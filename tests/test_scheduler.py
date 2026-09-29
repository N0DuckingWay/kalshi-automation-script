"""Tests for scheduler.py: exit-code logging, TimeoutExpired/OSError handling,
missed-run catch-up, the weekly job (_weekly_job, _cron_line) and the startup
host-clock check (_host_clock_realises_run). subprocess.run is mocked, and the
autouse `_tmp_project_root` fixture sends run_job's scheduler_state.json
writes to tmp_path instead of the repo root.

TestRunInProgress covers exit 50, a run stopped because another live trading
run held the live-run lock; tests/conftest.py points the lock's holder record
at each test's own tmp_path.
"""
import ast
import contextlib
import inspect
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
import schedule

from kalshi_betting import config, run_lock, scheduler
from kalshi_betting.config import (
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    SCHEDULER_BLIND_MAX_RETRIES,
    SCHEDULER_BLIND_RETRY_SECONDS,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
    ScheduledRun,
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


@pytest.fixture(autouse=True)
def _clean_global_schedule():
    """
    Empty the `schedule` library's GLOBAL job registry around every test.

    TS-01's blind-run retry calls schedule.every(...).do(...), which mutates
    process-wide state that outlives the test that created it: without this,
    a run_job() that exits 30 leaves a live job behind and the next test's
    `schedule.jobs` assertion (or a stray run_pending) sees it. Cleared in
    BOTH setup and teardown so the module is order-independent whether or not
    some other module registered a job first.
    """
    schedule.clear()
    yield
    schedule.clear()


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
    def test_time_series_skipped_logs_error(self, mock_run, caplog):
        # A held market could not be identified, so the run made no
        # time-series trade. It is an ERROR of its own, never a clean run or
        # the catch-all failure, and it points at the line naming the market.
        mock_run.return_value = _completed(EXIT_TIME_SERIES_SKIPPED, stderr="noise")

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "NO time-series trade" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        message = matches[0].getMessage()
        assert f"exit {EXIT_TIME_SERIES_SKIPPED}" in message
        assert "Same-title pairs were still searched" in message
        assert "kalshi_arb.log" in message
        assert "noise" not in message
        assert not any("Job completed successfully." in r.getMessage() for r in caplog.records)
        assert not any("Job failed" in r.getMessage() for r in caplog.records)

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_every_exit_code_has_a_message_of_its_own(self, mock_run, caplog):
        # Every EXIT_* code in config.py is distinct, none is the crash (1) or
        # usage-error (2) code, and none falls through to the catch-all
        # "Job failed" branch, so a new code cannot be added without a message.
        codes = {name: getattr(config, name) for name in dir(config)
                 if name.startswith("EXIT_")}
        assert EXIT_TIME_SERIES_SKIPPED in codes.values()
        assert EXIT_RUN_IN_PROGRESS in codes.values()
        assert len(set(codes.values())) == len(codes)
        assert not {1, 2} & set(codes.values())
        for name, code in codes.items():
            caplog.clear()
            schedule.clear()
            mock_run.return_value = _completed(code, stderr="x")
            with caplog.at_level(logging.INFO):
                scheduler.run_job()
            assert not any("Job failed" in r.getMessage() for r in caplog.records), name

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


def _write_lock_holder(record) -> None:
    """
    Write a holder record (any JSON value) where run_lock.holder() reads it.

    tests/conftest.py points config.LIVE_RUN_LOCK_FILE at the test's tmp_path,
    so this never touches the real lock in the home folder.

    Args:
        record: The value to write as JSON.
    """
    config.LIVE_RUN_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_RUN_LOCK_FILE.write_text(json.dumps(record), encoding="utf-8")


def _utc_text(moment: datetime) -> str:
    """
    Write a time the way the run lock's holder record stores it.

    Uses run_lock._TIME_FORMAT, the format LockHolder.age_seconds() reads
    back, so a record built here is always one the scheduler can date.

    Args:
        moment (datetime): A timezone-aware time.

    Returns:
        str: The time in UTC, as run_lock._TIME_FORMAT text
            (e.g. "2026-09-29T16:00:04Z").
    """
    return moment.astimezone(UTC).strftime(run_lock._TIME_FORMAT)


class TestRunInProgress:
    """Exit 50: another live trading run held the run lock, so the scheduled
    run stopped before making any request. The slot counts as done and is
    never retried; the message names the run in the way, and is an ERROR when
    that run has held the lock for longer than any run should, or its start is
    not recorded."""

    _CHECKOUT = "/Users/me/Kalshi Betting App"

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_a_fresh_holder_is_a_warning_naming_it(self, mock_run, tmp_path, caplog):
        _write_lock_holder({"pid": 4242, "checkout": self._CHECKOUT,
                            "started_at": _utc_text(datetime.now(UTC) - timedelta(minutes=5))})
        mock_run.return_value = _completed(EXIT_RUN_IN_PROGRESS, stderr="noise")

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "did not trade" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.WARNING
        message = matches[0].getMessage()
        assert f"exit {EXIT_RUN_IN_PROGRESS}" in message
        assert "another live trading run on this machine was in progress" in message
        assert f"process 4242 in {self._CHECKOUT}, since" in message
        assert "The weekly slot counts as done" in message
        assert "not retried" in message
        assert "noise" not in message
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)
        assert not any("Job failed" in r.getMessage() for r in caplog.records)
        # Done, not reopened: finalized with its code and no retry registered
        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == EXIT_RUN_IN_PROGRESS
        assert state["finished_at"] is not None
        assert schedule.jobs == []

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_a_holder_older_than_the_job_timeout_is_an_error(self, mock_run, tmp_path, caplog):
        started = datetime.now(UTC) - timedelta(seconds=SCHEDULER_JOB_TIMEOUT_SECONDS + 60)
        _write_lock_holder({"pid": 4242, "checkout": self._CHECKOUT,
                            "started_at": _utc_text(started)})
        mock_run.return_value = _completed(EXIT_RUN_IN_PROGRESS)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "did not trade" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        message = matches[0].getMessage()
        assert f"exit {EXIT_RUN_IN_PROGRESS}" in message
        assert "process 4242" in message
        assert "may be hung or stopped" in message
        assert "No trade was made for this weekly slot" in message
        assert "every scheduled run will stop the same way until that run ends" in message
        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == EXIT_RUN_IN_PROGRESS
        assert schedule.jobs == []

    @pytest.mark.parametrize("record", [
        None,
        {"pid": 4242, "checkout": _CHECKOUT},
        {"pid": 4242, "started_at": "not a time"},
    ], ids=["no-record", "no-start-time", "unreadable-start-time"])
    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_a_holder_whose_start_is_unknown_is_an_error(
        self, mock_run, record, tmp_path, caplog,
    ):
        if record is not None:
            _write_lock_holder(record)
        mock_run.return_value = _completed(EXIT_RUN_IN_PROGRESS)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "did not trade" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        assert "its start is not recorded" in matches[0].getMessage()
        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == EXIT_RUN_IN_PROGRESS
        assert state["finished_at"] is not None
        assert schedule.jobs == []

    def test_a_slot_the_lock_stopped_is_not_caught_up(self, tmp_path):
        # A real-money run was trading, so the slot is done: a restart does
        # not re-run it
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(tmp_path, last_slot=current_slot.isoformat(),
                     exit_code=EXIT_RUN_IN_PROGRESS)

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()

    def test_the_state_file_name_comes_from_config(self, tmp_path):
        assert config.SCHEDULER_STATE_FILENAME == "scheduler_state.json"
        assert scheduler._state_file_path() == tmp_path / config.SCHEDULER_STATE_FILENAME


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
    """_most_recent_slot: the latest SCHEDULED_RUN slot <= now, on the host's clock."""

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

    @pytest.mark.parametrize(
        ("now", "expected"),
        [
            # Thursday 2026-09-03 before 14:30: the previous Thursday.
            (datetime(2026, 9, 3, 14, 29), datetime(2026, 8, 27, 14, 30)),
            # Thursday 2026-09-03 exactly at and after 14:30: that day.
            (datetime(2026, 9, 3, 14, 30), datetime(2026, 9, 3, 14, 30)),
            (datetime(2026, 9, 3, 23, 59), datetime(2026, 9, 3, 14, 30)),
            # Monday 2026-08-31 and Wednesday 2026-09-02: the Thursday before.
            (datetime(2026, 8, 31, 9, 0), datetime(2026, 8, 27, 14, 30)),
            (datetime(2026, 9, 2, 23, 0), datetime(2026, 8, 27, 14, 30)),
        ],
    )
    def test_a_patched_schedule_moves_the_slot(self, monkeypatch, now, expected):
        monkeypatch.setattr(
            scheduler, "SCHEDULED_RUN", ScheduledRun(3, 14, 30, "America/Los_Angeles"),
        )
        assert scheduler._most_recent_slot(now) == expected


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

    def test_the_catch_up_warning_names_the_slot_on_the_hosts_clock(self, tmp_path, caplog):
        # The slot is on the host's clock, so the WARNING says so and names no zone
        now = datetime(2026, 9, 2, 10, 0)

        with patch("kalshi_betting.scheduler.run_job"), caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        [message] = [r.getMessage() for r in caplog.records if "no recorded run" in r.getMessage()]
        assert "Monday 09:00 on the host's clock, 2026-08-31T09:00:00" in message
        assert "America/Los_Angeles" not in message

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

    def test_blind_run_slot_is_retried_on_startup(self, tmp_path, caplog):
        # TS-01: the old test was purely temporal (last_slot < slot) and never
        # read the exit code, so a slot whose only attempt scanned NOTHING
        # counted as satisfied and the bot waited a week.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS, retries=1,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once_with(retries=2)
        assert any("blind-run" in r.getMessage() for r in caplog.records)

    def test_blind_run_at_the_cap_is_not_retried(self, tmp_path):
        # The daemon-restart path shares run_job's cap, so a multi-day outage
        # cannot make every restart re-run the same dead slot forever.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS, retries=SCHEDULER_BLIND_MAX_RETRIES,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()

    def test_blind_run_state_without_retries_key_counts_as_zero(self, tmp_path):
        # A state file written before TS-01 has no "retries" key. DR-24's new
        # _load_state validation only fires on a key that is PRESENT and
        # non-int, so an absent one is still untouched and must load and
        # read as attempt 0.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once_with(retries=1)

    def test_stale_slot_catch_up_starts_at_zero_retries(self, tmp_path):
        # A never-attempted slot is a fresh first run, not a retry — its count
        # must not inherit anything from the previous slot's record.
        now = datetime(2026, 9, 2, 10, 0)
        stale_slot = scheduler._most_recent_slot(now) - timedelta(days=7)
        _write_state(
            tmp_path, last_slot=stale_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS, retries=3,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once_with(retries=0)

    def test_a_run_that_skipped_time_series_is_not_retried(self, tmp_path):
        # It scanned and could trade same-title pairs, so the slot is done: a
        # restart does not re-run it.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_TIME_SERIES_SKIPPED,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()

    def test_failed_but_scanning_run_is_still_not_retried(self, tmp_path):
        # BS-17's deliberate behaviour, unchanged: only a BLIND run reopens a
        # slot. A run that scanned and merely failed stays satisfied.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(tmp_path, last_slot=current_slot.isoformat(), exit_code=1)

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()

    def test_non_int_retries_string_warns_and_is_read_as_zero(self, tmp_path, caplog):
        # DR-24: a hand-edited state file with a string "retries" used to make
        # _maybe_catch_up's `retries < SCHEDULER_BLIND_MAX_RETRIES` comparison
        # raise TypeError. _load_state now degrades it to 0 (with a WARNING)
        # before _maybe_catch_up ever sees it, so the blind slot is retried
        # exactly as it would be for a freshly-absent "retries" key.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS, retries="2",
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once_with(retries=1)
        assert any(
            "non-integer retries" in r.getMessage() for r in caplog.records
        )

    def test_null_retries_warns_and_is_read_as_zero(self, tmp_path, caplog):
        # Same as above for a JSON null (Python None) rather than a string —
        # the other shape a hand-edited or partially-written field can take.
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(),
            exit_code=EXIT_NO_TRADEABLE_SHARDS, retries=None,
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once_with(retries=1)
        assert any(
            "non-integer retries" in r.getMessage() for r in caplog.records
        )

    def test_non_int_exit_code_is_read_as_unknown_and_slot_not_retried(
        self, tmp_path, caplog,
    ):
        # DR-24's exit_code guard never fixed a crash: `==` against a
        # mismatched type doesn't raise, so a non-int exit_code always
        # compared unequal to EXIT_NO_TRADEABLE_SHARDS and left the slot
        # unretried, both before and after this guard existed. For the
        # string case pinned here that makes it a typing-hygiene change
        # only — the WARNING is new, the outcome ("30" == 30 is False) is
        # not. (It is not a no-op for every non-int shape: a JSON float
        # 30.0 compared EQUAL before this guard and is now read as unknown,
        # so a hand-edited float exit_code would newly lose its blind-run
        # retry. No writer produces one — _save_state always writes an int
        # — so only a hand-edited file can reach that case.)
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(
            tmp_path, last_slot=current_slot.isoformat(), exit_code="30",
        )

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()
        assert any(
            "non-integer exit_code" in r.getMessage() for r in caplog.records
        )


class TestStartupCatchUp:
    """DR-24: main() calls _startup_catch_up() instead of a bare
    _maybe_catch_up(), so an exception _load_state's own validation didn't
    catch still can't exit the daemon before the weekly job is registered."""

    def test_maybe_catch_up_raising_is_logged_and_swallowed(self, caplog):
        with patch(
            "kalshi_betting.scheduler._maybe_catch_up", side_effect=TypeError("boom"),
        ), caplog.at_level(logging.ERROR):
            scheduler._startup_catch_up()  # must not raise

        assert any(
            "Startup catch-up check raised" in r.getMessage() for r in caplog.records
        )

    def test_main_calls_the_guarded_wrapper(self):
        """DR-24: main() must call _startup_catch_up(), not _maybe_catch_up()
        directly — a bare call can raise out of main() before the weekly job
        is registered, exactly the failure this commit fixes. main() itself
        spawns a real prod run and is never invoked by this suite, so the
        call site is pinned on the source instead."""
        tree = ast.parse(inspect.getsource(scheduler))
        main_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        called = {
            sub.func.id
            for sub in ast.walk(main_fn)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        assert "_startup_catch_up" in called
        assert "_maybe_catch_up" not in called


class TestCheckLiveDefaults:
    """_check_live_defaults: at daemon start, one ERROR when every scheduled run
    would exit 2 for want of usable saved live defaults, naming the fix; nothing
    when a usable file is saved. It never raises, so main() still registers the
    weekly job, and main() calls it before registering that job."""

    @staticmethod
    def _errors(caplog) -> list[str]:
        """
        The messages of every ERROR (or worse) record captured so far.

        Args:
            caplog (pytest.LogCaptureFixture): The test's captured log.

        Returns:
            list[str]: Each such record's message, in order.
        """
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]

    def test_no_saved_file_logs_the_missing_file_error(self, caplog):
        assert not config.LIVE_DEFAULTS_FILE.exists()
        with caplog.at_level(logging.INFO):
            scheduler._check_live_defaults()
        (error,) = self._errors(caplog)
        # It names the checkout the spawned runs read the file in
        assert error.startswith(f"No live defaults are saved in {scheduler.PROJECT_ROOT} "
                                "(the checkout this daemon runs): every scheduled run will "
                                "exit 2")
        assert "python3 -m kalshi_betting.defaults_server --seed" in error
        assert "Save as live defaults…" in error

    def test_a_refused_file_logs_the_refused_file_error(self, caplog):
        config.LIVE_DEFAULTS_FILE.write_text("not json", encoding="utf-8")
        with caplog.at_level(logging.INFO):
            scheduler._check_live_defaults()
        (error,) = self._errors(caplog)
        assert error.startswith("The saved live defaults are refused (")
        assert str(config.LIVE_DEFAULTS_FILE) in error
        # The server will not save over a refused file: fix it, or delete it first
        assert error.endswith("every scheduled run will exit 2 until the file is fixed, "
                              "or deleted and saved again")

    @pytest.mark.usefixtures("saved_live_defaults")
    def test_a_usable_file_logs_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG):
            scheduler._check_live_defaults()
        assert not [r for r in caplog.records if r.name == "root"]

    def test_it_never_raises(self, monkeypatch, caplog):
        def broken():
            """
            Stand in for a reader that fails in a way no refusal covers.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("an unexpected failure")

        monkeypatch.setattr(scheduler, "read_saved_live_defaults", broken)
        with caplog.at_level(logging.INFO):
            scheduler._check_live_defaults()  # must not raise
        (error,) = self._errors(caplog)
        assert "an unexpected failure" in error

    def test_main_calls_it_before_registering_the_weekly_job(self):
        # main() spawns a real prod run and is never invoked by this suite, so
        # the order is pinned on the source: after the host-clock check, before
        # the catch-up (which can spawn a run) and the weekly job's registration
        tree = ast.parse(inspect.getsource(scheduler))
        main_fn = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        checks = [n.lineno for n in ast.walk(main_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "_check_live_defaults"]
        registrations = [n.lineno for n in ast.walk(main_fn)
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "do"]
        assert len(checks) == 1 and len(registrations) == 1
        assert checks[0] < registrations[0]
        lines = {n.func.id: n.lineno for n in ast.walk(main_fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in ("_host_clock_realises_run", "_startup_catch_up")}
        assert (lines["_host_clock_realises_run"] < checks[0]
                < lines["_startup_catch_up"] < registrations[0])


class TestBlindRunRetry:
    """TS-01/VI-02: EXIT_NO_TRADEABLE_SHARDS means the run scanned nothing —
    either an exchange-wide halt dropped every market at ingest, or the ingest
    came back empty for a cause /exchange/status could not name. The scheduler
    sees only the exit code, so its messages name both possibilities and point
    at kalshi_arb.log, where main._blind_run_reason logged which one fired. The
    bot trades only on the weekly fire, so that slot must not count as
    satisfied — it is retried hourly, a bounded number of times."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_blind_exit_warns_and_registers_one_retry(self, mock_run, tmp_path, caplog):
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [
            r for r in caplog.records
            if "Job scanned nothing (exit 30)" in r.getMessage()
        ]
        assert len(matches) == 1
        assert matches[0].levelno == logging.WARNING
        assert "NOT satisfied" in matches[0].getMessage()
        # The daemon has only the exit code, which no longer identifies one
        # cause: the message must offer both and point at the log that does.
        assert "the market ingest came back empty" in matches[0].getMessage()
        assert "kalshi_arb.log" in matches[0].getMessage()
        assert f"attempt 1 of {SCHEDULER_BLIND_MAX_RETRIES}" in matches[0].getMessage()
        # Never the generic "Job failed" branch, and never "successfully".
        assert "Job failed" not in caplog.text
        assert "Job completed successfully." not in caplog.text

        assert len(schedule.jobs) == 1
        assert schedule.jobs[0].interval == SCHEDULER_BLIND_RETRY_SECONDS

        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == EXIT_NO_TRADEABLE_SHARDS
        assert state["retries"] == 0

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_firing_the_retry_reenters_run_job_and_cancels_itself(self, mock_run):
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)

        scheduler.run_job()
        job = schedule.jobs[0]

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            result = job.run()

        # The retry carries the incremented count — the cap is enforced by the
        # argument each attempt passes on, not by the caller.
        mock_run_job.assert_called_once_with(retries=1)
        assert result is schedule.CancelJob

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_retry_is_one_shot_not_recurring(self, mock_run):
        # schedule removes a job whose function returns CancelJob, so a blind
        # retry can never become a permanent hourly job.
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)

        scheduler.run_job(retries=SCHEDULER_BLIND_MAX_RETRIES)  # registers nothing
        assert schedule.jobs == []

        scheduler.run_job()
        assert len(schedule.jobs) == 1

        # Dispatch through the library itself (run_pending is what honours
        # CancelJob; Job.run() alone does not deregister), with the due time
        # pulled into the past so the hourly job fires now.
        schedule.jobs[0].next_run = datetime.now() - timedelta(seconds=1)
        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            schedule.run_pending()

        mock_run_job.assert_called_once_with(retries=1)
        assert schedule.jobs == [], "a blind retry must never become a recurring job"

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_at_the_cap_logs_error_and_registers_nothing(
        self, mock_run, tmp_path, caplog,
    ):
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)

        with caplog.at_level(logging.INFO):
            scheduler.run_job(retries=SCHEDULER_BLIND_MAX_RETRIES)

        matches = [
            r for r in caplog.records
            if "Job scanned nothing on" in r.getMessage()
        ]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        assert f"{SCHEDULER_BLIND_MAX_RETRIES + 1} attempts" in matches[0].getMessage()
        # Terminal message for the week: it must not assert the halt as fact
        # when an empty ingest is equally possible behind exit 30.
        assert "kept coming back empty" in matches[0].getMessage()
        assert "kalshi_arb.log" in matches[0].getMessage()
        assert schedule.jobs == [], "the cap must stop the retry chain"

        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["retries"] == SCHEDULER_BLIND_MAX_RETRIES

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_one_below_the_cap_still_retries(self, mock_run):
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)

        scheduler.run_job(retries=SCHEDULER_BLIND_MAX_RETRIES - 1)

        assert len(schedule.jobs) == 1

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_other_exit_codes_register_no_retry(self, mock_run):
        # Only a BLIND run reopens the slot. A clean run, a low-balance skip,
        # a manual-review run, a run that skipped time-series trading and a
        # crash all leave the schedule empty.
        for code in (EXIT_OK, EXIT_SKIPPED_LOW_BALANCE, EXIT_TRADES_NEED_ATTENTION,
                     EXIT_TIME_SERIES_SKIPPED, 1):
            mock_run.return_value = _completed(code, stderr="x")
            scheduler.run_job()
            assert schedule.jobs == [], f"exit {code} must not schedule a retry"

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_retry_count_is_persisted_on_the_claim(self, mock_run, tmp_path):
        # The claim happens BEFORE the subprocess runs, so a host reboot
        # mid-retry leaves the attempt number on disk for the catch-up check.
        claimed: list = []

        def spy(*args, **kwargs):
            claimed.append(json.loads((tmp_path / "scheduler_state.json").read_text()))
            return _completed(EXIT_OK)

        mock_run.side_effect = spy
        scheduler.run_job(retries=2)

        assert claimed[0]["retries"] == 2
        assert claimed[0]["finished_at"] is None


def _register_weekly_job_as_main_does():
    """
    Register the weekly job exactly as scheduler.main() does, due in the past.

    main() itself spawns a real prod run and is never invoked by this suite, so
    the registration it performs is reproduced here verbatim and its next_run
    is pulled back one second so schedule.run_pending() fires it immediately.

    Because this is a REPRODUCTION rather than a read of main(), the tests
    built on it prove the guard WORKS, not that main() is wired to it — that
    half is pinned on the source by
    test_ast_weekly_job_is_registered_through_the_guard, and only that pin
    fails if main() reverts to a bare `.do(run_job)`.

    Returns:
        schedule.Job: The registered, already-overdue weekly job.
    """
    job = scheduler._weekly_job().do(scheduler._guarded_job, scheduler.run_job)
    job.next_run = datetime.now() - timedelta(seconds=1)
    return job


def _do_calls_in(func_name: str):
    """
    Collect every `<something>.do(...)` Call node inside one scheduler function.

    Args:
        func_name (str): The scheduler.py function to parse.

    Returns:
        list[ast.Call]: The `.do(...)` calls found, in source order.
    """
    tree = ast.parse(inspect.getsource(scheduler))
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    return [
        sub for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "do"
    ]


class TestGuardedJobRegistration:
    """DR-59: schedule.Job.run() assigns last_run and calls
    _schedule_next_run() only AFTER job_func() RETURNS, so any exception
    escaping a job leaves next_run in the past. main()'s poll loop catches it
    and keeps the daemon alive, but the job is still overdue — so it re-enters
    on the very next 60 s poll tick, turning the weekly production trading run
    into a once-a-minute one for as long as the fault persists. Every job is
    therefore registered through _guarded_job()."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_state_write_failure_advances_next_run(self, mock_run, monkeypatch):
        # _save_state's tmp+replace is the slot CLAIM, and it sits OUTSIDE
        # run_job's try — so an OSError there escapes the job entirely.
        mock_run.return_value = _completed(EXIT_OK)

        def _failing_replace(self, target):
            raise OSError("state write failed")

        monkeypatch.setattr(scheduler.pathlib.Path, "replace", _failing_replace)
        job = _register_weekly_job_as_main_does()

        schedule.run_pending()  # must not raise

        assert job.next_run > datetime.now()
        assert not job.should_run, (
            "an escaping exception must not leave the weekly job overdue — "
            "that is what re-fires a prod run every 60 s poll tick"
        )
        # The claim raised before the spawn, so no prod run was attempted.
        mock_run.assert_not_called()

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_unicode_decode_error_from_subprocess_run_is_contained(
        self, mock_run, caplog,
    ):
        # subprocess.run(..., text=True) decodes the child's streams strictly,
        # so one non-UTF-8 byte from the bot raises straight out of run_job.
        mock_run.side_effect = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte",
        )
        job = _register_weekly_job_as_main_does()

        with caplog.at_level(logging.ERROR):
            schedule.run_pending()  # must not raise

        assert job.next_run > datetime.now()
        assert not job.should_run
        assert any(
            "Scheduled job run_job raised" in r.getMessage() for r in caplog.records
        )

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_raising_blind_retry_still_cancels_itself(self, mock_run):
        # A retry signals "run me once" by RETURNING schedule.CancelJob; one
        # that raises returns nothing, so without on_error it would stay
        # registered, re-fire every poll tick, and never advance the
        # SCHEDULER_BLIND_MAX_RETRIES count the `retries` argument carries.
        mock_run.return_value = _completed(EXIT_NO_TRADEABLE_SHARDS)
        scheduler.run_job()
        assert len(schedule.jobs) == 1

        schedule.jobs[0].next_run = datetime.now() - timedelta(seconds=1)
        with patch(
            "kalshi_betting.scheduler.run_job", side_effect=OSError("disk write failed"),
        ) as mock_run_job:
            schedule.run_pending()  # must not raise

        mock_run_job.assert_called_once_with(retries=1)
        assert schedule.jobs == [], (
            "a blind retry that RAISES must still deregister itself"
        )

    def test_guarded_job_forwards_arguments_and_return_value(self):
        # .do(_guarded_job, job, *args, **kwargs) forwards everything after the
        # first argument through functools.partial, so the guard must be
        # transparent on the healthy path.
        seen = {}

        def record(a, b, c=None):
            seen.update({"a": a, "b": b, "c": c})
            return "ok"

        assert scheduler._guarded_job(record, 1, 2, c=3) == "ok"
        assert seen == {"a": 1, "b": 2, "c": 3}

    def test_guarded_job_returns_on_error_and_logs_once(self, caplog):
        def boom():
            raise OSError("nope")

        with caplog.at_level(logging.ERROR):
            assert scheduler._guarded_job(boom, on_error=schedule.CancelJob) is (
                schedule.CancelJob
            )

        matches = [
            r for r in caplog.records if "Scheduled job boom raised" in r.getMessage()
        ]
        assert len(matches) == 1
        assert matches[0].exc_info is not None, "the traceback must be logged"

    def test_guarded_job_default_on_error_is_none(self):
        def boom():
            raise ValueError("nope")

        # The weekly job passes no on_error: a recurring job must NOT be
        # cancelled by a failure, only rescheduled.
        assert scheduler._guarded_job(boom) is None

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
    def test_guarded_job_lets_ctrl_c_and_sys_exit_through(self, exc):
        """The guard catches Exception, NOT BaseException — Ctrl-C and
        sys.exit() must still stop the daemon.

        Nothing else in the suite pins this: widening the except clause to
        BaseException passes every other test, and would leave a daemon that
        cannot be interrupted while a job is running (each Ctrl-C would be
        swallowed, logged as a failed job, and the poll loop would carry on).
        The guard exists to stop a RAISING job re-firing every poll tick
        (DR-59), not to make the daemon unkillable.
        """
        def boom():
            raise exc("stop")

        with pytest.raises(exc):
            scheduler._guarded_job(boom, on_error=schedule.CancelJob)

    def test_ast_weekly_job_is_registered_through_the_guard(self):
        """main() must register run_job through _guarded_job on _weekly_job()'s
        job. main() is never invoked by this suite, so only this pin catches a
        revert to `.do(run_job)` (a raising job would re-fire every 60-second
        poll tick) or another job (which would move the live run)."""
        calls = _do_calls_in("main")
        assert len(calls) == 1, "main() should register exactly one job"
        args = calls[0].args
        assert isinstance(args[0], ast.Name) and args[0].id == "_guarded_job"
        assert isinstance(args[1], ast.Name) and args[1].id == "run_job"
        receiver = calls[0].func.value
        assert isinstance(receiver, ast.Call), ast.unparse(receiver)
        assert isinstance(receiver.func, ast.Name) and receiver.func.id == "_weekly_job"
        assert not receiver.args and not receiver.keywords

    def test_ast_blind_retry_is_registered_through_the_guard_with_canceljob(self):
        """DR-59: the blind retry must route through _guarded_job with
        on_error=schedule.CancelJob, so a RAISING retry still deregisters
        itself and the retry cap keeps advancing."""
        calls = _do_calls_in("run_job")
        assert len(calls) == 1, "run_job() should register exactly one retry"
        call = calls[0]
        args = call.args
        assert isinstance(args[0], ast.Name) and args[0].id == "_guarded_job"
        assert isinstance(args[1], ast.Name) and args[1].id == "_blind_retry"
        on_error = next(k.value for k in call.keywords if k.arg == "on_error")
        assert isinstance(on_error, ast.Attribute)
        assert on_error.attr == "CancelJob"
        assert isinstance(on_error.value, ast.Name) and on_error.value.id == "schedule"


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


class TestSetupLogging:
    """The daemon's own logging setup (C2).

    scheduler._setup_logging used to install a plain logging.FileHandler on
    PROJECT_ROOT / "kalshi_arb.log" — the very file its main.py subprocess
    rotates with a RotatingFileHandler. A long-lived daemon holding an open
    handle on that inode keeps writing into the renamed kalshi_arb.log.1 after
    every rotation, so its lines silently vanish from the live log. The daemon
    now logs to its own kalshi_scheduler.log, and rotates it itself.
    """

    def test_setup_logging_uses_rotating_file_handler(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            log_path = tmp_path / "x.log"
            scheduler._setup_logging(log_path)

            handler_types = [type(h) for h in root.handlers]
            assert logging.StreamHandler in handler_types
            # Rotating, not plain: this daemon runs forever, so its own log
            # must be bounded exactly like main.py's is (BS-25's reasoning).
            assert logging.handlers.RotatingFileHandler in handler_types
            # Exactly one of each — basicConfig should not have added extras
            assert sum(1 for h in root.handlers if type(h) is logging.StreamHandler) == 1
            assert sum(1 for h in root.handlers if isinstance(h, logging.FileHandler)) == 1

            rotating = next(
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            )
            # Same 5 MB x 3 sizing as main._setup_logging.
            assert rotating.maxBytes == 5 * 1024 * 1024
            assert rotating.backupCount == 3

            # delay=True: the file must not be created until a record is emitted
            assert not log_path.exists()
            logging.getLogger("test_setup_logging").info("trigger the delayed file open")
            assert log_path.exists()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level

    def test_scheduler_logs_to_its_own_file(self):
        # The whole point of C2: a file this process alone owns, never the one
        # the spawned main.py subprocess rotates out from under it.
        assert scheduler._SCHEDULER_LOG_PATH.name == "kalshi_scheduler.log"
        assert scheduler._SCHEDULER_LOG_PATH.name != "kalshi_arb.log"


class TestWeeklyJobIsTheOldRegistration:
    """On the shipped schedule, _weekly_job() builds the same job as the literal
    `schedule.every().monday.at("09:00")`: Monday 09:00 on the host's clock,
    with no time zone. It must never take one: schedule's zone option uses
    pytz, whose Los Angeles table has no daylight time after 2037."""

    def test_the_job_matches_the_literal_registration(self):
        job = scheduler._weekly_job()
        literal = schedule.every().monday.at("09:00")
        for attr in ("at_time", "unit", "start_day", "interval", "at_time_zone"):
            assert getattr(job, attr) == getattr(literal, attr), attr
        assert job.at_time_zone is None
        # Built, not registered: main() registers it with .do().
        assert job.job_func is None
        assert schedule.jobs == []

    def test_the_first_fire_is_the_literal_registrations(self):
        job = scheduler._weekly_job().do(lambda: None)
        literal = schedule.every().monday.at("09:00").do(lambda: None)
        assert job.next_run == literal.next_run

    def test_a_patched_schedule_moves_the_job(self, monkeypatch):
        monkeypatch.setattr(
            scheduler, "SCHEDULED_RUN", ScheduledRun(3, 14, 30, "America/Los_Angeles"),
        )
        job = scheduler._weekly_job()
        assert (job.unit, job.start_day, job.at_time.hour, job.at_time.minute) == (
            "weeks", "thursday", 14, 30,
        )
        assert job.at_time_zone is None


class TestOneRunSchedule:
    """scheduler.py takes the run's weekday and time from config.SCHEDULED_RUN
    alone: no "09:00" literal, no `.monday`, and no `.at()` call with a zone."""

    @staticmethod
    def _tree():
        """Parse scheduler.py's source."""
        return ast.parse(inspect.getsource(scheduler))

    def test_ast_no_literal_run_time(self):
        assert not [
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.Constant) and n.value == "09:00"
        ]

    def test_ast_no_literal_weekday(self):
        assert not [
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.Attribute) and n.attr == "monday"
        ]

    def test_ast_no_at_call_passes_a_zone(self):
        at_calls = [
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "at"
        ]
        assert at_calls, "the weekly job's .at() call should be found"
        for call in at_calls:
            assert len(call.args) == 1 and not call.keywords, ast.unparse(call)

    def test_the_scheduler_reads_the_config_schedule(self):
        assert scheduler.SCHEDULED_RUN is config.SCHEDULED_RUN


@contextlib.contextmanager
def _host_zone(name: str):
    """
    Run the block with the host's clock set to IANA zone `name`.

    Sets TZ and calls time.tzset(): the scheduler's host-clock check depends on
    the host zone, and test_backtester imports this to show the backtest does
    not. Restores TZ on exit; skips where the platform cannot load `name`.

    Args:
        name (str): IANA zone name to set as TZ.

    Yields:
        None
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")
    saved = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        zone = ZoneInfo(name)
        for probe in (datetime(2026, 1, 15, 12, 0), datetime(2026, 7, 15, 12, 0)):
            if probe.astimezone(UTC) != probe.replace(tzinfo=zone).astimezone(UTC):
                pytest.skip(f"the C library cannot load TZ={name}")
        yield
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()


def _criticals(caplog) -> list[str]:
    """Messages of the CRITICAL records caplog captured."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL]


def _late_on(late: date) -> ScheduledRun:
    """
    Build the shipped schedule with instant() one minute late on `late` only,
    so a host keeping the schedule's zone sees exactly one mismatch.

    Args:
        late (date): The run date whose instant is moved.

    Returns:
        ScheduledRun: Equal in every field to config.SCHEDULED_RUN.
    """
    class _Late(ScheduledRun):
        def instant(self, d: date) -> datetime:
            moment = super().instant(d)
            return moment + timedelta(minutes=1) if d == late else moment

    return _Late(0, 9, 0, "America/Los_Angeles")


class TestHostClockCheck:
    """_host_clock_realises_run(): fires land at SCHEDULED_RUN.instant(d) only if
    the host's clock keeps SCHEDULED_RUN's zone. A mismatch, bad zone or
    unplaceable date is logged CRITICAL, never raised (main() runs it at startup)."""

    def test_a_host_in_the_schedules_zone_passes(self, caplog):
        with _host_zone("America/Los_Angeles"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 9, 21)) is True
        assert _criticals(caplog) == []
        assert any(
            "keeps config.SCHEDULED_RUN" in r.getMessage() for r in caplog.records
        )

    def test_a_utc_host_is_flagged_on_the_first_run_date(self, caplog):
        # The first run date checked is Monday 2026-09-28: 09:00 UTC on this host
        with _host_zone("UTC"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 9, 23)) is False
        [message] = _criticals(caplog)
        assert "on 2026-09-28" in message
        assert "2026-09-28 09:00 UTC, not 2026-09-28 16:00 UTC" in message
        assert "the backtest" in message and "no longer replays this host's runs" in message

    def test_a_host_without_daylight_time_is_flagged_at_the_clock_change(self, caplog):
        # Phoenix has no daylight time: it first differs when Los Angeles falls back
        with _host_zone("America/Phoenix"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 9, 21)) is False
        [message] = _criticals(caplog)
        assert "on 2026-11-02" in message

    def test_a_fixed_offset_host_is_flagged_at_the_spring_change(self, caplog):
        # A UTC-8 host matches Los Angeles all winter: the first mismatch is 18 weeks out
        with _host_zone("Etc/GMT+8"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 11, 9)) is False
        [message] = _criticals(caplog)
        assert "on 2027-03-15" in message
        assert "2027-03-15 17:00 UTC, not 2027-03-15 16:00 UTC" in message

    @pytest.mark.parametrize(
        ("weeks_out", "passes"),
        [(103, False), (104, True)],
        ids=["last-checked-run", "first-run-after-the-horizon"],
    )
    def test_the_check_covers_exactly_the_next_104_runs(
        self, monkeypatch, caplog, weeks_out, passes,
    ):
        # The 104th run date from 2026-09-21 is checked; the 105th is not
        late = date(2026, 9, 21) + timedelta(weeks=weeks_out)
        monkeypatch.setattr(scheduler, "SCHEDULED_RUN", _late_on(late))
        with _host_zone("America/Los_Angeles"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 9, 21)) is passes
        if passes:
            assert _criticals(caplog) == []
            assert any("next 104 weekly runs" in r.getMessage() for r in caplog.records)
        else:
            [message] = _criticals(caplog)
            assert f"on {late.isoformat()}" in message

    def test_a_skipped_wall_time_is_the_schedules_problem_not_the_hosts(
        self, monkeypatch, caplog,
    ):
        # Sunday 02:30 is skipped each spring: the CRITICAL blames the schedule, not the host
        monkeypatch.setattr(
            scheduler, "SCHEDULED_RUN", ScheduledRun(6, 2, 30, "America/Los_Angeles"),
        )
        with _host_zone("America/Los_Angeles"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 3, 1)) is False
        [message] = _criticals(caplog)
        assert "2026-03-08, 2027-03-14" in message and "skips" in message
        assert "set the host's time zone" not in message

    def test_a_skipped_wall_time_does_not_hide_a_host_mismatch(self, monkeypatch, caplog):
        # Sunday 2026-03-08 is skipped; 2026-03-15 then exposes the UTC host
        monkeypatch.setattr(
            scheduler, "SCHEDULED_RUN", ScheduledRun(6, 2, 30, "America/Los_Angeles"),
        )
        with _host_zone("UTC"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 3, 2)) is False
        skipped, host = _criticals(caplog)
        assert "2026-03-08" in skipped and "skips" in skipped
        assert "on 2026-03-15" in host and "set the host's time zone" in host

    @pytest.mark.parametrize(
        "zone",
        [
            "No/Such_Zone",    # ZoneInfoNotFoundError
            "America",         # a directory: OSError
            "/etc/localtime",  # an absolute path: ValueError
        ],
    )
    def test_an_unresolvable_zone_is_critical_and_never_raises(self, monkeypatch, caplog, zone):
        monkeypatch.setattr(scheduler, "SCHEDULED_RUN", ScheduledRun(0, 9, 0, zone))
        with caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(2026, 9, 21)) is False
        [message] = _criticals(caplog)
        assert repr(zone) in message and "cannot be resolved" in message

    def test_the_end_of_the_calendar_is_critical_and_never_raises(self, caplog):
        # 9999-12-27 is the last Monday datetime holds; the next cannot be placed
        with _host_zone("America/Los_Angeles"), caplog.at_level(logging.INFO):
            assert scheduler._host_clock_realises_run(date(9999, 12, 27)) is False
        [message] = _criticals(caplog)
        assert "could not be checked" in message

    def test_ast_main_checks_the_host_clock_first_and_never_gates_on_it(self):
        # main() is never invoked here, so its order is pinned on the source: after
        # _setup_logging (an earlier log call would stop the log file being set up),
        # before the catch-up and registration, and gating nothing.
        tree = ast.parse(inspect.getsource(scheduler))
        main_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        lines = {}
        for stmt in main_fn.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name):
                    lines[func.id] = stmt.lineno
                elif isinstance(func, ast.Attribute) and func.attr == "do":
                    lines["do"] = stmt.lineno
        assert (lines["_setup_logging"] < lines["_host_clock_realises_run"]
                < lines["_startup_catch_up"] < lines["do"])


class TestCronLine:
    """_cron_line(): the crontab line main() logs for operators who prefer cron,
    at SCHEDULED_RUN's weekday and time on the host's clock. Cron's day 0 is
    Sunday, datetime.weekday()'s is Monday: a wrong mapping moves the live run."""

    def test_the_shipped_schedule_is_monday_0900(self):
        assert scheduler._cron_line("/repo", "/py") == (
            "0 9 * * 1 cd '/repo' && /py -m kalshi_betting.main --mode prod "
            ">> /tmp/kalshi_arb.log 2>&1"
        )

    @pytest.mark.parametrize(
        ("run", "prefix"),
        [
            (ScheduledRun(6, 7, 5, "America/Los_Angeles"), "5 7 * * 0 "),
            (ScheduledRun(3, 14, 30, "America/Los_Angeles"), "30 14 * * 4 "),
            (ScheduledRun(5, 23, 59, "America/Los_Angeles"), "59 23 * * 6 "),
        ],
        ids=["sunday", "thursday", "saturday"],
    )
    def test_a_patched_schedule_moves_the_line(self, monkeypatch, run, prefix):
        monkeypatch.setattr(scheduler, "SCHEDULED_RUN", run)
        assert scheduler._cron_line("/repo", "/py").startswith(prefix)

    def test_ast_main_logs_the_helpers_line(self):
        # main() is never invoked here: pin that it logs _cron_line()'s line
        tree = ast.parse(inspect.getsource(scheduler.main))
        assert [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_cron_line"
        ]
