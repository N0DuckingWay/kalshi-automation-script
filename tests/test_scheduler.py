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
import ast
import inspect
import json
import logging
import logging.handlers
import subprocess
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import schedule

from kalshi_betting import scheduler
from kalshi_betting.config import (
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    SCHEDULER_BLIND_MAX_RETRIES,
    SCHEDULER_BLIND_RETRY_SECONDS,
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
        # a manual-review run and a crash all leave the schedule empty.
        for code in (EXIT_OK, EXIT_SKIPPED_LOW_BALANCE, EXIT_TRADES_NEED_ATTENTION, 1):
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
    job = schedule.every().monday.at("09:00").do(scheduler._guarded_job, scheduler.run_job)
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
        """DR-59: main() must register run_job through _guarded_job. main()
        spawns a real prod run and is never invoked by this suite, so no
        runtime test can reach this line — without the pin, an edit reverting
        to `.do(run_job)` restores DR-59 with a green suite."""
        calls = _do_calls_in("main")
        assert len(calls) == 1, "main() should register exactly one job"
        args = calls[0].args
        assert isinstance(args[0], ast.Name) and args[0].id == "_guarded_job"
        assert isinstance(args[1], ast.Name) and args[1].id == "run_job"

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
