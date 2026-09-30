"""
File: test_run_lock.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for run_lock.py, the machine-wide lock that lets one live trading
    run place orders at a time: acquire() gives the lock to one holder,
    retries for the configured wait and then gives up, makes the lock file's
    folder and raises when it cannot; closing the descriptor frees the lock;
    held() reports a holder without keeping the lock, fails closed when the
    file cannot be opened, and never makes a starting run give up; holder()
    reads the holder record whatever the file holds; LockHolder describes the
    holder and measures its age; and a real process that holds the lock and
    is killed leaves it free.

Dependencies:
    Imports kalshi_betting.config (the lock file's path and waits) and
    kalshi_betting.run_lock. tests/conftest.py's autouse _isolate_live_runs
    points config.LIVE_RUN_LOCK_FILE at each test's tmp_path and shortens the
    waits, so no test here touches the real lock in the home folder.

Notes:
    An flock belongs to one open file, so a second open of the lock file in
    the same process is refused while the first holds it: most tests hold
    the lock in-process with acquire() and check a second acquire() or
    held() against it. The one test that starts another process hands it
    the redirected path on its command line, since the child cannot see the
    monkeypatch, and the child imports nothing from this package.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from kalshi_betting import config, run_lock


def _write_record(record) -> None:
    """
    Write a holder record (any JSON value) into the redirected lock file.

    Args:
        record: The value to write as JSON.
    """
    config.LIVE_RUN_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_RUN_LOCK_FILE.write_text(json.dumps(record), encoding="utf-8")


class TestAcquire:
    """acquire() takes the lock once, writes who holds it, and gives up on a held lock."""

    def test_the_lock_file_is_the_redirected_one(self, tmp_path):
        # Not vacuous: every test here works on its own tmp_path, never the home folder
        assert config.LIVE_RUN_LOCK_FILE == tmp_path / "lock" / "live_run.lock"
        assert config.LIVE_RUN_LOCK_WAIT_SECONDS == 0.2

    def test_a_missing_folder_is_created(self):
        folder = config.LIVE_RUN_LOCK_FILE.parent
        assert not folder.exists()
        fd = run_lock.acquire()
        try:
            assert fd is not None
            assert folder.is_dir() and config.LIVE_RUN_LOCK_FILE.is_file()
            assert (folder.stat().st_mode & 0o777) == 0o700
        finally:
            os.close(fd)

    def test_it_records_who_holds_it(self):
        fd = run_lock.acquire()
        try:
            record = json.loads(config.LIVE_RUN_LOCK_FILE.read_text(encoding="utf-8"))
            assert record["pid"] == os.getpid()
            assert record["checkout"] == str(config.PROJECT_ROOT)
            # The recorded format is spelled out here, not read from run_lock,
            # so a change to what the file holds fails this test
            started = datetime.strptime(record["started_at"], "%Y-%m-%dT%H:%M:%SZ")
            assert abs((datetime.now(UTC) - started.replace(tzinfo=UTC)).total_seconds()) < 60
        finally:
            os.close(fd)

    def test_a_new_holder_replaces_the_old_record(self):
        # A longer, stale record is cut off, not left trailing after the new one
        _write_record({"pid": 1, "checkout": "/x" * 200, "started_at": "2000-01-01T00:00:00Z"})
        fd = run_lock.acquire()
        try:
            record = json.loads(config.LIVE_RUN_LOCK_FILE.read_text(encoding="utf-8"))
            assert record["pid"] == os.getpid()
        finally:
            os.close(fd)

    def test_a_second_acquire_gives_none_once_the_wait_runs_out(self):
        first = run_lock.acquire()
        try:
            began = time.monotonic()
            assert run_lock.acquire() is None
            waited = time.monotonic() - began
            # It kept trying for the whole (patched) wait before giving up
            assert waited >= config.LIVE_RUN_LOCK_WAIT_SECONDS
            assert waited < config.LIVE_RUN_LOCK_WAIT_SECONDS + 5
        finally:
            os.close(first)

    def test_closing_the_descriptor_frees_the_lock(self):
        first = run_lock.acquire()
        os.close(first)
        second = run_lock.acquire()
        assert second is not None
        os.close(second)
        assert not run_lock.held()

    def test_a_refused_acquire_leaves_the_holder_record_alone(self):
        first = run_lock.acquire()
        try:
            before = config.LIVE_RUN_LOCK_FILE.read_text(encoding="utf-8")
            assert run_lock.acquire() is None
            assert config.LIVE_RUN_LOCK_FILE.read_text(encoding="utf-8") == before
        finally:
            os.close(first)

    def test_an_unwritable_folder_raises(self, tmp_path, monkeypatch):
        # No lock can be taken, so the run must stop with an error, never with
        # EXIT_RUN_IN_PROGRESS (which the scheduler counts as a done slot)
        locked = tmp_path / "read-only"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            monkeypatch.setattr(config, "LIVE_RUN_LOCK_FILE", locked / "sub" / "live_run.lock")
            with pytest.raises(PermissionError):
                run_lock.acquire()
            monkeypatch.setattr(config, "LIVE_RUN_LOCK_FILE", locked / "live_run.lock")
            with pytest.raises(PermissionError):
                run_lock.acquire()
        finally:
            locked.chmod(0o700)

    def test_a_momentary_shared_lock_does_not_stop_a_starting_run(self):
        # held()'s shared lock is taken and dropped at once; acquire() retries
        # through it. Here a shared lock is held for 50 ms of a 200 ms wait.
        config.LIVE_RUN_LOCK_FILE.parent.mkdir(parents=True)
        config.LIVE_RUN_LOCK_FILE.touch()
        probe = os.open(config.LIVE_RUN_LOCK_FILE, os.O_RDONLY)
        fcntl.flock(probe, fcntl.LOCK_SH)
        release = threading.Timer(0.05, os.close, args=(probe,))
        release.start()
        try:
            fd = run_lock.acquire()
            assert fd is not None
            os.close(fd)
        finally:
            release.join()


class TestHeld:
    """held() reports a holder without keeping the lock, and fails closed."""

    def test_no_file_is_free(self):
        assert not config.LIVE_RUN_LOCK_FILE.exists()
        assert run_lock.held() is False

    def test_a_free_lock_is_free(self):
        os.close(run_lock.acquire())
        assert config.LIVE_RUN_LOCK_FILE.exists()
        assert run_lock.held() is False

    def test_a_holder_is_seen(self):
        fd = run_lock.acquire()
        try:
            assert run_lock.held() is True
        finally:
            os.close(fd)
        assert run_lock.held() is False

    def test_held_keeps_no_lock(self):
        # After any number of checks a run can still take the lock at once
        os.close(run_lock.acquire())
        for _ in range(5):
            assert run_lock.held() is False
        fd = run_lock.acquire()
        assert fd is not None
        os.close(fd)

    def test_another_check_at_the_same_moment_is_not_a_run(self):
        # Two checks share the lock, so one never reads the other as a run
        # holding it
        os.close(run_lock.acquire())
        other_check = os.open(config.LIVE_RUN_LOCK_FILE, os.O_RDONLY)
        try:
            fcntl.flock(other_check, fcntl.LOCK_SH)
            assert run_lock.held() is False
        finally:
            os.close(other_check)

    def test_a_file_it_cannot_open_reads_as_held(self):
        # A caller refuses rather than guesses
        os.close(run_lock.acquire())
        config.LIVE_RUN_LOCK_FILE.chmod(0)
        try:
            assert run_lock.held() is True
        finally:
            config.LIVE_RUN_LOCK_FILE.chmod(0o600)

    def test_checks_in_a_loop_never_make_a_concurrent_acquire_fail(self, monkeypatch):
        # A thread checks held() as fast as it can while runs start one after
        # another. Each start retries every millisecond for a second, far more
        # tries than a momentary shared lock can block.
        monkeypatch.setattr(config, "LIVE_RUN_LOCK_WAIT_SECONDS", 1.0)
        monkeypatch.setattr(config, "LIVE_RUN_LOCK_POLL_SECONDS", 0.001)
        os.close(run_lock.acquire())
        stop = threading.Event()
        checks = []

        def check_forever() -> None:
            """
            Call held() until told to stop, counting the calls.

            Runs on its own thread; each call appends to checks, so the test
            can tell the checks really ran alongside the acquires.

            Returns:
                None: It returns once stop is set.
            """
            while not stop.is_set():
                run_lock.held()
                checks.append(1)

        prober = threading.Thread(target=check_forever)
        prober.start()
        try:
            for _ in range(30):
                fd = run_lock.acquire()
                assert fd is not None
                os.close(fd)
        finally:
            stop.set()
            prober.join()
        # Not vacuous: the prober really was checking while the runs started
        assert len(checks) > 30


class TestHolder:
    """holder() reads the record whatever the file holds."""

    def test_a_missing_file_names_no_one(self):
        assert run_lock.holder() == run_lock.LockHolder(None, None, None)

    @pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "null", "42", "\xff\xfe"])
    def test_garbage_names_no_one(self, text):
        config.LIVE_RUN_LOCK_FILE.parent.mkdir(parents=True)
        config.LIVE_RUN_LOCK_FILE.write_bytes(text.encode("latin-1"))
        assert run_lock.holder() == run_lock.LockHolder(None, None, None)

    def test_a_good_record_is_read(self):
        _write_record({"pid": 4242, "checkout": "/Users/me/Kalshi Betting App",
                       "started_at": "2026-09-29T16:00:04Z"})
        holder = run_lock.holder()
        assert holder == run_lock.LockHolder(4242, "/Users/me/Kalshi Betting App",
                                             "2026-09-29T16:00:04Z")
        assert holder.describe() == (
            "process 4242 in /Users/me/Kalshi Betting App, since 2026-09-29T16:00:04Z")

    def test_the_holders_own_record_names_this_checkout(self):
        fd = run_lock.acquire()
        try:
            holder = run_lock.holder()
            assert holder.pid == os.getpid()
            assert str(config.PROJECT_ROOT) in holder.describe()
        finally:
            os.close(fd)

    @pytest.mark.parametrize("record", [
        {"pid": "4242", "checkout": 7, "started_at": 1},
        {"pid": True, "checkout": None, "started_at": None},
        {"pid": 4.5},
        {},
    ])
    def test_a_field_of_the_wrong_type_reads_as_unrecorded(self, record):
        _write_record(record)
        assert run_lock.holder() == run_lock.LockHolder(None, None, None)

    def test_describe_leaves_out_what_is_not_recorded(self):
        assert run_lock.LockHolder(None, None, None).describe() == "another process"
        assert run_lock.LockHolder(7, None, None).describe() == "process 7"
        assert run_lock.LockHolder(None, "/c", None).describe() == "another process in /c"
        assert (run_lock.LockHolder(None, None, "2026-01-01T00:00:00Z").describe()
                == "another process, since 2026-01-01T00:00:00Z")


class TestAgeSeconds:
    """LockHolder.age_seconds() measures from the recorded start, or says it cannot."""

    def test_a_good_time(self):
        holder = run_lock.LockHolder(1, None, "2026-09-29T16:00:04Z")
        now = datetime(2026, 9, 29, 17, 0, 4, tzinfo=UTC)
        assert holder.age_seconds(now) == 3600.0

    def test_now_defaults_to_the_current_time(self):
        started = (datetime.now(UTC) - timedelta(minutes=5)).strftime(run_lock._TIME_FORMAT)
        age = run_lock.LockHolder(1, None, started).age_seconds()
        assert 290 <= age <= 400

    @pytest.mark.parametrize("started", [None, "", "yesterday", "2026-09-29 16:00:04",
                                         "2026-13-40T00:00:00Z"])
    def test_a_missing_or_garbage_time_is_unknown(self, started):
        assert run_lock.LockHolder(1, None, started).age_seconds() is None


class TestAHolderThatDies:
    """The operating system frees the lock when its holder is killed."""

    def test_a_killed_holder_leaves_the_lock_free(self):
        path = config.LIVE_RUN_LOCK_FILE
        path.parent.mkdir(parents=True)
        # The child takes the lock on the path it is handed (it cannot see the
        # monkeypatch) and imports nothing from this package
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import fcntl, os, sys, time\n"
             "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
             "fcntl.flock(fd, fcntl.LOCK_EX)\n"
             "print('locked', flush=True)\n"
             "time.sleep(60)\n",
             str(path)],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert child.stdout.readline().strip() == "locked"
            assert run_lock.held() is True
            assert run_lock.acquire() is None
            child.send_signal(signal.SIGKILL)
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()
        assert run_lock.held() is False
        fd = run_lock.acquire()
        assert fd is not None
        os.close(fd)
