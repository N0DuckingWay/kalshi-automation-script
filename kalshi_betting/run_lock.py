"""
File: run_lock.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Lets only one live trading run place orders at a time on this machine.
    main.py takes the lock for a production run that sends orders (not a dry
    run or a dev run, which send none) and releases it when the run ends. A
    second such run finds it taken and stops before it builds a client, with
    config.EXIT_RUN_IN_PROGRESS. Scheduled runs and runs started by hand both
    go through main.py, from any checkout or worktree, so they all share this
    one lock: they trade one Kalshi account, and two at once would size on
    the same balance and could pick the same pairs.

Dependencies:
    Imports config (the lock file's path, the waits, and PROJECT_ROOT, which
    the holder record names). main.py imports it to take the lock;
    scheduler.py reads its holder record for its exit-50 message.

Notes:
    The lock is an flock on config.LIVE_RUN_LOCK_FILE, read at call time so
    tests point it elsewhere. The operating system releases it when the
    holding process ends in any way, a crash or a kill included, so it can
    never be left stuck; only a run that is still alive but hung can hold it
    for a long time. The file also records the holder's process id, checkout
    and start time, for messages only: whether a run may trade is decided by
    the flock, never by the record. held() is for a caller that only wants
    to know whether a run is trading: it takes a shared lock and drops it at
    once. acquire() keeps retrying for config.LIVE_RUN_LOCK_WAIT_SECONDS
    before it gives up, so such a momentary check can never make a starting
    run stop. An flock belongs to one open file, so two opens of the lock
    file in one process also exclude each other, which is what lets the
    tests hold the lock in-process.
"""
import fcntl
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from . import config

# How the holder records when it took the lock (UTC)
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class LockHolder:
    """
    Who holds the run lock, as its file records it.

    Attributes:
        pid (int | None): The holder's process id; None when the file does not say.
        checkout (str | None): The checkout the holder runs from; None when not recorded.
        started_at (str | None): When it took the lock, as UTC text in
            _TIME_FORMAT; None when not recorded.
    """
    pid: int | None
    checkout: str | None
    started_at: str | None

    def describe(self) -> str:
        """
        Say who holds the lock in one phrase, for a log line or a page.

        Returns:
            str: e.g. "process 4242 in /Users/me/Kalshi Betting App, since
                2026-09-29T16:00:04Z"; "another process" stands in for an
                unrecorded pid, and an unrecorded checkout or start time is
                left out.
        """
        text = f"process {self.pid}" if self.pid is not None else "another process"
        if self.checkout:
            text += f" in {self.checkout}"
        return text + (f", since {self.started_at}" if self.started_at else "")

    def age_seconds(self, now: datetime | None = None) -> float | None:
        """
        How long ago the holder took the lock.

        Args:
            now (datetime | None): The time to measure to, timezone-aware;
                None (default) is now, in UTC.

        Returns:
            float | None: Seconds since started_at; None when it is missing or unreadable.
        """
        try:
            started = datetime.strptime(self.started_at or "", _TIME_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            return None
        return ((now or datetime.now(UTC)) - started).total_seconds()


def acquire() -> int | None:
    """
    Take the run lock, retrying for a moment if another process holds it.

    Creates the lock file's folder when needed, then tries an exclusive flock
    every config.LIVE_RUN_LOCK_POLL_SECONDS until config.LIVE_RUN_LOCK_WAIT_SECONDS
    have passed. Once it holds the lock, it rewrites the file with this
    process's id, checkout (config.PROJECT_ROOT) and start time. Any error
    making or opening the file propagates: the run then stops with an error
    (exit 1), never with EXIT_RUN_IN_PROGRESS, which the scheduler counts as
    a done slot.

    Returns:
        int | None: The open file descriptor holding the lock (close it to
            release the lock), or None when another process still holds it
            after config.LIVE_RUN_LOCK_WAIT_SECONDS.

    Raises:
        OSError: When the lock file's folder or the file cannot be made or
            opened, or the lock or the record cannot be written.
    """
    path = config.LIVE_RUN_LOCK_FILE
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + config.LIVE_RUN_LOCK_WAIT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(config.LIVE_RUN_LOCK_POLL_SECONDS)
        # Who holds it, for another run's refusal and the scheduler's message
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({
            "pid": os.getpid(), "checkout": str(config.PROJECT_ROOT),
            "started_at": datetime.now(UTC).strftime(_TIME_FORMAT),
        }).encode("utf-8"))
        return fd
    except BaseException:
        os.close(fd)
        raise


def held() -> bool:
    """
    Tell whether a live trading run holds the lock now, without keeping it.

    Takes a shared lock and drops it at once, so a run starting at the same
    moment finds the lock taken for an instant only, which acquire()'s retry
    rides out.

    Returns:
        bool: True when the lock is held through any other open of the file
            (in practice another run's process; two opens in one process
            exclude each other too), or when the lock cannot be checked (a
            permission error, say), so a caller refuses rather than guesses;
            False when no lock file exists or it is free.
    """
    try:
        fd = os.open(config.LIVE_RUN_LOCK_FILE, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)  # closed once on every path, which also drops the shared lock


def holder() -> LockHolder:
    """
    Read the lock file's record of the current holder.

    The record is the last one a run wrote, so it names the current holder
    only while the lock is held; it is for messages, never for deciding
    whether a run may trade.

    Returns:
        LockHolder: The recorded pid, checkout and start time; each None when
            the file is missing, unreadable or does not say.
    """
    try:
        record = json.loads(config.LIVE_RUN_LOCK_FILE.read_text(encoding="utf-8"))
        pid, checkout, started = (record.get("pid"), record.get("checkout"),
                                  record.get("started_at"))
    except (OSError, ValueError, AttributeError):
        return LockHolder(None, None, None)
    return LockHolder(pid if type(pid) is int else None,
                      checkout if isinstance(checkout, str) else None,
                      started if isinstance(started, str) else None)
