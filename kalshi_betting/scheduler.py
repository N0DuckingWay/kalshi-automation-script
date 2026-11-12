"""
File: scheduler.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    A long-running daemon that runs the production bot every day at
    config.SCHEDULER_RUN_TIMES' times (07:00 and 21:00) on the host's clock.
    It polls `schedule` every 60 seconds, logs to its own
    kalshi_scheduler.log, and records each run's slot in scheduler_state.json
    so a restart can catch up on a slot missed while it was down.

Dependencies:
    Imports PROJECT_ROOT, SCHEDULER_RUN_TIMES, SCHEDULER_JOB_TIMEOUT_SECONDS,
    SCHEDULER_BLIND_RETRY_SECONDS / SCHEDULER_BLIND_MAX_RETRIES,
    SCHEDULER_STATE_FILENAME (the slot record's file name), and the
    EXIT_OK / EXIT_SKIPPED_LOW_BALANCE / EXIT_TRADES_NEED_ATTENTION /
    EXIT_NO_TRADEABLE_SHARDS / EXIT_TIME_SERIES_SKIPPED / EXIT_RUN_IN_PROGRESS
    exit-code constants from config.py — the EXIT_* imports are what let run_job() map the
    subprocess's exit code to a distinct log level/message rather than
    treating every nonzero code identically — and read_saved_live_defaults,
    which _check_live_defaults reads once at daemon start. Imports run_lock,
    whose holder record names the run that stopped a scheduled run with
    EXIT_RUN_IN_PROGRESS. Spawns kalshi_betting.main as a
    subprocess (via sys.executable) rather than importing it directly, to
    isolate run-time errors and capture stdout/stderr separately. Entry point for
    `python3 -m kalshi_betting.scheduler`.

Notes:
    The scheduler runs the bot in production mode (--mode prod) with no live
    toggle flag, so a scheduled run trades exactly the saved live defaults
    (live_defaults.json). For the bot to trade, valid prod credentials must be
    present in secrets.json and the PEM key file, and live defaults must be
    saved: with none saved, or a saved file that is refused, every run exits 2
    (logged as "Job failed (exit 2)", and the slot is spent), and main() logs
    an ERROR saying so as soon as the daemon starts (_check_live_defaults). To
    change the run times, edit config.SCHEDULER_RUN_TIMES. The backtest does
    not follow them: it enters trades once a week, at config.SCHEDULED_RUN.

    Host clock: `schedule` fires on the computer's local clock and must stay
    that way, never `.at(time, tz)` (see _daily_jobs). At startup
    _host_clock_realises_run() checks that clock against SCHEDULER_RUN_TIMES'
    zone and logs CRITICAL on a mismatch; the daemon fires either way.

    Log file: kalshi_scheduler.log, never kalshi_arb.log, which the spawned
    bot rotates (see _SCHEDULER_LOG_PATH).

    Every job is registered through _guarded_job(): `schedule` reschedules a
    job only after it returns, so a job that raised would re-run on every
    poll tick. A one-shot retry that raises is cancelled instead.

    Slots: run_job() records its slot in scheduler_state.json before spawning
    the bot, so a crashed or hung run still counts as attempted, and
    finalizes it after a success, a nonzero exit, a timeout or an OSError.
    _save_state() raises on purpose: a failed claim must stop the spawn. At
    startup _maybe_catch_up() (via _startup_catch_up, which never stops the
    daemon) re-runs a slot with no recorded attempt; with no state file yet,
    the first start runs the bot. An ordinary fire whose slot is already
    recorded spawns nothing (_slot_already_run), so a host that slept through
    both run times trades once on waking, not twice.

    Lock refusals: exit code EXIT_RUN_IN_PROGRESS (50) means another live
    trading run on this machine held the live-run lock (run_lock), so the
    scheduled run stopped before making any request. run_job() counts the
    slot as done and never retries it, since a real-money run was already
    trading; it logs a WARNING naming that run, or an ERROR when that run has
    held the lock for over SCHEDULER_JOB_TIMEOUT_SECONDS, or its start is not
    recorded, and so may be hung.

    Blind runs: exit code EXIT_NO_TRADEABLE_SHARDS (30) means nothing was
    scanned (kalshi_arb.log says why). run_job() then schedules a one-shot
    retry SCHEDULER_BLIND_RETRY_SECONDS later, which can schedule the next,
    at most SCHEDULER_BLIND_MAX_RETRIES per slot (counted in `retries` and
    the state file's "retries" key); _maybe_catch_up() re-runs a blind slot
    too.

    Subprocess quirks: TimeoutExpired's .stdout/.stderr are bytes even with
    text=True, so _decode() normalizes both before logging; a bare OSError
    from subprocess.run() (a missing interpreter, a fork failure, a bad cwd)
    is logged as "Failed to spawn" instead of escaping run_job().
"""
import json
import logging
import logging.handlers
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import schedule

from . import run_lock
from .config import (
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    PROJECT_ROOT,
    SCHEDULER_BLIND_MAX_RETRIES,
    SCHEDULER_BLIND_RETRY_SECONDS,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
    SCHEDULER_RUN_TIMES,
    SCHEDULER_STATE_FILENAME,
    LiveDefaultsOutdated,
    read_saved_live_defaults,
)

# Schema version for scheduler_state.json — bump if the record shape changes
# so a future reader can distinguish old files instead of guessing.
_STATE_SCHEMA_VERSION = 1

# The daemon's own log file, deliberately NOT kalshi_arb.log: that file is
# rotated by the main.py subprocess this daemon spawns, and a second process
# holding an open FileHandler on it keeps writing into the renamed backup
# (kalshi_arb.log.1) after each rotation — its lines silently disappear from
# the live log. A module-level constant is fine here (unlike
# _state_file_path()) because main() resolves it once at daemon startup and
# nothing in the tests needs it redirected.
_SCHEDULER_LOG_PATH = PROJECT_ROOT / "kalshi_scheduler.log"

# How many days of runs _host_clock_realises_run() checks at daemon start: two
# years, so the check spans at least three clock changes, in both directions
# (a host on other daylight-saving rules may disagree only after one).
_HOST_CLOCK_CHECK_DAYS = 730


def _state_file_path() -> Path:
    """
    Return the path to the persisted scheduler state file.

    A function rather than a module-level constant so tests can monkeypatch
    `scheduler.PROJECT_ROOT` to a tmp_path and have state reads/writes follow
    it — a constant computed once at import time would freeze the path
    before any test patch could take effect.

    Returns:
        Path: PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME
            ("scheduler_state.json").
    """
    return PROJECT_ROOT / SCHEDULER_STATE_FILENAME


def _decode(stream) -> str:
    """
    Normalize a subprocess.TimeoutExpired stdout/stderr stream to str.

    See the CPython quirk documented in this module's Notes: even with
    text=True, TimeoutExpired's captured streams arrive as bytes, not str.

    Args:
        stream (bytes | str | None): The raw stdout/stderr attribute from a
            caught subprocess.TimeoutExpired.

    Returns:
        str: Decoded text (invalid bytes replaced, never raises), or "" if
            stream is None.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def _host_clock_label() -> str:
    """
    Describe when the daemon fires, for log lines: its daily times on the host's clock.

    Returns:
        str: e.g. "every day at 07:00 and 21:00 on the host's clock".
    """
    return f"every day at {SCHEDULER_RUN_TIMES.times_label()} on the host's clock"


def _daily_jobs() -> list[schedule.Job]:
    """
    Build one daily `schedule` job per SCHEDULER_RUN_TIMES time.

    main() registers each with `.do(_guarded_job, run_job)`. They fire by the
    host's clock and must stay that way, never `.at(time, tz)`: that option
    goes through pytz, whose America/Los_Angeles table has no daylight-saving
    time after 2037. _host_clock_realises_run() checks the host's clock instead.

    Returns:
        list[schedule.Job]: `schedule.every().day.at("HH:MM")` for each time,
            earliest first, not yet registered.
    """
    return [schedule.every().day.at(at_time) for at_time in SCHEDULER_RUN_TIMES.at_times()]


def _cron_lines(project_path: str, python_path: str) -> list[str]:
    """
    Build the crontab lines main() logs, running the bot at the daemon's daily times.

    cron also fires by the host's clock. Times that share a minute share a
    line ("0 7,21 * * *"); each other minute gets a line of its own.

    Args:
        project_path (str): The repo root to cd into.
        python_path (str): The interpreter that runs kalshi_betting.main.

    Returns:
        list[str]: e.g. ["0 7,21 * * * cd '<project>' && <python> -m
            kalshi_betting.main --mode prod >> /tmp/kalshi_arb.log 2>&1"],
            ordered by minute.
    """
    hours_by_minute: dict[int, list[int]] = {}
    for hour, minute in SCHEDULER_RUN_TIMES.times:
        hours_by_minute.setdefault(minute, []).append(hour)
    return [
        f"{minute} {','.join(str(hour) for hour in hours)} * * * "
        f"cd '{project_path}' && {python_path} -m kalshi_betting.main --mode prod "
        f">> /tmp/kalshi_arb.log 2>&1"
        for minute, hours in sorted(hours_by_minute.items())
    ]


def _host_clock_realises_run(today: date | None = None) -> bool:
    """
    Check that the host's clock fires each run at the UTC moment SCHEDULER_RUN_TIMES names.

    The daemon fires when the host's clock reads one of SCHEDULER_RUN_TIMES'
    times. That is the moment the schedule names (DailyRuns.on(d), then
    ScheduledRun.instant(d)) only if the host keeps the schedule's zone;
    otherwise the bot trades at other hours than the ones set. Checks every
    run in the next _HOST_CLOCK_CHECK_DAYS days: logs CRITICAL on the first
    mismatch, on run times a clock change skips, or if the zone or a date
    cannot be checked; INFO when all match. Never raises, so main()'s
    catch-up and job registration always run.

    Args:
        today (date | None): First date to check. Defaults to date.today().

    Returns:
        bool: True when every checked run lands on its scheduled UTC moment.
    """
    runs = SCHEDULER_RUN_TIMES
    try:
        runs.zone()
    except Exception as exc:
        # Broad on purpose: this check must never stop the daemon.
        logging.critical(
            "config.SCHEDULER_RUN_TIMES' zone %r cannot be resolved (%s: %s), so the "
            "host's clock cannot be checked against it. The daemon still fires "
            "%s.",
            runs.timezone, type(exc).__name__, exc, _host_clock_label(),
        )
        return False
    start = today if today is not None else date.today()
    skipped: list[str] = []
    mismatch: tuple[date, str, datetime, datetime] | None = None
    try:
        for offset in range(_HOST_CLOCK_CHECK_DAYS):
            d = start + timedelta(days=offset)
            for run in runs.on(d):
                if run.clock_change(d) == "skipped":
                    # The zone skips this wall time, so a clock in that zone
                    # never reads it on d: a problem with the schedule, not
                    # the host.
                    skipped.append(f"{d.isoformat()} {run.at_time()}")
                    continue
                # A naive datetime converts to UTC by the host's own rules:
                # the moment the daemon fires.
                fired = datetime(d.year, d.month, d.day, run.hour, run.minute).astimezone(UTC)
                scheduled = run.instant(d)
                if fired != scheduled:
                    mismatch = (d, run.at_time(), fired, scheduled)
                    break
            if mismatch is not None:
                break
    except Exception as exc:
        # Broad on purpose (an out-of-range date raises OverflowError, the
        # host's clock may raise OSError): never stop the daemon.
        logging.critical(
            "The host's clock could not be checked against config.SCHEDULER_RUN_TIMES "
            "(%s) from %s (%s: %s). The daemon still fires %s.",
            runs.label(), start.isoformat(), type(exc).__name__, exc, _host_clock_label(),
        )
        return False
    if skipped:
        logging.critical(
            "config.SCHEDULER_RUN_TIMES (%s) names a wall time its zone skips at a "
            "clock change (%s), so no host's clock fires those runs at their "
            "scheduled instant. Move config.SCHEDULER_RUN_TIMES off the clock "
            "change. The daemon still fires %s.",
            runs.label(), ", ".join(skipped), _host_clock_label(),
        )
    if mismatch is not None:
        d, at_time, fired, scheduled = mismatch
        logging.critical(
            "The host's clock does not keep config.SCHEDULER_RUN_TIMES (%s): on "
            "%s the %s run fires at %s UTC, not %s UTC, so the bot trades at "
            "other hours than the ones set. It still fires %s; set the host's "
            "time zone to %s, or change config.SCHEDULER_RUN_TIMES.",
            runs.label(), d.isoformat(), at_time, f"{fired:%Y-%m-%d %H:%M}",
            f"{scheduled:%Y-%m-%d %H:%M}", _host_clock_label(), runs.timezone,
        )
    if skipped or mismatch is not None:
        return False
    logging.info(
        "The host's clock keeps config.SCHEDULER_RUN_TIMES (%s) for the next %d days.",
        runs.label(), _HOST_CLOCK_CHECK_DAYS,
    )
    return True


def _most_recent_slot(now: datetime) -> datetime:
    """
    Compute the latest run time ("slot") at or before `now`, on the host's clock.

    run_job() records it in scheduler_state.json, and _maybe_catch_up() uses
    it to find the slot that should already have a recorded run.

    Args:
        now (datetime): Naive local datetime to evaluate against.

    Returns:
        datetime: The most recent slot that should already have fired (naive,
            local, seconds/microseconds zeroed): today's latest run time at or
            before `now`, or yesterday's last run time when `now` is before
            today's first.
    """
    today = [now.replace(hour=hour, minute=minute, second=0, microsecond=0)
             for hour, minute in SCHEDULER_RUN_TIMES.times]
    passed = [slot for slot in today if slot <= now]
    if passed:
        # The times are earliest first, so the last one passed is the latest
        return passed[-1]
    # Before today's first run time: yesterday's last slot
    return today[-1] - timedelta(days=1)


def _load_state() -> dict | None:
    """
    Load the persisted scheduler state, tolerating absence or corruption.

    Returns None whenever the file can't be trusted to represent a real
    prior run — missing, unreadable, not valid JSON, or missing/unparseable
    `last_slot`. This is deliberately the same outcome as "no prior run is
    known": a corrupt state file must not silently suppress the BS-17
    catch-up check.

    The two optional numeric keys, `retries` and `exit_code`, get a WARNING
    of their own (DR-24), but degrade IN PLACE rather than discarding the
    whole state the way an unparseable `last_slot` does: a non-int `retries`
    (a hand-edited string, `null`) is read as 0, and a present-but-non-int
    `exit_code` is read as None. This closes a real crash: `_maybe_catch_up`
    compares `retries` with `<`, so a corrupt value there used to raise
    `TypeError` out of the daemon at startup before the run jobs were ever
    registered. A corrupt `exit_code` never raised — it only ever compared
    unequal to `EXIT_NO_TRADEABLE_SHARDS` via `==` — so typing it here is
    hygiene, not a crash fix; it is validated anyway so the dict always
    carries a typed value. A `retries` key that is simply ABSENT is
    untouched by this and is still read as 0 by every caller's own
    `.get("retries", 0)`.

    Returns:
        dict | None: The parsed state dict (guaranteed to have a parseable
            `last_slot`, and int-or-absent `retries` / int-or-None
            `exit_code`), or None.
    """
    path = _state_file_path()
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logging.warning(
            "Corrupt scheduler state file %s (%s) — treating as no prior run",
            path, exc,
        )
        return None
    if not isinstance(state, dict) or "last_slot" not in state:
        logging.warning(
            "Scheduler state file %s is missing expected fields — "
            "treating as no prior run",
            path,
        )
        return None
    try:
        datetime.fromisoformat(state["last_slot"])
    except (TypeError, ValueError) as exc:
        logging.warning(
            "Scheduler state file %s has an unparseable last_slot (%s) — "
            "treating as no prior run",
            path, exc,
        )
        return None
    # DR-24: `retries` is compared with `<` in _maybe_catch_up; a hand-edited
    # string or null there used to raise TypeError out of main() before the
    # run jobs were ever registered, killing the daemon at startup. A bad
    # value degrades to "unknown" instead — bool is a subclass of int, so
    # True/False are accepted as-is. `exit_code` is compared with `==`,
    # which never raises on a type mismatch — its guard below is typing
    # hygiene, not a crash fix, kept so the dict always carries a typed
    # value.
    if "retries" in state and not isinstance(state["retries"], int):
        logging.warning(
            "Scheduler state file %s has a non-integer retries (%r) — reading as 0",
            path, state["retries"],
        )
        state["retries"] = 0
    if state.get("exit_code") is not None and not isinstance(state["exit_code"], int):
        logging.warning(
            "Scheduler state file %s has a non-integer exit_code (%r) — reading as unknown",
            path, state["exit_code"],
        )
        state["exit_code"] = None
    return state


def _save_state(
    *,
    last_slot: datetime,
    started_at: datetime,
    finished_at: datetime | None,
    exit_code: int | None,
    retries: int = 0,
) -> None:
    """
    Persist scheduler run state atomically (tmp file + rename).

    Called twice per run_job() invocation: once to CLAIM the slot before the
    subprocess is spawned (finished_at=None, exit_code=None), and once to
    finalize it afterwards on every exit path. Same tmp+replace idiom as
    historical.py's `_save_json_cache` / `_day_store_save` — the write goes
    to a sibling temp file that is renamed over the destination, so a crash
    mid-write can never leave a truncated state file that a later run (or
    the catch-up check) would read back and trust.

    Args:
        last_slot (datetime): The run slot (a SCHEDULER_RUN_TIMES time on
            the host's clock) this run is satisfying.
        started_at (datetime): When this run began (claim time).
        finished_at (datetime | None): When this run ended, or None if the
            run has only been claimed (still in progress).
        exit_code (int | None): The subprocess exit code, or None if the run
            has only been claimed, or ended via timeout/OSError before a
            subprocess exit code existed.
        retries (int): How many blind-run retries (TS-01) preceded this
            attempt at this slot. Defaults to 0 — the ordinary first run —
            so existing callers are unaffected.
    """
    state = {
        "schema": _STATE_SCHEMA_VERSION,
        "last_slot": last_slot.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at is not None else None,
        "exit_code": exit_code,
        # Blind-run retry count for this slot (TS-01); absent in pre-existing
        # state files, which _load_state still accepts unchanged and every
        # reader treats as 0 via `.get("retries", 0)` — _load_state now also
        # degrades a PRESENT but non-int retries/exit_code to unknown
        # (0 / None) with a WARNING (DR-24), but that validation never fires
        # on this well-typed write.
        "retries": retries,
    }
    path = _state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _guarded_job(
    job: Callable[..., Any],
    *args: Any,
    on_error: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Invoke a scheduled job so that no exception can escape into `schedule`.

    This is the wrapper EVERY job in this module is registered through, and it
    sits at the registration boundary rather than inside any one helper on
    purpose (DR-59). `schedule.Job.run()` assigns `last_run` and calls
    `_schedule_next_run()` only AFTER `job_func()` RETURNS, so an exception
    that escapes the job leaves `next_run` in the past. main()'s poll loop
    catches it ("Scheduler tick raised — daemon continues") and keeps the
    daemon alive, but the job is still overdue — so it is re-entered on the
    very next 60 s poll tick, turning a twice-daily production trading run into
    a once-a-minute one for as long as the fault persists. The same mechanism
    defeats the blind-retry one-shot contract: _blind_retry signals "run me
    once" by RETURNING schedule.CancelJob, and a retry that raises never
    returns, so it is never deregistered AND the SCHEDULER_BLIND_MAX_RETRIES
    cap — carried solely on the `retries` argument each attempt hands its
    successor — never advances.

    Catching here covers every escape path a job has (a state-file OSError
    from the slot claim or finalize, an undecodable child stream out of
    subprocess.run's text=True decoding, anything added later), which is
    exactly why the guard is not buried in one of them. The individual
    helpers keep their current semantics: in particular _save_state() still
    raises, so a failed slot CLAIM still prevents the subprocess spawn.

    KeyboardInterrupt and SystemExit are NOT subclasses of Exception and are
    deliberately not caught — Ctrl-C must still stop the daemon.

    Args:
        job (Callable[..., Any]): The job to invoke. `schedule`'s
            `.do(_guarded_job, job, *args, **kwargs)` forwards everything
            after the first argument straight through to this call.
        *args (Any): Positional arguments forwarded to `job`.
        on_error (Any): The value to return when `job` raises. Defaults to
            None. A one-shot job passes `schedule.CancelJob` here so that a
            RAISING attempt still deregisters itself — `schedule` decides
            that on the value returned from this wrapper, not on the identity
            of the registered function. Keyword-only, so it can never collide
            positionally with a job's own arguments; it does, however, RESERVE
            that name — a future job function with its own `on_error` keyword
            would have it consumed here instead of forwarded. Neither
            `run_job` nor `_blind_retry` has one.
        **kwargs (Any): Keyword arguments forwarded to `job`.

    Returns:
        Any: Whatever `job` returned, or `on_error` if it raised. The raising
            case is logged once, with traceback, at ERROR level.
    """
    try:
        return job(*args, **kwargs)
    except Exception:
        # Deliberately says only what is true of BOTH registrations: the
        # recurring daily job is rescheduled for its next day, while a
        # raising blind retry is deregistered by on_error=CancelJob and
        # its retry chain for this slot ends. Claiming "the next scheduled
        # fire is unaffected" would read, after a failing retry, as though
        # another retry were still queued.
        logging.exception(
            "Scheduled job %s raised — this attempt is abandoned; "
            "the daemon keeps polling.",
            getattr(job, "__name__", job),
        )
        return on_error


def _slot_already_run(slot: datetime) -> bool:
    """
    Say whether scheduler_state.json already records a run for `slot` (or a later one).

    `schedule` fires every overdue job in one poll tick. When the host sleeps
    (or the daemon is paused) through two run times, both daily jobs are
    overdue on waking and fire one after the other, and both would take the
    same slot (_most_recent_slot reads the wall clock). run_job() calls this
    for an ordinary fire (retries == 0) so the second one does not trade
    again: one scheduled run per slot. A blind-run retry never asks, since it
    re-runs a slot on purpose. An unreadable state file reads as "not run".

    Args:
        slot (datetime): The slot the run would claim (naive, host clock).

    Returns:
        bool: True when the recorded last_slot is at or after `slot`; the
            skip is logged at INFO.
    """
    state = _load_state()
    if state is None:
        return False
    recorded = datetime.fromisoformat(state["last_slot"])
    if recorded < slot:
        return False
    logging.info(
        "Scheduler: the %s slot already has a recorded run (last slot %s), so this "
        "fire is skipped — one scheduled run per slot (two run times passed while "
        "the host slept or the daemon was paused).",
        slot.isoformat(), recorded.isoformat(),
    )
    return True


def run_job(retries: int = 0) -> None:
    """
    Execute a single production arbitrage bot run as an isolated subprocess.

    Spawns `python -m kalshi_betting.main --mode prod` using the same Python
    interpreter that is running this scheduler. Running in a subprocess isolates
    any unhandled exceptions in the bot from the scheduler process — a crash in
    one run does not prevent future runs from being triggered. A hung run is
    killed after SCHEDULER_JOB_TIMEOUT_SECONDS so it cannot block the daemon
    past the next scheduled fire.

    Claims its slot (_most_recent_slot) in scheduler_state.json before spawning the
    subprocess, and finalizes that record on every exit path — success,
    nonzero exit, timeout, or OSError (BS-17). See this module's Notes for
    why claiming happens up front. An ordinary fire (retries == 0) whose slot
    already has a recorded run returns at once, spawning nothing
    (_slot_already_run): both daily jobs fire on waking from a sleep that
    spanned both run times, and only the first may trade.

    Logs the subprocess's stdout unconditionally (whether the run succeeded or
    failed). Its stderr is logged only on the catch-all failure branch (an
    exit code that is none of the EXIT_* codes) and on TimeoutExpired; every
    EXIT_* branch logs its message without stderr. All of that re-logged
    output lands in THIS daemon's log (kalshi_scheduler.log), while the bot's
    own log lines go to kalshi_arb.log through the subprocess's own handler.
    The exit code is mapped to a distinct log level/message per the EXIT_*
    contract in config.py (BS-14): a low-balance skip and a run with trades
    needing manual review are no longer indistinguishable from a clean run in
    this log — previously the only signal was a WARNING inside kalshi_arb.log
    that this scheduler process never reads. EXIT_TIME_SERIES_SKIPPED is an
    ERROR that still counts the slot as done. EXIT_RUN_IN_PROGRESS (another
    live trading run held the live-run lock) counts the slot as done too and
    is never retried: a WARNING naming that run (run_lock.holder), or an
    ERROR when it has held the lock for over SCHEDULER_JOB_TIMEOUT_SECONDS or
    its start is not recorded, since it may then be hung and every scheduled
    run stops the same way until it ends.

    A subprocess.TimeoutExpired's stdout/stderr are decoded before logging
    (BS-16 — see _decode()), and both streams are logged (stderr, the hung
    run's traceback, was previously dropped entirely). A bare OSError from
    subprocess.run() itself (BS-31 — e.g. the interpreter can't be spawned)
    is logged with a specific message and does not escape this function.

    EXIT_NO_TRADEABLE_SHARDS (TS-01, VI-02) is the one code that does NOT
    satisfy the slot: it means nothing was scanned at all — an
    exchange-wide halt, or an ingest that came back empty for a cause
    /exchange/status could not name — and the bot trades only at its
    scheduled runs, so letting it stand would cost that run. Which of the two fired is
    named in kalshi_arb.log by the subprocess, not here: this daemon sees only
    the exit code, so its own messages must not claim one cause over the
    other. That branch registers a one-shot retry on the `schedule` library's
    global scheduler, SCHEDULER_BLIND_RETRY_SECONDS out, and the retry
    re-enters this function with `retries` incremented. The cap lives HERE,
    not in the caller: a retried run that exits 30 again schedules its own
    successor, so SCHEDULER_BLIND_MAX_RETRIES can only be enforced by the
    argument each attempt carries. That retry is registered through
    _guarded_job with on_error=schedule.CancelJob, so an attempt that RAISES
    still deregisters itself instead of becoming a recurring job that never
    advances the count (DR-59).

    This function may raise: the slot CLAIM's _save_state() sits outside the
    try below, the finalizing _save_state() sits after it, and
    subprocess.run(..., text=True) decodes the child's streams strictly. That
    is deliberate — a failed claim must prevent the spawn — and it is why
    every registration of this function goes through _guarded_job(), which
    keeps an escaping exception from freezing the job's next_run (DR-59).

    Args:
        retries (int): How many blind-run retries have already been spent on
            this slot. 0 for the ordinary scheduled fire and for a
            catch-up of a never-attempted slot; greater than 0 only when
            _blind_retry or _maybe_catch_up is re-running a slot that a blind
            run left unsatisfied. Persisted into scheduler_state.json so the
            count survives a daemon restart.

    Returns:
        None
    """
    started_at = datetime.now()
    slot = _most_recent_slot(started_at)
    if retries == 0 and _slot_already_run(slot):
        return
    logging.info("Scheduler: starting scheduled arbitrage scan.")
    # Claim the slot before spawning, so a crash/timeout/OSError mid-run
    # still leaves a recorded attempt for this slot — otherwise the startup
    # catch-up check would see no record at all and re-run it on every
    # restart until one attempt happens to finish cleanly.
    _save_state(last_slot=slot, started_at=started_at, finished_at=None,
                exit_code=None, retries=retries)

    try:
        # Use sys.executable to ensure the subprocess uses the same Python environment
        # (venv, conda, etc.) as the scheduler itself
        result = subprocess.run(
            [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod"],
            capture_output=True,
            text=True,
            # Run from the project root so relative paths in main.py resolve correctly
            cwd=str(PROJECT_ROOT),
            # Kill a hung run rather than blocking the daemon forever
            timeout=SCHEDULER_JOB_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logging.error(
            "Job killed after %ds timeout.\nPartial stdout:\n%s\nPartial stderr:\n%s",
            SCHEDULER_JOB_TIMEOUT_SECONDS, _decode(exc.stdout), _decode(exc.stderr),
        )
        _save_state(
            last_slot=slot, started_at=started_at,
            finished_at=datetime.now(), exit_code=None, retries=retries,
        )
        return
    except OSError as exc:
        logging.error("Failed to spawn bot subprocess: %s", exc)
        _save_state(
            last_slot=slot, started_at=started_at,
            finished_at=datetime.now(), exit_code=None, retries=retries,
        )
        return

    if result.stdout:
        logging.info("stdout:\n%s", result.stdout)

    if result.returncode == EXIT_OK:
        logging.info("Job completed successfully.")
    elif result.returncode == EXIT_SKIPPED_LOW_BALANCE:
        logging.warning("Job skipped: balance below minimum — no trades attempted.")
    elif result.returncode == EXIT_TRADES_NEED_ATTENTION:
        logging.error(
            "Job completed but one or more trades need MANUAL REVIEW — "
            "check kalshi_arb.log and trade_log.xlsx.",
        )
    elif result.returncode == EXIT_TIME_SERIES_SKIPPED:
        # Slot done: the run scanned the markets, and a retry would most likely
        # fail the same lookup
        logging.error(
            "Job bought NO pair (exit %d): a held market could not be identified, "
            "so the run could not tell which ladders it already holds. The slot "
            "counts as done — see the ERROR in kalshi_arb.log naming the market.",
            result.returncode,
        )
    elif result.returncode == EXIT_RUN_IN_PROGRESS:
        # Slot done, never retried: another real-money run held the live-run
        # lock. The lock file's record names that run
        holder = run_lock.holder()
        age = holder.age_seconds()
        if age is not None and age <= SCHEDULER_JOB_TIMEOUT_SECONDS:
            logging.warning(
                "Job did not trade (exit %d): another live trading run on this machine "
                "was in progress (%s). The slot counts as done — a real-money "
                "run was already trading, so it is not retried.",
                result.returncode, holder.describe(),
            )
        else:
            logging.error(
                "Job did not trade (exit %d): the run holding the live-run lock (%s) has "
                "been running for over %d s, or its start is not recorded — it may be "
                "hung or stopped. No trade was made for this slot, and every "
                "scheduled run will stop the same way until that run ends. Before "
                "stopping it, check the account in the Kalshi UI: a run stopped "
                "mid-trade can leave a leg unhedged.",
                result.returncode, holder.describe(), SCHEDULER_JOB_TIMEOUT_SECONDS,
            )
    elif result.returncode == EXIT_NO_TRADEABLE_SHARDS:
        # Not a satisfied slot: nothing was scanned (TS-01, VI-02). The exit
        # code does not say WHICH cause fired, so neither does this message —
        # the subprocess logs that sentence into kalshi_arb.log.
        if retries < SCHEDULER_BLIND_MAX_RETRIES:
            logging.warning(
                "Job scanned nothing (exit %d): every advertised exchange shard was "
                "trading-inactive, or the market ingest came back empty — "
                "kalshi_arb.log names which. The slot is NOT satisfied — "
                "retrying in %d s (attempt %d of %d).",
                result.returncode, SCHEDULER_BLIND_RETRY_SECONDS,
                retries + 1, SCHEDULER_BLIND_MAX_RETRIES,
            )
            # One-shot job on the schedule library's global scheduler; it
            # cancels itself when it fires (see _blind_retry), and the next
            # attempt decides for itself whether to schedule another.
            # Registered through _guarded_job with on_error=CancelJob so a
            # RAISING retry still deregisters itself (DR-59): `schedule`
            # cancels on the value RETURNED to it, and a retry that raises
            # returns nothing — it would stay registered, re-fire every poll
            # tick, and never advance the SCHEDULER_BLIND_MAX_RETRIES count,
            # which rides solely on the `retries` argument below.
            schedule.every(SCHEDULER_BLIND_RETRY_SECONDS).seconds.do(
                _guarded_job, _blind_retry, retries + 1,
                on_error=schedule.CancelJob,
            )
        else:
            logging.error(
                "Job scanned nothing on %d attempts: every advertised exchange shard "
                "stayed trading-inactive, or the market ingest kept coming back empty "
                "— giving up on this slot; check the exchange status and "
                "kalshi_arb.log, which names the cause.",
                retries + 1,
            )
    else:
        logging.error("Job failed (exit %d):\n%s", result.returncode, result.stderr)

    _save_state(
        last_slot=slot, started_at=started_at,
        finished_at=datetime.now(), exit_code=result.returncode, retries=retries,
    )


def _blind_retry(retries: int) -> type[schedule.CancelJob]:
    """
    One-shot retry of a blind run (TS-01).

    Returning schedule.CancelJob removes this job after it fires, so the retry
    never recurs on its own; run_job decides whether to schedule another and
    enforces SCHEDULER_BLIND_MAX_RETRIES.

    Args:
        retries (int): The attempt number this retry represents — passed
            straight through to run_job, which persists it and uses it to
            enforce the cap.

    Returns:
        schedule.CancelJob: The library's sentinel meaning "deregister me".
    """
    run_job(retries=retries)
    return schedule.CancelJob


def _maybe_catch_up(now: datetime | None = None) -> None:
    """
    Run a catch-up job now if the latest run slot has no recorded run, or
    its last run was blind and the blind-retry cap is not yet used up.

    `schedule` fires only while this process polls, so a slot that passes
    while the daemon is down would otherwise be skipped. main() calls this
    once at startup, through _startup_catch_up(). A blind run (one that
    exited EXIT_NO_TRADEABLE_SHARDS) is re-run with `retries + 1`, sharing
    run_job's SCHEDULER_BLIND_MAX_RETRIES cap; a run that failed any other
    way counts as done and is not re-run.

    Args:
        now (datetime | None): Override for the current local time, for
            testing. Defaults to datetime.now().

    Returns:
        None
    """
    now = now if now is not None else datetime.now()
    slot = _most_recent_slot(now)
    state = _load_state()
    stale = state is None or datetime.fromisoformat(state["last_slot"]) < slot
    # A slot finalized by a blind run is not satisfied (TS-01): re-run it on
    # daemon start, under the same bounded retry count run_job applies. A
    # pre-existing state file has no "retries" key, which reads as 0; a
    # PRESENT but corrupt retries/exit_code has already been degraded to a
    # typed value (0 / None) by _load_state (DR-24), so the `<` and `==`
    # below never see anything but an int or None.
    blind = (
        state is not None
        and state.get("exit_code") == EXIT_NO_TRADEABLE_SHARDS
        and state.get("retries", 0) < SCHEDULER_BLIND_MAX_RETRIES
    )
    if stale or blind:
        logging.warning(
            "Most recent run slot (%s on the host's clock; runs are at %s) has no %s "
            "run — running catch-up now",
            slot.isoformat(), SCHEDULER_RUN_TIMES.times_label(),
            "recorded" if stale else "successful (blind-run)",
        )
        run_job(retries=0 if stale else state.get("retries", 0) + 1)


def _startup_catch_up() -> None:
    """
    Run _maybe_catch_up() and never let it take the daemon down.

    main() calls this BEFORE registering the run jobs, so an exception
    raised here used to propagate straight out of main() with nothing
    scheduled — a corrupt-but-present `retries` in scheduler_state.json (a
    raw string or null) reaching _maybe_catch_up's `retries < ...`
    comparison was one way to trigger it (DR-24; _load_state now degrades
    that case, and types `exit_code` alongside it, before they get here,
    but this guard is the backstop for whatever else _maybe_catch_up might
    still raise).
    Extracted to its own function so the guard is testable without entering
    main()'s infinite poll loop.

    Returns:
        None
    """
    try:
        _maybe_catch_up()
    except Exception:
        logging.exception("Startup catch-up check raised — daemon continues")


def _check_live_defaults() -> None:
    """
    Log an ERROR at daemon start when the scheduled runs would refuse to start.

    A live run exits 2 when no live defaults are saved or the saved file is
    refused. The daemon says so as soon as it starts, naming the fix, rather
    than only after the next scheduled run has spent its slot: for a missing
    file, the checkout (PROJECT_ROOT) the defaults must be saved in, since the
    runs it spawns read the file there; for one an older version saved
    (LiveDefaultsOutdated), that it must be saved again through the defaults
    server, whose pages offer its settings in their new form; for any other
    refused one, that it must be fixed, or deleted before new defaults are
    saved (the defaults server will not save over it). Nothing is logged when
    a usable file is saved.

    Returns:
        None

    Raises:
        Nothing: any exception from the read is logged as an ERROR instead,
            so the daemon still registers its run jobs.
    """
    try:
        # The one read the daemon makes of the saved defaults; main.py reads
        # them again, itself, on every run it spawns
        saved = read_saved_live_defaults()
    except LiveDefaultsOutdated as exc:
        # Saved by an older version: one save through the defaults server
        # replaces it, so it is neither to be fixed by hand nor deleted
        logging.error("The saved live defaults are out of date (%s): every scheduled run "
                      "will exit 2 until they are saved again through the defaults server "
                      "(./start_dashboard.sh, run in %s); do not delete the file", exc,
                      PROJECT_ROOT)
        return
    except Exception as exc:     # a refused file, or anything unexpected
        # The defaults server will not save over a refused file, so it must be
        # fixed, or deleted before new defaults are saved
        logging.error("The saved live defaults are refused (%s): every scheduled run "
                      "will exit 2 until the file is fixed, or deleted and saved again", exc)
        return
    if saved is None:
        # Name the checkout: the runs this daemon spawns read the file there, so
        # defaults saved from any other checkout (a worktree, say) do not count
        logging.error("No live defaults are saved in %s (the checkout this daemon runs): "
                      "every scheduled run will exit 2 until they are — save them from "
                      "that directory with ./start_dashboard.sh --seed (or python3 -m "
                      "kalshi_betting.defaults_server --seed), or through that server and "
                      "the backtest dashboard's Save as live defaults… button", PROJECT_ROOT)


def _setup_logging(log_path: pathlib.Path) -> None:
    """
    Configure root logging with both a console handler and a rotating file
    handler.

    Same SHAPE as kalshi_betting.main._setup_logging (console + rotating file,
    delay=True) but pointed at a DIFFERENT file (_SCHEDULER_LOG_PATH): the
    daemon must never share a handle on a file its own subprocess rotates —
    after main.py's RotatingFileHandler renames kalshi_arb.log, this
    long-lived process would keep writing into the renamed backup. It is still
    not imported from main.py, for the process-isolation reason stated there:
    scheduler.py spawns main as an isolated subprocess so a crash in one run
    can't take down the daemon, and importing main directly would defeat that
    isolation and create a needless coupling.

    The file handler ROTATES for the same reason main.py's does: this daemon
    runs forever, and a plain FileHandler would grow the log without bound.
    5MB across 3 backups is logging infrastructure sized for this daemon's own
    verbosity, not a strategy constant, so it stays inline here rather than
    moving to config.py — exactly as main.py's does.

    Args:
        log_path (pathlib.Path): Path to the log file to append to.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True,
            ),
        ],
    )


def main() -> None:
    """
    Entry point for the scheduler daemon.

    Sets up logging, checks the host's clock (_host_clock_realises_run), logs
    an ERROR when no usable live defaults are saved (_check_live_defaults:
    every run would exit 2 until they are), runs the startup catch-up
    (_startup_catch_up), registers run_job() at each of SCHEDULER_RUN_TIMES'
    times through _guarded_job(), then polls every 60 seconds. The clock
    check, the live-defaults check and the catch-up never raise, so the run
    jobs are always registered. With no scheduler_state.json yet, or none
    recorded for the latest slot, the catch-up runs the bot at once.
    """
    # The daemon logs to its OWN file — see _SCHEDULER_LOG_PATH for why it
    # must not share kalshi_arb.log with the subprocess that rotates it.
    _setup_logging(_SCHEDULER_LOG_PATH)

    # Check the host's clock before any catch-up run, and only after
    # _setup_logging: an earlier log call would install Python's default
    # handler and make _setup_logging's basicConfig a no-op, so the log file
    # would never be written. It never raises, so what follows always runs.
    _host_clock_realises_run()

    # Say now, not after the next scheduled run, when every run would exit 2
    # for want of saved live defaults (never raises, so the run jobs are still
    # registered below)
    _check_live_defaults()

    # Catch up on a missed run before registering future ones.
    # _startup_catch_up logs any raise (e.g. from a corrupt
    # scheduler_state.json), so the run jobs below are always registered.
    _startup_catch_up()

    # Through _guarded_job: `schedule` reschedules a job only after it
    # returns, so a raise escaping run_job() would leave it overdue and re-run
    # it on every poll tick.
    for job in _daily_jobs():
        job.do(_guarded_job, run_job)

    python_path  = sys.executable
    project_path = str(PROJECT_ROOT)

    logging.info(
        "Scheduler started. Runs %s (config.SCHEDULER_RUN_TIMES: %s).",
        _host_clock_label(), SCHEDULER_RUN_TIMES.label(),
    )
    logging.info(
        "To run instead as a cron job, add these lines with `crontab -e` (cron also "
        "fires on the host's clock):\n%s",
        "\n".join(_cron_lines(project_path, python_path)),
    )
    logging.info("Waiting for the next run (%s)...", _host_clock_label())

    # Poll for pending scheduled jobs every 60 seconds.
    # The schedule library tracks the next fire time internally.
    while True:
        try:
            schedule.run_pending()
        except Exception:
            # Since DR-59 every job body is caught by _guarded_job, so what
            # reaches here is a failure in the scheduling machinery itself
            # (run_pending's own bookkeeping, a reschedule). That must not
            # kill the daemon — log it and keep waiting. Note this handler
            # keeps the daemon alive but does NOT reschedule anything; that
            # is exactly why the guard exists at the registration sites.
            logging.exception("Scheduler tick raised — daemon continues")
        time.sleep(60)


if __name__ == "__main__":
    main()
