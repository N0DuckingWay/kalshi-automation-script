"""
File: scheduler.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Provides a long-running daemon that automatically invokes the production
    arbitrage bot every Monday at 09:00 local time. Uses the `schedule` library
    to register the job and a polling loop with 60-second sleep intervals to
    check for pending jobs. Every job is registered through _guarded_job(), so
    an exception inside a job can never freeze that job's next_run and turn the
    weekly fire into a once-per-poll-tick fire (DR-59 — see that function).
    The daemon logs to its own kalshi_scheduler.log
    (and the console), separate from the kalshi_arb.log its subprocess writes
    and rotates. Also prints the equivalent cron job command to the
    log for users who prefer cron over a Python daemon. Persists the most
    recently satisfied run slot to scheduler_state.json so a daemon restart
    can detect and catch up on a Monday 09:00 slot that was missed while the
    process was offline (BS-17), rather than silently waiting up to a week
    for the next scheduled fire.

Dependencies:
    Imports PROJECT_ROOT, SCHEDULER_JOB_TIMEOUT_SECONDS,
    SCHEDULER_BLIND_RETRY_SECONDS / SCHEDULER_BLIND_MAX_RETRIES, and the
    EXIT_OK / EXIT_SKIPPED_LOW_BALANCE / EXIT_TRADES_NEED_ATTENTION /
    EXIT_NO_TRADEABLE_SHARDS exit-code constants from config.py — the EXIT_*
    imports are what let run_job() map the subprocess's exit code to a
    distinct log level/message (BS-14) rather than treating every nonzero code
    identically. Spawns kalshi_betting.main as a
    subprocess (via sys.executable) rather than importing it directly, to
    isolate run-time errors and capture stdout/stderr separately. Entry point for
    `python3 -m kalshi_betting.scheduler`.

Notes:
    The scheduler runs the bot in production mode (--mode prod). For the bot to
    trade, valid prod credentials must be present in secrets.json and the PEM key
    file. If you want the scheduler to run at a different time or interval, edit
    the `schedule.every().monday.at("09:00")` call in main().

    Log file (C2): the daemon writes to PROJECT_ROOT / "kalshi_scheduler.log"
    (rotating, 5MB x 3), NOT to kalshi_arb.log. kalshi_arb.log belongs to the
    spawned main.py subprocess, which rotates it; a long-lived daemon holding
    an open handle on that file would keep appending to the renamed backup
    after every rotation, so its own lines would vanish from the live log.
    Operator-facing messages that point at kalshi_arb.log / trade_log.xlsx
    stay correct — that is still where the bot's own lines and trade records
    go.

    The Monday 09:00 schedule fires in HOST-LOCAL time (the `schedule` library
    reads the system clock, no timezone conversion) — this is deliberate,
    existing behavior, not a bug to "fix" by adding UTC conversion.

    CPython quirk (BS-16, reproduced on this host): subprocess.TimeoutExpired's
    .stdout/.stderr are raw BYTES even when subprocess.run() was called with
    text=True — text=True only governs decoding of the CompletedProcess
    returned on a normal exit, not the partial output attached to the
    exception when the timeout fires. Logging those bytes directly renders as
    a bytes repr (b'...') with literal \\n escapes instead of real newlines.
    _decode() below normalizes both streams before logging, and the timeout
    handler now logs stderr too (previously dropped entirely, even though it
    is where a hung run's traceback would show up).

    A bare OSError from subprocess.run() (BS-31) — e.g. ENOENT, a fork
    failure, a bad cwd — is now caught with a specific "Failed to spawn"
    error log instead of escaping run_job() and being swallowed by main()'s
    generic "Scheduler tick raised" handler with no run-specific context.

    DR-59 registration guard: main()'s generic "Scheduler tick raised" handler
    keeps the daemon alive but does NOT reschedule the job that raised —
    schedule.Job.run() assigns last_run and calls _schedule_next_run() only
    AFTER job_func() returns, so an escaping exception leaves next_run in the
    past and the daemon re-enters the SAME job on the very next 60 s poll tick.
    Every job is therefore registered through _guarded_job(), which catches at
    the boundary `schedule` calls in — covering every escape path out of a job,
    not just the ones an individual helper knows about. _save_state() itself is
    deliberately unchanged: a failed CLAIM write must still prevent the spawn,
    since a real-money run with no recorded slot would break BS-17's invariant
    that a claimed slot always has a record.

    BS-17 catch-up: run_job() claims its Monday-09:00 slot in
    scheduler_state.json BEFORE spawning the subprocess, and finalizes that
    record (finished_at, exit_code) on every exit path — success, nonzero
    exit, timeout, or OSError. Claiming at the start (not just recording on
    success) means a crashing/hanging run still leaves a recorded attempt for
    its slot, so a daemon restart won't loop re-running a slot whose
    subprocess merely failed; only a slot with NO recorded attempt at all
    triggers catch-up. On timeout or OSError, exit_code is left as None in
    the finalized record — there was no subprocess exit code to record — with
    finished_at still set to distinguish "attempted and ended" from "claimed,
    still running" (which only appears if the process was killed mid-run,
    e.g. host reboot). main() runs the startup catch-up check
    (_startup_catch_up(), a logging-guarded wrapper around _maybe_catch_up())
    once, after logging is configured and before registering the weekly
    schedule: the very first daemon start after this feature was added will
    therefore always trigger an immediate prod run, since
    scheduler_state.json does not yet exist. _startup_catch_up() exists
    because a corrupt-but-present `retries` in that file used to raise
    TypeError out of _maybe_catch_up() (its `retries < ...` comparison)
    before the weekly job was ever registered, exiting the daemon at
    startup (DR-24) — _load_state now degrades a non-int `retries` to 0
    with a WARNING before it gets there, and types `exit_code` the same
    way for consistency, though a non-int `exit_code` only ever compared
    unequal and never raised. _startup_catch_up() is the backstop for
    whatever else _maybe_catch_up() might still raise.

    TS-01/VI-02 blind-run retry: EXIT_NO_TRADEABLE_SHARDS (30) means the run
    scanned NOTHING. main.py returns it for either of two causes (see
    main._blind_run_reason): every advertised exchange shard was
    trading-inactive, so ingest dropped every market — an exchange-wide
    maintenance window overlapping the 09:00 fire, observed live 2026-09-03 —
    or the market ingest came back empty for a cause /exchange/status could
    not name, which is the case scanner.fetch_shard_statuses' fail-soft None
    leaves undiagnosable. This daemon cannot tell the two apart from the exit
    code alone; kalshi_arb.log carries the WARNING naming which one fired. The
    bot trades only on the weekly fire, so that used to cost the entire week
    while both logs said the run succeeded. run_job() now registers a one-shot
    retry (SCHEDULER_BLIND_RETRY_SECONDS out, at most
    SCHEDULER_BLIND_MAX_RETRIES per slot) instead, and _maybe_catch_up()
    re-runs a slot whose recorded attempt exited 30. The count is carried on
    run_job's `retries` argument and persisted as the state file's optional
    "retries" key — the cap can only be enforced there, because a retry that
    exits 30 again schedules its own successor. A pre-existing state file has
    no such key and reads as 0; a PRESENT but non-int `retries` or `exit_code`
    (a hand-edited file) is degraded by _load_state to that same 0 / unknown
    reading with a WARNING rather than reaching this comparison unvalidated
    (DR-24).
    An hourly cadence bounded at four attempts covers a typical maintenance
    window while keeping the scan near its intended Monday-morning slot; a
    longer interval would trade on stale morning pricing.
"""
import json
import logging
import logging.handlers
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import schedule

from .config import (
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    PROJECT_ROOT,
    SCHEDULER_BLIND_MAX_RETRIES,
    SCHEDULER_BLIND_RETRY_SECONDS,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
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


def _state_file_path() -> Path:
    """
    Return the path to the persisted scheduler state file.

    A function rather than a module-level constant so tests can monkeypatch
    `scheduler.PROJECT_ROOT` to a tmp_path and have state reads/writes follow
    it — a constant computed once at import time would freeze the path
    before any test patch could take effect.

    Returns:
        Path: PROJECT_ROOT / "scheduler_state.json".
    """
    return PROJECT_ROOT / "scheduler_state.json"


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


def _most_recent_slot(now: datetime) -> datetime:
    """
    Compute the latest Monday-09:00-local slot at or before `now`.

    Mirrors the local-time semantics of `schedule.every().monday.at("09:00")`,
    which this daemon uses to register the weekly job. Used by the BS-17
    catch-up check to determine which slot should already have a recorded run.

    Args:
        now (datetime): Naive local datetime to evaluate against.

    Returns:
        datetime: The Monday 09:00 datetime (naive, local, seconds/microseconds
            zeroed) of the most recent slot that should already have fired.
            If `now` is itself a Monday before 09:00, this is the PREVIOUS
            week's Monday 09:00 — this week's slot has not fired yet.
    """
    candidate = now.replace(hour=9, minute=0, second=0, microsecond=0)
    # Monday == 0 .. Sunday == 6; walk back to this calendar week's Monday.
    candidate -= timedelta(days=candidate.weekday())
    if candidate > now:
        # now is Monday before 09:00 — this week's slot hasn't fired yet.
        candidate -= timedelta(days=7)
    return candidate


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
    `TypeError` out of the daemon at startup before the weekly job was ever
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
    # weekly job was ever registered, killing the daemon at startup. A bad
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
        last_slot (datetime): The Monday-09:00 slot this run is satisfying.
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
    very next 60 s poll tick, turning a weekly production trading run into a
    once-a-minute one for as long as the fault persists. The same mechanism
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
        # recurring weekly job is rescheduled for its next Monday, while a
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


def run_job(retries: int = 0) -> None:
    """
    Execute a single production arbitrage bot run as an isolated subprocess.

    Spawns `python -m kalshi_betting.main --mode prod` using the same Python
    interpreter that is running this scheduler. Running in a subprocess isolates
    any unhandled exceptions in the bot from the scheduler process — a crash in
    one run does not prevent future runs from being triggered. A hung run is
    killed after SCHEDULER_JOB_TIMEOUT_SECONDS so it cannot block the daemon
    past the next scheduled fire.

    Claims its Monday-09:00 slot in scheduler_state.json before spawning the
    subprocess, and finalizes that record on every exit path — success,
    nonzero exit, timeout, or OSError (BS-17). See this module's Notes for
    why claiming happens up front.

    Logs the subprocess's stdout unconditionally (whether the run succeeded or
    failed). Its stderr is logged only on the catch-all failure branch (an
    exit code that is none of EXIT_OK / EXIT_SKIPPED_LOW_BALANCE /
    EXIT_TRADES_NEED_ATTENTION) and on TimeoutExpired; the
    EXIT_TRADES_NEED_ATTENTION and EXIT_SKIPPED_LOW_BALANCE branches log their
    message without stderr. All of that re-logged output lands in THIS
    daemon's log (kalshi_scheduler.log), while the bot's own log lines go to
    kalshi_arb.log through the subprocess's own handler. The exit code is
    mapped to a distinct log level/message per the EXIT_* contract in
    config.py (BS-14): a low-balance skip and a run with trades needing manual
    review are no longer indistinguishable from a clean run in this log —
    previously the only signal was a WARNING inside kalshi_arb.log that this
    scheduler process never reads.

    A subprocess.TimeoutExpired's stdout/stderr are decoded before logging
    (BS-16 — see _decode()), and both streams are logged (stderr, the hung
    run's traceback, was previously dropped entirely). A bare OSError from
    subprocess.run() itself (BS-31 — e.g. the interpreter can't be spawned)
    is logged with a specific message and does not escape this function.

    EXIT_NO_TRADEABLE_SHARDS (TS-01, VI-02) is the one code that does NOT
    satisfy the weekly slot: it means nothing was scanned at all — an
    exchange-wide halt, or an ingest that came back empty for a cause
    /exchange/status could not name — and the bot trades only on the weekly
    fire, so letting it stand would cost the week. Which of the two fired is
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
            this Monday slot. 0 for the ordinary weekly fire and for a
            catch-up of a never-attempted slot; greater than 0 only when
            _blind_retry or _maybe_catch_up is re-running a slot that a blind
            run left unsatisfied. Persisted into scheduler_state.json so the
            count survives a daemon restart.

    Returns:
        None
    """
    logging.info("Scheduler: starting weekly arbitrage scan.")
    started_at = datetime.now()
    slot = _most_recent_slot(started_at)
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
    elif result.returncode == EXIT_NO_TRADEABLE_SHARDS:
        # Not a satisfied slot: nothing was scanned (TS-01, VI-02). The exit
        # code does not say WHICH cause fired, so neither does this message —
        # the subprocess logs that sentence into kalshi_arb.log.
        if retries < SCHEDULER_BLIND_MAX_RETRIES:
            logging.warning(
                "Job scanned nothing (exit %d): every advertised exchange shard was "
                "trading-inactive, or the market ingest came back empty — "
                "kalshi_arb.log names which. The weekly slot is NOT satisfied — "
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
    Run an immediate catch-up job if the most recent Monday-09:00 slot has
    no recorded run (BS-17), or was only "satisfied" by a blind run (TS-01).

    Guards against a missed run when the daemon was offline (not started
    yet, crashed, host down, mid-deploy) across a scheduled fire time —
    `schedule.run_pending()` only fires while this process is polling, so a
    slot that comes and goes while the daemon is down is otherwise silently
    skipped until the following Monday, up to a full week away for a bot
    whose edge is time-sensitive. Extracted from main() so it can be tested
    without entering the infinite poll loop.

    The test was purely temporal (`last_slot < slot`) and never read the exit
    code, so a slot whose only attempt exited EXIT_NO_TRADEABLE_SHARDS —
    nothing scanned at all — counted as satisfied (TS-01). It is now also
    re-run, carrying `retries + 1` so the daemon-restart path shares
    run_job's SCHEDULER_BLIND_MAX_RETRIES cap rather than looping forever
    through an outage. A slot whose attempt merely FAILED is still not
    retried — that is BS-17's deliberate behaviour and is unchanged.

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
            "Most recent Monday 09:00 slot (%s) has no %s run — running catch-up now",
            slot.isoformat(), "recorded" if stale else "successful (blind-run)",
        )
        run_job(retries=0 if stale else state.get("retries", 0) + 1)


def _startup_catch_up() -> None:
    """
    Run _maybe_catch_up() and never let it take the daemon down.

    main() calls this BEFORE registering the weekly job, so an exception
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
    Entry point for the weekly scheduler daemon.

    Configures logging, runs the BS-17 startup catch-up check through
    _startup_catch_up() (a guard around _maybe_catch_up() that logs and
    continues instead of propagating — DR-24), registers run_job() to fire
    every Monday at 09:00, then enters an infinite polling loop checking for
    pending jobs every 60 seconds.

    The weekly job is registered through _guarded_job() (DR-59), so an
    exception escaping run_job() cannot leave next_run in the past and make
    the poll loop below re-enter a production trading run on every 60-second
    tick. The loop's own "Scheduler tick raised" handler keeps the daemon
    alive but does NOT reschedule the job that raised — only the wrapper
    returning normally does that.

    Note: the catch-up check means the very first daemon start after BS-17
    was added will always trigger an immediate prod run, since
    scheduler_state.json does not yet exist on that first start.
    """
    # The daemon logs to its OWN file — see _SCHEDULER_LOG_PATH for why it
    # must not share kalshi_arb.log with the subprocess that rotates it.
    _setup_logging(_SCHEDULER_LOG_PATH)

    # BS-17: catch up on a missed run before registering future ones, so a
    # daemon that was offline across a scheduled Monday 09:00 doesn't wait
    # up to a week for the next fire. Wrapped in _startup_catch_up so a
    # raise here (e.g. a corrupt optional key in scheduler_state.json,
    # DR-24) can't exit the process before the weekly job is registered.
    _startup_catch_up()

    # Register run_job() to fire every Monday at 09:00 local time, through
    # _guarded_job so an exception escaping the job can never freeze its
    # next_run in the past and turn the weekly fire into a once-per-poll-tick
    # fire (DR-59). schedule.Job.run() reschedules only AFTER job_func()
    # RETURNS, and the poll loop below swallows the raise and keeps polling.
    schedule.every().monday.at("09:00").do(_guarded_job, run_job)

    python_path  = sys.executable
    project_path = str(PROJECT_ROOT)

    logging.info("Scheduler started. Runs every Monday at 09:00.")
    logging.info(
        "To run instead as a cron job, add this line with `crontab -e`: "
        "0 9 * * 1 cd '%s' && %s -m kalshi_betting.main --mode prod >> /tmp/kalshi_arb.log 2>&1",
        project_path, python_path,
    )
    logging.info("Waiting for next Monday 09:00...")

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
