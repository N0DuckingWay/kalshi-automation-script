"""
File: scheduler.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Provides a long-running daemon that automatically invokes the production
    arbitrage bot every Monday at 09:00 local time. Uses the `schedule` library
    to register the job and a polling loop with 60-second sleep intervals to
    check for pending jobs. Also prints the equivalent cron job command to the
    log for users who prefer cron over a Python daemon. Persists the most
    recently satisfied run slot to scheduler_state.json so a daemon restart
    can detect and catch up on a Monday 09:00 slot that was missed while the
    process was offline (BS-17), rather than silently waiting up to a week
    for the next scheduled fire.

Dependencies:
    Imports PROJECT_ROOT from config.py. Spawns kalshi_betting.main as a subprocess
    (via sys.executable) rather than importing it directly, to isolate run-time
    errors and capture stdout/stderr separately. Entry point for
    `python3 -m kalshi_betting.scheduler`.

Notes:
    The scheduler runs the bot in production mode (--mode prod). For the bot to
    trade, valid prod credentials must be present in secrets.json and the PEM key
    file. If you want the scheduler to run at a different time or interval, edit
    the `schedule.every().monday.at("09:00")` call in main().

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
    e.g. host reboot). main() calls the startup catch-up check
    (_maybe_catch_up()) once, after logging is configured and before
    registering the weekly schedule: the very first daemon start after this
    feature was added will therefore always trigger an immediate prod run,
    since scheduler_state.json does not yet exist.
"""
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import schedule

from .config import (
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    PROJECT_ROOT,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
)

# Schema version for scheduler_state.json — bump if the record shape changes
# so a future reader can distinguish old files instead of guessing.
_STATE_SCHEMA_VERSION = 1


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

    Returns:
        dict | None: The parsed state dict (guaranteed to have a parseable
            `last_slot`), or None.
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
    return state


def _save_state(
    *,
    last_slot: datetime,
    started_at: datetime,
    finished_at: datetime | None,
    exit_code: int | None,
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
    """
    state = {
        "schema": _STATE_SCHEMA_VERSION,
        "last_slot": last_slot.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at is not None else None,
        "exit_code": exit_code,
    }
    path = _state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def run_job() -> None:
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

    Logs stdout on success and stderr + exit code on failure so every run is
    traceable in kalshi_arb.log. The subprocess's exit code is mapped to a
    distinct log level/message per the EXIT_* contract in config.py (BS-14):
    a low-balance skip and a run with trades needing manual review are no
    longer indistinguishable from a clean run in this log — previously the
    only signal was a WARNING inside kalshi_arb.log that this scheduler
    process never reads.

    A subprocess.TimeoutExpired's stdout/stderr are decoded before logging
    (BS-16 — see _decode()), and both streams are logged (stderr, the hung
    run's traceback, was previously dropped entirely). A bare OSError from
    subprocess.run() itself (BS-31 — e.g. the interpreter can't be spawned)
    is logged with a specific message and does not escape this function.
    """
    logging.info("Scheduler: starting weekly arbitrage scan.")
    started_at = datetime.now()
    slot = _most_recent_slot(started_at)
    # Claim the slot before spawning, so a crash/timeout/OSError mid-run
    # still leaves a recorded attempt for this slot — otherwise the startup
    # catch-up check would see no record at all and re-run it on every
    # restart until one attempt happens to finish cleanly.
    _save_state(last_slot=slot, started_at=started_at, finished_at=None, exit_code=None)

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
            finished_at=datetime.now(), exit_code=None,
        )
        return
    except OSError as exc:
        logging.error("Failed to spawn bot subprocess: %s", exc)
        _save_state(
            last_slot=slot, started_at=started_at,
            finished_at=datetime.now(), exit_code=None,
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
    else:
        logging.error("Job failed (exit %d):\n%s", result.returncode, result.stderr)

    _save_state(
        last_slot=slot, started_at=started_at,
        finished_at=datetime.now(), exit_code=result.returncode,
    )


def _maybe_catch_up(now: datetime | None = None) -> None:
    """
    Run an immediate catch-up job if the most recent Monday-09:00 slot has
    no recorded run (BS-17).

    Guards against a missed run when the daemon was offline (not started
    yet, crashed, host down, mid-deploy) across a scheduled fire time —
    `schedule.run_pending()` only fires while this process is polling, so a
    slot that comes and goes while the daemon is down is otherwise silently
    skipped until the following Monday, up to a full week away for a bot
    whose edge is time-sensitive. Extracted from main() so it can be tested
    without entering the infinite poll loop.

    Args:
        now (datetime | None): Override for the current local time, for
            testing. Defaults to datetime.now().
    """
    now = now if now is not None else datetime.now()
    slot = _most_recent_slot(now)
    state = _load_state()
    if state is None or datetime.fromisoformat(state["last_slot"]) < slot:
        logging.warning(
            "Most recent Monday 09:00 slot (%s) has no recorded run — "
            "running catch-up now",
            slot.isoformat(),
        )
        run_job()


def main() -> None:
    """
    Entry point for the weekly scheduler daemon.

    Configures logging, runs the BS-17 startup catch-up check (see
    _maybe_catch_up()), registers run_job() to fire every Monday at 09:00,
    then enters an infinite polling loop checking for pending jobs every 60
    seconds.

    Note: the catch-up check means the very first daemon start after BS-17
    was added will always trigger an immediate prod run, since
    scheduler_state.json does not yet exist on that first start.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "kalshi_arb.log"),
        ],
    )

    # BS-17: catch up on a missed run before registering future ones, so a
    # daemon that was offline across a scheduled Monday 09:00 doesn't wait
    # up to a week for the next fire.
    _maybe_catch_up()

    # Register run_job() to fire every Monday at 09:00 local time
    schedule.every().monday.at("09:00").do(run_job)

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
            # The subprocess isolates bot crashes, but a host-level failure
            # (e.g. the spawn itself raising) must not kill the daemon —
            # log it and keep waiting for the next scheduled run.
            logging.exception("Scheduler tick raised — daemon continues")
        time.sleep(60)


if __name__ == "__main__":
    main()
