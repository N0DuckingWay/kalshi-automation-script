"""
File: scheduler.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Provides a long-running daemon that automatically invokes the production
    arbitrage bot every Monday at 09:00 local time. Uses the `schedule` library
    to register the job and a polling loop with 60-second sleep intervals to
    check for pending jobs. Also prints the equivalent cron job command to the
    log for users who prefer cron over a Python daemon.

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
"""
import logging
import subprocess
import sys
import time

import schedule

from .config import PROJECT_ROOT


def run_job() -> None:
    """
    Execute a single production arbitrage bot run as an isolated subprocess.

    Spawns `python -m kalshi_betting.main --mode prod` using the same Python
    interpreter that is running this scheduler. Running in a subprocess isolates
    any unhandled exceptions in the bot from the scheduler process — a crash in
    one run does not prevent future runs from being triggered.

    Logs stdout on success and stderr + exit code on failure so every run is
    traceable in kalshi_arb.log.
    """
    logging.info("Scheduler: starting weekly arbitrage scan.")
    # Use sys.executable to ensure the subprocess uses the same Python environment
    # (venv, conda, etc.) as the scheduler itself
    result = subprocess.run(
        [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod"],
        capture_output=True,
        text=True,
        # Run from the project root so relative paths in main.py resolve correctly
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    if result.stdout:
        logging.info("stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logging.error("Job failed (exit %d):\n%s", result.returncode, result.stderr)
    else:
        logging.info("Job completed successfully.")


def main() -> None:
    """
    Entry point for the weekly scheduler daemon.

    Configures the `schedule` library to call run_job() every Monday at 09:00,
    prints cron job instructions as an alternative, then enters an infinite
    polling loop checking for pending jobs every 60 seconds.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "kalshi_arb.log"),
        ],
    )

    # Register run_job() to fire every Monday at 09:00 local time
    schedule.every().monday.at("09:00").do(run_job)

    python_path  = sys.executable
    project_path = str(__import__("pathlib").Path(__file__).parent.parent)

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
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
