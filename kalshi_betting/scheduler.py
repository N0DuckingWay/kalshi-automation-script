"""Weekly scheduler: runs the arbitrage bot every Monday at 09:00."""
import logging
import subprocess
import sys
import time

import schedule


def run_job() -> None:
    """
    Execute a single production arbitrage bot run as a subprocess.

    Invokes `python -m kalshi_betting.main --mode prod` using the current Python
    interpreter, with the project root as the working directory. Logs stdout on
    success and stderr on non-zero exit codes.
    """
    logging.info("Scheduler: starting weekly arbitrage scan.")
    result = subprocess.run(
        [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod"],
        capture_output=True,
        text=True,
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
    )

    schedule.every().monday.at("09:00").do(run_job)

    python_path = sys.executable
    project_path = str(__import__("pathlib").Path(__file__).parent.parent)

    print("Scheduler started. Runs every Monday at 09:00.")
    print()
    print("To run instead as a cron job, add this line with `crontab -e`:")
    print(
        f"  0 9 * * 1 cd '{project_path}' && {python_path} -m kalshi_betting.main --mode prod"
        f" >> /tmp/kalshi_arb.log 2>&1"
    )
    print()
    print("Waiting for next Monday 09:00...")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
