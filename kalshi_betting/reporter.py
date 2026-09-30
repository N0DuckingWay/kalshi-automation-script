"""
File: reporter.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Writes the bot's Excel files. A production run adds a banner row and one
    row per trade to the shared trade_log.xlsx, which keeps every run's history.
    A dev run writes a new timestamped file with two sheets: the simulated
    trades and every candidate pair found.

    It also writes the run result of a production run started with
    `main.py --result-file`: a RunReport that main.py fills in as the run goes
    (its outcome, its cash before and after trading, the portfolio value it
    sized on, whether it began sending orders, one record per pair and its
    WARNING-or-worse log lines), saved as one JSON file when the run ends.

Dependencies:
    Imports display_title, leg_sides (which side each leg buys, rendered
    into the Notes prefix) and leg_prices (the price each leg was sized at,
    for the run result) from scanner.py and TradeSpec from strategy.py.
    Imports PROJECT_ROOT, create_new_output and the run result's constants
    (LIVE_RUN_RESULT_FORMAT, RUN_REPORT_MAX_WARNINGS,
    RUN_REPORT_LINE_MAX_CHARS) from config.py. Exports the TradeResult
    dataclass (consumed by trader.py), the two public write functions, and
    the run result's RunReport, TradeRecord, LegRecord, RunReportHandler,
    trade_record, report_trades and write_run_report (all consumed by
    main.py).

Notes:
    The TradeResult dataclass is defined here (not in trader.py) because reporter.py
    is the authoritative consumer of trade outcomes — trader.py only needs to
    construct and return these objects.

    append_to_prod_log() coordinates concurrent writers (e.g. the scheduler and a
    manual invocation racing) via a sidecar advisory lock (fcntl.flock) around the
    load -> append -> save sequence, and saves atomically (tmp file + os.replace)
    so a crash mid-save can never truncate the accumulated trade history. If the
    lock cannot be acquired within _LOCK_TIMEOUT_SECONDS, this run's rows are never
    silently dropped — they're written to a standalone timestamped fallback file
    instead of touching the shared log. See BS-18 in CLAUDE.md's bug-sweep history.

    Header rows are written only when a workbook is CREATED (_append_locked on
    its first run, _write_fallback_log always, write_dev_simulation always), so
    the 2026-09 strategy change — which renamed the x/y headers to the
    side-neutral "x — A leg" / "y — B leg" and the profit column to
    "Profit if won ($)" — shows up in new workbooks and fallback files, while
    an existing shared trade_log.xlsx keeps
    its old header row untouched. Column COUNT and order are unchanged (18), so
    old and new rows line up; the per-row Notes prefix
    ("[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=0.xxxx]] ") is what tells a
    reader which side each count bought and, for a time-series row, the traded
    NO-leg price — the retained "nA (NO ask)" column is reporting-only there.

    append_to_prod_log's keyword-only run_note goes on the run's separator
    banner, never in a column; main._run_prod passes the run's live toggles
    (config.describe_live_settings), with "(default: X)" after each toggle a
    flag moved away from the saved live defaults, ending " | defaults:
    <origin>" (which saved file, when and from what it was saved).

    Neither filling nor writing a run result may stop a run or change its exit
    code: report_trades never raises (a pair it cannot describe is logged and
    kept as a record of what could be read), and write_run_report catches
    every error, since main() calls it in a finally. The result is replaced
    whole (a staging file renamed over it), and is strict JSON.
    RunReportHandler keeps every ERROR and CRITICAL line whole, since those
    are the lines a person has to act on; only WARNING lines are cut and
    capped.
"""
import contextlib
import fcntl
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .config import (
    LIVE_RUN_RESULT_FORMAT,
    PROJECT_ROOT,
    RUN_REPORT_LINE_MAX_CHARS,
    RUN_REPORT_MAX_WARNINGS,
    create_new_output,
)
from .scanner import display_title, leg_prices, leg_sides
from .strategy import TradeSpec

PROD_LOG_PATH = PROJECT_ROOT / "trade_log.xlsx"

# Sidecar lock file coordinating concurrent writers to PROD_LOG_PATH (e.g. the
# weekly scheduler and a manual run racing) — see _acquire_lock().
_LOCK_PATH = PROD_LOG_PATH.with_name(PROD_LOG_PATH.name + ".lock")
# Bounded acquire loop: a non-blocking flock() attempt polled at this interval,
# up to this many total seconds, rather than a blocking flock() call — so a
# wedged lock holder (crashed mid-hold, or just slow) can never hang this run
# forever. Deliberately short relative to a full bot run.
_LOCK_TIMEOUT_SECONDS = 30
_LOCK_POLL_SECONDS = 0.5

# Column definitions shared by both sheets
_TRADE_COLUMNS = [
    ("Date",              14),
    ("Time",              10),
    ("Market A",          45),
    ("Ticker A",          18),
    ("Market B",          45),
    ("Ticker B",          18),
    ("A Deadline",        12),
    ("B Deadline",        12),
    ("pA (YES ask)",      13),
    ("pB (YES ask)",      13),
    ("nA (NO ask)",       13),
    # Side-neutral: x is the count on market A and y on market B whatever side
    # each bought (NO/YES for same-title, YES/NO for time-series) — the row's
    # Notes prefix names the sides. An existing workbook keeps its old headers.
    ("x — A leg",         12),
    ("y — B leg",         13),
    # Fee-INCLUSIVE, matching the cash the collateral planner actually
    # moves and the basis "Profit if won" is already net of (TS-12).
    # An existing workbook keeps its old header; every row's Notes
    # carries fees=$x.xx so a row under the old header is still readable.
    ("Total Cost incl. fees ($)", 22),
    ("Profit if won ($)", 14),
    ("Profit Ratio (%)",  16),
    ("Status",            12),
    ("Notes",             30),
]

_HEADER_FILL_PROD = PatternFill("solid", fgColor="1F4E79")   # dark blue for prod
_HEADER_FILL_DEV  = PatternFill("solid", fgColor="375623")   # dark green for dev
_HEADER_FONT      = Font(bold=True, color="FFFFFF", size=11)
_THIN_BORDER      = Border(
    bottom=Side(style="thin", color="BFBFBF"),
)


@dataclass
class TradeResult:
    """
    Outcome record for a single attempted or simulated trade.

    Attributes:
        spec (TradeSpec): The trade specification that was executed or simulated.
        status (str): What happened. "NO leg" and "YES leg" are the two
            orders in the order they are sent: the NO leg first (the one
            undone if the pair cannot be completed), then the YES leg (see
            trader._ordered_legs).
            "executed": both legs filled (confirmed by the order replies or
                by the change in the account's position).
            "simulated": dry run or dev mode; nothing was sent.
            "failed": the NO leg did not fill, or nothing was sent (a pair
                stopped because the V2 NO-leg mapping was disproven earlier
                in the run), so nothing is open.
            "rolled_back": the NO leg filled, the YES leg did not, and the NO
                position was fully closed again.
            "rollback_failed": that closing order filled only partly or not
                at all, so a NO position is left open for a person to handle.
            "manual_review": the outcome of a leg could not be tied to this
                order (the position read failed or moved by an unexplained
                amount, the one-time check found that a NO buy did not open
                a NO position, on this pair or on an earlier one, or the
                pair's worker raised), so no
                follow-up order was sent.
        error (Optional[str]): Error message when the leg(s) involved required
            explanation, worded as "NO leg …"/"YES leg …" — every
            non-"executed"/"simulated" status always sets this, and "executed"
            also sets it in the one case where the YES leg was ambiguous but
            its fill was confirmed by position delta (the error text says so).
            None only when nothing needed explaining. _result_to_row prefixes
            it with the pair type and sides in the Excel Notes cell.
    """
    spec: TradeSpec
    status: str            # "executed" | "failed" | "simulated" | "rolled_back" | "rollback_failed" | "manual_review"
    error: str | None = None



def _apply_header_row(ws, fill: PatternFill) -> None:
    """
    Write and style the column header row.

    Args:
        ws: The openpyxl worksheet to write the header into.
        fill (PatternFill): Background fill for the header row (prod vs dev
            use different colors — see _HEADER_FILL_PROD / _HEADER_FILL_DEV).

    Returns:
        None
    """
    for col_idx, (header, width) in enumerate(_TRADE_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font  = _HEADER_FONT
        cell.fill  = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"


def _result_to_row(result: TradeResult, run_ts: datetime) -> list:
    """
    Serialize a TradeResult to a flat list matching the _TRADE_COLUMNS column order.

    Extracts all fields needed for one Excel data row, formatting prices as rounded
    floats and datetimes as "YYYY-MM-DD" strings. Columns are in MARKET order
    (A then B): x is market A's count and y is market B's, whatever side each
    bought. The Notes cell carries a prefix naming the pair type, the side
    bought on each market (from scanner.leg_sides) and, for a time-series pair,
    the traded NO-leg price nB — "[time_series: YES A / NO B nB=0.4000] " or
    "[same_title: NO A / YES B] " — followed by result.error (if any). That
    prefix is what disambiguates rows in a workbook whose header row predates
    the side-neutral x/y headers, and it is the only place nB is recorded (the
    "nA (NO ask)" column is reporting-only for a time-series row).

    Args:
        result (TradeResult): The trade result to serialize.
        run_ts (datetime): Timestamp of the current bot run, used to populate the
            Date and Time columns.

    Returns:
        list: Ordered list of 18 values, one per column in _TRADE_COLUMNS.
    """
    spec = result.spec
    pair = spec.pair
    mA   = display_title(pair.market_a)
    mB   = display_title(pair.market_b)
    # Which side each market's leg bought — the only source of truth for sides
    side_a, side_b = leg_sides(pair.pair_type)
    # nB is a traded leg price only for time-series pairs; for same-title it
    # is reporting-only and would just be noise in the Notes cell
    nb_note = f" nB={pair.nB:.4f}" if pair.pair_type == "time_series" else ""
    # fees=... makes a row self-describing: _append_locked writes headers only
    # when CREATING the file, so a pre-existing shared trade_log.xlsx keeps the
    # old "Total Cost ($)" header above a column whose values are now
    # fee-inclusive. The suffix is what tells a reader which basis a row is on.
    fees = spec.total_cost_with_fees - spec.total_cost
    notes = (
        f"[{pair.pair_type}: {side_a.upper()} A / {side_b.upper()} B{nb_note}"
        f" fees=${fees:.2f}] "
        + (result.error or "")
    )

    def fmt_dt(dt) -> str:
        return dt.strftime("%Y-%m-%d") if dt else ""

    return [
        run_ts.strftime("%Y-%m-%d"),
        run_ts.strftime("%H:%M:%S"),
        mA,
        pair.market_a.ticker,
        mB,
        pair.market_b.ticker,
        fmt_dt(pair.market_a.close_time),
        fmt_dt(pair.market_b.close_time),
        round(pair.pA, 4),
        round(pair.pB, 4),
        round(pair.nA, 4),
        spec.x,
        spec.y,
        round(spec.total_cost_with_fees, 2),
        round(spec.min_payoff, 2),
        round(spec.profit_ratio, 4),
        result.status,
        notes,
    ]


def _apply_data_row_styles(ws, row_idx: int, status: str) -> None:
    """
    Apply background fill, border, and alignment styling to a single data row.

    Color-codes rows by trade status: green for "executed", blue for "simulated",
    red/orange for "failed", yellow for "rolled_back", strong red for
    "rollback_failed" and "manual_review", white for any unknown status.

    Args:
        ws: An openpyxl Worksheet object to apply styles to.
        row_idx (int): 1-based row index of the data row to style.
        status (str): Trade status string — "executed", "simulated", "failed",
            "rolled_back", "rollback_failed", or "manual_review".
    """
    status_colors = {
        "executed":        "E2EFDA",   # light green
        "simulated":       "EBF3FB",   # light blue
        "failed":          "FCE4D6",   # light red/orange
        "rolled_back":     "FFF2CC",   # light yellow — NO leg unwound, no net position
        "rollback_failed": "F4B7B4",   # strong red — orphaned position, manual review
        "manual_review":   "F4B7B4",   # strong red — fill state unknown, manual review
    }
    fill_color = status_colors.get(status, "FFFFFF")
    fill = PatternFill("solid", fgColor=fill_color)
    for col in range(1, len(_TRADE_COLUMNS) + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill   = fill
        cell.border = _THIN_BORDER
        cell.alignment = Alignment(vertical="center")


def _apply_number_formats(ws, row_idx: int) -> None:
    """
    Apply currency/percentage formats to a single data row's numeric columns.

    Args:
        ws: The openpyxl worksheet containing the row.
        row_idx (int): 1-indexed row number to format.

    Returns:
        None
    """
    # Columns: pA=9, pB=10, nA=11, TotalCost=14, ProfitIfWon=15, ProfitRatio=16
    # (indices unchanged by the 2026-09 header renames — still 18 columns)
    for col in (9, 10, 11):
        ws.cell(row=row_idx, column=col).number_format = "0.00%"
    for col in (14, 15):
        ws.cell(row=row_idx, column=col).number_format = '"$"#,##0.00'
    ws.cell(row=row_idx, column=16).number_format = "0.00%"


def _write_separator_row(
    ws, run_ts: datetime, balance_before: float, balance_after: float, n_results: int,
    *, run_note: str = "",
) -> None:
    """
    Add one grey banner row for this run: its time, cash before and after, and trade count.

    The shared trade log and the fallback file both use it; a non-empty run_note goes at the end.

    Args:
        ws: The worksheet to add the row to.
        run_ts (datetime): When this run happened.
        balance_before (float): Cash on all shards together before this run's trades, in dollars.
        balance_after (float): The same after this run's trades.
        n_results (int): How many trade results this run has.
        run_note (str): Keyword-only text for the end of the banner; empty adds nothing.
    """
    sep_row = ws.max_row + 1
    sep_cell = ws.cell(row=sep_row, column=1,
                       value=f"── Run: {run_ts.strftime('%Y-%m-%d %H:%M')}  |  "
                             f"Balance before: ${balance_before:.2f}  →  "
                             f"after: ${balance_after:.2f}  |  "
                             f"{n_results} trade(s)"
                             + (f"  |  {run_note}" if run_note else ""))
    sep_cell.font = Font(italic=True, color="595959", size=9)
    sep_cell.fill = PatternFill("solid", fgColor="F2F2F2")
    ws.merge_cells(
        start_row=sep_row, start_column=1,
        end_row=sep_row, end_column=len(_TRADE_COLUMNS)
    )


def _write_trade_rows(ws, results: list, run_ts: datetime) -> None:
    """
    Append one styled, color-coded data row per TradeResult.

    Shared by the normal prod-log append path and the lock-timeout fallback path.

    Args:
        ws: The openpyxl Worksheet to append to.
        results (list): List of TradeResult objects to write, in order.
        run_ts (datetime): Timestamp of this run, used for the Date/Time columns.
    """
    for result in results:
        row_idx = ws.max_row + 1
        row_data = _result_to_row(result, run_ts)
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
        _apply_data_row_styles(ws, row_idx, result.status)
        _apply_number_formats(ws, row_idx)


def _acquire_lock(lock_path: Path) -> IO | None:
    """
    Acquire an exclusive advisory lock on lock_path, polling up to _LOCK_TIMEOUT_SECONDS.

    Uses a bounded loop of non-blocking fcntl.flock() attempts rather than a single
    blocking flock() call, so a lock held by a hung (but still alive) process can
    never wedge this call forever — the caller falls back to a standalone file
    instead of waiting indefinitely.

    Args:
        lock_path (Path): Path to the sidecar lock file. Created if it doesn't exist.

    Returns:
        The open file object holding the lock (caller must close it — via
        _release_lock — when done), or None if the timeout elapsed first or the
        lock file could not be created/opened at all.
    """
    # Creating/opening the sidecar must never be fatal: this is a coordination
    # nicety, and an OSError here (read-only mount, permissions, ENOSPC, a
    # directory in the way) previously escaped all the way out of
    # append_to_prod_log and killed a save that would otherwise have gone
    # through. Treat it as a failed acquisition — the caller's fallback path
    # writes a standalone timestamped file, which needs no lock at all.
    try:
        lock_path.touch(exist_ok=True)
        fh = open(lock_path, "r+")
    except OSError as e:
        logging.warning(
            "Could not open lock file %s (%s) — proceeding without the lock",
            lock_path, e,
        )
        return None
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if time.monotonic() >= deadline:
                fh.close()
                return None
            time.sleep(_LOCK_POLL_SECONDS)


def _release_lock(lock_fh) -> None:
    """
    Release a lock acquired by _acquire_lock() and close its file handle.

    Deliberately does NOT unlink the lock file: removing a flock'd file while
    another process might be about to open/lock that same path is a TOCTOU race
    (the other process could end up locking a different, newly-created inode of
    the same name, defeating the lock entirely). The sidecar file is small and
    harmless to keep around indefinitely — it's gitignored.

    Args:
        lock_fh: The open file object returned by _acquire_lock().
    """
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fh.close()


def _append_locked(results: list, balance_before: float, balance_after: float, *,
                   run_note: str = "") -> Path:
    """
    Add this run's rows to the shared trade log (creating it if needed) and save it.

    Call it only while holding the log's lock (see append_to_prod_log).

    Args:
        results (list): This run's TradeResult objects.
        balance_before (float): Cash on all shards together before this run's trades, in dollars.
        balance_after (float): The same after this run's trades.
        run_note (str): Keyword-only note for the banner row; empty adds nothing.

    Returns:
        Path: The trade log's path (PROD_LOG_PATH).
    """
    if PROD_LOG_PATH.exists():
        wb = openpyxl.load_workbook(PROD_LOG_PATH)
        ws = wb.active
    else:
        # First run — create the workbook and apply the header row
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Trade Log"
        _apply_header_row(ws, _HEADER_FILL_PROD)

    # Use local time for the separator row so timestamps are human-readable
    run_ts = datetime.now(UTC).astimezone()
    _write_separator_row(ws, run_ts, balance_before, balance_after, len(results),
                         run_note=run_note)
    _write_trade_rows(ws, results, run_ts)

    # Atomic save: openpyxl's wb.save() writes directly to the target path,
    # truncating it first — a crash partway through would destroy the entire
    # accumulated history. Writing to a tmp file and renaming it into place
    # means the visible PROD_LOG_PATH only ever transitions between complete
    # states (os.replace is an atomic rename on POSIX).
    tmp_path = PROD_LOG_PATH.with_name(PROD_LOG_PATH.name + ".tmp")
    wb.save(tmp_path)
    os.replace(tmp_path, PROD_LOG_PATH)
    logging.info("Trade log updated: %s (%d new row(s))", PROD_LOG_PATH, len(results))
    return PROD_LOG_PATH


def _write_fallback_log(results: list, balance_before: float, balance_after: float, *,
                        run_note: str = "") -> Path:
    """
    Write this run's rows to a new timestamped file instead of the shared trade log.

    Used only when the shared log's lock could not be taken in time, so this run's
    rows are never lost.

    Args:
        results (list): This run's TradeResult objects.
        balance_before (float): Cash on all shards together before this run's trades, in dollars.
        balance_after (float): The same after this run's trades.
        run_note (str): Keyword-only note for the banner row, as in the shared log.

    Returns:
        Path: The new file, trade_log_<date>_<time>.xlsx in the project folder.
    """
    run_ts = datetime.now(UTC).astimezone()
    # Microseconds keep two near-simultaneous fallbacks off the collision path at
    # all (preserving the existing sort order), and create_new_output guarantees
    # they cannot overwrite each other even if they do collide (TS-18). The name
    # is suffixed on collision, never the timestamp: one run_ts serves the
    # filename AND every row below, so a fallback file's rows always match its
    # own name.
    fallback_path = PROJECT_ROOT / f"trade_log_{run_ts.strftime('%Y-%m-%d_%H%M%S_%f')}.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Trade Log"
    _apply_header_row(ws, _HEADER_FILL_PROD)
    _write_separator_row(ws, run_ts, balance_before, balance_after, len(results),
                         run_note=run_note)
    _write_trade_rows(ws, results, run_ts)

    fallback_path, fh = create_new_output(fallback_path)
    with fh:
        wb.save(fh)
    logging.info("Fallback trade log written: %s (%d row(s))", fallback_path, len(results))
    return fallback_path


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def append_to_prod_log(results: list, balance_before: float, balance_after: float, *,
                       run_note: str = "") -> Path:
    """
    Add this run's trades to the shared production trade log, trade_log.xlsx.

    Creates the file with a header row the first time. Each run adds a banner
    row, then one colour-coded row per trade. A lock stops two runs writing at
    once; if it cannot be taken in time, the rows go to a separate timestamped
    file instead, so they are never lost.

    Args:
        results (list): TradeResult objects from trader.execute_trades(); may be empty.
        balance_before (float): Cash on all shards together before this run's trades, in dollars.
        balance_after (float): The same after this run's trades.
        run_note (str): Keyword-only note for the banner row; empty adds nothing.

    Returns:
        Path: The file written: the shared log, or the fallback file if the lock timed out.
    """
    lock_fh = _acquire_lock(_LOCK_PATH)
    if lock_fh is None:
        # Never clobber the shared log and never lose this run's rows: another
        # process appears to be mid-save. Write a standalone file instead.
        logging.warning(
            "Could not acquire lock on %s within %ds — writing this run's rows "
            "to a standalone fallback file instead of the shared trade log",
            _LOCK_PATH, _LOCK_TIMEOUT_SECONDS,
        )
        return _write_fallback_log(results, balance_before, balance_after, run_note=run_note)

    try:
        return _append_locked(results, balance_before, balance_after, run_note=run_note)
    finally:
        _release_lock(lock_fh)


def write_dev_simulation(
    results: list,
    all_candidates: list,
    balance_cents: int,
) -> Path:
    """
    Write a new timestamped dev simulation Excel file with two sheets.

    Sheet 1 ("Simulated Trades") contains trades that would have been executed
    if this were a real production run. Sheet 2 ("All Candidates") lists every
    qualifying pair discovered by the scanner — tradeable and non-tradeable alike —
    so the developer can see what the bot found before sizing decisions were applied.

    A new file is created on each dev run (never appended to) so simulation outputs
    are preserved for later comparison. The filename embeds the run timestamp.

    Args:
        results (list): List of TradeResult objects from trader.execute_trades()
            with status="simulated". May be empty.
        all_candidates (list): List of CandidatePair objects from scanner.py,
            representing all pairs discovered this run (regardless of tradeability).
        balance_cents (int): Virtual account balance in cents used for trade sizing.

    Returns:
        Path: Absolute path to the simulation file actually created
            (PROJECT_ROOT / "dev_simulation_YYYY-MM-DD_HHMMSS_ffffff.xlsx", with
            a "-1", "-2", … stem suffix on collision — see
            config.create_new_output).
    """
    run_ts   = datetime.now(UTC).astimezone()
    # Microseconds plus exclusive creation, same reasoning as
    # _write_fallback_log: two dev runs finishing in one second must not
    # overwrite each other's simulation (TS-18)
    filename = f"dev_simulation_{run_ts.strftime('%Y-%m-%d_%H%M%S_%f')}.xlsx"
    out_path = PROJECT_ROOT / filename

    wb = openpyxl.Workbook()

    # ── Sheet 1: Simulated Trades ──────────────────────────
    ws_trades = wb.active
    ws_trades.title = "Simulated Trades"
    _apply_header_row(ws_trades, _HEADER_FILL_DEV)

    # Run summary in row 2
    summary_cell = ws_trades.cell(
        row=2, column=1,
        value=(f"DEV SIMULATION  |  Run: {run_ts.strftime('%Y-%m-%d %H:%M')}  |  "
               f"Balance: ${balance_cents / 100:.2f}  |  "
               f"{len(results)} simulated trade(s)")
    )
    summary_cell.font = Font(italic=True, bold=True, color="375623", size=9)
    summary_cell.fill = PatternFill("solid", fgColor="E2EFDA")
    ws_trades.merge_cells(
        start_row=2, start_column=1,
        end_row=2, end_column=len(_TRADE_COLUMNS)
    )

    for result in results:
        row_idx = ws_trades.max_row + 1
        row_data = _result_to_row(result, run_ts)
        for col_idx, value in enumerate(row_data, start=1):
            ws_trades.cell(row=row_idx, column=col_idx, value=value)
        _apply_data_row_styles(ws_trades, row_idx, result.status)
        _apply_number_formats(ws_trades, row_idx)

    # ── Sheet 2: All Candidates ────────────────────────────
    ws_cands = wb.create_sheet("All Candidates")
    # nB is the traded NO-leg price of a time-series pair (reporting-only for
    # same-title); "Price Diff" is the gap each finder actually tested — the
    # later leg's premium pB − pA for time-series, pA − pB for same-title.
    cand_headers = [
        ("Pair Type", 14),
        ("Market A", 45), ("Ticker A", 18),
        ("Market B", 45), ("Ticker B", 18),
        ("A Deadline", 12), ("B Deadline", 12),
        ("pA (YES ask)", 13), ("pB (YES ask)", 13), ("nA (NO ask)", 13),
        ("nB (NO ask)", 13),
        ("Price Diff", 12), ("Tradeable?", 12),
    ]
    for col_idx, (header, width) in enumerate(cand_headers, start=1):
        cell = ws_cands.cell(row=1, column=col_idx, value=header)
        cell.font  = Font(bold=True, color="FFFFFF", size=11)
        cell.fill  = PatternFill("solid", fgColor="375623")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws_cands.column_dimensions[get_column_letter(col_idx)].width = width
    ws_cands.freeze_panes = "A2"
    ws_cands.row_dimensions[1].height = 28

    for pair in all_candidates:
        row_idx = ws_cands.max_row + 1
        # The gap the finder tested: a time-series candidate needs the LATER
        # contract's YES ask above the earlier's; a same-title candidate needs
        # market_a (the pricier side) above market_b
        diff    = pair.pB - pair.pA if pair.pair_type == "time_series" else pair.pA - pair.pB
        row_data = [
            pair.pair_type,
            display_title(pair.market_a), pair.market_a.ticker,
            display_title(pair.market_b), pair.market_b.ticker,
            pair.market_a.close_time.strftime("%Y-%m-%d") if pair.market_a.close_time else "",
            pair.market_b.close_time.strftime("%Y-%m-%d") if pair.market_b.close_time else "",
            round(pair.pA, 4),
            round(pair.pB, 4),
            round(pair.nA, 4),
            round(pair.nB, 4),
            round(diff, 4),
            "YES" if pair.tradeable else "no",
        ]
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws_cands.cell(row=row_idx, column=col_idx, value=value)
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(vertical="center")

        # Highlight tradeable rows
        row_fill = PatternFill("solid", fgColor="E2EFDA" if pair.tradeable else "FFFFFF")
        for col in range(1, len(cand_headers) + 1):
            ws_cands.cell(row=row_idx, column=col).fill = row_fill

        # Format the five price columns as percentages: pA=8, pB=9, nA=10,
        # nB=11, Price Diff=12 (1-based; Pair Type occupies column 1)
        for col in (8, 9, 10, 11, 12):
            ws_cands.cell(row=row_idx, column=col).number_format = "0.00%"

    out_path, fh = create_new_output(out_path)
    with fh:
        wb.save(fh)
    logging.info("Dev simulation written: %s", out_path)
    return out_path


# ─────────────────────────────────────────────
# Run result (main.py --result-file)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class LegRecord:
    """
    One leg of a pair as a run result records it.

    None means "not known": every field but ticker is None in the record
    report_trades keeps for a pair it could not describe, and price is also
    None when the leg's price is not a finite number.

    Attributes:
        ticker (str): The market's ticker.
        market (str | None): The market's title with its outcome label
            (scanner.display_title).
        side (str | None): "yes" or "no", the side this leg buys (scanner.leg_sides).
        count (int | None): The contracts this leg was sized and ordered at.
            How many were actually bought depends on the pair's status: none
            on a pair that failed or was simulated, and a rolled-back leg was
            sold again.
        price (float | None): The per-contract price this leg was sized at, in
            dollars (scanner.leg_prices).
    """
    ticker: str
    market: str | None
    side: str | None
    count: int | None
    price: float | None


@dataclass(frozen=True)
class TradeRecord:
    """
    One pair's outcome as a run result records it.

    Attributes:
        status (str): The trader's status ("executed", "simulated", "failed",
            "rolled_back", "rollback_failed" or "manual_review"; see
            TradeResult), or "unknown" when it could not be read.
        error (str | None): The trader's one-line error, if any.
        pair_type (str | None): "same_title" or "time_series".
        title (str | None): The pair's title.
        a (LegRecord | None): Market A's leg; None only when it could not be described.
        b (LegRecord | None): Market B's leg; None only when it could not be described.
        cost_with_fees (float | None): What the pair was sized to cost, fees
            included, in dollars; None when not known.
        profit_if_won (float | None): Its profit if it wins, in dollars; None
            when not known.
    """
    status: str
    error: str | None
    pair_type: str | None
    title: str | None
    a: LegRecord | None
    b: LegRecord | None
    cost_with_fees: float | None
    profit_if_won: float | None


@dataclass
class RunReport:
    """
    What one production run did, saved as JSON for the program that started it.

    main.py fills it in as the run goes when started with --result-file, and
    writes it when the run ends. Nothing is written if the run stops before
    logging is set up (a usage error) or is killed.

    Attributes:
        dry_run (bool): True when no orders were sent.
        started_at (datetime): When the run started, in UTC.
        settings (str): The run's settings, as the "Live settings:" line words them.
        defaults (str): Where the saved live defaults the run started from came from.
        message (str): The line the run stopped or finished on, exactly as logged; "" until set.
        balance_before (float | None): Cash on all shards before trading, in dollars; None if not read.
        balance_after (float | None): The same after trading; None if not read.
        portfolio_value_before (float | None): Cash plus the open positions' value before trading,
            in dollars (cash alone if that value was unreadable); None if not read.
        submission_started (bool): True once a real-money run starts sending orders; with no
            trades listed, some orders may still have gone out.
        trades (list[TradeRecord]): One per pair sent or simulated, in order; filled in when trading ends.
        warnings (list[str]): Each WARNING, ERROR or CRITICAL line logged, oldest first; long WARNINGs are cut.
        warnings_dropped (int): How many WARNING lines were left out once the cap was reached.
        error (str | None): The exception that stopped the run, on one line.
    """
    dry_run: bool
    started_at: datetime
    settings: str = ""
    defaults: str = ""
    message: str = ""
    balance_before: float | None = None
    balance_after: float | None = None
    portfolio_value_before: float | None = None
    submission_started: bool = False
    trades: list[TradeRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warnings_dropped: int = 0
    error: str | None = None


class RunReportHandler(logging.Handler):
    """
    A logging handler that copies each WARNING-or-worse line a run logs into its RunReport.

    main() adds it to the root logger for a run started with --result-file and
    removes it once the result is written. Each record keeps only its first
    line. An ERROR or CRITICAL line is always kept whole: those are the lines
    a person has to act on (a failed rollback, a V2 mapping disproof that
    names every other position to check), and a run logs only a few of them.
    A WARNING line is cut at config.RUN_REPORT_LINE_MAX_CHARS, its last
    character then "…", and once config.RUN_REPORT_MAX_WARNINGS WARNING lines
    are kept the rest are only counted, so a burst of retry warnings neither
    fills the record nor pushes out a later CRITICAL.
    """

    def __init__(self, report: RunReport) -> None:
        """
        Attach the handler to one report.

        Args:
            report (RunReport): The report the lines go into.
        """
        super().__init__(level=logging.WARNING)
        self.report = report
        self._warnings_kept = 0

    def emit(self, record: logging.LogRecord) -> None:
        """
        Add one log record's first line to the report, or count it when no more WARNING lines fit.

        Args:
            record (logging.LogRecord): The record; its level is WARNING or
                worse (the handler's own level filters the rest).
        """
        try:
            first = (record.getMessage().splitlines() or [""])[0]
            if record.levelno < logging.ERROR:
                if self._warnings_kept >= RUN_REPORT_MAX_WARNINGS:
                    self.report.warnings_dropped += 1
                    return
                if len(first) > RUN_REPORT_LINE_MAX_CHARS:
                    first = first[:RUN_REPORT_LINE_MAX_CHARS - 1] + "…"
                self._warnings_kept += 1
            self.report.warnings.append(f"{record.levelname}: {first}")
        except Exception:
            # logging's own reporting of a handler that failed; never raises
            self.handleError(record)


def _json_number(value, digits: int) -> float | None:
    """
    Round a money figure for the run result, or None when it is not a finite number.

    The result is written as strict JSON, which has no NaN or infinity, so a
    figure that is not finite is recorded as not known rather than losing the
    whole result.

    Args:
        value: The figure (a float in practice).
        digits (int): Decimal places to keep.

    Returns:
        float | None: The figure as a float, rounded; None if it is not finite.

    Raises:
        TypeError, ValueError: When value is not a number (float() refuses it);
            trade_record lets this through, and report_trades catches it.
    """
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def _text_or_none(value) -> str | None:
    """
    Keep a value for a run result only if it is text.

    Args:
        value: Any value read off a trade result.

    Returns:
        str | None: The value if it is a str, else None.
    """
    return value if isinstance(value, str) else None


def trade_record(result: TradeResult) -> TradeRecord:
    """
    Describe one pair's outcome for a RunReport, in the trade log's terms.

    Every value is a plain str, int, float or None (a figure that is not a
    finite number is None), so the record can always be written as JSON.

    Args:
        result (TradeResult): The trader's result for one pair.

    Returns:
        TradeRecord: Its markets (A then B, as the trade log lists them), the
            side each leg buys and the count and price it was sized at, its
            cost with fees and its profit if it wins.

    Raises:
        Exception: Whatever reading the result raises when its spec, pair or
            figures cannot be read (an AttributeError, TypeError, ValueError
            or KeyError in practice); report_trades catches it and keeps a
            record of what could be read instead.
    """
    spec = result.spec
    pair = spec.pair
    # Which side each market's leg bought, and at what price: the one source of truth
    side_a, side_b = leg_sides(pair.pair_type)
    price_a, price_b = leg_prices(pair)
    return TradeRecord(
        status=str(result.status),
        error=None if result.error is None else str(result.error),
        pair_type=_text_or_none(pair.pair_type),
        title=_text_or_none(pair.canonical_title),
        a=LegRecord(str(pair.market_a.ticker), display_title(pair.market_a), side_a,
                    int(spec.x), _json_number(price_a, 4)),
        b=LegRecord(str(pair.market_b.ticker), display_title(pair.market_b), side_b,
                    int(spec.y), _json_number(price_b, 4)),
        cost_with_fees=_json_number(spec.total_cost_with_fees, 2),
        profit_if_won=_json_number(spec.min_payoff, 2),
    )


def _fallback_leg(market, count) -> LegRecord | None:
    """
    Record what can be read of one leg of a pair trade_record could not describe.

    Args:
        market: The leg's market, or None.
        count: The contracts it was sized at, or None.

    Returns:
        LegRecord | None: The ticker and count (None where not an int), the
            rest not known; None when the ticker cannot be read.

    Raises:
        Exception: Whatever reading market.ticker raises other than
            AttributeError; report_trades catches it.
    """
    ticker = _text_or_none(getattr(market, "ticker", None))
    if not ticker:
        return None
    return LegRecord(ticker, None, None,
                     count if type(count) is int else None, None)


def report_trades(results: list) -> list[TradeRecord]:
    """
    Describe every pair's outcome for a RunReport; never raises.

    A pair that cannot be described is logged as an ERROR and kept as a record
    holding what the trade log's rescue lines read of it (status, error, title
    and both tickers and counts, each only if it can be read), so the run goes
    on to write the trade log after it. If even that cannot be read, the
    record says only that its status is "unknown".

    Args:
        results (list[TradeResult]): The trader's results, in submission order.

    Returns:
        list[TradeRecord]: One record per result, in the same order.
    """
    records = []
    for result in results:
        try:
            records.append(trade_record(result))
            continue
        except Exception as exc:
            logging.error("Could not describe a pair for the run result: %s", exc)
        try:
            spec = getattr(result, "spec", None)
            pair = getattr(spec, "pair", None)
            records.append(TradeRecord(
                status=_text_or_none(getattr(result, "status", None)) or "unknown",
                error=_text_or_none(getattr(result, "error", None)),
                pair_type=_text_or_none(getattr(pair, "pair_type", None)),
                title=_text_or_none(getattr(pair, "canonical_title", None)),
                a=_fallback_leg(getattr(pair, "market_a", None), getattr(spec, "x", None)),
                b=_fallback_leg(getattr(pair, "market_b", None), getattr(spec, "y", None)),
                cost_with_fees=None, profit_if_won=None))
        except Exception:
            # A field that raises when read (not merely missing): keep the
            # pair's place in the list with nothing but an unknown status
            records.append(TradeRecord(
                status="unknown", error=None, pair_type=None, title=None, a=None, b=None,
                cost_with_fees=None, profit_if_won=None))
    return records


def write_run_report(path: Path, report: RunReport, exit_code: int | None) -> None:
    """
    Write a run's result as JSON, replacing the file whole; never raises.

    The record is written to a hidden staging file beside path and renamed
    over it, so a reader sees the old file or the whole new one, never part
    of one. It is strict JSON (no NaN). Anything that goes wrong is logged as
    an ERROR and leaves no staging file; main() calls this in its finally, so
    it must not raise.

    Args:
        path (Path): The result file.
        report (RunReport): What the run did.
        exit_code (int | None): The run's exit code; None when an exception
            stopped it (report.error says which).
    """
    staging = None
    try:
        record = {
            "format": LIVE_RUN_RESULT_FORMAT, "mode": "prod", "dry_run": report.dry_run,
            "started_at": report.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exit_code": exit_code, "settings": report.settings, "defaults": report.defaults,
            "message": report.message, "balance_before": report.balance_before,
            "balance_after": report.balance_after,
            "portfolio_value_before": report.portfolio_value_before,
            "submission_started": report.submission_started,
            "trades": [asdict(t) for t in report.trades],
            "warnings": list(report.warnings), "warnings_dropped": report.warnings_dropped,
            "error": report.error,
        }
        text = json.dumps(record, allow_nan=False, indent=1)
        staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staging.write_text(text, encoding="utf-8")
        os.replace(staging, path)
        logging.info("Run result written: %s", path)
    except Exception as exc:  # it runs in main()'s finally: it must never raise
        logging.error("Could not write the run result to %s: %s", path, exc)
        if staging is not None:
            with contextlib.suppress(OSError):
                staging.unlink(missing_ok=True)
