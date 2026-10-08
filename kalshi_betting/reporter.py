"""
File: reporter.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Writes the bot's Excel files. A production run adds a banner row and one
    row per trade to the shared trade_log.xlsx, which keeps every run's history.
    A dev run writes a new timestamped file with two sheets: the simulated
    trades and every candidate pair found.

    A production run that sells held positions (the take-profit rule,
    seller.py and trader.sell_positions) adds its sales to the same log, with
    a banner row of their own, before it buys anything: one row per position,
    under the same columns as a trade.

    It also writes the run result of a production run started with
    `main.py --result-file`: a RunReport that main.py fills in as the run goes
    (its outcome, its cash before and after trading, the portfolio value it
    sized on, whether it began sending orders, one record per sale and per
    pair, and its WARNING-or-worse log lines), saved as one JSON file when the
    run ends.

Dependencies:
    Imports display_title, leg_sides (which side each leg buys, rendered
    into the Notes prefix), leg_prices (the price each leg was sized at,
    for the run result) and pair_held (the held pair an add-on adds to, for
    the "adds to N held" marker) from scanner.py and TradeSpec from
    strategy.py.
    Imports PROJECT_ROOT, create_new_output, count_text (writes the held
    count of an add-on exactly), fee_leg_exact (the fee on a sale, for a sale
    row's Notes) and the run result's constants (LIVE_RUN_RESULT_FORMAT,
    RUN_REPORT_MAX_WARNINGS, RUN_REPORT_LINE_MAX_CHARS) from config.py.
    Exports the TradeResult and SaleResult dataclasses (built by trader.py: a
    pair's trade, and a held position's sale), the two public write
    functions, and the run result's RunReport, TradeRecord, LegRecord,
    SaleRecord, SaleLegRecord, RunReportHandler, trade_record, report_trades,
    sale_record, report_sales and write_run_report (all consumed by main.py).
    A sale's plan (seller.SalePlan) is read by its attributes only; seller.py
    is never imported here.

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
    ("[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=0.xxxx] fees=$x.xx[ adds to N
    held]] ") is what tells a reader which side each count bought, for a
    time-series row the traded NO-leg price — the retained "nA (NO ask)"
    column is reporting-only there — and, for a trade that adds to a pair the
    account already held, how many contracts a side it held.

    append_to_prod_log's keyword-only run_note goes on each separator banner
    the run writes (one above its sales, when it sells, and one above its
    trades), never in a column; main._run_prod passes the run's live toggles
    (config.describe_live_settings), with "(default: X)" after each toggle a
    flag moved away from the saved live defaults, ending " | defaults:
    <origin>" (which saved file, when and from what it was saved).

    A sale row fills the trade columns this way: Market A/B and Ticker A/B are
    the position's markets (a lone held market's paid-out partner is market B,
    named "(paid out)"), the deadlines are the held markets' close times, pA
    and pB are the average bids the sale was priced at (blank for a paid-out
    partner), nA is blank, x and y are the contracts sold on each market,
    the cost is what the whole position cost, "Profit if won" is the profit
    selling it realizes at those bids (with a paid-out partner's payout) and
    the ratio is that profit over the cost, both left blank unless the whole
    position sold (or, in a dry run, would have). Its Status is a plain word
    (_SALE_STATUS_WORDS) and its Notes start "[sale: YES A / NO B, 84% of
    potential profit (level 80%), 9 days left, fees=$0.42] ".

    Neither filling nor writing a run result may stop a run or change its exit
    code: report_trades and report_sales never raise (a pair or sale they
    cannot describe is logged and kept as a record of what could be read),
    and write_run_report catches every error, since main() calls it in a
    finally. The result is replaced
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
from typing import IO, Any

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .config import (
    LIVE_RUN_RESULT_FORMAT,
    PROJECT_ROOT,
    RUN_REPORT_LINE_MAX_CHARS,
    RUN_REPORT_MAX_WARNINGS,
    count_text,
    create_new_output,
    fee_leg_exact,
)
from .scanner import display_title, leg_prices, leg_sides, pair_held
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
                in the run, or a pair adding to a held pair whose held
                positions no longer matched it), so nothing of this pair is
                open.
            "rolled_back": the NO leg filled, the YES leg did not, and this
                pair's NO contracts were all sold back. Anything the account
                held on that market before the pair is left as it was.
            "rollback_failed": that closing order filled only partly or not
                at all, so some of this pair's NO contracts are left open for
                a person to handle.
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


@dataclass
class SaleResult:
    """
    Outcome record for one held position the take-profit rule sold, or tried to.

    Built by trader.sell_positions, one per seller.SalePlan.

    Attributes:
        plan (Any): The seller.SalePlan that was sold or simulated.
        status (str): What happened.
            "sold": every contract held on each of its held markets sold.
            "partly_sold": some sold; for a pair, the same count on each
                market, so the rest is still an exact pair.
            "not_sold": nothing sold, or nothing was sent (error says which);
                in a dry run too, for a plan whose orders cannot be built.
            "unbalanced": a pair's second order sold fewer contracts than its
                first, so the two markets no longer hold the same count.
            "manual_review": how many contracts an order sold could not be
                known, so no further order was sent for the position.
            "simulated": dry run; nothing was sent.
        sold (dict[str, int]): Held ticker -> contracts known to be sold
            there (0 for a market whose order filled nothing or was not
            sent). A market whose sale could not be known is left out, and
            the dict is empty when the plan could not be used as written (a
            held market's count, side, walked bid or ladder unreadable) or an
            unexpected error stopped its sale. A dry run lists every held
            market's full count.
        error (str | None): What went wrong or was left over; None for
            "sold" and "simulated".
        decided_by_account (bool): True when the reply of at least one of
            its orders did not say how many contracts it sold (the POST
            raised, or the reply had no usable fill count), so the account's
            position was read to find out (trader._sell_leg). A lagging
            ledger can show fewer sold than really were, so main._run_prod
            reads the positions back after such a sale and checks the count.
            False for a dry run and when every reply said.
    """
    plan: Any
    status: str            # "sold" | "partly_sold" | "not_sold" | "unbalanced" | "manual_review" | "simulated"
    sold: dict[str, int]
    error: str | None = None
    decided_by_account: bool = False


# A sale's status in plain words, as the trade log's Status column shows it
# ("check": a person must look at the position in the Kalshi UI)
_SALE_STATUS_WORDS = {
    "sold":          "sold",
    "partly_sold":   "partly sold",
    "not_sold":      "not sold",
    "unbalanced":    "unbalanced",
    "manual_review": "check",
    "simulated":     "simulated",
}

# The sale statuses in which the whole position sold (or, in a dry run,
# would have), so the plan's profit is what the sale realized
_SALE_WHOLE_STATUSES = ("sold", "simulated")


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
    the traded NO-leg price nB, the fees, and, for a trade that adds to a
    pair the account already held (scanner.pair_held), how many contracts a
    side it held — "[time_series: YES A / NO B nB=0.4000 fees=$0.10] ",
    "[same_title: NO A / YES B fees=$0.10] " or "[time_series: YES A / NO B
    nB=0.4000 fees=$0.10 adds to 30 held] " — followed by result.error (if
    any). That prefix is what disambiguates rows in a workbook whose header
    row predates the side-neutral x/y headers, and it is the only place nB is
    recorded (the "nA (NO ask)" column is reporting-only for a time-series
    row). No column is added for an add-on: every other row is unchanged.

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
    # Cross-module: a trade adding to a held pair names the count held a side
    held = pair_held(pair)
    held_note = f" adds to {count_text(held.count)} held" if held is not None else ""
    notes = (
        f"[{pair.pair_type}: {side_a.upper()} A / {side_b.upper()} B{nb_note}"
        f" fees=${fees:.2f}{held_note}] "
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


def _days_left_text(days_left: Any) -> str:
    """
    Say how many days a sold position had left before its last market stops trading.

    Args:
        days_left (Any): SalePlan.days_left: whole days, or None when a close
            date was unknown.

    Returns:
        str: "9 days left", "1 day left" or "days left unknown".
    """
    if isinstance(days_left, bool) or not isinstance(days_left, int):
        return "days left unknown"
    return f"{days_left} day{'' if days_left == 1 else 's'} left"


def _sale_leg_cells(leg: Any, plan: Any, sold: dict) -> list:
    """
    The five trade-log cells one market of a sold position fills.

    Args:
        leg (Any): One of the plan's markets (a seller.SaleLeg): held now
            (it has a market) or a paid-out partner (its market is None).
        plan (Any): The seller.SalePlan.
        sold (dict): SaleResult.sold: held ticker -> contracts known to be sold.

    Returns:
        list: [market, ticker, deadline, average bid sold at, contracts sold]
            for a held market (the count blank when it is not known); for a
            paid-out partner, "(paid out)" and its ticker, the rest blank.
    """
    if leg.market is None:
        return ["(paid out)", leg.ticker, "", "", ""]
    close = leg.market.close_time
    count = sold.get(leg.ticker)
    return [display_title(leg.market), leg.ticker,
            close.strftime("%Y-%m-%d") if close else "",
            round(plan.walked[leg.ticker][0], 4),
            count if isinstance(count, int) and not isinstance(count, bool) else ""]


def _sale_to_row(sale: SaleResult, run_ts: datetime) -> list:
    """
    Serialize one position's sale to a flat list matching the _TRADE_COLUMNS column order.

    The position's markets fill the A and B columns in the plan's order (its
    held markets in ticker order, then a lone held market's paid-out
    partner). pA/pB hold the average bid each held market was priced to sell
    at (SalePlan.walked), nA is blank, x/y hold the contracts sold on each
    market (SaleResult.sold), the cost is what the whole position cost,
    "Profit if won" is the profit selling it realizes at those bids (with a
    paid-out partner's payout: SalePlan.profits at the check now) and the
    ratio is that profit over the cost. Those are the plan's figures, so they
    are filled in only when the whole position sold ("sold"), or would have
    in a dry run ("simulated"), and left blank for any other status; the
    Status says how much really sold. The Notes start "[sale: YES A / NO B,
    84% of potential profit (level 80%), 9 days left, fees=$0.42] ", the fees
    being the taker fee on each held market's sale at its average bid, and
    end with the sale's error (if any).

    Args:
        sale (SaleResult): The sale, from trader.sell_positions.
        run_ts (datetime): Timestamp of this run, for the Date and Time columns.

    Returns:
        list: Ordered list of 18 values, one per column in _TRADE_COLUMNS.
    """
    plan = sale.plan
    legs = list(plan.legs)
    sold = sale.sold if isinstance(sale.sold, dict) else {}
    cells_a = _sale_leg_cells(legs[0], plan, sold)
    cells_b = _sale_leg_cells(legs[1], plan, sold) if len(legs) > 1 else [""] * 5
    # The side held on each market: market A first, then market B
    sides = " / ".join(f"{leg.side.upper()} {'AB'[i]}" for i, leg in enumerate(legs[:2]))
    # The sale's fees as planned. Cross-module: config.fee_leg_exact, the one
    # fee rule, on each held market's sale at its average bid
    fees = sum(fee_leg_exact(plan.count, plan.walked[leg.ticker][0])
               for leg in legs if leg.market is not None)
    realized, potential = plan.profits[0]
    # The plan's profit is what the sale realizes only if all of it sold
    whole = sale.status in _SALE_WHOLE_STATUSES
    notes = (f"[sale: {sides}, {realized / potential:.0%} of potential profit "
             f"(level {plan.level:.0%}), {_days_left_text(plan.days_left)}, "
             f"fees=${fees:.2f}] " + (sale.error or ""))
    return [
        run_ts.strftime("%Y-%m-%d"),
        run_ts.strftime("%H:%M:%S"),
        cells_a[0], cells_a[1], cells_b[0], cells_b[1],
        cells_a[2], cells_b[2],
        cells_a[3], cells_b[3],
        "",
        cells_a[4], cells_b[4],
        round(plan.cost_dollars, 2),
        round(realized, 2) if whole else "",
        round(realized / plan.cost_dollars, 4) if whole and plan.cost_dollars else "",
        _SALE_STATUS_WORDS.get(sale.status, str(sale.status)),
        notes,
    ]


def _apply_data_row_styles(ws, row_idx: int, status: str) -> None:
    """
    Apply background fill, border, and alignment styling to a single data row.

    Color-codes rows by status: green for "executed" and a sale "sold", blue
    for "simulated", red/orange for "failed" and a sale "not sold", yellow
    for "rolled_back" and a sale "partly sold", strong red for
    "rollback_failed", "manual_review" and a sale "unbalanced" or "check",
    white for any unknown status.

    Args:
        ws: An openpyxl Worksheet object to apply styles to.
        row_idx (int): 1-based row index of the data row to style.
        status (str): The row's Status cell: a trade's status ("executed",
            "simulated", "failed", "rolled_back", "rollback_failed" or
            "manual_review") or a sale's word (_SALE_STATUS_WORDS).
    """
    status_colors = {
        "executed":        "E2EFDA",   # light green
        "simulated":       "EBF3FB",   # light blue
        "failed":          "FCE4D6",   # light red/orange
        "rolled_back":     "FFF2CC",   # light yellow — this pair's NO leg unwound
        "rollback_failed": "F4B7B4",   # strong red — orphaned position, manual review
        "manual_review":   "F4B7B4",   # strong red — fill state unknown, manual review
        "sold":            "E2EFDA",   # light green — the whole position sold
        "partly sold":     "FFF2CC",   # light yellow — the rest is still held
        "not sold":        "FCE4D6",   # light red/orange — still held
        "unbalanced":      "F4B7B4",   # strong red — a pair left uneven, manual review
        "check":           "F4B7B4",   # strong red — what sold is unknown, manual review
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
    *, run_note: str = "", n_sales: int = 0,
) -> None:
    """
    Add one grey banner row above a batch of rows: time, cash before and after, and counts.

    The shared trade log and the fallback file both use it; the sale count
    follows the trade count when there are sales, and a non-empty run_note
    goes at the end.

    Args:
        ws: The worksheet to add the row to.
        run_ts (datetime): When this run happened.
        balance_before (float): Cash on all shards together before these rows' orders, in dollars.
        balance_after (float): The same after them.
        n_results (int): How many trade results the rows below hold.
        run_note (str): Keyword-only text for the end of the banner; empty adds nothing.
        n_sales (int): Keyword-only. How many sales the rows below hold; 0 adds nothing.
    """
    sep_row = ws.max_row + 1
    sep_cell = ws.cell(row=sep_row, column=1,
                       value=f"── Run: {run_ts.strftime('%Y-%m-%d %H:%M')}  |  "
                             f"Balance before: ${balance_before:.2f}  →  "
                             f"after: ${balance_after:.2f}  |  "
                             f"{n_results} trade(s)"
                             + (f", {n_sales} sale(s)" if n_sales else "")
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
        _write_row(ws, _result_to_row(result, run_ts))


def _write_row(ws, row_data: list) -> None:
    """
    Append one styled, color-coded data row, its colour read from its Status cell.

    Args:
        ws: The openpyxl Worksheet to append to.
        row_data (list): The row's 18 values, in _TRADE_COLUMNS order.
    """
    row_idx = ws.max_row + 1
    for col_idx, value in enumerate(row_data, start=1):
        ws.cell(row=row_idx, column=col_idx, value=value)
    # The Status column (17th): a trade's status or a sale's word
    _apply_data_row_styles(ws, row_idx, row_data[16])
    _apply_number_formats(ws, row_idx)


def _write_sale_rows(ws, sales: list, run_ts: datetime) -> None:
    """
    Append one styled, color-coded data row per SaleResult (_sale_to_row).

    Args:
        ws: The openpyxl Worksheet to append to.
        sales (list): SaleResult objects to write, in order.
        run_ts (datetime): Timestamp of this run, used for the Date/Time columns.
    """
    for sale in sales:
        _write_row(ws, _sale_to_row(sale, run_ts))


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
                   run_note: str = "", sales: tuple | list = ()) -> Path:
    """
    Add this run's rows to the shared trade log (creating it if needed) and save it.

    Call it only while holding the log's lock (see append_to_prod_log).

    Args:
        results (list): This run's TradeResult objects.
        balance_before (float): Cash on all shards together before these orders, in dollars.
        balance_after (float): The same after them.
        run_note (str): Keyword-only note for the banner row; empty adds nothing.
        sales (tuple | list): Keyword-only. This run's SaleResult objects,
            written before the trades; empty adds nothing.

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
                         run_note=run_note, n_sales=len(sales))
    _write_sale_rows(ws, sales, run_ts)
    _write_trade_rows(ws, results, run_ts)

    # Atomic save: openpyxl's wb.save() writes directly to the target path,
    # truncating it first — a crash partway through would destroy the entire
    # accumulated history. Writing to a tmp file and renaming it into place
    # means the visible PROD_LOG_PATH only ever transitions between complete
    # states (os.replace is an atomic rename on POSIX).
    tmp_path = PROD_LOG_PATH.with_name(PROD_LOG_PATH.name + ".tmp")
    wb.save(tmp_path)
    os.replace(tmp_path, PROD_LOG_PATH)
    logging.info("Trade log updated: %s (%d new row(s))", PROD_LOG_PATH,
                 len(results) + len(sales))
    return PROD_LOG_PATH


def _write_fallback_log(results: list, balance_before: float, balance_after: float, *,
                        run_note: str = "", sales: tuple | list = ()) -> Path:
    """
    Write this run's rows to a new timestamped file instead of the shared trade log.

    Used only when the shared log's lock could not be taken in time, so this run's
    rows are never lost.

    Args:
        results (list): This run's TradeResult objects.
        balance_before (float): Cash on all shards together before these orders, in dollars.
        balance_after (float): The same after them.
        run_note (str): Keyword-only note for the banner row, as in the shared log.
        sales (tuple | list): Keyword-only. This run's SaleResult objects,
            written before the trades; empty adds nothing.

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
                         run_note=run_note, n_sales=len(sales))
    _write_sale_rows(ws, sales, run_ts)
    _write_trade_rows(ws, results, run_ts)

    fallback_path, fh = create_new_output(fallback_path)
    with fh:
        wb.save(fh)
    logging.info("Fallback trade log written: %s (%d row(s))", fallback_path,
                 len(results) + len(sales))
    return fallback_path


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def append_to_prod_log(results: list, balance_before: float, balance_after: float, *,
                       run_note: str = "", sales: tuple | list = ()) -> Path:
    """
    Add this run's trades, or its sales, to the shared production trade log, trade_log.xlsx.

    Creates the file with a header row the first time. Each call adds a
    banner row, then one colour-coded row per sale (_sale_to_row) and per
    trade. A production run that sells writes its sales in a call of their
    own, as soon as they are done and before it buys anything, so a sale is on
    record however the run ends. A lock stops two runs writing at once; if it
    cannot be taken in time, the rows go to a separate timestamped file
    instead, so they are never lost.

    Args:
        results (list): TradeResult objects from trader.execute_trades(); may be empty.
        balance_before (float): Cash on all shards together before these orders, in dollars.
        balance_after (float): The same after them.
        run_note (str): Keyword-only note for the banner row; empty adds nothing.
        sales (tuple | list): Keyword-only. SaleResult objects from
            trader.sell_positions(); empty (the default) adds no sale row.

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
        return _write_fallback_log(results, balance_before, balance_after, run_note=run_note,
                                   sales=sales)

    try:
        return _append_locked(results, balance_before, balance_after, run_note=run_note,
                              sales=sales)
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
        adds_to_held (float | None): For a trade that adds to a pair the
            account already held (scanner.pair_held), the contracts it held
            on each market; None for any other trade. The defaults server's
            run page shows it as "(adds to N held)".
    """
    status: str
    error: str | None
    pair_type: str | None
    title: str | None
    a: LegRecord | None
    b: LegRecord | None
    cost_with_fees: float | None
    profit_if_won: float | None
    adds_to_held: float | None = None


@dataclass(frozen=True)
class SaleLegRecord:
    """
    One held market of a sold position as a run result records it.

    Attributes:
        ticker (str): The market's ticker.
        side (str | None): "yes" or "no", the side held there.
        held (int | None): The contracts held there when the sale was planned.
        sold (int | None): The contracts known to be sold there; None when
            that could not be known (SaleResult.sold leaves it out). A dry
            run records the whole count, as if it sold.
        price (float | None): The average bid the sale was priced at
            (SalePlan.walked), in dollars; None when not a finite number.
    """
    ticker: str
    side: str | None
    held: int | None
    sold: int | None
    price: float | None


@dataclass(frozen=True)
class SaleRecord:
    """
    One held position's sale as a run result records it.

    The money figures are the plan's, at the bids the sale was priced at;
    status says how much really sold, and the profit is given only when all
    of it did.

    Attributes:
        title (str | None): The position's title (SalePlan.title).
        status (str): The sale's status ("sold", "partly_sold", "not_sold",
            "unbalanced", "manual_review" or "simulated"; see SaleResult),
            or "unknown" when it could not be read.
        level (float | None): The share of potential profit it was sold at
            (LiveSettings.sell_at, e.g. 0.8).
        days_left (int | None): Days before its last held market stops trading.
        cost (float | None): What the whole position cost, fees included, in dollars.
        proceeds (float | None): What selling all of its held markets
            returns at those bids, after the sale's fees, in dollars, as
            planned (whatever really sold).
        profit (float | None): The profit the sale realized: those
            proceeds, plus a paid-out partner's payout, less the cost. None
            unless the whole position sold ("sold"), or would have in a dry
            run ("simulated").
        realized_percent (float | None): The profit selling all of it at
            those bids realizes, as a share of its potential profit, in
            percent (e.g. 84.0): what the take-profit rule checked.
        legs (tuple[SaleLegRecord, ...]): Its held markets, in the plan's order
            (a paid-out partner has nothing to sell and is left out).
        error (str | None): The trader's one-line error, if any.
    """
    title: str | None
    status: str
    level: float | None
    days_left: int | None
    cost: float | None
    proceeds: float | None
    profit: float | None
    realized_percent: float | None
    legs: tuple[SaleLegRecord, ...]
    error: str | None


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
        submission_started (bool): True once a real-money run starts sending orders (sale
            orders included); with no trades listed, some orders may still have gone out.
        trades (list[TradeRecord]): One per pair sent or simulated, in order; filled in when trading ends.
        warnings (list[str]): Each WARNING, ERROR or CRITICAL line logged, oldest first; long WARNINGs are cut.
        warnings_dropped (int): How many WARNING lines were left out once the cap was reached.
        error (str | None): The exception that stopped the run, on one line.
        sales (list[SaleRecord]): One per held position sold, tried or
            simulated, in order; filled in when the sales end, before any buying.
        cash_after_sales (float | None): Cash on all shards after the sales,
            in dollars: in a live run, read back from Kalshi when anything
            sold, else the cash read at the start of the run; in a dry run, an
            estimate: that cash plus each sale's estimated proceeds. None when
            no sale was tried, or the cash could not be read after the sales.
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
    sales: list[SaleRecord] = field(default_factory=list)
    cash_after_sales: float | None = None


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
            cost with fees, its profit if it wins, and, for a trade that adds
            to a held pair, the count held on each market.

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
    # Cross-module: the held pair this trade adds to, read by type
    held = pair_held(pair)
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
        adds_to_held=None if held is None else _json_number(held.count, 2),
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


def _int_or_none(value) -> int | None:
    """
    Keep a value for a run result only if it is an int (a bool is not).

    Args:
        value: Any value read off a sale.

    Returns:
        int | None: The value if it is an int, else None.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def sale_record(sale: SaleResult) -> SaleRecord:
    """
    Describe one held position's sale for a RunReport.

    Every value is a plain str, int, float or None (a figure that is not a
    finite number is None), so the record can always be written as JSON.

    Args:
        sale (SaleResult): The trader's result for one position.

    Returns:
        SaleRecord: Its title, status, level and days left, its cost, the
            proceeds selling it returns at the bids it was priced at, the
            profit that realizes (only when all of it sold) and its share of
            the potential profit, and each held market's side, count held,
            count sold and price.

    Raises:
        Exception: Whatever reading the sale raises when its plan or figures
            cannot be read (an AttributeError, TypeError, ValueError,
            IndexError or KeyError in practice); report_sales catches it and
            keeps a record of what could be read instead.
    """
    plan = sale.plan
    sold = sale.sold if isinstance(sale.sold, dict) else {}
    realized, potential = plan.profits[0]
    legs = tuple(
        SaleLegRecord(str(leg.ticker), _text_or_none(leg.side), _int_or_none(leg.count),
                      _int_or_none(sold.get(leg.ticker)),
                      _json_number(plan.walked[leg.ticker][0], 4))
        for leg in plan.legs if leg.market is not None)
    return SaleRecord(
        title=_text_or_none(plan.title),
        status=str(sale.status),
        level=_json_number(plan.level, 4),
        days_left=_int_or_none(plan.days_left),
        cost=_json_number(plan.cost_dollars, 2),
        proceeds=_json_number(plan.proceeds_dollars, 2),
        # The plan's profit is what the sale realized only if all of it sold
        profit=(_json_number(realized, 2) if sale.status in _SALE_WHOLE_STATUSES
                else None),
        realized_percent=_json_number(100.0 * realized / potential, 2),
        legs=legs,
        error=None if sale.error is None else str(sale.error),
    )


def report_sales(sales: list) -> list[SaleRecord]:
    """
    Describe every held position's sale for a RunReport; never raises.

    A sale that cannot be described is logged as an ERROR and kept as a
    record holding its status, error and title (each only if it can be
    read), so the run goes on after it. If even that cannot be read, the
    record says only that its status is "unknown".

    Args:
        sales (list[SaleResult]): The trader's results, in the order sold.

    Returns:
        list[SaleRecord]: One record per sale, in the same order.
    """
    records = []
    for sale in sales:
        try:
            records.append(sale_record(sale))
            continue
        except Exception as exc:
            logging.error("Could not describe a sale for the run result: %s", exc)
        try:
            records.append(SaleRecord(
                title=_text_or_none(getattr(getattr(sale, "plan", None), "title", None)),
                status=_text_or_none(getattr(sale, "status", None)) or "unknown",
                level=None, days_left=None, cost=None, proceeds=None, profit=None,
                realized_percent=None, legs=(),
                error=_text_or_none(getattr(sale, "error", None))))
        except Exception:
            # A field that raises when read (not merely missing): keep the
            # sale's place in the list with nothing but an unknown status
            records.append(SaleRecord(
                title=None, status="unknown", level=None, days_left=None, cost=None,
                proceeds=None, profit=None, realized_percent=None, legs=(), error=None))
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
            "sales": [asdict(s) for s in report.sales],
            "cash_after_sales": report.cash_after_sales,
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
