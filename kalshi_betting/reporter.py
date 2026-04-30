"""Excel reporting: logs executed trades (prod) or simulated trades (dev)."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .config import PROJECT_ROOT
from .scanner import market_title
from .strategy import TradeSpec

PROD_LOG_PATH = PROJECT_ROOT / "trade_log.xlsx"

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
    ("x — NO on A",       12),
    ("y — YES on B",      13),
    ("Total Cost ($)",    14),
    ("Min Profit ($)",    14),
    ("Profit Ratio (%)",  16),
    ("Status",            12),
    ("Notes",             30),
]

_HEADER_FILL_PROD = PatternFill("solid", fgColor="1F4E79")   # dark blue for prod
_HEADER_FILL_DEV  = PatternFill("solid", fgColor="375623")   # dark green for dev
_HEADER_FONT      = Font(bold=True, color="FFFFFF", size=11)
_SUBHEADER_FILL   = PatternFill("solid", fgColor="D6E4F7")   # light blue
_SUBHEADER_FONT   = Font(bold=True, color="1F4E79", size=10)
_THIN_BORDER      = Border(
    bottom=Side(style="thin", color="BFBFBF"),
)


@dataclass
class TradeResult:
    spec: TradeSpec
    status: str            # "executed" | "failed" | "simulated"
    error: Optional[str] = None


def _apply_header_row(ws, fill: PatternFill) -> None:
    """Write and style the column header row."""
    for col_idx, (header, width) in enumerate(_TRADE_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font  = _HEADER_FONT
        cell.fill  = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"


def _result_to_row(result: TradeResult, run_ts: datetime) -> list:
    spec = result.spec
    pair = spec.pair
    mA   = _market_title(pair.market_a)
    mB   = _market_title(pair.market_b)

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
        round(spec.total_cost, 2),
        round(spec.min_payoff, 2),
        round(spec.profit_ratio, 4),
        result.status,
        result.error or "",
    ]


def _apply_data_row_styles(ws, row_idx: int, status: str) -> None:
    status_colors = {
        "executed":  "E2EFDA",   # light green
        "simulated": "EBF3FB",   # light blue
        "failed":    "FCE4D6",   # light red/orange
    }
    fill_color = status_colors.get(status, "FFFFFF")
    fill = PatternFill("solid", fgColor=fill_color)
    for col in range(1, len(_TRADE_COLUMNS) + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill   = fill
        cell.border = _THIN_BORDER
        cell.alignment = Alignment(vertical="center")


def _apply_number_formats(ws, row_idx: int) -> None:
    """Apply currency/percentage formats to numeric columns."""
    # Columns: pA=9, pB=10, nA=11, TotalCost=14, MinProfit=15, ProfitRatio=16
    for col in (9, 10, 11):
        ws.cell(row=row_idx, column=col).number_format = "0.00%"
    for col in (14, 15):
        ws.cell(row=row_idx, column=col).number_format = '"$"#,##0.00'
    ws.cell(row=row_idx, column=16).number_format = "0.00%"


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def append_to_prod_log(results: list, balance_before: float, balance_after: float) -> Path:
    """
    Append executed trade results to the persistent production trade log.
    Creates the file with a header row if it doesn't exist.
    Returns the path to the file.
    """
    if PROD_LOG_PATH.exists():
        wb = openpyxl.load_workbook(PROD_LOG_PATH)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Trade Log"
        _apply_header_row(ws, _HEADER_FILL_PROD)

    run_ts = datetime.now(timezone.utc).astimezone()  # local time for readability
    first_new_row = ws.max_row + 1

    # Write a run-separator row
    sep_row = ws.max_row + 1
    sep_cell = ws.cell(row=sep_row, column=1,
                       value=f"── Run: {run_ts.strftime('%Y-%m-%d %H:%M')}  |  "
                             f"Balance before: ${balance_before:.2f}  →  "
                             f"after: ${balance_after:.2f}  |  "
                             f"{len(results)} trade(s)")
    sep_cell.font = Font(italic=True, color="595959", size=9)
    sep_cell.fill = PatternFill("solid", fgColor="F2F2F2")
    ws.merge_cells(
        start_row=sep_row, start_column=1,
        end_row=sep_row, end_column=len(_TRADE_COLUMNS)
    )

    for result in results:
        row_idx = ws.max_row + 1
        row_data = _result_to_row(result, run_ts)
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
        _apply_data_row_styles(ws, row_idx, result.status)
        _apply_number_formats(ws, row_idx)

    wb.save(PROD_LOG_PATH)
    logging.info("Trade log updated: %s (%d new row(s))", PROD_LOG_PATH, len(results))
    return PROD_LOG_PATH


def write_dev_simulation(
    results: list,
    all_candidates: list,
    balance_cents: int,
) -> Path:
    """
    Write a new dev simulation Excel file.
    Sheet 1 — "Simulated Trades": trades that WOULD have been executed.
    Sheet 2 — "All Candidates": every qualifying pair found (tradeable or not).
    Returns the path to the new file.
    """
    run_ts   = datetime.now(timezone.utc).astimezone()
    filename = f"dev_simulation_{run_ts.strftime('%Y-%m-%d_%H%M%S')}.xlsx"
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
        _apply_data_row_styles(ws_trades, row_idx, "simulated")
        _apply_number_formats(ws_trades, row_idx)

    # ── Sheet 2: All Candidates ────────────────────────────
    ws_cands = wb.create_sheet("All Candidates")
    cand_headers = [
        ("Pair Type", 14),
        ("Market A", 45), ("Ticker A", 18),
        ("Market B", 45), ("Ticker B", 18),
        ("A Deadline", 12), ("B Deadline", 12),
        ("pA (YES ask)", 13), ("pB (YES ask)", 13), ("nA (NO ask)", 13),
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
        diff    = pair.pA - pair.pB
        row_data = [
            pair.pair_type,
            _market_title(pair.market_a), pair.market_a.ticker,
            _market_title(pair.market_b), pair.market_b.ticker,
            pair.market_a.close_time.strftime("%Y-%m-%d") if pair.market_a.close_time else "",
            pair.market_b.close_time.strftime("%Y-%m-%d") if pair.market_b.close_time else "",
            round(pair.pA, 4),
            round(pair.pB, 4),
            round(pair.nA, 4),
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

        # Format price columns (shifted right by 1 due to new Pair Type column)
        for col in (8, 9, 10, 11):
            ws_cands.cell(row=row_idx, column=col).number_format = "0.00%"

    wb.save(out_path)
    logging.info("Dev simulation written: %s", out_path)
    return out_path
