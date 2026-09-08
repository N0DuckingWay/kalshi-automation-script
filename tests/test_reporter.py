"""Tests for reporter.py — sidecar locking, atomic save, and lock-timeout
fallback around append_to_prod_log() (BS-18), plus the row/Notes/candidates
sheet layout after the 2026-09 strategy change (side-neutral x/y headers, the
"[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=…]] " Notes prefix, and the "nB (NO
ask)" candidates column). All tests run offline against tmp_path; no real
Kalshi API interaction.

The lock-timeout test pre-acquires the sidecar lock file from a *separate*
open() call in the test itself. This genuinely conflicts with reporter's own
_acquire_lock() even though both run in the same process: flock() locks are
scoped to the open file description, not the process, so two independent
open() calls on the same path do contend for the lock.
"""
import fcntl
import logging
from datetime import datetime

import openpyxl
import pytest

from kalshi_betting import reporter
from kalshi_betting.reporter import TradeResult
from kalshi_betting.scanner import ApiMarket, CandidatePair
from kalshi_betting.strategy import TradeSpec

_STATUS_COL_INDEX = 16  # 0-based index of the "Status" column in a data row tuple


def make_market(ticker: str) -> ApiMarket:
    """Build a minimal real ApiMarket (not a MagicMock) so display_title() and
    the close_time formatting in _result_to_row behave exactly as in prod."""
    return ApiMarket(
        ticker=ticker,
        event_ticker=f"EVT-{ticker}",
        title=f"Will {ticker} happen?",
        subtitle="",
        status="active",
        close_time=None,
    )


def make_result(
    ticker_suffix: str,
    status: str = "executed",
    pair_type: str = "time_series",
    error: str | None = None,
) -> TradeResult:
    """Factory for a valid TradeResult with a real CandidatePair/TradeSpec,
    matching the fields reporter._result_to_row reads off spec.pair and spec.

    The default pair is a coherent time-series pair under the 2026-09
    direction: the LATER contract (market_b) is priced above the earlier
    (pA=0.30 → pB=0.60), the legs bought are YES on A at pA=0.30 and NO on B at
    nB=0.40, and nA=0.70 is A's reporting-only NO ask. `pair_type` can be
    switched to "same_title" to exercise the other Notes prefix — the prices
    are then the same-title legs nA/pB and nB is reporting-only."""
    pair = CandidatePair(
        market_a=make_market(f"TICK-A-{ticker_suffix}"),
        market_b=make_market(f"TICK-B-{ticker_suffix}"),
        pA=0.30,
        pB=0.60,
        nA=0.70,
        tradeable=True,
        canonical_title="test pair",
        pair_type=pair_type,
        nB=0.40,
    )
    spec = TradeSpec(
        pair=pair,
        x=5,
        y=5,
        total_cost=4.75,
        total_cost_with_fees=4.85,
        min_payoff=0.25,
        profit_ratio=0.05,
        days_to_close=10,
        monthly_profit_ratio=0.15,
        kelly_p=0.6,
        kelly_fraction=0.1,
    )
    return TradeResult(spec=spec, status=status, error=error)


def _count_data_rows(path) -> int:
    """Count actual trade data rows in a saved log, distinguishing them from
    the header row and the run-separator banner rows (which only populate
    column 1, leaving the Status column empty)."""
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    count = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[_STATUS_COL_INDEX]:
            count += 1
    return count


@pytest.fixture
def reporter_paths(tmp_path, monkeypatch):
    """Point reporter's module-level path constants at tmp_path so tests never
    touch the real repo-root trade_log.xlsx."""
    log_path = tmp_path / "trade_log.xlsx"
    lock_path = tmp_path / "trade_log.xlsx.lock"
    monkeypatch.setattr(reporter, "PROD_LOG_PATH", log_path)
    monkeypatch.setattr(reporter, "_LOCK_PATH", lock_path)
    monkeypatch.setattr(reporter, "PROJECT_ROOT", tmp_path)
    return log_path, lock_path


class TestAppendToProdLog:
    def test_two_sequential_appends_preserve_all_rows(self, reporter_paths):
        log_path, _ = reporter_paths
        run1 = [make_result("1"), make_result("2")]
        run2 = [make_result("3")]

        reporter.append_to_prod_log(run1, balance_before=100.0, balance_after=95.0)
        reporter.append_to_prod_log(run2, balance_before=95.0, balance_after=90.0)

        assert _count_data_rows(log_path) == 3

    def test_no_tmp_residue_after_save(self, reporter_paths):
        log_path, _ = reporter_paths
        reporter.append_to_prod_log([make_result("1")], balance_before=100.0, balance_after=95.0)

        tmp_file = log_path.with_name(log_path.name + ".tmp")
        assert not tmp_file.exists()
        assert log_path.exists()

    def test_returns_prod_log_path_on_success(self, reporter_paths):
        log_path, _ = reporter_paths
        result_path = reporter.append_to_prod_log([make_result("1")], 100.0, 95.0)
        assert result_path == log_path

    def test_lock_timeout_writes_fallback_and_warns(self, reporter_paths, monkeypatch, caplog, tmp_path):
        log_path, lock_path = reporter_paths
        # Small deadline/poll so the test doesn't actually wait ~30s.
        monkeypatch.setattr(reporter, "_LOCK_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(reporter, "_LOCK_POLL_SECONDS", 0.05)

        lock_path.touch()
        held_fh = open(lock_path, "r+")
        fcntl.flock(held_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with caplog.at_level(logging.WARNING):
                result_path = reporter.append_to_prod_log([make_result("x")], 50.0, 45.0)
        finally:
            fcntl.flock(held_fh.fileno(), fcntl.LOCK_UN)
            held_fh.close()

        # Never touches the shared log — a fresh timestamped file instead.
        assert result_path != log_path
        assert not log_path.exists()
        assert result_path.parent == tmp_path
        assert result_path.name.startswith("trade_log_")
        assert result_path.exists()
        assert _count_data_rows(result_path) == 1

        assert any(
            rec.levelno == logging.WARNING and "Could not acquire lock" in rec.message
            for rec in caplog.records
        )

        # No .tmp residue from the fallback path either.
        assert not result_path.with_name(result_path.name + ".tmp").exists()

    def test_lock_open_oserror_falls_back_without_raising(
        self, reporter_paths, monkeypatch, caplog, tmp_path
    ):
        # A filesystem error while creating/opening the sidecar (read-only
        # mount, permissions, ENOSPC) must not kill the save — it degrades to
        # the lock-free standalone fallback file.
        log_path, _ = reporter_paths

        def boom(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        # reporter's module globals are consulted before builtins, so this
        # replaces only reporter's own open() call in _acquire_lock — openpyxl
        # still saves normally through its own module's open().
        monkeypatch.setattr(reporter, "open", boom, raising=False)

        with caplog.at_level(logging.WARNING):
            result_path = reporter.append_to_prod_log([make_result("x")], 50.0, 45.0)

        assert result_path != log_path
        assert not log_path.exists()
        assert result_path.parent == tmp_path
        assert result_path.name.startswith("trade_log_")
        assert _count_data_rows(result_path) == 1

        assert any(
            rec.levelno == logging.WARNING and "Could not open lock file" in rec.message
            for rec in caplog.records
        )


class TestResultToRow:
    """The 18-column row contract and the Notes prefix that names each
    market's side (from scanner.leg_sides) — the only place a workbook whose
    header row predates the side-neutral x/y headers records which side each
    count bought, and the only place a time-series row records nB."""

    def test_row_has_18_values_in_column_order(self):
        result = make_result("1", error="YES leg FoK not filled")
        run_ts = datetime(2026, 9, 8, 9, 0, 0)

        row = reporter._result_to_row(result, run_ts)

        assert len(row) == 18
        assert len(row) == len(reporter._TRADE_COLUMNS)
        assert row[0] == "2026-09-08"
        assert row[1] == "09:00:00"
        assert row[3] == "TICK-A-1"
        assert row[5] == "TICK-B-1"
        # pA, pB, nA in MARKET order; x/y are market A's and market B's counts
        assert row[8] == pytest.approx(0.30)
        assert row[9] == pytest.approx(0.60)
        assert row[10] == pytest.approx(0.70)
        assert row[11] == 5
        assert row[12] == 5
        assert row[13] == pytest.approx(4.75)
        assert row[14] == pytest.approx(0.25)
        assert row[15] == pytest.approx(0.05)
        assert row[16] == "executed"
        # Notes = prefix + the trader's error text, verbatim
        assert row[17] == "[time_series: YES A / NO B nB=0.4000] YES leg FoK not filled"

    def test_notes_prefix_time_series_names_sides_and_nb(self):
        row = reporter._result_to_row(make_result("1"), datetime(2026, 9, 8))
        assert row[17] == "[time_series: YES A / NO B nB=0.4000] "

    def test_notes_prefix_same_title_names_sides_without_nb(self):
        # nB is reporting-only for a same-title pair, so it is not in the prefix
        row = reporter._result_to_row(
            make_result("1", pair_type="same_title"), datetime(2026, 9, 8),
        )
        assert row[17] == "[same_title: NO A / YES B] "

    def test_trade_column_headers_are_side_neutral(self):
        headers = [h for h, _ in reporter._TRADE_COLUMNS]
        assert len(headers) == 18
        assert headers[11] == "x — A leg"
        assert headers[12] == "y — B leg"
        assert headers[14] == "Profit if won ($)"
        # _apply_number_formats' hardcoded indices depend on this order
        assert headers[8:11] == ["pA (YES ask)", "pB (YES ask)", "nA (NO ask)"]
        assert headers[13] == "Total Cost ($)"
        assert headers[15] == "Profit Ratio (%)"


class TestExistingWorkbookHeaderRow:
    def test_second_append_leaves_original_header_row_untouched(self, reporter_paths):
        # _append_locked writes headers only when it CREATES the workbook, so a
        # shared log written under an older header vocabulary keeps it — the
        # per-row Notes prefix is what disambiguates the new rows.
        log_path, _ = reporter_paths
        reporter.append_to_prod_log([make_result("1")], balance_before=100.0, balance_after=95.0)

        legacy_headers = [f"Legacy header {i}" for i in range(1, 19)]
        wb = openpyxl.load_workbook(log_path)
        ws = wb.active
        for col_idx, header in enumerate(legacy_headers, start=1):
            ws.cell(row=1, column=col_idx, value=header)
        wb.save(log_path)

        reporter.append_to_prod_log([make_result("2")], balance_before=95.0, balance_after=90.0)

        ws = openpyxl.load_workbook(log_path).active
        assert [ws.cell(row=1, column=c).value for c in range(1, 19)] == legacy_headers
        assert _count_data_rows(log_path) == 2

    def test_fallback_workbook_gets_the_new_headers(self, reporter_paths):
        _, lock_path = reporter_paths
        # A fresh workbook — the fallback path always creates one — carries
        # the current header vocabulary.
        path = reporter._write_fallback_log([make_result("1")], 100.0, 95.0)
        ws = openpyxl.load_workbook(path).active
        headers = [ws.cell(row=1, column=c).value for c in range(1, 19)]
        assert headers == [h for h, _ in reporter._TRADE_COLUMNS]
        assert headers[11] == "x — A leg"


class TestWriteDevSimulationCandidatesSheet:
    def test_candidates_sheet_has_nb_column_and_percent_formats(self, reporter_paths):
        ts_result = make_result("1")
        st_result = make_result("2", pair_type="same_title")
        path = reporter.write_dev_simulation(
            [ts_result, st_result],
            [ts_result.spec.pair, st_result.spec.pair],
            balance_cents=100_000,
        )

        wb = openpyxl.load_workbook(path)
        ws = wb["All Candidates"]
        # 1-based worksheet columns throughout, matching _apply_number_formats
        header = lambda col: ws.cell(row=1, column=col).value  # noqa: E731
        assert ws.max_column == 13
        assert header(10) == "nA (NO ask)"
        assert header(11) == "nB (NO ask)"
        assert header(12) == "Price Diff"
        assert header(13) == "Tradeable?"

        # Row 2 is the time-series candidate, row 3 the same-title one
        assert ws.cell(row=2, column=1).value == "time_series"
        assert ws.cell(row=2, column=11).value == pytest.approx(0.40)
        # Price Diff is the gap the finder tested: pB − pA for time-series...
        assert ws.cell(row=2, column=12).value == pytest.approx(0.30)
        # ...and pA − pB for same-title (negative here because the fixture's
        # prices are a time-series shape — the sign is what's being pinned)
        assert ws.cell(row=3, column=1).value == "same_title"
        assert ws.cell(row=3, column=11).value == pytest.approx(0.40)
        assert ws.cell(row=3, column=12).value == pytest.approx(-0.30)

        # Percent formats on the five price columns pA, pB, nA, nB, Price Diff
        for col in (8, 9, 10, 11, 12):
            assert ws.cell(row=2, column=col).number_format == "0.00%"
        assert ws.cell(row=2, column=13).number_format != "0.00%"

    def test_simulated_trades_sheet_rows_carry_notes_prefix(self, reporter_paths):
        result = make_result("1")
        path = reporter.write_dev_simulation([result], [result.spec.pair], balance_cents=100_000)
        ws = openpyxl.load_workbook(path)["Simulated Trades"]
        # Row 1 headers, row 2 the merged summary banner, row 3 the trade
        assert ws.cell(row=3, column=17).value == "executed"
        assert ws.cell(row=3, column=18).value == "[time_series: YES A / NO B nB=0.4000] "
