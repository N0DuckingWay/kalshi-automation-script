"""Tests for reporter.py — sidecar locking, atomic save, and lock-timeout
fallback around append_to_prod_log() (BS-18), the run note (main._run_prod's
live toggles) it appends to the separator row on both paths, plus the row/Notes/candidates
sheet layout after the 2026-09 strategy change (side-neutral x/y headers, the
"[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=…]] " Notes prefix, and the "nB (NO
ask)" candidates column), and the run result main.py --result-file writes
(trade_record, report_trades, write_run_report and RunReportHandler). All
tests run offline against tmp_path; no real Kalshi API interaction.

The lock-timeout test pre-acquires the sidecar lock file from a *separate*
open() call in the test itself. This genuinely conflicts with reporter's own
_acquire_lock() even though both run in the same process: flock() locks are
scoped to the open file description, not the process, so two independent
open() calls on the same path do contend for the lock.
"""
import dataclasses
import fcntl
import json
import logging
import re
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import openpyxl
import pytest

from kalshi_betting import config, reporter
from kalshi_betting.reporter import TradeResult
from kalshi_betting.scanner import ApiMarket, CandidatePair, leg_prices, leg_sides
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
        # TS-12: the column is fee-INCLUSIVE now (total_cost_with_fees,
        # 4.85), not the contract-only total_cost of 4.75.
        assert row[13] == pytest.approx(4.85)
        assert row[14] == pytest.approx(0.25)
        assert row[15] == pytest.approx(0.05)
        assert row[16] == "executed"
        # Notes = prefix + the trader's error text, verbatim
        assert row[17] == "[time_series: YES A / NO B nB=0.4000 fees=$0.10] YES leg FoK not filled"

    def test_notes_prefix_time_series_names_sides_and_nb(self):
        row = reporter._result_to_row(make_result("1"), datetime(2026, 9, 8))
        assert row[17] == "[time_series: YES A / NO B nB=0.4000 fees=$0.10] "

    def test_notes_prefix_same_title_names_sides_without_nb(self):
        # nB is reporting-only for a same-title pair, so it is not in the prefix
        row = reporter._result_to_row(
            make_result("1", pair_type="same_title"), datetime(2026, 9, 8),
        )
        assert row[17] == "[same_title: NO A / YES B fees=$0.10] "

    def test_trade_column_headers_are_side_neutral(self):
        headers = [h for h, _ in reporter._TRADE_COLUMNS]
        assert len(headers) == 18
        assert headers[11] == "x — A leg"
        assert headers[12] == "y — B leg"
        assert headers[14] == "Profit if won ($)"
        # _apply_number_formats' hardcoded indices depend on this order
        assert headers[8:11] == ["pA (YES ask)", "pB (YES ask)", "nA (NO ask)"]
        assert headers[13] == "Total Cost incl. fees ($)"
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


def _separator_rows(path) -> list:
    """Every run-separator banner in a saved log, in order: the column-1
    values beginning "── Run:" (a banner populates column 1 only)."""
    ws = openpyxl.load_workbook(path).active
    return [row[0] for row in ws.iter_rows(min_row=2, values_only=True)
            if isinstance(row[0], str) and row[0].startswith("── Run:")]


class TestSeparatorRunNote:
    """append_to_prod_log's run_note (main._run_prod passes the run's live
    toggles) is appended to the separator banner on BOTH paths — the shared
    log and the lock-timeout fallback file — never as a column; with no note
    the banner ends at its trade count."""

    _NOTE = "settings: tier floors off | spread band 0-0.5 | k 0.8"
    _BANNER = re.compile(
        r"^── Run: \d{4}-\d{2}-\d{2} \d{2}:\d{2}  \|  Balance before: \$100\.00  →  "
        r"after: \$95\.00  \|  1 trade\(s\)$")

    def test_the_shared_log_carries_the_note(self, reporter_paths):
        log_path, _ = reporter_paths
        reporter.append_to_prod_log([make_result("1")], 100.0, 95.0, run_note=self._NOTE)
        (banner,) = _separator_rows(log_path)
        assert banner.endswith(f"1 trade(s)  |  {self._NOTE}")
        assert self._BANNER.match(banner.removesuffix(f"  |  {self._NOTE}"))

    def test_no_note_leaves_the_banner_as_it_was(self, reporter_paths):
        log_path, _ = reporter_paths
        reporter.append_to_prod_log([make_result("1")], 100.0, 95.0)
        reporter.append_to_prod_log([make_result("2")], 100.0, 95.0, run_note="")
        banners = _separator_rows(log_path)
        assert len(banners) == 2
        assert all(self._BANNER.match(b) for b in banners), banners

    def test_the_note_adds_no_column(self, reporter_paths):
        log_path, _ = reporter_paths
        reporter.append_to_prod_log([make_result("1")], 100.0, 95.0, run_note=self._NOTE)
        ws = openpyxl.load_workbook(log_path).active
        assert ws.max_column == len(reporter._TRADE_COLUMNS)
        assert _count_data_rows(log_path) == 1

    def test_the_fallback_file_carries_the_note(self, reporter_paths):
        path = reporter._write_fallback_log([make_result("1")], 100.0, 95.0, run_note=self._NOTE)
        (banner,) = _separator_rows(path)
        assert banner.endswith(f"1 trade(s)  |  {self._NOTE}")
        path = reporter._write_fallback_log([make_result("2")], 100.0, 95.0)
        (banner,) = _separator_rows(path)
        assert self._BANNER.match(banner)

    def test_a_lock_timeout_hands_the_note_to_the_fallback(self, reporter_paths, monkeypatch):
        # The path append_to_prod_log takes when another process holds the log
        _, lock_path = reporter_paths
        monkeypatch.setattr(reporter, "_LOCK_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(reporter, "_LOCK_POLL_SECONDS", 0.05)
        lock_path.touch()
        held_fh = open(lock_path, "r+")
        fcntl.flock(held_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            path = reporter.append_to_prod_log([make_result("x")], 100.0, 95.0,
                                               run_note=self._NOTE)
        finally:
            fcntl.flock(held_fh.fileno(), fcntl.LOCK_UN)
            held_fh.close()
        (banner,) = _separator_rows(path)
        assert banner.endswith(f"  |  {self._NOTE}")


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
        assert ws.cell(row=3, column=18).value == "[time_series: YES A / NO B nB=0.4000 fees=$0.10] "


class _FrozenDatetime:
    """datetime stand-in whose now() always returns one fixed instant.

    Installed over reporter's module-global `datetime` (reporter.py imports it
    at module scope), so two successive writes render the IDENTICAL filename
    timestamp. That forces the TS-18 collision deterministically instead of
    waiting for two real writes to land in one microsecond — which is only a
    workable test because the fix is exclusive creation, not merely a finer
    timestamp. Under a microseconds-only fix this fixture would make even
    correct code emit a single filename.
    """

    FIXED = datetime(2026, 9, 13, 1, 2, 3, 456789, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.FIXED


class TestOutputFilenameCollisions:
    """TS-18: a second output written in the same instant must never overwrite
    the first. The fallback log is the sharpest case — it exists precisely to
    guarantee 'this run's rows are never silently dropped' (reporter.py module
    docstring), and before this fix it was the one write path in the module with
    neither a lock, an atomic rename, nor a unique name."""

    def test_fallback_filename_carries_microseconds(self, reporter_paths):
        # Six extra zero-padded digits keep two near-simultaneous fallbacks off
        # the collision path at all. %f is always six digits, so this pins the
        # format with no clock control.
        path = reporter._write_fallback_log([make_result("1")], 100.0, 95.0)
        assert re.fullmatch(r"trade_log_\d{4}-\d{2}-\d{2}_\d{6}_\d{6}\.xlsx", path.name)

    def test_two_fallback_writes_at_one_instant_both_survive(
        self, reporter_paths, monkeypatch
    ):
        # The load-bearing test: distinct paths alone would only restate that a
        # suffix was typed. Each file must still hold ITS OWN row — that is what
        # proves no run's rows were dropped.
        monkeypatch.setattr(reporter, "datetime", _FrozenDatetime)

        first = reporter._write_fallback_log([make_result("A")], 100.0, 95.0)
        second = reporter._write_fallback_log([make_result("B")], 95.0, 90.0)

        assert first != second
        assert second.name.endswith("-1.xlsx")
        assert first.exists()
        assert second.exists()
        assert _count_data_rows(first) == 1
        assert _count_data_rows(second) == 1

    def test_two_dev_simulations_at_one_instant_both_survive(
        self, reporter_paths, monkeypatch
    ):
        monkeypatch.setattr(reporter, "datetime", _FrozenDatetime)

        first = reporter.write_dev_simulation(
            [make_result("A")], [], balance_cents=100_000
        )
        second = reporter.write_dev_simulation(
            [make_result("B")], [], balance_cents=100_000
        )

        assert first != second
        assert second.name.endswith("-1.xlsx")
        assert first.exists()
        assert second.exists()


# ─────────────────────────────────────────────
# Run result (main.py --result-file)
# ─────────────────────────────────────────────

def _strict_json(path) -> dict:
    """
    Parse a run result file as strict JSON.

    Args:
        path (Path): The result file.

    Returns:
        dict: The parsed record.

    Raises:
        AssertionError: If the file holds NaN or an infinity.
    """
    def refuse(token):
        """
        Fail on a constant strict JSON has not.

        Args:
            token (str): "NaN", "Infinity" or "-Infinity".

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(f"not strict JSON: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _run_report(**changes) -> reporter.RunReport:
    """
    Build a filled-in RunReport for write_run_report.

    Args:
        **changes: Fields to set instead of the defaults here.

    Returns:
        reporter.RunReport: A report of a real-money run with one executed pair.
    """
    fields = {
        "dry_run": False, "started_at": datetime(2026, 9, 28, 16, 0, 5, tzinfo=UTC),
        "settings": "tier floors off | spread band 0-0.5",
        "defaults": "live_defaults.json, saved …",
        "message": "Submitted 1 of 1 order pair(s) successfully.",
        "balance_before": 100.0, "balance_after": 99.5, "portfolio_value_before": 180.25,
        "trades": [reporter.trade_record(make_result("R"))],
        "warnings": ["WARNING: Running in PRODUCTION mode — real money will be used!"],
    }
    fields.update(changes)
    return reporter.RunReport(**fields)


class TestTradeRecord:
    """trade_record describes a pair in the trade log's terms: market A then
    B, the side each leg bought (scanner.leg_sides) and the price it was sized
    at (scanner.leg_prices)."""

    @pytest.mark.parametrize("pair_type, sides, prices", [
        ("time_series", ("yes", "no"), (0.30, 0.40)),
        ("same_title", ("no", "yes"), (0.70, 0.60)),
    ])
    def test_each_leg_is_described_by_the_pair_type(self, pair_type, sides, prices):
        result = make_result("X", status="rolled_back", pair_type=pair_type,
                             error="YES leg FoK not filled")
        pair = result.spec.pair
        record = reporter.trade_record(result)
        # The one source of truth for sides and prices, and what they give here
        assert (record.a.side, record.b.side) == leg_sides(pair_type) == sides
        assert (record.a.price, record.b.price) == tuple(
            round(p, 4) for p in leg_prices(pair)) == prices
        assert record == reporter.TradeRecord(
            status="rolled_back", error="YES leg FoK not filled", pair_type=pair_type,
            title="test pair",
            a=reporter.LegRecord("TICK-A-X", "Will TICK-A-X happen?", sides[0], 5, prices[0]),
            b=reporter.LegRecord("TICK-B-X", "Will TICK-B-X happen?", sides[1], 5, prices[1]),
            cost_with_fees=4.85, profit_if_won=0.25)

    def test_a_figure_that_is_not_finite_is_recorded_as_not_known(self, tmp_path):
        result = make_result("N")
        result.spec.total_cost_with_fees = float("nan")
        result.spec.min_payoff = float("inf")
        record = reporter.trade_record(result)
        assert record.cost_with_fees is None and record.profit_if_won is None
        # So the result can still be written as strict JSON
        path = tmp_path / "result.json"
        reporter.write_run_report(path, _run_report(trades=[record]), 0)
        assert _strict_json(path)["trades"][0]["cost_with_fees"] is None

    def test_records_are_frozen(self):
        record = reporter.trade_record(make_result("F"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.status = "executed"
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.a.count = 0


class TestReportTrades:
    """report_trades never raises: a pair trade_record cannot describe is
    logged as an ERROR and kept, in its place, as what could be read of it."""

    def test_a_pair_that_cannot_be_described_is_logged_and_kept_in_order(self, caplog):
        first = make_result("1")
        last = make_result("3", status="failed", error="NO leg FoK not filled")
        # leg_prices reads nB directly, so a pair without it cannot be priced
        broken_pair = SimpleNamespace(
            pair_type="time_series", canonical_title="broken pair", pA=0.3,
            market_a=SimpleNamespace(ticker="TA"), market_b=SimpleNamespace(ticker="TB"))
        broken = TradeResult(spec=SimpleNamespace(pair=broken_pair, x=3, y=4),
                             status="manual_review", error="position lookup failed")
        with caplog.at_level(logging.ERROR):
            records = reporter.report_trades([first, broken, last])
        assert [r.status for r in records] == ["executed", "manual_review", "failed"]
        assert records[0] == reporter.trade_record(first)
        assert records[2] == reporter.trade_record(last)
        assert records[1] == reporter.TradeRecord(
            status="manual_review", error="position lookup failed", pair_type="time_series",
            title="broken pair", a=reporter.LegRecord("TA", None, None, 3, None),
            b=reporter.LegRecord("TB", None, None, 4, None),
            cost_with_fees=None, profit_if_won=None)
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert errors[0].getMessage().startswith("Could not describe a pair for the run result:")

    def test_only_plain_values_are_kept_from_a_result_it_cannot_read(self, caplog):
        # A result that is not a TradeResult at all, and one whose fields are
        # mocks: neither raises, and nothing kept would stop the JSON
        # A pair with no nB, so leg_prices raises; every other field is a mock
        pair = MagicMock(spec=["pair_type", "pA", "canonical_title", "market_a", "market_b"])
        pair.pair_type = "time_series"
        mocked = TradeResult(spec=MagicMock(pair=pair), status="executed")
        with caplog.at_level(logging.ERROR):
            records = reporter.report_trades([object(), mocked])
        assert records[0] == reporter.TradeRecord(
            status="unknown", error=None, pair_type=None, title=None, a=None, b=None,
            cost_with_fees=None, profit_if_won=None)
        assert records[1].status == "executed" and records[1].pair_type == "time_series"
        # A mock's ticker and counts are not text or ints, so no leg is kept
        assert records[1].title is None and records[1].a is None and records[1].b is None
        json.dumps([dataclasses.asdict(r) for r in records], allow_nan=False)
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 2

    def test_no_results_give_no_records(self):
        assert reporter.report_trades([]) == []

    def test_a_result_whose_fields_raise_when_read_keeps_its_place(self, caplog):
        # getattr's default covers only a missing field; a field that raises
        # anything else when read must not escape either, or the trade log
        # after it would never be written
        class Unreadable:
            """A result whose every field raises when read."""

            def __getattr__(self, name):
                """
                Fail on every field, the way a broken property would.

                Args:
                    name (str): The field read.

                Raises:
                    RuntimeError: Always.
                """
                raise RuntimeError(f"cannot read {name}")

        first = make_result("1")
        with caplog.at_level(logging.ERROR):
            records = reporter.report_trades([first, Unreadable()])
        assert records[0] == reporter.trade_record(first)
        assert records[1] == reporter.TradeRecord(
            status="unknown", error=None, pair_type=None, title=None, a=None, b=None,
            cost_with_fees=None, profit_if_won=None)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors == ["Could not describe a pair for the run result: cannot read spec"]


class TestWriteRunReport:
    """write_run_report writes the whole record as strict JSON, replacing the
    file at once, and never raises: a failure is an ERROR and leaves no
    staging file."""

    def test_the_record_is_strict_json_with_every_field(self, tmp_path):
        path = tmp_path / "result.json"
        report = _run_report()
        reporter.write_run_report(path, report, 20)
        record = _strict_json(path)
        assert record == {
            "format": config.LIVE_RUN_RESULT_FORMAT, "mode": "prod", "dry_run": False,
            "started_at": "2026-09-28T16:00:05Z", "finished_at": record["finished_at"],
            "exit_code": 20, "settings": report.settings, "defaults": report.defaults,
            "message": report.message, "balance_before": 100.0, "balance_after": 99.5,
            "portfolio_value_before": 180.25, "submission_started": False,
            "trades": [dataclasses.asdict(t) for t in report.trades],
            "warnings": report.warnings, "warnings_dropped": 0, "error": None,
        }
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["finished_at"])
        assert record["trades"][0]["a"]["ticker"] == "TICK-A-R"
        # Only the result is left in its folder
        assert list(tmp_path.iterdir()) == [path]

    def test_a_run_that_never_read_the_balance_records_no_portfolio_value(self, tmp_path):
        # The field defaults to None, written as null, never as 0: a run that
        # stopped before its balance read sized on nothing
        path = tmp_path / "result.json"
        report = reporter.RunReport(dry_run=True,
                                    started_at=datetime(2026, 9, 28, 16, 0, 5, tzinfo=UTC))
        reporter.write_run_report(path, report, 0)
        record = _strict_json(path)
        assert record["portfolio_value_before"] is None
        assert record["balance_before"] is None

    def test_an_exception_is_written_as_no_exit_code(self, tmp_path):
        path = tmp_path / "result.json"
        reporter.write_run_report(path, _run_report(error="RuntimeError: boom"), None)
        record = _strict_json(path)
        assert record["exit_code"] is None and record["error"] == "RuntimeError: boom"

    def test_an_old_file_is_replaced_whole(self, tmp_path):
        path = tmp_path / "result.json"
        path.write_text("x" * 100_000, encoding="utf-8")
        reporter.write_run_report(path, _run_report(trades=[]), 0)
        assert _strict_json(path)["trades"] == []
        assert list(tmp_path.iterdir()) == [path]

    @pytest.mark.parametrize("folder", ["missing", "read-only"])
    def test_an_unwritable_path_is_an_error_not_an_exception(self, tmp_path, caplog, folder):
        where = tmp_path / folder
        if folder == "read-only":
            where.mkdir()
            where.chmod(0o500)
        path = where / "result.json"
        try:
            with caplog.at_level(logging.ERROR):
                reporter.write_run_report(path, _run_report(), 0)
        finally:
            if where.exists():
                where.chmod(0o700)
        assert not path.exists()
        assert [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")] == []
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1 and errors[0].startswith(
            f"Could not write the run result to {path}")

    @pytest.mark.parametrize("value", [Decimal("1.50"), float("nan")], ids=["decimal", "nan"])
    def test_a_value_strict_json_cannot_hold_keeps_the_old_file(self, tmp_path, caplog, value):
        path = tmp_path / "result.json"
        path.write_text('{"old": true}', encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            reporter.write_run_report(path, _run_report(balance_before=value), 0)
        assert path.read_text(encoding="utf-8") == '{"old": true}'
        assert list(tmp_path.iterdir()) == [path]
        assert any(r.levelno == logging.ERROR and "Could not write the run result" in
                   r.getMessage() for r in caplog.records)

    def test_a_rename_that_fails_removes_its_staging_file(self, tmp_path, caplog):
        # A folder where the file should be: the staging file is written, the
        # rename over the folder fails, and the staging file is removed
        path = tmp_path / "result.json"
        path.mkdir()
        with caplog.at_level(logging.ERROR):
            reporter.write_run_report(path, _run_report(), 0)
        assert path.is_dir() and list(tmp_path.iterdir()) == [path]
        assert any("Could not write the run result" in r.getMessage() for r in caplog.records)

    def test_a_report_that_cannot_be_read_is_an_error_not_an_exception(self, tmp_path, caplog):
        path = tmp_path / "result.json"
        with caplog.at_level(logging.ERROR):
            reporter.write_run_report(path, _run_report(started_at="not a datetime",
                                                        trades=[object()]), 0)
        assert not path.exists()
        assert any("Could not write the run result" in r.getMessage() for r in caplog.records)


class TestRunReportHandler:
    """RunReportHandler copies each WARNING-or-worse line into the report,
    first line only: every ERROR and CRITICAL line whole, and the first
    config.RUN_REPORT_MAX_WARNINGS WARNING lines, each cut at
    config.RUN_REPORT_LINE_MAX_CHARS with "…" as its last character; the
    WARNING lines left out are counted."""

    @staticmethod
    def _log(emit) -> reporter.RunReport:
        """
        Attach a handler to a logger of its own, log through it, and detach it.

        Args:
            emit (Callable[[logging.Logger], None]): Logs through the logger.

        Returns:
            reporter.RunReport: The report the handler filled.
        """
        report = reporter.RunReport(dry_run=True, started_at=datetime.now(UTC))
        logger = logging.getLogger("test_reporter.run_report_handler")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = reporter.RunReportHandler(report)
        logger.addHandler(handler)
        try:
            emit(logger)
        finally:
            logger.removeHandler(handler)
        return report

    def test_it_keeps_warning_and_worse_only(self):
        def emit(logger):
            """
            Log one line at each level.

            Args:
                logger (logging.Logger): The logger to log through.
            """
            logger.debug("debug")
            logger.info("info")
            logger.warning("warned %d", 1)
            logger.error("errored")
            logger.critical("critical")

        report = self._log(emit)
        assert report.warnings == ["WARNING: warned 1", "ERROR: errored", "CRITICAL: critical"]
        assert report.warnings_dropped == 0

    def test_a_warning_is_cut_to_length_and_marked(self):
        limit = config.RUN_REPORT_LINE_MAX_CHARS
        report = self._log(lambda logger: (
            logger.warning("x" * (limit + 100) + "\nthe second line"),
            logger.warning("y" * limit),
            logger.error(""),
            logger.error("short\nsecond")))
        cut = "x" * (limit - 1) + "…"
        assert len(cut) == limit
        # A line of exactly the limit is not cut
        assert report.warnings == ["WARNING: " + cut, "WARNING: " + "y" * limit,
                                   "ERROR: ", "ERROR: short"]

    @pytest.mark.parametrize("level", [logging.ERROR, logging.CRITICAL])
    def test_an_error_or_critical_line_is_kept_whole(self, level):
        long_line = "z" * (config.RUN_REPORT_LINE_MAX_CHARS * 3) + " END"
        report = self._log(lambda logger: logger.log(level, "%s\nsecond line", long_line))
        assert report.warnings == [f"{logging.getLevelName(level)}: {long_line}"]

    def test_it_stops_at_the_cap_and_counts_the_rest(self):
        cap = config.RUN_REPORT_MAX_WARNINGS
        report = self._log(lambda logger: [logger.warning("line %d", i) for i in range(cap + 3)])
        assert report.warnings == [f"WARNING: line {i}" for i in range(cap)]
        assert report.warnings_dropped == 3

    def test_the_cap_counts_warnings_only_and_never_drops_an_error_or_critical(self):
        cap = config.RUN_REPORT_MAX_WARNINGS

        def emit(logger):
            """
            Log two ERRORs among the WARNINGs, then a burst of WARNINGs past
            the cap, then a CRITICAL and one more WARNING.

            Args:
                logger (logging.Logger): The logger to log through.
            """
            logger.error("early error")
            for i in range(cap + 10):
                logger.warning("retry %d", i)
                if i == 5:
                    logger.error("error among the retries")
            logger.critical("late critical")
            logger.warning("late warning")

        report = self._log(emit)
        assert [w for w in report.warnings if not w.startswith("WARNING: ")] == [
            "ERROR: early error", "ERROR: error among the retries", "CRITICAL: late critical"]
        # Oldest first, the WARNINGs capped at their own count
        assert report.warnings[-1] == "CRITICAL: late critical"
        assert sum(w.startswith("WARNING: ") for w in report.warnings) == cap
        assert report.warnings_dropped == 11

    def test_a_real_v2_disproof_critical_survives_a_burst_of_retry_warnings(self, monkeypatch):
        # The disproof CRITICAL names every earlier pair that went ahead on the
        # same mapping; those pairs are "executed" in the result, so this line
        # is the only record of them. It is far longer than a WARNING's cut.
        from kalshi_betting import trader

        spec = make_result("D").spec
        no_leg, _ = trader._ordered_legs(spec)
        unchecked = [f"KXSENATEREC-26MAY-LONGTICKER{i:02d}" for i in range(12)]
        monkeypatch.setattr(trader, "_V2_UNCHECKED_NO_LEGS", list(unchecked))
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", False)
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", False)
        # The NO buy moved the position the wrong way: a disproof
        monkeypatch.setattr(trader, "_position_count_once",
                            lambda client, ticker: float(no_leg.count))
        report = reporter.RunReport(dry_run=False, started_at=datetime.now(UTC))
        handler = reporter.RunReportHandler(report)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            for i in range(60):
                logging.warning("HTTP 429 on attempt %d, retrying", i)
            outcome = trader._confirm_v2_no_mapping(MagicMock(), spec, no_leg, 0.0)
        finally:
            root.removeHandler(handler)
        assert outcome.status == "manual_review"
        critical = [w for w in report.warnings if w.startswith("CRITICAL: ")]
        assert len(critical) == 1
        assert len(critical[0]) > config.RUN_REPORT_LINE_MAX_CHARS
        assert "flatten this position by hand in the Kalshi UI" in critical[0]
        assert all(ticker in critical[0] for ticker in unchecked)
        assert critical[0].endswith(" too.")
        assert report.warnings_dropped == 60 - config.RUN_REPORT_MAX_WARNINGS

    def test_a_line_that_cannot_be_formatted_does_not_raise(self, monkeypatch):
        # logging's own handleError reports it (silenced here); the run goes on
        monkeypatch.setattr(logging, "raiseExceptions", False)
        report = self._log(lambda logger: logger.warning("%d", "not a number"))
        assert report.warnings == [] and report.warnings_dropped == 0
