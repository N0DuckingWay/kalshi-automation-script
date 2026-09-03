"""Tests for reporter.py — sidecar locking, atomic save, and lock-timeout
fallback around append_to_prod_log() (BS-18). All tests run offline against
tmp_path; no real Kalshi API interaction.

The lock-timeout test pre-acquires the sidecar lock file from a *separate*
open() call in the test itself. This genuinely conflicts with reporter's own
_acquire_lock() even though both run in the same process: flock() locks are
scoped to the open file description, not the process, so two independent
open() calls on the same path do contend for the lock.
"""
import fcntl
import logging

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


def make_result(ticker_suffix: str, status: str = "executed") -> TradeResult:
    """Factory for a valid TradeResult with a real CandidatePair/TradeSpec,
    matching the fields reporter._result_to_row reads off spec.pair and spec."""
    pair = CandidatePair(
        market_a=make_market(f"TICK-A-{ticker_suffix}"),
        market_b=make_market(f"TICK-B-{ticker_suffix}"),
        pA=0.40,
        pB=0.35,
        nA=0.60,
        tradeable=True,
        canonical_title="test pair",
        pair_type="time_series",
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
    return TradeResult(spec=spec, status=status, error=None)


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
