"""Tests for main.py — the live-pipeline orchestrator's pure helpers
(_truncate, _format_deadline, _dedup_pairs, _compute_trade_specs,
_no_pairs_msg) and its logging setup (_setup_logging). All Kalshi API
interaction is mocked per project policy (tests must run offline).

This file is intentionally minimal and extensible — a later commit adds
end-to-end replays of _run_dev/_run_prod on top of these helper tests."""
import logging
from types import SimpleNamespace

from kalshi_betting import main
from kalshi_betting.config import (
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    SAME_TITLE_MIN_PRICE_DIFF,
)


def make_pair(ticker_a: str, ticker_b: str, pair_type: str = "time_series"):
    """Minimal stand-in for a CandidatePair — only the attributes
    _dedup_pairs actually reads (market_a.ticker, market_b.ticker)."""
    return SimpleNamespace(
        market_a=SimpleNamespace(ticker=ticker_a),
        market_b=SimpleNamespace(ticker=ticker_b),
        pair_type=pair_type,
    )


class TestTruncate:
    def test_truncate_boundary(self):
        # Exactly n chars: no truncation, no ellipsis appended
        text_at_boundary = "x" * 40
        assert main._truncate(text_at_boundary) == text_at_boundary

        # One char over: truncated to n chars plus the ellipsis marker
        text_over_boundary = "x" * 41
        result = main._truncate(text_over_boundary)
        assert result == "x" * 40 + "…"
        assert len(result) == 41


class TestFormatDeadline:
    def test_format_deadline_none(self):
        assert main._format_deadline(None) == "?"


class TestDedupPairs:
    def test_dedup_pairs_prefers_same_title(self):
        # Same ticker pair detected by both scanners — the same-title (primary)
        # entry must win, and the time-series (secondary) duplicate must be dropped.
        same_title = make_pair("TICK-A", "TICK-B", pair_type="same_title")
        time_series_dup = make_pair("TICK-A", "TICK-B", pair_type="time_series")

        result = main._dedup_pairs([same_title], [time_series_dup])

        assert result == [same_title]

    def test_dedup_pairs_preserves_order_and_appends_unique_secondary(self):
        primary_1 = make_pair("A1", "A2", pair_type="same_title")
        primary_2 = make_pair("B1", "B2", pair_type="same_title")
        # Duplicate of primary_1's ticker pair — must be dropped, order-independent of tickers
        secondary_dup = make_pair("A2", "A1", pair_type="time_series")
        # Genuinely unique pair — must be appended after all primary entries
        secondary_unique = make_pair("C1", "C2", pair_type="time_series")

        result = main._dedup_pairs(
            [primary_1, primary_2], [secondary_dup, secondary_unique],
        )

        assert result == [primary_1, primary_2, secondary_unique]


class TestComputeTradeSpecs:
    def test_compute_trade_specs_excludes_none(self, monkeypatch):
        pair_ok = make_pair("A1", "A2")
        pair_none = make_pair("B1", "B2")

        def fake_compute_trade(pair, balance_cents):
            # Only pair_ok produces a spec — pair_none has no edge (returns None)
            if pair is pair_ok:
                return SimpleNamespace(pair=pair)
            return None

        monkeypatch.setattr(main, "compute_trade", fake_compute_trade)

        specs = main._compute_trade_specs([pair_ok, pair_none], balance_cents=100_000)

        assert list(specs.keys()) == [id(pair_ok)]
        assert specs[id(pair_ok)].pair is pair_ok


class TestNoPairsMsg:
    def test_no_pairs_log_lines_format_thresholds_from_config(self):
        prod_msg = main._no_pairs_msg()
        dev_msg = main._no_pairs_msg(sandbox=True)

        for msg in (prod_msg, dev_msg):
            assert f"{MIN_PRICE_DIFF_SHORT_GAP:.0%}" in msg
            assert f"{MIN_PRICE_DIFF_LONG_GAP:.0%}" in msg
            assert f"{SAME_TITLE_MIN_PRICE_DIFF:.0%}" in msg

        assert "sandbox" in dev_msg
        assert "sandbox" not in prod_msg


class TestSetupLogging:
    def test_setup_logging_has_console_and_file_handlers(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            log_path = tmp_path / "x.log"
            main._setup_logging(log_path)

            handler_types = [type(h) for h in root.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.FileHandler in handler_types
            # Exactly one of each — basicConfig should not have added extras
            assert sum(1 for h in root.handlers if type(h) is logging.StreamHandler) == 1
            assert sum(1 for h in root.handlers if isinstance(h, logging.FileHandler)) == 1

            # delay=True: the file must not be created until a record is emitted
            assert not log_path.exists()
            logging.getLogger("test_setup_logging").info("trigger the delayed file open")
            assert log_path.exists()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level
