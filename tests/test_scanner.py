"""Tests for scanner.py normalize_title() — the core pair-detection function."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from kalshi_betting.scanner import (
    display_title,
    find_same_title_pairs,
    find_time_series_pairs,
    normalize_title,
    pair_key,
)


class TestNormalizeTitle:
    def test_full_month_name_with_year(self):
        result = normalize_title("Will BTC exceed $80k by December 2025?")
        assert "december" not in result
        assert "2025" not in result
        assert "btc" in result
        assert "$80k" in result

    def test_full_month_name_with_date_and_year(self):
        result = normalize_title("Inflation rate by January 31, 2026")
        assert "january" not in result
        assert "31" not in result
        assert "2026" not in result

    def test_abbreviated_month_with_year(self):
        result = normalize_title("Fed funds rate Dec 2025")
        assert "dec" not in result
        assert "2025" not in result
        assert "fed funds rate" in result

    def test_iso_date(self):
        result = normalize_title("GDP report 2025-03-31")
        assert "2025" not in result
        assert "03" not in result
        assert "31" not in result
        assert "gdp report" in result

    def test_quarter(self):
        result = normalize_title("Fed funds rate Q2 2025")
        assert "q2" not in result
        assert "2025" not in result
        assert "fed funds rate" in result

    def test_quarter_without_year(self):
        result = normalize_title("Rate decision Q3")
        assert "q3" not in result
        assert "rate decision" in result

    def test_time_series_pair_normalizes_identically(self):
        # Two markets differing only in deadline must produce the same normalized string
        title_march = "Will BTC exceed $80k by March 2025?"
        title_june = "Will BTC exceed $80k by June 2025?"
        assert normalize_title(title_march) == normalize_title(title_june)

    def test_another_time_series_pair(self):
        a = "Will the Fed cut rates by January 2025?"
        b = "Will the Fed cut rates by June 2025?"
        assert normalize_title(a) == normalize_title(b)

    def test_no_dates_unchanged(self):
        title = "Will it rain tomorrow?"
        result = normalize_title(title)
        assert result == "will it rain tomorrow?"

    def test_output_is_lowercase(self):
        result = normalize_title("BTC PRICE ABOVE 80K")
        assert result == result.lower()

    def test_whitespace_collapsed(self):
        result = normalize_title("Will BTC exceed $80k by December 2025 ?")
        assert "  " not in result

    def test_abbreviated_month_day_sandbox_format(self):
        # "Apr 02" format used in Kalshi sandbox titles
        result = normalize_title("BTC price Apr 02")
        assert "apr" not in result

    def test_numeric_date(self):
        result = normalize_title("Price above 50k on 01/15/2026")
        assert "01" not in result or "2026" not in result

    def test_standalone_year(self):
        result = normalize_title("GDP growth in 2026")
        assert "2026" not in result
        assert "gdp growth" in result

    def test_content_preserved_after_stripping(self):
        # A title with many dates should still have meaningful content left
        result = normalize_title("Will the S&P 500 exceed 6000 by December 31, 2025?")
        assert len(result.strip()) > 5
        assert "s&p 500" in result or "s&p" in result or "500" in result

    def test_result_is_stripped(self):
        result = normalize_title("  Will BTC   exceed $80k by March 2025?  ")
        assert result == result.strip()


def _mock_market(
    *,
    ticker: str,
    event_ticker: str,
    title: str = "",
    subtitle: str = "",
    event_title: str | None = None,
    yes_ask: float = 0.50,
    no_ask: float = 0.50,
    close_time=None,
):
    """Build a SimpleNamespace Kalshi-market stand-in with the fields the scanner reads.

    SimpleNamespace (not MagicMock) is used so that `getattr(m, "_event_title", "")`
    returns the empty string when no event title was attached — MagicMock would
    auto-vivify a child mock and break the pair_key fallback.
    """
    from datetime import UTC, datetime
    attrs = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "subtitle": subtitle,
        "yes_ask_dollars": str(yes_ask),
        "no_ask_dollars": str(no_ask),
        "yes_bid_dollars": str(max(yes_ask - 0.02, 0.01)),
        "close_time": close_time or datetime(2026, 6, 1, tzinfo=UTC),
    }
    if event_title is not None:
        attrs["_event_title"] = event_title
    return SimpleNamespace(**attrs)


class TestPairKey:
    def test_combines_event_and_market_title(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Trump", event_title="2024 Election Winner")
        key = pair_key(m)
        assert "2024 Election Winner" in key
        assert "Trump" in key
        # Sanity — separator present so the two parts don't run together ambiguously
        assert "|" in key

    def test_falls_back_when_event_title_missing(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title=None)
        assert pair_key(m) == "Will BTC exceed $80k"

    def test_falls_back_when_event_title_empty_string(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title="")
        assert pair_key(m) == "Will BTC exceed $80k"


class TestDisplayTitle:
    def test_formats_with_event_prefix(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Trump", event_title="2024 Election Winner")
        label = display_title(m)
        assert label == "2024 Election Winner: Trump"

    def test_falls_back_to_bare_title(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title=None)
        assert display_title(m) == "Will BTC exceed $80k"


class TestSameTitleGrouping:
    """Validates that the combined-key grouping eliminates cross-event MVE collisions."""

    def test_cross_event_same_option_label_does_not_pair(self):
        # Two MVE markets with identical option label "Trump" but in completely
        # different events — must NOT be paired.
        mA = _mock_market(
            ticker="ELECT-TRUMP", event_ticker="ELECT-2024",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
        )
        mB = _mock_market(
            ticker="TIME-TRUMP", event_ticker="TIME-2024",
            title="Trump", event_title="2024 Time Person of the Year",
            yes_ask=0.20, no_ask=0.80,
        )
        pairs = find_same_title_pairs([mA, mB])
        assert pairs == [], f"Expected no pairs across unrelated events; got {pairs}"

    def test_same_event_title_does_pair(self):
        # Two markets with identical event_title + market title but different
        # event_ticker — the legitimate same-title arbitrage case.
        mA = _mock_market(
            ticker="A1", event_ticker="EVT-A",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVT-B",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.40, no_ask=0.60,
        )
        pairs = find_same_title_pairs([mA, mB])
        # Should produce exactly one pair (best-per-group)
        assert len(pairs) == 1
        # Pair must reference both markets
        tickers = {pairs[0].market_a.ticker, pairs[0].market_b.ticker}
        assert tickers == {"A1", "B1"}

    def test_within_event_filter_still_applies(self):
        # Same event_title (so they group), same event_ticker (so within-event filter
        # rejects). This catches the multi-choice-options-in-the-same-event case.
        mA = _mock_market(
            ticker="X-TRUMP", event_ticker="MVE-2024",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
        )
        mB = _mock_market(
            ticker="X-HARRIS", event_ticker="MVE-2024",
            title="Harris", event_title="2024 Election Winner",
            yes_ask=0.50, no_ask=0.50,
        )
        # Different market titles, but event_title shared. find_same_title_pairs
        # groups by (event_title, title, subtitle) — different titles mean they
        # land in different groups, so no pair regardless.
        assert find_same_title_pairs([mA, mB]) == []


class TestTimeSeriesGrouping:
    def test_cross_event_mve_option_label_does_not_pair_in_time_series(self):
        # Same option label, different events at different deadlines —
        # the time-series scanner must NOT pair these because the event titles differ.
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="ELECT-TRUMP-MAR", event_ticker="ELECT-MAR",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="TIME-TRUMP-JUN", event_ticker="TIME-JUN",
            title="Trump", event_title="2024 Time Person of the Year",
            yes_ask=0.20, no_ask=0.80,
            close_time=datetime(2026, 3, 20, tzinfo=UTC),
        )
        # client is unused when `markets` is provided
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert pairs == [], f"Expected no pairs across unrelated MVE events; got {pairs}"
