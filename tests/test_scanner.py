"""Tests for scanner.py normalize_title() — the core pair-detection function."""
from kalshi_betting.scanner import normalize_title


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
