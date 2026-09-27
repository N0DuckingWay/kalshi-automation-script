"""
File: test_treasury.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for treasury.py: parsing the Treasury Fiscal Data API's auction
    records, paging the download, the never-raising load/cache-fallback
    contract of load_risk_free_rates(), the per-day lookup in
    RiskFreeRates.annual_on() and its day-number form (annual_on_days over
    day_numbers, which the dashboard's deployed-capital hurdle shares), and
    that no live-trading module can import this reporting-only module.

Dependencies:
    Imports kalshi_betting.treasury. Runs fully offline: tests/conftest.py's
    autouse _isolate_treasury_rates fixture redirects treasury._RATES_CACHE
    into a per-test tmp_path and stubs treasury._get_json to raise, so no
    test here reaches the real Treasury API or a real backtest_cache/ file
    unless it explicitly re-patches _get_json itself.

Notes:
    A fake failure passed to _get_json must be a non-transient exception
    (RuntimeError, ValueError, ...), never ConnectionError/TimeoutError or an
    HTTP 429/500/502/503/504 status — _http.api_call_with_retry retries
    those (and only those: any other status is fatal at once) and sleeps
    through ~62s of exponential backoff before giving up.
"""
import ast
import importlib
import inspect
import json
import logging
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from kalshi_betting import treasury

# A minimal, realistic two-record download: the bill's actual first auction
# and a later one, oldest first as the API serves them.
GOOD_ROWS = [
    {"auction_date": "2018-10-16", treasury.RISK_FREE_RATE_FIELD: "2.207"},
    {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "4.071"},
]


def _stub_fetch(monkeypatch, rows_or_exc):
    """
    Stand treasury._get_json in for a fake single-page download or failure.

    Args:
        monkeypatch (pytest.MonkeyPatch): Owns the patch for this test.
        rows_or_exc (list[dict] | Exception): A list of API records (served
            as one page, meta["total-pages"]=1) or an exception every call
            raises.
    """
    if isinstance(rows_or_exc, Exception):
        def fake(url):
            raise rows_or_exc
    else:
        def fake(url):
            return {"data": rows_or_exc, "meta": {"total-pages": 1}}

    monkeypatch.setattr(treasury, "_get_json", fake)


class TestParseAuctions:
    """_parse_auctions reads (date, decimal yield) pairs, skipping and
    counting anything unreadable or out of [0%, 100%)."""

    def test_percent_string_becomes_a_decimal(self):
        rows = [{"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "4.071000"}]
        out = treasury._parse_auctions(rows)
        assert out == ((date(2026, 9, 24), pytest.approx(0.04071)),)

    def test_output_is_ascending_even_when_the_input_is_not(self):
        rows = [
            {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "4.0"},
            {"auction_date": "2018-10-16", treasury.RISK_FREE_RATE_FIELD: "2.2"},
            {"auction_date": "2020-01-01", treasury.RISK_FREE_RATE_FIELD: "0.5"},
        ]
        out = treasury._parse_auctions(rows)
        assert [d for d, _ in out] == [date(2018, 10, 16), date(2020, 1, 1), date(2026, 9, 24)]

    def test_a_duplicate_date_is_averaged(self):
        rows = [
            {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "4.0"},
            {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "6.0"},
        ]
        out = treasury._parse_auctions(rows)
        assert out == ((date(2026, 9, 24), pytest.approx(0.05)),)

    @pytest.mark.parametrize("bad_row", [
        {"auction_date": None, treasury.RISK_FREE_RATE_FIELD: "4.0"},           # null date
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: None},    # null rate
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: ""},      # empty string
        {"auction_date": "2026-09-24"},                                        # missing key
        "not a dict",                                                          # non-dict row
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: float("nan")},  # NaN
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "-0.1"},  # negative
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: "150"},   # >= 100%
        # bool is an int subclass: float(True) is 1.0, a 1% yield, not a record
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: True},    # JSON true
        {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: False},   # JSON false
    ], ids=["null-date", "null-rate", "empty-string", "missing-key", "non-dict",
            "nan", "negative", "over-100pct", "json-true", "json-false"])
    def test_unreadable_or_out_of_range_records_are_skipped_and_counted(self, bad_row, caplog):
        good = {"auction_date": "2020-01-01", treasury.RISK_FREE_RATE_FIELD: "1.0"}
        with caplog.at_level(logging.INFO):
            out = treasury._parse_auctions([bad_row, good])
        assert out == ((date(2020, 1, 1), pytest.approx(0.01)),)
        assert "1 record(s) skipped — an unreadable auction_date, an unreadable " \
            "high_investment_rate, or a yield outside [0%, 100%)" in caplog.text

    def test_a_yield_too_large_for_a_float_is_skipped(self):
        # A JSON integer of 400 digits (a hand-edited cache) parses to an int
        # that float() cannot hold: OverflowError, not ValueError
        huge = {"auction_date": "2026-09-24", treasury.RISK_FREE_RATE_FIELD: int("9" * 400)}
        good = {"auction_date": "2020-01-01", treasury.RISK_FREE_RATE_FIELD: "1.0"}
        assert treasury._parse_auctions([huge, good]) == ((date(2020, 1, 1),
                                                          pytest.approx(0.01)),)

    def test_all_records_unreadable_raises(self):
        with pytest.raises(ValueError):
            treasury._parse_auctions([{"auction_date": None}])


class TestFetch:
    """_page_url / _fetch_rows: the request shape and the paging loop."""

    def test_page_url_decodes_to_exactly_the_expected_params(self):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(treasury._page_url(3))
        assert parse_qs(parsed.query) == {
            "fields": [f"auction_date,{treasury.RISK_FREE_RATE_FIELD}"],
            "filter": [f"security_term:eq:{treasury.RISK_FREE_BILL_TERM},"
                       "cash_management_bill_cmb:eq:No"],
            "sort": ["auction_date"],
            "page[size]": [str(treasury.TREASURY_API_PAGE_SIZE)],
            "page[number]": ["3"],
        }

    def test_a_two_page_response_is_called_twice_and_concatenated_in_order(self, monkeypatch):
        from urllib.parse import parse_qs, urlparse

        pages = {
            1: {"data": [{"auction_date": "2018-10-16"}], "meta": {"total-pages": 2}},
            2: {"data": [{"auction_date": "2020-01-01"}], "meta": {"total-pages": 2}},
        }
        calls = []

        def fake(url):
            page = int(parse_qs(urlparse(url).query)["page[number]"][0])
            calls.append(page)
            return pages[page]

        monkeypatch.setattr(treasury, "_get_json", fake)
        rows = treasury._fetch_rows()
        assert [r["auction_date"] for r in rows] == ["2018-10-16", "2020-01-01"]
        assert calls == [1, 2]

    @pytest.mark.parametrize("body", [
        ["a", "list", "not", "an", "object"],
        {"data": "not a list"},
        {"meta": {"total-pages": 1}},  # no "data" key at all
    ], ids=["list-body", "data-not-a-list", "no-data-key"])
    def test_a_non_conforming_page_raises(self, monkeypatch, body):
        monkeypatch.setattr(treasury, "_get_json", lambda url: body)
        with pytest.raises(ValueError):
            treasury._fetch_rows()

    def test_a_page_count_that_never_reaches_total_pages_raises_after_the_cap(self, monkeypatch):
        monkeypatch.setattr(
            treasury, "_get_json",
            lambda url: {"data": [], "meta": {"total-pages": 999}},
        )
        with pytest.raises(ValueError, match=str(treasury.TREASURY_API_MAX_PAGES)):
            treasury._fetch_rows()


class TestLoad:
    """load_risk_free_rates(): API first, cache fallback, never raises."""

    def test_success_returns_source_api_and_saves_the_cache(self, monkeypatch, caplog):
        _stub_fetch(monkeypatch, GOOD_ROWS)
        with caplog.at_level(logging.INFO):
            got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_API
        assert got.fetched_at is not None and got.fetched_at.tzinfo is not None
        assert got.auctions == (
            (date(2018, 10, 16), pytest.approx(0.02207)),
            (date(2026, 9, 24), pytest.approx(0.04071)),
        )
        on_disk = json.loads(treasury._RATES_CACHE.read_text())
        assert on_disk["term"] == treasury.RISK_FREE_BILL_TERM
        assert on_disk["field"] == treasury.RISK_FREE_RATE_FIELD
        assert on_disk["records"] == GOOD_ROWS
        assert "2 8-Week Treasury bill auctions" in caplog.text
        assert "2026-09-24" in caplog.text

    def test_failure_with_a_saved_copy_falls_back_to_it(self, monkeypatch, caplog):
        saved_at = datetime(2026, 9, 1, tzinfo=UTC)
        treasury._RATES_CACHE.write_text(json.dumps({
            "fetched_at": saved_at.isoformat(),
            "term": treasury.RISK_FREE_BILL_TERM,
            "field": treasury.RISK_FREE_RATE_FIELD,
            "records": GOOD_ROWS,
        }))
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_CACHE
        assert got.fetched_at == saved_at
        assert "2026-09-01" in caplog.text

    def test_failure_with_no_copy_is_unavailable(self, monkeypatch, caplog):
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got == treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        assert "no usable earlier download is saved" in caplog.text

    @pytest.mark.parametrize("bad_cache", [
        {"fetched_at": "2026-09-01T00:00:00+00:00", "term": "13-Week",
         "field": treasury.RISK_FREE_RATE_FIELD, "records": GOOD_ROWS},
        {"fetched_at": "2026-09-01T00:00:00+00:00", "term": treasury.RISK_FREE_BILL_TERM,
         "field": "high_discnt_rate", "records": GOOD_ROWS},
        {"fetched_at": "not a date", "term": treasury.RISK_FREE_BILL_TERM,
         "field": treasury.RISK_FREE_RATE_FIELD, "records": GOOD_ROWS},
        {"fetched_at": "2026-09-01T00:00:00", "term": treasury.RISK_FREE_BILL_TERM,
         "field": treasury.RISK_FREE_RATE_FIELD, "records": GOOD_ROWS},
    ], ids=["wrong-term", "wrong-field", "unparseable-fetched_at", "naive-fetched_at"])
    def test_a_copy_for_another_term_field_or_a_bad_stamp_is_ignored(self, monkeypatch, bad_cache):
        treasury._RATES_CACHE.write_text(json.dumps(bad_cache))
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_UNAVAILABLE

    @pytest.mark.parametrize("stamp", ["9999-12-31T23:30:00-01:00",
                                       "0001-01-01T00:30:00+01:00"])
    def test_a_stamp_with_no_utc_instant_is_no_usable_copy(self, monkeypatch, stamp):
        # Aware, but within a day of datetime's range: astimezone(UTC)
        # overflows. Served, it raised OverflowError out of the dashboard's
        # header after a finished backtest; now it is no usable copy
        treasury._RATES_CACHE.write_text(json.dumps({
            "fetched_at": stamp, "term": treasury.RISK_FREE_BILL_TERM,
            "field": treasury.RISK_FREE_RATE_FIELD, "records": GOOD_ROWS}))
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        assert treasury._read_cache() is None
        got = treasury.load_risk_free_rates()
        assert got == treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)

    def test_a_cached_stamp_is_returned_in_utc(self, monkeypatch):
        treasury._RATES_CACHE.write_text(json.dumps({
            "fetched_at": "2026-09-01T02:30:00-04:00", "term": treasury.RISK_FREE_BILL_TERM,
            "field": treasury.RISK_FREE_RATE_FIELD, "records": GOOD_ROWS}))
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_CACHE
        assert got.fetched_at == datetime(2026, 9, 1, 6, 30, tzinfo=UTC)
        assert got.fetched_at.utcoffset() == timedelta(0)

    def test_a_corrupt_cache_is_ignored(self, monkeypatch):
        treasury._RATES_CACHE.write_text("not valid json at all {")
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_UNAVAILABLE

    def _save_copy(self, records: list) -> None:
        treasury._RATES_CACHE.write_text(json.dumps({
            "fetched_at": "2026-09-01T00:00:00+00:00", "term": treasury.RISK_FREE_BILL_TERM,
            "field": treasury.RISK_FREE_RATE_FIELD, "records": records}))

    def test_a_cached_yield_too_large_for_a_float_never_raises(self, monkeypatch):
        # Reproduced: this raised OverflowError out of the loader
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        huge = {"auction_date": "2026-01-01", treasury.RISK_FREE_RATE_FIELD: int("9" * 400)}
        self._save_copy([huge])
        assert treasury.load_risk_free_rates().source == treasury.SOURCE_UNAVAILABLE
        # ... and beside a readable record, the copy is still served
        self._save_copy([huge, GOOD_ROWS[1]])
        got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_CACHE
        assert got.auctions == ((date(2026, 9, 24), pytest.approx(0.04071)),)

    def test_a_cache_nested_too_deep_to_parse_never_raises(self, monkeypatch, caplog):
        # Reproduced: json.loads raised RecursionError, which _load_json_cache
        # does not catch
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        treasury._RATES_CACHE.write_text("[" * 100_000 + "]" * 100_000)
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_UNAVAILABLE
        assert "Could not read the saved 8-Week bill yields" in caplog.text

    def test_an_unreadable_cache_directory_never_raises(self, monkeypatch, caplog):
        # Reproduced with an unsearchable directory: path.exists() raises
        # PermissionError out of _load_json_cache
        _stub_fetch(monkeypatch, RuntimeError("offline"))
        monkeypatch.setattr(treasury, "_load_json_cache",
                            MagicMock(side_effect=PermissionError("denied")))
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got == treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        assert "Could not read the saved 8-Week bill yields" in caplog.text
        assert "no usable earlier download is saved" in caplog.text

    def test_any_error_saving_still_returns_source_api(self, monkeypatch, caplog):
        # Not only OSError: a TypeError out of the serializer must not cost the
        # run the yields it just downloaded
        _stub_fetch(monkeypatch, GOOD_ROWS)
        monkeypatch.setattr(treasury, "_save_json_cache",
                            MagicMock(side_effect=TypeError("not serializable")))
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_API
        assert len(got.auctions) == 2
        assert "Could not save" in caplog.text

    def test_an_oserror_saving_still_returns_source_api(self, monkeypatch, caplog):
        _stub_fetch(monkeypatch, GOOD_ROWS)
        monkeypatch.setattr(treasury, "_save_json_cache",
                            MagicMock(side_effect=OSError("disk full")))
        with caplog.at_level(logging.WARNING):
            got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_API
        assert "Could not save" in caplog.text

    def test_an_unexpected_error_inside_parsing_still_returns_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(treasury, "_get_json",
                            lambda url: {"data": [{"auction_date": "2026-09-24"}],
                                        "meta": {"total-pages": 1}})
        monkeypatch.setattr(treasury, "_parse_auctions",
                            MagicMock(side_effect=KeyError("boom")))
        got = treasury.load_risk_free_rates()  # must not raise
        assert got.source == treasury.SOURCE_UNAVAILABLE

    def test_a_successful_download_is_preferred_over_an_existing_copy(self, monkeypatch):
        treasury._RATES_CACHE.write_text(json.dumps({
            "fetched_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            "term": treasury.RISK_FREE_BILL_TERM,
            "field": treasury.RISK_FREE_RATE_FIELD,
            "records": [{"auction_date": "2020-01-01", treasury.RISK_FREE_RATE_FIELD: "0.5"}],
        }))
        _stub_fetch(monkeypatch, GOOD_ROWS)
        got = treasury.load_risk_free_rates()
        assert got.source == treasury.SOURCE_API
        assert got.auctions[0][0] == date(2018, 10, 16)


class TestAnnualOn:
    """RiskFreeRates.annual_on(): the latest auction on or before each date —
    and its day-number form, annual_on_days(day_numbers(...))."""

    RATES = treasury.RiskFreeRates(
        ((date(2020, 1, 6), 0.01), (date(2020, 1, 13), 0.02), (date(2020, 1, 20), 0.03)),
        treasury.SOURCE_API, datetime(2020, 1, 21, tzinfo=UTC),
    )

    def test_a_day_before_the_first_auction_takes_the_first_yield(self):
        assert self.RATES.annual_on([date(2019, 1, 1)]) == pytest.approx([0.01])

    def test_the_auction_day_itself_takes_that_auctions_yield(self):
        assert self.RATES.annual_on([date(2020, 1, 13)]) == pytest.approx([0.02])

    def test_a_day_between_two_auctions_takes_the_earlier_one(self):
        assert self.RATES.annual_on([date(2020, 1, 15)]) == pytest.approx([0.02])

    def test_a_day_after_the_last_auction_takes_the_last_yield(self):
        assert self.RATES.annual_on([date(2020, 6, 1)]) == pytest.approx([0.03])

    def test_input_can_be_a_series_of_date_objects(self):
        s = pd.Series([date(2020, 1, 6), date(2020, 1, 20)])
        assert self.RATES.annual_on(s) == pytest.approx([0.01, 0.03])

    def test_input_can_be_a_naive_trading_day_datetimeindex(self):
        idx = pd.DatetimeIndex([datetime(2020, 1, 13), datetime(2020, 1, 21)])
        assert self.RATES.annual_on(idx) == pytest.approx([0.02, 0.03])

    def test_a_tz_aware_index_reads_its_own_calendar_day(self):
        # 23:00 America/New_York on Jan 13 is 04:00 UTC on Jan 14 — the lookup
        # must use the LOCAL date (Jan 13, rate 0.02), never the UTC one.
        idx = pd.DatetimeIndex([datetime(2020, 1, 13, 23, 0)]).tz_localize("America/New_York")
        assert self.RATES.annual_on(idx) == pytest.approx([0.02])

    def test_unavailable_rates_return_zeros_of_the_input_length(self):
        unavailable = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        out = unavailable.annual_on([date(2020, 1, 1), date(2020, 1, 2)])
        assert out == pytest.approx([0.0, 0.0])

    def test_empty_input_returns_an_empty_array(self):
        assert len(self.RATES.annual_on([])) == 0

    def test_day_numbers_are_date_ordinals(self):
        # The one day numbering the lookups and the dashboard's
        # capital-deployed helper share: date.toordinal, a timestamp read as
        # its own calendar date (as datetime.date() reads it)
        days = [date(2019, 12, 31), date(2020, 1, 13), date(2020, 3, 1)]
        want = [d.toordinal() for d in days]
        assert list(treasury.day_numbers(days)) == want
        assert list(treasury.day_numbers(pd.Series(days))) == want
        stamped = pd.to_datetime(pd.Series(days)) + pd.Timedelta(hours=23)
        assert list(treasury.day_numbers(stamped)) == want
        local = stamped.dt.tz_localize("America/New_York")
        assert list(treasury.day_numbers(local)) == [t.date().toordinal() for t in local]
        assert list(treasury.day_numbers(local)) == want
        assert len(treasury.day_numbers([])) == 0

    def test_day_numbers_falls_back_to_pandas_for_what_has_no_toordinal(self):
        # Strings and numpy datetime64 values carry no toordinal(): read
        # through pd.to_datetime instead, on the same numbering
        days = [date(2019, 12, 31), date(2020, 1, 13), date(2020, 3, 1)]
        want = [d.toordinal() for d in days]
        assert list(treasury.day_numbers([d.isoformat() for d in days])) == want
        assert list(treasury.day_numbers(np.array(days, dtype="datetime64[D]"))) == want
        stamped = np.array([f"{d.isoformat()}T23:15" for d in days], dtype="datetime64[m]")
        assert list(treasury.day_numbers(stamped)) == want
        assert list(treasury.day_numbers(list(stamped))) == want
        assert list(self.RATES.annual_on(np.array(["2019-01-01", "2020-01-15"],
                                                  dtype="datetime64[D]"))) == [0.01, 0.02]

    def test_annual_on_days_is_annual_on_for_day_numbers(self):
        days = [date(2019, 1, 1), date(2020, 1, 13), date(2020, 1, 15), date(2020, 6, 1)]
        assert list(self.RATES.annual_on_days(treasury.day_numbers(days))) \
            == list(self.RATES.annual_on(days)) == [0.01, 0.02, 0.02, 0.03]
        unavailable = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        assert list(unavailable.annual_on_days(treasury.day_numbers(days))) == [0.0] * 4


_PIPELINE_MODULES = [
    "main", "scanner", "strategy", "trader", "reporter", "scheduler", "auth", "v2_probe",
]


class TestReportingOnly:
    """treasury.py is reporting-only: nothing sizes, prices or settles on it,
    and no live-pipeline module may import it (the v2_probe import-pin
    precedent in tests/test_v2_probe.py::TestPipelineIsolation)."""

    @pytest.mark.parametrize("module", _PIPELINE_MODULES)
    def test_no_pipeline_module_imports_treasury(self, module):
        mod = importlib.import_module(f"kalshi_betting.{module}")
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[-1]}
                names |= {a.name for a in node.names}
            else:
                continue
            assert "treasury" not in names, (
                f"kalshi_betting/{module}.py imports treasury — it must stay "
                "reporting-only and unreachable from the live pipeline"
            )
