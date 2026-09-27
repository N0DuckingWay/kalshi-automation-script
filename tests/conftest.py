"""
File: conftest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Suite-wide pytest fixtures. Holds two, both autouse: a guard that points
    the backtester's event-title accumulator at a per-test temporary directory,
    so no test can read, rewrite, migrate or delete the operator's real
    backtest_cache/event_titles_v2.json or its legacy event_titles.json; and a
    guard that keeps every test off the Treasury API and the real rates cache.

Dependencies:
    Imports kalshi_betting.historical (its _EVENT_TITLES_CACHE and
    _LEGACY_EVENT_TITLES_CACHE module paths) and kalshi_betting.treasury (its
    _RATES_CACHE path and _get_json). Imported by pytest only.

Notes:
    Before DR-51 four tests in test_historical.py (TestFetchAllSettledMarkets'
    min_settled_ts and cutoff-warning tests) reached the real
    historical._load_or_build_event_titles through fetch_all_settled_markets
    without redirecting its cache path — measured by wrapping the function on
    the pre-DR-51 tree — so a suite run inside a checkout that held a real
    accumulator parsed it whole on each of them (374 MB on 2026-09-24), even
    though each asked for no ticker at all. DR-51's empty-request early return
    already stops that read; this guard makes it impossible for any test,
    because that function now also MIGRATES a legacy file and then deletes it,
    which a test must never do to real data. Tests that seed the accumulator
    still use test_historical.py's isolated_cache fixture, which patches the
    same two names to the same tmp_path and returns the v2 path.
"""
import pytest

from kalshi_betting import historical, treasury


@pytest.fixture(autouse=True)
def _isolate_event_title_accumulator(tmp_path, monkeypatch):
    """
    Redirect both event-title accumulator paths into this test's tmp_path.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real paths afterwards.
    """
    monkeypatch.setattr(historical, "_EVENT_TITLES_CACHE",
                        tmp_path / "event_titles_v2.json")
    monkeypatch.setattr(historical, "_LEGACY_EVENT_TITLES_CACHE",
                        tmp_path / "event_titles.json")


@pytest.fixture(autouse=True)
def _isolate_treasury_rates(tmp_path, monkeypatch):
    """
    Keep every test off the Treasury API and away from the real rates cache.

    The stub raises a plain RuntimeError, which api_call_with_retry treats as
    fatal, so a test that reaches the loader falls through at once instead of
    sleeping through the retry backoff.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real path and function
            afterwards.
    """
    monkeypatch.setattr(treasury, "_RATES_CACHE", tmp_path / "treasury_bill_rates.json")

    def _offline(url):
        raise RuntimeError("the Treasury API is not reachable from the test suite")

    monkeypatch.setattr(treasury, "_get_json", _offline)
