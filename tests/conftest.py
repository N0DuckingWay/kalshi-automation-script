"""
File: conftest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Suite-wide pytest fixtures. Holds exactly one: an autouse guard that points
    the backtester's event-title accumulator, and Kalshi's cached /series
    listing, at a per-test temporary directory, so no test can read, rewrite,
    migrate or delete the operator's real backtest_cache/event_titles_v2.json,
    its legacy event_titles.json or series_categories.json.

Dependencies:
    Imports kalshi_betting.historical (its _EVENT_TITLES_CACHE,
    _LEGACY_EVENT_TITLES_CACHE and _SERIES_CATEGORIES_CACHE module paths).
    Imported by pytest only.

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

    The /series listing joined the guard with main.py's live category/tag
    filter: historical.load_series_categories, which main._filter_by_category
    calls whenever a run sets a filter, reads the cached listing and rewrites
    it after a fetch, so a test that reached it unredirected would read the
    operator's real listing — its answer would then depend on the machine —
    and could overwrite it. A test that needs a listing seeds
    historical._SERIES_CATEGORIES_CACHE (this test's tmp_path) itself.
"""
import pytest

from kalshi_betting import historical


@pytest.fixture(autouse=True)
def _isolate_event_title_accumulator(tmp_path, monkeypatch):
    """
    Redirect both event-title accumulator paths, and the cached /series
    listing, into this test's tmp_path.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real paths afterwards.
    """
    monkeypatch.setattr(historical, "_EVENT_TITLES_CACHE",
                        tmp_path / "event_titles_v2.json")
    monkeypatch.setattr(historical, "_LEGACY_EVENT_TITLES_CACHE",
                        tmp_path / "event_titles.json")
    monkeypatch.setattr(historical, "_SERIES_CATEGORIES_CACHE",
                        tmp_path / "series_categories.json")
