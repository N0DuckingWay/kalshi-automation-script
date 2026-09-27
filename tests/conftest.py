"""
File: conftest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Suite-wide pytest fixtures and one helper. Two autouse guards: one points
    the event-title accumulator and Kalshi's cached /series listing at a
    per-test tmp_path, so no test can touch the operator's real event-title
    accumulators or series_categories.json; the other keeps every test off the
    Treasury API and the real rates cache. pre_toggle_defaults pins the live
    toggles for a test whose figures assume fixed values (they pin arithmetic,
    not config.py's policy); its helper, apply_pre_toggle_defaults, also
    serves class-scoped fixtures.

Dependencies:
    Imports kalshi_betting.historical (the three cache paths),
    kalshi_betting.treasury (its _RATES_CACHE path and _get_json), config, and
    backtester and backtest (the by-value copies they bind). Imported by
    pytest, and by test modules for apply_pre_toggle_defaults.

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

    A test that needs a /series listing seeds historical._SERIES_CATEGORIES_CACHE
    (its tmp_path) itself.
"""
import pytest

from kalshi_betting import backtest, backtester, config, historical, treasury


@pytest.fixture(autouse=True)
def _isolate_event_title_accumulator(tmp_path, monkeypatch):
    """
    Redirect both event-title accumulator paths and the /series listing to tmp_path.

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


def apply_pre_toggle_defaults(mp) -> None:
    """
    Pin every binding of the seven live toggles to fixed values.

    Tier floors on, no spread band, k 0.75, a 20% per-trade cap, no extra
    same-title cap, no category or tag filter. Patches config's constants,
    read at call time, AND the by-value copies backtester and backtest bind at
    import, or a test would price with one value and size with another; no
    other module binds one by value (pinned by test_config.py's
    TestPreToggleDefaults).

    Args:
        mp (pytest.MonkeyPatch): The patcher; a class-scoped fixture passes its
            own, since it undoes its other patcher before its yield and its lazy
            size-cap cells must still see these values.
    """
    mp.setattr(config, "TIME_SERIES_TIER_FLOORS", True)
    mp.setattr(config, "TIME_SERIES_SPREAD_BAND", (0.0, 1.0))
    mp.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.75)
    mp.setattr(config, "BUDGET_FRACTION", 0.20)
    mp.setattr(config, "SAME_TITLE_SIZE_CAP", 1.0)
    mp.setattr(config, "TRADE_CATEGORIES", None)
    mp.setattr(config, "TRADE_TAGS", None)
    mp.setattr(backtester, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.75)
    mp.setattr(backtester, "BUDGET_FRACTION", 0.20)
    mp.setattr(backtester, "SAME_TITLE_SIZE_CAP", 1.0)
    mp.setattr(backtest, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.75)


@pytest.fixture
def pre_toggle_defaults(monkeypatch):
    """
    Apply apply_pre_toggle_defaults for one test, undone when it ends.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
    """
    apply_pre_toggle_defaults(monkeypatch)


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
