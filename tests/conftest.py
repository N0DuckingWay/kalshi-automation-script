"""
File: conftest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Suite-wide pytest fixtures, and one helper they share. Two fixtures: an
    autouse guard that points the backtester's event-title accumulator, and
    Kalshi's cached /series listing, at a per-test temporary directory, so no
    test can read, rewrite, migrate or delete the operator's real
    backtest_cache/event_titles_v2.json, its legacy event_titles.json or
    series_categories.json; and pre_toggle_defaults, which a test built on the
    live toggles as they stood before the 2026-09-27 flip requests by name.
    The helper, apply_pre_toggle_defaults, is what that fixture applies, and
    what a class-scoped fixture applies on a MonkeyPatch of its own.

Dependencies:
    Imports kalshi_betting.historical (its _EVENT_TITLES_CACHE,
    _LEGACY_EVENT_TITLES_CACHE and _SERIES_CATEGORIES_CACHE module paths), and
    kalshi_betting.config, backtester and backtest (the live toggles and the
    by-value copies two of them bind). Imported by pytest, and by test modules
    for apply_pre_toggle_defaults (from .conftest import ...).

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

    The live toggles were flipped on 2026-09-27 (config.py: tier floors off,
    spread band 0-0.5, k 0.80, no per-trade cap, a 20% same-title cap). A test
    whose expected figures were derived under the old values (tier floors on,
    no band, k 0.75, a 20% cap for every pair, no extra same-title cap) keeps
    them by requesting pre_toggle_defaults, never by re-deriving them at the
    new values: the figures pin arithmetic, not a policy. Every binding must
    move together — config's constants, which live_settings(),
    max_affordable_pairs(fraction=None) and time_series_profit_prob(k=None)
    read at call time, AND the copies backtester and backtest bind by value at
    import — or a test would price with one value and size with another.
"""
import pytest

from kalshi_betting import backtest, backtester, config, historical


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


def apply_pre_toggle_defaults(mp) -> None:
    """
    Put every binding of the live toggles back to its pre-2026-09-27 value.

    Before the flip: tier floors on, no spread band, k 0.75, a 20% per-trade
    cap for every pair, no extra same-title cap, and no category or tag filter
    (the last two were None before the flip and still are; they are pinned so
    one call states all seven live toggles). This patches config's own
    constants, which live_settings(), max_affordable_pairs(fraction=None) and
    time_series_profit_prob(k=None) read at call time — so the backtester's
    and backtest.py's live-rule reports, which read live_settings() at call
    time, follow too — AND the by-value copies backtester (BUDGET_FRACTION,
    SAME_TITLE_SIZE_CAP, TIME_SERIES_INTERVAL_PROB_DISCOUNT) and backtest
    (TIME_SERIES_INTERVAL_PROB_DISCOUNT) bind at import. No other module binds
    one of these by value: scanner, strategy, trader and main read them only
    through a LiveSettings (pinned by the live-path AST walk in
    test_strategy.py), and dashboard reads none.

    Args:
        mp (pytest.MonkeyPatch): The patcher. A class-scoped sweep fixture
            passes one of its own, created at fixture start and undone only
            after its yield, because those fixtures undo their other patches
            mid-fixture and the lazy size-cap cells their tests read must
            still see these values.
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
    The live toggles as they stood before the 2026-09-27 flip, for one test.

    Applies apply_pre_toggle_defaults on the test's own monkeypatch, which
    undoes every patch when the test ends.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
    """
    apply_pre_toggle_defaults(monkeypatch)
