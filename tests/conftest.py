"""
File: conftest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Suite-wide pytest fixtures and two helpers. Eight autouse guards
    (_isolate_event_title_accumulator, _isolate_treasury_rates,
    _fresh_order_write_pacer, _fresh_v2_mapping_disproof_state,
    _isolate_live_defaults_for_the_session, _isolate_live_defaults,
    _isolate_live_runs and _isolate_live_dashboard, in the order described
    here): one points
    the event-title accumulator and Kalshi's cached /series listing at a
    per-test tmp_path, so no test can touch the operator's real event-title
    accumulators or series_categories.json; one keeps every test off the
    Treasury API and the real rates cache; one gives every test a full,
    fresh order-write pacer, so no test waits on writes an earlier test made;
    one clears trader's disproven-mapping latch and its record of unchecked
    NO legs, so a test that disproves the V2 NO-leg mapping cannot stop every
    later test's trades; two point the saved live defaults file
    (config.LIVE_DEFAULTS_FILE) away from the repo's live_defaults.json — at
    an empty session directory for the whole run, so class-scoped fixtures
    never see the real file, and at each test's own tmp_path, so a test that
    saves defaults never leaks them into the next; and one points the
    live-run lock (config.LIVE_RUN_LOCK_FILE) at each test's own tmp_path and
    shortens its waits, so no test takes or waits on the machine's real lock
    in the home folder, and points the defaults server's run folders
    (config.LIVE_RUNS_DIR) there too, so no test writes a run folder into the
    checkout; and one points the live dashboard's files (its JSON log of
    reads, its kept daily prices and its server log) and the trade log it
    reads (reporter.PROD_LOG_PATH, with its lock) at each test's own
    tmp_path, so no test writes them into the checkout or reads the
    operator's trade log. pre_toggle_defaults pins
    the live toggles for a test whose figures assume fixed values (they pin
    arithmetic, not config.py's policy); its helper, apply_pre_toggle_defaults,
    also serves class-scoped fixtures. save_config_live_defaults (and the
    saved_live_defaults fixture that calls it) saves config.py's toggles as
    the live defaults, for a test that needs a saved file.

Dependencies:
    Imports kalshi_betting.historical (the three cache paths),
    kalshi_betting.treasury (its _RATES_CACHE path and _get_json),
    kalshi_betting.trader (its _WritePacer, _ORDER_WRITE_PACER,
    _V2_NO_MAPPING_DISPROVEN, _V2_UNCHECKED_NO_LEGS and
    _V2_UNCHECKED_BASELINES), config (the toggle
    constants, the order-write rate and burst, LIVE_DEFAULTS_FILE,
    live_settings and save_live_defaults, the live-run lock's
    LIVE_RUN_LOCK_FILE, LIVE_RUN_LOCK_WAIT_SECONDS and
    LIVE_RUN_LOCK_POLL_SECONDS, the defaults server's LIVE_RUNS_DIR, and the
    live dashboard's LIVE_PORTFOLIO_LOG_FILE, LIVE_MARKS_CACHE_DIR and
    LIVE_DASHBOARD_LOG_FILE), kalshi_betting.reporter (PROD_LOG_PATH and
    _LOCK_PATH, the trade log the live dashboard reads), and
    backtester and backtest (the
    by-value copies they bind). Imported by pytest, and by test modules for
    apply_pre_toggle_defaults and save_config_live_defaults.

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

from kalshi_betting import backtest, backtester, config, historical, reporter, trader, treasury


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
    Pin every binding of the ten live toggles to fixed values.

    Tier floors on, no spread band, k 0.75, a 20% per-trade cap, no extra
    same-title cap, no category or tag filter, no adding to held pairs and no
    selling (a sell level of None and no minimum of days). The writer leaves
    the keys of the last three out while they are off, so a file saved from
    these values holds only the other seven toggles. Patches config's constants,
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
    mp.setattr(config, "ADD_TO_HELD_PAIRS", False)
    mp.setattr(config, "SELL_AT", None)
    mp.setattr(config, "SELL_MIN_DAYS", None)
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


@pytest.fixture(autouse=True)
def _fresh_order_write_pacer(monkeypatch):
    """
    Give each test its own full order-write pacer.

    trader._ORDER_WRITE_PACER is one module-level token bucket shared by every
    order and transfer POST, so without this a test could wait on writes an
    earlier test made. A fresh pacer lets a test take ORDER_WRITE_BURST places
    with no wait (a pair takes two: its NO leg's and one held for its YES leg).

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores the module's pacer afterwards.
    """
    monkeypatch.setattr(
        trader, "_ORDER_WRITE_PACER",
        trader._WritePacer(config.ORDER_WRITES_PER_SECOND, config.ORDER_WRITE_BURST),
    )


@pytest.fixture(autouse=True)
def _fresh_v2_mapping_disproof_state(monkeypatch):
    """
    Start each test with trader's disproven-mapping latch clear and no
    unchecked NO legs (or their earlier holdings) on record.

    trader._V2_NO_MAPPING_DISPROVEN lasts for the process, and once set,
    _execute_one sends nothing for any pair. A test anywhere in the suite that
    disproves the V2 NO-leg mapping would otherwise make every later test's
    trades come back "failed" with nothing sent. trader._V2_UNCHECKED_NO_LEGS
    also lasts for the process, and a later test's disproof CRITICAL would
    name tickers an earlier test recorded, and trader._V2_UNCHECKED_BASELINES
    (what each of those markets held before its pair) likewise. monkeypatch
    puts back the values from before the test, so none can carry over in
    either direction.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores all three afterwards.
    """
    monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", False)
    monkeypatch.setattr(trader, "_V2_UNCHECKED_NO_LEGS", [])
    monkeypatch.setattr(trader, "_V2_UNCHECKED_BASELINES", {})


@pytest.fixture(scope="session", autouse=True)
def _isolate_live_defaults_for_the_session(tmp_path_factory):
    """
    Point the saved live defaults file at an empty session directory.

    In place before class-scoped fixtures run: those set up before any
    function-scoped fixture is active (the backtester's sweep fixtures run
    whole backtests there), so the per-test redirect below cannot cover them.
    A class fixture that needs saved defaults points the file at a directory
    of its own, never this one. That redirect is in force during class setup
    only: each test body sees its own tmp_path (the per-test redirect below),
    so a test that reads the saved defaults itself saves them there (the
    saved_live_defaults fixture).

    Args:
        tmp_path_factory (pytest.TempPathFactory): pytest's session-wide
            temporary-directory maker.

    Yields:
        None: The redirect holds until the session ends.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "LIVE_DEFAULTS_FILE",
               tmp_path_factory.mktemp("live-defaults") / "live_defaults.json")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolate_live_defaults(tmp_path, monkeypatch):
    """
    Give each test its own saved live defaults path, under its tmp_path.

    So a test that saves defaults never leaks them into the next one, and a
    test with nothing saved reads "none saved".

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the session path afterwards.
    """
    monkeypatch.setattr(config, "LIVE_DEFAULTS_FILE", tmp_path / "live_defaults.json")


@pytest.fixture(autouse=True)
def _isolate_live_runs(tmp_path, monkeypatch):
    """
    Point the live-run lock and the defaults server's run folders at this test's tmp_path.

    config.LIVE_RUN_LOCK_FILE is read at call time (run_lock.py), so every
    test that runs main.main() in production without --dry-run takes a lock
    of its own, never the one in the home folder that real runs share. The
    wait drops to 0.2 s and the retry interval to 0.01 s, so a test that
    holds the lock sees a second run refused quickly. The folder under
    tmp_path is not made here: run_lock.acquire() makes it, as it would the
    real one. config.LIVE_RUNS_DIR, where defaults_server keeps each run it
    starts, is read at call time too, so no test writes one into the
    checkout's live_runs/.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real path and waits afterwards.
    """
    monkeypatch.setattr(config, "LIVE_RUN_LOCK_FILE", tmp_path / "lock" / "live_run.lock")
    monkeypatch.setattr(config, "LIVE_RUN_LOCK_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(config, "LIVE_RUN_LOCK_POLL_SECONDS", 0.01)
    monkeypatch.setattr(config, "LIVE_RUNS_DIR", tmp_path / "live_runs")


@pytest.fixture(autouse=True)
def _isolate_live_dashboard(tmp_path, monkeypatch):
    """
    Point the live dashboard's files at this test's tmp_path.

    config.LIVE_PORTFOLIO_LOG_FILE (one JSON line per read of the account),
    config.LIVE_MARKS_CACHE_DIR (finalized markets' daily prices) and
    config.LIVE_DASHBOARD_LOG_FILE (the server's log) are all read when they
    are used, so no test writes one into the checkout or reads the
    operator's own. The trade log the Live trading tab reads,
    reporter.PROD_LOG_PATH (looked up when live_portfolio.trade_log_paths
    is called, its fallback copies beside it), and its lock file go there
    too, so no test reads the operator's real trade log or its fallback
    copies.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real paths afterwards.
    """
    monkeypatch.setattr(config, "LIVE_PORTFOLIO_LOG_FILE", tmp_path / "live_portfolio_log.jsonl")
    monkeypatch.setattr(config, "LIVE_MARKS_CACHE_DIR", tmp_path / "live_marks")
    monkeypatch.setattr(config, "LIVE_DASHBOARD_LOG_FILE", tmp_path / "kalshi_live_dashboard.log")
    monkeypatch.setattr(reporter, "PROD_LOG_PATH", tmp_path / "trade_log.xlsx")
    monkeypatch.setattr(reporter, "_LOCK_PATH", tmp_path / "trade_log.xlsx.lock")


def save_config_live_defaults() -> None:
    """
    Save config.py's toggles, as live_settings() reads them now, as the live defaults.

    Saves exactly the values config.py's constants give (after any patch a
    test made), for a test that needs a saved file. Writes wherever
    config.LIVE_DEFAULTS_FILE points (a test's tmp_path under
    _isolate_live_defaults).

    Raises:
        LiveDefaultsError: If the save fails (config.save_live_defaults).
    """
    config.save_live_defaults(config.live_settings(), source="")


@pytest.fixture
def saved_live_defaults(_isolate_live_defaults):
    """
    Run save_config_live_defaults() for one test, into that test's own path.

    Requests _isolate_live_defaults by name so the file is written only after
    the path points at the test's tmp_path.

    Args:
        _isolate_live_defaults (None): The per-test redirect, set up first.
    """
    save_config_live_defaults()
