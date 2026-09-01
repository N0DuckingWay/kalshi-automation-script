"""
File: config.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Central store for all tunable constants, file paths, and fee-formula helpers
    used throughout the kalshi_betting package. Every threshold that controls
    whether a pair is traded, how large a position is, or how fees are estimated
    lives here so they can be adjusted without touching business logic in other
    modules. Both the live trading pipeline (main.py) and the backtest pipeline
    (backtest.py, backtester.py) import from this file.

Dependencies:
    No project imports. Imported by auth.py, scanner.py, strategy.py, trader.py,
    reporter.py, historical.py, backtester.py, dashboard.py, backtest.py,
    scheduler.py, and main.py.

Notes:
    PROJECT_ROOT is derived from __file__ so the package works correctly on any
    machine regardless of where the repo is cloned.
    The sandbox URL (demo-api.kalshi.co) requires a completely separate account
    registered at demo.kalshi.co — the production API key will return 401 there.
"""
import math
import pathlib

# ── API base URLs ─────────────────────────────────────────────────────────────

# Production Kalshi REST API — requires a live account and real RSA credentials.
PROD_URL    = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi sandbox API — requires a SEPARATE account at demo.kalshi.co.
# The production API key is NOT accepted here; it will return 401.
SANDBOX_URL = "https://demo-api.kalshi.co/trade-api/v2"

# ── Filesystem paths ──────────────────────────────────────────────────────────

# Path to project root (where secrets.json and the PEM key live).
# Derived from __file__ so the package works on any machine after cloning.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# JSON file with API key IDs. Expected keys: "Kalshi-api-key" (prod) and
# optionally "dev_api_key" (sandbox). See README for the full format.
SECRETS_FILE = PROJECT_ROOT / "secrets.json"

# RSA private key PEM file used to sign Kalshi API requests.
PEM_FILE     = PROJECT_ROOT / "kalshi_private_key.pem"

# RSA private key PEM file for the sandbox (demo) account. Separate from prod
# because the sandbox account is registered independently at demo.kalshi.co.
DEV_PEM_FILE = PROJECT_ROOT / "kalshi_demo_private_key.pem"

# ── Trading parameters ────────────────────────────────────────────────────────

# Hard cap on the Kelly fraction allocated to any single trade. Even if the
# mathematical Kelly says to bet more, we never exceed 20% of the balance on one pair.
BUDGET_FRACTION               = 0.20

# Tiered minimum YES ask price difference for time-series pairs, keyed by the
# deadline gap between the two legs. The wider the deadline gap, the weaker the
# correlation assumption, so a larger price gap is required to justify the trade:
#   gap <= SHORT_DEADLINE_GAP_DAYS (15 days)  -> MIN_PRICE_DIFF_SHORT_GAP (15%)
#   gap 16..MAX_DEADLINE_GAP_DAYS  (30 days)  -> MIN_PRICE_DIFF_LONG_GAP  (30%)
# Use min_price_diff_for_gap() below to pick the tier — never hardcode these.
MIN_PRICE_DIFF_SHORT_GAP      = 0.15
MIN_PRICE_DIFF_LONG_GAP       = 0.30

# Inclusive boundary (in calendar days) between the two time-series price-gap
# tiers: deadline gaps up to and including this many days use the short tier.
SHORT_DEADLINE_GAP_DAYS       = 15

# Minimum YES ask price difference for same-title pairs. These are markets asking
# the exact same question, so even a small divergence (5%) is anomalous and worth trading.
SAME_TITLE_MIN_PRICE_DIFF     = 0.05

# Prior probability that a same-title pair co-resolves (i.e. both YES or both NO).
# Set at 95% — divergence is an anomaly, so we assume high correlation by default.
SAME_TITLE_CO_RESOLVE_PROB    = 0.95

# Maximum number of calendar days allowed between the deadlines of the two legs
# in a time-series pair. Pairs with a larger gap are too far apart in time to
# be reliably correlated.
MAX_DEADLINE_GAP_DAYS         = 30

# Minimum account balance in cents required to run the bot. Below $50 the bot
# aborts to avoid wasting API calls when there is insufficient capital to trade.
MIN_BALANCE_CENTS             = 5000

# Warn (never cap/drop) when a backtester pair-extraction group still has more
# than this many members after the eligibility prefilter and (for time-series
# groups) the close-time windowing. Purely observability — lets us confirm
# the O(n^2)-avoidance measures in backtester._extract_pairs are actually
# keeping group sizes tractable at current Kalshi market volumes.
LARGE_GROUP_WARN_THRESHOLD    = 1000

# Whether to include multivariate (multi-choice) markets in scanning and backtesting.
# When True, markets are grouped by (event_title + market_title) so cross-event
# option-label collisions (e.g. "Trump" in two unrelated events) cannot false-positive
# into a same-title or time-series pair. When False, mve_filter="exclude" is passed
# to all market-fetch APIs and the bot operates only on binary events.
INCLUDE_MVE_MARKETS           = True

# Kalshi taker fee rate. The exact per-leg fee is:
#   ceil(TAKER_FEE_RATE × n_contracts × price × (1 − price) × 100) / 100
# The quadratic P*(1-P) factor means fees are highest near 50¢ and lowest near 1¢/99¢.
TAKER_FEE_RATE                = 0.07

# Slippage allowance, in cents per contract, added on top of the scanned price
# when computing the buy_max_cost cap for each market FoK order leg. The cap
# protects against the order book moving between the pre-execution check and
# submission: the order fills at or below (scanned price + allowance) or not at all.
BUY_MAX_COST_SLIPPAGE_CENTS   = 1

# Maximum accepted per-contract loss (cents) when unwinding leg A after a
# failed leg B, relative to leg A's scanned NO entry price. The rollback is a
# fill-or-kill LIMIT sell at (entry - this), so a book that has collapsed past
# the floor kills the unwind instead of realizing an unbounded loss; the
# orphaned position then surfaces as status="rollback_failed" for manual
# review — the same path an unfilled market unwind already took.
ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT = 5

# Maximum seconds a scheduler-spawned bot run may take before being killed.
# Prevents a hung run (e.g. a network stall inside the SDK) from blocking the
# weekly scheduler daemon forever.
SCHEDULER_JOB_TIMEOUT_SECONDS = 3600

# ── Process exit-code contract ────────────────────────────────────────────────
# main.py's process exit code is the only signal the scheduler (a separate
# subprocess, per scheduler.run_job) has for what happened in a run beyond a
# generic pass/fail — a prod run skipped for insufficient balance used to exit
# 0 just like a clean run, so the scheduler logged "Job completed successfully."
# and the WARNING explaining why nothing happened was buried in a log the
# scheduler never reads (BS-14). These three codes are shared between main.py
# (which returns/exits them) and scheduler.py (which maps them to distinct log
# levels/messages) — they live in config.py so both modules import the same
# values instead of duplicating magic numbers. An unhandled exception in
# main.py is NOT covered here: it still propagates and the interpreter exits
# 1, same as always.
EXIT_OK                       = 0
EXIT_SKIPPED_LOW_BALANCE      = 10
EXIT_TRADES_NEED_ATTENTION    = 20

# ── API pagination ────────────────────────────────────────────────────────────

# Number of items to request per page when paginating market/event endpoints.
# The API rejects limits above 200 with HTTP 400 (observed 2026-07; it used to
# accept 1000), so this must stay <= 200.
MARKET_PAGE_SIZE   = 200

# Number of positions to request per page when paginating the /portfolio/positions endpoint.
POSITION_PAGE_SIZE = 500

# Cap on the number of multivariate-events pages the backtester's event-title
# lookup will scan (historical._load_or_build_event_titles). The MVE listing is
# effectively unbounded, so titles not found within this many pages fall back
# to bounded per-ticker /events/{ticker} lookups instead of paging for hours.
MVE_TITLE_LOOKUP_MAX_PAGES = 500

# Number of worker threads used by historical.fetch_all_settled_markets to
# fetch archive day-slices and live settled-day windows in parallel. The
# settled-market history is tens of millions of records at 1000/page (the
# API's hard page-size cap), so a sequential walk takes hours; sharding the
# fetch across workers was live-verified 2026-07-13 at 12 workers / ~20 req/s
# with zero HTTP 429s. Each worker's calls still go through
# api_call_with_retry, so if Kalshi does start throttling, the normal
# exponential backoff applies per worker. Raise cautiously.
SETTLED_FETCH_MAX_WORKERS = 8

# How many compact market dicts one day-slice worker buffers in memory before
# streaming them out to its slice file (historical._DayStreamWriter). A UTC day
# of settled Kalshi markets reached ~4.4–4.8M records in 2026-08, and each
# worker used to hold an entire day in one Python list before serializing it:
# live telemetry from the 2026-08-31 sweep showed RSS sawtoothing 3.4 → 4.3 →
# 7.89 GB on a 16 GB host as workers filled their day buffers concurrently.
# Chunking bounds a worker's buffer at this many records regardless of how big
# the day is, so peak memory scales with (workers x chunk), not with day size.
# Flushes land on API-page boundaries, so the true bound is this plus one page
# (<= 1000 records). Bigger chunks amortize the per-write overhead slightly;
# smaller ones bound memory tighter. 50k records is roughly 50 API pages, i.e.
# a few tens of MB.
SETTLED_FETCH_CHUNK_RECORDS = 50_000

# Number of worker threads used by the backtester's per-ticker candlestick
# fetch (backtester._fetch_candles_parallel). Each fetch is an independent
# read-only GET routed through api_call_with_retry, so a 429 degrades to that
# worker's own exponential backoff rather than failing the run — live-verified
# 2026-08-03 at ~20 req/s with 12 workers and zero HTTP 429s. Fetching
# sequentially was live-measured the same day at ~4.3 tickers/sec, i.e. ~34
# hours for a 3-month window, which makes this loop the dominant cost of a
# backtest. Cache files are keyed per ticker, so two workers can never target
# the same path — _save_json_cache writes atomically (tmp+replace), but its tmp
# name is derived from the destination, so a shared cache path would still
# collide; never introduce a fetch whose cache path is shared across workers. Each worker keeps
# fetch_candlesticks' own rate_limit_sleep default (0.15s) between its pages.
CANDLESTICK_FETCH_MAX_WORKERS = 8

# Number of worker threads used by trader.py for both of its pools: the
# pre-execution order-book re-checks (pre_execution_check) and the per-pair
# execution of the selected portfolio (execute_trades). Each pool is sized
# min(this, len(work)), so a small portfolio never over-provisions threads.
# The portfolio is at most a handful of pairs in practice, so this is a
# ceiling rather than a tuned throughput figure; unlike the fetch pools it has
# never been exercised at scale against the live API. Raise cautiously — the
# execution pool submits real orders, so each extra worker is another
# concurrent write against the account.
TRADER_MAX_WORKERS = 8

# Names the SEMANTICS of backtester._can_ever_enter(), which run_backtest()
# passes to historical.fetch_all_settled_markets() as a prefilter so ineligible
# markets are dropped during assembly instead of being held in memory and
# written to the assembled cache. The tag is part of that cache's filename
# (settled_markets_<start_date>_<tag>.json), so a cache built under one filter
# can never be served to code expecting another.
#
# MUST be bumped whenever _can_ever_enter's behaviour changes — otherwise a
# stale prefiltered cache is silently reused and the backtest sees a market set
# the current predicate would not have produced.
SETTLED_PREFILTER_CACHE_TAG = "monday-eligibility-v1"

# Hard cap on the per-ticker event-title fallback in
# historical._load_or_build_event_titles. That fallback exists for the handful
# of archived events the bulk listings no longer carry, and it costs ONE HTTP
# GET per ticker. At current Kalshi volumes a 3-week backtest can reach the
# fallback with hundreds of thousands of unresolved tickers (live-measured
# 2026-08-03: 289,235 unique event_tickers for a 21-day window) — sequentially
# that is many hours with no visible progress, which reads as a hang.
#
# Tickers past the cap are recorded as "" (the same poison pill used for a
# failed lookup): the backtester then treats those markets as ungrouped, which
# is the identical outcome a failed lookup already produced. Correctness is
# unaffected; only MVE grouping coverage degrades, and the log says by how much.
EVENT_TITLE_FALLBACK_MAX_LOOKUPS = 5_000

# Worker threads for the per-ticker event-title fallback. Each lookup is an
# independent read-only GET, so this is pure I/O overlap — the same rationale
# (and the same retry-per-worker behaviour) as SETTLED_FETCH_MAX_WORKERS.
EVENT_TITLE_FALLBACK_MAX_WORKERS = 8

# Abandon a bulk event listing after this many CONSECUTIVE pages that resolve
# no new titles. Same "productivity bail-out" idiom as MVE_MAX_EMPTY_PAGES.
#
# The bulk listings are an O(all events) scan looking for a specific ticker
# set, so they only pay off while they keep hitting wanted tickers. Live-
# measured 2026-08-03 on a 21-day window: the `settled` listing resolved 8,696
# of 289,235 tickers in its first ~500 pages, then just 9 more over the next
# 400 pages — because the overwhelming majority of those tickers are
# auto-generated MVE collection events, which get_events EXCLUDES by API
# design and therefore can never return. Without this bail-out the phase keeps
# paging a listing that structurally cannot contain what it is looking for,
# for all three statuses, before the MVE listing is even reached.
EVENT_TITLE_LISTING_MAX_BARREN_PAGES = 50

# Stop the multivariate-events pull after this many CONSECUTIVE pages that
# contain no nested markets. The MVE listing is effectively unbounded (Kalshi
# auto-generates hundreds of thousands of collection events) and as of 2026-07
# the API returns zero nested markets on it regardless of with_nested_markets —
# without this cap the scan pages forever (observed: 75+ minutes, no end).
# Pages that DO contain markets reset the counter, so real MVE coverage
# resumes automatically if the API starts sending nested markets again.
MVE_MAX_EMPTY_PAGES = 25

# Stop an archive walk (historical._fetch_archive_tail and
# _fetch_archive_sequential) after this many CONSECUTIVE pages containing zero
# settlements inside the backtest window.
#
# /historical/markets is ordered by created_time DESC, ticker DESC — NOT by
# settlement time — so no EXACT stop rule exists: a market created arbitrarily
# early can settle arbitrarily late, i.e. inside the window. The walks used to
# stop at the first page whose FIRST record (the newest-CREATED one on that
# page) settled before start_ts, which is not a valid proof that deeper pages
# hold nothing: it silently dropped long-lived in-window settlers, and because
# most Kalshi markets are short-lived it typically fired after one or two
# pages. Since correctness can't be proven, bound the walk by productivity
# instead — the same idiom as EVENT_TITLE_LISTING_MAX_BARREN_PAGES and
# MVE_MAX_EMPTY_PAGES. At the archive's 1000-record page cap, 50 consecutive
# barren pages is ~50k records of created-time depth searched past the last
# page that produced anything.
ARCHIVE_MAX_BARREN_PAGES = 50

# Emit a progress log line every this many pages in scanner.py's three
# pagination loops (fetch_open_events_with_markets's standard-events and MVE
# loops, get_held_tickers). A live dev-mode run paged 125,538 sandbox markets
# in 13m27s with zero log lines in kalshi_arb.log — indistinguishable from a
# hang, the exact misdiagnosis class the sharded historical fetcher's
# "[sharded]"/"[windowed]" progress labels exist to prevent (see the
# sharded-fetch gotcha in CLAUDE.md). Purely a logging cadence, not a bound.
SCANNER_PROGRESS_LOG_EVERY_PAGES = 25

# ── Backtest candlestick granularity ────────────────────────────────────────

# Minutes per candle requested from /historical/markets/{ticker}/candlesticks.
# Daily (1440) only emits a bar for a market whose lifespan crosses a UTC
# midnight boundary — confirmed 2026-07 by direct API testing: a 2-hour-long
# market entirely within one day returned 0 daily candles but 2 hourly ones.
# Most Kalshi markets are single-game/few-hour windows within one calendar
# day, so daily granularity structurally produced zero price data for most
# markets regardless of liquidity, which silently zeroed out backtest entries.
# 60 (hourly) is the finest granularity actually available — period_interval=1
# (minute) returns HTTP 400.
CANDLESTICK_PERIOD_INTERVAL_MINUTES = 60


def min_price_diff_for_gap(gap_days: int) -> float:
    """
    Return the minimum time-series YES price gap required for a deadline gap.

    Picks the price-gap tier for a time-series pair based on how many calendar
    days separate the two legs' deadlines: gaps up to SHORT_DEADLINE_GAP_DAYS
    (15 days, inclusive) require MIN_PRICE_DIFF_SHORT_GAP (15%); anything wider
    requires MIN_PRICE_DIFF_LONG_GAP (30%). Callers must already have enforced
    gap_days <= MAX_DEADLINE_GAP_DAYS — this helper only selects the tier and
    does not reject over-cap gaps itself.

    Args:
        gap_days (int): Calendar days between the two legs' deadlines.
            Range: 0..MAX_DEADLINE_GAP_DAYS (caller-enforced).

    Returns:
        float: The minimum required YES ask price difference (dollars, 0-1).
    """
    if gap_days <= SHORT_DEADLINE_GAP_DAYS:
        return MIN_PRICE_DIFF_SHORT_GAP
    return MIN_PRICE_DIFF_LONG_GAP


def fee_per_pair_approx(nA: float, pB: float) -> float:
    """
    Compute a continuous approximation of the total Kalshi taker fee for one pair trade.

    Used during pair filtering and Kelly sizing (before the exact integer contract
    count is known). The exact fee formula uses ceiling rounding per leg; this
    approximation treats the contract count as a continuous quantity, making it
    suitable for threshold comparisons.

    The formula is: TAKER_FEE_RATE * (nA*(1-nA) + pB*(1-pB)), which sums the
    quadratic fee contribution from the NO leg (market A) and the YES leg (market B).

    Args:
        nA (float): NO ask price of market A in dollars. Range: (0, 1).
        pB (float): YES ask price of market B in dollars. Range: (0, 1).

    Returns:
        float: Approximate total taker fee per contract pair (in dollars).
            This is an underestimate relative to the exact ceiling formula —
            i.e. it is OPTIMISTIC (makes pairs look slightly more profitable
            than they actually are at small sizes). That is why it must only
            be used for filtering: final validation always re-checks with
            fee_leg_exact() so an underestimated fee cannot admit a bad trade.
    """
    return TAKER_FEE_RATE * (nA * (1.0 - nA) + pB * (1.0 - pB))


def fee_leg_exact(n: int, p: float) -> float:
    """
    Compute the exact Kalshi taker fee for one order leg of n contracts at price p.

    Kalshi applies a ceiling rounding per leg: the fee is rounded up to the nearest
    cent. This means small positions are slightly over-charged relative to the
    continuous approximation. Use this function once the final contract count n is
    known (e.g. in strategy.py and backtester.py).

    Args:
        n (int): Number of contracts for this leg. Should be >= 1.
        p (float): Price in dollars for this leg (YES price for the YES leg,
            NO price for the NO leg). Range: (0, 1).

    Returns:
        float: Taker fee in dollars, rounded up to the nearest cent.
    """
    # Round to 6 decimals before the ceiling so binary floating-point noise
    # (e.g. 0.07*100*0.25*100 = 175.00000000000003) cannot bump an exact
    # cent amount up an extra cent — Kalshi charges ceil of the TRUE value.
    return math.ceil(round(TAKER_FEE_RATE * n * p * (1.0 - p) * 100, 6)) / 100
