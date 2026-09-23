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
    scheduler.py, and main.py — plus the standalone, human-run verification
    CLI kept deliberately outside the pipeline's import graph (see CLAUDE.md's
    pipeline-isolation rule).

Notes:
    PROJECT_ROOT is derived from __file__ so the package works correctly on any
    machine regardless of where the repo is cloned.
    The sandbox URL (demo-api.kalshi.co) requires a completely separate account
    registered at demo.kalshi.co — the production API key will return 401 there.
"""
import math
import pathlib
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

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

# Collision-suffix cap for create_new_output(). Past this many outputs competing
# for one name, the helper takes a uuid tail rather than looping further: the cap
# bounds a pathological retry, not how many outputs an operator may keep.
OUTPUT_NAME_MAX_ATTEMPTS = 100

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

# Defensive ceiling on strategy.compute_trade's marginal-price descent. That
# loop re-prices a pair at each candidate contract count and takes the SMALLER
# of the count it started with and the one Kelly then allows, so the sequence is
# strictly decreasing and terminates on its own — this cap only exists so a
# future edit that breaks the decrease invariant surfaces as one WARNING and a
# skipped pair rather than a hung weekly run. Convergence takes 1-3 passes in
# practice; 64 strict decreases without settling means a pathological book, and
# refusing to trade it is the safe answer.
SIZE_SOLVE_MAX_ITERATIONS     = 64

# Tiered minimum YES ask price difference for time-series pairs, keyed by the
# deadline gap between the two legs. The LATER contract's YES ask must
# exceed the earlier's by at least the tier — later/earlier by close_time,
# or by STATED deadline for a same-event ladder (DR-73): that gap is the market-implied
# probability that the event first happens BETWEEN the two deadlines, which is
# the trade's single loss scenario. A wider deadline gap leaves more time for
# exactly that, so more of the market's in-between mass is genuine and a
# bigger price gap is demanded before the strategy disputes it:
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
# in a time-series pair. The wider the gap, the more of the market-implied
# in-between probability (the later YES ask minus the earlier) is genuine
# rather than mispricing; past 30 days there is too much room for the event to
# land between the deadlines for the trade to dispute the market's number.
MAX_DEADLINE_GAP_DAYS         = 30

# Whether the LIVE time-series finder may pair two rungs of ONE event's
# cumulative deadline LADDER (DR-73; the backtester mirrors it in DR-73c).
# Kalshi often lists a question's several deadlines as separate markets
# inside a SINGLE event — "Will SpaceX launch another Starship
# by Sep 23, 2026?" and "... by Oct 16, 2026?" are both KXSPACEXSTARSHIP-14 —
# and both finders have refused every same-event candidate since the first
# commit, on the rationale that a shared event ticker means multi-choice
# OPTIONS. That is true of an MVE event's option labels and false of a dated
# ladder, whose two rungs are the time-series premise itself: the earlier
# deadline's event nests inside the later one's. A ladder pair is ordered and
# tiered on the two STATED deadlines (scanner.stated_deadline /
# same_event_ladder), never on close_time, which a settled or single-instant
# event gives every rung alike.
#
# READ THIS BEFORE FLIPPING IT. What the switch buys and what it puts at risk,
# measured rather than assumed:
#
#   Nesting holds for ladders, and does not for what we trade today. Over 284
#   cached day slices (257 archive, 27 live), 1,821 same-event cumulative
#   pairs read to two different stated deadlines and the impossible
#   A=YES/B=NO cell occurs 0
#   times (0 of the 975 within MAX_DEADLINE_GAP_DAYS). The CROSS-EVENT
#   baseline on the same corpus is 2,872 of 22,080 — 13.01%. The premise
#   violations in the archive come from the pairs this finder admits TODAY.
#
#   Live funnel (2026-09-22 snapshot, 113,303 markets, on which the finder
#   emits 0 time-series and 0 same-title pairs with the switch off): 3,354
#   same-event candidates -> 2,840 past the identical-wording check (-502)
#   and the cumulative-wording one (-12 snapshot) -> 2,774 reading as two
#   different calendar days (-31 undated, -35 field conflict) -> 350 within
#   the 30-day stated-gap cap -> 89 past the price tier -> 87
#   past the pA + nB < $1 guard -> 24 pairs emitted, in 24 events and 23
#   series, all tradeable.
#
#   Exposure. strategy.compute_trade + select_portfolio over those 24 pairs at
#   a $10,000 balance size 17 trades and SELECT 6, deploying $9,803.66 — 98%
#   of the balance — with an aggregate market-implied EV of -$3,039.94, i.e.
#   -31% of what is deployed. Every one of the 24 is market-EV-negative, and
#   NOT because of k: at market prices this structure's EV per contract pair
#   is pA + (1 - pB) - (pA + nB) = 1 - pB - nB, which is exactly MINUS the
#   LATER leg's own bid-ask spread and contains neither pA nor k. Of the
#   -$3,039.94, -$2,455.23 is that spread (4-9c crossed on 5-12c contracts)
#   and -$584.71 is taker fees; the two sum to the total exactly.
#   Re-calibrating k changes WHICH pairs are selected and how large, never
#   this rate — a spread or liquidity screen is the lever that would. The
#   capital concentrates on the widest spreads, which is Kelly behaving correctly (the haircut is worth
#   0.25 x (pB - pA) in absolute probability, so a 0.94 spread carries a
#   23.5-point claimed edge against a 12c stake) — FOUR of the six sit exactly
#   at the BUDGET_FRACTION cap and a fifth at f* = 0.17, the five together
#   deploying $9,716.81. BUDGET_FRACTION caps each PAIR, not the portfolio,
#   and under the MODEL's own probabilities those five lose together 12.7% of
#   the time IF THE FIVE UNDERLYINGS ARE INDEPENDENT — the figure is the
#   PRODUCT of the five marginal loss probabilities, and nothing here models
#   correlation, which can only raise it (53.5% at market prices, by the same
#   product). Each loses its full stake in that cell.
#
#   Selection effect. The one-best-pair-per-group rule picks the LARGEST
#   pB - pA in a group, and a stale quote is by definition one out of line
#   with its neighbours, so the pair containing it has the inflated spread and
#   is the one selected. It is not hypothetical: 45 of the 300 live events
#   holding >= 2 dated cumulative rungs on two or more DISTINCT days price a
#   LATER deadline BELOW an earlier one, which is impossible if the rungs
#   nest, so at least one quote in each is wrong (45 of 484 counting every
#   event with >= 2 dated rungs, the other 184 of which name one day apiece
#   and so cannot be non-monotone at all; 42 of the 251 same-event GROUPS the
#   one-best rule actually contests). Those are counts of EVENTS and GROUPS,
#   not of emitted pairs: on the same snapshot NONE of the 24 emitted pairs
#   has an intervening rung of its own event priced outside [pA, pB], so the
#   guard below would change nothing today — the effect is latent in the
#   population, not present in the current selection. An intervening-rung
#   staleness guard is the natural answer and is deliberately not built here.
#
# Off by default until the interval discount is calibrated where the capital
# actually goes — k-hat in the widest (pB - pA > 0.60) band, not pooled: p and
# b depend on the spread, not on the gap in days, and the only k-hat ever
# computed (1.114) was measured on snapshot pairs DR-67 refuses and is void.
#
# BOTH PATHS IMPLEMENT THIS since DR-73c: backtester._extract_pairs forms the
# same pairs from a per-event sub-pass and _find_entry orders and gaps them on
# the same stated deadlines, so a ladder-enabled backtest measures the strategy
# a ladder-enabled live run would trade — with the one standing caveat the code
# already records at backtester._simulate_at_discount's one-best dedup: the
# backtest's one-best-per-group winner is the largest entry_monthly_ratio, not
# the live finder's tradeable-then-largest-gap, so the two paths can replay
# DIFFERENT rungs of the same ladder (on the 2026-09-22 snapshot the live
# funnel narrows 87 eligible ladder candidates to 24 emitted, so that contest
# decides 63 of them). backtest.py's
# --same-event-ladders / --no-same-event-ladders overrides this constant for
# ONE run, which is how the k-hat the gate above demands gets measured without
# flipping the switch first; scanner.py binds the constant at import, so that
# override never reaches the live finder. The backtest's HTML dashboard does
# NOT render the setting — it reaches kalshi_backtest.log only — so a
# ladder-enabled run's dashboard is indistinguishable from a switch-off one and
# must be labelled by hand (recorded, not fixed: dashboard.py is outside DR-73's
# blast radius).
#
# scanner.py, backtester.py AND backtest.py each bind this by VALUE at import
# (the SCANNER_MAX_PAGES idiom), so a test or harness flipping it at runtime
# must patch the constant on the MODULE it wants to affect — scanner for the
# live finder, backtester for _extract_pairs/_find_entry, backtest for the CLI
# echo — and never on this module: patching config here is a silent no-op that
# reads as a switch-ON run and produces a switch-OFF result. This is NOT the
# config.time_series_profit_prob(k=None) idiom, which works only because that
# helper lives here and reads THIS module's global; the sentinel arguments named
# same_event_ladders resolve their own module's binding at call time, which is
# what makes a run-level override and a module monkeypatch take effect where a
# def-time default would not. For a backtest the supported lever needs no
# patching at all: run_backtest_sweep(same_event_ladders=...) or
# backtest.py --same-event-ladders / --no-same-event-ladders.
TIME_SERIES_SAME_EVENT_LADDERS = False

# ── Time-series strategy model (2026-09 inversion) ────────────────────────────
#
# A time-series pair buys YES on the EARLIER contract (market_a) and NO
# on the LATER one (market_b) — earlier/later by close_time, or by STATED
# deadline for a same-event ladder (DR-73) — when the later contract's YES ask exceeds the
# earlier's by at least the deadline-gap tier, and when BOTH legs are worded as
# cumulative "by <date>" deadlines (scanner.deadline_phrasing) — only then does
# the earlier deadline's event nest inside the later one's, which is what makes
# the model below meaningful at all. The market-implied probability
# that the event first happens BETWEEN the two deadlines is (pB - pA); that is
# the trade's single loss scenario (earlier NO, later YES). This constant is the
# fraction of that market-implied in-between mass we believe — 0.75 means "the
# market overstates it by a quarter; prices will converge by 25%". It is an
# operator-tunable ESTIMATE, not a measured quantity: at 1.0 (take the market at
# face value) the Kelly fraction is <= 0 for every candidate and the strategy
# never fires; smaller values size more aggressively. Measure it against
# settled history with `backtest.py --interval-discount K` (overrides k for
# that backtest run only; this constant is what live sizing always reads) and
# read the dashboard's "Interval Discount (k) Calibration" section, or the
# calibration block in kalshi_backtest.log — see CLAUDE.md, "Interval-discount
# calibration (2026-09 follow-up)" for the full mechanism.
TIME_SERIES_INTERVAL_PROB_DISCOUNT = 0.75

# Grid of k values backtester.run_backtest_sweep() re-simulates so the dashboard
# can offer a k selector without a re-run. Spans "size very aggressively" (0.40)
# through "take the market at face value" (1.00, where Kelly is <= 0 for every
# pair and nothing trades — the boundary is informative, so it stays in). Each
# point costs one extra sizing+selection pass over already-fetched candidates;
# the market fetch and candlestick fetch happen once regardless.
INTERVAL_DISCOUNT_SWEEP = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

# Which side each leg of a pair buys, as (side bought on market_a, side bought
# on market_b). scanner.leg_sides() is the ONLY reader — never hardcode a side
# elsewhere. Same-title: NO on the pricier contract (market_a), YES on the
# cheaper (market_b). Time-series: YES on the earlier contract (market_a), NO on
# the later (market_b). The trader always SUBMITS the NO leg first, whichever
# market it sits on (see trader._ordered_legs).
SAME_TITLE_LEG_SIDES  = ("no", "yes")
TIME_SERIES_LEG_SIDES = ("yes", "no")

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
# to all market-fetch APIs, the backtester's MVE event-title listing is skipped,
# the assembled backtest cache filename gains a trailing "_nomve" marker (DR-57 —
# the flag changes WHAT IS FETCHED, so a cache built under one setting must never
# be served to a run under the other; only the False case is marked, so the
# default True keeps the pre-DR-57 filename and no cached assembly is orphaned),
# and the bot operates only on binary events. Event titles are still resolved for
# binary events in both modes, so live and backtest grouping keys match.
INCLUDE_MVE_MARKETS           = True

# Series-prefix FAMILY that scanner.event_series() collapses onto one series
# (DR-55). Kalshi lists its multi-leg COMBO (parlay) markets under SEVERAL
# series prefixes, not one. A combo ticket's wording names its legs but never
# its date, so one wording recurs across fixture instances and two tickets with
# identical leg wording are two DIFFERENT tickets — exactly the shape the
# DR-02/DR-54 one-series rule exists to refuse. event_series() used to return
# the LITERAL prefix, which DID refuse two combos listed under ONE KXMVE*
# prefix; what it could not see was a pair spanning TWO of them, which read as
# two different series and was priced on the 0.95 SAME_TITLE_CO_RESOLVE_PROB
# co-resolution prior (and, identically worded, as a time-series pair).
#
# Census of backtest_cache/event_titles.json (3,996,906 keys, 2,444 distinct
# prefixes), measured 2026-09-16. Reproduce with:
#   python3 -c "import json,collections;c=collections.Counter(k.split('-')[0] for k in json.load(open('backtest_cache/event_titles.json')));print([(k,v) for k,v in c.most_common() if k.startswith('KXMVE')])"
#   KXMVECROSSCATEGORY            2,960,840
#   KXMVESPORTSMULTIGAMEEXTENDED    906,157
#   KXMVECROSSCATEGORY0              94,230
#   KXMVENBASINGLEGAME                   61
# No KXMV* prefix exists in that file that is not also KXMVE*.
#
# The cross-prefix exposure is measured, not assumed. In
# backtest_cache/live_days/2026-09-14.json.gz (9,009,087 records, 8,939,229 of
# them KXMVE*), 10,643 distinct (event_title, title, subtitle) wordings appear
# under TWO literal KXMVE prefixes — every one of them KXMVECROSSCATEGORY x
# KXMVECROSSCATEGORY0. Those wordings form 26,072 cross-prefix market pairs, of
# which 17,829 co-resolved and 8,243 settled DIFFERENTLY: a 68.4%
# co-resolution rate against the 0.95 prior the same-title finder would have
# priced them on.
#
# A FAMILY PREFIX rather than an enumerated allowlist, deliberately. Enumerating
# the combo series would BE an allowlist, and the codebase's own prior belief
# ("every combo sits under KXMVECROSSCATEGORY") was already such an allowlist
# with one of the four entries — so an allowlist is exactly the artefact that
# was measurably wrong here, and it rots every time Kalshi adds a combo series
# (KXMVENBASINGLEGAME, 61 keys, does not appear in the 2026-09-14 day slice at
# all — a family that is barely listed today and could be listed in bulk
# tomorrow).
# Over-collapsing is the SAFE direction because _same_series() only ever
# REFUSES a pair: the worst case is a missed trade between two genuinely
# independent KXMVE* series, or — where the refused pair was a group's best
# candidate — a different, still-eligible runner-up promoted in its place,
# never a trade priced on a premise that does not hold. The family literal is
# deliberately narrow: prefix containment among ordinary series is common
# (the same census holds KXART, KXARTISTSTREAMS and KXARTISTVS, which are
# unrelated series), so this must not later be widened to KXMV or to a generic
# containment rule.
MVE_SERIES_FAMILY_PREFIX      = "KXMVE"

# Kalshi taker fee rate. The exact per-leg fee is:
#   ceil(TAKER_FEE_RATE × n_contracts × price × (1 − price) × 100) / 100
# The quadratic P*(1-P) factor means fees are highest near 50¢ and lowest near 1¢/99¢.
TAKER_FEE_RATE                = 0.07

# LEGACY ORDER PATH ONLY — the V2 order path uses BUY_SLIPPAGE_TICKS below.
# Slippage allowance, in cents per contract, added on top of the scanned price
# when computing the buy_max_cost cap for each market FoK order leg. The cap
# protects against the order book moving between the pre-execution check and
# submission: the order fills at or below (scanned price + allowance) or not at all.
# This stays a whole-cent value because `buy_max_cost` is an integer-cents field
# on the legacy /portfolio/orders create-order endpoint — a sub-cent-aware cap
# can't be expressed there no matter how finely a market's own tick grid is
# subdivided (see ApiMarket.price_level_structure / price_ranges in scanner.py).
# On 2026-08-17, all MVE/combo markets migrated to the
# `center_deci_edge_centi_cent` tick regime — $0.0001 ticks below $0.01 and
# above $0.99, $0.001 ticks in between — so this 1c tolerance permits roughly
# 10-100 ticks of price drift on those markets, depending on where in the band
# the price sits, rather than the intended ~1. That is precisely why the
# default order path is now ORDER_API_VERSION = "v2" (below), whose dollar-
# string limit price is capped in ticks; this constant only still applies when
# that switch is flipped back to "legacy" as a rollback.
BUY_MAX_COST_SLIPPAGE_CENTS   = 1

# Maximum accepted per-contract loss (cents) when unwinding the NO leg (the
# first-submitted leg: market_a for a same-title pair, market_b for a
# time-series pair) after the YES leg failed, relative to the NO leg's scanned
# NO entry price. The rollback is a fill-or-kill LIMIT sell at (entry - this),
# so a book that has collapsed past the floor kills the unwind instead of
# realizing an unbounded loss; the orphaned position then surfaces as
# status="rollback_failed" for manual review — the same path an unfilled
# market unwind already took.
#
# This allowance must cover the market's ENTIRE bid-ask spread, not just the
# "acceptable loss": the NO leg entered at the NO ASK, but the unwind is a sell
# that only fills against the NO BID, so (NO ask - NO bid) — the spread
# itself — is a floor on the loss even with zero adverse price movement.
# Any adverse move since entry is additive on top of that spread. At 5 cents
# this was narrower than the spread on the illiquid markets this strategy
# targets, so killed unwinds (rollback_failed orphans) were the normal
# outcome, not the tail case. 12 cents lets a normal-spread book fill the
# unwind while a genuinely collapsed book still kills it and surfaces
# rollback_failed for manual review — the deliberate bounded-loss trade-off.
#
# The floor applies to BOTH order paths. On the legacy path it is the NO limit
# sell price directly. On the V2 path a held NO position is a short YES, so the
# unwind is a YES BUY and the same bound becomes a bid CEILING of
# (1 - floor/100) dollars, ceiling-quantized onto the market's tick grid and
# clamped by V2_ROLLBACK_BID_PRICE_DOLLARS (see trader._v2_rollback_price(no_leg)).
ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT = 12

# Slippage allowance for the V2 order path, denominated in TICKS of the market's
# own price grid rather than in whole cents. The V2 endpoint
# (/portfolio/events/orders) takes dollar-string limit prices, so a FoK cap can
# finally be expressed at the market's real resolution: cap = scanned price +
# BUY_SLIPPAGE_TICKS × tick size, where the tick size comes from
# scanner.tick_size_for_price(). One tick restores the original intent of
# BUY_MAX_COST_SLIPPAGE_CENTS = 1, which meant ~1 tick back when every market
# was on a 1c grid but means roughly 10-100 ticks on the centi-cent regimes
# MVE/combo markets migrated to on 2026-08-17.
BUY_SLIPPAGE_TICKS            = 1

# Fallback tick size, in dollars, for a market whose tick structure is unknown
# or uniform-cent ("linear_cent", or no price_ranges bands at all). Kalshi's
# tick grids are nested ($0.01 ⊂ $0.001 ⊂ $0.0001), so the coarsest grid is
# always a safe, always-valid fallback. Stored as a dollar STRING and parsed
# with Decimal at the point of use — same rule as the balance parsing in
# auth.py: float literals reintroduce exactly the binary representation noise
# the API's dollar-string fields exist to avoid.
DEFAULT_TICK_SIZE_DOLLARS     = "0.01"

# The extreme tradeable price levels on Kalshi's FINEST grid ($0.0001 ticks,
# the center_deci_edge_centi_cent edge bands). Used by
# scanner._bids_to_ask_levels to decide which ORDER-BOOK LEVELS are real
# quotes: a level's complement outside this range is a settled or nonsensical
# price, not depth.
#
# Deliberately NOT the same bound as scanner's market-eligibility check
# (_MIN_ACTIVE_PRICE/_MAX_ACTIVE_PRICE, still 0.01/0.99), and the two must not
# be unified. Eligibility asks "is this MARKET worth trading at all", where a
# sub-cent YES ask means a near-settled market and admitting it would make a
# 0.9999 quote into a $0.0001 hedge leg. This bound asks "is this LEVEL a real
# quote on a market we already accepted" — and on the deci-cent and
# centi-cent regimes, whose entire point is sub-cent ticks, the old 0.01/0.99
# level bound silently discarded genuine depth (TS-14).
MIN_ACTIVE_PRICE_DOLLARS      = 0.0001
MAX_ACTIVE_PRICE_DOLLARS      = 0.9999

# Tolerance for comparing two contract prices for equality-or-better.
#
# Every price in the pipeline is a float parsed from a cent-quantized dollar
# string, so exact arithmetic on them does not hold: 0.35 - 0.30 evaluates to
# 0.04999999999999999, and a bare `< 0.05` therefore REJECTS a pair that sits
# exactly on the documented 5% threshold. Measured over live books: the
# same-title 5c test rejected 50 of 94 qualifying pairs, the time-series 15c
# tier 21 of 84, and the 30c tier 15 of 69 (TS-09).
#
# 1e-6 is two orders of magnitude below the FINEST tick any regime uses
# ($0.0001), so it can only absorb representation noise — never a real price
# difference, which is at least one tick. DO NOT TUNE UPWARD: this is an
# ABSOLUTE tolerance, so its weight relative to the quantity being compared
# grows as that quantity shrinks, and a larger value would start admitting
# genuinely sub-threshold pairs at the bottom of the book.
PRICE_EPSILON                 = 1e-6

# Which create-order endpoint trader.py submits through. Allowed values:
#   "v2"     — POST V2_ORDER_PATH below: dollar-string fill-or-kill LIMIT prices
#              (the limit price IS the price protection), fixed-point counts,
#              bid/ask sides on the single YES book, explicit exchange_index.
#              Only this path can express a cap at the market's real tick
#              resolution (see BUY_SLIPPAGE_TICKS above).
#   "legacy" — the original /portfolio/orders create-order call
#              (CreateOrderRequest, type="market", integer-cents buy_max_cost
#              via BUY_MAX_COST_SLIPPAGE_CENTS).
# The legacy path is retained UNMODIFIED in trader.py purely so flipping this
# constant to "legacy" is the instant rollback procedure if the first live or
# sandbox V2 submission misbehaves — no code change, no redeploy of logic.
# Default is "v2" because the legacy endpoint is past its "no earlier than
# 2026-05-06" deprecation window and costs 5x rate-limit tokens per request.
# Note that dev/sandbox V2 support is UNVERIFIED (dev mode never submits
# orders), so the first real production submission is the true verification of
# the V2 request/response mapping — see the V2 gotcha in CLAUDE.md.
ORDER_API_VERSION             = "v2"

# Full API path of the V2 create-order endpoint, including the /trade-api/v2
# prefix. A constant (not an inline literal) because the path is signed as part
# of every request — _http.signed_request_json signs timestamp + method + path,
# so the string used to build the URL and the string that is signed must be one
# and the same value.
V2_ORDER_PATH                 = "/trade-api/v2/portfolio/events/orders"

# TOP-OF-GRID CEILING CLAMP, as a dollar string, on the V2 reduce-only rollback
# bid that unwinds a filled NO leg (market_a for same-title, market_b for
# time-series — see trader._ordered_legs). On the single-YES-book model a held
# NO position is a short YES, so closing it is a YES BUY (bid).
#
# This is NOT the price submitted. The submitted bid is the bounded-loss
# ceiling derived from ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT — (1 - floor/100),
# ceiling-quantized onto the market's own tick grid — and this constant is only
# the upper clamp applied to it (trader._v2_rollback_price(no_leg), via
# min(derived_cap, this_clamp)). With the current ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
# value and _rollback_floor_cents(no_leg)'s [1, 99]-cent range, the derived cap can
# never exceed 0.99 — i.e. this clamp cannot actually bind today, since 0.99
# is already <= every regime's top-of-grid level. It is kept as a defensive
# invariant (a future change to the floor's bound could otherwise push the
# cap above a tradeable level) and it defines what "as aggressive as
# possible" means IF the clamp ever does bind: this is the finest-grid target
# ($0.0001 ticks), floored onto the grid of the market actually being
# unwound (0.99 on linear-cent, 0.999 on deci-cent, 0.9999 on a centi-cent
# edge band) — a flat 0.99 would fail to cross asks resting in (0.99, 1) on
# sub-cent regimes. Deliberately not "1" — that is a settlement value, not a
# tradeable level.
V2_ROLLBACK_BID_PRICE_DOLLARS = "0.9999"

# The DEFAULT exchange shard. Kalshi partitions the exchange into parallel
# instances keyed by `exchange_index` (on markets and in the balance breakdown;
# combos migrated to shard 1 on 2026-08-17, crypto to shard 2 and
# tennis/baseball to shard 3 on 2026-08-24). "Default" carries three
# path-independent meanings, which is why this is not named "routable" —
# routability depends on the order path (the legacy endpoint reaches only this
# shard; V2 takes an explicit per-order exchange_index):
#   1. the shard assumed when a market payload omits `exchange_index`
#      (fail-safe — absence of the field must never drop markets);
#   2. the shard the legacy/sandbox single-scalar balance shapes are
#      attributed to (auth.py fallback tiers 2-3);
#   3. the only shard the legacy order path may route to.
DEFAULT_EXCHANGE_INDEX       = 0

# The JSON re-typings of an /exchange/status boolean that scanner._status_flag()
# is allowed to RECOGNISE, matched case-insensitively after .strip(). Kalshi has
# already retyped or dropped required fields on markets, positions, orders,
# events and balance; the retyping that silently INVERTS a halt flag is the
# string "false", because Python's bool("false") is True — a halted shard would
# read as open and keep being scanned and traded. The NULL tokens are the
# stringified spellings of a JSON null, which are truthy strings for the same
# reason and carry no more information than an absent key, so they resolve to
# "unknown" exactly as an absent key does. The empty string is deliberately in
# NONE of these sets: bool("") is already False, and re-reading it as unknown
# would UN-DROP a halted shard.
EXCHANGE_FLAG_FALSE_TOKENS   = frozenset({"false", "f", "no", "n", "off", "0"})
EXCHANGE_FLAG_TRUE_TOKENS    = frozenset({"true", "t", "yes", "y", "on", "1"})
EXCHANGE_FLAG_NULL_TOKENS    = frozenset({"null", "none", "nil", "undefined"})

# Maximum characters of a drifted flag value's repr() in the drift WARNING
# scanner.fetch_shard_statuses() emits. The raw value is whatever the API sent,
# so an unbounded repr of (say) a 300-element array is a multi-KB log line
# repeated every run — the same per-line bloat TS-02 removed from the candlestick
# and event-title paths. 80 characters is enough to identify any plausible
# re-typing of a boolean.
EXCHANGE_FLAG_DRIFT_REPR_MAX_CHARS = 80

# ── Cross-shard collateral transfers ──────────────────────────────────────────

# Full API path of the intra-exchange (shard-to-shard) collateral transfer
# endpoint, used by trader.ensure_shard_collateral() to move cash onto whichever
# shard a selected trade's legs actually settle against. The pinned SDK has no
# generated method for this route, so it is reached through
# _http.signed_request_json(), which signs the path VERBATIM — hence the full
# string including the /trade-api/v2 prefix, not a suffix appended to PROD_URL.
# WARNING: the request body's `amount` field is denominated in CENTICENTS
# (1/100 of a cent), the codebase's THIRD money unit after integer cents and
# fixed-point dollar strings. Convert with trader._cents_to_centicents(), never
# by inlining a factor at the call site.
TRANSFER_PATH = "/trade-api/v2/portfolio/intra_exchange_instance_transfer"

# Seconds to keep waiting for accepted transfers to actually SETTLE. The
# transfer endpoint is ASYNCHRONOUS: a 2xx means "accepted", not "the funds have
# landed", so an order submitted immediately after could still be rejected for
# insufficient collateral on its shard. ensure_shard_collateral() therefore
# re-reads the per-shard balance until every under-funded shard is covered, and
# this bounds that wait. 30s is long enough for an in-flight transfer to land,
# short enough that a stuck transfer doesn't hold the run open while its scanned
# prices go stale; on timeout the affected trades are dropped rather than
# submitted against money that may not be there.
TRANSFER_SETTLE_TIMEOUT_SECONDS = 30

# Seconds between per-shard balance re-reads while waiting for transfers to
# settle. Each poll costs one GET /portfolio/balance, so 2s gives ~15 reads
# inside TRANSFER_SETTLE_TIMEOUT_SECONDS — responsive enough to proceed promptly
# once the funds land, infrequent enough not to spend rate-limit tokens on a hot
# loop while money is in flight.
TRANSFER_POLL_INTERVAL_SECONDS = 2

# Maximum seconds a scheduler-spawned bot run may take before being killed.
# Prevents a hung run (e.g. a network stall inside the SDK) from blocking the
# weekly scheduler daemon forever.
SCHEDULER_JOB_TIMEOUT_SECONDS = 3600

# A run that exits EXIT_NO_TRADEABLE_SHARDS (exchange-wide halt) is retried
# after this many seconds, at most SCHEDULER_BLIND_MAX_RETRIES times per
# Monday slot, so a maintenance window overlapping the 09:00 fire no longer
# silently costs the week (TS-01). Hourly x4 covers a four-hour outage while
# keeping the scan close to the intended slot; a longer interval would trade
# on stale morning pricing. Bounded on purpose: a multi-day outage stops
# retrying after the cap and scheduler_state.json records the attempts.
SCHEDULER_BLIND_RETRY_SECONDS = 3600
SCHEDULER_BLIND_MAX_RETRIES   = 4

# ── Process exit-code contract ────────────────────────────────────────────────
# main.py's process exit code is the only signal the scheduler (a separate
# subprocess, per scheduler.run_job) has for what happened in a run beyond a
# generic pass/fail — a prod run skipped for insufficient balance used to exit
# 0 just like a clean run, so the scheduler logged "Job completed successfully."
# and the WARNING explaining why nothing happened was buried in a log the
# scheduler never reads (BS-14). These codes are shared between main.py
# (which returns/exits them) and scheduler.py (which maps them to distinct log
# levels/messages) — they live in config.py so both modules import the same
# values instead of duplicating magic numbers. An unhandled exception in
# main.py is NOT covered here: it still propagates and the interpreter exits
# 1, same as always.
EXIT_OK                       = 0
EXIT_SKIPPED_LOW_BALANCE      = 10
EXIT_TRADES_NEED_ATTENTION    = 20
# NOTHING was scanned this run. Two causes, both reported with this code
# because the scheduler's decision is the same for either: every advertised
# shard reported trading_active=false (an exchange-wide halt or maintenance
# window — observed live 2026-09-03 00:29 PDT), so ingest dropped every market
# (TS-01); or ingest produced zero markets for a reason /exchange/status could
# not name — scanner.fetch_shard_statuses is fail-soft and returns None on ANY
# internal failure, which makes the all-halted test unevaluable exactly when
# something has gone wrong (VI-02). Distinct from EXIT_OK's "scanned
# everything, found no edge": scheduler.run_job maps this to a WARNING, never
# counts the weekly slot as satisfied, and retries it.
EXIT_NO_TRADEABLE_SHARDS      = 30

# ── API pagination ────────────────────────────────────────────────────────────

# Number of items to request per page when paginating market/event endpoints.
# The API rejects limits above 200 with HTTP 400 (observed 2026-07; it used to
# accept 1000), so this must stay <= 200.
MARKET_PAGE_SIZE   = 200

# Number of positions to request per page when paginating the /portfolio/positions endpoint.
POSITION_PAGE_SIZE = 500

# Hard ceiling on pages walked by scanner.py's cursor loops (open events, MVE
# events, positions). The stuck-cursor guard catches a cursor that repeats
# consecutively, but a keyset cycling with period > 1 (A, B, A, B, ...) never
# does; the guard now remembers every cursor it has used, and this cap bounds
# the walk regardless (TS-05). At MARKET_PAGE_SIZE (200) this is 1,000,000
# markets; a 2026-09 prod ingest is ~135k markets (~700 pages). Same
# bound-the-work-then-say-so idiom as ARCHIVE_TAIL_MAX_PAGES and
# EVENT_TITLE_FALLBACK_MAX_LOOKUPS.
SCANNER_MAX_PAGES  = 5000

# Cap on the number of multivariate-events pages the backtester's event-title
# lookup will scan (historical._load_or_build_event_titles). The MVE listing is
# effectively unbounded, so titles not found within this many pages fall back
# to bounded per-ticker /events/{ticker} lookups instead of paging for hours.
MVE_TITLE_LOOKUP_MAX_PAGES = 500

# Number of worker threads used by historical.fetch_all_settled_markets to
# fetch archive day-slices and live settled-day windows in parallel. Currently
# 8. The settled-market history is tens of millions of records at 1000/page
# (the API's hard page-size cap), so a sequential walk takes hours; sharding
# the fetch across workers is what makes it tractable. Note the specific
# figure below is a historical datapoint from a DIFFERENT worker count, not a
# measurement of this constant's current value: at 12 workers, a 2026-07-13
# live run measured ~20 req/s with zero HTTP 429s. Each worker's calls still
# go through api_call_with_retry, so if Kalshi does start throttling, the
# normal exponential backoff applies per worker. Raise cautiously.
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
# fetch (backtester._fetch_candles_parallel). Currently 8. Each fetch is an
# independent read-only GET routed through api_call_with_retry, so a 429
# degrades to that worker's own exponential backoff rather than failing the
# run. Note the specific figure below is a historical datapoint from a
# DIFFERENT worker count, not a measurement of this constant's current value:
# at 12 workers, a 2026-08-03 live run measured ~20 req/s with zero HTTP 429s.
# Fetching sequentially was live-measured the same day at ~4.3 tickers/sec,
# i.e. ~34 hours for a 3-month window, which makes this loop the dominant cost
# of a backtest. Cache files are keyed per ticker, so two workers can never
# target the same path — _save_json_cache writes atomically (tmp+replace), but
# its tmp name is derived from the destination, so a shared cache path would
# still collide; never introduce a fetch whose cache path is shared across
# workers. Each worker keeps fetch_candlesticks' own rate_limit_sleep default
# (0.15s) between its pages.
CANDLESTICK_FETCH_MAX_WORKERS = 8

# Eligible-market count above which backtester._prepare_entries warns the
# operator about the RAM the grouping/pairing step is about to need, and the
# per-record estimate the warning multiplies by.
#
# BS-15 hardened the settled-market FETCH to stream day slices to disk, but the
# phase right after it holds the whole window as one list and builds two group
# maps and two candidate-pair lists over it. This comment is the ONLY place
# (with the matching CLAUDE.md note) that records historical measurements: the
# warning itself prints only the running process's own numbers, so it can never
# quote a figure from some other run. Two runs on a 16 GB host, both over the
# same 1,089,165-record FIVE-DAY window — the cheapest run the tool supports —
# measured this phase from opposite sides, and neither figure explains the
# other (TS-07):
#   * 2026-09-12, cache MISS: fetch_all_settled_markets assembled the list from
#     the streamed day slices, and the recorded peak for that run is 6.05 GiB.
#     That run predates the _log_rss() lines, so its own log carries no RSS
#     line to confirm it — the figure was observed outside the log.
#   * 2026-09-13, cache HIT: fetch_all_settled_markets returned at its
#     use_cache early return ("Loaded 1089165 settled markets from cache") and
#     assembled nothing, yet the run logged "Peak RSS before grouping: 3816
#     MiB" and "Peak RSS after pair extraction: 3977 MiB". Grouping and pair
#     extraction added only 161 MiB to the high-water mark there; the rest was
#     the cache read itself, since historical._load_json_cache does
#     json.loads(path.read_text()) over a 1.43 GB assembled cache file.
# The warning is the operator's budget line on a smaller host; it never caps or
# drops anything.
#
# BACKTEST_RECORD_BYTES_ESTIMATE is the PARSED footprint of one cached market
# record — what the list of dicts itself costs in memory. It is NOT the
# record's size on disk, and it is NOT a peak-RSS predictor: peak additionally
# covers whatever transients are live at the same instant (on the cache-hit
# path, the whole decoded JSON string that json.loads is reading from; during
# pairing, the group maps and the pair lists). Two measurements taken on the
# real 2026-09-07 assembled cache bracket the value, which is why it stays a
# round estimate: decoding 195,038 sampled records under tracemalloc allocates
# ~3.0 KB per record (the sample averages 1,290 JSON bytes/record against the
# file-wide 1,310, so it is representative), while subtracting that 1.43 GB
# decoded string from the 3,816 MiB pre-grouping peak above leaves no more than
# ~2.4 KB per record actually resident. 2,700 sits between the two.
BACKTEST_MARKETS_RAM_WARN      = 500_000
BACKTEST_RECORD_BYTES_ESTIMATE = 2_700

# Outcome-label (subtitle) coverage below which backtester._prepare_entries
# escalates its coverage census from INFO to WARNING (DR-66).
#
# The subtitle is the outcome discriminator in BOTH backtest grouping keys —
# the time-series key scanner.time_series_group_key() builds, and the third
# component of the same-title key (event_title, title, subtitle). A record whose
# subtitle is blank keys by title alone, which is exactly the pre-DR-01
# strike-blind grouping the live scanner was fixed to stop using. So a backtest
# over a cache written before the 2026-08-14 yes_sub_title ingest fix silently
# replays that defect: its potential-pair counts, trade counts, return figure
# and — worst — its empirical k-hat recommendation for the real-money constant
# TIME_SERIES_INTERVAL_PROB_DISCOUNT all describe a strategy the shipped code
# does not implement, and nothing in the log or the dashboard said so. The
# census is the signal; the defect it closes is the silence, not the grouping.
#
# The threshold sits in the middle of an enormous measured gap, so its exact
# value is not load-bearing. Measured 2026-09-17 by streaming the assembled
# caches on disk (this is the ONLY place, with the matching CLAUDE.md note, that
# records these figures — the census itself prints only the running process's
# own numbers, per TS-07):
#   * settled_markets_2026-05-01_... , written 2026-08-03, i.e. PRE-fix:
#     2,344,886 records, 2,301,327 of them blank — 1.86% coverage.
#   * settled_markets_2026-09-07_... and settled_markets_2026-09-14_... , both
#     written POST-fix: 1,089,165 and 3,138,115 records, and ZERO blank
#     subtitles in either — 100.00% coverage.
# Half is therefore ~48 points below the observed healthy floor and ~48 above
# the observed defective value. Below it, most of the corpus groups on the wrong
# key regardless of what the rest of it does.
#
# event_title coverage is censused on the same INFO line but deliberately does
# NOT escalate, even though a blank event_title collapses the same-title key
# toward (title, subtitle) — the TS-11 direction. It is legitimately near zero
# on a HEALTHY cache (0.57% and 2.87% on the two post-fix caches above), because
# the corpus is overwhelmingly MVE combo markets, whose titles the bulk
# get_events listings exclude by API design and whose per-ticker fallback is
# capped at EVENT_TITLE_FALLBACK_MAX_LOOKUPS. Warning on it would fire on every
# run and train the operator to ignore the line.
#
# Advisory only: the census drops, filters and alters nothing.
BACKTEST_OUTCOME_LABEL_WARN_FRACTION = 0.50

# Annualization bases for the dashboard's risk-adjusted return metrics
# (dashboard._sharpe / _sortino). TWO of them exist because the dashboard's
# Benchmark Comparison table puts two series with DIFFERENT periodicities in
# adjacent rows, and they must never share one factor:
#
#   * The strategy equity curve from backtester._build_equity_curve() has one
#     row per CALENDAR day (it opens one day before start_date and runs through
#     today, weekends and holidays included), i.e. ~365 periods per year.
#   * The ^GSPC benchmark series is yfinance's daily close pct_change, which
#     serves TRADING days only, i.e. ~252 periods per year.
#
# Annualizing a calendar-day series at 252 understates its MAGNITUDE by exactly
# sqrt(365/252) = 1.2035 whenever the risk-free hurdle is 0 — the sign is
# unchanged, so a negative Sharpe reads LESS bad at 252, not better-looking in
# any meaningful sense. Verified on a series any reader can re-run, the negation
# of tests/test_dashboard.py's _RETURNS: sharpe@252 = -2.4820064 vs
# sharpe@365 = -2.9870951, ratio 1.2035002. At rf != 0 it is not a constant
# rescale at all, because the per-period hurdle rf/periods_per_year moves too.
#
# dashboard._sharpe/_sortino default to the CALENDAR base: four of their five
# call sites consume _build_equity_curve output, and the single trading-day
# consumer is the external ^GSPC row, which passes TRADING_DAYS_PER_YEAR
# explicitly. Defaulting to the majority case is the same fail-safe-default
# rule scanner.leg_sides() and scanner._shard_index() follow — a future
# in-module caller inherits the correct base rather than the wrong one.
TRADING_DAYS_PER_YEAR: int  = 252
CALENDAR_DAYS_PER_YEAR: int = 365

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
# (settled_markets_<start_date>_<tag>[_nomve].json — the trailing marker is
# INCLUDE_MVE_MARKETS=False's, DR-57), so a cache built under one filter can
# never be served to code expecting another. Because that marker is a bare
# suffix rather than a delimited field, a tag ending in "_nomve" would collide
# with the same tag minus the suffix under the other flag setting; harmless
# while the tag is this single hand-edited constant, worth a delimiter if tags
# ever become caller-supplied.
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

# Absolute ceiling on how many pages the archive TAIL walk (the sequential
# downward walk below created_time == start_date, historical._fetch_archive_tail)
# may request — roughly 2M records of created-time depth below start_date at the
# archive's 1000-record page cap. ARCHIVE_MAX_BARREN_PAGES above is the PRIMARY
# stop rule; this is the backstop, because that rule only bounds depth PAST the
# last productive page: a single long-dated in-window settlement resets the
# barren counter, so without a ceiling the tail can crawl most of created-time
# history one serial request at a time, uncached, on every run. When the cap is
# hit, a WARNING names how many pages were walked and that very-long-lived
# pre-start markets beyond it may be missed — the same bounded-scan idiom as
# EVENT_TITLE_FALLBACK_MAX_LOOKUPS (bound the work, then say loudly what the
# bound cost).
ARCHIVE_TAIL_MAX_PAGES = 2000

# Hard ceiling on RECORDS the archive tail accumulates in memory, independent of
# the page cap above. The tail is the one fetch path with no chunked `emit`
# sink — every other phase streams its records to a day slice and drops them —
# so its whole result is resident at once. ARCHIVE_TAIL_MAX_PAGES alone bounds
# that at 2000 x 1000 x ~BACKTEST_RECORD_BYTES_ESTIMATE, i.e. roughly 5 GB,
# which is the same OOM shape the sharded fetch was rewritten to avoid
# (BS-15). The two caps COMPOSE: whichever binds first stops the walk, so the
# real bound is min(pages x 1000, this) records. 500k at ~2.7 KB each is about
# 1.3 GB — large enough that no realistic window reaches it, small enough that
# a pathological one cannot take the host down. Hitting it logs a WARNING
# naming the count, the same bound-the-work-then-say-so idiom as the page cap
# and EVENT_TITLE_FALLBACK_MAX_LOOKUPS (TS-15).
ARCHIVE_TAIL_MAX_RECORDS = 500_000

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
    days separate the two legs' deadlines: the later leg's YES ask must exceed
    the earlier's by MIN_PRICE_DIFF_SHORT_GAP (15%) when the deadlines are up
    to SHORT_DEADLINE_GAP_DAYS (15 days, inclusive) apart, and by
    MIN_PRICE_DIFF_LONG_GAP (30%) for anything wider. The gap is a distance,
    not a direction — both of the things that measure it,
    scanner.deadline_gap_days() over two close_times and
    scanner.same_event_ladder() over two STATED deadlines (DR-73), are
    order-independent, and the direction (later leg pricier) is enforced by
    the caller's own filter. Downstream of pair formation the gap is read
    through scanner.pair_gap_days(), which returns whichever of the two the
    pair was admitted on. Callers must already have enforced
    gap_days <= MAX_DEADLINE_GAP_DAYS — this helper only selects the tier and
    does not reject over-cap gaps itself.

    Args:
        gap_days (int): Calendar days between the two legs' deadlines —
            their close_times for a cross-event pair, their stated deadlines
            for a same-event ladder. Range: 0..MAX_DEADLINE_GAP_DAYS
            (caller-enforced).

    Returns:
        float: The minimum required YES ask price difference (dollars, 0-1)
            by which the later leg must exceed the earlier one (later by
            close_time, or by stated deadline for a DR-73 ladder).
    """
    if gap_days <= SHORT_DEADLINE_GAP_DAYS:
        return MIN_PRICE_DIFF_SHORT_GAP
    return MIN_PRICE_DIFF_LONG_GAP


def time_series_profit_prob(pA: float, pB: float, k: float | None = None) -> float:
    """
    Return the modelled probability that a time-series pair trade is profitable.

    The trade (YES on the earlier contract at pA, NO on the later at ~1 - pB)
    loses only when the event first happens BETWEEN the two deadlines — earlier
    NO, later YES. The market-implied probability of that in-between scenario
    is the YES-ask gap (pB - pA); the strategy disputes it, believing only
    TIME_SERIES_INTERVAL_PROB_DISCOUNT of that mass. So:

        p = 1 - TIME_SERIES_INTERVAL_PROB_DISCOUNT * max(0, pB - pA)

    The gap is clamped at zero so a pair whose earlier contract is pricier
    (never a candidate, but reachable from reporting code) models as riskless
    rather than as a negative loss probability. This is the single definition
    of the model — strategy._kelly_p, backtester.run_backtest and
    dashboard._kelly_fraction all call it, so the three can never drift.

    The optional k overrides that constant for one call. It exists ONLY for the
    backtester's calibration sweep (backtester.run_backtest_sweep): the live
    sizer (strategy._kelly_p) never passes it, so live sizing always reads the
    config constant. It is resolved at call time rather than bound as a default
    argument, so tests that monkeypatch the constant still take effect.

    Args:
        pA (float): YES ask of the earlier contract, dollars in [0, 1].
        pB (float): YES ask of the later contract, dollars in [0, 1]. Earlier
            and later by close_time, or by STATED deadline for a same-event
            ladder (DR-73).
        k (float | None): Interval-discount override in [0, 1]. None (default)
            reads TIME_SERIES_INTERVAL_PROB_DISCOUNT.

    Returns:
        float: Probability of profit in (0, 1] for a discount in [0, 1].
    """
    discount = TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k
    return 1.0 - discount * max(0.0, pB - pA)


def fee_per_pair_approx(price_a: float, price_b: float) -> float:
    """
    Compute a continuous approximation of the total Kalshi taker fee for one pair trade.

    Used during pair filtering and Kelly sizing (before the exact integer contract
    count is known). The exact fee formula uses ceiling rounding per leg; this
    approximation treats the contract count as a continuous quantity, making it
    suitable for threshold comparisons.

    The formula is: TAKER_FEE_RATE * (price_a*(1-price_a) + price_b*(1-price_b)),
    which sums the quadratic fee contribution of the two legs. It is symmetric
    in its arguments and side-agnostic: pass the per-contract cost of whatever
    side each leg buys — (nA, pB) for a same-title pair, (pA, nB) for a
    time-series pair, i.e. exactly scanner.leg_prices(pair).

    Args:
        price_a (float): Cost in dollars of the side bought on the first leg.
            Range: (0, 1).
        price_b (float): Cost in dollars of the side bought on the second leg.
            Range: (0, 1).

    Returns:
        float: Approximate total taker fee per contract pair (in dollars).
            This is an underestimate relative to the exact ceiling formula —
            i.e. it is OPTIMISTIC (makes pairs look slightly more profitable
            than they actually are at small sizes). That is why it must only
            be used for filtering: final validation always re-checks with
            fee_leg_exact() so an underestimated fee cannot admit a bad trade.
    """
    return TAKER_FEE_RATE * (price_a * (1.0 - price_a) + price_b * (1.0 - price_b))


def fee_leg_exact(n: int, p: float) -> float:
    """
    Compute the exact Kalshi taker fee for one order leg of n contracts at price p.

    Kalshi applies a ceiling rounding per leg: the fee is rounded up to the nearest
    cent. This means small positions are slightly over-charged relative to the
    continuous approximation. Use this function once the final contract count n is
    known (e.g. in strategy.py and backtester.py).

    Side-agnostic, like fee_per_pair_approx(): p is the cost of the side the
    leg actually buys, whichever market and side that is.

    Args:
        n (int): Number of contracts for this leg. Should be >= 1.
        p (float): Price in dollars of the side bought on this leg (the YES ask
            for a YES buy, the NO ask for a NO buy). Range: (0, 1).

    Returns:
        float: Taker fee in dollars, rounded up to the nearest cent.
    """
    # Round to 6 decimals before the ceiling so binary floating-point noise
    # (e.g. 0.07*100*0.25*100 = 175.00000000000003) cannot bump an exact
    # cent amount up an extra cent — Kalshi charges ceil of the TRUE value.
    return math.ceil(round(TAKER_FEE_RATE * n * p * (1.0 - p) * 100, 6)) / 100


def max_affordable_pairs(
    balance_cents: int, price_sum: float, fraction: float | None = None,
) -> int:
    """
    Return the largest whole contract-pair count a fraction of the balance buys.

    The single definition of the budget -> contracts step, called from both ends
    of the sizing pipeline so the two can never drift:

      * scanner.enrich_with_orderbook_prices() passes the default
        BUDGET_FRACTION and the BEST qualifying level's price sum, to bound how
        much order-book depth it averages into the pair's fill price.
      * strategy.compute_trade() passes the capped Kelly fraction and the actual
        prefix-average price sum, to size the trade itself.

    The scanner's call is therefore an UPPER BOUND on the sizer's: its fraction
    is the maximum any Kelly result can be capped to, and its price sum the
    minimum any prefix average can reach (levels are ascending, so every deeper
    prefix costs at least as much per pair). That bound is what lets enrichment
    price a pair at a size the sizer can never exceed.

    fraction is resolved at CALL time rather than bound as a default argument,
    so a test that monkeypatches BUDGET_FRACTION still takes effect — the same
    rule, for the same reason, as time_series_profit_prob's k.

    Args:
        balance_cents (int): Account balance in integer cents. Range: >= 0.
        price_sum (float): Combined per-contract cost of the two legs, in
            dollars. Range: (0, 2); a nonpositive value returns 0 rather than
            raising, since it means the book carried no usable level.
        fraction (float | None): Fraction of the balance to spend. None (the
            default) reads BUDGET_FRACTION. Range: [0, 1].

    Returns:
        int: Floor of (balance_dollars * fraction / price_sum). 0 when the
            budget cannot afford a single contract pair, or when price_sum is
            nonpositive.
    """
    f = BUDGET_FRACTION if fraction is None else fraction
    if price_sum <= 0:
        # A nonpositive sum means no usable level; "affords nothing" is the
        # right answer and keeps every caller free of a ZeroDivisionError guard
        return 0
    # Same expression order as the sizing this replaced, so the float result is
    # identical: dollars first, then the fraction, then the division.
    return int((balance_cents / 100.0) * f / price_sum)


def create_new_output(path: Path) -> tuple[Path, BinaryIO]:
    """
    Exclusively create `path`, suffixing "-1", "-2", … if that exact name exists.

    Every generated output in this project is named from a local timestamp. Two
    writers that render the same timestamp string resolve to one path, and the
    second truncates the first — silently discarding a run's rows on the very
    path that exists to guarantee they are never dropped (TS-18). Exclusive
    creation (O_EXCL, via Path.open("xb")) makes that impossible rather than
    merely improbable: two processes racing for one name cannot both win,
    however fine the timestamp in it.

    Only FileExistsError is retried. Any other OSError — a permission failure, a
    full disk — propagates on the first attempt rather than being retried
    OUTPUT_NAME_MAX_ATTEMPTS times against a cause that will not change.

    Args:
        path (Path): Desired path. Used verbatim when free; otherwise its stem
            gains a "-N" suffix, so "trade_log_….xlsx" becomes
            "trade_log_…-1.xlsx". The suffix is "-", never "_", so it can never
            be misread as another timestamp component.

    Returns:
        tuple[Path, BinaryIO]: The path actually created and its open binary
            handle, positioned at byte 0. The CALLER closes the handle.

    Raises:
        OSError: If creation fails for any reason other than a name collision,
            or if the final uuid-suffixed attempt also fails.
    """
    for attempt in range(OUTPUT_NAME_MAX_ATTEMPTS):
        candidate = (
            path if attempt == 0
            else path.with_name(f"{path.stem}-{attempt}{path.suffix}")
        )
        try:
            return candidate, candidate.open("xb")
        except FileExistsError:
            continue
    # Pathological. One unguarded attempt on a random name, so an operator gets a
    # real OSError rather than a silent loop.
    final = path.with_name(f"{path.stem}-{uuid4().hex[:8]}{path.suffix}")
    return final, final.open("xb")
