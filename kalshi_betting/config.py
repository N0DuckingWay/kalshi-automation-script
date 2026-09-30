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
    scheduler.py, treasury.py, run_lock.py, and main.py — plus two standalone,
    human-run tools kept deliberately outside the pipeline's import graph: the
    verification CLI (see CLAUDE.md's pipeline-isolation rule) and
    defaults_server.py, the local pages that save the live defaults and start
    live trading runs with them.

Notes:
    PROJECT_ROOT is derived from __file__ so the package works correctly on any
    machine regardless of where the repo is cloned.
    The sandbox URL (demo-api.kalshi.co) requires a completely separate account
    registered at demo.kalshi.co — the production API key will return 401 there.
    The saved live defaults (LIVE_DEFAULTS_FILE, live_defaults.json in the repo
    root), which every live run starts from, are read and written here too:
    read_saved_live_defaults and live_defaults read the file, and
    save_live_defaults writes it (on defaults_server's Confirm and save, or
    Confirm and trade). The toggle constants here are the backtest's k and
    caps and the fallback live_settings() returns, never a live run's
    defaults. The runs defaults_server starts keep their files under
    LIVE_RUNS_DIR, which the server reads at call time, so tests redirect it.
"""
import fcntl
import json
import math
import numbers
import os
import pathlib
import stat
from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4
from zoneinfo import ZoneInfo

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

# The per-trade Kelly cap for EVERY pair, as a fraction of the bankroll Kelly
# sizes on (the portfolio value strategy.compute_trade is handed; kelly_budget
# never lets one trade spend more than the cash on hand): a multiple of
# SIZE_CAP_STEP from 5% to 100%, where 1.0 is no cap (f* <= p <= 1).
# backtester binds it by value at import for its eager points, and
# live_settings() reads it (see "Live trading toggles" below: a live run sizes
# at the saved live defaults' size_cap instead). At 1.0, with this file's k and
# SAME_TITLE_SIZE_CAP, one pair still stakes at most 20%: a time-series pair
# under 1 - k (max_kelly_fraction), a same-title pair under SAME_TITLE_SIZE_CAP.
BUDGET_FRACTION               = 1.0

# An EXTRA per-trade cap on SAME-TITLE pairs, on the same grid: a same-title
# pair is capped at min(BUDGET_FRACTION, this), a time-series pair never reads
# it (pair_size_cap); 1.0 adds no cap. Under a BUDGET_FRACTION of 1.0 it bounds
# a same-title pair, whose f* reaches about 0.89 on a wide divergence.
# backtester binds it by value and caps through pair_size_cap, and
# live_settings() reads it; a live run caps at the saved live defaults'
# same_title_size_cap instead.
SAME_TITLE_SIZE_CAP           = 0.20

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

# The BACKTEST's default time-series spread band (floor, ceiling) on pB - pA —
# (0.0, 1.0) is "no band": the floor is the deadline-gap tier alone and there
# is no ceiling; the LIVE band is TIME_SERIES_SPREAD_BAND. Read ONLY by
# time_series_spread_band(). No module outside config, backtester, backtest and
# dashboard may reference a band helper or constant, import one of those three
# modules or call min_price_diff_for_gap (the live path reaches the floor through
# live_time_series_floor) — pinned by tests/test_strategy.py::
# TestTimeSeriesKellyParity::test_ast_live_path_reads_toggles_only_through_live_settings.
BACKTEST_DEFAULT_SPREAD_BAND  = (0.0, 1.0)

# Band grid for the backtest's band x k scenario sweep, crossed with
# INTERVAL_DISCOUNT_SWEEP: 6 floors x 6 ceilings = 36 bands x 13 k = 468
# scenarios over ONE fetch. Read only by backtester._sweep_from_candidates
# (reached through run_backtest_sweep(band_sweep=True)), which runs one
# _find_entry pass per band — _find_entry applies ONE band per call, its
# spread_band resolved through time_series_spread_band — and then simulates
# every band at every k of the same k grid. Every floor sits below every
# ceiling, so all 36 bands are valid; every ceiling sits above both
# deadline-gap tiers, so no grid band empties a tier (see
# time_series_spread_band); and the default band above is a member, so a
# default run adds no 37th (an off-grid primary band does, as its own exact
# member).
# Cost measured 2026-09-23 on the DR-73 calibration corpus (10,733 time-series
# pairs, 10,530 with candles on both legs, start 2020-01-01, ladders on): ~1 s
# for ONE full time-series _find_entry pass (~91-97 us per pair across
# repeated runs, before and after _find_entry gained the band alike, at no
# band and at 0.30-0.60 and 0.40-0.50 — it scales with the window's pair
# count; 330 entries at no band, 183 at 0.30-0.60) and ~2-11 ms per
# simulation over its 330 entries (best of 3; it falls with the trade count,
# from 94 trades at k = 0.40 to none at k = 1.00). A band sweep pays the full
# pass ONCE (the tier-floors-off family below pays its own), at the no-band
# band, and every other band rescans only the pairs that entered there (every
# band's entries are a subset of the no-band band's — see
# backtester._sweep_from_candidates): on that corpus 330 of the 10,733 pairs,
# 0.056-0.073 s per band against 1.02-1.06 s for the full pass.
# The whole band sweep over that corpus (the production _sweep_from_candidates,
# measured 2026-09-23 with the pre-pass and the time-series population in
# place: 468 "all", 468 time-series, 468 ladder and 468 cross-event points,
# each "all" and time-series point's two halves, and an ex-top re-simulation
# on each of the 423 "all" and 423 time-series points with a traded event —
# the other 45 of each entered no trade — 4,590 simulations in all) took 26.4
# and 26.6 s in two runs, 3.2-3.3 s of it the entry passes (before the
# pre-pass and the time-series population, the 2,763-simulation sweep of the
# same corpus took 52.2 s, 37.7 s of it the entry passes), and kept 1,872 equity
# frames of ~138 KB each (2,460 daily rows), ~258 MB in all. A floor at or
# below a pair's tier is inert for it.
# The tier-floors-off family (run_backtest_sweep(tier_off_sweep=True), which
# backtest.py turns on together with the band sweep) enters and simulates
# again, with min_price_diff_for_gap's tier_floors=False, the 18 bands whose
# floor sits below a deadline-gap tier (floors 0, 0.20 and 0.25): a second
# full pass of its own at the no-band band, with the tiers off, plus 17
# rescans on the shipped grid, then 234 more "all" cells (18 bands x 13 k)
# and up to 234 in each of the other three populations (an empty one is
# skipped, as on the tier-on grid), with a split-half check on each "all" and
# "time_series" point and an ex-top re-simulation on each of those that
# traded an event.
# Measured 2026-09-26 on the same corpus (the production
# _sweep_from_candidates, start 2020-01-01, ladders on, zero API calls; all
# 234 cells of every population non-empty there):
# about 16 s more (42.8-44.4 s with it against 26.9-28.2 s without, over
# four runs each), 6.4-6.5 s of it the entry passes against 3.2-3.3 s (55
# passes against 37), 6,882 simulations against 4,590 — +2,292: 936 points,
# 936 halves and 420 ex-top re-simulations — and 2,808 kept equity frames
# against 1,872 (+936), 387.5 MB of frames against 258.3 MB (+129.2 MB of
# retained equity frames). Peak RSS is not quoted: it varied more between
# repeat runs of one setting than between the two settings. Every tier-on
# cell and k point (trade and win counts, final balance, mean return per
# trade, both halves, the ex-top check) and every band's calibration (the
# pooled row's n, realised rate, mean implied spread and k-hat; each
# bucket's label, tier, n and k-hat) came out identical to both the run
# without it and the code before it existed.
SPREAD_BAND_SWEEP_FLOORS      = (0.0, 0.20, 0.25, 0.30, 0.35, 0.40)
SPREAD_BAND_SWEEP_CEILINGS    = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

# Minimum YES ask price difference for same-title pairs. These are markets asking
# the exact same question, so even a small divergence (5%) is anomalous and worth trading.
SAME_TITLE_MIN_PRICE_DIFF     = 0.05

# Prior probability that a same-title pair co-resolves (i.e. both YES or both NO).
# Set at 95% — divergence is an anomaly, so we assume high correlation by default.
# It applies only to a pair that passed BOTH pairing rules: its two event
# tickers belong to two DIFFERENT series (the one-series rule, DR-02/DR-54),
# and its two markets close within SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each
# other (DR-74). Identical wording that fails either rule is two different
# fixtures of one question, for which this prior is simply false.
SAME_TITLE_CO_RESOLVE_PROB    = 0.95

# Largest gap between the two markets' close times that a same-title pair may
# have, in seconds (DR-74). Identical wording on two DIFFERENT series is one
# question only when both markets resolve at the same moment: a men's and a
# women's college basketball game between the same two schools (KXNCAAMBGAME /
# KXNCAAWBGAME, "Western Illinois at Eastern Illinois Winner?") share title,
# subtitle and event title, and so do the Champions League and La Liga
# "Atletico vs Barcelona" matches. On the 365-day backtest (start 2025-09-24,
# the backtest's population: event_title blank on ~95% of eligible records)
# the 1,906 same-title candidates closed either at the identical instant (245
# pairs, 243 settled the same way — including 226/226 commodity and 15/15
# financial; the other 2 are the KXRTBLACKPHONE2/KXBLACKPHONE2 anomaly) or at
# least 1.17 h (4,217 s) apart (1,661 pairs, 60.7% settled the same way).
# Live close_time is the SCHEDULED close — for a game, tip-off plus a fixed
# number of days, which keeps a doubleheader's two games hours apart (checked
# on archived pairs) — and the backtest's is the REALIZED close. Not covered:
# futures decided early, whose live close_time is a family-wide placeholder
# (all KXOSCAR* at 2027-12-31T15:00Z), pass live at a 0 s gap.
SAME_TITLE_MAX_CLOSE_GAP_SECONDS = 60 * 60

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
# and both finders refused every same-event candidate from the first commit
# until DR-73, on the rationale that a shared event ticker means multi-choice
# OPTIONS. That is true of an MVE event's option labels and false of a dated
# ladder, whose two rungs are the time-series premise itself: the earlier
# deadline's event nests inside the later one's. A ladder pair is ordered and
# tiered on the two STATED deadlines (scanner.stated_deadline /
# same_event_ladder), never on close_time, which a settled or single-instant
# event gives every rung alike.
#
# ON BY OPERATOR DECISION OF 2026-09-26, to make same-event ladder runs the
# default, live and backtest (it shipped False with DR-73). What the switch
# buys and what it puts at risk, measured rather than assumed — read this
# before relying on it. To turn it back off, set it False, re-pin
# tests/test_config.py::TestSameEventLadderSwitch to ship it off and reword
# this header, keeping the evidence below:
#
#   Nesting holds for ladders, and does not for the cross-event pairs. Over 284
#   cached day slices (257 archive, 27 live), 1,821 same-event cumulative
#   pairs read to two different stated deadlines and the impossible
#   A=YES/B=NO cell occurs 0
#   times (0 of the 975 within MAX_DEADLINE_GAP_DAYS). The CROSS-EVENT
#   baseline on the same corpus is 2,872 of 22,080 — 13.01%. The premise
#   violations in the archive come from the CROSS-EVENT pairs this finder admits.
#
#   Live funnel (2026-09-22 snapshot, 113,303 markets, on which the finder
#   emits 0 time-series and 0 same-title pairs with the switch off; tier
#   floors on, no band — for the shipped defaults see CLAUDE.md: "The live
#   defaults of 2026-09-27 — decision record"): 3,354
#   same-event candidates -> 2,840 past the identical-wording check (-502)
#   and the cumulative-wording one (-12 snapshot) -> 2,774 reading as two
#   different calendar days (-31 undated, -35 field conflict) -> 350 within
#   the 30-day stated-gap cap -> 89 past the price tier -> 87
#   past the pA + nB < $1 guard -> 24 pairs emitted, in 24 events and 23
#   series, all tradeable.
#
#   Exposure, at k 0.75 and a 20% cap for every pair (tier floors on, no band):
#   strategy.compute_trade + select_portfolio over those 24 pairs at a $10,000
#   balance size 17 trades and SELECT 6, deploying $9,803.66 — 98%
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
#   at that 20% cap and a fifth at f* = 0.17, the five together deploying
#   $9,716.81. The per-trade cap caps each PAIR, not the portfolio, and under
#   the MODEL's own probabilities those five lose together 12.7% of the time
#   IF THE FIVE UNDERLYINGS ARE INDEPENDENT — the figure is the
#   PRODUCT of the five marginal loss probabilities, and nothing here models
#   correlation, which can only raise it (53.5% at market prices, by the same
#   product). Each loses its full stake in that cell. At the account's real
#   balance ($64.44 at the 2026-09-25 prod dry run) the same snapshot selects
#   6 trades deploying $63.84 — 99% of it — at a market-implied EV of -$19.58,
#   leaving the cash under MIN_BALANCE_CENTS. The $50 gate reads the portfolio
#   value (cash plus the open positions' value), so later runs still scan, but
#   none spends more than the cash left until those positions settle.
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
#   not of emitted pairs: on the same snapshot NONE of those 24 emitted pairs
#   has an intervening rung of its own event priced outside [pA, pB], so the
#   guard below would have changed nothing there (at the shipped defaults 1 of
#   the 48 emitted pairs has one: see the decision record). An
#   intervening-rung staleness guard is the natural answer and is deliberately
#   not built here.
#
#   The gate it shipped behind, and its result. DR-73 left the switch off
#   until k-hat was calibrated where the capital goes — the widest
#   (pB - pA > 0.60) band, not pooled, since p and b depend on the spread
#   rather than the gap in days (the earlier 1.114 was measured on snapshot
#   pairs DR-67 refuses and is void) — and was to stay off unless that k-hat
#   was materially below 1: enabling it bets the operator's belief, k, against
#   real scheduled-event information. Entries measured 2026-09-23 with the real
#   _find_entry on the pre-cutoff same-event pairs, banded 2026-09-26 on exact
#   decimal spreads (float band edges, as first recorded, read 0.93 on n=52 for
#   the first run): over the archive extended back to 2021 (start 2020-01-01,
#   299 ladder entries) k-hat is 0.87 above 0.60 (n=71), 0.70 at 0.30-0.60
#   (n=174), 0.56 below 0.30 (n=54), 0.75 pooled; the first, smaller run (start
#   2024-01-01; 178 entries dated 2025-10-27 to 2026-07-20, from 40 events, 0
#   premise violations) read 0.92 above 0.60 (51 entries from 22 events), 0.81
#   pooled — n counts entries, not events. The two runs are not independent:
#   only 22 of the extended archive's 1,448 ladder pairs close before 2025, so
#   both, and the backtest below, largely share one 2025-2026 population (see
#   backtester._stated_deadline_dict for how an archive corpus differs from the
#   live ladder population). Live sizing then assumed k = 0.75, below the k-hat
#   above a 0.60 spread in both runs, so it sized those trades on more edge than
#   was measured; the operator turned the switch on regardless. The live k is
#   0.80, still below both, and the live band's 0.5 ceiling refuses those spreads.
#
#   Backtest. The 365-day window from 2025-09-24 (corpus assembled 2026-09-25
#   11:06 UTC; k 0.75, default band; run 2026-09-26 on main @ ba00633) with
#   ladders on: 61 trades, 45.9% won, +111.8% ($10,000 -> $21,181.24), pooled
#   k-hat 0.889 (over this corpus's 399 entries, all ladders — not the
#   calibration runs above), Sharpe 0.90 (at rf = 0). It rests on one trade: YES on
#   KXFISAEXTEND-26MAY "before Jun 1" at $0.04 and NO on "before Jun 15" at
#   $0.06 (27,347 contracts, $2,916.18, entered 2026-05-04) made +$24,430.82,
#   2.2x the run's whole net gain; that event is 41% of the run's POSITIVE
#   event P&L, and the run re-simulated without its entries returns -92.8%.
#   The median trade lost 100%. Re-simulated alone from $10,000, the entries
#   before 2026-04-20 return +98.9% and those on or after it -15.3%. The max
#   drawdown, -82.2% on 2026-06-12, is read off the cost-basis curve, which
#   carried that winner at its cost until it paid out on 2026-06-15, so it
#   bounds the marked-to-market drawdown in neither direction. Of the 468
#   time-series band x k cells, 20.3% are positive (22.5% of the 423 that
#   traded), and the split-half Spearman of their returns is -0.519. The same
#   window with the switch off: 3 trades, all same-title, +4.8%.
#   Under DR-75 (a pair enters on its first Kelly-passing Monday) the
#   ladders-on run gives 61 trades, 49.2% won, +117.08%, and -90.05% without
#   the top event; its pooled k-hat and the switch-off run are unchanged, and
#   this paragraph's other figures were not re-measured.
#
# BOTH PATHS IMPLEMENT THIS since DR-73c: backtester._extract_pairs forms the
# same pairs from a per-event sub-pass and _find_entry orders and gaps them on
# the same stated deadlines, so a ladder-enabled backtest measures the strategy
# a ladder-enabled live run would trade — with one caveat: they pick the rung
# differently (live: the largest pB - pA per group, before Kelly; backtest:
# the Kelly-passing candidate with the largest entry_monthly_ratio whose
# ladder is free, Monday by Monday). So the two paths can replay
# DIFFERENT rungs of the same ladder (on the 2026-09-22 snapshot, with the tier
# floors on and no band, the live funnel narrows 87 eligible ladder candidates
# to 24 emitted, so that contest decides 63 of them). backtest.py's
# --same-event-ladders / --no-same-event-ladders overrides this constant for
# ONE run — --no-same-event-ladders replays the rule as it stood before the
# switch was turned on; scanner.py binds the constant at import, so that
# override never reaches the live finder. The backtest's HTML dashboard also
# renders the resolved setting, not just kalshi_backtest.log:
# dashboard._run_settings_html prints "same-event ladders: on / off / not
# recorded" in the page header (BacktestSweep.same_event_ladders), alongside
# the primary spread band, so a ladder-enabled run's dashboard is no longer
# indistinguishable from a switch-off one. A recorded on/off is also read
# against this switch: "(same as this checkout's config)", or, when the run
# departs from it, a note that it is not a replay of the live rule. The
# pre-fetch echo flags a departure too ("ladders=off (config: on)" or the
# reverse), from backtest.py's binding, and run_backtest_sweep's ladder line
# names it, from backtester's; the header reads backtester's binding through
# BacktestSweep.config_same_event_ladders, which the sweep records.
#
# scanner.py, backtester.py AND backtest.py each bind this by VALUE at import
# (the SCANNER_MAX_PAGES idiom), so a test or harness flipping it at runtime
# must patch the constant on the MODULE it wants to affect — scanner for the
# live finder, backtester for _extract_pairs/_find_entry, backtest for the CLI
# echo — and never on this module, where a patch is a silent no-op: the harness
# looks as if it set the patched value, but every module still runs, and
# reports, the shipped one (since the flip, an "off" harness that patches config
# still pairs ladders, and its logs say "on"). This is NOT the
# config.time_series_profit_prob(k=None) idiom, which works only because that
# helper lives here and reads THIS module's global; the sentinel arguments named
# same_event_ladders resolve their own module's binding at call time, which is
# what makes a run-level override and a module monkeypatch take effect where a
# def-time default would not. For a backtest the supported lever needs no
# patching at all: run_backtest_sweep(same_event_ladders=...) or
# backtest.py --same-event-ladders / --no-same-event-ladders.
TIME_SERIES_SAME_EVENT_LADDERS = True

# ── Time-series strategy model (2026-09 inversion) ────────────────────────────
#
# A time-series pair buys YES on the EARLIER contract (market_a) and NO
# on the LATER one (market_b) — earlier/later by close_time, or by STATED
# deadline for a same-event ladder (DR-73) — when the later contract's YES ask exceeds the
# earlier's by at least the run's entry floor and by no more than its spread
# band's ceiling (the live toggles below), and when BOTH legs are worded as
# cumulative "by <date>" deadlines (scanner.deadline_phrasing) — only then does
# the earlier deadline's event nest inside the later one's, which is what makes
# the model below meaningful at all. The market-implied probability
# that the event first happens BETWEEN the two deadlines is (pB - pA); that is
# the trade's single loss scenario (earlier NO, later YES). This constant is the
# fraction of that market-implied in-between mass we believe — 0.80 means "the
# market overstates it by a fifth; prices will converge by 20%". It is an
# operator-tunable ESTIMATE, not a measured quantity: at 1.0 (take the market at
# face value) the Kelly fraction is <= 0 for every candidate and the strategy
# never fires; smaller values size more aggressively. Measure it against
# settled history with `backtest.py --interval-discount K` (overrides k for
# that backtest run only; live runs price at the saved live defaults' k, which
# main.py --interval-discount K overrides for one run) and
# read the dashboard's "Interval Discount (k) Calibration" section, or the
# calibration block in kalshi_backtest.log — see CLAUDE.md, "Interval-discount
# calibration (2026-09 follow-up)" for the full mechanism.
#
# k also bounds every live time-series stake below 1 - k on the books
# enrichment keeps (max_kelly_fraction); live_rule_warnings warns when a lower k
# lifts min(per-trade cap, 1 - k) above LIVE_EXPOSURE_WARN_FRACTION.
TIME_SERIES_INTERVAL_PROB_DISCOUNT = 0.80

# Grid of k values backtester.run_backtest_sweep() re-simulates so the dashboard
# can offer a k selector without a re-run and, on a band sweep, as the scenario
# explorer's k axis. Spans "size very aggressively" (0.40)
# through "take the market at face value" (1.00, where Kelly is <= 0 for every
# pair and nothing trades — the boundary is informative, so it stays in). Each
# point costs one extra sizing+selection pass over already-fetched candidates —
# on a band sweep, one per point per band (and again at each band the tier
# floors bind, on a tier-floors-off sweep), plus that band's population,
# split-half and ex-top runs (see SPREAD_BAND_SWEEP_FLOORS for the measured
# total); the market fetch and candlestick fetch happen once regardless.
INTERVAL_DISCOUNT_SWEEP = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

# ── Live trading toggles ──────────────────────────────────────────────────────
#
# Seven live toggles (TIME_SERIES_TIER_FLOORS, TIME_SERIES_SPREAD_BAND,
# TRADE_CATEGORIES and TRADE_TAGS below; k, BUDGET_FRACTION and
# SAME_TITLE_SIZE_CAP above). They are NOT the live defaults: a live run starts
# only from the saved ones (LIVE_DEFAULTS_FILE, below) and never falls back to
# these. They are the backtest's k and caps (backtester and backtest bind them
# by value), and what live_settings() returns: the settings a live entry point
# falls back to when a caller hands it none, which only tests and direct
# library calls do. No live module reads them but through LiveSettings (see
# there). tests/test_config.py::TestShippedLiveToggles pins the values; see
# CLAUDE.md: "The live defaults of 2026-09-27 — decision record".

# Whether the time-series entry rule applies the deadline-gap tier floors
# (MIN_PRICE_DIFF_SHORT_GAP / MIN_PRICE_DIFF_LONG_GAP): True -> pB - pA must
# clear max(tier, band floor); False -> the band floor alone, and pB - pA must
# still be strictly positive. The floor also sets the leg-price-sum ceiling,
# 1 - floor; at 1.0 scanner._levels_with_edge_after_fee is the edge check
# (residual: a book that moves after validate_pair_price can still fill within
# the FoK caps' tick of slippage per leg, and a spec whose edge is thinner than
# that then loses in its win cells too). A live run reads the saved live
# defaults' tier_floors instead, which main.py --tier-floors / --no-tier-floors
# overrides for one run.
TIME_SERIES_TIER_FLOORS = False

# The time-series spread band (floor, ceiling) on pB - pA, validated like
# the backtest's (0 <= floor < ceiling <= 1); (0.0, 1.0) is no band. The floor
# is layered on the tier, or stands alone with the tier floors off. A spread
# above the ceiling is refused at scan time (the two YES asks), in enrichment
# (the top of the refreshed book) and before submission (a fresh book). A live
# run reads the saved live defaults' spread_band instead, and main.py
# --spread-min / --spread-max overrides either bound of it for one run.
TIME_SERIES_SPREAD_BAND = (0.0, 0.5)

# Kalshi categories a pair may trade in ("Economics", ...), or None for
# any: a non-empty tuple of names, matched case-insensitively against market A's
# series category as the dashboard files it (historical.series_labels). Applied
# by main._filter_by_category, which fails CLOSED (no listing to file by: no
# trades). A live run reads the saved live defaults' categories instead, which
# main.py --category NAME (repeatable) / --any-category overrides for one run.
TRADE_CATEGORIES: tuple[str, ...] | None = None

# Kalshi tags a pair may trade in, or None for any: matched like
# TRADE_CATEGORIES against the series' FIRST tag under every category, and ANDed
# with it (so the dashboard's category-scoped Tag option "Sports · Basketball"
# is both filters set). A live run reads the saved live defaults' tags instead,
# which main.py --tag NAME (repeatable) / --any-tag overrides for one run.
TRADE_TAGS: tuple[str, ...] | None = None

# The size caps' grid: BUDGET_FRACTION, SAME_TITLE_SIZE_CAP, the saved live
# defaults' two caps and main.py's --size-cap / --same-title-size-cap (in
# percent) must each be a multiple of it
# from 5% to 100%; LiveSettings normalises each onto a backtester.SIZE_CAP_SWEEP
# cell (float-equal). Same value as SAME_TITLE_MIN_PRICE_DIFF, not the same
# constant.
SIZE_CAP_STEP = 0.05

# The largest per-pair stake (max_kelly_fraction, a fraction of the portfolio
# value) live_rule_warnings accepts without a WARNING; this file's values and
# LIVE_DEFAULTS_SEED stay within it. It bounds no trade and is not the per-trade
# cap: it only decides when a live run is warned.
LIVE_EXPOSURE_WARN_FRACTION = 0.20

# ── Saved live defaults ───────────────────────────────────────────────────────

# The live defaults every live run starts from: one JSON record of the seven
# live toggles, saved through defaults_server's confirmation page
# (python3 -m kalshi_betting.defaults_server; its --seed proposes
# LIVE_DEFAULTS_SEED for a first save). save_live_defaults writes it, on that
# page's Confirm and save (or Confirm and trade, which saves before it runs
# the bot), and read_saved_live_defaults / live_defaults read it. A live
# run refuses to start without it; it never falls back to the toggle constants
# above. main.py's flags still override it for one run. Read at call time
# through this module's global, so tests can point it elsewhere
# (tests/conftest.py). Operator state, like scheduler_state.json: gitignored,
# in the checkout the scheduler runs from.
LIVE_DEFAULTS_FILE = PROJECT_ROOT / "live_defaults.json"

# The format tag the saved file carries; a file with any other is refused.
LIVE_DEFAULTS_FORMAT = "live-defaults-v1"

# The largest saved file read; its record is well under 1 KB.
LIVE_DEFAULTS_MAX_BYTES = 65_536

# LiveSettings.origin for toggles built from this module's constants.
LIVE_DEFAULTS_FROM_CONFIG = "config.py"

# The longest note the saved file may keep about what it was saved from.
LIVE_DEFAULTS_SOURCE_MAX_CHARS = 300

# The source note written when the defaults are saved from LIVE_DEFAULTS_SEED.
LIVE_DEFAULTS_SEED_SOURCE = "seed values (config.LIVE_DEFAULTS_SEED)"

# The only two note shapes defaults_server's confirmation page accepts, besides
# no note at all: the backtest dashboard's own wording (with an optional note
# when that run's same-event ladder switch differed from config.py's), or the
# seed's (which the page accepts only on the seed values themselves). So a
# crafted link cannot choose the note's words, which the confirmation page
# shows and every later live run logs in its "Live defaults:" line. This
# restricts the note alone: the page checks a category or tag name for form
# only (one printable name), so a link can still put words of its own there,
# shown on the page as a highlighted change and logged by later runs once
# saved; a name no Kalshi series is filed under matches no pair. It is meant
# for re.fullmatch with re.ASCII, as defaults_server applies it: re.match or
# re.search would also accept a valid note followed by any other words, and
# without re.ASCII its \d would also read the digits of other scripts.
# Nothing in this module checks a note against it:
# save_live_defaults and read_saved_live_defaults accept any note
# live_defaults_source allows, an empty one included.
LIVE_DEFAULTS_SOURCE_PATTERN = (
    r"backtest dashboard for \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}"
    r"( \(same-event ladders (on|off), config\.py (on|off): its pairs are not "
    r"the live bot's\))?"
    r"|seed values \(config\.LIVE_DEFAULTS_SEED\)")

# ── Defaults server ───────────────────────────────────────────────────────────

# Where defaults_server.py listens: the pages that save the live defaults and
# start live runs with them. Loopback only. Changing the port needs a new
# dashboard, since the page's save and trade addresses are written into it when
# it is built.
DEFAULTS_SERVER_HOST = "127.0.0.1"
DEFAULTS_SERVER_PORT = 8765

# The longest request (path plus body) the server reads; its form is under 2 KB.
DEFAULTS_SERVER_MAX_REQUEST_BYTES = 16_384

# Seconds the server waits on a silent connection before dropping it: it
# answers one request at a time, and a browser can hold a connection open
# without sending anything.
DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS = 5

# How long, in milliseconds, a page of the server must be visible before a
# mouse move or key press enables its buttons: a click aimed at another page
# cannot land on one (the DoubleClickjacking defence).
DEFAULTS_SERVER_CONFIRM_ARM_MS = 1000

# The one dashboard file every backtest run writes (and overwrites) in
# PROJECT_ROOT; defaults_server opens it when it starts.
DASHBOARD_FILENAME = "backtest_dashboard.html"

# How much of the dashboard file, from its start, defaults_server reads to tell
# whether the page has the filter bar's Save and Trade buttons (it looks for
# the Trade link's id). The bar sits in the page's opening part, before every
# section and data block, so the first MiB holds it on any page, however large.
DASHBOARD_MARKER_SCAN_BYTES = 1_048_576

# When defaults_server finds its port taken, it asks the server already there
# which checkout it serves (GET /checkout): how long it waits, in seconds, for
# the connection and for each read of the answer, and the most of the answer it
# reads (the real one is a short JSON object). The running server answers one
# connection at a time, and a browser can leave connections to it open that
# send nothing, each held for DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS before it
# is dropped, so the wait leaves room for two of them ahead of the question;
# a shorter wait would take this checkout's own busy server for a stranger.
DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS = 2 * DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS + 2
DEFAULTS_SERVER_CHECKOUT_MAX_BYTES = 65_536

# Where defaults_server keeps each live trading run it starts: one folder per
# run, named by its UTC start time and its id, holding run.json (what the run
# is, written before it starts), output.log (everything it prints) and
# result.json (what main.py --result-file writes when it ends). Operator state
# like scheduler_state.json, so gitignored. Read at call time, so tests point
# it elsewhere (tests/conftest.py).
LIVE_RUNS_DIR = PROJECT_ROOT / "live_runs"

# How often, in seconds, the page of a run that is still going reloads itself.
DEFAULTS_SERVER_RUN_REFRESH_SECONDS = 2

# How much of a run's output.log its page shows: the last this-many lines,
# read from at most its last this-many bytes, so a long log is never read whole.
DEFAULTS_SERVER_RUN_LOG_TAIL_LINES = 25
DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES = 65_536

# How many of the newest run folders the server's index page lists.
DEFAULTS_SERVER_INDEX_RUNS = 10

# Which side each leg of a pair buys, as (side bought on market_a, side bought
# on market_b). scanner.leg_sides() is the ONLY reader — never hardcode a side
# elsewhere. Same-title: NO on the pricier contract (market_a), YES on the
# cheaper (market_b). Time-series: YES on the earlier contract (market_a), NO on
# the later (market_b). The trader always SUBMITS the NO leg first, whichever
# market it sits on (see trader._ordered_legs).
SAME_TITLE_LEG_SIDES  = ("no", "yes")
TIME_SERIES_LEG_SIDES = ("yes", "no")

# Minimum portfolio value in cents a production run needs before it scans:
# the cash on every shard plus Kalshi's value of the open positions
# (main._bankroll_cents, the value every Kelly fraction is taken of). Below $50
# the run stops before any scan (EXIT_SKIPPED_LOW_BALANCE), so it spends no
# API calls when there is too little capital to trade. Cash alone below it
# only draws a WARNING: the run goes on, and no trade spends more than the
# cash left.
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
# prefixes), measured 2026-09-16. That file was mostly "" entries for combo
# tickers nobody looked up; DR-51 migrates its titled entries into
# event_titles_v2.json and deletes it, so the command below reproduces the
# census only on a pre-DR-51 copy. Reproduce with:
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

# Largest loss per contract, in cents below the NO leg's entry price, that
# the unwind of a filled NO leg may take when the YES leg did not fill. The
# unwind buys the YES side back (holding NO is the same as being short YES)
# with a reduce-only immediate-or-cancel bid (it fills what it can right away
# and cancels the rest) capped at 1 - floor/100 dollars, where floor is the
# entry price in cents less this, kept within 1..99 cents; see
# trader._rollback_floor_cents and trader._v2_rollback_price. If the book has
# moved further than that, what is left stays open and the pair is reported
# as "rollback_failed" for a person to handle.
#
# The allowance must cover the whole bid-ask spread, because the NO leg was
# bought at the ask and is closed at the bid: the spread alone is lost even
# if prices do not move. 12 cents lets a normal book close the position while
# a collapsed book still stops the unwind.
ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT = 12

# How far above the scanned price, in ticks of the market's own price grid,
# each buy leg may fill. The V2 order's limit price is its price protection:
# scanned price rounded up onto the grid plus this many ticks (see
# scanner.v2_limit_price, which gets the tick size from
# scanner.tick_size_for_price). One tick lets a book that moved up by one
# tick since the pre-execution check still fill.
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

# The order path the bot sends orders through. "v2" is the only allowed value:
# POST V2_ORDER_PATH below, the only endpoint Kalshi accepts orders on. Its
# orders carry dollar-string limit prices, fixed-point counts, a bid/ask side
# on the YES book and each market's own exchange_index.
#
# main.py and the human-run order-path probe check it at startup, before
# logging is configured or any request is made, and exit 2 on any other
# value (order_api_version_error). If the V2 path misbehaves, stop trading
# and flatten positions by hand in the Kalshi UI; there is no other path.
ORDER_API_VERSION             = "v2"

# Full API path of the V2 create-order endpoint, including the /trade-api/v2
# prefix. A constant (not an inline literal) because the path is signed as part
# of every request — _http.signed_request_json signs timestamp + method + path,
# so the string used to build the URL and the string that is signed must be one
# and the same value.
V2_ORDER_PATH                 = "/trade-api/v2/portfolio/events/orders"

# Self-trade prevention for every V2 order. The V2 create-order endpoint
# REQUIRES this field ("taker_at_cross" | "maker") and rejects a body without
# it. "taker_at_cross" cancels OUR incoming order if it would trade against
# another order on this account; "maker" would cancel the account's resting
# order instead. The bot never leaves an order resting, so the only order it
# could meet is one placed outside the bot (by hand, or by another client on
# this account), and taker_at_cross leaves that order alone. What the endpoint
# reports for the bot's cancelled order has not been observed; the trader
# handles each shape through its existing paths. Nothing filled is an ordinary
# non-fill. On a buy leg, part filled or an error response goes to the
# position-delta check (a part fill the account shows ends as manual_review).
# On the unwind, either one is rollback_failed.
V2_SELF_TRADE_PREVENTION_TYPE = "taker_at_cross"

# What the V2 create-order endpoint sends when a fill_or_kill order cannot fill
# in full: an HTTP 409 error whose JSON body reads
# {"error": {"code": V2_FOK_KILL_ERROR_CODE, "message": ...}}. The exchange
# rejects such an order before it matches, so nothing filled and nothing rests
# — it is the endpoint's kill. trader._submit_order_v2 reads exactly this
# status AND this code, on a fill_or_kill body only, as a kill ("canceled").
# Every other error response still raises — on a buy leg into the caller's
# position check, on the unwind into rollback_failed — because an error the
# bot cannot name is no proof that nothing filled.
V2_FOK_KILL_HTTP_STATUS       = 409
V2_FOK_KILL_ERROR_CODE        = "fill_or_kill_insufficient_resting_volume"

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

# Pauses before each re-read, in seconds, when the V2 NO-leg mapping check
# (trader._confirm_v2_no_mapping) finds the account position UNMOVED right
# after a filled NO leg. The positions ledger lags a fill (a lag of about 1 s
# was seen live on 2026-09-28), so an unmoved reading is most often lag. The
# check re-reads after each pause in turn (up to 7 s in all) and judges the
# mapping disproven only if every re-read is still unmoved, which stops the
# rest of the run. A wrong side mapping moves the position the WRONG way
# rather than not at all, and while pairs run one at a time (see
# V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS below) no other pair sends anything
# during the wait, so waiting longer on a zero risks no extra wrong-side
# position then. The cost is that a filled NO leg checked while the mapping
# is still unverified can wait up to 7 s unhedged — usually only the first of
# a process, and only when the ledger lags.
V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS = (1.0, 2.0, 4.0)

# While the V2 NO-leg mapping is neither confirmed nor disproven in a process,
# trader.execute_trades runs pairs one at a time, so a wrong mapping costs one
# wrong-side position rather than one per pair already in flight. A pair can
# finish without a verdict (its NO leg killed or ambiguous, or the check
# unable to read the account), and when position reads keep failing every
# pair would do so behind about two minutes of retried reads, so the phase is
# bounded in TIME: once it has lasted this many seconds, the pair still
# running is no longer waited for and the rest start together, each still
# checking its own fill (operator decision, 2026-09-28). A time bound rather
# than a count of pairs without a verdict, because killed NO legs cost one
# round trip each and should not use up the protection: three kills in a row
# under a 3-pair count let a wrong mapping open a wrong-side position on
# every pair started together after them. The clock starts with the run, so
# slow position reads spend it before any NO leg is checked: at 60 s, reads
# that each succeed only after ~35 s (a 429 storm that clears) used the whole
# budget during the first pair's two baseline reads, and a wrong mapping then
# opened 7 wrong-side positions instead of 1 (review measurement,
# 2026-09-28). Three hundred seconds covers a first pair behind such reads
# (~105 s to a verdict) plus two or three killed legs at that speed, and
# dozens of killed legs at normal speed (operator decision, 2026-09-28). When
# every position read fails, pairs take ~2 min each, so about 3 run alone
# before the rest start: 7 pairs take ~10-13 min instead of ~2, and 30
# pairs ~15 min, inside the scheduler's one-hour
# SCHEDULER_JOB_TIMEOUT_SECONDS. A stuck order POST holds the other pairs
# back for at most this long; execute_trades itself still returns only when
# that POST does, since order POSTs carry no request timeout.
V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS = 300.0

# How often, in seconds, trader.execute_trades looks at whether the pair
# running alone has settled the V2 NO-leg mapping. The next pair starts as
# soon as it has, without waiting for the rest of that pair (its YES leg, or
# an unwind with its position reads). Short enough to add no noticeable delay,
# long enough that the waiting thread costs nothing.
V2_MAPPING_VERDICT_POLL_SECONDS = 0.05

# The DEFAULT exchange shard. Kalshi splits the exchange into parallel shards,
# numbered by `exchange_index` on markets and in the balance breakdown.
# Orders do not use this constant: every V2 order carries its own market's
# exchange_index. It is:
#   1. the shard assumed when a market payload omits `exchange_index`, so a
#      missing field never drops a market;
#   2. the shard a balance reply with a single total (the sandbox shape) is
#      credited to (auth.py);
#   3. the shard the human-run order-path probe's one-cent collateral-transfer
#      check moves money out of and back into.
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

# ── Weekly scheduler ──────────────────────────────────────────────────────────

# Weekday names, index 0 = Monday, spelled as the `schedule` library's methods.
_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class ScheduledRun:
    """
    When the weekly live run happens: a weekday, a wall-clock time and a time zone.

    Wall-clock means what a clock in that zone reads, so the UTC moment moves
    with daylight-saving time. scheduler.py fires the live run at
    config.SCHEDULED_RUN's weekday and time on the host's clock, and the
    backtest opens simulated trades only at its UTC moments (its ENTRY
    CHECKPOINTS). A spring clock change SKIPS the wall times in its jump and
    an autumn one REPEATS them (clock_change()). The zone is looked up only
    in zone(), so importing config needs no tz database. Frozen, so it can
    key backtester._checkpoint_floor's cache.

    Attributes:
        weekday (int): datetime.weekday() of the run, 0 = Monday.
        hour (int): Wall-clock hour, 0-23.
        minute (int): Wall-clock minute, 0-59.
        timezone (str): IANA zone name, e.g. "America/Los_Angeles".
    """
    weekday: int
    hour: int
    minute: int
    timezone: str

    def __post_init__(self) -> None:
        """
        Check each field's type and range, so a bad schedule fails where it is built.

        Raises:
            ValueError: weekday, hour or minute is not an int in range (a bool
                is refused), or timezone is not a non-empty str.
        """
        for name, value, top in (("weekday", self.weekday, 6), ("hour", self.hour, 23),
                                 ("minute", self.minute, 59)):
            if type(value) is not int or not 0 <= value <= top:
                raise ValueError(f"ScheduledRun.{name} must be an int in [0, {top}], got {value!r}")
        if type(self.timezone) is not str or not self.timezone:
            raise ValueError(f"ScheduledRun.timezone must be an IANA zone name, got {self.timezone!r}")

    def zone(self) -> ZoneInfo:
        """
        Look up the IANA time zone named by `timezone`.

        Returns:
            ZoneInfo: The zone.

        Raises:
            zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: The name is
                unknown, malformed (a path) or not a zone file ("America").
        """
        return ZoneInfo(self.timezone)

    def weekday_name(self) -> str:
        """
        Name the run weekday as the `schedule` library spells it (schedule.every().monday).

        Returns:
            str: Lower-case weekday, e.g. "monday".
        """
        return _WEEKDAY_NAMES[self.weekday]

    def at_time(self) -> str:
        """
        Format the run time as "HH:MM", the form `schedule`'s Job.at() takes.

        Returns:
            str: e.g. "09:00".
        """
        return f"{self.hour:02d}:{self.minute:02d}"

    def label(self) -> str:
        """
        Describe the schedule for log lines and the backtest dashboard's header.

        Returns:
            str: e.g. "Monday 09:00 America/Los_Angeles".
        """
        return f"{self.weekday_name().capitalize()} {self.at_time()} {self.timezone}"

    def cache_slug(self) -> str:
        """
        Name the schedule in a file-name-safe form, for the backtest's cache tag.

        Returns:
            str: e.g. "mon0900-America-Los_Angeles".
        """
        return (f"{self.weekday_name()[:3]}{self.hour:02d}{self.minute:02d}"
                f"-{self.timezone.replace('/', '-')}")

    def wall_time(self, d: date) -> datetime:
        """
        Build the run's wall-clock time on date `d` as a datetime carrying the zone.

        When a clock change skips or repeats that time, Python's default applies
        (fold=0: the UTC offset before the change).

        Args:
            d (date): A date in the zone; any weekday.

        Returns:
            datetime: tz-aware, tzinfo = zone().

        Raises:
            zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: As zone().
        """
        return datetime(d.year, d.month, d.day, self.hour, self.minute, tzinfo=self.zone())

    def instant(self, d: date) -> datetime:
        """
        Convert the run's wall-clock time on date `d` to the UTC moment it happens.

        The one definition of when the run happens (date_problems() repeats it inline).

        Args:
            d (date): A date in the zone; any weekday.

        Returns:
            datetime: tz-aware, in UTC.

        Raises:
            zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: As zone().
            OverflowError: The UTC moment is outside datetime's range.
        """
        return self.wall_time(d).astimezone(UTC)

    def clock_change(self, d: date) -> str | None:
        """
        Say whether a clock change skips or repeats the run's wall time on date `d`.

        It compares the UTC offsets before (fold=0) and after (fold=1) a
        change: larger after means skipped, smaller means repeated.

        Args:
            d (date): A date in the zone; any weekday.

        Returns:
            str | None: "skipped", "repeated", or None when it occurs once.

        Raises:
            zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: As zone().
        """
        wall = self.wall_time(d)
        before, after = wall.utcoffset(), wall.replace(fold=1).utcoffset()
        if before == after:
            return None
        return "skipped" if before < after else "repeated"

    def date_problems(self, first: date, last: date) -> list[str]:
        """
        List the run dates in [first, last] whose run is not one UTC moment on that same date.

        The backtest works in UTC dates, so backtester._prepare_candidates()
        refuses the schedule, before fetching, if this list is not empty. Only
        run weekdays are checked; a bad one is listed once, with its first
        problem: out of datetime's range, skipped or repeated by a clock
        change, or on another UTC date. Never raises OverflowError: an
        out-of-range date is listed, and the walk never steps past date.max.

        Args:
            first (date): First date, inclusive.
            last (date): Last date, inclusive.

        Returns:
            list[str]: "YYYY-MM-DD: <problem>" entries in date order.

        Raises:
            zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: As zone().
        """
        zone = self.zone()
        problems: list[str] = []
        ahead = (self.weekday - first.weekday()) % 7
        if (date.max - first).days < ahead:
            return problems
        d = first + timedelta(days=ahead)
        where = f"{self.at_time()} {self.timezone}"
        while d <= last:
            wall = datetime(d.year, d.month, d.day, self.hour, self.minute, tzinfo=zone)
            try:
                utc = wall.astimezone(UTC)
            except OverflowError:
                problems.append(f"{d.isoformat()}: {where} falls outside datetime's range in UTC")
            else:
                change = self.clock_change(d)
                if change is not None:
                    problems.append(f"{d.isoformat()}: {where} is {change} by a clock change")
                elif utc.date() != d:
                    problems.append(
                        f"{d.isoformat()}: {where} is {utc:%Y-%m-%d %H:%M} UTC, "
                        f"on another date in UTC"
                    )
            if (date.max - d).days < 7:
                break
            d += timedelta(days=7)
        return problems


# The weekly live run: Monday 09:00 Los Angeles time (16:00 UTC under daylight
# time, 17:00 under standard time). scheduler.py fires it on the host's clock;
# the backtest opens simulated trades only at these moments, so a change here
# moves every backtest entry and renames its cached market list. scheduler.py
# and backtester.py bind this at import: tests patch scheduler.SCHEDULED_RUN /
# backtester.SCHEDULED_RUN, never config.*.
SCHEDULED_RUN = ScheduledRun(weekday=0, hour=9, minute=0, timezone="America/Los_Angeles")

# Maximum seconds a scheduler-spawned bot run may take before being killed.
# Prevents a hung run (e.g. a network stall inside the SDK) from blocking the
# weekly scheduler daemon forever.
SCHEDULER_JOB_TIMEOUT_SECONDS = 3600

# The scheduler's record of its claimed weekly slot (scheduler.py), under
# PROJECT_ROOT: the slot, when its run started and finished, its exit code and
# the blind-run retries spent on it.
SCHEDULER_STATE_FILENAME = "scheduler_state.json"

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
# The portfolio value (cash on every shard plus Kalshi's value of the open
# positions) was below MIN_BALANCE_CENTS, so the run stopped before any scan.
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
# No time-series trade: a held market's ladder could not be identified.
# Same-title still ran, so the scheduler logs an ERROR but counts the slot as
# done (a retry would most likely fail the same lookup).
EXIT_TIME_SERIES_SKIPPED      = 40
# Another live trading run on this machine held the run lock (run_lock.py), so
# this one stopped before building a client or making any request. Only a
# production run that sends orders takes the lock, so the scheduler counts its
# slot as done (logging an ERROR instead of a WARNING when the holder has run
# for over SCHEDULER_JOB_TIMEOUT_SECONDS, or its start is not recorded, and may
# be hung).
EXIT_RUN_IN_PROGRESS          = 50

# ── Live run lock ─────────────────────────────────────────────────────────────

# The file a production run that sends orders keeps locked while it runs
# (run_lock.py). It lives in the user's home, not the checkout, so every
# checkout and worktree trading this account shares it. Read at call time, so
# tests point it elsewhere (tests/conftest.py).
LIVE_RUN_LOCK_FILE = pathlib.Path.home() / ".kalshi_betting" / "live_run.lock"

# How long a run keeps trying a taken lock before it stops with
# EXIT_RUN_IN_PROGRESS, and how often it tries. This is long enough to ride out
# a momentary check of the lock (run_lock.held — the defaults server's check
# before it offers a real-money run), and far shorter than any run.
LIVE_RUN_LOCK_WAIT_SECONDS = 2.0
LIVE_RUN_LOCK_POLL_SECONDS = 0.1

# ── Live run result ───────────────────────────────────────────────────────────

# The "format" key of the JSON record `main.py --result-file` writes when a
# production run ends (reporter.write_run_report), so a reader can tell this
# layout from any later one.
LIVE_RUN_RESULT_FORMAT = "live-run-result-v1"

# The most WARNING lines a run result keeps (the rest are only counted), and
# the most characters it keeps of each one's first line, the last of them "…"
# when the line is cut (reporter.RunReportHandler). They keep the record small
# enough to show on one page when a run logs a burst of retry warnings. They
# never apply to an ERROR or CRITICAL line: those are few (at most a handful
# per pair) and are the lines a person has to act on — a failed rollback, a
# V2 mapping disproof naming every position to check — so each is kept whole.
RUN_REPORT_MAX_WARNINGS = 50
RUN_REPORT_LINE_MAX_CHARS = 500

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

# GROUPABLE-market count above which backtester._prepare_candidates (the first
# half of _prepare_entries) warns the operator about the RAM the
# grouping/pairing step holds, and the per-record estimate the warning
# multiplies by. Groupable = the eligible markets (those passing
# _can_ever_enter) whose time-series or same-title grouping key is shared with
# another eligible market: since SS-1 they are the only records
# _prepare_candidates builds a list of for grouping, because every other
# eligible record would form a single-member group, which both groupings drop.
# The warning used to count every ELIGIBLE record, which matched what was then
# held for grouping; since SS-1 the subset is what stays resident from the
# warning onward, so the eligible count would over-state it. A 7-day window
# (--start-date 2026-09-17) measured 7,260,952 eligible records of which
# 184,255 share a key — the gap between those two numbers is what counting
# eligible records over-stated on that run. Both were measured under the pre-P5
# prefilter (SETTLED_PREFILTER_CACHE_TAG "monday-eligibility-v1"), which also
# admitted markets opened after that window's only checkpoint; the current
# prefilter admits a subset, so both numbers and the gap between them are
# smaller now (not re-measured).
#
# Known residual, narrowed by SS-1's Commit C: fetch_all_settled_markets now
# returns a historical.SettledCorpus that streams the assembled
# settled_markets_*.jsonl.gz cache on every walk, so a fetched or
# streamed-cache corpus is never resident (during its assembly only the
# archive tail, capped by ARCHIVE_TAIL_MAX_RECORDS, and a sequential
# fallback's result are lists; day slices and the live frontier stream off
# disk, the frontier through an anonymous spool file). Only a hit on a LEGACY
# settled_markets_*.json cache still hands over ONE list — read whole, and
# since the prefilter was applied during its assembly it IS the eligible set —
# resident, and counted by the "Peak RSS before grouping" line, until
# _prepare_candidates releases it right after its second walk, just before
# this warning. The warning does not count those records, so such a run whose
# eligible count is far above this threshold but whose groupable count is not
# gets no warning for the list that set its peak (its eligible count is still
# on the "Markets to analyze" and "Groupable subset" lines).
#
# BS-15 hardened the settled-market FETCH to stream day slices to disk, but the
# phase right after it held the whole window as one list and built two group
# maps and two candidate-pair lists over it (the two TS-07 measurements below
# predate SS-1 and describe that shape). This comment is the ONLY place
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
#     (Since SS-1's Commit C that whole-file read is the LEGACY .json path
#     only; assembled caches are now written as settled_markets_*.jsonl.gz
#     and streamed record by record, so this figure describes a legacy hit.)
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
# ~2.4 KB per record actually resident. 2,700 sits between the two. A third
# measurement, on the eligible records of the 2026-09-17 7-day window (99.75%
# MVE combos), found 3,926 B/record under tracemalloc over 60,000 samples, so
# on a combo-heavy corpus this estimate UNDERSTATES the footprint. It is left
# at 2,700 deliberately: it only scales an advisory warning, and moving it
# would change what that line reports on every other corpus.
BACKTEST_MARKETS_RAM_WARN      = 500_000
BACKTEST_RECORD_BYTES_ESTIMATE = 2_700

# Outcome-label (subtitle) coverage below which backtester._prepare_candidates
# (the first half of _prepare_entries) escalates its coverage census from INFO
# to WARNING (DR-66).
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
# get_events listings exclude by API design and whose per-ticker lookups are
# deferred whenever a run's unresolved set exceeds
# EVENT_TITLE_FALLBACK_MAX_LOOKUPS (DR-51 — every bulk window at 2026-09
# volume, so expect it lower still after that change). Warning on it would
# fire on every run and train the operator to ignore the line.
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
# rescale at all, because the per-period hurdle rf/periods_per_year moves too —
# and a page built with rates (the risk-free block below) has rf != 0, so the
# identity holds only on a page built without them.
#
# dashboard._sharpe/_sortino default to the CALENDAR base: every call site but
# one consumes _build_equity_curve output, and the single trading-day
# consumer is the external ^GSPC row, which passes TRADING_DAYS_PER_YEAR
# explicitly. Defaulting to the majority case is the same fail-safe-default
# rule scanner.leg_sides() and scanner._shard_index() follow — a future
# in-module caller inherits the correct base rather than the wrong one.
TRADING_DAYS_PER_YEAR: int  = 252
CALENDAR_DAYS_PER_YEAR: int = 365

# ─── Risk-free rate (backtest dashboard only) ─────────────────────────────────
# The dashboard's Sharpe and Sortino subtract, on each day, the yield of this
# Treasury bill's latest auction on or before it (treasury.py) — per day, since
# a multi-year window spans very different rates. A strategy curve is charged
# it only on its capital in open trades (dashboard._rf_hurdle); the ^GSPC row in
# full. REPORTING ONLY: the live bot never imports it.
TREASURY_AUCTIONS_URL: str = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    "/v1/accounting/od/auctions_query"
)
# auctions_query's security_term for the bill (first auctioned 2018-10-16).
RISK_FREE_BILL_TERM: str = "8-Week"
# The auction's stop-out yield on the investment-rate (bond-equivalent) basis, in
# percent: the yield a winning bidder earns. The discount rate (high_discnt_rate)
# is a bank-discount quote that understates it.
RISK_FREE_RATE_FIELD: str = "high_investment_rate"
# Per socket operation, per resolved address — not per request. With
# api_call_with_retry's 6 attempts and 62 s of backoff, an unresponsive host
# with one address costs about 4 minutes before the fallback to the saved copy
TREASURY_API_TIMEOUT_SECONDS: int = 30
# Records per page, and the most pages one download reads (a bound: the whole
# history fits one page)
TREASURY_API_PAGE_SIZE: int = 1000
TREASURY_API_MAX_PAGES: int = 20
# How far apart a curve's largest and smallest daily return must be for the
# dashboard to compute a Sharpe or Sortino at all (dashboard._varies): a curve
# whose returns differ by less than this never really moved, and its ratios
# read 0.0. A tolerance rather than rounding, since rounding can split two
# nearly equal returns across a rounding boundary. 1e-12 of the balance is
# $0.00000001 on $10,000, far below any real day's move and far above the
# float noise the equity curve can carry (~1e-16 of the balance).
FLAT_RETURN_TOLERANCE: float = 1e-12

# Number of worker threads used by trader.py for both of its pools: the
# pre-execution order-book re-checks (pre_execution_check) and the per-pair
# execution of the selected portfolio (execute_trades). Each pool is sized
# min(this, len(work)), so a small portfolio never over-provisions threads.
# The portfolio is at most a handful of pairs in practice, so this is a
# ceiling rather than a tuned throughput figure; unlike the fetch pools it has
# never been exercised at scale against the live API. Raise cautiously — the
# execution pool submits real orders, so each extra worker is another
# concurrent write against the account. The workers' writes share one pacer
# (ORDER_WRITES_PER_SECOND below), so more workers mean longer waits for a
# pair's NO leg, never faster writes. A pair's YES leg never waits, and an
# unwind waits only behind other unwinds, 1/ORDER_WRITES_PER_SECOND s each
# (trader._PairWrites). On the V2 path the execution pool runs pairs one at a
# time until the process's NO-leg mapping check has given a verdict (for at
# most V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS, above), and concurrently after
# that.
TRADER_MAX_WORKERS = 8

# How fast trader.py sends order and collateral-transfer POSTs, across every
# worker thread together: at most ORDER_WRITE_BURST back to back, then one
# every 1/ORDER_WRITES_PER_SECOND seconds (trader._ORDER_WRITE_PACER, a token
# bucket that refills at this rate up to this burst). Kalshi limits writes per
# account with a token bucket of its own that refills continuously, not per
# window: the Basic tier refills 100 tokens a second into a 100-token bucket
# (GET /account/limits), and an order or transfer POST costs the default 10
# tokens (GET /account/endpoint_costs lists no override for either), so the
# exchange accepts 10 orders a second and 10 back to back. Beyond that it
# answers HTTP 429 and rejects the request outright, unprocessed, which on a
# YES leg means an unhedged NO leg and a rollback, and on the rollback itself
# an open position. In any T seconds the pacer admits at most 8 + 8*T writes,
# never more than the exchange's 10 + 10*T; 8 and 8 leave 20% of that budget
# for writes the bot does not see (another client on the account, a manual
# order). Pacing sets when a request is sent, not when it arrives, so large
# network jitter can still bunch arrivals, and the pacer is per process, so a
# second process writing to the account (a probe's transfer, or a run on
# another computer; run_lock keeps two real-money runs on one machine from
# overlapping) paces itself separately at the full rate. A higher
# usage tier (GET /account/limits names the account's own) allows more; these
# are safe to raise only up to that tier's write budget divided by the order
# cost (10 tokens). The burst must be at least 2: a pair's NO leg also holds
# a place for its YES leg (trader._WritePacer refuses less at import).
ORDER_WRITES_PER_SECOND = 8
ORDER_WRITE_BURST = 8

# Version name of backtester._can_ever_enter(), the backtest's eligibility
# PREFILTER (a per-market test that drops settled markets no simulated trade
# could use). The backtest's market list is cached on disk under this name
# plus SCHEDULED_RUN.cache_slug() (backtester._prefilter_cache_tag), in the
# file name (settled_markets_<start_date>_<tag>[_nomve].jsonl.gz, or .json
# when legacy) and meta block, so a list filtered one way is never served to
# another. MUST get a new version name whenever _can_ever_enter's logic or
# CANDLESTICK_PERIOD_INTERVAL_MINUTES changes, or a stale cache is served
# (_prepare_candidates' re-check WARNs on a stricter prefilter, but a looser
# one goes unnoticed). Past versions: CLAUDE.md's prefilter gotcha.
SETTLED_PREFILTER_CACHE_TAG = "checkpoint-v3"

# How young an EMPTY assembled settled-market cache must be to still be served
# (DR-13, P2 of the 2026-09-24 review). An empty corpus is not a result: it
# records only that nothing qualified when it was assembled (or that a run was
# cut short), and a cache hit makes zero network calls, so an empty cache used
# to be a PERMANENT hit — the 2-byte "[]" settled_markets_2026-08-29_*.json on
# disk, last written 2026-09-01 00:16 UTC (its file time), was still being
# served on 2026-09-24. Under this age (measured from the assembly time the
# streamed cache's meta records, or a legacy .json's file time) an empty cache
# is served with a WARNING; at or over it, when its assembly time cannot be
# read, or when that time is in the future, it is a miss and the corpus is
# re-assembled. So a legitimately empty window re-checks at most once per this
# interval — and each re-check is an ordinary miss: a full re-assembly of the
# window from the day-slice stores (fetching any day not stored or no longer
# valid, and always the current day), not a top-up, so for a long window it is
# a full-volume run (the 2026-08-29 file's rebuild covers 26 days; the fresh
# 7-day 2026-09-17 run of 2026-09-24 assembled 7,274,215 records, took 2,382 s
# and peaked at 2,916,679,680 bytes max RSS). A NON-empty cache is never expired
# by age — it is announced (its assembly time and what --no-cache costs to
# extend it) and served; that is the operator decision "announce, don't
# enforce". One day, in seconds; the same value as, but deliberately not the
# same constant as, historical's private _DAY_SECONDS (a day-slice geometry
# fact) and backtester's private _DAY_SECONDS (a calendar step), and as
# historical's private _EMPTY_CANDLE_TTL_SECONDS — the older, separate staleness
# rule for an EMPTY candlestick cache file, which (unlike this one) serves a
# future-dated file; that rule predates this one and is left as it is.
EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS = 86_400

# How long historical.load_series_categories reuses its cached copy of Kalshi's
# /series listing (every series' official category and tags — 14,391 series in
# ONE response, measured 2026-09-25) before fetching it again. It feeds the
# backtest dashboard's category/tag views and, with a live filter set,
# main._filter_by_category (a dev run reads the cached copy however old). A
# series' category rarely changes, so a week is plenty; a series the copy lacks
# is filed by historical.series_labels' fallback until the next refresh.
# Seven days in seconds.
SERIES_CATEGORY_CACHE_MAX_AGE_SECONDS = 7 * 86_400

# Hard cap on the per-ticker event-title fallback in
# historical._load_or_build_event_titles. That fallback exists for the handful
# of archived events the bulk listings no longer carry, and it costs ONE HTTP
# GET per ticker. At current Kalshi volumes a 3-week backtest can reach the
# fallback with hundreds of thousands of unresolved tickers (live-measured
# 2026-08-03: 289,235 unique event_tickers for a 21-day window) — sequentially
# that is many hours with no visible progress, which reads as a hang.
#
# How the cap is spent (DR-51, 2026-09). When every ticker still unresolved
# after the bulk listings fits under the cap, all of them are looked up. When
# they do not, the cap is spent on NON-combo tickers only, and every combo
# ticker — the KXMVE family, MVE_SERIES_FAMILY_PREFIX, read through
# scanner.event_series — is deferred. The non-combo slice is taken in ticker
# order, tickers the accumulator has never answered first, so a later run
# resolves the next slice: with the cache on because answered tickers are no
# longer unresolved, and under --no-cache (which re-resolves every requested
# ticker) because the never-answered ones outrank those it already holds.
# A combo's event title has no measured effect on which pairs form: the
# one-series rule refuses every combo-vs-combo same-title pair (DR-54/DR-55),
# and on the 2026-09-08/09 day slices no combo record shared even the coarser
# time-series grouping key with any non-combo record (CLAUDE.md, DR-67 notes).
# A non-combo title can (TS-11), and the old single sorted list spent the whole
# cap on the tickers sorting before "KXMVE" plus the head of the combo block:
# on the 2026-09-24 7-day run 3,987,139 tickers reached the fallback, 5,000
# were looked up and 3,982,139 skipped, and one non-combo event sorting after
# "KXMVE" (KXNFLEVERYWEEKCOMPETE-27, 34 markets) was skipped with them.
#
# Deferring combos changes an INPUT to grouping, not a rule. A combo the old
# cap reached was looked up and titled: that run's census logged event_title on
# 27,934 of 7,274,215 eligible records, and 18,671 of those were among its
# 18,705 non-combo records, so 9,263 combo records carried a title. A deferred
# combo now stays blank unless the accumulator already holds its title, which
# moves those markets' grouping keys and the backtest's event_title and
# deadline-phrasing census figures (a post-DR-51 fresh backtest's outcome-label
# census is not comparable with a pre-DR-51 one) but not which pairs form: the
# 2026-09-08/09 measurement above was taken on day-slice records, which carry
# no event_title at all (it is patched in at assembly), i.e. with every
# combo's event title already blank.
#
# A deferred ticker is not looked up and nothing is stored for it: it is
# untitled for that run unless the accumulator already holds its title, and a
# later run tries it again. Storing "" for it grew
# backtest_cache/event_titles.json from 3,996,906 to 7,986,570 keys (202 MB to
# 374 MB) in that one run, 7,918,449 of them KXMVE tickers mapped to "", which
# every later fetch loaded whole. Only a genuine answer is stored — a title,
# or "" for a lookup that failed or an event listed without one. Correctness is
# unaffected either way: an untitled market groups by market title alone,
# exactly as after a failed lookup, and the log names how many were deferred.
EVENT_TITLE_FALLBACK_MAX_LOOKUPS = 5_000

# Worker threads for the per-ticker event-title fallback. Each lookup is an
# independent read-only GET, so this is pure I/O overlap — the same rationale
# (and the same retry-per-worker behaviour) as SETTLED_FETCH_MAX_WORKERS.
EVENT_TITLE_FALLBACK_MAX_WORKERS = 8

# Pause, in seconds, that each event-title fallback worker takes after every
# lookup (DR-51) — the same 0.15 s each fetch_candlesticks worker takes between
# pages (its rate_limit_sleep default), against the same API with the same
# worker count. Unpaced, the fallback's 5,000 lookups on the 2026-09-24 7-day
# run took 2m03s (about 41 requests/s) and drew 268 HTTP 429s. Read the
# evidence for what pacing does and does not buy: the candlestick fetch paced
# this way ran at about 24 requests/s (37,326 single-request tickers in 25m56s
# on 2026-09-13) and still drew 429s on 4-5% of its requests, all retried by
# api_call_with_retry, as the 268 were. So this lowers the aggregate rate; it is
# not a guarantee of zero 429s. What removes the storm at bulk-window volume is
# the budget rule above: combo tickers no longer reach the fallback at all.
EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS = 0.15

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
# the page cap above. The tail and the two sequential fallbacks are the fetch
# walks with no chunked `emit` sink (the day workers stream into slice files;
# the live frontier streams through a prefilter-applying sink into an
# anonymous temporary spool file, historical._FrontierSpool), and
# the tail is the one of them that applies no prefilter either, so its whole
# unfiltered result is resident at once. ARCHIVE_TAIL_MAX_PAGES alone bounds
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
# 60 (hourly) is the finest the endpoint serves (period_interval=1 returns
# HTTP 400). Daily (1440) will not do: it emits a bar only for a market open
# across a UTC midnight, and most Kalshi markets open and close within one
# day, so daily candles would leave most markets with no prices at all.
# The backtest's prefilter reads this period too (backtester._checkpoint_floor):
# changing it requires a new SETTLED_PREFILTER_CACHE_TAG.
CANDLESTICK_PERIOD_INTERVAL_MINUTES = 60

# The most candles /historical/markets/{ticker}/candlesticks serves in ONE
# request. A request spanning more is rejected with HTTP 400 "max
# candlesticks: 5000" — observed during the DR-73 calibration fetch and
# recorded in .git/dr73-calib/fetch_ladder_candles.py. The boundary is
# consistent with that number: across the 3,704 time-series legs of the
# 2026-09-23 calibration corpus, the longest request with a cached series
# spanned 4,993.97 hours, while all 115 requests spanning more than 5,000
# hours (the shortest 5,005.33) came back with no candles and left no cache
# file, which is what a failed request leaves. historical.fetch_candlesticks —
# the only reader — therefore sends any window longer than (this - 1) candle
# periods as consecutive requests of at most (this - 1) periods each,
# overlapping by one period, and merges them ascending by timestamp with the
# overlap's repeats dropped (historical._candle_request_windows /
# _merge_candle_pages); one period short of the cap so a request stays within
# it whether the endpoint counts a span's two ends inclusively or not. A
# window that fits is still ONE request, sent exactly as before. Until that
# paging, a longer window went out as a single
# request whose 400 fetch_candlesticks reads as "no candles" (it fail-softs
# every fetch error to [], never cached), so every market whose window — then
# opened at the backtest's --start-date and run to a day past its close —
# spanned more than about 208 days of hourly candles silently had no price
# series and could never enter a backtest trade.
CANDLESTICK_MAX_CANDLES_PER_REQUEST = 5000


def min_price_diff_for_gap(gap_days: int, spread_min: float | None = None, *,
                           tier_floors: bool = True) -> float:
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

    spread_min is a band floor layered ON TOP of the tier: max(tier,
    spread_min), or the tier alone for None. Only an explicit
    tier_floors=False drops the tier (the band floor alone, 0.0 with none),
    so a stray value falls back to the tiered rule. The live path reaches
    this only through live_time_series_floor. It does not validate
    spread_min: a caller resolves it through
    time_series_spread_band() first, which does — as backtester._find_entry,
    the one backtest caller that filters on it, does (backtester's other two,
    _interval_calibration's labels and _tier_floors_bind, are handed bands
    already resolved that way), and as LiveSettings does for the live band.

    Args:
        gap_days (int): Calendar days between the two legs' deadlines —
            their close_times for a cross-event pair, their stated deadlines
            for a same-event ladder. Range: 0..MAX_DEADLINE_GAP_DAYS
            (caller-enforced).
        spread_min (float | None): Band floor on pB - pA, dollars in
            [0, 1) — the first element of a band resolved by
            time_series_spread_band(). None (default) means "the tier alone".
        tier_floors (bool): Keyword-only; False drops the tier (default True).

    Returns:
        float: The minimum required YES ask price difference (dollars, 0-1)
            by which the later leg must exceed the earlier one (later by
            close_time, or by stated deadline for a DR-73 ladder): the tier
            when spread_min is None, else the larger of the tier and
            spread_min — or, with tier_floors False, spread_min alone (0.0
            when it is None).
    """
    if tier_floors is False:
        # The band floor alone, as a float even when there is no floor: the
        # tier is not consulted at all (see above)
        return 0.0 if spread_min is None else spread_min
    tier = (MIN_PRICE_DIFF_SHORT_GAP if gap_days <= SHORT_DEADLINE_GAP_DAYS
            else MIN_PRICE_DIFF_LONG_GAP)
    return tier if spread_min is None else max(tier, spread_min)


def time_series_spread_band(band: tuple[float, float] | None = None) -> tuple[float, float]:
    """
    Resolve and validate a time-series spread band (floor, ceiling).

    The band bounds the YES-ask spread pB - pA at which
    backtester._find_entry may enter a time-series candidate: its floor is
    layered on the deadline-gap tier through min_price_diff_for_gap's
    spread_min (so it also sets that pass's leg-price-sum ceiling, 1 minus
    the raised floor), and its ceiling is tested per Monday by
    time_series_spread_too_wide. It validates the backtest's bands and the
    live band (LiveSettings.__post_init__); no live module calls it.

    Validation is deliberately TIER-AGNOSTIC: it guarantees floor < ceiling,
    not a non-empty EFFECTIVE band. The effective floor is max(tier, floor)
    (the floor alone in the backtest's tier-floors-off family, where no tier
    can empty a band), so a ceiling below MIN_PRICE_DIFF_LONG_GAP refuses
    every 16-30-day pair,
    and one below MIN_PRICE_DIFF_SHORT_GAP refuses every pair — e.g.
    (0.20, 0.25) empties the long tier and (0.0, 0.10) empties both; a
    ceiling exactly ON a tier keeps only spreads sitting on that tier. No
    SPREAD_BAND_SWEEP_* band can do this (every grid ceiling sits above both
    tiers); a caller that accepts an operator-typed ceiling should warn when
    it sits at or below a tier, so an emptied tier is not read as a strategy
    result — backtest.main does for the backtest's primary band, and
    live_rule_warnings for the live band.

    The default is resolved at CALL time, never bound as a default argument,
    so a test that monkeypatches BACKTEST_DEFAULT_SPREAD_BAND still takes
    effect — the same idiom as time_series_profit_prob's k. Both elements are
    returned as floats, and a negative-zero floor is normalised to +0.0, so
    bands given as (0, 1), (0.0, 1.0) and (-0.0, 1.0) resolve to the same
    tuple, print the same ("%g-%g" gives "0-1") and label the same scenario.

    Args:
        band (tuple[float, float] | None): (floor, ceiling) override, dollars.
            None (default) reads BACKTEST_DEFAULT_SPREAD_BAND.

    Returns:
        tuple[float, float]: The resolved (floor, ceiling), with
            0 <= floor < ceiling <= 1.

    Raises:
        ValueError: If the band does not unpack to exactly two values, or
            unless 0 <= floor < ceiling <= 1 (a NaN fails every comparison
            and is refused too). This is a caller bug, not a user-input path:
            a CLI that accepts a band validates what the operator typed
            first, with its own parser error.
        TypeError: If the band is not iterable, or an element cannot be
            compared with a float.
    """
    lo, hi = BACKTEST_DEFAULT_SPREAD_BAND if band is None else band
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            "time-series spread band must satisfy 0 <= floor < ceiling <= 1, "
            f"got ({lo!r}, {hi!r})"
        )
    # + 0.0 turns a -0.0 floor (which passes 0.0 <= -0.0) into +0.0, so it
    # cannot print as "-0"; the ceiling is > floor >= 0, so it is never a zero.
    return float(lo) + 0.0, float(hi)


def time_series_spread_too_wide(spread: float, spread_max: float | None) -> bool:
    """
    Return True when a time-series YES-ask spread exceeds the band's ceiling.

    spread is pB - pA, the market-implied in-between mass the strategy
    disputes. PRICE_EPSILON is absorbed on the KEEP side (TS-09): a spread is
    refused only when it exceeds spread_max by MORE than the tolerance, so
    0.90 - 0.30 == 0.6000000000000001 is kept at a 0.60 ceiling — a pair
    sitting exactly on the documented bound is never dropped for float noise.
    This is the ONE place the ceiling's epsilon lives; callers test the
    result and add no tolerance of their own: backtester._find_entry;
    backtest.main and live_rule_warnings, to warn when a ceiling empties a
    range; and time_series_spread_refusal, through which the live finder,
    enrichment and validate_pair_price reach it. No live module calls it
    directly.

    Args:
        spread (float): pB - pA, dollars.
        spread_max (float | None): The band ceiling, dollars in (0, 1] — the
            second element of a band resolved by time_series_spread_band().
            None means no ceiling.

    Returns:
        bool: True when spread > spread_max + PRICE_EPSILON; always False
            when spread_max is None.
    """
    if spread_max is None:
        return False
    return spread > spread_max + PRICE_EPSILON


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

    The optional k overrides that constant for one call; every live call
    passes the run's LiveSettings.interval_discount (strategy._kelly_p_at), or
    main.py --interval-discount would be ignored. None is resolved at call
    time, so a test that monkeypatches the constant takes effect.

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


def kelly_budget(bankroll: float, fraction: float, cash: float | None = None) -> float:
    """
    Return what one trade may spend: a fraction of the bankroll, never more than the cash.

    Kelly fractions are taken of the whole bankroll — the portfolio value, cash
    plus what the open positions are worth — but only cash buys contracts, so
    the budget is min(bankroll * fraction, cash). The one definition of that
    rule: max_affordable_pairs turns it into a contract count for enrichment's
    depth bound and for the sizer, and strategy._evaluate_size reads it as the
    budget compute_trade's fee shrink fits the trade into.

    Units in are units out: pass dollars and get dollars, or cents and get
    cents.

    Args:
        bankroll (float): The value Kelly fractions are taken of. Range: >= 0.
        fraction (float): The (capped) Kelly fraction to spend. Range: [0, 1].
        cash (float | None): The cash on hand, in the bankroll's units. None
            means the bankroll is all cash, so nothing further bounds the budget.

    Returns:
        float: bankroll * fraction, or cash when that is smaller. With no cash
            the product is returned exactly as computed, in that order.
    """
    budget = bankroll * fraction
    return budget if cash is None else min(budget, cash)


def leg_cash_cents(cost_dollars: float) -> int:
    """
    Return the whole cents one order leg draws: its dollar cost rounded UP to the cent.

    The one rounding shared by trader._required_cents_by_shard (what each
    exchange shard must hold before the orders go out) and
    strategy.select_portfolio (what each trade takes from the cash left), so
    the portfolio the walk admits is never a cent short when its shards are
    funded. Up, never down: an order a fraction of a cent short of collateral
    is rejected, while a cent to spare costs nothing. The round() to 6 places
    first stops binary float noise (0.07 * 100 is 7.000000000000001) from
    claiming a whole extra cent — the same guard fee_leg_exact uses.

    Args:
        cost_dollars (float): One leg's fee-inclusive cost, in dollars. >= 0.

    Returns:
        int: The smallest whole number of cents that covers it.
    """
    return math.ceil(round(cost_dollars * 100, 6))


def max_affordable_pairs(
    bankroll_cents: int, price_sum: float, fraction: float | None = None, *,
    cash_cents: int | None = None,
) -> int:
    """
    Return the largest whole contract-pair count one trade's budget buys.

    The budget is kelly_budget: a fraction of the bankroll (the portfolio
    value), never more than the cash on hand. This is the single definition of
    the budget -> contracts step, called from both ends of the sizing pipeline
    so the two can never drift:

      * scanner.enrich_with_orderbook_prices() passes max_kelly_fraction and
        the BEST qualifying level's price sum, to bound the depth it averages.
      * strategy.compute_trade() passes the capped Kelly fraction and the actual
        prefix-average price sum, to size the trade itself.

    The scanner's call is therefore an UPPER BOUND on the sizer's when both
    are handed the same bankroll and cash: its fraction is the largest the
    sizer can return under the same settings, its price sum the minimum any
    prefix average can reach (levels ascend), and a smaller fraction can only
    lower min(bankroll * fraction, cash).

    Both live callers pass fraction, since None reads BUDGET_FRACTION rather
    than the run's own per-trade cap. None is resolved at CALL time,
    so a test that monkeypatches BUDGET_FRACTION still takes effect — the same
    rule, for the same reason, as time_series_profit_prob's k.

    Args:
        bankroll_cents (int): The portfolio value Kelly fractions are taken
            of, in integer cents. Range: >= 0.
        price_sum (float): Combined per-contract cost of the two legs, in
            dollars. Range: (0, 2); a nonpositive value returns 0 rather than
            raising, since it means the book carried no usable level.
        fraction (float | None): Fraction of the bankroll to spend. None (the
            default) reads BUDGET_FRACTION. Range: [0, 1].
        cash_cents (int | None): Keyword-only. The cash on hand, in integer
            cents; the budget never exceeds it. None (the default) means the
            bankroll is all cash.

    Returns:
        int: Floor of (min(bankroll_dollars * fraction, cash_dollars) /
            price_sum). 0 when the budget cannot afford a single contract
            pair, or when price_sum is nonpositive.
    """
    f = BUDGET_FRACTION if fraction is None else fraction
    if price_sum <= 0:
        # A nonpositive sum means no usable level; "affords nothing" is the
        # right answer and keeps every caller free of a ZeroDivisionError guard
        return 0
    cash = None if cash_cents is None else cash_cents / 100.0
    # Dollars first, then the fraction, then the division: with no cash this is
    # exactly the float (bankroll_cents / 100.0) * f / price_sum
    return int(kelly_budget(bankroll_cents / 100.0, f, cash) / price_sum)


# Why time_series_spread_refusal refused a spread (None: admitted). Compare
# these with ==, like scanner's REFUSED_* reasons.
SPREAD_NOT_POSITIVE = "not positive"
SPREAD_BELOW_FLOOR = "below floor"
SPREAD_ABOVE_CEILING = "above ceiling"


def _step_cap(value, name: str) -> float:
    """
    Validate a size cap and normalise it onto the SIZE_CAP_STEP grid.

    Normalised with round(SIZE_CAP_STEP * steps, 2), backtester.SIZE_CAP_SWEEP's
    own expression, so a cap from any source is float-equal to a grid cell.

    Args:
        value: The cap, a real number in (0, 1].
        name (str): The field's name, for the error message.

    Returns:
        float: The cap on the grid.

    Raises:
        ValueError: If the value is not a real number, is outside (0, 1], or is
            not a whole number (one or more) of SIZE_CAP_STEPs.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    steps = round(value / SIZE_CAP_STEP)
    # steps < 1: a positive cap within PRICE_EPSILON of 0 would otherwise pass
    # the multiple test as zero steps and return 0.0, on no cell of the grid
    if steps < 1 or abs(value - steps * SIZE_CAP_STEP) > PRICE_EPSILON:
        raise ValueError(f"{name} must be a multiple of {SIZE_CAP_STEP:.0%} "
                         f"from {SIZE_CAP_STEP:.0%} to 100%, got {value!r}")
    return round(SIZE_CAP_STEP * steps, 2)


def _names(value, name: str) -> tuple[str, ...] | None:
    """
    Validate a category or tag filter and normalise it to a tuple of names.

    None (any) passes through; anything else must be a non-empty tuple or list
    of strings, each non-empty once stripped. A bare str is refused (iterating
    it would filter on single characters), and so is "any" in any case (it
    would render exactly like None on the "Live settings:" line and the
    trade-log note).

    Args:
        value: The filter (TRADE_CATEGORIES / TRADE_TAGS, main.py --category / --tag,
            or the saved live defaults file's "categories" / "tags").
        name (str): The field's name, for the error message.

    Returns:
        tuple[str, ...] | None: The stripped names in order, or None for any.

    Raises:
        ValueError: If the value is a str, not a tuple or list, empty, or holds
            a non-string or blank name, or a name reading "any".
    """
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be None (any) or a tuple of names, got {value!r}")
    if not value:
        raise ValueError(f"{name} must name at least one, or be None for any, got {value!r}")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must hold non-empty names, got {item!r} in {value!r}")
        if item.strip().casefold() == "any":
            raise ValueError(
                f"{name} cannot hold the name {item!r} (in {value!r}): any category or "
                f"tag is null in {LIVE_DEFAULTS_FILE.name}, --any-category / --any-tag on "
                "main.py, or None in config.py")
        names.append(item.strip())
    return tuple(names)


@dataclass(frozen=True)
class LiveSettings:
    """
    One live run's strategy toggles, validated and normalised on construction.

    One per run: main._resolve_live_settings builds it from the saved live
    defaults (live_defaults()) and lays main.py's toggle flags over it with
    dataclasses.replace (re-running __post_init__), and each run mode hands
    it, always as the bare name `settings` (never the `reference` beside it,
    the saved defaults themselves), to every site that reads a toggle.
    Internal helpers REQUIRE it; a live entry point handed none resolves
    live_settings() once, which only tests and direct library calls rely on
    (pinned by tests/test_strategy.py's
    test_ast_live_path_reads_toggles_only_through_live_settings and
    tests/test_main.py::TestLiveSettingsReachEverySite).

    Attributes:
        tier_floors (bool): Whether the tier floors apply; a real bool, since
            min_price_diff_for_gap drops the tier only on an explicit False.
        spread_band (tuple[float, float]): (floor, ceiling) on pB - pA,
            0 <= floor < ceiling <= 1.
        interval_discount (float): k, in (0, 1]; at 0 every time-series pair
            would price as riskless (p = 1).
        size_cap (float): The per-trade Kelly cap for every pair, on the
            SIZE_CAP_STEP grid; 1.0 is no cap.
        same_title_size_cap (float): The extra same-title cap, on the same
            grid; default 1.0 (no extra cap).
        categories (tuple[str, ...] | None): Categories to trade; None (default) for any.
        tags (tuple[str, ...] | None): Series first tags, ANDed with categories; None for any.
        origin (str): Where these toggles' defaults were read: the saved defaults
            file with when (and from what) it was saved, or LIVE_DEFAULTS_FROM_CONFIG
            for toggles built from this module's constants (the default).
            Not compared: equal toggles are equal wherever they came from, and
            main.py's flags laid over the defaults keep it.

    Raises:
        ValueError: If any field is out of range or of the wrong type.
    """
    tier_floors: bool
    spread_band: tuple[float, float]
    interval_discount: float
    size_cap: float
    same_title_size_cap: float = 1.0
    categories: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None
    origin: str = field(default=LIVE_DEFAULTS_FROM_CONFIG, compare=False)

    def __post_init__(self) -> None:
        """
        Validate every field and normalise it in place (the object is frozen).

        Raises:
            ValueError: If any field is out of range or of the wrong type.
        """
        if type(self.tier_floors) is not bool:
            raise ValueError(f"tier_floors must be True or False, got {self.tier_floors!r}")
        # None first: time_series_spread_band would read it as the backtest's
        # default band. Its TypeError/ValueError become one ValueError naming
        # the field, as every other field reports.
        if self.spread_band is None:
            raise ValueError("spread_band must be (floor, ceiling), got None")
        try:
            band = time_series_spread_band(self.spread_band)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"spread_band must be (floor, ceiling) with 0 <= floor < ceiling <= 1, "
                f"got {self.spread_band!r}: {exc}"
            ) from exc
        object.__setattr__(self, "spread_band", band)
        k = self.interval_discount
        if isinstance(k, bool) or not isinstance(k, numbers.Real) or not 0.0 < k <= 1.0:
            raise ValueError(f"interval_discount (k) must be in (0, 1], got {k!r}")
        object.__setattr__(self, "interval_discount", float(k))
        object.__setattr__(self, "size_cap", _step_cap(self.size_cap, "size_cap"))
        object.__setattr__(self, "same_title_size_cap",
                           _step_cap(self.same_title_size_cap, "same_title_size_cap"))
        object.__setattr__(self, "categories", _names(self.categories, "categories"))
        object.__setattr__(self, "tags", _names(self.tags, "tags"))
        # One printable line: no newline, control, zero-width or text-direction
        # character, so it prints on one log line and cannot pass for other
        # text. Printable is not HTML-safe: a web page must still escape it
        if not isinstance(self.origin, str) or not self.origin.strip() \
                or not self.origin.isprintable():
            raise ValueError(f"origin must be a printable description, got {self.origin!r}")


# The seven toggles by field name: every LiveSettings field except origin
LIVE_TOGGLE_FIELDS = tuple(f.name for f in fields(LiveSettings) if f.compare)

# The live defaults `python3 -m kalshi_betting.defaults_server --seed` offers to
# save, the starting values for a first save: tier floors off, spread band
# 0-0.5, k 0.80, a 10% per-trade cap, any category or tag. One pair stakes at
# most 10% of the portfolio value: a time-series pair under the cap (1 - k is
# 0.20), a same-title pair under the lower of the cap and the 20% same-title
# cap.
# Nothing trades on it until it is confirmed on the confirmation page and
# written to LIVE_DEFAULTS_FILE (save_live_defaults, with
# LIVE_DEFAULTS_SEED_SOURCE).
LIVE_DEFAULTS_SEED = LiveSettings(
    tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.80,
    size_cap=0.10, same_title_size_cap=0.20, categories=None, tags=None)


class LiveDefaultsError(ValueError):
    """
    The saved live defaults cannot be used, or could not be saved.

    Reading: something exists at the path but is not a regular file (a link
    to a file that does not exist included), is unreadable, too large, not
    UTF-8, malformed or too deeply nested, or holds a value LiveSettings
    refuses. Saving: the source note is refused, the settings would make a
    file the reader refuses, the write fails, or the file does not read back
    as what was written. The message names the file. It is a ValueError, so a
    caller handling an invalid constant also stops on it.
    """


class LiveDefaultsMissing(LiveDefaultsError):
    """No live defaults are saved (LIVE_DEFAULTS_FILE does not exist)."""


def live_settings() -> LiveSettings:
    """
    Return config.py's own seven toggle constants as LiveSettings, validated.

    Read at call time, so a test that monkeypatches a constant here takes
    effect. It is NOT the live defaults: a live run starts only from the saved
    ones (live_defaults()). It is what a live entry point falls back to when a
    caller hands it no settings — tests and direct library calls only, since
    the AST pin makes every live call path hand the run's settings — and what
    tests save as the defaults they run under (tests/conftest.py).

    Returns:
        LiveSettings: Built from this module's toggle constants; its origin is
            LIVE_DEFAULTS_FROM_CONFIG.

    Raises:
        ValueError: If any constant is out of range.
    """
    return LiveSettings(
        tier_floors=TIME_SERIES_TIER_FLOORS,
        spread_band=TIME_SERIES_SPREAD_BAND,
        interval_discount=TIME_SERIES_INTERVAL_PROB_DISCOUNT,
        size_cap=BUDGET_FRACTION,
        same_title_size_cap=SAME_TITLE_SIZE_CAP,
        categories=TRADE_CATEGORIES,
        tags=TRADE_TAGS,
    )


# The saved file's top-level keys, in the order save_live_defaults writes them
_SAVED_KEYS = ("format", "saved_at", "source", "settings")
# How the file stamps when it was saved: UTC, to the second
_SAVED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def live_defaults_source(source) -> str:
    """
    Check the note the saved live defaults keep about what they were saved from.

    The note becomes part of LiveSettings.origin, so it must be short and one
    printable line (no newline, tab, zero-width or text-direction character):
    it then prints on one log line and cannot pass for other text. Printable
    is not HTML-safe, so a web page that shows it must still escape it.

    Args:
        source: The note; must be a str.

    Returns:
        str: The note with surrounding spaces removed; "" (no note) is allowed.

    Raises:
        ValueError: If it is not a str, is not printable, or is longer than
            LIVE_DEFAULTS_SOURCE_MAX_CHARS once stripped.
    """
    if not isinstance(source, str):
        raise ValueError(f'"source" must be a string, got {source!r}')
    if not source.isprintable():
        raise ValueError(f'"source" must be one printable line, got {source!r}')
    stripped = source.strip()
    if len(stripped) > LIVE_DEFAULTS_SOURCE_MAX_CHARS:
        raise ValueError(f'"source" must be at most {LIVE_DEFAULTS_SOURCE_MAX_CHARS} '
                         f"characters, got {len(stripped)}")
    return stripped


def _unique_keys(pairs: list) -> dict:
    """
    Build a JSON object for json.loads, refusing a key given twice.

    json.loads' default keeps the last of two equal keys without a word; the
    saved defaults must say one thing only.

    Args:
        pairs (list): The object's (key, value) pairs, in file order.

    Returns:
        dict: The object.

    Raises:
        ValueError: If a key appears more than once.
    """
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"the key {key!r} is given twice")
        out[key] = value
    return out


def _no_constant(name: str):
    """
    Refuse NaN, Infinity and -Infinity, which json.loads would otherwise read as floats.

    Args:
        name (str): The constant's spelling in the file.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"{name} is not a number the saved live defaults may hold")


def _filter_names(value, name: str) -> None:
    """
    Refuse a category or tag name that is not printable, before LiveSettings strips it.

    A zero-width or text-direction character would make two different
    filters look alike on every line that prints them. None and anything that
    is not a list pass through for LiveSettings to judge.

    Args:
        value: The filter as the file holds it (null or a list of names).
        name (str): The field's name ("categories" or "tags"), for the message.

    Raises:
        ValueError: If a name in the list is a str that is not printable.
    """
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, str) and not item.isprintable():
            raise ValueError(f'"{name}" must hold printable names, got {item!r}')


def _saved_settings(record) -> LiveSettings:
    """
    Turn a parsed saved-defaults record into LiveSettings, or refuse it.

    Checks what LiveSettings cannot see in JSON: the file's keys and format, the
    save time, the source note, exactly the seven toggle names, a spread band of
    two real numbers (LiveSettings would read a JSON true as 1) and printable
    filter names. LiveSettings then validates every value.

    Args:
        record: What json.loads returned for the file.

    Returns:
        LiveSettings: The toggles, their origin naming the file, when it was
            saved and (when there is one) the source note.

    Raises:
        ValueError: Naming the first rule the record breaks.
    """
    if not isinstance(record, dict) or set(record) != set(_SAVED_KEYS):
        raise ValueError(f"must be one JSON object with exactly the keys {', '.join(_SAVED_KEYS)}")
    if record["format"] != LIVE_DEFAULTS_FORMAT:
        raise ValueError(f'"format" must be {LIVE_DEFAULTS_FORMAT!r}, got {record["format"]!r}')
    saved_at = record["saved_at"]
    try:
        # Written back in the same format, it must be the same text: strptime
        # alone would take "2026-9-7T1:2:3Z"
        exact = datetime.strptime(saved_at, _SAVED_AT_FORMAT).strftime(_SAVED_AT_FORMAT) == saved_at
    except (TypeError, ValueError) as exc:
        raise ValueError(f'"saved_at" must be a UTC time like 2026-09-27T21:05:13Z, '
                         f"got {saved_at!r}") from exc
    if not exact:
        raise ValueError(f'"saved_at" must be a UTC time like 2026-09-27T21:05:13Z, '
                         f"got {saved_at!r}")
    source = live_defaults_source(record["source"])
    raw = record["settings"]
    if not isinstance(raw, dict) or set(raw) != set(LIVE_TOGGLE_FIELDS):
        raise ValueError(f'"settings" must hold exactly {", ".join(LIVE_TOGGLE_FIELDS)}')
    band = raw["spread_band"]
    if not (isinstance(band, list) and len(band) == 2 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in band)):
        raise ValueError(f'"spread_band" must be [floor, ceiling], got {band!r}')
    for name in ("categories", "tags"):
        _filter_names(raw[name], name)
    origin = f"{LIVE_DEFAULTS_FILE.name}, saved {saved_at}" + (f" from {source}" if source else "")
    return LiveSettings(**{**raw, "spread_band": tuple(band)}, origin=origin)


def _settings_from_bytes(data: bytes) -> LiveSettings:
    """
    Parse a saved live defaults file's bytes into LiveSettings, or refuse them.

    The one parse read_saved_live_defaults applies to the file on disk and
    save_live_defaults applies to the text it is about to write, so a save
    can never put in place a file the reader refuses.

    Args:
        data (bytes): The file's bytes.

    Returns:
        LiveSettings: The toggles, their origin naming the file, when it was
            saved and (when there is one) the source note.

    Raises:
        ValueError: The bytes are over LIVE_DEFAULTS_MAX_BYTES, are not UTF-8
            or not JSON (UnicodeDecodeError and JSONDecodeError are
            ValueErrors), hold a repeated key or NaN / Infinity, or break a
            rule of _saved_settings or LiveSettings.
        TypeError: A value's type trips a check that raises TypeError rather
            than ValueError; both callers treat it as a refusal.
        RecursionError: The document is nested too deeply to parse.
    """
    if len(data) > LIVE_DEFAULTS_MAX_BYTES:
        raise ValueError(f"over {LIVE_DEFAULTS_MAX_BYTES} bytes")
    record = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys,
                        parse_constant=_no_constant)
    return _saved_settings(record)


def read_saved_live_defaults() -> LiveSettings | None:
    """
    Read the saved live defaults (LIVE_DEFAULTS_FILE), or None when none are saved.

    live_defaults reads through it; scheduler._check_live_defaults and
    defaults_server._current_defaults call it directly.

    Strict, because a live run trades what it returns. The file is one JSON
    object with these keys:
    - "format": LIVE_DEFAULTS_FORMAT;
    - "saved_at": UTC, e.g. 2026-09-27T21:05:13Z;
    - "source": a note (live_defaults_source's rules);
    - "settings": the seven toggles by LiveSettings field name. tier_floors is
      true/false, spread_band is [floor, ceiling], interval_discount, size_cap
      and same_title_size_cap are numbers (the caps as fractions), and
      categories and tags are null (any) or a list of names.

    Refused on top of that: a repeated key, NaN or Infinity, a file over
    LIVE_DEFAULTS_MAX_BYTES, any value LiveSettings rejects, and anything at
    the path that is not a regular file (a directory or a FIFO, say, which is
    refused without waiting on it, or a link to a file that does not exist).
    A link to a regular file is read through.

    Returns:
        LiveSettings | None: The toggles, their origin naming the file and when
            they were saved; None when nothing is at the path.

    Raises:
        LiveDefaultsError: Something exists at the path but breaks a rule
            above; the message names the file.
    """
    path = LIVE_DEFAULTS_FILE
    try:
        # O_NONBLOCK: a FIFO (named pipe) at the path opens at once instead of
        # waiting for a writer, and is refused below; a regular file reads as usual
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        # A link whose target does not exist is something at the path, so it
        # is refused rather than read as "no defaults saved yet"
        if os.path.islink(path):
            raise LiveDefaultsError(
                f"{path}: cannot be read (a link to a file that does not exist)") from exc
        return None
    except OSError as exc:
        raise LiveDefaultsError(f"{path}: cannot be read ({exc})") from exc
    try:
        # A directory, FIFO or device at the path is not a saved file
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise LiveDefaultsError(f"{path}: cannot be read (not a regular file)")
        with os.fdopen(fd, "rb") as handle:
            # The file object now owns the descriptor and closes it
            fd = None
            data = handle.read(LIVE_DEFAULTS_MAX_BYTES + 1)
    except OSError as exc:
        raise LiveDefaultsError(f"{path}: cannot be read ({exc})") from exc
    finally:
        if fd is not None:
            os.close(fd)
    try:
        return _settings_from_bytes(data)
    except (TypeError, ValueError, RecursionError) as exc:
        raise LiveDefaultsError(f"{path}: {exc}") from exc


def live_defaults() -> LiveSettings:
    """
    Return the live defaults a live run starts from: the saved ones, which must exist.

    main._resolve_live_settings is its one live caller, so a live run never
    falls back to this module's constants; the backtest's report
    (backtester._live_settings_for_report) and backtest.py's pre-fetch echo
    also read it, failing soft. With no file saved it raises, naming the two
    ways to save one.

    Returns:
        LiveSettings: The saved defaults (read_saved_live_defaults).

    Raises:
        LiveDefaultsMissing: No file is saved; the message names the file and
            the two ways to create it.
        LiveDefaultsError: The file is refused.
    """
    saved = read_saved_live_defaults()
    if saved is None:
        raise LiveDefaultsMissing(
            f"no live defaults are saved at {LIVE_DEFAULTS_FILE}: save them from the backtest "
            "dashboard's \"Save as live defaults…\" button (with ./start_dashboard.sh "
            "running), or start from the seed values with ./start_dashboard.sh --seed (or "
            "python3 -m kalshi_betting.defaults_server --seed) (live runs never fall back to "
            "config.py's toggles)")
    return saved


def _sync_directory(directory: Path) -> None:
    """
    Flush a directory's entries (a rename just made in it) to disk.

    Uses F_FULLFSYNC where the platform has it (macOS, whose plain fsync does
    not flush the drive's own cache), and plain fsync otherwise or when the
    filesystem refuses F_FULLFSYNC.

    Args:
        directory (Path): The directory to flush.

    Raises:
        OSError: If the directory cannot be opened or flushed.
    """
    fd = os.open(directory, os.O_RDONLY)
    try:
        full_sync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_sync is not None:
            try:
                fcntl.fcntl(fd, full_sync)
                return
            except OSError:
                # Some filesystems (network or FAT volumes) refuse it: fall
                # back to fsync below
                pass
        os.fsync(fd)
    finally:
        os.close(fd)


def _saved_text(record: dict) -> str:
    """
    Render a saved-defaults record as the file's text.

    Valid JSON, laid out for a person to read: one key per line, and each
    toggle's value (the spread band and any names too) on its own key's line.

    Args:
        record (dict): The record, with _SAVED_KEYS in that order and
            "settings" holding the seven toggles as JSON values.

    Returns:
        str: The text, ending in a newline.

    Raises:
        ValueError: If a value is NaN or infinite (never true of a LiveSettings').
    """
    # json.dumps prints each value (a list included) on one line
    toggles = ",\n".join(f"    {json.dumps(name)}: {json.dumps(v, allow_nan=False)}"
                         for name, v in record["settings"].items())
    top = [f"  {json.dumps(key)}: {json.dumps(record[key], allow_nan=False)}"
           for key in _SAVED_KEYS if key != "settings"]
    top.append(f'  "settings": {{\n{toggles}\n  }}')
    return "{\n" + ",\n".join(top) + "\n}\n"


def save_live_defaults(settings: LiveSettings, *, source: str) -> LiveSettings:
    """
    Write settings as the saved live defaults, and return them as read back.

    Its one caller outside the tests is defaults_server's
    _App._post_confirm, on Confirm and save and on Confirm and trade. The
    record's text is first parsed exactly as read_saved_live_defaults
    will parse it, and must equal settings, so a file the reader would refuse
    is never written. It is then written next to the file under a name holding
    this process id, flushed, and renamed over the file, so a reader at the
    same moment sees the old file or the new one, never part of one. The
    directory is flushed too, so the rename survives a power cut. The file is
    then read back from disk and must equal settings (the seven toggles;
    origin is not compared).

    Args:
        settings (LiveSettings): The new defaults.
        source (str): What they were saved from (live_defaults_source's rules).

    Returns:
        LiveSettings: The saved defaults as read back (origin names the file).

    Raises:
        LiveDefaultsError: source is refused, or settings would make a file
            read_saved_live_defaults refuses or reads differently (nothing is
            written in either case); the write fails; or the file on disk does
            not read back as settings.
    """
    path = LIVE_DEFAULTS_FILE
    try:
        source = live_defaults_source(source)
    except ValueError as exc:
        raise LiveDefaultsError(f"{path}: not saved: {exc}") from exc
    values = {name: getattr(settings, name) for name in LIVE_TOGGLE_FIELDS}
    record = {
        "format": LIVE_DEFAULTS_FORMAT,
        "saved_at": datetime.now(UTC).strftime(_SAVED_AT_FORMAT),
        "source": source,
        # JSON has no tuples: the band and any names as lists
        "settings": {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()},
    }
    # Parse the text exactly as the reader will before anything is written:
    # LiveSettings takes some values the file's rules refuse (a name that is
    # not printable, a filter too long for LIVE_DEFAULTS_MAX_BYTES), and such a
    # file must never replace the one in place
    try:
        text = _saved_text(record)
        staged = _settings_from_bytes(text.encode("utf-8"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise LiveDefaultsError(f"{path}: not saved: {exc}") from exc
    if staged != settings:
        raise LiveDefaultsError(
            f"{path}: not saved: it would read back as {staged!r}, not the settings given")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    replaced = False
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        replaced = True
        _sync_directory(path.parent)
    except OSError as exc:
        if replaced:
            raise LiveDefaultsError(
                f"{path}: written, but its directory could not be flushed to disk ({exc}), "
                "so a power cut could still undo the save") from exc
        raise LiveDefaultsError(f"{path}: could not be written ({exc})") from exc
    finally:
        tmp.unlink(missing_ok=True)
    saved = read_saved_live_defaults()
    if saved != settings:
        raise LiveDefaultsError(f"{path}: read back as {saved!r}, not the settings written")
    return saved


def live_time_series_floor(gap_days: int, settings: LiveSettings) -> float:
    """
    Return the live time-series entry floor on pB - pA for a deadline gap.

    max(tier, band floor) with the tier floors on, the band floor alone with
    them off — through min_price_diff_for_gap with backtester._find_entry's
    keywords, so the two paths cannot disagree about a floor. It also sets the
    price-sum ceiling, 1 - floor (scanner._pair_max_sum).

    Args:
        gap_days (int): The deadline gap (scanner.pair_gap_days), 0..MAX_DEADLINE_GAP_DAYS.
        settings (LiveSettings): The run's toggles.

    Returns:
        float: The floor, in dollars.
    """
    return min_price_diff_for_gap(gap_days, spread_min=settings.spread_band[0],
                                  tier_floors=settings.tier_floors)


def time_series_spread_refusal(
    spread: float, gap_days: int, settings: LiveSettings,
) -> str | None:
    """
    Apply the live time-series spread rule: return why a spread pB - pA is refused, or None.

    The one live definition (the finder, enrichment and validate_pair_price),
    in the order and PRICE_EPSILON placement of backtester._find_entry's tests:
    strictly positive (epsilon on the REJECT side, tightening); at or above the
    entry floor (epsilon on the KEEP side, TS-09); at or under the band's
    ceiling (time_series_spread_too_wide, where that epsilon lives).

    Args:
        spread (float): pB - pA, in dollars.
        gap_days (int): The deadline gap (scanner.pair_gap_days).
        settings (LiveSettings): The run's toggles.

    Returns:
        str | None: A SPREAD_* reason, or None when the spread is admitted.
    """
    if spread <= PRICE_EPSILON:
        return SPREAD_NOT_POSITIVE
    if spread < live_time_series_floor(gap_days, settings) - PRICE_EPSILON:
        return SPREAD_BELOW_FLOOR
    if time_series_spread_too_wide(spread, settings.spread_band[1]):
        return SPREAD_ABOVE_CEILING
    return None


def pair_size_cap(pair_type: str, size_cap: float, same_title_size_cap: float) -> float:
    """
    Return the per-trade Kelly cap for a pair of this type.

    The one definition, shared by strategy._evaluate_size, max_kelly_fraction
    and backtester._simulate_at_discount, so no path caps a pair differently.
    Anything but the exact string "time_series" reads as same-title
    (scanner.leg_sides' rule).

    Args:
        pair_type (str): The pair's type.
        size_cap (float): The cap for every pair, in (0, 1].
        same_title_size_cap (float): The extra cap on same-title pairs, in (0, 1].

    Returns:
        float: size_cap for time-series, else min(size_cap, same_title_size_cap).
    """
    if pair_type == "time_series":
        return size_cap
    return min(size_cap, same_title_size_cap)


def max_kelly_fraction(pair_type: str, settings: LiveSettings) -> float:
    """
    Return the largest capped Kelly fraction a pair of this type sizes at under settings.

    Enrichment bounds the depth it averages with this, so a lifted cap never
    averages depth no trade can use (#51); it bounds the sizer only because
    both read the run's one LiveSettings. The cap is pair_size_cap's, further
    bounded by the pair type's ceiling on f* (anything but "time_series" is
    same-title):

    time_series: f* = 1 - k*(pB - pA)/(1 - c), c = pA + nB + fee, is below
        1 - k whenever pB is at or above 1 - nB (that market's YES bid) to
        within PRICE_EPSILON, since the fee exceeds PRICE_EPSILON — which is
        why enrichment drops a pair whose later book is crossed by more than
        that, or has no current ask. The bound is 1 - k rounded to 12
        places: 1.0 - 0.8 is 0.19999999999999996, which would shift
        max_contracts down by one on round-number books. At k = 1 it is 0,
        and no time-series trade can size.
    same_title: f* = (p - c)/(1 - c) < p = SAME_TITLE_CO_RESOLVE_PROB.

    Args:
        pair_type (str): The pair's type.
        settings (LiveSettings): The run's toggles.

    Returns:
        float: The run's cap for this pair type, bounded by the ceiling above.
    """
    cap = pair_size_cap(pair_type, settings.size_cap, settings.same_title_size_cap)
    if pair_type == "time_series":
        return min(cap, round(1.0 - settings.interval_discount, 12))
    return min(cap, SAME_TITLE_CO_RESOLVE_PROB)


def _exact_number(value: float) -> str:
    """
    Render a number in %g form when that form reads back as the same number, else exactly.

    %g keeps six significant digits, so two values can print alike (0.3 and
    0.3000001); repr keeps a departure visible in describe_live_settings.

    Args:
        value (float): The number to render.

    Returns:
        str: f"{value:g}" when float() of it equals value, else repr(value).
    """
    short = f"{value:g}"
    return short if float(short) == value else repr(value)


def describe_time_series_rule(tier_floors: bool, spread_band: tuple[float, float]) -> str:
    """
    Describe the time-series entry rule in words.

    For the finder's rule and refusal lines and main._no_pairs_msg. It takes
    the two fields, not a LiveSettings, so a report from recorded values need
    not invent the rest; bounds render through _exact_number.

    Args:
        tier_floors (bool): Whether the deadline-gap tier floors apply.
        spread_band (tuple[float, float]): The band, already validated.

    Returns:
        str: e.g. "tier floors off (pB - pA must still be positive), spread
            band 0-0.5 on pB - pA".
    """
    lo, hi = spread_band
    if tier_floors:
        tiers = (f"tier floors on (≥{MIN_PRICE_DIFF_SHORT_GAP:.0%} up to "
                 f"{SHORT_DEADLINE_GAP_DAYS} days apart, ≥{MIN_PRICE_DIFF_LONG_GAP:.0%} "
                 f"for {SHORT_DEADLINE_GAP_DAYS + 1}-{MAX_DEADLINE_GAP_DAYS})")
    else:
        tiers = "tier floors off (pB - pA must still be positive)"
    if (lo, hi) == (0.0, 1.0):
        band = "no spread band"
    else:
        band = f"spread band {_exact_number(lo)}-{_exact_number(hi)} on pB - pA"
    return f"{tiers}, {band}"


def _band_text(spread_band: tuple[float, float]) -> str:
    """
    Name a live spread band on the "Live settings:" line.

    Args:
        spread_band (tuple[float, float]): A validated band (LiveSettings').

    Returns:
        str: "none" for (0, 1), else "floor-ceiling" by _exact_number, e.g. "0-0.5".
    """
    lo, hi = spread_band
    if (lo, hi) == (0.0, 1.0):
        return "none"
    return f"{_exact_number(lo)}-{_exact_number(hi)}"


def _percent_text(fraction: float) -> str:
    """
    Render a fraction of the portfolio value as a percentage, to six significant digits (%g).

    Args:
        fraction (float): A fraction in [0, 1].

    Returns:
        str: e.g. "20%" or "24.9%"; a bound live_rule_warnings flags (above
            the threshold by more than PRICE_EPSILON) never prints as the threshold.
    """
    return f"{fraction * 100:g}%"


def _cap_text(cap: float, no_cap: str) -> str:
    """
    Name a validated size cap as a percentage.

    Args:
        cap (float): A cap on the SIZE_CAP_STEP grid, in (0, 1].
        no_cap (str): What 1.0 means for this cap, in words.

    Returns:
        str: "100% (<no_cap>)" at 1.0, otherwise e.g. "20%".
    """
    if cap >= 1.0:
        return f"100% ({no_cap})"
    return _percent_text(cap)


def _names_text(names: tuple[str, ...] | None) -> str:
    """
    Name a validated category or tag filter.

    Exact where a plain join would print two filters alike: a name holding ",",
    ";" or "|" (each a separator where the filter is printed) switches every
    name to its repr. None renders "any", a name _names refuses.

    Args:
        names (tuple[str, ...] | None): A filter as LiveSettings holds it.

    Returns:
        str: "any" for None, else the names joined by ", " (e.g. "Economics, Sports").
    """
    if names is None:
        return "any"
    if any(sep in name for name in names for sep in (",", ";", "|")):
        return ", ".join(repr(name) for name in names)
    return ", ".join(names)


# Every live toggle as the "Live settings:" line (describe_live_settings) and
# the comparison of two sets of defaults (live_settings_changes) name it:
# (label, field, renderer), each exact where a short form would print two
# values alike.
_LIVE_SETTING_FIELDS = (
    ("tier floors", "tier_floors", lambda v: "on" if v else "off"),
    ("spread band", "spread_band", _band_text),
    ("k", "interval_discount", repr),
    ("per-trade cap", "size_cap", lambda v: _cap_text(v, "no cap")),
    ("same-title cap", "same_title_size_cap", lambda v: _cap_text(v, "no extra cap")),
    ("categories", "categories", _names_text),
    ("tags", "tags", _names_text),
)


def describe_trade_filter(settings: LiveSettings) -> str:
    """
    Name a run's category/tag filter, as describe_live_settings renders it.

    main._filter_by_category's lines and main._no_pairs_msg use it, so each
    spells the filter as the "Live settings:" line does.

    Args:
        settings (LiveSettings): The run's toggles.

    Returns:
        str: e.g. "categories Economics, Sports; tags any".
    """
    return f"categories {_names_text(settings.categories)}; tags {_names_text(settings.tags)}"


def describe_live_settings(settings: LiveSettings, reference: LiveSettings | None = None) -> str:
    """
    Name every live toggle on one line, marking each that departs from reference.

    main._log_live_settings logs it on every live run and main._run_prod hands
    it to the prod trade log's separator row, both against the run's reference
    toggles. A field departs when its RAW value differs, and every renderer is
    exact where a short form would print two values alike. The same-title cap
    reads "100% (no extra cap)" at 1.0: the per-trade cap still applies.

    Args:
        settings (LiveSettings): The run's toggles.
        reference (LiveSettings | None): Toggles to compare against; None
            (default) marks nothing.

    Returns:
        str: e.g. "tier floors off | spread band 0-0.5 | k 0.8 | per-trade cap
            100% (no cap) | same-title cap 20% | categories any | tags any",
            with " (default: X)" after each field that differs from reference's
            when reference is the saved live defaults (its origin is anything
            but LIVE_DEFAULTS_FROM_CONFIG), " (config: X)" when it was built
            from config.py's constants.
    """
    # The mark names what the reference is: the saved defaults, or config.py
    mark = ("config" if reference is None or reference.origin == LIVE_DEFAULTS_FROM_CONFIG
            else "default")
    parts = []
    for label, name, render in _LIVE_SETTING_FIELDS:
        value = getattr(settings, name)
        text = f"{label} {render(value)}"
        if reference is not None and getattr(reference, name) != value:
            text += f" ({mark}: {render(getattr(reference, name))})"
        parts.append(text)
    return " | ".join(parts)


def live_settings_changes(current: LiveSettings | None,
                          proposed: LiveSettings) -> list[tuple[str, str, str, bool]]:
    """
    Compare the live defaults in force with proposed ones, toggle by toggle.

    For a page that asks before new defaults are saved: in the "Live settings:"
    line's order and words (_LIVE_SETTING_FIELDS), and, like that line's marks,
    a toggle changes when its RAW value differs.

    Args:
        current (LiveSettings | None): The saved defaults, or None when none are saved.
        proposed (LiveSettings): What a save would write.

    Returns:
        list[tuple[str, str, str, bool]]: (label, current value, proposed value,
            changed), one per toggle. With no current defaults, every current
            value is "—" and every row is changed.
    """
    rows = []
    for label, name, render in _LIVE_SETTING_FIELDS:
        new = getattr(proposed, name)
        if current is None:
            rows.append((label, "—", render(new), True))
        else:
            old = getattr(current, name)
            rows.append((label, render(old), render(new), old != new))
    return rows


def live_settings_argv(settings: LiveSettings) -> list[str]:
    """
    Spell a run's settings as main.py's toggle flags, all seven of them.

    A program that starts main.py with these flags gets a run that trades
    exactly these settings, whatever the saved live defaults say:
    main._resolve_live_settings lays every flag over the saved defaults and
    gets back settings equal to these (tests/test_main.py checks the round
    trip). Caps are written as whole percents, which LiveSettings' 5% grid
    makes exact. The other numbers are written as their repr, which float()
    reads back exactly. Each category or tag is written as --category=NAME or
    --tag=NAME, so a name that begins with "-" still reads as a name; no
    filter is written as --any-category or --any-tag.

    Args:
        settings (LiveSettings): The settings to spell.

    Returns:
        list[str]: The flags, in the order main.py lists them: the tier-floor
            switch, --spread-min, --spread-max, --interval-discount,
            --size-cap, --same-title-size-cap, then the category flags and the
            tag flags.
    """
    argv = ["--tier-floors" if settings.tier_floors else "--no-tier-floors",
            f"--spread-min={settings.spread_band[0]!r}",
            f"--spread-max={settings.spread_band[1]!r}",
            f"--interval-discount={settings.interval_discount!r}",
            f"--size-cap={round(settings.size_cap * 100)}",
            f"--same-title-size-cap={round(settings.same_title_size_cap * 100)}"]
    argv += ([f"--category={name}" for name in settings.categories]
             if settings.categories is not None else ["--any-category"])
    argv += ([f"--tag={name}" for name in settings.tags]
             if settings.tags is not None else ["--any-tag"])
    return argv


def live_rule_warnings(settings: LiveSettings) -> list[str]:
    """
    Name every setting that empties part of the time-series strategy or lifts one pair's exposure.

    main._log_live_settings logs each sentence as a WARNING on every live run.
    The first two cases judge the band's ceiling against the entry floor
    (live_time_series_floor) once per tier's gap range with the tier floors on,
    once for every gap with them off:
      EMPTIED: the ceiling refuses a spread exactly on the floor
          (time_series_spread_too_wide), so no pair in that range can trade;
          it cannot fire with the tiers off, where floor < ceiling.
      ON THE FLOOR: the ceiling is within PRICE_EPSILON of the floor, so only
          spreads on it can trade — none on a floor within PRICE_EPSILON of
          0, where such a spread is not positive.
      EXPOSURE: max_kelly_fraction of a pair type exceeds
          LIVE_EXPOSURE_WARN_FRACTION.
      k = 1: max_kelly_fraction("time_series") is 0, so no time-series trade sizes.

    Args:
        settings (LiveSettings): The run's toggles.

    Returns:
        list[str]: One sentence per warning, in the order above; empty if none.
    """
    out = []
    ceiling = settings.spread_band[1]
    if settings.tier_floors:
        # One gap from each tier's range: the entry floor is constant across
        # the range (max(tier, band floor)), so one gap judges all of it
        ranges = ((0, f"time-series pairs 0-{SHORT_DEADLINE_GAP_DAYS} days apart"),
                  (SHORT_DEADLINE_GAP_DAYS + 1,
                   f"time-series pairs {SHORT_DEADLINE_GAP_DAYS + 1}-"
                   f"{MAX_DEADLINE_GAP_DAYS} days apart"))
    else:
        # The band's own floor is the entry floor at every gap
        ranges = ((0, "time-series pairs at any deadline gap"),)
    for gap, scope in ranges:
        floor = live_time_series_floor(gap, settings)
        if time_series_spread_too_wide(floor, ceiling):
            out.append(
                f"the spread band's {_exact_number(ceiling)} ceiling is below the "
                f"{_exact_number(floor)} entry floor for {scope}, so none of them can trade")
        elif abs(ceiling - floor) <= PRICE_EPSILON:
            if floor <= PRICE_EPSILON:
                # A spread on a floor of 0 is not positive, and the live rule
                # refuses that before it tests the floor or the ceiling
                out.append(
                    f"the spread band's {_exact_number(ceiling)} ceiling sits on the "
                    f"{_exact_number(floor)} entry floor for {scope}, and pB - pA must be "
                    "positive, so none of them can trade")
            else:
                out.append(
                    f"the spread band's {_exact_number(ceiling)} ceiling sits on the "
                    f"{_exact_number(floor)} entry floor for {scope}, so only spreads "
                    "exactly on it can trade")
    for pair_type, label in (("time_series", "time-series"), ("same_title", "same-title")):
        bound = max_kelly_fraction(pair_type, settings)
        if bound > LIVE_EXPOSURE_WARN_FRACTION + PRICE_EPSILON:
            out.append(
                f"one {label} pair may stake up to {_percent_text(bound)} of the portfolio "
                f"value, above the {_percent_text(LIVE_EXPOSURE_WARN_FRACTION)} this check "
                "accepts")
    if max_kelly_fraction("time_series", settings) == 0:
        out.append(f"k = {settings.interval_discount!r}: time-series Kelly cannot be "
                   "positive, so no time-series trade can size")
    return out


def order_api_version_error() -> str | None:
    """
    Return an error message if ORDER_API_VERSION is not exactly the str "v2".

    main.main() and the human-run order-path probe's main() call this right
    after parsing their arguments, before logging is configured or any
    request is made, and pass a message to parser.error, which exits 2. Reads
    ORDER_API_VERSION at call time, so a test can monkeypatch it.

    Returns:
        str | None: None if ORDER_API_VERSION is exactly "v2"; otherwise one
            line naming the value and saying to set it to "v2".
    """
    value = ORDER_API_VERSION
    if type(value) is str and value == "v2":
        return None
    return (
        f"config.ORDER_API_VERSION is {value!r}, but \"v2\" (POST {V2_ORDER_PATH}) is the "
        "only order path this bot has: Kalshi retired the legacy /portfolio/orders order "
        "endpoint, so there is nothing to switch to. Set ORDER_API_VERSION = \"v2\" in "
        "config.py."
    )


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
