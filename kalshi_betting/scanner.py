"""
File: scanner.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Fetches all open Kalshi markets from the REST API and identifies pairs of
    contracts for the bot's two pair strategies: (1) time-series pairs —
    contracts that ask the same question about the same OUTCOME at different
    deadlines, identified by stripping date tokens from their titles and
    exact-matching the remainder together with the market's outcome label
    (time_series_group_key), traded as a directional bet (YES on the earlier
    contract, NO on the later) when the later contract is priced well above
    the earlier AND both legs' wording states a CUMULATIVE deadline
    ("by <date>") — deadline_phrasing refuses a snapshot family
    ("price ON <date>"), whose probabilities do not nest and which the date
    stripping would otherwise collapse into one group; and (2)
    same-title pairs — contracts with identical title and subtitle on
    different event tickers of DIFFERENT event series, traded as a
    near-arbitrage (NO on the pricier, YES on the cheaper) when their prices
    diverge. Both paths then check the live order book to replace best-ask
    prices with depth-weighted fill prices and confirm the edge survives real
    liquidity.

Dependencies:
    Imports constants, the leg-side tuples, and fee helpers from config.py and
    the retry/raw-fetch helpers from _http.py. Exports the CandidatePair and
    ApiMarket dataclasses, the leg helpers leg_sides()/leg_prices()/
    deadline_gap_days() (the only source of truth for which side each leg
    buys and what it costs — consumed by strategy.py, trader.py, reporter.py,
    main.py and backtester.py), and the scanning functions consumed by
    main.py and backtester.py (which also imports time_series_group_key,
    event_series, leg_sides, deadline_profile/cumulative_deadline_pair, so
    the live scanner and the backtester group time-series candidates AND
    decide their eligibility through one definition each, and
    deadline_pair_refusal with its REFUSED_SNAPSHOT/REFUSED_NO_STATED_DEADLINE/
    REFUSED_SAME_DEADLINE constants and the DEADLINE_CUMULATIVE/
    DEADLINE_SNAPSHOT/DEADLINE_UNKNOWN verdict constants, so the two can agree
    on WHY a candidate was refused (DR-72), and stated_deadline/
    same_event_ladder with the SAME_DAY sentinel, the one definition of a
    same-event deadline ladder's leg order and gap (DR-73)). pair_gap_days()
    is the single reader of that gap for everything downstream of pair
    formation. historical.py imports event_series too, so the backtest's
    event-title lookup budget tells a combo ticker from any other exactly as
    the one-series rule does (DR-51). Depends on the KalshiClient produced by
    auth.py.

Notes:
    The normalize_title() approach avoids fuzzy matching entirely — it relies on
    the observation that Kalshi titles differ only in date tokens when the same
    question is asked across multiple deadline-indexed markets. The _DATE_PATTERNS
    list must cover all Kalshi date formats to avoid missed pairs or false positives.
    It is only half of the time-series grouping key: the market's outcome label
    (its subtitle) is the other half, because a daily strike family shares one
    title and differs only there — see time_series_group_key().

    Neither finder pairs two events of ONE series (event_series(): the ticker
    prefix before the first "-", with every combo (KXMVE*) prefix collapsed
    onto one family — see config.MVE_SERIES_FAMILY_PREFIX) whose wording is
    identical. Two events of one series are two instances of one recurring
    fixture — two ball games, two 15-minute price windows, two combo tickets
    whose wording names their legs but never their date — so identical wording
    across them is the same question about two DIFFERENT events, and neither
    the 95% co-resolution prior nor the cumulative-deadline premise applies. Both
    finders carry the rule because the same two tickers qualify for both; see
    _same_series()/_identical_wording() and CLAUDE.md's one-series gotcha.

    The time-series finder additionally requires both legs to be CUMULATIVE-
    deadline markets stating two DIFFERENT deadlines (deadline_phrasing,
    cumulative_deadline_pair) — compared as normalized strings, not parsed
    calendar dates, so one deadline spelled two ways reads as two and a
    truncated spelling can read two deadlines as one (see the DR-67 Known
    residuals in CLAUDE.md). Kalshi lists SNAPSHOT markets too — "Bitcoin
    price ON Sep 15, 2026?" — whose probabilities do not nest, so the trade has
    no premise between two of them; and normalize_title erases a dated snapshot
    title just as readily as a dated deadline one, which is what put such a
    family in one group in the first place. The rule reads WORDING, not market
    class: an MVE market whose sub-contract label is a "by <date>" phrase is
    eligible like any other. It fails closed, and the backtester mirrors it —
    see CLAUDE.md's cumulative-deadline gotcha.

    Reading a cumulative deadline as a CALENDAR DATE is a separate step from
    classifying one: stated_deadline() turns a rung's wording into the last
    day its deadline includes (refusing year-less wording, and refusing when
    two of the market's own fields cannot be naming one day — judged on the
    SET of days each phrase can mean, since Kalshi spells one deadline both
    "before D" and "By D" inside a single market), and
    same_event_ladder() orders two such rungs and measures the gap between
    them — the arithmetic DR-73 needs to pair two rungs of ONE event's
    deadline ladder, where close_time cannot order them because a settled
    event closes every rung at one instant.

    Market fetching deliberately bypasses the SDK's response models: as of
    2026-07 the API stopped sending the legacy integer-cent price fields the
    pinned SDK's Market model requires, so fetch_open_events_with_markets()
    uses the *_without_preload_content raw-response variants and parses JSON
    into ApiMarket itself (see fetch_json_page for the error-handling contract).

    The exchange is sharded. Market data is cross-shard, so ingest TAGS every
    market with its exchange_index (_shard_index) and keeps it; the one
    ingest-time shard exclusion is a shard the exchange reports as not
    trading-active (fetch_shard_statuses -> the caller's inactive_shards).
    check_shard_coverage() is the pure audit that compares the advertised
    shards against the ones a run actually observed, so a shard we silently
    stopped seeing markets on cannot pass unnoticed.
"""
import calendar
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

from ._http import api_call_with_retry, fetch_json_page
from .config import (
    BUY_SLIPPAGE_TICKS,
    DEFAULT_EXCHANGE_INDEX,
    DEFAULT_TICK_SIZE_DOLLARS,
    EXCHANGE_FLAG_DRIFT_REPR_MAX_CHARS,
    EXCHANGE_FLAG_FALSE_TOKENS,
    EXCHANGE_FLAG_NULL_TOKENS,
    EXCHANGE_FLAG_TRUE_TOKENS,
    INCLUDE_MVE_MARKETS,
    MARKET_PAGE_SIZE,
    MAX_ACTIVE_PRICE_DOLLARS,
    MAX_DEADLINE_GAP_DAYS,
    MIN_ACTIVE_PRICE_DOLLARS,
    MVE_MAX_EMPTY_PAGES,
    MVE_SERIES_FAMILY_PREFIX,
    ORDER_API_VERSION,
    POSITION_PAGE_SIZE,
    PRICE_EPSILON,
    SAME_TITLE_LEG_SIDES,
    SAME_TITLE_MIN_PRICE_DIFF,
    SCANNER_MAX_PAGES,
    SCANNER_PROGRESS_LOG_EVERY_PAGES,
    TIME_SERIES_LEG_SIDES,
    TIME_SERIES_SAME_EVENT_LADDERS,
    fee_per_pair_approx,
    max_affordable_pairs,
    min_price_diff_for_gap,
)

# ---------------------------------------------------------------------------
# Date patterns stripped from titles before exact-match grouping.
# After stripping, two contracts that differ ONLY in their deadline will
# produce the same normalized string — no fuzzy matching needed. The
# time-series key appends the outcome label to that string so two different
# OUTCOMES cannot share a group (see time_series_group_key).
# ---------------------------------------------------------------------------
_DATE_PATTERNS = [
    # Full month-name dates: "December 1, 2026" / "January 31, 2027"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    # Full month-name + year only: "December 2026"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    # Abbreviated months with date+year: "Dec 1, 2026"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
    # Abbreviated months + year only: "Dec 2026"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}\b",
    # Numeric dates: "01/01/2027"
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    # ISO dates: "2026-12-31"
    r"\b\d{4}-\d{2}-\d{2}\b",
    # Deadline preposition + full month name + day, no year: "by June 30".
    # Must sit AHEAD of the "by/before/until/through/after" clause below,
    # whose optional day group matches only when a year follows: from behind,
    # that clause consumes "by June" and strands the day, leaving "30" against
    # "31" to split two titles that differ only in their deadline.
    # Anchored to the preposition deliberately. A bare "<Month> <day>" would
    # also erase the date from SNAPSHOT titles ("... on June 30"), which would
    # put two snapshot markets of one family into a single time-series group —
    # the premise violation DR-67 now screens at pair formation on the legs'
    # wording (this pattern only controls whether the two titles share a
    # group; the backtester's counter is defence in depth behind it).
    # Anchored, this pattern fires only where the clause below already fired
    # and stranded a day, so no title that used to stay apart is merged.
    # The abbreviated spelling ("by Oct 1") is deliberately NOT covered: its
    # own pattern still sits after that clause and still strands the day
    # there, exactly as before this commit.
    r"\b(?:by|before|until|through|after)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    # "by/before/until/through/after [month] [optional date+year]"
    r"\b(?:by|before|until|through|after)\s+(?:end\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?:\s+\d{1,2},?\s+\d{4}|\s+\d{4})?\b",
    # "end of [the] year"
    r"\bend\s+of\s+(?:the\s+)?year\b",
    # Quarters: "Q1 2026", "Q4"
    r"\bQ[1-4](?:\s+\d{4})?\b",
    # "in 2026"
    r"\bin\s+20\d{2}\b",
    # Standalone 4-digit years
    r"\b20\d{2}\b",
    # Short date-like suffixes often embedded in Kalshi sandbox titles:
    # "Apr 02", "Mar 21", "Apr 16" (month abbreviation + day, no year)
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b",
    # Time expressions: "at 12:00", "at 21:00", "H0650" style
    r"\bat\s+\d{1,2}:\d{2}\b",
    r"\bH\d{4}\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _DATE_PATTERNS]

# The unambiguous DATE shapes only — what _normalize_subtitle may strip from an
# outcome label. Every entry is taken BY INDEX from _DATE_PATTERNS and none is
# re-spelled here, so editing a pattern there can never leave a stale copy
# behind: indexes 0-5 are the month-name / abbreviated-month / numeric / ISO
# shapes that carry a year, index 6 is the deadline-anchored month + day shape
# added above, and index 12 is the pre-existing bare abbreviated month + day.
# Deliberately NOT the whole list. Excluded are the bare "by <Month>" clause
# (7), "end of year" (8), quarters (9), "in 20xx" (10), the standalone 20xx
# year (11) and the two time shapes (13, 14). A subtitle is where a bare
# 4-digit strike ("2050 or above") lives, so the year and quarter shapes would
# erase real outcome labels; an hour inside an outcome label may BE the
# outcome; and the bare "by <Month>" clause adds nothing a dated label needs,
# since 0-6 already cover every shape carrying a day or a year. One consequence
# to know: index 12 is unanchored, so a bare "Sep 14" outcome label normalizes
# to "" and two such labels share a key.
_EXPLICIT_DATE_PATTERNS = (
    *_DATE_PATTERNS[0:7],
    _DATE_PATTERNS[12],
)
_COMPILED_EXPLICIT_DATES = [re.compile(p, re.IGNORECASE) for p in _EXPLICIT_DATE_PATTERNS]

# ---------------------------------------------------------------------------
# Cumulative-deadline vs. snapshot phrasing.
#
# A time-series pair buys YES on the earlier contract and NO on the later one.
# That is only coherent when BOTH contracts are cumulative-deadline markets —
# "will X happen BY <date>" — because only then is the event by the earlier
# deadline nested inside the event by the later one. That nesting is what
# rules out the A=YES/B=NO settlement cell entirely, and it is what makes the
# FAIR VALUE of YES-A + NO-B at most $1 — not a guarantee that the quoted asks
# do: a wide book can still quote pA + nB >= 1 on a genuine ladder, which is
# why find_time_series_pairs skips that case explicitly rather than relying on
# nesting to rule it out.
#
# Kalshi also lists SNAPSHOT markets — "what is X ON <date>", "X IN <Month>" —
# which are NOT nested (SOL >= $180 on Sep 14 does not imply SOL >= $180 on
# Sep 18). normalize_title strips a dated snapshot title just as readily as a
# dated deadline one, so such a family lands in ONE time-series group and was
# sized and traded on a premise it does not have. These tables are what
# separates the two.
#
# DELIBERATELY SEPARATE FROM _DATE_PATTERNS, and must stay that way. That list
# exists to ERASE dates so two deadlines of one question collapse onto one
# grouping key; these exist to READ the surrounding preposition and decide what
# KIND of question it is. Merging them would couple two different ordering
# contracts — see CLAUDE.md's normalize_title pattern-order gotcha.
# ---------------------------------------------------------------------------

# Month and weekday spellings used only by the phrasing tables below.
#
# The month fragment takes an OPTIONAL day and an OPTIONAL year so a matched
# span names the WHOLE deadline rather than just its month: _deadline_spans
# compares those spans between two legs, and "by Dec 31" vs "by Dec 20" has to
# read as two different deadlines. The `(?!\d)` on the day group stops it
# eating the first two digits of a bare 4-digit year ("by March 2026" would
# otherwise capture "by March 20"), which would make every deadline in a given
# month look identical and silently refuse genuine pairs.
_DEADLINE_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)"
)
_DEADLINE_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
# The \b ends the month at a word boundary. Without it the abbreviations
# matched the front of ordinary words: "by Dec(ision)", "produced by
# Mar(vel) Studios" and "endorsed by Mar(co) Rubio" read as cumulative
# deadlines (DR-68). The optional period still follows ("Sept."). \b also
# rejects a digit or underscore directly after the month, so an unspaced
# "by Sep30" reads unknown — the fail-closed direction on this CUMULATIVE
# side (no market on the 2026-09-22 live snapshot is written that way). The
# snapshot entries use _SNAPSHOT_MONTH below, which keeps that unspaced date.
_DEADLINE_MONTH_FULL = rf"{_DEADLINE_MONTH}\b\.?(?:\s+\d{{1,2}}(?!\d))?(?:,?\s+\d{{4}})?"

# The SNAPSHOT (reject) side ends a month at the next LETTER instead of at a
# word boundary. "on Mars", "on declaring", "after Marvel" and "after Mayor"
# are still not dates, but an unspaced "on Sep30" / "after Sep30" still is:
# \b would reject that genuine date too, and narrowing a reject marker for a
# genuine date fails OPEN (snapshot -> cumulative). The tables compile with
# IGNORECASE, so [a-z] covers upper case as well (DR-68).
_SNAPSHOT_MONTH = rf"{_DEADLINE_MONTH}(?![a-z])"

# What may follow a deadline preposition for it to name a POINT IN TIME.
# Requiring one of these is what stops "cut BY 50 bps", "win BY 10 points" and
# "pass BY a 2/3 majority" from reading as deadlines. (That last one is stopped
# by the article "a" alone; the m/d alternative's vote/majority/of lookahead is
# what stops the article-less "pass by 2/3 majority" — DR-68.)
#
# The NOUN forms are not optional extras: _DATE_PATTERNS already strips
# "by end of <Month>", "end of [the] year" and "Q1 2026", so a table that
# accepted only "by <Month>" would refuse markets the grouping key already
# treats as deadline-dated.
#
# A bare CLOCK TIME is deliberately ABSENT. "$82,750 or above by 5pm" must not
# read as a cumulative deadline: the subtitle has top precedence in
# deadline_phrasing, so an intraday snapshot family would otherwise fail OPEN
# through its own sub-contract label instead of falling through to the
# "price on <date>" title where its real shape is spelled.
_DEADLINE_DATE_TOKEN = (
    rf"(?:{_DEADLINE_MONTH_FULL}"
    # A weekday keeps a MONTH-NAME date that follows it. Otherwise "by Friday,
    # Sep 19, 2026" and "by Friday, Sep 26, 2026" both truncate to "by friday",
    # read as one deadline stated twice, and a genuine pair 7 days apart is
    # refused (DR-68). A numeric date after a weekday is still truncated.
    # With no year or a 2-digit one the two titles do not share a group; with
    # a 4-digit year ("by Friday, 9/19/2026") or an ISO date they do, both
    # span "by friday", and the pair is refused as one deadline stated twice —
    # a missed trade, never a false one (CLAUDE.md DR-67 Known residuals).
    rf"|{_DEADLINE_WEEKDAY}\b(?:,?\s+{_DEADLINE_MONTH_FULL})?"
    # A calendar m/d (month 1-12, day 1-31), not a fraction or vote share:
    # "by 2/3 vote" and "ratified by 3/4 of states" are quantities, and
    # "by 3/2027" is a month/year, not March 20 (DR-68).
    r"|(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?:/(?:\d{4}|\d{2}))?\b"
    r"(?!\s*(?:votes?|majority|supermajority|of)\b)"
    r"|\d{4}-\d{2}-\d{2}"
    # 2020-2099 only: "decrease by 2000" is an amount, not a year (DR-68).
    # Season ranges ("before the 2027-28 season") deliberately stay a
    # deadline: rejecting them refused 63 genuine markets (retirements,
    # relocations, conference moves, a rule change) on the 2026-09-22 live
    # snapshot, as on the 2026-09-21 one. The cost of keeping them is one
    # recorded false positive ("2018-19 through 2025-26"; CLAUDE.md DR-67
    # Known residuals).
    r"|20[2-9]\d\b"
    r"|Q[1-4](?:\s+20\d{2})?\b"
    r"|(?:the\s+)?end\s+of\s+(?:the\s+)?(?:year|month|week|day)"
    r"|year[-\s]?end"
    r"|EOY\b)"
)

# CUMULATIVE: the event may happen at ANY time up to a deadline, so a later
# deadline can only add probability.
#
# Index 0 is load-bearing beyond this table: it is the ONE pattern whose match
# _deadline_spans reuses as a comparable deadline span, read from the field
# that DECIDED the verdict and no other (DR-69). The others establish the kind
# of question without naming a single comparable date, so a leg whose deciding
# field carries only them has no span, and a pair resting on such a leg cannot
# prove its two deadlines differ and is refused — even when a less specific
# field of that leg does name a date.
_CUMULATIVE_DEADLINE_PATTERNS = [
    rf"\b(?:by|before|prior\s+to|no\s+later\s+than|on\s+or\s+before|up\s+to|through|until)"
    rf"\s+(?:the\s+)?(?:end\s+of\s+(?:the\s+)?)?{_DEADLINE_DATE_TOKEN}",
    r"\bwithin\s+\d+\s+(?:hour|day|week|month|year)s?\b",
    r"\bat\s+any\s+(?:time|point)\b",
    # "ever" is deliberately NOT a marker (DR-68): it fired on names ("Worst
    # Neighbor Ever", "Chaco For Ever") and on superlatives ("the hottest year
    # ever"), none of which state a deadline.
]

# SNAPSHOT: a state measured AT one instant, or over a WINDOW that does not
# nest. Two entries deserve their reasoning spelled out:
#   * "after <date>" INVERTS the monotonicity the trade rests on — a later
#     "after" date carries LOWER probability, so the earlier/later leg
#     assignment means the opposite of what it says.
#   * "in <Month>" / "in <Year>" is a window, not a deadline: top-10 in October
#     does not nest inside top-10 in November.
# "after" requires a date token for the same reason "by" does — otherwise
# "Will BTC top $100k by Dec 31, 2026, after the halving?" is wrongly refused.
_SNAPSHOT_PATTERNS = [
    # "on THE Dec 6 CFP rankings" is the same snapshot as "on Dec 6", but only
    # with a day: "on the May ballot" names no instant, and (?!\d) stops the
    # day eating a year ("on the November 2026 ballot"). The month ends at the
    # next letter (_SNAPSHOT_MONTH), so "on Mars" and "on declaring" are not
    # dates while an unspaced "on Sep30" still is (DR-68).
    rf"\bon\s+(?:the\s+{_SNAPSHOT_MONTH}\.?\s*\d{{1,2}}(?!\d)"
    rf"|{_SNAPSHOT_MONTH}\.?\s*\d{{0,2}}|{_DEADLINE_WEEKDAY}"
    r"|\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2})",
    rf"\b(?:in|during|for)\s+(?:{_DEADLINE_MONTH}\b|Q[1-4]\b|20\d{{2}}\b)",
    r"\bat\s+(?:the\s+)?(?:close|open|end)\b",
    # The dotted forms end in a period, so a trailing \b required a following
    # WORD character ("at 5 p.m.ET" matched; "at 5 p.m. ET" did not). They now
    # need no boundary; the undotted forms keep theirs (DR-68).
    r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:(?:am|pm|ET|EDT|EST|CT|PT|UTC|GMT)\b|a\.m\.|p\.m\.)",
    r"\bat\s+\d{1,2}:\d{2}\b",
    r"\bas\s+of\b",
    r"\bend\s+of\s+day\b",
    # A REJECT marker, so its DATE alternatives stay as wide as they were
    # before DR-68: narrowing a reject marker for a genuine date fails OPEN
    # (snapshot -> cumulative), the direction this rule must never move for a
    # genuine date. Hence the wide numeric alternatives (any m/d, any 20xx
    # year) beside the tightened token, and _SNAPSHOT_MONTH beside the token's
    # \b-bounded month so an unspaced "after Sep30" still matches. What DOES
    # stop matching, deliberately, is a month-prefixed NON-date word — "after
    # Mayor Adams resigns", "after Marvel's release", "after Decision Day" —
    # which moves such a title snapshot -> cumulative, exactly as "on Mars"
    # does in the "on" entry above (DR-68).
    rf"\bafter\s+(?:the\s+)?(?:{_DEADLINE_DATE_TOKEN}"
    rf"|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|20\d{{2}}\b|{_SNAPSHOT_MONTH})",
    r"\bbetween\s+\d",
]

_COMPILED_CUMULATIVE = [re.compile(p, re.IGNORECASE) for p in _CUMULATIVE_DEADLINE_PATTERNS]
_COMPILED_SNAPSHOT = [re.compile(p, re.IGNORECASE) for p in _SNAPSHOT_PATTERNS]
# One alternation per table for _field_phrasing's existence tests (DR-70).
# "Some pattern matches somewhere" equals "the alternation matches somewhere"
# as long as every entry compiles with zero capturing groups and no inline
# flag, which tests/test_scanner.py::TestPhrasingTableInvariants pins. A group
# would renumber any backreference after it; a global inline flag is legal
# only at position 0 of a pattern and every entry is wrapped in (?:...), so
# inside the alternation it fails to compile (Python 3.11+, the project
# floor) and this module would not import. The test forbids scoped inline
# flags too, positive or negative, to keep the rule one line. The two
# searches per field that replace up to twelve (nine snapshot entries, three
# cumulative) measured about 2x on their own: 45.7 -> 23.0 us/record through
# deadline_profile on 200,000 records of the 2026-09-08 day slice
# (2026-09-22); with _deciding_field's duplicate-field skip, 12.3. The
# per-pattern lists stay: _deadline_spans needs index 0 alone, and the tests
# use them as the equivalence reference.
_ANY_CUMULATIVE = re.compile(
    "|".join(f"(?:{p})" for p in _CUMULATIVE_DEADLINE_PATTERNS), re.IGNORECASE,
)
_ANY_SNAPSHOT = re.compile(
    "|".join(f"(?:{p})" for p in _SNAPSHOT_PATTERNS), re.IGNORECASE,
)

# deadline_phrasing()'s three verdicts. Bare module-level strings, like the
# pair_type and TradeResult.status vocabularies — config.py owns the tunable
# constants, not the internal enumerations.
#
# NOTE ON THE NAME: "snapshot" here means a MARKET SHAPE ("what is X on <date>"),
# which is the sense CLAUDE.md, README.md and backtester.py's premise-violation
# WARNING all use. It is unrelated to enrich_with_orderbook_prices' "mixed
# snapshot" comment, which is about a stale vs. fresh PRICE quote.
DEADLINE_CUMULATIVE = "cumulative"
DEADLINE_SNAPSHOT = "snapshot"
DEADLINE_UNKNOWN = "unknown"

# Minimum ask price to consider a MARKET actively priced (not settled/illiquid).
# Distinct from config.MIN/MAX_ACTIVE_PRICE_DOLLARS (0.0001/0.9999), which
# bounds an order-book LEVEL. Do not unify them: widening this one would admit
# near-settlement markets, turning a 0.9999 YES quote into a $0.0001 hedge leg
# (TS-14, market half — a held operator decision, not an oversight).
_MIN_ACTIVE_PRICE = 0.01
_MAX_ACTIVE_PRICE = 0.99


@dataclass(frozen=True)
class PriceRange:
    """
    One tick-size band from a market's price_ranges array, in dollars.

    Kalshi markets can have a non-uniform tick grid across their price range
    (e.g. finer ticks near 0 and 1, coarser in the middle — "tapered" tick
    structure). Each band names its own step size; a market's full grid is
    the ordered list of bands covering [0, 1]. Read by tick_size_for_price().

    Attributes:
        start (float): Lower bound of this band, in dollars (e.g. 0.0).
        end (float): Upper bound of this band, in dollars (e.g. 1.0).
        step (float): Tick size within this band, in dollars (e.g. 0.001).
    """
    start: float
    end: float
    step: float


def _parse_price_ranges(raw: Any) -> list | None:
    """
    Parse a raw price_ranges array into PriceRange bands, or None if unknown.

    Fail-soft per the return-None convention: a missing, empty, or malformed
    array means "tick structure unknown", never an error — tick_size_for_price()
    then falls back to the default $0.01 grid, which is valid on every regime
    because Kalshi's grids are nested (combo markets moved to the
    center_deci_edge_centi_cent tick regime on 2026-08-17).

    Args:
        raw (Any): The raw `price_ranges` value from a market JSON dict —
            expected to be a list of {"start": str, "end": str, "step": str}
            dollar-string dicts, but may be missing, empty, or malformed.

    Returns:
        list[PriceRange] | None: Parsed bands in their original order, or
            None when raw is not a non-empty list, or when every band in it
            fails to parse. Individual malformed bands within an otherwise
            valid list are skipped rather than failing the whole array.
    """
    if not isinstance(raw, list) or not raw:
        return None
    bands = []
    for band in raw:
        try:
            bands.append(PriceRange(
                start=float(band["start"]), end=float(band["end"]), step=float(band["step"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return bands or None


def tick_size_for_price(market: Any, price_dollars: float) -> Decimal:
    """
    Return the tick size, in dollars, that applies at a given price on a market.

    Kalshi markets no longer share one uniform price grid. Known regimes, named
    by `price_level_structure`: "linear_cent" (uniform $0.01), "deci_cent"
    (uniform $0.001), "tapered_deci_cent" (banded), and
    "center_deci_edge_centi_cent" ($0.0001 below $0.01 and above $0.99, $0.001
    in between). The authoritative grid is the market's own `price_ranges`
    bands; the structure name is only used to short-circuit the uniform-cent
    case. A price sitting exactly on a band boundary belongs to TWO bands, and
    the FINEST step among the bands containing it wins. First-match used to be
    the rule, and it is only "finer" at a band's LOWER edge — at an UPPER edge
    the earlier band is 10x COARSER, which multiplied BUY_SLIPPAGE_TICKS by a
    10x tick exactly there and LOOSENED a buy cap that is a bid (TS-10).

    These grids are NESTED: $0.01 ⊂ $0.001 ⊂ $0.0001, so every point of a
    coarser grid is also a point of any finer one. Later price math relies on
    that — a cap computed as price + n × tick that crosses into a neighbouring
    band still lands on a valid grid point of that band, and the $0.01 fallback
    below is valid on every regime — so this function does not need to
    re-quantize across band edges.

    Returns a Decimal (never a float) because the V2 order endpoint takes
    dollar-string prices: binary float noise in a tick size would propagate
    into a price string the exchange rejects as off-grid.

    Args:
        market (Any): A market object — ApiMarket or any object with
            `price_level_structure` (str) and `price_ranges`
            (list[PriceRange] | None) attributes. Missing attributes are
            treated as unknown, not as an error.
        price_dollars (float): The price at which the tick size is needed, in
            dollars. Range: [0, 1].

    Returns:
        Decimal: The FINEST tick size in dollars among the bands containing
            that price. Falls back to Decimal(config.DEFAULT_TICK_SIZE_DOLLARS)
            when the structure is uniform-cent or unknown, when no band
            contains the price, or when every containing band's step is
            nonpositive — logging a warning in the latter two cases, which
            indicate a payload that drifted from the shapes above.
    """
    default = Decimal(DEFAULT_TICK_SIZE_DOLLARS)
    structure = getattr(market, "price_level_structure", "") or ""
    bands = getattr(market, "price_ranges", None)
    # Uniform-cent markets (and any market whose bands failed to parse — see
    # _parse_price_ranges' fail-soft contract) use the coarse default grid.
    if structure in ("", "linear_cent") or not bands:
        return default

    # Finest containing band wins (see docstring). Scanning every band instead
    # of returning on the first match also means one malformed band no longer
    # discards a valid later one — the old `break` abandoned the whole list.
    finest: Decimal | None = None
    for band in bands:
        try:
            if band.start <= price_dollars <= band.end:
                # Decimal(str(...)), never Decimal(float): the float came from
                # parsing a dollar string and str() round-trips it back exactly.
                step = Decimal(str(band.step))
                if step > 0 and (finest is None or step < finest):
                    finest = step
        except (AttributeError, TypeError, InvalidOperation):
            # This band is unreadable; keep scanning the rest rather than
            # throwing away bands that may well be intact.
            continue
    if finest is not None:
        return finest

    # Only reached on a malformed or non-covering band list; called once per
    # order leg at build time, so a warning here cannot spam the log.
    logging.warning(
        "No usable tick band for %s at price %.4f (structure=%r) — falling back to $%s",
        getattr(market, "ticker", "<unknown>"), price_dollars, structure, DEFAULT_TICK_SIZE_DOLLARS,
    )
    return default



# Lowest valid V2 limit price, in dollars. Kalshi prices live in the open unit
# interval — 0 and 1 are settlement values, not tradeable levels — and the
# finest grid in any regime is $0.0001, so this is the extreme valid bottom
# tick. The top of grid is not a module constant: it depends on the market's
# own tick regime and is derived per market by _v2_top_of_grid_price() from
# config.V2_ROLLBACK_BID_PRICE_DOLLARS.
_V2_MIN_PRICE = Decimal("0.0001")

# Quantum applied to the scanned price BEFORE it is ceiled onto the tick grid —
# the same round-before-ceil guard as _buy_max_cost_cents and
# config.fee_leg_exact. No Kalshi grid point has a 7th decimal (the finest is
# $0.0001), so quantizing can only remove binary float noise: it tightens or
# keeps the cap, never loosens it (TS-03).
_SCANNED_PRICE_QUANTUM = Decimal("0.000001")

def ceil_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """
    Round a price UP to the next point of a tick grid.

    Ceiling, never nearest or floor: this is the first half of a buy leg's price
    cap, and it mirrors the legacy _buy_max_cost_cents' math.ceil for exactly
    the same reason — a cap rounded BELOW the scanned depth-weighted price could
    never fill at the price we actually scanned, so a fill-or-kill order carrying
    it would be structurally killed every time rather than protected.

    Args:
        price (Decimal): Price in dollars to round. Range: [0, 1].
        tick (Decimal): Tick size in dollars for the grid to land on. Must be
            > 0 (tick_size_for_price guarantees this).

    Returns:
        Decimal: The smallest grid point >= price. Returns price unchanged when
            it already sits exactly on the grid.
    """
    return (price / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def v2_limit_price(leg_kind: str, scanned_price_dollars: float, market: Any) -> Decimal:
    """
    Compute the fill-or-kill LIMIT price for one V2 buy leg, in dollars.

    V2 has no "market" order type, so a taker order is a marketable FoK limit
    and this price IS the price protection that buy_max_cost provided on the
    legacy path: the order fills at or better than the cap, or not at all.
    The cap is the scanned price ceiled onto the market's own tick grid plus
    BUY_SLIPPAGE_TICKS ticks of tolerance for a book that moved since the
    pre-execution check.

    The scanned price is quantized to 6 decimals before that ceiling, the same
    round-before-ceil guard the legacy cap applies in _buy_max_cost_cents (and
    config.fee_leg_exact before it). Every scanned ask level is the complement
    of a resting bid (1.0 - float(bid)), and 20 of the 99 whole-cent
    complements land one ULP ABOVE the exact cent, which would otherwise ceil a
    whole extra tick and hand the order 2 x BUY_SLIPPAGE_TICKS of tolerance.
    No Kalshi grid point has a 7th decimal, so the quantize can only remove
    float noise: it tightens or keeps the cap, never loosens it (TS-03).

    Because the cap (or, for the NO leg, its complement 1 - cap) can land in a
    DIFFERENT band of the market's grid than the scanned price — stepping up
    across a band edge, or being mirrored to the other end of the book — the
    final price is re-quantized onto the grid of the band that actually
    contains it, rounding UP. Ceiling is chosen because a floor could round a
    YES cap BELOW the scanned price and make the order structurally unfillable.
    Kalshi's nested grids ($0.01 subset of $0.001 subset of $0.0001) mean a
    price landing in a FINER band than it was computed on is already on that
    band's grid, so the snap is then a no-op.

    That re-quantization is NOT purely protective, and this docstring used to
    claim it was. When the final price lands in a COARSER band than the one it
    was computed on, ceiling moves it AWAY from the scanned price and LOOSENS
    the cap by up to one destination-band tick. Reachable examples on the live
    band layouts: scanned 0.10 on tapered_deci_cent submits 0.11 where 0.101
    was intended ($0.0090/contract of extra tolerance), and scanned 0.01 on
    center_deci_edge_centi_cent submits 0.011 where 0.0101 was intended
    ($0.0009). The loosening is bounded by one tick of the destination band and
    is a known, accepted cost of keeping the order fillable; it is a separate
    finding from TS-10 and is deliberately not fixed here. TS-10 fixed the
    other half — tick_size_for_price now resolves a boundary price to the
    FINEST containing band, so the slippage allowance is no longer multiplied
    by a 10x tick at a band's upper edge.

    The clamp bounds are grid-aware for the same reason: the extreme tradeable
    levels are one tick inside 0 and 1 ON THIS MARKET'S GRID (e.g. 0.99, not
    0.9999, on a linear-cent market), so the bounds are derived from the tick
    size at each end of the book rather than the global finest-grid constants.

    This mapping (which side, and the complement for the NO leg) is the single
    assumption most in need of verification at the first live submission; see
    _V2_LEG_SIDE, which is where a correction would be made.

    Args:
        leg_kind (str): Which KIND of leg is being priced — "buy_yes" or
            "buy_no". Any other value is treated as a YES-style leg (the price
            is used as-is). A plain string, not a _Leg: this helper prices one
            side and knows nothing about which of the pair's markets it is on.
        scanned_price_dollars (float): The scanned depth-weighted per-contract
            price for this leg, in dollars. Range: (0, 1). For "buy_no" this is
            the NO price (the NO leg's _Leg.price_dollars), which is
            complemented into a YES-book ask price.
        market (Any): The market object the leg trades, used only to look up its
            tick grid. Any object exposing price_level_structure / price_ranges.

    Returns:
        Decimal: The limit price in dollars — a valid grid point of the band
            containing it, clamped one tick inside the open unit interval on
            this market's grid.
    """
    # Every scanned level is 1.0 - float(bid) (scanner._bids_to_ask_levels),
    # and 20 of the 99 whole-cent complements land one ULP ABOVE the exact
    # cent (0.43 -> 0.5700000000000001). Un-quantized, _ceil_to_tick steps a
    # whole extra tick and the FoK limit carries 2 x BUY_SLIPPAGE_TICKS of
    # tolerance instead of 1 (TS-03).
    scanned = Decimal(str(scanned_price_dollars)).quantize(_SCANNED_PRICE_QUANTUM)
    # Cross-module: the market's own tick grid is the only authority on what
    # price levels the exchange will accept for this leg
    tick = tick_size_for_price(market, scanned_price_dollars)
    cap = ceil_to_tick(scanned, tick) + BUY_SLIPPAGE_TICKS * tick
    # Buying NO is selling YES on the single YES book, so the YES-side price is
    # the complement of the capped NO price
    price = Decimal("1") - cap if leg_kind == "buy_no" else cap
    # Re-quantize onto the grid of the band the FINAL price sits in (see
    # docstring: ceiling is protective — worst case is a killed FoK)
    final_tick = tick_size_for_price(market, float(price))
    price = ceil_to_tick(price, final_tick)
    # Grid-aware clamp: the extreme valid levels are one tick inside 0 and 1
    # on this market's own grid at each end of the book
    bottom_tick = tick_size_for_price(market, float(_V2_MIN_PRICE))
    top_tick = tick_size_for_price(market, float(Decimal("1") - _V2_MIN_PRICE))
    return min(max(price, bottom_tick), Decimal("1") - top_tick)


def v2_effective_cap(leg_kind: str, scanned_price_dollars: float, market: Any) -> Decimal:
    """
    The V2 FoK cap for one buy leg, expressed in that leg's OWN side terms.

    v2_limit_price returns the price that goes on the WIRE, which for a NO leg
    is a YES-book ask of 1 - cap. Callers that need to compare the cap against
    NO-side level prices — strategy's reachability gate, validate_pair_price's
    pre-execution re-check — need the NO price, so this undoes that complement
    and nothing else.

    It exists so those callers never re-derive the cap themselves. TS-08 is
    precisely "the size and the cap disagree", and a second copy of the formula
    guarantees they disagree again; going through v2_limit_price means the
    number tested here is the number the order body will carry, including the
    NO leg's complement round-trip, which can TIGHTEN the effective NO cap on a
    sub-cent grid.

    Args:
        leg_kind (str): "buy_yes" or "buy_no", as for v2_limit_price.
        scanned_price_dollars (float): The leg's per-contract price in dollars
            — for "buy_no" the NO price. Range: (0, 1).
        market (Any): The market the leg trades, for its tick grid.

    Returns:
        Decimal: The highest per-contract price this leg can pay, in the leg's
            own side's terms — a YES price for "buy_yes", a NO price for
            "buy_no". A level priced above it cannot fill under this order.
    """
    wire = v2_limit_price(leg_kind, scanned_price_dollars, market)
    # Buying NO is selling YES, so the NO-side cap is the complement of the
    # YES-book ask that actually gets submitted.
    return Decimal("1") - wire if leg_kind == "buy_no" else wire

@dataclass
class CandidatePair:
    """
    A candidate pair of correlated markets with a detectable price gap.

    Which side each leg buys, and therefore which two of the four quoted
    prices are LEG prices, depends on pair_type — read them through
    leg_sides() / leg_prices(), never by position:

      same_title:  NO on market_a (the pricier side by YES ask) at nA, YES on
                   market_b (the cheaper side) at pB. Both legs pay when the
                   two identical questions co-resolve, so the trade is a
                   near-arbitrage priced on the SAME_TITLE_CO_RESOLVE_PROB
                   prior — which is why the finder only forms such a pair
                   across two DIFFERENT event series (see _same_series).
      time_series: YES on market_a (the EARLIER contract) at pA, NO on
                   market_b (the later one) at nB. "Earlier" means earlier by
                   close_time for a cross-event pair, and earlier by STATED
                   DEADLINE for a same-event ladder (DR-73), whose rungs a
                   settled event closes at one instant — read the gap through
                   pair_gap_days(), never by re-deriving it from close_time.
                   Three settlement cells
                   exist: event by A's deadline (A=YES, hence B=YES; YES-on-A
                   pays), never by B's (A=NO, B=NO; NO-on-B pays), or in
                   between (A=NO, B=YES; both legs worthless — the one loss
                   cell). A=YES with B=NO is impossible for a
                   cumulative-deadline pair, and find_time_series_pairs now
                   screens both legs' WORDING for that premise (a heuristic,
                   scanner.deadline_phrasing) instead of assuming it. This is
                   a directional bet, not an arbitrage: it profits only if the
                   market overstates the in-between probability (see
                   config.time_series_profit_prob).

    Attributes:
        market_a (Any): same_title: the market with the higher YES ask (the
            expensive side). time_series: the earlier contract — by close_time
            for a cross-event pair, by STATED deadline for a same-event
            ladder (DR-73).
        market_b (Any): same_title: the market with the lower YES ask (the
            cheap side). time_series: the later contract, by the same
            ordering.
        pA (float): YES ask price of market A in dollars (cost to buy YES on A).
            Range: [0, 1]. A leg price for time_series; reporting-only for
            same_title.
        pB (float): YES ask price of market B in dollars (what a YES contract
            on market B costs). Range: [0, 1]. A leg price for same_title; for
            time_series it feeds the price-gap filter and the Kelly model but
            is not a leg price.
        nA (float): NO ask price of market A in dollars (what a NO contract on
            market A costs). Range: [0, 1]. A leg price for same_title; reporting-only for
            time_series (still read so the prod log's "nA (NO ask)" column
            stays meaningful).
        tradeable (bool): True when the two LEG prices sum to less than
            1 - fee_per_pair_approx(leg prices) — i.e. a win scenario pays more
            than the pair costs — and, for time_series, pB > pA. Neither pair
            type's flag is a settlement guarantee: same-title rests on the
            co-resolution prior, time-series on the in-between probability
            being overstated.
        canonical_title (str): Grouping key used to identify the pair — the
            normalized title+outcome key (scanner.time_series_group_key) for
            time-series pairs, raw title for same-title pairs.
        pair_type (str): Strategy variant: "time_series" for pairs differing only in deadline,
            "same_title" for pairs with identical title/subtitle across different event tickers.
        max_contracts (int): How many contracts the pair's written leg prices
            are valid for — the affordability-bounded qualifying depth set by
            enrich_with_orderbook_prices(). 0 means NOT ENRICHED (uncapped);
            enrichment never writes 0, dropping such a pair instead.
        nB (float): NO ask price of market B in dollars (cost to buy NO on B).
            Range: [0, 1]. A leg price for time_series; populated fail-soft
            (0.0 when unparseable) for same_title, where it is reporting-only
            and never priced.
        stated_gap_days (int | None): For a same-event ladder (DR-73), the
            calendar-day gap between the two legs' STATED deadlines — the gap
            that ordered the legs, chose the price tier and was checked
            against MAX_DEADLINE_GAP_DAYS at pair formation, carried out
            rather than recomputed because a settled or same-day-closing
            event gives its rungs a close_time gap of zero. None for every
            other pair, meaning "tier this pair on close_time". Read it
            through pair_gap_days(), never by testing truthiness: a genuine
            0-day stated gap is falsy.
        depth_levels (tuple): The pair's qualifying order-book depth as
            (price_a, price_b, qty) triples in MARKET order, ascending by
            combined price — oriented once by enrich_with_orderbook_prices via
            leg_sides, the same source of truth leg_prices follows, so readers
            need no pair-type logic. Empty () before enrichment. This is what
            lets strategy.compute_trade price the exact n it sizes
            (scanner.prefix_fill_prices) instead of reusing one scalar average
            computed over depth the trade could never reach.
    """
    market_a: Any           # same_title: pricier side by YES ask | time_series: EARLIER contract (close_time, or STATED deadline for a DR-73 ladder)
    market_b: Any           # same_title: cheaper side by YES ask  | time_series: the later one, by the same ordering
    pA: float               # yes_ask_dollars of A (cost to buy YES on A) — a LEG price for time_series
    pB: float               # yes_ask_dollars of B (cost of a YES contract on B) — a LEG price for same_title
    nA: float               # no_ask_dollars of A  (cost of a NO contract on A)  — a LEG price for same_title
    tradeable: bool         # True when a win scenario pays more than the leg prices + approx fees
    canonical_title: str    # grouping key (normalized for time-series, raw for same-title)
    pair_type: str          # "time_series" | "same_title"
    max_contracts: int = 0  # qualifying contracts from order book (0 = not yet enriched)
    nB: float = 0.0         # no_ask_dollars of B (cost to buy NO on B) — a LEG price for time_series; reporting-only for same_title
    # Qualifying (price_a, price_b, qty) depth in MARKET order, ascending by
    # combined price; () = not enriched. Read via prefix_fill_prices().
    depth_levels: tuple[tuple[float, float, float], ...] = ()
    # DR-73: a same-event ladder's STATED deadline gap, which ordered the legs
    # and chose the tier; None = tier this pair on close_time. Read via
    # pair_gap_days(), never by truthiness (a 0-day stated gap is falsy).
    stated_gap_days: int | None = None


def leg_sides(pair_type: str) -> tuple[str, str]:
    """
    Return which side each leg of a pair buys, as (side on market_a, side on market_b).

    Time-series pairs buy YES on the earlier contract (market_a) and NO on the
    later one (market_b); same-title pairs buy NO on the pricier contract
    (market_a) and YES on the cheaper (market_b). Anything other than the exact
    string "time_series" — including None or a test double's auto-attribute —
    resolves to the same-title sides, so a pair whose type is unknown can never
    be silently traded as the directional time-series bet.

    Args:
        pair_type (str): CandidatePair.pair_type — "time_series" or "same_title".

    Returns:
        tuple[str, str]: config.TIME_SERIES_LEG_SIDES for time-series pairs,
            config.SAME_TITLE_LEG_SIDES otherwise. Each element is "yes" or "no".
    """
    if pair_type == "time_series":
        return TIME_SERIES_LEG_SIDES
    return SAME_TITLE_LEG_SIDES


def leg_prices(pair: Any) -> tuple[float, float]:
    """
    Return the per-contract cost of the side actually bought on each leg.

    This is the only mapping from a pair's four quoted prices to the two prices
    the pipeline sizes, fees and submits against: (nA, pB) for a same-title pair
    (NO on market_a, YES on market_b) and (pA, nB) for a time-series pair (YES on
    market_a, NO on market_b). Reads nB directly rather than defaulting it — a
    real CandidatePair always carries it, and a mock that lacks it should fail
    loudly instead of pricing a leg at a placeholder.

    Args:
        pair (Any): A CandidatePair (or an object exposing pair_type, pA, pB,
            nA and nB). pair_type is read fail-safe: anything other than
            "time_series" is treated as same-title (see leg_sides).

    Returns:
        tuple[float, float]: (price of the leg on market_a, price of the leg on
            market_b), dollars in (0, 1).
    """
    if getattr(pair, "pair_type", None) == "time_series":
        return pair.pA, pair.nB
    return pair.nA, pair.pB


def deadline_gap_days(market_a: Any, market_b: Any) -> int:
    """
    Return the whole-day gap between two markets' deadlines, independent of order.

    Uses timedelta.days on the close_time datetimes (the live scanner's
    arithmetic, which the backtester mirrors) on the absolute difference, so
    the result is the same whichever market is passed first. Both markets must
    carry a real datetime close_time — callers run downstream of
    _filter_active_markets, which drops markets without one.

    Args:
        market_a (Any): One leg's market, exposing a datetime close_time.
        market_b (Any): The other leg's market, exposing a datetime close_time.

    Returns:
        int: abs(market_b.close_time - market_a.close_time).days, >= 0.
    """
    return abs(market_b.close_time - market_a.close_time).days


def pair_gap_days(pair: Any) -> int:
    """
    Return the deadline gap a pair's price tier and ceiling must be measured on.

    One reader for the two quantities that can differ: a cross-event
    time-series pair is tiered on its legs' close_time gap
    (deadline_gap_days), while a same-event LADDER is tiered on the STATED
    deadline gap the finder ordered its legs by and already checked against
    MAX_DEADLINE_GAP_DAYS (DR-73). The two are not interchangeable — a
    settled or single-instant event closes every rung of a ladder at the same
    moment, so close_time would report a 0-day gap and silently drop such a
    pair from the 30% tier to the 15% one — so every downstream re-derivation
    of the tier goes through this helper rather than calling
    deadline_gap_days() itself.

    The stated gap is read by TYPE, not truthiness: a genuine 0-day stated gap
    is falsy, and bool is deliberately not accepted (type(True) is bool, not
    int), the same fail-safe-by-type rule leg_sides() and
    strategy._depth_levels() follow — anything that is not exactly an int
    (a MagicMock's auto-attribute, None, a float) falls back to close_time.

    Args:
        pair (Any): A CandidatePair, or any object exposing market_a/market_b
            and optionally stated_gap_days.

    Returns:
        int: The pair's deadline gap in whole calendar days, >= 0.
    """
    stated = getattr(pair, "stated_gap_days", None)
    # Exact type, not isinstance: bool is an int subclass, and True would
    # otherwise be read as a one-day stated gap.
    if type(stated) is int:
        return stated
    return deadline_gap_days(pair.market_a, pair.market_b)


def normalize_title(title: str) -> str:
    """
    Strip all date and time tokens from a market title, returning a normalized string.

    Removes patterns such as month names, ISO dates, quarters, and relative time
    expressions, then collapses whitespace and lowercases the result. Two markets
    that differ only in their deadline will produce the same normalized string,
    enabling exact-match grouping without fuzzy matching.

    This is the TITLE half of the time-series grouping key only. Callers that
    group time-series candidates must go through time_series_group_key(),
    which appends the market's outcome discriminator — a title alone does not
    separate the strikes of one daily family (DR-01).

    Args:
        title (str): Raw market title from the Kalshi API.

    Returns:
        str: Lowercased, whitespace-collapsed title with all date tokens removed.
    """
    result = title
    for pat in _COMPILED:
        result = pat.sub(" ", result)
    return re.sub(r"\s+", " ", result).strip().lower()


def market_title(market: Any) -> str:
    """
    Return the best available display title for a market object.

    Prefers `.title`, falls back to `.subtitle`, then `.ticker` as a last resort.

    Since the 2026-08 API drift, `.subtitle` is sourced from `yes_sub_title`
    (see `_market_from_dict`), so a title-less market now falls back to its
    outcome label (e.g. a candidate name) rather than dropping through to the
    opaque ticker as it did while subtitle was always "".

    Args:
        market (Any): A Kalshi market API object with `.title`, `.subtitle`, and `.ticker` attributes.

    Returns:
        str: The first non-falsy value among title, subtitle, and ticker.
    """
    return market.title or market.subtitle or market.ticker


def pair_key(market: Any) -> str:
    """
    Combined grouping key — event title joined with market title.

    Used by the pair-finders to ensure two markets with the same option label in
    different unrelated events (e.g. "Trump" in "2024 Election" vs. "Trump" in
    "2024 Time Person of the Year") do NOT get grouped as the same question.
    Falls back to the bare market_title when `_event_title` is missing — this
    keeps backwards-compat for test fixtures and any caller that hasn't been
    migrated to attach event titles, but disables cross-event collision
    protection for that market.

    Args:
        market (Any): A Kalshi market object. May have an `_event_title` attribute
            attached by `fetch_open_events_with_markets`.

    Returns:
        str: "<event_title> | <market_title>" when an event title is attached,
            otherwise just the market title.
    """
    event_title = getattr(market, "_event_title", "") or ""
    if not event_title:
        return market_title(market)
    return f"{event_title} | {market_title(market)}"


def event_series(event_ticker: Any) -> str:
    """
    Return the series identity of a Kalshi event ticker — its prefix before the
    first "-", except that every combo (KXMVE*) prefix collapses onto one family.

    Kalshi event tickers are a series prefix followed by the instance stamp, so
    "KXNPBRFI-26SEP160500FUKORI" -> "KXNPBRFI". Two event tickers that share a
    series are two instances of ONE recurring fixture (two ball games, two
    15-minute price windows, two combo tickets), which is what makes identical
    wording across them two different questions rather than one question listed
    twice — see find_same_title_pairs and CLAUDE.md's one-series gotcha
    (DR-02, DR-54).

    Kalshi lists its multi-leg COMBO (parlay) markets under several prefixes,
    all beginning config.MVE_SERIES_FAMILY_PREFIX ("KXMVE"). The literal prefix
    did refuse two combos listed under ONE of them, but split the family into
    four, so a pair spanning TWO combo prefixes was never seen by the
    one-series rule (DR-55). Any prefix in that family therefore answers with
    the family name, collapsing them onto one series:

        "KXMVECROSSCATEGORY-SHARD1-S6471E4699E9"       -> "KXMVE"
        "KXMVESPORTSMULTIGAMEEXTENDED-SHARD1-S93FF..." -> "KXMVE"

    That is deliberately an over-collapse: a combo ticket's wording names its
    legs but never its date, so two identically-worded tickets are two
    different tickets whichever KXMVE* series they were listed under, and
    _same_series() only ever REFUSES pairs — see
    config.MVE_SERIES_FAMILY_PREFIX for the census and the measured
    cross-prefix co-resolution rate behind the family rule, and for why an
    allowlist was rejected.

    It has a second reader outside pairing: historical._is_combo_event, which
    decides which event-title lookups the backtest spends its capped budget on
    (DR-51), so "combo" means the same thing there as it does to the
    one-series rule.

    Args:
        event_ticker (Any): The market's event_ticker. Anything that is not a
            str is read as unknown.

    Returns:
        str: config.MVE_SERIES_FAMILY_PREFIX when the ticker's prefix starts
            with it; otherwise the prefix before the first "-", stripped of
            surrounding whitespace and upper-cased. "" for an empty,
            all-whitespace, hyphen-leading or non-string ticker — every shape
            _same_series then reads as an unknown series and fails closed on. A
            hyphen-less ticker is its own series (returned stripped and
            upper-cased, and collapsed to the family if it is a KXMVE* one).
    """
    if not isinstance(event_ticker, str):
        return ""
    prefix = event_ticker.split("-", 1)[0].strip().upper()
    # One family for every combo series: their literal prefixes differ, but two
    # identically-worded combo tickets are different tickets regardless (DR-55).
    if prefix.startswith(MVE_SERIES_FAMILY_PREFIX):
        return MVE_SERIES_FAMILY_PREFIX
    return prefix


def _identical_wording(mA: Any, mB: Any) -> bool:
    """
    True when two markets carry the same raw (title, subtitle, event title).

    Compares the RAW strings, not the date-stripped ones: the point of the test
    is that the deadline does not appear anywhere in the wording, so two
    contracts worded identically cannot be "the same question at two
    deadlines". Unreadable evidence reads as NOT identical rather than as a
    rejection: two values that are not equal are two different wordings (a
    MagicMock's auto-attributes are distinct objects), and every one of the
    three attributes is read through getattr with a "" default — the same
    fail-soft read find_time_series_pairs already uses for .subtitle when it
    builds the grouping key — so a partial stub leaves the pair for the
    ordinary filters instead of raising out of the finder.

    Args:
        mA (Any): First market object (.title, .subtitle, optional ._event_title).
        mB (Any): Second market object, same shape.

    Returns:
        bool: True only when all three strings match exactly.
    """
    return (
        (getattr(mA, "title", "") or "", getattr(mA, "subtitle", "") or "",
         getattr(mA, "_event_title", "") or "")
        == (getattr(mB, "title", "") or "", getattr(mB, "subtitle", "") or "",
            getattr(mB, "_event_title", "") or "")
    )


def _same_series(mA: Any, mB: Any) -> bool:
    """
    True when two markets' event tickers resolve to the same series, or either
    series is unknown.

    The series is event_series()'s verdict, not a literal prefix match: every
    combo (KXMVE*) prefix collapses onto one family, so two combo tickets share
    a series while sharing no literal prefix (DR-55).

    Fails CLOSED: a pair whose fixture identity cannot be read (an empty or
    non-string event_ticker on either side) reads as the same series and is
    therefore refused, because pricing it on the 95% co-resolution prior
    requires PROVING the two events are different fixtures.

    Args:
        mA (Any): First market object with an .event_ticker attribute.
        mB (Any): Second market object, same shape.

    Returns:
        bool: True when both markets resolve to the same series, or when either
            is unreadable.
    """
    sa, sb = event_series(mA.event_ticker), event_series(mB.event_ticker)
    return not sa or not sb or sa == sb


def _field_phrasing(text: Any) -> str | None:
    """
    Classify ONE wording field, or None when it names no deadline shape at all.

    Within a single field a SNAPSHOT marker beats a cumulative one, because the
    snapshot markers name what the contract RESOLVES on while a stray "by" may
    only be describing the route to it. Across fields the precedence is
    different and deliberately so — see deadline_phrasing.

    Args:
        text (Any): One of a market's wording fields (event title, market
            title, or outcome label). Anything that is not a non-empty str
            reads as absent rather than raising — the same fail-safe-by-type
            rule leg_sides and strategy._depth_levels follow, and the case is
            real: historical._market_to_dict stores subtitle as
            `m.get("subtitle") or m.get("yes_sub_title")`, which is None when
            both are absent.

    Returns:
        str | None: DEADLINE_SNAPSHOT or DEADLINE_CUMULATIVE when the field
            carries a marker; None when it carries none, so the caller can fall
            through to a less specific field.
    """
    if not isinstance(text, str) or not text:
        return None
    # Snapshot first: a field naming a measurement instant is a snapshot
    # question regardless of what else it says. Each table is tested with ONE
    # search of its precompiled alternation, which matches exactly when some
    # entry of that table does (DR-70; see _ANY_SNAPSHOT).
    if _ANY_SNAPSHOT.search(text):
        return DEADLINE_SNAPSHOT
    if _ANY_CUMULATIVE.search(text):
        return DEADLINE_CUMULATIVE
    return None


def deadline_phrasing(event_title: Any, title: Any, subtitle: Any) -> str:
    """
    Classify a market's wording as a cumulative deadline, a snapshot, or unknown.

    The single definition of "is this a 'by <date>' market?", shared by the live
    finder and the backtester's mirror (pinned by AST in tests/test_strategy.py)
    so the two paths can never disagree about which pairs are eligible.

    The three fields are scanned SEPARATELY and combined by MOST-SPECIFIC FIELD
    WINS: subtitle, then title, then event title; the first field to yield a
    verdict decides. They are never concatenated — joining them can manufacture
    a match across a field boundary (a title ending "...on" beside a subtitle
    beginning "June 30") and destroys the explanation of where a verdict came
    from.

    The subtitle leads because it is the SUB-CONTRACT — the thing that actually
    resolves — so its own wording is the most authoritative statement of what
    this contract settles on, and it must be able to override a parent event
    title. That is what keeps an MVE market whose sub-contract is a
    "by <date>" label eligible even when its parent event is titled in the
    snapshot style. It does not weaken the snapshot refusals this exists for:
    in the KXBTCD/KXSOLD daily families the subtitle is a bare strike
    ("$82,750 or above") carrying no marker, so the verdict falls straight
    through to the title, which is where "price ON Sep 15, 2026" lives.

    Returns DEADLINE_UNKNOWN when no field names a deadline shape — the
    deadline may still live in the event ticker or the close time, but nothing
    the pair-finders read can prove it, and cumulative_deadline_pair fails
    closed on that.

    Args:
        event_title (Any): The market's parent event title. Non-str reads as absent.
        title (Any): The market's own question text. Non-str reads as absent.
        subtitle (Any): The market's outcome label / sub-contract, ingested from
            `subtitle` with a fallback to `yes_sub_title`. Non-str reads as absent.

    Returns:
        str: DEADLINE_CUMULATIVE, DEADLINE_SNAPSHOT, or DEADLINE_UNKNOWN.
    """
    # The field walk lives in _deciding_field, the one definition shared with
    # deadline_profile, so the verdict and the spans can never be read off
    # different fields (DR-69).
    return _deciding_field(event_title, title, subtitle)[0]


def _deciding_field(event_title: Any, title: Any, subtitle: Any) -> tuple:
    """
    The verdict and the one field that produced it: the classifier's single definition.

    Most specific field first (subtitle, then title, then event title); the
    first field carrying a marker decides. deadline_phrasing returns the
    verdict. deadline_profile also takes the deadline spans from the SAME
    field, so a field that did not decide the verdict cannot lend a leg a span
    (DR-69). Before this change, a spanless subtitle marker could decide
    "cumulative" while the event titles' different "by <date>" phrases made the
    two legs look like two deadlines.

    A field identical to the one scanned just before it is skipped (DR-70):
    the walk only reaches a field when every earlier one carried no marker,
    so re-scanning the same string could only return that same None. Combo
    titles repeat their subtitle verbatim — every one of the 163,898 combo
    records among the first 200,000 of the 2026-09-08 day slice — which is
    why the skip alone measured 45.7 -> 23.9 us/record there (2026-09-22),
    and 12.3 together with _ANY_SNAPSHOT's one-search table test.

    Args:
        event_title (Any): Parent event title; a non-str reads as absent.
        title (Any): The market's question text; a non-str reads as absent.
        subtitle (Any): The outcome label / sub-contract; a non-str reads as absent.

    Returns:
        tuple[str, str | None]: (verdict, deciding field text), or
            (DEADLINE_UNKNOWN, None) when no field names a deadline shape.
    """
    # The duplicate-field skip compares only str against str, so a non-str
    # field still reads as absent and never raises: its __eq__ is never
    # called, neither directly nor reflected from a str comparison, because a
    # non-str field is never kept as `previous` either.
    previous = None
    for text in (subtitle, title, event_title):
        if isinstance(text, str) and text == previous:
            continue
        previous = text if isinstance(text, str) else None
        verdict = _field_phrasing(text)
        if verdict is not None:
            return verdict, text
    return DEADLINE_UNKNOWN, None


def _deadline_spans(text: Any) -> tuple:
    """
    The distinct deadline phrases ONE wording field spells out, normalized.

    Read from _COMPILED_CUMULATIVE[0], the only pattern whose match names a
    comparable point in time. Called on the deciding field only (see
    _deciding_field and DR-69). cumulative_deadline_pair compares these as
    normalized STRINGS: one deadline spelled two ways reads as two deadlines,
    and a truncated span can read two deadlines as one.

    Args:
        text (Any): The deciding field's text, or None. A non-str yields ().

    Returns:
        tuple[str, ...]: Sorted, de-duplicated, lower-cased,
            whitespace-collapsed deadline phrases; () when there are none.
    """
    if not isinstance(text, str) or not text:
        return ()
    return tuple(sorted({
        re.sub(r"\s+", " ", match.group(0)).strip().lower()
        for match in _COMPILED_CUMULATIVE[0].finditer(text)
    }))


def deadline_profile(event_title: Any, title: Any, subtitle: Any) -> tuple:
    """
    A market's phrasing verdict and its deadline spans, computed in one call.

    The memoizable unit: both finders classify each market ONCE into one of
    these and then compare the cheap results pairwise. That is a hard
    requirement rather than a tidy-up — the backtester's pair sweep runs on the
    order of a million comparisons for a single large group, where re-running
    the regex tables per pair makes about 60x more classifier calls than per
    market (2 legs x ~31 in-window neighbours per member on
    TestExtractPairsPerformanceSmoke's 50,000-member group).

    Both paths call this with their own three strings (the live scanner off
    ApiMarket attributes, the backtester off cached dict keys), so the FIELD
    EXTRACTION differs between them but the classification does not.

    The spans come from the SAME field that decided the verdict, never from
    the other two (DR-69): a less specific field that restates, or
    contradicts, the deadline can neither lend a spanless deciding field a
    date nor make two differing deciding fields look alike.

    Args:
        event_title (Any): The market's parent event title.
        title (Any): The market's own question text.
        subtitle (Any): The market's outcome label / sub-contract.

    Returns:
        tuple[str, tuple[str, ...]]: (verdict, deadline spans of the deciding
            field). The spans are () when the verdict is unknown, and also when
            the deciding field established the KIND of question without naming
            a comparable date ("within 30 days", "at any time"), even if a less
            specific field names one — see deadline_phrasing, _deciding_field
            and _deadline_spans.
    """
    verdict, text = _deciding_field(event_title, title, subtitle)
    return verdict, _deadline_spans(text)


def _market_deadline_profile(market: Any) -> tuple:
    """
    deadline_profile() for a live market object, read fail-soft by attribute.

    Args:
        market (Any): An ApiMarket (or any stand-in) exposing .title, .subtitle
            and optionally ._event_title. Every attribute is read through
            getattr with a "" default, the same partial-stub tolerance
            _identical_wording applies.

    Returns:
        tuple[str, tuple[str, ...]]: The market's (verdict, deadline spans).
    """
    return deadline_profile(
        getattr(market, "_event_title", "") or "",
        getattr(market, "title", "") or "",
        getattr(market, "subtitle", "") or "",
    )


# deadline_pair_refusal()'s three refusal reasons, named by the DECIDING
# FIELD'S shape (DR-69) rather than by the pair's whole wording — each
# constant's own docstring below says exactly what it means and, for
# REFUSED_SAME_DEADLINE, the one caveat on reading it. Bare module-level
# strings, the same internal-vocabulary idiom as DEADLINE_CUMULATIVE /
# DEADLINE_SNAPSHOT / DEADLINE_UNKNOWN above — callers import the constants
# and deadline_pair_refusal rather than compare against string literals
# (DR-72).
REFUSED_SNAPSHOT = "snapshot"
"""Either leg's deciding field verdict is DEADLINE_SNAPSHOT: a snapshot
wording ("price ON <date>", "top 10 in October", "after <date>") does not
nest, so the pair has no cumulative-deadline premise regardless of what the
other leg's field says."""

REFUSED_NO_STATED_DEADLINE = "no_stated_deadline"
"""Either leg is not DEADLINE_CUMULATIVE at all (DEADLINE_UNKNOWN — no field
named a deadline shape), OR its deciding field named the KIND of question
without a comparable date ("within 30 days", "at any time" — its
deadline_profile() spans are then empty). Both read as "nothing here to
compare a deadline against"."""

REFUSED_SAME_DEADLINE = "same_deadline"
"""Both legs are cumulative and both name a comparable deadline, but the two
DECIDING fields' span sets are equal: one deadline stated twice, not two
different ones — a same-title shape, not a two-deadline family. Caveat: the
spans compared are normalized strings (_deadline_spans), so a deadline phrase
that TRUNCATES the same way on both legs can read as equal even when the raw
wording differs; this reason cannot distinguish "truly one deadline" from
"two deadlines the parser happened to truncate alike"."""


def deadline_pair_refusal(profile_a: tuple, profile_b: tuple) -> str | None:
    """
    Why a candidate is not a two-cumulative-deadline pair, or None when it is.

    The single definition of time-series eligibility (DR-72): cumulative_
    deadline_pair() is built on this function
    (`deadline_pair_refusal(...) is None`), so the boolean verdict and the
    refusal REASON can never disagree — there is exactly one place a future
    change to the rule has to happen. Precedence is order-independent and
    total, so exactly one of four outcomes always applies:

      1. REFUSED_SNAPSHOT — either leg's DECIDING field verdict is
         DEADLINE_SNAPSHOT. Checked first: a snapshot verdict disproves the
         nesting premise outright regardless of the other leg's field or
         whether either names a comparable date.
      2. REFUSED_NO_STATED_DEADLINE — otherwise, either leg is not
         DEADLINE_CUMULATIVE, or its deciding field's deadline spans are
         empty (DR-69: the spans come from the deciding field alone, never a
         less specific field that happens to name a date).
      3. REFUSED_SAME_DEADLINE — otherwise, both legs are cumulative with
         comparable spans, but the two span sets are equal.
      4. None — both legs are cumulative, both name a comparable deadline,
         and the two span sets differ: a genuine two-cumulative-deadline pair.

    Args:
        profile_a (tuple): First leg's (verdict, spans) from deadline_profile().
        profile_b (tuple): Second leg's, same shape.

    Returns:
        str | None: One of REFUSED_SNAPSHOT, REFUSED_NO_STATED_DEADLINE,
            REFUSED_SAME_DEADLINE, or None when the pair is eligible.
    """
    verdict_a, spans_a = profile_a
    verdict_b, spans_b = profile_b
    if verdict_a == DEADLINE_SNAPSHOT or verdict_b == DEADLINE_SNAPSHOT:
        return REFUSED_SNAPSHOT
    if (
        verdict_a != DEADLINE_CUMULATIVE
        or verdict_b != DEADLINE_CUMULATIVE
        or not spans_a
        or not spans_b
    ):
        return REFUSED_NO_STATED_DEADLINE
    if spans_a == spans_b:
        return REFUSED_SAME_DEADLINE
    return None


def cumulative_deadline_pair(profile_a: tuple, profile_b: tuple) -> bool:
    """
    True when two markets are one question asked at two different CUMULATIVE deadlines.

    The eligibility rule for a time-series pair, applied by BOTH finders on
    precomputed deadline_profile() results. Defined through
    deadline_pair_refusal() (DR-72): True exactly when that function returns
    None, so the two can never disagree about which pairs are eligible. In
    plain terms this requires, in order:

      1. Both legs classify DEADLINE_CUMULATIVE. Fails CLOSED — a leg whose
         wording is a snapshot ("price on <date>", "top 10 in October",
         "after <date>") or names no deadline at all is refused, because the
         YES-on-earlier / NO-on-later trade has no premise without the nesting
         that a cumulative deadline provides. Same direction as _same_series:
         pricing on a premise requires proving it, not merely failing to
         disprove it.
      2. Both legs name at least one comparable deadline span, and the two sets
         DIFFER. The spans are those of each leg's DECIDING field only — the
         field that produced its verdict (DR-69). Identical spans mean one
         deadline stated twice — two listings of the same question, which is a
         same-title shape, not a two-deadline family. Empty spans mean the
         deciding field established the KIND of question ("within 30 days",
         "at any time") without naming a date, even if a less specific field
         names one (DR-69), so the two deadlines cannot be shown to differ and
         the pair is refused. Spans are compared as normalized STRINGS
         (_deadline_spans), not as parsed calendar dates: one deadline spelled
         two ways reads as two deadlines ("Oct 1, 2026" vs "October 1, 2026"),
         and two deadlines truncated to one span read as one — fixed for a
         weekday plus month-name date by DR-68 ("by Friday, Sep 19, 2026" no
         longer truncates to "by friday"), other truncations remain (CLAUDE.md
         DR-67 Known residuals).

    Args:
        profile_a (tuple): First leg's (verdict, spans) from deadline_profile().
        profile_b (tuple): Second leg's, same shape.

    Returns:
        bool: True only when both legs are cumulative and the deadlines their
            deciding fields state differ. Order-independent.
    """
    return deadline_pair_refusal(profile_a, profile_b) is None


# ---------------------------------------------------------------------------
# DR-73a: reading a rung's deadline as a CALENDAR DATE.
#
# These tables are NOT the phrasing tables above and must never be merged into
# them, for the same reason _DATE_PATTERNS and the phrasing tables stay apart:
# _CUMULATIVE_DEADLINE_PATTERNS answers "is this a 'by <date>' market?" and
# hands _deadline_spans a normalized PHRASE, which cumulative_deadline_pair
# then compares as an opaque string; the tables here answer the different
# question "which calendar day is the last one that phrase includes?", which
# is what orders two rungs of ONE event's cumulative deadline ladder and
# measures the gap between them (DR-73). Everything below consumes a span
# string that _COMPILED_CUMULATIVE[0] already produced, so this reader can
# never widen what counts as a cumulative deadline — it can only refuse a
# phrase the phrasing table already accepted.
#
# The prepositions split three ways, and the split is the whole point: the
# same named day is the LAST INCLUDED day under "by" and the day AFTER it
# under "before", so a ladder mixing the two is mis-ordered by a day if this
# distinction is dropped. Measured on the 2026-09-22 live snapshot: 1 of the
# 492 events holding >= 2 dated cumulative rungs mixes them (SCOTREF-27; see
# same_event_ladder).

# The named day is the last one INCLUDED ("by Sep 23, 2026" includes Sep 23).
_STATED_INCLUSIVE = frozenset({"by", "no later than", "on or before"})
# The named day is EXCLUDED ("before Sep 23, 2026" ends on Sep 22).
_STATED_EXCLUSIVE = frozenset({"before", "prior to"})
# Inclusive like "by", but a BARE YEAR behind it is refused: CLAUDE.md's DR-67
# Known residuals record "2018-19 through 2025-26" reading as the span
# "through 2025", a season range rather than a deadline. Turning that span
# into 2025-12-31 would promote a recorded false-cumulative residual into a
# usable calendar date. Exactly two live markets carry the shape and neither
# is a deadline: KXNBANEWCHAMPION-27-Y spans "through 2025" off a title
# reading "(2018-19 through 2025-26)", which IS that residual; the other,
# KXNCAAWBCONFSTREAK-30BIGEASTCONN-Y, spans "through 2030" off a SUBTITLE
# reading "UConn to win the Big East Tournament through 2030" — a conjunction
# over every season up to that year, anti-monotone in the date and so not a
# deadline either. (Its TITLE's "through the 2029-30 season" spans "through
# the 2029" and is refused by the article guard instead, which is why the
# subtitle is the field to look at.) A DATED "through" stays accepted and is a real
# fiscal-year shape: 14 live rungs span "through june 30, 2027"
# (KXNYCSTAT-HOME27, KXNYCAFFORDABLE-27JUN30), 6 "through april 30, 2027".
_STATED_THROUGH = frozenset({"through"})
# Refused outright: whether the named day is included is simply not stated,
# and the two readings differ by exactly the day that orders a ladder. Neither
# preposition spans any market on the 2026-09-22 live snapshot.
_STATED_AMBIGUOUS = frozenset({"until", "up to"})

# Splits a span into (preposition, article, "end of", date token). The three
# optional pieces are captured rather than skipped because two of them decide
# a refusal: an article with no "end of" means the span TRUNCATED a longer
# phrase ("Before the 2027-28 season" -> "before the 2027", 115 live markets),
# and "end of" is only meaningful behind an inclusive preposition.
_STATED_SPAN = re.compile(
    r"^(before|prior to|by|no later than|on or before|through|until|up to)\s+"
    r"(the\s+)?(end\s+of\s+(?:the\s+)?)?(.+)$"
)

# A weekday in front of the date is noise for this reader — _DEADLINE_DATE_TOKEN
# keeps it so that "by Friday, Sep 19, 2026" spans the whole date instead of
# truncating to "by friday" (DR-68), and the date behind it is what names the
# day.
_STATED_WEEKDAY_PREFIX = re.compile(rf"^{_DEADLINE_WEEKDAY}\b\.?,?\s*", re.IGNORECASE)

# The accepted tokens. Every one states an explicit YEAR, which is the rule
# that makes this reader safe without an anchor date: a year-less deadline
# ("before Nov 4") cannot be placed, and guessing mis-orders a ladder that
# crosses New Year — KXAPCALLSENATE-26AUG20 lists "Before Nov 4" (2026)
# beside a sibling's "Before Jan 5" (2027). Year-less spans therefore read as
# None and the pair fails closed: 69 of the 4,161 live ladder candidate rungs
# on the 2026-09-22 snapshot (42 naming a month and day, 27 a bare month), of
# which 3,951 read to a date in all.
_STATED_MONTH_DAY_YEAR = re.compile(
    rf"^({_DEADLINE_MONTH})\b\.?\s+(\d{{1,2}})(?!\d)\s*,?\s*(\d{{4}})$", re.IGNORECASE
)
_STATED_MONTH_YEAR = re.compile(
    rf"^({_DEADLINE_MONTH})\b\.?\s*,?\s*(\d{{4}})$", re.IGNORECASE
)
# 2020-2099, the same bound _DEADLINE_DATE_TOKEN's bare-year alternative uses.
_STATED_BARE_YEAR = re.compile(r"^(20[2-9]\d)$")
_STATED_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Month spelling -> month number, covering exactly the spellings
# _DEADLINE_MONTH matches (full names, three-letter abbreviations, and the
# four-letter "Sept"). Spelled out rather than read off calendar.month_name /
# calendar.month_abbr: those are _localized_month objects backed by
# strftime("%B"/"%b") and therefore LC_TIME-dependent, and this table is built
# once at import — so one locale.setlocale() anywhere in the process (debugpy's
# launcher calls it, so running under a debugger on a localized host is enough)
# would replace every English key and silently blind _span_deadline to every
# month-named deadline while bare years kept reading. It fails CLOSED (an
# unknown spelling returns None), but invisibly, which is the DR-66 shape the
# rest of DR-73 is careful about. TestStatedDeadline pins this table against
# _DEADLINE_MONTH so the two still cannot drift apart by a typo.
_STATED_MONTH_NUMBERS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _last_included_day(start: date, end: date, inclusive: bool) -> date:
    """
    Collapse a stated deadline's period into the last calendar day it includes.

    The one place the inclusive/exclusive distinction is applied, so every
    token granularity (a named day, a named month, a named year) expresses it
    the same way: the caller says which period the wording NAMES, and this
    returns the last day the deadline covers.

    Args:
        start (date): First day of the named period — the named day itself for
            a day-granularity token, the 1st for a month, January 1st for a year.
        end (date): Last day of the named period — the same day, the month's
            last day, or December 31st.
        inclusive (bool): True for "by"/"no later than"/"on or before"/
            "through" (the named period is covered), False for
            "before"/"prior to" (the deadline stops the day before it starts).

    Returns:
        date: The last calendar day the deadline includes.

    Raises:
        OverflowError: When an exclusive deadline names date.min, so the day
            before it is not representable. Callers catch it and refuse.
    """
    return end if inclusive else start - timedelta(days=1)


def _span_deadline(span: Any) -> date | None:
    """
    Turn ONE deadline span into the last calendar day it includes, or None.

    Consumes a span as _deadline_spans produces it (a match of
    _COMPILED_CUMULATIVE[0], lower-cased and whitespace-collapsed). The
    case/whitespace normalization is repeated here so a hand-written span
    still reads, NOT so raw wording can be passed: _STATED_SPAN is anchored at
    both ends, so the string must BEGIN with the preposition and END with the
    date token. "Will X happen by Dec 31, 2026?" reads as None, and so does
    "by Dec 31, 2026?" on its one character of trailing punctuation. Callers
    hand it spans from _deadline_spans(), never a raw field.

    Refuses — returning None — in seven measured cases, each of which fails
    CLOSED into "this rung has no readable deadline":

      1. A YEAR-LESS token ("before Nov 4"). Nothing in the wording anchors
         the year and guessing mis-orders a ladder crossing New Year; see
         _STATED_MONTH_DAY_YEAR's comment.
      2. An article with no "end of" ("before the 2027"). That span is what a
         season range truncates to ("Before the 2027-28 season"), not a
         deadline. The guard is `article and not end_of` precisely so
         "by the end of 2027" — an unambiguous 2027-12-31 — still reads.
      3. An article IMMEDIATELY IN FRONT OF A BARE YEAR, which _STATED_SPAN
         captures inside the "end of" group rather than in `article`
         ("by the end of the 2030", one live market, KXCANADACUP-30's "by the
         end of the 2030 season"). That is the same season-range truncation
         as (2), reached past its guard because "end of" disarms it; the
         discriminator is the "the" adjacent to the year, so
         "by the end of 2027" still reads and "by end of the 2030" does not.
      4. "until" / "up to", whose inclusivity is not stated.
      5. A bare year behind "through" (see _STATED_THROUGH).
      6. A bare ISO date behind an EXCLUSIVE preposition (see below).
      7. An impossible calendar date ("Feb 30"), "end of" at day granularity,
         or a NUMERIC calendar date ("by 12/31/2026" —
         _DEADLINE_DATE_TOKEN's m/d alternative spans one, so the phrase
         reaches this reader, but no accepted token is numeric-m/d and the
         shape occurs zero times in either candidate population, so refusing
         keeps the reader to the month-name/ISO forms that were measured).

    Why an ISO date is accepted only behind an INCLUSIVE preposition:
    _COMPILED_CUMULATIVE[0] cuts a bare ISO date out of a full timestamp, so
    "before 2026-04-01T14:00:00.000Z" spans "before 2026-04-01" — whose last
    INCLUDED day is the named one (the market is open through 14:00 on it),
    not the day before. The shape is real and it is exactly what DR-73
    targets: KXDIAZOUT-MDC lists FIVE such rungs under one event ticker in
    backtest_cache/archive_days (titles "before 2026-02-01" … "before
    2026-06-01", each closing at 14:00–15:00 UTC on the named day), where a
    blank cached subtitle makes the truncated ISO the DECIDING span, and six
    actively priced markets across KXDIAZOUT-MDC and KXLAIOUT-LCHI carry the
    same wording live (there in a NON-deciding field, which stated_deadline's
    cross-check still reads). Dating those one day early is the one FAIL-OPEN
    direction in this reader — an ISO rung beside a sibling worded "by Apr 1,
    2026" would read as a 1-day ladder instead of the SAME_DAY refusal it is —
    so the exclusive side is refused rather than guessed. An inclusive ISO is
    right under both readings (a timestamp's day and a bare date's day are the
    same day) and stays accepted. No genuine, non-truncated exclusive ISO span
    occurs in either candidate population, so nothing measured is lost.

    Args:
        span (Any): One deadline phrase. A non-str reads as absent rather than
            raising — the same fail-safe-by-type rule leg_sides and
            strategy._depth_levels follow.

    Returns:
        date | None: The last calendar day the deadline includes, or None when
            the span states no placeable day.
    """
    if not isinstance(span, str) or not span:
        return None
    text = re.sub(r"\s+", " ", span).strip().lower()
    match = _STATED_SPAN.match(text)
    if match is None:
        return None
    preposition, article, end_of, token = match.groups()
    if preposition in _STATED_INCLUSIVE or preposition in _STATED_THROUGH:
        inclusive = True
    elif preposition in _STATED_EXCLUSIVE:
        inclusive = False
    else:
        # _STATED_AMBIGUOUS, or a preposition added to _STATED_SPAN without a
        # table entry: fail closed rather than pick a reading.
        return None
    if article and not end_of:
        return None
    if end_of and not inclusive:
        # "before the end of 2027" names the day before an unstated last day.
        return None
    token = _STATED_WEEKDAY_PREFIX.sub("", token, count=1).strip()
    if not token:
        return None

    # Every raise inside this block is a date-construction failure — an
    # impossible calendar day, or a year at the very edge of date's range
    # whose exclusive predecessor is not representable. Both mean the span
    # states no placeable day.
    try:
        month_day_year = _STATED_MONTH_DAY_YEAR.match(token)
        if month_day_year is not None:
            if end_of:
                return None  # "end of" is month/year granularity only.
            month = _STATED_MONTH_NUMBERS.get(month_day_year.group(1))
            if month is None:
                return None
            named = date(
                int(month_day_year.group(3)), month, int(month_day_year.group(2))
            )
            return _last_included_day(named, named, inclusive)

        month_year = _STATED_MONTH_YEAR.match(token)
        if month_year is not None:
            month = _STATED_MONTH_NUMBERS.get(month_year.group(1))
            if month is None:
                return None
            year = int(month_year.group(2))
            first = date(year, month, 1)
            return _last_included_day(
                first, date(year, month, calendar.monthrange(year, month)[1]), inclusive
            )

        bare_year = _STATED_BARE_YEAR.match(token)
        if bare_year is not None:
            if preposition in _STATED_THROUGH:
                return None
            if end_of is not None and end_of.strip().endswith("the"):
                # "by the end of THE 2030" is what "... the 2030-31 season"
                # truncates to (KXCANADACUP-30) — the same season range the
                # article guard above refuses, reached past it because
                # "end of" disarms that guard. The signature is the article
                # ADJACENT to the bare year, which _STATED_SPAN captures
                # inside this group, so the leading-article test cannot see
                # it. "by the end of 2027" keeps its own article and still
                # reads.
                return None
            year = int(bare_year.group(1))
            return _last_included_day(
                date(year, 1, 1), date(year, 12, 31), inclusive
            )

        iso = _STATED_ISO.match(token)
        if iso is not None:
            if end_of or not inclusive:
                # A bare ISO date is indistinguishable from one
                # _COMPILED_CUMULATIVE[0] cut out of a full timestamp, whose
                # last INCLUDED day is the named one rather than the day
                # before it — see the docstring's ISO paragraph and
                # KXDIAZOUT-MDC's five archive rungs. Refuse rather than date
                # an exclusive one a day early, which is this reader's one
                # fail-OPEN direction.
                return None
            named = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return _last_included_day(named, named, inclusive)
    except (ValueError, OverflowError):
        return None
    return None


# Rewrites the exclusive prepositions into "by" so _span_deadline can be asked
# the OTHER question a span answers — which day does it NAME — without a
# second parser. Anchored, so it only ever touches a span's own leading word.
_STATED_EXCLUSIVE_PREFIX = re.compile(r"^(?:before|prior to)\b")


def _span_days(span: Any) -> frozenset:
    """
    Both calendar days one span can be pointing at: the last it INCLUDES and the one it NAMES.

    Kalshi routinely spells ONE deadline both ways inside a single market —
    KXSPACEXSTARSHIP-14-26SEP23 is titled "... before Sep 23, 2026?" and
    sub-titled "By Sep 23, 2026", and closes 2026-09-23T03:59Z, i.e. 23:59 on
    Sep 22 in New York. Read as last-included days alone those are Sep 22 and
    Sep 23, a one-day disagreement; read as alternatives they are one deadline.
    stated_deadline's cross-check therefore compares field SETS and refuses
    only when they are disjoint. The deciding field's own date is still the
    single last-included day _span_deadline returns — this set widens what
    counts as AGREEMENT, never what the helper reports.

    Args:
        span (Any): One deadline phrase, as _deadline_spans produces it. A
            non-str, or one this reader cannot place, yields an empty set.

    Returns:
        frozenset[date]: Zero, one or two days. Empty means the span states no
            placeable day, which the caller treats as silence rather than
            disagreement.
    """
    last = _span_deadline(span)
    if last is None:
        return frozenset()
    text = re.sub(r"\s+", " ", span).strip().lower()
    named = _span_deadline(_STATED_EXCLUSIVE_PREFIX.sub("by", text, count=1))
    return frozenset(day for day in (last, named) if day is not None)


def stated_deadline(
    profile: tuple, event_title: Any, title: Any, subtitle: Any
) -> date | None:
    """
    The last calendar day a CUMULATIVE market's deadline includes, cross-checked.

    DR-69 made deadline_profile read its spans from the DECIDING field alone,
    because a less specific field must not lend a spanless leg a date. That
    precedence was designed to pick a VERDICT. DR-73 reuses it for a DATE that
    sets leg order, the price tier and the 30-day cap, where being wrong is a
    mis-ordered or mis-tiered real-money trade rather than a missed pair — so
    this function additionally reads every OTHER wording field and refuses
    when one of them cannot be pointing at the deciding field's deadline at
    all. That is not a reversal of DR-69: a non-deciding field can only ever
    REFUSE here, never supply the date.

    Agreement is judged on _span_days SETS, not on single last-included days,
    because Kalshi's own ladder template spells one deadline both ways in one
    market ("before Sep 23, 2026" in the title, "By Sep 23, 2026" in the
    subtitle). Compared as last-included days those look a day apart, and
    refusing on it deleted every rung of KXSPACEXSTARSHIP-14 — the widest live
    ladder — so the check asks whether the two fields have ANY reading in
    common and refuses only when they are disjoint.

    It is not hypothetical in either direction. Of the 4,697 actively priced
    cumulative markets on the 2026-09-22 snapshot, 47 read two different
    last-included days across their own fields: 30 name genuinely
    irreconcilable days and are refused — KXSTARSHIPFL-26JUN-26OCT01 puts a
    stale subtitle "Before 2026" -> 2025-12-31 against a title "before Oct 1,
    2026", nine months apart — while 17 are one named day spelled two ways
    (the five KXSPACEXSTARSHIP-14 rungs, EUEXIT-30, EUEXPANSION-30,
    KXSATOSHIBTCYEAR-27, six KXGOVTCUTS-26 strikes and three more) and read
    normally.

    Fails CLOSED, like cumulative_deadline_pair: a non-cumulative verdict, a
    deciding field naming no span, a span this reader cannot place, two
    different days inside the deciding field, or any other field whose
    readings are disjoint from the deciding field's all return None.

    Args:
        profile (tuple): The market's (verdict, spans) from deadline_profile()
            — passed in rather than recomputed, because both finders classify
            each market exactly ONCE and compare the cheap results pairwise
            (DR-70).
        event_title (Any): The market's parent event title. Non-str reads as absent.
        title (Any): The market's own question text. Non-str reads as absent.
        subtitle (Any): The market's outcome label / sub-contract. Non-str reads
            as absent.

    Returns:
        date | None: The last calendar day this market's deadline includes, or
            None when no single day can be read from its wording.
    """
    verdict, spans = profile
    if verdict != DEADLINE_CUMULATIVE or not spans:
        return None
    # The deciding field must name exactly one placeable day. Two spans in the
    # one field that decides is ambiguity in the authoritative place, so it is
    # refused rather than resolved by precedence. The `len != 1` half is the
    # ONLY thing that catches it now: the day-SET cross-check below asks
    # whether each field has a reading in common with the deciding field's
    # own set, and a field is trivially in common with itself, so it passes a
    # two-day deciding field straight through. (Under the pre-fix
    # single-day cross-check this clause was a recorded EQUIVALENT mutant; it
    # is load-bearing again, and separately load-bearing for DETERMINISM,
    # since pop() off a two-element set returns an arbitrary one of them.)
    deciding_days = {_span_deadline(span) for span in spans}
    if None in deciding_days or len(deciding_days) != 1:
        return None
    deadline = deciding_days.pop()
    # The cross-check, on _span_days SETS. A span this reader cannot place
    # yields an EMPTY set and is IGNORED (it names no day to disagree with);
    # only a field with readings entirely disjoint from the deciding field's
    # refuses. Comparing single last-included days instead read Kalshi's own
    # "before D" / "By D" template as a one-day conflict — see _span_days.
    mine = frozenset().union(*(_span_days(span) for span in spans))
    for text in (subtitle, title, event_title):
        for span in _deadline_spans(text):
            other = _span_days(span)
            if other and not (other & mine):
                return None
    return deadline


# same_event_ladder()'s third outcome: both rungs name the same calendar day.
# Distinct from None (a rung whose deadline could not be read) because the two
# have different remedies and the callers count them on separate lines — one
# is a parser gap, the other is two listings of one deadline. A bare
# module-level string, the same internal-vocabulary idiom as DEADLINE_* and
# REFUSED_*; callers compare against the constant, never a literal.
SAME_DAY = "same_day"


def same_event_ladder(deadline_a: Any, deadline_b: Any) -> tuple | str | None:
    """
    Order two rungs of one event's deadline ladder and measure the gap between them.

    Pure arithmetic over two already-read deadlines (see stated_deadline), so
    the live finder and the backtester can share one definition of which rung
    is the earlier one and how far apart the two are. DR-73 tiers and caps a
    same-event ladder on this STATED gap rather than on close_time, because a
    settled event closes every rung at one instant: in the archive 681 of
    1,821 dated same-event pairs (37.4%) have a close gap of zero days and 13
    are ordered the wrong way round by realized close.

    The inclusive/exclusive distinction matters here and is not theoretical:
    1 of the 492 live events holding >= 2 dated cumulative rungs mixes the two
    prepositions (2026-09-22 snapshot) — SCOTREF-27 lists a rung labelled
    "Before 2027" beside ones labelled "By Jan 1, 2028" / "2029" / "2030" — so
    a reader that ignored the split would read those two rungs as 1 day apart
    (2027-12-31 against 2028-01-01) instead of 366 (2026-12-31 against
    2028-01-01), turning a pair MAX_DEADLINE_GAP_DAYS correctly refuses into
    one admitted at the short-gap tier. (Counted over rungs dated from their
    DECIDING field alone: three of SCOTREF-27's four rungs additionally name
    a bare year in their titles — "called before 2028" against a subtitle "By
    Jan 1, 2028" — whose readings are disjoint from the subtitle's, so
    stated_deadline's cross-check refuses them and the mixing count over the
    cross-checked population is 0. That is a fact about the cross-check, not
    about the prepositions.) The SAME_DAY outcome is what stops
    the mirror image of that: two rungs whose wording differs ("by mar 31,
    2027" beside "before apr 1, 2027") but which name the identical last
    included day are one deadline spelled two ways, not a two-rung ladder.

    Args:
        deadline_a (Any): First rung's last included day, from stated_deadline().
            Anything that is not exactly a date reads as unreadable rather than
            raising — the fail-safe-by-type rule leg_sides follows. datetime is
            a date subclass and is deliberately excluded: mixing the two raises
            TypeError on subtraction.
        deadline_b (Any): Second rung's, same shape.

    Returns:
        tuple[bool, int] | str | None: (swap, gap_days) when the two name
            different days, where swap is True exactly when the SECOND
            argument states the EARLIER deadline and gap_days is the
            order-independent calendar-day gap; the SAME_DAY constant when
            both name the same day; None when either deadline is unreadable.
            Callers must branch on `is None` and `is SAME_DAY`: BOTH non-None
            outcomes are truthy, so `if result: swap, gap = result` unpacks
            the eight characters of the SAME_DAY string and raises ValueError.
    """
    # Exact-type tests, not isinstance: datetime passes isinstance(x, date) and
    # would raise TypeError when subtracted from a date.
    if type(deadline_a) is not date or type(deadline_b) is not date:
        return None
    if deadline_a == deadline_b:
        return SAME_DAY
    return deadline_b < deadline_a, abs((deadline_b - deadline_a).days)


def _normalize_subtitle(subtitle: str) -> str:
    """
    Normalize an outcome label for use inside the time-series grouping key.

    Lower-cases, collapses whitespace, strips explicit dates
    (_COMPILED_EXPLICIT_DATES) and trailing punctuation, so that
    "Donald Trump" and "Donald Trump." are one option label and
    "$80,000 by June 30" and "$80,000 by July 31" are one strike at two
    deadlines. Trailing whitespace and punctuation are trimmed together, so
    a label that ends in a stripped date FOLLOWED by punctuation
    ("$80,000 by June 30.") still matches the same label written without
    either.

    Deliberately NOT normalize_title(): that pass strips any bare 20xx token as
    a year, and a subtitle is exactly where a bare 4-digit strike
    ("2050 or above") lives — running it over an outcome label would merge two
    strikes in 2000-2099 into one key, i.e. reintroduce DR-01 through the
    subtitle.

    Args:
        subtitle (str): Raw outcome label (ApiMarket.subtitle, sourced from the
            API's `yes_sub_title` — see _market_from_dict).

    Returns:
        str: Lower-cased, date-stripped, punctuation-trimmed label. "" when
            nothing survives.
    """
    result = subtitle
    for pat in _COMPILED_EXPLICIT_DATES:
        result = pat.sub(" ", result)
    # Trim the whitespace and the punctuation in ONE rstrip, not whitespace
    # first and punctuation second: a stripped date leaves a space in FRONT of
    # any trailing punctuation ("$80,000 by June 30." -> "$80,000 by ."), and
    # trimming the punctuation last would leave that space in the key, so the
    # label would not match the same one written without the period.
    return re.sub(r"\s+", " ", result).strip().rstrip(" .,;:!").lower()


def time_series_group_key(combined_title: str, subtitle: Any) -> str:
    """
    Grouping key for time-series pair detection: the date-stripped
    event_title | title PLUS the market's outcome discriminator.

    Two contracts are "the same question at two deadlines" only when they ask
    about the same OUTCOME. The outcome label — a strike such as "$82,750 or
    above", a temperature band, an MVE option name — lives in the market's
    subtitle (ingested from `subtitle`, falling back to `yes_sub_title`), and
    market_title() reaches `.subtitle` only when `.title` is empty, so a key
    built from the title alone merged every strike of a daily family into one
    group and the best-pair rule then selected the widest strike mismatch
    (DR-01 — CLAUDE.md's cross-strike gotcha carries the measurement). A strike
    spelled inside the TITLE is a separate case, not handled here.

    Single source of truth for BOTH paths — scanner.find_time_series_pairs and
    backtester._group_by_normalized_title must call this, the latter through
    backtester._ts_group_key, which backtester._index_eligible_keys also reads
    (pinned by AST in tests/test_strategy.py; a two-link chain on the
    backtester side). Fail-safe on a non-string subtitle (a MagicMock's
    auto-attribute, None from an old cache record): treated as absent, never
    raised on — the same rule leg_sides and strategy._depth_levels follow.

    Args:
        combined_title (str): The pair_key() of the market — "<event_title> |
            <market_title>", or the bare market title when no event title is
            attached.
        subtitle (Any): The market's outcome label. Anything that is not a str
            is read as absent.

    Returns:
        str: "<normalized title> | <normalized subtitle>", the normalized title
            alone when the subtitle normalizes to nothing, or "" when the title
            itself normalizes to nothing (the caller drops such markets).
    """
    base = normalize_title(combined_title)
    if not base:
        return ""
    sub = _normalize_subtitle(subtitle) if isinstance(subtitle, str) else ""
    return f"{base} | {sub}" if sub else base


def display_title(market: Any) -> str:
    """
    Human-readable label for console and Excel output.

    Returns "<event_title>: <market_title>" when an event title is attached
    (so multivariate option labels like "Trump" or "Above $80k" carry their
    event context for manual spot-checking). Falls back to the bare market
    title for non-MVE markets. The outcome label is appended as
    " — <subtitle>" whenever it is not already the whole label, so two strikes
    of one daily family are distinguishable in the Excel rows instead of
    rendering as the same title twice (DR-17).

    Args:
        market (Any): A Kalshi market object. May have an `_event_title` attribute.

    Returns:
        str: Display label suitable for user-facing tables and Excel rows.
    """
    event_title = getattr(market, "_event_title", "") or ""
    base = market_title(market)
    # Append the outcome label unless it IS the label already — market_title
    # falls back to the subtitle when the title is empty, in which case the
    # two are equal and nothing is appended. The test is exact equality, so a
    # title that merely ENDS with the subtitle ("Biggest Mover: <name>" with
    # subtitle "<name>") renders the name twice; cosmetic only, because
    # display_title feeds Excel cells and the pairs table and is never a
    # grouping key — DR-17
    sub = getattr(market, "subtitle", "") or ""
    if isinstance(sub, str) and sub and sub != base:
        base = f"{base} — {sub}"
    return f"{event_title}: {base}" if event_title else base


def _filter_active_markets(
    markets: list,
    excluded_tickers: set | None = None,
    *,
    warn_missing_close: bool = True,
) -> list:
    """
    Filter markets to those that are actively priced, deadline-known, and not
    already held.

    A market is considered actively priced when its YES ask is between 1¢ and 99¢.
    Markets at 0¢ or 100¢ are effectively settled or completely illiquid — trading
    them offers no edge. Markets whose tickers are in excluded_tickers are already
    held in the portfolio and must not be traded again. A market whose close_time
    is None (missing or unparseable in the API payload, see _market_from_dict) is
    also dropped here: this filter is the first thing both find_time_series_pairs
    and find_same_title_pairs call, and every downstream consumer treats
    close_time as a real datetime — the close_time sort and deadline-gap
    arithmetic in the finders, _pair_max_sum's tiered ceiling, and
    strategy.compute_trade's days-to-close normalization all raise on None.
    Dropping such markets keeps the finders' "skip bad markets, don't raise"
    contract.

    Args:
        markets (list): List of Kalshi market API objects to filter.
        excluded_tickers (set | None): Set of ticker strings to skip. If None,
            no tickers are excluded.
        warn_missing_close (bool): Whether to emit the missing-close_time
            summary WARNING. Keyword-only, and True by default so any
            standalone caller keeps the signal. Both run modes call this on the
            SAME market list twice — once per finder — so the second caller
            passes False to keep it at ONE line per run, which is what
            CLAUDE.md specifies. This flag NEVER changes which markets are
            dropped, only whether the drop is reported (TS-22).

    Returns:
        list: Subset of markets that have a non-None close_time and a parseable
            YES ask in [0.01, 0.99], and whose ticker is not in
            excluded_tickers. Markets with a missing/unparseable close_time are
            ALWAYS dropped; they are reported once as a single summary WARNING
            with the count (silent when none were dropped, or when
            warn_missing_close is False).
    """
    excluded = excluded_tickers or set()
    active = []
    missing_close_time = 0
    for m in markets:
        if m.ticker in excluded:
            continue
        # An unknown deadline can't be sorted, gap-tiered, or Kelly-normalized;
        # count it for one summary warning rather than logging per market.
        if getattr(m, "close_time", None) is None:
            missing_close_time += 1
            continue
        try:
            ya = float(m.yes_ask_dollars)
            # Skip markets at 0¢ (already settled NO) or 100¢ (already settled YES)
            if _MIN_ACTIVE_PRICE <= ya <= _MAX_ACTIVE_PRICE:
                active.append(m)
        except (ValueError, TypeError):
            pass
    if missing_close_time and warn_missing_close:
        # Same silent-at-zero idiom as the trading-inactive shard skip count.
        logging.warning(
            "Skipped %d markets with missing/unparseable close_time", missing_close_time
        )
    return active


def filter_markets_within_horizon(markets: list, max_horizon_days: int | None) -> list:
    """
    Filter markets to those closing within a given number of days from now.

    Optional opt-in bet-horizon cap for the live trading path: when set, drops
    any market whose close_time is farther out than max_horizon_days from the
    moment this function runs. Applied upstream of both find_time_series_pairs
    and find_same_title_pairs so both bet types see an identical filtered list
    and a single cutoff timestamp.

    Args:
        markets (list): List of ApiMarket objects to filter.
        max_horizon_days (int | None): Maximum number of days from now a
            market's close_time may be. If None, no filtering is applied and
            markets is returned unchanged.

    Returns:
        list: Subset of markets whose close_time is <= now + max_horizon_days.
            A market with a missing or unparseable close_time is dropped when
            a horizon is active, since "closes within X days" can't be proven
            true for an unknown deadline. Returns markets unchanged when
            max_horizon_days is None.
    """
    if max_horizon_days is None:
        return markets

    cutoff = datetime.now(UTC) + timedelta(days=max_horizon_days)
    within_horizon = []
    for m in markets:
        try:
            # None close_time or a naive datetime (can't compare to a tz-aware
            # cutoff) both fail this comparison and are dropped, not raised
            if m.close_time is not None and m.close_time <= cutoff:
                within_horizon.append(m)
        except TypeError:
            pass
    # Silent when the flag is absent (the early return above), so this line
    # appears only on a run that actually asked for a horizon. Without it the
    # flag left NO evidence it had taken effect, and a live run whose pair
    # counts differed could not be attributed to it (TS-24). isoformat to the
    # second, not .date(): the cutoff is now + N days, a time of day, and
    # printing only the date implies a midnight boundary it does not have.
    logging.info(
        "Horizon filter: kept %d of %d markets closing on or before %s (--max-horizon-days %d)",
        len(within_horizon), len(markets),
        cutoff.isoformat(timespec="seconds"), max_horizon_days,
    )
    return within_horizon


def get_held_tickers(client: Any) -> set:
    """
    Fetch all tickers where the account currently holds a non-zero position.

    Iterates through all pages of the portfolio positions endpoint (via
    api_call_with_retry so a transient 429/5xx doesn't abort the run) and
    collects tickers with a non-zero position (signed contract count: negative
    = NO side, positive = YES side). These tickers are excluded from new trades
    to avoid doubling up on an existing position.

    Uses the raw-response variant + JSON parsing because the pinned SDK's
    MarketPosition model requires legacy integer-cent fields the API stopped
    sending in 2026-07 (position counts now arrive as the `position_fp`
    string) — the modeled get_positions call raises ValidationError on any
    non-empty positions page.

    The cursor loop is BOUNDED twice over (TS-05): it remembers every cursor it
    has requested, so a keyset that cycles (A, B, A, B, ...) rather than
    repeating consecutively still stops with a warning, and it stops
    unconditionally at SCANNER_MAX_PAGES. This runs before any pairing, so an
    unbounded walk here means the run never scans at all.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().

    Returns:
        set: Set of ticker strings (e.g. {"KXBTC-23DEC-T40000", ...}) where the
            user currently holds a non-zero position. Empty set if no positions exist.
    """
    held: set = set()
    cursor: str | None = None
    # Every cursor already requested, so a keyset that CYCLES (A, B, A, B, ...)
    # rather than repeating consecutively is still caught — see the guard below.
    seen_cursors: set[str] = set()
    pages = 0
    while True:
        kwargs: dict = {"limit": POSITION_PAGE_SIZE, "count_filter": "position"}
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        # Raw-response call: bypasses the broken MarketPosition model, keeps retry
        data = api_call_with_retry(
            fetch_json_page, client.get_positions_without_preload_content, **kwargs
        )
        pages += 1
        for pos in data.get("market_positions") or []:
            ticker = pos.get("ticker") or ""
            try:
                # position_fp is the signed (possibly fractional) contract count
                # the API now sends; fall back to the legacy integer field for
                # any account still receiving it
                raw = pos.get("position_fp")
                if raw is None:
                    raw = pos.get("position")
                if float(raw) != 0:
                    held.add(ticker)
            except (ValueError, TypeError):
                # If we can't parse the position, be conservative and treat it as held
                if ticker:
                    held.add(ticker)
        if pages % SCANNER_PROGRESS_LOG_EVERY_PAGES == 0:
            logging.info("Positions fetch: %d pages, %d held tickers so far", pages, len(held))
        new_cursor = data.get("cursor")
        # Stuck-cursor guard, widened from "same as the last cursor" to "any
        # cursor already used": the cursor is a keyset position, so a repeat of
        # one we already requested is proof the server isn't advancing — but a
        # keyset that CYCLES with period > 1 never repeats consecutively and
        # used to page forever (TS-05). Same bounded-scan idiom as the
        # MVE_MAX_EMPTY_PAGES bail-out below.
        if new_cursor and (new_cursor == cursor or new_cursor in seen_cursors):
            logging.warning(
                "Positions fetch: cursor did not advance (repeated) on page %d — "
                "stopping pagination to avoid an infinite loop",
                pages,
            )
            break
        # Hard page cap: bounds the walk against any cursor pathology, named or
        # not — the only unbounded scans left in the ingest path were here.
        # Guarded on new_cursor: a stream that ENDS on page SCANNER_MAX_PAGES
        # was not truncated, and must not claim it was.
        if new_cursor and pages >= SCANNER_MAX_PAGES:
            logging.warning(
                "Positions fetch: reached SCANNER_MAX_PAGES (%d) — stopping "
                "pagination; raise the constant if the account genuinely holds more",
                SCANNER_MAX_PAGES,
            )
            break
        if new_cursor:
            seen_cursors.add(new_cursor)
        cursor = new_cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break
    logging.info("Held positions: %d tickers", len(held))
    return held


@dataclass
class ApiMarket:
    """
    Lightweight market object parsed from a raw /events JSON dict.

    Stands in for the SDK's Market model, which can no longer deserialize live
    responses: as of 2026-07 the API stopped populating the legacy integer-cent
    price fields (yes_ask, no_bid, ...) that the pinned SDK (3.2.0) types as
    required, so every nested-markets page raises pydantic ValidationError.
    Carries exactly the attributes the pipeline reads off a market object
    (scanner, strategy, trader, reporter, main) — prices stay the raw
    `*_dollars` strings the API sends, matching what downstream float() calls
    already expect.

    Attributes:
        ticker (str): Market ticker, e.g. "KXBTC-24MAR-T80".
        event_ticker (str): Parent event ticker.
        title (str): Market question text.
        subtitle (str): Market subtitle, from the legacy `subtitle` key with a
            fallback to `yes_sub_title` (the API dropped `subtitle` in the
            2026-08 drift). "" when both are absent.
        status (str): SDK-style status string; open markets are "active".
        close_time (datetime | None): Parsed tz-aware close time, or None if
            missing/unparseable.
        yes_ask_dollars: YES ask as a dollar string (e.g. "0.35") or None.
        no_ask_dollars: NO ask as a dollar string or None.
        yes_bid_dollars: YES bid as a dollar string or None.
        price_level_structure (str): Tick regime name, e.g. "linear_cent",
            "tapered_deci_cent", "deci_cent". "" if absent. Read by
            tick_size_for_price(), which trader.py uses to cap V2 order prices.
        price_ranges (list[PriceRange] | None): Parsed tick-size bands (see
            _parse_price_ranges), or None when unknown/unparseable/absent.
            Also read by tick_size_for_price() — the authoritative grid.
        exchange_index (int): The exchange shard this market lives on (see
            _shard_index). Market data is cross-shard, so every shard's
            markets are ingested and simply tagged with this; on the V2 order
            path it routes the order, and while the legacy path is in use it
            is trader._legacy_routable that refuses to submit an order for a
            non-DEFAULT_EXCHANGE_INDEX market. DEFAULT_EXCHANGE_INDEX when the
            payload omits the field (fail-safe).
        _event_title (str): Parent event title attached for pair_key grouping.
    """
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    status: str
    close_time: Any
    yes_ask_dollars: Any = None
    no_ask_dollars: Any = None
    yes_bid_dollars: Any = None
    price_level_structure: str = ""
    price_ranges: Any = None  # list[PriceRange] | None — None = unknown
    exchange_index: int = DEFAULT_EXCHANGE_INDEX
    _event_title: str = field(default="")


def _market_from_dict(m: dict, event_title: str) -> ApiMarket:
    """
    Build an ApiMarket from one raw market JSON dict.

    Missing fields become the same falsy defaults the SDK model would have
    produced (None for prices, "" for strings) so downstream filters behave
    identically. close_time is parsed with datetime.fromisoformat, which
    handles the API's trailing-"Z" UTC format on Python >= 3.11.

    Args:
        m (dict): One market object from a raw events-endpoint JSON payload.
        event_title (str): Parent event's title, attached as _event_title.

    Returns:
        ApiMarket: Parsed market ready for the pair-detection pipeline.
    """
    close_dt = None
    if m.get("close_time"):
        try:
            close_dt = datetime.fromisoformat(m["close_time"])
        except (ValueError, TypeError):
            close_dt = None
    return ApiMarket(
        ticker=m.get("ticker") or "",
        event_ticker=m.get("event_ticker") or "",
        title=m.get("title") or "",
        # The API dropped `subtitle` from market payloads (2026-08 drift);
        # yes_sub_title carries the same intra-title outcome label (e.g. the
        # option name in a multi-choice event) and is the discriminator
        # same-title grouping depends on. no_sub_title is deliberately NOT
        # used — it is the negated phrasing and would yield asymmetric keys.
        subtitle=m.get("subtitle") or m.get("yes_sub_title") or "",
        status=m.get("status") or "",
        close_time=close_dt,
        yes_ask_dollars=m.get("yes_ask_dollars"),
        no_ask_dollars=m.get("no_ask_dollars"),
        yes_bid_dollars=m.get("yes_bid_dollars"),
        price_level_structure=m.get("price_level_structure") or "",
        price_ranges=_parse_price_ranges(m.get("price_ranges")),
        # Tag (never filter) the shard so downstream code — V2 order routing,
        # trader._legacy_routable, the collateral planner — can decide what
        # to do with it.
        exchange_index=_shard_index(m),
        _event_title=event_title,
    )


def _shard_index(m: dict) -> int:
    """
    Read the exchange shard a raw market dict lives on, fail-safe.

    Kalshi partitions the exchange into parallel instances keyed by an integer
    `exchange_index` on every market payload. Market-data endpoints are
    cross-shard (they return every shard's markets, tagged), so this is purely
    a labelling read — it never decides whether a market is kept.

    A missing, null, or unparseable value is reported as DEFAULT_EXCHANGE_INDEX
    rather than raising: absence of the field is the pre-sharding / sandbox
    shape and must never crash ingest. The check is deliberately explicit
    (`is None`, then a guarded `int()`) rather than the falsy idiom
    `int(m.get(...) or DEFAULT_EXCHANGE_INDEX)` — that idiom would conflate an
    explicitly declared shard 0 with a missing field, which is harmless only
    while DEFAULT_EXCHANGE_INDEX is itself 0 and silently wrong the day it
    changes to a non-zero shard.

    Args:
        m (dict): One raw market JSON dict from an events-endpoint payload.

    Returns:
        int: The market's exchange shard index, or DEFAULT_EXCHANGE_INDEX when
            the field is absent, null, or unparseable.
    """
    raw = m.get("exchange_index")
    if raw is None:
        return DEFAULT_EXCHANGE_INDEX
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_EXCHANGE_INDEX


def _status_flag(raw: Any) -> bool | None:
    """
    Normalise one /exchange/status boolean flag, tolerating JSON re-typing.

    fetch_shard_statuses() is the only producer of these status dicts, so this
    is the single place where a re-typed flag can be given back its real
    meaning: a bare bool() of the raw payload value reads the drifted string
    "false" as True, which silently un-halts a shard the exchange has halted
    and lets a collateral POST reach a shard whose transfers are disabled
    (TS-04b).

    The resolution order is: a real bool; the conventional numeric spellings
    0 and 1; then, for strings, the closed token sets in config
    (EXCHANGE_FLAG_NULL_TOKENS / _FALSE_TOKENS / _TRUE_TOKENS, stripped and
    lower-cased). Anything left is UNRECOGNISED and resolves by its truthiness
    in the one direction that cannot break a correct reading: a FALSY
    unrecognised value (0.0, "", [], {}) keeps the False that bool() already
    gave it, because turning that into "unknown" would un-drop a halted shard;
    a TRUTHY unrecognised value (2, "maybe", [1]) becomes None, which every
    trading_active consumer already treats exactly as it treated True (the
    shard stays in the ingest) and which makes the ENABLING flags — stored as
    `_status_flag(...) is True` — fail closed rather than move money on a
    value nobody can read.

    An absent, null or stringified-null flag returns None for the same reason:
    it is unknown, never halted. The caller decides what unknown means for
    each flag (keep the shard for trading_active per TS-04; refuse to move
    money for intra_exchange_transfers_active).

    Args:
        raw (Any): The value as it arrived in the JSON payload, or None when
            the key was absent.

    Returns:
        bool | None: The flag's boolean meaning, or None when it is unknown
            (absent, null, a stringified null, or an unrecognised truthy
            value).

    Raises:
        Nothing. Every input shape, including lists and dicts, resolves to a
        bool or None.
    """
    if raw is None:
        return None
    # Real bools pass through untouched.
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        # 0/1 are the other conventional JSON spelling of a wire boolean, and
        # this branch gives them exactly the reading bool() already gave them.
        # Without it, 1 would fall through to the unrecognised-truthy rule
        # below and read as unknown, which would make the enabling flags
        # (which demand `is True`) refuse every transfer under an int retyping.
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in EXCHANGE_FLAG_NULL_TOKENS:
            # A stringified null carries no more information than an absent
            # key, so it resolves the same way: unknown, never halted.
            return None
        if token in EXCHANGE_FLAG_FALSE_TOKENS:
            return False
        if token in EXCHANGE_FLAG_TRUE_TOKENS:
            return True
    # Unrecognised. Falsy keeps the pre-existing bool() reading (un-dropping a
    # halted shard is the one change this must never make); truthy is unknown.
    return False if not raw else None


def _drift_repr(value: Any) -> str:
    """
    Render a drifted flag value for a log line, with a bounded length.

    The value is whatever the API sent, so an unbounded repr() of (for
    instance) a large array would emit a multi-KB line on every run — the
    per-line log bloat TS-02 removed from the candlestick and event-title
    paths.

    Args:
        value (Any): The raw payload value to render.

    Returns:
        str: repr(value), truncated to config.EXCHANGE_FLAG_DRIFT_REPR_MAX_CHARS
            characters with a "(truncated)" marker appended when it was longer.
    """
    text = repr(value)
    if len(text) > EXCHANGE_FLAG_DRIFT_REPR_MAX_CHARS:
        return text[:EXCHANGE_FLAG_DRIFT_REPR_MAX_CHARS] + "…(truncated)"
    return text


def fetch_shard_statuses(client: Any) -> dict | None:
    """
    Read the per-exchange-shard status breakdown from GET /exchange/status.

    Kalshi's sharded exchange reports one status record per shard under
    `exchange_index_statuses`. The pinned SDK's ExchangeStatus pydantic model
    silently DROPS that field (its config ignores extras), so this reads the
    raw body through the `*_without_preload_content` variant instead — the
    same raw-read pattern the events/orders/positions/balance calls already
    use for drift reasons.

    The field is documented as "absent when the per-index breakdown is
    unavailable" — the sandbox and pre-sharding shape. That case, a malformed
    payload, and ANY exception (including the HTTP call itself failing) all
    return None, meaning "assume single-shard semantics". This is deliberately
    fail-soft: an exchange-status hiccup must degrade to the pre-sharding
    behaviour, never abort a scan.

    `trading_active` is deliberately TRI-STATE on the parsed result: True,
    False, or None when the payload carried no such key at all. An absent
    field is "unknown", never "halted" — normalising it to False would mark
    every shard inactive and empty the entire ingest while the exchange is
    open (TS-04), the same fail-safe policy _shard_index applies to a missing
    exchange_index. Only a False halts a shard.

    Every flag here is read through _status_flag(), not bool(): a wire boolean
    that drifts into its string form ("false") is TRUTHY in Python, so the
    bare coercion read a halted shard as open and an un-transferable shard as
    movable (TS-04b). The three flags then differ in what UNKNOWN means, which
    is why the last step is not uniform: `trading_active` stores the tri-state
    verdict as-is, so unknown keeps the shard in the ingest (TS-04), while
    `exchange_active` and `intra_exchange_transfers_active` store
    `_status_flag(...) is True`, so anything but a recognised true — absent,
    null, false, or unreadable — is stored as False. Not moving money is the
    safe direction, and for an absent flag it is also exactly what the
    previous bool() coercion did.

    Args:
        client (Any): An authenticated KalshiClient produced by
            auth.build_client().

    Returns:
        dict | None: Mapping of exchange_index (int) -> {"trading_active":
            bool | None, "exchange_active": bool,
            "intra_exchange_transfers_active": bool, "description": str}.
            `trading_active` is None when the field was absent, null or
            unreadable; the other two flags are True only when the payload
            affirmatively said so. Malformed entries are skipped. None when
            the breakdown is unavailable or anything at all went wrong.
    """
    try:
        # Read-only GET, so api_call_with_retry's 429/5xx backoff is correct
        # here (the no-retry rule applies only to order submission).
        data = api_call_with_retry(
            fetch_json_page, client.get_exchange_status_without_preload_content
        )
        raw = data.get("exchange_index_statuses")
        if not isinstance(raw, list):
            logging.info(
                "per-shard exchange status unavailable — assuming single-shard semantics"
            )
            return None
        statuses: dict = {}
        unknown_active = 0
        drifted: list = []
        for entry in raw:
            try:
                # A non-dict entry raises AttributeError on .get; a missing or
                # non-numeric exchange_index raises TypeError/ValueError. One
                # malformed record must not discard the well-formed ones.
                idx = int(entry.get("exchange_index"))
            except (AttributeError, TypeError, ValueError):
                continue
            # ABSENT is not FALSE. A renamed or dropped field (the documented
            # API-drift class that already hit markets, positions, orders,
            # events and balance) must read as "unknown" and KEEP the shard —
            # normalising it to False empties the entire ingest at exit 0
            # (TS-04). Same fail-safe policy as _shard_index for a missing
            # exchange_index. Only a False halts a shard. A RE-TYPED field is
            # a different case: it still carries a meaning, and _status_flag
            # recovers it rather than bool()-coercing a drifted "false" (which
            # is truthy in Python) into an open shard (TS-04b).
            raw_ta = entry.get("trading_active")
            raw_ea = entry.get("exchange_active")
            raw_tx = entry.get("intra_exchange_transfers_active")
            ta = _status_flag(raw_ta)
            # Counted on the RAW value, not on `ta`: this warning says the key
            # was absent, and a key that is present but unreadable is named
            # individually by the drift warning below instead.
            if raw_ta is None:
                unknown_active += 1
            # `is True` makes unknown fail CLOSED on these two: only a
            # recognised true enables them. For an absent flag that matches
            # the previous bool() reading exactly; for an unreadable one it
            # refuses to move money on a value nobody can interpret.
            ea = _status_flag(raw_ea) is True
            # Read by trader.ensure_shard_collateral() to refuse moving funds
            # to or from a shard where transfers are disabled.
            tx = _status_flag(raw_tx) is True
            for name, raw_value, resolved in (
                ("trading_active", raw_ta, ta),
                ("intra_exchange_transfers_active", raw_tx, tx),
                ("exchange_active", raw_ea, ea),
            ):
                if raw_value is not None and not isinstance(raw_value, bool):
                    # Name the shard, the flag and the (length-bounded) raw
                    # value: a merged count says drift happened but not where,
                    # which is not actionable.
                    drifted.append(
                        f"shard {idx} {name}={_drift_repr(raw_value)} -> {resolved}"
                    )
            statuses[idx] = {
                "trading_active": ta,
                "exchange_active": ea,
                "intra_exchange_transfers_active": tx,
                "description": entry.get("description") or "",
            }
        if drifted:
            # Separate from the unknown-flag counter below: a re-typed flag
            # was READ (and may have just halted a shard), where an absent one
            # was not. Summary line, silent when nothing drifted.
            logging.warning(
                "Exchange status: non-boolean flag value(s) — API drift, read as: %s",
                "; ".join(drifted),
            )
        if unknown_active:
            # Summary WARNING, silent at zero — the drift signal an operator
            # needs, without one line per shard per run.
            logging.warning(
                "Exchange status: %d shard(s) carry no trading_active flag — "
                "treated as tradeable (unknown, not halted)",
                unknown_active,
            )
        if not statuses:
            # An empty list (or one with no parseable entries) is the same
            # "breakdown unavailable" shape as an absent field — returning {}
            # instead of None would falsely CRITICAL every ingested shard in
            # check_shard_coverage and block every collateral transfer.
            logging.info(
                "per-shard exchange status empty — assuming single-shard semantics"
            )
            return None
        return statuses
    except Exception as exc:
        logging.info(
            "per-shard exchange status unavailable — assuming single-shard "
            "semantics (%s)", exc,
        )
        return None


def inactive_shard_indexes(shard_statuses: dict | None) -> set:
    """
    Derive the shards ingest must drop from a fetch_shard_statuses() result.

    The single definition of "trading-inactive" for BOTH run modes — main.py's
    dev and prod paths call this rather than each keeping its own comprehension,
    so the two can never silently disagree about which shards a run scans.

    Args:
        shard_statuses (dict | None): Return value of fetch_shard_statuses().
            None (breakdown unavailable) means no shard is known inactive.

    Returns:
        set: exchange_index values whose trading_active flag is False — a real
            bool by this point, since fetch_shard_statuses normalises a
            re-typed "false" into one (TS-04b). None (flag absent or
            unreadable, TS-04) and True both keep the shard — unknown is not
            halted, and must never empty the ingest.
    """
    return {
        idx for idx, st in (shard_statuses or {}).items()
        if st.get("trading_active") is False
    }


def check_shard_coverage(
    advertised: dict | None, market_shards: set, balance_shards: set
) -> tuple:
    """
    Compare the shards /exchange/status advertises against what a run actually
    saw, and classify any mismatch as a real blind spot or an expected gap.

    This is a pure function — no I/O, no logging — so the caller decides how
    loudly to report each problem string; see main._log_shard_coverage for the
    logging split (critical vs warning) this function's two return lists map
    onto directly.

    Severity rationale:
        A shard the exchange advertises as `trading_active=True` but that
        produced zero ingested markets is a genuine coverage gap ONLY when it
        also holds account funds — money sitting on a shard whose order book
        we never scanned is a real blind spot, because a same-title or
        time-series pair on that shard could exist and go undetected. When no
        funds are parked there, an empty active shard is business-as-usual
        during the shard rollout — it would be wrong to CRITICAL-alert every
        single week on an expected transient state, so that case is a warning
        instead. The same empty-vs-funded split applies to a market or balance
        shard the exchange doesn't even advertise: markets ingested from an
        unadvertised shard mean the payloads and /exchange/status disagree with
        each other, which is always treated as critical (it signals the two
        data sources are out of sync, independent of money); account funds
        sitting on an unadvertised shard are logged as a warning, since funds
        alone (with no market activity to miss) are an accounting curiosity,
        not a missed trading opportunity. A `trading_active=False` advertised
        shard is never flagged at all — fetch_open_events_with_markets() drops
        its markets deliberately, and that drop already logs its own warning.
        The flag is tri-state (TS-04): only a False is skipped here (a real
        bool by this point — fetch_shard_statuses normalises a re-typed
        "false" into one), because a shard whose flag is unknown (None) is
        still scanned at ingest and must therefore still be audited for
        coverage.

    Args:
        advertised (dict | None): The per-shard status breakdown from
            fetch_shard_statuses(), or None when the breakdown was
            unavailable (sandbox / pre-sharding shape). None makes coverage
            unknowable — this function returns ([], []) unconditionally in
            that case, regardless of what market_shards/balance_shards show,
            because an observed shard with no status data to compare against
            is not actionable.
        market_shards (set): The set of exchange_index values actually seen
            among ingested ApiMarket objects for this run.
        balance_shards (set): The set of exchange_index values holding a
            NONZERO balance. Contract: the caller is responsible for this
            filtering — pass `{s for s, c in shard_balances.items() if c > 0}`
            so a shard with a zero-cent breakdown entry (still "present" in
            the raw balance dict) does not count as fund presence here. This
            function only checks set membership; it has no concept of an
            amount.

    Returns:
        tuple[list, list]: (critical, warnings) — human-readable problem
            strings. ([], []) means full coverage (or status unavailable).
    """
    if advertised is None:
        return [], []

    critical: list = []
    warnings: list = []

    for idx, status in advertised.items():
        if status.get("trading_active") is False:
            # Deliberately dropped at ingest; fetch_open_events_with_markets
            # already warns about this — not this function's job to repeat it.
            # False only — including one fetch_shard_statuses recovered from
            # a re-typed "false". An unknown flag (None) leaves the shard in
            # the ingest, so its coverage still has to be audited (TS-04).
            continue
        if idx in market_shards:
            continue
        description = status.get("description") or ""
        if idx in balance_shards:
            critical.append(
                f"advertised active shard {idx} ({description}) produced zero "
                "ingested markets but holds account funds"
            )
        else:
            warnings.append(
                f"advertised active shard {idx} ({description}) produced zero "
                "ingested markets (may be legitimately empty)"
            )

    for idx in market_shards - set(advertised):
        critical.append(
            f"markets ingested from shard {idx} which /exchange/status does "
            "not advertise — payloads and status disagree"
        )

    for idx in balance_shards - set(advertised):
        warnings.append(
            f"account funds on shard {idx} which /exchange/status does not "
            "advertise"
        )

    return critical, warnings


def fetch_open_events_with_markets(
    client: Any, inactive_shards: set | None = None
) -> list:
    """
    Fetch all open Kalshi markets via the events endpoint (with nested markets).

    Iterates the /events endpoint with `with_nested_markets=True` so each event
    response contains its constituent markets. Each market gets the parent event's
    title attached as a `_event_title` attribute — this is the key change that
    allows MVE (multivariate) markets to be safely scanned: pair-grouping uses
    `event_title + market_title`, so cross-event option-label collisions (e.g.
    "Trump" in two unrelated events) no longer false-positive into a same-title
    or time-series pair.

    Uses the SDK's raw-response variants + ApiMarket parsing instead of the
    modeled `get_events` call — the pinned SDK's Market model can no longer
    deserialize live responses (see ApiMarket docstring). Page size must be
    <= 200 (MARKET_PAGE_SIZE): the API returns HTTP 400 for larger limits.

    When `INCLUDE_MVE_MARKETS` is True, also pulls the dedicated multivariate
    events endpoint, since `get_events` excludes MVE events by API design.
    When False, only the standard endpoint is hit and the previous binary-only
    behaviour is preserved.

    Both cursor loops are BOUNDED twice over (TS-05): each remembers every
    cursor it has requested, so a keyset that cycles (A, B, A, B, ...) rather
    than repeating consecutively still stops with a warning; and each stops
    unconditionally at SCANNER_MAX_PAGES, which bounds the walk against any
    cursor pathology whether or not it was anticipated. The MVE loop keeps its
    independent MVE_MAX_EMPTY_PAGES productivity bail-out unchanged.

    Markets are TAGGED with their exchange shard, not filtered by it. The
    market-data endpoints are cross-shard, so every shard's bids and asks
    reach the pair pipeline and each ApiMarket carries its own
    `exchange_index` (see _shard_index). The ONLY ingest-time shard exclusion
    is `inactive_shards`: nothing on a shard the exchange itself reports as
    not trading-active can be traded, nor should it be left to linger as a
    stale candidate, so those markets are dropped here. Whether a
    trading-active shard's market can actually be ordered is decided at
    submission time (per-leg routing on the V2 path, trader._legacy_routable
    on the legacy one), not here.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().
        inactive_shards (set | None): Exchange shard indexes to drop at ingest
            because the exchange reports trading_active=false for them.
            None/empty (the default, and what a single-shard exchange yields)
            keeps every market.

    Returns:
        list: Flat list of ApiMarket objects for all open markets, each with
            `_event_title` set to its parent event's full title (may be empty
            string if the event had no title) and `exchange_index` set to its
            shard. May contain thousands of items.
    """
    markets: list = []
    # Normalize once so both loops below do a plain set membership test
    inactive = inactive_shards or set()
    # Counts markets dropped for living on a trading-inactive shard — shared
    # across both the standard and MVE loops below so the summary warning
    # reflects the whole fetch.
    skipped_shard = 0
    # Standard (non-MVE) events with nested markets
    cursor: str | None = None
    # Every cursor already requested, so a keyset that CYCLES (A, B, A, B, ...)
    # rather than repeating consecutively is still caught — see the guard below.
    seen_cursors: set[str] = set()
    pages = 0
    while True:
        kwargs: dict = {
            "status": "open",
            "limit": MARKET_PAGE_SIZE,
            "with_nested_markets": True,
        }
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        # Raw-response call: bypasses the broken Market model, keeps 429/5xx retry
        data = api_call_with_retry(
            fetch_json_page, client.get_events_without_preload_content, **kwargs
        )
        pages += 1
        for ev in data.get("events") or []:
            ev_title = ev.get("title") or ""
            for m in ev.get("markets") or []:
                # An "open" event can still nest closed/settled/determined markets
                # (e.g. one option in a multi-choice event already resolved while
                # the event stays open) — status="open" on get_events() filters
                # events, not their nested markets. Same "active" check as the
                # MVE branch below; the 1%-99% price filter downstream catches
                # most stale markets too, but this stops them from being paired
                # (and shown as tradeable candidates) in the first place.
                if (m.get("status") or "") != "active":
                    continue
                # A shard the exchange reports as not trading-active has no
                # live book worth pairing — drop before it becomes a candidate
                if _shard_index(m) in inactive:
                    skipped_shard += 1
                    continue
                # Parse into ApiMarket with the parent event title attached so
                # pair_key()/display_title() can find it
                markets.append(_market_from_dict(m, ev_title))
        if pages % SCANNER_PROGRESS_LOG_EVERY_PAGES == 0:
            logging.info("Open-events fetch: %d pages, %d markets so far", pages, len(markets))
        new_cursor = data.get("cursor")
        # Stuck-cursor guard, widened from "same as the last cursor" to "any
        # cursor already used": the cursor is a keyset position, so a repeat of
        # one we already requested already proves the server isn't advancing —
        # but a keyset that CYCLES with period > 1 never repeats consecutively
        # and used to page forever (TS-05). Same bounded-scan idiom as the
        # MVE_MAX_EMPTY_PAGES bail-out below.
        if new_cursor and (new_cursor == cursor or new_cursor in seen_cursors):
            logging.warning(
                "Open-events fetch: cursor did not advance (repeated) on page %d — "
                "stopping pagination to avoid an infinite loop",
                pages,
            )
            break
        # Hard page cap: bounds the walk against any cursor pathology, named or
        # not — the only unbounded scans left in the ingest path were here.
        # Guarded on new_cursor: a stream that ENDS on page SCANNER_MAX_PAGES
        # was not truncated, and must not claim it was.
        if new_cursor and pages >= SCANNER_MAX_PAGES:
            logging.warning(
                "Open-events fetch: reached SCANNER_MAX_PAGES (%d) — stopping "
                "pagination; raise the constant if the exchange genuinely lists more",
                SCANNER_MAX_PAGES,
            )
            break
        if new_cursor:
            seen_cursors.add(new_cursor)
        cursor = new_cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break

    # Multivariate events — fetched only when MVE inclusion is enabled.
    # The /events endpoint excludes these by API design, so they need their own pull.
    if INCLUDE_MVE_MARKETS:
        cursor = None
        # The MVE listing is effectively unbounded (hundreds of thousands of
        # auto-generated collection events), so the pull bails out after
        # MVE_MAX_EMPTY_PAGES consecutive unproductive pages instead of paging
        # for hours. A page is "productive" only if it contributes at least one
        # ACTIVE nested market — a page can be full of nested markets that are
        # all closed/settled/determined (observed live in sandbox: thousands of
        # such pages in a row), and those must NOT reset the counter, or the
        # bail-out never fires. Pages that contribute active markets reset it.
        empty_pages = 0
        mve_pages = 0
        mve_market_count = 0
        # Independent of empty_pages above: that counter bails on unproductive
        # pages, this set catches a keyset that CYCLES (A, B, A, B, ...) while
        # every page stays productive — see the guard below.
        seen_cursors = set()
        while True:
            kwargs = {"limit": MARKET_PAGE_SIZE, "with_nested_markets": True}
            if cursor:
                kwargs["cursor"] = cursor
            data = api_call_with_retry(
                fetch_json_page,
                client.get_multivariate_events_without_preload_content,
                **kwargs,
            )
            mve_pages += 1
            events = data.get("events") or []
            page_active_count = 0
            for ev in events:
                ev_title = ev.get("title") or ""
                for m in ev.get("markets") or []:
                    # Only include markets that are still open — MVE response may
                    # include closed/settled options inside an otherwise-open event.
                    # The API status string for an open market is "active"
                    # (allowed values: initialized/active/closed/settled/determined
                    # — there is no "open" status).
                    if (m.get("status") or "") == "active":
                        # Same trading-inactive shard drop as the standard loop
                        # above — the two share one skip counter
                        if _shard_index(m) in inactive:
                            skipped_shard += 1
                            continue
                        markets.append(_market_from_dict(m, ev_title))
                        mve_market_count += 1
                        page_active_count += 1
            if mve_pages % SCANNER_PROGRESS_LOG_EVERY_PAGES == 0:
                logging.info(
                    "MVE events fetch: %d pages, %d markets so far", mve_pages, mve_market_count
                )
            if page_active_count == 0:
                empty_pages += 1
                if empty_pages >= MVE_MAX_EMPTY_PAGES:
                    logging.warning(
                        "MVE fetch: %d consecutive pages with no active nested "
                        "markets — stopping the MVE pull early (nested markets "
                        "may still be present but none are active/usable; the "
                        "listing itself is effectively unbounded)",
                        empty_pages,
                    )
                    break
            else:
                empty_pages = 0
            new_cursor = data.get("cursor")
            # Stuck-cursor guard, widened from "same as the last cursor" to
            # "any cursor already used": the cursor is a keyset position, so a
            # repeat of one we already requested proves the server isn't
            # advancing — but a keyset that CYCLES with period > 1 never
            # repeats consecutively and used to page forever (TS-05). Same
            # bounded-scan idiom as the empty-pages bail-out just above, which
            # keeps its own independent counter and reset semantics.
            if new_cursor and (new_cursor == cursor or new_cursor in seen_cursors):
                logging.warning(
                    "MVE events fetch: cursor did not advance (repeated) on page %d — "
                    "stopping pagination to avoid an infinite loop",
                    mve_pages,
                )
                break
            # Hard page cap: bounds the walk against any cursor pathology,
            # named or not, independently of the productivity bail-out above.
            # Guarded on new_cursor: a stream that ENDS on page
            # SCANNER_MAX_PAGES was not truncated, and must not claim it was.
            if new_cursor and mve_pages >= SCANNER_MAX_PAGES:
                logging.warning(
                    "MVE events fetch: reached SCANNER_MAX_PAGES (%d) — stopping "
                    "pagination; raise the constant if the exchange genuinely lists more",
                    SCANNER_MAX_PAGES,
                )
                break
            if new_cursor:
                seen_cursors.add(new_cursor)
            cursor = new_cursor
            if not cursor:
                break

    if skipped_shard:
        logging.warning(
            "Skipped %d markets on trading-inactive exchange shards %s",
            skipped_shard, sorted(inactive),
        )
    # Per-shard ingest counts are the only signal of which shards we actually
    # saw markets on — load-bearing for diagnosing a coverage gap after a
    # market category migrates to a new shard. Always logged, even single-shard.
    logging.info(
        "Ingested markets by shard: %s",
        dict(sorted(Counter(m.exchange_index for m in markets).items())),
    )
    logging.info("Fetched %d open markets (MVE included: %s)", len(markets), INCLUDE_MVE_MARKETS)
    return markets


def find_time_series_pairs(
    client: Any,
    held_tickers: set | None = None,
    markets: list | None = None,
    inactive_shards: set | None = None,
) -> list:
    """
    Find time-series candidate pairs (YES on the earlier contract, NO on the later).

    Grouping strategy: EXACT matching over
    `time_series_group_key(pair_key(m), m.subtitle)` — the date-stripped
    `event_title + market_title` key (see `pair_key`) PLUS the market's outcome
    discriminator. If two contracts differ ONLY in their deadline, stripping
    all date tokens from the combined key yields the exact same string; the
    event-title prefix keeps multivariate option labels (e.g. "Trump" appearing
    in unrelated events) from false-positive pairing, and the subtitle keeps
    two different OUTCOMES — two strikes of one daily price family, two
    temperature bands — out of one group, where the best-pair rule below would
    otherwise have paired the widest mismatch (DR-01).

    A pair is eligible when:
      1. Both markets are actively priced: ask price in [1%, 99%]
      2. Different event_tickers (which rules out an MVE event's multi-choice
         option labels) — UNLESS config.TIME_SERIES_SAME_EVENT_LADDERS is on
         and the two markets are two rungs of ONE event's cumulative deadline
         LADDER (DR-73). Kalshi lists a question's several deadlines as
         separate markets inside a single event, and two such rungs are the
         time-series premise itself. That branch demands a shared NON-EMPTY
         event ticker, wording that is not identical, item 4's cumulative
         verdict, and two deadlines that read to two DIFFERENT calendar dates
         (stated_deadline / same_event_ladder) no more than
         MAX_DEADLINE_GAP_DAYS apart — then orders the legs and tiers the
         pair on those STATED deadlines, never on close_time, and carries the
         gap out on CandidatePair.stated_gap_days. With the switch off (the
         shipped default) every same-event candidate is skipped and counted,
         exactly as it always was.
      3. NOT identical wording across two events of one series: when the raw
         (title, subtitle, event title) triple matches on both legs AND the two
         event tickers resolve to one series (_same_series, which collapses
         every combo (KXMVE*) prefix onto one family — DR-55), the deadline
         lives outside the wording entirely, so these are two instances of one
         recurring fixture rather than one question at two deadlines and there
         is no cumulative-deadline premise (DR-02, DR-54). find_same_title_pairs
         refuses the same shape; if only it did, this finder would simply
         relabel the pair as a time-series bet and main._dedup_pairs — which
         drops the time-series copy only when a same-title copy exists — would
         have nothing to drop it against. Such candidates are counted and
         reported once after the loop on their own silent-at-zero INFO line
         (M10), as find_same_title_pairs reports its own one-series skips;
         backtester._extract_pairs logs its sweep's count on a line that
         differs only in its parenthesis.
      4. Both legs are CUMULATIVE-deadline markets ("will X happen BY
         <date>") stating two DIFFERENT deadlines — cumulative_deadline_pair
         over deadline_phrasing. Kalshi also lists SNAPSHOT markets ("Bitcoin
         price ON Sep 15, 2026?"), whose probabilities do not nest: SOL >= $180
         on Sep 14 does not imply SOL >= $180 on Sep 18, so there is no
         in-between mass to dispute and nesting is what would make the fair
         value of YES-A + NO-B at most $1 — on a snapshot pair it doesn't.
         normalize_title strips a dated snapshot title exactly as it strips a
         dated deadline one, so such a family lands in ONE group here and used
         to be sized and traded. Fails CLOSED: wording that names no deadline
         at all is refused, because nothing the finder reads can prove the
         premise. Each leg's deadlines are read from the one field that
         decided its verdict, never from the other two (DR-69).
      5. Skips candidates whose leg ASK prices already sum to $1 or more. A
         win pays only $1, so such a pair cannot profit in any cell. It is not
         impossible: wide books quote it even on genuine ladders
         (KXDEFAULT-28DEC31/-29DEC31 at 0.11 + 0.96 on 2026-09-21). It only
         changes which NON-tradeable row represents a group, because tradeable
         already requires 1 - pA - nB > fee.
      6. Deadline gap <= MAX_DEADLINE_GAP_DAYS (30 days), measured
         order-independently by deadline_gap_days() for a cross-event pair and
         by the two STATED deadlines for a same-event ladder — one number
         either way afterwards, through pair_gap_days()
      7. pB - pA >= min_price_diff_for_gap(gap_days) — directional: the
         LATER contract (B) — later by close_time for a cross-event pair, by
         STATED deadline for a same-event ladder — must be priced higher than
         the earlier one (A) by at least the tier (15% when the deadlines are <= 15 days
         apart, 30% for 16-30 days). That gap is the market-implied
         probability that the event first happens between the two deadlines;
         the strategy disputes it. A pricier EARLIER contract is never a
         candidate — there is no in-between mass to dispute.

    Per normalized title+outcome key, keeps the single best pair (tradeable
    preferred, then largest pB - pA). NOTE: since DR-01 that key carries the
    outcome label, so the rule no longer bounds one FAMILY to one pair — a
    daily family of N strikes now yields up to N pairs, every one on the same
    underlying over the same window, and nothing downstream caps that
    concentration (strategy.select_portfolio dedups tickers only).

    The legs are YES on A at pA and NO on B at nB, so tradeable=True when
    pA + nB < 1 - fee_per_pair_approx(pA, nB) AND pB > pA. A cumulative-deadline
    pair has exactly THREE settlement cells:
      - event by A's deadline: A=YES, hence B=YES — YES-on-A pays $1, win;
      - never by B's deadline: A=NO, B=NO — NO-on-B pays $1, win;
      - in between: A=NO, B=YES — both legs worthless, the full stake
        (pA + nB plus fees) is lost.
    A=YES with B=NO cannot occur for a cumulative-deadline pair (YES by the
    earlier deadline implies YES by the later one). That premise is now
    screened on WORDING at pair formation by item 4 (a heuristic) rather than
    merely assumed; the backtester's premise-violation counter remains as
    defence in depth, since a text classifier over market wording is a
    heuristic, not a proof. The flag therefore says a win pays more than the
    pair costs, NOT that the pair
    cannot lose: this is a directional bet whose expected value is negative
    at market prices unless the market overstates the in-between probability
    (config.time_series_profit_prob). Same-title pairs (no deadline gap,
    simpler co-resolution model) are preferred over time-series ones wherever
    both exist.

    Args:
        client (Any): Authenticated KalshiClient, used only when markets is None.
        held_tickers (set | None): Tickers to exclude (currently-held positions).
            None or empty means exclude nothing.
        markets (list | None): Pre-fetched ApiMarket list to scan. When None,
            fetches all open markets via fetch_open_events_with_markets(client).
        inactive_shards (set | None): exchange_index values the exchange
            reports trading_active=false for, forwarded to that fallback fetch
            so this path applies the same single ingest-time shard exclusion
            both run modes do. IGNORED when markets is supplied — which is what
            every caller does today, making the fallback unreachable. None
            excludes no shard.

    Returns:
        list: CandidatePair objects, one per normalized title+outcome group
            that produced a pair, each carrying pair_type="time_series". Empty
            if no group has two markets on different event_tickers — or, with
            TIME_SERIES_SAME_EVENT_LADDERS on, two dated rungs of ONE event —
            within the
            deadline-gap cap whose wording is not identical across one series
            and states two different cumulative deadlines — the two deadlines
            compared as normalized strings, not parsed calendar dates (see the
            DR-67 Known residuals in CLAUDE.md). A pair formed by the
            same-event ladder branch (item 2, off by default) additionally
            carries stated_gap_days, the calendar-day gap between its two
            STATED deadlines; every other pair carries None there.
    """
    if markets is None:
        # Fetch all open markets from the Kalshi API if not supplied by the
        # caller. The exclusion must match main's: nothing on a shard the
        # exchange reports trading_active=false for can be traded, and it must
        # not linger as a stale candidate either. Without the forward this path
        # would ingest and PAIR markets on halted shards — the one ingest-time
        # exclusion that is mandatory (TS-27).
        markets = fetch_open_events_with_markets(client, inactive_shards=inactive_shards)

    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    active = _filter_active_markets(markets, held_tickers)
    logging.info("Actively priced markets (ask in 1%%–99%%): %d", len(active))

    # Group by exact normalized title + outcome label over the combined
    # (event + market) key. Stripping date tokens means two markets that differ
    # ONLY in their deadline produce the same key; event_title in the key
    # prevents two unrelated MVE events sharing an option label (e.g. "Trump")
    # from being grouped; and the subtitle keeps two different OUTCOMES of one
    # title (e.g. two strikes of a daily crypto family) apart — without it the
    # best-pair rule below picks the widest strike mismatch in the group and
    # sizes it as one cumulative-deadline pair (DR-01).
    by_title: dict = defaultdict(list)
    for m in active:
        norm = time_series_group_key(pair_key(m), getattr(m, "subtitle", "") or "")
        # Skip markets whose title collapses entirely to an empty string after stripping
        if norm:
            by_title[norm].append(m)

    logging.info("Distinct normalized title+outcome keys with >= 1 market: %d", len(by_title))

    # Classify each market's wording ONCE. Two reasons this is not done inside
    # the candidate loop below: a group of N markets produces O(N^2) candidate
    # pairs, so per-pair classification re-runs the regex tables an order of
    # magnitude more often than needed; and the verdict is a property of the
    # market, not of the pair.
    profiles: dict = {m.ticker: _market_deadline_profile(m) for m in active}

    # ALWAYS logged, not only when something is skipped. A rule that can empty
    # the strategy must not be detectable solely by the absence of a warning —
    # "cumulative: 0" is what separates "the exchange lists no cumulative
    # families right now" from "the phrasing tables are broken" (the DR-66
    # lesson applied to this rule).
    phrasing_census: dict = defaultdict(int)
    for verdict, _spans in profiles.values():
        phrasing_census[verdict] += 1
    logging.info(
        "Deadline phrasing of actively priced markets: %d cumulative, %d snapshot, %d unknown",
        phrasing_census[DEADLINE_CUMULATIVE],
        phrasing_census[DEADLINE_SNAPSHOT],
        phrasing_census[DEADLINE_UNKNOWN],
    )

    # DR-73: each rung's STATED deadline as a calendar date, read ONCE per
    # market (DR-70's rule — a group of N markets produces O(N^2) candidate
    # pairs, and one event's deadline ladder is the densest such group there
    # is), never per candidate pair. Not computed at all while the switch is
    # off, so a run with ladders disabled does exactly the work it did before.
    #
    # Each value is a PAIR of readings: the cross-checked deadline
    # stated_deadline() reports, and the same read with the cross-check
    # disarmed — it only ever reads the three FIELD arguments, so passing ""
    # for all three leaves the deciding field's own day. The second is used
    # for nothing but naming WHY a refused candidate was refused: a rung
    # stating no placeable day at all reads None both ways, while one whose
    # own fields name irreconcilable days reads None only with the
    # cross-check on.
    ladder_deadlines: dict = {}
    if TIME_SERIES_SAME_EVENT_LADDERS:
        for m in active:
            profile = profiles[m.ticker]
            ladder_deadlines[m.ticker] = (
                stated_deadline(
                    profile,
                    getattr(m, "_event_title", "") or "",
                    getattr(m, "title", "") or "",
                    getattr(m, "subtitle", "") or "",
                ),
                stated_deadline(profile, "", "", ""),
            )

    # Candidates refused because the two legs are not one question at two
    # cumulative deadlines, split by REASON (DR-72) rather than folded into
    # one counter: a snapshot leg, a leg naming no comparable deadline, and
    # two legs naming the SAME deadline are three different findings with
    # three different remedies, and collapsing them hid which one applied.
    # Plus the price-sum skip below. All reported once after the loop
    # (silent at zero) — the same summary idiom find_same_title_pairs uses
    # for its series skips. The checks themselves are NOT reordered: the
    # wording check still runs strictly before the price-sum check, so this
    # split cannot shift any candidate's count into price_sum_skips.
    snapshot_skips = 0
    no_deadline_skips = 0
    same_deadline_skips = 0
    # Candidates that named two DIFFERENT cumulative deadlines but were
    # refused for being more than MAX_DEADLINE_GAP_DAYS apart — the wording
    # was fine, only the gap wasn't. Reported separately below, since "worded
    # right but too far apart" is a different signal than "not worded as two
    # deadlines at all".
    gap_cap_skips = 0
    price_sum_skips = 0
    # Cross-event candidates refused by the one-series conjunct (M10):
    # identical wording across two events of one series. It was the one
    # refusal ahead of the price filters in this branch with no count, so a
    # run whose time-series zero it caused logged no cause; counted where it
    # fires, before the wording check, exactly as find_same_title_pairs
    # counts its own series skips. (The price-parse and tier `continue`s
    # further down stay uncounted, as they always were.)
    series_skips = 0
    # DR-73's same-event ladder branch keeps its OWN counters rather than
    # adding to the six above. They count a DIFFERENT population — candidates
    # inside one event, which every previous version of this finder refused
    # outright — and folding them in was measured to move snapshot_skips by 12
    # and price_sum_skips by 2 on the 2026-09-22 snapshot, blurring exactly
    # the distinction DR-72 had just split apart. Every one is silent at zero,
    # so a run with the switch off adds a single line (the disabled count).
    ladder_disabled_skips = 0
    ladder_no_event_skips = 0
    ladder_identical_wording_skips = 0
    ladder_snapshot_skips = 0
    ladder_no_deadline_skips = 0
    ladder_same_deadline_skips = 0
    ladder_undated_skips = 0
    ladder_field_conflict_skips = 0
    ladder_same_day_skips = 0
    ladder_gap_cap_skips = 0
    ladder_price_sum_skips = 0

    candidate_pairs: list = []
    for norm_title, members in by_title.items():
        # Need at least two markets in a group to form any pair
        if len(members) < 2:
            continue

        # Sort ascending by close_time so a cross-event candidate's mA is
        # always the earlier-closing contract. A same-event LADDER re-orders
        # its own two legs on their STATED deadlines below (DR-73) — this
        # sort cannot order those, because a ladder's rungs may close at one
        # instant or in the opposite order to their deadlines.
        members_sorted = sorted(members, key=lambda m: m.close_time)
        group_pairs: list = []

        for i, m_outer in enumerate(members_sorted):
            for m_inner in members_sorted[i + 1:]:
                # Fresh per-candidate locals, NEVER the loop variables: a
                # same-event ladder is ordered by STATED deadline and may swap
                # its legs below, and swapping the OUTER loop variable in
                # place would leak into every later m_inner of this iteration.
                # find_same_title_pairs already uses this idiom for exactly
                # the same reason.
                mA, mB = m_outer, m_inner
                # A same-event ladder's gap is its STATED deadline gap; None
                # means "this pair is tiered on close_time" (pair_gap_days).
                stated_gap = None

                if mA.event_ticker == mB.event_ticker:
                    # DR-73: two markets of ONE event. Every previous version
                    # of this finder skipped these outright as "options
                    # within a multi-choice event" — true of an MVE event's
                    # option labels, but NOT of a cumulative deadline LADDER,
                    # where Kalshi lists one question's several deadlines as
                    # separate markets inside a single event ("Will SpaceX
                    # launch another Starship by Sep 23, 2026?" and "... by
                    # Oct 16, 2026?" are both KXSPACEXSTARSHIP-14). Two such
                    # rungs are precisely the time-series premise: the earlier
                    # deadline's event nests inside the later one's. The
                    # switch ships OFF — see
                    # config.TIME_SERIES_SAME_EVENT_LADDERS for the nesting
                    # evidence and for the exposure turning it on would take.
                    if not TIME_SERIES_SAME_EVENT_LADDERS:
                        # Counts the whole same-event candidate POPULATION,
                        # deliberately ahead of every eligibility check below:
                        # it is the first arrow of the funnel in
                        # config.TIME_SERIES_SAME_EVENT_LADDERS' comment, and
                        # an operator must be able to read that number off a
                        # switch-OFF log. It says what the switch held back,
                        # not that every one of them would have paired.
                        ladder_disabled_skips += 1
                        continue
                    if not mA.event_ticker:
                        # Two markets sharing an EMPTY event ticker share no
                        # event at all, so nothing identifies the ladder they
                        # would belong to. Fails closed, exactly as it did
                        # before this branch existed.
                        ladder_no_event_skips += 1
                        continue
                    if _identical_wording(mA, mB):
                        # Same event AND identical wording: the deadline is
                        # not in the wording, so there is nothing here to
                        # order two rungs by — DR-02's reasoning, reached one
                        # level in. cumulative_deadline_pair below would
                        # refuse it too (identical wording states identical
                        # spans), but refusing it here keeps the ladder
                        # counters about ladders — and it is the LARGEST
                        # single ladder refusal on the 2026-09-22 snapshot
                        # (502 of 3,354), so it gets its own line rather than
                        # vanishing out of the funnel (DR-72's rule).
                        ladder_identical_wording_skips += 1
                        continue
                    if not cumulative_deadline_pair(
                        profiles[mA.ticker], profiles[mB.ticker]
                    ):
                        # The same three DR-72 reasons as the cross-event
                        # branch, on their own counters.
                        reason = deadline_pair_refusal(
                            profiles[mA.ticker], profiles[mB.ticker]
                        )
                        if reason == REFUSED_SNAPSHOT:
                            ladder_snapshot_skips += 1
                        elif reason == REFUSED_NO_STATED_DEADLINE:
                            ladder_no_deadline_skips += 1
                        elif reason == REFUSED_SAME_DEADLINE:
                            ladder_same_deadline_skips += 1
                        continue

                    # Both rungs' deadlines as CALENDAR DATES, read once per
                    # market above. cumulative_deadline_pair compares the two
                    # spans as opaque strings, which establishes that they
                    # DIFFER but not which is earlier nor by how many days —
                    # and close_time answers neither for a ladder, whose rungs
                    # a settled event closes at one instant (in the archive
                    # 681 of 1,821 dated same-event pairs have a close gap of
                    # zero days and 13 are ordered the wrong way round by
                    # realized close).
                    deadline_a, deciding_a = ladder_deadlines[mA.ticker]
                    deadline_b, deciding_b = ladder_deadlines[mB.ticker]
                    ladder = same_event_ladder(deadline_a, deadline_b)
                    if ladder is None:
                        # Two different findings, counted apart: a rung whose
                        # wording states no placeable day (a parser gap —
                        # year-less wording, mostly), and a rung whose own
                        # fields name irreconcilable days (stale wording in
                        # one of them). deciding_* is the same read with the
                        # cross-check disarmed, so it is what separates them.
                        if (deadline_a is None and deciding_a is not None) or (
                            deadline_b is None and deciding_b is not None
                        ):
                            ladder_field_conflict_skips += 1
                        else:
                            ladder_undated_skips += 1
                        continue
                    if ladder is SAME_DAY:
                        # Both rungs name ONE calendar day — one deadline
                        # spelled two ways ("by Mar 31, 2027" beside "before
                        # Apr 1, 2027"), not a two-rung ladder. Compared with
                        # `is`, never truthiness: SAME_DAY is a non-empty
                        # string and so is truthy.
                        ladder_same_day_skips += 1
                        continue
                    swap, stated_gap = ladder
                    if stated_gap > MAX_DEADLINE_GAP_DAYS:
                        # The ladder's OWN gap cap, deliberately inside this
                        # branch so gap_cap_skips below keeps counting exactly
                        # the cross-event candidates it counts today.
                        ladder_gap_cap_skips += 1
                        continue
                    if swap:
                        # market_a must be the EARLIER contract, which for a
                        # ladder means the earlier STATED deadline:
                        # members_sorted ordered this group by close_time,
                        # which a ladder's rungs can share outright.
                        mA, mB = mB, mA
                else:
                    # Identical wording across two events of one series is the
                    # same-title shape find_same_title_pairs now refuses (DR-02,
                    # DR-54): the deadline lives outside the wording, so these are
                    # two instances of one recurring fixture and there is no
                    # cumulative-deadline premise to trade. Without this conjunct
                    # the same-title gate would merely RELABEL such a pair as a
                    # time-series bet — main._dedup_pairs only ever dropped the
                    # time-series copy because a same-title copy existed.
                    if _identical_wording(mA, mB) and _same_series(mA, mB):
                        series_skips += 1
                        continue

                    # The pair must be one question asked at two different
                    # CUMULATIVE deadlines. Kalshi also lists SNAPSHOT markets
                    # ("Bitcoin price ON Sep 15, 2026?"), whose probabilities do
                    # not nest — SOL >= $180 on Sep 14 does not imply SOL >= $180
                    # on Sep 18 — so the YES-on-earlier / NO-on-later trade has no
                    # premise there at all, and nesting — which is what would make
                    # the fair value of YES-A + NO-B at most $1 — does not hold on
                    # a snapshot pair. normalize_title strips a dated snapshot
                    # title just as readily as a dated deadline one, so such a
                    # family lands in ONE group here and was previously sized and
                    # traded. Fails CLOSED on wording that names no deadline.
                    if not cumulative_deadline_pair(
                        profiles[mA.ticker], profiles[mB.ticker]
                    ):
                        # Same decision as cumulative_deadline_pair, re-asked for
                        # its REASON (DR-72) — deadline_pair_refusal is the one
                        # definition both calls share, so the verdict here and the
                        # boolean just tested can never disagree.
                        reason = deadline_pair_refusal(
                            profiles[mA.ticker], profiles[mB.ticker]
                        )
                        if reason == REFUSED_SNAPSHOT:
                            snapshot_skips += 1
                        elif reason == REFUSED_NO_STATED_DEADLINE:
                            no_deadline_skips += 1
                        elif reason == REFUSED_SAME_DEADLINE:
                            same_deadline_skips += 1
                        continue

                    # Deadline gap check: past 30 days too much of the market-implied
                    # in-between probability is genuine for the trade to dispute it.
                    # Order-independent (same helper _pair_max_sum and the backtester use)
                    gap_days = deadline_gap_days(mA, mB)
                    if gap_days > MAX_DEADLINE_GAP_DAYS:
                        # Wording was fine — the pair passed cumulative_deadline_pair
                        # just above — only the gap wasn't. Counted separately from
                        # the wording refusals above (DR-72): this is the live signal
                        # that cumulative families exist but sit too far apart to
                        # trade, not that the wording screen is failing.
                        gap_cap_skips += 1
                        continue

                # A ladder's gap is its STATED gap, already capped inside the
                # branch above; a cross-event pair re-reads the close_time gap
                # the else branch just checked. One expression, so everything
                # below — the tier, the ceiling, the pair itself — is measured
                # on the same number the admission check used.
                gap_days = stated_gap if stated_gap is not None else deadline_gap_days(mA, mB)

                try:
                    pA = float(mA.yes_ask_dollars)
                    pB = float(mB.yes_ask_dollars)
                    # nA is NOT a leg price here — it is read so the prod log's
                    # "nA (NO ask)" column stays meaningful (reporting only)
                    nA = float(mA.no_ask_dollars)
                    # nB IS a leg price: the cost of the NO bought on the later contract
                    nB = float(mB.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # Enforce the minimum YES price difference required for time-series
                # pairs, tiered by deadline gap (15% for gaps <= 15 days, 30% for
                # 16-30 days — a wider gap leaves more room for the event to land
                # between the deadlines, so more of the market's in-between mass is
                # genuine and a bigger gap is demanded before disputing it).
                # Directional, not abs(): mA is always the EARLIER contract —
                # by close_time for a cross-event pair (the sort above), by
                # STATED deadline for a same-event ladder (swapped inside the
                # branch above, because that sort cannot order one) — and the
                # bet only exists when the LATER contract is
                # priced higher (pB > pA) — the gap is the market-implied in-between
                # probability we dispute. A pricier earlier contract (pA > pB) has
                # no in-between mass to dispute, is not a candidate, and using abs()
                # here would let such pairs through as untradeable placeholders that
                # could still win the group's one-pair-per-title slot below. Mirrors
                # the directional check in backtester._find_entry.
                # PRICE_EPSILON, not a bare <: both prices are floats parsed
                # from cent-quantized dollar strings, so an exactly-at-tier
                # gap can evaluate a hair under it (0.45 - 0.30 ->
                # 0.15000000000000002 is fine, but 0.35 - 0.20 ->
                # 0.14999999999999997 is not) and the pair is rejected for
                # representation noise rather than for its price (TS-09).
                if pB - pA < min_price_diff_for_gap(gap_days) - PRICE_EPSILON:
                    continue

                # Skips candidates whose leg ASK prices already sum to $1 or
                # more. A win pays only $1, so such a pair cannot profit in
                # any cell. It is not impossible: wide books quote it even on
                # genuine ladders (KXDEFAULT-28DEC31/-29DEC31 at 0.11 + 0.96
                # on 2026-09-21) — nesting makes the fair value of YES-A +
                # NO-B at most $1, not the quoted asks. It is already IMPLIED
                # by tradeable below (which requires 1 - pA - nB > fee), but
                # only as a flag — a violating pair was still constructed and
                # could win this group's one-pair slot at the sort below,
                # blocking a sound runner-up. Skipping it outright only
                # changes which NON-tradeable row represents such a group.
                #
                # The epsilon runs the OPPOSITE way from the qualifying-level
                # filters in enrich_with_orderbook_prices/validate_pair_price,
                # which ADD it to a keep bound so a level exactly on the bound
                # is not dropped for float noise. This is the REJECT side: the
                # skip condition is strictly pA + nB < 1, so a sum that is
                # really 1.0 but evaluates to 0.9999999 must still be refused.
                # Subtracting tightens by at most PRICE_EPSILON (1e-6), two
                # orders of magnitude below the finest tick, so it can never
                # reject a genuinely sub-$1 pair (TS-09's rule is "never
                # reject for representation noise", and this honours it) — in
                # practice it is defensive and inert on every Kalshi grid
                # (4-dp prices; complementary pairs sum to exactly 1.0).
                if pA + nB >= 1.0 - PRICE_EPSILON:
                    # Counted on the ladder's own line when it is one, so a
                    # ladder run cannot silently inflate the cross-event
                    # count (DR-73).
                    if stated_gap is not None:
                        ladder_price_sum_skips += 1
                    else:
                        price_sum_skips += 1
                    continue

                # tradeable=True when a win scenario (YES-on-A or NO-on-B paying $1)
                # covers both leg prices plus the approximate fees — the leg prices
                # are pA and nB, not nA/pB. This is not a guarantee against the
                # in-between loss cell; the pB > pA conjunct restates the direction.
                # fee_per_pair_approx returns a continuous estimate of total taker fees.
                tradeable = ((1.0 - pA - nB) > fee_per_pair_approx(pA, nB)) and (pB > pA)

                group_pairs.append(
                    CandidatePair(
                        market_a=mA,
                        market_b=mB,
                        pA=pA,
                        pB=pB,
                        nA=nA,
                        tradeable=tradeable,
                        canonical_title=norm_title,
                        pair_type="time_series",
                        nB=nB,
                        # None for a cross-event pair (tier it on close_time);
                        # the stated gap for a ladder, so pair_gap_days can
                        # hand every downstream re-derivation the very number
                        # this pair was admitted and tiered on (DR-73).
                        stated_gap_days=stated_gap,
                    )
                )

        if not group_pairs:
            continue

        # Keep only the single best pair per normalized title+outcome group to avoid flooding
        # the portfolio with many near-identical positions. Tradeable pairs rank above
        # non-tradeable ones; within each tier, the largest pB - pA (the disputed
        # in-between probability) wins. pB > pA holds for every entry in group_pairs
        # (see the directional filter above), so no abs() is needed.
        group_pairs.sort(key=lambda p: (p.tradeable, p.pB - p.pA), reverse=True)
        candidate_pairs.append(group_pairs[0])

    # The one-series conjunct's count (M10), silent at zero: the same line
    # backtester._extract_pairs logs for its sweep, whose parenthesis says
    # "within the deadline-gap window" instead, as its DR-72 lines do. The
    # words "two instances of one event series" are shared on purpose with
    # find_same_title_pairs' line, so one grep finds the rule on both paths
    # and in both finders.
    if series_skips:
        logging.info(
            "Time-series candidates skipped as two instances of one event "
            "series (identical wording, different fixture; counted before the "
            "gap and price filters): %d",
            series_skips,
        )
    # Three separate INFO lines, each silent at zero (DR-72) — the same
    # summary idiom find_same_title_pairs uses for its series skips, split by
    # REASON instead of folded into one "not a cumulative-deadline pair"
    # count so a reader can tell a wording problem (snapshot families, no
    # recognised deadline, one deadline said twice) from a distance problem
    # (the gap-cap line below). All three count candidate pairs within a
    # group, before the gap and price filters run.
    if snapshot_skips:
        logging.info(
            "Time-series candidates refused because a leg's deciding field "
            "is snapshot wording (counted before the gap and price filters): %d",
            snapshot_skips,
        )
    if no_deadline_skips:
        logging.info(
            "Time-series candidates refused because a leg's deciding field "
            "carries no recognised deadline wording or no comparable date "
            "(counted before the gap and price filters): %d",
            no_deadline_skips,
        )
    if same_deadline_skips:
        logging.info(
            "Time-series candidates refused because the two deciding fields "
            "state the same deadline, or truncate to one (counted before the "
            "gap and price filters): %d",
            same_deadline_skips,
        )
    if gap_cap_skips:
        # The live signal that cumulative families exist but sit too far
        # apart to trade — distinct from the wording refusals above, which
        # mean the families themselves were never found to be cumulative at
        # all. Can include wording false positives (a heuristic misread as
        # cumulative that happens to also miss the gap cap).
        logging.info(
            "Time-series candidates worded as two different cumulative "
            "deadlines, refused at the %d-day gap cap (tier and price not "
            "evaluated): %d",
            MAX_DEADLINE_GAP_DAYS,
            gap_cap_skips,
        )
    if price_sum_skips:
        logging.info(
            "Time-series candidates skipped for a leg price sum at or above $1: %d",
            price_sum_skips,
        )

    # DR-73's own reporting. Every line below counts candidates INSIDE one
    # event — a population none of the counters above has ever seen — and each
    # is silent at zero, so a run with the switch off adds exactly one line
    # (the disabled count) to this finder's output.
    if ladder_disabled_skips:
        logging.info(
            "Same-event candidates skipped because same-event deadline "
            "ladders are disabled (config.TIME_SERIES_SAME_EVENT_LADDERS): %d",
            ladder_disabled_skips,
        )
    if ladder_no_event_skips:
        logging.info(
            "Same-event ladder candidates refused because the shared event "
            "ticker is empty: %d",
            ladder_no_event_skips,
        )
    if ladder_identical_wording_skips:
        logging.info(
            "Same-event ladder candidates refused because the two rungs' "
            "wording is identical (the deadline is not in the wording): %d",
            ladder_identical_wording_skips,
        )
    if ladder_snapshot_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deciding "
            "field is snapshot wording: %d",
            ladder_snapshot_skips,
        )
    if ladder_no_deadline_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deciding "
            "field carries no recognised deadline wording or no comparable "
            "date: %d",
            ladder_no_deadline_skips,
        )
    if ladder_same_deadline_skips:
        logging.info(
            "Same-event ladder candidates refused because the two rungs' "
            "deciding fields state the same deadline, or truncate to one: %d",
            ladder_same_deadline_skips,
        )
    if ladder_undated_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deadline "
            "states no placeable calendar day (year-less wording, mostly — "
            "see scanner._span_deadline): %d",
            ladder_undated_skips,
        )
    if ladder_field_conflict_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's own "
            "wording fields name irreconcilable days (stated_deadline's "
            "cross-check): %d",
            ladder_field_conflict_skips,
        )
    if ladder_same_day_skips:
        logging.info(
            "Same-event ladder candidates refused because both rungs name "
            "one calendar day (one deadline spelled two ways): %d",
            ladder_same_day_skips,
        )
    if ladder_gap_cap_skips:
        logging.info(
            "Same-event ladder candidates worded as two different cumulative "
            "deadlines, refused at the %d-day STATED gap cap (tier and price "
            "not evaluated): %d",
            MAX_DEADLINE_GAP_DAYS,
            ladder_gap_cap_skips,
        )
    if ladder_price_sum_skips:
        logging.info(
            "Same-event ladder candidates skipped for a leg price sum at or "
            "above $1: %d",
            ladder_price_sum_skips,
        )
    if TIME_SERIES_SAME_EVENT_LADDERS:
        # ALWAYS logged while the switch is on, zero included: a switch that
        # silently produces nothing must be distinguishable from one that is
        # working and finding nothing (DR-66). Counted off the RETURNED pairs,
        # after the one-best-per-group contest, so it says how many ladders
        # actually reach the sizer rather than how many were proposed.
        logging.info(
            "Same-event ladder pairs among the time-series pairs: %d",
            sum(1 for p in candidate_pairs if type(p.stated_gap_days) is int),
        )

    logging.info(
        "Time-series pairs: %d total, %d tradeable",
        len(candidate_pairs),
        sum(1 for p in candidate_pairs if p.tradeable),
    )
    return candidate_pairs


def find_same_title_pairs(
    markets: list,
    held_tickers: set | None = None,
) -> list:
    """
    Find pairs of markets with *exactly* the same title (no normalization),
    where the YES ask price differs by >= SAME_TITLE_MIN_PRICE_DIFF (5%).

    Both markets should resolve identically (same question), so buying NO on the
    expensive market and YES on the cheap market pays on at least one leg
    whenever they co-resolve — a profit when nA+pB<1 (a near-arbitrage priced
    on the SAME_TITLE_CO_RESOLVE_PROB prior). The legs are NO on market_a at nA
    and YES on market_b at pB; nB is populated fail-soft for reporting only.

    The two event tickers must belong to DIFFERENT series (_same_series). Two
    events of one series are two instances of one recurring fixture — two ball
    games, two 15-minute price windows, two MVE combos of different games — so
    identical wording across them is one question asked about two different
    events and the co-resolution prior does not apply at all: the 2026-09-15
    sweep's own NPB pair was quoted 0.97 and 0.01 (DR-02, DR-54). Such
    candidates are skipped and counted, and the count is reported once at the
    end as an INFO line. So are candidates whose two markets carry the SAME
    event ticker (M10), on a line of their own: before M10 that skip was the
    one refusal ahead of the price filters here with no count (the price and
    5% gate `continue`s are still uncounted). Both lines are silent at zero,
    and backtester._extract_pairs logs both verbatim for its same-title
    branch.

    Grouping key is (event_title, title, subtitle). The event_title component is
    what prevents cross-event option-label collisions in MVE markets — e.g. two
    markets both titled "Trump" in unrelated events will have different event
    titles and therefore won't be grouped together. The subtitle component is
    sourced from the API's `yes_sub_title` field post-2026-08 drift (see
    `_market_from_dict`) — it is the only intra-title discriminator, so without
    it two DIFFERENT outcomes sharing one question title (e.g. two candidates
    under "Who will the next Pope be?") on different event tickers would be
    falsely paired as the same contract under the 95% co-resolution assumption.

    Filters: different event_ticker (to exclude multi-choice options), different
    event SERIES (to exclude two instances of one recurring fixture), both
    actively priced (1%-99%), not in held_tickers. One best pair per title
    group.

    Args:
        markets (list): ApiMarket objects to scan (already fetched by the
            caller — unlike find_time_series_pairs, this function never
            fetches).
        held_tickers (set | None): Tickers to exclude (currently-held
            positions). None or empty means exclude nothing.

    Returns:
        list: CandidatePair objects, one per (event_title, title, subtitle)
            group that produced a pair, each carrying pair_type="same_title".
            Empty if no group has two markets on different event_tickers of
            different event series.
    """
    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    # warn_missing_close=False: both run modes call find_time_series_pairs on
    # this SAME list immediately before this call (main._run_dev, _run_prod),
    # and it has already emitted the summary WARNING. One line per run, as
    # CLAUDE.md specifies — not one per finder. The markets are still dropped
    # here either way; only the report is suppressed (TS-22).
    active = _filter_active_markets(markets, held_tickers, warn_missing_close=False)

    # Group by exact (event_title, title, subtitle) tuple — no normalization.
    # Three-element key: event_title prevents MVE cross-event collisions; the
    # (title, subtitle) pair distinguishes markets within an event.
    by_terms: dict = defaultdict(list)
    for m in active:
        event_title = getattr(m, "_event_title", "") or ""
        title    = m.title or ""
        subtitle = m.subtitle or ""
        if title or subtitle:
            by_terms[(event_title, title, subtitle)].append(m)

    candidate_pairs: list = []
    # Counted, not logged per pair. The rejection below fires per CANDIDATE
    # pair — before the price parse and the 5% gate, so more often than the
    # one-best-pair-per-group rows a run finally reports, of which the
    # 2026-09-15 prod dry run had 71 of 75 pairing two DIFFERENT deadlines,
    # i.e. two instances of one recurring fixture. One line each would bury the
    # run's real output; one summary INFO after the loop instead.
    series_skips = 0
    # Same event ticker on both markets (M10) — counted for the same reason,
    # with the same one-summary-line idiom: a refusal nothing reports cannot
    # be told apart, in the log, from a grouping that never formed a group.
    same_event_skips = 0
    # members = all active markets that share this exact (event_title, title, subtitle)
    # key. Each entry is a separate market object from a different event — any two of
    # them are candidates for a same-title pair if their prices diverge.
    for (_event_title, title, subtitle), members in by_terms.items():
        # Use whichever of title or subtitle is non-empty as the display label
        raw_title = title or subtitle
        # Need at least two markets in a group to form any pair
        if len(members) < 2:
            continue

        group_pairs: list = []
        for i, m_outer in enumerate(members):
            for m_inner in members[i + 1:]:
                # Same event_ticker means these are options in the same multi-choice event,
                # not separate markets asking the same question — skip them
                if m_outer.event_ticker == m_inner.event_ticker:
                    same_event_skips += 1
                    continue

                # Same wording, same series, different event: two instances of
                # one recurring fixture (NPB game days, table-tennis matches,
                # 15-minute price windows, MVE combos of different games) —
                # two questions about two events, not one question listed
                # twice, so the SAME_TITLE_CO_RESOLVE_PROB prior does not apply
                # (DR-02, DR-54). The group key already guarantees the wording
                # is identical, so the series test is the whole rule here.
                if _same_series(m_outer, m_inner):
                    series_skips += 1
                    continue

                try:
                    p_outer = float(m_outer.yes_ask_dollars)
                    p_inner = float(m_inner.yes_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # Canonicalize so market_a is always the more expensive side (higher YES ask).
                # Use fresh locals per iteration so the swap does not leak into the next mB.
                if p_outer >= p_inner:
                    mA, mB, pA, pB = m_outer, m_inner, p_outer, p_inner
                else:
                    mA, mB, pA, pB = m_inner, m_outer, p_inner, p_outer

                # Enforce the minimum 5% YES price difference for same-title pairs.
                # A smaller gap is within normal bid-ask spread noise.
                # Same float-noise tolerance as the time-series tier test:
                # 0.35 - 0.30 == 0.04999999999999999, which a bare < rejects
                # at the documented 5% threshold (TS-09).
                if pA - pB < SAME_TITLE_MIN_PRICE_DIFF - PRICE_EPSILON:
                    continue

                try:
                    nA = float(mA.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # nB is reporting-only for a same-title pair (never priced or
                # submitted), so an unparseable value must not cost a candidate:
                # fail-soft to the dataclass default instead of `continue`.
                try:
                    nB = float(mB.no_ask_dollars)
                except (ValueError, TypeError, AttributeError):
                    nB = 0.0

                # tradeable=True when buying NO on A and YES on B covers all costs.
                # fee_per_pair_approx returns a continuous estimate of total taker fees.
                tradeable = (1.0 - nA - pB) > fee_per_pair_approx(nA, pB)
                group_pairs.append(
                    CandidatePair(
                        market_a=mA,
                        market_b=mB,
                        pA=pA,
                        pB=pB,
                        nA=nA,
                        tradeable=tradeable,
                        canonical_title=raw_title,
                        pair_type="same_title",
                        nB=nB,
                    )
                )

        if not group_pairs:
            continue

        # Keep only the best pair per exact title group — largest price gap among
        # tradeable pairs wins; tradeable is preferred over non-tradeable
        group_pairs.sort(key=lambda p: (p.tradeable, p.pA - p.pB), reverse=True)
        candidate_pairs.append(group_pairs[0])

    if same_event_skips:
        logging.info(
            "Same-title candidates skipped because both markets carry the "
            "same event ticker (one event's own markets, not one question "
            "listed by two events): %d",
            same_event_skips,
        )
    if series_skips:
        logging.info(
            "Same-title candidates skipped as two instances of one event series "
            "(identical wording, different fixture): %d", series_skips,
        )

    logging.info(
        "Same-title pairs: %d total, %d tradeable",
        len(candidate_pairs),
        sum(1 for p in candidate_pairs if p.tradeable),
    )
    return candidate_pairs


# ─── Order-book depth pricing ─────────────────────────────────────────────────

def _pair_orderbooks(
    no_levels: list[tuple[float, float]],
    yes_levels: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """
    Merge-pair the NO leg's ask levels with the YES leg's ask levels.

    Both lists sorted ascending by price. Two-pointer sweep: at each step take
    min(remaining_no, remaining_yes) contracts and emit (yes_price, no_price, qty).
    Contracts left over in one book with no counterpart in the other are dropped.
    Which market each side comes from is the caller's concern (see
    _leg_ask_levels) — this sweep only knows "the NO leg" and "the YES leg".

    Args:
        no_levels (list[tuple[float, float]]): (price, quantity) levels for
            the NO leg's NO ask (whichever market that leg is on), ascending
            by price.
        yes_levels (list[tuple[float, float]]): (price, quantity) levels for
            the YES leg's YES ask, ascending by price.

    Returns:
        list[tuple[float, float, float]]: (yes_price, no_price, qty) tuples,
            one per matched quantity slice. Empty if either input is empty.
    """
    pairs: list[tuple[float, float, float]] = []
    i, j = 0, 0
    # Default to 0.0 when empty so the while loop below never indexes into it
    rem_no  = no_levels[i][1]  if no_levels  else 0.0
    rem_yes = yes_levels[j][1] if yes_levels else 0.0

    while i < len(no_levels) and j < len(yes_levels):
        qty = min(rem_no, rem_yes)  # contracts matchable at these two price levels
        pairs.append((yes_levels[j][0], no_levels[i][0], qty))
        rem_no  -= qty
        rem_yes -= qty
        if rem_no == 0:
            i += 1  # NO level exhausted, advance to next cheapest
            if i < len(no_levels):
                rem_no = no_levels[i][1]
        if rem_yes == 0:
            j += 1  # YES level exhausted, advance to next cheapest
            if j < len(yes_levels):
                rem_yes = yes_levels[j][1]

    return pairs


def prefix_fill_prices(
    levels: Sequence[tuple[float, float, float]], n: int,
) -> tuple[float, float] | None:
    """
    Quantity-weighted average fill price of the FIRST n contracts of a book.

    The single definition of "what would n contract pairs actually cost", shared
    by enrich_with_orderbook_prices (which prices a pair at the most contracts
    the budget could ever buy) and strategy.compute_trade (which prices the
    exact n it sizes). Averaging the WHOLE qualifying book instead — what this
    replaced — priced every pair against depth no single trade could reach,
    which both inflated the fill price and killed pairs at the profitability
    gate on levels they would never have touched.

    Levels are consumed cheapest-first, taking min(remaining, qty) at each, so
    the result is exactly the volume-weighted price of a marketable order for n
    contracts. Because levels ascend by combined price, the returned sum is
    non-decreasing in n: a larger n can only reach further down the book into
    worse-priced levels.

    Args:
        levels (Sequence[tuple[float, float, float]]): The pair's qualifying
            depth as (price_a, price_b, qty) in MARKET order, ascending by
            combined price — CandidatePair.depth_levels, or the freshly
            oriented levels enrichment is about to store there.
        n (int): Whole contract pairs to price. Range: >= 1.

    Returns:
        tuple[float, float] | None: (avg price on market_a, avg price on
            market_b) in dollars. None when n < 1, or when the levels hold
            fewer than n contracts in total — the caller decides whether that
            is a dropped pair or a smaller size.
    """
    if n < 1:
        return None
    remaining = float(n)
    sum_a = 0.0
    sum_b = 0.0
    for price_a, price_b, qty in levels:
        take = min(remaining, qty)
        if take <= 0:
            # A zero/negative level cannot contribute; _bids_to_ask_levels
            # already drops these, so this only guards hand-built input
            continue
        sum_a += price_a * take
        sum_b += price_b * take
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        # Fewer than n contracts available. No epsilon is needed: the final
        # take is exactly `remaining` whenever a level can cover it, so
        # remaining reaches exactly 0.0 on every sufficient book.
        return None
    return sum_a / n, sum_b / n


def _pair_ticker(pair: Any, attr: str) -> str:
    """
    Read one of a pair's market tickers without assuming the market is there.

    Used only to name a market in a log line. _leg_ask_levels and
    _reference_yes_ask are deliberately tolerant of a bare pair stub (the same
    fail-safe rule leg_sides applies to an unknown pair_type), so a missing
    market must degrade to a placeholder rather than raise inside the scan.

    Args:
        pair (Any): A CandidatePair or any stub exposing market_a/market_b.
        attr (str): "market_a" or "market_b".

    Returns:
        str: The market's ticker, or "<unknown>" when it cannot be read.
    """
    return getattr(getattr(pair, attr, None), "ticker", "<unknown>")


def _bids_to_ask_levels(bids_raw: list, ticker: str = "<unknown>") -> list[tuple[float, float]]:
    """
    Convert bid levels to ask levels via the complement price (1 − P).

    Applies to both sides: YES bid at P → NO ask at (1−P);
                           NO bid at P → YES ask at (1−P).
    Descending bids naturally yield ascending asks after the complement.

    Bounded by config.MIN/MAX_ACTIVE_PRICE_DOLLARS (0.0001/0.9999), the extreme
    tradeable levels on Kalshi's FINEST grid — NOT by the 0.01/0.99
    market-eligibility bound. Those are different questions, and using the
    coarse one here silently discarded real depth on exactly the regimes whose
    point is sub-cent ticks: every level of a deci-cent or centi-cent book
    priced under a cent, or over 99c, vanished before pairing (TS-14).

    Args:
        bids_raw (list): [[price_str, qty_str], ...] sorted descending by
            price, as parsed from the orderbook payload.
        ticker (str): The market the book came from, named in the drop
            WARNING only. Defaults to a placeholder for hand-built input.

    Returns:
        list[tuple[float, float]]: [(ask_price, qty), ...] sorted ascending
            (cheapest ask first). A level whose complement price falls outside
            [MIN_ACTIVE_PRICE_DOLLARS, MAX_ACTIVE_PRICE_DOLLARS] or whose qty
            is <= 0 is dropped, as is a malformed entry (bad price/qty string);
            one summary WARNING names the total, silent at zero.
    """
    levels = []
    dropped = 0
    for entry in bids_raw:
        try:
            bid_price = float(entry[0])
            qty = float(entry[1])
            ask_price = 1.0 - bid_price
            if MIN_ACTIVE_PRICE_DOLLARS <= ask_price <= MAX_ACTIVE_PRICE_DOLLARS and qty > 0:
                levels.append((ask_price, qty))
            else:
                dropped += 1
        except (ValueError, TypeError, IndexError):
            dropped += 1
            continue
    if dropped:
        # Same silent-at-zero summary idiom as the trading-inactive shard skip
        # count. These drops used to be entirely invisible, so a book thinned
        # by a drifted payload read downstream as genuinely thin depth (TS-14).
        logging.warning(
            "Orderbook for %s: dropped %d of %d bid levels as unusable "
            "(complement outside [%s, %s], nonpositive qty, or unparseable)",
            ticker, dropped, len(bids_raw),
            MIN_ACTIVE_PRICE_DOLLARS, MAX_ACTIVE_PRICE_DOLLARS,
        )
    levels.sort(key=lambda x: x[0])
    return levels


def _leg_ask_levels(
    pair: Any,
    ob_a: dict,
    ob_b: dict,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Derive the (NO leg, YES leg) ask levels a pair would consume from its two books.

    Buying side S on a market consumes that market's OPPOSITE-side resting
    bids: a NO buy fills against the YES bids (NO ask = 1 - YES bid) and a YES
    buy fills against the NO bids (YES ask = 1 - NO bid), both via
    _bids_to_ask_levels. Which market carries which leg comes from
    leg_sides(pair.pair_type), so this is the single place the book sides are
    chosen for enrich_with_orderbook_prices and validate_pair_price:

      same_title  (NO on A, YES on B): NO asks from A's YES bids, YES asks
                                       from B's NO bids — today's behaviour.
      time_series (YES on A, NO on B): NO asks from B's YES bids, YES asks
                                       from A's NO bids.

    Args:
        pair (Any): CandidatePair (or anything exposing pair_type) whose legs
            are being priced.
        ob_a (dict): market_a's parsed order book from _fetch_orderbook —
            {"yes": [[price, qty], ...], "no": [...]} (bids, dollar strings).
        ob_b (dict): market_b's parsed order book, same shape.

    Returns:
        tuple[list, list]: (no_levels, yes_levels), each an ascending
            [(ask_price, qty), ...] list as produced by _bids_to_ask_levels —
            exactly the two arguments _pair_orderbooks takes. Either may be
            empty when the relevant side has no resting bids.
    """
    # Map each market to the side bought there; the only source of truth for
    # which market carries the NO leg is config's side tuples via leg_sides
    side_a, _side_b = leg_sides(getattr(pair, "pair_type", None))
    if side_a == "no":
        # NO on A consumes A's YES bids; YES on B consumes B's NO bids
        return (_bids_to_ask_levels(ob_a["yes"], _pair_ticker(pair, "market_a")),
                _bids_to_ask_levels(ob_b["no"], _pair_ticker(pair, "market_b")))
    # YES on A consumes A's NO bids; NO on B consumes B's YES bids
    return (_bids_to_ask_levels(ob_b["yes"], _pair_ticker(pair, "market_b")),
            _bids_to_ask_levels(ob_a["no"], _pair_ticker(pair, "market_a")))


def _reference_yes_ask(pair: Any, ob_a: dict, ob_b: dict) -> float | None:
    """
    Best YES ask on the market that does NOT carry the pair's YES leg.

    Each pair type buys YES on one market and NO on the other, so exactly one
    market's YES ask is a LEG price; the OTHER market's YES ask is the model's
    reference quote (pB for time_series, pA for same_title). _leg_ask_levels
    reads only one side of each book, which leaves the side that yields this
    quote fetched but unused — so deriving it here costs no extra request.

    A YES ask is the complement of a resting NO bid, so the reference comes
    from the non-YES-leg market's NO bids via _bids_to_ask_levels.

    Args:
        pair (Any): CandidatePair (or anything exposing pair_type); only
            pair_type is read, via leg_sides, so an unknown type resolves to
            the same-title sides exactly as leg_sides specifies.
        ob_a (dict): market_a's parsed book from _fetch_orderbook —
            {"yes": [[price, qty], ...], "no": [...]} (bids, dollar strings).
        ob_b (dict): market_b's parsed book, same shape.

    Returns:
        float | None: The lowest YES ask on the non-YES-leg market, or None
            when that side carries no usable resting bids (the caller then
            keeps the pair's scan-time value).
    """
    # leg_sides is the only source of truth for which market carries the YES
    # leg: YES on A for time_series, YES on B for same_title
    side_a, _side_b = leg_sides(getattr(pair, "pair_type", None))
    ob_ref = ob_b if side_a == "yes" else ob_a
    ref_ticker = _pair_ticker(pair, "market_b" if side_a == "yes" else "market_a")
    levels = _bids_to_ask_levels(ob_ref["no"], ref_ticker)
    # _bids_to_ask_levels returns ASCENDING asks, so [0] is the best (lowest)
    return levels[0][0] if levels else None


# Unit tags for an orderbook side-key candidate set. Dollar sets carry bid prices
# as 0-1 dollar strings (parsed straight by _bids_to_ask_levels); cents sets carry
# integer cents and MUST be converted to dollars before that parser sees them.
_UNIT_DOLLARS = "dollars"
_UNIT_CENTS = "cents"

# Valid inclusive bounds for an integer-cent bid price on the legacy arrays. A
# level outside this range is not a price Kalshi can quote (0 and 100 are the
# settled extremes), so it signals a unit/shape drift rather than a real bid.
_MIN_BID_CENTS = 1
_MAX_BID_CENTS = 99

# Matched wire-format generations for the orderbook response: each container key
# owns an ordered tuple of candidate (yes_key, no_key, unit) side-key sets, and a
# container is NEVER read with another container's sets. Verified against the
# pinned SDK's models/orderbook.py:
#   * `orderbook_fp` — live-observed only (absent from the SDK entirely); serves
#     `yes_dollars`/`no_dollars` dollar-string bid arrays. This is what production
#     receives today, so its handling must stay behaviour-identical.
#   * `orderbook` — the SDK-modeled container. Its model REQUIRES
#     `yes_dollars`/`no_dollars` (dollar strings) and aliases the legacy integer-cent
#     arrays as `"true"`/`"false"` — never `"yes"`/`"no"`, which this table wrongly
#     assumed before. Both shapes are accepted, dollars first, cents second.
# Resolving container and side keys independently (a plain `or` across both
# generations) risked cross-reading a cents array through the dollars parser, so
# the pairing is enforced explicitly in _fetch_orderbook. Order matters within a
# container too: the first candidate set whose BOTH keys are present wins.
#
# CONTEXT: Kalshi removed the legacy integer price fields from its REST/WS
# payloads on 2026-03-12, and production serves `orderbook_fp` only — so the
# `orderbook` container and its cents arrays are not expected on the wire today.
# They are kept as SDK-shape-derived DEFENSE, not as a live code path: the pinned
# SDK still models exactly those shapes, sandbox and future replays may serve
# them, and the alternative to parsing them correctly is not "no code" but a
# silently empty book (BS-12). Nothing here weakens the fail-closed rule — an
# unknown container, or a known container matching none of its own candidate
# sets, still returns None with a mismatch warning.
_ORDERBOOK_SIDE_KEYS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "orderbook_fp": (
        ("yes_dollars", "no_dollars", _UNIT_DOLLARS),
    ),
    "orderbook": (
        ("yes_dollars", "no_dollars", _UNIT_DOLLARS),
        ("true", "false", _UNIT_CENTS),
    ),
}


def _coerce_int_cents(value: Any) -> int | None:
    """
    Coerce one raw legacy-array bid price to an integer number of cents.

    Accepts a genuine int, an integral float (45.0), or a string spelling either
    ("45", "45.0"). Anything fractional, non-numeric, or boolean is rejected —
    a fractional "cent" means the array is really carrying dollars and must not
    be multiplied through the cents path.

    Args:
        value (Any): Raw price element from a legacy `true`/`false` bid level.

    Returns:
        int | None: The value as whole cents, or None if it is not an integral
            number (which the caller treats as a malformed level).
    """
    # bool is an int subclass; True would silently become 1 cent
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            num = float(value.strip())
        except ValueError:
            return None
        return int(num) if num.is_integer() else None
    return None


def _cents_bids_to_dollar_bids(ticker: str, side_key: str, bids_raw: list) -> list:
    """
    Convert one legacy integer-cent bid array to the dollar form the parser expects.

    _bids_to_ask_levels is dollars-only by contract: it computes 1 - price, so a
    cents value fed to it straight yields a wildly negative ask that its own
    range check silently discards — a full cents book would parse as an empty
    book with no warning at all. This converts cents to dollars first and drops
    (loudly) any level whose price is not a whole cent in [1, 99], so a unit
    drift surfaces as a warning naming the offending value instead of silence.

    Args:
        ticker (str): Market ticker, for the drop warning.
        side_key (str): The side key being converted ("true"/"false"), for the
            drop warning.
        bids_raw (list): Raw bid levels, each expected as [price_cents, qty].

    Returns:
        list: [[price_dollars, qty], ...] in the original order, with malformed
            levels omitted. Returns [] when every level was malformed.
    """
    converted: list = []
    for entry in bids_raw:
        try:
            raw_price = entry[0]
            qty = entry[1]
        except (TypeError, IndexError, KeyError):
            logging.warning(
                "Dropping malformed legacy cents array level for %s side '%s': %r "
                "(level is not a [price, qty] pair)",
                ticker,
                side_key,
                entry,
            )
            continue
        cents = _coerce_int_cents(raw_price)
        if cents is None or not (_MIN_BID_CENTS <= cents <= _MAX_BID_CENTS):
            logging.warning(
                "Dropping malformed legacy cents array level for %s side '%s': price %r "
                "is not a whole cent in [%d, %d]",
                ticker,
                side_key,
                raw_price,
                _MIN_BID_CENTS,
                _MAX_BID_CENTS,
            )
            continue
        converted.append([cents / 100.0, qty])
    return converted


def _fetch_orderbook(client: Any, ticker: str) -> dict | None:
    """
    Fetch the order book for a market.

    The book arrives under one of two matched key generations (see
    _ORDERBOOK_SIDE_KEYS). The container key strictly selects which side-key sets
    are eligible — generations are never mixed:

      * `orderbook_fp` (what production receives today) → `yes_dollars`/`no_dollars`
        dollar-string bid arrays.
      * `orderbook` (the SDK-modeled container) → `yes_dollars`/`no_dollars` dollar
        strings if present, else the legacy `true`/`false` INTEGER-CENT arrays,
        which are converted to dollars here before any parsing.

    A container present but null or empty (`{}`) is an empty book — a market with
    no resting bids, not a shape change. A container that is a non-empty dict but
    matches none of its own candidate side-key sets (or is some other non-dict
    value) signals a Kalshi API shape change and fails closed with a "potential
    orderbook key mismatch" warning, rather than cross-reading another
    generation's keys (which could feed a cents array through the dollars parser).

    Args:
        client (Any): Authenticated KalshiClient exposing the raw-response
            get_market_orderbook_without_preload_content variant.
        ticker (str): Market ticker whose order book to fetch.

    Returns:
        dict | None: {'yes': [[price, qty], ...], 'no': [...]} where 'yes' is YES
            bids and 'no' is NO bids and every price is in DOLLARS (dollar-string
            levels pass through untouched; cents levels are converted). Returns
            None when no recognized container key is present, or when the selected
            container matches none of its candidate side-key sets (a potential key
            mismatch — logged as such).
    """
    try:
        # Raw-response call: the live API returns only the orderbook_fp key,
        # which the pinned SDK's modeled response (orderbook required) rejects
        data = api_call_with_retry(
            fetch_json_page, client.get_market_orderbook_without_preload_content, ticker=ticker
        )
        # Select the container by key PRESENCE (not truthiness) so its matched
        # side keys can be enforced below — an empty book still selects its own
        # container instead of falling through to the other generation.
        container_key = next((k for k in _ORDERBOOK_SIDE_KEYS if k in data), None)
        if container_key is None:
            # No recognized container key at all. Log the keys that WERE present:
            # if the API renames the orderbook key again (as it did when
            # orderbook -> orderbook_fp), every book resolves to None and every
            # pair is silently marked non-tradeable — a 0-trade run with nothing
            # but generic "unavailable" warnings. Naming the actual keys makes
            # that drift diagnosable instead of looking like an empty book.
            logging.warning(
                "No usable orderbook for %s (response keys: %s) — treating as unavailable",
                ticker,
                sorted(data.keys()),
            )
            return None
        ob = data[container_key]
        # The container->side-key mapping below is validated against the pinned
        # SDK's documented Orderbook shapes (models/orderbook.py) by
        # TestFetchOrderbookKeyMapping in tests/test_scanner.py, which exercises
        # every generation, unit, and degenerate container this branch handles.
        if ob is None or ob == {}:
            # Container present but carries no book at all. That is a market with
            # no resting bids on either side, NOT a shape change — return an empty
            # book so the mismatch warning stays reserved for genuine drift.
            return {"yes": [], "no": []}
        selected = None
        if isinstance(ob, dict):
            # First candidate set whose BOTH keys are present wins; a set is only
            # ever taken whole, so units can never be inferred from the wrong keys.
            selected = next(
                (
                    keys
                    for keys in _ORDERBOOK_SIDE_KEYS[container_key]
                    if keys[0] in ob and keys[1] in ob
                ),
                None,
            )
        if selected is None:
            # Container present and non-empty but matches none of ITS OWN candidate
            # side-key sets (or is not even a dict). This is a potential key
            # mismatch — the container and side keys have drifted out of sync, or
            # Kalshi changed the response shape. Fail closed instead of
            # cross-reading another generation's keys (which could feed a cents
            # array through the dollars parser).
            logging.warning(
                "Potential orderbook key mismatch for %s: container '%s' present but matches "
                "none of its expected side-key sets %s (got %s) — check for bugs or changes "
                "to Kalshi's API",
                ticker,
                container_key,
                [(y, n) for y, n, _ in _ORDERBOOK_SIDE_KEYS[container_key]],
                sorted(ob.keys()) if isinstance(ob, dict) else type(ob).__name__,
            )
            return None
        yes_key, no_key, unit = selected
        # Matched side keys are always present; a side with no resting bids
        # arrives as null or [] and must read as an empty (not missing) side.
        yes_raw = list(ob[yes_key] or [])
        no_raw  = list(ob[no_key] or [])
        if unit == _UNIT_CENTS:
            # Convert to dollars HERE — _bids_to_ask_levels is dollars-only, and
            # cents fed through it parse as an empty book with no warning at all
            yes_raw = _cents_bids_to_dollar_bids(ticker, yes_key, yes_raw)
            no_raw  = _cents_bids_to_dollar_bids(ticker, no_key, no_raw)
        return {"yes": yes_raw, "no": no_raw}
    except Exception as exc:
        logging.warning("Orderbook fetch failed for %s: %s", ticker, exc)
        return None


def _pair_max_sum(pair: Any) -> float:
    """
    Return the maximum allowed yes_price + no_price sum for one pair's orderbook
    depth levels — the complement of the pair's minimum price-gap threshold.

    The sum is over the two LEG prices (the YES leg's YES ask plus the NO leg's
    NO ask, whichever markets those sit on). same_title pairs use the flat
    SAME_TITLE_MIN_PRICE_DIFF threshold (sum <= 1 - SAME_TITLE_MIN_PRICE_DIFF).
    time_series pairs use the deadline-gap-tiered threshold from
    min_price_diff_for_gap() (sum <= 1 - MIN_PRICE_DIFF_SHORT_GAP when the
    deadlines are <= SHORT_DEADLINE_GAP_DAYS apart, sum <= 1 -
    MIN_PRICE_DIFF_LONG_GAP for wider gaps up to MAX_DEADLINE_GAP_DAYS). The
    gap comes from pair_gap_days(), which is order-independent, so the
    ceiling does not depend on which leg closes first — and which returns the
    pair's STATED deadline gap for a same-event ladder (DR-73), whose rungs
    can share a close_time entirely.

    Args:
        pair (CandidatePair): The pair whose ceiling is needed.

    Returns:
        float: Maximum qualifying yes_price + no_price sum (dollars, 0-1).
    """
    if pair.pair_type == "time_series":
        # Tier the ceiling by the same deadline gap used at candidate
        # detection — the STATED gap for a same-event ladder, close_time
        # otherwise (DR-73). Re-deriving it from close_time here would
        # re-tier a ladder the finder already tiered.
        gap_days = pair_gap_days(pair)
        return 1.0 - min_price_diff_for_gap(gap_days)
    return 1.0 - SAME_TITLE_MIN_PRICE_DIFF


def enrich_with_orderbook_prices(
    client: Any, pairs: list, balance_cents: int,
) -> list:
    """
    For each tradeable pair, fetch both order books, pair the NO leg's asks
    with the YES leg's asks using a merge sweep (the legs' markets and sides
    come from _leg_ask_levels), then filter to only contract pairs whose
    combined LEG price meets the pair's gap threshold (see _pair_max_sum):

      same_title:  yes_price + no_price <= 1 - SAME_TITLE_MIN_PRICE_DIFF
      time_series: yes_price + no_price <= 1 - min_price_diff_for_gap(gap)

    The two leg prices are replaced with weighted-average fill prices over the
    contracts this account could actually BUY — not over the whole qualifying
    book. One pair is capped at BUDGET_FRACTION of the balance, so averaging a
    liquid market's full depth priced every pair against levels no single trade
    can reach: it inflated the fill price and killed pairs at the profitability
    gate below on contracts we would never have bought. The bound is
    config.max_affordable_pairs(balance_cents, best level's price sum) — the
    maximum fraction over the minimum price sum, so it is an upper bound on
    whatever strategy.compute_trade sizes, and the price written here can never
    be optimistic relative to the one that trade is finally priced at. The
    qualifying levels themselves are kept on the pair (depth_levels) so
    compute_trade can re-price at the exact n it settles on. Prices are written
    back to nA/pB for a same-title pair and to pA/nB for a time-series pair. The pair's REFERENCE quote (the non-leg
    market's YES ask: pB for time_series, pA for same_title) is refreshed from
    the same books via _reference_yes_ask, so downstream models never subtract
    a scan-time quote from a depth-weighted one; the remaining quote (nA for
    time_series, nB for same_title) is reporting-only and stays untouched.
    max_contracts is set to that affordability-bounded count, i.e. how many
    contracts the written prices are valid for. Pairs with no qualifying
    contracts are marked tradeable=False, as are pairs whose refreshed
    reference no longer sits above the YES leg's fill, and pairs the budget
    cannot afford a single contract of.

    Args:
        client (Any): Authenticated KalshiClient used to fetch each pair's
            order books (cached per ticker across the whole call).
        pairs (list): CandidatePair objects to enrich. A pair already marked
            tradeable=False is passed through unchanged.
        balance_cents (int): Account balance in integer cents — the real
            per-shard sum in prod, the virtual --sandbox-balance in dev. Bounds
            how much book depth is averaged into each pair's fill price.
            Required rather than defaulted: both call sites already hold it,
            and a default would silently restore whole-book pricing on a
            real-money path with no signal that it had.

    Returns:
        list: One CandidatePair per input pair, in the same order, with the
            leg prices (nA/pB for same_title, pA/nB for time_series), the
            reference quote (pB for time_series, pA for same_title, refreshed
            only when the reference book side had resting bids), tradeable,
            max_contracts and depth_levels replaced by depth-validated values.
    """
    # Cache order books by ticker to avoid fetching the same book twice
    # when the same market appears in multiple pairs
    ob_cache: dict[str, dict | None] = {}

    def get_ob(ticker: str) -> dict | None:
        if ticker not in ob_cache:
            # Fetch the order book from the Kalshi API and cache the result
            ob_cache[ticker] = _fetch_orderbook(client, ticker)
        return ob_cache[ticker]

    enriched = []
    for pair in pairs:
        # Non-tradeable pairs (failed best-ask check) are passed through unchanged —
        # they still appear in the dev simulation Excel sheet for transparency
        if not pair.tradeable:
            enriched.append(pair)
            continue

        ob_a = get_ob(pair.market_a.ticker)
        ob_b = get_ob(pair.market_b.ticker)

        if ob_a is None or ob_b is None:
            # If either order book is unavailable we cannot validate depth, so
            # mark the pair non-tradeable. Passing best-ask prices through with
            # tradeable=True would let strategy.compute_trade size against unbounded
            # depth (max_contracts=0 is treated as "no cap" downstream) — the
            # opposite of what pre_execution_check is meant to catch.
            logging.warning(
                "Orderbook unavailable for '%s' — marking non-tradeable",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Derive the ask levels each leg would consume — each buy fills against
        # the OPPOSITE-side bids of ITS OWN market, and which market carries the
        # NO leg depends on the pair type (see _leg_ask_levels)
        no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)

        # Merge-pair the NO and YES depth levels into (yes_price, no_price, qty) tuples
        paired     = _pair_orderbooks(no_levels, yes_levels)

        # Keep only contract pairs where the combined fill price leaves the required
        # gap: same_title requires >= SAME_TITLE_MIN_PRICE_DIFF; time_series requires
        # the deadline-gap-tiered threshold from min_price_diff_for_gap() — see
        # _pair_max_sum for the exact per-tier ceilings.
        max_sum    = _pair_max_sum(pair)
        qualifying = [
            (yp, np_, qty)
            for yp, np_, qty in paired
            if yp + np_ <= max_sum + PRICE_EPSILON
        ]

        if not qualifying:
            # No depth available at the required gap — mark untradeable to skip execution
            logging.info(
                "No qualifying contract pairs for '%s' after price gap filter — skipping",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Orient the levels into MARKET order — (market_a's leg price,
        # market_b's leg price, qty) — so everything downstream reads them the
        # way leg_prices reads the pair's scalars. leg_sides is the only source
        # of truth for which market buys which side.
        side_a, _side_b = leg_sides(pair.pair_type)
        a_is_no = side_a == "no"
        depth_levels = tuple(
            (np_, yp, q) if a_is_no else (yp, np_, q) for yp, np_, q in qualifying
        )

        total_qty = sum(qty for _, _, qty in qualifying)
        # Bound the average at the most contracts any Kelly result could ever
        # afford, rather than averaging the whole book. Levels ascend by
        # combined price, so BUDGET_FRACTION (the largest fraction compute_trade
        # can cap to) over the BEST level's sum (the cheapest any prefix average
        # can be) is an upper bound on the n that trade finally sizes — the
        # price written here is therefore never optimistic relative to it.
        best_a, best_b, _ = depth_levels[0]
        affordable = max_affordable_pairs(balance_cents, best_a + best_b)
        cap = min(int(total_qty), affordable)
        fills = prefix_fill_prices(depth_levels, cap)

        if fills is None:
            # cap < 1: the budget cannot afford one contract pair, or the book
            # holds under one contract of qualifying depth. Drop the pair rather
            # than write max_contracts=0, which compute_trade reads as UNCAPPED
            # — the sub-one-contract hole that overloaded sentinel used to have.
            # Both figures are named because they are different faults with
            # different fixes (add funds vs. the book is too thin), and the
            # binding one is whichever is smaller.
            logging.info(
                "No affordable contract pairs for '%s' — %.2f contract(s) rest at "
                "the gap and the budget affords %d; skipping",
                pair.canonical_title, total_qty, affordable,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Back to SIDE order for the direction guard, the fee check and the
        # writeback below, all of which speak in "the YES leg"/"the NO leg"
        avg_yes, avg_no = (fills[1], fills[0]) if a_is_no else (fills[0], fills[1])

        # The REFERENCE quote — the non-leg market's YES ask — refreshed from the
        # book already in hand. Left at its scan-time value it would be compared
        # against a fresh avg_yes by strategy._kelly_p, whose subtraction runs
        # through config.time_series_profit_prob's max(0, pB - pA) clamp: a stale
        # pB at or below the fresh pA clamps to zero, returning p = 1.0, so the
        # pair models as RISKLESS and Kelly sizes it at the BUDGET_FRACTION cap.
        ref_yes = _reference_yes_ask(pair, ob_a, ob_b)

        # Direction, re-asserted for TIME_SERIES ONLY. avg_yes is the YES leg's
        # fill and ref_yes the later contract's YES ask, so `ref_yes > avg_yes`
        # is exactly the `pB > pA` conjunct find_time_series_pairs applies at
        # scan time — re-applied now that the fill price has moved.
        #
        # same_title is deliberately NOT guarded here. Its model is the fixed
        # co-resolution prior (strategy._kelly_p), so pA is not a model input and
        # there is no clamp to protect: a guard would buy nothing, while its
        # stale-fallback branch could drop a sound near-arbitrage on exactly the
        # quantity this change exists to distrust.
        #
        # With a refreshed reference the test is implied by the ceiling applied
        # above: every qualifying level satisfies yes + no <= 1 - tier, so
        # avg_yes <= (1 - tier) - avg_no <= max(YES bid on the later market)
        # - tier, and an UNCROSSED book puts that market's YES ask at or above
        # its YES bid. On fresh data it can therefore only fire on a CROSSED book.
        # The affordability bound above does not weaken that: the ceiling holds
        # for EVERY qualifying level individually, so it holds for any PREFIX
        # average of them, and truncating only lowers avg_yes. The guard is no
        # more likely to fire than it was over the whole book.
        is_time_series = leg_sides(pair.pair_type) == TIME_SERIES_LEG_SIDES
        direction_ok = True
        if is_time_series:
            if ref_yes is not None:
                direction_ok = ref_yes > avg_yes
                basis = f"fresh reference ask {ref_yes:.4f}"
            else:
                # No fresh reference (the later market had no resting NO bids),
                # so the only quote available is the scan-time pB — seconds to
                # tens of seconds old. Demand the FULL tier rather than a bare
                # `>`: a mixed-snapshot gap of a thousandth would otherwise pass,
                # and as the gap shrinks time_series_profit_prob rises toward 1.0
                # and Kelly sizes toward the BUDGET_FRACTION cap.
                # pair_gap_days, not deadline_gap_days: a same-event
                # ladder is tiered on its STATED deadline gap (DR-73).
                tier = min_price_diff_for_gap(pair_gap_days(pair))
                direction_ok = (pair.pB - avg_yes) >= tier - PRICE_EPSILON
                basis = (
                    f"scan-time reference ask {pair.pB:.4f} (later book's NO side "
                    f"empty), which must clear the {tier:.2f} tier"
                )
            if not direction_ok:
                logging.warning(
                    "Pair '%s' dropped: the later contract no longer prices above "
                    "the YES leg fill %.4f — %s",
                    pair.canonical_title, avg_yes, basis,
                )

        # Re-validate tradeability at the depth-weighted prices (the pair may still be
        # unprofitable if all qualifying contracts are at the edge of the gap threshold).
        profitable = (1.0 - avg_no - avg_yes) > fee_per_pair_approx(avg_no, avg_yes)

        if not profitable:
            # Kept as its own arm so this line only ever reports the profitability
            # verdict — a direction drop has already logged its own WARNING above
            logging.info(
                "Pair '%s' unprofitable after depth adjustment: avg_no=%.3f avg_yes=%.3f",
                pair.canonical_title, avg_no, avg_yes,
            )

        new_tradeable = profitable and direction_ok

        # Replace the LEG prices and contract count with depth-accurate values,
        # writing back to whichever fields are the leg prices for this pair type
        # (nA/pB for same_title, pA/nB for time_series — the fields
        # leg_prices() reads); strategy.py sizes the final Kelly trade on them
        if leg_sides(pair.pair_type) == TIME_SERIES_LEG_SIDES:
            leg_updates = {"pA": avg_yes, "nB": avg_no}
            # pB is the model's reference quote, not a leg price — refreshed so
            # strategy._kelly_p's pB - pA subtraction has both operands from one
            # snapshot. Left alone when the reference side had no resting bids.
            if ref_yes is not None:
                leg_updates["pB"] = ref_yes
        else:
            leg_updates = {"nA": avg_no, "pB": avg_yes}
            # Mirror: pA is same_title's reference quote. Nothing sizes on it
            # (_kelly_p uses the fixed co-resolution prior here) and it is NOT
            # guarded above, but it is the operand of the reported pA - pB
            # "Price Diff", which otherwise subtracts a scan-time quote from a
            # depth-weighted fill and can print a negative gap for a pair that
            # passed the finder's directional filter. Refreshed for coherence.
            if ref_yes is not None:
                leg_updates["pA"] = ref_yes
        enriched.append(dc_replace(
            pair,
            tradeable=new_tradeable,
            # How many contracts the prices just written are valid for — the
            # bound compute_trade's own depth clamp then honours
            max_contracts=cap,
            depth_levels=depth_levels,
            **leg_updates,
        ))

    tradeable_after = sum(1 for p in enriched if p.tradeable)
    logging.info(
        "Orderbook depth check: %d/%d pairs remain tradeable after price gap filter",
        tradeable_after,
        sum(1 for p in pairs if p.tradeable),
    )
    return enriched


def validate_pair_price(client: Any, spec: Any) -> bool:
    """
    Re-fetch both order books for a TradeSpec immediately before execution and
    confirm the gap threshold still holds at the required contract depth.

    Returns True only if qualifying depth >= spec.x contracts remain at the
    pair's gap threshold. A False result means prices have moved since the
    scan and the trade should be skipped. Every False return is logged here,
    once, at WARNING, with its reason — callers must not log the drop again.

    Args:
        client: Authenticated KalshiClient from auth.build_client().
        spec: TradeSpec whose pair prices should be re-validated.

    Returns:
        bool: True if the pair still qualifies; False if prices moved or order
            books are unavailable.
    """
    pair   = spec.pair
    ob_a   = _fetch_orderbook(client, pair.market_a.ticker)
    ob_b   = _fetch_orderbook(client, pair.market_b.ticker)

    if ob_a is None or ob_b is None:
        logging.warning(
            "Pre-execution orderbook unavailable for '%s' — dropping",
            pair.canonical_title,
        )
        return False

    # Same side selection as enrichment: each leg consumes its own market's
    # opposite-side bids, and the NO leg's market depends on the pair type
    no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)
    paired     = _pair_orderbooks(no_levels, yes_levels)
    # Same per-pair gap ceiling used at scan time (deadline-gap-tiered for
    # time_series) — the trade must still qualify at execution time
    max_sum    = _pair_max_sum(pair)
    qualifying = [
        (yp, np_, qty)
        for yp, np_, qty in paired
        if yp + np_ <= max_sum + PRICE_EPSILON
    ]

    if not qualifying:
        # WARNING, not INFO: this is a SELECTED trade being dropped seconds
        # before submission. This is the single log line for the drop —
        # pre_execution_check deliberately does not log a second one.
        logging.warning(
            "Pre-execution check failed for '%s' — gap no longer qualifies; dropping",
            pair.canonical_title,
        )
        return False

    # Require enough depth to fill our full intended contract count via FoK.
    # On the V2 path "enough depth" means depth the ORDER CAN REACH, not depth
    # that merely clears the gap: the order is one fill-or-kill limit per leg,
    # priced from this spec's own leg prices, and it buys nothing resting above
    # that limit. Counting the whole qualifying book here would let a trade
    # whose top levels sit above its cap pass the pre-execution check and then
    # be killed on the wire, reported as "NO leg FoK not filled" — the same
    # confusion between cap and size that TS-08 fixed on the sizing side.
    #
    # The caps come from leg_prices(spec.pair) — the price the trader is about
    # to submit at — NOT from a freshly recomputed average of this book. That
    # is the question that actually matters: will the order we are about to
    # send fill against the book as it stands now?
    if ORDER_API_VERSION == "v2":
        side_a, side_b = leg_sides(pair.pair_type)
        price_a, price_b = leg_prices(spec.pair)
        cap_a = float(v2_effective_cap(f"buy_{side_a}", price_a, pair.market_a))
        cap_b = float(v2_effective_cap(f"buy_{side_b}", price_b, pair.market_b))
        # qualifying is in SIDE order (yes, no, qty); orient the caps to match
        cap_yes, cap_no = (cap_b, cap_a) if side_a == "no" else (cap_a, cap_b)
        total_qty = sum(
            qty for yp, np_, qty in qualifying
            if yp <= cap_yes + PRICE_EPSILON and np_ <= cap_no + PRICE_EPSILON
        )
        basis = "reachable at the FoK limit"
    else:
        # Legacy buy_max_cost is a TOTAL-cost cap and can sweep a ladder, so
        # every qualifying contract is reachable on that path.
        total_qty = sum(qty for _, _, qty in qualifying)
        basis = "at gap"

    if total_qty < spec.x:
        logging.warning(
            "Pre-execution check failed for '%s' — only %.1f contracts %s (need %d); dropping",
            pair.canonical_title, total_qty, basis, spec.x,
        )
        return False

    return True
