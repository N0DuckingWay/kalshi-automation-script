"""Market scanning: fetch open markets and find arbitrage candidate pairs."""
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import (
    MARKET_PAGE_SIZE,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF,
    POSITION_PAGE_SIZE,
    SAME_TITLE_MIN_PRICE_DIFF,
)

# ---------------------------------------------------------------------------
# Date patterns stripped from titles before exact-match grouping.
# After stripping, two contracts that differ ONLY in their deadline will
# produce the same normalized string — no fuzzy matching needed.
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

# Minimum ask price to consider a market actively priced (not settled/illiquid)
_MIN_ACTIVE_PRICE = 0.01
_MAX_ACTIVE_PRICE = 0.99


@dataclass
class CandidatePair:
    market_a: Any           # Market with higher YES ask (expensive side)
    market_b: Any           # Market with lower YES ask (cheap side)
    pA: float               # yes_ask_dollars of A (cost to buy YES on A)
    pB: float               # yes_ask_dollars of B (cost to buy YES on B)
    nA: float               # no_ask_dollars of A  (cost to buy NO on A)
    tradeable: bool         # True when guaranteed arbitrage exists
    canonical_title: str    # grouping key (normalized for time-series, raw for same-title)
    pair_type: str          # "time_series" | "same_title"


def normalize_title(title: str) -> str:
    """Strip date tokens; return lowercased, collapsed-whitespace string."""
    result = title
    for pat in _COMPILED:
        result = pat.sub(" ", result)
    return re.sub(r"\s+", " ", result).strip().lower()


def _market_title(market: Any) -> str:
    return market.title or market.subtitle or market.ticker


def get_held_tickers(client: Any) -> set:
    """Return set of tickers where the user currently holds a non-zero position."""
    held: set = set()
    cursor: str | None = None
    while True:
        kwargs: dict = dict(limit=POSITION_PAGE_SIZE, count_filter="position")
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.get_positions(**kwargs)
        for pos in resp.market_positions:
            try:
                if float(pos.position_fp) != 0:
                    held.add(pos.ticker)
            except (ValueError, AttributeError):
                held.add(pos.ticker)
        cursor = resp.cursor
        if not cursor:
            break
    logging.info("Held positions: %d tickers", len(held))
    return held


def _api_call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on 429 responses."""
    delay = 2.0
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                logging.warning("Rate limited — retrying in %.0fs (attempt %d/5)", delay, attempt + 1)
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


def fetch_open_markets(client: Any) -> list:
    """Fetch all open, non-multivariate markets via paginated API calls."""
    markets = []
    cursor: str | None = None
    while True:
        kwargs: dict = dict(status="open", limit=MARKET_PAGE_SIZE, mve_filter="exclude")
        if cursor:
            kwargs["cursor"] = cursor
        resp = _api_call_with_retry(client.get_markets, **kwargs)
        markets.extend(resp.markets)
        cursor = resp.cursor
        if not cursor:
            break
    logging.info("Fetched %d open markets", len(markets))
    return markets


def find_candidate_pairs(
    client: Any,
    held_tickers: set | None = None,
    markets: list | None = None,
) -> list:
    """
    Find time-series arbitrage candidate pairs.

    Grouping strategy: EXACT normalized-title matching.
    If two contracts differ ONLY in their deadline, stripping all date tokens
    from the title yields the exact same string. This is more precise than
    fuzzy matching (which groups unrelated contracts with similar templates).

    A pair is eligible when:
      1. Both markets are actively priced: ask price in [1%, 99%]
      2. Different event_tickers (rules out multi-choice options in the same event)
      3. Deadline gap <= MAX_DEADLINE_GAP_DAYS (30 days)
      4. |pA - pB| >= MIN_PRICE_DIFF (5%)

    Per normalized title, keeps the single best pair (tradeable preferred, then
    largest price gap) to avoid flooding the portfolio with dozens of similar pairs.

    tradeable=True when nA + pB < 1 AND pA > pB (all 3 outcome scenarios profitable).
    """
    if held_tickers is None:
        held_tickers = set()

    if markets is None:
        markets = fetch_open_markets(client)

    markets = [m for m in markets if m.ticker not in held_tickers]
    logging.info("Markets after excluding held positions: %d", len(markets))

    # Pre-filter: only keep actively priced markets (not settled / illiquid)
    active = []
    for m in markets:
        try:
            ya = float(m.yes_ask_dollars)
            if _MIN_ACTIVE_PRICE <= ya <= _MAX_ACTIVE_PRICE:
                active.append(m)
        except (ValueError, TypeError):
            pass
    logging.info("Actively priced markets (ask in 1%%–99%%): %d", len(active))

    # Group by exact normalized title — O(n) hash, no fuzzy matching
    by_title: dict = defaultdict(list)
    for m in active:
        norm = normalize_title(_market_title(m))
        if norm:  # skip empty strings
            by_title[norm].append(m)

    logging.info("Distinct normalized titles with >= 1 market: %d", len(by_title))

    candidate_pairs: list = []
    for norm_title, members in by_title.items():
        if len(members) < 2:
            continue

        members_sorted = sorted(members, key=lambda m: m.close_time)
        group_pairs: list = []

        for i, mA in enumerate(members_sorted):
            for mB in members_sorted[i + 1:]:
                # Same event = options within a multi-choice event, not time-series
                if mA.event_ticker == mB.event_ticker:
                    continue

                gap_days = (mB.close_time - mA.close_time).days
                if gap_days > MAX_DEADLINE_GAP_DAYS:
                    continue

                try:
                    pA = float(mA.yes_ask_dollars)
                    pB = float(mB.yes_ask_dollars)
                    nA = float(mA.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

                if abs(pA - pB) < MIN_PRICE_DIFF:
                    continue

                tradeable = (nA + pB < 1.0) and (pA > pB)

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
                    )
                )

        if not group_pairs:
            continue

        # Keep the single best pair per title group
        group_pairs.sort(key=lambda p: (p.tradeable, abs(p.pA - p.pB)), reverse=True)
        candidate_pairs.append(group_pairs[0])

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
    expensive market and YES on the cheap market is guaranteed profit when nA+pB<1.

    Filters: different event_ticker (to exclude multi-choice options), both actively
    priced (1%-99%), not in held_tickers. One best pair per title group.
    """
    if held_tickers is None:
        held_tickers = set()

    active = []
    for m in markets:
        if m.ticker in held_tickers:
            continue
        try:
            ya = float(m.yes_ask_dollars)
            if _MIN_ACTIVE_PRICE <= ya <= _MAX_ACTIVE_PRICE:
                active.append(m)
        except (ValueError, TypeError):
            pass

    by_terms: dict = defaultdict(list)
    for m in active:
        title    = m.title or ""
        subtitle = m.subtitle or ""
        if title or subtitle:
            by_terms[(title, subtitle)].append(m)

    candidate_pairs: list = []
    for (title, subtitle), members in by_terms.items():
        raw_title = title or subtitle
        if len(members) < 2:
            continue

        group_pairs: list = []
        for i, mA in enumerate(members):
            for mB in members[i + 1:]:
                if mA.event_ticker == mB.event_ticker:
                    continue
                try:
                    pA = float(mA.yes_ask_dollars)
                    pB = float(mB.yes_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # Ensure A is the expensive side (higher YES ask)
                if pA < pB:
                    mA, mB, pA, pB = mB, mA, pB, pA

                if pA - pB < SAME_TITLE_MIN_PRICE_DIFF:
                    continue

                try:
                    nA = float(mA.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

                tradeable = (nA + pB < 1.0)
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
                    )
                )

        if not group_pairs:
            continue

        group_pairs.sort(key=lambda p: (p.tradeable, p.pA - p.pB), reverse=True)
        candidate_pairs.append(group_pairs[0])

    logging.info(
        "Same-title pairs: %d total, %d tradeable",
        len(candidate_pairs),
        sum(1 for p in candidate_pairs if p.tradeable),
    )
    return candidate_pairs
