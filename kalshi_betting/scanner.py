"""Market scanning: fetch open markets and find arbitrage candidate pairs."""
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from typing import Any

from .config import (
    MARKET_PAGE_SIZE,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF,
    POSITION_PAGE_SIZE,
    SAME_TITLE_MIN_PRICE_DIFF,
    TAKER_FEE_RATE,
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
    max_contracts: int = 0  # qualifying contracts from order book (0 = not yet enriched)


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

                fee_per_unit = TAKER_FEE_RATE * (nA * (1.0 - nA) + pB * (1.0 - pB))
                tradeable = ((1.0 - nA - pB) > fee_per_unit) and (pA > pB)

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

                fee_per_unit = TAKER_FEE_RATE * (nA * (1.0 - nA) + pB * (1.0 - pB))
                tradeable = (1.0 - nA - pB) > fee_per_unit
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


# ─── Order-book depth pricing ─────────────────────────────────────────────────

def _pair_orderbooks(
    no_levels: list[tuple[float, float]],
    yes_levels: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """
    Merge-pair NO ask levels (market A) with YES ask levels (market B).

    Both lists must be sorted ascending by price. Uses a two-pointer sweep:
    at each step take min(remaining_no, remaining_yes) contracts and emit a
    (yes_price, no_price, qty) tuple. Contracts in one book with no counterpart
    in the other are silently dropped.

    Returns [(yes_price, no_price, qty), ...] in ascending price order.
    """
    pairs: list[tuple[float, float, float]] = []
    i, j = 0, 0
    rem_no  = no_levels[i][1]  if no_levels  else 0.0
    rem_yes = yes_levels[j][1] if yes_levels else 0.0

    while i < len(no_levels) and j < len(yes_levels):
        qty = min(rem_no, rem_yes)
        pairs.append((yes_levels[j][0], no_levels[i][0], qty))
        rem_no  -= qty
        rem_yes -= qty
        if rem_no == 0:
            i += 1
            if i < len(no_levels):
                rem_no = no_levels[i][1]
        if rem_yes == 0:
            j += 1
            if j < len(yes_levels):
                rem_yes = yes_levels[j][1]

    return pairs


def _no_ask_levels(yes_bids_raw: list) -> list[tuple[float, float]]:
    """
    Derive NO ask levels from YES bids returned by the orderbook API.

    yes_bids_raw: [[price_str, qty_str], ...] from orderbook_fp.yes_dollars,
                  sorted descending (highest YES bid first).
    Returns: [(no_ask_price, qty), ...] sorted ascending (cheapest NO ask first).

    Relationship: YES bid at P → NO ask at (1 − P).
    Descending YES bids naturally produce ascending NO asks after the complement.
    """
    levels = []
    for entry in yes_bids_raw:
        try:
            yes_price = float(entry[0])
            qty = float(entry[1])
            no_ask = 1.0 - yes_price
            if _MIN_ACTIVE_PRICE <= no_ask <= _MAX_ACTIVE_PRICE and qty > 0:
                levels.append((no_ask, qty))
        except (ValueError, TypeError, IndexError):
            continue
    levels.sort(key=lambda x: x[0])
    return levels


def _yes_ask_levels(no_bids_raw: list) -> list[tuple[float, float]]:
    """
    Derive YES ask levels from NO bids returned by the orderbook API.

    no_bids_raw: [[price_str, qty_str], ...] from orderbook_fp.no_dollars,
                 sorted descending (highest NO bid first).
    Returns: [(yes_ask_price, qty), ...] sorted ascending (cheapest YES ask first).

    Relationship: NO bid at P → YES ask at (1 − P).
    """
    levels = []
    for entry in no_bids_raw:
        try:
            no_price = float(entry[0])
            qty = float(entry[1])
            yes_ask = 1.0 - no_price
            if _MIN_ACTIVE_PRICE <= yes_ask <= _MAX_ACTIVE_PRICE and qty > 0:
                levels.append((yes_ask, qty))
        except (ValueError, TypeError, IndexError):
            continue
    levels.sort(key=lambda x: x[0])
    return levels


def _fetch_orderbook(client: Any, ticker: str) -> dict | None:
    """
    Fetch the order book for a market.

    Returns {'yes': [[price_str, qty_str], ...], 'no': [...]} where 'yes' is
    YES bids and 'no' is NO bids (both sorted descending by price), or None on
    failure.
    """
    try:
        resp = _api_call_with_retry(client.get_market_orderbook, ticker=ticker)
        ob = getattr(resp, "orderbook_fp", None) or getattr(resp, "orderbook", None)
        if ob is None:
            return None
        yes_raw = list(getattr(ob, "yes_dollars", None) or getattr(ob, "yes", None) or [])
        no_raw  = list(getattr(ob, "no_dollars",  None) or getattr(ob, "no",  None) or [])
        return {"yes": yes_raw, "no": no_raw}
    except Exception as exc:
        logging.warning("Orderbook fetch failed for %s: %s", ticker, exc)
        return None


def enrich_with_orderbook_prices(client: Any, pairs: list) -> list:
    """
    For each tradeable pair, fetch both order books, pair NO asks (market A)
    with YES asks (market B) using a merge sweep, then filter to only contract
    pairs whose combined price meets the per-type gap threshold:

      same_title:  yes_price + no_price <= 1 - SAME_TITLE_MIN_PRICE_DIFF (0.95)
      time_series: yes_price + no_price <= 1 - MIN_PRICE_DIFF             (0.85)

    nA and pB are replaced with the weighted-average fill prices over the
    qualifying contracts. max_contracts is set to the total qualifying count,
    capping how many contracts strategy.py may size. Pairs with no qualifying
    contracts are marked tradeable=False.
    """
    # Per pair-type: max allowed (yes_price + no_price) for a contract pair
    _max_sum = {
        "same_title":  1.0 - SAME_TITLE_MIN_PRICE_DIFF,  # 0.95
        "time_series": 1.0 - MIN_PRICE_DIFF,              # 0.85
    }

    ob_cache: dict[str, dict | None] = {}

    def get_ob(ticker: str) -> dict | None:
        if ticker not in ob_cache:
            ob_cache[ticker] = _fetch_orderbook(client, ticker)
        return ob_cache[ticker]

    enriched = []
    for pair in pairs:
        if not pair.tradeable:
            enriched.append(pair)
            continue

        ob_a = get_ob(pair.market_a.ticker)
        ob_b = get_ob(pair.market_b.ticker)

        if ob_a is None or ob_b is None:
            logging.warning(
                "Orderbook unavailable for '%s' — keeping best-ask prices",
                pair.canonical_title,
            )
            enriched.append(pair)
            continue

        no_levels  = _no_ask_levels(ob_a["yes"])
        yes_levels = _yes_ask_levels(ob_b["no"])

        paired = _pair_orderbooks(no_levels, yes_levels)

        max_sum = _max_sum.get(pair.pair_type, 0.95)
        qualifying = [
            (yp, np_, qty) for yp, np_, qty in paired if yp + np_ <= max_sum
        ]

        if not qualifying:
            logging.info(
                "No qualifying contract pairs for '%s' after price gap filter — skipping",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        total_qty = sum(qty for _, _, qty in qualifying)
        avg_pB = sum(yp  * qty for yp, _,   qty in qualifying) / total_qty
        avg_nA = sum(np_ * qty for _,  np_, qty in qualifying) / total_qty

        fee_per_unit = TAKER_FEE_RATE * (avg_nA * (1.0 - avg_nA) + avg_pB * (1.0 - avg_pB))
        new_tradeable = (1.0 - avg_nA - avg_pB) > fee_per_unit

        if not new_tradeable:
            logging.info(
                "Pair '%s' unprofitable after depth adjustment: avg_nA=%.3f avg_pB=%.3f",
                pair.canonical_title, avg_nA, avg_pB,
            )

        enriched.append(dc_replace(
            pair,
            nA=avg_nA,
            pB=avg_pB,
            tradeable=new_tradeable,
            max_contracts=int(total_qty),
        ))

    tradeable_after = sum(1 for p in enriched if p.tradeable)
    logging.info(
        "Orderbook depth check: %d/%d pairs remain tradeable after price gap filter",
        tradeable_after,
        sum(1 for p in pairs if p.tradeable),
    )
    return enriched
