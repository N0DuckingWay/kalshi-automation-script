"""
File: scanner.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Fetches all open Kalshi markets from the REST API and identifies pairs of
    contracts that are candidates for arbitrage. Two detection paths exist:
    (1) time-series pairs — contracts that ask the same question at different
    deadlines, identified by stripping date tokens from their titles and
    exact-matching the remainder; and (2) same-title pairs — contracts with
    identical title and subtitle on different event tickers. Both paths then
    check the live order book to replace best-ask prices with depth-weighted
    fill prices and confirm the edge survives real liquidity.

Dependencies:
    Imports constants and fee helpers from config.py. Exports the CandidatePair
    dataclass and scanning functions consumed by main.py, backtester.py, and
    (via normalize_title) historical.py. Depends on the KalshiClient produced
    by auth.py.

Notes:
    The normalize_title() approach avoids fuzzy matching entirely — it relies on
    the observation that Kalshi titles differ only in date tokens when the same
    question is asked across multiple deadline-indexed markets. The _DATE_PATTERNS
    list must cover all Kalshi date formats to avoid missed pairs or false positives.
"""
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any

from .config import (
    MARKET_PAGE_SIZE,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF,
    POSITION_PAGE_SIZE,
    SAME_TITLE_MIN_PRICE_DIFF,
    fee_per_pair_approx,
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
    """
    An arbitrage candidate consisting of two correlated markets with a detectable price gap.

    Attributes:
        market_a (Any): The market with the higher YES ask (the expensive side of the pair).
        market_b (Any): The market with the lower YES ask (the cheap side of the pair).
        pA (float): YES ask price of market A in dollars (cost to buy YES on A). Range: [0, 1].
        pB (float): YES ask price of market B in dollars (cost to buy YES on B). Range: [0, 1].
        nA (float): NO ask price of market A in dollars (cost to buy NO on A). Range: [0, 1].
        tradeable (bool): True when a risk-free arbitrage exists (nA + pB < 1 and pA > pB).
        canonical_title (str): Grouping key used to identify the pair — normalized title for
            time-series pairs, raw title for same-title pairs.
        pair_type (str): Strategy variant: "time_series" for pairs differing only in deadline,
            "same_title" for pairs with identical title/subtitle across different event tickers.
    """
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
    """
    Strip all date and time tokens from a market title, returning a normalized string.

    Removes patterns such as month names, ISO dates, quarters, and relative time
    expressions, then collapses whitespace and lowercases the result. Two markets
    that differ only in their deadline will produce the same normalized string,
    enabling exact-match grouping without fuzzy matching.

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

    Args:
        market (Any): A Kalshi market API object with `.title`, `.subtitle`, and `.ticker` attributes.

    Returns:
        str: The first non-falsy value among title, subtitle, and ticker.
    """
    return market.title or market.subtitle or market.ticker


def _filter_active_markets(markets: list, excluded_tickers: set | None = None) -> list:
    """
    Filter markets to those that are actively priced and not already held.

    A market is considered actively priced when its YES ask is between 1¢ and 99¢.
    Markets at 0¢ or 100¢ are effectively settled or completely illiquid — trading
    them offers no edge. Markets whose tickers are in excluded_tickers are already
    held in the portfolio and must not be traded again.

    Args:
        markets (list): List of Kalshi market API objects to filter.
        excluded_tickers (set | None): Set of ticker strings to skip. If None,
            no tickers are excluded.

    Returns:
        list: Subset of markets that have a parseable YES ask in [0.01, 0.99]
            and whose ticker is not in excluded_tickers.
    """
    excluded = excluded_tickers or set()
    active = []
    for m in markets:
        if m.ticker in excluded:
            continue
        try:
            ya = float(m.yes_ask_dollars)
            # Skip markets at 0¢ (already settled NO) or 100¢ (already settled YES)
            if _MIN_ACTIVE_PRICE <= ya <= _MAX_ACTIVE_PRICE:
                active.append(m)
        except (ValueError, TypeError):
            pass
    return active


def get_held_tickers(client: Any) -> set:
    """
    Fetch all tickers where the account currently holds a non-zero position.

    Iterates through all pages of the portfolio positions endpoint and collects
    tickers with a non-zero position_fp (fractional position). These tickers are
    excluded from new trades to avoid doubling up on an existing position.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().

    Returns:
        set: Set of ticker strings (e.g. {"KXBTC-23DEC-T40000", ...}) where the
            user currently holds a non-zero position. Empty set if no positions exist.
    """
    held: set = set()
    cursor: str | None = None
    while True:
        kwargs: dict = dict(limit=POSITION_PAGE_SIZE, count_filter="position")
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.get_positions(**kwargs)
        for pos in resp.market_positions:
            try:
                # position_fp is a float string — add to held set if non-zero
                if float(pos.position_fp) != 0:
                    held.add(pos.ticker)
            except (ValueError, AttributeError):
                # If we can't parse the position, be conservative and treat it as held
                held.add(pos.ticker)
        cursor = resp.cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break
    logging.info("Held positions: %d tickers", len(held))
    return held


def _api_call_with_retry(fn, *args, **kwargs):
    """
    Call an API function with exponential backoff on HTTP 429 (rate limit) errors.

    Retries up to 5 times with a starting delay of 2 seconds, doubling each time
    up to a maximum of 60 seconds. Any non-429 exception is re-raised immediately
    without retrying (e.g. 401 auth errors, 404 not found).

    Args:
        fn: A callable that performs an API call (e.g. client.get_markets).
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        The return value of fn(*args, **kwargs) on success.

    Raises:
        Exception: Re-raises the original exception if it is not a 429 error, or
            if all 5 retries are exhausted.
    """
    delay = 2.0
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                logging.warning("Rate limited — retrying in %.0fs (attempt %d/5)", delay, attempt + 1)
                time.sleep(delay)
                # Exponential backoff capped at 60 seconds to avoid waiting indefinitely
                delay = min(delay * 2, 60)
            else:
                raise


def fetch_open_markets(client: Any) -> list:
    """
    Fetch all open, non-multivariate Kalshi markets via paginated API calls.

    Uses the mve_filter="exclude" parameter to skip multivariate (multi-choice)
    markets, which have fundamentally different pricing semantics and would
    generate false positives in the arbitrage scanner. Iterates through all
    pages of results using cursor-based pagination.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().

    Returns:
        list: All open, non-multivariate Kalshi market API objects. May contain
            thousands of items depending on the current state of the platform.
    """
    markets = []
    cursor: str | None = None
    while True:
        kwargs: dict = dict(status="open", limit=MARKET_PAGE_SIZE, mve_filter="exclude")
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        resp = _api_call_with_retry(client.get_markets, **kwargs)
        markets.extend(resp.markets)
        cursor = resp.cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break
    logging.info("Fetched %d open markets", len(markets))
    return markets


def find_time_series_pairs(
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
      4. |pA - pB| >= MIN_PRICE_DIFF (15%)

    Per normalized title, keeps the single best pair (tradeable preferred, then
    largest price gap) to avoid flooding the portfolio with dozens of similar pairs.

    tradeable=True when nA + pB < 1 AND pA > pB (all 3 outcome scenarios profitable).
    """
    if markets is None:
        # Fetch all open markets from the Kalshi API if not supplied by the caller
        markets = fetch_open_markets(client)

    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    active = _filter_active_markets(markets, held_tickers)
    logging.info("Actively priced markets (ask in 1%%–99%%): %d", len(active))

    # Group by exact normalized title — O(n) hash, no fuzzy matching.
    # Stripping date tokens from titles means two markets that differ ONLY in their
    # deadline produce the same key, so they end up in the same group.
    by_title: dict = defaultdict(list)
    for m in active:
        norm = normalize_title(market_title(m))
        # Skip markets whose title collapses entirely to an empty string after stripping
        if norm:
            by_title[norm].append(m)

    logging.info("Distinct normalized titles with >= 1 market: %d", len(by_title))

    candidate_pairs: list = []
    for norm_title, members in by_title.items():
        # Need at least two markets in a group to form any pair
        if len(members) < 2:
            continue

        # Sort ascending by close_time so mA is always the earlier-closing contract
        members_sorted = sorted(members, key=lambda m: m.close_time)
        group_pairs: list = []

        for i, mA in enumerate(members_sorted):
            for mB in members_sorted[i + 1:]:
                # Same event_ticker means these are options within a multi-choice event,
                # not separate time-series markets — skip them
                if mA.event_ticker == mB.event_ticker:
                    continue

                # Deadline gap check: pairs more than 30 days apart are too weakly
                # correlated for the time-series arbitrage assumption to hold reliably
                gap_days = (mB.close_time - mA.close_time).days
                if gap_days > MAX_DEADLINE_GAP_DAYS:
                    continue

                try:
                    pA = float(mA.yes_ask_dollars)
                    pB = float(mB.yes_ask_dollars)
                    nA = float(mA.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # Enforce the minimum 15% YES price difference required for time-series pairs.
                # Smaller gaps don't provide enough edge to cover fees and uncertainty.
                if abs(pA - pB) < MIN_PRICE_DIFF:
                    continue

                # tradeable=True when buying NO on A and YES on B is guaranteed profitable
                # in all three resolution scenarios (before exact fee computation).
                # fee_per_pair_approx returns a continuous estimate of total taker fees.
                tradeable = ((1.0 - nA - pB) > fee_per_pair_approx(nA, pB)) and (pA > pB)

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

        # Keep only the single best pair per normalized title group to avoid flooding
        # the portfolio with many near-identical positions. Tradeable pairs rank above
        # non-tradeable ones; within each tier, the largest price gap wins.
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
    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    active = _filter_active_markets(markets, held_tickers)

    # Group by exact (title, subtitle) tuple — no normalization.
    # Two markets with identical text in both fields are asking the same question.
    by_terms: dict = defaultdict(list)
    for m in active:
        title    = m.title or ""
        subtitle = m.subtitle or ""
        if title or subtitle:
            by_terms[(title, subtitle)].append(m)

    candidate_pairs: list = []
    for (title, subtitle), members in by_terms.items():
        # Use whichever of title or subtitle is non-empty as the display label
        raw_title = title or subtitle
        # Need at least two markets in a group to form any pair
        if len(members) < 2:
            continue

        group_pairs: list = []
        for i, mA in enumerate(members):
            for mB in members[i + 1:]:
                # Same event_ticker means these are options in the same multi-choice event,
                # not separate markets asking the same question — skip them
                if mA.event_ticker == mB.event_ticker:
                    continue
                try:
                    pA = float(mA.yes_ask_dollars)
                    pB = float(mB.yes_ask_dollars)
                except (ValueError, TypeError):
                    continue

                # Canonicalize so market_a is always the more expensive side (higher YES ask).
                # The strategy buys NO on the expensive side and YES on the cheap side.
                if pA < pB:
                    mA, mB, pA, pB = mB, mA, pB, pA

                # Enforce the minimum 5% YES price difference for same-title pairs.
                # A smaller gap is within normal bid-ask spread noise.
                if pA - pB < SAME_TITLE_MIN_PRICE_DIFF:
                    continue

                try:
                    nA = float(mA.no_ask_dollars)
                except (ValueError, TypeError):
                    continue

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
                    )
                )

        if not group_pairs:
            continue

        # Keep only the best pair per exact title group — largest price gap among
        # tradeable pairs wins; tradeable is preferred over non-tradeable
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

    Both lists sorted ascending by price. Two-pointer sweep: at each step take
    min(remaining_no, remaining_yes) contracts and emit (yes_price, no_price, qty).
    Contracts left over in one book with no counterpart in the other are dropped.

    Returns [(yes_price, no_price, qty), ...].
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


def _bids_to_ask_levels(bids_raw: list) -> list[tuple[float, float]]:
    """
    Convert bid levels to ask levels via the complement price (1 − P).

    bids_raw: [[price_str, qty_str], ...] sorted descending by price.
    Returns: [(ask_price, qty), ...] sorted ascending (cheapest ask first).

    Applies to both sides: YES bid at P → NO ask at (1−P);
                           NO bid at P → YES ask at (1−P).
    Descending bids naturally yield ascending asks after the complement.
    """
    levels = []
    for entry in bids_raw:
        try:
            bid_price = float(entry[0])
            qty = float(entry[1])
            ask_price = 1.0 - bid_price
            if _MIN_ACTIVE_PRICE <= ask_price <= _MAX_ACTIVE_PRICE and qty > 0:
                levels.append((ask_price, qty))
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

    nA and pB are replaced with weighted-average fill prices over qualifying
    contracts. max_contracts is set to the total qualifying count. Pairs with
    no qualifying contracts are marked tradeable=False.
    """
    _max_sum = {
        "same_title":  1.0 - SAME_TITLE_MIN_PRICE_DIFF,  # 0.95
        "time_series": 1.0 - MIN_PRICE_DIFF,              # 0.85
    }
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
            # If either order book is unavailable, keep the pair with best-ask prices
            # rather than marking it untradeable — the scanner already validated the gap
            logging.warning(
                "Orderbook unavailable for '%s' — keeping best-ask prices",
                pair.canonical_title,
            )
            enriched.append(pair)
            continue

        # Convert YES bids of market A into NO ask levels (complement prices)
        # because we are buying NO on market A — YES bids are the counterparties
        no_levels  = _bids_to_ask_levels(ob_a["yes"])
        # Convert NO bids of market B into YES ask levels (complement prices)
        # because we are buying YES on market B — NO bids are the counterparties
        yes_levels = _bids_to_ask_levels(ob_b["no"])

        # Merge-pair the NO and YES depth levels into (yes_price, no_price, qty) tuples
        paired     = _pair_orderbooks(no_levels, yes_levels)

        max_sum    = _max_sum.get(pair.pair_type, 0.95)
        # Keep only contract pairs where the combined fill price leaves the required gap:
        # same_title requires ≥5% gap (sum ≤ 0.95); time_series requires ≥15% gap (sum ≤ 0.85)
        qualifying = [
            (yp, np_, qty) for yp, np_, qty in paired if yp + np_ <= max_sum
        ]

        if not qualifying:
            # No depth available at the required gap — mark untradeable to skip execution
            logging.info(
                "No qualifying contract pairs for '%s' after price gap filter — skipping",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        total_qty = sum(qty for _, _, qty in qualifying)
        # Compute depth-weighted average fill prices to replace the best-ask estimates
        avg_pB    = sum(yp  * qty for yp, _,   qty in qualifying) / total_qty
        avg_nA    = sum(np_ * qty for _,  np_, qty in qualifying) / total_qty

        # Re-validate tradeability at the depth-weighted prices (the pair may still be
        # unprofitable if all qualifying contracts are at the edge of the gap threshold)
        new_tradeable = (1.0 - avg_nA - avg_pB) > fee_per_pair_approx(avg_nA, avg_pB)

        if not new_tradeable:
            logging.info(
                "Pair '%s' unprofitable after depth adjustment: avg_nA=%.3f avg_pB=%.3f",
                pair.canonical_title, avg_nA, avg_pB,
            )

        # Replace best-ask prices and contract count with depth-accurate values;
        # strategy.py will use these to compute the final Kelly-sized trade
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


def validate_pair_price(client: Any, spec: Any) -> bool:
    """
    Re-fetch both order books for a TradeSpec immediately before execution and
    confirm the gap threshold still holds at the required contract depth.

    Returns True only if qualifying depth >= spec.x contracts remain at the
    pair's gap threshold. A False result means prices have moved since the
    scan and the trade should be skipped.

    Args:
        client: Authenticated KalshiClient from auth.build_client().
        spec: TradeSpec whose pair prices should be re-validated.

    Returns:
        bool: True if the pair still qualifies; False if prices moved or order
            books are unavailable.
    """
    _max_sum = {
        "same_title":  1.0 - SAME_TITLE_MIN_PRICE_DIFF,
        "time_series": 1.0 - MIN_PRICE_DIFF,
    }
    pair   = spec.pair
    ob_a   = _fetch_orderbook(client, pair.market_a.ticker)
    ob_b   = _fetch_orderbook(client, pair.market_b.ticker)

    if ob_a is None or ob_b is None:
        logging.warning(
            "Pre-execution orderbook unavailable for '%s' — skipping",
            pair.canonical_title,
        )
        return False

    no_levels  = _bids_to_ask_levels(ob_a["yes"])
    yes_levels = _bids_to_ask_levels(ob_b["no"])
    paired     = _pair_orderbooks(no_levels, yes_levels)
    max_sum    = _max_sum.get(pair.pair_type, 0.95)
    qualifying = [(yp, np_, qty) for yp, np_, qty in paired if yp + np_ <= max_sum]

    if not qualifying:
        logging.info(
            "Pre-execution check failed for '%s' — gap no longer qualifies",
            pair.canonical_title,
        )
        return False

    # Require enough depth to fill our full intended contract count via FoK
    total_qty = sum(qty for _, _, qty in qualifying)
    if total_qty < spec.x:
        logging.info(
            "Pre-execution check failed for '%s' — only %.1f contracts at gap (need %d)",
            pair.canonical_title, total_qty, spec.x,
        )
        return False

    return True
