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
    Imports constants and fee helpers from config.py and the retry/raw-fetch
    helpers from _http.py. Exports the CandidatePair and ApiMarket dataclasses
    and scanning functions consumed by main.py, backtester.py, and (via
    normalize_title) historical.py. Depends on the KalshiClient produced by
    auth.py.

Notes:
    The normalize_title() approach avoids fuzzy matching entirely — it relies on
    the observation that Kalshi titles differ only in date tokens when the same
    question is asked across multiple deadline-indexed markets. The _DATE_PATTERNS
    list must cover all Kalshi date formats to avoid missed pairs or false positives.

    Market fetching deliberately bypasses the SDK's response models: as of
    2026-07 the API stopped sending the legacy integer-cent price fields the
    pinned SDK's Market model requires, so fetch_open_events_with_markets()
    uses the *_without_preload_content raw-response variants and parses JSON
    into ApiMarket itself (see fetch_json_page for the error-handling contract).
    fetch_shard_statuses() reads GET /exchange/status the same way, because the
    pinned SDK's ExchangeStatus model silently drops exchange_index_statuses.

    Exchange shards are TAGGED here, not filtered. The market-data endpoints
    are cross-shard, so every shard's markets are ingested and each ApiMarket
    carries its own exchange_index; the only shard-based drop is of shards the
    exchange itself reports as not trading-active. Whether an order can
    actually be routed to a given shard is decided at submission time by
    trader._legacy_routable, deliberately not here.
"""
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import UTC, datetime, timedelta
from typing import Any

from ._http import api_call_with_retry, fetch_json_page
from .config import (
    DEFAULT_EXCHANGE_INDEX,
    INCLUDE_MVE_MARKETS,
    MARKET_PAGE_SIZE,
    MAX_DEADLINE_GAP_DAYS,
    MVE_MAX_EMPTY_PAGES,
    POSITION_PAGE_SIZE,
    SAME_TITLE_MIN_PRICE_DIFF,
    fee_per_pair_approx,
    min_price_diff_for_gap,
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


@dataclass(frozen=True)
class PriceRange:
    """
    One tick-size band from a market's price_ranges array, in dollars.

    Kalshi markets can have a non-uniform tick grid across their price range
    (e.g. finer ticks near 0 and 1, coarser in the middle — "tapered" tick
    structure). Each band names its own step size; a market's full grid is
    the ordered list of bands covering [0, 1]. Nothing consumes these bands
    yet — see _parse_price_ranges.

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
    array means "tick structure unknown", never an error — nothing consumes
    these bands yet (groundwork for tick-aware order caps after the V2 order
    migration; combo markets move to $0.0001 ticks on 2026-08-17).

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
        tradeable (bool): True when nA + pB < 1 - fee_per_pair_approx(nA, pB) and pA > pB.
            For time-series pairs this is only truly risk-free under the monotonicity
            assumption that P(B) >= P(A) — see find_time_series_pairs for the scenario
            this doesn't cover.
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


def display_title(market: Any) -> str:
    """
    Human-readable label for console and Excel output.

    Returns "<event_title>: <market_title>" when an event title is attached
    (so multivariate option labels like "Trump" or "Above $80k" carry their
    event context for manual spot-checking). Falls back to the bare market
    title for non-MVE markets.

    Args:
        market (Any): A Kalshi market object. May have an `_event_title` attribute.

    Returns:
        str: Display label suitable for user-facing tables and Excel rows.
    """
    event_title = getattr(market, "_event_title", "") or ""
    base = market_title(market)
    return f"{event_title}: {base}" if event_title else base


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

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().

    Returns:
        set: Set of ticker strings (e.g. {"KXBTC-23DEC-T40000", ...}) where the
            user currently holds a non-zero position. Empty set if no positions exist.
    """
    held: set = set()
    cursor: str | None = None
    while True:
        kwargs: dict = {"limit": POSITION_PAGE_SIZE, "count_filter": "position"}
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        # Raw-response call: bypasses the broken MarketPosition model, keeps retry
        data = api_call_with_retry(
            fetch_json_page, client.get_positions_without_preload_content, **kwargs
        )
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
        cursor = data.get("cursor")
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
            "tapered_deci_cent", "deci_cent". "" if absent — nothing reads
            this field yet (groundwork for a future tick-aware order cap).
        price_ranges (list[PriceRange] | None): Parsed tick-size bands (see
            _parse_price_ranges), or None when unknown/unparseable/absent.
            Nothing reads this field yet either.
        exchange_index (int): The exchange shard this market lives on (see
            _shard_index). Market data is cross-shard, so every shard's
            markets are ingested and simply tagged with this; it is
            trader._legacy_routable that refuses to submit an order for a
            non-DEFAULT_EXCHANGE_INDEX market until the V2 migration lands.
            Defaults to DEFAULT_EXCHANGE_INDEX's value when the payload omits
            the field.
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
        # Tag (never filter) the shard so downstream code — today only
        # trader._legacy_routable — can decide what to do with it
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

    A missing, null, or unparseable value is reported as
    DEFAULT_EXCHANGE_INDEX rather than raising: absence of the field is the
    pre-sharding / sandbox shape and must never crash ingest, and a payload
    that simply omits the key is overwhelmingly likely to be an ordinary
    shard-0 market, so mislabelling it as some other shard would wrongly bar
    it from trading.

    Args:
        m (dict): One raw market JSON dict from an events-endpoint payload.

    Returns:
        int: The market's exchange shard index, or DEFAULT_EXCHANGE_INDEX when
            the field is absent, null, or unparseable.
    """
    try:
        return int(m.get("exchange_index") or DEFAULT_EXCHANGE_INDEX)
    except (TypeError, ValueError):
        return DEFAULT_EXCHANGE_INDEX


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

    Args:
        client (Any): An authenticated KalshiClient produced by
            auth.build_client().

    Returns:
        dict | None: Mapping of exchange_index (int) -> {"trading_active":
            bool, "exchange_active": bool, "intra_exchange_transfers_active":
            bool, "description": str}. Malformed entries are skipped. None
            when the breakdown is unavailable or anything at all went wrong.
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
        for entry in raw:
            try:
                # A non-dict entry raises AttributeError on .get; a missing or
                # non-numeric exchange_index raises TypeError/ValueError. One
                # malformed record must not discard the well-formed ones.
                idx = int(entry.get("exchange_index"))
            except (AttributeError, TypeError, ValueError):
                continue
            statuses[idx] = {
                "trading_active": bool(entry.get("trading_active")),
                "exchange_active": bool(entry.get("exchange_active")),
                # Read by the collateral-transfer path in a later commit; kept
                # here so the parsed shape doesn't have to change then.
                "intra_exchange_transfers_active": bool(
                    entry.get("intra_exchange_transfers_active")
                ),
                "description": entry.get("description") or "",
            }
        return statuses
    except Exception as exc:
        logging.info(
            "per-shard exchange status unavailable — assuming single-shard "
            "semantics (%s)", exc,
        )
        return None


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

    Markets are TAGGED with their exchange shard, not filtered by it. The
    market-data endpoints are cross-shard, so every shard's bids and asks
    reach the pair pipeline and each ApiMarket carries its own
    `exchange_index` (see _shard_index). The only shard-based drop here is
    `inactive_shards`: shards the exchange itself reports as not
    trading-active (see fetch_shard_statuses) carry no live book worth
    pairing. Whether a *routable* shard's market can actually be ordered is
    decided at submission time by trader._legacy_routable, not here.

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
        cursor = data.get("cursor")
        # A None or empty cursor signals the last page
        if not cursor:
            break

    # Multivariate events — fetched only when MVE inclusion is enabled.
    # The /events endpoint excludes these by API design, so they need their own pull.
    if INCLUDE_MVE_MARKETS:
        cursor = None
        # The MVE listing is effectively unbounded (hundreds of thousands of
        # auto-generated collection events) and currently returns no nested
        # markets at all, so the pull bails out after MVE_MAX_EMPTY_PAGES
        # consecutive marketless pages instead of paging for hours. Pages that
        # contain markets reset the counter.
        empty_pages = 0
        while True:
            kwargs = {"limit": MARKET_PAGE_SIZE, "with_nested_markets": True}
            if cursor:
                kwargs["cursor"] = cursor
            data = api_call_with_retry(
                fetch_json_page,
                client.get_multivariate_events_without_preload_content,
                **kwargs,
            )
            events = data.get("events") or []
            page_market_count = sum(len(ev.get("markets") or []) for ev in events)
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
            if page_market_count == 0:
                empty_pages += 1
                if empty_pages >= MVE_MAX_EMPTY_PAGES:
                    logging.warning(
                        "MVE fetch: %d consecutive pages with no nested markets — "
                        "stopping the MVE pull early (the API currently returns "
                        "none; the listing itself is effectively unbounded)",
                        empty_pages,
                    )
                    break
            else:
                empty_pages = 0
            cursor = data.get("cursor")
            if not cursor:
                break

    if skipped_shard:
        logging.warning(
            "Skipped %d markets on trading-inactive shards %s",
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
) -> list:
    """
    Find time-series arbitrage candidate pairs.

    Grouping strategy: EXACT normalized-title matching over the combined
    `event_title + market_title` key (see `pair_key`). If two contracts differ
    ONLY in their deadline, stripping all date tokens from the combined key
    yields the exact same string. The event-title prefix is what keeps
    multivariate option labels (e.g. "Trump" appearing in unrelated events)
    from false-positive pairing.

    A pair is eligible when:
      1. Both markets are actively priced: ask price in [1%, 99%]
      2. Different event_tickers (rules out multi-choice options in the same event)
      3. Deadline gap <= MAX_DEADLINE_GAP_DAYS (30 days)
      4. pA - pB >= min_price_diff_for_gap(gap_days) — directional: the
         earlier-closing contract (A) must be priced higher than the later
         one (B), the anomaly this strategy exploits. The required gap is
         tiered by deadline distance (15% when the deadlines are <= 15 days
         apart, 30% for 16-30 days). A pricier later contract is normal term
         structure and is never a candidate.

    Per normalized title, keeps the single best pair (tradeable preferred, then
    largest price gap) to avoid flooding the portfolio with dozens of similar pairs.

    tradeable=True when nA + pB < 1 - fee_per_pair_approx(nA, pB) AND pA > pB. Buying
    NO on A (earlier close) and YES on B (later close) is profitable in 3 of the 4
    outcome combinations (A=NO/B=NO, A=YES/B=YES, A=NO/B=YES all pay out >= the cost).
    The 4th, A=YES/B=NO — the underlying crosses the threshold by A's deadline and
    falls back below it by B's — pays out $0 on both legs, a total loss of nA + pB.
    This flag assumes that reversal has ~zero probability (P(B) >= P(A), i.e. the
    later deadline is monotonically at least as likely to resolve YES); it does not
    hedge against it. That's the time-series independence assumption CLAUDE.md flags,
    and why same-title pairs (no deadline gap, simpler co-resolution model) are
    preferred over time-series ones wherever both exist.
    """
    if markets is None:
        # Fetch all open markets from the Kalshi API if not supplied by the caller
        markets = fetch_open_events_with_markets(client)

    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    active = _filter_active_markets(markets, held_tickers)
    logging.info("Actively priced markets (ask in 1%%–99%%): %d", len(active))

    # Group by exact normalized title over the combined (event + market) key.
    # Stripping date tokens means two markets that differ ONLY in their deadline
    # produce the same key. Using event_title in the key prevents two unrelated
    # MVE events sharing an option label (e.g. "Trump") from being grouped.
    by_title: dict = defaultdict(list)
    for m in active:
        norm = normalize_title(pair_key(m))
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

                # Enforce the minimum YES price difference required for time-series
                # pairs, tiered by deadline gap (15% for gaps <= 15 days, 30% for
                # 16-30 days — wider gaps mean weaker correlation, so more edge is
                # required). Directional, not abs(): mA is always the earlier-closing
                # contract (sorted above), and the tradeable arbitrage only exists
                # when the EARLIER contract is priced higher (pA > pB) — that's the
                # anomaly being exploited. A pricier later contract (pA < pB) is
                # normal term structure, not a candidate, and using abs() here let
                # such pairs through as untradeable placeholders that could still
                # win the group's one-pair-per-title slot below. Mirrors the
                # directional check in backtester._find_entry.
                if pA - pB < min_price_diff_for_gap(gap_days):
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
        # non-tradeable ones; within each tier, the largest price gap wins. pA > pB is
        # now guaranteed for every entry in group_pairs (see the directional filter
        # above), so no abs() is needed.
        group_pairs.sort(key=lambda p: (p.tradeable, p.pA - p.pB), reverse=True)
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

    Grouping key is (event_title, title, subtitle). The event_title component is
    what prevents cross-event option-label collisions in MVE markets — e.g. two
    markets both titled "Trump" in unrelated events will have different event
    titles and therefore won't be grouped together. The subtitle component is
    sourced from the API's `yes_sub_title` field post-2026-08 drift (see
    `_market_from_dict`) — it is the only intra-title discriminator, so without
    it two DIFFERENT outcomes sharing one question title (e.g. two candidates
    under "Who will the next Pope be?") on different event tickers would be
    falsely paired as the same contract under the 95% co-resolution assumption.

    Filters: different event_ticker (to exclude multi-choice options), both actively
    priced (1%-99%), not in held_tickers. One best pair per title group.
    """
    # Remove markets already held and those priced at 0¢/100¢ (settled/illiquid)
    active = _filter_active_markets(markets, held_tickers)

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
    # members = all active markets that share this exact (event_title, title, subtitle)
    # key. Each entry is a separate market object from a different event — any two of
    # them are candidates for a same-title arbitrage if their prices diverge.
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


# Matched wire-format generations for the orderbook response: each container key
# pairs with its OWN side keys and units, and the two must never be mixed. The
# current live API returns the book under `orderbook_fp` with dollar-string bid
# arrays; the legacy `orderbook` container carried integer-cent arrays. Resolving
# container and side keys independently (a plain `or` across both generations)
# risked cross-reading a cents array through the dollars parser, so the pairing is
# enforced explicitly in _fetch_orderbook. Order matters: orderbook_fp (current)
# is tried before orderbook (legacy).
_ORDERBOOK_SIDE_KEYS: dict[str, tuple[str, str]] = {
    "orderbook_fp": ("yes_dollars", "no_dollars"),
    "orderbook": ("yes", "no"),
}


def _fetch_orderbook(client: Any, ticker: str) -> dict | None:
    """
    Fetch the order book for a market.

    The live Kalshi API returns the book under one of two matched key generations
    (see _ORDERBOOK_SIDE_KEYS): the current `orderbook_fp` container with
    `yes_dollars`/`no_dollars` dollar-string bid arrays, or the legacy `orderbook`
    container with `yes`/`no` arrays. The container key selects which side keys are
    expected — the two generations are never mixed. Once a container is found, its
    matched side keys MUST be present (either may be empty or null for a side with
    no resting bids); a container whose matched side keys are absent, or which is
    not even a dict, signals a Kalshi API shape change and is treated as a hard
    failure rather than silently cross-reading the other generation's keys.

    Args:
        client (Any): Authenticated KalshiClient exposing the raw-response
            get_market_orderbook_without_preload_content variant.
        ticker (str): Market ticker whose order book to fetch.

    Returns:
        dict | None: {'yes': [[price_str, qty_str], ...], 'no': [...]} where 'yes'
            is YES bids and 'no' is NO bids, or None on failure. Returns None when
            no recognized container key is present, when the selected container is
            not a dict, or when the container's matched side keys are missing (a
            potential key mismatch — logged as such).
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
        yes_key, no_key = _ORDERBOOK_SIDE_KEYS[container_key]
        # TODO: validate this strict container->side-key mapping against captured
        # test data from Kalshi's live API.
        if not isinstance(ob, dict) or yes_key not in ob or no_key not in ob:
            # Container present but its matched side keys are absent (or it is not
            # even a dict). This is a potential key mismatch — the container and
            # side keys have drifted out of sync, or Kalshi changed the response
            # shape. Fail closed instead of cross-reading the other generation's
            # keys (which could feed a cents array through the dollars parser).
            logging.warning(
                "Potential orderbook key mismatch for %s: container '%s' present but its "
                "expected side keys %s are missing (got %s) — check for bugs or changes to "
                "Kalshi's API",
                ticker,
                container_key,
                (yes_key, no_key),
                sorted(ob.keys()) if isinstance(ob, dict) else type(ob).__name__,
            )
            return None
        # Matched side keys are guaranteed present; a side with no resting bids
        # arrives as null or [] and must read as an empty (not missing) side.
        yes_raw = list(ob[yes_key] or [])
        no_raw  = list(ob[no_key] or [])
        return {"yes": yes_raw, "no": no_raw}
    except Exception as exc:
        logging.warning("Orderbook fetch failed for %s: %s", ticker, exc)
        return None


def _pair_max_sum(pair: Any) -> float:
    """
    Return the maximum allowed yes_price + no_price sum for one pair's orderbook
    depth levels — the complement of the pair's minimum price-gap threshold.

    same_title pairs use the flat 5% threshold (sum <= 0.95). time_series pairs
    use the deadline-gap-tiered threshold: sum <= 0.85 when the deadlines are
    <= 15 days apart, sum <= 0.70 for 16-30 days. market_a is always the
    earlier-closing leg for time_series pairs (guaranteed by the sort in
    find_time_series_pairs), so the gap is non-negative.

    Args:
        pair (CandidatePair): The pair whose ceiling is needed.

    Returns:
        float: Maximum qualifying yes_price + no_price sum (dollars, 0-1).
    """
    if pair.pair_type == "time_series":
        # Tier the ceiling by the same deadline gap used at candidate detection
        gap_days = (pair.market_b.close_time - pair.market_a.close_time).days
        return 1.0 - min_price_diff_for_gap(gap_days)
    return 1.0 - SAME_TITLE_MIN_PRICE_DIFF


def enrich_with_orderbook_prices(client: Any, pairs: list) -> list:
    """
    For each tradeable pair, fetch both order books, pair NO asks (market A)
    with YES asks (market B) using a merge sweep, then filter to only contract
    pairs whose combined price meets the pair's gap threshold (see
    _pair_max_sum):

      same_title:  yes_price + no_price <= 1 - SAME_TITLE_MIN_PRICE_DIFF (0.95)
      time_series: yes_price + no_price <= 1 - min_price_diff_for_gap(gap)
                   (0.85 for deadline gaps <= 15 days, 0.70 for 16-30 days)

    nA and pB are replaced with weighted-average fill prices over qualifying
    contracts. max_contracts is set to the total qualifying count. Pairs with
    no qualifying contracts are marked tradeable=False.
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

        # Convert YES bids of market A into NO ask levels (complement prices)
        # because we are buying NO on market A — YES bids are the counterparties
        no_levels  = _bids_to_ask_levels(ob_a["yes"])
        # Convert NO bids of market B into YES ask levels (complement prices)
        # because we are buying YES on market B — NO bids are the counterparties
        yes_levels = _bids_to_ask_levels(ob_b["no"])

        # Merge-pair the NO and YES depth levels into (yes_price, no_price, qty) tuples
        paired     = _pair_orderbooks(no_levels, yes_levels)

        # Keep only contract pairs where the combined fill price leaves the required
        # gap: same_title requires ≥5% (sum ≤ 0.95); time_series requires the
        # deadline-gap-tiered 15%/30% (sum ≤ 0.85 / 0.70)
        max_sum    = _pair_max_sum(pair)
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
    # Same per-pair gap ceiling used at scan time (deadline-gap-tiered for
    # time_series) — the trade must still qualify at execution time
    max_sum    = _pair_max_sum(pair)
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
