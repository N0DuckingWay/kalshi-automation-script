"""
File: backtester.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Replays both pair strategies — same-title co-resolution pairs and the
    directional time-series bet — on the full history of settled Kalshi markets.
    Groups settled markets into potential time-series and same-title pairs, fetches
    hourly candlestick price series for each involved market, then scans weekly
    Monday snapshots to find the first date each pair was tradeable at the required
    threshold. Applies Kelly sizing to compute trade size, records actual P&L from
    settlement outcomes, deduplicates overlapping pairs by priority, and builds
    a daily equity curve. Results feed into dashboard.py for visualization.

Dependencies:
    Imports normalize_title and leg_sides from scanner.py; fee/model helpers
    (fee_leg_exact, fee_per_pair_approx, min_price_diff_for_gap,
    time_series_profit_prob) plus BUDGET_FRACTION,
    CANDLESTICK_FETCH_MAX_WORKERS, LARGE_GROUP_WARN_THRESHOLD,
    MAX_DEADLINE_GAP_DAYS, SAME_TITLE_CO_RESOLVE_PROB, SAME_TITLE_MIN_PRICE_DIFF,
    and SETTLED_PREFILTER_CACHE_TAG from config.py; fetch_all_settled_markets(),
    fetch_candlesticks(), and infer_category() from historical.py. Also
    depends on pandas (external) for the equity-curve DataFrame. Does NOT
    import strategy.py — Kelly sizing and portfolio selection are
    re-implemented inline against the same config.py constants, so a change
    to either sizing formula must be made in both places to keep live/backtest
    parity. Exports BacktestTrade (consumed by dashboard.py) and run_backtest()
    (called by backtest.py).

Notes:
    The backtester uses a two-pass approach: Pass 1 collects all potential entries
    (prices, dates, Kelly fraction — no sizing), keeps only the best entry per
    title group (mirroring the live scanners' one-pair-per-group rule), and then
    drops any time-series candidate whose ticker pair was also found as a
    same-title candidate — the same preference main._dedup_pairs applies live,
    since the same-title co-resolution model (identical questions must
    co-resolve) is simpler than the directional time-series bet, whose edge
    rests on the operator-tuned interval discount behind
    config.time_series_profit_prob. Pass 2 walks
    entries in chronological order (priority-ordered within a date using the
    ENTRY-TIME expected return, never realized results), maintains a running cash
    balance — sizing every candidate of an entry date against that checkpoint's
    opening balance, admitting them greedily against the running cash (mirroring
    main._run_prod + strategy.select_portfolio) and releasing settlement receipts
    on exit dates — and applies a greedy ticker-conflict filter
    so each market ticker appears in at most one OPEN trade at a time (the ticker
    is released on its trade's exit date, alongside the cash). This mirrors the
    live bot's Kelly sizing against one per-run balance snapshot and its
    one-active-position-per-ticker rule: get_held_tickers() reads positions with
    count_filter="position", so a settled ticker leaves the blocked set live too.

    Before grouping, run_backtest() filters markets through _can_ever_enter(),
    a necessary-condition prefilter: _find_entry() can only open a trade at a
    Monday-09:00-UTC checkpoint on/after start_date, and requires both legs to
    have an hourly candle at-or-before that Monday (i.e. opened by then). A
    market whose [open_time, close_time - 1 day] window contains no such
    Monday can never appear in any entered pair, as either leg, in either pair
    type — dropping it up front avoids materializing it into any group at all.
    This matters because normalized-title groups can have 10,000+ members at
    current Kalshi volumes (hourly/intraday crypto ladders collapsing into one
    group) — without the prefilter, and without the close-time-windowed
    enumeration in _extract_pairs() for time-series groups, pair extraction is
    O(n^2) per group and infeasible (500B+ iterations observed on a single
    53k-member group). Neither optimization changes results: both only skip
    work that provably cannot produce an entry.

    Time-series pairs buy YES on the earlier-closing contract (market A) and
    NO on the later one (market B) — scanner.leg_sides is the only source of
    truth for the sides, and _settlement_receipt pays by side. Their
    settlement table therefore has exactly three cells: event by A (A=YES,
    B=YES — YES-on-A pays n), never by B (A=NO, B=NO — NO-on-B pays n), and in
    between (A=NO, B=YES — both legs worthless, the full stake is lost). A=YES
    with B=NO is impossible for a cumulative-deadline pair: a candidate that
    settled that way is excluded from Pass 1 (never traded, never paid) and
    counted, and one summary WARNING reports the count. Kalshi does list
    snapshot-style markets ("on <date>"), so that counter is the only signal
    that the normalized-title grouping admitted a non-cumulative pair — the
    live scanner cannot detect it from prices.
"""
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from .config import (
    BUDGET_FRACTION,
    CANDLESTICK_FETCH_MAX_WORKERS,
    LARGE_GROUP_WARN_THRESHOLD,
    MAX_DEADLINE_GAP_DAYS,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
    SETTLED_PREFILTER_CACHE_TAG,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
    time_series_profit_prob,
)
from .historical import (
    fetch_all_settled_markets,
    fetch_candlesticks,
    infer_category,
)
from .scanner import leg_sides, normalize_title

# Seconds in one UTC day. Same value as historical._DAY_SECONDS, kept local
# rather than importing a private name.
_DAY_SECONDS = 86_400

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """
    Complete record of a single simulated pair trade from the backtest.

    Captures both the trade setup (entry prices, sizing, pair metadata) and the
    final outcome (settlement results, P&L, slippage) for use in performance analysis
    and dashboard generation. Which side each leg bought depends on pair_type
    (scanner.leg_sides): same_title buys NO on market A and YES on market B;
    time_series buys YES on market A (the earlier-closing contract) and NO on
    market B (the later one).

    Attributes:
        pair_type (str): Strategy variant used: "time_series" or "same_title".
        ticker_a (str): Kalshi ticker of market A — the pricier side, NO bought
            (same_title), or the earlier-closing contract, YES bought (time_series).
        ticker_b (str): Kalshi ticker of market B — the cheaper side, YES bought
            (same_title), or the later-closing contract, NO bought (time_series).
        title_a (str): Display title of market A.
        title_b (str): Display title of market B.
        category (str): Human-readable market category inferred from event_ticker prefix
            (e.g. "Crypto", "Sports", "Politics").
        entry_date (date): The Monday on which the trade was first tradeable and sized.
        exit_date (date): The date the later-settling market resolved; marks when cash returned.
        entry_pA (float): YES ask price of market A at entry. Range: [0.01, 0.99].
            The traded price of the market-A leg for time_series; for same_title
            it is the quote that made A the pricier side (reporting only).
        entry_pB (float): YES ask price of market B at entry. Range: [0.01, 0.99].
            The traded price of the market-B leg for same_title; for time_series
            it feeds the profit model together with entry_pA but is not traded.
        entry_nA (float): NO ask price of market A at entry (≈ 1 − yes_bid_A).
            Range: [0.01, 0.99]. The traded price of the market-A leg for
            same_title; reporting only for time_series.
        entry_nB (float): NO ask price of market B at entry (≈ 1 − yes_bid_B).
            Range: [0.01, 0.99]. The traded price of the market-B leg for
            time_series; reporting only for same_title.
        n (int): Number of contracts bought on each leg (x = y = n). Always >= 1.
            Both legs are always the same size.
        total_cost (float): Dollar cost of the contracts: n * (price_a + price_b),
            where the leg prices are (entry_nA, entry_pB) for same_title and
            (entry_pA, entry_nB) for time_series. Excludes taker fees (see fees).
        fees (float): Exact ceiling-rounded taker fee for both legs, charged at entry.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        actual_payoff (float): Gross dollar value received at settlement: $1 per
            contract for each leg whose market resolved to the side it bought
            (see _settlement_receipt for the per-type tables). Fees and entry
            cost are NOT deducted here — they are accounted in profit.
        profit (float): actual_payoff − total_cost − fees. same_title: negative
            only when A=YES and B=NO. time_series: negative only in the
            in-between cell (A=NO, B=YES); both win cells (A=YES, B=YES and
            A=NO, B=NO) realize exactly expected_payoff.
        profit_ratio (float): profit / (total_cost + fees). Return on the cash
            actually invested.
        monthly_profit_ratio (float): Realized profit_ratio scaled to 30 days:
            profit_ratio * 30 / holding_days. Reporting only — trade selection
            uses the entry-time expected ratio to avoid look-ahead bias.
        kelly_fraction (float): Capped Kelly fraction used for sizing, <= BUDGET_FRACTION.
        expected_payoff (float): NET profit in a win scenario after exact fees:
            n * (1 − price_a − price_b) − fees on the leg prices above. For
            same_title this is the guaranteed floor (every co-resolution
            outcome pays at least n); for time_series it is the profit realized
            in either win cell, while the in-between cell loses total_cost +
            fees instead. Always > 0 for recorded trades.
        slippage (float): profit − expected_payoff. same_title: positive when
            both legs paid (A=NO, B=YES), zero in the co-resolution cells,
            negative only in the loss cell. time_series: zero in both win cells
            and negative in the loss cell — there is no positive-slippage cell.
        holding_days (int): Calendar days between entry_date and exit_date. Always >= 1.
        balance_at_entry (float): Simulated cash balance in dollars at the OPEN
            of this trade's entry-date checkpoint — the base the Kelly budget
            was sized against, shared by every trade entering that same date
            (mirroring the single balance read at the top of a live run).
        deadline_gap_days (int | None): Calendar days between the two legs'
            close_times, carried out of _find_entry (which selected the price
            tier and applied the MAX_DEADLINE_GAP_DAYS cutoff on exactly this
            number) rather than recomputed. None for same_title, which has no
            deadline-gap concept, and for any trade constructed without it
            (test fixtures). Reporting only — nothing sizes, prices or settles
            on this field.
    """
    pair_type: str       # "time_series" | "same_title"
    ticker_a: str
    ticker_b: str
    title_a: str
    title_b: str
    category: str
    entry_date: date
    exit_date: date      # date the last-settling market resolved
    entry_pA: float      # YES ask of A at entry — the market-A leg price for time_series
    entry_pB: float      # YES ask of B at entry — the market-B leg price for same_title
    entry_nA: float      # NO ask of A at entry (≈ 1 - yes_bid_A) — the market-A leg price for same_title
    entry_nB: float      # NO ask of B at entry (≈ 1 - yes_bid_B) — the market-B leg price for time_series
    n: int               # contracts bought on each leg (x = y = n)
    total_cost: float
    fees: float          # exact both-leg taker fees, charged at entry
    outcome_a: str       # "yes" | "no"
    outcome_b: str       # "yes" | "no"
    actual_payoff: float
    profit: float
    profit_ratio: float
    monthly_profit_ratio: float  # realized profit_ratio * 30 / holding_days (reporting only)
    kelly_fraction: float        # capped Kelly fraction used for sizing
    expected_payoff: float  # n * (1 - price_a - price_b) minus fees — same_title floor / time_series win-cell profit
    slippage: float         # profit - expected_payoff
    holding_days: int
    balance_at_entry: float  # checkpoint opening balance the Kelly budget used
    # Calendar days between the two legs' close_times, carried out of
    # _find_entry rather than recomputed. None for same_title (no deadline-gap
    # concept) and for any trade constructed without it (test fixtures).
    # Reporting only — nothing sizes, prices or settles on this field.
    deadline_gap_days: int | None = None


def _settlement_receipt(n: int, outcome_a: str, outcome_b: str, pair_type: str) -> float:
    """
    Compute the gross dollar amount received at settlement for one pair trade.

    Each contract pays exactly $1 when its side wins and $0 otherwise, so the
    receipt is n per leg whose market resolved to the side that leg bought —
    scanner.leg_sides(pair_type) is the only source of the sides — and is
    independent of entry prices.

    same_title (n NO on market A + n YES on market B):

      A=YES, B=YES: n   [YES on B pays; NO on A worthless]
      A=NO,  B=YES: 2n  [both legs pay — best scenario]
      A=NO,  B=NO:  n   [NO on A pays; YES on B worthless]
      A=YES, B=NO:  0   [loss scenario — both legs worthless]

    time_series (n YES on the earlier market A + n NO on the later market B)
    has exactly three cells:

      A=YES, B=YES: n   [event by A — YES on A pays; NO on B worthless]
      A=NO,  B=NO:  n   [never by B — NO on B pays; YES on A worthless]
      A=NO,  B=YES: 0   [in between — the loss cell; both legs worthless]

    A=YES with B=NO is impossible for a cumulative-deadline pair (YES by the
    earlier deadline implies YES by the later one), so that combination is not
    a payout cell at all: it raises. run_backtest excludes such candidates in
    Pass 1 before ever sizing them, so the guard here is defensive and
    unreachable from the pipeline — it exists so no caller can ever price the
    fourth cell by accident.

    Entry cost and taker fees are deliberately NOT deducted here — callers
    subtract them exactly once when computing profit and the equity curve.

    Args:
        n (int): Number of contracts bought on each leg.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        pair_type (str): "time_series" or "same_title" — selects the sides via
            scanner.leg_sides (anything but "time_series" is same-title).

    Returns:
        float: Gross settlement receipt in dollars: n per leg that paid.

    Raises:
        ValueError: For a time_series pair whose earlier contract settled YES
            while the later settled NO — a premise violation, never a payout.
    """
    if pair_type == "time_series" and outcome_a == "yes" and outcome_b == "no":
        raise ValueError(
            "time-series premise violated: earlier contract settled YES but later settled NO"
        )
    # Sides bought on (market_a, market_b) — the pipeline's single side mapping
    side_a, side_b = leg_sides(pair_type)
    receipt = 0.0
    if outcome_a == side_a:
        # Market A resolved to the side bought there: $1 per contract
        receipt += n
    if outcome_b == side_b:
        # Market B resolved to the side bought there: $1 per contract
        receipt += n
    return receipt


def _leg_prices_for(
    pair_type: str, pA: float, nA: float, pB: float, nB: float,
) -> tuple[float, float]:
    """
    Return the per-contract cost of the side bought on each leg, from the four quotes.

    Dict-world mirror of scanner.leg_prices for the backtester, which carries a
    market's quotes as loose floats rather than a CandidatePair: (nA, pB) for a
    same-title pair (NO on market A, YES on market B) and (pA, nB) for a
    time-series pair (YES on the earlier market A, NO on the later market B).
    Every price the backtester sizes, fees or pays out on comes through here,
    so the leg mapping is written exactly once.

    Args:
        pair_type (str): "time_series" or "same_title" — anything but the exact
            string "time_series" is treated as same-title, matching
            scanner.leg_sides.
        pA (float): YES ask of market A, dollars in [0.01, 0.99].
        nA (float): NO ask of market A, dollars in [0.01, 0.99].
        pB (float): YES ask of market B, dollars in [0.01, 0.99].
        nB (float): NO ask of market B, dollars in [0.01, 0.99].

    Returns:
        tuple[float, float]: (price_a, price_b) — the cost of the side bought on
            market A and on market B respectively.
    """
    if pair_type == "time_series":
        return pA, nB
    return nA, pB


# ─── Eligibility prefilter ─────────────────────────────────────────────────────

def _parse_iso_date(value: str | None) -> date | None:
    """
    Parse an ISO 8601 timestamp string to a date, tolerating missing/bad input.

    Shared by _can_ever_enter() and _extract_pairs()'s close-time windowing so
    the "can't parse it → treat as unknown, not an error" behavior is written
    exactly once.

    Args:
        value (str | None): An ISO 8601 timestamp string (e.g. "close_time" or
            "open_time" from a market dict), or None/empty if absent.

    Returns:
        date | None: The parsed date, or None if value is falsy or fails to
            parse. Never raises.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO 8601 timestamp string to a datetime, tolerating missing/bad input.

    Sibling of _parse_iso_date for the callers that need the time-of-day
    component, which the date-only helper throws away: the candlestick fetch
    window is expressed in unix seconds, so truncating a close_time to midnight
    would silently move the window. Same "can't parse it → treat as unknown,
    not an error" contract as _parse_iso_date.

    Args:
        value (str | None): An ISO 8601 timestamp string (e.g. "close_time"
            from a market dict), or None/empty if absent.

    Returns:
        datetime | None: The parsed datetime (naive or aware, exactly as the
            string expressed it), or None if value is falsy or fails to parse.
            Never raises.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _can_ever_enter(m: dict, start_date: date) -> bool:
    """
    Necessary-condition prefilter: could this market possibly appear in any
    entered pair, as either leg, of either pair type?

    _find_entry() only opens a trade at a Monday-09:00-UTC checkpoint inside
    [start_date, min(close_a, close_b) - 1 day], and requires an hourly candle
    at-or-before that Monday for BOTH legs — which requires each market to
    have opened on or before it. So a market whose own
    [open_time, close_time - 1 day] window contains no Monday on/after
    start_date can never satisfy that condition for any partner market,
    regardless of pair_type. Dropping such a market before grouping/pairing
    is therefore provably safe — it would have contributed entry=None to
    every possible pair anyway (see CLAUDE.md for the full invariant).

    Args:
        m (dict): Market dict as produced by historical._market_to_dict().
        start_date (date): Backtest start date — _find_entry() never scans a
            Monday before this.

    Returns:
        bool: True if the market MIGHT be enterable (keep it) — this includes
            the case where open_time or close_time is missing/unparseable,
            since then we can't prove ineligibility (also keeps older cache
            files, written before open_time was added, working correctly —
            just without the speedup). False only when we can prove no
            Monday checkpoint falls in the market's eligible window.
    """
    open_d = _parse_iso_date(m.get("open_time"))
    close_d = _parse_iso_date(m.get("close_time"))
    if open_d is None or close_d is None:
        # Can't prove ineligibility — keep it rather than risk dropping a
        # market that could actually enter a pair.
        return True

    lower = max(open_d, start_date)
    upper = close_d - timedelta(days=1)  # mirrors _find_entry's scan_end
    if lower > upper:
        return False

    # Advance to the first Monday on/after `lower` — identical convention to
    # _monday_timestamps' own advance-to-Monday step, so the two stay in sync.
    d = lower
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d <= upper


# ─── Pair grouping (metadata only, no prices) ─────────────────────────────────

def _pair_key(m: dict) -> str:
    """
    Combined grouping key for a market dict — event title joined with market title.

    Mirrors scanner.pair_key() for the backtester's dict-based market representation.
    The event_title prefix prevents cross-event option-label collisions in MVE
    markets — e.g. two markets both titled "Trump" in unrelated events will have
    different event titles and therefore won't be grouped together.

    Falls back to the bare title when event_title is missing (older cache files
    or non-MVE markets).

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.

    Returns:
        str: "{event_title} | {title}" when event_title is present, otherwise
            just the title (falling back to subtitle, then ticker, if the
            market has no title).
    """
    event_title = m.get("event_title") or ""
    title = m.get("title") or m.get("subtitle") or m.get("ticker", "")
    if not event_title:
        return title
    return f"{event_title} | {title}"


def _group_by_exact_title(markets: list[dict]) -> dict[tuple, list[dict]]:
    """
    Group markets by exact (event_title, title, subtitle) tuple for same-title pair detection.

    Three-element key: the event_title component prevents cross-event option-label
    collisions in MVE markets; (title, subtitle) distinguishes markets within an event.

    Args:
        markets (list[dict]): Market dicts in the compact historical._market_to_dict
            form.

    Returns:
        dict[tuple, list[dict]]: Mapping of (event_title, title, subtitle) ->
            member markets, for groups with >= 2 members and at least one of
            title/subtitle non-empty. Single-member groups are dropped.
    """
    groups: dict = defaultdict(list)
    for m in markets:
        event_title = m.get("event_title") or ""
        title    = m.get("title") or ""
        subtitle = m.get("subtitle") or ""
        if title or subtitle:
            groups[(event_title, title, subtitle)].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_by_normalized_title(markets: list[dict]) -> dict[str, list[dict]]:
    """
    Group markets by date-stripped combined key (event_title + title) for time-series pair detection.

    Args:
        markets (list[dict]): Market dicts in the compact historical._market_to_dict
            form.

    Returns:
        dict[str, list[dict]]: Mapping of normalized (event_title + title) key
            -> member markets, for groups with >= 2 members. A market whose
            key normalizes to an empty string is dropped.
    """
    groups: dict = defaultdict(list)
    for m in markets:
        # _pair_key combines event_title + market title before normalization so that
        # two MVE markets sharing an option label across unrelated events do not collide.
        norm = normalize_title(_pair_key(m))
        if norm:
            groups[norm].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _drop_cross_type_duplicates(candidates: list[dict]) -> list[dict]:
    """
    Drop time-series candidates whose ticker pair was also found as a same-title candidate.

    Mirrors main._dedup_pairs on the live path: when both scanners detect the
    same two markets, the same-title pair is kept because its co-resolution
    model is simpler (identical questions must co-resolve) than the directional
    time-series bet, whose edge depends on the interval discount behind
    config.time_series_profit_prob — see CLAUDE.md, "Pair dedup prefers
    same-title", and the same_title > time_series tie-break in
    strategy.select_portfolio(). Duplicate identity is the frozenset of the two
    tickers, so a pair discovered with its legs in the opposite order still
    matches.

    This has to happen in Pass 1, not Pass 2: Pass 2's ticker-conflict filter
    only prevents both copies being OPEN at the same time. Whenever the
    same-title copy is skipped there for sizing reasons (n < 1, or a
    non-positive expected payoff), it never claims the tickers, so the
    time-series duplicate is entered instead — a candidate the live pipeline
    would never have had at all.

    Args:
        candidates (list[dict]): Pass 1 candidate dicts, each carrying
            "pair_type" ("same_title" or "time_series") and the "mA"/"mB"
            market dicts (which carry "ticker").

    Returns:
        list[dict]: Every same-title candidate plus every time-series candidate
            whose ticker pair is not already covered by a same-title candidate,
            in the input's original order.
    """
    same_title_keys = {
        frozenset((c["mA"]["ticker"], c["mB"]["ticker"]))
        for c in candidates
        if c["pair_type"] == "same_title"
    }
    kept = [
        c for c in candidates
        if c["pair_type"] == "same_title"
        or frozenset((c["mA"]["ticker"], c["mB"]["ticker"])) not in same_title_keys
    ]
    dropped = len(candidates) - len(kept)
    if dropped:
        logging.info(
            "Dropped %d time-series candidate(s) already covered by a same-title candidate",
            dropped,
        )
    return kept


def _extract_pairs(groups: dict) -> list[tuple[dict, dict, str, object]]:
    """
    Return list of (market_a, market_b, canonical_title, group_key) tuples where
    the two markets have different event_tickers. No price filtering at this stage.

    The pair type is NOT a parameter: the shape of each group key (see below)
    decides which sweep applies, and run_backtest() attaches the pair_type
    label to each returned tuple itself.

    Group keys may be:
      - a string (normalized-title group from _group_by_normalized_title), or
      - a 3-tuple (event_title, title, subtitle) from _group_by_exact_title.
    For the 3-tuple form, the display canonical is taken from title-or-subtitle,
    but the FULL 3-tuple (including event_title) is also returned as group_key —
    the one-pair-per-group dedup in run_backtest must key on the full group, not
    just the display title, or two unrelated events sharing an option label
    (e.g. "Trump" in two different events) would collide into a single group
    and silently drop one of the two legitimate pairs.

    For string-keyed (time-series) groups, members are sorted ascending by
    close_time and swept with a two-pointer window bounded by
    MAX_DEADLINE_GAP_DAYS + 1 day of margin: _find_entry() unconditionally
    rejects any time-series pair whose close dates differ by more than
    MAX_DEADLINE_GAP_DAYS, so pairs outside that window can never produce an
    entry and are skipped without ever being materialized as a candidate
    pair. The +1 day margin is slack only (it can never cause a pair within
    the true limit to be skipped) — _find_entry() still applies the exact
    `.days > MAX_DEADLINE_GAP_DAYS` cutoff itself. Members with a missing or
    unparseable close_time are dropped from this sweep (group-local only —
    _group_by_exact_title's same-title groups are untouched), because
    _find_entry() unconditionally requires close_time on both legs and
    returns None immediately without it, regardless of pair type.

    3-tuple-keyed (same-title) groups have no deadline-gap concept, so they
    stay naive — the eligibility prefilter (_can_ever_enter, applied in
    run_backtest before grouping) keeps these groups small in practice.

    Args:
        groups (dict): Mapping of group key -> list of market dicts (the
            compact historical._market_to_dict form). Keys are either a
            normalized-title string (time-series groups) or an
            (event_title, title, subtitle) 3-tuple (same-title groups).

    Returns:
        list[tuple[dict, dict, str, object]]: One (market_a, market_b,
            canonical_title, group_key) tuple per candidate pair, in group
            iteration order. Empty if no group has two members on different
            event_tickers.
    """
    pairs = []
    for key, members in groups.items():
        if isinstance(key, str):
            canon = key
        else:
            # 3-tuple (event_title, title, subtitle) — use title-or-subtitle for display
            canon = key[1] or key[2]

        if len(members) > LARGE_GROUP_WARN_THRESHOLD:
            # Visibility only — not a cap. Confirms the prefilter/windowing
            # above are actually keeping group sizes tractable in practice.
            logging.warning(
                "Pair-extraction group %r has %d members after filtering — "
                "still large; verify the eligibility prefilter is firing as expected",
                canon, len(members),
            )

        seen: set[frozenset] = set()

        if isinstance(key, str):
            # Time-series: sort by close_time and sweep only the pairs within
            # the deadline-gap window (see docstring above) instead of the
            # naive O(n^2) double loop over the whole group.
            dated = [(_parse_iso_date(m.get("close_time")), m) for m in members]
            dated = [(d, m) for d, m in dated if d is not None]
            dated.sort(key=lambda pair: pair[0])
            margin = timedelta(days=MAX_DEADLINE_GAP_DAYS + 1)
            n = len(dated)
            for i in range(n):
                close_a, mA = dated[i]
                for j in range(i + 1, n):
                    close_b, mB = dated[j]
                    if close_b - close_a > margin:
                        # Sorted ascending by close_time — every further j is
                        # at least this far from mA, so nothing later qualifies.
                        break
                    if mA["event_ticker"] == mB["event_ticker"]:
                        continue
                    pair_key = frozenset([mA["ticker"], mB["ticker"]])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((mA, mB, canon, key))
        else:
            # Same-title: no deadline-gap constraint, stays naive.
            for i, mA in enumerate(members):
                for mB in members[i + 1:]:
                    if mA["event_ticker"] == mB["event_ticker"]:
                        continue
                    pair_key = frozenset([mA["ticker"], mB["ticker"]])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((mA, mB, canon, key))
    return pairs


# ─── Entry point detection from candlestick data ──────────────────────────────

def _monday_timestamps(start_date: date, end_date: date) -> list[int]:
    """
    Generate a list of Unix timestamps for every Monday in the given date range.

    Each timestamp corresponds to 09:00 UTC on the Monday. The backtest scans
    these weekly checkpoints to simulate the bot's Monday morning trading schedule.

    Args:
        start_date (date): First date of the scan range (inclusive). The function
            advances to the first Monday on or after this date.
        end_date (date): Last date of the scan range (inclusive).

    Returns:
        list[int]: Unix timestamps (09:00 UTC) for every Monday in [start_date, end_date].
            Returns an empty list if start_date > end_date or no Monday falls in range.
    """
    d = start_date
    # Advance to the first Monday (weekday() == 0) if start_date is not already one
    while d.weekday() != 0:
        d += timedelta(days=1)
    ts_list = []
    while d <= end_date:
        ts_list.append(int(datetime(d.year, d.month, d.day, 9, 0, tzinfo=UTC).timestamp()))
        d += timedelta(weeks=1)
    return ts_list


def _candle_at_or_before(candles: list[dict], ts: int) -> dict | None:
    """
    Find the most recent candlestick at or before a given Unix timestamp.

    Candles are assumed to be sorted ascending by their "ts" field. Returns the
    last candle whose "ts" is <= ts, or None if no such candle exists. ("ts" is
    derived from the API's end_period_ts when the candle is fetched — see
    historical.fetch_candlesticks — so this is equivalently "at or before the
    candle's period end", just read off the dict's own key rather than the
    wire field name.) This is used to read the closing price on or before a
    given Monday snapshot.

    Args:
        candles (list[dict]): List of candle dicts with a "ts" key (unix timestamp).
            Must be sorted ascending by "ts".
        ts (int): The target Unix timestamp to search at or before.

    Returns:
        Optional[dict]: The last candle dict with ts <= the target, or None if all
            candles are after the target timestamp or the list is empty.
    """
    result = None
    for c in candles:
        if c["ts"] <= ts:
            result = c
        else:
            # Candles are sorted ascending, so once we exceed ts we can stop
            break
    return result


def _find_entry(
    candles_a: list[dict],
    candles_b: list[dict],
    mA: dict,
    mB: dict,
    pair_type: str,
    start_date: date,
    max_horizon_days: int | None = None,
) -> dict | None:
    """
    Find the first Monday where a potential pair was tradeable at the required threshold.

    Scans weekly Monday snapshots up to (not including) the day before the
    earlier of the two market close dates. The scan's start is the LATER of
    the backtest start date and one calendar year before that end point — the
    one-year figure bounds how far back the window can reach, it does not
    describe where the window ends. At each Monday, reads the candlestick
    prices, applies the price gap, price-sum, and fee filters from the live
    trading logic, and returns the entry data for the first qualifying week.

    Direction rules mirror the live scanner exactly:
      - time_series: market A is fixed as the EARLIER-closing contract, and an
        entry requires pB − pA >= the deadline-gap-tiered threshold from
        min_price_diff_for_gap (15% for gaps <= 15 days, 30% for 16-30 days) —
        the LATER contract priced higher by at least the tier is the anomaly
        the strategy disputes (the market implies an outsized probability that
        the event first happens between the two deadlines). A pricier earlier
        contract is never a candidate. The legs are YES on A at pA and NO on B
        at nB, so the price-sum ceiling (pA + nB <= 1 − threshold) and the fee
        check are applied to those two leg prices — never to (nA, pB), which
        are reporting-only quotes for this pair type. The deadline gap driving
        the tier (and the 30-day cutoff) is the ABSOLUTE timedelta.days on the
        two close_time datetimes, exactly as scanner.deadline_gap_days computes
        it (order-independent) — not calendar-date subtraction, which counts a
        day boundary the live path does not.
      - same_title: market A is canonicalized per Monday as the more expensive
        side (the two contracts ask the identical question, so direction is
        price-only); the legs are NO on market A (nA) and YES on market B (pB),
        and the entry requires pA − pB >= SAME_TITLE_MIN_PRICE_DIFF with the
        ceiling and fee check on (nA, pB).
    In both cases the traded pair of prices comes from _leg_prices_for, and
    both of them must be live [0.01, 0.99] quotes.

    Scanning stops at the earlier close date (not the later one) because after
    the first market closes, the pair is no longer open for entry.

    Args:
        candles_a (list[dict]): Hourly candlestick dicts for market A (sorted by ts).
        candles_b (list[dict]): Hourly candlestick dicts for market B (sorted by ts).
        mA (dict): Market A metadata dict (with "close_time", "result", etc.).
        mB (dict): Market B metadata dict (with "close_time", "result", etc.).
        pair_type (str): "time_series" or "same_title" — controls price gap threshold
            and deadline gap check.
        start_date (date): Backtest start date; no entry is recorded before this date.
        max_horizon_days (int | None): Optional cap on how far a checkpoint's Monday
            may be from the later-closing leg's close date. A Monday where
            (later close date − Monday) exceeds this is skipped (not rejected
            outright — a later Monday closer to the close dates may still
            qualify). None means no cap (default), matching live-path semantics.

    Returns:
        Optional[dict]: A dict with keys "entry_date" (date), "pA" (float), "pB"
            (float), "nA" (float), "nB" (float), "mA" (dict), "mB" (dict),
            "gap_days" (int | None) for the first qualifying Monday — all four
            quotes of the canonicalized A and B (YES ask and NO ask of each),
            of which _leg_prices_for picks the two that were actually traded.
            "gap_days" is the loop-invariant deadline gap the price tier and
            the MAX_DEADLINE_GAP_DAYS cutoff were applied on, carried out for
            reporting so a report cannot bucket a pair under a gap it was not
            filtered by; it is None for same_title, which has no deadline-gap
            concept.
            Returns None if no qualifying Monday was found in the scan window, or
            if either leg's close_time is missing or unparseable (no scan window
            can be derived, so the pair is simply not enterable).
    """
    # Both markets must have a PARSEABLE close_time; without one we can't
    # determine the scan window. A malformed timestamp is treated exactly like
    # a missing one (the file-wide "can't parse it = unknown, not an error"
    # convention) rather than raising out of the caller's candidate loop.
    close_a = _parse_iso_date(mA.get("close_time"))
    close_b = _parse_iso_date(mB.get("close_time"))
    if close_a is None or close_b is None:
        return None

    # Scan up to (but not including) the day the earlier market closes —
    # after that, the pair is no longer fully open for entry
    scan_end   = min(close_a, close_b) - timedelta(days=1)
    # Look back at most 1 year from scan_end to keep the scan window manageable
    scan_start = max(start_date, scan_end - timedelta(days=365))

    # A single-day window (scan_start == scan_end) is still a valid scan window
    if scan_start > scan_end:
        return None

    if pair_type == "time_series":
        # Live-scanner invariant: market A is the EARLIER-closing contract.
        # Never swap by price — the trade only exists when the LATER contract
        # is priced higher (checked per Monday below).
        if close_b < close_a:
            mA, mB = mB, mA
            candles_a, candles_b = candles_b, candles_a
            close_a, close_b = close_b, close_a
        # Deadline gap is loop-invariant: a wider gap carries more genuine
        # in-between probability mass, so 16-30 day gaps demand the larger
        # tier and gaps beyond MAX_DEADLINE_GAP_DAYS are never disputed.
        # Measure it on the close_time DATETIMES, not the dates parsed above:
        # timedelta.days floors, while calendar-date subtraction counts day
        # boundaries, so the two disagree by up to a day whenever the closes
        # straddle midnight (2026-02-01T23:00Z vs 2026-02-17T01:00Z is gap 15
        # live but 16 by date). That one day flips both the tier boundary and
        # the 30-day cutoff, so a backtest that is supposed to replay the live
        # strategy must use the live arithmetic.
        dt_a = _parse_iso_datetime(mA.get("close_time"))
        dt_b = _parse_iso_datetime(mB.get("close_time"))
        try:
            # Identical arithmetic to scanner.deadline_gap_days (used by
            # find_time_series_pairs / _pair_max_sum): absolute timedelta.days
            # on tz-aware datetimes, so the result is order-independent
            gap_days = abs(dt_b - dt_a).days
        except TypeError:
            # A naive/aware mix (only reachable from a hand-edited cache) can't
            # be subtracted; fall back to the dates rather than raising, per
            # this file's "can't parse it = unknown, not an error" convention
            gap_days = abs(close_b - close_a).days
        if gap_days > MAX_DEADLINE_GAP_DAYS:
            return None
        # Tier the required price gap by deadline distance (15% for gaps
        # <= 15 days, 30% for 16-30 days) — the same tiering, computed off the
        # same gap arithmetic, as scanner.find_time_series_pairs
        threshold = min_price_diff_for_gap(gap_days)
    else:
        # same_title pairs have no deadline-gap concept — flat 5% threshold
        threshold = SAME_TITLE_MIN_PRICE_DIFF
        # No deadline gap to report for same_title. The time_series branch
        # above assigns gap_days on both its try and except paths, so this is
        # the only branch where the name would otherwise be unbound below.
        gap_days = None

    for ts in _monday_timestamps(scan_start, scan_end):
        entry_date = datetime.fromtimestamp(ts, tz=UTC).date()

        # Optional opt-in bet-horizon cap: skip checkpoints where the
        # later-closing leg (close_b, since time_series always keeps
        # close_b >= close_a and same_title has no ordering) would close
        # further out than max_horizon_days from THIS simulated checkpoint —
        # a cheap comparison done before touching candle data
        if max_horizon_days is not None and (max(close_a, close_b) - entry_date).days > max_horizon_days:
            continue

        # Read the closing prices at this Monday snapshot
        ca = _candle_at_or_before(candles_a, ts)
        cb = _candle_at_or_before(candles_b, ts)
        if ca is None or cb is None:
            continue

        try:
            p_a_raw = float(ca["yes_ask_close"])
            p_b_raw = float(cb["yes_ask_close"])
            n_a_raw = float(ca["no_ask_close"])
            n_b_raw = float(cb["no_ask_close"])
        except (ValueError, TypeError):
            continue

        # Skip settled or illiquid candles (prices at the extreme ends of the
        # range). Both YES asks are checked regardless of pair type: for
        # time_series pB is not a leg price, but it still drives the gap and
        # the profit model, so a dead quote there is just as disqualifying.
        if not (0.01 <= p_a_raw <= 0.99 and 0.01 <= p_b_raw <= 0.99):
            continue

        if pair_type == "time_series":
            # A stays the earlier-closing market — no price canonicalization.
            # The anomaly is the LATER contract priced higher: gap = pB - pA.
            mA_i, mB_i = mA, mB
            pA, pB, nA, nB = p_a_raw, p_b_raw, n_a_raw, n_b_raw
            gap = pB - pA
        else:
            # same_title: canonicalize per iteration so the swap never leaks to
            # the next Monday. Market A is the more expensive side this week,
            # and the anomaly is its YES ask exceeding B's: gap = pA - pB.
            if p_a_raw >= p_b_raw:
                mA_i, mB_i = mA, mB
                pA, pB, nA, nB = p_a_raw, p_b_raw, n_a_raw, n_b_raw
            else:
                mA_i, mB_i = mB, mA
                pA, pB, nA, nB = p_b_raw, p_a_raw, n_b_raw, n_a_raw
            gap = pA - pB

        # Enforce the minimum price gap for this pair type (directional in
        # both cases — the gap above is signed, never an absolute value)
        if gap < threshold:
            continue

        # The two prices actually paid — (nA, pB) for same_title, (pA, nB) for
        # time_series — via the module's single leg mapping
        price_a, price_b = _leg_prices_for(pair_type, pA, nA, pB, nB)

        # Both traded leg prices must be live (0.01–0.99) quotes — mirrors the
        # live pipeline's (0, 1) price validation in compute_trade. This adds
        # the NO-leg check (nA for same_title, nB for time_series) to the YES
        # asks banded above. Note the candle no_ask_close is already clamped
        # into [0.01, 0.99] by historical.fetch_candlesticks, so on candle
        # data this cannot fire for the NO leg — it is parity, not a filter.
        if not (0.01 <= price_a <= 0.99 and 0.01 <= price_b <= 0.99):
            continue

        # Live orderbook-depth parity: enrich_with_orderbook_prices only keeps
        # contracts whose combined LEG price leaves the required gap
        # (price_a + price_b <= 1 - threshold) — apply the same cut to candle entries
        if price_a + price_b > 1.0 - threshold:
            continue

        # Check that the gross spread on the leg prices exceeds the continuous fee estimate
        if (1.0 - price_a - price_b) <= fee_per_pair_approx(price_a, price_b):
            continue

        return {
            "entry_date": entry_date,
            "pA": pA, "pB": pB, "nA": nA, "nB": nB,
            "mA": mA_i, "mB": mB_i,
            # The gap the tier above was selected from, carried out so the
            # calibration report buckets each pair under exactly the gap it
            # was filtered by. None for same_title. Reporting only — every
            # decision this value drives was already made above.
            "gap_days": gap_days,
        }

    return None


# ─── Candlestick fetching ─────────────────────────────────────────────────────

def _fetch_candles_parallel(
    hist_client: Any,
    needed_tickers: dict[str, dict],
    start_date: date,
    use_cache: bool,
) -> dict[str, list[dict]]:
    """
    Fetch the hourly candlestick series for every needed ticker, in parallel.

    One HTTP fetch per ticker, spread across CANDLESTICK_FETCH_MAX_WORKERS
    threads. Parallelism is result-neutral here for three reasons: the returned
    mapping is only ever read by key (never iterated), so completion order
    cannot matter; each ticker's disk cache path is derived from its ticker, so
    two workers can never write the same file (historical._save_json_cache is an
    atomic tmp+replace, but its tmp name is derived from the destination, so a
    shared path WOULD still collide — path uniqueness stays load-bearing); and
    each fetch is
    an independent read-only GET whose retry/backoff already lives per-call
    inside api_call_with_retry.

    Markets with no close_time get an empty series without any HTTP call, which
    is what the sequential version did — there is no window to request. A
    present-but-unparseable close_time is handled the same way (with a warning):
    it is a data defect in one market, not a reason to abort the whole run from
    the main thread before any worker starts.

    Worker exceptions are deliberately NOT caught: fetch_candlesticks already
    fail-softs network errors to an empty list internally, so anything that
    still escapes is a real defect (e.g. a market that should have been
    prefiltered out) and must surface rather than be silently degraded into
    "this ticker has no prices".

    Args:
        hist_client (Any): Historical KalshiClient, shared across worker
            threads (the same pattern historical.py's fetch pools use).
        needed_tickers (dict[str, dict]): Ticker -> market dict, for exactly
            the markets appearing in at least one candidate pair.
        start_date (date): Start of the backtest window; the fetch window's
            lower bound, identical for every ticker.
        use_cache (bool): Passed through to fetch_candlesticks — whether the
            per-ticker disk cache may be reused.

    Returns:
        dict[str, list[dict]]: Ticker -> candle list (keys: ts, yes_ask_close,
        no_ask_close). Every key of needed_tickers is present; the value is an
        empty list for markets with no usable (missing or unparseable) close_time.

    Raises:
        Exception: Whatever a worker's fetch_candlesticks call raises, after
            the pool has been torn down without draining its queue.
    """
    candles_by_ticker: dict[str, list[dict]] = {}

    # open_ts: start of the backtest window. Depends only on start_date, so it
    # is identical for every ticker and computed once.
    open_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                           tzinfo=UTC).timestamp())

    # Split the work first: no-close_time markets resolve without any HTTP, so
    # they never occupy a worker slot.
    work: list[tuple[str, int]] = []
    for ticker, m in needed_tickers.items():
        close_time = m.get("close_time")
        if not close_time:
            candles_by_ticker[ticker] = []
            continue
        close_dt = _parse_iso_datetime(close_time)
        if close_dt is None:
            # A malformed timestamp yields no requestable window, exactly like a
            # missing one — resolve the ticker to an empty series instead of
            # raising on the main thread and killing a multi-hour run. Logged
            # (not silent) because an empty series is otherwise indistinguishable
            # from "this market genuinely has no price history".
            logging.warning(
                "Unparseable close_time %r for %s — fetching no candles for it",
                close_time, ticker,
            )
            candles_by_ticker[ticker] = []
            continue
        # close_ts: one day past market close to include the final candle
        close_ts = int(close_dt.timestamp()) + _DAY_SECONDS
        work.append((ticker, close_ts))

    if work:
        with ThreadPoolExecutor(max_workers=CANDLESTICK_FETCH_MAX_WORKERS) as pool:
            # Returns list[dict] with keys: ts (unix int), yes_ask_close (float),
            # no_ask_close (float) — cached per ticker, so a second run is much faster
            futures = {
                pool.submit(fetch_candlesticks, hist_client, ticker,
                            open_ts, close_ts, use_cache): ticker
                for ticker, close_ts in work
            }
            done = 0
            try:
                for future in as_completed(futures):
                    candles_by_ticker[futures[future]] = future.result()
                    done += 1
                    if done % 50 == 0:
                        # Denominator is len(work), not len(needed_tickers): tickers
                        # with a missing/unparseable close_time never enter `work`
                        # (they're resolved to [] above without a worker), so
                        # `done` can never reach len(needed_tickers) whenever any
                        # were skipped.
                        logging.info("  Candlestick progress: %d / %d",
                                     done, len(work))
            except BaseException:
                # Same tear-down as historical.py's fetch pools: without it the
                # executor's __exit__ drains every still-queued ticker (hours of
                # work) before the error ever reaches the caller.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    return candles_by_ticker


# ─── Main backtest loop ───────────────────────────────────────────────────────

def run_backtest(
    hist_client: Any,
    live_client,
    start_date: date = date(2024, 1, 1),
    initial_balance: float = 10_000.0,
    use_cache: bool = True,
    max_horizon_days: int | None = None,
) -> tuple[list[BacktestTrade], pd.DataFrame]:
    """
    Replay both pair strategies on all settled Kalshi markets from start_date.

    Algorithm:
      1. Fetch all settled markets since start_date.
      2. Drop markets that provably can never enter any pair (_can_ever_enter).
      3. Group into potential time-series and same-title pairs (metadata only).
      4. Fetch hourly candlesticks for every ticker appearing in a potential
         pair, in parallel across CANDLESTICK_FETCH_MAX_WORKERS threads.
      5. Find the first Monday where the pair was tradeable at the threshold;
         exclude (and count, with one summary WARNING) any time-series
         candidate whose settlement was earlier-YES/later-NO — impossible for
         a cumulative-deadline pair, so a premise violation rather than a
         payout; keep only the best entry per title group (live
         one-pair-per-group rule), then drop any time-series candidate whose
         ticker pair was also found as a same-title candidate (live
         main._dedup_pairs rule).
      6. Walk entries chronologically with a running cash balance: Kelly-size
         every candidate of an entry date against that checkpoint's opening
         balance, admit it only while its fee-inclusive cost still fits the
         running cash (mirroring main._run_prod + strategy.select_portfolio),
         and record actual P&L from settlement outcomes.
      7. Build an equity curve from the trade timeline.

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        initial_balance (float): Simulated starting cash balance in dollars.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap mirroring
            scanner.filter_markets_within_horizon on the live path, but relative
            to each simulated checkpoint rather than real-world now: at a given
            Monday checkpoint, a pair can only enter if the later-closing leg
            closes within max_horizon_days of THAT checkpoint. None (default)
            applies no cap, matching current behavior. Passed straight through
            to _find_entry() for each candidate pair.

    Returns:
        tuple[list[BacktestTrade], pd.DataFrame]: (trades, equity_df).
            trades is one BacktestTrade per entered pair, in entry-date order
            (empty if none were ever entered). equity_df has columns
            [date, portfolio_value, daily_return], one row per day, flat at
            initial_balance if trades is empty.

    Raises:
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — this is
            treated as a real defect (a market that should never have reached
            this stage), not degraded into "no price history".

    Note:
        Before any network call, this function checks whether [start_date,
        today] contains at least one Monday 09:00 UTC checkpoint (the
        only kind _find_entry() can ever act on). If not, no trade can ever
        be entered regardless of what the fetch would return, so the fetch is
        skipped entirely and this returns the same empty-result shape as the
        zero-trade path ([], an equity curve flat at initial_balance) with a
        WARNING logged.
    """
    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    # Feasibility pre-check, BEFORE any network call: a trade can only ever be
    # entered at a Monday 09:00 UTC checkpoint. If [start_date, today] contains
    # no Monday at all, no trade can ever be entered by construction, no matter
    # what the fetch returns — this is exactly the class of run that burned ~59
    # minutes fetching 9.2M records into a 2-byte assembled cache (0 markets
    # survived the Monday-eligibility prefilter). Detecting it up front skips
    # the fetch entirely instead of discovering it only after paying for it.
    #
    # The window end is today, NOT yesterday. _find_entry()'s per-pair scan
    # ends at (min(close_a, close_b) - 1 day), but a market that settled early
    # can still carry a close_time in the future, which makes TODAY a
    # legitimate checkpoint for that pair. This guard exists only to catch the
    # structurally-impossible case, so it must be strictly conservative: an
    # over-tight end date would wrongly skip a real run (e.g. today is Monday
    # and start_date is within the last week).
    feasibility_end = date.today()
    if not _monday_timestamps(start_date, feasibility_end):
        logging.warning(
            "No Monday 09:00 UTC entry checkpoint exists in [%s, %s] — no "
            "trade can ever be entered; skipping the fetch entirely",
            start_date, feasibility_end,
        )
        # Same empty-result shape the zero-trade path at the bottom of this
        # function already produces, so backtest.py / generate_dashboard need
        # no changes to handle this early-exit.
        return [], _build_equity_curve([], start_date, initial_balance)

    # Fetch all settled markets from start_date onward (uses disk cache if
    # available). The eligibility predicate below is handed to the fetch so
    # ineligible markets are dropped during assembly rather than materialized
    # and cached first — result-neutral, since the very next statement would
    # discard exactly those records anyway, but it keeps peak memory and the
    # assembled cache proportional to what the backtest can actually use.
    # SETTLED_PREFILTER_CACHE_TAG keys that cache to _can_ever_enter's current
    # semantics and MUST be bumped if this predicate changes.
    markets = fetch_all_settled_markets(
        hist_client, live_client, start_date, use_cache,
        prefilter=lambda m: _can_ever_enter(m, start_date),
        prefilter_tag=SETTLED_PREFILTER_CACHE_TAG,
    )
    logging.info("Total settled markets to analyze: %d", len(markets))

    # Necessary-condition prefilter: drop markets whose [open_time, close_time
    # - 1 day] window contains no Monday checkpoint on/after start_date, since
    # _find_entry() can then never enter them as either leg of either pair
    # type. This is what makes grouping/pairing tractable at current Kalshi
    # volumes (hourly/intraday ladders are the overwhelming majority of
    # settled markets and almost never span a scannable Monday).
    #
    # Retained even though the same predicate was passed into the fetch above:
    # it is idempotent, it costs one pass, and it keeps this guarantee local to
    # the code that depends on it (a cached unfiltered list, a caller that
    # skips the prefilter argument, or a future fetch path would otherwise
    # reach the O(n^2) pairing unfiltered).
    eligible_markets = [m for m in markets if _can_ever_enter(m, start_date)]
    logging.info(
        "Eligibility prefilter: skipping %d/%d markets that cannot appear in any tradeable pair",
        len(markets) - len(eligible_markets), len(markets),
    )
    markets = eligible_markets

    # Group settled markets into potential pairs using the same logic as the live scanner
    ts_groups    = _group_by_normalized_title(markets)
    same_groups  = _group_by_exact_title(markets)
    ts_pairs     = _extract_pairs(ts_groups)
    same_pairs   = _extract_pairs(same_groups)

    logging.info("Potential pairs: %d time-series, %d same-title", len(ts_pairs), len(same_pairs))

    # Collect only the tickers that actually appear in a potential pair to avoid
    # fetching candlesticks for thousands of unrelated markets
    needed_tickers: dict[str, dict] = {}
    for mA, mB, _, _ in ts_pairs + same_pairs:
        needed_tickers[mA["ticker"]] = mA
        needed_tickers[mB["ticker"]] = mB

    logging.info("Fetching candlesticks for %d markets (cached per ticker)...", len(needed_tickers))

    # Fetch hourly candlestick price series for all needed tickers, in parallel
    # across tickers — one independent read-only GET each, cached per ticker so
    # a second run is much faster. Sequentially this loop dominated the whole
    # backtest (~4.3 tickers/sec live-measured).
    candles_by_ticker = _fetch_candles_parallel(
        hist_client, needed_tickers, start_date, use_cache
    )

    logging.info("Candlestick fetch complete.")

    # ── Pass 1: collect all tradeable entries (no sizing, no conflict filter) ──

    # Combine both pair types for the scan loop
    all_pairs = [(p, "time_series") for p in ts_pairs] + [(p, "same_title") for p in same_pairs]
    candidates = []
    # Time-series candidates that settled earlier-YES/later-NO — impossible
    # for a cumulative-deadline pair, so the grouping admitted a
    # non-cumulative one. Counted here, reported once after the loop.
    premise_violations = 0

    for (mA_orig, mB_orig, canon, group_key), pair_type in all_pairs:
        candles_a = candles_by_ticker.get(mA_orig["ticker"], [])
        candles_b = candles_by_ticker.get(mB_orig["ticker"], [])

        # Find the first Monday where this pair was tradeable at the threshold
        # prices — max_horizon_days (if set) restricts entries to checkpoints
        # close enough to the legs' close dates
        entry = _find_entry(
            candles_a, candles_b, mA_orig, mB_orig, pair_type, start_date,
            max_horizon_days=max_horizon_days,
        )
        if entry is None:
            continue

        # Unpack entry — mA/mB may have been swapped inside _find_entry to canonicalize
        mA = entry["mA"]
        mB = entry["mB"]
        pA, pB, nA, nB = entry["pA"], entry["pB"], entry["nA"], entry["nB"]
        entry_date = entry["entry_date"]

        # The two prices actually paid — (nA, pB) for same_title, (pA, nB) for
        # time_series — the backtester's mirror of scanner.leg_prices
        price_a, price_b = _leg_prices_for(pair_type, pA, nA, pB, nB)

        # ── Kelly fraction (sizing happens in Pass 2 against the checkpoint) ──
        # Compute the net spread on the leg prices after the continuous fee approximation
        net_spread = (1.0 - price_a - price_b) - fee_per_pair_approx(price_a, price_b)
        profit_ratio_entry = net_spread / (price_a + price_b) if net_spread > 0 else 0.0

        # Probability model. time_series: the discounted market-implied
        # in-between mass, 1 - k * (pB - pA), from config.time_series_profit_prob
        # — the single definition strategy._kelly_p and dashboard._kelly_fraction
        # also call, so the three can never drift (called directly by name here;
        # a test pins that). same_title: the fixed co-resolution prior.
        p = time_series_profit_prob(pA, pB) if pair_type == "time_series" else SAME_TITLE_CO_RESOLVE_PROB
        q = 1.0 - p

        # Kelly formula: f* = p - q/b; negative means no edge
        kelly_f = (p - q / profit_ratio_entry) if profit_ratio_entry > 0 else -1.0
        if kelly_f <= 0:
            # Kelly fraction is non-positive — the pair has no positive expected value
            continue
        # Cap at BUDGET_FRACTION (20%) to avoid over-concentration
        kelly_f_capped = min(BUDGET_FRACTION, kelly_f)

        # Skip pairs where the settlement result is missing or non-binary
        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        # Earlier YES with later NO cannot happen for a cumulative-deadline
        # pair (YES by the earlier deadline implies YES by the later one), so
        # this pair was not one: skip it — never traded, never paid — and
        # count it for the summary WARNING after the loop. _settlement_receipt
        # would raise on this cell; excluding it here keeps that guard
        # unreachable from the pipeline.
        if pair_type == "time_series" and outcome_a == "yes" and outcome_b == "no":
            premise_violations += 1
            continue

        # Determine the exit date as the later of the two settlement timestamps.
        # Belt-and-braces: fetch_all_settled_markets only keeps records that
        # carry a settlement_ts, so a miss here should be impossible — but an
        # unparseable value must not raise out of the candidate loop, and a
        # candidate with no exit date can't be cash-simulated at all (its
        # capital would be released on an invented day), so it is skipped.
        exit_date_a = _parse_iso_date(mA.get("settlement_ts"))
        exit_date_b = _parse_iso_date(mB.get("settlement_ts"))
        if exit_date_a is None or exit_date_b is None:
            logging.debug(
                "Skipping candidate %s/%s: missing or unparseable settlement_ts",
                mA.get("ticker"), mB.get("ticker"),
            )
            continue
        exit_date   = max(exit_date_a, exit_date_b)

        holding_days = max(1, (exit_date - entry_date).days)

        # Entry-time priority metric: the expected return normalized to 30 days
        # using only information available at entry (entry prices and the market
        # close dates). Sorting Pass 2 by REALIZED returns would leak settlement
        # outcomes into trade selection (look-ahead bias).
        # Belt-and-braces again: _find_entry already required a parseable
        # close_time on both legs to derive its scan window, so neither guard
        # can fire in practice — but a bare [...] index plus an unguarded
        # fromisoformat would turn any future data drift into a mid-run crash.
        close_a_d = _parse_iso_date(mA.get("close_time"))
        close_b_d = _parse_iso_date(mB.get("close_time"))
        if close_a_d is None or close_b_d is None:
            logging.debug(
                "Skipping candidate %s/%s: missing or unparseable close_time",
                mA.get("ticker"), mB.get("ticker"),
            )
            continue
        expected_days = max(1, (max(close_a_d, close_b_d) - entry_date).days)
        entry_monthly_ratio = profit_ratio_entry * 30.0 / expected_days

        # Use title > subtitle > ticker as the display label for each market
        title_a = mA.get("title") or mA.get("subtitle") or mA.get("ticker", "")
        title_b = mB.get("title") or mB.get("subtitle") or mB.get("ticker", "")

        candidates.append({
            "pair_type": pair_type,
            "canon": canon,
            "group_key": group_key,
            "mA": mA, "mB": mB,
            "pA": pA, "pB": pB, "nA": nA, "nB": nB,
            # Leg prices Pass 2 sizes, fees and pays out on
            "price_a": price_a, "price_b": price_b,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "kelly_f_capped": kelly_f_capped,
            "entry_monthly_ratio": entry_monthly_ratio,
            "holding_days": holding_days,
            "title_a": title_a,
            "title_b": title_b,
            # Carried straight from _find_entry (None for same_title) so the
            # recorded trade reports the same gap the tier was chosen from
            "gap_days": entry["gap_days"],
        })

    if premise_violations:
        # Summary-warning idiom (silent at zero): the only signal that the
        # normalized-title grouping admitted non-cumulative pairs — the live
        # scanner cannot detect this from prices.
        logging.warning(
            "Excluded %d time-series candidate(s) whose settlement violated the "
            "cumulative-deadline premise (earlier YES, later NO) — the "
            "normalized-title group likely mixes snapshot markets ('on <date>') "
            "with cumulative ones ('by <date>')",
            premise_violations,
        )

    # Keep only the single best candidate per title group — mirrors the live
    # scanners' CONCEPT (one pair per normalized-title / exact-title group so
    # the portfolio isn't flooded with near-identical correlated positions),
    # not their tie-break rule: the live scanners rank by tradeable-first then
    # largest price gap (scanner.py's group_pairs.sort), while this ranks by
    # largest entry_monthly_ratio — a different quantity, chosen here because
    # entry-time expected return is what Pass 2 below sizes and orders on.
    # Dedup on the FULL group_key (which includes event_title for same_title
    # groups), not the display-only canon — two unrelated events sharing an
    # option label (e.g. "Trump" in two different events) have the same canon
    # but distinct group_keys, and must remain two separate candidates.
    best_by_group: dict = {}
    for c in candidates:
        key = (c["pair_type"], c["group_key"])
        cur = best_by_group.get(key)
        if cur is None or c["entry_monthly_ratio"] > cur["entry_monthly_ratio"]:
            best_by_group[key] = c
    candidates = list(best_by_group.values())

    # Cross-type dedup, mirroring main._dedup_pairs on the live path: the same
    # two tickers can be discovered by BOTH groupings, and the live pipeline
    # keeps only the same-title copy (simpler co-resolution model). Must run
    # here rather than relying on Pass 2's ticker-conflict filter, which only
    # stops both copies being open at once — not the time-series copy being
    # entered whenever the same-title one is skipped for sizing reasons.
    candidates = _drop_cross_type_duplicates(candidates)

    # Chronological order for the cash simulation; within one entry date, take
    # the best ENTRY-TIME expected return first (same_title preferred at ties,
    # matching strategy.select_portfolio)
    candidates.sort(
        key=lambda c: (c["entry_date"], -c["entry_monthly_ratio"], c["pair_type"] != "same_title"),
    )

    # ── Pass 2: chronological cash-constrained greedy selection ───────────────
    # Walk entries in date order, maintaining a running cash balance. Sizing
    # mirrors main._run_prod exactly: every candidate on a given entry date
    # (one Monday checkpoint = one live run) is Kelly-sized against that
    # checkpoint's OPENING balance — the cash after that day's settlement
    # receipts have returned, i.e. what verify_auth would report at the top
    # of the run — and then admitted greedily, in the same order
    # select_portfolio uses, only while its fee-inclusive cost still fits the
    # RUNNING cash (select_portfolio's `total_cost_with_fees > available`).
    # Sizing later same-day trades off the running cash instead (the old
    # behaviour) made every trade after the first on a busy Monday smaller
    # than live would make it. Settlement receipts return to cash on their
    # exit dates. A ticker-conflict
    # filter mirrors the live
    # bot's rule precisely: at most one ACTIVE position per ticker. Live, that
    # rule comes from scanner.get_held_tickers(), which queries positions with
    # count_filter="position" — so a ticker leaves the held set once its market
    # settles and may be entered again. The backtest therefore holds a ticker
    # only until its trade's exit date, released below on the same schedule as
    # the settlement receipts.
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()
    cash = initial_balance
    # Opening balance of the checkpoint currently being walked — the Kelly
    # base for every candidate entering on that date (see comment above).
    checkpoint_date: date | None = None
    checkpoint_cash = cash
    pending_exits: list[tuple[date, float]] = []  # (exit_date, settlement receipt)
    # (exit_date, ticker) for every leg of a still-open trade — the release
    # ledger for active_tickers, kept alongside pending_exits so cash and
    # ticker availability are always freed on exactly the same day.
    active_until: list[tuple[date, str]] = []

    for c in candidates:
        d = c["entry_date"]
        # Release settlement receipts from trades that exited on or before this entry
        cash += sum(amt for ed, amt in pending_exits if ed <= d)
        pending_exits = [(ed, amt) for ed, amt in pending_exits if ed > d]
        # Release the tickers of those same settled trades — the position is
        # closed, so (as live) the ticker is no longer blocked. Set difference
        # is safe because the conflict filter below guarantees a ticker is in
        # at most one open trade at a time.
        # NOTE the `<= d` (not `< d`): a trade exiting ON date d frees its
        # tickers for a later candidate entering that same date d. This is a
        # deliberate symmetry with the cash rule directly above, which likewise
        # returns that trade's settlement receipt on d — both resources are
        # freed on exactly the same day, so a same-day re-entry is funded and
        # unblocked together rather than one without the other.
        active_tickers.difference_update(tk for ed, tk in active_until if ed <= d)
        active_until = [(ed, tk) for ed, tk in active_until if ed > d]

        if d != checkpoint_date:
            # First candidate of a new checkpoint: receipts for this date have
            # just been returned above, so this is the balance a live run
            # starting today would read and size everything against.
            checkpoint_date = d
            checkpoint_cash = cash

        mA, mB = c["mA"], c["mB"]
        # Skip if either ticker is still committed to a trade that hasn't settled
        if mA["ticker"] in active_tickers or mB["ticker"] in active_tickers:
            continue

        # The traded leg prices (see _leg_prices_for) — every dollar figure
        # below is computed on these, never on the reporting-only quotes
        price_a, price_b = c["price_a"], c["price_b"]

        # Kelly sizing against the checkpoint's opening balance (live:
        # compute_trade(pair, balance_cents) with one balance for the run)
        budget = checkpoint_cash * c["kelly_f_capped"]
        n = int(budget / (price_a + price_b))
        if n < 1:
            # Kelly budget can't afford one contract — live compute_trade skips too
            continue

        # Mirror live compute_trade's shrink loop exactly (strategy.py): the
        # budget above covers the CONTRACTS only, while the exact ceiling-rounded
        # fees ride on top — so the raw n systematically overshoots the Kelly cap.
        # Shrink until the fee-inclusive cost actually fits the Kelly budget.
        # (No max_contracts analog here: the backtest has no orderbook depth to
        # cap against, only candle closes.)
        fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        while n > 0 and n * (price_a + price_b) + fee_a + fee_b > budget:
            n -= 1
            fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        if n < 1:
            # Fees ate the entire Kelly budget — no contract count fits
            continue

        total_cost = n * (price_a + price_b)
        # Exact ceiling-rounded taker fees for both legs, charged at entry
        fees = fee_a + fee_b
        # Win-scenario NET profit after exact fees (same_title: the floor every
        # co-resolution outcome clears; time_series: the profit of either win
        # cell) — reject if the ceiling rounding ate the margin (mirrors live
        # compute_trade's min_payoff gate)
        expected_payoff = n * (1.0 - price_a - price_b) - fees
        if expected_payoff <= 0:
            continue
        # Greedy fit against the RUNNING cash — select_portfolio's admission
        # rule; a spec that no longer fits is skipped, later cheaper ones may
        # still be admitted.
        if total_cost + fees > cash:
            continue

        # Realized P&L from the settlement outcomes — n per leg whose market
        # resolved to the side bought (sides depend on the pair type)
        receipt      = _settlement_receipt(n, c["outcome_a"], c["outcome_b"], c["pair_type"])
        profit       = receipt - total_cost - fees
        invested     = total_cost + fees
        profit_ratio = profit / invested if invested > 0 else 0.0
        # Normalize realized return to a 30-day equivalent (reporting only)
        monthly_profit_ratio = profit_ratio * 30.0 / c["holding_days"]
        # Slippage = realized profit vs. the win-scenario payoff (net vs. net)
        slippage = profit - expected_payoff

        trades.append(BacktestTrade(
            pair_type=c["pair_type"],
            ticker_a=mA["ticker"],
            ticker_b=mB["ticker"],
            title_a=c["title_a"],
            title_b=c["title_b"],
            # infer_category maps the event_ticker prefix to a human-readable label (e.g. "Crypto")
            category=infer_category(mA.get("event_ticker", "")),
            entry_date=c["entry_date"],
            exit_date=c["exit_date"],
            entry_pA=c["pA"],
            entry_pB=c["pB"],
            entry_nA=c["nA"],
            entry_nB=c["nB"],
            n=n,
            total_cost=total_cost,
            fees=fees,
            outcome_a=c["outcome_a"],
            outcome_b=c["outcome_b"],
            actual_payoff=receipt,
            profit=profit,
            profit_ratio=profit_ratio,
            monthly_profit_ratio=monthly_profit_ratio,
            kelly_fraction=c["kelly_f_capped"],
            expected_payoff=expected_payoff,
            slippage=slippage,
            holding_days=c["holding_days"],
            balance_at_entry=checkpoint_cash,
            deadline_gap_days=c["gap_days"],
        ))

        # Cash out the door: contracts plus fees; the receipt comes back at exit
        cash -= invested
        pending_exits.append((c["exit_date"], receipt))

        # Mark both tickers as active so no OVERLAPPING pair is added later;
        # the release ledger frees them again on this trade's exit date.
        active_tickers.add(mA["ticker"])
        active_tickers.add(mB["ticker"])
        active_until.append((c["exit_date"], mA["ticker"]))
        active_until.append((c["exit_date"], mB["ticker"]))

    logging.info(
        "Backtest complete: %d trades, %d profitable",
        len(trades),
        sum(1 for t in trades if t.profit > 0),
    )

    equity_df = _build_equity_curve(trades, start_date, initial_balance)
    return trades, equity_df


# ─── Equity curve construction ────────────────────────────────────────────────

def _build_equity_curve(
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
) -> pd.DataFrame:
    """
    Construct a daily equity curve DataFrame from the list of backtest trades.

    For each trade, subtracts the full cash outlay (total_cost + fees, both paid
    at execution) from cash on the entry_date and adds the gross settlement
    receipt (actual_payoff) on the exit_date. This models a simple accounting
    treatment where capital is deployed on entry and returned at settlement,
    with each dollar counted exactly once.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades with entry_date,
            exit_date, total_cost, and actual_payoff populated.
        start_date (date): The first date of the equity curve (initial balance day).
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        pd.DataFrame: DataFrame with one row per calendar day from start_date to
            today (UTC), with columns:
            - "date" (date): Calendar date.
            - "portfolio_value" (float): Cumulative portfolio value in dollars.
            - "daily_return" (float): Fractional daily return (pct_change of portfolio_value).
    """
    # entry_date and exit_date come from UTC-derived timestamps, so use UTC today
    # here as well — otherwise `date.today()` in a non-UTC timezone can drop or add
    # a day around the boundary and misalign the equity curve.
    today = datetime.now(UTC).date()
    dates = [start_date + timedelta(days=i) for i in range((today - start_date).days + 1)]

    # Accumulate cash inflows and outflows per date
    cash_changes: dict[date, float] = defaultdict(float)
    for t in trades:
        # Capital leaves the portfolio on entry day (contract cost + taker fees)
        cash_changes[t.entry_date] -= t.total_cost + t.fees
        # Gross settlement receipt returns to the portfolio on exit day
        cash_changes[t.exit_date]  += t.actual_payoff

    rows = []
    cash = initial_balance
    for d in dates:
        # Apply any net cash change for this day (may be zero if no trades entered/exited)
        cash += cash_changes.get(d, 0.0)
        rows.append({"date": d, "portfolio_value": cash})

    df = pd.DataFrame(rows)
    # Compute fractional daily returns; the first row has no prior day so it gets 0.0
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df
