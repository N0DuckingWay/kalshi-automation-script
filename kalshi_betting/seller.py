"""
File: seller.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Decides which held positions the take-profit rule sells this run and plans
    each sale; main.py hands the plans to trader.sell_positions. A position is an
    exact held pair (one YES and one NO market of equal count), or a lone held
    market with its paid-out partner (a settled market the account held on the other
    side, asking the same question, at the same count or, when no market of that
    count fits, at least as many (a lone market partly sold after its partner paid
    out holds fewer)). Its held markets are alone on their ladder (one question
    asked at several deadlines). It is sold when its realized profit (what
    selling its held markets returns after fees, plus any partner's payout, less
    its cost) has reached LiveSettings.sell_at of its potential
    profit (what it pays if it wins, less its cost) at each of
    config.TAKE_PROFIT_HOLD_DAYS daily checks, the last one now. The rule in
    full, and where it differs from the backtest: CLAUDE.md (live selling).

    It also plans Kelly trims (plan_trims): selling part of a held position
    that is worth more than its Kelly share of the portfolio value (the most
    Kelly sizing would put in it), by the backtest's rule
    (config.kelly_trim_count). A position is an exact held pair, or a lone
    leg: the later market of a time-series pair, still held after the
    earlier market paid out $0 (the trim finds that market among the
    account's settlements as a take-profit sale finds a partner). The
    paid-out market then reads as its settlement price
    (config.settled_bids), so the pair arithmetic runs unchanged. A position
    is trimmed only after enough calendar days without a fill (a trade) on
    its markets and, with LiveSettings.min_price_move set, once its price
    has moved that far since, up or down. A trim sells only into bids within
    config.KELLY_TRIM_MAX_BID_DROP of the best bid, never into a gap in a
    book.

    And it decides which held pairs a run may add to (plan_add_ons): an
    add-on is the other half of a rebalance, so it waits as a trim does,
    the same number of calendar days after the last fill on the pair's
    markets and, with a minimum price move, until the pair's price has moved
    that far since, up or down. It sends nothing and sizes nothing: the
    finders and the sizer do the rest. With
    LiveSettings.rebalance_min_days_to_maturity set ("Rebalance until"),
    neither a trim nor an add-on is made once the position's later market
    closes fewer than that many days from today (UTC dates).

Dependencies:
    Imports the standard library, config, scanner and historical only, never
    trader, strategy, the backtester or main (pinned by tests/test_seller.py).
    main.py alone imports it: a production run calls plan_sales before it
    buys anything and, with rebalance on, plan_trims right after it and
    plan_add_ons once it has found the held pairs it may add to.

Notes:
    Every request is a read-only GET. Each planner call reads for itself,
    each thing at most once. Fails closed: anything unknown or unreadable
    means no sale, no trim and no add-on, with a log line.
"""
import logging
import math
import numbers
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from .config import (
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    CONTRACT_PAYOUT_DOLLARS,
    KELLY_TRIM_MAX_BID_DROP,
    MIN_ACTIVE_PRICE_DOLLARS,
    PRICE_THEN_MAX_AGE_DAYS,
    REBALANCE_MIN_DAYS_SINCE_TRADE,
    SALE_KIND_KELLY_TRIM,
    SALE_KIND_TAKE_PROFIT,
    SALE_PARTNER_MAX_AGE_DAYS,
    SALE_PARTNER_MAX_LOOKUPS,
    SALE_PARTNER_MAX_WIDER_LOOKUPS,
    SALE_PARTNER_REVENUE_TOLERANCE_DOLLARS,
    SCHEDULED_RUN,
    TAKE_PROFIT_HOLD_DAYS,
    TAKE_PROFIT_HOLD_DAYS_MAX,
    LiveSettings,
    confirmed_fee_multiplier,
    count_text,
    days_left_reached,
    days_to_maturity,
    fee_leg_exact,
    fee_per_pair_approx,
    held_pair_win_prob,
    kelly_hold_fraction,
    kelly_trim_bids,
    kelly_trim_count,
    pair_price_move,
    pair_size_cap,
    price_move_reached,
    reached_every_check,
    rebalance_min_days,
    rebalance_rested,
    settled_bids,
    take_profit_reached,
    trade_day,
)
from .historical import ask_before, bid_before, candle_reads_bid, recent_candles, series_ticker
from .scanner import (
    FillTimes,
    HeldPair,
    HeldPosition,
    Settlement,
    _fetch_orderbook,
    bid_ladder,
    display_title,
    get_fill_times,
    get_settlements,
    held_pair_kind,
    held_pairs,
    held_positions_bid_value,
    leg_sides,
    listed_ask,
    listed_fee_multiplier,
    market_event_fee_override,
    market_fee_multiplier,
    market_for_labels,
    market_ladder_keys,
    walk_bids,
)

# One day in seconds: the gap between checks, and how far back a check looks for a candle
_DAY_SECONDS = 86_400

# One candle period in seconds: the candle fetch starts this much before the oldest
# check's 24 hours, so no candle a check reads is cut off
_CANDLE_PERIOD_SECONDS = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60

_OTHER_SIDE = {"yes": "no", "no": "yes"}


@dataclass(frozen=True)
class SaleLeg:
    """
    One market of a position: held now, or a paid-out partner.

    Attributes:
        ticker (str): The market's ticker.
        event_ticker (str): Its event's ticker.
        side (str): "yes" or "no": the side held (for a partner, before it paid out).
        count (int): Contracts held: the position's contract pairs; for a Kelly
            trim, the contracts sold there (for a lone leg's paid-out partner,
            the share of it that goes with them).
        cost_dollars (float): What they cost, fees included.
        market (Any): The ApiMarket from this run's market list while held; None for a partner.
        payout_dollars (float | None): What a partner paid when it settled, for
            `count` of the contracts it settled with (_partner_leg); None while held.
        paid_at (datetime | None): When a partner settled (UTC); None while held.
        partner_fee_multiplier (float): For a paid-out partner (no market), the
            fee multiplier it is listed at (scanner.listed_fee_multiplier of its
            series' and its event's override, 1.0 when either is unknown), which
            an earlier day's check charges its sale at; 1.0 for a held market,
            whose own market carries its multiplier.
        held (int | None): For a Kelly trim, the contracts held there; None when
            `count` is all of them, and for a paid-out partner.
    """
    ticker: str
    event_ticker: str
    side: str
    count: int
    cost_dollars: float
    market: Any = None
    payout_dollars: float | None = None
    paid_at: datetime | None = None
    partner_fee_multiplier: float = 1.0
    held: int | None = None


@dataclass(frozen=True)
class SalePlan:
    """
    Held contracts a run sells, with what the sale orders need.

    A take-profit sale (plan_sales) sells a whole position. A Kelly trim
    (plan_trims) sells part of an exact held pair, or of a lone leg (the later
    market of a time-series pair, held after its earlier market paid out $0);
    its count and money figures are those of the part sold, a lone leg's
    including that share of its paid-out partner.

    Attributes:
        title (str): The first held market's display title.
        legs (tuple[SaleLeg, ...]): Its held markets in ticker order, then its partner if any.
        count (int): What is sold: contract pairs of an exact pair, contracts of
            a lone leg's one held market (for a take-profit sale, all held on
            each market).
        cost_dollars (float): What the whole position cost, fees included; for a trim,
            the part sold.
        ladders (dict): Held ticker -> bids for the side held, best first ([[price, qty], ...]);
            a trim keeps only the bids near the best one (config.kelly_trim_bids).
        walked (dict): Held ticker -> (average, lowest price reached) selling the count down them.
        proceeds_dollars (float): What selling the count on the held markets now returns after fees.
        profits (tuple): (realized, potential profit) at each check, now first; a trim
            has one check, now.
        days_left (int | None): Days from today (UTC) until its last held market closes, if known.
        level (float | None): The share of potential profit a take-profit sale sells at
            (LiveSettings.sell_at); None for a trim.
        kind (str): config.SALE_KIND_TAKE_PROFIT or config.SALE_KIND_KELLY_TRIM.
    """
    title: str
    legs: tuple[SaleLeg, ...]
    count: int
    cost_dollars: float
    ladders: dict
    walked: dict
    proceeds_dollars: float
    profits: tuple
    days_left: int | None
    level: float | None
    kind: str = SALE_KIND_TAKE_PROFIT


def _hold_days() -> int:
    """
    Check how many daily checks a position must pass before it is sold.

    Reads seller.TAKE_PROFIT_HOLD_DAYS when called (a test patches it there, never
    config's) and checks it as backtester._resolve_hold_days does: change both together.

    Returns:
        int: A whole number from 1 to TAKE_PROFIT_HOLD_DAYS_MAX.

    Raises:
        ValueError: For any other value, a bool included.
    """
    days = TAKE_PROFIT_HOLD_DAYS
    if (isinstance(days, bool) or not isinstance(days, numbers.Integral)
            or not 1 <= days <= TAKE_PROFIT_HOLD_DAYS_MAX):
        raise ValueError(f"TAKE_PROFIT_HOLD_DAYS must be a whole number from 1 to "
                         f"{TAKE_PROFIT_HOLD_DAYS_MAX}, got {days!r}")
    return int(days)


def _rebalance_min_days() -> int:
    """
    Check how many days a held pair must go without a trade before a rebalance: a Kelly trim or an add-on.

    Reads seller.REBALANCE_MIN_DAYS_SINCE_TRADE when called: a test patches it
    here, never config's.

    Returns:
        int: A whole number from 0 (no wait) to config.REBALANCE_MIN_DAYS_SINCE_TRADE_MAX.

    Raises:
        ValueError: For any other value, a bool or a float included.
    """
    # Cross-module: config's one check of this number
    return rebalance_min_days(REBALANCE_MIN_DAYS_SINCE_TRADE)


def _day(moment: datetime) -> date:
    """
    The calendar day a moment falls on, on the bot's own clock (its schedule's time zone).

    A rebalance's wait (a Kelly trim's, and an add-on's) is counted between
    days read here: today's and a fill's. Reads seller.SCHEDULED_RUN when
    called: a test patches it here, never config's.

    Args:
        moment (datetime): A time that carries its zone.

    Returns:
        date: The day, as a plain datetime.date.

    Raises:
        TypeError: When the moment carries no zone.
    """
    # Cross-module: config's one rule for which day a moment falls on
    return trade_day(moment, SCHEDULED_RUN)


def _other(side: str) -> str:
    """
    The other side of a market's book.

    Args:
        side (str): "yes" or "no".

    Returns:
        str: "no" for "yes", "yes" for "no".
    """
    return _OTHER_SIDE[side]


def _utc_date(close_time: Any) -> date | None:
    """
    The calendar date (UTC) a market's close time names, or None when it cannot be read.

    Args:
        close_time (Any): ApiMarket.close_time.

    Returns:
        date | None: Its UTC date; None unless it is an aware datetime that converts to UTC.
    """
    if not isinstance(close_time, datetime) or close_time.utcoffset() is None:
        return None
    try:
        return close_time.astimezone(UTC).date()
    except (OverflowError, ValueError):
        return None


def _dollars(amount: float) -> str:
    """
    Write a dollar amount for a log line, with its sign before the dollar sign.

    Args:
        amount (float): Dollars.

    Returns:
        str: e.g. "$18.90" or "-$0.20".
    """
    return f"-${-amount:.2f}" if amount < 0 else f"${amount:.2f}"


def _sides_text(sides: tuple[tuple[str, str], ...]) -> str:
    """
    Name held markets by side and ticker, for a log line.

    Args:
        sides (tuple[tuple[str, str], ...]): (ticker, side) per held market.

    Returns:
        str: e.g. "YES KX-A / NO KX-B".
    """
    return " / ".join(f"{side.upper()} {ticker}" for ticker, side in sides)


def _held_text(pair: HeldPair) -> str:
    """
    Name a held position by side and ticker, for a log line.

    Args:
        pair (HeldPair): An exact held pair or a lone leg.

    Returns:
        str: e.g. "YES KX-A / NO KX-B", or "NO KX-B (alone on its ladder)" for a lone leg.
    """
    return _sides_text(pair.sides) + (" (alone on its ladder)" if pair.lone else "")


def _held_market_text(ticker: Any, position: HeldPosition) -> str:
    """
    Name one held market by side and ticker, for a "Not selling" line.

    Args:
        ticker (Any): The held market's ticker.
        position (HeldPosition): Its listing row.

    Returns:
        str: e.g. "YES KX-A"; the side reads "?" when the count is unknown.
    """
    count = position.count
    side = "?" if count is None else ("YES" if count > 0 else "NO")
    return f"{side} {ticker}"


def _shape_reason(position: HeldPosition) -> tuple[str, str]:
    """
    Say why scanner.held_pairs left a held market out of every position.

    Args:
        position (HeldPosition): The held market's listing row.

    Returns:
        tuple[str, str]: (the reason in words, a short label for the summary line).
    """
    if position.count is None:
        return "its contract count could not be read", "count unreadable"
    if position.exposure_dollars is None or position.fees_dollars is None:
        return "its cost could not be read", "cost unreadable"
    if position.exposure_dollars < MIN_ACTIVE_PRICE_DOLLARS * abs(position.count):
        return "its cost is below the finest price times its count", "cost too small"
    return ("other held markets share its ladder, but not as one YES and one NO of "
            "equal count with readable costs"), "not one exact pair"


def _settled_count(settlement: Settlement, side: str) -> float:
    """
    The contracts of one side the account held when a settled market paid out.

    Kalshi's settlement lists the contracts bought on each side over the
    market's life. A sale before the payout is listed as bought on the other
    side (selling YES is buying NO), and the two net out, so what was held at
    payout is their difference: 180 YES and 52 NO listed is 128 YES held. For a
    record that lists one side only, it is that side's count.

    Args:
        settlement (Settlement): One settled market.
        side (str): "yes" or "no".

    Returns:
        float: That side's count less the other side's, never below 0.
    """
    bought = _bought_count(settlement, side)
    return max(0.0, bought - _bought_count(settlement, _OTHER_SIDE[side]))


def _bought_count(settlement: Settlement, side: str) -> float:
    """
    The contracts of one side a settled market's settlement lists as bought.

    Args:
        settlement (Settlement): One settled market.
        side (str): "yes" or "no".

    Returns:
        float: Settlement.yes_count or no_count.
    """
    return settlement.yes_count if side == "yes" else settlement.no_count


class _Reads:
    """
    One planner call's reads beyond the positions listing, each made once, when first needed.

    Attributes:
        client (Any): The authenticated Kalshi client.
        held_positions (dict): Ticker -> HeldPosition for every held market (empty
            for plan_add_ons, which reads no position).
        markets_by_ticker (dict): This run's whole market list by ticker.
        now_ts (float): The run's moment, in Unix seconds.
        hold_days (int): How many daily checks there are (0 for a Kelly trim or an
            add-on, which make none).
        lookups (int): Label lookups from the exchange so far (SALE_PARTNER_MAX_LOOKUPS at most).
        wider_lookups (int): Label lookups from the exchange so far in _find_partner's
            search for a partner settled with more contracts than are held
            (SALE_PARTNER_MAX_WIDER_LOOKUPS at most), counted apart from `lookups`.
        fee_multipliers (dict): Series ticker -> fee multiplier
            (historical.load_series_fee_multipliers), for a paid-out partner's fee.
        purpose (str): "selling" (plan_sales) or "trimming" (plan_trims): what a
            lone held market whose partner has paid out is looked at for, which
            the WARNING of settlements() names when the settlements cannot be used.

    Besides the reads below, label_market(ticker) gives the market labels()
    found for a settled market, so a trim can read a paid-out partner's market.
    """

    def __init__(self, client: Any, held_positions: dict, markets_by_ticker: dict,
                 now_ts: float, hold_days: int, *, fee_multipliers: dict | None = None,
                 purpose: str = "selling") -> None:
        """Start with nothing read; each argument is the attribute of the same name."""
        self.client = client
        self.fee_multipliers = fee_multipliers or {}
        self.purpose = purpose
        # Each settled market looked up for its labels (None: not found)
        self._label_markets: dict = {}
        self.held_positions = held_positions
        self.markets_by_ticker = markets_by_ticker
        self.now_ts = now_ts
        self.hold_days = hold_days
        self.lookups = 0
        self.wider_lookups = 0
        self._candles: dict = {}
        self._then_candles: dict = {}
        self._settlements: list | None = None
        self._settlements_read = False
        self._settlements_problem = ""
        self._labels: dict = {}
        self._event_titles: dict = {}

    def candles(self, ticker: str, event_ticker: str) -> list | None:
        """
        One market's candles over the days the earlier checks read, fetched once.

        Args:
            ticker (str): The market's ticker.
            event_ticker (str): Its event's ticker (its series names the candle path).

        Returns:
            list | None: The candles; None when they could not be read.
        """
        if ticker not in self._candles:
            end = math.floor(self.now_ts)
            start = end - self.hold_days * _DAY_SECONDS - _CANDLE_PERIOD_SECONDS
            # Cross-module: hourly candles read fresh, never from the backtest's candle cache
            self._candles[ticker] = recent_candles(self.client, ticker, event_ticker,
                                                   start, end)
        return self._candles[ticker]

    def ask_then(self, market: Any, side: str, moment: int) -> float:
        """
        One side's ask (its buy price) on a market at a past moment, from hourly candles.

        A rebalance's minimum price move (a Kelly trim's, and an add-on's)
        compares a pair's price now with its price at its last fill; this
        reads one market's part of the price then.
        The candles cover the config.PRICE_THEN_MAX_AGE_DAYS days before the
        moment and are fetched once.

        Args:
            market (Any): The held market.
            side (str): The side held there, "yes" or "no".
            moment (int): The past moment, in Unix seconds.

        Returns:
            float: The side's latest usable ask at or before the moment, in dollars;
                NaN when the candles could not be read or hold none in those days.
        """
        window = PRICE_THEN_MAX_AGE_DAYS * _DAY_SECONDS
        key = (market.ticker, moment)
        if key not in self._then_candles:
            # Cross-module: fresh hourly candles (never the backtest's cache), starting
            # one candle period early so none the window reaches is cut off
            self._then_candles[key] = recent_candles(
                self.client, market.ticker, market.event_ticker,
                moment - window - _CANDLE_PERIOD_SECONDS, moment)
        candles = self._then_candles[key]
        if candles is None:
            return float("nan")
        # Cross-module: the side's latest usable ask, by the rule the backtest prices with
        return ask_before(candles, moment, side, window=window)

    def settlements(self) -> tuple[list | None, str]:
        """
        The account's recent settlements, read once, when every record could be read.

        An unreadable record may be a lone held market's partner, so then none is
        used and no lone held market is sold (or, for plan_trims, trimmed) this
        run (fails closed, one WARNING, worded for `purpose`).

        Returns:
            tuple[list | None, str]: (the settlements, "") or (None, why not usable).
        """
        if not self._settlements_read:
            self._settlements_read = True
            unreadable: dict = {}
            # Only the window a partner may come from, so an older unreadable record cannot
            # stop a sale; a second wider than _find_partner's own age check
            oldest = math.floor(self.now_ts - SALE_PARTNER_MAX_AGE_DAYS * _DAY_SECONDS) - 1
            # Cross-module: the settlements in that window, counting unreadable records
            found = get_settlements(self.client, unreadable_out=unreadable, min_ts=oldest)
            if found is None:
                self._settlements_problem = "the account's settlements could not be read"
            elif unreadable.get("unreadable", 0) > 0:
                self._settlements_problem = (
                    f"{unreadable['unreadable']} of the account's settlements could not be "
                    "read, and any of them may be the partner")
            else:
                self._settlements = found
            if self._settlements_problem and self.purpose == "trimming":
                logging.warning("Not trimming any lone leg whose partner has paid out this "
                                "run: %s", self._settlements_problem)
            elif self._settlements_problem:
                logging.warning("Not selling any held market whose partner has paid out this "
                                "run: %s", self._settlements_problem)
        return self._settlements, self._settlements_problem

    def labels(self, ticker: str, *, wider: bool = False) -> tuple[frozenset | None, str]:
        """
        A settled market's ladder labels, found once: in this run's market list, or looked up.

        A lookup counts against SALE_PARTNER_MAX_LOOKUPS, or, for _find_partner's search
        for a partner settled with more contracts (`wider`), against
        SALE_PARTNER_MAX_WIDER_LOOKUPS, so that search never uses up the lookups a
        same-count partner needs. Labels found by either are kept for both.

        Args:
            ticker (str): The settled market's ticker.
            wider (bool): True when the search for a partner settled with more
                contracts asks.

        Returns:
            tuple[frozenset | None, str]: (its labels, "") or (None, why not found).
        """
        if ticker in self._labels:
            return self._labels[ticker]
        listed = ticker in self.markets_by_ticker
        if not listed and wider:
            if self.wider_lookups >= SALE_PARTNER_MAX_WIDER_LOOKUPS:
                return None, (f"the limit of {SALE_PARTNER_MAX_WIDER_LOOKUPS} lookups for a "
                              "partner settled with more contracts this run was reached")
            self.wider_lookups += 1
        elif not listed:
            if self.lookups >= SALE_PARTNER_MAX_LOOKUPS:
                return None, (f"the limit of {SALE_PARTNER_MAX_LOOKUPS} partner lookups "
                              "this run was reached")
            self.lookups += 1
        # Cross-module: from the run's list, else looked up as held markets are, so its labels
        # match the held market's own (_find_partner compares their question labels)
        market = market_for_labels(self.client, ticker, self.markets_by_ticker,
                                   self._event_titles)
        self._label_markets[ticker] = market
        if market is None:
            result = (None, f"settled market {ticker} could not be looked up, and it may "
                            "be the partner")
        else:
            # Cross-module: the one definition of a market's ladder labels
            result = (market_ladder_keys(market), "")
        self._labels[ticker] = result
        return result

    def label_market(self, ticker: str) -> Any:
        """
        The market labels() found for a settled market, so a trim can read a paid-out partner.

        Args:
            ticker (str): The settled market's ticker.

        Returns:
            Any: Its ApiMarket (from this run's market list, or looked up); None
                when labels() did not look it up or did not find it.
        """
        return self._label_markets.get(ticker)

    def partner_fee_multiplier(self, ticker: str, event_ticker: str) -> float:
        """
        The fee multiplier a paid-out partner is listed at.

        scanner.listed_fee_multiplier of its series' multiplier and its event's
        override, read from the market labels() found for it: 1.0, the full
        fee, when the market was not found or its event's override is not
        known, which includes a partner looked up outside the run's list
        (scanner._fetch_held_market reads no override). A partner in the
        run's list is charged as the backtest charges a pair's paid-out leg.

        Args:
            ticker (str): The partner's ticker.
            event_ticker (str): Its event's ticker (its series names its multiplier).

        Returns:
            float: The listed multiplier (config.confirmed_fee_multiplier charges it).
        """
        # Cross-module: the event's override, else the series'; unknown -> the full fee
        return listed_fee_multiplier(
            self.fee_multipliers.get(series_ticker(event_ticker or ""), 1.0),
            market_event_fee_override(self._label_markets.get(ticker)))


def _find_partner(reads: _Reads, ticker: str, side: str, count: int,
                  labels: frozenset) -> tuple[Settlement | None, str]:
    """
    Find a lone held market's one paid-out partner among the account's settlements.

    The partner is the one settled market that is no longer held, was held on
    the other side at payout (_settled_count), asks the held market's question,
    and settled within SALE_PARTNER_MAX_AGE_DAYS. Its count is the held count;
    only if no market has that, at least the held count (a lone market sold in
    part after payout holds fewer than its partner settled with). Exactly one
    market must fit; the second search has its own lookup allowance
    (SALE_PARTNER_MAX_WIDER_LOOKUPS). Used by _plan_one and _trim_one. Fails
    closed on no or several partners, an unreadable label, a result other than
    yes or no, or a two-sided record whose payout does not match its net count.

    Args:
        reads (_Reads): This run's reads.
        ticker (str): The lone held market's ticker.
        side (str): The side held there, "yes" or "no".
        count (int): Its count.
        labels (frozenset): Its ladder labels.

    Returns:
        tuple[Settlement | None, str]: (the partner, "") or (None, why not).
    """
    questions = [label for label in labels if label[0] == "question"]
    if not questions:
        return None, "its question is unknown, so no partner can be shown to ask it"
    question = questions[0]
    settlements, problem = reads.settlements()
    if settlements is None:
        return None, problem
    other = _OTHER_SIDE[side]
    oldest = reads.now_ts - SALE_PARTNER_MAX_AGE_DAYS * _DAY_SECONDS

    def question_matches_with_count(fits, wider: bool) -> tuple[list[Settlement] | None, str]:
        """Up to two settled markets that pass every test above and whose count on the
        other side fits; (None, why) when a candidate's labels cannot be found. `wider`
        names the search for a partner settled with more (its own lookup allowance)."""
        candidates = [s for s in settlements
                      if s.ticker not in reads.held_positions and s.ticker != ticker
                      # The second test is implied by the net reading (a market
                      # held on `other` at payout held none on `side`); kept as a guard
                      and fits(_settled_count(s, other)) and _settled_count(s, side) == 0
                      and s.settled_at.timestamp() >= oldest]
        partners: list[Settlement] = []
        for settlement in candidates:
            found, problem = reads.labels(settlement.ticker, wider=wider)
            if found is None:
                return None, problem
            if question not in found:
                continue
            partners.append(settlement)
            if len(partners) > 1:
                break
        return partners, ""

    # The same count first; only when no market fits it, one settled with at least
    # as many (a label already found is not looked up again)
    partners, problem = question_matches_with_count(lambda settled: settled == count, False)
    if partners == []:
        partners, problem = question_matches_with_count(lambda settled: settled >= count, True)
    if partners is None:
        return None, problem
    if not partners:
        # config.count_text: a contract count written exactly
        return None, (f"no paid-out partner: no market it no longer holds was held "
                      f"{other.upper()} with at least {count_text(count)} contracts on its "
                      f"question and paid out in the last {SALE_PARTNER_MAX_AGE_DAYS} days")
    if len(partners) > 1:
        return None, ("more than one settled market could be its paid-out partner ("
                      + ", ".join(s.ticker for s in partners) + ")")
    partner = partners[0]
    if partner.result not in ("yes", "no"):
        return None, (f"its partner {partner.ticker} settled {partner.result} rather than "
                      "yes or no, which the rule does not judge")
    if partner.yes_count > 0 and partner.no_count > 0:
        # Contracts bought on both sides: what was held at payout is their
        # difference only when the payout says so ($1 for each contract held on
        # the winning side, $0 when the side held lost)
        expected = _settled_count(partner, partner.result) * CONTRACT_PAYOUT_DOLLARS
        if abs(partner.revenue_dollars - expected) > SALE_PARTNER_REVENUE_TOLERANCE_DOLLARS:
            # config.count_text: a contract count written exactly
            return None, (f"its partner {partner.ticker} lists contracts bought on both "
                          f"sides ({count_text(partner.yes_count)} YES, "
                          f"{count_text(partner.no_count)} NO), and its payout "
                          f"({_dollars(partner.revenue_dollars)}) does not match the "
                          f"{count_text(_settled_count(partner, other))} {other.upper()} "
                          "held when it paid out, so what was held cannot be read")
    return partner, ""


def _partner_leg(reads: _Reads, partner: Settlement, other: str, count: int) -> SaleLeg:
    """
    A lone held market's paid-out partner as one leg of a position (or the part of
    one a Kelly trim sells) of `count` contract pairs.

    The partner's cost is `count` contracts at the average cost of a contract
    bought on the side held there, fees included: (that side's cost + the fees) x
    count / the contracts bought on that side. Contracts sold before the payout
    were a trade of their own, so the ones held keep their own average cost. Its
    payout is scaled to `count` of the contracts held when it paid out
    (_settled_count): a lone market partly sold after its partner paid out holds
    fewer contracts than the partner settled with, so the position is that share
    of the pair. For a partner whose record lists one side, bought and held are
    the same count, so when it settled with exactly `count` contracts the
    figures are the settlement's own. _plan_one (the take-profit sale) carries this leg
    beside the held market, and its payout values the partner at every check it
    had paid out by; _trim_one (a Kelly trim of a lone leg) carries it for the
    contracts the trim sells, so the part sold's cost includes that share of
    the partner's.

    Args:
        reads (_Reads): This run's reads (for the partner's fee multiplier).
        partner (Settlement): The partner _find_partner found: holding at least
            `count` contracts on `other` when it paid out.
        other (str): The side held on the partner, "yes" or "no".
        count (int): The lone market's contracts this leg goes with: all held
            for a take-profit sale (_plan_one), the contracts sold for a Kelly
            trim (_trim_one); at least one.

    Returns:
        SaleLeg: The partner's leg, with no market.
    """
    cost_basis = partner.yes_cost_dollars if other == "yes" else partner.no_cost_dollars
    return SaleLeg(partner.ticker, partner.event_ticker, other, count,
                   (cost_basis + partner.fees_dollars) * (count / _bought_count(partner, other)),
                   payout_dollars=partner.revenue_dollars * (count / _settled_count(partner, other)),
                   paid_at=partner.settled_at,
                   partner_fee_multiplier=reads.partner_fee_multiplier(
                       partner.ticker, partner.event_ticker))


def _checks_text(profits: list[tuple[float, float]]) -> str:
    """
    Write each check's realized share of potential profit, now first, for a log line.

    Args:
        profits (list[tuple[float, float]]): (realized, potential profit) per check, potential > 0.

    Returns:
        str: e.g. "now 84%, 1 day before 82%, 2 days before 85%".
    """
    parts = []
    for back, (realized, potential) in enumerate(profits):
        when = "now" if back == 0 else f"{back} day{'' if back == 1 else 's'} before"
        parts.append(f"{when} {realized / potential:.0%}")
    return ", ".join(parts)


def _plan_one(pair: HeldPair, reads: _Reads, held_labels: dict, *, sell_at: float,
              min_days: int | None, now_date: date) -> SalePlan | None:
    """
    Judge one position, logging its verdict, and plan its sale; it stops at its first failed step.

    Args:
        pair (HeldPair): An exact held pair or a lone held market (held_pairs).
        reads (_Reads): This run's reads.
        held_labels (dict): Ticker -> ladder labels for every held market.
        sell_at (float): The share of potential profit to sell at.
        min_days (int | None): The minimum of days left before its last held market closes, if any.
        now_date (date): Today's date, UTC.

    Returns:
        SalePlan | None: The plan when the rule sells it; None otherwise.
    """
    text = _held_text(pair)

    def not_selling(reason: str) -> None:
        """Log why this position is not sold."""
        logging.info("Not selling %s: %s", text, reason)

    count_value = pair.count
    if not float(count_value).is_integer():
        # config.count_text: a contract count written exactly
        not_selling(f"its count {count_text(count_value)} is not a whole number of contracts")
        return None
    count = int(count_value)
    missing = [ticker for ticker, _side in pair.sides if ticker not in reads.markets_by_ticker]
    if missing:
        not_selling(f"{missing[0]} is not in this run's market list")
        return None
    markets = {ticker: reads.markets_by_ticker[ticker] for ticker, _side in pair.sides}
    # Cross-module: days to maturity from each held market's scheduled close date
    days_left = days_to_maturity([_utc_date(m.close_time) for m in markets.values()], now_date)
    if min_days is not None:
        # The days rule comes first, so a position too near maturity costs no request
        if days_left is None:
            not_selling(f"a held market's close date is unknown, so the minimum of "
                        f"{min_days} days before maturity cannot be checked")
            return None
        if days_left < min_days:
            not_selling(f"{days_left} day(s) left before its last market stops trading, "
                        f"fewer than the minimum of {min_days}")
            return None

    # The check now: each held market's real bids for the side held, walked for the count
    ladders: dict = {}
    walked: dict = {}
    for ticker, side in pair.sides:
        # Cross-module: the market's order book, a retried read-only GET
        book = _fetch_orderbook(reads.client, ticker)
        if book is None:
            not_selling(f"the order book of {ticker} could not be read")
            return None
        # Cross-module: the side's bids, best first, then sold down for the count
        ladder = bid_ladder(book, side, ticker=ticker)
        walk = walk_bids(ladder, count)
        if walk is None:
            not_selling(f"the {side.upper()} bids on {ticker} hold fewer than "
                        f"{count} contracts")
            return None
        ladders[ticker] = ladder
        walked[ticker] = walk

    legs = [SaleLeg(ticker, markets[ticker].event_ticker, side, count,
                    reads.held_positions[ticker].exposure_dollars
                    + reads.held_positions[ticker].fees_dollars,
                    market=markets[ticker])
            for ticker, side in pair.sides]
    if pair.lone:
        ticker, side = pair.sides[0]
        partner, problem = _find_partner(reads, ticker, side, count, held_labels[ticker])
        if partner is None:
            not_selling(problem)
            return None
        if partner.settled_at.timestamp() > reads.now_ts:
            # A payout dated after now (the exchange's clock runs ahead) has nothing to value it by
            not_selling(f"its partner {partner.ticker} is recorded as settling after now")
            return None
        other = _OTHER_SIDE[side]
        # The partner's cost and payout, scaled to the contracts held here
        legs.append(_partner_leg(reads, partner, other, count))
        settled = _settled_count(partner, other)
        # config.count_text: named only when the partner settled with more than is held here
        more = "" if settled == count else f" with {count_text(settled)} contracts"
        text = (f"{side.upper()} {ticker} with paid-out {other.upper()} {partner.ticker} "
                f"(settled {partner.result}{more}, paid {_dollars(partner.revenue_dollars)})")

    cost = sum(leg.cost_dollars for leg in legs)
    potential = count * CONTRACT_PAYOUT_DOLLARS - cost
    # config.count_text: the count written exactly
    head = (f"Take-profit check (sell at {sell_at:.0%}): {text}, {count_text(count)} each, "
            f"cost {_dollars(cost)}, potential profit {_dollars(potential)}"
            + ("" if days_left is None else f", {days_left} day(s) left"))
    if potential <= 0:
        logging.info("%s -> keep (no potential profit)", head)
        return None

    # Each held market sold at its walk's average price, less the sale's taker fee
    # at that market's own fee multiplier
    proceeds = sum(count * walked[leg.ticker][0]
                   - fee_leg_exact(count, walked[leg.ticker][0],
                                   multiplier=market_fee_multiplier(leg.market))
                   for leg in legs if leg.market is not None)
    # A partner paid out by now (one recorded later was refused above)
    value = proceeds + sum(leg.payout_dollars for leg in legs if leg.market is None)
    profits = [(value - cost, potential)]
    # Cross-module: the rule's one test at one check; stop at the first below the level
    if not take_profit_reached(sell_at, value - cost, potential):
        logging.info("%s -> keep (%s)", head, _checks_text(profits))
        return None

    # The earlier checks, 24 hours apart: each market's last candle bid in the
    # 24 hours before the check, in any size, or a partner's payout once paid
    for back in range(1, reads.hold_days):
        moment = reads.now_ts - back * _DAY_SECONDS
        value = 0.0
        for leg in legs:
            if leg.market is None and leg.paid_at.timestamp() <= moment:
                value += leg.payout_dollars
                continue
            candles = reads.candles(leg.ticker, leg.event_ticker)
            if candles is None:
                not_selling(f"the recent candles of {leg.ticker} could not be read")
                return None
            # Cross-module: the side's last candle bid in the 24 hours before this check
            bid = bid_before(candles, moment, leg.side, window=_DAY_SECONDS)
            if bid != bid:
                not_selling(f"no {leg.side.upper()} bid on {leg.ticker} in the 24 hours "
                            f"before the check {back} day(s) before")
                return None
            # Cross-module: the taker fee on selling `count` at that bid, at the
            # market's own fee multiplier (a partner not yet paid out by this
            # check, which carries no market, at the one it is listed at)
            multiplier = (market_fee_multiplier(leg.market) if leg.market is not None
                          else confirmed_fee_multiplier(leg.partner_fee_multiplier))
            value += count * bid - fee_leg_exact(count, bid, multiplier=multiplier)
        profits.append((value - cost, potential))
        # Cross-module: config.take_profit_reached, the rule's test at this check
        if not take_profit_reached(sell_at, value - cost, potential):
            logging.info("%s -> keep (%s)", head, _checks_text(profits))
            return None

    # Cross-module: the rule's test over every check, as the backtest's _reached_every_day
    if not reached_every_check(sell_at, profits):
        logging.info("%s -> keep (%s)", head, _checks_text(profits))
        return None
    logging.info("%s; %s -> sell", head, _checks_text(profits))
    # Cross-module: the held market's title as the run's tables show it
    title = display_title(markets[pair.sides[0][0]])
    return SalePlan(title=title, legs=tuple(legs), count=count, cost_dollars=cost,
                    ladders=ladders, walked=walked, proceeds_dollars=proceeds,
                    profits=tuple(profits), days_left=days_left, level=sell_at)


def plan_sales(client: Any, held_positions: dict, held_labels: dict, markets_by_ticker: dict,
               *, settings: LiveSettings, now: datetime,
               fee_multipliers: dict | None = None) -> list[SalePlan]:
    """
    Find the held positions the take-profit rule sells this run, and plan each sale.

    Fails closed: when any held market's ladder is unknown (it has no labels),
    no position is judged.

    Args:
        client (Any): The authenticated production Kalshi client, only read from.
        held_positions (dict): Ticker -> HeldPosition for every held market.
        held_labels (dict): Ticker -> ladder labels (scanner.resolve_held_ladders' labels_out).
        markets_by_ticker (dict): This run's whole market list by ticker, held markets included.
        settings (LiveSettings): sell_at (None: sell nothing, read nothing) and sell_min_days.
        now (datetime): The run's moment, timezone-aware: the last check.
        fee_multipliers (dict | None): Keyword-only. Series ticker -> fee
            multiplier (historical.load_series_fee_multipliers, the map
            main._run_prod reads), which a paid-out partner's fee is charged
            at (Reads.partner_fee_multiplier); None charges every partner the
            full fee.

    Returns:
        list[SalePlan]: The positions to sell, in ticker order; empty when none.

    Raises:
        ValueError: When now is not timezone-aware, or TAKE_PROFIT_HOLD_DAYS is invalid.
    """
    sell_at = settings.sell_at
    if sell_at is None:
        return []
    hold_days = _hold_days()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    reads = _Reads(client, held_positions, markets_by_ticker, now.timestamp(), hold_days,
                   fee_multipliers=fee_multipliers)
    never: Counter = Counter()

    unknown = sorted((t for t in held_positions if not held_labels.get(t)), key=str)
    if unknown:
        # An unknown ladder could be any position's, so none is provably alone on its ladder
        logging.info("Not selling this run: the ladder of %d held market(s) is unknown "
                     "(%s)", len(unknown), ", ".join(str(t) for t in unknown))
        never["ladder unknown"] = len(unknown)
        if len(held_positions) > len(unknown):
            never["not judged"] = len(held_positions) - len(unknown)
        _log_summary(0, 0, never)
        return []

    # Cross-module: the one definition of exact held pairs and lone held markets, quietly
    found = held_pairs(held_positions, held_labels, markets_by_ticker, log=False)
    in_position = set().union(*found) if found else set()
    for ticker in sorted(held_positions, key=str):
        if ticker in in_position:
            continue
        position = held_positions[ticker]
        reason, label = _shape_reason(position)
        logging.info("Not selling %s: %s", _held_market_text(ticker, position), reason)
        never[label] += 1

    now_date = now.astimezone(UTC).date()
    plans = []
    for key in sorted(found, key=sorted):
        plan = _plan_one(found[key], reads, held_labels, sell_at=sell_at,
                         min_days=settings.sell_min_days, now_date=now_date)
        if plan is not None:
            plans.append(plan)
    _log_summary(len(plans), len(found), never)
    return plans


def _log_summary(sold: int, considered: int, never: Counter) -> None:
    """
    Log how many positions the run sells, and how many held markets are in none.

    Args:
        sold (int): Positions to sell.
        considered (int): Positions judged (exact pairs and lone held markets).
        never (Counter): Short reason -> held markets in no position for it.
    """
    if never:
        reasons = ", ".join(f"{n} {why}" for why, n in sorted(never.items()))
        logging.info("Positions to sell this run: %d of %d (%d held market(s) never sold: %s)",
                     sold, considered, sum(never.values()), reasons)
    else:
        logging.info("Positions to sell this run: %d of %d", sold, considered)


# ─── Trimming a held pair to its Kelly size ───────────────────────────────────

# How a log line names a pair's kind
_KIND_WORDS = {"time_series": "time-series", "same_title": "same-title"}


def _cents_text(dollars: float) -> str:
    """
    Write a price change in cents with its sign, for a log line.

    Written to the hundredth of a cent (Kalshi's finest price step), so a move
    just short of a minimum never reads as the minimum.

    Args:
        dollars (float): The change, in dollars.

    Returns:
        str: e.g. "+13c", "-5c" or "+9.6c"; a change that rounds to 0 is "+0c", never "-0c".
    """
    cents = f"{abs(dollars) * 100:.2f}".rstrip("0").rstrip(".")
    sign = "-" if dollars < 0 and cents != "0" else "+"
    return f"{sign}{cents}c"


def _price_text(dollars: float) -> str:
    """
    Write a price for a log line: whole cents, or finer when needed.

    Args:
        dollars (float): The price, in dollars.

    Returns:
        str: e.g. "$0.97" or "$0.874"; never finer than a hundredth of a cent.
    """
    whole, _, part = f"{dollars:.4f}".partition(".")
    return f"${whole}.{part.rstrip('0').ljust(2, '0')}"


def _log_trim_summary(trimmed: int, considered: int, lone_trimmed: int,
                      lone_considered: int) -> None:
    """
    Log how many exact held pairs, and how many lone legs, the run trims.

    The line for exact pairs is logged on every call; the line for lone legs
    (one held market whose partner has paid out) only when at least one was
    looked at.

    Args:
        trimmed (int): Exact pairs part of which is sold.
        considered (int): Exact held pairs not picked for a take-profit sale.
        lone_trimmed (int): Lone legs part of which is sold.
        lone_considered (int): Lone legs not picked for a take-profit sale.
    """
    logging.info("Pairs to trim this run: %d of %d", trimmed, considered)
    if lone_considered:
        logging.info("Lone legs to trim this run (partner paid out): %d of %d",
                     lone_trimmed, lone_considered)


def _last_fill(pair: HeldPair, reads: _Reads, fills: FillTimes | None) -> datetime | None:
    """
    When a held position last traded: its newest fill, by the bot or by hand.

    Counts fills on the position's own markets and, for a lone leg, on any
    unheld market of its ladder in this run's list. Trims and add-ons both
    wait and measure price moves from this moment.

    Args:
        pair (HeldPair): An exact held pair or a lone leg.
        reads (_Reads): This planner call's reads (its market list is used).
        fills (FillTimes | None): Each market's last fill; None if not read.

    Returns:
        datetime | None: The newest such fill; None if fills is None or none is listed.
    """
    if fills is None:
        return None
    listed = [fills.latest[ticker] for ticker, _side in pair.sides if ticker in fills.latest]
    if pair.lone:
        held = {ticker for ticker, _side in pair.sides}
        # Cross-module: scanner's one reading of a market's ladder labels
        listed += [when for ticker, when in fills.latest.items()
                   if ticker not in held and ticker in reads.markets_by_ticker
                   and market_ladder_keys(reads.markets_by_ticker[ticker]) & pair.labels]
    return max(listed) if listed else None


def _too_near_maturity(pair: HeldPair, markets_by_ticker: dict, utc_date: date,
                       min_days: int | None) -> str | None:
    """
    Why a held position may not be rebalanced this run because it is too near maturity.

    "Rebalance until" (LiveSettings.rebalance_min_days_to_maturity): a Kelly
    trim or an add-on is made only while the position's later market closes
    at least `min_days` days from today. The later market is the latest close
    among its held markets: a pair's later market, or a lone leg's own (an
    add-on to a lone leg is judged on the market held, since the market it
    would be paired with is not known yet). Days are whole calendar days
    between UTC dates, as a take-profit sale counts them
    (config.days_to_maturity). Fails closed: under a minimum, a held market
    missing from this run's list, or whose close date cannot be read, holds
    the position back. Reads nothing from the exchange, so both planners ask
    it before anything is read for a position.

    Args:
        pair (HeldPair): An exact held pair or a lone leg.
        markets_by_ticker (dict): This run's market list by ticker.
        utc_date (date): Today, UTC.
        min_days (int | None): The fewest days left that still allow a
            rebalance (1 or 7); None rebalances however near maturity.

    Returns:
        str | None: The reason, for a "Not trimming" or "Not adding to" line;
            None when the position may be rebalanced.
    """
    if min_days is None:
        return None
    unchecked = f"so rebalancing until {min_days} day(s) before maturity cannot be checked"
    missing = [ticker for ticker, _side in pair.sides if ticker not in markets_by_ticker]
    if missing:
        return f"{missing[0]} is not in this run's market list, {unchecked}"
    # Cross-module: whole days from today (UTC) to the latest held close date
    left = days_to_maturity([_utc_date(markets_by_ticker[ticker].close_time)
                             for ticker, _side in pair.sides], utc_date)
    # Cross-module: config's one test of the days left
    if days_left_reached(left, min_days):
        return None
    if left is None:
        return f"a held market's close date is unknown, {unchecked}"
    if left == 0:
        closes = "closes today"
    else:
        closes = f"closes in {left} day(s)" if left > 0 else f"closed {-left} day(s) ago"
    return (f"its later market {closes}, and rebalancing stops {min_days} day(s) "
            "before maturity")


def _trim_one(pair: HeldPair, reads: _Reads, fills: FillTimes | None, *,
              settings: LiveSettings, min_days: int, now_date: date, utc_date: date,
              value: float, positions_value: float,
              fetched: list) -> tuple[SalePlan | None, str | None]:
    """
    Judge one exact held pair or lone leg for a Kelly trim, log the verdict, and plan the part sale.

    It is sold down when worth more than its Kelly share of the portfolio value
    (config.kelly_trim_count, the backtest's rule). A lone leg (its partner paid
    out $0) is trimmed as the later market of a time-series pair, with the
    partner read at its settlement price (config.settled_bids). It stops at the
    first failing step with one "Not trimming" line: markets listed, the wait
    since its last trade (_last_fill), the partner (_find_partner), its kind,
    readable books (their fetch value goes into `fetched`), the minimum price
    move, and a readable chance of paying. Only bids within
    KELLY_TRIM_MAX_BID_DROP of each best bid are sold into.

    Args:
        pair (HeldPair): An exact held pair or a lone leg, from scanner.held_pairs.
        reads (_Reads): This planner call's reads.
        fills (FillTimes | None): When each market last traded; None only with no
            wait in days and no minimum price move.
        settings (LiveSettings): The run's settings: k, both size caps and the
            minimum price move.
        min_days (int): Days a position must go without a trade (_rebalance_min_days).
        now_date (date): Today on the bot's own clock (_day).
        utc_date (date): Today in UTC, for the days left before maturity.
        value (float): What the trim is sized on, in dollars: cash plus open positions.
        positions_value (float): The open positions' part of that value, in dollars.
        fetched (list): Once its best bids are read, what it would fetch at its
            held sides' best bids after the fee, in dollars, is appended here.

    Returns:
        tuple[SalePlan | None, str | None]: (plan, None) when part of it is
            sold; (None, None) when it is kept; (None, why) when it alone would
            fetch more than all open positions are valued at, so that value is
            wrong and plan_trims trims nothing this run.

    Raises:
        ValueError: When KELLY_TRIM_MAX_BID_DROP is not a finite number at least 0.
    """
    text = _held_text(pair)

    def not_trimming(reason: str) -> None:
        """Log why this position is not trimmed."""
        logging.info("Not trimming %s: %s", text, reason)

    count_value = pair.count
    if not float(count_value).is_integer():
        # config.count_text: a contract count written exactly
        not_trimming(f"its count {count_text(count_value)} is not a whole number of contracts")
        return None, None
    count = int(count_value)
    missing = [ticker for ticker, _side in pair.sides if ticker not in reads.markets_by_ticker]
    if missing:
        not_trimming(f"{missing[0]} is not in this run's market list")
        return None, None
    # The held markets only: a lone leg's paid-out partner is read apart
    markets = {ticker: reads.markets_by_ticker[ticker] for ticker, _side in pair.sides}

    # Its last trade (None when no fill is listed). The wait and the price move
    # both count from it
    last_fill = _last_fill(pair, reads, fills)
    if min_days == 0:
        waited = "no wait"
    else:
        # No fill listed: its last trade is no later than the list's oldest fill
        since = (last_fill if last_fill is not None
                 else (fills.oldest if fills is not None else None))
        if since is None:
            not_trimming("when it last traded is not known")
            return None, None
        last = _day(since)
        days = (now_date - last).days
        # Cross-module: config's one test of the wait
        if not rebalance_rested(last, now_date, min_days):
            ago = (f"{days} day(s) ago" if days >= 0
                   else f"after today ({now_date}) on the bot's own clock")
            not_trimming(f"it last traded {last}, {ago}, and a trim waits until {min_days} "
                         "day(s) have passed since a pair's last trade")
            return None, None
        waited = (f"last traded {last}, {days} day(s) ago" if last_fill is not None
                  else f"no fill listed (the fills list starts {last})")

    by_side = {side: markets[ticker] for ticker, side in pair.sides}
    partner = None
    if pair.lone:
        lone_ticker, lone_side = pair.sides[0]
        other = _OTHER_SIDE[lone_side]
        # The one settled market held on the other side that asked this market's
        # question: its count is the held count, or at least as many once the leg
        # was trimmed (a trim leaves it holding fewer than its partner settled with)
        partner, problem = _find_partner(reads, lone_ticker, lone_side, count,
                                         pair.labels)
        if partner is None:
            not_trimming(problem)
            return None, None
        if partner.settled_at.timestamp() > reads.now_ts:
            # A payout dated after now (the exchange's clock runs ahead) has nothing to read it by
            not_trimming(f"its partner {partner.ticker} is recorded as settling after now")
            return None, None
        if partner.result == other:
            # The partner's side won, so the event this leg bets against has happened
            not_trimming(f"its partner {partner.ticker} paid out on its {other.upper()} side, "
                         "so this leg cannot win")
            return None, None
        text = (f"{lone_side.upper()} {lone_ticker} (alone: {other.upper()} "
                f"{partner.ticker} paid out $0)")
        partner_market = reads.label_market(partner.ticker)
        if partner_market is None:
            # Defence only: _find_partner keeps a partner only once its market was found
            not_trimming(f"the market of its partner {partner.ticker} could not be read")
            return None, None
        by_side[other] = partner_market
    # Cross-module: the pair's kind and which market is A (a lone leg's partner
    # standing in for the market that paid out)
    kind = held_pair_kind(by_side["yes"], by_side["no"])
    if kind is None:
        not_trimming("its two markets cannot be shown to be a same-title or a time-series "
                     "pair held this way round, so its chance of paying cannot be read")
        return None, None
    pair_type, market_a, market_b = kind
    if partner is not None and (pair_type != "time_series" or market_b.ticker != lone_ticker):
        # The forecast for what is left is the later market's chance of paying once
        # the earlier one has paid $0; no other lone leg has one
        not_trimming("a lone leg is trimmed only as the later market of a time-series pair")
        return None, None
    # Cross-module: the side each leg holds
    side_a, side_b = leg_sides(pair_type)

    # All four sides' bids, best first: the held sides price the sale, and a side's
    # ask is 1 minus the other side's best bid
    bids: dict = {}
    for market in (market_a, market_b):
        if partner is not None and market.ticker == partner.ticker:
            # Cross-module: a paid-out market reads as its settlement price, one level
            # deep without end, and no book or candle is read for it
            bids[(market.ticker, "yes")], bids[(market.ticker, "no")] = [
                [(price, math.inf)] for price in settled_bids(partner.result)]
            continue
        # Cross-module: the market's order book, a retried read-only GET
        book = _fetch_orderbook(reads.client, market.ticker)
        if book is None:
            not_trimming(f"the order book of {market.ticker} could not be read")
            return None, None
        for side in ("yes", "no"):
            # Cross-module: the side's usable bids, best first
            ladder = bid_ladder(book, side, ticker=market.ticker)
            if not ladder:
                not_trimming(f"{market.ticker} has no {side.upper()} bid")
                return None, None
            # Cross-module: whether an hourly candle could show this best bid
            if not candle_reads_bid(ladder[0][0], side):
                not_trimming(f"the best {side.upper()} bid on {market.ticker} "
                             f"(${ladder[0][0]:.4f}) is one the hourly price history cannot "
                             "show (a YES bid of 1c or less), and the backtest never trims "
                             "on such a quote")
                return None, None
            bids[(market.ticker, side)] = ladder
    yes_bid_a, no_bid_a = bids[(market_a.ticker, "yes")][0][0], bids[(market_a.ticker, "no")][0][0]
    yes_bid_b, no_bid_b = bids[(market_b.ticker, "yes")][0][0], bids[(market_b.ticker, "no")][0][0]

    # Cross-module: only each held side's bids near its best bid, so a gap in a book
    # is never sold into (a paid-out partner sells nothing, so it has no ladder)
    ladders = {market.ticker: kelly_trim_bids(bids[(market.ticker, side)],
                                              KELLY_TRIM_MAX_BID_DROP)
               for market, side in ((market_a, side_a), (market_b, side_b))
               if market.ticker in markets}
    # Each side held's best bid; a paid-out partner's is its settlement price ($0)
    bid_a, bid_b = bids[(market_a.ticker, side_a)][0][0], bids[(market_b.ticker, side_b)][0][0]
    # One contract pair's sale at the best bids, after the fee. Tested before the
    # price move and the chance, so a pair they hold back still flags a wrong value
    # Each market's fee multiplier, by ticker (market_a's goes with bid_a)
    fee_mults = {market.ticker: market_fee_multiplier(market)
                 for market in (market_a, market_b)}
    # Cross-module: the approximate fee on that sale
    top_value = bid_a + bid_b - fee_per_pair_approx(
        bid_a, bid_b, multiplier_a=fee_mults[market_a.ticker],
        multiplier_b=fee_mults[market_b.ticker])
    # What a lone leg sells is contracts of its one market; a pair's, contract pairs
    unit = "contracts" if pair.lone else "contract pairs"
    # plan_trims also tests these pairs together
    fetched.append(count * top_value)
    if count * top_value > positions_value:
        # One pair cannot be worth more than every open position together
        return None, (f"{text} alone would fetch {_dollars(count * top_value)} at its best "
                      f"bids ({count_text(count)} {unit} at {_dollars(top_value)} "
                      f"each), more than the {_dollars(positions_value)} all open positions "
                      "are valued at")

    move_text = ""
    if settings.min_price_move is not None:
        # Its price (the asks of the sides held, added) must have moved that far,
        # up or down, since its last trade
        if last_fill is None:
            where = "its market or its ladder" if pair.lone else "either market"
            not_trimming(f"no fill is listed on {where}, so its price when it last "
                         "traded cannot be read, and a minimum price move is set")
            return None, None
        moment = math.floor(last_fill.timestamp())
        # Its price then: each held side's ask at the last trade, from hourly candles. A
        # paid-out partner is not held: its part of the price is 0 then, as now (its
        # settled ask), so its payout never reads as a move
        then_a = reads.ask_then(market_a, side_a, moment) if market_a.ticker in markets else 0.0
        then_b = reads.ask_then(market_b, side_b, moment) if market_b.ticker in markets else 0.0
        # NaN (never equal to itself): no usable quote, or unreadable candles
        if then_a != then_a or then_b != then_b:
            unread, side = (market_a, side_a) if then_a != then_a else (market_b, side_b)
            who = "it" if pair.lone else "the pair"
            not_trimming(f"the {side.upper()} ask on {unread.ticker} when {who} last "
                         f"traded ({_day(last_fill)}) cannot be read (no usable quote in the "
                         f"{PRICE_THEN_MAX_AGE_DAYS} days before it, or its price "
                         "history could not be read), and a minimum price move is set")
            return None, None
        # Its price now, from the books just read (a paid-out partner's: 0)
        now_a = 1.0 - bids[(market_a.ticker, _other(side_a))][0][0]
        now_b = 1.0 - bids[(market_b.ticker, _other(side_b))][0][0]
        # Cross-module: the move and its test, both shared with the backtest (a
        # rise and a fall of the minimum count alike)
        move = pair_price_move(then_a + then_b, now_a + now_b)
        prices = (f"price {_price_text(then_a + then_b)} then, {_price_text(now_a + now_b)} now "
                  f"({_cents_text(move)}, minimum {settings.min_price_move * 100:.0f}c)")
        if not price_move_reached(move, settings.min_price_move):
            total = "price" if pair.lone else "total price"
            not_trimming(f"its {total} has not moved by the minimum since it last "
                         f"traded: {prices}")
            return None, None
        move_text = prices + ", "

    # Cross-module: the chance the pair pays, as a new trade of it would be sized
    # (each ask is 1 minus the other side's best bid; a paid-out partner's are its
    # settlement price)
    win_prob = held_pair_win_prob(pair_type, 1.0 - no_bid_a, 1.0 - yes_bid_a,
                                  1.0 - no_bid_b, 1.0 - yes_bid_b, settings.interval_discount)
    if win_prob is None:
        not_trimming("its chance of paying cannot be read from its quotes (a market's YES "
                     "ask sits below its own YES bid, or the two markets' midpoints leave "
                     "no gap between them)")
        return None, None

    def net_sale(pairs: int) -> float | None:
        """What selling `pairs` contract pairs (contracts, for a lone leg) returns
        after fees; None when the bids within reach hold fewer."""
        total = 0.0
        for ticker, ladder in ladders.items():
            # Cross-module: the one bid walk, then the taker fee on that market's
            # sale at its own fee multiplier
            walk = walk_bids(ladder, pairs)
            if walk is None:
                return None
            total += pairs * walk[0] - fee_leg_exact(pairs, walk[0],
                                                     multiplier=fee_mults[ticker])
        return total

    # Cross-module: the pair's size cap, as a new trade of its kind is capped
    cap = pair_size_cap(pair_type, settings.size_cap, settings.same_title_size_cap)
    # Cross-module: the backtest's trim rule
    decision = kelly_trim_count(count, value, win_prob, cap, top_value, net_sale)
    # Cross-module: the share Kelly lets the pair keep at its best bids, for the log line
    keep = min(cap, kelly_hold_fraction(win_prob, top_value))
    held_words = (f"{count_text(count)} contracts" if pair.lone
                  else f"{count_text(count)} each")
    each = "a contract" if pair.lone else "a pair"
    head = (f"Kelly trim check: {text} ({_KIND_WORDS.get(pair_type, pair_type)}), "
            f"{held_words}, {waited}, {move_text}pays {win_prob:.1%}, "
            f"{_dollars(top_value)} {each} at the best bids, may keep {keep:.1%} of "
            f"{_dollars(value)}")
    sell = decision.sell
    if not sell:
        logging.info("%s -> keep", head)
        return None, None
    walked = {ticker: walk_bids(ladder, sell) for ticker, ladder in ladders.items()}
    proceeds = net_sale(sell)
    if proceeds is None or None in walked.values():
        # Defence only: the rule returns a count the bids can take
        not_trimming(f"the bids within reach hold fewer than the {sell} {unit} a "
                     "trim would sell")
        return None, None
    logging.info("%s -> sell %d", head, sell)

    # Each held leg sells `sell` of the `count` held; its cost is that share of the
    # market's cost
    legs = tuple(
        SaleLeg(ticker, markets[ticker].event_ticker, side, sell,
                (reads.held_positions[ticker].exposure_dollars
                 + reads.held_positions[ticker].fees_dollars) * sell / count,
                market=markets[ticker], held=count)
        for ticker, side in pair.sides)
    if partner is not None:
        # The paid-out partner goes with them: its cost and $0 payout for the
        # `sell` contracts, as a lone take-profit sale carries it
        legs += (_partner_leg(reads, partner, other, sell),)
    cost = sum(leg.cost_dollars for leg in legs)
    # Cross-module: days to maturity from each held market's scheduled close date
    days_left = days_to_maturity([_utc_date(m.close_time) for m in markets.values()], utc_date)
    # Cross-module: the held market's title as the run's tables show it
    title = display_title(markets[pair.sides[0][0]])
    return SalePlan(title=title, legs=legs, count=sell, cost_dollars=cost, ladders=ladders,
                    walked=walked, proceeds_dollars=proceeds,
                    profits=((proceeds - cost, sell * CONTRACT_PAYOUT_DOLLARS - cost),),
                    days_left=days_left, level=None, kind=SALE_KIND_KELLY_TRIM), None


def plan_trims(client: Any, held_positions: dict, held_labels: dict, markets_by_ticker: dict,
               *, settings: LiveSettings, now: datetime, cash_cents: float,
               positions_value_cents: float | None,
               skip: frozenset = frozenset()) -> list[SalePlan]:
    """
    Plan this run's Kelly trims: part sales of held positions grown past their Kelly share.

    A position is an exact held pair or a lone leg (one held market whose
    partner has paid out), and each is judged by _trim_one. A lone leg is
    trimmed only as the later market of a time-series pair whose earlier
    market is found among the account's settlements, paid out $0 (the side
    held there lost). Fails closed: nothing is trimmed, with one log line
    saying why, unless all of these hold:

      - Kalshi's value of the open positions is known and above zero (a trim
        sized on too small a value sells too much).
      - Every held market's ladder is known.
      - With a wait in days or a minimum price move set, the account's fills
        could be read, and with a wait, the list reaches back that many days.
        With neither set, the fills are not read.
      - The positions whose best bids were read would not, alone or together,
        fetch more at those bids than all open positions are valued at. If they
        would, that value is wrong and no trim stands, even one logged "-> sell".

    And no lone leg is trimmed, with one WARNING, when the account's
    settlements cannot all be read (any record left out may be its partner).
    With settings.rebalance_min_days_to_maturity set, a position whose later
    market closes fewer than that many days from today (UTC dates) is left
    out first, with one "Not trimming" line, before anything is read for it
    (_too_near_maturity).
    A trim is sized on the cash plus the larger of Kalshi's value of the open
    positions and what the held markets would fetch at this run's bids, which
    can only trim less.

    Args:
        client (Any): The authenticated production Kalshi client, only read from.
        held_positions (dict): Ticker -> HeldPosition for every held market.
        held_labels (dict): Ticker -> ladder labels (scanner.resolve_held_ladders' labels_out).
        markets_by_ticker (dict): This run's whole market list by ticker, held markets included.
        settings (LiveSettings): rebalance (off: trim and read nothing), min_price_move,
            rebalance_min_days_to_maturity, k and both size caps.
        now (datetime): The run's moment, timezone-aware.
        cash_cents (float): The account's cash on every shard, in cents.
        positions_value_cents (float | None): Kalshi's value of the open positions, in
            cents; None when it was not read or was refused.
        skip (frozenset): Held tickers picked for a take-profit sale this run; a pair
            or lone leg with one is not looked at.

    Returns:
        list[SalePlan]: The part sales in ticker order (kind config.SALE_KIND_KELLY_TRIM);
            empty when none.

    Raises:
        ValueError: Before any request, when now is not timezone-aware, cash_cents is
            not a finite number at least 0, or REBALANCE_MIN_DAYS_SINCE_TRADE is
            invalid; later, when KELLY_TRIM_MAX_BID_DROP is invalid.
    """
    if not settings.rebalance:
        return []
    # Checked first, so a bad number stops the run before any request
    min_days = _rebalance_min_days()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if (isinstance(cash_cents, bool) or not isinstance(cash_cents, numbers.Real)
            or not math.isfinite(cash_cents) or cash_cents < 0):
        raise ValueError(f"cash_cents must be a finite number at least 0, got {cash_cents!r}")
    if not held_positions:
        _log_trim_summary(0, 0, 0, 0)
        return []
    # `not ... > 0` is also true for NaN
    if (isinstance(positions_value_cents, bool)
            or not isinstance(positions_value_cents, numbers.Real)
            or not positions_value_cents > 0 or not math.isfinite(positions_value_cents)):
        logging.warning(
            "Not trimming any held pair this run: Kalshi's value of the open positions is "
            "not known or is not above $0, and a trim sized on too small a value would "
            "sell too much")
        _log_trim_summary(0, 0, 0, 0)
        return []
    unknown = sorted((t for t in held_positions if not held_labels.get(t)), key=str)
    if unknown:
        # An unknown ladder could be any pair's, so no pair is provably alone on its ladder
        logging.info("Not trimming this run: the ladder of %d held market(s) is unknown (%s)",
                     len(unknown), ", ".join(str(t) for t in unknown))
        _log_trim_summary(0, 0, 0, 0)
        return []

    # Cross-module: the one definition of exact held pairs and lone held markets, quietly
    found = held_pairs(held_positions, held_labels, markets_by_ticker, log=False)
    # Exact pairs and lone legs (a lone leg is judged once its paid-out partner is
    # found), none whose markets a take-profit sale already takes
    pairs = {key: pair for key, pair in found.items() if not key & skip}
    for key in sorted(found, key=sorted):
        if key & skip:
            logging.info("Not trimming %s: it is picked for a take-profit sale this run",
                         _held_text(found[key]))
    # The summary counts exact pairs and lone legs apart
    lone = sum(1 for pair in pairs.values() if pair.lone)
    exact = len(pairs) - lone
    if not pairs:
        _log_trim_summary(0, 0, 0, 0)
        return []
    # "Rebalance until": a position whose later market closes too soon is left out
    # first, so nothing is read for it
    utc_date = now.astimezone(UTC).date()
    for key in sorted(pairs, key=sorted):
        near = _too_near_maturity(pairs[key], markets_by_ticker, utc_date,
                                  settings.rebalance_min_days_to_maturity)
        if near is not None:
            logging.info("Not trimming %s: %s", _held_text(pairs[key]), near)
            del pairs[key]
    if not pairs:
        _log_trim_summary(0, exact, 0, lone)
        return []

    # Today on the bot's own clock, as every fill's day is read
    now_date = _day(now)
    fills = None
    if min_days > 0 or settings.min_price_move is not None:
        # Cross-module: one read of the account's fills serves the wait and the price move
        fills = get_fill_times(client)
        if fills is None:
            logging.warning(
                "Not trimming any held pair this run: the account's fills could not be "
                "read, so when each pair last traded is not known")
            _log_trim_summary(0, exact, 0, lone)
            return []
    if min_days > 0:
        logging.info(
            "Kelly trim: a pair is looked at once %d day(s) have passed since its last "
            "trade, counted in calendar days on the bot's own clock (%s), where today is %s",
            min_days, SCHEDULED_RUN.timezone, now_date)
        # The list must reach back past the wait: a pair with no fill listed is dated
        # by the list's oldest fill
        problem = None
        if fills.oldest is None:
            problem = "the account's fills list is empty"
        # Cross-module: config's one test of the wait, asked of the list's oldest day
        elif not rebalance_rested(_day(fills.oldest), now_date, min_days):
            problem = (f"the account's fills list starts {_day(fills.oldest)} on the bot's "
                       f"own clock, fewer than {min_days} day(s) ago")
        if problem is not None:
            logging.warning(
                "Not trimming any held pair this run: %s, so a trade in the last %d day(s) "
                "cannot be ruled out", problem, min_days)
            _log_trim_summary(0, exact, 0, lone)
            return []

    # Cross-module: what the held markets would fetch at this run's bids. A trim is
    # sized on the larger of that and Kalshi's value, which can only trim less
    floor = held_positions_bid_value(held_positions, markets_by_ticker)
    kalshi_value = positions_value_cents / 100
    positions_value = max(kalshi_value, floor)
    value = cash_cents / 100 + positions_value
    logging.info(
        "Kelly trim sizes on %s = cash %s + open positions %s (Kalshi's value %s; %s at "
        "this run's bids)", _dollars(value), _dollars(cash_cents / 100),
        _dollars(positions_value), _dollars(kalshi_value), _dollars(floor))
    if kalshi_value < floor:
        logging.warning(
            "Kalshi's value of the open positions (%s) is below what they would fetch at "
            "this run's bids (%s): the Kelly trim is sized on the larger, so it can only "
            "sell less", _dollars(kalshi_value), _dollars(floor))

    # 0 daily checks: a trim reads only the books now, a lone leg's settled partner
    # and, with a minimum price move, each market's price at its last fill
    reads = _Reads(client, held_positions, markets_by_ticker, now.timestamp(), 0,
                   purpose="trimming")
    plans: list[SalePlan] = []
    lone_trimmed = 0
    doubts: list[str] = []
    # What each pair whose books were read would fetch at its best bids after the fee
    fetched: list[float] = []
    for key in sorted(pairs, key=sorted):
        plan, doubt = _trim_one(pairs[key], reads, fills, settings=settings,
                                min_days=min_days, now_date=now_date, utc_date=utc_date,
                                value=value, positions_value=positions_value,
                                fetched=fetched)
        if doubt is not None:
            doubts.append(doubt)
        elif plan is not None:
            plans.append(plan)
            if pairs[key].lone:
                lone_trimmed += 1
    total = sum(fetched)
    # A lone leg is one market, not a pair, so a run that looked at one names both "positions"
    noun = "positions" if lone else "pairs"
    if not doubts and total > positions_value:
        # Together they would fetch more than all open positions: the value is just as wrong
        doubts.append(f"the {len(fetched)} {noun} whose order books were read would fetch "
                      f"{_dollars(total)} at their best bids together, more than the "
                      f"{_dollars(positions_value)} all open positions are valued at")
    if doubts:
        # Every trim above was sized on that wrong value, so none stands
        alike = "position(s)" if lone else "pair(s)"
        more = f" (and {len(doubts) - 1} more {alike} alike)" if len(doubts) > 1 else ""
        logging.warning(
            "Not trimming any held pair this run, including any marked '-> sell' above: "
            "%s%s, so that value cannot be right, and a trim sized on too small a value "
            "would sell too much", doubts[0], more)
        _log_trim_summary(0, exact, 0, lone)
        return []
    _log_trim_summary(len(plans) - lone_trimmed, exact, lone_trimmed, lone)
    return plans


# ─── Adding to a held pair: the wait and the price move ───────────────────────

def _log_add_on_summary(kept: int, considered: int) -> None:
    """
    Log how many held pairs (and lone legs) a run may add to.

    Args:
        kept (int): Those that passed the wait and, with a minimum, the price move.
        considered (int): Those handed in.
    """
    logging.info("Held pairs that may be added to this run: %d of %d", kept, considered)


def _add_on_one(pair: HeldPair, reads: _Reads, fills: FillTimes, *,
                min_move: float | None, min_days: int, now_date: date) -> bool:
    """
    Judge one held pair (or lone leg) for an add-on, and log the verdict.

    It stops at the first step that fails, with one "Not adding to" line:

      1. The wait: enough calendar days since its last trade (_last_fill, the
         Kelly trim's own reading: the last fill on both of an exact pair's
         markets; for a lone leg, on its own market and any market of its
         ladder in this run's list, since the finders pair a lone leg only
         with one of those, so a partner sold by hand restarts the wait).
         With no fill listed, it last traded no later than the fills list's
         oldest fill.
      2. With a minimum price move set: a fill is listed, so its price when it
         last traded can be read.
      3. Its price now: the ask of the side held on each market, from this
         run's market list (scanner.listed_ask, no request). Read before the
         price then, so a market the list lacks costs no candle read.
      4. Its price then: each held side's ask at its last fill, from hourly
         candles (_Reads.ask_then, as the Kelly trim reads it).
      5. The move: the price now less the price then, up or down, at least
         the minimum (config.pair_price_move, price_move_reached, the rule
         the backtest's add-ons and every trim share).

    An exact pair's price is the two held sides' asks, added; a lone leg's
    is its one market's ask.

    Args:
        pair (HeldPair): An exact held pair or a lone leg, from scanner.held_pairs.
        reads (_Reads): This planner call's reads (only the candles and this
            run's market list are used).
        fills (FillTimes): When each market last traded (plan_add_ons reads it
            once, and calls this only when it was read).
        min_move (float | None): The minimum price move in dollars; None for none.
        min_days (int): Days a pair must go without a trade (_rebalance_min_days).
        now_date (date): Today on the bot's own clock (_day).

    Returns:
        bool: True when the run may add to it.
    """
    text = _sides_text(pair.sides)

    def not_adding(reason: str) -> None:
        """Log why this pair is not added to."""
        logging.info("Not adding to %s: %s", text, reason)

    # config.count_text: a contract count written exactly
    count = f"{count_text(pair.count)} contracts" + ("" if pair.lone else " each")
    where = "its market" if pair.lone else "either market"
    # The last trade: the latest fill on its markets (for a lone leg, also on any
    # market of its ladder in this run's list; None when none is listed). The wait
    # and the price move both count from it, as the Kelly trim's do
    last_fill = _last_fill(pair, reads, fills)
    if min_days == 0:
        waited = "no wait"
    else:
        # No fill listed: its last trade is no later than the list's oldest fill
        since = last_fill if last_fill is not None else fills.oldest
        if since is None:
            not_adding("when it last traded is not known")
            return False
        last = _day(since)
        days = (now_date - last).days
        # Cross-module: config's one test of the wait, the trim's too
        if not rebalance_rested(last, now_date, min_days):
            ago = (f"{days} day(s) ago" if days >= 0
                   else f"after today ({now_date}) on the bot's own clock")
            not_adding(f"it last traded {last}, {ago}, and an add-on waits until {min_days} "
                       "day(s) have passed since a pair's last trade")
            return False
        waited = (f"last traded {last}, {days} day(s) ago" if last_fill is not None
                  else f"no fill listed (the fills list starts {last})")

    prices = ""
    if min_move is not None:
        if last_fill is None:
            not_adding(f"no fill is listed on {where}, so its price when it last traded "
                       "cannot be read, and a minimum price movement is set")
            return False
        # Its price now: the ask of each side held, as this run's market list shows it
        now_total = 0.0
        for ticker, side in pair.sides:
            # Cross-module: the one reading of a held market's ask now
            ask = listed_ask(reads.markets_by_ticker.get(ticker), side)
            if ask is None:
                not_adding(f"the {side.upper()} ask on {ticker} is not in this run's market "
                           "list or not a price strictly between 0 and 1, and a minimum "
                           "price movement is set")
                return False
            now_total += ask
        # Its price then: each side's ask at the last fill, from hourly candles
        moment = math.floor(last_fill.timestamp())
        then_total = 0.0
        for ticker, side in pair.sides:
            then = reads.ask_then(reads.markets_by_ticker[ticker], side, moment)
            # NaN (never equal to itself): no usable quote, or unreadable candles
            if then != then:
                not_adding(f"the {side.upper()} ask on {ticker} when it last traded "
                           f"({_day(last_fill)}) cannot be read (no usable quote in the "
                           f"{PRICE_THEN_MAX_AGE_DAYS} days before it, or its price history "
                           "could not be read), and a minimum price movement is set")
                return False
            then_total += then
        # Cross-module: the move and its test, both shared with the backtest and the
        # trim (a rise and a fall of the minimum count alike)
        move = pair_price_move(then_total, now_total)
        prices = (f"price {_price_text(then_total)} then, {_price_text(now_total)} now "
                  f"({_cents_text(move)}, minimum {min_move * 100:.0f}c)")
        if not price_move_reached(move, min_move):
            total = "price" if pair.lone else "total price"
            not_adding(f"its {total} has not moved by the minimum since it last traded: "
                       f"{prices}")
            return False
        prices = ", " + prices
    logging.info("Add-on check: %s, %s, %s%s -> may add", text, count, waited, prices)
    return True


def plan_add_ons(client: Any, add_on_pairs: dict, markets_by_ticker: dict, *,
                 settings: LiveSettings, now: datetime) -> dict:
    """
    Keep the held pairs and lone legs this run may add to: rested and, with a minimum, moved.

    An add-on is a rebalance, so it follows the Kelly trim's two rules
    (plan_trims): at least REBALANCE_MIN_DAYS_SINCE_TRADE calendar days since
    the last fill on any of its markets, by the bot or by hand
    (config.rebalance_rested, on the bot's own clock; for a lone leg, also on
    any market of its ladder in this run's list, the one kind of market the
    finders pair it with, so a partner sold by hand is not bought back the
    next day), and, with
    settings.min_price_move, a move of at least that many dollars, up or
    down, since that fill (config.pair_price_move, price_move_reached). An
    exact pair's price is the asks of the sides held on its two markets,
    added; a lone leg's is its one market's. The price then comes from
    hourly candles at the last fill (_Reads.ask_then, as the trim reads it);
    the price now from this run's market list (scanner.listed_ask). Each pair
    is judged by _add_on_one. Before any of that, with
    settings.rebalance_min_days_to_maturity set, a pair whose later market
    closes fewer than that many days from today (UTC dates; a lone leg's own
    market) is left out, with one "Not adding to" line (_too_near_maturity).

    Fails closed: a pair whose wait, price then or price now cannot be read
    is left out, and when the account's fills cannot be read, or (with a
    wait) the list does not reach back that many days, none is kept, with
    one WARNING. The account's fills are read once, and only when there is
    something to add to and a wait or a minimum is set.

    Args:
        client (Any): The authenticated production Kalshi client, only read from.
        add_on_pairs (dict): frozenset of tickers -> HeldPair, what scanner.held_pairs
            found the run may add to (less any pair at its cap or picked for a sale).
        markets_by_ticker (dict): This run's whole market list by ticker, held markets included.
        settings (LiveSettings): rebalance (off: keep nothing, read nothing),
            min_price_move and rebalance_min_days_to_maturity.
        now (datetime): The run's moment, timezone-aware.

    Returns:
        dict: The entries of add_on_pairs the run may add to, the same HeldPair
            objects under the same keys; empty when none.

    Raises:
        ValueError: Before any request, when REBALANCE_MIN_DAYS_SINCE_TRADE is
            invalid or now is not timezone-aware.
    """
    if not settings.rebalance:
        return {}
    # Checked first, so a bad number stops the run before any request
    min_days = _rebalance_min_days()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if not add_on_pairs:
        return {}
    # "Rebalance until": a position whose later market closes too soon is left out
    # first, so nothing is read for it
    near_maturity = set()
    if settings.rebalance_min_days_to_maturity is not None:
        utc_date = now.astimezone(UTC).date()
        for key in sorted(add_on_pairs, key=sorted):
            near = _too_near_maturity(add_on_pairs[key], markets_by_ticker, utc_date,
                                      settings.rebalance_min_days_to_maturity)
            if near is not None:
                logging.info("Not adding to %s: %s", _sides_text(add_on_pairs[key].sides),
                             near)
                near_maturity.add(key)
    pairs = {key: pair for key, pair in add_on_pairs.items() if key not in near_maturity}
    if not pairs:
        _log_add_on_summary(0, len(add_on_pairs))
        return {}
    min_move = settings.min_price_move
    if min_days == 0 and min_move is None:
        # Nothing to wait for, so nothing is read
        _log_add_on_summary(len(pairs), len(add_on_pairs))
        return pairs

    # Today on the bot's own clock, as every fill's day is read
    now_date = _day(now)
    # Cross-module: one read of the account's fills serves the wait and the price move
    fills = get_fill_times(client)
    if fills is None:
        logging.warning(
            "Not adding to held pairs this run: the account's fills could not be read, so "
            "when each pair last traded is not known")
        _log_add_on_summary(0, len(add_on_pairs))
        return {}
    if min_days > 0:
        # The list must reach back past the wait: a pair with no fill listed is dated
        # by the list's oldest fill
        problem = None
        if fills.oldest is None:
            problem = "the account's fills list is empty"
        # Cross-module: config's one test of the wait, asked of the list's oldest day
        elif not rebalance_rested(_day(fills.oldest), now_date, min_days):
            problem = (f"the account's fills list starts {_day(fills.oldest)} on the bot's "
                       f"own clock, fewer than {min_days} day(s) ago")
        if problem is not None:
            logging.warning(
                "Not adding to held pairs this run: %s, so a trade in the last %d day(s) "
                "cannot be ruled out", problem, min_days)
            _log_add_on_summary(0, len(add_on_pairs))
            return {}

    # 0 daily checks: an add-on reads only each market's price at its pair's last fill;
    # no position is read
    reads = _Reads(client, {}, markets_by_ticker, now.timestamp(), 0)
    kept: dict = {}
    for key in sorted(pairs, key=sorted):
        if _add_on_one(pairs[key], reads, fills, min_move=min_move,
                       min_days=min_days, now_date=now_date):
            kept[key] = pairs[key]
    _log_add_on_summary(len(kept), len(add_on_pairs))
    return kept
