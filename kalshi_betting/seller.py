"""
File: seller.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Decides which held positions the take-profit rule sells this run and plans
    each sale; main.py hands the plans to trader.sell_positions. A position is an
    exact held pair (one YES and one NO market of equal count), or a lone held
    market with its paid-out partner (a settled market the account held on the other
    side at the same count, asking the same question). Its held markets are alone on
    their ladder (one question asked at several deadlines). It is sold when its realized
    profit (what selling its held markets returns after fees, plus any partner's
    payout, less its cost) has reached LiveSettings.sell_at of its potential
    profit (what it pays if it wins, less its cost) at each of
    config.TAKE_PROFIT_HOLD_DAYS daily checks, the last one now. The rule in
    full, and where it differs from the backtest: CLAUDE.md (live selling).

    It also plans Kelly trims (plan_trims): selling part of an exact held pair
    that is worth more than its Kelly share of the portfolio value (the most
    Kelly sizing would put in it), by the backtest's rule
    (config.kelly_trim_count). A pair is trimmed only after enough calendar
    days without a fill (a trade) on either market and, with
    LiveSettings.trim_min_rise set, once its price has risen that far since.
    A trim sells only into bids within config.KELLY_TRIM_MAX_BID_DROP of the
    best bid, never into a gap in a book.

Dependencies:
    Imports the standard library, config, scanner and historical only, never
    trader, strategy, the backtester or main (pinned by tests/test_seller.py).
    main.py alone imports it: a production run calls plan_sales before it
    buys anything and, with trim_to_kelly on, plan_trims right after it.

Notes:
    Every request is a read-only GET. Each planner call reads for itself,
    each thing at most once. Fails closed: anything unknown or unreadable
    means no sale and no trim, with a log line.
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
    KELLY_TRIM_MIN_DAYS_SINCE_TRADE,
    MIN_ACTIVE_PRICE_DOLLARS,
    SALE_KIND_KELLY_TRIM,
    SALE_KIND_TAKE_PROFIT,
    SALE_PARTNER_MAX_AGE_DAYS,
    SALE_PARTNER_MAX_LOOKUPS,
    SCHEDULED_RUN,
    TAKE_PROFIT_HOLD_DAYS,
    TAKE_PROFIT_HOLD_DAYS_MAX,
    TRIM_PRICE_THEN_MAX_AGE_DAYS,
    LiveSettings,
    count_text,
    days_to_maturity,
    fee_leg_exact,
    fee_per_pair_approx,
    held_pair_win_prob,
    kelly_hold_fraction,
    kelly_trim_bids,
    kelly_trim_count,
    kelly_trim_min_days,
    kelly_trim_rested,
    pair_price_rise,
    pair_size_cap,
    reached_every_check,
    take_profit_reached,
    trade_day,
    trim_rise_reached,
)
from .historical import ask_before, bid_before, candle_reads_bid, recent_candles
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
            trim, the contracts sold there.
        cost_dollars (float): What they cost, fees included.
        market (Any): The ApiMarket from this run's market list while held; None for a partner.
        payout_dollars (float | None): What a partner paid when it settled; None while held.
        paid_at (datetime | None): When a partner settled (UTC); None while held.
        held (int | None): For a Kelly trim, the contracts held there; None when
            `count` is all of them.
    """
    ticker: str
    event_ticker: str
    side: str
    count: int
    cost_dollars: float
    market: Any = None
    payout_dollars: float | None = None
    paid_at: datetime | None = None
    held: int | None = None


@dataclass(frozen=True)
class SalePlan:
    """
    Held contracts a run sells, with what the sale orders need.

    A take-profit sale (plan_sales) sells a whole position. A Kelly trim
    (plan_trims) sells part of an exact held pair; its count and money figures
    are those of the part sold.

    Attributes:
        title (str): The first held market's display title.
        legs (tuple[SaleLeg, ...]): Its held markets in ticker order, then its partner if any.
        count (int): Contract pairs sold (for a take-profit sale, all held on each market).
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


def _trim_min_days() -> int:
    """
    Check how many days a held pair must go without a trade before a Kelly trim.

    Reads seller.KELLY_TRIM_MIN_DAYS_SINCE_TRADE when called: a test patches it
    here, never config's.

    Returns:
        int: A whole number from 0 (no wait) to config.KELLY_TRIM_MIN_DAYS_SINCE_TRADE_MAX.

    Raises:
        ValueError: For any other value, a bool or a float included.
    """
    # Cross-module: config's one check of this number
    return kelly_trim_min_days(KELLY_TRIM_MIN_DAYS_SINCE_TRADE)


def _day(moment: datetime) -> date:
    """
    The calendar day a moment falls on, on the bot's own clock (its schedule's time zone).

    The Kelly trim's wait is counted between days read here: today's and a
    fill's. Reads seller.SCHEDULED_RUN when called: a test patches it here,
    never config's.

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
    The contracts of one side a settled market's settlement says the account held.

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
        held_positions (dict): Ticker -> HeldPosition for every held market.
        markets_by_ticker (dict): This run's whole market list by ticker.
        now_ts (float): The run's moment, in Unix seconds.
        hold_days (int): How many daily checks there are (0 for a Kelly trim, which
            makes none).
        lookups (int): Label lookups from the exchange so far (SALE_PARTNER_MAX_LOOKUPS at most).
    """

    def __init__(self, client: Any, held_positions: dict, markets_by_ticker: dict,
                 now_ts: float, hold_days: int) -> None:
        """Start with nothing read; each argument is the attribute of the same name."""
        self.client = client
        self.held_positions = held_positions
        self.markets_by_ticker = markets_by_ticker
        self.now_ts = now_ts
        self.hold_days = hold_days
        self.lookups = 0
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

        The Kelly trim's minimum price rise compares a pair's price now with its
        price at its last fill; this reads one market's part of the price then.
        The candles cover the config.TRIM_PRICE_THEN_MAX_AGE_DAYS days before the
        moment and are fetched once.

        Args:
            market (Any): The held market.
            side (str): The side held there, "yes" or "no".
            moment (int): The past moment, in Unix seconds.

        Returns:
            float: The side's latest usable ask at or before the moment, in dollars;
                NaN when the candles could not be read or hold none in those days.
        """
        window = TRIM_PRICE_THEN_MAX_AGE_DAYS * _DAY_SECONDS
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
        used and no lone held market is sold this run (fails closed, one WARNING).

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
            if self._settlements_problem:
                logging.warning("Not selling any held market whose partner has paid out this "
                                "run: %s", self._settlements_problem)
        return self._settlements, self._settlements_problem

    def labels(self, ticker: str) -> tuple[frozenset | None, str]:
        """
        A settled market's ladder labels, found once: in this run's market list, or looked up.

        Args:
            ticker (str): The settled market's ticker.

        Returns:
            tuple[frozenset | None, str]: (its labels, "") or (None, why not found).
        """
        if ticker in self._labels:
            return self._labels[ticker]
        listed = ticker in self.markets_by_ticker
        if not listed:
            if self.lookups >= SALE_PARTNER_MAX_LOOKUPS:
                return None, (f"the limit of {SALE_PARTNER_MAX_LOOKUPS} partner lookups "
                              "this run was reached")
            self.lookups += 1
        # Cross-module: from the run's list, else looked up as held markets are, so its labels
        # match the held market's own (_find_partner compares their question labels)
        market = market_for_labels(self.client, ticker, self.markets_by_ticker,
                                   self._event_titles)
        if market is None:
            result = (None, f"settled market {ticker} could not be looked up, and it may "
                            "be the partner")
        else:
            # Cross-module: the one definition of a market's ladder labels
            result = (market_ladder_keys(market), "")
        self._labels[ticker] = result
        return result


def _find_partner(reads: _Reads, ticker: str, side: str, count: int,
                  labels: frozenset) -> tuple[Settlement | None, str]:
    """
    Find a lone held market's one paid-out partner among the account's settlements.

    It is the one settled market no longer held, held on the other side with the
    same count (none on the held side), asking the held market's question (every
    pair the bot buys asks one question, so a market of its event that asks
    another was never its pair-mate), and settled no earlier than
    SALE_PARTNER_MAX_AGE_DAYS before now (one dated after now is returned, and
    _plan_one refuses it). Fails closed: a held market with no question label, no
    partner, two or more, an unreadable label, or a partner settled other than yes
    or no means no sale.

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
    candidates = [s for s in settlements
                  if s.ticker not in reads.held_positions and s.ticker != ticker
                  and _settled_count(s, other) == count and _settled_count(s, side) == 0
                  and s.settled_at.timestamp() >= oldest]
    partners: list[Settlement] = []
    for settlement in candidates:
        found, problem = reads.labels(settlement.ticker)
        if found is None:
            return None, problem
        if question not in found:
            continue
        partners.append(settlement)
        if len(partners) > 1:
            break
    if not partners:
        # config.count_text: a contract count written exactly
        return None, (f"no paid-out partner: no market it no longer holds was held "
                      f"{other.upper()} with {count_text(count)} contracts on its question "
                      f"and paid out in the last {SALE_PARTNER_MAX_AGE_DAYS} days")
    if len(partners) > 1:
        return None, ("more than one settled market could be its paid-out partner ("
                      + ", ".join(s.ticker for s in partners) + ")")
    partner = partners[0]
    if partner.result not in ("yes", "no"):
        return None, (f"its partner {partner.ticker} settled {partner.result} rather than "
                      "yes or no, which the rule does not judge")
    return partner, ""


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
    text = _sides_text(pair.sides) + (" (alone on its ladder)" if pair.lone else "")

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
        # The partner's cost: what was paid for the side held there, plus its fees
        cost_basis = partner.yes_cost_dollars if other == "yes" else partner.no_cost_dollars
        legs.append(SaleLeg(partner.ticker, partner.event_ticker, other, count,
                            cost_basis + partner.fees_dollars,
                            payout_dollars=partner.revenue_dollars,
                            paid_at=partner.settled_at))
        text = (f"{side.upper()} {ticker} with paid-out {other.upper()} {partner.ticker} "
                f"(settled {partner.result}, paid {_dollars(partner.revenue_dollars)})")

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
    proceeds = sum(count * walked[leg.ticker][0] - fee_leg_exact(count, walked[leg.ticker][0])
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
            # Cross-module: the taker fee on selling `count` at that bid
            value += count * bid - fee_leg_exact(count, bid)
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
               *, settings: LiveSettings, now: datetime) -> list[SalePlan]:
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
    reads = _Reads(client, held_positions, markets_by_ticker, now.timestamp(), hold_days)
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

    Written to the hundredth of a cent (Kalshi's finest price step), so a rise
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


def _log_trim_summary(trimmed: int, considered: int) -> None:
    """
    Log how many held pairs the run trims.

    Args:
        trimmed (int): Pairs part of which is sold.
        considered (int): Exact held pairs not picked for a take-profit sale.
    """
    logging.info("Pairs to trim this run: %d of %d", trimmed, considered)


def _trim_one(pair: HeldPair, reads: _Reads, fills: FillTimes | None, *,
              settings: LiveSettings, min_days: int, now_date: date, utc_date: date,
              value: float, positions_value: float,
              fetched: list) -> tuple[SalePlan | None, str | None]:
    """
    Judge one exact held pair for a Kelly trim, log the verdict, and plan the part sale.

    The pair is sold down when it is worth more than its Kelly share of the
    portfolio value, each contract pair counted at what selling it now returns
    (config.kelly_trim_count, the backtest's rule). It stops at the first step
    that fails, with one "Not trimming" line:

      1. A whole count, and both markets in this run's market list.
      2. The wait: enough calendar days since the last fill on either market.
         A pair with no fill listed counts from the list's oldest fill.
      3. Its kind (same-title or time-series) and which market is A.
      4. Both order books, with a best bid on all four sides that an hourly
         candle could show, so it is judged only on quotes the backtest could
         see. What the pair would fetch at its held sides' best bids, after the
         fee, goes into `fetched`; if that alone is more than all open
         positions are valued at, it stops here (see Returns).
      5. With a minimum price rise set: the pair's total price (the asks of
         the sides held, added) has risen that far since its last fill.
      6. Its chance of paying can be read (config.held_pair_win_prob).

    The sale is sized and priced only on bids within KELLY_TRIM_MAX_BID_DROP of
    each held side's best bid (patch seller.KELLY_TRIM_MAX_BID_DROP in a test),
    so it never sells into a gap in a book.

    Args:
        pair (HeldPair): An exact held pair, never a lone market.
        reads (_Reads): This planner call's reads.
        fills (FillTimes | None): When each market last traded; None only with no
            wait in days and no minimum price rise.
        settings (LiveSettings): The run's settings: k, both size caps and the
            minimum price rise.
        min_days (int): Days a pair must go without a trade (_trim_min_days).
        now_date (date): Today on the bot's own clock (_day).
        utc_date (date): Today in UTC, for the days left before maturity.
        value (float): What the trim is sized on, in dollars: cash plus open positions.
        positions_value (float): The open positions' part of that value, in dollars.
        fetched (list): Once the four best bids are read, what the pair would fetch
            at its held sides' best bids after the fee, in dollars, is appended here.

    Returns:
        tuple[SalePlan | None, str | None]: (plan, None) when part of the pair is
            sold; (None, None) when it is kept; (None, why) when the pair alone
            would fetch more than all open positions are valued at, so that value
            is wrong and plan_trims trims nothing this run.

    Raises:
        ValueError: When KELLY_TRIM_MAX_BID_DROP is not a finite number at least 0.
    """
    text = _sides_text(pair.sides)

    def not_trimming(reason: str) -> None:
        """Log why this pair is not trimmed."""
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
    markets = {ticker: reads.markets_by_ticker[ticker] for ticker, _side in pair.sides}

    # The pair's last trade: the latest fill on either market (None when none is
    # listed). The wait and the price rise both count from it
    listed = ([fills.latest[ticker] for ticker, _side in pair.sides if ticker in fills.latest]
              if fills is not None else [])
    last_fill = max(listed) if listed else None
    if min_days == 0:
        waited = "no wait"
    else:
        # No fill listed: its last trade is no later than the list's oldest fill
        since = last_fill if listed else (fills.oldest if fills is not None else None)
        if since is None:
            not_trimming("when it last traded is not known")
            return None, None
        last = _day(since)
        days = (now_date - last).days
        # Cross-module: config's one test of the wait
        if not kelly_trim_rested(last, now_date, min_days):
            ago = (f"{days} day(s) ago" if days >= 0
                   else f"after today ({now_date}) on the bot's own clock")
            not_trimming(f"it last traded {last}, {ago}, and a trim waits until {min_days} "
                         "day(s) have passed since a pair's last trade")
            return None, None
        waited = (f"last traded {last}, {days} day(s) ago" if listed
                  else f"no fill listed (the fills list starts {last})")

    by_side = {side: markets[ticker] for ticker, side in pair.sides}
    # Cross-module: the pair's kind and which market is A
    kind = held_pair_kind(by_side["yes"], by_side["no"])
    if kind is None:
        not_trimming("its two markets cannot be shown to be a same-title or a time-series "
                     "pair held this way round, so its chance of paying cannot be read")
        return None, None
    pair_type, market_a, market_b = kind
    # Cross-module: the side each leg holds
    side_a, side_b = leg_sides(pair_type)

    # All four sides' bids, best first: the held sides price the sale, and a side's
    # ask is 1 minus the other side's best bid
    bids: dict = {}
    for market in (market_a, market_b):
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

    # Cross-module: only the held side's bids near its best bid, so a gap in a book
    # is never sold into
    ladders = {market.ticker: kelly_trim_bids(bids[(market.ticker, side)],
                                              KELLY_TRIM_MAX_BID_DROP)
               for market, side in ((market_a, side_a), (market_b, side_b))}
    bid_a, bid_b = ladders[market_a.ticker][0][0], ladders[market_b.ticker][0][0]
    # One contract pair's sale at the best bids, after the fee. Tested before the
    # price rise and the chance, so a pair they hold back still flags a wrong value
    # Cross-module: the approximate fee on that sale
    top_value = bid_a + bid_b - fee_per_pair_approx(bid_a, bid_b)
    # plan_trims also tests these pairs together
    fetched.append(count * top_value)
    if count * top_value > positions_value:
        # One pair cannot be worth more than every open position together
        return None, (f"{text} alone would fetch {_dollars(count * top_value)} at its best "
                      f"bids ({count_text(count)} contract pairs at {_dollars(top_value)} "
                      f"each), more than the {_dollars(positions_value)} all open positions "
                      "are valued at")

    rise_text = ""
    if settings.trim_min_rise is not None:
        # The pair's total price (the asks of the sides held, added) must have risen
        # that far since its last fill
        if last_fill is None:
            not_trimming("no fill is listed on either market, so its price when it last "
                         "traded cannot be read, and a minimum price rise is set")
            return None, None
        moment = math.floor(last_fill.timestamp())
        # Its price then: each held side's ask at the last fill, from hourly candles
        then_a = reads.ask_then(market_a, side_a, moment)
        then_b = reads.ask_then(market_b, side_b, moment)
        # NaN (never equal to itself): no usable quote, or unreadable candles
        if then_a != then_a or then_b != then_b:
            unread, side = (market_a, side_a) if then_a != then_a else (market_b, side_b)
            not_trimming(f"the {side.upper()} ask on {unread.ticker} when the pair last "
                         f"traded ({_day(last_fill)}) cannot be read (no usable quote in the "
                         f"{TRIM_PRICE_THEN_MAX_AGE_DAYS} days before it, or its price "
                         "history could not be read), and a minimum price rise is set")
            return None, None
        # Its price now, from the books just read
        now_a = 1.0 - bids[(market_a.ticker, _other(side_a))][0][0]
        now_b = 1.0 - bids[(market_b.ticker, _other(side_b))][0][0]
        # Cross-module: the rise and its test, both shared with the backtest
        rise = pair_price_rise(then_a + then_b, now_a + now_b)
        prices = (f"price {_price_text(then_a + then_b)} then, {_price_text(now_a + now_b)} now "
                  f"({_cents_text(rise)}, minimum {settings.trim_min_rise * 100:.0f}c)")
        if not trim_rise_reached(rise, settings.trim_min_rise):
            not_trimming("its total price has not risen by the minimum since it last "
                         f"traded: {prices}")
            return None, None
        rise_text = prices + ", "

    # Cross-module: the chance the pair pays, as a new trade of it would be sized
    # (each ask is 1 minus the other side's best bid)
    win_prob = held_pair_win_prob(pair_type, 1.0 - no_bid_a, 1.0 - yes_bid_a,
                                  1.0 - no_bid_b, 1.0 - yes_bid_b, settings.interval_discount)
    if win_prob is None:
        not_trimming("its chance of paying cannot be read from its quotes (a market's YES "
                     "ask sits below its own YES bid, or the two markets' midpoints leave "
                     "no gap between them)")
        return None, None

    def net_sale(pairs: int) -> float | None:
        """What selling `pairs` contract pairs returns after fees; None when the bids
        within reach hold fewer."""
        total = 0.0
        for ladder in ladders.values():
            # Cross-module: the one bid walk, then the taker fee on that market's sale
            walk = walk_bids(ladder, pairs)
            if walk is None:
                return None
            total += pairs * walk[0] - fee_leg_exact(pairs, walk[0])
        return total

    # Cross-module: the pair's size cap, as a new trade of its kind is capped
    cap = pair_size_cap(pair_type, settings.size_cap, settings.same_title_size_cap)
    # Cross-module: the backtest's trim rule
    decision = kelly_trim_count(count, value, win_prob, cap, top_value, net_sale)
    # Cross-module: the share Kelly lets the pair keep at its best bids, for the log line
    keep = min(cap, kelly_hold_fraction(win_prob, top_value))
    head = (f"Kelly trim check: {text} ({_KIND_WORDS.get(pair_type, pair_type)}), "
            f"{count_text(count)} each, {waited}, {rise_text}pays {win_prob:.1%}, "
            f"{_dollars(top_value)} a pair at the best bids, may keep {keep:.1%} of "
            f"{_dollars(value)}")
    sell = decision.sell
    if not sell:
        logging.info("%s -> keep", head)
        return None, None
    walked = {ticker: walk_bids(ladder, sell) for ticker, ladder in ladders.items()}
    proceeds = net_sale(sell)
    if proceeds is None or None in walked.values():
        # Defence only: the rule returns a count the bids can take
        not_trimming(f"the bids within reach hold fewer than the {sell} contract pairs a "
                     "trim would sell")
        return None, None
    logging.info("%s -> sell %d", head, sell)

    # Each leg sells `sell` of the `count` held; its cost is that share of the market's cost
    legs = tuple(
        SaleLeg(ticker, markets[ticker].event_ticker, side, sell,
                (reads.held_positions[ticker].exposure_dollars
                 + reads.held_positions[ticker].fees_dollars) * sell / count,
                market=markets[ticker], held=count)
        for ticker, side in pair.sides)
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
    Plan this run's Kelly trims: part sales of exact held pairs grown past their Kelly share.

    Each pair is judged by _trim_one. Fails closed: nothing is trimmed, with one
    log line saying why, unless all of these hold:

      - Kalshi's value of the open positions is known and above zero (a trim
        sized on too small a value sells too much).
      - Every held market's ladder is known.
      - With a wait in days or a minimum price rise set, the account's fills
        could be read, and with a wait, the list reaches back that many days.
        With neither set, the fills are not read.
      - The pairs whose best bids were read would not, alone or together, fetch
        more at those bids than all open positions are valued at. If they
        would, that value is wrong and no trim stands, even one logged "-> sell".

    A trim is sized on the cash plus the larger of Kalshi's value of the open
    positions and what the held markets would fetch at this run's bids, which
    can only trim less. Lone held markets are never trimmed.

    Args:
        client (Any): The authenticated production Kalshi client, only read from.
        held_positions (dict): Ticker -> HeldPosition for every held market.
        held_labels (dict): Ticker -> ladder labels (scanner.resolve_held_ladders' labels_out).
        markets_by_ticker (dict): This run's whole market list by ticker, held markets included.
        settings (LiveSettings): trim_to_kelly (off: trim and read nothing), trim_min_rise,
            k and both size caps.
        now (datetime): The run's moment, timezone-aware.
        cash_cents (float): The account's cash on every shard, in cents.
        positions_value_cents (float | None): Kalshi's value of the open positions, in
            cents; None when it was not read or was refused.
        skip (frozenset): Held tickers picked for a take-profit sale this run; a pair
            with one is not looked at.

    Returns:
        list[SalePlan]: The part sales in ticker order (kind config.SALE_KIND_KELLY_TRIM);
            empty when none.

    Raises:
        ValueError: Before any request, when now is not timezone-aware, cash_cents is
            not a finite number at least 0, or KELLY_TRIM_MIN_DAYS_SINCE_TRADE is
            invalid; later, when KELLY_TRIM_MAX_BID_DROP is invalid.
    """
    if not settings.trim_to_kelly:
        return []
    # Checked first, so a bad number stops the run before any request
    min_days = _trim_min_days()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if (isinstance(cash_cents, bool) or not isinstance(cash_cents, numbers.Real)
            or not math.isfinite(cash_cents) or cash_cents < 0):
        raise ValueError(f"cash_cents must be a finite number at least 0, got {cash_cents!r}")
    if not held_positions:
        _log_trim_summary(0, 0)
        return []
    # `not ... > 0` is also true for NaN
    if (isinstance(positions_value_cents, bool)
            or not isinstance(positions_value_cents, numbers.Real)
            or not positions_value_cents > 0 or not math.isfinite(positions_value_cents)):
        logging.warning(
            "Not trimming any held pair this run: Kalshi's value of the open positions is "
            "not known or is not above $0, and a trim sized on too small a value would "
            "sell too much")
        _log_trim_summary(0, 0)
        return []
    unknown = sorted((t for t in held_positions if not held_labels.get(t)), key=str)
    if unknown:
        # An unknown ladder could be any pair's, so no pair is provably alone on its ladder
        logging.info("Not trimming this run: the ladder of %d held market(s) is unknown (%s)",
                     len(unknown), ", ".join(str(t) for t in unknown))
        _log_trim_summary(0, 0)
        return []

    # Cross-module: the one definition of exact held pairs and lone held markets, quietly
    found = held_pairs(held_positions, held_labels, markets_by_ticker, log=False)
    # Exact pairs only, and none whose markets a take-profit sale already takes
    pairs = {key: pair for key, pair in found.items() if not pair.lone and not key & skip}
    for key in sorted(found, key=sorted):
        if not found[key].lone and key & skip:
            logging.info("Not trimming %s: it is picked for a take-profit sale this run",
                         _sides_text(found[key].sides))
    if not pairs:
        _log_trim_summary(0, 0)
        return []

    # Today on the bot's own clock, as every fill's day is read
    now_date = _day(now)
    fills = None
    if min_days > 0 or settings.trim_min_rise is not None:
        # Cross-module: one read of the account's fills serves the wait and the price rise
        fills = get_fill_times(client)
        if fills is None:
            logging.warning(
                "Not trimming any held pair this run: the account's fills could not be "
                "read, so when each pair last traded is not known")
            _log_trim_summary(0, len(pairs))
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
        elif not kelly_trim_rested(_day(fills.oldest), now_date, min_days):
            problem = (f"the account's fills list starts {_day(fills.oldest)} on the bot's "
                       f"own clock, fewer than {min_days} day(s) ago")
        if problem is not None:
            logging.warning(
                "Not trimming any held pair this run: %s, so a trade in the last %d day(s) "
                "cannot be ruled out", problem, min_days)
            _log_trim_summary(0, len(pairs))
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

    # 0 daily checks: a trim reads only the books now and, with a minimum price
    # rise, each market's price at its pair's last fill
    reads = _Reads(client, held_positions, markets_by_ticker, now.timestamp(), 0)
    utc_date = now.astimezone(UTC).date()
    plans: list[SalePlan] = []
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
    total = sum(fetched)
    if not doubts and total > positions_value:
        # Together they would fetch more than all open positions: the value is just as wrong
        doubts.append(f"the {len(fetched)} pairs whose order books were read would fetch "
                      f"{_dollars(total)} at their best bids together, more than the "
                      f"{_dollars(positions_value)} all open positions are valued at")
    if doubts:
        # Every trim above was sized on that wrong value, so none stands
        more = f" (and {len(doubts) - 1} more pair(s) alike)" if len(doubts) > 1 else ""
        logging.warning(
            "Not trimming any held pair this run, including any marked '-> sell' above: "
            "%s%s, so that value cannot be right, and a trim sized on too small a value "
            "would sell too much", doubts[0], more)
        _log_trim_summary(0, len(pairs))
        return []
    _log_trim_summary(len(plans), len(pairs))
    return plans
