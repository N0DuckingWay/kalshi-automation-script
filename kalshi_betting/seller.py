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

Dependencies:
    Imports the standard library, config, scanner and historical only, never
    trader, strategy, the backtester or main (pinned by tests/test_seller.py).
    main.py alone imports it, and calls plan_sales before it buys anything.

Notes:
    Every request is a read-only GET, made at most once per run. It fails
    closed: anything unknown or unreadable means no sale, with a log line.
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
    MIN_ACTIVE_PRICE_DOLLARS,
    SALE_PARTNER_MAX_AGE_DAYS,
    SALE_PARTNER_MAX_LOOKUPS,
    TAKE_PROFIT_HOLD_DAYS,
    TAKE_PROFIT_HOLD_DAYS_MAX,
    LiveSettings,
    count_text,
    days_to_maturity,
    fee_leg_exact,
    reached_every_check,
    take_profit_reached,
)
from .historical import bid_before, recent_candles
from .scanner import (
    HeldPair,
    HeldPosition,
    Settlement,
    _fetch_orderbook,
    bid_ladder,
    display_title,
    get_settlements,
    held_pairs,
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
        count (int): Contracts held: the position's contract pairs.
        cost_dollars (float): What they cost, fees included.
        market (Any): The ApiMarket from this run's market list while held; None for a partner.
        payout_dollars (float | None): What a partner paid when it settled; None while held.
        paid_at (datetime | None): When a partner settled (UTC); None while held.
    """
    ticker: str
    event_ticker: str
    side: str
    count: int
    cost_dollars: float
    market: Any = None
    payout_dollars: float | None = None
    paid_at: datetime | None = None


@dataclass(frozen=True)
class SalePlan:
    """
    A position the take-profit rule sells this run, with what its sale orders need.

    Attributes:
        title (str): The first held market's display title.
        legs (tuple[SaleLeg, ...]): Its held markets in ticker order, then its partner if any.
        count (int): Its contract pairs (the contracts held on each market).
        cost_dollars (float): What the whole position cost, fees included.
        ladders (dict): Held ticker -> bids for the side held, best first ([[price, qty], ...]).
        walked (dict): Held ticker -> (average, lowest price reached) selling the count down them.
        proceeds_dollars (float): What selling the held markets now returns after fees.
        profits (tuple): (realized, potential profit) at each check, now first.
        days_left (int | None): Days from today (UTC) until its last held market closes, if known.
        level (float): The share of potential profit it sold at (LiveSettings.sell_at).
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
    level: float


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
    This run's reads beyond the positions listing, each made once, when first needed.

    Attributes:
        client (Any): The authenticated Kalshi client.
        held_positions (dict): Ticker -> HeldPosition for every held market.
        markets_by_ticker (dict): This run's whole market list by ticker.
        now_ts (float): The run's moment, in Unix seconds.
        hold_days (int): How many daily checks there are.
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
