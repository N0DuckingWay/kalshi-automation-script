"""
File: seller.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Decides which held positions the take-profit rule sells this run, and
    plans each sale. It sends no order: main.py hands the plans to
    trader.sell_positions.

    A position is either an exact held pair (scanner.held_pairs: two held
    markets alone on one ladder, one YES and one NO, of equal count, with
    readable costs) or a lone held market (alone on its ladder) with exactly
    one paid-out partner: a settled market the account no longer holds, that
    it held on the other side with the same count, asking the same question,
    and that paid out (settled yes or no) at most
    config.SALE_PARTNER_MAX_AGE_DAYS ago. Anything else is never sold, and a
    log line says why. (A ladder is one question asked at several deadlines:
    two markets are on one ladder when they share an event or ask the same
    question, scanner.ladder_keys.)

    The rule is the backtest's. Potential profit is what the position pays if
    it wins (its contract pairs at CONTRACT_PAYOUT_DOLLARS) less what it cost;
    realized profit is what selling its markets would return after the sale's
    fees (a partner that has paid out counts at its payout), less that cost.
    A position is sold when its realized profit reaches the share of its
    potential profit set by LiveSettings.sell_at at every one of
    config.TAKE_PROFIT_HOLD_DAYS daily checks, 24 hours apart, the last one
    now. The check now walks each held market's real order book (sells the
    count down its bids, best first); an earlier check reads each market's
    last candle bid in the 24 hours before it, in any size. The tests are config.take_profit_reached and
    config.reached_every_check, the ones the backtest decides with. With
    LiveSettings.sell_min_days, a position with fewer days left before its
    last held market stops trading is neither valued nor sold.

Dependencies:
    Imports the standard library, config, scanner and historical only, never
    the backtester, backtest, dashboard, depth model, trader, strategy or
    main (pinned by tests/test_seller.py). main.py alone imports it, and
    calls plan_sales before it buys anything.

Notes:
    Every request is a read-only GET, each made at most once per run. A
    position's steps run in order and stop at the first that fails, so a
    refused position reads nothing past that point. It fails closed: an
    unknown ladder, a missing book or candle, settlements that cannot all be
    read, or a partner that cannot be shown to be the only one each mean no
    sale, with a log line. Where live selling differs from the backtest is
    recorded in CLAUDE.md (live selling).
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

# One day in seconds: the gap between daily checks, and how far back an
# earlier check looks for a candle
_DAY_SECONDS = 86_400

# One candle period in seconds: the candles fetched for the earlier checks
# start this much before the oldest check's 24 hours, so none a check could
# read is cut off
_CANDLE_PERIOD_SECONDS = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60

# The side a lone held market's partner was held on: the other one
_OTHER_SIDE = {"yes": "no", "no": "yes"}


@dataclass(frozen=True)
class SaleLeg:
    """
    One market of a position the take-profit rule judges: held now, or paid out.

    Attributes:
        ticker (str): The market's ticker.
        event_ticker (str): Its event's ticker.
        side (str): The side the account holds there (or held, for a paid-out
            partner), "yes" or "no".
        count (int): Contracts held on it: the position's contract pairs.
        cost_dollars (float): What the account paid for them, fees included:
            a held market's exposure plus fees paid (HeldPosition), a
            paid-out partner's cost basis on the side held plus its fees
            (Settlement).
        market (Any): The ApiMarket from this run's market list while held;
            None for a paid-out partner.
        payout_dollars (float | None): What a paid-out partner paid when it
            settled (Settlement.revenue_dollars); None while held.
        paid_at (datetime | None): When a paid-out partner settled, in UTC;
            None while held.
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
        title (str): The first held market's display title
            (scanner.display_title).
        legs (tuple[SaleLeg, ...]): The position's markets: its held markets
            in ticker order, then a lone held market's paid-out partner.
        count (int): Its contract pairs: the contracts held on each market.
        cost_dollars (float): What the whole position cost, fees included
            (the sum of its legs' costs).
        ladders (dict): Held ticker -> its bids for the side held now, best
            first ([[price, quantity], ...], scanner.bid_ladder).
        walked (dict): Held ticker -> (average price, lowest price reached)
            selling the count down those bids (scanner.walk_bids).
        proceeds_dollars (float): What selling the held markets now returns
            after the sale's fees (config.fee_leg_exact).
        profits (tuple): (realized profit, potential profit) at each check,
            now first, then 1, 2, ... days before.
        days_left (int | None): Days from today (UTC) to the date its last
            held market stops trading (config.days_to_maturity); None when a
            close date is unknown.
        level (float): The share of potential profit it was sold at
            (LiveSettings.sell_at).
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

    Reads this module's TAKE_PROFIT_HOLD_DAYS when called, so a test patches
    seller.TAKE_PROFIT_HOLD_DAYS, never config's. It checks the value as
    backtester._resolve_hold_days does: change the two together.

    Returns:
        int: A whole number from 1 to TAKE_PROFIT_HOLD_DAYS_MAX.

    Raises:
        ValueError: For a bool, a number that is not whole, or one outside 1
            to TAKE_PROFIT_HOLD_DAYS_MAX.
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
        close_time (Any): ApiMarket.close_time, a timezone-aware datetime when
            the market list could parse it.

    Returns:
        date | None: Its UTC date; None for anything but an aware datetime,
            or one that cannot be placed in UTC.
    """
    if not isinstance(close_time, datetime) or close_time.utcoffset() is None:
        return None
    try:
        return close_time.astimezone(UTC).date()
    except (OverflowError, ValueError):
        # A time near the ends of datetime's range that UTC cannot hold
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
    Say why a held market is in no position the rule can sell.

    held_pairs leaves a held market out when its count or cost cannot be
    read, its cost is below the finest price times its count, or other held
    markets share its ladder without forming one exact pair with it.

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
    This run's reads for selling beyond the positions listing, each made at most once.

    Each is made only when first needed: a market's candles, the account's
    settlements, and a settled market's ladder labels (at most
    SALE_PARTNER_MAX_LOOKUPS exchange lookups per run).

    Attributes:
        client (Any): The authenticated Kalshi client.
        held_positions (dict): Ticker -> HeldPosition for every held market.
        markets_by_ticker (dict): This run's whole market list by ticker.
        now_ts (float): The run's moment, in Unix seconds.
        hold_days (int): How many daily checks there are.
        lookups (int): Exchange lookups of settled markets' labels made so far.
    """

    def __init__(self, client: Any, held_positions: dict, markets_by_ticker: dict,
                 now_ts: float, hold_days: int) -> None:
        """
        Start a run's reads with nothing read yet.

        Args:
            client (Any): The authenticated Kalshi client.
            held_positions (dict): Ticker -> HeldPosition.
            markets_by_ticker (dict): This run's whole market list by ticker.
            now_ts (float): The run's moment, in Unix seconds.
            hold_days (int): How many daily checks there are.
        """
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
            event_ticker (str): Its event's ticker (its series names the
                candlestick path).

        Returns:
            list | None: The candles; None when they could not be read.
        """
        if ticker not in self._candles:
            end = math.floor(self.now_ts)
            start = end - self.hold_days * _DAY_SECONDS - _CANDLE_PERIOD_SECONDS
            # Cross-module: hourly candles read fresh, never through the
            # backtest's candle cache
            self._candles[ticker] = recent_candles(self.client, ticker, event_ticker,
                                                   start, end)
        return self._candles[ticker]

    def settlements(self) -> tuple[list | None, str]:
        """
        The account's recent settlements, read once, when every record could be read.

        A record that could not be read may be the partner a lone held market
        needs, so then none is used and no lone held market is sold this run
        (fails closed); one WARNING says so.

        Returns:
            tuple[list | None, str]: (the settlements, "") or (None, why they
                cannot be used).
        """
        if not self._settlements_read:
            self._settlements_read = True
            unreadable: dict = {}
            # Ask only for the window a partner may come from, so an older
            # unreadable record cannot stop a sale; one second wider, so it is
            # never narrower than _find_partner's own age check
            oldest = math.floor(self.now_ts - SALE_PARTNER_MAX_AGE_DAYS * _DAY_SECONDS) - 1
            # Cross-module: the settlements in that window, counting the
            # records that could not be read
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

        A lookup from the exchange counts against SALE_PARTNER_MAX_LOOKUPS;
        past it, no further market is looked up this run.

        Args:
            ticker (str): The settled market's ticker.

        Returns:
            tuple[frozenset | None, str]: (its labels, "") or (None, why they
                could not be read).
        """
        if ticker in self._labels:
            return self._labels[ticker]
        listed = ticker in self.markets_by_ticker
        if not listed:
            if self.lookups >= SALE_PARTNER_MAX_LOOKUPS:
                return None, (f"the limit of {SALE_PARTNER_MAX_LOOKUPS} partner lookups "
                              "this run was reached")
            self.lookups += 1
        # Cross-module: from the run's list, else looked up the way a held
        # market the list lacks is, so its labels match its ladder-mates'
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

    The settlements do not say which markets were bought together, so the
    partner is the one settled market that fits. Both markets of every pair
    the bot buys ask one question (scanner.time_series_group_key), so the
    partner must carry the held market's question label; another option of
    the same multi-choice event does not. A candidate is a settled market the
    account no longer holds, held on the other side with the same count (and
    none on the held side), that paid out at most SALE_PARTNER_MAX_AGE_DAYS
    ago (one recorded as paying out after now passes here; _plan_one
    refuses it). As the held market is alone on its ladder, no other held
    market can claim the same partner.

    Fails closed: no partner, two or more, a candidate whose labels cannot be
    read, a held market with no question label, or a partner that settled
    other than yes or no each mean the held market is not sold.

    Args:
        reads (_Reads): This run's reads.
        ticker (str): The lone held market's ticker.
        side (str): The side held there, "yes" or "no".
        count (int): Its count.
        labels (frozenset): Its ladder labels.

    Returns:
        tuple[Settlement | None, str]: (the partner, "") or (None, why there
            is not exactly one usable partner).
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
            # Another question: never its pair-mate
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
        profits (list[tuple[float, float]]): (realized, potential profit) per
            check; potential profit above 0.

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
    Judge one position by the take-profit rule, logging its verdict, and plan its sale.

    Steps run in order and stop at the first that fails, logging one "Not
    selling" line (or "-> keep" for no potential profit or a check below the
    level): a whole count; every held market in the run's market list (for
    its close time and title); the days rule; each held market's book now,
    walked for the count; a lone held market's one paid-out partner, not
    recorded as paying out after now; a potential profit above 0; the check
    now; then each earlier check.

    Args:
        pair (HeldPair): An exact held pair or a lone held market (held_pairs).
        reads (_Reads): This run's reads.
        held_labels (dict): Ticker -> ladder labels for every held market.
        sell_at (float): The share of potential profit to sell at.
        min_days (int | None): The fewest days a position must have left
            before its last held market stops trading; None for no minimum.
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
    # Cross-module: the one definition of days to maturity, from each held
    # market's scheduled close date
    days_left = days_to_maturity([_utc_date(m.close_time) for m in markets.values()], now_date)
    if min_days is not None:
        # The days rule comes before any valuation, so a position too near
        # maturity costs no request
        if days_left is None:
            not_selling(f"a held market's close date is unknown, so the minimum of "
                        f"{min_days} days before maturity cannot be checked")
            return None
        if days_left < min_days:
            not_selling(f"{days_left} day(s) left before its last market stops trading, "
                        f"fewer than the minimum of {min_days}")
            return None

    # Check now, held markets: each one's real bids for the side held, walked for the count
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
            # A payout recorded after now (the exchange's clock ahead of this
            # machine's) has no book or candle rule to value it by (the
            # backtest would value that leg at its bid then)
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

    # The check now: each held market sold at its walk's average price, less
    # the taker fee on that sale (config.fee_leg_exact)
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
                # Paid out by this check: worth its payout, with nothing to sell
                value += leg.payout_dollars
                continue
            candles = reads.candles(leg.ticker, leg.event_ticker)
            if candles is None:
                not_selling(f"the recent candles of {leg.ticker} could not be read")
                return None
            # Cross-module: the side's bid on the last candle that ended in
            # the 24 hours before this check, read the backtest's way
            bid = bid_before(candles, moment, leg.side, window=_DAY_SECONDS)
            if bid != bid:
                not_selling(f"no {leg.side.upper()} bid on {leg.ticker} in the 24 hours "
                            f"before the check {back} day(s) before")
                return None
            # Cross-module: config.fee_leg_exact, the taker fee on selling
            # `count` at that one bid
            value += count * bid - fee_leg_exact(count, bid)
        profits.append((value - cost, potential))
        # Cross-module: config.take_profit_reached, the rule's test at this check
        if not take_profit_reached(sell_at, value - cost, potential):
            logging.info("%s -> keep (%s)", head, _checks_text(profits))
            return None

    # Cross-module: the rule's test at every check, the one the backtest's
    # _reached_every_day applies
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

    Logs a verdict line per position, a "Not selling" line per held market in
    no position, and a summary. When any held market's ladder is unknown, no
    position is judged (fails closed): one line names those markets, and the
    summary counts the other held markets as "not judged".

    Args:
        client (Any): The authenticated Kalshi client (production): its
            order books, candles, settlements and market lookups are read.
        held_positions (dict): Ticker -> HeldPosition for every held market
            (scanner.get_held_positions).
        held_labels (dict): Ticker -> ladder labels for every held market
            (scanner.resolve_held_ladders' labels_out). A held market with no
            labels leaves its ladder unknown, so nothing is sold.
        markets_by_ticker (dict): This run's whole market list by ticker,
            before held markets are dropped from it.
        settings (LiveSettings): Keyword-only. The run's settings:
            sell_at (None never sells, and nothing is read) and
            sell_min_days.
        now (datetime): Keyword-only. The run's moment, timezone-aware: the
            last check, from which the earlier ones are counted back.

    Returns:
        list[SalePlan]: The positions to sell, in ticker order; empty when
            none, or when settings.sell_at is None.

    Raises:
        ValueError: When now is not a timezone-aware datetime, or
            TAKE_PROFIT_HOLD_DAYS is not a whole number from 1 to
            TAKE_PROFIT_HOLD_DAYS_MAX.
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
        # A held market whose ladder is unknown could share one with any
        # position, so none can be shown to be alone on its ladder
        logging.info("Not selling this run: the ladder of %d held market(s) is unknown "
                     "(%s)", len(unknown), ", ".join(str(t) for t in unknown))
        never["ladder unknown"] = len(unknown)
        if len(held_positions) > len(unknown):
            never["not judged"] = len(held_positions) - len(unknown)
        _log_summary(0, 0, never)
        return []

    # Cross-module: the one definition of an exact held pair and a lone held
    # market, without its lines about adding to them
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
