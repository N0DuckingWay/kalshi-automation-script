"""
File: trader.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Turns TradeSpec objects (from strategy.py) into Kalshi orders and sends
    them. Each pair is two orders ("legs") sent one after the other: the NO
    leg first, then the YES leg only if the NO leg filled. Which market gets
    which side depends on the pair type and comes only from _ordered_legs()
    (through scanner.leg_sides / scanner.leg_prices): for a same_title pair
    the NO leg is market_a (the pricier contract), for a time_series pair it
    is market_b (the later deadline — by close_time, or by stated deadline
    for a same-event ladder, DR-73). Never hardcode spec.pair.market_a as
    the NO leg.

    Both buy legs carry a price limit, so an order against a book that moved
    since the pre-execution check is killed instead of filling at a loss. If
    the YES leg does not fill, the NO leg is undone at once (the "rollback"
    or "unwind") by an order whose price is capped at the NO leg's entry less
    config.ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT, never an unpriced order. An
    unwind that does not close the whole position is reported as
    "rollback_failed", naming how many NO contracts are still open when the
    reply says. Pairs run in parallel threads (ThreadPoolExecutor) — except
    that until this process has confirmed or disproven the NO-leg side
    mapping they run one at a time (see execute_trades) — and every
    order and collateral-transfer POST first waits its turn on
    _ORDER_WRITE_PACER, a shared rate limiter (config.ORDER_WRITES_PER_SECOND,
    bursts of config.ORDER_WRITE_BURST), so the account stays under Kalshi's
    write limit.

    Every order goes to Kalshi's V2 create-order endpoint, POST
    config.V2_ORDER_PATH (/portfolio/events/orders), the bot's only order
    path; main.py and the human-run order-path probe refuse to start on any
    other config.ORDER_API_VERSION (config.order_api_version_error). Each
    order is a JSON body with a bid or ask side on the market's YES book, a
    dollar-string limit price, a count, the self_trade_prevention_type the
    endpoint requires, and the exchange_index (shard) of that leg's own
    market. The buy legs are fill_or_kill (the whole count fills at once or
    nothing does), and their limit price is their price protection: the
    scanned price rounded up onto the market's tick grid plus
    config.BUY_SLIPPAGE_TICKS ticks (_v2_limit_price). The unwind is
    reduce_only and immediate_or_cancel (it fills what it can at once and
    cancels the rest), the only time-in-force the endpoint accepts with
    reduce_only. Kalshi answers a fill_or_kill order it could not fill with
    an HTTP 409 error, which _submit_order_v2 returns as "canceled"
    (_is_fok_kill).

    An exception from a submission does not prove the order failed (a
    timeout can arrive after the fill), so the outcome is then judged from
    the account's position: by how it CHANGED, never by what it holds. Both
    tickers' positions are read before either order is sent and compared
    with a reading taken after the exception, so an unrelated holding on the
    same ticker is never mistaken for this order.

    The write pacer adds no wait before the YES leg: a pair's NO leg waits for
    two free tokens and holds the second, so the YES leg is sent at once. An
    unwind takes the pacer's hedge lane, ahead of every waiting NO leg, and
    waits at most 1/rate seconds for each unwind in that lane, its own
    included (_PairWrites).

    pre_execution_check() re-fetches order books for each spec in the portfolio
    concurrently and drops any whose prices have moved since the scan, reducing
    the chance of submitting orders against a stale price.

    ensure_shard_collateral() moves cash between shards before trading:
    sizing uses the whole balance, but each order is paid from its own
    market's shard. It totals what each shard needs, plans transfers from
    shards with spare cash, sends each transfer once to config.TRANSFER_PATH,
    and waits until the money has arrived; trades on a shard it could not
    fund are dropped. main._run_prod calls it right after pre_execution_check.

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py;
    ApiException (to recognise the V2 kill reply) from the kalshi_python_sync
    SDK; fetch_json_page (position reads), signed_request_json and
    api_call_with_retry (position reads only, never orders or transfers) from
    _http.py, with api_error_payload (reads the error details Kalshi sends
    back; _is_fok_kill checks their code) and api_error_summary (the one-line
    description every failed request is logged and recorded with here);
    ceil_to_tick, leg_prices, leg_sides, tick_size_for_price,
    v2_limit_price and validate_pair_price from scanner.py (leg_sides and
    leg_prices decide which market gets which side; ceil_to_tick and
    v2_limit_price are re-exported here as _ceil_to_tick and _v2_limit_price);
    read_shard_balances from auth.py (the balance re-read while waiting for a
    transfer); and ORDER_WRITES_PER_SECOND, ORDER_WRITE_BURST,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT, TRADER_MAX_WORKERS, TRANSFER_PATH,
    TRANSFER_POLL_INTERVAL_SECONDS, TRANSFER_SETTLE_TIMEOUT_SECONDS,
    V2_FOK_KILL_ERROR_CODE, V2_FOK_KILL_HTTP_STATUS,
    V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS, V2_MAPPING_VERDICT_POLL_SECONDS,
    V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS, V2_ORDER_PATH,
    V2_ROLLBACK_BID_PRICE_DOLLARS, V2_SELF_TRADE_PREVENTION_TYPE, LiveSettings,
    leg_cash_cents (the one rounding of a leg's cost up to the whole cent,
    shared with strategy.select_portfolio) and live_settings from config.py.
    Called by main.py after
    select_portfolio() picks the trades. Uses the KalshiClient built by
    auth.py.

Notes:
    Never retry an order or a collateral transfer: a resent order can fill
    twice or at a worse price, and a resent transfer moves the money twice.
    _submit_order_v2 and _execute_transfer call signed_request_json directly
    and must never be wrapped in api_call_with_retry. Each POST first takes
    one place on _ORDER_WRITE_PACER (a YES leg uses the place its NO leg held
    for it); that only delays the request, which is still sent once. An HTTP
    429 raises like any other error reply.

    Position reads (_position_count) ARE retried, since a read cannot trade;
    do not make reads and submissions alike in either direction. Two reads
    are single-shot on purpose (_position_count_once), because they run while
    a NO position may be open and unhedged and must not wait through retry
    backoff: _confirm_v2_no_mapping's check and _execute_one's NO-leg
    re-read. Both parse through _read_position.

    A position delta of ZERO across an ambiguous submission is not a confirmed
    non-fill on its own: a transport error can be raised after the exchange
    filled the order, and the ledger is read-after-write lagged. Every such
    branch re-reads once after _V2_MAPPING_RECHECK_DELAY_SECONDS and judges the
    re-read, so the bot no longer unwinds a live hedge (DR-63) or walks away
    from an unhedged fill reporting a clean non-fill (DR-64).

    Money units: contract prices are DOLLARS (float, or Decimal in the
    order-price math), balances and order costs are integer CENTS, and the
    transfer endpoint's `amount` is CENTICENTS (1/100 of a cent). Convert to
    centicents only through _cents_to_centicents, never inline.

    The V2 NO-leg mapping (an `ask` on the YES book opening a NO position) is
    doc-derived and cannot be proven offline, so the FIRST V2 NO-leg fill of
    each process is checked against the account's signed position DELTA across
    that fill (_confirm_v2_no_mapping): any movement other than -no_leg.count
    disproves the mapping and stops the pair at "manual_review" with the YES
    leg unsubmitted and the NO leg deliberately left in place. It is a delta,
    never an absolute sign — an unrelated holding in the same ticker would
    otherwise both fake a disproof and mask a real one. A process-lifetime
    latch (_V2_NO_MAPPING_CONFIRMED) keeps the cost at one extra positions
    read per run; the latch is shared across pair types, because it verifies
    the exchange's side mapping, not any particular market. A disproof sets a
    second process-lifetime latch (_V2_NO_MAPPING_DISPROVEN), and every later
    pair of the run is then stopped before anything is read or sent (status
    "failed"). Until one of the two latches is set, execute_trades runs pairs
    one at a time (for at most config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS),
    so a disproof costs one wrong-side position rather than one per pair
    already in flight.

    Every order body carries its own leg's market's exchange_index, never the
    -1 "auto-route" value, so each order goes to its own market's shard.

    Position reads use the SDK's raw-response method and parse the JSON
    themselves (_read_position), because the SDK's position model cannot read
    live replies. The SDK has no method for the V2 order route, so the
    _build_*_order_v2 functions build the JSON body and _submit_order_v2 sends
    it through _http.signed_request_json.

    All V2 price math is done in Decimal, never float: the endpoint takes
    dollar-string prices, and binary float noise would produce a string the
    exchange rejects as off-grid.
"""
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, Inexact, localcontext
from json import JSONDecodeError
from typing import Any

from kalshi_python_sync.exceptions import ApiException

from ._http import (
    api_call_with_retry,
    api_error_payload,
    api_error_summary,
    fetch_json_page,
    signed_request_json,
)
from .auth import read_shard_balances
from .config import (
    ORDER_WRITE_BURST,
    ORDER_WRITES_PER_SECOND,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT,
    TRADER_MAX_WORKERS,
    TRANSFER_PATH,
    TRANSFER_POLL_INTERVAL_SECONDS,
    TRANSFER_SETTLE_TIMEOUT_SECONDS,
    V2_FOK_KILL_ERROR_CODE,
    V2_FOK_KILL_HTTP_STATUS,
    V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS,
    V2_MAPPING_VERDICT_POLL_SECONDS,
    V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS,
    V2_ORDER_PATH,
    V2_ROLLBACK_BID_PRICE_DOLLARS,
    V2_SELF_TRADE_PREVENTION_TYPE,
    LiveSettings,
    leg_cash_cents,
    live_settings,
)
from .reporter import TradeResult
from .scanner import (
    ceil_to_tick,
    leg_prices,
    leg_sides,
    tick_size_for_price,
    v2_limit_price,
    validate_pair_price,
)
from .strategy import TradeSpec

# The V2 buy-cap arithmetic LIVES IN scanner.py and is only re-exported here.
# It moved there so strategy.py could size against the very cap this module
# submits, without importing trader (a real cycle — trader imports both
# scanner and strategy). Duplicating the formula was not an option: TS-08 is
# "the size and the cap disagree", and a second copy guarantees a second
# disagreement. Kept under the historical private names because every call
# site, docstring and test in this module already refers to them that way.
_ceil_to_tick = ceil_to_tick
_v2_limit_price = v2_limit_price

# Tolerance for comparing position deltas against whole-contract expectations.
# Contract counts are always whole numbers on the wire, but the API now sends
# them as the `position_fp` STRING, which _position_count parses with float() —
# so an exact `delta == -no_leg.count` comparison would be at the mercy of decimal
# round-tripping. Any real mismatch is at least one whole contract, many orders
# of magnitude above this epsilon.
_DELTA_EPS = 1e-6

# Side each order leg takes on Kalshi's SINGLE YES order book, where every
# order is quoted in YES terms:
#   buying YES        = bidding for YES at the YES price;
#   buying NO         = ASKING (selling YES you do not hold) at 1 - the NO
#                       price — a short YES position IS a long NO position;
#   closing a held NO = buying that YES short back, i.e. a reduce-only BID.
# Kept as one table so the mapping is a SINGLE point of correction: if the
# first live submission shows the exchange interprets a leg the other way
# round, only these three values change — no builder logic moves.
_V2_LEG_SIDE: dict[str, str] = {
    "buy_yes":  "bid",  # buy YES n @ p   -> bid at capped p
    "buy_no":   "ask",  # buy NO n @ p    -> ask at 1 - capped p
    "close_no": "bid",  # unwind held NO  -> reduce-only bid at an aggressive price
}


@dataclass(frozen=True)
class _Leg:
    """
    One submission leg of a pair: which market, which side, at what price.

    Built only by _ordered_legs(), the single place a pair type is translated
    into sides and prices. Every order builder, rollback-price helper and the
    execution state machine below takes a _Leg rather than a TradeSpec, so
    none of them can hardcode "market_a is the NO leg" — true for a same_title
    pair, false for a time_series pair.

    Attributes:
        market (Any): The market this leg trades — supplies the ticker, the
            tick grid (price_level_structure / price_ranges) and the
            exchange_index the order routes to.
        side (str): "no" or "yes" — the side BOUGHT on this market.
        price_dollars (float): The scanned, depth-weighted per-contract price
            of that side, in dollars (0, 1). This is the price the buy cap
            protects and, for the NO leg, the entry the rollback floor is
            measured from.
        count (int): Whole contracts on this leg — spec.x for the market_a
            leg, spec.y for the market_b leg (counts follow the MARKET).
        label (str): Human-readable "<SIDE> on <title or ticker>" for logs.
    """
    market: Any
    side: str
    price_dollars: float
    count: int
    label: str


def _ordered_legs(spec: TradeSpec) -> tuple[_Leg, _Leg]:
    """
    Resolve a spec's two legs into SUBMISSION order: (no_leg, yes_leg).

    The NO leg is always submitted first and is the leg the rollback unwinds;
    the YES leg follows only once the NO leg has filled. Which market carries
    which side comes from scanner.leg_sides (by pair type) and the matching
    prices from scanner.leg_prices, so this function — never a builder — is
    the only source of truth for the side/market assignment:

      * same_title:  NO on market_a (spec.x @ pair.nA), then YES on market_b
                     (spec.y @ pair.pB) — the wire behaviour this path has
                     always had, unchanged.
      * time_series: NO on market_b (spec.y @ pair.nB), then YES on market_a
                     (spec.x @ pair.pA).

    Counts follow the MARKET, not the side: spec.x is always market_a's
    contract count and spec.y market_b's (strategy.TradeSpec's invariant), so
    the market_a leg carries x and the market_b leg carries y whichever side
    each buys.

    Args:
        spec (TradeSpec): The trade specification. Reads spec.pair (pair_type,
            market_a, market_b and the quoted prices leg_prices selects from),
            spec.x and spec.y.

    Returns:
        tuple[_Leg, _Leg]: (no_leg, yes_leg). The two legs sit on different
            markets and buy opposite sides by construction — leg_sides returns
            exactly one "no" and one "yes" for every pair type.
    """
    # Cross-module: the pair type alone decides which market buys which side
    side_a, side_b = leg_sides(spec.pair.pair_type)
    # Cross-module: the per-contract prices of exactly those sides, in the
    # same (market_a, market_b) order
    price_a, price_b = leg_prices(spec.pair)
    market_a, market_b = spec.pair.market_a, spec.pair.market_b
    leg_a = _Leg(
        market=market_a, side=side_a, price_dollars=price_a, count=spec.x,
        label=f"{side_a.upper()} on {market_a.title or market_a.ticker}",
    )
    leg_b = _Leg(
        market=market_b, side=side_b, price_dollars=price_b, count=spec.y,
        label=f"{side_b.upper()} on {market_b.title or market_b.ticker}",
    )
    # NO leg first — the unwind side — whichever market it happens to be on
    return (leg_a, leg_b) if leg_a.side == "no" else (leg_b, leg_a)


# Number of decimal places in a V2 dollar-string price. Four places exactly
# represents every grid point of every known regime ($0.01 / $0.001 / $0.0001),
# so quantizing here can never move a price off-grid.
_V2_PRICE_QUANTUM = Decimal("0.0001")

# Process-lifetime latch for the V2 NO-leg mapping backstop in _execute_one().
# False until a V2 NO buy has been observed to move the account position by
# exactly -count (i.e. an `ask` really did open a NO position, as _V2_LEG_SIDE
# hypothesises). One confirmation proves the mapping for every later trade
# this run — so once confirmed the check is skipped and the backstop costs
# one extra positions read per PROCESS, not one per trade. Shared across pair
# types on purpose:
# it verifies the exchange's side mapping, not any market, so a same_title NO
# fill (on market_a) proves it for a time_series NO fill (on market_b) and
# vice versa. Deliberately not persisted anywhere: a fresh process
# re-verifies, which is cheap and keeps the check honest across restarts and
# API changes.
_V2_NO_MAPPING_CONFIRMED = False

# Process-lifetime latch set when the V2 NO-leg mapping is DISPROVEN, by the
# mapping check or by an ambiguous NO leg whose position moved in a way a NO
# buy cannot explain. From then on no pair in this process sends anything:
# _execute_one stops each later pair before it reads a position or builds an
# order (status "failed", nothing submitted), and a pair whose NO leg filled
# before the stop reached it stops at manual_review before its YES leg. The
# CRITICAL logged at the disproof tells the operator to stop trading and
# flatten by hand, so the bot must not keep opening positions on the same
# mapping for the rest of the run. Only a new process clears it; like the
# confirmation latch it is never persisted, so a scheduled run the next week
# starts clear.
# Each run the defaults server's Confirm and trade starts is a new process too,
# so both disproof CRITICALs (the mapping check's and the ambiguous NO leg's)
# name every way a real-money run starts: the scheduler daemon, main.py by
# hand, and that server's button.
_V2_NO_MAPPING_DISPROVEN = False

# NO-leg tickers of pairs that went ahead in this process although the mapping
# check could not read the account after their NO leg filled (the check's
# "unknown" outcome). Such a pair sends its YES leg as usual, so if a later
# pair then disproves the mapping, these positions rest on the same wrong
# mapping while the pairs report "executed". The disproof's CRITICAL names
# them so a human checks them too. Appended to from worker threads (a list
# append is atomic under the GIL) and cleared only with the process.
_V2_UNCHECKED_NO_LEGS: list[str] = []

# Pause before re-reading a ZERO position delta in _execute_one's two
# ambiguous-leg branches (the human-run V2 probe reads it too): a position
# that has not moved immediately after a fill the exchange may already have
# processed is most often read-after-write lag in the positions ledger, not
# evidence of a non-fill. The name is historical: the V2 NO-mapping backstop
# used this one pause too, and now re-reads on its own schedule,
# config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS. One second is about the lag
# observed live and far below any price-staleness concern.
_V2_MAPPING_RECHECK_DELAY_SECONDS = 1.0

# Pacing waits this short or shorter are not logged, so an ordinary queue
# behind a full burst does not bury the execution log.
_PACE_LOG_MIN_WAIT_SECONDS = 0.25

# Float slack on token comparisons: a waiter that wakes a hair short of a
# whole token is served rather than sent back to sleep.
_PACE_TOKEN_EPS = 1e-9


# eq=False so line.remove(request) removes this exact request, never another
# caller's request with the same fields.
@dataclass(eq=False)
class _PaceRequest:
    """
    One caller waiting on a _WritePacer.

    Attributes:
        need (int): Tokens to take: 1, or 2 for a write now plus one held.
        hold (bool): True when one of those tokens is held for later.
        hedge (bool): True for the hedge lane, served before the in-turn line.
        served_at (float | None): Clock reading when served; None while waiting.
    """
    need: int
    hold: bool
    hedge: bool
    served_at: float | None = None


class _HeldWrite:
    """
    A write set aside on a _WritePacer for its holder to send later.

    Its token counts against the bucket until the holder calls send() (the
    POST is going out) or release() (it is not). Only the first call counts.
    """

    def __init__(self, pacer: "_WritePacer") -> None:
        """
        Start a live hold (only _WritePacer._take makes one).

        Args:
            pacer (_WritePacer): The pacer the token is held on.
        """
        self._pacer = pacer
        self._live = True

    def send(self) -> bool:
        """
        Use the held write for a POST about to be sent.

        Returns:
            bool: True if it was still held, so the POST may go with no wait;
                False if it was already used or released, so the caller must
                take a place of its own.
        """
        return self._pacer._end_hold(self, sent=True)

    def release(self) -> None:
        """Give the held write back unsent, if it is still held."""
        self._pacer._end_hold(self, sent=False)


class _WritePacer:
    """
    A thread-safe token bucket that paces the account's order and transfer POSTs.

    Holds up to `burst` tokens and refills at `rate` a second; every POST takes
    a token first. Callers wait their turn, first come, first served, in one
    of two lines: the in-turn line (acquire, acquire_with_hold)
    and the hedge lane (acquire_hedge), which is always served first and is
    what an unwind of a pair's NO leg uses.

    acquire_with_hold takes two tokens: one for a POST now (a pair's NO leg)
    and one held for a POST later (its YES leg). Held tokens count against the
    bucket until sent or released, so the refill stops at `burst` minus the
    held tokens (the bucket's "room"). So in any T seconds at most
    burst + rate*T POSTs go out, however late a held write is sent. A hold
    needs two free tokens, so at most burst - 1 are ever held and a hedge-lane
    write can always be served by the refill alone.

    Waiting releases the lock (Condition.wait). Pacing only delays a request,
    which is still sent exactly once. Tests may pass their own clock and wait.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        *,
        clock: Callable[[], float] | None = None,
        wait: Callable[[float | None], Any] | None = None,
    ) -> None:
        """
        Build a full bucket.

        Args:
            rate (float): Tokens added per second; a finite number > 0.
            burst (int): Bucket size, i.e. writes allowed back to back; an int
                >= 2, so a pair's two writes fit together.
            clock (Callable[[], float] | None): Monotonic seconds; None uses
                time.monotonic.
            wait (Callable[[float | None], Any] | None): Called with the
                condition held; blocks up to the given seconds (None: until
                woken) and releases the condition meanwhile, as
                Condition.wait does. None uses the condition's own wait.

        Raises:
            ValueError: If rate or burst is out of range (a configuration
                error, so it is raised).
        """
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"write pacer rate must be a finite number > 0, got {rate!r}")
        if isinstance(burst, bool) or not isinstance(burst, int) or burst < 2:
            raise ValueError(
                f"write pacer burst must be an int >= 2 (a pair's two writes"
                f" must fit in the bucket together), got {burst!r}"
            )
        self._rate = float(rate)
        self._burst = float(burst)
        self._tokens = float(burst)  # free tokens, never above burst - held
        self._held = 0               # tokens held by acquire_with_hold
        # Last refill time; None until the first request, so it comes from the
        # pacer's own clock rather than a reading taken at import
        self._stamp: float | None = None
        self._clock = clock
        self._wait = wait
        self._cond = threading.Condition()
        # Waiting callers, first come, first served; hedges go first
        self._hedges: deque[_PaceRequest] = deque()
        self._in_turn: deque[_PaceRequest] = deque()

    def acquire(self) -> float:
        """
        Take a token for a POST now, waiting its turn behind earlier callers.

        Returns:
            float: Seconds waited (0.0 when a token was free).
        """
        return self._take(_PaceRequest(need=1, hold=False, hedge=False))[0]

    def acquire_hedge(self) -> float:
        """
        Take a token for a POST now, ahead of every caller waiting in turn.

        Returns:
            float: Seconds waited (0.0 when a token was free).
        """
        return self._take(_PaceRequest(need=1, hold=False, hedge=True))[0]

    def acquire_with_hold(self) -> tuple[float, _HeldWrite]:
        """
        Take a token for a POST now and hold a second for later, waiting in
        turn until two are free.

        Returns:
            tuple[float, _HeldWrite]: Seconds waited, and the held write (end
                it with send() or release()).
        """
        # A request with hold=True always comes back with its held write
        return self._take(_PaceRequest(need=2, hold=True, hedge=False))  # type: ignore[return-value]

    def _now(self) -> float:
        """
        Read the pacer's clock.

        Returns:
            float: Seconds on the given clock, or on time.monotonic.
        """
        return self._clock() if self._clock is not None else time.monotonic()

    def _refill(self, now: float) -> None:
        """
        Add the tokens earned since the last update, up to burst minus held.
        Call with the lock held.

        Args:
            now (float): The pacer's clock reading.
        """
        if self._stamp is None:
            self._stamp = now
        elif now > self._stamp:
            self._tokens = min(
                self._burst - self._held,
                self._tokens + (now - self._stamp) * self._rate,
            )
            self._stamp = now

    def _head(self) -> _PaceRequest | None:
        """
        Find the next caller to serve (hedge lane first). Call with the lock held.

        Returns:
            _PaceRequest | None: Its request, or None when nobody is waiting.
        """
        if self._hedges:
            return self._hedges[0]
        if self._in_turn:
            return self._in_turn[0]
        return None

    def _serve(self, now: float) -> None:
        """
        Serve callers from the front of the line while tokens cover them, and
        wake the rest if anyone was served. Call with the lock held.

        Stops at the first caller not covered, so nobody behind it goes first.

        Args:
            now (float): The pacer's clock reading.
        """
        self._refill(now)
        served = False
        while (head := self._head()) is not None and (
            self._tokens + _PACE_TOKEN_EPS >= head.need
        ):
            (self._hedges if head.hedge else self._in_turn).popleft()
            self._tokens -= head.need
            if head.hold:
                self._held += 1
            head.served_at = now
            served = True
        if served:
            self._cond.notify_all()

    def _delay(self) -> float | None:
        """
        Work out how long the front caller must wait. Call with the lock held,
        right after _serve.

        Returns:
            float | None: Seconds until the refill covers it; None when nobody
                is waiting, or when only ending a hold can make room for it.
        """
        head = self._head()
        if head is None or self._burst - self._held + _PACE_TOKEN_EPS < head.need:
            return None
        return max(0.0, (head.need - self._tokens) / self._rate)

    def _block(self, timeout: float | None) -> None:
        """
        Wait, with the lock released, until woken or until timeout.

        Args:
            timeout (float | None): Most seconds to wait; None waits until woken.
        """
        if self._wait is not None:
            self._wait(timeout)
        else:
            self._cond.wait(timeout)

    def _take(self, request: _PaceRequest) -> tuple[float, _HeldWrite | None]:
        """
        Join the request's line and wait until it is served.

        Any waiting thread that wakes serves the front of the line, so a
        slow-waking thread never holds the line up.

        Args:
            request (_PaceRequest): What the caller needs.

        Returns:
            tuple[float, _HeldWrite | None]: Seconds waited, and the held write
                when request.hold is True (else None).
        """
        line = self._hedges if request.hedge else self._in_turn
        with self._cond:
            arrived = self._now()
            line.append(request)
            try:
                self._serve(arrived)
                while request.served_at is None:
                    self._block(self._delay())
                    self._serve(self._now())
            except BaseException:
                if request.served_at is None:
                    # Leave the line, so nobody waits behind an empty place
                    line.remove(request)
                    self._cond.notify_all()
                elif request.hold:
                    # Served just before stopping: return the held token,
                    # which nobody else could ever send or release
                    self._refill(self._now())
                    self._held -= 1
                    self._tokens += 1.0
                    self._cond.notify_all()
                raise
            waited = request.served_at - arrived
        held = _HeldWrite(self) if request.hold else None
        if waited > _PACE_LOG_MIN_WAIT_SECONDS:
            logging.info(
                "Paced an order or transfer write to the exchange's write limit:"
                " waited %.2fs"
                " (at most %g writes a second, %d back to back)",
                waited, self._rate, int(self._burst),
            )
        return waited, held

    def _end_hold(self, held: _HeldWrite, *, sent: bool) -> bool:
        """
        End a hold. A sent hold's token is spent; a released one's returns to
        the free tokens. Either way the room grows, so waiters are woken.

        Args:
            held (_HeldWrite): The hold to end.
            sent (bool): True when the holder is about to send the write.

        Returns:
            bool: True if the hold was live and is now ended; False if it had
                already ended (nothing changes).
        """
        with self._cond:
            if not held._live:
                return False
            held._live = False
            # Refill to now under the current room first; freeing the room
            # before refilling would credit idle time to the whole bucket
            now = self._now()
            self._refill(now)
            self._held -= 1
            if not sent:
                self._tokens += 1.0
            self._serve(now)
            self._cond.notify_all()
        return True


# The one pacer shared by every order and transfer POST in this module, from
# every worker thread (see config.ORDER_WRITES_PER_SECOND). Built at import,
# so a bad config value fails the import.
_ORDER_WRITE_PACER = _WritePacer(ORDER_WRITES_PER_SECOND, ORDER_WRITE_BURST)


class _PairWrites:
    """
    One pair's places on the write pacer (used by _execute_one).

    opening() is the NO leg's place: it waits for two free tokens and holds
    the second for the YES leg, so the YES leg never waits while the NO leg is
    unhedged. hedge() is for the YES leg or an unwind: it uses the held place
    if still unsent, otherwise the hedge lane. close() returns an unsent held
    place; _execute_legs calls it as soon as the NO leg's POST raises, since
    checking whether that leg filled can take a minute of retried reads and
    the hold would block other pairs meanwhile.
    """

    def __init__(self, pacer: _WritePacer) -> None:
        """
        Start with no place taken.

        Args:
            pacer (_WritePacer): The pacer every write of this pair uses.
        """
        self._pacer = pacer
        self._held: _HeldWrite | None = None

    def opening(self) -> float:
        """
        Wait for the NO leg's place and hold one for the YES leg.

        Returns:
            float: Seconds waited.
        """
        # At most one hold per pair
        if self._held is not None:
            self._held.release()
        wait, self._held = self._pacer.acquire_with_hold()
        return wait

    def hedge(self) -> float:
        """
        Take a hedge write's place: the held one if unsent, else the hedge lane.

        Returns:
            float: Seconds waited (0.0 on the held place).
        """
        if self._held is not None and self._held.send():
            return 0.0
        return self._pacer.acquire_hedge()

    def close(self) -> None:
        """Return an unsent held place to the pacer."""
        if self._held is not None:
            self._held.release()


def _rollback_floor_cents(no_leg: _Leg) -> int:
    """
    Lowest NO price per contract, in cents, the unwind of a NO leg may take.

    It is the NO leg's scanned entry price less
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT, kept within 1..99 cents.
    _v2_rollback_price turns it into the unwind bid's cap of 1 - floor/100 on
    the YES book, so a collapsed book stops the unwind instead of taking an
    unbounded loss; _rollback_no_leg reports whatever is left open as
    "rollback_failed".

    Args:
        no_leg (_Leg): The NO leg being unwound (see _ordered_legs). Uses
            no_leg.price_dollars, its scanned entry price in dollars (0, 1).

    Returns:
        int: The floor in cents, always within [1, 99].
    """
    # round() BEFORE int: this bound is deliberately cent-quantized regardless
    # of the market's own tick grid, and no_leg.price_dollars (pair.nA for a
    # same_title pair, pair.nB for a time_series pair) is generally NOT a clean
    # cent value by the time it reaches here — scanner.enrich_with_orderbook_prices
    # replaces the raw best-ask with a depth-weighted average across qualifying
    # book levels, so it can land anywhere in (0, 1) as a float (e.g. 0.57
    # stored as 0.5699999999999998), which int() alone would truncate to 56.
    entry_cents = int(round(no_leg.price_dollars * 100))
    return max(1, min(99, entry_cents - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT))


def _v2_top_of_grid_price(market: Any) -> Decimal:
    """
    Highest tradeable bid level on one market's own tick grid, in dollars.

    config.V2_ROLLBACK_BID_PRICE_DOLLARS is the finest-grid target ($0.0001
    ticks); this floors it onto the grid of the band that contains it — floor,
    not ceiling, because rounding up would leave the open unit interval (1 is a
    settlement value, not a tradeable level). The result is 0.99 on a
    linear-cent grid, 0.999 on deci-cent, 0.9999 on a centi-cent edge band.

    Split out of _v2_rollback_price so the top-of-grid level has ONE definition:
    the rollback uses it as the upper clamp on its loss-floored bid, and the
    human-run V2 probe's unfillable-ask step submits at exactly this level.
    (Named without the module reference on purpose — a pipeline module must
    stay textually as well as structurally free of that tool; a test asserts
    it.) A flat 0.99 would fail to cross asks resting in (0.99, 1) on sub-cent
    regimes.

    Args:
        market (Any): The market whose grid to use. Any object exposing
            price_level_structure / price_ranges.

    Returns:
        Decimal: The highest valid price level on this market's grid.
    """
    target = Decimal(V2_ROLLBACK_BID_PRICE_DOLLARS)
    tick = tick_size_for_price(market, float(target))
    return (target / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def _v2_rollback_price(no_leg: _Leg) -> Decimal:
    """
    Price of the bid that unwinds a filled NO leg, in dollars.

    Holding NO is the same as being short YES, so the unwind buys YES back.
    Its price is capped at 1 - _rollback_floor_cents(no_leg)/100: paying more
    for the YES would lose more than ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT per
    contract. The bid fills only what rests at or under the cap, so a
    collapsed book leaves part or all of the position open, which is reported
    as "rollback_failed".

    The cap is rounded UP onto the grid of the price band it falls in, so the
    bid is always a valid level; that can loosen the cap by less than one
    tick. The cap is a whole-cent price and every grid contains whole cents,
    so today the rounding changes nothing. It is then kept at or below
    _v2_top_of_grid_price(market), the market's highest valid level.

    Args:
        no_leg (_Leg): The NO leg being unwound. Uses no_leg.price_dollars
            (its scanned NO entry, dollars) and no_leg.market (the tick grid).

    Returns:
        Decimal: The bid price in dollars: a valid grid level, no higher than
            the capped price or the market's top level.
    """
    market = no_leg.market
    # The loss floor mirrored onto the YES book: closing the NO at floor cents
    # is a YES buy-back at (1 - floor) dollars
    cap = Decimal("1") - Decimal(_rollback_floor_cents(no_leg)) / Decimal("100")
    # Round up onto the grid of the band the cap sits in, so the bid is a
    # valid level (a no-op today: the cap is whole cents; see docstring)
    cap = _ceil_to_tick(cap, tick_size_for_price(market, float(cap)))
    # Top-of-grid clamp: the highest level this market can actually quote
    return min(cap, _v2_top_of_grid_price(market))


def _format_price(p: Decimal) -> str:
    """
    Serialize a dollar price as the fixed-point string the V2 endpoint expects.

    Always four decimal places (e.g. "0.5600"), which exactly represents every
    grid point of every known tick regime, so the quantization here can never
    move an on-grid price off-grid. Decimal in, string out — a float never
    touches the wire format.

    Args:
        p (Decimal): Price in dollars. Range: [0, 1].

    Returns:
        str: The price as a 4-decimal fixed-point string.
    """
    return str(p.quantize(_V2_PRICE_QUANTUM))


def _format_count(n: int) -> str:
    """
    Serialize a contract count as the fixed-point string the V2 endpoint expects.

    V2 counts are fixed-point strings (e.g. "10.00") because the endpoint also
    supports fractional contracts. The bot still sizes in whole contracts —
    strategy.compute_trade() returns integer counts — so this always emits a
    ".00" fraction. Fractional sizing is a deliberately deferred follow-up.

    Args:
        n (int): Whole contract count for one leg. >= 1.

    Returns:
        str: The count as a fixed-point string with two decimal places.
    """
    return f"{n}.00"


def _build_no_order_v2(leg: _Leg) -> dict:
    """
    Build the V2 request body for the first-submitted leg: buy NO on its market.

    On the single YES book, buying NO is expressed as an ASK (selling YES we do
    not hold — a short YES position is a long NO position), priced at
    1 - (capped NO price). The order is fill_or_kill so it executes in full or
    not at all, and the limit price is the price protection (see
    _v2_limit_price). This side/price conversion is the assumption most needing
    verification at the first live submission — correct it in _V2_LEG_SIDE.

    Args:
        leg (_Leg): The NO leg from _ordered_legs (market_a for a same_title
            pair, market_b for a time_series pair). Uses leg.market for the
            ticker, tick grid and exchange shard, leg.price_dollars for the
            price cap, and leg.count for the contract count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": leg.market.ticker,
        # Client-generated idempotency key — lets a human match a log line to
        # an order in the account when a submission outcome is ambiguous
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["buy_no"],
        "price": _format_price(_v2_limit_price("buy_no", leg.price_dollars, leg.market)),
        "count": _format_count(leg.count),
        # fill_or_kill: execute the full count immediately or cancel with no fill
        "time_in_force": "fill_or_kill",
        # Required by the V2 endpoint — see config.V2_SELF_TRADE_PREVENTION_TYPE
        "self_trade_prevention_type": V2_SELF_TRADE_PREVENTION_TYPE,
        # This leg's OWN market's shard, read from the market itself — legs of
        # one pair can live on different shards. Explicit, never the -1
        # auto-route sentinel: if our notion of a market's shard is ever wrong
        # we want the exchange to reject the order loudly rather than silently
        # settle it against a shard we never modelled. An explicit-shard write
        # also bills only that shard's rate-limit bucket, where auto-route bills
        # every nonzero shard's.
        "exchange_index": leg.market.exchange_index,
        # Opening exposure, not closing it
        "reduce_only": False,
        # We are deliberately takers — a post-only order would be rejected
        # rather than crossing the book
        "post_only": False,
    }


def _build_yes_order_v2(leg: _Leg) -> dict:
    """
    Build the V2 request body for the second-submitted leg: buy YES on its market.

    Buying YES is a BID on the YES book at the capped YES price — no complement
    is involved, unlike the NO leg. fill_or_kill with the limit price acting as
    the price protection (see _v2_limit_price). The side/price mapping is the
    assumption most needing verification at the first live submission — correct
    it in _V2_LEG_SIDE.

    Args:
        leg (_Leg): The YES leg from _ordered_legs (market_b for a same_title
            pair, market_a for a time_series pair). Uses leg.market for the
            ticker, tick grid and exchange shard, leg.price_dollars for the
            price cap, and leg.count for the contract count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": leg.market.ticker,
        # Client-generated idempotency key — see _build_no_order_v2
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["buy_yes"],
        "price": _format_price(_v2_limit_price("buy_yes", leg.price_dollars, leg.market)),
        "count": _format_count(leg.count),
        # fill_or_kill: execute the full count immediately or cancel with no fill
        "time_in_force": "fill_or_kill",
        # Required by the V2 endpoint — see config.V2_SELF_TRADE_PREVENTION_TYPE
        "self_trade_prevention_type": V2_SELF_TRADE_PREVENTION_TYPE,
        # The YES leg's own market's shard — a pair's two markets may sit on
        # different shards. Explicit, never -1 auto-route — see _build_no_order_v2
        "exchange_index": leg.market.exchange_index,
        "reduce_only": False,
        "post_only": False,
    }


def _build_rollback_order_v2(no_leg: _Leg) -> dict:
    """
    Build the V2 order body that unwinds (closes) a filled NO leg.

    Holding NO is being short YES, so the unwind is a YES bid ("close_no" in
    _V2_LEG_SIDE). reduce_only means it can only shrink an existing position,
    so it is safe even when it is unclear whether the NO leg filled. Its
    price is the loss-capped bid from _v2_rollback_price. It is
    immediate_or_cancel (fills what it can at once and cancels the rest), the
    only time-in-force the endpoint accepts with reduce_only, so it may close
    only part of the position; _rollback_no_leg reports anything short of a
    full close as rollback_failed.

    Args:
        no_leg (_Leg): The NO leg to unwind (market_a for a same_title pair,
            market_b for a time_series pair). Supplies the price, ticker,
            shard and count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": no_leg.market.ticker,
        # Client-generated idempotency key — see _build_no_order_v2
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["close_no"],
        # Loss-floored bid cap on THIS market's grid (see _v2_rollback_price)
        "price": _format_price(_v2_rollback_price(no_leg)),
        "count": _format_count(no_leg.count),
        # immediate_or_cancel, not fill_or_kill: the exchange rejects a
        # reduce_only order with any other time in force. It fills what rests
        # at or under the loss-floored cap and cancels the rest (nothing is
        # left resting), so it can close only part of the position;
        # _rollback_no_leg reports anything short of a full close as
        # rollback_failed.
        "time_in_force": "immediate_or_cancel",
        # Required by the V2 endpoint — see config.V2_SELF_TRADE_PREVENTION_TYPE
        "self_trade_prevention_type": V2_SELF_TRADE_PREVENTION_TYPE,
        # The NO leg's own market's shard — the unwind must route to the same
        # shard the NO-leg order opened the position on. Explicit, never -1
        # auto-route — see _build_no_order_v2
        "exchange_index": no_leg.market.exchange_index,
        # Can only reduce an existing position — never opens exposure even if
        # the NO leg turns out not to have filled after all
        "reduce_only": True,
        "post_only": False,
    }


def _parse_fixed_point(payload: dict, key: str) -> Decimal | None:
    """
    Read a count field that may arrive as a fixed-point string or a raw number.

    The V2 responses observed in the docs carry counts both ways — `fill_count`
    as an integer and `fill_count_fp` as a fixed-point string (Get Orders shows
    the _fp form) — and which one a given deployment sends is not yet verified
    live. The _fp variant is preferred when present because it is the newer,
    non-truncating representation. Decimal(str(v)) keeps float noise out.

    Args:
        payload (dict): The order object from a V2 response.
        key (str): Base field name, e.g. "fill_count". The f"{key}_fp" variant
            is tried first.

    Returns:
        Decimal | None: The parsed value, or None when neither key is present or
            neither value can be parsed as a number.
    """
    for candidate in (f"{key}_fp", key):
        value = payload.get(candidate)
        if value is None:
            continue
        try:
            return Decimal(str(value))
        except (ArithmeticError, TypeError, ValueError):
            continue
    return None


def _v2_fill_status(data: dict, requested_count: int) -> str:
    """
    Read a V2 order reply as "executed" (all filled) or "canceled" (none).

    _execute_one and _rollback_no_leg branch on these two. A 2xx reply with a
    fill count of zero is "canceled"; so is Kalshi's HTTP 409 kill of a
    fill-or-kill order, which _submit_order_v2 converts before this runs.

    Anything else (a part fill, or no readable fill count) is logged at
    CRITICAL and raised, never guessed. On a buy leg the exception sends
    _execute_one to check the account's position. On the unwind, where a
    part fill is a real outcome (it is immediate-or-cancel), _rollback_no_leg
    reads the fill count from the reply _submit_order_v2 attaches to the
    error, reports rollback_failed and sends no further order. Never add a
    part-fill status here: this reader also classifies the buy legs, where a
    part fill must raise so the position change decides.

    Args:
        data (dict): The parsed V2 response body. The order object may be
            wrapped under an "order" key or sent flat; both are accepted.
        requested_count (int): Contract count the order asked for. >= 1.

    Returns:
        str: "executed" when the full count filled, "canceled" when nothing did.

    Raises:
        ValueError: When the fill count is missing, unparseable, or is a
            partial fill — deliberately routing a buy leg into the
            position-lookup disambiguation path, and an unwind into
            rollback_failed.
    """
    inner = data.get("order")
    order = inner if isinstance(inner, dict) else data
    fill = _parse_fixed_point(order, "fill_count")
    if fill is not None:
        if fill == requested_count:
            return "executed"
        if fill == 0:
            return "canceled"
    logging.critical(
        "V2 order response could not be classified (fill_count=%s, requested=%d,"
        " response keys=%s, order keys=%s) — raising: on a buy leg the account's"
        " position change decides the outcome; on the unwind the pair is"
        " reported rollback_failed.",
        fill, requested_count, sorted(data.keys()), sorted(order.keys()),
    )
    raise ValueError(
        f"Unclassifiable V2 order response: fill_count={fill}, requested={requested_count}"
    )


def _is_fok_kill(exc: BaseException) -> bool:
    """
    Tell whether a submission's exception is the V2 endpoint's fill-or-kill kill.

    The V2 create-order endpoint answers a fill_or_kill order that cannot fill
    in full with HTTP config.V2_FOK_KILL_HTTP_STATUS and a JSON body of the
    form {"error": {"code": config.V2_FOK_KILL_ERROR_CODE, "message": ...}}.
    The exchange rejects such an order before it matches, so nothing filled:
    it is a clean kill, not an ambiguous submission. Only that exact status
    AND code count — any other error response says nothing certain about
    whether the order filled, so it must keep reaching the caller's position
    check.

    Args:
        exc (BaseException): The exception a submission raised.

    Returns:
        bool: True only for an ApiException (the SDK raises its
            ConflictException subclass for a 409) whose status is the kill
            status and whose body — a str, or UTF-8 bytes — parses as a JSON
            object carrying the kill code under ["error"]["code"]. False for
            anything else: another status or code, a missing or unparseable
            body, a body that is not a JSON object, or any other exception
            type. Never raises.
    """
    if not isinstance(exc, ApiException):
        return False
    if getattr(exc, "status", None) != V2_FOK_KILL_HTTP_STATUS:
        return False
    # The error details Kalshi sent back
    error = api_error_payload(exc)
    return error is not None and error.get("code") == V2_FOK_KILL_ERROR_CODE


class _UnclassifiableV2Response(ValueError):
    """
    The error _submit_order_v2 raises when _v2_fill_status cannot classify a
    2xx V2 order response, with the parsed response body attached.

    It is a ValueError carrying _v2_fill_status's own message, so a caller
    that treats any exception as an ambiguous submission (the two buy legs in
    _execute_one) sees the same error text it would see from _v2_fill_status
    itself. The one caller that reads the body is _rollback_no_leg: the unwind
    is immediate_or_cancel, so a fill count between zero and the full count
    means it closed part of the position, and _rollback_no_leg reads that
    count from `response` to report how many NO contracts are still open.

    Attributes:
        response (Any): The parsed response body _v2_fill_status rejected. It
            is a dict, because _v2_fill_status reads it with .get before it
            can raise; _rollback_no_leg still checks the type before reading.
    """

    def __init__(self, message: str, response: Any = None) -> None:
        """
        Build the error from the classifier's message and the rejected body.

        Args:
            message (str): _v2_fill_status's own error message, kept word for
                word so the buy legs record the same error text.
            response (Any): The parsed response body. Defaults to None only so
                copy and pickle can rebuild the error from its message; they
                restore the body afterwards from the instance's attributes.
        """
        super().__init__(message)
        self.response = response


def _submit_order_v2(
    client: Any, body: dict, *, pace: Callable[[], float] | None = None,
) -> str:
    """
    Send one V2 order and return "executed" or "canceled".

    The SDK has no method for this route, so the request is signed and sent
    by _http.signed_request_json, which raises ApiException on any non-2xx
    reply. One error reply is returned instead: Kalshi's HTTP 409 kill of a
    fill_or_kill order (_is_fok_kill) means nothing filled, so it is
    "canceled". Any error on the immediate_or_cancel unwind raises.

    Never retried: a resent order can fill twice or at a worse price. It
    waits its turn on the write pacer right before the POST; an HTTP 429
    still raises like any other error.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        body (dict): Request body from one of the _build_*_order_v2 builders.
        pace (Callable[[], float] | None): Keyword-only. Called once, right
            before the POST, to wait for its place on the write pacer (e.g. a
            _PairWrites method from _execute_one). None takes a place in turn
            (_ORDER_WRITE_PACER.acquire).

    Returns:
        str: "executed" (full fill) or "canceled" (killed with no fill — a 2xx
            with a fill count of zero, or the exchange's kill response to a
            fill_or_kill body).

    Raises:
        ApiException: On any other error reply, and on every error reply to
            the immediate_or_cancel unwind.
        ValueError: A reply whose fill count is missing or partial, raised as
            _UnclassifiableV2Response with the reply attached (_execute_one
            then checks the position on a buy leg; _rollback_no_leg reads the
            count on the unwind). A 2xx reply that is not JSON raises
            json.JSONDecodeError, with nothing attached.
    """
    requested = int(Decimal(body["count"]))
    # One pacer place per POST, taken before logging so the log time is the send time
    (pace if pace is not None else _ORDER_WRITE_PACER.acquire)()
    # Log before submitting: the client_order_id is the only handle a human has
    # for finding this order in the account if the outcome turns out ambiguous
    logging.info(
        "Submitting V2 order: ticker=%s side=%s price=%s count=%s client_order_id=%s",
        body["ticker"], body["side"], body["price"], body["count"], body["client_order_id"],
    )
    try:
        # Retry-free by design (see docstring); signed_request_json contains no
        # retry logic of its own precisely so this call site stays single-shot
        data = signed_request_json(client, "POST", V2_ORDER_PATH, body=body)
    except ApiException as exc:
        # The exchange's kill of a fill-or-kill order is a clean non-fill, not
        # an ambiguous error. Only on a fill_or_kill body: every other error,
        # and any error on the immediate_or_cancel unwind, still raises.
        if body.get("time_in_force") == "fill_or_kill" and _is_fok_kill(exc):
            logging.info(
                "V2 order killed by the exchange (HTTP %d %s), nothing filled:"
                " ticker=%s side=%s price=%s count=%s client_order_id=%s",
                V2_FOK_KILL_HTTP_STATUS, V2_FOK_KILL_ERROR_CODE,
                body["ticker"], body["side"], body["price"], body["count"],
                body["client_order_id"],
            )
            return "canceled"
        raise
    try:
        return _v2_fill_status(data, requested)
    except ValueError as exc:
        # Re-raised with the same message, still a ValueError, carrying the
        # body: only the unwind's caller reads it, to count what a partial
        # close left open
        raise _UnclassifiableV2Response(str(exc), data) from exc


def _read_position(client: Any, ticker: str) -> float:
    """
    Fetch and parse the signed contract position for one ticker, once.

    The single place the positions wire format is decoded, so the retried
    (_position_count) and single-shot (_position_count_once) readers can never
    drift apart in how they interpret a page. This function performs exactly
    ONE HTTP GET and does not swallow anything — its callers decide what a
    failure means.

    Uses the raw-response variant + JSON parsing because the pinned SDK's
    MarketPosition model requires legacy integer fields the API stopped
    sending in 2026-07 (the count now arrives as the `position_fp` string) —
    the modeled get_positions call raises ValidationError on any non-empty
    page, which would turn every ambiguous order into manual_review.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up; also the server-side filter, so
            a single page is certain to contain it if a position exists.

    Returns:
        float: Signed contract count (0 when the account holds nothing in this
            ticker; may be fractional because position_fp is float-parsed).

    Raises:
        Exception: Any transport, HTTP (ApiException on non-2xx) or parse
            error. Callers translate this into "state unknown".
    """
    data = fetch_json_page(
        client.get_positions_without_preload_content, ticker=ticker
    )
    for pos in data.get("market_positions") or []:
        if pos.get("ticker") == ticker:
            # position_fp is the signed contract count the API now sends;
            # fall back to the legacy integer field if it ever reappears
            raw = pos.get("position_fp")
            if raw is None:
                raw = pos.get("position")
            return float(raw)
    return 0.0


def _position_count_once(client: Any, ticker: str) -> float | None:
    """
    Single-shot, deliberately UNRETRIED signed position read for one ticker.

    Same parse as _position_count (both go through _read_position) but with no
    backoff at all: exactly one request, then success or None. It exists for
    callers whose latency budget is bounded by something other than the read —
    the same reasoning _await_transfer_settlement uses for its single-shot
    balance reads. Both callers run inside the window where the NO leg may be
    filled and is certainly unhedged, because the YES leg has not been
    submitted yet: _confirm_v2_no_mapping's mapping check, and _execute_one's
    ledger-lag re-read on an ambiguous NO leg. api_call_with_retry can hold a
    single call for ~62s of sleeps during a 429 storm, and because a failing
    endpoint never latches the mapping, EVERY V2 trade in such a storm would
    pay that stall with a naked NO-leg position open. A failed read costs
    neither caller anything it cannot absorb: the mapping check proceeds
    unlatched and re-arms, and the re-read degrades to an unknown delta, which
    _execute_one already handles as manual_review with no order submitted.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up.

    Returns:
        float | None: Signed contract count, or None when the single attempt
            failed and the state is unknown.
    """
    try:
        return _read_position(client, ticker)
    except Exception as exc:
        logging.warning(
            "Single-shot position lookup failed for %s: %s",
            ticker, api_error_summary(exc),
        )
        return None


def _position_count(client: Any, ticker: str) -> float | None:
    """
    Fetch the signed contract position for one ticker, or None if the lookup fails.

    Used to disambiguate order-submission exceptions: an exception does not
    prove the order was rejected (a timeout can arrive after the fill), so the
    account's actual position is the ground truth. Kalshi convention: negative
    counts are NO contracts, positive counts are YES contracts.

    The returned number is the account's ABSOLUTE holding, which may include
    contracts this bot never bought (an earlier run, or a manual trade).
    Callers therefore never interpret a single reading: _execute_one snapshots
    before and after each submission and attributes only the DELTA (see
    _fill_delta). A single reading is meaningful only as a baseline.

    Parsing and the wire-format quirks live in _read_position (the pinned SDK's
    MarketPosition model can no longer deserialize live pages); this function
    adds the retry policy on top of it.

    The read goes through api_call_with_retry because it is a read-only GET:
    the project's no-retry rule covers order submission only, and retrying a
    GET cannot duplicate a trade. Without the retry a single transient 429 here
    reads as "state unknown" and escalates a recoverable ambiguity into a
    rollback or manual_review. The single-shot sibling _position_count_once is
    for the two callers that cannot afford the backoff (see its docstring).

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up.

    Returns:
        float | None: Signed contract count (0 = confirmed no position; may be
            fractional because position_fp is float-parsed), or None when the
            lookup itself failed on every attempt and the state is unknown.
    """
    try:
        # Retried: read-only GET, so the no-retry rule (order submission only)
        # does not apply, and a transient 429 must not be mistaken for
        # "position unknown" — see _execute_one's ambiguity handling.
        return api_call_with_retry(_read_position, client, ticker)
    except Exception as exc:
        logging.warning(
            "Position lookup failed for %s: %s", ticker, api_error_summary(exc)
        )
        return None


def _fill_delta(before: float | None, after: float | None) -> float | None:
    """
    Signed position change attributable to one order, or None if unknown.

    The account's absolute position is not evidence about a specific order —
    it may already hold contracts in the same ticker from an earlier run or a
    manual trade. The change across the submission is, provided both readings
    succeeded. Either reading being None (the lookup failed after retries)
    makes the delta unknowable, and the caller must treat that as unknown
    state rather than as zero.

    Args:
        before (float | None): Signed position immediately before submission,
            or None if that lookup failed.
        after (float | None): Signed position after the ambiguous submission,
            or None if that lookup failed.

    Returns:
        float | None: after - before, or None when either snapshot is missing.
            Deltas produced by real fills are whole contracts, but callers
            compare with _DELTA_EPS because position_fp is float-parsed.
    """
    if before is None or after is None:
        return None
    return after - before


def _partial_unwind_counts(exc: BaseException, count: int) -> tuple[str, str] | None:
    """
    Return how many NO contracts a partial unwind closed and how many are
    still open.

    The unwind is immediate-or-cancel, so it may close only part of the
    position. When it does, _submit_order_v2 raises _UnclassifiableV2Response
    carrying the order response, and this reads the fill count from it. Only
    _rollback_no_leg calls it.

    Args:
        exc (BaseException): The error the unwind raised.
        count (int): How many NO contracts the pair bought.

    Returns:
        tuple[str, str] | None: Contracts closed and contracts still open, as
            plain numbers ("3", "2.5"), or None when the error carries no
            usable fill count. Never raises.
    """
    try:
        if not (isinstance(exc, _UnclassifiableV2Response)
                and isinstance(exc.response, dict)):
            return None
        # The order may be nested under "order" or sent at the top level
        inner = exc.response.get("order")
        order = inner if isinstance(inner, dict) else exc.response
        fill = _parse_fixed_point(order, "fill_count")
        # Only a finite count above zero and below the full count is a
        # partial fill
        if fill is None or not fill.is_finite() or not 0 < fill < count:
            return None
        # Exact arithmetic only: a count too long to subtract exactly is
        # treated as unknown
        with localcontext() as ctx:
            ctx.traps[Inexact] = True
            still_open = count - fill
            return (
                format(fill.normalize(), "f"),
                format(still_open.normalize(), "f"),
            )
    except Exception:
        return None


def _rollback_no_leg(
    client: Any, spec: TradeSpec, no_leg: _Leg, reason: str, *,
    pace: Callable[[], float] | None = None,
) -> TradeResult:
    """
    Close a filled NO leg after the YES leg failed, and check that it closed.

    The NO leg (see _ordered_legs) is the only leg that can be left open when
    the YES leg fails. The unwind is the reduce-only immediate-or-cancel bid
    from _build_rollback_order_v2, price-capped by
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT, never an unpriced order. Being
    reduce-only, it cannot open a new position.

    Its own result is checked: a full close is "rolled_back"; no fill, a part
    fill or an error is "rollback_failed", and no second order is sent. For a
    part close the CRITICAL names how many NO contracts are still open (the
    NO count minus the fill count in the unwind's reply, read by
    _partial_unwind_counts); when that count cannot be read, or the unwind
    raised for any other reason, it says "up to" the NO count. Both of those
    alerts say to check the account. An unwind that filled nothing logs
    ROLLBACK NOT FILLED with the full NO count.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade being unwound; carried into the TradeResult.
        no_leg (_Leg): The first-submitted NO leg whose position must be
            closed — supplies the ticker, count, entry price and shard.
        reason (str): The upstream failure that triggered the rollback; recorded
            in the TradeResult error field.
        pace (Callable[[], float] | None): Keyword-only. Waits for the
            unwind's place on the write pacer (_execute_one passes
            _PairWrites.hedge). None takes the hedge lane
            (_ORDER_WRITE_PACER.acquire_hedge), ahead of writes waiting in
            turn.

    Returns:
        TradeResult: status="rolled_back" when the unwind filled in full;
            status="rollback_failed" when it did not fill, filled only part
            of the position, or raised (the error text then ends with how
            many NO contracts are still open, when known).
    """
    # A reduce-only bid capped at the ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT loss
    # floor, closing the NO-leg position
    rollback = _build_rollback_order_v2(no_leg)
    try:
        # Signed, single-shot submission (see _submit_order_v2), in the
        # pacer's hedge lane unless the caller hands in the pair's own place
        rb_status = _submit_order_v2(
            client, rollback,
            pace=pace if pace is not None else _ORDER_WRITE_PACER.acquire_hedge,
        )
    except Exception as exc:
        # One-line description of the error
        rb_error = api_error_summary(exc)
        # Contracts closed and still open after a partial unwind; None when
        # not known
        counts = _partial_unwind_counts(exc, no_leg.count)
        if counts is not None:
            closed_text, open_text = counts
            logging.critical(
                "ROLLBACK FAILED for '%s' — ORPHANED POSITION: the unwind's"
                " response reports it closed %s of the %d NO contracts this"
                " pair bought on %s, so %s of them are still open (an"
                " immediate-or-cancel order leaves nothing resting) — check the"
                " account. Manual review required. Error: %s",
                spec.pair.canonical_title, closed_text, no_leg.count,
                no_leg.market.ticker, open_text, rb_error,
            )
            return TradeResult(
                spec=spec, status="rollback_failed",
                error=(
                    f"{reason}; rollback error: {rb_error}; {open_text} of"
                    f" {no_leg.count} NO contracts still open"
                ),
            )
        logging.critical(
            "ROLLBACK FAILED for '%s' — ORPHANED POSITION: up to %d NO contracts"
            " on %s (the unwind can close part of the position before it stops —"
            " check the account). Manual review required. Error: %s",
            spec.pair.canonical_title, no_leg.count, no_leg.market.ticker, rb_error,
        )
        return TradeResult(
            spec=spec, status="rollback_failed",
            error=f"{reason}; rollback error: {rb_error}",
        )
    if rb_status != "executed":
        logging.critical(
            "ROLLBACK NOT FILLED (status=%s) for '%s' — ORPHANED POSITION: %d NO"
            " contracts on %s. Manual review required.",
            rb_status, spec.pair.canonical_title, no_leg.count, no_leg.market.ticker,
        )
        return TradeResult(
            spec=spec, status="rollback_failed",
            error=f"{reason}; rollback FoK not filled: status={rb_status}",
        )
    logging.warning(
        "Rollback executed for '%s' — sold %d NO contracts on %s",
        spec.pair.canonical_title, no_leg.count, no_leg.market.ticker,
    )
    return TradeResult(spec=spec, status="rolled_back", error=reason)


def pre_execution_check(client: Any, portfolio: list, *,
                        settings: LiveSettings | None = None) -> list:
    """
    Re-validate order book prices for all specs concurrently before execution.

    Fetches both order books for each spec in parallel and drops any whose gap
    threshold is no longer met or whose available depth is less than the intended
    contract count, or whose fresh book keeps no level with an edge after the
    fee, or (time-series) whose later book has no YES ask or whose spread tops
    the run's band ceiling (scanner.validate_pair_price). This reduces the
    window between price observation and order submission, lowering the chance
    of submitting against a stale price. Each drop is logged once, with its
    reason, by validate_pair_price (or by the exception handler here); this
    function adds only a summary count.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        portfolio (list): List of TradeSpec objects selected by select_portfolio().
        settings (LiveSettings | None): Keyword-only; the run's toggles. None
            reads config.py's (tests and direct calls only: a live run always
            hands the run's, built from the saved live defaults).

    Returns:
        list: Filtered list of TradeSpec objects that still pass the price check.
            May be empty if all pairs' prices moved since the scan.

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    if not portfolio:
        return []

    # One rule for every spec's re-check: the run's LiveSettings, or config.py's
    # for a caller that hands none
    settings = live_settings() if settings is None else settings

    valid = []
    with ThreadPoolExecutor(max_workers=min(TRADER_MAX_WORKERS, len(portfolio))) as pool:
        future_to_spec = {
            # Re-check each spec's books under the run's one LiveSettings
            pool.submit(validate_pair_price, client, spec, settings=settings): spec
            for spec in portfolio
        }
        # Iterate as futures complete so one raising thread does not swallow the others;
        # a raised exception is caught per-spec and the spec is dropped like a False check.
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                ok = future.result()
            except Exception as exc:
                logging.warning(
                    "Pre-execution check raised for '%s' — dropping: %s",
                    spec.pair.canonical_title, api_error_summary(exc),
                )
                continue
            if ok:
                valid.append(spec)
            # A False result was already logged, with its reason, by
            # validate_pair_price — don't log the same drop a second time.
    logging.info(
        "Pre-execution check: %d of %d pair(s) still qualify", len(valid), len(portfolio),
    )
    return valid


def _cents_to_centicents(cents: int) -> int:
    """
    Convert whole cents to CENTICENTS, the transfer endpoint's money unit.

    A centicent is 1/100 of a cent (so $1.00 = 100 cents = 10,000 centicents).
    This is the THIRD money unit in the codebase and it exists in exactly one
    place — the `amount` field of POST config.TRANSFER_PATH:

      * integer CENTS        — balances, MIN_BALANCE_CENTS
      * fixed-point DOLLARS  — the *_dollars API strings (auth._dollar_str_to_cents)
      * integer CENTICENTS   — intra-exchange transfer amounts (here, only here)

    The arithmetic is trivial on purpose; the function exists so the unit
    conversion is NAMED at the one call site that needs it. Never inline the
    factor: a missing ×100 moves 1% of the intended collateral (an unexplained
    insufficient-funds rejection later), and a doubled one moves 100× the
    intended amount out of a shard.

    Args:
        cents (int): Amount in whole cents. >= 0.

    Returns:
        int: The same amount expressed in centicents (cents × 100).
    """
    return cents * 100


def _required_cents_by_shard(portfolio: list) -> dict[int, int]:
    """
    Sum each exchange shard's cash requirement across a selected portfolio.

    Every leg draws its cash from the shard its own market lives on, so the
    market_a leg charges spec.pair.market_a.exchange_index (cost_with_fees_a)
    and the market_b leg charges spec.pair.market_b.exchange_index
    (cost_with_fees_b) — whichever side each buys. TradeSpec.cost_with_fees_a
    / cost_with_fees_b are MARKET costs, not side costs, which is what lets
    this pairing stay the same for both pair types. A pair whose legs share a
    shard simply adds both costs to that one shard.

    Args:
        portfolio (list): TradeSpec objects selected for execution. Each must
            carry per-leg cash requirements in cost_with_fees_a / cost_with_fees_b
            (dollars, fee-inclusive — see strategy.TradeSpec).

    Returns:
        dict[int, int]: exchange_index -> required cash in whole cents. Shards
            no leg touches are absent (not zero-filled).
    """
    required: dict[int, int] = {}
    for spec in portfolio:
        for market, cost_dollars in (
            (spec.pair.market_a, spec.cost_with_fees_a),
            (spec.pair.market_b, spec.cost_with_fees_b),
        ):
            # Rounded up to the cent, never down (an order a fraction of a cent
            # short of collateral is rejected), by the one rounding
            # strategy.select_portfolio budgets the cash with, so a portfolio
            # it admitted never finds its shards a cent short
            cents = leg_cash_cents(cost_dollars)
            required[market.exchange_index] = required.get(market.exchange_index, 0) + cents
    return required


def _unfunded_shards(required: dict[int, int], available: dict[int, int]) -> set[int]:
    """
    Return the shards whose available balance does not cover their requirement.

    Pure comparison — a shard missing from `available` counts as holding zero,
    which is the safe reading (we never assume money we could not observe).

    Args:
        required (dict[int, int]): exchange_index -> required cents.
        available (dict[int, int]): exchange_index -> observed balance in cents.

    Returns:
        set[int]: exchange_index values still short of cash. Empty when every
            requirement is covered.
    """
    return {shard for shard, need in required.items() if available.get(shard, 0) < need}


def _plan_transfers(
    required_by_shard: dict[int, int], available_by_shard: dict[int, int]
) -> list[tuple[int, int, int]]:
    """
    Plan the shard-to-shard transfers that would fund every shard's requirement.

    PURE function — no I/O, no logging, no clock. Deficit per shard is
    max(0, required - available); surplus is max(0, available - required). Each
    deficit is filled greedily from the largest REMAINING surplus first, which
    minimizes the number of transfers (every transfer is a non-idempotent POST,
    so fewer is strictly better).

    Ordering is fully deterministic so the same balances always produce the same
    plan (and so tests can assert on it): deficits are processed in ascending
    shard-index order, and candidate sources are ranked by remaining surplus
    descending, ties broken by ascending shard index.

    When total surplus is less than total deficit, this plans what IS coverable
    and leaves the rest short rather than failing outright — the caller detects
    the still-unfunded shard(s) from the post-transfer balances and drops only
    the trades that depend on them.

    Args:
        required_by_shard (dict[int, int]): exchange_index -> required cents.
        available_by_shard (dict[int, int]): exchange_index -> available cents.

    Returns:
        list[tuple[int, int, int]]: (source_shard, destination_shard, cents)
            transfers to perform, in execution order. Empty when no shard is
            short, and also when nothing can be moved (no surplus anywhere) —
            so an empty plan alone must NOT be read as "everything is funded".
    """
    deficits = sorted(
        (shard, need - available_by_shard.get(shard, 0))
        for shard, need in required_by_shard.items()
        if need - available_by_shard.get(shard, 0) > 0
    )
    remaining_surplus = {
        shard: avail - required_by_shard.get(shard, 0)
        for shard, avail in available_by_shard.items()
        if avail - required_by_shard.get(shard, 0) > 0
    }

    plan: list[tuple[int, int, int]] = []
    for dest, shortfall in deficits:
        # Re-rank per deficit so "largest remaining surplus first" stays true
        # after earlier deficits have drawn a source down.
        for source in sorted(remaining_surplus, key=lambda s: (-remaining_surplus[s], s)):
            if shortfall <= 0:
                break
            take = min(remaining_surplus[source], shortfall)
            if take <= 0:
                continue
            plan.append((source, dest, take))
            remaining_surplus[source] -= take
            shortfall -= take
    return plan


def _transfers_active(shard_statuses: dict | None, shard: int) -> bool:
    """
    Report whether the exchange says intra-shard transfers are usable on a shard.

    shard_statuses=None means the per-shard breakdown was unavailable (sandbox
    or pre-sharding shape, see scanner.fetch_shard_statuses) — there is nothing
    to gate on, so transfers are attempted and the POST itself is allowed to
    fail loudly if the endpoint is unsupported.

    When a breakdown IS available, a shard missing from it is treated as
    inactive: refusing to move money to or from a shard the exchange did not
    advertise costs us at most a dropped trade, while attempting it risks a
    transfer into a shard whose state we cannot reason about.

    Args:
        shard_statuses (dict | None): scanner.fetch_shard_statuses() output.
        shard (int): exchange_index to check.

    Returns:
        bool: True if a transfer touching this shard may be attempted.
    """
    if shard_statuses is None:
        return True
    return bool((shard_statuses.get(shard) or {}).get("intra_exchange_transfers_active"))


def _spec_shards(spec: TradeSpec) -> set[int]:
    """
    Return the set of exchange shards a single trade's two legs settle against.

    Args:
        spec (TradeSpec): The trade specification.

    Returns:
        set[int]: One element when both legs share a shard, two otherwise.
    """
    return {spec.pair.market_a.exchange_index, spec.pair.market_b.exchange_index}


def _partition_by_funding(portfolio: list, unfunded: set[int]) -> tuple[list, list]:
    """
    Split a portfolio into the trades that are fully funded and those that aren't.

    PURE function. A trade is droppable iff ANY of its legs sits on an unfunded
    shard — both legs must be payable, since a funded NO leg with an unpayable
    YES leg is precisely the unhedged half-fill the whole rollback machinery
    exists to avoid.

    Args:
        portfolio (list): TradeSpec objects selected for execution.
        unfunded (set[int]): exchange_index values still short of cash.

    Returns:
        tuple[list, list]: (kept, dropped), each preserving the input order.
    """
    kept: list = []
    dropped: list = []
    for spec in portfolio:
        (dropped if _spec_shards(spec) & unfunded else kept).append(spec)
    return kept, dropped


def _execute_transfer(client: Any, source: int, dest: int, cents: int) -> str | None:
    """
    Submit one intra-exchange collateral transfer and return its transfer id.

    Deliberately NOT wrapped in api_call_with_retry: the transfer endpoint is
    not idempotent, so retrying an ambiguous failure (a timeout that actually
    landed) would move the money a second time. One attempt, and a failure is
    reported to the caller as "this shard did not get funded".

    The SDK has no generated method for this route, so the request goes through
    _http.signed_request_json — a signed POST with a verbatim JSON body, which
    applies the shared non-2xx -> ApiException + JSON-parse contract and, by
    design, contains NO retry logic of its own. That is exactly the single-shot
    contract this call site needs (the same reason _submit_order_v2 uses it).
    The POST first takes a place in turn on _ORDER_WRITE_PACER.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        source (int): exchange_index the funds leave.
        dest (int): exchange_index the funds arrive on.
        cents (int): Amount to move, in whole CENTS (converted to the
            endpoint's centicents here — see _cents_to_centicents).

    Returns:
        str | None: The accepted transfer's id, or None when the response
            carried none (the transfer may still have been accepted, so the
            caller must treat this as "in flight", not "failed"). A 2xx whose
            body cannot be parsed (TS-17) — or which parses to something other
            than a JSON object, e.g. a bare string, array, number or null
            (DR-05) — also returns None, for the same reason: the status check
            ran first, so the exchange accepted the transfer and the money has
            moved. Both cases log a MONEY IS IN FLIGHT critical.

    Raises:
        ApiException: On a non-2xx status from the transfer endpoint.
        Exception: Any transport-level error — the transfer's fate is then
            unknown and it is NEVER re-sent.
    """
    # Wait for a place under the account's write limit; the transfer is still sent once
    _ORDER_WRITE_PACER.acquire()
    # Retry-free by design (see docstring): signed_request_json signs the path
    # verbatim, raises ApiException on non-2xx, and never retries.
    try:
        data = signed_request_json(client, "POST", TRANSFER_PATH, body=_transfer_body(source, dest, cents))
    except (JSONDecodeError, TypeError) as exc:
        # _check_and_parse validates the status BEFORE parsing, so a parse
        # failure proves the exchange returned 2xx: the transfer was ACCEPTED
        # and the funds have moved. Letting this propagate would land in
        # ensure_shard_collateral's generic handler, which logs a FAILED POST,
        # skips the settlement poll entirely and suppresses the MONEY IN FLIGHT
        # critical — the worst of both worlds. Returning None routes it into
        # the existing id-less-acceptance path instead. Still single-shot: the
        # request is NOT re-sent (TS-17).
        logging.critical(
            "Transfer POST of $%.2f shard %d→%d was ACCEPTED (2xx) but its response "
            "could not be parsed — MONEY IS IN FLIGHT, CHECK THE ACCOUNT. Treating as "
            "accepted with no transfer_id; NOT re-sent. Parse error: %s",
            cents / 100, source, dest, api_error_summary(exc),
        )
        return None
    if not isinstance(data, dict):
        # Same reasoning as the parse-failure branch above: the status check ran
        # first, so the exchange ACCEPTED the transfer — a body that parses to
        # anything but an object (a string, array, number, boolean or null) is
        # an unrecognised acknowledgement, not a rejection. Letting the
        # AttributeError from .get() propagate reported a FAILED POST, skipped
        # the settlement poll and suppressed this critical (DR-05) — the exact
        # misreport TS-17 fixed, one exception type over. Still single-shot:
        # the request is NOT re-sent.
        logging.critical(
            "Transfer POST of $%.2f shard %d→%d was ACCEPTED (2xx) but its response "
            "was not a JSON object (%s) — MONEY IS IN FLIGHT, CHECK THE ACCOUNT. "
            "Treating as accepted with no transfer_id; NOT re-sent.",
            cents / 100, source, dest, type(data).__name__,
        )
        return None
    return data.get("transfer_id")


def _transfer_body(source: int, dest: int, cents: int) -> dict:
    """
    Build the JSON body for one intra-exchange collateral transfer.

    PURE builder, split out of _execute_transfer so anything that needs to show
    the exact request (the live probe CLI's evidence log) can print the body that is
    actually sent rather than a hand-copied duplicate that could drift.

    Args:
        source (int): exchange_index the funds leave.
        dest (int): exchange_index the funds arrive on.
        cents (int): Amount to move, in whole CENTS.

    Returns:
        dict: The request body for POST config.TRANSFER_PATH.
    """
    return {
        # Both endpoints of an intra-exchange transfer are the trading balance;
        # "event_contract" is Kalshi's name for that collateral pool.
        "source": "event_contract",
        "destination": "event_contract",
        # CENTICENTS (1/100 cent) — NOT cents. See _cents_to_centicents.
        "amount": _cents_to_centicents(cents),
        "source_exchange_shard": source,
        "destination_exchange_shard": dest,
    }


def _await_transfer_settlement(client: Any, required: dict[int, int]) -> dict[int, int]:
    """
    Re-read the per-shard balance until every requirement is covered, or time out.

    Transfers are asynchronous — acceptance is not settlement — so this is the
    only thing that proves the collateral actually arrived. Polls every
    config.TRANSFER_POLL_INTERVAL_SECONDS and gives up after
    config.TRANSFER_SETTLE_TIMEOUT_SECONDS (a time.monotonic deadline, immune to
    wall-clock jumps), returning whatever the last read showed so the caller can
    decide which trades are still fundable.

    The balance read is auth.read_shard_balances — the SINGLE-SHOT variant of
    the shard-aware parse the run's opening balance came from, imported at
    module scope (trader.py already sits BELOW auth.py in the dependency order,
    so no cycle). Single-shot matters here: verify_auth wraps its read in
    api_call_with_retry, whose exponential backoff can hold one call for ~60s
    of sleeps during an outage — which would stretch this "bounded" wait far
    past its deadline, since the monotonic deadline is only checked between
    reads. A failed single read just costs one poll interval instead. Tests
    substitute the reader by monkeypatching trader.read_shard_balances.

    A balance read that raises is treated as "nothing observed" (empty dict) and
    retried until the deadline: it is never treated as success, because
    submitting orders against an unverifiable balance is exactly what this
    function exists to prevent.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        required (dict[int, int]): exchange_index -> required cents; the loop
            ends as soon as every one of these is covered.

    Returns:
        dict[int, int]: The most recent per-shard balance in cents (possibly
            empty if every read failed).
    """
    deadline = time.monotonic() + TRANSFER_SETTLE_TIMEOUT_SECONDS
    balances: dict[int, int] = {}
    while True:
        try:
            # Single-shot shard-aware read (same parse the run started with);
            # never the retry-wrapped verify_auth — see docstring for why.
            balances = read_shard_balances(client)
        except Exception as exc:
            logging.warning(
                "Balance re-read failed while awaiting transfers: %s",
                api_error_summary(exc),
            )
            balances = {}
        if not _unfunded_shards(required, balances):
            return balances
        if time.monotonic() >= deadline:
            return balances
        time.sleep(TRANSFER_POLL_INTERVAL_SECONDS)


def ensure_shard_collateral(
    client: Any,
    portfolio: list,
    shard_balances: dict[int, int],
    shard_statuses: dict | None,
    dry_run: bool = False,
) -> list:
    """
    Move collateral onto the exchange shards the selected portfolio draws from.

    Sizing is portfolio-wide — strategy.select_portfolio spends the SUM of every
    shard's cash — but an order settles against its own market's shard only. This
    function closes that gap: it totals each shard's cash requirement from the
    legs' cost_with_fees_* (_required_cents_by_shard), plans greedy transfers
    out of surplus shards (_plan_transfers), executes them, and confirms the
    asynchronous funds have actually landed before letting execution proceed.

    Failure is always degradation, never an abort: a blocked, failed, or
    unsettled transfer results in the affected trades being dropped from the
    returned portfolio while the rest execute normally. Specifically:

      * No shard is short  -> returns the portfolio unchanged, no API calls.
      * dry_run            -> logs the planned transfers and returns the
                              portfolio unchanged. NEVER POSTs.
      * Transfers inactive on either endpoint shard (per shard_statuses) ->
        that transfer is not attempted; a warning tells the operator to move
        the funds manually in the Kalshi UI.
      * A transfer POST raises -> logged as an error and NOT retried (the
        endpoint is not idempotent); its shard simply stays unfunded.
      * Transfers accepted but NOT OBSERVED to land within
        config.TRANSFER_SETTLE_TIMEOUT_SECONDS -> logged CRITICAL with the
        in-flight transfer ids ("money is in flight"), and only the trades
        needing a still-unfunded shard are dropped. The verdict is the
        settlement OBSERVATION, not the fact that a transfer was headed there:
        a shard whose accepted transfers were all observed to land and merely
        fell short of its deficit gets a WARNING naming the shortfall instead.
        That warning states only what was observed and asserts no cause — the
        shard may be short because a leg was blocked, because a POST raised,
        or because there was no surplus left, and this function cannot tell
        which.

    This is genuinely live, not a placeholder for a future migration: Kalshi
    moved all combo/MVE markets to shard 1, crypto to shard 2, and
    tennis/baseball to shard 3 (see the exchange-sharding gotcha in
    CLAUDE.md), so a selected portfolio spanning shards — and an account
    balance concentrated on one shard — is the normal case today, not an
    edge case. The zero-deficit fast path still returns immediately with no
    API calls whenever a run's selected legs happen to already sit on a
    funded shard; it is a fast path for that case, not evidence collateral
    movement is inactive in general.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        portfolio (list): TradeSpec objects selected for execution, each
            carrying cost_with_fees_a / cost_with_fees_b.
        shard_balances (dict[int, int]): exchange_index -> cash in cents, the
            shard_cash_cents of the auth.read_account_balance() read
            main._run_prod made before this run's orders.
        shard_statuses (dict | None): scanner.fetch_shard_statuses() output,
            used to skip shards whose intra_exchange_transfers_active is false.
            None (breakdown unavailable) means transfers are attempted anyway
            and the POST is allowed to fail loudly.
        dry_run (bool): When True, plan and log but never POST. Defaults to False.

    Returns:
        list: The subset of `portfolio` whose every leg sits on a shard
            confirmed to hold its required cash, in the input order. Equal to
            `portfolio` whenever nothing needed funding or every transfer
            settled; possibly empty when nothing could be funded.
    """
    if not portfolio:
        return []

    # Per-leg cash requirements totalled onto the shard each leg settles against
    required = _required_cents_by_shard(portfolio)
    # An empty plan is ambiguous (nothing needed vs. nothing movable), so the
    # fast path keys off the deficit set itself, not off the plan being empty.
    if not _unfunded_shards(required, shard_balances):
        logging.info(
            "All shards sufficiently funded for %d selected trade(s) — no collateral "
            "transfers needed (required by shard: %s)", len(portfolio), required,
        )
        return portfolio

    # Pure, deterministic greedy plan — no I/O happens until the loop below
    plan = _plan_transfers(required, shard_balances)

    if dry_run:
        for source, dest, cents in plan:
            logging.info(
                "DRY RUN: would transfer $%.2f shard %d→%d", cents / 100, source, dest,
            )
        if not plan:
            logging.info("DRY RUN: shard(s) %s under-funded and no surplus to draw on",
                         sorted(_unfunded_shards(required, shard_balances)))
        return portfolio

    accepted: list[str] = []
    # Cents actually accepted for movement INTO each destination shard — the
    # settle wait must target only these, not the full requirement map: a
    # deficit shard whose transfer was skipped (transfers inactive) or whose
    # POST failed can never settle, and waiting on it would burn the whole
    # timeout and then miscast transfers that DID settle as in-flight.
    accepted_cents: dict[int, int] = {}
    for source, dest, cents in plan:
        if not _transfers_active(shard_statuses, source) or not _transfers_active(
            shard_statuses, dest
        ):
            logging.warning(
                "Intra-exchange transfers are not active on shard %d and/or %d — NOT "
                "moving $%.2f; trades needing shard %d will be dropped. Move the funds "
                "manually in the Kalshi UI to enable them.",
                source, dest, cents / 100, dest,
            )
            continue
        try:
            # Single attempt by design — see _execute_transfer (non-idempotent).
            transfer_id = _execute_transfer(client, source, dest, cents)
        except Exception as exc:
            logging.error(
                "Collateral transfer of $%.2f from shard %d to shard %d FAILED (not "
                "retried — the endpoint is not idempotent): %s",
                cents / 100, source, dest, api_error_summary(exc),
            )
            continue
        accepted.append(str(transfer_id))
        accepted_cents[dest] = accepted_cents.get(dest, 0) + cents
        logging.info(
            "Collateral transfer accepted: $%.2f shard %d→%d (transfer_id=%s) — "
            "asynchronous, awaiting settlement",
            cents / 100, source, dest, transfer_id,
        )

    # What the accepted transfers can actually deliver per shard — the lesser
    # of its requirement and its prior balance plus the cents moved toward it —
    # so an unfundable shard can't stall the wait. This is also the yardstick
    # the in-flight verdict below is measured against, which is why it is
    # defined for both branches rather than only inside the wait.
    awaitable = {
        dest: min(required[dest], shard_balances.get(dest, 0) + moved)
        for dest, moved in accepted_cents.items()
        if dest in required
    }
    if accepted_cents:
        # Acceptance is not settlement: block (bounded) until a fresh balance
        # read proves the money landed before any order relies on it.
        confirmed = _await_transfer_settlement(client, awaitable)
    else:
        # Nothing moved, so the opening balances are still the truth — don't
        # burn the settle timeout waiting for transfers that were never sent.
        confirmed = shard_balances

    unfunded = _unfunded_shards(required, confirmed)
    if not unfunded:
        logging.info("All shard collateral requirements confirmed funded.")
        return portfolio

    # MONEY IS IN FLIGHT means "an accepted transfer was not OBSERVED to land",
    # and that is decided by the SETTLEMENT OBSERVATION, never by a shard's
    # membership in accepted_cents (DR-65). `awaitable` is exactly what the
    # accepted transfers could deliver, so a confirmed balance at or above it
    # proves the money arrived; such a shard reads short only because the plan
    # could not cover its whole deficit — an ordinary degraded outcome, not an
    # ambiguous one. Keying on membership alone fired the critical on a
    # transfer whose arrival the same log line's own `confirmed` map showed.
    #
    # Known residual: "landed" is inferred from a balance THRESHOLD against
    # this run's opening shard_balances, not from the transfer's own status.
    # An unrelated credit to the deficit shard of at least the accepted cents,
    # arriving inside the poll while the transfer is genuinely stuck, would
    # suppress this critical. The inverse (a debit) only over-reports, which
    # is the safe direction, and the settle wait's success condition already
    # carried the same aliasing.
    in_flight = sorted(
        s for s in unfunded if confirmed.get(s, 0) < awaitable.get(s, 0)
    )
    if in_flight:
        logging.critical(
            "Collateral transfer(s) did not settle within %ss — MONEY IS IN FLIGHT, "
            "CHECK THE ACCOUNT. transfer_ids=%s; shard(s) still under-funded: %s "
            "(required %s, confirmed %s)",
            TRANSFER_SETTLE_TIMEOUT_SECONDS, accepted, in_flight, required, confirmed,
        )
    # Accepted, OBSERVED to land, and still short. Report only that — the code
    # knows the shard is short but nothing about WHY: a planned leg may have
    # been blocked by inactive transfers, its POST may have raised (which the
    # server could still have accepted, leaving that shard out of
    # accepted_cents entirely), or there may simply have been no surplus left.
    # Asserting a cause here, or asserting the negative "nothing is in flight",
    # would be an affirmative denial the observation does not support — and in
    # the raised-POST case a false one. The shortfall is a normal degraded
    # outcome, so it is a warning rather than a money-in-flight alarm.
    settled_short = sorted(
        s for s in unfunded if s in accepted_cents and s not in set(in_flight)
    )
    if settled_short:
        logging.warning(
            "Collateral transfer(s) to shard(s) %s were OBSERVED to land but did "
            "not cover the full requirement. Shortfall in cents by shard: %s "
            "(required %s, confirmed %s). See any transfer warnings/errors above "
            "for what could not be moved. "
            "Trades needing these shards are dropped below.",
            settled_short,
            {s: required[s] - confirmed.get(s, 0) for s in settled_short},
            {s: required[s] for s in settled_short},
            {s: confirmed.get(s, 0) for s in settled_short},
        )

    # Both legs must be payable — a funded NO leg with an unpayable YES leg is
    # the unhedged half-fill the rollback machinery exists to avoid
    kept, dropped = _partition_by_funding(portfolio, unfunded)
    for spec in dropped:
        logging.warning(
            "Dropping '%s' — leg shard(s) %s include an under-funded shard %s",
            spec.pair.canonical_title, sorted(_spec_shards(spec)), sorted(unfunded),
        )
    logging.warning(
        "Collateral funding incomplete: %d of %d selected trade(s) dropped.",
        len(dropped), len(portfolio),
    )
    return kept


def _stop_run_on_v2_mapping_disproof() -> str:
    """
    Set the disproven latch, so no later pair of this process sends anything.

    Called wherever the V2 NO-leg side mapping is found disproven: by the
    mapping check (_confirm_v2_no_mapping) and by an ambiguous NO leg whose
    position moved in a way a NO buy cannot explain (_execute_legs).

    Returns:
        str: A sentence to end the disproof's CRITICAL with, naming the NO
            legs that earlier pairs of this process sent while the mapping
            check could not read the account (_V2_UNCHECKED_NO_LEGS). Those
            pairs went ahead as if the mapping held, so their positions rest
            on the same wrong mapping. An empty string when there are none.
    """
    global _V2_NO_MAPPING_DISPROVEN
    _V2_NO_MAPPING_DISPROVEN = True
    if not _V2_UNCHECKED_NO_LEGS:
        return ""
    return (
        " Earlier pairs of this run went ahead after their NO-leg fill could"
        " not be checked, so they rest on the same mapping: check the"
        f" positions on {', '.join(_V2_UNCHECKED_NO_LEGS)} too."
    )


def _confirm_v2_no_mapping(
    client: Any, spec: TradeSpec, no_leg: _Leg, before_no: float | None,
) -> TradeResult | None:
    """
    Check, once per process, that a filled V2 NO buy really opened a NO position.

    _V2_LEG_SIDE sends a NO buy as an `ask` on the YES book. This checks that
    against the account: the NO buy of no_leg.count contracts must change the
    signed position on its market by exactly -no_leg.count. It judges the
    CHANGE from the baseline read before the NO leg was sent, never the
    holding itself, because the account may already hold that market.

    _execute_one calls it after the NO leg filled and before the YES leg is
    sent. After one confirmation (_V2_NO_MAPPING_CONFIRMED) it does nothing
    for the rest of the process, for every market and pair type, unless the
    mapping has been disproven (below). Its reads (_position_count_once) are
    single-shot, never retried, because they happen while the NO position
    is unhedged.

    Outcomes:
      * change of -no_leg.count: confirmed; remember it for the rest of the
        run and go on to the YES leg.
      * change unknown (a read failed): warn and go on without remembering
        it, so the next NO fill checks again. The NO leg's ticker is
        recorded (_V2_UNCHECKED_NO_LEGS), so that if a later pair disproves
        the mapping its CRITICAL names this position too.
      * change of zero: read again after each pause in
        config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS (1, 2 and 4 s, so up
        to 7 s), since the ledger can lag behind a fill (about 1 s was seen
        live); the first reading that moved is judged, and a failed re-read
        is the unknown outcome above. A wrong mapping moves the position the
        wrong way rather than not at all, so the wait costs no extra
        wrong-side position while pairs run one at a time; it can leave this
        NO leg unhedged up to 7 s, and only when the ledger lags.
      * any other change, a zero that survives every re-read included:
        disproven. Set the disproven latch (_V2_NO_MAPPING_DISPROVEN), so
        _execute_one sends nothing for any later pair of this process, and
        return manual_review: the YES leg is not sent and the NO leg is not
        unwound, since the unwind relies on the same side mapping. The
        CRITICAL log says the rest of the run is stopped, and to stop
        trading and flatten the position by hand in the Kalshi UI.

    If the mapping was already disproven in this process when this runs —
    this pair's NO leg filled after another pair's disproof — nothing is
    read: the pair stops at manual_review with the YES leg unsent and the NO
    leg left in place, as the disproving pair did. The disproven latch is
    checked before the confirmed one, so a disproof always wins.

    execute_trades runs pairs one at a time until this check confirms or
    disproves the mapping, so while it does no two pairs reach the check
    together. After its time budget, or for a caller running _execute_one
    concurrently some other way, two pairs can: that only costs an extra
    read, and each pair still stops on its own evidence or on the latch.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade whose NO leg just filled; used only for
            the manual_review TradeResult.
        no_leg (_Leg): The NO leg from _ordered_legs: its market's ticker is
            read and no_leg.count is the expected change.
        before_no (float | None): The NO leg's market's position read before
            the NO leg was sent, or None if that read failed (then no change
            is knowable and the trade goes on without remembering a result).

    Returns:
        TradeResult | None: None to go on to the YES leg (confirmed, already
            confirmed, or not checkable); a status="manual_review"
            TradeResult when the mapping was disproven, by this pair or
            earlier in this process.
    """
    global _V2_NO_MAPPING_CONFIRMED
    ticker = no_leg.market.ticker
    if _V2_NO_MAPPING_DISPROVEN:
        # Another pair already disproved the mapping and this pair's NO leg
        # filled anyway (it was past _execute_one's stop when the latch was
        # set). Treat it like the disproving pair: its YES leg and any unwind
        # would rest on the same disproven mapping.
        logging.critical(
            "V2 NO leg on %s filled after the NO-leg mapping was disproven"
            " earlier in this run — NOT submitting the YES leg and NOT"
            " auto-unwinding. A human must flatten this account position too.",
            ticker,
        )
        return TradeResult(
            spec=spec, status="manual_review",
            error=(
                f"V2 NO-leg mapping disproven earlier in this run; NO leg on"
                f" {ticker} filled, YES leg not submitted and NO leg not unwound"
            ),
        )
    if _V2_NO_MAPPING_CONFIRMED:
        return None
    # Ground truth for the mapping: how the account's own signed position
    # MOVED across the fill. Single-shot on purpose — see the docstring.
    delta = _fill_delta(before_no, _position_count_once(client, ticker))
    if delta is None:
        logging.warning(
            "Could not verify the V2 NO-leg position delta on %s — proceeding"
            " unlatched; the NO-leg fill itself was confirmed by the order"
            " response, and the check re-arms on the next V2 NO fill",
            ticker,
        )
        # Named in the CRITICAL if a later pair disproves the mapping
        _V2_UNCHECKED_NO_LEGS.append(ticker)
        return None
    if abs(delta) < _DELTA_EPS:
        # A delta of exactly zero right after a confirmed fill is ambiguous:
        # genuine disproof MOVES the position (the wrong way), while an
        # unchanged ledger can simply be read-after-write lag. Re-reading
        # after each pause in turn separates the two — without it, ledger
        # lag would falsely stop the run with this pair's real NO leg
        # unhedged. Stops at the first re-read that moved.
        for pause in V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS:
            time.sleep(pause)
            delta = _fill_delta(before_no, _position_count_once(client, ticker))
            if delta is None:
                logging.warning(
                    "V2 NO-leg re-read failed on %s after a zero first delta —"
                    " proceeding unlatched; the check re-arms on the next fill",
                    ticker,
                )
                _V2_UNCHECKED_NO_LEGS.append(ticker)
                return None
            if abs(delta) >= _DELTA_EPS:
                break
    if abs(delta + no_leg.count) < _DELTA_EPS:
        _V2_NO_MAPPING_CONFIRMED = True
        logging.info(
            "V2 NO-leg mapping confirmed live: the NO buy moved the position on"
            " %s by %s (our %d NO buy, short-YES by Kalshi's signed convention)"
            " — not re-checked this process",
            ticker, delta, no_leg.count,
        )
        return None
    # Latch before logging, so no later pair of this process sends anything
    unchecked = _stop_run_on_v2_mapping_disproof()
    logging.critical(
        "V2 NO-LEG MAPPING DISPROVEN on %s — the NO leg's ask did not open NO"
        " exposure: expected a position delta of %d, got %s. NOT submitting the"
        " YES leg and NOT auto-unwinding (the unwind is a bid resting on the"
        " same disproven hypothesis, so it could double the error). The rest"
        " of this run is stopped: no later pair sends any order. Stop"
        " trading until this is understood (stop the scheduler daemon if it is"
        " running, and do not run main.py --mode prod; if the defaults server"
        " is running, stop it with Ctrl-C in the terminal running"
        " ./start_dashboard.sh or python3 -m kalshi_betting.defaults_server,"
        " and do not press Confirm and trade), and flatten this"
        " position by hand in the Kalshi UI; there is no other order path to"
        " fall back on.%s",
        ticker, -no_leg.count, delta, unchecked,
    )
    return TradeResult(
        spec=spec, status="manual_review",
        error=(
            f"V2 NO-leg mapping disproven: position delta {delta} after the"
            f" NO-leg fill on {ticker}, expected {-no_leg.count}; YES leg not"
            f" submitted and NO leg not unwound"
        ),
    )


def _execute_one(client: Any, spec: TradeSpec) -> TradeResult:
    """
    Send one pair's two orders, undoing the first if the second fails.

    Gets the NO leg and YES leg from _ordered_legs, sends the NO leg
    (fill_or_kill), then, if it filled, the YES leg (fill_or_kill). If the
    YES leg does not fill, it unwinds the NO leg with _rollback_no_leg, which
    checks that the unwind closed it. Every order goes through
    _submit_order_v2, and the rules are the same for both pair types.

    A "canceled" reply (including Kalshi's 409 kill) means the order did not
    fill. An exception is uncertain — the order may have filled before a
    timeout — so the outcome is judged by how the account's position on that
    ticker CHANGED from the baselines read for both tickers before the NO
    leg was sent. Reading both first keeps retried reads out of the gap
    between the NO fill and the YES order, while the NO position is
    unhedged. A change of zero is read again once after
    _V2_MAPPING_RECHECK_DELAY_SECONDS, because the ledger can lag behind a
    fill (DR-63/DR-64), and the second reading decides. The NO leg's re-read
    is single-shot (_position_count_once); the YES leg's is retried like the
    read beside it, since a failed re-read would leave the NO leg unhedged.

    NO leg uncertain: zero change on both readings → "failed"; exactly
    -no_leg.count (a held NO reads negative) → unwind; anything else →
    "manual_review" with no order sent, since an unwind could close a
    holding this order does not own.

    YES leg uncertain: exactly +yes_leg.count → "executed"; zero on both
    readings → unwind the NO leg; anything else, including a failed read →
    "manual_review" with no order sent, since an unwind could undo a real
    fill.

    On the first NO fill of the process, _confirm_v2_no_mapping checks the
    NO leg's position change; if that disproves the side mapping, the pair
    stops at "manual_review" before the YES leg is sent. While the mapping
    is unconfirmed, an uncertain NO leg whose position made a known change
    other than 0 or -no_leg.count also counts as a disproof (a wrong mapping
    moves it by +no_leg.count).

    Once the mapping has been disproven in this process
    (_V2_NO_MAPPING_DISPROVEN), a pair is stopped first of all, before any
    position is read or any order is built: status "failed", because
    nothing was sent and there is nothing to unwind. The check sits at the
    top rather than just before the NO leg's POST because execute_trades
    runs pairs one at a time until the mapping is settled, so while it does
    no pair can be between the two points when the latch is set; checking
    first also spares each stopped pair its two baseline reads. A pair
    already past this check when another pair sets the latch — possible only
    after execute_trades' time budget
    (config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS), or for a caller that
    runs pairs concurrently some other way — still sends its NO leg; if it
    fills, the pair stops at manual_review before its YES leg (see
    _confirm_v2_no_mapping).

    The pair's pacer places come from one _PairWrites: the NO leg waits for
    two tokens and keeps one for the YES leg, and an unwind takes the hedge
    lane ahead of waiting NO legs. An unsent held place is returned as soon
    as the NO leg's POST raises, or when the pair ends.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.

    Returns:
        TradeResult: status "executed", "failed" (including a pair stopped,
            with nothing sent, because the V2 NO-leg mapping was disproven
            earlier in this process), "rolled_back", "rollback_failed" or
            "manual_review" (see reporter.TradeResult).
    """
    # However the pair ends, an unsent held place goes back to the pacer
    writes = _PairWrites(_ORDER_WRITE_PACER)
    try:
        return _execute_legs(client, spec, writes)
    finally:
        writes.close()


def _execute_legs(client: Any, spec: TradeSpec, writes: _PairWrites) -> TradeResult:
    """
    The body of _execute_one (see its docstring for every path).

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.
        writes (_PairWrites): The pair's pacer places: writes.opening for the
            NO leg, writes.hedge for the YES leg and any unwind.

    Returns:
        TradeResult: As _execute_one.
    """
    # Once a pair has disproven the V2 NO-leg mapping, nothing else is sent in
    # this process: stop before any position read or order build, so the stop
    # costs no API call. Nothing was sent, so "failed" (nothing to unwind);
    # the run still exits EXIT_TRADES_NEED_ATTENTION through the disproving
    # pair's own manual_review result.
    if _V2_NO_MAPPING_DISPROVEN:
        logging.warning(
            "Not sending '%s' (A=%s B=%s) — the V2 NO-leg mapping was disproven"
            " earlier in this run; nothing submitted",
            spec.pair.canonical_title, spec.pair.market_a.ticker,
            spec.pair.market_b.ticker,
        )
        return TradeResult(
            spec=spec, status="failed",
            error=(
                "NO leg not sent: V2 NO-leg mapping disproven earlier in this"
                " run; nothing submitted"
            ),
        )

    # Submission order is a property of the pair type, resolved in exactly one
    # place: the NO leg is always first and is the leg the rollback unwinds.
    no_leg, yes_leg = _ordered_legs(spec)
    order_no  = _build_no_order_v2(no_leg)
    order_yes = _build_yes_order_v2(yes_leg)

    # Read both tickers' positions BEFORE sending anything, so an uncertain
    # outcome is judged by how the position changed, and so no retried read
    # sits between a confirmed NO fill and the YES order. The single-shot
    # reads are _confirm_v2_no_mapping's check (until the run's first
    # confirmation; up to three more reads over 7 s when the ledger reads
    # unchanged) and the NO leg's re-read after a zero change (DR-64). The YES leg has no
    # pacer wait either: it uses the place the NO leg held for it (see
    # _PairWrites).
    before_no = _position_count(client, no_leg.market.ticker)
    before_yes = _position_count(client, yes_leg.market.ticker)

    # Submit the NO leg (single-shot; see _submit_order_v2)
    no_leg_error: str | None = None
    try:
        # Waits for two free tokens and holds the second for the YES leg
        status_no = _submit_order_v2(client, order_no, pace=writes.opening)
        if status_no != "executed":
            # FoK rejection is a confirmed non-fill — safe to walk away
            logging.info(
                "NO leg (%s) not filled (status=%s) — aborting pair",
                no_leg.label, status_no,
            )
            return TradeResult(
                spec=spec, status="failed",
                error=f"NO leg FoK not filled: status={status_no}",
            )
    except Exception as e:
        # One-line description of the error, for the logs and TradeResult.error
        no_leg_error = api_error_summary(e)

    # Disambiguation runs OUTSIDE the except block (mirroring the YES leg
    # below) so the position lookup is not executed while the NO leg's
    # exception is still the active one: anything raised in there would
    # inherit it as __context__, and api_call_with_retry walks that chain — a
    # fatal lookup error would be misread as transient and retried through the
    # full backoff schedule (~62s) before this already-urgent decision could
    # be made.
    if no_leg_error is not None:
        # The YES leg is never sent on this path and the reads below can take a
        # minute, so return the held place now rather than block other pairs
        writes.close()
        # Ambiguous: the order may have filled before the exception (e.g. a
        # timeout after the fill). Attribute by delta against the baseline.
        after_no = _position_count(client, no_leg.market.ticker)
        delta = _fill_delta(before_no, after_no)
        if delta is not None and abs(delta) < _DELTA_EPS:
            # A zero delta is only a CONFIRMED non-fill once the ledger has had
            # a chance to catch up (DR-64). A transport error can be raised
            # milliseconds after the exchange processed — and FILLED — the
            # order, and the positions ledger is read-after-write lagged, so an
            # unmoved first reading is ambiguous exactly as it is in
            # _confirm_v2_no_mapping. Without this re-read the run walked away
            # reporting "failed — nothing to unwind" while a full-size,
            # UNHEDGED NO position was open, and no status in the
            # EXIT_TRADES_NEED_ATTENTION set escalated it.
            #
            # The re-read is _position_count_once — SINGLE-SHOT. This is the
            # unhedged window: the YES leg has not been submitted, so if the NO
            # leg did fill the account is one-sided while we wait, and
            # api_call_with_retry can hold one call for ~62s of backoff. The
            # first read above stays retried (unchanged).
            logging.info(
                "NO leg (%s) raised for '%s' and the ledger reads unchanged —"
                " re-reading once after %ss before calling it a non-fill",
                no_leg.label, spec.pair.canonical_title,
                _V2_MAPPING_RECHECK_DELAY_SECONDS,
            )
            time.sleep(_V2_MAPPING_RECHECK_DELAY_SECONDS)
            delta = _fill_delta(
                before_no, _position_count_once(client, no_leg.market.ticker)
            )
        if delta is not None and abs(delta) < _DELTA_EPS:
            # Confirmed non-fill: the position did not move at all, on two
            # readings a short pause apart
            logging.error(
                "NO leg (%s) submission failed for '%s' (position unchanged —"
                " no fill): %s",
                no_leg.label, spec.pair.canonical_title, no_leg_error,
            )
            return TradeResult(
                spec=spec, status="failed", error=f"NO leg error: {no_leg_error}",
            )
        if delta is not None and abs(delta + no_leg.count) < _DELTA_EPS:
            # Moved by exactly -no_leg.count: our NO buy filled (NO contracts
            # are negative by Kalshi convention). Unwind the now-unhedged leg.
            logging.error(
                "NO leg (%s) raised for '%s' but position moved by %s (our %d"
                " NO buy) — unwinding: %s",
                no_leg.label, spec.pair.canonical_title, delta, no_leg.count,
                no_leg_error,
            )
            return _rollback_no_leg(
                client, spec, no_leg, f"NO leg ambiguous error: {no_leg_error}",
                pace=writes.hedge,
            )
        # Unknown or unexplained change (e.g. an unrelated trade landed in
        # between): send nothing, since an unwind could close a holding this
        # order does not own.
        #
        # A KNOWN change of anything but 0 or -no_leg.count is also what a
        # wrong side mapping makes (an ask that opened YES moves it by
        # +count). While the mapping is not yet confirmed in this process,
        # treat it as a disproof and stop the rest of the run, exactly as the
        # mapping check does: otherwise every later pair whose NO POST also
        # raised would open another wrong-side position without the check
        # ever running. If the change was an unrelated trade, the cost is only
        # the run's remaining trades.
        stops_run = delta is not None and not _V2_NO_MAPPING_CONFIRMED
        stop_note = ""
        if stops_run:
            stop_note = (
                " The V2 NO-leg mapping is not yet confirmed in this process and"
                " a NO buy cannot move the position this way, so the mapping is"
                " treated as disproven: the rest of this run is stopped and no"
                " later pair sends any order. Stop the bot and flatten this"
                " position by hand in the Kalshi UI."
                " To stop the bot, stop the scheduler daemon if it is running,"
                " and do not run main.py --mode prod; if the defaults server is"
                " running, stop it with Ctrl-C in the terminal running"
                " ./start_dashboard.sh or python3 -m kalshi_betting.defaults_server,"
                " and do not press Confirm and trade."
                + _stop_run_on_v2_mapping_disproof()
            )
        logging.critical(
            "NO leg (%s) raised for '%s' and the fill could NOT be attributed"
            " (position delta=%s, expected 0 or %d) — NOT unwinding, since a"
            " reduce-only unwind of a position this order may not own could"
            " close an unrelated holding. Manual review required: %s%s",
            no_leg.label, spec.pair.canonical_title, delta, -no_leg.count,
            no_leg_error, stop_note,
        )
        error = f"NO leg ambiguous, delta={delta}: {no_leg_error}"
        if stops_run:
            error += "; V2 NO-leg mapping treated as disproven, rest of run stopped"
        return TradeResult(spec=spec, status="manual_review", error=error)

    # The NO leg is now a confirmed fill. On the process's first NO fill,
    # check that it really opened a NO position (_confirm_v2_no_mapping); if
    # not, stop here rather than hedge or unwind a position we do not hold.
    # This must run after the NO-leg outcome is settled and before the YES
    # leg is sent, and it judges the change from the NO baseline.
    backstop = _confirm_v2_no_mapping(client, spec, no_leg, before_no)
    if backstop is not None:
        return backstop

    # Submit the YES leg (single-shot); its baseline was read above
    yes_leg_error: str | None = None
    yes_leg_ambiguous = False
    try:
        # Sent on the place the NO leg held for it: no pacer wait
        status_yes = _submit_order_v2(client, order_yes, pace=writes.hedge)
        if status_yes != "executed":
            yes_leg_error = f"YES leg FoK not filled: status={status_yes}"
    except Exception as e:
        # One-line description of the error
        yes_leg_error = f"YES leg error: {api_error_summary(e)}"
        yes_leg_ambiguous = True

    if yes_leg_error:
        if yes_leg_ambiguous:
            # The exception may have arrived after the fill — attribute by
            # delta before rolling the NO leg back, or we'd reverse a completed
            # hedge.
            after_yes = _position_count(client, yes_leg.market.ticker)
            delta = _fill_delta(before_yes, after_yes)
            if delta is not None and abs(delta) < _DELTA_EPS:
                # A zero delta is only a CONFIRMED non-fill once the ledger has
                # had a chance to catch up (DR-63). The YES leg's POST can reach
                # the exchange and FILL milliseconds before the client sees a
                # transport error, and the positions ledger is
                # read-after-write lagged — so an unmoved first reading looked
                # identical to a clean non-fill and sent the reduce-only unwind
                # of a NO leg that was, in truth, hedging a real YES fill. That
                # sold the hedge, left a full-size naked YES position open, and
                # reported it as "rolled_back", which means flat.
                #
                # Retried (_position_count), unlike the NO-leg re-read above:
                # this read's immediate neighbour is already retried, and the
                # decision being protected — do not sell a live hedge — is
                # exactly the ambiguity CLAUDE.md says the retry exists to
                # preserve. The deciding argument is what a FAILED single-shot
                # re-read would cost: an unknown delta, hence manual_review,
                # which leaves the NO leg unhedged INDEFINITELY — strictly
                # worse than a bounded wait and a correct unwind. The residual
                # is the mirror of that: in the branch where the YES leg truly
                # did not fill, the NO leg IS unhedged and this read can hold
                # its loss-floored unwind for up to ~62s of backoff, long
                # enough to turn a rolled_back into a rollback_failed orphan.
                # The first read above already carried that same exposure.
                logging.info(
                    "YES leg (%s) raised for '%s' and the ledger reads"
                    " unchanged — re-reading once after %ss before rolling the"
                    " NO leg back",
                    yes_leg.label, spec.pair.canonical_title,
                    _V2_MAPPING_RECHECK_DELAY_SECONDS,
                )
                time.sleep(_V2_MAPPING_RECHECK_DELAY_SECONDS)
                delta = _fill_delta(
                    before_yes, _position_count(client, yes_leg.market.ticker)
                )
            if delta is not None and abs(delta - yes_leg.count) < _DELTA_EPS:
                # Moved by exactly +yes_leg.count: our YES buy filled, pair
                # complete
                logging.warning(
                    "YES leg (%s) raised for '%s' but position moved by %s (our"
                    " %d YES buy) — pair is complete: %s",
                    yes_leg.label, spec.pair.canonical_title, delta,
                    yes_leg.count, yes_leg_error,
                )
                return TradeResult(
                    spec=spec, status="executed",
                    error=(
                        "YES leg ambiguous but fill confirmed by position delta:"
                        f" {yes_leg_error}"
                    ),
                )
            if delta is None or abs(delta) >= _DELTA_EPS:
                # Either the lookup failed (state genuinely unknown) or the
                # position moved by an amount this order cannot explain.
                # Rolling the NO leg back here would be wrong if the YES leg
                # actually did fill (we'd sell the hedge and be left with a
                # naked YES position while the log says "rolled_back",
                # implying flat). Do NOT auto-rollback; surface for manual
                # review instead.
                logging.critical(
                    "YES leg (%s) raised for '%s' and the fill could NOT be"
                    " attributed (position delta=%s, expected 0 or %d) — NOT"
                    " auto-rolling-back the NO leg to avoid reversing a possible"
                    " real fill. Manual review required: %s",
                    yes_leg.label, spec.pair.canonical_title, delta,
                    yes_leg.count, yes_leg_error,
                )
                return TradeResult(
                    spec=spec, status="manual_review",
                    error=f"YES leg ambiguous, delta={delta}: {yes_leg_error}",
                )
            # delta == 0 → confirmed non-fill; fall through to the rollback below
        logging.error(
            "YES leg (%s) failed after the NO leg filled — attempting rollback: %s",
            yes_leg.label, yes_leg_error,
        )
        return _rollback_no_leg(client, spec, no_leg, yes_leg_error, pace=writes.hedge)

    logging.info(
        "Both legs filled: '%s'  %dx %s, then %dx %s",
        spec.pair.canonical_title, no_leg.count, no_leg.label,
        yes_leg.count, yes_leg.label,
    )
    return TradeResult(spec=spec, status="executed")


def _v2_mapping_unverified() -> bool:
    """
    Whether this process has yet to settle the V2 NO-leg mapping.

    execute_trades runs pairs one at a time while this is True, for at most
    config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS.

    Returns:
        bool: True while this process has neither confirmed nor disproven
            the NO-leg side mapping; False once either latch is set (neither
            is ever cleared within a process).
    """
    return not _V2_NO_MAPPING_CONFIRMED and not _V2_NO_MAPPING_DISPROVEN


def execute_trades(client: Any, specs: list, dry_run: bool = False) -> list:
    """
    Execute each TradeSpec as a sequential two-leg trade, with pairs running
    concurrently across specs once the V2 NO-leg mapping is settled.

    In live mode, each spec is handled by _execute_one(): the NO leg (market_a
    for a same_title pair, market_b for a time_series pair — see
    _ordered_legs) is submitted first, then the YES leg only if the NO leg
    filled, with a floored-limit rollback of the NO leg if the YES leg fails.
    Specs run concurrently via ThreadPoolExecutor, and their POSTs share one
    pacer (_ORDER_WRITE_PACER), where a pair's YES leg and unwind go ahead of
    other pairs' NO legs.

    The exception is the time before this process has confirmed or disproven
    the NO-leg side mapping (_confirm_v2_no_mapping): until then a pair runs
    alone, and the next one starts only when it has settled the mapping or
    finished. The mapping is checked on a pair's own NO fill, and concurrent
    pairs all send their NO legs before the first check can finish (on the
    first live run, 2026-09-28, all 7 NO legs went out before the first
    confirmation), so a disproof would otherwise find a wrong-side position
    on every pair already in flight. One at a time, a disproof costs one
    position: its latch stops every later pair before anything is sent
    (status "failed"). The next pair starts the moment the mapping is
    settled, even while the pair that settled it is still sending its YES
    leg or an unwind; once it is confirmed the remaining pairs start
    together, as before.

    A pair can also finish without a verdict: its NO leg killed, its NO POST
    raising (an uncertain leg the check never reaches, unless its position
    moved in a way that disproves the mapping), its worker raising, or the
    check unable to read the account; the next pair then runs alone in turn.
    When position reads keep failing every pair would finish that way behind
    about two minutes of retried reads, so the phase is bounded in time:
    once it has lasted config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS, the
    pair still running is no longer waited for, a WARNING is logged, and the
    rest start together, each still checking its own NO fill; a disproof
    after that stops only the pairs that start after it. The bound is time
    rather than a count of such pairs because a killed NO leg costs only a
    round trip and must not use up the protection. The cost is paid once
    per process: usually the first pair's NO POST and mapping check (a few
    round trips, up to 7 s more when the ledger lags), and never more than
    the budget. A stuck order POST likewise holds the rest back for at most
    the budget (execute_trades itself still returns only when that POST
    does).

    In dry_run mode, no orders are submitted. The function logs the intended
    trade — both legs in SUBMISSION order (NO leg first), with each leg's own
    count and traded price, which for a time_series pair are NOT the pair's
    nA/pB — and returns TradeResult objects with status="simulated", which are
    still written to the dev simulation Excel file by reporter.py.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().
            Must be pointed at the correct endpoint (prod vs. sandbox).
        specs (list): List of TradeSpec objects from strategy.select_portfolio().
            Each spec encodes one pair with a final integer contract count.
        dry_run (bool): If True, skip actual order submission and return simulated
            results. Defaults to False. Always True in dev/sandbox mode.

    Returns:
        list: List of TradeResult objects (from reporter.py), one per spec. Each
            result has status="executed" (both legs filled), "simulated" (dry
            run), "failed" (NO leg confirmed unfilled, or nothing sent — a
            pair stopped after the V2 NO-leg mapping was disproven),
            "rolled_back" (YES leg
            confirmed unfilled, NO leg unwound), "rollback_failed" (NO-leg
            unwind did not fill, or closed only part of the position —
            orphaned position), or "manual_review" (a
            leg's fill state could not be attributed to this order, or an
            exception escaped the worker — no automated order was submitted in
            response). The list is in SUBMISSION order: results[i] corresponds
            to specs[i].
    """
    # ThreadPoolExecutor(max_workers=0) raises ValueError, so short-circuit empty input
    if not specs:
        return []

    if dry_run:
        results = []
        for spec in specs:
            # Same resolution the live path uses, so the dry-run line shows
            # exactly the legs (sides, counts, prices) that WOULD be submitted,
            # NO leg first
            no_leg, yes_leg = _ordered_legs(spec)
            logging.info(
                "[DRY RUN] Pair order (NO leg first, then YES — there is no batch "
                "endpoint): Buy %dx %s @ %.2f%% | Buy %dx %s @ %.2f%% | "
                "Total cost: $%.2f incl. fees | Profit if won: $%.2f",
                no_leg.count, no_leg.label, no_leg.price_dollars * 100,
                yes_leg.count, yes_leg.label, yes_leg.price_dollars * 100,
                spec.total_cost_with_fees, spec.min_payoff,
            )
            results.append(TradeResult(spec=spec, status="simulated"))
        return results

    with ThreadPoolExecutor(max_workers=min(TRADER_MAX_WORKERS, len(specs))) as pool:
        future_to_spec: dict = {}
        # The one-at-a-time phase ends when the mapping is settled or at this
        # deadline, whichever comes first (see the docstring)
        deadline = time.monotonic() + V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS
        budget_logged = False
        for spec in specs:
            future = pool.submit(_execute_one, client, spec)
            future_to_spec[future] = spec
            # Checked after submitting: once the mapping is settled there is
            # nothing to wait for
            if not _v2_mapping_unverified():
                continue
            if time.monotonic() >= deadline:
                if not budget_logged:
                    logging.warning(
                        "V2 NO-leg mapping still unverified after %gs of"
                        " running pairs one at a time — starting the remaining"
                        " pairs together; each still checks its own NO fill,"
                        " but a disproof now stops only the pairs that start"
                        " after it",
                        V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS,
                    )
                    budget_logged = True
                continue
            # Hold the next pair back until this one settles the mapping,
            # finishes, or the deadline passes. The wait never raises; a
            # worker's exception is collected below.
            while _v2_mapping_unverified() and not future.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_for_futures(
                    [future], timeout=min(V2_MAPPING_VERDICT_POLL_SECONDS, remaining),
                )
        results = []
        # Collect per-future so one worker's exception cannot discard every other
        # pair's TradeResult — including confirmed real fills. A list
        # comprehension over .result() re-raised out of execute_trades, and the
        # caller (main._run_prod) reads results only AFTER this returns: the
        # Excel rows, the CRITICAL manual-review summary and the
        # EXIT_TRADES_NEED_ATTENTION exit code were all lost with it. The pool's
        # `with` block still waits for every worker, so the OTHER pairs' orders
        # and rollbacks have already completed by the time we get here — this
        # only preserves the record, it never changes what is bought or sold.
        # The dict preserves insertion (submission) order, which is the order
        # the caller pairs results[i] with specs[i] in.
        for future, spec in future_to_spec.items():
            try:
                results.append(future.result())
            except Exception as exc:
                # The raising pair's own fill state is unattributable (it may
                # have filled the NO leg and died before the YES leg or the
                # rollback), so no order is submitted in response — an unwind
                # could reverse a real fill, the same reasoning behind every
                # other manual_review case in _execute_one. "A"/"B" are MARKET
                # labels (market_a / market_b), not submission legs.
                # One-line description in the message and the result;
                # exc_info adds the traceback
                logging.critical(
                    "Unhandled exception executing '%s' (A=%s B=%s) — fill state "
                    "UNKNOWN, manual review required: %s",
                    spec.pair.canonical_title,
                    spec.pair.market_a.ticker,
                    spec.pair.market_b.ticker,
                    api_error_summary(exc),
                    exc_info=True,
                )
                results.append(
                    TradeResult(
                        spec=spec,
                        status="manual_review",
                        error=f"Unhandled exception in _execute_one: {api_error_summary(exc)}",
                    )
                )

    return results
