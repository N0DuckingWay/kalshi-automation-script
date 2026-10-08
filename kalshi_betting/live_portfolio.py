"""
File: live_portfolio.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Works out how the live account is doing, for the live dashboard's Live
    trading tab. Reads the account from Kalshi with read-only GETs (every
    fill, settlement, deposit and withdrawal, then the balance and the
    positions) and reads the bot's trade log, which lists the purchases the
    bot made. Each of the bot's purchases is matched to the one Kalshi order
    that made it; every other fill in the account counts as "Other bets"
    (your own trades). Then every fill and payout is replayed, oldest first,
    into a ledger of the contracts each owner bought, sold and was paid for,
    with the cash each one moved.

Dependencies:
    config (the LIVE_* page sizes, matching windows and read retries,
    SCANNER_MAX_PAGES, count_text); auth (_positions_value_cents, its reading
    of what the balance reply says the positions are worth); historical
    (_historical_get, the retried, signed, read-only GET, looked up when it is
    called so tests can replace it); reporter (PROD_LOG_PATH, where the
    trade log lives, also looked up when it is called). Nothing in the
    trading pipeline imports this module: it only reads.

Notes:
    A fill's direction comes from its book_side alone: "bid" buys YES (or
    closes NO held), "ask" sells YES (or opens NO). Kalshi's side and action
    fields are never read, since they disagree with Kalshi's own documents
    (a NO purchase arrives with action "sell").

    Money and counts are Decimals throughout, read from Kalshi's text, so the
    ledger's cash adds up exactly. Where a fee or a cost is shared between
    pieces of one fill, each piece is rounded to config.LIVE_SHARE_STEP_DOLLARS
    and the last piece takes what is left, so the pieces add back to the whole.

    A record from Kalshi that cannot be read stops the whole read with an
    error, rather than being left out: a fill left out would make every later
    figure wrong without saying so.

    The trade log's dates and times are this computer's local time (reporter
    writes them that way), so they are read in the local time zone.
"""
from __future__ import annotations

import dataclasses
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import openpyxl

from . import auth, config, historical, reporter

# The owner of every contract the bot's trade log does not account for
OTHER_BETS = "Other bets"

_ZERO, _ONE = Decimal(0), Decimal(1)

# Trade-log statuses whose orders reached Kalshi: these rows are the bot's
_BOT_STATUSES = frozenset({"executed", "rolled_back", "rollback_failed", "manual_review"})

# The bot's other trade-log statuses: a pair that sent nothing that filled
# ("failed") and a dry run's pair ("simulated")
_OTHER_STATUSES = frozenset({"failed", "simulated"})

# Statuses whose rows sent an order to buy their NO leg back (an unwind; on
# "rollback_failed" it bought back part of the leg, or none)
_UNWOUND_STATUSES = frozenset({"rolled_back", "rollback_failed"})

# The start of a trade-log row's Notes: "[time_series: YES A / NO B ..."
_NOTE = re.compile(r"^\[(\w+): (YES|NO) A / (YES|NO) B")

# The trade-log header cells this module reads (0-based column: header).
# Columns 11 and 12 (the counts) are left out: older logs name them
# differently, and their place has never changed.
_LOG_HEADERS = {0: "Date", 1: "Time", 2: "Market A", 3: "Ticker A", 5: "Ticker B",
                16: "Status", 17: "Notes"}

# Column positions in a trade-log row (0-based)
_COL_DATE, _COL_TIME, _COL_TITLE, _COL_TICKER_A, _COL_TICKER_B = 0, 1, 2, 3, 5
_COL_COUNT_A, _COL_COUNT_B, _COL_STATUS, _COL_NOTES = 11, 12, 16, 17
_LOG_COLUMNS = 18

# A run's banner row: "── Run: 2026-09-28 02:27  |  Balance before: $132.45  →  ..."
_BANNER_START = "── Run:"
_BANNER = re.compile(r"^── Run: .*?Balance before: \$(-?[\d,]+\.\d+)")

# The names reporter gives its fallback copies of the trade log:
# trade_log_<date>_<time>_<microseconds>.xlsx, with "-N" (or, rarely, eight
# hex digits) added when that name is taken. A copy you made yourself, such as
# Finder's "trade_log copy.xlsx", never matches.
_FALLBACK_LOG = re.compile(
    r"trade_log_\d{4}-\d{2}-\d{2}_\d{6}_\d{6}(-(\d+|[0-9a-f]{8}))?\.xlsx")


# ---- what Kalshi says ------------------------------------------------------

@dataclass(frozen=True)
class Fill:
    """
    One fill: part (or all) of one order, as Kalshi lists it.

    Attributes:
        fill_id (str): Kalshi's id of the fill.
        order_id (str): The order the fill belongs to ("" when Kalshi gave none).
        ticker (str): The market.
        time (datetime): When it filled (UTC).
        buys_yes (bool): True for a bid (buys YES, or closes NO held), False
            for an ask (sells YES held, or opens NO).
        count (Decimal): Contracts filled, always above 0.
        yes_price (Decimal): The YES price of the fill, in dollars.
        no_price (Decimal): The NO price of the fill (1 less the YES price).
        fee (Decimal): The fee Kalshi charged for this fill, in dollars.
    """
    fill_id: str
    order_id: str
    ticker: str
    time: datetime
    buys_yes: bool
    count: Decimal
    yes_price: Decimal
    no_price: Decimal
    fee: Decimal


@dataclass(frozen=True)
class Payout:
    """
    One market's settlement: what each side's contracts were paid.

    Attributes:
        ticker (str): The market.
        time (datetime): When it settled (UTC).
        result (str): "yes", "no" or "scalar".
        yes_value (Decimal | None): What one YES contract pays on a scalar
            result, in dollars (a NO contract pays 1 less this).
        revenue (Decimal | None): What Kalshi credited the account, in
            dollars; None for a settlement rebuilt from the market itself.
    """
    ticker: str
    time: datetime
    result: str
    yes_value: Decimal | None
    revenue: Decimal | None


@dataclass(frozen=True)
class CashFlow:
    """
    One deposit or withdrawal that reached the account.

    Attributes:
        time (datetime): When it was applied (UTC).
        amount (Decimal): Dollars into the account (a deposit, less its fee)
            or out of it (a withdrawal plus its fee, negative).
    """
    time: datetime
    amount: Decimal


@dataclass(frozen=True)
class Market:
    """
    What the live dashboard needs to know about one market.

    Attributes:
        ticker (str): The market.
        event_ticker (str): Its event ("" when Kalshi gave none).
        title (str): Its title (the ticker when it has none).
        status (str): Kalshi's status for it, e.g. "active" or "finalized".
        result (str): "yes", "no", "scalar", or "" while undecided.
        settled_at (datetime | None): When it settled, if it has.
        yes_value (Decimal | None): What one YES contract pays on settlement.
        yes_bid (Decimal | None): The best YES bid now.
        yes_ask (Decimal | None): The best YES ask now.
        last_price (Decimal | None): The YES price of the last trade.
    """
    ticker: str
    event_ticker: str
    title: str
    status: str
    result: str
    settled_at: datetime | None
    yes_value: Decimal | None
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    last_price: Decimal | None


@dataclass(frozen=True)
class Account:
    """
    One read of the account.

    Attributes:
        read_at (datetime): When the balance was read (UTC).
        cash (Decimal): Cash on every shard together, to the hundredth of a
            cent the account keeps.
        shards (int): How many shards the balance reply listed (the trade
            log's "Balance before" rounds each one down to the cent).
        kalshi_positions_value (Decimal | None): What Kalshi says the open
            positions are worth, in dollars; None when the reply has no
            usable value.
        positions (dict[str, Decimal]): Contracts held now, by market
            (YES positive, NO negative); markets held at 0 are left out.
        fills (tuple[Fill, ...]): Every fill, oldest first.
        payouts (tuple[Payout, ...]): The settlements Kalshi still lists.
        flows (tuple[CashFlow, ...]): Deposits and withdrawals, oldest first.
        changing (bool): True when the account kept changing while it was
            read, so the read may not be consistent.
        warnings (tuple[str, ...]): Anything odd in the replies that the read
            worked around, for the page to show (empty when nothing was).
    """
    read_at: datetime
    cash: Decimal
    shards: int
    kalshi_positions_value: Decimal | None
    positions: dict[str, Decimal]
    fills: tuple[Fill, ...]
    payouts: tuple[Payout, ...]
    flows: tuple[CashFlow, ...]
    changing: bool
    warnings: tuple[str, ...] = ()


def _dec(value: Any) -> Decimal:
    """
    Read a Kalshi number as a Decimal.

    Kalshi sends dollars and counts as text ("0.4700", "13.00") and some
    amounts as whole numbers of cents; both read exactly.

    Args:
        value (Any): The number as Kalshi sent it (text, an int or a float).

    Returns:
        Decimal: The number.

    Raises:
        ValueError: If it is missing, True or False, not a number, or not finite.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"not a number: {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _opt_dec(value: Any) -> Decimal | None:
    """
    Read a Kalshi number as a Decimal, or None when it cannot be read.

    Args:
        value (Any): The number as Kalshi sent it, or None.

    Returns:
        Decimal | None: The number; None if it is missing or unreadable.
    """
    try:
        return _dec(value)
    except ValueError:
        return None


def _when(value: Any) -> datetime:
    """
    Read a Kalshi time as an aware UTC datetime.

    Args:
        value (Any): ISO text ("2026-09-28T09:27:00.123456Z") or seconds
            since 1970 as a number.

    Returns:
        datetime: The moment, in UTC.

    Raises:
        ValueError: If the text is not a time, or names no time zone.
        OverflowError: If a number of seconds is beyond the range of a datetime.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, UTC)
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"time without a zone: {value!r}")
    return moment.astimezone(UTC)


def _get(client: Any, path: str, **params: Any) -> dict:
    """
    Send one read-only GET to Kalshi and return its JSON object.

    Every Kalshi request this module makes goes through here, to
    historical._historical_get: signed, retried on rate limits and server
    errors, and looked up when it is called so tests can replace it. The
    path is put after historical._API_PREFIX, the production API's own path
    ("/trade-api/v2"), the same prefix the backtest's requests use.

    Args:
        client (Any): A client from auth.build_client.
        path (str): The path after the API prefix, e.g. "/portfolio/fills".
        **params (Any): Query parameters; None values are left out.

    Returns:
        dict: The reply.

    Raises:
        ValueError: If the reply is not a JSON object.
        ApiException: If Kalshi answers with an error status after the retries.
    """
    data = historical._historical_get(client, historical._API_PREFIX + path, **params)
    if not isinstance(data, dict):
        raise ValueError(f"{path} answered with {type(data).__name__}, not an object")
    return data


def _pages(client: Any, path: str, key: str, **params: Any) -> list[dict]:
    """
    Read every record of a Kalshi listing, page by page.

    Follows the reply's cursor until a page comes back without one.

    Args:
        client (Any): A client from auth.build_client.
        path (str): The listing's path after the API prefix.
        key (str): The reply's key that holds the records, e.g. "fills".
        **params (Any): Query parameters sent with every page.

    Returns:
        list[dict]: Every record, in the order Kalshi listed them (records
            that are not JSON objects are left out).

    Raises:
        ValueError: If a reply has no list under `key`, a cursor comes back a
            second time, or the listing runs past config.SCANNER_MAX_PAGES pages.
    """
    records: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(config.SCANNER_MAX_PAGES):
        data = _get(client, path, cursor=cursor, **params)
        rows = data.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"{path} answered without a {key!r} list")
        records.extend(row for row in rows if isinstance(row, dict))
        cursor = data.get("cursor") or None
        if cursor is None:
            return records
        if cursor in seen:
            raise ValueError(f"{path} repeated a page cursor")
        seen.add(cursor)
    raise ValueError(f"{path} did not finish within {config.SCANNER_MAX_PAGES} pages")


def _fill(row: dict) -> Fill:
    """
    Read one fill record.

    Args:
        row (dict): A record from /portfolio/fills or /historical/fills.

    Returns:
        Fill: The fill.

    Raises:
        ValueError: If its book_side is neither "bid" nor "ask", its count is
            not above 0, or a number or time cannot be read.
        KeyError: If a field it needs is missing.
    """
    side = row.get("book_side")
    if side not in ("bid", "ask"):
        raise ValueError(f"fill {row.get('fill_id')!r} has book_side {side!r}")
    count = _dec(row["count_fp"])
    if count <= 0:
        raise ValueError(f"fill {row.get('fill_id')!r} has count {count}")
    return Fill(str(row["fill_id"]), str(row.get("order_id") or ""), str(row["ticker"]),
                _when(row["created_time"]), side == "bid", count,
                _dec(row["yes_price_dollars"]), _dec(row["no_price_dollars"]),
                _dec(row.get("fee_cost") or "0"))


def _payout_record(row: dict) -> Payout:
    """
    Read one settlement record.

    Kalshi gives its revenue and value in whole cents; both are read as dollars.

    Args:
        row (dict): A record from /portfolio/settlements.

    Returns:
        Payout: The settlement, with the revenue Kalshi credited.

    Raises:
        ValueError: If a number or time cannot be read.
        KeyError: If the ticker or the settlement time is missing.
    """
    value = row.get("value")
    return Payout(str(row["ticker"]), _when(row["settled_time"]),
                  str(row.get("market_result") or ""),
                  None if value is None else _dec(value) / 100,
                  _dec(row.get("revenue") or 0) / 100)


def _payout_key(row: dict) -> tuple[datetime, str]:
    """
    Name one settlement record by its time and market.

    Args:
        row (dict): A record from /portfolio/settlements.

    Returns:
        tuple[datetime, str]: Its settlement time and its ticker.
    """
    return _when(row["settled_time"]), str(row["ticker"])


def _cash(balance: dict) -> tuple[Decimal, int, list[str]]:
    """
    Read the cash on every shard from a balance reply.

    Each shard's entry is in dollars (balance_dollars, else balance, which
    inside an entry is dollar text too). A shard listed twice counts once,
    by its last entry, as auth._balance_cents_by_shard (the bot's own
    reading) counts it, with a warning. With no shard list, the reply's own
    balance_dollars, else its balance in whole cents, is the one shard.

    Args:
        balance (dict): The reply from /portfolio/balance.

    Returns:
        tuple[Decimal, int, list[str]]: The cash in dollars, how many shards
            the reply lists, and a warning per shard listed more than once.

    Raises:
        ValueError: If the reply has no cash this can read.
    """
    shards = balance.get("balance_breakdown")
    if isinstance(shards, list) and shards:
        by_shard: dict[str, Decimal] = {}
        warnings: list[str] = []
        for shard in shards:
            if not isinstance(shard, dict):
                raise ValueError(f"a balance shard is {type(shard).__name__}, not an object")
            index = str(shard.get("exchange_index"))
            if index in by_shard:
                warnings.append(f"Kalshi's balance reply lists shard {index} twice: "
                                f"only its last entry is counted")
            by_shard[index] = _dec(shard["balance_dollars"] if "balance_dollars" in shard
                                   else shard.get("balance"))
        return sum(by_shard.values(), _ZERO), len(by_shard), warnings
    if "balance_dollars" in balance:
        return _dec(balance["balance_dollars"]), 1, []
    return _dec(balance.get("balance")) / 100, 1, []


def _read_once(client: Any) -> tuple[Account, frozenset[str], frozenset[tuple[datetime, str]]]:
    """
    Read the whole account once.

    Reads the records first (live fills, then archived ones; settlements;
    deposits and withdrawals), then the balance and the positions, so the
    balance is never older than the records. Live fills come first because
    Kalshi moves a fill from its live listing to its archive, never back: a
    fill that moves while the two are read is in the archive by the time
    the archive is read. A fill in both is taken from the live listing.

    Args:
        client (Any): A client from auth.build_client.

    Returns:
        tuple: The account (changing False); the ids of every fill read; and
            every settlement read, named by its time and ticker.

    Raises:
        ValueError: If a reply or a record cannot be read.
        KeyError: If a record lacks a field this needs.
        ApiException: If Kalshi answers with an error status after the retries.
    """
    fills: dict[str, Fill] = {}
    for path in ("/portfolio/fills", "/historical/fills"):          # live, then archived
        for row in _pages(client, path, "fills", limit=config.LIVE_PAGE_SIZE):
            fill = _fill(row)
            fills.setdefault(fill.fill_id, fill)
    payouts = tuple(_payout_record(row) for row in _pages(
        client, "/portfolio/settlements", "settlements", limit=config.LIVE_PAGE_SIZE))
    flows = []
    for path, key, deposit in (("/portfolio/deposits", "deposits", True),
                               ("/portfolio/withdrawals", "withdrawals", False)):
        for row in _pages(client, path, key, limit=config.LIVE_TRANSFERS_PAGE_SIZE):
            if row.get("status") != "applied":
                continue                      # pending or failed: no money moved
            amount = _dec(row["amount_cents"]) / 100
            fee = _dec(row.get("fee_cents") or 0) / 100
            flows.append(CashFlow(_when(row.get("finalized_ts") or row["created_ts"]),
                                  amount - fee if deposit else -(amount + fee)))
    balance = _get(client, "/portfolio/balance")
    read_at = datetime.now(UTC)
    positions: dict[str, Decimal] = {}
    for row in _pages(client, "/portfolio/positions", "market_positions",
                      limit=config.LIVE_PAGE_SIZE, count_filter="position"):
        if count := _dec(row["position_fp"]):
            positions[str(row["ticker"])] = count
    cash, shards, warnings = _cash(balance)
    # Cross-module: auth's own reading of the positions value, in whole cents
    cents = auth._positions_value_cents(balance)
    ordered = tuple(sorted(fills.values(), key=lambda f: (f.time, f.fill_id)))
    account = Account(read_at, cash, shards, None if cents is None else Decimal(cents) / 100,
                      positions, ordered, payouts, tuple(sorted(flows, key=lambda f: f.time)),
                      False, tuple(warnings))
    return account, frozenset(fills), frozenset((p.time, p.ticker) for p in payouts)


def _pages_first(client: Any, path: str, key: str) -> dict | None:
    """
    Read the newest record of a Kalshi listing (Kalshi lists newest first).

    Args:
        client (Any): A client from auth.build_client.
        path (str): The listing's path after the API prefix.
        key (str): The reply's key that holds the records.

    Returns:
        dict | None: The newest record, or None when the listing is empty.

    Raises:
        ValueError: If the reply is not a JSON object.
    """
    rows = _get(client, path, limit=1).get(key)
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None


def read_account(client: Any) -> Account:
    """
    Read the account, reading it again when it changed while it was read.

    After each read, the newest fill and the newest settlement are read once
    more. When either is one the read did not include, or the newest fill is
    under config.LIVE_READ_SETTLE_SECONDS old (Kalshi's balance takes about a
    second to include a fill), it waits config.LIVE_READ_RETRY_PAUSE_SECONDS
    and reads again, at most config.LIVE_READ_ATTEMPTS times in all.

    Deposits and withdrawals are not checked again: one applied in the
    moment between the read of their listings and the balance read would be
    in the cash but not among the deposits, and the cash check would show
    the difference.

    Args:
        client (Any): A client from auth.build_client.

    Returns:
        Account: The last read; its `changing` is True when every read found
            the account still changing.

    Raises:
        ValueError: If a reply or a record cannot be read.
        KeyError: If a record lacks a field this needs.
        ApiException: If Kalshi answers with an error status after the retries.
    """
    settle = timedelta(seconds=config.LIVE_READ_SETTLE_SECONDS)
    attempts = max(1, config.LIVE_READ_ATTEMPTS)
    for attempt in range(attempts):
        account, fill_ids, payout_keys = _read_once(client)
        latest_fill = _pages_first(client, "/portfolio/fills", "fills")
        latest_payout = _pages_first(client, "/portfolio/settlements", "settlements")
        # Kalshi lists newest first, so a record made after the read is the
        # first one now, and the read does not have it
        unchanged = ((latest_fill is None or str(latest_fill.get("fill_id")) in fill_ids)
                     and (latest_payout is None or _payout_key(latest_payout) in payout_keys))
        fresh = bool(account.fills) and account.read_at - account.fills[-1].time < settle
        if unchanged and not fresh:
            return account
        if attempt + 1 < attempts:
            time.sleep(config.LIVE_READ_RETRY_PAUSE_SECONDS)
    return dataclasses.replace(account, changing=True)


def _market(row: dict) -> Market:
    """
    Read one market record from /markets or /historical/markets.

    Args:
        row (dict): The market record.

    Returns:
        Market: The market; a price it cannot read is None.

    Raises:
        ValueError: If its settlement time cannot be read.
        KeyError: If it has no ticker.
    """
    settled = row.get("settlement_ts")
    return Market(str(row["ticker"]), str(row.get("event_ticker") or ""),
                  str(row.get("title") or row["ticker"]), str(row.get("status") or ""),
                  str(row.get("result") or ""), _when(settled) if settled else None,
                  _opt_dec(row.get("settlement_value_dollars")),
                  _opt_dec(row.get("yes_bid_dollars")), _opt_dec(row.get("yes_ask_dollars")),
                  _opt_dec(row.get("last_price_dollars")))


def read_markets(client: Any, tickers: Iterable[str]) -> dict[str, Market]:
    """
    Look markets up: on Kalshi's live listing first, then in its archive for the rest.

    Asks for config.LIVE_MARKET_TICKERS_PER_REQUEST markets per request.

    Args:
        client (Any): A client from auth.build_client.
        tickers (Iterable[str]): The markets wanted.

    Returns:
        dict[str, Market]: Each market found, by ticker; a market neither
            listing has is left out.

    Raises:
        ValueError: If a reply or a record cannot be read.
        ApiException: If Kalshi answers with an error status after the retries.
    """
    wanted = sorted(set(tickers))
    found: dict[str, Market] = {}
    step = config.LIVE_MARKET_TICKERS_PER_REQUEST
    for path in ("/markets", "/historical/markets"):
        missing = [t for t in wanted if t not in found]
        for i in range(0, len(missing), step):
            for row in _pages(client, path, "markets", tickers=",".join(missing[i:i + step]),
                              limit=config.LIVE_PAGE_SIZE):
                market = _market(row)
                found[market.ticker] = market
    return found


# ---- the bot's trade log ---------------------------------------------------

@dataclass(frozen=True)
class BotLeg:
    """
    One leg of a bot purchase, as the trade log records it.

    Attributes:
        ticker (str): The market.
        side (str): "yes" or "no": the side the leg bought.
        count (Decimal): Contracts the leg bought.
    """
    ticker: str
    side: str
    count: Decimal


@dataclass(frozen=True)
class BotTrade:
    """
    One purchase the bot sent to Kalshi, from one trade-log row.

    Attributes:
        trade_id (str): "<file name>#<row number>", unique across the logs.
        status (str): The row's status: "executed", "rolled_back",
            "rollback_failed" or "manual_review".
        title (str): Market A's title as the log shows it.
        legs (tuple[BotLeg, BotLeg]): Market A's leg, then market B's.
        run_after (datetime): The previous real run's log time (UTC), or the
            log time less config.LIVE_BOT_RUN_WINDOW_SECONDS if that is
            later: this run's orders come after it.
        logged_at (datetime): When this run wrote the log (UTC): its orders
            come before it.
    """
    trade_id: str
    status: str
    title: str
    legs: tuple[BotLeg, BotLeg]
    run_after: datetime
    logged_at: datetime


@dataclass(frozen=True)
class RunStart:
    """
    One real run of the bot, as its banner row in the trade log records it.

    Attributes:
        run_after (datetime): As in BotTrade: this run's orders come after it.
        logged_at (datetime): When the run wrote the log (UTC).
        cash_before (Decimal): The banner's "Balance before": the cash before
            the run traded, each shard rounded down to the cent.
    """
    run_after: datetime
    logged_at: datetime
    cash_before: Decimal


def trade_log_paths() -> list[Path]:
    """
    List the bot's trade logs: the shared one and reporter's fallback copies beside it.

    The shared log is reporter.PROD_LOG_PATH, looked up when this is called.
    A fallback copy is one reporter wrote when it could not take the shared
    log's lock, named trade_log_<date>_<time>_<microseconds>.xlsx; a copy
    you made yourself is never read.

    Returns:
        list[Path]: The shared log (when it exists), then the fallback copies by name.
    """
    main_log = reporter.PROD_LOG_PATH
    copies = sorted(p for p in main_log.parent.glob("trade_log_*.xlsx")
                    if _FALLBACK_LOG.fullmatch(p.name) and p.is_file())
    return ([main_log] if main_log.is_file() else []) + copies


def _log_readings(day: Any, clock: Any) -> tuple[datetime, datetime]:
    """
    Read a trade-log row's Date and Time as UTC moments.

    reporter writes them as text in this computer's local time. A workbook
    saved by hand can turn them into a date and a time instead; those read
    the same. In the hour that repeats when clocks go back, the same Date
    and Time name two moments an hour apart; everywhere else they name one.

    Args:
        day (Any): The Date cell, e.g. "2026-09-28" (or a date).
        clock (Any): The Time cell, e.g. "02:27:00" (or a time).

    Returns:
        tuple[datetime, datetime]: The earlier and the later reading, in UTC;
            the same moment twice outside the repeated hour.

    Raises:
        ValueError: If the cells are not a date and a time.
    """
    if hasattr(day, "strftime"):
        day = day.strftime("%Y-%m-%d")
    if hasattr(clock, "strftime"):
        clock = clock.strftime("%H:%M:%S")
    text = f"{day} {clock}".strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        first, second = (naive.replace(fold=fold).astimezone(UTC) for fold in (0, 1))
        return min(first, second), max(first, second)
    raise ValueError(f"unreadable date and time: {text!r}")


def _status(row: tuple) -> str:
    """
    A trade-log row's Status, with any spaces around it removed.

    Args:
        row (tuple): The row's values.

    Returns:
        str: The status ("" for an empty cell).
    """
    return str(row[_COL_STATUS] or "").strip()


class _NotATradeLog(Exception):
    """A workbook whose first row is not the trade log's header."""


def _log_rows(path: Path, file_index: int) -> tuple[list[tuple], list[str]]:
    """
    Read one trade log's trade rows.

    reporter adds rows in time order, so each row's time is taken as the
    latest of its readings (_log_readings) that is not after the next row's
    time. That tells the two passes through the hour that repeats when
    clocks go back apart; a row with nothing after it in that hour takes the
    later reading, which still comes after its run's orders.

    Args:
        path (Path): The workbook.
        file_index (int): Its place among the logs read, for ordering rows.

    Returns:
        tuple: The rows, each (log time, (file_index, row number), trade id,
            row values, the cash on the run's banner row or None), and a
            warning for each row whose date or time cannot be read.

    Raises:
        Exception: Whatever openpyxl raises for a file it cannot read, or
            _NotATradeLog when the first row is not the trade log's header.
    """
    read: list[tuple] = []
    warnings: list[str] = []
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book.worksheets[0].iter_rows(values_only=True)
        header = next(rows, ())
        if any(len(header) <= i or header[i] != name for i, name in _LOG_HEADERS.items()):
            raise _NotATradeLog(path.name)
        banner_cash = None
        for number, row in enumerate(rows, start=2):
            first = row[0] if row else None
            if isinstance(first, str) and first.startswith(_BANNER_START):
                match = _BANNER.match(first)
                banner_cash = _dec(match.group(1).replace(",", "")) if match else None
                continue
            if len(row) < _LOG_COLUMNS or not _status(row):
                continue                                      # a blank row
            try:
                readings = _log_readings(row[_COL_DATE], row[_COL_TIME])
            except ValueError:
                warnings.append(f"{path.name} row {number}: unreadable date or time")
                continue
            read.append((readings, number, tuple(row[:_LOG_COLUMNS]), banner_cash))
    finally:
        book.close()
    rows_out: list[tuple] = []
    next_time: datetime | None = None
    for (earlier, later), number, values, banner_cash in reversed(read):
        logged = earlier if next_time is not None and later > next_time >= earlier else later
        next_time = logged
        rows_out.append((logged, (file_index, number), f"{path.name}#{number}", values,
                         banner_cash))
    rows_out.reverse()
    return rows_out, warnings


def read_trade_logs(paths: Iterable[Path]) -> tuple[list[BotTrade], list[RunStart], list[str]]:
    """
    Read the bot's purchases and real runs from its trade logs.

    A row is a real run's when its status is anything but "simulated" (a dry
    run's rows are simulated and never bound a window). Each real run's
    orders are looked for after the previous real run's log time (and at
    most config.LIVE_BOT_RUN_WINDOW_SECONDS before its own). A row repeated
    exactly, in the same log or another, is read once. A file or row that
    cannot be read, or a status this does not know, is a warning, never an
    exception: those trades then count as Other bets.

    Args:
        paths (Iterable[Path]): The trade logs, e.g. trade_log_paths().

    Returns:
        tuple: The bot's purchases (rows whose status is in _BOT_STATUSES),
            oldest first; one RunStart per real run whose banner row gives
            its cash before; and the warnings.
    """
    entries: list[tuple] = []
    warnings: list[str] = []
    seen_rows: set[tuple] = set()
    for file_index, path in enumerate(paths):
        try:
            rows, row_warnings = _log_rows(path, file_index)
        except _NotATradeLog:
            warnings.append(f"{path.name} is not a trade log this page can read: "
                            f"its trades count as {OTHER_BETS}")
            continue
        except Exception as exc:  # a damaged file: its trades count as Other bets, and the page says so
            warnings.append(f"Could not read {path.name} ({type(exc).__name__}): "
                            f"its trades count as {OTHER_BETS}")
            continue
        warnings.extend(row_warnings)
        for entry in rows:
            if entry[3] in seen_rows:
                continue                              # the same row in another log
            seen_rows.add(entry[3])
            entries.append(entry)
    real = sorted({e[0] for e in entries if _status(e[3]) != "simulated"})
    window = timedelta(seconds=config.LIVE_BOT_RUN_WINDOW_SECONDS)
    after = {t: max(real[i - 1], t - window) if i else t - window for i, t in enumerate(real)}
    trades: list[BotTrade] = []
    starts: list[RunStart] = []
    started: set[datetime] = set()
    for logged, _, trade_id, row, banner_cash in sorted(entries, key=lambda e: (e[0], e[1])):
        status = _status(row)
        if status == "simulated":
            continue
        if logged not in started and banner_cash is not None:
            starts.append(RunStart(after[logged], logged, banner_cash))
            started.add(logged)
        if status not in _BOT_STATUSES:
            if status not in _OTHER_STATUSES:
                warnings.append(f"{trade_id}: its status {status!r} is not one this page "
                                f"knows; its contracts count as {OTHER_BETS}")
            continue
        note = _NOTE.match(str(row[_COL_NOTES] or ""))
        try:
            if note is None:
                raise ValueError("no sides in its notes")
            legs = (BotLeg(str(row[_COL_TICKER_A]), note.group(2).lower(), _dec(row[_COL_COUNT_A])),
                    BotLeg(str(row[_COL_TICKER_B]), note.group(3).lower(), _dec(row[_COL_COUNT_B])))
        except ValueError:
            warnings.append(f"{trade_id}: its sides or counts cannot be read; "
                            f"its contracts count as {OTHER_BETS}")
            continue
        trades.append(BotTrade(trade_id, status, str(row[_COL_TITLE] or row[_COL_TICKER_A]),
                               legs, after[logged], logged))
    return trades, starts, warnings


# ---- which fills are the bot's ---------------------------------------------

def _order_for(orders: dict[str, list[Fill]], count: Decimal, low: datetime, high: datetime,
               claimed: dict[str, str], *, latest: bool) -> list[Fill] | None:
    """
    Find the order that made one bot leg.

    Args:
        orders (dict[str, list[Fill]]): One market and direction's orders,
            each its fills, oldest first.
        count (Decimal): The leg's contracts: the order's fills must add up
            to exactly this.
        low (datetime): Every fill must come after this.
        high (datetime): Every fill must come at or before this.
        claimed (dict[str, str]): Fills already matched to a bot purchase.
        latest (bool): Keyword-only: take the latest such order (True) or
            the earliest (False).

    Returns:
        list[Fill] | None: The order's fills, or None when no unclaimed order fits.
    """
    picks = [fills for fills in orders.values()
             if not any(f.fill_id in claimed for f in fills)
             and all(low < f.time <= high for f in fills)
             and sum((f.count for f in fills), _ZERO) == count]
    if not picks:
        return None
    pick = max if latest else min
    return pick(picks, key=lambda fills: fills[0].time)


def match_bot_fills(trades: list[BotTrade], fills: Iterable[Fill]
                    ) -> tuple[dict[str, str], list[str]]:
    """
    Match each bot purchase leg to the one Kalshi order that made it.

    A leg matches an order of exactly its market, direction and size, in
    three passes. The trade log shows its time to the whole second, rounded
    down, so a log time here means the end of that second
    (config.LIVE_TRADE_LOG_TIME_STEP_SECONDS later): a run's last orders can
    land inside it. 1: inside the run's own window (after the end of
    run_after's second, up to the end of logged_at's), the latest such
    order, since a run's orders come just before it writes the log. 2: for
    legs still unmatched, up to config.LIVE_BOT_RUN_CLOCK_SLACK_SECONDS
    further (this computer's clock running behind Kalshi's), the earliest.
    3: a row whose unwind was sent (rolled back, or unwind failed) then
    claims the bids that bought its NO leg back, from its NO purchase to the
    end of its pass-2 allowance, up to the NO leg's count; the newest row
    claims first, so a retry soon after keeps its own unwind. A rolled-back
    or failed-unwind row's YES leg never filled, so only its NO leg is
    looked for. A row on manual review sent no unwind, so pass 3 skips it.

    Args:
        trades (list[BotTrade]): The bot's purchases, from read_trade_logs.
        fills (Iterable[Fill]): Every fill in the account.

    Returns:
        tuple: The owner of each bot fill (fill_id -> trade_id; every other
            fill is Other bets), and a warning for each leg of an executed
            purchase no order explains.
    """
    step = timedelta(seconds=config.LIVE_TRADE_LOG_TIME_STEP_SECONDS)
    slack = timedelta(seconds=config.LIVE_BOT_RUN_CLOCK_SLACK_SECONDS)
    orders: dict[tuple[str, bool], dict[str, list[Fill]]] = defaultdict(lambda: defaultdict(list))
    for fill in fills:
        # A fill with no order id is an order of its own
        orders[(fill.ticker, fill.buys_yes)][fill.order_id or fill.fill_id].append(fill)
    for by_order in orders.values():
        for order in by_order.values():
            order.sort(key=lambda f: (f.time, f.fill_id))
    ordered = sorted(trades, key=lambda t: t.logged_at)
    wanted = [(t, leg) for t in ordered for leg in t.legs
              if t.status in ("executed", "manual_review") or leg.side == "no"]
    owner: dict[str, str] = {}
    no_opened: dict[str, datetime] = {}
    left: list[tuple[BotTrade, BotLeg]] = []
    warnings: list[str] = []

    def claim(trade: BotTrade, leg: BotLeg, order: list[Fill]) -> None:
        owner.update((f.fill_id, trade.trade_id) for f in order)
        if leg.side == "no":
            no_opened[trade.trade_id] = min(f.time for f in order)

    for trade, leg in wanted:            # 1: inside the run's own window, the latest order
        order = _order_for(orders[(leg.ticker, leg.side == "yes")], leg.count,
                           trade.run_after + step, trade.logged_at + step, owner, latest=True)
        if order:
            claim(trade, leg, order)
        else:
            left.append((trade, leg))
    for trade, leg in left:              # 2: just past the log time (clock skew), the earliest
        order = _order_for(orders[(leg.ticker, leg.side == "yes")], leg.count,
                           trade.logged_at + step, trade.logged_at + step + slack, owner,
                           latest=False)
        if order:
            claim(trade, leg, order)
        elif trade.status == "executed":
            warnings.append(f"The bot's purchase of {config.count_text(float(leg.count))} "
                            f"{leg.side.upper()} on {leg.ticker} matches no Kalshi order: "
                            f"those contracts count as {OTHER_BETS}")
    for trade in reversed(ordered):      # 3: unwinds, newest row first
        if trade.status not in _UNWOUND_STATUSES or trade.trade_id not in no_opened:
            continue
        no_leg = next(leg for leg in trade.legs if leg.side == "no")
        opened = no_opened[trade.trade_id]
        until = trade.logged_at + step + slack
        bought_back = _ZERO
        for order in sorted(orders[(no_leg.ticker, True)].values(), key=lambda fs: fs[0].time):
            size = sum((f.count for f in order), _ZERO)
            if (not any(f.fill_id in owner for f in order)
                    and bought_back + size <= no_leg.count
                    and all(opened <= f.time <= until for f in order)):
                owner.update((f.fill_id, trade.trade_id) for f in order)
                bought_back += size
    return owner, warnings


def manual_partners(fills: Iterable[Fill], owner_by_fill: dict[str, str]
                    ) -> dict[str, frozenset[str]]:
    """
    For each of your own fills, the other markets you sold close to it.

    Two of your own sales within config.LIVE_MANUAL_PAIR_SECONDS of each
    other on different markets are read as one pair sold by hand, which is
    how a sale of a bot pair finds the purchase it closes. A sale is a fill
    that closes at least part of what the account holds in its market (a
    bid while NO is held, an ask while YES is held), the account's holding
    being every fill before it, the bot's included. A purchase is never a
    partner.

    Args:
        fills (Iterable[Fill]): Every fill in the account.
        owner_by_fill (dict[str, str]): The bot's fills, from match_bot_fills.

    Returns:
        dict[str, frozenset[str]]: For each fill not the bot's, by fill_id,
            the other markets you sold within the gap of it.
    """
    held: dict[str, Decimal] = defaultdict(Decimal)
    mine: list[Fill] = []
    sales: list[Fill] = []
    for fill in sorted(fills, key=lambda f: (f.time, f.fill_id)):
        before = held[fill.ticker]
        held[fill.ticker] = before + (fill.count if fill.buys_yes else -fill.count)
        if fill.fill_id in owner_by_fill:
            continue
        mine.append(fill)
        if (before < 0) if fill.buys_yes else (before > 0):
            sales.append(fill)
    gap = timedelta(seconds=config.LIVE_MANUAL_PAIR_SECONDS)
    return {f.fill_id: frozenset(g.ticker for g in sales if abs(g.time - f.time) <= gap)
            - {f.ticker} for f in mine}


# ---- the ledger ------------------------------------------------------------

@dataclass(frozen=True)
class LedgerEvent:
    """
    One change to one owner's contracts in one market.

    Attributes:
        time (datetime): When it happened (UTC).
        ticker (str): The market.
        owner (str): A bot purchase's trade_id, or OTHER_BETS.
        side (str): "yes" or "no": the side of the contracts opened, closed
            or paid out.
        contracts (Decimal): Contracts opened (positive) or closed or paid
            out (negative).
        cash (Decimal): Dollars into the account (positive) or out of it
            (negative), fees included.
        spent (Decimal): Dollars spent opening contracts, fees included; 0
            for a close or a payout.
        basis (Decimal): The change in what the owner's open contracts in
            this market cost: the cost added by an open, less the share of
            the cost a close or payout takes away.
    """
    time: datetime
    ticker: str
    owner: str
    side: str
    contracts: Decimal
    cash: Decimal
    spent: Decimal
    basis: Decimal


@dataclass(frozen=True)
class Ledger:
    """
    Every fill and payout, replayed into each owner's contracts and cash.

    Attributes:
        events (tuple[LedgerEvent, ...]): Every change, oldest first.
        held_now (dict[str, Decimal]): Contracts the ledger holds now, by
            market (YES positive, NO negative); markets at 0 are left out.
        warnings (tuple[str, ...]): Payouts it could not read, or whose
            revenue disagrees with the contracts held.
    """
    events: tuple[LedgerEvent, ...]
    held_now: dict[str, Decimal]
    warnings: tuple[str, ...]


@dataclass
class _Lot:
    """
    One owner's open contracts in one market, from one opening fill.

    Attributes:
        owner (str): A bot purchase's trade_id, or OTHER_BETS.
        side (str): "yes" or "no".
        count (Decimal): Contracts still open.
        cost (Decimal): What the open contracts cost, fees included.
    """
    owner: str
    side: str
    count: Decimal
    cost: Decimal


def tickers_needing_payout(account: Account) -> set[str]:
    """
    Find the markets whose payout must be rebuilt from the market itself.

    A market qualifies when its fills leave contracts held, Kalshi holds none
    there now, and Kalshi no longer lists a settlement for it (it lists only
    recent ones).

    Args:
        account (Account): The account, from read_account.

    Returns:
        set[str]: Their tickers.
    """
    net: dict[str, Decimal] = defaultdict(Decimal)
    for fill in account.fills:
        net[fill.ticker] += fill.count if fill.buys_yes else -fill.count
    listed = {p.ticker for p in account.payouts}
    return {t for t, n in net.items() if n and t not in listed and t not in account.positions}


def all_payouts(account: Account, markets: dict[str, Market]) -> tuple[list[Payout], list[str]]:
    """
    List every payout: Kalshi's settlements, plus one rebuilt per market it no longer lists.

    A rebuilt payout takes the market's result, settlement time and YES
    value; it has no revenue to check. A market that cannot be looked up, or
    that shows no settlement with a result this can read ("yes", "no" or
    "scalar") and a settlement time, is a warning, and its contracts stay
    held in the ledger. A market that settled after the account was read is
    left as it is, with no warning: when the account was read its contracts
    were still held.

    Args:
        account (Account): The account, from read_account.
        markets (dict[str, Market]): Markets looked up, from read_markets.

    Returns:
        tuple: The payouts, and the warnings.
    """
    payouts, warnings = list(account.payouts), []
    for ticker in sorted(tickers_needing_payout(account)):
        market = markets.get(ticker)
        if market is None:
            warnings.append(f"Market {ticker} could not be looked up: "
                            f"its contracts count as still held")
        elif market.settled_at is not None and market.settled_at > account.read_at:
            continue
        elif market.settled_at is not None and market.result in ("yes", "no", "scalar"):
            payouts.append(Payout(ticker, market.settled_at, market.result, market.yes_value, None))
        else:
            warnings.append(f"Market {ticker} has no settlement this page can read: "
                            f"its contracts count as still held")
    return payouts, warnings


def _paid_per_contract(side: str, result: str, yes_value: Decimal | None) -> Decimal | None:
    """
    What one contract of a side is paid on a result.

    Args:
        side (str): "yes" or "no".
        result (str): "yes", "no" or "scalar".
        yes_value (Decimal | None): What one YES contract pays on a scalar result.

    Returns:
        Decimal | None: Dollars per contract; None for a result this cannot
            read (or a scalar result with no value).
    """
    if result in ("yes", "no"):
        return _ONE if side == result else _ZERO
    if result == "scalar" and yes_value is not None:
        return yes_value if side == "yes" else _ONE - yes_value
    return None


def _close_rank(lot: _Lot, owner: str, partners: frozenset[str],
                owner_legs: dict[str, frozenset[str]]) -> int:
    """
    Rank an open lot for a close: lower ranks close first (oldest first within a rank).

    The bot's own close (an unwind) takes its own lots first. Your own sale
    takes, first, the lots of a bot purchase whose other market you also
    sold within config.LIVE_MANUAL_PAIR_SECONDS (one pair sold by hand),
    then your own lots, then the oldest.

    Args:
        lot (_Lot): The open lot.
        owner (str): The closing fill's owner.
        partners (frozenset[str]): For your own fill, the other markets you
            sold close to it (manual_partners).
        owner_legs (dict[str, frozenset[str]]): Each bot purchase's markets, by trade_id.

    Returns:
        int: The rank.
    """
    if owner != OTHER_BETS:                                  # the bot's own close (an unwind)
        return 0 if lot.owner == owner else 1
    if owner_legs.get(lot.owner, frozenset()) & partners:    # the pair you sold by hand
        return 0
    return 1 if lot.owner == OTHER_BETS else 2


def _split(total: Decimal, parts: list[Decimal], whole: Decimal) -> list[Decimal]:
    """
    Share an amount over parts in proportion to them, so the shares add back exactly.

    Each share but the last is rounded to config.LIVE_SHARE_STEP_DOLLARS; the
    last takes what is left.

    Args:
        total (Decimal): The amount to share.
        parts (list[Decimal]): The parts' sizes.
        whole (Decimal): What the parts add up to.

    Returns:
        list[Decimal]: One share per part (empty for no parts).
    """
    if not parts:
        return []
    step = Decimal(config.LIVE_SHARE_STEP_DOLLARS)
    shares = [(total * part / whole).quantize(step) for part in parts[:-1]]
    return [*shares, total - sum(shares, _ZERO)]


def _book_fill(lots: list[_Lot], fill: Fill, owner: str, partners: frozenset[str],
               owner_legs: dict[str, frozenset[str]]) -> list[LedgerEvent]:
    """
    Book one fill: close what it can of the side held, then open the rest.

    A bid closes NO held and opens YES; an ask closes YES held and opens NO.
    A close is paid the fill's price of the side closed (a bid closing NO is
    paid its NO price), and an open costs the price of the side opened. The
    fill's fee is shared over the pieces by their contracts.

    Args:
        lots (list[_Lot]): The market's open lots (all on one side), oldest
            first; changed in place.
        fill (Fill): The fill.
        owner (str): The fill's owner: a bot purchase's trade_id or OTHER_BETS.
        partners (frozenset[str]): For your own fill, the other markets you
            sold close to it.
        owner_legs (dict[str, frozenset[str]]): Each bot purchase's markets, by trade_id.

    Returns:
        list[LedgerEvent]: One event per lot closed, then one for the contracts opened.
    """
    takes: list[tuple[_Lot, Decimal]] = []
    left = fill.count
    if lots and lots[0].side == ("no" if fill.buys_yes else "yes"):   # a bid closes NO; an ask closes YES
        for lot in sorted(lots, key=lambda lot: _close_rank(lot, owner, partners, owner_legs)):
            if not left:
                break
            take = min(left, lot.count)
            takes.append((lot, take))
            left -= take
    fees = _split(fill.fee, [take for _, take in takes] + ([left] if left else []), fill.count)
    close_price = fill.no_price if fill.buys_yes else fill.yes_price
    events = []
    for (lot, take), fee in zip(takes, fees[:len(takes)], strict=True):
        basis = (lot.cost if take == lot.count
                 else (lot.cost * take / lot.count).quantize(Decimal(config.LIVE_SHARE_STEP_DOLLARS)))
        lot.count, lot.cost = lot.count - take, lot.cost - basis
        events.append(LedgerEvent(fill.time, fill.ticker, lot.owner, lot.side, -take,
                                  take * close_price - fee, _ZERO, -basis))
    lots[:] = [lot for lot in lots if lot.count]
    if left:
        side = "yes" if fill.buys_yes else "no"
        cost = left * (fill.yes_price if fill.buys_yes else fill.no_price) + fees[-1]
        lots.append(_Lot(owner, side, left, cost))
        events.append(LedgerEvent(fill.time, fill.ticker, owner, side, left, -cost, cost, cost))
    return events


def _book_payout(lots: list[_Lot], payout: Payout) -> tuple[list[LedgerEvent], str | None]:
    """
    Book one market's payout: every open lot is paid for its side and closed.

    Args:
        lots (list[_Lot]): The market's open lots; emptied when the payout is read.
        payout (Payout): The settlement.

    Returns:
        tuple: One event per lot, and a warning (or None): when the result
            cannot be read (the lots then stay open), or when Kalshi's
            revenue differs from what the lots are paid by more than
            config.LIVE_PAYOUT_TOLERANCE_DOLLARS.
    """
    events, paid = [], _ZERO
    for lot in lots:
        per = _paid_per_contract(lot.side, payout.result, payout.yes_value)
        if per is None:
            return [], (f"{payout.ticker} settled as {payout.result!r} with no payout "
                        f"this page can read")
        paid += lot.count * per
        events.append(LedgerEvent(payout.time, payout.ticker, lot.owner, lot.side, -lot.count,
                                  lot.count * per, _ZERO, -lot.cost))
    lots.clear()
    tolerance = Decimal(config.LIVE_PAYOUT_TOLERANCE_DOLLARS)
    if payout.revenue is not None and abs(paid - payout.revenue) > tolerance:
        return events, (f"{payout.ticker} paid ${payout.revenue} but the contracts held "
                        f"add up to ${paid}")
    return events, None


def build_ledger(fills: Iterable[Fill], payouts: Iterable[Payout], owner_by_fill: dict[str, str],
                 owner_legs: dict[str, frozenset[str]]) -> Ledger:
    """
    Replay every fill and payout, oldest first, into one ledger of owners' contracts and cash.

    Fills and payouts are taken in time order, market by market; at the same
    moment a fill comes before a payout.

    Args:
        fills (Iterable[Fill]): Every fill in the account.
        payouts (Iterable[Payout]): Every payout, from all_payouts.
        owner_by_fill (dict[str, str]): The bot's fills, from match_bot_fills;
            every other fill is Other bets.
        owner_legs (dict[str, frozenset[str]]): Each bot purchase's markets, by trade_id.

    Returns:
        Ledger: The events, the contracts held now and the warnings.
    """
    fills = list(fills)
    partners = manual_partners(fills, owner_by_fill)
    lots: dict[str, list[_Lot]] = defaultdict(list)
    events: list[LedgerEvent] = []
    warnings: list[str] = []
    timeline = sorted([(f.time, 0, f.fill_id, f) for f in fills]
                      + [(p.time, 1, p.ticker, p) for p in payouts],
                      key=lambda item: item[:3])
    for _, _, _, item in timeline:
        if isinstance(item, Fill):
            events += _book_fill(lots[item.ticker], item, owner_by_fill.get(item.fill_id, OTHER_BETS),
                                 partners.get(item.fill_id, frozenset()), owner_legs)
        else:
            booked, problem = _book_payout(lots[item.ticker], item)
            events += booked
            if problem:
                warnings.append(problem)
    held: dict[str, Decimal] = {}
    for ticker, open_lots in lots.items():
        if net := sum((lot.count if lot.side == "yes" else -lot.count for lot in open_lots), _ZERO):
            held[ticker] = net
    return Ledger(tuple(events), held, tuple(warnings))
