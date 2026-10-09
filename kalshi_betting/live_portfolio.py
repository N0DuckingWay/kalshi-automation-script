"""
File: live_portfolio.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Works out how the live account is doing, for the live dashboard's Live
    trading tab. Reads the account from Kalshi with read-only GETs (every
    fill, settlement, deposit and withdrawal, then the balance and the
    positions) and reads the bot's trade log, which lists the purchases the
    bot made (and, on a run that sells, the held positions it sold first).
    Each of the bot's purchases is matched to the one Kalshi order that made
    it; every other fill in the account counts as "Other bets" (your own
    trades). Then every fill and payout is replayed, oldest first, into a
    ledger of the contracts each owner bought, sold and was paid for, with
    the cash each one moved. The bot's sale orders are among those other
    fills, and are read as a sale by hand is: a pair's two sale orders, on
    two markets within config.LIVE_MANUAL_PAIR_SECONDS of each other, close
    that pair's purchase first, so what a sale brings in goes to the
    purchase it closes; a lone held market's sale (its partner already paid
    out) closes your own contracts in that market first, if you hold any,
    and then the bot's.

    From the ledger it values the account over time: the cash, and each
    holding at its midpoint, just before the bot's first trade, at each of
    Kalshi's daily closes (midnight New York time), at each deposit or
    withdrawal, and now, by category
    (a bot purchase's category is its market A's Kalshi category; everything
    else is Other bets). From those values it works out each period's
    statistics (total return, profit, Sharpe and Sortino, each category's
    return), each bot purchase's return, the holdings now, and whether the
    cash each run logged before it traded matches the cash rebuilt from
    Kalshi's records. build_live_view puts it all together for the page, and
    append_snapshot writes one JSON line per read.

Dependencies:
    config (the LIVE_* page sizes, matching windows, read retries, candle
    limits, periods and files, SCANNER_MAX_PAGES,
    CANDLESTICK_MAX_CANDLES_PER_REQUEST, CALENDAR_DAYS_PER_YEAR, count_text);
    auth (_positions_value_cents, its reading of what the balance reply says
    the positions are worth); historical (_historical_get, the retried,
    signed, read-only GET, looked up when it is called so tests can replace
    it; _candle_close, a candle's closing price; _load_json_cache and
    _save_json_cache for the finalized markets' daily prices; series_labels
    and infer_category, the backtest page's filing rule); reporter
    (PROD_LOG_PATH, where the trade log lives, also looked up when it is
    called); dashboard (_sharpe and _sortino, the backtest page's own ratios,
    looked up when called); treasury (RiskFreeRates, the 8-week T-bill
    yields); _http (api_error_summary, one line per failed request).
    Imported by live_dashboard (the Live trading tab's server) only; nothing
    in the trading pipeline imports this module: it only reads.

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

    A holding is valued at, in order: its market's decided result; the
    midpoint of its quote (Kalshi shows an empty side as 0 for the bid or 1
    for the ask, and the midpoint counts it there, so a one-sided quote is
    halfway from its real side to that edge; a quote with both sides empty
    gives none); the last trade, when strictly between 0 and 1; the latest
    earlier daily price; and, never priced at all, what it cost.
"""
from __future__ import annotations

import bisect
import dataclasses
import json
import logging
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import openpyxl
import pandas as pd

from . import auth, config, dashboard, historical, reporter, treasury
from ._http import api_error_summary

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

# Statuses whose rows may have bought both legs: a completed purchase, and
# one left for manual review. Both their legs are looked for among the
# orders, and such a row counts as a purchase in the trade figures when both
# its orders were found.
_PAIR_STATUSES = frozenset({"executed", "manual_review"})

# The start of a trade-log row's Notes: "[time_series: YES A / NO B ..."
_NOTE = re.compile(r"^\[(\w+): (YES|NO) A / (YES|NO) B")

# The start of a sale row's Notes ("[sale: YES A / NO B, 84% of potential
# profit ..."): a held position the bot sold (reporter writes these rows when
# a run sells, under a banner of their own, before it buys anything). A sale
# is not a purchase: its orders are read as fills that close what the bot
# bought (see build_ledger), and its banner's cash is checked just before
# the run's first fill on the markets it sold (check_logged_cash).
_SALE_NOTE = "[sale: "

# The trade-log header cells this module reads (0-based column: header).
# Columns 11 and 12 (the counts) are left out: older logs name them
# differently, and their place has never changed.
_LOG_HEADERS = {0: "Date", 1: "Time", 2: "Market A", 3: "Ticker A", 5: "Ticker B",
                16: "Status", 17: "Notes"}

# Column positions in a trade-log row (0-based)
_COL_DATE, _COL_TIME, _COL_TITLE, _COL_TICKER_A, _COL_TICKER_B = 0, 1, 2, 3, 5
_COL_MARKET_B = 4
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

# Kalshi's daily candles close at midnight in this time zone
_KALSHI_DAY = ZoneInfo(config.LIVE_CANDLE_DAY_ZONE)

# A ticker that can name its own file in config.LIVE_MARKS_CACHE_DIR: letters,
# digits, ".", "_" and "-" only, never starting with "." (so never "..")
_CACHE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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
        archived (bool): True for a market found only in Kalshi's archive
            (/historical/markets), whose daily prices only the archive serves.
        subtitle (str): Its outcome label ("" when Kalshi gave none): what
            tells two markets of one question apart, such as the deadline of
            one rung of a ladder. Kalshi's subtitle, else its yes_sub_title.
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
    archived: bool = False
    subtitle: str = ""


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


def _market(row: dict, *, archived: bool) -> Market:
    """
    Read one market record from /markets or /historical/markets.

    Args:
        row (dict): The market record.
        archived (bool): Keyword-only: True when it came from /historical/markets.

    Returns:
        Market: The market; a price it cannot read is None.

    Raises:
        ValueError: If its settlement time cannot be read.
        KeyError: If it has no ticker.
    """
    settled = row.get("settlement_ts")
    # The outcome label: Kalshi's subtitle, else its yes_sub_title (the field
    # the scanner reads the same label from); anything but text reads as none
    label = row.get("subtitle") or row.get("yes_sub_title") or ""
    return Market(str(row["ticker"]), str(row.get("event_ticker") or ""),
                  str(row.get("title") or row["ticker"]), str(row.get("status") or ""),
                  str(row.get("result") or ""), _when(settled) if settled else None,
                  _opt_dec(row.get("settlement_value_dollars")),
                  _opt_dec(row.get("yes_bid_dollars")), _opt_dec(row.get("yes_ask_dollars")),
                  _opt_dec(row.get("last_price_dollars")), archived,
                  label if isinstance(label, str) else "")


def read_markets(client: Any, tickers: Iterable[str]) -> dict[str, Market]:
    """
    Look markets up: on Kalshi's live listing first, then in its archive for the rest.

    Asks for config.LIVE_MARKET_TICKERS_PER_REQUEST markets per request.

    Args:
        client (Any): A client from auth.build_client.
        tickers (Iterable[str]): The markets wanted.

    Returns:
        dict[str, Market]: Each market found, by ticker (Market.archived is
            True for one only the archive has); a market neither listing has
            is left out.

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
                market = _market(row, archived=path == "/historical/markets")
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
        sold_markets (frozenset[str]): For the banner above a run's sales,
            the markets its sale orders may have filled on (_sale_markets;
            empty for a banner above purchases): those orders went before
            anything else, so its cash is checked just before its first fill
            on them.
    """
    run_after: datetime
    logged_at: datetime
    cash_before: Decimal
    sold_markets: frozenset[str] = frozenset()


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


def _sale_markets(row: tuple) -> set[str]:
    """
    The markets a sale row's orders may have filled on.

    A held market counts when its count cell is blank (how many sold is not
    known) or above 0. A market whose count is 0 does not, nor does a
    paid-out partner (its Market cell is reporter.PAID_OUT_MARKET: no order
    goes to it), nor any market of a "not sold" row, whose orders filled
    nothing or were never sent.

    Args:
        row (tuple): A sale row's values (its Notes start with _SALE_NOTE).

    Returns:
        set[str]: The tickers.
    """
    if _status(row) == reporter._SALE_STATUS_WORDS["not_sold"]:
        return set()
    markets = set()
    for market_col, ticker_col, count_col in ((_COL_TITLE, _COL_TICKER_A, _COL_COUNT_A),
                                              (_COL_MARKET_B, _COL_TICKER_B, _COL_COUNT_B)):
        if not row[ticker_col] or row[market_col] == reporter.PAID_OUT_MARKET:
            continue
        count = row[count_col]
        try:
            if count not in (None, "") and _dec(count) <= 0:
                continue
        except ValueError:
            pass                                      # unreadable: it may have sold
        markets.add(str(row[ticker_col]))
    return markets


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
    run's rows are simulated and never bound a window, but for a dry run's
    sale row whose plan could not be used ("not sold") or whose sale raised
    ("check"): such a row bounds a window and has a RunStart, as a real
    run's does). Each real run's
    orders are looked for after the previous real run's log time (and at
    most config.LIVE_BOT_RUN_WINDOW_SECONDS before its own). A row repeated
    exactly, in the same log or another, is read once. A file or row that
    cannot be read, or a status this does not know, is a warning, never an
    exception: those trades then count as Other bets.

    A sale row (its Notes start with _SALE_NOTE) is a real run's row too,
    but never a purchase and never a warning: a run that sells writes its
    sales, under a banner of their own, before it buys, so its sales and its
    purchases are two log times, each with its own RunStart. The sales'
    RunStart names the markets their orders may have filled on
    (RunStart.sold_markets), and the purchases that follow are looked for
    after the sales' log time.

    Args:
        paths (Iterable[Path]): The trade logs, e.g. trade_log_paths().

    Returns:
        tuple: The bot's purchases (rows whose status is in _BOT_STATUSES),
            oldest first; one RunStart per real run's log time whose banner
            row gives its cash before; and the warnings.
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
    # Each real run's banner cash (from its first row that has one), and the
    # markets its sale rows name, by its log time
    cash_before: dict[datetime, Decimal] = {}
    sold_markets: dict[datetime, set[str]] = defaultdict(set)
    for logged, _, trade_id, row, banner_cash in sorted(entries, key=lambda e: (e[0], e[1])):
        status = _status(row)
        if status == "simulated":
            continue
        if logged not in cash_before and banner_cash is not None:
            cash_before[logged] = banner_cash
        if str(row[_COL_NOTES] or "").startswith(_SALE_NOTE):
            # A sale, not a purchase: the markets its orders may have filled
            # on are where the run's first orders went
            sold_markets[logged].update(_sale_markets(row))
            continue
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
    starts = [RunStart(after[logged], logged, cash, frozenset(sold_markets.get(logged, ())))
              for logged, cash in cash_before.items()]
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
              if t.status in _PAIR_STATUSES or leg.side == "no"]
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
    then your own lots, then the oldest. The bot's sale orders (a live run
    that sells held positions) are not matched to the bot, so they are
    ranked here as your own sales are.

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


# ---- prices ----------------------------------------------------------------

def day_ends(start: datetime, end: datetime) -> list[datetime]:
    """
    List each daily close strictly between two moments.

    Kalshi's daily candles close at midnight in config.LIVE_CANDLE_DAY_ZONE
    (New York): 04:00 UTC in summer, 05:00 UTC in winter. These are the
    moments, besides the start and now, the account is valued at.

    Args:
        start (datetime): An aware moment; a close exactly at it is left out.
        end (datetime): An aware moment; a close exactly at it is left out.

    Returns:
        list[datetime]: The closes, in UTC, oldest first.
    """
    ends: list[datetime] = []
    day = start.astimezone(_KALSHI_DAY).date()
    while (moment := _day_close(day)) < end:
        if moment > start:
            ends.append(moment)
        day += timedelta(days=1)
    return ends


def _day_close(day: date) -> datetime:
    """
    The moment one day begins on Kalshi's daily clock (midnight New York time).

    Args:
        day (date): The day.

    Returns:
        datetime: Midnight at the start of that day, in UTC.
    """
    return datetime.combine(day, dtime(0), _KALSHI_DAY).astimezone(UTC)


def _is_close(moment: datetime) -> bool:
    """
    Whether a moment is one of Kalshi's daily closes (midnight New York time).

    Args:
        moment (datetime): An aware moment.

    Returns:
        bool: True when it is exactly midnight in config.LIVE_CANDLE_DAY_ZONE.
    """
    return moment == _day_close(moment.astimezone(_KALSHI_DAY).date())


def _mid(bid: Decimal | None, ask: Decimal | None, last: Decimal | None) -> Decimal | None:
    """
    A YES price from a quote: its midpoint, else its last trade.

    Kalshi shows a side with no orders as 0 (bid) or 1 (ask), the edge of
    the price range, and the midpoint counts it there: a quote with no bid
    is worth half its ask, and one with no ask is halfway from its bid to 1.
    A quote with both sides empty (bid 0, ask 1), or whose bid is not below
    its ask, or that lies outside 0 to 1, gives no midpoint; the last trade
    is then used when it is strictly between 0 and 1.

    Args:
        bid (Decimal | None): The best YES bid.
        ask (Decimal | None): The best YES ask.
        last (Decimal | None): The YES price of the last trade.

    Returns:
        Decimal | None: The YES price, or None when the quote gives none.
    """
    if (bid is not None and ask is not None and _ZERO <= bid < ask <= _ONE
            and (bid > _ZERO or ask < _ONE)):
        return (bid + ask) / 2
    if last is not None and _ZERO < last < _ONE:
        return last
    return None


def _candle_side(candle: dict, key: str) -> Decimal | None:
    """
    One side of a candle at its close.

    The closing price is read by historical._candle_close, the backtest's own
    reading: close_dollars when present, else close (both dollars).

    Args:
        candle (dict): One candle as Kalshi sends it.
        key (str): "yes_bid", "yes_ask" or "price" (the trades).

    Returns:
        Decimal | None: The closing price in dollars, or None when the candle
            has none this can read.
    """
    side = candle.get(key)
    if not isinstance(side, dict):
        return None
    return _opt_dec(historical._candle_close(side))


def _candle_last(candle: dict) -> Decimal | None:
    """
    The last trade at a daily candle's close.

    On a day with trades it is the trades' close (_candle_side). On a day
    with none, Kalshi's candle carries only the last price before the day,
    which is still the last trade at the close: previous_dollars, else
    previous, read as historical._candle_close reads close_dollars, else close
    (a present but unreadable previous_dollars gives None).

    Args:
        candle (dict): One candle as Kalshi sends it.

    Returns:
        Decimal | None: The last trade's YES price in dollars, or None when
            the candle has none this can read.
    """
    close = _candle_side(candle, "price")
    if close is not None:
        return close
    price = candle.get("price")
    if not isinstance(price, dict):
        return None
    for key in ("previous_dollars", "previous"):
        raw = price.get(key)
        if raw is not None and raw != "":
            return _opt_dec(raw)
    return None


def _candle_mid(candle: dict) -> tuple[datetime, Decimal] | None:
    """
    A daily candle's YES price at its close: the midpoint, else the last trade.

    Args:
        candle (dict): One candle as Kalshi sends it.

    Returns:
        tuple[datetime, Decimal] | None: When the candle closed (UTC) and the
            YES price then; None when it has no readable close time or price.
    """
    try:
        when = _when(candle["end_period_ts"])
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    mid = _mid(_candle_side(candle, "yes_bid"), _candle_side(candle, "yes_ask"),
               _candle_last(candle))
    return None if mid is None else (when, mid)


def quote_mid(market: Market) -> Decimal | None:
    """
    A market's YES value now.

    Its payout once decided (1 or 0, or a scalar's YES value); otherwise the
    midpoint of its quote, else its last trade (_mid).

    Args:
        market (Market): The market, from read_markets.

    Returns:
        Decimal | None: What one YES contract is worth, or None when nothing
            gives a price.
    """
    if market.result in ("yes", "no"):
        return _ONE if market.result == "yes" else _ZERO
    if market.result == "scalar" and market.yes_value is not None:
        return market.yes_value
    return _mid(market.yes_bid, market.yes_ask, market.last_price)


@dataclass(frozen=True)
class Marks:
    """
    The YES value of each market the account held, over time.

    Attributes:
        daily (dict[str, tuple[tuple[datetime, Decimal], ...]]): By ticker,
            each daily close (UTC) with the YES price then, oldest first.
        now (dict[str, Decimal]): By ticker, the YES value now (quote_mid),
            for markets that have one.
        read_at (datetime): When the account was read.
    """
    daily: dict[str, tuple[tuple[datetime, Decimal], ...]]
    now: dict[str, Decimal]
    read_at: datetime
    _times: dict[str, list[datetime]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index each market's daily closes once, for at()'s lookups."""
        object.__setattr__(self, "_times", {ticker: [when for when, _ in points]
                                            for ticker, points in self.daily.items()})

    def at(self, ticker: str, moment: datetime) -> Decimal | None:
        """
        A market's YES value at a moment.

        From the moment the account was read on, the value now when there is
        one; otherwise the latest daily price at or before the moment.

        Args:
            ticker (str): The market.
            moment (datetime): An aware moment.

        Returns:
            Decimal | None: The YES value, or None when the market has none yet
                (its contracts are then valued at what they cost).
        """
        if moment >= self.read_at and ticker in self.now:
            return self.now[ticker]
        i = bisect.bisect_right(self._times.get(ticker, []), moment)
        return self.daily[ticker][i - 1][1] if i else None


def _now_marks(tickers: Iterable[str], markets: dict[str, Market]) -> dict[str, Decimal]:
    """
    The YES value now of each market that has one.

    Args:
        tickers (Iterable[str]): The markets wanted.
        markets (dict[str, Market]): Markets looked up, from read_markets.

    Returns:
        dict[str, Decimal]: By ticker, quote_mid of each market found that has a value.
    """
    now: dict[str, Decimal] = {}
    for ticker in tickers:
        market = markets.get(ticker)
        mid = quote_mid(market) if market is not None else None
        if mid is not None:
            now[ticker] = mid
    return now


def _candle_windows(start_ts: int, end_ts: int, most: int) -> list[tuple[int, int]]:
    """
    Split a span of time into windows that each hold at most `most` daily candles per market.

    A window from lo to hi holds at most _candles_in(lo, hi) candles of one
    market. Neighbouring windows share their end second, so a candle ending
    exactly there can come back twice; the caller keeps one.

    Args:
        start_ts (int): The span's start, in seconds since 1970.
        end_ts (int): The span's end, in seconds since 1970.
        most (int): The most candles of one market a window may hold.

    Returns:
        list[tuple[int, int]]: The windows (start, end), oldest first.
    """
    period = config.LIVE_CANDLE_PERIOD_MINUTES * 60
    span = max(1, most - 2) * period
    windows: list[tuple[int, int]] = []
    lo = start_ts
    while True:
        hi = min(lo + span, end_ts)
        windows.append((lo, max(lo, hi)))
        if hi >= end_ts:
            return windows
        lo = hi


def _candles_in(lo: int, hi: int) -> int:
    """
    The most daily candles of one market a request from lo to hi can return.

    One per daily close inside the window, one for the day still open at its
    end, and one more for a day a clock change shortened.

    Args:
        lo (int): The window's start, in seconds since 1970.
        hi (int): The window's end, in seconds since 1970.

    Returns:
        int: The most candles.
    """
    return (hi - lo) // (config.LIVE_CANDLE_PERIOD_MINUTES * 60) + 2


def _read_batch(client: Any, tickers: list[str], start_ts: int, end_ts: int,
                warnings: list[str]) -> tuple[dict[str, list[dict]], set[str]]:
    """
    Read daily candles for many markets through Kalshi's batch candlestick listing.

    GET /markets/candlesticks, with at most config.LIVE_CANDLE_TICKERS_PER_REQUEST
    markets per request and never more than config.LIVE_CANDLE_MAX_PER_REQUEST
    candles in one answer. A request that fails, or whose reply has no list
    of markets, is one warning, and its markets are left without daily
    prices; an entry of the list that is not an object naming one of the
    markets asked for is skipped.

    Args:
        client (Any): A client from auth.build_client.
        tickers (list[str]): The markets.
        start_ts (int): The first moment wanted, in seconds since 1970.
        end_ts (int): The last moment wanted, in seconds since 1970.
        warnings (list[str]): Gets one line per request that failed.

    Returns:
        tuple: The raw candles of each market the answers listed, by ticker;
            and the markets whose request failed.
    """
    served: dict[str, list[dict]] = defaultdict(list)
    failed: set[str] = set()
    most = config.LIVE_CANDLE_MAX_PER_REQUEST
    for lo, hi in _candle_windows(start_ts, end_ts, most):
        size = max(1, min(config.LIVE_CANDLE_TICKERS_PER_REQUEST, most // _candles_in(lo, hi)))
        for i in range(0, len(tickers), size):
            chunk = tickers[i:i + size]
            try:
                data = _get(client, "/markets/candlesticks", market_tickers=",".join(chunk),
                            start_ts=lo, end_ts=hi,
                            period_interval=config.LIVE_CANDLE_PERIOD_MINUTES)
                entries = data.get("markets")
                if not isinstance(entries, list):
                    raise ValueError("the batch candle reply came without a market list")
            except Exception as exc:  # one failed request costs its markets' daily prices only
                failed.update(chunk)
                warnings.append(f"Daily prices of {len(chunk)} market(s) could not be read "
                                f"({api_error_summary(exc)}): they are valued at their last "
                                f"known price, or at what they cost")
                continue
            wanted = set(chunk)
            for entry in entries:
                ticker = entry.get("market_ticker") if isinstance(entry, dict) else None
                if not isinstance(ticker, str) or ticker not in wanted:
                    continue
                candles = entry.get("candlesticks")
                if isinstance(candles, list):
                    served[ticker].extend(candle for candle in candles if isinstance(candle, dict))
    return dict(served), failed


def _read_archive(client: Any, ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    """
    Read one market's daily candles from Kalshi's archive.

    GET /historical/markets/{ticker}/candlesticks, in requests of at most
    config.CANDLESTICK_MAX_CANDLES_PER_REQUEST candles each.

    Args:
        client (Any): A client from auth.build_client.
        ticker (str): The market.
        start_ts (int): The first moment wanted, in seconds since 1970.
        end_ts (int): The last moment wanted, in seconds since 1970.

    Returns:
        list[dict]: The raw candles.

    Raises:
        ValueError: If an answer has no candle list.
        Exception: Whatever the request raises (an ApiException after the retries).
    """
    candles: list[dict] = []
    for lo, hi in _candle_windows(start_ts, end_ts, config.CANDLESTICK_MAX_CANDLES_PER_REQUEST):
        data = _get(client, f"/historical/markets/{ticker}/candlesticks", start_ts=lo, end_ts=hi,
                    period_interval=config.LIVE_CANDLE_PERIOD_MINUTES)
        rows = data.get("candlesticks")
        if not isinstance(rows, list):
            raise ValueError(f"the archive's candles for {ticker} came without a candle list")
        candles.extend(row for row in rows if isinstance(row, dict))
    return candles


def _marks_file(ticker: str) -> Path | None:
    """
    The file a finalized market's daily candles are kept in.

    config.LIVE_MARKS_CACHE_DIR is read when this is called, so tests can
    point it elsewhere.

    Args:
        ticker (str): The market.

    Returns:
        Path | None: LIVE_MARKS_CACHE_DIR/<ticker>.json, or None for a ticker
            that cannot safely name a file.
    """
    if not _CACHE_NAME.fullmatch(ticker):
        return None
    return Path(config.LIVE_MARKS_CACHE_DIR) / f"{ticker}.json"


def _finalized(market: Market | None) -> bool:
    """
    Whether a market's prices can no longer change (Kalshi has finalized it).

    Args:
        market (Market | None): The market, or None when it was not found.

    Returns:
        bool: True for a finalized market.
    """
    return market is not None and market.status == "finalized"


def _kept_candles(ticker: str, start_ts: int) -> list[dict] | None:
    """
    A finalized market's daily candles from its file, when they go back far enough.

    Args:
        ticker (str): The market.
        start_ts (int): The first moment wanted, in seconds since 1970.

    Returns:
        list[dict] | None: The candles, or None when there is no usable file
            or it starts later than `start_ts`.
    """
    path = _marks_file(ticker)
    if path is None:
        return None
    # Cross-module: the backtest's own small-cache reader; a damaged file reads as no file
    data = historical._load_json_cache(path)
    if (not isinstance(data, dict) or data.get("ticker") != ticker
            or type(data.get("start_ts")) is not int or data["start_ts"] > start_ts
            or not isinstance(data.get("candles"), list)):
        return None
    return [candle for candle in data["candles"] if isinstance(candle, dict)]


def _keep_candles(ticker: str, start_ts: int, end_ts: int, candles: list[dict]) -> None:
    """
    Write a finalized market's daily candles to its file, for later reads.

    A file that cannot be written is logged and otherwise ignored: the
    market is read from Kalshi again next time.

    Args:
        ticker (str): The market.
        start_ts (int): The first moment the candles cover, in seconds since 1970.
        end_ts (int): The last moment, in seconds since 1970.
        candles (list[dict]): The raw candles.
    """
    path = _marks_file(ticker)
    if path is None:
        return
    try:
        # Cross-module: the backtest's own atomic writer (a temporary file, then a rename)
        historical._save_json_cache(path, {"ticker": ticker, "start_ts": start_ts,
                                           "end_ts": end_ts, "candles": candles})
    except (OSError, TypeError, ValueError) as exc:
        logging.warning("Could not keep the daily prices of %s in %s (%s)", ticker, path,
                        api_error_summary(exc))


def _daily_points(candles: Iterable[dict]) -> tuple[tuple[datetime, Decimal], ...]:
    """
    Turn raw daily candles into (close time, YES price) points.

    Args:
        candles (Iterable[dict]): Raw candles, in any order, possibly repeated.

    Returns:
        tuple: One point per close time (the last candle read wins), oldest
            first; candles with no price are left out.
    """
    points: dict[datetime, Decimal] = {}
    for candle in candles:
        point = _candle_mid(candle)
        if point is not None:
            points[point[0]] = point[1]
    return tuple(sorted(points.items()))


def read_marks(client: Any, tickers: Iterable[str], start: datetime, markets: dict[str, Market],
               read_at: datetime) -> tuple[Marks, list[str]]:
    """
    Read the daily prices, and the prices now, of the markets the account held.

    The daily candles run from the day before `start` to the read. A
    finalized market whose file in config.LIVE_MARKS_CACHE_DIR goes back far
    enough is read from it, with no request. A market read_markets found only
    in Kalshi's archive (Market.archived) is read from the archive's own
    candle listing, one market at a time; the other markets read_markets
    found are asked of Kalshi's batch listing (GET /markets/candlesticks),
    and one that comes back with no candles there and has settled is then
    read from the archive too. A market read from the archive is kept in its
    file when it is finalized; a market that is still open is never kept.
    A market read_markets did not find is never asked for: neither listing
    has it, so it has no candles to read. A request that fails is a warning
    (one line, through _http.api_error_summary), never an error: those
    markets are valued at their last known price, or at what they cost. So
    a failed batch request never costs the markets only the archive has.

    Args:
        client (Any): A client from auth.build_client.
        tickers (Iterable[str]): The markets.
        start (datetime): The first moment the account is valued at.
        markets (dict[str, Market]): Markets looked up, from read_markets.
        read_at (datetime): When the account was read.

    Returns:
        tuple: The Marks, and the warnings.
    """
    wanted = sorted(set(tickers))
    now = _now_marks(wanted, markets)
    if not wanted:
        return Marks({}, now, read_at), []
    day_before = start.astimezone(_KALSHI_DAY).date() - timedelta(days=1)
    start_ts = int(_day_close(day_before).timestamp())
    end_ts = max(start_ts, math.ceil(read_at.timestamp()))
    warnings: list[str] = []
    candles: dict[str, list[dict]] = {}
    batch: list[str] = []
    archive: list[str] = []
    for ticker in wanted:
        market = markets.get(ticker)
        if market is None:
            continue                     # found in neither listing: no candles to read
        kept = _kept_candles(ticker, start_ts) if _finalized(market) else None
        if kept is not None:
            candles[ticker] = kept
        elif market.archived:
            archive.append(ticker)
        else:
            batch.append(ticker)
    served, failed = _read_batch(client, batch, start_ts, end_ts, warnings)
    candles.update(served)
    # A settled market the batch gave no candles is looked for in the archive;
    # one whose batch request failed, or that is still open, is not
    archive += [ticker for ticker in batch if ticker not in failed and not served.get(ticker)
                and markets[ticker].settled_at is not None]
    for ticker in sorted(archive):
        market = markets[ticker]
        try:
            archived = _read_archive(client, ticker, start_ts, end_ts)
        except Exception as exc:  # one market's failure costs that market's daily prices only
            warnings.append(f"Daily prices of {ticker} could not be read "
                            f"({api_error_summary(exc)}): it is valued at its last known "
                            f"price, or at what it cost")
            continue
        candles[ticker] = archived
        if _finalized(market):
            _keep_candles(ticker, start_ts, end_ts, archived)
    daily = {ticker: points for ticker, rows in candles.items() if (points := _daily_points(rows))}
    return Marks(daily, now, read_at), warnings


def _value(side: str, count: Decimal, basis: Decimal, mark: Decimal | None) -> Decimal:
    """
    What some contracts are worth at a YES value.

    Args:
        side (str): "yes" or "no".
        count (Decimal): Contracts held.
        basis (Decimal): What they cost.
        mark (Decimal | None): The YES value, or None when the market has never been priced.

    Returns:
        Decimal: count times the side's value (1 less the YES value for NO),
            or what they cost when there is no YES value.
    """
    if mark is None:
        return basis
    return count * (mark if side == "yes" else _ONE - mark)


# ---- the account over time -------------------------------------------------

@dataclass(frozen=True)
class History:
    """
    The account's value at each moment it is valued at, by group.

    A group is a Kalshi category (for the bot's purchases) or OTHER_BETS.
    Every series has one value per moment in `times`.

    Attributes:
        times (tuple[datetime, ...]): Just before the bot's first fill, each
            daily close since, each deposit or withdrawal since (the moment it
            landed, with it counted), and the read (UTC), oldest first.
        cash (tuple[float, ...]): The cash.
        value (dict[str, tuple[float, ...]]): Each group's holdings' value.
        net_cash (dict[str, tuple[float, ...]]): Each group's cash in, less
            cash out, since the start (fees included).
        spent (dict[str, tuple[float, ...]]): Each group's dollars spent
            opening contracts since the start (fees included).
        steps (dict[str, tuple[tuple[datetime, float], ...]]): Each group's
            net cash after each of its changes since the start.
        flows (tuple[float, ...]): Deposits less withdrawals since the
            previous moment (0 at the first). Each lands exactly at its own
            moment, so none earns or loses anything before the next moment.
    """
    times: tuple[datetime, ...]
    cash: tuple[float, ...]
    value: dict[str, tuple[float, ...]]
    net_cash: dict[str, tuple[float, ...]]
    spent: dict[str, tuple[float, ...]]
    steps: dict[str, tuple[tuple[datetime, float], ...]]
    flows: tuple[float, ...]

    def total(self, i: int) -> float:
        """
        The account's whole value at one moment: the cash and every holding.

        Args:
            i (int): The moment's place in `times`.

        Returns:
            float: The value in dollars.
        """
        return self.cash[i] + sum(series[i] for series in self.value.values())


def _cash_reader(account: Account, ledger: Ledger) -> Callable[[datetime], Decimal]:
    """
    Build the reader of the cash at any moment: the cash now, less every change after it.

    The changes are the ledger's cash (fills and payouts) and the deposits
    and withdrawals.

    Args:
        account (Account): The account, from read_account.
        ledger (Ledger): The ledger, from build_ledger.

    Returns:
        Callable[[datetime], Decimal]: The cash at a moment, in dollars.
    """
    changes = sorted([(e.time, e.cash) for e in ledger.events]
                     + [(f.time, f.amount) for f in account.flows], key=lambda change: change[0])
    times = [when for when, _ in changes]
    after = [_ZERO] * (len(changes) + 1)          # after[i]: changes i, i+1, ... added up
    for i in range(len(changes) - 1, -1, -1):
        after[i] = after[i + 1] + changes[i][1]

    def cash(moment: datetime) -> Decimal:
        """The cash at `moment`: the cash now, less every change after it."""
        return account.cash - after[bisect.bisect_right(times, moment)]

    return cash


def cash_at(account: Account, ledger: Ledger, moment: datetime) -> Decimal:
    """
    The cash at a moment: the cash now, less every cash change after it.

    Args:
        account (Account): The account, from read_account.
        ledger (Ledger): The ledger, from build_ledger.
        moment (datetime): An aware moment.

    Returns:
        Decimal: The cash then, in dollars.
    """
    return _cash_reader(account, ledger)(moment)


def build_history(account: Account, ledger: Ledger, marks: Marks,
                  group_of: Callable[[str], str], start: datetime) -> History:
    """
    Value the account at the start, at each daily close since and now, group by group.

    It is also valued at the moment of each deposit or withdrawal since the
    start (with it counted), so the money moved in or out is in the value
    from that moment on and the time-weighted return never credits its gain
    or loss to the money already there. Each holding is valued at its
    market's YES value then (Marks.at), or at what it cost when its market
    has never been priced. A group's net cash and spending count only what
    happened after `start`.

    Args:
        account (Account): The account, from read_account.
        ledger (Ledger): The ledger, from build_ledger.
        marks (Marks): The markets' values over time, from read_marks.
        group_of (Callable[[str], str]): A ledger owner's group.
        start (datetime): Just before the bot's first fill.

    Returns:
        History: The values; a group whose value, net cash and spending are
            0 at every moment is left out.
    """
    landed = {f.time for f in account.flows if start < f.time < account.read_at}
    times = tuple(sorted({start, *day_ends(start, account.read_at), *landed, account.read_at}))
    cash_then = _cash_reader(account, ledger)
    held: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    basis: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    net: dict[str, Decimal] = defaultdict(Decimal)
    spent: dict[str, Decimal] = defaultdict(Decimal)
    columns: dict[str, dict[str, list[float]]] = {name: defaultdict(list)
                                                  for name in ("value", "net", "spent")}
    steps: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    groups = sorted({group_of(e.owner) for e in ledger.events})
    events, i = ledger.events, 0
    for moment in times:
        while i < len(events) and events[i].time <= moment:
            event = events[i]
            group, key = group_of(event.owner), (event.ticker, event.owner, event.side)
            held[key] += event.contracts
            basis[key] += event.basis
            if event.time > start:                   # only what happened since the start
                net[group] += event.cash
                spent[group] += event.spent
                steps[group].append((event.time, float(net[group])))
            i += 1
        worth: dict[str, Decimal] = defaultdict(Decimal)
        for (ticker, owner, side), count in held.items():
            if count:
                worth[group_of(owner)] += _value(side, count, basis[(ticker, owner, side)],
                                                 marks.at(ticker, moment))
        for group in groups:
            columns["value"][group].append(float(worth[group]))
            columns["net"][group].append(float(net[group]))
            columns["spent"][group].append(float(spent[group]))
    flows = [0.0] + [float(sum((f.amount for f in account.flows if lo < f.time <= hi), _ZERO))
                     for lo, hi in pairwise(times)]
    keep = [group for group in groups
            if any(columns["value"][group]) or any(columns["net"][group])
            or any(columns["spent"][group])]
    return History(times, tuple(float(cash_then(moment)) for moment in times),
                   {g: tuple(columns["value"][g]) for g in keep},
                   {g: tuple(columns["net"][g]) for g in keep},
                   {g: tuple(columns["spent"][g]) for g in keep},
                   {g: tuple(steps[g]) for g in keep}, tuple(flows))


# ---- statistics ------------------------------------------------------------

@dataclass(frozen=True)
class TradeReturn:
    """
    How one bot purchase has done so far.

    Attributes:
        trade_id (str): The purchase (BotTrade.trade_id).
        opened (datetime): Its first fill (UTC).
        pairs (Decimal): Contract pairs it bought.
        ret (float): Its return so far: cash back plus value now, less what it
            spent, over what it spent (fees, sales and payouts included).
        open_pairs (Decimal): Pairs still held: the fewer of its contracts
            still held on its two markets.
        still_open (bool): True while any of its contracts is still held.
    """
    trade_id: str
    opened: datetime
    pairs: Decimal
    ret: float
    open_pairs: Decimal
    still_open: bool


def trade_returns(trades: list[BotTrade], ledger: Ledger, marks: Marks) -> list[TradeReturn]:
    """
    The return so far of each bot purchase that bought a contract pair.

    That is a row whose status is "executed" or "manual_review"
    (_PAIR_STATUSES) and whose two orders were both found: a manual-review
    row whose two orders filled holds a pair like any other. A purchase's
    contracts still held are valued at their market's value now (Marks.at
    at the read), or at what they cost when never priced. A sale of its
    contracts, by the bot or by you, counts as cash it got back. Rolled-back
    and failed-unwind rows are left out (their YES leg never filled, so they
    hold no pair, though their category still counts them), and so is any
    purchase only one of whose orders was found.

    Args:
        trades (list[BotTrade]): The bot's purchases, from read_trade_logs.
        ledger (Ledger): The ledger, from build_ledger.
        marks (Marks): The markets' values, from read_marks.

    Returns:
        list[TradeReturn]: One per such purchase, in the trade log's order.
    """
    cash: dict[str, Decimal] = defaultdict(Decimal)
    spent: dict[str, Decimal] = defaultdict(Decimal)
    opened: dict[str, datetime] = {}
    markets: dict[str, set[str]] = defaultdict(set)
    held: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    basis: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    for e in ledger.events:
        cash[e.owner] += e.cash
        spent[e.owner] += e.spent
        if e.spent:
            opened[e.owner] = min(opened.get(e.owner, e.time), e.time)
            markets[e.owner].add(e.ticker)
        held[(e.ticker, e.owner, e.side)] += e.contracts
        basis[(e.ticker, e.owner, e.side)] += e.basis
    value: dict[str, Decimal] = defaultdict(Decimal)
    per_market: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for (ticker, owner, side), count in held.items():
        if count:
            value[owner] += _value(side, count, basis[(ticker, owner, side)],
                                   marks.at(ticker, marks.read_at))
            per_market[(owner, ticker)] += count
    out: list[TradeReturn] = []
    for trade in trades:
        legs = {leg.ticker for leg in trade.legs}
        if (trade.status not in _PAIR_STATUSES or markets[trade.trade_id] != legs
                or not spent[trade.trade_id]):
            continue
        a, b = (per_market[(trade.trade_id, leg.ticker)] for leg in trade.legs)
        out.append(TradeReturn(
            trade.trade_id, opened[trade.trade_id], trade.legs[0].count,
            float((cash[trade.trade_id] + value[trade.trade_id]) / spent[trade.trade_id]),
            min(a, b), bool(a or b)))
    return out


def pair_weighted(returns: list[TradeReturn]) -> tuple[float | None, float | None]:
    """
    The mean and median return over contract pairs: a purchase of N pairs counts N times.

    The median is the middle pair's return, or the average of the two middle
    pairs' returns when the pairs are an even number (as numpy's median of
    the list with each return repeated once per pair).

    Args:
        returns (list[TradeReturn]): The purchases.

    Returns:
        tuple[float | None, float | None]: The mean and the median; None and
            None when there is no pair.
    """
    weights = [float(r.pairs) for r in returns]
    total = sum(weights)
    if not returns or total <= 0:
        return None, None
    mean = sum(w * r.ret for w, r in zip(weights, returns, strict=True)) / total
    ordered = sorted(zip((r.ret for r in returns), weights, strict=True))
    run = 0.0
    for i, (ret, weight) in enumerate(ordered):
        run += weight
        if run > total / 2:
            return mean, ret
        if run == total / 2:
            return mean, (ret + ordered[i + 1][0]) / 2
    return mean, ordered[-1][0]


@dataclass(frozen=True)
class GroupReturn:
    """
    How one group (a category, or Other bets) did over a period.

    Attributes:
        group (str): The group.
        pnl (float): Its profit: value plus net cash at the end, less the same
            at the period's start.
        put_in (float): Its value at the start, plus the most cash it had out
            at any moment since, so money reinvested is not counted twice.
        ret (float | None): pnl over put_in; None when nothing was put in.
    """
    group: str
    pnl: float
    put_in: float
    ret: float | None


@dataclass(frozen=True)
class PeriodStats:
    """
    The Live trading tab's figures for one period.

    Attributes:
        label (str): The period, e.g. "All" or "1M".
        first (datetime): Its first moment (UTC): the start, or the first
            daily close in the period.
        last (datetime): Its last moment: the read.
        total_return (float | None): The return over the period with
            deposits and withdrawals taken out (each moment-to-moment return
            chained); None for a period with no time in it.
        pnl (float): The value's change, less deposits plus withdrawals.
        sharpe (float | None): dashboard._sharpe over its whole days' returns;
            None with fewer than config.LIVE_RATIO_MIN_WHOLE_DAYS whole days.
        sortino (float | None): dashboard._sortino, likewise.
        whole_days (int): How many whole days (one daily close to the next)
            the two ratios used.
        mean_trade (float | None): The pair-weighted mean return of the bot
            purchases made in the period (pair_weighted).
        median_trade (float | None): Their pair-weighted median return.
        pairs (Decimal): Contract pairs those purchases bought.
        purchases (int): How many purchases.
        open_pairs (Decimal): Their pairs still held.
        groups (tuple[GroupReturn, ...]): Each group's return over the period.
    """
    label: str
    first: datetime
    last: datetime
    total_return: float | None
    pnl: float
    sharpe: float | None
    sortino: float | None
    whole_days: int
    mean_trade: float | None
    median_trade: float | None
    pairs: Decimal
    purchases: int
    open_pairs: Decimal
    groups: tuple[GroupReturn, ...]


def _group_returns(history: History, a: int) -> tuple[GroupReturn, ...]:
    """
    Each group's return from moment `a` of the history to the read.

    Profit is the group's value plus its net cash at the end, less the same
    at moment `a`. What it put in is its value at `a` plus the most cash it
    had out at any moment since (its net cash at `a` less its lowest net cash
    after), so a payout put back to work is not counted twice: spending $10,
    getting $12 back, spending the $12 and getting $14.40 back put in $10.

    Args:
        history (History): The history, from build_history.
        a (int): The period's first moment's place in history.times.

    Returns:
        tuple[GroupReturn, ...]: One per group, in history.value's order.
    """
    times, end = history.times, len(history.times) - 1
    out: list[GroupReturn] = []
    for group in history.value:
        start_net = history.net_cash[group][a]
        gain = ((history.value[group][end] + history.net_cash[group][end])
                - (history.value[group][a] + start_net))
        out_most = max([0.0] + [start_net - n for when, n in history.steps[group]
                                if times[a] < when <= times[end]])
        put_in = history.value[group][a] + out_most
        out.append(GroupReturn(group, gain, put_in, gain / put_in if put_in > 0 else None))
    return tuple(out)


def _months_before(moment: datetime, months: int) -> datetime:
    """
    The moment some months before another, on New York's calendar.

    The daily closes are midnights in config.LIVE_CANDLE_DAY_ZONE, so a
    month back is counted on that zone's wall clock (pandas' DateOffset, so
    March 31 less one month is February 28 or 29): between 8 pm and midnight
    New York time the UTC date is already the next day's. A wall time a
    clock change skips is read as zoneinfo reads it.

    Args:
        moment (datetime): An aware moment.
        months (int): How many months back.

    Returns:
        datetime: The moment that many months earlier, in UTC.
    """
    local = moment.astimezone(_KALSHI_DAY).replace(tzinfo=None)
    earlier = (pd.Timestamp(local) - pd.DateOffset(months=months)).to_pydatetime()
    return earlier.replace(tzinfo=_KALSHI_DAY).astimezone(UTC)


def period_stats(history: History, returns: list[TradeReturn],
                 risk_free: treasury.RiskFreeRates | None, label: str, months: int) -> PeriodStats:
    """
    Work out one period's figures.

    The period runs from the first daily close at or after `months` months
    before the read, counted on New York's calendar as the daily closes are
    (the history's start for 0, or when the history is shorter), to the
    read. The total return chains each moment-to-moment return, taking out
    the deposits and withdrawals: each lands at a moment of its own
    (build_history), so it is taken out at the end of the step it lands in.
    Sharpe and Sortino are the backtest page's own (dashboard._sharpe and
    _sortino, at config.CALENDAR_DAYS_PER_YEAR periods a year), over whole
    days only: a day runs from one daily close to the next, its return
    chained over the moments inside it, and the part-days before the
    period's first close and after its last are left out; with fewer than
    config.LIVE_RATIO_MIN_WHOLE_DAYS whole days there are none. They subtract the
    8-week T-bill yield on the share of the value held in positions at the
    start of each day (the backtest page's rule); with no yields, 0%. The
    trade figures cover the bot purchases first filled in the period.

    Args:
        history (History): The history, from build_history.
        returns (list[TradeReturn]): Every purchase's return, from trade_returns.
        risk_free (treasury.RiskFreeRates | None): The T-bill yields, or None.
        label (str): The period's label.
        months (int): How many months back the period reaches; 0 for all of it.

    Returns:
        PeriodStats: The figures.
    """
    times, now = history.times, history.times[-1]
    closes = [_is_close(moment) for moment in times]
    a, end = 0, len(times) - 1
    if months and times[0] < (cut := _months_before(now, months)):
        after_cut = [i for i, moment in enumerate(times) if moment >= cut]   # the read at least
        a = next((i for i in after_cut if closes[i]), after_cut[0])
    growth, pnl = 1.0, 0.0
    daily: list[float] = []
    day_starts: list[date] = []
    deployed: list[float] = []
    day: tuple[int, float] | None = None       # (the day's opening close, its growth so far)
    for k in range(a + 1, len(times)):
        before, after = history.total(k - 1), history.total(k) - history.flows[k]
        pnl += after - before
        step = after / before if before > 0 else None
        if step is not None:
            growth *= step
        if closes[k - 1]:
            day = (k - 1, 1.0)                  # a daily close opens a day
        if day is not None:
            day = None if step is None else (day[0], day[1] * step)
        if closes[k] and day is not None:       # the next close ends it: a whole day
            opened = day[0]
            daily.append(day[1] - 1)
            day_starts.append(times[opened].astimezone(_KALSHI_DAY).date())
            deployed.append(1 - history.cash[opened] / history.total(opened))
            day = None
    sharpe = sortino = None
    if len(daily) >= config.LIVE_RATIO_MIN_WHOLE_DAYS:
        rf: float | np.ndarray = (0.0 if risk_free is None
                                  else risk_free.annual_on(day_starts) * np.array(deployed))
        series = pd.Series(daily)
        # Cross-module: the backtest page's own ratios, looked up when called
        sharpe = dashboard._sharpe(series, rf, periods_per_year=config.CALENDAR_DAYS_PER_YEAR)
        sortino = dashboard._sortino(series, rf, periods_per_year=config.CALENDAR_DAYS_PER_YEAR)
    inside = [r for r in returns if r.opened >= times[a]]
    mean, median = pair_weighted(inside)
    return PeriodStats(label, times[a], now, growth - 1 if a < end else None, pnl, sharpe,
                       sortino, len(daily), mean, median, sum((r.pairs for r in inside), _ZERO),
                       len(inside), sum((r.open_pairs for r in inside), _ZERO),
                       _group_returns(history, a))


# ---- the cash each run logged ----------------------------------------------

@dataclass(frozen=True)
class CashCheck:
    """
    Whether the cash each real run logged before trading matches Kalshi's records.

    Attributes:
        matched (int): Runs whose logged cash matches.
        checked (int): Runs checked.
        worst (Decimal): The largest difference found, in dollars (0 with none).
        misses (tuple[tuple[datetime, Decimal, Decimal], ...]): Each run that
            does not match: its log time (UTC), the cash it logged, and the
            cash rebuilt then.
    """
    matched: int
    checked: int
    worst: Decimal
    misses: tuple[tuple[datetime, Decimal, Decimal], ...]


def check_logged_cash(account: Account, ledger: Ledger, runs: list[RunStart],
                      first_bot_fill: dict[datetime, datetime], *,
                      bot_fill_ids: frozenset[str] = frozenset()) -> CashCheck:
    """
    Check each real run's logged "Balance before" against the cash rebuilt from Kalshi's records.

    Every real run is checked, those before the bot's first fill included:
    the cash is rebuilt (cash_at, which holds at any moment) just before the
    run's first bot fill, or at its log time when it had none. For the
    banner above a run's sales (RunStart.sold_markets), that first fill is
    the earlier of its first bot fill and its first fill on the markets it
    sold, read in the window match_bot_fills reads a run's purchases in:
    after the end of the previous real run's logged second and up to the end
    of its own (the log keeps whole seconds, so a fill inside a run's logged
    second is that run's). Its sale orders count as Other bets fills, so
    first_bot_fill holds none of them, and its banner's cash is the cash
    before those orders; a fill the bot's purchases own (bot_fill_ids) is
    never taken for a sale. The logged
    figure rounds each shard's cash down to the cent, so it matches when the
    rebuilt cash is at most config.LIVE_CASH_CHECK_BELOW_DOLLARS below it and
    less than config.LIVE_CASH_CHECK_ABOVE_PER_SHARD_DOLLARS per shard above
    it. Which shards held cash at each run is not known, so every shard the
    balance reply lists now is counted, empty ones included: with four
    shards listed, a rebuilt cash just under 4 cents above the logged figure
    is a match even when only one shard held cash then. The largest gap
    found (worst) is reported beside the count, so such a gap still shows.

    Args:
        account (Account): The account, from read_account.
        ledger (Ledger): The ledger, from build_ledger.
        runs (list[RunStart]): The real runs, from read_trade_logs.
        first_bot_fill (dict[datetime, datetime]): Each run's first bot fill,
            by the run's log time.
        bot_fill_ids (frozenset[str]): Keyword-only. The fills the bot's
            purchases own (match_bot_fills), never read as a sale.

    Returns:
        CashCheck: The result.
    """
    cash_then = _cash_reader(account, ledger)
    step = timedelta(seconds=config.LIVE_TRADE_LOG_TIME_STEP_SECONDS)
    low = -Decimal(config.LIVE_CASH_CHECK_BELOW_DOLLARS)
    high = Decimal(config.LIVE_CASH_CHECK_ABOVE_PER_SHARD_DOLLARS) * max(1, account.shards)
    matched = checked = 0
    worst = _ZERO
    misses: list[tuple[datetime, Decimal, Decimal]] = []
    for run in runs:
        first = first_bot_fill.get(run.logged_at)
        if run.sold_markets:
            sale = min((f.time for f in account.fills if f.ticker in run.sold_markets
                        and f.fill_id not in bot_fill_ids
                        and run.run_after + step < f.time <= run.logged_at + step),
                       default=None)
            if sale is not None and (first is None or sale < first):
                first = sale
        rebuilt = cash_then(first - timedelta(microseconds=1) if first else run.logged_at)
        gap = rebuilt - run.cash_before
        checked += 1
        worst = max(worst, abs(gap))
        if low <= gap < high:
            matched += 1
        else:
            misses.append((run.logged_at, run.cash_before, rebuilt))
    return CashCheck(matched, checked, worst, tuple(misses))


def _cash_miss_warnings(check: CashCheck) -> list[str]:
    """
    One sentence per run whose logged cash does not match, for the page to show as it is.

    The logged cash is shown as the trade log wrote it; the rebuilt cash is
    rounded to config.LIVE_CASH_SHOWN_STEP_DOLLARS (the hundredth of a cent
    the account keeps its cash to), since it can carry the ledger's finer
    shares of a fee or a cost.

    Args:
        check (CashCheck): The check, from check_logged_cash.

    Returns:
        list[str]: The sentences, in the order of the runs.
    """
    step = Decimal(config.LIVE_CASH_SHOWN_STEP_DOLLARS)
    return [f"The run logged at {logged:%Y-%m-%d %H:%M} UTC wrote ${banner} as its cash before "
            f"trading; Kalshi's records rebuild ${rebuilt.quantize(step)} then"
            for logged, banner, rebuilt in check.misses]


# ---- what the page shows ---------------------------------------------------

@dataclass(frozen=True)
class Holding:
    """
    Contracts of one side of one market held now, by one group.

    Attributes:
        ticker (str): The market.
        title (str): Its title (the ticker when it was not found).
        group (str): A Kalshi category, or OTHER_BETS.
        side (str): "yes" or "no".
        contracts (Decimal): Contracts held.
        price (Decimal | None): What one contract of the side held is worth
            now; None when its market has never been priced.
        value (Decimal): What they are worth now (at what they cost when
            never priced).
        cost (Decimal): What they cost, fees included.
        subtitle (str): The market's outcome label (Market.subtitle; "" when
            it has none or was not found).
        event_ticker (str): The market's event (_event_of), so the markets
            of one question can be listed together.
    """
    ticker: str
    title: str
    group: str
    side: str
    contracts: Decimal
    price: Decimal | None
    value: Decimal
    cost: Decimal
    subtitle: str = ""
    event_ticker: str = ""


@dataclass(frozen=True)
class LiveView:
    """
    Everything the Live trading tab shows, from one read of the account.

    Attributes:
        read_at (datetime): When the account was read (UTC).
        cash (Decimal): The cash on every shard.
        kalshi_positions_value (Decimal | None): What Kalshi says the
            positions are worth; None when its reply had no usable value.
        holdings (tuple[Holding, ...]): What is held now, in group_order,
            then by event and market, so the markets of one question (the
            rungs of a ladder, both legs of a pair on one event) sit together.
        group_order (tuple[str, ...]): The groups, each category in the order
            it first appeared (its first bot purchase), ties by name, with
            OTHER_BETS last. A new category always comes after the ones
            before it, so each keeps its place (and its color on the page)
            from one read to the next, while the matched purchases and
            their categories stay the same (_group_order).
        history (History | None): The account over time; None before the
            bot's first live trade.
        periods (tuple[PeriodStats, ...] | None): One per
            config.LIVE_DASHBOARD_PERIODS; None before the first trade.
        trades (tuple[TradeReturn, ...]): Each bot purchase's return (trade_returns).
        cash_check (CashCheck): The runs' logged cash against Kalshi's records.
        risk_free (treasury.RiskFreeRates | None): The T-bill yields the
            ratios used, or None (0%).
        changing (bool): True when the account kept changing while it was read.
        warnings (tuple[str, ...]): Every warning of the read and of each step
            after it, for the page to show.
    """
    read_at: datetime
    cash: Decimal
    kalshi_positions_value: Decimal | None
    holdings: tuple[Holding, ...]
    group_order: tuple[str, ...]
    history: History | None
    periods: tuple[PeriodStats, ...] | None
    trades: tuple[TradeReturn, ...]
    cash_check: CashCheck
    risk_free: treasury.RiskFreeRates | None
    changing: bool
    warnings: tuple[str, ...]

    @property
    def holdings_value(self) -> Decimal:
        """What every holding is worth now, added up."""
        return sum((holding.value for holding in self.holdings), _ZERO)


def _holdings(ledger: Ledger, marks: Marks, markets: dict[str, Market],
              group_of: Callable[[str], str]) -> list[Holding]:
    """
    What is held now, one Holding per market, side and group.

    Args:
        ledger (Ledger): The ledger, from build_ledger.
        marks (Marks): The markets' values, from read_marks.
        markets (dict[str, Market]): Markets looked up, for their titles.
        group_of (Callable[[str], str]): A ledger owner's group.

    Returns:
        list[Holding]: The holdings, in no particular order.
    """
    held: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    basis: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    for e in ledger.events:
        key = (e.ticker, group_of(e.owner), e.side)
        held[key] += e.contracts
        basis[key] += e.basis
    out: list[Holding] = []
    for (ticker, group, side), count in held.items():
        if not count:
            continue
        mark = marks.at(ticker, marks.read_at)
        market = markets.get(ticker)
        out.append(Holding(ticker, market.title if market else ticker, group, side, count,
                           None if mark is None else (mark if side == "yes" else _ONE - mark),
                           _value(side, count, basis[(ticker, group, side)], mark),
                           basis[(ticker, group, side)], market.subtitle if market else "",
                           _event_of(ticker, markets)))
    return out


def _group_order(history: History | None, holdings: list[Holding]) -> tuple[str, ...]:
    """
    Order the groups: each category by when it first appeared, ties by name, OTHER_BETS last.

    A category first appears at its earliest change after the history's
    start, which is the first fill of its first bot purchase (the first of
    history.steps). The history always starts just before the bot's first
    fill, and a newer purchase only ever adds a later change, so a category
    that appears later always comes after every category before it, however
    much it puts in or is worth: a category keeps its place, and so its color
    on the page, from one read to the next. That holds while the matched
    purchases and their categories stay the same: Kalshi filing a series
    under another category, a /series listing that cannot be read with no
    copy cached (the categories then come from infer_category), or an older
    trade log found later can rename or reorder them. A group with no change
    in the history comes after those that have one, by name. OTHER_BETS can
    have none (without a history every holding is OTHER_BETS; with one, a
    position held since before the bot's first fill and untouched since),
    and it comes last in any case.

    Args:
        history (History | None): The history, or None.
        holdings (list[Holding]): What is held now.

    Returns:
        tuple[str, ...]: Every group in the history or the holdings.
    """
    first = ({group: steps[0][0] for group, steps in history.steps.items() if steps}
             if history else {})
    groups = {holding.group for holding in holdings} | (set(history.value) if history else set())
    never = datetime.min.replace(tzinfo=UTC)     # only beside groups with no change at all
    return tuple(sorted(groups, key=lambda g: (g == OTHER_BETS, g not in first,
                                               first.get(g, never), g)))


def _sorted_holdings(holdings: list[Holding], group_order: tuple[str, ...]) -> list[Holding]:
    """
    Order the holdings for the page: by group, in group_order, then by event and market.

    Within a group the markets of one question (the rungs of a ladder, both
    legs of a pair on one event) sit together, whatever each is worth.

    Args:
        holdings (list[Holding]): The holdings, from _holdings.
        group_order (tuple[str, ...]): The groups' order, from _group_order
            (it lists every holding's group).

    Returns:
        list[Holding]: The holdings in that order.
    """
    place = {group: i for i, group in enumerate(group_order)}
    return sorted(holdings, key=lambda h: (place[h.group], h.event_ticker, h.ticker, h.side))


def _held_words(count: Decimal) -> str:
    """
    A signed holding in words: "10 YES", "4 NO" or "none".

    Args:
        count (Decimal): Contracts held, YES positive and NO negative.

    Returns:
        str: The words.
    """
    if not count:
        return "none"
    return f"{config.count_text(float(abs(count)))} {'YES' if count > 0 else 'NO'}"


def _event_of(ticker: str, markets: dict[str, Market]) -> str:
    """
    A market's event: as Kalshi lists it, else the ticker less its last "-" part.

    Args:
        ticker (str): The market.
        markets (dict[str, Market]): Markets looked up.

    Returns:
        str: The event ticker.
    """
    market = markets.get(ticker)
    if market is not None and market.event_ticker:
        return market.event_ticker
    return ticker.rsplit("-", 1)[0]


def _held_since(ledger: Ledger, start: datetime | None) -> set[str]:
    """
    The markets held at some moment from `start` on.

    Args:
        ledger (Ledger): The ledger, from build_ledger.
        start (datetime | None): The first moment; None for the markets held now.

    Returns:
        set[str]: Markets held at `start`, or with a change after it (all
            those held now when `start` is None).
    """
    if start is None:
        return set(ledger.held_now)
    held: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    later: set[str] = set()
    for e in ledger.events:
        if e.time > start:
            later.add(e.ticker)
        else:
            held[(e.ticker, e.owner, e.side)] += e.contracts
    return later | {ticker for (ticker, _, _), count in held.items() if count}


def build_live_view(client: Any, *, risk_free: treasury.RiskFreeRates | None,
                    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
                    trade_logs: Iterable[Path]) -> LiveView:
    """
    Read the account and work out everything the Live trading tab shows.

    In order: read the account and the trade logs; match the bot's orders;
    look up the markets (those held now, those traded since the bot's first
    fill, those whose payout must be rebuilt, every bot leg, then any held
    at the start that are still missing); rebuild the payouts and build the
    ledger; compare what it holds with Kalshi's positions (a warning per
    market that differs); read the daily prices; then the history, each
    period of config.LIVE_DASHBOARD_PERIODS, the purchases' returns, the
    holdings now and the cash check. A bot purchase's group is its market
    A's Kalshi category (historical.series_labels, the backtest page's
    filing rule, with historical.infer_category for a series Kalshi's
    listing lacks); everything else is OTHER_BETS. Before the bot's first
    fill there is no history and no period.

    Args:
        client (Any): A client from auth.build_client.
        risk_free (treasury.RiskFreeRates | None): Keyword-only: the T-bill
            yields for the ratios, or None (0%).
        series_categories (dict | None): Keyword-only:
            historical.load_series_categories' map, or None.
        trade_logs (Iterable[Path]): Keyword-only: the trade logs
            (trade_log_paths()).

    Returns:
        LiveView: The view.

    Raises:
        ValueError: If a reply or record from Kalshi cannot be read.
        KeyError: If a record lacks a field this needs.
        ApiException: If Kalshi answers with an error status after the retries.
    """
    account = read_account(client)
    trades, runs, log_warnings = read_trade_logs(trade_logs)
    owner_by_fill, match_warnings = match_bot_fills(trades, account.fills)
    bot_fills = [f for f in account.fills if f.fill_id in owner_by_fill]
    start = bot_fills[0].time - timedelta(microseconds=1) if bot_fills else None
    wanted = (set(account.positions) | tickers_needing_payout(account)
              | {leg.ticker for trade in trades for leg in trade.legs})
    if start is not None:
        wanted |= {f.ticker for f in account.fills if f.time > start}
    markets = read_markets(client, wanted)
    payouts, payout_warnings = all_payouts(account, markets)
    owner_legs = {trade.trade_id: frozenset(leg.ticker for leg in trade.legs) for trade in trades}
    ledger = build_ledger(account.fills, payouts, owner_by_fill, owner_legs)
    priced = _held_since(ledger, start)
    if missing := priced - wanted:
        markets = {**markets, **read_markets(client, missing)}

    position_warnings = [
        f"{ticker}: Kalshi holds {_held_words(account.positions.get(ticker, _ZERO))}, but the "
        f"fills and payouts add up to {_held_words(ledger.held_now.get(ticker, _ZERO))}"
        for ticker in sorted(set(ledger.held_now) | set(account.positions))
        if ledger.held_now.get(ticker, _ZERO) != account.positions.get(ticker, _ZERO)]

    if start is None:
        marks, mark_warnings = Marks({}, _now_marks(priced, markets), account.read_at), []
    else:
        marks, mark_warnings = read_marks(client, priced, start, markets, account.read_at)

    category: dict[str, str] = {}
    for trade in trades:
        event = _event_of(trade.legs[0].ticker, markets)
        # Cross-module: the backtest page's filing rule, so a category means the same on both tabs
        category[trade.trade_id] = historical.series_labels(
            event, historical.infer_category(event), series_categories)[0]

    def group_of(owner: str) -> str:
        """A ledger owner's group: its purchase's category, or OTHER_BETS."""
        return category.get(owner, OTHER_BETS)

    history = None if start is None else build_history(account, ledger, marks, group_of, start)
    returns = trade_returns(trades, ledger, marks)
    periods = None if history is None else tuple(
        period_stats(history, returns, risk_free, label, months)
        for label, months in config.LIVE_DASHBOARD_PERIODS)
    holdings = _holdings(ledger, marks, markets, group_of)
    group_order = _group_order(history, holdings)
    holdings = _sorted_holdings(holdings, group_order)

    run_of = {trade.trade_id: trade.logged_at for trade in trades}
    first_bot_fill: dict[datetime, datetime] = {}
    for fill in bot_fills:
        run = run_of[owner_by_fill[fill.fill_id]]
        first_bot_fill.setdefault(run, fill.time)          # bot_fills are oldest first
    # The bot's purchases' own fills are never read as a run's sale orders
    cash_check = check_logged_cash(account, ledger, runs, first_bot_fill,
                                   bot_fill_ids=frozenset(owner_by_fill))
    cash_warnings = _cash_miss_warnings(cash_check)

    changing = (["The account kept changing while it was read: the figures may not all be "
                 "from one moment"] if account.changing else [])
    warnings = (*account.warnings, *changing, *log_warnings, *match_warnings, *payout_warnings,
                *ledger.warnings, *position_warnings, *mark_warnings, *cash_warnings)
    return LiveView(account.read_at, account.cash, account.kalshi_positions_value,
                    tuple(holdings), group_order, history, periods, tuple(returns), cash_check,
                    risk_free, account.changing, tuple(warnings))


# ---- the log line ----------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """
    Turn a record into plain JSON values: a Decimal into its exact text, a datetime into ISO text.

    Args:
        value (Any): A Decimal, datetime, dict, list, tuple, or a plain JSON value.

    Returns:
        Any: The same record in JSON's own types.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def snapshot_record(view: LiveView) -> dict:
    """
    One read of the account as a JSON-ready record: the holdings and cash at that moment.

    Args:
        view (LiveView): The view, from build_live_view.

    Returns:
        dict: "read_at", "cash", "kalshi_positions_value", "holdings_value",
            "changing" and "holdings" (each holding's fields); every Decimal
            as its exact text and every time as ISO text.
    """
    return _jsonable({
        "read_at": view.read_at,
        "cash": view.cash,
        "kalshi_positions_value": view.kalshi_positions_value,
        "holdings_value": view.holdings_value,
        "changing": view.changing,
        "holdings": [dataclasses.asdict(holding) for holding in view.holdings],
    })


def append_snapshot(view: LiveView) -> None:
    """
    Add one line, snapshot_record(view) as JSON, to config.LIVE_PORTFOLIO_LOG_FILE.

    The path is read when this is called, so tests point it elsewhere. A line
    that cannot be written is a WARNING in the log, never an error: the page
    still shows the read.

    Args:
        view (LiveView): The view, from build_live_view.
    """
    path = Path(config.LIVE_PORTFOLIO_LOG_FILE)
    try:
        line = json.dumps(snapshot_record(view), allow_nan=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logging.warning("Could not add this read of the account to %s (%s)", path,
                        api_error_summary(exc))
