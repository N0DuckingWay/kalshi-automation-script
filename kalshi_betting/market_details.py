"""
File: market_details.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The backtest's side data about each market, and the checks that it agrees
    with itself. Three kinds of data are fetched and kept apart from the
    settled-market corpus, so the corpus and its caches are untouched:
      - each pair market's extra fields (its timing fields, and the numbers,
        strikes and rules text), from /markets and /historical/markets, 100
        tickers a request;
      - every trade of each pair market, from /historical/trades and
        /markets/trades;
      - a record of live markets' scheduled closes, which a command saves so a
        later run can learn which field gives the close a live run sees.
    The checks compare the three volume sources (the market's own total, the
    hourly candles, the trades), the candles' open/high/low/close order, the
    trades' prices and ids, the settlement value against the result and the
    order of the timing fields. The backtest calls the loaders and logs the
    coverage and the checks. One use decides trades: the scheduled close.
    learn_close_rules works out, per series, which timing field (with a fixed
    offset) reproduces every live close the captures saw, and scheduled_close
    applies that rule to a settled market's own timing fields; with
    config.BACKTEST_SCHEDULED_CLOSE on, the backtest reads that close where a
    live run reads its close (backtester._maturity_close). Nothing else here
    decides a trade.

Dependencies:
    config (the MARKET_DETAILS_* and INTEGRITY_* settings), _http (one-line
    error text) and historical (the retried read-only signed GET, the cache
    folder, the gzip JSON helpers, series_ticker, the client builder). Every
    request is a read-only GET: it never places an order and never imports
    trader. Backtest-only: imported by backtester.py alone,
    never by a live module (tests/test_market_details.py::TestIsolation).

Notes:
    Caches, all under historical.CACHE_DIR (backtest_cache/):
      market_details.jsonl.gz  one record per line, settled markets only (a
                               settled record does not change); each call
                               appends its new records as a new gzip member,
                               and a read keeps the latest line of a ticker
      trades/<ticker>.json.gz  one market's trades, only when the walk
                               reached its end and the trades add up to the
                               market's volume_fp
      scheduled_closes/<UTC stamp>.json.gz  one capture of live closes
    Values are kept as Kalshi sends them (dollar and contract counts as
    strings); this module validates types and parses nothing else.
    A scheduled-close rule is kept only when it reproduces every observation
    of its series, on at least config.SCHEDULED_CLOSE_MIN_OBSERVATIONS
    distinct markets from config.SCHEDULED_CLOSE_MIN_EVENTS events (the same
    market captured twice counts once), and is applied only to a market
    whose latest-minus-expected expiration gap is one the rule saw.
    Two commands, both read-only:
        python3 -m kalshi_betting.market_details capture-closes
        python3 -m kalshi_betting.market_details check
"""
import argparse
import bisect
import gzip
import json
import logging
import math
import re
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config, historical
from ._http import api_error_summary

# Where each cache lives. Looked up through these module names when used, so a
# test points them at its own folder.
_DETAILS_CACHE = historical.CACHE_DIR / "market_details.jsonl.gz"
_TRADES_DIR = historical.CACHE_DIR / "trades"
_CLOSES_DIR = historical.CACHE_DIR / "scheduled_closes"

# The format tags of the two files this module defines
_TRADES_FORMAT = "trades-v1"
CLOSES_FORMAT = "scheduled-closes-v1"

# A market in one of these states has its final record; it will not change. (A
# "determined" market can still lack its settlement fields, so it is not frozen.)
SETTLED_STATUSES = frozenset({"settled", "finalized"})

# Seconds one hourly candle covers
_PERIOD_SECONDS = config.CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60

# A market's ticker as a file name; anything else is fetched but never cached
_FILE_SAFE_TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The expected JSON type of every field this module keeps. "number" means an
# int or a float but never a bool.
_NUMBER = "number"
TIMING_FIELDS: dict[str, Any] = {
    "ticker": str,
    "event_ticker": str,
    "status": str,
    "open_time": str,
    "close_time": str,
    "expiration_time": str,
    "latest_expiration_time": str,
    "expected_expiration_time": str,
    "occurrence_datetime": str,
    "can_close_early": bool,
    "settlement_timer_seconds": _NUMBER,
    "settlement_ts": str,
    "settlement_value_dollars": str,
    "result": str,
}
FULL_ONLY_FIELDS: dict[str, Any] = {
    "volume_fp": str,
    "volume_24h_fp": str,
    "open_interest_fp": str,
    "last_price_dollars": str,
    "previous_price_dollars": str,
    "previous_yes_ask_dollars": str,
    "previous_yes_bid_dollars": str,
    "yes_bid_size_fp": str,
    "yes_ask_size_fp": str,
    "notional_value_dollars": str,
    "strike_type": str,
    "floor_strike": _NUMBER,
    "cap_strike": _NUMBER,
    "custom_strike": dict,
    "expiration_value": str,
    "rules_primary": str,
    "rules_secondary": str,
    "early_close_condition": str,
    "settlement_bounds_type": str,
    "market_type": str,
}
LEVEL_TIMING = "timing"
LEVEL_FULL = "full"

# The checks, by the name each failure sentence starts with
CHECK_CANDLE_VOLUME = "candle_volume"      # candle volumes sum to the market's volume_fp
CHECK_TRADE_VOLUME = "trade_volume"        # trades sum to the market's volume_fp
CHECK_HOUR_VOLUME = "hourly_volume"        # each hour's trades sum to that hour's candle volume
CHECK_TRADE_ID = "trade_id"                # no trade_id appears twice
CHECK_TRADE_PRICE = "trade_price"          # yes + no = 1 and both in (0, 1)
CHECK_CANDLE_RANGE = "candle_range"        # low <= open, close <= high; bid <= ask
CHECK_SETTLEMENT = "settlement_value"      # settlement_value_dollars agrees with result
CHECK_TIMING = "timing_order"              # open_time <= close_time <= latest_expiration_time
CHECK_UNREADABLE = "unreadable_record"     # a record so malformed a check could not run
# A data anomaly is a finding about Kalshi's own data that no correct parse
# could have changed (a crossed close: the YES bid above the YES ask). It is
# counted and named beside the failures but is never itself a failure.
CHECK_CANDLE_CROSSED = "candle_crossed"    # YES bid close above YES ask close
ANOMALY_NOTE = "Kalshi's data, not counted as a failure"
CHECK_NAMES = (
    CHECK_CANDLE_VOLUME, CHECK_TRADE_VOLUME, CHECK_HOUR_VOLUME, CHECK_TRADE_ID,
    CHECK_TRADE_PRICE, CHECK_CANDLE_RANGE, CHECK_SETTLEMENT, CHECK_TIMING,
    CHECK_UNREADABLE,
)
ANOMALY_NAMES = (CHECK_CANDLE_CROSSED,)  # not in CHECK_NAMES: the failure checks only



class _WalkIncomplete(Exception):
    """A trades walk that cannot be trusted to have reached its end."""


def _get(client: Any, path: str, **params: Any) -> Any:
    """
    One retried, read-only signed GET (the module's only way to Kalshi).

    Looked up through historical when called, so a test replaces this one name
    to answer every request, and nothing here can reach an order endpoint.

    Args:
        client (Any): An authenticated KalshiClient.
        path (str): Full API path including the /trade-api/v2 prefix.
        **params: Query parameters; None values are left out.

    Returns:
        Any: The parsed JSON body.

    Raises:
        Exception: Whatever historical._historical_get raises after its retries.
    """
    return historical._historical_get(client, path, **params)


def _pause() -> None:
    """Sleep the pause that follows each request (config.MARKET_DETAILS_REQUEST_PAUSE_SECONDS)."""
    time.sleep(config.MARKET_DETAILS_REQUEST_PAUSE_SECONDS)


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, or None."""
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _number(value: Any) -> float | None:
    """
    A finite float from a number or a numeric string, else None.

    Args:
        value (Any): A JSON value.

    Returns:
        float | None: The value, or None for a bool, a missing value or text
            that is not a finite number.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_time(value: Any) -> datetime | None:
    """
    An aware datetime from one of Kalshi's ISO timestamps, else None.

    Kalshi writes 0 to 6 fractional digits and a trailing Z; a time with no
    offset is read as UTC.

    Args:
        value (Any): A JSON value.

    Returns:
        datetime | None: The moment, or None when it cannot be read.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _time_ms(value: Any) -> int | None:
    """
    Unix milliseconds of an ISO timestamp, whole-integer exact.

    Args:
        value (Any): A JSON value.

    Returns:
        int | None: Milliseconds since the epoch, or None when unreadable.
    """
    moment = _parse_time(value)
    if moment is None:
        return None
    try:
        delta = moment - datetime(1970, 1, 1, tzinfo=UTC)
    except OverflowError:
        return None
    return (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000


def timing_epochs(record: dict) -> tuple[int | None, int | None, int | None]:
    """
    A market record's three close-related times as whole Unix seconds.

    The compact form the backtest holds for every groupable market, in place
    of the record's strings: (close_time, latest_expiration_time,
    expected_expiration_time). Each ISO timestamp is parsed once here, in any
    of Kalshi's spellings (a trailing Z, 0 to 6 fractional digits).

    Args:
        record (dict): A record from load_market_details.

    Returns:
        tuple[int | None, int | None, int | None]: The three times; None for
            one that is absent or cannot be read.
    """
    out: list[int | None] = []
    for name in ("close_time", "latest_expiration_time", "expected_expiration_time"):
        moment = _parse_time(record.get(name)) if isinstance(record, dict) else None
        try:
            out.append(None if moment is None else int(moment.timestamp()))
        except (OverflowError, OSError, ValueError):
            out.append(None)
    return out[0], out[1], out[2]


def _type_ok(value: Any, expected: Any) -> bool:
    """Whether a value has the type a field is expected to have."""
    if expected == _NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    return isinstance(value, expected)


def _record(raw: dict, full: bool) -> tuple[dict, int]:
    """
    Cut one market payload down to the fields this module keeps.

    Args:
        raw (dict): A market object from /markets or /historical/markets.
        full (bool): Keep the numbers, strikes and rules text as well as the
            timing fields.

    Returns:
        tuple[dict, int]: The record (with a "level" of "full" or "timing")
            and how many present fields had an unexpected type and were left
            out. A null or absent field is simply left out.
    """
    fields = {**TIMING_FIELDS, **FULL_ONLY_FIELDS} if full else TIMING_FIELDS
    record: dict[str, Any] = {}
    bad = 0
    for name, expected in fields.items():
        value = raw.get(name)
        if value is None:
            continue
        if _type_ok(value, expected):
            record[name] = value
        else:
            bad += 1
    record["level"] = LEVEL_FULL if full else LEVEL_TIMING
    return record, bad


def _markets_of(data: Any) -> list[dict]:
    """The market objects of a /markets-style reply; raises ValueError when it has no list."""
    markets = data.get("markets") if isinstance(data, dict) else None
    if not isinstance(markets, list):
        raise ValueError("reply has no markets list")
    return [m for m in markets if isinstance(m, dict)]


def _fetch_batch(client: Any, batch: list[str], full: frozenset[str]
                 ) -> tuple[dict[str, dict], list[str], int, str | None]:
    """
    Read one batch of tickers: /markets first, /historical/markets for the rest.

    Never raises. A request that fails ends the batch there: what was read so
    far is kept, the tickers still unread come back as failed.

    Args:
        client (Any): An authenticated KalshiClient.
        batch (list[str]): Up to MARKET_DETAILS_BATCH_SIZE tickers.
        full (frozenset[str]): Tickers whose full record is wanted.

    Returns:
        tuple: (records read by ticker, tickers neither endpoint listed,
            fields left out for an unexpected type, a one-line failure text or
            None). The tickers a failure left unread are not in the second
            element; the caller counts them as batch minus read minus
            not-listed.
    """
    prefix = historical._API_PREFIX
    found: dict[str, dict] = {}
    bad = 0
    pending = list(batch)
    for path in (f"{prefix}/markets", f"{prefix}/historical/markets"):
        if not pending:
            break
        try:
            data = _get(client, path, tickers=",".join(pending),
                        limit=config.MARKET_DETAILS_PAGE_SIZE)
            listed = _markets_of(data)
        except Exception as exc:
            return found, [], bad, api_error_summary(exc)
        _pause()
        wanted = set(pending)
        for raw in listed:
            ticker = raw.get("ticker")
            if isinstance(ticker, str) and ticker in wanted and ticker not in found:
                found[ticker], extra = _record(raw, ticker in full)
                found[ticker]["ticker"] = ticker
                bad += extra
        pending = [t for t in pending if t not in found]
    return found, pending, bad, None


class _BadCache(Exception):
    """A details cache file that cannot be read past some line (damaged, truncated or not JSON lines)."""


def _stream_details(path: Path) -> Iterator[dict]:
    """
    Yield the records of the details cache, one per line, oldest line first.

    The file is gzip JSON lines, possibly several gzip members back to back
    (each call appends one). It is streamed, never held.

    Args:
        path (Path): The cache file; must exist.

    Yields:
        dict: One record per good line.

    Raises:
        _BadCache: At the first line that is not a JSON object with a ticker,
            or when the gzip stream is damaged or cut short. Every record
            before it has already been yielded.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise _BadCache(f"line {number} is not JSON") from exc
                if not isinstance(record, dict) or not isinstance(record.get("ticker"), str):
                    raise _BadCache(f"line {number} is not a record with a ticker")
                yield record
    except (OSError, EOFError, zlib.error, ValueError) as exc:
        if isinstance(exc, _BadCache):
            raise
        raise _BadCache(f"the gzip stream is damaged ({type(exc).__name__})") from exc


def _read_details(path: Path, wanted: set[str], project: Callable[[dict], Any] | None
                  ) -> tuple[dict[str, tuple[str | None, Any]], bool]:
    """
    Read the requested tickers' latest records from the details cache.

    Streams the file once and keeps only the wanted tickers, each as
    (level, record or project(record)); a later line of a ticker replaces an
    earlier one. A damaged or truncated tail ends the read there with one
    WARNING; every record before it is kept.

    Args:
        path (Path): The cache file.
        wanted (set[str]): The tickers to keep.
        project (Callable | None): Applied to each kept record as it is read.

    Returns:
        tuple: ({ticker: (level, kept)}, whether the file's tail was bad).
    """
    kept: dict[str, tuple[str | None, Any]] = {}
    if not path.exists():
        return kept, False
    try:
        for record in _stream_details(path):
            ticker = record["ticker"]
            if ticker in wanted:
                kept[ticker] = (record.get("level"), record if project is None else project(record))
    except _BadCache as exc:
        logging.warning("Market details cache %s is damaged (%s): keeping the records "
                        "before it", path.name, exc)
        return kept, True
    return kept, False


def _append_details(path: Path, records: list[dict], repair: bool) -> None:
    """
    Append records to the details cache as one new gzip member.

    The existing file is never rewritten, with one exception: when its tail was
    found damaged (repair), the good prefix is first copied to a new file, so
    the appended member is reachable (a gzip reader stops at a damaged member).

    Args:
        path (Path): The cache file; its folder is created.
        records (list[dict]): The records to add.
        repair (bool): Cut a damaged tail off before appending.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if repair and path.exists():
        good: list[dict] = []
        try:
            good.extend(_stream_details(path))
        except _BadCache:
            pass
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(b"".join(json.dumps(r).encode("utf-8") + b"\n" for r in good))
        tmp.replace(path)
    with gzip.open(path, "ab") as fh:
        fh.write(b"".join(json.dumps(r).encode("utf-8") + b"\n" for r in records))


def load_market_details(client: Any, tickers: Iterable[str], *,
                        full: frozenset[str] = frozenset(),
                        stats_out: dict | None = None,
                        project: Callable[[dict], Any] | None = None) -> dict[str, Any]:
    """
    Each market's extra fields, from the cache and then from Kalshi.

    Every ticker gets its timing fields (status, open/close/expiration times,
    can_close_early, settlement fields, result, event ticker); tickers in
    `full` also get the numbers (volume, last and previous prices, open
    interest), strikes, rules text and the rest of FULL_ONLY_FIELDS. A record
    is marked "level": "full" or "timing". A cached timing-only record is
    fetched again when the full one is wanted.

    The cache is backtest_cache/market_details.jsonl.gz, one record per line.
    A call streams it once, keeping only the requested tickers (the latest line
    of a ticker wins), and appends the settled records it fetched (status
    settled or finalized) as a new gzip member; the file is never rewritten.
    A damaged tail costs one WARNING and none of the records before it.

    Tickers are read 100 at a time (MARKET_DETAILS_BATCH_SIZE) on
    CANDLESTICK_FETCH_MAX_WORKERS threads: /markets first, then
    /historical/markets for the ones it does not list. With client None only
    the cache is read. Never raises: a batch that fails is one WARNING (one
    line of text) and its tickers are left out; a ticker neither endpoint
    lists is left out too.

    Args:
        client (Any): An authenticated KalshiClient, or None to read the cache only.
        tickers (Iterable[str]): The markets wanted.
        full (frozenset[str]): The tickers whose full record is wanted.
        stats_out (dict | None): When given, filled with "requested",
            "from_cache", "fetched", "not_found", "failed" (tickers a failed
            request left unread), "bad_fields" and "full".
        project (Callable | None): When given, the result holds
            project(record) in place of each record (timing_epochs keeps three
            ints per ticker instead of a dict of strings). The cache file is
            unchanged.

    Returns:
        dict[str, Any]: Ticker -> record (or project(record)), for the tickers
            that were read.
    """
    wanted = list(dict.fromkeys(t for t in tickers if isinstance(t, str) and t))
    cached, bad_tail = _read_details(_DETAILS_CACHE, set(wanted), project)
    result: dict[str, Any] = {}
    todo: list[str] = []
    full_count = 0
    for ticker in wanted:
        entry = cached.get(ticker)
        if entry is not None and (ticker not in full or entry[0] == LEVEL_FULL):
            result[ticker] = entry[1]
            full_count += entry[0] == LEVEL_FULL
        else:
            todo.append(ticker)
    del cached
    stats = {"requested": len(wanted), "from_cache": len(result), "fetched": 0,
             "not_found": 0, "failed": 0, "bad_fields": 0, "full": full_count}

    if client is not None and todo:
        size = config.MARKET_DETAILS_BATCH_SIZE
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        fresh: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.CANDLESTICK_FETCH_MAX_WORKERS) as pool:
            futures = [(batch, pool.submit(_fetch_batch, client, batch, full))
                       for batch in batches]
            for batch, future in futures:
                try:
                    found, unlisted, bad, failure = future.result()
                except Exception as exc:  # _fetch_batch never raises; stay loud-free if it does
                    found, unlisted, bad, failure = {}, [], 0, api_error_summary(exc)
                stats["bad_fields"] += bad
                stats["fetched"] += len(found)
                stats["not_found"] += len(unlisted)
                if failure is not None:
                    stats["failed"] += len(batch) - len(found) - len(unlisted)
                    logging.warning(
                        "Market details: a batch of %d tickers could not be fully read "
                        "(%s); %d left out", len(batch), failure,
                        len(batch) - len(found) - len(unlisted))
                for ticker, record in found.items():
                    result[ticker] = record if project is None else project(record)
                    stats["full"] += record.get("level") == LEVEL_FULL
                    if record.get("status") in SETTLED_STATUSES:
                        fresh.append(record)
        if fresh:
            try:
                _append_details(_DETAILS_CACHE, fresh, bad_tail)
            except OSError as exc:
                logging.warning("Market details: could not save the cache: %s",
                                api_error_summary(exc))
    else:
        stats["not_found"] = len(todo)
    if stats_out is not None:
        stats_out.update(stats)
    return result


def read_trades_cutoff(client: Any) -> int | None:
    """
    The time before which Kalshi has moved its trades to the archive.

    One retried read-only GET of /historical/cutoff, field trades_created_ts.

    Args:
        client (Any): An authenticated KalshiClient.

    Returns:
        int | None: Unix seconds, or None when it cannot be read.
    """
    try:
        data = _get(client, f"{historical._API_PREFIX}/historical/cutoff")
    except Exception:
        return None
    moment = _parse_time(data.get("trades_created_ts") if isinstance(data, dict) else None)
    return None if moment is None else int(moment.timestamp())


def _trades_path(ticker: str) -> Path | None:
    """The cache file of one market's trades, or None for a ticker that cannot name a file."""
    return _TRADES_DIR / f"{ticker}.json.gz" if _FILE_SAFE_TICKER.match(ticker) else None


def _parse_trade(raw: Any) -> tuple[tuple[int, float, float, str], str, bool] | None:
    """
    Read one trade.

    Args:
        raw (Any): An object from the trades listing.

    Returns:
        tuple | None: ((time in ms, YES price, contracts, taker side), trade id,
            whether the prices are sound: both in (0, 1) and summing to 1), or
            None when the trade cannot be read at all.
    """
    if not isinstance(raw, dict):
        return None
    when = _time_ms(raw.get("created_time"))
    yes = _number(raw.get("yes_price_dollars"))
    no = _number(raw.get("no_price_dollars"))
    count = _number(raw.get("count_fp"))
    trade_id = raw.get("trade_id")
    if when is None or yes is None or count is None or count < 0 or not isinstance(trade_id, str):
        return None
    side = raw.get("taker_side")
    sound = (0.0 < yes < 1.0 and no is not None and 0.0 < no < 1.0
             and abs(yes + no - 1.0) <= config.PRICE_EPSILON)
    return (when, yes, count, side if isinstance(side, str) else ""), trade_id, sound


def _walk_trades(client: Any, path: str, ticker: str) -> tuple[list[tuple], int]:
    """
    Read every page of one market's trades from one endpoint.

    Bounded twice (the scanner's rule): a cursor seen before ends the walk as
    incomplete, and so does config.SCANNER_MAX_PAGES pages.

    Args:
        client (Any): An authenticated KalshiClient.
        path (str): The endpoint path.
        ticker (str): The market.

    Returns:
        tuple[list[tuple], int]: The trades as (trade, trade id, whether its
            prices were sound) in the order the endpoint served them (newest
            first), and how many repeated trade ids this endpoint repeated
            (each repeat is dropped).

    Raises:
        _WalkIncomplete: A cursor repeated, the page cap was hit, a reply had
            no trades list or a trade could not be read.
        Exception: Whatever a request raises (a 404 carries .status).
    """
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    records: list[tuple] = []
    duplicates = 0
    cursor: str | None = None
    for _ in range(config.SCANNER_MAX_PAGES):
        data = _get(client, path, ticker=ticker, limit=config.MARKET_DETAILS_PAGE_SIZE,
                    cursor=cursor)
        page = data.get("trades") if isinstance(data, dict) else None
        if not isinstance(page, list):
            raise _WalkIncomplete("reply has no trades list")
        for raw in page:
            parsed = _parse_trade(raw)
            if parsed is None:
                raise _WalkIncomplete("a trade could not be read")
            trade, trade_id, sound = parsed
            if trade_id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(trade_id)
            records.append((trade, trade_id, sound))
        _pause()
        cursor = data.get("cursor") or None
        if cursor is None:
            return records, duplicates
        if cursor in seen_cursors:
            raise _WalkIncomplete("a cursor repeated")
        seen_cursors.add(cursor)
    raise _WalkIncomplete(f"more than {config.SCANNER_MAX_PAGES} pages")


def _merge_walks(walks: Sequence[list[tuple]]) -> tuple[list[tuple], list[int]]:
    """
    Merge the walks of one market's endpoints into its trades, oldest first.

    A trade id served by more than one endpoint is kept once, from the earlier
    walk in the sequence (Kalshi may list a trade near the cutoff in both).

    Args:
        walks (Sequence[list[tuple]]): _walk_trades' records, one list per endpoint.

    Returns:
        tuple[list[tuple], list[int]]: The trades oldest first, and the times (ms)
            of those whose prices were unsound, oldest first.
    """
    seen: set[str] = set()
    trades: list[tuple] = []
    unsound: list[int] = []
    for records in walks:
        for trade, trade_id, sound in records:
            if trade_id in seen:
                continue
            seen.add(trade_id)
            trades.append(trade)
            if not sound:
                unsound.append(trade[0])
    trades.reverse()
    trades.sort(key=lambda t: t[0])
    unsound.sort()
    return trades, unsound


def load_market_trades(client: Any, ticker: str, *, settled_before_trades_cutoff: bool,
                       volume_fp: Any = None, spans_trades_cutoff: bool | None = None,
                       stats_out: dict | None = None) -> tuple[list[tuple], int] | None:
    """
    Every trade of one market, as (time in ms, YES price, contracts, taker side).

    Oldest first. A cached walk is returned as it was saved. Otherwise the
    trades are paged from /historical/trades (a market settled before Kalshi's
    trades cutoff) or /markets/trades (after it), the other endpoint when the
    first answers 404. Kalshi moves the trades CREATED before the cutoff to the
    archive, so a market that opened before the cutoff and settled after it has
    its earlier trades only in /historical/trades and its later ones only in
    /markets/trades; such a market needs both endpoints, merged, each trade id
    kept once (a trade both endpoints list is kept from the first one walked).
    spans_trades_cutoff says which case this is: True walks both endpoints,
    False walks one, and None (the default, for a caller that does not know)
    walks the first and, only when its trades fall short of volume_fp by more
    than INTEGRITY_VOLUME_TOLERANCE, the other as well (a failure of that extra
    walk keeps the first walk's trades, uncached, as a market read from one
    endpoint always was). With spans_trades_cutoff True a 404 from one
    endpoint means it holds nothing for the market, a failure of either walk
    otherwise returns None.

    The repeated-id count is a repeat within ONE endpoint's own listing (each
    repeat is dropped and counted); an id the two endpoints share is expected
    near the cutoff, is dropped silently and is not counted, so the count
    stays a sign of a listing fault. The merged trades are returned; they are
    cached under backtest_cache/trades/<ticker>.json.gz only when every walk
    reached its end and, as well, they add up to the market's volume_fp within
    INTEGRITY_VOLUME_TOLERANCE (with no readable volume_fp nothing is cached),
    so a cache file is always a consistent one and an off-volume market is read
    again next run (the integrity check still reports it). A cut-short walk (a
    failed request, a repeated cursor, the page cap, a reply or trade that
    cannot be read) returns None and caches nothing; the reason goes to
    stats_out and a DEBUG line, never a WARNING (the caller counts them). With
    client None only the cache is read.

    Args:
        client (Any): An authenticated KalshiClient, or None to read the cache only.
        ticker (str): The market.
        settled_before_trades_cutoff (bool): Ask the archive endpoint first.
        volume_fp (Any): The market's volume_fp (a string as Kalshi sends it, or
            a number), from its full record; None when not known.
        spans_trades_cutoff (bool | None): Whether the market opened before the
            trades cutoff and settled at or after it (see above); None when not known.
        stats_out (dict | None): When given, filled with "duplicates",
            "price_failures" (trades whose YES and NO prices are not both in
            (0, 1) or do not sum to 1), "unsound_ms" (their times), "from_cache",
            "cached" (a file now holds this walk), "bytes" (the cache file's
            size), "endpoints" (how many endpoints were walked to their end) and,
            when the walk did not reach its end, "failure" (one line of text
            saying why).

    Returns:
        tuple[list[tuple], int] | None: (trades, repeated ids dropped), or None
            when the trades are not known in full.
    """
    path = _trades_path(ticker)
    cached = historical._load_gz_json_cache(path) if path is not None else None
    if (isinstance(cached, dict) and cached.get("format") == _TRADES_FORMAT
            and isinstance(cached.get("trades"), list)):
        trades = [tuple(t) for t in cached["trades"]]
        if stats_out is not None:
            stats_out.update(duplicates=int(cached.get("duplicates", 0)),
                             price_failures=int(cached.get("price_failures", 0)),
                             unsound_ms=list(cached.get("unsound_ms", [])),
                             from_cache=True, cached=True, bytes=path.stat().st_size)
        return trades, int(cached.get("duplicates", 0))

    def fail(reason: str) -> None:
        logging.debug("Trades for %s not read to their end: %s", ticker, reason)
        if stats_out is not None:
            stats_out["failure"] = reason

    if client is None:
        fail("not cached and no client to read them with")
        return None

    prefix = historical._API_PREFIX
    archive = f"{prefix}/historical/trades"
    live = f"{prefix}/markets/trades"
    first, other = (archive, live) if settled_before_trades_cutoff else (live, archive)
    total = _number(volume_fp)
    tol = config.INTEGRITY_VOLUME_TOLERANCE

    walks: list[list[tuple]] = []
    duplicates = 0
    try:
        try:
            records, repeated = _walk_trades(client, first, ticker)
            walks.append(records)
            duplicates += repeated
            asked_other = False
        except _WalkIncomplete:
            raise
        except Exception as exc:
            if _status_of(exc) != 404:
                raise
            # The first endpoint knows nothing of the market: the other holds all of it
            _pause()
            records, repeated = _walk_trades(client, other, ticker)
            walks.append(records)
            duplicates += repeated
            asked_other = True
        if not asked_other:
            if spans_trades_cutoff is True:
                try:
                    records, repeated = _walk_trades(client, other, ticker)
                except _WalkIncomplete:
                    raise
                except Exception as exc:
                    # A 404 here means this endpoint holds none of the market's trades
                    if _status_of(exc) != 404:
                        raise
                    _pause()
                else:
                    walks.append(records)
                    duplicates += repeated
            elif spans_trades_cutoff is None and total is not None:
                so_far = math.fsum(r[0][2] for r in walks[0])
                if total - so_far > tol:
                    try:
                        records, repeated = _walk_trades(client, other, ticker)
                    except Exception as exc:  # the first walk stands, as it always did
                        logging.debug("Trades for %s: the other endpoint was not read: %s",
                                      ticker, api_error_summary(exc))
                    else:
                        walks.append(records)
                        duplicates += repeated
    except _WalkIncomplete as exc:
        fail(str(exc))
        return None
    except Exception as exc:
        fail(api_error_summary(exc))
        return None

    trades, unsound = _merge_walks(walks)
    size = 0
    consistent = total is not None and abs(
        math.fsum(t[2] for t in trades) - total) <= tol
    if path is not None and consistent:
        try:
            historical._save_gz_json_cache(path, {
                "format": _TRADES_FORMAT, "ticker": ticker, "duplicates": duplicates,
                "price_failures": len(unsound), "unsound_ms": unsound,
                "trades": [list(t) for t in trades]})
            size = path.stat().st_size
        except OSError as exc:
            logging.warning("Trades for %s not cached: %s", ticker, api_error_summary(exc))
    if stats_out is not None:
        stats_out.update(duplicates=duplicates, price_failures=len(unsound),
                         unsound_ms=unsound, from_cache=False, cached=size > 0, bytes=size,
                         endpoints=len(walks))
    return trades, duplicates


def failing_check(sentence: str) -> str:
    """The name of the check a failure sentence belongs to (the text before its first colon)."""
    return sentence.split(":", 1)[0]


def is_anomaly(sentence: str) -> bool:
    """Whether a finding sentence names a data anomaly (ANOMALY_NAMES) rather than a failure."""
    return failing_check(sentence) in ANOMALY_NAMES


def split_candle_findings(findings: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split check_candles' sentences into integrity failures and data anomalies.

    Args:
        findings (Sequence[str]): check_candles' result.

    Returns:
        tuple[list[str], list[str]]: (failures, anomalies), each in the order given.
    """
    failures = [f for f in findings if not is_anomaly(f)]
    return failures, [f for f in findings if is_anomaly(f)]


def check_candles(candles: Sequence[Any], *, ticker: str = "") -> list[str]:
    """
    Check the order of every full hourly record.

    For each side of the book the candle carries (YES ask, YES bid), the low
    must not exceed the open or the close and the high must not fall below
    either; the traded-price block gets the same test over its open, close and
    mean. Those are integrity failures: values that cannot come from a correct
    parse. The YES bid at the close above the YES ask at the close is a crossed
    quote in Kalshi's own data; it is reported as a data anomaly (check name
    candle_crossed, with ANOMALY_NOTE at the end of the sentence), which
    split_candle_findings separates from the failures. A field a candle lacks is
    not checked. Never raises.

    Args:
        candles (Sequence[Any]): Full candle dicts from historical.fetch_candlesticks
            (or the candlesticks_v3 cache), where (open, high, low) are tuples
            or lists.
        ticker (str): Named in each sentence when given.

    Returns:
        list[str]: One sentence per finding, each starting "candle_range: "
            (a failure) or "candle_crossed: " (an anomaly); [] when every candle holds.
    """
    out: list[str] = []
    where = f" {ticker}" if ticker else ""
    eps = config.PRICE_EPSILON
    for candle in candles:
        try:
            if not isinstance(candle, dict):
                continue
            ts = candle.get("ts")
            for label, ohl_key, close_key in (("YES ask", "yes_ask_ohl", "yes_ask_close"),
                                              ("YES bid", "yes_bid_ohl", "yes_bid_close")):
                ohl = candle.get(ohl_key)
                close = _number(candle.get(close_key))
                if not isinstance(ohl, (list, tuple)) or len(ohl) != 3:
                    continue
                open_, high, low = (_number(v) for v in ohl)
                if None in (open_, high, low):
                    continue
                for name, value in (("open", open_), ("close", close)):
                    if value is None:
                        continue
                    if low > value + eps:
                        out.append(f"{CHECK_CANDLE_RANGE}:{where} hour ending {ts}: {label} low "
                                   f"{low} is above its {name} {value}")
                    if high < value - eps:
                        out.append(f"{CHECK_CANDLE_RANGE}:{where} hour ending {ts}: {label} high "
                                   f"{high} is below its {name} {value}")
            price = candle.get("price")
            if isinstance(price, (list, tuple)) and len(price) == 6:
                p_open, p_high, p_low, p_close, p_mean, _prev = (_number(v) for v in price)
                if p_high is not None and p_low is not None:
                    if p_low > p_high + eps:
                        out.append(f"{CHECK_CANDLE_RANGE}:{where} hour ending {ts}: traded "
                                   f"low {p_low} is above traded high {p_high}")
                    for name, value in (("open", p_open), ("close", p_close), ("mean", p_mean)):
                        if value is not None and not (p_low - eps <= value <= p_high + eps):
                            out.append(f"{CHECK_CANDLE_RANGE}:{where} hour ending {ts}: traded "
                                       f"{name} {value} is outside its low {p_low} to high {p_high}")
            bid = _number(candle.get("yes_bid_close"))
            ask = _number(candle.get("yes_ask_close"))
            if bid is not None and ask is not None and bid > ask + eps:
                out.append(f"{CHECK_CANDLE_CROSSED}:{where} hour ending {ts}: YES bid close "
                           f"{bid} is above YES ask close {ask} ({ANOMALY_NOTE})")
        except Exception:
            continue
    return out


def _check_timing(details: dict, name: str) -> list[str]:
    """Check 8: open_time <= close_time <= latest_expiration_time, where the fields are present."""
    out: list[str] = []
    open_t = _parse_time(details.get("open_time"))
    close_t = _parse_time(details.get("close_time"))
    latest = _parse_time(details.get("latest_expiration_time"))
    if open_t is not None and close_t is not None and open_t > close_t:
        out.append(f"{CHECK_TIMING}: {name}: open_time {details['open_time']} is after "
                   f"close_time {details['close_time']}")
    if close_t is not None and latest is not None and close_t > latest:
        out.append(f"{CHECK_TIMING}: {name}: close_time {details['close_time']} is after "
                   f"latest_expiration_time {details['latest_expiration_time']}")
    return out


def _check_settlement(details: dict, name: str) -> list[str]:
    """Check 7: settlement_value_dollars agrees with result (1 for yes, 0 for no)."""
    result = details.get("result")
    value = _number(details.get("settlement_value_dollars"))
    if result not in ("yes", "no") or value is None:
        return []
    expected = 1.0 if result == "yes" else 0.0
    if abs(value - expected) > config.PRICE_EPSILON:
        return [f"{CHECK_SETTLEMENT}: {name}: result is {result} but settlement_value_dollars "
                f"is {details.get('settlement_value_dollars')}"]
    return []


def _check_volumes(details: dict, candles: Sequence[dict], trades: Sequence[tuple] | None,
                   name: str) -> list[str]:
    """Checks 1, 2 and 3: the market's volume, its candles' volumes and its trades' counts agree."""
    out: list[str] = []
    tol = config.INTEGRITY_VOLUME_TOLERANCE
    slack = config.INTEGRITY_LIFE_SLACK_SECONDS
    total = _number(details.get("volume_fp"))
    open_t = _parse_time(details.get("open_time"))
    close_t = _parse_time(details.get("close_time"))
    hours = sorted((c["ts"], c["volume"]) for c in candles
                   if isinstance(c.get("ts"), int) and _number(c.get("volume")) is not None)

    if total is not None and hours and open_t is not None and close_t is not None:
        # Whole life with no hour missing: a gap in the series means candle
        # volume is absent, not wrong, so the sum is not compared then
        whole_life = (len(hours) == len(candles)
                      and hours[0][0] <= open_t.timestamp() + slack
                      and hours[-1][0] >= close_t.timestamp()
                      and all(b[0] - a[0] == _PERIOD_SECONDS
                              for a, b in zip(hours, hours[1:], strict=False)))
        if whole_life:
            summed = math.fsum(float(v) for _, v in hours)
            if abs(summed - total) > tol:
                out.append(f"{CHECK_CANDLE_VOLUME}: {name}: the candles' volumes add up to "
                           f"{summed:.2f} but volume_fp is {total:.2f}")
    if trades is None:
        return out
    if total is not None:
        summed = math.fsum(t[2] for t in trades)
        if abs(summed - total) > tol:
            out.append(f"{CHECK_TRADE_VOLUME}: {name}: the {len(trades)} trades add up to "
                       f"{summed:.2f} but volume_fp is {total:.2f}")
    if hours:
        stamps = [h[0] for h in hours]
        per_hour = [0.0] * len(hours)
        parts: list[list[float]] = [[] for _ in hours]
        for when_ms, _price, count, _side in trades:
            seconds = when_ms / 1000.0
            # A candle ending at T covers [T - 1 h, T): Kalshi files a trade made
            # exactly on the hour in the candle that starts then
            index = bisect.bisect_right(stamps, seconds)
            # Only the hours the candles cover: a trade before the first candle's
            # period, or in a gap between candles, belongs to no candle
            if index < len(stamps) and stamps[index] - _PERIOD_SECONDS <= seconds < stamps[index]:
                parts[index].append(count)
        for index in range(len(hours)):
            per_hour[index] = math.fsum(parts[index])
        bad = [(stamps[i], per_hour[i], float(hours[i][1])) for i in range(len(hours))
               if abs(per_hour[i] - float(hours[i][1])) > tol]
        if bad:
            ts, from_trades, from_candle = bad[0]
            out.append(f"{CHECK_HOUR_VOLUME}: {name}: {len(bad)} of {len(hours)} hours differ; "
                       f"the first, ending {ts}, has {from_trades:.2f} contracts in its trades "
                       f"and {from_candle:.2f} in its candle")
    return out


def check_market(details: dict | None, candles: Sequence[dict], trades: Sequence[tuple] | None,
                 *, trade_stats: dict | None = None) -> list[str]:
    """
    What does not add up for one market.

    Checks, each skipped when the data it needs is absent:
      1. the candles' volumes sum to the market's volume_fp (only when the
         candles cover the market's whole life with no hour missing between
         the first and the last; otherwise the check is skipped, not failed);
      2. the trades sum to volume_fp;
      3. each hour's trades sum to that hour's candle volume (only hours both
         cover; a trade belongs to the first candle whose end is at or after it);
      4. no trade_id repeated (from trade_stats["duplicates"]);
      5. trade prices: every trade in the market's life has its YES price in
         (0, 1), and none had YES and NO prices that fail to sum to 1 (from
         trade_stats["unsound_ms"], the times of such trades; only those
         inside the life count);
      7. settlement_value_dollars agrees with result;
      8. open_time <= close_time <= latest_expiration_time.
    (Check 6, the order inside each candle, is check_candles.) Never raises
    on malformed input.

    Args:
        details (dict | None): The market's record from load_market_details.
        candles (Sequence[dict]): Its hourly candles (ts and volume are read).
        trades (Sequence[tuple] | None): Its trades from load_market_trades, or
            None when they are not known in full.
        trade_stats (dict | None): load_market_trades' stats_out.

    Returns:
        list[str]: One sentence per failure, each starting "<check name>: ";
            [] when every check holds.
    """
    try:
        info = details if isinstance(details, dict) else {}
        name = str(info.get("ticker") or "market")
        out: list[str] = []
        out += _check_volumes(info, candles or [], trades, name)
        stats = trade_stats or {}
        if stats.get("duplicates"):
            out.append(f"{CHECK_TRADE_ID}: {name}: {stats['duplicates']} trades repeated an "
                       f"earlier trade_id")
        slack_ms = config.INTEGRITY_LIFE_SLACK_SECONDS * 1000
        begin = _time_ms(info.get("open_time"))
        end = _time_ms(info.get("settlement_ts")) or _time_ms(info.get("close_time"))

        def in_life(when_ms: int) -> bool:
            return ((begin is None or when_ms >= begin - slack_ms)
                    and (end is None or when_ms <= end + slack_ms))

        # Trades whose YES and NO prices were unsound, only those inside the
        # market's life (from their times when known, else the plain count)
        if "unsound_ms" in stats:
            unsound = sum(1 for when in stats["unsound_ms"] if in_life(when))
        else:
            unsound = int(stats.get("price_failures") or 0)
        if unsound:
            out.append(f"{CHECK_TRADE_PRICE}: {name}: {unsound} trades in the market's life "
                       f"had YES and NO prices not both in (0, 1) or not summing to 1")
        if trades is not None:
            outside = sum(1 for t in trades if in_life(t[0]) and not 0.0 < t[1] < 1.0)
            if outside:
                out.append(f"{CHECK_TRADE_PRICE}: {name}: {outside} trades in the market's "
                           f"life have a YES price outside (0, 1)")
        out += _check_settlement(info, name)
        out += _check_timing(info, name)
        return out
    except Exception as exc:  # a malformed record must never end a worker
        return [f"{CHECK_UNREADABLE}: market could not be checked ({type(exc).__name__})"]


def audit_market(client: Any, ticker: str, details: dict | None, candles: Sequence[dict], *,
                 settled_before_trades_cutoff: bool,
                 spans_trades_cutoff: bool | None = None) -> dict:
    """
    Load one market's trades, check it against its details and candles, and keep a summary.

    The trades are dropped when this returns, so a caller can run it over
    thousands of markets holding only the small summaries.

    Args:
        client (Any): An authenticated KalshiClient, or None to read the cache only.
        ticker (str): The market.
        details (dict | None): Its record from load_market_details.
        candles (Sequence[dict]): Its hourly candles (ts and volume are read).
        settled_before_trades_cutoff (bool): load_market_trades' routing.
        spans_trades_cutoff (bool | None): load_market_trades' spans_trades_cutoff.

    Returns:
        dict: "trades" (count), "contracts" (their total), "duplicates",
            "price_failures", "complete" (the walk reached its end), "from_cache",
            "bytes" (the cache file's size), "failure" (one line saying why the
            walk did not reach its end, else None; the caller counts them and
            logs one summary WARNING) and "failures" (check_market's sentences).
    """
    stats: dict = {}
    loaded = load_market_trades(client, ticker,
                                settled_before_trades_cutoff=settled_before_trades_cutoff,
                                volume_fp=(details or {}).get("volume_fp"),
                                spans_trades_cutoff=spans_trades_cutoff,
                                stats_out=stats)
    trades = None if loaded is None else loaded[0]
    summary = {
        "trades": 0 if trades is None else len(trades),
        "contracts": 0.0 if trades is None else math.fsum(t[2] for t in trades),
        "duplicates": int(stats.get("duplicates", 0)),
        "price_failures": int(stats.get("price_failures", 0)),
        "complete": trades is not None,
        "from_cache": bool(stats.get("from_cache", False)),
        "bytes": int(stats.get("bytes", 0)),
        "failure": stats.get("failure"),
        "failures": check_market(
            {**(details or {}), "ticker": ticker}, candles, trades, trade_stats=stats),
    }
    return summary


def capture_scheduled_closes(client: Any) -> Path:
    """
    Save every open market's ticker, series and live close_time.

    Pages /markets?status=open&mve_filter=exclude (read-only, retried), with
    the scanner's two bounds on the cursor loop (a cursor seen twice, then
    SCANNER_MAX_PAGES pages; either stops the walk with a WARNING and the
    markets read so far are saved). The file is
    backtest_cache/scheduled_closes/<UTC stamp>.json.gz, in the
    scheduled-closes-v1 format:
        {"meta": {"source", "taken_at", "format"},
         "observations": [{"ticker", "series", "live_close_time", "record": null}, ...]}
    "record" is filled in later, from the market's settled record, by whoever
    learns the close rule.

    Args:
        client (Any): An authenticated KalshiClient.

    Returns:
        Path: The file written.

    Raises:
        Exception: Whatever a request raises after its retries; nothing is
            written then.
    """
    observations: list[dict] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    for _ in range(config.SCANNER_MAX_PAGES):
        data = _get(client, f"{historical._API_PREFIX}/markets", status="open",
                    mve_filter="exclude", limit=config.MARKET_DETAILS_PAGE_SIZE, cursor=cursor)
        for market in _markets_of(data):
            ticker = market.get("ticker")
            close = market.get("close_time")
            if isinstance(ticker, str) and ticker and isinstance(close, str) and close:
                observations.append({
                    "ticker": ticker,
                    "series": historical.series_ticker(market.get("event_ticker") or ""),
                    "live_close_time": close,
                    "record": None,
                })
        _pause()
        cursor = data.get("cursor") or None
        if cursor is None:
            break
        if cursor in seen_cursors:
            logging.warning("Scheduled-close capture: a cursor repeated; saving the %d "
                            "markets read so far", len(observations))
            break
        seen_cursors.add(cursor)
    else:
        logging.warning("Scheduled-close capture: stopped at %d pages; saving the %d "
                        "markets read so far", config.SCANNER_MAX_PAGES, len(observations))
    taken = datetime.now(UTC)
    path = _CLOSES_DIR / f"{taken.strftime('%Y-%m-%dT%H%M%SZ')}.json.gz"
    historical._save_gz_json_cache(path, {
        "meta": {"source": "market_details capture-closes",
                 "taken_at": taken.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "format": CLOSES_FORMAT},
        "observations": observations,
    })
    return path


# The three fields a scheduled-close rule may read, by their place in
# timing_epochs' tuple, and the order a rule is chosen in when several fit a
# series: the close_time itself first (it fits only when every observation
# closed on schedule, and then the rule changes nothing — the safe choice),
# then the latest expiration, then the expected expiration.
CLOSE_RULE_FIELDS: dict[str, int] = {
    "close_time": 0,
    "latest_expiration_time": 1,
    "expected_expiration_time": 2,
}
CLOSE_RULE_PREFERENCE = ("close_time", "latest_expiration_time", "expected_expiration_time")


def _epoch_seconds(value: Any) -> int | None:
    """Whole Unix seconds of an ISO timestamp, or None when it cannot be read."""
    moment = _parse_time(value)
    if moment is None:
        return None
    try:
        return int(moment.timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _observation_times(observation: Any, timing: tuple | None = None
                       ) -> tuple[str, str, int, tuple] | None:
    """
    One capture observation as (ticker, series, live close, the record's timing), or None.

    Args:
        observation (Any): One entry of a capture's "observations" list.
        timing (tuple | None): The settled record's timing_epochs, when the
            caller looked it up because the observation's own record is null.

    Returns:
        tuple | None: None when the ticker, the series or the live close cannot be read,
            or when there is no settled record (a record whose status is not
            settled or finalized is not a final answer).
    """
    if not isinstance(observation, dict):
        return None
    ticker = observation.get("ticker")
    series = observation.get("series")
    live = _epoch_seconds(observation.get("live_close_time"))
    if (not isinstance(ticker, str) or not ticker or not isinstance(series, str)
            or not series or live is None):
        return None
    if timing is None:
        record = observation.get("record")
        if not isinstance(record, dict) or record.get("status") not in SETTLED_STATUSES:
            return None
        timing = timing_epochs(record)
    return ticker, series, live, timing


def _event_of(ticker: str) -> str:
    """A market ticker's event ticker: everything before its last hyphen (Kalshi's naming)."""
    return ticker.rsplit("-", 1)[0]


def learn_close_rules(observations: Iterable[tuple[str, str, int, tuple]],
                      stats_out: dict | None = None
                      ) -> dict[str, tuple[str, int, frozenset, int]]:
    """
    Per series, the field and fixed offset that reproduce every observed live close.

    Each observation is one market seen open by a live run and then settled:
    (its ticker, its series, the close_time the live run saw, in Unix
    seconds, and the settled record's timing_epochs tuple). For every series,
    each of the three fields of CLOSE_RULE_FIELDS is tried, in the order of
    CLOSE_RULE_PREFERENCE, with the one offset (seconds) its first
    observation implies (live close minus that field); the first field that
    every observation of the series has, and that plus the offset equals
    every live close, is the series' rule.

    A rule also records the FINGERPRINT of the markets it was learned on:
    each observation's latest_expiration_time minus its
    expected_expiration_time. Kalshi has changed how it sets these fields
    over time (older markets of one series, and markets across a clock
    change, can carry another gap between them), and a rule learned on one
    convention gives a close that is too LATE on another, which the "not
    before the realized close" guard cannot catch. So scheduled_close applies
    a rule only to a market whose own fingerprint is one the rule saw. Only
    observations carrying both fields support the rule (give it a
    fingerprint and count toward its minimums): a series needs at least
    config.SCHEDULED_CLOSE_MIN_OBSERVATIONS of them, from at least
    config.SCHEDULED_CLOSE_MIN_EVENTS distinct events (an event ticker is
    the market ticker less its last hyphenated part). A series with a miss
    on every field, or too little support, has no rule.

    The same market seen twice with the same live close counts once.

    Args:
        observations (Iterable): (ticker, series, live close seconds, timing tuple).
        stats_out (dict | None): When given, filled with "observations" (the
            distinct ones read), "covered" (those of a series with a rule) and
            "by_field" (series count per field).

    Returns:
        dict: Series ticker -> (field name, offset seconds, frozenset of the
            fingerprints seen, number of distinct events supporting it).
    """
    by_series: dict[str, list[tuple[str, int, tuple]]] = {}
    seen: set[tuple] = set()
    for ticker, series, live, timing in observations:
        # Two captures can hold one market at one live close: one fact
        key = (ticker, live)
        if key in seen:
            continue
        seen.add(key)
        by_series.setdefault(series, []).append((ticker, live, tuple(timing)))
    rules: dict[str, tuple[str, int, frozenset, int]] = {}
    for series, rows in by_series.items():
        support = [(ticker, t) for ticker, _, t in rows if t[1] is not None and t[2] is not None]
        events = {_event_of(ticker) for ticker, _ in support}
        if (len(support) < config.SCHEDULED_CLOSE_MIN_OBSERVATIONS
                or len(events) < config.SCHEDULED_CLOSE_MIN_EVENTS):
            continue
        for name in CLOSE_RULE_PREFERENCE:
            index = CLOSE_RULE_FIELDS[name]
            first = rows[0][2][index]
            if first is None:
                continue
            offset = rows[0][1] - first
            if all(t[index] is not None and t[index] + offset == live for _, live, t in rows):
                prints = frozenset(t[1] - t[2] for _, t in support)
                rules[series] = (name, offset, prints, len(events))
                break
    if stats_out is not None:
        stats_out["observations"] = sum(len(rows) for rows in by_series.values())
        stats_out["covered"] = sum(len(by_series[s]) for s in rules)
        stats_out["by_field"] = {name: sum(1 for rule in rules.values() if rule[0] == name)
                                 for name in CLOSE_RULE_PREFERENCE}
    return rules


def close_rule_applies(timing: tuple | None, rules: dict, series: str) -> bool:
    """
    Whether a series' rule may be applied to this market: its fingerprint is one the rule saw.

    Args:
        timing (tuple | None): The market's timing_epochs tuple.
        rules (dict): learn_close_rules' result.
        series (str): The market's series ticker.

    Returns:
        bool: True when the series has a rule and the market's
            latest_expiration_time minus expected_expiration_time (both
            present) is among the fingerprints the rule was learned on.
    """
    rule = rules.get(series) if isinstance(rules, dict) else None
    if rule is None or not isinstance(timing, tuple) or len(timing) != 3:
        return False
    latest, expected = timing[1], timing[2]
    return latest is not None and expected is not None and (latest - expected) in rule[2]


def scheduled_close(timing: tuple | None, rules: dict, series: str) -> int | None:
    """
    A settled market's scheduled close, in Unix seconds, by its series' rule.

    Args:
        timing (tuple | None): The market's timing_epochs tuple
            (close_time, latest_expiration_time, expected_expiration_time).
        rules (dict): learn_close_rules' result.
        series (str): The market's series ticker.

    Returns:
        int | None: The rule's field plus its offset. None (use the realized
            close) when the series has no rule, the market's fingerprint is
            not one the rule saw (close_rule_applies), the field or the
            realized close is not known, or the rule's answer falls before
            the realized close (a market does not close after its schedule
            here).
    """
    if not close_rule_applies(timing, rules, series):
        return None
    name, offset = rules[series][0], rules[series][1]
    value = timing[CLOSE_RULE_FIELDS[name]]
    realized = timing[0]
    if value is None or realized is None:
        return None
    answer = value + offset
    return answer if answer >= realized else None


def load_close_rules(client: Any = None) -> dict[str, tuple[str, int, frozenset, int]]:
    """
    Learn the scheduled-close rules from every capture under backtest_cache/scheduled_closes/.

    Reads each *.json.gz capture (scheduled-closes-v1; another format, or a
    file that cannot be read, is one WARNING and left out). An observation
    whose record is null is looked up through load_market_details (timing
    only, cached; settled records only). With client None only the cache is
    read. Never raises. Logs one INFO line naming the rules learned.

    Args:
        client (Any): An authenticated KalshiClient, or None.

    Returns:
        dict: learn_close_rules' result (series ticker -> (field, offset
            seconds, fingerprints, events)).
    """
    rows: list[tuple[str, str, int, tuple]] = []
    missing: list[dict] = []
    folder = _CLOSES_DIR
    paths = sorted(folder.glob("*.json.gz")) if folder.is_dir() else []
    for path in paths:
        try:
            data = _read_gz_json(path)
        except (OSError, EOFError, zlib.error, ValueError) as exc:
            logging.warning("Scheduled close: capture %s cannot be read (%s); left out",
                            path.name, type(exc).__name__)
            continue
        meta = data.get("meta") if isinstance(data, dict) else None
        observations = data.get("observations") if isinstance(data, dict) else None
        if (not isinstance(meta, dict) or meta.get("format") != CLOSES_FORMAT
                or not isinstance(observations, list)):
            logging.warning("Scheduled close: capture %s is not in the %s format; left out",
                            path.name, CLOSES_FORMAT)
            continue
        for observation in observations:
            if isinstance(observation, dict) and observation.get("record") is None:
                missing.append(observation)
                continue
            row = _observation_times(observation)
            if row is not None:
                rows.append(row)
    if missing:
        # Cross-module: the settled records of the markets captured open,
        # from the details cache (and Kalshi, with a client)
        looked_up = load_market_details(
            client, [o.get("ticker") for o in missing],
            project=lambda r: (r.get("status"), timing_epochs(r)))
        for observation in missing:
            found = looked_up.get(observation.get("ticker"))
            if found is None or found[0] not in SETTLED_STATUSES:
                continue
            row = _observation_times(observation, found[1])
            if row is not None:
                rows.append(row)
    stats: dict = {}
    rules = learn_close_rules(rows, stats_out=stats)
    by_field = stats["by_field"]
    logging.info(
        "Scheduled close: rules for %d series from %d observations "
        "(close_time %d, latest_expiration_time %d, expected_expiration_time %d)",
        len(rules), stats["observations"], by_field["close_time"],
        by_field["latest_expiration_time"], by_field["expected_expiration_time"])
    return rules


def _read_gz_json(path: Path) -> Any:
    """Parse one gzipped JSON file, raising on any damage (unlike the cache loaders, which hide it)."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


class _Audit:
    """
    The failures one `check` run has found, by check name, with a few examples of each.

    Data anomalies (ANOMALY_NAMES) are kept apart in anomaly_counts and
    anomaly_examples: they are reported but are not failures.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.examples: dict[str, list[str]] = {}
        self.anomaly_counts: dict[str, int] = {}
        self.anomaly_examples: dict[str, list[str]] = {}
        self.scanned: dict[str, int] = {}

    def scan(self, kind: str) -> None:
        """Count one file scanned."""
        self.scanned[kind] = self.scanned.get(kind, 0) + 1

    def fail(self, name: str, sentence: str) -> None:
        """Record one failure (or, for a name in ANOMALY_NAMES, one data anomaly)."""
        if name in ANOMALY_NAMES:
            counts, examples = self.anomaly_counts, self.anomaly_examples
        else:
            counts, examples = self.counts, self.examples
        counts[name] = counts.get(name, 0) + 1
        kept = examples.setdefault(name, [])
        if len(kept) < config.INTEGRITY_MAX_EXAMPLES:
            kept.append(sentence)


def _audit_details(audit: _Audit) -> dict[str, dict]:
    """Audit the details cache, line by line; returns the latest good record of each ticker."""
    if not _DETAILS_CACHE.exists():
        return {}
    audit.scan("details file")
    allowed = {**TIMING_FIELDS, **FULL_ONLY_FIELDS}
    good: dict[str, dict] = {}
    try:
        for record in _stream_details(_DETAILS_CACHE):
            ticker = record["ticker"]
            audit.scan("details record")
            if record.get("level") not in (LEVEL_TIMING, LEVEL_FULL):
                audit.fail("details_record", f"{ticker}: level is {record.get('level')!r}")
            if record.get("status") not in SETTLED_STATUSES:
                audit.fail("details_record",
                           f"{ticker}: cached with status {record.get('status')!r}")
            for field, value in record.items():
                if field == "level":
                    continue
                if field not in allowed:
                    audit.fail("details_record", f"{ticker}: unexpected field {field}")
                elif not _type_ok(value, allowed[field]):
                    audit.fail("details_record",
                               f"{ticker}: {field} has type {type(value).__name__}")
            for sentence in (_check_settlement(record, ticker) + _check_timing(record, ticker)):
                audit.fail(failing_check(sentence), sentence)
            good[ticker] = record
    except _BadCache as exc:
        audit.fail("details_file", f"{_DETAILS_CACHE.name} is damaged past some line ({exc})")
    return good


def _audit_trades(audit: _Audit, details: dict[str, dict]) -> None:
    """Audit every trades file, against its details and its cached candles where they exist."""
    if not _TRADES_DIR.is_dir():
        return
    for path in sorted(_TRADES_DIR.glob("*.json.gz")):
        audit.scan("trades file")
        ticker = path.name[:-len(".json.gz")]
        try:
            data = _read_gz_json(path)
        except (OSError, EOFError, zlib.error, ValueError) as exc:
            audit.fail("trades_file", f"{ticker}: cannot be read ({type(exc).__name__})")
            continue
        rows = data.get("trades") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("format") != _TRADES_FORMAT \
                or not isinstance(rows, list):
            audit.fail("trades_file", f"{ticker}: not a {_TRADES_FORMAT} file")
            continue
        trades: list[tuple] = []
        shaped = True
        for row in rows:
            if (not isinstance(row, list) or len(row) != 4 or not isinstance(row[0], int)
                    or _number(row[1]) is None or _number(row[2]) is None
                    or not isinstance(row[3], str)):
                shaped = False
                break
            trades.append(tuple(row))
        if not shaped:
            audit.fail("trades_file", f"{ticker}: a trade is not (ms, price, count, side)")
            continue
        if any(a[0] > b[0] for a, b in zip(trades, trades[1:], strict=False)):
            audit.fail("trades_file", f"{ticker}: trades are not oldest first")
        stats = {"duplicates": data.get("duplicates", 0),
                 "price_failures": data.get("price_failures", 0),
                 "unsound_ms": data.get("unsound_ms", [])}
        candles: list[dict] = []
        candle_path = historical._CANDLES_DIR / f"{ticker}.json.gz"
        if candle_path.exists():
            try:
                cached = _read_gz_json(candle_path)
                candles = cached.get("candles", []) if isinstance(cached, dict) else []
            except (OSError, EOFError, zlib.error, ValueError):
                candles = []
        record = details.get(ticker)
        for sentence in check_market(record if record is not None else {"ticker": ticker},
                                     candles, trades, trade_stats=stats):
            audit.fail(failing_check(sentence), sentence)


def _audit_candles(audit: _Audit) -> None:
    """Audit every cached full hourly record."""
    folder = historical._CANDLES_DIR
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.json.gz")):
        audit.scan("candle file")
        ticker = path.name[:-len(".json.gz")]
        try:
            data = _read_gz_json(path)
        except (OSError, EOFError, zlib.error, ValueError) as exc:
            audit.fail("candles_file", f"{ticker}: cannot be read ({type(exc).__name__})")
            continue
        candles = data.get("candles") if isinstance(data, dict) else None
        if not isinstance(candles, list):
            audit.fail("candles_file", f"{ticker}: no candles list")
            continue
        for sentence in check_candles(candles, ticker=ticker):
            audit.fail(failing_check(sentence), sentence)


def run_check(out: Any = None) -> int:
    """
    Audit every cache this module and the candle fetch write.

    Reads the details file (field types, status, result against settlement
    value, timing order), every trades file (shape, order, duplicates, prices,
    volume against the market's total and the cached candles) and every cached
    candle file (the order inside each candle). Prints the counts and a few
    examples of each failing check.

    Args:
        out (Any): Where to print; stdout when None.

    Returns:
        int: 1 when any integrity check failed, else 0. Data anomalies (a
            crossed candle close) are printed as NOTE lines and do not count.
    """
    stream = out if out is not None else sys.stdout
    audit = _Audit()
    details = _audit_details(audit)
    _audit_trades(audit, details)
    _audit_candles(audit)
    print("Scanned: " + (", ".join(f"{n} {kind}(s)" for kind, n in sorted(audit.scanned.items()))
                         or "nothing (no cache found)"), file=stream)
    for name, count in sorted(audit.counts.items()):
        print(f"FAIL {name}: {count} failing", file=stream)
        for sentence in audit.examples[name]:
            print(f"    {sentence}", file=stream)
    for name, count in sorted(audit.anomaly_counts.items()):
        print(f"NOTE {name}: {count} found ({ANOMALY_NOTE})", file=stream)
        for sentence in audit.anomaly_examples[name]:
            print(f"    {sentence}", file=stream)
    if not audit.counts:
        print("All checks pass.", file=stream)
    return 1 if audit.counts else 0


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point: `capture-closes` and `check`.

    Args:
        argv (list[str] | None): Arguments; the process's own when None.

    Returns:
        int: 0 on success, 1 when the capture failed or a check failed.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.market_details",
        description="The backtest's side data about markets (read-only).")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capture-closes",
                        help="save every open market's live close_time for the close rule")
    commands.add_parser("check", help="audit every cache; exit 1 when a check fails")
    args = parser.parse_args(argv)

    if args.command == "check":
        return run_check()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The production client reads open markets; nothing here can place an order
    client = historical.build_prod_live_client()
    try:
        path = capture_scheduled_closes(client)
    except Exception as exc:
        logging.error("Scheduled-close capture failed: %s", api_error_summary(exc))
        return 1
    with gzip.open(path, "rb") as fh:
        count = len(json.loads(fh.read())["observations"])
    print(f"Saved {count} live closes to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
