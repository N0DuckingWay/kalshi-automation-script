"""
File: depth_model.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The backtest's model of order-book depth. Kalshi keeps no historical order
    books, so this module saves snapshots of live books, fits a table of how
    many contracts typically rest near the best bid, and builds a synthetic
    book from that table for any market at any moment. Backtest only: no
    live-trading module imports it.

Dependencies:
    config (the DEPTH_* constants and the combo series prefix), historical
    (cache folder, candle fetch, series ticker, client builder), scanner
    (open-market listing, order-book read, event_series, shard status) and
    _http (one-line error text). Every request is a read-only GET: it never
    places an order and never imports trader.

Notes:
    A book has two bid ladders, YES bids and NO bids. Buying YES walks the NO
    bids (ask = 1 - bid) and selling YES walks the YES bids, so one table, the
    contracts resting within a distance of the best bid, serves buys and sales
    alike. Save a snapshot with:
        python3 -m kalshi_betting.depth_model snapshot [--markets N]
"""
import argparse
import gzip
import hashlib
import json
import logging
import math
import numbers
import random
import sys
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import config, historical, scanner
from ._http import api_error_summary

# The format tag stored in every snapshot file's meta block
_SNAPSHOT_FORMAT = "depth-snapshot-v1"

# Float noise allowed when asking whether a level sits within a distance of the
# best bid (0.49 against 0.50 - 0.01)
_FLOAT_NOISE = 1e-9

# The lowest and highest bid a synthetic ladder may start at or reach
_LOWEST_BID = 0.01
_HIGHEST_BID = 0.99


@dataclass(frozen=True)
class DepthModel:
    """
    Typical resting contracts near the best bid, by trading activity and price.

    Every row holds the cumulative contracts resting within each of
    config.DEPTH_MODEL_DISTANCES of the best bid. A cell or volume row fitted
    to too few ladders is left out, so a lookup falls back: cell, then volume
    row, then overall.

    Attributes:
        cells (dict): (volume bucket, price band) -> row.
        volume_rows (dict): volume bucket -> row, over all prices.
        overall (tuple): The row over every ladder.
        snapshots (int): How many snapshots the ladders came from.
        ladders (int): How many bid ladders were fitted.
        first_taken (str): Earliest snapshot time (ISO UTC); "" if unknown.
        last_taken (str): Latest snapshot time; "" if unknown.
        digest (str): Identifies the table; equal tables share it.
    """
    cells: dict
    volume_rows: dict
    overall: tuple
    snapshots: int
    ladders: int
    first_taken: str
    last_taken: str
    digest: str

    def __hash__(self) -> int:
        """Hash by the table's digest, since the dict fields cannot be hashed."""
        return hash(self.digest)


@dataclass
class _Ladders:
    """The fitted ladders' rows, grouped for the table (internal to fit)."""
    cells: dict = field(default_factory=dict)
    volumes: dict = field(default_factory=dict)
    everything: list = field(default_factory=list)
    stamps: set = field(default_factory=set)


def _finite(value) -> float | None:
    """A real number as a float; None for anything else, NaN and infinity included."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        # An integer too large to hold as a float
        return None


def _rounded(price) -> float | None:
    """A price rounded to the 4 decimals of Kalshi's finest grid; None if not a number."""
    number = _finite(price)
    return None if number is None else round(number, 4)


def _volume_bucket(volume: float) -> int:
    """Index of the 24-hour volume bucket that holds this many contracts."""
    return bisect_right(config.DEPTH_MODEL_VOLUME_EDGES, volume)


def _price_band(price: float) -> int:
    """Index of the best-bid price band that holds this price."""
    return bisect_right(config.DEPTH_MODEL_PRICE_EDGES, round(price, 6))


def volume_24h(candles: list[dict], ts: int) -> float | None:
    """
    Contracts traded in the window up to and including ts.

    Counts candles with ts - DEPTH_VOLUME_WINDOW_SECONDS < candle ts <= ts,
    skipping any whose "volume" is None. The backtest and the snapshot both
    read a market's volume feature through this one function.

    Args:
        candles (list[dict]): The market's hourly candles, ascending by "ts".
        ts (int): Unix time the window ends at.

    Returns:
        float | None: The contracts traded. 0.0 when the market has earlier
            candles but none in the window. None when no candle is at or
            before ts, or when every candle in the window has no volume.
    """
    end = bisect_right(candles, ts, key=lambda c: c["ts"])
    if end == 0:
        return None
    start = bisect_right(candles, ts - config.DEPTH_VOLUME_WINDOW_SECONDS,
                         key=lambda c: c["ts"])
    if start >= end:
        return 0.0
    volumes = [v for v in (_finite(c.get("volume")) for c in candles[start:end])
               if v is not None]
    return sum(volumes) if volumes else None


def _ladder(levels) -> tuple[float, tuple[float, ...]] | None:
    """
    Read one side of a book as its best bid and cumulative contracts per distance.

    Args:
        levels (list): [[price, qty], ...] bids, as strings or numbers.

    Returns:
        tuple | None: (best bid, contracts resting within each of
            DEPTH_MODEL_DISTANCES of it). None when no level is usable.
    """
    if not isinstance(levels, (list, tuple)):
        return None
    parsed = []
    for entry in levels:
        try:
            price, qty = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            continue
        if 0.0 < price < 1.0 and qty > 0.0 and math.isfinite(qty):
            parsed.append((price, qty))
    if not parsed:
        return None
    best = max(price for price, _ in parsed)
    row = tuple(
        round(sum(qty for price, qty in parsed if price >= best - distance - _FLOAT_NOISE), 6)
        for distance in config.DEPTH_MODEL_DISTANCES
    )
    return best, row


def _gather(records: Iterable[dict]) -> _Ladders:
    """
    Sort the ladders of snapshot records into the table's groups.

    Args:
        records (Iterable[dict]): Snapshot records; one with no volume reading
            is skipped, as is a side with no usable level.

    Returns:
        _Ladders: Each ladder's row under its (volume bucket, price band), its
            volume bucket and overall, and the snapshot times seen.
    """
    found = _Ladders()
    for record in records:
        if not isinstance(record, dict):
            continue
        volume = _finite(record.get("volume_24h"))
        if volume is None:
            continue
        bucket = _volume_bucket(volume)
        used = False
        for side in ("yes", "no"):
            ladder = _ladder(record.get(side))
            if ladder is None:
                continue
            best, row = ladder
            found.cells.setdefault((bucket, _price_band(best)), []).append(row)
            found.volumes.setdefault(bucket, []).append(row)
            found.everything.append(row)
            used = True
        stamp = record.get("taken_at")
        if used and isinstance(stamp, str) and stamp:
            found.stamps.add(stamp)
    return found


def _quantile_row(rows: list) -> tuple[float, ...]:
    """The configured quantile of each distance's contracts across the rows."""
    values = np.quantile(np.asarray(rows, dtype=float), config.DEPTH_MODEL_QUANTILE, axis=0)
    # Cumulative rows never fall as the distance grows; this only guards float rounding
    values = np.maximum.accumulate(values)
    return tuple(round(float(v), 6) for v in values)


def _digest(cells: dict, volume_rows: dict, overall: tuple) -> str:
    """A short hash of the table and its distances, equal for equal tables."""
    text = repr((config.DEPTH_MODEL_DISTANCES, sorted(cells.items()),
                 sorted(volume_rows.items()), overall))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _build(found: _Ladders) -> DepthModel | None:
    """The model for gathered ladders; None when there are none."""
    if not found.everything:
        return None
    minimum = config.DEPTH_MODEL_MIN_LADDERS
    cells = {key: _quantile_row(rows) for key, rows in sorted(found.cells.items())
             if len(rows) >= minimum}
    volume_rows = {key: _quantile_row(rows) for key, rows in sorted(found.volumes.items())
                   if len(rows) >= minimum}
    overall = _quantile_row(found.everything)
    stamps = sorted(found.stamps)
    return DepthModel(
        cells=cells,
        volume_rows=volume_rows,
        overall=overall,
        snapshots=len(stamps),
        ladders=len(found.everything),
        first_taken=stamps[0] if stamps else "",
        last_taken=stamps[-1] if stamps else "",
        digest=_digest(cells, volume_rows, overall),
    )


def fit(records: Iterable[dict]) -> DepthModel | None:
    """
    Fit the depth table to snapshot records.

    Each non-empty side of a book is one ladder, filed by the market's 24-hour
    volume and the ladder's best bid. A cell reports the configured quantile
    of its ladders at each distance; a cell with too few ladders is left out
    and read from its volume row, and a thin volume row from the overall row.

    Args:
        records (Iterable[dict]): One market per record: {"ticker", "taken_at"
            (ISO UTC), "volume_24h" (float or None), "yes": [[price, qty], ...],
            "no": [[price, qty], ...]}. A record with no volume is skipped.

    Returns:
        DepthModel | None: The fitted model, or None when no ladder is usable.
    """
    return _build(_gather(records))


def _row(model: DepthModel, volume: float, best_bid: float) -> tuple[float, ...]:
    """The table row for a ladder: its cell, else its volume row, else overall."""
    bucket = _volume_bucket(volume)
    row = model.cells.get((bucket, _price_band(best_bid)))
    if row is None:
        row = model.volume_rows.get(bucket)
    return model.overall if row is None else row


def can_start_at(best_bid: float) -> bool:
    """
    Whether bid_ladder can start a ladder at this best bid.

    Args:
        best_bid (float): A side's best bid, in dollars.

    Returns:
        bool: True for a number from 0.01 to 0.99 (rounded to 4 decimals).
    """
    start = _rounded(best_bid)
    return start is not None and _LOWEST_BID <= start <= _HIGHEST_BID


def bid_ladder(model: DepthModel, best_bid: float, volume: float) -> list[list[float]]:
    """
    Synthetic bid levels for one side of a book, best first.

    The level at each table distance below the best bid holds the contracts
    the table adds between that distance and the one before. Levels that add
    nothing are dropped, and the ladder stops before any price under 1c.

    Args:
        model (DepthModel): The fitted table.
        best_bid (float): The side's best bid, in dollars.
        volume (float | None): The market's 24-hour volume.

    Returns:
        list[list[float]]: [[price, contracts], ...]. Empty when best_bid is
            one the table cannot start at (can_start_at) or the volume is
            unknown.
    """
    volume = _finite(volume)
    if model is None or volume is None or not can_start_at(best_bid):
        return []
    start = _rounded(best_bid)
    levels = []
    before = 0.0
    row = _row(model, volume, start)
    for distance, cumulative in zip(config.DEPTH_MODEL_DISTANCES, row, strict=True):
        price = round(start - distance, 4)
        if price < _LOWEST_BID:
            break
        cumulative = float(cumulative)
        added = round(cumulative - before, 6)
        before = cumulative
        if added > 0:
            levels.append([price, added])
    return levels


def book(model: DepthModel | None, yes_ask: float, yes_bid: float,
         volume: float | None) -> dict | None:
    """
    A synthetic book in the shape scanner._fetch_orderbook returns.

    The YES bids start at the YES bid. The NO bids start at 1 - the YES ask,
    so the cheapest YES ask walked from them is the YES ask itself.

    Args:
        model (DepthModel | None): The fitted table.
        yes_ask (float): The market's YES ask, in dollars.
        yes_bid (float): The market's YES bid, in dollars.
        volume (float | None): The market's 24-hour volume.

    Returns:
        dict | None: {"yes": YES bids, "no": NO bids}, each [[price, contracts],
            ...] best first. None when there is no model or no volume.
    """
    if model is None or _finite(volume) is None:
        return None
    ask = _finite(yes_ask)
    no_bid = None if ask is None else round(1 - ask, 4)
    return {
        "yes": bid_ladder(model, yes_bid, volume),
        "no": bid_ladder(model, no_bid, volume),
    }


def _snapshot_dir() -> Path:
    """The folder that holds the saved snapshots."""
    return historical.CACHE_DIR / config.DEPTH_SNAPSHOTS_DIRNAME


def _snapshot_files() -> list[Path]:
    """
    The saved snapshot files, oldest first; empty when the folder is missing.

    Raises:
        OSError: If the folder exists but cannot be read.
    """
    folder = _snapshot_dir()
    if not folder.is_dir():
        return []
    # iterdir raises on an unreadable folder, where glob would quietly find nothing
    return sorted(path for path in folder.iterdir() if path.name.endswith(".json.gz"))


def _saved_records(files: list[Path]) -> Iterator[dict]:
    """
    Yield the records of each snapshot file, one file in memory at a time.

    A market saved twice in one UTC hour (a rerun picks the same markets) is
    yielded once, from the earliest file, so it is not counted twice.

    Args:
        files (list[Path]): Snapshot files. One that cannot be read, or is not
            a depth snapshot, is skipped with a WARNING naming it.

    Yields:
        dict: One market's record.
    """
    seen = set()
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload["meta"]["format"] != _SNAPSHOT_FORMAT:
                raise ValueError("not a depth snapshot")
            records = payload["records"]
            if not isinstance(records, list):
                raise ValueError("records is not a list")
        except Exception as exc:
            logging.warning("Depth snapshot %s skipped: %s", path.name, api_error_summary(exc))
            continue
        for record in records:
            if isinstance(record, dict):
                ticker, stamp = record.get("ticker"), record.get("taken_at")
                if isinstance(ticker, str) and ticker and isinstance(stamp, str):
                    # The stamp's first 13 characters are its date and hour
                    key = (ticker, stamp[:13])
                    if key in seen:
                        continue
                    seen.add(key)
            yield record


def _volume_bucket_labels() -> list[str]:
    """Names of the volume buckets, e.g. "0", "1-99", "10,000+"."""
    edges = config.DEPTH_MODEL_VOLUME_EDGES
    labels = ["0" if edges[0] == 1 else f"under {edges[0]:,}"]
    labels += [f"{low:,}-{high - 1:,}" for low, high in zip(edges, edges[1:], strict=False)]
    labels.append(f"{edges[-1]:,}+")
    return labels


def _price_band_labels() -> list[str]:
    """Names of the best-bid price bands, e.g. "<5c", "5-20c", "95c+"."""
    cents = [f"{edge * 100:g}" for edge in config.DEPTH_MODEL_PRICE_EDGES]
    labels = [f"<{cents[0]}c"]
    labels += [f"{low}-{high}c" for low, high in zip(cents, cents[1:], strict=False)]
    labels.append(f"{cents[-1]}c+")
    return labels


def _log_cell_counts(found: _Ladders) -> None:
    """Log how many ladders each table cell was fitted to, as a short table."""
    bands = _price_band_labels()
    logging.info("Ladders per cell (rows: contracts traded in 24 h; columns: best bid)")
    logging.info("%-12s %s", "", " ".join(f"{band:>8}" for band in bands))
    for bucket, label in enumerate(_volume_bucket_labels()):
        counts = [len(found.cells.get((bucket, band), ())) for band in range(len(bands))]
        logging.info("%-12s %s", label, " ".join(f"{count:>8d}" for count in counts))


def _log_model(model: DepthModel) -> None:
    """Log which snapshots a model was fitted to."""
    logging.info("Depth model: %d snapshot(s), %d ladders, taken %s to %s",
                 model.snapshots, model.ladders, model.first_taken, model.last_taken)


def load_depth_model() -> DepthModel | None:
    """
    Fit a model from every saved snapshot.

    Never raises. A file it cannot read is skipped with one WARNING naming it,
    and with nothing usable it returns None, also with a WARNING, so the
    backtest falls back to the top of the book. A market saved twice in one
    hour counts once.

    Returns:
        DepthModel | None: The fitted model, or None when there is none.
    """
    try:
        files = _snapshot_files()
        if not files:
            logging.warning(
                "no depth snapshot saved — the backtest fills every trade at the top of "
                "the book; run: python3 -m kalshi_betting.depth_model snapshot")
            return None
        model = fit(_saved_records(files))
    except Exception as exc:
        logging.warning("Depth model not loaded (%s) — the backtest fills every trade "
                        "at the top of the book", api_error_summary(exc))
        return None
    if model is None:
        logging.warning("the saved depth snapshots hold no ladder with a volume reading — "
                        "the backtest fills every trade at the top of the book")
        return None
    _log_model(model)
    return model


def _has_yes_ask(market) -> bool:
    """Whether the market quotes a YES ask strictly between $0 and $1."""
    try:
        ask = float(market.yes_ask_dollars)
    except (TypeError, ValueError):
        return False
    return 0.0 < ask < 1.0


def _sample(markets: list, count: int, taken: datetime) -> list:
    """
    Pick the markets a snapshot reads: a random sample of the open non-combo
    markets that quote a YES ask.

    The sample is seeded from the snapshot's UTC hour and drawn from the
    markets sorted by ticker, so a rerun in the same hour picks the same ones.

    Args:
        markets (list): ApiMarket objects from the open-markets listing.
        count (int): The most markets to pick.
        taken (datetime): When the snapshot is taken (UTC).

    Returns:
        list: The chosen markets.
    """
    pool = {}
    for market in markets:
        if not market.ticker or market.ticker in pool:
            continue
        # The combo (KXMVE) family is skipped: its books are not what a trade walks
        if scanner.event_series(market.event_ticker) == config.MVE_SERIES_FAMILY_PREFIX:
            continue
        if _has_yes_ask(market):
            pool[market.ticker] = market
    ordered = [pool[ticker] for ticker in sorted(pool)]
    rng = random.Random(int(taken.strftime("%Y%m%d%H")))
    return rng.sample(ordered, min(count, len(ordered)))


def _read_market(client, market, now_ts: int, taken_at: str) -> dict | None:
    """
    Read one market's book and trailing volume.

    Args:
        client: An authenticated production client.
        market: The ApiMarket to read.
        now_ts (int): Unix time of the snapshot.
        taken_at (str): The snapshot's ISO UTC time, stored on the record.

    Returns:
        dict | None: The snapshot record, or None when the book could not be
            read or is empty, or anything failed (one WARNING says which).
    """
    try:
        # The same book read the live run prices a trade off
        ob = scanner._fetch_orderbook(client, market.ticker)
        if ob is None or not (ob["yes"] or ob["no"]):
            return None
        window = config.DEPTH_VOLUME_WINDOW_SECONDS + config.CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60
        # The market's recent hourly candles carry its volume; the live endpoint
        # is asked first because an open market has not reached the archive
        candles = historical.fetch_candlesticks(
            client, market.ticker, now_ts - window, now_ts, use_cache=False,
            series=historical.series_ticker(market.event_ticker), live_first=True)
        return {
            "ticker": market.ticker,
            "taken_at": taken_at,
            "volume_24h": volume_24h(candles, now_ts),
            "yes": ob["yes"],
            "no": ob["no"],
        }
    except Exception as exc:
        logging.warning("Depth snapshot: %s skipped: %s", market.ticker, api_error_summary(exc))
        return None


def _write_snapshot(path: Path, taken_at: str, records: list[dict]) -> None:
    """
    Write a snapshot as gzip JSON, through a temp file so it appears whole or not at all.

    Args:
        path (Path): The file to write.
        taken_at (str): The snapshot's ISO UTC time.
        records (list[dict]): The market records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "meta": {"format": _SNAPSHOT_FORMAT, "taken_at": taken_at, "markets": len(records)},
        "records": records,
    }
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def snapshot(client, markets: int = config.DEPTH_SNAPSHOT_MARKETS, *,
             now: datetime | None = None) -> Path:
    """
    Save live books and trailing volume for a random sample of open markets.

    Reads the open markets as a production run does, keeps the non-combo ones
    with a YES ask, samples up to `markets` of them, and for each reads its
    order book and its last 25 hours of hourly candles. Saves the books and
    each market's 24-hour volume under the backtest cache, then fits all saved
    snapshots and logs how many ladders each cell holds. Every request is a
    read-only GET; the candle reads also refresh each sampled market's own
    candle cache file.

    Args:
        client: An authenticated production client.
        markets (int): The most markets to sample.
        now (datetime | None): The snapshot's time; the current time when None.
            A naive time is read as UTC.

    Returns:
        Path: The file written.

    Raises:
        ValueError: If markets is below 1.
        RuntimeError: If no market could be sampled, or none had a usable book
            (nothing is saved).
    """
    if markets < 1:
        raise ValueError("markets must be at least 1")
    taken = datetime.now(UTC) if now is None else now
    taken = (taken.replace(tzinfo=UTC) if taken.tzinfo is None else taken.astimezone(UTC))
    taken = taken.replace(microsecond=0)
    taken_at = taken.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ts = int(taken.timestamp())

    # A run skips markets on shards that are not trading; so does the snapshot
    inactive = scanner.inactive_shard_indexes(scanner.fetch_shard_statuses(client))
    # The same listing a run scans
    open_markets = scanner.fetch_open_events_with_markets(client, inactive_shards=inactive)
    chosen = _sample(open_markets, markets, taken)
    if not chosen:
        raise RuntimeError("no open non-combo market with a YES ask to sample")
    logging.info("Depth snapshot: reading %d of %d open markets", len(chosen), len(open_markets))

    records: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=config.DEPTH_SNAPSHOT_MAX_WORKERS)
    try:
        futures = [pool.submit(_read_market, client, market, now_ts, taken_at)
                   for market in chosen]
        for done, future in enumerate(futures, start=1):
            record = future.result()
            if record is not None:
                records.append(record)
            if done % config.DEPTH_SNAPSHOT_PROGRESS_EVERY == 0:
                logging.info("Depth snapshot: %d of %d markets read", done, len(chosen))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not records:
        raise RuntimeError("no sampled market had a usable book; nothing saved")

    path = _snapshot_dir() / f"{taken.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    _write_snapshot(path, taken_at, records)
    logging.info("Depth snapshot saved: %s, %d markets kept of %d sampled (%d skipped: no usable book)",
                 path, len(records), len(chosen), len(chosen) - len(records))

    # Fit everything saved so far and show how well each cell is covered. The
    # snapshot is already saved, so a failure here is only reported
    try:
        found = _gather(_saved_records(_snapshot_files()))
        _log_cell_counts(found)
        model = _build(found)
    except Exception as exc:
        logging.warning("Depth snapshot saved, but the saved snapshots could not be fitted: %s",
                        api_error_summary(exc))
        return path
    if model is None:
        logging.warning("no saved ladder has a volume reading yet, so the depth model is empty")
    else:
        _log_model(model)
    return path


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point: `snapshot [--markets N]`.

    Args:
        argv (list[str] | None): Arguments; the process's own when None.

    Returns:
        int: 0 when a snapshot was saved, 1 when it could not be.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.depth_model",
        description="Save live order books for the backtest's depth model (read-only).")
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot", help="save books and volume for a sample of open markets")
    snap.add_argument(
        "--markets", type=int, default=config.DEPTH_SNAPSHOT_MARKETS,
        help="how many open markets to sample (default %(default)s)")
    args = parser.parse_args(argv)
    if args.markets < 1:
        parser.error("--markets must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The production client reads the live books; nothing here can place an order
    client = historical.build_prod_live_client()
    try:
        snapshot(client, args.markets)
    except Exception as exc:
        logging.error("Depth snapshot failed: %s", api_error_summary(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
