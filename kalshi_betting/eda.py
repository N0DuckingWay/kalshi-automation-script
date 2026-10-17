"""
File: eda.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The counting layer of the exploratory data page: it describes the settled
    markets kept on disk under backtest_cache/ (the archive and live day
    slices). Each day slice is read once and boiled down to a small JSON
    summary under backtest_cache/<config.EDA_SUMMARY_DIRNAME>/<archive|live>/
    <day>.json; a later run reuses every summary whose slice has not changed
    and reads only the new or changed slices. merge() then adds every summary
    up into one dict. Reporting only: nothing that trades, sizes or
    backtests reads any of it.

    A summary counts, per settlement week (the Monday, UTC, of the week a
    market settled in), combo or not, and series (the literal series ticker,
    historical.series_ticker): how many markets settled, how many Yes, No or
    otherwise; how many have no outcome label (subtitle), no event title
    before and after the event titles saved in event_titles_v2.json are
    joined in, or no readable open time; a lifetime histogram (close minus
    open) and a settlement-lag histogram (settlement minus close); the tick
    regimes (price_level_structure) and the exchange shards. For markets that
    are not combos it also counts the deadline wording
    (scanner.deadline_profile: cumulative, snapshot or unknown) and, for
    cumulative ones, collects the day each one's deadline names
    (scanner.stated_deadline) under its event, so the merge can count
    deadline ladders (an event with two or more stated days) even when its
    rungs sit in different slices. It also keeps, for every market that has a
    candle file under backtest_cache/candlesticks/, its result, close and
    settlement times, series and whether it is a combo, for the price
    calibration a later step reads from those candles. merge() files each
    series under its Kalshi category and first tag (historical.series_labels,
    with infer_category as the fallback) from the listing it is handed, so a
    new listing re-files every summary without reading a slice again.

    Three more readers sit beside the day-slice counts. summarise_candles()
    reads the hourly candle files under backtest_cache/candlesticks/ and, for
    the markets merge() joined to a result, works out each market's YES ask
    (and, where both sides are usable, its midpoint) 1 hour, 24 hours and 7
    days (config.EDA_PRICE_LEADS_HOURS) before it closed; it files those prices
    into price bins per category (the reliability chart and the
    favourite-longshot gap), sums the Brier score per category
    and the traded volume per category, and counts the candle files by field
    version, empty and unreadable. inventory() lists everything at the top of
    the cache folder: size, file count, date range, a plain description, the
    meta line of each assembled corpus and the depth snapshot count. khat_view()
    reduces a backtest's k-hat observations to the figures of the k view
    (mid-spread bins against the realised in-between rate, k-hat with its 95%
    interval by gap band and by category); save_khat() and load_khat() keep it
    in <cache>/eda_days/khat.json so a rebuild without a backtest can show it.

Dependencies:
    The standard library, config (EDA_SUMMARY_DIRNAME, EDA_SUMMARY_VERSION,
    EDA_MAX_WORKERS, EDA_WORKER_START_TIMEOUT_SECONDS,
    EDA_LIFETIME_EDGES_HOURS, EDA_LAG_EDGES_HOURS, EDA_CALIBRATION_BIN,
    EDA_PRICE_LEADS_HOURS, EDA_PRICE_MAX_AGE_HOURS, EDA_CANDLE_FACTS_VERSION,
    EDA_KHAT_SPREAD_BIN, KHAT_CONFIDENCE_LEVEL,
    CANDLESTICK_CACHE_FIELDS_VERSION, DEPTH_SNAPSHOTS_DIRNAME and
    MVE_SERIES_FAMILY_PREFIX), historical (the slice reader _day_store_iter,
    _SliceUnreadable, _iso_epoch, _read_title_file with _TITLE_FILE_ABSENT,
    the file names of _EVENT_TITLES_CACHE and _CANDLES_DIR, series_labels,
    series_ticker, infer_category, usable_candle_ask) and scanner
    (event_series, deadline_profile, stated_deadline, DEADLINE_CUMULATIVE).
    Imports no order-path module (trader, main, scheduler, run_lock,
    defaults_server, v2_probe, seller), and no module of the package imports
    this one (tests/test_eda.py::TestIsolation). The one exception to "only
    the standard library, config, historical and scanner" is khat_view(),
    which imports backtester and dashboard INSIDE the function, so the
    worker processes that count the day slices (which import this module
    afresh) never load pandas or plotly.

Notes:
    Which slice counts which market. The archive phase of a fetch holds only
    markets settled BEFORE the archive cutoff (each archive slice's meta
    cutoff_ts), and a live day only markets settled at or after it: the rule
    historical._assembled_records applies. So an archive summary counts only
    its records settled before its own slice's cutoff, and names how many it
    left out (`dropped_after_cutoff`). With M the largest cutoff among the
    readable archive summaries, a live summary is counted only when its day
    ends after M (its day's midnight plus one day): a live day the archive
    already covers is left out. No ticker is de-duplicated across slices.

    Kalshi's cutoff is a UTC midnight (2026-08-17T00:00:00Z on the cache read
    2026-10-17), so the two kinds meet at a day boundary. The rule above
    covers two cases where they would not. (1) The cutoff advances and a run
    extends the assembled cache (historical._extend_assembled_cache) or
    fetches a window that starts after the cutoff: the live days before the
    new cutoff are deleted, but the archive days are not fetched again, so
    they keep the old cutoff and those markets sit in neither kind on disk. A
    live day's summary is therefore kept, and counted, after its slice is
    deleted, until an archive cutoff reaches the end of its day (only
    complete days were ever saved as slices, so the summary is a complete
    record of the day). (2) A fetch interrupted after the archive was fetched
    again but before the old live days were deleted leaves those days in both
    kinds; they end at or before M, so they are not counted. merge() returns
    the archive cutoff against the first live day counted (`coverage`), so a
    gap the summaries cannot fill (a live day deleted before it was ever
    summarised) shows. What can still count a market twice: a record settled
    exactly on a UTC midnight, which two adjacent live windows may both hold
    (the fetch leaves those boundary overlaps to the assembly's ticker
    dedup), and a cutoff that is ever not a midnight.

    Reuse. A summary is reused while its key still matches: the slice's kind
    and day, its size and modification time (st_mtime_ns), its meta cutoff_ts
    (archive) and config.EDA_SUMMARY_VERSION. The cutoff is read from the
    slice's first line, which costs one small read for a streamed
    ("jsonl-v1") slice; for a legacy single-document slice that line is the
    whole file, so there only the size and time are compared (any rewrite of
    a slice changes its time). A slice that changes while it is read, or that
    cannot be read, gives a summary naming it in `unreadable`, which is not
    saved, so the next run tries it again; nothing else stops.

    Residuals: what a summary takes from other files is fixed when it is
    built. A candle file created after its slice was summarised is not joined
    to its market until that slice is summarised again (a changed slice, or a
    version bump); merge() counts the candle files with no joined market. And
    blank_title_after, the deadline wording and the ladders read the event
    titles in event_titles_v2.json as it was then.

    Summaries are built in worker processes (spawn, never fork: safe on
    macOS), each of which loads the event titles and the candle tickers once,
    through _init_worker; the parent writes each summary to disk. A spawned
    worker imports the caller's main module again, so a script that calls
    load_or_build_summaries with more than one worker must do so under
    `if __name__ == "__main__":`. Workers that do not start (that guard
    missing, or a sandbox refusing new processes) or that die are replaced by
    counting the remaining slices in this process, with one WARNING.

    Candle files. Each is read once and its facts (the YES ask and midpoint at
    each lead, whether that lead is stale, the traded volume, the field
    version, whether it is empty) are kept in
    <cache>/eda_days/candles.json under the file name, with the file's size
    and modification time, the market close time the leads were worked out
    from and config.EDA_CANDLE_FACTS_VERSION (its own version, so changing the
    facts never rebuilds the day-slice summaries); a file whose key no longer
    matches
    is read again, and the join with a market's result and category happens
    after that cache, so a new category listing or result needs no re-read.
    Only markets that are not combos are calibrated or counted for volume (a
    combo's price is not what the strategy trades), and the number left out
    is reported. A candle's "ts" is the END of its hour. A candle price is
    used only when historical.usable_candle_ask accepts it (an ask of 1.00,
    or a NO ask at the 0.99 clamp, is no quote). The primary price is the YES
    ask, what betting Yes costs. The midpoint, halfway between the YES ask and
    the YES bid (1 minus the NO ask), is a second figure that exists only where
    both sides are usable: a long shot's YES bid is under a cent, so its NO ask
    sits at the clamp and it has an ask but no midpoint, which is why the
    midpoint alone would drop most long shots. The price for a lead must come
    from a candle that ended within config.EDA_PRICE_MAX_AGE_HOURS before
    close minus the lead; an older last candle gives no price (counted as
    stale). The close time used is the market's REALIZED close (the cache
    records when the market actually closed), so where trading carries on after
    the outcome is known, the 1 hour price can already reflect the outcome. The
    candles are those the backtest priced, so they are not a random sample of
    the markets.

    The cache folder is listed with os.scandir, once for the candle folder
    and once per top-level entry for inventory(); nothing globs or sorts the
    candle folder. The candle files are read in this process, one at a time.
"""
import gzip
import json
import logging
import math
import multiprocessing
import os
import re
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from . import config, historical, scanner

# The two kinds of day slice and the folder under the cache each lives in,
# as historical._day_store_path names them.
SLICE_DIRS = {"live": "live_days", "archive": "archive_days"}
# A day slice's file name ends in this; the rest is its UTC day.
_SLICE_SUFFIX = ".json.gz"
# A summary file's name is "<day>" plus this; an interrupted save leaves this
# plus ".tmp".
_SUMMARY_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"
# A candle file's name is "<ticker>" plus this (historical.fetch_candlesticks).
_CANDLE_SUFFIX = ".json"
# The week key of a record whose settlement time cannot be read.
UNKNOWN_WEEK = "unknown"
# Joins the parts of a key: a summary's "week|combo|series", merge()'s
# "week|combo|category|tag".
KEY_SEPARATOR = "|"
# What a part's own KEY_SEPARATOR is written as, so a key always splits evenly.
_SEPARATOR_STAND_IN = "/"
# The longest reason an unreadable slice is named with.
_REASON_MAX_CHARS = 200
# Read at most this many bytes of a slice's first line when looking for its
# meta block during the reuse check: a streamed slice's meta line is a few
# hundred bytes, and a legacy slice's first line is the whole file.
_META_LINE_MAX_BYTES = 65_536
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86_400

# Set in each worker process by _init_worker and read by _summarise_task, so
# the event titles and the candle tickers are sent to a worker once rather
# than with every slice.
_WORKER_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class SliceKey:
    """
    What one summary was built from; a summary is reused only while it still matches.

    Attributes:
        kind (str): "archive" or "live".
        day (str): The slice's UTC day, from its file name ("2026-08-16").
        size (int): The slice file's size in bytes when it was read.
        mtime_ns (int): Its modification time, in nanoseconds.
        cutoff_ts (int | None): The archive cutoff in its meta block (epoch
            seconds); None for a live slice, or an archive slice with none.
        version (int): config.EDA_SUMMARY_VERSION when it was built.
    """
    kind: str
    day: str
    size: int
    mtime_ns: int
    cutoff_ts: int | None
    version: int


def _one_line(exc: BaseException) -> str:
    """
    Describe an exception on one short line.

    Args:
        exc (BaseException): The exception.

    Returns:
        str: "<type>: <first line of its message>", at most _REASON_MAX_CHARS long.
    """
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    line = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return line[:_REASON_MAX_CHARS]


def _bucket(value: float, edges: tuple) -> int:
    """
    The histogram bucket a value falls in.

    Args:
        value (float): The value, in the edges' unit.
        edges (tuple): Ascending bucket edges.

    Returns:
        int: 0 below the first edge, len(edges) at or above the last, and
            otherwise i for edges[i - 1] <= value < edges[i].
    """
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _week(epoch: float | None) -> str:
    """
    The Monday (UTC) of the week a moment falls in.

    Args:
        epoch (float | None): Epoch seconds, or None.

    Returns:
        str: The Monday's ISO date, or UNKNOWN_WEEK for None or a time no
            date can hold.
    """
    if epoch is None:
        return UNKNOWN_WEEK
    try:
        day = datetime.fromtimestamp(epoch, tz=UTC).date()
    except (OverflowError, OSError, ValueError):
        return UNKNOWN_WEEK
    return (day - timedelta(days=day.weekday())).isoformat()


def _day_end(day: Any) -> int | None:
    """
    The end of a UTC day: its midnight plus one day, in epoch seconds.

    Args:
        day (Any): The day as an ISO date ("2026-08-17").

    Returns:
        int | None: The epoch second the day ends at, or None when `day` is
            not an ISO date.
    """
    try:
        d = date.fromisoformat(day)
    except (TypeError, ValueError):
        return None
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) + _SECONDS_PER_DAY


def _iso_day(epoch: int | None) -> str | None:
    """
    The UTC date of an epoch second, as text.

    Args:
        epoch (int | None): Epoch seconds.

    Returns:
        str | None: The ISO date, or None for None.
    """
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()


def _text(value: Any) -> str:
    """
    A record field as text, "" for anything that is not a string.

    Args:
        value (Any): The field's value.

    Returns:
        str: The value, or "".
    """
    return value if isinstance(value, str) else ""


def _label(value: str) -> str:
    """
    A series, category or tag safe to use as one part of a key.

    Args:
        value (str): The label.

    Returns:
        str: The label with any KEY_SEPARATOR written as _SEPARATOR_STAND_IN.
    """
    return value.replace(KEY_SEPARATOR, _SEPARATOR_STAND_IN)


def _new_counts() -> dict:
    """
    The counters one key starts with.

    Returns:
        dict: Every counter at zero; see summarise_slice for what each counts.
    """
    return {
        "n": 0, "yes": 0, "no": 0, "other": 0,
        "blank_subtitle": 0,
        "blank_title_before": 0, "blank_title_after": 0,
        "no_open": 0,
        "lifetime_h": [0] * (len(config.EDA_LIFETIME_EDGES_HOURS) + 1),
        "lifetime_unknown": 0,
        "lag_h": [0] * (len(config.EDA_LAG_EDGES_HOURS) + 1),
        "lag_unknown": 0,
        "regime": {},
        "shard": {},
        "phrasing": {},
    }


def candle_tickers(cache_dir: Path) -> frozenset:
    """
    The tickers that have a candle file in the cache.

    Lists the names only (os.scandir); no candle file is opened.

    Args:
        cache_dir (Path): The cache folder (backtest_cache/).

    Returns:
        frozenset: Each "<ticker>.json" file's ticker; empty when there is no
            candle folder.
    """
    folder = Path(cache_dir) / historical._CANDLES_DIR.name
    try:
        with os.scandir(folder) as entries:
            return frozenset(
                entry.name[: -len(_CANDLE_SUFFIX)] for entry in entries
                if entry.name.endswith(_CANDLE_SUFFIX) and entry.is_file()
            )
    except OSError:
        return frozenset()


def load_event_titles(cache_dir: Path) -> dict:
    """
    The event titles saved in the cache's event_titles_v2.json, read only.

    Args:
        cache_dir (Path): The cache folder.

    Returns:
        dict: event_ticker -> title; empty when the file is absent, cannot be
            read or does not hold a JSON object (historical._read_title_file
            tells those apart; a missing join only raises the "blank event
            title after the join" count).
    """
    titles, outcome, detail = historical._read_title_file(
        Path(cache_dir) / historical._EVENT_TITLES_CACHE.name)
    if titles is None:
        if outcome != historical._TITLE_FILE_ABSENT:
            logging.warning("Data page: the event titles could not be read (%s: %s); "
                            "no event title is joined", outcome, detail)
        return {}
    return titles


def _slice_stat(path: Path) -> tuple[int, int]:
    """
    A slice file's size and modification time.

    Args:
        path (Path): The slice.

    Returns:
        tuple[int, int]: (size in bytes, st_mtime_ns).

    Raises:
        OSError: When the file cannot be stat'ed (for example it was deleted).
    """
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _head_cutoff(path: Path) -> tuple[bool, int | None]:
    """
    The archive cutoff in a streamed slice's meta line, read without the records.

    Args:
        path (Path): The slice.

    Returns:
        tuple[bool, int | None]: (known, cutoff_ts). known is False when the
            first line is not a self-contained meta line (a legacy
            single-document slice, or a file that cannot be read for any
            reason): the caller then compares size and time alone.
    """
    try:
        with gzip.open(path, "rb") as fh:
            line = fh.readline(_META_LINE_MAX_BYTES)
        if not line.endswith(b"\n"):
            return False, None
        head = json.loads(line)
        if not isinstance(head, dict) or "markets" in head or not isinstance(
                head.get("meta"), dict):
            return False, None
        return True, _cutoff_of(head["meta"])
    except Exception:
        return False, None


def _cutoff_of(meta: dict) -> int | None:
    """
    A meta block's cutoff_ts as an int, or None.

    Args:
        meta (dict): A slice's meta block.

    Returns:
        int | None: The cutoff in epoch seconds, or None when absent or not
            a whole number.
    """
    value = meta.get("cutoff_ts")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def summarise_slice(path: Path, kind: str, *, titles: dict, candle_set: frozenset) -> dict:
    """
    Read one day slice once and count it.

    Streams the slice through historical._day_store_iter (one record at a
    time; no meta check, the meta block is read into the key). An archive
    slice counts only the records settled before its own meta cutoff_ts
    (records at or after it are counted in `dropped_after_cutoff`); a live
    slice counts every record (merge() decides whether the live day counts).
    A record whose settlement time cannot be read is counted under the week
    UNKNOWN_WEEK in either kind.

    Per counted record, under the key "week|combo|series" (combo is "1" or
    "0", from scanner.event_series against config.MVE_SERIES_FAMILY_PREFIX;
    series is historical.series_ticker of the event ticker): n; yes, no or
    other (any other result); blank_subtitle; blank_title_before (the
    record's own event_title is blank) and blank_title_after (still blank
    once `titles` is joined in by event ticker); no_open (open_time missing
    or unreadable); lifetime_h (close minus open, in
    config.EDA_LIFETIME_EDGES_HOURS buckets) or lifetime_unknown; lag_h
    (settlement minus close, in config.EDA_LAG_EDGES_HOURS buckets) or
    lag_unknown; regime (price_level_structure -> count, "" for none); shard
    (exchange_index as text -> count, "" for none); and, for a market that
    is not a combo, phrasing (scanner.deadline_profile's verdict -> count).

    Args:
        path (Path): The slice file.
        kind (str): "archive" or "live".
        titles (dict): event_ticker -> event title, joined where a record has none.
        candle_set (frozenset): Tickers with a candle file.

    Returns:
        dict: The summary: "key" (SliceKey as a dict), "file", "unreadable"
            (None, or "<file name>: <one-line reason>", in which case every
            count is empty), "records" (records read), "counted",
            "dropped_after_cutoff", "counts" (key -> counters), "ladders"
            (event_ticker -> sorted ISO days its cumulative non-combo markets
            state) and "candles" (ticker -> [result, close_time,
            settlement_ts, series, combo]).

    Raises:
        Exception: Only for a fault that is not the slice's (a bug); a slice
            that cannot be read, or that changes while it is read, is named
            in "unreadable" instead.
    """
    path = Path(path)
    day = path.name[: -len(_SLICE_SUFFIX)] if path.name.endswith(_SLICE_SUFFIX) else path.stem
    summary = _empty_summary(path, kind, day)
    try:
        before = _slice_stat(path)
    except OSError as exc:
        summary["unreadable"] = f"{path.name}: {_one_line(exc)}"
        return summary

    meta: dict = {}
    counts: dict[str, dict] = {}
    ladders: dict[str, set] = {}
    candles: dict[str, list] = {}
    records = counted = dropped = 0
    cutoff: int | None = None
    lifetime_edges = config.EDA_LIFETIME_EDGES_HOURS
    lag_edges = config.EDA_LAG_EDGES_HOURS
    combo_family = config.MVE_SERIES_FAMILY_PREFIX
    iso_epoch = historical._iso_epoch
    try:
        for m in historical._day_store_iter(path, {}, meta_out=meta):
            if records == 0:
                cutoff = _cutoff_of(meta) if kind == "archive" else None
            records += 1
            settle = iso_epoch(m.get("settlement_ts"))
            if cutoff is not None and settle is not None and settle >= cutoff:
                dropped += 1
                continue
            counted += 1
            event = _text(m.get("event_ticker"))
            series = _label(historical.series_ticker(event))
            combo = scanner.event_series(event) == combo_family
            key = KEY_SEPARATOR.join((_week(settle), "1" if combo else "0", series))
            c = counts.get(key)
            if c is None:
                c = counts[key] = _new_counts()
            c["n"] += 1
            result = m.get("result")
            if result == "yes":
                c["yes"] += 1
            elif result == "no":
                c["no"] += 1
            else:
                c["other"] += 1
            title = _text(m.get("title"))
            subtitle = _text(m.get("subtitle"))
            if not subtitle:
                c["blank_subtitle"] += 1
            own_title = _text(m.get("event_title"))
            joined = own_title or (_text(titles.get(event)) if event else "")
            if not own_title:
                c["blank_title_before"] += 1
            if not joined:
                c["blank_title_after"] += 1
            opened = iso_epoch(m.get("open_time"))
            closed = iso_epoch(m.get("close_time"))
            if opened is None:
                c["no_open"] += 1
            if opened is not None and closed is not None:
                c["lifetime_h"][_bucket((closed - opened) / _SECONDS_PER_HOUR,
                                        lifetime_edges)] += 1
            else:
                c["lifetime_unknown"] += 1
            if closed is not None and settle is not None:
                c["lag_h"][_bucket((settle - closed) / _SECONDS_PER_HOUR, lag_edges)] += 1
            else:
                c["lag_unknown"] += 1
            regime = _text(m.get("price_level_structure"))
            c["regime"][regime] = c["regime"].get(regime, 0) + 1
            shard_raw = m.get("exchange_index")
            shard = "" if shard_raw is None else str(shard_raw)
            c["shard"][shard] = c["shard"].get(shard, 0) + 1
            if not combo:
                profile = scanner.deadline_profile(joined, title, subtitle)
                verdict = profile[0]
                c["phrasing"][verdict] = c["phrasing"].get(verdict, 0) + 1
                if verdict == scanner.DEADLINE_CUMULATIVE and event:
                    stated = scanner.stated_deadline(profile, joined, title, subtitle)
                    if stated is not None:
                        ladders.setdefault(event, set()).add(stated.isoformat())
            ticker = m.get("ticker")
            if isinstance(ticker, str) and ticker in candle_set and ticker not in candles:
                candles[ticker] = [result, m.get("close_time"), m.get("settlement_ts"),
                                   series, combo]
        after = _slice_stat(path)
    except (historical._SliceUnreadable, OSError, EOFError, zlib.error) as exc:
        summary["unreadable"] = f"{path.name}: {_one_line(exc)}"
        return summary
    if after != before:
        summary["unreadable"] = f"{path.name}: the file changed while it was read"
        return summary
    if records == 0:
        # A slice with no record still has a meta block (filled before the first record)
        cutoff = _cutoff_of(meta) if kind == "archive" else None
    summary["key"] = asdict(SliceKey(kind, day, before[0], before[1], cutoff,
                                     config.EDA_SUMMARY_VERSION))
    summary.update(records=records, counted=counted, dropped_after_cutoff=dropped,
                   counts=counts,
                   ladders={event: sorted(days) for event, days in ladders.items()},
                   candles=candles)
    return summary


def _empty_summary(path: Path, kind: str, day: str) -> dict:
    """
    A summary with nothing counted, the shape every summary has.

    Args:
        path (Path): The slice file.
        kind (str): "archive" or "live".
        day (str): The slice's day.

    Returns:
        dict: The summary's fields, every count empty, "key" holding the kind
            and day only.
    """
    return {"key": {"kind": kind, "day": day}, "file": path.name, "unreadable": None,
            "records": 0, "counted": 0, "dropped_after_cutoff": 0,
            "counts": {}, "ladders": {}, "candles": {}}


def _summary_path(cache_dir: Path, kind: str, day: str) -> Path:
    """
    Where one slice's summary is kept.

    Args:
        cache_dir (Path): The cache folder.
        kind (str): "archive" or "live".
        day (str): The slice's day.

    Returns:
        Path: <cache_dir>/<config.EDA_SUMMARY_DIRNAME>/<kind>/<day>.json.
    """
    return Path(cache_dir) / config.EDA_SUMMARY_DIRNAME / kind / f"{day}{_SUMMARY_SUFFIX}"


def _write_summary(path: Path, summary: dict) -> None:
    """
    Save a summary atomically: a temporary file renamed over the old one.

    Args:
        path (Path): Where it goes.
        summary (dict): The summary.

    Raises:
        OSError: When it cannot be written; the temporary file is removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    try:
        tmp.write_text(json.dumps(summary, separators=(",", ":")))
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_summary(path: Path) -> dict | None:
    """
    A saved summary of the current version, or None.

    Args:
        path (Path): The saved summary.

    Returns:
        dict | None: The summary, or None when it cannot be read, is not a
            JSON object with a key, or was built under another
            config.EDA_SUMMARY_VERSION.
    """
    try:
        saved = json.loads(path.read_text())
        key = saved.get("key")
        if not isinstance(key, dict) or key.get("version") != config.EDA_SUMMARY_VERSION:
            return None
        return saved
    except Exception:
        return None


def _reusable(summary_path: Path, kind: str, day: str, slice_path: Path) -> dict | None:
    """
    The saved summary of a slice, when its key still matches the slice.

    Anything wrong with the saved summary (a damaged file, an odd shape)
    only means it is not reused.

    Args:
        summary_path (Path): The saved summary.
        kind (str): "archive" or "live".
        day (str): The slice's day.
        slice_path (Path): The slice.

    Returns:
        dict | None: The saved summary, or None when there is none, it cannot
            be read, or its kind, day, size, time, cutoff (when the slice's
            meta line can be read cheaply) or version differ.
    """
    try:
        saved = _read_summary(summary_path)
        if saved is None:
            return None
        key = saved["key"]
        size, mtime_ns = _slice_stat(slice_path)
        if (key.get("kind"), key.get("day"), key.get("size"),
                key.get("mtime_ns")) != (kind, day, size, mtime_ns):
            return None
        if kind == "archive":
            known, cutoff = _head_cutoff(slice_path)
            if known and cutoff != key.get("cutoff_ts"):
                return None
        return saved
    except Exception:
        return None


def _list_slices(cache_dir: Path) -> tuple[list[tuple[str, str, Path]], dict[str, str]]:
    """
    Every day slice in the cache, live days first, each kind newest day first.

    Args:
        cache_dir (Path): The cache folder.

    Returns:
        tuple[list, dict[str, str]]: (kind, day, path) per slice; and, per
            kind whose folder exists but cannot be listed, the reason. A
            missing folder lists nothing and is no error.
    """
    found = []
    errors: dict[str, str] = {}
    for kind, folder in SLICE_DIRS.items():
        try:
            with os.scandir(Path(cache_dir) / folder) as entries:
                names = [e.name for e in entries if e.name.endswith(_SLICE_SUFFIX)]
        except FileNotFoundError:
            names = []
        except OSError as exc:
            errors[kind] = f"{folder}/: {_one_line(exc)}"
            names = []
        for name in sorted(names, reverse=True):
            found.append((kind, name[: -len(_SLICE_SUFFIX)], Path(cache_dir) / folder / name))
    return found, errors


def _archive_cutoff(summaries) -> int | None:
    """
    The largest archive cutoff among readable archive summaries (M).

    Args:
        summaries (Iterable[dict]): Summaries.

    Returns:
        int | None: The cutoff in epoch seconds, or None when no readable
            archive summary records one.
    """
    cutoffs = [s["key"].get("cutoff_ts") for s in summaries
               if not s.get("unreadable") and (s.get("key") or {}).get("kind") == "archive"
               and isinstance(s["key"].get("cutoff_ts"), int)]
    return max(cutoffs) if cutoffs else None


def _live_counted(day: Any, cutoff: int | None) -> bool:
    """
    Whether a live day counts: it ends after the archive cutoff.

    Args:
        day (Any): The live day (ISO date).
        cutoff (int | None): The archive cutoff M, or None for none.

    Returns:
        bool: True when there is no archive cutoff, the day cannot be read,
            or the day ends after the cutoff.
    """
    end = _day_end(day)
    return cutoff is None or end is None or end > cutoff


def _tidy_summaries(cache_dir: Path, slices: list, list_errors: dict[str, str],
                    cutoff: int | None) -> tuple[list[dict], int]:
    """
    Handle the saved summaries whose slice is not in the cache.

    Leftover temporary files of an interrupted save are deleted. An archive
    summary whose slice is gone is deleted. A live summary whose slice is
    gone is kept, and returned to be counted, while its day still ends after
    the archive cutoff (the archive does not cover it yet, and only complete
    days were ever saved as slices); once the cutoff reaches the end of its
    day, or when it was built under another version, it is deleted. A kind
    whose slice folder could not be listed is left alone. Best effort: a
    file that cannot be deleted is left.

    Args:
        cache_dir (Path): The cache folder.
        slices (list): _list_slices' slices.
        list_errors (dict[str, str]): _list_slices' errors.
        cutoff (int | None): The archive cutoff M.

    Returns:
        tuple[list[dict], int]: The kept live summaries (each marked
            "slice_gone": True), and how many files were deleted.
    """
    wanted = {(kind, day) for kind, day, _ in slices}
    kept: list[dict] = []
    removed = 0
    for kind in SLICE_DIRS:
        if kind in list_errors:
            continue
        folder = Path(cache_dir) / config.EDA_SUMMARY_DIRNAME / kind
        try:
            with os.scandir(folder) as entries:
                names = [e.name for e in entries]
        except OSError:
            continue
        for name in names:
            path = folder / name
            if name.endswith(_TMP_SUFFIX):
                drop = True
            elif not name.endswith(_SUMMARY_SUFFIX):
                continue
            else:
                day = name[: -len(_SUMMARY_SUFFIX)]
                if (kind, day) in wanted:
                    continue
                drop = True
                if kind == "live" and _live_counted(day, cutoff):
                    saved = _read_summary(path)
                    if saved is not None and not saved.get("unreadable"):
                        saved["slice_gone"] = True
                        kept.append(saved)
                        drop = False
            if drop:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    return kept, removed


def _init_worker(cache_dir: str, candle_set: frozenset) -> None:
    """
    Load, once per worker process, what every slice's count needs.

    Args:
        cache_dir (str): The cache folder; the event titles are read from it here.
        candle_set (frozenset): Tickers with a candle file.
    """
    _WORKER_STATE.update(titles=load_event_titles(Path(cache_dir)), candle_set=candle_set)


def _worker_ready() -> bool:
    """
    A trivial task: it returns once a worker process has started.

    Returns:
        bool: Always True.
    """
    return True


def _summarise_task(path: str, kind: str) -> dict:
    """
    Count one slice in a worker process, with what _init_worker loaded.

    Args:
        path (str): The slice file.
        kind (str): "archive" or "live".

    Returns:
        dict: summarise_slice's summary.
    """
    return summarise_slice(Path(path), kind, titles=_WORKER_STATE["titles"],
                           candle_set=_WORKER_STATE["candle_set"])


def _failed(path: Path, kind: str, day: str, exc: BaseException) -> dict:
    """
    The summary of a slice whose count raised: nothing counted, the slice named.

    Args:
        path (Path): The slice.
        kind (str): "archive" or "live".
        day (str): Its day.
        exc (BaseException): What was raised.

    Returns:
        dict: An empty summary with "unreadable" set.
    """
    summary = _empty_summary(path, kind, day)
    summary["unreadable"] = f"{path.name}: {_one_line(exc)}"
    return summary


def load_or_build_summaries(cache_dir: Path, *, workers: int) -> list[dict]:
    """
    Every day slice's summary: the saved ones that still match, the rest built now.

    Lists archive_days/ and live_days/ under the cache (a missing folder is
    simply empty; one that cannot be listed is named in a summary of its
    own), reuses each saved summary whose key still matches its slice
    (_reusable), builds the others — in this process with workers=1,
    otherwise in up to min(workers, config.EDA_MAX_WORKERS) spawned worker
    processes — and saves each newly built summary atomically, from this
    process. A summary that names its slice as unreadable is returned but
    never saved, so the next run reads that slice again. Saved summaries of
    slices that are gone are handled by _tidy_summaries: a live day the
    archive does not yet cover is kept and returned (marked "slice_gone"),
    anything else is deleted. Never raises for one slice: a count that
    raises gives that slice an unreadable summary and the rest go on.

    With more than one worker, the worker processes are spawned, and a
    spawned process imports the caller's main module again: a script must
    call this under `if __name__ == "__main__":`. If the workers do not
    start within config.EDA_WORKER_START_TIMEOUT_SECONDS, or the pool cannot
    be made or breaks, its processes are killed and the remaining slices are
    counted in this process, with one WARNING.

    Args:
        cache_dir (Path): The cache folder (backtest_cache/).
        workers (int): The most worker processes; 1 counts in this process.

    Returns:
        list[dict]: One summary per slice or kept live day, live days first,
            each kind newest day first (the order does not depend on which
            worker finished first), then one per slice folder that could not
            be listed.
    """
    cache_dir = Path(cache_dir)
    slices, list_errors = _list_slices(cache_dir)
    results: dict[int, dict] = {}
    todo: list[int] = []
    for i, (kind, day, path) in enumerate(slices):
        saved = _reusable(_summary_path(cache_dir, kind, day), kind, day, path)
        if saved is not None:
            results[i] = saved
        else:
            todo.append(i)
    candle_set = candle_tickers(cache_dir) if todo else frozenset()

    def finished(i: int, summary: dict) -> None:
        """
        Keep one newly built summary, and save it unless it names its slice unreadable.

        Args:
            i (int): The slice's index in `slices`.
            summary (dict): Its summary.
        """
        kind, day, _ = slices[i]
        results[i] = summary
        if summary.get("unreadable"):
            logging.warning("Data page: %s slice %s", kind, summary["unreadable"])
            return
        try:
            _write_summary(_summary_path(cache_dir, kind, day), summary)
        except OSError as exc:
            logging.warning("Data page: the summary of %s slice %s could not be saved (%s); "
                            "it is built again next run", kind, day, _one_line(exc))

    def build_here(indexes: list[int]) -> None:
        """
        Count the given slices in this process.

        Args:
            indexes (list[int]): Indexes into `slices`.
        """
        titles = load_event_titles(cache_dir) if indexes else {}
        for i in indexes:
            kind, day, path = slices[i]
            try:
                summary = summarise_slice(path, kind, titles=titles, candle_set=candle_set)
            except Exception as exc:
                summary = _failed(path, kind, day, exc)
            finished(i, summary)

    pool_size = min(workers, config.EDA_MAX_WORKERS, len(todo))
    if pool_size <= 1:
        build_here(todo)
    else:
        build_here(_build_in_workers(slices, todo, pool_size, cache_dir, candle_set, finished))

    built = [results[i] for i in range(len(slices))]
    kept, removed = _tidy_summaries(cache_dir, slices, list_errors, _archive_cutoff(built))
    live = sorted([s for s in built if s["key"]["kind"] == "live"] + kept,
                  key=lambda s: str(s["key"].get("day")), reverse=True)
    archive = [s for s in built if s["key"]["kind"] != "live"]
    folders = []
    for kind, reason in list_errors.items():
        summary = _empty_summary(Path(SLICE_DIRS[kind]), kind, "")
        summary["unreadable"] = reason
        logging.warning("Data page: the %s slice folder could not be listed: %s", kind, reason)
        folders.append(summary)
    unreadable = sum(1 for s in built if s.get("unreadable"))
    logging.info("Data page: %d day slice summaries: %d reused, %d built, %d unreadable; "
                 "%d live days counted from a kept summary after their slice was deleted; "
                 "%d files tidied away", len(slices), len(slices) - len(todo),
                 len(todo) - unreadable, unreadable, len(kept), removed)
    return live + archive + folders


def _stop_pool(pool: ProcessPoolExecutor | None) -> None:
    """
    Kill a worker pool's processes and shut it down without waiting.

    Never raises: it runs on the way out of a pool that already failed.

    Args:
        pool (ProcessPoolExecutor | None): The pool, or None when it was never made.
    """
    if pool is None:
        return
    try:
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            process.kill()
    except Exception:
        pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _build_in_workers(slices, todo, pool_size, cache_dir, candle_set, finished) -> list[int]:
    """
    Count the listed slices across spawned worker processes.

    Waits up to config.EDA_WORKER_START_TIMEOUT_SECONDS for a worker to
    answer a trivial task before handing out the slices. A pool that cannot
    be made, workers that do not start in time and a pool that breaks (a
    worker died) all end the same way: the processes are killed, one
    WARNING names the cause, and the slices not yet counted are handed back
    for the caller to count in its own process. A slice whose own count
    raises is named unreadable, as in-process.

    Args:
        slices (list): _list_slices' slices.
        todo (list[int]): Indexes of the slices to count.
        pool_size (int): How many worker processes.
        cache_dir (Path): The cache folder.
        candle_set (frozenset): Tickers with a candle file.
        finished (Callable[[int, dict], None]): Called once per slice with its summary.

    Returns:
        list[int]: The indexes still to count; empty when the workers counted all.
    """
    remaining = list(todo)
    pool = None
    try:
        pool = ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(cache_dir), candle_set),
        )
        pool.submit(_worker_ready).result(timeout=config.EDA_WORKER_START_TIMEOUT_SECONDS)
        futures = {pool.submit(_summarise_task, str(slices[i][2]), slices[i][0]): i
                   for i in todo}
        for future in as_completed(futures):
            i = futures[future]
            kind, day, path = slices[i]
            try:
                summary = future.result()
            except BrokenProcessPool:
                raise
            except Exception as exc:
                summary = _failed(path, kind, day, exc)
            finished(i, summary)
            remaining.remove(i)
    except (TimeoutError, BrokenProcessPool, RuntimeError, OSError) as exc:
        logging.warning(
            "Data page: the worker processes could not be used (%s); counting the %d "
            "remaining day slices in this process instead. Worker processes are spawned "
            "and import the caller's main module again, so a script must build the "
            "summaries under `if __name__ == \"__main__\":`", _one_line(exc), len(remaining))
        _stop_pool(pool)
        return remaining
    pool.shutdown(wait=True)
    return []


def _add_counts(into: dict, more: dict) -> None:
    """
    Add one key's counters into another's, in place.

    Args:
        into (dict): The running counters (_new_counts' shape).
        more (dict): Counters to add.
    """
    for name, value in more.items():
        if isinstance(value, int) and not isinstance(value, bool):
            into[name] = into.get(name, 0) + value
        elif isinstance(value, list):
            have = into.setdefault(name, [0] * len(value))
            for j, v in enumerate(value):
                if j < len(have):
                    have[j] += v
                else:
                    have.append(v)
        elif isinstance(value, dict):
            have = into.setdefault(name, {})
            for k, v in value.items():
                have[k] = have.get(k, 0) + v


def _coverage(cutoff: int | None, live_days: list[str], skipped: int, gone: int) -> dict:
    """
    How the archive and the live days counted meet.

    Args:
        cutoff (int | None): The archive cutoff M.
        live_days (list[str]): The live days counted.
        skipped (int): Live days left out as covered by the archive.
        gone (int): Live days counted from a summary whose slice was deleted.

    Returns:
        dict: "archive_cutoff" (the cutoff's UTC date, or None),
            "first_live_day" and "last_live_day" (or None), "gap_days" (whole
            days between the cutoff and the first live day counted, 0 when
            they meet; None when either is missing), "live_days_skipped",
            "live_days_without_slice" and "note", one plain sentence.
    """
    first = min(live_days) if live_days else None
    last = max(live_days) if live_days else None
    gap = None
    if cutoff is not None and first is not None:
        start = _day_end(first)
        gap = max(0, (start - _SECONDS_PER_DAY - cutoff) // _SECONDS_PER_DAY) if start else None
    cutoff_day = _iso_day(cutoff)
    if cutoff_day is None:
        note = "No archive cutoff was recorded, so every live day is counted."
    elif first is None:
        note = f"The archive runs to {cutoff_day}; no live day is counted."
    elif gap:
        note = (f"Gap: the archive runs to {cutoff_day} but the first live day counted is "
                f"{first}; markets settled between them are not counted.")
    else:
        note = f"No gap: the archive runs to {cutoff_day} and the live days start {first}."
    return {"archive_cutoff": cutoff_day, "first_live_day": first, "last_live_day": last,
            "gap_days": gap, "live_days_skipped": skipped,
            "live_days_without_slice": gone, "note": note}


def merge(summaries: list[dict], *, series_categories: dict | None,
          candle_set: frozenset = frozenset()) -> dict:
    """
    Add every summary up into the one dict the data page draws.

    Each summary's "week|combo|series" counters are filed under
    "week|combo|category|tag" (historical.series_labels of the series, with
    infer_category as the fallback, from `series_categories`) and added. A
    live summary is counted only when its day ends after the largest
    readable archive cutoff (see the module Notes). The ladder days of one
    event are joined across slices, so the two rungs of one ladder listed in
    different slices make one ladder: an event with two or more distinct
    stated days, filed under the month ("YYYY-MM") of its earliest one. A
    candle ticker met in more than one slice keeps the first one in the
    summaries' order (load_or_build_summaries': live before archive, newest
    day first). Summaries naming an unreadable slice add nothing but their
    name.

    Args:
        summaries (list[dict]): load_or_build_summaries' result.
        series_categories (dict | None): Kalshi's /series listing
            (historical.load_series_categories), or None.
        candle_set (frozenset): The cache's candle tickers, to count those no
            summary joined to a market (see the module Notes' residuals).

    Returns:
        dict: "counts" (key -> counters), "ladders" (month -> {"events",
            "rungs"}), "ladder_events", "candles" (ticker -> [result,
            close_time, settlement_ts, category, tag, combo]),
            "candle_files" and "candles_unjoined", "slices" ({"live": n,
            "archive": n}), "records", "counted", "dropped_after_cutoff",
            "archive_cutoffs" (every distinct archive cutoff seen, ascending),
            "coverage" (_coverage's dict) and "unreadable" (each unreadable
            slice or folder, named).
    """
    cutoff = _archive_cutoff(summaries)
    labels_by_series: dict[str, tuple[str, str]] = {}

    def labels(series: str) -> tuple[str, str]:
        """
        A series' category and tag, worked out once per series.

        Args:
            series (str): The series ticker.

        Returns:
            tuple[str, str]: (category, tag), each safe as a key part.
        """
        found = labels_by_series.get(series)
        if found is None:
            category, tag = historical.series_labels(
                series, historical.infer_category(series), series_categories)
            found = labels_by_series[series] = (_label(category), _label(tag))
        return found

    counts: dict[str, dict] = {}
    days_by_event: dict[str, set] = {}
    candles: dict[str, list] = {}
    slices = dict.fromkeys(SLICE_DIRS, 0)
    records = counted = dropped = skipped = gone = 0
    live_days: list[str] = []
    cutoffs: set = set()
    unreadable: list[str] = []
    for summary in summaries:
        key = summary.get("key") or {}
        kind = key.get("kind")
        if kind in slices and key.get("day"):
            slices[kind] += 1
        if summary.get("unreadable"):
            unreadable.append(f"{kind}: {summary['unreadable']}")
            continue
        if kind == "live":
            if not _live_counted(key.get("day"), cutoff):
                skipped += 1
                continue
            live_days.append(str(key.get("day")))
            gone += bool(summary.get("slice_gone"))
        records += summary.get("records", 0)
        counted += summary.get("counted", 0)
        dropped += summary.get("dropped_after_cutoff", 0)
        if kind == "archive" and key.get("cutoff_ts") is not None:
            cutoffs.add(key["cutoff_ts"])
        for name, more in (summary.get("counts") or {}).items():
            week, combo, series = (name.split(KEY_SEPARATOR, 2) + ["", ""])[:3]
            merged_key = KEY_SEPARATOR.join((week, combo, *labels(series)))
            _add_counts(counts.setdefault(merged_key, _new_counts()), more)
        for event, days in (summary.get("ladders") or {}).items():
            days_by_event.setdefault(event, set()).update(days)
        for ticker, info in (summary.get("candles") or {}).items():
            if ticker not in candles:
                result, close_time, settle, series, combo = info
                candles[ticker] = [result, close_time, settle, *labels(series), combo]
    ladders: dict[str, dict] = {}
    ladder_events = 0
    for days in days_by_event.values():
        if len(days) < 2:
            continue
        ladder_events += 1
        month = min(days)[:7]
        entry = ladders.setdefault(month, {"events": 0, "rungs": 0})
        entry["events"] += 1
        entry["rungs"] += len(days)
    return {
        "counts": counts,
        "ladders": dict(sorted(ladders.items())),
        "ladder_events": ladder_events,
        "candles": candles,
        "candle_files": len(candle_set),
        "candles_unjoined": sum(1 for t in candle_set if t not in candles),
        "slices": slices,
        "records": records,
        "counted": counted,
        "dropped_after_cutoff": dropped,
        "archive_cutoffs": sorted(cutoffs),
        "coverage": _coverage(cutoff, live_days, skipped, gone),
        "unreadable": unreadable,
    }


# ─── Candle calibration ───────────────────────────────────────────────────────

# The per-file candle facts and the saved k view live in the summary folder
# under these names (<cache>/<config.EDA_SUMMARY_DIRNAME>/<name>).
_CANDLE_CACHE_NAME = "candles.json"
_KHAT_NAME = "khat.json"
# Added to a price before it is floored onto the bin grid, so that a price on a
# bin edge (0.30 / 0.05 is 5.999999999999999) lands in the bin above the edge.
_BIN_ROUNDING_DIGITS = 9


def _bin_count(width: float) -> int:
    """
    How many bins of a given width cover the dollar range 0 to 1.

    Args:
        width (float): The bin width in dollars (for example 0.05).

    Returns:
        int: round(1 / width), at least 1.
    """
    return max(1, round(1.0 / width))


def _bin_index(value: float, width: float, count: int) -> int:
    """
    The bin a price falls in.

    Args:
        value (float): The price, in dollars.
        width (float): The bin width.
        count (int): How many bins there are.

    Returns:
        int: floor(value / width), held to 0 .. count - 1 (a price of exactly 1
            goes in the last bin). The division is rounded to nine decimals
            first, so a price on a bin edge goes in the bin above it.
    """
    return min(count - 1, max(0, math.floor(round(value / width, _BIN_ROUNDING_DIGITS))))


def _finite(value: Any) -> float | None:
    """
    A number as a float, None when it is not a finite real number.

    Args:
        value (Any): Anything.

    Returns:
        float | None: float(value) for an int or float that is finite (a bool
            is not a number here), otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _candle_prices(candle: Any) -> tuple[float | None, float | None]:
    """
    The two prices one candle gives: the YES ask, and the midpoint.

    The YES ask is what betting Yes costs. The midpoint is halfway between the
    YES ask and the YES bid (1 minus the NO ask). Each ask counts only when
    historical.usable_candle_ask accepts it (strictly between 0 and 1, and a
    NO ask also below the 0.99 clamp that stands for an empty book). A long
    shot's YES bid is under a cent, so its NO ask sits at the clamp: it still
    has a YES ask but no midpoint.

    Args:
        candle (Any): One candle of a candle file.

    Returns:
        tuple[float | None, float | None]: (YES ask, midpoint); each None when
            not usable. A midpoint is never given without a YES ask.
    """
    if not isinstance(candle, dict):
        return None, None
    yes = historical.usable_candle_ask(candle.get("yes_ask_close"), "yes")
    if math.isnan(yes):
        return None, None
    no = historical.usable_candle_ask(candle.get("no_ask_close"), "no")
    if math.isnan(no):
        return yes, None
    return yes, (yes + (1.0 - no)) / 2.0


def _read_candle_file(path: Path, close_epoch: float | None, leads: tuple) -> dict:
    """
    Read one candle file and work out everything the page needs from it.

    For each lead, the candle used is the last one whose end "ts" is at or
    before close - lead; when that candle ended more than
    config.EDA_PRICE_MAX_AGE_HOURS before that moment the lead is "stale" and
    has no price.

    Args:
        path (Path): The candle file (historical.fetch_candlesticks' format:
            {"open_ts", "close_ts", "period_interval", "fields", "candles"}; a
            bare list is the oldest format; a file with no "fields" key
            predates the traded volume and counts as field version 1).
        close_epoch (float | None): The market's close time in epoch seconds,
            or None when unknown (then no lead has a price).
        leads (tuple): Hours before close at which a price is read.

    Returns:
        dict: "f" (field version: the file's "fields" if a whole number, 1
            when absent, 0 for anything else), "e" (True when the file holds
            no candle), "vol" (the traded volume summed over the candles that
            state one, None when none does), "asks" and "mids" (per lead, the
            candle's YES ask and midpoint, or None) and "stale" (per lead,
            True when the last candle at or before the lead's moment is too
            old to use).

    Raises:
        Exception: When the file cannot be read or is not a candle file; the
            caller counts it as unreadable.
    """
    data = json.loads(path.read_bytes())
    if isinstance(data, list):
        candles, fields = data, 1
    elif isinstance(data, dict) and isinstance(data.get("candles"), list):
        candles = data["candles"]
        raw = data.get("fields", 1)
        fields = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
    else:
        raise ValueError("not a candle file")
    targets = [None if close_epoch is None else close_epoch - lead * _SECONDS_PER_HOUR
               for lead in leads]
    best_ts: list[float | None] = [None] * len(leads)
    best: list[Any] = [None] * len(leads)
    volume = 0.0
    has_volume = False
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        number = _finite(candle.get("volume"))
        if number is not None:
            volume += number
            has_volume = True
        ts = _finite(candle.get("ts"))
        if ts is None:
            continue
        for i, target in enumerate(targets):
            if target is not None and ts <= target and (best_ts[i] is None or ts >= best_ts[i]):
                best_ts[i], best[i] = ts, candle
    max_age = config.EDA_PRICE_MAX_AGE_HOURS * _SECONDS_PER_HOUR
    stale = [best_ts[i] is not None and best_ts[i] <= targets[i] - max_age
             for i in range(len(leads))]
    prices = [(None, None) if stale[i] else _candle_prices(best[i]) for i in range(len(leads))]
    return {"f": fields, "e": not candles, "vol": volume if has_volume else None,
            "asks": [p[0] for p in prices], "mids": [p[1] for p in prices], "stale": stale}


def _load_candle_cache(path: Path, leads: tuple) -> dict:
    """
    The saved per-file candle facts, or nothing when they cannot be trusted.

    Args:
        path (Path): <cache>/eda_days/candles.json.
        leads (tuple): The leads the facts must have been worked out for.

    Returns:
        dict: file name -> facts; empty when the file is absent, damaged, of
            another config.EDA_CANDLE_FACTS_VERSION or built for other leads
            or another config.EDA_PRICE_MAX_AGE_HOURS.
    """
    try:
        saved = json.loads(path.read_text())
        if (saved.get("version") != config.EDA_CANDLE_FACTS_VERSION
                or saved.get("leads") != list(leads)
                or saved.get("max_age_hours") != config.EDA_PRICE_MAX_AGE_HOURS
                or not isinstance(saved.get("files"), dict)):
            return {}
        return saved["files"]
    except Exception:
        return {}


def _write_candle_cache(path: Path, leads: tuple, files: dict) -> None:
    """
    Save the per-file candle facts atomically, logging (never raising) a failure.

    Args:
        path (Path): <cache>/eda_days/candles.json.
        leads (tuple): The leads the facts were worked out for.
        files (dict): file name -> facts.
    """
    try:
        _write_summary(path, {"version": config.EDA_CANDLE_FACTS_VERSION, "leads": list(leads),
                              "max_age_hours": config.EDA_PRICE_MAX_AGE_HOURS, "files": files})
    except (OSError, ValueError, TypeError) as exc:
        logging.warning("Data page: the candle summary could not be saved (%s); the candle "
                        "files are read again next run", _one_line(exc))


def _is_int(value: Any) -> bool:
    """
    Whether a value is a whole number (a bool is not).

    Args:
        value (Any): Anything.

    Returns:
        bool: True for an int that is not a bool.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _price_list(value: Any, count: int) -> bool:
    """
    Whether a value is a list of `count` prices, each None or a finite number in (0, 1).

    Args:
        value (Any): A saved list of prices.
        count (int): How many leads there are.

    Returns:
        bool: True when it is well formed.
    """
    if not isinstance(value, list) or len(value) != count:
        return False
    for item in value:
        if item is None:
            continue
        number = _finite(item)
        if number is None or not 0.0 < number < 1.0:
            return False
    return True


def _fact_matches(fact: Any, size: int, mtime_ns: int, close: Any, leads: tuple) -> bool:
    """
    Whether a saved per-file fact still describes the file and the close time used.

    A fact is also refused when it is not well formed (a hand-edited or
    damaged entry), so it is simply read again rather than raising later.

    Args:
        fact (Any): The saved entry.
        size (int): The file's size now.
        mtime_ns (int): Its modification time now.
        close (Any): The market's close time text now (None when unjoined).
        leads (tuple): The leads in force.

    Returns:
        bool: True when size, time, close time and version all match and the
            entry is well formed: "f" a whole number, "e" a bool, "vol" None
            or a finite number, "asks" and "mids" lists of one price (None or
            a finite number in (0, 1)) per lead, "stale" a list of one bool
            per lead.
    """
    if not isinstance(fact, dict):
        return False
    if not (fact.get("s") == size and fact.get("m") == mtime_ns and fact.get("c") == close
            and fact.get("v") == config.EDA_CANDLE_FACTS_VERSION):
        return False
    count = len(leads)
    stale = fact.get("stale")
    return (_is_int(fact.get("f")) and isinstance(fact.get("e"), bool)
            and "vol" in fact and (fact["vol"] is None or _finite(fact["vol"]) is not None)
            and _price_list(fact.get("asks"), count) and _price_list(fact.get("mids"), count)
            and isinstance(stale, list) and len(stale) == count
            and all(isinstance(x, bool) for x in stale))


def _new_cell(nbins: int) -> dict:
    """
    An empty calibration cell.

    Args:
        nbins (int): How many price bins.

    Returns:
        dict: {"bins": [[markets, yes], ...], "n": 0, "brier_sum": 0.0}.
    """
    return {"bins": [[0, 0] for _ in range(nbins)], "n": 0, "brier_sum": 0.0}


def _add_price(cell: dict, price: float, outcome: float, width: float) -> None:
    """
    File one market's price and outcome into a calibration cell.

    Args:
        cell (dict): A _new_cell.
        price (float): The price, in dollars.
        outcome (float): 1.0 if the market settled Yes, 0.0 if No.
        width (float): The bin width.
    """
    slot = cell["bins"][_bin_index(price, width, len(cell["bins"]))]
    slot[0] += 1
    slot[1] += int(outcome)
    cell["n"] += 1
    cell["brier_sum"] += (price - outcome) ** 2


def summarise_candles(cache_dir: Path, candles: dict) -> dict:
    """
    Calibrate the cached candles against how their markets settled.

    Lists <cache_dir>/candlesticks/ once with os.scandir, reads each file
    whose saved facts no longer match (or has none), and adds up, per market
    that merge() joined (`candles`: ticker -> [result, close_time,
    settlement_ts, category, tag, combo]) and is not a combo, for each lead in
    config.EDA_PRICE_LEADS_HOURS: the YES ask (the primary price: what betting
    Yes costs) and the midpoint (only where the NO side is usable too) of the
    last candle at or before close - lead, used only when it ended within
    config.EDA_PRICE_MAX_AGE_HOURS of that moment. Each price is filed in a
    bin of config.EDA_CALIBRATION_BIN dollars with how many of its markets
    settled Yes, and its Brier sum (price - outcome)^2 is kept, for a market
    whose result is "yes" or "no". The traded volume per category comes from
    files at config.CANDLESTICK_CACHE_FIELDS_VERSION only. Every file, joined
    or not, is counted by field version, and as empty or unreadable. A file
    that cannot be read is counted and skipped; this never raises for one.

    Args:
        cache_dir (Path): The cache folder (backtest_cache/).
        candles (dict): merge()'s "candles" map; the close time is its
            second entry (an ISO text), the category its fourth, the combo
            flag its sixth.

    Returns:
        dict: "leads" (the hours), "bin_width", "bins" (the bin count),
            "calibration" (str(lead) -> category -> {"ask": cell, "mid": cell},
            a cell being {"bins": [[markets, yes], ...], "n": markets priced,
            "brier_sum": float}; a category appears at a lead once one of its
            markets has an ask there), "coverage" (str(lead) -> {"markets":
            joined non-combo markets with a yes or no result, "no_ask": of
            them with no YES ask price at the lead, "no_mid": with no
            midpoint, "stale": whose last candle was too old}), "volume"
            (category -> contracts), "volume_files" (files that added to it),
            "files" (candle files seen), "by_fields" (field version as text,
            "0" for an odd one -> files), "empty", "unreadable", "unjoined"
            (files whose ticker no summary joined), "combos_left_out" (joined
            combos not calibrated), "read" and "reused" (files read now, and
            kept from the saved facts), and "folder_error" (None, or why the
            candle folder could not be listed).
    """
    cache_dir = Path(cache_dir)
    leads = tuple(config.EDA_PRICE_LEADS_HOURS)
    width = config.EDA_CALIBRATION_BIN
    nbins = _bin_count(width)
    folder = cache_dir / historical._CANDLES_DIR.name
    cache_path = cache_dir / config.EDA_SUMMARY_DIRNAME / _CANDLE_CACHE_NAME
    saved = _load_candle_cache(cache_path, leads)
    fresh: dict[str, dict] = {}
    calibration: dict[str, dict[str, dict]] = {str(lead): {} for lead in leads}
    coverage = {str(lead): {"markets": 0, "no_ask": 0, "no_mid": 0, "stale": 0}
                for lead in leads}
    volume: dict[str, float] = {}
    out = {"leads": list(leads), "bin_width": width, "bins": nbins,
           "calibration": calibration, "coverage": coverage, "volume": volume,
           "volume_files": 0, "files": 0, "by_fields": {}, "empty": 0, "unreadable": 0,
           "unjoined": 0, "combos_left_out": 0, "read": 0, "reused": 0, "folder_error": None}
    current = config.CANDLESTICK_CACHE_FIELDS_VERSION
    try:
        scan = os.scandir(folder)
    except FileNotFoundError:
        return out
    except OSError as exc:
        out["folder_error"] = f"{folder.name}/: {_one_line(exc)}"
        return out
    with scan:
        for entry in scan:
            name = entry.name
            if not name.endswith(_CANDLE_SUFFIX):
                continue
            try:
                if not entry.is_file():
                    continue
                st = entry.stat()
            except OSError:
                out["files"] += 1
                out["unreadable"] += 1
                continue
            out["files"] += 1
            info = candles.get(name[: -len(_CANDLE_SUFFIX)])
            close = info[1] if info and isinstance(info[1], str) else None
            fact = saved.get(name)
            if _fact_matches(fact, st.st_size, st.st_mtime_ns, close, leads):
                out["reused"] += 1
            else:
                try:
                    fact = _read_candle_file(Path(entry.path), historical._iso_epoch(close),
                                             leads)
                except Exception:
                    out["unreadable"] += 1
                    continue
                fact.update(s=st.st_size, m=st.st_mtime_ns, c=close,
                            v=config.EDA_CANDLE_FACTS_VERSION)
                out["read"] += 1
            fresh[name] = fact
            fields_key = str(fact["f"])
            out["by_fields"][fields_key] = out["by_fields"].get(fields_key, 0) + 1
            out["empty"] += bool(fact["e"])
            if info is None:
                out["unjoined"] += 1
                continue
            if info[5]:
                out["combos_left_out"] += 1
                continue
            category = info[3]
            if fact["f"] == current and fact["vol"] is not None:
                volume[category] = volume.get(category, 0.0) + fact["vol"]
                out["volume_files"] += 1
            outcome = 1.0 if info[0] == "yes" else 0.0 if info[0] == "no" else None
            if outcome is None:
                continue
            for i, lead in enumerate(leads):
                seen = coverage[str(lead)]
                seen["markets"] += 1
                seen["stale"] += bool(fact["stale"][i])
                ask, mid = fact["asks"][i], fact["mids"][i]
                seen["no_ask"] += ask is None
                seen["no_mid"] += mid is None
                if ask is None and mid is None:
                    continue
                cell = calibration[str(lead)].setdefault(
                    category, {"ask": _new_cell(nbins), "mid": _new_cell(nbins)})
                if ask is not None:
                    _add_price(cell["ask"], ask, outcome, width)
                if mid is not None:
                    _add_price(cell["mid"], mid, outcome, width)
    if out["read"] or set(fresh) != set(saved):
        _write_candle_cache(cache_path, leads, fresh)
    logging.info("Data page: %d candle files: %d read, %d reused, %d unreadable, %d with no "
                 "joined market", out["files"], out["read"], out["reused"], out["unreadable"],
                 out["unjoined"])
    return out


# ─── Inventory of the cache folder ────────────────────────────────────────────

# The word for an entry this module has no description of (a file or folder
# another part of the project added): it is listed, never skipped.
INVENTORY_NOT_DESCRIBED = "not described"
# One plain line per known top-level entry of the cache folder, shown by the page.
INVENTORY_DESCRIPTIONS = {
    "archive_days": ("Settled markets from Kalshi's archive, one file per day the "
                     "market was created, kept only up to the archive cutoff."),
    "live_days": ("Markets settled since the archive cutoff, one file per settlement day, "
                  "from Kalshi's live listing."),
    "candlesticks": ("Hourly price candles (YES and NO ask, hourly volume) for the markets "
                     "the backtest priced, one file per market."),
    config.DEPTH_SNAPSHOTS_DIRNAME: ("Saved copies of live order books, used to model how many "
                                     "contracts rest near the best price."),
    config.EDA_SUMMARY_DIRNAME: ("This page's own small summaries of the day slices and "
                                 "candles, and the saved k view."),
    "event_titles_v2.json": ("Event titles looked up for event tickers, so markets can be "
                             "grouped by event."),
    "event_titles.json": ("The older event title file, read once to bring its titles into "
                          "event_titles_v2.json; no longer read."),
    "event_competitions.json": ("The competition each event names (a league, for example), "
                                "used by the same-title sport check."),
    "series_categories.json": "Kalshi's category and tags for each series, refreshed weekly.",
    "treasury_bill_rates.json": ("The 8-week Treasury bill auction yields the dashboards "
                                 "subtract from Sharpe and Sortino."),
    "live_marks": ("Daily prices of finalized markets the Live trading tab has read, kept so "
                   "they are not fetched again."),
}
# Assembled corpora: settled_markets_<start date>[...].jsonl.gz, and the older
# single-document form of the same.
INVENTORY_CORPUS_DESCRIPTION = ("An assembled set of settled markets for one backtest start "
                                "date, already filtered to markets that could be traded.")
INVENTORY_LEGACY_CORPUS_DESCRIPTION = ("The older single-document form of an assembled set "
                                       "of settled markets; no longer read.")
_CORPUS_PREFIX = "settled_markets_"
_CORPUS_SUFFIX = ".jsonl.gz"
_LEGACY_CORPUS_SUFFIX = ".json"
# The meta keys of an assembled corpus the page shows.
_CORPUS_META_KEYS = ("start_date", "prefilter_tag", "assembled_at", "assembly_counts")
# A day file's name: "<YYYY-MM-DD>.json" or "<YYYY-MM-DD>.json.gz".
_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json(?:\.gz)?$")
# A depth snapshot's name: the UTC time it was taken.
_SNAPSHOT_FILE = re.compile(r"^(\d{8}T\d{6}Z)\.json\.gz$")


def _inventory_row(name: str, kind: str, **fields: Any) -> dict:
    """
    One inventory row with every field present.

    Args:
        name (str): The entry's name.
        kind (str): "folder" or "file".
        **fields: Any of files, bytes, first_day, last_day, description,
            detail, error.

    Returns:
        dict: name, kind, files (file count; 1 for a file), bytes, first_day,
            last_day (the ISO dates of the oldest and newest day file inside
            a folder, else None), description (a plain line, or
            INVENTORY_NOT_DESCRIBED), detail (a dict of extra facts, or None)
            and error (None, or why the entry could not be read).
    """
    row = {"name": name, "kind": kind, "files": None, "bytes": None, "first_day": None,
           "last_day": None, "description": INVENTORY_NOT_DESCRIBED, "detail": None,
           "error": None}
    row.update(fields)
    return row


def _walk_folder(path: str, top_names: list | None = None
                 ) -> tuple[int, int, str | None, str | None, int]:
    """
    Count the files in a folder and all below it, one os.scandir pass per folder.

    Args:
        path (str): The folder.
        top_names (list | None): When given, the names of the files directly in
            the folder are appended to it (so one pass also serves the depth
            snapshot count).

    Returns:
        tuple: (files, bytes, oldest day file's ISO date, newest, folders
            that could not be listed below the first). Symbolic links are not
            followed. Day files are those named <YYYY-MM-DD>.json[.gz].

    Raises:
        OSError: When the folder itself cannot be listed.
    """
    files = total = skipped = 0
    first = last = None
    stack = [(path, True)]
    while stack:
        current, top = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((entry.path, False))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        skipped += 1
                        continue
                    files += 1
                    if top and top_names is not None:
                        top_names.append(entry.name)
                    match = _DAY_FILE.match(entry.name)
                    if match:
                        day = match.group(1)
                        first = day if first is None or day < first else first
                        last = day if last is None or day > last else last
        except OSError:
            if top:
                raise
            skipped += 1
    return files, total, first, last, skipped


def _corpus_meta(path: str) -> dict:
    """
    The meta block of an assembled corpus, read from its first line only.

    Args:
        path (str): The settled_markets_*.jsonl.gz file.

    Returns:
        dict: The keys of _CORPUS_META_KEYS that the meta block has, or
            {"meta_error": "<one-line reason>"} when the first line cannot be
            read or is not a meta line.
    """
    try:
        with gzip.open(path, "rb") as fh:
            line = fh.readline(_META_LINE_MAX_BYTES)
        head = json.loads(line)
        meta = head.get("meta") if isinstance(head, dict) else None
        if not isinstance(meta, dict):
            return {"meta_error": "the first line is not a meta block"}
        return {key: meta[key] for key in _CORPUS_META_KEYS if key in meta}
    except Exception as exc:
        return {"meta_error": _one_line(exc)}


def _snapshot_detail(names: list) -> dict:
    """
    How many depth snapshots a folder holds and when the newest was taken.

    Args:
        names (list): The names of the files directly in the depth snapshot
            folder (collected by _walk_folder's one pass).

    Returns:
        dict: {"snapshots": count of <UTC time>.json.gz files, "newest": the
            newest one's UTC time (from its file name) as an ISO text, or
            None}.
    """
    stamps = [m.group(1) for m in map(_SNAPSHOT_FILE.match, names) if m]
    newest = None
    if stamps:
        try:
            newest = datetime.strptime(max(stamps), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=UTC).isoformat()
        except ValueError:
            newest = None
    return {"snapshots": len(stamps), "newest": newest}


def _describe(name: str, is_dir: bool) -> str:
    """
    The plain line for a known top-level entry.

    Args:
        name (str): The entry's name.
        is_dir (bool): Whether it is a folder.

    Returns:
        str: Its description, or INVENTORY_NOT_DESCRIBED.
    """
    if name in INVENTORY_DESCRIPTIONS:
        return INVENTORY_DESCRIPTIONS[name]
    if not is_dir and name.startswith(_CORPUS_PREFIX):
        if name.endswith(_CORPUS_SUFFIX):
            return INVENTORY_CORPUS_DESCRIPTION
        if name.endswith(_LEGACY_CORPUS_SUFFIX):
            return INVENTORY_LEGACY_CORPUS_DESCRIPTION
    return INVENTORY_NOT_DESCRIBED


def inventory(cache_dir: Path) -> list[dict]:
    """
    One row per top-level entry of the cache folder.

    A folder is walked with one os.scandir pass per folder (the candle folder,
    over 100,000 files, is one pass). For each assembled corpus
    (settled_markets_*.jsonl.gz) only the first line is read, for its meta
    block; for the depth snapshot folder the snapshots are counted and the
    newest one's time is taken from its file name. An entry this module does
    not know is listed with INVENTORY_NOT_DESCRIBED, never skipped, and an
    entry that cannot be read gets a row whose "error" says so. A name that
    starts with "." (.DS_Store, a tool's hidden folder) is not part of the
    cache and is left out. Never raises.

    Args:
        cache_dir (Path): The cache folder (backtest_cache/).

    Returns:
        list[dict]: _inventory_row rows, sorted by name; a single row naming
            the cache folder with an "error" when the folder cannot be listed.
    """
    cache_dir = Path(cache_dir)
    try:
        with os.scandir(cache_dir) as listing:
            entries = sorted(listing, key=lambda e: e.name)
    except OSError as exc:
        return [_inventory_row(cache_dir.name or str(cache_dir), "folder",
                               error=_one_line(exc))]
    rows = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            rows.append(_row_for(entry))
        except Exception as exc:
            rows.append(_inventory_row(entry.name, "file", error=_one_line(exc)))
    return rows


def _row_for(entry: os.DirEntry) -> dict:
    """
    The inventory row of one top-level entry.

    Args:
        entry (os.DirEntry): The entry.

    Returns:
        dict: Its row; a folder or file that cannot be read has "error" set.
    """
    name = entry.name
    if entry.is_dir(follow_symlinks=False):
        row = _inventory_row(name, "folder", description=_describe(name, True))
        depth = name == config.DEPTH_SNAPSHOTS_DIRNAME
        top_names: list = []
        try:
            files, total, first, last, skipped = _walk_folder(
                entry.path, top_names if depth else None)
        except OSError as exc:
            row["error"] = _one_line(exc)
            return row
        row.update(files=files, bytes=total, first_day=first, last_day=last)
        detail: dict = {}
        if skipped:
            detail["unreadable_parts"] = skipped
        if depth:
            detail.update(_snapshot_detail(top_names))
        row["detail"] = detail or None
        return row
    row = _inventory_row(name, "file", files=1, description=_describe(name, False))
    try:
        row["bytes"] = entry.stat(follow_symlinks=False).st_size
    except OSError as exc:
        row["error"] = _one_line(exc)
        return row
    if name.startswith(_CORPUS_PREFIX) and name.endswith(_CORPUS_SUFFIX):
        row["detail"] = _corpus_meta(entry.path)
    return row


# ─── The k view ───────────────────────────────────────────────────────────────

def _khat_stat(label: str, observations: list, backtester: Any, dashboard: Any) -> dict:
    """
    Reduce a group of k-hat observations to numbers.

    The arithmetic is the backtest page's own: backtester._calibration_bucket
    for the rate, the mean mid spread and k-hat, and dashboard._khat_interval
    over dashboard._khat_cluster_sums for the interval (each event counted
    once), so a group holding a band's whole population reproduces its card.

    Args:
        label (str): The group's name.
        observations (list): backtester.CalibrationObservation records.
        backtester (Any): The backtester module (imported by the caller).
        dashboard (Any): The dashboard module (imported by the caller).

    Returns:
        dict: "label", "n", "events" (distinct market-A events), "rate"
            (realised in-between rate), "implied" (mean mid spread), "k" (k-hat,
            None when the mean mid spread is not above 0), "ci_low" and
            "ci_high" (the 95% interval, None when there is none); a number
            that is not finite is None.
    """
    bucket = backtester._calibration_bucket(label, 0.0, observations)
    events = len({o.event_ticker for o in observations})
    gaps = sum(o.implied for o in observations)
    interval = dashboard._khat_interval(bucket.empirical_k, gaps,
                                        dashboard._khat_cluster_sums(observations), events)
    return {"label": label, "n": bucket.n, "events": events,
            "rate": _finite(bucket.realised_rate), "implied": _finite(bucket.mean_implied),
            "k": _finite(bucket.empirical_k),
            "ci_low": _finite(interval[0]) if interval else None,
            "ci_high": _finite(interval[1]) if interval else None}


def khat_view(sweep: Any, *, series_categories: dict | None = None) -> dict | None:
    """
    The figures of the k view, from a backtest's own k-hat observations.

    Uses the primary band's calibration (dashboard._primary_calibration, the
    backtest page's one reading of it) and its carried
    observations: one per time-series entry with a binary settlement. Returns
    mid-spread bins (config.EDA_KHAT_SPREAD_BIN wide, entries with a mid spread
    above 0; one that is not above 0, a NaN included, is counted in `non_positive`) with how many entries and how often they settled in between;
    the pooled k-hat with its confidence interval (config.KHAT_CONFIDENCE_LEVEL)
    next to the run's k (sweep.primary.k); and k-hat with its interval by
    deadline-gap band and by Kalshi category (filed as the backtest page files
    trades: historical.series_labels over `series_categories`, the ticker-prefix
    category when a series is missing or no listing is given).

    backtester and dashboard are imported inside this function, so the worker
    processes that count day slices (which import this module) never load
    pandas or plotly.

    Args:
        sweep (Any): A backtester.BacktestSweep, or None.
        series_categories (dict | None): Kalshi's /series listing, or None.

    Returns:
        dict | None: "k", "confidence_level", "spread_bin", "observations",
            "non_positive" (entries whose mid spread is not above 0, NaN
            included, left out of the bins), "excluded_premise_violations",
            "pooled" (a _khat_stat), "bins" (non-empty ones, ascending:
            "lo", "hi", "n", "in_between", "rate", "implied"), "gap_bands"
            and "categories" (lists of _khat_stat). None when the sweep has no
            calibration or it carries no observation.
    """
    if sweep is None:
        return None
    from . import backtester, dashboard  # local: keeps pandas and plotly out of the workers

    calibration = dashboard._primary_calibration(sweep)
    observations = tuple(getattr(calibration, "observations", None) or ())
    if not observations:
        return None

    width = config.EDA_KHAT_SPREAD_BIN
    nbins = _bin_count(width)
    slots: dict[int, list] = {}
    non_positive = 0
    for o in observations:
        if not o.implied > 0:
            non_positive += 1
            continue
        slot = slots.setdefault(_bin_index(o.implied, width, nbins), [0, 0, 0.0])
        slot[0] += 1
        slot[1] += int(bool(o.in_between))
        slot[2] += o.implied
    bins = [{"lo": round(i * width, 9), "hi": round((i + 1) * width, 9), "n": n,
             "in_between": hits, "rate": hits / n, "implied": gap / n}
            for i, (n, hits, gap) in sorted(slots.items())]
    bands = []
    for lo, hi in backtester._CALIBRATION_GAP_BANDS:
        members = [o for o in observations if o.gap_days is not None and lo <= o.gap_days <= hi]
        if members:
            bands.append(_khat_stat(f"{lo}-{hi}d", members, backtester, dashboard))
    by_category: dict[str, list] = {}
    for o in observations:
        category, _ = historical.series_labels(o.event_ticker, o.category, series_categories)
        by_category.setdefault(category, []).append(o)
    categories = [_khat_stat(name, members, backtester, dashboard)
                  for name, members in sorted(by_category.items())]
    return {
        "k": _finite(getattr(getattr(sweep, "primary", None), "k", None)),
        "confidence_level": config.KHAT_CONFIDENCE_LEVEL,
        "spread_bin": width,
        "observations": len(observations),
        "non_positive": non_positive,
        "excluded_premise_violations": getattr(calibration, "excluded_premise_violations", None),
        "pooled": _khat_stat("POOLED", list(observations), backtester, dashboard),
        "bins": bins,
        "gap_bands": bands,
        "categories": categories,
    }


def _khat_path(cache_dir: Path) -> Path:
    """
    Where the saved k view is kept.

    Args:
        cache_dir (Path): The cache folder.

    Returns:
        Path: <cache_dir>/<config.EDA_SUMMARY_DIRNAME>/khat.json.
    """
    return Path(cache_dir) / config.EDA_SUMMARY_DIRNAME / _KHAT_NAME


def save_khat(cache_dir: Path, view: dict) -> None:
    """
    Save the k view atomically as strict JSON.

    Args:
        cache_dir (Path): The cache folder.
        view (dict): khat_view's result.

    Raises:
        ValueError: When the view holds a NaN or infinite number (strict JSON
            has none); nothing is written and any earlier file stays.
        OSError: When the file cannot be written.
    """
    text = json.dumps(view, allow_nan=False, separators=(",", ":"))
    path = _khat_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _refuse_constant(name: str) -> None:
    """
    json.loads' hook for NaN and Infinity, which strict JSON does not have.

    Args:
        name (str): The constant's text.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"not strict JSON: {name}")


def load_khat(cache_dir: Path) -> dict | None:
    """
    The saved k view.

    Args:
        cache_dir (Path): The cache folder.

    Returns:
        dict | None: The view, or None when the file is missing, cannot be
            read, is not strict JSON or is not a JSON object.
    """
    try:
        view = json.loads(_khat_path(cache_dir).read_text(), parse_constant=_refuse_constant)
    except Exception:
        return None
    return view if isinstance(view, dict) else None
