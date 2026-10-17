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

Dependencies:
    The standard library, config (EDA_SUMMARY_DIRNAME, EDA_SUMMARY_VERSION,
    EDA_MAX_WORKERS, EDA_WORKER_START_TIMEOUT_SECONDS,
    EDA_LIFETIME_EDGES_HOURS, EDA_LAG_EDGES_HOURS and
    MVE_SERIES_FAMILY_PREFIX), historical (the slice reader _day_store_iter,
    _SliceUnreadable, _iso_epoch, _read_title_file with _TITLE_FILE_ABSENT,
    the file names of _EVENT_TITLES_CACHE and _CANDLES_DIR, series_labels,
    series_ticker, infer_category) and scanner (event_series,
    deadline_profile, stated_deadline, DEADLINE_CUMULATIVE). Imports no
    order-path module (trader, main, scheduler, run_lock, defaults_server,
    v2_probe, seller), and no module of the package imports this one
    (tests/test_eda.py::TestIsolation).

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
"""
import gzip
import json
import logging
import multiprocessing
import os
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
