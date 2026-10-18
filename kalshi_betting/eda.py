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

    build_eda_page() draws all of that as the Data page, config.EDA_DASHBOARD_FILENAME
    under config.PROJECT_ROOT: an inventory; the market universe; price
    calibration; deadline ladders and k; data quality; and the Terms of
    glossary.PAGE_DATA. Charts are plotly figures embedded with no copy of
    Plotly (the page loads it once, from config.LIVE_DASHBOARD_PLOTLY_URL with
    its integrity hash). Every switch (Non-combo / Combo / All; the calibration's
    lead and price) is a group of buttons whose views Python has already drawn
    and hidden; the page's one small script only shows one view and hides the
    others. It never raises: a failed step is one WARNING and a line on the
    page, and the rest is still written. `python3 -m kalshi_betting.eda
    [--workers N]` (main) rebuilds the page from the cache without a backtest.
    Every word of the page is a constant in the Data page block, except the
    sentences of the coverage note, which merge() (_coverage) writes and the
    page shows as they are.

Dependencies:
    The standard library, config (EDA_SUMMARY_DIRNAME, EDA_SUMMARY_VERSION,
    EDA_MAX_WORKERS, EDA_WORKER_START_TIMEOUT_SECONDS,
    EDA_LIFETIME_EDGES_HOURS, EDA_LAG_EDGES_HOURS, EDA_CALIBRATION_BIN,
    EDA_PRICE_LEADS_HOURS, EDA_PRICE_MAX_AGE_HOURS, EDA_CANDLE_FACTS_VERSION,
    EDA_KHAT_SPREAD_BIN, EDA_KHAT_VERSION, EDA_MAX_BOOK_WIDTH,
    EDA_MIN_MARKETS_PER_POINT, EDA_DASHBOARD_FILENAME, PRICE_EPSILON, PROJECT_ROOT,
    KHAT_CONFIDENCE_LEVEL, LIVE_DASHBOARD_PLOTLY_URL, LIVE_DASHBOARD_PLOTLY_SRI,
    CANDLESTICK_CACHE_FIELDS_VERSION, DEPTH_SNAPSHOTS_DIRNAME and
    MVE_SERIES_FAMILY_PREFIX), glossary (the Data page's Terms), historical (the slice reader _day_store_iter,
    _SliceUnreadable, _iso_epoch, _read_title_file with _TITLE_FILE_ABSENT,
    the file names of _EVENT_TITLES_CACHE and _CANDLES_DIR, series_labels,
    series_ticker, infer_category, usable_candle_ask) and scanner
    (event_series, deadline_profile, stated_deadline, DEADLINE_CUMULATIVE).
    Imports no order-path module (trader, main, scheduler, run_lock,
    defaults_server, v2_probe, seller), and no module of the package imports
    this one (tests/test_eda.py::TestIsolation). The two exceptions to
    "only the standard library, config, glossary, historical and scanner"
    are khat_view(), which imports backtester and dashboard INSIDE the
    function, and _plotly(), which imports plotly there, so the worker
    processes that count the day slices (which import this module afresh)
    never load pandas or plotly.

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
    stale). A one-sided book says little (a 99c ask with nobody bidding only
    means nobody sold lower), so a price counts only where the book was at
    most config.EDA_MAX_BOOK_WIDTH wide, a missing bid counting as 0 (counted
    as wide otherwise). The close time used is the market's REALIZED close (the cache
    records when the market actually closed), so where trading carries on after
    the outcome is known, the 1 hour price can already reflect the outcome. The
    candles are those the backtest priced, so they are not a random sample of
    the markets.

    The cache folder is listed with os.scandir, once for the candle folder
    and once per top-level entry for inventory(); nothing globs or sorts the
    candle folder. The candle files are read in this process, one at a time.
"""
import argparse
import gzip
import html
import json
import logging
import logging.handlers
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

from . import config, glossary, historical, scanner

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


def _candle_prices(candle: Any) -> tuple[float | None, float | None, bool]:
    """
    The prices one candle gives: the YES ask, the midpoint, and whether the book was too wide.

    The YES ask is what betting Yes costs. The midpoint is halfway between the
    YES ask and the YES bid (1 minus the NO ask). Each ask counts only when
    historical.usable_candle_ask accepts it (strictly between 0 and 1, and a
    NO ask also below the 0.99 clamp that stands for an empty book). A long
    shot's YES bid is under a cent, so its NO ask sits at the clamp: it still
    has a YES ask but no midpoint.

    A one-sided book says little: a 99c ask with nobody bidding only means
    nobody sold below 99c. So a price counts only where the book was at most
    config.EDA_MAX_BOOK_WIDTH wide: the YES ask minus the YES bid, with a
    missing bid (an unusable NO ask) counted as a bid of 0 (so a long shot
    asked at 3c with no bid is 3c wide and counts, a 99c ask with no bid does
    not). A wider book gives no price at all, and is reported as wide.

    Residual: a real 1c YES bid reads as no bid (its NO ask sits at the 0.99
    clamp, which is no quote), so a book 10 to 11c wide with a 1c bid counts
    as wide.

    Args:
        candle (Any): One candle of a candle file.

    Returns:
        tuple[float | None, float | None, bool]: (YES ask, midpoint, wide);
            each price None when not usable or when the book was wide. A
            midpoint is never given without a YES ask.
    """
    if not isinstance(candle, dict):
        return None, None, False
    yes = historical.usable_candle_ask(candle.get("yes_ask_close"), "yes")
    if math.isnan(yes):
        return None, None, False
    no = historical.usable_candle_ask(candle.get("no_ask_close"), "no")
    bid = 0.0 if math.isnan(no) else 1.0 - no
    if yes - bid > config.EDA_MAX_BOOK_WIDTH + config.PRICE_EPSILON:
        return None, None, True
    if math.isnan(no):
        return yes, None, False
    return yes, (yes + (1.0 - no)) / 2.0, False


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
            candle's YES ask and midpoint, or None), "stale" (per lead,
            True when the last candle at or before the lead's moment is too
            old to use) and "wide" (per lead, True when the candle's book was
            wider than config.EDA_MAX_BOOK_WIDTH, so no price is kept).

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
    prices = [(None, None, False) if stale[i] else _candle_prices(best[i])
              for i in range(len(leads))]
    return {"f": fields, "e": not candles, "vol": volume if has_volume else None,
            "asks": [p[0] for p in prices], "mids": [p[1] for p in prices],
            "stale": stale, "wide": [p[2] for p in prices]}


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
            per lead, "wide" likewise.
    """
    if not isinstance(fact, dict):
        return False
    if not (fact.get("s") == size and fact.get("m") == mtime_ns and fact.get("c") == close
            and fact.get("v") == config.EDA_CANDLE_FACTS_VERSION):
        return False
    count = len(leads)
    stale, wide = fact.get("stale"), fact.get("wide")
    return (_is_int(fact.get("f")) and isinstance(fact.get("e"), bool)
            and "vol" in fact and (fact["vol"] is None or _finite(fact["vol"]) is not None)
            and _price_list(fact.get("asks"), count) and _price_list(fact.get("mids"), count)
            and isinstance(stale, list) and len(stale) == count
            and all(isinstance(x, bool) for x in stale)
            and isinstance(wide, list) and len(wide) == count
            and all(isinstance(x, bool) for x in wide))


def _new_cell(nbins: int) -> dict:
    """
    An empty calibration cell.

    Args:
        nbins (int): How many price bins.

    Returns:
        dict: {"bins": [[markets, yes, price sum], ...], "n": 0, "brier_sum": 0.0}.
    """
    return {"bins": [[0, 0, 0.0] for _ in range(nbins)], "n": 0, "brier_sum": 0.0}


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
    slot[2] += price
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
    config.EDA_PRICE_MAX_AGE_HOURS of that moment and only when its book was at
    most config.EDA_MAX_BOOK_WIDTH wide (YES ask minus YES bid, a missing bid
    counting as 0; see _candle_prices). Each price is filed in a
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
            a cell being {"bins": [[markets, yes, sum of their prices], ...],
            "n": markets priced,
            "brier_sum": float}; a category appears at a lead once one of its
            markets has an ask there), "coverage" (str(lead) -> {"markets":
            joined non-combo markets with a yes or no result, "no_ask": of
            them with no YES ask price at the lead, "no_mid": with no
            midpoint, "stale": whose last candle was too old, "wide": whose book
            was too wide to price; a wide market is not also counted in no_ask
            or no_mid}), "volume"
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
    coverage = {str(lead): {"markets": 0, "no_ask": 0, "no_mid": 0, "stale": 0, "wide": 0}
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
                if fact["wide"][i]:
                    seen["wide"] += 1
                    continue
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
            has none); nothing is written and any earlier file stays. The file
            also records config.EDA_KHAT_VERSION, which load_khat checks.
        OSError: When the file cannot be written.
    """
    text = json.dumps({**view, "version": config.EDA_KHAT_VERSION}, allow_nan=False,
                      separators=(",", ":"))
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
        dict | None: The view as save_khat was given it (the version key is
            taken off), or None when the file is missing, cannot be read, is not
            strict JSON, is not a JSON object, or has a version other than
            config.EDA_KHAT_VERSION or none.
    """
    try:
        view = json.loads(_khat_path(cache_dir).read_text(), parse_constant=_refuse_constant)
    except Exception:
        return None
    if not isinstance(view, dict) or view.get("version") != config.EDA_KHAT_VERSION:
        return None
    return {key: value for key, value in view.items() if key != "version"}


# ─── The Data page ────────────────────────────────────────────────────────────
#
# Everything below builds the page; nothing above depends on it. Every word the
# page shows is one of the constants in this block, so the script that switches
# views writes no text of its own.

PAGE_TITLE = "Data"
PAGE_BUILT = "Built {when} from {files:,} day files ({records:,} records)."
PAGE_BUILT_NO_COUNTS = ("Built {when}. The day files could not be counted; see the notes "
                        "below.")
BUILT_TIME_FORMAT = "%Y-%m-%d %H:%M UTC"
FAILURES_HEAD = "Some steps of this build failed, so parts of the page are missing:"
FAILURE_LINE = "{step}: {reason}"
STEP_SUMMARIES = "Counting the day files"
STEP_CANDLES = "Reading the candle files"
STEP_INVENTORY = "Listing the cache folder"
STEP_KHAT = "Preparing the k figures"
STEP_LISTING = "Reading Kalshi's series listing"
STEP_CHARTS = "Loading the chart library"
STEP_KHAT_SAVE = "Saving the k figures"
CHARTS_UNAVAILABLE = "the chart library could not be loaded"
PAGE_FAILED = "The page could not be put together ({reason})."
MAIN_NO_PAGE = "No page could be written; see kalshi_eda.log."
EDA_LOG_FILENAME = "kalshi_eda.log"
SECTION_FAILED = "The {section} section could not be built ({reason})."
NO_DAY_COUNTS = "The day-file counts are missing, so there is nothing to draw here."
NO_CANDLES = "The candle figures are missing, so there is nothing to draw here."
NO_MARKETS_IN_VIEW = "No markets in this view."
NO_DATA_FOR_CHART = "Nothing to draw."

HEADING_INVENTORY = "Inventory"
HEADING_UNIVERSE = "Market universe"
HEADING_CALIBRATION = "Price calibration"
HEADING_LADDERS = "Deadline ladders and k"
HEADING_QUALITY = "Data quality"

INVENTORY_HEADERS = ("Item", "Files", "Size", "Dates", "What it is")
INVENTORY_NOTE = ("Assembled corpora are filtered to the markets the backtest could trade, "
                  "and candle files cover only the markets the backtest priced, so neither "
                  "is the whole exchange. The day files are the unfiltered record of what "
                  "settled.")
INVENTORY_EMPTY = "The cache folder holds nothing."
INVENTORY_DATES = "{first} to {last}"
INVENTORY_START = "starts {value}"
INVENTORY_FILTER = "filter {value}"
INVENTORY_ASSEMBLED = "assembled {value}"
INVENTORY_SNAPSHOTS = "{count:,} snapshots"
INVENTORY_SNAPSHOT_ONE = "1 snapshot"
INVENTORY_NEWEST = "newest {value} UTC"
INVENTORY_META_ERROR = "first line unreadable: {reason}"
INVENTORY_UNREADABLE_PARTS = "{count:,} unreadable parts"
INVENTORY_ERROR = "could not be read: {reason}"
NOT_AVAILABLE = "—"
# Units, written in one place: an hour edge or lead below HOURS_AS_DAYS_FROM
# hours is written in hours, otherwise in whole days when it is a whole number
# of them; sizes step up every BYTES_PER_STEP.
HOURS_TEXT = "{value:g} h"
DAYS_TEXT = "{value:g} d"
HOURS_AS_DAYS_FROM = 48
HOURS_PER_DAY = 24
BYTE_UNITS = ("B", "KB", "MB", "GB")
BYTES_PER_STEP = 1000
BYTES_TEXT = "{value:d} {unit}"
BYTES_TEXT_DECIMAL = "{value:.1f} {unit}"

SWITCH_COMBO_LABEL = "Show"
VIEW_LABELS = {"noncombo": "Non-combo", "combo": "Combo", "all": "All"}
LEAD_SWITCH_LABEL = "Price taken"
LEAD_LABEL = "{span} before close"
PRICE_SWITCH_LABEL = "Price"
PRICE_LABELS = {"ask": "Yes ask", "mid": "Midpoint"}

UNIVERSE_INTRO = ("Markets that have settled, by the week they settled in (weeks start on "
                  "Monday, UTC). The Combo switch separates multi-leg tickets from "
                  "ordinary markets; Non-combo is the default.")
UNIVERSE_RESIDUAL = ("Each market is counted once per day file. The day files do not "
                     "overlap, except that a market settled exactly on a UTC midnight may "
                     "sit in two neighbouring live days and be counted twice.")
UNIVERSE_UNKNOWN_WEEK = ("{count:,} markets have no readable settlement time; they are in "
                         "the totals but not in the weekly charts.")
CHART_SETTLED = "Markets settled per week"
CHART_SETTLED_NOTE = ("Stacked by Kalshi category: the {top} largest, and the rest as "
                      "More.")
CHART_YES_BY_CATEGORY = "Share settled Yes, by category"
CHART_YES_BY_CATEGORY_NOTE = ("Of the markets that settled Yes or No, the fraction that "
                              "settled Yes, largest category first. A low share means the "
                              "questions mostly ask about things that do not happen. "
                              "Categories with no settled Yes or No market are left out.")
CHART_YES_OVER_TIME = "Share settled Yes, by month"
CHART_YES_OVER_TIME_NOTE = ("The largest categories, by the month of the settlement week. A "
                            "category's month with fewer than {minimum} settled markets is "
                            "left out, so a few markets do not make a 0% or 100% spike.")
CHART_LIFETIME = "Market lifetime"
CHART_LIFETIME_NOTE = "Close time minus open time."
CHART_LAG = "Settlement lag"
CHART_LAG_NOTE = "Settlement time minus close time."
CHART_REGIME = "Tick regime over time"
CHART_REGIME_NOTE = "Share of each week's settled markets by price tick structure."
CHART_SHARD = "Exchange shard over time"
CHART_SHARD_NOTE = "Share of each week's settled markets by exchange shard."
CHART_VOLUME = "Traded volume, by category"
CHART_VOLUME_NOTE = ("Contracts traded, summed from the candle files that record volume, "
                     "for markets that are not combos: {files:,} of the {all:,} candle "
                     "files joined to a settled non-combo market carry it.")
AXIS_MARKETS = "Markets"
AXIS_LADDERS = "Ladders"
AXIS_SHARE_YES = "Share settled Yes (%)"
AXIS_PERCENT = "Share (%)"
AXIS_CONTRACTS = "Contracts"
AXIS_MONTH = "Month"
AXIS_WEEK = "Week"
LABEL_MORE = "More"
LABEL_NONE_RECORDED = "none recorded"
LABEL_SHARD = "Shard {value}"
LABEL_UNKNOWN_BUCKET = "unknown"
LAG_FIRST_BUCKET = "before close"
LIFETIME_FIRST_BUCKET = "under {edge}"
BUCKET_MIDDLE = "{low} to {high}"
BUCKET_LAST = "{edge} or more"
HOVER_YES_SHARE = "%{x}: %{y:.1f}% Yes of %{customdata:,} settled"
HOVER_YES_BAR = "%{y}: %{x:.1f}% Yes of %{customdata:,} settled"
HOVER_COUNT_BAR = "%{y}: %{x:,}"
TEXT_YES_BAR = "{share:.1f}% of {count:,}"

CALIBRATION_INTRO = ("Each market's price at a set time before close, against how it "
                     "settled. Pick when the price was taken and which price. Combo "
                     "markets are not calibrated, so there is no Combo switch here: "
                     "their prices are not what the strategy trades.")
CALIBRATION_SAMPLE_NOTE = ("The markets are the ones the backtest priced, so they are not "
                           "a random sample of Kalshi.")
CALIBRATION_ASK_NOTE = ("The Yes ask is what betting Yes costs, so it includes the cost of "
                        "trading; the midpoint is halfway between the Yes ask and the Yes "
                        "bid and exists only where both sides had a quote.")
CALIBRATION_CLOSE_NOTE = ("Close time is the realized close, so where trading went on "
                          "after the outcome was known, the 1 h price can already show it.")
CHART_RELIABILITY = "Price against share settled Yes"
CHART_RELIABILITY_NOTE = ("Each dot is one price bin, placed at the average price in the bin; "
                          "its size shows how many markets it holds. The line is where a "
                          "fair price would sit. A price counts "
                          "only where the gap between the Yes ask and the Yes bid was "
                          "{width}c or less (a missing bid counts as 0), so one-sided "
                          "quotes are left out.")
CHART_GAP = "Share settled Yes minus price"
CHART_GAP_NOTE = ("In percentage points, by price bin, at the average price in the bin: "
                  "the share settled Yes minus that price. Below 0 at low prices means "
                  "long shots are overpriced.")
AXIS_PRICE = "Price ($)"
AXIS_GAP = "Points"
HOVER_GAP = "average price %{x:.3f}: %{y:+.1f} points, %{customdata:,} markets"
HOVER_BIN = "average price %{x:.3f}: %{y:.1f}% Yes of %{customdata:,} markets"
LABEL_FAIR = "Fair price"
LABEL_BINS = "Price bins"
BRIER_HEADERS = ("Category", "Markets", "Brier")
BRIER_ALL = "All"
BRIER_CAPTION = "Brier score by category for this price and time (lower is better)."
COVERAGE_HEADERS = ("Lead", "Markets", "Priced (ask)", "No ask", "No midpoint", "Wide book",
                    "Stale")
COVERAGE_CAPTION = ("Markets with a yes or no result, by how many had no Yes ask, no "
                    "midpoint, a book too wide to price, or a last candle too old to use "
                    "at each time. Stale ones are also counted under No ask, and every "
                    "market with no ask is also under No midpoint; a wide book is in "
                    "neither. Priced (ask) is Markets minus No ask minus Wide book.")

LADDERS_INTRO = ("Markets that ask the same question at several deadlines. The census "
                 "and ladder charts are for markets that are not combos.")
CHART_CENSUS = "Deadline wording, by category"
CHART_CENSUS_NOTE = ("Whether each market's wording states a cumulative deadline (by a "
                     "date), a snapshot (on a date) or neither.")
CHART_LADDERS = "Deadline ladders per month"
CHART_LADDERS_NOTE = ("Events with two or more cumulative rungs stating different days, "
                      "by the month of the earliest day. Total: {total:,}.")
CENSUS_NAMES = {"cumulative": "Cumulative", "snapshot": "Snapshot", "unknown": "Unknown"}
CENSUS_COLORS = {"cumulative": "#2a78d6", "snapshot": "#eb6834", "unknown": "#898781"}
KHAT_HEAD = "The k view"
KHAT_NONE = "No backtest has saved its k figures yet."
KHAT_FAILED = "The k figures could not be drawn ({reason})."
KHAT_INTRO = ("From the last backtest's own deadline pairs: {observations:,} entries, "
              "{non_positive:,} with a mid spread of 0 or less left out of the bins. The "
              "mid spread is the later market's midpoint minus the earlier one's; a pair "
              "is in between if the event happened between the two deadlines.")
CHART_KHAT = "Mid spread against how often the event fell in between"
CHART_KHAT_NOTE = ("Each dot is one mid-spread bin. The lines are y = k̂ x (what happened) "
                   "and y = k x (what the strategy assumed, k = {k}).")
LABEL_KHAT_LINE = "k̂ x"
LABEL_K_LINE = "k x"
LABEL_KHAT_DOTS = "Mid-spread bins"
AXIS_MID_SPREAD = "Mid spread ($)"
AXIS_IN_BETWEEN = "In between (%)"
HOVER_KHAT_BIN = "mid spread %{x:.3f}: %{y:.1f}% in between of %{customdata:,} pairs"
KHAT_HEADERS = ("Group", "Pairs", "Events", "k̂", "{level} CI")
KHAT_POOLED = "All pairs"
KHAT_BY_GAP = "k̂ by deadline gap"
KHAT_BY_CATEGORY = "k̂ by category"
KHAT_NOTE = ("k̂ is how often events fell in between, divided by the average mid spread. "
             "The figures are the backtest page's own, saved by the last run; each event "
             "counts once in the interval.")

QUALITY_INTRO = ("How complete the stored records are, by the week the market settled. "
                 "The Combo switch works as above.")
CHART_QUALITY = "Records with a field missing"
CHART_QUALITY_NOTE = ("Share of each week's markets. The event title is also shown after "
                      "the saved title file is joined in.")
QUALITY_SERIES = (("blank_subtitle", "Blank subtitle", "#2a78d6"),
                  ("blank_title_before", "Blank event title", "#eb6834"),
                  ("blank_title_after", "Blank event title after the join", "#1baf7a"),
                  ("no_open", "Missing open time", "#eda100"))
AXIS_MISSING = "Missing (%)"
DAYFILES_HEADERS = ("Day files", "Count")
DAYFILES_ARCHIVE = "Archive day files"
DAYFILES_LIVE = "Live day files"
DAYFILES_SKIPPED = "Live days left out (the archive covers them)"
DAYFILES_GONE = "Live days kept after their file was deleted"
DAYFILES_DROPPED = "Records at or after the archive cutoff, left out of archive days"
CANDLEFILES_HEADERS = ("Candle files", "Count")
CANDLEFILES_VERSION = "Field version {version}"
CANDLEFILES_VERSION_CURRENT = "Field version {version} (current)"
CANDLEFILES_EMPTY = "Empty"
CANDLEFILES_UNREADABLE = "Unreadable"
CANDLEFILES_UNJOINED = "No market in the day files"
CANDLEFILES_COMBOS = "Combos, left out of the calibration"
DEPTH_LINE = "Depth snapshots: {count:,}; newest {newest} UTC."
DEPTH_LINE_NONE = "Depth snapshots: none saved."
DEPTH_LINE_UNKNOWN = "Depth snapshots: not listed."
UNREADABLE_HEAD = "Day files that could not be read"
UNREADABLE_NONE = "Every day file was read."

# How many of the largest categories get a colour of their own; the rest are
# drawn together in grey as More.
_TOP_CATEGORIES = 8
# How many categories the share-over-time chart draws.
_TOP_LINES = 6
_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
            "#4a3aa7", "#7a5c3e")
_MORE_COLOR = "#898781"
_INK = "#0b0b0b"
_GRID = "#e1e0d9"
_PLOT_CONFIG = {"displaylogo": False, "responsive": True}
_CHART_HEIGHT = 340
# Legend entries that fit on one row above a chart, and the height of a row.
_LEGEND_PER_ROW = 5
_LEGEND_ROW_PX = 22
# Pixels per category of a horizontal bar chart, and its least height.
_BAR_ROW_PX = 26
# The view the page opens on for each switch.
_DEFAULT_VIEW = "noncombo"
_DEFAULT_LEAD = 24
_DEFAULT_PRICE = "ask"
# Combo flags of the merged keys that each view adds up.
_VIEW_COMBOS = {"noncombo": ("0",), "combo": ("1",), "all": ("0", "1")}
_VIEW_ORDER = ("noncombo", "combo", "all")
_SCRIPT_ID = "eda-script"

_CSS = """
body { font-family: sans-serif; color: #0b0b0b; background: #ffffff; margin: 0;
       padding: 0 16px 48px 16px; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 26px; margin: 20px 0 4px 0; }
h2 { font-size: 22px; margin: 40px 0 8px 0; padding-bottom: 8px;
     border-bottom: 2px solid #E0E0E0; scroll-margin-top: 16px; }
h3 { font-size: 16px; margin: 24px 0 2px 0; }
p { font-size: 14px; line-height: 1.4; margin: 6px 0; }
.note { color: #52514e; font-size: 13px; }
.warn { background: #fff4d6; border: 1px solid #eda100; padding: 8px 12px;
        border-radius: 4px; }
.switch { margin: 12px 0 4px 0; font-size: 14px; }
.switch span { margin-right: 8px; color: #52514e; }
.switch button { font: inherit; padding: 4px 12px; margin-right: 4px; cursor: pointer;
                 background: #f9f9f7; border: 1px solid #c3c2b7; border-radius: 4px; }
.switch button[aria-pressed="true"] { background: #2a78d6; color: #ffffff;
                                      border-color: #2a78d6; }
table { font-size: 14px; border-collapse: collapse; margin: 12px 0; width: 100%; }
caption { text-align: left; color: #52514e; font-size: 13px; padding: 4px 0; }
th { padding: 6px 12px; text-align: left; background: #E3F2FD; }
td { padding: 5px 12px; vertical-align: top; border-top: 1px solid #E0E0E0; }
td.num, th.num { text-align: right; }
.chart { margin: 8px 0 16px 0; }
"""


def _esc(value: Any) -> str:
    """
    Escape a value for use as HTML text or inside a double-quoted attribute.

    Args:
        value (Any): Anything; it is shown as text.

    Returns:
        str: The text with &, <, > and quotes escaped.
    """
    return html.escape(str(value), quote=True)


def _plotly() -> Any:
    """
    Import plotly's graph objects.

    Imported here rather than at the top of the module so the worker
    processes that count day slices (which import this module) never load it.

    Returns:
        module: plotly.graph_objects.
    """
    import plotly.graph_objects as go  # local: keeps plotly out of the workers
    return go


def _human_bytes(count: Any) -> str:
    """
    A size in bytes as a short decimal figure.

    Args:
        count (Any): A byte count, or None.

    Returns:
        str: For example "512 B", "28.4 MB" or "24.6 GB"; NOT_AVAILABLE for
            None or a value that is not a number.
    """
    if not isinstance(count, (int, float)) or isinstance(count, bool) or count < 0:
        return NOT_AVAILABLE
    size = float(count)
    for unit in BYTE_UNITS:
        if size < BYTES_PER_STEP or unit == BYTE_UNITS[-1]:
            return (BYTES_TEXT.format(value=int(size), unit=unit) if unit == BYTE_UNITS[0]
                    else BYTES_TEXT_DECIMAL.format(value=size, unit=unit))
        size /= BYTES_PER_STEP
    return NOT_AVAILABLE


def _edge_text(hours: float) -> str:
    """
    A bucket edge, in hours, written as hours or days.

    Args:
        hours (float): The edge.

    Returns:
        str: "6 h" below two days, otherwise whole days ("3 d") when it is
            a whole number of days.
    """
    if hours >= HOURS_AS_DAYS_FROM and hours % HOURS_PER_DAY == 0:
        return DAYS_TEXT.format(value=hours // HOURS_PER_DAY)
    return HOURS_TEXT.format(value=hours)


def _bucket_labels(edges: tuple, first: str) -> list[str]:
    """
    The labels of a histogram's buckets, from its edges.

    Args:
        edges (tuple): Ascending edges, in hours.
        first (str): The first bucket's label, or a template with {edge}.

    Returns:
        list[str]: len(edges) + 1 labels, matching _bucket's buckets.
    """
    labels = [first.format(edge=_edge_text(edges[0]))]
    labels += [BUCKET_MIDDLE.format(low=_edge_text(lo), high=_edge_text(hi))
               for lo, hi in zip(edges, edges[1:], strict=False)]
    labels.append(BUCKET_LAST.format(edge=_edge_text(edges[-1])))
    return labels


def _lead_label(hours: Any) -> str:
    """
    The name of a lead on the page.

    Args:
        hours (Any): The lead, in hours.

    Returns:
        str: "1 h before close", "24 h before close" or "7 d before close".
    """
    return LEAD_LABEL.format(span=_edge_text(hours))


def _short_lead(hours: Any) -> str:
    """
    A lead as the switch button names it.

    Args:
        hours (Any): The lead, in hours.

    Returns:
        str: "1 h", "24 h" or "7 d".
    """
    return _edge_text(hours)


class _Reg:
    """What one section tells the page script: its switches and the views they show."""

    def __init__(self) -> None:
        """Start with no switches and no panels."""
        self.switches: dict[str, dict] = {}
        self.panels: list[dict] = []

    def switch(self, name: str, label: str, options: list[tuple[str, str]],
               default: str) -> str:
        """
        Register a switch and return its buttons.

        Args:
            name (str): The switch's name (part of every button id).
            label (str): The words in front of the buttons.
            options (list[tuple[str, str]]): (value, button text) pairs.
            default (str): The value that starts pressed.

        Returns:
            str: HTML for the switch; each button has aria-pressed.
        """
        self.switches[name] = {"options": {v: f"btn-{name}-{v}" for v, _ in options},
                               "default": default}
        buttons = "".join(
            f'<button type="button" id="btn-{_esc(name)}-{_esc(v)}" '
            f'aria-pressed="{"true" if v == default else "false"}">{_esc(text)}</button>'
            for v, text in options)
        return (f'<div class="switch" role="group" aria-label="{_esc(label)}">'
                f'<span>{_esc(label)}</span>{buttons}</div>')

    def panel(self, names: list[str], views: dict[str, dict]) -> None:
        """
        Register the views a set of switches chooses between.

        Args:
            names (list[str]): The switches, in the order their values make a key.
            views (dict[str, dict]): key ("|"-joined values) -> {"div": the
                view's id, "charts": ids of the charts inside it}.
        """
        self.panels.append({"switches": names, "views": views})

    def absorb(self, other: "_Reg") -> None:
        """
        Take another registry's switches and panels.

        Args:
            other (_Reg): A finished section's registry.
        """
        self.switches.update(other.switches)
        self.panels.extend(other.panels)


def _table(headers: tuple, rows: list[list[str]], *, caption: str | None = None,
           numeric: tuple = ()) -> str:
    """
    A table whose every cell is escaped.

    Args:
        headers (tuple): Column headers.
        rows (list[list[str]]): Rows of cell text.
        caption (str | None): A short line above the table.
        numeric (tuple): Indexes of columns to right-align.

    Returns:
        str: HTML for the table.
    """
    head = "".join(f'<th{" class=num" if i in numeric else ""}>{_esc(h)}</th>'
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td{" class=num" if i in numeric else ""}>{_esc(c)}</td>'
                         for i, c in enumerate(row)) + "</tr>"
        for row in rows)
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f"<table>{cap}<tr>{head}</tr>{body}</table>"


def _note(text: str, css: str = "note") -> str:
    """
    A paragraph.

    Args:
        text (str): Its text, escaped here.
        css (str): Its class: "note" (grey) or "warn" (amber box).

    Returns:
        str: HTML.
    """
    return f'<p class="{css}">{_esc(text)}</p>'


def _chart(go: Any, title: str, note: str | None, traces: list, chart_id: str, ids: list,
           *, x_title: str | None = None, y_title: str | None = None,
           layout: dict | None = None, xaxis: dict | None = None,
           yaxis: dict | None = None, height: int = _CHART_HEIGHT,
           right: int = 16) -> str:
    """
    One chart: a heading, an optional note and the figure.

    The figure carries no copy of Plotly itself (the page loads it once) and
    no template, which keeps each chart a few KB.

    Args:
        go (Any): plotly.graph_objects.
        title (str): The heading above the chart.
        note (str | None): A grey line under the heading.
        traces (list): The figure's traces.
        chart_id (str): The chart's div id, added to `ids`.
        ids (list): Collects the ids of the charts in one view, so the page
            script can resize them when the view is shown.
        x_title (str | None): The x axis title.
        y_title (str | None): The y axis title.
        layout (dict | None): Extra layout settings.
        xaxis (dict | None): Extra x axis settings.
        yaxis (dict | None): Extra y axis settings.
        height (int): The plot's height in pixels; the legend's room is added.
        right (int): The right margin in pixels (room for bar labels).

    Returns:
        str: HTML.
    """
    fig = go.Figure(data=traces)
    # The legend sits above the plot, clear of the axis titles below it; with
    # many entries it wraps onto more rows, so the top margin grows with them.
    legend_rows = 0 if len(traces) < 2 else 1 + len(traces) // _LEGEND_PER_ROW
    top = 12 + _LEGEND_ROW_PX * legend_rows
    fig.update_layout(
        template="none", height=height + top + 48,
        margin={"l": 64, "r": right, "t": top, "b": 48},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": _INK, "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        xaxis={"gridcolor": _GRID, **({"title": {"text": x_title}} if x_title else {}),
               **(xaxis or {})},
        yaxis={"gridcolor": _GRID, **({"title": {"text": y_title}} if y_title else {}),
               **(yaxis or {})},
        **(layout or {}))
    ids.append(chart_id)
    figure = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id,
                         config=_PLOT_CONFIG, default_height=f"{height + top + 48}px",
                         default_width="100%")
    note_html = _note(note) if note else ""
    return f'<div class="chart"><h3>{_esc(title)}</h3>{note_html}{figure}</div>'


def _split_key(key: str) -> list[str]:
    """
    Split a merged key into week, combo flag, category and tag.

    Args:
        key (str): "week|combo|category|tag".

    Returns:
        list[str]: Exactly four parts.
    """
    return (str(key).split(KEY_SEPARATOR) + ["", "", "", ""])[:4]


def _add_list(into: list, more: Any) -> None:
    """
    Add a list of counts into another, element by element.

    Args:
        into (list): The running counts.
        more (Any): Counts to add; ignored when not a list.
    """
    if isinstance(more, list):
        for i, value in enumerate(more[: len(into)]):
            into[i] += value


def _aggregate(counts: dict, combos: tuple) -> dict:
    """
    Add up the merged counters of the keys whose combo flag is in `combos`.

    Args:
        counts (dict): merge()'s "counts".
        combos (tuple): The combo flags to add ("0", "1").

    Returns:
        dict: "n"; "unknown" (markets with no readable week); "weekly"
            (week -> category -> n); "cat_n"; "cat_yes" (category -> [yes,
            yes + no]); "month" ((month, category) -> [yes, yes + no]); "life"
            and "lag" (bucket counts) with "life_unknown" and "lag_unknown";
            "regime" and "shard" (week -> name -> n); "quality" (week -> [n,
            then the counts of the fields of QUALITY_SERIES]); "phrasing" (category ->
            verdict -> n).
    """
    agg: dict = {"n": 0, "unknown": 0, "weekly": {}, "cat_n": {}, "cat_yes": {},
                 "month": {}, "life": [0] * (len(config.EDA_LIFETIME_EDGES_HOURS) + 1),
                 "life_unknown": 0, "lag": [0] * (len(config.EDA_LAG_EDGES_HOURS) + 1),
                 "lag_unknown": 0, "regime": {}, "shard": {}, "quality": {}, "phrasing": {}}
    for key, c in counts.items():
        week, combo, category, _tag = _split_key(key)
        if combo not in combos:
            continue
        n = c.get("n", 0)
        agg["n"] += n
        agg["cat_n"][category] = agg["cat_n"].get(category, 0) + n
        pair = agg["cat_yes"].setdefault(category, [0, 0])
        pair[0] += c.get("yes", 0)
        pair[1] += c.get("yes", 0) + c.get("no", 0)
        _add_list(agg["life"], c.get("lifetime_h"))
        _add_list(agg["lag"], c.get("lag_h"))
        agg["life_unknown"] += c.get("lifetime_unknown", 0)
        agg["lag_unknown"] += c.get("lag_unknown", 0)
        for verdict, count in (c.get("phrasing") or {}).items():
            bucket = agg["phrasing"].setdefault(category, {})
            bucket[verdict] = bucket.get(verdict, 0) + count
        if week == UNKNOWN_WEEK:
            agg["unknown"] += n
            continue
        by_cat = agg["weekly"].setdefault(week, {})
        by_cat[category] = by_cat.get(category, 0) + n
        month = agg["month"].setdefault((week[:7], category), [0, 0])
        month[0] += c.get("yes", 0)
        month[1] += c.get("yes", 0) + c.get("no", 0)
        for name in ("regime", "shard"):
            bucket = agg[name].setdefault(week, {})
            for value, count in (c.get(name) or {}).items():
                bucket[value] = bucket.get(value, 0) + count
        quality = agg["quality"].setdefault(week, [0] * (1 + len(QUALITY_SERIES)))
        quality[0] += n
        for i, (field, _name, _color) in enumerate(QUALITY_SERIES, start=1):
            quality[i] += c.get(field, 0)
    return agg


def _shares(pairs: dict) -> dict:
    """
    Turn [yes, settled] pairs into percentages.

    Args:
        pairs (dict): name -> [yes, yes + no].

    Returns:
        dict: name -> (percent Yes, markets), for names with at least one.
    """
    return {name: (100.0 * yes / total, total) for name, (yes, total) in pairs.items()
            if total > 0}


def _stack_traces(go: Any, weeks: list, series: list[tuple[str, list, str]]) -> list:
    """
    One stacked bar trace per series over the same weeks.

    Args:
        go (Any): plotly.graph_objects.
        weeks (list): The x values.
        series (list[tuple[str, list, str]]): (name, values, colour).

    Returns:
        list: go.Bar traces.
    """
    return [go.Bar(x=weeks, y=values, name=name, marker_color=color)
            for name, values, color in series]


def _histogram(go: Any, labels: list[str], values: list, unknown: int) -> list:
    """
    A one-trace bar histogram, with an extra bar for markets with no value.

    Args:
        go (Any): plotly.graph_objects.
        labels (list[str]): Bucket labels.
        values (list): Bucket counts.
        unknown (int): Markets with no readable value.

    Returns:
        list: One go.Bar trace.
    """
    x, y = list(labels), list(values)
    if unknown:
        x.append(LABEL_UNKNOWN_BUCKET)
        y.append(unknown)
    return [go.Bar(x=x, y=y, marker_color=_PALETTE[0], showlegend=False)]


def _universe_view(go: Any, agg: dict, view: str, ids: list) -> str:
    """
    The charts of the market universe for one combo view.

    Args:
        go (Any): plotly.graph_objects.
        agg (dict): _aggregate's result for the view.
        view (str): "noncombo", "combo" or "all" (part of the chart ids).
        ids (list): Collects the chart ids.

    Returns:
        str: HTML for the view's charts.
    """
    if not agg["n"]:
        return _note(NO_MARKETS_IN_VIEW)
    parts = []
    weeks = sorted(agg["weekly"])
    order = sorted(agg["cat_n"], key=lambda c: (-agg["cat_n"][c], c))
    top, rest = order[:_TOP_CATEGORIES], order[_TOP_CATEGORIES:]
    colors = {c: _PALETTE[i] for i, c in enumerate(top)}
    if agg["unknown"]:
        parts.append(_note(UNIVERSE_UNKNOWN_WEEK.format(count=agg["unknown"])))
    series = [(c, [agg["weekly"][w].get(c, 0) for w in weeks], colors[c]) for c in top]
    if rest:
        series.append((LABEL_MORE, [sum(n for c, n in agg["weekly"][w].items() if c in rest)
                                    for w in weeks], _MORE_COLOR))
    parts.append(_chart(go, CHART_SETTLED, CHART_SETTLED_NOTE.format(top=_TOP_CATEGORIES),
                        _stack_traces(go, weeks, series),
                        f"eda-{view}-settled", ids, x_title=AXIS_WEEK, y_title=AXIS_MARKETS,
                        layout={"barmode": "stack"}))
    shares = _shares(agg["cat_yes"])
    names = [c for c in order if c in shares]
    parts.append(_chart(
        go, CHART_YES_BY_CATEGORY, CHART_YES_BY_CATEGORY_NOTE,
        [go.Bar(y=names, x=[shares[c][0] for c in names], orientation="h",
                customdata=[shares[c][1] for c in names], hovertemplate=HOVER_YES_BAR
                + "<extra></extra>", marker_color=_PALETTE[0], showlegend=False,
                text=[TEXT_YES_BAR.format(share=shares[c][0], count=shares[c][1])
                      for c in names], textposition="outside", cliponaxis=False)],
        f"eda-{view}-yes", ids, x_title=AXIS_SHARE_YES, xaxis={"range": [0, 100]},
        yaxis={"autorange": "reversed", "automargin": True},
        height=max(_CHART_HEIGHT, _BAR_ROW_PX * len(names) + 40), right=110))
    months = sorted({m for m, _c in agg["month"]})
    lines = []
    for category in order[:_TOP_LINES]:
        points = [(m, agg["month"][(m, category)]) for m in months
                  if (m, category) in agg["month"]
                  and agg["month"][(m, category)][1] >= config.EDA_MIN_MARKETS_PER_POINT]
        lines.append(go.Scatter(
            x=[m for m, _ in points], y=[100.0 * p[0] / p[1] for _, p in points],
            customdata=[p[1] for _, p in points], mode="lines+markers", name=category,
            line={"color": colors[category]}, hovertemplate=HOVER_YES_SHARE
            + "<extra></extra>"))
    parts.append(_chart(go, CHART_YES_OVER_TIME,
                        CHART_YES_OVER_TIME_NOTE.format(minimum=config.EDA_MIN_MARKETS_PER_POINT),
                        lines,
                        f"eda-{view}-yes-time", ids, x_title=AXIS_MONTH,
                        y_title=AXIS_SHARE_YES, yaxis={"range": [0, 100]}))
    parts.append(_chart(
        go, CHART_LIFETIME, CHART_LIFETIME_NOTE,
        _histogram(go, _bucket_labels(config.EDA_LIFETIME_EDGES_HOURS, LIFETIME_FIRST_BUCKET),
                   agg["life"], agg["life_unknown"]),
        f"eda-{view}-life", ids, y_title=AXIS_MARKETS))
    parts.append(_chart(
        go, CHART_LAG, CHART_LAG_NOTE,
        _histogram(go, _bucket_labels(config.EDA_LAG_EDGES_HOURS, LAG_FIRST_BUCKET),
                   agg["lag"], agg["lag_unknown"]),
        f"eda-{view}-lag", ids, y_title=AXIS_MARKETS))
    for name, title, note, label in (
            ("regime", CHART_REGIME, CHART_REGIME_NOTE, lambda v: v or LABEL_NONE_RECORDED),
            ("shard", CHART_SHARD, CHART_SHARD_NOTE,
             lambda v: LABEL_SHARD.format(value=v) if v != "" else LABEL_NONE_RECORDED)):
        totals: dict = {}
        for by_name in agg[name].values():
            for value, count in by_name.items():
                totals[value] = totals.get(value, 0) + count
        kinds = sorted(totals, key=lambda v: (-totals[v], v))
        traces = [go.Bar(x=weeks, y=[agg[name][w].get(kind, 0) for w in weeks],
                         name=label(kind), marker_color=_PALETTE[i % len(_PALETTE)])
                  for i, kind in enumerate(kinds)]
        parts.append(_chart(go, title, note, traces, f"eda-{view}-{name}", ids,
                            x_title=AXIS_WEEK, y_title=AXIS_PERCENT,
                            layout={"barmode": "stack", "barnorm": "percent"}))
    return "".join(parts)


def _volume_block(go: Any, cal: dict | None) -> str:
    """
    The traded volume by category, from the candle files that record volume.

    Args:
        go (Any): plotly.graph_objects.
        cal (dict | None): summarise_candles' result.

    Returns:
        str: HTML for the chart, or a note when there is nothing to draw.
    """
    if not cal or not cal.get("volume"):
        return _note(NO_CANDLES if not cal else NO_DATA_FOR_CHART)
    names = sorted(cal["volume"], key=lambda c: (-cal["volume"][c], c))
    joined = (cal.get("files", 0) - cal.get("unreadable", 0) - cal.get("unjoined", 0)
              - cal.get("combos_left_out", 0))
    note = CHART_VOLUME_NOTE.format(files=cal.get("volume_files", 0), all=joined)
    return _chart(go, CHART_VOLUME, note,
                  [go.Bar(y=names, x=[cal["volume"][c] for c in names], orientation="h",
                          hovertemplate=HOVER_COUNT_BAR + "<extra></extra>",
                          marker_color=_PALETTE[0], showlegend=False)],
                  "eda-volume", [], x_title=AXIS_CONTRACTS,
                  yaxis={"autorange": "reversed", "automargin": True},
                  height=max(_CHART_HEIGHT, _BAR_ROW_PX * len(names) + 40))


def _view_div(section: str, key: str, default: bool, body: str) -> str:
    """
    One pre-drawn view, hidden unless it is the default.

    Args:
        section (str): The section's short name (part of the id).
        key (str): The view's key, made safe for an id.
        default (bool): Whether it starts shown.
        body (str): Its HTML.

    Returns:
        str: HTML.
    """
    return f'<div id="view-{section}-{key}"{"" if default else " hidden"}>{body}</div>'


def _combo_views(go: Any, counts: dict, section: str, reg: _Reg, draw) -> str:
    """
    The Non-combo / Combo / All switch and its three pre-drawn views.

    Args:
        go (Any): plotly.graph_objects.
        counts (dict): merge()'s "counts".
        section (str): The section's short name.
        reg (_Reg): Where the switch and panel are registered.
        draw (Callable): draw(go, agg, view, ids) -> the view's HTML.

    Returns:
        str: HTML for the switch and the views.
    """
    switch_name = f"combo-{section}"
    switch = reg.switch(switch_name, SWITCH_COMBO_LABEL,
                        [(v, VIEW_LABELS[v]) for v in _VIEW_ORDER], _DEFAULT_VIEW)
    views, divs = {}, []
    for view in _VIEW_ORDER:
        ids: list = []
        body = draw(go, _aggregate(counts, _VIEW_COMBOS[view]), view, ids)
        divs.append(_view_div(section, view, view == _DEFAULT_VIEW, body))
        views[view] = {"div": f"view-{section}-{view}", "charts": ids}
    reg.panel([switch_name], views)
    return switch + "".join(divs)


def _universe_section(go: Any, merged: dict | None, cal: dict | None, reg: _Reg) -> str:
    """
    The market universe section.

    Args:
        go (Any): plotly.graph_objects.
        merged (dict | None): merge()'s result.
        cal (dict | None): summarise_candles' result.
        reg (_Reg): Where the section's switch is registered.

    Returns:
        str: HTML for the section's body.
    """
    if not merged:
        return _note(NO_DAY_COUNTS, "warn")
    coverage = merged.get("coverage") or {}
    css = "warn" if coverage.get("gap_days") else "note"
    body = [_note(UNIVERSE_INTRO)]
    if coverage.get("note"):
        body.append(_note(coverage["note"], css))
    body.append(_note(UNIVERSE_RESIDUAL))
    body.append(_combo_views(go, merged.get("counts") or {}, "universe", reg, _universe_view))
    body.append(_volume_block(go, cal))
    return "".join(body)


def _bins_total(cells: dict, price: str, nbins: int) -> tuple[list, int, float]:
    """
    Add one price's bins up over every category of a lead.

    Args:
        cells (dict): category -> {"ask": cell, "mid": cell}.
        price (str): "ask" or "mid".
        nbins (int): How many bins.

    Returns:
        tuple[list, int, float]: the summed [markets, yes, price sum] bins, the
            markets priced and the Brier sum.
    """
    bins = [[0, 0, 0.0] for _ in range(nbins)]
    n, brier = 0, 0.0
    for cell in cells.values():
        part = cell.get(price) or {}
        for i, pair in enumerate((part.get("bins") or [])[:nbins]):
            bins[i][0] += pair[0]
            bins[i][1] += pair[1]
            bins[i][2] += pair[2] if len(pair) > 2 else 0.0
        n += part.get("n", 0)
        brier += part.get("brier_sum", 0.0)
    return bins, n, brier


def _calibration_view(go: Any, cal: dict, lead: Any, price: str, ids: list) -> str:
    """
    The charts and the Brier table of one lead and price.

    Args:
        go (Any): plotly.graph_objects.
        cal (dict): summarise_candles' result.
        lead (Any): The lead, in hours.
        price (str): "ask" or "mid".
        ids (list): Collects the chart ids.

    Returns:
        str: HTML for the view.
    """
    cells = (cal.get("calibration") or {}).get(str(lead)) or {}
    width = cal.get("bin_width") or config.EDA_CALIBRATION_BIN
    nbins = cal.get("bins") or _bin_count(width)
    bins, n, brier = _bins_total(cells, price, nbins)
    if not n:
        return _note(NO_MARKETS_IN_VIEW)
    stamp = f"{lead}-{price}"
    # Each dot sits at the average price of the markets in its bin, not at the
    # bin's centre, so a bin of 1c prices is not drawn at 2.5c
    points = [(pair[2] / pair[0], pair[1] / pair[0] * 100.0, pair[0])
              for pair in bins if pair[0] > 0]
    most = max(p[2] for p in points)
    sizes = [6 + 26 * math.sqrt(p[2] / most) for p in points]
    reliability = [
        go.Scatter(x=[0, 1], y=[0, 100], mode="lines", name=LABEL_FAIR,
                   line={"color": _MORE_COLOR, "dash": "dash"}, hoverinfo="skip"),
        go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="markers",
                   name=LABEL_BINS, customdata=[p[2] for p in points],
                   marker={"size": sizes, "color": _PALETTE[0]},
                   hovertemplate=HOVER_BIN + "<extra></extra>")]
    gap = [go.Bar(x=[p[0] for p in points], y=[p[1] - p[0] * 100.0 for p in points],
                  customdata=[p[2] for p in points], marker_color=_PALETTE[1],
                  width=width * 0.8, hovertemplate=HOVER_GAP + "<extra></extra>",
                  showlegend=False)]
    rows = sorted(((cat, cell[price]["n"], cell[price]["brier_sum"])
                   for cat, cell in cells.items()
                   if (cell.get(price) or {}).get("n")),
                  key=lambda r: (-r[1], r[0]))
    table_rows = [[BRIER_ALL, f"{n:,}", f"{brier / n:.3f}"]]
    table_rows += [[cat, f"{count:,}", f"{total / count:.3f}"] for cat, count, total in rows]
    width_note = CHART_RELIABILITY_NOTE.format(
        width=round(config.EDA_MAX_BOOK_WIDTH * 100))
    return (_chart(go, CHART_RELIABILITY, width_note, reliability,
                   f"eda-cal-{stamp}-reliability", ids, x_title=AXIS_PRICE,
                   y_title=AXIS_SHARE_YES, xaxis={"range": [0, 1]},
                   yaxis={"range": [0, 100]})
            + _chart(go, CHART_GAP, CHART_GAP_NOTE, gap, f"eda-cal-{stamp}-gap", ids,
                     x_title=AXIS_PRICE, y_title=AXIS_GAP, xaxis={"range": [0, 1]})
            + _table(BRIER_HEADERS, table_rows, caption=BRIER_CAPTION, numeric=(1, 2)))


def _calibration_section(go: Any, cal: dict | None, reg: _Reg) -> str:
    """
    The price calibration section.

    Args:
        go (Any): plotly.graph_objects.
        cal (dict | None): summarise_candles' result.
        reg (_Reg): Where the Lead and Price switches are registered.

    Returns:
        str: HTML for the section's body.
    """
    if not cal:
        return _note(NO_CANDLES, "warn")
    leads = list(cal.get("leads") or [])
    if not leads:
        return _note(NO_DATA_FOR_CHART)
    default_lead = str(_DEFAULT_LEAD if _DEFAULT_LEAD in leads else leads[0])
    parts = [_note(CALIBRATION_INTRO), _note(CALIBRATION_SAMPLE_NOTE),
             _note(CALIBRATION_ASK_NOTE), _note(CALIBRATION_CLOSE_NOTE)]
    parts.append(reg.switch("lead", LEAD_SWITCH_LABEL,
                            [(str(h), _short_lead(h)) for h in leads], default_lead))
    parts.append(reg.switch("price", PRICE_SWITCH_LABEL,
                            [(p, PRICE_LABELS[p]) for p in PRICE_LABELS], _DEFAULT_PRICE))
    views = {}
    for lead in leads:
        for price in PRICE_LABELS:
            ids: list = []
            body = _calibration_view(go, cal, lead, price, ids)
            shown = str(lead) == default_lead and price == _DEFAULT_PRICE
            parts.append(_view_div("cal", f"{lead}-{price}", shown, body))
            views[f"{lead}|{price}"] = {"div": f"view-cal-{lead}-{price}", "charts": ids}
    reg.panel(["lead", "price"], views)
    coverage = cal.get("coverage") or {}
    rows = []
    for h in leads:
        seen = coverage.get(str(h), {})
        priced = seen.get("markets", 0) - seen.get("no_ask", 0) - seen.get("wide", 0)
        rows.append([_lead_label(h), f"{seen.get('markets', 0):,}", f"{priced:,}",
                     f"{seen.get('no_ask', 0):,}", f"{seen.get('no_mid', 0):,}",
                     f"{seen.get('wide', 0):,}", f"{seen.get('stale', 0):,}"])
    parts.append(_table(COVERAGE_HEADERS, rows, caption=COVERAGE_CAPTION,
                        numeric=(1, 2, 3, 4, 5, 6)))
    return "".join(parts)


def _khat_rows(stats: list) -> list[list[str]]:
    """
    Table rows for a list of k-hat figures.

    Args:
        stats (list): khat_view's _khat_stat dicts.

    Returns:
        list[list[str]]: Group, pairs, events, k-hat, interval.
    """
    rows = []
    for s in stats:
        k = s.get("k")
        low, high = s.get("ci_low"), s.get("ci_high")
        interval = NOT_AVAILABLE if low is None or high is None else f"{low:.3f} to {high:.3f}"
        rows.append([str(s.get("label", "")), f"{s.get('n', 0):,}", f"{s.get('events', 0):,}",
                     NOT_AVAILABLE if k is None else f"{k:.3f}", interval])
    return rows


def _khat_block(go: Any, khat: dict | None) -> str:
    """
    The k view: the mid-spread chart and the k-hat tables.

    Args:
        go (Any): plotly.graph_objects.
        khat (dict | None): khat_view's result, or None.

    Returns:
        str: HTML.
    """
    head = f"<h3>{_esc(KHAT_HEAD)}</h3>"
    if not khat:
        return head + _note(KHAT_NONE)
    parts = [head, _note(KHAT_INTRO.format(observations=khat.get("observations", 0),
                                           non_positive=khat.get("non_positive", 0)))]
    bins = khat.get("bins") or []
    pooled = khat.get("pooled") or {}
    k_hat, k = pooled.get("k"), khat.get("k")
    if bins:
        reach = max(b["implied"] for b in bins) * 1.05
        traces = [go.Scatter(x=[b["implied"] for b in bins], y=[b["rate"] * 100.0 for b in bins],
                             customdata=[b["n"] for b in bins], mode="markers",
                             name=LABEL_KHAT_DOTS, hovertemplate=HOVER_KHAT_BIN
                             + "<extra></extra>",
                             marker={"size": [6 + 26 * math.sqrt(b["n"] / max(x["n"] for x in bins))
                                              for b in bins], "color": _PALETTE[0]})]
        for value, name, color in ((k_hat, LABEL_KHAT_LINE, _PALETTE[1]),
                                   (k, LABEL_K_LINE, _PALETTE[2])):
            if value is not None:
                traces.append(go.Scatter(x=[0, reach], y=[0, value * reach * 100.0],
                                         mode="lines", name=name, line={"color": color},
                                         hoverinfo="skip"))
        shown_k = NOT_AVAILABLE if k is None else f"{k:g}"
        parts.append(_chart(go, CHART_KHAT, CHART_KHAT_NOTE.format(k=shown_k), traces,
                            "eda-khat", [], x_title=AXIS_MID_SPREAD, y_title=AXIS_IN_BETWEEN,
                            xaxis={"range": [0, reach]}))
    level = khat.get("confidence_level", config.KHAT_CONFIDENCE_LEVEL)
    headers = tuple(h.format(level=f"{level:.0%}") for h in KHAT_HEADERS)
    if pooled:
        parts.append(_table(headers, [[KHAT_POOLED, *_khat_rows([pooled])[0][1:]]],
                            numeric=(1, 2, 3, 4)))
    for caption, key in ((KHAT_BY_GAP, "gap_bands"), (KHAT_BY_CATEGORY, "categories")):
        rows = _khat_rows(khat.get(key) or [])
        if rows:
            parts.append(_table(headers, rows, caption=caption, numeric=(1, 2, 3, 4)))
    parts.append(_note(KHAT_NOTE))
    return "".join(parts)


def _ladders_section(go: Any, merged: dict | None, khat: dict | None) -> str:
    """
    The deadline ladders and k section.

    Args:
        go (Any): plotly.graph_objects.
        merged (dict | None): merge()'s result.
        khat (dict | None): khat_view's result or the saved one.

    Returns:
        str: HTML for the section's body.
    """
    parts = [_note(LADDERS_INTRO)]
    if not merged:
        parts.append(_note(NO_DAY_COUNTS, "warn"))
    else:
        agg = _aggregate(merged.get("counts") or {}, _VIEW_COMBOS["noncombo"])
        order = sorted(agg["phrasing"], key=lambda c: (-sum(agg["phrasing"][c].values()), c))
        if order:
            traces = [go.Bar(y=order, x=[agg["phrasing"][c].get(v, 0) for c in order],
                             orientation="h", name=CENSUS_NAMES[v],
                             marker_color=CENSUS_COLORS[v])
                      for v in CENSUS_NAMES]
            parts.append(_chart(go, CHART_CENSUS, CHART_CENSUS_NOTE, traces, "eda-census", [],
                                x_title=AXIS_MARKETS, layout={"barmode": "stack"},
                                yaxis={"autorange": "reversed", "automargin": True},
                                height=max(_CHART_HEIGHT, _BAR_ROW_PX * len(order) + 40)))
        ladders = merged.get("ladders") or {}
        if ladders:
            months = sorted(ladders)
            note = CHART_LADDERS_NOTE.format(total=merged.get("ladder_events", 0))
            parts.append(_chart(
                go, CHART_LADDERS, note,
                [go.Bar(x=months, y=[ladders[m]["events"] for m in months],
                        marker_color=_PALETTE[0], showlegend=False)],
                "eda-ladders", [], x_title=AXIS_MONTH, y_title=AXIS_LADDERS))
    try:
        parts.append(_khat_block(go, khat))
    except Exception as exc:
        # A saved view in an unexpected shape costs the k block alone
        logging.warning("Data page: the k view could not be drawn", exc_info=True)
        parts.append(f"<h3>{_esc(KHAT_HEAD)}</h3>"
                     + _note(KHAT_FAILED.format(reason=_one_line(exc)), "warn"))
    return "".join(parts)


def _quality_view(go: Any, agg: dict, view: str, ids: list) -> str:
    """
    The data quality chart for one combo view.

    Args:
        go (Any): plotly.graph_objects.
        agg (dict): _aggregate's result for the view.
        view (str): "noncombo", "combo" or "all".
        ids (list): Collects the chart ids.

    Returns:
        str: HTML for the view.
    """
    weeks = sorted(w for w, counts in agg["quality"].items() if counts[0] > 0)
    if not weeks:
        return _note(NO_MARKETS_IN_VIEW)
    traces = [go.Scatter(x=weeks, y=[100.0 * agg["quality"][w][i] / agg["quality"][w][0]
                                     for w in weeks],
                         mode="lines", name=name, line={"color": color})
              for i, (_field, name, color) in enumerate(QUALITY_SERIES, start=1)]
    return _chart(go, CHART_QUALITY, CHART_QUALITY_NOTE, traces, f"eda-{view}-quality", ids,
                  x_title=AXIS_WEEK, y_title=AXIS_MISSING, yaxis={"range": [0, 100]})


def _quality_section(go: Any, merged: dict | None, cal: dict | None, rows: list,
                     reg: _Reg) -> str:
    """
    The data quality section.

    Args:
        go (Any): plotly.graph_objects.
        merged (dict | None): merge()'s result.
        cal (dict | None): summarise_candles' result.
        rows (list): inventory()'s rows.
        reg (_Reg): Where the section's switch is registered.

    Returns:
        str: HTML for the section's body.
    """
    parts = [_note(QUALITY_INTRO)]
    if not merged:
        parts.append(_note(NO_DAY_COUNTS, "warn"))
    else:
        parts.append(_combo_views(go, merged.get("counts") or {}, "quality", reg,
                                  _quality_view))
        slices, coverage = merged.get("slices") or {}, merged.get("coverage") or {}
        parts.append(_table(DAYFILES_HEADERS, [
            [DAYFILES_ARCHIVE, f"{slices.get('archive', 0):,}"],
            [DAYFILES_LIVE, f"{slices.get('live', 0):,}"],
            [DAYFILES_SKIPPED, f"{coverage.get('live_days_skipped', 0):,}"],
            [DAYFILES_GONE, f"{coverage.get('live_days_without_slice', 0):,}"],
            [DAYFILES_DROPPED, f"{merged.get('dropped_after_cutoff', 0):,}"]], numeric=(1,)))
    if cal:
        current = str(config.CANDLESTICK_CACHE_FIELDS_VERSION)
        by_fields = cal.get("by_fields") or {}
        table = [[(CANDLEFILES_VERSION_CURRENT if v == current else CANDLEFILES_VERSION)
                  .format(version=v), f"{by_fields[v]:,}"] for v in sorted(by_fields)]
        table += [[CANDLEFILES_EMPTY, f"{cal.get('empty', 0):,}"],
                  [CANDLEFILES_UNREADABLE, f"{cal.get('unreadable', 0):,}"],
                  [CANDLEFILES_UNJOINED, f"{cal.get('unjoined', 0):,}"],
                  [CANDLEFILES_COMBOS, f"{cal.get('combos_left_out', 0):,}"]]
        parts.append(_table(CANDLEFILES_HEADERS, table, numeric=(1,)))
    depth = next((r for r in rows if r.get("name") == config.DEPTH_SNAPSHOTS_DIRNAME), None)
    detail = (depth or {}).get("detail") or {}
    if depth is None:
        parts.append(_note(DEPTH_LINE_NONE))
    elif "snapshots" not in detail:
        parts.append(_note(DEPTH_LINE_UNKNOWN))
    else:
        newest = str(detail.get("newest") or NOT_AVAILABLE)[:16].replace("T", " ")
        parts.append(_note(DEPTH_LINE.format(count=detail["snapshots"], newest=newest)))
    unreadable = (merged or {}).get("unreadable") or []
    parts.append(f"<h3>{_esc(UNREADABLE_HEAD)}</h3>")
    if unreadable:
        parts.append("<ul>" + "".join(f"<li>{_esc(u)}</li>" for u in unreadable) + "</ul>")
    else:
        parts.append(_note(UNREADABLE_NONE))
    return "".join(parts)


def _inventory_extra(row: dict) -> str:
    """
    The detail line under an inventory row's description.

    Args:
        row (dict): One inventory() row.

    Returns:
        str: The corpus meta or snapshot facts, or the row's error, joined by
            semicolons; "" when there is none.
    """
    detail = row.get("detail") or {}
    bits = []
    if "meta_error" in detail:
        bits.append(INVENTORY_META_ERROR.format(reason=detail["meta_error"]))
    if detail.get("start_date") is not None:
        bits.append(INVENTORY_START.format(value=detail["start_date"]))
    if detail.get("prefilter_tag") is not None:
        bits.append(INVENTORY_FILTER.format(value=detail["prefilter_tag"]))
    if detail.get("assembled_at") is not None:
        bits.append(INVENTORY_ASSEMBLED.format(value=str(detail["assembled_at"])[:19]))
    if "snapshots" in detail:
        bits.append(INVENTORY_SNAPSHOT_ONE if detail["snapshots"] == 1
                    else INVENTORY_SNAPSHOTS.format(count=detail["snapshots"]))
        if detail.get("newest"):
            bits.append(INVENTORY_NEWEST.format(
                value=str(detail["newest"])[:16].replace("T", " ")))
    if detail.get("unreadable_parts"):
        bits.append(INVENTORY_UNREADABLE_PARTS.format(count=detail["unreadable_parts"]))
    if row.get("error"):
        bits.append(INVENTORY_ERROR.format(reason=row["error"]))
    return "; ".join(bits)


def _inventory_section(rows: list | None) -> str:
    """
    The inventory section.

    Args:
        rows (list | None): inventory()'s rows.

    Returns:
        str: HTML for the section's body.
    """
    if not rows:
        return _note(INVENTORY_EMPTY)
    table = []
    for row in rows:
        name = str(row.get("name", "")) + ("/" if row.get("kind") == "folder" else "")
        first, last = row.get("first_day"), row.get("last_day")
        dates = INVENTORY_DATES.format(first=first, last=last) if first and last else NOT_AVAILABLE
        files = row.get("files")
        extra = _inventory_extra(row)
        what = str(row.get("description", ""))
        table.append([name, NOT_AVAILABLE if files is None else f"{files:,}",
                      _human_bytes(row.get("bytes")), dates,
                      f"{what} ({extra})" if extra else what])
    return _table(INVENTORY_HEADERS, table, numeric=(1, 2)) + _note(INVENTORY_NOTE)


# The page's one script. It only switches views: it writes the `hidden`
# attribute and aria-pressed, never any text, and asks Plotly to resize the
# charts of a view it reveals (a chart drawn while hidden has no width). It
# does nothing else, and returns quietly if Plotly did not load. __CONFIG__ is
# replaced by the switches and panels the sections registered.
_SCRIPT = """(function () {
  var cfg = __CONFIG__;
  var state = {};
  function byId(id) { return document.getElementById(id); }
  function resize(ids) {
    if (typeof Plotly === 'undefined' || !Plotly.Plots) { return; }
    ids.forEach(function (id) {
      var el = byId(id);
      if (el) { try { Plotly.Plots.resize(el); } catch (e) { /* a chart that is not drawn */ } }
    });
  }
  function show(panel) {
    var key = panel.switches.map(function (s) { return state[s]; }).join('|');
    Object.keys(panel.views).forEach(function (k) {
      var view = panel.views[k];
      var div = byId(view.div);
      if (!div) { return; }
      div.hidden = k !== key;
      if (k === key) { resize(view.charts); }
    });
  }
  function choose(name, option) {
    var sw = cfg.switches[name];
    state[name] = option;
    Object.keys(sw.options).forEach(function (o) {
      var button = byId(sw.options[o]);
      if (button) { button.setAttribute('aria-pressed', o === option ? 'true' : 'false'); }
    });
    cfg.panels.forEach(function (p) { if (p.switches.indexOf(name) >= 0) { show(p); } });
  }
  Object.keys(cfg.switches).forEach(function (name) {
    var sw = cfg.switches[name];
    state[name] = sw['default'];
    Object.keys(sw.options).forEach(function (o) {
      var button = byId(sw.options[o]);
      if (button) { button.addEventListener('click', function () { choose(name, o); }); }
    });
  });
})();"""


def _section(sec_id: str, heading: str, build, needs_charts: bool, go: Any,
             reg: _Reg) -> str:
    """
    One section of the page; a section that cannot be built says so and the rest go on.

    Args:
        sec_id (str): The section's short name (its id is "sec-<this>").
        heading (str): Its heading.
        build (Callable): build(sub_registry) -> the section's HTML body.
        needs_charts (bool): Whether it needs plotly.
        go (Any): plotly.graph_objects, or None when it could not be imported.
        reg (_Reg): The page's registry; the section's switches are added only
            when it builds.

    Returns:
        str: HTML for the section.
    """
    sub = _Reg()
    try:
        if needs_charts and go is None:
            raise RuntimeError(CHARTS_UNAVAILABLE)
        body = build(sub)
        reg.absorb(sub)
    except Exception as exc:
        logging.warning("Data page: the %s section could not be built", heading, exc_info=True)
        body = _note(SECTION_FAILED.format(section=heading, reason=_one_line(exc)), "warn")
    return f'<section id="sec-{_esc(sec_id)}"><h2>{_esc(heading)}</h2>{body}</section>'


def _render(data: dict, failures: list[str], built: str) -> str:
    """
    The whole page.

    Args:
        data (dict): What the build steps returned: "merged", "candles",
            "inventory" and "khat" (each None when its step failed or was
            not run).
        failures (list[str]): One line per failed step.
        built (str): The build time as text.

    Returns:
        str: The page's HTML.
    """
    reg = _Reg()
    go = None
    try:
        go = _plotly()
    except Exception as exc:
        logging.warning("Data page: plotly could not be imported", exc_info=True)
        failures.append(FAILURE_LINE.format(step=STEP_CHARTS, reason=_one_line(exc)))
    merged, candles = data.get("merged"), data.get("candles")
    sections = [
        _section("inventory", HEADING_INVENTORY,
                 lambda r: _inventory_section(data.get("inventory")), False, go, reg),
        _section("universe", HEADING_UNIVERSE,
                 lambda r: _universe_section(go, merged, candles, r), True, go, reg),
        _section("calibration", HEADING_CALIBRATION,
                 lambda r: _calibration_section(go, candles, r), True, go, reg),
        _section("ladders", HEADING_LADDERS,
                 lambda r: _ladders_section(go, merged, data.get("khat")), True, go, reg),
        _section("quality", HEADING_QUALITY,
                 lambda r: _quality_section(go, merged, candles, data.get("inventory") or [], r),
                 True, go, reg),
    ]
    if merged:
        counted = sum((merged.get("slices") or {}).values())
        built_line = PAGE_BUILT.format(when=built, files=counted,
                                       records=merged.get("records", 0))
    else:
        built_line = PAGE_BUILT_NO_COUNTS.format(when=built)
    failed = ""
    if failures:
        failed = (f'<div class="warn"><p>{_esc(FAILURES_HEAD)}</p><ul>'
                  + "".join(f"<li>{_esc(f)}</li>" for f in failures) + "</ul></div>")
    config_json = json.dumps({"switches": reg.switches, "panels": reg.panels},
                             separators=(",", ":")).replace("</", "<\\/")
    script = _SCRIPT.replace("__CONFIG__", config_json)
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(PAGE_TITLE)}</title>"
        f'<script src="{_esc(config.LIVE_DASHBOARD_PLOTLY_URL)}" '
        f'integrity="{_esc(config.LIVE_DASHBOARD_PLOTLY_SRI)}" crossorigin="anonymous">'
        f"</script><style>{_CSS}</style></head><body><main>"
        f"<h1>{_esc(PAGE_TITLE)}</h1><p>{glossary.terms_link_html(glossary.PAGE_DATA)}</p>"
        f'<p class="note" id="built">{_esc(built_line)}</p>{failed}'
        + "".join(sections) + glossary.terms_section_html(glossary.PAGE_DATA)
        + f'</main><script id="{_SCRIPT_ID}">{script}</script></body></html>\n')


def _write_page(out: Path, text: str) -> Path:
    """
    Write the page beside its final name and rename it into place.

    Args:
        out (Path): The page's path.
        text (str): Its HTML.

    Returns:
        Path: `out`.

    Raises:
        OSError: When the folder or file cannot be written; any earlier page
            is left as it was. Whatever is raised, the temporary file is removed.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + _TMP_SUFFIX)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out


def _step(failures: list[str], name: str, func, *args, **kwargs) -> Any:
    """
    Run one build step; a failure is logged, named for the page and becomes None.

    Args:
        failures (list[str]): Collects one line per failed step.
        name (str): The step's name on the page.
        func (Callable): The step.
        *args: Its arguments.
        **kwargs: Its keyword arguments.

    Returns:
        Any: What the step returned, or None when it raised.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logging.warning("Data page: %s failed", name.lower(), exc_info=True)
        failures.append(FAILURE_LINE.format(step=name, reason=_one_line(exc)))
        return None


def _count_days(cache_dir: Path, series_categories: dict | None, workers: int) -> dict:
    """
    The day-file counts: every summary, built or reused, merged.

    Args:
        cache_dir (Path): The cache folder.
        series_categories (dict | None): Kalshi's /series listing.
        workers (int): The most worker processes for the day files.

    Returns:
        dict: merge()'s result.
    """
    summaries = load_or_build_summaries(cache_dir, workers=workers)
    return merge(summaries, series_categories=series_categories,
                 candle_set=candle_tickers(cache_dir))


def _khat_for_page(cache_dir: Path, sweep: Any, series_categories: dict | None,
                   failures: list[str]) -> dict | None:
    """
    The k view: this run's, saved for later, or the last one saved.

    Args:
        cache_dir (Path): The cache folder.
        sweep (Any): A BacktestSweep, or None.
        series_categories (dict | None): Kalshi's /series listing.
        failures (list[str]): Collects failed steps.

    Returns:
        dict | None: khat_view's result for a sweep (None when the sweep has
            no observations), else the saved view (None when none is saved).
    """
    if sweep is None:
        return load_khat(cache_dir)
    view = _step(failures, STEP_KHAT, khat_view, sweep, series_categories=series_categories)
    if view is not None:
        _step(failures, STEP_KHAT_SAVE, save_khat, cache_dir, view)
    return view


def build_eda_page(*, sweep: Any = None, series_categories: dict | None = None,
                   workers: int = 1, cache_dir: Path | None = None,
                   out_path: Path | None = None) -> Path | None:
    """
    Build the Data page from the cache and write it.

    Counts the day files (reusing every saved summary that still matches),
    joins them to the candle files, lists the cache folder and takes the k
    figures from the sweep (and saves them) or, with no sweep, from the last
    saved view; then writes the page to a temporary file beside `out_path` and
    renames it into place. Reporting only, and it never raises: a step that
    fails is logged as one WARNING with its traceback, named on the page, and
    the rest of the page is still written.

    Args:
        sweep (Any): A backtester.BacktestSweep for the k view, or None to show
            the last saved one.
        series_categories (dict | None): Kalshi's /series listing; None reads
            the cached copy (historical.load_series_categories(None)).
        workers (int): The most worker processes for the day files (1 counts in
            this process). With more than one, the caller must be under
            `if __name__ == "__main__":`.
        cache_dir (Path | None): The cache folder; None is historical.CACHE_DIR,
            read now.
        out_path (Path | None): Where to write the page; None is
            config.PROJECT_ROOT / config.EDA_DASHBOARD_FILENAME, read now.

    Returns:
        Path | None: The page's path, or None when nothing could be written.
    """
    out: Path | None = None
    failures: list[str] = []
    try:
        cache = Path(cache_dir) if cache_dir is not None else Path(historical.CACHE_DIR)
        out = (Path(out_path) if out_path is not None
               else Path(config.PROJECT_ROOT) / config.EDA_DASHBOARD_FILENAME)
        if series_categories is None:
            series_categories = _step(failures, STEP_LISTING,
                                      historical.load_series_categories, None)
        data: dict = {"merged": None, "candles": None}
        data["merged"] = _step(failures, STEP_SUMMARIES, _count_days, cache,
                               series_categories, workers)
        if data["merged"] is not None:
            data["candles"] = _step(failures, STEP_CANDLES, summarise_candles, cache,
                                    data["merged"]["candles"])
        data["inventory"] = _step(failures, STEP_INVENTORY, inventory, cache)
        data["khat"] = _khat_for_page(cache, sweep, series_categories, failures)
        built = datetime.now(UTC).strftime(BUILT_TIME_FORMAT)
        text = _render(data, failures, built)
    except Exception as exc:
        logging.warning("Data page: the page could not be put together", exc_info=True)
        if out is None:
            return None
        text = _fallback_page(_one_line(exc))
    try:
        return _write_page(out, text)
    except Exception:
        logging.warning("Data page: the page could not be written to %s", out, exc_info=True)
        return None


def _fallback_page(reason: str) -> str:
    """
    A page that only says the build failed.

    Args:
        reason (str): One line saying why.

    Returns:
        str: HTML.
    """
    return ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{_esc(PAGE_TITLE)}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{_esc(PAGE_TITLE)}</h1>{_note(PAGE_FAILED.format(reason=reason), 'warn')}"
            "</main></body></html>\n")


def _configure_logging() -> None:
    """
    Send the log to kalshi_eda.log in PROJECT_ROOT, set up as backtest.py sets up its log.

    No console handler: the path of the page is printed instead.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # delay=True: the file is opened on the first message, so a run that
            # exits before logging leaves no file (as backtest.py and main.py do)
            logging.handlers.RotatingFileHandler(
                Path(config.PROJECT_ROOT) / EDA_LOG_FILENAME, maxBytes=20 * 1024 * 1024,
                backupCount=3, delay=True),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """
    Rebuild the Data page from the cache, without a backtest.

    It shows the k figures the last backtest saved (none yet is said on the
    page). The page is written to config.PROJECT_ROOT /
    config.EDA_DASHBOARD_FILENAME and its path printed.

    Args:
        argv (list[str] | None): Command-line arguments; None reads sys.argv.

    Returns:
        int: 0 when a page was written, 1 when none could be.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.eda",
        description="Rebuild the Data page from backtest_cache/ (no backtest, no request "
                    "to Kalshi). Reuses every saved day-file summary that still matches.")
    parser.add_argument(
        "--workers", type=int, default=_default_workers(),
        help="Worker processes for counting day files (default: one less than this "
             f"machine's CPUs, at most {config.EDA_MAX_WORKERS})")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    _configure_logging()
    page = build_eda_page(workers=min(args.workers, config.EDA_MAX_WORKERS))
    if page is None:
        print(MAIN_NO_PAGE)
        return 1
    print(page)
    return 0


def _default_workers() -> int:
    """
    The default number of worker processes.

    Returns:
        int: One less than the CPU count, at least 1 and at most config.EDA_MAX_WORKERS.
    """
    return max(1, min(config.EDA_MAX_WORKERS, (os.cpu_count() or 2) - 1))


if __name__ == "__main__":
    # Under this guard because summary workers are spawned and import this
    # module again; without it each worker would run main() too.
    raise SystemExit(main())
