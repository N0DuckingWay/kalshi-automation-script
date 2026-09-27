"""
File: treasury.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Downloads the 8-week Treasury bill's auction yields from the Treasury's
    Fiscal Data API: the risk-free rate the backtest dashboard's Sharpe and
    Sortino subtract, on each day the yield of the latest auction on or
    before it. A download is saved under backtest_cache/; if it fails, the
    saved copy is used, and with none the rate is unavailable (the dashboard
    subtracts 0% and says so). Reporting only: no live-trading module
    imports this one.

Dependencies:
    config (URL, bill term, rate field, TREASURY_API_* bounds), _http
    (api_call_with_retry) and historical (cache helpers). Imported by
    backtest (load_risk_free_rates) and dashboard (RiskFreeRates,
    SOURCE_CACHE, day_numbers).

Notes:
    Cash-management bills are filtered out. A date before the first auction
    takes its yield. The cache stores the API's own records, so one parser
    (_parse_auctions) reads a download and a cached copy alike.
"""
import json
import logging
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ._http import api_call_with_retry
from .config import (
    RISK_FREE_BILL_TERM,
    RISK_FREE_RATE_FIELD,
    TREASURY_API_MAX_PAGES,
    TREASURY_API_PAGE_SIZE,
    TREASURY_API_TIMEOUT_SECONDS,
    TREASURY_AUCTIONS_URL,
)
from .historical import CACHE_DIR, _exception_summary, _load_json_cache, _save_json_cache

# Every successful download, for a later run that cannot reach the API
_RATES_CACHE = CACHE_DIR / "treasury_bill_rates.json"

# RiskFreeRates.source
SOURCE_API = "api"
SOURCE_CACHE = "cache"
SOURCE_UNAVAILABLE = "unavailable"

# date.toordinal() of numpy's datetime64 day 0, so the two day numberings meet
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


@dataclass(frozen=True)
class RiskFreeRates:
    """
    The bill's auction yields, and where they came from.

    Attributes:
        auctions (tuple[tuple[date, float], ...]): (auction date, yield as an
            annual decimal — 0.04071 for 4.071%), one per date, ascending.
            Empty when source is SOURCE_UNAVAILABLE.
        source (str): SOURCE_API (downloaded by this run), SOURCE_CACHE (the
            download failed: the copy an earlier run saved) or
            SOURCE_UNAVAILABLE (neither).
        fetched_at (datetime | None): When the yields were downloaded, in UTC
            (an earlier run's time for a cached copy); None when unavailable.
    """
    auctions: tuple[tuple[date, float], ...]
    source: str
    fetched_at: datetime | None
    _days: np.ndarray = field(init=False, repr=False, compare=False)
    _rates: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index the auctions by day number once, for annual_on's lookups."""
        object.__setattr__(self, "_days", np.array(
            [d.toordinal() for d, _ in self.auctions], dtype=np.int64))
        object.__setattr__(self, "_rates", np.array(
            [r for _, r in self.auctions], dtype=float))

    @property
    def latest(self) -> tuple[date, float] | None:
        """The most recent auction's (date, yield), or None when there is none."""
        return self.auctions[-1] if self.auctions else None

    def annual_on(self, dates) -> np.ndarray:
        """
        The annual yield in force on each date: the latest auction on or before it.

        The result is positional — element i belongs to dates[i] — so a caller
        subtracts it by position, never by index.

        Args:
            dates: Dates, one per return row (anything day_numbers reads).

        Returns:
            np.ndarray: One annual decimal yield per date; all zeros when the
                rates are unavailable.
        """
        return self.annual_on_days(day_numbers(dates))

    def annual_on_days(self, days: np.ndarray) -> np.ndarray:
        """
        annual_on, for dates already converted by day_numbers.

        Args:
            days (np.ndarray): Proleptic Gregorian ordinals (day_numbers'),
                one per return row.

        Returns:
            np.ndarray: One annual decimal yield per day, positional; all
                zeros when the rates are unavailable.
        """
        if not self.auctions:
            return np.zeros(len(days))
        pos = np.searchsorted(self._days, days, side="right") - 1
        return self._rates[np.maximum(pos, 0)]


def day_numbers(dates) -> np.ndarray:
    """
    Dates as proleptic Gregorian ordinals (date.toordinal), vectorized.

    The one day numbering the yield lookup and dashboard._deployed_on_days
    share, so a row's yield and capital are read on the same calendar day.
    A datetime64 Series or a DatetimeIndex is converted in numpy; anything
    else by each value's own toordinal() (a tz-aware timestamp on its own
    wall clock), falling back to pd.to_datetime when a value has none (a
    string, or a numpy datetime64 array's elements).

    Args:
        dates: Dates — a column of datetime.date, of datetimes or
            pd.Timestamps, or a DatetimeIndex (a tz-aware one is read on its
            own calendar date).

    Returns:
        np.ndarray: One int64 ordinal per date, positional.
    """
    if isinstance(dates, pd.DatetimeIndex | pd.Series) and dates.dtype.kind == "M":
        idx = pd.DatetimeIndex(dates)
    else:
        values = dates if hasattr(dates, "__len__") else list(dates)
        try:
            return np.fromiter((d.toordinal() for d in values), np.int64, len(values))
        except (AttributeError, TypeError, ValueError):
            idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize().values.astype("datetime64[D]").astype(np.int64) + _EPOCH_ORDINAL


def _page_url(page: int) -> str:
    """
    The auctions_query URL for one page of the bill's auctions, oldest first.

    Args:
        page (int): 1-based page number.

    Returns:
        str: The full URL, parameters percent-encoded.
    """
    params = {
        "fields": f"auction_date,{RISK_FREE_RATE_FIELD}",
        "filter": f"security_term:eq:{RISK_FREE_BILL_TERM},cash_management_bill_cmb:eq:No",
        "sort": "auction_date",
        "page[size]": str(TREASURY_API_PAGE_SIZE),
        "page[number]": str(page),
    }
    return f"{TREASURY_AUCTIONS_URL}?{urllib.parse.urlencode(params)}"


def _get_json(url: str) -> Any:
    """
    GET one Fiscal Data page and parse its JSON body. Single-shot: the caller
    wraps it in api_call_with_retry.

    Args:
        url (str): The page's URL (_page_url).

    Returns:
        Any: The parsed body.

    Raises:
        urllib.error.HTTPError: On a non-2xx status (its .status is what
            api_call_with_retry reads).
        urllib.error.URLError, TimeoutError: On a transport failure.
        ValueError: If the body is not JSON.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=TREASURY_API_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def _fetch_rows() -> list[dict]:
    """
    Download every auction record of the bill, following the API's pages.

    Returns:
        list[dict]: The API's records ({"auction_date", RISK_FREE_RATE_FIELD}),
            as served.

    Raises:
        ValueError: If a page is not an object carrying a data list, or the
            pages run past TREASURY_API_MAX_PAGES.
        Exception: Whatever _get_json raised on its last attempt.
    """
    rows: list[dict] = []
    for page in range(1, TREASURY_API_MAX_PAGES + 1):
        # A read-only GET, so retried like every other data read
        body = api_call_with_retry(_get_json, _page_url(page))
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ValueError(f"auctions_query page {page} carried no data list")
        rows.extend(r for r in body["data"] if isinstance(r, dict))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        if page >= int(meta.get("total-pages") or 0):
            return rows
    raise ValueError(f"auctions_query ran past {TREASURY_API_MAX_PAGES} pages")


def _parse_auctions(rows) -> tuple[tuple[date, float], ...]:
    """
    Read (auction date, annual decimal yield) pairs out of the API's records.

    A record without a readable date or yield (a boolean, or a number too
    large for a float, included) or with a yield outside [0%, 100%) is
    skipped and counted. Two records on one date are averaged.

    Args:
        rows: The API's records (a download's, or a cached copy's).

    Returns:
        tuple[tuple[date, float], ...]: One pair per date, ascending.

    Raises:
        ValueError: If no record is readable.
    """
    by_date: dict[date, list[float]] = {}
    skipped = 0
    for row in rows:
        try:
            day = date.fromisoformat(row["auction_date"])
            raw = row[RISK_FREE_RATE_FIELD]
            # A boolean is not a yield: float(True) is 1.0, which would read
            # as a 1% auction rather than as an unreadable record
            rate = None if isinstance(raw, bool) else float(raw) / 100.0
        except (KeyError, TypeError, ValueError, OverflowError):
            skipped += 1
            continue
        if rate is None or not (math.isfinite(rate) and 0.0 <= rate < 1.0):
            skipped += 1
            continue
        by_date.setdefault(day, []).append(rate)
    if skipped:
        logging.info("Treasury %s bill auctions: %d record(s) skipped — an unreadable "
                     "auction_date, an unreadable %s, or a yield outside [0%%, 100%%)",
                     RISK_FREE_BILL_TERM, skipped, RISK_FREE_RATE_FIELD)
    if not by_date:
        raise ValueError(f"no {RISK_FREE_BILL_TERM} auction with a readable {RISK_FREE_RATE_FIELD}")
    return tuple((day, sum(v) / len(v)) for day, v in sorted(by_date.items()))


def _read_cache() -> RiskFreeRates | None:
    """
    The copy an earlier run saved, or None when there is none usable.

    A copy for another bill term or rate field is not used, nor one whose
    download time is naive or has no UTC instant (it is returned in UTC, for
    the dashboard header).

    Returns:
        RiskFreeRates | None: source SOURCE_CACHE, stamped with the copy's
            download time in UTC; None when there is no usable copy.
    """
    cached = _load_json_cache(_RATES_CACHE)
    if not isinstance(cached, dict) or cached.get("term") != RISK_FREE_BILL_TERM \
            or cached.get("field") != RISK_FREE_RATE_FIELD:
        return None
    try:
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
        auctions = _parse_auctions(cached["records"])
    except (KeyError, TypeError, ValueError):
        return None
    if fetched_at.tzinfo is None:
        return None
    try:
        # Normalized here so no later reader can overflow converting it
        fetched_at = fetched_at.astimezone(UTC)
    except OverflowError:
        return None
    return RiskFreeRates(auctions, SOURCE_CACHE, fetched_at)


def load_risk_free_rates() -> RiskFreeRates:
    """
    The bill's auction yields for the dashboard's Sharpe and Sortino, never raising.

    Always tries the API first and saves what it downloads; on any failure
    falls back to the saved copy, then to SOURCE_UNAVAILABLE. It never
    raises — it runs after a multi-hour backtest — so a failed download,
    cache read or save each degrade with a WARNING. The worst case is time:
    an unresponsive host costs 6 attempts of TREASURY_API_TIMEOUT_SECONDS per
    resolved address plus 62 s of backoff — about 4 minutes for one address.

    Returns:
        RiskFreeRates: The yields, with their source and download time.
    """
    try:
        rows = _fetch_rows()
        auctions = _parse_auctions(rows)
    except Exception as exc:  # reporting only — never end a backtest over it
        try:
            cached = _read_cache()
        except Exception as read_exc:
            # _load_json_cache catches a corrupt file, not e.g. a
            # PermissionError or RecursionError
            logging.warning("Could not read the saved %s bill yields at %s (%s)",
                            RISK_FREE_BILL_TERM, _RATES_CACHE, _exception_summary(read_exc))
            cached = None
        if cached is not None:
            logging.warning(
                "Treasury Fiscal Data unavailable (%s) — using the %s bill yields downloaded "
                "%s (latest auction %s)", _exception_summary(exc), RISK_FREE_BILL_TERM,
                cached.fetched_at.isoformat(timespec="minutes"), cached.latest[0])
            return cached
        logging.warning(
            "Treasury Fiscal Data unavailable (%s) and no usable earlier download is saved "
            "— the dashboard's Sharpe and Sortino ratios subtract 0%%",
            _exception_summary(exc))
        return RiskFreeRates((), SOURCE_UNAVAILABLE, None)
    now = datetime.now(UTC)
    try:
        _save_json_cache(_RATES_CACHE, {
            "fetched_at": now.isoformat(), "term": RISK_FREE_BILL_TERM,
            "field": RISK_FREE_RATE_FIELD, "records": rows,
        })
    except Exception as exc:  # a failed save must not cost the run its fresh yields
        logging.warning("Could not save the %s bill yields to %s (%s) — a run that cannot "
                        "reach the API will not find them", RISK_FREE_BILL_TERM, _RATES_CACHE,
                        _exception_summary(exc))
    (first_day, _), (last_day, last_rate) = auctions[0], auctions[-1]
    logging.info("Risk-free rate: %d %s Treasury bill auctions from Treasury Fiscal Data, "
                 "%s to %s (latest %.3f%%)", len(auctions), RISK_FREE_BILL_TERM,
                 first_day, last_day, last_rate * 100)
    return RiskFreeRates(auctions, SOURCE_API, now)
