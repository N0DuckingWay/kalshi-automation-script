"""
File: treasury.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Downloads the 8-week U.S. Treasury bill's auction yields from the Treasury's
    Fiscal Data API and turns them into the risk-free rate the backtest
    dashboard's Sharpe and Sortino ratios subtract: on each day of a curve, the
    yield of the most recent auction on or before that day
    (RiskFreeRates.annual_on). Every successful download is saved under
    backtest_cache/; when the API cannot be reached the last saved copy is used,
    and with no copy the rate is reported unavailable (the dashboard then
    subtracts 0% and its header says so). Reporting only: nothing sizes, prices
    or settles on it, and no live-trading module imports this one.

Dependencies:
    Imports TREASURY_AUCTIONS_URL, RISK_FREE_BILL_TERM, RISK_FREE_RATE_FIELD and
    the TREASURY_API_* bounds from config.py, api_call_with_retry from _http.py
    (the retried read-only GET wrapper), and CACHE_DIR, _load_json_cache,
    _save_json_cache and _exception_summary from historical.py. Imported by
    backtest.py (load_risk_free_rates) and dashboard.py (RiskFreeRates,
    SOURCE_CACHE and day_numbers — the last so the dashboard's
    deployed-capital hurdle reads a curve's dates on the same day numbering
    as the yield lookup, converting them once).

Notes:
    The endpoint is "Treasury Securities Auctions Data"
    (v1/accounting/od/auctions_query), paged by page[number]/page[size] with
    meta["total-pages"]. The rate read is high_investment_rate (see config).
    Cash-management bills are filtered out, so the series is the regular
    weekly bill. A date before the first auction (2018-10-16) takes that
    auction's yield. The cache stores the API's own records, so the one
    parser (_parse_auctions) reads a download and a cached copy alike.
    Dates are looked up as day numbers (day_numbers: date.toordinal,
    vectorized); annual_on converts and looks up, annual_on_days looks up
    day numbers a caller already converted.
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
            API could not be reached: the copy an earlier run saved) or
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

        A date before the first auction takes the first auction's yield. The
        result is positional — element i belongs to dates[i] — so a caller
        subtracts it from a returns Series by position, never by index.

        Args:
            dates: Dates, one per return row — a column of datetime.date
                (_build_equity_curve's), or a DatetimeIndex (the ^GSPC series'
                trading days; a tz-aware one is read on its own calendar date).

        Returns:
            np.ndarray: One annual decimal yield per date; all zeros when the
                rates are unavailable.
        """
        return self.annual_on_days(day_numbers(dates))

    def annual_on_days(self, days: np.ndarray) -> np.ndarray:
        """
        annual_on, for dates already converted to day numbers (day_numbers).

        For a caller that reads the same dates twice — the dashboard's
        deployed-capital hurdle converts a curve's dates once and looks up
        both the yield and the capital in open trades on them.

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

    The one day numbering RiskFreeRates' lookups use, and the one the
    dashboard's capital-deployed helper matches trades' entry and exit dates
    against (dashboard._deployed_on_days), so the yield and the capital a
    curve's row is charged on are read on the same calendar day.

    A datetime64 column or a DatetimeIndex is converted in numpy; anything
    else — _build_equity_curve's column of datetime.date, or datetimes and
    pd.Timestamps — by each value's own toordinal(), which reads a timestamp's
    calendar date exactly as datetime.date() does (a tz-aware one on its own
    wall clock) and costs about 60% of a pd.to_datetime pass over the same
    column. A value without a toordinal (a date string) falls back to
    pd.to_datetime.

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

    A record without a readable date or yield, or with a yield outside
    [0%, 100%), is skipped and counted. Two records on one date are averaged.

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
            rate = float(row[RISK_FREE_RATE_FIELD]) / 100.0
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not (math.isfinite(rate) and 0.0 <= rate < 1.0):
            skipped += 1
            continue
        by_date.setdefault(day, []).append(rate)
    if skipped:
        logging.info("Treasury %s bill auctions: %d record(s) without a readable %s skipped",
                     RISK_FREE_BILL_TERM, skipped, RISK_FREE_RATE_FIELD)
    if not by_date:
        raise ValueError(f"no {RISK_FREE_BILL_TERM} auction with a readable {RISK_FREE_RATE_FIELD}")
    return tuple((day, sum(v) / len(v)) for day, v in sorted(by_date.items()))


def _read_cache() -> RiskFreeRates | None:
    """
    The copy an earlier run saved, or None when there is none usable.

    A copy saved for another bill term or rate field is not used: it describes
    a different rate from the one config names.

    Returns:
        RiskFreeRates | None: source SOURCE_CACHE, stamped with the copy's own
            download time; None when the file is absent, unreadable, for
            another term or field, or holds no readable auction.
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
    return RiskFreeRates(auctions, SOURCE_CACHE, fetched_at)


def load_risk_free_rates() -> RiskFreeRates:
    """
    The bill's auction yields for the dashboard's Sharpe and Sortino, never raising.

    Downloads every auction from the Fiscal Data API and saves the records
    under backtest_cache/. On any failure — the API unreachable, an error
    status after retries, a body that cannot be read — it falls back to the
    last saved copy, and with none to SOURCE_UNAVAILABLE (no auctions: the
    dashboard subtracts 0% and says so). Always tries the API first: a saved
    copy is only a fallback, never served in preference to a fresh download.

    Returns:
        RiskFreeRates: The yields, with their source and download time.
    """
    try:
        rows = _fetch_rows()
        auctions = _parse_auctions(rows)
    except Exception as exc:  # reporting only — never end a backtest over it
        cached = _read_cache()
        if cached is not None:
            logging.warning(
                "Treasury Fiscal Data unavailable (%s) — using the %s bill yields downloaded "
                "%s (latest auction %s)", _exception_summary(exc), RISK_FREE_BILL_TERM,
                cached.fetched_at.isoformat(timespec="minutes"), cached.latest[0])
            return cached
        logging.warning(
            "Treasury Fiscal Data unavailable (%s) and no copy is saved — the dashboard's "
            "Sharpe and Sortino ratios subtract 0%%", _exception_summary(exc))
        return RiskFreeRates((), SOURCE_UNAVAILABLE, None)
    now = datetime.now(UTC)
    try:
        _save_json_cache(_RATES_CACHE, {
            "fetched_at": now.isoformat(), "term": RISK_FREE_BILL_TERM,
            "field": RISK_FREE_RATE_FIELD, "records": rows,
        })
    except OSError as exc:
        logging.warning("Could not save the %s bill yields to %s (%s) — a run that cannot "
                        "reach the API will not find them", RISK_FREE_BILL_TERM, _RATES_CACHE,
                        _exception_summary(exc))
    (first_day, _), (last_day, last_rate) = auctions[0], auctions[-1]
    logging.info("Risk-free rate: %d %s Treasury bill auctions from Treasury Fiscal Data, "
                 "%s to %s (latest %.3f%%)", len(auctions), RISK_FREE_BILL_TERM,
                 first_day, last_day, last_rate * 100)
    return RiskFreeRates(auctions, SOURCE_API, now)
