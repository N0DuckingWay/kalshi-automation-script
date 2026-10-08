"""
File: live_dashboard.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Serves the dashboard's two tabs on this computer. 127.0.0.1:8766
    (config.LIVE_DASHBOARD_PORT) serves the tab page and the Live trading
    tab's data: GET /api/live reads the account from Kalshi through
    live_portfolio on every load and answers everything the tab shows, as
    JSON (payload). 127.0.0.1:8767 (config.LIVE_BACKTEST_PORT) serves the
    backtest page, unchanged, with the chunk files beside it; the Backtest
    tab loads it in a frame the first time the tab is chosen. Run it with

        python3 -m kalshi_betting.live_dashboard [--no-browser]

    It opens the page in the browser unless told not to; Ctrl-C stops both
    servers. When its port is already taken by this checkout's own live
    dashboard, running this checkout's current code, it opens that page and
    exits, starting nothing; otherwise a taken port is refused (exit 2).

Dependencies:
    config (the LIVE_DASHBOARD_* and LIVE_BACKTEST_PORT settings, the Plotly
    build and its hash, PROJECT_ROOT, DASHBOARD_FILENAME,
    DASHBOARD_FILES_DIRNAME, LIVE_DASHBOARD_PERIODS, LIVE_CANDLE_DAY_ZONE,
    RISK_FREE_BILL_TERM, count_text); live_portfolio (build_live_view,
    append_snapshot, trade_log_paths, OTHER_BETS and the view's types);
    historical (build_prod_live_client, the production client, built on the
    first read; load_series_categories, the categories the bot's purchases
    are filed under); treasury (load_risk_free_rates, the T-bill yields the
    Sharpe and Sortino ratios subtract; RiskFreeRates and SOURCE_CACHE);
    _http (api_error_summary, the one-line description of a failed read).
    Nothing imports this module: a person runs it.

Notes:
    Read-only: it sends Kalshi read-only requests only (through
    live_portfolio and historical) and never loads the order path. It
    answers only GET and HEAD, only requests addressed to its own host and
    port (so a web page on another name cannot reach it through DNS), sends
    no CORS headers, and its tab page may never be shown in a frame. The
    account data also needs the page's own request header and, from a
    browser, a same-origin request, and the tab page itself is refused when
    another web site sent the browser to it, so no other web page can make
    it read the account. Its own Kalshi client refuses every method but GET.
    The backtest page is served from the second port, a different web
    origin, so its scripts (and the chart library it loads from the web)
    cannot read the account data; it may be framed only by the tab page,
    in a sandboxed frame that cannot navigate the tab page, and no other
    web site may load it or its chunk files except by navigating to it.

    Every number and sentence the page shows is formatted here (payload);
    the page's script only places them. One read of the account runs at a
    time: a request that arrives during a read waits for that read's result
    (_SingleFlight).

    Known limits, recorded rather than fixed: a read has no overall deadline
    (each request to Kalshi has its timeouts, but their retries can stretch
    one read past config.LIVE_DASHBOARD_BUILD_WAIT_SECONDS, and requests that
    arrive meanwhile are answered 503 until it ends); and a category's color
    follows its place in live_portfolio's group order (most money put in
    first), so a purchase that changes that order can change the colors
    between two reads, while within one read every chart and table agrees.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import http.client
import json
import logging
import logging.handlers
import math
import os
import signal
import socket
import threading
import urllib.parse
import urllib.request
import webbrowser
from base64 import b64encode
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from email.utils import formatdate, parsedate_to_datetime
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import config, historical, live_portfolio, treasury
from ._http import api_error_summary

# What GET /health says this server is, so a second start can tell it from
# another program on the port
_APP = "kalshi-live-dashboard"

# The colors of the named category bands: the dataviz reference palette's
# light-mode slots 1 to 6, in that order, which keeps neighbouring bands
# apart for color-blind readers too. A category keeps its slot on every
# chart of the page.
_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
# Other bets (violet, the reference's slot 7), the categories past the named
# ones (muted gray) and cash (the light axis gray, so the money not in a bet
# recedes)
_OTHER_BETS_COLOR, _MORE_COLOR, _CASH_COLOR = "#4a3aa7", "#898781", "#c3c2b7"
# The chart surface, and the colors of a gain and a loss
_SURFACE, _UP, _DOWN = "#fcfcfb", "#006300", "#d03b3b"
# The reference chrome: the page behind the charts, primary and secondary
# ink, gridlines and the axis line
_PLANE, _INK, _INK_2, _GRID, _AXIS = "#f9f9f7", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# The bands that are not a single category
CASH = "Cash"
MORE_CATEGORIES = "More categories"

# The six statistic cards: (key, label), in page order
CARD_LABELS = (("return", "Total return"), ("pnl", "Profit"), ("sharpe", "Sharpe"),
               ("sortino", "Sortino"), ("mean", "Mean trade"), ("median", "Median trade"))

# What each card's label says when pointed at (each value has a tooltip of its own)
_CARD_HINTS = {
    "return": "Growth over the period, with deposits and withdrawals taken out",
    "pnl": "Dollars gained or lost over the period",
    "sharpe": "Return for each unit of day-to-day swing, scaled to a year",
    "sortino": "Like Sharpe, counting only the losing days as risk",
    "mean": "Average return of the bot's purchases in the period",
    "median": "Middle return of the bot's purchases in the period",
}

# Kalshi's days close at midnight here; the chart's times are in this zone
_DAY_ZONE = ZoneInfo(config.LIVE_CANDLE_DAY_ZONE)

# A chart's options: it resizes with the window, without Plotly's logo
_CHART_CONFIG = {"responsive": True, "displaylogo": False}

_CENT = Decimal("0.01")


# ---- formatting -------------------------------------------------------------

def _finite(value: Any) -> float | None:
    """
    A number as a finite float, or None.

    Args:
        value (Any): A float, int, Decimal or None.

    Returns:
        float | None: The number; None for None, NaN or an infinity.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pct(value: Any) -> str:
    """
    A return as signed percent text: "+12.3%", "-4.1%", "0.0%", or "—" when there is none.

    Args:
        value (Any): The return as a fraction (0.123 for 12.3%), or None.

    Returns:
        str: The text.
    """
    number = _finite(value)
    if number is None:
        return "—"
    text = f"{number:+.1%}"
    return "0.0%" if text in ("+0.0%", "-0.0%") else text


def _money(value: Any) -> str:
    """
    Dollars to the cent: "$1,234.56".

    A Decimal is rounded half up, as money is ($90.9050 shows as $90.91);
    a float as Python rounds it.

    Args:
        value (Any): The amount, a Decimal or a float.

    Returns:
        str: The text.
    """
    if isinstance(value, Decimal):
        value = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    return f"${value:,.2f}"


def _signed_money(value: Any) -> str:
    """
    A gain or loss in dollars: "+$4.56", "-$4.56", or "$0.00".

    Args:
        value (Any): The amount, or None.

    Returns:
        str: The text; "—" for None.
    """
    number = _finite(value)
    if number is None:
        return "—"
    cents = round(number, 2)
    if cents == 0:
        return "$0.00"
    return f"{'+' if cents > 0 else '-'}${abs(cents):,.2f}"


def _ratio(value: Any) -> str:
    """
    A Sharpe or Sortino ratio to two decimals, or "—" when there is none.

    Args:
        value (Any): The ratio, or None.

    Returns:
        str: The text.
    """
    number = _finite(value)
    if number is None:
        return "—"
    text = f"{number:.2f}"
    return "0.00" if text == "-0.00" else text


def _tone(text: str) -> str:
    """
    How a figure's text is colored: "up" for a gain, "down" for a loss, "flat" otherwise.

    Read from the text as shown, so a figure that rounds to zero is never
    colored as a gain or a loss.

    Args:
        text (str): The figure as shown.

    Returns:
        str: "up", "down" or "flat".
    """
    try:
        number = float(text.replace("$", "").replace("%", "").replace(",", ""))
    except ValueError:
        return "flat"
    return "up" if number > 0 else "down" if number < 0 else "flat"


def _price(price: Decimal | None) -> str:
    """
    One contract's value: "$0.45", "$0.455", or "—" when never priced.

    Args:
        price (Decimal | None): The value of one contract of the side held.

    Returns:
        str: The text, to at most four decimals and at least two.
    """
    if price is None:
        return "—"
    text = f"{float(price):.4f}".rstrip("0")
    whole, _, cents = text.partition(".")
    return f"${whole}.{cents.ljust(2, '0')}"


def _count_words(count: Any, one: str, many: str) -> str:
    """
    A count and its noun: "1 purchase", "12 purchases", "12.5 contract pairs".

    Args:
        count (Any): The count.
        one (str): The noun for exactly one.
        many (str): The noun for any other count.

    Returns:
        str: The text.
    """
    text = config.count_text(float(count))
    return f"{text} {one if text == '1' else many}"


def _chart_time(moment: datetime) -> str:
    """
    A moment as the chart shows it: New York wall-clock time, where Kalshi's days close.

    Plotly reads a date without its time zone, so the zone is applied here.

    Args:
        moment (datetime): An aware moment.

    Returns:
        str: "YYYY-MM-DD HH:MM:SS".
    """
    return moment.astimezone(_DAY_ZONE).strftime("%Y-%m-%d %H:%M:%S")


def _day(moment: datetime) -> str:
    """
    A moment's New York date, short: "Oct 8".

    Args:
        moment (datetime): An aware moment.

    Returns:
        str: The month's short name and the day.
    """
    local = moment.astimezone(_DAY_ZONE)
    return f"{local:%b} {local.day}"


def _span(first: datetime, last: datetime) -> tuple[str, str]:
    """
    A period's dates and length: ("Sep 28 – Oct 8, 2026", "10 days").

    Dates are New York's, as Kalshi's days are.

    Args:
        first (datetime): The period's first moment.
        last (datetime): Its last moment (the read).

    Returns:
        tuple[str, str]: The dates, and the length in whole days, rounded
            ("under a day" for less than one).
    """
    a, b = first.astimezone(_DAY_ZONE), last.astimezone(_DAY_ZONE)
    if a.date() == b.date():
        span = f"{_day(b)}, {b.year}"
    elif a.year == b.year:
        span = f"{_day(a)} – {_day(b)}, {b.year}"
    else:
        span = f"{_day(a)}, {a.year} – {_day(b)}, {b.year}"
    days = (last - first).total_seconds() / 86_400
    length = "under a day" if days < 1 else _count_words(round(days), "day", "days")
    return span, length


def _range_note(first: datetime, last: datetime) -> str:
    """
    A period's span as the line beside the period buttons shows it: "Sep 28 – Oct 8, 2026 (10 days)".

    Args:
        first (datetime): The period's first moment.
        last (datetime): Its last moment (the read).

    Returns:
        str: The dates, then the length in parentheses.
    """
    span, length = _span(first, last)
    return f"{span} ({length})"


def _read_time(read_at: datetime) -> str:
    """
    A read's time in this computer's own time zone: "Oct 8, 01:23 PDT".

    Args:
        read_at (datetime): The read (UTC).

    Returns:
        str: The text.
    """
    local = read_at.astimezone()
    return f"{local:%b} {local.day}, {local:%H:%M %Z}".rstrip()


def _read_text(read_at: datetime) -> str:
    """
    When the account was read: "Read from Kalshi Oct 8, 01:23 PDT".

    Args:
        read_at (datetime): The read (UTC).

    Returns:
        str: The text.
    """
    return f"Read from Kalshi {_read_time(read_at)}"


def _stale_text(read_at: datetime) -> str:
    """
    What a refresh that failed says beside its error: "Still showing the read from Oct 8, 01:23 PDT".

    Args:
        read_at (datetime): The read still shown (UTC).

    Returns:
        str: The text.
    """
    return f"Still showing the read from {_read_time(read_at)}"


# ---- what the page shows ------------------------------------------------------

@dataclass(frozen=True)
class _Bands:
    """
    How the groups are drawn: which categories have a band of their own, and every group's color.

    Attributes:
        named (tuple[str, ...]): Categories drawn by name, most money put in
            first (a lone category past the palette's slots last, in gray).
        folded (tuple[str, ...]): Categories drawn together as MORE_CATEGORIES
            (two or more, or none).
        colors (dict[str, str]): Each group's color (a category past the
            palette's slots has the MORE_CATEGORIES gray).
    """
    named: tuple[str, ...]
    folded: tuple[str, ...]
    colors: dict[str, str]


def _bands(view: live_portfolio.LiveView) -> _Bands:
    """
    Choose the named bands and every group's color, the same for every chart and period.

    The categories in view.group_order (most money put in first) take the
    palette's slots in order, at most config.LIVE_DASHBOARD_MAX_CATEGORY_BANDS
    of them; the rest are drawn together as MORE_CATEGORIES, in gray. When
    only one category is left past the slots, it is drawn by its own name in
    that gray instead, so no band hides a single category's name. A group
    the history holds that group_order lacks (live_portfolio's group order
    lists every one) comes after the listed ones: drawn by name while a slot
    is free, else with the rest.

    Args:
        view (live_portfolio.LiveView): The view.

    Returns:
        _Bands: The bands.
    """
    other = live_portfolio.OTHER_BETS
    categories = [g for g in view.group_order if g != other]
    if view.history is not None:
        categories += [g for g in view.history.value if g != other and g not in categories]
    most = min(config.LIVE_DASHBOARD_MAX_CATEGORY_BANDS, len(_PALETTE))
    named, folded = tuple(categories[:most]), tuple(categories[most:])
    if len(folded) == 1:
        named, folded = named + folded, ()
    colors = {g: _PALETTE[i] if i < most else _MORE_COLOR for i, g in enumerate(named)}
    colors.update(dict.fromkeys(folded, _MORE_COLOR))
    colors[other] = _OTHER_BETS_COLOR
    return _Bands(named, folded, colors)


@dataclass(frozen=True)
class _Band:
    """
    One band of the area chart: its values at each of the history's moments.

    Attributes:
        name (str): Cash, a category, MORE_CATEGORIES or Other bets.
        values (list[float]): Its value at each moment.
        color (str): Its color.
        members (tuple[str, ...]): For MORE_CATEGORIES, the categories it
            holds; empty otherwise.
    """
    name: str
    values: list[float]
    color: str
    members: tuple[str, ...] = ()


def _area_bands(history: live_portfolio.History, bands: _Bands) -> list[_Band]:
    """
    The area chart's bands, bottom first: Cash, each named category, MORE_CATEGORIES, Other bets.

    Each band is there only when the history holds it; each moment's bands
    add up to the account's whole value then.

    Args:
        history (live_portfolio.History): The history.
        bands (_Bands): The bands.

    Returns:
        list[_Band]: The bands, in stack order.
    """
    out = [_Band(CASH, list(history.cash), _CASH_COLOR)]
    for group in bands.named:
        if group in history.value:
            out.append(_Band(group, list(history.value[group]), bands.colors[group]))
    folded = [g for g in bands.folded if g in history.value]
    if folded:
        values = [sum(point) for point in zip(*(history.value[g] for g in folded), strict=True)]
        out.append(_Band(MORE_CATEGORIES, values, _MORE_COLOR, tuple(folded)))
    other = live_portfolio.OTHER_BETS
    if other in history.value:
        out.append(_Band(other, list(history.value[other]), _OTHER_BETS_COLOR))
    return out


def _members_text(members: Iterable[str]) -> str:
    """
    The categories MORE_CATEGORIES holds, as one line: "Mentions, Weather".

    Args:
        members (Iterable[str]): The categories.

    Returns:
        str: Their names, comma separated.
    """
    return ", ".join(members)


def _area(view: live_portfolio.LiveView, bands: _Bands) -> dict | None:
    """
    The stacked area chart of the account's value over time, by band.

    Cash first (at the bottom), then each named category, MORE_CATEGORIES
    and Other bets last (_area_bands). A 2-pixel line in the surface color
    separates the bands. Hovering shows every band at that moment and the
    account's total (a line drawn in no width, kept out of the legend), and
    MORE_CATEGORIES names the categories it holds; a slider below the chart
    chooses the span shown, as the period buttons above it do. The legend
    sits to the right, its entries in the stack's order (the top band
    first).

    Args:
        view (live_portfolio.LiveView): The view.
        bands (_Bands): The bands.

    Returns:
        dict | None: {"data", "layout"} for Plotly; None without a history.
    """
    history = view.history
    if history is None:
        return None
    x = [_chart_time(moment) for moment in history.times]
    data = []
    for band in _area_bands(history, bands):
        trace = {"type": "scatter", "mode": "lines", "name": band.name, "x": x,
                 "y": band.values, "stackgroup": "one", "line": {"width": 2, "color": _SURFACE},
                 "fillcolor": band.color, "hovertemplate": "%{y:$,.2f}"}
        if band.members:
            trace["customdata"] = [_members_text(band.members)] * len(x)
            trace["hovertemplate"] = "%{y:$,.2f} (%{customdata})"
        data.append(trace)
    data.append({"type": "scatter", "mode": "lines", "name": "Total", "x": x,
                 "y": [history.total(i) for i in range(len(history.times))],
                 "line": {"width": 0, "color": _INK}, "showlegend": False,
                 "hovertemplate": "%{y:$,.2f}"})
    layout = {
        "height": 440, "margin": {"l": 70, "r": 10, "t": 20, "b": 30},
        "paper_bgcolor": _SURFACE, "plot_bgcolor": _SURFACE,
        "font": {"family": _FONT, "color": _INK, "size": 12},
        "hovermode": "x unified",
        "legend": {"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top",
                   "traceorder": "reversed"},
        "xaxis": {"type": "date", "gridcolor": _GRID, "linecolor": _AXIS,
                  "hoverformat": "%b %-d, %Y %H:%M",
                  "rangeslider": {"visible": True, "bgcolor": _PLANE}},
        "yaxis": {"tickprefix": "$", "tickformat": ",.2~f", "gridcolor": _GRID,
                  "linecolor": _AXIS, "rangemode": "tozero"},
    }
    return {"data": data, "layout": layout}


def _area_table(view: live_portfolio.LiveView, bands: _Bands) -> dict | None:
    """
    The area chart's figures as a table: one row per point, each band's value and the total.

    Args:
        view (live_portfolio.LiveView): The view.
        bands (_Bands): The bands.

    Returns:
        dict | None: "head" (the column names: the time in New York, each
            band in stack order, then Total) and "rows" (each {"cells",
            "tip", "swatch"}, oldest first); None without a history.
    """
    history = view.history
    if history is None:
        return None
    area = _area_bands(history, bands)
    rows = []
    for i, moment in enumerate(history.times):
        local = moment.astimezone(_DAY_ZONE)
        when = f"{local:%b} {local.day}, {local.year} {local:%H:%M}"
        rows.append({"cells": [when, *(_money(band.values[i]) for band in area),
                               _money(history.total(i))], "tip": None, "swatch": None})
    return {"head": ["Time (New York)", *(band.name for band in area), "Total"], "rows": rows}


@dataclass(frozen=True)
class _GroupRow:
    """
    One band's return over a period, folded categories already summed.

    Attributes:
        label (str): The category, MORE_CATEGORIES or Other bets.
        pnl (float): Its profit.
        put_in (float): What it put in.
        ret (float | None): pnl over put_in; None when nothing was put in.
        color (str): Its band's color.
        members (tuple[str, ...]): For MORE_CATEGORIES, the categories it
            holds in the period; empty otherwise.
    """
    label: str
    pnl: float
    put_in: float
    ret: float | None
    color: str
    members: tuple[str, ...] = ()


def _moved(pnl: float, put_in: float) -> bool:
    """
    Whether a band did anything in a period: a profit or money put in that shows as more than $0.00.

    Args:
        pnl (float): Its profit.
        put_in (float): What it put in.

    Returns:
        bool: False when both round to zero cents.
    """
    return round(pnl, 2) != 0 or round(put_in, 2) != 0


def _group_rows(groups: Iterable[live_portfolio.GroupReturn], bands: _Bands) -> list[_GroupRow]:
    """
    A period's returns by band: each named category, MORE_CATEGORIES, then Other bets.

    The folded categories' profits and what they put in are each added up,
    and their return is the one over the other (never an average of returns).

    Args:
        groups (Iterable[live_portfolio.GroupReturn]): The period's returns by group.
        bands (_Bands): The bands.

    Returns:
        list[_GroupRow]: The rows, in band order; a band the period lacks,
            or one with no profit and nothing put in (both $0.00), is left out.
    """
    by_group = {r.group: r for r in groups}
    rows = [_GroupRow(g, by_group[g].pnl, by_group[g].put_in, by_group[g].ret, bands.colors[g])
            for g in bands.named if g in by_group]
    other = live_portfolio.OTHER_BETS
    folded = [(g, r) for g, r in by_group.items() if g != other and g not in bands.named]
    if folded:
        pnl = sum(r.pnl for _, r in folded)
        put_in = sum(r.put_in for _, r in folded)
        rows.append(_GroupRow(MORE_CATEGORIES, pnl, put_in, pnl / put_in if put_in > 0 else None,
                              _MORE_COLOR, tuple(g for g, _ in folded)))
    if other in by_group:
        r = by_group[other]
        rows.append(_GroupRow(other, r.pnl, r.put_in, r.ret, _OTHER_BETS_COLOR))
    return [row for row in rows if _moved(row.pnl, row.put_in)]


def _bars(rows: list[_GroupRow]) -> dict:
    """
    The bar chart of a period's returns by band, each bar labelled "+12.3% · +$4.56".

    Args:
        rows (list[_GroupRow]): The rows, from _group_rows.

    The x axis reaches half as far again past the longest bar on each side
    that has bars, so each bar's label (drawn past its end) clears the
    category names and the chart's edge. Each bar's end away from zero is
    rounded; hovering on MORE_CATEGORIES names the categories it holds.

    Returns:
        dict: {"data", "layout"} for Plotly; the bars run top to bottom in
            the rows' order, each in its band's color.
    """
    returns = [_finite(r.ret) for r in rows]
    xs = [0.0 if ret is None else ret * 100 for ret in returns]
    low, high = min([0.0, *xs]), max([0.0, *xs])
    span = max(high - low, 10.0)
    reach = [low - (0.5 if low < 0 else 0.05) * span, high + (0.5 if high > 0 else 0.05) * span]
    data = [{"type": "bar", "orientation": "h",
             "y": [r.label for r in rows],
             "x": xs,
             "marker": {"color": [r.color for r in rows], "cornerradius": 4},
             "text": [f"{_pct(r.ret)} · {_signed_money(r.pnl)}" for r in rows],
             "customdata": [f" ({_members_text(r.members)})" if r.members else ""
                            for r in rows],
             "textposition": "outside", "cliponaxis": False,
             "hovertemplate": "%{y}: %{text}%{customdata}<extra></extra>"}]
    layout = {
        "height": max(140, 50 + 34 * len(rows)),
        "margin": {"l": 10, "r": 130, "t": 10, "b": 30},
        "paper_bgcolor": _SURFACE, "plot_bgcolor": _SURFACE,
        "font": {"family": _FONT, "color": _INK, "size": 12},
        "showlegend": False, "bargap": 0.35,
        "xaxis": {"ticksuffix": "%", "gridcolor": _GRID, "zeroline": True,
                  "zerolinecolor": _AXIS, "range": reach},
        "yaxis": {"autorange": "reversed", "automargin": True},
    }
    return {"data": data, "layout": layout}


def _cards(stats: live_portfolio.PeriodStats) -> dict[str, dict[str, str]]:
    """
    The six cards' text, color and tooltip for one period.

    Args:
        stats (live_portfolio.PeriodStats): The period's figures.

    Returns:
        dict[str, dict[str, str]]: Each card's {"text", "tone", "title"},
            keyed as in CARD_LABELS, in that order.
    """
    days = _count_words(stats.whole_days, "whole day", "whole days")
    least = config.LIVE_RATIO_MIN_WHOLE_DAYS
    if stats.whole_days >= least:
        ratio_note = f"over {days}"
    else:
        ratio_note = (f"needs at least {_count_words(least, 'whole day', 'whole days')} (one "
                      f"daily close to the next); this period has {days}")
    span, length = _span(stats.first, stats.last)
    term = config.RISK_FREE_BILL_TERM.lower()
    pairs = _count_words(stats.pairs, "contract pair", "contract pairs")
    trade_note = ("each pair counted once: cash back plus value now, less what it cost, over "
                  "what it cost (fees, sales and payouts included; open pairs at today's value)")
    texts = {
        "return": (_pct(stats.total_return),
                   "Time-weighted: each step's return chained, with deposits and withdrawals "
                   f"taken out, over {span}, {length}"),
        "pnl": (_signed_money(stats.pnl),
                "The account's value at the end less its value at the start, less deposits "
                "and plus withdrawals"),
        "sharpe": (_ratio(stats.sharpe),
                   "Mean daily return over its standard deviation, times √365, after taking "
                   f"out the {term} T-bill yield on the money held in positions; {ratio_note}"),
        "sortino": (_ratio(stats.sortino),
                    "As Sharpe, but over the deviation of the losing days only; "
                    f"{ratio_note}"),
        "mean": (_pct(stats.mean_trade),
                 f"The mean return of the {pairs} the bot bought in this period, {trade_note}"
                 if stats.purchases else "The bot bought no contract pair in this period"),
        "median": (_pct(stats.median_trade),
                   f"The median return of the {pairs} the bot bought in this period, "
                   f"{trade_note}" if stats.purchases
                   else "The bot bought no contract pair in this period"),
    }
    return {key: {"text": text, "tone": _tone(text), "title": title}
            for key, (text, title) in ((k, texts[k]) for k, _ in CARD_LABELS)}


def _trades_note(stats: live_portfolio.PeriodStats) -> str:
    """
    The line under the cards: "312 contract pairs in 41 purchases (247 still open) · Sharpe and Sortino use 9 whole days".

    Args:
        stats (live_portfolio.PeriodStats): The period's figures.

    Returns:
        str: The line ("No bot purchase in this period · ..." when there was none).
    """
    days = _count_words(stats.whole_days, "whole day", "whole days")
    if not stats.purchases:
        return f"No bot purchase in this period · Sharpe and Sortino use {days}"
    return (f"{_count_words(stats.pairs, 'contract pair', 'contract pairs')} in "
            f"{_count_words(stats.purchases, 'purchase', 'purchases')} "
            f"({config.count_text(float(stats.open_pairs))} still open) · Sharpe and Sortino use "
            f"{days}")


def _period(stats: live_portfolio.PeriodStats, bands: _Bands) -> dict:
    """
    Everything one period button shows.

    Args:
        stats (live_portfolio.PeriodStats): The period's figures.
        bands (_Bands): The bands.

    Returns:
        dict: "label", "note" (its span), "cards", "trades" (the line under
            the cards), "range" (its first and last moment as the area
            chart's x values), "groups" (the bar chart) and "group_rows" (the
            table under it: band, return, profit, put in, each row with its
            band's color as a swatch on the first cell, and MORE_CATEGORIES
            naming its categories as that cell's tooltip).
    """
    rows = _group_rows(stats.groups, bands)
    return {
        "label": stats.label,
        "note": _range_note(stats.first, stats.last),
        "cards": _cards(stats),
        "trades": _trades_note(stats),
        "range": [_chart_time(stats.first), _chart_time(stats.last)],
        "groups": _bars(rows),
        "group_rows": [{"cells": [r.label, _pct(r.ret), _signed_money(r.pnl), _money(r.put_in)],
                        "tip": _members_text(r.members) or None, "swatch": [0, r.color]}
                       for r in rows],
    }


def _market_label(holding: live_portfolio.Holding) -> str:
    """
    The holdings table's Market cell: the title, then " — " and the outcome label when it says more.

    The outcome label (Kalshi's subtitle, such as one rung's deadline) is
    what tells two markets of one question apart; it is left off when it is
    empty or the title itself, as the scanner's display_title does.

    Args:
        holding (live_portfolio.Holding): The holding.

    Returns:
        str: The label.
    """
    if holding.subtitle and holding.subtitle != holding.title:
        return f"{holding.title} — {holding.subtitle}"
    return holding.title


def _holdings(view: live_portfolio.LiveView, bands: _Bands) -> dict:
    """
    The holdings table: one row per holding, then the holdings' totals, the cash and the account's total.

    The holdings' row sets their value beside their cost; the Cash row and
    the Total row (holdings and cash) give a value only, since cash has no
    cost to compare it with.

    Args:
        view (live_portfolio.LiveView): The view.
        bands (_Bands): The bands (each row's category swatch).

    Returns:
        dict: "rows" and "foot" (each row {"cells", "tip", "swatch"}: the
            cells Market, Side, Contracts, Price, Value, Cost and Category,
            the market's ticker as the first cell's tooltip, and the
            category's color on the last cell), and "empty" (a sentence for
            when nothing is held, else None).
    """
    rows = [{"cells": [_market_label(h), h.side.upper(), config.count_text(float(h.contracts)),
                       _price(h.price), _money(h.value), _money(h.cost), h.group],
             "tip": h.ticker, "swatch": [6, bands.colors.get(h.group, _MORE_COLOR)]}
            for h in view.holdings]
    cost = sum((h.cost for h in view.holdings), Decimal(0))
    foot = [
        {"cells": ["Holdings", "", "", "", _money(view.holdings_value), _money(cost), ""],
         "tip": None, "swatch": None},
        {"cells": [CASH, "", "", "", _money(view.cash), "", ""], "tip": None,
         "swatch": [6, _CASH_COLOR]},
        {"cells": ["Total", "", "", "", _money(view.cash + view.holdings_value), "", ""],
         "tip": None, "swatch": None},
    ]
    return {"rows": rows, "foot": foot,
            "empty": None if view.holdings else "Nothing is held now."}


def _risk_free_note(rates: treasury.RiskFreeRates | None) -> str:
    """
    What the Sharpe and Sortino ratios subtract: the T-bill yield, or 0% and why.

    Args:
        rates (treasury.RiskFreeRates | None): The yields the view used.

    Returns:
        str: One sentence.
    """
    term = config.RISK_FREE_BILL_TERM.lower()
    if rates is None:
        return (f"Sharpe and Sortino subtract 0%: the {term} Treasury bill's yields have not "
                "been downloaded yet.")
    if rates.latest is None:
        return (f"Sharpe and Sortino subtract 0%: the {term} Treasury bill's yields could not "
                "be downloaded, and no earlier download is saved.")
    day, rate = rates.latest
    text = (f"Sharpe and Sortino subtract the {term} Treasury bill's yield (latest auction "
            f"{rate:.2%}, {day:%b} {day.day}, {day.year}) on the share of the account held in "
            "positions.")
    if rates.source == treasury.SOURCE_CACHE:
        text += " The download failed, so an earlier saved download is used."
    return text


def _notes(view: live_portfolio.LiveView) -> list[str]:
    """
    The notes under the holdings: how positions are valued, the cash check, Kalshi's own value, the ratios' yield.

    Args:
        view (live_portfolio.LiveView): The view.

    Returns:
        list[str]: The sentences, in page order.
    """
    notes = ["Positions are valued at the midpoint of the best bid and ask; an empty side "
             "counts at 0 or 1; with both sides empty, the last trade. A decided market is "
             "valued at its payout, and one never priced at what it cost."]
    check = view.cash_check
    if check.checked:
        worst = check.worst.quantize(Decimal(config.LIVE_CASH_SHOWN_STEP_DOLLARS))
        notes.append(f"Cash rebuilt from Kalshi's records matches {check.matched} of "
                     f"{check.checked} runs' logged balances (largest gap ${worst}).")
    else:
        notes.append("No run's logged balance to check the rebuilt cash against yet.")
    held = _money(view.holdings_value)
    if view.kalshi_positions_value is None:
        notes.append(f"Holdings at the midpoint: {held}. Kalshi's own value of the positions "
                     "could not be read.")
    else:
        notes.append(f"Holdings at the midpoint: {held}. Kalshi's own value of the positions: "
                     f"{_money(view.kalshi_positions_value)}.")
    notes.append(_risk_free_note(view.risk_free))
    if view.history is not None:
        notes.append("The chart's times are New York time. It has a point at each daily close "
                     "(midnight there, where Kalshi's days close) and now, joined by straight "
                     "lines, so a trade made during a day shows as a slope up to that day's "
                     "close.")
    return notes


def payload(view: live_portfolio.LiveView) -> dict:
    """
    Everything the Live trading tab shows, as JSON-ready text and Plotly figures.

    Every number is formatted here (every Decimal and float), so the page's
    script only places what it is given. Before the bot's first live trade
    there is no history: no chart, no period, and "empty" says why.

    Args:
        view (live_portfolio.LiveView): The view, from build_live_view.

    Returns:
        dict: "status" (when the account was read), "stale" (what a later
            refresh that fails says beside its error), "read_at" (ISO time),
            "empty" (a sentence, or None), "periods" (one per
            config.LIVE_DASHBOARD_PERIODS, see _period; [] without a
            history), "area" (the stacked chart, or None), "area_table" (its
            figures, see _area_table, or None), "holdings" (see _holdings),
            "notes" and "warnings" (sentences).
    """
    bands = _bands(view)
    periods = [_period(stats, bands) for stats in view.periods or ()]
    empty = None
    if view.history is None or not periods:
        empty = ("The bot has made no live trade yet. The holdings below are read from Kalshi; "
                 "the chart and the statistics start at the bot's first trade.")
    return {
        "status": _read_text(view.read_at),
        "stale": _stale_text(view.read_at),
        "read_at": view.read_at.isoformat(),
        "empty": empty,
        "periods": periods,
        "area": _area(view, bands),
        "area_table": _area_table(view, bands),
        "holdings": _holdings(view, bands),
        "notes": _notes(view),
        "warnings": list(view.warnings),
    }


# ---- one read at a time ---------------------------------------------------------

class _Busy(Exception):
    """A read of the account was already going and did not finish in time."""


@dataclass
class _Flight:
    """
    One read of the account and its outcome.

    Attributes:
        done (threading.Event): Set when the read finished.
        result (Any): What it returned.
        error (BaseException | None): What it raised, or None.
    """
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class _SingleFlight:
    """
    One read of the account at a time: a request arriving during a read waits for that read's result.

    The first request starts a read; every request that arrives before it
    finishes gets the same result, or the same error, waiting at most
    config.LIVE_DASHBOARD_BUILD_WAIT_SECONDS (read when it waits). A request
    after the read finished starts a new one, so every page load still reads
    the account afresh.
    """

    def __init__(self, build: Callable[[], Any]) -> None:
        """
        Set up with no read going.

        Args:
            build (Callable[[], Any]): The read: called with no arguments.
        """
        self._build = build
        self._lock = threading.Lock()
        self._current: _Flight | None = None
        self.waiting = 0

    def get(self) -> Any:
        """
        Read the account, or wait for the read already going.

        Returns:
            Any: What the read returned.

        Raises:
            _Busy: When this request waited the whole wait for another's read.
            Exception: Whatever the read raised.
        """
        with self._lock:
            flight = self._current
            leader = flight is None
            if leader:
                flight = self._current = _Flight()
            else:
                self.waiting += 1
        if leader:
            try:
                flight.result = self._build()
            except BaseException as exc:
                flight.error = exc
                raise
            finally:
                with self._lock:
                    self._current = None
                flight.done.set()
            return flight.result
        try:
            finished = flight.done.wait(config.LIVE_DASHBOARD_BUILD_WAIT_SECONDS)
        finally:
            with self._lock:
                self.waiting -= 1
        if not finished:
            raise _Busy
        if flight.error is not None:
            raise flight.error
        return flight.result


def _duration_words(seconds: float) -> str:
    """
    A wait in words: whole minutes from two minutes up ("5 minutes"), else seconds ("30 seconds").

    Args:
        seconds (float): The wait.

    Returns:
        str: The words.
    """
    if seconds >= 120:
        return _count_words(round(seconds / 60), "minute", "minutes")
    return _count_words(seconds, "second", "seconds")


# ---- the two applications -----------------------------------------------------

@dataclass(frozen=True)
class _Request:
    """
    One HTTP request, as the applications see it.

    Attributes:
        method (str): The method, e.g. "GET".
        target (str): The request target, e.g. "/api/live".
        host (str | None): The Host header; None when missing or given twice.
        headers (dict[str, str]): Every header given exactly once, by its
            lower-case name.
        repeated (frozenset[str]): The lower-case names of the headers given
            more than once (in any mix of cases).
    """
    method: str
    target: str
    host: str | None
    headers: dict[str, str] = field(default_factory=dict)
    repeated: frozenset[str] = frozenset()

    def header(self, name: str) -> str | None:
        """
        A header given exactly once.

        Args:
            name (str): Its name, in any case.

        Returns:
            str | None: Its value; None when missing or given more than once.
        """
        return self.headers.get(name.lower())


# What _fetch_site answers for a Sec-Fetch-Site header given more than once:
# not a value a browser sends, so it is refused wherever the header is checked
_SITE_GIVEN_TWICE = "given more than once"

# The Sec-Fetch-Site values of a request that does not come from another web
# site: none (typed, a bookmark, the browser opened by the server), and this
# server or another port of this computer
_NOT_ANOTHER_SITE = (None, "none", "same-origin", "same-site")


def _fetch_site(request: _Request) -> str | None:
    """
    Where a browser says a request came from (its Sec-Fetch-Site header).

    Args:
        request (_Request): The request.

    Returns:
        str | None: The header's value; None when it was not sent (a program
            other than a browser, or an older browser); _SITE_GIVEN_TWICE when
            it was sent more than once.
    """
    if "sec-fetch-site" in request.repeated:
        return _SITE_GIVEN_TWICE
    return request.header("Sec-Fetch-Site")


@dataclass(frozen=True)
class _Response:
    """
    One HTTP response.

    Attributes:
        status (int): The HTTP status.
        body (str | bytes | Path): The body; a Path is a file, sent from disk.
        content_type (str): The Content-Type header.
        headers (tuple[tuple[str, str], ...]): Every other header (Content-
            Length aside), in sending order.
        file_id (tuple[int, int] | None): For a file, the (device, inode)
            of the file that was checked; the file opened to send must be
            that same one. None for no check.
    """
    status: int
    body: str | bytes | Path
    content_type: str
    headers: tuple[tuple[str, str], ...] = ()
    file_id: tuple[int, int] | None = None


_HTML_TYPE = "text/html; charset=utf-8"
_TEXT_TYPE = "text/plain; charset=utf-8"
_JSON_TYPE = "application/json; charset=utf-8"
_JS_TYPE = "text/javascript; charset=utf-8"

# The tab page's address and the backtest page's address
_PAGE_ORIGIN = f"http://{config.LIVE_DASHBOARD_HOST}:{config.LIVE_DASHBOARD_PORT}"
_BACKTEST_ORIGIN = f"http://{config.LIVE_DASHBOARD_HOST}:{config.LIVE_BACKTEST_PORT}"
BACKTEST_URL = f"{_BACKTEST_ORIGIN}/backtest/"

# Every answer on the tab page's port that is not the page itself: never
# stored, never sniffed, never framed, never loaded by another origin's page,
# and allowed to load nothing
_PLAIN_HEADERS = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
)

# Every answer on the backtest page's port: checked again on each load (the
# page is rebuilt in place), shown in a frame only by the tab page, and loaded
# as a script or image by no page of another site
_FRAMED_HEADERS = (
    ("Cache-Control", "no-cache"),
    ("X-Content-Type-Options", "nosniff"),
    ("Content-Security-Policy",
     f"frame-ancestors {_PAGE_ORIGIN} http://localhost:{config.LIVE_DASHBOARD_PORT}"),
    ("Referrer-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-site"),
)


def _text(status: int, message: str, headers: tuple[tuple[str, str], ...]) -> _Response:
    """
    A short plain-text answer.

    Args:
        status (int): The HTTP status.
        message (str): The text.
        headers (tuple[tuple[str, str], ...]): The headers to send with it.

    Returns:
        _Response: The answer; a 405 also says which methods are allowed.
    """
    extra = (("Allow", "GET, HEAD"),) if status == HTTPStatus.METHOD_NOT_ALLOWED else ()
    return _Response(status, message + "\n", _TEXT_TYPE, headers + extra)


def _json(status: int, record: Any) -> _Response:
    """
    A JSON answer on the tab page's port.

    Args:
        status (int): The HTTP status.
        record (Any): The record, or its JSON text already.

    Returns:
        _Response: The answer.
    """
    body = record if isinstance(record, str) else json.dumps(record, allow_nan=False)
    return _Response(status, body, _JSON_TYPE, _PLAIN_HEADERS)


class PageApp:
    """
    127.0.0.1:8766: the tab page, the account data (/api/live) and /health.

    One instance lives as long as its server. Its read of the account runs
    through a _SingleFlight, so one read runs at a time.
    """

    def __init__(self, port: int | None = None, *, build: Callable[[], Any] | None = None) -> None:
        """
        Set up the routes for the tab page's server.

        Args:
            port (int | None): The port it listens on, which a request's Host
                must name; None (default) is config.LIVE_DASHBOARD_PORT.
            build (Callable[[], Any] | None): Keyword-only. The read of the
                account: returns the JSON text /api/live answers. None
                (default) refuses every read (main passes the real one).
        """
        self.port = config.LIVE_DASHBOARD_PORT if port is None else port
        self._hosts = frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})
        self._flight = _SingleFlight(build if build is not None else _no_reader)

    def handle(self, request: _Request) -> _Response:
        """
        Answer one request.

        It must name this server in its Host header (403 otherwise), be a
        GET or HEAD (405) and ask for a path (400 for any other form of
        address). "/" is the tab page, except when another web site sent the
        browser there (403, a short note with no script, so no other site
        can make it read the account by opening the page); "/health" says
        what this server is, which checkout it serves and the fingerprint of
        the code it loaded; "/api/live" reads the account, for the page's own
        script only (its header, and a same-origin request or one with no
        Sec-Fetch-Site header, sent at most once), never on a HEAD. Anything
        else is 404.

        Args:
            request (_Request): The request.

        Returns:
            _Response: The answer.
        """
        if request.host is None or request.host.lower() not in self._hosts:
            return _text(403, f"This server answers only 127.0.0.1:{self.port} or "
                         f"localhost:{self.port}.", _PLAIN_HEADERS)
        if request.method not in ("GET", "HEAD"):
            return _text(405, "This server only shows pages.", _PLAIN_HEADERS)
        if not request.target.startswith("/"):
            return _text(400, "Ask for a path on this server, such as /.", _PLAIN_HEADERS)
        path = urllib.parse.urlsplit(request.target).path
        site = _fetch_site(request)
        if path == "/":
            if site not in _NOT_ANOTHER_SITE:
                return _text(403, "This dashboard opens only from this computer: type "
                             f"http://{config.LIVE_DASHBOARD_HOST}:{self.port}/ in the address "
                             "bar, or run ./start_dashboard.sh.", _PLAIN_HEADERS)
            return _Response(200, _SHELL, _HTML_TYPE, _PAGE_HEADERS)
        if path == "/health":
            return _json(200, {"app": _APP, "project_root": str(config.PROJECT_ROOT.resolve()),
                               "code": _LOADED_CODE})
        if path == "/api/live":
            if request.method == "HEAD":
                return _text(405, "Use GET.", _PLAIN_HEADERS)
            if request.header(config.LIVE_DASHBOARD_REQUEST_HEADER) != "1" or site not in (
                    None, "same-origin"):
                return _json(403, {"error": "Only this dashboard's own page may read the account."})
            return self._live()
        return self.not_found()

    def _live(self) -> _Response:
        """
        Read the account (or wait for the read already going) and answer its data.

        Returns:
            _Response: 200 with the data; 503 when another read did not
                finish in time; 502 with one line saying why when the read
                failed.
        """
        try:
            body = self._flight.get()
        except _Busy:
            waited = _duration_words(config.LIVE_DASHBOARD_BUILD_WAIT_SECONDS)
            return _json(503, {"error": f"Another read of the account is still going after "
                                        f"{waited}: Kalshi may be slow to answer. Try Refresh "
                                        "again in a few minutes."})
        except Exception as exc:
            reason = api_error_summary(exc)
            logging.warning("Could not read the account: %s", reason)
            logging.debug("The read's traceback", exc_info=True)
            return _json(502, {"error": f"Could not read the account from Kalshi: {reason}"})
        return _json(200, body)

    def not_found(self) -> _Response:
        """
        The 404 answer.

        Returns:
            _Response: The answer.
        """
        return _text(404, "No such page.", _PLAIN_HEADERS)

    def failed(self, request: _Request | None = None) -> _Response:
        """
        The answer when answering a request raised.

        Args:
            request (_Request | None): The request, when known: on
                /api/live the answer is JSON, as the page's script reads it.

        Returns:
            _Response: A 500.
        """
        message = ("Something went wrong answering this request; the server's log has the "
                   "details.")
        if request is not None and urllib.parse.urlsplit(request.target).path == "/api/live":
            return _json(500, {"error": message})
        return _text(500, message, _PLAIN_HEADERS)

    @property
    def error_headers(self) -> tuple[tuple[str, str], ...]:
        """The headers a refusal made before the application sees a request carries."""
        return _PLAIN_HEADERS


def _no_reader() -> Any:
    """
    The read a PageApp built without one has: it refuses.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("this server was started without a reader of the account")


class BacktestApp:
    """
    127.0.0.1:8767: the backtest page and its chunk files, for the Backtest tab only.

    It serves config.DASHBOARD_FILENAME from config.PROJECT_ROOT (read when
    called) at /backtest/, and the .js files under
    config.DASHBOARD_FILES_DIRNAME beside it, which the page loads by
    relative address. Every answer may be framed only by the tab page.
    """

    def __init__(self, port: int | None = None) -> None:
        """
        Set up the routes for the backtest page's server.

        Args:
            port (int | None): The port it listens on, which a request's Host
                must name; None (default) is config.LIVE_BACKTEST_PORT.
        """
        self.port = config.LIVE_BACKTEST_PORT if port is None else port
        self._hosts = frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})

    def handle(self, request: _Request) -> _Response:
        """
        Answer one request.

        It must name this server in its Host header (403), be a GET or HEAD
        (405) and ask for a path (400). A page of another web site may not
        load anything from it except by navigating to it (403), so no other
        site can read a chunk file by loading it as a script, or make the
        browser download the page in the background; the tab page framing
        it is a navigation, and the page's own chunk loads come from its own
        origin. "/backtest" is sent on to "/backtest/" (308), since the
        page's chunk addresses are relative to it; "/backtest/" is the page,
        or a short page saying there is none yet; "/backtest/<the files
        folder>/..." is a chunk file (_sidecar). Anything else is 404. A
        file answers If-None-Match (its ETag) and If-Modified-Since with 304
        when it has not changed.

        Args:
            request (_Request): The request.

        Returns:
            _Response: The answer.
        """
        if request.host is None or request.host.lower() not in self._hosts:
            return _text(403, f"This server answers only 127.0.0.1:{self.port} or "
                         f"localhost:{self.port}.", _FRAMED_HEADERS)
        if request.method not in ("GET", "HEAD"):
            return _text(405, "This server only shows pages.", _FRAMED_HEADERS)
        if not request.target.startswith("/"):
            return _text(400, "Ask for a path on this server, such as /backtest/.",
                         _FRAMED_HEADERS)
        site = _fetch_site(request)
        if site == _SITE_GIVEN_TWICE or (site not in _NOT_ANOTHER_SITE
                                         and request.header("Sec-Fetch-Mode") != "navigate"):
            return _text(403, "Only the dashboard's own page may load this.", _FRAMED_HEADERS)
        path = urllib.parse.urlsplit(request.target).path
        if path == "/backtest":
            return _Response(308, "See /backtest/\n", _TEXT_TYPE,
                             _FRAMED_HEADERS + (("Location", "/backtest/"),))
        if path == "/backtest/":
            page = config.PROJECT_ROOT / config.DASHBOARD_FILENAME
            if not page.is_file():
                return _Response(200, _NO_BACKTEST_HTML, _HTML_TYPE, _FRAMED_HEADERS)
            return _file(page, _HTML_TYPE, request)
        if path.startswith(f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/"):
            target = _sidecar(path)
            if target is None:
                return self.not_found()
            return _file(target, _JS_TYPE, request)
        return self.not_found()

    def not_found(self) -> _Response:
        """
        The 404 answer.

        Returns:
            _Response: The answer.
        """
        return _text(404, "No such file.", _FRAMED_HEADERS)

    def failed(self, request: _Request | None = None) -> _Response:
        """
        The answer when answering a request raised.

        Args:
            request (_Request | None): The request, when known (unused: every
                answer here is text).

        Returns:
            _Response: A 500.
        """
        return _text(500, "Something went wrong answering this request; the server's log "
                     "has the details.", _FRAMED_HEADERS)

    @property
    def error_headers(self) -> tuple[tuple[str, str], ...]:
        """The headers a refusal made before the application sees a request carries."""
        return _FRAMED_HEADERS


def _sidecar(path: str) -> Path | None:
    """
    The chunk file a /backtest/ address names: a .js file under the files folder, never anything outside it.

    The address after "/backtest/" is decoded once; a control character
    (NUL included) or a backslash in it refuses it. It must then resolve,
    links followed, to an existing .js file inside the resolved files
    folder, so "..", an encoded "..", an encoded "/" and a link pointing
    outside the folder all reach nothing.

    Args:
        path (str): The request's path, still percent-encoded.

    Returns:
        Path | None: The file; None when the address names no such file.
    """
    relative = urllib.parse.unquote(path[len("/backtest/"):])
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in relative) or "\\" in relative:
        return None
    root = (config.PROJECT_ROOT / config.DASHBOARD_FILES_DIRNAME).resolve()
    try:
        target = (config.PROJECT_ROOT / relative).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not target.is_relative_to(root) or target.suffix != ".js" or not target.is_file():
        return None
    return target


def _file(path: Path, content_type: str, request: _Request) -> _Response:
    """
    A file of the backtest page, with its Last-Modified time and ETag, or 304 when the browser's copy is current.

    The ETag names the file's inode, size and modification time to the
    nanosecond, so a page published again within one second (by rename, a
    new inode) is never mistaken for the browser's copy. The answer carries
    the file's (device, inode), which the handler checks against the file it
    opens to send.

    Args:
        path (Path): The file.
        content_type (str): Its Content-Type.
        request (_Request): The request (its If-None-Match and If-Modified-Since).

    Returns:
        _Response: The file (sent from disk by the handler), or a 304 with no body.
    """
    try:
        stat = path.stat()
    except OSError:
        return _text(404, "No such file.", _FRAMED_HEADERS)
    modified = int(stat.st_mtime)
    tag = f'"{stat.st_ino:x}-{stat.st_size:x}-{stat.st_mtime_ns:x}"'
    headers = _FRAMED_HEADERS + (("Last-Modified", formatdate(modified, usegmt=True)),
                                 ("ETag", tag))
    if _unchanged(request, tag, modified):
        return _Response(304, b"", content_type, headers)
    return _Response(200, path, content_type, headers, (stat.st_dev, stat.st_ino))


def _unchanged(request: _Request, tag: str, modified: int) -> bool:
    """
    Whether the browser's copy of a file is current: by If-None-Match when it is sent, else by If-Modified-Since.

    Args:
        request (_Request): The request.
        tag (str): The file's ETag.
        modified (int): Its modification time, in whole seconds.

    Returns:
        bool: True when the browser's copy is the file as it is now.
    """
    match = request.header("If-None-Match")
    if match is not None:
        tags = [part.strip().removeprefix("W/") for part in match.split(",")]
        return "*" in tags or tag in tags
    since = request.header("If-Modified-Since")
    if since is None:
        return False
    try:
        return modified <= parsedate_to_datetime(since).timestamp()
    except (TypeError, ValueError, OverflowError, IndexError):
        return False


_NO_BACKTEST_HTML = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>No backtest yet</title>'
    f'</head><body style="font-family:{escape(_FONT)};color:{_INK};background:{_PLANE};'
    'padding:24px"><p>There is no backtest dashboard yet. Build one with <code>python3 -m '
    "kalshi_betting.backtest</code>, then choose the Backtest tab again.</p></body></html>")


# ---- the tab page ---------------------------------------------------------------

# The page's own sentences, which its script shows (it words nothing itself)
_SCRIPT_TEXT = {
    "loading": "Reading the account from Kalshi…",
    "unreachable": "Could not reach this dashboard's server: is it still running?",
    "unreadable": "The server's answer could not be read.",
    "no_charts": ("The chart library could not be loaded from cdn.plot.ly, so the charts are "
                  "missing; the figures and tables below are complete."),
}

_SCRIPT_TEMPLATE = """
(function () {
  'use strict';
  var BACKTEST_URL = __BACKTEST_URL__;
  var TEXT = __TEXT__;
  var HEADERS = __HEADERS__;
  var PERIODS = __PERIODS__;
  var CARDS = __CARDS__;
  var CHART_CONFIG = __CHART_CONFIG__;
  var DATA = null;
  var CHOSEN = 0;
  var OPENED = false;
  var LOADING = false;

  function byId(id) { return document.getElementById(id); }
  function charts() { return typeof Plotly === 'undefined' ? null : Plotly; }

  function fill(id, items) {
    var list = byId(id);
    list.textContent = '';
    items.forEach(function (text) {
      var item = document.createElement('li');
      item.textContent = text;
      list.appendChild(item);
    });
    list.hidden = !items.length;
  }

  function rows(id, list) {
    var body = byId(id);
    body.textContent = '';
    list.forEach(function (row) {
      var line = document.createElement('tr');
      row.cells.forEach(function (text, j) {
        var cell = document.createElement('td');
        cell.textContent = text;
        if (j === 0 && row.tip) { cell.title = row.tip; }
        if (row.swatch && j === row.swatch[0]) {
          cell.style.borderLeft = '4px solid ' + row.swatch[1];
        }
        line.appendChild(cell);
      });
      body.appendChild(line);
    });
  }

  function head(id, names) {
    var line = byId(id);
    line.textContent = '';
    names.forEach(function (text) {
      var cell = document.createElement('th');
      cell.textContent = text;
      line.appendChild(cell);
    });
  }

  function choose(i) {
    var period = DATA && DATA.periods[i];
    if (!period) { return; }
    CHOSEN = i;
    for (var k = 0; k < PERIODS; k++) {
      byId('period-' + k).setAttribute('aria-pressed', k === i ? 'true' : 'false');
    }
    byId('live-range').textContent = period.note;
    CARDS.forEach(function (key) {
      var card = period.cards[key];
      var value = byId('card-' + key);
      value.textContent = card.text;
      value.className = 'card-value ' + card.tone;
      value.title = card.title;
    });
    byId('live-trades').textContent = period.trades;
    rows('live-group-rows', period.group_rows);
    var P = charts();
    if (P) {
      P.react('live-groups', period.groups.data, period.groups.layout, CHART_CONFIG);
      if (DATA.area) { P.relayout('live-area', {'xaxis.range': period.range}); }
    }
  }

  function show(data) {
    DATA = data;
    var status = byId('live-status');
    status.textContent = data.status;
    status.className = 'status';
    var empty = byId('live-empty');
    empty.textContent = data.empty || '';
    empty.hidden = !data.empty;
    byId('live-stats').hidden = !data.periods.length;
    rows('live-holdings', data.holdings.rows);
    rows('live-holdings-foot', data.holdings.foot);
    var none = byId('live-holdings-empty');
    none.textContent = data.holdings.empty || '';
    none.hidden = !data.holdings.empty;
    fill('live-notes', data.notes);
    fill('live-warnings', data.warnings);
    var table = data.area_table;
    byId('live-area-details').hidden = !table;
    if (table) {
      head('live-area-head', table.head);
      rows('live-area-rows', table.rows);
    }
    var P = charts();
    if (!P) {
      status.textContent = data.status + ' · ' + TEXT.no_charts;
      status.className = 'status error';
    } else if (data.area) {
      P.react('live-area', data.area.data, data.area.layout, CHART_CONFIG);
    }
    choose(CHOSEN < data.periods.length ? CHOSEN : 0);
  }

  function failed(message) {
    var status = byId('live-status');
    status.textContent = DATA ? message + ' · ' + DATA.stale : message;
    status.className = 'status error';
  }

  function load() {
    if (LOADING) { return; }
    LOADING = true;
    var button = byId('live-refresh');
    button.disabled = true;
    var status = byId('live-status');
    status.textContent = TEXT.loading;
    status.className = 'status';
    function done() {
      LOADING = false;
      button.disabled = false;
    }
    fetch('/api/live', {cache: 'no-store', headers: HEADERS}).then(function (answer) {
      return answer.json().then(function (body) {
        if (answer.ok && body && !body.error) { show(body); }
        else { failed((body && body.error) || TEXT.unreadable); }
      }, function () { failed(TEXT.unreadable); });
    }, function () { failed(TEXT.unreachable); }).then(done, function () {
      failed(TEXT.unreadable);
      done();
    });
  }

  function tab(name) {
    var live = name === 'live';
    byId('tab-live').setAttribute('aria-selected', live ? 'true' : 'false');
    byId('tab-backtest').setAttribute('aria-selected', live ? 'false' : 'true');
    byId('panel-live').hidden = !live;
    byId('panel-backtest').hidden = live;
    if (!live && !OPENED) {
      byId('backtest-frame').src = BACKTEST_URL;
      OPENED = true;
    }
    var P = charts();
    if (live && P && DATA) {
      if (DATA.area) { P.Plots.resize(byId('live-area')); }
      if (DATA.periods.length) { P.Plots.resize(byId('live-groups')); }
    }
  }

  byId('tab-live').addEventListener('click', function () { tab('live'); });
  byId('tab-backtest').addEventListener('click', function () { tab('backtest'); });
  byId('live-refresh').addEventListener('click', load);
  for (var k = 0; k < PERIODS; k++) {
    (function (i) {
      byId('period-' + i).addEventListener('click', function () { choose(i); });
    })(k);
  }
  load();
})();
"""


def _script() -> str:
    """
    The tab page's script, with this server's settings written in.

    Returns:
        str: The script's text, exactly as it sits between the page's
            <script> tags (its hash is the page's Content-Security-Policy's).
    """
    values = {
        "__BACKTEST_URL__": json.dumps(BACKTEST_URL),
        "__TEXT__": json.dumps(_SCRIPT_TEXT, ensure_ascii=False),
        "__HEADERS__": json.dumps({config.LIVE_DASHBOARD_REQUEST_HEADER: "1"}),
        "__PERIODS__": str(len(config.LIVE_DASHBOARD_PERIODS)),
        "__CARDS__": json.dumps([key for key, _ in CARD_LABELS]),
        "__CHART_CONFIG__": json.dumps(_CHART_CONFIG),
    }
    text = _SCRIPT_TEMPLATE
    for name, value in values.items():
        text = text.replace(name, value)
    return text


_SCRIPT = _script()
_SCRIPT_HASH = "sha256-" + b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode()

# The page's Content-Security-Policy: its own script (by hash) and the one
# Plotly file, no other script; its own data only; the backtest page as its
# only frame; Plotly's PNG download (data: and blob: images)
_PAGE_CSP = (f"default-src 'none'; script-src {config.LIVE_DASHBOARD_PLOTLY_URL} "
             f"'{_SCRIPT_HASH}'; style-src 'unsafe-inline'; connect-src 'self'; "
             f"frame-src {_BACKTEST_ORIGIN}; img-src data: blob:; base-uri 'none'; "
             "form-action 'none'; frame-ancestors 'none'")

_PAGE_HEADERS = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Content-Security-Policy", _PAGE_CSP),
)

_STYLE = f"""
:root {{
  color-scheme: light;
  --plane: {_PLANE}; --surface: {_SURFACE}; --ink: {_INK}; --ink-2: {_INK_2};
  --muted: #898781; --grid: {_GRID}; --axis: {_AXIS}; --up: {_UP}; --down: {_DOWN};
  --ring: rgba(11,11,11,0.10); --accent: {_PALETTE[0]};
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--plane); color: var(--ink); font-family: {_FONT}; }}
[hidden] {{ display: none !important; }}
.tabs {{ display: flex; gap: 4px; padding: 8px 16px 0; border-bottom: 1px solid var(--grid);
  background: var(--surface); }}
.tabs button {{ font: inherit; font-size: 15px; padding: 8px 16px; border: 1px solid transparent;
  border-bottom: none; border-radius: 8px 8px 0 0; background: none; color: var(--ink-2);
  cursor: pointer; }}
.tabs button[aria-selected="true"] {{ background: var(--plane); color: var(--ink);
  border-color: var(--grid); font-weight: 600; }}
#panel-live {{ max-width: 1200px; margin: 0 auto; padding: 16px 24px 48px; }}
.head {{ display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
h1 {{ font-size: 24px; margin: 8px 0; }}
h2 {{ font-size: 17px; margin: 28px 0 8px; }}
.status {{ color: var(--ink-2); }}
.status.error {{ color: var(--down); font-weight: 600; }}
button.action {{ font: inherit; padding: 4px 12px; border: 1px solid var(--axis);
  border-radius: 6px; background: var(--surface); color: var(--ink); cursor: pointer; }}
button.action:disabled {{ color: var(--muted); cursor: default; }}
.periods {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 12px 0; }}
.periods button {{ font: inherit; padding: 4px 12px; border: 1px solid var(--axis);
  border-radius: 6px; background: var(--surface); color: var(--ink); cursor: pointer; }}
.periods button[aria-pressed="true"] {{ background: var(--ink); color: var(--surface);
  border-color: var(--ink); }}
.periods .range {{ margin-left: auto; color: var(--ink-2); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px;
  padding: 10px 14px; }}
.card-label {{ color: var(--ink-2); font-size: 13px; }}
.card-value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
.card-value.up {{ color: var(--up); }}
.card-value.down {{ color: var(--down); }}
.trades {{ color: var(--ink-2); font-size: 14px; margin: 10px 0 0; }}
.chart {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; background: var(--surface);
  font-variant-numeric: tabular-nums; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--ink-2); font-weight: 600; }}
tfoot td {{ font-weight: 600; }}
#holdings-table td:nth-child(n+3):nth-child(-n+6), #holdings-table th:nth-child(n+3):nth-child(-n+6),
#group-table td:nth-child(n+2), #group-table th:nth-child(n+2),
#area-table td:nth-child(n+2), #area-table th:nth-child(n+2) {{ text-align: right; }}
#holdings-table td:nth-child(n+2), #holdings-table th:nth-child(n+2),
#area-table td, #area-table th {{ white-space: nowrap; }}
details {{ margin: 8px 0; }}
summary {{ color: var(--ink-2); cursor: pointer; }}
#area-table {{ display: block; overflow-x: auto; }}
.notes li {{ color: var(--ink-2); margin: 4px 0; }}
.warnings li {{ color: var(--down); margin: 4px 0; }}
.empty {{ color: var(--ink-2); font-size: 15px; }}
#panel-backtest iframe {{ display: block; width: 100%; height: calc(100vh - 48px); border: 0;
  background: var(--surface); }}
"""


# What the backtest page may do in its frame: run its scripts in its own
# origin (its chunk files), open the defaults server's pages in a window of
# their own (their forms then work, outside the frame's limits) and save a
# chart's picture; never navigate the tab page itself
_FRAME_SANDBOX = ("allow-scripts allow-same-origin allow-popups "
                  "allow-popups-to-escape-sandbox allow-downloads")


def _shell() -> str:
    """
    The tab page: the two tabs, the Live trading panel's places, the backtest frame and the script.

    Built once, from config: the period buttons from
    config.LIVE_DASHBOARD_PERIODS, the cards from CARD_LABELS, the Plotly
    file with its integrity hash.

    Returns:
        str: The page's HTML.
    """
    periods = "".join(
        f'<button type="button" id="period-{i}" aria-pressed="{"true" if i == 0 else "false"}">'
        f"{escape(label)}</button>"
        for i, (label, _) in enumerate(config.LIVE_DASHBOARD_PERIODS))
    cards = "".join(
        f'<div class="card"><div class="card-label" title="{escape(_CARD_HINTS[key])}">'
        f'{escape(label)}</div><div class="card-value flat" id="card-{key}">—</div></div>'
        for key, label in CARD_LABELS)
    holdings_head = "".join(f"<th>{name}</th>" for name in (
        "Market", "Side", "Contracts", "Price", "Value", "Cost", "Category"))
    group_head = "".join(f"<th>{name}</th>" for name in ("Category", "Return", "Profit", "Put in"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalshi dashboard</title>
<style>{_STYLE}</style>
<script src="{escape(config.LIVE_DASHBOARD_PLOTLY_URL)}" integrity="{config.LIVE_DASHBOARD_PLOTLY_SRI}" crossorigin="anonymous"></script>
</head>
<body>
<div class="tabs" role="tablist">
<button type="button" id="tab-live" role="tab" aria-selected="true" aria-controls="panel-live">Live trading</button>
<button type="button" id="tab-backtest" role="tab" aria-selected="false" aria-controls="panel-backtest">Backtest</button>
</div>
<section id="panel-live" role="tabpanel" aria-labelledby="tab-live">
<div class="head"><h1>Live trading</h1><span id="live-status" class="status">{escape(_SCRIPT_TEXT["loading"])}</span>
<button type="button" class="action" id="live-refresh">Refresh</button></div>
<ul id="live-warnings" class="warnings" hidden></ul>
<p id="live-empty" class="empty" hidden></p>
<div id="live-stats">
<div class="periods" role="group" aria-label="Period">{periods}<span id="live-range" class="range"></span></div>
<div class="cards">{cards}</div>
<p id="live-trades" class="trades"></p>
<h2>Account value by category</h2>
<div id="live-area" class="chart"></div>
<details id="live-area-details"><summary>The chart's figures</summary>
<table id="area-table"><thead><tr id="live-area-head"></tr></thead><tbody id="live-area-rows"></tbody></table></details>
<h2>Return by category</h2>
<div id="live-groups" class="chart"></div>
<table id="group-table"><thead><tr>{group_head}</tr></thead><tbody id="live-group-rows"></tbody></table>
</div>
<h2>Holdings now</h2>
<table id="holdings-table"><thead><tr>{holdings_head}</tr></thead><tbody id="live-holdings"></tbody><tfoot id="live-holdings-foot"></tfoot></table>
<p id="live-holdings-empty" class="empty" hidden></p>
<h2>Notes</h2>
<ul id="live-notes" class="notes"></ul>
</section>
<section id="panel-backtest" role="tabpanel" aria-labelledby="tab-backtest" hidden>
<iframe id="backtest-frame" title="Backtest dashboard" sandbox="{_FRAME_SANDBOX}"></iframe>
</section>
<script>{_SCRIPT}</script>
</body>
</html>
"""


_SHELL = _shell()


# ---- the socket side --------------------------------------------------------------

def _log_safe(text: str) -> str:
    """
    Make text safe to write on one log line: every non-printable character as its escape.

    Args:
        text (str): The text.

    Returns:
        str: The text with each non-printable character as \\xNN, \\uNNNN or \\UNNNNNNNN.
    """
    out = []
    for ch in text:
        if ch.isprintable():
            out.append(ch)
        elif ord(ch) < 0x100:
            out.append(f"\\x{ord(ch):02x}")
        elif ord(ch) < 0x10000:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return "".join(out)


class _Handler(BaseHTTPRequestHandler):
    """
    The socket side of both servers: reads one request, hands it to the server's app, sends the answer.

    The server's app is its `app` attribute (a PageApp or a BacktestApp). A
    connection that sends nothing is dropped after `timeout` seconds. Only
    GET and HEAD reach the app; any other method gets 405 and a request the
    base class cannot read its own 400, 414, 431 or 505, each with the app's
    headers. A file body is opened first (a file gone by then is 404), its
    length taken from the open file, and it is copied with no time limit,
    since the backtest page can be hundreds of MB. A browser that stops
    reading part way is let go quietly.
    """

    timeout = config.LIVE_DASHBOARD_SOCKET_TIMEOUT_SECONDS
    server_version = "KalshiLiveDashboard"
    sys_version = ""

    def do_GET(self) -> None:
        """Answer a GET."""
        self._serve()

    def do_HEAD(self) -> None:
        """Answer a HEAD: the GET's status and headers, with no body."""
        self._serve()

    def _headers_once(self) -> tuple[dict[str, str], frozenset[str]]:
        """
        Every header given exactly once, by its lower-case name, and the names of those given more often.

        Returns:
            tuple[dict[str, str], frozenset[str]]: The headers given once,
                and the lower-case names of the headers given twice or more
                (in any mix of cases).
        """
        once, repeated = {}, set()
        for name in {key.lower() for key in self.headers.keys()}:
            values = self.headers.get_all(name) or []
            if len(values) == 1:
                once[name] = values[0]
            else:
                repeated.add(name)
        return once, frozenset(repeated)

    def _serve(self) -> None:
        """Hand the request to the server's app and send its answer; an app that raises answers 500."""
        headers, repeated = self._headers_once()
        request = _Request(self.command, self.path, headers.get("host"), headers, repeated)
        app = self.server.app
        try:
            response = app.handle(request)
        except Exception:
            logging.exception("The live dashboard failed on %s %s", self.command,
                              _log_safe(self.path))
            response = app.failed(request)
        self._send(response)

    def _start(self, response: _Response, length: int) -> None:
        """
        Send the status line and headers.

        Args:
            response (_Response): The answer.
            length (int): Its body's length in bytes.
        """
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        for name, value in response.headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send(self, response: _Response) -> None:
        """
        Send an answer, its body from memory or from disk.

        A browser that closes the connection, resets it or stops reading
        before the answer is sent is let go with a DEBUG line.

        Args:
            response (_Response): The answer.
        """
        if self.request_version == "HTTP/0.9":
            self.request_version = self.protocol_version
        try:
            if isinstance(response.body, Path):
                self._send_file(response)
                return
            body = (response.body if isinstance(response.body, bytes)
                    else response.body.encode("utf-8"))
            self._start(response, len(body))
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as exc:
            logging.debug("The browser left before the answer was sent (%s)", type(exc).__name__)

    def _send_file(self, response: _Response) -> None:
        """
        Send a file from disk: opened first, its length from the open file, copied in blocks.

        A file gone before it was opened (a rebuild deleted it) is answered
        404; one that cannot be opened for another reason, 500. A file that
        is not the one the application checked (its device and inode differ:
        it was replaced, or a link put in its place, after the check) is
        answered 404, so nothing outside the checked files is ever sent. Once
        open, the connection has no time limit, so a slow browser is never
        cut off.

        Args:
            response (_Response): The answer; its body is the file's Path.
        """
        try:
            handle = response.body.open("rb")
        except FileNotFoundError:
            self._send(self.server.app.not_found())
            return
        except OSError:
            logging.exception("Could not open %s", response.body)
            self._send(self.server.app.failed())
            return
        with handle:
            opened = os.fstat(handle.fileno())
            if response.file_id is not None and (opened.st_dev, opened.st_ino) != response.file_id:
                self._send(self.server.app.not_found())
                return
            length = opened.st_size
            self._start(response, length)
            if self.command == "HEAD":
                return
            self.connection.settimeout(None)
            left = length
            while left > 0:
                block = handle.read(min(config.LIVE_DASHBOARD_FILE_BLOCK_BYTES, left))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)

    def send_error(self, code: int, message: str | None = None,
                   explain: str | None = None) -> None:
        """
        Answer a request the base class refuses before the app sees it.

        A method this server does not answer (the base class's 501) is a 405;
        a request it cannot read keeps its own status. Each is a short text
        with the app's headers, and the connection is then closed.

        Args:
            code (int): The HTTP status.
            message (str | None): The base class's short reason; None shows
                the status's own phrase.
            explain (str | None): The base class's longer explanation; not shown.
        """
        self.close_connection = True
        headers = self.server.app.error_headers
        if code == HTTPStatus.NOT_IMPLEMENTED:
            response = _text(405, "This server only shows pages.", headers)
        else:
            response = _text(code, message or HTTPStatus(code).phrase, headers)
        self._send(response)

    def log_message(self, format: str, *args) -> None:
        """
        Log one line about a request at DEBUG, made safe first.

        Args:
            format (str): The base class's format string.
            *args: Its arguments.
        """
        logging.debug("%s", _log_safe(f"{self.address_string()} {format % args}"))


# ---- reading the account ------------------------------------------------------------

def _with_timeouts(client: Any) -> Any:
    """
    Give every request this client sends the connect and read timeouts of config.LIVE_KALSHI_TIMEOUT_SECONDS.

    Only this client's own transport is changed (its rest_client's request),
    so the timeouts reach every read live_portfolio and historical make with
    it and no other client in the process. The transport also refuses every
    method but GET, before anything is sent, so this server's client can
    only read, whatever code calls it.

    Args:
        client (Any): A client from historical.build_prod_live_client.

    Returns:
        Any: The same client.
    """
    rest = client.rest_client
    request = rest.request

    def timed(method, url, headers=None, body=None, post_params=None, _request_timeout=None):
        """
        The client's own request, a GET only, always with this server's timeouts.

        Raises:
            RuntimeError: For any method but GET; nothing is sent.
        """
        if method != "GET":
            raise RuntimeError(f"the live dashboard sends GET requests only, not {method}")
        return request(method, url, headers=headers, body=body, post_params=post_params,
                       _request_timeout=config.LIVE_KALSHI_TIMEOUT_SECONDS)

    rest.request = timed
    return client


class _RiskFree:
    """
    The T-bill yields the ratios subtract: downloaded now and again every config.LIVE_RISK_FREE_REFRESH_SECONDS.

    Attributes:
        current (treasury.RiskFreeRates | None): The latest download; None
            until the first one lands (the ratios then subtract 0%, and the
            page says so).
        tried (threading.Event): Set once the first download has finished,
            whether it worked or not.
    """

    def __init__(self) -> None:
        """Start with no yields."""
        self.current: treasury.RiskFreeRates | None = None
        self.tried = threading.Event()

    def first(self) -> treasury.RiskFreeRates | None:
        """
        The yields, waiting for the first download when it has not finished yet.

        A read of the account that starts just after the server does (the
        page the server opens) would otherwise always subtract 0%. The wait
        is at most config.LIVE_RISK_FREE_FIRST_WAIT_SECONDS (read when
        called), since the download can take minutes when the Treasury's
        server does not answer.

        Returns:
            treasury.RiskFreeRates | None: The yields; None when none has
                landed in time.
        """
        if self.current is None:
            self.tried.wait(config.LIVE_RISK_FREE_FIRST_WAIT_SECONDS)
        return self.current

    def keep_fresh(self, stop: threading.Event) -> None:
        """
        Download the yields now, then again every refresh interval, until `stop` is set.

        treasury.load_risk_free_rates never raises; an unexpected error is a
        WARNING and the yields already held are kept.

        Args:
            stop (threading.Event): Set when the server stops.
        """
        while True:
            try:
                self.current = treasury.load_risk_free_rates()
            except Exception as exc:
                logging.warning("Could not download the T-bill yields: %s",
                                api_error_summary(exc))
            self.tried.set()
            if stop.wait(config.LIVE_RISK_FREE_REFRESH_SECONDS):
                return


class _LiveReader:
    """
    The read behind /api/live: the account from Kalshi, worked out by live_portfolio, as the page's JSON.

    The production client is built on the first read (historical.
    build_prod_live_client, with this server's timeouts), so the page comes
    up even when the credentials cannot be read; such a read fails and is
    answered 502, and the next read tries again.
    """

    def __init__(self, risk_free: _RiskFree) -> None:
        """
        Set up with no client yet: it is built on the first read.

        Args:
            risk_free (_RiskFree): Where the T-bill yields are kept.
        """
        self._risk_free = risk_free
        self._client: Any = None

    def _kalshi(self) -> Any:
        """
        The production client, built on first use.

        Returns:
            Any: The client.
        """
        if self._client is None:
            # Cross-module: the production client, as the backtest's reads use
            self._client = _with_timeouts(historical.build_prod_live_client())
        return self._client

    def __call__(self) -> str:
        """
        Read the account and answer the page's data.

        Reads Kalshi's categories (historical.load_series_categories, cached a
        week), takes the T-bill yields (waiting briefly for the first
        download, _RiskFree.first), builds the view
        (live_portfolio.build_live_view over the trade logs), adds the read
        to config.LIVE_PORTFOLIO_LOG_FILE, logs one line and returns
        payload(view) as JSON.

        Returns:
            str: The JSON text.
        """
        client = self._kalshi()
        # Cross-module: the categories the bot's purchases are filed under, as on the backtest page
        series = historical.load_series_categories(client)
        risk_free = self._risk_free.first()
        view = live_portfolio.build_live_view(client, risk_free=risk_free,
                                              series_categories=series,
                                              trade_logs=live_portfolio.trade_log_paths())
        # Cross-module: one JSON line per read; a write that fails is only a WARNING
        live_portfolio.append_snapshot(view)
        kalshi = ("not read" if view.kalshi_positions_value is None
                  else _money(view.kalshi_positions_value))
        logging.info("Read the account: cash %s, %s worth %s at the midpoint (Kalshi: %s)",
                     _money(view.cash), _count_words(len(view.holdings), "holding", "holdings"),
                     _money(view.holdings_value), kalshi)
        return json.dumps(payload(view), allow_nan=False)


# ---- starting ---------------------------------------------------------------------

class _Server(ThreadingHTTPServer):
    """
    One of the live dashboard's two servers.

    A thread per connection (daemon threads, so stopping never waits for
    one), and a queue of config.LIVE_DASHBOARD_LISTEN_BACKLOG connections
    waiting to be accepted, so a burst (a page load beside the backtest
    page's chunk files) is never refused.
    """

    request_queue_size = config.LIVE_DASHBOARD_LISTEN_BACKLOG


def _port_answers(host: str, port: int) -> bool:
    """
    Whether a program already accepts connections on this host and port.

    Asked before binding: on macOS a bind to 127.0.0.1 succeeds while
    another program listens on every address (0.0.0.0) of the same port,
    and would then take that program's connections from this computer. A
    connection that is accepted shows the port is in use either way. Waits
    at most config.LIVE_DASHBOARD_HEALTH_TIMEOUT_SECONDS; never raises.

    Args:
        host (str): The address.
        port (int): The port.

    Returns:
        bool: True when a connection was accepted.
    """
    try:
        with socket.create_connection((host, port),
                                      timeout=config.LIVE_DASHBOARD_HEALTH_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def _code_fingerprint() -> str:
    """
    Fingerprint this package's Python code as it is on disk now.

    A SHA-256 over every .py file in the package's own folder, in name
    order, each as its name and then its length and bytes (the defaults
    server's recipe). A file that cannot be read counts as its name and the
    error's type, and a folder that cannot be listed as no files, so it
    never raises.

    Returns:
        str: The fingerprint, 64 hex digits.
    """
    digest = hashlib.sha256()
    folder = Path(__file__).resolve().parent
    try:
        files = sorted(folder.glob("*.py"))
    except OSError:
        files = []
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        try:
            data = path.read_bytes()
        except OSError as exc:
            data = f"unreadable: {type(exc).__name__}".encode()
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


# The fingerprint of the code this process loaded, taken when this module is
# imported; GET /health reports it, so a second start can tell a server left
# running across a code change
_LOADED_CODE = _code_fingerprint()


@dataclass(frozen=True)
class _Running:
    """
    What the live dashboard already on the port says about itself (GET /health).

    Attributes:
        project_root (str): The resolved checkout root it serves.
        code (str | None): The fingerprint of the code it loaded; None when
            its answer carries none.
    """
    project_root: str
    code: str | None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect handler that follows none: an answer must come from the listener itself."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Refuse every redirect, so urllib raises HTTPError for the 3xx instead.

        Returns:
            None: Never a new request.
        """
        return None


def _running_dashboard(base: str) -> _Running | None:
    """
    Ask the server listening at `base` what it is, or None when it is not a live dashboard that answers.

    One GET of base/health, through no proxy and following no redirect, with
    a config.LIVE_DASHBOARD_HEALTH_TIMEOUT_SECONDS timeout, reading at most
    config.LIVE_DASHBOARD_HEALTH_MAX_BYTES. Only a success whose body is a
    JSON object naming this app and a non-empty string "project_root"
    counts. It never raises.

    Args:
        base (str): The server's address, "http://host:port".

    Returns:
        _Running | None: What it says; None for anything else.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects)
    try:
        with opener.open(f"{base}/health",
                         timeout=config.LIVE_DASHBOARD_HEALTH_TIMEOUT_SECONDS) as answer:
            body = answer.read(config.LIVE_DASHBOARD_HEALTH_MAX_BYTES + 1)
        if len(body) > config.LIVE_DASHBOARD_HEALTH_MAX_BYTES:
            return None
        record = json.loads(body.decode("utf-8"))
    except (OSError, ValueError, RecursionError, http.client.HTTPException):
        return None
    if not isinstance(record, dict) or record.get("app") != _APP:
        return None
    root, code = record.get("project_root"), record.get("code")
    if not isinstance(root, str) or not root:
        return None
    return _Running(root, code if isinstance(code, str) else None)


def _setup_logging(log_path: Path | None) -> None:
    """
    Log to the terminal and, when given one, to a rotating file (5 MB across 3 backups).

    Args:
        log_path (Path | None): The log file; None logs to the terminal only
            (a start that found this checkout's dashboard already running
            leaves the log file to it).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)


def _open_page(base: str, *, no_browser: bool) -> None:
    """
    Log the page's address and, unless told not to, open it in the browser.

    Args:
        base (str): The tab page's address, "http://host:port".
        no_browser (bool): Whether to open nothing.
    """
    logging.info("Open %s/ for the Live trading and Backtest tabs", base)
    if not no_browser:
        webbrowser.open(f"{base}/")


def _reuse_running_dashboard(parser: argparse.ArgumentParser, base: str, *,
                             no_browser: bool) -> None:
    """
    Answer a start whose port is taken: reopen this checkout's running dashboard, or refuse.

    It binds nothing and starts nothing. When the listener's /health names
    this app, this checkout's resolved root and this checkout's code now, it
    opens the page (unless no_browser) and returns. Otherwise it exits 2:
    this checkout's dashboard running older code is told to stop and start
    again, and anything else is named.

    Args:
        parser (argparse.ArgumentParser): main's parser, for its error exit.
        base (str): The tab page's address.
        no_browser (bool): Whether to open nothing.

    Raises:
        SystemExit: Status 2 (parser.error) unless the listener is this
            checkout's dashboard running this checkout's code.
    """
    port = config.LIVE_DASHBOARD_PORT
    running = _running_dashboard(base)
    if running is None:
        parser.error(f"port {port} is in use by another program — stop it, or change "
                     "config.LIVE_DASHBOARD_PORT")
    if running.project_root != str(config.PROJECT_ROOT.resolve()):
        parser.error(f"port {port} is served by the live dashboard of "
                     f"{_log_safe(running.project_root)} — stop it (Ctrl-C in its terminal) "
                     "before starting this checkout's")
    if running.code != _code_fingerprint():
        parser.error(f"port {port} is served by this checkout's live dashboard, but it is "
                     "running code from before a change to this checkout — stop it (Ctrl-C "
                     "in its terminal) and start it again")
    _setup_logging(None)
    logging.info("This checkout's live dashboard is already running at %s/", base)
    _open_page(base, no_browser=no_browser)


def main(argv: list[str] | None = None) -> None:
    """
    Serve both tabs until Ctrl-C, or reopen this checkout's running dashboard's page.

    It binds config.LIVE_DASHBOARD_PORT, then config.LIVE_BACKTEST_PORT, on
    config.LIVE_DASHBOARD_HOST. A port is taken when a program already
    accepts a connection to it (_port_answers, asked first) or it cannot be
    bound. When the first is taken it binds nothing: this checkout's own
    dashboard running this checkout's code has its page opened again;
    anything else exits 2. Only once both are bound does it
    log, to the terminal and config.LIVE_DASHBOARD_LOG_FILE. It then starts
    downloading the T-bill yields (in the background, again daily), serves
    the backtest page from a background thread and the tab page from this
    one, and opens the page unless --no-browser. The Kalshi client is built
    on the first read of the account.

    Args:
        argv (list[str] | None): The arguments; None reads the command line.

    Raises:
        SystemExit: Status 2 when a port is held by anything but this
            checkout's dashboard running this checkout's code, or an argument
            is invalid.
        OSError: When a port cannot be bound for another reason.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.live_dashboard",
        description="Serve the dashboard's Live trading tab (the account read from Kalshi on "
                    "every load, read-only) and its Backtest tab. When this checkout's "
                    "dashboard is already running, open its page again instead.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Open nothing; only log the address to open")
    args = parser.parse_args(argv)
    # A job started in the background has Ctrl-C ignored; Ctrl-C must stop the servers
    signal.signal(signal.SIGINT, signal.default_int_handler)
    host = config.LIVE_DASHBOARD_HOST
    base = f"http://{host}:{config.LIVE_DASHBOARD_PORT}"
    backtest_taken = (f"port {config.LIVE_BACKTEST_PORT} is in use by another program — stop "
                      "it, or change config.LIVE_BACKTEST_PORT")
    if _port_answers(host, config.LIVE_DASHBOARD_PORT):
        _reuse_running_dashboard(parser, base, no_browser=args.no_browser)
        return
    if _port_answers(host, config.LIVE_BACKTEST_PORT):
        parser.error(backtest_taken)
    try:
        page_server = _Server((host, config.LIVE_DASHBOARD_PORT), _Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        _reuse_running_dashboard(parser, base, no_browser=args.no_browser)
        return
    try:
        backtest_server = _Server((host, config.LIVE_BACKTEST_PORT), _Handler)
    except OSError as exc:
        page_server.server_close()
        if exc.errno != errno.EADDRINUSE:
            raise
        parser.error(backtest_taken)
    _setup_logging(config.LIVE_DASHBOARD_LOG_FILE)
    risk_free, stop = _RiskFree(), threading.Event()
    page_server.app = PageApp(build=_LiveReader(risk_free))
    backtest_server.app = BacktestApp()
    threading.Thread(target=risk_free.keep_fresh, args=(stop,), name="risk-free",
                     daemon=True).start()
    threading.Thread(target=backtest_server.serve_forever, name="backtest-page",
                     daemon=True).start()
    logging.info("Live dashboard at %s/ and the backtest page at %s — it reads the account "
                 "from Kalshi on every load and places no orders. Ctrl-C stops it.", base,
                 BACKTEST_URL)
    _open_page(base, no_browser=args.no_browser)
    try:
        page_server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Live dashboard stopped")
    finally:
        stop.set()
        backtest_server.shutdown()
        backtest_server.server_close()
        page_server.server_close()


if __name__ == "__main__":
    main()
