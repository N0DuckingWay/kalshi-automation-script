"""Tests for live_dashboard.py: what the Live trading tab shows (payload), the
tab page's server (one read of the account at a time, refusals of every
request not from its own page), the backtest page's server (the page and its
chunk files, and nothing outside them), the socket side, main()'s start and
its reuse of a running dashboard, the page's script (run under
tests/js/live_harness.js with node or jsc, skipping without either), and the
rules that keep the dashboard read-only and away from the order path.

All offline: the Kalshi client is never built for real (historical's builder
and live_portfolio's view are replaced), servers bind loopback ports the
operating system chooses and are closed again, and the browser is replaced.
"""
import ast
import dataclasses
import errno
import hashlib
import http.client
import importlib
import inspect
import json
import logging
import os
import pkgutil
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from email.utils import formatdate
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin, urlsplit

import pytest
from kalshi_python_sync.exceptions import ApiException

import kalshi_betting
from kalshi_betting import config, historical, live_dashboard, live_portfolio, treasury
from kalshi_betting.live_dashboard import (
    CASH,
    MORE_CATEGORIES,
    BacktestApp,
    PageApp,
    _Request,
    _Response,
    payload,
)
from kalshi_betting.live_portfolio import (
    OTHER_BETS,
    CashCheck,
    GroupReturn,
    History,
    Holding,
    LiveView,
    PeriodStats,
)

from .test_scheduler import _host_zone

D = Decimal
PAGE_HOST = f"127.0.0.1:{config.LIVE_DASHBOARD_PORT}"
BACKTEST_HOST = f"127.0.0.1:{config.LIVE_BACKTEST_PORT}"
# A read of the account at 01:23 PDT on Oct 8, and the bot's first fill
READ = datetime(2026, 10, 8, 8, 23, tzinfo=UTC)
START = datetime(2026, 9, 28, 6, 30, tzinfo=UTC)
# The page script's own request header, as config names it
LIVE_HEADER = {config.LIVE_DASHBOARD_REQUEST_HEADER: "1"}


# ---- views built from Decimal fixtures ------------------------------------------

def _times() -> tuple[datetime, ...]:
    """The moments a history from START to READ is valued at."""
    return (START, *live_portfolio.day_ends(START, READ), READ)


def _history(groups: dict[str, float], cash: float = 100.0) -> History:
    """
    A history in which each group starts at its value and gains 1% a moment, and the cash stays put.

    The groups first appear in the order given, a minute apart (one change
    each, moving no cash), which is the order live_portfolio lists them in.

    Args:
        groups (dict[str, float]): Each group's value at the start.
        cash (float): The cash at every moment.

    Returns:
        History: The history (no cash in or out by any group, no deposit).
    """
    times = _times()
    value = {g: tuple(start * 1.01 ** i for i in range(len(times))) for g, start in groups.items()}
    zeros = tuple(0.0 for _ in times)
    steps = {g: ((START + timedelta(minutes=i + 1), 0.0),) for i, g in enumerate(groups)}
    return History(times, tuple(cash for _ in times), value, dict.fromkeys(groups, zeros),
                   dict.fromkeys(groups, zeros), steps, zeros)


def _holding(ticker, group, side="yes", contracts="10", price="0.45", cost="3.07",
             subtitle="") -> Holding:
    """One holding, its value contracts × price."""
    count = D(contracts)
    worth = None if price is None else count * D(price)
    return Holding(ticker, f"Will {ticker}?", group, side, count,
                   None if price is None else D(price),
                   worth if worth is not None else D(cost), D(cost), subtitle)


def _view(groups: dict[str, float] | None = None, *, holdings=None, warnings=(),
          history=True, **changes) -> LiveView:
    """
    A LiveView as build_live_view returns one, its periods worked out by live_portfolio itself.

    Args:
        groups (dict[str, float] | None): Each group's value at the start;
            None is Crypto, Climate and Other bets.
        holdings: The holdings; None is one per group.
        warnings: The view's warnings.
        history (bool): False for a view from before the bot's first trade.
        **changes: Any other field.

    Returns:
        LiveView: The view.
    """
    groups = {"Crypto": 20.0, "Climate": 10.0, OTHER_BETS: 5.0} if groups is None else groups
    held = (tuple(_holding(f"T{i}", g) for i, g in enumerate(groups))
            if holdings is None else tuple(holdings))
    hist = _history(groups) if history else None
    periods = None if hist is None else tuple(
        live_portfolio.period_stats(hist, [], None, label, months)
        for label, months in config.LIVE_DASHBOARD_PERIODS)
    order = live_portfolio._group_order(hist, list(held))
    view = LiveView(READ, D("90.9050"), D("12.25"), held, order, hist, periods, (),
                    CashCheck(22, 22, D("0.0075"), ()), None, False, tuple(warnings))
    return dataclasses.replace(view, **changes)


def _stats(**changes) -> PeriodStats:
    """One period's figures, set by hand."""
    stats = PeriodStats("All", START, READ, 0.123, 4.56, 1.234, 2.5, 9, 0.05, -0.02,
                        D("312"), 41, D("247"), (GroupReturn("Crypto", 4.56, 37.08, 0.123),))
    return dataclasses.replace(stats, **changes)


# ---- payload -----------------------------------------------------------------------

class TestCards:
    """The six cards' text, color and tooltip."""

    def test_each_card_is_formatted_and_colored_by_its_sign(self):
        cards = live_dashboard._cards(_stats())
        assert list(cards) == [key for key, _ in live_dashboard.CARD_LABELS]
        assert {k: (c["text"], c["tone"]) for k, c in cards.items()} == {
            "return": ("+12.3%", "up"), "pnl": ("+$4.56", "up"), "sharpe": ("1.23", "up"),
            "sortino": ("2.50", "up"), "mean": ("+5.0%", "up"), "median": ("-2.0%", "down")}
        assert "9 whole days" in cards["sharpe"]["title"]
        assert "312 contract pairs" in cards["mean"]["title"]
        # The span, without parentheses inside the sentence's own
        assert cards["return"]["title"].endswith(
            "with deposits and withdrawals taken out, over Sep 28 – Oct 8, 2026, 10 days")

    def test_the_card_labels_explain_themselves(self):
        assert live_dashboard._CARD_HINTS["sharpe"] == (
            "Return for each unit of day-to-day swing, scaled to a year")

    def test_losses_and_missing_figures(self):
        cards = live_dashboard._cards(_stats(
            total_return=-0.0412, pnl=-4.56, sharpe=None, sortino=-0.5, whole_days=1,
            mean_trade=None, median_trade=None, purchases=0, pairs=D(0)))
        assert {k: (c["text"], c["tone"]) for k, c in cards.items()} == {
            "return": ("-4.1%", "down"), "pnl": ("-$4.56", "down"), "sharpe": ("—", "flat"),
            "sortino": ("-0.50", "down"), "mean": ("—", "flat"), "median": ("—", "flat")}
        assert "needs at least 2 whole days" in cards["sharpe"]["title"]
        assert "this period has 1 whole day" in cards["sharpe"]["title"]
        assert cards["mean"]["title"] == "The bot bought no contract pair in this period"

    @pytest.mark.parametrize("least, shown", [(9, True), (10, False)])
    def test_the_ratio_minimum_is_one_setting(self, monkeypatch, least, shown):
        # The figure (live_portfolio) and its card's tooltip read one setting
        monkeypatch.setattr(config, "LIVE_RATIO_MIN_WHOLE_DAYS", least)
        stats = _view().periods[0]
        assert stats.whole_days == 9
        assert (stats.sharpe is not None, stats.sortino is not None) == (shown, shown)
        title = live_dashboard._cards(stats)["sharpe"]["title"]
        if shown:
            assert title.endswith("; over 9 whole days")
        else:
            assert title.endswith("needs at least 10 whole days (one daily close to the next); "
                                  "this period has 9 whole days")

    def test_a_figure_that_rounds_to_zero_is_not_colored(self):
        cards = live_dashboard._cards(_stats(total_return=-0.00004, pnl=0.004, sharpe=-0.001))
        assert (cards["return"]["text"], cards["return"]["tone"]) == ("0.0%", "flat")
        assert (cards["pnl"]["text"], cards["pnl"]["tone"]) == ("$0.00", "flat")
        assert (cards["sharpe"]["text"], cards["sharpe"]["tone"]) == ("0.00", "flat")

    def test_the_trades_note(self):
        assert live_dashboard._trades_note(_stats()) == (
            "312 contract pairs in 41 purchases (247 still open) · Sharpe and Sortino use "
            "9 whole days")
        assert live_dashboard._trades_note(_stats(pairs=D(1), purchases=1, open_pairs=D(0),
                                                  whole_days=1)) == (
            "1 contract pair in 1 purchase (0 still open) · Sharpe and Sortino use 1 whole day")
        assert live_dashboard._trades_note(_stats(pairs=D(0), purchases=0, open_pairs=D(0))) == (
            "No bot purchase in this period · Sharpe and Sortino use 9 whole days")

    def test_cards_and_tones_per_period(self):
        out = payload(_view())
        assert [p["label"] for p in out["periods"]] == [
            label for label, _ in config.LIVE_DASHBOARD_PERIODS]
        for stats, period in zip(_view().periods, out["periods"], strict=True):
            assert period["cards"] == live_dashboard._cards(stats)
            assert period["trades"] == live_dashboard._trades_note(stats)
            assert period["range"] == [live_dashboard._chart_time(stats.first),
                                       live_dashboard._chart_time(stats.last)]
        # Every value gains 1% a moment, so every period's return is a gain
        assert out["periods"][0]["cards"]["return"]["tone"] == "up"
        # A 1M period over a 10-day history is the whole history
        assert out["periods"][-1]["note"] == out["periods"][0]["note"] == (
            "Sep 28 – Oct 8, 2026 (10 days)")


class TestFormats:
    def test_money_and_prices(self):
        assert live_dashboard._money(D("1234.5")) == "$1,234.50"
        assert live_dashboard._signed_money(-0.004) == "$0.00"
        assert live_dashboard._signed_money(1234.567) == "+$1,234.57"
        assert live_dashboard._price(D("0.45")) == "$0.45"
        assert live_dashboard._price(D("0.455")) == "$0.455"
        assert live_dashboard._price(D("0.0001")) == "$0.0001"
        assert live_dashboard._price(D("1")) == "$1.00"
        assert live_dashboard._price(None) == "—"

    def test_range_notes(self):
        note = live_dashboard._range_note
        assert note(START, READ) == "Sep 28 – Oct 8, 2026 (10 days)"
        assert note(READ - timedelta(hours=3), READ) == "Oct 8, 2026 (under a day)"
        new_year = datetime(2027, 1, 2, 12, tzinfo=UTC)
        assert note(datetime(2026, 12, 31, 12, tzinfo=UTC), new_year) == (
            "Dec 31, 2026 – Jan 2, 2027 (2 days)")
        assert note(READ - timedelta(days=1), READ) == "Oct 7 – Oct 8, 2026 (1 day)"

    def test_the_chart_reads_new_york_wall_time(self):
        # A daily close is midnight in New York, so it sits on the day's line
        close = live_portfolio.day_ends(START, READ)[0]
        assert live_dashboard._chart_time(close) == "2026-09-29 00:00:00"

    def test_the_read_time_is_this_computer_s(self):
        with _host_zone("America/Los_Angeles"):
            assert live_dashboard._read_text(READ) == "Read from Kalshi Oct 8, 01:23 PDT"
            assert payload(_view())["stale"] == "Still showing the read from Oct 8, 01:23 PDT"

    def test_a_wait_in_words(self):
        assert live_dashboard._duration_words(300) == "5 minutes"
        assert live_dashboard._duration_words(30) == "30 seconds"
        assert live_dashboard._duration_words(1) == "1 second"


class TestArea:
    def test_bands_run_cash_first_and_other_bets_last_and_add_up(self):
        view = _view()
        area = payload(view)["area"]
        stacked = [t for t in area["data"] if "stackgroup" in t]
        assert [t["name"] for t in stacked] == [CASH, "Crypto", "Climate", OTHER_BETS]
        for i in range(len(view.history.times)):
            assert sum(t["y"][i] for t in stacked) == pytest.approx(view.history.total(i))
        assert all(t["stackgroup"] == "one" for t in stacked)
        # A 2-pixel gap in the surface color between bands; each band its own color
        assert {(t["line"]["color"], t["line"]["width"]) for t in stacked} == {
            (live_dashboard._SURFACE, 2)}
        assert [t["fillcolor"] for t in stacked] == ["#c3c2b7", "#2a78d6", "#eb6834", "#4a3aa7"]
        # The hover also shows the account's total, from a line drawn in no width
        total = area["data"][-1]
        assert area["data"][:-1] == stacked and total["name"] == "Total"
        assert total["y"] == pytest.approx([view.history.total(i)
                                            for i in range(len(view.history.times))])
        assert (total["showlegend"], total["line"]["width"], "stackgroup" in total) == (
            False, 0, False)
        layout = area["layout"]
        assert layout["hovermode"] == "x unified"
        # One set of range controls: the period buttons above, the slider below
        assert "rangeselector" not in layout["xaxis"]
        assert layout["xaxis"]["rangeslider"]["visible"] is True
        assert layout["yaxis"]["tickformat"] == ",.2~f"
        # The legend lists the bands as they stack, the top band first
        assert layout["legend"]["traceorder"] == "reversed"
        assert area["data"][0]["x"][0] == live_dashboard._chart_time(START)

    def test_eight_categories_fold_into_six_and_more(self):
        groups = {f"Cat{i}": 80.0 - 5 * i for i in range(8)}
        groups[OTHER_BETS] = 3.0
        view = _view(groups)
        assert view.group_order[:8] == tuple(f"Cat{i}" for i in range(8))
        out = payload(view)
        stacked = out["area"]["data"][:-1]
        names = [t["name"] for t in stacked]
        assert names == [CASH, *(f"Cat{i}" for i in range(6)), MORE_CATEGORIES, OTHER_BETS]
        more = stacked[7]["y"]
        assert more == pytest.approx([view.history.value["Cat6"][i] + view.history.value["Cat7"][i]
                                      for i in range(len(view.history.times))])
        for i in range(len(view.history.times)):
            assert sum(t["y"][i] for t in stacked) == pytest.approx(view.history.total(i))
        # The band names the categories it holds when hovered
        assert stacked[7]["customdata"] == ["Cat6, Cat7"] * len(view.history.times)
        assert stacked[7]["hovertemplate"] == "%{y:$,.2f} (%{customdata})"
        assert "customdata" not in stacked[6]
        # The folded returns: profit and money put in added up, then divided
        stats = view.periods[0]
        folded = [r for r in stats.groups if r.group in ("Cat6", "Cat7")]
        pnl, put_in = sum(r.pnl for r in folded), sum(r.put_in for r in folded)
        rows = live_dashboard._group_rows(stats.groups, live_dashboard._bands(view))
        assert [r.label for r in rows] == [*(f"Cat{i}" for i in range(6)), MORE_CATEGORIES,
                                           OTHER_BETS]
        assert rows[6].pnl == pytest.approx(pnl)
        assert rows[6].put_in == pytest.approx(put_in)
        assert rows[6].ret == pytest.approx(pnl / put_in)
        assert rows[6].members == ("Cat6", "Cat7")
        bars = out["periods"][0]["groups"]["data"][0]
        assert bars["y"] == [r.label for r in rows]
        assert bars["text"][6] == (f"{live_dashboard._pct(pnl / put_in)} · "
                                   f"{live_dashboard._signed_money(pnl)}")
        assert bars["customdata"][6] == " (Cat6, Cat7)" and bars["customdata"][5] == ""
        assert bars["hovertemplate"] == "%{y}: %{text}%{customdata}<extra></extra>"
        assert bars["marker"]["cornerradius"] == 4
        row = out["periods"][0]["group_rows"][6]
        assert row["cells"] == [
            MORE_CATEGORIES, live_dashboard._pct(pnl / put_in),
            live_dashboard._signed_money(pnl), live_dashboard._money(put_in)]
        assert row["tip"] == "Cat6, Cat7"
        assert out["periods"][0]["group_rows"][5]["tip"] is None

    def test_a_lone_category_past_the_six_keeps_its_name(self):
        groups = {f"Cat{i}": 80.0 - 5 * i for i in range(7)}
        view = _view(groups)
        out = payload(view)
        stacked = out["area"]["data"][:-1]
        assert [t["name"] for t in stacked] == [CASH, *(f"Cat{i}" for i in range(7))]
        assert stacked[7]["fillcolor"] == live_dashboard._MORE_COLOR
        rows = out["periods"][0]["group_rows"]
        assert [r["cells"][0] for r in rows] == [f"Cat{i}" for i in range(7)]
        assert rows[6]["swatch"] == [0, live_dashboard._MORE_COLOR] and rows[6]["tip"] is None
        swatches = {r["cells"][6]: r["swatch"][1] for r in out["holdings"]["rows"]}
        assert swatches["Cat6"] == live_dashboard._MORE_COLOR
        assert swatches["Cat0"] == live_dashboard._PALETTE[0]

    def test_a_group_the_order_lacks_takes_a_free_slot(self):
        # live_portfolio's group order lists every group; one it lacked would
        # follow the listed ones, by name while a slot is free
        view = _view()
        view = dataclasses.replace(view, group_order=("Crypto", OTHER_BETS))
        bands = live_dashboard._bands(view)
        assert (bands.named, bands.folded) == (("Crypto", "Climate"), ())
        assert bands.colors["Climate"] == live_dashboard._PALETTE[1]

    def test_returns_are_folded_from_profit_and_money_put_in_not_averaged(self):
        bands = live_dashboard._Bands(("A",), ("B", "C"), {"A": "#2a78d6", "B": "#898781",
                                                           "C": "#898781",
                                                           OTHER_BETS: "#4a3aa7"})
        rows = live_dashboard._group_rows(
            (GroupReturn("A", 1.0, 10.0, 0.1), GroupReturn("B", 9.0, 10.0, 0.9),
             GroupReturn("C", -1.0, 90.0, -1 / 90), GroupReturn(OTHER_BETS, 0.5, 0.0, None)),
            bands)
        more = rows[1]
        assert (more.label, more.pnl, more.put_in, more.ret) == (MORE_CATEGORIES, 8.0, 100.0,
                                                                  0.08)
        assert rows[2].ret is None
        bars = live_dashboard._bars(rows)
        assert bars["data"][0]["text"][2] == "— · +$0.50"
        assert bars["data"][0]["x"] == pytest.approx([10.0, 8.0, 0.0])
        # Room past the longest bar for its label; a side with no bar barely any
        assert bars["layout"]["xaxis"]["range"] == pytest.approx([-0.5, 15.0])
        losing = live_dashboard._bars([dataclasses.replace(rows[0], ret=-0.4)])
        assert losing["layout"]["xaxis"]["range"] == pytest.approx([-60.0, 2.0])

    def test_a_category_keeps_its_color_across_periods(self):
        out = payload(_view())
        colors = [dict(zip(p["groups"]["data"][0]["y"], p["groups"]["data"][0]["marker"]["color"],
                           strict=True)) for p in out["periods"]]
        assert all(c == colors[0] for c in colors)
        assert colors[0] == {"Crypto": "#2a78d6", "Climate": "#eb6834", OTHER_BETS: "#4a3aa7"}
        area = {t["name"]: t["fillcolor"] for t in out["area"]["data"] if "fillcolor" in t}
        assert all(area[name] == color for name, color in colors[0].items())
        swatches = {row["cells"][6]: row["swatch"][1] for row in out["holdings"]["rows"]}
        assert swatches == colors[0]

    def test_a_later_bigger_category_does_not_take_an_earlier_ones_color(self):
        # Crypto first, then Climate; then Politics arrives worth far more than
        # both, and later still a seventh category, past the six slots: every
        # earlier category keeps its color, and each new one takes the next
        # slot, then the gray
        def colors(groups):
            out = payload(_view(groups))
            return {t["name"]: t["fillcolor"] for t in out["area"]["data"] if "fillcolor" in t}

        before = colors({"Crypto": 20.0, "Climate": 10.0, OTHER_BETS: 5.0})
        after = colors({"Crypto": 20.0, "Climate": 10.0, "Politics": 900.0, OTHER_BETS: 5.0})
        assert before == {CASH: live_dashboard._CASH_COLOR, "Crypto": live_dashboard._PALETTE[0],
                          "Climate": live_dashboard._PALETTE[1],
                          OTHER_BETS: live_dashboard._OTHER_BETS_COLOR}
        assert after == {**before, "Politics": live_dashboard._PALETTE[2]}
        six = {f"Cat{i}": 1.0 + i for i in range(6)}
        full = colors({**six, "Huge": 1000.0})
        assert {g: full[g] for g in six} == {f"Cat{i}": live_dashboard._PALETTE[i]
                                             for i in range(6)}
        assert full["Huge"] == live_dashboard._MORE_COLOR

    def test_a_group_held_but_not_in_the_history_still_has_a_color(self):
        view = _view(holdings=(_holding("T0", "Crypto"), _holding("N", "Newcomer")))
        out = payload(view)
        assert "Newcomer" in view.group_order
        assert "Newcomer" not in [t["name"] for t in out["area"]["data"]]
        row = next(r for r in out["holdings"]["rows"] if r["cells"][6] == "Newcomer")
        assert row["swatch"][1] in live_dashboard._PALETTE


class TestHoldings:
    def test_rows_cash_and_totals(self):
        held = (_holding("A1", "Crypto", "yes", "13", "0.45", "3.90"),
                _holding("B1", "Crypto", "no", "13", "0.40", "5.20"),
                _holding("X", OTHER_BETS, "yes", "5", None, "1.00"))
        view = _view(holdings=held)
        table = payload(view)["holdings"]
        assert [r["cells"] for r in table["rows"]] == [
            ["Will A1?", "YES", "13", "$0.45", "$5.85", "$3.90", "Crypto"],
            ["Will B1?", "NO", "13", "$0.40", "$5.20", "$5.20", "Crypto"],
            ["Will X?", "YES", "5", "—", "$1.00", "$1.00", OTHER_BETS]]
        assert [r["tip"] for r in table["rows"]] == ["A1", "B1", "X"]
        # The holdings' value beside their cost; cash and the total have no cost
        assert [r["cells"] for r in table["foot"]] == [
            ["Holdings", "", "", "", "$12.05", "$10.10", ""],
            [CASH, "", "", "", "$90.91", "", ""],
            ["Total", "", "", "", "$102.96", "", ""]]
        assert table["empty"] is None

    def test_the_outcome_label_tells_two_rungs_apart(self):
        held = (_holding("Q-26NOV01", "Tech", "yes", subtitle="Before Nov 1, 2026"),
                _holding("Q-26DEC01", "Tech", "no", subtitle="Before Dec 1, 2026"),
                _holding("P", "Tech", subtitle="Will P?"))
        table = payload(_view({"Tech": 10.0}, holdings=held))["holdings"]
        assert [r["cells"][0] for r in table["rows"]] == [
            "Will Q-26NOV01? — Before Nov 1, 2026", "Will Q-26DEC01? — Before Dec 1, 2026",
            "Will P?"]

    def test_nothing_held(self):
        table = payload(_view(holdings=()))["holdings"]
        assert table["rows"] == [] and table["empty"] == "Nothing is held now."


class TestNotes:
    def test_the_notes(self):
        notes = payload(_view())["notes"]
        assert notes[0].startswith("Positions are valued at the midpoint of the best bid and "
                                   "ask; an empty side counts at 0 or 1; with both sides "
                                   "empty, the last trade.")
        assert notes[1] == ("Cash rebuilt from Kalshi's records matches 22 of 22 runs' "
                            "logged balances (largest gap $0.0075).")
        assert notes[2] == ("Holdings at the midpoint: $13.50. Kalshi's own value of the "
                            "positions: $12.25.")
        assert notes[3].startswith("Sharpe and Sortino subtract 0%")
        assert notes[4] == (
            "The chart's times are New York time. It has a point at each daily close (midnight "
            "there, where Kalshi's days close) and now, joined by straight lines, so a trade made "
            "during a day shows as a slope up to that day's close.")

    def test_unread_positions_value_and_no_run_checked(self):
        notes = payload(_view(kalshi_positions_value=None,
                              cash_check=CashCheck(0, 0, D(0), ())))["notes"]
        assert notes[1] == "No run's logged balance to check the rebuilt cash against yet."
        assert notes[2].endswith("Kalshi's own value of the positions could not be read.")

    def test_the_risk_free_note(self):
        note = live_dashboard._risk_free_note
        api = treasury.RiskFreeRates(((datetime(2026, 10, 6).date(), 0.04071),),
                                     treasury.SOURCE_API, READ)
        assert note(api) == ("Sharpe and Sortino subtract the 8-week Treasury bill's yield "
                             "(latest auction 4.07%, Oct 6, 2026) on the share of the account "
                             "held in positions.")
        cached = dataclasses.replace(api, source=treasury.SOURCE_CACHE)
        assert note(cached).endswith("an earlier saved download is used.")
        none = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        assert "could not be downloaded" in note(none)
        assert "not been downloaded yet" in note(None)


class TestPayload:
    def test_it_is_plain_json(self):
        view = _view(warnings=("A1: Kalshi holds 12 YES, but the fills and payouts add up to "
                               "13 YES",))
        out = payload(view)
        assert json.loads(json.dumps(out, allow_nan=False)) == out
        assert out["warnings"] == list(view.warnings)
        assert out["empty"] is None
        assert out["read_at"] == "2026-10-08T08:23:00+00:00"
        assert out["stale"] == live_dashboard._stale_text(READ)

    def test_the_area_s_figures_as_a_table(self):
        view = _view()
        table = payload(view)["area_table"]
        assert table["head"] == ["Time (New York)", CASH, "Crypto", "Climate", OTHER_BETS,
                                 "Total"]
        assert len(table["rows"]) == len(view.history.times)
        first, last = table["rows"][0]["cells"], table["rows"][-1]["cells"]
        assert first[0] == "Sep 28, 2026 02:30" and last[0] == "Oct 8, 2026 04:23"
        assert first[1:] == ["$100.00", "$20.00", "$10.00", "$5.00", "$135.00"]
        assert last[-1] == live_dashboard._money(view.history.total(len(view.history.times) - 1))

    def test_no_bot_trade_yet_shows_the_holdings_and_says_so(self):
        view = _view({OTHER_BETS: 5.0}, history=False)
        out = payload(view)
        assert out["periods"] == [] and out["area"] is None and out["area_table"] is None
        assert out["empty"].startswith("The bot has made no live trade yet.")
        assert [r["cells"][0] for r in out["holdings"]["rows"]] == ["Will T0?"]
        assert not any("New York time" in note for note in out["notes"])
        assert json.loads(json.dumps(out, allow_nan=False)) == out

    def test_the_palette_covers_every_named_band(self):
        assert config.LIVE_DASHBOARD_MAX_CATEGORY_BANDS <= len(live_dashboard._PALETTE)


# ---- the tab page's server ----------------------------------------------------------

def _get(app, target: str, *, host: str | None = PAGE_HOST, method: str = "GET",
         headers: dict | None = None) -> _Response:
    """Send one request straight to an application."""
    sent = {name.lower(): value for name, value in (headers or {}).items()}
    if host is not None:
        sent["host"] = host
    return app.handle(_Request(method, target, host, sent))


class _Reads:
    """A read of the account that records its calls."""

    def __init__(self, answer='{"status": "ok"}', *, fail=None, gate=None):
        self.calls, self.answer, self.fail, self.gate = 0, answer, fail, gate
        self.started = threading.Event()

    def __call__(self):
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.fail is not None:
            raise self.fail
        return self.answer


class TestPageApp:
    @pytest.mark.parametrize("host", ["evil.com", "127.0.0.1:8765", BACKTEST_HOST,
                                      "127.0.0.1", None])
    def test_a_wrong_host_is_refused(self, host):
        reads = _Reads()
        response = _get(PageApp(build=reads), "/api/live", host=host, headers=LIVE_HEADER)
        assert response.status == 403 and reads.calls == 0

    def test_localhost_is_this_server_too(self):
        response = _get(PageApp(), "/", host=f"LOCALHOST:{config.LIVE_DASHBOARD_PORT}")
        assert response.status == 200

    def test_only_get_and_head(self):
        response = _get(PageApp(), "/", method="POST")
        assert response.status == 405
        assert ("Allow", "GET, HEAD") in response.headers

    def test_the_page(self):
        response = _get(PageApp(), "/")
        assert (response.status, response.content_type) == (200, "text/html; charset=utf-8")
        assert response.body == live_dashboard._SHELL
        headers = dict(response.headers)
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Cache-Control"] == "no-store"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert not any(name.lower().startswith("access-control") for name, _ in response.headers)
        tag = re.search(r'<script src="([^"]+)" integrity="([^"]+)" crossorigin="anonymous">',
                        response.body)
        assert tag.groups() == (config.LIVE_DASHBOARD_PLOTLY_URL,
                                config.LIVE_DASHBOARD_PLOTLY_SRI)

    def test_the_csp_hash_is_the_script_s_own(self):
        page = live_dashboard._SHELL
        [script] = re.findall(r"<script>(.*?)</script>", page, re.S)
        digest = b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode()
        csp = dict(_get(PageApp(), "/").headers)["Content-Security-Policy"]
        assert f"'sha256-{digest}'" in csp
        assert f"script-src {config.LIVE_DASHBOARD_PLOTLY_URL} 'sha256-{digest}';" in csp
        assert "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp
        assert f"frame-src http://127.0.0.1:{config.LIVE_BACKTEST_PORT};" in csp
        # The backtest address and the request header are written into the script
        assert json.dumps(live_dashboard.BACKTEST_URL) in script
        assert json.dumps(config.LIVE_DASHBOARD_REQUEST_HEADER) in script

    def test_the_page_has_its_tabs_buttons_and_frame(self):
        page = live_dashboard._SHELL
        assert re.search(r'id="tab-live" role="tab" aria-selected="true"', page)
        assert re.search(r'id="tab-backtest" role="tab" aria-selected="false"', page)
        for i, (label, _) in enumerate(config.LIVE_DASHBOARD_PERIODS):
            pressed = "true" if i == 0 else "false"
            assert f'id="period-{i}" aria-pressed="{pressed}">{label}</button>' in page
        for key, label in live_dashboard.CARD_LABELS:
            assert f'>{label}</div><div class="card-value flat" id="card-{key}">' in page
        frame = re.search(r'<iframe id="backtest-frame" title="Backtest dashboard" '
                          r'sandbox="([^"]+)"></iframe>', page)
        # Scripts in its own origin, the defaults server's pages in windows of
        # their own and a chart's picture saved; never the tab page navigated
        assert frame.group(1).split() == ["allow-scripts", "allow-same-origin", "allow-popups",
                                          "allow-popups-to-escape-sandbox", "allow-downloads"]
        assert "allow-top-navigation" not in page
        # The warnings come before every figure they may qualify
        assert page.index('id="live-warnings"') < page.index('id="live-stats"')
        assert ('<details id="live-area-details"><summary>The chart\'s figures</summary>'
                in page)

    def test_health_names_this_checkout_and_its_code(self):
        response = _get(PageApp(), "/health")
        assert response.status == 200
        assert json.loads(response.body) == {
            "app": "kalshi-live-dashboard", "project_root": str(config.PROJECT_ROOT.resolve()),
            "code": live_dashboard._LOADED_CODE}
        assert live_dashboard._LOADED_CODE == live_dashboard._code_fingerprint()

    def test_the_account_needs_the_page_s_own_header(self):
        reads = _Reads()
        app = PageApp(build=reads)
        assert _get(app, "/api/live").status == 403
        wrong = {config.LIVE_DASHBOARD_REQUEST_HEADER: "0"}
        assert _get(app, "/api/live", headers=wrong).status == 403
        response = _get(app, "/api/live", headers={**LIVE_HEADER, "Sec-Fetch-Site": "cross-site"})
        assert response.status == 403
        assert json.loads(response.body) == {
            "error": "Only this dashboard's own page may read the account."}
        assert _get(app, "/api/live", headers={**LIVE_HEADER, "Sec-Fetch-Site": "same-site"}
                    ).status == 403
        assert _get(app, "/api/live", method="HEAD", headers=LIVE_HEADER).status == 405
        assert reads.calls == 0
        ok = _get(app, "/api/live", headers={**LIVE_HEADER, "Sec-Fetch-Site": "same-origin"})
        assert (ok.status, ok.body, reads.calls) == (200, '{"status": "ok"}', 1)
        assert ok.content_type == "application/json; charset=utf-8"

    def test_five_requests_at_once_make_one_read(self):
        gate = threading.Event()
        reads = _Reads(gate=gate)
        app = PageApp(build=reads)
        answers = []

        def ask():
            answers.append(_get(app, "/api/live", headers=LIVE_HEADER))

        threads = [threading.Thread(target=ask) for _ in range(5)]
        threads[0].start()
        assert reads.started.wait(5)
        for thread in threads[1:]:
            thread.start()
        deadline = time.monotonic() + 5
        while app._flight.waiting < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app._flight.waiting == 4
        gate.set()
        for thread in threads:
            thread.join(5)
        assert reads.calls == 1
        assert [(a.status, a.body) for a in answers] == [(200, '{"status": "ok"}')] * 5
        # A request after the read finished reads the account afresh
        assert _get(app, "/api/live", headers=LIVE_HEADER).status == 200
        assert reads.calls == 2

    @pytest.mark.parametrize("site", [None, "none", "same-origin", "same-site"])
    def test_the_page_opens_from_this_computer(self, site):
        headers = {} if site is None else {"Sec-Fetch-Site": site}
        assert _get(PageApp(), "/", headers=headers).body == live_dashboard._SHELL

    @pytest.mark.parametrize("method", ["GET", "HEAD"])
    def test_another_site_sending_the_browser_to_the_page_gets_no_script(self, method):
        response = _get(PageApp(), "/", method=method, headers={"Sec-Fetch-Site": "cross-site"})
        assert response.status == 403 and "<script" not in response.body
        assert response.body == ("This dashboard opens only from this computer: type "
                                 f"http://127.0.0.1:{config.LIVE_DASHBOARD_PORT}/ in the address "
                                 "bar, or run ./start_dashboard.sh.\n")
        given_twice = PageApp().handle(_Request("GET", "/", PAGE_HOST, {"host": PAGE_HOST},
                                                frozenset({"sec-fetch-site"})))
        assert given_twice.status == 403

    def test_a_sec_fetch_site_given_twice_reads_nothing(self):
        reads = _Reads()
        app = PageApp(build=reads)
        request = _Request("GET", "/api/live", PAGE_HOST,
                           {"host": PAGE_HOST, config.LIVE_DASHBOARD_REQUEST_HEADER.lower(): "1"},
                           frozenset({"sec-fetch-site"}))
        assert app.handle(request).status == 403 and reads.calls == 0

    @pytest.mark.parametrize("target", ["http://evil.com/api/live", "evil", "*"])
    def test_only_a_path_is_answered(self, target):
        reads = _Reads()
        response = _get(PageApp(build=reads), target, headers=LIVE_HEADER)
        assert response.status == 400 and reads.calls == 0

    def test_a_failure_on_the_account_data_is_json(self):
        app = PageApp()
        failed = app.failed(_Request("GET", "/api/live?x=1", PAGE_HOST))
        assert failed.status == 500 and failed.content_type == "application/json; charset=utf-8"
        assert json.loads(failed.body)["error"].startswith("Something went wrong")
        assert app.failed(_Request("GET", "/", PAGE_HOST)).content_type == (
            "text/plain; charset=utf-8")
        assert app.failed().status == 500

    def test_a_request_that_waits_too_long_gets_503(self, monkeypatch):
        monkeypatch.setattr(config, "LIVE_DASHBOARD_BUILD_WAIT_SECONDS", 0.05)
        gate = threading.Event()
        reads = _Reads(gate=gate)
        app = PageApp(build=reads)
        first = []
        thread = threading.Thread(target=lambda: first.append(
            _get(app, "/api/live", headers=LIVE_HEADER)))
        thread.start()
        assert reads.started.wait(5)
        late = _get(app, "/api/live", headers=LIVE_HEADER)
        assert late.status == 503
        assert json.loads(late.body)["error"] == (
            "Another read of the account is still going after 0.05 seconds: Kalshi may be slow "
            "to answer. Try Refresh again in a few minutes.")
        gate.set()
        thread.join(5)
        assert first[0].status == 200 and reads.calls == 1

    @pytest.mark.parametrize("fail, reason", [
        (ApiException(status=500, reason="Internal Server Error"), "HTTP 500 Internal Server Error"),
        (FileNotFoundError("secrets.json is missing\nsecond line"),
         "FileNotFoundError: secrets.json is missing")])
    def test_a_read_that_fails_is_a_502_with_one_line(self, fail, reason, caplog):
        app = PageApp(build=_Reads(fail=fail))
        with caplog.at_level(logging.WARNING):
            response = _get(app, "/api/live", headers=LIVE_HEADER)
        assert response.status == 502
        error = json.loads(response.body)["error"]
        assert error.startswith(f"Could not read the account from Kalshi: {reason}")
        assert "\n" not in error
        assert [r.levelno for r in caplog.records] == [logging.WARNING]

    def test_a_waiting_request_gets_the_read_s_error_too(self):
        gate = threading.Event()
        reads = _Reads(gate=gate, fail=ValueError("bad reply"))
        app = PageApp(build=reads)
        answers = []
        threads = [threading.Thread(target=lambda: answers.append(
            _get(app, "/api/live", headers=LIVE_HEADER))) for _ in range(2)]
        threads[0].start()
        assert reads.started.wait(5)
        threads[1].start()
        deadline = time.monotonic() + 5
        while app._flight.waiting < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        gate.set()
        for thread in threads:
            thread.join(5)
        assert [a.status for a in answers] == [502, 502] and reads.calls == 1

    def test_unknown_paths(self):
        assert _get(PageApp(), "/favicon.ico").status == 404
        assert _get(PageApp(), "/backtest/").status == 404

    def test_a_page_app_built_without_a_reader_refuses_to_read(self):
        response = _get(PageApp(), "/api/live", headers=LIVE_HEADER)
        assert response.status == 502


# ---- the backtest page's server --------------------------------------------------

@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """config.PROJECT_ROOT at a folder of its own, holding a backtest page and one chunk file."""
    root = tmp_path / "checkout"
    chunks = root / config.DASHBOARD_FILES_DIRNAME / "build1"
    chunks.mkdir(parents=True)
    (root / config.DASHBOARD_FILENAME).write_text("<html>the backtest</html>", encoding="utf-8")
    (chunks / "chunk-3.js").write_text("window.__dashChunk(1);", encoding="utf-8")
    (chunks / "notes.txt").write_text("not a chunk", encoding="utf-8")
    (root / "secrets.js").write_text("outside", encoding="utf-8")
    (tmp_path / "outside.js").write_text("outside", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    return root


def _backtest(target, **kwargs) -> _Response:
    """Ask the backtest page's application."""
    return _get(BacktestApp(), target, host=kwargs.pop("host", BACKTEST_HOST), **kwargs)


class TestBacktestApp:
    def test_the_address_without_its_slash_is_sent_on(self, checkout):
        response = _backtest("/backtest")
        assert response.status == 308
        assert ("Location", "/backtest/") in response.headers

    def test_the_page(self, checkout):
        response = _backtest("/backtest/")
        assert response.status == 200
        assert response.body == checkout / config.DASHBOARD_FILENAME
        assert response.content_type == "text/html; charset=utf-8"

    def test_no_backtest_yet(self, checkout):
        (checkout / config.DASHBOARD_FILENAME).unlink()
        response = _backtest("/backtest/")
        assert response.status == 200
        assert "There is no backtest dashboard yet" in response.body

    def test_a_chunk_file(self, checkout):
        response = _backtest(f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js")
        assert response.status == 200
        assert response.body == (checkout / config.DASHBOARD_FILES_DIRNAME / "build1"
                                 / "chunk-3.js").resolve()
        assert response.content_type == "text/javascript; charset=utf-8"

    def test_a_chunk_address_resolved_against_the_page_reaches_the_chunk(self, checkout):
        url = urljoin(f"http://{BACKTEST_HOST}/backtest/",
                      f"{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js")
        assert _backtest(urlsplit(url).path).status == 200

    def test_a_files_folder_that_is_a_link_is_served(self, checkout, tmp_path):
        # As in a worktree whose files folder links to the main checkout's
        real = tmp_path / "elsewhere"
        shutil.move(str(checkout / config.DASHBOARD_FILES_DIRNAME), str(real))
        (checkout / config.DASHBOARD_FILES_DIRNAME).symlink_to(real)
        assert _backtest(f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js"
                         ).status == 200

    @pytest.mark.parametrize("tail", [
        "../secrets.js", "%2e%2e/secrets.js", "build1/%2e%2e/%2e%2e/secrets.js",
        "build1%2f..%2f..%2fsecrets.js", "build1/chunk-3.js%00.js", "build1/a%5cb.js",
        "build1/notes.txt", "build1/outside.js", "build1/missing.js", "build1/chunk-3.js%0a",
        "build1/%7f.js"])
    def test_nothing_outside_the_chunk_files(self, checkout, tmp_path, tail):
        folder = checkout / config.DASHBOARD_FILES_DIRNAME / "build1"
        (folder / "outside.js").symlink_to(tmp_path / "outside.js")
        (folder / "a\\b.js").write_text("a backslash in its name", encoding="utf-8")
        (folder / "\x7f.js").write_text("a control character in its name", encoding="utf-8")
        response = _backtest(f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/{tail}")
        assert response.status == 404

    def test_other_paths_are_404(self, checkout):
        for target in ("/", "/backtest_dashboard.html", "/backtest/secrets.js", "/api/live"):
            assert _backtest(target).status == 404, target

    def test_an_unchanged_file_is_304(self, checkout):
        page = checkout / config.DASHBOARD_FILENAME
        os.utime(page, (1_700_000_000, 1_700_000_000))
        first = _backtest("/backtest/")
        modified = dict(first.headers)["Last-Modified"]
        assert modified == formatdate(1_700_000_000, usegmt=True)
        again = _backtest("/backtest/", headers={"If-Modified-Since": modified})
        assert (again.status, again.body) == (304, b"")
        older = _backtest("/backtest/", headers={
            "If-Modified-Since": formatdate(1_600_000_000, usegmt=True)})
        assert older.status == 200
        assert _backtest("/backtest/", headers={"If-Modified-Since": "not a date"}).status == 200
        tag = dict(first.headers)["ETag"]
        for match in (tag, f"W/{tag}", f'"other", {tag}', "*"):
            again = _backtest("/backtest/", headers={"If-None-Match": match})
            assert (again.status, dict(again.headers)["ETag"]) == (304, tag), match
        # If-None-Match decides when both are sent
        assert _backtest("/backtest/", headers={"If-None-Match": '"other"',
                                                "If-Modified-Since": modified}).status == 200

    def test_a_page_published_again_within_the_second_is_sent_again(self, checkout):
        page = checkout / config.DASHBOARD_FILENAME
        os.utime(page, (1_700_000_000, 1_700_000_000))
        first = dict(_backtest("/backtest/").headers)
        # A rebuild publishes by rename: a new file, the same whole second
        new = checkout / "new.html"
        new.write_text("<html>the new backtest, longer</html>", encoding="utf-8")
        os.utime(new, ns=(1_700_000_000_500_000_000, 1_700_000_000_500_000_000))
        os.replace(new, page)
        again = _backtest("/backtest/", headers={"If-None-Match": first["ETag"],
                                                 "If-Modified-Since": first["Last-Modified"]})
        assert again.status == 200 and dict(again.headers)["ETag"] != first["ETag"]

    def test_every_answer_may_be_framed_by_the_tab_page_only(self, checkout):
        for target in ("/backtest/", "/backtest", "/nothing",
                       f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js"):
            headers = dict(_backtest(target).headers)
            ancestors = headers["Content-Security-Policy"].removeprefix("frame-ancestors ")
            assert ancestors.split() == [f"http://127.0.0.1:{config.LIVE_DASHBOARD_PORT}",
                                         f"http://localhost:{config.LIVE_DASHBOARD_PORT}"]
            assert "X-Frame-Options" not in headers
            assert not any(n.lower().startswith("access-control") for n in headers)
            assert headers["Cross-Origin-Resource-Policy"] == "same-site"

    @pytest.mark.parametrize("host", ["evil.com", PAGE_HOST, "127.0.0.1:8765", None])
    def test_a_wrong_host_is_refused(self, checkout, host):
        assert _backtest("/backtest/", host=host).status == 403

    def test_only_get_and_head(self, checkout):
        assert _backtest("/backtest/", method="POST").status == 405
        assert _backtest("/backtest/", method="HEAD").status == 200

    def test_only_a_path_is_answered(self, checkout):
        target = f"http://evil/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js"
        assert _backtest(target).status == 400

    @pytest.mark.parametrize("site, mode, status", [
        ("cross-site", "no-cors", 403),          # another site's <script src>
        ("cross-site", "cors", 403),             # another site's fetch
        ("cross-site", None, 403),
        ("cross-site", "navigate", 200),         # localhost:8766 framing it, or a link
        ("same-site", "no-cors", 200),
        ("same-origin", "no-cors", 200),         # the page's own chunk files
        ("none", "navigate", 200),
        (None, None, 200)])
    def test_no_other_site_may_load_it(self, checkout, site, mode, status):
        headers = {name: value for name, value in (("Sec-Fetch-Site", site),
                                                    ("Sec-Fetch-Mode", mode)) if value}
        chunk = f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js"
        for target in (chunk, "/backtest/"):
            assert _backtest(target, headers=headers).status == status, target

    def test_a_sec_fetch_site_given_twice_loads_nothing(self, checkout):
        request = _Request("GET", "/backtest/", BACKTEST_HOST,
                           {"host": BACKTEST_HOST, "sec-fetch-mode": "navigate"},
                           frozenset({"sec-fetch-site"}))
        assert BacktestApp().handle(request).status == 403


# ---- the socket side --------------------------------------------------------------

@pytest.fixture
def serve():
    """
    Start a real server for an application on a loopback port the operating system chooses.

    Skipped where the sandbox refuses to bind a port. Every server is
    stopped and closed after the test.

    Yields:
        Callable: start(make) takes a function of the port returning the
            application, and returns the port.
    """
    started = []

    def start(make) -> int:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), live_dashboard._Handler)
        except PermissionError as exc:
            pytest.skip(f"this sandbox does not allow binding a local port: {exc}")
        server.app = make(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                  daemon=True)
        thread.start()
        started.append((server, thread))
        return server.server_address[1]

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _over_socket(port: int, method: str, target: str, headers: dict | None = None):
    """Send one request over the socket and read the whole answer."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, target, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class TestOverASocket:
    def test_get_and_head_of_the_page(self, serve):
        port = serve(lambda port: PageApp(port))
        status, headers, body = _over_socket(port, "GET", "/")
        assert status == 200 and body.decode("utf-8") == live_dashboard._SHELL
        assert int(headers["Content-Length"]) == len(body)
        status, headers, body = _over_socket(port, "HEAD", "/")
        assert status == 200 and body == b""
        assert int(headers["Content-Length"]) == len(live_dashboard._SHELL.encode("utf-8"))

    def test_another_method_is_405(self, serve):
        port = serve(lambda port: PageApp(port))
        status, headers, _ = _over_socket(port, "POST", "/api/live", LIVE_HEADER)
        assert status == 405 and headers["X-Frame-Options"] == "DENY"

    def test_a_streamed_file_is_sent_whole(self, serve, checkout, monkeypatch):
        monkeypatch.setattr(config, "LIVE_DASHBOARD_FILE_BLOCK_BYTES", 4096)
        page = checkout / config.DASHBOARD_FILENAME
        data = os.urandom(300_000)
        page.write_bytes(data)
        port = serve(lambda port: BacktestApp(port))
        status, headers, body = _over_socket(port, "GET", "/backtest/")
        assert status == 200 and body == data and int(headers["Content-Length"]) == len(data)
        status, headers, body = _over_socket(port, "HEAD", "/backtest/")
        assert status == 200 and body == b"" and int(headers["Content-Length"]) == len(data)

    def test_a_file_gone_before_it_is_opened_is_404(self, serve, tmp_path):
        gone = tmp_path / "gone.js"

        class Gone(BacktestApp):
            def handle(self, request):
                return _Response(200, gone, "text/javascript; charset=utf-8",
                                 live_dashboard._FRAMED_HEADERS)

        port = serve(lambda port: Gone(port))
        status, headers, body = _over_socket(port, "GET", "/backtest/x.js")
        assert status == 404 and body == b"No such file.\n"
        assert headers["Content-Security-Policy"].startswith("frame-ancestors")

    def test_a_header_given_twice_is_not_read(self, serve):
        reads = _Reads()
        port = serve(lambda port: PageApp(port, build=reads))
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.putrequest("GET", "/api/live")
            connection.putheader(config.LIVE_DASHBOARD_REQUEST_HEADER, "1")
            connection.putheader(config.LIVE_DASHBOARD_REQUEST_HEADER, "1")
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 403
            response.read()
        finally:
            connection.close()
        assert reads.calls == 0
        status, _, body = _over_socket(port, "GET", "/api/live", LIVE_HEADER)
        assert (status, body, reads.calls) == (200, b'{"status": "ok"}', 1)

    @pytest.mark.parametrize("sites", [("cross-site", "cross-site"),
                                       ("cross-site", "same-origin")])
    def test_a_sec_fetch_site_given_twice_reads_nothing(self, serve, sites):
        reads = _Reads()
        port = serve(lambda port: PageApp(port, build=reads))
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.putrequest("GET", "/api/live")
            connection.putheader(config.LIVE_DASHBOARD_REQUEST_HEADER, "1")
            connection.putheader("Sec-Fetch-Site", sites[0])
            connection.putheader("sec-fetch-site", sites[1])
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 403
            response.read()
        finally:
            connection.close()
        assert reads.calls == 0

    def test_a_failure_on_the_account_data_is_json(self, serve):
        class Broken(PageApp):
            def _live(self):
                raise RuntimeError("broken")

        port = serve(lambda port: Broken(port))
        status, headers, body = _over_socket(port, "GET", "/api/live", LIVE_HEADER)
        assert status == 500 and headers["Content-Type"] == "application/json; charset=utf-8"
        assert json.loads(body)["error"].startswith("Something went wrong answering")

    def test_a_file_swapped_after_its_check_is_not_sent(self, serve, checkout, tmp_path,
                                                        monkeypatch):
        folder = checkout / config.DASHBOARD_FILES_DIRNAME / "build1"
        (tmp_path / "secret.js").write_text("SECRET-OUTSIDE\n", encoding="utf-8")
        checked = live_dashboard._file

        def swap_after_the_check(path, content_type, request):
            response = checked(path, content_type, request)
            (folder / "chunk-3.js").unlink()
            (folder / "chunk-3.js").symlink_to(tmp_path / "secret.js")
            return response

        monkeypatch.setattr(live_dashboard, "_file", swap_after_the_check)
        port = serve(lambda port: BacktestApp(port))
        status, _, body = _over_socket(
            port, "GET", f"/backtest/{config.DASHBOARD_FILES_DIRNAME}/build1/chunk-3.js")
        assert status == 404 and b"SECRET" not in body

    def test_the_health_of_a_running_dashboard(self, serve):
        port = serve(lambda port: PageApp(port))
        running = live_dashboard._running_dashboard(f"http://127.0.0.1:{port}")
        assert running == live_dashboard._Running(str(config.PROJECT_ROOT.resolve()),
                                                  live_dashboard._LOADED_CODE)

    def test_another_program_s_health_is_none(self, serve):
        port = serve(lambda port: BacktestApp(port))
        assert live_dashboard._running_dashboard(f"http://127.0.0.1:{port}") is None


# ---- reading the account ----------------------------------------------------------

class _Transport:
    """A client's rest_client stand-in: records what each request was given."""

    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, body=None, post_params=None,
                _request_timeout=None):
        self.calls.append({"method": method, "url": url, "timeout": _request_timeout})
        return SimpleNamespace(status=200, data=b'{"ok": true}', reason="OK")


def _client() -> SimpleNamespace:
    """A stand-in KalshiClient: its transport and its signing."""
    auth = SimpleNamespace(create_auth_headers=lambda method, path: {"sig": "x"})
    return SimpleNamespace(rest_client=_Transport(), kalshi_auth=auth)


class TestReader:
    def test_every_request_of_this_server_s_client_has_its_timeouts(self):
        client = live_dashboard._with_timeouts(_client())
        # The signed read every Kalshi request of live_portfolio and historical goes through
        historical._signed_raw_get(client, "/trade-api/v2/portfolio/fills", limit=1)
        client.rest_client.request("GET", "https://x", headers={}, _request_timeout=999)
        assert [c["timeout"] for c in client.rest_client.calls] == [
            config.LIVE_KALSHI_TIMEOUT_SECONDS] * 2
        assert config.LIVE_KALSHI_TIMEOUT_SECONDS == (10, 60)
        # Only this client's transport is changed
        other = _client()
        other.rest_client.request("GET", "https://x")
        assert other.rest_client.calls[-1]["timeout"] is None

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "HEAD", "get"])
    def test_this_server_s_client_sends_nothing_but_gets(self, method):
        client = live_dashboard._with_timeouts(_client())
        with pytest.raises(RuntimeError, match="GET requests only"):
            client.rest_client.request(method, "https://x/trade-api/v2/portfolio/events/orders",
                                        headers={}, body={"x": 1})
        assert client.rest_client.calls == []

    def test_a_read(self, monkeypatch, caplog):
        built, seen = [], {}
        client = _client()
        monkeypatch.setattr(historical, "build_prod_live_client",
                            lambda: built.append(1) or client)
        monkeypatch.setattr(historical, "load_series_categories",
                            lambda c: {"KXCRYPTO": ("Crypto", ("BTC",))})
        view = _view()

        def build_live_view(c, *, risk_free, series_categories, trade_logs):
            seen.update(client=c, risk_free=risk_free, series=series_categories,
                        logs=trade_logs)
            return view

        monkeypatch.setattr(live_portfolio, "build_live_view", build_live_view)
        rates = live_dashboard._RiskFree()
        rates.current = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        rates.tried.set()
        reader = live_dashboard._LiveReader(rates)
        with caplog.at_level(logging.INFO):
            text = reader()
            reader()
        assert json.loads(text) == json.loads(json.dumps(payload(view)))
        assert built == [1]                       # built once, on the first read
        assert seen == {"client": client, "risk_free": rates.current,
                        "series": {"KXCRYPTO": ("Crypto", ("BTC",))},
                        "logs": live_portfolio.trade_log_paths()}
        lines = config.LIVE_PORTFOLIO_LOG_FILE.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert ("Read the account: cash $90.91, 3 holdings worth $13.50 at the midpoint "
                "(Kalshi: $12.25)") in caplog.text
        # The client was given this server's timeouts
        client.rest_client.request("GET", "https://x")
        assert client.rest_client.calls[-1]["timeout"] == config.LIVE_KALSHI_TIMEOUT_SECONDS

    def test_credentials_that_cannot_be_read_are_a_502_and_tried_again(self, monkeypatch):
        attempts = []

        def build():
            attempts.append(1)
            raise FileNotFoundError("secrets.json")

        monkeypatch.setattr(historical, "build_prod_live_client", build)
        app = PageApp(build=live_dashboard._LiveReader(live_dashboard._RiskFree()))
        for _ in range(2):
            response = _get(app, "/api/live", headers=LIVE_HEADER)
            assert response.status == 502
            assert json.loads(response.body)["error"] == (
                "Could not read the account from Kalshi: FileNotFoundError: secrets.json")
        assert attempts == [1, 1]
        # The page itself still comes up
        assert _get(app, "/").status == 200

    def test_the_t_bill_yields_are_kept_fresh(self, monkeypatch):
        rates = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        calls = []
        monkeypatch.setattr(treasury, "load_risk_free_rates", lambda: calls.append(1) or rates)
        monkeypatch.setattr(config, "LIVE_RISK_FREE_REFRESH_SECONDS", 0.01)
        holder, stop = live_dashboard._RiskFree(), threading.Event()
        assert holder.current is None
        thread = threading.Thread(target=holder.keep_fresh, args=(stop,), daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(5)
        assert len(calls) >= 3 and holder.current is rates and not thread.is_alive()
        assert holder.tried.is_set()

    def test_the_first_read_waits_for_the_first_download(self, monkeypatch):
        rates = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
        gate = threading.Event()

        def download():
            assert gate.wait(10)
            return rates

        monkeypatch.setattr(treasury, "load_risk_free_rates", download)
        monkeypatch.setattr(config, "LIVE_RISK_FREE_REFRESH_SECONDS", 60)
        holder, stop = live_dashboard._RiskFree(), threading.Event()
        thread = threading.Thread(target=holder.keep_fresh, args=(stop,), daemon=True)
        thread.start()
        got = []
        reader = threading.Thread(target=lambda: got.append(holder.first()))
        reader.start()
        time.sleep(0.2)
        assert got == []                          # still waiting for the download
        gate.set()
        reader.join(5)
        stop.set()
        thread.join(5)
        assert got == [rates]

    def test_a_download_that_does_not_land_in_time_is_not_waited_for(self, monkeypatch):
        monkeypatch.setattr(config, "LIVE_RISK_FREE_FIRST_WAIT_SECONDS", 0.05)
        holder = live_dashboard._RiskFree()
        started = time.monotonic()
        assert holder.first() is None
        assert time.monotonic() - started < 5


# ---- main ---------------------------------------------------------------------------

def _run_main(argv: list[str]) -> str:
    """
    Run live_dashboard.main with the root logger's handlers set aside.

    main configures logging with basicConfig, which does nothing while the
    root logger has handlers (pytest adds its own), so they are set aside for
    the call and put back after it, whatever it raises.

    Returns:
        str: The log file's text ("" when it was never written).
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        live_dashboard.main(argv)
    finally:
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
    log = Path(config.LIVE_DASHBOARD_LOG_FILE)
    return log.read_text(encoding="utf-8") if log.exists() else ""


@pytest.fixture
def run_main(monkeypatch):
    """
    Run main with the servers, the browser, the client builder and the T-bill download replaced.

    Yields:
        dict: "servers" (each fake server, by port), "busy" (ports whose bind
            fails as taken), "listening" (ports another program already
            answers on), "running" (what the stand-in for _running_dashboard
            answers), "opened" (each address opened), "rates" (set when the
            yields were downloaded) and "interrupt" (whether serve_forever
            raises KeyboardInterrupt, True by default).
    """
    state = {"servers": {}, "busy": set(), "listening": set(), "running": None, "opened": [],
             "rates": threading.Event(), "interrupt": True, "asked": []}

    class FakeServer:
        """Records how it was made and used; serves nothing."""

        def __init__(self, address, handler):
            if address[1] in state["busy"]:
                raise OSError(errno.EADDRINUSE, "Address already in use")
            self.address, self.handler = address, handler
            self.served = self.shut = self.closed = False
            state["servers"][address[1]] = self

        def serve_forever(self, poll_interval=0.5):
            self.served = True
            if self is state["servers"].get(config.LIVE_DASHBOARD_PORT) and state["interrupt"]:
                raise KeyboardInterrupt

        def shutdown(self):
            self.shut = True

        def server_close(self):
            self.closed = True

    def running_dashboard(base):
        state["asked"].append(base)
        return state["running"]

    def no_client():
        raise AssertionError("main built a Kalshi client")

    rates = treasury.RiskFreeRates((), treasury.SOURCE_UNAVAILABLE, None)
    monkeypatch.setattr(live_dashboard, "_Server", FakeServer)
    monkeypatch.setattr(live_dashboard, "_port_answers",
                        lambda host, port: port in state["listening"])
    monkeypatch.setattr(live_dashboard, "_running_dashboard", running_dashboard)
    monkeypatch.setattr(live_dashboard.webbrowser, "open", state["opened"].append)
    monkeypatch.setattr(historical, "build_prod_live_client", no_client)
    monkeypatch.setattr(treasury, "load_risk_free_rates",
                        lambda: state["rates"].set() or rates)
    yield state


class TestMain:
    BASE = f"http://127.0.0.1:{config.LIVE_DASHBOARD_PORT}"

    def test_it_serves_both_tabs_until_ctrl_c(self, run_main):
        log = _run_main([])
        servers = run_main["servers"]
        assert set(servers) == {config.LIVE_DASHBOARD_PORT, config.LIVE_BACKTEST_PORT}
        page, backtest = servers[config.LIVE_DASHBOARD_PORT], servers[config.LIVE_BACKTEST_PORT]
        assert page.address == (config.LIVE_DASHBOARD_HOST, config.LIVE_DASHBOARD_PORT)
        assert isinstance(page.app, PageApp) and isinstance(backtest.app, BacktestApp)
        assert page.handler is backtest.handler is live_dashboard._Handler
        assert page.served and page.closed and backtest.shut and backtest.closed
        assert run_main["opened"] == [f"{self.BASE}/"]
        assert run_main["rates"].wait(5)
        assert f"Live dashboard at {self.BASE}/" in log and "Live dashboard stopped" in log
        assert run_main["asked"] == []

    def test_no_browser(self, run_main):
        _run_main(["--no-browser"])
        assert run_main["opened"] == []

    def test_it_turns_ctrl_c_back_on(self, run_main, monkeypatch):
        # start_dashboard.sh starts it in the background, where a job begins
        # with Ctrl-C ignored: main makes Ctrl-C stop it again
        calls = []
        monkeypatch.setattr(live_dashboard.signal, "signal",
                            lambda signum, handler: calls.append((signum, handler)))
        _run_main(["--no-browser"])
        assert calls == [(signal.SIGINT, signal.default_int_handler)]

    def test_this_checkout_s_running_dashboard_has_its_page_opened(self, run_main, capsys):
        run_main["busy"].add(config.LIVE_DASHBOARD_PORT)
        run_main["running"] = live_dashboard._Running(str(config.PROJECT_ROOT.resolve()),
                                                      live_dashboard._code_fingerprint())
        assert _run_main([]) == ""
        assert run_main["opened"] == [f"{self.BASE}/"]
        assert run_main["asked"] == [self.BASE]
        assert run_main["servers"] == {}
        assert "This checkout's live dashboard is already running" in capsys.readouterr().err
        assert not Path(config.LIVE_DASHBOARD_LOG_FILE).exists()

    @pytest.mark.parametrize("running, words", [
        (None, "is in use by another program"),
        ("older code", "stop it (Ctrl-C in its terminal) and start it again"),
        ("another checkout", "is served by the live dashboard of /somewhere/else")])
    def test_anything_else_on_the_port_is_refused(self, run_main, capsys, running, words):
        run_main["busy"].add(config.LIVE_DASHBOARD_PORT)
        root = str(config.PROJECT_ROOT.resolve())
        run_main["running"] = {
            None: None,
            "older code": live_dashboard._Running(root, "0" * 64),
            "another checkout": live_dashboard._Running("/somewhere/else",
                                                        live_dashboard._LOADED_CODE)}[running]
        with pytest.raises(SystemExit) as stopped:
            _run_main([])
        assert stopped.value.code == 2
        assert words in capsys.readouterr().err
        assert run_main["opened"] == [] and run_main["servers"] == {}
        assert not Path(config.LIVE_DASHBOARD_LOG_FILE).exists()

    def test_the_backtest_port_taken_is_refused(self, run_main, capsys):
        run_main["busy"].add(config.LIVE_BACKTEST_PORT)
        with pytest.raises(SystemExit) as stopped:
            _run_main([])
        assert stopped.value.code == 2
        assert f"port {config.LIVE_BACKTEST_PORT} is in use" in capsys.readouterr().err
        assert run_main["servers"][config.LIVE_DASHBOARD_PORT].closed
        assert not Path(config.LIVE_DASHBOARD_LOG_FILE).exists()

    def test_a_program_already_answering_on_the_page_port_is_asked_what_it_is(self, run_main,
                                                                             capsys):
        # On macOS binding 127.0.0.1 succeeds beside a program listening on
        # 0.0.0.0, so a port that accepts a connection counts as taken
        run_main["listening"].add(config.LIVE_DASHBOARD_PORT)
        with pytest.raises(SystemExit) as stopped:
            _run_main([])
        assert stopped.value.code == 2
        assert "is in use by another program" in capsys.readouterr().err
        assert run_main["asked"] == [self.BASE] and run_main["servers"] == {}

    def test_a_program_already_answering_on_the_backtest_port_is_refused(self, run_main,
                                                                        capsys):
        run_main["listening"].add(config.LIVE_BACKTEST_PORT)
        with pytest.raises(SystemExit) as stopped:
            _run_main([])
        assert stopped.value.code == 2
        assert f"port {config.LIVE_BACKTEST_PORT} is in use" in capsys.readouterr().err
        assert run_main["servers"] == {}

    def test_the_servers_let_a_burst_of_connections_wait(self):
        assert issubclass(live_dashboard._Server, ThreadingHTTPServer)
        assert live_dashboard._Server.request_queue_size == config.LIVE_DASHBOARD_LISTEN_BACKLOG
        assert config.LIVE_DASHBOARD_LISTEN_BACKLOG >= 64
        assert live_dashboard._Server.daemon_threads is True


@pytest.fixture
def guarded_bind(monkeypatch):
    """main's binds: real, but one that succeeds fails the test (the port should have been taken)."""
    real = ThreadingHTTPServer

    def bind(address, handler):
        server = real(address, handler)
        server.server_close()
        raise AssertionError(f"main bound {address} although the port was taken")

    monkeypatch.setattr(live_dashboard, "_Server", bind)
    monkeypatch.setattr(historical, "build_prod_live_client",
                        lambda: pytest.fail("main built a Kalshi client"))


class TestBusyPortOverASocket:
    def test_this_checkout_s_dashboard_is_reused(self, serve, guarded_bind, monkeypatch, capsys):
        port = serve(lambda port: PageApp(port))
        monkeypatch.setattr(config, "LIVE_DASHBOARD_PORT", port)
        opened = []
        monkeypatch.setattr(live_dashboard.webbrowser, "open", opened.append)
        _run_main([])
        assert opened == [f"http://127.0.0.1:{port}/"]
        assert not Path(config.LIVE_DASHBOARD_LOG_FILE).exists()

    def test_the_same_dashboard_running_older_code_is_refused(self, serve, guarded_bind,
                                                              monkeypatch, capsys):
        port = serve(lambda port: PageApp(port))
        monkeypatch.setattr(config, "LIVE_DASHBOARD_PORT", port)
        monkeypatch.setattr(live_dashboard, "_code_fingerprint", lambda: "f" * 64)
        with pytest.raises(SystemExit) as stopped:
            _run_main([])
        assert stopped.value.code == 2
        assert "and start it again" in capsys.readouterr().err

    def test_a_program_listening_on_every_address_is_not_taken_over(self, guarded_bind,
                                                                     monkeypatch, capsys):
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                other.bind(("0.0.0.0", 0))
            except PermissionError as exc:
                pytest.skip(f"this sandbox does not allow binding a local port: {exc}")
            other.listen(5)
            monkeypatch.setattr(config, "LIVE_DASHBOARD_PORT", other.getsockname()[1])
            monkeypatch.setattr(config, "LIVE_DASHBOARD_HEALTH_TIMEOUT_SECONDS", 0.5)
            with pytest.raises(SystemExit) as stopped:
                _run_main(["--no-browser"])
        finally:
            other.close()
        assert stopped.value.code == 2
        assert "is in use by another program" in capsys.readouterr().err


# ---- the page's script ----------------------------------------------------------------

_HARNESS = Path(__file__).parent / "js" / "live_harness.js"
_JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def _js_runtime() -> str | None:
    """node when installed (CI's runners have it), else macOS's JavaScriptCore shell; None when neither is."""
    node = shutil.which("node")
    if node:
        return node
    return str(_JSC) if _JSC.exists() else None


def _run_live_script(tmp_path: Path, answers: list, steps: list, *,
                     plotly: bool = True) -> dict:
    """
    Run the tab page's script under tests/js/live_harness.js.

    Args:
        tmp_path (Path): Where the assembled program is written.
        answers (list): The answers its requests get, in order (see the harness).
        steps (list): The harness's steps (["settle"], ["click", id], ["snap", name]).
        plotly (bool): Keyword-only. Whether the chart library loaded.

    Returns:
        dict: The snapshots the steps took, by name.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("no JavaScript runtime (node or jsc) to run the page script with")
    ids = sorted(set(re.findall(r'\bid="([^"]+)"', live_dashboard._SHELL)))
    program = "\n".join([
        _HARNESS.read_text(encoding="utf-8"),
        *(f"__IDS[{json.dumps(i)}] = true;" for i in ids),
        f"__ANSWERS = {json.dumps(answers)};",
        f"__NO_PLOTLY = {json.dumps(not plotly)};",
        "__install();",
        live_dashboard._SCRIPT,
        f"__step({json.dumps(steps)}, 0);",
    ])
    path = tmp_path / "live_run.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _cells(snap: dict, element: str) -> list[list[str]]:
    """A table body's cell texts, row by row."""
    return [[c["text"] for c in row["cells"]] for row in snap["el"][element]["rows"]]


class TestPageScript:
    def test_the_first_render(self, tmp_path):
        data = payload(_view(warnings=("A1: Kalshi holds 12 YES",)))
        snap = _run_live_script(tmp_path, [{"status": 200, "body": data}],
                                [["settle"], ["snap", "shown"]])["shown"]
        assert snap["fetches"] == [{"url": "/api/live", "cache": "no-store",
                                    "headers": LIVE_HEADER}]
        el = snap["el"]
        assert el["live-status"]["text"] == data["status"]
        assert el["live-status"]["className"] == "status"
        assert el["live-refresh"]["disabled"] is False
        first = data["periods"][0]
        for key, _ in live_dashboard.CARD_LABELS:
            card = first["cards"][key]
            assert (el[f"card-{key}"]["text"], el[f"card-{key}"]["className"],
                    el[f"card-{key}"]["title"]) == (card["text"], f"card-value {card['tone']}",
                                                    card["title"])
        assert el["live-range"]["text"] == first["note"]
        assert el["live-trades"]["text"] == first["trades"]
        assert [el[f"period-{i}"]["attrs"]["aria-pressed"]
                for i in range(len(data["periods"]))] == ["true", "false", "false", "false",
                                                          "false"]
        assert [(p["call"], p["id"]) for p in snap["plots"]] == [
            ("react", "live-area"), ("react", "live-groups"), ("relayout", "live-area")]
        assert snap["plots"][0]["data"] == data["area"]["data"]
        assert snap["plots"][1]["data"] == first["groups"]["data"]
        assert snap["plots"][2]["update"] == {"xaxis.range": first["range"]}
        assert snap["plots"][0]["config"] == {"responsive": True, "displaylogo": False}
        assert _cells(snap, "live-holdings") == [r["cells"] for r in data["holdings"]["rows"]]
        assert _cells(snap, "live-holdings-foot") == [r["cells"] for r in data["holdings"]["foot"]]
        first_row = el["live-holdings"]["rows"][0]["cells"]
        assert first_row[0]["title"] == data["holdings"]["rows"][0]["tip"]
        assert first_row[6]["border"] == f"4px solid {data['holdings']['rows'][0]['swatch'][1]}"
        assert _cells(snap, "live-group-rows") == [r["cells"] for r in first["group_rows"]]
        assert [r["text"] for r in el["live-notes"]["rows"]] == data["notes"]
        assert [r["text"] for r in el["live-warnings"]["rows"]] == data["warnings"]
        assert el["live-warnings"]["hidden"] is False
        assert el["live-empty"]["hidden"] is True and el["live-stats"]["hidden"] is False
        # The area's figures, as a table
        assert el["live-area-details"]["hidden"] is False
        assert [r["text"] for r in el["live-area-head"]["rows"]] == data["area_table"]["head"]
        assert _cells(snap, "live-area-rows") == [r["cells"] for r in data["area_table"]["rows"]]

    def test_a_period_click(self, tmp_path):
        view = _view()
        view = dataclasses.replace(view, periods=(
            view.periods[0], dataclasses.replace(view.periods[1], total_return=-0.05,
                                                 label="1Y"), *view.periods[2:]))
        data = payload(view)
        snap = _run_live_script(tmp_path, [{"status": 200, "body": data}],
                                [["settle"], ["snap", "shown"], ["click", "period-1"],
                                 ["snap", "chosen"]])["chosen"]
        chosen = data["periods"][1]
        assert snap["el"]["card-return"]["text"] == "-5.0%"
        assert snap["el"]["card-return"]["className"] == "card-value down"
        assert [snap["el"][f"period-{i}"]["attrs"]["aria-pressed"] for i in range(2)] == [
            "false", "true"]
        assert [(p["call"], p["id"]) for p in snap["plots"]] == [
            ("react", "live-groups"), ("relayout", "live-area")]
        assert snap["plots"][1]["update"] == {"xaxis.range": chosen["range"]}
        assert snap["fetches"] == []

    def test_the_backtest_tab_loads_its_page_once(self, tmp_path):
        data = payload(_view())
        snaps = _run_live_script(tmp_path, [{"status": 200, "body": data}], [
            ["settle"], ["snap", "shown"], ["click", "tab-backtest"], ["snap", "backtest"],
            ["click", "tab-live"], ["snap", "live"], ["click", "tab-backtest"],
            ["snap", "again"]])
        # Nothing loads the backtest page until its tab is chosen
        assert snaps["shown"]["srcSets"] == 0
        backtest = snaps["backtest"]
        assert backtest["el"]["backtest-frame"]["src"] == live_dashboard.BACKTEST_URL
        assert backtest["el"]["panel-live"]["hidden"] is True
        assert backtest["el"]["panel-backtest"]["hidden"] is False
        assert backtest["el"]["tab-backtest"]["attrs"]["aria-selected"] == "true"
        assert backtest["el"]["tab-live"]["attrs"]["aria-selected"] == "false"
        live = snaps["live"]
        assert [(p["call"], p["id"]) for p in live["plots"]] == [
            ("resize", "live-area"), ("resize", "live-groups")]
        assert live["el"]["panel-live"]["hidden"] is False
        assert snaps["again"]["srcSets"] == 1

    def test_an_error_shows_its_message_and_refresh_reads_again(self, tmp_path):
        data = payload(_view())
        error = "Could not read the account from Kalshi: HTTP 500 Internal Server Error"
        snaps = _run_live_script(tmp_path, [{"status": 502, "body": {"error": error}},
                                            {"status": 200, "body": data}],
                                 [["settle"], ["snap", "failed"], ["click", "live-refresh"],
                                  ["settle"], ["snap", "again"]])
        failed = snaps["failed"]
        assert failed["el"]["live-status"]["text"] == error
        assert failed["el"]["live-status"]["className"] == "status error"
        assert failed["el"]["live-refresh"]["disabled"] is False
        assert failed["plots"] == []
        again = snaps["again"]
        assert len(again["fetches"]) == 1 and again["fetches"][0]["headers"] == LIVE_HEADER
        assert again["el"]["live-status"]["text"] == data["status"]
        assert again["el"]["live-status"]["className"] == "status"

    def test_a_refresh_that_fails_says_which_read_is_still_shown(self, tmp_path):
        data = payload(_view())
        error = "Could not read the account from Kalshi: HTTP 500 Internal Server Error"
        snaps = _run_live_script(tmp_path, [{"status": 200, "body": data},
                                            {"status": 502, "body": {"error": error}}],
                                 [["settle"], ["click", "live-refresh"], ["settle"],
                                  ["snap", "failed"]])
        status = snaps["failed"]["el"]["live-status"]
        assert status["text"] == f"{error} · {data['stale']}"
        assert status["className"] == "status error"
        # The figures already shown stay
        assert snaps["failed"]["el"]["card-return"]["text"] == (
            data["periods"][0]["cards"]["return"]["text"])

    def test_no_answer_and_an_unreadable_answer(self, tmp_path):
        snaps = _run_live_script(tmp_path, [{"network_error": True},
                                            {"status": 200, "bad_json": True}],
                                 [["settle"], ["snap", "down"], ["click", "live-refresh"],
                                  ["settle"], ["snap", "garbled"]])
        text = live_dashboard._SCRIPT_TEXT
        assert snaps["down"]["el"]["live-status"]["text"] == text["unreachable"]
        assert snaps["garbled"]["el"]["live-status"]["text"] == text["unreadable"]

    def test_without_the_chart_library_the_figures_still_show(self, tmp_path):
        data = payload(_view())
        snap = _run_live_script(tmp_path, [{"status": 200, "body": data}],
                                [["settle"], ["snap", "shown"]], plotly=False)["shown"]
        assert snap["el"]["live-status"]["text"] == (
            f"{data['status']} · {live_dashboard._SCRIPT_TEXT['no_charts']}")
        assert snap["el"]["card-return"]["text"] == data["periods"][0]["cards"]["return"]["text"]
        assert _cells(snap, "live-holdings") == [r["cells"] for r in data["holdings"]["rows"]]

    def test_before_the_bot_s_first_trade(self, tmp_path):
        data = payload(_view({OTHER_BETS: 5.0}, history=False))
        snap = _run_live_script(tmp_path, [{"status": 200, "body": data}],
                                [["settle"], ["snap", "shown"], ["click", "period-1"],
                                 ["snap", "clicked"]])
        shown = snap["shown"]
        assert shown["el"]["live-stats"]["hidden"] is True
        assert shown["el"]["live-empty"]["hidden"] is False
        assert shown["el"]["live-empty"]["text"] == data["empty"]
        assert shown["plots"] == []
        assert _cells(shown, "live-holdings") == [r["cells"] for r in data["holdings"]["rows"]]
        assert shown["el"]["live-area-details"]["hidden"] is True
        assert snap["clicked"]["plots"] == []

    def test_the_script_words_nothing_itself(self):
        # Every sentence it can show comes from the payload or _SCRIPT_TEXT
        body = live_dashboard._SCRIPT_TEMPLATE
        strings = re.findall(r"'([^']*)'", body)
        assert all(not re.search(r"[A-Z][a-z]+ [a-z]+ ", s) for s in strings), strings


# ---- keeping it read-only and away from the order path -----------------------------

def _imports_in(source: str) -> set[str]:
    """
    The package's own modules some source imports, however it spells the import.

    "from . import x", "from .x import y", "from kalshi_betting import x",
    "from kalshi_betting.x import y" and "import kalshi_betting.x" each name x.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                found.add(node.module)
            else:
                found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module == "kalshi_betting":
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kalshi_betting."):
            found.add(node.module.split(".", 1)[-1])
        elif isinstance(node, ast.Import):
            found |= {alias.name.split(".", 1)[-1] for alias in node.names
                      if alias.name.startswith("kalshi_betting.")}
    return found


def _package_imports(module) -> set[str]:
    """The package's own modules a module imports."""
    return _imports_in(inspect.getsource(module))


def _strings(module) -> list[str]:
    """Every string constant in a module's source, docstrings included."""
    return [node.value for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


class TestIsolation:
    @pytest.mark.parametrize("source", [
        "from . import live_dashboard", "from .live_dashboard import payload",
        "from kalshi_betting import live_dashboard", "from kalshi_betting import config, live_dashboard",
        "from kalshi_betting.live_dashboard import payload", "import kalshi_betting.live_dashboard",
        "import kalshi_betting.live_dashboard as d", "def f():\n    from kalshi_betting import live_dashboard"])
    def test_every_spelling_of_an_import_is_found(self, source):
        assert "live_dashboard" in _imports_in(source)

    def test_what_each_module_imports_from_the_package(self):
        assert _package_imports(live_portfolio) == {
            "auth", "config", "historical", "reporter", "dashboard", "treasury", "_http"}
        assert _package_imports(live_dashboard) == {
            "config", "live_portfolio", "historical", "treasury", "_http"}

    def test_no_other_module_imports_either(self):
        names = [m.name for m in pkgutil.iter_modules(kalshi_betting.__path__)]
        assert {"live_portfolio", "live_dashboard"} <= set(names)
        for name in names:
            module = importlib.import_module(f"kalshi_betting.{name}")
            imported = _package_imports(module)
            if name != "live_dashboard":
                assert "live_portfolio" not in imported, name
            assert "live_dashboard" not in imported, name

    @pytest.mark.parametrize("module", [live_portfolio, live_dashboard])
    def test_no_write_method_and_no_order_path(self, module):
        for text in _strings(module):
            assert not re.search(r"\b(POST|PUT|DELETE|PATCH)\b", text), text
            assert "/portfolio/orders" not in text, text

    def test_every_kalshi_read_goes_through_one_get(self):
        names = {"signed_request_json", "_signed_raw_get", "fetch_json_page",
                 "api_call_with_retry", "create_order"}

        def named(module) -> list[tuple[str, str]]:
            out = []
            parents = {}
            tree = ast.parse(inspect.getsource(module))
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                found = (node.attr if isinstance(node, ast.Attribute)
                         else node.id if isinstance(node, ast.Name) else None)
                if found is None:
                    continue
                owner = node
                while owner is not None and not isinstance(owner, ast.FunctionDef):
                    owner = parents.get(owner)
                out.append((found, owner.name if owner is not None else ""))
            return out

        portfolio = named(live_portfolio)
        assert {owner for name, owner in portfolio if name == "_historical_get"} == {"_get"}
        assert not {name for name, _ in portfolio} & names
        dashboard_names = {name for name, _ in named(live_dashboard)}
        assert not dashboard_names & (names | {"_historical_get", "_get"})

    def test_the_record_of_each_read_stays_out_of_git(self):
        # Every page load writes the cash and every holding there
        root = Path(__file__).resolve().parents[1]
        git = shutil.which("git")
        if git is None or not (root / ".git").exists():
            pytest.skip("not a git checkout")
        for name in (config.LIVE_PORTFOLIO_LOG_FILE.name, config.LIVE_DASHBOARD_LOG_FILE.name):
            done = subprocess.run([git, "check-ignore", "-q", "--no-index", name], cwd=root,
                                  capture_output=True, text=True, timeout=60)
            assert done.returncode == 0, f"{name} is not ignored by git"

    def test_importing_it_loads_no_order_path_module(self):
        code = ("import json, sys, kalshi_betting.live_dashboard\n"
                "print(json.dumps(sorted(m for m in sys.modules "
                "if m.startswith('kalshi_betting'))))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=120, cwd=str(Path(__file__).resolve().parents[1]))
        assert done.returncode == 0, done.stderr
        loaded = set(json.loads(done.stdout.strip().splitlines()[-1]))
        assert "kalshi_betting.live_dashboard" in loaded
        for name in ("trader", "main", "scheduler", "run_lock", "defaults_server", "v2_probe"):
            assert f"kalshi_betting.{name}" not in loaded, name
