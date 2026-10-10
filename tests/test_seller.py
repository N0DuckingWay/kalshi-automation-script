"""
Tests for seller.py: which held positions the take-profit rule sells this run.

The core check is parity with the backtest: one position built twice, as a
backtest trade valued by backtester._hold_readings and as the live account
judged by seller.plan_sales, must show equal profits at every check and the
same verdict at 50%, at the highest 1% level the backtest sells at, and at
the level just above it. Everything runs offline: the exchange reads
(order books, candles, settlements, market lookups) are stubs on the seller
module, and the client is a MagicMock that is never called.
"""
import ast
import inspect
import logging
import math
import sys
from dataclasses import replace as dc_replace
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kalshi_betting import backtester, config, seller
from kalshi_betting.config import fee_leg_exact
from kalshi_betting.scanner import (
    ApiMarket,
    HeldPosition,
    Settlement,
    leg_sides,
    market_ladder_keys,
)

# The backtest's checkpoint date, and the live run's moment set to that
# checkpoint (Monday 2026-01-12, 09:00 Los Angeles = 17:00 UTC)
_D = date(2026, 1, 12)
_START = date(2026, 1, 1)
_NOW = backtester._checkpoint_datetime(_D)
_CK = int(_NOW.timestamp())
_DAY = 86_400
# The previous Monday's checkpoint: every market's first candle, so each
# market's backtest quotes put _D on their weekly checkpoint grid
_FIRST = int(backtester._checkpoint_datetime(date(2026, 1, 5)).timestamp())

_A, _EV_A = "KXRKTA-26-JAN20", "KXRKTA-26"
_B, _EV_B = "KXRKTB-26-JAN25", "KXRKTB-26"
_A_CLOSE = datetime(2026, 1, 20, tzinfo=UTC)
_B_CLOSE = datetime(2026, 1, 25, tzinfo=UTC)


def _candle(ts: int, yes_ask: float, no_ask: float) -> dict:
    """One hourly candle ending at ts, with its YES and NO asks."""
    return {"ts": ts, "yes_ask_close": yes_ask, "no_ask_close": no_ask, "volume": 10.0}


def _rung(ticker: str, event: str, deadline: str, close: datetime,
          title: str | None = None, subtitle: str = "") -> ApiMarket:
    """A market of the rocket ladder (one question at several deadlines), as the run lists it."""
    return ApiMarket(ticker=ticker, event_ticker=event,
                     title=title or f"Will the rocket launch by {deadline}?",
                     subtitle=subtitle, status="active", close_time=close)


def _book(side: str, *levels: tuple[str, str]) -> dict:
    """An order book as scanner._fetch_orderbook returns it: bids on one side, ascending."""
    book = {"yes": [], "no": []}
    book[side] = [[price, size] for price, size in sorted(levels, key=lambda lv: float(lv[0]))]
    return book


def _settings(level: float | None, min_days: int | None = None) -> config.LiveSettings:
    """The run's settings with a sell level (and a minimum of days)."""
    return dc_replace(config.live_settings(), sell_at=level, sell_min_days=min_days)


class _World:
    """
    One live account and exchange, read through stubs on the seller module.

    Attributes:
        markets (dict): This run's market list by ticker.
        known (dict): Every market by ticker, for held markets' ladder labels.
        positions (dict): Ticker -> HeldPosition.
        books (dict): Ticker -> order book (absent: the book cannot be read).
        candles (dict): Ticker -> candles (absent: they cannot be read).
        settlements (list | None): The account's settlements; None: unreadable.
        unreadable (int): Settlement records left out as unreadable.
        unreadable_old (int): Unreadable records that paid out before any
            window the seller asks for: left out only when a call names no
            min_ts.
        lookup (dict): Ticker -> market the exchange returns on a lookup.
        calls (dict): Every stubbed read made, by kind ("min_ts": each
            settlements call's window).
    """

    def __init__(self) -> None:
        self.markets: dict = {}
        self.known: dict = {}
        self.positions: dict = {}
        self.books: dict = {}
        self.candles: dict = {}
        self.settlements: list | None = []
        self.unreadable = 0
        self.unreadable_old = 0
        self.lookup: dict = {}
        self.calls: dict = {"book": [], "candles": [], "settlements": 0, "lookup": [],
                            "min_ts": []}

    def list(self, market: ApiMarket) -> None:
        """Put a market in this run's market list."""
        self.markets[market.ticker] = market
        self.known[market.ticker] = market

    def hold(self, market: ApiMarket, count: float, exposure: float, fees: float,
             *, listed: bool = True) -> None:
        """Hold a market: count > 0 is YES, < 0 is NO."""
        if listed:
            self.list(market)
        self.known[market.ticker] = market
        self.positions[market.ticker] = HeldPosition(market.ticker, count, exposure, fees)

    def labels(self) -> dict:
        """Each held market's ladder labels, as resolve_held_ladders' labels_out gives them."""
        return {t: market_ladder_keys(self.known[t]) for t in self.positions}

    def plan(self, monkeypatch, level: float | None = 0.01, *, min_days: int | None = None,
             now: datetime = _NOW, labels: dict | None = None) -> list:
        """Run seller.plan_sales against this world, its reads stubbed and recorded."""
        def book(client, ticker):
            self.calls["book"].append(ticker)
            return self.books.get(ticker)

        def candles(client, ticker, event_ticker, start, end):
            self.calls["candles"].append((ticker, event_ticker, start, end))
            return self.candles.get(ticker)

        def settlements(client, *, unreadable_out=None, min_ts=None):
            self.calls["settlements"] += 1
            self.calls["min_ts"].append(min_ts)
            if unreadable_out is not None:
                # An unreadable record older than the window asked for is not
                # sent at all, as the exchange leaves it out
                unreadable_out["unreadable"] = (
                    self.unreadable + (self.unreadable_old if min_ts is None else 0))
            return self.settlements

        def lookup(client, ticker, markets_by_ticker, event_titles):
            if ticker in markets_by_ticker:
                return markets_by_ticker[ticker]
            self.calls["lookup"].append(ticker)
            return self.lookup.get(ticker)

        monkeypatch.setattr(seller, "_fetch_orderbook", book)
        monkeypatch.setattr(seller, "recent_candles", candles)
        monkeypatch.setattr(seller, "get_settlements", settlements)
        monkeypatch.setattr(seller, "market_for_labels", lookup)
        client = MagicMock()
        plans = seller.plan_sales(client, dict(self.positions),
                                  self.labels() if labels is None else labels,
                                  dict(self.markets), settings=_settings(level, min_days),
                                  now=now)
        # Every read went through a stub: the client itself is never touched
        assert client.mock_calls == []
        return plans


# ─── The same position, in the backtest and live ──────────────────────────────

class _Parity:
    """
    One position built twice: a backtest trade with quotes from its candles,
    and the live account holding it, with books at the backtest's
    checkpoint bids and the same candles behind the stubbed recent_candles.

    For a lone case, market A has paid out before the run: the live account
    holds B alone, and A is one of its settlements, costing what the trade's
    A leg cost.
    """

    def __init__(self, pair_type: str, *, candles_a: list, candles_b: list,
                 price_a: float, price_b: float, n: int = 30, lone: bool = False,
                 a_settles: datetime | None = None, a_result: str = "no",
                 titles: tuple[str, str] | None = None) -> None:
        self.n = n
        self.side_a, self.side_b = leg_sides(pair_type)
        settle_a = a_settles or datetime(2026, 2, 20, 12, tzinfo=UTC)
        if titles is None:
            market_a = _rung(_A, _EV_A, "Jan 20, 2026", _A_CLOSE)
            market_b = _rung(_B, _EV_B, "Jan 25, 2026", _B_CLOSE)
        else:
            market_a = _rung(_A, _EV_A, "", _A_CLOSE, title=titles[0], subtitle="Team A")
            market_b = _rung(_B, _EV_B, "", _B_CLOSE, title=titles[1], subtitle="Team A")
        # The backtest: one trade, each market's quotes from its candles
        quotes_a, _ = backtester._leg_quotes(
            {"ticker": _A, "settlement_ts": settle_a.isoformat(), "result": a_result},
            candles_a, _START)
        quotes_b, _ = backtester._leg_quotes(
            {"ticker": _B, "settlement_ts": "2026-02-25T12:00:00+00:00", "result": "yes"},
            candles_b, _START)
        fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        self.trade = backtester.BacktestTrade(
            pair_type=pair_type, ticker_a=_A, ticker_b=_B, title_a="", title_b="",
            category="Other", entry_date=date(2026, 1, 5), exit_date=date(2026, 2, 25),
            entry_pA=price_a, entry_pB=1 - price_b, entry_nA=1 - price_a, entry_nB=price_b,
            n=n, total_cost=n * price_a + n * price_b, fees=fee_a + fee_b,
            outcome_a=a_result, outcome_b="yes", actual_payoff=0.0, profit=0.0,
            profit_ratio=0.0, monthly_profit_ratio=0.0, kelly_fraction=0.1,
            expected_payoff=0.0, slippage=0.0, holding_days=10, balance_at_entry=10_000.0,
            close_date_a=_A_CLOSE.date(), close_date_b=_B_CLOSE.date(),
            marks=(quotes_a, quotes_b))
        # Live: the same markets held, books at the backtest's checkpoint bids
        self.world = _World()
        signed = {"yes": n, "no": -n}
        if lone:
            self.world.hold(market_b, signed[self.side_b], n * price_b, fee_b)
            self.world.lookup[_A] = market_a
            self.world.known[_A] = market_a
            revenue = n * (1.0 if a_result == self.side_a else 0.0)
            self.world.settlements = [Settlement(
                _A, _EV_A, a_result,
                n if self.side_a == "yes" else 0.0, n if self.side_a == "no" else 0.0,
                n * price_a if self.side_a == "yes" else 0.0,
                n * price_a if self.side_a == "no" else 0.0,
                fee_a, revenue, settle_a)]
        else:
            self.world.hold(market_a, signed[self.side_a], n * price_a, fee_a)
            self.world.hold(market_b, signed[self.side_b], n * price_b, fee_b)
        for ticker, quotes, side in ((_A, quotes_a, self.side_a), (_B, quotes_b, self.side_b)):
            if ticker in self.world.positions:
                bid = quotes.bid_at_checkpoint(_D, side)
                assert bid == bid, "the backtest needs a fresh bid at its checkpoint"
                self.world.books[ticker] = _book(side, (repr(bid), "100000"))
        self.world.candles = {_A: candles_a, _B: candles_b}

    def readings(self):
        """The backtest's sale and its profit at each check (None: not valued)."""
        return backtester._hold_readings([self.trade], _D, backtester._resolve_hold_days())

    def boundary(self, profits) -> float:
        """The highest 1% level the backtest sells at."""
        levels = [round(k / 100, 2) for k in range(1, 101)]
        selling = [lv for lv in levels if backtester._reached_every_day(lv, profits)]
        assert selling, "the case must sell at some level"
        return max(selling)


def _time_series_candles() -> tuple[list, list]:
    """YES on A and NO on B, priced at every check (A's NO ask sets its YES bid,
    B's YES ask its NO bid)."""
    a = [_candle(_FIRST, 0.30, 0.71),
         _candle(_CK - 2 * _DAY - 3600, 0.44, 0.57),     # check 2: YES bid 0.43
         _candle(_CK - _DAY - 1800, 0.43, 0.59),         # check 1: YES bid 0.41
         _candle(_CK, 0.44, 0.58)]                       # now: YES bid 0.42
    b = [_candle(_FIRST, 0.59, 0.41),
         _candle(_CK - 2 * _DAY - 3600, 0.50, 0.52),     # check 2: NO bid 0.50
         _candle(_CK - _DAY - 1800, 0.47, 0.54),         # check 1: NO bid 0.53
         _candle(_CK, 0.48, 0.53)]                       # now: NO bid 0.52
    return a, b


def _same_title_candles() -> tuple[list, list]:
    """NO on A and YES on B (A's YES ask sets its NO bid, B's NO ask its YES bid)."""
    a = [_candle(_FIRST, 0.50, 0.51),
         _candle(_CK - 2 * _DAY - 3600, 0.49, 0.52),     # check 2: NO bid 0.51
         _candle(_CK - _DAY - 1800, 0.47, 0.54),         # check 1: NO bid 0.53
         _candle(_CK, 0.48, 0.53)]                       # now: NO bid 0.52
    b = [_candle(_FIRST, 0.40, 0.61),
         _candle(_CK - 2 * _DAY - 3600, 0.49, 0.52),     # check 2: YES bid 0.48
         _candle(_CK - _DAY - 1800, 0.48, 0.54),         # check 1: YES bid 0.46
         _candle(_CK, 0.48, 0.53)]                       # now: YES bid 0.47
    return a, b


def _lone_case(a_settles: datetime, a_result: str, b_candles: list | None = None,
               a_candles: list | None = None) -> _Parity:
    """B held NO alone, its partner A (YES) paid out at a_settles."""
    a = a_candles or [_candle(_FIRST, 0.30, 0.71),
                      _candle(_CK - 4 * _DAY, 0.20, 0.81)]
    b = b_candles or [_candle(_FIRST, 0.59, 0.41),
                      _candle(_CK - 2 * _DAY - 3600, 0.16, 0.86),   # NO bid 0.84
                      _candle(_CK - _DAY - 1800, 0.14, 0.87),       # NO bid 0.86
                      _candle(_CK, 0.15, 0.86)]                     # NO bid 0.85
    return _Parity("time_series", candles_a=a, candles_b=b, price_a=0.30, price_b=0.40,
                   lone=True, a_settles=a_settles, a_result=a_result)


def _same_title_lone_case(a_settles: datetime, a_result: str, a_candles: list,
                          b_candles: list) -> _Parity:
    """B held YES alone, its partner A (held NO) paid out at a_settles: the
    same-title sides, so the partner's NO cost, count and NO bid are read."""
    return _Parity("same_title", candles_a=a_candles, candles_b=b_candles, price_a=0.40,
                   price_b=0.35, lone=True, a_settles=a_settles, a_result=a_result,
                   titles=("Who wins the game?", "Who wins the game?"))


def _same_title_lone_paid_before() -> _Parity:
    """B held YES alone; its partner A (NO) settled yes, paying $0, before every check."""
    a = [_candle(_FIRST, 0.50, 0.51),
         _candle(_CK - 4 * _DAY, 0.80, 0.21)]
    b = [_candle(_FIRST, 0.40, 0.61),
         _candle(_CK - 2 * _DAY - 3600, 0.93, 0.08),                # YES bid 0.92
         _candle(_CK - _DAY - 1800, 0.92, 0.09),                    # YES bid 0.91
         _candle(_CK, 0.94, 0.07)]                                  # YES bid 0.93
    return _same_title_lone_case(datetime.fromtimestamp(_CK - 3 * _DAY, UTC), "yes", a, b)


def _same_title_lone_paid_between() -> _Parity:
    """B held YES alone; its partner A (NO) settled no, paying $30, between the
    checks 2 and 1 days before, so A counts at its NO bid at the first."""
    a = [_candle(_FIRST, 0.50, 0.51),
         _candle(_CK - 2 * _DAY - 3600, 0.10, 0.91)]                # NO bid 0.90
    b = [_candle(_FIRST, 0.40, 0.61),
         _candle(_CK - 2 * _DAY - 3600, 0.04, 0.97),                # YES bid 0.03
         _candle(_CK - _DAY - 1800, 0.03, 0.98),                    # YES bid 0.02
         _candle(_CK, 0.03, 0.98)]                                  # YES bid 0.02
    return _same_title_lone_case(datetime.fromtimestamp(_CK - _DAY - 12 * 3600, UTC), "no",
                                 a, b)


def _paid_between_case(settles_ts: int = _CK - _DAY - 12 * 3600) -> _Parity:
    """B held NO alone; its partner A (YES) paid out $30 between the checks 2
    and 1 days before (by default; settles_ts sets when), so it counts at its
    candle bid at the first and at its payout from then on."""
    a = [_candle(_FIRST, 0.30, 0.71),
         _candle(_CK - 2 * _DAY - 3600, 0.91, 0.10)]                # YES bid 0.90
    b = [_candle(_FIRST, 0.59, 0.41),
         _candle(_CK - 2 * _DAY - 3600, 0.97, 0.04),                # NO bid 0.03
         _candle(_CK - _DAY - 1800, 0.98, 0.03),                    # NO bid 0.02
         _candle(_CK, 0.98, 0.03)]                                  # NO bid 0.02
    return _lone_case(datetime.fromtimestamp(settles_ts, UTC), "yes", b_candles=b,
                      a_candles=a)


_PARITY_CASES = {
    "exact time-series pair (YES A / NO B)": lambda: _Parity(
        "time_series", candles_a=_time_series_candles()[0],
        candles_b=_time_series_candles()[1], price_a=0.30, price_b=0.40),
    "exact same-title pair (NO A / YES B)": lambda: _Parity(
        "same_title", candles_a=_same_title_candles()[0], candles_b=_same_title_candles()[1],
        price_a=0.50, price_b=0.40, titles=("Who wins the game?", "Who wins the game?")),
    "lone market with a $0 partner paid out before every check": lambda: _lone_case(
        datetime.fromtimestamp(_CK - 3 * _DAY, UTC), "no"),
    "lone market whose partner paid out between the checks": _paid_between_case,
    # Paid out at the very moment of the check a day before: paid there
    "lone market whose partner paid out exactly at a check": lambda: _paid_between_case(
        _CK - _DAY),
    "lone YES market with a $0 NO partner paid out before every check":
        _same_title_lone_paid_before,
    "lone YES market whose NO partner paid out between the checks":
        _same_title_lone_paid_between,
}


class TestParityWithTheBacktest:
    """seller.plan_sales values a position at each check exactly as the
    backtest values the same position (backtester._hold_readings, with no
    depth model, so each sale is at one bid), and so sells at exactly the
    levels the backtest sells at."""

    @pytest.mark.parametrize("name", list(_PARITY_CASES))
    def test_equal_profits_at_every_check_and_the_same_verdict(self, monkeypatch, name):
        case = _PARITY_CASES[name]()
        readings = case.readings()
        assert readings is not None
        sale, profits = readings
        _per_trade, value, cost, potential_return = sale
        # A level every check reaches: the plan carries every check's profit
        (plan,) = case.world.plan(monkeypatch, 0.01)
        assert len(plan.profits) == len(profits) == backtester.TAKE_PROFIT_HOLD_DAYS
        for (live_realized, live_potential), (realized, potential) in zip(
                plan.profits, profits, strict=True):
            assert live_realized == pytest.approx(realized, abs=1e-9)
            assert live_potential == pytest.approx(potential, abs=1e-9)
        assert plan.cost_dollars == pytest.approx(cost, abs=1e-9)
        assert plan.count * config.CONTRACT_PAYOUT_DOLLARS == pytest.approx(potential_return)
        # The held markets' proceeds now: the sale's value less what a
        # partner that had paid out by now returned
        paid = sum(leg.payout_dollars for leg in plan.legs if leg.market is None)
        assert plan.proceeds_dollars == pytest.approx(value - paid, abs=1e-9)
        # The same verdict at 50%, at the highest level the backtest sells
        # at, and at the level just above it
        top = case.boundary(profits)
        for level in sorted({0.5, top, round(top + 0.01, 2)} - {1.01}):
            sells = backtester._reached_every_day(level, profits)
            assert bool(case.world.plan(monkeypatch, level)) == sells, level
        assert case.world.plan(monkeypatch, top)
        if top < 1.0:
            assert not case.world.plan(monkeypatch, round(top + 0.01, 2))

    def test_the_levels_tell_the_cases_apart(self):
        # Not vacuous: each case's highest selling level lies strictly inside
        # the grid, so the level just above it is one neither side sells at
        for name, build in _PARITY_CASES.items():
            case = build()
            _sale, profits = case.readings()
            top = case.boundary(profits)
            assert 0.01 < top < 1.0, name

    def test_a_partner_paid_between_the_checks_counts_at_its_bid_then_its_payout(
            self, monkeypatch):
        case = _paid_between_case()
        (plan,) = case.world.plan(monkeypatch, 0.01)
        cost = plan.cost_dollars
        held = 30 * 0.02 - fee_leg_exact(30, 0.02)
        # Now and a day before: the $30 payout; two days before: A's YES bid 0.90
        assert plan.profits[0][0] == pytest.approx(30 + held - cost, abs=1e-9)
        two_back = 30 * 0.90 - fee_leg_exact(30, 0.90) + 30 * 0.03 - fee_leg_exact(30, 0.03)
        assert plan.profits[2][0] == pytest.approx(two_back - cost, abs=1e-9)
        # A's candles were read for that check, under its own event
        assert (_A, _EV_A) in {(t, e) for t, e, _s, _e in case.world.calls["candles"]}
        (partner,) = [leg for leg in plan.legs if leg.market is None]
        assert partner.ticker == _A and partner.side == "yes" and partner.payout_dollars == 30

    def test_a_no_partner_counts_at_its_no_cost_and_its_no_bid(self, monkeypatch):
        # Same-title sides: B held YES, its partner A was held NO
        case = _same_title_lone_paid_between()
        (plan,) = case.world.plan(monkeypatch, 0.01)
        (partner,) = [leg for leg in plan.legs if leg.market is None]
        assert (partner.ticker, partner.side, partner.payout_dollars) == (_A, "no", 30.0)
        assert partner.cost_dollars == pytest.approx(30 * 0.40 + fee_leg_exact(30, 0.40))
        # Two days before, A had not paid out: its NO bid 0.90, B's YES bid 0.03
        two_back = 30 * 0.90 - fee_leg_exact(30, 0.90) + 30 * 0.03 - fee_leg_exact(30, 0.03)
        assert plan.profits[2][0] == pytest.approx(two_back - plan.cost_dollars, abs=1e-9)
        assert (_A, _EV_A) in {(t, e) for t, e, _s, _e in case.world.calls["candles"]}

    def test_a_partner_paid_a_second_after_a_check_is_priced_there_by_its_candle(
            self, monkeypatch, caplog):
        # Still unpaid at the check a day before, so it needs a candle bid
        # there, and A has no candle in the 24 hours before it: neither the
        # backtest nor the live run can value that check
        case = _paid_between_case(_CK - _DAY + 1)
        assert case.readings() is None
        with caplog.at_level(logging.INFO):
            assert case.world.plan(monkeypatch, 0.01) == []
        assert f"no YES bid on {_A} in the 24 hours before the check 1 day(s) before" in (
            caplog.text)

    def test_a_check_with_no_candle_in_its_24_hours_sells_on_neither_side(
            self, monkeypatch, caplog):
        a, b = _time_series_candles()
        # B's candle before check 1 is exactly 24 hours old there: it
        # belongs to the check before, so check 1 has no NO bid on B
        b = [c for c in b if c["ts"] != _CK - _DAY - 1800]
        b = [_candle(_CK - 2 * _DAY, 0.50, 0.52) if c["ts"] == _CK - 2 * _DAY - 3600 else c
             for c in b]
        case = _Parity("time_series", candles_a=a, candles_b=b, price_a=0.30, price_b=0.40)
        assert case.readings() is None
        with caplog.at_level(logging.INFO):
            assert case.world.plan(monkeypatch, 0.01) == []
        assert f"no NO bid on {_B} in the 24 hours before the check 1 day(s) before" in (
            caplog.text)

    @pytest.mark.parametrize("lone", [False, True])
    @pytest.mark.parametrize("min_days", [12, 13, 14])
    def test_the_days_rule_at_below_and_above_its_boundary(self, monkeypatch, lone, min_days):
        case = (_lone_case(datetime.fromtimestamp(_CK - 3 * _DAY, UTC), "no") if lone
                else _Parity("time_series", candles_a=_time_series_candles()[0],
                             candles_b=_time_series_candles()[1], price_a=0.30,
                             price_b=0.40))
        # 13 days from Monday 01-12 to the last held market's close on 01-25
        assert backtester._days_left([case.trade], _D) == 13
        sells = (backtester._far_enough([case.trade], _D, min_days)
                 and case.readings() is not None
                 and backtester._reached_every_day(0.01, case.readings()[1]))
        plans = case.world.plan(monkeypatch, 0.01, min_days=min_days)
        assert bool(plans) == sells == (min_days <= 13)
        if plans:
            assert plans[0].days_left == 13
        else:
            # Refused before any valuation: no book, settlement or candle read
            assert case.world.calls["book"] == []
            assert case.world.calls["settlements"] == 0
            assert case.world.calls["candles"] == []


# ─── Walking the book now ─────────────────────────────────────────────────────

def _exact_pair_world() -> _World:
    """YES on A and NO on B, 30 each, at the parity case's prices and candles."""
    case = _Parity("time_series", candles_a=_time_series_candles()[0],
                   candles_b=_time_series_candles()[1], price_a=0.30, price_b=0.40)
    return case.world


class TestWalk:
    def test_a_book_thinner_than_the_count_is_not_sold(self, monkeypatch, caplog):
        world = _exact_pair_world()
        world.books[_A] = _book("yes", ("0.42", "20"), ("0.41", "9"))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert f"the YES bids on {_A} hold fewer than 30 contracts" in caplog.text
        # Refused at the check now: no earlier day's candles were read
        assert world.calls["candles"] == []

    def test_a_book_that_cannot_be_read_is_not_sold(self, monkeypatch, caplog):
        world = _exact_pair_world()
        del world.books[_B]
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert f"the order book of {_B} could not be read" in caplog.text

    def test_a_two_level_book_gives_the_average_and_the_fee_on_it(self, monkeypatch):
        world = _exact_pair_world()
        world.books[_A] = _book("yes", ("0.60", "10"), ("0.55", "40"), ("0.80", "0"))
        (plan,) = world.plan(monkeypatch, 0.01)
        average = (0.60 * 10 + 0.55 * 20) / 30
        assert plan.walked[_A] == (pytest.approx(average), 0.55)
        assert plan.ladders[_A] == [[0.60, 10.0], [0.55, 40.0]]
        b_bid = plan.walked[_B][0]
        assert b_bid == 0.52
        expected = (30 * average - fee_leg_exact(30, average)
                    + 30 * b_bid - fee_leg_exact(30, b_bid))
        assert plan.proceeds_dollars == pytest.approx(expected, abs=1e-12)
        assert plan.profits[0][0] == pytest.approx(expected - plan.cost_dollars, abs=1e-12)


# ─── A lone held market's partner ─────────────────────────────────────────────

# Rungs of B's own event (the rocket question at other deadlines): what a
# lookup of each returns
_SAME_EVENT = "KXRKTB-26-JAN18"
_SAME_EVENT_2 = "KXRKTB-26-JAN19"
_OWN_EVENT_RUNGS = {
    _SAME_EVENT: _rung(_SAME_EVENT, _EV_B, "Jan 18, 2026", datetime(2026, 1, 18, tzinfo=UTC)),
    _SAME_EVENT_2: _rung(_SAME_EVENT_2, _EV_B, "Jan 19, 2026",
                         datetime(2026, 1, 19, tzinfo=UTC)),
}


def _lone_world(*settlements: Settlement, lookups: dict | None = None) -> _World:
    """B held NO alone (30 contracts), priced to sell at any level up to 36%.
    A lookup finds the rungs of B's own event, and whatever `lookups` adds."""
    case = _lone_case(datetime.fromtimestamp(_CK - 3 * _DAY, UTC), "no")
    world = case.world
    world.settlements = list(settlements)
    world.lookup = {**_OWN_EVENT_RUNGS, **(lookups or {})}
    return world


def _settled(ticker: str, event: str, *, yes: float = 30.0, no: float = 0.0,
             cost: float = 9.0, fees: float = 0.45, revenue: float = 0.0,
             at: datetime | None = None, result: str = "no") -> Settlement:
    """One settlement: what the account held when the market paid out."""
    return Settlement(ticker, event, result, yes, no, cost if yes else 0.0, cost if no else 0.0,
                      fees, revenue, at or datetime.fromtimestamp(_CK - 3 * _DAY, UTC))


class TestPartners:
    def test_a_partner_on_the_held_markets_own_event_is_looked_up_for_its_question(
            self, monkeypatch):
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        (plan,) = world.plan(monkeypatch, 0.01)
        assert world.calls["lookup"] == [_SAME_EVENT]
        assert [leg.ticker for leg in plan.legs] == [_B, _SAME_EVENT]

    def test_another_question_on_the_held_markets_own_event_is_not_its_partner(
            self, monkeypatch, caplog):
        # Another option of B's event: it shares B's event, but never paired with B
        option = _rung("KXRKTB-26-BOOM", _EV_B, "Jan 18, 2026", _A_CLOSE,
                       title="Will the rocket explode by Jan 18, 2026?")
        world = _lone_world(_settled("KXRKTB-26-BOOM", _EV_B),
                            lookups={"KXRKTB-26-BOOM": option})
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert world.calls["lookup"] == ["KXRKTB-26-BOOM"]
        assert "no paid-out partner" in caplog.text
        # With the real partner beside it, the option makes nothing ambiguous
        world = _lone_world(_settled("KXRKTB-26-BOOM", _EV_B), _settled(_SAME_EVENT, _EV_B),
                            lookups={"KXRKTB-26-BOOM": option})
        (plan,) = world.plan(monkeypatch, 0.01)
        assert plan.legs[1].ticker == _SAME_EVENT

    def test_a_settlement_is_the_partner_of_the_held_market_asking_its_question(
            self, monkeypatch, caplog):
        # Two lone held markets on two events of one multi-choice question:
        # L asks about Alpha on event E, M about Beta on event E2. P, settled
        # on E, asks about Beta: it shares L's event but M's question, so it
        # can only have been M's pair-mate, and L has no partner
        def option(ticker: str, event: str, name: str) -> ApiMarket:
            return _rung(ticker, event, "", _B_CLOSE, title="Who wins the cup?",
                         subtitle=name)

        world = _World()
        lone_l = option("KXCUP-E-ALPHA", "KXCUP-E", "Alpha")
        lone_m = option("KXCUP-E2-BETA", "KXCUP-E2", "Beta")
        world.hold(lone_l, -30.0, 12.0, 0.51)
        world.hold(lone_m, -30.0, 12.0, 0.51)
        for market in (lone_l, lone_m):
            world.books[market.ticker] = _book("no", ("0.85", "1000"))
            world.candles[market.ticker] = _lone_case(
                datetime.fromtimestamp(_CK - 3 * _DAY, UTC), "no").world.candles[_B]
        world.settlements = [_settled("KXCUP-E-BETA", "KXCUP-E")]
        world.lookup = {"KXCUP-E-BETA": option("KXCUP-E-BETA", "KXCUP-E", "Beta")}
        with caplog.at_level(logging.INFO):
            (plan,) = world.plan(monkeypatch, 0.01)
        assert [leg.ticker for leg in plan.legs] == ["KXCUP-E2-BETA", "KXCUP-E-BETA"]
        assert "Not selling NO KXCUP-E-ALPHA (alone on its ladder): no paid-out partner" in (
            caplog.text)

    def test_a_partner_on_another_held_markets_event_is_still_its_partner(
            self, monkeypatch, caplog):
        # The settled market asks B's question, on the event of C, which is
        # held but asks another question: it was B's pair-mate, never C's
        world = _lone_world(_settled("KXCEV-26-X", "KXCEV-26"),
                            lookups={"KXCEV-26-X": _rung("KXCEV-26-X", "KXCEV-26",
                                                         "Jan 20, 2026", _A_CLOSE)})
        held_c = _rung("KXCEV-26-Y", "KXCEV-26", "", _B_CLOSE, title="Who wins the game?")
        world.hold(held_c, 10.0, 4.0, 0.2)
        world.books["KXCEV-26-Y"] = _book("yes", ("0.50", "100"))
        with caplog.at_level(logging.INFO):
            (plan,) = world.plan(monkeypatch, 0.01)
        assert [leg.ticker for leg in plan.legs] == [_B, "KXCEV-26-X"]
        assert "Not selling YES KXCEV-26-Y (alone on its ladder): no paid-out partner" in (
            caplog.text)

    def test_a_settlement_older_than_the_age_limit_is_not_a_partner(self, monkeypatch,
                                                                      caplog):
        limit = seller.SALE_PARTNER_MAX_AGE_DAYS * _DAY
        at_limit = datetime.fromtimestamp(_CK - limit, UTC)
        world = _lone_world(_settled(_SAME_EVENT, _EV_B, at=at_limit))
        (plan,) = world.plan(monkeypatch, 0.01)
        assert plan.legs[1].ticker == _SAME_EVENT
        older = datetime.fromtimestamp(_CK - limit - 1, UTC)
        world = _lone_world(_settled(_SAME_EVENT, _EV_B, at=older))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert "no paid-out partner" in caplog.text
        # Not even asked about
        assert world.calls["lookup"] == []

    def test_an_old_settlement_makes_the_real_partner_neither_ambiguous_nor_blocked(
            self, monkeypatch):
        old = datetime.fromtimestamp(_CK - seller.SALE_PARTNER_MAX_AGE_DAYS * _DAY - 1, UTC)
        # An earlier trade of B's question at the same count, and an old
        # unrelated settlement no lookup can find: both left out
        world = _lone_world(_settled(_SAME_EVENT_2, _EV_B, at=old),
                            _settled("KXOLDNEWS-25MAR-X", "KXOLDNEWS-25MAR", at=old),
                            _settled(_SAME_EVENT, _EV_B))
        (plan,) = world.plan(monkeypatch, 0.01)
        assert plan.legs[1].ticker == _SAME_EVENT
        assert world.calls["lookup"] == [_SAME_EVENT]

    @pytest.mark.parametrize("result", ["void", "scalar"])
    def test_a_partner_that_settled_other_than_yes_or_no_is_not_used(self, monkeypatch,
                                                                       caplog, result):
        # A voided market refunds its cost, which would count as profit
        world = _lone_world(_settled(_SAME_EVENT, _EV_B, result=result, revenue=9.0))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert (f"its partner {_SAME_EVENT} settled {result} rather than yes or no"
                in caplog.text)
        assert world.calls["candles"] == []

    def test_a_held_market_with_no_question_has_no_partner(self, monkeypatch, caplog):
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        labels = world.labels()
        labels[_B] = frozenset(label for label in labels[_B] if label[0] == "event")
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01, labels=labels) == []
        assert "its question is unknown" in caplog.text
        assert world.calls["settlements"] == 0

    def test_one_partner_on_another_event_is_looked_up(self, monkeypatch):
        world = _lone_world(_settled(_A, _EV_A),
                            lookups={_A: _rung(_A, _EV_A, "Jan 20, 2026", _A_CLOSE)})
        (plan,) = world.plan(monkeypatch, 0.01)
        assert world.calls["lookup"] == [_A]
        partner = plan.legs[1]
        assert (partner.side, partner.count, partner.cost_dollars) == ("yes", 30, 9.0 + 0.45)
        assert partner.payout_dollars == 0.0 and partner.market is None

    @pytest.mark.parametrize("settlements, reason", [
        ([], "no paid-out partner"),
        # Held on the same side as B (NO), not the other
        ([_settled(_SAME_EVENT, _EV_B, yes=0.0, no=30.0)], "no paid-out partner"),
        # Another count
        ([_settled(_SAME_EVENT, _EV_B, yes=20.0)], "no paid-out partner"),
        # Some of both sides
        ([_settled(_SAME_EVENT, _EV_B, yes=30.0, no=5.0)], "no paid-out partner"),
        # Two that could be its partner
        ([_settled(_SAME_EVENT, _EV_B), _settled(_SAME_EVENT_2, _EV_B)],
         "more than one settled market could be its paid-out partner"),
    ])
    def test_not_exactly_one_partner_is_not_sold(self, monkeypatch, caplog, settlements,
                                                 reason):
        world = _lone_world(*settlements)
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert reason in caplog.text
        # Refused before any earlier day's candle is read
        assert world.calls["candles"] == []

    def test_a_partner_on_another_ladder_is_not_its_partner(self, monkeypatch, caplog):
        other = _rung("KXRAIN-26", "KXRAIN", "Jan 20, 2026", _A_CLOSE,
                      title="Will it rain by Jan 20, 2026?")
        world = _lone_world(_settled("KXRAIN-26", "KXRAIN"), lookups={"KXRAIN-26": other})
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert world.calls["lookup"] == ["KXRAIN-26"]
        assert "no paid-out partner" in caplog.text

    def test_a_candidate_that_cannot_be_looked_up_may_be_the_partner(self, monkeypatch,
                                                                     caplog):
        world = _lone_world(_settled(_A, _EV_A))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert f"settled market {_A} could not be looked up" in caplog.text

    def test_past_the_lookup_limit_the_rest_are_not_sold(self, monkeypatch, caplog):
        monkeypatch.setattr(seller, "SALE_PARTNER_MAX_LOOKUPS", 1)
        rain = _rung("KXRAIN-26", "KXRAIN", "Jan 20, 2026", _A_CLOSE,
                     title="Will it rain by Jan 20, 2026?")
        world = _lone_world(_settled("KXRAIN-26", "KXRAIN"), _settled(_A, _EV_A),
                            lookups={"KXRAIN-26": rain,
                                     _A: _rung(_A, _EV_A, "Jan 20, 2026", _A_CLOSE)})
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        # One lookup, then the limit: the second candidate was never asked about
        assert world.calls["lookup"] == ["KXRAIN-26"]
        assert "the limit of 1 partner lookups this run was reached" in caplog.text

    def test_the_limit_counts_lookups_across_held_markets(self, monkeypatch, caplog):
        # Two lone held markets on two ladders (30 and 20 contracts), each
        # with its partner on another event: with a limit of one lookup, the
        # first in ticker order is sold and the second is not
        monkeypatch.setattr(seller, "SALE_PARTNER_MAX_LOOKUPS", 1)
        world = _lone_world(_settled(_A, _EV_A),
                            lookups={_A: _rung(_A, _EV_A, "Jan 20, 2026", _A_CLOSE)})
        snow = _rung("KXSNOWB-26", "KXSNOWB", "Jan 25, 2026", _B_CLOSE,
                     title="Will it snow by Jan 25, 2026?")
        world.hold(snow, -20.0, 8.0, 0.40)
        world.books["KXSNOWB-26"] = world.books[_B]
        world.settlements.append(_settled("KXSNOWA-26", "KXSNOWA", yes=20.0, cost=6.0))
        world.lookup["KXSNOWA-26"] = _rung("KXSNOWA-26", "KXSNOWA", "Jan 20, 2026", _A_CLOSE,
                                           title="Will it snow by Jan 20, 2026?")
        with caplog.at_level(logging.INFO):
            plans = world.plan(monkeypatch, 0.01)
        assert [plan.legs[0].ticker for plan in plans] == [_B]
        assert world.calls["lookup"] == [_A]
        assert ("Not selling NO KXSNOWB-26 (alone on its ladder): the limit of 1 partner "
                "lookups this run was reached") in caplog.text

    def test_unreadable_settlements_sell_no_lone_market_and_warn_once(self, monkeypatch,
                                                                      caplog):
        world = _lone_world()
        world.settlements = None
        snow = _rung("KXSNOWB-26", "KXSNOWB", "Jan 25, 2026", _B_CLOSE,
                     title="Will it snow by Jan 25, 2026?")
        world.hold(snow, -30.0, 12.0, 0.51)
        world.books["KXSNOWB-26"] = world.books[_B]
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert world.calls["settlements"] == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert [r.getMessage() for r in warnings] == [
            "Not selling any held market whose partner has paid out this run: the "
            "account's settlements could not be read"]

    def test_a_settlement_record_left_out_may_be_the_partner(self, monkeypatch, caplog):
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        world.unreadable = 1
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert "1 of the account's settlements could not be read" in caplog.text

    def test_only_the_partner_window_of_settlements_is_asked_for(self, monkeypatch):
        # The read asks for the markets that paid out in the last
        # SALE_PARTNER_MAX_AGE_DAYS (a second more, never less), so an
        # unreadable record older than that is never sent and cannot stop
        # the sale; one inside the window still does
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        world.unreadable_old = 1
        (plan,) = world.plan(monkeypatch, 0.01)
        assert plan.legs[1].ticker == _SAME_EVENT
        limit = seller.SALE_PARTNER_MAX_AGE_DAYS * _DAY
        assert world.calls["min_ts"] == [_CK - limit - 1]
        assert all(type(ts) is int for ts in world.calls["min_ts"])

    def test_the_age_check_still_applies_to_what_the_window_returns(self, monkeypatch,
                                                                     caplog):
        # The exchange's window reaches a second past the age limit, so a
        # settlement in that second is returned; the seller's own check
        # still leaves it out
        limit = seller.SALE_PARTNER_MAX_AGE_DAYS * _DAY
        world = _lone_world(_settled(_SAME_EVENT, _EV_B,
                                     at=datetime.fromtimestamp(_CK - limit - 1, UTC)))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert world.calls["min_ts"] == [_CK - limit - 1]
        assert "no paid-out partner" in caplog.text

    def test_a_partner_recorded_as_settling_after_now_is_not_used(self, monkeypatch, caplog):
        world = _lone_world(_settled(_SAME_EVENT, _EV_B,
                                     at=datetime.fromtimestamp(_CK + 60, UTC)))
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert "is recorded as settling after now" in caplog.text


# ─── Which held markets are judged, and what is read ──────────────────────────

class TestPositionsAndReads:
    def test_no_sell_level_reads_nothing(self, monkeypatch):
        world = _exact_pair_world()
        assert world.plan(monkeypatch, None) == []
        assert world.calls == {"book": [], "candles": [], "settlements": 0, "lookup": [],
                               "min_ts": []}

    def test_a_held_market_missing_from_the_market_list_is_not_sold(self, monkeypatch,
                                                                    caplog):
        world = _exact_pair_world()
        del world.markets[_B]
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert f"{_B} is not in this run's market list" in caplog.text
        assert world.calls["book"] == []

    def test_an_unknown_ladder_sells_nothing(self, monkeypatch, caplog):
        world = _exact_pair_world()
        labels = world.labels()
        labels[_B] = frozenset()
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01, labels=labels) == []
        assert f"the ladder of 1 held market(s) is unknown ({_B})" in caplog.text
        assert ("Positions to sell this run: 0 of 0 (2 held market(s) never sold: "
                "1 ladder unknown, 1 not judged)") in caplog.text
        assert world.calls["book"] == []

    def test_three_held_markets_on_one_ladder_are_never_sold(self, monkeypatch, caplog):
        world = _exact_pair_world()
        world.hold(_rung("KXRKTC-26-FEB1", "KXRKTC-26", "Feb 1, 2026",
                         datetime(2026, 2, 1, tzinfo=UTC)), 30.0, 9.0, 0.45)
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert caplog.text.count("other held markets share its ladder") == 3
        assert ("Positions to sell this run: 0 of 0 (3 held market(s) never sold: "
                "3 not one exact pair)") in caplog.text
        assert world.calls["book"] == []

    @pytest.mark.parametrize("position, reason", [
        (HeldPosition("KXLONE", None, 9.0, 0.45), "count unreadable"),
        (HeldPosition("KXLONE", 30.0, None, 0.45), "cost unreadable"),
        (HeldPosition("KXLONE", 30.0, 0.0, 0.45), "cost too small"),
    ])
    def test_a_held_market_held_pairs_leaves_out_says_why(self, monkeypatch, caplog, position,
                                                          reason):
        world = _World()
        world.list(_rung("KXLONE", "KXLONE-EV", "Jan 20, 2026", _A_CLOSE))
        world.positions["KXLONE"] = position
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert f"(1 held market(s) never sold: 1 {reason})" in caplog.text

    def test_a_check_below_the_level_now_reads_no_candle(self, monkeypatch, caplog):
        world = _exact_pair_world()
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.99) == []
        assert world.calls["candles"] == []
        assert "-> keep (now 65%)" in caplog.text
        # A lone held market too: its partner is found (the check now needs
        # its cost and payout, so the settlements are read), then no earlier
        # day's candle is read for either market
        lone = _lone_world(_settled(_SAME_EVENT, _EV_B))
        with caplog.at_level(logging.INFO):
            assert lone.plan(monkeypatch, 0.5) == []
        assert lone.calls["settlements"] == 1 and lone.calls["candles"] == []
        assert lone.calls["lookup"] == [_SAME_EVENT]
        assert (f"NO {_B} with paid-out YES {_SAME_EVENT} (settled no, paid $0.00), 30 each"
                in caplog.text)

    def test_a_lone_market_refused_before_its_partner_reads_no_settlement(
            self, monkeypatch):
        # The days rule, then a book too thin: neither reads the settlements
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        assert world.plan(monkeypatch, 0.01, min_days=14) == []
        world.books[_B] = _book("no", ("0.85", "5"))
        assert world.plan(monkeypatch, 0.01) == []
        assert world.calls["settlements"] == 0
        assert world.calls["candles"] == []

    def test_one_check_reads_no_candle(self, monkeypatch):
        monkeypatch.setattr(seller, "TAKE_PROFIT_HOLD_DAYS", 1)
        world = _exact_pair_world()
        (plan,) = world.plan(monkeypatch, 0.01)
        assert len(plan.profits) == 1 and world.calls["candles"] == []

    def test_candles_are_read_once_per_market_over_the_checks_window(self, monkeypatch):
        world = _exact_pair_world()
        world.plan(monkeypatch, 0.01)
        start = _CK - 3 * _DAY - 3600
        assert world.calls["candles"] == [(_A, _EV_A, start, _CK), (_B, _EV_B, start, _CK)]

    @pytest.mark.parametrize("days", [0, 8, True, 2.0, "3", None])
    def test_a_bad_number_of_checks_is_refused(self, monkeypatch, days):
        monkeypatch.setattr(seller, "TAKE_PROFIT_HOLD_DAYS", days)
        with pytest.raises(ValueError, match="TAKE_PROFIT_HOLD_DAYS must be a whole number"):
            _exact_pair_world().plan(monkeypatch, 0.01)

    def test_now_must_carry_a_time_zone(self, monkeypatch):
        with pytest.raises(ValueError, match="timezone-aware"):
            _exact_pair_world().plan(monkeypatch, 0.01, now=_NOW.replace(tzinfo=None))

    def test_a_count_that_is_not_whole_is_not_sold(self, monkeypatch, caplog):
        world = _exact_pair_world()
        world.positions[_A] = HeldPosition(_A, 30.5, 9.15, 0.45)
        world.positions[_B] = HeldPosition(_B, -30.5, 12.2, 0.51)
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert "its count 30.5 is not a whole number of contracts" in caplog.text


class TestLogLines:
    def test_a_sale_names_every_check_and_the_summary_counts_it(self, monkeypatch, caplog):
        world = _exact_pair_world()
        with caplog.at_level(logging.INFO):
            (plan,) = world.plan(monkeypatch, 0.6)
        lines = [r.getMessage() for r in caplog.records]
        assert (f"Take-profit check (sell at 60%): YES {_A} / NO {_B}, 30 each, cost $21.96, "
                "potential profit $8.04, 13 day(s) left; now 65%, 1 day before 65%, "
                "2 days before 61% -> sell") in lines
        assert lines[-1] == "Positions to sell this run: 1 of 1"
        assert plan.title == "Will the rocket launch by Jan 20, 2026?"
        assert plan.level == 0.6 and plan.count == 30 and plan.days_left == 13
        assert [(leg.ticker, leg.side) for leg in plan.legs] == [(_A, "yes"), (_B, "no")]

    def test_a_keep_names_the_checks_read(self, monkeypatch, caplog):
        world = _exact_pair_world()
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.62) == []
        assert ("potential profit $8.04, 13 day(s) left -> keep (now 65%, 1 day before 65%, "
                "2 days before 61%)") in caplog.text

    def test_no_potential_profit_is_kept(self, monkeypatch, caplog):
        world = _exact_pair_world()
        world.positions[_A] = HeldPosition(_A, 30.0, 20.0, 0.45)
        with caplog.at_level(logging.INFO):
            assert world.plan(monkeypatch, 0.01) == []
        assert "potential profit -$2.96" in caplog.text
        assert "-> keep (no potential profit)" in caplog.text


class TestHeldPairsQuietForSelling:
    def test_held_pairs_with_log_off_logs_nothing_and_finds_the_same(self, caplog):
        from kalshi_betting.scanner import held_pairs

        world = _exact_pair_world()
        labels = world.labels()
        with caplog.at_level(logging.DEBUG):
            quiet = held_pairs(world.positions, labels, world.markets, log=False)
            assert caplog.records == []
            assert quiet == held_pairs(world.positions, labels, world.markets)
            # Not vacuous: with log on, the same call logs its lines
            assert caplog.records
            # The unknown-ladder line too
            caplog.clear()
            assert held_pairs(world.positions, {_A: labels[_A]}, world.markets,
                              log=False) == {}
            assert caplog.records == []


# ─── The module's shape ───────────────────────────────────────────────────────

def _tree():
    return ast.parse(inspect.getsource(seller))


def _calls(func_name: str) -> set[str]:
    """Every name a function of seller calls, bare or as an attribute."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {(sub.func.id if isinstance(sub.func, ast.Name)
                     else getattr(sub.func, "attr", None))
                    for sub in ast.walk(node) if isinstance(sub, ast.Call)}
    raise AssertionError(f"seller.{func_name} not found")


class TestIsolation:
    def test_it_imports_only_the_standard_library_config_scanner_and_historical(self):
        project = set()
        for node in ast.walk(_tree()):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    project.add(node.module)
                else:
                    assert node.module.split(".")[0] in sys.stdlib_module_names, node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        assert project == {"config", "scanner", "historical"}

    def test_main_alone_imports_it(self):
        # The live run (main._run_prod) is its one user; no other pipeline,
        # backtest or report module imports it
        import importlib
        import pkgutil

        import kalshi_betting

        importers = set()
        for info in pkgutil.iter_modules(kalshi_betting.__path__):
            if info.name == "seller":
                continue
            module = importlib.import_module(f"kalshi_betting.{info.name}")
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.ImportFrom):
                    names = {(node.module or "").split(".")[-1]} | {a.name for a in node.names}
                elif isinstance(node, ast.Import):
                    names = {a.name.split(".")[-1] for a in node.names}
                else:
                    continue
                if "seller" in names:
                    importers.add(info.name)
        assert importers == {"main"}

    def test_it_decides_and_prices_through_the_shared_rule(self):
        # The decision: config's one test at each check and at every check
        plan_one = _calls("_plan_one")
        assert {"take_profit_reached", "reached_every_check", "days_to_maturity",
                "fee_leg_exact", "bid_ladder", "walk_bids", "bid_before",
                "_fetch_orderbook"} <= plan_one
        assert {"held_pairs"} <= _calls("plan_sales")
        assert {"get_settlements"} <= _calls("settlements")
        assert {"recent_candles"} <= _calls("candles")
        assert {"market_for_labels", "market_ladder_keys"} <= _calls("labels")
        # held_pairs is asked to stay quiet
        for node in ast.walk(_tree()):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "held_pairs"):
                assert any(kw.arg == "log" and isinstance(kw.value, ast.Constant)
                           and kw.value.value is False for kw in node.keywords)

    def test_it_holds_no_copy_of_the_rule_or_the_fee(self):
        names = {node.id for node in ast.walk(_tree()) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(_tree()) if isinstance(node, ast.Attribute)}
        # No float-noise allowance of its own, no fee rate, no candle bid rule
        for name in ("PRICE_EPSILON", "TAKER_FEE_RATE", "candle_sale_bids",
                     "usable_candle_ask"):
            assert name not in names, name
        own_bids = [n for n in ast.walk(_tree()) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "round"]
        assert own_bids == []

    def test_plan_sales_requires_the_runs_settings(self):
        signature = inspect.signature(seller.plan_sales)
        settings = signature.parameters["settings"]
        assert settings.kind is inspect.Parameter.KEYWORD_ONLY
        assert settings.default is inspect.Parameter.empty
        assert signature.parameters["now"].default is inspect.Parameter.empty


def test_plans_are_frozen():
    leg = seller.SaleLeg("T", "E", "yes", 1, 0.5)
    with pytest.raises(AttributeError):
        leg.count = 2
    assert math.isclose(leg.cost_dollars, 0.5) and leg.market is None
    plan = seller.SalePlan("T", (leg,), 1, 0.5, {}, {}, 0.6, ((0.1, 0.5),), None, 0.2)
    with pytest.raises(AttributeError):
        plan.count = 2
    assert timedelta(seconds=seller._DAY_SECONDS) == timedelta(days=1)
