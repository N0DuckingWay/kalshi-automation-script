"""
Tests for seller.py: which held positions the take-profit rule sells this run.

The core check is parity with the backtest: one position built twice, as a
backtest trade valued by backtester._hold_readings and as the live account
judged by seller.plan_sales, must show equal profits at every check and the
same verdict at 50%, at the highest 1% level the backtest sells at, and at
the level just above it. Everything runs offline: the exchange reads
(order books, candles, settlements, market lookups) are stubs on the seller
module, and the client is a MagicMock that is never called.

seller.plan_trims (selling part of a held pair down to its Kelly size) is
held to the backtest the same way: one pair built as a backtest trade and
as the live account must give backtester._kelly_trim's count, and its wait
after a trade is checked on every day from 1 to 8 after a purchase.
"""
import ast
import dataclasses
import inspect
import logging
import math
import sys
from dataclasses import replace as dc_replace
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from kalshi_betting import backtester, config, depth_model, seller
from kalshi_betting.config import fee_leg_exact
from kalshi_betting.scanner import (
    ApiMarket,
    FillTimes,
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
        fills (FillTimes | None): When each market last traded, as the
            account's fills list says (None: it cannot be read).
        fill_reads (int): How many times the fills list was read.
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
        self.fills: FillTimes | None = None
        self.fill_reads = 0

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

    def trims(self, monkeypatch, settings: config.LiveSettings | None = None, *,
              now: datetime = _NOW, cash_cents: float = 5_000,
              positions_value_cents: float | None = 5_000, skip=frozenset(),
              labels: dict | None = None) -> list:
        """Run seller.plan_trims against this world, its reads stubbed and recorded.

        By default the Kelly trim is on with no minimum price rise, and it is
        sized on $100: $50 of cash and open positions Kalshi values at $50."""
        def book(client, ticker):
            self.calls["book"].append(ticker)
            return self.books.get(ticker)

        def candles(client, ticker, event_ticker, start, end):
            self.calls["candles"].append((ticker, event_ticker, start, end))
            return self.candles.get(ticker)

        def fills(client):
            self.fill_reads += 1
            return self.fills

        monkeypatch.setattr(seller, "_fetch_orderbook", book)
        monkeypatch.setattr(seller, "recent_candles", candles)
        monkeypatch.setattr(seller, "get_fill_times", fills)
        client = MagicMock()
        plans = seller.plan_trims(
            client, dict(self.positions), self.labels() if labels is None else labels,
            dict(self.markets), settings=_trim_settings() if settings is None else settings,
            now=now, cash_cents=cash_cents, positions_value_cents=positions_value_cents,
            skip=skip)
        # Every read went through a stub: the client itself is never touched
        assert client.mock_calls == []
        return plans


def _trim_settings(*, k: float = 0.8, cap: float = 1.0, st_cap: float = 0.2,
                   min_rise: float | None = None, on: bool = True) -> config.LiveSettings:
    """The run's settings with the Kelly trim on (or off), and its k, caps and minimum rise."""
    return dc_replace(config.live_settings(), trim_to_kelly=on, trim_min_rise=min_rise,
                      interval_discount=k, size_cap=cap, same_title_size_cap=st_cap)


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


# ─── Trimming a held pair to its Kelly size ───────────────────────────────────

# When the standard pair last traded: the previous Monday's checkpoint, seven
# calendar days before the run (_NOW) on the bot's own clock
_BOUGHT = datetime.fromtimestamp(_FIRST, UTC)
# The fills list's oldest fill: far enough back to rule out any later trade
_LONG_AGO = _NOW - timedelta(days=30)
_PACIFIC = ZoneInfo("America/Los_Angeles")

# The standard time-series pair's quotes at the run, as (YES ask, NO ask) on
# each market: YES is held on A (best YES bid 0.42) and NO on B (best NO bid
# 0.52), so one contract pair sells for 0.94 before the fee
_A_NOW = (0.44, 0.58)
_B_NOW = (0.48, 0.53)


def _fills(latest: dict | None = None, oldest: datetime | None = _LONG_AGO) -> FillTimes:
    """The account's fill times, as scanner.get_fill_times returns them."""
    return FillTimes(latest=dict(latest or {}), oldest=oldest)


def _ladder_model(levels) -> depth_model.DepthModel:
    """A depth model whose every bid ladder holds `levels`: (distance below
    the best bid, contracts), each distance one of DEPTH_MODEL_DISTANCES."""
    ladder = [[round(0.50 - distance, 2), contracts] for distance, contracts in levels]
    model = depth_model.fit([{"ticker": "L", "taken_at": "2026-09-28T16:00:00Z",
                              "volume_24h": 400.0, "yes": ladder, "no": ladder}])
    assert model is not None
    return model


class _TrimCase:
    """
    One exact held pair built twice for the Kelly trim: a backtest trade whose
    quotes come from its candles, and the live account holding it, with order
    books whose best bids on both sides of both markets are the candles' own
    at the backtest's checkpoint, and the same candles behind the stubbed
    recent_candles. The pair was bought at the previous Monday's checkpoint,
    which is also when the account's fills list says both markets last traded.

    Attributes:
        n (int): Contract pairs held.
        trade (BacktestTrade): The pair as the backtest holds it.
        world (_World): The live account and exchange.
        positions_cents (int): What Kalshi values the open positions at.
    """

    def __init__(self, pair_type: str = "time_series", a_now: tuple = _A_NOW,
                 b_now: tuple = _B_NOW, *, n: int = 30, a_then: tuple | None = None,
                 b_then: tuple | None = None, model=None, price_a: float = 0.30,
                 price_b: float = 0.40, positions_cents: int = 5_000) -> None:
        self.n = n
        self.positions_cents = positions_cents
        self.side_a, self.side_b = leg_sides(pair_type)
        if pair_type == "time_series":
            market_a = _rung(_A, _EV_A, "Jan 20, 2026", _A_CLOSE)
            market_b = _rung(_B, _EV_B, "Jan 25, 2026", _B_CLOSE)
        else:
            # One question listed on two series, both closing at one moment
            market_a = _rung(_A, _EV_A, "", _B_CLOSE, title="Who wins the game?",
                             subtitle="Team A")
            market_b = _rung(_B, _EV_B, "", _B_CLOSE, title="Who wins the game?",
                             subtitle="Team A")
        candles_a = [_candle(_FIRST, *(a_then or a_now)), _candle(_CK, *a_now)]
        candles_b = [_candle(_FIRST, *(b_then or b_now)), _candle(_CK, *b_now)]
        # The backtest: one trade, each market's quotes from its candles
        quotes_a, _ = backtester._leg_quotes(
            {"ticker": _A, "settlement_ts": "2026-02-20T12:00:00+00:00", "result": "no"},
            candles_a, _START, model)
        quotes_b, _ = backtester._leg_quotes(
            {"ticker": _B, "settlement_ts": "2026-02-25T12:00:00+00:00", "result": "yes"},
            candles_b, _START, model)
        fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        self.trade = backtester.BacktestTrade(
            pair_type=pair_type, ticker_a=_A, ticker_b=_B, title_a="", title_b="",
            category="Other", entry_date=date(2026, 1, 5), exit_date=date(2026, 2, 25),
            entry_pA=price_a, entry_pB=1 - price_b, entry_nA=1 - price_a, entry_nB=price_b,
            n=n, total_cost=n * price_a + n * price_b, fees=fee_a + fee_b,
            outcome_a="no", outcome_b="yes", actual_payoff=0.0, profit=0.0,
            profit_ratio=0.0, monthly_profit_ratio=0.0, kelly_fraction=0.1,
            expected_payoff=0.0, slippage=0.0, holding_days=10, balance_at_entry=10_000.0,
            close_date_a=_A_CLOSE.date(), close_date_b=_B_CLOSE.date(),
            marks=(quotes_a, quotes_b))
        # Live: the same two markets held, with both sides of both books
        self.world = _World()
        signed = {"yes": n, "no": -n}
        self.world.hold(market_a, signed[self.side_a], n * price_a, fee_a)
        self.world.hold(market_b, signed[self.side_b], n * price_b, fee_b)
        for ticker, quotes, held_side, (yes_ask, no_ask) in (
                (_A, quotes_a, self.side_a, a_now), (_B, quotes_b, self.side_b, b_now)):
            # A side's best bid is 1 minus the other side's ask, as a candle shows it
            best = {"yes": round(1.0 - no_ask, 6), "no": round(1.0 - yes_ask, 6)}
            book = {}
            for side in ("yes", "no"):
                fresh = quotes.bid_at_checkpoint(_D, side)
                # Where the backtest has a bid, the book's best bid is that very number
                assert fresh != fresh or fresh == best[side]
                # The side held sells down the backtest's own modeled ladder when it
                # has one; otherwise one bid in any size, as the backtest sells
                ladder = quotes.sale_ladder(_D, best[side]) if side == held_side else None
                book[side] = ladder if ladder is not None else [[best[side], 100_000.0]]
            self.world.books[ticker] = book
        self.world.candles = {_A: candles_a, _B: candles_b}
        self.world.fills = _fills({_A: _BOUGHT, _B: _BOUGHT})

    def value(self, value_cents: int) -> float:
        """The portfolio value plan_trims sizes on for this many cents in all."""
        return (value_cents - self.positions_cents) / 100 + self.positions_cents / 100

    def backtest(self, value_cents: int, *, k: float = 0.8, cap: float = 1.0,
                 st_cap: float = 0.2):
        """backtester._kelly_trim's (decision, sale) for the pair at one portfolio value."""
        return backtester._kelly_trim([self.trade], _D, self.value(value_cents), k, cap, st_cap)

    def live(self, monkeypatch, value_cents: int, *, k: float = 0.8, cap: float = 1.0,
             st_cap: float = 0.2, min_rise: float | None = None, **kwargs) -> list:
        """seller.plan_trims' plans for the pair at the same portfolio value."""
        return self.world.trims(
            monkeypatch, _trim_settings(k=k, cap=cap, st_cap=st_cap, min_rise=min_rise),
            cash_cents=value_cents - self.positions_cents,
            positions_value_cents=self.positions_cents, **kwargs)


# Each case: (how the pair is built, the portfolio value in cents, the caps).
# The backtest and the live plan must sell the same number of contract pairs
_TRIM_PARITY_CASES = {
    # Worth $27 of a $100 portfolio under a 10% cap: sold down to the cap
    "a time-series pair past its size cap": (
        lambda: _TrimCase(), 10_000, {"cap": 0.10}),
    # No cap in the way: Kelly's own share of a $50 portfolio is the limit
    "a time-series pair past its Kelly share, under its cap": (
        lambda: _TrimCase(a_now=(0.30, 0.72), b_now=(0.62, 0.40)), 5_000, {}),
    "a time-series pair within its share": (
        lambda: _TrimCase(), 100_000, {}),
    # NO on A (best NO bid 0.50) and YES on B (best YES bid 0.40): held to the
    # 20% same-title cap of a $100 portfolio
    "a same-title pair past its size cap": (
        lambda: _TrimCase("same_title", a_now=(0.50, 0.52), b_now=(0.42, 0.60)), 10_000, {}),
    # Selling returns more than the pair is expected to pay: nothing is kept
    "a converged same-title pair": (
        lambda: _TrimCase("same_title", a_now=(0.03, 0.98), b_now=(0.03, 0.98)), 10_000, {}),
    # Market A's YES ask (0.40) sits below its own YES bid (0.45)
    "a crossed book": (
        lambda: _TrimCase(a_now=(0.40, 0.55)), 10_000, {"cap": 0.10}),
    # The bids hold 14 contracts a market, fewer than the 19 the best bids want sold
    "a thin ladder": (
        lambda: _TrimCase(model=_ladder_model(((0.0, 5.0), (0.01, 5.0), (0.02, 4.0)))),
        10_000, {"cap": 0.10}),
    # Five contracts at each best bid, the rest 20c below it: the last distance
    # the backtest's modeled ladder reaches, which live must not leave out
    "a ladder that reaches its last modeled distance": (
        lambda: _TrimCase(b_now=(0.45, 0.56),
                          model=_ladder_model(((0.0, 5.0), (0.20, 100.0)))),
        10_000, {"cap": 0.10}),
    # A YES bid of 1c: a candle's NO ask of 0.99 is no quote, so neither side sees a bid
    "a 1c YES bid on market A": (
        lambda: _TrimCase(a_now=(0.02, 0.99)), 10_000, {"cap": 0.10}),
    "a 1c YES bid on market B": (
        lambda: _TrimCase(b_now=(0.02, 0.99)), 10_000, {"cap": 0.10}),
}

# How many contract pairs each case sells, so no case passes by selling nothing
# on both sides when it is meant to sell
_TRIM_PARITY_SELLS = {
    "a time-series pair past its size cap": "part",
    "a time-series pair past its Kelly share, under its cap": "part",
    "a time-series pair within its share": "none",
    "a same-title pair past its size cap": "part",
    "a converged same-title pair": "all",
    "a crossed book": "none",
    "a thin ladder": "part",
    "a ladder that reaches its last modeled distance": "part",
    "a 1c YES bid on market A": "none",
    "a 1c YES bid on market B": "none",
}


class TestTrimParityWithTheBacktest:
    """seller.plan_trims sells exactly the contract pairs backtester._kelly_trim
    sells of the same pair at the same quotes, portfolio value, k and caps,
    and values the part sale as the backtest does."""

    @pytest.mark.parametrize("name", list(_TRIM_PARITY_CASES))
    def test_the_same_count_and_the_same_sale(self, monkeypatch, name):
        build, value_cents, caps = _TRIM_PARITY_CASES[name]
        case = build()
        decision, sale = case.backtest(value_cents, **caps)
        plans = case.live(monkeypatch, value_cents, **caps)
        assert [plan.count for plan in plans] == ([decision.sell] if decision.sell else [])
        expected = {"none": 0, "all": case.n}.get(_TRIM_PARITY_SELLS[name])
        if expected is None:
            assert 0 < decision.sell < case.n, name
        else:
            assert decision.sell == expected, name
        if not decision.sell:
            return
        (plan,) = plans
        assert plan.kind == config.SALE_KIND_KELLY_TRIM and plan.level is None
        # What the account holds is on each leg; the plan's counts are the part sold
        assert [(leg.count, leg.held) for leg in plan.legs] == [(decision.sell, case.n)] * 2
        # The backtest's own valuation of selling that many
        sold, per_trade = sale
        assert sold == [decision.sell]
        ((sale_value, _fees, (price_a, price_b)),) = per_trade
        _, value, cost, potential = backtester._position_sale_value(
            [case.trade], _D, sold=[decision.sell])
        assert plan.proceeds_dollars == pytest.approx(sale_value, abs=1e-9)
        assert plan.proceeds_dollars == pytest.approx(value, abs=1e-9)
        assert plan.cost_dollars == pytest.approx(cost, abs=1e-9)
        assert plan.profits == ((pytest.approx(value - cost, abs=1e-9),
                                 pytest.approx(potential - cost, abs=1e-9)),)
        assert plan.walked[_A][0] == pytest.approx(price_a, abs=1e-12)
        assert plan.walked[_B][0] == pytest.approx(price_b, abs=1e-12)

    def test_a_thin_ladder_sells_fewer_than_the_best_bids_want(self, monkeypatch):
        # Not vacuous: with deep bids the same pair sells 19, and the ladder's 14
        # contracts a market hold the thin one to fewer
        _, value_cents, caps = _TRIM_PARITY_CASES["a thin ladder"]
        deep = _TrimCase().backtest(value_cents, **caps)[0].sell
        thin_case = _TRIM_PARITY_CASES["a thin ladder"][0]()
        thin = thin_case.backtest(value_cents, **caps)[0].sell
        assert (deep, thin) == (19, 14)
        (plan,) = thin_case.live(monkeypatch, value_cents, **caps)
        assert plan.count == 14
        # The walk reached the ladder's third level on both markets
        assert plan.walked[_A][1] == pytest.approx(0.40) and plan.walked[_B][1] == pytest.approx(0.50)
        assert plan.walked[_A][0] < 0.42 and plan.walked[_B][0] < 0.52

    def test_the_deepest_level_the_backtest_models_is_walked_live(self, monkeypatch):
        # Both sides' second level sits exactly 20c under the best bid. In
        # floats 0.55 - 0.20 is a hair above 0.35, so without its allowance
        # for float noise the live cut would leave market B's level out and
        # sell 5 where the backtest sells 15
        build, value_cents, caps = _TRIM_PARITY_CASES[
            "a ladder that reaches its last modeled distance"]
        case = build()
        assert case.world.books[_A]["yes"] == [[0.42, 5.0], [0.22, 100.0]]
        assert case.world.books[_B]["no"] == [[0.55, 5.0], [0.35, 100.0]]
        assert 0.55 - seller.KELLY_TRIM_MAX_BID_DROP > 0.35
        (plan,) = case.live(monkeypatch, value_cents, **caps)
        assert plan.count == case.backtest(value_cents, **caps)[0].sell == 15
        assert plan.walked[_A][1] == pytest.approx(0.22)
        assert plan.walked[_B][1] == pytest.approx(0.35)
        assert plan.ladders == {_A: [[0.42, 5.0], [0.22, 100.0]],
                                _B: [[0.55, 5.0], [0.35, 100.0]]}

    @pytest.mark.parametrize("ticker, side, book, sells, lowest", [
        # One contract at the best bid, then nothing until a stub bid far below
        (_A, "yes", [[0.42, 1.0], [0.02, 100_000.0]], 1, 0.42),
        (_A, "yes", [[0.42, 5.0], [0.10, 100_000.0]], 5, 0.42),
        # A bid exactly the reach below the best bid is still walked; a cent lower is not
        (_A, "yes", [[0.42, 1.0], [0.22, 100_000.0]], 17, 0.22),
        (_A, "yes", [[0.42, 1.0], [0.21, 100_000.0]], 1, 0.42),
        (_B, "no", [[0.52, 1.0], [0.32, 100_000.0]], 17, 0.32),
        (_B, "no", [[0.52, 1.0], [0.31, 100_000.0]], 1, 0.52),
    ])
    def test_a_trim_does_not_sell_into_a_gap_under_the_best_bid(
            self, monkeypatch, ticker, side, book, sells, lowest):
        # The backtest's modeled ladder ends 20c below the best bid, so it can
        # never sell lower. Live leaves out the bids further below than that,
        # and sells only what the bids within reach hold (the best bids alone
        # want 19 sold)
        assert seller.KELLY_TRIM_MAX_BID_DROP == config.DEPTH_MODEL_DISTANCES[-1] == 0.20
        case = _TrimCase()
        case.world.books[ticker][side] = book
        (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        assert plan.count == sells
        assert [leg.count for leg in plan.legs] == [sells, sells]
        assert plan.walked[ticker][1] == pytest.approx(lowest)
        # The ladder the trader prices the second order on ends there too
        assert min(price for price, _size in plan.ladders[ticker]) == pytest.approx(lowest)
        assert all(price >= 0.42 - 0.20 - 1e-9 for price, _size in plan.ladders[_A])
        # The other market sells at its one deep best bid
        other = _B if ticker == _A else _A
        assert plan.walked[other][0] == plan.walked[other][1]

    def test_the_reach_under_the_best_bid_is_sellers_own_binding(self, monkeypatch):
        # With the reach widened to 40c the stub bid at 2c is within it, and the
        # rule, left to the whole book under a 10% cap, sells 12 pairs into it:
        # the sale the 20c reach exists to stop
        case = _TrimCase()
        case.world.books[_A]["yes"] = [[0.42, 1.0], [0.02, 100_000.0]]
        monkeypatch.setattr(seller, "KELLY_TRIM_MAX_BID_DROP", 0.40)
        (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        assert plan.count == 12 and plan.walked[_A][1] == pytest.approx(0.02)
        # A reach of 0 sells only at the best bid's own price
        monkeypatch.setattr(seller, "KELLY_TRIM_MAX_BID_DROP", 0)
        (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        assert plan.count == 1 and plan.ladders[_A] == [[0.42, 1.0]]
        # A value that is no distance stops the plan, before any order exists
        monkeypatch.setattr(seller, "KELLY_TRIM_MAX_BID_DROP", None)
        with pytest.raises(ValueError, match="KELLY_TRIM_MAX_BID_DROP"):
            case.live(monkeypatch, 10_000, cap=0.10)

    def test_the_live_price_rise_is_the_backtests(self, monkeypatch):
        # The pair's total price (YES ask on A plus NO ask on B) was 0.77 when it
        # was bought and is 0.97 now. The backtest reads both from the candles
        # at its two checkpoints; live reads "then" from the candle at the
        # pair's last fill and "now" from the order books just read
        case = _TrimCase(a_then=(0.34, 0.68), b_then=(0.58, 0.43))
        expected = backtester._price_rise([case.trade], _D, {})
        assert expected == pytest.approx(0.20, abs=1e-9)
        measured = []
        real = seller.pair_price_rise

        def spy(then, now):
            measured.append(real(then, now))
            return measured[-1]

        monkeypatch.setattr(seller, "pair_price_rise", spy)
        # A 20c rise meets a 20c minimum, on both sides
        assert config.trim_rise_reached(expected, 0.20)
        (plan,) = case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.20)
        assert measured == [pytest.approx(expected, abs=1e-9)]
        assert plan.count == case.backtest(10_000, cap=0.10)[0].sell == 19
        # ... and falls short of a 25c one, on both sides
        assert not config.trim_rise_reached(expected, 0.25)
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.25) == []
        # Each market's price then was read from its candles over the week
        # before the pair's last fill
        week = config.TRIM_PRICE_THEN_MAX_AGE_DAYS * _DAY
        assert case.world.calls["candles"][:2] == [
            (_A, _EV_A, _FIRST - week - 3600, _FIRST), (_B, _EV_B, _FIRST - week - 3600, _FIRST)]


# ─── The wait after a trade, however often the bot runs ───────────────────────

def _pacific(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """A moment of 2026 on the bot's own clock (Los Angeles time)."""
    return datetime(2026, month, day, hour, minute, tzinfo=_PACIFIC)


def _waiting_case(*fills_at: datetime, only: str | None = None) -> _TrimCase:
    """The standard pair, which a run trims (19 of 30 under a 10% cap) once it
    has rested: its markets last traded at the latest of `fills_at` (both of
    them, or `only` the named one, the other at the first)."""
    case = _TrimCase()
    first, last = min(fills_at), max(fills_at)
    latest = {_A: last, _B: last}
    if only is not None:
        latest = {_A: first, _B: first, only: last}
    case.world.fills = _fills(latest, oldest=first - timedelta(days=30))
    return case


def _trims_at(case: _TrimCase, monkeypatch, now: datetime) -> list:
    """The run's trims at `now`, sized so the standard pair sells once it has rested."""
    return case.live(monkeypatch, 10_000, cap=0.10, now=now)


class TestTrimWaitAtAnyFrequency:
    """A pair is trimmed only once KELLY_TRIM_MIN_DAYS_SINCE_TRADE (7) calendar
    days, on the bot's own clock, have passed since the last fill on either of
    its markets. The rule reads two dates and no schedule, so it holds whether
    the bot runs weekly, daily or several times in one day."""

    @pytest.mark.parametrize("days_after", range(1, 9))
    def test_a_run_on_each_of_the_eight_days_after_a_purchase(self, monkeypatch, caplog,
                                                              days_after):
        # Bought Tuesday January 6 at 10:00; a run at 09:00 on each later day
        case = _waiting_case(_pacific(1, 6, 10))
        with caplog.at_level(logging.INFO):
            plans = _trims_at(case, monkeypatch, _pacific(1, 6 + days_after, 9))
        # Days 1 to 6: left alone. Day 7 (the next Tuesday) and day 8: trimmed
        assert bool(plans) == (days_after >= 7)
        if plans:
            assert plans[0].count == 19
            assert f"last traded 2026-01-06, {days_after} day(s) ago" in caplog.text
        else:
            assert (f"it last traded 2026-01-06, {days_after} day(s) ago, and a trim waits "
                    "until 7 day(s) have passed") in caplog.text
            # Refused before any order book is read
            assert case.world.calls["book"] == []

    @pytest.mark.parametrize("what, only", [
        ("an add-on (a fill on both markets)", None),
        ("a trim (a fill on both markets)", None),
        ("a fill by hand on market A alone", _A),
        ("a fill by hand on market B alone", _B),
    ])
    def test_any_fill_on_day_three_moves_the_first_trim_to_day_ten(self, monkeypatch, what,
                                                                    only):
        # Bought on January 6; on January 9 (day 3) the pair trades again.
        # Kalshi's fills list does not say why, and the rule does not ask
        case = _waiting_case(_pacific(1, 6, 10), _pacific(1, 9, 14), only=only)
        for day, trims in ((13, False), (15, False), (16, True), (17, True)):
            assert bool(_trims_at(case, monkeypatch, _pacific(1, day, 9))) == trims, (what, day)

    def test_a_second_run_the_same_day_sees_the_firsts_fills(self, monkeypatch):
        case = _waiting_case(_pacific(1, 6, 10))
        morning = _pacific(1, 13, 9)
        assert _trims_at(case, monkeypatch, morning)
        # The morning run's trim filled at 09:01 on both markets
        case.world.fills = _fills({_A: morning + timedelta(minutes=1),
                                   _B: morning + timedelta(minutes=1)})
        assert _trims_at(case, monkeypatch, _pacific(1, 13, 15)) == []
        # ... and the pair rests again until the same weekday a week later
        assert _trims_at(case, monkeypatch, _pacific(1, 19, 15)) == []
        assert _trims_at(case, monkeypatch, _pacific(1, 20, 6))

    def test_late_evening_to_just_after_midnight_counts_calendar_days(self, monkeypatch):
        # A fill at 11:50pm on January 5; a run at 12:10am seven dates later is
        # allowed (a little under 7 x 24 hours), six dates later is not
        case = _waiting_case(_pacific(1, 5, 23, 50))
        assert _trims_at(case, monkeypatch, _pacific(1, 11, 0, 10)) == []
        assert _trims_at(case, monkeypatch, _pacific(1, 12, 0, 10))

    def test_an_evening_fill_belongs_to_its_pacific_day(self, monkeypatch):
        # A fill at 6pm Pacific on Monday January 5 is already Tuesday in UTC;
        # the run at 9am the next Monday still finds seven days gone
        bought = _pacific(1, 5, 18)
        assert bought.astimezone(UTC).date() == date(2026, 1, 6)
        case = _waiting_case(bought)
        assert _trims_at(case, monkeypatch, _pacific(1, 12, 9))

    def test_the_days_follow_the_schedules_own_zone(self, monkeypatch):
        # The same moments with the bot's clock set to UTC: the 6pm Pacific fill
        # is a Tuesday there, so the next Monday morning is only six days on
        utc_run = config.ScheduledRun(weekday=0, hour=9, minute=0, timezone="UTC")
        evening = _waiting_case(_pacific(1, 5, 18))
        late = _waiting_case(_pacific(1, 5, 23, 50))
        # On UTC's clock, where Pacific's says no: a fill at 3pm Pacific on
        # January 5 and a run at 5pm Pacific on January 11 (January 12 in UTC)
        afternoon = _waiting_case(_pacific(1, 5, 15))
        assert _trims_at(afternoon, monkeypatch, _pacific(1, 11, 17)) == []
        monkeypatch.setattr(seller, "SCHEDULED_RUN", utc_run)
        assert _trims_at(evening, monkeypatch, _pacific(1, 12, 9)) == []
        assert _trims_at(late, monkeypatch, _pacific(1, 12, 0, 10)) == []
        assert _trims_at(afternoon, monkeypatch, _pacific(1, 11, 17))
        # config's own schedule is not what the seller reads
        assert config.SCHEDULED_RUN.timezone == "America/Los_Angeles"


# ─── plan_trims: each gate and each step ──────────────────────────────────────

_NO_READS = {"book": [], "candles": [], "settlements": 0, "lookup": [], "min_ts": []}
_PAIR_TEXT = f"YES {_A} / NO {_B}"


def _warnings(caplog) -> list[str]:
    """Every WARNING line logged."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _weather_sell(n: int, value: float, cap: float) -> int:
    """What the shared rule sells of a weather pair (_weather_pair) of `n` contract
    pairs: config.kelly_trim_count, handed what that pair's books show."""
    win = config.held_pair_win_prob("time_series", 0.44, 0.58, 0.48, 0.53, 0.8)
    top = 0.42 + 0.52 - config.fee_per_pair_approx(0.42, 0.52)

    def net_sale(pairs):
        return (pairs * 0.42 - fee_leg_exact(pairs, 0.42)
                + pairs * 0.52 - fee_leg_exact(pairs, 0.52))

    return config.kelly_trim_count(n, value, win, cap, top, net_sale).sell


def _weather_pair(world: _World, name: str, n: int, *, quotes: bool = True,
                  yes_bid: str = "0.42", no_bid: str = "0.52") -> tuple[str, str]:
    """Hold one more time-series pair, on a ladder of its own: YES on the
    January 20 market and NO on the January 25 one, `n` contracts each, with
    deep books at those best bids and, with `quotes`, the same bids on the
    run's market list. Both markets last traded when the fills list says."""
    tickers = []
    for day, close, held, count in ((20, _A_CLOSE, "yes", n), (25, _B_CLOSE, "no", -n)):
        ticker, event = f"KX{name.upper()}{day}-26", f"KX{name.upper()}{day}"
        market = ApiMarket(
            ticker=ticker, event_ticker=event, title=f"Will it {name} by Jan {day}, 2026?",
            subtitle="", status="active", close_time=close,
            # The list's YES bid, and the YES ask whose complement is the NO bid
            yes_bid_dollars=yes_bid if quotes else None,
            yes_ask_dollars=str(round(1 - float(no_bid), 4)) if quotes else None)
        world.hold(market, count, n * 0.35, 0.50)
        # The side not held: A's NO bid (so its YES ask is 0.44) and B's YES bid
        other = "0.56" if held == "yes" else "0.47"
        world.books[ticker] = {"yes": [[yes_bid if held == "yes" else other, "100000"]],
                               "no": [[no_bid if held == "no" else other, "100000"]]}
        tickers.append(ticker)
    return tickers[0], tickers[1]


class TestPlanTrims:
    """seller.plan_trims: nothing is trimmed unless every run-level gate holds,
    each pair stops at its first failed step with one line saying why, and a
    trim's plan names the contracts sold apart from the contracts held."""

    def test_the_standard_pair_is_sold_down_to_its_cap(self, monkeypatch, caplog):
        case = _TrimCase()
        with caplog.at_level(logging.INFO):
            (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        lines = [r.getMessage() for r in caplog.records]
        assert ("Kelly trim sizes on $100.00 = cash $50.00 + open positions $50.00 "
                "(Kalshi's value $50.00; $0.00 at this run's bids)") in lines
        assert ("Kelly trim: a pair is looked at once 7 day(s) have passed since its last "
                "trade, counted in calendar days on the bot's own clock "
                "(America/Los_Angeles), where today is 2026-01-12") in lines
        assert (f"Kelly trim check: {_PAIR_TEXT} (time-series), 30 each, last traded "
                "2026-01-05, 7 day(s) ago, pays 96.4%, $0.91 a pair at the best bids, "
                "may keep 10.0% of $100.00 -> sell 19") in lines
        assert lines[-1] == "Pairs to trim this run: 1 of 1"
        assert _warnings(caplog) == []
        # The plan: the 19 sold, with the 30 held on each leg
        assert plan.title == "Will the rocket launch by Jan 20, 2026?"
        assert (plan.kind, plan.level, plan.count, plan.days_left) == (
            config.SALE_KIND_KELLY_TRIM, None, 19, 13)
        assert [(leg.ticker, leg.side, leg.count, leg.held) for leg in plan.legs] == [
            (_A, "yes", 19, 30), (_B, "no", 19, 30)]
        assert all(leg.market is case.world.markets[leg.ticker] for leg in plan.legs)
        # Each leg's cost is the sold share of what its market cost, fees included
        cost_a = (30 * 0.30 + fee_leg_exact(30, 0.30)) * 19 / 30
        cost_b = (30 * 0.40 + fee_leg_exact(30, 0.40)) * 19 / 30
        assert [leg.cost_dollars for leg in plan.legs] == [pytest.approx(cost_a),
                                                           pytest.approx(cost_b)]
        assert plan.cost_dollars == pytest.approx(cost_a + cost_b)
        # Sold at the best bids, less the fee on each market's sale of 19
        assert plan.ladders == {_A: [[0.42, 100_000.0]], _B: [[0.52, 100_000.0]]}
        assert plan.walked == {_A: (0.42, 0.42), _B: (0.52, 0.52)}
        proceeds = (19 * 0.42 - fee_leg_exact(19, 0.42)) + (19 * 0.52 - fee_leg_exact(19, 0.52))
        assert plan.proceeds_dollars == pytest.approx(proceeds, abs=1e-12)
        assert plan.profits == ((pytest.approx(proceeds - plan.cost_dollars),
                                 pytest.approx(19 * 1.0 - plan.cost_dollars)),)
        # One fills read, each book once, and no candle (no minimum price rise)
        assert case.world.fill_reads == 1
        assert case.world.calls["book"] == [_A, _B] and case.world.calls["candles"] == []

    def test_the_count_is_the_shared_rules(self, monkeypatch):
        # config.kelly_trim_count, handed what the books show
        case = _TrimCase()
        (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        win = config.held_pair_win_prob("time_series", 0.44, 0.58, 0.48, 0.53, 0.8)
        top = 0.42 + 0.52 - config.fee_per_pair_approx(0.42, 0.52)

        def net_sale(pairs):
            return (pairs * 0.42 - fee_leg_exact(pairs, 0.42)
                    + pairs * 0.52 - fee_leg_exact(pairs, 0.52))

        assert plan.count == config.kelly_trim_count(30, 100.0, win, 0.10, top, net_sale).sell

    def test_the_plan_is_one_the_trader_and_the_reporter_read(self, monkeypatch, caplog):
        # The real plan, handed on as a run would hand it: the trader builds
        # its two orders from it (a dry run sends neither), and the reporter
        # writes its row and its record
        from kalshi_betting import reporter, trader

        (plan,) = _TrimCase().live(monkeypatch, 10_000, cap=0.10)
        legs, problem = trader._sale_legs(plan)
        assert problem is None and sorted(leg.ticker for leg in legs) == [_A, _B]
        assert [trader._plan_holding(leg) for leg in legs] == [30, 30]
        with caplog.at_level(logging.INFO):
            [result] = trader.sell_positions(MagicMock(), [plan], dry_run=True)
        assert (result.status, result.sold) == ("simulated", {_A: 19, _B: 19})
        assert ("19 YES contracts (of 30 held) on " + _A) in caplog.text
        assert ("19 NO contracts (of 30 held) on " + _B) in caplog.text
        fees = fee_leg_exact(19, 0.42) + fee_leg_exact(19, 0.52)
        row = reporter._sale_to_row(result, _NOW)
        assert row[17] == ("[sale: YES A / NO B, Kelly trim: 19 of 30 contract pairs, "
                           f"13 days left, fees=${fees:.2f}] ")
        assert row[11:13] == [19, 19] and row[13] == round(plan.cost_dollars, 2)
        [record] = reporter.report_sales([result])
        assert (record.kind, record.level, record.realized_percent) == (
            config.SALE_KIND_KELLY_TRIM, None, None)
        assert [(leg.ticker, leg.held, leg.sold) for leg in record.legs] == [
            (_A, 30, 19), (_B, 30, 19)]
        assert record.profit == round(plan.profits[0][0], 2)

    def test_a_pair_within_its_share_is_kept(self, monkeypatch, caplog):
        case = _TrimCase()
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 100_000) == []
        assert ("$0.91 a pair at the best bids, may keep 61.9% of $1000.00 -> keep"
                in caplog.text)
        assert "Pairs to trim this run: 0 of 1" in caplog.text

    def test_a_pair_sold_whole_still_names_what_it_held(self, monkeypatch):
        build, value_cents, caps = _TRIM_PARITY_CASES["a converged same-title pair"]
        (plan,) = build().live(monkeypatch, value_cents, **caps)
        assert [(leg.ticker, leg.side, leg.count, leg.held) for leg in plan.legs] == [
            (_A, "no", 30, 30), (_B, "yes", 30, 30)]
        assert plan.kind == config.SALE_KIND_KELLY_TRIM

    # ── Switched off, and what stops the run ─────────────────────────────────

    def test_off_reads_nothing_and_checks_nothing(self, monkeypatch, caplog):
        case = _TrimCase()
        # Not even the number of days is checked: a run that does not trim never stops on it
        monkeypatch.setattr(seller, "KELLY_TRIM_MIN_DAYS_SINCE_TRADE", 99)
        with caplog.at_level(logging.DEBUG):
            assert case.world.trims(monkeypatch, _trim_settings(on=False)) == []
        assert case.world.calls == _NO_READS and case.world.fill_reads == 0
        assert caplog.records == []

    @pytest.mark.parametrize("days", [True, 7.5, -1, 31, None, "7"])
    def test_a_bad_number_of_days_stops_the_run_before_any_request(self, monkeypatch, days):
        monkeypatch.setattr(seller, "KELLY_TRIM_MIN_DAYS_SINCE_TRADE", days)
        case = _TrimCase()
        with pytest.raises(ValueError, match="KELLY_TRIM_MIN_DAYS_SINCE_TRADE must be a "
                                             "whole number from 0 to 30"):
            case.live(monkeypatch, 10_000, cap=0.10)
        assert case.world.calls == _NO_READS and case.world.fill_reads == 0

    def test_now_must_carry_a_time_zone(self, monkeypatch):
        with pytest.raises(ValueError, match="timezone-aware"):
            _TrimCase().live(monkeypatch, 10_000, now=_NOW.replace(tzinfo=None))

    @pytest.mark.parametrize("cash", [None, True, -1, float("nan"), float("inf"), "5000"])
    def test_cash_that_is_not_an_amount_is_refused(self, monkeypatch, cash):
        with pytest.raises(ValueError, match="cash_cents must be a finite number"):
            _TrimCase().world.trims(monkeypatch, cash_cents=cash)

    # ── The run-level gates ──────────────────────────────────────────────────

    def test_nothing_held_trims_nothing(self, monkeypatch, caplog):
        world = _World()
        with caplog.at_level(logging.INFO):
            assert world.trims(monkeypatch, positions_value_cents=None) == []
        assert [r.getMessage() for r in caplog.records] == ["Pairs to trim this run: 0 of 0"]
        assert world.fill_reads == 0

    @pytest.mark.parametrize("value", [None, 0, -5, float("nan"), float("inf"), True, "5000"])
    def test_an_unknown_positions_value_trims_nothing(self, monkeypatch, caplog, value):
        # A trim grows as the value shrinks, so a missing value is never read as $0
        case = _TrimCase()
        with caplog.at_level(logging.INFO):
            assert case.world.trims(monkeypatch, positions_value_cents=value) == []
        assert _warnings(caplog) == [
            "Not trimming any held pair this run: Kalshi's value of the open positions is "
            "not known or is not above $0, and a trim sized on too small a value would "
            "sell too much"]
        assert case.world.calls == _NO_READS and case.world.fill_reads == 0

    def test_an_unknown_ladder_trims_nothing(self, monkeypatch, caplog):
        case = _TrimCase()
        labels = case.world.labels()
        labels[_B] = frozenset()
        with caplog.at_level(logging.INFO):
            assert case.world.trims(monkeypatch, labels=labels) == []
        assert (f"Not trimming this run: the ladder of 1 held market(s) is unknown ({_B})"
                in caplog.text)
        assert case.world.calls == _NO_READS and case.world.fill_reads == 0

    def test_a_lone_held_market_is_never_trimmed(self, monkeypatch, caplog):
        world = _lone_world(_settled(_SAME_EVENT, _EV_B))
        world.fills = _fills()
        with caplog.at_level(logging.INFO):
            assert world.trims(monkeypatch) == []
        assert "Pairs to trim this run: 0 of 0" in caplog.text
        assert world.calls == _NO_READS and world.fill_reads == 0

    @pytest.mark.parametrize("skip", [frozenset({_A}), frozenset({_B}), frozenset({_A, _B, "X"})])
    def test_a_pair_picked_for_a_take_profit_sale_is_not_looked_at(self, monkeypatch, caplog,
                                                                    skip):
        case = _TrimCase()
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, skip=skip) == []
        assert (f"Not trimming {_PAIR_TEXT}: it is picked for a take-profit sale this run"
                in caplog.text)
        assert "Pairs to trim this run: 0 of 0" in caplog.text
        assert case.world.calls == _NO_READS and case.world.fill_reads == 0
        # A ticker of another position skips nothing
        assert case.live(monkeypatch, 10_000, cap=0.10, skip=frozenset({"X"}))

    def test_unreadable_fills_trim_nothing(self, monkeypatch, caplog):
        case = _TrimCase()
        case.world.fills = None
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert _warnings(caplog) == [
            "Not trimming any held pair this run: the account's fills could not be read, "
            "so when each pair last traded is not known"]
        assert "Pairs to trim this run: 0 of 1" in caplog.text
        assert case.world.fill_reads == 1 and case.world.calls == _NO_READS

    def test_an_empty_fills_list_trims_nothing(self, monkeypatch, caplog):
        case = _TrimCase()
        case.world.fills = _fills(oldest=None)
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert _warnings(caplog) == [
            "Not trimming any held pair this run: the account's fills list is empty, so a "
            "trade in the last 7 day(s) cannot be ruled out"]
        assert case.world.calls == _NO_READS

    @pytest.mark.parametrize("days_back, trims", [(3, False), (6, False), (7, True), (30, True)])
    def test_the_fills_list_must_reach_back_past_the_wait(self, monkeypatch, caplog,
                                                          days_back, trims):
        # Neither market has a fill listed, so the pair last traded no later than
        # the list's oldest fill: the list must start at least 7 days ago
        case = _TrimCase()
        case.world.fills = _fills(oldest=_NOW - timedelta(days=days_back))
        with caplog.at_level(logging.INFO):
            plans = case.live(monkeypatch, 10_000, cap=0.10)
        assert bool(plans) == trims
        start = (_NOW - timedelta(days=days_back)).astimezone(_PACIFIC).date()
        if trims:
            assert f"30 each, no fill listed (the fills list starts {start}), pays" in caplog.text
        else:
            assert _warnings(caplog) == [
                f"Not trimming any held pair this run: the account's fills list starts {start} "
                f"on the bot's own clock, fewer than 7 day(s) ago, so a trade in the last 7 "
                "day(s) cannot be ruled out"]
            assert case.world.calls == _NO_READS

    def test_the_fills_lists_start_is_a_day_on_the_bots_own_clock(self, monkeypatch, caplog):
        # 01:00 UTC on January 6 is still January 5 in Los Angeles: seven days
        # before a run on the 12th there, though only six UTC dates before it
        case = _TrimCase()
        case.world.fills = _fills(oldest=datetime(2026, 1, 6, 1, tzinfo=UTC))
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10)
        assert "no fill listed (the fills list starts 2026-01-05)" in caplog.text

    # ── The wait, per pair ───────────────────────────────────────────────────

    @pytest.mark.parametrize("market", [_A, _B])
    def test_a_fill_on_either_market_six_days_ago_blocks(self, monkeypatch, caplog, market):
        case = _TrimCase()
        recent = _NOW - timedelta(days=6)
        case.world.fills = _fills({_A: _BOUGHT, _B: _BOUGHT, market: recent})
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert (f"Not trimming {_PAIR_TEXT}: it last traded 2026-01-06, 6 day(s) ago, and a "
                "trim waits until 7 day(s) have passed since a pair's last trade"
                in caplog.text)
        assert case.world.calls["book"] == []
        # Seven days ago on that market: trimmed
        case.world.fills = _fills({_A: _BOUGHT, _B: _BOUGHT, market: _NOW - timedelta(days=7)})
        assert case.live(monkeypatch, 10_000, cap=0.10)

    def test_a_fill_listed_on_one_market_alone_is_the_pairs_last_trade(self, monkeypatch,
                                                                       caplog):
        case = _TrimCase()
        case.world.fills = _fills({_B: _NOW - timedelta(days=9)})
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10)
        assert "30 each, last traded 2026-01-03, 9 day(s) ago, pays" in caplog.text

    def test_a_fill_dated_after_the_run_blocks(self, monkeypatch, caplog):
        case = _TrimCase()
        case.world.fills = _fills({_A: _NOW + timedelta(days=1)})
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert ("it last traded 2026-01-13, after today (2026-01-12) on the bot's own clock"
                in caplog.text)

    def test_a_wait_of_eight_days_blocks_a_fill_seven_days_old(self, monkeypatch):
        monkeypatch.setattr(seller, "KELLY_TRIM_MIN_DAYS_SINCE_TRADE", 8)
        case = _TrimCase()
        assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert case.live(monkeypatch, 10_000, cap=0.10, now=_NOW + timedelta(days=1))
        # config's own number is not what the seller reads
        assert config.KELLY_TRIM_MIN_DAYS_SINCE_TRADE == 7

    def test_no_wait_asks_for_no_fills(self, monkeypatch, caplog):
        monkeypatch.setattr(seller, "KELLY_TRIM_MIN_DAYS_SINCE_TRADE", 0)
        case = _TrimCase()
        # Traded a minute ago, and the fills list could not even be read
        case.world.fills = None
        with caplog.at_level(logging.INFO):
            (plan,) = case.live(monkeypatch, 10_000, cap=0.10)
        assert plan.count == 19 and case.world.fill_reads == 0
        assert f"Kelly trim check: {_PAIR_TEXT} (time-series), 30 each, no wait, pays" in (
            caplog.text)
        assert "a pair is looked at once" not in caplog.text

    # ── The value a trim is sized on ─────────────────────────────────────────

    @pytest.mark.parametrize("share", [0.10, 0.50])
    def test_a_kalshi_value_below_the_bids_is_raised_to_them(self, monkeypatch, caplog, share):
        # Three pairs, 450 contracts a side in all, whose markets the run's list
        # quotes at 0.42 (YES bid) and 0.52 (NO bid): they would fetch $423 at
        # those bids. Kalshi values them at a tenth, or a half, of that
        world = _World()
        world.fills = _fills()
        held = {"hail": 200, "rain": 100, "snow": 150}
        for name, n in held.items():
            _weather_pair(world, name, n)
        floor = 450 * 0.42 + 450 * 0.52
        cash = 100_000
        settings = _trim_settings(cap=0.05)
        with caplog.at_level(logging.INFO):
            low = world.trims(monkeypatch, settings, cash_cents=cash,
                              positions_value_cents=int(floor * 100 * share))
        assert _warnings(caplog) == [
            f"Kalshi's value of the open positions (${floor * share:.2f}) is below what they "
            f"would fetch at this run's bids (${floor:.2f}): the Kelly trim is sized on the "
            "larger, so it can only sell less"]
        assert (f"Kelly trim sizes on ${cash / 100 + floor:.2f} = cash $1000.00 + open "
                f"positions ${floor:.2f} (Kalshi's value ${floor * share:.2f}; ${floor:.2f} "
                "at this run's bids)") in caplog.text
        # Sized as if Kalshi had valued them at what they fetch (a cent over,
        # so its value is plainly the larger and nothing is warned about)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            at_floor = world.trims(monkeypatch, settings, cash_cents=cash,
                                   positions_value_cents=math.ceil(floor * 100) + 1)
        assert _warnings(caplog) == []
        assert [(p.legs[0].ticker, p.count) for p in low] == [
            (p.legs[0].ticker, p.count) for p in at_floor]
        # ... which is what the shared rule sells of each pair at cash plus the floor
        value = cash / 100 + floor
        assert [p.count for p in low] == [_weather_sell(n, value, 0.05) for n in held.values()]
        # Not vacuous: sized on Kalshi's lower value, every pair would sell more
        understated = cash / 100 + floor * share
        for plan, n in zip(low, held.values(), strict=True):
            assert 0 < plan.count < _weather_sell(n, understated, 0.05) < n

    def test_a_pair_worth_more_than_every_open_position_stops_every_trim(self, monkeypatch,
                                                                         caplog):
        # The rain pair would trim. The snow pair's markets are missing their
        # quotes on the run's list, so the floor counts them at $0, and its
        # order books put it alone at about $36, above the $30 Kalshi values
        # every open position at: that value cannot be right
        world = _World()
        world.fills = _fills()
        _weather_pair(world, "rain", 30)
        _weather_pair(world, "snow", 40, quotes=False)
        settings = _trim_settings(cap=0.05)
        with caplog.at_level(logging.INFO):
            assert world.trims(monkeypatch, settings, cash_cents=5_000,
                               positions_value_cents=3_000) == []
        (warning,) = _warnings(caplog)
        top = 0.42 + 0.52 - config.fee_per_pair_approx(0.42, 0.52)
        assert warning == (
            "Not trimming any held pair this run: YES KXSNOW20-26 / NO KXSNOW25-26 alone "
            f"would fetch ${40 * top:.2f} at its best bids (40 contract pairs at ${top:.2f} "
            "each), more than the $30.00 all open positions are valued at, so that value "
            "cannot be right, and a trim sized on too small a value would sell too much")
        assert "Pairs to trim this run: 0 of 2" in caplog.text
        # The rain pair was judged, and would have been trimmed
        assert "YES KXRAIN20-26 / NO KXRAIN25-26 (time-series), 30 each" in caplog.text
        assert "-> sell" in caplog.text
        # With a value that covers the snow pair, both trim
        assert len(world.trims(monkeypatch, settings, cash_cents=5_000,
                               positions_value_cents=4_000)) == 2

    def test_a_minimum_price_rise_never_lets_through_a_trim_the_value_guard_stops(
            self, monkeypatch, caplog):
        # The snow pair alone fetches about $36, above the $30 every open position is
        # valued at, so with no minimum nothing is trimmed. A minimum may only hold a
        # trim back: the rain pair (risen 20c) must not be trimmed on that value
        # because the snow pair's own price has not risen
        world = _World()
        rain = _weather_pair(world, "rain", 30)
        snow = _weather_pair(world, "snow", 40, quotes=False)
        world.fills = _fills(dict.fromkeys((*rain, *snow), _BOUGHT))
        world.candles = {rain[0]: [_candle(_FIRST, 0.34, 0.68)],
                         rain[1]: [_candle(_FIRST, 0.58, 0.43)],
                         snow[0]: [_candle(_FIRST, 0.44, 0.58)],
                         snow[1]: [_candle(_FIRST, 0.48, 0.53)]}
        sized = {"cash_cents": 5_000, "positions_value_cents": 3_000}
        assert world.trims(monkeypatch, _trim_settings(cap=0.05), **sized) == []
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert world.trims(monkeypatch, _trim_settings(cap=0.05, min_rise=0.10),
                               **sized) == []
        (warning,) = _warnings(caplog)
        assert "YES KXSNOW20-26 / NO KXSNOW25-26 alone would fetch" in warning
        assert "so that value cannot be right" in warning
        assert "Pairs to trim this run: 0 of 2" in caplog.text
        # Not vacuous: on a value that covers the snow pair, the minimum holds the
        # snow pair back and the rain pair alone is trimmed
        plans = world.trims(monkeypatch, _trim_settings(cap=0.05, min_rise=0.10),
                            cash_cents=5_000, positions_value_cents=4_000)
        assert [plan.legs[0].ticker for plan in plans] == [rain[0]]

    def test_a_crossed_book_on_the_outsized_pair_still_stops_every_trim(self, monkeypatch,
                                                                        caplog):
        world = _World()
        world.fills = _fills()
        _weather_pair(world, "rain", 30)
        snow = _weather_pair(world, "snow", 40, quotes=False)
        # A NO bid of 0.60 beside the YES bid of 0.42: the YES ask sits under its own
        # bid, so the snow pair's chance cannot be read
        world.books[snow[0]]["no"] = [["0.60", "100000"]]
        with caplog.at_level(logging.INFO):
            assert world.trims(monkeypatch, _trim_settings(cap=0.05), cash_cents=5_000,
                               positions_value_cents=3_000) == []
        (warning,) = _warnings(caplog)
        assert "YES KXSNOW20-26 / NO KXSNOW25-26 alone would fetch" in warning
        assert "so that value cannot be right" in warning

    # ── Each step of one pair ────────────────────────────────────────────────

    def test_a_count_that_is_not_whole_is_not_trimmed(self, monkeypatch, caplog):
        case = _TrimCase()
        case.world.positions[_A] = HeldPosition(_A, 30.5, 9.15, 0.45)
        case.world.positions[_B] = HeldPosition(_B, -30.5, 12.2, 0.51)
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert (f"Not trimming {_PAIR_TEXT}: its count 30.5 is not a whole number of "
                "contracts") in caplog.text

    def test_a_market_missing_from_the_market_list_is_not_trimmed(self, monkeypatch, caplog):
        case = _TrimCase()
        del case.world.markets[_B]
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert f"Not trimming {_PAIR_TEXT}: {_B} is not in this run's market list" in caplog.text
        assert case.world.calls["book"] == []

    def test_a_pair_whose_kind_cannot_be_read_is_not_trimmed(self, monkeypatch, caplog):
        # NO on the earlier deadline and YES on the later: no pair the bot buys
        case = _TrimCase()
        case.world.positions[_A] = HeldPosition(_A, -30.0, 9.0, 0.45)
        case.world.positions[_B] = HeldPosition(_B, 30.0, 12.0, 0.51)
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert (f"Not trimming NO {_A} / YES {_B}: its two markets cannot be shown to be a "
                "same-title or a time-series pair held this way round") in caplog.text
        assert case.world.calls["book"] == []

    def test_a_book_that_cannot_be_read_is_not_trimmed(self, monkeypatch, caplog):
        case = _TrimCase()
        del case.world.books[_B]
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert f"Not trimming {_PAIR_TEXT}: the order book of {_B} could not be read" in (
            caplog.text)

    @pytest.mark.parametrize("ticker, side", [(_A, "yes"), (_A, "no"), (_B, "yes"), (_B, "no")])
    def test_a_missing_bid_on_any_of_the_four_sides_is_not_trimmed(self, monkeypatch, caplog,
                                                                    ticker, side):
        # The two sides held price the sale; the other two give each market's asks
        case = _TrimCase()
        case.world.books[ticker][side] = []
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10) == []
        assert f"Not trimming {_PAIR_TEXT}: {ticker} has no {side.upper()} bid" in caplog.text

    @pytest.mark.parametrize("ticker", [_A, _B])
    @pytest.mark.parametrize("bid, trims", [(0.01, False), (0.0099, False), (0.0101, True),
                                            (0.02, True)])
    def test_a_yes_bid_of_one_cent_or_less_is_not_trimmed(self, monkeypatch, caplog, ticker,
                                                          bid, trims):
        # The backtest's candles cannot show such a bid, so it never trims on one
        case = _TrimCase()
        case.world.books[ticker]["yes"] = [[bid, 100_000.0]]
        with caplog.at_level(logging.INFO):
            plans = case.live(monkeypatch, 100_000_000, cap=0.05)
        refused = (f"Not trimming {_PAIR_TEXT}: the best YES bid on {ticker} (${bid:.4f}) is "
                   "one the hourly price history cannot show")
        assert (refused in caplog.text) == (not trims)
        assert plans == []
        # A NO bid of 1c is one a candle shows (its YES ask is 0.99)
        assert "the best NO bid" not in caplog.text

    def test_a_no_bid_of_one_cent_is_read(self, monkeypatch, caplog):
        case = _TrimCase()
        case.world.books[_A]["no"] = [[0.01, 100_000.0]]
        with caplog.at_level(logging.INFO):
            case.live(monkeypatch, 10_000, cap=0.10)
        assert "hourly price history cannot show" not in caplog.text
        assert "Kelly trim check" in caplog.text or "chance of paying" in caplog.text

    def test_a_crossed_book_is_not_trimmed(self, monkeypatch, caplog):
        build, value_cents, caps = _TRIM_PARITY_CASES["a crossed book"]
        with caplog.at_level(logging.INFO):
            assert build().live(monkeypatch, value_cents, **caps) == []
        assert (f"Not trimming {_PAIR_TEXT}: its chance of paying cannot be read from its "
                "quotes") in caplog.text

    def test_pairs_are_judged_in_ticker_order_each_on_its_own(self, monkeypatch, caplog):
        # Two pairs past the cap and one that waits: the plans come in ticker order
        world = _World()
        hail, rain, snow = (_weather_pair(world, name, 60) for name in ("hail", "rain", "snow"))
        world.fills = _fills({rain[0]: _NOW - timedelta(days=2)})
        with caplog.at_level(logging.INFO):
            plans = world.trims(monkeypatch, _trim_settings(cap=0.05), cash_cents=5_000,
                                positions_value_cents=20_000)
        assert [plan.legs[0].ticker for plan in plans] == [hail[0], snow[0]]
        assert "Pairs to trim this run: 2 of 3" in caplog.text
        assert f"Not trimming YES {rain[0]} / NO {rain[1]}: it last traded 2026-01-10" in (
            caplog.text)
        # The waiting pair's books were never read
        assert world.calls["book"] == [*hail, *snow]

    # ── The minimum price rise ───────────────────────────────────────────────

    @staticmethod
    def _risen(cents_a: int, cents_b: int = 0) -> _TrimCase:
        """The standard pair, whose asks of the sides held (YES on A at 0.44, NO on
        B at 0.53, so $0.97 a pair now) stood that many cents lower when it
        last traded."""
        return _TrimCase(a_then=(round(0.44 - cents_a / 100, 2), 0.58),
                         b_then=(0.48, round(0.53 - cents_b / 100, 2)))

    @pytest.mark.parametrize("cents, trims", [(9, False), (10, True), (11, True), (-5, False),
                                              (0, False)])
    def test_a_rise_under_at_and_over_the_minimum(self, monkeypatch, caplog, cents, trims):
        case = self._risen(cents)
        with caplog.at_level(logging.INFO):
            plans = case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)
        assert bool(plans) == trims
        then = 0.97 - cents / 100
        prices = f"price ${then:.2f} then, $0.97 now ({cents:+d}c, minimum 10c)"
        if trims:
            assert (f"30 each, last traded 2026-01-05, 7 day(s) ago, {prices}, pays 96.4%"
                    in caplog.text)
            assert plans[0].count == 19
        else:
            assert (f"Not trimming {_PAIR_TEXT}: its total price has not risen by the minimum "
                    f"since it last traded: {prices}") in caplog.text

    @pytest.mark.parametrize("yes_ask_then, prices", [
        # Prices on a grid finer than a cent: the rise is short of the 10c
        # minimum by 0.4c, by a tenth of a cent and by a hundredth of a cent
        (0.344, "price $0.874 then, $0.97 now (+9.6c, minimum 10c)"),
        (0.341, "price $0.871 then, $0.97 now (+9.9c, minimum 10c)"),
        (0.3401, "price $0.8701 then, $0.97 now (+9.99c, minimum 10c)"),
    ])
    def test_a_rise_just_short_of_the_minimum_is_not_written_as_the_minimum(
            self, monkeypatch, caplog, yes_ask_then, prices):
        case = _TrimCase(a_then=(yes_ask_then, 0.58), b_then=(0.48, 0.53))
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        (line,) = [r.getMessage() for r in caplog.records
                   if "has not risen by the minimum" in r.getMessage()]
        assert "(+10c, minimum 10c)" not in line
        assert line.endswith(f"since it last traded: {prices}")

    def test_a_change_in_price_is_written_exactly(self):
        # Whole cents as before; anything finer to the hundredth of a cent
        assert [seller._cents_text(d) for d in (0.13, -0.05, 0.0, -0.0, 0.10, 1.0)] == [
            "+13c", "-5c", "+0c", "+0c", "+10c", "+100c"]
        assert [seller._cents_text(d) for d in (0.096, 0.099, 0.0999, -0.0001, 0.2001)] == [
            "+9.6c", "+9.9c", "+9.99c", "-0.01c", "+20.01c"]
        # Float noise around a whole cent is a whole cent, and never "-0c"
        assert seller._cents_text(0.97 - 0.87) == "+10c"
        assert seller._cents_text(0.87 - 0.97) == "-10c"
        assert seller._cents_text(-1e-9) == "+0c"
        assert [seller._price_text(d) for d in (0.97, 0.874, 0.8701, 1.0, 0.5, 0.34 + 0.53)] == [
            "$0.97", "$0.874", "$0.8701", "$1.00", "$0.50", "$0.87"]

    def test_the_rise_is_the_two_markets_together(self, monkeypatch):
        # 6c on one market and 4c on the other is a 10c rise of the pair
        assert self._risen(6, 4).live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)
        assert self._risen(6, 3).live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        # A rise on one market is offset by a fall on the other
        assert self._risen(15, -6).live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []

    def test_the_rise_is_measured_from_the_later_of_the_two_markets_fills(self, monkeypatch):
        # Three candles a market: the pair stood at $0.77 two weeks ago and at
        # $0.92 a week ago; market B's fill a week ago is the pair's last trade,
        # so the rise is 5c, not 20c
        case = _TrimCase()
        earlier = _FIRST - 7 * _DAY
        case.world.candles = {
            _A: [_candle(earlier, 0.34, 0.68), _candle(_FIRST, 0.42, 0.60), _candle(_CK, *_A_NOW)],
            _B: [_candle(earlier, 0.58, 0.43), _candle(_FIRST, 0.51, 0.50), _candle(_CK, *_B_NOW)]}
        case.world.fills = _fills({_A: datetime.fromtimestamp(earlier, UTC), _B: _BOUGHT})
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.05)
        # Both markets' prices were read at that one moment
        assert {(t, end) for t, _e, _s, end in case.world.calls["candles"]} == {
            (_A, _FIRST), (_B, _FIRST)}

    def test_after_a_trim_the_rise_is_measured_from_the_trim(self, monkeypatch):
        # Bought at $0.77; a trim on January 5 sold part of it at $0.92. A run
        # two weeks after the purchase measures from the trim: 5c, not 20c
        case = _TrimCase()
        bought = _FIRST - 7 * _DAY
        case.world.candles = {
            _A: [_candle(bought, 0.34, 0.68), _candle(_FIRST, 0.42, 0.60), _candle(_CK, *_A_NOW)],
            _B: [_candle(bought, 0.58, 0.43), _candle(_FIRST, 0.51, 0.50), _candle(_CK, *_B_NOW)]}
        # Kalshi lists each market's latest fill: the trim's
        case.world.fills = _fills({_A: _BOUGHT, _B: _BOUGHT})
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        # Had it not been trimmed, the same run would measure from the purchase
        case.world.fills = _fills({_A: datetime.fromtimestamp(bought, UTC),
                                   _B: datetime.fromtimestamp(bought, UTC)})
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)

    def test_no_fill_listed_is_not_trimmed_and_reads_no_candle(self, monkeypatch, caplog):
        # The pair's last fill is older than the fills list reaches: it has
        # rested, but its price then cannot be read
        case = self._risen(20)
        case.world.fills = _fills()
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        assert (f"Not trimming {_PAIR_TEXT}: no fill is listed on either market, so its price "
                "when it last traded cannot be read, and a minimum price rise is set"
                in caplog.text)
        assert case.world.calls["candles"] == []
        # With no minimum the same pair trims
        assert case.live(monkeypatch, 10_000, cap=0.10)

    @pytest.mark.parametrize("ticker, side", [(_A, "YES"), (_B, "NO")])
    def test_candles_that_cannot_be_read_are_not_a_rise(self, monkeypatch, caplog, ticker, side):
        case = self._risen(20)
        del case.world.candles[ticker]
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        assert (f"Not trimming {_PAIR_TEXT}: the {side} ask on {ticker} when the pair last "
                "traded (2026-01-05) cannot be read (no usable quote in the 7 days before "
                "it, or its price history could not be read), and a minimum price rise is "
                "set") in caplog.text

    def test_no_usable_quote_in_the_week_before_is_not_a_rise(self, monkeypatch, caplog):
        # Market A's only candle before the pair's last fill ended eight days
        # before it, and the one at the fill itself has an empty YES ask book
        case = self._risen(20)
        case.world.candles[_A] = [_candle(_FIRST - 8 * _DAY, 0.24, 0.78),
                                  _candle(_FIRST, 1.00, 0.58), _candle(_CK, *_A_NOW)]
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        assert f"the YES ask on {_A} when the pair last traded (2026-01-05) cannot be read" in (
            caplog.text)
        # A usable quote six days before the fill is read
        case.world.candles[_A].insert(1, _candle(_FIRST - 6 * _DAY, 0.24, 0.78))
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)

    def test_a_candle_after_the_last_fill_is_never_the_price_then(self, monkeypatch):
        # An hour after the fill the ask had jumped: the price then is still the
        # one at the fill, so the rise since is 20c
        case = self._risen(20)
        case.world.candles[_A].insert(1, _candle(_FIRST + 3600, 0.44, 0.58))
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.20)

    def test_with_no_minimum_no_candle_is_read(self, monkeypatch):
        case = self._risen(-30)
        # Not even readable: nothing asks for them
        case.world.candles = {}
        assert case.live(monkeypatch, 10_000, cap=0.10)
        assert case.world.calls["candles"] == []

    def test_a_minimum_with_no_wait_still_reads_the_fills(self, monkeypatch, caplog):
        monkeypatch.setattr(seller, "KELLY_TRIM_MIN_DAYS_SINCE_TRADE", 0)
        case = self._risen(20)
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)
        assert case.world.fill_reads == 1
        assert "30 each, no wait, price $0.77 then, $0.97 now (+20c, minimum 10c), pays" in (
            caplog.text)
        # Unreadable fills: nothing is trimmed, wait or no wait
        case.world.fills = None
        assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        # An empty list is a valid read with no wait; the pair has no fill listed
        case.world.fills = _fills(oldest=None)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10) == []
        assert "no fill is listed on either market" in caplog.text

    def test_each_markets_price_then_is_read_once(self, monkeypatch):
        case = self._risen(20)
        case.live(monkeypatch, 10_000, cap=0.10, min_rise=0.10)
        week = config.TRIM_PRICE_THEN_MAX_AGE_DAYS * _DAY
        assert case.world.calls["candles"] == [
            (_A, _EV_A, _FIRST - week - 3600, _FIRST), (_B, _EV_B, _FIRST - week - 3600, _FIRST)]
        assert all(type(value) is int for call in case.world.calls["candles"]
                   for value in call[2:])


# ─── The module's shape ───────────────────────────────────────────────────────

def _tree():
    return ast.parse(inspect.getsource(seller))


def _def(func_name: str) -> ast.FunctionDef:
    """One function (or method) of seller, as its syntax tree."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"seller.{func_name} not found")


def _calls(func_name: str) -> set[str]:
    """Every name a function of seller calls, bare or as an attribute."""
    return {(sub.func.id if isinstance(sub.func, ast.Name)
             else getattr(sub.func, "attr", None))
            for sub in ast.walk(_def(func_name)) if isinstance(sub, ast.Call)}


def _calls_of(func_name: str, callee: str) -> list[ast.Call]:
    """Every call a function of seller makes to `callee`, bare or as an attribute."""
    return [sub for sub in ast.walk(_def(func_name)) if isinstance(sub, ast.Call)
            and (sub.func.id if isinstance(sub.func, ast.Name)
                 else getattr(sub.func, "attr", None)) == callee]


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

    def test_plan_trims_requires_the_runs_settings_and_both_amounts(self):
        # Nothing a trim is sized on has a default: the run's settings, its
        # moment, its cash and Kalshi's value of the open positions
        signature = inspect.signature(seller.plan_trims)
        for name in ("settings", "now", "cash_cents", "positions_value_cents"):
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert parameter.default is inspect.Parameter.empty, name
        # Only the positions a take-profit sale takes may be left out
        assert signature.parameters["skip"].default == frozenset()

    def test_a_trim_is_decided_through_the_shared_rule(self):
        # The rule, the chance of paying, both waits and the cap are config's
        # own functions, the very ones the backtest trims with
        for name in ("kelly_trim_count", "kelly_trim_bids", "held_pair_win_prob",
                     "kelly_trim_rested", "kelly_trim_min_days", "trade_day", "pair_price_rise",
                     "trim_rise_reached", "pair_size_cap", "fee_per_pair_approx"):
            assert getattr(seller, name) is getattr(config, name), name
            assert getattr(backtester, name, getattr(config, name)) is getattr(config, name)
        trim_one = _calls("_trim_one")
        assert {"kelly_trim_count", "kelly_trim_bids", "held_pair_win_prob",
                "kelly_trim_rested", "pair_price_rise", "trim_rise_reached", "pair_size_cap",
                "fee_per_pair_approx", "fee_leg_exact", "walk_bids", "bid_ladder",
                "candle_reads_bid", "held_pair_kind", "leg_sides", "_fetch_orderbook",
                "ask_then", "_day", "_other"} <= trim_one
        # It reads no candle and no fills list itself, and turns no time into a
        # day but through _day
        assert not {"recent_candles", "ask_before", "get_fill_times", "trade_day"} & trim_one
        assert {"recent_candles", "ask_before"} <= _calls("ask_then")
        assert {"get_fill_times", "held_positions_bid_value", "held_pairs", "_trim_min_days",
                "kelly_trim_rested", "_day", "_trim_one"} <= _calls("plan_trims")
        # One fills read, one floor and one count of days a run
        for callee in ("get_fill_times", "held_positions_bid_value", "_trim_min_days"):
            assert len(_calls_of("plan_trims", callee)) == 1, callee
        # The wait's number of days is seller's own binding, checked by config
        assert ast.unparse(_def("_trim_min_days").body[-1]) == (
            "return kelly_trim_min_days(KELLY_TRIM_MIN_DAYS_SINCE_TRADE)")
        # ... and a day is read on seller's own binding of the schedule
        assert ast.unparse(_def("_day").body[-1]) == "return trade_day(moment, SCHEDULED_RUN)"

    def test_the_trims_steps_come_in_the_rules_order(self):
        # The wait before any order book is read; the price rise measured, then
        # tested, before the trim is sized
        (rested,) = _calls_of("_trim_one", "kelly_trim_rested")
        (kind,) = _calls_of("_trim_one", "held_pair_kind")
        (book,) = _calls_of("_trim_one", "_fetch_orderbook")
        (measured,) = _calls_of("_trim_one", "pair_price_rise")
        (tested,) = _calls_of("_trim_one", "trim_rise_reached")
        (forecast,) = _calls_of("_trim_one", "held_pair_win_prob")
        (decided,) = _calls_of("_trim_one", "kelly_trim_count")
        assert (rested.lineno < kind.lineno < book.lineno < measured.lineno < tested.lineno
                < forecast.lineno < decided.lineno)
        # What the pair alone fetches is worked out (and the run-wide value check
        # made) once the books are read, before the price rise and the chance
        (valued,) = _calls_of("_trim_one", "fee_per_pair_approx")
        (guard,) = [node for node in ast.walk(_def("_trim_one")) if isinstance(node, ast.Compare)
                    and ast.unparse(node) == "count * top_value > positions_value"]
        assert book.lineno < valued.lineno < guard.lineno < measured.lineno
        # The bids too far under the best bid are left out before anything is
        # valued or sized on them, at seller's own binding of the reach
        (reach,) = _calls_of("_trim_one", "kelly_trim_bids")
        assert book.lineno < reach.lineno < valued.lineno < decided.lineno
        assert ast.unparse(reach.args[1]) == "KELLY_TRIM_MAX_BID_DROP"
        # The rise goes from the price then to the price now, and is tested
        # against the run's own minimum
        assert [ast.unparse(arg) for arg in measured.args] == [
            "then_a + then_b", "now_a + now_b"]
        assert [ast.unparse(arg) for arg in tested.args] == ["rise", "settings.trim_min_rise"]
        assert ast.unparse(rested) == "kelly_trim_rested(last, now_date, min_days)"

    def test_the_four_quotes_go_in_as_the_backtest_hands_them(self):
        # Each market's YES ask, then its NO ask, an ask being 1 minus the other
        # side's bid: the order backtester._kelly_trim is pinned to. Swapping
        # two can keep the mid spread and still misread a crossed book
        (forecast,) = _calls_of("_trim_one", "held_pair_win_prob")
        assert [ast.unparse(arg) for arg in forecast.args] == [
            "pair_type", "1.0 - no_bid_a", "1.0 - yes_bid_a", "1.0 - no_bid_b",
            "1.0 - yes_bid_b", "settings.interval_discount"] and not forecast.keywords
        source = ast.parse(inspect.getsource(backtester))
        (theirs,) = [sub for node in ast.walk(source)
                     if isinstance(node, ast.FunctionDef) and node.name == "_kelly_trim"
                     for sub in ast.walk(node) if isinstance(sub, ast.Call)
                     and isinstance(sub.func, ast.Name) and sub.func.id == "held_pair_win_prob"]
        assert [ast.unparse(arg) for arg in theirs.args][1:5] == [
            ast.unparse(arg) for arg in forecast.args][1:5]
        # The rule's own arguments, in its order
        (decided,) = _calls_of("_trim_one", "kelly_trim_count")
        assert [ast.unparse(arg) for arg in decided.args] == [
            "count", "value", "win_prob", "cap", "top_value", "net_sale"]
        assert not decided.keywords
        (cap,) = _calls_of("_trim_one", "pair_size_cap")
        assert [ast.unparse(arg) for arg in cap.args] == [
            "pair_type", "settings.size_cap", "settings.same_title_size_cap"]


def test_plans_are_frozen():
    leg = seller.SaleLeg("T", "E", "yes", 1, 0.5)
    with pytest.raises(AttributeError):
        leg.count = 2
    assert math.isclose(leg.cost_dollars, 0.5) and leg.market is None
    plan = seller.SalePlan("T", (leg,), 1, 0.5, {}, {}, 0.6, ((0.1, 0.5),), None, 0.2)
    with pytest.raises(AttributeError):
        plan.count = 2
    assert timedelta(seconds=seller._DAY_SECONDS) == timedelta(days=1)
    # Built as before, a leg's count is everything held and the plan is a
    # take-profit sale: the fields a Kelly trim sets come last, with defaults
    assert leg.held is None and plan.kind == config.SALE_KIND_TAKE_PROFIT and plan.level == 0.2
    assert [f.name for f in dataclasses.fields(seller.SaleLeg)][-1] == "held"
    assert [f.name for f in dataclasses.fields(seller.SalePlan)][-2:] == ["level", "kind"]
    assert (config.SALE_KIND_TAKE_PROFIT, config.SALE_KIND_KELLY_TRIM) == (
        "take_profit", "kelly_trim")
