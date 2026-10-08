"""Tests for live_portfolio.py: reading the account from Kalshi, reading the
bot's trade log, matching the bot's purchases to Kalshi orders, and the ledger
of every contract bought, sold and paid out.

All offline. historical._historical_get is replaced by FakeKalshi, which
serves each listing page by page by cursor, so no request ever leaves the
machine. The trade logs are written by reporter.append_to_prod_log itself,
with its paths pointed at tmp_path (as tests/test_reporter.py does) and its
clock and the host's time zone pinned. The ledger's cash is checked against
a separate running-position formula over 300 random fill sequences.
"""
import copy
import random
import shutil
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace

import openpyxl
import pytest

from kalshi_betting import auth, config, historical, live_portfolio, reporter
from kalshi_betting.live_portfolio import (
    OTHER_BETS,
    Account,
    BotLeg,
    BotTrade,
    Fill,
    Market,
    Payout,
)
from kalshi_betting.reporter import TradeResult
from kalshi_betting.scanner import ApiMarket, CandidatePair
from kalshi_betting.strategy import TradeSpec

from .test_scheduler import _host_zone

D = Decimal
_API = "/trade-api/v2"
_KEYS = {
    "/historical/fills": "fills",
    "/portfolio/fills": "fills",
    "/portfolio/settlements": "settlements",
    "/portfolio/deposits": "deposits",
    "/portfolio/withdrawals": "withdrawals",
    "/portfolio/positions": "market_positions",
    "/markets": "markets",
    "/historical/markets": "markets",
}
# A fixed moment for tests that never compare with the clock
T0 = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)


def _iso(moment: datetime) -> str:
    """A moment as Kalshi writes it: ISO text in UTC ending in Z."""
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _ago(seconds: float) -> datetime:
    """The moment `seconds` before the real clock now (for reads that compare with it)."""
    return datetime.now(UTC) - timedelta(seconds=seconds)


def fill_row(fill_id, ticker, when, book_side, count, yes_price, *, fee="0.0000",
             order_id=None, **extra) -> dict:
    """A fill record as Kalshi's fills listings send it."""
    yes = D(str(yes_price))
    row = {"fill_id": fill_id, "order_id": order_id or f"order-{fill_id}", "ticker": ticker,
           "created_time": _iso(when), "book_side": book_side,
           "count_fp": f"{D(str(count)):.2f}", "yes_price_dollars": f"{yes:.4f}",
           "no_price_dollars": f"{1 - yes:.4f}", "fee_cost": fee}
    row.update(extra)
    return row


class FakeKalshi:
    """
    Stands in for historical._historical_get: serves each listing page by page.

    rows maps a path (after /trade-api/v2) to its records, newest first as
    Kalshi lists them; balance is the /portfolio/balance reply. Every call is
    recorded, and on_balance (if set) runs before each balance read, so a test
    can change the account while it is being read.
    """

    def __init__(self):
        self.rows = {path: [] for path in _KEYS}
        self.balance = {"balance_breakdown": [{"exchange_index": 0, "balance_dollars": "100.0000"}],
                        "portfolio_value": 0}
        self.calls = []
        self.on_balance = None

    def __call__(self, client, path, **params):
        assert path.startswith(_API), path
        short = path[len(_API):]
        self.calls.append((short, dict(params)))
        if short == "/portfolio/balance":
            if self.on_balance:
                self.on_balance()
            return copy.deepcopy(self.balance)
        rows = self.rows[short]
        if params.get("tickers"):
            wanted = set(params["tickers"].split(","))
            rows = [r for r in rows if r["ticker"] in wanted]
        start = int(params.get("cursor") or 0)
        limit = params["limit"]
        nxt = start + limit
        return {_KEYS[short]: copy.deepcopy(rows[start:nxt]),
                "cursor": str(nxt) if nxt < len(rows) else ""}

    def count(self, path: str) -> int:
        """How many requests went to `path`."""
        return sum(1 for short, _ in self.calls if short == path)


@pytest.fixture
def kalshi(monkeypatch):
    """A FakeKalshi in place of historical._historical_get, and sleeps that only record."""
    fake = FakeKalshi()
    fake.sleeps = []
    fake.on_sleep = None

    def sleep(seconds):
        fake.sleeps.append(seconds)
        if fake.on_sleep:
            fake.on_sleep()

    monkeypatch.setattr(historical, "_historical_get", fake)
    monkeypatch.setattr(live_portfolio, "time", SimpleNamespace(sleep=sleep))
    return fake


# ---- reading the account ---------------------------------------------------

class TestReadAccount:
    def test_live_and_archived_fills_are_deduped_and_sorted(self, kalshi):
        kalshi.rows["/historical/fills"] = [
            fill_row("f1", "M1", _ago(3000), "bid", 2, "0.30"),
            fill_row("f2", "M1", _ago(4000), "bid", 1, "0.30", fee="0.0100"),
        ]
        kalshi.rows["/portfolio/fills"] = [
            fill_row("f3", "M2", _ago(1000), "ask", 3, "0.60"),
            fill_row("f2", "M1", _ago(4000), "bid", 1, "0.30", fee="0.0200"),
        ]
        account = live_portfolio.read_account(object())
        assert [f.fill_id for f in account.fills] == ["f2", "f1", "f3"]
        # The live copy of a fill is kept over its archived copy
        assert account.fills[0].fee == D("0.0200")
        assert account.changing is False
        assert account.warnings == ()
        assert kalshi.sleeps == []

    def test_a_fill_moving_to_the_archive_mid_read_is_kept(self, kalshi, monkeypatch):
        # Kalshi's archive cutoff moves while the fills are read: the old fill
        # leaves the live listing for the archive between the two reads
        kalshi.rows["/portfolio/fills"] = [fill_row("new", "M1", _ago(3600), "bid", 1, "0.30"),
                                           fill_row("old", "M1", _ago(3 * 86400), "bid", 2, "0.30")]
        moved = []

        def get(client, path, **params):
            reply = kalshi(client, path, **params)
            if path.endswith("fills") and params.get("limit") != 1 and not moved:
                moved.append(path)
                kalshi.rows["/historical/fills"].append(kalshi.rows["/portfolio/fills"].pop())
            return reply

        monkeypatch.setattr(historical, "_historical_get", get)
        account = live_portfolio.read_account(object())
        assert [f.fill_id for f in account.fills] == ["old", "new"]
        assert account.changing is False

    def test_a_no_purchase_is_read_from_book_side_alone(self, kalshi):
        # Kalshi sends a NO purchase with action "sell": only book_side is reliable
        kalshi.rows["/portfolio/fills"] = [
            fill_row("f2", "M1", _ago(500), "bid", 1, "0.30", action="sell", side="no"),
            fill_row("f1", "M1", _ago(600), "ask", 4, "0.25", action="sell", side="no"),
        ]
        account = live_portfolio.read_account(object())
        no_buy, yes_buy = account.fills
        assert no_buy.buys_yes is False
        assert (no_buy.count, no_buy.yes_price, no_buy.no_price) == (D("4.00"), D("0.25"), D("0.75"))
        assert yes_buy.buys_yes is True

    @pytest.mark.parametrize("book_side", [None, "", "yes", "buy"])
    def test_a_fill_without_a_book_side_is_refused(self, book_side):
        row = fill_row("f1", "M1", T0, "bid", 1, "0.30")
        row["book_side"] = book_side
        with pytest.raises(ValueError, match="book_side"):
            live_portfolio._fill(row)

    def test_cash_comes_from_every_shard(self, kalshi):
        kalshi.balance = {"balance": 9999, "balance_dollars": "99.99", "balance_breakdown": [
            {"exchange_index": 0, "balance_dollars": "10.1234", "balance": "77.0000"},
            {"exchange_index": 1, "balance": "5.0001"},
        ]}
        account = live_portfolio.read_account(object())
        assert account.cash == D("15.1235")
        assert account.shards == 2
        assert account.warnings == ()

    def test_a_shard_listed_twice_counts_once(self, kalshi):
        # auth._balance_cents_by_shard keeps a repeated shard's last entry; so does this
        kalshi.balance = {"balance_breakdown": [
            {"exchange_index": 0, "balance_dollars": "5.0000"},
            {"exchange_index": 1, "balance_dollars": "2.0000"},
            {"exchange_index": 0, "balance_dollars": "6.0000"},
        ]}
        account = live_portfolio.read_account(object())
        assert (account.cash, account.shards) == (D("8.0000"), 2)
        assert account.warnings == (
            "Kalshi's balance reply lists shard 0 twice: only its last entry is counted",)
        assert auth._balance_cents_by_shard(kalshi.balance) == {0: 600, 1: 200}

    @pytest.mark.parametrize("balance, cash", [
        ({"balance_dollars": "7.5000", "balance": 1}, D("7.5000")),
        ({"balance": 1234}, D("12.34")),
        ({"balance": 1234, "balance_breakdown": []}, D("12.34")),
    ])
    def test_cash_without_shards_is_one_shard(self, balance, cash):
        assert live_portfolio._cash(balance) == (cash, 1, [])

    def test_unreadable_cash_is_refused(self):
        with pytest.raises(ValueError):
            live_portfolio._cash({"portfolio_value": 10})
        with pytest.raises(ValueError):
            live_portfolio._cash({"balance_breakdown": [{"exchange_index": 0}]})

    @pytest.mark.parametrize("reply, value", [
        ({"portfolio_value": 7850}, D("78.50")),
        ({"portfolio_value_dollars": "78.5099", "portfolio_value": 1}, D("78.50")),
        ({}, None),
        ({"portfolio_value": -5}, None),
    ])
    def test_the_positions_value_is_auths_reading(self, kalshi, reply, value):
        kalshi.balance = {"balance": 100, **reply}
        account = live_portfolio.read_account(object())
        assert account.kalshi_positions_value == value

    def test_the_positions_value_goes_through_auth_at_call_time(self, kalshi, monkeypatch):
        seen = []

        def reading(data):
            seen.append(data)
            return 4321

        monkeypatch.setattr(auth, "_positions_value_cents", reading)
        account = live_portfolio.read_account(object())
        assert account.kalshi_positions_value == D("43.21")
        assert seen == [kalshi.balance]

    def test_deposits_and_withdrawals(self, kalshi):
        kalshi.rows["/portfolio/deposits"] = [
            {"status": "applied", "amount_cents": 10000, "fee_cents": 25,
             "created_ts": int(T0.timestamp()), "finalized_ts": int(T0.timestamp()) + 60},
            {"status": "pending", "amount_cents": 5000, "fee_cents": 0,
             "created_ts": int(T0.timestamp())},
        ]
        kalshi.rows["/portfolio/withdrawals"] = [
            {"status": "applied", "amount_cents": 5000, "fee_cents": 100,
             "created_ts": int(T0.timestamp()) - 3600},
        ]
        account = live_portfolio.read_account(object())
        assert [(f.time, f.amount) for f in account.flows] == [
            (T0 - timedelta(hours=1), D("-51")),
            (T0 + timedelta(seconds=60), D("99.75")),
        ]

    def test_settlement_revenue_and_value_are_read_in_cents(self, kalshi):
        kalshi.rows["/portfolio/settlements"] = [
            {"ticker": "M1", "settled_time": _iso(T0), "market_result": "yes",
             "revenue": 1300, "value": 100, "fee_cost": "0.3100"},
            {"ticker": "M2", "settled_time": _iso(T0), "market_result": "scalar",
             "revenue": 0, "value": 35},
        ]
        account = live_portfolio.read_account(object())
        first, second = account.payouts
        assert (first.ticker, first.result, first.revenue, first.yes_value) == (
            "M1", "yes", D("13.00"), D("1.00"))
        assert (second.result, second.revenue, second.yes_value) == ("scalar", D("0"), D("0.35"))

    def test_positions_are_signed_and_zero_is_left_out(self, kalshi):
        kalshi.rows["/portfolio/positions"] = [
            {"ticker": "M1", "position_fp": "-13.00"},
            {"ticker": "M2", "position_fp": "4.00"},
            {"ticker": "M3", "position_fp": "0.00"},
        ]
        account = live_portfolio.read_account(object())
        assert account.positions == {"M1": D("-13.00"), "M2": D("4.00")}
        assert ("/portfolio/positions", {"cursor": None, "limit": config.LIVE_PAGE_SIZE,
                                         "count_filter": "position"}) in kalshi.calls

    def test_every_listing_is_read_page_by_page(self, kalshi, monkeypatch):
        monkeypatch.setattr(config, "LIVE_PAGE_SIZE", 2)
        kalshi.rows["/portfolio/fills"] = [
            fill_row(f"f{i}", "M1", _ago(1000 + i), "bid", 1, "0.30") for i in range(5)]
        account = live_portfolio.read_account(object())
        assert len(account.fills) == 5
        cursors = [p["cursor"] for path, p in kalshi.calls
                   if path == "/portfolio/fills" and p["limit"] == 2]
        assert cursors == [None, "2", "4"]

    def test_the_records_are_read_before_the_balance_and_positions(self, kalshi):
        live_portfolio.read_account(object())
        order = [path for path, params in kalshi.calls if params.get("limit") != 1]
        # Live fills before archived ones: a fill moving to the archive while
        # they are read is then in one of the two reads
        assert order == ["/portfolio/fills", "/historical/fills", "/portfolio/settlements",
                         "/portfolio/deposits", "/portfolio/withdrawals",
                         "/portfolio/balance", "/portfolio/positions"]


class TestPages:
    def test_a_repeated_cursor_raises(self, monkeypatch):
        cursors = iter(["A", "B", "A"])
        monkeypatch.setattr(historical, "_historical_get",
                            lambda client, path, **params: {"fills": [], "cursor": next(cursors)})
        with pytest.raises(ValueError, match="repeated a page cursor"):
            live_portfolio._pages(object(), "/portfolio/fills", "fills", limit=5)

    def test_a_listing_that_never_ends_raises(self, monkeypatch):
        numbers = iter(range(10_000))
        monkeypatch.setattr(config, "SCANNER_MAX_PAGES", 3)
        monkeypatch.setattr(historical, "_historical_get",
                            lambda client, path, **params: {"fills": [],
                                                            "cursor": str(next(numbers))})
        with pytest.raises(ValueError, match="did not finish within 3 pages"):
            live_portfolio._pages(object(), "/portfolio/fills", "fills", limit=5)

    @pytest.mark.parametrize("reply", [{"cursor": ""}, {"fills": None}, {"fills": "x"}])
    def test_a_reply_without_its_list_raises(self, monkeypatch, reply):
        monkeypatch.setattr(historical, "_historical_get", lambda client, path, **params: reply)
        with pytest.raises(ValueError, match="without a 'fills' list"):
            live_portfolio._pages(object(), "/portfolio/fills", "fills", limit=5)

    @pytest.mark.parametrize("reply", [[], "text", None])
    def test_a_reply_that_is_not_an_object_raises(self, monkeypatch, reply):
        monkeypatch.setattr(historical, "_historical_get", lambda client, path, **params: reply)
        with pytest.raises(ValueError, match="not an object"):
            live_portfolio._get(object(), "/portfolio/balance")

    def test_every_request_goes_to_the_signed_get_with_the_api_prefix(self, monkeypatch):
        calls = []

        def fake(client, path, **params):
            calls.append((client, path, params))
            return {"fills": [], "cursor": ""}

        monkeypatch.setattr(historical, "_historical_get", fake)
        client = object()
        live_portfolio._pages(client, "/portfolio/fills", "fills", limit=7)
        assert calls == [(client, "/trade-api/v2/portfolio/fills", {"cursor": None, "limit": 7})]


class TestReadsAgainWhenTheAccountChanges:
    def test_a_fill_arriving_mid_read_reads_again(self, kalshi):
        kalshi.rows["/portfolio/fills"] = [fill_row("f1", "M1", _ago(600), "bid", 1, "0.30")]

        def arrive():
            if kalshi.count("/portfolio/balance") == 1:
                kalshi.rows["/portfolio/fills"].insert(
                    0, fill_row("f2", "M1", _ago(300), "bid", 2, "0.31"))

        kalshi.on_balance = arrive
        account = live_portfolio.read_account(object())
        assert [f.fill_id for f in account.fills] == ["f1", "f2"]
        assert account.changing is False
        assert kalshi.count("/portfolio/balance") == 2
        assert kalshi.sleeps == [config.LIVE_READ_RETRY_PAUSE_SECONDS]

    def test_a_settlement_arriving_mid_read_reads_again(self, kalshi):
        def arrive():
            if kalshi.count("/portfolio/balance") == 1:
                kalshi.rows["/portfolio/settlements"].insert(0, {
                    "ticker": "M9", "settled_time": _iso(_ago(60)), "market_result": "no",
                    "revenue": 500})

        kalshi.on_balance = arrive
        account = live_portfolio.read_account(object())
        assert [p.ticker for p in account.payouts] == ["M9"]
        assert kalshi.count("/portfolio/balance") == 2
        assert account.changing is False

    def test_a_fill_under_the_settle_age_waits_and_reads_again(self, kalshi):
        kalshi.rows["/portfolio/fills"] = [fill_row("f1", "M1", _ago(1), "bid", 1, "0.30")]

        def age():                    # the pause lets the fill grow old enough
            kalshi.rows["/portfolio/fills"][0]["created_time"] = _iso(
                _ago(config.LIVE_READ_SETTLE_SECONDS + 5))

        kalshi.on_sleep = age
        account = live_portfolio.read_account(object())
        assert account.changing is False
        assert kalshi.count("/portfolio/balance") == 2
        assert kalshi.sleeps == [config.LIVE_READ_RETRY_PAUSE_SECONDS]

    def test_an_account_that_keeps_changing_is_marked(self, kalshi):
        def arrive():
            n = kalshi.count("/portfolio/balance")
            kalshi.rows["/portfolio/fills"].insert(
                0, fill_row(f"new{n}", "M1", _ago(600 - n), "bid", 1, "0.30"))

        kalshi.on_balance = arrive
        account = live_portfolio.read_account(object())
        assert account.changing is True
        assert kalshi.count("/portfolio/balance") == config.LIVE_READ_ATTEMPTS
        # A pause between reads, none after the last
        assert kalshi.sleeps == [config.LIVE_READ_RETRY_PAUSE_SECONDS] * (
            config.LIVE_READ_ATTEMPTS - 1)

    def test_fills_sharing_the_newest_moment_are_not_a_change(self, kalshi):
        # One order that walked three levels: three fills at one moment, listed
        # in an order that is not their ids' order
        when = _ago(900)
        kalshi.rows["/portfolio/fills"] = [
            fill_row("fill-a", "M1", when, "bid", 1, "0.30", order_id="o1"),
            fill_row("fill-c", "M1", when, "bid", 1, "0.31", order_id="o1"),
            fill_row("fill-b", "M1", when, "bid", 1, "0.32", order_id="o1"),
        ]
        account = live_portfolio.read_account(object())
        assert account.changing is False
        assert kalshi.count("/portfolio/balance") == 1
        assert kalshi.sleeps == []


class TestReadMarkets:
    def test_live_first_then_the_archive(self, kalshi, monkeypatch):
        monkeypatch.setattr(config, "LIVE_MARKET_TICKERS_PER_REQUEST", 2)
        kalshi.rows["/markets"] = [
            {"ticker": "M1", "event_ticker": "E1", "title": "One?", "status": "active",
             "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.45", "last_price_dollars": "0.42",
             "result": ""},
            {"ticker": "M3", "event_ticker": "E3", "status": "active", "yes_bid_dollars": "x"},
        ]
        kalshi.rows["/historical/markets"] = [
            {"ticker": "M2", "event_ticker": "E2", "title": "Two?", "status": "finalized",
             "result": "scalar", "settlement_ts": _iso(T0), "settlement_value_dollars": "0.3500"},
            {"ticker": "M1", "event_ticker": "E1", "title": "Old copy", "status": "finalized"},
        ]
        found = live_portfolio.read_markets(object(), ["M2", "M1", "M4", "M3", "M1"])
        assert set(found) == {"M1", "M2", "M3"}
        assert found["M1"] == Market("M1", "E1", "One?", "active", "", None, None,
                                     D("0.40"), D("0.45"), D("0.42"))
        assert found["M2"].settled_at == T0
        assert (found["M2"].result, found["M2"].yes_value) == ("scalar", D("0.3500"))
        assert found["M3"].title == "M3" and found["M3"].yes_bid is None
        asked = [(path, params["tickers"]) for path, params in kalshi.calls]
        # Two per request; the archive is asked only for what the live listing lacked
        assert asked == [("/markets", "M1,M2"), ("/markets", "M3,M4"),
                         ("/historical/markets", "M2,M4")]

    def test_nothing_wanted_sends_nothing(self, kalshi):
        assert live_portfolio.read_markets(object(), []) == {}
        assert kalshi.calls == []


# ---- the bot's trade log ---------------------------------------------------

def _trade_result(ticker_a: str, ticker_b: str, status: str, *, pair_type: str = "time_series",
                  count: int = 13) -> TradeResult:
    """A TradeResult as trader.execute_trades hands reporter one."""
    def market(ticker):
        return ApiMarket(ticker=ticker, event_ticker=f"EV-{ticker}", title=f"Will {ticker}?",
                         subtitle="", status="active", close_time=None)
    pair = CandidatePair(market_a=market(ticker_a), market_b=market(ticker_b), pA=0.30, pB=0.60,
                         nA=0.70, tradeable=True, canonical_title="pair", pair_type=pair_type,
                         nB=0.40)
    spec = TradeSpec(pair=pair, x=count, y=count, total_cost=9.10, total_cost_with_fees=9.41,
                     min_payoff=3.0, profit_ratio=0.3, days_to_close=20,
                     monthly_profit_ratio=0.4, kelly_p=0.7, kelly_fraction=0.1)
    return TradeResult(spec=spec, status=status,
                       error=None if status == "executed" else "YES leg error: killed")


@pytest.fixture
def log_paths(tmp_path, monkeypatch):
    """Point reporter's trade log, its lock and its fallback folder at tmp_path."""
    log_path = tmp_path / "trade_log.xlsx"
    monkeypatch.setattr(reporter, "PROD_LOG_PATH", log_path)
    monkeypatch.setattr(reporter, "_LOCK_PATH", tmp_path / "trade_log.xlsx.lock")
    monkeypatch.setattr(reporter, "PROJECT_ROOT", tmp_path)
    return log_path


@pytest.fixture
def pacific():
    """The host's clock on Los Angeles time (as the bot's Mac keeps it), restored afterwards."""
    with _host_zone("America/Los_Angeles"):
        yield


def _write_run(monkeypatch, at: datetime, results: list, cash_before: float) -> None:
    """Have reporter append one run to the trade log as if its clock read `at`."""
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at if tz is None else at.astimezone(tz)

    monkeypatch.setattr(reporter, "datetime", _Clock)
    reporter.append_to_prod_log(results, cash_before, cash_before)


class TestReadTradeLogs:
    def test_runs_trades_and_windows(self, log_paths, pacific, monkeypatch):
        run1, dry, run2 = T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)
        _write_run(monkeypatch, run1, [
            _trade_result("A1", "B1", "executed"),
            _trade_result("A2", "B2", "rolled_back"),
            _trade_result("A3", "B3", "executed", pair_type="same_title", count=4),
        ], 132.45)
        _write_run(monkeypatch, dry, [_trade_result("A1", "B1", "simulated")], 120.00)
        _write_run(monkeypatch, run2, [
            _trade_result("A4", "B4", "failed"),
            _trade_result("A5", "B5", "manual_review"),
        ], 110.07)
        trades, starts, warnings = live_portfolio.read_trade_logs(
            live_portfolio.trade_log_paths())
        assert warnings == []
        window = timedelta(seconds=config.LIVE_BOT_RUN_WINDOW_SECONDS)
        assert [(t.trade_id, t.status) for t in trades] == [
            ("trade_log.xlsx#3", "executed"), ("trade_log.xlsx#4", "rolled_back"),
            ("trade_log.xlsx#5", "executed"), ("trade_log.xlsx#10", "manual_review")]
        first, _, same_title, review = trades
        assert first.legs == (BotLeg("A1", "yes", D(13)), BotLeg("B1", "no", D(13)))
        assert same_title.legs == (BotLeg("A3", "no", D(4)), BotLeg("B3", "yes", D(4)))
        assert first.title == "Will A1?"
        assert (first.logged_at, first.run_after) == (run1, run1 - window)
        # The dry run in between does not move the second run's window
        assert (review.logged_at, review.run_after) == (run2, run1)
        assert [(s.logged_at, s.run_after, s.cash_before) for s in starts] == [
            (run1, run1 - window, D("132.45")), (run2, run1, D("110.07"))]

    def test_an_old_header_workbook_is_read(self, log_paths, pacific, monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed")], 50.0)
        book = openpyxl.load_workbook(log_paths)
        sheet = book.active
        for column, header in ((12, "x — NO on A"), (13, "y — YES on B"),
                               (14, "Total Cost ($)"), (15, "Min Profit ($)")):
            sheet.cell(row=1, column=column, value=header)
        book.save(log_paths)
        trades, starts, warnings = live_portfolio.read_trade_logs([log_paths])
        assert warnings == []
        assert [t.legs for t in trades] == [(BotLeg("A1", "yes", D(13)), BotLeg("B1", "no", D(13)))]
        assert [s.cash_before for s in starts] == [D("50.00")]

    def test_a_row_repeated_in_a_fallback_copy_is_read_once(self, log_paths, pacific, monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed")], 50.0)
        shutil.copy(log_paths, log_paths.parent / "trade_log_2026-09-28_020000_123456.xlsx")
        trades, starts, warnings = live_portfolio.read_trade_logs(
            live_portfolio.trade_log_paths())
        assert [t.trade_id for t in trades] == ["trade_log.xlsx#3"]
        assert len(starts) == 1 and warnings == []

    def test_a_fallback_copys_own_runs_are_read(self, log_paths, pacific, monkeypatch):
        run1, run2 = T0, T0 + timedelta(hours=1)
        _write_run(monkeypatch, run1, [_trade_result("A1", "B1", "executed")], 50.0)
        # The shared log's lock cannot be taken, so reporter writes run 2 to a fallback copy
        monkeypatch.setattr(reporter, "_acquire_lock", lambda path: None)
        _write_run(monkeypatch, run2, [_trade_result("A2", "B2", "rolled_back", count=7)], 40.0)
        paths = live_portfolio.trade_log_paths()
        assert [p.name for p in paths] == ["trade_log.xlsx",
                                           "trade_log_2026-09-28_030000_000000.xlsx"]
        trades, starts, warnings = live_portfolio.read_trade_logs(paths)
        assert warnings == []
        window = timedelta(seconds=config.LIVE_BOT_RUN_WINDOW_SECONDS)
        assert [(t.trade_id, t.status, t.legs, t.logged_at, t.run_after) for t in trades] == [
            ("trade_log.xlsx#3", "executed",
             (BotLeg("A1", "yes", D(13)), BotLeg("B1", "no", D(13))), run1, run1 - window),
            (f"{paths[1].name}#3", "rolled_back",
             (BotLeg("A2", "yes", D(7)), BotLeg("B2", "no", D(7))), run2, run1)]
        assert [(s.logged_at, s.run_after, s.cash_before) for s in starts] == [
            (run1, run1 - window, D("50.00")), (run2, run1, D("40.00"))]

    def test_date_and_time_cells_of_other_kinds_are_read(self, log_paths, pacific, monkeypatch):
        # A workbook saved by hand can turn the text cells into a date and a time
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed"),
                                     _trade_result("A2", "B2", "executed")], 50.0)
        book = openpyxl.load_workbook(log_paths)
        sheet = book.active
        sheet.cell(row=3, column=1, value=datetime(2026, 9, 28))
        sheet.cell(row=3, column=2, value=time(2, 0, 0))
        sheet.cell(row=4, column=1, value=date(2026, 9, 28))
        book.save(log_paths)
        trades, _, warnings = live_portfolio.read_trade_logs([log_paths])
        assert warnings == []
        assert [(t.trade_id, t.logged_at) for t in trades] == [
            ("trade_log.xlsx#3", T0), ("trade_log.xlsx#4", T0)]

    def test_a_status_with_spaces_is_read_and_an_unknown_one_warns(self, log_paths, pacific,
                                                                   monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed"),
                                     _trade_result("A2", "B2", "executed")], 50.0)
        book = openpyxl.load_workbook(log_paths)
        sheet = book.active
        sheet.cell(row=3, column=17, value="executed ")
        sheet.cell(row=4, column=17, value="half-done")
        book.save(log_paths)
        trades, starts, warnings = live_portfolio.read_trade_logs([log_paths])
        assert [(t.trade_id, t.status) for t in trades] == [("trade_log.xlsx#3", "executed")]
        assert warnings == [f"trade_log.xlsx#4: its status 'half-done' is not one this page "
                            f"knows; its contracts count as {OTHER_BETS}"]
        assert len(starts) == 1

    def test_two_runs_in_the_repeated_hour_keep_their_order(self, log_paths, pacific,
                                                            monkeypatch):
        # On 2026-11-01 Los Angeles reads 01:00-01:59 twice. Run 1 logs at 01:50
        # the first time round, run 2 at 01:10 the second time, 20 minutes later
        run1 = datetime(2026, 11, 1, 8, 50, tzinfo=UTC)
        run2 = datetime(2026, 11, 1, 9, 10, tzinfo=UTC)
        _write_run(monkeypatch, run1, [_trade_result("A1", "B1", "executed")], 50.0)
        _write_run(monkeypatch, run2, [_trade_result("A2", "B2", "executed")], 40.0)
        trades, starts, warnings = live_portfolio.read_trade_logs([log_paths])
        assert warnings == []
        assert [(t.logged_at, t.run_after) for t in trades] == [
            (run1, run1 - timedelta(seconds=config.LIVE_BOT_RUN_WINDOW_SECONDS)), (run2, run1)]
        fills = [_fill("1b", "B1", run1 - timedelta(seconds=2), False, 13),
                 _fill("1a", "A1", run1 - timedelta(seconds=1), True, 13),
                 _fill("2b", "B2", run2 - timedelta(seconds=2), False, 13),
                 _fill("2a", "A2", run2 - timedelta(seconds=1), True, 13)]
        owner, unmatched = live_portfolio.match_bot_fills(trades, fills)
        assert owner == {"1a": "trade_log.xlsx#3", "1b": "trade_log.xlsx#3",
                         "2a": "trade_log.xlsx#5", "2b": "trade_log.xlsx#5"}
        assert unmatched == []

    def test_a_lone_run_in_the_repeated_hour_reads_as_the_later_time(self, log_paths, pacific,
                                                                     monkeypatch):
        # With nothing after it to say which 01:30 it was, the later one is
        # taken: its orders still come before it
        _write_run(monkeypatch, datetime(2026, 11, 1, 8, 30, tzinfo=UTC),
                   [_trade_result("A1", "B1", "executed")], 50.0)
        trades, _, _ = live_portfolio.read_trade_logs([log_paths])
        assert [t.logged_at for t in trades] == [datetime(2026, 11, 1, 9, 30, tzinfo=UTC)]

    def test_only_reporters_fallback_copies_are_read(self, tmp_path, log_paths):
        names = ["trade_log.xlsx", "trade_log copy.xlsx", "trade_log_mine.xlsx",
                 "trade_log_2026-09-28_020000_123456.xlsx",
                 "trade_log_2026-09-28_020000_123456-1.xlsx",
                 "trade_log_2026-09-28_020000_123456-0a1b2c3d.xlsx",
                 "trade_log_2026-09-28_020000.xlsx",
                 "trade_log_2026-09-28_020000_123456.xlsx.tmp"]
        for name in names:
            (tmp_path / name).write_bytes(b"")
        (tmp_path / "trade_log_2026-09-29_020000_123456.xlsx").mkdir()
        assert [p.name for p in live_portfolio.trade_log_paths()] == [
            "trade_log.xlsx", "trade_log_2026-09-28_020000_123456-0a1b2c3d.xlsx",
            "trade_log_2026-09-28_020000_123456-1.xlsx",
            "trade_log_2026-09-28_020000_123456.xlsx"]

    def test_no_trade_log_is_no_path(self, log_paths):
        assert live_portfolio.trade_log_paths() == []

    def test_local_time_is_read_as_utc(self, pacific):
        once = datetime(2026, 9, 28, 9, 27, tzinfo=UTC)
        assert live_portfolio._log_readings("2026-09-28", "02:27:00") == (once, once)
        assert live_portfolio._log_readings("2026-09-28", "02:27") == (once, once)
        winter = datetime(2026, 12, 1, 10, 27, tzinfo=UTC)
        assert live_portfolio._log_readings("2026-12-01", "02:27:00") == (winter, winter)

    def test_the_repeated_hour_has_two_readings(self, pacific):
        # 01:30 happens twice on 2026-11-01 in Los Angeles: 08:30 UTC, then 09:30 UTC
        assert live_portfolio._log_readings("2026-11-01", "01:30:00") == (
            datetime(2026, 11, 1, 8, 30, tzinfo=UTC), datetime(2026, 11, 1, 9, 30, tzinfo=UTC))
        before = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
        assert live_portfolio._log_readings("2026-11-01", "00:30:00") == (before, before)

    @pytest.mark.parametrize("day, clock", [("2026-09-28", None), ("", ""), ("bad", "02:27:00")])
    def test_an_unreadable_time_raises(self, day, clock):
        with pytest.raises(ValueError):
            live_portfolio._log_readings(day, clock)

    def test_a_damaged_file_is_a_warning(self, tmp_path, log_paths, pacific, monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed")], 50.0)
        broken = tmp_path / "trade_log_2026-09-28_020000_123456.xlsx"
        broken.write_bytes(b"not a workbook")
        trades, _, warnings = live_portfolio.read_trade_logs(live_portfolio.trade_log_paths())
        assert [t.trade_id for t in trades] == ["trade_log.xlsx#3"]
        assert warnings == [f"Could not read {broken.name} (BadZipFile): its trades count as "
                            f"{OTHER_BETS}"]

    def test_a_workbook_that_is_not_a_trade_log_is_a_warning(self, tmp_path):
        path = tmp_path / "trade_log.xlsx"
        book = openpyxl.Workbook()
        book.active.append(["Something", "else"])
        book.save(path)
        assert live_portfolio.read_trade_logs([path]) == (
            [], [], [f"trade_log.xlsx is not a trade log this page can read: its trades count "
                     f"as {OTHER_BETS}"])

    def test_unreadable_rows_are_warnings(self, log_paths, pacific, monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed"),
                                     _trade_result("A2", "B2", "executed")], 50.0)
        book = openpyxl.load_workbook(log_paths)
        sheet = book.active
        sheet.cell(row=3, column=2, value="25:99:00")       # time
        sheet.cell(row=4, column=18, value="no sides here")  # notes
        book.save(log_paths)
        trades, _, warnings = live_portfolio.read_trade_logs([log_paths])
        assert trades == []
        assert warnings == [
            "trade_log.xlsx row 3: unreadable date or time",
            f"trade_log.xlsx#4: its sides or counts cannot be read; its contracts count as "
            f"{OTHER_BETS}"]


# ---- which fills are the bot's ---------------------------------------------

def _fill(fill_id, ticker, when, buys_yes, count, yes_price="0.30", fee="0", order_id=None):
    """A Fill as read_account builds one."""
    yes = D(yes_price)
    return Fill(fill_id, f"order-{fill_id}" if order_id is None else order_id, ticker, when,
                buys_yes, D(count), yes, 1 - yes, D(fee))


def _bot(trade_id, legs, logged_at, *, status="executed", run_after=None):
    """A BotTrade: legs are (ticker, side, count) triples, market A's first."""
    return BotTrade(trade_id, status, trade_id, tuple(BotLeg(t, s, D(c)) for t, s, c in legs),
                    run_after or logged_at - timedelta(seconds=config.LIVE_BOT_RUN_WINDOW_SECONDS),
                    logged_at)


class TestMatchBotFills:
    def test_an_exact_order_is_taken_and_no_other(self):
        logged = T0 + timedelta(hours=1)
        trade = _bot("T1", [("A", "yes", 10), ("B", "no", 10)], logged)
        fills = [
            _fill("a1", "A", logged - timedelta(seconds=30), True, 6, order_id="oA"),
            _fill("a2", "A", logged - timedelta(seconds=30), True, 4, "0.31", order_id="oA"),
            _fill("b1", "B", logged - timedelta(seconds=31), False, 8, order_id="partial"),
            _fill("b2", "B", logged - timedelta(seconds=32), False, 12, order_id="oversized"),
            _fill("b3", "B", logged - timedelta(seconds=33), True, 10, order_id="other-way"),
        ]
        owner, warnings = live_portfolio.match_bot_fills([trade], fills)
        assert owner == {"a1": "T1", "a2": "T1"}
        assert warnings == [f"The bot's purchase of 10 NO on B matches no Kalshi order: those "
                            f"contracts count as {OTHER_BETS}"]

    def test_consecutive_runs_of_one_pair_each_keep_their_own_orders(self):
        legs = [("A", "yes", 5), ("B", "no", 5)]
        logged_n = T0
        logged_next = T0 + timedelta(hours=1)
        run_n = _bot("N", legs, logged_n)
        run_next = _bot("N+1", legs, logged_next, run_after=logged_n)
        fills = [
            _fill("nB", "B", logged_n - timedelta(seconds=31), False, 5),
            _fill("nA", "A", logged_n - timedelta(seconds=30), True, 5),
            _fill("xB", "B", logged_n + timedelta(seconds=60), False, 5),
            _fill("xA", "A", logged_n + timedelta(seconds=61), True, 5),
        ]
        owner, warnings = live_portfolio.match_bot_fills([run_next, run_n], fills)
        assert owner == {"nA": "N", "nB": "N", "xA": "N+1", "xB": "N+1"}
        assert warnings == []

    @pytest.mark.parametrize("after", [0.2, 0.4, 0.6, 0.999])
    def test_orders_inside_the_logged_second_stay_with_their_run(self, after):
        # The Time cell shows the whole second, rounded down, so a run's last
        # orders can land just after the time shown (0 to 0.7 s on the real
        # account). Two runs 30 minutes apart buy the same legs and sizes.
        legs = [("A", "yes", 5), ("B", "no", 5)]
        logged_n = datetime(2026, 10, 5, 16, 0, 5, tzinfo=UTC)
        logged_next = logged_n + timedelta(minutes=30)
        run_n = _bot("N", legs, logged_n)
        run_next = _bot("N+1", legs, logged_next, run_after=logged_n)
        fills = []
        for name, logged in (("n", logged_n), ("m", logged_next)):
            fills += [_fill(f"{name}B", "B", logged - timedelta(seconds=1.5), False, 5),
                      _fill(f"{name}A", "A", logged + timedelta(seconds=after), True, 5)]
        owner, warnings = live_portfolio.match_bot_fills([run_n, run_next], fills)
        assert owner == {"nA": "N", "nB": "N", "mA": "N+1", "mB": "N+1"}
        assert warnings == []

    def test_the_previous_runs_logged_second_is_not_the_next_runs(self):
        # The previous run (whose pair sent nothing that filled) logged at
        # 16:00:05; an order inside that second is not the next run's, whose
        # own order lands 30 s past its log time (its clock running behind)
        logged_prev = datetime(2026, 10, 5, 16, 0, 5, tzinfo=UTC)
        logged = logged_prev + timedelta(minutes=30)
        run = _bot("N+1", [("A", "yes", 5), ("B", "no", 5)], logged, run_after=logged_prev)
        fills = [_fill("yours", "A", logged_prev + timedelta(seconds=0.5), True, 5),
                 _fill("b", "B", logged - timedelta(seconds=2), False, 5),
                 _fill("a", "A", logged + timedelta(seconds=30), True, 5)]
        owner, warnings = live_portfolio.match_bot_fills([run], fills)
        assert owner == {"a": "N+1", "b": "N+1"}
        assert warnings == []

    def test_an_order_just_past_the_log_time_is_found(self):
        logged = T0
        trade = _bot("T1", [("A", "yes", 5), ("B", "no", 5)], logged)
        fills = [
            _fill("b", "B", logged + timedelta(seconds=90), False, 5),
            _fill("a", "A", logged + timedelta(seconds=91), True, 5),
            # Past the logged second and the clock allowance: never the bot's
            _fill("late", "A", logged + timedelta(
                seconds=config.LIVE_TRADE_LOG_TIME_STEP_SECONDS
                + config.LIVE_BOT_RUN_CLOCK_SLACK_SECONDS + 1), True, 5),
        ]
        owner, warnings = live_portfolio.match_bot_fills([trade], fills)
        assert owner == {"a": "T1", "b": "T1"}
        assert warnings == []

    def test_in_the_window_the_latest_order_wins_and_past_it_the_earliest(self):
        logged = T0
        inside = _bot("inside", [("A", "yes", 5)] * 2, logged)
        fills = [_fill("early", "A", logged - timedelta(minutes=30), True, 5),
                 _fill("late", "A", logged - timedelta(minutes=1), True, 5)]
        owner, _ = live_portfolio.match_bot_fills([inside], fills)
        assert owner == {"early": "inside", "late": "inside"}
        one_leg = BotTrade("one", "executed", "one",
                           (BotLeg("A", "yes", D(5)), BotLeg("C", "no", D(1))),
                           logged - timedelta(hours=1), logged)
        assert live_portfolio.match_bot_fills([one_leg], fills)[0] == {"late": "one"}
        past = [_fill("p1", "A", logged + timedelta(seconds=10), True, 5),
                _fill("p2", "A", logged + timedelta(seconds=20), True, 5)]
        assert live_portfolio.match_bot_fills([one_leg], past)[0] == {"p1": "one"}

    def test_a_rolled_back_row_claims_its_no_purchase_and_its_unwind(self):
        logged = T0
        rolled = _bot("R", [("A", "yes", 13), ("B", "no", 13)], logged, status="rolled_back")
        retry = _bot("S", [("A", "yes", 13), ("B", "no", 13)], logged + timedelta(seconds=90),
                     status="rolled_back", run_after=logged)
        fills = [
            _fill("r-no", "B", logged - timedelta(seconds=20), False, 13),
            _fill("someone", "A", logged - timedelta(seconds=19), True, 13),
            _fill("r-unwind", "B", logged - timedelta(seconds=18), True, 13, "0.70"),
            # The retry, within the first row's clock allowance: its own orders
            _fill("s-no", "B", logged + timedelta(seconds=60), False, 13),
            _fill("s-unwind", "B", logged + timedelta(seconds=62), True, 13, "0.70"),
        ]
        owner, warnings = live_portfolio.match_bot_fills([rolled, retry], fills)
        assert owner == {"r-no": "R", "r-unwind": "R", "s-no": "S", "s-unwind": "S"}
        assert warnings == []

    def test_an_executed_retry_keeps_its_own_orders(self):
        logged = T0
        rolled = _bot("R", [("A", "yes", 13), ("B", "no", 13)], logged, status="rolled_back")
        retry = _bot("S", [("A", "yes", 13), ("B", "no", 13)], logged + timedelta(seconds=90),
                     run_after=logged)
        fills = [
            _fill("r-no", "B", logged - timedelta(seconds=20), False, 13),
            _fill("r-unwind", "B", logged - timedelta(seconds=18), True, 13, "0.70"),
            _fill("s-no", "B", logged + timedelta(seconds=60), False, 13),
            _fill("s-yes", "A", logged + timedelta(seconds=61), True, 13),
        ]
        owner, warnings = live_portfolio.match_bot_fills([rolled, retry], fills)
        assert owner == {"r-no": "R", "r-unwind": "R", "s-no": "S", "s-yes": "S"}
        assert warnings == []

    def test_a_partial_unwind_claims_only_up_to_the_no_count(self):
        logged = T0
        failed = _bot("F", [("A", "yes", 10), ("B", "no", 10)], logged, status="rollback_failed")
        fills = [
            _fill("no", "B", logged - timedelta(seconds=20), False, 10),
            _fill("part", "B", logged - timedelta(seconds=19), True, 4, "0.70"),
            # Your own buy-back later the same minute: more than the bot's NO left
            _fill("yours", "B", logged + timedelta(seconds=30), True, 7, "0.70"),
        ]
        owner, warnings = live_portfolio.match_bot_fills([failed], fills)
        assert owner == {"no": "F", "part": "F"}
        assert warnings == []

    @pytest.mark.parametrize("first_unwound, retry_status, retry_unwound", [
        (0, "rolled_back", 10), (4, "rollback_failed", 6)])
    def test_a_retrys_unwind_stays_with_the_retry(self, first_unwound, retry_status,
                                                  retry_unwound):
        # The first row's unwind failed (wholly, or after 4 of 10); 90 s later a
        # retry rolls back. Its unwind lies inside the first row's allowance too.
        logged = T0
        legs = [("A", "yes", 10), ("B", "no", 10)]
        first = _bot("R1", legs, logged, status="rollback_failed")
        retry = _bot("R2", legs, logged + timedelta(seconds=90), status=retry_status,
                     run_after=logged)
        fills = [_fill("r1-no", "B", logged - timedelta(seconds=20), False, 10)]
        if first_unwound:
            fills.append(_fill("r1-unwind", "B", logged - timedelta(seconds=19), True,
                               first_unwound, "0.70"))
        fills += [_fill("r2-no", "B", logged + timedelta(seconds=60), False, 10),
                  _fill("r2-unwind", "B", logged + timedelta(seconds=62), True, retry_unwound,
                        "0.70")]
        owner, warnings = live_portfolio.match_bot_fills([first, retry], fills)
        expected = {"r1-no": "R1", "r2-no": "R2", "r2-unwind": "R2"}
        if first_unwound:
            expected["r1-unwind"] = "R1"
        assert owner == expected
        assert warnings == []

    def test_a_manual_review_row_claims_no_buy_back(self):
        # On manual review the bot sends no unwind: a bid on its NO market just
        # after the alert is your own (a flatten by hand), not the bot's
        review = _bot("MR", [("A", "yes", 7), ("B", "no", 7)], T0, status="manual_review")
        fills = [_fill("mr-no", "B", T0 - timedelta(seconds=20), False, 7),
                 _fill("flatten", "B", T0 + timedelta(seconds=60), True, 7, "0.70")]
        owner, warnings = live_portfolio.match_bot_fills([review], fills)
        assert owner == {"mr-no": "MR"}
        assert warnings == []

    def test_unmatched_rows_that_were_not_executed_are_silent(self):
        review = _bot("M", [("A", "yes", 3), ("B", "no", 3)], T0, status="manual_review")
        assert live_portfolio.match_bot_fills([review], []) == ({}, [])

    def test_a_fill_without_an_order_id_is_its_own_order(self):
        trade = _bot("T1", [("A", "yes", 5), ("B", "no", 3)], T0)
        fills = [_fill("x", "A", T0 - timedelta(seconds=9), True, 2, order_id=""),
                 _fill("y", "A", T0 - timedelta(seconds=9), True, 3, order_id=""),
                 _fill("z", "B", T0 - timedelta(seconds=9), False, 3, order_id="")]
        owner, _ = live_portfolio.match_bot_fills([trade], fills)
        assert owner == {"z": "T1"}

    def test_your_close_partners_are_your_sales(self):
        fills = [_fill("bot", "A", T0, True, 1),
                 _fill("bot-c", "C", T0, False, 2),
                 _fill("y1", "A", T0 + timedelta(minutes=1), False, 1),   # sells YES A
                 _fill("y2", "B", T0 + timedelta(minutes=5), True, 1),    # buys YES B
                 _fill("y3", "C", T0 + timedelta(minutes=8), True, 1),    # sells NO C
                 _fill("y4", "D", T0 + timedelta(minutes=30), True, 1)]   # buys YES D
        partners = live_portfolio.manual_partners(fills, {"bot": "T1", "bot-c": "T1"})
        # Only a sale (a fill that closes what the account holds) is a partner
        assert partners == {"y1": frozenset({"C"}), "y2": frozenset({"A", "C"}),
                            "y3": frozenset({"A"}), "y4": frozenset()}


class TestTradeLogToOwners:
    def test_rows_written_by_reporter_own_their_orders(self, log_paths, pacific, monkeypatch):
        _write_run(monkeypatch, T0, [_trade_result("A1", "B1", "executed", count=13)], 50.0)
        trades, _, _ = live_portfolio.read_trade_logs(live_portfolio.trade_log_paths())
        fills = [_fill("no", "B1", T0 - timedelta(seconds=3), False, 13),
                 _fill("yes", "A1", T0 - timedelta(seconds=2), True, 13)]
        owner, warnings = live_portfolio.match_bot_fills(trades, fills)
        assert owner == {"no": "trade_log.xlsx#3", "yes": "trade_log.xlsx#3"}
        assert warnings == []


# ---- the ledger ------------------------------------------------------------

def _ledger(fills, owners=None, payouts=(), legs=None):
    """build_ledger over fills; owners maps fill_id -> trade_id (the rest are Other bets)."""
    return live_portfolio.build_ledger(fills, payouts, owners or {}, legs or {})


def _holdings(ledger) -> dict:
    """Contracts each (owner, ticker, side) holds after every event, without zeros."""
    held = defaultdict(D)
    for e in ledger.events:
        held[(e.owner, e.ticker, e.side)] += e.contracts
    return {key: count for key, count in held.items() if count}


def _tuples(ledger):
    """Each event as (owner, ticker, side, contracts, cash, spent, basis)."""
    return [(e.owner, e.ticker, e.side, e.contracts, e.cash, e.spent, e.basis)
            for e in ledger.events]


class TestLedger:
    def test_cash_for_each_kind_of_fill(self):
        t = [T0 + timedelta(minutes=i) for i in range(4)]
        ledger = _ledger([
            _fill("yes-open", "Y", t[0], True, 10, "0.30", "0.07"),
            _fill("yes-close", "Y", t[1], False, 10, "0.45", "0.05"),
            _fill("no-open", "N", t[2], False, 5, "0.60", "0.03"),
            _fill("no-close", "N", t[3], True, 5, "0.70", "0.02"),
        ])
        assert _tuples(ledger) == [
            (OTHER_BETS, "Y", "yes", D(10), D("-3.07"), D("3.07"), D("3.07")),
            (OTHER_BETS, "Y", "yes", D(-10), D("4.45"), D(0), D("-3.07")),
            (OTHER_BETS, "N", "no", D(5), D("-2.03"), D("2.03"), D("2.03")),
            (OTHER_BETS, "N", "no", D(-5), D("1.48"), D(0), D("-2.03")),
        ]
        assert ledger.held_now == {} and ledger.warnings == ()

    def test_a_fill_crossing_zero_is_split(self):
        ledger = _ledger([_fill("open", "M", T0, False, 5, "0.60"),
                          _fill("cross", "M", T0 + timedelta(minutes=1), True, 8, "0.70", "0.08")])
        assert _tuples(ledger)[1:] == [
            (OTHER_BETS, "M", "no", D(-5), D("1.45"), D(0), D("-2.00")),
            (OTHER_BETS, "M", "yes", D(3), D("-2.13"), D("2.13"), D("2.13")),
        ]
        assert ledger.held_now == {"M": D(3)}

    def test_fees_add_back_exactly_over_many_pieces(self):
        # Three lots closed and the rest opened: four pieces of one fill's fee
        opens = [_fill(f"o{i}", "M", T0 + timedelta(seconds=i), True, 1, "0.30") for i in range(3)]
        sell = _fill("sell", "M", T0 + timedelta(minutes=1), False, 6, "0.40", "0.07")
        ledger = _ledger([*opens, sell], {"o0": "T1", "o1": "T2"})
        pieces = [e for e in ledger.events if e.time == sell.time]
        # What each piece's contracts are worth at the fill's prices, less its cash
        fees = [(-e.contracts * sell.yes_price if e.side == "yes"
                 else -e.contracts * sell.no_price) - e.cash for e in pieces]
        assert fees == [D("0.011666666667")] * 3 + [D("0.034999999999")]
        assert sum(fees, D(0)) == D("0.07")
        assert [e.owner for e in pieces] == [OTHER_BETS, "T1", "T2", OTHER_BETS]
        assert ledger.held_now == {"M": D(-3)}

    def test_a_closed_lot_keeps_no_cost(self):
        ledger = _ledger([_fill("open", "M", T0, True, 3, "0.33", "0.01"),
                          *[_fill(f"c{i}", "M", T0 + timedelta(minutes=i + 1), False, 1, "0.50")
                            for i in range(3)]])
        assert sum((e.basis for e in ledger.events), D(0)) == 0
        closes = [e.basis for e in ledger.events[1:]]
        assert closes == [D("-0.333333333333"), D("-0.333333333334"), D("-0.333333333333")]
        assert ledger.held_now == {}

    def test_an_unwind_takes_its_own_lots_first(self):
        fills = [_fill("yours", "B", T0, False, 5),
                 _fill("bot-no", "B", T0 + timedelta(minutes=1), False, 5),
                 _fill("unwind", "B", T0 + timedelta(minutes=2), True, 5, "0.70")]
        ledger = _ledger(fills, {"bot-no": "T", "unwind": "T"})
        assert _holdings(ledger) == {(OTHER_BETS, "B", "no"): D(5)}

    def test_a_pair_sold_by_hand_comes_out_of_its_own_purchase(self):
        # Two bot pairs share market A; you sell the newer pair (A and B2) by hand
        fills = [_fill("t1-b", "B1", T0, False, 10),
                 _fill("t1-a", "A", T0 + timedelta(seconds=1), True, 10),
                 _fill("t2-b", "B2", T0 + timedelta(hours=1), False, 10),
                 _fill("t2-a", "A", T0 + timedelta(hours=1, seconds=1), True, 10),
                 _fill("sell-a", "A", T0 + timedelta(days=1), False, 10, "0.50"),
                 _fill("buy-b2", "B2", T0 + timedelta(days=1, minutes=4), True, 10, "0.80")]
        owners = {"t1-b": "T1", "t1-a": "T1", "t2-b": "T2", "t2-a": "T2"}
        legs = {"T1": frozenset({"A", "B1"}), "T2": frozenset({"A", "B2"})}
        ledger = _ledger(fills, owners, legs=legs)
        assert _holdings(ledger) == {("T1", "B1", "no"): D(10), ("T1", "A", "yes"): D(10)}

    def test_without_a_partner_sale_the_oldest_purchase_goes_first(self):
        fills = [_fill("t1-a", "A", T0, True, 10),
                 _fill("t2-a", "A", T0 + timedelta(hours=1), True, 10),
                 _fill("sell-a", "A", T0 + timedelta(days=1), False, 10, "0.50")]
        legs = {"T1": frozenset({"A", "B1"}), "T2": frozenset({"A", "B2"})}
        ledger = _ledger(fills, {"t1-a": "T1", "t2-a": "T2"}, legs=legs)
        assert _holdings(ledger) == {("T2", "A", "yes"): D(10)}

    def test_a_purchase_beside_your_sale_does_not_pair_it(self):
        # You sell A, then 3 minutes later BUY NO on T2's other market: that is
        # no sale of T2's pair, so the sale closes the oldest purchase
        fills = [_fill("t1-a", "A", T0, True, 10),
                 _fill("t2-a", "A", T0 + timedelta(hours=1), True, 10),
                 _fill("sell-a", "A", T0 + timedelta(days=1), False, 10, "0.50"),
                 _fill("open-b2", "B2", T0 + timedelta(days=1, minutes=3), False, 5, "0.40")]
        legs = {"T1": frozenset({"A", "B1"}), "T2": frozenset({"A", "B2"})}
        ledger = _ledger(fills, {"t1-a": "T1", "t2-a": "T2"}, legs=legs)
        assert _holdings(ledger) == {("T2", "A", "yes"): D(10), (OTHER_BETS, "B2", "no"): D(5)}

    def test_your_sale_beside_a_purchase_takes_your_own_lots(self):
        # You sell the 4 YES A you bought yourself, then add 1 NO to B1, which
        # the bot holds NO of: the sale still comes out of your own lot
        fills = [_fill("t1-b", "B1", T0, False, 10),
                 _fill("t1-a", "A", T0 + timedelta(seconds=1), True, 10),
                 _fill("yours", "A", T0 + timedelta(hours=1), True, 4),
                 _fill("sell", "A", T0 + timedelta(days=1), False, 4, "0.50"),
                 _fill("more-b1", "B1", T0 + timedelta(days=1, minutes=2), False, 1, "0.40")]
        ledger = _ledger(fills, {"t1-b": "T1", "t1-a": "T1"},
                         legs={"T1": frozenset({"A", "B1"})})
        assert _holdings(ledger) == {("T1", "B1", "no"): D(10), ("T1", "A", "yes"): D(10),
                                     (OTHER_BETS, "B1", "no"): D(1)}

    def test_your_sale_takes_your_own_lots_before_the_bots(self):
        fills = [_fill("bot", "A", T0, True, 10),
                 _fill("yours", "A", T0 + timedelta(hours=1), True, 5),
                 _fill("sell", "A", T0 + timedelta(days=1), False, 7, "0.50")]
        ledger = _ledger(fills, {"bot": "T"}, legs={"T": frozenset({"A", "B"})})
        assert _holdings(ledger) == {("T", "A", "yes"): D(8)}

    @pytest.mark.parametrize("result, yes_value, paid_yes, paid_no", [
        ("yes", None, D(10), D(0)),
        ("no", None, D(0), D(4)),
        ("scalar", D("0.35"), D("3.50"), D("2.60")),
    ])
    def test_a_payout_pays_the_winning_side(self, result, yes_value, paid_yes, paid_no):
        fills = [_fill("y", "Y", T0, True, 10), _fill("n", "N", T0, False, 4)]
        settled = T0 + timedelta(days=1)
        payouts = [Payout("Y", settled, result, yes_value, None),
                   Payout("N", settled, result, yes_value, None)]
        ledger = _ledger(fills, {"y": "T"}, payouts)
        paid = {(e.ticker, e.owner): (e.contracts, e.cash, e.basis)
                for e in ledger.events if e.time == settled}
        assert paid == {("Y", "T"): (D(-10), paid_yes, D("-3.00")),
                        ("N", OTHER_BETS): (D(-4), paid_no, D("-2.80"))}
        assert ledger.held_now == {} and ledger.warnings == ()

    def test_revenue_that_disagrees_warns(self):
        fills = [_fill("y", "Y", T0, True, 10)]
        settled = T0 + timedelta(days=1)
        assert _ledger(fills, payouts=[Payout("Y", settled, "yes", None, D("10.00"))]).warnings == ()
        assert _ledger(fills, payouts=[Payout("Y", settled, "yes", None, D("9.99"))]).warnings == ()
        assert _ledger(fills, payouts=[Payout("Y", settled, "yes", None, D("9.98"))]).warnings == (
            "Y paid $9.98 but the contracts held add up to $10",)

    def test_an_unreadable_result_warns_and_keeps_the_contracts(self):
        ledger = _ledger([_fill("y", "Y", T0, True, 10)],
                         payouts=[Payout("Y", T0 + timedelta(days=1), "void", None, D(0))])
        assert ledger.held_now == {"Y": D(10)}
        assert ledger.warnings == ("Y settled as 'void' with no payout this page can read",)

    def test_a_fill_and_a_payout_at_one_moment_take_the_fill_first(self):
        ledger = _ledger([_fill("y", "Y", T0, True, 10)],
                         payouts=[Payout("Y", T0, "yes", None, D(10))])
        assert ledger.held_now == {} and ledger.warnings == ()


def _account(fills=(), payouts=(), positions=None, read_at=None) -> Account:
    """An Account built directly, as read_account would return it."""
    return Account(read_at or T0 + timedelta(days=10), D(100), 1, None, positions or {},
                   tuple(fills), tuple(payouts), (), False)


class TestRebuiltPayouts:
    def test_payouts_kalshi_no_longer_lists_are_rebuilt(self):
        settled = T0 + timedelta(days=2)
        fills = [_fill(f"f{t}", t, T0, True, 3) for t in ("M1", "M2", "M3", "M4", "M5", "M6")]
        fills.append(_fill("f-M6-close", "M6", T0 + timedelta(hours=1), False, 3))
        listed = Payout("M4", settled, "no", None, D(0))
        account = _account(fills, [listed], positions={"M5": D(3)})
        assert live_portfolio.tickers_needing_payout(account) == {"M1", "M2", "M3"}
        markets = {
            "M1": Market("M1", "E1", "One", "finalized", "yes", settled, D(1), None, None, None),
            "M3": Market("M3", "E3", "Three", "closed", "", None, None, None, None, None),
        }
        payouts, warnings = live_portfolio.all_payouts(account, markets)
        assert payouts == [listed, Payout("M1", settled, "yes", D(1), None)]
        assert warnings == [
            "Market M2 could not be looked up: its contracts count as still held",
            "Market M3 has no settlement this page can read: its contracts count as still held"]

    @pytest.mark.parametrize("result, settled", [
        ("void", T0 + timedelta(days=2)), ("", T0 + timedelta(days=2)), ("yes", None)])
    def test_a_found_market_with_no_readable_settlement_warns(self, result, settled):
        # Kalshi holds none of these contracts and lists no settlement for them
        account = _account([_fill("f", "OLD", T0, True, 3)])
        market = Market("OLD", "E", "Old", "finalized", result, settled, None, None, None, None)
        assert live_portfolio.all_payouts(account, {"OLD": market}) == (
            [], ["Market OLD has no settlement this page can read: its contracts count as "
                 "still held"])

    def test_a_settlement_after_the_read_is_not_rebuilt(self):
        account = _account([_fill("f", "M1", T0, True, 3)], read_at=T0 + timedelta(days=1))
        later = Market("M1", "E1", "One", "finalized", "yes", T0 + timedelta(days=2), D(1),
                       None, None, None)
        assert live_portfolio.all_payouts(account, {"M1": later}) == ([], [])


# ---- the ledger's cash against a running-position formula ------------------

def _oracle(fills, payouts):
    """
    Cash and holdings by a simple running position, independent of the ledger's lots.

    A bid closes NO held (paid its NO price) then opens YES (its YES price);
    an ask closes YES held (paid its YES price) then opens NO (its NO price);
    the fee comes off; a payout pays the position for its side.
    """
    position = defaultdict(D)
    cash = D(0)
    for fill in sorted(fills, key=lambda f: (f.time, f.fill_id)):
        held = position[fill.ticker]
        if fill.buys_yes:
            close = min(fill.count, max(-held, D(0)))
            cash += close * fill.no_price - (fill.count - close) * fill.yes_price
            position[fill.ticker] = held + fill.count
        else:
            close = min(fill.count, max(held, D(0)))
            cash += close * fill.yes_price - (fill.count - close) * fill.no_price
            position[fill.ticker] = held - fill.count
        cash -= fill.fee
    for payout in payouts:
        held = position[payout.ticker]
        yes = {"yes": D(1), "no": D(0), "scalar": payout.yes_value}[payout.result]
        cash += max(held, D(0)) * yes + max(-held, D(0)) * (1 - yes)
        position[payout.ticker] = D(0)
    return cash, {t: n for t, n in position.items() if n}


def _random_case(seed: int):
    """Random fills (and sometimes payouts) over three markets, with random owners."""
    rng = random.Random(seed)
    tickers = ["M1", "M2", "M3"]
    owners = ["T1", "T2", "T3", OTHER_BETS]
    moment = T0
    fills = []
    for i in range(rng.randint(1, 40)):
        moment += timedelta(seconds=rng.choice([0, 0, 1, 30, 300, 3600]))
        count = rng.choice([D(1), D(2), D(3), D(5), D(8), D("0.5"), D("2.25"), D("13")])
        fee = D(rng.randint(0, 300)) / 10000
        fills.append(_fill(f"f{i:04d}", rng.choice(tickers), moment, rng.random() < 0.5,
                           count, f"{rng.randint(1, 99) / 100:.2f}", str(fee)))
    owner_by_fill = {}
    for fill in fills:
        owner = rng.choice(owners)
        if owner != OTHER_BETS:
            owner_by_fill[fill.fill_id] = owner
    legs = {o: frozenset(rng.sample(tickers, 2)) for o in owners[:3]}
    payouts = []
    if rng.random() < 0.5:
        for ticker in rng.sample(tickers, rng.randint(1, 3)):
            result = rng.choice(["yes", "no", "scalar"])
            value = D(rng.randint(0, 100)) / 100 if result == "scalar" else None
            payouts.append(Payout(ticker, moment + timedelta(days=1), result, value, None))
    return fills, owner_by_fill, legs, payouts


class TestLedgerMatchesARunningPosition:
    @pytest.mark.parametrize("seed", range(300))
    def test_cash_and_holdings(self, seed):
        fills, owner_by_fill, legs, payouts = _random_case(seed)
        ledger = live_portfolio.build_ledger(fills, payouts, owner_by_fill, legs)
        cash, held = _oracle(fills, payouts)
        assert sum((e.cash for e in ledger.events), D(0)) == cash
        assert ledger.held_now == held
        if not payouts:
            signed = defaultdict(D)
            for fill in fills:
                signed[fill.ticker] += fill.count if fill.buys_yes else -fill.count
            assert ledger.held_now == {t: n for t, n in signed.items() if n}
        # Contracts an owner no longer holds keep no cost
        contracts, basis = defaultdict(D), defaultdict(D)
        for e in ledger.events:
            contracts[(e.owner, e.ticker, e.side)] += e.contracts
            basis[(e.owner, e.ticker, e.side)] += e.basis
        for key, count in contracts.items():
            assert count >= 0
            if count == 0:
                assert basis[key] == 0
        assert ledger.warnings == ()
