"""
File: test_market_details.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for market_details.py: the batched, cached market-details loader
    (timing and full records, the /historical fallback, only settled records
    cached), the trades walk (paging, both endpoints, the two bounds, a
    cut-short walk cached nowhere, repeated ids), each integrity check firing
    on a crafted failure and staying silent on a real market whose candles,
    trades and total volume agree, the candle-order check, the scheduled-close
    capture, the two commands, the backtester's use of all of it
    (_fetch_projected_candles, _audit_pair_markets, _log_extra_data and one
    _prepare_candidates run), and that no live module imports the module.

Dependencies:
    Imports kalshi_betting.market_details, historical, backtester and config.
    Fully offline: a FakeExchange stands in for market_details._get (the
    conftest session guard has already replaced it with an offline stub), and
    the three caches point at the test's tmp_path.

Notes:
    The real-market fixtures are Kalshi payloads saved under
    tests/fixtures/kalshi_payloads/: market_archive_mlb_game.json and
    candles_archive_mlb_game.json are one MLB game whose hourly candles add up
    to its volume_fp (2,075,659.36) exactly. The saved trade page is short, so
    the trades of that game are built from its candles: two trades an hour,
    adding up to the hour's volume.
"""
import ast
import gzip
import importlib
import inspect
import json
import logging
import pkgutil
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import kalshi_betting
from kalshi_betting import backtester, config, historical, market_details

FIXTURES = Path(__file__).parent / "fixtures" / "kalshi_payloads"
PREFIX = historical._API_PREFIX
MLB = "KXMLBGAME-26AUG161605COLSF-COL"


def _load(name):
    """Read one saved Kalshi payload."""
    return json.loads((FIXTURES / name).read_text())


def _market(ticker, **over):
    """A settled market payload shaped like the saved MLB game's, under another ticker."""
    market = dict(_load("market_archive_mlb_game.json")["market"])
    market["ticker"] = ticker
    market.update(over)
    return market


def _mlb_candles():
    """The saved MLB game's full hourly candles, parsed by the real candle reader."""
    payload = _load("candles_archive_mlb_game.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(historical, "_historical_get", lambda client, path, **params: payload)
        candles, _raw, dropped = historical._fetch_candle_pages(
            None, "p", [(0, 1)], 0, {"request": 0})
    assert dropped == 0
    return candles


def _contiguous(candles):
    """The candles with a zero-volume hour inserted for every hour the series skips.

    The saved game's series has no candle for some hours (nothing traded), so
    the whole-life volume check, which wants every hour from the first to the
    last, skips it; this fills the hours in without changing the total."""
    by_ts = {c["ts"]: c for c in candles}
    first, last = min(by_ts), max(by_ts)
    return [by_ts.get(t) or {"ts": t, "volume": 0.0}
            for t in range(first, last + 3600, 3600)]


def _stamp(seconds):
    """An ISO time with microseconds and a Z, as Kalshi writes trade times."""
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _trade_payload(ticker, trade_id, seconds, count, yes="0.5000", no="0.5000"):
    """One trade object as the trades endpoints send it."""
    return {"count_fp": str(count), "created_time": _stamp(seconds), "is_block_trade": False,
            "no_price_dollars": no, "taker_book_side": "ask", "taker_outcome_side": "no",
            "taker_side": "no", "ticker": ticker, "trade_id": trade_id,
            "yes_price_dollars": yes}


def _mlb_trade_payloads(candles, ticker=MLB):
    """Two trades inside each hour that traded, adding up to the hour's volume exactly."""
    out = []
    for candle in candles:
        volume = Decimal(repr(candle["volume"]))
        if volume == 0:
            continue
        first = (volume * Decimal("0.4")).quantize(Decimal("0.01"))
        for k, part in enumerate((first, volume - first)):
            out.append(_trade_payload(
                ticker, f"{ticker}-{candle['ts']}-{k}", candle["ts"] - 3000 + k * 1200, part))
    return out


def _time_ms_of(text):
    """Unix milliseconds of an ISO time."""
    return market_details._time_ms(text)


def _tuples(payloads):
    """The (ms, YES price, contracts, side) form of trade payloads, oldest first."""
    parsed = [market_details._parse_trade(p)[0] for p in payloads]
    return sorted(parsed, key=lambda t: t[0])


class FakeExchange:
    """
    Stand-in for market_details._get: serves markets, trades, the cutoff and open markets.

    Attributes:
        markets (dict): ticker -> payload served by /markets.
        archived (dict): ticker -> payload served by /historical/markets.
        trades (dict): ticker -> trade payloads (any order) served by both trades endpoints.
        trade_status (dict): endpoint suffix ("historical/trades", "markets/trades") ->
            an HTTP status to fail with instead of serving.
        broken (dict): path suffix -> exception to raise.
        open_pages (list): pages for /markets?status=open, each (markets, cursor).
        requests (list): (path, params) of every request.
    """

    def __init__(self):
        self.markets = {}
        self.archived = {}
        self.trades = {}
        self.trade_status = {}
        self.broken = {}
        self.open_pages = []
        self.cutoff = {"trades_created_ts": "2026-08-17T00:00:00Z"}
        self.requests = []

    def paths(self, suffix):
        """The requests whose path ends with suffix."""
        return [(p, q) for p, q in self.requests if p.endswith(suffix)]

    def __call__(self, client, path, **params):
        self.requests.append((path, dict(params)))
        for suffix, exc in self.broken.items():
            if path.endswith(suffix):
                raise exc
        if path.endswith("/historical/cutoff"):
            return self.cutoff
        if path.endswith("/historical/markets"):
            return {"markets": [self.archived[t] for t in params["tickers"].split(",")
                                if t in self.archived]}
        if path.endswith("/markets") and params.get("status") == "open":
            markets, cursor = self.open_pages[int(params.get("cursor") or 0)]
            return {"markets": markets, "cursor": cursor}
        if path.endswith("/markets"):
            return {"markets": [self.markets[t] for t in params["tickers"].split(",")
                                if t in self.markets]}
        for suffix in ("historical/trades", "markets/trades"):
            if path.endswith(suffix):
                if suffix in self.trade_status:
                    raise _HttpError(self.trade_status[suffix])
                everything = sorted(self.trades.get(params["ticker"], []),
                                    key=lambda t: t["created_time"], reverse=True)
                start = int(params.get("cursor") or 0)
                limit = params["limit"]
                page = everything[start:start + limit]
                more = start + limit < len(everything)
                return {"trades": page, "cursor": str(start + limit) if more else ""}
        raise AssertionError(f"unexpected request {path}")


class _HttpError(Exception):
    """An error carrying an HTTP status, as the SDK's ApiException does."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


@pytest.fixture
def exchange(monkeypatch):
    """A FakeExchange in place of market_details._get, with no pause after requests."""
    fake = FakeExchange()
    monkeypatch.setattr(market_details, "_get", fake)
    return fake


def _read_gz(path):
    """Parse one gzipped JSON file."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def _details_lines():
    """Every line of the details cache, oldest first."""
    with gzip.open(market_details._DETAILS_CACHE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _cached_details():
    """The details cache as ticker -> latest record."""
    return {r["ticker"]: r for r in _details_lines()}


def _write_details(records):
    """Replace the details cache with these records, one gzip JSON line each."""
    market_details._DETAILS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(market_details._DETAILS_CACHE, "wb") as fh:
        fh.write(b"".join(json.dumps(r).encode() + b"\n" for r in records))


class TestLoadMarketDetails:
    """Batches of 100, /markets then /historical/markets, timing versus full records."""

    def test_tickers_are_read_100_at_a_time(self, exchange):
        tickers = [f"T{i:03d}" for i in range(250)]
        exchange.markets = {t: _market(t) for t in tickers}
        stats = {}
        got = market_details.load_market_details(MagicMock(), tickers, stats_out=stats)
        assert sorted(got) == sorted(tickers)
        asked = exchange.paths("/markets")
        assert sorted(len(q["tickers"].split(",")) for _, q in asked) == [50, 100, 100]
        assert all(q["limit"] == config.MARKET_DETAILS_PAGE_SIZE for _, q in asked)
        assert exchange.paths("/historical/markets") == []
        assert stats["requested"] == 250
        assert stats["fetched"] == 250
        assert stats["from_cache"] == 0

    def test_markets_the_live_endpoint_lacks_come_from_the_archive(self, exchange):
        live = [f"L{i}" for i in range(3)]
        old = [f"O{i}" for i in range(2)]
        exchange.markets = {t: _market(t) for t in live}
        exchange.archived = {t: _market(t) for t in old}
        stats = {}
        got = market_details.load_market_details(MagicMock(), live + old + ["GONE"],
                                                 stats_out=stats)
        assert sorted(got) == sorted(live + old)
        [(_, query)] = exchange.paths("/historical/markets")
        assert sorted(query["tickers"].split(",")) == sorted(old + ["GONE"])
        assert stats["not_found"] == 1
        assert stats["failed"] == 0

    def test_timing_records_hold_timing_fields_and_full_records_the_rest(self, exchange):
        exchange.markets = {"A": _market("A"), "B": _market("B")}
        got = market_details.load_market_details(MagicMock(), ["A", "B"], full=frozenset({"B"}))
        timing, full = got["A"], got["B"]
        assert timing["level"] == "timing"
        assert full["level"] == "full"
        for field in ("close_time", "expiration_time", "latest_expiration_time",
                      "expected_expiration_time", "occurrence_datetime", "can_close_early",
                      "settlement_ts", "settlement_value_dollars", "result", "status",
                      "open_time", "event_ticker", "settlement_timer_seconds"):
            assert field in timing, field
        for field in ("volume_fp", "rules_primary", "custom_strike", "strike_type",
                      "previous_yes_ask_dollars", "notional_value_dollars"):
            assert field not in timing, field
            assert field in full, field
        assert full["volume_fp"] == "2075659.36"
        assert full["close_time"] == timing["close_time"]

    def test_a_timing_record_is_fetched_again_when_the_full_one_is_wanted(self, exchange):
        exchange.markets = {"A": _market("A")}
        market_details.load_market_details(MagicMock(), ["A"])
        assert len(exchange.requests) == 1
        stats = {}
        got = market_details.load_market_details(
            MagicMock(), ["A"], full=frozenset({"A"}), stats_out=stats)
        assert len(exchange.requests) == 2
        assert got["A"]["level"] == "full"
        assert stats["from_cache"] == 0
        # Now cached in full: neither level asks again
        again = market_details.load_market_details(MagicMock(), ["A"], full=frozenset({"A"}))
        market_details.load_market_details(MagicMock(), ["A"])
        assert len(exchange.requests) == 2
        assert again["A"] == got["A"]
        assert _cached_details()["A"]["level"] == "full"

    def test_a_full_cached_record_serves_a_timing_request(self, exchange):
        exchange.markets = {"A": _market("A")}
        market_details.load_market_details(MagicMock(), ["A"], full=frozenset({"A"}))
        got = market_details.load_market_details(MagicMock(), ["A"])
        assert len(exchange.requests) == 1
        assert got["A"]["level"] == "full"

    def test_only_settled_records_are_cached(self, exchange):
        exchange.markets = {
            "DONE": _market("DONE"),
            "OPEN": _market("OPEN", status="active", result="", settlement_ts=None),
            "DET": _market("DET", status="determined"),
            "SET": _market("SET", status="settled"),
        }
        got = market_details.load_market_details(MagicMock(), ["DONE", "OPEN", "DET", "SET"])
        assert sorted(got) == ["DET", "DONE", "OPEN", "SET"]
        # A determined market can still lack its settlement fields: not frozen
        assert sorted(_cached_details()) == ["DONE", "SET"]
        # The open market is asked about again, the settled ones are not
        exchange.requests.clear()
        market_details.load_market_details(MagicMock(), ["DONE", "OPEN", "DET"])
        [(_, query)] = exchange.paths("/markets")
        assert sorted(query["tickers"].split(",")) == ["DET", "OPEN"]

    def test_a_new_call_merges_into_the_cache_file(self, exchange):
        exchange.markets = {"A": _market("A"), "B": _market("B")}
        market_details.load_market_details(MagicMock(), ["A"])
        market_details.load_market_details(MagicMock(), ["B"])
        assert sorted(_cached_details()) == ["A", "B"]

    def test_with_no_client_only_the_cache_is_read(self, exchange):
        exchange.markets = {"A": _market("A")}
        market_details.load_market_details(MagicMock(), ["A"])
        exchange.requests.clear()
        stats = {}
        got = market_details.load_market_details(None, ["A", "B"], stats_out=stats)
        assert sorted(got) == ["A"]
        assert exchange.requests == []
        assert stats["from_cache"] == 1
        assert stats["not_found"] == 1

    def test_a_failed_batch_is_one_warning_and_the_rest_is_read(self, exchange, caplog,
                                                                monkeypatch):
        monkeypatch.setattr(config, "MARKET_DETAILS_BATCH_SIZE", 2)
        tickers = ["A", "B", "C", "D"]
        exchange.markets = {t: _market(t) for t in tickers}
        real = exchange.__call__

        def flaky(client, path, **params):
            if path.endswith("/markets") and params["tickers"] == "C,D":
                raise _HttpError(503)
            return real(client, path, **params)

        monkeypatch.setattr(market_details, "_get", flaky)
        stats = {}
        with caplog.at_level(logging.WARNING):
            got = market_details.load_market_details(MagicMock(), tickers, stats_out=stats)
        assert sorted(got) == ["A", "B"]
        assert stats["failed"] == 2
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "\n" not in warnings[0]
        assert "2 left out" in warnings[0]
        # Nothing was cached for the failed batch
        assert sorted(_cached_details()) == ["A", "B"]

    def test_a_field_of_the_wrong_type_is_left_out_and_counted(self, exchange):
        exchange.markets = {"A": _market("A", can_close_early="yes", volume_fp=2075659.36)}
        stats = {}
        got = market_details.load_market_details(MagicMock(), ["A"], full=frozenset({"A"}),
                                                 stats_out=stats)
        assert "can_close_early" not in got["A"]
        assert "volume_fp" not in got["A"]
        assert stats["bad_fields"] == 2

    def test_a_null_field_is_simply_absent(self, exchange):
        exchange.markets = {"A": _market("A", occurrence_datetime=None)}
        stats = {}
        got = market_details.load_market_details(MagicMock(), ["A"], stats_out=stats)
        assert "occurrence_datetime" not in got["A"]
        assert stats["bad_fields"] == 0

    def test_a_reply_with_no_markets_list_is_a_failed_batch(self, exchange, monkeypatch, caplog):
        monkeypatch.setattr(market_details, "_get", lambda *a, **k: {"oops": 1})
        with caplog.at_level(logging.WARNING):
            got = market_details.load_market_details(MagicMock(), ["A"])
        assert got == {}
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_the_session_stub_keeps_every_test_offline(self):
        # No FakeExchange here: the conftest stub is in place, so a MagicMock
        # client reads nothing and raises nothing for markets and trades
        assert market_details.load_market_details(MagicMock(), ["A", "B"]) == {}
        assert market_details.load_market_trades(
            MagicMock(), "A", settled_before_trades_cutoff=True) == ([], 0)
        with pytest.raises(RuntimeError):
            market_details._get(MagicMock(), f"{PREFIX}/historical/cutoff")

    def test_a_call_appends_a_member_and_never_rewrites_the_file(self, exchange):
        exchange.markets = {"A": _market("A"), "B": _market("B")}
        market_details.load_market_details(MagicMock(), ["A"])
        path = market_details._DETAILS_CACHE
        first = path.read_bytes()
        market_details.load_market_details(MagicMock(), ["B"])
        second = path.read_bytes()
        assert second.startswith(first)
        assert len(second) > len(first)
        assert [r["ticker"] for r in _details_lines()] == ["A", "B"]
        # Nothing new to add: the file is not touched
        market_details.load_market_details(MagicMock(), ["A", "B"])
        assert path.read_bytes() == second

    def test_the_latest_line_of_a_ticker_wins(self, exchange):
        exchange.markets = {"A": _market("A")}
        market_details.load_market_details(MagicMock(), ["A"])
        market_details.load_market_details(MagicMock(), ["A"], full=frozenset({"A"}))
        assert [r["level"] for r in _details_lines()] == ["timing", "full"]
        got = market_details.load_market_details(None, ["A"])
        assert got["A"]["level"] == "full"
        assert got["A"] == _cached_details()["A"]

    def test_only_the_requested_tickers_are_kept(self, exchange):
        exchange.markets = {t: _market(t) for t in "ABC"}
        market_details.load_market_details(MagicMock(), list("ABC"))
        assert sorted(market_details.load_market_details(None, ["B"])) == ["B"]

    def test_a_damaged_tail_keeps_the_records_before_it_and_warns_once(
            self, exchange, caplog):
        exchange.markets = {t: _market(t) for t in "ABC"}
        market_details.load_market_details(MagicMock(), ["A"])
        market_details.load_market_details(MagicMock(), ["B"])
        market_details.load_market_details(MagicMock(), ["C"])
        path = market_details._DETAILS_CACHE
        raw = path.read_bytes()
        path.write_bytes(raw[:-12])      # the last member is cut short
        with caplog.at_level(logging.WARNING):
            got = market_details.load_market_details(None, list("ABC"))
        assert sorted(got) == ["A", "B"]
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_a_bad_line_stops_the_read_there(self, exchange, caplog):
        good = market_details._record(_market("A"), False)[0] | {"ticker": "A"}
        later = market_details._record(_market("C"), False)[0] | {"ticker": "C"}
        market_details._DETAILS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(market_details._DETAILS_CACHE, "wb") as fh:
            fh.write(json.dumps(good).encode() + b"\nthis is not json\n"
                     + json.dumps(later).encode() + b"\n")
        with caplog.at_level(logging.WARNING):
            got = market_details.load_market_details(None, ["A", "C"])
        assert sorted(got) == ["A"]
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_appending_after_a_damaged_tail_keeps_the_good_prefix_reachable(self, exchange):
        exchange.markets = {t: _market(t) for t in "ABC"}
        market_details.load_market_details(MagicMock(), ["A"])
        market_details.load_market_details(MagicMock(), ["B"])
        path = market_details._DETAILS_CACHE
        path.write_bytes(path.read_bytes()[:-12])
        # B was lost with the tail; fetching C now must not hide A behind the damage
        market_details.load_market_details(MagicMock(), ["C"])
        got = market_details.load_market_details(None, list("ABC"))
        assert sorted(got) == ["A", "C"]
        # ... and B is read again when asked for
        assert sorted(market_details.load_market_details(MagicMock(), ["B"])) == ["B"]
        assert sorted(market_details.load_market_details(None, list("ABC"))) == list("ABC")


class TestLoadMarketTrades:
    """The trades walk: oldest first, both endpoints, two bounds, cached only when complete."""

    # The seven trades _serve makes carry 1 + 2 + ... + 7 contracts
    VOLUME = "28.00"

    def _serve(self, exchange, count=7, ticker="M"):
        exchange.trades[ticker] = [
            _trade_payload(ticker, f"id{i}", 1_786_000_000 + i * 60, i + 1) for i in range(count)]

    def test_trades_come_back_oldest_first_across_pages(self, exchange, monkeypatch):
        monkeypatch.setattr(config, "MARKET_DETAILS_PAGE_SIZE", 3)
        self._serve(exchange)
        stats = {}
        trades, duplicates = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=self.VOLUME,
            stats_out=stats)
        assert duplicates == 0
        assert [t[2] for t in trades] == [1, 2, 3, 4, 5, 6, 7]
        assert [t[0] for t in trades] == sorted(t[0] for t in trades)
        assert trades[0] == (1_786_000_000_000, 0.5, 1.0, "no")
        assert len(exchange.paths("historical/trades")) == 3
        assert stats["from_cache"] is False
        assert stats["bytes"] > 0

    def test_a_complete_walk_is_cached_and_read_without_a_request(self, exchange):
        self._serve(exchange)
        first, _ = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=self.VOLUME)
        assert (market_details._TRADES_DIR / "M.json.gz").exists()
        exchange.requests.clear()
        stats = {}
        second, _ = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=self.VOLUME,
            stats_out=stats)
        assert exchange.requests == []
        assert second == first
        assert stats["from_cache"] is True

    @pytest.mark.parametrize("volume", ["28.01", "27.99", "100.00", None, "lots"])
    def test_a_walk_that_does_not_add_up_to_the_volume_is_returned_but_not_cached(
            self, exchange, volume):
        self._serve(exchange)
        stats = {}
        got = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=volume,
            stats_out=stats)
        assert got is not None
        assert len(got[0]) == 7
        assert stats["cached"] is False
        assert not market_details._TRADES_DIR.exists()
        # Read again next time
        exchange.requests.clear()
        market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=volume)
        assert exchange.requests

    def test_a_volume_within_the_tolerance_is_cached(self, exchange):
        self._serve(exchange)
        market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp="28.004")
        assert (market_details._TRADES_DIR / "M.json.gz").exists()

    def test_with_no_client_only_the_cache_is_read(self, exchange):
        self._serve(exchange)
        assert market_details.load_market_trades(
            None, "M", settled_before_trades_cutoff=True) is None
        market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp=self.VOLUME)
        assert market_details.load_market_trades(
            None, "M", settled_before_trades_cutoff=True) is not None

    @pytest.mark.parametrize(("before", "first", "second"), [
        (True, "historical/trades", "markets/trades"),
        (False, "markets/trades", "historical/trades"),
    ])
    def test_the_endpoint_asked_first_follows_the_settlement(self, exchange, before, first,
                                                             second):
        self._serve(exchange)
        market_details.load_market_trades(MagicMock(), "M", settled_before_trades_cutoff=before)
        assert exchange.paths(first) and not exchange.paths(second)

    def test_a_404_asks_the_other_endpoint(self, exchange):
        self._serve(exchange)
        exchange.trade_status["historical/trades"] = 404
        trades, _ = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True)
        assert len(trades) == 7
        assert exchange.paths("markets/trades")

    def test_a_404_from_both_is_not_known_and_not_cached(self, exchange):
        self._serve(exchange)
        exchange.trade_status = {"historical/trades": 404, "markets/trades": 404}
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True) is None
        assert not market_details._TRADES_DIR.exists()

    def test_another_error_does_not_ask_the_other_endpoint(self, exchange):
        self._serve(exchange)
        exchange.trade_status["historical/trades"] = 500
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True) is None
        assert not exchange.paths("markets/trades")

    def test_a_walk_cut_short_by_an_error_caches_nothing(self, exchange, monkeypatch):
        monkeypatch.setattr(config, "MARKET_DETAILS_PAGE_SIZE", 3)
        self._serve(exchange)
        real = exchange.__call__
        calls = []

        def dies_on_page_two(client, path, **params):
            calls.append(path)
            if len(calls) == 2:
                raise _HttpError(500)
            return real(client, path, **params)

        monkeypatch.setattr(market_details, "_get", dies_on_page_two)
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True) is None
        assert not market_details._TRADES_DIR.exists()

    def test_a_cursor_seen_twice_stops_the_walk_and_caches_nothing(self, exchange, monkeypatch,
                                                                   caplog):
        page = {"trades": [_trade_payload("M", "x", 1_786_000_000, 1)], "cursor": "same"}
        monkeypatch.setattr(market_details, "_get", lambda *a, **k: page)
        stats = {}
        with caplog.at_level(logging.DEBUG):
            assert market_details.load_market_trades(
                MagicMock(), "M", settled_before_trades_cutoff=True, stats_out=stats) is None
        # The reason goes to the caller and to DEBUG, never to a per-market WARNING
        assert stats["failure"] == "a cursor repeated"
        assert [r.levelno for r in caplog.records
                if "cursor repeated" in r.getMessage()] == [logging.DEBUG]
        assert not market_details._TRADES_DIR.exists()

    def test_an_http_error_is_a_one_line_reason(self, exchange):
        exchange.trade_status["historical/trades"] = 500
        stats = {}
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, stats_out=stats) is None
        assert "\n" not in stats["failure"]
        assert stats["failure"]

    def test_the_page_cap_stops_the_walk_and_caches_nothing(self, exchange, monkeypatch):
        monkeypatch.setattr(config, "SCANNER_MAX_PAGES", 3)
        counter = iter(range(100))

        def endless(*a, **k):
            n = next(counter)
            return {"trades": [_trade_payload("M", f"id{n}", 1_786_000_000 + n, 1)],
                    "cursor": f"c{n}"}

        monkeypatch.setattr(market_details, "_get", endless)
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True) is None
        assert next(counter) == 3
        assert not market_details._TRADES_DIR.exists()

    def test_a_repeated_trade_id_is_dropped_and_counted(self, exchange):
        trades = [_trade_payload("M", "dup", 1_786_000_000, 2),
                  _trade_payload("M", "dup", 1_786_000_000, 2),
                  _trade_payload("M", "other", 1_786_000_100, 3)]
        exchange.trades["M"] = trades
        stats = {}
        got, duplicates = market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True, volume_fp="5.00",
            stats_out=stats)
        assert duplicates == 1
        assert [t[2] for t in got] == [2.0, 3.0]
        assert stats["duplicates"] == 1
        # The count survives the cache
        again = {}
        market_details.load_market_trades(MagicMock(), "M", settled_before_trades_cutoff=True,
                                          volume_fp="5.00", stats_out=again)
        assert again["duplicates"] == 1

    def test_unsound_prices_are_counted(self, exchange):
        exchange.trades["M"] = [
            _trade_payload("M", "a", 1_786_000_000, 1, yes="0.4000", no="0.5000"),
            _trade_payload("M", "b", 1_786_000_001, 1, yes="1.0000", no="0.0000"),
            _trade_payload("M", "c", 1_786_000_002, 1)]
        stats = {}
        market_details.load_market_trades(MagicMock(), "M", settled_before_trades_cutoff=True,
                                          volume_fp="3.00", stats_out=stats)
        assert stats["price_failures"] == 2
        assert stats["unsound_ms"] == [1_786_000_000_000, 1_786_000_001_000]
        # The times survive the cache
        again = {}
        market_details.load_market_trades(MagicMock(), "M", settled_before_trades_cutoff=True,
                                          volume_fp="3.00", stats_out=again)
        assert again["unsound_ms"] == stats["unsound_ms"]

    def test_a_trade_that_cannot_be_read_ends_the_walk_uncached(self, exchange):
        bad = _trade_payload("M", "a", 1_786_000_000, 1)
        bad["count_fp"] = "lots"
        exchange.trades["M"] = [bad]
        assert market_details.load_market_trades(
            MagicMock(), "M", settled_before_trades_cutoff=True) is None
        assert not market_details._TRADES_DIR.exists()

    def test_a_ticker_that_cannot_name_a_file_is_read_but_not_cached(self, exchange):
        exchange.trades["A/B"] = [_trade_payload("A/B", "a", 1_786_000_000, 1)]
        got = market_details.load_market_trades(
            MagicMock(), "A/B", settled_before_trades_cutoff=True)
        assert got is not None
        assert not market_details._TRADES_DIR.exists()

    def test_kalshi_time_strings_with_any_fraction_are_read(self):
        # 6, 5 and 0 digits of fraction, as the live payloads write them
        base = int(datetime(2026, 8, 16, 23, 15, 20, tzinfo=UTC).timestamp()) * 1000
        assert market_details._time_ms("2026-08-16T23:15:20.572093Z") == base + 572
        assert market_details._time_ms("2026-10-17T14:17:20.69747Z") is not None
        assert market_details._time_ms("2026-08-16T23:15:20Z") == base
        assert market_details._time_ms("not a time") is None
        assert market_details._time_ms(None) is None

    def test_the_real_trade_pages_parse(self, exchange):
        for name in ("trades_hist_page.json", "trades_live_page.json"):
            for raw in _load(name)["trades"]:
                parsed = market_details._parse_trade(raw)
                assert parsed is not None
                (when, yes, count, side), trade_id, sound = parsed
                assert sound
                assert 0 < yes < 1
                assert count > 0
                assert side in ("yes", "no")
                assert trade_id == raw["trade_id"]

    def test_the_trades_cutoff_is_read_as_seconds(self, exchange):
        assert market_details.read_trades_cutoff(MagicMock()) == int(
            datetime(2026, 8, 17, tzinfo=UTC).timestamp())

    def test_an_unreadable_cutoff_is_none(self, exchange):
        exchange.broken["/historical/cutoff"] = _HttpError(500)
        assert market_details.read_trades_cutoff(MagicMock()) is None
        exchange.broken.clear()
        exchange.cutoff = {"trades_created_ts": "soon"}
        assert market_details.read_trades_cutoff(MagicMock()) is None


class TestTimingEpochs:
    """The compact form of a record's close-related times."""

    def test_three_times_as_unix_seconds(self):
        record = {"close_time": "2026-08-16T23:19:39Z",
                  "latest_expiration_time": "2026-08-31T22:00:00.5Z",
                  "expected_expiration_time": "2026-08-17T01:00:00.123456+00:00"}
        base = int(datetime(2026, 8, 16, 23, 19, 39, tzinfo=UTC).timestamp())
        assert market_details.timing_epochs(record) == (
            base, int(datetime(2026, 8, 31, 22, tzinfo=UTC).timestamp()),
            int(datetime(2026, 8, 17, 1, tzinfo=UTC).timestamp()))
        assert base == 1_786_922_379

    def test_a_missing_or_unreadable_time_is_none(self):
        assert market_details.timing_epochs({"close_time": "soon"}) == (None, None, None)
        assert market_details.timing_epochs({}) == (None, None, None)
        assert market_details.timing_epochs(None) == (None, None, None)

    def test_project_replaces_each_record_and_leaves_the_cache_whole(self, exchange):
        exchange.markets = {"A": _market("A")}
        stats = {}
        got = market_details.load_market_details(
            MagicMock(), ["A"], full=frozenset({"A"}), stats_out=stats,
            project=market_details.timing_epochs)
        assert got["A"] == market_details.timing_epochs(_market("A"))
        assert stats["full"] == 1
        assert _cached_details()["A"]["level"] == "full"
        again = market_details.load_market_details(
            None, ["A"], full=frozenset({"A"}), stats_out=stats,
            project=market_details.timing_epochs)
        assert again == got
        assert stats["from_cache"] == 1
        assert stats["full"] == 1


class TestCheckMarket:
    """Every check fires on a crafted failure and none fires on the real MLB game."""

    @pytest.fixture
    def game(self):
        candles = _mlb_candles()
        details = market_details._record(_load("market_archive_mlb_game.json")["market"], True)[0]
        trades = _tuples(_mlb_trade_payloads(candles))
        return details, candles, trades

    def test_the_real_game_passes_every_check(self, game):
        details, candles, trades = game
        assert details["volume_fp"] == "2075659.36"
        assert market_details.check_market(details, candles, trades) == []
        assert market_details.check_candles(candles) == []

    def test_candle_volumes_that_miss_the_total_fail(self, game):
        details, candles, trades = game
        details = {**details, "volume_fp": "2075659.99"}
        failed = market_details.check_market(details, _contiguous(candles), None)
        assert [market_details.failing_check(s) for s in failed] == ["candle_volume"]
        # The saved series skips hours nothing traded in, so it is not compared
        assert market_details.check_market(details, candles, None) == []

    def test_the_candle_total_is_skipped_when_the_candles_miss_part_of_the_life(self, game):
        details, candles, trades = game
        details = {**details, "volume_fp": "5.00"}
        assert market_details.check_market(details, candles[5:], None) == []
        assert market_details.check_market(details, candles[:-5], None) == []
        # An hour whose volume is unknown also leaves the sum unchecked
        unknown = [dict(c) for c in candles]
        unknown[3]["volume"] = None
        assert market_details.check_market(details, unknown, None) == []

    def test_a_missing_hour_skips_the_whole_life_candle_check_instead_of_failing_it(self, game):
        details, candles, trades = game
        details = {**details, "volume_fp": "5.00"}
        whole = _contiguous(candles)
        gap = whole[:30] + whole[31:]
        assert market_details.check_market(details, gap, None) == []
        # Without the gap the same wrong total does fail
        assert [market_details.failing_check(s)
                for s in market_details.check_market(details, whole, None)] == ["candle_volume"]

    def test_unsound_prices_count_only_inside_the_markets_life(self, game):
        details, candles, trades = game
        inside = trades[10][0]
        long_after = trades[-1][0] + 10 * 86_400_000
        long_before = _time_ms_of(details["open_time"]) - 10 * 86_400_000
        failed = market_details.check_market(
            details, candles, trades, trade_stats={"unsound_ms": [inside, long_after]})
        assert [market_details.failing_check(s) for s in failed] == ["trade_price"]
        assert "1 trades in the market's life" in failed[0]
        assert market_details.check_market(
            details, candles, trades,
            trade_stats={"price_failures": 2, "unsound_ms": [long_after, long_before]}) == []

    def test_trades_that_miss_the_total_fail(self, game):
        details, candles, trades = game
        failed = market_details.check_market(details, candles, trades[:-1])
        names = {market_details.failing_check(s) for s in failed}
        assert "trade_volume" in names

    def test_a_trade_in_the_wrong_hour_fails_only_the_hourly_check(self, game):
        details, candles, trades = game
        moved = list(trades)
        when, price, count, side = moved[-1]
        moved[-1] = (when - 5_000_000, price, count, side)
        failed = market_details.check_market(details, candles, moved)
        assert {market_details.failing_check(s) for s in failed} == {"hourly_volume"}
        assert "hours differ" in failed[0]

    def test_a_trade_on_an_hour_boundary_belongs_to_the_hour_that_ends_there(self, game):
        details, candles, trades = game
        shifted = [(t[0], *t[1:]) for t in trades]
        # Move one trade exactly onto the end of its own hour
        for index, (when, price, count, side) in enumerate(shifted):
            end = -(-when // 3_600_000) * 3_600_000
            shifted[index] = (end, price, count, side)
        assert market_details.check_market(details, candles, shifted) == []

    def test_trades_outside_the_candles_hours_are_not_compared(self, game):
        details, candles, trades = game
        early = (candles[0]["ts"] * 1000 - 10_000_000, 0.5, 4.0, "yes")
        # They add to the total, so only the trade-volume check sees them
        failed = market_details.check_market(details, candles, [early, *trades])
        assert {market_details.failing_check(s) for s in failed} == {"trade_volume"}

    def test_a_repeated_trade_id_fails(self, game):
        details, candles, trades = game
        failed = market_details.check_market(details, candles, trades,
                                             trade_stats={"duplicates": 2})
        assert [market_details.failing_check(s) for s in failed] == ["trade_id"]

    def test_unsound_prices_fail(self, game):
        details, candles, trades = game
        failed = market_details.check_market(details, candles, trades,
                                             trade_stats={"price_failures": 3})
        assert [market_details.failing_check(s) for s in failed] == ["trade_price"]
        # A YES price outside (0, 1) inside the market's life fails on its own
        bad = [*trades, (trades[-1][0], 1.0, 0.0, "yes")]
        failed = market_details.check_market(details, candles, bad)
        assert [market_details.failing_check(s) for s in failed] == ["trade_price"]

    def test_a_bad_price_long_after_settlement_is_not_in_the_life(self, game):
        details, candles, trades = game
        late = (trades[-1][0] + 10 * 86_400_000, 1.0, 0.0, "yes")
        failed = market_details.check_market(details, candles, [*trades, late])
        assert "trade_price" not in {market_details.failing_check(s) for s in failed}

    def test_a_settlement_value_that_disagrees_with_the_result_fails(self, game):
        details, candles, trades = game
        failed = market_details.check_market(
            {**details, "settlement_value_dollars": "0.0000"}, candles, trades)
        assert [market_details.failing_check(s) for s in failed] == ["settlement_value"]
        # A result that is neither yes nor no is not compared
        assert market_details.check_market(
            {**details, "result": "scalar", "settlement_value_dollars": "0.4200"},
            candles, trades) == []

    def test_timing_fields_out_of_order_fail(self, game):
        details, candles, trades = game
        failed = market_details.check_market(
            {**details, "open_time": "2026-08-18T00:00:00Z"}, candles, trades)
        assert "timing_order" in {market_details.failing_check(s) for s in failed}
        failed = market_details.check_market(
            {**details, "latest_expiration_time": "2026-08-01T00:00:00Z"}, candles, trades)
        assert [market_details.failing_check(s) for s in failed] == ["timing_order"]

    def test_missing_fields_skip_their_checks(self, game):
        details, candles, trades = game
        assert market_details.check_market({}, [], None) == []
        assert market_details.check_market(None, candles, None) == []
        # No total and no timing: only the hour-by-hour comparison can run, and it holds
        assert market_details.check_market({"ticker": "X"}, candles, trades) == []

    def test_a_malformed_record_is_one_unreadable_sentence_not_a_crash(self, game):
        details, candles, trades = game
        failed = market_details.check_market(details, [7, 8], trades)
        assert [market_details.failing_check(s) for s in failed] == ["unreadable_record"]

    def test_every_check_name_is_listed(self):
        assert set(market_details.CHECK_NAMES) == {
            "candle_volume", "trade_volume", "hourly_volume", "trade_id", "trade_price",
            "candle_range", "settlement_value", "timing_order", "unreadable_record"}


class TestCheckCandles:
    """The order inside each hour, on either side of the book."""

    def _candle(self, **over):
        candle = {"ts": 1000, "yes_ask_close": 0.50, "yes_bid_close": 0.40,
                  "yes_ask_ohl": (0.60, 0.70, 0.50), "yes_bid_ohl": (0.40, 0.45, 0.35),
                  "price": (0.45, 0.55, 0.40, 0.50, 0.48, None), "volume": 3.0}
        candle.update(over)
        return candle

    def test_the_real_candles_hold(self):
        assert market_details.check_candles(_mlb_candles(), ticker=MLB) == []

    def test_a_clean_candle_holds(self):
        assert market_details.check_candles([self._candle()]) == []

    @pytest.mark.parametrize("over", [
        {"yes_ask_ohl": (0.60, 0.70, 0.65)},       # low above the open
        {"yes_ask_ohl": (0.60, 0.70, 0.55)},       # low above the close 0.50
        {"yes_ask_ohl": (0.60, 0.55, 0.50)},       # high below the open
        {"yes_ask_ohl": (0.60, 0.70, 0.50), "yes_ask_close": 0.80},   # close above the high
        {"yes_bid_ohl": (0.40, 0.45, 0.42)},       # bid low above the open
        {"yes_bid_ohl": (0.40, 0.45, 0.35), "yes_bid_close": 0.50},   # bid close above the high
        {"price": (0.45, 0.55, 0.60, 0.50, 0.48, None)},   # traded low above traded high
        {"price": (0.45, 0.55, 0.40, 0.80, 0.48, None)},   # traded close outside the range
        {"yes_bid_close": 0.55, "yes_bid_ohl": (0.55, 0.60, 0.50)},   # bid above the ask close
    ])
    def test_each_out_of_order_candle_fails(self, over):
        failed = market_details.check_candles([self._candle(**over)], ticker="T")
        assert failed
        assert all(s.startswith("candle_range: T") for s in failed)

    def test_a_candle_without_the_fields_is_not_checked(self):
        bare = {"ts": 1, "yes_ask_close": 0.5, "no_ask_close": 0.5, "volume": 2.0}
        assert market_details.check_candles([bare]) == []
        assert market_details.check_candles([self._candle(yes_ask_ohl=None, price=None)]) == []

    def test_junk_never_raises(self):
        junk = [None, 3, "x", {"yes_ask_ohl": "abc"}, {"yes_ask_ohl": (1, 2)},
                {"price": (1, 2, 3)}, {"yes_bid_close": object(), "yes_ask_close": 0.5}]
        assert market_details.check_candles(junk) == []


class TestAuditMarket:
    """One market's trades loaded, checked and reduced to a summary."""

    def test_a_consistent_market_has_a_summary_with_no_failures(self, exchange):
        candles = _mlb_candles()
        exchange.trades[MLB] = _mlb_trade_payloads(candles)
        details = market_details._record(_load("market_archive_mlb_game.json")["market"], True)[0]
        summary = market_details.audit_market(
            MagicMock(), MLB, details, candles, settled_before_trades_cutoff=True)
        assert summary["complete"] is True
        assert summary["trades"] == len(exchange.trades[MLB])
        assert summary["contracts"] == pytest.approx(2075659.36)
        assert summary["duplicates"] == 0
        assert summary["failures"] == []
        assert summary["from_cache"] is False
        assert summary["bytes"] > 0
        again = market_details.audit_market(
            MagicMock(), MLB, details, candles, settled_before_trades_cutoff=True)
        assert again["from_cache"] is True
        assert again["trades"] == summary["trades"]

    def test_failures_and_duplicates_are_carried(self, exchange):
        candles = _mlb_candles()
        payloads = _mlb_trade_payloads(candles)
        exchange.trades[MLB] = [*payloads, payloads[0]]
        details = market_details._record(_load("market_archive_mlb_game.json")["market"], True)[0]
        summary = market_details.audit_market(
            MagicMock(), MLB, {**details, "volume_fp": "1.00"}, _contiguous(candles),
            settled_before_trades_cutoff=True)
        names = {market_details.failing_check(s) for s in summary["failures"]}
        assert {"candle_volume", "trade_volume", "trade_id"} <= names
        assert summary["duplicates"] == 1

    def test_trades_that_could_not_be_read_make_an_incomplete_summary(self, exchange):
        exchange.trade_status = {"historical/trades": 500}
        summary = market_details.audit_market(
            MagicMock(), "M", None, [], settled_before_trades_cutoff=True)
        assert summary["complete"] is False
        assert summary["trades"] == 0
        assert summary["failures"] == []


class TestCaptureScheduledCloses:
    """The command that records live closes, in the scheduled-closes-v1 format."""

    def test_it_writes_every_open_market_in_the_v1_format(self, exchange):
        def open_market(ticker, event, close):
            return {"ticker": ticker, "event_ticker": event, "close_time": close}

        exchange.open_pages = [
            ([open_market("KXA-1-X", "KXA-1", "2026-10-20T00:00:00Z"),
              open_market("KXB-2-Y", "KXB-2", "2026-10-21T00:00:00Z")], "1"),
            ([open_market("KXC-3-Z", "KXC-3", "2026-10-22T00:00:00Z"),
              {"ticker": "NO-CLOSE", "event_ticker": "KXD-1"}], ""),
        ]
        path = market_details.capture_scheduled_closes(MagicMock())
        assert path.parent == market_details._CLOSES_DIR
        assert path.name.endswith(".json.gz")
        data = _read_gz(path)
        assert set(data) == {"meta", "observations"}
        assert set(data["meta"]) == {"source", "taken_at", "format"}
        assert data["meta"]["format"] == "scheduled-closes-v1"
        assert data["meta"]["taken_at"].endswith("Z")
        assert data["observations"] == [
            {"ticker": "KXA-1-X", "series": "KXA", "live_close_time": "2026-10-20T00:00:00Z",
             "record": None},
            {"ticker": "KXB-2-Y", "series": "KXB", "live_close_time": "2026-10-21T00:00:00Z",
             "record": None},
            {"ticker": "KXC-3-Z", "series": "KXC", "live_close_time": "2026-10-22T00:00:00Z",
             "record": None},
        ]
        first = exchange.paths("/markets")[0][1]
        assert first["status"] == "open"
        assert first["mve_filter"] == "exclude"
        assert first["limit"] == config.MARKET_DETAILS_PAGE_SIZE

    def test_a_repeated_cursor_saves_what_was_read(self, exchange, caplog):
        market = {"ticker": "KXA-1-X", "event_ticker": "KXA-1", "close_time": "2026-10-20T00:00:00Z"}
        exchange.open_pages = [([market], "1"), ([market], "1"), ([market], "1")]
        with caplog.at_level(logging.WARNING):
            path = market_details.capture_scheduled_closes(MagicMock())
        assert any("cursor repeated" in r.getMessage() for r in caplog.records)
        assert len(_read_gz(path)["observations"]) == 2

    def test_the_page_cap_stops_it(self, exchange, monkeypatch, caplog):
        monkeypatch.setattr(config, "SCANNER_MAX_PAGES", 2)
        market = {"ticker": "KXA-1-X", "event_ticker": "KXA-1", "close_time": "2026-10-20T00:00:00Z"}
        exchange.open_pages = [([market], "1"), ([market], "2"), ([market], "3")]
        with caplog.at_level(logging.WARNING):
            path = market_details.capture_scheduled_closes(MagicMock())
        assert any("stopped at 2 pages" in r.getMessage() for r in caplog.records)
        assert len(_read_gz(path)["observations"]) == 2

    def test_a_failed_request_writes_nothing(self, exchange):
        exchange.broken["/markets"] = _HttpError(500)
        with pytest.raises(_HttpError):
            market_details.capture_scheduled_closes(MagicMock())
        assert not market_details._CLOSES_DIR.exists()


SAMPLE_CLOSES = Path(__file__).parent / "fixtures" / "scheduled_closes_sample.json"


def _sample_observations():
    """Real capture observations (scheduled-closes-v1): 30 MLB games, 30 natural-gas dailies,
    24 Nobel Peace markets whose live closes no one field reproduces, 16 of another series."""
    return json.loads(SAMPLE_CLOSES.read_text())["observations"]


def _rows(observations):
    """The observations as learn_close_rules reads them."""
    return [r for r in (market_details._observation_times(o) for o in observations)
            if r is not None]


def _of(series, observations=None):
    """The sample's observations of one series."""
    return [o for o in (observations or _sample_observations()) if o["series"] == series]


def _without(observations, *fields):
    """A copy of the observations with the named fields taken off each record."""
    return [dict(o, record={k: v for k, v in o["record"].items() if k not in fields})
            for o in observations]


def _two_events(observations):
    """The observations with every other one moved to a second event of its series
    (its ticker's date part renamed), times unchanged."""
    out = []
    for i, o in enumerate(observations):
        event, strike = o["ticker"].rsplit("-", 1)
        out.append(dict(o, ticker=f"{event}{'B' if i % 2 else ''}-{strike}"))
    return out


def _rule(rules, series):
    """A rule's field and offset."""
    return rules[series][:2]


class TestCloseRules:
    """learn_close_rules and scheduled_close on real observations."""

    def test_a_game_series_learns_latest_expiration_time(self):
        rules = market_details.learn_close_rules(_rows(_of("KXMLBGAME")))
        assert set(rules) == {"KXMLBGAME"}
        field, offset, prints, events = rules["KXMLBGAME"]
        assert (field, offset) == ("latest_expiration_time", 0)
        # Every game's latest expiration sits 69 h after its expected one
        assert prints == frozenset({248_400}) and events == 15

    def test_close_time_first_then_latest_then_expected(self):
        # On these natural-gas dailies, spread over two events, all three
        # fields reproduce the live close: close_time (the rule that changes
        # nothing) is chosen
        gas = _two_events(_of("KXNATGASD"))
        rules = market_details.learn_close_rules(_rows(gas))
        assert _rule(rules, "KXNATGASD") == ("close_time", 0)
        # One market closed early: close_time no longer fits, latest does
        early = [dict(o, record=dict(o["record"])) for o in gas]
        early[3]["record"]["close_time"] = "2026-09-23T20:00:00Z"
        assert _rule(market_details.learn_close_rules(_rows(early)), "KXNATGASD") == (
            "latest_expiration_time", -608_400)
        # Without close_time and latest fitting, expected is the rule — but an
        # observation without latest cannot support it (no fingerprint)
        no_latest = _without(early, "latest_expiration_time")
        assert market_details.learn_close_rules(_rows(no_latest)) == {}

    def test_one_event_learns_nothing(self):
        # The sample's 30 dailies are one event: under the event floor
        gas = _of("KXNATGASD")
        assert len({market_details._event_of(o["ticker"]) for o in gas}) == 1
        assert config.SCHEDULED_CLOSE_MIN_EVENTS == 2
        assert market_details.learn_close_rules(_rows(gas)) == {}
        assert set(market_details.learn_close_rules(_rows(_two_events(gas)))) == {"KXNATGASD"}

    def test_a_series_no_field_reproduces_learns_nothing(self):
        nobel = _of("KXNOBELPEACE")
        assert len(nobel) >= config.SCHEDULED_CLOSE_MIN_OBSERVATIONS
        assert market_details.learn_close_rules(_rows(nobel)) == {}

    def test_one_miss_is_enough_to_learn_nothing(self):
        moved = [dict(o) for o in _of("KXMLBGAME")]
        moved[7]["live_close_time"] = "2026-09-28T18:35:00Z"
        assert market_details.learn_close_rules(_rows(moved)) == {}

    def test_too_few_observations_learn_nothing(self):
        few = _of("KXNEXTLABORSEC")
        assert len(few) < config.SCHEDULED_CLOSE_MIN_OBSERVATIONS
        assert market_details.learn_close_rules(_rows(few)) == {}
        games = _of("KXMLBGAME")[:config.SCHEDULED_CLOSE_MIN_OBSERVATIONS - 1]
        assert market_details.learn_close_rules(_rows(games)) == {}
        games = _of("KXMLBGAME")[:config.SCHEDULED_CLOSE_MIN_OBSERVATIONS]
        assert _rule(market_details.learn_close_rules(_rows(games)), "KXMLBGAME") == (
            "latest_expiration_time", 0)

    def test_observations_without_expected_do_not_count(self):
        games = _of("KXMLBGAME")[:config.SCHEDULED_CLOSE_MIN_OBSERVATIONS]
        # One of the 20 has no expected expiration: 19 support the rule, under the minimum
        stripped = [*_without(games[:1], "expected_expiration_time"), *games[1:]]
        assert market_details.learn_close_rules(_rows(stripped)) == {}

    def test_one_market_seen_twice_counts_once(self):
        games = _of("KXMLBGAME")[:config.SCHEDULED_CLOSE_MIN_OBSERVATIONS - 1]
        stats = {}
        assert market_details.learn_close_rules(_rows(games + games), stats_out=stats) == {}
        assert stats["observations"] == len(games)

    def test_the_whole_sample_and_its_stats(self):
        stats = {}
        rules = market_details.learn_close_rules(_rows(_sample_observations()),
                                                 stats_out=stats)
        assert {s: _rule(rules, s) for s in rules} == {
            "KXMLBGAME": ("latest_expiration_time", 0)}
        assert stats == {"observations": 100, "covered": 30,
                         "by_field": {"close_time": 0, "latest_expiration_time": 1,
                                      "expected_expiration_time": 0}}

    def test_a_game_that_closed_early_gets_its_scheduled_close(self):
        rules = market_details.learn_close_rules(_rows(_of("KXMLBGAME")))
        game = _of("KXMLBGAME")[16]
        timing = market_details.timing_epochs(game["record"])
        assert market_details.close_rule_applies(timing, rules, "KXMLBGAME")
        answer = market_details.scheduled_close(timing, rules, "KXMLBGAME")
        assert answer == market_details._epoch_seconds(game["live_close_time"])
        assert answer > timing[0]

    def test_a_market_of_another_convention_keeps_its_realized_close(self):
        # An older game whose latest expiration sits 14 days after its start
        # (Kalshi's earlier convention): the rule learned on 69 h would put its
        # close days too late, so it is not applied
        rules = market_details.learn_close_rules(_rows(_of("KXMLBGAME")))
        close, latest, expected = market_details.timing_epochs(_of("KXMLBGAME")[16]["record"])
        old = (close, expected + 14 * 86_400, expected)
        assert not market_details.close_rule_applies(old, rules, "KXMLBGAME")
        assert market_details.scheduled_close(old, rules, "KXMLBGAME") is None
        # No expected expiration: no fingerprint, no rule
        assert market_details.scheduled_close((close, latest, None), rules, "KXMLBGAME") is None

    def test_a_clock_change_needs_both_gaps_seen(self):
        # Two observations a clock change apart: latest − expected differs by an
        # hour. Seen both, the rule takes both; a new market with a third gap
        # keeps its realized close
        base = datetime(2026, 10, 1, 20, tzinfo=UTC)

        def obs(i, gap):
            expected = base + timedelta(days=i)
            latest = expected + timedelta(seconds=gap)
            return (f"KXA-{i // 2}-{i}", "KXA", int(latest.timestamp()),
                    (int(expected.timestamp()) - 3_600, int(latest.timestamp()),
                     int(expected.timestamp())))

        rows = [obs(i, 604_800 if i < 12 else 608_400) for i in range(24)]
        rules = market_details.learn_close_rules(rows)
        assert rules["KXA"][:2] == ("latest_expiration_time", 0)
        assert rules["KXA"][2] == frozenset({604_800, 608_400})
        for gap in (604_800, 608_400):
            timing = obs(30, gap)[3]
            assert market_details.scheduled_close(timing, rules, "KXA") == timing[1]
        assert market_details.scheduled_close(obs(30, 612_000)[3], rules, "KXA") is None
        # Learned on one side of the change only, the other side keeps its
        # realized close
        one_side = market_details.learn_close_rules([obs(i, 604_800) for i in range(24)])
        assert market_details.scheduled_close(obs(30, 608_400)[3], one_side, "KXA") is None

    def test_an_answer_before_the_realized_close_is_not_used(self):
        rules = {"KXA": ("latest_expiration_time", -7200, frozenset({3600, 7200, 9000}), 2)}
        assert market_details.scheduled_close((1_000_000, 1_003_600, 1_000_000), rules,
                                              "KXA") is None
        assert market_details.scheduled_close((1_000_000, 1_007_200, 1_000_000), rules,
                                              "KXA") == 1_000_000
        assert market_details.scheduled_close((1_000_000, 1_009_000, 1_000_000), rules,
                                              "KXA") == 1_001_800

    def test_no_rule_no_field_or_no_realized_close_is_none(self):
        rules = {"KXA": ("expected_expiration_time", 0, frozenset({-1}), 2)}
        assert market_details.scheduled_close((1, 2, 3), rules, "KXB") is None
        assert market_details.scheduled_close((1, 2, None), rules, "KXA") is None
        assert market_details.scheduled_close((None, 2, 3), rules, "KXA") is None
        assert market_details.scheduled_close(None, rules, "KXA") is None
        assert market_details.scheduled_close((1, 2, 3), rules, "KXA") == 3


class TestLoadCloseRules:
    """load_close_rules reads every capture, looking up observations with no record."""

    def _write(self, name, observations, fmt=market_details.CLOSES_FORMAT):
        market_details._CLOSES_DIR.mkdir(parents=True, exist_ok=True)
        historical._save_gz_json_cache(market_details._CLOSES_DIR / name, {
            "meta": {"source": "test", "taken_at": "2026-09-22", "format": fmt},
            "observations": observations})

    def test_no_capture_learns_nothing(self, caplog):
        with caplog.at_level(logging.INFO):
            assert market_details.load_close_rules(None) == {}
        assert ("Scheduled close: rules for 0 series from 0 observations "
                "(close_time 0, latest_expiration_time 0, expected_expiration_time 0)"
                in caplog.text)

    def test_it_reads_every_capture_and_logs_the_rules(self, caplog):
        sample = _sample_observations()
        self._write("2026-09-22.json.gz", sample[:50])
        self._write("2026-09-23.json.gz", sample[40:])
        self._write("bad.json.gz", sample, fmt="other")
        with caplog.at_level(logging.INFO):
            rules = market_details.load_close_rules(None)
        assert {s: _rule(rules, s) for s in rules} == {
            "KXMLBGAME": ("latest_expiration_time", 0)}
        assert ("Scheduled close: rules for 1 series from 100 observations "
                "(close_time 0, latest_expiration_time 1, expected_expiration_time 0)"
                in caplog.text)
        assert "bad.json.gz is not in the scheduled-closes-v1 format" in caplog.text

    def test_a_null_record_is_looked_up_settled_only(self, exchange):
        games = _of("KXMLBGAME")[:config.SCHEDULED_CLOSE_MIN_OBSERVATIONS]
        open_one = dict(games[0]["record"], ticker=games[0]["ticker"], status="active")
        for o in games:
            exchange.markets[o["ticker"]] = dict(o["record"], ticker=o["ticker"])
        exchange.markets[games[0]["ticker"]] = open_one
        self._write("x.json.gz", [dict(o, record=None) for o in games])
        # One market still open: 19 settled observations, under the minimum
        assert market_details.load_close_rules(MagicMock()) == {}
        exchange.markets[games[0]["ticker"]] = dict(games[0]["record"],
                                                     ticker=games[0]["ticker"])
        assert _rule(market_details.load_close_rules(MagicMock()), "KXMLBGAME") == (
            "latest_expiration_time", 0)


class TestCommands:
    """The two commands: capture-closes and check."""

    def test_capture_closes_prints_the_path_and_the_count(self, exchange, monkeypatch, capsys):
        market = {"ticker": "KXA-1-X", "event_ticker": "KXA-1", "close_time": "2026-10-20T00:00:00Z"}
        exchange.open_pages = [([market], "")]
        monkeypatch.setattr(historical, "build_prod_live_client", lambda: MagicMock())
        assert market_details.main(["capture-closes"]) == 0
        out = capsys.readouterr().out
        assert "Saved 1 live closes to" in out
        assert str(market_details._CLOSES_DIR) in out

    def test_capture_closes_failing_exits_1(self, exchange, monkeypatch):
        exchange.broken["/markets"] = _HttpError(500)
        monkeypatch.setattr(historical, "build_prod_live_client", lambda: MagicMock())
        assert market_details.main(["capture-closes"]) == 1

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            market_details.main([])

    @pytest.fixture
    def caches(self, exchange, tmp_path, monkeypatch):
        """Real caches written the way the loaders write them, plus a candle folder."""
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        candles = _mlb_candles()
        exchange.markets = {MLB: _market(MLB)}
        exchange.trades[MLB] = _mlb_trade_payloads(candles)
        market_details.load_market_details(MagicMock(), [MLB], full=frozenset({MLB}))
        market_details.load_market_trades(MagicMock(), MLB, settled_before_trades_cutoff=True,
                                          volume_fp="2075659.36")
        historical._save_gz_json_cache(
            historical._CANDLES_DIR / f"{MLB}.json.gz", {"candles": candles})
        return candles

    def test_check_passes_on_consistent_caches(self, caches, capsys):
        assert market_details.main(["check"]) == 0
        out = capsys.readouterr().out
        assert "All checks pass." in out
        assert "1 trades file(s)" in out
        assert "1 candle file(s)" in out

    def test_check_passes_when_there_is_no_cache(self, capsys):
        assert market_details.main(["check"]) == 0
        assert "nothing" in capsys.readouterr().out

    def test_check_fails_on_a_details_record_of_the_wrong_type(self, caches, capsys):
        data = _cached_details()
        data[MLB]["volume_fp"] = 2075659.36
        _write_details(data.values())
        assert market_details.main(["check"]) == 1
        assert "details_record" in capsys.readouterr().out

    def test_check_fails_on_a_damaged_details_file(self, caches, capsys):
        path = market_details._DETAILS_CACHE
        path.write_bytes(path.read_bytes()[:-8])
        assert market_details.main(["check"]) == 1
        assert "details_file" in capsys.readouterr().out

    def test_check_fails_on_an_unsettled_record_in_the_cache(self, caches, capsys):
        data = _cached_details()
        data[MLB]["status"] = "active"
        _write_details(data.values())
        assert market_details.main(["check"]) == 1

    def test_check_fails_when_trades_miss_the_market_total(self, caches, capsys):
        path = market_details._TRADES_DIR / f"{MLB}.json.gz"
        data = _read_gz(path)
        data["trades"].pop()
        historical._save_gz_json_cache(path, data)
        assert market_details.main(["check"]) == 1
        assert "trade_volume" in capsys.readouterr().out

    def test_check_fails_on_a_damaged_trades_file(self, caches, capsys):
        path = market_details._TRADES_DIR / f"{MLB}.json.gz"
        path.write_bytes(b"not gzip")
        assert market_details.main(["check"]) == 1
        assert "trades_file" in capsys.readouterr().out

    def test_check_fails_on_an_out_of_order_candle(self, caches, capsys):
        path = historical._CANDLES_DIR / f"{MLB}.json.gz"
        data = _read_gz(path)
        data["candles"][5]["yes_ask_ohl"] = [0.6, 0.7, 0.9]
        historical._save_gz_json_cache(path, data)
        assert market_details.main(["check"]) == 1
        assert "candle_range" in capsys.readouterr().out

    def test_check_shows_at_most_ten_examples_of_a_check(self, caches, capsys):
        path = historical._CANDLES_DIR / f"{MLB}.json.gz"
        data = _read_gz(path)
        for candle in data["candles"]:
            candle["yes_ask_ohl"] = [0.6, 0.7, 0.9]
        historical._save_gz_json_cache(path, data)
        assert market_details.main(["check"]) == 1
        out = capsys.readouterr().out
        assert out.count("    candle_range:") == config.INTEGRITY_MAX_EXAMPLES
        assert "candle_range: " in out


class TestBacktesterUse:
    """The backtest calls the module, logs what it found, and decides nothing with it."""

    def test_the_candle_worker_reports_failures_and_still_returns_the_same_candles(
            self, monkeypatch):
        candles = _mlb_candles()
        bad = [dict(c) for c in candles]
        bad[4]["yes_ask_ohl"] = (0.6, 0.7, 0.9)
        monkeypatch.setattr(backtester, "fetch_candlesticks", lambda *a, **k: bad)
        out: dict = {}
        with_check = backtester._fetch_projected_candles(
            MagicMock(), "T", 0, 1, True, check_out=out)
        without = backtester._fetch_projected_candles(MagicMock(), "T", 0, 1, True)
        assert with_check == without
        assert set(out) == {"T"}
        count, examples = out["T"]
        assert count >= 1
        assert len(examples) <= config.INTEGRITY_MAX_EXAMPLES
        assert examples[0].startswith("candle_range: T")

    def test_a_clean_series_leaves_the_check_out_empty(self, monkeypatch):
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda *a, **k: _mlb_candles())
        out: dict = {}
        backtester._fetch_projected_candles(MagicMock(), "T", 0, 1, True, check_out=out)
        assert out == {}

    def test_the_failures_are_capped_per_ticker(self, monkeypatch):
        bad = [dict(c, yes_ask_ohl=(0.6, 0.7, 0.9)) for c in _mlb_candles()]
        monkeypatch.setattr(backtester, "fetch_candlesticks", lambda *a, **k: bad)
        out: dict = {}
        backtester._fetch_projected_candles(MagicMock(), "T", 0, 1, True, check_out=out)
        count, examples = out["T"]
        assert count > config.INTEGRITY_MAX_EXAMPLES
        assert len(examples) == config.INTEGRITY_MAX_EXAMPLES

    def test_the_pair_markets_are_audited_into_small_summaries(self, exchange):
        candles = _mlb_candles()
        exchange.trades[MLB] = _mlb_trade_payloads(candles)
        details = market_details._record(_load("market_archive_mlb_game.json")["market"], True)[0]
        needed = {MLB: {"ticker": MLB, "settlement_ts": "2026-08-16T23:22:42Z"},
                  "NOPE": {"ticker": "NOPE"}}
        cutoff = int(datetime(2026, 8, 17, tzinfo=UTC).timestamp())
        summaries = backtester._audit_pair_markets(
            MagicMock(), needed, {MLB: details}, {MLB: candles}, cutoff)
        assert summaries[MLB]["complete"] is True
        assert summaries[MLB]["failures"] == []
        # The archive endpoint was asked first for a market settled before the cutoff
        assert exchange.paths("historical/trades")
        # A market with no trades at all is a complete walk of nothing
        assert summaries["NOPE"]["complete"] is True
        assert summaries["NOPE"]["trades"] == 0
        # Only counts are kept, never the trades themselves
        assert all(set(s) == {"trades", "contracts", "duplicates", "price_failures",
                              "complete", "from_cache", "bytes", "failure", "failures"}
                   for s in summaries.values())

    def test_a_market_settled_after_the_cutoff_asks_the_live_endpoint_first(self, exchange):
        exchange.trades["L"] = [_trade_payload("L", "a", 1_787_100_000, 1)]
        needed = {"L": {"ticker": "L", "settlement_ts": "2026-09-01T00:00:00Z"}}
        cutoff = int(datetime(2026, 8, 17, tzinfo=UTC).timestamp())
        backtester._audit_pair_markets(MagicMock(), needed, {}, {}, cutoff)
        paths = [p for p, _ in exchange.requests]
        assert paths[0].endswith("markets/trades")

    def test_an_unknown_cutoff_asks_the_archive_first(self, exchange):
        exchange.trades["L"] = [_trade_payload("L", "a", 1_787_100_000, 1)]
        needed = {"L": {"ticker": "L", "settlement_ts": "2026-09-01T00:00:00Z"}}
        backtester._audit_pair_markets(MagicMock(), needed, {}, {}, None)
        assert [p for p, _ in exchange.requests][0].endswith("historical/trades")

    def test_a_worker_that_fails_costs_its_market_a_summary_and_one_warning(
            self, monkeypatch, caplog):
        def boom(*a, **k):
            raise ValueError("odd")

        monkeypatch.setattr(market_details, "audit_market", boom)
        with caplog.at_level(logging.WARNING):
            summaries = backtester._audit_pair_markets(
                MagicMock(), {"T": {"ticker": "T"}}, {}, {}, None)
        assert summaries["T"]["complete"] is False
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_the_log_lines_always_show_every_check(self, caplog):
        summaries = {
            "A": {"trades": 10, "contracts": 25.0, "duplicates": 0, "price_failures": 0,
                  "complete": True, "from_cache": False, "bytes": 1_500_000, "failure": None,
                  "failures": []},
            "B": {"trades": 0, "contracts": 0.0, "duplicates": 0, "price_failures": 0,
                  "complete": False, "from_cache": False, "bytes": 0,
                  "failure": "HTTP 500 Server Error",
                  "failures": ["trade_volume: B: the trades differ", "trade_volume: B: again"]},
        }
        with caplog.at_level(logging.INFO):
            backtester._log_extra_data(
                {"full": 2, "from_cache": 1}, 2, summaries,
                {"A": (3, ["candle_range: A: hour ending 1: low above open"])})
        lines = [r.getMessage() for r in caplog.records]
        assert "Market details: full records for 2 of 2 pair markets (1 from the cache)" in lines
        # One WARNING for every market whose trades were not read to their end
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Trades: 1 of 2 pair markets could not be read to their end (first: B: HTTP 500 "
            "Server Error); they are not cached and are asked again next run"]
        assert ("Trades: 1 of 2 pair markets read to their end (10 trades, 25 contracts; "
                "1.5 MB cached)") in lines
        for name in market_details.CHECK_NAMES:
            assert any(line.startswith(f"Integrity check {name} over the pair markets: ")
                       for line in lines), name
        assert "Integrity check trade_volume over the pair markets: 1 failing" in lines
        assert "Integrity check candle_range over the pair markets: 1 failing" in lines
        assert "Integrity check trade_id over the pair markets: 0 failing" in lines
        assert "  trade_volume: B: the trades differ" in lines
        assert "  candle_range: A: hour ending 1: low above open" in lines

    def test_prepare_candidates_loads_reports_and_carries_the_side_data(
            self, exchange, monkeypatch, caplog):
        da = {"ticker": "DA", "event_ticker": "EA", "event_title": "EV", "title": "Q",
              "subtitle": "", "result": "yes", "close_time": "2026-02-01T00:00:00+00:00",
              "settlement_ts": "2026-02-01T12:00:00+00:00"}
        db = {**da, "ticker": "DB", "event_ticker": "EB"}
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: [dict(da), dict(db)])
        full = _mlb_candles()
        monkeypatch.setattr(backtester, "fetch_candlesticks", lambda *a, **k: full)
        exchange.markets = {t: _market(t, volume_fp="0.00", status="finalized")
                            for t in ("DA", "DB")}
        exchange.trades = {t: [] for t in ("DA", "DB")}
        with caplog.at_level(logging.INFO):
            candidates = backtester._prepare_candidates(
                MagicMock(), MagicMock(), date(2026, 1, 1), True, None)
        assert candidates is not None
        # Only the pair markets are loaded (a groupable market that forms no pair
        # costs nothing), each with its full record
        assert sorted(candidates.details) == ["DA", "DB"]
        assert all(d["level"] == "full" for d in candidates.details.values())
        assert sorted(candidates.trade_summaries) == ["DA", "DB"]
        assert candidates.trade_summaries["DA"]["complete"] is True
        assert candidates.candle_failures == {}
        lines = [r.getMessage() for r in caplog.records]
        assert "Market details: full records for 2 of 2 pair markets (0 from the cache)" in lines
        assert not [line for line in lines if "groupable markets (" in line]
        assert any(line.startswith("Peak RSS after the side-data loads") for line in lines)
        assert any(line.startswith("Trades: 2 of 2 pair markets read to their end")
                   for line in lines)
        for name in market_details.CHECK_NAMES:
            assert any(line.startswith(f"Integrity check {name} over the pair markets: ")
                       for line in lines), name
        # Candles are exactly what the fetch returned, cut to the six kept keys
        assert candidates.candles_by_ticker["DA"] == backtester._project_candles(full)


class TestIsolation:
    """The module is read-only and backtest-only."""

    TREE = ast.parse(inspect.getsource(market_details))

    def test_it_imports_no_order_path_or_report_module(self):
        imported = set()
        for node in ast.walk(self.TREE):
            if isinstance(node, ast.ImportFrom):
                imported |= {(node.module or "").split(".")[-1]}
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name.split(".")[-1] for a in node.names}
        for name in ("trader", "main", "scheduler", "v2_probe", "defaults_server", "scanner",
                     "backtester", "backtest", "dashboard", "strategy", "reporter", "seller",
                     "depth_model", "live_portfolio", "live_dashboard"):
            assert name not in imported, name

    def test_no_live_module_imports_it(self):
        names = [m.name for m in pkgutil.iter_modules(kalshi_betting.__path__)]
        assert "market_details" in names
        for name in names:
            if name in {"market_details", "backtester"}:
                continue
            module = importlib.import_module(f"kalshi_betting.{name}")
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.Import):
                    imported = {a.name.split(".")[-1] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported = {(node.module or "").split(".")[-1]}
                    imported |= {a.name for a in node.names}
                else:
                    continue
                assert "market_details" not in imported, f"kalshi_betting/{name}.py imports it"

    def test_it_sends_no_order(self):
        text = inspect.getsource(market_details)
        for forbidden in ("create_order", "submit_order", "V2_ORDER_PATH", "signed_request_json",
                          "_submit_", "ORDER_API_VERSION", "POST"):
            assert forbidden not in text

    def test_its_only_way_to_kalshi_is_one_read_only_get(self):
        sites = [n for n in ast.walk(self.TREE)
                 if isinstance(n, ast.Attribute) and n.attr in ("_historical_get", "_signed_raw_get",
                                                                "rest_client", "request")]
        assert len(sites) == 1
        getter = next(n for n in self.TREE.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_get")
        assert sites[0] in list(ast.walk(getter))
        assert sites[0].attr == "_historical_get"
        # Every request goes through _get, looked up by name when called
        calls = [n for n in ast.walk(self.TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_get"]
        assert len(calls) == 4

    def test_the_backtester_never_reads_the_carried_fields_off_an_object(self):
        # The fields are written once, by _prepare_candidates, as keyword
        # arguments of _Candidates; no function reads them back off a
        # _Candidates, so no decision can depend on them
        tree = ast.parse(inspect.getsource(backtester))
        reads = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and n.attr in ("details", "trade_summaries", "candle_failures")]
        assert reads == []
