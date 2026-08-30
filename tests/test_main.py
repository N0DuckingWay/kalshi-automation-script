"""
File: test_main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline tests for kalshi_betting.main — the live-pipeline orchestrator's
    pure helpers (_truncate, _format_deadline, _dedup_pairs,
    _compute_trade_specs, _no_pairs_msg), its logging setup (_setup_logging),
    and end-to-end "live-shape replay" runs of _run_dev/_run_prod against a
    MagicMock client wired with CURRENT-generation (2026-08+) Kalshi payload
    shapes — dollar-string prices, yes_sub_title, orderbook_fp books,
    shard-aware balance_breakdown, position_fp positions, V2 order responses —
    the closest offline substitute for a live smoke test.

Dependencies:
    Imports _run_dev/_run_prod and the pure helpers from kalshi_betting.main,
    plus config constants asserted against. All Kalshi API interaction is
    mocked at the HTTP boundary (raw-response mocks and rest_client.request);
    reporter Excel writers are patched out so no files are written.

Notes:
    The V2 live-execution replays run with dry_run=False against the config
    default ORDER_API_VERSION="v2" — deliberately not monkeypatched, so they
    prove the default path. Order responses are generated from each request's
    own submitted count, so the tests don't depend on exact Kelly sizing.
"""
import json
import logging
import pathlib
import re
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import main
from kalshi_betting import scanner as scanner_mod
from kalshi_betting.config import (
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    ORDER_API_VERSION,
    SAME_TITLE_MIN_PRICE_DIFF,
    V2_ORDER_PATH,
)


def make_pair(ticker_a: str, ticker_b: str, pair_type: str = "time_series"):
    """Minimal stand-in for a CandidatePair — only the attributes
    _dedup_pairs actually reads (market_a.ticker, market_b.ticker)."""
    return SimpleNamespace(
        market_a=SimpleNamespace(ticker=ticker_a),
        market_b=SimpleNamespace(ticker=ticker_b),
        pair_type=pair_type,
    )


class TestTruncate:
    def test_truncate_boundary(self):
        # Exactly n chars: no truncation, no ellipsis appended
        text_at_boundary = "x" * 40
        assert main._truncate(text_at_boundary) == text_at_boundary

        # One char over: truncated to n chars plus the ellipsis marker
        text_over_boundary = "x" * 41
        result = main._truncate(text_over_boundary)
        assert result == "x" * 40 + "…"
        assert len(result) == 41


class TestFormatDeadline:
    def test_format_deadline_none(self):
        assert main._format_deadline(None) == "?"


class TestDedupPairs:
    def test_dedup_pairs_prefers_same_title(self):
        # Same ticker pair detected by both scanners — the same-title (primary)
        # entry must win, and the time-series (secondary) duplicate must be dropped.
        same_title = make_pair("TICK-A", "TICK-B", pair_type="same_title")
        time_series_dup = make_pair("TICK-A", "TICK-B", pair_type="time_series")

        result = main._dedup_pairs([same_title], [time_series_dup])

        assert result == [same_title]

    def test_dedup_pairs_preserves_order_and_appends_unique_secondary(self):
        primary_1 = make_pair("A1", "A2", pair_type="same_title")
        primary_2 = make_pair("B1", "B2", pair_type="same_title")
        # Duplicate of primary_1's ticker pair — must be dropped, order-independent of tickers
        secondary_dup = make_pair("A2", "A1", pair_type="time_series")
        # Genuinely unique pair — must be appended after all primary entries
        secondary_unique = make_pair("C1", "C2", pair_type="time_series")

        result = main._dedup_pairs(
            [primary_1, primary_2], [secondary_dup, secondary_unique],
        )

        assert result == [primary_1, primary_2, secondary_unique]


class TestComputeTradeSpecs:
    def test_compute_trade_specs_excludes_none(self, monkeypatch):
        pair_ok = make_pair("A1", "A2")
        pair_none = make_pair("B1", "B2")

        def fake_compute_trade(pair, balance_cents):
            # Only pair_ok produces a spec — pair_none has no edge (returns None)
            if pair is pair_ok:
                return SimpleNamespace(pair=pair)
            return None

        monkeypatch.setattr(main, "compute_trade", fake_compute_trade)

        specs = main._compute_trade_specs([pair_ok, pair_none], balance_cents=100_000)

        assert list(specs.keys()) == [id(pair_ok)]
        assert specs[id(pair_ok)].pair is pair_ok


class TestNoPairsMsg:
    def test_no_pairs_log_lines_format_thresholds_from_config(self):
        prod_msg = main._no_pairs_msg()
        dev_msg = main._no_pairs_msg(sandbox=True)

        for msg in (prod_msg, dev_msg):
            assert f"{MIN_PRICE_DIFF_SHORT_GAP:.0%}" in msg
            assert f"{MIN_PRICE_DIFF_LONG_GAP:.0%}" in msg
            assert f"{SAME_TITLE_MIN_PRICE_DIFF:.0%}" in msg

        assert "sandbox" in dev_msg
        assert "sandbox" not in prod_msg


class TestSetupLogging:
    def test_setup_logging_has_console_and_file_handlers(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            log_path = tmp_path / "x.log"
            main._setup_logging(log_path)

            handler_types = [type(h) for h in root.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.FileHandler in handler_types
            # Exactly one of each — basicConfig should not have added extras
            assert sum(1 for h in root.handlers if type(h) is logging.StreamHandler) == 1
            assert sum(1 for h in root.handlers if isinstance(h, logging.FileHandler)) == 1

            # delay=True: the file must not be created until a record is emitted
            assert not log_path.exists()
            logging.getLogger("test_setup_logging").info("trigger the delayed file open")
            assert log_path.exists()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level


# ═══════════════════════════════════════════════════════════════════════════
# Live-shape replay suite — end-to-end _run_dev / _run_prod runs against a
# MagicMock client wired with current-generation (2026-08+) Kalshi payload
# shapes only: yes_sub_title (no "subtitle" key), *_dollars price strings,
# balance_breakdown shard entries, position_fp counts, and orderbook_fp
# dollar-string book levels. This exercises the real scanner ingest/pairing,
# real strategy sizing, and real trader execution (mocked at the HTTP
# boundary only) — the closest offline substitute for a live smoke test.
#
# Fixed 4-group market set built by _live_shape_client():
#   Recurring Q / Will X happen?  -> SAME-EXP (0.50) / SAME-CHEAP (0.20)
#       the one tradeable same-title pair every test exercises.
#   Tick Event / Tick Test Market -> TICK-A (0.90, sub-cent tick fields) /
#       TICK-B (0.50) — priced to be non-tradeable (nA+pB > 1), so it always
#       appears in scan output but never in a portfolio.
#   Shard Event / Shard Test Market -> SHARD1-A (exchange_index=1, dropped
#       at ingest) / SHARD1-B (left without a partner, so no pair forms).
#   Held Event / Held Question -> HELD-A (held in prod) / HELD-B — forms a
#       tradeable pair in dev (no held-ticker filtering) but never in prod.
# ═══════════════════════════════════════════════════════════════════════════

_CLOSE = "2026-12-01T00:00:00Z"

_TICKER_SAME_EXP = "SAME-EXP"
_TICKER_SAME_CHEAP = "SAME-CHEAP"
_TICKER_TICK_A = "TICK-A"
_TICKER_TICK_B = "TICK-B"
_TICKER_SHARD_A = "SHARD1-A"
_TICKER_SHARD_B = "SHARD1-B"
_TICKER_HELD_A = "HELD-A"
_TICKER_HELD_B = "HELD-B"

_TICK_PRICE_RANGES = [
    {"start": "0", "end": "0.01", "step": "0.0001"},
    {"start": "0.01", "end": "0.99", "step": "0.001"},
    {"start": "0.99", "end": "1", "step": "0.0001"},
]

# Live-shape balance payload (2026-08-14 shard scoping): shard 0 holds
# $250.00 (25000 cents) — the only shard our order path can reach — while
# shard 1 and the cross-shard aggregate both carry much larger, WRONG values
# that Kelly sizing must never see.
_LIVE_BALANCE_PAYLOAD = {
    "balance": 114,
    "balance_dollars": "10250.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
    ],
}

# Same shape, shard-0 entry below MIN_BALANCE_CENTS ($50) so _run_prod aborts.
_LOW_BALANCE_PAYLOAD = {
    "balance": 1,
    "balance_dollars": "100.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
    ],
}


def _raw_json_response(payload: dict, status: int = 200, reason: str = "OK") -> SimpleNamespace:
    """Raw-response stand-in matching every *_without_preload_content call and
    a hand-built rest_client.request call. reason/getheaders are what
    ApiException.from_response reads off a real RESTResponse on the non-2xx
    path (see test_http.py's TestSignedRequestJson._response)."""
    return SimpleNamespace(
        status=status,
        data=json.dumps(payload).encode("utf-8"),
        reason=reason,
        getheaders=lambda: {},
    )


def _mk_market(
    ticker: str,
    event_ticker: str,
    title: str,
    sub: str,
    yes_ask: str,
    no_ask: str,
    *,
    exchange_index: int = 0,
    price_level_structure: str = "",
    price_ranges: list | None = None,
) -> dict:
    """One raw market JSON dict in the current (2026-08+) wire shape: no
    "subtitle" key at all (only yes_sub_title), *_dollars price strings, and
    an explicit exchange_index."""
    d = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "yes_sub_title": sub,
        "status": "active",
        "close_time": _CLOSE,
        "yes_ask_dollars": yes_ask,
        "no_ask_dollars": no_ask,
        "yes_bid_dollars": no_ask,
        "price_level_structure": price_level_structure,
        "exchange_index": exchange_index,
    }
    if price_ranges is not None:
        d["price_ranges"] = price_ranges
    return d


def _ev(title: str, market: dict) -> dict:
    return {"title": title, "markets": [market]}


def _build_events() -> list:
    return [
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_EXP, "EVT-EXP", "Will X happen?", "Outcome Main",
            "0.50", "0.45", price_level_structure="linear_cent",
        )),
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_CHEAP, "EVT-CHEAP", "Will X happen?", "Outcome Main",
            "0.20", "0.75", price_level_structure="linear_cent",
        )),
        _ev("Tick Event", _mk_market(
            _TICKER_TICK_A, "EVT-TICK-A", "Tick Test Market", "Outcome",
            "0.90", "0.85", price_level_structure="center_deci_edge_centi_cent",
            price_ranges=_TICK_PRICE_RANGES,
        )),
        _ev("Tick Event", _mk_market(
            _TICKER_TICK_B, "EVT-TICK-B", "Tick Test Market", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Shard Event", _mk_market(
            _TICKER_SHARD_A, "EVT-SHARD-A", "Shard Test Market", "Outcome",
            "0.60", "0.35", exchange_index=1,
        )),
        _ev("Shard Event", _mk_market(
            _TICKER_SHARD_B, "EVT-SHARD-B", "Shard Test Market", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Held Event", _mk_market(
            _TICKER_HELD_A, "EVT-HELD-A", "Held Question", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Held Event", _mk_market(
            _TICKER_HELD_B, "EVT-HELD-B", "Held Question", "Outcome",
            "0.20", "0.75",
        )),
    ]


def _raw_events_page(events: list, cursor: str | None = None) -> SimpleNamespace:
    return _raw_json_response({"events": events, "cursor": cursor})


# Orderbook depth for the one tradeable pair: market A's YES bids complement
# to a NO ask of 0.45 (matching nA above); market B's NO bids complement to a
# YES ask of 0.20 (matching pB above) — see scanner._bids_to_ask_levels.
_ORDERBOOK_PAYLOADS = {
    _TICKER_SAME_EXP: {"orderbook_fp": {"yes_dollars": [["0.55", "100"]], "no_dollars": []}},
    _TICKER_SAME_CHEAP: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.80", "100"]]}},
}


def _orderbook_side_effect(ticker: str) -> SimpleNamespace:
    payload = _ORDERBOOK_PAYLOADS.get(
        ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    )
    return _raw_json_response(payload)


def _positions_side_effect(held_payload: dict, lookup_map: dict):
    """Dispatches get_positions_without_preload_content calls: a "ticker"
    kwarg means trader._position_count's per-ticker ambiguity lookup;
    otherwise it's scanner.get_held_tickers' paginated listing."""
    def _effect(**kwargs):
        if "ticker" in kwargs:
            payload = lookup_map.get(kwargs["ticker"], {"market_positions": []})
        else:
            payload = held_payload
        return _raw_json_response(payload)

    return _effect


def _order_side_effect(fill_pattern: list):
    """Programs client.rest_client.request for a sequence of V2 order POSTs.

    Each tag in fill_pattern is "full" (fills the exact requested count,
    read back from the submitted body so the response is always self-
    consistent regardless of the sized contract count), "kill" (fill_count
    0), or "error" (HTTP 500, exercising the ambiguous-response path)."""
    state = {"i": 0}

    def _effect(verb, url, headers=None, body=None):
        idx = state["i"]
        state["i"] += 1
        tag = fill_pattern[idx]
        if tag == "error":
            return _raw_json_response(
                {"error": "internal"}, status=500, reason="Internal Server Error"
            )
        requested = int(Decimal(body["count"]))
        fill = requested if tag == "full" else 0
        payload = {"order": {"order_id": f"ord-{idx}", "fill_count": fill, "remaining_count": 0}}
        return _raw_json_response(payload)

    return _effect


def _live_shape_client(
    monkeypatch,
    *,
    balance_payload: dict,
    include_held_position: bool = True,
    position_lookup_responses: dict | None = None,
    order_side_effect=None,
    mve_bailout: bool = False,
):
    """Build a MagicMock KalshiClient wired end-to-end with current-generation
    payload shapes over the fixed 4-group market set described above.

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to configure
            scanner.INCLUDE_MVE_MARKETS / MVE_MAX_EMPTY_PAGES.
        balance_payload (dict): Body for get_balance_without_preload_content.
        include_held_position (bool): Whether the account holds HELD-A —
            True (default) matches every test's expectation that the held
            pair never trades in prod.
        position_lookup_responses (dict | None): ticker -> positions body,
            consulted only by trader._position_count's ambiguous-leg lookup.
        order_side_effect: Optional callable for client.rest_client.request
            (see _order_side_effect). Only the V2 live-execution tests need
            this — dry-run and dev paths never submit orders.
        mve_bailout (bool): When True, leaves INCLUDE_MVE_MARKETS at its
            default (True) and shrinks MVE_MAX_EMPTY_PAGES to 1 so the MVE
            pull's bail-out fires almost immediately instead of the loop
            terminating on a None cursor — this is what actually exercises
            the bail-out path rather than the ordinary end-of-listing exit.
            When False (default), INCLUDE_MVE_MARKETS is turned off so the
            MVE loop is skipped entirely — the cleaner setup for every test
            that isn't specifically about the MVE bail-out.
    """
    if mve_bailout:
        monkeypatch.setattr(scanner_mod, "INCLUDE_MVE_MARKETS", True)
        monkeypatch.setattr(scanner_mod, "MVE_MAX_EMPTY_PAGES", 1)
    else:
        monkeypatch.setattr(scanner_mod, "INCLUDE_MVE_MARKETS", False)

    client = MagicMock()

    client.get_events_without_preload_content = MagicMock(
        return_value=_raw_events_page(_build_events())
    )
    if mve_bailout:
        # A non-None cursor on every page means only the consecutive-empty-
        # page bail-out (not "no cursor left") can end this loop.
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_events_page([], cursor="MVE-CURSOR-1")
        )

    client.get_balance_without_preload_content = MagicMock(
        return_value=_raw_json_response(balance_payload)
    )

    held_payload = {
        "market_positions": (
            [{"ticker": _TICKER_HELD_A, "position_fp": "3.00"}] if include_held_position else []
        ),
        "cursor": None,
    }
    client.get_positions_without_preload_content = MagicMock(
        side_effect=_positions_side_effect(held_payload, position_lookup_responses or {})
    )

    client.get_market_orderbook_without_preload_content = MagicMock(
        side_effect=_orderbook_side_effect
    )

    # Legacy order path must never be touched — ORDER_API_VERSION defaults to "v2".
    client.create_order_without_preload_content = MagicMock()

    # V2 submission plumbing — signed_request_json reads these directly.
    client.configuration.host = "https://api.elections.kalshi.com/trade-api/v2"
    client.kalshi_auth.create_auth_headers = MagicMock(return_value={})
    client.rest_client.request = MagicMock(
        side_effect=order_side_effect if order_side_effect is not None else []
    )

    return client


class TestRunDevLiveShapeReplay:
    def test_run_dev_dry_run_end_to_end_current_payload_shapes(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,  # unused in dev, harmless
            mve_bailout=True,
        )
        captured: dict = {}

        def fake_write_dev_simulation(results, all_candidates, balance_cents):
            captured["results"] = results
            captured["all_candidates"] = all_candidates
            captured["balance_cents"] = balance_cents
            return pathlib.Path("/fake/dev_sim.xlsx")

        monkeypatch.setattr(main, "write_dev_simulation", fake_write_dev_simulation)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert "results" in captured
        simulated = [r for r in captured["results"] if r.status == "simulated"]
        assert simulated, "expected at least one simulated TradeResult"

        # The simulated trade's prices came straight from the *_dollars payloads.
        main_spec = next(
            r.spec for r in simulated if r.spec.pair.market_a.ticker == _TICKER_SAME_EXP
        )
        assert main_spec.pair.nA == pytest.approx(0.45)
        assert main_spec.pair.pB == pytest.approx(0.20)

        # The exchange_index=1 market was dropped at ingest — it appears in no pair.
        all_tickers: set = set()
        for pair in captured["all_candidates"]:
            all_tickers.add(pair.market_a.ticker)
            all_tickers.add(pair.market_b.ticker)
        assert _TICKER_SHARD_A not in all_tickers

        # Sub-cent tick fields flowed through ingest without breaking the run.
        tick_pair = next(
            p for p in captured["all_candidates"] if p.market_a.ticker == _TICKER_TICK_A
        )
        assert tick_pair.market_a.price_level_structure == "center_deci_edge_centi_cent"
        assert tick_pair.market_a.price_ranges is not None

        assert "Skipped 1 markets on non-routable exchange shards" in caplog.text
        assert "MVE fetch:" in caplog.text  # proves the bail-out path actually fired

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0


class TestRunProdDryRunLiveShapeReplay:
    def test_run_prod_dry_run_end_to_end_current_payload_shapes(self, monkeypatch, caplog):
        client = _live_shape_client(monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD)
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after):
            captured["results"] = results
            captured["balance_before"] = balance_before
            captured["balance_after"] = balance_after
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)

        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        # Shard-0 balance ($250.00), never the $10,250 aggregate or the $9,999 shard-1 entry.
        assert "$250.00" in caplog.text

        assert "results" in captured
        results = captured["results"]
        assert results, "expected at least one result"
        assert all(r.status == "simulated" for r in results)

        # HELD-A is held — the pair it would have formed with HELD-B must never surface.
        tickers: set = set()
        for r in results:
            tickers.add(r.spec.pair.market_a.ticker)
            tickers.add(r.spec.pair.market_b.ticker)
        assert _TICKER_HELD_A not in tickers
        assert _TICKER_HELD_B not in tickers

        held_lookup_calls = [
            c for c in client.get_positions_without_preload_content.call_args_list
            if "ticker" not in c.kwargs
        ]
        assert held_lookup_calls, "expected get_held_tickers to have queried positions"

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0

    def test_run_prod_aborts_below_min_balance(self, monkeypatch, caplog):
        client = _live_shape_client(monkeypatch, balance_payload=_LOW_BALANCE_PAYLOAD)
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.WARNING):
            main._run_prod(client, args)

        assert "below minimum" in caplog.text
        client.get_events_without_preload_content.assert_not_called()
        client.get_positions_without_preload_content.assert_not_called()
        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0


class TestRunProdLiveV2Replay:
    """dry_run=False replays against the default ORDER_API_VERSION="v2" path
    (deliberately not monkeypatched, per the task — this proves the default
    live-execution path end-to-end). Every test tunes the market set down to
    exactly one tradeable, selectable pair (SAME-EXP / SAME-CHEAP) so order
    counts are fully deterministic."""

    @staticmethod
    def _run(monkeypatch, fill_pattern, position_lookup_responses=None):
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            order_side_effect=_order_side_effect(fill_pattern),
            position_lookup_responses=position_lookup_responses,
        )
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        args = SimpleNamespace(dry_run=False, max_horizon_days=None)
        main._run_prod(client, args)
        return client, captured

    def test_run_prod_live_v2_all_filled_end_to_end(self, monkeypatch):
        # Not monkeypatched anywhere in this class — proves the DEFAULT order
        # path (not a forced-on "v2") is what gets exercised end-to-end.
        assert ORDER_API_VERSION == "v2"

        client, captured = self._run(monkeypatch, ["full", "full"])

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 2

        price_re = re.compile(r"^\d\.\d{4}$")
        count_re = re.compile(r"^\d+\.00$")
        for call in calls:
            url = call.args[1]
            assert V2_ORDER_PATH in url
            body = call.kwargs["body"]
            assert price_re.match(body["price"])
            assert count_re.match(body["count"])
            assert body["time_in_force"] == "fill_or_kill"
            assert body["exchange_index"] == 0

        assert len(captured["results"]) == 1
        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_b_killed_rolls_back_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["full", "kill", "full"])

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 3

        rollback_body = calls[2].kwargs["body"]
        assert rollback_body["side"] == "bid"
        assert rollback_body["reduce_only"] is True
        # Replay markets use the default $0.01 grid, so the finest-grid
        # rollback target floors to the highest cent-grid level
        assert rollback_body["price"] == "0.9900"

        assert captured["results"][0].status == "rolled_back"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_a_killed_is_failed_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["kill"])

        assert client.rest_client.request.call_count == 1
        assert captured["results"][0].status == "failed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_error_response_routes_to_position_lookup(self, monkeypatch):
        client, captured = self._run(
            monkeypatch,
            ["full", "error"],
            position_lookup_responses={
                _TICKER_SAME_CHEAP: {
                    "market_positions": [{"ticker": _TICKER_SAME_CHEAP, "position_fp": "12.00"}]
                },
            },
        )

        # Leg A filled, leg B raised — exactly two order POSTs, no rollback.
        assert client.rest_client.request.call_count == 2

        lookup_calls = [
            c for c in client.get_positions_without_preload_content.call_args_list
            if c.kwargs.get("ticker") == _TICKER_SAME_CHEAP
        ]
        assert lookup_calls, "expected a position lookup for the leg-B ticker"

        result = captured["results"][0]
        assert result.status == "executed"
        assert "ambiguous" in (result.error or "").lower()
        client.create_order_without_preload_content.assert_not_called()
