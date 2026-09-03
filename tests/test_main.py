"""
File: test_main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline tests for kalshi_betting.main — the live-pipeline orchestrator's
    pure helpers (_truncate, _format_deadline, _dedup_pairs,
    _compute_trade_specs, _no_pairs_msg), its logging setup (_setup_logging,
    including the BS-25 rotating file handler), its process exit-code contract
    (BS-14), and end-to-end "live-shape replay" runs of _run_dev/_run_prod
    against a MagicMock client wired with CURRENT-generation (2026-08+) Kalshi
    payload shapes — dollar-string prices, yes_sub_title, orderbook_fp books,
    shard-aware balance_breakdown, position_fp positions, V2 order responses —
    the closest offline substitute for a live smoke test.

    On the exit-code half: _run_prod / _run_dev return an int outcome code and
    main() propagates it to the OS via sys.exit(), so the scheduler (a separate
    subprocess, see scheduler.run_job) can distinguish a clean run from a
    low-balance skip or a run whose trades need manual review.

Dependencies:
    Imports _run_dev/_run_prod and the pure helpers from kalshi_betting.main,
    plus config constants asserted against. The live-shape replays mock all
    Kalshi API interaction at the HTTP boundary (raw-response mocks and
    rest_client.request); the exit-code tests mock the heavy collaborators
    (auth, scanner, strategy, trader, reporter) at their main-module import
    sites per project policy. Reporter Excel writers are patched out so no
    files are written, and PROJECT_ROOT is redirected to tmp_path wherever
    main() configures logging, so the real kalshi_arb.log / trade_log.xlsx are
    never touched.

Notes:
    The V2 live-execution replays run with dry_run=False against the config
    default ORDER_API_VERSION="v2" — deliberately not monkeypatched, so they
    prove the default path. Order responses are generated from each request's
    own submitted count, so the tests don't depend on exact Kelly sizing.

    verify_auth() returns dict[int, int] (exchange_index -> cents), never a
    scalar — every mock of it here must return a dict, and _run_prod sizes on
    sum(...) of it.
"""
import json
import logging
import logging.handlers
import pathlib
import re
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kalshi_betting import main
from kalshi_betting import scanner as scanner_mod
from kalshi_betting import trader as trader_mod
from kalshi_betting.config import (
    DEFAULT_EXCHANGE_INDEX,
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    MIN_BALANCE_CENTS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    ORDER_API_VERSION,
    SAME_TITLE_MIN_PRICE_DIFF,
    TRANSFER_PATH,
    V2_ORDER_PATH,
)
from kalshi_betting.reporter import TradeResult


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
            # RotatingFileHandler, not a plain FileHandler (BS-25) — a
            # scheduler daemon runs this weekly forever, so the log must rotate
            assert logging.handlers.RotatingFileHandler in handler_types
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
#   Shard Event / Shard Test Market -> SHARD1-A (exchange_index=1) /
#       SHARD1-B (shard 0) — a cross-shard pair that IS ingested (market data
#       is cross-shard) but is priced non-tradeable, so it only ever proves
#       ingest tagging, never execution.
#   Held Event / Held Question -> HELD-A (held in prod) / HELD-B — forms a
#       tradeable pair in dev (no held-ticker filtering) but never in prod.
#
# The one tradeable pair's cheap leg can be moved onto another shard with
# _live_shape_client(same_cheap_shard=...), which is what the cross-shard
# routing and collateral-transfer replays use.
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
# $250.00 and shard 1 holds $9,999.00 — sizing sums the BREAKDOWN
# (1024900 cents = $10,249.00), deliberately != the $10,250.00 top-level
# balance_dollars aggregate, so the replay proves the breakdown sum is what
# Kelly sizing sees, not the top-level field.
_LIVE_BALANCE_PAYLOAD = {
    "balance": 114,
    "balance_dollars": "10250.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
    ],
}

# Same shape, but the breakdown SUM is below MIN_BALANCE_CENTS ($50 = 5000
# cents): shard0 $5.00 + shard1 $3.00 = 800 cents. Under multi-shard
# semantics an account with money parked on shard 1 must still be summed in
# — it's the total across shards that must clear the floor, not any single
# shard — so this fixture only aborts because the TOTAL is sub-minimum.
_LOW_BALANCE_PAYLOAD = {
    "balance": 1,
    "balance_dollars": "8.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5.0000"},
        {"exchange_index": 1, "balance": "3.0000"},
    ],
}

# Everything on shard 0, nothing on shard 1 — the collateral-transfer replays
# put the cheap leg on shard 1, so leg B's cash requirement is a pure deficit
# that only an intra-exchange transfer out of shard 0's surplus can cover.
_SHARD1_EMPTY_BALANCE = {
    "balance": 114,
    "balance_dollars": "10000.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "10000.0000"},
        {"exchange_index": 1, "balance": "0.0000"},
    ],
}

# What the account reads back AFTER a transfer settles — trader's settlement
# poll re-reads the balance through auth.verify_auth, so this is how the
# replay proves the money landed before any order relies on it.
_SHARD1_SETTLED_BALANCE = {
    "balance": 114,
    "balance_dollars": "10000.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5000.0000"},
        {"exchange_index": 1, "balance": "5000.0000"},
    ],
}

# Funds parked on an advertised, trading-active shard that produced ZERO
# ingested markets — the one coverage gap check_shard_coverage calls CRITICAL.
_SHARD2_FUNDED_BALANCE = {
    "balance": 114,
    "balance_dollars": "10349.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
        {"exchange_index": 2, "balance": "100.0000"},
    ],
}


def _status_entry(
    idx: int, *, trading_active: bool = True, transfers_active: bool = True,
    description: str = "Main",
) -> dict:
    """One entry of GET /exchange/status's exchange_index_statuses array."""
    return {
        "exchange_index": idx,
        "trading_active": trading_active,
        "exchange_active": True,
        "intra_exchange_transfers_active": transfers_active,
        "description": description,
    }


def _status_payload(*entries: dict) -> dict:
    """A full GET /exchange/status body wrapping the given shard entries."""
    return {
        "exchange_active": True,
        "trading_active": True,
        "exchange_index_statuses": list(entries),
    }


# Default advertised topology: shards 0 and 1, both trading and both accepting
# intra-exchange transfers — matching the fixed market set, which spans exactly
# those two shards.
_EXCHANGE_STATUS_PAYLOAD = _status_payload(
    _status_entry(0, description="Main"),
    _status_entry(1, description="Combos"),
)


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


def _build_events(same_cheap_shard: int = 0) -> list:
    """The fixed market set. `same_cheap_shard` moves the tradeable pair's
    cheap leg (leg B) onto another exchange shard, which is what turns the
    one selectable trade into a cross-shard one."""
    return [
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_EXP, "EVT-EXP", "Will X happen?", "Outcome Main",
            "0.50", "0.45", price_level_structure="linear_cent",
        )),
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_CHEAP, "EVT-CHEAP", "Will X happen?", "Outcome Main",
            "0.20", "0.75", price_level_structure="linear_cent",
            exchange_index=same_cheap_shard,
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
    kwarg means trader._position_count's per-ticker lookup; otherwise it's
    scanner.get_held_tickers' paginated listing.

    A lookup_map value may be a single payload (every read of that ticker sees
    it) or a LIST of payloads consumed in order, which is what models a
    position CHANGING across reads. trader attributes an ambiguous fill by
    DELTA, never by the absolute holding, so a leg whose fill must read as
    "filled" needs a baseline read (taken before either order is submitted)
    followed by a post-exception read that differs by exactly the leg's count.
    The last entry of a list repeats once exhausted."""
    sequences = {t: list(v) for t, v in lookup_map.items() if isinstance(v, list)}

    def _effect(**kwargs):
        if "ticker" in kwargs:
            ticker = kwargs["ticker"]
            if ticker in sequences:
                seq = sequences[ticker]
                payload = seq.pop(0) if len(seq) > 1 else seq[0]
            else:
                payload = lookup_map.get(ticker, {"market_positions": []})
            # A callable entry is resolved at read time, so a payload can be
            # derived from what was actually submitted rather than hardcoding
            # a count the Kelly sizing might change
            if callable(payload):
                payload = payload()
        else:
            payload = held_payload
        return _raw_json_response(payload)

    return _effect


def _balance_side_effect(first: dict, later: dict | None):
    """Programs get_balance_without_preload_content. The FIRST read (the one
    _run_prod sizes on) returns `first`; every later read — trader's transfer
    settlement poll, then the post-trade balance — returns `later`, which is
    how a replay models funds actually landing on a shard."""
    state = {"n": 0}

    def _effect(*args, **kwargs):
        state["n"] += 1
        if later is None or state["n"] == 1:
            return _raw_json_response(first)
        return _raw_json_response(later)

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


def _transfer_and_order_side_effect(fill_pattern: list):
    """client.rest_client.request carries BOTH signed POSTs the live path
    makes — the collateral transfer and the V2 orders — so dispatch on the
    URL. Transfers are accepted with a transfer_id; orders fall through to
    _order_side_effect, whose fill_pattern therefore only counts orders."""
    orders = _order_side_effect(fill_pattern)

    def _effect(verb, url, headers=None, body=None):
        if TRANSFER_PATH in url:
            return _raw_json_response({"transfer_id": "xfer-1"})
        return orders(verb, url, headers=headers, body=body)

    return _effect


def _live_shape_client(
    monkeypatch,
    *,
    balance_payload: dict,
    balance_payload_after: dict | None = None,
    exchange_status_payload: dict | None = _EXCHANGE_STATUS_PAYLOAD,
    same_cheap_shard: int = 0,
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
        balance_payload (dict): Body for the FIRST get_balance_without_preload_content
            read — the one prod sizing is based on.
        balance_payload_after (dict | None): Body for every LATER balance read
            (trader's transfer settlement poll, then the post-trade balance).
            None (default) keeps returning balance_payload forever.
        exchange_status_payload (dict | None): Body for
            get_exchange_status_without_preload_content, defaulting to shards
            0 and 1 both trading-active with transfers active. Pass None to
            leave the attribute unwired, which is what every OTHER test module's
            MagicMock client looks like: fetch_json_page then chokes on the
            auto-created MagicMock attribute, scanner.fetch_shard_statuses'
            broad except swallows it, and the run degrades to single-shard
            semantics (see test_unwired_exchange_status_degrades_to_single_shard).
        same_cheap_shard (int): exchange_index for the tradeable pair's cheap
            leg (SAME-CHEAP / leg B). 0 (default) keeps the pair on one shard;
            1 makes it the cross-shard pair the routing and collateral replays
            need.
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
        return_value=_raw_events_page(_build_events(same_cheap_shard=same_cheap_shard))
    )
    if mve_bailout:
        # A non-None cursor on every page means only the consecutive-empty-
        # page bail-out (not "no cursor left") can end this loop.
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_events_page([], cursor="MVE-CURSOR-1")
        )

    # Per-shard exchange status: read raw, so the run derives real
    # trading_active/transfers_active facts rather than falling back to
    # single-shard semantics.
    if exchange_status_payload is not None:
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=_raw_json_response(exchange_status_payload)
        )

    client.get_balance_without_preload_content = MagicMock(
        side_effect=_balance_side_effect(balance_payload, balance_payload_after)
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


def _capture_dev_simulation(monkeypatch) -> dict:
    """Patch out the Excel writer and capture what _run_dev hands it."""
    captured: dict = {}

    def fake_write_dev_simulation(results, all_candidates, balance_cents):
        captured["results"] = results
        captured["all_candidates"] = all_candidates
        captured["balance_cents"] = balance_cents
        return pathlib.Path("/fake/dev_sim.xlsx")

    monkeypatch.setattr(main, "write_dev_simulation", fake_write_dev_simulation)
    return captured


def _candidate_tickers(candidates) -> set:
    tickers: set = set()
    for pair in candidates:
        tickers.add(pair.market_a.ticker)
        tickers.add(pair.market_b.ticker)
    return tickers


class TestRunDevLiveShapeReplay:
    def test_run_dev_dry_run_end_to_end_current_payload_shapes(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,  # unused in dev, harmless
            mve_bailout=True,
        )
        captured = _capture_dev_simulation(monkeypatch)

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

        # Market data is cross-shard: the exchange_index=1 market is INGESTED
        # and tagged, not dropped. (It was dropped before the multi-shard flip.)
        all_candidates = captured["all_candidates"]
        assert _TICKER_SHARD_A in _candidate_tickers(all_candidates)
        shard_pair = next(
            p for p in all_candidates if p.market_a.ticker == _TICKER_SHARD_A
        )
        assert shard_pair.market_a.exchange_index == 1
        assert shard_pair.market_b.exchange_index == 0

        # Sub-cent tick fields flowed through ingest without breaking the run.
        tick_pair = next(
            p for p in all_candidates if p.market_a.ticker == _TICKER_TICK_A
        )
        assert tick_pair.market_a.price_level_structure == "center_deci_edge_centi_cent"
        assert tick_pair.market_a.price_ranges is not None

        # The old drop-at-ingest warning is gone, and the per-shard ingest
        # census (the only signal of which shards a run actually saw) is on.
        assert "non-routable" not in caplog.text
        assert "Ingested markets by shard: {0: 7, 1: 1}" in caplog.text
        # Both advertised shards produced markets — nothing to report.
        assert "Full shard coverage: shards [0, 1] scanned" in caplog.text
        assert "SHARD COVERAGE FAILURE" not in caplog.text

        assert "MVE fetch:" in caplog.text  # proves the bail-out path actually fired

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0

    def test_run_dev_drops_markets_on_trading_inactive_shard(self, monkeypatch, caplog):
        # The ONE ingest-time shard exclusion: nothing on a halted shard can be
        # traded, nor should it linger as a stale candidate.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, trading_active=False, description="Combos"),
            ),
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert _TICKER_SHARD_A not in _candidate_tickers(captured["all_candidates"])
        assert "Skipped 1 markets on trading-inactive exchange shards [1]" in caplog.text
        assert "Ingested markets by shard: {0: 7}" in caplog.text
        # A deliberately-halted shard is never re-reported as a coverage gap —
        # the ingest drop above already warned about it.
        assert "SHARD COVERAGE FAILURE" not in caplog.text
        assert "Shard coverage:" not in caplog.text

    def test_unwired_exchange_status_degrades_to_single_shard(self, monkeypatch, caplog):
        # Every other test module's MagicMock client leaves
        # get_exchange_status_without_preload_content unwired. That must remain
        # harmless: the auto-created MagicMock attribute makes fetch_json_page
        # raise, scanner.fetch_shard_statuses swallows it, and the run keeps
        # every market and simply cannot assess coverage.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=None,
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert _TICKER_SHARD_A in _candidate_tickers(captured["all_candidates"])
        assert "Per-shard exchange status unavailable — coverage not assessable." in caplog.text
        assert "trading-inactive" not in caplog.text
        assert "SHARD COVERAGE FAILURE" not in caplog.text


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

        # Sum of the breakdown ($250.00 + $9,999.00 = $10,249.00) — never the
        # $10,250.00 top-level balance_dollars aggregate. Sizing is
        # portfolio-wide across shards, but must read the breakdown, not the
        # aggregate field, so this pins the distinction.
        assert "$10249.00" in caplog.text

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

    def test_run_prod_flags_funded_advertised_shard_with_no_markets(self, monkeypatch, caplog):
        # Shard 2 is advertised, trading-active, holds $100 — and produced zero
        # ingested markets. That is a real blind spot (a pair could exist there
        # and go undetected), so it is CRITICAL. The run still continues.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_SHARD2_FUNDED_BALANCE,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, description="Combos"),
                _status_entry(2, description="Crypto"),
            ),
        )
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert critical, "expected a CRITICAL coverage record"
        assert any("SHARD COVERAGE FAILURE" in r.getMessage() for r in critical)
        assert any(
            "shard 2" in r.getMessage() and "holds account funds" in r.getMessage()
            for r in critical
        )
        # Never an abort — the run trades on whatever shards WERE covered.
        assert captured.get("results"), "coverage failure must not stop the run"

    def test_run_prod_warns_when_empty_advertised_shard_holds_no_funds(self, monkeypatch, caplog):
        # Same topology minus the money on shard 2: an advertised-but-empty
        # shard is expected during the rollout, so it must not cry wolf.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, description="Combos"),
                _status_entry(2, description="Crypto"),
            ),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        assert "SHARD COVERAGE FAILURE" not in caplog.text
        assert "Shard coverage: advertised active shard 2" in caplog.text
        assert "may be legitimately empty" in caplog.text

    def test_run_prod_aborts_below_min_balance(self, monkeypatch, caplog):
        client = _live_shape_client(monkeypatch, balance_payload=_LOW_BALANCE_PAYLOAD)
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.WARNING):
            main._run_prod(client, args)

        assert "below minimum" in caplog.text
        client.get_events_without_preload_content.assert_not_called()
        client.get_positions_without_preload_content.assert_not_called()
        # The shard-status read happens after the balance gate — an account
        # that can't trade shouldn't spend an API call on exchange status.
        client.get_exchange_status_without_preload_content.assert_not_called()
        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0


class TestRunProdLiveV2Replay:
    """dry_run=False replays against the default ORDER_API_VERSION="v2" path
    (deliberately not monkeypatched, per the task — this proves the default
    live-execution path end-to-end). Every test tunes the market set down to
    exactly one tradeable, selectable pair (SAME-EXP / SAME-CHEAP) so order
    counts are fully deterministic."""

    @pytest.fixture(autouse=True)
    def _fresh_v2_mapping_latch(self, monkeypatch):
        """Every replay starts as a fresh process does: the V2 NO-leg mapping
        unconfirmed, so trader._confirm_v2_no_mapping's first-fill check is
        genuinely exercised end-to-end. The latch is a process global that real
        execution flips, so monkeypatch also keeps it from leaking between
        tests."""
        monkeypatch.setattr(trader_mod, "_V2_NO_MAPPING_CONFIRMED", False)

    @staticmethod
    def _run(
        monkeypatch,
        fill_pattern,
        position_lookup_responses=None,
        *,
        same_cheap_shard: int = 0,
        balance_payload: dict = _LIVE_BALANCE_PAYLOAD,
        balance_payload_after: dict | None = None,
        order_side_effect=None,
        dry_run: bool = False,
    ):
        # After a filled NO buy the exchange reports a NEGATIVE (short-YES)
        # position on the leg-A ticker — that sign is what trader's first-fill
        # backstop verifies before leg B is submitted, so the replay account
        # must model it or every pair would stop at manual_review.
        lookups = {
            _TICKER_SAME_EXP: {
                "market_positions": [
                    {"ticker": _TICKER_SAME_EXP, "position_fp": "-12.00"}
                ]
            },
        }
        lookups.update(position_lookup_responses or {})
        client = _live_shape_client(
            monkeypatch,
            balance_payload=balance_payload,
            balance_payload_after=balance_payload_after,
            same_cheap_shard=same_cheap_shard,
            order_side_effect=(
                order_side_effect if order_side_effect is not None
                else _order_side_effect(fill_pattern)
            ),
            position_lookup_responses=lookups,
        )
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        args = SimpleNamespace(dry_run=dry_run, max_horizon_days=None)
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
        # The unwind is LOSS-FLOORED, not a flat top-of-grid bid: leg A's
        # scanned NO entry is 0.45, so the floor is 45 - 12 = 33c and the bid
        # cap is its YES-book mirror, 1 - 0.33 = 0.67, already on the replay
        # markets' default $0.01 grid and well under the 0.99 top-of-grid clamp
        assert rollback_body["price"] == "0.6700"

        assert captured["results"][0].status == "rolled_back"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_a_killed_is_failed_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["kill"])

        assert client.rest_client.request.call_count == 1
        assert captured["results"][0].status == "failed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_error_response_routes_to_position_lookup(self, monkeypatch):
        # Record what each order actually asked for, so the modelled position
        # move can be leg B's OWN count rather than a hardcoded number that
        # would silently drift with Kelly sizing.
        submitted: list = []
        base_orders = _order_side_effect(["full", "error"])

        def recording_orders(verb, url, headers=None, body=None):
            if TRANSFER_PATH not in url:
                submitted.append(body["count"])
            return base_orders(verb, url, headers=headers, body=body)

        client, captured = self._run(
            monkeypatch,
            ["full", "error"],
            order_side_effect=recording_orders,
            position_lookup_responses={
                # Delta semantics: the baseline read (taken before leg A is
                # submitted) must show FLAT, and the post-exception read must
                # show exactly the contracts leg B bought. A single static
                # payload would give a delta of 0 — a confirmed non-fill —
                # and roll leg A back, the opposite of what this pins.
                _TICKER_SAME_CHEAP: [
                    {"market_positions": []},
                    lambda: {
                        "market_positions": [
                            {"ticker": _TICKER_SAME_CHEAP, "position_fp": submitted[1]}
                        ]
                    },
                ],
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

    def test_run_prod_live_v2_routes_each_leg_to_its_own_shard(self, monkeypatch):
        # The behavioral point of multi-shard support: a pair whose legs live
        # on different shards is now tradeable, and each V2 body must carry
        # ITS OWN market's exchange_index — never one shared value, never the
        # -1 auto-route sentinel. Shard 1 is already funded here ($9,999), so
        # no collateral transfer is involved.
        client, captured = self._run(monkeypatch, ["full", "full"], same_cheap_shard=1)

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 2
        assert all(V2_ORDER_PATH in c.args[1] for c in calls)
        assert all(TRANSFER_PATH not in c.args[1] for c in calls)

        by_ticker = {c.kwargs["body"]["ticker"]: c.kwargs["body"] for c in calls}
        assert by_ticker[_TICKER_SAME_EXP]["exchange_index"] == 0
        assert by_ticker[_TICKER_SAME_CHEAP]["exchange_index"] == 1

        assert len(captured["results"]) == 1
        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_dry_run_plans_but_never_posts_a_collateral_transfer(
        self, monkeypatch, caplog,
    ):
        # Leg B sits on shard 1, which holds $0 — a genuine deficit. In dry-run
        # the plan must be logged and NOTHING posted: not the transfer, not the
        # orders.
        with caplog.at_level(logging.INFO):
            client, _ = self._run(
                monkeypatch,
                [],
                same_cheap_shard=1,
                balance_payload=_SHARD1_EMPTY_BALANCE,
                dry_run=True,
            )

        assert "DRY RUN: would transfer" in caplog.text
        assert "shard 0→1" in caplog.text
        assert client.rest_client.request.call_count == 0
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_transfers_collateral_before_ordering(self, monkeypatch, caplog):
        # Same deficit, live: the transfer POST must land BEFORE any order
        # POST, and the settlement poll must see the funds actually arrive
        # (the second balance read) before execution proceeds.
        with caplog.at_level(logging.INFO):
            client, captured = self._run(
                monkeypatch,
                ["full", "full"],
                same_cheap_shard=1,
                balance_payload=_SHARD1_EMPTY_BALANCE,
                balance_payload_after=_SHARD1_SETTLED_BALANCE,
                order_side_effect=_transfer_and_order_side_effect(["full", "full"]),
            )

        urls = [c.args[1] for c in client.rest_client.request.call_args_list]
        assert len(urls) == 3
        assert TRANSFER_PATH in urls[0], "collateral must move before any order"
        assert all(V2_ORDER_PATH in u for u in urls[1:])

        transfer_body = client.rest_client.request.call_args_list[0].kwargs["body"]
        assert transfer_body["source_exchange_shard"] == 0
        assert transfer_body["destination_exchange_shard"] == 1
        assert transfer_body["amount"] > 0

        assert "Collateral transfer accepted" in caplog.text
        assert "All shard collateral requirements confirmed funded." in caplog.text

        order_bodies = {
            c.kwargs["body"]["ticker"]: c.kwargs["body"]
            for c in client.rest_client.request.call_args_list[1:]
        }
        assert order_bodies[_TICKER_SAME_EXP]["exchange_index"] == 0
        assert order_bodies[_TICKER_SAME_CHEAP]["exchange_index"] == 1

        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_drops_the_trade_when_its_shard_cannot_be_funded(
        self, monkeypatch, caplog,
    ):
        # Transfers reported inactive on shard 1 => the transfer is never
        # attempted, the only selected trade is dropped, and NO order is
        # submitted underfunded.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_SHARD1_EMPTY_BALANCE,
            same_cheap_shard=1,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, transfers_active=False, description="Combos"),
            ),
            order_side_effect=_transfer_and_order_side_effect(["full", "full"]),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=False, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        assert "Intra-exchange transfers are not active" in caplog.text
        assert "No selected pair could be funded on its exchange shard" in caplog.text
        assert client.rest_client.request.call_count == 0
        client.create_order_without_preload_content.assert_not_called()


def _args(dry_run: bool = False, max_horizon_days=None) -> SimpleNamespace:
    """Minimal stand-in for the argparse.Namespace _run_prod/_run_dev read."""
    return SimpleNamespace(dry_run=dry_run, max_horizon_days=max_horizon_days)


def make_spec() -> SimpleNamespace:
    """Minimal TradeSpec-like stub with concrete (non-Mock) scalar fields.

    print_pairs_table / _print_portfolio format several fields with a format
    spec (e.g. f"{pair.pA:.2%}") — a bare MagicMock's default __format__
    support is unreliable, so pair/spec fields are plain SimpleNamespace
    values instead of auto-attributing MagicMocks.
    """
    pair = SimpleNamespace(
        pair_type="time_series",
        market_a=SimpleNamespace(
            ticker="TICK-A", title="Market A", subtitle="", close_time=None,
            exchange_index=DEFAULT_EXCHANGE_INDEX,
        ),
        market_b=SimpleNamespace(
            ticker="TICK-B", title="Market B", subtitle="", close_time=None,
            exchange_index=DEFAULT_EXCHANGE_INDEX,
        ),
        pA=0.60,
        pB=0.30,
        nA=0.40,
        tradeable=True,
        canonical_title="Test pair",
    )
    return SimpleNamespace(
        pair=pair,
        x=5,
        y=5,
        total_cost=2.0,
        # Per-leg fee-inclusive costs — trader._required_cents_by_shard reads
        # these to total the collateral each shard must hold before execution
        cost_with_fees_a=1.0,
        cost_with_fees_b=1.0,
        min_payoff=0.50,
        profit_ratio=0.10,
        monthly_profit_ratio=0.20,
        kelly_fraction=0.10,
        kelly_p=0.60,
    )


class TestRunProdExitCodes:
    @patch("kalshi_betting.main.verify_auth")
    def test_low_balance_returns_skip_code(self, mock_verify_auth):
        # Balance below MIN_BALANCE_CENTS must short-circuit before any scan —
        # the bare `return` this used to be silently exited 0.
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1}
        client = MagicMock()

        code = main._run_prod(client, _args())

        assert code == EXIT_SKIPPED_LOW_BALANCE
        assert code == 10

    @patch("kalshi_betting.main.append_to_prod_log")
    @patch("kalshi_betting.main.execute_trades")
    @patch("kalshi_betting.main.pre_execution_check")
    @patch("kalshi_betting.main.select_portfolio")
    @patch("kalshi_betting.main.compute_trade")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_manual_review_result_returns_attention_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_compute,
        mock_select,
        mock_pre_exec,
        mock_execute,
        mock_append_log,
    ):
        # Two verify_auth calls: pre-trade balance, then post-trade balance
        # for the log's separator row.
        mock_verify_auth.side_effect = [
            {DEFAULT_EXCHANGE_INDEX: 100_000},
            {DEFAULT_EXCHANGE_INDEX: 100_000},
        ]
        mock_held.return_value = set()
        mock_fetch.return_value = []
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio: portfolio
        mock_execute.return_value = [
            TradeResult(spec=spec, status="manual_review", error="position lookup failed"),
        ]
        mock_append_log.return_value = "trade_log.xlsx"

        client = MagicMock()
        code = main._run_prod(client, _args(dry_run=False))

        assert code == EXIT_TRADES_NEED_ATTENTION
        assert code == 20

    @patch("kalshi_betting.main.append_to_prod_log")
    @patch("kalshi_betting.main.execute_trades")
    @patch("kalshi_betting.main.pre_execution_check")
    @patch("kalshi_betting.main.select_portfolio")
    @patch("kalshi_betting.main.compute_trade")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_clean_dry_run_returns_ok_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_compute,
        mock_select,
        mock_pre_exec,
        mock_execute,
        mock_append_log,
    ):
        mock_verify_auth.side_effect = [
            {DEFAULT_EXCHANGE_INDEX: 100_000},
            {DEFAULT_EXCHANGE_INDEX: 100_000},
        ]
        mock_held.return_value = set()
        mock_fetch.return_value = []
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio: portfolio
        mock_execute.return_value = [TradeResult(spec=spec, status="simulated")]
        mock_append_log.return_value = "trade_log.xlsx"

        client = MagicMock()
        code = main._run_prod(client, _args(dry_run=True))

        assert code == EXIT_OK
        assert code == 0

    @patch("kalshi_betting.main.verify_auth")
    def test_no_qualifying_pairs_returns_ok_code(self, mock_verify_auth):
        # No-pairs / no-executable-trades paths must also resolve to EXIT_OK,
        # not just the low-balance and post-execution paths.
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: 100_000}
        with (
            patch("kalshi_betting.main.get_held_tickers", return_value=set()),
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
        ):
            code = main._run_prod(MagicMock(), _args())

        assert code == EXIT_OK


class TestRunDevExitCode:
    def test_run_dev_returns_ok_code(self):
        client = MagicMock()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK


class TestMainEntryPoint:
    @patch("kalshi_betting.main.write_dev_simulation")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.build_client")
    def test_main_dev_mode_exits_ok(
        self,
        mock_build_client,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_write_sim,
        tmp_path,
        monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_fetch.return_value = []
        mock_filter_horizon.side_effect = lambda m, d: m
        mock_find_ts.return_value = []
        mock_find_st.return_value = []
        mock_enrich.return_value = []
        mock_write_sim.return_value = "dev_sim.xlsx"

        # main() configures logging with a FileHandler under PROJECT_ROOT —
        # point that at tmp_path so this test never touches the real
        # repo-root kalshi_arb.log.
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_OK

    @patch("kalshi_betting.main.verify_auth")
    @patch("kalshi_betting.main.build_client")
    def test_main_prod_mode_low_balance_exits_skip_code(
        self, mock_build_client, mock_verify_auth, tmp_path, monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1}

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_SKIPPED_LOW_BALANCE
        assert exc_info.value.code == 10


def _fake_write_dev_simulation(results, candidate_pairs, balance_cents):
    """Stand-in for reporter.write_dev_simulation() that reproduces its one
    real log line (reporter.py:544, "Dev simulation written: %s") so the
    BS-26 tests below can assert main._run_dev's caller side does not log a
    duplicate of it on any exit path.
    """
    logging.info("Dev simulation written: %s", "dev_sim.xlsx")
    return "dev_sim.xlsx"


class TestDevSimulationLoggedOnce:
    """BS-26: write_dev_simulation() already logs "Dev simulation written: %s"
    itself — main._run_dev must not repeat that line (with or without an
    "(empty)"/"(candidates only)" qualifier) on any of its three exit paths.
    """

    def test_empty_candidate_pairs_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1

    def test_empty_portfolio_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        pair = make_spec().pair
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[pair]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[pair]),
            patch("kalshi_betting.main.compute_trade", return_value=None),
            patch("kalshi_betting.main.select_portfolio", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1

    def test_full_run_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        spec = make_spec()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[spec.pair]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[spec.pair]),
            patch("kalshi_betting.main.compute_trade", return_value=spec),
            patch("kalshi_betting.main.select_portfolio", return_value=[spec]),
            patch(
                "kalshi_betting.main.execute_trades",
                return_value=[TradeResult(spec=spec, status="simulated")],
            ),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1


class TestLoggingRotation:
    """BS-25: kalshi_arb.log must rotate (5MB x 3 backups) instead of growing
    unbounded — a scheduler daemon re-runs this process weekly forever.

    Targets _setup_logging() directly rather than main(): logging.basicConfig()
    is a no-op once the root logger already has handlers (pytest installs its
    own), so the root logger is cleared first to actually exercise the handler
    configuration.
    """

    def test_setup_logging_file_handler_rotates(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            main._setup_logging(tmp_path / "kalshi_arb.log")

            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
            handler = file_handlers[0]
            assert handler.maxBytes == 5 * 1024 * 1024
            assert handler.backupCount == 3
            # A plain FileHandler would satisfy isinstance(h, FileHandler) too,
            # so pin the concrete type — that is the whole point of BS-25
            assert type(handler) is logging.handlers.RotatingFileHandler
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level

    def test_main_installs_the_rotating_handler(self, tmp_path, monkeypatch):
        # End-to-end: main() must route through _setup_logging, so the
        # rotating handler is what a real run actually gets.
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        for h in saved_handlers:
            root.removeHandler(h)

        try:
            monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
            monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev"])
            with (
                patch("kalshi_betting.main.build_client", return_value=MagicMock()),
                patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
                patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
                patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
                patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
                patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
                patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
            ):
                with pytest.raises(SystemExit):
                    main.main()

            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
        finally:
            # Close whatever main() attached so tmp_path teardown isn't blocked
            # by an open file handle on any platform, then restore the root
            # logger exactly as pytest had it configured.
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)


class TestDryRunInertInDev:
    """BS-32: --dry-run has no effect in dev mode (dev always simulates), so
    main() logs a warning naming that rather than leaving it silently
    ignored.
    """

    def test_dev_dry_run_logs_inert_warning(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev", "--dry-run"])

        with (
            patch("kalshi_betting.main.build_client", return_value=MagicMock()),
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

        assert exc_info.value.code == EXIT_OK
        assert "--dry-run is inert in dev mode" in caplog.text

    def test_prod_dry_run_does_not_log_inert_warning(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod", "--dry-run"])

        with (
            patch("kalshi_betting.main.build_client", return_value=MagicMock()),
            patch(
                "kalshi_betting.main.verify_auth",
                return_value={DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1},
            ),
        ):
            with pytest.raises(SystemExit):
                main.main()

        assert "--dry-run is inert in dev mode" not in caplog.text


def test_exit_code_constants_distinct():
    # Guard against a future accidental collision between the three codes —
    # the scheduler's log-level mapping depends on them being distinguishable.
    assert len({EXIT_OK, EXIT_SKIPPED_LOW_BALANCE, EXIT_TRADES_NEED_ATTENTION}) == 3
