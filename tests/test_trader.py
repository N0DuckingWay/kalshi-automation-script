"""Tests for trader.py: V2 order bodies and price/tick math, the rollback and
its loss floor, how uncertain outcomes are judged, the collateral transfers,
and the write pacer. All Kalshi calls are mocked, so the tests run offline.

Legs are named by the order they are sent: the NO leg goes first and is the
one the rollback undoes, the YES leg second. For same_title pairs (make_spec's
default) the NO leg is market_a (TICK-A); for time_series pairs it is
market_b (TICK-B), covered by TestTimeSeriesLegOrder.

_execute_one reads BOTH legs' positions before sending anything (the NO
leg's ticker first) and judges an exception by how a position changed
afterwards. Mocks therefore script get_positions replies in order with
side_effect (see positions_seq); every _execute_one call uses two baseline
reads first.

Every order goes through signed_request_json, which the tests mock.
TestV2IsTheOnlyOrderPath checks, on the syntax tree, that nothing reaches the
SDK's create-order methods, that only config.py reads ORDER_API_VERSION, and
that no code string starts with the retired /portfolio/orders path.

The write pacer (trader._WritePacer) is covered from `class _FakeClock` to
the end. Single-thread tests use _FakeClock; multi-thread tests use _SimTime,
simulated time that only moves once every thread is blocked, so their times
are exact on any machine.
"""
import ast
import copy
import dataclasses
import inspect
import json
import logging
import math
import pickle
import random
import textwrap
import threading
import time
import uuid
from decimal import Decimal
from json import JSONDecodeError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import _http, config, trader
from kalshi_betting.config import (
    BUY_SLIPPAGE_TICKS,
    DEFAULT_EXCHANGE_INDEX,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT,
    TRANSFER_PATH,
    V2_ORDER_PATH,
)
from kalshi_betting.reporter import TradeResult
from kalshi_betting.scanner import PriceRange
from kalshi_betting.trader import (
    _await_transfer_settlement,
    _build_no_order_v2,
    _build_rollback_order_v2,
    _build_yes_order_v2,
    _ceil_to_tick,
    _cents_to_centicents,
    _execute_one,
    _execute_transfer,
    _format_count,
    _format_price,
    _is_fok_kill,
    _ordered_legs,
    _partition_by_funding,
    _plan_transfers,
    _position_count,
    _required_cents_by_shard,
    _rollback_floor_cents,
    _submit_order_v2,
    _transfers_active,
    _unfunded_shards,
    _v2_fill_status,
    _v2_limit_price,
    _v2_rollback_price,
    _v2_top_of_grid_price,
    _WritePacer,
    ensure_shard_collateral,
    execute_trades,
    pre_execution_check,
)

# Tick grids used by the V2 price-math tests, mirroring the regimes named by
# live `price_level_structure` values (see scanner.tick_size_for_price).
DECI_CENT_BANDS = [PriceRange(start=0.0, end=1.0, step=0.001)]
CENTER_DECI_EDGE_CENTI_BANDS = [
    PriceRange(start=0.0, end=0.01, step=0.0001),
    PriceRange(start=0.01, end=0.99, step=0.001),
    PriceRange(start=0.99, end=1.0, step=0.0001),
]


def _calls_retry_wrapper(fn) -> bool:
    """True when fn's body actually CALLS api_call_with_retry.

    Parses the AST rather than grepping the source text: several of these
    functions explain the no-retry rule in their own docstrings, so a
    substring search reports every one of them as a violation. Only a real
    call node counts.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "api_call_with_retry"
        for node in ast.walk(tree)
    )


class _StatusError(Exception):
    """Minimal stand-in for an SDK exception carrying an HTTP status.

    Mirrors tests/test_http.py's helper — api_call_with_retry classifies by
    the .status attribute, not by exception type.
    """

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def make_market(structure: str = "", ranges: list | None = None) -> SimpleNamespace:
    """Market stand-in exposing only the two fields tick_size_for_price reads."""
    return SimpleNamespace(price_level_structure=structure, price_ranges=ranges)


def make_spec(
    x: int = 5,
    nA: float = 0.40,
    pB: float = 0.35,
    pair_type: str = "same_title",
    pA: float = 0.60,
    nB: float = 0.65,
    structure: str = "",
    ranges: list | None = None,
    shard_a: int = 0,
    shard_b: int = 0,
    cost_a: float = 0.0,
    cost_b: float = 0.0,
    title: str = "test pair",
) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads.

    `pair_type` decides the leg order (see _ordered_legs): the default
    "same_title" makes market_a/TICK-A the NO leg priced at `nA` and
    market_b/TICK-B the YES leg at `pB` — today's wire behaviour — while
    "time_series" makes TICK-B the NO leg at `nB` and TICK-A the YES leg at
    `pA`. All four prices are pinned as REAL floats: scanner.leg_prices reads
    `nB` directly, and a MagicMock auto-attribute there would TypeError in
    the price math.

    Both markets carry explicit tick-structure attributes because the V2 order
    builders price against the market's own grid via scanner.tick_size_for_price;
    the defaults ("" / None) mean "unknown", i.e. the $0.01 fallback grid.

    `shard_a`/`shard_b` pin REAL ints on each leg's exchange_index. This must
    never be left to MagicMock's auto-attributes: an auto-attr is a truthy Mock
    that compares unequal to DEFAULT_EXCHANGE_INDEX (so every spec would look
    non-routable) and is not JSON-serializable in a V2 order body.

    `cost_a`/`cost_b` are the per-leg fee-inclusive DOLLAR costs the collateral
    planner sizes transfers from, pinned as REAL floats for the same reason —
    an auto-attr Mock would blow up (or silently mis-size) the ceil-to-cents
    conversion in _required_cents_by_shard.
    """
    pair = MagicMock()
    pair.market_a.ticker = "TICK-A"
    pair.market_a.title = "Market A"
    pair.market_a.price_level_structure = structure
    pair.market_a.price_ranges = ranges
    pair.market_a.exchange_index = shard_a
    pair.market_b.ticker = "TICK-B"
    pair.market_b.title = "Market B"
    pair.market_b.price_level_structure = structure
    pair.market_b.price_ranges = ranges
    pair.market_b.exchange_index = shard_b
    pair.pair_type = pair_type
    pair.pA = pA
    pair.pB = pB
    pair.nA = nA
    pair.nB = nB
    pair.canonical_title = title
    spec = MagicMock()
    spec.pair = pair
    spec.x = x
    spec.y = x
    spec.cost_with_fees_a = cost_a
    spec.cost_with_fees_b = cost_b
    return spec


def _no_leg(spec) -> trader._Leg:
    """The first-submitted NO leg of a spec, exactly as _execute_one resolves it."""
    return _ordered_legs(spec)[0]


def _yes_leg(spec) -> trader._Leg:
    """The second-submitted YES leg of a spec, exactly as _execute_one resolves it."""
    return _ordered_legs(spec)[1]


def shard_status(transfers_active: bool = True) -> dict:
    """One parsed scanner.fetch_shard_statuses() entry."""
    return {
        "trading_active": True,
        "exchange_active": True,
        "intra_exchange_transfers_active": transfers_active,
        "description": "",
    }


def transfer_resp(transfer_id: str = "tr_abc123") -> dict:
    """Parsed POST /portfolio/intra_exchange_instance_transfer body.

    The SDK models no such route, so trader submits it through
    _http.signed_request_json, whose return value is the ALREADY-PARSED JSON
    body — mocks of that helper therefore return a plain dict, not a raw
    RESTResponse stand-in.
    """
    return {"transfer_id": transfer_id}


def v2_resp(fill_count, requested: int = 5) -> dict:
    """Parsed V2 create-order response body (signed_request_json's return)."""
    return {
        "order": {
            "order_id": "ord-1",
            "client_order_id": "cid-1",
            "fill_count": fill_count,
            "remaining_count": requested - fill_count,
            "ts_ms": 1_700_000_000_000,
        }
    }


# The body the V2 endpoint sent, verbatim, when it killed a fill-or-kill ask
# that could not fill on the production API (2026-09-28).
FOK_KILL_BODY = (
    '{"error":{"code":"fill_or_kill_insufficient_resting_volume",'
    '"message":"fill or kill insufficient resting volume"}}'
)


def fok_kill_error() -> ApiException:
    """The HTTP 409 the V2 endpoint answers a fill-or-kill that cannot fill."""
    return ApiException(status=409, reason="Conflict", body=FOK_KILL_BODY)


def positions_resp(ticker: str | None = None, position: float = 0) -> SimpleNamespace:
    """Raw get_positions response with zero or one market position.

    _position_count parses the raw JSON body (the SDK's MarketPosition model
    can't deserialize live responses anymore), with the count in the
    position_fp string field — mocks mirror that wire format.
    """
    mps = [] if ticker is None else [{"ticker": ticker, "position_fp": str(position)}]
    payload = {"market_positions": mps, "cursor": None}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def positions_seq(*readings) -> MagicMock:
    """Mock get_positions that answers successive calls from a script.

    Each element is either a (ticker, position) tuple, None for "no position
    on file", or an Exception instance to raise for that call. _execute_one
    reads BOTH baselines up front — the NO leg's ticker first, then the YES
    leg's, before either order is submitted, so no blocking call sits in the
    unhedged window between the NO leg's fill and the YES leg's submission —
    and then once (or twice) more after an ambiguous leg, so the script is
    consumed in that order:
        before_no, before_yes, [backstop], [after_no], [no lag re-read],
        [after_yes], [yes lag re-read]
    (for the same-title default that is TICK-A, TICK-B, ...; for a
    time_series spec it is TICK-B, TICK-A, ...).

    The optional [backstop] slot is the V2 NO-mapping check's own single-shot
    read (see TestV2NoMappingBackstop); it hits the same client method, so it
    consumes a script entry like any other, but only on the V2 path and only
    while the mapping is unlatched.

    The optional lag re-read slots (DR-63/DR-64) are consumed only when the
    corresponding leg's first post-failure delta is ZERO: _execute_one then
    re-reads that leg's ticker once, after a pause, before concluding
    "confirmed non-fill". An ambiguous-leg test whose first reading is
    unmoved therefore needs one MORE script entry than it did before those
    findings; under-providing one raises StopIteration rather than repeating
    the last reading.

    A flat return_value cannot express this: before and after would be equal,
    which is precisely the delta-0 "confirmed non-fill" case.
    """
    effects = []
    for reading in readings:
        if isinstance(reading, BaseException):
            effects.append(reading)
        elif reading is None:
            effects.append(positions_resp())
        else:
            effects.append(positions_resp(*reading))
    return MagicMock(side_effect=effects)


class TestRollbackPriceFloor:
    """_rollback_floor_cents: the lowest NO price per contract the unwind may
    take, from the NO leg's own entry price (`nA` here, since market_a is the
    NO leg). The unwind bid is capped at 1 - floor/100 (see
    TestV2OrderBuilders)."""

    def test_floor_is_entry_less_max_loss(self):
        spec = make_spec(nA=0.62)
        assert _rollback_floor_cents(_no_leg(spec)) == 62 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
        # Literal guard: at the current calibration (BS-05, 12 cents to cover
        # the full bid-ask spread plus adverse movement) this must be 50 cents
        # exactly, so a silent change to the constant fails this test even
        # though the assertion above would float along with it.
        assert ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT == 12
        assert _rollback_floor_cents(_no_leg(spec)) == 50

    def test_floor_rounds_before_truncating(self):
        # 0.57 is stored as 0.5699999999999998, so 0.57 * 100 == 56.99999999999999.
        # int() alone would truncate the entry to 56 and floor a cent too low.
        spec = make_spec(nA=0.57)
        assert _rollback_floor_cents(_no_leg(spec)) == 57 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT

    def test_floor_clamped_to_valid_limit_price(self):
        # 3 - 5 would be a negative limit price the API rejects outright.
        assert _rollback_floor_cents(_no_leg(make_spec(nA=0.03))) == 1
        # And the upper clamp keeps the price inside the API's 1..99 range.
        assert _rollback_floor_cents(_no_leg(make_spec(nA=1.20))) == 99


@pytest.fixture(autouse=True)
def _reset_v2_mapping_latch(monkeypatch):
    """Start every test from a fresh process's unlatched NO-mapping state.

    trader._V2_NO_MAPPING_CONFIRMED is a PROCESS-lifetime latch that real
    execution flips, so without this a single test that confirms the mapping
    would silently disable the backstop for every test that runs after it.
    monkeypatch restores the pre-test value at teardown, so the latch can never
    leak across tests in either direction. Its twin, the disproven latch
    (trader._V2_NO_MAPPING_DISPROVEN), is cleared the same way for the whole
    suite by conftest.py.
    """
    monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", False)


@pytest.fixture
def v2_mapping_confirmed(monkeypatch):
    """Pretend the V2 NO-leg mapping has already been confirmed this process.

    _execute_one()'s backstop reads the account position after the first V2
    NO-leg fill (see TestV2NoMappingBackstop). Classes exercising the rollback /
    disambiguation state machine on the V2 wire format aren't testing that
    check and must not have their position mocks consumed by it, so they start
    from the latched state a second trade would see.
    """
    monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", True)


def assert_disproof_names_the_remedy(caplog) -> None:
    """Assert the disproven-mapping CRITICAL says to stop trading and flatten
    by hand in the Kalshi UI, and names no other order path."""
    criticals = [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(criticals) == 1, criticals
    message = criticals[0]
    assert "MAPPING DISPROVEN" in message
    assert "Stop trading" in message and "Kalshi UI" in message
    assert "no other order path to fall back on" in message
    assert "legacy" not in message.lower()
    assert "ORDER_API_VERSION" not in message


class TestRollbackVerification:
    """_execute_one around the unwind: the capped unwind's own result is
    checked, a clean fill-or-kill rejection needs no position read, and both
    baselines are read before any order is sent. The NO-mapping check is
    pre-latched so it does not use up the position scripts."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """The V2 NO-leg mapping is already confirmed, as on a second trade."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_floored_rollback_killed_by_price_reports_rollback_failed(self, post):
        # The unwind finds nothing at or under its cap: "rollback_failed",
        # and the unwind sent carried the cap (1 - (62c entry - max loss))
        post.side_effect = [
            v2_resp(5),   # NO leg
            v2_resp(0),   # YES leg rejected
            v2_resp(0),   # floored unwind finds nothing at or under its cap
        ]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec(nA=0.62))
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error
        rollback_body = post.call_args_list[2].kwargs["body"]
        assert rollback_body["price"] == _format_price(
            Decimal("1") - Decimal(62 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT) / Decimal("100")
        )
        assert rollback_body["reduce_only"] is True
        assert rollback_body["time_in_force"] == "immediate_or_cancel"

    def test_leg_a_fok_rejection_is_failed_without_position_check(self, post):
        # A 2xx with nothing filled is a non-fill: no extra read, no
        # rollback, no YES leg
        post.side_effect = [v2_resp(0)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert "NO leg FoK not filled" in result.error
        assert post.call_count == 1
        # Only the two up-front baselines were read — no ambiguity snapshot
        assert client.get_positions_without_preload_content.call_count == 2

    def test_both_baselines_are_read_before_any_order_is_submitted(self, post):
        # Both baselines are read before the NO leg is sent, so no retried
        # read sits between the NO fill and the YES order
        calls: list[str] = []

        def record_positions(*args, **kwargs):
            calls.append("positions")
            return positions_resp()

        def record_order(*args, **kwargs):
            calls.append("order")
            return v2_resp(5)

        post.side_effect = record_order
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=record_positions)

        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert calls == ["positions", "positions", "order", "order"]


class TestNoLegExceptionDisambiguation:
    """The NO leg (TICK-A) raised: the outcome follows the position change,
    an existing holding is never read as this order's fill, and an
    unexplained change is never traded against."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """The V2 NO-leg mapping is already confirmed, as on a second trade."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_external_no_position_unchanged_is_failed_not_unwound(self, post, monkeypatch):
        # BS-01: the account already holds 10 NO on TICK-A and the order did
        # not fill. The change is 0 on both reads (DR-64), so "failed" with
        # no unwind
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = TimeoutError("timeout")
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -10),   # after_no — unmoved
            ("TICK-A", -10),   # lag re-read — still unmoved
        )
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert post.call_count == 1

    def test_delta_of_our_no_buy_is_unwound(self, post):
        # Exception but the position moved by exactly -spec.x (timeout AFTER
        # the fill) — the half-filled pair must be unwound, not abandoned.
        # The account also held 10 unrelated NO contracts, which the delta
        # correctly ignores.
        post.side_effect = [
            TimeoutError("timeout"),  # NO leg raises after actually filling
            v2_resp(5),               # rollback fills
        ]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -15),   # after_no — moved by -5 == -spec.x
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        # Exactly one submission attempt for the NO leg, plus the rollback
        assert post.call_count == 2
        assert post.call_args_list[1].kwargs["body"]["reduce_only"] is True

    def test_unattributable_delta_is_manual_review(self, post):
        # An unexplained change: manual_review, no unwind
        post.side_effect = TimeoutError("timeout")
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", 0),     # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -3),    # after_no — -3, but spec.x is 7
        )
        result = _execute_one(client, make_spec(x=7))
        assert result.status == "manual_review"
        assert "delta=-3" in result.error
        # No unwind order was submitted
        assert post.call_count == 1

    def test_snapshot_failure_is_manual_review(self, post):
        # The read failed, so the state is unknown: no unwind. The call count
        # also checks the read runs outside the except block, where the
        # TimeoutError context would make the retry wrapper retry it
        post.side_effect = TimeoutError("timeout")
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None,                            # before_no
            None,                            # before_yes (taken up front)
            RuntimeError("lookup failed"),   # after_no — non-retryable → fail fast
        )
        with patch.object(_http.time, "sleep") as sleep:
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "delta=None" in result.error
        assert post.call_count == 1
        assert client.get_positions_without_preload_content.call_count == 3
        sleep.assert_not_called()


class TestYesLegExceptionDisambiguation:
    """The YES leg (TICK-B) raised: judged by the position change, and never
    unwound on a change the order cannot explain."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """The V2 NO-leg mapping is already confirmed, as on a second trade, so
        the NO leg's fill costs no extra position read."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_external_yes_position_unchanged_rolls_back(self, post, monkeypatch):
        # BS-01: the account already holds 5 YES on TICK-B and the YES leg did
        # not fill. The change is 0 on both reads (DR-63), so the NO leg is
        # rolled back
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = [
            v2_resp(5),               # NO leg
            TimeoutError("timeout"),  # YES leg raises, truly unfilled
            v2_resp(5),               # rollback fills
        ]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_no
            ("TICK-B", 5),     # before_yes — pre-existing external position
            ("TICK-B", 5),     # after_yes — unmoved
            ("TICK-B", 5),     # lag re-read — still unmoved
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        assert post.call_count == 3

    def test_no_position_at_all_rolls_back(self, post, monkeypatch):
        # The ticker is absent on every read, re-read included: a non-fill,
        # so the NO leg is unwound
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = [
            v2_resp(5),               # NO leg
            TimeoutError("timeout"),  # YES leg raises, truly unfilled
            v2_resp(5),               # rollback fills
        ]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_unexpected_delta_is_manual_review_without_rollback(self, post):
        # The position moved by +2 but we ordered 7 — unattributable. Never
        # auto-rollback on an outcome we cannot explain.
        post.side_effect = [
            v2_resp(7, 7),            # NO leg
            TimeoutError("timeout"),  # YES leg raises
        ]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None,
            ("TICK-B", 0),
            ("TICK-B", 2),     # +2, but spec.y is 7
        )
        result = _execute_one(client, make_spec(x=7))
        assert result.status == "manual_review"
        assert "delta=2" in result.error
        # No third (rollback) order was submitted
        assert post.call_count == 2


class TestLedgerLagOnAmbiguousLegs:
    """DR-63/DR-64: a position delta of ZERO is not on its own a confirmed
    non-fill.

    A transport error — urllib3.ProtocolError / ConnectionError /
    ReadTimeoutError, the classes _http._TRANSIENT_NETWORK_ERRORS lists as
    observed live — can be raised milliseconds AFTER the exchange processed and
    FILLED the order, and Kalshi's positions ledger is read-after-write lagged.
    A single post-failure read therefore returns the PRE-fill value, and the
    bot concluded "no fill" on a leg that had filled:

      * YES leg (DR-63): it submitted the reduce-only unwind of a NO leg that
        was in truth hedging a real YES fill — selling the hedge, leaving a
        full-size naked YES position open, and reporting "rolled_back", which
        means flat.
      * NO leg (DR-64): it returned "failed" ("nothing to unwind") with a
        full-size UNHEDGED NO position open. "failed" is not in the set
        main._run_prod maps to EXIT_TRADES_NEED_ATTENTION, so the run exited 0
        and the Excel row read as a pair that never traded.

    Each ambiguous branch now re-reads its own ticker ONCE after
    trader._V2_MAPPING_RECHECK_DELAY_SECONDS and judges the re-read — the same
    reasoning _confirm_v2_no_mapping applies (DR-21) and v2_probe applies twice
    over (DR-60, DR-20). It does NOT touch the absolute-vs-delta rule, which
    settles which QUANTITY is evidence, not how many times it is read.

    The NO-leg re-read must be SINGLE-SHOT: it sits in the unhedged window (the
    YES leg has not been submitted), where api_call_with_retry's ~62s of
    backoff is the worse outcome. The YES-leg re-read is the ordinary retried
    read, matching the first read beside it.
    """

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """The NO-mapping backstop already latched, so it cannot consume these
        cases' position scripts."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    @pytest.fixture
    def slept(self, monkeypatch):
        """Record (and skip) the ledger-lag pause."""
        recorded: list[float] = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: recorded.append(s))
        return recorded

    @pytest.fixture
    def readers(self, monkeypatch):
        """Record which reader each position lookup went through.

        Both readers hit the same client method, so a call count alone cannot
        tell them apart; this wraps the module-level names _execute_one
        resolves at call time and delegates to the real implementations, so
        retry policy is unchanged and only the routing is observed.
        """
        seen: list[tuple[str, str]] = []
        real_once = trader._position_count_once
        real_retried = trader._position_count

        def once(client, ticker):
            seen.append(("once", ticker))
            return real_once(client, ticker)

        def retried(client, ticker):
            seen.append(("retried", ticker))
            return real_retried(client, ticker)

        monkeypatch.setattr(trader, "_position_count_once", once)
        monkeypatch.setattr(trader, "_position_count", retried)
        return seen

    def test_yes_leg_lagging_ledger_is_not_rolled_back(self, post, slept):
        # DR-63: the YES POST reached the exchange and FILLED, then the client
        # saw a transport error. The ledger lags on the first read and catches
        # up on the second, so the pair is complete and the hedge must stay.
        post.side_effect = [v2_resp(5), ConnectionError("Connection broken")]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,              # before_no baseline
            None,              # before_yes baseline
            ("TICK-B", 0),     # after_yes — ledger has not caught up yet
            ("TICK-B", 5),     # lag re-read — the fill is visible: delta +5
        ))
        result = _execute_one(client, make_spec(x=5))
        assert result.status != "rolled_back"
        assert result.status == "executed"
        # Exactly the two leg submissions — NO unwind of a live hedge
        assert post.call_count == 2
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert client.get_positions_without_preload_content.call_count == 4

    def test_yes_leg_genuinely_unfilled_still_rolls_back(self, post, slept):
        # Control: the ledger really is unmoved on BOTH readings, so today's
        # rollback behaviour is unchanged.
        post.side_effect = [v2_resp(5), ConnectionError("Connection broken"),
                            v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-B", 0), ("TICK-B", 0),
        ))
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        # NO leg, YES leg, then the unwind
        assert post.call_count == 3
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    def test_yes_leg_reread_uses_the_retried_reader(self, post, slept, readers):
        # The hedge is already in place here and the first read beside it is
        # retried, so a transient 429 on the re-read must not escalate a
        # recoverable ambiguity into manual_review.
        post.side_effect = [v2_resp(5), ConnectionError("Connection broken")]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-B", 0), ("TICK-B", 5),
        ))
        assert _execute_one(client, make_spec(x=5)).status == "executed"
        assert readers == [
            ("retried", "TICK-A"),   # before_no baseline
            ("retried", "TICK-B"),   # before_yes baseline
            ("retried", "TICK-B"),   # after_yes
            ("retried", "TICK-B"),   # the lag re-read
        ]

    def test_yes_leg_failed_reread_is_manual_review_not_a_rollback(
        self, post, slept
    ):
        # An unreadable re-read is the existing unattributable case: no unwind
        # is submitted, because the YES leg may really have filled.
        post.side_effect = [v2_resp(5), ConnectionError("Connection broken")]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-B", 0), RuntimeError("positions endpoint down"),
        ))
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "manual_review"
        assert post.call_count == 2

    def test_no_leg_lagging_ledger_is_unwound_not_reported_failed(
        self, post, slept
    ):
        # DR-64: the NO POST reached the exchange and FILLED, then the client
        # saw a transport error. The ledger lags on the first read; the re-read
        # shows our -5, so the now-unhedged leg is unwound instead of being
        # abandoned under a clean "failed".
        post.side_effect = [ConnectionError("Connection broken"), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,              # before_no baseline
            None,              # before_yes baseline
            ("TICK-A", 0),     # after_no — ledger has not caught up yet
            ("TICK-A", -5),    # lag re-read — our NO buy is visible
        ))
        result = _execute_one(client, make_spec(x=5))
        assert result.status != "failed"
        assert result.status == "rolled_back"
        # The NO leg's (raising) submission, then the unwind — the YES leg is
        # never sent, exactly as on the pre-existing delta=-count path
        assert post.call_count == 2
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    def test_no_leg_genuinely_unfilled_still_fails(self, post, slept):
        # Control: unmoved on BOTH readings, so today's "failed" is unchanged
        # and nothing is submitted after the raising leg.
        post.side_effect = [ConnectionError("Connection broken")]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", 0),
        ))
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "failed"
        assert "NO leg error" in result.error
        assert post.call_count == 1
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    def test_no_leg_reread_uses_the_single_shot_reader(self, post, slept, readers):
        # Load-bearing: this read sits in the unhedged window — the YES leg has
        # not been submitted, so if the NO leg filled the account is one-sided
        # while we wait, and api_call_with_retry can hold one call for ~62s.
        post.side_effect = [ConnectionError("Connection broken"), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", -5),
        ))
        assert _execute_one(client, make_spec(x=5)).status == "rolled_back"
        assert readers == [
            ("retried", "TICK-A"),   # before_no baseline
            ("retried", "TICK-B"),   # before_yes baseline
            ("retried", "TICK-A"),   # after_no (unchanged, still retried)
            ("once", "TICK-A"),      # the lag re-read — SINGLE-SHOT
        ]
        # And specifically: the re-read did not go through the retried reader
        assert ("retried", "TICK-A") not in readers[3:]

    def test_no_leg_reread_carries_no_backoff(self, post, slept):
        # Behavioural counterpart to the routing assertion above: a 429 on the
        # re-read costs exactly ONE request and no sleeps from the retry
        # wrapper, rather than up to six with ~62s of backoff between them.
        #
        # trader.time, _http.time and time are the same module object, so the
        # `slept` fixture records BOTH the ledger-lag pause and anything
        # api_call_with_retry would sleep — the assertion below is that the
        # pause is the only one.
        post.side_effect = [ConnectionError("Connection broken")]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), _StatusError(429),
        ))
        result = _execute_one(client, make_spec(x=5))
        # Unreadable re-read -> unknown delta -> manual_review, no order sent
        assert result.status == "manual_review"
        assert "delta=None" in result.error
        assert post.call_count == 1
        assert client.get_positions_without_preload_content.call_count == 4
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    def test_a_moved_ledger_is_judged_without_a_re_read(self, post, slept):
        # The re-read is triggered by a ZERO delta only: a first read that
        # already answers the question must not pay the pause.
        post.side_effect = [ConnectionError("Connection broken"), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", -5),
        ))
        assert _execute_one(client, make_spec(x=5)).status == "rolled_back"
        assert slept == []
        assert client.get_positions_without_preload_content.call_count == 3


class TestPositionCountRetry:
    """BS-04: the position read is a read-only GET, so it retries."""

    def test_retries_429_then_succeeds(self):
        # A transient rate-limit on the position lookup must not read as
        # "state unknown" — that would escalate an ambiguous order into a
        # rollback or manual_review for no reason.
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=[
            _StatusError(429),
            positions_resp("TICK-A", position=-4),
        ])
        with patch.object(_http.time, "sleep"):
            assert _position_count(client, "TICK-A") == -4
        assert client.get_positions_without_preload_content.call_count == 2

    def test_non_retryable_failure_returns_none(self):
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("boom")
        )
        with patch.object(_http.time, "sleep") as sleep:
            assert _position_count(client, "TICK-A") is None
        sleep.assert_not_called()
        assert client.get_positions_without_preload_content.call_count == 1

    def test_missing_ticker_is_confirmed_zero(self):
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp()
        )
        assert _position_count(client, "TICK-A") == 0


class TestCentsToCenticents:
    """The transfer endpoint's `amount` is CENTICENTS (1/100 of a cent) — the
    codebase's third money unit. The conversion must exist exactly once, named,
    so no call site ever inlines a bare factor."""

    def test_cents_convert_to_centicents(self):
        # $1.14 = 114 cents = 11,400 centicents
        assert _cents_to_centicents(114) == 11_400

    def test_zero_is_zero(self):
        assert _cents_to_centicents(0) == 0


class TestPlanTransfers:
    """Pure planner: deficits filled greedily from the largest remaining
    surplus, deterministically ordered, partial when surplus runs out."""

    def test_no_deficit_plans_nothing(self):
        assert _plan_transfers({0: 500, 1: 200}, {0: 1000, 1: 1000}) == []

    def test_single_deficit_from_single_surplus(self):
        # Shard 0 is 400 short; shard 1 has 900 spare.
        assert _plan_transfers({0: 500, 1: 100}, {0: 100, 1: 1000}) == [(1, 0, 400)]

    def test_deficit_drawn_from_largest_surplus_first(self):
        # Surpluses: shard 1 = 300, shard 2 = 900. The 500 deficit must come
        # entirely out of shard 2 (one transfer beats two — each POST is a
        # non-idempotent money movement).
        plan = _plan_transfers({0: 500}, {0: 0, 1: 300, 2: 900})
        assert plan == [(2, 0, 500)]

    def test_multiple_sources_for_one_deficit(self):
        # 1000 needed, no single surplus covers it: 600 then 400, largest first.
        plan = _plan_transfers({0: 1000}, {0: 0, 1: 400, 2: 600})
        assert plan == [(2, 0, 600), (1, 0, 400)]

    def test_insufficient_total_surplus_plans_what_is_coverable(self):
        # Only 250 exists to move against a 1000 deficit — the planner moves it
        # anyway and leaves the shard short; the caller detects that from the
        # post-transfer balances and drops only the affected trades.
        plan = _plan_transfers({0: 1000}, {0: 0, 1: 250})
        assert plan == [(1, 0, 250)]
        assert sum(cents for _, _, cents in plan) == 250

    def test_multiple_deficits_processed_in_shard_order(self):
        plan = _plan_transfers({1: 400, 2: 300}, {1: 0, 2: 0, 5: 1000})
        assert plan == [(5, 1, 400), (5, 2, 300)]

    def test_plan_is_deterministic(self):
        required = {0: 900, 3: 500}
        available = {0: 100, 1: 700, 2: 700, 3: 0, 4: 400}
        first = _plan_transfers(required, available)
        assert all(_plan_transfers(required, available) == first for _ in range(5))

    def test_missing_shard_in_available_counts_as_zero(self):
        assert _plan_transfers({7: 300}, {0: 1000}) == [(0, 7, 300)]


class TestRequiredCentsByShard:
    def test_same_shard_legs_sum_onto_one_shard(self):
        spec = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)
        assert _required_cents_by_shard([spec]) == {0: 1500}

    def test_cross_shard_spec_splits_requirement_across_both_shards(self):
        # Each leg draws collateral from its OWN market's shard.
        spec = make_spec(shard_a=0, shard_b=1, cost_a=10.00, cost_b=5.00)
        assert _required_cents_by_shard([spec]) == {0: 1000, 1: 500}

    def test_partial_cents_round_up_never_down(self):
        # Flooring would under-fund the shard and get the order rejected for
        # insufficient collateral; the ceiling costs at most a spare cent.
        spec = make_spec(shard_a=0, shard_b=0, cost_a=1.001, cost_b=0.0)
        assert _required_cents_by_shard([spec]) == {0: 101}

    def test_float_noise_does_not_inflate_by_a_cent(self):
        # 0.07 * 100 == 7.000000000000001 in binary floating point; the round()
        # before the ceiling must keep this at 7 cents, not 8.
        spec = make_spec(shard_a=0, shard_b=0, cost_a=0.07, cost_b=0.0)
        assert _required_cents_by_shard([spec]) == {0: 7}

    def test_requirements_accumulate_across_specs(self):
        specs = [
            make_spec(shard_a=0, shard_b=0, cost_a=1.00, cost_b=2.00),
            make_spec(shard_a=0, shard_b=1, cost_a=3.00, cost_b=4.00),
        ]
        assert _required_cents_by_shard(specs) == {0: 600, 1: 400}


class TestUnfundedShardsAndPartitioning:
    """The two pure helpers that decide which trades survive a funding
    shortfall. A shard we could not observe holds nothing, and a spec dies if
    EITHER leg's shard is short."""

    def test_covered_requirement_leaves_nothing_unfunded(self):
        assert _unfunded_shards({0: 500, 1: 200}, {0: 500, 1: 999}) == set()

    def test_missing_shard_in_available_is_unfunded(self):
        # Absence is never read as "surely it's fine" — it is zero.
        assert _unfunded_shards({3: 1}, {0: 10_000}) == {3}

    def test_transfers_active_defaults_true_when_statuses_unavailable(self):
        # None = sandbox / pre-sharding shape: nothing to gate on.
        assert _transfers_active(None, 0) is True

    def test_transfers_inactive_flag_is_respected(self):
        statuses = {0: shard_status(True), 1: shard_status(False)}
        assert _transfers_active(statuses, 0) is True
        assert _transfers_active(statuses, 1) is False

    def test_drifted_false_transfers_flag_refuses_the_transfer(self):
        # scanner.fetch_shard_statuses normalises a re-typed "false" (and any
        # value it cannot read) to a real False before it ever reaches here;
        # this pins that the money gate then refuses to move, where the old
        # bool("false") would have returned True and POSTed the transfer.
        statuses = {0: shard_status(False)}
        assert _transfers_active(statuses, 0) is False

    def test_shard_absent_from_statuses_is_treated_as_inactive(self):
        # Refusing a shard the exchange never advertised costs at most a
        # dropped trade; attempting it moves money into an unmodelled state.
        assert _transfers_active({0: shard_status(True)}, 9) is False

    def test_partition_drops_a_spec_if_either_leg_shard_is_short(self):
        both_ok = make_spec(shard_a=0, shard_b=0, title="both ok")
        leg_b_bad = make_spec(shard_a=0, shard_b=1, title="leg b bad")
        leg_a_bad = make_spec(shard_a=1, shard_b=0, title="leg a bad")
        kept, dropped = _partition_by_funding([both_ok, leg_b_bad, leg_a_bad], {1})
        assert kept == [both_ok]
        assert dropped == [leg_b_bad, leg_a_bad]

    def test_partition_with_no_unfunded_shards_keeps_everything(self):
        specs = [make_spec(shard_a=0, shard_b=1), make_spec(shard_a=2, shard_b=2)]
        kept, dropped = _partition_by_funding(specs, set())
        assert kept == specs
        assert dropped == []


class TestExecuteTransfer:
    """One POST, verbatim body, centicent amount, never retried."""

    def test_body_and_path_are_exact(self, monkeypatch):
        post = MagicMock(return_value=transfer_resp("tr_1"))
        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock()
        assert _execute_transfer(client, 1, 0, 1400) == "tr_1"
        args, kwargs = post.call_args
        assert args[0] is client
        assert args[1] == "POST"
        assert args[2] == TRANSFER_PATH
        assert kwargs["body"] == {
            "source": "event_contract",
            "destination": "event_contract",
            # 1400 cents == 140,000 CENTICENTS — not 1400, not 14.00
            "amount": 140_000,
            "source_exchange_shard": 1,
            "destination_exchange_shard": 0,
        }

    def test_missing_transfer_id_returns_none(self, monkeypatch):
        # "Accepted but id-less" is in-flight, not failed — the caller must not
        # read None as "nothing moved".
        monkeypatch.setattr(trader, "signed_request_json", MagicMock(return_value={}))
        assert _execute_transfer(MagicMock(), 0, 1, 100) is None

    def test_api_exception_propagates_unretried(self, monkeypatch):
        post = MagicMock(side_effect=ApiException(status=500))
        monkeypatch.setattr(trader, "signed_request_json", post)
        with pytest.raises(ApiException):
            _execute_transfer(MagicMock(), 0, 1, 100)
        assert post.call_count == 1


class TestAwaitTransferSettlement:
    """Acceptance is not settlement: the poll re-reads the shard-aware balance
    until every requirement is covered or the bounded deadline passes."""

    @pytest.fixture(autouse=True)
    def _fast_poll(self, monkeypatch):
        """Compress the poll so the async-settlement contract is exercised for
        real (a real monotonic deadline, a real sleep) without the suite paying
        the production 30s bound."""
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0.05)

    def test_returns_as_soon_as_every_shard_is_covered(self, monkeypatch):
        reader = MagicMock(side_effect=[{0: 100}, {0: 1500}])
        monkeypatch.setattr(trader, "read_shard_balances", reader)
        assert _await_transfer_settlement(MagicMock(), {0: 1500}) == {0: 1500}
        assert reader.call_count == 2

    def test_timeout_returns_the_last_observed_balances(self, monkeypatch):
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0)
        reader = MagicMock(return_value={0: 100})
        monkeypatch.setattr(trader, "read_shard_balances", reader)
        assert _await_transfer_settlement(MagicMock(), {0: 1500}) == {0: 100}
        assert reader.call_count == 1

    def test_failed_balance_read_warns_and_observes_nothing(self, monkeypatch, caplog):
        # A read that raises must never be read as success — an unverifiable
        # balance is precisely what this function exists to refuse.
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0)
        monkeypatch.setattr(
            trader, "read_shard_balances", MagicMock(side_effect=RuntimeError("boom"))
        )
        with caplog.at_level(logging.WARNING, logger="root"):
            assert _await_transfer_settlement(MagicMock(), {0: 1500}) == {}
        assert any("Balance re-read failed" in r.getMessage() for r in caplog.records)


class TestEnsureShardCollateral:
    """Collateral must be on the shard an order settles against before that
    order is submitted. Every failure mode degrades to dropping the affected
    trades — never to submitting them underfunded, and never to a retry."""

    def _patch_io(self, monkeypatch, *, transfer=None, balances=None, settle_timeout=0.05):
        """Patch trader's two outbound calls and return the mocks.

        `transfer` is the signed_request_json stand-in (return_value or
        side_effect already configured); `balances` is read_shard_balances's. The
        settle poll is compressed to milliseconds so the async-settlement
        contract is exercised for real (a real deadline, a real sleep) without
        the suite paying the production 30s bound.
        """
        post = MagicMock(return_value=transfer_resp()) if transfer is None else transfer
        va = MagicMock(return_value={}) if balances is None else balances
        monkeypatch.setattr(trader, "signed_request_json", post)
        monkeypatch.setattr(trader, "read_shard_balances", va)
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", settle_timeout)
        return post, va

    def test_zero_deficit_is_a_no_op(self, monkeypatch):
        # The universal case today: everything is on shard 0 and shard 0 is
        # funded. No transfer, and no balance re-poll either.
        post, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(MagicMock(), portfolio, {0: 100_000}, None)
        assert result == portfolio
        post.assert_not_called()
        va.assert_not_called()

    def test_empty_portfolio_short_circuits(self, monkeypatch):
        post, va = self._patch_io(monkeypatch)
        assert ensure_shard_collateral(MagicMock(), [], {0: 100_000}, None) == []
        post.assert_not_called()
        va.assert_not_called()

    def test_dry_run_plans_but_never_posts(self, monkeypatch, caplog):
        post, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), portfolio, {0: 100, 1: 100_000}, None, dry_run=True
            )
        assert result == portfolio
        post.assert_not_called()
        va.assert_not_called()
        assert any("DRY RUN" in r.getMessage() for r in caplog.records)

    def test_dry_run_with_no_surplus_says_so_and_keeps_the_portfolio(
        self, monkeypatch, caplog
    ):
        post, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), portfolio, {0: 100}, None, dry_run=True
            )
        assert result == portfolio
        post.assert_not_called()
        assert any("no surplus" in r.getMessage() for r in caplog.records)

    def test_funded_deficit_posts_exact_body_and_returns_full_portfolio(self, monkeypatch):
        # Shard 0 needs 1500c but holds 100c; shard 1 has the rest.
        post, va = self._patch_io(
            monkeypatch,
            # Insufficient on the first re-read, sufficient on the second —
            # acceptance is not settlement, so the poll must keep looking.
            balances=MagicMock(side_effect=[{0: 100, 1: 100_000}, {0: 1500, 1: 98_600}]),
        )
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, None
        )
        assert result == portfolio
        assert post.call_count == 1
        args, kwargs = post.call_args
        assert args[1] == "POST"
        assert args[2] == TRANSFER_PATH
        assert kwargs["body"] == {
            "source": "event_contract",
            "destination": "event_contract",
            # 1400 cents == 140,000 CENTICENTS — not 1400, not 14.00
            "amount": 140_000,
            "source_exchange_shard": 1,
            "destination_exchange_shard": 0,
        }
        assert va.call_count == 2

    def test_settle_timeout_drops_only_unfunded_shard_specs(self, monkeypatch, caplog):
        # Transfer accepted but never lands: money is in flight, so the run
        # must shout and drop ONLY the trades that were waiting on it.
        post, va = self._patch_io(
            monkeypatch,
            transfer=MagicMock(return_value=transfer_resp("tr_stuck")),
            balances=MagicMock(return_value={0: 100, 1: 100_000}),
            # Deadline already elapsed: one re-read, then give up.
            settle_timeout=0,
        )
        needy = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00,
                          title="needy pair")
        funded = make_spec(shard_a=1, shard_b=1, cost_a=1.00, cost_b=1.00,
                           title="funded pair")
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), [needy, funded], {0: 100, 1: 100_000}, None
            )
        assert result == [funded]
        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert criticals, "an unsettled transfer must be logged at CRITICAL"
        blob = " ".join(r.getMessage() for r in criticals)
        assert "tr_stuck" in blob
        assert "IN FLIGHT" in blob

    def test_inactive_transfers_block_the_post_and_drop_affected_specs(
        self, monkeypatch, caplog
    ):
        # Shard 0 (the destination) is not accepting intra-exchange transfers.
        post, va = self._patch_io(monkeypatch)
        statuses = {0: shard_status(False), 1: shard_status(True)}
        needy = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00,
                          title="needy pair")
        funded = make_spec(shard_a=1, shard_b=1, cost_a=1.00, cost_b=1.00,
                           title="funded pair")
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), [needy, funded], {0: 100, 1: 100_000}, statuses
            )
        assert result == [funded]
        post.assert_not_called()
        # Nothing was sent, so there is nothing to wait for either.
        va.assert_not_called()
        warnings = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "manually" in warnings

    def test_inactive_source_shard_also_blocks_the_post(self, monkeypatch):
        # Same gate from the other end: the SOURCE shard can't send.
        post, va = self._patch_io(monkeypatch)
        statuses = {0: shard_status(True), 1: shard_status(False)}
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, statuses
        )
        assert result == []
        post.assert_not_called()

    def test_none_statuses_still_attempts_the_transfer(self, monkeypatch):
        # No per-shard breakdown (sandbox / pre-sharding shape) means there is
        # nothing to gate on — attempt it and let the POST fail loudly if the
        # endpoint is unsupported.
        post, va = self._patch_io(
            monkeypatch, balances=MagicMock(return_value={0: 1500, 1: 98_600})
        )
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, None
        )
        assert result == portfolio
        assert post.call_count == 1

    def test_failed_post_drops_affected_specs_and_keeps_the_rest(self, monkeypatch, caplog):
        post, va = self._patch_io(
            monkeypatch, transfer=MagicMock(side_effect=RuntimeError("boom"))
        )
        needy = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00,
                          title="needy pair")
        funded = make_spec(shard_a=1, shard_b=1, cost_a=1.00, cost_b=1.00,
                           title="funded pair")
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), [needy, funded], {0: 100, 1: 100_000}, None
            )
        assert result == [funded]
        errors = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
        )
        assert "FAILED" in errors
        # Nothing was accepted, so no settle poll is owed.
        va.assert_not_called()

    def test_failed_post_is_never_retried(self, monkeypatch):
        # A retried transfer moves the money TWICE — the endpoint is not
        # idempotent. Exactly one attempt, no matter what it raises.
        post, _ = self._patch_io(
            monkeypatch, transfer=MagicMock(side_effect=TimeoutError("timeout"))
        )
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        ensure_shard_collateral(MagicMock(), portfolio, {0: 100, 1: 100_000}, None)
        assert post.call_count == 1

    def test_transfer_path_bypasses_the_retry_wrapper_entirely(self):
        # Checked per function: order and transfer POSTs never go through the
        # retry wrapper, while the read-only position lookup does
        for fn in (trader._submit_order_v2, trader._execute_transfer):
            assert not _calls_retry_wrapper(fn), (
                f"{fn.__name__} must not be wrapped in retry/backoff — "
                "a retried submission can double-fill and a retried transfer "
                "moves the money twice"
            )
        # The asymmetry is deliberate and is itself pinned: the read-only
        # position lookup DOES retry (see TestPositionCountRetry).
        assert _calls_retry_wrapper(trader._position_count)

    def test_order_submission_call_sites_bypass_the_retry_wrapper(self):
        # Neither caller of _submit_order_v2 retries it
        for fn in (trader._execute_one, trader._rollback_no_leg):
            assert not _calls_retry_wrapper(fn), (
                f"{fn.__name__} must not wrap an order submission in "
                "retry/backoff — a retried order POST can fill twice"
            )

    def test_cross_shard_spec_funds_both_legs_shards(self, monkeypatch):
        # Legs on different shards: BOTH must be covered or the pair is a
        # half-fill risk, so a shortfall on either one drops the whole spec.
        post, va = self._patch_io(
            monkeypatch, balances=MagicMock(return_value={0: 100, 1: 100_000})
        )
        spec = make_spec(shard_a=0, shard_b=1, cost_a=10.00, cost_b=5.00)
        result = ensure_shard_collateral(
            MagicMock(), [spec], {0: 100, 1: 100_000}, None
        )
        # Shard 0 needs 1000c and only ever holds 100c → the pair is dropped
        # even though shard 1's leg is amply funded.
        assert result == []

    def test_no_surplus_anywhere_drops_without_posting(self, monkeypatch):
        # An empty plan here means "nothing movable", NOT "nothing needed" —
        # the specs must still be dropped rather than sailing through.
        post, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)]
        assert ensure_shard_collateral(MagicMock(), portfolio, {0: 100}, None) == []
        post.assert_not_called()
        va.assert_not_called()


class TestV2PriceMath:
    def test_ceil_to_tick_on_grid_unchanged(self):
        # A price already on the grid must not be nudged up a tick — that would
        # widen the cap by one extra tick on every single order.
        assert _ceil_to_tick(Decimal("0.35"), Decimal("0.01")) == Decimal("0.35")
        assert _ceil_to_tick(Decimal("0.501"), Decimal("0.001")) == Decimal("0.501")

    def test_ceil_to_tick_off_grid_rounds_up(self):
        # Ceiling, never nearest/floor: a cap below the scanned price could
        # never fill at the price we scanned.
        assert _ceil_to_tick(Decimal("0.3512"), Decimal("0.01")) == Decimal("0.36")
        assert _ceil_to_tick(Decimal("0.50051"), Decimal("0.001")) == Decimal("0.501")

    def test_buy_yes_limit_adds_n_ticks_above_ceiled_price(self):
        market = make_market()  # unknown structure -> $0.01 fallback grid
        price = _v2_limit_price("buy_yes", 0.35, market)
        assert price == Decimal("0.35") + BUY_SLIPPAGE_TICKS * Decimal("0.01")

    def test_buy_no_limit_is_complement_of_capped_no_price(self):
        # Buying NO is an ASK on the single YES book at 1 - (capped NO price).
        market = make_market()
        price = _v2_limit_price("buy_no", 0.40, market)
        assert price == Decimal("1") - (Decimal("0.40") + BUY_SLIPPAGE_TICKS * Decimal("0.01"))
        assert price == Decimal("0.59")

    def test_linear_cent_cap_is_one_cent_above_the_scanned_price(self):
        # On a 1c grid the cap is the price rounded up to the cent plus
        # BUY_SLIPPAGE_TICKS cents
        market = make_market("linear_cent")
        price = _v2_limit_price("buy_yes", 0.35, market)
        cap_cents = math.ceil(round(0.35 * 100, 6)) + BUY_SLIPPAGE_TICKS
        for count in (1, 5, 17):
            assert price * count * 100 == count * cap_cents

    def test_v2_float_noise_does_not_loosen_the_cap(self):
        # 1.0 - 0.43 is 0.5700000000000001; the cap must still be 0.58, not
        # 0.59 (TS-03)
        market = make_market("linear_cent")
        assert _v2_limit_price("buy_yes", 1.0 - 0.43, market) == Decimal("0.58")
        # buy_no: NO price 0.30000000000000004 -> cap 0.31 -> YES-book ask 0.69
        assert _v2_limit_price("buy_no", 1.0 - 0.70, market) == Decimal("0.69")

    def test_linear_cent_cap_is_one_cent_above_every_whole_cent_ask(self):
        # For every whole-cent bid, the ask 1.0 - bid gets a cap of that ask
        # rounded up to the cent plus BUY_SLIPPAGE_TICKS cents
        market = make_market("linear_cent")
        for cents in range(2, 100):
            p = 1.0 - cents / 100          # the exact form the scanner produces
            cap = _v2_limit_price("buy_yes", p, market)
            assert cap * 100 == math.ceil(round(p * 100, 6)) + BUY_SLIPPAGE_TICKS, cents
        # Except 0.99: one cent above it is $1.00, not a tradeable level, so
        # the cap is the top of the grid
        assert _v2_limit_price("buy_yes", 1.0 - 0.01, market) == Decimal("0.99")

    def test_deci_cent_cap_moves_one_deci_cent_not_one_cent(self):
        market = make_market("deci_cent", DECI_CENT_BANDS)
        assert _v2_limit_price("buy_yes", 0.5, market) == Decimal("0.501")

    def test_centi_cent_edge_band_cap_moves_one_centi_cent(self):
        # 0.995 sits in the 0.99-1.0 step-0.0001 edge band.
        market = make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        assert _v2_limit_price("buy_yes", 0.995, market) == Decimal("0.9951")

    def test_cross_band_cap_lands_on_valid_finer_grid(self):
        # 0.9895 is in the 0.01-0.99 step-0.001 band, so the cap steps up to
        # 0.991 — across the band edge into the finer 0.0001 edge band. The
        # grids are nested, so that is still a valid, tradeable price level.
        market = make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        price = _v2_limit_price("buy_yes", 0.9895, market)
        assert price == Decimal("0.991")
        assert price % Decimal("0.0001") == 0

    def test_cap_clamped_inside_open_unit_interval_on_grid(self):
        # 0 and 1 are settlement values, not tradeable price levels — and the
        # clamp bounds must be valid levels of THIS market's grid: on a
        # linear-cent market the extremes are 0.99/0.01, not the finest-grid
        # 0.9999/0.0001 (which a cent-tick book would reject as off-grid).
        market = make_market("linear_cent")
        assert _v2_limit_price("buy_yes", 0.999, market) == Decimal("0.99")
        assert _v2_limit_price("buy_no", 0.999, market) == Decimal("0.01")

    def test_cap_stepping_into_coarser_band_requantizes(self):
        # Regression (found in adversarial review): scanned 0.00995 sits in the
        # $0.0001 edge band, but cap = ceil + 1 tick = 0.0101 lands in the
        # $0.001 middle band, where 0.0101 is NOT a valid level. The final
        # price must be re-quantized onto the destination band's grid
        # (ceiling: worst case a killed FoK, never a worse fill).
        market = make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        assert _v2_limit_price("buy_yes", 0.00995, market) == Decimal("0.011")

    def test_no_leg_complement_requantized_onto_containing_band(self):
        # Same regression, NO side: 1 - 0.0101 = 0.9899 is off the $0.001 grid
        # of the middle band containing it; must snap up to 0.990 — which is
        # still fillable at the scanned NO price (0.990 <= 1 - 0.00995).
        market = make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        assert _v2_limit_price("buy_no", 0.00995, market) == Decimal("0.990")

    def test_format_price_always_four_decimals(self):
        assert _format_price(Decimal("0.56")) == "0.5600"
        assert _format_price(Decimal("0.9951")) == "0.9951"
        assert _format_price(Decimal("0.5")) == "0.5000"

    def test_format_count_fixed_point_string(self):
        assert _format_count(10) == "10.00"
        assert _format_count(1) == "1.00"


class TestV2OrderBuilders:
    def test_no_leg_is_ask_at_one_minus_capped_no_price(self):
        body = _build_no_order_v2(_no_leg(make_spec(x=5, nA=0.40)))
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "ask"
        assert body["price"] == "0.5900"

    def test_yes_leg_is_bid_at_capped_yes_price(self):
        body = _build_yes_order_v2(_yes_leg(make_spec(x=5, pB=0.35)))
        assert body["ticker"] == "TICK-B"
        assert body["side"] == "bid"
        assert body["price"] == "0.3600"

    def test_rollback_is_reduce_only_bid_at_the_loss_floored_price(self):
        # The unwind is a reduce-only YES bid capped at 1 - the NO loss floor:
        # nA=0.40 -> floor 40-12=28c -> cap 0.72
        body = _build_rollback_order_v2(_no_leg(make_spec()))
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "bid"
        assert body["reduce_only"] is True
        assert body["price"] == "0.7200"

    def test_rollback_price_is_the_yes_book_mirror_of_the_loss_floor(self):
        # The bid cap is 1 - the NO loss floor
        for nA in (0.40, 0.57, 0.62, 0.85):
            spec = make_spec(nA=nA)
            floor_cents = _rollback_floor_cents(_no_leg(spec))
            assert _v2_rollback_price(_no_leg(spec)) == (
                Decimal("1") - Decimal(floor_cents) / Decimal("100")
            )

    def test_rollback_price_never_exceeds_the_markets_top_of_grid(self):
        # A very cheap NO leg clamps the loss floor to 1c, which mirrors to a
        # 0.99 cap — the highest tradeable level on a whole-cent grid. The
        # invariant asserted here is the INEQUALITY: the cap may never exceed
        # the market's own top-of-grid level on any regime, which is what keeps
        # the bid a quotable price rather than a settlement value.
        #
        # On every grid Kalshi actually serves today the clamp does not BIND:
        # on linear_cent the cap TIES top-of-grid (0.99 == 0.99), and on the
        # finer regimes top-of-grid is strictly higher (0.999 / 0.9999), so the
        # cap sits strictly below it. The clamp exists for a hypothetical
        # coarser-than-cent band, where the mirrored floor could land above the
        # highest quotable level. Hence `<=`, not `==` — a change that made the
        # clamp bind would still be correct, and this test would still hold.
        assert _v2_rollback_price(_no_leg(make_spec(nA=0.05))) == Decimal("0.99")
        for structure, bands in (
            ("linear_cent", None),
            ("deci_cent", DECI_CENT_BANDS),
            ("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS),
        ):
            spec = make_spec(nA=0.05, structure=structure, ranges=bands)
            assert _v2_rollback_price(_no_leg(spec)) <= _v2_top_of_grid_price(_no_leg(spec).market)

    def test_rollback_price_lands_on_the_markets_own_tick_grid(self):
        # Ceiling-quantization onto the band containing the cap. With a whole-
        # cent floor this is a no-op on every nested grid, which is exactly the
        # point: the cap must never end up BELOW every level of its band, which
        # is what a floor-quantized off-grid cap would do — structurally killing
        # the unwind and orphaning the position.
        for structure, bands, tick in (
            ("linear_cent", None, Decimal("0.01")),
            ("deci_cent", DECI_CENT_BANDS, Decimal("0.001")),
            ("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS, Decimal("0.001")),
        ):
            price = _v2_rollback_price(_no_leg(make_spec(nA=0.40, structure=structure, ranges=bands)))
            assert price == Decimal("0.72")
            assert price % tick == 0

    def test_top_of_grid_price_floors_to_market_grid(self):
        # Regression (found in adversarial review): a flat $0.99 bid cannot
        # cross asks resting in (0.99, 1) on sub-cent regimes. The top-of-grid
        # level — the rollback bid's upper clamp, and what v2_probe's
        # unfillable-ask step submits — must be THIS market's highest level.
        assert _v2_top_of_grid_price(make_market("linear_cent")) == Decimal("0.99")
        assert _v2_top_of_grid_price(
            make_market("deci_cent", DECI_CENT_BANDS)
        ) == Decimal("0.999")
        assert _v2_top_of_grid_price(
            make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        ) == Decimal("0.9999")

    def test_buy_legs_fill_or_kill_and_the_unwind_immediate_or_cancel(self):
        # The two buy legs fill in full or not at all. The unwind is
        # reduce_only, which the V2 endpoint accepts only with
        # immediate_or_cancel, so it closes what it can and cancels the rest.
        spec = make_spec()
        no_body = _build_no_order_v2(_no_leg(spec))
        yes_body = _build_yes_order_v2(_yes_leg(spec))
        rollback_body = _build_rollback_order_v2(_no_leg(spec))
        assert no_body["time_in_force"] == "fill_or_kill"
        assert yes_body["time_in_force"] == "fill_or_kill"
        assert rollback_body["time_in_force"] == "immediate_or_cancel"
        assert rollback_body["reduce_only"] is True
        for body in (no_body, yes_body, rollback_body):
            assert body["post_only"] is False

    # Every value the V2 create-order endpoint documents for the fields it
    # requires (https://docs.kalshi.com/api-reference/orders/create-order-v2).
    # A body missing one is rejected with HTTP 400.
    _V2_REQUIRED_FIELDS = {
        "ticker", "side", "count", "price", "time_in_force", "self_trade_prevention_type",
    }
    _V2_SIDES = {"bid", "ask"}
    _V2_TIME_IN_FORCE = {"fill_or_kill", "good_till_canceled", "immediate_or_cancel"}
    _V2_SELF_TRADE_PREVENTION = {"taker_at_cross", "maker"}

    @staticmethod
    def _all_v2_bodies(spec) -> list[dict]:
        """The three V2 bodies _execute_one can send for one spec."""
        return [
            _build_no_order_v2(_no_leg(spec)),
            _build_yes_order_v2(_yes_leg(spec)),
            _build_rollback_order_v2(_no_leg(spec)),
        ]

    def test_every_v2_body_carries_the_documented_required_fields(self):
        for spec in (make_spec(), TestTimeSeriesLegOrder._ts_spec()):
            for body in self._all_v2_bodies(spec):
                assert self._V2_REQUIRED_FIELDS <= body.keys()
                assert body["side"] in self._V2_SIDES
                assert body["time_in_force"] in self._V2_TIME_IN_FORCE
                assert body["self_trade_prevention_type"] in self._V2_SELF_TRADE_PREVENTION

    def test_reduce_only_only_with_immediate_or_cancel(self):
        # The docs, verbatim: "Orders with reduce_only set to true will be
        # rejected unless time_in_force is immediate_or_cancel."
        for structure, bands in (
            ("linear_cent", None),
            ("deci_cent", DECI_CENT_BANDS),
            ("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS),
        ):
            for spec in (
                make_spec(structure=structure, ranges=bands),
                TestTimeSeriesLegOrder._ts_spec(structure=structure, ranges=bands),
            ):
                for body in self._all_v2_bodies(spec):
                    assert (
                        not body["reduce_only"]
                        or body["time_in_force"] == "immediate_or_cancel"
                    )

    def test_self_trade_prevention_type_comes_from_config(self, monkeypatch):
        # The builders read the module binding of the config constant, so a
        # change there reaches every V2 body.
        monkeypatch.setattr(trader, "V2_SELF_TRADE_PREVENTION_TYPE", "maker")
        for body in self._all_v2_bodies(make_spec()):
            assert body["self_trade_prevention_type"] == "maker"

    def test_each_leg_carries_its_own_markets_shard(self):
        # Per-leg routing: a pair's two legs can live on different shards, so
        # each body takes exchange_index from its OWN market — and the rollback
        # routes to market A, the shard the position was opened on. Never -1
        # (auto-route): a wrong shard must be rejected loudly by the exchange,
        # not silently papered over.
        spec = make_spec(shard_a=2, shard_b=3)
        no_body = _build_no_order_v2(_no_leg(spec))
        yes_body = _build_yes_order_v2(_yes_leg(spec))
        rollback_body = _build_rollback_order_v2(_no_leg(spec))
        assert no_body["exchange_index"] == 2
        assert yes_body["exchange_index"] == 3
        assert rollback_body["exchange_index"] == 2
        for body in (no_body, yes_body, rollback_body):
            assert body["exchange_index"] != -1

    def test_an_off_default_shard_pair_is_submitted_on_each_legs_own_shard(
        self, v2_mapping_confirmed, monkeypatch
    ):
        # A leg off DEFAULT_EXCHANGE_INDEX is traded normally, each order on
        # its own market's shard
        post = MagicMock(side_effect=[v2_resp(5), v2_resp(5)])
        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock()
        result = _execute_one(client, make_spec(shard_a=1))
        assert result.status == "executed"
        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["body"]["exchange_index"] == 1
        assert post.call_args_list[1].kwargs["body"]["exchange_index"] == DEFAULT_EXCHANGE_INDEX

    def test_default_shard_spec_carries_the_default_exchange_index(self):
        # The universal case today: everything is on DEFAULT_EXCHANGE_INDEX.
        spec = make_spec()
        for body in (
            _build_no_order_v2(_no_leg(spec)), _build_yes_order_v2(_yes_leg(spec)), _build_rollback_order_v2(_no_leg(spec)),
        ):
            assert body["exchange_index"] == DEFAULT_EXCHANGE_INDEX

    def test_client_order_ids_are_unique_uuids(self):
        spec = make_spec()
        ids = [
            _build_no_order_v2(_no_leg(spec))["client_order_id"],
            _build_yes_order_v2(_yes_leg(spec))["client_order_id"],
            _build_rollback_order_v2(_no_leg(spec))["client_order_id"],
            _build_no_order_v2(_no_leg(spec))["client_order_id"],
        ]
        assert len(set(ids)) == len(ids)
        for cid in ids:
            uuid.UUID(cid)  # raises if not a valid UUID string

    def test_counts_serialized_as_fixed_point_strings(self):
        spec = make_spec(x=7)
        assert _build_no_order_v2(_no_leg(spec))["count"] == "7.00"
        assert _build_yes_order_v2(_yes_leg(spec))["count"] == "7.00"
        assert _build_rollback_order_v2(_no_leg(spec))["count"] == "7.00"

    def test_buy_legs_are_not_reduce_only(self):
        # Only the unwind closes exposure; a reduce_only buy leg would never fill.
        spec = make_spec()
        assert _build_no_order_v2(_no_leg(spec))["reduce_only"] is False
        assert _build_yes_order_v2(_yes_leg(spec))["reduce_only"] is False


class TestV2FillStatus:
    def test_full_fill_plain_int_is_executed(self):
        assert _v2_fill_status({"order": {"fill_count": 10}}, 10) == "executed"

    def test_full_fill_fp_string_is_executed(self):
        assert _v2_fill_status({"order": {"fill_count_fp": "10.00"}}, 10) == "executed"

    def test_zero_fill_is_canceled(self):
        assert _v2_fill_status({"order": {"fill_count": 0}}, 10) == "canceled"
        assert _v2_fill_status({"order": {"fill_count_fp": "0.00"}}, 10) == "canceled"

    def test_partial_fill_raises_for_ambiguous_path(self):
        # A partial fill is neither a clean fill nor a clean kill, so it raises.
        # On a buy leg (fill-or-kill) that routes _execute_one into the
        # position lookup; on the immediate-or-cancel unwind, _rollback_no_leg
        # reports it as rollback_failed.
        with pytest.raises(ValueError):
            _v2_fill_status({"order": {"fill_count": 4}}, 10)

    def test_missing_fill_count_raises(self):
        with pytest.raises(ValueError):
            _v2_fill_status({"order": {"order_id": "ord-1"}}, 10)

    def test_wrapped_order_key_unwrapped(self):
        # The inner order object wins over any same-named top-level field.
        data = {"order": {"fill_count": 10}, "fill_count": 0}
        assert _v2_fill_status(data, 10) == "executed"

    def test_flat_response_accepted(self):
        assert _v2_fill_status({"fill_count": 10}, 10) == "executed"


class TestIsFokKill:
    """_is_fok_kill recognises exactly one response: HTTP 409 whose JSON body
    carries the fill-or-kill kill code under ["error"]["code"]."""

    def test_the_exchanges_kill_response_is_a_kill(self):
        assert _is_fok_kill(fok_kill_error()) is True

    def test_a_bytes_body_is_read_as_utf_8(self):
        exc = ApiException(status=409, reason="Conflict", body=FOK_KILL_BODY.encode())
        assert _is_fok_kill(exc) is True

    def test_the_sdk_exception_for_a_409_response_is_a_kill(self):
        # The exception the live transport really raises: _check_and_parse
        # hands a 409 to ApiException.from_response, which raises the SDK's
        # ConflictException subclass with the decoded body.
        resp = SimpleNamespace(
            status=409, reason="Conflict", data=FOK_KILL_BODY.encode("utf-8"),
            getheaders=lambda: {"content-type": "application/json"},
        )
        with pytest.raises(ApiException) as exc_info:
            _http._check_and_parse(resp)
        assert type(exc_info.value) is not ApiException
        assert _is_fok_kill(exc_info.value) is True

    @pytest.mark.parametrize("exc", [
        ApiException(status=409, reason="Conflict", body=None),
        ApiException(status=409, reason="Conflict"),
        ApiException(
            status=409, reason="Conflict",
            body='{"error":{"code":"insufficient_balance","message":"x"}}',
        ),
        ApiException(status=400, reason="Bad Request", body=FOK_KILL_BODY),
        ApiException(status=409, reason="Conflict", body="not json"),
        ApiException(status=409, reason="Conflict", body=""),
        ApiException(status=409, reason="Conflict", body=b"\xff\xfe"),
        ApiException(status=409, reason="Conflict", body=json.dumps([FOK_KILL_BODY])),
        ApiException(
            status=409, reason="Conflict",
            body='{"error":"fill_or_kill_insufficient_resting_volume"}',
        ),
        ApiException(
            status=409, reason="Conflict",
            body='{"code":"fill_or_kill_insufficient_resting_volume"}',
        ),
        ApiException(status=409, reason="Conflict", body=123),
        ApiException(status="409", reason="Conflict", body=FOK_KILL_BODY),
        Exception(FOK_KILL_BODY),
        TimeoutError("timeout"),
    ], ids=[
        "none-body", "no-body", "another-code", "status-400", "non-json",
        "empty-body", "non-utf8-bytes", "json-list", "error-not-object",
        "code-not-under-error", "int-body", "string-status", "plain-exception",
        "timeout",
    ])
    def test_anything_else_is_not_a_kill(self, exc):
        assert _is_fok_kill(exc) is False

    def test_an_exception_that_is_not_an_api_exception_is_not_a_kill(self):
        # Carrying the kill's status and body is not enough: only the SDK's
        # ApiException comes from the exchange's HTTP response.
        exc = Exception("look-alike")
        exc.status = 409
        exc.body = FOK_KILL_BODY
        assert _is_fok_kill(exc) is False

    def test_a_subclass_with_another_status_is_not_a_kill(self):
        # The SDK's own 400 subclass carrying the kill code is still not the
        # kill: the status and the code must both match.
        from kalshi_python_sync.exceptions import BadRequestException
        exc = BadRequestException(status=400, reason="Bad Request", body=FOK_KILL_BODY)
        assert _is_fok_kill(exc) is False

    def test_the_constants_are_the_observed_response(self):
        # The wire values the exchange sent; config is the one place they live.
        assert config.V2_FOK_KILL_HTTP_STATUS == 409
        assert config.V2_FOK_KILL_ERROR_CODE == "fill_or_kill_insufficient_resting_volume"


class TestSubmitOrderV2KillResponse:
    """_submit_order_v2 returns the kill response to a fill_or_kill body as
    "canceled", single-shot, and re-raises every other error."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    @pytest.mark.parametrize("builder, leg", [
        (_build_no_order_v2, _no_leg), (_build_yes_order_v2, _yes_leg),
    ], ids=["no-leg", "yes-leg"])
    def test_a_killed_fill_or_kill_is_canceled(self, post, caplog, builder, leg):
        body = builder(leg(make_spec()))
        assert body["time_in_force"] == "fill_or_kill"
        post.side_effect = fok_kill_error()
        with caplog.at_level(logging.INFO):
            assert _submit_order_v2(MagicMock(), body) == "canceled"
        # One POST: the kill is an answer, never a reason to resubmit
        assert post.call_count == 1
        kill_lines = [r for r in caplog.records if "killed by the exchange" in r.getMessage()]
        assert len(kill_lines) == 1
        assert kill_lines[0].levelno == logging.INFO
        line = kill_lines[0].getMessage()
        for part in (
            "409", "fill_or_kill_insufficient_resting_volume", body["ticker"],
            body["side"], body["price"], body["count"], body["client_order_id"],
        ):
            assert part in line
        # Nothing at WARNING or above: a kill is routine
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_the_same_response_to_the_unwind_raises(self, post):
        # The immediate_or_cancel unwind never gets the fill-or-kill reading:
        # an error on it stays an error, which _rollback_no_leg reports as
        # rollback_failed.
        body = _build_rollback_order_v2(_no_leg(make_spec()))
        assert body["time_in_force"] == "immediate_or_cancel"
        err = fok_kill_error()
        post.side_effect = err
        with pytest.raises(ApiException) as exc_info:
            _submit_order_v2(MagicMock(), body)
        assert exc_info.value is err
        assert post.call_count == 1

    @pytest.mark.parametrize("err", [
        ApiException(
            status=409, reason="Conflict",
            body='{"error":{"code":"insufficient_balance","message":"x"}}',
        ),
        ApiException(status=400, reason="Bad Request", body=FOK_KILL_BODY),
        ApiException(status=500, reason="Internal Server Error"),
    ], ids=["409-another-code", "400-kill-code", "500"])
    def test_any_other_error_raises(self, post, err):
        post.side_effect = err
        with pytest.raises(ApiException) as exc_info:
            _submit_order_v2(MagicMock(), _build_no_order_v2(_no_leg(make_spec())))
        assert exc_info.value is err
        assert post.call_count == 1


class TestV2ExecuteOne:
    """_execute_one's outcomes: every status, how many orders are sent, and
    the unwind's body."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """The NO-leg backstop already latched: these cases test the state
        machine, not the first-fill mapping check, and must not have their
        position mocks consumed by it."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_v2_both_legs_filled_is_executed(self, post):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "executed"
        assert post.call_count == 2
        # Every submission goes to the V2 route, by POST
        for call in post.call_args_list:
            assert call.args[1:] == ("POST", V2_ORDER_PATH)

    def test_v2_leg_a_killed_is_failed_no_leg_b_submitted(self, post):
        post.side_effect = [v2_resp(0)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "failed"
        assert post.call_count == 1

    def test_v2_leg_b_killed_rolls_back_with_reduce_only_bid(self, post):
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(5)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rolled_back"
        rollback_body = post.call_args_list[2].kwargs["body"]
        assert rollback_body["ticker"] == "TICK-A"
        assert rollback_body["side"] == "bid"
        assert rollback_body["reduce_only"] is True
        # reduce_only is accepted only with immediate_or_cancel, and every V2
        # body carries the required self-trade-prevention field
        assert rollback_body["time_in_force"] == "immediate_or_cancel"
        assert rollback_body["self_trade_prevention_type"] == "taker_at_cross"
        # Loss-floored, not a flat top-of-grid bid: default spec nA=0.40 ->
        # floor 40-12=28c -> bid cap 1 - 0.28 = 0.72 on the $0.01 grid
        assert rollback_body["price"] == "0.7200"

    def test_v2_unfilled_rollback_is_rollback_failed(self, post):
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(0)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error

    def test_v2_partial_unwind_is_rollback_failed(self, post, caplog):
        # The immediate-or-cancel unwind closes 3 of the 5 NO contracts and
        # cancels the rest. That is never reported as flat: the fill count
        # makes _v2_fill_status raise, _rollback_no_leg reports
        # rollback_failed, and no second order is sent. Its alert and the
        # error text name the exact count still open, 5 - 3 = 2, read from the
        # unwind's own response, not the "up to 5" said when it is unknown.
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(3)]
        with caplog.at_level(logging.CRITICAL):
            result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "fill_count=3" in result.error
        assert result.error.endswith("; 2 of 5 NO contracts still open")
        assert post.call_count == 3
        orphan = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.CRITICAL and "ORPHANED POSITION" in r.getMessage()]
        assert len(orphan) == 1
        assert "closed 3 of the 5 NO contracts this pair bought on TICK-A," in orphan[0]
        assert "so 2 of them are still open" in orphan[0]
        assert "up to" not in orphan[0]

    def test_v2_leg_a_exception_with_position_is_unwound(self, post):
        post.side_effect = [TimeoutError("timeout"), v2_resp(5)]
        client = MagicMock()
        # Delta, not the absolute holding: flat baselines for both legs, then
        # -5 on TICK-A after the exception = exactly our NO buy (spec.x=5).
        # A flat return_value would make before == after, i.e. delta 0.
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-A", -5),
        )
        assert _execute_one(client, make_spec()).status == "rolled_back"

    def test_v2_leg_a_exception_with_no_position_is_failed(self, post, monkeypatch):
        # The flat return_value answers the DR-64 lag re-read too — the ledger
        # is unmoved on both readings, so this stays a confirmed non-fill; the
        # sleep is patched out so the suite does not pay the real delay.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = TimeoutError("timeout")
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert post.call_count == 1

    def test_v2_leg_b_exception_with_position_is_executed(self, post):
        post.side_effect = [v2_resp(5), TimeoutError("timeout")]
        client = MagicMock()
        # Delta, not the absolute holding: flat baselines, then +5 on TICK-B
        # after the exception = exactly our YES buy (spec.y=5), so the pair
        # actually completed and rolling NO leg back would REVERSE the hedge.
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-B", 5),
        )
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        # No rollback was submitted
        assert post.call_count == 2

    def test_v2_leg_b_exception_with_unknown_position_is_manual_review(self, post):
        # The read failed, so the state is unknown: no rollback. The call
        # count and sleeps also check the read runs outside the except block,
        # where the TimeoutError context would make the retry wrapper retry it
        post.side_effect = [v2_resp(5), TimeoutError("timeout")]
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        with patch.object(_http.time, "sleep") as sleep:
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert post.call_count == 2
        # The two up-front baselines and one read after the YES leg raised
        assert client.get_positions_without_preload_content.call_count == 3
        sleep.assert_not_called()

    def test_v2_leg_a_kill_response_is_failed_at_once(self, post, monkeypatch, caplog):
        # The exchange's HTTP 409 kill of the NO leg is a confirmed non-fill:
        # the pair ends "failed" with no position read after the submission
        # and no pause, and nothing is logged at ERROR.
        slept: list = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        post.side_effect = [fok_kill_error()]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None)
        with caplog.at_level(logging.INFO):
            result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert result.error == "NO leg FoK not filled: status=canceled"
        assert post.call_count == 1
        # The two up-front baselines only
        assert client.get_positions_without_preload_content.call_count == 2
        assert slept == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_v2_leg_b_kill_response_rolls_back_at_once(self, post, monkeypatch):
        # The exchange's HTTP 409 kill of the YES leg goes straight to the
        # unwind: no position read after the submission and no pause before
        # the rollback, which is immediate_or_cancel as always.
        slept: list = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        post.side_effect = [v2_resp(5), fok_kill_error(), v2_resp(5)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"
        assert result.error == "YES leg FoK not filled: status=canceled"
        assert post.call_count == 3
        rollback_body = post.call_args_list[2].kwargs["body"]
        assert rollback_body["ticker"] == "TICK-A"
        assert rollback_body["reduce_only"] is True
        assert rollback_body["time_in_force"] == "immediate_or_cancel"
        assert client.get_positions_without_preload_content.call_count == 2
        assert slept == []

    def test_v2_exactly_one_post_per_leg_no_retry_on_5xx(self, post, monkeypatch):
        # A 5xx on an order submission must NEVER be retried: a second FoK
        # could fill the leg twice at a different price.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = ApiException(status=500, reason="server error")
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert post.call_count == 1

    def test_the_sdk_create_order_endpoint_is_never_called(self, post):
        # All three orders go through signed_request_json; the SDK's
        # create-order methods are never called
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(5)]
        client = MagicMock()
        assert _execute_one(client, make_spec()).status == "rolled_back"
        assert post.call_count == 3
        for call in post.call_args_list:
            assert call.args[1:] == ("POST", V2_ORDER_PATH)
        for name in ("create_order", "create_order_without_preload_content",
                     "batch_create_orders", "batch_create_orders_without_preload_content"):
            getattr(client, name).assert_not_called()


class TestPartialUnwindCount:
    """How many NO contracts a failed V2 unwind leaves open, as its alert says.

    The V2 unwind is immediate_or_cancel, so its 2xx response can report a
    fill count strictly between zero and the NO leg's count: it closed that
    many and left nothing resting. _rollback_no_leg reads that count from the
    body _submit_order_v2 attaches to its error and names the exact number
    still open (count minus fill). Every other failure leaves the number
    unknown and the alert says "up to" the count. Either way the pair is
    rollback_failed and no further order is sent. The two buy legs keep
    raising on a partial fill into _execute_one's position check.
    """

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mapping_confirmed):
        """NO-leg mapping already latched (see TestV2ExecuteOne)."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    @pytest.fixture
    def slept(self, monkeypatch):
        """Every pause _execute_one takes, recorded instead of slept."""
        pauses: list = []
        monkeypatch.setattr(trader.time, "sleep", pauses.append)
        return pauses

    @staticmethod
    def _orphan_alerts(caplog) -> list[str]:
        """The CRITICAL lines that report an orphaned NO position."""
        return [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.CRITICAL and "ORPHANED POSITION" in r.getMessage()
        ]

    def _unwind(self, post, caplog, rollback_reply, spec=None):
        """Run one pair whose NO leg fills, whose YES leg is killed, and whose
        unwind answers with `rollback_reply` (a body, or an exception to
        raise). Returns the result and the client, whose position reads are
        scripted for the two up-front baselines only."""
        post.side_effect = [v2_resp(5), v2_resp(0), rollback_reply]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None)
        with caplog.at_level(logging.CRITICAL):
            result = _execute_one(client, spec or make_spec())
        return result, client

    @pytest.mark.parametrize(
        "body, closed, still_open",
        [
            pytest.param({"order": {"fill_count": 1}}, "1", "4", id="int-1"),
            pytest.param({"order": {"fill_count": 4}}, "4", "1", id="int-4"),
            pytest.param({"order": {"fill_count_fp": "3.00"}}, "3", "2", id="fp-string"),
            # A flat body, not wrapped under "order"
            pytest.param({"fill_count_fp": "3.00"}, "3", "2", id="flat"),
            # Fractional contracts are counted exactly
            pytest.param({"order": {"fill_count_fp": "2.50"}}, "2.5", "2.5", id="fractional"),
            pytest.param({"order": {"fill_count_fp": "0.01"}}, "0.01", "4.99", id="one-hundredth"),
            # The _fp field wins when both are present, as in _v2_fill_status
            pytest.param(
                {"order": {"fill_count_fp": "1.00", "fill_count": 4}}, "1", "4",
                id="fp-over-int",
            ),
        ],
    )
    def test_a_partial_close_names_the_exact_count_still_open(
        self, post, caplog, slept, body, closed, still_open,
    ):
        result, client = self._unwind(post, caplog, body)
        assert result.status == "rollback_failed"
        assert result.error.startswith(
            "YES leg FoK not filled: status=canceled; rollback error: "
            "Unclassifiable V2 order response: fill_count="
        )
        assert result.error.endswith(f"; {still_open} of 5 NO contracts still open")
        (alert,) = self._orphan_alerts(caplog)
        assert (
            f"closed {closed} of the 5 NO contracts this pair bought on TICK-A,"
            f" so {still_open} of them are still open"
        ) in alert
        assert "check the account" in alert
        assert "up to" not in alert
        # No further order, no position read after the unwind, and no pause
        assert post.call_count == 3
        assert client.get_positions_without_preload_content.call_count == 2
        assert slept == []

    def test_a_time_series_partial_close_names_market_b(self, post, caplog, slept):
        # For a time-series pair the NO leg is market_b, so the unwind — and
        # its alert — are on TICK-B.
        result, client = self._unwind(
            post, caplog, v2_resp(2), spec=make_spec(pair_type="time_series"),
        )
        assert result.status == "rollback_failed"
        assert post.call_args_list[2].kwargs["body"]["ticker"] == "TICK-B"
        (alert,) = self._orphan_alerts(caplog)
        assert "closed 2 of the 5 NO contracts this pair bought on TICK-B" in alert
        assert "so 3 of them are still open" in alert
        assert result.error.endswith("; 3 of 5 NO contracts still open")
        assert post.call_count == 3
        assert client.get_positions_without_preload_content.call_count == 2
        assert slept == []

    def test_a_partial_close_after_an_ambiguous_no_leg_is_counted_too(
        self, post, caplog, slept,
    ):
        # The other way into the unwind: the NO leg's submission raised, and
        # the ledger moved by exactly -5, our NO buy. The partial unwind is
        # counted the same way, with no further order or read after it.
        post.side_effect = [TimeoutError("read timed out"), v2_resp(2)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-A", -5),
        )
        with caplog.at_level(logging.CRITICAL):
            result = _execute_one(client, make_spec())
        assert result.status == "rollback_failed"
        assert result.error.startswith("NO leg ambiguous error: read timed out;")
        assert result.error.endswith("; 3 of 5 NO contracts still open")
        (alert,) = self._orphan_alerts(caplog)
        assert "closed 2 of the 5 NO contracts this pair bought on TICK-A" in alert
        assert post.call_count == 2
        assert client.get_positions_without_preload_content.call_count == 3
        assert slept == []

    @pytest.mark.parametrize(
        "reply",
        [
            # 2xx objects whose fill count is unreadable or impossible: they
            # reach the count reader, which finds nothing it can use
            pytest.param({"order": {"order_id": "ord-1"}}, id="no-count"),
            pytest.param({"order": {"fill_count": 7}}, id="over-count"),
            pytest.param({"order": {"fill_count": -1}}, id="negative"),
            pytest.param({"order": {"fill_count": "three"}}, id="not-a-number"),
            pytest.param({"order": {"fill_count": True}}, id="bool"),
            pytest.param({"order": {"fill_count_fp": "NaN"}}, id="nan"),
            pytest.param({"order": {"fill_count_fp": "Infinity"}}, id="infinity"),
            pytest.param({"order": [{"fill_count": 3}]}, id="order-not-an-object"),
            # Counts the default 28-digit decimal precision would round: never
            # printed as an exact count that is wrong or a line that is huge
            pytest.param(
                {"order": {"fill_count_fp": "4.9999999999999999999999999999999"}},
                id="too-many-digits",
            ),
            pytest.param({"order": {"fill_count_fp": "1E-40"}}, id="tiny"),
            pytest.param({"order": {"fill_count_fp": "1E-999999999"}}, id="underflow"),
            # Replies that fail before the count reader: the classifier raises
            # something other than its ValueError (a signalling NaN), the body
            # is not a JSON object or not JSON at all, the exchange answered
            # with an error (the 409 kill is read as a clean non-fill only on
            # a fill_or_kill body, and the unwind is not one), or the
            # connection failed
            pytest.param({"order": {"fill_count_fp": "sNaN"}}, id="signalling-nan"),
            pytest.param([{"fill_count": 3}], id="list-body"),
            pytest.param("accepted", id="string-body"),
            pytest.param(3, id="number-body"),
            pytest.param(None, id="null-body"),
            pytest.param(JSONDecodeError("Expecting value", "", 0), id="not-json"),
            pytest.param(ApiException(status=400, reason="Bad Request"), id="http-400"),
            pytest.param(ApiException(status=500, reason="Server Error"), id="http-500"),
            pytest.param(fok_kill_error(), id="http-409-kill"),
            pytest.param(TimeoutError("read timed out"), id="timeout"),
            pytest.param(ConnectionError("connection reset"), id="connection-error"),
        ],
    )
    def test_an_unknown_count_still_says_up_to(self, post, caplog, slept, reply):
        result, client = self._unwind(post, caplog, reply)
        assert result.status == "rollback_failed"
        assert result.error.startswith(
            "YES leg FoK not filled: status=canceled; rollback error: "
        )
        assert "still open" not in result.error
        (alert,) = self._orphan_alerts(caplog)
        assert "up to 5 NO contracts on TICK-A" in alert
        assert "still open" not in alert
        assert len(alert) < 1000
        assert post.call_count == 3
        assert client.get_positions_without_preload_content.call_count == 2
        assert slept == []

    def test_a_carried_body_that_is_not_an_object_still_says_up_to(
        self, monkeypatch, caplog,
    ):
        # The reader checks the carried body's type rather than assuming it:
        # an error carrying something other than a dict leaves the count
        # unknown.
        submitted: list = []

        def submit(client, order, *, pace=None):
            submitted.append(order)
            raise trader._UnclassifiableV2Response("unclassifiable", [{"fill_count": 3}])

        monkeypatch.setattr(trader, "_submit_order_v2", submit)
        with caplog.at_level(logging.CRITICAL):
            result = trader._rollback_no_leg(
                MagicMock(), make_spec(), _no_leg(make_spec()), "YES leg failed",
            )
        assert result.status == "rollback_failed"
        (alert,) = self._orphan_alerts(caplog)
        assert "up to 5 NO contracts on TICK-A" in alert
        assert len(submitted) == 1

    def test_a_count_that_cannot_be_compared_still_says_up_to(
        self, monkeypatch, caplog,
    ):
        # Reading the count never costs the orphan alert: a NO-leg count that
        # a Decimal refuses to compare with (numpy's int64 here) falls back to
        # "up to" instead of raising out of _rollback_no_leg.
        np = pytest.importorskip("numpy")
        submitted: list = []

        def submit(client, order, *, pace=None):
            submitted.append(order)
            raise trader._UnclassifiableV2Response("unclassifiable", v2_resp(3))

        monkeypatch.setattr(trader, "_submit_order_v2", submit)
        spec = make_spec()
        leg = dataclasses.replace(_no_leg(spec), count=np.int64(5))
        with caplog.at_level(logging.CRITICAL):
            result = trader._rollback_no_leg(MagicMock(), spec, leg, "YES leg failed")
        assert result.status == "rollback_failed"
        (alert,) = self._orphan_alerts(caplog)
        assert "up to 5 NO contracts on TICK-A" in alert
        assert len(submitted) == 1

    def test_a_zero_fill_names_the_full_count(self, post, caplog):
        # A 2xx fill count of zero is a clean "canceled", not an unknown: the
        # unwind closed nothing, so all 5 are open, and the alert says so.
        result, _ = self._unwind(post, caplog, {"order": {"fill_count_fp": "0.00"}})
        assert result.status == "rollback_failed"
        assert result.error == (
            "YES leg FoK not filled: status=canceled; rollback FoK not filled:"
            " status=canceled"
        )
        (alert,) = self._orphan_alerts(caplog)
        assert "ROLLBACK NOT FILLED (status=canceled)" in alert
        assert "5 NO contracts on TICK-A" in alert
        assert "up to" not in alert

    def test_a_full_fill_is_rolled_back(self, post, caplog):
        result, _ = self._unwind(post, caplog, {"order": {"fill_count_fp": "5.00"}})
        assert result.status == "rolled_back"
        assert self._orphan_alerts(caplog) == []

    def test_a_partial_fill_error_carries_the_response(self, post):
        # _submit_order_v2 re-raises _v2_fill_status's error with the body
        # attached. It is still a ValueError with the same message, so a buy
        # leg's caller sees the error it always did.
        body = v2_resp(3)
        post.return_value = body
        with pytest.raises(ValueError) as exc_info:
            _submit_order_v2(MagicMock(), _build_rollback_order_v2(_no_leg(make_spec())))
        err = exc_info.value
        assert isinstance(err, trader._UnclassifiableV2Response)
        assert err.response is body
        assert str(err) == "Unclassifiable V2 order response: fill_count=3, requested=5"
        assert type(err.__cause__) is ValueError

    def test_the_carrying_error_survives_copy_and_pickle(self):
        err = trader._UnclassifiableV2Response("unclassifiable", {"order": {"fill_count": 3}})
        for clone in (copy.copy(err), copy.deepcopy(err), pickle.loads(pickle.dumps(err))):
            assert type(clone) is trader._UnclassifiableV2Response
            assert str(clone) == "unclassifiable"
            assert clone.response == {"order": {"fill_count": 3}}

    def test_a_partial_no_leg_fill_is_still_judged_by_the_position(self, post):
        # A partial fill on the NO leg (a fill-or-kill buy) is not a partial
        # unwind: it raises into the position check, which cannot attribute
        # a -3 move to a 5-contract order, so nothing further is sent.
        post.side_effect = [v2_resp(3)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-A", -3),
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "fill_count=3" in result.error
        assert post.call_count == 1

    def test_a_partial_yes_leg_fill_is_never_rolled_back(self, post):
        # A partial fill on the YES leg must not read as a non-fill: the
        # position moved +3, which this 5-contract order cannot explain, so
        # the NO leg is left in place for a human rather than unwound.
        post.side_effect = [v2_resp(5), v2_resp(3)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-B", 3),
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "fill_count=3" in result.error
        assert post.call_count == 2



class TestV2NoMappingBackstop:
    """_confirm_v2_no_mapping: on a process's first V2 NO fill, the NO leg's
    position must change by exactly -no_leg.count (a held NO reads negative).
    Any other change stops the pair at manual_review, with the YES leg not
    sent and the NO leg left in place. The change is measured from
    _execute_one's NO baseline, never the holding: an existing long position
    must not fake a disproof, nor a short one hide a real one. The check's own
    read is single-shot (_position_count_once) but uses the same client
    method, so call counts include it. Same-title default, so the NO leg is
    TICK-A (TestTimeSeriesLegOrder covers TICK-B); each case starts with the
    latch False, as a fresh process does."""

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_delta_of_minus_x_confirms_the_mapping_and_completes_the_pair(self, post):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                    # before_no baseline: flat
            None,                    # before_yes baseline
            ("TICK-A", -5),          # backstop: moved by -5, our 5-contract NO buy
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        # YES leg still went out — the check must not disturb the state machine
        assert post.call_count == 2
        assert trader._V2_NO_MAPPING_CONFIRMED is True

    def test_external_long_position_still_confirms_via_the_delta(self, post):
        # Regression: the account already holds +100 YES on market A from an
        # earlier run or a manual trade. Our 10-contract NO buy nets it to +90,
        # which is POSITIVE — the old absolute-sign test read that as "mapping
        # disproven" and halted the pair at manual_review with a real, unhedged
        # NO-leg position open. The delta (-10) is unambiguous and confirms.
        post.side_effect = [v2_resp(10, 10), v2_resp(10, 10)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            ("TICK-A", 100),         # before_no: external long YES
            None,                    # before_yes baseline
            ("TICK-A", 90),          # backstop: +100 - 10 = +90
        ))
        assert _execute_one(client, make_spec(x=10)).status == "executed"
        assert post.call_count == 2
        assert trader._V2_NO_MAPPING_CONFIRMED is True

    def test_external_short_position_cannot_mask_a_disproof(self, post, caplog):
        # The mirror-image regression: the account is already -100 on market A,
        # and the fill moved it the WRONG way (+5, i.e. the ask opened YES
        # exposure). The reading is still negative in absolute terms, so the
        # old sign test would have CONFIRMED — and latched that false
        # confirmation for the rest of the process. The delta (+5, not -5)
        # disproves.
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            ("TICK-A", -100),        # before_no: external short
            None,                    # before_yes baseline
            ("TICK-A", -95),         # backstop: moved +5, the wrong direction
        ))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "mapping disproven" in result.error
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
        assert_disproof_names_the_remedy(caplog)
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        # YES leg was never submitted, so a false confirmation cannot have latched
        assert post.call_count == 1

    def test_confirmation_latches_for_the_process(self, post):
        post.side_effect = [v2_resp(5)] * 4
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", -5),   # trade 1: baselines + backstop
            None, None,                   # trade 2: baselines only
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert _execute_one(client, make_spec()).status == "executed"
        assert post.call_count == 4
        # One BACKSTOP read across two trades: the mapping is a property of the
        # exchange, so it costs one read per PROCESS, not per trade. The other
        # four reads are the two ambiguity baselines _execute_one takes up
        # front on every pair (2 trades x 2 legs) — those are unconditional and
        # unrelated to the backstop.
        assert client.get_positions_without_preload_content.call_count == 5

    def test_latched_state_skips_the_lookup_entirely(self, post, monkeypatch):
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", True)
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock()
        assert _execute_one(client, make_spec()).status == "executed"
        # Exactly the two up-front ambiguity baselines and nothing else — no
        # third read, i.e. the backstop was skipped entirely.
        assert client.get_positions_without_preload_content.call_count == 2

    def test_wrong_direction_delta_disproves_the_mapping_and_stops_the_pair(
        self, post, caplog
    ):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 5),    # flat -> +5: the ask opened YES
        ))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "mapping disproven" in result.error
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
        assert_disproof_names_the_remedy(caplog)
        # A disproven mapping must NOT latch — nothing was confirmed
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_wrong_magnitude_delta_disproves_the_mapping(self, post):
        # Right direction, wrong size: a -1 move cannot be our 5-contract buy,
        # so the mapping is not proven and the pair must not proceed.
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", -1),
        ))
        assert _execute_one(client, make_spec()).status == "manual_review"
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_persistent_zero_delta_disproves_the_mapping(self, post, monkeypatch):
        # An unmoved ledger on EVERY read after a "filled" NO buy is
        # contradictory (fill reported, position unchanged) — still
        # manual_review, but only after the lag re-reads have had their chance.
        slept = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", 0), ("TICK-A", 0), ("TICK-A", 0),
        ))
        assert _execute_one(client, make_spec()).status == "manual_review"
        # Two up-front baselines, then the backstop's first read and one
        # re-read after each pause of its schedule
        assert client.get_positions_without_preload_content.call_count == 6
        assert slept == list(config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS)

    def test_transient_zero_delta_recovers_on_reread_and_latches(self, post, monkeypatch):
        # Regression (adversarial review): an unmoved FIRST read is usually
        # read-after-write lag in the positions ledger, not disproof. The
        # re-read sees the real move, latches, and the pair completes — instead
        # of falsely halting at manual_review with a real unhedged NO-leg
        # position left open on an unattended run.
        slept = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                          # before_no baseline
            None,                          # before_yes baseline
            ("TICK-A", 0),                 # lagging first read -> delta 0
            ("TICK-A", -5.0),              # ledger catches up -> delta -5
        ))
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is True
        assert slept == [config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS[0]]

    def test_zero_then_failed_reread_proceeds_unlatched(self, post, monkeypatch):
        # A zero delta then a failed re-read is UNKNOWN, not disproven —
        # proceed to YES leg unlatched, same as a failed first read.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                          # before_no baseline
            None,                          # before_yes baseline
            ("TICK-A", 0),
            RuntimeError("positions endpoint down"),
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_disproven_mapping_submits_no_leg_b_and_no_rollback(self, post):
        post.side_effect = [v2_resp(5), v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 5),
        ))
        assert _execute_one(client, make_spec()).status == "manual_review"
        # Only NO leg went out: YES leg is not submitted (it would hedge a
        # position we don't hold) and NO unwind is attempted (the unwind is a
        # bid resting on the same disproven hypothesis).
        assert post.call_count == 1
        only_body = post.call_args_list[0].kwargs["body"]
        assert only_body["ticker"] == "TICK-A"
        assert only_body["side"] == "ask"
        # NO leg's position is left exactly as it is for a human to flatten
        client.create_order_without_preload_content.assert_not_called()

    def test_failed_position_lookup_proceeds_without_latching(self, post, caplog):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        with caplog.at_level(logging.INFO, logger="root"):
            assert _execute_one(client, make_spec()).status == "executed"
        # Unknown is not disproven — the fill itself was confirmed by the FoK
        # response, so the pair proceeds and one flaky read cannot stall trading
        assert post.call_count == 2
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_missing_baseline_makes_the_delta_unknown_not_disproven(self, post):
        # The backstop's own read succeeds, but the NO-leg BASELINE failed, so
        # no delta exists. That is unknown — proceed unlatched rather than
        # judging the absolute reading, which is exactly what this check is not
        # allowed to do.
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            RuntimeError("baseline lookup failed"),   # before_no -> None
            None,                                     # before_yes baseline
            ("TICK-A", -5),                           # backstop read succeeds
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_backstop_read_is_single_shot_and_never_retried(self, post):
        # The backstop's read is the ONLY blocking call inside the window where
        # NO leg is filled and unhedged, so it must not carry
        # api_call_with_retry's ~62s of backoff — and because a failing endpoint
        # never latches, every V2 trade in a 429 storm would pay it. A 429 here
        # costs exactly ONE request and one unlatched pass; a retried read would
        # issue up to six and sleep between them.
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, _StatusError(429),
        ))
        with patch.object(_http.time, "sleep") as sleep:
            assert _execute_one(client, make_spec()).status == "executed"
        assert client.get_positions_without_preload_content.call_count == 3
        sleep.assert_not_called()
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_backstop_source_carries_no_retry_wrapper(self):
        # Structural counterpart to the call-count case above: neither the
        # backstop nor the single-shot reader it uses may reach for the
        # backoff wrapper. The retried sibling _position_count still does (see
        # TestPositionCountRetry) — the asymmetry is the point.
        assert not _calls_retry_wrapper(trader._confirm_v2_no_mapping)
        assert not _calls_retry_wrapper(trader._position_count_once)
        assert _calls_retry_wrapper(trader._position_count)

    def test_check_rearms_after_a_failed_lookup(self, post):
        # Unlatched means the NEXT V2 NO fill re-checks: the first trade's
        # lookup fails, the second's succeeds and confirms.
        post.side_effect = [v2_resp(5)] * 4
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                              # trade 1: before_no baseline
            None,                              # trade 1: before_yes baseline
            RuntimeError("lookup failed"),     # trade 1: backstop read fails
            None,                              # trade 2: before_no baseline
            None,                              # trade 2: before_yes baseline
            ("TICK-A", -5),                    # trade 2: backstop confirms
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert _execute_one(client, make_spec()).status == "executed"
        # 4 baselines (2 trades x 2 legs) + 2 backstop reads — the backstop
        # genuinely RE-ARMED after the first trade's failed lookup
        assert client.get_positions_without_preload_content.call_count == 6
        assert trader._V2_NO_MAPPING_CONFIRMED is True


class _SharedLedgerExchange:
    """A V2 order endpoint and positions ledger that concurrent pairs share.

    Each filled order moves the account's signed position on its ticker: a
    bid by +count, and an ask by ask_sign * count. An ask_sign of -1 is the
    mapping the bot assumes (an ask opens NO, a short YES); +1 is a wrong
    mapping the backstop must disprove. A ticker in `killed` gets the
    exchange's HTTP 409 kill response instead of a fill. A ticker in
    `unreadable` reads fine the first time (the pair's baseline) and fails on
    every later read (the mapping check's). `ask_reply` is what a filled ask
    answers: "fill" (a normal 2xx), "raise" (the fill lands, then the client
    sees a transport error) or "no-fill-count" (the fill lands, then a 2xx
    body the trader cannot classify). `on_post`, if given, is called with
    each body before the exchange handles it.
    """

    def __init__(
        self, *, ask_sign: int = -1, killed=(), unreadable=(), ask_reply: str = "fill",
        on_post=None,
    ):
        self.ask_sign = ask_sign
        self.killed = set(killed)
        self.unreadable = set(unreadable)
        self.ask_reply = ask_reply
        self.on_post = on_post
        self.positions: dict[str, float] = {}
        self.posts: list[dict] = []
        self.reads: list[str] = []
        self._lock = threading.Lock()

    def post(self, client, method, path, body):
        """Stand-in for trader.signed_request_json."""
        with self._lock:
            self.posts.append(dict(body))
        if self.on_post is not None:
            self.on_post(body)
        if body["ticker"] in self.killed:
            raise fok_kill_error()
        count = int(Decimal(body["count"]))
        sign = 1 if body["side"] == "bid" else self.ask_sign
        with self._lock:
            self.positions[body["ticker"]] = (
                self.positions.get(body["ticker"], 0) + sign * count
            )
        if body["side"] == "ask" and self.ask_reply == "raise":
            raise ConnectionError("connection reset after the fill")
        if body["side"] == "ask" and self.ask_reply == "no-fill-count":
            return {"order": {"order_id": "ord-1"}}
        return v2_resp(count, count)

    def read(self, **kwargs):
        """Stand-in for client.get_positions_without_preload_content."""
        ticker = kwargs["ticker"]
        with self._lock:
            earlier = self.reads.count(ticker)
            self.reads.append(ticker)
            position = self.positions.get(ticker, 0)
        if ticker in self.unreadable and earlier:
            raise RuntimeError("positions endpoint down")
        return positions_resp(ticker, position)

    def client(self) -> MagicMock:
        """A client whose position reads come from this ledger."""
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=self.read)
        return client


class TestV2MappingDisproofStopsTheRun:
    """Once the backstop disproves the V2 NO-leg mapping, the process sends
    nothing more: _V2_NO_MAPPING_DISPROVEN is set, every later pair stops
    before it reads a position or builds an order (status "failed", nothing
    submitted), and execute_trades runs pairs one at a time until the mapping
    is confirmed or disproven, so a disproof costs one wrong-side position
    instead of one per pair in flight.

    Without the latch, 12 pairs whose NO fills all moved the position the wrong
    way gave 12 manual_review results, 12 NO-leg POSTs and 12 CRITICALs. And
    concurrency alone is enough to lose the protection: on the first live run
    (2026-09-28 02:26) all 7 NO legs were sent before the first mapping check
    finished, so a latch that pairs check only when they start would have
    stopped none of them."""

    @staticmethod
    def _specs(n: int, types=("same_title", "time_series")) -> list:
        """n specs on distinct tickers, cycling through the given pair types."""
        specs = []
        for i in range(n):
            spec = make_spec(title=f"pair {i}", pair_type=types[i % len(types)])
            spec.pair.market_a.ticker = f"TICK-A{i}"
            spec.pair.market_b.ticker = f"TICK-B{i}"
            specs.append(spec)
        return specs

    @pytest.mark.parametrize(
        "types",
        [("same_title", "time_series"), ("time_series", "same_title")],
        ids=["same-title-disproves", "time-series-disproves"],
    )
    def test_a_disproof_stops_every_later_pair_before_its_no_leg(
        self, monkeypatch, caplog, types,
    ):
        # Every NO fill moves the position the wrong way. The first pair's
        # check disproves the mapping; the other eleven, of both pair types,
        # send nothing and read nothing.
        specs = self._specs(12, types)
        # The first NO leg waits a moment for any other pair to start. One at
        # a time none can, so it waits the full 0.2 s; were pairs let run
        # together, others would start and send NO legs meanwhile.
        index = {id(spec): i for i, spec in enumerate(specs)}
        another_started = threading.Event()
        seen_during_first_post: list[bool] = []

        def first_post_waits(body):
            if not seen_during_first_post:
                seen_during_first_post.append(another_started.wait(0.2))

        exchange = _SharedLedgerExchange(ask_sign=+1, on_post=first_post_waits)
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        real = trader._execute_one

        def run(client, spec):
            if index[id(spec)] > 0:
                another_started.set()
            return real(client, spec)

        monkeypatch.setattr(trader, "_execute_one", run)
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(exchange.client(), specs, dry_run=False)

        assert seen_during_first_post == [False]

        assert [r.spec for r in results] == specs
        assert results[0].status == "manual_review"
        assert "mapping disproven" in results[0].error
        assert [r.status for r in results[1:]] == ["failed"] * 11
        assert {r.error for r in results[1:]} == {
            "NO leg not sent: V2 NO-leg mapping disproven earlier in this run;"
            " nothing submitted"
        }
        # One order in the whole run: the first pair's NO leg
        assert len(exchange.posts) == 1
        assert exchange.posts[0]["ticker"] == _no_leg(specs[0]).market.ticker
        assert exchange.posts[0]["side"] == "ask"
        # Only the first pair read the account: its two baselines and the check
        assert len(exchange.reads) == 3
        assert trader._V2_NO_MAPPING_DISPROVEN is True
        assert trader._V2_NO_MAPPING_CONFIRMED is False

        disproofs = [
            r for r in caplog.records
            if r.levelno == logging.CRITICAL and "DISPROVEN" in r.getMessage()
        ]
        assert len(disproofs) == 1
        assert "The rest of this run is stopped" in disproofs[0].getMessage()
        assert not [
            r for r in caplog.records
            if r.levelno == logging.CRITICAL and r not in disproofs
        ]
        stopped = [r.getMessage() for r in caplog.records if "Not sending" in r.getMessage()]
        assert len(stopped) == 11
        assert all("A=TICK-A" in m and "B=TICK-B" in m for m in stopped)

    def test_pairs_run_one_at_a_time_until_the_mapping_is_confirmed(self, monkeypatch):
        # Pair 0's NO leg is killed and pair 1's check cannot read the
        # account, so neither settles the mapping and each runs alone: while
        # each is running it waits a moment for the next pair to start, and
        # none does. Pair 2 confirms the mapping, and pairs 3-5 then start
        # together: they can only get past the barrier together, so run one
        # at a time the first of them would wait out its timeout and fail.
        specs = self._specs(6)
        pair2_no = _no_leg(specs[2]).market.ticker
        exchange = _SharedLedgerExchange(
            killed=[_no_leg(specs[0]).market.ticker],
            unreadable=[_no_leg(specs[1]).market.ticker],
        )
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        # A roomy pacer, so no pair waits on the write limit in real time
        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", trader._WritePacer(1000, 100))
        index = {id(spec): i for i, spec in enumerate(specs)}
        started = [threading.Event() for _ in specs]
        next_started_early: dict[int, bool] = {}
        together = threading.Barrier(3, timeout=5)
        real = trader._execute_one

        def run(client, spec):
            i = index[id(spec)]
            started[i].set()
            if i < 2:
                next_started_early[i] = started[i + 1].wait(0.2)
            if i >= 3:
                together.wait()
            return real(client, spec)

        monkeypatch.setattr(trader, "_execute_one", run)
        results = execute_trades(exchange.client(), specs, dry_run=False)

        assert [r.status for r in results] == ["failed"] + ["executed"] * 5
        assert next_started_early == {0: False, 1: False}
        # Pairs 3-5 read nothing until pair 2's check (its second read of its
        # NO ticker) had confirmed the mapping
        reads = exchange.reads
        confirmed_at = [k for k, t in enumerate(reads) if t == pair2_no][1]
        later = {f"TICK-A{i}" for i in (3, 4, 5)} | {f"TICK-B{i}" for i in (3, 4, 5)}
        assert all(k > confirmed_at for k, t in enumerate(reads) if t in later)
        assert trader._V2_NO_MAPPING_CONFIRMED is True
        assert trader._V2_NO_MAPPING_DISPROVEN is False

    def test_the_next_pair_starts_as_soon_as_the_mapping_is_confirmed(self, monkeypatch):
        # Pair 0 confirms the mapping, then its YES leg waits for pair 1 to
        # start. Pair 1 does, while pair 0 is still in its YES leg: the wait
        # ends at the verdict, not at the end of pair 0. Pair 0's NO POST
        # takes a moment, so execute_trades is already waiting on pair 0
        # (the mapping still unverified) when the verdict comes.
        specs = self._specs(2)
        pair1_started = threading.Event()
        yes_saw_pair1: list[bool] = []

        def yes_leg_waits(body):
            if body["side"] == "ask" and body["ticker"] == _no_leg(specs[0]).market.ticker:
                time.sleep(0.2)
            if body["side"] == "bid" and body["ticker"] == _yes_leg(specs[0]).market.ticker:
                yes_saw_pair1.append(pair1_started.wait(5))

        exchange = _SharedLedgerExchange(on_post=yes_leg_waits)
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        real = trader._execute_one

        def run(client, spec):
            if spec is specs[1]:
                pair1_started.set()
            return real(client, spec)

        monkeypatch.setattr(trader, "_execute_one", run)
        results = execute_trades(exchange.client(), specs, dry_run=False)
        assert [r.status for r in results] == ["executed", "executed"]
        assert yes_saw_pair1 == [True]

    def test_killed_no_legs_do_not_use_up_the_one_at_a_time_phase(self, monkeypatch):
        # The first three NO legs are killed (no verdict, one round trip
        # each) and the mapping is wrong. Pair 3 still runs alone, disproves
        # the mapping, and the other eight send nothing: one wrong-side
        # position, where a count of three pairs without a verdict started
        # the rest together and opened one per pair in flight.
        specs = self._specs(12)
        killed = [_no_leg(spec).market.ticker for spec in specs[:3]]
        pair3_no = _no_leg(specs[3]).market.ticker
        index = {id(spec): i for i, spec in enumerate(specs)}
        # Pair 3's NO leg waits a moment for a later pair to start. Run alone,
        # none does; started together, they would send NO legs meanwhile.
        later_started = threading.Event()
        seen_during_pair3: list[bool] = []

        def pair3_waits(body):
            if body["ticker"] == pair3_no:
                seen_during_pair3.append(later_started.wait(0.2))

        exchange = _SharedLedgerExchange(ask_sign=+1, killed=killed, on_post=pair3_waits)
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        real = trader._execute_one

        def run(client, spec):
            if index[id(spec)] > 3:
                later_started.set()
            return real(client, spec)

        monkeypatch.setattr(trader, "_execute_one", run)
        results = execute_trades(exchange.client(), specs, dry_run=False)
        assert seen_during_pair3 == [False]
        assert [r.status for r in results] == (
            ["failed"] * 3 + ["manual_review"] + ["failed"] * 8
        )
        assert all("not sent" in r.error for r in results[4:])
        assert [b["ticker"] for b in exchange.posts] == (
            killed + [_no_leg(specs[3]).market.ticker]
        )
        wrong_side = [t for t, v in exchange.positions.items() if v]
        assert wrong_side == [_no_leg(specs[3]).market.ticker]

    def test_the_time_budget_ends_the_phase_even_while_a_pair_is_stuck(
        self, monkeypatch, caplog,
    ):
        # Every check fails to read the account, so no pair gives a verdict.
        # Pairs 0 and 1 run alone. Pair 2's NO POST hangs until another pair
        # starts: once the (shortened) budget has passed, execute_trades stops
        # waiting for it, logs one WARNING, and starts pairs 3-5 together.
        monkeypatch.setattr(trader, "V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS", 0.5)
        specs = self._specs(6)
        stuck = _no_leg(specs[2]).market.ticker
        index = {id(spec): i for i, spec in enumerate(specs)}
        pair3_started = threading.Event()
        stuck_released: list[bool] = []

        def hang_until_pair3(body):
            if body["ticker"] == stuck:
                stuck_released.append(pair3_started.wait(5))

        exchange = _SharedLedgerExchange(
            unreadable=[_no_leg(spec).market.ticker for spec in specs],
            on_post=hang_until_pair3,
        )
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", trader._WritePacer(1000, 100))
        events: list[tuple[str, int]] = []
        lock = threading.Lock()
        together = threading.Barrier(3, timeout=5)
        real = trader._execute_one

        def run(client, spec):
            i = index[id(spec)]
            with lock:
                events.append(("start", i))
            if i == 3:
                pair3_started.set()
            try:
                if i >= 3:
                    together.wait()
                return real(client, spec)
            finally:
                with lock:
                    events.append(("end", i))

        monkeypatch.setattr(trader, "_execute_one", run)
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(exchange.client(), specs, dry_run=False)

        assert [r.status for r in results] == ["executed"] * 6
        assert stuck_released == [True]
        assert events[:5] == [("start", 0), ("end", 0), ("start", 1), ("end", 1), ("start", 2)]
        # Pair 3 started while pair 2 was still stuck
        assert events.index(("start", 3)) < events.index(("end", 2))
        warnings = [
            r.getMessage() for r in caplog.records
            if "still unverified after 0.5s of running pairs one at a time" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        assert trader._V2_NO_MAPPING_DISPROVEN is False

    def test_a_disproof_names_the_pairs_that_went_ahead_unchecked(self, monkeypatch, caplog):
        # Pair 0's check cannot read the account, so it sends its YES leg on
        # the wrong mapping and reports "executed". Pair 1 disproves the
        # mapping, and its CRITICAL names pair 0's position as well.
        specs = self._specs(4)
        pair0_no = _no_leg(specs[0]).market.ticker
        exchange = _SharedLedgerExchange(ask_sign=+1, unreadable=[pair0_no])
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(exchange.client(), specs, dry_run=False)
        assert [r.status for r in results] == ["executed", "manual_review", "failed", "failed"]
        assert trader._V2_UNCHECKED_NO_LEGS == [pair0_no]
        disproof = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.CRITICAL and "DISPROVEN" in r.getMessage()
        ]
        assert len(disproof) == 1
        assert f"check the positions on {pair0_no} too" in disproof[0]
        # Pair 0's two legs and pair 1's NO leg; nothing after the disproof
        assert [b["ticker"] for b in exchange.posts] == [
            pair0_no, _yes_leg(specs[0]).market.ticker, _no_leg(specs[1]).market.ticker,
        ]

    def test_an_ambiguous_disproof_names_the_pairs_that_went_ahead_unchecked(
        self, monkeypatch, caplog,
    ):
        # Pair 0's check cannot read the account, so it goes ahead. Pair 1's
        # NO POST raises after a wrong-way fill, and the CRITICAL of that
        # ambiguous-leg disproof names pair 0's position as well.
        specs = self._specs(3)
        pair0_no = _no_leg(specs[0]).market.ticker
        pair1_no = _no_leg(specs[1]).market.ticker
        exchange = _SharedLedgerExchange(ask_sign=+1, unreadable=[pair0_no])
        post = exchange.post

        def raise_on_pair1(client, method, path, body):
            reply = post(client, method, path, body)
            if body["ticker"] == pair1_no:
                raise ConnectionError("connection reset after the fill")
            return reply

        monkeypatch.setattr(trader, "signed_request_json", raise_on_pair1)
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(exchange.client(), specs, dry_run=False)
        assert [r.status for r in results] == ["executed", "manual_review", "failed"]
        assert "treated as disproven" in results[1].error
        stop = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.CRITICAL and "treated as disproven" in r.getMessage()
        ]
        assert len(stop) == 1
        assert f"check the positions on {pair0_no} too" in stop[0]

    @pytest.mark.parametrize("ask_reply", ["raise", "no-fill-count"])
    def test_an_ambiguous_no_leg_that_moved_the_wrong_way_stops_the_run(
        self, monkeypatch, caplog, ask_reply,
    ):
        # The NO POST fills the wrong way and then either raises or answers a
        # body the trader cannot read, so the pair takes the ambiguous path
        # and the mapping check never runs. The position moved by +5, which a
        # NO buy cannot do: the run stops there, with one wrong-side position.
        specs = self._specs(5)
        exchange = _SharedLedgerExchange(ask_sign=+1, ask_reply=ask_reply)
        monkeypatch.setattr(trader, "signed_request_json", exchange.post)
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(exchange.client(), specs, dry_run=False)
        assert results[0].status == "manual_review"
        assert "treated as disproven" in results[0].error
        assert [r.status for r in results[1:]] == ["failed"] * 4
        assert len(exchange.posts) == 1
        assert trader._V2_NO_MAPPING_DISPROVEN is True
        assert any(
            r.levelno == logging.CRITICAL
            and "the rest of this run is stopped" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize(
        "confirmed, after", [(True, ("TICK-A", 5)), (False, RuntimeError("down"))],
        ids=["already-confirmed", "position-unknown"],
    )
    def test_an_ambiguous_no_leg_stops_the_run_only_on_a_known_move_while_unconfirmed(
        self, monkeypatch, confirmed, after,
    ):
        # With the mapping already confirmed, an unexplained move is an
        # unrelated trade, not a disproof; with the position unknown there is
        # no move to judge. Either way the pair stops at manual_review alone.
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", confirmed)
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(side_effect=ConnectionError("reset")),
        )
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, after,
        ))
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "treated as disproven" not in result.error
        assert trader._V2_NO_MAPPING_DISPROVEN is False

    def test_a_zero_is_re_read_on_the_schedule_until_it_moves(self, monkeypatch):
        slept = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(side_effect=[v2_resp(5), v2_resp(5)]),
        )
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", 0), ("TICK-A", -5),
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is True
        assert slept == list(config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS[:2])

    def test_a_failed_re_read_is_unknown_and_on_record(self, monkeypatch):
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(side_effect=[v2_resp(5), v2_resp(5)]),
        )
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", 0), RuntimeError("positions endpoint down"),
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        assert trader._V2_NO_MAPPING_DISPROVEN is False
        assert trader._V2_UNCHECKED_NO_LEGS == ["TICK-A"]

    def test_pairs_start_together_once_the_mapping_is_confirmed(self, monkeypatch):
        # A confirmed mapping is not re-checked, so nothing makes a pair wait
        # for another: all four must be running at once to get past the
        # barrier.
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", True)
        specs = self._specs(4)
        together = threading.Barrier(4, timeout=5)

        def run(client, spec):
            together.wait()
            return TradeResult(spec=spec, status="executed")

        monkeypatch.setattr(trader, "_execute_one", run)
        results = execute_trades(MagicMock(), specs, dry_run=False)
        assert [r.status for r in results] == ["executed"] * 4

    @pytest.mark.parametrize("pair_type", ["same_title", "time_series"])
    def test_a_stopped_pair_reads_builds_and_sends_nothing(
        self, monkeypatch, caplog, pair_type,
    ):
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", True)
        post = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", post)
        builder = MagicMock(side_effect=AssertionError("an order was built"))
        monkeypatch.setattr(trader, "_build_no_order_any", builder)
        client = MagicMock()
        pacer = trader._ORDER_WRITE_PACER
        with caplog.at_level(logging.WARNING, logger="root"):
            result = _execute_one(client, make_spec(pair_type=pair_type))
        assert result.status == "failed"
        assert result.error == (
            "NO leg not sent: V2 NO-leg mapping disproven earlier in this run;"
            " nothing submitted"
        )
        builder.assert_not_called()
        post.assert_not_called()
        client.get_positions_without_preload_content.assert_not_called()
        client.create_order_without_preload_content.assert_not_called()
        # Not even a place on the write pacer was taken
        assert pacer._held == 0
        assert pacer._tokens == float(config.ORDER_WRITE_BURST)
        assert any(
            r.levelno == logging.WARNING and "Not sending" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize(
        "readings",
        [
            (("TICK-A", 5),),                    # moved the wrong way
            (("TICK-A", -1),),                   # right way, wrong size
            (("TICK-A", 0),) * 4,                # unmoved, even after every re-read
        ],
        ids=["wrong-direction", "wrong-size", "persistent-zero"],
    )
    def test_every_disproof_sets_the_latch(self, monkeypatch, readings):
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post = MagicMock(side_effect=[v2_resp(5)])
        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, *readings,
        ))
        assert _execute_one(client, make_spec()).status == "manual_review"
        assert trader._V2_NO_MAPPING_DISPROVEN is True
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        assert post.call_count == 1

    @pytest.mark.parametrize(
        "readings",
        [
            (("TICK-A", -5),),                                   # confirmed
            (RuntimeError("positions endpoint down"),),          # unknown
            (("TICK-A", 0), RuntimeError("positions endpoint down")),  # zero, then unknown
        ],
        ids=["confirmed", "unknown", "zero-then-unknown"],
    )
    def test_no_other_outcome_sets_the_latch(self, monkeypatch, readings):
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(side_effect=[v2_resp(5), v2_resp(5)]),
        )
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, *readings,
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_DISPROVEN is False

    @pytest.mark.parametrize("confirmed", [False, True], ids=["unverified", "also-confirmed"])
    def test_a_no_fill_after_another_pairs_disproof_stops_before_the_yes_leg(
        self, monkeypatch, caplog, confirmed,
    ):
        # This pair was already past the stop when another pair disproved the
        # mapping (the latch is set while its NO leg is in flight). That
        # cannot happen through execute_trades, which runs pairs one at a time
        # until the mapping is verified; it guards a caller that does not. The
        # disproof outweighs a confirmation from yet another pair.
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", confirmed)
        bodies: list[dict] = []

        def post(client, method, path, body):
            bodies.append(body)
            monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", True)
            return v2_resp(5)

        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock(get_positions_without_preload_content=positions_seq(None, None))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, make_spec(pair_type="time_series"))
        assert result.status == "manual_review"
        assert "disproven earlier in this run" in result.error
        assert "TICK-B" in result.error          # the time-series NO leg's market
        # The NO leg only: no YES leg, and no unwind on the disproven mapping
        assert [b["side"] for b in bodies] == ["ask"]
        # The two baselines only: no mapping-check read
        assert client.get_positions_without_preload_content.call_count == 2
        assert any(
            r.levelno == logging.CRITICAL and "earlier in this run" in r.getMessage()
            for r in caplog.records
        )

    def test_a_dry_run_is_unaffected_by_the_latch(self, monkeypatch):
        # A dry run never reaches _execute_one, so the latch changes nothing
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", True)
        post = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock()
        results = execute_trades(client, self._specs(3), dry_run=True)
        assert [r.status for r in results] == ["simulated"] * 3
        post.assert_not_called()
        client.get_positions_without_preload_content.assert_not_called()

    def test_which_states_run_pairs_one_at_a_time(self, monkeypatch):
        assert trader._v2_mapping_unverified() is True
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", True)
        assert trader._v2_mapping_unverified() is False
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_CONFIRMED", False)
        monkeypatch.setattr(trader, "_V2_NO_MAPPING_DISPROVEN", True)
        assert trader._v2_mapping_unverified() is False


class TestV2IsTheOnlyOrderPath:
    """Syntax-tree checks over the package that keep the retired
    /portfolio/orders endpoint out: no name contains "createorder" (ignoring
    underscores and case), only config.py reads ORDER_API_VERSION, and no
    non-docstring string starts with "/portfolio/orders" or
    "/trade-api/v2/portfolio/orders". Comments and docstrings do not count.
    Each finder is first run on sample snippets so it cannot pass by seeing
    nothing."""

    # Modules the walk must find: a moved package or an empty glob fails
    # rather than passing vacuously
    _EXPECTED = {
        "config.py", "trader.py", "main.py", "scanner.py", "strategy.py",
        "v2_probe.py", "_http.py", "auth.py",
    }

    @classmethod
    def _modules(cls) -> dict[str, ast.Module]:
        package = Path(inspect.getsourcefile(trader)).parent
        found = {
            path.name: ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted(package.glob("*.py"))
        }
        assert cls._EXPECTED <= found.keys(), sorted(found)
        return found

    @staticmethod
    def _names_and_strings(tree: ast.AST):
        """Every identifier in the code (names, attributes, imports and
        aliases), plus the attribute name a getattr/setattr/hasattr/delattr
        call passes as a string."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                yield node.id
            elif isinstance(node, ast.Attribute):
                yield node.attr
            elif isinstance(node, ast.ImportFrom):
                yield node.module or ""
                for alias in node.names:
                    yield alias.name
                    yield alias.asname or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name
                    yield alias.asname or ""
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in {"getattr", "setattr", "hasattr", "delattr"}
                  and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                  and isinstance(node.args[1].value, str)):
                yield node.args[1].value

    @classmethod
    def _create_order_uses(cls, tree: ast.AST) -> list[str]:
        """Every name containing "createorder", ignoring underscores and case
        (CreateOrderRequest, create_order*, batch_create_orders*)."""
        return [
            name for name in cls._names_and_strings(tree)
            if "createorder" in name.replace("_", "").lower()
        ]

    @classmethod
    def _order_api_version_uses(cls, tree: ast.AST) -> list[str]:
        return [name for name in cls._names_and_strings(tree) if name == "ORDER_API_VERSION"]

    # The retired endpoint's path, in the two spellings a request could use
    _RETIRED_ORDER_PATHS = ("/trade-api/v2/portfolio/orders", "/portfolio/orders")

    @staticmethod
    def _docstring_ids(tree: ast.AST) -> set[int]:
        """ids of the string constants that are docstrings: the first
        statement of a module, class or function, when it is a bare string."""
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    found.add(id(first.value))
        return found

    @classmethod
    def _retired_order_path_strings(cls, tree: ast.AST) -> list[str]:
        """Every non-docstring string constant (f-string pieces included)
        that starts, after leading whitespace, with the retired path."""
        docstrings = cls._docstring_ids(tree)
        return [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value.lstrip().startswith(cls._RETIRED_ORDER_PATHS)
        ]

    def test_the_finders_see_every_shape(self):
        for snippet in (
            "from kalshi_python_sync.models import CreateOrderRequest",
            "from kalshi_python_sync.models.create_order_request import X",
            "import kalshi_python_sync.models.create_order_request",
            "client.create_order_without_preload_content(create_order_request=o)",
            "client.batch_create_orders(orders)",
            "f = getattr(client, 'create_order')",
            "CreateOrderRequest(ticker='T')",
        ):
            assert self._create_order_uses(ast.parse(snippet)), snippet
        for snippet in (
            "from .config import ORDER_API_VERSION",
            "from .config import OTHER as ORDER_API_VERSION",
            "from .config import ORDER_API_VERSION as other",
            "if config.ORDER_API_VERSION != 'v2': pass",
            "x = ORDER_API_VERSION",
            "getattr(config, 'ORDER_API_VERSION')",
        ):
            assert self._order_api_version_uses(ast.parse(snippet)), snippet
        # Words in a docstring or comment are not code
        prose = ast.parse('"""create_order and CreateOrderRequest, ORDER_API_VERSION"""\n# create_order')
        assert not self._create_order_uses(prose)
        assert not self._order_api_version_uses(prose)

    def test_no_module_reaches_the_sdk_create_order_surface(self):
        hits = {
            name: uses for name, tree in self._modules().items()
            if (uses := self._create_order_uses(tree))
        }
        assert hits == {}

    def test_only_config_references_order_api_version(self):
        referencing = {
            name for name, tree in self._modules().items()
            if self._order_api_version_uses(tree)
        }
        # config.py itself must be found
        assert referencing == {"config.py"}

    def test_the_retired_path_finder_sees_every_shape(self):
        for snippet in (
            'PATH = "/portfolio/orders"',
            'signed_request_json(client, "POST", "/trade-api/v2/portfolio/orders", body=b)',
            'url = f"/portfolio/orders/{order_id}"',
            'x = "  /portfolio/orders/batched"',
            'def f():\n    """Doc."""\n    return "/portfolio/orders"',
        ):
            assert self._retired_order_path_strings(ast.parse(snippet)), snippet
        # A docstring naming the path is prose, not a request path
        for snippet in (
            '"""/portfolio/orders is retired."""',
            'def f():\n    """/portfolio/orders is retired."""',
            'class C:\n    """/trade-api/v2/portfolio/orders"""',
        ):
            assert not self._retired_order_path_strings(ast.parse(snippet)), snippet
        # The startup check's message names the path mid-string, so it is
        # not a hit
        source = textwrap.dedent(inspect.getsource(config.order_api_version_error))
        assert "/portfolio/orders" in source
        assert not self._retired_order_path_strings(ast.parse(source))

    def test_no_code_string_spells_the_retired_order_path(self):
        hits = {
            name: paths for name, tree in self._modules().items()
            if (paths := self._retired_order_path_strings(tree))
        }
        assert hits == {}


class TestSettleAwaitTargeting:
    """Regression (adversarial review): the settle wait targets only shards an
    accepted transfer was headed for — an unfundable deficit shard must not
    burn the timeout or miscast settled transfers as money-in-flight.

    DR-65 is the second half of the same idea, one step later: the in-flight
    VERDICT must come from the settlement observation, not from a shard's
    membership in accepted_cents. A shard whose accepted transfers all landed
    and merely fell short of its deficit gets a shortfall WARNING; only cents
    that were accepted and NOT observed to arrive keep the CRITICAL."""

    def _statuses(self, inactive_shard: int) -> dict:
        st = {i: shard_status() for i in (0, 1, 2)}
        st[inactive_shard] = shard_status(transfers_active=False)
        return st

    def test_await_targets_only_accepted_destinations(self, monkeypatch):
        # Deficits on shards 1 (transferable) and 2 (transfers inactive):
        # shard 2 can never settle, so the await must not include it.
        specs = [
            make_spec(shard_a=0, shard_b=1, cost_a=0.0, cost_b=2.00, title="s1"),
            make_spec(shard_a=0, shard_b=2, cost_a=0.0, cost_b=3.00, title="s2"),
        ]
        balances = {0: 10_000, 1: 0, 2: 0}
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(return_value=transfer_resp())
        )
        awaited = {}

        def fake_await(client, required):
            awaited.update(required)
            return {0: 9_800, 1: 200, 2: 0}

        monkeypatch.setattr(trader, "_await_transfer_settlement", fake_await)
        kept = ensure_shard_collateral(
            MagicMock(), specs, balances, self._statuses(inactive_shard=2)
        )
        assert set(awaited) == {1}
        assert [s.pair.canonical_title for s in kept] == ["s1"]

    def test_settled_but_short_transfer_is_a_shortfall_not_money_in_flight(
        self, monkeypatch, caplog
    ):
        # DR-65: shard 3 needs $100 and holds nothing; shards 0 and 1 each hold
        # $50 of surplus, but shard 1 cannot send. The shard-0 leg POSTs and
        # its $50 LANDS — the settle wait targets min(required, prior + moved)
        # = 5000c, sees it on the first read and returns immediately. Shard 3
        # is still short of its full $100, and the in-flight test used to be
        # bare membership in accepted_cents, so the run logged MONEY IS IN
        # FLIGHT while the same line's own `confirmed` map showed the money had
        # arrived. The money that WAS accepted is accounted for; why shard 3 is
        # still short (here: shard 1 could not send, and its own warning says
        # so) is not something the verdict knows or claims.
        specs = [make_spec(shard_a=3, shard_b=3, cost_a=50.00, cost_b=50.00,
                           title="needs shard 3")]
        balances = {0: 5_000, 1: 5_000, 3: 0}
        statuses = {0: shard_status(True), 1: shard_status(False),
                    3: shard_status(True)}
        post = MagicMock(return_value=transfer_resp("tidA"))
        monkeypatch.setattr(trader, "signed_request_json", post)
        monkeypatch.setattr(
            trader, "_await_transfer_settlement",
            lambda client, required: {0: 0, 1: 5_000, 3: 5_000},
        )
        with caplog.at_level(logging.INFO, logger="root"):
            kept = ensure_shard_collateral(MagicMock(), specs, balances, statuses)
        # Only the fundable leg was POSTed; the blocked one warned separately
        assert post.call_count == 1
        assert "MONEY IS IN FLIGHT" not in caplog.text
        assert not [r for r in caplog.records if r.levelno == logging.CRITICAL]
        warnings = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "OBSERVED to land but did not cover" in warnings
        # The shortfall itself is named: 10000c required, 5000c confirmed
        assert "{3: 5000}" in warnings
        # Degradation is unchanged — the underfunded spec is still dropped
        assert kept == []

    def test_accepted_transfer_that_never_lands_still_shouts(
        self, monkeypatch, caplog
    ):
        # Control for the case above, same plan and the same partial coverage —
        # but the accepted $50 is NOT observed on the shard. That is genuinely
        # ambiguous and must keep the CRITICAL.
        specs = [make_spec(shard_a=3, shard_b=3, cost_a=50.00, cost_b=50.00,
                           title="needs shard 3")]
        balances = {0: 5_000, 1: 5_000, 3: 0}
        statuses = {0: shard_status(True), 1: shard_status(False),
                    3: shard_status(True)}
        monkeypatch.setattr(
            trader, "signed_request_json",
            MagicMock(return_value=transfer_resp("tidA")),
        )
        monkeypatch.setattr(
            trader, "_await_transfer_settlement",
            lambda client, required: {0: 0, 1: 5_000, 3: 0},
        )
        with caplog.at_level(logging.INFO, logger="root"):
            kept = ensure_shard_collateral(MagicMock(), specs, balances, statuses)
        criticals = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL
        )
        assert "MONEY IS IN FLIGHT" in criticals
        assert "tidA" in criticals
        # A shard counted as in-flight is never ALSO reported as a settled
        # shortfall — the two verdicts partition the unfunded shards an
        # accepted transfer was headed for. (A shard whose transfer was
        # never accepted is in NEITHER set; see
        # test_no_false_money_in_flight_for_never_accepted_shards.)
        assert "OBSERVED to land but did not cover" not in caplog.text
        assert kept == []

    def test_partially_landed_transfer_is_still_in_flight(self, monkeypatch, caplog):
        # Half the accepted cents arrived, so part of the money is genuinely
        # unaccounted for: the CRITICAL must not be softened to a shortfall.
        specs = [make_spec(shard_a=3, shard_b=3, cost_a=50.00, cost_b=50.00,
                           title="needs shard 3")]
        balances = {0: 5_000, 1: 5_000, 3: 0}
        statuses = {0: shard_status(True), 1: shard_status(False),
                    3: shard_status(True)}
        monkeypatch.setattr(
            trader, "signed_request_json",
            MagicMock(return_value=transfer_resp("tidA")),
        )
        monkeypatch.setattr(
            trader, "_await_transfer_settlement",
            lambda client, required: {0: 0, 1: 5_000, 3: 2_500},
        )
        with caplog.at_level(logging.INFO, logger="root"):
            ensure_shard_collateral(MagicMock(), specs, balances, statuses)
        assert "MONEY IS IN FLIGHT" in caplog.text
        assert "OBSERVED to land but did not cover" not in caplog.text

    def test_no_false_money_in_flight_for_never_accepted_shards(self, monkeypatch, caplog):
        # Shard 2's transfer was never accepted (inactive) — its underfunding
        # is a plain drop, never the MONEY IS IN FLIGHT critical.
        specs = [make_spec(shard_a=0, shard_b=2, cost_a=0.0, cost_b=3.00)]
        balances = {0: 10_000, 2: 0}
        monkeypatch.setattr(
            trader, "signed_request_json",
            MagicMock(side_effect=AssertionError("nothing should be POSTed")),
        )
        with caplog.at_level(logging.INFO):
            kept = ensure_shard_collateral(
                MagicMock(), specs, balances, self._statuses(inactive_shard=2)
            )
        assert kept == []
        assert "MONEY IS IN FLIGHT" not in caplog.text


class TestExecuteTradesWorkerIsolation:
    """execute_trades must return one TradeResult per spec, in SUBMISSION order,
    even when one worker thread raises.

    The pool's `with` block already waits for every worker, so orders and
    rollbacks complete regardless; what a swallowed exception destroys is the
    RECORD — the Excel rows, the CRITICAL manual-review alert, and the
    EXIT_TRADES_NEED_ATTENTION exit code main._run_prod derives from these
    statuses.

    Run with the V2 NO-leg mapping unverified (pairs one at a time, since the
    stand-in workers never settle it) and confirmed (pairs concurrently).
    """

    @pytest.fixture(autouse=True, params=["unverified", "confirmed"])
    def _mode(self, request, monkeypatch):
        """Set whether the mapping is already confirmed for each test."""
        monkeypatch.setattr(
            trader, "_V2_NO_MAPPING_CONFIRMED", request.param == "confirmed",
        )

    @staticmethod
    def _specs() -> list:
        """Three specs whose NO-leg tickers are distinct, so a fake worker can
        single one out (make_spec pins the same TICK-A/TICK-B on every spec)."""
        specs = []
        for i in (1, 2, 3):
            spec = make_spec(title=f"pair {i}")
            spec.pair.market_a.ticker = f"TICK-A{i}"
            spec.pair.market_b.ticker = f"TICK-B{i}"
            specs.append(spec)
        return specs

    def test_execute_trades_isolates_one_raising_worker(self, monkeypatch, caplog):
        specs = self._specs()

        def fake(client, spec):
            if spec.pair.market_a.ticker == "TICK-A2":
                raise RuntimeError("boom")
            return TradeResult(spec=spec, status="executed")

        monkeypatch.setattr(trader, "_execute_one", fake)
        with caplog.at_level(logging.CRITICAL, logger="root"):
            results = execute_trades(MagicMock(), specs, dry_run=False)

        assert len(results) == 3
        # Submission order is the caller's contract: reporter rows and the
        # summary counts pair results[i] with specs[i].
        assert [r.spec for r in results] == specs
        assert results[0].status == "executed"
        assert results[2].status == "executed"
        assert results[1].status == "manual_review"
        assert "boom" in results[1].error

        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(criticals) == 1
        assert "TICK-A2" in criticals[0].getMessage()

    def test_execute_trades_all_ok_unchanged(self, monkeypatch):
        specs = self._specs()
        monkeypatch.setattr(
            trader, "_execute_one",
            lambda client, spec: TradeResult(spec=spec, status="executed"),
        )
        results = execute_trades(MagicMock(), specs, dry_run=False)
        assert [r.spec for r in results] == specs
        assert {r.status for r in results} == {"executed"}


class TestPreExecutionCheckLogging:
    """pre_execution_check must not log a second, reason-less line for a drop
    already logged (with its reason) by validate_pair_price — and must emit
    one INFO summary of how many selected pairs still qualify."""

    def test_each_drop_logged_once_with_summary(self, monkeypatch, caplog):
        keep = make_spec(title="keep me")
        drop = make_spec(title="drop me")

        def fake_validate(client, spec, *, settings):
            # The run's settings are always handed on, by keyword: with none
            # given, config.py's, resolved once by pre_execution_check
            assert isinstance(settings, config.LiveSettings)
            if spec.pair.canonical_title == "drop me":
                logging.warning(
                    "Pre-execution check failed for '%s' — gap no longer qualifies; dropping",
                    spec.pair.canonical_title,
                )
                return False
            return True

        monkeypatch.setattr(trader, "validate_pair_price", fake_validate)

        with caplog.at_level(logging.INFO, logger=""):
            result = pre_execution_check(MagicMock(), [keep, drop])

        assert result == [keep]

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        drop_warnings = [r for r in warnings if "drop me" in r.getMessage()]
        assert len(drop_warnings) == 1

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        summary = [
            r for r in infos
            if r.getMessage() == "Pre-execution check: 1 of 2 pair(s) still qualify"
        ]
        assert len(summary) == 1

    def test_exception_path_still_logs_and_drops(self, monkeypatch, caplog):
        keep = make_spec(title="keep me")
        drop = make_spec(title="drop me")

        def fake_validate(client, spec, *, settings):
            assert isinstance(settings, config.LiveSettings)
            if spec.pair.canonical_title == "drop me":
                raise RuntimeError("boom")
            return True

        monkeypatch.setattr(trader, "validate_pair_price", fake_validate)

        with caplog.at_level(logging.INFO, logger=""):
            result = pre_execution_check(MagicMock(), [keep, drop])

        assert result == [keep]

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        drop_warnings = [
            r for r in warnings if "raised" in r.getMessage() and "drop me" in r.getMessage()
        ]
        assert len(drop_warnings) == 1

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        summary = [
            r for r in infos
            if r.getMessage() == "Pre-execution check: 1 of 2 pair(s) still qualify"
        ]
        assert len(summary) == 1


class TestPreExecutionCheckSettings:
    """pre_execution_check re-checks every selected spec under ONE rule: the
    run's LiveSettings, handed to every validate_pair_price it submits to the
    pool. A run's override must never apply to one spec's re-check while
    another's reads config.py."""

    @staticmethod
    def _specs(n=4):
        return [make_spec(title=f"spec {i}") for i in range(n)]

    def test_one_settings_object_reaches_every_call(self, monkeypatch):
        explicit = config.LiveSettings(
            tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8, size_cap=1.0,
        )
        seen = []

        def fake_validate(client, spec, *, settings):
            seen.append(settings)
            return True

        def no_config(*a, **kw):
            raise AssertionError("pre_execution_check read config.py despite explicit settings")

        monkeypatch.setattr(trader, "validate_pair_price", fake_validate)
        monkeypatch.setattr(trader, "live_settings", no_config)
        specs = self._specs()
        # Specs come back in completion order, so compare them as a set
        result = pre_execution_check(MagicMock(), specs, settings=explicit)
        assert {id(s) for s in result} == {id(s) for s in specs}
        assert len(seen) == len(specs)
        assert all(s is explicit for s in seen)

    def test_no_settings_resolves_config_once_for_every_call(self, monkeypatch):
        seen = []
        resolved = []

        def fake_validate(client, spec, *, settings):
            seen.append(settings)
            return True

        def counting_live_settings():
            resolved.append(config.live_settings())
            return resolved[-1]

        monkeypatch.setattr(trader, "validate_pair_price", fake_validate)
        monkeypatch.setattr(trader, "live_settings", counting_live_settings)
        specs = self._specs()
        pre_execution_check(MagicMock(), specs)
        assert len(resolved) == 1
        assert len(seen) == len(specs)
        assert all(s is resolved[0] for s in seen)


class TestSameTitleWireIdentity:
    """Every field (except the random client_order_id) of the three V2
    bodies built for make_spec()'s default same-title spec (x=5, nA=0.40,
    pB=0.35, shard 0), as literal values: NO on market_a, YES on market_b,
    and a reduce-only immediate_or_cancel unwind."""

    def test_v2_bodies_are_unchanged(self):
        spec = make_spec()
        no_body = _build_no_order_v2(_no_leg(spec))
        yes_body = _build_yes_order_v2(_yes_leg(spec))
        rollback_body = _build_rollback_order_v2(_no_leg(spec))
        for body in (no_body, yes_body, rollback_body):
            uuid.UUID(body.pop("client_order_id"))
        assert no_body == {
            "ticker": "TICK-A", "side": "ask", "price": "0.5900", "count": "5.00",
            "time_in_force": "fill_or_kill",
            "self_trade_prevention_type": "taker_at_cross", "exchange_index": 0,
            "reduce_only": False, "post_only": False,
        }
        assert yes_body == {
            "ticker": "TICK-B", "side": "bid", "price": "0.3600", "count": "5.00",
            "time_in_force": "fill_or_kill",
            "self_trade_prevention_type": "taker_at_cross", "exchange_index": 0,
            "reduce_only": False, "post_only": False,
        }
        assert rollback_body == {
            "ticker": "TICK-A", "side": "bid", "price": "0.7200", "count": "5.00",
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross", "exchange_index": 0,
            "reduce_only": True, "post_only": False,
        }

    def test_same_title_baselines_read_market_a_then_market_b(
        self, v2_mapping_confirmed, monkeypatch
    ):
        # Nothing used to pin the ORDER of the two baseline reads; now that the
        # NO leg can be either market, pin it for both pair types (the
        # time-series mirror lives in TestTimeSeriesLegOrder).
        monkeypatch.setattr(trader, "signed_request_json",
                            MagicMock(side_effect=[v2_resp(5), v2_resp(5)]))
        tickers: list[str] = []

        def record(**kwargs):
            tickers.append(kwargs["ticker"])
            return positions_resp()

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=record)
        assert _execute_one(client, make_spec()).status == "executed"
        assert tickers == ["TICK-A", "TICK-B"]


class TestTimeSeriesLegOrder:
    """A time_series pair buys NO on the later contract (market_b) and YES on
    the earlier one (market_a), so TICK-B is sent first, read first, unwound
    on failure and checked by the NO-mapping check. Expected prices come from
    each leg's own scanned price, as in TestV2PriceMath."""

    @staticmethod
    def _ts_spec(**overrides) -> MagicMock:
        """Time-series spec: earlier YES ask 0.30 on TICK-A, later NO ask 0.40
        on TICK-B, legs on different shards so routing is observable."""
        kwargs = {"pair_type": "time_series", "pA": 0.30, "nB": 0.40, "shard_a": 2, "shard_b": 3}
        kwargs.update(overrides)
        return make_spec(**kwargs)

    @pytest.fixture
    def post(self, monkeypatch):
        """Mock of signed_request_json as imported into trader's namespace."""
        mock = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    def test_ordered_legs_puts_no_on_market_b_first(self):
        spec = self._ts_spec()
        no_leg, yes_leg = _ordered_legs(spec)
        assert no_leg.market is spec.pair.market_b
        assert (no_leg.side, no_leg.price_dollars, no_leg.count) == ("no", 0.40, spec.y)
        assert no_leg.label == "NO on Market B"
        assert yes_leg.market is spec.pair.market_a
        assert (yes_leg.side, yes_leg.price_dollars, yes_leg.count) == ("yes", 0.30, spec.x)
        assert yes_leg.label == "YES on Market A"

    def test_ordered_legs_keeps_market_a_as_the_no_leg_for_same_title(self):
        spec = make_spec()
        no_leg, yes_leg = _ordered_legs(spec)
        assert no_leg.market is spec.pair.market_a and no_leg.price_dollars == 0.40
        assert yes_leg.market is spec.pair.market_b and yes_leg.price_dollars == 0.35
        assert (no_leg.label, yes_leg.label) == ("NO on Market A", "YES on Market B")

    def test_legs_are_frozen(self):
        with pytest.raises(AttributeError):
            _no_leg(self._ts_spec()).count = 99

    def test_v2_submits_no_on_market_b_then_yes_on_market_a(
        self, v2_mapping_confirmed, post
    ):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        spec = self._ts_spec()
        assert _execute_one(MagicMock(), spec).status == "executed"
        first, second = (c.kwargs["body"] for c in post.call_args_list)
        # NO leg: an ask on TICK-B at 1 - (capped later NO ask), routed to
        # market_b's shard, sized by y (market_b's count)
        assert (first["ticker"], first["side"], first["exchange_index"]) == ("TICK-B", "ask", 3)
        assert first["price"] == _format_price(
            Decimal("1") - (Decimal("0.40") + BUY_SLIPPAGE_TICKS * Decimal("0.01"))
        )
        assert first["count"] == _format_count(spec.y)
        # YES leg: a bid on TICK-A at the capped earlier YES ask, shard 2, x
        assert (second["ticker"], second["side"], second["exchange_index"]) == ("TICK-A", "bid", 2)
        assert second["price"] == _format_price(
            Decimal("0.30") + BUY_SLIPPAGE_TICKS * Decimal("0.01")
        )
        assert second["count"] == _format_count(spec.x)
        for body in (first, second):
            assert body["reduce_only"] is False and body["exchange_index"] != -1

    def test_baselines_read_market_b_then_market_a(self, v2_mapping_confirmed, post):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        tickers: list[str] = []

        def record(**kwargs):
            tickers.append(kwargs["ticker"])
            return positions_resp()

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=record)
        assert _execute_one(client, self._ts_spec()).status == "executed"
        assert tickers == ["TICK-B", "TICK-A"]

    def test_yes_leg_kill_rolls_back_the_no_leg_on_market_b(
        self, v2_mapping_confirmed, post
    ):
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(5)]
        spec = self._ts_spec()
        assert _execute_one(MagicMock(), spec).status == "rolled_back"
        rollback = post.call_args_list[2].kwargs["body"]
        assert (rollback["ticker"], rollback["side"], rollback["exchange_index"]) == (
            "TICK-B", "bid", 3)
        assert rollback["reduce_only"] is True
        assert rollback["count"] == _format_count(spec.y)
        # Loss floor from the LATER contract's NO entry (nB), not from nA
        floor_cents = _rollback_floor_cents(_no_leg(spec))
        assert floor_cents == 40 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
        assert rollback["price"] == _format_price(
            Decimal("1") - Decimal(floor_cents) / Decimal("100")
        )

    def test_no_leg_kill_is_failed_before_anything_touches_market_a(
        self, v2_mapping_confirmed, post
    ):
        post.side_effect = [v2_resp(0)]
        result = _execute_one(MagicMock(), self._ts_spec())
        assert result.status == "failed"
        assert "NO leg FoK not filled" in result.error
        assert post.call_count == 1
        assert post.call_args_list[0].kwargs["body"]["ticker"] == "TICK-B"

    def test_no_leg_exception_delta_is_judged_on_market_b(
        self, v2_mapping_confirmed, post
    ):
        # The NO leg raised after actually filling: the delta on TICK-B (not
        # TICK-A) is -y, so the NO leg is unwound on TICK-B.
        post.side_effect = [TimeoutError("timeout"), v2_resp(5)]
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_no  (TICK-B)
            None,              # before_yes (TICK-A)
            ("TICK-B", -5),    # after_no   (TICK-B) — moved by -y
        )
        assert _execute_one(client, self._ts_spec()).status == "rolled_back"
        assert post.call_args_list[1].kwargs["body"]["ticker"] == "TICK-B"

    def test_backstop_reads_market_b_and_confirms(self, post):
        post.side_effect = [v2_resp(5), v2_resp(5)]
        readings = iter([positions_resp(), positions_resp(), positions_resp("TICK-B", -5)])
        tickers: list[str] = []

        def record(**kwargs):
            tickers.append(kwargs["ticker"])
            return next(readings)

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=record)
        assert _execute_one(client, self._ts_spec()).status == "executed"
        # before_no (TICK-B), before_yes (TICK-A), then the backstop on the
        # NO leg's market — TICK-B, never TICK-A
        assert tickers == ["TICK-B", "TICK-A", "TICK-B"]
        assert trader._V2_NO_MAPPING_CONFIRMED is True
        assert post.call_count == 2

    def test_backstop_disproof_on_market_b_stops_after_one_post(self, post, caplog):
        post.side_effect = [v2_resp(5), v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-B", 5),   # flat -> +5 on TICK-B: the ask opened YES
        ))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, self._ts_spec())
        assert result.status == "manual_review"
        assert "mapping disproven" in result.error and "TICK-B" in result.error
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
        assert_disproof_names_the_remedy(caplog)
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        # Only the NO leg went out — no YES leg, no unwind
        assert post.call_count == 1
        only = post.call_args_list[0].kwargs["body"]
        assert (only["ticker"], only["side"]) == ("TICK-B", "ask")

    def test_dry_run_lists_the_no_leg_first_with_leg_prices(self, caplog):
        spec = self._ts_spec()
        spec.total_cost = 3.50
        # Deliberately DIFFERENT from total_cost. make_spec returns a MagicMock,
        # so leaving this unset auto-vivifies a child mock and the %.2f format
        # would still render something — the assertion below would pass
        # vacuously against a regression back to total_cost (TS-12).
        spec.total_cost_with_fees = 3.78
        spec.min_payoff = 1.50
        with caplog.at_level(logging.INFO, logger="root"):
            results = execute_trades(MagicMock(), [spec], dry_run=True)
        assert [r.status for r in results] == ["simulated"]
        line = next(r.getMessage() for r in caplog.records if "[DRY RUN]" in r.getMessage())
        # Counts AND prices come from the legs (nB / pA), NO leg first —
        # not from the pair's nA / pB, which are not the traded prices here
        assert "Buy 5x NO on Market B @ 40.00%" in line
        assert "Buy 5x YES on Market A @ 30.00%" in line
        assert line.index("NO on Market B") < line.index("YES on Market A")
        # TS-12: the fee-INCLUSIVE figure, not the 3.50 contract-only total
        assert "Total cost: $3.78 incl. fees" in line
        assert "$3.50" not in line
        assert "Profit if won: $1.50" in line
        assert "Min profit" not in line
        # TS-23: the line names a PAIR order in submission order. "Batch order"
        # implied an atomic two-leg submission; there is no batch endpoint, and
        # the whole rollback machinery exists because the legs go one at a time.
        assert "Pair order (NO leg first" in line
        assert "Batch order" not in line


class TestUnparseableTransferResponse:
    """TS-17: a transfer the exchange ACCEPTED but answered unparseably.

    _check_and_parse validates the status BEFORE parsing, so a parse error out
    of signed_request_json proves a 2xx came back — the transfer was accepted
    and the money has moved. Before the fix that exception propagated into
    ensure_shard_collateral's generic handler, which logged a FAILED POST, left
    the destination out of accepted_cents, SKIPPED the settlement poll entirely
    and therefore never fired the MONEY IS IN FLIGHT critical. Money gone,
    balance never re-read, nothing alerted.
    """

    @staticmethod
    def _parse_error() -> JSONDecodeError:
        return JSONDecodeError("Expecting value", "", 0)

    def test_unparseable_2xx_returns_none_not_raise(self, monkeypatch):
        post = MagicMock(side_effect=self._parse_error())
        monkeypatch.setattr(trader, "signed_request_json", post)
        # None is the value that already means "accepted, in flight" — the
        # contract ensure_shard_collateral implements by NOT branching on it.
        assert _execute_transfer(MagicMock(), 1, 0, 1400) is None
        # Still single-shot: a retried transfer moves the money twice.
        assert post.call_count == 1

    def test_unparseable_2xx_logs_money_in_flight_critical(self, monkeypatch, caplog):
        monkeypatch.setattr(
            trader, "signed_request_json", MagicMock(side_effect=self._parse_error())
        )
        with caplog.at_level(logging.CRITICAL):
            _execute_transfer(MagicMock(), 1, 0, 1400)
        criticals = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL
        )
        assert "MONEY IS IN FLIGHT" in criticals
        assert "NOT re-sent" in criticals

    def test_unparseable_2xx_is_awaited_and_specs_survive(self, monkeypatch):
        # End to end: the destination must reach accepted_cents so the settle
        # poll actually runs. Before the fix the poll was skipped and every
        # spec needing that shard was dropped while the funds were in flight.
        post = MagicMock(side_effect=self._parse_error())
        va = MagicMock(return_value={0: 100_000, 1: 100_000})
        monkeypatch.setattr(trader, "signed_request_json", post)
        monkeypatch.setattr(trader, "read_shard_balances", va)
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0.05)

        spec = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)
        result = ensure_shard_collateral(
            MagicMock(), [spec], {0: 100, 1: 100_000}, None
        )
        assert result == [spec]          # not dropped
        va.assert_called()               # the settle poll DID run
        assert post.call_count == 1      # and was never re-sent

    def test_non_2xx_still_takes_the_failed_path(self, monkeypatch, caplog):
        # ApiException is not a ValueError, so the new handler must not catch
        # it: a genuine non-2xx is still a FAILED POST and still drops specs.
        post = MagicMock(side_effect=ApiException(status=500, reason="boom"))
        va = MagicMock(return_value={0: 100, 1: 100_000})
        monkeypatch.setattr(trader, "signed_request_json", post)
        monkeypatch.setattr(trader, "read_shard_balances", va)
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0.05)

        spec = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), [spec], {0: 100, 1: 100_000}, None
            )
        assert result == []
        errors = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
        )
        assert "FAILED" in errors
        assert post.call_count == 1


class TestNonObject2xxTransferResponse:
    """DR-05: a transfer the exchange ACCEPTED but answered with a non-object body.

    _check_and_parse validates the status BEFORE parsing, so anything that comes
    back from signed_request_json — parsed or not — proves a 2xx and therefore
    that the money has already moved. TS-17 covered the bodies that FAIL to
    parse; these are the ones that parse fine but to something other than a JSON
    object: b'"accepted"' -> str, b'[]' -> list, b'null' -> None, b'123' -> int,
    b'true' -> bool (every JSON type a body can carry that is NOT an object).
    Before the isinstance guard, `data.get("transfer_id")` raised AttributeError
    on every one of them, which landed in ensure_shard_collateral's generic
    `except Exception` handler: logged "FAILED (not retried…)", left the
    destination out of accepted_cents, SKIPPED the settlement poll and never
    fired the MONEY IS IN FLIGHT critical — the same misreport TS-17 fixed, one
    exception type over.

    Every case below drives the REAL _http._check_and_parse over a raw-response
    stand-in, so the value reaching _execute_transfer is produced by production
    parsing code rather than hand-written into the stub.
    """

    # (body bytes, the type name _execute_transfer must name in its critical)
    NON_OBJECT_BODIES = [
        (b'"accepted"', "str"),
        (b"[]", "list"),
        (b"null", "NoneType"),
        (b"123", "int"),
        (b"true", "bool"),
    ]

    class _RawResponse:
        """Minimal RESTResponse stand-in: .status and .data are all
        _check_and_parse reads (it falls back to .read() only when .data is
        None, which never happens here)."""

        def __init__(self, status: int, data: bytes):
            self.status = status
            self.data = data
            self.reason = "OK"

        def getheaders(self) -> dict:
            return {}

    @classmethod
    def _posting(cls, body: bytes) -> MagicMock:
        """A signed_request_json stand-in that runs the REAL 2xx parse over
        `body` — returning the parsed value, or raising exactly as production
        would on an unparseable one — and records each call so the single-shot
        contract can be asserted."""
        return MagicMock(
            side_effect=lambda *a, **kw: _http._check_and_parse(cls._RawResponse(200, body))
        )

    @pytest.mark.parametrize("body,type_name", NON_OBJECT_BODIES)
    def test_non_object_2xx_returns_none_not_raise(self, monkeypatch, body, type_name):
        post = self._posting(body)
        monkeypatch.setattr(trader, "signed_request_json", post)
        # None is the value that already means "accepted, id unknown" — the same
        # contract the id-less-dict and unparseable-body cases return.
        assert _execute_transfer(MagicMock(), 0, 1, 9662) is None
        # Still single-shot: a retried transfer moves the money twice.
        assert post.call_count == 1

    @pytest.mark.parametrize("body,type_name", NON_OBJECT_BODIES)
    def test_non_object_2xx_logs_one_critical_naming_the_type(
        self, monkeypatch, caplog, body, type_name
    ):
        monkeypatch.setattr(trader, "signed_request_json", self._posting(body))
        with caplog.at_level(logging.CRITICAL):
            _execute_transfer(MagicMock(), 0, 1, 9662)
        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        # Exactly one — the guard logs and returns, so one POST produces one
        # alert; a second record would mean it also fell through to the
        # parse-failure wording (or vice versa).
        assert len(criticals) == 1
        message = criticals[0].getMessage()
        assert "MONEY IS IN FLIGHT" in message
        assert "NOT re-sent" in message
        assert "was not a JSON object" in message
        # Naming the type is what tells the operator this is a payload-shape
        # drift to chase, not a parse failure or an outage.
        assert f"({type_name})" in message
        # $96.62 and the direction, so the account can be reconciled by hand.
        assert "96.62" in message
        assert "shard 0→1" in message

    @pytest.mark.parametrize("body,type_name", NON_OBJECT_BODIES)
    def test_non_object_2xx_is_awaited_and_specs_survive(
        self, monkeypatch, body, type_name
    ):
        # End to end: the destination must reach accepted_cents so the settle
        # poll actually runs. Before the fix the AttributeError skipped the poll
        # and dropped every spec needing that shard while the funds were moving.
        post = self._posting(body)
        balances = MagicMock(return_value={0: 100_000, 1: 100_000})
        monkeypatch.setattr(trader, "signed_request_json", post)
        monkeypatch.setattr(trader, "read_shard_balances", balances)
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", 0.05)

        spec = make_spec(shard_a=0, shard_b=0, cost_a=10.00, cost_b=5.00)
        result = ensure_shard_collateral(
            MagicMock(), [spec], {0: 100, 1: 100_000}, None
        )
        assert result == [spec]          # not dropped
        balances.assert_called()         # the settle poll DID run
        assert post.call_count == 1      # and was never re-sent

    def test_empty_2xx_body_still_takes_the_parse_failure_branch(
        self, monkeypatch, caplog
    ):
        # The isinstance guard sits AFTER the except clause, so an empty body —
        # which never produces a value at all — must keep its TS-17 wording.
        monkeypatch.setattr(trader, "signed_request_json", self._posting(b""))
        with caplog.at_level(logging.CRITICAL):
            assert _execute_transfer(MagicMock(), 0, 1, 9662) is None
        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(criticals) == 1
        message = criticals[0].getMessage()
        assert "could not be parsed" in message
        assert "was not a JSON object" not in message

    def test_object_2xx_body_still_yields_the_transfer_id(self, monkeypatch, caplog):
        # The guard must not fire on the normal shape: driven through the same
        # real parser, a JSON object still returns its id and logs no critical.
        monkeypatch.setattr(
            trader, "signed_request_json", self._posting(b'{"transfer_id": "tr_1"}')
        )
        with caplog.at_level(logging.CRITICAL):
            assert _execute_transfer(MagicMock(), 0, 1, 9662) == "tr_1"
        assert [r for r in caplog.records if r.levelno == logging.CRITICAL] == []

    def test_object_2xx_body_without_an_id_is_unchanged(self, monkeypatch, caplog):
        # An id-less OBJECT is the pre-existing "accepted, id unknown" case and
        # is NOT the new branch: same None, but no critical.
        monkeypatch.setattr(trader, "signed_request_json", self._posting(b"{}"))
        with caplog.at_level(logging.CRITICAL):
            assert _execute_transfer(MagicMock(), 0, 1, 9662) is None
        assert [r for r in caplog.records if r.levelno == logging.CRITICAL] == []


class _FakeClock:
    """A clock and a wait for driving a _WritePacer from one thread.

    Each wait moves the clock on by its timeout. A wait with no timeout would
    never end in one thread, so it fails the test instead of hanging.
    """

    def __init__(self, start: float = 0.0):
        self.now = start
        self.waited: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def wait(self, timeout) -> None:
        if timeout is None:
            raise AssertionError("the pacer would wait forever for a held write to end")
        self.waited.append(timeout)
        self.now += timeout


def _fake_pacer(rate, burst, clock: _FakeClock) -> _WritePacer:
    """A _WritePacer on a single-thread fake clock."""
    return _WritePacer(rate, burst, clock=clock.monotonic, wait=clock.wait)


class _SimStuck(BaseException):
    """Raised in every simulated thread once the simulation cannot move on
    (a BaseException, so no `except Exception` in trader turns it into a
    trade result)."""


class _SimCondition(threading.Condition):
    """The simulation's condition: notify_all() counts every blocked thread as
    running until it blocks again."""

    def __init__(self, sim: "_SimTime"):
        super().__init__()
        self._sim = sim

    def notify_all(self) -> None:
        self._sim._blocked.clear()
        super().notify_all()


class _SimTime:
    """Simulated time for threads that block only through wait() (the
    pacer's) or sleep() (a round trip, a pause).

    Time stands still while any thread runs and jumps to the earliest
    deadline once all are blocked, so times are exact. Threads woken together
    run in OS order, so tests check times, not which thread got which.
    `tasks` pieces of work run on `workers` threads, and each task calls
    done() when it ends. The pacer must share `cond` (see _sim_pacer).
    """

    REAL_LIMIT_SECONDS = 10.0

    def __init__(self, tasks: int, workers: int | None = None):
        self.now = 0.0
        self.cond = _SimCondition(self)
        self._tasks = tasks
        self._workers = tasks if workers is None else workers
        self._blocked: dict[int, float] = {}
        self.stuck = False

    def monotonic(self) -> float:
        return self.now

    def done(self) -> None:
        """One task has ended."""
        with self.cond:
            self._tasks -= 1
            self._advance_if_all_blocked()

    def wait(self, timeout) -> None:
        """The pacer's wait: called with self.cond held."""
        self._block(math.inf if timeout is None else self.now + timeout)

    def sleep(self, seconds: float) -> None:
        with self.cond:
            deadline = self.now + seconds
            while self.now < deadline:
                self._block(deadline)

    def _block(self, deadline: float) -> None:
        if self.stuck:
            raise _SimStuck
        me = threading.get_ident()
        self._blocked[me] = deadline
        if self._advance_if_all_blocked():
            return
        if not self.cond.wait(self.REAL_LIMIT_SECONDS):
            self.stuck = True
            self.cond.notify_all()
        self._blocked.pop(me, None)
        if self.stuck:
            raise _SimStuck

    def _advance_if_all_blocked(self) -> bool:
        running = min(self._workers, self._tasks)
        if not self._blocked or len(self._blocked) < running:
            return False
        nxt = min(self._blocked.values())
        if nxt == math.inf:
            self.stuck = True          # everyone waits on someone: deadlock
        else:
            self.now = max(self.now, nxt)
        self.cond.notify_all()
        return True


def _sim_pacer(rate, burst, sim: _SimTime) -> _WritePacer:
    """A _WritePacer running in the simulation's time."""
    pacer = _WritePacer(rate, burst, clock=sim.monotonic, wait=sim.wait)
    pacer._cond = sim.cond
    return pacer


def _run_sim_threads(sim: _SimTime, bodies) -> None:
    """Run each body on its own thread in the simulation and wait for all."""
    errors: list[BaseException] = []

    def run(body):
        try:
            body()
        except BaseException as exc:   # noqa: BLE001 — reported below
            errors.append(exc)
        finally:
            sim.done()

    threads = [threading.Thread(target=run, args=(b,)) for b in bodies]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not sim.stuck, "the simulation stopped: every thread was waiting for another"
    assert errors == []


def _assert_within_bucket(send_times, rate, burst) -> None:
    """No stretch of time holds more than burst + rate * (its length) sends."""
    times = sorted(send_times)
    for i in range(len(times)):
        for j in range(i, len(times)):
            assert j - i + 1 <= burst + rate * (times[j] - times[i]) + 1e-6, (
                f"{j - i + 1} sends in {times[j] - times[i]:.4f}s: {times}"
            )


class TestWritePacer:
    """The bucket itself: `burst` writes back to back, then one per 1/rate s,
    first come, first served within a lane."""

    def test_the_first_burst_does_not_wait(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        assert [pacer.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
        assert clock.waited == []

    def test_the_next_caller_waits_one_over_the_rate(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        for _ in range(3):
            pacer.acquire()
        assert pacer.acquire() == pytest.approx(0.25)
        assert clock.waited == [pytest.approx(0.25)]

    def test_a_caller_after_the_line_drains_waits_only_its_own_turn(self):
        # The 4th arrives after the 3rd was served, so it waits one 1/rate, not two
        clock = _FakeClock()
        pacer = _fake_pacer(4, 2, clock)
        assert [pacer.acquire(), pacer.acquire()] == [0.0, 0.0]
        assert pacer.acquire() == pytest.approx(0.25)
        assert pacer.acquire() == pytest.approx(0.25)
        assert clock.now == pytest.approx(0.5)

    def test_tokens_refill_with_elapsed_time(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        for _ in range(3):
            pacer.acquire()
        clock.now = 0.5          # half a second at 4 a second refills 2 tokens
        assert pacer.acquire() == 0.0
        assert pacer.acquire() == 0.0
        assert pacer.acquire() == pytest.approx(0.25)

    def test_the_refill_stops_at_the_burst(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        pacer.acquire()
        clock.now = 1_000.0      # a long idle spell holds a full bucket, no more
        assert [pacer.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
        assert pacer.acquire() == pytest.approx(0.25)

    def test_the_first_acquire_starts_the_clock(self):
        # Refill starts from the first request's clock reading, whatever clock
        clock = _FakeClock(start=0.0)
        pacer = _fake_pacer(1, 2, clock)
        assert [pacer.acquire(), pacer.acquire()] == [0.0, 0.0]
        assert pacer.acquire() == pytest.approx(1.0)
        clock.now = 10.0
        assert pacer.acquire() == 0.0

    @pytest.mark.parametrize(
        "rate", [0, -1, 0.0, float("nan"), float("inf"), True, "8", None],
    )
    def test_an_invalid_rate_raises(self, rate):
        with pytest.raises(ValueError, match="rate"):
            _WritePacer(rate, 8)

    @pytest.mark.parametrize("burst", [0, -1, 1, 1.5, 8.0, True, "8", None])
    def test_an_invalid_burst_raises(self, burst):
        # 1 too: a pair's NO leg takes two places at once
        with pytest.raises(ValueError, match="burst"):
            _WritePacer(8, burst)

    def test_the_shipped_pacer_uses_the_config_constants(self):
        # conftest replaces the module's pacer, so build the same call afresh
        pacer = _WritePacer(config.ORDER_WRITES_PER_SECOND, config.ORDER_WRITE_BURST)
        assert pacer._rate == float(config.ORDER_WRITES_PER_SECOND)
        assert pacer._burst == float(config.ORDER_WRITE_BURST)
        assert trader.ORDER_WRITES_PER_SECOND == config.ORDER_WRITES_PER_SECOND
        assert trader.ORDER_WRITE_BURST == config.ORDER_WRITE_BURST

    def test_the_module_pacer_is_built_from_the_config_names(self):
        # By syntax tree: bound once, to _WritePacer(ORDER_WRITES_PER_SECOND,
        # ORDER_WRITE_BURST), never literal numbers
        tree = ast.parse(inspect.getsource(trader))
        bindings = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if any(isinstance(t, ast.Name) and t.id == "_ORDER_WRITE_PACER"
                   for t in targets):
                bindings.append(node)
        assert len(bindings) == 1
        (node,) = bindings
        assert isinstance(node, ast.Assign) and len(node.targets) == 1
        call = node.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name) and call.func.id == "_WritePacer"
        assert [type(a) for a in call.args] == [ast.Name, ast.Name]
        assert [a.id for a in call.args] == ["ORDER_WRITES_PER_SECOND", "ORDER_WRITE_BURST"]
        assert call.keywords == []

    def test_a_long_wait_is_logged_once(self, caplog):
        clock = _FakeClock()
        pacer = _fake_pacer(2, 2, clock)
        pacer.acquire()
        pacer.acquire()
        with caplog.at_level(logging.INFO):
            assert pacer.acquire() == pytest.approx(0.5)
        lines = [r for r in caplog.records
                 if "Paced an order or transfer write" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert "0.50s" in lines[0].getMessage()

    def test_a_short_wait_is_not_logged(self, caplog):
        clock = _FakeClock()
        pacer = _fake_pacer(8, 2, clock)
        pacer.acquire()
        pacer.acquire()
        with caplog.at_level(logging.INFO):
            assert pacer.acquire() == pytest.approx(0.125)
        assert "Paced an order or transfer write" not in caplog.text

    def test_the_balance_is_read_under_the_lock_and_the_wait_is_handed_the_lock(self):
        # The clock is read inside the locked block, and the wait hook gets
        # the lock so Condition.wait can release it while waiting
        clock = _FakeClock()
        owned_at_clock: list[bool] = []
        owned_at_wait: list[bool] = []

        def locked_clock():
            owned_at_clock.append(pacer._cond._is_owned())
            return clock.monotonic()

        def wait(timeout):
            owned_at_wait.append(pacer._cond._is_owned())
            clock.wait(timeout)

        pacer = _WritePacer(4, 2, clock=locked_clock, wait=wait)
        pacer.acquire()
        pacer.acquire()
        pacer.acquire()
        assert owned_at_clock and all(owned_at_clock)
        assert owned_at_wait == [True]

    def test_a_caller_that_stops_waiting_leaves_the_line(self):
        # An interrupted caller (Ctrl-C) leaves the line, or those behind wait forever
        clock = _FakeClock()
        interrupted = [False]

        def wait(timeout):
            if not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt
            clock.wait(timeout)

        pacer = _WritePacer(4, 2, clock=clock.monotonic, wait=wait)
        pacer.acquire()
        pacer.acquire()
        with pytest.raises(KeyboardInterrupt):
            pacer.acquire()
        assert not pacer._in_turn and not pacer._hedges
        assert pacer.acquire() == pytest.approx(0.25)

    def test_a_caller_that_stops_waiting_removes_its_own_place_not_another(self):
        # Two identical waiting requests; the second stops. Only its own place
        # may go, or the first waits forever (why _PaceRequest is eq=False).
        sim = _SimTime(tasks=2)
        pacer = _sim_pacer(4, 2, sim)
        pacer.acquire()
        pacer.acquire()                       # bucket empty at t=0
        served: dict[str, float] = {}
        second_arrived = threading.Event()

        def first():
            pacer.acquire()
            served["first"] = sim.now

        def second():
            real_wait = pacer._wait

            def wait_once(timeout):
                pacer._wait = real_wait
                second_arrived.set()
                raise KeyboardInterrupt

            with sim.cond:
                while not pacer._in_turn:     # let the first caller join first
                    sim.cond.wait(0.01)
                pacer._wait = wait_once
            try:
                pacer.acquire()
            except KeyboardInterrupt:
                served["second"] = -1.0

        _run_sim_threads(sim, [first, second])
        assert second_arrived.is_set()
        assert served == {"first": pytest.approx(0.25), "second": -1.0}

    def test_a_held_write_served_just_before_its_caller_stops_goes_back(self):
        # Served by another waiter, then stopped before getting its held write:
        # the pacer takes the held token back
        clock = _FakeClock()

        def wait(timeout):
            clock.wait(timeout)
            pacer._serve(clock.now)           # served, as another waiter would
            raise KeyboardInterrupt

        pacer = _WritePacer(4, 2, clock=clock.monotonic, wait=wait)
        pacer.acquire()
        pacer.acquire()
        with pytest.raises(KeyboardInterrupt):
            pacer.acquire_with_hold()
        assert pacer._held == 0
        assert not pacer._in_turn
        # The held token is back; the one for the unsent POST stays spent
        assert pacer._tokens == pytest.approx(1.0)


class TestHeldWrites:
    """acquire_with_hold: a place now plus one held for later, counted
    against the bucket until sent or released, so a late send stays in bounds."""

    def test_it_takes_two_places(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 4, clock)
        wait, held = pacer.acquire_with_hold()
        assert wait == 0.0 and held is not None
        assert [pacer.acquire(), pacer.acquire()] == [0.0, 0.0]
        assert pacer.acquire() == pytest.approx(0.25)

    def test_it_waits_until_two_places_are_free_together(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        pacer.acquire()
        pacer.acquire()
        wait, _ = pacer.acquire_with_hold()   # one token left: needs one more
        assert wait == pytest.approx(0.25)

    def test_a_held_write_is_sent_with_no_wait_on_an_empty_bucket(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 2, clock)
        _, held = pacer.acquire_with_hold()   # both tokens gone
        assert held.send() is True
        assert clock.waited == []

    def test_a_held_write_counts_against_the_bucket_until_it_ends(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        _, held = pacer.acquire_with_hold()
        clock.now = 1_000.0      # refills to 3 - 1 held, not 3
        assert [pacer.acquire(), pacer.acquire()] == [0.0, 0.0]
        assert pacer.acquire() == pytest.approx(0.25)
        assert held.send() is True
        # Once sent, the refill may reach 3 again
        clock.now += 1_000.0
        assert [pacer.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]

    def test_a_held_write_sent_late_frees_its_room_only_after_the_refill(self):
        # Hold, idle, send late, then two singles at that instant: the held
        # send and the first single fill the burst of 2, so the second waits.
        # Freeing the hold's room before the refill would let all three go.
        clock = _FakeClock()
        pacer = _fake_pacer(8, 2, clock)
        _, held = pacer.acquire_with_hold()
        clock.now = 10.0
        assert held.send() is True
        assert pacer.acquire() == 0.0
        assert pacer.acquire() == pytest.approx(0.125)

    def test_a_released_write_gives_its_token_back(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 2, clock)
        _, held = pacer.acquire_with_hold()
        held.release()
        assert pacer.acquire() == 0.0
        assert pacer.acquire() == pytest.approx(0.25)

    def test_a_hold_ends_once(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 4, clock)
        _, released = pacer.acquire_with_hold()
        released.release()
        assert released.send() is False       # the caller must take its own place
        released.release()                    # and a second release adds nothing
        _, sent = pacer.acquire_with_hold()
        assert sent.send() is True
        assert sent.send() is False
        sent.release()                        # a sent write is never given back
        assert pacer._held == 0
        assert pacer._tokens == pytest.approx(4 - 1 - 2)

    def test_at_most_burst_minus_one_writes_are_ever_held(self):
        # A hold needs two free tokens, so room never drops below one: a hedge still goes
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        pacer.acquire_with_hold()
        clock.now += 100.0
        pacer.acquire_with_hold()
        assert pacer._held == 2
        clock.now += 100.0
        assert pacer._tokens + pacer._held <= pacer._burst
        with pytest.raises(AssertionError, match="forever"):
            pacer.acquire_with_hold()         # room for 1, never for 2 until a hold ends
        assert pacer.acquire_hedge() == 0.0   # a single write still goes

    def test_a_hold_waiting_for_room_is_woken_when_another_hold_ends(self):
        sim = _SimTime(tasks=2)
        pacer = _sim_pacer(4, 2, sim)
        served: dict[str, float] = {}

        def first():
            _, held = pacer.acquire_with_hold()   # bucket empty, one held
            sim.sleep(0.5)
            held.release()

        def second():
            sim.sleep(0.1)
            pacer.acquire_with_hold()[1].release()
            served["second"] = sim.now

        _run_sim_threads(sim, [first, second])
        # Released at 0.5: its token plus the refill covers two at once
        assert served["second"] == pytest.approx(0.5)

    def test_a_pair_waiting_for_room_is_woken_when_a_held_write_is_sent(self):
        # Real threads. The second pair can only wait to be woken (room for
        # one, needs two); sending the held write (which adds no token) must
        # still wake it.
        pacer = _WritePacer(20, 2)
        _, held = pacer.acquire_with_hold()
        served = threading.Event()

        def second():
            pacer.acquire_with_hold()[1].release()
            served.set()

        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:     # until the second pair is in line
            with pacer._cond:
                if pacer._in_turn:
                    break
            time.sleep(0.001)
        with pacer._cond:
            assert list(pacer._in_turn), "the second pair never joined the line"
        assert not served.is_set()
        assert held.send() is True
        assert served.wait(timeout=5.0)
        thread.join(timeout=5.0)

    @pytest.mark.parametrize("seed", range(12))
    def test_sends_never_exceed_the_bucket_however_late_held_writes_go(self, seed):
        # Random mixes of holds sent late or released, singles and hedges.
        # Plans are drawn up front so a seed always replays the same plans.
        rng = random.Random(seed)
        rate, burst, callers = 8.0, 4, 8
        plans = [
            [(rng.random() * 0.3, rng.choice(["pair", "pair", "single", "hedge"]),
              rng.random() * 2.0, rng.random() < 0.8) for _ in range(4)]
            for _ in range(callers)
        ]
        sim = _SimTime(tasks=callers)
        pacer = _sim_pacer(rate, burst, sim)
        sends: list[float] = []

        def caller(plan):
            for pause, kind, hold_for, send in plan:
                sim.sleep(pause)
                if kind == "pair":
                    _, held = pacer.acquire_with_hold()
                    sends.append(sim.now)
                    sim.sleep(hold_for)
                    if send:
                        held.send()
                        sends.append(sim.now)
                    else:
                        held.release()
                elif kind == "single":
                    pacer.acquire()
                    sends.append(sim.now)
                else:
                    pacer.acquire_hedge()
                    sends.append(sim.now)

        _run_sim_threads(sim, [lambda plan=plan: caller(plan) for plan in plans])
        assert len(sends) >= callers * 4
        _assert_within_bucket(sends, rate, burst)
        assert pacer._held == 0


class TestHedgeLane:
    """acquire_hedge is served before every waiting in-turn caller."""

    def test_a_hedge_goes_ahead_of_a_waiting_pair(self):
        sim = _SimTime(tasks=3)
        pacer = _sim_pacer(4, 2, sim)
        served: dict[str, float] = {}

        def holder():
            _, held = pacer.acquire_with_hold()   # t=0: bucket empty, one held
            sim.sleep(0.5)
            held.send()

        def pair():
            sim.sleep(0.01)
            pacer.acquire_with_hold()[1].send()
            served["pair"] = sim.now

        def hedge():
            sim.sleep(0.1)
            pacer.acquire_hedge()
            served["hedge"] = sim.now

        _run_sim_threads(sim, [holder, pair, hedge])
        # The hedge came later but gets the next token (0.25); the pair gets
        # room at 0.5 when the hold is sent, and two tokens at 0.75
        assert served["hedge"] == pytest.approx(0.25)
        assert served["pair"] == pytest.approx(0.75)

    def test_without_the_hedge_lane_the_same_write_would_wait_behind_the_pair(self):
        # Control for the test above: taken in turn, it goes after the pair
        sim = _SimTime(tasks=3)
        pacer = _sim_pacer(4, 2, sim)
        served: dict[str, float] = {}

        def holder():
            _, held = pacer.acquire_with_hold()
            sim.sleep(0.5)
            held.send()

        def pair():
            sim.sleep(0.01)
            pacer.acquire_with_hold()[1].send()
            served["pair"] = sim.now

        def in_turn():
            sim.sleep(0.1)
            pacer.acquire()
            served["in_turn"] = sim.now

        _run_sim_threads(sim, [holder, pair, in_turn])
        assert served["pair"] == pytest.approx(0.75)
        assert served["in_turn"] == pytest.approx(1.0)

    def test_hedges_are_served_in_the_order_they_arrive(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 2, clock)
        pacer.acquire()
        pacer.acquire()
        assert pacer.acquire_hedge() == pytest.approx(0.25)
        assert pacer.acquire_hedge() == pytest.approx(0.25)

    def test_concurrent_callers_each_get_their_own_place(self):
        # 20 callers at once get 20 distinct places (a lost update repeats a wait)
        rate, burst, callers = 4, 3, 20
        sim = _SimTime(tasks=callers)
        pacer = _sim_pacer(rate, burst, sim)
        waits: list[float] = []

        def caller():
            waits.append(pacer.acquire())

        _run_sim_threads(sim, [caller for _ in range(callers)])
        expected = [0.0] * burst + [k / rate for k in range(1, callers - burst + 1)]
        assert sorted(waits) == [pytest.approx(w) for w in expected]

    def test_real_threads_are_never_served_faster_than_the_bucket(self):
        # Real threads at a high rate, some holding and sending late. Send
        # times come from the pacer's own clock readings, not OS wake-ups.
        rate, burst, callers = 400.0, 4, 16
        local = threading.local()

        def clock():
            now = time.monotonic()
            if getattr(local, "first", None) is None:
                local.first = now
            return now

        pacer = _WritePacer(rate, burst, clock=clock)
        start = threading.Barrier(callers)
        sends: list[float] = []
        lock = threading.Lock()

        def timed(call):
            local.first = None
            result = call()
            return local.first, result

        def run(i):
            start.wait()
            if i % 2:
                arrived, (wait, held) = timed(pacer.acquire_with_hold)
                time.sleep(0.001 * (i % 5))
                sent_at, ok = timed(held.send)
                assert ok
                times = [arrived + wait, sent_at]
            else:
                take = pacer.acquire_hedge if i % 4 == 0 else pacer.acquire
                arrived, wait = timed(take)
                times = [arrived + wait]
            with lock:
                sends.extend(times)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not any(t.is_alive() for t in threads)
        assert len(sends) == callers + callers // 2
        _assert_within_bucket(sends, rate, burst)
        assert pacer._held == 0


class TestWritesArePaced:
    """Each POST takes exactly one pacer place first and is sent once."""

    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def pacer(self, monkeypatch, events):
        """A stand-in pacer recording "acquire", "hedge", "opening" (a place
        plus a held one), "held" (a held place sent) and "released" in `events`."""
        mock = MagicMock()
        mock.acquire.side_effect = lambda: events.append("acquire") or 0.0
        mock.acquire_hedge.side_effect = lambda: events.append("hedge") or 0.0

        def with_hold():
            events.append("opening")
            held = MagicMock()
            live = [True]

            def send():
                if not live[0]:
                    return False
                live[0] = False
                events.append("held")
                return True

            def release():
                if live[0]:
                    live[0] = False
                    events.append("released")

            held.send.side_effect = send
            held.release.side_effect = release
            return 0.0, held

        mock.acquire_with_hold.side_effect = with_hold
        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", mock)
        return mock

    @pytest.fixture
    def post(self, monkeypatch, events):
        """signed_request_json, recording each POST in `events`."""
        mock = MagicMock()
        replies: list = []

        def answer(*args, **kwargs):
            events.append("post")
            reply = replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        mock.side_effect = answer
        mock.replies = replies
        monkeypatch.setattr(trader, "signed_request_json", mock)
        return mock

    @staticmethod
    def _too_many_requests() -> ApiException:
        # The body the exchange sends with a 429
        return ApiException(
            status=429, reason="Too Many Requests",
            body='{"error":{"code":"too_many_requests","message":"too many requests"}}',
        )

    def test_a_v2_order_waits_before_its_post(self, pacer, post, events):
        post.replies.append(v2_resp(5))
        assert _submit_order_v2(MagicMock(), _build_no_order_v2(_no_leg(make_spec()))) == "executed"
        assert events == ["acquire", "post"]

    def test_a_v2_order_takes_the_place_it_is_given(self, pacer, post, events):
        # The given place is used instead of one in turn
        post.replies.append(v2_resp(5))
        pace = MagicMock(side_effect=lambda: events.append("given") or 0.0)
        body = _build_no_order_v2(_no_leg(make_spec()))
        assert _submit_order_v2(MagicMock(), body, pace=pace) == "executed"
        assert events == ["given", "post"]
        pacer.acquire.assert_not_called()

    def test_the_v2_log_line_is_written_after_the_wait(self, pacer, post, caplog):
        # So the "Submitting V2 order" line's time is the send time
        logged_before_wait = []
        pacer.acquire.side_effect = lambda: logged_before_wait.append(
            "Submitting V2 order" in caplog.text
        ) or 0.0
        post.replies.append(v2_resp(5))
        with caplog.at_level(logging.INFO):
            _submit_order_v2(MagicMock(), _build_no_order_v2(_no_leg(make_spec())))
        assert logged_before_wait == [False]
        assert "Submitting V2 order" in caplog.text

    def test_a_transfer_waits_before_its_post(self, pacer, post, events):
        post.replies.append(transfer_resp("tr_9"))
        assert _execute_transfer(MagicMock(), 1, 0, 1400) == "tr_9"
        assert events == ["acquire", "post"]

    def test_a_v2_kill_takes_one_place_and_one_post(self, pacer, post, events):
        post.replies.append(fok_kill_error())
        assert _submit_order_v2(MagicMock(), _build_no_order_v2(_no_leg(make_spec()))) == "canceled"
        assert events == ["acquire", "post"]

    def test_a_v2_429_takes_one_place_and_one_post_and_still_raises(
        self, pacer, post, events,
    ):
        # Pacing is not a retry: a 429 still raises
        err = self._too_many_requests()
        post.replies.append(err)
        with pytest.raises(ApiException) as exc_info:
            _submit_order_v2(MagicMock(), _build_yes_order_v2(_yes_leg(make_spec())))
        assert exc_info.value is err
        assert events == ["acquire", "post"]

    def test_a_transfer_429_takes_one_place_and_one_post(self, pacer, post, events):
        post.replies.append(self._too_many_requests())
        with pytest.raises(ApiException):
            _execute_transfer(MagicMock(), 0, 1, 100)
        assert events == ["acquire", "post"]

    def test_an_unwind_takes_the_hedge_lane_by_default(self, pacer, post, events):
        post.replies.append(v2_resp(5))
        spec = make_spec()
        result = trader._rollback_no_leg(MagicMock(), spec, _no_leg(spec), "why")
        assert result.status == "rolled_back"
        assert events == ["hedge", "post"]
        pacer.acquire.assert_not_called()

    def test_a_429_on_the_yes_leg_paces_the_rollback_too(
        self, pacer, post, events, v2_mapping_confirmed, monkeypatch,
    ):
        # YES leg 429'd and unfilled, so the NO leg is unwound: NO takes two
        # places, YES sends the held one, the unwind takes the hedge lane
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.replies.extend([v2_resp(5), self._too_many_requests(), v2_resp(5)])
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None,   # before_no
            None,   # before_yes
            None,   # after_yes — unmoved
            None,   # lag re-read — still unmoved
        )
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"
        assert events == ["opening", "post", "held", "post", "hedge", "post"]
        assert post.call_args_list[2].kwargs["body"]["reduce_only"] is True
        pacer.acquire.assert_not_called()

    def test_a_dry_run_takes_no_place(self, pacer):
        results = execute_trades(MagicMock(), [make_spec()], dry_run=True)
        assert [r.status for r in results] == ["simulated"]
        pacer.acquire.assert_not_called()
        pacer.acquire_with_hold.assert_not_called()
        pacer.acquire_hedge.assert_not_called()


class TestPairWrites:
    """Which pacer places each _execute_one path takes, read from the places
    held at each POST, the hedge-lane places and the balance afterwards (a
    real pacer that never waits: every case fits in one burst)."""

    BURST = 8

    @pytest.fixture
    def pacer(self, monkeypatch):
        clock = _FakeClock()
        pacer = _fake_pacer(8, self.BURST, clock)
        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", pacer)
        return pacer

    @pytest.fixture
    def posts(self, monkeypatch):
        """signed_request_json answering from a script, recording each body."""
        bodies: list[dict] = []
        replies: list = []

        held_at_post: list[int] = []

        def answer(client, method, path, body):
            bodies.append(body)
            held_at_post.append(trader._ORDER_WRITE_PACER._held)
            reply = replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        monkeypatch.setattr(trader, "signed_request_json", answer)
        return SimpleNamespace(bodies=bodies, replies=replies, held_at_post=held_at_post)

    @staticmethod
    def _count_hedges(pacer, monkeypatch) -> list:
        """Record each hedge-lane place the pacer hands out."""
        hedges: list = []
        real = pacer.acquire_hedge
        monkeypatch.setattr(pacer, "acquire_hedge", lambda: hedges.append(1) or real())
        return hedges

    @staticmethod
    def _spent(pacer) -> float:
        """Tokens taken so far (the fake clock never moves, so no refill)."""
        return pacer._burst - pacer._tokens - pacer._held

    def test_a_filled_pair_sends_its_yes_leg_on_the_held_place(
        self, pacer, posts, v2_mapping_confirmed, monkeypatch,
    ):
        posts.replies.extend([v2_resp(5), v2_resp(5)])
        hedges = self._count_hedges(pacer, monkeypatch)
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert [b["side"] for b in posts.bodies] == ["ask", "bid"]
        # One place held while the NO leg posts, spent by the YES leg
        assert posts.held_at_post == [1, 0]
        assert hedges == []
        assert pacer._held == 0 and self._spent(pacer) == 2

    def test_a_killed_no_leg_gives_the_held_place_back(
        self, pacer, posts, v2_mapping_confirmed,
    ):
        posts.replies.append(fok_kill_error())
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert pacer._held == 0 and self._spent(pacer) == 1

    def test_a_killed_yes_leg_is_unwound_through_the_hedge_lane(
        self, pacer, posts, v2_mapping_confirmed, monkeypatch,
    ):
        posts.replies.extend([v2_resp(5), fok_kill_error(), v2_resp(5)])
        hedges = self._count_hedges(pacer, monkeypatch)
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"
        assert posts.held_at_post == [1, 0, 0]
        assert hedges == [1]
        assert pacer._held == 0 and self._spent(pacer) == 3

    def test_a_no_leg_whose_fill_was_unclear_gives_its_place_back_before_the_reads(
        self, pacer, posts, v2_mapping_confirmed, monkeypatch,
    ):
        # The NO leg raised: its held place goes back before the (possibly
        # slow) reads; they show it filled, so it is unwound through the hedge lane
        posts.replies.extend([ConnectionError("reset"), v2_resp(5)])
        hedges = self._count_hedges(pacer, monkeypatch)
        held_at_read: list[int] = []
        script = positions_seq(None, None, ("TICK-A", -5))

        def read(**kwargs):
            held_at_read.append(pacer._held)
            return script(**kwargs)

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=read)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"
        assert posts.bodies[-1]["reduce_only"] is True
        assert held_at_read == [0, 0, 0]   # the baselines, and the read after the error
        assert posts.held_at_post == [1, 0]
        assert hedges == [1]
        assert pacer._held == 0 and self._spent(pacer) == 2

    def test_a_no_leg_that_did_not_fill_gives_the_held_place_back(
        self, pacer, posts, v2_mapping_confirmed, monkeypatch,
    ):
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        posts.replies.append(ConnectionError("reset"))
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(None, None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert pacer._held == 0 and self._spent(pacer) == 1

    def test_a_disproven_mapping_gives_the_held_place_back(
        self, pacer, posts, monkeypatch,
    ):
        posts.replies.append(v2_resp(5))
        client = MagicMock()
        client.get_positions_without_preload_content = positions_seq(
            None, None, ("TICK-A", 5),     # the NO buy moved the position the wrong way
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert len(posts.bodies) == 1
        assert pacer._held == 0 and self._spent(pacer) == 1

    def test_an_exception_escaping_the_pair_gives_the_held_place_back(
        self, pacer, posts, monkeypatch,
    ):
        posts.replies.append(v2_resp(5))
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())

        def broken(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(trader, "_confirm_v2_no_mapping", broken)
        results = execute_trades(client, [make_spec()], dry_run=False)
        assert [r.status for r in results] == ["manual_review"]
        assert pacer._held == 0 and self._spent(pacer) == 1

    def test_a_pair_never_holds_two_places(self, pacer):
        writes = trader._PairWrites(pacer)
        writes.opening()
        writes.opening()
        assert pacer._held == 1
        writes.close()
        assert pacer._held == 0

    def test_the_second_hedge_write_takes_the_hedge_lane(self, pacer, monkeypatch):
        hedges = self._count_hedges(pacer, monkeypatch)
        writes = trader._PairWrites(pacer)
        writes.opening()
        assert writes.hedge() == 0.0 and hedges == []
        writes.hedge()
        assert hedges == [1]
        writes.close()
        assert pacer._held == 0 and self._spent(pacer) == 3


class TestExecuteTradesArePaced:
    """Whole portfolios through execute_trades in simulated time, at the
    shipped rate and burst: sends stay in bounds, YES legs never wait, and
    unwinds go ahead of waiting NO legs."""

    POST_SECONDS = 0.1
    READ_SECONDS = 0.05

    @staticmethod
    def _specs(n: int) -> list:
        """n specs on distinct tickers, like a real portfolio's."""
        specs = []
        for i in range(n):
            spec = make_spec(title=f"pair {i}")
            spec.pair.market_a.ticker = f"TICK-A{i}"
            spec.pair.market_b.ticker = f"TICK-B{i}"
            specs.append(spec)
        return specs

    def _run(self, monkeypatch, pairs: int, *, kill_yes: bool = False) -> list:
        """Run the pairs; return each POST as (sent, returned, ticker, kind),
        kind being "no", "yes" or "unwind"."""
        rate, burst = config.ORDER_WRITES_PER_SECOND, config.ORDER_WRITE_BURST
        workers = min(config.TRADER_MAX_WORKERS, pairs)
        sim = _SimTime(tasks=pairs, workers=workers)
        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", _sim_pacer(rate, burst, sim))
        monkeypatch.setattr(trader.time, "sleep", sim.sleep)
        posts: list[tuple[float, float, str, str]] = []
        lock = threading.Lock()

        def post(client, method, path, body):
            sent = sim.now
            kind = ("unwind" if body.get("reduce_only")
                    else "no" if body["side"] == "ask" else "yes")
            sim.sleep(self.POST_SECONDS)
            with lock:
                posts.append((sent, sim.now, body["ticker"], kind))
            if kind == "yes" and kill_yes:
                raise fok_kill_error()
            return v2_resp(5)

        def read(**kwargs):
            sim.sleep(self.READ_SECONDS)
            return positions_resp()

        monkeypatch.setattr(trader, "signed_request_json", post)
        real = trader._execute_one

        def run(client, spec):
            try:
                return real(client, spec)
            finally:
                sim.done()

        monkeypatch.setattr(trader, "_execute_one", run)
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=read)
        results = execute_trades(client, self._specs(pairs), dry_run=False)
        assert not sim.stuck
        expected = "rolled_back" if kill_yes else "executed"
        assert [r.status for r in results] == [expected] * pairs
        _assert_within_bucket([p[0] for p in posts], rate, burst)
        return posts

    @staticmethod
    def _by_pair(posts) -> dict:
        """Each pair's POSTs by kind, keyed by the pair's index."""
        pairs: dict = {}
        for sent, returned, ticker, kind in posts:
            index = ticker.split("TICK-")[1][1:]
            pairs.setdefault(index, {})[kind] = (sent, returned)
        return pairs

    @pytest.mark.parametrize("pairs", [7, 14])
    def test_every_yes_leg_goes_out_the_moment_its_no_leg_returns(
        self, monkeypatch, v2_mapping_confirmed, pairs,
    ):
        posts = self._run(monkeypatch, pairs)
        by_pair = self._by_pair(posts)
        assert len(by_pair) == pairs
        for legs in by_pair.values():
            no_returned = legs["no"][1]
            yes_sent = legs["yes"][0]
            assert yes_sent == pytest.approx(no_returned, abs=1e-9)

    def test_the_opening_no_legs_are_spread_to_the_write_limit(
        self, monkeypatch, v2_mapping_confirmed,
    ):
        # All pairs ask at 0.1 (two 0.05 s reads). Four fit in the burst of 8
        # (two places each); each later pair goes 2/8 s after the one before.
        posts = self._run(monkeypatch, 7)
        rate, burst = config.ORDER_WRITES_PER_SECOND, config.ORDER_WRITE_BURST
        start = 2 * self.READ_SECONDS
        first = burst // 2
        expected = [start] * first + [
            start + 2 * k / rate for k in range(1, 7 - first + 1)
        ]
        no_sent = sorted(p[0] for p in posts if p[3] == "no")
        assert no_sent == [pytest.approx(t) for t in expected]

    @pytest.mark.parametrize("pairs", [7, 14])
    def test_every_unwind_goes_ahead_of_the_waiting_no_legs(
        self, monkeypatch, v2_mapping_confirmed, pairs,
    ):
        # Every YES leg is killed. No NO leg may go out between an unwind's
        # request and its POST, and it waits at most 1/rate for itself plus
        # 1/rate per unwind ahead of it.
        posts = self._run(monkeypatch, pairs, kill_yes=True)
        rate = config.ORDER_WRITES_PER_SECOND
        by_pair = self._by_pair(posts)
        no_sent = [legs["no"][0] for legs in by_pair.values()]
        asked = sorted(legs["yes"][1] for legs in by_pair.values())
        for legs in by_pair.values():
            requested = legs["yes"][1]
            sent = legs["unwind"][0]
            assert not [t for t in no_sent if requested < t < sent - 1e-9]
            ahead = sum(1 for t in asked if t <= requested)
            assert sent - requested <= ahead / rate + 1e-9
