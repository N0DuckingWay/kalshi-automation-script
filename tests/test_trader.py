"""Tests for trader.py — order construction (both the V2 and the retained
legacy endpoint), V2 price/tick math, rollback verification and its loss
floor, exception disambiguation, and the cross-shard collateral transfer
machinery. All Kalshi API interaction is mocked per project policy (tests must
run offline).

Legs are named by SUBMISSION order, not by market: the NO leg is always
submitted first and is the leg the rollback unwinds, the YES leg second. Which
market carries which side is the pair type's business (trader._ordered_legs):
same_title puts the NO leg on market_a (TICK-A) — make_spec's default, so the
long-standing same-title cases below read "TICK-A" for the NO leg unchanged —
while time_series puts it on market_b (TICK-B); TestTimeSeriesLegOrder pins
that flip end to end and TestSameTitleWireIdentity pins the same-title bodies
byte for byte.

Ambiguity handling is DELTA-based: _execute_one reads BOTH legs' baseline
positions up front (NO leg's ticker first, then the YES leg's), before either
order is submitted, and compares each against a reading taken after an
exception, attributing the outcome to the change. Mocks therefore sequence
get_positions responses with side_effect (see positions_seq) rather than
returning one flat payload — a single return_value would make before and
after identical, i.e. delta 0. Every _execute_one call consumes TWO baseline
reads before anything else, so a mock sequence written for the old
read-on-demand protocol will fail with StopIteration or a wrong status; read
such failures through that lens first.

The merged default is ORDER_API_VERSION="v2", so any class that drives
_execute_one down the legacy path opts in via the `legacy_mode` fixture.
"""
import ast
import inspect
import json
import logging
import math
import textwrap
import threading
import time
import uuid
from decimal import Decimal
from json import JSONDecodeError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import _http, config, trader
from kalshi_betting.config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
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
    _build_no_order,
    _build_no_order_v2,
    _build_rollback_order_any,
    _build_rollback_order_v2,
    _build_yes_order,
    _build_yes_order_v2,
    _buy_max_cost_cents,
    _ceil_to_tick,
    _cents_to_centicents,
    _execute_one,
    _execute_transfer,
    _format_count,
    _format_price,
    _is_fok_kill,
    _legacy_routable,
    _ordered_legs,
    _partition_by_funding,
    _plan_transfers,
    _position_count,
    _required_cents_by_shard,
    _rollback_floor_cents,
    _submit_order,
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


def order_resp(status: str) -> SimpleNamespace:
    """Raw create_order response — trader parses the JSON body directly
    because the SDK's Order response model can't deserialize live payloads."""
    payload = {"order": {"status": status}}
    return SimpleNamespace(status=201, data=json.dumps(payload).encode("utf-8"))


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


class TestOrderPriceProtection:
    def test_no_leg_has_buy_max_cost(self):
        spec = make_spec(x=5, nA=0.40)
        order = _build_no_order(_no_leg(spec))
        expected = math.ceil(5 * 0.40 * 100) + 5 * BUY_MAX_COST_SLIPPAGE_CENTS
        assert order.buy_max_cost == expected
        assert order.side == "no"
        assert order.action == "buy"
        assert order.time_in_force == "fill_or_kill"

    def test_yes_leg_has_buy_max_cost(self):
        spec = make_spec(x=5, pB=0.35)
        order = _build_yes_order(_yes_leg(spec))
        expected = math.ceil(5 * 0.35 * 100) + 5 * BUY_MAX_COST_SLIPPAGE_CENTS
        assert order.buy_max_cost == expected
        assert order.side == "yes"

    def test_float_noise_does_not_loosen_the_cap(self):
        # 7 * 0.07 * 100 == 49.00000000000001 in binary float, so a bare
        # ceil() would hand the order a spurious extra cent of headroom.
        # Rounding to 6 decimals first keeps the cap at the true 49 cents —
        # strictly tighter price protection, never looser.
        assert _buy_max_cost_cents(7, 0.07) == 49 + 7 * BUY_MAX_COST_SLIPPAGE_CENTS
        assert math.ceil(7 * 0.07 * 100) == 50  # what the un-rounded form gave

    def test_genuine_fraction_still_rounds_up(self):
        # The guard must only remove noise: a real sub-cent remainder still
        # ceils, or the cap could reject a fill at the scanned price.
        assert _buy_max_cost_cents(3, 0.335) == 101 + 3 * BUY_MAX_COST_SLIPPAGE_CENTS


class TestRollbackPriceFloor:
    """The NO-leg unwind is floored, never an unbounded market order: on the
    legacy path the floor is the price of a fill-or-kill limit sell, and on V2
    the same floor caps the immediate-or-cancel bid. The floor is read from
    the NO leg's own scanned entry — `nA` for the same-title default used here
    (market_a is the NO leg)."""

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


@pytest.fixture
def legacy_mode(monkeypatch):
    """Pin trader to the retained legacy /portfolio/orders order path.

    ORDER_API_VERSION now defaults to "v2", so the legacy-endpoint cases below
    must select their path explicitly rather than relying on the default —
    otherwise they would silently stop covering the legacy code they exist for.
    """
    monkeypatch.setattr(trader, "ORDER_API_VERSION", "legacy")


@pytest.fixture
def v2_mode(monkeypatch):
    """Pin trader to the V2 order path (the config default, made explicit)."""
    monkeypatch.setattr(trader, "ORDER_API_VERSION", "v2")


@pytest.fixture(autouse=True)
def _reset_v2_mapping_latch(monkeypatch):
    """Start every test from a fresh process's unlatched NO-mapping state.

    trader._V2_NO_MAPPING_CONFIRMED is a PROCESS-lifetime latch that real
    execution flips, so without this a single test that confirms the mapping
    would silently disable the backstop for every test that runs after it.
    monkeypatch restores the pre-test value at teardown, so the latch can never
    leak across tests in either direction.
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


class TestRollbackVerification:
    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """These cases assert on the legacy CreateOrderRequest wire format."""

    def test_unfilled_rollback_reports_rollback_failed(self):
        # NO leg fills, YES leg FoK is rejected, and the rollback FoK is ALSO
        # rejected — the orphaned NO-leg position must surface as
        # "rollback_failed", never be logged away as a successful rollback.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            order_resp("canceled"),   # YES leg rejected (confirmed non-fill)
            order_resp("canceled"),   # rollback rejected → orphaned position
        ])
        # Only the two pre-submission baselines are read: neither leg raised,
        # so no ambiguity snapshot is taken.
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error

    def test_filled_rollback_reports_rolled_back(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            order_resp("canceled"),   # YES leg rejected
            order_resp("executed"),   # rollback filled
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_rollback_order_is_floored_reduce_only_limit_sell(self):
        # The unwind must be a LIMIT sell carrying a proceeds floor: a market
        # sell has no such knob, so a collapsed book would realize an unbounded
        # loss on a position we only hold because YES leg failed.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),
            order_resp("canceled"),
            order_resp("executed"),
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        spec = make_spec(nA=0.40)
        _execute_one(client, spec)
        rollback_call = client.create_order_without_preload_content.call_args_list[2]
        rollback_req = rollback_call.kwargs["create_order_request"]
        assert rollback_req.action == "sell"
        assert rollback_req.side == "no"
        assert rollback_req.type == "limit"
        assert rollback_req.no_price == 40 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
        assert rollback_req.count == spec.x
        assert rollback_req.time_in_force == "fill_or_kill"
        assert rollback_req.reduce_only is True

    def test_floored_rollback_killed_by_price_reports_rollback_failed(self):
        # A book below the floor kills the FoK limit sell. The position is
        # still open, so the outcome must stay "rollback_failed" for manual
        # review — the same contract the old market unwind had when unfilled.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            order_resp("canceled"),   # YES leg rejected
            order_resp("canceled"),   # floored unwind killed by the price floor
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec(nA=0.62))
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error
        rollback_req = client.create_order_without_preload_content.call_args_list[2].kwargs[
            "create_order_request"
        ]
        assert rollback_req.no_price == 62 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT

    def test_clean_double_fill_submits_exactly_two_orders(self):
        # The happy path must be untouched by the delta protocol: two orders,
        # no rollback, and no submit-retry.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            order_resp("executed"),   # YES leg
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 2

    def test_leg_a_fok_rejection_is_failed_without_position_check(self):
        # A clean FoK rejection is a confirmed non-fill — no ambiguity
        # snapshot, no rollback, and YES leg is never submitted.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(
            return_value=order_resp("canceled")
        )
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert "NO leg FoK not filled" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        # Only the two up-front baselines were read — no ambiguity snapshot
        assert client.get_positions_without_preload_content.call_count == 2

    def test_both_baselines_are_read_before_any_order_is_submitted(self):
        # The unhedged window is the gap between NO leg's fill and YES leg's
        # submission. A position read in there is a blocking network call that
        # can burn the full ~62s retry schedule while the account holds a naked
        # NO on market A, so BOTH baselines must be taken up front. YES leg's is
        # equally valid there: it reads a different ticker, and no fill on that
        # ticker can have happened yet.
        calls: list[str] = []

        def record_positions(*args, **kwargs):
            calls.append("positions")
            return positions_resp()

        def record_order(*args, **kwargs):
            calls.append("order")
            return order_resp("executed")

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=record_positions)
        client.create_order_without_preload_content = MagicMock(side_effect=record_order)

        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert calls == ["positions", "positions", "order", "order"]


class TestNoLegExceptionDisambiguation:
    """The NO leg raised: the outcome is attributed to the position DELTA.

    Same-title default, so the NO leg is TICK-A (market_a)."""

    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """Exercises the legacy submission path's exception handling."""

    def test_no_movement_is_failed(self, monkeypatch):
        # Exception + position unchanged → confirmed non-fill, no rollback sent.
        # RE-PINNED (DR-64) from a three-entry script: a zero delta is only a
        # confirmed non-fill once the LAG RE-READ has also come back zero, so
        # the genuine-non-fill case now scripts a fourth, still-unmoved reading.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        # before_no, before_yes (both up front), then after_no, then the re-read
        client.get_positions_without_preload_content = positions_seq(None, None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert client.create_order_without_preload_content.call_count == 1

    def test_external_no_position_unchanged_is_failed_not_unwound(self, monkeypatch):
        # REGRESSION (BS-01): the account already holds 10 NO contracts on
        # TICK-A from an earlier run, and our order genuinely did not fill.
        # The old absolute check (held_a != 0) unwound that unrelated holding;
        # the delta is 0, so this must be a clean "failed" with NO sell order.
        # RE-PINNED (DR-64) from a three-entry script — the fourth reading is
        # the lag re-read, still unmoved because this really is a non-fill.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -10),   # after_no — unmoved
            ("TICK-A", -10),   # lag re-read — still unmoved
        )
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert client.create_order_without_preload_content.call_count == 1

    def test_delta_of_our_no_buy_is_unwound(self):
        # Exception but the position moved by exactly -spec.x (timeout AFTER
        # the fill) — the half-filled pair must be unwound, not abandoned.
        # The account also held 10 unrelated NO contracts, which the delta
        # correctly ignores.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            TimeoutError("timeout"),  # NO leg raises after actually filling
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -15),   # after_no — moved by -5 == -spec.x
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        # Exactly one submission attempt per NO-leg order plus the rollback
        assert client.create_order_without_preload_content.call_count == 2

    def test_unattributable_delta_is_manual_review(self):
        # The position moved, but by an amount our order cannot explain (an
        # unrelated trade landed in the snapshot window). A reduce-only sell
        # would liquidate a position we may not own — surface it instead.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", 0),     # before_no
            None,              # before_yes (taken up front, unused here)
            ("TICK-A", -3),    # after_no — -3, but spec.x is 7
        )
        result = _execute_one(client, make_spec(x=7))
        assert result.status == "manual_review"
        assert "delta=-3" in result.error
        # No unwind order was submitted
        assert client.create_order_without_preload_content.call_count == 1

    def test_snapshot_failure_is_manual_review(self):
        # The lookup itself failed, so the state is unknown. This is the
        # behavior change: the old code unwound blindly here.
        #
        # The call_count assertion is load-bearing beyond "no retry loop": the
        # snapshot must run OUTSIDE NO leg's except block. Inside it, the
        # RuntimeError would inherit the submission's TimeoutError as
        # __context__, api_call_with_retry's cause-chain walk would classify it
        # as transient, and this decision would stall for the full ~62s backoff.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            None,                            # before_no
            None,                            # before_yes (taken up front)
            RuntimeError("lookup failed"),   # after_no — non-retryable → fail fast
        )
        with patch.object(_http.time, "sleep") as sleep:
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "delta=None" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        assert client.get_positions_without_preload_content.call_count == 3
        sleep.assert_not_called()


class TestYesLegExceptionDisambiguation:
    """The YES leg raised: same delta protocol, but never auto-rollback on
    unknown. Same-title default, so the YES leg is TICK-B (market_b)."""

    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """Exercises the legacy submission path's exception handling."""

    def test_delta_of_our_yes_buy_is_executed(self):
        # YES leg raises but the position moved by exactly +spec.y — the pair
        # actually completed; rolling back NO leg would REVERSE the hedge.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            TimeoutError("timeout"),  # YES leg raises after actually filling
        ])
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_no
            None,              # before_yes
            ("TICK-B", 5),     # after_yes — moved by +5 == spec.y
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "executed"
        # No rollback order was submitted
        assert client.create_order_without_preload_content.call_count == 2

    def test_external_yes_position_unchanged_rolls_back(self, monkeypatch):
        # HEADLINE REGRESSION (BS-01): the account already holds 5 YES
        # contracts on TICK-B, and YES leg did NOT fill. The old truthiness
        # check (`if held_b:`) read that stale holding as our fill and
        # reported "executed", leaving NO leg unhedged and the log claiming a
        # complete pair. The delta is 0, so NO leg must be rolled back.
        # RE-PINNED (DR-63) from a three-entry script: a zero delta is only a
        # confirmed non-fill once the LAG RE-READ has also come back zero, so
        # the genuine-non-fill case scripts a fourth, still-unmoved reading.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            TimeoutError("timeout"),  # YES leg raises, truly unfilled
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_no
            ("TICK-B", 5),     # before_yes — pre-existing external position
            ("TICK-B", 5),     # after_yes — unmoved
            ("TICK-B", 5),     # lag re-read — still unmoved
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        assert client.create_order_without_preload_content.call_count == 3

    def test_no_position_at_all_rolls_back(self, monkeypatch):
        # RE-PINNED (DR-63) from a three-entry script — the fourth reading is
        # the lag re-read, still flat because this really is a non-fill.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            TimeoutError("timeout"),  # YES leg raises, truly unfilled
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(None, None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_unexpected_delta_is_manual_review_without_rollback(self):
        # The position moved by +2 but we ordered 7 — unattributable. Never
        # auto-rollback on an outcome we cannot explain.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            TimeoutError("timeout"),  # YES leg raises
        ])
        client.get_positions_without_preload_content = positions_seq(
            None,
            ("TICK-B", 0),
            ("TICK-B", 2),     # +2, but spec.y is 7
        )
        result = _execute_one(client, make_spec(x=7))
        assert result.status == "manual_review"
        assert "delta=2" in result.error
        # No third (rollback) order was submitted
        assert client.create_order_without_preload_content.call_count == 2

    def test_unknown_position_does_not_auto_rollback(self):
        # YES leg raises AND the position lookup itself fails — the fill state
        # is genuinely unknown. Auto-rolling-back here would be wrong if YES leg
        # actually filled: it would sell the NO-leg hedge and leave a naked YES
        # position on B while reporting "rolled_back" (which implies flat).
        # RuntimeError is non-retryable, so api_call_with_retry fails fast and
        # the lookup returns None on the first attempt.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            TimeoutError("timeout"),  # YES leg raises
        ])
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        # No rollback order was submitted — only NO leg and YES leg's attempt
        assert client.create_order_without_preload_content.call_count == 2


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
    def _use_v2(self, v2_mode, v2_mapping_confirmed):
        """The production default path, with the NO-mapping backstop already
        latched so it cannot consume these cases' position scripts."""

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


class TestLegacyShardGuard:
    """The legacy /portfolio/orders endpoint has no shard-routing parameter, so
    while it is the selected path a pair with a leg off DEFAULT_EXCHANGE_INDEX
    must be refused BEFORE anything is submitted. Markets are tagged with their
    shard at ingest, so this guard is the only thing standing between an
    unreachable shard and a misrouted real-money order. The V2 path is exempt:
    every V2 body routes itself via its own market's exchange_index."""

    def test_both_legs_default_shard_is_routable(self):
        assert _legacy_routable(make_spec(shard_a=0, shard_b=0)) is True

    def test_leg_a_off_default_shard_is_not_routable(self):
        assert _legacy_routable(make_spec(shard_a=1, shard_b=0)) is False

    def test_leg_b_off_default_shard_is_not_routable(self):
        assert _legacy_routable(make_spec(shard_a=0, shard_b=1)) is False

    def test_both_legs_off_default_shard_is_not_routable(self):
        assert _legacy_routable(make_spec(shard_a=2, shard_b=2)) is False

    def test_routable_check_is_gated_on_config_constant(self):
        # Not a hardcoded 0 that would silently diverge from config.py.
        assert _legacy_routable(
            make_spec(shard_a=DEFAULT_EXCHANGE_INDEX, shard_b=DEFAULT_EXCHANGE_INDEX)
        ) is True

    def test_legacy_mode_off_shard_spec_fails_before_any_submission(
        self, legacy_mode
    ):
        client = MagicMock()
        result = _execute_one(client, make_spec(shard_b=1))
        assert result.status == "failed", (
            'nothing was submitted, so there is nothing to unwind — "failed" '
            "is the correct status vocabulary, not manual_review"
        )
        assert "shard" in result.error
        # The guard must run BEFORE anything reaches either order path
        client.create_order_without_preload_content.assert_not_called()
        client.rest_client.request.assert_not_called()

    def test_legacy_mode_default_shard_spec_proceeds(self, legacy_mode):
        # Sanity: the guard must not block the ordinary single-shard case.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg
            order_resp("executed"),   # YES leg
        ])
        result = _execute_one(client, make_spec(shard_a=0, shard_b=0))
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 2

    def test_v2_mode_off_shard_spec_proceeds(
        self, v2_mode, v2_mapping_confirmed, monkeypatch
    ):
        # V2 bodies carry their own market's shard, so an off-shard pair is
        # perfectly routable there — the guard must not fire.
        post = MagicMock(side_effect=[v2_resp(5), v2_resp(5)])
        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock()
        result = _execute_one(client, make_spec(shard_a=1))
        assert result.status == "executed"
        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["body"]["exchange_index"] == 1


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
        # Structural guarantee, not just a call count — but asserted PER
        # FUNCTION, not as a module-wide import ban. trader.py legitimately
        # imports api_call_with_retry for the read-only position lookups in
        # _position_count (a GET cannot duplicate a trade, and an unretried
        # transient 429 there escalates a resolvable ambiguity into a rollback
        # or manual_review). What must never be retried is the state-changing
        # side: the two submission paths and the non-idempotent transfer POST.
        for fn in (trader._submit_order, trader._submit_order_v2, trader._execute_transfer):
            assert not _calls_retry_wrapper(fn), (
                f"{fn.__name__} must not be wrapped in retry/backoff — "
                "a retried submission can double-fill and a retried transfer "
                "moves the money twice"
            )
        # The asymmetry is deliberate and is itself pinned: the read-only
        # position lookup DOES retry (see TestPositionCountRetry).
        assert _calls_retry_wrapper(trader._position_count)

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

    def test_linear_cent_cap_equals_legacy_one_cent_slippage(self):
        # On a 1c-grid market the V2 cap must be exactly the legacy intent:
        # scanned price + $0.01 per contract, i.e. the same total buy_max_cost.
        market = make_market("linear_cent")
        price = _v2_limit_price("buy_yes", 0.35, market)
        for count in (1, 5, 17):
            assert price * count * 100 == _buy_max_cost_cents(count, 0.35)

    def test_v2_float_noise_does_not_loosen_the_cap(self):
        # 1.0 - 0.43 == 0.5700000000000001: the cap must be 0.58 (one tick of
        # slippage), not 0.59 — the V2 twin of the legacy round-before-ceil
        # guard pinned by test_float_noise_does_not_loosen_the_cap (TS-03).
        market = make_market("linear_cent")
        assert _v2_limit_price("buy_yes", 1.0 - 0.43, market) == Decimal("0.58")
        # buy_no: NO price 0.30000000000000004 -> cap 0.31 -> YES-book ask 0.69
        assert _v2_limit_price("buy_no", 1.0 - 0.70, market) == Decimal("0.69")

    def test_v2_cap_parity_with_legacy_over_every_whole_cent_bid(self):
        # Every scanned ask is 1.0 - float(bid), so walk all 99 whole-cent bids
        # in exactly the form scanner._bids_to_ask_levels produces and require
        # the V2 cap to equal the legacy one-cent cap to the cent.
        market = make_market("linear_cent")
        for cents in range(2, 100):
            p = 1.0 - cents / 100          # the exact form the scanner produces
            cap = _v2_limit_price("buy_yes", p, market)
            assert cap * 100 == _buy_max_cost_cents(1, p), cents
        # cents == 1 (scanned 0.99) is the one deliberate divergence and is NOT
        # float noise: the legacy cap is $1.00, which is a settlement value and
        # not a tradeable level, so the V2 cap clamps to the top of this
        # market's grid. That clamp is stricter, which is the allowed direction.
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
        # Closing a held NO position is buying the YES short back — a bid —
        # and reduce_only keeps it from ever opening new exposure. The price is
        # NOT a flat top-of-grid bid: it is the legacy limit sell's loss floor
        # mirrored onto the YES book. Default spec nA=0.40 -> floor 40-12=28c
        # -> bid cap 1 - 0.28 = 0.72.
        body = _build_rollback_order_v2(_no_leg(make_spec()))
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "bid"
        assert body["reduce_only"] is True
        assert body["price"] == "0.7200"

    def test_rollback_price_is_the_yes_book_mirror_of_the_legacy_floor(self):
        # One bound, two expressions — this is the invariant that keeps the two
        # order paths from diverging in how much loss an unwind may realize.
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
    """The full legacy outcome matrix, replayed against the V2 order path."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mode, v2_mapping_confirmed):
        """V2 path, with the NO-leg backstop already latched: these cases test
        the state machine, not the first-fill mapping check, and must not have
        their position mocks consumed by it."""

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
        # rollback_failed, and no second order is sent.
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(3)]
        with caplog.at_level(logging.CRITICAL):
            result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "fill_count=3" in result.error
        assert post.call_count == 3
        assert any(
            r.levelno == logging.CRITICAL and "up to 5 NO contracts" in r.getMessage()
            for r in caplog.records
        )

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
        post.side_effect = [v2_resp(5), TimeoutError("timeout")]
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert post.call_count == 2

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


class TestV2NoMappingBackstop:
    """_V2_LEG_SIDE's NO-leg mapping (an `ask` on the YES book OPENS a NO
    position) is doc-derived and unverifiable offline, so the first V2 NO-leg
    fill of a process must prove it: the account position has to MOVE by
    exactly -no_leg.count across the fill (Kalshi's ledger is signed — a long
    NO reads negative). Any other movement disproves the mapping, and the pair
    stops at manual_review with the YES leg unsubmitted and the NO leg
    deliberately left in place. These cases use the same-title default, so the
    NO leg is TICK-A; TestTimeSeriesLegOrder replays the check on TICK-B. The
    latch is shared across pair types — it proves the exchange's side mapping,
    not a market.

    The evidence is the DELTA against _execute_one's up-front NO-leg baseline,
    never the absolute holding — the same rule the rest of the module's
    ambiguity handling follows. The two regression cases below pin why: an
    external LONG position fakes a disproof under an absolute-sign test, and an
    external SHORT one masks a real disproof.

    The backstop's own read is SINGLE-SHOT (_position_count_once), unlike the
    two baselines around it, because it sits in the window where NO leg is
    filled and unhedged. Both readers call the same client method, so the
    call-count assertions below still count every read on one mock; what
    changes is that the backstop's read never retries."""

    @pytest.fixture(autouse=True)
    def _use_v2(self, v2_mode):
        """V2 path with the latch left False — the state a fresh process is in
        on its first trade (the module-level fixture resets it)."""

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
        # An unmoved ledger on BOTH reads after a "filled" NO buy is
        # contradictory (fill reported, position unchanged) — still
        # manual_review, but only after the lag re-read below has had its chance.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-A", 0), ("TICK-A", 0),
        ))
        assert _execute_one(client, make_spec()).status == "manual_review"
        # Two up-front baselines, then BOTH the backstop's first read and its
        # post-delay re-read
        assert client.get_positions_without_preload_content.call_count == 4

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
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

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

    def test_legacy_mode_never_consults_positions_on_a_fill(self, legacy_mode):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"), order_resp("executed"),
        ])
        assert _execute_one(client, make_spec()).status == "executed"
        # The backstop verifies the V2 mapping; on the legacy path there is
        # nothing to verify, and an extra positions read would be pure cost.
        # The two up-front ambiguity baselines are path-independent, so exactly
        # those two reads happen and no third.
        assert client.get_positions_without_preload_content.call_count == 2
        assert trader._V2_NO_MAPPING_CONFIRMED is False


class TestOrderVersionDispatch:
    def test_rollback_dispatcher_floors_the_loss_on_both_paths(self, monkeypatch):
        # _build_rollback_order_any must never hand back an UNPRICED order on
        # either path: an unwind with no proceeds bound realizes an unbounded
        # loss on a book that collapsed since NO leg filled. One bound, two
        # expressions — the legacy NO sell prices AT the floor, the V2 YES
        # buy-back caps at its mirror (1 - floor).
        spec = make_spec(nA=0.62)
        floor_cents = _rollback_floor_cents(_no_leg(spec))

        monkeypatch.setattr(trader, "ORDER_API_VERSION", "legacy")
        legacy = _build_rollback_order_any(_no_leg(spec))
        assert legacy.type == "limit"          # never "market" — no floor there
        assert legacy.no_price == floor_cents
        assert legacy.reduce_only is True
        assert legacy.time_in_force == "fill_or_kill"

        monkeypatch.setattr(trader, "ORDER_API_VERSION", "v2")
        v2 = _build_rollback_order_any(_no_leg(spec))
        assert v2["side"] == "bid"
        assert v2["reduce_only"] is True
        assert v2["time_in_force"] == "immediate_or_cancel"
        assert v2["price"] == _format_price(
            Decimal("1") - Decimal(floor_cents) / Decimal("100")
        )

    def test_legacy_mode_uses_create_order_endpoint_unchanged(self, legacy_mode, monkeypatch):
        posted = MagicMock()
        monkeypatch.setattr(trader, "signed_request_json", posted)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"), order_resp("executed"),
        ])
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 2
        # The V2 route is never touched on the rollback path
        posted.assert_not_called()

    def test_v2_mode_never_touches_legacy_endpoint(
        self, v2_mode, v2_mapping_confirmed, monkeypatch
    ):
        posted = MagicMock(side_effect=[v2_resp(5), v2_resp(5)])
        monkeypatch.setattr(trader, "signed_request_json", posted)
        client = MagicMock()
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 0

    def test_config_default_is_v2(self):
        # The default must be V2; "legacy" is only ever a deliberate rollback.
        assert config.ORDER_API_VERSION == "v2"
        assert trader.ORDER_API_VERSION == "v2"


class TestDropLegacyUnroutable:
    """Regression (adversarial review): on the legacy path, unroutable specs
    must be dropped BEFORE collateral moves — never funded and then refused."""

    def test_v2_mode_is_a_no_op(self, v2_mode):
        portfolio = [make_spec(shard_a=0, shard_b=3)]
        assert trader.drop_legacy_unroutable(portfolio) == portfolio

    def test_legacy_mode_drops_off_shard_specs_with_a_warning(self, legacy_mode, caplog):
        keep = make_spec(shard_a=0, shard_b=0, title="routable")
        drop = make_spec(shard_a=0, shard_b=1, title="off-shard")
        with caplog.at_level(logging.WARNING):
            kept = trader.drop_legacy_unroutable([keep, drop])
        assert kept == [keep]
        assert "before collateral funding" in caplog.text

    def test_legacy_mode_keeps_default_shard_specs(self, legacy_mode):
        portfolio = [make_spec(shard_a=0, shard_b=0)]
        assert trader.drop_legacy_unroutable(portfolio) == portfolio


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
    """

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
    """Pins every field of every V2 body (except the random client_order_id)
    and every field of every legacy request the builders produce for
    make_spec()'s default same-title spec (x=5, nA=0.40, pB=0.35, shard 0), as
    literal values, so nothing a same-title order sends can move by accident.
    A same_title pair buys NO on market_a and YES on market_b.

    Every V2 body carries self_trade_prevention_type (a required field of the
    V2 create-order endpoint), and the reduce_only unwind is
    immediate_or_cancel (the only time in force the endpoint accepts with
    reduce_only)."""

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

    def test_legacy_requests_are_unchanged(self, legacy_mode):
        spec = make_spec()
        no = _build_no_order(_no_leg(spec))
        assert (no.ticker, no.side, no.action, no.type, no.count, no.time_in_force,
                no.buy_max_cost) == ("TICK-A", "no", "buy", "market", 5, "fill_or_kill", 205)
        yes = _build_yes_order(_yes_leg(spec))
        assert (yes.ticker, yes.side, yes.action, yes.type, yes.count,
                yes.time_in_force, yes.buy_max_cost) == (
            "TICK-B", "yes", "buy", "market", 5, "fill_or_kill", 180)
        rb = _build_rollback_order_any(_no_leg(spec))
        assert (rb.ticker, rb.side, rb.action, rb.type, rb.no_price, rb.count,
                rb.time_in_force, rb.reduce_only) == (
            "TICK-A", "no", "sell", "limit", 28, 5, "fill_or_kill", True)

    def test_same_title_baselines_read_market_a_then_market_b(
        self, v2_mode, v2_mapping_confirmed, monkeypatch
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
    """A time_series pair buys NO on the LATER contract (market_b) and YES on
    the EARLIER one (market_a), and the NO leg is always submitted first — so
    for this pair type the whole state machine runs "backwards" across the
    markets: TICK-B is submitted first, baselined first, unwound on failure,
    and is the market the V2 NO-mapping backstop reads. Every expected price is
    derived the way TestV2PriceMath / TestOrderPriceProtection derive theirs,
    from the leg's own scanned price, never from nA/pB."""

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
        self, v2_mode, v2_mapping_confirmed, post
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

    def test_baselines_read_market_b_then_market_a(self, v2_mode, v2_mapping_confirmed, post):
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
        self, v2_mode, v2_mapping_confirmed, post
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
        self, v2_mode, v2_mapping_confirmed, post
    ):
        post.side_effect = [v2_resp(0)]
        result = _execute_one(MagicMock(), self._ts_spec())
        assert result.status == "failed"
        assert "NO leg FoK not filled" in result.error
        assert post.call_count == 1
        assert post.call_args_list[0].kwargs["body"]["ticker"] == "TICK-B"

    def test_no_leg_exception_delta_is_judged_on_market_b(
        self, v2_mode, v2_mapping_confirmed, post
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

    def test_legacy_path_submits_no_on_market_b_then_yes_on_market_a(self, legacy_mode):
        # Both legs on DEFAULT_EXCHANGE_INDEX: the legacy endpoint routes
        # nothing else, and this case is about leg ORDER, not the shard guard.
        spec = self._ts_spec(shard_a=0, shard_b=0)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # NO leg on TICK-B
            order_resp("canceled"),   # YES leg on TICK-A rejected
            order_resp("executed"),   # rollback on TICK-B fills
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        assert _execute_one(client, spec).status == "rolled_back"
        reqs = [
            c.kwargs["create_order_request"]
            for c in client.create_order_without_preload_content.call_args_list
        ]
        assert (reqs[0].ticker, reqs[0].side, reqs[0].action, reqs[0].count) == (
            "TICK-B", "no", "buy", spec.y)
        assert reqs[0].buy_max_cost == _buy_max_cost_cents(spec.y, 0.40)
        assert (reqs[1].ticker, reqs[1].side, reqs[1].action, reqs[1].count) == (
            "TICK-A", "yes", "buy", spec.x)
        assert reqs[1].buy_max_cost == _buy_max_cost_cents(spec.x, 0.30)
        assert (reqs[2].ticker, reqs[2].side, reqs[2].action, reqs[2].type) == (
            "TICK-B", "no", "sell", "limit")
        assert reqs[2].no_price == 40 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
        assert reqs[2].count == spec.y
        assert reqs[2].reduce_only is True

    def test_backstop_reads_market_b_and_confirms(self, v2_mode, post):
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

    def test_backstop_disproof_on_market_b_stops_after_one_post(self, v2_mode, post, caplog):
        post.side_effect = [v2_resp(5), v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None, None, ("TICK-B", 5),   # flat -> +5 on TICK-B: the ask opened YES
        ))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, self._ts_spec())
        assert result.status == "manual_review"
        assert "mapping disproven" in result.error and "TICK-B" in result.error
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
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
    """A clock and a sleep for driving a _WritePacer without real time.

    `advance_on_sleep` decides whether a sleep moves the clock (one caller
    at a time, as a single thread sees it) or leaves it where it is (every
    caller arriving at the same instant, so each wait is its place in line).
    """

    def __init__(self, start: float = 0.0, *, advance_on_sleep: bool = True):
        self.now = start
        self.advance_on_sleep = advance_on_sleep
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.slept.append(seconds)
            if self.advance_on_sleep:
                self.now += seconds


def _fake_pacer(rate, burst, clock: _FakeClock) -> _WritePacer:
    """A _WritePacer on a fake clock."""
    return _WritePacer(rate, burst, clock=clock.monotonic, sleep=clock.sleep)


class TestWritePacer:
    """The token bucket every order and transfer POST waits on: a burst of
    `burst` writes back to back, then one every 1/rate seconds, callers
    queued in the order they take its lock."""

    def test_the_first_burst_does_not_wait(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        assert [pacer.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
        assert clock.slept == []

    def test_the_next_caller_waits_one_over_the_rate(self):
        clock = _FakeClock()
        pacer = _fake_pacer(4, 3, clock)
        for _ in range(3):
            pacer.acquire()
        assert pacer.acquire() == pytest.approx(0.25)
        assert clock.slept == [pytest.approx(0.25)]

    def test_queued_callers_wait_in_turn(self):
        # Every caller arrives at the same instant: the k-th past the burst
        # is admitted k/rate after it, so the waits accumulate.
        clock = _FakeClock(advance_on_sleep=False)
        pacer = _fake_pacer(4, 3, clock)
        waits = [pacer.acquire() for _ in range(7)]
        assert waits == [0.0, 0.0, 0.0,
                         pytest.approx(0.25), pytest.approx(0.5),
                         pytest.approx(0.75), pytest.approx(1.0)]

    def test_a_caller_after_the_queue_drains_waits_only_its_own_turn(self):
        # One caller sleeps its 0.25 s and the clock moves with it; the next
        # caller arrives then and waits one more 1/rate, not the sum of both.
        clock = _FakeClock()
        pacer = _fake_pacer(4, 1, clock)
        assert pacer.acquire() == 0.0
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
        # The pacer reads its clock for the first time on its first acquire,
        # so it refills from there on whatever clock it was given — here one
        # that starts at zero, far below the real monotonic clock.
        clock = _FakeClock(start=0.0, advance_on_sleep=False)
        pacer = _fake_pacer(1, 1, clock)
        assert pacer.acquire() == 0.0
        assert pacer.acquire() == pytest.approx(1.0)
        clock.now = 10.0
        assert pacer.acquire() == 0.0

    @pytest.mark.parametrize(
        "rate", [0, -1, 0.0, float("nan"), float("inf"), True, "8", None],
    )
    def test_an_invalid_rate_raises(self, rate):
        with pytest.raises(ValueError, match="rate"):
            _WritePacer(rate, 8)

    @pytest.mark.parametrize("burst", [0, -1, 1.5, 8.0, True, "8", None])
    def test_an_invalid_burst_raises(self, burst):
        with pytest.raises(ValueError, match="burst"):
            _WritePacer(8, burst)

    def test_the_shipped_pacer_uses_the_config_constants(self):
        # conftest replaces trader._ORDER_WRITE_PACER per test, so the module's
        # construction is checked through a fresh build of the same call.
        pacer = _WritePacer(config.ORDER_WRITES_PER_SECOND, config.ORDER_WRITE_BURST)
        assert pacer._rate == float(config.ORDER_WRITES_PER_SECOND)
        assert pacer._burst == float(config.ORDER_WRITE_BURST)
        assert trader.ORDER_WRITES_PER_SECOND == config.ORDER_WRITES_PER_SECOND
        assert trader.ORDER_WRITE_BURST == config.ORDER_WRITE_BURST

    def test_the_module_pacer_is_built_from_the_config_names(self):
        # Read trader.py's syntax tree, not its text, so a comment or a
        # docstring spelling the call cannot stand in for the assignment: the
        # module binds _ORDER_WRITE_PACER exactly once at top level, to
        # _WritePacer called with the two config names, positionally, and
        # with nothing else — never a literal rate or burst.
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
        pacer = _fake_pacer(2, 1, clock)
        pacer.acquire()
        with caplog.at_level(logging.INFO):
            assert pacer.acquire() == pytest.approx(0.5)
        lines = [r for r in caplog.records if "Pacing order and transfer writes" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert "0.50s" in lines[0].getMessage()

    def test_a_short_wait_is_not_logged(self, caplog):
        clock = _FakeClock()
        pacer = _fake_pacer(8, 1, clock)
        pacer.acquire()
        with caplog.at_level(logging.INFO):
            assert pacer.acquire() == pytest.approx(0.125)
        assert "Pacing order and transfer writes" not in caplog.text

    def test_the_sleep_happens_outside_the_lock(self):
        # One caller's wait must never stop another caller from taking its
        # place in line, so the lock is released before the sleep.
        held = []
        clock = _FakeClock()
        pacer = _WritePacer(
            4, 1, clock=clock.monotonic,
            sleep=lambda s: held.append(pacer._lock.locked()),
        )
        pacer.acquire()
        pacer.acquire()
        assert held == [False]

    def test_concurrent_callers_each_get_their_own_place(self):
        # 20 threads at one frozen instant: the lock must hand out 20 distinct
        # places, so the waits are exactly 0 for the burst and then 1/rate,
        # 2/rate, ... — a lost update would repeat a wait. A lost update is a
        # race, which a run can miss, so the clock also records whether the
        # lock is held each time the pacer reads it: acquire() reads the clock
        # first thing in the block that updates the balance, so a reading
        # taken without the lock means that block is not guarded.
        clock = _FakeClock(advance_on_sleep=False)
        rate, burst, callers = 4, 3, 20
        lock_held: list[bool] = []

        def locked_clock():
            lock_held.append(pacer._lock.locked())
            return clock.monotonic()

        pacer = _WritePacer(rate, burst, clock=locked_clock, sleep=clock.sleep)
        start = threading.Barrier(callers)
        waits: list[float] = []
        waits_lock = threading.Lock()

        def run():
            start.wait()
            wait = pacer.acquire()
            with waits_lock:
                waits.append(wait)

        threads = [threading.Thread(target=run) for _ in range(callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expected = [0.0] * burst + [k / rate for k in range(1, callers - burst + 1)]
        assert sorted(waits) == [pytest.approx(w) for w in expected]
        assert lock_held == [True] * callers

    def test_real_threads_are_never_admitted_faster_than_the_bucket(self):
        # Real time and real sleeps, at a rate high enough to finish at once.
        # Each thread's admission time is the clock reading the pacer took
        # under its lock plus the wait it returned, so the check reads the
        # pacer's own schedule rather than when the OS happened to wake a
        # thread. In any stretch from one admission to a later one, the
        # bucket admits at most burst + rate * (stretch).
        rate, burst, callers = 400.0, 4, 20
        local = threading.local()

        def clock():
            local.now = time.monotonic()
            return local.now

        pacer = _WritePacer(rate, burst, clock=clock)
        start = threading.Barrier(callers)
        admitted: list[float] = []
        returned: list[tuple[float, float]] = []
        lock = threading.Lock()

        def run():
            start.wait()
            wait = pacer.acquire()
            done = time.monotonic()
            with lock:
                admitted.append(local.now + wait)
                returned.append((local.now + wait, done))

        threads = [threading.Thread(target=run) for _ in range(callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(admitted) == callers
        admitted.sort()
        for i in range(callers):
            for j in range(i, callers):
                assert j - i + 1 <= burst + rate * (admitted[j] - admitted[i]) + 1e-6
        # Nobody returned before its admission time: the sleep covered it
        for due, done in returned:
            assert done >= due - 1e-3


class TestWritesArePaced:
    """Every order and transfer POST takes one place on the pacer, before
    the POST, and is still sent exactly once."""

    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def pacer(self, monkeypatch, events):
        """A stand-in pacer that records each acquire in `events`."""
        mock = MagicMock()
        mock.acquire.side_effect = lambda: events.append("acquire") or 0.0
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

    def test_the_v2_log_line_is_written_after_the_wait(self, pacer, post, caplog):
        # The "Submitting V2 order" line's time is the send time, so a pacing
        # wait shows up as a gap before it, never between it and the POST.
        logged_before_wait = []
        pacer.acquire.side_effect = lambda: logged_before_wait.append(
            "Submitting V2 order" in caplog.text
        ) or 0.0
        post.replies.append(v2_resp(5))
        with caplog.at_level(logging.INFO):
            _submit_order_v2(MagicMock(), _build_no_order_v2(_no_leg(make_spec())))
        assert logged_before_wait == [False]
        assert "Submitting V2 order" in caplog.text

    def test_a_legacy_order_waits_before_its_post(self, pacer, events):
        client = MagicMock()

        def create(**kwargs):
            events.append("post")
            return order_resp("executed")

        client.create_order_without_preload_content.side_effect = create
        assert _submit_order(client, _build_no_order(_no_leg(make_spec()))) == "executed"
        assert events == ["acquire", "post"]

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
        # Pacing is not a retry: a 429 that comes back anyway raises into the
        # caller's ambiguous path exactly as before.
        err = self._too_many_requests()
        post.replies.append(err)
        with pytest.raises(ApiException) as exc_info:
            _submit_order_v2(MagicMock(), _build_yes_order_v2(_yes_leg(make_spec())))
        assert exc_info.value is err
        assert events == ["acquire", "post"]

    def test_a_legacy_429_takes_one_place_and_one_post(self, pacer, events):
        client = MagicMock()

        def create(**kwargs):
            events.append("post")
            return SimpleNamespace(
                status=429, reason="Too Many Requests",
                data=b'{"error":{"code":"too_many_requests","message":"too many requests"}}',
                getheaders=lambda: {"content-type": "application/json"},
            )

        client.create_order_without_preload_content.side_effect = create
        with pytest.raises(ApiException):
            _submit_order(client, _build_no_order(_no_leg(make_spec())))
        assert events == ["acquire", "post"]

    def test_a_transfer_429_takes_one_place_and_one_post(self, pacer, post, events):
        post.replies.append(self._too_many_requests())
        with pytest.raises(ApiException):
            _execute_transfer(MagicMock(), 0, 1, 100)
        assert events == ["acquire", "post"]

    def test_a_429_on_the_yes_leg_paces_the_rollback_too(
        self, pacer, post, events, v2_mode, v2_mapping_confirmed, monkeypatch,
    ):
        # One pair whose YES leg the exchange rejects with a 429: the position
        # check finds nothing filled and the NO leg is unwound. Each of the
        # three orders takes its own place on the pacer before its own POST,
        # and none is sent twice.
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
        assert events == ["acquire", "post"] * 3
        assert post.call_args_list[2].kwargs["body"]["reduce_only"] is True

    def test_a_dry_run_takes_no_place(self, pacer):
        results = execute_trades(MagicMock(), [make_spec()], dry_run=True)
        assert [r.status for r in results] == ["simulated"]
        pacer.acquire.assert_not_called()


class TestExecuteTradesArePaced:
    """execute_trades runs its pairs on concurrent workers, and every one of
    their POSTs goes through the one shared pacer, so seven pairs' 14 orders
    leave no faster than the account's write limit instead of inside one
    second."""

    RATE = 2
    BURST = 2
    PAIRS = 7

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

    def test_the_pairs_orders_are_spread_to_the_write_limit(
        self, monkeypatch, v2_mode, v2_mapping_confirmed,
    ):
        # A frozen clock: every worker arrives at the same instant, so each
        # order's admission time is exactly the wait the pacer gave it. Each
        # worker records the wait of its own last acquire, and each POST reads
        # it off that worker, which proves the POST came after the acquire.
        clock = _FakeClock(advance_on_sleep=False)
        shared = _fake_pacer(self.RATE, self.BURST, clock)
        local = threading.local()

        class _Recording:
            def acquire(self):
                wait = shared.acquire()
                local.admitted = wait
                return wait

        monkeypatch.setattr(trader, "_ORDER_WRITE_PACER", _Recording())
        admissions: list = []
        lock = threading.Lock()

        def post(client, method, path, body):
            admitted = local.__dict__.pop("admitted", None)
            with lock:
                admissions.append(admitted)
            return v2_resp(5)

        monkeypatch.setattr(trader, "signed_request_json", post)
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp()
        )

        results = execute_trades(client, self._specs(self.PAIRS), dry_run=False)

        assert [r.status for r in results] == ["executed"] * self.PAIRS
        orders = 2 * self.PAIRS
        assert len(admissions) == orders
        assert None not in admissions            # every POST had its own acquire
        admissions.sort()
        # The burst goes at once, then one order every 1/rate seconds
        expected = [max(0.0, (i - self.BURST + 1) / self.RATE) for i in range(orders)]
        assert admissions == [pytest.approx(a) for a in expected]
        # No 1-second window holds more than burst + rate orders
        for start in admissions:
            in_window = [a for a in admissions if start <= a <= start + 1.0 + 1e-9]
            assert len(in_window) <= self.BURST + self.RATE
        # Each worker slept its own wait, outside the POST
        assert sorted(clock.slept) == [pytest.approx(a) for a in expected if a > 0]
