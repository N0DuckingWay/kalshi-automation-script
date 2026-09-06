"""Tests for trader.py — order construction (both the V2 and the retained
legacy endpoint), V2 price/tick math, rollback verification and its loss
floor, exception disambiguation, and the cross-shard collateral transfer
machinery. All Kalshi API interaction is mocked per project policy (tests must
run offline).

Ambiguity handling is DELTA-based: _execute_one reads BOTH legs' baseline
positions up front, before either order is submitted, and compares each against
a reading taken after an exception, attributing the outcome to the change.
Mocks therefore sequence get_positions responses with side_effect (see
positions_seq) rather than returning one flat payload — a single return_value
would make before and after identical, i.e. delta 0. Every _execute_one call
consumes TWO baseline reads before anything else, so a mock sequence written
for the old read-on-demand protocol will fail with StopIteration or a wrong
status; read such failures through that lens first.

The merged default is ORDER_API_VERSION="v2", so any class that drives
_execute_one down the legacy path opts in via the `legacy_mode` fixture.
"""
import ast
import inspect
import json
import logging
import math
import textwrap
import uuid
from decimal import Decimal
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
    _legacy_routable,
    _partition_by_funding,
    _plan_transfers,
    _position_count,
    _required_cents_by_shard,
    _rollback_floor_cents,
    _transfers_active,
    _unfunded_shards,
    _v2_fill_status,
    _v2_limit_price,
    _v2_rollback_price,
    _v2_top_of_grid_price,
    ensure_shard_collateral,
    execute_trades,
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
    structure: str = "",
    ranges: list | None = None,
    shard_a: int = 0,
    shard_b: int = 0,
    cost_a: float = 0.0,
    cost_b: float = 0.0,
    title: str = "test pair",
) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads.

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
    pair.nA = nA
    pair.pB = pB
    pair.canonical_title = title
    spec = MagicMock()
    spec.pair = pair
    spec.x = x
    spec.y = x
    spec.cost_with_fees_a = cost_a
    spec.cost_with_fees_b = cost_b
    return spec


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
    reads BOTH baselines up front — before either order is submitted, so no
    blocking call sits in the unhedged window between leg A's fill and leg B's
    submission — and then once more after an ambiguous leg, so the script is
    consumed in that order:
        before_a, before_b, [backstop], [after_a], [after_b]

    The optional [backstop] slot is the V2 NO-mapping check's own single-shot
    read (see TestV2NoMappingBackstop); it hits the same client method, so it
    consumes a script entry like any other, but only on the V2 path and only
    while the mapping is unlatched.

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
        order = _build_no_order(spec)
        expected = math.ceil(5 * 0.40 * 100) + 5 * BUY_MAX_COST_SLIPPAGE_CENTS
        assert order.buy_max_cost == expected
        assert order.side == "no"
        assert order.action == "buy"
        assert order.time_in_force == "fill_or_kill"

    def test_yes_leg_has_buy_max_cost(self):
        spec = make_spec(x=5, pB=0.35)
        order = _build_yes_order(spec)
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
    """The leg-A unwind is a floored FoK limit sell, not an unbounded market sell."""

    def test_floor_is_entry_less_max_loss(self):
        spec = make_spec(nA=0.62)
        assert _rollback_floor_cents(spec) == 62 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT
        # Literal guard: at the current calibration (BS-05, 12 cents to cover
        # the full bid-ask spread plus adverse movement) this must be 50 cents
        # exactly, so a silent change to the constant fails this test even
        # though the assertion above would float along with it.
        assert ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT == 12
        assert _rollback_floor_cents(spec) == 50

    def test_floor_rounds_before_truncating(self):
        # 0.57 is stored as 0.5699999999999998, so 0.57 * 100 == 56.99999999999999.
        # int() alone would truncate the entry to 56 and floor a cent too low.
        spec = make_spec(nA=0.57)
        assert _rollback_floor_cents(spec) == 57 - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT

    def test_floor_clamped_to_valid_limit_price(self):
        # 3 - 5 would be a negative limit price the API rejects outright.
        assert _rollback_floor_cents(make_spec(nA=0.03)) == 1
        # And the upper clamp keeps the price inside the API's 1..99 range.
        assert _rollback_floor_cents(make_spec(nA=1.20)) == 99


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
    leg-A fill (see TestV2NoMappingBackstop). Classes exercising the rollback /
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
        # Leg A fills, leg B FoK is rejected, and the rollback FoK is ALSO
        # rejected — the orphaned leg-A position must surface as
        # "rollback_failed", never be logged away as a successful rollback.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            order_resp("canceled"),   # leg B rejected (confirmed non-fill)
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
            order_resp("executed"),   # leg A
            order_resp("canceled"),   # leg B rejected
            order_resp("executed"),   # rollback filled
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_rollback_order_is_floored_reduce_only_limit_sell(self):
        # The unwind must be a LIMIT sell carrying a proceeds floor: a market
        # sell has no such knob, so a collapsed book would realize an unbounded
        # loss on a position we only hold because leg B failed.
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
            order_resp("executed"),   # leg A
            order_resp("canceled"),   # leg B rejected
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
            order_resp("executed"),   # leg A
            order_resp("executed"),   # leg B
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 2

    def test_leg_a_fok_rejection_is_failed_without_position_check(self):
        # A clean FoK rejection is a confirmed non-fill — no ambiguity
        # snapshot, no rollback, and leg B is never submitted.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(
            return_value=order_resp("canceled")
        )
        client.get_positions_without_preload_content = positions_seq(None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert "Leg A FoK not filled" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        # Only the two up-front baselines were read — no ambiguity snapshot
        assert client.get_positions_without_preload_content.call_count == 2

    def test_both_baselines_are_read_before_any_order_is_submitted(self):
        # The unhedged window is the gap between leg A's fill and leg B's
        # submission. A position read in there is a blocking network call that
        # can burn the full ~62s retry schedule while the account holds a naked
        # NO on market A, so BOTH baselines must be taken up front. Leg B's is
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


class TestLegAExceptionDisambiguation:
    """Leg A raised: the outcome is attributed to the position DELTA."""

    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """Exercises the legacy submission path's exception handling."""

    def test_no_movement_is_failed(self):
        # Exception + position unchanged → confirmed non-fill, no rollback sent
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        # before_a, before_b (both up front), then after_a
        client.get_positions_without_preload_content = positions_seq(None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert client.create_order_without_preload_content.call_count == 1

    def test_external_no_position_unchanged_is_failed_not_unwound(self):
        # REGRESSION (BS-01): the account already holds 10 NO contracts on
        # TICK-A from an earlier run, and our order genuinely did not fill.
        # The old absolute check (held_a != 0) unwound that unrelated holding;
        # the delta is 0, so this must be a clean "failed" with NO sell order.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_a
            None,              # before_b (taken up front, unused here)
            ("TICK-A", -10),   # after_a — unmoved
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
            TimeoutError("timeout"),  # leg A raises after actually filling
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", -10),   # before_a
            None,              # before_b (taken up front, unused here)
            ("TICK-A", -15),   # after_a — moved by -5 == -spec.x
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        # Exactly one submission attempt per leg-A order plus the rollback
        assert client.create_order_without_preload_content.call_count == 2

    def test_unattributable_delta_is_manual_review(self):
        # The position moved, but by an amount our order cannot explain (an
        # unrelated trade landed in the snapshot window). A reduce-only sell
        # would liquidate a position we may not own — surface it instead.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            ("TICK-A", 0),     # before_a
            None,              # before_b (taken up front, unused here)
            ("TICK-A", -3),    # after_a — -3, but spec.x is 7
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
        # snapshot must run OUTSIDE leg A's except block. Inside it, the
        # RuntimeError would inherit the submission's TimeoutError as
        # __context__, api_call_with_retry's cause-chain walk would classify it
        # as transient, and this decision would stall for the full ~62s backoff.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(
            None,                            # before_a
            None,                            # before_b (taken up front)
            RuntimeError("lookup failed"),   # after_a — non-retryable → fail fast
        )
        with patch.object(_http.time, "sleep") as sleep:
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "delta=None" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        assert client.get_positions_without_preload_content.call_count == 3
        sleep.assert_not_called()


class TestLegBExceptionDisambiguation:
    """Leg B raised: same delta protocol, but never auto-rollback on unknown."""

    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """Exercises the legacy submission path's exception handling."""

    def test_delta_of_our_yes_buy_is_executed(self):
        # Leg B raises but the position moved by exactly +spec.y — the pair
        # actually completed; rolling back leg A would REVERSE the hedge.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises after actually filling
        ])
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_a
            None,              # before_b
            ("TICK-B", 5),     # after_b — moved by +5 == spec.y
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "executed"
        # No rollback order was submitted
        assert client.create_order_without_preload_content.call_count == 2

    def test_external_yes_position_unchanged_rolls_back(self):
        # HEADLINE REGRESSION (BS-01): the account already holds 5 YES
        # contracts on TICK-B, and leg B did NOT fill. The old truthiness
        # check (`if held_b:`) read that stale holding as our fill and
        # reported "executed", leaving leg A unhedged and the log claiming a
        # complete pair. The delta is 0, so leg A must be rolled back.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises, truly unfilled
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(
            None,              # before_a
            ("TICK-B", 5),     # before_b — pre-existing external position
            ("TICK-B", 5),     # after_b — unmoved
        )
        result = _execute_one(client, make_spec(x=5))
        assert result.status == "rolled_back"
        assert client.create_order_without_preload_content.call_count == 3

    def test_no_position_at_all_rolls_back(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises, truly unfilled
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = positions_seq(None, None, None)
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_unexpected_delta_is_manual_review_without_rollback(self):
        # The position moved by +2 but we ordered 7 — unattributable. Never
        # auto-rollback on an outcome we cannot explain.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises
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
        # Leg B raises AND the position lookup itself fails — the fill state
        # is genuinely unknown. Auto-rolling-back here would be wrong if leg B
        # actually filled: it would sell the leg-A hedge and leave a naked YES
        # position on B while reporting "rolled_back" (which implies flat).
        # RuntimeError is non-retryable, so api_call_with_retry fails fast and
        # the lookup returns None on the first attempt.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises
        ])
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        # No rollback order was submitted — only leg A and leg B's attempt
        assert client.create_order_without_preload_content.call_count == 2


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
            order_resp("executed"),   # leg A
            order_resp("executed"),   # leg B
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
        body = _build_no_order_v2(make_spec(x=5, nA=0.40))
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "ask"
        assert body["price"] == "0.5900"

    def test_yes_leg_is_bid_at_capped_yes_price(self):
        body = _build_yes_order_v2(make_spec(x=5, pB=0.35))
        assert body["ticker"] == "TICK-B"
        assert body["side"] == "bid"
        assert body["price"] == "0.3600"

    def test_rollback_is_reduce_only_bid_at_the_loss_floored_price(self):
        # Closing a held NO position is buying the YES short back — a bid —
        # and reduce_only keeps it from ever opening new exposure. The price is
        # NOT a flat top-of-grid bid: it is the legacy limit sell's loss floor
        # mirrored onto the YES book. Default spec nA=0.40 -> floor 40-12=28c
        # -> bid cap 1 - 0.28 = 0.72.
        body = _build_rollback_order_v2(make_spec())
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "bid"
        assert body["reduce_only"] is True
        assert body["price"] == "0.7200"

    def test_rollback_price_is_the_yes_book_mirror_of_the_legacy_floor(self):
        # One bound, two expressions — this is the invariant that keeps the two
        # order paths from diverging in how much loss an unwind may realize.
        for nA in (0.40, 0.57, 0.62, 0.85):
            spec = make_spec(nA=nA)
            floor_cents = _rollback_floor_cents(spec)
            assert _v2_rollback_price(spec) == (
                Decimal("1") - Decimal(floor_cents) / Decimal("100")
            )

    def test_rollback_price_never_exceeds_the_markets_top_of_grid(self):
        # A very cheap leg A clamps the loss floor to 1c, which mirrors to a
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
        assert _v2_rollback_price(make_spec(nA=0.05)) == Decimal("0.99")
        for structure, bands in (
            ("linear_cent", None),
            ("deci_cent", DECI_CENT_BANDS),
            ("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS),
        ):
            spec = make_spec(nA=0.05, structure=structure, ranges=bands)
            assert _v2_rollback_price(spec) <= _v2_top_of_grid_price(spec.pair.market_a)

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
            price = _v2_rollback_price(make_spec(nA=0.40, structure=structure, ranges=bands))
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

    def test_all_legs_fill_or_kill(self):
        spec = make_spec()
        for body in (
            _build_no_order_v2(spec), _build_yes_order_v2(spec), _build_rollback_order_v2(spec),
        ):
            assert body["time_in_force"] == "fill_or_kill"
            assert body["post_only"] is False

    def test_each_leg_carries_its_own_markets_shard(self):
        # Per-leg routing: a pair's two legs can live on different shards, so
        # each body takes exchange_index from its OWN market — and the rollback
        # routes to market A, the shard the position was opened on. Never -1
        # (auto-route): a wrong shard must be rejected loudly by the exchange,
        # not silently papered over.
        spec = make_spec(shard_a=2, shard_b=3)
        no_body = _build_no_order_v2(spec)
        yes_body = _build_yes_order_v2(spec)
        rollback_body = _build_rollback_order_v2(spec)
        assert no_body["exchange_index"] == 2
        assert yes_body["exchange_index"] == 3
        assert rollback_body["exchange_index"] == 2
        for body in (no_body, yes_body, rollback_body):
            assert body["exchange_index"] != -1

    def test_default_shard_spec_carries_the_default_exchange_index(self):
        # The universal case today: everything is on DEFAULT_EXCHANGE_INDEX.
        spec = make_spec()
        for body in (
            _build_no_order_v2(spec), _build_yes_order_v2(spec), _build_rollback_order_v2(spec),
        ):
            assert body["exchange_index"] == DEFAULT_EXCHANGE_INDEX

    def test_client_order_ids_are_unique_uuids(self):
        spec = make_spec()
        ids = [
            _build_no_order_v2(spec)["client_order_id"],
            _build_yes_order_v2(spec)["client_order_id"],
            _build_rollback_order_v2(spec)["client_order_id"],
            _build_no_order_v2(spec)["client_order_id"],
        ]
        assert len(set(ids)) == len(ids)
        for cid in ids:
            uuid.UUID(cid)  # raises if not a valid UUID string

    def test_counts_serialized_as_fixed_point_strings(self):
        spec = make_spec(x=7)
        assert _build_no_order_v2(spec)["count"] == "7.00"
        assert _build_yes_order_v2(spec)["count"] == "7.00"
        assert _build_rollback_order_v2(spec)["count"] == "7.00"

    def test_buy_legs_are_not_reduce_only(self):
        # Only the unwind closes exposure; a reduce_only buy leg would never fill.
        spec = make_spec()
        assert _build_no_order_v2(spec)["reduce_only"] is False
        assert _build_yes_order_v2(spec)["reduce_only"] is False


class TestV2FillStatus:
    def test_full_fill_plain_int_is_executed(self):
        assert _v2_fill_status({"order": {"fill_count": 10}}, 10) == "executed"

    def test_full_fill_fp_string_is_executed(self):
        assert _v2_fill_status({"order": {"fill_count_fp": "10.00"}}, 10) == "executed"

    def test_zero_fill_is_canceled(self):
        assert _v2_fill_status({"order": {"fill_count": 0}}, 10) == "canceled"
        assert _v2_fill_status({"order": {"fill_count_fp": "0.00"}}, 10) == "canceled"

    def test_partial_fill_raises_for_ambiguous_path(self):
        # A partial fill violates the fill-or-kill invariant, so the fill state
        # is not trustworthy — raising routes _execute_one into the position
        # lookup instead of reporting a clean fill or a clean kill.
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
        # Loss-floored, not a flat top-of-grid bid: default spec nA=0.40 ->
        # floor 40-12=28c -> bid cap 1 - 0.28 = 0.72 on the $0.01 grid
        assert rollback_body["price"] == "0.7200"

    def test_v2_unfilled_rollback_is_rollback_failed(self, post):
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(0)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error

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

    def test_v2_leg_a_exception_with_no_position_is_failed(self, post):
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
        # actually completed and rolling leg A back would REVERSE the hedge.
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

    def test_v2_exactly_one_post_per_leg_no_retry_on_5xx(self, post):
        # A 5xx on an order submission must NEVER be retried: a second FoK
        # could fill the leg twice at a different price.
        post.side_effect = ApiException(status=500, reason="server error")
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert post.call_count == 1


class TestV2NoMappingBackstop:
    """_V2_LEG_SIDE's NO-leg mapping (an `ask` on the YES book OPENS a NO
    position) is doc-derived and unverifiable offline, so the first V2 leg-A
    fill of a process must prove it: the account position has to MOVE by
    exactly -spec.x across the fill (Kalshi's ledger is signed — a long NO
    reads negative). Any other movement disproves the mapping, and the pair
    stops at manual_review with leg B unsubmitted and leg A deliberately left
    in place.

    The evidence is the DELTA against _execute_one's up-front leg-A baseline,
    never the absolute holding — the same rule the rest of the module's
    ambiguity handling follows. The two regression cases below pin why: an
    external LONG position fakes a disproof under an absolute-sign test, and an
    external SHORT one masks a real disproof.

    The backstop's own read is SINGLE-SHOT (_position_count_once), unlike the
    two baselines around it, because it sits in the window where leg A is
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
            None,                    # before_a baseline: flat
            None,                    # before_b baseline
            ("TICK-A", -5),          # backstop: moved by -5, our 5-contract NO buy
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        # Leg B still went out — the check must not disturb the state machine
        assert post.call_count == 2
        assert trader._V2_NO_MAPPING_CONFIRMED is True

    def test_external_long_position_still_confirms_via_the_delta(self, post):
        # Regression: the account already holds +100 YES on market A from an
        # earlier run or a manual trade. Our 10-contract NO buy nets it to +90,
        # which is POSITIVE — the old absolute-sign test read that as "mapping
        # disproven" and halted the pair at manual_review with a real, unhedged
        # leg-A position open. The delta (-10) is unambiguous and confirms.
        post.side_effect = [v2_resp(10, 10), v2_resp(10, 10)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            ("TICK-A", 100),         # before_a: external long YES
            None,                    # before_b baseline
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
            ("TICK-A", -100),        # before_a: external short
            None,                    # before_b baseline
            ("TICK-A", -95),         # backstop: moved +5, the wrong direction
        ))
        with caplog.at_level(logging.INFO, logger="root"):
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "mapping disproven" in result.error
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
        assert trader._V2_NO_MAPPING_CONFIRMED is False
        # Leg B was never submitted, so a false confirmation cannot have latched
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
        # of falsely halting at manual_review with a real unhedged leg-A
        # position left open on an unattended run.
        slept = []
        monkeypatch.setattr(trader.time, "sleep", lambda s: slept.append(s))
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                          # before_a baseline
            None,                          # before_b baseline
            ("TICK-A", 0),                 # lagging first read -> delta 0
            ("TICK-A", -5.0),              # ledger catches up -> delta -5
        ))
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is True
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    def test_zero_then_failed_reread_proceeds_unlatched(self, post, monkeypatch):
        # A zero delta then a failed re-read is UNKNOWN, not disproven —
        # proceed to leg B unlatched, same as a failed first read.
        monkeypatch.setattr(trader.time, "sleep", lambda s: None)
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            None,                          # before_a baseline
            None,                          # before_b baseline
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
        # Only leg A went out: leg B is not submitted (it would hedge a
        # position we don't hold) and NO unwind is attempted (the unwind is a
        # bid resting on the same disproven hypothesis).
        assert post.call_count == 1
        only_body = post.call_args_list[0].kwargs["body"]
        assert only_body["ticker"] == "TICK-A"
        assert only_body["side"] == "ask"
        # Leg A's position is left exactly as it is for a human to flatten
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
        # The backstop's own read succeeds, but the leg-A BASELINE failed, so
        # no delta exists. That is unknown — proceed unlatched rather than
        # judging the absolute reading, which is exactly what this check is not
        # allowed to do.
        post.side_effect = [v2_resp(5), v2_resp(5)]
        client = MagicMock(get_positions_without_preload_content=positions_seq(
            RuntimeError("baseline lookup failed"),   # before_a -> None
            None,                                     # before_b baseline
            ("TICK-A", -5),                           # backstop read succeeds
        ))
        assert _execute_one(client, make_spec()).status == "executed"
        assert trader._V2_NO_MAPPING_CONFIRMED is False

    def test_backstop_read_is_single_shot_and_never_retried(self, post):
        # The backstop's read is the ONLY blocking call inside the window where
        # leg A is filled and unhedged, so it must not carry
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
            None,                              # trade 1: before_a baseline
            None,                              # trade 1: before_b baseline
            RuntimeError("lookup failed"),     # trade 1: backstop read fails
            None,                              # trade 2: before_a baseline
            None,                              # trade 2: before_b baseline
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
        # loss on a book that collapsed since leg A filled. One bound, two
        # expressions — the legacy NO sell prices AT the floor, the V2 YES
        # buy-back caps at its mirror (1 - floor).
        spec = make_spec(nA=0.62)
        floor_cents = _rollback_floor_cents(spec)

        monkeypatch.setattr(trader, "ORDER_API_VERSION", "legacy")
        legacy = _build_rollback_order_any(spec)
        assert legacy.type == "limit"          # never "market" — no floor there
        assert legacy.no_price == floor_cents
        assert legacy.reduce_only is True
        assert legacy.time_in_force == "fill_or_kill"

        monkeypatch.setattr(trader, "ORDER_API_VERSION", "v2")
        v2 = _build_rollback_order_any(spec)
        assert v2["side"] == "bid"
        assert v2["reduce_only"] is True
        assert v2["time_in_force"] == "fill_or_kill"
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
    burn the timeout or miscast settled transfers as money-in-flight."""

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
        """Three specs whose leg-A tickers are distinct, so a fake worker can
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
