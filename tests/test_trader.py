"""Tests for trader.py — order construction, rollback verification, and
exception disambiguation. All Kalshi API interaction is mocked per project
policy (tests must run offline)."""
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from kalshi_betting.config import BUY_MAX_COST_SLIPPAGE_CENTS, DEFAULT_EXCHANGE_INDEX
from kalshi_betting.trader import (
    _build_no_order,
    _build_yes_order,
    _execute_one,
    _legacy_routable,
)


def make_spec(
    x: int = 5,
    nA: float = 0.40,
    pB: float = 0.35,
    shards: tuple[int, int] = (0, 0),
) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads.

    `shards` pins REAL ints on both legs' exchange_index. This must never be
    left to MagicMock's auto-attributes: an auto-attr is a truthy Mock object
    that compares unequal to DEFAULT_EXCHANGE_INDEX, which would make every
    spec look non-routable (or, with a different comparison, silently defeat
    the shard check entirely).
    """
    pair = MagicMock()
    pair.market_a.ticker = "TICK-A"
    pair.market_a.title = "Market A"
    pair.market_a.exchange_index = shards[0]
    pair.market_b.ticker = "TICK-B"
    pair.market_b.title = "Market B"
    pair.market_b.exchange_index = shards[1]
    pair.nA = nA
    pair.pB = pB
    pair.canonical_title = "test pair"
    spec = MagicMock()
    spec.pair = pair
    spec.x = x
    spec.y = x
    return spec


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


class TestRollbackVerification:
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
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_rollback_order_is_reduce_only_sell(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),
            order_resp("canceled"),
            order_resp("executed"),
        ])
        _execute_one(client, make_spec())
        rollback_call = client.create_order_without_preload_content.call_args_list[2]
        rollback_req = rollback_call.kwargs["create_order_request"]
        assert rollback_req.action == "sell"
        assert rollback_req.side == "no"
        assert rollback_req.reduce_only is True


class TestExceptionDisambiguation:
    def test_leg_a_exception_with_no_position_is_failed(self):
        # Exception + confirmed zero position → clean failure, no rollback sent
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert client.create_order_without_preload_content.call_count == 1

    def test_leg_a_exception_with_position_is_unwound(self):
        # Exception but the position exists (timeout AFTER the fill) — the
        # half-filled pair must be unwound, not abandoned as "failed".
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            TimeoutError("timeout"),  # leg A raises after actually filling
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-A", position=-5)
        )
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_leg_b_exception_with_position_is_executed(self):
        # Leg B raises but the YES position exists — the pair actually
        # completed; rolling back leg A would REVERSE the unhedged exposure.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises after actually filling
        ])
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-B", position=5)
        )
        result = _execute_one(client, make_spec())
        assert result.status == "executed"
        # No rollback order was submitted
        assert client.create_order_without_preload_content.call_count == 2

    def test_leg_b_exception_with_no_position_rolls_back(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises, truly unfilled
            order_resp("executed"),   # rollback fills
        ])
        client.get_positions_without_preload_content = MagicMock(return_value=positions_resp())
        result = _execute_one(client, make_spec())
        assert result.status == "rolled_back"

    def test_leg_b_exception_with_unknown_position_does_not_auto_rollback(self):
        # Leg B raises AND the position lookup itself fails — the fill state
        # is genuinely unknown (not confirmed zero, not confirmed non-zero).
        # Auto-rolling-back here would be wrong if leg B actually filled: it
        # would sell the leg-A hedge and leave a naked YES position on B while
        # reporting "rolled_back" (which implies flat). Must surface for
        # manual review instead of guessing.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            TimeoutError("timeout"),  # leg B raises
        ])
        client.get_positions_without_preload_content = MagicMock(side_effect=RuntimeError("lookup failed"))
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        # No rollback order was submitted — only leg A and leg B's attempt
        assert client.create_order_without_preload_content.call_count == 2


class TestLegacyShardRouting:
    """TEMPORARY guard, dies with the V2 order migration: the legacy
    /portfolio/orders endpoint has no shard routing, so a pair with a leg off
    DEFAULT_EXCHANGE_INDEX must be refused BEFORE anything is submitted.
    The scanner deliberately ingests such markets (market data is cross-shard),
    so this is the only thing standing between them and a misrouted order."""

    def test_both_legs_shard_zero_is_routable(self):
        assert _legacy_routable(make_spec(shards=(0, 0))) is True

    def test_leg_a_off_shard_zero_is_not_routable(self):
        assert _legacy_routable(make_spec(shards=(1, 0))) is False

    def test_leg_b_off_shard_zero_is_not_routable(self):
        assert _legacy_routable(make_spec(shards=(0, 1))) is False

    def test_both_legs_off_shard_zero_is_not_routable(self):
        assert _legacy_routable(make_spec(shards=(2, 2))) is False

    def test_routable_check_is_gated_on_config_constant(self):
        # Not a hardcoded 0 that would silently diverge from config.py.
        assert _legacy_routable(
            make_spec(shards=(DEFAULT_EXCHANGE_INDEX, DEFAULT_EXCHANGE_INDEX))
        ) is True

    def test_non_routable_spec_submits_nothing_and_fails(self):
        client = MagicMock()
        result = _execute_one(client, make_spec(shards=(0, 1)))
        assert result.status == "failed", (
            'nothing was submitted, so there is nothing to unwind — "failed" '
            "is the correct status vocabulary, not manual_review"
        )
        assert "exchange shard" in result.error
        assert "V2 migration pending" in result.error
        # The guard must run BEFORE any order goes out
        client.create_order_without_preload_content.assert_not_called()

    def test_non_routable_leg_a_also_submits_nothing(self):
        client = MagicMock()
        result = _execute_one(client, make_spec(shards=(3, 0)))
        assert result.status == "failed"
        client.create_order_without_preload_content.assert_not_called()

    def test_shard_zero_pair_proceeds_to_submission(self):
        # Sanity: the guard must not block the ordinary single-shard case.
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),   # leg A
            order_resp("executed"),   # leg B
        ])
        result = _execute_one(client, make_spec(shards=(0, 0)))
        assert result.status == "executed"
        assert client.create_order_without_preload_content.call_count == 2
