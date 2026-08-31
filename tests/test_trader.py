"""Tests for trader.py — order construction (both the V2 and the retained
legacy endpoint), V2 price/tick math, rollback verification, and exception
disambiguation. All Kalshi API interaction is mocked per project policy (tests
must run offline)."""
import json
import math
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import config, trader
from kalshi_betting.config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    BUY_SLIPPAGE_TICKS,
    DEFAULT_EXCHANGE_INDEX,
    V2_ORDER_PATH,
)
from kalshi_betting.scanner import PriceRange
from kalshi_betting.trader import (
    _build_no_order,
    _build_no_order_v2,
    _build_rollback_order_v2,
    _build_yes_order,
    _build_yes_order_v2,
    _buy_max_cost_cents,
    _ceil_to_tick,
    _execute_one,
    _format_count,
    _format_price,
    _v2_fill_status,
    _v2_limit_price,
    _v2_rollback_price,
)

# Tick grids used by the V2 price-math tests, mirroring the regimes named by
# live `price_level_structure` values (see scanner.tick_size_for_price).
DECI_CENT_BANDS = [PriceRange(start=0.0, end=1.0, step=0.001)]
CENTER_DECI_EDGE_CENTI_BANDS = [
    PriceRange(start=0.0, end=0.01, step=0.0001),
    PriceRange(start=0.01, end=0.99, step=0.001),
    PriceRange(start=0.99, end=1.0, step=0.0001),
]


def make_market(structure: str = "", ranges: list | None = None) -> SimpleNamespace:
    """Market stand-in exposing only the two fields tick_size_for_price reads."""
    return SimpleNamespace(price_level_structure=structure, price_ranges=ranges)


def make_spec(
    x: int = 5,
    nA: float = 0.40,
    pB: float = 0.35,
    structure: str = "",
    ranges: list | None = None,
) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads.

    Both markets carry explicit tick-structure attributes because the V2 order
    builders price against the market's own grid via scanner.tick_size_for_price;
    the defaults ("" / None) mean "unknown", i.e. the $0.01 fallback grid.
    """
    pair = MagicMock()
    pair.market_a.ticker = "TICK-A"
    pair.market_a.title = "Market A"
    pair.market_a.price_level_structure = structure
    pair.market_a.price_ranges = ranges
    pair.market_b.ticker = "TICK-B"
    pair.market_b.title = "Market B"
    pair.market_b.price_level_structure = structure
    pair.market_b.price_ranges = ranges
    pair.nA = nA
    pair.pB = pB
    pair.canonical_title = "test pair"
    spec = MagicMock()
    spec.pair = pair
    spec.x = x
    spec.y = x
    return spec


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
    @pytest.fixture(autouse=True)
    def _use_legacy(self, legacy_mode):
        """Exercises the legacy submission path's exception handling."""

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

    def test_rollback_is_reduce_only_bid_at_aggressive_price(self):
        # Closing a held NO position is buying the YES short back — a bid —
        # and reduce_only keeps it from ever opening new exposure.
        body = _build_rollback_order_v2(make_spec())
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "bid"
        assert body["reduce_only"] is True
        # Default-grid market: the finest-grid target floors to the $0.01
        # grid's highest tradeable level
        assert body["price"] == "0.9900"

    def test_rollback_price_floors_to_market_grid(self):
        # Regression (found in adversarial review): a flat $0.99 bid cannot
        # cross asks resting in (0.99, 1) on sub-cent regimes, structurally
        # killing an unwind the legacy market order always filled. The bid
        # must be the highest tradeable level of THIS market's grid.
        assert _v2_rollback_price(make_market("linear_cent")) == Decimal("0.99")
        assert _v2_rollback_price(
            make_market("deci_cent", DECI_CENT_BANDS)
        ) == Decimal("0.999")
        assert _v2_rollback_price(
            make_market("center_deci_edge_centi_cent", CENTER_DECI_EDGE_CENTI_BANDS)
        ) == Decimal("0.9999")

    def test_all_legs_fill_or_kill(self):
        spec = make_spec()
        for body in (
            _build_no_order_v2(spec), _build_yes_order_v2(spec), _build_rollback_order_v2(spec),
        ):
            assert body["time_in_force"] == "fill_or_kill"
            assert body["post_only"] is False

    def test_all_legs_carry_explicit_routable_exchange_index(self):
        # Never -1 (auto-route): the order's shard must agree with the shard
        # the ingest-time guard admitted the market on.
        spec = make_spec()
        for body in (
            _build_no_order_v2(spec), _build_yes_order_v2(spec), _build_rollback_order_v2(spec),
        ):
            assert body["exchange_index"] == DEFAULT_EXCHANGE_INDEX
            assert body["exchange_index"] != -1

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
    def _use_v2(self, v2_mode):
        pass

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
        # Default-grid spec market: finest-grid target floored to $0.99
        assert rollback_body["price"] == "0.9900"

    def test_v2_unfilled_rollback_is_rollback_failed(self, post):
        post.side_effect = [v2_resp(5), v2_resp(0), v2_resp(0)]
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error

    def test_v2_leg_a_exception_with_position_is_unwound(self, post):
        post.side_effect = [TimeoutError("timeout"), v2_resp(5)]
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-A", position=-5)
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
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-B", position=5)
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


class TestOrderVersionDispatch:
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

    def test_v2_mode_never_touches_legacy_endpoint(self, v2_mode, monkeypatch):
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
