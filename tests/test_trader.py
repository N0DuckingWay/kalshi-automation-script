"""Tests for trader.py — order construction, tick-aware V2 order bodies,
rollback verification, exception disambiguation, and per-shard collateral
funding. All Kalshi API interaction is mocked per project policy (tests must
run offline)."""
import json
import logging
import math
import uuid
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import trader
from kalshi_betting.config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    BUY_PRICE_SLIPPAGE_DOLLARS,
    DEFAULT_EXCHANGE_INDEX,
    TRANSFER_PATH,
    V2_CREATE_ORDER_PATH,
    V2_SELF_TRADE_PREVENTION_TYPE,
)
from kalshi_betting.scanner import PriceRange
from kalshi_betting.trader import (
    _build_no_order,
    _build_v2_order_body,
    _build_yes_order,
    _cents_to_centicents,
    _execute_one,
    _fp,
    _legacy_routable,
    _no_buy_as_v2,
    _no_close_as_v2,
    _plan_transfers,
    _quantize_to_tick,
    _required_cents_by_shard,
    _submit_leg,
    _v2_filled,
    _yes_buy_as_v2,
    ensure_shard_collateral,
)


def make_spec(
    x: int = 5,
    nA: float = 0.40,
    pB: float = 0.35,
    shards: tuple[int, int] = (0, 0),
    cost_a: float = 0.0,
    cost_b: float = 0.0,
    title: str = "test pair",
    price_ranges: list | None = None,
) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads.

    `shards` pins REAL ints on both legs' exchange_index. This must never be
    left to MagicMock's auto-attributes: an auto-attr is a truthy Mock object
    that compares unequal to DEFAULT_EXCHANGE_INDEX, which would make every
    spec look non-routable (or, with a different comparison, silently defeat
    the shard check entirely).

    `cost_a`/`cost_b` are the per-leg fee-inclusive dollar costs the collateral
    planner sizes transfers from, pinned as REAL floats for the same reason —
    an auto-attr Mock would blow up (or silently mis-size) the ceil-to-cents
    conversion in _required_cents_by_shard.

    `price_ranges` is pinned for the same reason again: trader._tick_bands
    tests it with isinstance(..., list), so an auto-attr Mock would silently
    take the whole-cent FALLBACK grid — a test asserting a sub-cent tick would
    then pass or fail for the wrong reason. None is the real "tick structure
    unknown" value scanner._parse_price_ranges produces.
    """
    pair = MagicMock()
    pair.market_a.ticker = "TICK-A"
    pair.market_a.title = "Market A"
    pair.market_a.exchange_index = shards[0]
    pair.market_a.price_ranges = price_ranges
    pair.market_b.ticker = "TICK-B"
    pair.market_b.title = "Market B"
    pair.market_b.exchange_index = shards[1]
    pair.market_b.price_ranges = price_ranges
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


def make_market(price_ranges: list | None = None, exchange_index: int = 0,
                ticker: str = "TICK-A") -> SimpleNamespace:
    """A minimal ApiMarket stand-in carrying only what the V2 path reads.

    SimpleNamespace rather than MagicMock so a typo'd attribute raises instead
    of quietly producing a Mock that isinstance() checks then route to the
    fallback tick grid.
    """
    return SimpleNamespace(
        ticker=ticker, price_ranges=price_ranges, exchange_index=exchange_index,
    )


# A tapered grid of the shape Kalshi ships for MVE/combo markets: fine ticks
# near the certainty ends, coarser through the middle. Dollar STRINGS, because
# that is what the live payload sends and what scanner._parse_price_ranges
# floats out of.
TAPERED_BANDS = [
    {"start": "0.0001", "end": "0.05", "step": "0.0001"},
    {"start": "0.05", "end": "0.95", "step": "0.001"},
    {"start": "0.95", "end": "0.9999", "step": "0.0001"},
]

# The uniform $0.0001 grid all MVE/combo markets move to on 2026-08-17 — the
# grid on which the legacy integer-cent buy_max_cost permits ~100 ticks of drift.
DECI_CENT_BANDS = [{"start": "0.0001", "end": "0.9999", "step": "0.0001"}]


def v2_order_resp(fill_count: str = "5.00", remaining_count: str = "0.00",
                  status: int = 201) -> SimpleNamespace:
    """Raw POST /portfolio/events/orders response.

    V2 carries NO `status` field — fill success is read off
    remaining_count/fill_count, so these mocks deliberately omit one.
    """
    payload = {
        "order_id": "ord_abc",
        "client_order_id": "cid_abc",
        "fill_count": fill_count,
        "remaining_count": remaining_count,
        "average_fill_price": "0.4000",
        "average_fee_paid": "0.0100",
        "ts_ms": 1_700_000_000_000,
    }
    return SimpleNamespace(status=status, data=json.dumps(payload).encode("utf-8"))


def v2_filled_resp(count: int = 5) -> SimpleNamespace:
    """A V2 response for a fill-or-kill order that filled completely."""
    return v2_order_resp(fill_count=f"{count}.00", remaining_count="0.00")


def v2_killed_resp(count: int = 5) -> SimpleNamespace:
    """A V2 response for a fill-or-kill order that was killed unfilled."""
    return v2_order_resp(fill_count="0.00", remaining_count=f"{count}.00")


def transfer_resp(transfer_id: str = "tr_abc123") -> SimpleNamespace:
    """Raw POST /portfolio/intra_exchange_instance_transfer response.

    The SDK models no such route, so trader signs a raw request and parses the
    body itself through fetch_json_page — mocks mirror that wire format.
    """
    return SimpleNamespace(
        status=200, data=json.dumps({"transfer_id": transfer_id}).encode("utf-8")
    )


def shard_status(transfers_active: bool = True) -> dict:
    """One parsed scanner.fetch_shard_statuses() entry."""
    return {
        "trading_active": True,
        "exchange_active": True,
        "intra_exchange_transfers_active": transfers_active,
        "description": "",
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
        spec = make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)
        assert _required_cents_by_shard([spec]) == {0: 1500}

    def test_cross_shard_spec_splits_requirement_across_both_shards(self):
        # Each leg draws collateral from its OWN market's shard.
        spec = make_spec(shards=(0, 1), cost_a=10.00, cost_b=5.00)
        assert _required_cents_by_shard([spec]) == {0: 1000, 1: 500}

    def test_partial_cents_round_up_never_down(self):
        # Flooring would under-fund the shard and get the order rejected for
        # insufficient collateral; the ceiling costs at most a spare cent.
        spec = make_spec(shards=(0, 0), cost_a=1.001, cost_b=0.0)
        assert _required_cents_by_shard([spec]) == {0: 101}

    def test_float_noise_does_not_inflate_by_a_cent(self):
        # 0.07 * 100 == 7.000000000000001 in binary floating point; the round()
        # before the ceiling must keep this at 7 cents, not 8.
        spec = make_spec(shards=(0, 0), cost_a=0.07, cost_b=0.0)
        assert _required_cents_by_shard([spec]) == {0: 7}

    def test_requirements_accumulate_across_specs(self):
        specs = [
            make_spec(shards=(0, 0), cost_a=1.00, cost_b=2.00),
            make_spec(shards=(0, 1), cost_a=3.00, cost_b=4.00),
        ]
        assert _required_cents_by_shard(specs) == {0: 600, 1: 400}


class TestEnsureShardCollateral:
    """Collateral must be on the shard an order settles against before that
    order is submitted. Every failure mode degrades to dropping the affected
    trades — never to submitting them underfunded, and never to a retry."""

    def _patch_io(self, monkeypatch, *, transfer=None, balances=None, settle_timeout=0.05):
        """Patch trader's two outbound calls and return the mocks.

        `transfer` is the signed_raw_request stand-in (return_value or
        side_effect already configured); `balances` is verify_auth's. The
        settle poll is compressed to milliseconds so the async-settlement
        contract is exercised for real (a real deadline, a real sleep) without
        the suite paying the production 30s bound.
        """
        srr = MagicMock(return_value=transfer_resp()) if transfer is None else transfer
        va = MagicMock(return_value={}) if balances is None else balances
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        monkeypatch.setattr(trader, "verify_auth", va)
        monkeypatch.setattr(trader, "TRANSFER_POLL_INTERVAL_SECONDS", 0.001)
        monkeypatch.setattr(trader, "TRANSFER_SETTLE_TIMEOUT_SECONDS", settle_timeout)
        return srr, va

    def test_zero_deficit_is_a_no_op(self, monkeypatch):
        # The universal case today: everything is on shard 0 and shard 0 is
        # funded. No transfer, and no balance re-poll either.
        srr, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(MagicMock(), portfolio, {0: 100_000}, None)
        assert result == portfolio
        srr.assert_not_called()
        va.assert_not_called()

    def test_empty_portfolio_short_circuits(self, monkeypatch):
        srr, va = self._patch_io(monkeypatch)
        assert ensure_shard_collateral(MagicMock(), [], {0: 100_000}, None) == []
        srr.assert_not_called()
        va.assert_not_called()

    def test_dry_run_plans_but_never_posts(self, monkeypatch, caplog):
        srr, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), portfolio, {0: 100, 1: 100_000}, None, dry_run=True
            )
        assert result == portfolio
        srr.assert_not_called()
        va.assert_not_called()
        assert any("DRY RUN" in r.getMessage() for r in caplog.records)

    def test_funded_deficit_posts_exact_body_and_returns_full_portfolio(self, monkeypatch):
        # Shard 0 needs 1500c but holds 100c; shard 1 has the rest.
        srr, va = self._patch_io(
            monkeypatch,
            # Insufficient on the first re-read, sufficient on the second —
            # acceptance is not settlement, so the poll must keep looking.
            balances=MagicMock(side_effect=[{0: 100, 1: 100_000}, {0: 1500, 1: 98_600}]),
        )
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, None
        )
        assert result == portfolio
        assert srr.call_count == 1
        args, kwargs = srr.call_args
        # (client, method, path) are bound positionally by the partial
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
        srr, va = self._patch_io(
            monkeypatch,
            transfer=MagicMock(return_value=transfer_resp("tr_stuck")),
            balances=MagicMock(return_value={0: 100, 1: 100_000}),
            # Deadline already elapsed: one re-read, then give up.
            settle_timeout=0,
        )
        needy = make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00, title="needy pair")
        funded = make_spec(shards=(1, 1), cost_a=1.00, cost_b=1.00, title="funded pair")
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
        srr, va = self._patch_io(monkeypatch)
        statuses = {0: shard_status(False), 1: shard_status(True)}
        needy = make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00, title="needy pair")
        funded = make_spec(shards=(1, 1), cost_a=1.00, cost_b=1.00, title="funded pair")
        with caplog.at_level(logging.INFO, logger="root"):
            result = ensure_shard_collateral(
                MagicMock(), [needy, funded], {0: 100, 1: 100_000}, statuses
            )
        assert result == [funded]
        srr.assert_not_called()
        # Nothing was sent, so there is nothing to wait for either.
        va.assert_not_called()
        warnings = " ".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "manually" in warnings

    def test_inactive_source_shard_also_blocks_the_post(self, monkeypatch):
        # Same gate from the other end: the SOURCE shard can't send.
        srr, va = self._patch_io(monkeypatch)
        statuses = {0: shard_status(True), 1: shard_status(False)}
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, statuses
        )
        assert result == []
        srr.assert_not_called()

    def test_none_statuses_still_attempts_the_transfer(self, monkeypatch):
        # No per-shard breakdown (sandbox / pre-sharding shape) means there is
        # nothing to gate on — attempt it and let the POST fail loudly if the
        # endpoint is unsupported.
        srr, va = self._patch_io(
            monkeypatch, balances=MagicMock(return_value={0: 1500, 1: 98_600})
        )
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        result = ensure_shard_collateral(
            MagicMock(), portfolio, {0: 100, 1: 100_000}, None
        )
        assert result == portfolio
        assert srr.call_count == 1

    def test_failed_post_drops_affected_specs_and_keeps_the_rest(self, monkeypatch, caplog):
        srr, va = self._patch_io(
            monkeypatch, transfer=MagicMock(side_effect=RuntimeError("boom"))
        )
        needy = make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00, title="needy pair")
        funded = make_spec(shards=(1, 1), cost_a=1.00, cost_b=1.00, title="funded pair")
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
        srr, _ = self._patch_io(
            monkeypatch, transfer=MagicMock(side_effect=TimeoutError("timeout"))
        )
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        ensure_shard_collateral(MagicMock(), portfolio, {0: 100, 1: 100_000}, None)
        assert srr.call_count == 1

    def test_transfer_path_bypasses_the_retry_wrapper_entirely(self):
        # Structural guarantee, not just a call count: trader.py must not even
        # import api_call_with_retry, so no future edit can accidentally wrap
        # the non-idempotent transfer POST in backoff.
        assert not hasattr(trader, "api_call_with_retry")

    def test_cross_shard_spec_funds_both_legs_shards(self, monkeypatch):
        # Legs on different shards: BOTH must be covered or the pair is a
        # half-fill risk, so a shortfall on either one drops the whole spec.
        srr, va = self._patch_io(
            monkeypatch, balances=MagicMock(return_value={0: 100, 1: 100_000})
        )
        spec = make_spec(shards=(0, 1), cost_a=10.00, cost_b=5.00)
        result = ensure_shard_collateral(
            MagicMock(), [spec], {0: 100, 1: 100_000}, None
        )
        # Shard 0 needs 1000c and only ever holds 100c → the pair is dropped
        # even though shard 1's leg is amply funded.
        assert result == []

    def test_no_surplus_anywhere_drops_without_posting(self, monkeypatch):
        # An empty plan here means "nothing movable", NOT "nothing needed" —
        # the specs must still be dropped rather than sailing through.
        srr, va = self._patch_io(monkeypatch)
        portfolio = [make_spec(shards=(0, 0), cost_a=10.00, cost_b=5.00)]
        assert ensure_shard_collateral(MagicMock(), portfolio, {0: 100}, None) == []
        srr.assert_not_called()
        va.assert_not_called()


class TestQuantizeToTick:
    """V2 limit prices must land EXACTLY on a tick of the market's own grid or
    the exchange rejects the order. This is the first reader of the
    price_level_structure / price_ranges fields scanner.py ingests."""

    def test_linear_cent_floors_toward_zero(self):
        # No price_ranges → whole-cent fallback grid; a bid floors, which can
        # only TIGHTEN the cap.
        assert _quantize_to_tick(
            Decimal("0.4567"), make_market(), ROUND_FLOOR
        ) == Decimal("0.45")

    def test_linear_cent_ceils_toward_one(self):
        assert _quantize_to_tick(
            Decimal("0.4567"), make_market(), ROUND_CEILING
        ) == Decimal("0.46")

    def test_price_already_on_grid_is_unchanged(self):
        assert _quantize_to_tick(
            Decimal("0.45"), make_market(), ROUND_FLOOR
        ) == Decimal("0.45")
        assert _quantize_to_tick(
            Decimal("0.45"), make_market(), ROUND_CEILING
        ) == Decimal("0.45")

    def test_tapered_low_band_uses_its_own_fine_step(self):
        m = make_market(TAPERED_BANDS)
        # 0.02345 sits in the $0.0001 band, NOT the coarser middle one.
        assert _quantize_to_tick(Decimal("0.02345"), m, ROUND_FLOOR) == Decimal("0.0234")

    def test_tapered_middle_band_uses_the_coarse_step(self):
        m = make_market(TAPERED_BANDS)
        assert _quantize_to_tick(Decimal("0.5234"), m, ROUND_FLOOR) == Decimal("0.523")
        assert _quantize_to_tick(Decimal("0.5234"), m, ROUND_CEILING) == Decimal("0.524")

    def test_tapered_high_band_uses_its_own_fine_step(self):
        m = make_market(TAPERED_BANDS)
        assert _quantize_to_tick(Decimal("0.96234"), m, ROUND_FLOOR) == Decimal("0.9623")

    def test_band_boundary_resolves_to_the_lower_band(self):
        # 0.05 is the shared edge of bands 1 and 2. Selection is deterministic
        # (first band whose end reaches the price) and the boundary price is a
        # tick of both bands, so neither rounding direction moves it.
        m = make_market(TAPERED_BANDS)
        assert _quantize_to_tick(Decimal("0.05"), m, ROUND_FLOOR) == Decimal("0.05")
        assert _quantize_to_tick(Decimal("0.05"), m, ROUND_CEILING) == Decimal("0.05")

    def test_price_range_dataclasses_are_read_like_dicts(self):
        # The live path hands over scanner.PriceRange objects; the backtest
        # cache hands over raw dicts. Both must produce the same grid.
        as_objects = [PriceRange(start=float(b["start"]), end=float(b["end"]),
                                 step=float(b["step"])) for b in TAPERED_BANDS]
        assert _quantize_to_tick(
            Decimal("0.5234"), make_market(as_objects), ROUND_FLOOR
        ) == _quantize_to_tick(
            Decimal("0.5234"), make_market(TAPERED_BANDS), ROUND_FLOOR
        )

    def test_none_price_ranges_falls_back_to_whole_cents(self):
        assert _quantize_to_tick(
            Decimal("0.456"), make_market(None), ROUND_FLOOR
        ) == Decimal("0.45")

    def test_empty_price_ranges_falls_back_to_whole_cents(self):
        assert _quantize_to_tick(
            Decimal("0.456"), make_market([]), ROUND_FLOOR
        ) == Decimal("0.45")

    def test_malformed_price_ranges_fall_back_to_whole_cents(self):
        # Fail-SAFE, not fail-open: a coarser grid than the market's real one
        # still lands on a price the exchange accepts.
        junk = [{"start": "x", "end": "y", "step": "z"}, {"nope": 1}, None]
        assert _quantize_to_tick(
            Decimal("0.456"), make_market(junk), ROUND_FLOOR
        ) == Decimal("0.45")

    def test_non_positive_step_band_is_skipped(self):
        bands = [{"start": "0.01", "end": "0.99", "step": "0"}]
        assert _quantize_to_tick(
            Decimal("0.456"), make_market(bands), ROUND_FLOOR
        ) == Decimal("0.45")

    def test_never_returns_zero(self):
        for market in (make_market(), make_market(DECI_CENT_BANDS),
                       make_market(TAPERED_BANDS)):
            for rounding in (ROUND_FLOOR, ROUND_CEILING):
                assert _quantize_to_tick(Decimal("0"), market, rounding) > 0
                assert _quantize_to_tick(Decimal("-3"), market, rounding) > 0

    def test_never_returns_one(self):
        for market in (make_market(), make_market(DECI_CENT_BANDS),
                       make_market(TAPERED_BANDS)):
            for rounding in (ROUND_FLOOR, ROUND_CEILING):
                assert _quantize_to_tick(Decimal("1"), market, rounding) < 1
                assert _quantize_to_tick(Decimal("4"), market, rounding) < 1

    def test_out_of_range_high_clamps_to_top_of_grid(self):
        assert _quantize_to_tick(
            Decimal("1.5"), make_market(), ROUND_CEILING
        ) == Decimal("0.99")

    def test_out_of_range_low_clamps_to_bottom_of_grid(self):
        assert _quantize_to_tick(
            Decimal("-0.5"), make_market(), ROUND_FLOOR
        ) == Decimal("0.01")

    def test_deci_cent_grid_produces_four_decimal_prices(self):
        # The $0.0001 grid all MVE/combo markets move to on 2026-08-17 — the
        # exact case the legacy integer-cent cap could not express.
        m = make_market(DECI_CENT_BANDS)
        # Ticks run from the band's own start (0.0001), so 0.56047 floors onto
        # 0.0001 + 5603 * 0.0001 — a price the whole-cent cap could not name.
        assert _fp(_quantize_to_tick(Decimal("0.56047"), m, ROUND_FLOOR)) == "0.5604"
        assert _fp(_quantize_to_tick(Decimal("0.56047"), m, ROUND_CEILING)) == "0.5605"

    def test_quantized_price_is_a_whole_number_of_ticks_from_band_start(self):
        # A band is only guaranteed to be aligned to its OWN start, so ticks
        # are measured from there rather than from zero.
        bands = [{"start": "0.0003", "end": "0.9999", "step": "0.0005"}]
        got = _quantize_to_tick(Decimal("0.5"), make_market(bands), ROUND_FLOOR)
        assert (got - Decimal("0.0003")) % Decimal("0.0005") == 0


class TestFpFormat:
    """V2 replaced integer-cent fields with validated fixed-point strings:
    prices carry 2-4 decimals, counts 0-2, and an exponent form is a rejection."""

    def test_two_decimal_price(self):
        assert _fp(Decimal("0.56")) == "0.56"

    def test_four_decimal_price(self):
        assert _fp(Decimal("0.5605")) == "0.5605"

    def test_count_is_two_decimals(self):
        assert _fp(Decimal(5), 2) == "5.00"

    def test_at_least_two_decimals(self):
        assert _fp(Decimal("0.5")) == "0.50"
        assert _fp(Decimal(1), 2) == "1.00"

    def test_trailing_zeros_beyond_two_are_dropped(self):
        assert _fp(Decimal("0.5600")) == "0.56"
        assert _fp(Decimal("0.5650")) == "0.565"

    def test_never_scientific_notation(self):
        # Decimal.normalize() turns 100 into 1E+2 — the re-quantize must undo it.
        for value in (Decimal("1E+2"), Decimal("100"), Decimal("1000"),
                      Decimal("0.0001")):
            assert "E" not in _fp(value) and "e" not in _fp(value)
        assert _fp(Decimal("1E+2"), 2) == "100.00"

    def test_excess_precision_is_rounded_to_the_field_maximum(self):
        assert _fp(Decimal("0.123456")) == "0.1235"


class TestV2LegMappings:
    """Leg kind -> (V2 side, tick-quantized limit price). The NO mappings
    encode the UNVERIFIED hypothesis that an ask opens a NO position."""

    def test_yes_buy_is_a_floored_bid(self):
        side, price = _yes_buy_as_v2(0.35, make_market())
        assert side == "bid"
        assert price == "0.36"

    def test_yes_buy_floor_only_tightens_the_cap(self):
        side, price = _yes_buy_as_v2(0.3547, make_market())
        assert side == "bid"
        # 0.3647 floored to the cent grid = 0.36, i.e. we permit LESS slippage
        # than the raw allowance, never more.
        assert price == "0.36"
        assert Decimal(price) <= Decimal("0.3547") + Decimal(str(BUY_PRICE_SLIPPAGE_DOLLARS))

    def test_no_buy_is_a_ceiled_ask_at_one_minus_price(self):
        side, price = _no_buy_as_v2(0.40, make_market())
        assert side == "ask"
        assert price == "0.59"

    def test_no_buy_ceiling_only_tightens_the_implied_cap(self):
        side, price = _no_buy_as_v2(0.4033, make_market())
        assert side == "ask"
        # 1 - 0.4133 = 0.5867, ceiled to the cent grid = 0.59: a HIGHER minimum
        # accepted proceeds, i.e. a stricter implied NO cost cap.
        assert price == "0.59"
        assert Decimal(price) >= Decimal(1) - (
            Decimal("0.4033") + Decimal(str(BUY_PRICE_SLIPPAGE_DOLLARS))
        )

    def test_no_close_is_a_bid_at_the_top_of_the_grid(self):
        # Reproduces the legacy rollback's "unwind at any price" urgency.
        assert _no_close_as_v2(make_market()) == ("bid", "0.99")

    def test_no_close_respects_a_sub_cent_grid(self):
        assert _no_close_as_v2(make_market(DECI_CENT_BANDS)) == ("bid", "0.9999")

    def test_mappings_use_the_market_tick_grid(self):
        m = make_market(DECI_CENT_BANDS)
        # cap = 0.36047; the bid floors onto the $0.0001 grid and the ask
        # ceils 1 - cap onto it, so each stays on the conservative side.
        assert _yes_buy_as_v2(0.35047, m) == ("bid", "0.3604")
        assert _no_buy_as_v2(0.35047, m) == ("ask", "0.6396")


class TestV2BodyConstruction:
    """Plain-dict bodies with an EXPLICIT per-order exchange_index — the SDK's
    CreateOrderRequest has no slot for it and would silently drop it."""

    def test_body_carries_every_required_field(self):
        body = _build_v2_order_body("TICK-A", "bid", "0.36", 5, 0)
        assert body["ticker"] == "TICK-A"
        assert body["side"] == "bid"
        assert body["price"] == "0.36"
        assert body["count"] == "5.00"
        assert body["time_in_force"] == "fill_or_kill"
        assert body["self_trade_prevention_type"] == V2_SELF_TRADE_PREVENTION_TYPE
        assert body["post_only"] is False
        assert body["cancel_order_on_pause"] is True
        assert body["reduce_only"] is False
        assert body["exchange_index"] == 0

    def test_body_is_a_plain_dict_not_a_pydantic_model(self):
        # signed_raw_request json.dumps() this verbatim; a model would drop
        # exchange_index on the way to the wire.
        assert type(_build_v2_order_body("TICK-A", "bid", "0.36", 5, 0)) is dict

    def test_exchange_index_passes_through_verbatim(self):
        # Deliberately NOT the -1 auto-route sentinel: a wrong shard tag must
        # be rejected loudly rather than silently papered over.
        assert _build_v2_order_body("TICK-A", "bid", "0.36", 5, 7)["exchange_index"] == 7

    def test_reduce_only_is_opt_in(self):
        body = _build_v2_order_body("TICK-A", "bid", "0.99", 5, 0, reduce_only=True)
        assert body["reduce_only"] is True

    def test_client_order_id_is_a_fresh_uuid_per_order(self):
        first = _build_v2_order_body("TICK-A", "bid", "0.36", 5, 0)["client_order_id"]
        second = _build_v2_order_body("TICK-A", "bid", "0.36", 5, 0)["client_order_id"]
        assert first != second
        # It is the exchange's dedupe handle, so it must be a real uuid.
        assert uuid.UUID(first).version == 4

    def test_count_is_a_fixed_point_string(self):
        assert _build_v2_order_body("TICK-A", "bid", "0.36", 12, 0)["count"] == "12.00"

    def test_cross_shard_legs_get_different_exchange_indices(self, monkeypatch):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(side_effect=[v2_filled_resp(), v2_filled_resp()])
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        # Submitted leg-by-leg rather than through _execute_one because the
        # TEMPORARY _legacy_routable guard still refuses cross-shard pairs; V2
        # is what removes that guard, and the bodies must already be correct.
        spec = make_spec(shards=(0, 1))
        _submit_leg(trader_client := MagicMock(), ticker=spec.pair.market_a.ticker,
                    leg="no_buy", count=spec.x, price=spec.pair.nA,
                    market=spec.pair.market_a)
        _submit_leg(trader_client, ticker=spec.pair.market_b.ticker,
                    leg="yes_buy", count=spec.y, price=spec.pair.pB,
                    market=spec.pair.market_b)
        bodies = [call.kwargs["body"] for call in srr.call_args_list]
        assert [b["exchange_index"] for b in bodies] == [0, 1]
        assert [b["ticker"] for b in bodies] == ["TICK-A", "TICK-B"]

    def test_v2_path_never_touches_create_order_request(self, monkeypatch):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        monkeypatch.setattr(
            trader, "signed_raw_request",
            MagicMock(side_effect=[v2_filled_resp(), v2_filled_resp()]),
        )
        client = MagicMock()
        assert _execute_one(client, make_spec()).status == "executed"
        client.create_order_without_preload_content.assert_not_called()


class TestV2FillParsing:
    """V2 responses have NO `status` field — a complete fill-or-kill fill is
    exactly remaining_count == 0 AND fill_count == count."""

    def test_complete_fill_is_true(self):
        assert _v2_filled({"remaining_count": "0.00", "fill_count": "5.00"}, 5) is True

    def test_killed_order_is_false(self):
        assert _v2_filled({"remaining_count": "5.00", "fill_count": "0.00"}, 5) is False

    def test_partial_fill_is_false(self):
        assert _v2_filled({"remaining_count": "2.00", "fill_count": "3.00"}, 5) is False

    def test_full_fill_count_with_leftover_is_false(self):
        # Both conditions are checked, so a response that reported only one of
        # them honestly cannot read as a fill.
        assert _v2_filled({"remaining_count": "1.00", "fill_count": "5.00"}, 5) is False

    def test_missing_remaining_count_raises(self):
        # Raising routes the caller into its ambiguous-exception path (position
        # lookup, then rollback or manual_review) — the same handling a legacy
        # response missing order.status already got.
        with pytest.raises(KeyError):
            _v2_filled({"fill_count": "5.00"}, 5)

    def test_missing_fill_count_raises(self):
        with pytest.raises(KeyError):
            _v2_filled({"remaining_count": "0.00"}, 5)

    def test_unparseable_count_raises(self):
        with pytest.raises(ArithmeticError):
            _v2_filled({"remaining_count": "not-a-number", "fill_count": "5.00"}, 5)

    def test_null_count_raises(self):
        with pytest.raises(TypeError):
            _v2_filled({"remaining_count": None, "fill_count": "5.00"}, 5)


class TestV2SubmitPath:
    """The V2 request goes out as a signed raw POST — and exactly once."""

    def test_submits_a_signed_post_to_the_v2_path(self, monkeypatch):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(return_value=v2_filled_resp())
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        client = MagicMock()
        assert _submit_leg(client, ticker="TICK-A", leg="yes_buy", count=5,
                           price=0.35, market=make_market()) is True
        assert srr.call_count == 1
        args, kwargs = srr.call_args
        assert args[0] is client
        assert args[1] == "POST"
        assert args[2] == V2_CREATE_ORDER_PATH
        body = kwargs["body"]
        assert body["side"] == "bid"
        assert body["price"] == "0.36"
        assert body["count"] == "5.00"
        assert body["reduce_only"] is False

    def test_close_leg_forces_reduce_only(self, monkeypatch):
        # A closing order that could OPEN the opposite exposure is the one
        # thing the seam must never emit, whatever the caller passed.
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(return_value=v2_filled_resp())
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        _submit_leg(MagicMock(), ticker="TICK-A", leg="no_close", count=5,
                    price=0.40, market=make_market(), reduce_only=False)
        body = srr.call_args.kwargs["body"]
        assert body["reduce_only"] is True
        assert body["side"] == "bid"

    def test_killed_order_returns_false(self, monkeypatch):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        monkeypatch.setattr(trader, "signed_raw_request",
                            MagicMock(return_value=v2_killed_resp()))
        assert _submit_leg(MagicMock(), ticker="TICK-A", leg="no_buy", count=5,
                           price=0.40, market=make_market()) is False

    def test_non_2xx_raises_and_is_never_retried(self, monkeypatch):
        # A retryable 5xx: if this were wrapped in api_call_with_retry it would
        # be submitted up to 6 times — double-submitting a fill-or-kill leg.
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        # reason/getheaders are what the SDK's ApiException reads off a failed
        # response — a bare status/data namespace would blow up in the
        # exception constructor instead of in the code under test.
        srr = MagicMock(return_value=SimpleNamespace(
            status=503, data=b"{}", reason="Service Unavailable",
            getheaders=lambda: {},
        ))
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        # fetch_json_page re-raises non-2xx as ApiException, so the exception
        # type callers already handle is unchanged between the two paths.
        with pytest.raises(ApiException):
            _submit_leg(MagicMock(), ticker="TICK-A", leg="no_buy", count=5,
                        price=0.40, market=make_market())
        assert srr.call_count == 1

    def test_transport_error_is_never_retried(self, monkeypatch):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(side_effect=TimeoutError("timeout"))
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        with pytest.raises(TimeoutError):
            _submit_leg(MagicMock(), ticker="TICK-A", leg="no_buy", count=5,
                        price=0.40, market=make_market())
        assert srr.call_count == 1

    def test_unknown_leg_kind_is_rejected(self):
        with pytest.raises(ValueError):
            _submit_leg(MagicMock(), ticker="TICK-A", leg="yes_sell", count=5,
                        price=0.40, market=make_market())


class TestLegacyLegSubmissionIsUnchanged:
    """With the flag OFF (the default), the seam must put byte-identical
    CreateOrderRequests on the wire — the whole point of landing V2 disabled."""

    def test_no_buy_matches_the_original_builder(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(
            return_value=order_resp("executed")
        )
        spec = make_spec(x=5, nA=0.40)
        assert _submit_leg(client, ticker="TICK-A", leg="no_buy", count=5,
                           price=0.40, market=spec.pair.market_a) is True
        sent = client.create_order_without_preload_content.call_args.kwargs[
            "create_order_request"
        ]
        assert sent == _build_no_order(spec)

    def test_yes_buy_matches_the_original_builder(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(
            return_value=order_resp("executed")
        )
        spec = make_spec(x=5, pB=0.35)
        assert _submit_leg(client, ticker="TICK-B", leg="yes_buy", count=5,
                           price=0.35, market=spec.pair.market_b) is True
        sent = client.create_order_without_preload_content.call_args.kwargs[
            "create_order_request"
        ]
        assert sent == _build_yes_order(spec)

    def test_close_leg_is_a_reduce_only_market_sell(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(
            return_value=order_resp("executed")
        )
        _submit_leg(client, ticker="TICK-A", leg="no_close", count=5,
                    price=0.40, market=make_market())
        sent = client.create_order_without_preload_content.call_args.kwargs[
            "create_order_request"
        ]
        assert (sent.side, sent.action, sent.reduce_only) == ("no", "sell", True)
        assert sent.buy_max_cost is None

    def test_legacy_path_never_signs_a_raw_post(self, monkeypatch):
        srr = MagicMock()
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"), order_resp("executed"),
        ])
        assert _execute_one(client, make_spec()).status == "executed"
        srr.assert_not_called()

    def test_flag_defaults_to_off(self):
        # The live probe verifying the NO-leg mapping has not run; nothing but
        # that probe may flip this.
        assert trader.config.V2_ORDERS_ENABLED is False


class TestV2RollbackVerification:
    """Flag-flipped duplicate of TestRollbackVerification: the rollback state
    machine must behave IDENTICALLY on the V2 wire format."""

    def _patch(self, monkeypatch, responses):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(side_effect=responses)
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        return srr

    def test_unfilled_rollback_reports_rollback_failed(self, monkeypatch):
        self._patch(monkeypatch, [
            v2_filled_resp(),   # leg A
            v2_killed_resp(),   # leg B killed (confirmed non-fill)
            v2_killed_resp(),   # rollback killed → orphaned position
        ])
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "rollback_failed"
        assert "rollback FoK not filled" in result.error

    def test_filled_rollback_reports_rolled_back(self, monkeypatch):
        self._patch(monkeypatch, [
            v2_filled_resp(), v2_killed_resp(), v2_filled_resp(),
        ])
        assert _execute_one(MagicMock(), make_spec()).status == "rolled_back"

    def test_rollback_order_is_a_reduce_only_close(self, monkeypatch):
        srr = self._patch(monkeypatch, [
            v2_filled_resp(), v2_killed_resp(), v2_filled_resp(),
        ])
        _execute_one(MagicMock(), make_spec())
        rollback_body = srr.call_args_list[2].kwargs["body"]
        assert rollback_body["ticker"] == "TICK-A"
        assert rollback_body["reduce_only"] is True
        assert rollback_body["side"] == "bid"
        assert rollback_body["price"] == "0.99"
        assert rollback_body["time_in_force"] == "fill_or_kill"

    def test_both_legs_filled_is_executed(self, monkeypatch):
        srr = self._patch(monkeypatch, [v2_filled_resp(), v2_filled_resp()])
        assert _execute_one(MagicMock(), make_spec()).status == "executed"
        assert srr.call_count == 2

    def test_leg_a_killed_aborts_without_rollback(self, monkeypatch):
        srr = self._patch(monkeypatch, [v2_killed_resp()])
        result = _execute_one(MagicMock(), make_spec())
        assert result.status == "failed"
        assert "Leg A FoK not filled" in result.error
        # Nothing filled, so nothing to unwind.
        assert srr.call_count == 1

    def test_leg_bodies_carry_the_expected_sides(self, monkeypatch):
        srr = self._patch(monkeypatch, [v2_filled_resp(), v2_filled_resp()])
        _execute_one(MagicMock(), make_spec())
        bodies = [c.kwargs["body"] for c in srr.call_args_list]
        # Leg A buys NO → ask (the UNVERIFIED mapping); leg B buys YES → bid.
        assert bodies[0]["side"] == "ask" and bodies[0]["price"] == "0.59"
        assert bodies[1]["side"] == "bid" and bodies[1]["price"] == "0.36"


class TestV2ExceptionDisambiguation:
    """Flag-flipped duplicate of TestExceptionDisambiguation. Ambiguity is
    still resolved by the account position, and an unknown leg-B fill state
    still refuses to auto-roll-back."""

    def _patch(self, monkeypatch, responses):
        monkeypatch.setattr(trader.config, "V2_ORDERS_ENABLED", True)
        srr = MagicMock(side_effect=responses)
        monkeypatch.setattr(trader, "signed_raw_request", srr)
        return srr

    def test_leg_a_exception_with_no_position_is_failed(self, monkeypatch):
        srr = self._patch(monkeypatch, [TimeoutError("timeout")])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp()
        )
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert srr.call_count == 1

    def test_leg_a_exception_with_position_is_unwound(self, monkeypatch):
        self._patch(monkeypatch, [TimeoutError("timeout"), v2_filled_resp()])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-A", position=-5)
        )
        assert _execute_one(client, make_spec()).status == "rolled_back"

    def test_leg_b_exception_with_position_is_executed(self, monkeypatch):
        srr = self._patch(monkeypatch, [v2_filled_resp(), TimeoutError("timeout")])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp("TICK-B", position=5)
        )
        assert _execute_one(client, make_spec()).status == "executed"
        assert srr.call_count == 2

    def test_leg_b_exception_with_no_position_rolls_back(self, monkeypatch):
        self._patch(monkeypatch, [
            v2_filled_resp(), TimeoutError("timeout"), v2_filled_resp(),
        ])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp()
        )
        assert _execute_one(client, make_spec()).status == "rolled_back"

    def test_leg_b_exception_with_unknown_position_does_not_auto_rollback(
        self, monkeypatch
    ):
        srr = self._patch(monkeypatch, [v2_filled_resp(), TimeoutError("timeout")])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=RuntimeError("lookup failed")
        )
        result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        # No rollback order was submitted — only leg A and leg B's attempt.
        assert srr.call_count == 2

    def test_unreadable_fill_state_is_treated_as_ambiguous(self, monkeypatch):
        # A V2 body missing remaining_count is "we cannot tell whether this
        # filled" — _v2_filled raises, which must land in the SAME position-
        # consulting path a transport error does, not read as a clean non-fill.
        malformed = SimpleNamespace(
            status=201, data=json.dumps({"fill_count": "5.00"}).encode("utf-8")
        )
        srr = self._patch(monkeypatch, [malformed])
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=positions_resp()
        )
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert srr.call_count == 1
        client.get_positions_without_preload_content.assert_called_once()
