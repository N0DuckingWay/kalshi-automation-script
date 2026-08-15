"""Tests for trader.py — order construction, rollback verification, exception
disambiguation, and per-shard collateral funding. All Kalshi API interaction is
mocked per project policy (tests must run offline)."""
import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from kalshi_betting import trader
from kalshi_betting.config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    DEFAULT_EXCHANGE_INDEX,
    TRANSFER_PATH,
)
from kalshi_betting.trader import (
    _build_no_order,
    _build_yes_order,
    _cents_to_centicents,
    _execute_one,
    _legacy_routable,
    _plan_transfers,
    _required_cents_by_shard,
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
    pair.canonical_title = title
    spec = MagicMock()
    spec.pair = pair
    spec.x = x
    spec.y = x
    spec.cost_with_fees_a = cost_a
    spec.cost_with_fees_b = cost_b
    return spec


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
