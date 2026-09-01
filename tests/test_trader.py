"""Tests for trader.py — order construction, rollback verification, and
exception disambiguation. All Kalshi API interaction is mocked per project
policy (tests must run offline).

Ambiguity handling is DELTA-based: _execute_one snapshots the position
immediately before each submission and again after an exception, and attributes
the outcome to the change. Mocks therefore sequence get_positions responses with
side_effect (see positions_seq) rather than returning one flat payload — a
single return_value would make before and after identical, i.e. delta 0.
"""
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kalshi_betting import _http
from kalshi_betting.config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT,
)
from kalshi_betting.trader import (
    _build_no_order,
    _build_yes_order,
    _buy_max_cost_cents,
    _execute_one,
    _position_count,
    _rollback_floor_cents,
)


class _StatusError(Exception):
    """Minimal stand-in for an SDK exception carrying an HTTP status.

    Mirrors tests/test_http.py's helper — api_call_with_retry classifies by
    the .status attribute, not by exception type.
    """

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def make_spec(x: int = 5, nA: float = 0.40, pB: float = 0.35) -> MagicMock:
    """Factory for a TradeSpec-like mock with the fields trader.py reads."""
    pair = MagicMock()
    pair.market_a.ticker = "TICK-A"
    pair.market_a.title = "Market A"
    pair.market_b.ticker = "TICK-B"
    pair.market_b.title = "Market B"
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


def positions_seq(*readings) -> MagicMock:
    """Mock get_positions that answers successive calls from a script.

    Each element is either a (ticker, position) tuple, None for "no position
    on file", or an Exception instance to raise for that call. _execute_one
    reads BOTH baselines up front — before either order is submitted, so no
    blocking call sits in the unhedged window between leg A's fill and leg B's
    submission — and then once more after an ambiguous leg, so the script is
    consumed in that order:
        before_a, before_b, [after_a], [after_b]

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
