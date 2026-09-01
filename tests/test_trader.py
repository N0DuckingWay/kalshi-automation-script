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
from kalshi_betting.config import BUY_MAX_COST_SLIPPAGE_CENTS
from kalshi_betting.trader import (
    _build_no_order,
    _build_yes_order,
    _execute_one,
    _position_count,
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
    reads the position once before each submission and once more after an
    ambiguous one, so the script is consumed in that order:
        before_a, [after_a], before_b, [after_b]

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

    def test_rollback_order_is_reduce_only_sell(self):
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=[
            order_resp("executed"),
            order_resp("canceled"),
            order_resp("executed"),
        ])
        client.get_positions_without_preload_content = positions_seq(None, None)
        _execute_one(client, make_spec())
        rollback_call = client.create_order_without_preload_content.call_args_list[2]
        rollback_req = rollback_call.kwargs["create_order_request"]
        assert rollback_req.action == "sell"
        assert rollback_req.side == "no"
        assert rollback_req.reduce_only is True

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
        client.get_positions_without_preload_content = positions_seq(None)
        result = _execute_one(client, make_spec())
        assert result.status == "failed"
        assert "Leg A FoK not filled" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        # Only the pre-submission baseline was read
        assert client.get_positions_without_preload_content.call_count == 1


class TestLegAExceptionDisambiguation:
    """Leg A raised: the outcome is attributed to the position DELTA."""

    def test_no_movement_is_failed(self):
        # Exception + position unchanged → confirmed non-fill, no rollback sent
        client = MagicMock()
        client.create_order_without_preload_content = MagicMock(side_effect=TimeoutError("timeout"))
        client.get_positions_without_preload_content = positions_seq(None, None)
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
            ("TICK-A", -10),   # before
            ("TICK-A", -10),   # after — unmoved
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
            ("TICK-A", -10),   # before
            ("TICK-A", -15),   # after — moved by -5 == -spec.x
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
            ("TICK-A", 0),
            ("TICK-A", -3),    # -3, but spec.x is 7
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
            None,
            RuntimeError("lookup failed"),   # non-retryable → must fail fast
        )
        with patch.object(_http.time, "sleep") as sleep:
            result = _execute_one(client, make_spec())
        assert result.status == "manual_review"
        assert "delta=None" in result.error
        assert client.create_order_without_preload_content.call_count == 1
        assert client.get_positions_without_preload_content.call_count == 2
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
