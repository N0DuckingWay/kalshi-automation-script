"""Tests for v2_probe.py — the user-run live gate for the V2 order mapping.

The probe itself submits real orders; these tests never do. Every Kalshi
interaction is a MagicMock, and the two submission seams the probe uses
(trader._submit_order_v2, trader._execute_transfer) are patched, so nothing
here can reach an endpoint.

What matters most: the probe must build its bodies through the REAL trader
functions (otherwise it verifies a reimplementation rather than the code that
will run), it must never submit before confirmation, and a position going the
WRONG way after the ask — the exact failure the probe exists to catch — must
be a hard FAIL that does not go on to submit the unwind.
"""
import inspect
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import trader, v2_probe

TICKER = "PROBE-TICKER"


def market_resp(exchange_index: int = 0, price_ranges: list | None = None,
                status: int = 200) -> SimpleNamespace:
    """Raw GET /markets/{ticker} response — the probe parses the JSON itself
    through scanner._market_from_dict, so the mock mirrors the wire format."""
    payload = {
        "market": {
            "ticker": TICKER,
            "event_ticker": "PROBE-EVENT",
            "title": "Probe market",
            "yes_sub_title": "Yes",
            "status": "active",
            "close_time": "2026-12-31T00:00:00Z",
            "yes_ask_dollars": "0.60",
            "no_ask_dollars": "0.40",
            "yes_bid_dollars": "0.59",
            "price_level_structure": "linear_cent",
            "price_ranges": price_ranges,
            "exchange_index": exchange_index,
        }
    }
    return SimpleNamespace(status=status, data=json.dumps(payload).encode("utf-8"))


def orderbook_resp(yes_bid: str = "0.59", qty: str = "500") -> SimpleNamespace:
    """Raw orderbook response in the current orderbook_fp/dollar-string shape.

    A resting YES bid at 0.59 is what an ask can cross with, and it is what
    scanner._bids_to_ask_levels turns into a NO ask at 0.41.
    """
    payload = {"orderbook_fp": {"yes_dollars": [[yes_bid, qty]], "no_dollars": []}}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def positions_resp(position: float | None) -> SimpleNamespace:
    """Raw get_positions response; None means "no position record at all"."""
    mps = [] if position is None else [{"ticker": TICKER, "position_fp": str(position)}]
    payload = {"market_positions": mps, "cursor": None}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def probe_client(positions: list, exchange_index: int = 0,
                 price_ranges: list | None = None) -> MagicMock:
    """Mock client whose position endpoint walks `positions` in call order.

    `positions` is the sequence of signed counts the probe will observe: e.g.
    [0, -0.01, 0] for start-flat, NO opened, closed back to flat. A None entry
    yields an empty positions page, which _position_count reports as 0.
    """
    client = MagicMock()
    client.get_market_without_preload_content = MagicMock(
        return_value=market_resp(exchange_index=exchange_index, price_ranges=price_ranges)
    )
    client.get_market_orderbook_without_preload_content = MagicMock(
        return_value=orderbook_resp()
    )
    client.get_positions_without_preload_content = MagicMock(
        side_effect=[positions_resp(p) for p in positions]
    )
    return client


def v2_resp(fill_count: str, remaining_count: str) -> dict:
    """Parsed V2 order response — what a patched _submit_order_v2 returns.

    V2 carries no `status` field; fill state lives in the two count strings.
    """
    return {
        "order_id": "ord_probe",
        "client_order_id": "cid_probe",
        "fill_count": fill_count,
        "remaining_count": remaining_count,
        "average_fill_price": "0.4100",
        "average_fee_paid": "0.0002",
    }


FILLED = v2_resp("0.01", "0.00")
KILLED = v2_resp("0.00", "0.01")


@pytest.fixture
def submits(monkeypatch) -> list:
    """Capture every body the probe submits, returning fills by default.

    Patched at trader._submit_order_v2 — its definition site — which is exactly
    why the probe calls it as a module attribute.
    """
    captured: list = []

    def fake_submit(client, body):
        captured.append(body)
        return FILLED

    monkeypatch.setattr(trader, "_submit_order_v2", fake_submit)
    return captured


def answer(monkeypatch, value: str) -> None:
    """Point the confirmation prompt at a canned answer."""
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: value)


class TestBodyConstruction:
    """The probe must submit bodies built by the real trader functions — a
    reimplementation would verify itself rather than the live order path."""

    def test_no_mapping_submits_ask_then_reduce_only_bid(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, 0])

        outcome = v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        assert outcome == v2_probe._PASS
        assert len(submits) == 2
        buy, close = submits
        # The hypothesis under test: buying NO is an ASK on the YES book.
        assert buy["side"] == "ask"
        assert buy["reduce_only"] is False
        # Closing it is a bid that may only reduce.
        assert close["side"] == "bid"
        assert close["reduce_only"] is True

    def test_both_bodies_carry_the_fractional_probe_count(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, 0])

        v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        assert [b["count"] for b in submits] == ["0.01", "0.01"]

    def test_bodies_carry_the_markets_own_exchange_index(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, 0], exchange_index=1)

        v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        # Explicit shard from the fetched market — never the -1 auto-route
        # sentinel, same rule as the live path.
        assert [b["exchange_index"] for b in submits] == [1, 1]

    def test_buy_price_matches_trader_no_buy_mapping(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, 0])

        v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        # Book has a YES bid at 0.59 -> NO ask at 0.41; the probe must price
        # through _no_buy_as_v2, not by hand.
        market = SimpleNamespace(price_ranges=None)
        _side, expected = trader._no_buy_as_v2(0.41, market)
        assert submits[0]["price"] == expected

    def test_build_body_uses_the_real_builder_then_overrides_only_count(self):
        # If _build_v2_order_body ever starts accepting fractional counts, the
        # override in _probe_body should go away — this pins the reason it
        # exists (the builder's int-typed count) rather than the workaround.
        params = inspect.signature(trader._build_v2_order_body).parameters
        assert params["count"].annotation is int
        body = v2_probe._probe_body(TICKER, "ask", "0.58", 0)
        reference = trader._build_v2_order_body(TICKER, "ask", "0.58", 1, 0)
        assert body["count"] == "0.01"
        assert {k: v for k, v in body.items() if k not in ("count", "client_order_id")} == {
            k: v for k, v in reference.items() if k not in ("count", "client_order_id")
        }


class TestConfirmationGate:
    """Nothing may reach the wire before a human types yes."""

    def test_declining_aborts_before_any_submit(self, submits, monkeypatch):
        answer(monkeypatch, "no")
        client = probe_client([0])

        outcome = v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        assert submits == []
        assert outcome == v2_probe._NEUTRAL

    def test_declining_the_unwind_is_a_failure_not_a_neutral(self, submits, monkeypatch):
        # Two prompts: accept the buy, decline the unwind. A NO position is then
        # left open, which is never an inconclusive outcome.
        answers = iter(["yes", "no"])
        monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
        client = probe_client([0, -0.01])

        outcome = v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        assert len(submits) == 1
        assert outcome == v2_probe._FAIL

    def test_yes_flag_skips_the_prompt(self, submits, monkeypatch):
        def explode(*_a, **_k):
            raise AssertionError("input() must not be called when --yes was passed")

        monkeypatch.setattr("builtins.input", explode)
        client = probe_client([0, -0.01, 0])

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=True) == v2_probe._PASS


class TestNoMappingVerdict:
    """The verdict is read off the ACCOUNT POSITION, not off the response."""

    def test_negative_then_flat_passes(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, 0])

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=False) == v2_probe._PASS

    def test_positive_position_after_the_ask_is_a_hard_fail(self, submits, monkeypatch, capsys):
        # THE failure this probe exists to catch: the ask opened YES exposure,
        # so "ask == buy NO" is false and the whole V2 NO leg is wrong.
        answer(monkeypatch, "yes")
        client = probe_client([0, 0.01])

        outcome = v2_probe._step_no_mapping(client, TICKER, assume_yes=False)

        assert outcome == v2_probe._FAIL
        # The unwind must NOT be submitted: it rests on the same disproven
        # mapping and would add to the wrong exposure.
        assert len(submits) == 1
        assert "HYPOTHESIS DISPROVEN" in capsys.readouterr().out

    def test_no_fill_is_neutral_and_submits_no_unwind(self, monkeypatch):
        answer(monkeypatch, "yes")
        monkeypatch.setattr(trader, "_submit_order_v2", lambda *_a, **_k: KILLED)
        client = probe_client([0, 0])

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=False) == v2_probe._NEUTRAL

    def test_unwind_that_leaves_a_position_fails(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0, -0.01, -0.01])

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=False) == v2_probe._FAIL

    def test_non_flat_start_aborts_before_submitting(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([-5])

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=False) == v2_probe._FAIL
        assert submits == []

    def test_empty_book_is_neutral(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        client = probe_client([0])
        empty = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=SimpleNamespace(status=200, data=json.dumps(empty).encode("utf-8"))
        )

        assert v2_probe._step_no_mapping(client, TICKER, assume_yes=False) == v2_probe._NEUTRAL
        assert submits == []


class TestUnfillableAskStep:
    """FoK kill semantics, and the ask limit's direction (a floor on proceeds)."""

    def test_body_is_priced_one_tick_below_one(self, submits, monkeypatch):
        answer(monkeypatch, "yes")
        monkeypatch.setattr(trader, "_submit_order_v2", lambda c, b: submits.append(b) or KILLED)
        client = probe_client([0, 0])

        v2_probe._step_unfillable_ask(client, TICKER, assume_yes=False)

        assert len(submits) == 1
        assert submits[0]["side"] == "ask"
        # Whole-cent fallback grid (no price_ranges on the mock market).
        assert submits[0]["price"] == "0.99"
        assert submits[0]["count"] == "0.01"

    def test_kill_response_passes(self, monkeypatch):
        answer(monkeypatch, "yes")
        monkeypatch.setattr(trader, "_submit_order_v2", lambda *_a, **_k: KILLED)
        client = probe_client([0, 0])

        assert v2_probe._step_unfillable_ask(client, TICKER, assume_yes=False) == v2_probe._PASS

    def test_any_fill_fails(self, monkeypatch):
        answer(monkeypatch, "yes")
        monkeypatch.setattr(trader, "_submit_order_v2", lambda *_a, **_k: FILLED)
        client = probe_client([0, -0.01])

        assert v2_probe._step_unfillable_ask(client, TICKER, assume_yes=False) == v2_probe._FAIL

    def test_kill_but_position_moved_fails(self, monkeypatch):
        answer(monkeypatch, "yes")
        monkeypatch.setattr(trader, "_submit_order_v2", lambda *_a, **_k: KILLED)
        client = probe_client([0, -0.01])

        assert v2_probe._step_unfillable_ask(client, TICKER, assume_yes=False) == v2_probe._FAIL


class TestTransferStep:
    """Skipped whenever the exchange says the move isn't available; a pass means
    the cent actually round-tripped."""

    def _statuses(self, transfers_active: bool = True, shards=(0, 1)) -> dict:
        return {
            s: {
                "trading_active": True,
                "exchange_active": True,
                "intra_exchange_transfers_active": transfers_active,
                "description": "",
            }
            for s in shards
        }

    def test_skipped_when_no_breakdown(self, monkeypatch):
        monkeypatch.setattr(v2_probe.scanner, "fetch_shard_statuses", lambda _c: None)

        outcome = v2_probe._step_transfer(MagicMock(), None, assume_yes=True)

        assert outcome == v2_probe._NEUTRAL

    def test_skipped_when_transfers_inactive(self, monkeypatch):
        monkeypatch.setattr(
            v2_probe.scanner, "fetch_shard_statuses",
            lambda _c: self._statuses(transfers_active=False),
        )
        executed = MagicMock()
        monkeypatch.setattr(trader, "_execute_transfer", executed)

        outcome = v2_probe._step_transfer(MagicMock(), None, assume_yes=True)

        assert outcome == v2_probe._NEUTRAL
        executed.assert_not_called()

    def test_skipped_when_no_second_shard_advertised(self, monkeypatch):
        monkeypatch.setattr(
            v2_probe.scanner, "fetch_shard_statuses", lambda _c: self._statuses(shards=(0,))
        )

        assert v2_probe._step_transfer(MagicMock(), None, assume_yes=True) == v2_probe._NEUTRAL

    def test_round_trip_passes(self, monkeypatch):
        monkeypatch.setattr(v2_probe.scanner, "fetch_shard_statuses", lambda _c: self._statuses())
        # Opening balances, then the post-outbound and post-return reads the
        # real bounded poller performs through auth.verify_auth.
        balances = iter([{0: 500, 1: 100}, {0: 499, 1: 101}, {0: 500, 1: 100}])
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda _c: next(balances))
        monkeypatch.setattr(trader, "verify_auth", lambda _c: next(balances))
        moves: list = []
        monkeypatch.setattr(
            trader, "_execute_transfer",
            lambda _c, src, dst, cents: moves.append((src, dst, cents)) or f"tr_{src}{dst}",
        )

        outcome = v2_probe._step_transfer(MagicMock(), None, assume_yes=True)

        assert outcome == v2_probe._PASS
        assert moves == [(0, 1, 1), (1, 0, 1)]

    def test_outbound_that_never_settles_fails_without_sending_it_back(self, monkeypatch):
        monkeypatch.setattr(v2_probe.scanner, "fetch_shard_statuses", lambda _c: self._statuses())
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda _c: {0: 500, 1: 100})
        # The poller never sees the destination grow, so it burns its timeout.
        monkeypatch.setattr(trader, "verify_auth", lambda _c: {0: 500, 1: 100})
        monkeypatch.setattr(trader, "_await_transfer_settlement", lambda _c, _r: {0: 500, 1: 100})
        moves: list = []
        monkeypatch.setattr(
            trader, "_execute_transfer",
            lambda _c, src, dst, cents: moves.append((src, dst, cents)) or "tr_x",
        )

        outcome = v2_probe._step_transfer(MagicMock(), None, assume_yes=True)

        assert outcome == v2_probe._FAIL
        # Only the outbound leg was attempted — never send money back into a
        # transfer whose first leg is still in flight.
        assert moves == [(0, 1, 1)]


class TestCountSemantics:
    """The probe's fractional count must be read with the same Decimal rule
    trader._v2_filled uses for whole counts."""

    def test_probe_count_is_the_v2_fractional_minimum(self):
        assert v2_probe.PROBE_COUNT == Decimal("0.01")
        assert v2_probe.PROBE_COUNT_STR == "0.01"

    def test_count_readers_raise_on_missing_fields(self):
        # Same semantics as trader._v2_filled: unreadable counts mean the
        # outcome is UNKNOWN, never "did not fill".
        with pytest.raises(KeyError):
            v2_probe._filled_count({})
        with pytest.raises(KeyError):
            v2_probe._remaining_count({})


class TestMainDispatch:
    def test_order_steps_require_a_ticker(self, monkeypatch):
        built = MagicMock()
        monkeypatch.setattr(v2_probe.auth, "build_client", built)

        code = v2_probe.main(["--step", "no-mapping"])

        assert code == v2_probe._EXIT_CODES[v2_probe._NEUTRAL]
        # Refused before even authenticating — no client, no API call.
        built.assert_not_called()

    def test_exit_code_maps_the_outcome(self, monkeypatch):
        monkeypatch.setattr(v2_probe.auth, "build_client", lambda _m: MagicMock())
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda _c: {0: 1000})
        monkeypatch.setitem(v2_probe._STEPS, "no-mapping", lambda *_a: v2_probe._FAIL)

        assert v2_probe.main(["--ticker", TICKER]) == 1

    def test_uses_the_production_client(self, monkeypatch):
        built = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(v2_probe.auth, "build_client", built)
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda _c: {0: 1000})
        monkeypatch.setitem(v2_probe._STEPS, "no-mapping", lambda *_a: v2_probe._PASS)

        assert v2_probe.main(["--ticker", TICKER]) == 0
        # The sandbox implements neither V2 orders nor sharding, so a sandbox
        # result would prove nothing.
        built.assert_called_once_with("prod")


class TestPipelineIsolation:
    """The probe places real orders. It must stay unreachable from the pipeline
    — one import away from the scheduler is one import too many."""

    @pytest.mark.parametrize("module", ["main", "scheduler", "trader", "scanner", "strategy"])
    def test_no_pipeline_module_imports_v2_probe(self, module):
        import ast
        import importlib

        source = inspect.getsource(importlib.import_module(f"kalshi_betting.{module}"))
        # AST rather than a substring search: trader.py legitimately NAMES the
        # probe in prose (its docstring says the flag stays False until the
        # probe passes). What must never appear is an actual import edge, in
        # any form — plain, from-, or a lazy importlib call.
        imported: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.Call):
                # A lazy import passes the module path as a string literal,
                # which the two branches above cannot see.
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in ("import_module", "__import__"):
                    imported += [
                        a.value for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ]
        assert not [name for name in imported if "v2_probe" in name]

    def test_importing_main_does_not_import_the_probe(self):
        import importlib
        import sys

        for name in ("kalshi_betting.v2_probe", "kalshi_betting.main"):
            sys.modules.pop(name, None)
        importlib.import_module("kalshi_betting.main")
        assert "kalshi_betting.v2_probe" not in sys.modules
