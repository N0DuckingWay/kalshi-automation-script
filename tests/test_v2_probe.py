"""
File: test_v2_probe.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for v2_probe.py — the user-run live verification CLI for the V2
    order mapping. The probe itself submits real orders; these tests never do.
    Every Kalshi interaction is a MagicMock, and the probe's submission seam
    (v2_probe.signed_request_json) plus trader._execute_transfer are patched,
    so nothing here can reach an endpoint.

    What matters most: the probe must build its bodies through the REAL trader
    builders (otherwise it verifies a reimplementation rather than the code
    that will run), it must never submit before confirmation, and a position
    going the WRONG way after the ask — the exact failure the probe exists to
    catch — must be a hard FAIL that does not go on to submit the unwind.

Dependencies:
    Imports v2_probe and trader; patches at each function's definition site.
    Offline-only per project policy.

Notes:
    The confirmation prompt is driven by patching builtins.input; --yes paths
    bypass it entirely.
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
    """Parsed V2 order response — what a patched signed_request_json returns.

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

    Patched at v2_probe.signed_request_json — the probe's one submission seam
    (it deliberately bypasses trader._submit_order_v2, whose int-count fill
    classifier cannot express the fractional probe count).
    """
    captured: list = []

    def fake_post(client, method, path, *, query=None, body=None):
        assert method == "POST"
        captured.append({"path": path, "body": body})
        return FILLED

    monkeypatch.setattr(v2_probe, "signed_request_json", fake_post)
    return captured


def answer(monkeypatch, value: str) -> None:
    """Point the confirmation prompt at a canned answer."""
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: value)


class TestBodyConstruction:
    """The probe must exercise the REAL trader builders, overriding only what
    its fractional count and unfillable price require."""

    def test_no_mapping_submits_ask_then_reduce_only_bid(self, submits, monkeypatch):
        client = probe_client([0, -0.01, 0])
        out = v2_probe._step_no_mapping(client, TICKER, True, 1)
        assert out == v2_probe._PASS
        assert [b["body"]["side"] for b in submits] == ["ask", "bid"]
        assert submits[0]["body"]["reduce_only"] is False
        assert submits[1]["body"]["reduce_only"] is True

    def test_both_bodies_carry_the_fractional_probe_count(self, submits, monkeypatch):
        client = probe_client([0, -0.01, 0])
        v2_probe._step_no_mapping(client, TICKER, True, 1)
        assert all(b["body"]["count"] == v2_probe.PROBE_COUNT_STR for b in submits)

    def test_bodies_carry_the_markets_own_exchange_index(self, submits, monkeypatch):
        client = probe_client([0, -0.01, 0], exchange_index=2)
        v2_probe._step_no_mapping(client, TICKER, True, 1)
        assert all(b["body"]["exchange_index"] == 2 for b in submits)

    def test_orders_post_to_the_v2_order_path(self, submits, monkeypatch):
        from kalshi_betting.config import V2_ORDER_PATH
        client = probe_client([0, -0.01, 0])
        v2_probe._step_no_mapping(client, TICKER, True, 1)
        assert all(b["path"] == V2_ORDER_PATH for b in submits)

    def test_no_buy_body_uses_the_real_builder_then_overrides_only_count(self):
        market = SimpleNamespace(
            ticker=TICKER, price_level_structure="", price_ranges=None, exchange_index=0,
        )
        body = v2_probe._no_buy_body(market, 0.41)
        reference = trader._build_no_order_v2(v2_probe._probe_spec(market, 0.41))
        # Everything except count (random client_order_id aside) is the
        # builder's own output — the probe verifies the real code.
        for key in ("ticker", "side", "price", "time_in_force", "exchange_index",
                    "reduce_only", "post_only"):
            assert body[key] == reference[key]
        assert body["count"] == v2_probe.PROBE_COUNT_STR


class TestConfirmationGate:
    """Nothing is submitted until the operator confirms (or passed --yes)."""

    def test_declining_aborts_before_any_submit(self, submits, monkeypatch):
        answer(monkeypatch, "no")
        client = probe_client([0])
        out = v2_probe._step_no_mapping(client, TICKER, False, 1)
        assert out == v2_probe._NEUTRAL
        assert submits == []

    def test_declining_the_unwind_is_a_failure_not_a_neutral(self, submits, monkeypatch):
        # First prompt yes, second prompt no — a NO position is then OPEN, so
        # walking away is a FAIL with a flatten-it-manually warning.
        answers = iter(["yes", "no"])
        monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
        client = probe_client([0, -0.01])
        out = v2_probe._step_no_mapping(client, TICKER, False, 1)
        assert out == v2_probe._FAIL
        assert len(submits) == 1  # only the opening ask went out

    def test_yes_flag_skips_the_prompt(self, submits, monkeypatch):
        monkeypatch.setattr(
            "builtins.input",
            lambda *_a, **_k: pytest.fail("prompt must not be shown with --yes"),
        )
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS


class TestNoMappingVerdict:
    """The position sign after the ask is the whole verdict."""

    def test_negative_then_flat_passes(self, submits, monkeypatch):
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS

    def test_positive_position_after_the_ask_is_a_hard_fail(self, submits, monkeypatch, capsys):
        # THE failure this probe exists to catch: the ask opened YES exposure.
        client = probe_client([0, 0.01])
        out = v2_probe._step_no_mapping(client, TICKER, True, 1)
        assert out == v2_probe._FAIL
        # The unwind must NOT be submitted — it rests on the same disproven mapping.
        assert len(submits) == 1
        printed = capsys.readouterr().out
        assert "HYPOTHESIS DISPROVEN" in printed
        assert "legacy" in printed

    def test_no_fill_is_neutral_and_submits_no_unwind(self, monkeypatch):
        submitted = []

        def killed_post(client, method, path, *, query=None, body=None):
            submitted.append(body)
            return KILLED

        monkeypatch.setattr(v2_probe, "signed_request_json", killed_post)
        client = probe_client([0, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._NEUTRAL
        assert len(submitted) == 1

    def test_unwind_that_leaves_a_position_fails(self, submits, monkeypatch):
        client = probe_client([0, -0.01, -0.01])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._FAIL

    def test_non_flat_start_aborts_before_submitting(self, submits, monkeypatch):
        client = probe_client([3.0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._FAIL
        assert submits == []

    def test_empty_book_is_neutral(self, submits, monkeypatch):
        client = probe_client([0])
        empty = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=SimpleNamespace(status=200, data=json.dumps(empty).encode())
        )
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._NEUTRAL
        assert submits == []

    def test_unreadable_fill_counts_fail_never_guess(self, monkeypatch):
        monkeypatch.setattr(
            v2_probe, "signed_request_json",
            lambda *a, **k: {"order_id": "x"},  # no fill_count at all
        )
        client = probe_client([0, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._FAIL


class TestUnfillableAskStep:
    """FoK kill semantics: a top-of-grid ask must come back killed."""

    def test_body_is_priced_at_the_top_of_the_grid(self, monkeypatch):
        captured = []

        def killed_post(client, method, path, *, query=None, body=None):
            captured.append(body)
            return KILLED

        monkeypatch.setattr(v2_probe, "signed_request_json", killed_post)
        client = probe_client([0, 0])
        out = v2_probe._step_unfillable_ask(client, TICKER, True, 1)
        assert out == v2_probe._PASS
        # linear_cent market: top of the $0.01 grid is 0.99
        assert captured[0]["price"] == "0.9900"
        assert captured[0]["side"] == "ask"

    def test_any_fill_fails(self, submits, monkeypatch):
        # `submits` returns FILLED — an unfillable ask that fills is a FAIL.
        client = probe_client([0, -0.01])
        assert v2_probe._step_unfillable_ask(client, TICKER, True, 1) == v2_probe._FAIL

    def test_kill_but_position_moved_fails(self, monkeypatch):
        monkeypatch.setattr(v2_probe, "signed_request_json", lambda *a, **k: KILLED)
        client = probe_client([0, -0.01])
        assert v2_probe._step_unfillable_ask(client, TICKER, True, 1) == v2_probe._FAIL


def shard_statuses(transfers_active: bool = True, shards: tuple = (0, 1)) -> dict:
    """Parsed fetch_shard_statuses shape for the transfer step."""
    return {
        idx: {
            "trading_active": True,
            "exchange_active": True,
            "intra_exchange_transfers_active": transfers_active,
            "description": f"shard {idx}",
        }
        for idx in shards
    }


class TestTransferStep:
    """One cent out and back through the real transfer functions."""

    def _arm(self, monkeypatch, statuses, balances_seq, transfer_ids=("t1", "t2")):
        monkeypatch.setattr(v2_probe.scanner, "fetch_shard_statuses", lambda c: statuses)
        balances = iter(balances_seq)
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda c: next(balances))
        executed = []

        def fake_transfer(client, source, dest, cents):
            executed.append((source, dest, cents))
            return transfer_ids[len(executed) - 1]

        monkeypatch.setattr(trader, "_execute_transfer", fake_transfer)
        # The settle poll re-reads via trader.verify_auth — feed it directly.
        monkeypatch.setattr(trader, "_await_transfer_settlement",
                            lambda c, req: next(balances))
        return executed

    def test_skipped_when_no_breakdown(self, monkeypatch):
        monkeypatch.setattr(v2_probe.scanner, "fetch_shard_statuses", lambda c: None)
        assert v2_probe._step_transfer(MagicMock(), None, True, 1) == v2_probe._NEUTRAL

    def test_skipped_when_transfers_inactive(self, monkeypatch):
        executed = self._arm(
            monkeypatch, shard_statuses(transfers_active=False), [{0: 100, 1: 0}],
        )
        assert v2_probe._step_transfer(MagicMock(), None, True, 1) == v2_probe._NEUTRAL
        assert executed == []

    def test_skipped_when_no_dest_shard_advertised(self, monkeypatch):
        executed = self._arm(monkeypatch, shard_statuses(shards=(0,)), [{0: 100}])
        assert v2_probe._step_transfer(MagicMock(), None, True, 1) == v2_probe._NEUTRAL
        assert executed == []

    def test_round_trip_passes(self, monkeypatch):
        executed = self._arm(
            monkeypatch, shard_statuses(),
            [{0: 100, 1: 0},        # before
             {0: 99, 1: 1},         # after outbound settles
             {0: 100, 1: 0}],       # after return settles
        )
        assert v2_probe._step_transfer(MagicMock(), None, True, 1) == v2_probe._PASS
        assert executed == [(0, 1, 1), (1, 0, 1)]

    def test_outbound_that_never_settles_fails_without_sending_it_back(self, monkeypatch):
        executed = self._arm(
            monkeypatch, shard_statuses(),
            [{0: 100, 1: 0},        # before
             {0: 99, 1: 0}],        # outbound never lands
        )
        assert v2_probe._step_transfer(MagicMock(), None, True, 1) == v2_probe._FAIL
        # The return leg must NOT be attempted while the cent is in flight.
        assert executed == [(0, 1, 1)]

    def test_dest_shard_argument_is_honored(self, monkeypatch):
        executed = self._arm(
            monkeypatch, shard_statuses(shards=(0, 3)),
            [{0: 100, 3: 0}, {0: 99, 3: 1}, {0: 100, 3: 0}],
        )
        assert v2_probe._step_transfer(MagicMock(), None, True, 3) == v2_probe._PASS
        assert executed == [(0, 3, 1), (3, 0, 1)]


class TestCountSemantics:
    def test_probe_count_is_the_v2_fractional_minimum(self):
        assert v2_probe.PROBE_COUNT_STR == "0.01"
        assert v2_probe.PROBE_COUNT == Decimal("0.01")

    def test_fill_counts_return_none_on_missing_fields(self):
        # None means "cannot tell", which every caller treats as a FAIL — the
        # same never-guess rule trader._v2_fill_status enforces by raising.
        assert v2_probe._fill_counts({"order_id": "x"}) == (None, None)

    def test_fill_counts_prefer_fp_variants_and_unwrap_order(self):
        wrapped = {"order": {"fill_count_fp": "0.01", "remaining_count_fp": "0.00"}}
        assert v2_probe._fill_counts(wrapped) == (Decimal("0.01"), Decimal("0"))


class TestMainDispatch:
    def test_order_steps_require_a_ticker(self, monkeypatch):
        monkeypatch.setattr(
            v2_probe.auth, "build_client",
            lambda mode: pytest.fail("must refuse before building a client"),
        )
        assert v2_probe.main(["--step", "no-mapping"]) == 2
        assert v2_probe.main(["--step", "unfillable-ask"]) == 2

    def test_unknown_step_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            v2_probe.main(["--step", "definitely-not-a-step"])

    def test_exit_code_maps_the_outcome(self, monkeypatch):
        monkeypatch.setattr(v2_probe.auth, "build_client", lambda mode: MagicMock())
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda c: {0: 100})
        for outcome, code in ((v2_probe._PASS, 0), (v2_probe._FAIL, 1),
                              (v2_probe._NEUTRAL, 2)):
            monkeypatch.setitem(
                v2_probe._STEPS, "no-mapping", lambda c, t, y, d, _o=outcome: _o
            )
            assert v2_probe.main(["--ticker", TICKER]) == code

    def test_uses_the_production_client(self, monkeypatch):
        modes = []

        def fake_build(mode):
            modes.append(mode)
            return MagicMock()

        monkeypatch.setattr(v2_probe.auth, "build_client", fake_build)
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda c: {0: 100})
        monkeypatch.setitem(v2_probe._STEPS, "no-mapping", lambda c, t, y, d: v2_probe._PASS)
        v2_probe.main(["--ticker", TICKER])
        # PROD only — a sandbox pass would prove nothing about the mapping.
        assert modes == ["prod"]

    def test_failed_auth_is_a_fail_exit(self, monkeypatch):
        monkeypatch.setattr(v2_probe.auth, "build_client", lambda mode: MagicMock())

        def boom(client):
            raise RuntimeError("bad credentials")

        monkeypatch.setattr(v2_probe.auth, "verify_auth", boom)
        assert v2_probe.main(["--ticker", TICKER]) == 1


_PIPELINE_MODULES = [
    "main", "trader", "scanner", "auth", "strategy", "reporter", "scheduler",
    "historical", "backtester", "backtest", "dashboard", "config", "_http",
]


class TestPipelineIsolation:
    """v2_probe submits real orders; the pipeline must never be able to reach
    it. A single import would put probe submissions one code path away from
    the weekly scheduler."""

    @pytest.mark.parametrize("module", _PIPELINE_MODULES)
    def test_no_pipeline_module_imports_v2_probe(self, module):
        import importlib
        mod = importlib.import_module(f"kalshi_betting.{module}")
        source = inspect.getsource(mod)
        assert "v2_probe" not in source, (
            f"kalshi_betting/{module}.py references v2_probe — the probe must "
            "never be reachable from the pipeline"
        )


class TestUnfillableAskCrossingGuard:
    def test_top_bid_meeting_the_limit_refuses_to_run(self, monkeypatch):
        # Regression (adversarial review): on a near-settled market the top
        # YES bid can sit AT the top of the grid — the "unfillable" ask would
        # actually fill (a real position) and the step would misreport broken
        # FoK semantics. The step must refuse (NEUTRAL) instead.
        submitted = []
        monkeypatch.setattr(
            v2_probe, "signed_request_json",
            lambda *a, **k: submitted.append(k) or KILLED,
        )
        client = probe_client([0])
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=orderbook_resp(yes_bid="0.99", qty="10")
        )
        out = v2_probe._step_unfillable_ask(client, TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        assert submitted == []

    def test_unreadable_book_refuses_to_run(self, monkeypatch):
        submitted = []
        monkeypatch.setattr(
            v2_probe, "signed_request_json",
            lambda *a, **k: submitted.append(k) or KILLED,
        )
        client = probe_client([0])
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=SimpleNamespace(status=200, data=b"{}")
        )
        assert v2_probe._step_unfillable_ask(client, TICKER, True, 1) == v2_probe._NEUTRAL
        assert submitted == []
