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
    The probe exits 2 on any ORDER_API_VERSION but "v2" and never tells the
    operator to switch order paths. The three FAILs that doubt the V2 path
    after a submission on a ticker the probe found flat say to stop trading
    and flatten by hand in the Kalshi UI; main()'s closing
    line says to stop trading after a FAIL but never to flatten, and asks for
    nothing after a NEUTRAL.

Dependencies:
    Imports v2_probe, trader and config (the fee model); patches at each
    function's definition site.
    Offline-only per project policy.

Notes:
    The confirmation prompt is driven by patching builtins.input; --yes paths
    bypass it entirely.

    Two seams beyond the mock client are patched where a test needs the
    account to answer dynamically rather than from a fixed sequence:
    trader._position_count (the probe's ground truth) and v2_probe.time.sleep
    (the one-second recheck pause, so the suite never actually waits). A test
    that patches _position_count passes probe_client([]) — an empty positions
    sequence, so an unpatched read would raise rather than silently pass.
"""
import inspect
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import config, trader, v2_probe
from kalshi_betting.scanner import PriceRange

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
        # Per contract, including Kalshi's rounding of the order's total fee —
        # what the exchange reported for a 0.01-contract fill in live trading
        "average_fee_paid": "0.0200",
    }


FILLED = v2_resp("0.01", "0.00")
KILLED = v2_resp("0.00", "0.01")

# The code the V2 endpoint sent when it killed the probe's fill-or-kill ask on
# the production API (2026-09-28); with the default code, fok_kill_error's body
# is the compact body it sent, byte for byte. KILLED above is the other kill
# shape, a 2xx with nothing filled, which the probe still accepts.
FOK_KILL_CODE = "fill_or_kill_insufficient_resting_volume"


def fok_kill_error(code: str = FOK_KILL_CODE) -> ApiException:
    """The HTTP 409 the V2 endpoint answers a fill-or-kill that cannot fill."""
    body = json.dumps(
        {"error": {"code": code, "message": "fill or kill insufficient resting volume"}},
        separators=(",", ":"),
    )
    return ApiException(status=409, reason="Conflict", body=body)

# The values the V2 create-order endpoint accepts for its required
# self_trade_prevention_type field
# (https://docs.kalshi.com/api-reference/orders/create-order-v2).
_V2_SELF_TRADE_PREVENTION = {"taker_at_cross", "maker"}


def reject_like_the_endpoint(body: dict) -> None:
    """Raise the HTTP 400 the V2 endpoint returns for a body it refuses.

    Two of the endpoint's own checks: self_trade_prevention_type is required
    and must be one of its two values, and "Orders with reduce_only set to
    true will be rejected unless time_in_force is immediate_or_cancel." A
    stand-in submission seam calls this first, so a probe test fails if a
    trader builder stops sending either.
    """
    if body.get("self_trade_prevention_type") not in _V2_SELF_TRADE_PREVENTION:
        raise ApiException(status=400, reason="self_trade_prevention_type is required")
    if body.get("reduce_only") and body.get("time_in_force") != "immediate_or_cancel":
        raise ApiException(
            status=400, reason="reduce_only requires time_in_force immediate_or_cancel",
        )


class FakeExchange:
    """A one-price book that honours limit prices, plus the position it moves.

    Everywhere else in this file the submission seam returns a canned fill or
    a canned kill, which cannot show whether a price would actually have
    crossed. This models the single fact DR-04 turns on: an ASK (sell YES)
    fills only at or BELOW the resting YES bid, a BID (buy YES) fills only at
    or ABOVE the resting YES ask. An order that does not cross is answered as
    the exchange answers it: a fill_or_kill order with the HTTP 409 kill
    response (fok_kill_error), an immediate_or_cancel order with a 2xx that
    has nothing filled and the full count remaining. A body the endpoint
    itself would refuse (reject_like_the_endpoint) raises its HTTP 400 before
    any of that. `position` is a Decimal so -0.01 + 0.01 is exactly 0.
    """

    def __init__(self, yes_bid: str, yes_ask: str):
        self.yes_bid = Decimal(yes_bid)
        self.yes_ask = Decimal(yes_ask)
        self.position = Decimal("0")
        self.submitted: list = []

    def submit(self, client, method, path, *, query=None, body=None):
        """Stand-in for v2_probe.signed_request_json."""
        assert method == "POST"
        self.submitted.append(body)
        reject_like_the_endpoint(body)
        count = Decimal(body["count"])
        price = Decimal(body["price"])
        if body["side"] == "ask":
            # Selling YES: the limit is a FLOOR on proceeds, so it crosses
            # only a resting bid at or above it.
            signed = -count if price <= self.yes_bid else Decimal("0")
        else:
            # Buying YES: the limit is a CEILING, so it crosses only a resting
            # ask at or below it.
            signed = count if price >= self.yes_ask else Decimal("0")
        if body["time_in_force"] == "fill_or_kill" and signed == 0:
            # The exchange rejects a fill-or-kill that cannot fill before it
            # matches: an error response, and the position does not move.
            raise fok_kill_error()
        if body["reduce_only"]:
            # reduce_only can only close existing exposure: a YES bid buys
            # back no more than the NO position actually held, and cannot
            # touch an account that is flat or already long (a bare
            # min(signed, -position) would turn the bid into a SALE out of a
            # long holding and report it as a fill).
            signed = min(signed, max(-self.position, Decimal("0")))
        self.position += signed
        return v2_resp(str(abs(signed)), str(count - abs(signed)))

    def position_count(self, client, ticker):
        """Stand-in for trader._position_count — the account's ground truth."""
        return self.position


@pytest.fixture
def submits(monkeypatch) -> list:
    """Capture every body the probe submits, returning fills by default.

    A body the endpoint would refuse is answered with its HTTP 400 instead
    (reject_like_the_endpoint), after it is captured.

    Patched at v2_probe.signed_request_json — the probe's one submission seam
    (it deliberately bypasses trader._submit_order_v2, whose int-count fill
    classifier cannot express the fractional probe count).
    """
    captured: list = []

    def fake_post(client, method, path, *, query=None, body=None):
        assert method == "POST"
        captured.append({"path": path, "body": body})
        reject_like_the_endpoint(body)
        return FILLED

    monkeypatch.setattr(v2_probe, "signed_request_json", fake_post)
    return captured


def answer(monkeypatch, value: str) -> None:
    """Point the confirmation prompt at a canned answer."""
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: value)


def assert_names_no_other_order_path(printed: str) -> None:
    """Assert the output names no other order path to switch to."""
    assert "legacy" not in printed.lower()
    assert "ORDER_API_VERSION" not in printed


# The stop-trading words, spelled out rather than read back from v2_probe, so
# a change to the probe's text fails here: the scheduler daemon and main.py by
# hand, then the defaults server, whose Confirm and trade starts a new process
# with its disproof latch clear.
_STOP_SCHEDULER_AND_MAIN = (
    "Stop trading until this is understood: stop the scheduler daemon if it is running, "
    "and do not run main.py --mode prod."
)
_STOP_DEFAULTS_SERVER = (
    "If the defaults server is running, stop it with Ctrl-C in the terminal running "
    "./start_dashboard.sh or python3 -m kalshi_betting.defaults_server, and do not "
    "press Confirm and trade."
)
_FLATTEN_BY_HAND = (
    "Flatten any position on the probed ticker by hand in the Kalshi UI; there is no "
    "other order path to fall back on."
)


def assert_names_stop_trading(printed: str) -> None:
    """
    Assert the output says to stop trading every way a real-money run starts.

    Those ways are the scheduler daemon, main.py by hand and the defaults
    server's Confirm and trade, each named in its own sentence.

    Args:
        printed (str): Everything the probe printed, or one closing line.

    Raises:
        AssertionError: When either sentence is missing.
    """
    assert _STOP_SCHEDULER_AND_MAIN in printed
    assert _STOP_DEFAULTS_SERVER in printed


def assert_names_the_remedy(printed: str) -> None:
    """Assert the output carries v2_probe._REMEDY (stop trading, flatten by
    hand in the Kalshi UI) and names no other order path."""
    assert v2_probe._REMEDY in printed
    assert "Kalshi UI" in printed
    # Stopping trading names the defaults server as well as the scheduler
    assert_names_stop_trading(printed)
    assert _FLATTEN_BY_HAND in printed
    assert_names_no_other_order_path(printed)


class TestBodyConstruction:
    """The probe must exercise the REAL trader builders, overriding only the
    fields its fractional count and its price-independent verdict require:
    the count on both bodies, and the price on the close and on the
    deliberately unfillable ask."""

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
        reference = trader._build_no_order_v2(v2_probe._probe_leg(market, 0.41))
        # Everything except count (random client_order_id aside) is the
        # builder's own output — the probe verifies the real code.
        for key in ("ticker", "side", "price", "time_in_force",
                    "self_trade_prevention_type", "exchange_index",
                    "reduce_only", "post_only"):
            assert body[key] == reference[key]
        assert body["count"] == v2_probe.PROBE_COUNT_STR

    def test_probe_leg_is_a_real_trader_leg(self):
        # The probe hands the builders the SAME leg type the live path builds
        # (trader._ordered_legs), so nothing about the probe body can come from
        # a stand-in the builders would read differently.
        market = SimpleNamespace(
            ticker=TICKER, price_level_structure="", price_ranges=None, exchange_index=0,
        )
        leg = v2_probe._probe_leg(market, 0.41)
        assert isinstance(leg, trader._Leg)
        assert leg.market is market
        assert (leg.side, leg.price_dollars, leg.count) == ("no", 0.41, 1)
        assert leg.label == "NO on v2-probe"

    # (price_level_structure, price_ranges, the top-of-grid price the close
    # must carry). The four structure strings CLAUDE.md lists, plus an
    # absent/empty one. The expected price follows from the price_ranges each
    # fixture below declares — scanner.tick_size_for_price reads the bands,
    # not the name — so the empty and "linear_cent" fixtures land on the $0.01
    # grid, the "deci_cent" and "tapered_deci_cent" fixtures put a $0.001 band
    # under 0.9999, and only the "center_deci_edge_centi_cent" fixture puts a
    # $0.0001 band there.
    _TOP_OF_GRID_BY_REGIME = [
        ("", None, "0.9900"),
        ("linear_cent", [PriceRange(start=0.0, end=1.0, step=0.01)], "0.9900"),
        ("deci_cent", [PriceRange(start=0.0, end=1.0, step=0.001)], "0.9990"),
        ("tapered_deci_cent", [
            PriceRange(start=0.0, end=0.05, step=0.001),
            PriceRange(start=0.05, end=0.95, step=0.01),
            PriceRange(start=0.95, end=1.0, step=0.001),
        ], "0.9990"),
        ("center_deci_edge_centi_cent", [
            PriceRange(start=0.0, end=0.01, step=0.0001),
            PriceRange(start=0.01, end=0.99, step=0.001),
            PriceRange(start=0.99, end=1.0, step=0.0001),
        ], "0.9999"),
    ]

    @pytest.mark.parametrize("structure, ranges, expected_price", _TOP_OF_GRID_BY_REGIME)
    def test_close_body_uses_the_real_rollback_builder_then_overrides_count_and_price(
        self, structure, ranges, expected_price,
    ):
        """Re-pinned and renamed from
        test_close_body_uses_the_real_rollback_builder_then_overrides_only_count
        (DR-04): the old expectation that the close carries the BUILDER's price
        was wrong. The builder prices a real unwind at a loss floor derived
        from the leg's scanned entry, and the probe's placeholder entry of 0.5
        puts that floor at $0.62 on every regime below (pinned here) — a bid
        structurally killed on any market whose YES ask is above 62c, which
        FAILed the mapping gate for a reason unrelated to the mapping and left
        a real 0.01 NO position open. Every other key still comes from the
        builder, so the probe still submits the real unwind body.
        """
        market = SimpleNamespace(
            ticker=TICKER, price_level_structure=structure, price_ranges=ranges,
            exchange_index=0,
        )
        body = v2_probe._no_close_body(market)
        reference = trader._build_rollback_order_v2(v2_probe._probe_leg(market, 0.5))
        for key in ("ticker", "side", "time_in_force",
                    "self_trade_prevention_type", "exchange_index",
                    "reduce_only", "post_only"):
            assert body[key] == reference[key]
        assert body["reduce_only"] is True
        # The endpoint accepts reduce_only only with immediate_or_cancel
        assert body["time_in_force"] == "immediate_or_cancel"
        assert body["count"] == v2_probe.PROBE_COUNT_STR
        # The price is the one key that must NOT be the builder's.
        assert body["price"] == expected_price
        assert body["price"] == trader._format_price(
            trader._v2_top_of_grid_price(market)
        )
        assert reference["price"] == "0.6200"


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

    def test_close_prompt_names_the_close_bodys_time_in_force(
        self, submits, monkeypatch, capsys,
    ):
        # The close's confirmation text is built from the close body, so it
        # names the time in force the endpoint actually receives.
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS
        close_body = submits[1]["body"]
        printed = capsys.readouterr().out
        assert close_body["time_in_force"] == "immediate_or_cancel"
        assert f"time_in_force={close_body['time_in_force']}" in printed

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
        assert f"A POSITION IS OPEN ON {TICKER}" in printed
        assert_names_the_remedy(printed)
        # The message says "flatten" once, in the remedy
        assert printed.lower().count("flatten") == 1

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
        # The close's verdict is judged on a re-read after the pause, so a
        # position that stays open needs a fourth read (TestCloseVerdictReRead).
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: None)
        client = probe_client([0, -0.01, -0.01, -0.01])
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
        # DR-60 gave this branch the re-read _non_object_body_fail already had,
        # so it now makes a third position read and sleeps once. The assertion
        # below is unchanged, and the fixture edit is hygiene rather than a
        # necessity: with the old two-entry list the third read exhausts the
        # MagicMock side_effect, trader._position_count fail-softs the
        # StopIteration to None and this still passes — but it would really
        # sleep 1s and silently exercise that swallow. The re-read behaviour
        # itself is pinned by TestUnreadableFillCountsChecksTheAccount.
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: None)
        client = probe_client([0, 0, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._FAIL


class TestCloseCrossesTheBook:
    """DR-04: the closing bid must actually cross the resting YES ask.

    The other no-mapping tests feed the probe a canned fill, so they pass
    whatever the close is priced at. This one runs the whole step against a
    book that honours limit prices.
    """

    def test_close_fills_against_a_resting_ask_the_builder_price_could_not(
        self, monkeypatch,
    ):
        """End-to-end on a linear-cent market quoting 0.89 / 0.90.

        The NO buy goes out as an ask at 0.8800 and crosses the resting YES
        bid of 0.89, opening -0.01. The close then goes out as a bid at
        0.9900 and crosses the resting YES ask of 0.90, returning the account
        to flat — PASS. The builder's own loss-floored price of $0.62 on the
        same market sits BELOW that ask, so before DR-04 the close was killed,
        the step reported FAIL against the mapping, and a real 0.01 NO
        position was left open.
        """
        exchange = FakeExchange(yes_bid="0.89", yes_ask="0.90")
        monkeypatch.setattr(v2_probe, "signed_request_json", exchange.submit)
        # The book, not a scripted sequence, is what moves the position here.
        monkeypatch.setattr(trader, "_position_count", exchange.position_count)
        client = probe_client([])
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=orderbook_resp(yes_bid="0.89", qty="500")
        )
        _, market = v2_probe._fetch_market(client, TICKER)

        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS
        assert exchange.position == 0
        ask_body, close_body = exchange.submitted
        assert (ask_body["side"], ask_body["price"]) == ("ask", "0.8800")
        assert (close_body["side"], close_body["price"]) == ("bid", "0.9900")

        # The counter-factual, on the same market: the rollback builder's
        # loss-floored price cannot reach the resting ask this close crossed.
        builder = trader._build_rollback_order_v2(v2_probe._probe_leg(market, 0.5))
        assert builder["price"] == "0.6200"
        assert Decimal(builder["price"]) < exchange.yes_ask

    def test_the_old_builder_price_would_have_left_the_position_open(
        self, monkeypatch,
    ):
        """The same book, with the close forced back to the builder's price.

        Pins the failure DR-04 describes rather than asserting it in prose: a
        $0.62 bid does not cross a 0.90 ask, the position stays at -0.01, and
        the step reports FAIL — indistinguishable from a broken side mapping.
        """
        exchange = FakeExchange(yes_bid="0.89", yes_ask="0.90")
        monkeypatch.setattr(v2_probe, "signed_request_json", exchange.submit)
        monkeypatch.setattr(trader, "_position_count", exchange.position_count)
        # The open position is read again after the close's re-read pause
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: None)

        def builder_priced_close(market):
            body = trader._build_rollback_order_v2(v2_probe._probe_leg(market, 0.5))
            body["count"] = v2_probe.PROBE_COUNT_STR
            return body

        monkeypatch.setattr(v2_probe, "_no_close_body", builder_priced_close)
        client = probe_client([])
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=orderbook_resp(yes_bid="0.89", qty="500")
        )

        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._FAIL
        assert exchange.submitted[1]["price"] == "0.6200"
        assert exchange.position == Decimal("-0.01")


class TestZeroPositionIsReReadOnce:
    """DR-21: a ledger that reads 0 right after a reported full fill is read
    once more, after trader._V2_MAPPING_RECHECK_DELAY_SECONDS, before any of
    the sign branches run — so the re-read's own value decides the verdict."""

    @staticmethod
    def _arm(monkeypatch, reads: list):
        """Script trader._position_count and neutralize the recheck sleep.

        Returns (observed reads, slept durations) so a test can assert both
        that the extra read happened and that it was the ONLY one.
        """
        seq = iter(reads)
        observed: list = []

        def scripted(client, ticker):
            value = next(seq)
            observed.append(value)
            return value

        slept: list = []
        monkeypatch.setattr(trader, "_position_count", scripted)
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: slept.append(s))
        return observed, slept

    def test_zero_then_negative_confirms_and_submits_the_close(self, submits, monkeypatch):
        # start flat, ledger lags at 0, re-read shows the NO position, close flattens
        observed, slept = self._arm(monkeypatch, [0, 0, -0.01, 0])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._PASS
        assert observed == [0, 0, -0.01, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert len(submits) == 2  # the close really was submitted

    def test_zero_then_positive_takes_the_disproven_branch(self, submits, monkeypatch, capsys):
        # The re-read must reach the sign branches, not fall through to the
        # terminal zero FAIL: a positive position is the disproof this probe
        # exists to catch, and the close must NOT be submitted.
        observed, slept = self._arm(monkeypatch, [0, 0, 0.01])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, 0.01]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "HYPOTHESIS DISPROVEN" in printed
        assert_names_the_remedy(printed)
        assert len(submits) == 1

    def test_zero_then_none_is_a_lookup_failure_never_a_confirmation(
        self, submits, monkeypatch, capsys,
    ):
        # A failed re-read is "state unknown", which must take the None branch
        # rather than being read as either half of the mapping.
        observed, slept = self._arm(monkeypatch, [0, 0, None])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, None]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "position lookup failed" in printed
        assert "CONFIRMED" not in printed
        assert len(submits) == 1

    def test_persistent_zero_still_fails_and_says_it_was_re_read(
        self, submits, monkeypatch, capsys,
    ):
        # Re-pinned wording (DR-21): a ledger still flat after the re-read is
        # genuinely contradictory, so the verdict is unchanged — but the
        # message now says the re-read happened and tells the operator to
        # flatten anything they find.
        observed, slept = self._arm(monkeypatch, [0, 0, 0])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "still 0 after a re-read" in printed
        assert "FLATTEN ANY POSITION YOU FIND" in printed
        assert len(submits) == 1

    def test_a_nonzero_first_read_is_not_re_read(self, submits, monkeypatch):
        # The extra read is spent only on the ambiguous case; a ledger that
        # already moved is evidence and must not be re-polled.
        observed, slept = self._arm(monkeypatch, [0, -0.01, 0])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._PASS
        assert observed == [0, -0.01, 0]
        assert slept == []

    def test_an_unfilled_order_is_not_re_read(self, monkeypatch):
        # No fill was reported, so a zero position is the expected outcome,
        # not a lagging ledger: the step stays NEUTRAL without an extra read.
        monkeypatch.setattr(v2_probe, "signed_request_json", lambda *a, **k: KILLED)
        observed, slept = self._arm(monkeypatch, [0, 0])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        assert observed == [0, 0]
        assert slept == []


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


class TestKillResponse:
    """The exchange answers a fill-or-kill that cannot fill with HTTP 409 and
    the code fill_or_kill_insufficient_resting_volume, not with a 2xx that has
    nothing filled. Both order steps read that response through
    trader._is_fok_kill as a kill and judge the account, re-reading once after
    the pause when the first read is not exactly 0 (a failed lookup, or a
    position the kill cannot explain)."""

    @staticmethod
    def _arm(monkeypatch, error, reads: list):
        """Answer every submission with `error` and script the position reads.

        Returns (submitted bodies, observed reads, sleep durations).
        """
        submitted: list = []

        def post(client, method, path, *, query=None, body=None):
            submitted.append(body)
            reject_like_the_endpoint(body)
            raise error

        monkeypatch.setattr(v2_probe, "signed_request_json", post)

        seq = iter(reads)
        observed: list = []

        def scripted(client, ticker):
            value = next(seq)
            observed.append(value)
            return value

        slept: list = []
        monkeypatch.setattr(trader, "_position_count", scripted)
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: slept.append(s))
        return submitted, observed, slept

    def test_unfillable_ask_kill_with_a_flat_account_passes(self, monkeypatch, capsys):
        submitted, observed, slept = self._arm(monkeypatch, fok_kill_error(), [0, 0])
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._PASS
        assert len(submitted) == 1
        # A flat first read is the expected answer and is not re-polled
        assert observed == [0, 0]
        assert slept == []
        printed = capsys.readouterr().out
        assert "HTTP 409" in printed
        assert FOK_KILL_CODE in printed

    def test_unfillable_ask_kill_with_a_non_flat_first_read_is_judged_on_the_re_read(
        self, monkeypatch, capsys,
    ):
        _, observed, slept = self._arm(monkeypatch, fok_kill_error(), [0, -0.01, 0])
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._PASS
        assert observed == [0, -0.01, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "kill response: -0.01" in printed
        assert "Position after re-read: 0" in printed
        # A position a kill cannot explain stays in the evidence
        assert "NOTE: the first read (-0.01) was not flat" in printed

    def test_unfillable_ask_kill_with_a_failed_first_read_is_re_read(self, monkeypatch, capsys):
        _, observed, slept = self._arm(monkeypatch, fok_kill_error(), [0, None, 0])
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._PASS
        assert observed == [0, None, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        # A failed lookup explains itself: no note
        assert "NOTE:" not in capsys.readouterr().out

    @pytest.mark.parametrize("reads", [[0, -0.01, -0.01], [0, None, None]],
                             ids=["position-stays-open", "lookup-keeps-failing"])
    def test_unfillable_ask_kill_with_an_account_that_is_not_flat_fails(
        self, monkeypatch, capsys, reads,
    ):
        _, observed, slept = self._arm(monkeypatch, fok_kill_error(), reads)
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert observed == reads
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert "CHECK THE ACCOUNT" in capsys.readouterr().out

    def test_unfillable_ask_with_another_409_code_fails(self, monkeypatch, capsys):
        # Only the kill code is a kill; any other error keeps today's FAIL.
        _, observed, slept = self._arm(
            monkeypatch, fok_kill_error("insufficient_balance"), [0, 0],
        )
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert observed == [0, 0]
        assert slept == []
        assert "submission raised" in capsys.readouterr().out

    def test_no_buy_kill_with_a_flat_account_is_neutral(self, monkeypatch, capsys):
        submitted, observed, slept = self._arm(monkeypatch, fok_kill_error(), [0, 0])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        # The close is never submitted: nothing was opened
        assert len(submitted) == 1
        assert observed == [0, 0]
        assert slept == []
        printed = capsys.readouterr().out
        assert "killed unfilled" in printed
        assert "more liquid ticker" in printed

    def test_no_buy_kill_with_a_non_flat_first_read_is_judged_on_the_re_read(
        self, monkeypatch, capsys,
    ):
        submitted, observed, slept = self._arm(monkeypatch, fok_kill_error(), [0, -0.01, 0])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        assert len(submitted) == 1
        assert observed == [0, -0.01, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert "NOTE: the first read (-0.01) was not flat" in capsys.readouterr().out

    def test_no_buy_kill_with_a_position_that_stays_open_fails(self, monkeypatch, capsys):
        submitted, observed, slept = self._arm(
            monkeypatch, fok_kill_error(), [0, -0.01, -0.01],
        )
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert len(submitted) == 1
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert "FLATTEN ANY POSITION YOU FIND" in capsys.readouterr().out

    def test_no_buy_with_another_409_code_fails(self, monkeypatch, capsys):
        submitted, observed, slept = self._arm(
            monkeypatch, fok_kill_error("insufficient_balance"), [0, 0],
        )
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert len(submitted) == 1
        assert slept == []
        assert "NO-buy submission raised" in capsys.readouterr().out

    def test_the_stand_in_exchange_kills_the_unfillable_ask(self, monkeypatch):
        # End to end against a book that honours limit prices: a top-of-grid
        # ask crosses no resting bid, so the exchange answers with its 409.
        exchange = FakeExchange(yes_bid="0.59", yes_ask="0.60")
        monkeypatch.setattr(v2_probe, "signed_request_json", exchange.submit)
        monkeypatch.setattr(trader, "_position_count", exchange.position_count)
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._PASS
        assert exchange.position == 0
        assert len(exchange.submitted) == 1

    def test_a_no_buy_that_no_longer_crosses_is_neutral_end_to_end(self, monkeypatch):
        # The book the probe priced against shows a YES bid of 0.59, but the
        # exchange's bid has dropped to 0.50: the 0.5800 ask cannot fill, the
        # exchange kills it with its 409, and nothing is opened or closed.
        exchange = FakeExchange(yes_bid="0.50", yes_ask="0.60")
        monkeypatch.setattr(v2_probe, "signed_request_json", exchange.submit)
        monkeypatch.setattr(trader, "_position_count", exchange.position_count)
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        assert exchange.position == 0
        [ask_body] = exchange.submitted
        assert (ask_body["side"], ask_body["price"]) == ("ask", "0.5800")


class TestCloseVerdictReRead:
    """A position that is not exactly 0 straight after the reduce-only close is
    read once more after trader._V2_MAPPING_RECHECK_DELAY_SECONDS, and the
    close is judged on the re-read: the positions ledger lags a fill, so the
    lone read after a close that did flatten the account can still show the
    open position."""

    def test_a_lagging_ledger_after_the_close_passes(self, submits, monkeypatch, capsys):
        observed, slept = TestZeroPositionIsReReadOnce._arm(
            monkeypatch, [0, -0.01, -0.01, 0],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._PASS
        assert observed == [0, -0.01, -0.01, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert len(submits) == 2
        printed = capsys.readouterr().out
        assert "Position after the close: -0.01" in printed
        assert "Position after re-read: 0" in printed

    def test_a_failed_read_after_the_close_is_re_read(self, submits, monkeypatch):
        observed, slept = TestZeroPositionIsReReadOnce._arm(
            monkeypatch, [0, -0.01, None, 0],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._PASS
        assert observed == [0, -0.01, None, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]

    @pytest.mark.parametrize("last", [-0.01, None], ids=["stays-open", "lookup-fails"])
    def test_a_position_still_open_after_the_re_read_fails(
        self, submits, monkeypatch, capsys, last,
    ):
        observed, slept = TestZeroPositionIsReReadOnce._arm(
            monkeypatch, [0, -0.01, -0.01, last],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, -0.01, -0.01, last]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "did NOT return the position to flat" in printed
        assert f"Position after re-read: {last}" in printed
        assert_names_the_remedy(printed)

    def test_a_flat_read_after_the_close_is_not_re_read(self, submits, monkeypatch):
        observed, slept = TestZeroPositionIsReReadOnce._arm(monkeypatch, [0, -0.01, 0])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._PASS
        assert observed == [0, -0.01, 0]
        assert slept == []


class TestNonObjectOrderBody:
    """DR-58: a 2xx order body that is not a JSON object must FAIL cleanly.

    _fill_counts and _report_fee both call data.get(...), so a body of
    "accepted" / [] / 123 / true / null raised an uncaught AttributeError
    IMMEDIATELY AFTER A REAL ASK HAD BEEN SUBMITTED — the probe died on a
    traceback with no position read, no reduce-only close and no
    flatten-it-manually warning, leaving a real position open with nothing to
    tell the operator it existed. Same reading trader._execute_transfer takes
    of a non-object 2xx transfer body (DR-05); the trader.py ORDER readers
    deliberately keep raising, because _execute_one resolves them against the
    position delta, and nothing like _execute_one sits above the probe.
    """

    # Every shape _http.signed_request_json can hand back from a 2xx that is
    # not a JSON object. `None` is the literal `null` body; 1.5 is a bare JSON
    # number that is not an int, so the matrix covers both numeric spellings.
    _NON_OBJECT_BODIES = ["accepted", [], 123, 1.5, True, None]

    @staticmethod
    def _arm(monkeypatch, response, reads: list):
        """Hand `response` back from the submission seam and script the position
        reads.

        Returns (submitted bodies, observed reads, sleep durations) so a test
        can assert that the position really was looked up, that the extra
        re-read happened only when the first read was flat or unreadable, and
        that the step stopped before any further submission.
        """
        submitted: list = []

        def post(client, method, path, *, query=None, body=None):
            submitted.append(body)
            return response

        monkeypatch.setattr(v2_probe, "signed_request_json", post)

        seq = iter(reads)
        observed: list = []

        def scripted(client, ticker):
            value = next(seq)
            observed.append(value)
            return value

        slept: list = []
        monkeypatch.setattr(trader, "_position_count", scripted)
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: slept.append(s))
        return submitted, observed, slept

    @pytest.mark.parametrize("body", _NON_OBJECT_BODIES)
    def test_no_mapping_fails_and_checks_the_account(self, body, monkeypatch, capsys):
        # start flat, then the guard's own read finds the position the
        # unreadable response may have opened.
        submitted, observed, _ = self._arm(monkeypatch, body, [0, -0.01])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        # The position WAS looked up — that lookup is the whole remedy.
        assert observed == [0, -0.01]
        # Only the opening ask went out: the close rests on a mapping this
        # response proved nothing about.
        assert len(submitted) == 1
        printed = capsys.readouterr().out
        assert type(body).__name__ in printed
        assert "FLATTEN IT MANUALLY" in printed

    @pytest.mark.parametrize("body", _NON_OBJECT_BODIES)
    def test_unfillable_ask_fails_and_checks_the_account(self, body, monkeypatch, capsys):
        submitted, observed, _ = self._arm(monkeypatch, body, [0, -0.01])
        out = v2_probe._step_unfillable_ask(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert observed == [0, -0.01]
        assert len(submitted) == 1
        printed = capsys.readouterr().out
        assert type(body).__name__ in printed
        assert "FLATTEN IT MANUALLY" in printed

    def test_a_flat_first_read_is_re_read_once_before_concluding(
        self, monkeypatch, capsys,
    ):
        # DR-21's reasoning applies here too: a ledger that reads flat straight
        # after a submission is usually lag, and the re-read is what surfaces
        # the stranded position.
        _, observed, slept = self._arm(monkeypatch, "accepted", [0, 0, -0.01])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, -0.01]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        assert "FLATTEN IT MANUALLY" in capsys.readouterr().out

    def test_a_persistently_flat_account_says_so_rather_than_warning(
        self, monkeypatch, capsys,
    ):
        # Still a FAIL (nothing was proven), but the operator must be able to
        # tell "flat" from "unreadable" and from "open".
        _, observed, slept = self._arm(monkeypatch, [], [0, 0, 0])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "nothing to flatten" in printed
        assert "FLATTEN IT MANUALLY" not in printed

    def test_an_unreadable_position_says_check_it_manually(self, monkeypatch, capsys):
        # None is "the lookup failed", which must not be collapsed into "flat".
        _, observed, slept = self._arm(monkeypatch, 123, [0, None, None])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, None, None]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "CHECK IT MANUALLY" in printed
        assert "nothing to flatten" not in printed

    def test_a_nonzero_first_read_is_not_re_read(self, monkeypatch):
        # A ledger that already moved is evidence; don't spend a second read.
        _, observed, slept = self._arm(monkeypatch, True, [0, -0.01])
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, -0.01]
        assert slept == []

    def test_a_genuine_json_object_body_is_untouched(self, submits):
        # The guard must not disturb the normal path: a real object body still
        # opens, confirms and closes the position exactly as before.
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS
        assert [b["body"]["side"] for b in submits] == ["ask", "bid"]

    def test_a_genuine_json_object_kill_body_is_untouched(self, monkeypatch):
        monkeypatch.setattr(v2_probe, "signed_request_json", lambda *a, **k: KILLED)
        client = probe_client([0, 0])
        assert v2_probe._step_unfillable_ask(client, TICKER, True, 1) == v2_probe._PASS


class TestUnreadableFillCountsChecksTheAccount:
    """DR-60: a JSON-OBJECT body whose fill counts cannot be read leaves the
    probe in exactly the state _non_object_body_fail handles one step earlier —
    a real ask submitted, a 2xx back, and no idea what it did — so it takes the
    same remedy.

    Before this, the branch decided from a SINGLE un-refreshed position read
    (DR-21's re-read is gated on `filled`, which is None and therefore falsy
    here) and printed a warning only when that read was truthy. A lagging
    ledger printed `Position after the NO buy: 0.0`, no warning at all, and
    returned FAIL while a real 0.01 NO position was open on the production
    account; a FAILED lookup (None) printed nothing either, collapsed into
    "flat" by the same bare truthiness test DR-58 forbade.
    """

    # Both shapes _fill_counts reports as unreadable on a real dict body: no
    # count fields at all, and one of the two present without the other.
    _UNREADABLE_BODIES = [
        {"order": {"status": "executed"}},
        {"fill_count": "0.01"},
    ]

    @staticmethod
    def _arm(monkeypatch, response, reads: list):
        """Hand `response` back from the submission seam and script the reads.

        Returns (submitted bodies, observed reads, sleep durations) — the same
        shape TestNonObjectOrderBody._arm returns, because the two classes pin
        the same helper from its two call sites.
        """
        submitted: list = []

        def post(client, method, path, *, query=None, body=None):
            submitted.append(body)
            return response

        monkeypatch.setattr(v2_probe, "signed_request_json", post)

        seq = iter(reads)
        observed: list = []

        def scripted(client, ticker):
            value = next(seq)
            observed.append(value)
            return value

        slept: list = []
        monkeypatch.setattr(trader, "_position_count", scripted)
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: slept.append(s))
        return submitted, observed, slept

    @pytest.mark.parametrize("body", _UNREADABLE_BODIES)
    def test_a_flat_first_read_is_re_read_and_surfaces_the_position(
        self, body, monkeypatch, capsys,
    ):
        # start flat, the ledger lags at 0 straight after the fill, the re-read
        # finds the 0.01 NO position the operator has to flatten.
        submitted, observed, slept = self._arm(monkeypatch, body, [0, 0, -0.01])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        assert observed == [0, 0, -0.01]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        # Only the opening ask went out — the close rests on a mapping this
        # unreadable response proved nothing about.
        assert len(submitted) == 1
        printed = capsys.readouterr().out
        assert "FLATTEN IT MANUALLY" in printed

    def test_an_unreadable_position_says_check_it_manually(self, monkeypatch, capsys):
        # None is "the lookup failed", never "the account is flat".
        _, observed, slept = self._arm(
            monkeypatch, {"order": {"status": "executed"}}, [0, None, None],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, None, None]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "CHECK IT MANUALLY" in printed
        assert "FLATTEN IT MANUALLY" not in printed

    def test_a_persistently_flat_account_says_so_rather_than_staying_silent(
        self, monkeypatch, capsys,
    ):
        # Still FAIL (nothing was proven), but a checked-and-flat account must
        # be distinguishable from a step that never looked.
        _, observed, slept = self._arm(
            monkeypatch, {"order": {"status": "executed"}}, [0, 0, 0],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "nothing to flatten" in printed
        assert "FLATTEN IT MANUALLY" not in printed

    def test_a_nonzero_first_read_is_not_re_read(self, monkeypatch, capsys):
        # A ledger that already moved is evidence; don't spend a second read.
        _, observed, slept = self._arm(
            monkeypatch, {"order": {"status": "executed"}}, [0, -0.01],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, -0.01]
        assert slept == []
        assert "FLATTEN IT MANUALLY" in capsys.readouterr().out


class TestNonConformingFillOrKill:
    """DR-20: a fill-or-kill response that is neither a complete fill nor a
    true kill is a PROTOCOL VIOLATION, not a clean kill.

    `filled = remaining == 0 and fill == PROBE_COUNT` collapsed every non-full
    outcome into one boolean. A partial fill therefore took the `not filled`
    branch, where DR-21's re-read does not run (it is gated on `filled`), so a
    lagging ledger printed "killed unfilled and the account is still flat" and
    exited NEUTRAL while a real fraction of a contract was open on the
    production account. The sibling step _step_unfillable_ask already got this
    right; the two now agree about what a partial fill is.
    """

    # Every readable shape that is neither a complete fill nor a true kill,
    # against a count of 0.01: a partial, an over-fill, and a stale remainder
    # (the counts do not even sum to the order).
    _NON_CONFORMING = [
        ("0.005", "0.005"),
        ("0.02", "0.00"),
        ("0.01", "0.005"),
    ]

    @staticmethod
    def _arm(monkeypatch, response, reads: list):
        """Hand `response` back from the submission seam and script the reads.

        Same shape as TestNonObjectOrderBody._arm /
        TestUnreadableFillCountsChecksTheAccount._arm — all three classes pin
        branches that end in the shared _recheck_and_report_position tail.
        """
        submitted: list = []

        def post(client, method, path, *, query=None, body=None):
            submitted.append(body)
            return response

        monkeypatch.setattr(v2_probe, "signed_request_json", post)

        seq = iter(reads)
        observed: list = []

        def scripted(client, ticker):
            value = next(seq)
            observed.append(value)
            return value

        slept: list = []
        monkeypatch.setattr(trader, "_position_count", scripted)
        monkeypatch.setattr(v2_probe.time, "sleep", lambda s: slept.append(s))
        return submitted, observed, slept

    def test_partial_fill_with_a_lagging_ledger_fails_and_surfaces_the_position(
        self, monkeypatch, capsys,
    ):
        # THE DR-20 case: half the probe count filled, the ledger has not
        # caught up yet, and the old code called that "killed unfilled and the
        # account is still flat" — NEUTRAL, no re-read, no warning.
        submitted, observed, slept = self._arm(
            monkeypatch, v2_resp("0.005", "0.005"), [0, 0, -0.005],
        )
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        # The ledger really was re-read once before anything was reported.
        assert observed == [0, 0, -0.005]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        # Only the opening ask went out — the close rests on a mapping this
        # response proved nothing about.
        assert len(submitted) == 1
        printed = capsys.readouterr().out
        assert "fill_count=0.005" in printed
        assert "remaining_count=0.005" in printed
        assert "FLATTEN IT MANUALLY" in printed
        # The false claim this bug was made of must be gone.
        assert "still flat" not in printed
        assert_names_the_remedy(printed)

    def test_partial_fill_with_a_genuinely_flat_ledger_still_fails(
        self, monkeypatch, capsys,
    ):
        # A fill-or-kill invariant violation is a violation even if nothing
        # ended up landing: the response shape is what is being judged.
        _, observed, slept = self._arm(
            monkeypatch, v2_resp("0.005", "0.005"), [0, 0, 0],
        )
        assert v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1) == v2_probe._FAIL
        assert observed == [0, 0, 0]
        assert slept == [trader._V2_MAPPING_RECHECK_DELAY_SECONDS]
        printed = capsys.readouterr().out
        assert "fill_count=0.005" in printed
        assert "remaining_count=0.005" in printed
        # Checked-and-flat is reported, but never as "the kill left us flat".
        assert "nothing to flatten" in printed
        assert "still flat" not in printed

    @pytest.mark.parametrize("fill, remaining", _NON_CONFORMING)
    def test_no_non_conforming_shape_is_read_as_a_kill_or_a_fill(
        self, fill, remaining, monkeypatch, capsys,
    ):
        # An over-fill and a stale remainder are as impossible as a partial;
        # none of them may reach the kill NEUTRAL or the fill PASS.
        submitted, observed, slept = self._arm(
            monkeypatch, v2_resp(fill, remaining), [0, -0.01, -0.01],
        )
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._FAIL
        # Only TWO of the three scripted reads are consumed — the baseline and
        # the post-buy read — because that post-buy read has already MOVED, and
        # _recheck_and_report_position does not re-poll a read that is itself
        # evidence. No re-read means no delay either.
        assert observed == [0, -0.01]
        assert slept == []
        assert len(submitted) == 1  # never goes on to the close
        printed = capsys.readouterr().out
        assert f"fill_count={fill}" in printed
        assert f"remaining_count={remaining}" in printed
        assert "still flat" not in printed
        assert "CONFIRMED" not in printed

    def test_a_true_kill_with_a_flat_account_is_still_neutral(self, monkeypatch, capsys):
        # The correct path must not have regressed: a genuine kill against a
        # genuinely flat account keeps its NEUTRAL verdict and its wording.
        submitted, observed, slept = self._arm(monkeypatch, KILLED, [0, 0])
        out = v2_probe._step_no_mapping(probe_client([]), TICKER, True, 1)
        assert out == v2_probe._NEUTRAL
        assert observed == [0, 0]
        assert slept == []
        assert len(submitted) == 1
        printed = capsys.readouterr().out
        assert "killed unfilled and the account is still flat" in printed

    def test_a_full_fill_still_passes(self, submits, monkeypatch):
        # The conforming-fill path is untouched.
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS
        assert [b["body"]["side"] for b in submits] == ["ask", "bid"]

    def test_the_fee_line_does_not_claim_nothing_filled(self, capsys):
        # _report_fee reads no fill counts, so it cannot assert an empty fill —
        # on a partial that was a second, independent false claim printed right
        # above the verdict (DR-20).
        v2_probe._report_fee({"order_id": "x"}, "0.4100")
        printed = capsys.readouterr().out
        assert "nothing filled" not in printed
        assert "no average_fee_paid" in printed


class TestFeeCheckIsPerContract:
    """_report_fee prints the exchange's average_fee_paid, which is per
    contract, beside the fee model per contract three ways: before rounding,
    for one whole contract rounded up to the cent, and rounded as Kalshi
    rounds this order — the figure a tiny order's charge should match."""

    @staticmethod
    def _unrounded(price: float) -> str:
        return f"${config.TAKER_FEE_RATE * price * (1 - price):.6f}"

    def test_every_figure_is_labelled_per_contract_at_the_fill_price(self, capsys):
        # The model is worked out at the fill price, not the limit price
        v2_probe._report_fee(FILLED, "0.4300")
        first, second = capsys.readouterr().out.strip().split("\n")
        assert "per contract at p=0.41 (average fill price)" in first
        assert "exchange average_fee_paid=$0.0200 per contract" in first
        assert f"TAKER_FEE_RATE*p*(1-p) = {self._unrounded(0.41)} per contract before rounding" in first
        assert (
            f"config.fee_leg_exact(1, 0.41) = ${config.fee_leg_exact(1, 0.41):.2f}"
            " for one whole contract, rounded up to the cent"
        ) in first
        assert second.startswith("  Kalshi rounds each order's total fee up to the account's balance precision")
        # No "probe traded" hint to scale the figures by the contract count
        assert "probe traded" not in first + second

    def test_the_exchange_rounding_reproduces_the_live_probe_charge(self, capsys):
        # Fills at 0.58 and 0.59 model to $0.0200 per contract at $0.0001 (the
        # figure the exchange charged) and $1.00 at $0.01
        for fill in ("0.5800", "0.5900"):
            v2_probe._report_fee(dict(FILLED, average_fill_price=fill), "0.5700")
            second = capsys.readouterr().out.strip().split("\n")[1]
            assert (
                "on this 0.01-contract order the model comes to $0.0200 per contract at"
                " $0.0001 precision or $1.0000 per contract at $0.01 precision."
            ) in second

    def test_the_order_rounding_is_exact_decimal_arithmetic(self):
        # For 0.01 contracts, rounding the order's total fee up to $0.0001
        # gives the same figure as config.fee_leg_exact(1, p), at every cent
        # price
        assert v2_probe.PROBE_COUNT_STR == "0.01"
        for cents in range(1, 100):
            price = cents / 100
            rounded = v2_probe._order_rounded_fee_per_contract(price, "0.0001")
            assert rounded == Decimal(str(config.fee_leg_exact(1, price))), price
        assert v2_probe._order_rounded_fee_per_contract(0.5, "0.01") == Decimal("1")

    def test_an_order_object_nested_under_order_is_read(self, capsys):
        v2_probe._report_fee({"order": FILLED}, "0.4300")
        assert "per contract at p=0.41 (average fill price)" in capsys.readouterr().out

    @pytest.mark.parametrize("fill_price", [
        None, "garbage", True, "1.5", "-0.1", "nan", "inf", {}, 10**400, "0", "0.0000", "1",
    ])
    def test_the_limit_price_stands_in_for_an_unreadable_fill_price(self, fill_price, capsys):
        # Anything that is not a number strictly between 0 and 1 (including 0
        # and 1, true/false and a number too large for a float) falls back to
        # the limit price
        body = dict(FILLED, average_fill_price=fill_price)
        if fill_price is None:
            del body["average_fill_price"]
        v2_probe._report_fee(body, "0.4300")
        printed = capsys.readouterr().out
        assert "per contract at p=0.43 (limit price)" in printed
        assert f"= {self._unrounded(0.43)} per contract before rounding" in printed

    def test_with_no_price_at_all_the_charge_is_still_printed(self, capsys):
        body = dict(FILLED, average_fill_price="garbage")
        v2_probe._report_fee(body, "not a price")
        printed = capsys.readouterr().out
        assert "exchange average_fee_paid=$0.0200 per contract" in printed
        assert "'not a price'" in printed
        assert "before rounding" not in printed
        assert "Kalshi rounds" not in printed

    def test_a_full_probe_run_prints_the_per_contract_lines(self, submits, capsys):
        client = probe_client([0, -0.01, 0])
        assert v2_probe._step_no_mapping(client, TICKER, True, 1) == v2_probe._PASS
        printed = capsys.readouterr().out
        assert "exchange average_fee_paid=$0.0200 per contract" in printed
        assert "per contract before rounding" in printed
        assert "the model comes to $0.0200 per contract at $0.0001 precision" in printed


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

    def test_self_transfer_is_refused_before_any_post(self, monkeypatch, capsys):
        # DR-22: --dest-shard 0 IS the source shard. Both existing guards (is
        # the shard advertised, are its transfers active) are satisfied by the
        # source shard by construction, so this used to POST a real,
        # non-idempotent, never-retried transfer that could not raise the
        # shard's balance — the settlement poll then burned the full timeout
        # and reported a FALSE "MONEY MAY BE IN FLIGHT".
        statuses_read: list = []
        monkeypatch.setattr(
            v2_probe.scanner, "fetch_shard_statuses",
            lambda c: statuses_read.append(c) or shard_statuses(),
        )
        monkeypatch.setattr(
            trader, "_execute_transfer",
            lambda *a, **k: pytest.fail("a self-transfer must never be POSTed"),
        )
        out = v2_probe._step_transfer(
            MagicMock(), None, True, v2_probe._TRANSFER_SOURCE_SHARD
        )
        assert out == v2_probe._NEUTRAL
        # Refused before ANY I/O, not merely before the POST.
        assert statuses_read == []
        printed = capsys.readouterr().out
        assert "IS the source shard" in printed
        assert "MONEY MAY BE IN FLIGHT" not in printed

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

    @pytest.mark.parametrize("value", ["legacy", "V2", "", None])
    @pytest.mark.parametrize("step", sorted(v2_probe._STEPS))
    def test_a_non_v2_order_path_is_refused_before_anything_runs(
        self, monkeypatch, capsys, value, step,
    ):
        # Any value but "v2" exits 2 before the banner, logging, a client or
        # any step
        monkeypatch.setattr(config, "ORDER_API_VERSION", value)
        monkeypatch.setattr(
            v2_probe.auth, "build_client",
            lambda mode: pytest.fail("must refuse before building a client"),
        )
        monkeypatch.setattr(
            v2_probe.logging, "basicConfig",
            lambda *a, **k: pytest.fail("must refuse before configuring logging"),
        )
        monkeypatch.setitem(
            v2_probe._STEPS, step, lambda *a: pytest.fail("must refuse before any step"),
        )
        with pytest.raises(SystemExit) as exc_info:
            v2_probe.main(["--ticker", TICKER, "--step", step, "--yes"])
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "KALSHI V2 ORDER-PATH LIVE PROBE" not in captured.out
        assert config.order_api_version_error() in captured.err
        assert repr(value) in captured.err

    @staticmethod
    def _closing(monkeypatch, capsys, step: str, outcome: str, argv: list) -> str:
        """Run main() with `step` stubbed to return `outcome`; return what it
        printed after the RESULT banner (the closing line)."""
        monkeypatch.setattr(v2_probe.auth, "build_client", lambda mode: MagicMock())
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda c: {0: 100})
        monkeypatch.setitem(v2_probe._STEPS, step, lambda c, t, y, d: outcome)
        v2_probe.main(argv)
        printed = capsys.readouterr().out
        return printed.split(f"RESULT: {step} -> {outcome}", 1)[1]

    @pytest.mark.parametrize("step", sorted(v2_probe._TICKER_STEPS))
    def test_an_order_step_fail_ends_on_stop_trading_and_never_says_flatten(
        self, monkeypatch, capsys, step,
    ):
        # The closing line after an order-step FAIL says to stop trading and
        # to act only on the step's own warnings, never to flatten
        closing = self._closing(
            monkeypatch, capsys, step, v2_probe._FAIL, ["--ticker", TICKER, "--step", step],
        )
        assert v2_probe._STOP_TRADING in closing
        assert_names_stop_trading(closing)
        assert "Act only on the position warnings printed above" in closing
        assert v2_probe._FLATTEN not in closing
        assert "flatten any position" not in closing.lower()
        assert ("Both --step no-mapping and --step unfillable-ask have to PASS before the "
                "V2 order path should be trusted to run unsupervised") in closing
        assert_names_no_other_order_path(closing)

    def test_a_transfer_fail_ends_on_stop_trading_and_the_shard_balances(
        self, monkeypatch, capsys,
    ):
        # The closing line after a transfer FAIL says to stop trading and
        # check the shard balances, and names no ticker
        closing = self._closing(
            monkeypatch, capsys, "transfer", v2_probe._FAIL,
            ["--step", "transfer", "--ticker", TICKER],
        )
        assert v2_probe._STOP_TRADING in closing
        assert_names_stop_trading(closing)
        assert "Check each shard's balance in the Kalshi UI" in closing
        assert "flatten" not in closing.lower()
        assert TICKER not in closing and "ticker" not in closing.lower()
        assert_names_no_other_order_path(closing)

    @pytest.mark.parametrize("step", sorted(v2_probe._STEPS))
    def test_a_neutral_calls_for_no_action(self, monkeypatch, capsys, step):
        # The closing line after a NEUTRAL asks for no halt and no flatten
        # (the transfer step runs with no ticker)
        argv = ["--step", step] + (["--ticker", TICKER] if step in v2_probe._TICKER_STEPS
                                   else [])
        closing = self._closing(monkeypatch, capsys, step, v2_probe._NEUTRAL, argv)
        assert "inconclusive" in closing
        assert "stop trading" not in closing.lower()
        assert "defaults server" not in closing
        assert "flatten" not in closing.lower()
        assert ("Both --step no-mapping and --step unfillable-ask have to PASS before the "
                "V2 order path should be trusted to run unsupervised") in closing
        assert_names_no_other_order_path(closing)

    @pytest.mark.parametrize("step", sorted(v2_probe._TICKER_STEPS))
    @pytest.mark.parametrize("cause", ["position_not_flat", "no_market"])
    def test_a_fail_before_any_submission_never_says_to_flatten(
        self, monkeypatch, capsys, submits, step, cause,
    ):
        # A real step that stops before submitting (the ticker already holds
        # a position, or the market cannot be read) submits nothing and never
        # says to flatten
        client = probe_client([5])
        if cause == "no_market":
            client.get_market_without_preload_content = MagicMock(
                return_value=SimpleNamespace(status=200, data=b'{"market": {}}'),
            )
        monkeypatch.setattr(v2_probe.auth, "build_client", lambda mode: client)
        monkeypatch.setattr(v2_probe.auth, "verify_auth", lambda c: {0: 100})
        assert v2_probe.main(["--ticker", TICKER, "--step", step, "--yes"]) == 1
        assert submits == []
        printed = capsys.readouterr().out
        assert ("probe must start FLAT" if cause == "position_not_flat"
                else "no market returned") in printed
        assert v2_probe._FLATTEN not in printed
        assert "flatten any position" not in printed.lower()
        assert "FLATTEN" not in printed
        assert "close the position" not in printed.lower()
        assert "never flatten a position the bot holds" in printed
        assert_names_no_other_order_path(printed)

    def test_a_step_that_passes_prints_no_remedy(self, monkeypatch, capsys):
        closing = self._closing(
            monkeypatch, capsys, "no-mapping", v2_probe._PASS, ["--ticker", TICKER],
        )
        assert "Record this output" in closing
        assert v2_probe._STOP_TRADING not in closing
        assert "defaults server" not in closing
        assert "flatten" not in closing.lower()

    def test_stop_trading_names_every_way_a_real_money_run_starts(self):
        # The stop-trading sentence keeps its scheduler-and-main.py words and
        # then names the defaults server, whose Confirm and trade starts a new
        # process; the remedy and both FAIL closing lines start with it.
        assert v2_probe._STOP_TRADING == f"{_STOP_SCHEDULER_AND_MAIN} {_STOP_DEFAULTS_SERVER}"
        assert v2_probe._FLATTEN == _FLATTEN_BY_HAND
        assert v2_probe._REMEDY == (
            f"{_STOP_SCHEDULER_AND_MAIN} {_STOP_DEFAULTS_SERVER} {_FLATTEN_BY_HAND}"
        )
        assert v2_probe._ORDER_FAIL_CLOSING.startswith(v2_probe._STOP_TRADING)
        assert v2_probe._TRANSFER_FAIL_CLOSING.startswith(v2_probe._STOP_TRADING)

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
