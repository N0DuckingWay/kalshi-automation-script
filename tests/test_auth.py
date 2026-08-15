"""
File: test_auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting.auth — the shard-aware balance
    parsing added when Kalshi scoped GET /portfolio/balance per exchange shard
    (2026-08-13). Covers the dollar-string → floored-cents converter, the
    tiered fallback chain in _balance_cents_from_payload (shard-0 breakdown
    entry → aggregate balance_dollars → legacy integer cents), and verify_auth
    end-to-end against a faked raw response.

Dependencies:
    Imports _balance_cents_from_payload, _dollar_str_to_cents, and verify_auth
    from kalshi_betting.auth, and ROUTABLE_EXCHANGE_INDEX from
    kalshi_betting.config. Uses unittest.mock / SimpleNamespace to stand in for
    the SDK client and its RESTResponse — no network access.

Notes:
    The same-key-different-units trap is asserted explicitly: inside a
    balance_breakdown entry "balance" is a fixed-point DOLLAR STRING, while the
    TOP-LEVEL "balance" is legacy INTEGER CENTS. A test pins each.
"""
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting.auth import (
    _balance_cents_from_payload,
    _dollar_str_to_cents,
    verify_auth,
)
from kalshi_betting.config import ROUTABLE_EXCHANGE_INDEX

# The exact live body observed on this account 2026-08-14. Top-level "balance"
# is integer cents; the breakdown entries carry dollar strings under "balance".
LIVE_PAYLOAD = {
    "balance": 114,
    "balance_breakdown": [
        {"balance": "1.1407", "exchange_index": 0},
        {"balance": "0.0000", "exchange_index": 1},
    ],
    "balance_dollars": "1.1407",
    "portfolio_value": 39796,
    "updated_ts": 1786699278,
}


def _raw_balance_response(payload: dict, status: int = 200) -> SimpleNamespace:
    """Build a raw-response stand-in for get_balance_without_preload_content.

    verify_auth bypasses the SDK's strict balance model and parses the JSON
    body itself, so mocks provide (status, data-bytes) exactly like the SDK's
    RESTResponse.
    """
    return SimpleNamespace(
        status=status,
        data=json.dumps(payload).encode("utf-8"),
        getheaders=lambda: {},
        reason="OK" if status == 200 else "Error",
    )


class TestDollarStrToCents:
    def test_live_fixed_point_string_is_floored(self):
        # "1.1407" is 114.07 cents — flooring must never report the extra cent
        # the account cannot actually spend.
        assert _dollar_str_to_cents("1.1407") == 114

    def test_sub_cent_precision_floors_not_rounds(self):
        # 12345.6 cents rounds to 12346 but must floor to 12345.
        assert _dollar_str_to_cents("123.456") == 12345

    def test_zero(self):
        assert _dollar_str_to_cents("0.0000") == 0

    def test_none_returns_none(self):
        assert _dollar_str_to_cents(None) is None

    def test_garbage_returns_none(self):
        assert _dollar_str_to_cents("garbage") is None

    def test_float_input_handled(self):
        assert _dollar_str_to_cents(1.14) == 114

    def test_int_input_handled(self):
        assert _dollar_str_to_cents(2) == 200


class TestBalanceCentsFromPayload:
    def test_live_payload_uses_shard_zero_entry(self):
        assert _balance_cents_from_payload(LIVE_PAYLOAD) == 114

    def test_shard_zero_preferred_over_aggregate(self):
        # The whole point of shard-awareness: $100 parked on shard 1 is
        # unreachable by our order path and must NOT inflate Kelly sizing.
        payload = {
            "balance_breakdown": [
                {"balance": "5.00", "exchange_index": 0},
                {"balance": "100.00", "exchange_index": 1},
            ],
            "balance_dollars": "105.00",
            "balance": 10500,
        }
        assert _balance_cents_from_payload(payload) == 500

    def test_no_breakdown_falls_back_to_balance_dollars(self):
        assert _balance_cents_from_payload({"balance_dollars": "42.50"}) == 4250

    def test_legacy_integer_payload_is_already_cents(self):
        # Sandbox shape. 5000 means $50.00 — running it through the dollar
        # converter would report $5,000.00.
        assert _balance_cents_from_payload({"balance": 5000}) == 5000

    def test_breakdown_without_shard_zero_warns_and_uses_aggregate(self, caplog):
        payload = {
            "balance_breakdown": [{"balance": "100.00", "exchange_index": 1}],
            "balance_dollars": "100.00",
        }
        with caplog.at_level(logging.WARNING):
            assert _balance_cents_from_payload(payload) == 10000
        assert any(
            "balance_breakdown" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_malformed_entries_do_not_break_shard_zero_lookup(self):
        payload = {
            "balance_breakdown": [
                "not-a-dict",
                None,
                {"exchange_index": None, "balance": "9.99"},
                {"exchange_index": "abc", "balance": "8.88"},
                {"balance": "7.77"},
                {"balance": "3.50", "exchange_index": ROUTABLE_EXCHANGE_INDEX},
            ],
            "balance_dollars": "999.00",
        }
        assert _balance_cents_from_payload(payload) == 350

    def test_shard_zero_entry_with_unparseable_balance_falls_through(self):
        payload = {
            "balance_breakdown": [{"balance": "garbage", "exchange_index": 0}],
            "balance_dollars": "12.00",
        }
        assert _balance_cents_from_payload(payload) == 1200

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_from_payload({})

    def test_nonsense_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_from_payload({"nonsense": 1})

    def test_non_int_legacy_balance_raises(self):
        # A bool is an int subclass but a string is not — a stringy top-level
        # balance is not silently treated as cents.
        with pytest.raises(ValueError):
            _balance_cents_from_payload({"balance": "114"})


class TestVerifyAuth:
    def test_returns_shard_zero_cents_from_raw_response(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response(LIVE_PAYLOAD)
        )

        result = verify_auth(client)

        assert result == 114
        assert isinstance(result, int)
        # The modeled call must not be used — it deserializes through the
        # pinned SDK's strict pydantic balance model.
        client.get_balance.assert_not_called()
        client.get_balance_without_preload_content.assert_called_once()

    def test_legacy_sandbox_payload(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"balance": 100000})
        )
        assert verify_auth(client) == 100000

    def test_non_2xx_raises_api_exception(self):
        # fetch_json_page must convert non-2xx statuses into ApiException so
        # bad credentials still fail loudly rather than parsing an error body.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"error": "unauthorized"}, status=401)
        )
        with pytest.raises(ApiException):
            verify_auth(client)
