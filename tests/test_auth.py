"""
File: test_auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting.auth — client construction (dev-key
    fallback semantics) and the shard-aware balance parsing added when Kalshi
    scoped GET /portfolio/balance per exchange shard (2026-08-13). Covers the
    BS-19 fix (an eager .get() default crashed dev-only setups), the BS-10 fix
    (verify_auth's modeled get_balance() raises pydantic ValidationError on
    live 2026-07+ responses, so it reads the raw body instead), the
    dollar-string -> floored-cents converter, the tiered fallback chain in
    _balance_cents_by_shard (full breakdown -> aggregate balance_dollars ->
    legacy integer cents), and verify_auth end-to-end against a faked raw
    response.

Dependencies:
    Imports build_client, verify_auth, _balance_cents_by_shard and
    _dollar_str_to_cents from kalshi_betting.auth, and DEFAULT_EXCHANGE_INDEX
    from kalshi_betting.config. Patches kalshi_betting.auth.SECRETS_FILE /
    PEM_FILE / DEV_PEM_FILE (module-level names, imported directly from
    config.py) with tmp_path fixture files, and patches
    kalshi_betting.auth.KalshiClient with a MagicMock so no real client is
    constructed. Uses unittest.mock / SimpleNamespace to stand in for the SDK
    client and its RESTResponse — no network access.

Notes:
    build_client() monkey-patches cfg.api_key_id / cfg.private_key_pem onto a
    real Configuration object (see the CLAUDE.md "KalshiClient monkey-patch
    pattern" gotcha) — that pattern is intentional and must NOT be "fixed";
    these tests patch KalshiClient itself instead, so Configuration is still
    built for real (proving cfg.api_key_id/cfg.private_key_pem get set) but
    never handed to a live client.

    The same-key-different-units trap is asserted explicitly: inside a
    balance_breakdown entry "balance" is a fixed-point DOLLAR STRING, while the
    TOP-LEVEL "balance" is legacy INTEGER CENTS. A test pins each.

    verify_auth() returns the FULL per-shard dict, not a single scalar — the
    old "shard-0 preferred over aggregate" behavior inverts here: shard-1
    funds are now visible in the result, not excluded from it. Sizing is
    portfolio-wide and lives in main.py (sum(shard_balances.values())), which
    is exercised in test_main.py, not here.
"""
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import _http, auth
from kalshi_betting.auth import (
    _balance_cents_by_shard,
    _dollar_str_to_cents,
    verify_auth,
)
from kalshi_betting.config import DEFAULT_EXCHANGE_INDEX


def _write_secrets(tmp_path, payload: dict):
    """
    Write a secrets.json-shaped fixture file and point auth.SECRETS_FILE at it.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        payload (dict): Contents to serialize as the secrets file.

    Returns:
        pathlib.Path: Path to the written file.
    """
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps(payload))
    return path


def balance_resp(status: int, body: dict) -> MagicMock:
    """
    Build a fake raw get_balance_without_preload_content response.

    Mirrors the .status / .data pattern used by fetch_json_page and by the
    fake responses in tests/test_http.py — the raw SDK variants return an
    object with a .status int and a .data bytes body rather than a modeled
    pydantic object.

    Args:
        status (int): HTTP status code.
        body (dict): JSON-serializable response body.

    Returns:
        MagicMock: Object with .status and .data set.
    """
    resp = MagicMock()
    resp.status = status
    resp.data = json.dumps(body).encode("utf-8")
    return resp


class TestBuildClientDevKeyFallback:
    """BS-19: dev_api_key is optional and must not be evaluated eagerly."""

    def test_dev_only_secrets_builds_client(self, tmp_path, monkeypatch):
        # Only dev_api_key present — the old eager .get() default evaluated
        # secrets["Kalshi-api-key"] regardless, raising KeyError even though
        # dev_api_key existed. This must now succeed.
        _write_secrets(tmp_path, {"dev_api_key": "dev-key-123"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")
        dev_pem = tmp_path / "kalshi_demo_private_key.pem"  # deliberately absent

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", dev_pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("dev")

        # cfg is the sole positional/keyword arg passed to KalshiClient(...)
        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "dev-key-123"
        assert cfg.private_key_pem == "prod-pem"  # falls back to PEM_FILE (no dev PEM)

    def test_missing_dev_key_falls_back_to_prod_key(self, tmp_path, monkeypatch):
        # No dev_api_key at all — dev mode must fall back to Kalshi-api-key
        # rather than raising.
        _write_secrets(tmp_path, {"Kalshi-api-key": "prod-key-456"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")
        dev_pem = tmp_path / "kalshi_demo_private_key.pem"

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", dev_pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("dev")

        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "prod-key-456"

    def test_neither_key_raises_keyerror(self, tmp_path, monkeypatch):
        _write_secrets(tmp_path, {"some_other_field": "irrelevant"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", tmp_path / "no_such_dev.pem")

        with pytest.raises(KeyError):
            auth.build_client("dev")

    def test_prod_mode_requires_kalshi_api_key(self, tmp_path, monkeypatch):
        # dev_api_key present is irrelevant in prod mode — prod always requires
        # Kalshi-api-key specifically.
        _write_secrets(tmp_path, {"dev_api_key": "dev-key-123"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)

        with pytest.raises(KeyError):
            auth.build_client("prod")

    def test_prod_mode_builds_client_with_kalshi_api_key(self, tmp_path, monkeypatch):
        _write_secrets(tmp_path, {"Kalshi-api-key": "prod-key-789"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem-text")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("prod")

        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "prod-key-789"
        assert cfg.private_key_pem == "prod-pem-text"


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


class TestBalanceCentsByShard:
    def test_live_payload_returns_all_shards(self):
        # Every shard is now visible — shard 1's $0.00 is included, not
        # silently excluded the way the old shard-0-only lookup would have.
        assert _balance_cents_by_shard(LIVE_PAYLOAD) == {0: 114, 1: 0}

    def test_multi_shard_both_nonzero(self):
        payload = {
            "balance_breakdown": [
                {"balance": "5.00", "exchange_index": 0},
                {"balance": "100.00", "exchange_index": 1},
            ],
            "balance_dollars": "105.00",
            "balance": 10500,
        }
        # Both shards' funds are visible — unlike the pre-multi-shard reader,
        # shard-1 collateral is no longer dropped from the result.
        assert _balance_cents_by_shard(payload) == {0: 500, 1: 10000}

    def test_no_breakdown_falls_back_to_balance_dollars(self):
        assert _balance_cents_by_shard({"balance_dollars": "42.50"}) == {
            DEFAULT_EXCHANGE_INDEX: 4250
        }

    def test_legacy_integer_payload_is_already_cents(self):
        # Sandbox shape. 5000 means $50.00 — running it through the dollar
        # converter would report $5,000.00.
        assert _balance_cents_by_shard({"balance": 5000}) == {
            DEFAULT_EXCHANGE_INDEX: 5000
        }

    def test_breakdown_with_no_parseable_entries_warns_and_uses_aggregate(self, caplog):
        payload = {
            "balance_breakdown": [{"exchange_index": "not-an-int", "balance": "100.00"}],
            "balance_dollars": "100.00",
        }
        with caplog.at_level(logging.WARNING):
            assert _balance_cents_by_shard(payload) == {DEFAULT_EXCHANGE_INDEX: 10000}
        assert any(
            "balance_breakdown" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_malformed_entries_do_not_break_good_entries(self):
        payload = {
            "balance_breakdown": [
                "not-a-dict",
                None,
                {"exchange_index": None, "balance": "9.99"},
                {"exchange_index": "abc", "balance": "8.88"},
                {"balance": "7.77"},
                {"balance": "3.50", "exchange_index": DEFAULT_EXCHANGE_INDEX},
                {"balance": "6.25", "exchange_index": 1},
            ],
            "balance_dollars": "999.00",
        }
        # Only the two well-formed entries survive; the malformed ones are
        # skipped without aborting the scan of the good ones.
        assert _balance_cents_by_shard(payload) == {
            DEFAULT_EXCHANGE_INDEX: 350,
            1: 625,
        }

    def test_unparseable_entry_balance_is_skipped(self):
        # The only entry present is unparseable, so the breakdown yields
        # nothing and falls all the way through to the aggregate.
        payload = {
            "balance_breakdown": [{"balance": "garbage", "exchange_index": 0}],
            "balance_dollars": "12.00",
        }
        assert _balance_cents_by_shard(payload) == {DEFAULT_EXCHANGE_INDEX: 1200}

    def test_one_unparseable_entry_among_others_is_just_dropped(self):
        # A mix of one bad entry and one good entry: the good entry alone
        # is returned — no fallback to the aggregate, since something parsed.
        payload = {
            "balance_breakdown": [
                {"balance": "garbage", "exchange_index": 0},
                {"balance": "2.00", "exchange_index": 1},
            ],
            "balance_dollars": "999.00",
        }
        assert _balance_cents_by_shard(payload) == {1: 200}

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_by_shard({})

    def test_nonsense_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_by_shard({"nonsense": 1})

    def test_non_int_legacy_balance_raises(self):
        # A bool is an int subclass but a string is not — a stringy top-level
        # balance is not silently treated as cents.
        with pytest.raises(ValueError):
            _balance_cents_by_shard({"balance": "114"})


class TestVerifyAuth:
    def test_returns_full_shard_dict_from_raw_response(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response(LIVE_PAYLOAD)
        )

        result = verify_auth(client)

        assert result == {0: 114, 1: 0}
        assert isinstance(result, dict)
        # The modeled call must not be used — it deserializes through the
        # pinned SDK's strict pydantic balance model.
        client.get_balance.assert_not_called()
        client.get_balance_without_preload_content.assert_called_once()

    def test_log_line_contains_per_shard_breakdown_and_total(self, caplog):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response(LIVE_PAYLOAD)
        )
        with caplog.at_level(logging.INFO):
            verify_auth(client)
        assert any(
            "0: 114" in rec.message and "1: 0" in rec.message and "1.14" in rec.message
            for rec in caplog.records
        )

    def test_legacy_sandbox_payload(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"balance": 100000})
        )
        assert verify_auth(client) == {DEFAULT_EXCHANGE_INDEX: 100000}

    def test_non_2xx_raises_api_exception(self):
        # fetch_json_page must convert non-2xx statuses into ApiException so
        # bad credentials still fail loudly rather than parsing an error body.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"error": "unauthorized"}, status=401)
        )
        with pytest.raises(ApiException):
            verify_auth(client)


class TestUnparseableEntryBalanceWarns:
    def test_dropped_shard_funds_are_never_invisible(self, caplog):
        # Regression (adversarial review): an entry with a parseable index but
        # an unparseable balance silently vanished that shard's real funds
        # from sizing, the coverage audit, and the transfer planner. The drop
        # is still the safe behavior (under-sizing), but it must be LOUD.
        import logging as _logging
        payload = {
            "balance_breakdown": [
                {"exchange_index": 0, "balance": "10.0000"},
                {"exchange_index": 2, "amount_dollars": "500.0000"},  # drifted key
            ]
        }
        with caplog.at_level(_logging.WARNING):
            out = _balance_cents_by_shard(payload)
        assert out == {0: 1000}
        assert "shard 2" in caplog.text
        assert "NOT counted" in caplog.text


class TestVerifyAuthRetryAndDrift:
    """verify_auth's transport-level behaviour, distinct from the payload-shape
    coverage in TestVerifyAuth above: the api_call_with_retry wrapper it goes
    through, and the loud failure required of an unrecognized response shape.

    Both cases assert the SHARD DICT return type — verify_auth is no longer
    scalar-returning, so a 429 that recovers must yield {shard: cents}, not an
    int."""

    def test_retries_429_then_succeeds(self):
        # A transient 429 on a read-only GET must be retried, not fatal —
        # verify_auth is wrapped in api_call_with_retry (unlike order
        # submission and read_shard_balances, both deliberately single-shot).
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            side_effect=[
                balance_resp(429, {"error": "slow down"}),
                balance_resp(200, {"balance": 100000}),
            ]
        )
        with patch.object(_http.time, "sleep") as sleep:
            assert auth.verify_auth(client) == {DEFAULT_EXCHANGE_INDEX: 100000}
        assert client.get_balance_without_preload_content.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_unknown_shape_raises(self):
        # No recognizable balance field at all — must fail loudly rather than
        # silently returning a wrong number. _balance_cents_by_shard raises
        # ValueError once every fallback tier has been exhausted.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"totally_unexpected_field": 1})
        )
        with patch.object(_http.time, "sleep"):
            with pytest.raises(ValueError):
                auth.verify_auth(client)
