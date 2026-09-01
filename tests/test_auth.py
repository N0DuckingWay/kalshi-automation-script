"""
File: test_auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting.auth — client construction (dev-key
    fallback semantics) and verify_auth's raw-response balance parsing.
    auth.py previously had zero coverage; this file exercises the BS-19
    (eager .get() default crashed dev-only setups) and BS-10 (verify_auth's
    modeled get_balance() call raises pydantic ValidationError on live 2026-07+
    responses) fixes.

Dependencies:
    Imports build_client and verify_auth from kalshi_betting.auth. Patches
    kalshi_betting.auth.SECRETS_FILE / PEM_FILE / DEV_PEM_FILE (module-level
    names, imported directly from config.py) with tmp_path fixture files, and
    patches kalshi_betting.auth.KalshiClient with a MagicMock so no real
    client is constructed. No network access.

Notes:
    build_client() monkey-patches cfg.api_key_id / cfg.private_key_pem onto a
    real Configuration object (see the CLAUDE.md "KalshiClient monkey-patch
    pattern" gotcha) — that pattern is intentional and must NOT be "fixed";
    these tests patch KalshiClient itself instead, so Configuration is still
    built for real (proving cfg.api_key_id/cfg.private_key_pem get set) but
    never handed to a live client.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from kalshi_betting import _http, auth


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


class TestVerifyAuth:
    """BS-10: verify_auth must use the raw response variant, not the modeled
    get_balance() (whose pydantic model no longer matches live responses)."""

    def test_parses_balance_field(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"balance": 25677})
        )
        with patch.object(_http.time, "sleep"):
            assert auth.verify_auth(client) == 25677

    def test_parses_balance_fp_fallback(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"balance_fp": "25677"})
        )
        with patch.object(_http.time, "sleep"):
            assert auth.verify_auth(client) == 25677

    def test_parses_balance_dollars_fallback(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"balance_dollars": "256.77"})
        )
        with patch.object(_http.time, "sleep"):
            assert auth.verify_auth(client) == 25677

    def test_balance_dollars_rounds_not_truncates(self):
        # float("2.03") * 100 == 202.99999999999997; int() would truncate that
        # to 202, silently losing a cent. verify_auth must round.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"balance_dollars": "2.03"})
        )
        with patch.object(_http.time, "sleep"):
            assert auth.verify_auth(client) == 203

    def test_retries_429_then_succeeds(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            side_effect=[
                balance_resp(429, {"error": "slow down"}),
                balance_resp(200, {"balance": 100000}),
            ]
        )
        with patch.object(_http.time, "sleep") as sleep:
            assert auth.verify_auth(client) == 100000
        assert client.get_balance_without_preload_content.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_unknown_shape_raises_keyerror(self):
        # No recognizable balance field at all — must fail loudly rather than
        # silently returning None-derived garbage (e.g. int(float(None))).
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"totally_unexpected_field": 1})
        )
        with patch.object(_http.time, "sleep"):
            with pytest.raises(KeyError):
                auth.verify_auth(client)
