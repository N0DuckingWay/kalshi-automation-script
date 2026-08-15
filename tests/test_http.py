"""
File: test_http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting._http — the shared retry wrapper,
    the raw-response JSON fetcher, and the generic signed raw HTTP request
    builder. Covers the retry classification rules that every market-data
    call depends on: HTTP 429/5xx back off, transient transport failures
    (connection reset, truncated body, read timeout) back off, and everything
    else fails fast. Also covers signed_raw_request's URL/header/body
    construction — the mechanism the upcoming V2 order-submission migration
    depends on to send fields (e.g. exchange_index) the pinned SDK's pydantic
    request models would silently drop.

Dependencies:
    Imports api_call_with_retry, fetch_json_page, signed_raw_request, and
    _is_transient_network_error from kalshi_betting._http, and historical
    (to test _signed_raw_get's delegation onto signed_raw_request).
    Uses unittest.mock to stand in for SDK calls — no network access.

Notes:
    time.sleep is patched out in every retry test so the suite doesn't actually
    wait the 2s/4s/8s backoff schedule.
"""
from http.client import IncompleteRead
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from kalshi_betting import _http, historical
from kalshi_betting._http import (
    _is_transient_network_error,
    api_call_with_retry,
    fetch_json_page,
    signed_raw_request,
)


class _StatusError(Exception):
    """Minimal stand-in for an SDK exception carrying an HTTP status."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class TestTransientNetworkDetection:
    """_is_transient_network_error's classification of transport failures."""

    def test_direct_protocol_error(self):
        assert _is_transient_network_error(ProtocolError("Connection broken"))

    def test_wrapped_incomplete_read(self):
        # The exact shape observed live: urllib3 raises ProtocolError *from*
        # an http.client.IncompleteRead during resp.read().
        inner = IncompleteRead(partial=b"", expected=100)
        outer = ProtocolError(f"Connection broken: {inner!r}", inner)
        outer.__cause__ = inner
        assert _is_transient_network_error(outer)

    def test_builtin_connection_and_timeout_errors(self):
        assert _is_transient_network_error(ConnectionResetError("reset by peer"))
        assert _is_transient_network_error(TimeoutError("timed out"))
        assert _is_transient_network_error(ReadTimeoutError(None, "url", "read timeout"))

    def test_cause_chain_is_walked(self):
        root = ConnectionResetError("reset by peer")
        middle = OSError("io failed")
        middle.__cause__ = root
        top = RuntimeError("wrapped twice")
        top.__cause__ = middle
        assert _is_transient_network_error(top)

    def test_unrelated_exception_is_not_transient(self):
        assert not _is_transient_network_error(ValueError("bad payload"))
        assert not _is_transient_network_error(KeyError("missing"))

    def test_self_referential_chain_terminates(self):
        # A cycle in the cause chain must not hang the classifier.
        exc = RuntimeError("loop")
        exc.__cause__ = exc
        assert not _is_transient_network_error(exc)


class TestApiCallWithRetry:
    """Retry/backoff behavior of the shared wrapper."""

    def test_returns_immediately_on_success(self):
        fn = MagicMock(return_value={"ok": True})
        assert api_call_with_retry(fn, "arg", key="value") == {"ok": True}
        fn.assert_called_once_with("arg", key="value")

    def test_retries_429_then_succeeds(self):
        fn = MagicMock(side_effect=[_StatusError(429), {"ok": True}])
        with patch.object(_http.time, "sleep") as sleep:
            assert api_call_with_retry(fn) == {"ok": True}
        assert fn.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_retries_5xx_then_succeeds(self):
        fn = MagicMock(side_effect=[_StatusError(503), _StatusError(500), {"ok": True}])
        with patch.object(_http.time, "sleep") as sleep:
            assert api_call_with_retry(fn) == {"ok": True}
        assert fn.call_count == 3
        # Backoff doubles between attempts.
        assert [c.args[0] for c in sleep.call_args_list] == [2.0, 4.0]

    def test_retries_protocol_error_then_succeeds(self):
        # Regression test for the crash that killed a multi-hour backtest fetch:
        # a status-less ProtocolError used to be re-raised on first occurrence.
        inner = IncompleteRead(partial=b"", expected=100)
        broken = ProtocolError(f"Connection broken: {inner!r}", inner)
        broken.__cause__ = inner
        fn = MagicMock(side_effect=[broken, {"ok": True}])
        with patch.object(_http.time, "sleep") as sleep:
            assert api_call_with_retry(fn) == {"ok": True}
        assert fn.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_retries_connection_reset_then_succeeds(self):
        fn = MagicMock(side_effect=[ConnectionResetError("reset by peer"), {"ok": True}])
        with patch.object(_http.time, "sleep"):
            assert api_call_with_retry(fn) == {"ok": True}
        assert fn.call_count == 2

    def test_non_retryable_status_fails_fast(self):
        fn = MagicMock(side_effect=_StatusError(400))
        with patch.object(_http.time, "sleep") as sleep:
            with pytest.raises(_StatusError):
                api_call_with_retry(fn)
        fn.assert_called_once()
        sleep.assert_not_called()

    def test_non_network_exception_fails_fast(self):
        fn = MagicMock(side_effect=ValueError("bad payload"))
        with patch.object(_http.time, "sleep") as sleep:
            with pytest.raises(ValueError):
                api_call_with_retry(fn)
        fn.assert_called_once()
        sleep.assert_not_called()

    def test_exhausts_attempts_and_reraises_transient_error(self):
        fn = MagicMock(side_effect=ProtocolError("Connection broken"))
        with patch.object(_http.time, "sleep"):
            with pytest.raises(ProtocolError):
                api_call_with_retry(fn)
        assert fn.call_count == _http._MAX_ATTEMPTS


class TestFetchJsonPage:
    """Raw-response status checking and JSON parsing."""

    @staticmethod
    def _response(status: int, body: bytes) -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.data = body
        return resp

    def test_parses_2xx_body(self):
        fetch_fn = MagicMock(return_value=self._response(200, b'{"markets": [1, 2]}'))
        assert fetch_json_page(fetch_fn, limit=10) == {"markets": [1, 2]}
        fetch_fn.assert_called_once_with(limit=10)

    def test_reads_body_when_data_unset(self):
        resp = MagicMock()
        resp.status = 200
        resp.data = None
        resp.read.return_value = b'{"ok": true}'
        assert fetch_json_page(MagicMock(return_value=resp)) == {"ok": True}
        resp.read.assert_called_once()

    def test_non_2xx_raises_api_exception(self):
        # The raw SDK variants don't raise on 4xx/5xx; fetch_json_page restores
        # that so api_call_with_retry can still see a retryable status.
        fetch_fn = MagicMock(return_value=self._response(429, b'{"error": "slow down"}'))
        with pytest.raises(ApiException):
            fetch_json_page(fetch_fn)


def _mock_client(host: str = "https://api.elections.kalshi.com/trade-api/v2") -> MagicMock:
    """
    Minimal KalshiClient stand-in exposing only the two SDK primitives
    signed_raw_request touches: client.configuration.host (for host
    derivation) and client.kalshi_auth / client.rest_client (for signing and
    transport). No generated endpoint method is ever called.
    """
    client = MagicMock()
    client.configuration.host = host
    client.kalshi_auth.create_auth_headers = MagicMock(
        return_value={"KALSHI-ACCESS-SIGNATURE": "sig", "KALSHI-ACCESS-TIMESTAMP": "123"}
    )
    client.rest_client.request = MagicMock(return_value=MagicMock(status=200, data=b"{}"))
    return client


class TestSignedRawRequest:
    """URL/header/body construction for the generic signed raw HTTP helper."""

    def test_get_with_params_builds_query_and_signs_bare_path(self):
        client = _mock_client()
        path = "/trade-api/v2/historical/markets"
        signed_raw_request(client, "GET", path, params={"limit": 1000, "cursor": None})

        # Query string carries only the non-None param; None-valued params
        # are omitted entirely (same convention the old _signed_raw_get had).
        called_url = client.rest_client.request.call_args.args[1]
        assert called_url == f"https://api.elections.kalshi.com{path}?limit=1000"

        # Kalshi's signature covers only timestamp+METHOD+path — the query
        # string must NOT be part of what gets signed.
        client.kalshi_auth.create_auth_headers.assert_called_once_with("GET", path)

    def test_post_with_body_sets_content_type_and_forwards_dict_verbatim(self):
        client = _mock_client()
        path = "/trade-api/v2/portfolio/events/orders"
        body = {"ticker": "ABC", "side": "yes", "exchange_index": 1}
        signed_raw_request(client, "POST", path, body=body)

        call = client.rest_client.request.call_args
        assert call.args[0] == "POST"
        assert call.args[1] == f"https://api.elections.kalshi.com{path}"
        assert call.kwargs["headers"]["Content-Type"] == "application/json"
        # The exact dict is passed through — this is the mechanism that lets
        # a field like exchange_index (which a pydantic request model would
        # silently drop) actually reach the wire.
        assert call.kwargs["body"] is body
        assert call.kwargs["body"]["exchange_index"] == 1

    def test_sandbox_host_produces_sandbox_origin_url(self):
        client = _mock_client(host="https://demo-api.kalshi.co/trade-api/v2")
        path = "/trade-api/v2/portfolio/balance"
        signed_raw_request(client, "GET", path)

        called_url = client.rest_client.request.call_args.args[1]
        assert called_url.startswith("https://demo-api.kalshi.co")
        assert "api.elections.kalshi.com" not in called_url

    def test_no_params_no_body_produces_bare_url_no_content_type(self):
        client = _mock_client()
        path = "/trade-api/v2/historical/cutoff"
        signed_raw_request(client, "GET", path)

        call = client.rest_client.request.call_args
        called_url = call.args[1]
        assert called_url == f"https://api.elections.kalshi.com{path}"
        assert "?" not in called_url
        assert "Content-Type" not in call.kwargs["headers"]
        # No body kwarg at all when body is None — matches the pre-existing
        # GET-only call shape (_signed_raw_get never passed body=).
        assert "body" not in call.kwargs

    def test_empty_params_dict_produces_bare_url(self):
        client = _mock_client()
        path = "/trade-api/v2/historical/markets"
        signed_raw_request(client, "GET", path, params={})

        called_url = client.rest_client.request.call_args.args[1]
        assert called_url == f"https://api.elections.kalshi.com{path}"


class TestSignedRawGetDelegation:
    """historical._signed_raw_get must forward onto the generalized helper."""

    def test_delegates_to_signed_raw_request(self, monkeypatch):
        mock = MagicMock(return_value=MagicMock(status=200, data=b"{}"))
        monkeypatch.setattr(historical, "signed_raw_request", mock)

        client = MagicMock()
        path = "/trade-api/v2/historical/markets"
        historical._signed_raw_get(client, path, limit=1000, cursor="abc")

        mock.assert_called_once_with(client, "GET", path, params={"limit": 1000, "cursor": "abc"})
