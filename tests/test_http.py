"""
File: test_http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting._http — the shared retry wrapper, the
    raw-response JSON fetcher, and the signed arbitrary-method request helper.
    Covers the retry classification rules that every market-data call depends
    on (HTTP 429/5xx back off, transient transport failures back off, everything
    else fails fast) plus signed_request_json's URL/header/signature contract
    and its deliberate absence of internal retries.

Dependencies:
    Imports api_call_with_retry, fetch_json_page, signed_request_json, and
    _is_transient_network_error from kalshi_betting._http. Uses unittest.mock
    to stand in for SDK calls — no network access.

Notes:
    time.sleep is patched out in every retry test so the suite doesn't actually
    wait the 2s/4s/8s backoff schedule.
"""
import json
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from kalshi_betting import _http
from kalshi_betting._http import (
    _is_transient_network_error,
    api_call_with_retry,
    fetch_json_page,
    signed_request_json,
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


class _RecordingAuth:
    """Stand-in for KalshiAuth that records how create_auth_headers was called."""

    def __init__(self):
        self.calls: list[tuple] = []

    def create_auth_headers(self, *args, **kwargs) -> dict:
        self.calls.append((args, kwargs))
        return {"KALSHI-SIG": "x"}


class TestSignedRequestJson:
    """URL building, signing scope, headers, and no-retry contract."""

    _HOST = "https://demo-api.kalshi.co/trade-api/v2"
    _PATH = "/trade-api/v2/portfolio/events/orders"

    @staticmethod
    def _client(host: str, response=None, request_side_effect=None) -> SimpleNamespace:
        """Build a fake KalshiClient exposing configuration/kalshi_auth/rest_client."""
        rest_client = MagicMock()
        if request_side_effect is not None:
            rest_client.request.side_effect = request_side_effect
        else:
            rest_client.request.return_value = response
        return SimpleNamespace(
            configuration=SimpleNamespace(host=host),
            kalshi_auth=_RecordingAuth(),
            rest_client=rest_client,
        )

    @staticmethod
    def _response(status: int, payload: dict) -> SimpleNamespace:
        # reason/getheaders are what ApiException.from_response reads off a real
        # RESTResponse on the non-2xx path, so the stand-in must carry them too.
        return SimpleNamespace(
            status=status,
            data=json.dumps(payload).encode("utf-8"),
            reason="",
            getheaders=lambda: {},
        )

    def test_auth_headers_signed_over_method_and_path_only(self):
        # Kalshi signs timestamp + METHOD + PATH; the body must never reach the
        # signer, or a body change would invalidate an otherwise-valid signature.
        client = self._client(self._HOST, self._response(201, {"order": {"id": "a"}}))
        body = {"count": 3, "ticker": "KX-1"}
        signed_request_json(client, "POST", self._PATH, body=body)

        assert len(client.kalshi_auth.calls) == 1
        args, kwargs = client.kalshi_auth.calls[0]
        assert args == ("POST", self._PATH)
        assert kwargs == {}
        # The body dict is not passed to the signer in any form.
        assert body not in args and body not in kwargs.values()

        _, sent_kwargs = client.rest_client.request.call_args
        assert sent_kwargs["headers"]["KALSHI-SIG"] == "x"

    def test_body_and_content_type_passed_to_rest_client(self):
        client = self._client(self._HOST, self._response(200, {"ok": True}))
        body = {"action": "buy", "count": 2}
        signed_request_json(client, "POST", self._PATH, body=body)

        _, kwargs = client.rest_client.request.call_args
        # The SDK json.dumps-serializes a dict body itself — pass it unchanged.
        assert kwargs["body"] is body
        assert kwargs["headers"]["Content-Type"] == "application/json"

    def test_no_content_type_when_body_is_none(self):
        client = self._client(self._HOST, self._response(200, {"ok": True}))
        signed_request_json(client, "GET", self._PATH)

        _, kwargs = client.rest_client.request.call_args
        assert kwargs["body"] is None
        assert not [k for k in kwargs["headers"] if k.lower() == "content-type"]

    def test_query_params_appended_to_url_not_signature(self):
        client = self._client(self._HOST, self._response(200, {"ok": True}))
        signed_request_json(client, "GET", self._PATH, query={"a": 1, "b": None})

        args, _ = client.rest_client.request.call_args
        assert args[1].endswith("?a=1")          # None-valued params are omitted
        # Query strings are excluded from Kalshi's signature — bare path only.
        assert client.kalshi_auth.calls[0][0] == ("GET", self._PATH)

    def test_non_2xx_raises_apiexception(self):
        # Same contract as fetch_json_page: the raw transport doesn't raise, so
        # the helper must, or an order rejection would parse as a success body.
        client = self._client(self._HOST, self._response(400, {"error": "bad request"}))
        with pytest.raises(ApiException):
            signed_request_json(client, "POST", self._PATH, body={"count": 1})

    def test_2xx_returns_parsed_json(self):
        client = self._client(self._HOST, self._response(201, {"order": {"status": "executed"}}))
        result = signed_request_json(client, "POST", self._PATH, body={"count": 1})
        assert result == {"order": {"status": "executed"}}

    def test_exactly_one_transport_call_no_internal_retry(self):
        # A retried fill-or-kill order leg could double-fill; the helper must
        # never retry on its own (see the CLAUDE.md order-submission rule).
        client = self._client(self._HOST, request_side_effect=ConnectionResetError("reset"))
        with pytest.raises(ConnectionResetError):
            signed_request_json(client, "POST", self._PATH, body={"count": 1})
        assert client.rest_client.request.call_count == 1

    def test_host_taken_from_client_configuration(self):
        client = self._client(self._HOST, self._response(200, {"ok": True}))
        signed_request_json(client, "GET", self._PATH)

        url = client.rest_client.request.call_args[0][1]
        assert url.startswith("https://demo-api.kalshi.co")
        # configuration.host already carries /trade-api/v2 and so does `path`;
        # only scheme+netloc are taken from the host, so it appears exactly once.
        assert url.count("/trade-api/v2") == 1
        assert url == f"https://demo-api.kalshi.co{self._PATH}"
