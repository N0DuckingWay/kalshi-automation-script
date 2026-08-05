"""
File: test_http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting._http — the shared retry wrapper and
    the raw-response JSON fetcher. Covers the retry classification rules that
    every market-data call depends on: HTTP 429/5xx back off, transient
    transport failures (connection reset, truncated body, read timeout) back
    off, and everything else fails fast.

Dependencies:
    Imports api_call_with_retry, fetch_json_page, and _is_transient_network_error
    from kalshi_betting._http. Uses unittest.mock to stand in for SDK calls —
    no network access.

Notes:
    time.sleep is patched out in every retry test so the suite doesn't actually
    wait the 2s/4s/8s backoff schedule.
"""
from http.client import IncompleteRead
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from kalshi_betting import _http
from kalshi_betting._http import (
    _is_transient_network_error,
    api_call_with_retry,
    fetch_json_page,
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
