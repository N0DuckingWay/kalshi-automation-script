"""
File: _http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Shared HTTP helpers for Kalshi SDK calls: a retry wrapper (429/5xx plus
    transient connection failures, with exponential backoff) and a raw-response
    JSON fetcher for the SDK's `*_without_preload_content` endpoint variants.
    Both the live scanner and the historical fetch pipeline import from here so
    backoff behavior stays consistent and there is no scanner → historical
    reverse import.

Dependencies:
    No project imports — this module is a leaf so scanner.py, trader.py, and
    historical.py can import from it without introducing cycles. Imports
    ApiException from the kalshi_python_sync SDK to re-raise raw-response
    HTTP errors with the same exception types the modeled calls used, and
    (optionally) urllib3's exception types to recognize transport-level drops.

Notes:
    The Kalshi SDK's ApiException exposes .status; requests-based errors expose
    .response.status_code. We look for either. A status-less exception is
    retryable only if it is a recognized transient connection failure (see
    _TRANSIENT_NETWORK_ERRORS); anything else is re-raised immediately.

    fetch_json_page() exists because of 2026-07 API drift: several endpoints
    stopped sending the legacy integer-cent fields the pinned SDK's response
    models type as required, so modeled calls raise pydantic ValidationError.
    Raw-response variants bypass the models — but they also skip the SDK's
    status check, which fetch_json_page restores.
"""
import json
import logging
import time
from collections.abc import Callable
from http.client import IncompleteRead
from typing import Any

from kalshi_python_sync.exceptions import ApiException

# HTTP status codes worth retrying: 429 (rate limit) plus common transient 5xx errors.
# 500 / 502 / 503 / 504 sometimes appear during Kalshi maintenance or upstream blips.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Transport-level failures that carry no HTTP status but are just as transient
# as a 503 — the connection died before or during the response body. Observed
# live on 2026-08-03: a multi-hour backtest fetch was killed outright by
# `urllib3.exceptions.ProtocolError: Connection broken: IncompleteRead` raised
# from resp.read() inside fetch_json_page, because the retry wrapper only knew
# how to recognize status-carrying errors. Retrying is safe here: every caller
# of api_call_with_retry is a read-only market-data GET (order submission
# deliberately bypasses this wrapper — see trader._submit_order).
_urllib3_transient: tuple[type[BaseException], ...]
try:
    # urllib3 ships as a dependency of the Kalshi SDK's rest client, but guard
    # the import so this leaf module stays importable if that ever changes.
    from urllib3.exceptions import ProtocolError, ReadTimeoutError

    _urllib3_transient = (ProtocolError, ReadTimeoutError)
except ImportError:  # pragma: no cover - urllib3 is always present in practice
    _urllib3_transient = ()

_TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    *_urllib3_transient,
    IncompleteRead,     # truncated response body
    ConnectionError,    # builtin: reset / aborted / refused
    TimeoutError,       # builtin: socket timeout (socket.timeout is an alias)
)

# How far up an exception's __cause__/__context__ chain to look for a transient
# transport error. urllib3 wraps IncompleteRead in ProtocolError, and callers
# may wrap further; a shallow bounded walk catches those without risking a
# pathological loop on a deeply chained exception.
_CAUSE_CHAIN_DEPTH = 5

# Retry policy: 6 attempts total (1 initial + 5 retries), backoff doubling
# 2s/4s/8s/16s/32s between attempts. _MAX_DELAY is a defensive ceiling on the
# doubling, not a delay this policy actually reaches — with 5 retries the
# largest computed sleep is 32s; the cap only matters if _MAX_ATTEMPTS grows.
_MAX_ATTEMPTS = 6
_INITIAL_DELAY = 2.0
_MAX_DELAY = 60.0


def _extract_status(exc: BaseException) -> int | None:
    """
    Best-effort extraction of an HTTP status code from an exception.

    Kalshi's generated SDK raises ApiException with a .status attribute; libraries
    built on requests raise errors with .response.status_code. Returns None when
    no status can be found — the caller should treat that as non-retryable.

    Args:
        exc (BaseException): Exception raised by the API client.

    Returns:
        int | None: HTTP status code if discoverable, otherwise None.
    """
    for attr in ("status", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_transient_network_error(exc: BaseException) -> bool:
    """
    Report whether an exception is a transient transport-level failure.

    Checks the exception itself and up to _CAUSE_CHAIN_DEPTH levels of its
    __cause__/__context__ chain, because the transport error that actually
    matters is often wrapped (urllib3 raises ProtocolError *from* an
    http.client.IncompleteRead). These carry no HTTP status, so
    _extract_status() returns None for them and they would otherwise be
    treated as fatal.

    Args:
        exc (BaseException): Exception raised by the API client.

    Returns:
        bool: True if the exception (or a shallow cause of it) is one of
            _TRANSIENT_NETWORK_ERRORS and the call is worth retrying.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CAUSE_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return False
        if isinstance(current, _TRANSIENT_NETWORK_ERRORS):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def fetch_json_page(fetch_fn: Any, **kwargs) -> dict:
    """
    Call a `*_without_preload_content` SDK method and parse the JSON body.

    The raw-response variants bypass the SDK's response models (which 2026-07
    API drift broke — see module Notes) but they ALSO skip the SDK's status
    check and return 4xx/5xx bodies without raising. This helper restores the
    original error behavior by raising ApiException.from_response on non-2xx
    statuses, so api_call_with_retry keeps retrying 429/5xx exactly as it did
    for the modeled calls.

    Args:
        fetch_fn: A bound `*_without_preload_content` method on KalshiClient.
        **kwargs: Query parameters forwarded to the SDK method.

    Returns:
        dict: The parsed JSON response body.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
    """
    resp = fetch_fn(**kwargs)
    # RESTResponse.data may be unread until .read() is called, depending on
    # how the underlying urllib3 response was created
    body = getattr(resp, "data", None)
    if body is None and hasattr(resp, "read"):
        body = resp.read()
    if not 200 <= resp.status < 300:
        raise ApiException.from_response(
            http_resp=resp,
            body=body.decode("utf-8") if isinstance(body, bytes) else body,
            data=None,
        )
    return json.loads(body)


def api_call_with_retry(fn: Callable, *args, **kwargs):
    """
    Call a Kalshi SDK function with exponential backoff on retryable errors.

    Retryable = HTTP 429 or 5xx as reported by the exception, OR a transient
    transport-level failure with no status at all (connection reset, truncated
    body, read timeout — see _is_transient_network_error). Any other exception
    is re-raised on the first occurrence. On the final attempt, even a retryable
    error is re-raised so callers see a real failure rather than an infinite loop.

    Only read-only market-data calls go through this wrapper; order submission
    deliberately does not (retrying a fill-or-kill order could double-fill a
    leg), so retrying transport errors here cannot duplicate a trade.

    Args:
        fn (Callable): The API function to invoke (e.g. client.get_markets).
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Whatever fn returns on success.

    Raises:
        Exception: The original exception if it is not retryable, or if all
            attempts have been exhausted.
    """
    delay = _INITIAL_DELAY
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            status = _extract_status(exc)
            # A status-less exception may still be a transient connection drop;
            # those carry no HTTP code but are exactly as worth retrying as 503.
            transient = status is None and _is_transient_network_error(exc)
            retryable = status in _RETRYABLE_STATUS or transient
            last_attempt = attempt >= _MAX_ATTEMPTS - 1
            if not retryable or last_attempt:
                raise
            logging.warning(
                "Retryable API error (%s) — sleeping %.0fs (attempt %d/%d)",
                f"{type(exc).__name__}: {exc}" if transient else f"status={status}",
                delay, attempt + 1, _MAX_ATTEMPTS - 1,
            )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY)
