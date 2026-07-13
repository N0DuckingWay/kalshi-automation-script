"""
File: _http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Shared HTTP helpers for Kalshi SDK calls: a retry wrapper (429/5xx with
    exponential backoff) and a raw-response JSON fetcher for the SDK's
    `*_without_preload_content` endpoint variants. Both the live scanner and
    the historical fetch pipeline import from here so backoff behavior stays
    consistent and there is no scanner → historical reverse import.

Dependencies:
    No project imports — this module is a leaf so scanner.py, trader.py, and
    historical.py can import from it without introducing cycles. Imports
    ApiException from the kalshi_python_sync SDK to re-raise raw-response
    HTTP errors with the same exception types the modeled calls used.

Notes:
    The Kalshi SDK's ApiException exposes .status; requests-based errors expose
    .response.status_code. We look for either. Anything else is treated as
    non-retryable and re-raised immediately.

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
from typing import Any

from kalshi_python_sync.exceptions import ApiException

# HTTP status codes worth retrying: 429 (rate limit) plus common transient 5xx errors.
# 500 / 502 / 503 / 504 sometimes appear during Kalshi maintenance or upstream blips.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

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
    Call a Kalshi SDK function with exponential backoff on retryable HTTP errors.

    Retryable = HTTP 429 or 5xx as reported by the exception. Any other exception
    is re-raised on the first occurrence. On the final attempt, even a retryable
    error is re-raised so callers see a real failure rather than an infinite loop.

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
            retryable = status in _RETRYABLE_STATUS
            last_attempt = attempt >= _MAX_ATTEMPTS - 1
            if not retryable or last_attempt:
                raise
            logging.warning(
                "Retryable API error (status=%s) — sleeping %.0fs (attempt %d/%d)",
                status, delay, attempt + 1, _MAX_ATTEMPTS - 1,
            )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY)
