"""
File: _http.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Shared HTTP helpers for Kalshi SDK calls: a retry wrapper (429/5xx plus
    transient connection failures, with exponential backoff), a raw-response
    JSON fetcher for the SDK's `*_without_preload_content` endpoint variants,
    and signed_request_json() — a signed arbitrary-method (GET/POST/...) raw
    request for API routes the pinned SDK has no generated method for. Both the
    live scanner and the historical fetch pipeline import from here so backoff
    behavior stays consistent and there is no scanner → historical reverse
    import.

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

    signed_request_json() generalizes that to routes with no SDK method at all —
    notably the V2 order endpoint /portfolio/events/orders, which trader.py now
    submits through by default. It shares fetch_json_page's status-check + parse
    tail via _check_and_parse, so the non-2xx → ApiException contract is
    single-sourced. It contains NO retry logic on purpose: order submission
    calls it directly and retry-free, because a retried fill-or-kill leg can
    double-fill (see trader._submit_order_v2 and the CLAUDE.md rule). Read-only
    callers wrap it in api_call_with_retry themselves.
"""
import json
import logging
import time
from collections.abc import Callable
from http.client import IncompleteRead
from typing import Any
from urllib.parse import urlencode, urlparse

from kalshi_python_sync.exceptions import ApiException

# Optional acceleration: the backtest's settled-market fetch parses tens of
# millions of JSON records and is CPU-bound on exactly this call. orjson is an
# optional extra (`pip install -e ".[perf]"`); the stdlib fallback is fully
# equivalent, just slower. Both accept bytes or str.
try:
    import orjson

    _json_loads: Callable[[Any], Any] = orjson.loads
except ImportError:
    _json_loads = json.loads

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
# deliberately bypasses this wrapper on both paths — see trader._submit_order
# and trader._submit_order_v2).
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


def _check_and_parse(resp: Any) -> dict:
    """
    Apply the status check and JSON parse shared by every raw-response call.

    The SDK's `*_without_preload_content` variants (and a hand-built
    rest_client.request call) return the RESTResponse untouched: no response
    model, and no status check either. This restores the original error
    behavior by raising ApiException.from_response on non-2xx, which is what
    lets api_call_with_retry keep seeing retryable 429/5xx statuses exactly as
    it did for the modeled calls.

    Args:
        resp (Any): A RESTResponse-like object exposing .status and either a
            .data attribute or a .read() method for the body bytes.

    Returns:
        dict: The parsed JSON response body.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
    """
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
    return _json_loads(body)


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
    # Shared with signed_request_json so the non-2xx → ApiException contract
    # has exactly one definition
    return _check_and_parse(resp)


def signed_request_json(
    client: Any,
    method: str,
    path: str,
    *,
    query: dict | None = None,
    body: dict | None = None,
) -> dict:
    """
    Perform a signed request of any HTTP method against an arbitrary API path.

    The pinned SDK has no generated method for several routes the bot needs
    (the /historical archive, and the V2 order endpoint
    /portfolio/events/orders this is groundwork for), and its modeled calls
    deserialize through response models that live API drift keeps breaking.
    This helper signs the request the way every SDK call is signed (KalshiAuth:
    RSA-PSS over timestamp + method + path — method-agnostic, query string
    stripped, body NOT part of the signature), executes it with the client's own
    rest client, and applies the shared status-check + JSON-parse contract.

    Contains NO retry logic by design. Order submission calls this directly:
    retrying a rejected fill-or-kill leg could submit it twice at different
    prices and leave an unhedged position, so retries are the caller's decision
    (read-only callers wrap it in api_call_with_retry).

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        method (str): HTTP method, e.g. "GET" or "POST". Case-insensitive; it
            is upper-cased before signing so the signature matches the request.
        path (str): FULL API path including the /trade-api/v2 prefix (e.g.
            "/trade-api/v2/portfolio/events/orders"). This is what gets signed,
            matching how the SDK itself signs.
        query (dict | None): Query parameters; None values are omitted. Not
            part of the signature.
        body (dict | None): JSON request body. When not None, a
            Content-Type: application/json header is sent and the SDK's rest
            client json.dumps-serializes the dict.

    Returns:
        dict: The parsed JSON response body.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
    """
    verb = method.upper()
    # configuration.host already includes the /trade-api/v2 prefix, and `path`
    # carries it too (it must, to be signed correctly) — so take only the
    # scheme+netloc from the host to avoid emitting the prefix twice.
    parsed_host = urlparse(client.configuration.host)
    encoded = urlencode({k: v for k, v in (query or {}).items() if v is not None})
    url = f"{parsed_host.scheme}://{parsed_host.netloc}{path}" + (f"?{encoded}" if encoded else "")

    headers = {"accept": "application/json"}
    if body is not None:
        # The SDK's rest client only json.dumps a dict body when the content
        # type is JSON (or absent); set it explicitly so intent is on the wire.
        headers["Content-Type"] = "application/json"
    # Query strings are excluded from the signature by Kalshi's auth scheme, and
    # so is the body — only timestamp + method + path are signed
    headers.update(client.kalshi_auth.create_auth_headers(verb, path))

    resp = client.rest_client.request(verb, url, headers=headers, body=body)
    # Same status-check + parse tail as fetch_json_page (see _check_and_parse)
    return _check_and_parse(resp)


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
