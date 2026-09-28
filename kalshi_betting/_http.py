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
    import. It also reads a failed request back: api_error_payload() parses
    the exchange's JSON error object out of an ApiException, and
    api_error_summary() turns any failed request into one short log line
    (status, reason, the exchange's error code, message and details), which
    trader.py logs and records in place of the SDK's multi-line exception text.

Dependencies:
    No project imports — this module is a leaf so auth.py, scanner.py,
    trader.py, and historical.py can all import from it without introducing
    cycles (the standalone, human-run verification CLI kept deliberately
    outside the pipeline's import graph also imports from here — see
    CLAUDE.md's pipeline-isolation rule). Imports ApiException from the
    kalshi_python_sync SDK to re-raise raw-response HTTP errors with the same
    exception types the modeled calls used, and (optionally) urllib3's
    exception types to recognize transport-level drops.

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

    Neither public helper narrows its return type. Every Kalshi endpoint
    observed answers a 2xx with a JSON object, but _check_and_parse hands back
    whatever the parser produced, so both are annotated `-> Any`: a body of
    `"accepted"`, `[]`, `123`, `true` or a literal `null` (which parses to
    None) reaches the caller unchanged. Callers that immediately `.get()` it
    raise AttributeError on such a body, and callers that subscript it
    (trader._submit_order's `data["order"]["status"]`) raise TypeError. Both
    are deliberate loud failures at most call sites, order submission included:
    there an exception is what routes trader._execute_one into its
    ambiguous-submission path, which reconciles the outcome against the
    account's position ledger. trader._execute_transfer is one exception —
    an accepted transfer has no such ledger to reconcile it against and its
    caller's generic handler would report a FAILED POST — so it guards with
    isinstance(..., dict) instead (DR-05). A few read-only lookups also check
    the type, as a failed read.
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

# The longest line api_error_summary returns. Room for a status, a reason, and
# the exchange's error code, message and a sentence of details; a longer
# description is cut rather than let one error fill a log line.
_ERROR_SUMMARY_MAX_CHARS = 300


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


def _check_and_parse(resp: Any) -> Any:
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
        Any: The parsed JSON response body — a `dict` for every Kalshi
            endpoint response observed in practice, but the underlying JSON
            parser (orjson.loads or the stdlib json.loads) doesn't enforce
            that, so the type is not narrowed to `dict` here.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
        ValueError: (json.JSONDecodeError, or TypeError for a null body) when
            the status IS 2xx but the body is empty or is not JSON. Because the
            status check runs first, this can only mean the request itself
            SUCCEEDED — which matters at any call site where that implies work
            was already done server-side (see trader._execute_transfer).
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


def _body_text(exc: BaseException) -> str | None:
    """
    Return a failed request's response body as text.

    The SDK's ApiException keeps the body on its .body attribute as text; a
    hand-built one may hold UTF-8 bytes instead, which are decoded here.

    Args:
        exc (BaseException): The exception a request raised.

    Returns:
        str | None: The body, or None when there is none or it is not UTF-8
            text. Never raises.
    """
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, (bytes, bytearray)):
            body = bytes(body).decode("utf-8")
    except Exception:
        return None
    return body if isinstance(body, str) else None


def api_error_payload(exc: BaseException) -> dict | None:
    """
    Read the exchange's own error object out of a failed request's response.

    Kalshi answers a rejected request with a JSON body of the form
    {"error": {"code": ..., "message": ..., "details": ...}} ("details" is
    optional). This is the one place that body is parsed: trader._is_fok_kill
    reads the code from it to recognise the V2 kill response, and
    api_error_summary prints it in log lines.

    Args:
        exc (BaseException): The exception a request raised.

    Returns:
        dict | None: The object under "error", or None when the exception has
            no body, the body is not UTF-8 text, is not JSON, or has no
            "error" object. Never raises.
    """
    body = _body_text(exc)
    if body is None:
        return None
    try:
        # The stdlib parser, not _json_loads: an error body is tiny, and this
        # keeps the parse identical whether or not the optional orjson is installed
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else None


def _one_line(value: Any) -> str:
    """
    Render any value as printable text on a single line.

    Every character that is not printable — newlines and tabs, but also
    control characters such as a terminal escape or a NUL, which a JSON body
    can carry as \\u escapes — becomes a space, and runs of spaces collapse
    to one. The result is safe in a log line and in a spreadsheet cell (the
    trade log's Notes cell refuses control characters).

    Args:
        value (Any): The value to render.

    Returns:
        str: The value's text on one line, stripped at both ends.
    """
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    return " ".join(text.split())


def api_error_summary(exc: BaseException, limit: int = _ERROR_SUMMARY_MAX_CHARS) -> str:
    """
    Describe a failed request in one line, for log lines and error fields.

    The SDK's own str(ApiException) spreads the status, the reason, every
    response header and the body over several lines — about 900 bytes, most
    of it headers — so logging the exception whole buries the part that says
    what went wrong (TS-02 in CLAUDE.md). This keeps only that part:

      * An HTTP error (the exception has a .status, as ApiException does):
        "HTTP <status> <reason>", then the exchange's error code, message and,
        when present, details from the response body (see api_error_payload):
            HTTP 409 Conflict — fill_or_kill_insufficient_resting_volume: fill
            or kill insufficient resting volume
            HTTP 400 Bad Request — missing_parameters: missing parameters
            (Key: 'CreateOrderV2Request.SelfTradePreventionType' ...)
        A body that is not the exchange's JSON error object (a gateway's HTML
        page, plain text) is kept as it is after a colon, on one line, so
        nothing the exchange said is lost.
      * Anything else (a dropped connection, a response the caller could not
        read): the exception's class name and the first line of its message,
        e.g. "ProtocolError: ('Connection aborted.', ...)".

    trader.py uses it wherever it logs or records a failed request (order
    submissions, the unwind, position and balance reads, transfers), and
    scanner._fetch_orderbook for a failed order-book read.

    Args:
        exc (BaseException): The exception to describe.
        limit (int): The most characters to return; a longer description is
            cut and ends in "…".

    Returns:
        str: One line of printable text, at most `limit` characters. Never
            raises: an exception whose attributes cannot be read is named by
            its class.
    """
    try:
        name = _one_line(type(exc).__name__) or "Exception"
    except Exception:
        name = "Exception"
    try:
        status = getattr(exc, "status", None)
        if status is not None:
            reason = getattr(exc, "reason", None)
            text = f"HTTP {_one_line(status)}"
            if reason:
                text += f" {_one_line(reason)}"
            error = api_error_payload(exc)
            parts = [] if error is None else [
                _one_line(error[key]) for key in ("code", "message")
                if error.get(key) not in (None, "")
            ]
            details = None if error is None else error.get("details")
            if parts:
                text += " — " + ": ".join(parts)
            if details not in (None, ""):
                text += f" ({_one_line(details)})"
            if not parts and details in (None, ""):
                # Not the exchange's error object: keep whatever the body says
                body = _one_line(_body_text(exc) or "")
                if body:
                    text += f": {body}"
        else:
            lines = str(exc).strip().splitlines()
            first = _one_line(lines[0]) if lines else ""
            text = f"{name}: {first}" if first else name
    except Exception:
        # The promise is one line and no exception: a broken error object is
        # still named
        text = name
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…" if limit > 0 else ""


def fetch_json_page(fetch_fn: Any, **kwargs) -> Any:  # whatever the 2xx body parses to; _check_and_parse does not narrow (a literal null is None)
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
        Any: The parsed JSON response body — a `dict` for every Kalshi endpoint
            response observed in practice, but not narrowed: _check_and_parse
            returns whatever the JSON parser produced, so a body of `"ok"`,
            `[]`, `123`, `true` or a literal `null` (which parses to None)
            reaches the caller as-is. Callers that immediately `.get()` the
            result raise AttributeError on such a body, and callers that
            subscript it (trader._submit_order's `data["order"]["status"]`)
            raise TypeError. Both are deliberate loud failures: on the order
            path, raising is what routes trader._execute_one into its
            ambiguous-submission path, where the account position decides the
            outcome. One call site that instead guards with
            isinstance(..., dict) — because its 2xx has already moved money and
            nothing reconciles a transfer after the fact — is
            trader._execute_transfer, which reads signed_request_json rather
            than this helper (DR-05). A few read-only lookups also check the
            type, as a failed read.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
        ValueError: (json.JSONDecodeError, or TypeError for a null body) when
            the status IS 2xx but the body is empty or is not JSON. Because the
            status check runs first, this can only mean the request itself
            SUCCEEDED — which matters at any call site where that implies work
            was already done server-side (see trader._execute_transfer).
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
) -> Any:  # whatever the 2xx body parses to; _check_and_parse does not narrow (a literal null is None)
    """
    Perform a signed request of any HTTP method against an arbitrary API path.

    The pinned SDK has no generated method for the V2 order endpoint
    (/portfolio/events/orders, used by trader._submit_order_v2) or the
    intra-exchange collateral transfer endpoint (used by
    trader._execute_transfer), and its modeled calls deserialize through
    response models that live API drift keeps breaking. (The /historical
    archive is reached through a separate, dedicated helper,
    historical._signed_raw_get, not this function.) This helper signs the
    request the way every SDK call is signed (KalshiAuth:
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
        Any: The parsed JSON response body — a `dict` for every Kalshi endpoint
            response observed in practice, but not narrowed: _check_and_parse
            returns whatever the JSON parser produced, so a body of `"ok"`,
            `[]`, `123`, `true` or a literal `null` (which parses to None)
            reaches the caller as-is. trader._execute_transfer guards on
            isinstance(..., dict) for exactly that reason — by the time the
            body is read, the 2xx has already moved the money and nothing
            reconciles a transfer after the fact (DR-05). The other
            money-moving caller, trader._submit_order_v2, deliberately stays
            loud: its exception routes trader._execute_one into the
            ambiguous-submission path, where the account position decides.

    Raises:
        ApiException: (or a status-specific subclass) when the HTTP status is
            not 2xx.
        ValueError: (json.JSONDecodeError, or TypeError for a null body) when
            the status IS 2xx but the body is empty or is not JSON. Because the
            status check runs first, this can only mean the request itself
            SUCCEEDED — which matters at any call site where that implies work
            was already done server-side (see trader._execute_transfer).
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
