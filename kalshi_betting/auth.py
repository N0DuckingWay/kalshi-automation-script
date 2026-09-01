"""
File: auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Handles all authentication concerns for the Kalshi REST API. Reads the RSA
    private key and API key ID from the project's secrets.json and PEM files,
    constructs a KalshiClient instance pointed at either the production or sandbox
    endpoint, and provides a verify_auth() helper that confirms the credentials
    work by fetching the account balance. Every other module that talks to the
    Kalshi API receives a KalshiClient produced by this module.

Dependencies:
    Imports PROD_URL, SANDBOX_URL, SECRETS_FILE, and PEM_FILE from config.py, and
    api_call_with_retry / fetch_json_page from _http.py. build_client() is called
    by main.py, historical.py, and (indirectly) backtest.py. verify_auth() is
    called by main.py to confirm credentials and read balance.

Notes:
    KalshiClient does NOT accept api_key_id and private_key_pem as constructor
    parameters. Instead it detects them as monkey-patched attributes on the
    Configuration object via hasattr() and builds KalshiAuth internally.
    The sandbox endpoint (demo-api.kalshi.co) requires a completely separate
    account — the production key returns 401 there.
    verify_auth() calls get_balance_without_preload_content and parses the JSON
    body itself rather than the modeled get_balance() — see verify_auth's
    docstring and the CLAUDE.md API-drift gotcha for why.
"""
import json
import logging

from kalshi_python_sync import KalshiClient
from kalshi_python_sync.configuration import Configuration

from ._http import api_call_with_retry, fetch_json_page
from .config import DEV_PEM_FILE, PEM_FILE, PROD_URL, SANDBOX_URL, SECRETS_FILE


def build_client(mode: str) -> KalshiClient:
    """
    Construct an authenticated KalshiClient for the given operating mode.

    Reads credentials from secrets.json and the RSA PEM file defined in config.py,
    then builds a Configuration object that KalshiClient.__init__ will use to
    create the internal KalshiAuth signer for RSA-based request authentication.

    In dev mode, uses the "dev_api_key" from secrets.json if present, falling
    back to "Kalshi-api-key" otherwise. The sandbox API endpoint is used in dev
    mode; the production endpoint is used in prod mode.

    Args:
        mode (str): Operating mode — "prod" uses the live Kalshi API and the
            "Kalshi-api-key" from secrets.json; "dev" uses the sandbox API and
            the "dev_api_key" (or falls back to "Kalshi-api-key").

    Returns:
        KalshiClient: An authenticated client object ready to call Kalshi API
            methods such as get_markets(), get_balance(), batch_create_orders(), etc.

    Raises:
        FileNotFoundError: If secrets.json or the PEM file do not exist at the
            paths defined in config.py.
        KeyError: If "Kalshi-api-key" is missing from secrets.json in prod
            mode, or if BOTH "dev_api_key" and "Kalshi-api-key" are missing
            in dev mode (dev_api_key alone is sufficient).
        json.JSONDecodeError: If secrets.json cannot be parsed as JSON.
    """
    raw = SECRETS_FILE.read_text().strip()
    # secrets.json may be missing outer braces — wrap if needed so json.loads
    # always receives a valid JSON object regardless of how the file was saved
    if not raw.startswith("{"):
        raw = "{" + raw + "}"
    secrets  = json.loads(raw)

    if mode == "prod":
        url      = PROD_URL
        key_id   = secrets["Kalshi-api-key"]
        pem_text = PEM_FILE.read_text()
    else:
        url      = SANDBOX_URL
        # Lazy fallback: dev_api_key is optional; only require the prod key
        # when the dev key is absent (a .get() default is evaluated eagerly,
        # so `secrets.get("dev_api_key", secrets["Kalshi-api-key"])` raised
        # KeyError in dev mode even when dev_api_key WAS present)
        if "dev_api_key" in secrets:
            key_id = secrets["dev_api_key"]
        else:
            key_id = secrets["Kalshi-api-key"]
        pem_file = DEV_PEM_FILE if DEV_PEM_FILE.exists() else PEM_FILE
        pem_text = pem_file.read_text()

    cfg = Configuration(host=url)
    # KalshiClient.__init__ detects these attributes via hasattr() and builds
    # KalshiAuth internally — they are NOT standard Configuration constructor params
    cfg.api_key_id      = key_id
    cfg.private_key_pem = pem_text

    client = KalshiClient(configuration=cfg)
    logging.info("KalshiClient built for mode=%s  url=%s", mode, url)
    return client


def verify_auth(client: KalshiClient) -> int:
    """
    Verify that the client's credentials are valid and return the account balance.

    Calls the Kalshi get_balance() endpoint through api_call_with_retry() so a
    transient 429/5xx doesn't abort the run before scanning even starts. If
    authentication fails, this call will raise an exception from the underlying
    HTTP client. Used in production mode both at startup (to confirm auth works
    and read the pre-trade balance) and after trading (to read the post-trade
    balance for the Excel log).

    Uses the raw-response variant + JSON parsing, same as trader._position_count
    and scanner.get_held_tickers: the pinned SDK's GetBalanceResponse model
    types balance/portfolio_value/updated_ts as legacy StrictInt fields, which
    is the same field class that already drifted away for markets, positions,
    and orders (see the CLAUDE.md API-drift gotcha) — the modeled get_balance
    call is the last one of those still standing and would raise pydantic
    ValidationError on a live response that no longer sends them.

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        int: Current account balance in cents (e.g. 100000 = $1,000.00).

    Raises:
        Exception: Any exception raised by the Kalshi API client if the request
            fails (e.g. 401 Unauthorized if credentials are wrong, network error).
        KeyError: If the parsed response body has none of the recognized
            balance fields ("balance", "balance_fp", "balance_dollars") — an
            unrecognized response shape must fail loudly, not silently return
            a wrong value.
    """
    # Raw-response call: the raw variant skips the SDK's pydantic response
    # model entirely (see module Notes), so a future field rename can't crash
    # auth verification the way the modeled get_balance() would.
    data = api_call_with_retry(fetch_json_page, client.get_balance_without_preload_content)
    raw = data.get("balance")
    if raw is None:
        # Defensive drift fallback, mirroring the *_fp / *_dollars pattern
        # used elsewhere (e.g. trader._position_count's position_fp fallback)
        raw = data.get("balance_fp")
        if raw is None and data.get("balance_dollars") is not None:
            raw = float(data["balance_dollars"]) * 100
    if raw is None:
        raise KeyError(
            f"No recognizable balance field in get_balance response (keys: {sorted(data)})"
        )
    balance = int(float(raw))
    logging.info("Auth OK — balance: %d cents  ($%.2f)", balance, balance / 100)
    return balance
