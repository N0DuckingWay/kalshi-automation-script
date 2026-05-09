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
    Imports PROD_URL, SANDBOX_URL, SECRETS_FILE, and PEM_FILE from config.py.
    build_client() is called by main.py, historical.py, and (indirectly) backtest.py.
    verify_auth() is called by main.py to confirm credentials and read balance.

Notes:
    KalshiClient does NOT accept api_key_id and private_key_pem as constructor
    parameters. Instead it detects them as monkey-patched attributes on the
    Configuration object via hasattr() and builds KalshiAuth internally.
    The sandbox endpoint (demo-api.kalshi.co) requires a completely separate
    account — the production key returns 401 there.
"""
import json
import logging

from kalshi_python_sync import KalshiClient
from kalshi_python_sync.configuration import Configuration

from .config import PROD_URL, SANDBOX_URL, SECRETS_FILE, PEM_FILE


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
        KeyError: If "Kalshi-api-key" is missing from secrets.json in prod mode.
        json.JSONDecodeError: If secrets.json cannot be parsed as JSON.
    """
    raw = SECRETS_FILE.read_text().strip()
    # secrets.json may be missing outer braces — wrap if needed so json.loads
    # always receives a valid JSON object regardless of how the file was saved
    if not raw.startswith("{"):
        raw = "{" + raw + "}"
    secrets  = json.loads(raw)

    # Read the full PEM text (not a file path) — KalshiClient requires the
    # key content as a string, not a filesystem reference
    pem_text = PEM_FILE.read_text()

    if mode == "prod":
        url    = PROD_URL
        key_id = secrets["Kalshi-api-key"]
    else:
        # Prefer a dedicated sandbox key if configured; fall back to prod key
        url    = SANDBOX_URL
        key_id = secrets.get("dev_api_key", secrets["Kalshi-api-key"])

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

    Calls the Kalshi get_balance() endpoint. If authentication fails, this call
    will raise an exception from the underlying HTTP client. Used in production
    mode both at startup (to confirm auth works and read the pre-trade balance)
    and after trading (to read the post-trade balance for the Excel log).

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        int: Current account balance in cents (e.g. 100000 = $1,000.00).

    Raises:
        Exception: Any exception raised by the Kalshi API client if the request
            fails (e.g. 401 Unauthorized if credentials are wrong, network error).
    """
    resp = client.get_balance()
    logging.info(
        "Auth OK — balance: %d cents  ($%.2f)",
        resp.balance,
        resp.balance / 100,
    )
    return resp.balance
