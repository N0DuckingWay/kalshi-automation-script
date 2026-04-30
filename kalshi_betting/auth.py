"""Build an authenticated KalshiClient from secrets.json + RSA PEM key."""
import json
import logging

from kalshi_python_sync import KalshiClient
from kalshi_python_sync.configuration import Configuration

from .config import PROD_URL, SANDBOX_URL, SECRETS_FILE, PEM_FILE


def build_client(mode: str) -> KalshiClient:
    """
    Construct a KalshiClient for the given mode ('dev' or 'prod').

    KalshiClient expects cfg.api_key_id and cfg.private_key_pem as
    monkey-patched attributes on the Configuration object (not constructor params).
    KalshiClient.__init__ detects them via hasattr() and builds KalshiAuth internally.
    """
    raw = SECRETS_FILE.read_text().strip()
    # secrets.json may be missing outer braces — wrap if needed
    if not raw.startswith("{"):
        raw = "{" + raw + "}"
    secrets  = json.loads(raw)
    pem_text = PEM_FILE.read_text()

    if mode == "prod":
        url    = PROD_URL
        key_id = secrets["Kalshi-api-key"]
    else:
        url    = SANDBOX_URL
        key_id = secrets.get("dev_api_key", secrets["Kalshi-api-key"])

    cfg = Configuration(host=url)
    cfg.api_key_id      = key_id    # detected by KalshiClient.__init__ via hasattr
    cfg.private_key_pem = pem_text  # must be PEM string content, not a file path

    client = KalshiClient(configuration=cfg)
    logging.info("KalshiClient built for mode=%s  url=%s", mode, url)
    return client


def verify_auth(client: KalshiClient) -> int:
    """Call get_balance() to confirm auth works. Returns balance in cents."""
    resp = client.get_balance()
    logging.info(
        "Auth OK — balance: %d cents  ($%.2f)",
        resp.balance,
        resp.balance / 100,
    )
    return resp.balance
