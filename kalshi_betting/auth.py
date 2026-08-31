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
    Imports PROD_URL, SANDBOX_URL, SECRETS_FILE, PEM_FILE, and
    DEFAULT_EXCHANGE_INDEX from config.py, and api_call_with_retry /
    fetch_json_page from _http.py. build_client() is called by main.py,
    historical.py, and (indirectly) backtest.py. verify_auth() is called by
    main.py to confirm credentials and read balance.

Notes:
    KalshiClient does NOT accept api_key_id and private_key_pem as constructor
    parameters. Instead it detects them as monkey-patched attributes on the
    Configuration object via hasattr() and builds KalshiAuth internally.
    The sandbox endpoint (demo-api.kalshi.co) requires a completely separate
    account — the production key returns 401 there.

    verify_auth() deliberately does NOT use the modeled client.get_balance().
    That call deserializes through the pinned SDK's strict pydantic model,
    which types balance / portfolio_value / updated_ts as required ints — the
    exact drift-fragile pattern that already broke the events, orders, and
    positions endpoints when Kalshi moved money fields to *_dollars strings.
    It reads the raw body through _http.fetch_json_page() instead, which keeps
    the non-2xx → ApiException semantics so bad credentials still fail loudly.

    Balance is shard-aware. Kalshi's 2026-08-13 change scoped
    /portfolio/balance per exchange shard: the body carries a
    balance_breakdown list of {exchange_index, balance} entries alongside the
    aggregate. Beware the same-key-different-units trap — inside a breakdown
    entry "balance" is a fixed-point DOLLAR STRING, while the TOP-LEVEL
    "balance" is a legacy INTEGER CENTS field. They must never be parsed by
    the same code path.
"""
import json
import logging
from decimal import ROUND_FLOOR, Decimal, InvalidOperation

from kalshi_python_sync import KalshiClient
from kalshi_python_sync.configuration import Configuration

from ._http import api_call_with_retry, fetch_json_page
from .config import (
    DEFAULT_EXCHANGE_INDEX,
    DEV_PEM_FILE,
    PEM_FILE,
    PROD_URL,
    SANDBOX_URL,
    SECRETS_FILE,
)


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

    if mode == "prod":
        url      = PROD_URL
        key_id   = secrets["Kalshi-api-key"]
        pem_text = PEM_FILE.read_text()
    else:
        url      = SANDBOX_URL
        key_id   = secrets.get("dev_api_key", secrets["Kalshi-api-key"])
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


def _dollar_str_to_cents(value) -> int | None:
    """
    Convert a fixed-point dollar string (e.g. "1.1407") to whole cents, floored.

    Floors rather than rounds: the bot must never be told it holds sub-cent
    money it cannot actually spend, because Kelly sizing would then overshoot
    the real balance by a cent. Parsing goes through Decimal rather than float
    — binary float parsing would reintroduce exactly the truncation noise the
    *_dollars string fields exist to avoid.

    Args:
        value: The raw field value from the API body. Normally a fixed-point
            string; a float or int is accepted and converted via its str form.
            None or an unparseable value yields None.

    Returns:
        int | None: The value in whole cents (floored toward negative
            infinity), or None if value is missing or unparseable.
    """
    if value is None:
        return None
    try:
        # Decimal(str(...)) keeps the exact decimal digits the API sent; the
        # ROUND_FLOOR quantize is the "never overstate spendable funds" rule.
        return int((Decimal(str(value)) * 100).to_integral_value(rounding=ROUND_FLOOR))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _balance_cents_from_payload(data: dict) -> int:
    """
    Extract the spendable balance in cents from a raw /portfolio/balance body.

    Kalshi's 2026-08-13 change scoped the balance per exchange shard, so the
    body now carries a balance_breakdown list alongside the aggregate. The
    preference order is:

      1. The balance_breakdown entry for DEFAULT_EXCHANGE_INDEX — the only
         shard our order path can reach, so funds parked on any other shard
         must not inflate Kelly sizing. Entries carry the dollar string under
         the key "balance" (NOT "balance_dollars").
      2. Top-level balance_dollars — the fixed-point aggregate across shards.
      3. Legacy top-level integer balance, already in cents. This is the
         sandbox shape and the pre-drift production shape; it is returned
         as-is and must NOT be run through the dollar converter.

    Args:
        data (dict): Parsed JSON body of GET /portfolio/balance.

    Returns:
        int: Spendable balance in whole cents.

    Raises:
        ValueError: If no balance field in the payload parses. This is the
            deliberate loud-failure carve-out to the project's
            return-None-on-validation-failure convention: sizing real trades
            against an unknown balance is worse than aborting the run.
    """
    breakdown = data.get("balance_breakdown") or []
    for entry in breakdown:
        try:
            # A non-dict entry raises AttributeError on .get and a missing /
            # non-numeric exchange_index raises TypeError/ValueError — a
            # malformed entry must never abort the scan of the good ones.
            idx = int(entry.get("exchange_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        if idx == DEFAULT_EXCHANGE_INDEX:
            # Inside a breakdown entry "balance" is a DOLLAR STRING (unlike the
            # top-level "balance", which is integer cents) — hence the dollar
            # converter here. balance_dollars is accepted first in case the
            # field is ever added to the entries.
            cents = _dollar_str_to_cents(entry.get("balance_dollars") or entry.get("balance"))
            if cents is not None:
                return cents
    if breakdown:
        # Breakdown present but no usable shard-0 entry: fall through to the
        # aggregate rather than reading $0 (which would falsely trip the
        # MIN_BALANCE_CENTS abort), but say so.
        logging.warning("balance_breakdown had no parseable shard-%d entry; "
                        "falling back to aggregate balance", DEFAULT_EXCHANGE_INDEX)
    cents = _dollar_str_to_cents(data.get("balance_dollars"))
    if cents is not None:
        return cents
    legacy = data.get("balance")
    if isinstance(legacy, int):
        # Top-level legacy field is ALREADY cents — no dollar conversion.
        return legacy
    raise ValueError(f"Unparseable balance payload: keys={sorted(data)}")


def verify_auth(client: KalshiClient) -> int:
    """
    Verify that the client's credentials are valid and return the account balance.

    Reads GET /portfolio/balance through the SDK's raw-response variant wrapped
    in api_call_with_retry(), so a transient 429/5xx doesn't abort the run
    before scanning even starts, and parses the shard-aware body itself (see
    _balance_cents_from_payload). If authentication fails, fetch_json_page
    re-raises the non-2xx as ApiException. Used in production mode both at
    startup (to confirm auth works and read the pre-trade balance) and after
    trading (to read the post-trade balance for the Excel log).

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        int: Current spendable balance in cents on DEFAULT_EXCHANGE_INDEX
            (e.g. 100000 = $1,000.00).

    Raises:
        ApiException: If the request returns a non-2xx status (e.g. 401
            Unauthorized when credentials are wrong).
        ValueError: If the response body carries no parseable balance field.
        Exception: Any other exception raised by the underlying HTTP client
            (e.g. a network error that outlived the retry budget).
    """
    # Raw-response call: the modeled get_balance() deserializes through the
    # pinned SDK's strict pydantic model (balance/portfolio_value/updated_ts
    # all required) — the same drift failure class that already broke
    # events/orders/positions. fetch_json_page restores non-2xx semantics,
    # so bad credentials still raise ApiException loudly.
    data = api_call_with_retry(fetch_json_page, client.get_balance_without_preload_content)
    # Prefer the shard we can actually route orders to over the cross-shard
    # aggregate, so unreachable funds can't inflate Kelly sizing.
    balance_cents = _balance_cents_from_payload(data)
    logging.info(
        "Auth OK — balance: %d cents  ($%.2f)",
        balance_cents,
        balance_cents / 100,
    )
    return balance_cents
