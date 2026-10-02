"""
File: auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Logs the bot in to Kalshi. It reads the API key ID from secrets.json and the
    RSA private key from its PEM file, and builds a KalshiClient for either the
    live site or the practice (sandbox) site. Every module that talks to Kalshi
    uses a client made here.

    It also reads the account balance, which doubles as a check that the login
    works. One read gives two things: the cash on each shard (one of the
    exchange's separate sections, each holding its own cash) and what Kalshi
    says the open positions are worth.

Dependencies:
    Imports file paths, site addresses and DEFAULT_EXCHANGE_INDEX from config.py,
    and api_call_with_retry and fetch_json_page from _http.py.
    Used by main.py (build_client, read_account_balance), historical.py
    (build_client), trader.py (read_shard_balances), backtest.py
    (read_account_balance: a backtest's starting balance when --balance is
    not given) and the human-run verification tool (build_client, verify_auth).

Notes:
    - Set the key ID and private key as attributes on the SDK's Configuration,
      never as KalshiClient arguments: the SDK looks for them there.
    - The sandbox needs its own account; the production key is refused there.
    - Read the balance from the raw reply, never the SDK's get_balance(), which
      fails whenever Kalshi leaves out a field it expects.
    - Cash is always returned per shard, with no one-number version, so no
      caller can size on a single shard's cash by mistake.
    - Inside a balance_breakdown entry "balance" is a dollar string, while the
      top-level "balance" and "portfolio_value" are whole cents. Never parse
      them the same way.
    - The positions value is optional: when it cannot be read it is None,
      never an error.
"""
import json
import logging
from dataclasses import dataclass
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
            raw-response methods such as get_balance_without_preload_content()
            and get_positions_without_preload_content().

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


def _balance_cents_by_shard(data: dict) -> dict[int, int]:
    """
    Extract the spendable balance in cents PER SHARD from a raw
    /portfolio/balance body.

    Kalshi's 2026-08-13 change scoped the balance per exchange shard, so the
    body now carries a balance_breakdown list alongside the aggregate. Unlike
    the single-shard predecessor of this function, every parseable shard is
    returned — a later collateral-transfer planner and a shard coverage check
    both need the full picture, not just the routable shard. The preference
    order (tiers 2 and 3 only apply when the breakdown is absent or entirely
    unparseable):

      1. Every balance_breakdown entry that parses: {exchange_index: cents}
         for each entry, keyed by its own exchange_index (not just
         DEFAULT_EXCHANGE_INDEX). Entries carry the dollar string under the
         key "balance" (NOT "balance_dollars").
      2. Top-level balance_dollars — the fixed-point aggregate across shards
         — attributed to DEFAULT_EXCHANGE_INDEX, used only when the
         breakdown is missing/empty or nothing in it parsed.
      3. Legacy top-level integer balance, already in cents, attributed to
         DEFAULT_EXCHANGE_INDEX. This is the sandbox shape and the
         pre-drift production shape; it is returned as-is and must NOT be
         run through the dollar converter.

    Args:
        data (dict): Parsed JSON body of GET /portfolio/balance.

    Returns:
        dict[int, int]: Spendable balance in whole cents, keyed by
            exchange_index. Callers that need a single number for Kelly
            sizing must sum this dict themselves — sizing is deliberately
            portfolio-wide, not per-shard.

    Raises:
        ValueError: If no balance field in the payload parses. This is the
            deliberate loud-failure carve-out to the project's
            return-None-on-validation-failure convention: sizing real trades
            against an unknown balance is worse than aborting the run.
    """
    breakdown = data.get("balance_breakdown") or []
    by_shard: dict[int, int] = {}
    for entry in breakdown:
        try:
            # A non-dict entry raises AttributeError on .get and a missing /
            # non-numeric exchange_index raises TypeError/ValueError — a
            # malformed entry must never abort the scan of the good ones.
            idx = int(entry.get("exchange_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        # Inside a breakdown entry "balance" is a DOLLAR STRING (unlike the
        # top-level "balance", which is integer cents) — hence the dollar
        # converter here. balance_dollars is accepted first in case the
        # field is ever added to the entries.
        # PRESENCE, not truthiness: a numeric 0 or "" in balance_dollars is a
        # real answer, and `or` would discard it and read the stale legacy
        # field instead — reporting a genuinely-empty shard as funded. That is
        # the one OVER-statement this function can produce, in a module whose
        # flooring and drop-with-a-warning rules both exist to guarantee the
        # opposite (TS-16). An unparseable "" now correctly reaches the WARNING
        # branch below instead of silently reading another field.
        raw = entry["balance_dollars"] if "balance_dollars" in entry else entry.get("balance")
        cents = _dollar_str_to_cents(raw)
        if cents is not None:
            if idx in by_shard:
                # A repeated index means the payload shape changed (a
                # per-subaccount split, say). Last-win matches the previous
                # behaviour and is as defensible as first-win; being SILENT
                # about it is not.
                logging.warning(
                    "balance_breakdown lists exchange_index %d more than once — "
                    "keeping the last entry; earlier entry discarded", idx,
                )
            by_shard[idx] = cents
        else:
            # A parseable shard index with an unparseable balance means that
            # shard's REAL funds silently vanish from sizing, the coverage
            # audit, and the transfer planner — exactly the drift failure
            # class this codebase keeps hitting. Under-sizing is safe, but it
            # must never be invisible.
            logging.warning(
                "balance_breakdown entry for shard %d carried no parseable "
                "balance (keys=%s) — that shard's funds are NOT counted",
                idx, sorted(entry),
            )
    if by_shard:
        return by_shard
    if breakdown:
        # Breakdown present but nothing in it parsed: fall through to the
        # aggregate rather than reading $0 (which would falsely trip the
        # MIN_BALANCE_CENTS abort), but say so.
        logging.warning("balance_breakdown had no parseable entries; "
                        "falling back to aggregate balance")
    cents = _dollar_str_to_cents(data.get("balance_dollars"))
    if cents is not None:
        return {DEFAULT_EXCHANGE_INDEX: cents}
    legacy = data.get("balance")
    if isinstance(legacy, int):
        # Top-level legacy field is ALREADY cents — no dollar conversion.
        return {DEFAULT_EXCHANGE_INDEX: legacy}
    raise ValueError(f"Unparseable balance payload: keys={sorted(data)}")


# The largest positions value accepted: 2**53 cents (about $90 trillion). A float
# holds every whole number up to it exactly; anything larger means the field is
# not what this module expects. The second line is the same limit in dollars.
_MAX_POSITIONS_VALUE_CENTS = 2 ** 53
_MAX_POSITIONS_VALUE_DOLLARS = Decimal(_MAX_POSITIONS_VALUE_CENTS) / 100


@dataclass(frozen=True)
class AccountBalance:
    """
    One reading of the account balance: the cash on each shard and what Kalshi
    says the open positions are worth.

    Its fields cannot be replaced, but callers must not edit the cash dict.

    Attributes:
        shard_cash_cents (dict[int, int]): Cash in whole cents, keyed by shard number.
        positions_value_cents (int | None): The open positions' worth in whole cents; None if unreadable.
    """
    shard_cash_cents: dict[int, int]
    positions_value_cents: int | None


def _positions_value_cents(data: dict) -> int | None:
    """
    Read what Kalshi says the open positions are worth from a balance reply, in whole cents.

    Uses portfolio_value_dollars (a dollar string, rounded down to the cent) when
    the reply has that key, otherwise portfolio_value (whole cents). The value
    never includes cash. Never raises: a missing, unreadable, negative or
    impossibly large value gives None.

    Args:
        data (dict): The parsed reply from GET /portfolio/balance.

    Returns:
        int | None: The value in whole cents, or None if the reply has no usable value.
    """
    if "portfolio_value_dollars" in data:
        raw = data["portfolio_value_dollars"]
        # Accept only a string or a plain number (True/False count as numbers
        # in Python, so they are refused by name)
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            return None
        try:
            amount = Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return None
        # Check the range before converting, so a huge number is never turned into cents
        if not amount.is_finite() or not 0 <= amount <= _MAX_POSITIONS_VALUE_DOLLARS:
            return None
        cents = _dollar_str_to_cents(raw)
    else:
        raw = data.get("portfolio_value")
        # Only a real whole number counts; True would otherwise read as one cent
        cents = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    if cents is None or not 0 <= cents <= _MAX_POSITIONS_VALUE_CENTS:
        return None
    return cents


def read_account_balance(client: KalshiClient) -> AccountBalance:
    """
    Read the account balance: the cash on each shard and what Kalshi says the
    open positions are worth.

    Makes one GET /portfolio/balance, retried on temporary errors, and logs one
    "Auth OK" line with each shard's cash. Bad credentials raise, so this read
    is also the check that the login works.

    Args:
        client (KalshiClient): A client made by build_client().

    Returns:
        AccountBalance: Each shard's cash and the positions' value (None if unreadable).

    Raises:
        ApiException: If Kalshi answers with an error status, e.g. 401 for bad credentials.
        ValueError: If the reply has no readable cash balance.
        Exception: Any other request error, such as a network failure that outlasts the retries.
    """
    # One retried read of the raw reply; an error status raises ApiException
    data = api_call_with_retry(fetch_json_page, client.get_balance_without_preload_content)
    # The cash per shard; raises ValueError if the reply has none
    shard_balances = _balance_cents_by_shard(data)
    logging.info(
        "Auth OK — balance by shard: %s (total $%.2f)",
        shard_balances,
        sum(shard_balances.values()) / 100,
    )
    return AccountBalance(shard_balances, _positions_value_cents(data))


def verify_auth(client: KalshiClient) -> dict[int, int]:
    """
    Check that the client's credentials work and return the cash on each shard.

    Makes the same single read as read_account_balance() and keeps only the cash.

    Args:
        client (KalshiClient): A client made by build_client().

    Returns:
        dict[int, int]: Cash in whole cents by shard, e.g. {0: 100000} is $1,000.00 on shard 0.

    Raises:
        ApiException: If Kalshi answers with an error status, e.g. 401 for bad credentials.
        ValueError: If the reply has no readable cash balance.
        Exception: Any other request error, such as a network failure that outlasts the retries.
    """
    # Same read as read_account_balance, keeping only the cash
    return read_account_balance(client).shard_cash_cents


def read_shard_balances(client: KalshiClient) -> dict[int, int]:
    """
    Single-shot per-shard balance read — no retry, no backoff, and no logging
    of its own (the shared _balance_cents_by_shard parse it calls may still
    emit a WARNING on a malformed or unparseable balance_breakdown).

    The bounded-poll variant of verify_auth(): trader._await_transfer_settlement
    re-reads the balance every TRANSFER_POLL_INTERVAL_SECONDS against a
    monotonic deadline, and that deadline is only checked BETWEEN reads — so
    the read itself must return (or fail) quickly. verify_auth's
    api_call_with_retry wrapper can hold a single call for ~60s of exponential
    backoff during an outage, silently tripling the "bounded" wait; here a
    transient failure just raises and costs the caller one poll interval.

    Same parse, same shard-aware dict as verify_auth — "landed" is judged
    against exactly the numbers sizing was based on.

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        dict[int, int]: Spendable balance in whole cents, keyed by exchange_index.

    Raises:
        ApiException: On a non-2xx status.
        ValueError: If the response body carries no parseable balance field.
        Exception: Any other exception raised by the underlying HTTP client
            (e.g. a network error) — this function has no retry budget to
            absorb one, so a transient failure surfaces immediately. The
            caller (trader._await_transfer_settlement) catches bare
            Exception here and treats it as "nothing observed", never as
            success.
    """
    return _balance_cents_by_shard(fetch_json_page(client.get_balance_without_preload_content))
