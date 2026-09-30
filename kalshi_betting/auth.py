"""
File: auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Handles all authentication concerns for the Kalshi REST API. Reads the RSA
    private key and API key ID from the project's secrets.json and PEM files,
    constructs a KalshiClient instance pointed at either the production or sandbox
    endpoint, and reads the account balance, which doubles as the check that
    the credentials work. Every other module that talks to the Kalshi API
    receives a KalshiClient produced by this module.

    One balance read (GET /portfolio/balance) answers two questions:
    how much CASH each exchange shard holds (only cash can buy contracts),
    and what Kalshi says the account's OPEN POSITIONS are worth.
    read_account_balance() returns both as an AccountBalance;
    verify_auth() makes the same read and returns only the cash per shard;
    read_shard_balances() is a single-shot, retry-free cash read for a
    caller that polls against a deadline.

Dependencies:
    Imports PROD_URL, SANDBOX_URL, SECRETS_FILE, PEM_FILE, DEV_PEM_FILE, and
    DEFAULT_EXCHANGE_INDEX from config.py, and api_call_with_retry /
    fetch_json_page from _http.py. build_client() is called by main.py,
    historical.py, and (indirectly) backtest.py. read_account_balance() is
    called by main.py (the balance before trading, whose cash plus positions
    value the run sizes on, and the cash after trading) and
    read_shard_balances() by trader.py (the collateral-transfer settle poll).
    verify_auth() is read_account_balance() returning only the cash. (The
    standalone, human-run verification CLI kept deliberately outside the
    pipeline's import graph calls build_client() and verify_auth() — see
    CLAUDE.md's pipeline-isolation rule.)

Notes:
    KalshiClient does NOT accept api_key_id and private_key_pem as constructor
    parameters. Instead it detects them as monkey-patched attributes on the
    Configuration object via hasattr() and builds KalshiAuth internally.
    The sandbox endpoint (demo-api.kalshi.co) requires a completely separate
    account — the production key returns 401 there.

    The balance read deliberately does NOT use the modeled
    client.get_balance(). That call deserializes through the pinned SDK's
    strict pydantic model, which types balance / portfolio_value / updated_ts
    as required ints — the drift-fragile pattern that breaks as soon as
    Kalshi stops sending one of them. It reads the raw body through
    _http.fetch_json_page() instead, which keeps the non-2xx → ApiException
    semantics so bad credentials still fail loudly.

    Cash is shard-aware. The balance body carries a balance_breakdown list of
    {exchange_index, balance} entries alongside the account-wide aggregate,
    and the cash is returned as the FULL per-shard breakdown,
    dict[int, int] (exchange_index -> cents), rather than a single scalar:
    the collateral-transfer planner and the shard coverage check both need
    every shard's balance, not just the routable one. Callers that need one
    number sum the dict themselves. There is deliberately no
    scalar-returning wrapper here — a dual API would invite a future caller
    to size against the wrong (single-shard) number.

    The positions value is Kalshi's top-level portfolio_value field: integer
    cents, covering every shard (the request names no shard), and it does
    NOT include cash (the pinned SDK describes it as "the current value of
    all positions held"). It is optional — a reply without a readable value
    gives None, never an error — because the cash alone is enough to trade.

    Beware the same-key-different-units trap — inside a breakdown entry
    "balance" is a fixed-point DOLLAR STRING, while the TOP-LEVEL "balance"
    (and the top-level portfolio_value) is INTEGER CENTS. They must never be
    parsed by the same code path.
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


# The largest positions value this module accepts, in whole cents. A float
# holds every whole number up to 2**53 exactly, so any value up to it
# survives the float arithmetic a caller may do on it. It is about
# $90 trillion, far beyond any account, so a larger number means the field
# is not what _positions_value_cents expects. The dollar bound is the same
# amount, compared before any conversion to cents.
_MAX_POSITIONS_VALUE_CENTS = 2 ** 53
_MAX_POSITIONS_VALUE_DOLLARS = Decimal(_MAX_POSITIONS_VALUE_CENTS) / 100


@dataclass(frozen=True)
class AccountBalance:
    """
    One read of the account balance (GET /portfolio/balance): the cash on
    each exchange shard, and what Kalshi says the open positions are worth.

    Built by read_account_balance(). Frozen at the top level only: the
    per-shard dict is the parse's own dict, not a copy, so a caller must not
    change it.

    Attributes:
        shard_cash_cents (dict[int, int]): Spendable cash in whole cents,
            keyed by exchange_index — the _balance_cents_by_shard parse, the
            same dict verify_auth() returns. Only cash can buy contracts, and
            each order draws on its own market's shard.
        positions_value_cents (int | None): What Kalshi says the account's
            open positions are worth, in whole cents, all shards together
            (the reply's portfolio_value, which does not include cash). None
            when the reply carries no readable value (_positions_value_cents).
    """
    shard_cash_cents: dict[int, int]
    positions_value_cents: int | None


def _positions_value_cents(data: dict) -> int | None:
    """
    Read what Kalshi says the account's open positions are worth from a raw
    /portfolio/balance body, in whole cents.

    The reply's top-level portfolio_value is the value of the open positions
    across every shard, in integer cents. It does not include cash, which is
    the separate balance field (_balance_cents_by_shard). The field is chosen
    by PRESENCE, as _balance_cents_by_shard chooses a breakdown entry's field:

      1. portfolio_value_dollars, a fixed-point dollar string, whenever the
         key is present — converted by _dollar_str_to_cents, so floored to the
         cent like every balance. If it is present but unreadable the answer
         is None: the integer field is not consulted, so one reply's two
         spellings are never mixed. Neither the pinned SDK's model nor any
         reply seen carries this spelling; it is read in case Kalshi adds
         one beside the integer field, as it added balance_dollars beside
         balance. Only a string or a plain number can spell an amount, so
         anything else (a list, an object, a bool) gives None without being
         turned into text, and the amount must be finite and within the
         bound below before it is converted — a huge exponent would
         otherwise build an integer with that many digits.
      2. Otherwise portfolio_value, integer cents, returned as is (never run
         through the dollar converter — the same-key-different-units trap in
         the module notes). Only a real int counts: a bool (an int subclass
         in Python), a string or a float gives None.

    Either way the value must lie between 0 and _MAX_POSITIONS_VALUE_CENTS
    (2**53, about $90 trillion). Every Kalshi position is a held contract
    worth between $0 and $1, so the total cannot be negative, and no account
    comes near the upper bound; a number outside it means the field is not
    what this reader expects.

    This never raises. The value is optional — a caller without it can size
    on cash alone — so a value that cannot be read must not fail a balance
    read whose cash parsed.

    Args:
        data (dict): Parsed JSON body of GET /portfolio/balance.

    Returns:
        int | None: The positions' value in whole cents, from 0 to
            _MAX_POSITIONS_VALUE_CENTS, or None if the reply carries no
            readable value in that range.
    """
    if "portfolio_value_dollars" in data:
        raw = data["portfolio_value_dollars"]
        # Only a string or a plain number can spell an amount (a bool is an
        # int in Python, so it is refused by name). A list or an object is
        # never turned into text, which for a deeply nested one could raise
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            return None
        try:
            amount = Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return None
        # Range-checked before the cents conversion, which cannot then
        # overflow or build a huge integer; -0.00 passes and reads as 0
        if not amount.is_finite() or not 0 <= amount <= _MAX_POSITIONS_VALUE_DOLLARS:
            return None
        cents = _dollar_str_to_cents(raw)
    else:
        raw = data.get("portfolio_value")
        # bool first: True is an int in Python and must not read as one cent
        cents = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    if cents is None or not 0 <= cents <= _MAX_POSITIONS_VALUE_CENTS:
        return None
    return cents


def read_account_balance(client: KalshiClient) -> AccountBalance:
    """
    Read the account balance: the cash on each exchange shard, and what
    Kalshi says the open positions are worth.

    Makes one GET /portfolio/balance through the SDK's raw-response variant,
    wrapped in api_call_with_retry() so a transient 429/5xx or a dropped
    connection is retried rather than ending the run before it scans; the
    read is a GET, so retrying it cannot duplicate anything. If the
    credentials are bad, fetch_json_page re-raises the non-2xx as an
    ApiException, so this read is also the check that authentication works.
    Logs one INFO line, "Auth OK — balance by shard: ...", naming each
    shard's cash and their total.

    The raw variant is used, never the modeled client.get_balance(): the
    pinned SDK's GetBalanceResponse model types balance / portfolio_value /
    updated_ts as strict required ints, so the modeled call raises a pydantic
    ValidationError as soon as a reply leaves one of them out (see the
    CLAUDE.md API-drift gotcha). The body is parsed here instead — the cash
    by _balance_cents_by_shard, the positions value by _positions_value_cents.

    main._run_prod calls it before trading (it sizes on the cash plus the
    positions' value, and spends only the cash) and again after trading (the
    cash for the trade log). verify_auth() is this read returning only the
    cash.

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        AccountBalance: Each shard's spendable cash in whole cents (e.g.
            {0: 100000, 1: 5000} = $1,000.00 on shard 0 and $50.00 on
            shard 1), and the positions' value in whole cents, or None when
            the reply carries no readable value.

    Raises:
        ApiException: If the request returns a non-2xx status (e.g. 401
            Unauthorized when credentials are wrong), after any retries.
        ValueError: If the response body carries no parseable cash balance.
            A missing or unreadable positions value never raises.
        Exception: Any other exception raised by the underlying HTTP client
            (e.g. a network error that outlived the retry budget).
    """
    # Retried read-only GET on the raw variant; fetch_json_page turns a
    # non-2xx into ApiException, so bad credentials still fail loudly
    data = api_call_with_retry(fetch_json_page, client.get_balance_without_preload_content)
    # Cash first: a body with no parseable cash raises here
    shard_balances = _balance_cents_by_shard(data)
    logging.info(
        "Auth OK — balance by shard: %s (total $%.2f)",
        shard_balances,
        sum(shard_balances.values()) / 100,
    )
    return AccountBalance(shard_balances, _positions_value_cents(data))


def verify_auth(client: KalshiClient) -> dict[int, int]:
    """
    Confirm the client's credentials work and return the cash on each
    exchange shard.

    Makes exactly the read read_account_balance() makes — one retried
    GET /portfolio/balance and the same "Auth OK" log line — and returns only
    its per-shard cash. The human-run verification CLI reads each shard's
    cash with it; main.py calls read_account_balance() instead, since it
    also needs the positions' value.

    There is deliberately no scalar-returning variant: the dict is the single
    source of truth, and a caller that needs one number sums it explicitly.
    A single number here would invite a future caller to size against one
    shard's cash instead of the whole account's.

    Args:
        client (KalshiClient): An authenticated client produced by build_client().

    Returns:
        dict[int, int]: Spendable cash in whole cents, keyed by
            exchange_index (e.g. {0: 100000, 1: 5000} = $1,000.00 on shard 0
            and $50.00 on shard 1).

    Raises:
        ApiException: If the request returns a non-2xx status (e.g. 401
            Unauthorized when credentials are wrong).
        ValueError: If the response body carries no parseable cash balance.
        Exception: Any other exception raised by the underlying HTTP client
            (e.g. a network error that outlived the retry budget).
    """
    # The one balance read and its log line; this caller needs only the cash
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
