"""
File: test_auth.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline unit tests for kalshi_betting.auth — client construction (dev-key
    fallback semantics) and the shard-aware balance parsing added when Kalshi
    scoped GET /portfolio/balance per exchange shard (2026-08-13). Covers the
    BS-19 fix (an eager .get() default crashed dev-only setups), the BS-10 fix
    (verify_auth's modeled get_balance() raises pydantic ValidationError on
    live 2026-07+ responses, so it reads the raw body instead), the
    dollar-string -> floored-cents converter, the tiered fallback chain in
    _balance_cents_by_shard (full breakdown -> aggregate balance_dollars ->
    legacy integer cents), verify_auth end-to-end against a faked raw
    response, and read_account_balance — the same one read returning each
    shard's cash together with Kalshi's value of the open positions
    (_positions_value_cents: integer cents, or a dollar string by presence,
    None whenever the reply carries nothing readable or a value outside
    0 to 2**53 cents).

Dependencies:
    Imports the kalshi_betting.auth and kalshi_betting._http modules
    (build_client is called as auth.build_client; _http.time.sleep is
    patched in the retry tests), the names verify_auth, read_account_balance,
    AccountBalance, _balance_cents_by_shard, _positions_value_cents and
    _dollar_str_to_cents from kalshi_betting.auth, and DEFAULT_EXCHANGE_INDEX
    from kalshi_betting.config. Patches kalshi_betting.auth.SECRETS_FILE /
    PEM_FILE / DEV_PEM_FILE (module-level names, imported directly from
    config.py) with tmp_path fixture files, and patches
    kalshi_betting.auth.KalshiClient with a MagicMock so no real client is
    constructed. Uses unittest.mock / SimpleNamespace to stand in for the SDK
    client and its RESTResponse — no network access.

Notes:
    build_client() monkey-patches cfg.api_key_id / cfg.private_key_pem onto a
    real Configuration object (see the CLAUDE.md "KalshiClient monkey-patch
    pattern" gotcha) — that pattern is intentional and must NOT be "fixed";
    these tests patch KalshiClient itself instead, so Configuration is still
    built for real (proving cfg.api_key_id/cfg.private_key_pem get set) but
    never handed to a live client.

    The same-key-different-units trap is asserted explicitly: inside a
    balance_breakdown entry "balance" is a fixed-point DOLLAR STRING, while the
    TOP-LEVEL "balance" is legacy INTEGER CENTS. A test pins each.

    verify_auth() returns the FULL per-shard dict, not a single scalar — the
    old "shard-0 preferred over aggregate" behavior inverts here: shard-1
    funds are now visible in the result, not excluded from it. Sizing is
    portfolio-wide and lives in main.py (sum(shard_balances.values())), which
    is exercised in test_main.py, not here.

    The balance reply's top-level portfolio_value is integer cents and is the
    open positions' value alone — it never includes cash. The two
    HOLDING_PAYLOADS replies below, taken while positions were held, show it
    sitting under the cash balance, which a figure that included cash could
    not do; the pinned SDK's model describes it as "the current value of all
    positions held".
"""
import dataclasses
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import _http, auth
from kalshi_betting.auth import (
    AccountBalance,
    _balance_cents_by_shard,
    _dollar_str_to_cents,
    _positions_value_cents,
    read_account_balance,
    verify_auth,
)
from kalshi_betting.config import DEFAULT_EXCHANGE_INDEX


def _write_secrets(tmp_path, payload: dict):
    """
    Write a secrets.json-shaped fixture file and point auth.SECRETS_FILE at it.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        payload (dict): Contents to serialize as the secrets file.

    Returns:
        pathlib.Path: Path to the written file.
    """
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps(payload))
    return path


def balance_resp(status: int, body: dict) -> MagicMock:
    """
    Build a fake raw get_balance_without_preload_content response.

    Mirrors the .status / .data pattern used by fetch_json_page and by the
    fake responses in tests/test_http.py — the raw SDK variants return an
    object with a .status int and a .data bytes body rather than a modeled
    pydantic object.

    Args:
        status (int): HTTP status code.
        body (dict): JSON-serializable response body.

    Returns:
        MagicMock: Object with .status and .data set.
    """
    resp = MagicMock()
    resp.status = status
    resp.data = json.dumps(body).encode("utf-8")
    return resp


class TestBuildClientDevKeyFallback:
    """BS-19: dev_api_key is optional and must not be evaluated eagerly."""

    def test_dev_only_secrets_builds_client(self, tmp_path, monkeypatch):
        # Only dev_api_key present — the old eager .get() default evaluated
        # secrets["Kalshi-api-key"] regardless, raising KeyError even though
        # dev_api_key existed. This must now succeed.
        _write_secrets(tmp_path, {"dev_api_key": "dev-key-123"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")
        dev_pem = tmp_path / "kalshi_demo_private_key.pem"  # deliberately absent

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", dev_pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("dev")

        # cfg is the sole positional/keyword arg passed to KalshiClient(...)
        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "dev-key-123"
        assert cfg.private_key_pem == "prod-pem"  # falls back to PEM_FILE (no dev PEM)

    def test_missing_dev_key_falls_back_to_prod_key(self, tmp_path, monkeypatch):
        # No dev_api_key at all — dev mode must fall back to Kalshi-api-key
        # rather than raising.
        _write_secrets(tmp_path, {"Kalshi-api-key": "prod-key-456"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")
        dev_pem = tmp_path / "kalshi_demo_private_key.pem"

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", dev_pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("dev")

        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "prod-key-456"

    def test_neither_key_raises_keyerror(self, tmp_path, monkeypatch):
        _write_secrets(tmp_path, {"some_other_field": "irrelevant"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)
        monkeypatch.setattr(auth, "DEV_PEM_FILE", tmp_path / "no_such_dev.pem")

        with pytest.raises(KeyError):
            auth.build_client("dev")

    def test_prod_mode_requires_kalshi_api_key(self, tmp_path, monkeypatch):
        # dev_api_key present is irrelevant in prod mode — prod always requires
        # Kalshi-api-key specifically.
        _write_secrets(tmp_path, {"dev_api_key": "dev-key-123"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)

        with pytest.raises(KeyError):
            auth.build_client("prod")

    def test_prod_mode_builds_client_with_kalshi_api_key(self, tmp_path, monkeypatch):
        _write_secrets(tmp_path, {"Kalshi-api-key": "prod-key-789"})
        pem = tmp_path / "kalshi_private_key.pem"
        pem.write_text("prod-pem-text")

        monkeypatch.setattr(auth, "SECRETS_FILE", tmp_path / "secrets.json")
        monkeypatch.setattr(auth, "PEM_FILE", pem)

        fake_client_cls = MagicMock()
        with patch.object(auth, "KalshiClient", fake_client_cls):
            auth.build_client("prod")

        cfg = fake_client_cls.call_args.kwargs["configuration"]
        assert cfg.api_key_id == "prod-key-789"
        assert cfg.private_key_pem == "prod-pem-text"


# The exact live body observed on this account 2026-08-14. Top-level "balance"
# is integer cents; the breakdown entries carry dollar strings under "balance".
LIVE_PAYLOAD = {
    "balance": 114,
    "balance_breakdown": [
        {"balance": "1.1407", "exchange_index": 0},
        {"balance": "0.0000", "exchange_index": 1},
    ],
    "balance_dollars": "1.1407",
    "portfolio_value": 39796,
    "updated_ts": 1786699278,
}


def _raw_balance_response(payload: dict, status: int = 200) -> SimpleNamespace:
    """Build a raw-response stand-in for get_balance_without_preload_content.

    verify_auth bypasses the SDK's strict balance model and parses the JSON
    body itself, so mocks provide (status, data-bytes) exactly like the SDK's
    RESTResponse.
    """
    return SimpleNamespace(
        status=status,
        data=json.dumps(payload).encode("utf-8"),
        getheaders=lambda: {},
        reason="OK" if status == 200 else "Error",
    )


class TestDollarStrToCents:
    def test_live_fixed_point_string_is_floored(self):
        # "1.1407" is 114.07 cents — flooring must never report the extra cent
        # the account cannot actually spend.
        assert _dollar_str_to_cents("1.1407") == 114

    def test_sub_cent_precision_floors_not_rounds(self):
        # 12345.6 cents rounds to 12346 but must floor to 12345.
        assert _dollar_str_to_cents("123.456") == 12345

    def test_zero(self):
        assert _dollar_str_to_cents("0.0000") == 0

    def test_none_returns_none(self):
        assert _dollar_str_to_cents(None) is None

    def test_garbage_returns_none(self):
        assert _dollar_str_to_cents("garbage") is None

    def test_float_input_handled(self):
        assert _dollar_str_to_cents(1.14) == 114

    def test_int_input_handled(self):
        assert _dollar_str_to_cents(2) == 200


class TestBalanceCentsByShard:
    def test_live_payload_returns_all_shards(self):
        # Every shard is now visible — shard 1's $0.00 is included, not
        # silently excluded the way the old shard-0-only lookup would have.
        assert _balance_cents_by_shard(LIVE_PAYLOAD) == {0: 114, 1: 0}

    def test_multi_shard_both_nonzero(self):
        payload = {
            "balance_breakdown": [
                {"balance": "5.00", "exchange_index": 0},
                {"balance": "100.00", "exchange_index": 1},
            ],
            "balance_dollars": "105.00",
            "balance": 10500,
        }
        # Both shards' funds are visible — unlike the pre-multi-shard reader,
        # shard-1 collateral is no longer dropped from the result.
        assert _balance_cents_by_shard(payload) == {0: 500, 1: 10000}

    def test_no_breakdown_falls_back_to_balance_dollars(self):
        assert _balance_cents_by_shard({"balance_dollars": "42.50"}) == {
            DEFAULT_EXCHANGE_INDEX: 4250
        }

    def test_legacy_integer_payload_is_already_cents(self):
        # Sandbox shape. 5000 means $50.00 — running it through the dollar
        # converter would report $5,000.00.
        assert _balance_cents_by_shard({"balance": 5000}) == {
            DEFAULT_EXCHANGE_INDEX: 5000
        }

    def test_breakdown_with_no_parseable_entries_warns_and_uses_aggregate(self, caplog):
        payload = {
            "balance_breakdown": [{"exchange_index": "not-an-int", "balance": "100.00"}],
            "balance_dollars": "100.00",
        }
        with caplog.at_level(logging.WARNING):
            assert _balance_cents_by_shard(payload) == {DEFAULT_EXCHANGE_INDEX: 10000}
        assert any(
            "balance_breakdown" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_malformed_entries_do_not_break_good_entries(self):
        payload = {
            "balance_breakdown": [
                "not-a-dict",
                None,
                {"exchange_index": None, "balance": "9.99"},
                {"exchange_index": "abc", "balance": "8.88"},
                {"balance": "7.77"},
                {"balance": "3.50", "exchange_index": DEFAULT_EXCHANGE_INDEX},
                {"balance": "6.25", "exchange_index": 1},
            ],
            "balance_dollars": "999.00",
        }
        # Only the two well-formed entries survive; the malformed ones are
        # skipped without aborting the scan of the good ones.
        assert _balance_cents_by_shard(payload) == {
            DEFAULT_EXCHANGE_INDEX: 350,
            1: 625,
        }

    def test_unparseable_entry_balance_is_skipped(self):
        # The only entry present is unparseable, so the breakdown yields
        # nothing and falls all the way through to the aggregate.
        payload = {
            "balance_breakdown": [{"balance": "garbage", "exchange_index": 0}],
            "balance_dollars": "12.00",
        }
        assert _balance_cents_by_shard(payload) == {DEFAULT_EXCHANGE_INDEX: 1200}

    def test_one_unparseable_entry_among_others_is_just_dropped(self):
        # A mix of one bad entry and one good entry: the good entry alone
        # is returned — no fallback to the aggregate, since something parsed.
        payload = {
            "balance_breakdown": [
                {"balance": "garbage", "exchange_index": 0},
                {"balance": "2.00", "exchange_index": 1},
            ],
            "balance_dollars": "999.00",
        }
        assert _balance_cents_by_shard(payload) == {1: 200}

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_by_shard({})

    def test_nonsense_payload_raises(self):
        with pytest.raises(ValueError):
            _balance_cents_by_shard({"nonsense": 1})

    def test_non_int_legacy_balance_raises(self):
        # A bool is an int subclass but a string is not — a stringy top-level
        # balance is not silently treated as cents.
        with pytest.raises(ValueError):
            _balance_cents_by_shard({"balance": "114"})


class TestVerifyAuth:
    def test_returns_full_shard_dict_from_raw_response(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response(LIVE_PAYLOAD)
        )

        result = verify_auth(client)

        assert result == {0: 114, 1: 0}
        assert isinstance(result, dict)
        # The modeled call must not be used — it deserializes through the
        # pinned SDK's strict pydantic balance model.
        client.get_balance.assert_not_called()
        client.get_balance_without_preload_content.assert_called_once()

    def test_log_line_contains_per_shard_breakdown_and_total(self, caplog):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response(LIVE_PAYLOAD)
        )
        with caplog.at_level(logging.INFO):
            verify_auth(client)
        assert any(
            "0: 114" in rec.message and "1: 0" in rec.message and "1.14" in rec.message
            for rec in caplog.records
        )

    def test_legacy_sandbox_payload(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"balance": 100000})
        )
        assert verify_auth(client) == {DEFAULT_EXCHANGE_INDEX: 100000}

    def test_non_2xx_raises_api_exception(self):
        # fetch_json_page must convert non-2xx statuses into ApiException so
        # bad credentials still fail loudly rather than parsing an error body.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"error": "unauthorized"}, status=401)
        )
        with pytest.raises(ApiException):
            verify_auth(client)


class TestUnparseableEntryBalanceWarns:
    def test_dropped_shard_funds_are_never_invisible(self, caplog):
        # Regression (adversarial review): an entry with a parseable index but
        # an unparseable balance silently vanished that shard's real funds
        # from sizing, the coverage audit, and the transfer planner. The drop
        # is still the safe behavior (under-sizing), but it must be LOUD.
        import logging as _logging
        payload = {
            "balance_breakdown": [
                {"exchange_index": 0, "balance": "10.0000"},
                {"exchange_index": 2, "amount_dollars": "500.0000"},  # drifted key
            ]
        }
        with caplog.at_level(_logging.WARNING):
            out = _balance_cents_by_shard(payload)
        assert out == {0: 1000}
        assert "shard 2" in caplog.text
        assert "NOT counted" in caplog.text


class TestVerifyAuthRetryAndDrift:
    """verify_auth's transport-level behaviour, distinct from the payload-shape
    coverage in TestVerifyAuth above: the api_call_with_retry wrapper it goes
    through, and the loud failure required of an unrecognized response shape.

    Both cases assert the SHARD DICT return type — verify_auth is no longer
    scalar-returning, so a 429 that recovers must yield {shard: cents}, not an
    int."""

    def test_retries_429_then_succeeds(self):
        # A transient 429 on a read-only GET must be retried, not fatal —
        # verify_auth is wrapped in api_call_with_retry (unlike order
        # submission and read_shard_balances, both deliberately single-shot).
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            side_effect=[
                balance_resp(429, {"error": "slow down"}),
                balance_resp(200, {"balance": 100000}),
            ]
        )
        with patch.object(_http.time, "sleep") as sleep:
            assert auth.verify_auth(client) == {DEFAULT_EXCHANGE_INDEX: 100000}
        assert client.get_balance_without_preload_content.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_unknown_shape_raises(self):
        # No recognizable balance field at all — must fail loudly rather than
        # silently returning a wrong number. _balance_cents_by_shard raises
        # ValueError once every fallback tier has been exhausted.
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=balance_resp(200, {"totally_unexpected_field": 1})
        )
        with patch.object(_http.time, "sleep"):
            with pytest.raises(ValueError):
                auth.verify_auth(client)


class TestBalanceFieldChosenByPresence:
    """TS-16: the breakdown field is chosen by PRESENCE, not truthiness.

    `entry.get("balance_dollars") or entry.get("balance")` discarded a numeric
    zero and read the stale legacy field instead, reporting a genuinely-empty
    shard as funded. That is the one OVER-statement this function can produce,
    in a module whose flooring rule and drop-with-a-warning rule both exist to
    guarantee the opposite. It feeds Kelly sizing, the MIN_BALANCE_CENTS gate,
    the shard-coverage audit and the transfer planner.
    """

    def test_numeric_zero_balance_dollars_is_honoured_not_skipped(self):
        payload = {"balance_breakdown": [
            {"exchange_index": 0, "balance_dollars": 0, "balance": "5.00"},
        ]}
        assert _balance_cents_by_shard(payload) == {0: 0}

    def test_float_zero_balance_dollars_is_honoured(self):
        payload = {"balance_breakdown": [
            {"exchange_index": 0, "balance_dollars": 0.0, "balance": "5.00"},
        ]}
        assert _balance_cents_by_shard(payload) == {0: 0}

    def test_empty_balance_dollars_reaches_the_warning_not_the_legacy_field(self, caplog):
        # "" is unparseable, not zero — it must drop the shard LOUDLY through
        # the existing warning rather than silently reading another field.
        payload = {
            "balance_breakdown": [
                {"exchange_index": 0, "balance_dollars": "", "balance": "5.00"},
            ],
            "balance_dollars": "7.00",
        }
        with caplog.at_level(logging.WARNING):
            out = _balance_cents_by_shard(payload)
        assert out == {DEFAULT_EXCHANGE_INDEX: 700}
        assert "NOT counted" in caplog.text

    def test_string_zero_still_reads_as_zero(self):
        # Regression guard: the string form was always correct (it is truthy),
        # which is exactly why the live payload never exposed the bug.
        payload = {"balance_breakdown": [
            {"exchange_index": 0, "balance_dollars": "0.0000"},
        ]}
        assert _balance_cents_by_shard(payload) == {0: 0}

    def test_duplicate_exchange_index_warns_and_keeps_last(self, caplog):
        # Last-win matches the previous behaviour and is as defensible as
        # first-win; being SILENT about a changed payload shape is not.
        payload = {"balance_breakdown": [
            {"exchange_index": 0, "balance": "1.00"},
            {"exchange_index": 0, "balance": "9.00"},
        ]}
        with caplog.at_level(logging.WARNING):
            out = _balance_cents_by_shard(payload)
        assert out == {0: 900}
        assert "more than once" in caplog.text

    def test_dollar_converter_distinguishes_zero_from_unparseable(self):
        # The converter was never the problem — pin that, so the fix can't be
        # "corrected" back into the caller.
        assert _dollar_str_to_cents(0) == 0
        assert _dollar_str_to_cents(0.0) == 0
        assert _dollar_str_to_cents("") is None


# The top-level fields of two live balance replies from this account while it
# held open positions. Their balance_breakdown was left out when they were
# printed, so here the cash comes from the balance_dollars aggregate on the
# default shard. portfolio_value sits below the cash: it is the positions'
# value alone.
HOLDING_PAYLOADS = [
    (
        {"balance": 13245, "balance_dollars": "132.4521",
         "portfolio_value": 7850, "updated_ts": 1790587734},
        13245, 7850,
    ),
    (
        {"balance": 11615, "balance_dollars": "116.1584",
         "portfolio_value": 9527, "updated_ts": 1790613502},
        11615, 9527,
    ),
]


def _balance_client(payload: dict) -> MagicMock:
    """
    Build a client stand-in whose raw balance call answers `payload` with 200.

    Args:
        payload (dict): The JSON body the balance read returns.

    Returns:
        MagicMock: A client whose get_balance_without_preload_content returns
            a raw 200 response carrying `payload`.
    """
    client = MagicMock()
    client.get_balance_without_preload_content = MagicMock(
        return_value=_raw_balance_response(payload)
    )
    return client


class TestReadAccountBalance:
    """read_account_balance: one retried GET /portfolio/balance that returns
    each shard's cash and Kalshi's value of the open positions.

    The value is meant to be added to the cash to get the amount trades are
    sized on, so it must be read in the right unit (integer cents, or a
    dollar string floored like every balance), must never be mistaken for a
    cash figure, and must come back None — never 0, never an error, never a
    slow conversion — when the reply carries nothing readable, so the caller
    can fall back to cash alone.
    verify_auth makes this same read and returns only the cash, so both keep
    one log line and one request."""

    def test_the_live_reply_gives_cash_by_shard_and_the_positions_value(self):
        # The cash parse is unchanged; portfolio_value comes back as is, in cents
        client = _balance_client(LIVE_PAYLOAD)
        assert read_account_balance(client) == AccountBalance({0: 114, 1: 0}, 39796)

    @pytest.mark.parametrize("payload, cash, positions", HOLDING_PAYLOADS)
    def test_replies_holding_positions_read_the_value_as_sent(
        self, payload, cash, positions,
    ):
        # The value sits below the cash, so it cannot include it: it is read
        # as sent and never added to or taken from the cash
        result = read_account_balance(_balance_client(payload))
        assert result.shard_cash_cents == {DEFAULT_EXCHANGE_INDEX: cash}
        assert result.positions_value_cents == positions
        assert positions < cash

    def test_one_get_on_the_raw_variant(self):
        # The modeled get_balance deserializes through the SDK's strict model
        client = _balance_client(LIVE_PAYLOAD)
        read_account_balance(client)
        client.get_balance_without_preload_content.assert_called_once()
        client.get_balance.assert_not_called()

    def test_the_auth_ok_line_is_unchanged_and_logged_once(self, caplog):
        with caplog.at_level(logging.INFO):
            read_account_balance(_balance_client(LIVE_PAYLOAD))
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Auth OK")]
        assert lines == ["Auth OK — balance by shard: {0: 114, 1: 0} (total $1.14)"]

    def test_verify_auth_is_the_same_read_returning_only_the_cash(self, caplog):
        # One request and one "Auth OK" line, whichever function is called
        client = _balance_client(LIVE_PAYLOAD)
        with caplog.at_level(logging.INFO):
            assert verify_auth(client) == {0: 114, 1: 0}
        client.get_balance_without_preload_content.assert_called_once()
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Auth OK")]
        assert lines == ["Auth OK — balance by shard: {0: 114, 1: 0} (total $1.14)"]

    def test_verify_auth_returns_the_cash_of_read_account_balance(self, monkeypatch):
        # One definition of the read: verify_auth hands back exactly the cash
        # dict read_account_balance built, not a second parse of its own
        cash = {0: 1, 3: 2}
        monkeypatch.setattr(auth, "read_account_balance",
                            lambda client: AccountBalance(cash, 5))
        assert verify_auth(MagicMock()) is cash

    def test_a_429_is_retried(self):
        # A read-only GET: a transient 429 must not end the run before it scans
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            side_effect=[
                balance_resp(429, {"error": "slow down"}),
                balance_resp(200, LIVE_PAYLOAD),
            ]
        )
        with patch.object(_http.time, "sleep") as sleep:
            assert read_account_balance(client) == AccountBalance({0: 114, 1: 0}, 39796)
        assert client.get_balance_without_preload_content.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_bad_credentials_raise(self):
        client = MagicMock()
        client.get_balance_without_preload_content = MagicMock(
            return_value=_raw_balance_response({"error": "unauthorized"}, status=401)
        )
        with pytest.raises(ApiException):
            read_account_balance(client)

    def test_unparseable_cash_raises_whatever_the_positions_value(self):
        # Sizing on an unknown cash balance is worse than stopping the run
        client = _balance_client({"portfolio_value": 500, "nonsense": 1})
        with pytest.raises(ValueError):
            read_account_balance(client)

    def test_an_unreadable_positions_value_does_not_fail_the_read(self, caplog):
        # The cash parsed, so the read succeeds with no value and no WARNING:
        # the caller decides what sizing on cash alone means
        client = _balance_client({"balance": 100, "portfolio_value_dollars": "Infinity"})
        with caplog.at_level(logging.INFO):
            result = read_account_balance(client)
        assert result == AccountBalance({DEFAULT_EXCHANGE_INDEX: 100}, None)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_the_result_is_frozen(self):
        result = read_account_balance(_balance_client(LIVE_PAYLOAD))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.positions_value_cents = 0

    def test_integer_cents_are_read_as_is_and_zero_is_an_answer(self):
        assert _positions_value_cents({"portfolio_value": 39796}) == 39796
        # Nothing held reads 0, not None: a real answer, not a missing one
        assert _positions_value_cents({"portfolio_value": 0}) == 0

    @pytest.mark.parametrize("dollars, cents", [
        ("78.5099", 7850),   # floored, never rounded up to 7851
        ("0", 0),
        ("0.0000", 0),
        (0, 0),              # a falsy answer is still an answer
        ("132.4521", 13245),
    ])
    def test_the_dollar_string_wins_by_presence(self, dollars, cents):
        # The integer field beside it is ignored whenever the dollar key is present
        payload = {"portfolio_value_dollars": dollars, "portfolio_value": 1}
        assert _positions_value_cents(payload) == cents

    @pytest.mark.parametrize("dollars", [
        "", None, "garbage", "-1.00", "-0.01", True, False, "NaN", "sNaN",
        "Infinity", "-Infinity", "1e999999", "-1e999999", float("inf"),
        float("nan"), [7850], {"v": "78.50"},
        "1e400", 1e308,            # finite, but beyond the bound
        "1e999990",                # far beyond it: no million-digit integer is built
        "90071992547409.93",       # one cent above 2**53 cents
    ])
    def test_an_unreadable_dollar_string_is_none_and_never_falls_back(self, dollars):
        # Chosen by presence: an unreadable dollar value is no value, and the
        # integer field is not consulted, so one reply's two spellings are
        # never mixed; a non-finite, overflowing or out-of-range value, or
        # one that is not a string or a plain number, raises nothing
        payload = {"portfolio_value_dollars": dollars, "portfolio_value": 500}
        assert _positions_value_cents(payload) is None

    def test_the_bound_is_2_to_the_53_cents_on_both_spellings(self):
        # Every whole number of cents up to 2**53 is exact in a float; the
        # bound itself is accepted and one cent more is not
        bound = 2 ** 53
        assert _positions_value_cents({"portfolio_value": bound}) == bound
        assert _positions_value_cents({"portfolio_value": bound + 1}) is None
        assert _positions_value_cents({"portfolio_value": 10 ** 400}) is None
        assert _positions_value_cents(
            {"portfolio_value_dollars": "90071992547409.92"}) == bound
        # Negative zero is zero, and a whole dollar amount spelled as a number reads
        assert _positions_value_cents({"portfolio_value_dollars": "-0.00"}) == 0
        assert _positions_value_cents({"portfolio_value_dollars": 78}) == 7800

    @pytest.mark.parametrize("dollars", ["1e999990", "1e400", "-1e999999", "Infinity"])
    def test_an_out_of_range_dollar_value_is_refused_before_any_conversion(
        self, dollars, monkeypatch,
    ):
        # Converting "1e999990" to cents would build an integer of a million
        # digits, which takes many seconds; the range check comes first, so
        # the converter is never reached
        def must_not_convert(value):
            raise AssertionError(f"converted {value!r}")
        monkeypatch.setattr(auth, "_dollar_str_to_cents", must_not_convert)
        assert _positions_value_cents({"portfolio_value_dollars": dollars}) is None

    def test_a_deeply_nested_dollar_value_is_never_turned_into_text(self):
        # Turning a deeply nested list into text raises RecursionError; the
        # reader refuses a list by its type first, so the value is None and the
        # balance read around it still returns the cash
        nested: list = []
        for _ in range(100_000):
            nested = [nested]
        assert _positions_value_cents({"portfolio_value_dollars": nested}) is None
        # The body is handed over already parsed: no JSON text can nest this deep
        client = MagicMock()
        with patch.object(auth, "fetch_json_page", return_value={
            "balance": 100, "portfolio_value_dollars": nested,
        }):
            assert verify_auth(client) == {DEFAULT_EXCHANGE_INDEX: 100}
            assert read_account_balance(client) == AccountBalance(
                {DEFAULT_EXCHANGE_INDEX: 100}, None)

    @pytest.mark.parametrize("body", [
        {"balance": 100, "portfolio_value_dollars": "1e999990"},
        {"balance": 100, "portfolio_value_dollars": "Infinity"},
        {"balance": 100, "portfolio_value": 2 ** 60},   # a whole number past the bound
    ])
    def test_verify_auth_returns_the_cash_whatever_the_positions_value(self, body):
        # verify_auth reads the positions value only to discard it, so no
        # value in that field can fail or slow down the cash it returns
        assert verify_auth(_balance_client(body)) == {DEFAULT_EXCHANGE_INDEX: 100}

    @pytest.mark.parametrize("payload", [
        {},
        {"portfolio_value": None},
        {"portfolio_value": "397.96"},   # a string where integer cents belong
        {"portfolio_value": "7850"},
        {"portfolio_value": 7850.0},
        {"portfolio_value": True},       # an int subclass, not one cent
        {"portfolio_value": False},
        {"portfolio_value": -1},
        {"portfolio_value": [7850]},
        {"portfolio_value": {}},
    ])
    def test_an_unreadable_integer_field_is_none(self, payload):
        assert _positions_value_cents(payload) is None
