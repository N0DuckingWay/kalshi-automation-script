"""
File: v2_probe.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    USER-RUN LIVE VERIFICATION CLI — this module is never imported by the
    trading pipeline and must be invoked by a human, deliberately, from a
    terminal:

        python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step <name>] [--yes]

    It exists to satisfy the one gate on config.V2_ORDERS_ENABLED. The V2 order
    path in trader.py is complete and unit-tested, but its NO-leg mapping is an
    unverified HYPOTHESIS that cannot be proven offline: V2 has no
    side="no"/action="buy" vocabulary, so "buy NO at <= p" is mapped onto an
    `ask` (sell YES) at >= 1 - p, and closing that NO position onto a `bid`
    carrying reduce_only=True. If the hypothesis is wrong the bot would take
    the WRONG SIDE of the market, not merely a wrong price. This probe submits
    the smallest possible real orders through the EXACT trader functions the
    live path would use (trader._no_buy_as_v2, _no_close_as_v2,
    _build_v2_order_body, _submit_order_v2) and reports whether the account
    position moved the way the hypothesis predicts.

    *** REAL MONEY. *** Every step here submits live orders (or moves live
    collateral) against the PRODUCTION account. Worst-case exposure is the V2
    minimum fractional count of 0.01 contracts — roughly one cent — but it is
    real, it is not a simulation, and there is no dry-run mode: a probe that
    doesn't submit proves nothing. Every request body and every raw response is
    printed verbatim, and that printed output IS the evidence log that gates
    the V2_ORDERS_ENABLED flip.

Dependencies:
    Imports the auth, config, scanner and trader MODULES (module-style, so the
    probe always calls the same functions the live path calls and tests can
    monkeypatch them at their definition site), plus api_call_with_retry /
    fetch_json_page from _http.py for the single read-only market lookup.

    NOTHING IMPORTS THIS MODULE. It is not referenced by main.py, scheduler.py,
    trader.py, or any other pipeline module, and tests/test_v2_probe.py asserts
    that stays true. Adding an import of v2_probe anywhere in the pipeline would
    put real-money probe orders one code path away from the weekly scheduler.

Notes:
    PROD ONLY. auth.build_client("prod") is hardcoded — the sandbox implements
    neither the V2 order endpoint nor exchange sharding, so a sandbox "pass"
    would prove nothing about the mapping this probe exists to verify.

    COUNT OVERRIDE. trader._build_v2_order_body() takes `count: int` and
    renders "<n>.00", because the bot sizes in whole contracts. The probe wants
    the V2 minimum of 0.01 contracts, so _probe_body() builds the body through
    the real builder and then overrides the one field — see the loud comment
    there. trader.py is deliberately NOT modified for the probe's benefit: the
    thing being verified must stay byte-identical to the thing that will run.

    Likewise the fill test: trader._v2_filled() takes an int count, so the
    probe re-states its rule inline for a fractional count (_filled_count /
    _remaining_count), keeping the same Decimal comparison and the same
    "unparseable means ambiguous" semantics.

    CONFIRMATION. Nothing is submitted until the request body has been printed
    and the operator has typed "yes" — unless --yes was passed, which is for a
    second or third run once the operator has already seen the bodies.

    EXIT CODES: 0 = every executed step PASSED, 1 = something FAILED, 2 =
    NEUTRAL (the step could not be run to a conclusion — no fill, no liquidity,
    aborted at the prompt, or a skipped transfer step). Only a 0 from BOTH
    mapping steps (--step no-mapping and --step unfillable-ask) may be used to
    justify flipping config.V2_ORDERS_ENABLED.
"""
import argparse
import json
import logging
import sys
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any

from . import auth, config, scanner, trader
from ._http import api_call_with_retry, fetch_json_page

# The V2 minimum fractional contract count. This lives here rather than in
# config.py on purpose: it is a property of THIS diagnostic (make the exposure
# as close to zero as the exchange allows), not a tunable of the trading
# strategy, and nothing in the pipeline may ever size an order from it.
PROBE_COUNT_STR = "0.01"
PROBE_COUNT = Decimal(PROBE_COUNT_STR)

# Shards used by the transfer step. Source is the shard everything currently
# lives on; destination is the first shard Kalshi actually migrated markets to
# (combos, 2026-08-17). One cent is the smallest amount that is still a
# non-zero movement in the endpoint's centicent unit after conversion.
_TRANSFER_SOURCE_SHARD = config.DEFAULT_EXCHANGE_INDEX
_TRANSFER_DEST_SHARD = 1
_TRANSFER_PROBE_CENTS = 1

# Step outcomes. PASS/FAIL are verdicts on the hypothesis; NEUTRAL means the
# step never reached a verdict (and so proves nothing either way).
_PASS = "PASS"
_FAIL = "FAIL"
_NEUTRAL = "NEUTRAL"

_EXIT_CODES = {_PASS: 0, _FAIL: 1, _NEUTRAL: 2}


def _emit(label: str, payload: Any) -> None:
    """
    Print one labelled JSON block of the evidence log.

    Every request body and every raw response goes through here, pretty-printed
    with indent=2, because this output is the artifact a human reads when
    deciding whether to flip config.V2_ORDERS_ENABLED.

    Args:
        label (str): Short human-readable name for the block, e.g.
            "REQUEST BODY (NO buy)".
        payload (Any): Anything json.dumps can render; non-serializable values
            (e.g. dataclasses) fall back to their str() form rather than
            raising, since a probe must never die while printing evidence.

    Returns:
        None
    """
    print(f"\n----- {label} -----")
    print(json.dumps(payload, indent=2, default=str))


def _confirm(assume_yes: bool, summary: str) -> bool:
    """
    Show a plain-English summary of what is about to be submitted and gate on it.

    This is the only thing standing between the operator and a real order, so it
    runs BEFORE every submission and every transfer, after the exact request
    body has already been printed.

    Args:
        assume_yes (bool): True when --yes was passed, which skips the
            interactive prompt (and says so, so the log still records that no
            human confirmed this particular submission).
        summary (str): Plain-English description of the submission — what
            instrument, which side, what price, what count, what it costs at
            worst.

    Returns:
        bool: True to proceed with the submission, False if the operator
            declined (anything other than exactly "yes").
    """
    print("\n*** ABOUT TO SUBMIT AGAINST THE LIVE PRODUCTION ACCOUNT — REAL MONEY ***")
    print(summary)
    if assume_yes:
        print("--yes passed: proceeding without interactive confirmation.")
        return True
    answer = input('Type "yes" to submit, anything else to abort: ').strip().lower()
    if answer != "yes":
        print("Aborted at the confirmation prompt — nothing was submitted.")
        return False
    return True


def _fetch_market(client: Any, ticker: str) -> tuple:
    """
    Fetch one market by ticker and parse it the way live ingest would.

    Read-only GET, so it goes through the shared retry wrapper. The raw payload
    is returned alongside the parsed market because the probe prints it verbatim
    — the exchange_index, price_level_structure and price_ranges fields the V2
    order path depends on are exactly the ones a human needs to eyeball.

    Args:
        client (Any): Authenticated prod KalshiClient from auth.build_client().
        ticker (str): The market ticker the operator named on the command line.

    Returns:
        tuple: (raw market dict, scanner.ApiMarket) on success, or (None, None)
            when the response carried no market object at all.
    """
    # Raw-response variant + fetch_json_page, per the project-wide rule: the
    # pinned SDK's Market model can no longer deserialize live payloads.
    data = api_call_with_retry(
        fetch_json_page, client.get_market_without_preload_content, ticker=ticker
    )
    raw = data.get("market") or {}
    if not raw:
        return None, None
    # Same parser live ingest uses, so the ApiMarket the probe hands to the
    # trader functions is constructed identically to a real trading candidate
    # (notably exchange_index and the price_ranges tick grid).
    return raw, scanner._market_from_dict(raw, "")


def _probe_body(
    ticker: str,
    side: str,
    price_str: str,
    exchange_index: int,
    *,
    reduce_only: bool = False,
) -> dict:
    """
    Build a V2 order body for the probe: the real builder, with a 0.01 count.

    Args:
        ticker (str): Market ticker to trade.
        side (str): "bid" or "ask", as produced by one of trader's _*_as_v2
            mapping functions — never assembled here.
        price_str (str): Fixed-point dollar limit price, already quantized onto
            the market's tick grid by trader._quantize_to_tick.
        exchange_index (int): The market's own shard, from ApiMarket.
        reduce_only (bool): True only for the closing order. Defaults to False.

    Returns:
        dict: The request body, identical to what the live path would send
            except for the count field.
    """
    # Built through the REAL builder so every other field (time_in_force,
    # self_trade_prevention_type, client_order_id, post_only,
    # cancel_order_on_pause, exchange_index) is byte-identical to what the live
    # path would send. count=1 here is a placeholder, immediately replaced:
    body = trader._build_v2_order_body(
        ticker, side, price_str, 1, exchange_index, reduce_only=reduce_only
    )
    # !!! DELIBERATE OVERRIDE — DO NOT "FIX" THIS BY CHANGING trader.py !!!
    # _build_v2_order_body takes `count: int` and renders "<n>.00" because the
    # bot sizes in whole contracts. The probe wants the V2 fractional minimum
    # of 0.01 contracts (~one cent of exposure), which that signature cannot
    # express. Overriding the one field here keeps trader.py — the code being
    # verified — completely untouched by the verification.
    body["count"] = PROBE_COUNT_STR
    return body


def _filled_count(data: dict) -> Decimal:
    """
    Read a V2 response's fill_count as a Decimal.

    Mirrors trader._v2_filled's parsing rule (Decimal, never float — V2 counts
    are fixed-point strings and float() would be lossy for fractional counts)
    but as a plain accessor, because _v2_filled takes an int count and the probe
    trades a fractional one.

    Args:
        data (dict): Parsed V2 order response body.

    Returns:
        Decimal: The filled count.

    Raises:
        KeyError: If fill_count is absent.
        TypeError / decimal.InvalidOperation: If it is unparseable. Same as
            trader._v2_filled, an unreadable field is treated as "we cannot
            tell what happened", never as a non-fill.
    """
    return Decimal(data["fill_count"])


def _remaining_count(data: dict) -> Decimal:
    """
    Read a V2 response's remaining_count as a Decimal.

    See _filled_count — same rule, same raise-on-unparseable semantics.

    Args:
        data (dict): Parsed V2 order response body.

    Returns:
        Decimal: The unfilled remainder.

    Raises:
        KeyError: If remaining_count is absent.
        TypeError / decimal.InvalidOperation: If it is unparseable.
    """
    return Decimal(data["remaining_count"])


def _report_fee(data: dict, price_str: str) -> None:
    """
    Print the fee the exchange actually charged next to the bot's fee model.

    INFORMATIONAL ONLY — no pass/fail. The two numbers are not directly
    comparable: config.fee_leg_exact() is defined for a whole number of
    contracts and ceilings to a whole cent, while the probe trades 0.01 of one,
    so the model figure is printed for n=1 and the reader does the scaling. The
    point is to catch an order-of-magnitude surprise in the V2 fee shape before
    real size flows through it.

    Args:
        data (dict): Parsed V2 order response body; average_fee_paid may be
            absent when nothing filled.
        price_str (str): The limit price the order was submitted at, used as
            the model's price input.

    Returns:
        None
    """
    charged = data.get("average_fee_paid")
    if charged is None:
        print("Fee check: response carried no average_fee_paid (nothing filled).")
        return
    try:
        price = float(price_str)
    except (TypeError, ValueError):
        print(f"Fee check: charged={charged} (limit price {price_str!r} unparseable)")
        return
    # config.fee_leg_exact is the bot's own fee model — the same function
    # strategy.py sizes trades against.
    modelled = config.fee_leg_exact(1, price)
    print(
        f"Fee check (informational, no verdict): exchange average_fee_paid={charged} | "
        f"config.fee_leg_exact(1, {price}) = ${modelled:.4f} for ONE whole contract "
        f"(the probe traded {PROBE_COUNT_STR})"
    )


def _step_no_mapping(client: Any, ticker: str, assume_yes: bool) -> str:
    """
    THE CORE GATE: verify that an `ask` opens a NO position and reduce_only
    closes it.

    Sequence, all of it against the live account:
      1. Fetch the market and its order book; derive the NO ask price from the
         resting YES bids (buying NO at p == selling YES at 1 - p).
      2. Confirm the account is FLAT on this ticker — the probe's whole verdict
         is "which way did the position move", which is meaningless otherwise.
      3. Submit the NO-buy body built by trader._no_buy_as_v2 (side "ask").
      4. Re-read the position. PASS half one iff it went NEGATIVE, which is
         Kalshi's unified-ledger convention for a NO position. A POSITIVE
         position means the hypothesis is WRONG — the ask opened YES exposure —
         and the closing order is then NOT submitted, because its own mapping
         (reduce_only bid nets a NO position to flat) rests on the same
         disproven assumption and would add to the wrong exposure instead.
      5. Submit the closing body from trader._no_close_as_v2 (side "bid",
         reduce_only=True). PASS half two iff the position returns to 0.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Liquid, cheap market chosen by the operator.
        assume_yes (bool): Skip the interactive confirmations (--yes).

    Returns:
        str: _PASS only when the position went negative AND came back to zero;
            _FAIL on any contrary evidence, an error, or a position left open;
            _NEUTRAL when the step never reached a verdict (no liquidity, no
            fill, or the operator declined at the prompt).
    """
    print("\n===== STEP: no-mapping (the V2 NO-leg mapping gate) =====")

    raw, market = _fetch_market(client, ticker)
    if market is None:
        print(f"{_FAIL}: no market returned for ticker {ticker!r}.")
        return _FAIL
    _emit("RAW MARKET PAYLOAD", raw)
    print(
        f"Parsed: exchange_index={market.exchange_index} "
        f"price_level_structure={market.price_level_structure!r} "
        f"price_ranges={market.price_ranges}"
    )

    # Same order-book reader the live scanner uses, including its strict
    # container/side-key generation matching.
    book = scanner._fetch_orderbook(client, ticker)
    if book is None:
        print(f"{_NEUTRAL}: order book unavailable for {ticker} — nothing to price against.")
        return _NEUTRAL
    _emit("ORDER BOOK (raw bid arrays)", book)

    # YES bids are exactly the levels a sell-YES (ask) order can hit, and the
    # scanner already expresses them as NO ask levels via the 1-P complement.
    no_levels = scanner._bids_to_ask_levels(book["yes"])
    if not no_levels:
        print(
            f"{_NEUTRAL}: no resting YES bids on {ticker} — an ask has nothing to cross "
            "with. Pick a more liquid ticker."
        )
        return _NEUTRAL
    no_price, no_qty = no_levels[0]
    print(
        f"Best NO ask (from the top YES bid): {no_price:.4f} for {no_qty} contracts "
        f"— i.e. a resting YES bid at {1.0 - no_price:.4f}"
    )

    # Ground truth for the whole verdict: the account must start flat.
    start = trader._position_count(client, ticker)
    print(f"Starting position on {ticker}: {start}")
    if start != 0:
        print(
            f"{_FAIL}: probe must start FLAT on {ticker} (position={start}; None means the "
            "lookup itself failed). Close the position or pick another ticker."
        )
        return _FAIL

    # THE HYPOTHESIS UNDER TEST — the same function the live path would call.
    side, price_str = trader._no_buy_as_v2(no_price, market)
    body = _probe_body(ticker, side, price_str, market.exchange_index)
    _emit("REQUEST BODY (NO buy — hypothesis: an ask opens a NO position)", body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill {side.upper()} (sell YES) on {ticker} for "
        f"{PROBE_COUNT_STR} contracts at a limit of {price_str} on shard "
        f"{market.exchange_index}. Under the hypothesis this BUYS {PROBE_COUNT_STR} NO "
        f"at about {no_price:.4f} — worst case about one cent of exposure. The top "
        f"resting YES bid is {1.0 - no_price:.4f}, so this limit should cross and fill.",
    ):
        return _NEUTRAL

    try:
        # Exactly the submission function the live path uses — no retry, one attempt.
        data = trader._submit_order_v2(client, body)
    except Exception as exc:
        print(f"{_FAIL}: NO-buy submission raised: {exc}")
        after_err = trader._position_count(client, ticker)
        print(f"Position after the error: {after_err}")
        if after_err:
            print(
                "*** The order may still have filled — a position is open on "
                f"{ticker}. FLATTEN IT MANUALLY in the Kalshi UI. ***"
            )
        return _FAIL
    _emit("RAW RESPONSE (NO buy)", data)

    try:
        filled = _remaining_count(data) == 0 and _filled_count(data) == PROBE_COUNT
    except (KeyError, TypeError, InvalidOperation) as exc:
        # Same reading trader._v2_filled forces on the live path: unparseable
        # fill fields mean the outcome is UNKNOWN, not that nothing filled.
        print(f"{_FAIL}: response carried no readable fill_count/remaining_count ({exc}).")
        filled = None

    after = trader._position_count(client, ticker)
    print(f"Position after the NO buy: {after}")
    _report_fee(data, price_str)

    if filled is None:
        if after:
            print(f"*** Position open on {ticker} — FLATTEN IT MANUALLY. ***")
        return _FAIL
    if not filled:
        if after == 0:
            print(
                f"{_NEUTRAL}: the fill-or-kill was killed unfilled and the account is still "
                "flat, so the mapping is unproven. Pick a more liquid ticker and re-run."
            )
            return _NEUTRAL
        print(
            f"{_FAIL}: the response reported no fill but the position is {after}. "
            "Do not trust either signal — check the account manually."
        )
        return _FAIL
    if after is None:
        print(f"{_FAIL}: the order filled but the position lookup failed — state unknown.")
        return _FAIL
    if after > 0:
        print(
            f"{_FAIL}: *** HYPOTHESIS DISPROVEN *** the ask filled and opened a POSITIVE "
            f"(YES) position of {after} on {ticker}. 'ask' does NOT open a NO position, so "
            "trader._no_buy_as_v2 takes the wrong side of the market. The reduce_only "
            "unwind is NOT being submitted — it rests on the same disproven mapping and "
            "would add to this exposure. *** FLATTEN THE POSITION MANUALLY. *** "
            "config.V2_ORDERS_ENABLED must stay False."
        )
        return _FAIL
    if after == 0:
        print(
            f"{_FAIL}: the response reported a complete fill but the position on {ticker} is "
            "still 0. Neither signal is trustworthy — check the account manually."
        )
        return _FAIL

    print(
        f"Half one CONFIRMED: the ask filled and opened a NEGATIVE position ({after}), "
        "which is Kalshi's convention for a NO position."
    )

    # Half two: does a reduce_only bid net that NO position back to flat?
    close_side, close_price = trader._no_close_as_v2(market)
    close_body = _probe_body(
        ticker, close_side, close_price, market.exchange_index, reduce_only=True
    )
    _emit("REQUEST BODY (NO close — hypothesis: reduce_only bid nets to flat)", close_body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill {close_side.upper()} on {ticker} for {PROBE_COUNT_STR} "
        f"contracts at a limit of {close_price} with reduce_only=true, to close the "
        f"{after} position just opened.",
    ):
        print(
            f"{_FAIL}: declined at the prompt while a {after} position is OPEN on {ticker}. "
            "*** FLATTEN IT MANUALLY. ***"
        )
        return _FAIL

    try:
        close_data = trader._submit_order_v2(client, close_body)
    except Exception as exc:
        print(
            f"{_FAIL}: the closing order raised: {exc}. *** A {after} position may still be "
            f"OPEN on {ticker} — CHECK THE ACCOUNT. ***"
        )
        return _FAIL
    _emit("RAW RESPONSE (NO close)", close_data)

    final = trader._position_count(client, ticker)
    print(f"Position after the close: {final}")
    if final != 0:
        print(
            f"{_FAIL}: the reduce_only bid did NOT return the position to flat (position="
            f"{final}; None means the lookup failed). *** CHECK THE ACCOUNT MANUALLY. *** "
            "config.V2_ORDERS_ENABLED must stay False."
        )
        return _FAIL

    print(
        f"{_PASS}: an ask opened a NO position (negative) and a reduce_only bid returned "
        "the account to flat. BOTH halves of the mapping in trader._no_buy_as_v2 / "
        "_no_close_as_v2 are confirmed against the live API."
    )
    return _PASS


def _step_unfillable_ask(client: Any, ticker: str, assume_yes: bool) -> str:
    """
    Verify fill-or-kill kill semantics (and the ask's price-cap direction).

    Submits an ask one tick below 1 — the highest priceable limit on the grid.
    An ask's limit is the MINIMUM proceeds accepted per contract, so demanding
    ~$0.99 to sell one YES is above any realistic resting bid and the order
    cannot fill. Two things are proven at once: that a FoK that cannot fill
    comes back killed with the full count remaining (rather than resting, or
    partially filling), and that the ask's limit really is a floor on proceeds
    — the direction trader._no_buy_as_v2's ROUND_CEILING quantization assumes.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Market to submit against.
        assume_yes (bool): Skip the interactive confirmation (--yes).

    Returns:
        str: _PASS when the order came back with zero filled, the full count
            remaining, and the account still flat; _FAIL if anything filled or
            the response/position disagrees; _NEUTRAL if the operator declined
            or the market could not be read.
    """
    print("\n===== STEP: unfillable-ask (fill-or-kill kill semantics) =====")

    raw, market = _fetch_market(client, ticker)
    if market is None:
        print(f"{_FAIL}: no market returned for ticker {ticker!r}.")
        return _FAIL
    _emit("RAW MARKET PAYLOAD", raw)

    start = trader._position_count(client, ticker)
    print(f"Starting position on {ticker}: {start}")
    if start != 0:
        print(f"{_FAIL}: probe must start FLAT on {ticker} (position={start}).")
        return _FAIL

    # Top of the market's own grid, via the same quantizer the order path uses:
    # _quantize_to_tick clamps to one step below 1, so asking for 1 yields the
    # highest priceable tick (e.g. "0.99" on a whole-cent market).
    price_str = trader._fp(trader._quantize_to_tick(Decimal(1), market, ROUND_FLOOR))
    body = _probe_body(ticker, "ask", price_str, market.exchange_index)
    _emit("REQUEST BODY (deliberately unfillable ask)", body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill ASK on {ticker} for {PROBE_COUNT_STR} contracts at a limit "
        f"of {price_str} — one tick below 1, i.e. demanding almost the full dollar to sell "
        "one YES. No resting bid can meet that, so this MUST come back killed with nothing "
        "filled. Expected exposure: zero.",
    ):
        return _NEUTRAL

    try:
        data = trader._submit_order_v2(client, body)
    except Exception as exc:
        print(f"{_FAIL}: submission raised: {exc}")
        print(f"Position after the error: {trader._position_count(client, ticker)}")
        return _FAIL
    _emit("RAW RESPONSE (unfillable ask)", data)

    try:
        killed = _filled_count(data) == 0 and _remaining_count(data) == PROBE_COUNT
    except (KeyError, TypeError, InvalidOperation) as exc:
        print(f"{_FAIL}: response carried no readable fill_count/remaining_count ({exc}).")
        return _FAIL

    after = trader._position_count(client, ticker)
    print(f"Position after the unfillable ask: {after}")
    if not killed:
        print(
            f"{_FAIL}: the order did NOT come back killed — fill_count="
            f"{data.get('fill_count')} remaining_count={data.get('remaining_count')} against "
            f"a count of {PROBE_COUNT_STR}. Either fill-or-kill does not behave as assumed "
            "or an ask's limit is not a floor on proceeds. *** CHECK THE ACCOUNT. ***"
        )
        return _FAIL
    if after != 0:
        print(
            f"{_FAIL}: the response reported a kill but the position is {after}. "
            "*** CHECK THE ACCOUNT MANUALLY. ***"
        )
        return _FAIL

    print(
        f"{_PASS}: the fill-or-kill came back with nothing filled and the full "
        f"{PROBE_COUNT_STR} remaining, and the account stayed flat. Kill semantics and the "
        "ask's proceeds-floor direction are confirmed."
    )
    return _PASS


def _step_transfer(client: Any, ticker: str, assume_yes: bool) -> str:
    """
    Round-trip one cent between shard 0 and shard 1 to confirm the transfer path.

    Verifies the two things about trader.ensure_shard_collateral's plumbing that
    only a live call can show: that `amount` really is denominated in centicents
    (trader._cents_to_centicents), and that settlement is asynchronous but lands
    inside config.TRANSFER_SETTLE_TIMEOUT_SECONDS. Uses the real
    trader._execute_transfer and trader._await_transfer_settlement, so a pass
    here is a pass for the code the live path runs.

    Skipped (NEUTRAL, never FAIL) whenever the exchange says the move is not
    available: no per-shard status breakdown at all, no shard 1 advertised, or
    intra_exchange_transfers_active false on either endpoint.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Unused — present only so every step shares one dispatch
            signature.
        assume_yes (bool): Skip the interactive confirmation (--yes).

    Returns:
        str: _PASS when the cent lands on shard 1 and comes back to shard 0;
            _FAIL when a transfer raises or does not settle in time; _NEUTRAL
            when the step is skipped or the operator declines.
    """
    print("\n===== STEP: transfer (inter-shard collateral round trip) =====")

    # Same read main.py does before every run; None means single-shard semantics.
    statuses = scanner.fetch_shard_statuses(client)
    if statuses is None:
        print(
            f"{_NEUTRAL}: no per-shard status breakdown from /exchange/status — this account "
            "or environment is pre-sharding, so there is no second shard to transfer to."
        )
        return _NEUTRAL
    _emit("SHARD STATUSES", statuses)
    if _TRANSFER_DEST_SHARD not in statuses:
        print(
            f"{_NEUTRAL}: the exchange advertises no shard {_TRANSFER_DEST_SHARD} "
            f"(advertised: {sorted(statuses)}) — nothing to transfer to."
        )
        return _NEUTRAL
    # trader's own gate: never move money to or from a shard whose transfers
    # the exchange reports as inactive.
    if not trader._transfers_active(statuses, _TRANSFER_SOURCE_SHARD) or not (
        trader._transfers_active(statuses, _TRANSFER_DEST_SHARD)
    ):
        print(
            f"{_NEUTRAL}: intra-exchange transfers are not active on shard "
            f"{_TRANSFER_SOURCE_SHARD} and/or {_TRANSFER_DEST_SHARD} — skipping."
        )
        return _NEUTRAL

    # The shard-aware balance read the whole run is sized against.
    before = auth.verify_auth(client)
    _emit("BALANCE BY SHARD, CENTS (before)", before)
    if before.get(_TRANSFER_SOURCE_SHARD, 0) < _TRANSFER_PROBE_CENTS:
        print(
            f"{_NEUTRAL}: shard {_TRANSFER_SOURCE_SHARD} holds "
            f"{before.get(_TRANSFER_SOURCE_SHARD, 0)}c — not even one cent to move."
        )
        return _NEUTRAL

    # Mirrors the body trader._execute_transfer builds internally, printed here
    # so the evidence log shows the centicent conversion explicitly.
    preview = {
        "source": "event_contract",
        "destination": "event_contract",
        "amount": trader._cents_to_centicents(_TRANSFER_PROBE_CENTS),
        "source_exchange_shard": _TRANSFER_SOURCE_SHARD,
        "destination_exchange_shard": _TRANSFER_DEST_SHARD,
    }
    _emit("REQUEST BODY (outbound transfer, as trader._execute_transfer builds it)", preview)
    if not _confirm(
        assume_yes,
        f"Move {_TRANSFER_PROBE_CENTS}c ({preview['amount']} centicents) from shard "
        f"{_TRANSFER_SOURCE_SHARD} to shard {_TRANSFER_DEST_SHARD}, then move it back. "
        "Transfers are NOT retried and NOT idempotent — one attempt each.",
    ):
        return _NEUTRAL

    out_id = _transfer_leg(
        client, _TRANSFER_SOURCE_SHARD, _TRANSFER_DEST_SHARD, "outbound"
    )
    if out_id is False:
        return _FAIL
    # Wait for the ASYNCHRONOUS settlement using the real bounded poller: the
    # destination must hold its prior balance plus the cent we sent.
    target_out = {_TRANSFER_DEST_SHARD: before.get(_TRANSFER_DEST_SHARD, 0) + _TRANSFER_PROBE_CENTS}
    mid = trader._await_transfer_settlement(client, target_out)
    _emit("BALANCE BY SHARD, CENTS (after outbound)", mid)
    if mid.get(_TRANSFER_DEST_SHARD, 0) < target_out[_TRANSFER_DEST_SHARD]:
        print(
            f"{_FAIL}: the outbound transfer did not settle within "
            f"{config.TRANSFER_SETTLE_TIMEOUT_SECONDS}s (transfer_id={out_id}). *** MONEY MAY "
            "BE IN FLIGHT — CHECK THE ACCOUNT. *** Not sending it back; re-run once the "
            "balances have settled."
        )
        return _FAIL
    print(f"Outbound leg settled: {_TRANSFER_PROBE_CENTS}c landed on shard {_TRANSFER_DEST_SHARD}.")

    back_id = _transfer_leg(
        client, _TRANSFER_DEST_SHARD, _TRANSFER_SOURCE_SHARD, "return"
    )
    if back_id is False:
        print(
            f"*** {_TRANSFER_PROBE_CENTS}c is stranded on shard {_TRANSFER_DEST_SHARD} — move "
            "it back manually in the Kalshi UI. ***"
        )
        return _FAIL
    target_back = {_TRANSFER_SOURCE_SHARD: before.get(_TRANSFER_SOURCE_SHARD, 0)}
    end = trader._await_transfer_settlement(client, target_back)
    _emit("BALANCE BY SHARD, CENTS (after return)", end)
    if end.get(_TRANSFER_SOURCE_SHARD, 0) < target_back[_TRANSFER_SOURCE_SHARD]:
        print(
            f"{_FAIL}: the return transfer did not settle within "
            f"{config.TRANSFER_SETTLE_TIMEOUT_SECONDS}s (transfer_id={back_id}). *** MONEY MAY "
            "BE IN FLIGHT — CHECK THE ACCOUNT. ***"
        )
        return _FAIL

    print(
        f"{_PASS}: {_TRANSFER_PROBE_CENTS}c round-tripped shard {_TRANSFER_SOURCE_SHARD} -> "
        f"{_TRANSFER_DEST_SHARD} -> {_TRANSFER_SOURCE_SHARD} (transfer_ids: {out_id}, "
        f"{back_id}). The centicent unit and the asynchronous settle timing are confirmed."
    )
    return _PASS


def _transfer_leg(client: Any, source: int, dest: int, label: str):
    """
    Execute one leg of the transfer round trip and report its id.

    Args:
        client (Any): Authenticated prod KalshiClient.
        source (int): exchange_index the funds leave.
        dest (int): exchange_index the funds arrive on.
        label (str): "outbound" or "return", for the printed evidence line.

    Returns:
        str | None | bool: The accepted transfer's id (None if the response
            carried none — the transfer may still be in flight), or the literal
            False when the POST raised. False is distinguishable from a None id
            precisely because None does NOT mean failure here.
    """
    try:
        # The real, single-attempt, never-retried transfer call.
        transfer_id = trader._execute_transfer(client, source, dest, _TRANSFER_PROBE_CENTS)
    except Exception as exc:
        print(
            f"{_FAIL}: the {label} transfer POST raised: {exc}. It is NOT being retried "
            "(the endpoint is not idempotent). *** CHECK THE ACCOUNT — the transfer may "
            "still have landed. ***"
        )
        return False
    print(f"{label.capitalize()} transfer accepted: transfer_id={transfer_id}")
    return transfer_id


# Step name -> implementation. Every step shares the (client, ticker,
# assume_yes) signature so main() can dispatch without special cases.
_STEPS = {
    "no-mapping": _step_no_mapping,
    "unfillable-ask": _step_unfillable_ask,
    "transfer": _step_transfer,
}

# Steps that trade, and therefore require the operator to name a market.
_TICKER_STEPS = frozenset({"no-mapping", "unfillable-ask"})


def main(argv: list | None = None) -> int:
    """
    Parse arguments, run one probe step against the live account, and report.

    Builds a PRODUCTION client unconditionally (the sandbox implements neither
    the V2 order endpoint nor sharding, so a sandbox result would prove
    nothing), verifies credentials with a balance read, then runs the selected
    step.

    Args:
        argv (list | None): Argument vector for testing. None reads sys.argv.

    Returns:
        int: 0 when the step PASSED, 1 when it FAILED, 2 when it was NEUTRAL
            (no verdict reached — no fill, no liquidity, skipped, or aborted at
            the confirmation prompt). Only a 0 from BOTH no-mapping and
            unfillable-ask may be used to justify flipping
            config.V2_ORDERS_ENABLED.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.v2_probe",
        description=(
            "Live verification probe for the V2 order path's NO-leg mapping. SUBMITS REAL "
            "ORDERS against the production account (0.01 contracts, ~1c of exposure). Run "
            "by hand; never wired into the pipeline."
        ),
    )
    parser.add_argument(
        "--ticker",
        help=(
            "Market ticker to probe — pick a LIQUID, CHEAP market you are happy to trade "
            "one cent of. Required for the order steps; there is deliberately no default."
        ),
    )
    parser.add_argument(
        "--step", choices=sorted(_STEPS), default="no-mapping",
        help="Which check to run. Default: no-mapping (the V2 gate itself).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation before each submission.",
    )
    args = parser.parse_args(argv)

    # The scanner/trader/auth calls below log at INFO (order-book key mismatches,
    # rollback criticals, transfer acceptance); without a handler those lines
    # would be dropped, and they belong in the evidence log next to the bodies.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 78)
    print("KALSHI V2 ORDER-PATH LIVE PROBE — REAL MONEY, PRODUCTION ACCOUNT")
    print(f"step={args.step}  ticker={args.ticker}  worst-case exposure per order: "
          f"{PROBE_COUNT_STR} contracts (about one cent)")
    print("Every request body and raw response below is the evidence log for the "
          "config.V2_ORDERS_ENABLED decision — keep it.")
    print("=" * 78)

    if args.step in _TICKER_STEPS and not args.ticker:
        print(f"--ticker is required for --step {args.step}. Refusing to guess a market.")
        return _EXIT_CODES[_NEUTRAL]

    # Prod client, deliberately: the sandbox has neither V2 orders nor shards.
    client = auth.build_client("prod")
    try:
        # Confirms the credentials work AND records the opening balances.
        balances = auth.verify_auth(client)
    except Exception as exc:
        print(f"{_FAIL}: authentication / balance read failed: {exc}")
        return _EXIT_CODES[_FAIL]
    _emit("BALANCE BY SHARD, CENTS (opening)", balances)

    outcome = _STEPS[args.step](client, args.ticker, args.yes)

    print(f"\n================ RESULT: {args.step} -> {outcome} ================")
    if outcome != _PASS:
        print(
            "config.V2_ORDERS_ENABLED must stay False. Both --step no-mapping and --step "
            "unfillable-ask have to PASS before it may be flipped."
        )
    else:
        print(
            "Record this output. config.V2_ORDERS_ENABLED may only be flipped once BOTH "
            "--step no-mapping and --step unfillable-ask have PASSED."
        )
    return _EXIT_CODES[outcome]


if __name__ == "__main__":
    sys.exit(main())
