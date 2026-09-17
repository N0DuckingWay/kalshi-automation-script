"""
File: v2_probe.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    USER-RUN LIVE VERIFICATION CLI — this module is never imported by the
    trading pipeline and must be invoked by a human, deliberately, from a
    terminal:

        python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step <name>] [--dest-shard N] [--yes]

    It exists to verify, for about one cent, the one thing about the V2 order
    path that cannot be proven offline. The V2 order path in trader.py is
    complete and unit-tested, but its NO-leg mapping is an unverified
    HYPOTHESIS: V2 has no side="no"/action="buy" vocabulary, so "buy NO at
    <= p" is mapped onto an `ask` (sell YES) at >= 1 - p, and closing that NO
    position onto a `bid` carrying reduce_only=True (trader._V2_LEG_SIDE). If
    the hypothesis is wrong the bot would take the WRONG SIDE of the market,
    not merely a wrong price. This probe submits the smallest possible real
    orders built by the EXACT trader builders the live path uses
    (trader._build_no_order_v2, trader._build_rollback_order_v2) and reports
    whether the account position moved the way the hypothesis predicts. What
    it is verifying is a SIDE mapping, not a price, so the closing bid is
    priced at the top of the market's own grid and crosses any resting YES ask
    (see PRICE OVERRIDE below); reduce_only is what bounds it.

    STRONGLY RECOMMENDED, NOT A HARD GATE. The V2 path is on by default
    (config.ORDER_API_VERSION = "v2") and does not wait for this probe. What
    backstops it instead is trader._confirm_v2_no_mapping(), which checks the
    very same position MOVEMENT on the first live V2 NO fill of a process — but
    that check pays for its evidence with a REAL trade-sized position, while
    this probe buys the identical evidence for ~1c. If either mapping step
    FAILS, set config.ORDER_API_VERSION = "legacy" to hold the bot on the
    legacy order path.

    *** REAL MONEY. *** Every step here submits live orders (or moves live
    collateral) against the PRODUCTION account. Worst-case exposure is the V2
    minimum fractional count of 0.01 contracts — roughly one cent — but it is
    real, it is not a simulation, and there is no dry-run mode: a probe that
    doesn't submit proves nothing. Every request body and every raw response is
    printed verbatim, and that printed output IS the evidence log behind the
    decision to keep ORDER_API_VERSION = "v2" (or to flip it to "legacy").

Dependencies:
    Imports the auth, config, scanner and trader MODULES (module-style, so the
    probe always calls the same functions the live path calls and tests can
    monkeypatch them at their definition site), plus api_call_with_retry /
    fetch_json_page / signed_request_json from _http.py.

    NOTHING IMPORTS THIS MODULE. It is not referenced by main.py, scheduler.py,
    trader.py, or any other pipeline module, and tests/test_v2_probe.py asserts
    that stays true. Adding an import of v2_probe anywhere in the pipeline would
    put real-money probe orders one code path away from the weekly scheduler.

Notes:
    PROD ONLY. auth.build_client("prod") is hardcoded — the sandbox implements
    neither the V2 order endpoint nor exchange sharding, so a sandbox "pass"
    would prove nothing about the mapping this probe exists to verify.

    COUNT OVERRIDE. trader's builders read the whole-contract count from the
    trader._Leg they are handed (rendered "<n>.00") because the bot sizes in
    whole contracts. The probe wants the V2 minimum of 0.01 contracts, so
    _no_buy_body() / _no_close_body() build the body through the real builders
    on a real _Leg (_probe_leg: side "no", count 1) and then override the
    count — see the loud comment in _no_buy_body. trader.py is deliberately
    NOT modified for the probe's benefit: the thing being verified must stay
    byte-identical to the thing that will run.

    PRICE OVERRIDE. _no_close_body() overrides a second field, the closing
    bid's price, and _step_unfillable_ask overrides the same field on its own
    body. The rollback builder prices a real unwind at a loss floor derived
    from the leg's scanned entry, and the probe's placeholder entry of 0.5
    makes that floor $0.62 on every market and every tick regime — a bid that
    is structurally killed whenever the YES ask sits above 62c, which FAILed
    the mapping gate for a reason unrelated to the mapping and left a real
    position open (DR-04). The probe verifies a SIDE mapping, not a price, so
    the close bids the top of the market's own grid instead
    (trader._v2_top_of_grid_price: 0.99 on a linear-cent grid, 0.999 on
    deci-cent, 0.9999 on a centi-cent edge band), exactly the level the
    unfillable-ask step submits at. reduce_only=True is what bounds that bid:
    it can only close the 0.01 contracts the probe just opened.

    SUBMISSION AND FILL READING. trader._submit_order_v2 classifies fills
    against `int(Decimal(body["count"]))`, which truncates the probe's
    fractional 0.01 to 0 and would misread a kill as a full fill. The probe
    therefore submits through the SAME transport the trader uses
    (_http.signed_request_json — single-shot, retry-free, non-2xx raises) and
    reads fill_count/remaining_count itself via trader._parse_fixed_point,
    keeping the same Decimal comparison and the same "unparseable means
    ambiguous, never a non-fill" semantics. Neither order step collapses a
    non-conforming response into a clean kill, and both apply the SAME test for
    what a true kill is (nothing filled, the full count still remaining):
    _step_no_mapping classifies the counts into THREE outcomes rather than
    two — a complete fill, a true kill, and a fill-or-kill invariant violation
    (partial, over-fill, stale remainder) — where it used to call a partial
    fill a kill and report the account as "still flat" (DR-20);
    _step_unfillable_ask tests for the same true kill and FAILs anything else,
    naming the counts, as it always has. Their REMEDIES still differ, and only
    the kill test is shared: that step's ask is meant to be unfillable, so it
    folds a complete fill into the same FAIL, and it reports the account from a
    single un-refreshed read (a residual, below). Both are stricter than
    trader._v2_fill_status, which reads only fill_count and so cannot see a
    surprise confined to remaining_count. A 2xx body that is not a JSON
    OBJECT at all (`"accepted"`, `[]`, `123`, `true`, `null`) is handled one
    step earlier, by _non_object_body_fail: the fill readers would raise
    AttributeError out of a real, possibly-filled submission, killing the probe
    before it could read the position or warn about it (DR-58). That guard
    restores the position read and the warning, NOT the reduce-only close —
    an unreadable body proves nothing about the mapping that close rests on,
    so the step ends at FAIL with the position left for a human, exactly as a
    disproven mapping does. A body that IS an object but whose fill counts are
    absent or unparseable leaves the probe in that same state one step later;
    in the NO-buy step (_step_no_mapping) that branch now reports through the
    same helper _non_object_body_fail uses, _recheck_and_report_position: the
    ledger is re-read once when the first read is flat or unreadable, and
    lookup-failed, position-open and genuinely-flat are three distinct printed
    outcomes (DR-60). That is not every branch in this state:
    _step_unfillable_ask's own unreadable-fill-counts branch returns FAIL above
    its only post-submission position read, its `not killed` branch names the
    counts but reports the account from one un-refreshed read and never says
    FLATTEN, and both steps' submission-EXCEPTION handlers decide from one
    un-refreshed read too. Those four are recorded residuals — see CLAUDE.md's
    DR-60 bullet — so nothing here should be read as a module-wide guarantee.

    CONFIRMATION. Nothing is submitted until the request body has been printed
    and the operator has typed "yes" — unless --yes was passed, which is for a
    second or third run once the operator has already seen the bodies.

    EXIT CODES: 0 = the executed step PASSED, 1 = something FAILED, 2 =
    NEUTRAL (the step could not be run to a conclusion — a TRUE kill with
    nothing filled and the account flat, no liquidity, the OPENING confirmation
    was declined, or a skipped transfer step, including a --dest-shard that IS
    the source shard, refused before any POST — DR-22). A PARTIAL or
    otherwise non-conforming fill-or-kill response is NOT neutral: it is the
    protocol violation the probe exists to catch, so it is a FAIL that names
    the counts and re-reads the account (DR-20). A
    prompt declined once a position is already open (e.g. the unfillable-ask
    step's closing confirmation) is scored FAIL, not NEUTRAL — declining to
    close a real open position is not a neutral outcome, and the printed
    message tells the operator to flatten it manually. Only a 0 from BOTH
    mapping steps (--step no-mapping and --step unfillable-ask) is evidence
    that the V2 order path may be trusted to run unsupervised.
"""
import argparse
import json
import logging
import sys
import time
from decimal import Decimal
from typing import Any

from . import auth, config, scanner, trader
from ._http import api_call_with_retry, fetch_json_page, signed_request_json

# The V2 minimum fractional contract count. This lives here rather than in
# config.py on purpose: it is a property of THIS diagnostic (make the exposure
# as close to zero as the exchange allows), not a tunable of the trading
# strategy, and nothing in the pipeline may ever size an order from it.
PROBE_COUNT_STR = "0.01"
PROBE_COUNT = Decimal(PROBE_COUNT_STR)

# Default shards for the transfer step. Source is the shard everything
# historically lived on; the default destination is the first shard Kalshi
# migrated markets to (combos, 2026-08-17) and is overridable via --dest-shard
# now that shards 2 and 3 exist. One cent is the smallest amount that is still
# a non-zero movement in the endpoint's centicent unit after conversion.
_TRANSFER_SOURCE_SHARD = config.DEFAULT_EXCHANGE_INDEX
_TRANSFER_DEST_SHARD_DEFAULT = 1
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

    Every request body and every response goes through here, pretty-printed
    with indent=2, because this output is the artifact a human reads when
    deciding whether to trust the V2 order path.

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
    runs before every step that submits, after that step's exact request body
    has been printed.

    ONE gate per STEP, not per request. The transfer step confirms the whole
    round trip up front and its RETURN leg has no gate of its own — by the time
    that leg runs the cent is already on the far shard and leaving it there is
    the worse outcome, so the step is deliberately committed once the operator
    approves it. Read "before every submission" as "before every step"; this
    docstring used to say the former and mean the latter (TS-30).

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
    # trader builders is constructed identically to a real trading candidate
    # (notably exchange_index and the price_ranges tick grid).
    return raw, scanner._market_from_dict(raw, "")


def _probe_leg(market: Any, no_price: float) -> trader._Leg:
    """
    Build the trader._Leg the probe hands to trader's V2 order builders.

    The builders take a _Leg rather than a TradeSpec, so the probe constructs
    the real thing rather than a stand-in: `market` (ticker, tick grid,
    exchange shard — exactly what _build_no_order_v2 /
    _build_rollback_order_v2 read), side "no" (the probe verifies the NO-leg
    mapping and nothing else), `price_dollars` = the scanned NO price, `count`
    1 (a whole-contract placeholder, immediately overridden — see
    _no_buy_body) and a fixed "NO on v2-probe" label.

    Args:
        market (Any): The scanner.ApiMarket the probe trades.
        no_price (float): Scanned NO price in dollars, from the order book.

    Returns:
        trader._Leg: The NO leg for the probe's order.
    """
    # Cross-module: the same leg type the live path builds in _ordered_legs,
    # so the builders below cannot tell the probe from a real trade
    return trader._Leg(
        market=market, side="no", price_dollars=no_price, count=1, label="NO on v2-probe",
    )


def _no_buy_body(market: Any, no_price: float) -> dict:
    """
    Build the NO-buy request body: the real builder, with a 0.01 count.

    Built through trader._build_no_order_v2 so every other field (side from
    _V2_LEG_SIDE, the tick-aware limit price, time_in_force, client_order_id,
    exchange_index, reduce_only, post_only) is byte-identical to what the live
    path would send.

    Args:
        market (Any): The scanner.ApiMarket to trade.
        no_price (float): Scanned NO price in dollars.

    Returns:
        dict: The request body, identical to the live path's except for count.
    """
    # Cross-module: the real NO-leg builder, fed a real _Leg (see _probe_leg)
    body = trader._build_no_order_v2(_probe_leg(market, no_price))
    # !!! DELIBERATE OVERRIDE — DO NOT "FIX" THIS BY CHANGING trader.py !!!
    # The builders read a whole-contract count from the _Leg and render
    # "<n>.00" because the bot sizes in whole contracts. The probe wants the V2
    # fractional minimum of 0.01 contracts (~one cent of exposure), which that
    # contract cannot express. Overriding the one field here keeps trader.py —
    # the code being verified — completely untouched by the verification.
    body["count"] = PROBE_COUNT_STR
    return body


def _no_close_body(market: Any) -> dict:
    """
    Build the reduce-only closing bid: the real rollback builder, 0.01 count,
    priced at the top of the market's own grid.

    Everything that makes this body a CLOSE comes from
    trader._build_rollback_order_v2 — side "bid" (trader._V2_LEG_SIDE's
    "close_no"), reduce_only=True, time_in_force "fill_or_kill", the ticker and
    the market's own exchange shard — so the probe verifies the real unwind
    body rather than a reimplementation of it. TWO fields are then overridden:
    the count (the V2 fractional minimum, see _no_buy_body) and the price. See
    the module header's PRICE OVERRIDE note for why the builder's price cannot
    be used here.

    Args:
        market (Any): The scanner.ApiMarket whose NO position is being closed.

    Returns:
        dict: The request body — trader._build_rollback_order_v2's output with
            the count and the price overridden.
    """
    body = trader._build_rollback_order_v2(_probe_leg(market, 0.5))
    # The real unwind builder fixes the SHAPE of the close (reduce-only YES
    # bid, this market's shard; count overridden below). Its PRICE is a loss
    # floor derived from the 0.5 placeholder entry — $0.62 on every market —
    # and a bid at $0.62 is structurally killed whenever the YES ask is above
    # it, which made the mapping gate FAIL for a reason unrelated to the
    # mapping and left a real position open (DR-04). The probe wants a close
    # that fills regardless of price at 0.01 contracts, so it bids the top of
    # the market's own grid — the same LEVEL _step_unfillable_ask submits its
    # deliberately unfillable ASK at, on the opposite side. reduce_only bounds
    # the exposure to the held 0.01 contracts.
    body["price"] = trader._format_price(trader._v2_top_of_grid_price(market))
    body["count"] = PROBE_COUNT_STR
    return body


def _submit_probe_order(client: Any, body: dict) -> Any:
    """
    Submit one probe order and return the parsed response body.

    Uses _http.signed_request_json against config.V2_ORDER_PATH — the SAME
    transport, path, and single-shot/no-retry contract trader._submit_order_v2
    uses. The probe does not go through _submit_order_v2 itself because that
    function classifies the fill against int(Decimal(count)), which truncates
    the fractional probe count to 0 and would misread a kill as a full fill;
    the probe reads fill_count/remaining_count itself (see _fill_counts).

    Args:
        client (Any): Authenticated prod KalshiClient.
        body (dict): Request body from _no_buy_body/_no_close_body.

    Returns:
        Any: The parsed 2xx response body. Normally a dict (possibly wrapped
            under "order"), but DELIBERATELY not narrowed to one:
            _http.signed_request_json is annotated `-> Any` and does not narrow
            either, so a 2xx body of `"accepted"`, `[]`, `123`, `true` or
            `null` reaches the caller as a str/list/int/bool/None. Every caller
            must therefore check `isinstance(data, dict)` before reading fill
            fields off it — see _non_object_body_fail (DR-58).

    Raises:
        ApiException: On a non-2xx status — never retried.
    """
    logging.info(
        "Submitting V2 probe order: ticker=%s side=%s price=%s count=%s client_order_id=%s",
        body["ticker"], body["side"], body["price"], body["count"], body["client_order_id"],
    )
    return signed_request_json(client, "POST", config.V2_ORDER_PATH, body=body)


def _fill_counts(data: dict) -> tuple:
    """
    Read a V2 response's fill_count and remaining_count as Decimals.

    Mirrors the live path's parsing (trader._parse_fixed_point: prefer the _fp
    string variant, Decimal never float, tolerate the "order"-wrapped shape).
    Unparseable or absent counts return None — the caller must treat that as
    "we cannot tell what happened", never as a non-fill, exactly as
    trader._v2_fill_status raises rather than guesses on the live path.

    Args:
        data (dict): Parsed V2 order response body.

    Returns:
        tuple: (fill_count, remaining_count) as Decimal | None each.
    """
    inner = data.get("order")
    order = inner if isinstance(inner, dict) else data
    return (
        trader._parse_fixed_point(order, "fill_count"),
        trader._parse_fixed_point(order, "remaining_count"),
    )


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
        data (dict): Parsed V2 order response body; average_fee_paid is
            typically absent when nothing filled, but this function reads no
            fill counts and so cannot assert that — see the printed wording
            below (DR-20).
        price_str (str): The limit price the order was submitted at, used as
            the model's price input.

    Returns:
        None
    """
    inner = data.get("order")
    order = inner if isinstance(inner, dict) else data
    charged = order.get("average_fee_paid")
    if charged is None:
        # Deliberately does NOT claim "nothing filled": this function never
        # reads fill_count/remaining_count, and on a PARTIAL fill that claim
        # was a second, independent false assertion of an empty fill printed
        # right above the verdict (DR-20).
        print(
            "Fee check: response carried no average_fee_paid, so there is no charged fee "
            "to compare (the raw response body and the verdict below, not this line, say "
            "what filled)."
        )
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


def _recheck_and_report_position(
    client: Any, ticker: str, observed: float | None
) -> None:
    """
    Re-read a flat-or-unreadable position once, then name which of three states
    the account is in.

    THE SINGLE DEFINITION of the re-read-and-report tail, so that the callers
    that do run it cannot drift apart. It has exactly THREE call sites:
    `_non_object_body_fail` (a 2xx body that is not a JSON object at all —
    DR-58, itself reached from BOTH submission-response readers),
    `_step_no_mapping`'s unreadable-fill-counts branch (a genuine JSON object
    whose `fill_count`/`remaining_count` cannot be read — DR-60), and
    `_step_no_mapping`'s fill-or-kill invariant-violation branch (counts that
    ARE readable but describe a partial fill, an over-fill or a stale
    remainder — DR-20). The first two used to disagree: the first re-read and
    printed three distinct outcomes, while the second decided from a SINGLE
    un-refreshed read and printed nothing at all unless that read was truthy —
    so a lagging ledger and a FAILED lookup both came out as silence while a
    real 0.01 position was open on the production account.

    FOUR OTHER BRANCHES sit in the same state and deliberately do NOT call
    this — recorded as residuals, not as coverage: `_step_unfillable_ask`'s own
    unreadable-fill-counts branch, which returns _FAIL above its only
    post-submission position read (that step was explicitly out of DR-60's
    scope; its ask is designed to be unfillable, but its own DR-58 comment
    already argues an unreadable 2xx body is not proof it was killed);
    `_step_unfillable_ask`'s `not killed` branch, the direct sibling of DR-20's
    new one, which names the counts and FAILs but decides its account report
    from a single un-refreshed read and never says FLATTEN; and both steps'
    post-submission EXCEPTION handlers, which likewise decide from a single
    un-refreshed read. Widening to them is a separate change, and the word
    "SINGLE DEFINITION" above is about this tail's ONE implementation, never a
    claim that every such branch runs it.

    The re-read fires on `None` (the lookup itself failed) as well as on an
    exact `0`, which EXTENDS DR-21's read-after-write-lag reasoning rather than
    restating it: DR-21's own re-read in `_step_no_mapping` fires only on an
    exactly-zero read because a non-zero one is already the evidence that step
    wants. Here neither `None` nor `0` is evidence of anything, and collapsing
    `None` into "flat" is precisely the silent branch this exists to prevent.
    A first read that already moved is evidence and is NOT re-polled.

    The retried `trader._position_count` is used deliberately (never the
    single-shot `_position_count_once`): the probe is a human-supervised tool
    with 0.01 contracts at stake and no unhedged-window latency budget, so a
    transient 429 must not read as "state unknown". This is the same choice
    DR-21's re-read makes, and the same one that must never be copied into
    `trader._confirm_v2_no_mapping`.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): The market the order was submitted against.
        observed (float | None): The position the caller already read AND PRINTED —
            `float` (possibly 0) or `None` when that lookup failed. The caller
            prints it because only the caller knows what to call the read;
            this function prints only the re-read and the verdict.

    Returns:
        None: Everything it has to say it prints. The caller owns the verdict,
            which is _FAIL at both of today's call sites.
    """
    if observed is None or observed == 0:
        # A ledger that reads flat (or fails) straight after a submission is
        # usually read-after-write lag, not proof of a kill — re-read ONCE
        # before concluding anything, exactly as DR-21 established.
        time.sleep(trader._V2_MAPPING_RECHECK_DELAY_SECONDS)
        # Cross-module: the account's ledger is the only remaining evidence.
        observed = trader._position_count(client, ticker)
        print(f"Position after re-read: {observed}")
    if observed is None:
        print(
            f"*** Could not read the position on {ticker}. CHECK IT MANUALLY and "
            "FLATTEN ANYTHING YOU FIND. ***"
        )
    elif observed:
        print(
            f"*** A {observed} position is OPEN on {ticker}. FLATTEN IT MANUALLY in "
            "the Kalshi UI. ***"
        )
    else:
        print(f"The account reads flat on {ticker} — nothing to flatten.")


def _non_object_body_fail(client: Any, ticker: str, data: Any, label: str) -> str:
    """
    Report a submitted order whose 2xx body is not a JSON object, and check the
    account.

    This guards the probe's two submission-response READERS, which is not the
    same as its three SUBMISSIONS: `_step_no_mapping`'s NO buy and
    `_step_unfillable_ask`'s ask both read fill fields off the body, while
    `_step_no_mapping`'s reduce-only NO close only hands its body to `_emit`
    and takes its verdict from `trader._position_count`, so nothing there can
    raise on a non-object body.

    `_http.signed_request_json` is annotated `-> Any` and does not narrow a 2xx
    body to a dict, so `"accepted"`, `[]`, `123`, `true` or a literal `null` all
    reach the caller as a `str`/`list`/`int`/`bool`/`None`. Every consumer below
    the submission calls `data.get(...)` — `_fill_counts` at both guarded sites
    and `_report_fee` in `_step_no_mapping` — so such a body used to raise an
    uncaught `AttributeError` IMMEDIATELY AFTER A REAL ORDER HAD BEEN
    SUBMITTED: the probe died on a traceback with no position read, no
    flatten-it-manually warning (and, in `_step_no_mapping`, no reduce-only
    close), and a real position was left open with nothing to tell the operator
    it existed (DR-58). The close is NOT restored here — it rests on the very
    mapping this unreadable body proves nothing about, so the position is left
    for a human, as a disproven mapping already leaves it. A 2xx proves the
    order reached the exchange, so it MAY have filled
    — the same reading `trader._execute_transfer` takes of a non-object 2xx
    transfer body (DR-05), and the opposite of the ORDER-submission readers in
    `trader.py`, which must stay loud because raising there routes
    `_execute_one` into its position-DELTA resolution path. No such caller
    exists above the probe, so raising here resolves nothing.

    The position is read here and handed to `_recheck_and_report_position`,
    which re-reads ONCE after `trader._V2_MAPPING_RECHECK_DELAY_SECONDS` when
    that first read is flat or unreadable and then prints all three outcomes
    distinctly — an unreadable position, an open position, and a genuinely flat
    account. That helper is shared with `_step_no_mapping`'s
    unreadable-fill-counts branch, which is the same epistemic state one step
    later (DR-60); see its docstring for why the re-read fires on `None` too.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): The market the order was submitted against.
        data (Any): The non-dict 2xx body, for its type name.
        label (str): Which submission this was, e.g. "NO buy", for the message.

    Returns:
        str: Always _FAIL — the body cannot be read, so nothing is proven and a
            position may be open.
    """
    print(
        f"{_FAIL}: the 2xx {label} response body was {type(data).__name__}, not a JSON "
        "object, so its fill fields cannot be read — the order MAY have filled."
    )
    # Cross-module: the account's ledger is the only remaining evidence about
    # what this order did.
    stranded = trader._position_count(client, ticker)
    print(f"Position after the non-object {label} response: {stranded}")
    # Shared with _step_no_mapping's unreadable-fill-counts branch: the one
    # definition of "re-read a flat-or-unreadable ledger once, then say which
    # of the three states the account is in" (DR-60).
    _recheck_and_report_position(client, ticker, stranded)
    return _FAIL


def _step_no_mapping(client: Any, ticker: str, assume_yes: bool, dest_shard: int) -> str:
    """
    THE CORE GATE: verify that an `ask` opens a NO position and reduce_only
    closes it.

    Sequence, all of it against the live account:
      1. Fetch the market and its order book; derive the NO ask price from the
         resting YES bids (buying NO at p == selling YES at 1 - p).
      2. Confirm the account is FLAT on this ticker — the probe's whole verdict
         is "which way did the position move", which is meaningless otherwise.
      3. Submit the NO-buy body built by trader._build_no_order_v2 (side "ask"
         per _V2_LEG_SIDE). A 2xx body that is not a JSON object is a FAIL that
         still checks the account (_non_object_body_fail), never a traceback
         out of the fill readers (DR-58). A body that IS an object but whose
         fill_count/remaining_count cannot be read is the same epistemic state
         one step later — a real order, a 2xx, and no idea what it did — and
         takes the same remedy through the shared
         _recheck_and_report_position: the ledger is refreshed once and the
         account is reported as unreadable, open or genuinely flat, where this
         branch used to decide from one un-refreshed read and print nothing at
         all for the first two (DR-60). The verdict stays FAIL.
      3a. Classify the readable counts into THREE outcomes, not two: a
         complete fill, a true kill (nothing filled, the full count still
         remaining), and anything else — a partial fill, an over-fill or a
         stale remainder. That third class is a fill-or-kill invariant
         violation, so it FAILs naming the actual counts and reports the
         account through the same _recheck_and_report_position tail. It used
         to collapse into "not filled" and, with a lagging ledger, print
         "killed unfilled and the account is still flat" and exit NEUTRAL
         while a real fraction of a contract was open (DR-20). Judging on
         BOTH counts is deliberately stricter than trader._v2_fill_status,
         which reads fill_count alone — see the comment at the test.
      4. Re-read the position. PASS half one iff it went NEGATIVE, which is
         Kalshi's unified-ledger convention for a NO position. A position of
         exactly 0 after a reported full fill is read ONCE more, after
         trader._V2_MAPPING_RECHECK_DELAY_SECONDS, before any of the sign
         branches run: an unmoved ledger straight after a fill is usually
         read-after-write lag, the same assumption
         trader._confirm_v2_no_mapping makes (DR-21). A POSITIVE position
         means the hypothesis is WRONG — the ask opened YES exposure — and the
         closing order is then NOT submitted, because its own mapping
         (reduce_only bid nets a NO position to flat) rests on the same
         disproven assumption and would add to the wrong exposure instead.
      5. Submit the closing body from trader._build_rollback_order_v2 (side
         "bid", reduce_only=True), priced at the top of this market's own grid
         rather than at the builder's loss floor, so the close crosses
         whatever is resting on the book (see _no_close_body). PASS half two
         iff the position returns to 0. This is the step's SECOND submission
         — the probe's second order-submission site of three — and it needs
         no _non_object_body_fail guard: its 2xx body is only
         handed to _emit (typed Any — it just pretty-prints), and the verdict
         comes from trader._position_count, so there is no data.get(...) on
         it to raise. Add the guard here if a future edit starts reading fill
         fields off close_data (DR-58).

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Liquid, cheap market chosen by the operator.
        assume_yes (bool): Skip the interactive confirmations (--yes).
        dest_shard (int): Unused — shared step signature.

    Returns:
        str: _PASS only when the position went negative AND came back to zero;
            _FAIL on any contrary evidence, an error, a 2xx response body that
            cannot be read (a non-object body, or an object whose fill counts
            are missing), a readable response that is neither a complete fill
            nor a true kill (a fill-or-kill invariant violation — DR-20), or a
            position left open — this includes declining the SECOND (closing)
            confirmation, since a real position is open by then and declining
            to close it is not a neutral outcome; _NEUTRAL only when the step
            never reached a verdict with no position at risk: no liquidity, a
            TRUE kill (nothing filled, the full count remaining) against a
            genuinely flat account, or declining the FIRST (opening)
            confirmation, before any order was submitted.
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

    # THE HYPOTHESIS UNDER TEST — the same builder the live path would call.
    body = _no_buy_body(market, no_price)
    _emit("REQUEST BODY (NO buy — hypothesis: an ask opens a NO position)", body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill {body['side'].upper()} (sell YES) on {ticker} for "
        f"{PROBE_COUNT_STR} contracts at a limit of {body['price']} on shard "
        f"{body['exchange_index']}. Under the hypothesis this BUYS {PROBE_COUNT_STR} NO "
        f"at about {no_price:.4f} — worst case about one cent of exposure. The top "
        f"resting YES bid is {1.0 - no_price:.4f}, so this limit should cross and fill.",
    ):
        return _NEUTRAL

    try:
        # Same transport/path/no-retry contract as the live path — one attempt.
        data = _submit_probe_order(client, body)
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
    if not isinstance(data, dict):
        # A 2xx body that is not an object cannot be read by _fill_counts /
        # _report_fee below, and letting their .get() raise here would skip the
        # position read AND the reduce-only close, stranding a real position
        # with no operator warning (DR-58).
        return _non_object_body_fail(client, ticker, data, "NO buy")

    fill, remaining = _fill_counts(data)
    # TWO orthogonal questions, never one boolean: `filled` answers "was this a
    # complete fill" and `conforming` answers "is this a shape fill-or-kill is
    # allowed to produce at all". Together they name the three readable
    # outcomes the branches below dispatch on — complete fill, true kill,
    # invariant violation — plus the unreadable one (`filled = None`).
    # Collapsing everything non-full into `filled = False` reported a PARTIAL
    # fill (0.005 filled, 0.005 remaining) as a clean kill, skipped the DR-21
    # re-read — which is gated on `filled` — and printed "killed unfilled and
    # the account is still flat" while a real 0.005 position was open on the
    # production account (DR-20). The sibling step _step_unfillable_ask already
    # classified this way. `conforming` is left True in the unreadable case and
    # is only safe to read because the `filled is None` branch below returns
    # BEFORE it is tested — keep that order, or an unreadable body falls
    # through to the "still flat" NEUTRAL and reintroduces DR-60.
    filled: bool | None
    conforming = True
    if fill is None or remaining is None:
        # Same reading the live path forces: unparseable fill fields mean the
        # outcome is UNKNOWN, not that nothing filled.
        print(f"{_FAIL}: response carried no readable fill_count/remaining_count.")
        filled = None
    else:
        filled = remaining == 0 and fill == PROBE_COUNT
        # A fill-or-kill either fills completely or comes back with the full
        # count still remaining. Anything else is an invariant violation and is
        # exactly what this probe exists to catch before real size flows.
        # ATTRIBUTION, precisely, because the FAIL below quotes it: trader's
        # _v2_fill_status reads ONLY fill_count, so it raises — rather than
        # guess a status — on a PARTIAL or an OVER-fill, where fill_count is
        # neither the requested count nor zero. It never reads remaining_count
        # at all, so a STALE REMAINDER (fill_count == count with a non-zero
        # remaining_count) books as "executed" there, and a kill whose
        # remaining_count is not the full count books as "canceled". This test
        # reads BOTH counts, so the probe deliberately holds a STRICTER
        # contract than the live reader on exactly those two shapes: an
        # unexpected remaining_count is itself a finding on the one gate that
        # decides whether the bot stays on V2, and the sibling
        # _step_unfillable_ask has tested both counts since it was written, so
        # the two steps agree about what a kill is.
        conforming = filled or (fill == 0 and remaining == PROBE_COUNT)

    after = trader._position_count(client, ticker)
    print(f"Position after the NO buy: {after}")
    if filled and after == 0:
        # Read-after-write lag is the usual cause, exactly as
        # trader._confirm_v2_no_mapping assumes: re-read ONCE, here, before
        # any of the sign branches below, so a re-read that comes back
        # positive (mapping disproven) or None (lookup failed) still takes
        # its own branch instead of falling through (DR-21). This is
        # deliberately NOT _recheck_and_report_position: that helper reports a
        # terminal FAIL, while this re-read feeds the sign branches below, and
        # its gate is narrower (`filled` truthy AND exactly zero, never None —
        # a failed lookup here has its own `after is None` branch). Collapsing
        # the two would silently change the `filled is True` path.
        time.sleep(trader._V2_MAPPING_RECHECK_DELAY_SECONDS)
        after = trader._position_count(client, ticker)
        print(f"Position after re-read: {after}")
    _report_fee(data, body["price"])

    if filled is None:
        # The order went out, the exchange returned 2xx, and the body cannot
        # say what happened — the same state _non_object_body_fail handles one
        # step earlier, so it takes the same remedy (DR-60). This branch used
        # to decide from the single un-refreshed read above (the DR-21 re-read
        # is gated on `filled`, which is falsy here) and to print nothing for
        # either a lagging ledger or a FAILED lookup, while a real 0.01
        # position could be open.
        _recheck_and_report_position(client, ticker, after)
        return _FAIL
    if not conforming:
        # A real order, a 2xx, and a response the fill-or-kill contract says
        # cannot happen: the counts are readable but describe a partial fill,
        # an over-fill or a stale remainder. Never NEUTRAL and never a claim
        # that the account is flat — some of the order may have landed, so the
        # ledger is re-read and reported through the same shared tail the
        # unreadable-body branches use (DR-20).
        print(
            f"{_FAIL}: the fill-or-kill neither filled completely nor came back killed — "
            f"fill_count={fill} remaining_count={remaining} against a count of "
            f"{PROBE_COUNT_STR}. A fill-or-kill may only do one of those two things. The "
            "live path raises rather than guess a status whenever fill_count is neither "
            "the requested count nor zero (trader._v2_fill_status), but it reads no "
            "remaining_count at all — so a surprise confined to that field would pass it "
            "as a clean fill or a clean kill. This probe checks both counts deliberately. "
            "*** CHECK THE ACCOUNT. *** Set config.ORDER_API_VERSION = \"legacy\" to hold "
            "the bot on the legacy order path."
        )
        _recheck_and_report_position(client, ticker, after)
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
            "trader._V2_LEG_SIDE takes the wrong side of the market. The reduce_only "
            "unwind is NOT being submitted — it rests on the same disproven mapping and "
            "would add to this exposure. *** FLATTEN THE POSITION MANUALLY. *** "
            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy path.'
        )
        return _FAIL
    if after == 0:
        print(
            f"{_FAIL}: the response reported a complete fill but the position on {ticker} is "
            "still 0 after a re-read. Neither signal is trustworthy — check the "
            "account manually and FLATTEN ANY POSITION YOU FIND."
        )
        return _FAIL

    print(
        f"Half one CONFIRMED: the ask filled and opened a NEGATIVE position ({after}), "
        "which is Kalshi's convention for a NO position."
    )

    # Half two: does a reduce_only bid net that NO position back to flat?
    close_body = _no_close_body(market)
    _emit("REQUEST BODY (NO close — hypothesis: reduce_only bid nets to flat)", close_body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill {close_body['side'].upper()} on {ticker} for "
        f"{PROBE_COUNT_STR} contracts at a limit of {close_body['price']} with "
        f"reduce_only=true, to close the {after} position just opened.",
    ):
        print(
            f"{_FAIL}: declined at the prompt while a {after} position is OPEN on {ticker}. "
            "*** FLATTEN IT MANUALLY. ***"
        )
        return _FAIL

    try:
        close_data = _submit_probe_order(client, close_body)
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
            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy path.'
        )
        return _FAIL

    print(
        f"{_PASS}: an ask opened a NO position (negative) and a reduce_only bid returned "
        "the account to flat. BOTH halves of the trader._V2_LEG_SIDE mapping are "
        "confirmed against the live API."
    )
    return _PASS


def _step_unfillable_ask(client: Any, ticker: str, assume_yes: bool, dest_shard: int) -> str:
    """
    Verify fill-or-kill kill semantics (and the ask's price-cap direction).

    Submits an ask at the highest tradeable level of the market's own grid
    (trader._v2_top_of_grid_price — one tick below 1). An ask's limit is the
    MINIMUM proceeds accepted per contract, so demanding almost the full dollar
    to sell one YES is above any realistic resting bid and the order cannot
    fill. Two things are proven at once: that a FoK that cannot fill comes back
    killed with the full count remaining (rather than resting, or partially
    filling), and that the ask's limit really is a floor on proceeds — the
    direction trader._v2_limit_price's ceiling quantization assumes.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Market to submit against.
        assume_yes (bool): Skip the interactive confirmation (--yes).
        dest_shard (int): Unused — shared step signature.

    Returns:
        str: _PASS when the order came back with zero filled, the full count
            remaining, and the account still flat; _FAIL if anything filled,
            the response/position disagrees, the 2xx body is not a JSON object
            (unreadable is not proof of a kill — DR-58), or the market could
            not be read at all (no market means no order was even attempted);
            _NEUTRAL
            only if the operator declined the interactive confirmation.
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

    # Top of the market's own grid, via the same helper the live rollback uses:
    # the highest tradeable level (e.g. "0.9900" on a whole-cent market).
    price_str = trader._format_price(trader._v2_top_of_grid_price(market))

    # "Unfillable" is only true while no resting YES bid meets the limit. On a
    # near-settled market the top bid can sit AT the top of the grid — the ask
    # would then genuinely fill (a real position) and the step would misreport
    # broken FoK semantics. Refuse to run rather than manufacture a false FAIL.
    book = scanner._fetch_orderbook(client, ticker)
    if book is None:
        print(f"{_NEUTRAL}: order book unavailable for {ticker} — cannot prove the "
              "ask would be unfillable.")
        return _NEUTRAL
    yes_bids = [Decimal(str(level[0])) for level in (book.get("yes") or [])]
    if yes_bids and max(yes_bids) >= Decimal(price_str):
        print(
            f"{_NEUTRAL}: the top resting YES bid ({max(yes_bids)}) meets the "
            f"top-of-grid limit {price_str} — the ask would actually FILL here. "
            "Pick a market trading well below the top of its grid and re-run."
        )
        return _NEUTRAL

    body = _no_buy_body(market, 0.5)
    # Price override (module header, PRICE OVERRIDE): the deliberately
    # unfillable price replaces the builder's capped one — everything else in
    # the body stays the builder's.
    body["price"] = price_str
    _emit("REQUEST BODY (deliberately unfillable ask)", body)
    if not _confirm(
        assume_yes,
        f"Submit a fill-or-kill ASK on {ticker} for {PROBE_COUNT_STR} contracts at a limit "
        f"of {price_str} — the top of the grid, i.e. demanding almost the full dollar to "
        "sell one YES. No resting bid can meet that, so this MUST come back killed with "
        "nothing filled. Expected exposure: zero.",
    ):
        return _NEUTRAL

    try:
        data = _submit_probe_order(client, body)
    except Exception as exc:
        print(f"{_FAIL}: submission raised: {exc}")
        print(f"Position after the error: {trader._position_count(client, ticker)}")
        return _FAIL
    _emit("RAW RESPONSE (unfillable ask)", data)
    if not isinstance(data, dict):
        # The ask was MEANT to be unfillable, but an unreadable 2xx body is not
        # proof that it was killed — a 2xx proves only that the order reached
        # the exchange. Check the account instead of raising out of
        # _fill_counts' .get() (DR-58).
        return _non_object_body_fail(client, ticker, data, "unfillable ask")

    fill, remaining = _fill_counts(data)
    if fill is None or remaining is None:
        print(f"{_FAIL}: response carried no readable fill_count/remaining_count.")
        return _FAIL
    killed = fill == 0 and remaining == PROBE_COUNT

    after = trader._position_count(client, ticker)
    print(f"Position after the unfillable ask: {after}")
    if not killed:
        print(
            f"{_FAIL}: the order did NOT come back killed — fill_count={fill} "
            f"remaining_count={remaining} against a count of {PROBE_COUNT_STR}. Either "
            "fill-or-kill does not behave as assumed or an ask's limit is not a floor on "
            "proceeds. *** CHECK THE ACCOUNT. ***"
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


def _transfer_leg(client: Any, source: int, dest: int, label: str) -> str | None | bool:
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


def _step_transfer(client: Any, ticker: str, assume_yes: bool, dest_shard: int) -> str:
    """
    Round-trip one cent between shard 0 and another shard to confirm transfers.

    Verifies the two things about trader.ensure_shard_collateral's plumbing that
    only a live call can show: that `amount` really is denominated in centicents
    (trader._cents_to_centicents), and that settlement is asynchronous but lands
    inside config.TRANSFER_SETTLE_TIMEOUT_SECONDS. Uses the real
    trader._execute_transfer and trader._await_transfer_settlement, so a pass
    here is a pass for the code the live path runs.

    Skipped (NEUTRAL, never FAIL) whenever the move is not available: the
    destination IS the source shard (DR-22 — a self-transfer is refused before
    any POST), no per-shard status breakdown at all, no destination shard
    advertised, or intra_exchange_transfers_active false on either endpoint.

    Args:
        client (Any): Authenticated prod KalshiClient.
        ticker (str): Unused — present only so every step shares one dispatch
            signature.
        assume_yes (bool): Skip the interactive confirmation (--yes).
        dest_shard (int): Destination shard for the round trip (--dest-shard,
            default 1). Must differ from _TRANSFER_SOURCE_SHARD; equal is a
            NEUTRAL refusal, and a shard the exchange does not advertise (any
            negative value, or an unknown index) is refused by the advertised-
            shard guard below.

    Returns:
        str: _PASS when the cent lands on the destination shard and comes back;
            _FAIL when a transfer raises or does not settle in time; _NEUTRAL
            when the step is skipped or the operator declines.
    """
    print("\n===== STEP: transfer (inter-shard collateral round trip) =====")

    # Refuse a self-transfer BEFORE any I/O, and before any POST. The two
    # existing guards below (is the shard advertised, are its transfers active)
    # are both satisfied by the source shard by construction, so --dest-shard 0
    # used to POST a real, non-idempotent, never-retried transfer whose source
    # and destination were the same shard. Net-zero, so no money can go
    # missing, but it cannot raise the shard's balance by the probe cent the
    # settlement target demands: the poll burned the full
    # TRANSFER_SETTLE_TIMEOUT_SECONDS and then reported a FALSE "MONEY MAY BE
    # IN FLIGHT" (DR-22). Guarded here rather than in argparse because
    # _step_transfer is what spends the money and is reachable from main()'s
    # step dispatch and from any future caller, not only from the CLI.
    if dest_shard == _TRANSFER_SOURCE_SHARD:
        print(
            f"{_NEUTRAL}: --dest-shard {dest_shard} IS the source shard "
            f"({_TRANSFER_SOURCE_SHARD}), so this would POST a self-transfer that cannot "
            "raise that shard's balance — the settlement poll would then burn "
            f"{config.TRANSFER_SETTLE_TIMEOUT_SECONDS}s and report money in flight that "
            "never moved. Nothing was submitted; pick a different destination shard."
        )
        return _NEUTRAL

    # Same read main.py does before every run; None means single-shard semantics.
    statuses = scanner.fetch_shard_statuses(client)
    if statuses is None:
        print(
            f"{_NEUTRAL}: no per-shard status breakdown from /exchange/status — this account "
            "or environment is pre-sharding, so there is no second shard to transfer to."
        )
        return _NEUTRAL
    _emit("SHARD STATUSES", statuses)
    if dest_shard not in statuses:
        print(
            f"{_NEUTRAL}: the exchange advertises no shard {dest_shard} "
            f"(advertised: {sorted(statuses)}) — nothing to transfer to."
        )
        return _NEUTRAL
    # trader's own gate: never move money to or from a shard whose transfers
    # the exchange reports as inactive.
    if not trader._transfers_active(statuses, _TRANSFER_SOURCE_SHARD) or not (
        trader._transfers_active(statuses, dest_shard)
    ):
        print(
            f"{_NEUTRAL}: intra-exchange transfers are not active on shard "
            f"{_TRANSFER_SOURCE_SHARD} and/or {dest_shard} — skipping."
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

    # THE body trader._execute_transfer will send — built by the same function
    # (trader._transfer_body), never hand-copied, so the printed evidence can
    # never diverge from the request actually made.
    preview = trader._transfer_body(_TRANSFER_SOURCE_SHARD, dest_shard, _TRANSFER_PROBE_CENTS)
    _emit("REQUEST BODY (outbound transfer, as trader._execute_transfer builds it)", preview)
    if not _confirm(
        assume_yes,
        f"Move {_TRANSFER_PROBE_CENTS}c ({preview['amount']} centicents) from shard "
        f"{_TRANSFER_SOURCE_SHARD} to shard {dest_shard}, then move it back. "
        "Transfers are NOT retried and NOT idempotent — one attempt each.",
    ):
        return _NEUTRAL

    out_id = _transfer_leg(client, _TRANSFER_SOURCE_SHARD, dest_shard, "outbound")
    if out_id is False:
        return _FAIL
    # Wait for the ASYNCHRONOUS settlement using the real bounded poller: the
    # destination must hold its prior balance plus the cent we sent.
    target_out = {dest_shard: before.get(dest_shard, 0) + _TRANSFER_PROBE_CENTS}
    mid = trader._await_transfer_settlement(client, target_out)
    _emit("BALANCE BY SHARD, CENTS (after outbound)", mid)
    if mid.get(dest_shard, 0) < target_out[dest_shard]:
        print(
            f"{_FAIL}: the outbound transfer did not settle within "
            f"{config.TRANSFER_SETTLE_TIMEOUT_SECONDS}s (transfer_id={out_id}). *** MONEY MAY "
            "BE IN FLIGHT — CHECK THE ACCOUNT. *** Not sending it back; re-run once the "
            "balances have settled."
        )
        return _FAIL
    print(f"Outbound leg settled: {_TRANSFER_PROBE_CENTS}c landed on shard {dest_shard}.")

    # No _confirm here, deliberately. The operator approved the ROUND TRIP at
    # the top of this step, and by now the cent is sitting on the far shard:
    # stopping to ask would strand it there on a declined or unattended prompt,
    # which is strictly worse than completing the trip that was authorised
    # (TS-30).
    back_id = _transfer_leg(client, dest_shard, _TRANSFER_SOURCE_SHARD, "return")
    if back_id is False:
        print(
            f"*** {_TRANSFER_PROBE_CENTS}c is stranded on shard {dest_shard} — move "
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
        f"{dest_shard} -> {_TRANSFER_SOURCE_SHARD} (transfer_ids: {out_id}, "
        f"{back_id}). The centicent unit and the asynchronous settle timing are confirmed."
    )
    return _PASS


# Step name -> implementation. Every step shares the (client, ticker,
# assume_yes, dest_shard) signature so main() can dispatch without special
# cases.
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
            (no verdict reached — a TRUE kill with the account flat, no
            liquidity, skipped, or aborted at the confirmation prompt; a
            partial or otherwise non-conforming fill-or-kill response is a
            FAIL, not a NEUTRAL — DR-20). Only a 0 from BOTH no-mapping and
            unfillable-ask is evidence that the V2 order path may be trusted
            to run unsupervised.
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
        "--dest-shard", type=int, default=_TRANSFER_DEST_SHARD_DEFAULT,
        help=(
            "Destination shard for --step transfer (default: 1, the combos shard). Must "
            f"not be the source shard ({_TRANSFER_SOURCE_SHARD}) — a self-transfer is "
            "refused before any POST."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation before each submission.",
    )
    args = parser.parse_args(argv)

    # The scanner/trader/auth calls below log at INFO (order-book key mismatches,
    # transfer acceptance, submissions); without a handler those lines would be
    # dropped, and they belong in the evidence log next to the bodies.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 78)
    print("KALSHI V2 ORDER-PATH LIVE PROBE — REAL MONEY, PRODUCTION ACCOUNT")
    print(f"step={args.step}  ticker={args.ticker}  worst-case exposure per order: "
          f"{PROBE_COUNT_STR} contracts (about one cent)")
    print("Every request body and response below is the evidence log for trusting "
          "the V2 order path — keep it.")
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

    outcome = _STEPS[args.step](client, args.ticker, args.yes, args.dest_shard)

    print(f"\n================ RESULT: {args.step} -> {outcome} ================")
    if outcome != _PASS:
        print(
            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy order '
            "path. Both --step no-mapping and --step unfillable-ask have to PASS before "
            "the V2 order path should be trusted to run unsupervised."
        )
    else:
        print(
            "Record this output. The V2 order path is only known-good once BOTH "
            "--step no-mapping and --step unfillable-ask have PASSED."
        )
    return _EXIT_CODES[outcome]


if __name__ == "__main__":
    sys.exit(main())
