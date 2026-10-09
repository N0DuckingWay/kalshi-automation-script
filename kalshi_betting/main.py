"""
File: main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Runs the live trading bot from the command line. Dev mode scans the
    sandbox exchange and writes a simulation to Excel. Prod mode scans the
    real exchange, finds pairs of contracts to trade (same-title pairs: one
    question listed twice at different prices; time-series pairs: one
    question at two deadlines), sizes each trade with the Kelly rule (a
    formula for how much of the money to bet), sends the orders and adds the
    results to the Excel trade log. The exit code tells the scheduler how the
    run went (the EXIT_* constants in config.py); an unexpected error exits 1.

    A production run sizes each trade on the account's portfolio value (the
    cash on every exchange shard plus what Kalshi says the open positions are
    worth) but never spends more than the cash. The minimum-balance check
    reads the portfolio value. Dev mode holds nothing, so its virtual
    --sandbox-balance counts as both.

    Kalshi's value of the open positions counts only when the contracts held
    can back it: a contract pays at most $1 if it wins
    (config.CONTRACT_PAYOUT_DOLLARS), so a larger value, or one the positions
    listing cannot be checked against, is refused with a WARNING and the run
    sizes on the cash alone (_checked_positions_value). A value of 0 is always
    kept: it adds nothing.

    main() exits 2 before logging starts if config.ORDER_API_VERSION is not
    "v2". A production run that sends orders takes the machine-wide live-run
    lock (run_lock) before it builds a client and holds it until main() ends,
    so two such runs never trade at once; if another run holds it, this one
    exits 50 without making any request. Dry runs and dev runs skip the lock.

    With --result-file PATH (prod only) the run deletes any file at PATH, then
    writes a JSON summary there when it ends: its exit code, closing line,
    balances, whether it began sending orders, each sale's and each pair's
    outcome, its warnings and any error. A usage error before logging starts,
    or a kill signal, leaves no file, so a caller reads the exit code first.

    The live settings come from the saved live defaults (live_defaults.json);
    a toggle flag overrides one of them for this run only. With no saved
    file, a refused file or a bad flag, the run exits 2 before logging
    anything; it never falls back to config.py's values. The scheduler passes
    no toggle flags. Between finding pairs and pricing them, _dedup_pairs and
    _filter_by_category trim the list.

    A production run sells before it buys. With sell_at set, it sells each
    held position that has stayed at or above that share of its potential
    profit (what it pays if it wins, less its cost) at every daily check
    (seller.plan_sales decides, trader.sell_positions sends the orders) and
    writes the sales to the trade log before it buys. A live run that sold
    reads its positions and cash again and sizes the buys on what is left;
    a dry run sends nothing and adds each sale's estimated proceeds to the
    cash. The markets sold are not bought again that run, and no position
    picked for sale is added to. A sale left uneven, of unknown outcome, or
    found to have sold more than it recorded makes the run exit 20
    (EXIT_TRADES_NEED_ATTENTION).

    A production run keeps every market the account holds out of new trades,
    except, when the run's add_to_held_pairs setting is on, the markets it
    may add to (scanner.held_pairs): the two markets of an exact held pair,
    which each finder lets through only as that same pair, and a lone held
    leg whose partner has paid out, which each finder lets through only
    beside a market the account does not hold (add_on_pairs). It adds to
    none when Kalshi's value of the open positions was not read or was
    refused, since an add-on is sized on the portfolio value.

Dependencies:
    Imports auth.py (the client, and the one balance read, which also checks
    the credentials), config.py (constants, exit codes, the order-path check
    and the live-settings helpers), historical.py (Kalshi's series categories,
    for the category/tag filter), reporter.py (Excel output and the run
    summary), _http.py (a one-line description of an error), scanner.py
    (finding markets and pairs, and pricing them from the order book),
    seller.py (plan_sales: which held positions to sell), strategy.py
    (sizing and choosing trades), trader.py (sending orders, sell_positions
    included) and run_lock.py (the one-real-money-run-at-a-time lock). Run as
    `python3 -m kalshi_betting.main`.

    To add to held pairs it also reads scanner.get_held_positions,
    resolve_held_ladders and held_pairs (the positions, their ladders and the
    exact held pairs and lone held legs a run may add to), scanner.pair_held (which names the
    held pair a trade adds to in the pairs table, the portfolio lines and the
    rescue dump), config.held_pair_fraction (the one definition of an
    add-on's size, which leaves out a held pair with no room left),
    config.count_text (which writes a held count exactly) and
    config.CONTRACT_PAYOUT_DOLLARS (what one contract pays at most, the
    bound of the positions-value check).

Notes:
    Label rule for everything this module logs: "A"/"B" always mean
    market_a/market_b, and the pairs table, _print_portfolio and the rescue
    dump render legs in MARKET order (A then B) with the bought side next to
    each count. The trader's own execution logs list legs in SUBMISSION order
    (the NO leg first) — the two views describe the same trade.

    Both run modes read the exchange's per-shard status breakdown
    (scanner.fetch_shard_statuses) before fetching markets and pass the set of
    trading-inactive shards into the fetch. Markets on every other shard are
    ingested and tagged with their exchange_index — market data is cross-shard.
    The full parsed status dict is kept in a local (`shard_statuses`) because
    prod reads more than trading_active from it: it is handed to
    trader.ensure_shard_collateral(), which refuses to move collateral to or
    from a shard whose intra_exchange_transfers_active is false. Immediately
    after the fetch, `_log_shard_coverage` (wrapping
    scanner.check_shard_coverage) compares that advertised breakdown against
    the shards actually observed in ingested markets (and, in prod, the
    balance breakdown) and logs any gap at CRITICAL or WARNING — see the
    function docstring for the empty-vs-funded severity split. A run only
    claims "Full shard coverage" when the breakdown exists and nothing was
    wrong; the run always continues regardless of what the check finds.
"""
import argparse
import contextlib
import logging
import logging.handlers
import math
import os
import pathlib
import sys
import time
from collections import Counter
from dataclasses import replace as dc_replace
from datetime import UTC, datetime

from tabulate import tabulate

from . import run_lock
from ._http import api_error_summary
from .auth import build_client, read_account_balance
from .config import (
    CONTRACT_PAYOUT_DOLLARS,
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    LIVE_DEFAULTS_FROM_CONFIG,
    MIN_BALANCE_CENTS,
    PROJECT_ROOT,
    SALE_READ_BACK_RECHECK_SECONDS,
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS,
    SAME_TITLE_MIN_PRICE_DIFF,
    SELL_AT_STEP,
    SIZE_CAP_STEP,
    V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS,
    LiveDefaultsError,
    LiveDefaultsMissing,
    LiveSettings,
    count_text,
    describe_live_settings,
    describe_time_series_rule,
    describe_trade_filter,
    fee_leg_exact,
    held_pair_fraction,
    live_defaults,
    live_rule_warnings,
    live_settings,
    order_api_version_error,
)
from .historical import infer_category, load_series_categories, series_labels
from .reporter import (
    RunReport,
    RunReportHandler,
    append_to_prod_log,
    report_sales,
    report_trades,
    write_dev_simulation,
    write_run_report,
)
from .scanner import (
    check_shard_coverage,
    close_gap_bound_text,
    display_title,
    enrich_with_orderbook_prices,
    fetch_open_events_with_markets,
    fetch_shard_statuses,
    filter_markets_within_horizon,
    find_same_title_pairs,
    find_time_series_pairs,
    get_held_positions,
    held_pairs,
    inactive_shard_indexes,
    leg_sides,
    pair_held,
    resolve_held_ladders,
)
from .seller import plan_sales
from .strategy import compute_trade, select_portfolio
from .trader import (
    ensure_shard_collateral,
    execute_trades,
    pre_execution_check,
    sell_positions,
)


def _truncate(text: str, n: int = 40) -> str:
    """
    Truncate text to at most n characters, appending an ellipsis if truncated.

    Args:
        text (str): Input string to truncate.
        n (int): Maximum number of characters to keep. Defaults to 40.

    Returns:
        str: The original string if len(text) <= n, otherwise text[:n] + "…".
    """
    return text[:n] + "…" if len(text) > n else text



def _format_deadline(dt) -> str:
    """
    Format a market close datetime as a "YYYY-MM-DD" string, or "?" if None.

    Args:
        dt: A datetime object representing the market deadline, or None.

    Returns:
        str: ISO date string "YYYY-MM-DD" if dt is not None, otherwise "?".
    """
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _bankroll_cents(cash_cents: int, positions_value_cents: int | None) -> int:
    """
    Return the portfolio value: the cash plus what Kalshi says the open
    positions are worth, in cents.

    If the positions' value could not be read, returns the cash alone and logs
    a WARNING; that can only make trades smaller, never larger.

    Args:
        cash_cents (int): The cash on every shard together, in cents.
        positions_value_cents (int | None): The open positions' value in cents; None if unread.

    Returns:
        int: The cash plus the positions' value, or the cash alone when that value is None.
    """
    if positions_value_cents is None:
        logging.warning(
            "Kalshi's balance reply carried no readable portfolio_value — sizing on "
            "cash alone ($%.2f) this run, as if no position were held",
            cash_cents / 100,
        )
        return cash_cents
    return cash_cents + positions_value_cents


def _checked_positions_value(cash_cents: int, positions_value_cents: int | None,
                             held_positions: dict, listing_complete: bool, *,
                             not_shown: list | tuple = (),
                             value_before_sales: int | None = None) -> int | None:
    """
    Return Kalshi's value of the open positions if the contracts held can back it.

    A contract pays at most config.CONTRACT_PAYOUT_DOLLARS ($1) if it wins, so
    the open positions can be worth at most that much per contract held. The
    value is kept when it is no more than that, counted over a positions
    listing read to its end whose every count can be read. A value of 0 adds
    nothing to the portfolio value and is always kept. An unread value (None)
    is returned as it is, with no word here: _bankroll_cents has already
    counted it as $0 and said so. Any other value is refused: one WARNING
    names it, the contracts held and the reason, and the run sizes on the
    cash alone. The check keeps a value in the wrong unit, or otherwise too
    large, from making every trade too big. After a live run's sales the
    value is also refused while the listing does not yet show every sale
    (not_shown), or when it is not below the value kept at the start of the
    run (value_before_sales), since either way it may still count what was
    sold.

    Args:
        cash_cents (int): The cash on every shard together, in cents, named in the WARNING.
        positions_value_cents (int | None): Kalshi's value of the open
            positions, in cents; None when the balance reply had none.
        held_positions (dict): scanner.get_held_positions' result, ticker -> HeldPosition.
        listing_complete (bool): Whether that listing was read to its end.
        not_shown (list | tuple): Keyword-only. The markets whose listed
            position does not yet show this run's sale (_listing_misses_sales);
            empty at the start of a run.
        value_before_sales (int | None): Keyword-only. The value this check
            kept at the start of the run, in cents, passed after sales that
            sold or may have sold contracts; None otherwise.

    Returns:
        int | None: positions_value_cents when it may count in the portfolio
            value; None when it was not read, or is refused.
    """
    if positions_value_cents is None or positions_value_cents == 0:
        # Not read, or nothing to count: nothing to check
        return positions_value_cents
    counts = [position.count for position in held_positions.values()]
    if not listing_complete:
        reason = ("the list of the account's positions was cut short, so not every "
                  "contract held is known")
    elif not_shown:
        reason = (f"the list of the account's positions does not yet show this run's "
                  f"sales on {', '.join(not_shown)}, so the value may still count what "
                  f"was sold")
    elif value_before_sales is not None and positions_value_cents >= value_before_sales:
        # Selling turns positions into cash, so the value should fall. Only a
        # value that did not fall at all is caught; one that fell by less than
        # what was sold passes
        reason = (f"this run's sales sold, or may have sold, contracts, yet it is not "
                  f"below the ${value_before_sales / 100:.2f} it was before them, so it "
                  f"may still count what was sold")
    elif None in counts:
        reason = (f"the contract count of {sum(c is None for c in counts)} held "
                  f"market(s) could not be read")
    else:
        contracts = sum(abs(count) for count in counts)
        # Rounded to a millionth of a cent, so float noise in the sum of the
        # counts never refuses a value that is exactly at the bound
        if positions_value_cents <= round(contracts * CONTRACT_PAYOUT_DOLLARS * 100, 6):
            return positions_value_cents
        reason = (f"that is more than the {count_text(contracts)} contract(s) held can "
                  f"be worth at ${CONTRACT_PAYOUT_DOLLARS:.2f} each")
    logging.warning(
        "Kalshi's value of the open positions ($%.2f) is not used: %s — sizing on "
        "cash alone ($%.2f) this run, as if no position were held",
        positions_value_cents / 100, reason, cash_cents / 100,
    )
    return None


# The sale statuses that need a person to look at the account: a pair the
# sale left uneven, or one whose sale outcome could not be known
_SALE_ATTENTION_STATUSES = ("unbalanced", "manual_review")

# How far apart two contract counts may be and still be read as equal: counts
# are whole contracts, read from fixed-point strings as floats
_COUNT_TOLERANCE = 1e-6


def _sold_tickers(sales: list, *, dry_run: bool) -> set:
    """
    The held markets this run's sales keep out of every purchase for the rest of the run.

    A dry run counts every held market of a "simulated" sale, as if its
    orders had filled. A live run counts every market an order is known to
    have sold on, plus every held market of a sale left for a person
    (_SALE_ATTENTION_STATUSES), since some of it may have sold or what is
    left needs sorting out. Any market here also makes a live run read its
    cash and positions again before it buys. A market whose sale sold
    nothing is still held, and stays out of new trades like any held market.

    Args:
        sales (list): trader.sell_positions' results (reporter.SaleResult).
        dry_run (bool): Keyword-only. True for a run that sends no orders.

    Returns:
        set: Their tickers.
    """
    tickers: set = set()
    for sale in sales:
        if dry_run:
            if sale.status == "simulated":
                tickers |= set(sale.sold)
            continue
        tickers |= {ticker for ticker, count in sale.sold.items() if count}
        if sale.status in _SALE_ATTENTION_STATUSES:
            tickers |= {leg.ticker for leg in getattr(sale.plan, "legs", ())
                        if getattr(leg, "market", None) is not None}
    return tickers


def _planned_tickers(sales: list) -> set:
    """
    Every held market of every position the take-profit rule picked this run.

    A position picked for sale is never added to in the same run, even when
    its orders sold nothing, so a live run and a dry run of one account offer
    the same add-ons.

    Args:
        sales (list): trader.sell_positions' results (reporter.SaleResult).

    Returns:
        set: The tickers of their plans' held markets (a paid-out partner,
            which has no market, is left out).
    """
    return {leg.ticker for sale in sales for leg in getattr(sale.plan, "legs", ())
            if getattr(leg, "market", None) is not None}


def _account_decided_short(sales: list) -> list:
    """
    The markets of sales whose count an account reading decided and came up short.

    When a sale order's reply does not say how many it sold, the account's
    position decides (SaleResult.decided_by_account), and a ledger that
    trails the fills shows too few sold. Such a sale that reads as partly
    sold, not sold or unbalanced may have sold more, so the run checks it.

    Args:
        sales (list): A live run's sale results (reporter.SaleResult).

    Returns:
        list: Those sales' held tickers (the keys of SaleResult.sold), sorted.
    """
    return sorted({ticker for sale in sales
                   if getattr(sale, "decided_by_account", False) is True
                   and sale.status in ("partly_sold", "not_sold", "unbalanced")
                   for ticker in sale.sold}, key=str)


def _sale_attention_note(sales: list) -> str:
    """
    Give the words a run's closing line ends with when a sale needs a person to check it.

    Args:
        sales (list): trader.sell_positions' results (reporter.SaleResult).

    Returns:
        str: " Sales that need a person to check: N (see the CRITICAL lines
            above)." when any sale is unbalanced or its outcome unknown
            (_SALE_ATTENTION_STATUSES); "" otherwise.
    """
    count = sum(1 for sale in sales if sale.status in _SALE_ATTENTION_STATUSES)
    return (f" Sales that need a person to check: {count} (see the CRITICAL lines above)."
            if count else "")


def _with_simulated_proceeds(sales: list, cash_cents: int,
                             shard_balances: dict) -> tuple[int, dict]:
    """
    Add what a dry run's sales would return to the cash, as if they had filled.

    Each held market of a "simulated" sale returns its contracts times the
    average bid the sale was priced at (SalePlan.walked), less the taker fee,
    floored to the whole cent so the cash is never overstated. It goes on the
    cash total and on the market's own shard (exchange_index). Other sales
    add nothing.

    Args:
        sales (list): trader.sell_positions' results in a dry run.
        cash_cents (int): The cash on every shard together, in cents.
        shard_balances (dict): Exchange index -> cash on that shard, in cents.

    Returns:
        tuple[int, dict]: The new cash total, and a new per-shard dict (the
            one passed in is left as it was).
    """
    shards = dict(shard_balances)
    added = 0
    for sale in sales:
        if sale.status != "simulated":
            continue
        plan = sale.plan
        for leg in plan.legs:
            if leg.market is None:
                continue  # a paid-out partner has nothing to sell
            count = sale.sold.get(leg.ticker, 0)
            average = plan.walked[leg.ticker][0]
            # Cross-module: the one fee rule, on selling `count` at that price
            dollars = count * average - fee_leg_exact(count, average)
            # Rounded to a millionth of a cent first, so float noise never
            # costs a cent, then down to the whole cent
            cents = max(0, math.floor(round(dollars * 100, 6)))
            shard = leg.market.exchange_index
            shards[shard] = shards.get(shard, 0) + cents
            added += cents
    return cash_cents + added, shards


def _record_sales(sales: list, cash_before_cents: int, cash_after_cents: int,
                  run_note: str) -> None:
    """
    Add this run's sales to the trade log, or dump them to the log if that fails.

    Called before anything is bought, so a filled sale is on record however
    the run ends. If the write fails, every sale is dumped as a CRITICAL
    RESCUE line and the error is raised again, which stops the run before it
    buys.

    Args:
        sales (list): trader.sell_positions' results; not empty.
        cash_before_cents (int): The cash on every shard before the sales, in cents.
        cash_after_cents (int): The same after them, as read back from
            Kalshi. A dry run, which sent nothing, passes the cash before again.
        run_note (str): The trade log banner's note for this run.

    Raises:
        Exception: Whatever writing the trade log raised, after the rescue dump.
    """
    try:
        # Cross-module: one banner row, then one row per sale (reporter._sale_to_row)
        append_to_prod_log([], cash_before_cents / 100, cash_after_cents / 100,
                           run_note=run_note, sales=sales)
    except Exception as exc:
        logging.critical("Failed to write the sales to the trade log: %s — rescue dump "
                         "follows", exc)
        for sale in sales:
            logging.critical("  RESCUE SALE | %s | %s | sold %s | %s",
                             sale.status, getattr(sale.plan, "title", sale.plan),
                             sale.sold, sale.error or "")
        raise


def _listing_misses_sales(sales: list, before: dict, positions: dict, *,
                          listing_complete: bool = True) -> tuple[list, list]:
    """
    Compare the listed positions with what this run's sales recorded as sold there.

    A market an order sold n contracts on should now hold n fewer than at the
    start of the run. Fewer sold than recorded means the sale does not show
    yet (the positions ledger trails a fill). More sold than recorded means,
    for a sale whose count an account reading decided
    (SaleResult.decided_by_account), that the reading trailed the fills;
    such a sale's markets are checked even where it recorded 0. For a sale
    whose replies gave every count, either difference reads as not shown yet.

    A market is not checked when a count is unreadable or unknown, or when it
    is missing from a listing cut short (it may still be held). Missing from
    a listing read to its end means it holds nothing.

    Args:
        sales (list): A live run's sale results (reporter.SaleResult).
        before (dict): The positions listing read at the start of the run,
            ticker -> HeldPosition.
        positions (dict): The listing read after the sales, the same shape.
        listing_complete (bool): Keyword-only. Whether that listing was read
            to its end (scanner.get_held_positions' complete_out).

    Returns:
        tuple[list, list]: (markets whose sale does not show yet, markets of
            account-decided sales showing more sold than recorded), each
            sorted; both empty when every sale shows as recorded.
    """
    not_shown, more_sold = set(), set()
    for sale in sales:
        by_account = getattr(sale, "decided_by_account", False) is True
        for ticker, count in sale.sold.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                continue
            if count == 0 and not by_account:
                continue
            held = before.get(ticker)
            now = positions.get(ticker)
            if held is None or held.count is None or (now is not None and now.count is None):
                continue
            if now is None and not listing_complete:
                # Missing from a listing cut short: still held, or not, unknown
                continue
            # How far the listing says the position moved toward zero since
            # the run started: a YES holding falls toward 0, a NO one rises
            shown = ((held.count - (0.0 if now is None else now.count))
                     * math.copysign(1.0, held.count))
            if shown < count - _COUNT_TOLERANCE:
                not_shown.add(ticker)
            elif shown > count + _COUNT_TOLERANCE:
                (more_sold if by_account else not_shown).add(ticker)
    return sorted(not_shown, key=str), sorted(more_sold, key=str)


def _positions_after_sales(client, sales: list, before: dict,
                           held_listing: dict) -> tuple[dict, list, list]:
    """
    Read the positions again after a live run's sales, waiting while a sale may not show yet.

    Kalshi's positions ledger can trail a fill, so when the listing differs
    from what the sales recorded (_listing_misses_sales) it is read once more
    after config.SALE_READ_BACK_RECHECK_SECONDS. A sale whose account-decided
    count came up short (_account_decided_short) waits longer instead: the
    reading that set its count may have trailed the fills, and the listing
    reads the same ledger. The listing is then read once after each pause of
    config.V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS (1, 2 and 4 seconds),
    stopping early once a read after a pause shows more sold. A ledger still
    trailing after the last pause leaves the short count standing,
    unflagged. The last comparison is returned; _run_prod reads the balance
    only after this.

    Args:
        client: The production KalshiClient.
        sales (list): The run's sale results (reporter.SaleResult).
        before (dict): The positions listing read at the start of the run.
        held_listing (dict): Its "complete" key is set to whether the last
            listing read was read to its end.

    Returns:
        tuple[dict, list, list]: The positions (ticker -> HeldPosition), then
            _listing_misses_sales' two lists from the last read.
    """
    # Cross-module: the one reader of the positions listing, as at the start of the run
    positions = get_held_positions(client, complete_out=held_listing)
    not_shown, more_sold = _listing_misses_sales(
        sales, before, positions, listing_complete=held_listing.get("complete") is True)
    short = _account_decided_short(sales)
    if short:
        # A short account reading may have trailed the fills: the longer wait
        pauses = V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS
    elif not_shown or more_sold:
        pauses = (SALE_READ_BACK_RECHECK_SECONDS,)
    else:
        pauses = ()
    for number, pause in enumerate(pauses):
        if number and more_sold:
            # A read after a wait shows more sold: that sale gets checked
            # (_flag_sales_sold_more), so there is nothing left to wait for
            break
        if not_shown:
            logging.info("The positions listing does not yet show this run's sales on %s; "
                         "reading it again in %g s", ", ".join(not_shown), pause)
        elif more_sold:
            logging.info("The positions listing shows more sold than this run recorded on "
                         "%s; reading it again in %g s", ", ".join(more_sold), pause)
        else:
            logging.info("This run's sales on %s were counted from the account's "
                         "positions, which can trail a fill; reading the positions "
                         "listing again in %g s to check them", ", ".join(short), pause)
        time.sleep(pause)
        positions = get_held_positions(client, complete_out=held_listing)
        not_shown, more_sold = _listing_misses_sales(
            sales, before, positions, listing_complete=held_listing.get("complete") is True)
    return positions, not_shown, more_sold


def _flag_sales_sold_more(sales: list, more_sold: list, before: dict,
                          positions: dict) -> list:
    """
    Turn each sale the listing shows more sold on than recorded into one a person must check.

    Such a sale's count came from an account reading that trailed the fills,
    so a pair's second order then sold too few, or nothing, and the position
    may now be uneven. Each such sale logs one CRITICAL naming each market,
    what the run recorded and what the account shows, and becomes
    "manual_review" with an error saying the same: the run exits
    EXIT_TRADES_NEED_ATTENTION and its held markets stay out of every
    purchase (_sold_tickers). The CRITICAL and the error say they replace
    what the trader logged on the short count (an "unbalanced" sale's extra
    to sell by hand, say).

    Args:
        sales (list): A live run's sale results (reporter.SaleResult).
        more_sold (list): The markets the listing shows more sold on than recorded.
        before (dict): The positions listing read at the start of the run.
        positions (dict): The listing read after the sales.

    Returns:
        list: The sales in the same order, each one with such a market
            replaced by a "manual_review" copy; the others as they were.
    """
    flagged = []
    for sale in sales:
        tickers = sorted((t for t in sale.sold if t in more_sold), key=str)
        if not tickers:
            flagged.append(sale)
            continue
        parts = []
        for ticker in tickers:
            held = before[ticker].count
            now = positions.get(ticker)
            # Missing here only from a listing read to its end (a listing cut
            # short names no market it misses), so it holds nothing
            left = 0.0 if now is None else now.count
            side = "YES" if held > 0 else "NO"
            # config.count_text: a contract count written exactly
            parts.append(f"{ticker} held {count_text(abs(held))} {side}: this run recorded "
                         f"{count_text(sale.sold[ticker])} sold, the account now shows "
                         f"{count_text(abs(left))} left "
                         f"({count_text(abs(held) - abs(left))} sold)")
        mismatch = "the account shows more sold than recorded: " + "; ".join(parts)
        logging.critical(
            "SALE RECORDED SHORT for '%s': %s. The count was read from the account's "
            "positions, which trailed the fills, so the position may now be uneven. "
            "This replaces every earlier line about this sale, including any extra it "
            "said to sell by hand: read what each of its markets holds now in the Kalshi "
            "UI before acting; none is traded again this run. Manual review required.",
            getattr(sale.plan, "title", sale.plan), "; ".join(parts),
        )
        error = (mismatch if not sale.error
                 else f"{mismatch} (this replaces what was recorded before the check: "
                      f"{sale.error})")
        flagged.append(dc_replace(sale, status="manual_review", error=error))
    return flagged


def _value_after_sales(cash_cents: int, positions_value_cents: int | None,
                       held_positions: dict, listing_complete: bool,
                       not_shown: list, *,
                       value_before_sales: int | None = None) -> tuple[int | None, int]:
    """
    Check the open positions' value after a live run's sales, and work out the portfolio value.

    The value goes through _checked_positions_value with its after-sales
    checks (not_shown, value_before_sales). The portfolio value is the cash
    plus that value, or the cash alone when it was not read or is refused.
    The minimum-balance check is not applied again: the run passed it, and a
    sale only turns a position into cash. One INFO line gives all three
    numbers.

    Args:
        cash_cents (int): The cash on every shard, read after the sales, in cents.
        positions_value_cents (int | None): Kalshi's value of the open
            positions from that same read, in cents; None when not read.
        held_positions (dict): The listing read after the sales, ticker -> HeldPosition.
        listing_complete (bool): Whether that listing was read to its end.
        not_shown (list): The markets whose sale the listing does not show
            yet (_positions_after_sales).
        value_before_sales (int | None): Keyword-only. The value kept at the
            start of the run, in cents, when the sales sold or may have sold
            contracts; None otherwise.

    Returns:
        tuple[int | None, int]: The positions' value as checked (None when not
            read or refused) and the portfolio value, in cents.
    """
    checked_value_cents = _checked_positions_value(
        cash_cents, positions_value_cents, held_positions, listing_complete,
        not_shown=not_shown, value_before_sales=value_before_sales,
    )
    if positions_value_cents is not None and checked_value_cents is None:
        # Refused (its WARNING is logged): the cash alone, as at the start
        portfolio_value_cents = cash_cents
        counted = "not used (counted as $0.00)"
    else:
        portfolio_value_cents = _bankroll_cents(cash_cents, positions_value_cents)
        counted = ("not read (counted as $0.00)" if positions_value_cents is None
                   else f"${positions_value_cents / 100:.2f}")
    logging.info("After the sales: portfolio value $%.2f = cash $%.2f + open positions %s",
                 portfolio_value_cents / 100, cash_cents / 100, counted)
    return checked_value_cents, portfolio_value_cents


def _display_specs(trade_specs: dict, portfolio: list) -> dict:
    """
    Match each selected trade back to the candidate pair it came from, for the
    pairs table.

    Trades are matched by their two tickers, not by object, because
    select_portfolio returns a new, smaller copy of a trade it shrank to fit
    the cash. No two candidates share both tickers.

    Args:
        trade_specs (dict): id(candidate pair) -> TradeSpec, from _compute_trade_specs.
        portfolio (list): The TradeSpecs select_portfolio returned.

    Returns:
        dict: id(candidate pair) -> the selected TradeSpec, for selected pairs only.
    """
    selected = {frozenset((s.pair.market_a.ticker, s.pair.market_b.ticker)): s
                for s in portfolio}
    shown: dict = {}
    for pid, spec in trade_specs.items():
        chosen = selected.get(frozenset((spec.pair.market_a.ticker,
                                         spec.pair.market_b.ticker)))
        if chosen is not None:
            shown[pid] = chosen
    return shown


def _compute_trade_specs(
    candidate_pairs: list, portfolio_value_cents: int, settings: LiveSettings, *,
    cash_cents: int,
) -> dict:
    """
    Size a trade for every candidate pair that qualifies.

    Args:
        candidate_pairs (list): CandidatePair objects to size.
        portfolio_value_cents (int): The portfolio value in cents; each trade bets a share of it.
        settings (LiveSettings): The run's settings, the same ones enrichment used.
        cash_cents (int): Keyword-only. The cash on hand in cents; no trade spends more.

    Returns:
        dict: id(pair) -> TradeSpec for each pair that produced a trade.
    """
    specs: dict = {}
    for pair in candidate_pairs:
        # Size on the portfolio value, spending at most the cash; None if the pair does not qualify
        spec = compute_trade(pair, portfolio_value_cents, settings=settings,
                             cash_cents=cash_cents)
        if spec is not None:
            specs[id(pair)] = spec
    return specs


def _print_portfolio(portfolio: list, label: str) -> None:
    """
    Log a summary of selected portfolio trades to the log file.

    Legs are rendered in MARKET order (A then B) with the side bought on each
    market next to its count — e.g. "5× YES(A) + 5× NO(B)" for a time-series
    pair, "5× NO(A) + 5× YES(B)" for a same-title pair. The sides come from
    scanner.leg_sides, never from the pair type by hand. Each line also names
    both tickers: a time-series pair's canonical title now ends with the
    outcome label (scanner.time_series_group_key) and the 55-character cut
    applied below drops it, so two strikes of one daily family would otherwise
    log an identical line (DR-17). "profit if won" is
    spec.min_payoff: the guaranteed floor for a same-title pair, and the
    profit in either winning settlement of a time-series pair (event by A, or
    never by B) — the in-between settlement loses the whole stake. A trade
    that adds to a pair the account already holds (scanner.pair_held) ends
    " — adds to N held", N being the contracts held on each market; every
    other line is unchanged.

    Args:
        portfolio (list): List of TradeSpec objects representing the trades
            selected for execution.
        label (str): Header label printed before the trade list (e.g.
            "Executing" or "Dry-run:").

    Returns:
        None
    """
    logging.info("%s %d trade(s):", label, len(portfolio))
    for spec in portfolio:
        # Which side each market's leg buys — the only source of truth for sides
        side_a, side_b = leg_sides(spec.pair.pair_type)
        # Cross-module: the held pair this trade adds to, read by type
        held = pair_held(spec.pair)
        logging.info(
            "  [%s] %s (%s / %s) — %d× %s(A) + %d× %s(B) — "
            "cost $%.2f incl. fees, profit if won $%.2f (%.1f%% return)%s",
            spec.pair.pair_type,
            spec.pair.canonical_title[:55],
            # DR-17: the tickers identify the trade when the title cannot —
            # the 55-character cut above drops the outcome label a time-series
            # canonical title ends with, so two strikes of one daily family
            # produced the same line in the review of a dry run
            spec.pair.market_a.ticker, spec.pair.market_b.ticker,
            spec.x, side_a.upper(), spec.y, side_b.upper(),
            spec.total_cost_with_fees, spec.min_payoff,
            spec.profit_ratio * 100,
            f" — adds to {count_text(held.count)} held" if held is not None else "",
        )


def _no_pairs_msg(sandbox: bool = False, settings: LiveSettings | None = None, *,
                  time_series_searched: bool = True) -> str:
    """
    Build the "no qualifying pairs found" log message with the run's live rule.

    Words the time-series rule as the finder's "Time-series entry rule" line
    does, and names the cumulative-deadline requirement so a reader does not
    blame the price rule for what the wording decided. Names the
    same-title pairing rules for the same reason: a same-title pair also needs
    two different event series whose markets close within
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each other (DR-02/DR-54, DR-74), with
    the bound read from that constant and rendered by
    scanner.close_gap_bound_text. A set category/tag filter is named too,
    since it can empty the list on its own.

    Args:
        sandbox (bool): True to phrase the message for a dev/sandbox run
            ("... found in sandbox ..."), False for a production run.
            Defaults to False.
        settings (LiveSettings | None): The run's toggles; None reads config.py's.
        time_series_searched (bool): Keyword-only. False says the run did not
            look for time-series pairs, instead of naming their rule.

    Returns:
        str: The fully formatted log message, ready to pass to logging.info().

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # The run's toggles, or config.py's for a caller that hands none
    settings = live_settings() if settings is None else settings
    if time_series_searched:
        time_series = (
            "time-series: both legs worded as cumulative deadlines "
            "(“by <date>”, two different ones) with the later leg's YES ask above "
            "the earlier's by the run's entry rule — "
            # In the finder's own rule-line words, so the two cannot disagree
            f"{describe_time_series_rule(settings.tier_floors, settings.spread_band)}"
        )
    else:
        time_series = ("time-series: not searched this run, because a held market "
                       "could not be identified (see the ERROR above)")
    thresholds = (
        f"{time_series}"
        " — or same-title: "
        f"≥{SAME_TITLE_MIN_PRICE_DIFF:.0%} price diff on two different series "
        # This module's own binding, like the same-title threshold above; the scanner
        # helper only renders it, in the words the refusal lines use.
        f"closing within {close_gap_bound_text(SAME_TITLE_MAX_CLOSE_GAP_SECONDS)}"
    )
    if settings.categories is not None or settings.tags is not None:
        # The run's category/tag filter, in the "Live settings:" line's words
        thresholds += (f" — among pairs filed under the run's category/tag filter "
                       f"({describe_trade_filter(settings)})")
    if sandbox:
        return f"No qualifying pairs found in sandbox ({thresholds})."
    return f"No qualifying pairs found ({thresholds})."


def _dedup_pairs(primary: list, secondary: list) -> list:
    """
    Merge two pair lists, excluding any pair from secondary that already appears in primary.

    A duplicate is defined as any pair whose frozenset of {ticker_a, ticker_b} already
    exists in primary. Since DR-67 the two finders cannot produce a cross-type
    collision on one ticker pair: a same-title pair requires identical wording,
    a time-series pair requires two DIFFERENT stated deadlines, and identical
    wording states identical deadline spans, or none — never two different
    ones. This function stays as a guard against a collision arising another
    way (both finders have their own defences too — see the one-series and
    cumulative-deadline gotchas in CLAUDE.md), and its preference is the same
    as before: were a collision to occur, the same-title pair would be kept
    because it is the near-arbitrage — identical questions must co-resolve, so
    its payoff is a floor — whereas the time-series pair is a directional bet
    whose sizing rests on the discounted mid-spread estimate of the in-between
    probability (config.time_series_profit_prob). This matches the
    same_title > time_series tie-break already used by strategy.select_portfolio().

    Args:
        primary (list): List of CandidatePair objects from find_same_title_pairs().
            These are always kept.
        secondary (list): List of CandidatePair objects from find_time_series_pairs().
            Entries whose ticker pair already appears in primary are dropped.

    Returns:
        list: Combined list with primary entries first, then non-duplicate secondary
            entries appended in their original order.
    """
    seen: set = set()
    result = []
    for pair in primary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        seen.add(key)
        result.append(pair)
    for pair in secondary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        # Only add if this exact ticker pair was not already found by the
        # same-title scanner (the primary list, which is kept on conflict)
        if key not in seen:
            seen.add(key)
            result.append(pair)
    return result


def _filter_by_category(pairs: list, settings: LiveSettings, listing_client) -> list:
    """
    Keep only the pairs filed under the run's categories and tags.

    Files each pair by MARKET A's series through historical.series_labels,
    the dashboard's filing rule. Matching is case-insensitive, the two axes
    combine by AND (None = any), and a tag matches under EVERY category, unlike
    the dashboard's category-scoped Tag options. Fails CLOSED: with no
    listing nothing is kept, rather than filing nearly every KX ticker as "Other",
    and its WARNING is the one line saying why, since the run exits EXIT_OK.
    A name matching no listed label and no pair draws a typo WARNING.

    Args:
        pairs (list): CandidatePairs after dedup, before enrichment.
        settings (LiveSettings): The run's toggles.
        listing_client: A production KalshiClient, or None to read the cached
            /series listing only (dev: the sandbox key never signs a prod request).

    Returns:
        list: The kept pairs, in their original order — the input list itself,
            with no request, when no filter is set; [] when there is no listing.
    """
    if settings.categories is None and settings.tags is None:
        return pairs
    # Kalshi's category and tags per series, the map the dashboard files by
    series_categories = load_series_categories(listing_client)
    # The filter in the "Live settings:" line's own words
    wanted = describe_trade_filter(settings)
    if not series_categories:
        where = ("no cached copy exists (a dev run reads the cached copy only — a "
                 "production run or a backtest fetches it)" if listing_client is None
                 else "Kalshi's /series listing could not be read and no cached copy exists")
        logging.warning(
            "Category/tag filter (%s) is set but %s — no pair can be filed, so this "
            "run trades none", wanted, where)
        return []
    cats = None if settings.categories is None else {c.casefold() for c in settings.categories}
    tags = None if settings.tags is None else {t.casefold() for t in settings.tags}
    kept: list = []
    dropped: Counter = Counter()
    filed_cats: set = set()
    filed_tags: set = set()
    for pair in pairs:
        event = pair.market_a.event_ticker
        # Filed exactly as the dashboard files a trade of this event
        category, tag = series_labels(event, infer_category(event), series_categories)
        filed_cats.add(category.casefold())
        filed_tags.add(tag.casefold())
        if ((cats is None or category.casefold() in cats)
                and (tags is None or tag.casefold() in tags)):
            kept.append(pair)
        else:
            dropped[f"{category} · {tag}"] += 1
    logging.info(
        "Category/tag filter (%s): kept %d of %d candidate pairs%s", wanted,
        len(kept), len(pairs),
        "" if not dropped else " — dropped " + ", ".join(
            f"{name} {n}" for name, n in dropped.most_common()))
    # A name matching no listed label and no pair is almost certainly a typo
    known_cats = filed_cats | {(c or "Uncategorised").casefold()
                               for c, _ in series_categories.values()}
    known_tags = filed_tags | {(t[0] if t else "General").casefold()
                               for _, t in series_categories.values()}
    for name in settings.categories or ():
        if name.casefold() not in known_cats:
            logging.warning(
                "Category %r names no category in Kalshi's listing of %d series, nor "
                "any this run's pairs were filed under — check the spelling; it "
                "matches nothing", name, len(series_categories))
    for name in settings.tags or ():
        if name.casefold() not in known_tags:
            hint = ""
            if " · " in name:
                # The dashboard's Tag select names "category · tag"
                category, _, tag = name.partition(" · ")
                hint = (f" (the dashboard's Tag option {name!r} is --category "
                        f"{category!r} --tag {tag!r})")
            logging.warning(
                "Tag %r is no series' first tag in Kalshi's listing of %d series, nor "
                "any this run's pairs were filed under — check the spelling; it "
                "matches nothing%s", name, len(series_categories), hint)
    return kept


def print_pairs_table(candidate_pairs: list, display_specs: dict) -> None:
    """
    Log a table of every candidate pair, with the planned trade for the
    selected ones.

    Each row shows the two markets, each leg's outcome label and exchange
    shard, the deadlines, the prices, whether the pair is tradeable, and for a
    selected pair the contract counts, profit if won, monthly return and
    Kelly fraction (the share of the portfolio value Kelly calls for). The
    counts of a trade that adds to a pair the account already holds are
    followed by "(adds to N held)", N being the contracts held on each market.

    Args:
        candidate_pairs (list): Every CandidatePair the scanner returned.
        display_specs (dict): id(pair) -> the selected TradeSpec, at the size it will trade at.

    Returns:
        None
    """
    rows = []
    for pair in candidate_pairs:
        spec = display_specs.get(id(pair))
        # Price columns come from the spec's own pair when there is one: that is
        # the marginal fill price for the recommended size, so the prices and
        # the trade on a row always describe the same thing. Candidates with no
        # spec keep the enrichment-stage quote.
        priced = spec.pair if spec else pair
        if spec:
            # Sides rendered next to each count, in market order (A then B)
            side_a, side_b = leg_sides(pair.pair_type)
            trade_str   = f"{spec.x}× {side_a.upper()}(A) + {spec.y}× {side_b.upper()}(B)"
            # Cross-module: a trade adding to a held pair says how much is held
            held = pair_held(spec.pair)
            if held is not None:
                trade_str += f" (adds to {count_text(held.count)} held)"
            profit_str  = f"${spec.min_payoff:.2f}"
            monthly_str = f"{spec.monthly_profit_ratio:.2%}/mo"
            kelly_str   = f"{spec.kelly_fraction:.1%} (p={spec.kelly_p:.2f})"
        else:
            trade_str   = "—"
            profit_str  = "—"
            monthly_str = "—"
            kelly_str   = "—"

        rows.append([
            pair.pair_type,
            # display_title prefixes the event title for MVE markets so multi-choice
            # option labels (e.g. "Trump", "Above $80k") carry their event context
            _truncate(display_title(pair.market_a)),
            _truncate(display_title(pair.market_b)),
            # DR-17: the outcome label in its own cells — display_title
            # appends it at the END, so _truncate's 40-character cut on the
            # title cells above drops it first on any title that runs past 40
            # characters, which the daily families this exists for all do
            _truncate(getattr(pair.market_a, "subtitle", "") or "—", 24),
            _truncate(getattr(pair.market_b, "subtitle", "") or "—", 24),
            # Which exchange shard each leg's market is on ("a/b"); each
            # order goes to its own leg's shard
            f"{pair.market_a.exchange_index}/{pair.market_b.exchange_index}",
            _format_deadline(pair.market_a.close_time),
            _format_deadline(pair.market_b.close_time),
            f"{priced.pA:.2%}",
            f"{priced.pB:.2%}",
            # The NO ask of market B — the traded NO-leg price of a time-series
            # pair (depth-weighted after enrichment); reporting-only for same-title
            f"{priced.nB:.2%}",
            "YES ✓" if pair.tradeable else "no",
            trade_str,
            profit_str,
            monthly_str,
            kelly_str,
        ])

    headers = [
        "Type",
        "Market A", "Market B", "Outcome A", "Outcome B",
        "Shards",
        "A Deadline", "B Deadline",
        "pA (YES)", "pB (YES)", "nB (NO)",
        "Tradeable?", "Recommended Trade", "Profit (win)", "Monthly Return", "Kelly",
    ]
    table = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    for line in table.splitlines():
        logging.info(line)


def _log_shard_coverage(shard_statuses, market_shards: set, balance_shards: set) -> None:
    """
    Run scanner.check_shard_coverage() and emit its findings at the right
    severity, shared by both _run_dev and _run_prod so the logging split
    (critical vs warning vs "full coverage" vs "unassessable") lives in one
    place.

    A run must only ever CLAIM full coverage when every shard the exchange
    advertises was actually scanned; a missing shard is reported loudly but
    never aborts the run — trading continues on whatever was covered. When NO
    advertised shard was scannable (every one explicitly trading-inactive —
    an exchange-wide halt), check_shard_coverage's two loops are empty and it
    returns ([], []), which used to print the same "Full shard coverage" line
    a healthy run prints (TS-01); that case now warns instead. A shard whose
    trading_active is None (flag absent or unreadable, TS-04) counts as
    scannable, matching scanner.inactive_shard_indexes; a re-typed "false"
    does NOT, because scanner.fetch_shard_statuses normalises it to a real
    False first (TS-04b).

    Args:
        shard_statuses (dict | None): Return value of scanner.fetch_shard_statuses().
        market_shards (set): exchange_index values seen among ingested markets.
        balance_shards (set): exchange_index values holding a nonzero balance.

    Returns:
        None
    """
    if shard_statuses is None:
        logging.info("Per-shard exchange status unavailable — coverage not assessable.")
        return
    # Pure comparison of advertised vs. observed shards; this function only
    # decides how loudly to log what check_shard_coverage found.
    critical, warnings = check_shard_coverage(shard_statuses, market_shards, balance_shards)
    # Coverage may only be CLAIMED over shards that were scannable.
    # check_shard_coverage deliberately skips trading-inactive shards (their
    # ingest drop already warned), so an all-inactive exchange yields ([], [])
    # — which read as success (TS-01). Only a False is inactive (a real bool
    # by now: scanner.fetch_shard_statuses recovers a re-typed "false" into
    # one); None (flag unknown) is scannable, matching inactive_shard_indexes.
    scannable = sorted(
        idx for idx, st in shard_statuses.items() if st.get("trading_active") is not False
    )
    for problem in critical:
        # Same severity channel as orphaned positions — this must never be missable.
        logging.critical("SHARD COVERAGE FAILURE: %s", problem)
    for problem in warnings:
        logging.warning("Shard coverage: %s", problem)
    if not scannable:
        logging.warning(
            "Shard coverage NOT claimable: all %d advertised shards are "
            "trading-inactive — markets were ingested from %d shard(s)",
            len(shard_statuses), len(market_shards),
        )
    elif not critical and not warnings:
        # Lists the shards actually scanned, not every advertised one — a
        # funded-but-halted shard is no longer folded into "full".
        logging.info("Full shard coverage: shards %s scanned", scannable)


def _blind_run_reason(markets: list, shard_statuses, inactive_shards: set) -> str | None:
    """
    Decide whether this run scanned NOTHING at all, and name the reason.

    The single definition of "blind run" for BOTH run modes, so dev and prod
    can never silently disagree about what one is. A blind run is any run whose
    market ingest yielded nothing to pair, and it must be reported with
    EXIT_NO_TRADEABLE_SHARDS rather than EXIT_OK: EXIT_OK claims "scanned
    everything, found no edge", which would let scheduler.run_job record the
    weekly slot as satisfied by a run that never looked at a single book
    (TS-01).

    Two independent causes, checked in that order because the first is the
    more specific diagnosis:

    1. Every advertised exchange shard is trading-inactive (the exchange-wide
       halt observed live 2026-09-03) — ingest dropped every market by design.
       Kept as its OWN disjunct rather than folded into the census below,
       because the two are not equivalent: a halt that still leaves one stray
       market ingested (e.g. one tagged with a shard /exchange/status does not
       advertise) would pass a census test while every advertised book is shut.
    2. Ingest produced zero markets for any other reason — most importantly
       when scanner.fetch_shard_statuses() returned None, which it does on ANY
       internal failure (it is fail-soft by design, and correctly so). That
       makes `shard_statuses` falsy, so cause 1 cannot fire, and an ingest that
       came back empty used to exit EXIT_OK claiming a clean scan (VI-02).

    This must be evaluated on the RAW ingest, before held-ticker or horizon
    filtering: those filters legitimately empty the list on a healthy exchange
    (`--max-horizon-days 1` on a quiet week), which is "no edge", not blind.

    Args:
        markets (list): The ApiMarket objects ingest produced, BEFORE any
            held-ticker or horizon filtering.
        shard_statuses (dict | None): Return value of
            scanner.fetch_shard_statuses(). None (breakdown unavailable) makes
            cause 1 unknowable, which is why cause 2 exists.
        inactive_shards (set): Return value of
            scanner.inactive_shard_indexes(shard_statuses) — only ever a subset
            of the advertised shards, so equality with the advertised set means
            "every one of them".

    Returns:
        str | None: A ready-to-log sentence naming which cause fired, or None
            when the run actually scanned something and is not blind.

    Raises:
        Nothing. This is a pure comparison over already-fetched values; it
        performs no I/O and swallows nothing.
    """
    if shard_statuses and inactive_shards == set(shard_statuses):
        return (
            f"Every advertised exchange shard is trading-inactive "
            f"({sorted(inactive_shards)}) — nothing can be scanned this run"
        )
    if not markets:
        return "Ingest produced zero markets — nothing can be scanned this run"
    return None


# argparse destination -> the flag an operator types, for the usage error
_LIVE_FLAGS = (
    ("tier_floors", "--tier-floors/--no-tier-floors"),
    ("spread_min", "--spread-min"),
    ("spread_max", "--spread-max"),
    ("interval_discount", "--interval-discount"),
    ("size_cap", "--size-cap"),
    ("same_title_size_cap", "--same-title-size-cap"),
    ("add_to_held_pairs", "--add-to-held-pairs/--no-add-to-held-pairs"),
    ("sell_at", "--sell-at"),
    ("no_sell", "--no-sell"),
    ("sell_min_days", "--sell-min-days"),
    ("no_sell_min_days", "--no-sell-min-days"),
    ("category", "--category"),
    ("any_category", "--any-category"),
    ("tag", "--tag"),
    ("any_tag", "--any-tag"),
)

# The cap flags, which take a whole percent where LiveSettings holds a fraction
_LIVE_PERCENT_FLAGS = frozenset({"size_cap", "same_title_size_cap"})


def _resolve_live_settings(args, parser) -> tuple[LiveSettings, LiveSettings]:
    """
    Resolve this run's LiveSettings: the saved live defaults, with each given flag laid over.

    The run's one read of the defaults (config.live_defaults, the one caller
    the AST pin test_ast_live_path_reads_toggles_only_through_live_settings
    allows); exits 2 when none are saved or the file is refused. There is no
    fallback to config.py's toggle constants. LiveSettings validates the
    result (dataclasses.replace re-runs __post_init__), so a flag meets the
    same rule a saved value does; the result keeps the defaults' origin.
    Called before logging is configured, so a refusal logs nothing and makes
    no request.

    Args:
        args (argparse.Namespace): The parsed flags; a missing attribute reads
            as not given.
        parser (argparse.ArgumentParser): Used to report a refusal.

    Returns:
        tuple[LiveSettings, LiveSettings]: (this run's settings, the saved
            defaults they were built from), equal when no toggle flag was given.

    Raises:
        SystemExit: Status 2 (parser.error) when no live defaults are saved,
            the saved file is refused, or a flag's value is invalid.
    """
    try:
        # The saved live defaults: the reference, and the base the flags lay over.
        # There is no fallback: with none saved, the run does not start
        reference = live_defaults()
    except LiveDefaultsMissing as exc:
        parser.error(str(exc))
    except LiveDefaultsError as exc:
        # The defaults server will not save over a file it refuses, so the
        # remedy is to fix the file, or to delete it before saving new ones
        parser.error(f"the saved live defaults are refused ({exc}): fix the file, or delete "
                     f"it and then save new ones through the defaults server, "
                     f"./start_dashboard.sh (or python3 -m kalshi_betting.defaults_server): "
                     f"its --seed, or the backtest dashboard's \"Save as live defaults…\" "
                     f"button — the server will not save over a file it refuses")
    overrides: dict = {}
    if getattr(args, "tier_floors", None) is not None:
        overrides["tier_floors"] = args.tier_floors
    lo, hi = getattr(args, "spread_min", None), getattr(args, "spread_max", None)
    if lo is not None or hi is not None:
        overrides["spread_band"] = (reference.spread_band[0] if lo is None else lo,
                                    reference.spread_band[1] if hi is None else hi)
    if getattr(args, "interval_discount", None) is not None:
        overrides["interval_discount"] = args.interval_discount
    if getattr(args, "size_cap", None) is not None:
        overrides["size_cap"] = args.size_cap / 100
    if getattr(args, "same_title_size_cap", None) is not None:
        overrides["same_title_size_cap"] = args.same_title_size_cap / 100
    if getattr(args, "add_to_held_pairs", None) is not None:
        overrides["add_to_held_pairs"] = args.add_to_held_pairs
    # --no-sell clears both sell settings (a minimum of days means nothing
    # without a level); --sell-at and --sell-min-days then set one each, and
    # LiveSettings refuses a minimum left with no level (argparse keeps
    # --sell-at/--no-sell, and the two minimum-of-days flags, mutually exclusive)
    if getattr(args, "no_sell", None):
        overrides["sell_at"] = None
        overrides["sell_min_days"] = None
    if getattr(args, "sell_at", None) is not None:
        overrides["sell_at"] = args.sell_at / 100
    if getattr(args, "no_sell_min_days", None):
        overrides["sell_min_days"] = None
    if getattr(args, "sell_min_days", None) is not None:
        overrides["sell_min_days"] = args.sell_min_days
    # --category / --tag (repeatable) set the filter; --any-category / --any-tag
    # clear the saved one (argparse keeps each pair mutually exclusive)
    if getattr(args, "category", None) is not None:
        overrides["categories"] = tuple(args.category)
    elif getattr(args, "any_category", None):
        overrides["categories"] = None
    if getattr(args, "tag", None) is not None:
        overrides["tags"] = tuple(args.tag)
    elif getattr(args, "any_tag", None):
        overrides["tags"] = None
    try:
        # replace re-validates every field (LiveSettings.__post_init__)
        return dc_replace(reference, **overrides), reference
    except ValueError as exc:
        # Given means not None, never truthy: 0 is a given --spread-min and
        # False a given --no-tier-floors (the --any-* switches default to None)
        given = [(dest, flag) for dest, flag in _LIVE_FLAGS
                 if getattr(args, dest, None) is not None]
        message = (f"invalid live setting for this run "
                   f"({', '.join(flag for _, flag in given)}): {exc}")
        percent = [flag for dest, flag in given if dest in _LIVE_PERCENT_FLAGS]
        if percent:
            # The flag's own unit, from the one grid definition (SIZE_CAP_STEP)
            step = f"{SIZE_CAP_STEP * 100:g}"
            message += (f" ({' and '.join(percent)} take{'s' if len(percent) == 1 else ''} "
                        f"a whole percent, a multiple of {step} from {step} to 100, read "
                        "as that percent / 100)")
        if any(dest == "sell_at" for dest, _ in given):
            # --sell-at's unit, from its grid definition (SELL_AT_STEP)
            sell_step = f"{SELL_AT_STEP * 100:g}"
            message += (f" (--sell-at takes a whole percent, a multiple of {sell_step} "
                        f"from {sell_step} to 100, read as that percent / 100)")
        parser.error(message)


def _log_live_settings(settings: LiveSettings, reference: LiveSettings, *,
                       real_money: bool) -> None:
    """
    Log the run's defaults' origin, its toggles, any departure, and every rule warning.

    One INFO line names the defaults' origin (the saved file, when and from
    what it was saved). One INFO line marks each field that departs from
    reference: "(default: X)" when reference is the saved live defaults, as a
    live run's always is, "(config: X)" when it was built from config.py's
    constants (a reference a test or direct call builds, e.g. with
    config.live_settings()). Only a real-money run WARNs on a departure.
    Resolves neither object itself (TestLiveSettingsReachEverySite).

    Args:
        settings (LiveSettings): The run's toggles.
        reference (LiveSettings): The defaults the run's toggles were built
            from (the run's own when a run mode was handed none, which marks
            nothing).
        real_money (bool): True for a prod run that submits orders.
    """
    # Where the defaults came from: the saved file, when and from what
    logging.info("Live defaults: %s", reference.origin)
    # Every field, with a mark on each one a flag moved: "(default: X)" (the
    # saved defaults), or "(config: X)" for a reference built from config.py
    logging.info("Live settings: %s", describe_live_settings(settings, reference))
    if real_money and settings != reference:
        if reference.origin == LIVE_DEFAULTS_FROM_CONFIG:
            logging.warning(
                "This PRODUCTION run overrides config.py's live settings (see the "
                "\"(config: …)\" marks on the line above): its trades follow the "
                "flags, not the committed configuration")
        else:
            logging.warning(
                "This PRODUCTION run overrides the saved live defaults (see the "
                "\"(default: …)\" marks on the line above): its trades follow the "
                "flags, not the saved defaults")
    # A setting that empties part of the strategy or lifts one pair's stake
    for text in live_rule_warnings(settings):
        logging.warning("Live settings: %s", text)


def _run_dev(client, args, settings: LiveSettings | None = None,
             reference: LiveSettings | None = None) -> int:
    """
    Run a dev-mode scan of the sandbox exchange and write a simulation.

    Uses real sandbox market data but never sends orders. It holds no
    positions, so the virtual --sandbox-balance counts as both the portfolio
    value and the cash. Results go to a timestamped Excel file.

    Args:
        client: KalshiClient for the sandbox, from auth.build_client("dev").
        args: Parsed arguments (sandbox_balance, max_horizon_days).
        settings (LiveSettings | None): The run's settings; None means config.py's, for tests.
        reference (LiveSettings | None): The saved defaults they came from; None means settings.

    Returns:
        int: EXIT_NO_TRADEABLE_SHARDS when nothing could be scanned (every
            shard had stopped trading, or no markets came back), else EXIT_OK.

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # Resolved ONCE and handed to every site below that reads a toggle
    settings = live_settings() if settings is None else settings
    reference = settings if reference is None else reference
    sandbox_balance_cents = int(args.sandbox_balance * 100)
    # Dev holds nothing, so the virtual balance is both the portfolio value and the cash
    portfolio_value_cents = cash_cents = sandbox_balance_cents
    logging.info(
        "DEV mode: using real sandbox market data | virtual balance $%.2f",
        args.sandbox_balance,
    )
    # Log the run's toggles; dev never submits, so no departure WARNING
    _log_live_settings(settings, reference, real_money=False)

    # Read the exchange's per-shard status breakdown so ingest can drop shards
    # that aren't trading. Returns None on the sandbox / pre-sharding shape,
    # which degrades to single-shard semantics (keep everything).
    shard_statuses = fetch_shard_statuses(client)
    inactive_shards = inactive_shard_indexes(shard_statuses)

    # Fetch all open sandbox markets — the sandbox public endpoint does not require
    # valid authentication for read operations, so this works with the prod key too.
    # Markets from every shard are ingested and tagged; only trading-inactive
    # shards are dropped.
    markets = fetch_open_events_with_markets(client, inactive_shards=inactive_shards)
    logging.info("Sandbox markets fetched: %d", len(markets))

    # Dev mode has one virtual balance, not a real per-shard breakdown, so
    # there is no balance-shard set to compare against — pass empty and let
    # the market-coverage half of the check still catch a missing shard
    _log_shard_coverage(shard_statuses, {m.exchange_index for m in markets}, set())

    # A run that ingested nothing simulated nothing. Dev's exit code is not
    # consumed by the scheduler, but the two modes must not disagree about what
    # a blind run is, so both ask the same helper (TS-01, VI-02).
    blind_reason = _blind_run_reason(markets, shard_statuses, inactive_shards)
    if blind_reason:
        logging.warning("%s", blind_reason)
        return EXIT_NO_TRADEABLE_SHARDS

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Skip held-positions filter — sandbox requires a separate account and credentials.
    # Pass an empty set so _filter_active_markets does not exclude any tickers.
    # The run's settings decide the time-series entry rule (tier floors, band)
    time_series_pairs = find_time_series_pairs(
        client, held_tickers=set(), markets=markets, settings=settings,
    )
    # Detect same-title pairs separately — uses a different grouping key (exact title match)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers=set())
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Category/tag filter; no listing client, so the sandbox key never signs a prod request
    candidate_pairs   = _filter_by_category(candidate_pairs, settings, None)
    # Price each pair from its order book, up to what one trade's budget could buy
    candidate_pairs   = enrich_with_orderbook_prices(
        client, candidate_pairs, portfolio_value_cents, settings=settings,
        cash_cents=cash_cents,
    )

    if not candidate_pairs:
        # BS-26: write_dev_simulation() already logs "Dev simulation written: %s" —
        # this line carries the qualifier (why the file is empty) instead of
        # repeating the artifact path a second time. The thresholds come from
        # _no_pairs_msg, handed the run's settings, so they match the rule the run applied.
        logging.info(
            "%s Simulation file will contain headers only.",
            _no_pairs_msg(sandbox=True, settings=settings),
        )
        # Write an empty simulation file so the run is still recorded
        write_dev_simulation([], [], sandbox_balance_cents)
        return EXIT_OK

    # Apply Kelly sizing to each candidate pair using the virtual balance
    trade_specs   = _compute_trade_specs(candidate_pairs, portfolio_value_cents, settings,
                                         cash_cents=cash_cents)
    # Pick trades best-first within the cash, shrinking one that no longer fits
    portfolio     = select_portfolio(list(trade_specs.values()), cash_cents)
    # The selected trade for each candidate pair, for the pairs table
    display_specs = _display_specs(trade_specs, portfolio)

    logging.info("Kalshi Sandbox Scan — Virtual Balance: $%.2f | Mode: DEV", args.sandbox_balance)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        # BS-26: qualifier only — write_dev_simulation() logs the "written" line itself.
        logging.info(
            "No executable trades found — simulation file will "
            "contain candidates only."
        )
        # Write a simulation file showing candidates even though no trades were sized
        write_dev_simulation([], candidate_pairs, sandbox_balance_cents)
        return EXIT_OK

    _print_portfolio(portfolio, "Simulated")

    # Simulate order execution (dry_run=True is always enforced in dev mode)
    # Returns TradeResult objects with status="simulated" — no API orders are placed
    results = execute_trades(client, portfolio, dry_run=True)

    # Write the simulation Excel file: Sheet 1 = simulated trades, Sheet 2 = all
    # candidates. write_dev_simulation() logs "Dev simulation written: %s" itself
    # (BS-26) — don't duplicate that line here.
    write_dev_simulation(results, candidate_pairs, sandbox_balance_cents)
    return EXIT_OK


def _run_prod(client, args, settings: LiveSettings | None = None,
              reference: LiveSettings | None = None, *,
              report: RunReport | None = None) -> int:
    """
    Run one production scan and trade on the real Kalshi account.

    Reads the balance, sells the held positions the take-profit rule picks
    (when settings.sell_at is set), skips markets already held, finds
    same-title and time-series pairs, sizes each trade with the Kelly rule (a
    formula for how much to bet), sends the orders (the NO leg first, then
    the YES leg) and adds the results to trade_log.xlsx. With --dry-run
    everything runs except sending orders.

    Sales come before every purchase. seller.plan_sales picks the held
    positions (an exact held pair, or a lone held market whose partner has
    paid out) that have stayed at or above settings.sell_at of their
    potential profit at every daily check, trader.sell_positions sells them,
    and the sales go into the trade log before anything is bought. A live
    run that sold (or whose sale counts came from the account) then reads
    its positions again (_positions_after_sales), then its cash and Kalshi's
    value of the open positions, which _value_after_sales checks before the
    buys are sized on what is left; a dry run adds the estimated proceeds to the cash instead
    (_with_simulated_proceeds). The markets sold, and both markets of a pair
    a sale left uneven, stay out of every purchase this run, and no position
    picked for sale is added to. A position sold in full frees the rest of
    its ladder (one question at several deadlines) for new time-series
    pairs. Nothing is sold when a held market could not be identified or the
    positions listing was cut short (one WARNING).

    Each trade bets a share of the portfolio value (the cash plus Kalshi's
    value of the open positions) but never spends more than the cash left.
    The minimum-balance check reads the portfolio value; low cash alone only
    logs a WARNING. No new time-series trade is made on a ladder the account
    holds, and none at all if a held market cannot be identified.

    Kalshi's value of the open positions counts only if the contracts held
    can back it (_checked_positions_value, once the positions are read); a
    value refused at the start of the run leaves the cash alone as the
    portfolio value, and the minimum-balance check is applied to it again.

    With settings.add_to_held_pairs on, it may add to an exact held pair
    (scanner.held_pairs: the same two markets, the same side on each) or to a
    lone held leg whose partner has paid out (a new pair buying that market
    on its held side, beside a market not held), sized
    on the whole position (config.held_pair_fraction); only when the
    positions listing was read to its end, every held market was identified
    and Kalshi's value of the open positions was read and kept, and never to
    a pair whose stake (its worth at today's prices plus the fees paid for
    it) already fills its per-trade cap of the portfolio value.

    Args:
        client: KalshiClient for production, from auth.build_client("prod").
        args: Parsed arguments (dry_run, max_horizon_days).
        settings (LiveSettings | None): The run's settings; None means config.py's, for tests.
        reference (LiveSettings | None): The saved defaults they came from; None means settings.
        report (RunReport | None): Keyword-only. The --result-file summary to fill in.

    Returns:
        int: EXIT_SKIPPED_LOW_BALANCE if the portfolio value (the cash alone
            when Kalshi's value of the open positions is refused) is below
            MIN_BALANCE_CENTS; EXIT_NO_TRADEABLE_SHARDS if nothing could be
            scanned; EXIT_TRADES_NEED_ATTENTION if a trade or a sale needs a
            person to check it; otherwise EXIT_TIME_SERIES_SKIPPED if a held
            market could not be identified; else EXIT_OK.

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # What the run did, for main.py --result-file; without it nothing reads this one
    report = RunReport(dry_run=args.dry_run, started_at=datetime.now(UTC)) if report is None else report
    # Resolved ONCE, before any request, and handed to every site below that reads a toggle
    settings = live_settings() if settings is None else settings
    reference = settings if reference is None else reference
    logging.warning("Running in PRODUCTION mode — real money will be used!")
    # Log the run's toggles; a departure is a WARNING when orders go out
    _log_live_settings(settings, reference, real_money=not args.dry_run)

    # Read the balance (this also checks the credentials): cash per shard and the positions' value
    account = read_account_balance(client)
    # Cash per shard, for the shard coverage check and the cash transfers below
    shard_balances = account.shard_cash_cents
    # Cash to spend: all shards together, as cash is moved between shards before trading
    cash_cents = sum(shard_balances.values())
    # The portfolio value each trade bets a share of (just the cash if positions are unread)
    positions_value_cents = account.positions_value_cents
    portfolio_value_cents = _bankroll_cents(cash_cents, positions_value_cents)
    # Log all three numbers, marking an unread positions value as not read
    logging.info(
        "Sizing on portfolio value $%.2f = cash $%.2f + open positions %s",
        portfolio_value_cents / 100, cash_cents / 100,
        ("not read (counted as $0.00)" if positions_value_cents is None
         else f"${positions_value_cents / 100:.2f}"),
    )
    report.balance_before = cash_cents / 100
    report.portfolio_value_before = portfolio_value_cents / 100
    if portfolio_value_cents < MIN_BALANCE_CENTS:
        # Don't waste API calls scanning when there's insufficient capital to trade
        message = (f"Portfolio value ${portfolio_value_cents / 100:.2f} (cash "
                   f"${cash_cents / 100:.2f}) is below minimum "
                   f"${MIN_BALANCE_CENTS / 100:.2f} — skipping run.")
        logging.warning("%s", message)
        report.message = message
        return EXIT_SKIPPED_LOW_BALANCE

    # Current open positions, each with its side and cost. A held market is
    # never traded again, except an exact held pair or a lone held leg (its
    # partner paid out) this run's settings add to; held_listing["complete"]
    # says whether the listing was read to its end
    held_listing: dict = {}
    held_positions    = get_held_positions(client, complete_out=held_listing)
    held_tickers      = set(held_positions)
    # Kalshi's value of the open positions counts only if the contracts held
    # can back it; None when it was not read, or is refused (with a WARNING)
    checked_value_cents = _checked_positions_value(
        cash_cents, positions_value_cents, held_positions,
        held_listing.get("complete") is True,
    )
    if positions_value_cents is not None and checked_value_cents is None:
        # Refused: size on the cash alone, and apply the minimum to it again
        portfolio_value_cents = cash_cents
        report.portfolio_value_before = cash_cents / 100
        if portfolio_value_cents < MIN_BALANCE_CENTS:
            message = (f"Portfolio value ${portfolio_value_cents / 100:.2f} (cash "
                       f"${cash_cents / 100:.2f}) is below minimum "
                       f"${MIN_BALANCE_CENTS / 100:.2f} — skipping run.")
            logging.warning("%s", message)
            report.message = message
            return EXIT_SKIPPED_LOW_BALANCE
    if cash_cents < MIN_BALANCE_CENTS:
        # Low cash alone does not stop the run; no trade spends more than the cash left
        logging.warning(
            "Cash $%.2f is below the $%.2f minimum but the portfolio value is not: "
            "the run goes on, and no trade spends more than the cash left",
            cash_cents / 100, MIN_BALANCE_CENTS / 100,
        )

    # Read the exchange's per-shard status breakdown so ingest can drop shards
    # that aren't trading. Returns None on the pre-sharding shape, which
    # degrades to single-shard semantics (keep everything). The full dict is
    # kept — ensure_shard_collateral below also reads each shard's
    # intra_exchange_transfers_active off it.
    shard_statuses    = fetch_shard_statuses(client)
    inactive_shards   = inactive_shard_indexes(shard_statuses)

    # Fetch all open markets (every shard, tagged — only trading-inactive
    # shards are dropped) and pre-filter held tickers for downstream efficiency
    markets           = fetch_open_events_with_markets(client, inactive_shards=inactive_shards)

    # Verify the exchange's advertised shards were actually covered by this
    # fetch BEFORE any held-ticker/horizon filtering trims the market list —
    # a blind spot must be measured against what ingest actually saw, not a
    # subsequently-filtered view of it. Only shards holding money count as
    # funded, per check_shard_coverage's contract.
    _log_shard_coverage(
        shard_statuses,
        {m.exchange_index for m in markets},
        {s for s, c in shard_balances.items() if c > 0},
    )

    # An ingest that produced nothing scanned nothing — an exchange-wide halt
    # dropping every market, or an empty ingest whose cause the status
    # breakdown could not name. Either way that is not "no edge this week":
    # return the dedicated code so scheduler.run_job never records the Monday
    # slot as satisfied (TS-01, VI-02). Evaluated here, before the held-ticker
    # and horizon filters below, so a filter that legitimately empties the list
    # is never mistaken for a blind run.
    blind_reason = _blind_run_reason(markets, shard_statuses, inactive_shards)
    if blind_reason:
        logging.warning("%s", blind_reason)
        report.message = blind_reason
        return EXIT_NO_TRADEABLE_SHARDS

    # Our positions' ladders, read before held markets are dropped; None
    # means one could not be identified, so no time-series trade this run.
    # Each identified market's own labels land in held_labels, for held_pairs
    held_labels: dict = {}
    held_ladders      = resolve_held_ladders(client, markets, held_tickers,
                                             labels_out=held_labels)
    # Every market by ticker, before held ones are dropped: held_pairs reads
    # each held market's ask from it, to value a held pair at today's prices
    markets_by_ticker = {m.ticker: m for m in markets}
    # The note on this run's trade-log banners: the toggles a flag moved and
    # the saved defaults the run started from, so the workbook tells a flag's
    # rows from the defaults' rows, and one set of saved defaults from the next
    run_note = (f"settings: {describe_live_settings(settings, reference)}"
                + ("" if reference.origin == LIVE_DEFAULTS_FROM_CONFIG
                   else f" | defaults: {reference.origin}"))

    # Sales come before every purchase, as in the backtest: the cash they free
    # funds this run's buys
    sales: list = []
    sold_tickers: set = set()
    # A dry run's estimated sale proceeds, added to the cash its buys are
    # sized on but never received (0 in a live run)
    simulated_cents = 0
    if settings.sell_at is not None:
        if held_ladders is None or held_listing.get("complete") is not True:
            # Fails closed, as adding to held pairs does: a held market that
            # cannot be placed could share a ladder with any position
            logging.warning("Not selling this run: %s",
                            "a market the account holds could not be looked up"
                            if held_ladders is None
                            else "the list of the account's positions was cut short")
        else:
            cash_before_sales = cash_cents
            # Cross-module: the positions the take-profit rule sells this run,
            # each with what its sale orders need (read-only requests only)
            plans = plan_sales(client, held_positions, held_labels, markets_by_ticker,
                               settings=settings, now=datetime.now(UTC))
            if plans and not args.dry_run:
                # Recorded before the first sale order, as before the first buy
                report.submission_started = True
            # Cross-module: one position at a time, each order reduce-only and
            # immediate-or-cancel, sent once; a dry run sends nothing
            sales = sell_positions(client, plans, dry_run=args.dry_run)
            # The sales for the run result; never raises
            report.sales = report_sales(sales)
            sold_tickers = _sold_tickers(sales, dry_run=args.dry_run)
            # A live run reads its account back when anything sold or was left
            # for a person, or when an account reading, which may trail the
            # fills, decided a sale's count (even none)
            read_back = not args.dry_run and (
                bool(sold_tickers)
                or any(getattr(sale, "decided_by_account", False) is True for sale in sales))
            if args.dry_run:
                # Sized as if the sales filled: their estimated proceeds go on
                # the cash and its shards
                cash_cents, shard_balances = _with_simulated_proceeds(
                    sales, cash_cents, shard_balances)
                simulated_cents = cash_cents - cash_before_sales
                # The portfolio value is never below the cash, so one read as
                # the cash alone grows with the proceeds
                portfolio_value_cents = max(portfolio_value_cents, cash_cents)
                if sold_tickers:
                    logging.info("Dry run: sizing as if the sales filled — cash $%.2f "
                                 "(+$%.2f)", cash_cents / 100, simulated_cents / 100)
            elif read_back:
                # The start-of-run listing, to check the sales against
                positions_before_sales = held_positions
                # The positions first (waiting while they may not show the
                # sales yet), then the balance, so it no longer counts what
                # was sold
                try:
                    held_positions, not_shown, more_sold = _positions_after_sales(
                        client, sales, positions_before_sales, held_listing)
                    account = read_account_balance(client)
                except Exception:
                    # The sales go on record before the error stops the run
                    _record_sales(sales, cash_before_sales, cash_before_sales, run_note)
                    raise
                if more_sold:
                    # A sale counted short from a trailing ledger: a person
                    # must check it, and its markets stay out of every purchase
                    sales = _flag_sales_sold_more(sales, more_sold, positions_before_sales,
                                                  held_positions)
                    report.sales = report_sales(sales)
                    sold_tickers = _sold_tickers(sales, dry_run=False)
                shard_balances = account.shard_cash_cents
                cash_cents = sum(shard_balances.values())
                positions_value_cents = account.positions_value_cents
            if sales:
                report.cash_after_sales = cash_cents / 100
                # On record before any buy; a dry run sent nothing, so its
                # banner shows the cash before the sales
                _record_sales(sales, cash_before_sales,
                              cash_before_sales if args.dry_run else cash_cents, run_note)
            if read_back or (args.dry_run and sold_tickers):
                if not args.dry_run:
                    # The portfolio value the buys are sized on; when contracts
                    # were, or may have been, sold, the positions' value must
                    # have fallen below the one kept at the start
                    checked_value_cents, portfolio_value_cents = _value_after_sales(
                        cash_cents, positions_value_cents, held_positions,
                        held_listing.get("complete") is True, not_shown,
                        value_before_sales=checked_value_cents if sold_tickers else None)
                    # A listing cut short may leave out a market still held,
                    # so then every market held before the sales counts as
                    # held still (fails closed: its ladder stays blocked)
                    held_tickers = set(held_positions) | (
                        set() if held_listing.get("complete") is True else held_tickers)
                # A dry run's held markets still include what it would have
                # sold, so those are taken out; a live run holds what the new
                # listing shows (a part-sold market too)
                still_held = (held_tickers - sold_tickers) if args.dry_run else held_tickers
                unknown = sorted((t for t in still_held if t not in held_labels), key=str)
                if unknown:
                    # Only a position that appeared from elsewhere since the
                    # run started: its ladder is unknown, so no time-series trade
                    logging.error(
                        "Held market %r was not held when this run started, so its ladder "
                        "is unknown and no time-series trade will be made this run",
                        unknown[0])
                    held_ladders = None
                else:
                    # A sold position's ladders are free again; its own
                    # markets stay blocked through sold_tickers
                    held_ladders = frozenset().union(*(held_labels[t] for t in still_held))
    # The exit code of every clean return below: a sale left for a person
    # first, then a held market that could not be identified
    clean_exit        = (EXIT_TRADES_NEED_ATTENTION
                         if any(s.status in _SALE_ATTENTION_STATUSES for s in sales)
                         else EXIT_OK if held_ladders is not None
                         else EXIT_TIME_SERIES_SKIPPED)
    # The words the run's closing line ends with when a sale needs a person,
    # since that line otherwise speaks of the trades alone
    sale_note         = _sale_attention_note(sales)
    # What this run may add to (exact held pairs, and lone legs whose partner
    # has paid out): only with the setting on, and only when every held
    # market was listed and identified, since an unknown one could share a
    # pair's ladder, and being alone on its ladder is what makes adding safe; and only when Kalshi's value of the open
    # positions was read and kept, since an add-on is sized on the portfolio
    # value, which would otherwise leave out what the account holds
    add_on_pairs: dict = {}
    if settings.add_to_held_pairs:
        if held_ladders is None or held_listing.get("complete") is not True:
            logging.warning("Not adding to held pairs this run: %s",
                            "a market the account holds could not be looked up"
                            if held_ladders is None
                            else "the list of the account's positions was cut short, "
                                 "so a held market may be missing from it")
        elif held_positions and checked_value_cents is None:
            logging.warning("Not adding to held pairs this run: %s",
                            "Kalshi's value of the open positions could not be read, "
                            "so the portfolio value counts only the cash"
                            if positions_value_cents is None
                            else "Kalshi's value of the open positions was refused, "
                                 "so the portfolio value counts only the cash")
        else:
            # Cross-module: the one definition of what a run may add to (an
            # exact held pair, both markets alone on one ladder, one YES and
            # one NO of equal size; or a lone leg alone on its ladder, its
            # partner paid out), valued at today's prices from this run's
            # market list
            add_on_pairs = held_pairs(held_positions, held_labels, markets_by_ticker)
            # A pair whose stake (its worth at today's prices plus the fees
            # paid for it) already fills its per-trade cap of the portfolio
            # value can add nothing whatever Kelly says: it would only take
            # its group's one slot and size to nothing, so its markets stay
            # blocked like any other held market's
            full = {key for key, pair in add_on_pairs.items()
                    if held_pair_fraction(settings.size_cap, pair.stake_dollars,
                                          portfolio_value_cents / 100) <= 0}
            if full:
                logging.info("Held pairs already at their size cap, not added to this "
                             "run: %d", len(full))
                add_on_pairs = {key: pair for key, pair in add_on_pairs.items()
                                if key not in full}
            # A position picked for sale is not added to this run, whatever
            # its sale's outcome (_planned_tickers)
            if sales:
                planned = _planned_tickers(sales) | sold_tickers
                picked = {key for key in add_on_pairs if key & planned}
                if picked:
                    logging.info("Held pairs picked for sale this run, not added to: %d",
                                 len(picked))
                    add_on_pairs = {key: pair for key, pair in add_on_pairs.items()
                                    if key not in picked}
    # The markets of those pairs and lone legs stay in (a pair's market pairs
    # only with its own partner, a lone leg only with a market not held);
    # every other held market is dropped, and so is every market sold this
    # run, which a live run may no longer hold
    add_on_tickers    = {ticker for key in add_on_pairs for ticker in key}
    blocked_tickers   = (held_tickers - add_on_tickers) | sold_tickers
    markets           = [m for m in markets if m.ticker not in blocked_tickers]

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets           = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Run both pair detection paths: time-series (the run's entry rule, and no
    # pair on a ladder we hold but one adding to an exact held pair or a lone
    # held leg) and same-title (a held pair's markets pair only with each
    # other, a lone leg only with a market not held)
    if held_ladders is None:
        time_series_pairs = []
    else:
        time_series_pairs = find_time_series_pairs(
            client, blocked_tickers, markets, settings=settings, held_ladders=held_ladders,
            add_on_pairs=add_on_pairs,
        )
    same_title_pairs  = find_same_title_pairs(markets, blocked_tickers,
                                              add_on_pairs=add_on_pairs)
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Category/tag filter, before enrichment so a dropped pair costs no book request
    candidate_pairs   = _filter_by_category(candidate_pairs, settings, client)
    # Price each pair from its order book, up to what one trade's budget could buy
    candidate_pairs   = enrich_with_orderbook_prices(
        client, candidate_pairs, portfolio_value_cents, settings=settings,
        cash_cents=cash_cents,
    )

    if not candidate_pairs:
        # Names the run's entry rule, or says time-series was not searched
        message = (_no_pairs_msg(settings=settings,
                                 time_series_searched=held_ladders is not None)
                   + sale_note)
        logging.info("%s", message)
        report.message = message
        return clean_exit

    # Size each pair: a share of the portfolio value, spending at most the cash
    trade_specs   = _compute_trade_specs(candidate_pairs, portfolio_value_cents, settings,
                                         cash_cents=cash_cents)
    # Pick trades best-first within the cash, one time-series trade per ladder
    portfolio     = select_portfolio(list(trade_specs.values()), cash_cents,
                                     held_ladders=held_ladders or frozenset())
    # The selected trade for each candidate pair, for the pairs table
    display_specs = _display_specs(trade_specs, portfolio)

    logging.info("Kalshi Pair Scan — Portfolio value: $%.2f (cash $%.2f) | Mode: PROD",
                 portfolio_value_cents / 100, cash_cents / 100)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        message = "No executable trades found." + sale_note
        logging.info("%s", message)
        report.message = message
        return clean_exit

    _print_portfolio(portfolio, "Selected")

    # Re-fetch each pair's books and drop any that moved, under the run's rule
    portfolio = pre_execution_check(client, portfolio, settings=settings)
    if not portfolio:
        message = ("All selected pairs failed pre-execution price check — no trades "
                   "submitted." + sale_note)
        logging.info("%s", message)
        report.message = message
        return clean_exit

    # Move cash onto the shards the trades use; a trade whose shard cannot be funded is dropped
    portfolio = ensure_shard_collateral(
        client, portfolio, shard_balances, shard_statuses, dry_run=args.dry_run,
    )
    if not portfolio:
        message = (
            "No selected pair could be funded on its exchange shard — no trades submitted."
            + sale_note
        )
        logging.info("%s", message)
        report.message = message
        # Never a bare return: sys.exit(None) exits 0, which would hide
        # EXIT_TIME_SERIES_SKIPPED
        return clean_exit

    # Recorded before the first order is sent, so a result written after an
    # exception or Ctrl-C during execute_trades (whose trades are not yet
    # recorded) still says orders may have been placed
    if not args.dry_run:
        report.submission_started = True

    # Submit orders sequentially per leg, concurrently across pairs (on the V2
    # path, one pair at a time until a NO fill has confirmed the order-side
    # mapping, for at most config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS; a
    # disproof stops every pair that starts after it)
    results = execute_trades(client, portfolio, dry_run=args.dry_run)
    # Each pair's outcome for the run result, before the trade log is written;
    # it never raises, so the trade log and its rescue dump are always reached
    report.trades = report_trades(results)

    # The cash before trading as Kalshi holds it: a dry run's estimated sale
    # proceeds left out, since its sales sent nothing
    cash_read_cents = cash_cents - simulated_cents
    # Read the cash after trading for the trade log; if that fails, use the cash before
    try:
        balance_after = sum(read_account_balance(client).shard_cash_cents.values()) / 100
        # Only a balance actually read goes into the run result
        report.balance_after = balance_after
    except Exception as exc:
        logging.error(
            "Post-trade balance fetch failed: %s — logging with pre-trade balance", exc,
        )
        balance_after = cash_read_cents / 100

    # Append this run's results to the cumulative trade_log.xlsx file. If the
    # write fails (e.g. the file is open in Excel), dump every result to the log
    # so the record of real fills is never lost, then re-raise.
    try:
        # append_to_prod_log() already logs "Trade log updated: %s (%d new row(s))"
        # itself (BS-26) — don't duplicate that line here. The separator row
        # shows the cash before and after trading, with run_note as its note
        append_to_prod_log(results, cash_read_cents / 100, balance_after, run_note=run_note)
    except Exception as exc:
        logging.critical("Failed to write trade log: %s — rescue dump follows", exc)
        for r in results:
            # Cross-module: an add-on's counts sit on top of what the pair
            # already held, so its line names that held count too
            held = pair_held(r.spec.pair)
            # Both counts are printed: x is market A's leg and y is market B's,
            # and for a time-series pair the NO leg (the one that gets unwound)
            # is market B's, so y is the count a human must reconcile first.
            logging.critical(
                "  RESCUE | %s | %s | A=%s B=%s | x=%d y=%d%s cost=$%.2f incl. fees | %s",
                r.status,
                r.spec.pair.canonical_title,
                r.spec.pair.market_a.ticker,
                r.spec.pair.market_b.ticker,
                r.spec.x,
                r.spec.y,
                f" (adds to {count_text(held.count)} held)" if held is not None else "",
                r.spec.total_cost_with_fees,
                r.error or "",
            )
        raise

    if args.dry_run:
        message = "[DRY RUN] No orders were actually submitted." + sale_note
        logging.info("%s", message)
        report.message = message
        return clean_exit

    n_ok       = sum(1 for r in results if r.status == "executed")
    n_rolled   = sum(1 for r in results if r.status == "rolled_back")
    n_orphaned = sum(1 for r in results if r.status == "rollback_failed")
    # "manual_review": the trader could not tell what a leg did (see
    # reporter.TradeResult) and sent no follow-up order. It needs a person as
    # urgently as a failed rollback, so both are counted in one alert. A
    # pair stopped by a disproof of the V2 NO-leg mapping sent nothing and
    # comes back "failed", so it is not counted here; the disproving pair's
    # manual_review is what returns EXIT_TRADES_NEED_ATTENTION.
    n_unknown  = sum(1 for r in results if r.status == "manual_review")
    message = (f"Submitted {n_ok} of {len(results)} order pair(s) successfully. "
               f"{n_rolled} rolled back, {n_orphaned} rollback failure(s), "
               f"{n_unknown} unknown fill state(s).{sale_note}")
    logging.info("%s", message)
    report.message = message
    if n_orphaned or n_unknown:
        logging.critical(
            "%d pair(s) may have ORPHANED positions and %d pair(s) have an "
            "UNDETERMINED fill state — manual review required (see trade log).",
            n_orphaned, n_unknown,
        )
        # Surface via the process exit code too (BS-14) — the scheduler reads
        # this, not the log file, and needs to distinguish "trades happened
        # but need a human to check" from a clean run.
        return EXIT_TRADES_NEED_ATTENTION

    return clean_exit


def _build_parser() -> argparse.ArgumentParser:
    """
    Build main.py's command-line parser: every flag, with the "live trading toggles" group.

    main() parses sys.argv with it. A test builds it to parse a list of flags,
    such as config.live_settings_argv's, the way main() would.

    Returns:
        argparse.ArgumentParser: The parser, with nothing parsed yet.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Kalshi Arbitrage Bot — scans for two kinds of mispriced contract "
            "pairs (same-title near-arbitrage pairs, and directional "
            "time-series pairs betting against the market's implied chance "
            "that an event first happens between two deadlines), sizes them "
            "with the Kelly criterion, and submits fill-or-kill orders."
        ),
    )
    parser.add_argument(
        "--mode", choices=["dev", "prod"], default="dev",
        help="'dev' scans real sandbox markets and simulates; 'prod' uses real account",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="(prod only) Discover and size trades but do not submit orders",
    )
    parser.add_argument(
        "--sandbox-balance", type=float, default=1000.0, metavar="DOLLARS",
        help="Virtual balance in dollars used for trade sizing in dev mode (default: 1000)",
    )
    parser.add_argument(
        "--max-horizon-days", type=int, default=None, metavar="DAYS",
        help="Only consider markets closing within DAYS from now (both modes; default: no limit)",
    )
    parser.add_argument(
        "--result-file", type=pathlib.Path, default=None, metavar="PATH",
        help="(prod only) When the run ends, write what it did (outcome, trades, "
             "warnings) to PATH as JSON, replacing any file there; a usage error "
             "(exit 2, reason on stderr) or a kill signal leaves no file",
    )
    # Every flag defaults to None (not given); LiveSettings alone validates the
    # values — no choices, range check or literal here
    live = parser.add_argument_group(
        "live trading toggles",
        "Override one live default for THIS run only, in either mode "
        "(adding to held pairs and selling: production runs only). The live "
        "defaults are the ones saved through python3 -m kalshi_betting.defaults_server "
        "(live_defaults.json); a run refuses to start without them. The weekly scheduler "
        "passes none of these flags, so a scheduled run trades exactly the saved defaults.",
    )
    live.add_argument(
        "--tier-floors", action=argparse.BooleanOptionalAction, default=None,
        help="Apply (or, with --no-tier-floors, drop) the deadline-gap tier floors on "
             "time-series pairs (default: the saved live defaults)",
    )
    live.add_argument(
        "--spread-min", type=float, default=None, metavar="X",
        help="Time-series spread-band FLOOR on pB - pA, 0-1 "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--spread-max", type=float, default=None, metavar="Y",
        help="Time-series spread-band CEILING on pB - pA, 0-1 "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--interval-discount", type=float, default=None, metavar="K",
        help="Time-series interval discount k, in (0, 1] "
             "(default: the saved live defaults)",
    )
    # The caps' grid step in percent (SIZE_CAP_STEP); argparse %-formats help, hence "%%"
    cap_step = f"{SIZE_CAP_STEP * 100:g}"
    live.add_argument(
        "--size-cap", type=int, default=None, metavar="PCT",
        help=f"Per-trade Kelly cap for every pair, in whole percent, in {cap_step}%% "
             "steps; 100 = no cap (default: the saved live defaults)",
    )
    live.add_argument(
        "--same-title-size-cap", type=int, default=None, metavar="PCT",
        help=f"Extra per-trade cap on same-title pairs, in whole percent, in {cap_step}%% "
             "steps; 100 = no extra cap beyond --size-cap "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--add-to-held-pairs", action=argparse.BooleanOptionalAction, default=None,
        help="Let this run add to a pair the account already holds (exactly the same "
             "two markets, the same side on each), sizing the old and new contracts "
             "together as a share of the portfolio value (the old ones at today's "
             "prices plus the fees paid for them). --no-add-to-held-pairs never "
             "trades a held market. Production runs only: a dev run holds nothing "
             "(default: the saved live defaults)",
    )
    sell_level = live.add_mutually_exclusive_group()
    sell_level.add_argument(
        "--sell-at", type=int, default=None, metavar="PCT",
        help="Sell a held position once it has stayed at or above PCT%% of its potential profit "
             "for config.TAKE_PROFIT_HOLD_DAYS days in a row; a whole percent from 1 to "
             "100. Production runs only (default: the saved live defaults)",
    )
    sell_level.add_argument(
        "--no-sell", action="store_true", default=None,
        help="Sell nothing this run, whatever the saved live defaults say (also clears "
             "the saved minimum of days; cannot be combined with --sell-min-days)",
    )
    sell_days = live.add_mutually_exclusive_group()
    sell_days.add_argument(
        "--sell-min-days", type=int, default=None, metavar="N",
        help="Sell a position only while at least N days (1 or more) remain before "
             "its last market stops trading; needs a sell level, saved or given with "
             "--sell-at (default: the saved live defaults)",
    )
    sell_days.add_argument(
        "--no-sell-min-days", action="store_true", default=None,
        help="Set no minimum of days to maturity this run, whatever the saved live "
             "defaults say",
    )
    # Filed as the backtest dashboard files a trade (_filter_by_category)
    categories = live.add_mutually_exclusive_group()
    categories.add_argument(
        "--category", action="append", default=None, metavar="NAME",
        help="Trade only pairs filed under this Kalshi category, as the backtest "
             "dashboard's Category select names it (repeatable; case-insensitive; "
             "default: the saved live defaults)",
    )
    categories.add_argument(
        "--any-category", action="store_true", default=None,
        help="Trade any category this run, whatever the saved live defaults say",
    )
    tags = live.add_mutually_exclusive_group()
    tags.add_argument(
        "--tag", action="append", default=None, metavar="NAME",
        help="Trade only pairs whose series' FIRST Kalshi tag is NAME, under ANY "
             "category unless --category narrows it: the backtest dashboard's Tag "
             "option \"C · T\" is --category C --tag T (repeatable; case-insensitive; "
             "combined with --category by AND; default: the saved live defaults)",
    )
    tags.add_argument(
        "--any-tag", action="store_true", default=None,
        help="Trade any tag this run, whatever the saved live defaults say",
    )
    return parser


def _setup_logging(log_path: pathlib.Path) -> None:
    """
    Configure root logging with both a console handler and a file handler.

    Uses logging.basicConfig() with two handlers so every log line reaches
    both the terminal (for interactive/foreground runs) and the persistent
    log file (for later inspection, e.g. by the scheduler daemon). The file
    handler is opened with delay=True so merely calling this function — e.g.
    from a test or an import that doesn't go on to log anything — does not
    touch disk; the file is only created on the first emitted record.

    The file handler ROTATES (BS-25): a scheduler daemon runs this process
    weekly forever, and a plain FileHandler would grow kalshi_arb.log without
    bound (see backtest.py's kalshi_backtest.log for the failure mode that
    motivated this). 5MB across 3 backups is logging infrastructure sized for
    this CLI's own verbosity, not a strategy constant, so it stays inline here
    rather than moving to config.py.

    Args:
        log_path (pathlib.Path): Path to the log file to append to.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True,
            ),
        ],
    )


def main() -> None:
    """
    CLI entry point for the Kalshi Arbitrage Bot — the live pipeline that
    scans for same-title near-arbitrage pairs and directional time-series
    pairs, sizes them, and trades them.

    Parses command-line arguments (--mode, --dry-run, --sandbox-balance,
    --max-horizon-days, and the "live trading toggles" group), exits 2 if
    config.ORDER_API_VERSION is not "v2" (config.order_api_version_error),
    checks --max-horizon-days, resolves the run's settings and the saved live
    defaults they were built from (_resolve_live_settings: with no defaults
    saved, a refused file or a bad flag it exits 2) — all before logging is
    configured or any request is made — then configures logging, builds the
    Kalshi client and runs _run_dev (sandbox simulation) or _run_prod (real
    trading). Ends with sys.exit() and the run's return code (the EXIT_*
    constants in config.py), which the scheduler reads. An unhandled exception
    propagates and exits 1.

    A production run that sends orders (--mode prod without --dry-run) first
    takes the machine-wide live-run lock (run_lock.acquire), before it builds
    the client, and holds it until the run mode returns or raises; the lock is
    released in a finally, so a second call in the same process takes it
    afresh. When another run still holds the lock after
    config.LIVE_RUN_LOCK_WAIT_SECONDS, this run logs a WARNING naming the
    holder and exits EXIT_RUN_IN_PROGRESS (50) without building a client or
    making any request. An error making or opening the lock file propagates
    (exit 1). Dry runs and dev runs send no orders, so they neither take the
    lock nor wait for it.

    With --result-file PATH (--mode prod only: in dev it exits 2 right after
    the order-path check, before logging is configured, a client is built or
    any request is made), any file already at PATH is removed right after
    that check, and a reporter.RunReport is filled in as the run goes: the
    run's settings and where its defaults came from, every WARNING-or-worse
    line logged after logging is configured (reporter.RunReportHandler, on
    the root logger), and what _run_prod records. It is written to PATH as
    JSON (reporter.write_run_report) in a finally, so however the run ends
    once logging is configured — with the exit code, or with no exit code
    and a one-line error (_http.api_error_summary) when an exception stops
    it, which still propagates — before the handler is removed and the lock
    released. A usage error that exits 2 before logging is configured (a bad
    flag, no or refused saved live defaults, a non-v2 order path) and a kill
    signal leave no file, so a caller reads the exit code first. Neither
    filling nor writing it changes the exit code.

    Returns:
        None: This function never returns to its caller — it always ends by
            calling sys.exit(code), which raises SystemExit.
    """
    # Every flag, their help and the live-toggle group (_build_parser)
    parser = _build_parser()
    args = parser.parse_args()
    # Exit 2 unless ORDER_API_VERSION is "v2", before anything is logged, a
    # client is built or an order could be sent
    problem = order_api_version_error()
    if problem:
        parser.error(problem)
    # The run result describes a production run only; refused right after the
    # order-path check, before logging, a client or any request
    if args.result_file is not None and args.mode != "prod":
        parser.error("--result-file is for --mode prod only")
    if args.result_file is not None:
        # An older result at PATH must never pass for this run's: whatever is
        # there when main() ends was written by this run, or nothing is (a
        # usage error below, or a kill signal). A file that cannot be removed
        # is left to the final write, which logs why it cannot replace it.
        with contextlib.suppress(OSError):
            args.result_file.unlink(missing_ok=True)
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    # Read and validated BEFORE logging is configured: no saved live defaults,
    # a refused file or a bad flag exits 2 with nothing logged or requested
    settings, reference = _resolve_live_settings(args, parser)

    # Echo to the console (foreground/interactive runs) as well as the
    # persistent log file (later inspection, scheduler-spawned runs)
    _setup_logging(PROJECT_ROOT / "kalshi_arb.log")

    # With --result-file (prod only): what the run did, filled in as it goes
    # and written in the finally below, however the run ends from here on; the
    # handler copies in every WARNING-or-worse line logged from here on
    report = handler = None
    if args.result_file is not None:
        report = RunReport(dry_run=args.dry_run, started_at=datetime.now(UTC),
                           settings=describe_live_settings(settings, reference),
                           defaults=reference.origin)
        handler = RunReportHandler(report)
        logging.getLogger().addHandler(handler)
    # One production run that sends orders at a time on this machine
    # (run_lock); dry runs and dev runs send none, so they neither take the
    # lock nor wait for it
    lock_fd = None
    code = None
    try:
        if args.mode == "dev" and args.dry_run:
            # _run_dev always calls execute_trades(dry_run=True) regardless of
            # args.dry_run (dev never submits real orders), so --dry-run has no
            # effect in dev mode. Logged (not parser.error'd) after basicConfig so
            # it lands in kalshi_arb.log for anyone diagnosing "why didn't
            # --dry-run change anything" after the fact.
            logging.warning("--dry-run is inert in dev mode — dev never submits orders")

        if args.mode == "prod" and args.sandbox_balance != parser.get_default("sandbox_balance"):
            # Prod sizes on the real account, so warn that --sandbox-balance does nothing
            logging.warning(
                "--sandbox-balance is inert in prod mode — prod sizes on the account's "
                "portfolio value (its cash plus its open positions' value) and spends "
                "only its cash; use --dry-run to avoid submitting orders",
            )

        if args.mode == "prod" and not args.dry_run:
            # The machine-wide lock every real-money run shares; None when
            # another run still holds it after LIVE_RUN_LOCK_WAIT_SECONDS
            lock_fd = run_lock.acquire()
            if lock_fd is None:
                # The record names the run in the way (run_lock.holder)
                message = (f"Another live trading run is in progress "
                           f"({run_lock.holder().describe()}) — this run stops "
                           f"without contacting Kalshi, so nothing is sent "
                           f"(exit {EXIT_RUN_IN_PROGRESS}).")
                logging.warning("%s", message)
                if report is not None:
                    report.message = message
                code = EXIT_RUN_IN_PROGRESS
        if code is None:
            client = build_client(args.mode)  # returns KalshiClient authenticated via RSA key from secrets.json

            # Both run modes get the run's settings and the saved defaults they were
            # built from; a production run also gets the report to fill (None
            # without --result-file)
            if args.mode == "dev":
                code = _run_dev(client, args, settings, reference)
            else:
                code = _run_prod(client, args, settings, reference, report=report)
    except BaseException as exc:
        if report is not None:
            # One line, as the trade log records a failed order, never a traceback
            report.error = api_error_summary(exc)
        raise
    finally:
        if report is not None:
            # Written before the lock is released, with the exit code, or None
            # when an exception is on its way out (report.error names it);
            # never raises
            write_run_report(args.result_file, report, code)
            logging.getLogger().removeHandler(handler)
        if lock_fd is not None:
            # Released only once the run mode has returned or raised and its
            # result is written; nothing trades after this
            os.close(lock_fd)

    # Only sys.exit() communicates the outcome to a subprocess caller (the
    # scheduler) — a bare return here would always look like exit 0.
    sys.exit(code)


if __name__ == "__main__":
    main()
