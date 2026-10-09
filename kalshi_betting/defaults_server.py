"""
File: defaults_server.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    A small local web server that saves the live trading defaults
    (config.LIVE_DEFAULTS_FILE, live_defaults.json), which every live run
    starts from, and starts live trading runs with them. Run by a person,
    deliberately, from a terminal, through the launcher at the checkout's
    root (./start_dashboard.sh [--seed] [--no-browser], which checks that its
    Python can import the live bot and starts the live dashboard too) or directly:

        python3 -m kalshi_betting.defaults_server [--seed] [--no-browser]

    On start it opens one page: with --seed the confirmation page proposing
    the seed values; otherwise the backtest dashboard, or its own index when
    there is no dashboard, the dashboard was built before its Save and Trade
    buttons, or it cannot be read. When its port is already taken by this
    checkout's own server, running this checkout's current code, it opens
    that page from the running server and exits, starting nothing; when the
    port is held by that server running older code, by another checkout's
    server, or by anything else, it refuses (exit 2). With --no-browser it
    opens no page; the launcher passes it when the live dashboard opens one.

    Its pages:
      - /confirm, opened with the proposed settings in its address (by the
        backtest dashboard's filter bar, or by --seed, which proposes
        config.LIVE_DEFAULTS_SEED). It shows the live defaults in force
        beside the proposed ones with every change highlighted, and offers
        three buttons, always in this order: Dry run (runs the live bot with
        the proposed settings but sends no orders, and saves nothing),
        Confirm and save, and, last and set apart, Confirm and trade (saves
        the proposed settings, then runs the live bot with them, placing real
        orders). Closing the tab cancels; nothing is written on the way in.
        With no live defaults saved, every live run refuses to start, so this
        is also how the first file is made.
      - /trade shows the saved defaults and offers Dry run and Confirm and
        trade with them.
      - /runs/<id> shows one run: while it runs, its progress and the end of
        its output; once it ends, its outcome, read from the result file the
        run wrote (main.py --result-file) and the exit code.
      - / lists what the server is for and the newest runs; /saved shows the
        defaults just saved; /checkout answers which checkout this server
        serves and a fingerprint of the code it loaded, as JSON (a second
        start in the same checkout reads it to find its own server on the
        port).

    Each run is its own `python -m kalshi_betting.main --mode prod` process,
    started with all ten toggles as explicit flags (config.live_settings_argv)
    so it trades exactly what its page showed, in a new session (so Ctrl-C on
    this server, or closing its terminal, never reaches it), from
    config.PROJECT_ROOT, writing everything it prints to its own folder under
    config.LIVE_RUNS_DIR. The server never places an order itself.

Dependencies:
    Imports config and run_lock only (besides the standard library). From
    config: LiveSettings, LIVE_TOGGLE_FIELDS (the ten toggle names the
    fingerprint reads), LiveDefaultsError (a refused saved file) and the
    saved-defaults helpers (read_saved_live_defaults, save_live_defaults,
    live_settings_changes, describe_live_settings, live_rule_warnings,
    live_defaults_source), live_settings_argv (a run's flags), the seed values
    and their source note, the source-note pattern, the exit codes, the run
    result's format tag and the DEFAULTS_SERVER_*, DASHBOARD_FILENAME,
    DASHBOARD_MARKER_SCAN_BYTES and SCHEDULER_* constants, and count_text
    (a run page writes an add-on's held count exactly). From run_lock:
    held() and holder(), to refuse a real-money run while another live
    trading run holds the machine's lock.
    It reads config.LIVE_DEFAULTS_FILE, config.PROJECT_ROOT,
    config.LIVE_RUNS_DIR and config.LIVE_RUN_LOCK_FILE through the module at
    call time, so the tests' redirects reach it. Nothing imports this module:
    it is a tool a person runs, and tests/test_defaults_server.py checks that
    no other module of the package imports it, and that it imports nothing
    of the package but config and run_lock.

Notes:
    It listens on 127.0.0.1 only, answers one request at a time (so two
    saves or two starts can never interleave) and never serves the
    dashboard: any script on a page served from its own address could press
    its buttons. Every request it can read must name this server in its Host
    header (a DNS-rebinding defence); one it cannot read (a malformed request
    line, say) gets only a refusal page. A POST must also come from this
    server's own page (its Origin header), carry the token that page was
    built with (an HMAC, keyed by this process, over the page's purpose —
    "confirm" or "trade" — the fingerprint of the defaults in force, a
    one-time nonce and the proposed fields) and find the same defaults still
    in force; otherwise it is refused, or the page is shown again against the
    defaults now in force. A page's nonce works once: a second POST of the
    same page (a double click, a resubmit) is sent to what the first one
    produced, and nothing is done twice. Which button was pressed (the
    action) is not signed, since one form carries several buttons; every
    action's own checks are made again. Every response, a refusal included,
    is sent as HTTP/1.0 with the same six headers: it forbids framing and
    lets a browser run only the one script whose hash its
    Content-Security-Policy names. That script enables the buttons only after
    the page has been visible for DEFAULTS_SERVER_CONFIRM_ARM_MS and the
    mouse moves or a key is pressed, so a click aimed at another page cannot
    land on one, and disables them for good once one is pressed. A button
    that does not apply is still shown, disabled, in its place, with the
    reason beside it, so the real-money button is always last and never
    moves under a habitual click. The source note a page proposes must be
    one of the two shapes config.LIVE_DEFAULTS_SOURCE_PATTERN allows, or be
    left out, and the seed's note may label only the seed values, so a
    crafted link cannot choose the note's words. Category and tag names are
    checked for form only (one printable name each), so a link can still put
    words of its own there: the page shows them as a highlighted change, and
    a name no Kalshi series is filed under matches no pair. The headers never
    include "Referrer-Policy: no-referrer": under it a browser sends a POST's
    Origin as "null", and every honest save would be refused.

    These checks tell this server's own pages from other web pages. They do
    not identify local programs: anything on this machine that can connect
    to 127.0.0.1 on the server's port can fetch a page and post it back, and
    so save settings or start a run. Stop the server when you are done.

    A real-money run is refused while a run started from this checkout's
    server is still going (one started before a restart of the server
    included, found by the lock its process holds on its output.log) or
    while another live trading run holds the machine's lock (run_lock.held,
    a momentary check that a starting run's own wait rides out); a dry run
    only while a run started from this checkout's server is still going.
    Confirm and trade is also refused, until a box is ticked, when the newest
    finished real-money run that says anything about the account (this
    server's or the scheduler's; a dry run, or a run that stopped before it
    could send an order, is passed over) ended needing attention (exit
    config.EXIT_TRADES_NEED_ATTENTION) or without a clean result. The server
    never stops a run: killing one while it sends orders could leave half of
    a pair open.

    The one request it makes itself is to its own port, when a start finds
    that port taken: GET /checkout, with no proxy, no redirect followed, a
    bounded wait and a bounded read. Only an answer naming this checkout
    (config.PROJECT_ROOT, resolved) counts as its own server, and it is
    reused only when the code it loaded matches this checkout's code now
    (a SHA-256 over the package's .py files), so a server left running
    across a code change is restarted rather than trusted with a run;
    anything else gets a refusal. That answer is a claim, not proof: a local
    program listening on the port could make it, as it could already serve
    look-alike pages at the addresses the dashboard's buttons open.
"""
import argparse
import errno
import fcntl
import hashlib
import hmac
import html
import http.client
import json
import logging
import logging.handlers
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import urllib.request
import webbrowser
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from . import config, run_lock
from .config import (
    DASHBOARD_FILENAME,
    DASHBOARD_MARKER_SCAN_BYTES,
    DEFAULTS_SERVER_CHECKOUT_MAX_BYTES,
    DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS,
    DEFAULTS_SERVER_CONFIRM_ARM_MS,
    DEFAULTS_SERVER_HOST,
    DEFAULTS_SERVER_INDEX_RUNS,
    DEFAULTS_SERVER_MAX_REQUEST_BYTES,
    DEFAULTS_SERVER_PORT,
    DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES,
    DEFAULTS_SERVER_RUN_LOG_TAIL_LINES,
    DEFAULTS_SERVER_RUN_REFRESH_SECONDS,
    DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS,
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    LIVE_DEFAULTS_SEED,
    LIVE_DEFAULTS_SEED_SOURCE,
    LIVE_DEFAULTS_SOURCE_PATTERN,
    LIVE_RUN_RESULT_FORMAT,
    LIVE_TOGGLE_FIELDS,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
    SCHEDULER_STATE_FILENAME,
    LiveDefaultsError,
    LiveSettings,
    count_text,
    describe_live_settings,
    live_defaults_source,
    live_rule_warnings,
    live_settings_argv,
    live_settings_changes,
    read_saved_live_defaults,
    save_live_defaults,
)

# The fields a confirmation request carries, in the GET query and repeated in
# the POST body; the signed text is built from exactly these, as raw strings
_FIELDS = ("tier_floors", "spread_min", "spread_max", "k", "size_cap",
           "same_title_size_cap", "add_to_held_pairs", "sell_at", "sell_min_days",
           "category", "tag", "source")
# The fields a proposal must carry; every other one may be left out
_REQUIRED = ("tier_floors", "spread_min", "spread_max", "k", "size_cap")
# The largest number of fields a query or form may hold (its twelve fields plus
# the fingerprint, nonce, token, action and acknowledgement, with room to
# spare); more is refused unread
_MAX_FIELDS = 20
# A plain decimal number in ASCII digits: no underscores, spaces, digits of
# other scripts, "nan" or "inf"
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", re.ASCII)
# A whole number of days: one to six ASCII digits, with no sign or spaces
_WHOLE_NUMBER = re.compile(r"\d{1,6}", re.ASCII)
# A token or fingerprint: 64 lower-case hex digits (a SHA-256 hex digest)
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
# A page's one-time nonce: 32 lower-case hex digits
_HEX32 = re.compile(r"[0-9a-f]{32}", re.ASCII)
# A run's id: 16 lower-case hex digits
_RUN_ID = re.compile(r"[0-9a-f]{16}", re.ASCII)
# A run's folder under config.LIVE_RUNS_DIR: its UTC start time, then its id
_RUN_FOLDER = re.compile(r"\d{8}T\d{6}Z-([0-9a-f]{16})", re.ASCII)
# A Content-Length value: one or more ASCII digits
_DIGITS = re.compile(r"\d+", re.ASCII)
# The media type a browser gives a form it posts
_FORM_TYPE = "application/x-www-form-urlencoded"
# The rotating log file's name in PROJECT_ROOT
_LOG_NAME = "kalshi_defaults_server.log"
# What /confirm's buttons post as "action", and what /trade's post
_CONFIRM_ACTIONS = ("save", "trade", "dry_run")
_TRADE_ACTIONS = ("trade", "dry_run")
# The files of a run's folder: the result main.py --result-file writes, what
# the run printed, and what the run is (written before it starts)
_RESULT_NAME, _OUTPUT_NAME, _RUN_NAME = "result.json", "output.log", "run.json"
# The ids of the buttons the page script arms, in page order: Dry run, Confirm
# and save, Confirm and trade. A placeholder for a button that does not apply
# has no id, so it is never armed
_ARMED_BUTTONS = ("confirm-dry-run", "confirm", "confirm-trade")
# UTC times as a run's folder records them (run_lock and the run result write
# the same format), and the start time at the head of a run folder's name
_TIME = "%Y-%m-%dT%H:%M:%SZ"
_FOLDER_TIME = "%Y%m%dT%H%M%SZ"

# The one script any page runs: it enables the page's buttons (those of
# _ARMED_BUTTONS present) only after the page has been visible for
# DEFAULTS_SERVER_CONFIRM_ARM_MS and the mouse moves or a key is pressed,
# disables them again whenever the page is hidden, and disables them for good
# once the form is sent, so a second click sends nothing
_CONFIRM_JS = """
(function() {
  var IDS = __IDS__;
  var buttons = IDS.map(function(id) { return document.getElementById(id); })
                   .filter(function(b) { return b; });
  if (!buttons.length) { return; }
  var timer = null;
  var ready = false;
  var sent = false;
  function disableAll() {
    buttons.forEach(function(b) { b.disabled = true; });
  }
  var form = buttons[0].form;
  if (form) {
    form.addEventListener('submit', function() {
      sent = true;
      // Only after the submission has read which button was pressed: a
      // disabled button sends no value
      setTimeout(disableAll, 0);
    });
  }
  function watch() {
    if (document.visibilityState === 'visible') {
      if (timer === null && !ready) {
        timer = setTimeout(function() { ready = true; }, ARM_MS);
      }
    } else {
      if (timer !== null) { clearTimeout(timer); timer = null; }
      ready = false;
      disableAll();
    }
  }
  function arm() {
    if (ready && !sent && document.visibilityState === 'visible') {
      buttons.forEach(function(b) { b.disabled = false; });
    }
  }
  document.addEventListener('visibilitychange', watch);
  document.addEventListener('mousemove', arm);
  document.addEventListener('keydown', arm);
  watch();
})();
""".replace("__IDS__", json.dumps(list(_ARMED_BUTTONS))).replace(
    "ARM_MS", str(DEFAULTS_SERVER_CONFIRM_ARM_MS))

# The Content-Security-Policy hash of that script, so the browser runs it and
# nothing else
_CONFIRM_JS_HASH = "sha256-" + b64encode(
    hashlib.sha256(_CONFIRM_JS.encode("utf-8")).digest()).decode("ascii")

# The policy every response carries: nothing loads but inline styles and the
# one script above, no page may frame this one, and a form may post only here
_CSP = ("default-src 'none'; style-src 'unsafe-inline'; "
        f"script-src '{_CONFIRM_JS_HASH}'; form-action 'self'; base-uri 'none'; "
        "frame-ancestors 'none'")

# The type of every page; /checkout alone answers JSON
_HTML_TYPE = "text/html; charset=utf-8"
_JSON_TYPE = "application/json; charset=utf-8"

# The page style: the dashboard's font and heading colour, a changed row in amber
_STYLE = """
body { font-family: sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; color: #212121; }
h1 { color: #1A237E; }
table { border-collapse: collapse; font-size: 14px; margin: 12px 0; }
th, td { border: 1px solid #E0E0E0; padding: 6px 12px; text-align: left; }
th { background: #F5F5F5; }
tr.changed td { background: #FFF8E1; }
.tag { color: #E65100; font-size: 12px; font-weight: 700; }
.note { color: #616161; font-size: 13px; }
.warn { color: #B71C1C; }
.banner { color: #B71C1C; border: 1px solid #B71C1C; padding: 8px 12px; }
.ok { color: #1B5E20; }
.gap { margin-top: 32px; }
.money { color: #B71C1C; font-weight: 700; }
pre { background: #FAFAFA; border: 1px solid #E0E0E0; padding: 8px; font-size: 12px;
      white-space: pre-wrap; overflow-x: auto; }
button { font-size: 15px; padding: 8px 16px; }
"""

# The banner shown when a POST finds different defaults in force than the page was built on
_STALE_BANNER = ("The live defaults changed after this page was shown, so nothing was "
                 "saved or traded. The page below is built on the defaults in force now; "
                 "use it again only if you still want to.")

# The banner shown when Confirm and trade is pressed without ticking the
# acknowledgement the last run's attention banner asks for
_ACK_BANNER = ("Nothing was saved or traded: tick “I have read the last run's result and "
               "want to place real orders anyway” first, or use Dry run.")

# The acknowledgement a real-money run needs while the attention banner shows
_ACK_LABEL = "I have read the last run's result and want to place real orders anyway"

# How to start a server that can run trades: its own main() hands _App the
# process starter; an _App built without one refuses every run
_NOT_STARTED_TO_TRADE = ("This server was not started to run trades — start it with "
                         "./start_dashboard.sh")

# What a dashboard file holds when it has the filter bar's Save and Trade
# buttons: the Trade link's id, which dashboard.py writes in the page's
# opening part (in the filter bar, or under the notice that replaces it)
_DASHBOARD_MARKER = b'id="flt-trade"'

# What _dashboard_state finds at the dashboard's path
_DASHBOARD_MISSING, _DASHBOARD_UNREADABLE, _DASHBOARD_OLD, _DASHBOARD_READY = (
    "missing", "unreadable", "old", "ready")

# The reasons shown beside a button that does not apply
_SAVE_FIRST = "A dry run needs saved defaults — save these first"
_NOTHING_TO_SAVE = "Nothing to save — these are already the live defaults"

# The notes that follow the buttons
_DRY_RUN_NOTE = ("Dry run runs the same search and sizing but sends no orders and leaves "
                 "the saved defaults alone. It adds “simulated” rows to trade_log.xlsx.")
_ARMING_NOTE = ("The buttons become clickable a moment after this page is shown, once you "
                "move the mouse or press a key.")

# The notes a run gets about what it was started with
_SAVED_NOTE = "Saved and verified: the file was read back and matches these settings."
_NOTHING_SAVED_NOTE = "Nothing to save: these are already the live defaults."
_DRY_RUN_SAVED_NOTE = "Not saved: a dry run leaves the live defaults unchanged."

# The warning a run page gives when orders may have been sent before the run ended
_CHECK_POSITIONS = ("Orders may have been placed — check your positions in the Kalshi UI "
                    "and kalshi_arb.log before trading again.")

# What a run page says under "No trade completed"
_EACH_PAIR_BELOW = "Each pair's status and error are listed below."

# The statuses a run result's pairs can have, grouped as the run page lists them
_ATTENTION_STATUSES = ("rollback_failed", "manual_review")
_TRADE_GROUPS = (("Needs attention", _ATTENTION_STATUSES), ("Completed", ("executed",)),
                 ("Would have traded", ("simulated",)))

# Sale (take-profit) statuses that need a person: a pair left uneven, or an unknown outcome
_SALE_ATTENTION_STATUSES = ("unbalanced", "manual_review")

# A sale's status in plain words, as the trade log's Status column words it
_SALE_STATUS_WORDS = {"sold": "sold", "partly_sold": "partly sold", "not_sold": "not sold",
                      "unbalanced": "unbalanced", "manual_review": "check",
                      "simulated": "simulated"}

# What a run page says under its Sales table
_SALE_PROFIT_NOTE = ("Profit is what selling the whole position realizes at the bids the "
                     "sale was priced at, after fees; it is shown only for a position that "
                     "sold in full (or, in a dry run, would have).")

# The exit code argparse gives a command line it refuses: the run stopped
# before it logged anything, so it sent nothing
_USAGE_EXIT = 2

# The exit code Python gives an exception nothing caught
_ERROR_EXIT = 1

# Exit codes of a real-money run that stopped before sending any order: a bad
# command line, too little money, nothing to scan, or another run in progress.
# The attention warning skips them, since they say nothing about the account
_NO_ORDER_EXITS = frozenset({_USAGE_EXIT, EXIT_SKIPPED_LOW_BALANCE, EXIT_NO_TRADEABLE_SHARDS,
                             EXIT_RUN_IN_PROGRESS})

# The exit codes of a real-money run that finished trading with nothing left
# for a person to check: the newest such run clears the attention warning
_CLEAN_EXITS = frozenset({EXIT_OK, EXIT_TIME_SERIES_SKIPPED})

# How the attention warning ends, after "The last real-money run (<which>)":
# for a run that needed a person, and for one that ended without a clean
# result ({why} says how it ended). After a V2 order-mapping disproof every new
# real-money run is a new process that would open another wrong-side position,
# so _ATTENTION_TAIL names every way one starts: the scheduler daemon, main.py
# by hand and this page's Confirm and trade. What to undo by hand is what the
# run's CRITICAL names: a market the account held before the run (a pair it
# added to, or an earlier trade) goes back to what it held, never to 0.
_ATTENTION_TAIL = (" ended needing manual attention ({why}). Read its result first. If it "
                   "reports a V2 order-mapping disproof, stop trading (a new real-money run "
                   "would open another wrong-side position, so stop the scheduler daemon if "
                   "it is running, do not run main.py --mode prod, and do not press Confirm "
                   "and trade), then undo by hand in the Kalshi UI what its CRITICAL names: "
                   "close out a market the account did not hold before the run, and put one "
                   "it did hold back to what it held.")
_UNCLEAN_TAIL = " ended without a clean result ({why}). " + _CHECK_POSITIONS

# The attention warning when the runs' records could not be read at all: it
# fails closed, so a real-money run still needs the box ticked
_OUTCOME_UNREADABLE = ("The outcome of the last real-money run could not be read (see "
                       f"{_LOG_NAME}). " + _CHECK_POSITIONS)


@dataclass(frozen=True)
class _Request:
    """
    One HTTP request, as the application sees it.

    Attributes:
        method (str): "GET" or "POST".
        target (str): The request target, e.g. "/confirm?k=0.8".
        host (str | None): The Host header; None when missing or given twice.
        origin (str | None): The Origin header; None when missing or given twice.
        content_type (str | None): The Content-Type header; None when missing
            or given twice.
        body (bytes): The request body; empty for a GET.
    """
    method: str
    target: str
    host: str | None
    origin: str | None = None
    content_type: str | None = None
    body: bytes = b""


@dataclass(frozen=True)
class _Response:
    """
    One HTTP response: a status, a body and, for a redirect, where to.

    Attributes:
        status (int): The HTTP status code.
        body (str): The HTML page (or, for /checkout, the JSON).
        location (str | None): The Location header of a redirect; None otherwise.
        content_type (str): The Content-Type header; an HTML page unless said otherwise.
    """
    status: int
    body: str
    location: str | None = None
    content_type: str = _HTML_TYPE


@dataclass(frozen=True)
class _Notice:
    """
    One reason or warning a page shows, in two forms.

    Attributes:
        text (str): Plain text, for a log line or a refusal page.
        html (str): The same for a page, escaped, perhaps with a link.
    """
    text: str
    html: str


@dataclass(frozen=True)
class _Slot:
    """
    One button of a page's form, in its fixed place.

    Attributes:
        label (str): The button's text.
        button_id (str): Its id (one of _ARMED_BUTTONS) when it applies.
        action (str): The action it posts when it applies.
        reason (str | None): Why it does not apply, as HTML; None when it
            applies (a real button), else it is shown as a disabled
            placeholder with no id, name or value, the reason beside it.
    """
    label: str
    button_id: str
    action: str
    reason: str | None


@dataclass(frozen=True)
class _Controls:
    """
    What a page's form carries: its signed fields and its buttons.

    Attributes:
        fingerprint (str): _fingerprint of the defaults the page was built on.
        nonce (str): The page's one-time nonce (32 hex digits).
        token (str): The token signing the page's purpose, fingerprint, nonce
            and proposal fields.
        slots (tuple[_Slot, ...]): The buttons in page order; the last is
            Confirm and trade.
        attention (_Notice | None): The warning about the last real-money
            run, or None.
    """
    fingerprint: str
    nonce: str
    token: str
    slots: tuple
    attention: _Notice | None


@dataclass
class _Run:
    """
    One live trading run the server started, or found in config.LIVE_RUNS_DIR.

    Attributes:
        run_id (str): Its id (16 hex digits), the end of its folder's name.
        folder (Path): Its folder: run.json, output.log and, once it ends, result.json.
        process (subprocess.Popen | None): The process, for a run this server
            started; None for one read from disk (started before this server
            restarted, or by another server of this checkout).
        dry_run (bool): True when it sends no orders; a run whose folder does
            not say is taken as a real-money run.
        settings_text (str): Its settings in the "Live settings:" line's words.
        started_at (datetime | None): When it started, in UTC; None when not recorded.
        saved_note (str): What happened to the live defaults before it started.
        pid (int | None): Its process id; None when not recorded.
        start_error (str | None): Why it could not be started, when it could not.
    """
    run_id: str
    folder: Path
    process: subprocess.Popen | None
    dry_run: bool
    settings_text: str
    started_at: datetime | None
    saved_note: str
    pid: int | None = None
    start_error: str | None = None


@dataclass
class _NonceUse:
    """
    What a page's one-time nonce produced, once its action has been taken.

    Attributes:
        path (str | None): The page the action produced (/saved or /runs/<id>),
            where a second POST of the same page is sent; None when it
            produced none.
        outcome (str): What happened instead, for a second POST's refusal.
    """
    path: str | None
    outcome: str


@dataclass(frozen=True)
class _Result:
    """
    A run's result file (main.py --result-file), as read.

    Attributes:
        state (str): "ok", "missing" or "unreadable".
        record (dict): The file's fields, each type-checked; empty unless state is "ok":
            dry_run, exit_code, finished_at, message, error, submission_started,
            balance_before and balance_after (cash), portfolio_value_before (cash plus
            open positions), trades, warnings and warnings_dropped.
    """
    state: str
    record: dict


@dataclass(frozen=True)
class _Outcome:
    """
    How a run page sums a run up.

    Attributes:
        headline (str): The one-line verdict.
        css (str): Its style class ("banner", "warn", "ok" or "").
        detail (str): The paragraphs under it, as HTML.
    """
    headline: str
    css: str
    detail: str


@dataclass(frozen=True)
class _LastRun:
    """
    What one finished real-money run tells about the account, for the attention warning.

    Attributes:
        finished (datetime): When it ended, timezone-aware; the newest run decides.
        verdict (str): "clean" (it finished trading with nothing left to
            check), "attention" (a pair or sale needs a person) or "unclean" (it ended
            without a clean result, so orders may have been placed).
        why (str): How it ended, in words, e.g. "exit 20" or "it wrote no result".
        where (str): Which run, as plain text; a scheduled run, which has no
            page, also names the log its result is in.
        where_html (str): The same, escaped, linking to its page when it has one.
    """
    finished: datetime
    verdict: str
    why: str
    where: str
    where_html: str


def _response_headers(response: _Response) -> list[tuple[str, str]]:
    """
    List the headers a response is sent with (Content-Length aside).

    Every response carries the same six: its type, no caching, no type
    sniffing, no framing, the same-origin referrer policy (under which a
    browser still sends a POST's real Origin) and the Content-Security-
    Policy. A redirect adds its Location.

    Args:
        response (_Response): The response.

    Returns:
        list[tuple[str, str]]: (name, value) pairs, in sending order.
    """
    headers = [
        ("Content-Type", response.content_type),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "same-origin"),
        ("Content-Security-Policy", _CSP),
    ]
    if response.location is not None:
        headers.append(("Location", response.location))
    return headers


def _log_safe(text: str) -> str:
    """
    Make text safe to write on one log line.

    Every character that is not printable (a newline, an escape character, a
    text-direction mark) is written as its escape instead, so a crafted
    request cannot add lines to the log or hide what it says.

    Args:
        text (str): The text.

    Returns:
        str: The text with each non-printable character as \\xNN, \\uNNNN or
            \\UNNNNNNNN.
    """
    out = []
    for ch in text:
        if ch.isprintable():
            out.append(ch)
        elif ord(ch) < 0x100:
            out.append(f"\\x{ord(ch):02x}")
        elif ord(ch) < 0x10000:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return "".join(out)


def _params(text: str) -> dict[str, list[str]]:
    """
    Parse a query string or form body into its fields, strictly.

    Browsers percent-encode every byte of a query or form that is not plain
    ASCII, so any other character is refused, as are a field without "=",
    percent-encoded bytes that are not UTF-8 and more than _MAX_FIELDS
    fields. A field given with nothing after "=" is kept, as "".

    Args:
        text (str): The query string (after "?") or the form body.

    Returns:
        dict[str, list[str]]: Each field name with its values in order; {}
            for "".

    Raises:
        ValueError: If the text breaks one of the rules above (a
            UnicodeDecodeError is a ValueError).
    """
    if not text.isascii():
        raise ValueError("the request holds characters that are not percent-encoded")
    return parse_qs(text, keep_blank_values=True, strict_parsing=True,
                    max_num_fields=_MAX_FIELDS, errors="strict")


def _number(text: str, name: str) -> float:
    """
    Read a field as a plain, finite decimal number.

    Args:
        text (str): The field's value.
        name (str): The field's name, for the message.

    Returns:
        float: The number (a negative zero reads as 0.0).

    Raises:
        ValueError: If the value is not written as plain ASCII digits (with an
            optional leading minus, decimal point and exponent), is infinite,
            or is a non-zero number too small to hold (it would read as 0).
    """
    if not _NUMBER.fullmatch(text):
        raise ValueError(f"{name} must be a plain decimal number, got {text!r}")
    value = float(text)
    mantissa = re.split("[eE]", text)[0]
    if not math.isfinite(value) or (value == 0.0 and any(c in "123456789" for c in mantissa)):
        raise ValueError(f"{name} is out of range, got {text!r}")
    return value + 0.0


def _name(text: str, name: str) -> str:
    """
    Read a category or tag field as one name.

    Args:
        text (str): The field's value.
        name (str): The field's name, for the message.

    Returns:
        str: The name as given.

    Raises:
        ValueError: If the name is not one printable line (a zero-width or
            text-direction character, say) or has spaces around it.
    """
    if not text.isprintable() or text != text.strip():
        raise ValueError(f"{name} must be one printable name with no spaces around it, "
                         f"got {text!r}")
    return text


def _whole_number(text: str, name: str) -> int:
    """
    Read a field as a plain whole number.

    Args:
        text (str): The field's value.
        name (str): The field's name, for the message.

    Returns:
        int: The number.

    Raises:
        ValueError: If the value is not one to six ASCII digits.
    """
    if not _WHOLE_NUMBER.fullmatch(text):
        raise ValueError(f"{name} must be a whole number of at most six digits, got {text!r}")
    return int(text)


def _kept(name: str, current: LiveSettings | None, source: str) -> object:
    """
    Return the value of a setting a request left out.

    Args:
        name (str): The LiveSettings field's name.
        current (LiveSettings | None): The live defaults in force, or None if none are saved.
        source (str): The request's source note ("" when it has none).

    Returns:
        object: The saved value, or the seed's when none is saved or source is the seed's note.
    """
    if current is not None and source != LIVE_DEFAULTS_SEED_SOURCE:
        return getattr(current, name)
    return getattr(LIVE_DEFAULTS_SEED, name)


def _proposal(params: dict[str, list[str]],
              current: LiveSettings | None) -> tuple[LiveSettings, str]:
    """
    Turn a confirmation request's fields into the proposed live defaults.

    tier_floors ("on" / "off"), spread_min, spread_max, k and size_cap (a
    fraction, e.g. 0.2) are required. same_title_size_cap,
    add_to_held_pairs ("on" / "off"), sell_at ("off" or a share in (0, 1])
    and sell_min_days ("off" or a whole number of days) may be left out:
    each then keeps the saved value, or the seed's when none is saved or
    (all but the same-title cap) on a link carrying the seed's note, so such a link still
    proposes exactly the seed, and the page shows the change against what is
    saved. A missing category or tag means any, whatever is saved; a tag needs
    its category. source is the note the saved file will keep: left out, it
    is empty; given, it must be one of the two shapes
    config.LIVE_DEFAULTS_SOURCE_PATTERN allows, with ASCII digits only, and
    the seed's note (LIVE_DEFAULTS_SEED_SOURCE) may label only the seed
    values themselves.

    Args:
        params (dict[str, list[str]]): The request's fields (_params).
        current (LiveSettings | None): The live defaults in force, or None
            when none are saved.

    Returns:
        tuple[LiveSettings, str]: The proposed defaults and the source note.

    Raises:
        ValueError: Naming the first rule the request breaks: an unknown,
            repeated, blank or missing field, a value that is not a plain
            number or a printable name, a tier_floors or add_to_held_pairs
            other than on or off, a sell_at neither off nor a number, a sell_min_days
            neither off nor one to six digits, a tag without a category, a source
            of another shape, any value LiveSettings refuses, or the seed's
            note on other values.
    """
    unknown = sorted(set(params) - set(_FIELDS))
    if unknown:
        raise ValueError(f"unknown field {unknown[0]!r}")
    for field_name, values in params.items():
        if len(values) != 1:
            raise ValueError(f"{field_name} is given {len(values)} times")
        if values[0] == "":
            raise ValueError(f"{field_name} is blank")
    value = {field_name: values[0] for field_name, values in params.items()}
    missing = [field_name for field_name in _REQUIRED if field_name not in value]
    if missing:
        raise ValueError(f"{missing[0]} is missing")
    if value["tier_floors"] not in ("on", "off"):
        raise ValueError(f"tier_floors must be on or off, got {value['tier_floors']!r}")
    if "same_title_size_cap" in value:
        same_title = _number(value["same_title_size_cap"], "same_title_size_cap")
    elif current is not None:
        same_title = current.same_title_size_cap
    else:
        same_title = LIVE_DEFAULTS_SEED.same_title_size_cap
    source = value.get("source", "")
    if "add_to_held_pairs" in value:
        if value["add_to_held_pairs"] not in ("on", "off"):
            raise ValueError("add_to_held_pairs must be on or off, got "
                             f"{value['add_to_held_pairs']!r}")
        add_on = value["add_to_held_pairs"] == "on"
    else:
        add_on = _kept("add_to_held_pairs", current, source)
    # "off" means never selling, or no minimum of days
    if "sell_at" in value:
        sell_at = None if value["sell_at"] == "off" else _number(value["sell_at"], "sell_at")
    else:
        sell_at = _kept("sell_at", current, source)
    if "sell_min_days" in value:
        sell_min_days = (None if value["sell_min_days"] == "off"
                         else _whole_number(value["sell_min_days"], "sell_min_days"))
    else:
        sell_min_days = _kept("sell_min_days", current, source)
    if "tag" in value and "category" not in value:
        raise ValueError("a tag needs its category")
    categories = (_name(value["category"], "category"),) if "category" in value else None
    tags = (_name(value["tag"], "tag"),) if "tag" in value else None
    # re.ASCII: the pattern's digits are 0-9 only, never a digit of another
    # script (which could show a date out of order on the page)
    if source and not re.fullmatch(LIVE_DEFAULTS_SOURCE_PATTERN, source, re.ASCII):
        raise ValueError("source must be the backtest dashboard's note or the seed's, "
                         f"got {source!r}")
    # The note's own rules (one printable line, not too long), as config's save
    # applies them
    if live_defaults_source(source) != source:
        raise ValueError(f"source must have no spaces around it, got {source!r}")
    # LiveSettings validates every value (the band, k, each cap on its grid,
    # the names), naming the field it refuses
    settings = LiveSettings(
        tier_floors=value["tier_floors"] == "on",
        spread_band=(_number(value["spread_min"], "spread_min"),
                     _number(value["spread_max"], "spread_max")),
        interval_discount=_number(value["k"], "k"),
        size_cap=_number(value["size_cap"], "size_cap"),
        same_title_size_cap=same_title,
        categories=categories,
        tags=tags,
        add_to_held_pairs=add_on,
        sell_at=sell_at,
        sell_min_days=sell_min_days,
    )
    # The seed's note names the seed values, so it may label nothing else
    if source == LIVE_DEFAULTS_SEED_SOURCE and settings != LIVE_DEFAULTS_SEED:
        raise ValueError("the seed values' note may label only the seed values "
                         "(config.LIVE_DEFAULTS_SEED)")
    return settings, source


def _seed_query() -> str:
    """
    Build the confirmation page's query that proposes LIVE_DEFAULTS_SEED.

    Each number is written as its repr (the exact float), adding to held
    pairs as on or off, the source note is LIVE_DEFAULTS_SEED_SOURCE, and a
    category or tag is added only when the seed sets one (it sets none: any).
    The sell level and the minimum of days are written as off when unset.

    Returns:
        str: The query string, without the "?".
    """
    seed = LIVE_DEFAULTS_SEED
    pairs = [
        ("tier_floors", "on" if seed.tier_floors else "off"),
        ("spread_min", repr(seed.spread_band[0])),
        ("spread_max", repr(seed.spread_band[1])),
        ("k", repr(seed.interval_discount)),
        ("size_cap", repr(seed.size_cap)),
        ("same_title_size_cap", repr(seed.same_title_size_cap)),
        ("add_to_held_pairs", "on" if seed.add_to_held_pairs else "off"),
        ("sell_at", "off" if seed.sell_at is None else repr(seed.sell_at)),
        ("sell_min_days", "off" if seed.sell_min_days is None else str(seed.sell_min_days)),
    ]
    pairs += [("category", name) for name in seed.categories or ()]
    pairs += [("tag", name) for name in seed.tags or ()]
    pairs.append(("source", LIVE_DEFAULTS_SEED_SOURCE))
    return urlencode(pairs)


def _current_defaults() -> LiveSettings | None:
    """
    Read the live defaults in force: the server's one read of the saved file.

    Returns:
        LiveSettings | None: The saved defaults, or None when none are saved.

    Raises:
        LiveDefaultsError: If a file exists but is refused.
    """
    # config's strict read, the one a live run makes through live_defaults(),
    # but returning None rather than raising when no file is saved
    return read_saved_live_defaults()


def _fingerprint(current: LiveSettings | None) -> str:
    """
    Fingerprint the live defaults in force, so a page built on them can tell they changed.

    The origin names the second the file was saved and its source note, so
    a later save changes the fingerprint even when it writes the same
    values, unless it also has the same note and falls in the same second
    (and then the page still describes the defaults in force).

    Args:
        current (LiveSettings | None): The saved defaults, or None when none are saved.

    Returns:
        str: The SHA-256 hex digest of "none", or of the origin and the ten
            toggles as JSON.
    """
    if current is None:
        text = "none"
    else:
        text = json.dumps({"origin": current.origin,
                           **{name: getattr(current, name) for name in LIVE_TOGGLE_FIELDS}},
                          sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signed_text(purpose: str, fingerprint: str, nonce: str,
                 params: dict[str, list[str]]) -> str:
    """
    Build the text a page's token signs.

    The page's purpose ("confirm" or "trade", so one page's token cannot
    start the other page's action), the fingerprint of the defaults in force,
    the page's nonce, each on its own line, then every proposal field as the
    request gave it, sorted and URL-encoded (none for /trade). The raw strings
    are signed, never values re-rendered from LiveSettings, so the POST must
    repeat exactly what the page showed.

    Args:
        purpose (str): "confirm" or "trade".
        fingerprint (str): _fingerprint of the defaults the page was built on.
        nonce (str): The page's one-time nonce.
        params (dict[str, list[str]]): The request's fields; any other than
            _FIELDS is not signed.

    Returns:
        str: The text to sign.
    """
    pairs = sorted((name, value) for name in _FIELDS for value in params.get(name, ()))
    return "\n".join((purpose, fingerprint, nonce, urlencode(pairs)))


def _page(title: str, body: str, *, script: bool = False, refresh: int | None = None) -> str:
    """
    Wrap a page's body in the HTML document every page shares.

    Args:
        title (str): The page title and heading (escaped here).
        body (str): The body's HTML, every dynamic string in it already escaped.
        script (bool): Whether to add the page script (the pages with buttons).
        refresh (int | None): Reload the page after this many seconds (a run
            still going); None (default) never.

    Returns:
        str: The whole HTML document.
    """
    escaped = html.escape(title)
    head = (f"<meta http-equiv=\"refresh\" content=\"{int(refresh)}\">\n"
            if refresh is not None else "")
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            + head
            + f"<title>{escaped}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
            f"<h1>{escaped}</h1>\n{body}\n"
            + (f"<script>{_CONFIRM_JS}</script>\n" if script else "")
            + "</body>\n</html>\n")


def _where_html(*, saving: bool, trading: bool = False) -> str:
    """
    Say, as HTML, which file holds the live defaults, which runs read it, and where a run starts.

    Args:
        saving (bool): Whether the page offers a save; it then says the save
            writes the file.
        trading (bool): Whether the page can start a run; it then names the
            checkout the run's code, trade log and log come from and the
            machine's live-run lock.

    Returns:
        str: One or two paragraphs naming config.LIVE_DEFAULTS_FILE and
            config.PROJECT_ROOT (and config.LIVE_RUN_LOCK_FILE), read at call time.
    """
    path = html.escape(str(config.LIVE_DEFAULTS_FILE.absolute()))
    root = html.escape(str(config.PROJECT_ROOT.absolute()))
    if saving:
        text = (f"<p>This writes <code>{path}</code>; every live run of the code in "
                f"<code>{root}</code> reads it (the weekly scheduler's too, when it runs "
                "from this checkout).</p>")
    else:
        text = (f"<p>Every live run of the code in <code>{root}</code> reads its defaults "
                f"from <code>{path}</code> (the weekly scheduler's too, when it runs from "
                "this checkout).</p>")
    if trading:
        lock = html.escape(str(config.LIVE_RUN_LOCK_FILE.absolute()))
        text += (f"\n<p>A run started here is <code>python -m kalshi_betting.main --mode "
                 f"prod</code> from <code>{root}</code>: that checkout's code, its trade log "
                 f"and <code>kalshi_arb.log</code>. A real-money run holds the machine's "
                 f"live-run lock, <code>{lock}</code>, while it runs.</p>")
    return text


def _notes_html(*, confirming: bool) -> str:
    """
    List, as HTML, what a save does and does not change.

    Args:
        confirming (bool): Whether the page has buttons; only then does it say
            that nothing happens until one is clicked.

    Returns:
        str: A list of notes.
    """
    path = html.escape(config.LIVE_DEFAULTS_FILE.name)
    notes = (
        *(("Nothing is saved or traded until you click one of the buttons below; closing "
           "this tab cancels.",) if confirming else ()),
        "A live run already under way keeps the settings it started with; the next "
        "run reads the defaults saved by then.",
        "main.py's toggle flags still override any of these for one run.",
        f"{path} is the only source of live defaults: deleting it stops every live run "
        "until defaults are saved again.",
        "The backtest dashboard's figures for a category or tag are that slice's share "
        "of a run over every category, not a run over that category alone.",
    )
    return ("<ul class=\"note\">" + "".join(f"<li>{note}</li>" for note in notes) + "</ul>")


def _slot_html(slot: _Slot) -> str:
    """
    Render one button in its place: the real button, or a disabled placeholder and its reason.

    A real button is a submit button rendered disabled (the page script arms
    it), with its id and its action as the name "action". A placeholder has
    the same label but no id, name or value, is never armed, and is followed
    by the reason it does not apply.

    Args:
        slot (_Slot): The button.

    Returns:
        str: The button's HTML.
    """
    label = html.escape(slot.label)
    if slot.reason is None:
        return (f"<button type=\"submit\" name=\"action\" value=\"{html.escape(slot.action)}\" "
                f"id=\"{html.escape(slot.button_id)}\" disabled>{label}</button>")
    return (f"<button type=\"button\" disabled>{label}</button> "
            f"<span class=\"note\">{slot.reason}</span>")


def _form_html(target: str, params: dict[str, list[str]], controls: _Controls,
               money_line: str) -> str:
    """
    Build a page's one form: the hidden fields, then the buttons in their fixed order.

    The buttons before the last come first; after a visible gap, the red line
    saying the last button places real orders, the acknowledgement box when
    the attention banner shows and that button applies, and the last button.
    So Tab, or a habitual click, never lands on the real-money button because
    another button went missing.

    Args:
        target (str): Where the form posts ("/confirm" or "/trade").
        params (dict[str, list[str]]): The proposal fields to carry, as the
            request gave them (none for /trade).
        controls (_Controls): The signed fields and the buttons.
        money_line (str): The red line above the real-money button (plain text).

    Returns:
        str: The form's HTML.
    """
    hidden = [
        f"<input type=\"hidden\" name=\"{html.escape(name)}\" value=\"{html.escape(value)}\">"
        for name in _FIELDS for value in params.get(name, ())
    ]
    hidden += [f"<input type=\"hidden\" name=\"fingerprint\" value=\"{controls.fingerprint}\">",
               f"<input type=\"hidden\" name=\"nonce\" value=\"{controls.nonce}\">",
               f"<input type=\"hidden\" name=\"token\" value=\"{controls.token}\">"]
    *first, last = controls.slots
    parts = [f"<form method=\"post\" action=\"{target}\">", "".join(hidden),
             "<p>" + " ".join(_slot_html(slot) for slot in first) + "</p>",
             "<div class=\"gap\">",
             f"<p class=\"money\">{html.escape(money_line)}</p>"]
    if controls.attention is not None and last.reason is None:
        parts.append("<p><label><input type=\"checkbox\" name=\"ack_attention\" value=\"1\"> "
                     f"{html.escape(_ACK_LABEL)}</label></p>")
    parts += [f"<p>{_slot_html(last)}</p>", "</div>", "</form>"]
    return "\n".join(parts)


def _buttons_notes_html() -> str:
    """
    Say, as HTML, what Dry run does and when the buttons can be clicked.

    Returns:
        str: Two short notes.
    """
    return (f"<p class=\"note\">{html.escape(_DRY_RUN_NOTE)}</p>\n"
            f"<p class=\"note\">{html.escape(_ARMING_NOTE)}</p>")


def _settings_table(settings: LiveSettings) -> str:
    """
    Show one set of live settings as a Setting | Value table.

    Args:
        settings (LiveSettings): The settings.

    Returns:
        str: The table, one row per toggle in the "Live settings:" line's words.
    """
    # Each toggle's value in the "Live settings:" line's words (compared with
    # nothing, so only the new column is read)
    rows = live_settings_changes(None, settings)
    table = ["<table><tr><th>Setting</th><th>Value</th></tr>"]
    for label, _, value, _ in rows:
        table.append(f"<tr data-setting=\"{html.escape(label)}\"><td>{html.escape(label)}</td>"
                     f"<td>{html.escape(value)}</td></tr>")
    table.append("</table>")
    return "".join(table)


def _confirm_html(current: LiveSettings | None, settings: LiveSettings, source: str,
                  params: dict[str, list[str]], controls: _Controls,
                  banner: str | None) -> str:
    """
    Build the confirmation page: the defaults in force beside the proposed ones.

    Every row whose value changes is highlighted, with the new value in bold
    and a "changed" tag; any live_rule_warnings sentence is shown in red.
    The form posts back every proposal field as the request gave it, with the
    fingerprint, nonce and token, through three buttons rendered disabled
    (the page script arms them): Dry run, Confirm and save, and, set apart
    below a red line, Confirm and trade. A button that does not apply is a
    placeholder with its reason.

    Args:
        current (LiveSettings | None): The live defaults in force, or None when none are saved.
        settings (LiveSettings): The proposed defaults.
        source (str): The proposed source note ("" for none).
        params (dict[str, list[str]]): The request's fields, carried into the form as given.
        controls (_Controls): The signed fields, the buttons and any attention warning.
        banner (str | None): A notice to show above everything else, or None.

    Returns:
        str: The HTML page.
    """
    # Each toggle in the "Live settings:" line's words, flagged when it changes
    rows = live_settings_changes(current, settings)
    changes = sum(1 for *_, changed in rows if changed)
    title = ("Overwrite the live trading defaults?" if current is not None
             else "Save the first live trading defaults?")
    parts = []
    if banner is not None:
        parts.append(f"<p class=\"banner\">{html.escape(banner)}</p>")
    if controls.attention is not None:
        parts.append(f"<p class=\"banner\">{controls.attention.html}</p>")
    parts.append(_where_html(saving=bool(changes), trading=True))
    in_force = (html.escape(current.origin) if current is not None
                else "none saved — live runs refuse to start until defaults are saved")
    parts.append(f"<p>Live defaults in force: {in_force}</p>")
    parts.append("<p>New defaults from: "
                 + (html.escape(source) if source else "(no source note)") + "</p>")
    table = ["<table><tr><th>Setting</th><th>Current</th><th>New</th><th></th></tr>"]
    for label, old, new, changed in rows:
        new_cell = f"<b>{html.escape(new)}</b>" if changed else html.escape(new)
        tag = "<span class=\"tag\">changed</span>" if changed else ""
        table.append(f"<tr class=\"{'changed' if changed else 'same'}\" "
                     f"data-setting=\"{html.escape(label)}\"><td>{html.escape(label)}</td>"
                     f"<td>{html.escape(old)}</td><td>{new_cell}</td><td>{tag}</td></tr>")
    table.append("</table>")
    parts.append("".join(table))
    parts.append(f"<p>{changes} of {len(rows)} settings change.</p>")
    # The same warnings a live run on these settings would log
    for warning in live_rule_warnings(settings):
        parts.append(f"<p class=\"warn\">Warning: {html.escape(warning)}.</p>")
    if not changes:
        parts.append("<p>These are already the live defaults — nothing to save.</p>")
    parts.append(_notes_html(confirming=True))
    parts.append(_form_html(
        "/confirm", params, controls,
        "Confirm and trade saves these settings as the live defaults, then runs the live "
        "bot with them on the production account: it places real orders."))
    parts.append(_buttons_notes_html())
    return _page(title, "\n".join(parts), script=True)


def _trade_html(current: LiveSettings, controls: _Controls, banner: str | None) -> str:
    """
    Build the trade page: the saved live defaults, and the buttons that run the bot with them.

    Args:
        current (LiveSettings): The saved live defaults.
        controls (_Controls): The signed fields, the buttons and any attention warning.
        banner (str | None): A notice to show above everything else, or None.

    Returns:
        str: The HTML page.
    """
    settings = current
    parts = []
    if banner is not None:
        parts.append(f"<p class=\"banner\">{html.escape(banner)}</p>")
    if controls.attention is not None:
        parts.append(f"<p class=\"banner\">{controls.attention.html}</p>")
    parts.append(_where_html(saving=False, trading=True))
    parts.append(f"<p>Saved defaults: {html.escape(settings.origin)}</p>")
    parts.append(_settings_table(settings))
    # The same warnings a live run on these settings logs
    for warning in live_rule_warnings(settings):
        parts.append(f"<p class=\"warn\">Warning: {html.escape(warning)}.</p>")
    parts.append(_form_html(
        "/trade", {}, controls,
        "Confirm and trade runs the live bot with these saved defaults on the production "
        "account: it places real orders."))
    parts.append(_buttons_notes_html())
    return _page("Trade with the saved live defaults?", "\n".join(parts), script=True)


def _saved_html(settings: LiveSettings) -> str:
    """
    Build the page shown after a save: the live defaults now in force.

    Args:
        settings (LiveSettings): The saved defaults, as read back from the file.

    Returns:
        str: The HTML page.
    """
    parts = [
        f"<p class=\"ok\">Saved and verified: {html.escape(settings.origin)}. The file was "
        "read back and every setting matched what was confirmed.</p>",
        _settings_table(settings),
    ]
    # What a live run on these settings will log as a warning
    for warning in live_rule_warnings(settings):
        parts.append(f"<p class=\"warn\">Saved (valid), but note: {html.escape(warning)}.</p>")
    parts += [
        "<p><a href=\"/trade\">Trade with these defaults →</a></p>",
        _where_html(saving=False),
        _notes_html(confirming=False),
        "<p>You can close this tab.</p>",
    ]
    return _page("Live trading defaults saved", "\n".join(parts))


def _message_html(status: int, title: str, text: str) -> str:
    """
    Build a page that reports one outcome: a refusal, an error or "nothing to save".

    Args:
        status (int): The HTTP status the page is sent with.
        title (str): The heading.
        text (str): The explanation (escaped here).

    Returns:
        str: The HTML page.
    """
    phrase = HTTPStatus(status).phrase
    body = (f"<p>{html.escape(text)}</p>\n"
            f"<p class=\"note\">HTTP {status} {html.escape(phrase)}</p>")
    return _page(title, body)


def _write_json(path: Path, record: dict) -> None:
    """
    Write a JSON record to a file whole: to a staging file beside it, then renamed over it.

    Args:
        path (Path): The file.
        record (dict): The record.

    Raises:
        OSError: When the staging file cannot be written or renamed (it is
            removed first).
        TypeError, ValueError: When the record cannot be written as JSON.
    """
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(json.dumps(record, indent=1), encoding="utf-8")
        os.replace(staging, path)
    except BaseException:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parse_utc(text) -> datetime | None:
    """
    Read a UTC time as a run's folder and its result record it.

    Args:
        text: The recorded value.

    Returns:
        datetime | None: The time, timezone-aware; None when it is not text
            in _TIME's format.
    """
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, _TIME).replace(tzinfo=UTC)
    except ValueError:
        return None


def _read_json_file(path: Path):
    """
    Read a JSON file; any failure to read or parse it is a None.

    Args:
        path (Path): The file.

    Returns:
        The parsed value, or None when the file is missing or cannot be read
            or parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def _finite(value) -> float | None:
    """
    Keep a recorded amount only if it is a finite number.

    Args:
        value: The recorded value.

    Returns:
        float | None: The number, or None when it is not a finite int or
            float (a bool is not a number here).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _text(value) -> str | None:
    """
    Keep a recorded value only if it is text.

    Args:
        value: The recorded value.

    Returns:
        str | None: The value, or None when it is not a str.
    """
    return value if isinstance(value, str) else None


def _int(value) -> int | None:
    """
    Keep a recorded value only if it is an int (a bool is not).

    Args:
        value: The recorded value.

    Returns:
        int | None: The value, or None.
    """
    return value if type(value) is int else None


def _leg(value) -> dict | None:
    """
    Read one leg of a pair in a run result, each field checked for its type.

    Args:
        value: The recorded leg.

    Returns:
        dict | None: ticker, market, side (str | None), count (int | None)
            and price (float | None); None when the leg is not an object.
    """
    if not isinstance(value, dict):
        return None
    return {"ticker": _text(value.get("ticker")), "market": _text(value.get("market")),
            "side": _text(value.get("side")), "count": _int(value.get("count")),
            "price": _finite(value.get("price"))}


def _sale_leg(value) -> dict | None:
    """
    Read one held market of a sale in a run result, each field checked for its type.

    Args:
        value: The recorded market.

    Returns:
        dict | None: ticker, side, held, sold, price (None if unreadable); None if not an object.
    """
    if not isinstance(value, dict):
        return None
    return {"ticker": _text(value.get("ticker")), "side": _text(value.get("side")),
            "held": _int(value.get("held")), "sold": _int(value.get("sold")),
            "price": _finite(value.get("price"))}


def _sale(value) -> dict | None:
    """
    Read one sale in a run result, each field checked for its type.

    Args:
        value: The recorded sale.

    Returns:
        dict | None: title, status ("unknown" if unreadable), error, profit and
            legs (the readable ones, each _sale_leg); None if not an object.
    """
    if not isinstance(value, dict):
        return None
    legs = value.get("legs") if isinstance(value.get("legs"), list) else []
    return {"title": _text(value.get("title")),
            "status": _text(value.get("status")) or "unknown",
            "error": _text(value.get("error")),
            "profit": _finite(value.get("profit")),
            "legs": [leg for leg in (_sale_leg(item) for item in legs) if leg is not None]}


def _read_result(folder: Path) -> _Result:
    """
    Read a run's result file strictly: the record's type-checked fields, or why there are none.

    The file must be a JSON object in config.LIVE_RUN_RESULT_FORMAT, with a
    bool dry_run and an int or null exit_code; otherwise it is "unreadable".
    Every other field is kept only when it has its type (a pair's fields
    too), so a hand-edited or damaged file never breaks a page.
    A sale's fields are read the same way; a result with no list of sales has none.

    Args:
        folder (Path): The run's folder.

    Returns:
        _Result: state "missing" (no file), "unreadable", or "ok" with the record.
    """
    path = folder / _RESULT_NAME
    try:
        present = path.exists()
    except OSError:
        present = True
    if not present:
        return _Result("missing", {})
    raw = _read_json_file(path)
    if (not isinstance(raw, dict) or raw.get("format") != LIVE_RUN_RESULT_FORMAT
            or not isinstance(raw.get("dry_run"), bool)
            or not (raw.get("exit_code") is None or type(raw.get("exit_code")) is int)):
        return _Result("unreadable", {})
    trades = []
    listed = raw.get("trades") if isinstance(raw.get("trades"), list) else []
    for trade in listed:
        if not isinstance(trade, dict):
            continue
        trades.append({"status": _text(trade.get("status")) or "unknown",
                       "error": _text(trade.get("error")),
                       "pair_type": _text(trade.get("pair_type")),
                       "a": _leg(trade.get("a")), "b": _leg(trade.get("b")),
                       "cost_with_fees": _finite(trade.get("cost_with_fees")),
                       "profit_if_won": _finite(trade.get("profit_if_won")),
                       "adds_to_held": _finite(trade.get("adds_to_held"))})
    listed = raw.get("sales") if isinstance(raw.get("sales"), list) else []
    sales = [sale for sale in (_sale(item) for item in listed) if sale is not None]
    warnings = raw.get("warnings") if isinstance(raw.get("warnings"), list) else []
    dropped = _int(raw.get("warnings_dropped"))
    return _Result("ok", {
        "dry_run": raw["dry_run"],
        "exit_code": raw["exit_code"],
        "finished_at": _parse_utc(raw.get("finished_at")),
        "message": _text(raw.get("message")) or "",
        "error": _text(raw.get("error")),
        "submission_started": raw.get("submission_started") is True,
        "balance_before": _finite(raw.get("balance_before")),
        "balance_after": _finite(raw.get("balance_after")),
        "portfolio_value_before": _finite(raw.get("portfolio_value_before")),
        "trades": trades,
        "sales": sales,
        "cash_after_sales": _finite(raw.get("cash_after_sales")),
        "warnings": [line for line in warnings if isinstance(line, str)],
        "warnings_dropped": dropped if dropped is not None and dropped > 0 else 0,
    })


def _run_alive(folder: Path) -> bool:
    """
    Tell whether a run is still going, from the lock its process holds on its output.log.

    The server locks output.log before it starts the run, and the run's
    process inherits that lock through its output, so the lock is held
    exactly as long as the process lives, however it ends, and across a
    restart of the server.

    Args:
        folder (Path): The run's folder.

    Returns:
        bool: True when output.log is locked by another open of it (the run's
            process); False when it is free, missing or cannot be opened.
    """
    try:
        fd = os.open(folder / _OUTPUT_NAME, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)  # closed once on every path, which also drops the shared lock


def _log_tail(folder: Path) -> list[str]:
    """
    Read the last lines of a run's output.log, reading only the end of the file.

    Args:
        folder (Path): The run's folder.

    Returns:
        list[str]: At most DEFAULTS_SERVER_RUN_LOG_TAIL_LINES lines, from the
            last DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES bytes (a first line cut by
            that limit is left out); [] when the file cannot be read.
    """
    try:
        with (folder / _OUTPUT_NAME).open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES)
            handle.seek(start)
            data = handle.read(DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES)
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return lines[-DEFAULTS_SERVER_RUN_LOG_TAIL_LINES:]


def _error_line(lines: list[str]) -> str | None:
    """
    Find the last "…: error: …" line a run printed (its argument parser's refusal).

    Args:
        lines (list[str]): The end of the run's output.

    Returns:
        str | None: The line, or None when there is none.
    """
    for line in reversed(lines):
        if ": error: " in line:
            return line
    return None


def _refused_to_start(lines: list[str]) -> bool:
    """
    Tell from its output whether a run stopped at its argument parser (exit 2, before logging).

    The parser prints its "usage:" line, then "<prog>: error: <why>" as the
    last thing the run prints; a run that logged anything after would not end
    on that line.

    Args:
        lines (list[str]): The end of the run's output.

    Returns:
        bool: True when a line starts with "usage: " and the last non-empty
            line is an error line.
    """
    printed = [line for line in lines if line.strip()]
    return (bool(printed) and ": error: " in printed[-1]
            and any(line.startswith("usage: ") for line in printed))


def _runs_on_disk() -> list[Path]:
    """
    List the run folders under config.LIVE_RUNS_DIR, newest first.

    Returns:
        list[Path]: Every folder named like a run's (its UTC start time, then
            its id), sorted by name, newest first; [] when the folder is
            missing or cannot be read.
    """
    try:
        folders = [p for p in config.LIVE_RUNS_DIR.iterdir()
                   if _RUN_FOLDER.fullmatch(p.name) and p.is_dir()]
    except OSError:
        return []
    return sorted(folders, key=lambda p: p.name, reverse=True)


def _load_run(folder: Path) -> _Run:
    """
    Describe a run from its folder alone (one another server started, or this one before a restart).

    Its run.json says what it is; a run.json that cannot be read makes it a
    real-money run of unknown settings, so its page errs toward checking
    positions.

    Args:
        folder (Path): The run's folder.

    Returns:
        _Run: The run, with no process.
    """
    record = _read_json_file(folder / _RUN_NAME)
    record = record if isinstance(record, dict) else {}
    match = _RUN_FOLDER.fullmatch(folder.name)
    started = _parse_utc(record.get("started_at"))
    if started is None and match is not None:
        # The folder's name starts with the run's UTC start time
        try:
            started = datetime.strptime(folder.name[:16], _FOLDER_TIME).replace(tzinfo=UTC)
        except ValueError:
            started = None
    return _Run(
        run_id=match.group(1) if match else folder.name,
        folder=folder, process=None,
        dry_run=record.get("dry_run") is True,
        settings_text=_text(record.get("settings")) or "not recorded",
        started_at=started,
        saved_note=_text(record.get("saved_note")) or "",
        pid=_int(record.get("pid")),
        start_error=_text(record.get("start_error")),
    )


def _output_time(folder: Path) -> datetime | None:
    """
    Tell when a run last printed anything: its output.log's modification time.

    Args:
        folder (Path): The run's folder.

    Returns:
        datetime | None: The time, in UTC; None when the file cannot be read.
    """
    try:
        return datetime.fromtimestamp((folder / _OUTPUT_NAME).stat().st_mtime, UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _exit_verdict(code: int | None) -> tuple[str, str] | None:
    """
    Say what a finished real-money run's exit code tells about the account.

    Args:
        code (int | None): The exit code; None when none was recorded (a run
            stopped at a time limit, one that could not be started, or a
            damaged record).

    Returns:
        tuple[str, str] | None: (verdict, how it ended in words): "attention"
            for EXIT_TRADES_NEED_ATTENTION, "clean" for _CLEAN_EXITS, and
            "unclean" for anything else (no code, a signal, an error, a code
            no run of main.py returns); None for _NO_ORDER_EXITS, a run that
            stopped before it could send an order.
    """
    if code in _NO_ORDER_EXITS:
        return None
    if code == EXIT_TRADES_NEED_ATTENTION:
        return "attention", f"exit {code}"
    if code in _CLEAN_EXITS:
        return "clean", f"exit {code}"
    if code is None:
        return "unclean", "no exit code was recorded"
    if code < 0:
        return "unclean", f"stopped by a signal (exit {code})"
    return "unclean", f"exit {code}"


def _folder_last_run(folder: Path, process: subprocess.Popen | None) -> _LastRun | None:
    """
    Read what one run folder tells about the account, for the attention warning.

    It follows the run page's own summary (_outcome). Left out: a dry run, a
    run still going, one that could not be started, one its argument parser
    refused, and one that stopped before it could send an order
    (_NO_ORDER_EXITS, or an error before its first order). Of the rest, a pair
    (or a sale) left for a person, or exit EXIT_TRADES_NEED_ATTENTION, is "attention"; a
    run with no readable result, stopped by a signal, stopped by an error
    while or after sending orders, or ended with any other code but a clean
    one, is "unclean"; a clean exit (_CLEAN_EXITS) is "clean".

    Args:
        folder (Path): The run's folder (named like a run's).
        process (subprocess.Popen | None): Its process, when this server
            started it; its exit code wins over the result's, as on the run page.

    Returns:
        _LastRun | None: What it tells, or None when it is left out (or has
            no time to place it among the others).
    """
    code = None if process is None else process.poll()
    if process is not None and code is None:
        return None  # still going
    result = _read_result(folder)
    record = result.record
    run = _load_run(folder)
    if run.start_error is not None:
        return None  # nothing ran
    if result.state == "ok":
        if record["dry_run"]:
            return None
        if process is None:
            code = record["exit_code"]
    elif run.dry_run:
        return None
    elif process is None and result.state == "missing" and _run_alive(folder):
        return None  # still going: its process still holds output.log
    trades = record.get("trades", [])
    refused = (code == _USAGE_EXIT
               or (process is None and code is None and _refused_to_start(_log_tail(folder))))
    if result.state == "missing" and refused:
        return None  # its argument parser refused it before it logged anything
    sale_attention = any(s["status"] in _SALE_ATTENTION_STATUSES
                         for s in record.get("sales", []))
    if (any(t["status"] in _ATTENTION_STATUSES for t in trades) or sale_attention
            or code == EXIT_TRADES_NEED_ATTENTION):
        verdict = ("attention", f"exit {code}" if code == EXIT_TRADES_NEED_ATTENTION
                   else "a sale was left for a person to check" if sale_attention
                   else "a pair was left for a person to check")
    elif result.state != "ok":
        if code in _NO_ORDER_EXITS:
            return None  # its exit code says it stopped before it could send an order
        why = ("its result could not be read" if result.state == "unreadable"
               else "it wrote no result")
        verdict = ("unclean", why if code is None else f"{why}, exit {code}")
    elif code is not None and code < 0:
        verdict = _exit_verdict(code)
    elif record["submission_started"] and not trades and record["error"] is not None:
        verdict = ("unclean", "it stopped while sending orders")
    elif code in (None, _ERROR_EXIT) and not record["submission_started"]:
        return None  # an error before its first order: nothing was sent
    elif code is None:
        verdict = ("unclean", "an error stopped it after it began sending orders")
    else:
        verdict = _exit_verdict(code)
        if verdict is None:
            return None  # it stopped before it could send an order
    finished = record.get("finished_at") or _output_time(folder) or run.started_at
    if finished is None:
        return None
    when = f"{finished.astimezone(UTC):%Y-%m-%d %H:%M} UTC"
    run_id = run.run_id
    return _LastRun(finished, verdict[0], verdict[1], f"run {run_id} of {when}",
                    f"<a href=\"/runs/{html.escape(run_id)}\">run {html.escape(run_id)}</a> "
                    f"of {html.escape(when)}")


def _scheduled_last_run() -> _LastRun | None:
    """
    Read what the scheduler's record of its last run tells about the account.

    The scheduler writes config.PROJECT_ROOT / SCHEDULER_STATE_FILENAME in
    naive local time. A run it has not finished, one whose finish time cannot
    be read, and one that stopped before it could send an order
    (_NO_ORDER_EXITS) are left out; a finished run with no readable exit code
    (stopped at the scheduler's time limit, not started, or a damaged record)
    is "unclean".
    The run is named with its local finish time and kalshi_arb.log, where
    its result is, since it writes no result file and has no run page.

    Returns:
        _LastRun | None: What it tells, or None when it is left out.
    """
    state = _read_json_file(config.PROJECT_ROOT / SCHEDULER_STATE_FILENAME)
    if not isinstance(state, dict) or not isinstance(state.get("finished_at"), str):
        return None
    try:
        finished = datetime.fromisoformat(state["finished_at"]).astimezone()
    except (ValueError, OverflowError, OSError):
        return None
    code = _int(state.get("exit_code"))
    verdict = _exit_verdict(code)
    if verdict is None:
        return None
    why = verdict[1] if code is not None else (
        "the scheduler recorded no exit code: it was stopped at its time limit, could not "
        "be started, or its record is damaged")
    # A scheduled run writes no result file and has no run page; its result,
    # a disproof CRITICAL included, is in the log main.py writes
    when = f"the scheduled run of {finished:%Y-%m-%d %H:%M}, logged in kalshi_arb.log"
    return _LastRun(finished, verdict[0], why, when, html.escape(when))


def _duration_text(seconds: float) -> str:
    """
    Say a duration in whole hours, minutes and seconds.

    Args:
        seconds (float): The duration.

    Returns:
        str: e.g. "1 h 02 min", "3 min 07 s" or "12 s".
    """
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _hang_limit_text() -> str:
    """
    Say how long a run may take before a page calls it possibly hung.

    Returns:
        str: SCHEDULER_JOB_TIMEOUT_SECONDS in words, e.g. "1 h 00 min".
    """
    return _duration_text(SCHEDULER_JOB_TIMEOUT_SECONDS)


def _money(value: float | None) -> str:
    """
    Say an amount in dollars.

    Args:
        value (float | None): The amount; None when not known.

    Returns:
        str: e.g. "$12.30", or "—".
    """
    return "—" if value is None else f"${value:,.2f}"


def _leg_text(leg: dict | None) -> str:
    """
    Say what one leg of a pair bought, as "30 × YES @ 0.2100".

    Args:
        leg (dict | None): The leg (_leg).

    Returns:
        str: The count, side and price, "?" for any not recorded; "—" for no leg.
    """
    if leg is None:
        return "—"
    count = "?" if leg["count"] is None else str(leg["count"])
    side = "?" if leg["side"] is None else leg["side"].upper()
    price = "?" if leg["price"] is None else f"{leg['price']:.4f}"
    return f"{count} × {side} @ {price}"


def _market_text(leg: dict | None) -> str:
    """
    Name one leg's market, with its ticker.

    Args:
        leg (dict | None): The leg (_leg).

    Returns:
        str: "<market> (<ticker>)", whichever of the two is recorded, or "—".
    """
    if leg is None:
        return "—"
    names = [name for name in (leg["market"], leg["ticker"] and f"({leg['ticker']})") if name]
    return " ".join(names) or "—"


def _trades_html(trades: list[dict]) -> str:
    """
    Show a run's pairs as tables: needs attention, completed, would have traded, not completed.

    A pair's Note is the trader's error, if any, followed by "(adds to N
    held)" for a trade that added to a pair the account already held.

    Args:
        trades (list[dict]): The run result's pairs (_read_result).

    Returns:
        str: The tables' HTML, one per group that has a pair; "" when there are none.
    """
    grouped: list[tuple[str, list[dict]]] = []
    placed = set()
    for label, statuses in _TRADE_GROUPS:
        rows = [t for t in trades if t["status"] in statuses]
        placed |= {id(t) for t in rows}
        grouped.append((label, rows))
    grouped.append(("Not completed", [t for t in trades if id(t) not in placed]))
    parts = []
    head = ("<tr><th>Status</th><th>Type</th><th>Market A</th><th>Leg A</th>"
            "<th>Market B</th><th>Leg B</th><th>Cost incl. fees</th><th>Profit if won</th>"
            "<th>Note</th></tr>")
    for label, rows in grouped:
        if not rows:
            continue
        body = []
        for t in rows:
            # A trade that added to a held pair says how much was held there
            note = " ".join(part for part in (
                t["error"] or "",
                f"(adds to {count_text(t['adds_to_held'])} held)"
                if (t["adds_to_held"] or 0) > 0 else "",
            ) if part)
            cells = (t["status"], t["pair_type"] or "—", _market_text(t["a"]),
                     _leg_text(t["a"]), _market_text(t["b"]), _leg_text(t["b"]),
                     _money(t["cost_with_fees"]), _money(t["profit_if_won"]), note)
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        parts.append(f"<h2>{html.escape(label)}</h2>\n<table>{head}{''.join(body)}</table>")
    return "\n".join(parts)


def _sold_text(leg: dict) -> str:
    """
    Say what one held market of a sale sold, as "30 of 30 YES on KX-A".

    Args:
        leg (dict): The market (_sale_leg).

    Returns:
        str: That text, with "?" for anything not recorded.
    """
    sold = "?" if leg["sold"] is None else str(leg["sold"])
    held = "?" if leg["held"] is None else str(leg["held"])
    side = "?" if leg["side"] is None else leg["side"].upper()
    return f"{sold} of {held} {side} on {leg['ticker'] or '?'}"


def _sales_html(sales: list[dict], cash_after: float | None, *, dry_run: bool) -> str:
    """
    Show a run's sales (the take-profit rule) as one short table, with the cash they left.

    Args:
        sales (list[dict]): The run result's sales (_sale).
        cash_after (float | None): The cash after the sales, in dollars, or None.
        dry_run (bool): Keyword-only. A dry run's cash after the sales is an estimate.

    Returns:
        str: The table's HTML; "" when there are no sales.
    """
    if not sales:
        return ""
    head = "<tr><th>Position</th><th>Status</th><th>Sold</th><th>Profit</th></tr>"
    body = []
    for sale in sales:
        status = _SALE_STATUS_WORDS.get(sale["status"], sale["status"])
        sold = ", ".join(_sold_text(leg) for leg in sale["legs"]) or "—"
        if sale["error"]:
            sold += f" ({sale['error']})"
        profit = (_money(sale["profit"]) if sale["status"] in ("sold", "simulated")
                  else "—")
        cells = (sale["title"] or "—", status, sold, profit)
        body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    parts = [f"<h2>Sales</h2>\n<table>{head}{''.join(body)}</table>",
             f"<p class=\"note\">{html.escape(_SALE_PROFIT_NOTE)}</p>"]
    if cash_after is not None:
        parts.append(f"<p>Cash after the sales {_money(cash_after)}"
                     + (" (estimated: nothing was sent)" if dry_run else "") + "</p>")
    return "\n".join(parts)


def _outcome(run: _Run, exit_code: int | None, result: _Result, lines: list[str]) -> _Outcome:
    """
    Sum up a run that has ended: the first rule of the run page's table that applies.

    Args:
        run (_Run): The run.
        exit_code (int | None): Its exit code: the one its process returned
            (negative for a signal), or its result's for a run read from disk;
            None when not known.
        result (_Result): Its result file, as read.
        lines (list[str]): The end of its output.

    Returns:
        _Outcome: The headline, its style and the paragraphs under it.
    """
    record = result.record
    trades = record.get("trades", [])
    executed = [t for t in trades if t["status"] == "executed"]
    error_line = _error_line(lines)
    real_money = not run.dry_run
    if run.start_error is not None:
        return _Outcome("The run could not be started", "warn",
                        f"<p>{html.escape(run.start_error)}. Nothing ran and no order was "
                        "sent.</p>")
    # 0: a usage error (exit 2) writes no result: the run never started
    refused = (exit_code == _USAGE_EXIT or (exit_code is None and run.process is None
                                  and _refused_to_start(lines)))
    if result.state == "missing" and refused:
        return _Outcome("The run refused to start", "warn",
                        f"<p>{html.escape(error_line or 'No reason was printed.')}</p>"
                        "<p>Nothing was sent.</p>")
    # 1: a pair or a sale needs a person
    if (any(t["status"] in _ATTENTION_STATUSES for t in trades)
            or any(s["status"] in _SALE_ATTENTION_STATUSES for s in record.get("sales", []))
            or exit_code == EXIT_TRADES_NEED_ATTENTION):
        return _Outcome("Trades or sales need your attention", "banner",
                        "<p>Check these positions in the Kalshi UI. Do not start another "
                        "real-money run — from this page, the dashboard, a terminal or the "
                        "scheduler — until this is understood.</p>"
                        + (f"<p>{html.escape(record['message'])}</p>"
                           if record.get("message") else ""))
    # 2: no readable result (a kill signal, a crash before logging), or a
    # real-money run stopped while sending orders
    stopped_sending = (result.state == "ok" and real_money and record["submission_started"]
                       and not trades and record["error"] is not None)
    if result.state != "ok" or (exit_code is not None and exit_code < 0) or stopped_sending:
        if stopped_sending:
            headline = "The run stopped while sending orders"
            detail = f"<p>{html.escape(record['error'])}</p>"
        else:
            headline = ("The run's result could not be read" if result.state == "unreadable"
                        else "The run ended without writing a result")
            detail = (f"<p>Exit code: {'not recorded' if exit_code is None else exit_code}"
                      + (" (stopped by a signal)" if exit_code is not None and exit_code < 0
                         else "") + ".</p>")
            if error_line:
                detail += f"<p>{html.escape(error_line)}</p>"
        detail += (f"<p class=\"money\">{html.escape(_CHECK_POSITIONS)}</p>" if real_money
                   else "<p>It was a dry run: no orders were sent.</p>")
        return _Outcome(headline, "banner", detail)
    message = (f"<p>{html.escape(record['message'])}</p>" if record["message"] else "")
    # 3: orders went out, then the run failed
    if executed and exit_code not in (EXIT_OK, EXIT_TIME_SERIES_SKIPPED):
        return _Outcome("Orders were placed, then the run stopped with an error", "warn",
                        f"<p>{html.escape(record['error'] or 'No error was recorded.')}</p>")
    if exit_code in (EXIT_OK, EXIT_TIME_SERIES_SKIPPED):
        note = ("<p>No time-series pair was searched: a held market could not be "
                "identified.</p>" if exit_code == EXIT_TIME_SERIES_SKIPPED else "")
        # 7: nothing to trade
        if not trades:
            sales = record.get("sales", [])
            if any(s["status"] in ("sold", "partly_sold") for s in sales):
                return _Outcome("Positions sold; no trades to complete", "ok",
                                message + note)
            if sales and all(s["status"] == "simulated" for s in sales):
                return _Outcome("Dry run finished — no orders were sent", "",
                                "<p>These positions would have been sold:</p>"
                                + message + note)
            return _Outcome("No trades to complete", "", message + note)
        # 4, 5, 6 (and 8, the same with exit 40's note)
        if executed:
            return _Outcome("Trades completed", "ok",
                            f"<p>{len(executed)} of {len(trades)} pairs placed.</p>"
                            + message + note)
        if all(t["status"] == "simulated" for t in trades):
            return _Outcome("Dry run finished — no orders were sent", "",
                            "<p>These trades would have been placed:</p>" + message + note)
        return _Outcome("No trade completed", "warn",
                        f"<p>{html.escape(_EACH_PAIR_BELOW)}</p>" + message + note)
    # 9: stopped before trading, for a reason of its own
    reasons = {
        EXIT_SKIPPED_LOW_BALANCE: "the portfolio value is below the minimum",
        EXIT_NO_TRADEABLE_SHARDS: ("nothing could be scanned (every exchange shard was "
                                   "closed, or no market was read)"),
        EXIT_RUN_IN_PROGRESS: "another live trading run was in progress",
    }
    if exit_code in reasons:
        return _Outcome(f"Not traded: {reasons[exit_code]}", "", message)
    # 10: its parser refused it, though it wrote a result
    if exit_code == _USAGE_EXIT:
        return _Outcome("The run refused to start", "warn",
                        f"<p>{html.escape(error_line or 'No reason was printed.')}</p>")
    # 11: anything else
    error = record["error"] or "No error was recorded; the end of its output is below."
    return _Outcome("The run stopped with an error", "warn",
                    f"<p>{html.escape(error)}</p>" + message)


def _run_html(run: _Run) -> str:
    """
    Build a run's page: its progress while it runs, its outcome once it has ended.

    Args:
        run (_Run): The run.

    Returns:
        str: The HTML page; one still running reloads itself every
            DEFAULTS_SERVER_RUN_REFRESH_SECONDS.
    """
    result = _read_result(run.folder)
    if run.process is not None:
        exit_code = run.process.poll()
        running = exit_code is None and run.start_error is None
    else:
        exit_code = result.record.get("exit_code")
        running = (result.state == "missing" and run.start_error is None
                   and _run_alive(run.folder))
    lines = _log_tail(run.folder)
    kind = "a dry run: no orders" if run.dry_run else "real orders on the production account"
    started = (f"{run.started_at:%Y-%m-%d %H:%M:%S} UTC" if run.started_at is not None
               else "not recorded")
    about = [f"<p>Run <code>{html.escape(run.run_id)}</code> — {html.escape(kind)}. "
             f"Started {html.escape(started)}"
             + (f", process {run.pid}" if run.pid is not None else "") + ".</p>",
             f"<p>Settings: {html.escape(run.settings_text)}</p>"]
    if run.saved_note:
        about.append(f"<p>{html.escape(run.saved_note)}</p>")
    tail = ("<h2>End of its output</h2>\n<pre>" + html.escape("\n".join(lines)) + "</pre>"
            if lines else "<p class=\"note\">It has printed nothing yet.</p>")
    if running:
        headline = ("Dry run in progress — no orders" if run.dry_run
                    else "Trading run in progress — real orders on the production account")
        css = "" if run.dry_run else " class=\"money\""
        parts = [f"<p{css}>{html.escape(headline)}</p>", *about]
        if run.started_at is not None:
            elapsed = (datetime.now(UTC) - run.started_at).total_seconds()
            parts.append(f"<p>Running for {html.escape(_duration_text(elapsed))}.</p>")
            if elapsed > SCHEDULER_JOB_TIMEOUT_SECONDS:
                who = f"process {run.pid}" if run.pid is not None else "it"
                parts.append(f"<p class=\"warn\">It has been running for over "
                             f"{html.escape(_hang_limit_text())} — it may be hung; check "
                             f"Kalshi before stopping {html.escape(who)}.</p>")
        restarted = (" It was started before this server restarted."
                     if run.process is None else "")
        parts.append("<p class=\"note\">Closing this tab or stopping the server does not "
                     "stop the run. Keep this Mac awake until it finishes." + restarted
                     + "</p>")
        parts.append(tail)
        return _page("Live trading run", "\n".join(parts),
                     refresh=DEFAULTS_SERVER_RUN_REFRESH_SECONDS)
    outcome = _outcome(run, exit_code, result, lines)
    css = f" class=\"{outcome.css}\"" if outcome.css else ""
    parts = [f"<p{css}><b>{html.escape(outcome.headline)}</b></p>", outcome.detail, *about]
    record = result.record
    if record.get("finished_at") is not None:
        parts.append(f"<p>Finished {record['finished_at']:%Y-%m-%d %H:%M:%S} UTC, exit code "
                     f"{'not recorded' if exit_code is None else exit_code}.</p>")
    if record.get("balance_before") is not None and record.get("balance_after") is not None:
        parts.append(f"<p>Balance before {_money(record['balance_before'])} → after "
                     f"{_money(record['balance_after'])}</p>")
    if record.get("portfolio_value_before") is not None:
        # Cash plus open positions, read before trading: what the run sizes on
        # (worded so it also fits a run that stopped at the minimum)
        # A run that sold sizes on what its sales left, so this is the value before them
        parts.append(f"<p>Portfolio value {_money(record['portfolio_value_before'])} "
                     f"(cash {_money(record.get('balance_before'))}) — "
                     + ("before the sales" if record.get("sales")
                        else "what Kelly sizes on") + "</p>")
    # The sales come first, as the run made them before it bought anything
    sales = _sales_html(record.get("sales", []), record.get("cash_after_sales"),
                        dry_run=run.dry_run)
    if sales:
        parts.append(sales)
    trades = _trades_html(record.get("trades", []))
    if trades:
        parts.append(trades)
    warnings = record.get("warnings", [])
    if warnings or record.get("warnings_dropped"):
        parts.append("<h2>Warnings</h2>\n<ul>"
                     + "".join(f"<li>{html.escape(line)}</li>" for line in warnings) + "</ul>")
        if record.get("warnings_dropped"):
            parts.append(f"<p class=\"note\">{record['warnings_dropped']} more warning lines "
                         "were not kept.</p>")
    parts.append(tail)
    return _page("Live trading run", "\n".join(parts))


def _dashboard_state(dashboard: Path) -> tuple[str, str | None]:
    """
    Tell whether the backtest dashboard exists and carries the filter bar's Save and Trade buttons.

    The buttons sit in the page's opening part, so only the first
    DASHBOARD_MARKER_SCAN_BYTES are read, looking for the Trade link's id
    (_DASHBOARD_MARKER). A page built before the buttons existed lacks it.

    Args:
        dashboard (Path): The dashboard file.

    Returns:
        tuple[str, str | None]: The state — _DASHBOARD_MISSING,
            _DASHBOARD_UNREADABLE, _DASHBOARD_OLD or _DASHBOARD_READY — and,
            for an unreadable file only, why it could not be read (else None).
    """
    if not dashboard.exists():
        return _DASHBOARD_MISSING, None
    try:
        with dashboard.open("rb") as page:
            head = page.read(DASHBOARD_MARKER_SCAN_BYTES)
    except OSError as exc:
        return _DASHBOARD_UNREADABLE, f"{type(exc).__name__}: {exc}"
    return (_DASHBOARD_READY if _DASHBOARD_MARKER in head else _DASHBOARD_OLD), None


def _index_html() -> str:
    """
    Build the page at "/": what this server is for and the newest runs, with no form and no script.

    Returns:
        str: The HTML page.
    """
    dashboard = config.PROJECT_ROOT / DASHBOARD_FILENAME
    state, why = _dashboard_state(dashboard)
    shown = html.escape(str(dashboard.absolute()))
    if state == _DASHBOARD_READY:
        intro = ("<p>This server saves the live trading defaults and runs the live bot with "
                 f"them. Open <code>{shown}</code> and use its filter bar's Save as live "
                 "defaults… button to save a scenario, or its Trade using defaults… button to "
                 "trade the saved ones.</p>")
    elif state == _DASHBOARD_MISSING:
        intro = ("<p>No backtest dashboard yet: run <code>python3 -m kalshi_betting.backtest"
                 "</code> to build one.</p>")
    elif state == _DASHBOARD_OLD:
        intro = (f"<p>The backtest dashboard at <code>{shown}</code> was built before its "
                 "Save as live defaults… and Trade using defaults… buttons: rebuild it with "
                 "<code>python3 -m kalshi_betting.backtest</code> and the "
                 "<code>--start-date</code> you built it from. Until then, use the links "
                 "below.</p>")
    else:
        intro = (f"<p>The backtest dashboard at <code>{shown}</code> could not be read "
                 f"({html.escape(why or 'no reason given')}). Use the links below.</p>")
    links = (f"<p><a href=\"{html.escape('/confirm?' + _seed_query())}\">Start from the seed "
             "values</a> · <a href=\"/trade\">Trade using defaults</a></p>")
    rows = []
    for folder in _runs_on_disk()[:DEFAULTS_SERVER_INDEX_RUNS]:
        run = _load_run(folder)
        result = _read_result(folder)
        if run.start_error is not None:
            state = "never started"
        elif result.state != "missing":
            code = result.record.get("exit_code")
            state = (f"finished (exit {code})" if result.state == "ok" and code is not None
                     else "finished")
        elif _run_alive(folder):
            state = "running"
        else:
            state = "ended without a result"
        started = (f"{run.started_at:%Y-%m-%d %H:%M:%S} UTC" if run.started_at is not None
                   else run.run_id)
        kind = "dry run" if run.dry_run else "real orders"
        rows.append(f"<li><a href=\"/runs/{html.escape(run.run_id)}\">{html.escape(started)}"
                    f"</a> — {kind} — {html.escape(state)}</li>")
    runs = ("<h2>Runs started here</h2>\n<ul>" + "".join(rows) + "</ul>" if rows
            else "<p class=\"note\">No run has been started from this checkout's server.</p>")
    return _page("Live trading defaults",
                 "\n".join((intro, links, runs, _where_html(saving=False, trading=True))))


class _App:
    """
    The server's routes and checks, apart from any socket.

    One instance lives as long as its server, with a random key made at
    start; a page's token is valid only for the instance that built it, and
    each page's nonce works once. It remembers the runs it started and what
    each used nonce produced.
    """

    def __init__(self, port: int, key: bytes | None = None, *,
                 start_process: Callable[..., subprocess.Popen] | None = None) -> None:
        """
        Set up the routes for a server on a port.

        Args:
            port (int): The port the server listens on; a request's Host and a
                POST's Origin must name it.
            key (bytes | None): The token key; None (default) makes a random one.
            start_process (Callable | None): What starts a run's process, called
                like subprocess.Popen. Only main() passes one (Popen itself);
                with None (default) the server refuses every run, so nothing
                built without it can start main.py.
        """
        self.port = port
        self._key = secrets.token_bytes(32) if key is None else key
        self._hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        self._origins = frozenset(f"http://{host}" for host in self._hosts)
        self._start_process = start_process
        self._runs: dict[str, _Run] = {}
        self._used_nonces: dict[str, _NonceUse] = {}

    def handle(self, request: _Request) -> _Response:
        """
        Answer one request.

        Every request must name this server in its Host header. GET "/" says
        what the server is for and lists the newest runs; GET "/confirm" shows
        the confirmation page and POST "/confirm" saves, trades or dry-runs;
        GET "/saved" shows the defaults in force; GET "/trade" shows the saved
        defaults and POST "/trade" trades or dry-runs them; GET "/runs/<id>"
        shows a run; GET "/checkout" answers which checkout this is and the
        fingerprint of the code this process loaded. Anything else is 404.

        Args:
            request (_Request): The request.

        Returns:
            _Response: The response.
        """
        if request.host is None or request.host.lower() not in self._hosts:
            return self._refuse(403, "Refused", "This server answers only requests addressed "
                                f"to 127.0.0.1:{self.port} or localhost:{self.port}.")
        path, _, query = request.target.partition("?")
        if request.method == "GET":
            if path == "/":
                return _Response(200, _index_html())
            if path == "/confirm":
                return self._get_confirm(query)
            if path == "/saved":
                return self._get_saved()
            if path == "/trade":
                return self._get_trade()
            if path == "/checkout":
                return _Response(200, json.dumps(
                    {"project_root": str(config.PROJECT_ROOT.resolve()), "code": _LOADED_CODE}),
                    content_type=_JSON_TYPE)
            if path.startswith("/runs/"):
                return self._get_run(path[len("/runs/"):])
        elif request.method == "POST":
            if path == "/confirm":
                return self._post_confirm(request)
            if path == "/trade":
                return self._post_trade(request)
        return self._refuse(404, "Not found", "This server has no such page.")

    def _token(self, purpose: str, fingerprint: str, nonce: str,
               params: dict[str, list[str]]) -> str:
        """
        Sign a page's purpose, the fingerprint it was built on, its nonce and its proposal fields.

        Args:
            purpose (str): "confirm" or "trade".
            fingerprint (str): _fingerprint of the defaults the page was built on.
            nonce (str): The page's one-time nonce.
            params (dict[str, list[str]]): The proposal's fields.

        Returns:
            str: The HMAC-SHA-256 of _signed_text, as 64 hex digits.
        """
        return hmac.new(self._key, _signed_text(purpose, fingerprint, nonce, params)
                        .encode("utf-8"), hashlib.sha256).hexdigest()

    def _refuse(self, status: int, title: str, text: str) -> _Response:
        """
        Log a refusal and build its page.

        Args:
            status (int): The HTTP status.
            title (str): The page's heading.
            text (str): Why, for the page and the log.

        Returns:
            _Response: The message page.
        """
        logging.warning("Refused (%d %s): %s", status, HTTPStatus(status).phrase,
                        _log_safe(text))
        return _Response(status, _message_html(status, title, text))

    def _refused_file(self, exc: LiveDefaultsError) -> _Response:
        """
        Refuse a request because the saved live defaults file cannot be used.

        Args:
            exc (LiveDefaultsError): Why the file is refused (it names the file).

        Returns:
            _Response: A 409 page with no form.
        """
        return self._refuse(
            409, "The saved live defaults are refused",
            f"{exc}. Fix or delete {config.LIVE_DEFAULTS_FILE.absolute()} first; nothing "
            "can be saved over it or traded from here.")

    def _own_run(self) -> _Run | None:
        """
        Find a run started from this checkout's server that is still going.

        A run this server started is still going while its process is. A run
        known only from its folder (started before this server restarted) is
        still going when its page would say so: no result yet, no start
        error, and its output.log still locked by its process (_run_alive).

        Returns:
            _Run | None: The newest such run, or None.
        """
        alive = [run for run in self._runs.values()
                 if run.process is not None and run.process.poll() is None]
        if alive:
            return alive[-1]
        for folder in _runs_on_disk():
            if _RUN_FOLDER.fullmatch(folder.name).group(1) in self._runs:
                continue  # its process, checked above, is the better answer
            try:
                finished = (folder / _RESULT_NAME).exists()
            except OSError:
                finished = True
            if finished or not _run_alive(folder):
                continue
            run = _load_run(folder)
            if run.start_error is None:
                return run
        return None

    def _run_blocker(self, *, dry_run: bool) -> _Notice | None:
        """
        Say why a run cannot start now, or None when it can.

        In order: a server built without a process starter runs nothing; a
        run started from this checkout's server that is still going (_own_run,
        which also finds one started before this server restarted) blocks
        both kinds of run; and, for a real-money run only, another live
        trading run holding the machine's lock (run_lock.held, a momentary
        check) blocks it, with a warning once that run has held the lock for
        over SCHEDULER_JOB_TIMEOUT_SECONDS.

        Args:
            dry_run (bool): Whether the run would be a dry run.

        Returns:
            _Notice | None: The reason, or None.
        """
        if self._start_process is None:
            return _Notice(_NOT_STARTED_TO_TRADE, html.escape(_NOT_STARTED_TO_TRADE))
        own = self._own_run()
        if own is not None:
            link = f"/runs/{own.run_id}"
            return _Notice(f"A run started from this page is still going ({link})",
                           "A run started from this page is still going "
                           f"(<a href=\"{html.escape(link)}\">{html.escape(link)}</a>)")
        # The machine-wide lock a real-money run holds (dry runs never take it)
        if not dry_run and run_lock.held():
            # Who holds it, as its record says: for the message only
            holder = run_lock.holder()
            text = f"Another live trading run is in progress ({holder.describe()})"
            page = html.escape(text)
            age = holder.age_seconds()
            if age is not None and age > SCHEDULER_JOB_TIMEOUT_SECONDS:
                who = f"process {holder.pid}" if holder.pid is not None else "it"
                hung = (f"running for over {_hang_limit_text()} — it may be hung; check "
                        f"Kalshi before stopping {who}")
                text += f" — {hung}"
                page += f" — <span class=\"money\">{html.escape(hung)}</span>"
            return _Notice(text, page)
        return None

    def _attention(self) -> _Notice | None:
        """
        Warn when the last real-money run left something for a person to check; never raises.

        Every finished real-money run of this checkout is read: each run
        folder under config.LIVE_RUNS_DIR (_folder_last_run, which uses the
        exit code this server collected for a run it started) and the
        scheduler's record of its last run (_scheduled_last_run). A dry run,
        a run still going and a run that stopped before it could send an
        order are left out, so they neither raise the warning nor clear it.
        Of the rest the newest decides: a clean exit clears it; a run that
        needed attention (exit EXIT_TRADES_NEED_ATTENTION) or that ended
        without a clean result (no result, a signal, an error after it began
        sending orders, a scheduled run with no exit code) raises it. It
        never raises: a record that cannot be read is left out, and anything
        else going wrong fails closed, with a warning that the outcome could
        not be read.

        Returns:
            _Notice | None: The warning, or None when the newest run that
                decides ended cleanly, or there is none.
        """
        try:
            runs = []
            for folder in _runs_on_disk():
                own = self._runs.get(_RUN_FOLDER.fullmatch(folder.name).group(1))
                last = _folder_last_run(folder, None if own is None else own.process)
                if last is not None:
                    runs.append(last)
            scheduled = _scheduled_last_run()
            if scheduled is not None:
                runs.append(scheduled)
            # Ties go to the first listed: the newest run folder, then the scheduler
            newest = max(runs, key=lambda run: run.finished, default=None)
        except Exception:
            # Fails closed: a warning that cannot be worked out still asks for the box
            logging.exception("Could not read the last real-money run's outcome")
            return _Notice(_OUTCOME_UNREADABLE, html.escape(_OUTCOME_UNREADABLE))
        if newest is None or newest.verdict == "clean":
            return None
        template = _ATTENTION_TAIL if newest.verdict == "attention" else _UNCLEAN_TAIL
        tail = template.format(why=newest.why)
        return _Notice(f"The last real-money run ({newest.where}){tail}",
                       f"The last real-money run ({newest.where_html}){html.escape(tail)}")

    def _controls(self, *, purpose: str, fingerprint: str, params: dict[str, list[str]],
                  current: LiveSettings | None, changes: bool) -> _Controls:
        """
        Build a page's form: a fresh nonce, its token and its buttons in their fixed order.

        Every render issues a fresh nonce. On /confirm the buttons are Dry run
        (a placeholder when a run is blocked or no defaults are saved), Confirm
        and save (a placeholder when nothing would change) and Confirm and
        trade (a placeholder when a real-money run is blocked); on /trade, Dry
        run and Confirm and trade.

        Args:
            purpose (str): "confirm" or "trade".
            fingerprint (str): _fingerprint of the defaults the page is built on.
            params (dict[str, list[str]]): The proposal fields ({} for /trade).
            current (LiveSettings | None): The live defaults in force.
            changes (bool): Whether saving the proposal would change anything.

        Returns:
            _Controls: The page's form.
        """
        nonce = secrets.token_hex(16)
        dry_block = self._run_blocker(dry_run=True)
        trade_block = self._run_blocker(dry_run=False)
        if dry_block is not None:
            dry_reason = dry_block.html
        elif current is None:
            dry_reason = html.escape(_SAVE_FIRST)
        else:
            dry_reason = None
        trade = _Slot("Confirm and trade", "confirm-trade", "trade",
                      None if trade_block is None else trade_block.html)
        if purpose == "confirm":
            slots = (_Slot("Dry run (no orders; defaults unchanged)", "confirm-dry-run",
                           "dry_run", dry_reason),
                     _Slot("Confirm and save", "confirm", "save",
                           None if changes else html.escape(_NOTHING_TO_SAVE)),
                     trade)
        else:
            slots = (_Slot("Dry run (no orders)", "confirm-dry-run", "dry_run", dry_reason),
                     trade)
        return _Controls(fingerprint, nonce, self._token(purpose, fingerprint, nonce, params),
                         slots, self._attention())

    def _render_confirm(self, current: LiveSettings | None, *, settings: LiveSettings,
                        source: str, params: dict[str, list[str]], banner: str | None) -> str:
        """
        Build the confirmation page for a proposal, against the defaults in force.

        Args:
            current (LiveSettings | None): The live defaults in force.
            settings (LiveSettings): The proposed defaults.
            source (str): The proposed source note.
            params (dict[str, list[str]]): The request's proposal fields.
            banner (str | None): A notice to show above everything else.

        Returns:
            str: The HTML page.
        """
        changes = any(changed for *_, changed in live_settings_changes(current, settings))
        controls = self._controls(purpose="confirm", fingerprint=_fingerprint(current),
                                  params=params, current=current, changes=changes)
        return _confirm_html(current, settings, source, params, controls, banner)

    def _render_trade(self, current: LiveSettings, banner: str | None) -> str:
        """
        Build the trade page for the saved live defaults.

        Args:
            current (LiveSettings): The saved live defaults.
            banner (str | None): A notice to show above everything else.

        Returns:
            str: The HTML page.
        """
        controls = self._controls(purpose="trade", fingerprint=_fingerprint(current), params={},
                                  current=current, changes=False)
        return _trade_html(current, controls, banner)

    def _get_confirm(self, query: str) -> _Response:
        """
        Show the confirmation page for the proposal in a query string.

        Args:
            query (str): The query string (after "?").

        Returns:
            _Response: 200 with the page, 400 when the proposal is refused,
                409 when the saved file is refused.
        """
        try:
            current = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        try:
            params = _params(query)
            settings, source = _proposal(params, current)
        except ValueError as exc:
            return self._refuse(400, "These settings cannot be saved", str(exc))
        return _Response(200, self._render_confirm(current, settings=settings, source=source,
                                                   params=params, banner=None))

    def _get_saved(self) -> _Response:
        """
        Show the live defaults in force (the page a save redirects to).

        Returns:
            _Response: 200 with the page, 404 when none are saved, 409 when
                the saved file is refused.
        """
        try:
            settings = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        if settings is None:
            return self._refuse(404, "No live defaults are saved",
                                "No live defaults are saved yet.")
        return _Response(200, _saved_html(settings))

    def _no_defaults(self, status: int) -> _Response:
        """
        Refuse the trade page because no live defaults are saved.

        Args:
            status (int): 404 for the page, 409 for a POST.

        Returns:
            _Response: The refusal, naming how to save defaults.
        """
        return self._refuse(
            status, "No live defaults are saved",
            "No live defaults are saved, so there is nothing to trade. Save them first: "
            "the backtest dashboard's Save as live defaults… button, or run "
            "./start_dashboard.sh --seed in this checkout for the seed values.")

    def _get_trade(self) -> _Response:
        """
        Show the trade page for the saved live defaults.

        Returns:
            _Response: 200 with the page, 404 when none are saved, 409 when
                the saved file is refused.
        """
        try:
            current = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        if current is None:
            return self._no_defaults(404)
        return _Response(200, self._render_trade(current, None))

    def _get_run(self, run_id: str) -> _Response:
        """
        Show one run's page, from memory or from its folder.

        Args:
            run_id (str): The id in the address.

        Returns:
            _Response: 200 with the page; 404 for an id that is not 16 hex
                digits or names no single run folder.
        """
        if not _RUN_ID.fullmatch(run_id):
            return self._refuse(404, "Not found", "This server has no such page.")
        run = self._runs.get(run_id)
        if run is None:
            found = [folder for folder in _runs_on_disk() if folder.name.endswith(f"-{run_id}")]
            if len(found) != 1:
                return self._refuse(404, "Not found", "This server knows no such run.")
            run = _load_run(found[0])
        return _Response(200, _run_html(run))

    def _read_form(self, request: _Request) -> dict[str, list[str]] | _Response:
        """
        Check a POST comes from this server's own page as a form, and read its fields.

        Args:
            request (_Request): The POST.

        Returns:
            dict[str, list[str]] | _Response: The fields, or the refusal: 403
                for another Origin, 400 for a body that is not a readable form.
        """
        if request.origin not in self._origins:
            return self._refuse(403, "Refused", "A save or a run must come from this server's "
                                f"own page; its Origin was {request.origin!r}.")
        media = (request.content_type or "").split(";", 1)[0].strip().lower()
        if media != _FORM_TYPE:
            return self._refuse(400, "Refused", f"A save or a run must be a posted form, not "
                                f"{request.content_type!r}.")
        try:
            return _params(request.body.decode("ascii"))
        except ValueError as exc:
            return self._refuse(400, "Refused", f"The form cannot be read: {exc}")

    def _signed_fields(self, form: dict[str, list[str]], purpose: str,
                       actions: tuple) -> tuple[str, str, str, bool] | _Response:
        """
        Take a form's signed fields and action out of it, check them, and check its token.

        Pops fingerprint, nonce, token, action and ack_attention, so what is
        left is the proposal. Each of the first three must be given once, as
        64, 32 and 64 hex digits (else 403); the action once, as one of
        actions (else 400); ack_attention at most once, as "1" (else 400).
        The trade page carries no proposal, so any other field on its form is
        400. Then the token must sign this purpose, fingerprint, nonce and
        the proposal for this server (else 403).

        Args:
            form (dict[str, list[str]]): The form's fields; changed in place.
            purpose (str): "confirm" or "trade" (whose form has no proposal).
            actions (tuple): The actions the page's buttons post.

        Returns:
            tuple[str, str, str, bool] | _Response: (fingerprint, nonce,
                action, whether the attention box was ticked), or the refusal.
        """
        fingerprints, nonces, tokens = (form.pop("fingerprint", []), form.pop("nonce", []),
                                        form.pop("token", []))
        chosen, acks = form.pop("action", []), form.pop("ack_attention", [])
        if not (len(fingerprints) == 1 and len(nonces) == 1 and len(tokens) == 1
                and _HEX64.fullmatch(fingerprints[0]) and _HEX32.fullmatch(nonces[0])
                and _HEX64.fullmatch(tokens[0])):
            return self._refuse(403, "Refused", "The form does not carry this server's "
                                "fingerprint, nonce and token.")
        if len(chosen) != 1 or chosen[0] not in actions:
            return self._refuse(400, "Refused", "The form must name one action of "
                                f"{', '.join(actions)}; it named {chosen!r}.")
        if acks not in ([], ["1"]):
            return self._refuse(400, "Refused", "The acknowledgement box's value cannot be read.")
        if purpose == "trade" and form:
            return self._refuse(400, "Refused", "The trade page's form carries no other "
                                f"field; this one carried {sorted(form)[0]!r}.")
        fingerprint, nonce, token = fingerprints[0], nonces[0], tokens[0]
        expected = self._token(purpose, fingerprint, nonce, form)
        if not hmac.compare_digest(token.encode("ascii"), expected.encode("ascii")):
            return self._refuse(
                403, "Refused", "This page was issued by another server session or for "
                "another page, or its settings were changed after it was shown — open it "
                "again from the dashboard (or --seed).")
        return fingerprint, nonce, chosen[0], bool(acks)

    def _used(self, nonce: str) -> _Response | None:
        """
        Answer a POST of a page whose button was already used, or None for a fresh page.

        Args:
            nonce (str): The page's nonce.

        Returns:
            _Response | None: A 303 to the page the first POST produced, or a
                409 saying what happened when it produced none; None when the
                nonce was never used.
        """
        use = self._used_nonces.get(nonce)
        if use is None:
            return None
        if use.path is not None:
            logging.info("A page's button was used again; sent to %s", use.path)
            return _Response(303, _message_html(303, "Already done", "This page's button was "
                                                f"already used; see {use.path}."),
                             location=use.path)
        return self._refuse(409, "Already used", f"This page's button was already used — "
                            f"{use.outcome}; open the page again from the dashboard.")

    def _post_confirm(self, request: _Request) -> _Response:
        """
        Save, trade or dry-run the proposal a confirmation page posted, after every check.

        In order: the Origin must be this server; the body must be a form;
        its fingerprint, nonce, token and action must be readable and the
        token must sign exactly these fields for this server's confirmation
        page; a nonce already used is sent to what it produced; the saved
        file must be usable; the proposal must be valid; the defaults in force
        must still be the ones the page was built on (else the page is shown
        again against them). Then, by action: "save" with nothing to change
        writes nothing; "trade" and "dry_run" are refused while a run is
        blocked, "dry_run" while no defaults are saved, and "trade" while the
        last real-money run needs attention and its box is not ticked — every
        one of these before anything is saved. Only then is the nonce marked
        used: "save", and "trade" with a change, save the proposal (the one
        write of the saved file); "save" is sent to /saved, and "trade" and
        "dry_run" start the run with the proposal's settings and are sent to
        its page.

        Args:
            request (_Request): The POST.

        Returns:
            _Response: 303 to /saved or /runs/<id>; 200 when there is nothing
                to save; 400, 403, 409 or 500 when refused or when the save or
                the start fails.
        """
        form = self._read_form(request)
        if isinstance(form, _Response):
            return form
        signed = self._signed_fields(form, "confirm", _CONFIRM_ACTIONS)
        if isinstance(signed, _Response):
            return signed
        posted_fingerprint, nonce, action, acked = signed
        used = self._used(nonce)
        if used is not None:
            return used
        try:
            current = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        try:
            settings, source = _proposal(form, current)
        except ValueError as exc:
            return self._refuse(400, "These settings cannot be saved", str(exc))
        if _fingerprint(current) != posted_fingerprint:
            logging.warning("Refused a POST from a page built on other live defaults; "
                            "showing it again against the defaults in force")
            return _Response(409, self._render_confirm(current, settings=settings, source=source,
                                                       params=form, banner=_STALE_BANNER))
        # Nothing is written when every toggle already has the proposed value
        changes = any(changed for *_, changed in live_settings_changes(current, settings))
        if action == "save" and not changes:
            return _Response(200, _message_html(
                200, "Nothing to save",
                "These are already the live defaults, so nothing was written."))
        dry_run = action == "dry_run"
        if action != "save":
            blocker = self._run_blocker(dry_run=dry_run)
            if blocker is not None:
                return self._refuse(409, "Not started",
                                    f"{blocker.text}. Nothing was saved or traded.")
            if dry_run and current is None:
                return self._refuse(409, "Save these first",
                                    "A dry run needs saved live defaults — save these first. "
                                    "Nothing was saved or traded.")
            if not dry_run and not acked and self._attention() is not None:
                logging.warning("Refused Confirm and trade: the last real-money run needs "
                                "attention and its box was not ticked")
                return _Response(409, self._render_confirm(
                    current, settings=settings, source=source, params=form,
                    banner=_ACK_BANNER))
        # Every check has passed: from here this page's button counts as used
        use = _NonceUse(None, "the request did not finish")
        self._used_nonces[nonce] = use
        saved = False
        if action == "save" or (action == "trade" and changes):
            try:
                # The one write of the saved file: config parses the text as a
                # live run will before writing it, then reads it back
                save_live_defaults(settings, source=source)
            except LiveDefaultsError as exc:
                # The message says what happened: refused before writing, a
                # failed write that left the old file, or a write whose flush
                # failed
                logging.error("Saving the live defaults failed: %s", _log_safe(str(exc)))
                use.outcome = "saving the live defaults failed, so nothing was traded"
                return _Response(500, _message_html(500, "Saving the live defaults failed",
                                                    f"{exc}. Nothing was traded."))
            saved = True
            # Marked "(default: X)" against the defaults this save replaced
            logging.info("Saved live defaults to %s: %s", config.LIVE_DEFAULTS_FILE,
                         _log_safe(describe_live_settings(settings, current or settings)))
        if action == "save":
            use.path = "/saved"
            return _Response(303, _message_html(303, "Saved", "The live defaults were saved; "
                                                "see /saved."), location="/saved")
        if saved:
            note = _SAVED_NOTE
        elif dry_run:
            note = _DRY_RUN_SAVED_NOTE
        else:
            note = _NOTHING_SAVED_NOTE
        return self._launch(use, settings=settings, dry_run=dry_run, saved_note=note,
                            saved=saved)

    def _post_trade(self, request: _Request) -> _Response:
        """
        Trade or dry-run the saved live defaults the trade page showed, after every check.

        In order: the Origin, the form, exactly the fingerprint, nonce, token
        and action (plus an optional acknowledgement) and nothing else, the
        token signed for the trade page, a nonce already used, the saved file
        usable and something saved, the same defaults still in force (else the
        page is shown again), then the run's blockers and, for "trade", the
        attention acknowledgement. Only then is the nonce marked used and the
        run started with the saved defaults.

        Args:
            request (_Request): The POST.

        Returns:
            _Response: 303 to /runs/<id>; 400, 403, 409 or 500 when refused or
                when the start fails.
        """
        form = self._read_form(request)
        if isinstance(form, _Response):
            return form
        signed = self._signed_fields(form, "trade", _TRADE_ACTIONS)
        if isinstance(signed, _Response):
            return signed
        posted_fingerprint, nonce, action, acked = signed
        used = self._used(nonce)
        if used is not None:
            return used
        try:
            current = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        if current is None:
            return self._no_defaults(409)
        if _fingerprint(current) != posted_fingerprint:
            logging.warning("Refused a trade from a page built on other live defaults; "
                            "showing it again against the defaults in force")
            return _Response(409, self._render_trade(current, _STALE_BANNER))
        dry_run = action == "dry_run"
        blocker = self._run_blocker(dry_run=dry_run)
        if blocker is not None:
            return self._refuse(409, "Not started", f"{blocker.text}. Nothing was traded.")
        if not dry_run and not acked and self._attention() is not None:
            logging.warning("Refused Confirm and trade: the last real-money run needs "
                            "attention and its box was not ticked")
            return _Response(409, self._render_trade(current, _ACK_BANNER))
        use = _NonceUse(None, "the request did not finish")
        self._used_nonces[nonce] = use
        settings = current
        return self._launch(use, settings=settings, dry_run=dry_run,
                            saved_note=f"The saved live defaults ({current.origin}); nothing "
                            "was saved.", saved=False)

    def _launch(self, use: _NonceUse, *, settings: LiveSettings, dry_run: bool,
                saved_note: str, saved: bool) -> _Response:
        """
        Start a run and answer the POST that asked for it.

        Args:
            use (_NonceUse): The page's nonce record, given the run's page on success.
            settings (LiveSettings): The run's settings.
            dry_run (bool): Whether it sends no orders.
            saved_note (str): What happened to the live defaults before it.
            saved (bool): Whether this POST saved new defaults.

        Returns:
            _Response: 303 to the run's page, or 500 when it could not be
                started (naming the saved defaults when they were saved).
        """
        try:
            run = self._start_run(settings=settings, dry_run=dry_run, saved_note=saved_note)
        except (OSError, ValueError) as exc:
            logging.error("The run could not be started: %s", _log_safe(str(exc)))
            use.outcome = ("the run could not be started"
                           + ("; the new defaults were saved" if saved else ""))
            return _Response(500, _message_html(
                500, "The run could not be started",
                f"The run could not be started: {exc}."
                + (" The new defaults were saved." if saved else "")))
        path = f"/runs/{run.run_id}"
        use.path = path
        return _Response(303, _message_html(303, "Started", f"The run started; see {path}."),
                         location=path)

    def _start_run(self, *, settings: LiveSettings, dry_run: bool, saved_note: str) -> _Run:
        """
        Start one live trading run: main.py in production, as its own process in a new session.

        The one place the server starts a process. It makes the run's folder
        under config.LIVE_RUNS_DIR, writes run.json (what the run is) before
        anything starts, locks output.log and starts
        `python -m kalshi_betting.main --mode prod` with all ten toggles as
        flags (config.live_settings_argv), --result-file in the folder and,
        for a dry run, --dry-run, from config.PROJECT_ROOT, its output going
        to output.log. The process inherits the lock through its output, so
        _run_alive can tell it is still going however long it runs, across a
        restart of the server. Its own session keeps Ctrl-C on this server, or
        closing its terminal, from reaching it.

        Args:
            settings (LiveSettings): The run's settings.
            dry_run (bool): Whether it sends no orders.
            saved_note (str): What happened to the live defaults before it.

        Returns:
            _Run: The run, remembered by this server.

        Raises:
            RuntimeError: When the server was built without a process starter.
            OSError, ValueError: When the folder or the first run.json
                cannot be made (nothing is left behind), or when the process
                cannot be started (run.json then records why, as start_error,
                best effort).
        """
        if self._start_process is None:
            raise RuntimeError(_NOT_STARTED_TO_TRADE)
        run_id = secrets.token_hex(8)
        started = datetime.now(UTC)
        folder = config.LIVE_RUNS_DIR / f"{started.strftime(_FOLDER_TIME)}-{run_id}"
        folder.mkdir(parents=True)
        # All ten toggles as flags, so the run trades exactly these settings
        argv = [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
                *live_settings_argv(settings), "--result-file", str(folder / _RESULT_NAME)]
        if dry_run:
            argv.append("--dry-run")
        # Named by its words in the "Live settings:" line, as the run logs them
        settings_text = describe_live_settings(settings)
        record = {"run_id": run_id, "dry_run": dry_run, "argv": argv,
                  "settings": settings_text, "started_at": started.strftime(_TIME),
                  "saved_note": saved_note}
        # What the run is, written before it starts: a failure here starts nothing
        try:
            _write_json(folder / _RUN_NAME, record)
        except BaseException:
            try:
                folder.rmdir()
            except OSError:
                pass
            raise
        try:
            with (folder / _OUTPUT_NAME).open("wb") as output:
                # The run inherits this lock through its output, so the lock marks
                # it alive until it ends, however it ends, even across a server restart
                fcntl.flock(output.fileno(), fcntl.LOCK_EX)
                # Its own session: Ctrl-C on this server, or closing its terminal,
                # never reaches a run that is placing orders
                process = self._start_process(
                    argv, cwd=config.PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            try:
                _write_json(folder / _RUN_NAME, {**record, "start_error": str(exc)})
            except Exception:
                logging.warning("Could not record in %s that the run did not start", folder)
            raise
        run = _Run(run_id=run_id, folder=folder, process=process, dry_run=dry_run,
                   settings_text=settings_text, started_at=started, saved_note=saved_note,
                   pid=process.pid)
        self._runs[run_id] = run
        try:
            _write_json(folder / _RUN_NAME, {**record, "pid": process.pid})
        except Exception as exc:
            # Only the pid is lost; the run is going and its page works without it
            logging.warning("Could not add the process id to %s: %s", folder / _RUN_NAME, exc)
        logging.info("Started a live trading run (%s), process %s: %s; its folder is %s",
                     "dry run" if dry_run else "real orders", process.pid,
                     _log_safe(settings_text), folder)
        return run


class _Handler(BaseHTTPRequestHandler):
    """
    The socket side: reads one request, hands it to the server's _App, sends the answer.

    A connection that sends nothing is dropped after `timeout` seconds, so
    one silent browser connection cannot hold the one-request-at-a-time
    server. A request over DEFAULTS_SERVER_MAX_REQUEST_BYTES (path plus
    body) is refused before its body is read. Methods other than GET and
    POST get 501, and a request the base class cannot read (a malformed or
    over-long request line, unreadable headers) its own 400, 414, 431 or
    505; send_error sends each as this server's refusal page, with the same
    headers as every other answer.
    """

    timeout = DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS
    server_version = "KalshiDefaultsServer"
    sys_version = ""

    def do_GET(self) -> None:
        """
        Answer a GET, refusing one whose path is over the size limit (413).

        Returns:
            None
        """
        if len(self.path) > DEFAULTS_SERVER_MAX_REQUEST_BYTES:
            self._send(self._too_large())
            return
        self._serve(b"")

    def do_POST(self) -> None:
        """
        Read a POST's body, within the size limit, and answer it.

        The Content-Length header must be given once, as plain digits: a
        missing one is 411, an unreadable one 400, and one that would take
        the request over DEFAULTS_SERVER_MAX_REQUEST_BYTES 413, all before
        any of the body is read. A body shorter than its stated length is 400.

        Returns:
            None
        """
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            self._send(_Response(411, _message_html(
                411, "Refused", "A save or a run must say how long its form is.")))
            return
        text = lengths[0].strip()
        if len(lengths) > 1 or not _DIGITS.fullmatch(text):
            self._send(_Response(400, _message_html(
                400, "Refused", "The form's length cannot be read.")))
            return
        # A length with more digits (leading zeros aside) than the limit is
        # over it, so int() is never asked to read one (it refuses a string
        # of more than a few thousand digits)
        significant = text.lstrip("0") or "0"
        if len(significant) > len(str(DEFAULTS_SERVER_MAX_REQUEST_BYTES)):
            self._send(self._too_large())
            return
        length = int(significant)
        if len(self.path) + length > DEFAULTS_SERVER_MAX_REQUEST_BYTES:
            self._send(self._too_large())
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._send(_Response(400, _message_html(
                400, "Refused", "The form ended before its stated length.")))
            return
        self._serve(body)

    def _too_large(self) -> _Response:
        """
        Build the 413 answer to a request over the size limit.

        Returns:
            _Response: The 413 page.
        """
        return _Response(413, _message_html(
            413, "Refused", f"A request may hold at most {DEFAULTS_SERVER_MAX_REQUEST_BYTES} "
            "bytes."))

    def _app(self) -> _App:
        """
        Return the server's _App, making one for its port on first use.

        One made here has no process starter, so it refuses every run; main()
        sets the server's own before serving.

        Returns:
            _App: The one _App of this server.
        """
        app = getattr(self.server, "defaults_app", None)
        if app is None:
            app = _App(self.server.server_address[1])
            self.server.defaults_app = app
        return app

    def _header(self, name: str) -> str | None:
        """
        Read a header given exactly once.

        Args:
            name (str): The header's name.

        Returns:
            str | None: Its value, or None when it is missing or given more than once.
        """
        values = self.headers.get_all(name) or []
        return values[0] if len(values) == 1 else None

    def _serve(self, body: bytes) -> None:
        """
        Answer the request with its body read.

        Any exception out of the application is logged and answered with a
        500 page, so one bad request cannot stop the server.

        Args:
            body (bytes): The request body (empty for a GET).

        Returns:
            None
        """
        request = _Request(method=self.command, target=self.path, host=self._header("Host"),
                           origin=self._header("Origin"),
                           content_type=self._header("Content-Type"), body=body)
        try:
            response = self._app().handle(request)
        except Exception:
            logging.exception("The defaults server failed on %s %s", self.command,
                              _log_safe(self.path))
            response = _Response(500, _message_html(
                500, "Server error", "Something went wrong answering this request; the "
                "server's log has the details."))
        self._send(response)

    def _send(self, response: _Response) -> None:
        """
        Send a response with its status line and headers.

        The base class sends no status line or headers to a request it reads
        as HTTP/0.9 (a request line with no version, or one it could not
        read); every answer here is sent as this server's HTTP/1.0 instead, so
        each carries the six headers. A HEAD answer has no body.

        Args:
            response (_Response): The response.

        Returns:
            None
        """
        if self.request_version == "HTTP/0.9":
            self.request_version = self.protocol_version
        body = response.body.encode("utf-8")
        self.send_response(response.status)
        for name, value in _response_headers(response):
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None,
                   explain: str | None = None) -> None:
        """
        Answer a request the base class refuses before the application sees it.

        The base class calls this for a method with no do_ method here (501),
        a malformed or over-long request line (400, 414), headers it cannot
        read (400, 431) and an HTTP version it does not speak (505). The answer
        is this server's refusal page, sent through _send with the same six
        headers as every other, and the connection is then closed.

        Args:
            code (int): The HTTP status.
            message (str | None): The base class's short reason, shown on the
                page (escaped); None shows the status's own phrase.
            explain (str | None): The base class's longer explanation; not shown.

        Returns:
            None
        """
        self.log_error("code %d, message %s", code, message)
        self.close_connection = True
        self._send(_Response(code, _message_html(
            code, "Refused", message or HTTPStatus(code).phrase)))

    def log_message(self, format: str, *args) -> None:
        """
        Log one line about a request through the logging module, made safe first.

        The base class calls it for every request it answers and every error
        it meets; each non-printable character (a newline in a crafted path,
        say) is written as its escape, so one request is always one line.

        Args:
            format (str): The base class's format string.
            *args: Its arguments.

        Returns:
            None
        """
        logging.info("%s", _log_safe(f"{self.address_string()} {format % args}"))


def _setup_logging(log_path: Path | None) -> None:
    """
    Log to the console and, when given one, to a rotating file.

    5 MB across 3 backups, like main.py's and scheduler.py's own logs; the
    file is only created on the first record.

    Args:
        log_path (Path | None): The log file; None logs to the console only
            (a start that found this checkout's server already running
            leaves the log file to that server).

    Returns:
        None
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True,
        ))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def _code_fingerprint() -> str:
    """
    Fingerprint this package's Python code as it is on disk now.

    A SHA-256 over every .py file in the package's own folder, in name
    order, each as its name and then its length and bytes. Only the code
    counts: a change to the docs, the tests or a saved file leaves it as it
    is. A file that cannot be read counts as its name and the error's type,
    and a folder that cannot be listed as no files, so it never raises.

    Returns:
        str: The fingerprint, 64 hex digits.
    """
    digest = hashlib.sha256()
    folder = Path(__file__).resolve().parent
    try:
        files = sorted(folder.glob("*.py"))
    except OSError:
        files = []
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        try:
            data = path.read_bytes()
        except OSError as exc:
            data = f"unreadable: {type(exc).__name__}".encode()
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


# The fingerprint of the code this process loaded, taken when this module is
# imported (for the server, as it starts). GET /checkout reports it, so a
# second start can tell a server left running across a code change
_LOADED_CODE = _code_fingerprint()


@dataclass(frozen=True)
class _RunningServer:
    """
    What the defaults server already on the port says about itself (GET /checkout).

    Attributes:
        project_root (str): The resolved checkout root it serves.
        code (str | None): The fingerprint of the code it loaded
            (_code_fingerprint); None when its answer carries none.
    """
    project_root: str
    code: str | None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect handler that follows none: an answer must come from the listener itself."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Refuse every redirect, so urllib raises HTTPError for the 3xx instead.

        Args:
            req (urllib.request.Request): The request that was answered.
            fp: The answer's body.
            code (int): The 3xx status.
            msg (str): The status's reason.
            headers: The answer's headers.
            newurl (str): Where the answer points.

        Returns:
            None: Never a new request.
        """
        return None


def _running_checkout(base: str) -> _RunningServer | None:
    """
    Ask the server listening at base which checkout it serves, or None when it gives no such answer.

    One GET of base/checkout, through no proxy and following no redirect,
    with a DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS timeout on the connection
    and on each read, reading at most DEFAULTS_SERVER_CHECKOUT_MAX_BYTES.
    Only a success status whose body is a JSON object with a non-empty
    string "project_root" counts as an answer; anything else — no listener,
    a timeout, an error or redirect status, a longer body, or a body that is
    not that JSON (one nested too deeply to parse included) — is None. The
    answer's "code" is kept when it is a string, else read as None. It never
    raises.

    Args:
        base (str): The server's address, "http://host:port".

    Returns:
        _RunningServer | None: The resolved checkout root the server names
            and the fingerprint of the code it loaded; None when the listener
            is not a defaults server that answers.
    """
    # No proxy (the address is loopback) and no redirect followed
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects)
    try:
        with opener.open(f"{base}/checkout",
                         timeout=DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS) as answer:
            body = answer.read(DEFAULTS_SERVER_CHECKOUT_MAX_BYTES + 1)
        if len(body) > DEFAULTS_SERVER_CHECKOUT_MAX_BYTES:
            return None
        record = json.loads(body.decode("utf-8"))
    except (OSError, ValueError, RecursionError, http.client.HTTPException):
        return None
    if not isinstance(record, dict):
        return None
    root, code = record.get("project_root"), record.get("code")
    if not isinstance(root, str) or not root:
        return None
    return _RunningServer(project_root=root, code=code if isinstance(code, str) else None)


def _start_page(base: str, *, seed: bool) -> tuple[str, int, str]:
    """
    Choose the page a start opens, and what it logs about that page.

    With seed, the confirmation page proposing LIVE_DEFAULTS_SEED. Otherwise
    the backtest dashboard when it exists and has the Save and Trade buttons,
    and this server's index when there is no dashboard, when it was built
    before those buttons (a WARNING, naming the rebuild and the trade page)
    or when it cannot be read (a WARNING with the reason).

    Args:
        base (str): The server's address, "http://host:port".
        seed (bool): Whether this start was asked to propose the seed values.

    Returns:
        tuple[str, int, str]: The address to open, the log level and the
            message to log about it.
    """
    if seed:
        url = f"{base}/confirm?{_seed_query()}"
        return url, logging.INFO, f"Seed values: open {url} to review and confirm them"
    index = f"{base}/"
    dashboard = config.PROJECT_ROOT / DASHBOARD_FILENAME
    state, why = _dashboard_state(dashboard)
    if state == _DASHBOARD_READY:
        return (dashboard.as_uri(), logging.INFO,
                f"Open {dashboard} and use its filter bar's Save as live defaults… or Trade "
                "using defaults… button")
    if state == _DASHBOARD_MISSING:
        return (index, logging.INFO,
                f"No backtest dashboard at {dashboard}: run python3 -m kalshi_betting.backtest "
                f"to build one, or start from the seed values with --seed; opening {index}")
    if state == _DASHBOARD_OLD:
        return (index, logging.WARNING,
                "This dashboard was built before the Save/Trade buttons: rebuild it with "
                "python3 -m kalshi_betting.backtest and the --start-date you built it from; "
                f"until then use {base}/trade (opening {index})")
    return (index, logging.WARNING,
            f"The backtest dashboard at {dashboard} could not be read ({why}); opening {index}")


def _open_start_page(base: str, *, seed: bool, no_browser: bool) -> None:
    """
    Log the page this start is for and, unless told not to, open it in the browser.

    Args:
        base (str): The server's address, "http://host:port".
        seed (bool): Whether this start was asked to propose the seed values.
        no_browser (bool): Whether to open nothing and only log the address.

    Returns:
        None
    """
    url, level, message = _start_page(base, seed=seed)
    logging.log(level, "%s", message)
    if not no_browser:
        webbrowser.open(url)


def _reuse_running_server(parser: argparse.ArgumentParser, base: str, *, seed: bool,
                          no_browser: bool) -> None:
    """
    Answer a start whose port is taken: reopen this checkout's running server's page, or refuse.

    It binds nothing and starts nothing — no second server, no run. When the
    listener answers GET /checkout with this checkout's resolved root and
    the fingerprint of this checkout's code now (_code_fingerprint), it
    logs that this checkout's server is already running and opens the page
    this start asked for (the seed page with seed; else the dashboard, or the
    server's index), unless no_browser. When the listener names another
    checkout, names this one with other code (a server left running across
    a code change, whose pages would check and start runs with that older
    code) or gives no such answer, the start is refused.

    Args:
        parser (argparse.ArgumentParser): main()'s parser, for its error exit.
        base (str): The server's address, "http://host:port".
        seed (bool): Whether this start was asked to propose the seed values.
        no_browser (bool): Whether to open nothing and only log the address.

    Returns:
        None: When this checkout's server is running (the caller then exits 0).

    Raises:
        SystemExit: Status 2 (parser.error) when the port is held by another
            checkout's defaults server, by this checkout's server running
            other code, or by something that is not a defaults server.
    """
    running = _running_checkout(base)
    if running is None:
        parser.error(f"port {DEFAULTS_SERVER_PORT} is in use — stop the other server, "
                     "or change config.DEFAULTS_SERVER_PORT and rebuild the dashboard")
    if running.project_root != str(config.PROJECT_ROOT.resolve()):
        parser.error(f"port {DEFAULTS_SERVER_PORT} is served by the defaults server of "
                     f"{_log_safe(running.project_root)} — stop it (Ctrl-C in its terminal) "
                     "before starting this checkout's")
    if running.code != _code_fingerprint():
        parser.error(f"port {DEFAULTS_SERVER_PORT} is served by this checkout's defaults "
                     "server, but it is running code from before a change to this "
                     "checkout — stop it (Ctrl-C in its terminal) and start it again")
    # The running server keeps its own log file; this short start logs to the terminal
    _setup_logging(None)
    logging.info("This checkout's defaults server is already running at %s/", base)
    _open_start_page(base, seed=seed, no_browser=no_browser)


def main(argv: list[str] | None = None) -> None:
    """
    Run the defaults server until Ctrl-C, or reopen this checkout's server's page.

    It turns Ctrl-C back on first (start_dashboard.sh starts it in the
    background, where a job begins with Ctrl-C ignored), then binds
    DEFAULTS_SERVER_HOST:DEFAULTS_SERVER_PORT. When the port is taken it
    binds nothing and starts nothing: if the listener is this
    checkout's own defaults server running this checkout's current code (its
    GET /checkout names this resolved PROJECT_ROOT and this code's
    fingerprint), it opens the page asked for and returns; if it is that
    server running older code, another checkout's server, or not a defaults
    server, it exits 2 before anything is logged. Otherwise it logs where it
    saves and the live defaults in force. Its _App is built with
    subprocess.Popen as its process starter, the one place that is given, so
    its pages can start runs. With --seed it opens the confirmation page
    proposing LIVE_DEFAULTS_SEED; otherwise it opens the backtest dashboard,
    or its own index when there is no dashboard, the dashboard was built
    before its Save and Trade buttons, or it cannot be read. --no-browser
    only logs the address. When it stops, each run it started that is still
    going gets a WARNING: the run keeps going on its own.

    Args:
        argv (list[str] | None): The arguments; None (default) reads the
            command line.

    Returns:
        None

    Raises:
        SystemExit: Status 2 when the port is held by anything but this
            checkout's defaults server running this checkout's current code,
            or an argument is invalid.
        OSError: When the port cannot be bound for another reason.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.defaults_server",
        description="Serve the pages that save the live trading defaults "
                    "(live_defaults.json), which every live run starts from, and that run "
                    "the live bot with them. When this checkout's server is already "
                    "running, open its page again instead.",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Open the confirmation page proposing the seed values "
             "(config.LIVE_DEFAULTS_SEED) for a first save",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Open nothing; only log the address to open",
    )
    args = parser.parse_args(argv)
    # start_dashboard.sh starts it in the background, where a job begins with
    # Ctrl-C ignored; Ctrl-C must stop it through the KeyboardInterrupt below,
    # so the runs still going are named
    signal.signal(signal.SIGINT, signal.default_int_handler)
    base = f"http://{DEFAULTS_SERVER_HOST}:{DEFAULTS_SERVER_PORT}"
    try:
        # One request at a time, so two saves or two starts can never interleave
        server = HTTPServer((DEFAULTS_SERVER_HOST, DEFAULTS_SERVER_PORT), _Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        _reuse_running_server(parser, base, seed=args.seed, no_browser=args.no_browser)
        return
    # The one process starter any _App is given: this server's pages start runs
    app = _App(DEFAULTS_SERVER_PORT, start_process=subprocess.Popen)
    server.defaults_app = app
    _setup_logging(config.PROJECT_ROOT / _LOG_NAME)
    logging.info("Defaults server at %s/ — it saves the live defaults to %s and trades them "
                 "at %s/trade. Ctrl-C stops it; stop it when you are done.", base,
                 config.LIVE_DEFAULTS_FILE, base)
    try:
        settings = _current_defaults()
    except LiveDefaultsError as exc:
        logging.warning("The saved live defaults are refused (%s): fix or delete the file "
                        "before saving new ones here", exc)
    else:
        if settings is None:
            logging.info("No live defaults are saved: live runs refuse to start until they are")
        else:
            # Every toggle, in the words a live run's "Live settings:" line uses
            logging.info("Live defaults in force: %s — %s", settings.origin,
                         describe_live_settings(settings))
    _open_start_page(base, seed=args.seed, no_browser=args.no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Defaults server stopped")
    finally:
        server.server_close()
        for run in app._runs.values():
            if run.process is not None and run.process.poll() is None:
                logging.warning("A live trading run started here is still running (process "
                                "%s); it keeps running on its own — its result will be in %s",
                                run.process.pid, run.folder)


if __name__ == "__main__":
    main()
