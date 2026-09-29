"""
File: defaults_server.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    A small local web server that saves the live trading defaults
    (config.LIVE_DEFAULTS_FILE, live_defaults.json), which every live run
    starts from. Run by a person, deliberately, from a terminal:

        python3 -m kalshi_betting.defaults_server [--seed] [--no-browser]

    It serves one confirmation page. The page is opened with the proposed
    settings in its address (by the backtest dashboard's filter bar, or by
    --seed, which proposes config.LIVE_DEFAULTS_SEED), shows the live
    defaults in force beside the proposed ones with every change
    highlighted, and writes the proposed ones only when its Confirm button
    is clicked. Closing the tab cancels; nothing is written on the way in.
    With no live defaults saved, every live run refuses to start, so this is
    also how the first file is made.

Dependencies:
    Imports config only: LiveSettings, LIVE_TOGGLE_FIELDS (the seven toggle
    names the fingerprint reads), LiveDefaultsError (a refused saved file)
    and the saved-defaults helpers (read_saved_live_defaults,
    save_live_defaults, live_settings_changes, describe_live_settings,
    live_rule_warnings, live_defaults_source), the seed values and their
    source note, the source-note pattern and the DEFAULTS_SERVER_* and
    DASHBOARD_FILENAME constants. It reads config.LIVE_DEFAULTS_FILE and
    config.PROJECT_ROOT through the module at call time, so the tests'
    redirects reach it. Nothing imports this module: it is a tool a person
    runs, and tests/test_defaults_server.py checks that no other module of
    the package imports it.

Notes:
    It listens on 127.0.0.1 only, answers one request at a time (so two
    saves can never interleave) and never serves the dashboard: any script
    on a page served from its own address could press its button. Every
    request it can read must name this server in its Host header (a
    DNS-rebinding defence); one it cannot read (a malformed request line,
    say) gets only a refusal page. The save (a POST) must also come from
    this server's own page (its Origin header), carry the token that page
    was built with (an HMAC over the proposed fields and the defaults in
    force, keyed by this process), and find the same defaults still in
    force; otherwise it is refused, or the page is shown again against the
    defaults now in force. Every response, a refusal included, is sent as
    HTTP/1.0 with the same six headers: it forbids framing and lets a
    browser run only the one script whose hash its Content-Security-Policy
    names. That script enables Confirm only after the page has been visible
    for DEFAULTS_SERVER_CONFIRM_ARM_MS and the mouse moves or a key is
    pressed, so a click aimed at another page cannot land on it. The source
    note a page proposes must be one of the two shapes
    config.LIVE_DEFAULTS_SOURCE_PATTERN allows, or be left out, and the
    seed's note may label only the seed values, so a crafted link cannot
    choose the note's words. Category and tag names are checked for form
    only (one printable name each), so a link can still put words of its
    own there: the page shows them as a highlighted change, and a name no
    Kalshi series is filed under matches no pair. The headers never
    include "Referrer-Policy: no-referrer": under it a browser sends a
    POST's Origin as "null", and every honest save would be refused.
"""
import argparse
import errno
import hashlib
import hmac
import html
import json
import logging
import logging.handlers
import math
import re
import secrets
import webbrowser
from base64 import b64encode
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from . import config
from .config import (
    DASHBOARD_FILENAME,
    DEFAULTS_SERVER_CONFIRM_ARM_MS,
    DEFAULTS_SERVER_HOST,
    DEFAULTS_SERVER_MAX_REQUEST_BYTES,
    DEFAULTS_SERVER_PORT,
    DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS,
    LIVE_DEFAULTS_SEED,
    LIVE_DEFAULTS_SEED_SOURCE,
    LIVE_DEFAULTS_SOURCE_PATTERN,
    LIVE_TOGGLE_FIELDS,
    LiveDefaultsError,
    LiveSettings,
    describe_live_settings,
    live_defaults_source,
    live_rule_warnings,
    live_settings_changes,
    read_saved_live_defaults,
    save_live_defaults,
)

# The fields a confirmation request carries, in the GET query and repeated in
# the POST body; the signed text is built from exactly these, as raw strings
_FIELDS = ("tier_floors", "spread_min", "spread_max", "k", "size_cap",
           "same_title_size_cap", "category", "tag", "source")
# The fields a proposal must carry; every other one may be left out
_REQUIRED = ("tier_floors", "spread_min", "spread_max", "k", "size_cap")
# The largest number of fields a query or form may hold (its nine fields plus
# the fingerprint and token, with room to spare); more is refused unread
_MAX_FIELDS = 20
# A plain decimal number in ASCII digits: no underscores, spaces, digits of
# other scripts, "nan" or "inf"
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", re.ASCII)
# A token or fingerprint: 64 lower-case hex digits (a SHA-256 hex digest)
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
# A Content-Length value: one or more ASCII digits
_DIGITS = re.compile(r"\d+", re.ASCII)
# The media type a browser gives a form it posts
_FORM_TYPE = "application/x-www-form-urlencoded"
# The rotating log file's name in PROJECT_ROOT
_LOG_NAME = "kalshi_defaults_server.log"

# The one script any page runs: it enables the Confirm button only after the
# page has been visible for DEFAULTS_SERVER_CONFIRM_ARM_MS and the mouse moves
# or a key is pressed, and disables it again whenever the page is hidden
_CONFIRM_JS = """
(function() {
  var button = document.getElementById('confirm');
  if (!button) { return; }
  var timer = null;
  var ready = false;
  function watch() {
    if (document.visibilityState === 'visible') {
      if (timer === null && !ready) {
        timer = setTimeout(function() { ready = true; }, ARM_MS);
      }
    } else {
      if (timer !== null) { clearTimeout(timer); timer = null; }
      ready = false;
      button.disabled = true;
    }
  }
  function arm() {
    if (ready && document.visibilityState === 'visible') { button.disabled = false; }
  }
  document.addEventListener('visibilitychange', watch);
  document.addEventListener('mousemove', arm);
  document.addEventListener('keydown', arm);
  watch();
})();
""".replace("ARM_MS", str(DEFAULTS_SERVER_CONFIRM_ARM_MS))

# The Content-Security-Policy hash of that script, so the browser runs it and
# nothing else
_CONFIRM_JS_HASH = "sha256-" + b64encode(
    hashlib.sha256(_CONFIRM_JS.encode("utf-8")).digest()).decode("ascii")

# The policy every response carries: nothing loads but inline styles and the
# one script above, no page may frame this one, and a form may post only here
_CSP = ("default-src 'none'; style-src 'unsafe-inline'; "
        f"script-src '{_CONFIRM_JS_HASH}'; form-action 'self'; base-uri 'none'; "
        "frame-ancestors 'none'")

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
button { font-size: 15px; padding: 8px 16px; }
"""

# The banner shown when a save finds different defaults in force than the page was built on
_STALE_BANNER = ("The live defaults changed after this page was shown, so nothing was "
                 "saved. The comparison below is against the defaults in force now; "
                 "confirm again only if you still want these settings.")


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
    One HTTP response: a status, an HTML page and, for a redirect, where to.

    Attributes:
        status (int): The HTTP status code.
        body (str): The HTML page.
        location (str | None): The Location header of a redirect; None otherwise.
    """
    status: int
    body: str
    location: str | None = None


def _response_headers(response: _Response) -> list[tuple[str, str]]:
    """
    List the headers a response is sent with (Content-Length aside).

    Every response carries the same six: an HTML type, no caching, no
    type sniffing, no framing, the same-origin referrer policy (under which
    a browser still sends a POST's real Origin) and the Content-Security-
    Policy. A redirect adds its Location.

    Args:
        response (_Response): The response.

    Returns:
        list[tuple[str, str]]: (name, value) pairs, in sending order.
    """
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
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


def _proposal(params: dict[str, list[str]],
              current: LiveSettings | None) -> tuple[LiveSettings, str]:
    """
    Turn a confirmation request's fields into the proposed live defaults.

    tier_floors ("on" / "off"), spread_min, spread_max, k and size_cap (a
    fraction, e.g. 0.2) are required. same_title_size_cap may be left out,
    and then keeps the saved value (or the seed's when none is saved); it is
    the only field that falls back to what is saved. A missing category or
    tag means any, whatever is saved; a tag needs its category. source is
    the note the saved file will keep: left out, it is empty; given, it must
    be one of the two shapes config.LIVE_DEFAULTS_SOURCE_PATTERN allows,
    with ASCII digits only, and the seed's note (LIVE_DEFAULTS_SEED_SOURCE)
    may label only the seed values themselves.

    Args:
        params (dict[str, list[str]]): The request's fields (_params).
        current (LiveSettings | None): The live defaults in force, or None
            when none are saved.

    Returns:
        tuple[LiveSettings, str]: The proposed defaults and the source note.

    Raises:
        ValueError: Naming the first rule the request breaks: an unknown,
            repeated, blank or missing field, a value that is not a plain
            number or a printable name, a tag without a category, a source
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
    if "tag" in value and "category" not in value:
        raise ValueError("a tag needs its category")
    categories = (_name(value["category"], "category"),) if "category" in value else None
    tags = (_name(value["tag"], "tag"),) if "tag" in value else None
    source = value.get("source", "")
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
    )
    # The seed's note names the seed values, so it may label nothing else
    if source == LIVE_DEFAULTS_SEED_SOURCE and settings != LIVE_DEFAULTS_SEED:
        raise ValueError("the seed values' note may label only the seed values "
                         "(config.LIVE_DEFAULTS_SEED)")
    return settings, source


def _seed_query() -> str:
    """
    Build the confirmation page's query that proposes LIVE_DEFAULTS_SEED.

    Each number is written as its repr (the exact float), the source note is
    LIVE_DEFAULTS_SEED_SOURCE, and a category or tag is added only when the
    seed sets one (it sets none: any).

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
        str: The SHA-256 hex digest of "none", or of the origin and the seven
            toggles as JSON.
    """
    if current is None:
        text = "none"
    else:
        text = json.dumps({"origin": current.origin,
                           **{name: getattr(current, name) for name in LIVE_TOGGLE_FIELDS}},
                          sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signed_text(fingerprint: str, params: dict[str, list[str]]) -> str:
    """
    Build the text a confirmation page's token signs.

    The fingerprint of the defaults in force, a newline, then every
    proposal field as the request gave it, sorted and URL-encoded. The raw
    strings are signed, never values re-rendered from LiveSettings, so the
    POST must repeat exactly what the page showed.

    Args:
        fingerprint (str): _fingerprint of the defaults the page was built on.
        params (dict[str, list[str]]): The request's fields; any other than
            _FIELDS is not signed.

    Returns:
        str: The text to sign.
    """
    pairs = sorted((name, value) for name in _FIELDS for value in params.get(name, ()))
    return fingerprint + "\n" + urlencode(pairs)


def _page(title: str, body: str, *, script: bool = False) -> str:
    """
    Wrap a page's body in the HTML document every page shares.

    Args:
        title (str): The page title and heading (escaped here).
        body (str): The body's HTML, every dynamic string in it already escaped.
        script (bool): Whether to add the Confirm script (the confirmation page only).

    Returns:
        str: The whole HTML document.
    """
    escaped = html.escape(title)
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<title>{escaped}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
            f"<h1>{escaped}</h1>\n{body}\n"
            + (f"<script>{_CONFIRM_JS}</script>\n" if script else "")
            + "</body>\n</html>\n")


def _where_html(*, saving: bool) -> str:
    """
    Say, as HTML, which file holds the live defaults and which runs read it.

    Args:
        saving (bool): Whether the page offers a save (a Confirm button); it
            then says the save writes the file.

    Returns:
        str: One paragraph naming config.LIVE_DEFAULTS_FILE and
            config.PROJECT_ROOT, read at call time.
    """
    path = html.escape(str(config.LIVE_DEFAULTS_FILE.absolute()))
    root = html.escape(str(config.PROJECT_ROOT.absolute()))
    if saving:
        return (f"<p>This writes <code>{path}</code>; every live run of the code in "
                f"<code>{root}</code> reads it (the weekly scheduler's too, when it runs "
                "from this checkout).</p>")
    return (f"<p>Every live run of the code in <code>{root}</code> reads its defaults from "
            f"<code>{path}</code> (the weekly scheduler's too, when it runs from this "
            "checkout).</p>")


def _notes_html(*, confirming: bool) -> str:
    """
    List, as HTML, what a save does and does not change.

    Args:
        confirming (bool): Whether the page offers a save (a Confirm button);
            only then does it say that nothing changes until Confirm is clicked.

    Returns:
        str: A list of notes.
    """
    path = html.escape(config.LIVE_DEFAULTS_FILE.name)
    notes = (
        *(("Nothing changes until you click Confirm; closing this tab cancels.",)
          if confirming else ()),
        "A live run already under way keeps the settings it started with; the next "
        "run reads the defaults saved by then.",
        "main.py's toggle flags still override any of these for one run.",
        f"{path} is the only source of live defaults: deleting it stops every live run "
        "until defaults are saved again.",
        "The backtest dashboard's figures for a category or tag are that slice's share "
        "of a run over every category, not a run over that category alone.",
    )
    return ("<ul class=\"note\">" + "".join(f"<li>{note}</li>" for note in notes) + "</ul>")


def _confirm_html(current: LiveSettings | None, settings: LiveSettings, source: str,
                  params: dict[str, list[str]], fingerprint: str, token: str,
                  banner: str | None) -> str:
    """
    Build the confirmation page: the defaults in force beside the proposed ones.

    Every row whose value changes is highlighted, with the new value in bold
    and a "changed" tag; any live_rule_warnings sentence is shown in red.
    When something changes, a form posts back every proposal field as the
    request gave it, with the fingerprint and token, through a Confirm
    button rendered disabled (the page's script enables it); when nothing
    changes there is no button.

    Args:
        current (LiveSettings | None): The live defaults in force, or None when none are saved.
        settings (LiveSettings): The proposed defaults.
        source (str): The proposed source note ("" for none).
        params (dict[str, list[str]]): The request's fields, carried into the form as given.
        fingerprint (str): _fingerprint(current).
        token (str): The token signing fingerprint and params.
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
    parts.append(_where_html(saving=bool(changes)))
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
    parts.append(_notes_html(confirming=bool(changes)))
    if changes:
        hidden = [
            f"<input type=\"hidden\" name=\"{html.escape(name)}\" value=\"{html.escape(value)}\">"
            for name in _FIELDS for value in params.get(name, ())
        ]
        hidden.append(f"<input type=\"hidden\" name=\"fingerprint\" value=\"{fingerprint}\">")
        hidden.append(f"<input type=\"hidden\" name=\"token\" value=\"{token}\">")
        parts.append("<form method=\"post\" action=\"/confirm\">" + "".join(hidden)
                     + "<button id=\"confirm\" type=\"submit\" disabled>Confirm — save the "
                     "live defaults</button></form>")
        parts.append("<p class=\"note\">Confirm becomes clickable a moment after this page "
                     "is shown, once you move the mouse or press a key.</p>")
    else:
        parts.append("<p>These are already the live defaults — nothing to save.</p>")
    return _page(title, "\n".join(parts), script=True)


def _saved_html(settings: LiveSettings) -> str:
    """
    Build the page shown after a save: the live defaults now in force.

    Args:
        settings (LiveSettings): The saved defaults, as read back from the file.

    Returns:
        str: The HTML page.
    """
    # Each toggle's value in the "Live settings:" line's words (compared with
    # nothing, so only the new column is read)
    rows = live_settings_changes(None, settings)
    table = ["<table><tr><th>Setting</th><th>Value</th></tr>"]
    for label, _, value, _ in rows:
        table.append(f"<tr data-setting=\"{html.escape(label)}\"><td>{html.escape(label)}</td>"
                     f"<td>{html.escape(value)}</td></tr>")
    table.append("</table>")
    body = "\n".join((
        "<p>Saved. The live defaults are now:</p>",
        "".join(table),
        f"<p>From: {html.escape(settings.origin)}</p>",
        _where_html(saving=False),
        _notes_html(confirming=False),
        "<p>You can close this tab.</p>",
    ))
    return _page("Live trading defaults saved", body)


def _index_html() -> str:
    """
    Build the page at "/": what this server is for, with no form and no script.

    Returns:
        str: The HTML page.
    """
    dashboard = html.escape(str((config.PROJECT_ROOT / DASHBOARD_FILENAME).absolute()))
    body = ("<p>This server saves the live trading defaults. Open "
            f"<code>{dashboard}</code> and use its filter bar's Save as live defaults… "
            "button, or start from the seed values with "
            "<code>python3 -m kalshi_betting.defaults_server --seed</code>.</p>\n"
            + _where_html(saving=False))
    return _page("Live trading defaults", body)


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


class _App:
    """
    The server's routes and checks, apart from any socket.

    One instance lives as long as its server, with a random key made at
    start; a page's token is valid only for the instance that built it.
    """

    def __init__(self, port: int, key: bytes | None = None) -> None:
        """
        Set up the routes for a server on a port.

        Args:
            port (int): The port the server listens on; a request's Host and a
                save's Origin must name it.
            key (bytes | None): The token key; None (default) makes a random one.
        """
        self.port = port
        self._key = secrets.token_bytes(32) if key is None else key
        self._hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        self._origins = frozenset(f"http://{host}" for host in self._hosts)

    def handle(self, request: _Request) -> _Response:
        """
        Answer one request.

        Every request must name this server in its Host header. GET "/" says
        what the server is for; GET "/confirm" shows the confirmation page;
        POST "/confirm" saves; GET "/saved" shows the defaults in force.
        Anything else is 404.

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
        elif request.method == "POST" and path == "/confirm":
            return self._post_confirm(request)
        return self._refuse(404, "Not found", "This server has no such page.")

    def _token(self, fingerprint: str, params: dict[str, list[str]]) -> str:
        """
        Sign a page's proposal fields and the fingerprint it was built on.

        Args:
            fingerprint (str): _fingerprint of the defaults the page was built on.
            params (dict[str, list[str]]): The proposal's fields.

        Returns:
            str: The HMAC-SHA-256 of _signed_text, as 64 hex digits.
        """
        return hmac.new(self._key, _signed_text(fingerprint, params).encode("utf-8"),
                        hashlib.sha256).hexdigest()

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
            "can be saved over it from here.")

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
        fingerprint = _fingerprint(current)
        return _Response(200, _confirm_html(current, settings, source, params, fingerprint,
                                            self._token(fingerprint, params), None))

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

    def _post_confirm(self, request: _Request) -> _Response:
        """
        Save the proposal a confirmation page posted, after every check.

        In order: the Origin must be this server; the body must be a form;
        it must carry one fingerprint and one token of 64 hex digits; the
        token must sign exactly these fields and that fingerprint for this
        server; the saved file must be usable; the proposal must be valid;
        and the defaults in force must still be the ones the page was built
        on (else the page is shown again against them, with a banner). When
        nothing would change nothing is written. Otherwise the proposal is
        saved and the browser is sent to /saved.

        Args:
            request (_Request): The POST.

        Returns:
            _Response: 303 to /saved on a save; 200 when there is nothing to
                save; 400, 403, 409 or 500 when refused or when the save fails.
        """
        if request.origin not in self._origins:
            return self._refuse(403, "Refused", "A save must come from this server's own "
                                f"confirmation page; its Origin was {request.origin!r}.")
        media = (request.content_type or "").split(";", 1)[0].strip().lower()
        if media != _FORM_TYPE:
            return self._refuse(400, "Refused", f"A save must be a posted form, not "
                                f"{request.content_type!r}.")
        try:
            form = _params(request.body.decode("ascii"))
        except ValueError as exc:
            return self._refuse(400, "Refused", f"The form cannot be read: {exc}")
        fingerprints, tokens = form.pop("fingerprint", []), form.pop("token", [])
        if not (len(fingerprints) == 1 and len(tokens) == 1
                and _HEX64.fullmatch(fingerprints[0]) and _HEX64.fullmatch(tokens[0])):
            return self._refuse(403, "Refused", "The form does not carry this server's "
                                "fingerprint and token.")
        posted_fingerprint, token = fingerprints[0], tokens[0]
        expected = self._token(posted_fingerprint, form)
        if not hmac.compare_digest(token.encode("ascii"), expected.encode("ascii")):
            return self._refuse(
                403, "Refused", "This page was issued by another server session, or its "
                "settings were changed after it was shown — open it again from the dashboard "
                "(or --seed).")
        try:
            current = _current_defaults()
        except LiveDefaultsError as exc:
            return self._refused_file(exc)
        try:
            settings, source = _proposal(form, current)
        except ValueError as exc:
            return self._refuse(400, "These settings cannot be saved", str(exc))
        fingerprint = _fingerprint(current)
        if fingerprint != posted_fingerprint:
            logging.warning("Refused a save from a page built on other live defaults; "
                            "showing it again against the defaults in force")
            return _Response(409, _confirm_html(current, settings, source, form, fingerprint,
                                                self._token(fingerprint, form), _STALE_BANNER))
        # Nothing is written when every toggle already has the proposed value
        if not any(changed for *_, changed in live_settings_changes(current, settings)):
            return _Response(200, _message_html(
                200, "Nothing to save",
                "These are already the live defaults, so nothing was written."))
        try:
            # The one write of the saved file: config parses the text as a
            # live run will before writing it, then reads it back
            save_live_defaults(settings, source=source)
        except LiveDefaultsError as exc:
            # The message says what happened: refused before writing, a failed
            # write that left the old file, or a write whose flush failed
            logging.error("Saving the live defaults failed: %s", _log_safe(str(exc)))
            return _Response(500, _message_html(500, "Saving the live defaults failed",
                                                str(exc)))
        # Marked "(default: X)" against the defaults this save replaced
        logging.info("Saved live defaults to %s: %s", config.LIVE_DEFAULTS_FILE,
                     _log_safe(describe_live_settings(settings, current or settings)))
        return _Response(303, _message_html(303, "Saved", "The live defaults were saved; "
                                            "see /saved."), location="/saved")


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
                411, "Refused", "A save must say how long its form is.")))
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


def _setup_logging(log_path: Path) -> None:
    """
    Log to the console and to a rotating file.

    5 MB across 3 backups, like main.py's and scheduler.py's own logs; the
    file is only created on the first record.

    Args:
        log_path (Path): The log file.

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


def main(argv: list[str] | None = None) -> None:
    """
    Run the defaults server until Ctrl-C.

    It binds DEFAULTS_SERVER_HOST:DEFAULTS_SERVER_PORT first, so a busy port
    exits 2 before anything is logged, then logs where it saves and the
    live defaults in force. With --seed it opens the confirmation page
    proposing LIVE_DEFAULTS_SEED; otherwise it opens the backtest dashboard
    when one has been written. --no-browser only logs the address.

    Args:
        argv (list[str] | None): The arguments; None (default) reads the
            command line.

    Returns:
        None

    Raises:
        SystemExit: Status 2 when the port is in use or an argument is invalid.
        OSError: When the port cannot be bound for another reason.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m kalshi_betting.defaults_server",
        description="Serve the confirmation page that saves the live trading defaults "
                    "(live_defaults.json), which every live run starts from.",
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
    try:
        # One request at a time, so two saves can never interleave
        server = HTTPServer((DEFAULTS_SERVER_HOST, DEFAULTS_SERVER_PORT), _Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"port {DEFAULTS_SERVER_PORT} is in use — stop the other server, "
                         "or change config.DEFAULTS_SERVER_PORT and rebuild the dashboard")
        raise
    server.defaults_app = _App(DEFAULTS_SERVER_PORT)
    _setup_logging(config.PROJECT_ROOT / _LOG_NAME)
    base = f"http://{DEFAULTS_SERVER_HOST}:{DEFAULTS_SERVER_PORT}"
    logging.info("Defaults server at %s/ — it saves the live defaults to %s. Ctrl-C stops it.",
                 base, config.LIVE_DEFAULTS_FILE)
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
    if args.seed:
        url = f"{base}/confirm?{_seed_query()}"
        logging.info("Seed values: open %s to review and confirm them", url)
        if not args.no_browser:
            webbrowser.open(url)
    else:
        dashboard = config.PROJECT_ROOT / DASHBOARD_FILENAME
        if not dashboard.exists():
            logging.info("No backtest dashboard at %s: run a backtest to build one, or "
                         "start from the seed values with --seed", dashboard)
        else:
            logging.info("Open %s and use its filter bar's Save as live defaults… button",
                         dashboard)
            if not args.no_browser:
                webbrowser.open(dashboard.as_uri())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Defaults server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
