"""
File: test_defaults_server.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for defaults_server.py, the local web server whose confirmation
    page saves the live trading defaults. Most tests call _App.handle
    directly with a _Request, so they need no socket: the Host check, the
    instruction page, the confirmation page (every changed row highlighted,
    warnings, escaping, the script and its Content-Security-Policy hash),
    every refused proposal, the save and each check in front of it (Origin,
    content type, fingerprint and token, a stale page, a failed write), the
    saved page and the seed. A POST body is always built from the hidden
    inputs of the page the server rendered, never by hand, except where a
    test signs a request itself to reach a path no rendered page leads to.
    One class runs the Confirm script itself under node or macOS's jsc with
    a stand-in page, one runs the real handler over a loopback socket
    (skipped only if the sandbox refuses to bind a port), one runs main()
    with the server and the browser replaced, and one checks that no other
    module imports this one.

Dependencies:
    Imports kalshi_betting.config (the saved-defaults helpers and constants)
    and kalshi_betting.defaults_server. tests/conftest.py points
    config.LIVE_DEFAULTS_FILE at each test's own tmp_path, so every save
    here lands there.

Notes:
    The socket tests bind 127.0.0.1 on a free port; under a sandbox that
    forbids local binding they skip, and CI runs them. The Confirm script's
    tests skip when neither node nor jsc is present.
"""
import ast
import errno
import hashlib
import html
import http.client
import importlib
import inspect
import io
import json
import logging
import os
import pkgutil
import shutil
import socket
import subprocess
import threading
import time
from base64 import b64encode
from dataclasses import replace
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import urlencode

import pytest

import kalshi_betting
from kalshi_betting import config, defaults_server
from kalshi_betting.config import LiveSettings

PORT = config.DEFAULTS_SERVER_PORT
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://127.0.0.1:{PORT}"
FORM = "application/x-www-form-urlencoded"
KEY = b"k" * 32
# A source note of the dashboard's shape
DASHBOARD_NOTE = "backtest dashboard for 2025-09-24 to 2026-09-27"
# The seven rows of a comparison, in their order on the page
LABELS = ["tier floors", "spread band", "k", "per-trade cap", "same-title cap",
          "categories", "tags"]
# A proposal's fields: the seed's values, spelled as the dashboard spells them
_BASE = {"tier_floors": "off", "spread_min": "0", "spread_max": "0.5", "k": "0.8",
         "size_cap": "0.1"}
# The ASCII digits as Arabic-Indic and as full-width digits (str.translate tables)
_ARABIC_INDIC = str.maketrans("0123456789", "".join(chr(0x0660 + i) for i in range(10)))
_FULL_WIDTH = str.maketrans("0123456789", "".join(chr(0xFF10 + i) for i in range(10)))


class _PageParser(HTMLParser):
    """
    Read what a test needs off a rendered page.

    Collects the forms (method, action), the hidden inputs in page order,
    the buttons' attributes, each comparison row's class and cell texts by
    its data-setting label, and the text of every script.
    """

    def __init__(self) -> None:
        """
        Start with nothing collected, character references decoded in text.

        Returns:
            None
        """
        super().__init__(convert_charrefs=True)
        self.forms: list[tuple[str | None, str | None]] = []
        self.hidden: list[tuple[str, str]] = []
        self.buttons: list[dict] = []
        self.rows: dict[str, str | None] = {}
        self.cells: dict[str, list[str]] = {}
        self.scripts: list[str] = []
        self._row: str | None = None
        self._in_script = False

    def handle_starttag(self, tag, attrs) -> None:
        """
        Record a form, hidden input, button, row, cell or script as it opens.

        Args:
            tag (str): The tag's name, lower-cased.
            attrs (list[tuple[str, str | None]]): Its attributes, values
                unescaped (None for one given without a value).

        Returns:
            None
        """
        attributes = dict(attrs)
        if tag == "form":
            self.forms.append((attributes.get("method"), attributes.get("action")))
        elif tag == "input" and attributes.get("type") == "hidden":
            self.hidden.append((attributes["name"], attributes["value"]))
        elif tag == "button":
            self.buttons.append(attributes)
        elif tag == "tr" and "data-setting" in attributes:
            self._row = attributes["data-setting"]
            self.rows[self._row] = attributes.get("class")
            self.cells[self._row] = []
        elif tag == "td" and self._row is not None:
            self.cells[self._row].append("")
        elif tag == "script":
            self._in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag) -> None:
        """
        Close a row or a script.

        Args:
            tag (str): The tag's name, lower-cased.

        Returns:
            None
        """
        if tag == "tr":
            self._row = None
        elif tag == "script":
            self._in_script = False

    def handle_data(self, data) -> None:
        """
        Add text to the open script or table cell.

        Args:
            data (str): The text.

        Returns:
            None
        """
        if self._in_script:
            self.scripts[-1] += data
        elif self._row is not None and self.cells[self._row]:
            self.cells[self._row][-1] += data


def _parse(body: str) -> _PageParser:
    """
    Parse a rendered page.

    Args:
        body (str): The HTML.

    Returns:
        _PageParser: What the page holds.
    """
    parser = _PageParser()
    parser.feed(body)
    parser.close()
    return parser


def _query(**changes) -> str:
    """
    Build a confirmation query: _BASE with changes, a value of None removing a field.

    Args:
        **changes: Fields to set or (with None) remove.

    Returns:
        str: The query string.
    """
    fields = {**_BASE, **changes}
    return urlencode([(name, value) for name, value in fields.items() if value is not None])


def _app(key: bytes = KEY) -> defaults_server._App:
    """
    Make the application for the default port.

    Args:
        key (bytes): The token key.

    Returns:
        defaults_server._App: The application.
    """
    return defaults_server._App(PORT, key=key)


def _get(app, target: str, host: str | None = HOST) -> defaults_server._Response:
    """
    Send a GET straight to the application.

    Args:
        app (defaults_server._App): The application.
        target (str): The request target.
        host (str | None): The Host header.

    Returns:
        defaults_server._Response: The answer.
    """
    return app.handle(defaults_server._Request("GET", target, host))


def _post(app, fields, *, origin: str | None = ORIGIN, content_type: str | None = FORM,
          host: str | None = HOST, target: str = "/confirm") -> defaults_server._Response:
    """
    Send a POST straight to the application.

    Args:
        app (defaults_server._App): The application.
        fields: The form: a list of (name, value) pairs, or the body as bytes.
        origin (str | None): The Origin header.
        content_type (str | None): The Content-Type header.
        host (str | None): The Host header.
        target (str): The request target.

    Returns:
        defaults_server._Response: The answer.
    """
    body = fields if isinstance(fields, bytes) else urlencode(fields).encode("ascii")
    return app.handle(defaults_server._Request("POST", target, host, origin=origin,
                                               content_type=content_type, body=body))


def _page_form(app, query: str) -> list[tuple[str, str]]:
    """
    Open the confirmation page for a query and read back its form.

    Args:
        app (defaults_server._App): The application.
        query (str): The query string.

    Returns:
        list[tuple[str, str]]: The form's hidden inputs, in page order.
    """
    response = _get(app, f"/confirm?{query}")
    assert response.status == 200, response.body
    return _parse(response.body).hidden


def _save(settings: LiveSettings, source: str = "") -> LiveSettings:
    """
    Save live defaults to the test's own file.

    Args:
        settings (LiveSettings): The defaults.
        source (str): Their source note.

    Returns:
        LiveSettings: The saved defaults as read back.
    """
    return config.save_live_defaults(settings, source=source)


class TestHostCheck:
    """Every request must be addressed to this server (a DNS-rebinding defence)."""

    @pytest.mark.parametrize("host", [None, "127.0.0.1:9999", "evil.test:8765", "localhost",
                                      "127.0.0.1", "127.0.0.1:8765.evil.test"])
    def test_other_hosts_are_refused(self, host):
        response = _get(_app(), "/", host=host)
        assert response.status == 403
        assert "<form" not in response.body

    @pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}",
                                      f"LOCALHOST:{PORT}"])
    def test_the_loopback_names_pass(self, host):
        assert _get(_app(), "/", host=host).status == 200

    def test_a_save_to_another_host_is_refused_before_anything_else(self):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, form, host="evil.test:8765").status == 403
        assert config.read_saved_live_defaults() is None


class TestIndexPage:
    """GET / says what the server is for, with no form and no script."""

    def test_the_instruction_page(self):
        response = _get(_app(), "/")
        assert response.status == 200
        page = _parse(response.body)
        assert page.forms == [] and page.scripts == [] and page.buttons == []
        assert "Save as live defaults…" in response.body
        assert "python3 -m kalshi_betting.defaults_server --seed" in response.body
        assert str((config.PROJECT_ROOT / config.DASHBOARD_FILENAME).absolute()) in response.body
        assert f"reads its defaults from <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" \
            in response.body
        assert "This writes" not in response.body

    def test_anything_else_is_404(self):
        assert _get(_app(), "/nope").status == 404
        assert _post(_app(), [], target="/").status == 404


class TestConfirmPage:
    """GET /confirm: the defaults in force beside the proposed ones."""

    def test_with_a_file_saved_only_the_changed_rows_are_highlighted(self):
        saved = _save(LiveSettings(True, (0.0, 1.0), 0.8, 0.1, 0.5), source=DASHBOARD_NOTE)
        response = _get(_app(), "/confirm?" + _query(same_title_size_cap="0.5",
                                                   source=DASHBOARD_NOTE))
        assert response.status == 200
        page = _parse(response.body)
        assert list(page.rows) == LABELS
        changed = {label for label, css in page.rows.items() if css == "changed"}
        assert changed == {"tier floors", "spread band"}
        assert all(css == "same" for label, css in page.rows.items() if label not in changed)
        assert response.body.count('<span class="tag">changed</span>') == 2
        assert "2 of 7 settings change." in response.body
        assert "Overwrite the live trading defaults?" in response.body
        assert f"This writes <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" \
            in response.body
        assert "Nothing changes until you click Confirm; closing this tab cancels." \
            in response.body
        assert saved.origin in response.body
        assert f"New defaults from: {DASHBOARD_NOTE}" in response.body
        # Current and new values, in the "Live settings:" line's words
        assert page.cells["tier floors"][:3] == ["tier floors", "on", "off"]
        assert page.cells["spread band"][:3] == ["spread band", "none", "0-0.5"]

    def test_the_changed_value_is_bold(self):
        _save(LiveSettings(True, (0.0, 1.0), 0.8, 0.1, 0.5))
        body = _get(_app(), "/confirm?" + _query(same_title_size_cap="0.5")).body
        assert '<td><b>off</b></td>' in body
        assert '<td><b>0.8</b></td>' not in body

    def test_with_none_saved_every_row_changes(self):
        response = _get(_app(), "/confirm?" + _query())
        assert response.status == 200
        page = _parse(response.body)
        assert "Save the first live trading defaults?" in response.body
        assert "none saved — live runs refuse to start until defaults are saved" in response.body
        assert set(page.rows.values()) == {"changed"}
        assert all(cells[1] == "—" for cells in page.cells.values())
        assert "7 of 7 settings change." in response.body
        # The same-title cap falls back to the seed's
        assert page.cells["same-title cap"][2] == "20%"

    def test_the_exposure_warning_shows(self):
        response = _get(_app(), "/confirm?" + _query(k="0.4", size_cap="1"))
        warnings = config.live_rule_warnings(LiveSettings(False, (0.0, 0.5), 0.4, 1.0, 0.2))
        assert warnings
        for sentence in warnings:
            assert f'<p class="warn">Warning: {sentence}.</p>' in response.body

    def test_no_warning_for_the_seed(self):
        assert 'class="warn"' not in _get(_app(), "/confirm?" + _query()).body

    def test_a_missing_same_title_cap_keeps_the_saved_value(self):
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.5))
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.rows["same-title cap"] == "same"
        assert page.cells["same-title cap"][1:3] == ["50%", "50%"]

    def test_a_missing_category_or_tag_proposes_any(self):
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2, ("Sports",), ("Basketball",)))
        page = _parse(_get(_app(), "/confirm?" + _query(same_title_size_cap="0.2")).body)
        assert page.rows["categories"] == page.rows["tags"] == "changed"
        assert page.cells["categories"][1:3] == ["Sports", "any"]
        assert page.cells["tags"][1:3] == ["Basketball", "any"]

    def test_confirm_is_there_and_disabled_when_something_changes(self):
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.forms == [("post", "/confirm")]
        [button] = page.buttons
        assert button["id"] == "confirm" and button["type"] == "submit"
        assert "disabled" in button
        names = [name for name, _ in page.hidden]
        assert names == ["tier_floors", "spread_min", "spread_max", "k", "size_cap",
                         "fingerprint", "token"]

    def test_no_button_when_nothing_changes(self):
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2))
        response = _get(_app(), "/confirm?" + _query())
        page = _parse(response.body)
        assert page.buttons == [] and page.forms == [] and page.hidden == []
        assert "These are already the live defaults — nothing to save." in response.body
        assert "0 of 7 settings change." in response.body
        # With nothing to confirm, the page neither says it writes nor waits for Confirm
        assert "This writes" not in response.body and "click Confirm" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body

    def test_the_csp_hash_is_the_one_script_on_the_page(self):
        response = _get(_app(), "/confirm?" + _query())
        [script] = _parse(response.body).scripts
        assert script == defaults_server._CONFIRM_JS
        digest = b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
        csp = dict(defaults_server._response_headers(response))["Content-Security-Policy"]
        assert f"script-src 'sha256-{digest}'" in csp

    def test_the_script_arms_after_the_configured_delay(self):
        js = defaults_server._CONFIRM_JS
        assert f"}}, {config.DEFAULTS_SERVER_CONFIRM_ARM_MS});" in js
        assert "ARM_MS" not in js
        for event in ("visibilitychange", "mousemove", "keydown"):
            assert f"addEventListener('{event}'" in js

    def test_a_hostile_category_is_escaped(self):
        hostile = "<script>alert(1)</script>"
        response = _get(_app(), "/confirm?" + _query(category=hostile, tag='"><b>x'))
        assert response.status == 200
        assert "<script>alert" not in response.body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.body
        page = _parse(response.body)
        assert len(page.scripts) == 1
        assert ("category", hostile) in page.hidden and ("tag", '"><b>x') in page.hidden
        assert page.cells["categories"][2] == hostile


# macOS's JavaScriptCore shell, the runtime used when node is not installed
_JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")

# A stand-in for the few browser pieces the Confirm script touches: its button,
# the page's visibility, event listeners a step fires, and timers that run only
# when a step says so. The script runs inside a function whose parameters are
# these stand-ins, so neither runtime's own globals are replaced. The steps
# follow: ["fire", event], ["show", "visible" | "hidden"] (sets the visibility
# and fires visibilitychange), ["elapse"] (runs every pending timer) and
# ["snap", label] (records the button, the pending timers' delays and the
# events listened for); the records are printed as one JSON line.
_CONFIRM_HARNESS = """
var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __listeners = {};
var __timers = {};
var __next = 1;
var __button = __HAS_BUTTON__ ? {disabled: true} : null;
var __document = {
  visibilityState: __VISIBILITY__,
  getElementById: function(id) { return id === 'confirm' ? __button : null; },
  addEventListener: function(type, fn) {
    (__listeners[type] = __listeners[type] || []).push(fn);
  }
};
function __setTimeout(fn, ms) { var id = __next++; __timers[id] = {fn: fn, ms: ms}; return id; }
function __clearTimeout(id) { delete __timers[id]; }
function __fire(type) {
  (__listeners[type] || []).forEach(function(fn) { fn({type: type}); });
}
(function(document, setTimeout, clearTimeout) {
__SCRIPT__
})(__document, __setTimeout, __clearTimeout);
var __out = {};
__STEPS__.forEach(function(step) {
  if (step[0] === 'fire') {
    __fire(step[1]);
  } else if (step[0] === 'show') {
    __document.visibilityState = step[1];
    __fire('visibilitychange');
  } else if (step[0] === 'elapse') {
    Object.keys(__timers).forEach(function(id) {
      var timer = __timers[id];
      delete __timers[id];
      timer.fn();
    });
  } else if (step[0] === 'snap') {
    __out[step[1]] = {
      disabled: __button ? __button.disabled : null,
      timers: Object.keys(__timers).map(function(id) { return __timers[id].ms; }),
      listeners: Object.keys(__listeners).sort()
    };
  }
});
__emit(JSON.stringify(__out));
"""


def _js_runtime() -> str | None:
    """
    Find a JavaScript runtime for the Confirm script.

    The choice tests/test_dashboard.py's script tests make: node when
    installed (CI's runners have it), else macOS's JavaScriptCore shell.

    Returns:
        str | None: The runtime's path, or None when neither is present.
    """
    node = shutil.which("node")
    if node:
        return node
    return str(_JSC) if _JSC.exists() else None


def _run_confirm_script(tmp_path, steps, *, visible: bool = True,
                        has_button: bool = True) -> dict:
    """
    Run the page's Confirm script, byte for byte, through a list of steps.

    Skips the test when no JavaScript runtime is present.

    Args:
        tmp_path (Path): Where to write the program.
        steps (list[list[str]]): The steps (see _CONFIRM_HARNESS).
        visible (bool): Whether the page is visible when the script runs.
        has_button (bool): Whether the page has its Confirm button.

    Returns:
        dict: Each "snap" step's record, by its label.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("neither node nor macOS's jsc is here to run the Confirm script")
    program = (_CONFIRM_HARNESS
               .replace("__HAS_BUTTON__", "true" if has_button else "false")
               .replace("__VISIBILITY__", json.dumps("visible" if visible else "hidden"))
               .replace("__STEPS__", json.dumps(steps))
               .replace("__SCRIPT__", defaults_server._CONFIRM_JS))
    path = tmp_path / "confirm_script.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestConfirmScript:
    """The Confirm script itself, run under node or jsc with a stand-in page."""

    ARM = config.DEFAULTS_SERVER_CONFIRM_ARM_MS
    EVENTS = ["keydown", "mousemove", "visibilitychange"]

    def test_nothing_arms_it_before_the_delay(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["snap", "load"], ["fire", "mousemove"], ["fire", "keydown"], ["snap", "early"]])
        assert out["load"] == {"disabled": True, "timers": [self.ARM], "listeners": self.EVENTS}
        assert out["early"]["disabled"] is True

    @pytest.mark.parametrize("event", ["mousemove", "keydown"])
    def test_after_the_delay_a_mouse_move_or_a_key_arms_it(self, tmp_path, event):
        out = _run_confirm_script(tmp_path, [
            ["elapse"], ["snap", "ready"], ["fire", event], ["snap", "armed"]])
        # The delay alone arms nothing: it takes the reader's own move
        assert out["ready"]["disabled"] is True
        assert out["armed"]["disabled"] is False

    def test_hiding_the_page_disarms_it_and_a_return_needs_the_full_delay_again(
            self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["elapse"], ["fire", "mousemove"], ["snap", "armed"],
            ["show", "hidden"], ["snap", "hidden"], ["fire", "keydown"], ["snap", "moved"],
            ["show", "visible"], ["snap", "back"], ["fire", "mousemove"], ["snap", "early"],
            ["elapse"], ["fire", "keydown"], ["snap", "again"]])
        assert out["armed"]["disabled"] is False
        assert out["hidden"]["disabled"] is True and out["hidden"]["timers"] == []
        assert out["moved"]["disabled"] is True
        assert out["back"]["timers"] == [self.ARM]
        assert out["early"]["disabled"] is True
        assert out["again"]["disabled"] is False

    def test_hiding_the_page_before_the_delay_ends_restarts_it(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["show", "hidden"], ["snap", "hidden"], ["show", "visible"], ["snap", "back"],
            ["fire", "mousemove"], ["snap", "early"]])
        assert out["hidden"]["timers"] == [] and out["back"]["timers"] == [self.ARM]
        assert out["early"]["disabled"] is True

    def test_a_page_opened_hidden_starts_no_clock_until_it_is_shown(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["snap", "load"], ["fire", "mousemove"], ["snap", "moved"],
            ["show", "visible"], ["snap", "shown"]], visible=False)
        assert out["load"]["timers"] == [] and out["moved"]["disabled"] is True
        assert out["shown"]["timers"] == [self.ARM]

    def test_a_page_without_confirm_listens_for_nothing(self, tmp_path):
        out = _run_confirm_script(tmp_path, [["snap", "load"]], has_button=False)
        assert out["load"] == {"disabled": None, "timers": [], "listeners": []}


class TestConfirmRefusals:
    """A proposal the server cannot save is refused with 400, naming why."""

    @pytest.mark.parametrize("query", [
        _query(foo="1"),                                  # unknown field
        _query() + "&k=0.8",                              # repeated
        _query(k=""),                                     # blank
        _query(source=""),                                # a blank source
        _query(tier_floors="maybe"),
        _query(k="nan"), _query(k="inf"), _query(k="0_8"), _query(k="1e-400"),
        _query(k="1e400"), _query(k="\uff10.8"), _query(k=" 0.8"), _query(k="0"),
        _query(k="1.5"), _query(k="0x1"),
        _query(size_cap="0.33"), _query(same_title_size_cap="0"),
        _query(tag="Basketball"),                         # a tag without its category
        _query(category="any"), _query(category="Any"), _query(category=" Sports"),
        _query(category="Sports\u200b"), _query(category="Sports", tag="Bask\u202eetball"),
        _query(spread_min="0.5", spread_max="0.5"),
        _query(spread_min="0.6"),
        _query(spread_min="1e-400"),                      # would read as 0: refused, not rounded
        _query(source="my own words"),
        _query(source=DASHBOARD_NOTE + " and more"),
        _query(source=DASHBOARD_NOTE + "x" * 300),
        _query(source=" " + config.LIVE_DEFAULTS_SEED_SOURCE),
        _query(k="0.75", source=config.LIVE_DEFAULTS_SEED_SOURCE),  # not the seed's values
        _query(source=DASHBOARD_NOTE.translate(_ARABIC_INDIC)),     # other scripts' digits
        _query(source=DASHBOARD_NOTE.translate(_FULL_WIDTH)),
        _query() + "&category=Fútbol",                # not percent-encoded
        "k", "&&", _query() + "&" + "&".join(f"x{i}=1" for i in range(20)),
    ])
    def test_it_is_refused(self, query):
        response = _get(_app(), f"/confirm?{query}")
        assert response.status == 400, query
        assert "<form" not in response.body
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("field", ["tier_floors", "spread_min", "spread_max", "k",
                                       "size_cap"])
    def test_a_required_field_may_not_be_missing(self, field):
        response = _get(_app(), "/confirm?" + _query(**{field: None}))
        assert response.status == 400
        assert f"{field} is missing" in response.body

    def test_the_refusal_names_the_rule(self):
        body = _get(_app(), "/confirm?" + _query(tag="Basketball")).body
        assert "a tag needs its category" in body

    def test_the_seed_note_labels_the_seed_values_only(self):
        # The seed's values under its note are shown ...
        assert _get(_app(), "/confirm?" + _query(
            source=config.LIVE_DEFAULTS_SEED_SOURCE)).status == 200
        # ... but a same-title cap that falls back to a saved 50% is not the
        # seed's, so its note may not label it
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.5))
        response = _get(_app(), "/confirm?" + _query(source=config.LIVE_DEFAULTS_SEED_SOURCE))
        assert response.status == 400
        assert "may label only the seed values" in response.body


class TestRefusedFile:
    """A saved file that cannot be used blocks every read route with 409, no form."""

    def test_every_route_is_refused(self):
        app = _app()
        form = _page_form(app, _query())
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        for response in (_get(app, "/confirm?" + _query()), _post(app, form),
                         _get(app, "/saved")):
            assert response.status == 409
            assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body
            assert "<form" not in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == b"not json"


class TestSave:
    """POST /confirm writes exactly what its page showed, after every check."""

    def test_a_first_save_writes_exactly_the_proposal(self):
        app = _app()
        form = _page_form(app, _query(source=DASHBOARD_NOTE))
        response = _post(app, form)
        assert response.status == 303 and response.location == "/saved"
        saved = config.read_saved_live_defaults()
        assert saved == LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2)
        assert saved.origin.endswith(" from " + DASHBOARD_NOTE)

    def test_it_overwrites_the_defaults_in_force(self):
        _save(LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0))
        app = _app()
        form = _page_form(app, _query(category="Sports", tag="Basketball",
                                      same_title_size_cap="0.5"))
        assert _post(app, form).status == 303
        assert config.read_saved_live_defaults() == LiveSettings(
            False, (0.0, 0.5), 0.8, 0.1, 0.5, ("Sports",), ("Basketball",))

    def test_a_form_with_a_charset_is_accepted(self):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, form, content_type=FORM + "; charset=UTF-8").status == 303

    @pytest.mark.parametrize("origin", [None, "null", "http://evil.test",
                                        "http://127.0.0.1:9999", "https://127.0.0.1:8765",
                                        "http://127.0.0.1:8765/"])
    def test_the_origin_must_be_this_server(self, origin):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, form, origin=origin).status == 403
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("content_type", [None, "text/plain", "multipart/form-data",
                                              "application/json"])
    def test_the_body_must_be_a_form(self, content_type):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, form, content_type=content_type).status == 400
        assert config.read_saved_live_defaults() is None

    def test_an_unreadable_body_is_refused(self):
        app = _app()
        assert _post(app, "k=é".encode()).status == 400
        assert _post(app, b"no-equals-sign").status == 400

    @staticmethod
    def _without(form, name):
        """
        Drop every field of one name from a form.

        Args:
            form (list[tuple[str, str]]): The form's fields.
            name (str): The field name to drop.

        Returns:
            list[tuple[str, str]]: The other fields, in order.
        """
        return [(n, v) for n, v in form if n != name]

    @pytest.mark.parametrize("mangle", [
        lambda f: TestSave._without(f, "token"),
        lambda f: TestSave._without(f, "fingerprint"),
        lambda f: f + [pair for pair in f if pair[0] == "token"],
        lambda f: f + [pair for pair in f if pair[0] == "fingerprint"],
        lambda f: TestSave._without(f, "token") + [("token", "é" * 64)],
        lambda f: TestSave._without(f, "token") + [("token", "z" * 64)],
        lambda f: TestSave._without(f, "token") + [
            ("token", dict(f)["token"].upper())],
        lambda f: TestSave._without(f, "token") + [("token", dict(f)["token"][:63])],
        lambda f: TestSave._without(f, "fingerprint") + [("fingerprint", "0" * 64)],
    ])
    def test_the_fingerprint_and_token_must_be_this_page_s(self, mangle):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, mangle(form)).status == 403
        assert config.read_saved_live_defaults() is None

    def test_a_token_from_another_server_session_is_refused(self):
        form = _page_form(_app(KEY), _query())
        response = _post(_app(b"x" * 32), form)
        assert response.status == 403
        assert "another server session" in response.body
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("tamper", [
        lambda f: [(n, "0.4" if n == "k" else v) for n, v in f],
        lambda f: f + [("category", "Sports")],
        lambda f: [(n, v) for n, v in f if n != "tag"],
        lambda f: [(n, v) for n, v in f if n != "same_title_size_cap"],
    ])
    def test_a_field_changed_after_signing_is_refused(self, tamper):
        app = _app()
        form = _page_form(app, _query(category="Sports", tag="Basketball",
                                      same_title_size_cap="0.5"))
        assert _post(app, tamper(form)).status == 403
        assert config.read_saved_live_defaults() is None

    def test_an_unsigned_extra_field_is_refused(self):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, form + [("foo", "1")]).status == 400
        assert config.read_saved_live_defaults() is None

    def test_a_stale_page_is_shown_again_against_the_defaults_in_force(self):
        app = _app()
        form = _page_form(app, _query())
        other = _save(LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        response = _post(app, form)
        assert response.status == 409
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert defaults_server._STALE_BANNER in response.body
        assert other.origin in response.body
        again = _parse(response.body).hidden
        assert dict(again)["fingerprint"] != dict(form)["fingerprint"]
        # The page shown again saves
        assert _post(app, again).status == 303
        assert config.read_saved_live_defaults() == LiveSettings(False, (0.0, 0.5), 0.8,
                                                                 0.1, 1.0)

    def test_a_failed_save_is_a_500_and_keeps_the_old_file(self, monkeypatch):
        _save(LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        app = _app()
        form = _page_form(app, _query())

        def refuse(src, dst):
            """
            Stand in for os.replace, failing as a full or read-only disk would.

            Args:
                src: The file to rename.
                dst: Where to.

            Raises:
                OSError: Always.
            """
            raise OSError("no rename today")

        monkeypatch.setattr(os, "replace", refuse)
        response = _post(app, form)
        assert response.status == 500
        assert "no rename today" in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_a_zero_floor_and_a_non_ascii_tag_round_trip(self):
        app = _app()
        form = _page_form(app, _query(spread_min="0", category="Deportes", tag="Fútbol"))
        assert ("tag", "Fútbol") in form
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.spread_band == (0.0, 0.5) and saved.tags == ("Fútbol",)

    def test_nothing_to_save_writes_nothing(self):
        # No rendered page offers this (it has no form when nothing changes),
        # so the test signs the request itself
        saved = _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        app = _app()
        params = {name: [value] for name, value in _BASE.items()}
        fingerprint = defaults_server._fingerprint(saved)
        form = [(name, values[0]) for name, values in params.items()]
        form += [("fingerprint", fingerprint), ("token", app._token(fingerprint, params))]
        response = _post(app, form)
        assert response.status == 200
        assert "nothing was written" in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_the_save_is_logged_against_the_defaults_it_replaced(self, caplog):
        _save(LiveSettings(True, (0.0, 0.5), 0.8, 0.1, 0.2))
        app = _app()
        form = _page_form(app, _query())
        with caplog.at_level(logging.INFO):
            assert _post(app, form).status == 303
        [line] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("Saved live defaults to")]
        assert f"to {config.LIVE_DEFAULTS_FILE}: tier floors off (default: on) |" in line

    def test_a_refusal_is_logged_as_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _get(_app(), "/confirm?" + _query(tag="Basketball"))
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.getMessage() == "Refused (400 Bad Request): a tag needs its category"


class TestSavedPage:
    """GET /saved shows the live defaults in force."""

    def test_it_shows_the_new_defaults(self):
        app = _app()
        assert _post(app, _page_form(app, _query(source=DASHBOARD_NOTE))).status == 303
        response = _get(app, "/saved")
        assert response.status == 200
        assert "Saved. The live defaults are now:" in response.body
        page = _parse(response.body)
        assert list(page.rows) == LABELS
        assert page.cells["tier floors"] == ["tier floors", "off"]
        assert page.cells["per-trade cap"] == ["per-trade cap", "10%"]
        assert config.read_saved_live_defaults().origin in response.body
        assert page.forms == [] and page.scripts == []
        # The save is done: the page no longer speaks of writing or of Confirm
        assert "This writes" not in response.body and "click Confirm" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body

    def test_with_none_saved_it_is_404(self):
        assert _get(_app(), "/saved").status == 404


class TestSeed:
    """--seed's confirmation page proposes exactly LIVE_DEFAULTS_SEED."""

    def test_the_seed_query_proposes_the_seed(self):
        params = defaults_server._params(defaults_server._seed_query())
        settings, source = defaults_server._proposal(params, None)
        assert settings == config.LIVE_DEFAULTS_SEED
        assert source == config.LIVE_DEFAULTS_SEED_SOURCE
        # It names every field, so what is saved does not change it
        saved = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        assert defaults_server._proposal(params, saved)[0] == config.LIVE_DEFAULTS_SEED

    def test_confirming_the_seed_saves_it(self):
        app = _app()
        response = _get(app, "/confirm?" + defaults_server._seed_query())
        assert response.status == 200
        assert "Save the first live trading defaults?" in response.body
        assert 'class="warn"' not in response.body
        assert _post(app, _parse(response.body).hidden).status == 303
        live = config.live_defaults()
        assert live == config.LIVE_DEFAULTS_SEED
        assert live.origin.endswith(" from " + config.LIVE_DEFAULTS_SEED_SOURCE)


class TestHelpers:
    """The pieces the routes are built from."""

    def test_the_fingerprint_follows_the_defaults_and_when_they_were_saved(self):
        seed = config.LIVE_DEFAULTS_SEED
        assert defaults_server._fingerprint(None) == hashlib.sha256(b"none").hexdigest()
        a = replace(seed, origin="live_defaults.json, saved 2026-09-27T21:05:13Z")
        b = replace(seed, origin="live_defaults.json, saved 2026-09-27T21:05:14Z")
        assert defaults_server._fingerprint(a) == defaults_server._fingerprint(replace(a))
        assert defaults_server._fingerprint(a) != defaults_server._fingerprint(b)
        assert defaults_server._fingerprint(a) != defaults_server._fingerprint(
            replace(a, interval_discount=0.75))

    def test_the_signed_text_is_the_raw_fields_sorted(self):
        params = {"k": ["0.80"], "tier_floors": ["off"], "token": ["x"], "category": ["A B"]}
        assert defaults_server._signed_text("f" * 64, params) == (
            "f" * 64 + "\ncategory=A+B&k=0.80&tier_floors=off")

    def test_a_number_reads_plainly(self):
        assert defaults_server._number("-0", "k") == 0.0
        assert str(defaults_server._number("-0", "k")) == "0.0"
        assert defaults_server._number("1e-3", "k") == 0.001
        assert defaults_server._number("0.000", "k") == 0.0
        assert defaults_server._number("0e-400", "k") == 0.0
        with pytest.raises(ValueError, match="out of range"):
            defaults_server._number("1e-400", "spread_min")

    def test_log_safe_escapes_what_cannot_print(self):
        text = defaults_server._log_safe("a\nb\x1b[31m\u202ec\U000e0041 ok")
        assert text == "a\\x0ab\\x1b[31m\\u202ec\\U000e0041 ok"

    def test_the_handler_logs_through_logging_on_one_line(self, caplog):
        handler = object.__new__(defaults_server._Handler)
        handler.client_address = ("127.0.0.1", 5555)
        with caplog.at_level(logging.INFO):
            handler.log_message('"%s" %s %s', "GET /\r\nInjected: yes HTTP/1.0", "200", "-")
        [record] = caplog.records
        assert "\r" not in record.getMessage() and "\n" not in record.getMessage()
        assert "\\x0d\\x0aInjected" in record.getMessage()

    def test_every_response_carries_the_six_headers(self):
        headers = defaults_server._response_headers(defaults_server._Response(200, ""))
        assert [name for name, _ in headers] == [
            "Content-Type", "Cache-Control", "X-Content-Type-Options", "X-Frame-Options",
            "Referrer-Policy", "Content-Security-Policy"]
        values = dict(headers)
        assert values["Referrer-Policy"] == "same-origin"
        assert "frame-ancestors 'none'" in values["Content-Security-Policy"]
        assert "form-action 'self'" in values["Content-Security-Policy"]
        redirect = dict(defaults_server._response_headers(
            defaults_server._Response(303, "", location="/saved")))
        assert redirect["Location"] == "/saved"

    def test_a_silent_connection_is_dropped_after_the_configured_timeout(self):
        # The handler's own class attribute, not the base class's None (no
        # timeout at all), so one idle connection cannot hold the server
        timeout = defaults_server._Handler.__dict__["timeout"]
        assert timeout == config.DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS
        assert isinstance(timeout, (int, float)) and timeout > 0

    @staticmethod
    def _handler(command, request_version):
        """
        Make a handler that has read a request line, writing its answer to memory.

        Args:
            command (str | None): The method the base class read (None or ""
                when it could not read one).
            request_version (str): The version it read ("HTTP/0.9" is its
                default when the request line cannot be read).

        Returns:
            defaults_server._Handler: The handler, its wfile a BytesIO.
        """
        handler = object.__new__(defaults_server._Handler)
        handler.client_address = ("127.0.0.1", 5555)
        handler.command = command
        handler.request_version = request_version
        handler.requestline = f"{command} / {request_version}"
        handler.close_connection = False
        handler.wfile = io.BytesIO()
        return handler

    @pytest.mark.parametrize("command, version, code, message", [
        ("PUT", "HTTP/1.1", 501, "Unsupported method ('PUT')"),
        ("HEAD", "HTTP/1.1", 501, "Unsupported method ('HEAD')"),
        (None, "HTTP/0.9", 400, "Bad request syntax ('<b>GARBAGE</b>')"),
        ("", "", 414, None),
    ])
    def test_a_refusal_from_the_http_layer_is_this_server_s_page(
            self, command, version, code, message):
        handler = self._handler(command, version)
        handler.send_error(code, message)
        head, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        headers = dict(line.split(": ", 1) for line in header_lines)
        # A status line and the six headers, even on a request read as HTTP/0.9
        assert status_line.startswith(f"HTTP/1.0 {code} ")
        for name, value in defaults_server._response_headers(defaults_server._Response(code, "")):
            assert headers[name] == value, name
        assert handler.close_connection
        if command == "HEAD":
            assert body == b"" and int(headers["Content-Length"]) > 0
            return
        assert int(headers["Content-Length"]) == len(body)
        page = body.decode("utf-8")
        assert f"HTTP {code} " in page and "<script" not in page
        if message is None:
            assert html.escape(HTTPStatus(code).phrase) in page
        else:
            assert html.escape(message) in page


@pytest.fixture
def live_server(monkeypatch):
    """
    Run the real handler on a free loopback port in a thread.

    The silent-connection timeout is lowered so a test does not wait the
    configured seconds. Skipped when the sandbox refuses to bind a port.

    Args:
        monkeypatch (pytest.MonkeyPatch): Lowers the handler's timeout.

    Yields:
        int: The port.
    """
    monkeypatch.setattr(defaults_server._Handler, "timeout", 0.5)
    try:
        server = HTTPServer(("127.0.0.1", 0), defaults_server._Handler)
    except PermissionError as exc:
        pytest.skip(f"this sandbox does not allow binding a local port: {exc}")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(port: int, method: str, target: str, *, body: bytes | None = None,
             headers: dict | None = None, host: str | None = None):
    """
    Send one request over the socket and read the whole answer.

    Args:
        port (int): The server's port.
        method (str): The method.
        target (str): The request target.
        body (bytes | None): The body, sent with its Content-Length.
        headers (dict | None): More headers.
        host (str | None): A Host header to send instead of http.client's own.

    Returns:
        tuple[int, dict, str]: The status, the headers and the body.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest(method, target, skip_host=host is not None,
                              skip_accept_encoding=True)
        if host is not None:
            connection.putheader("Host", host)
        for name, value in (headers or {}).items():
            connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode("utf-8")
    finally:
        connection.close()


def _raw_request(port: int, data: bytes):
    """
    Send raw bytes as a request, as no well-behaved client would, and read the answer.

    Every byte sent is one the server reads before answering, so it closes
    the connection cleanly rather than resetting it.

    Args:
        port (int): The server's port.
        data (bytes): The request, exactly as sent.

    Returns:
        tuple[int, dict, str]: The status, the headers and the body.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(data)
        response = http.client.HTTPResponse(sock)
        try:
            response.begin()
            return response.status, dict(response.getheaders()), response.read().decode("utf-8")
        finally:
            response.close()


class TestOverASocket:
    """The real handler, over a loopback socket."""

    def test_a_confirm_page_and_its_save(self, live_server):
        port = live_server
        status, _, body = _request(port, "GET", "/confirm?" + _query(source=DASHBOARD_NOTE))
        assert status == 200
        form = urlencode(_parse(body).hidden).encode("ascii")
        status, headers, _ = _request(port, "POST", "/confirm", body=form, headers={
            "Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM})
        assert status == 303 and headers["Location"] == "/saved"
        status, _, body = _request(port, "GET", "/saved")
        assert status == 200 and "Saved. The live defaults are now:" in body
        assert config.read_saved_live_defaults() == LiveSettings(False, (0.0, 0.5), 0.8, 0.1,
                                                                 0.2)

    def test_every_route_sends_the_six_headers(self, live_server):
        port = live_server
        expected = dict(defaults_server._response_headers(defaults_server._Response(200, "")))
        answers = [
            _request(port, "GET", "/"),
            _request(port, "GET", "/confirm?" + _query()),
            _request(port, "GET", "/confirm?k"),
            _request(port, "GET", "/saved"),
            _request(port, "GET", "/nope"),
            _request(port, "GET", "/", host="evil.test"),
            _request(port, "POST", "/confirm", body=b"", headers={"Content-Type": FORM}),
        ]
        assert [status for status, _, _ in answers] == [200, 200, 400, 404, 404, 403, 403]
        for _, headers, _ in answers:
            for name, value in expected.items():
                assert headers[name] == value, name

    @pytest.mark.parametrize("length, status", [
        (None, 411), ("abc", 400), ("-1", 400), ("+5", 400),
        (str(config.DEFAULTS_SERVER_MAX_REQUEST_BYTES + 1), 413),
        # More digits than int() reads from a string
        ("9" * 5000, 413),
    ])
    def test_a_bad_content_length_is_refused(self, live_server, length, status):
        headers = {"Origin": f"http://127.0.0.1:{live_server}", "Content-Type": FORM}
        if length is not None:
            headers["Content-Length"] = length
        got, answer_headers, _ = _request(live_server, "POST", "/confirm", headers=headers)
        assert got == status
        assert answer_headers["X-Frame-Options"] == "DENY"

    def test_two_content_lengths_are_refused(self, live_server):
        headers = {"Content-Type": FORM, "Content-Length": "0"}
        status, _, _ = _request(live_server, "POST", "/confirm", body=b"", headers=headers)
        assert status == 400

    def test_a_path_over_the_limit_is_refused(self, live_server):
        target = "/confirm?" + "k=" + "1" * config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        assert _request(live_server, "GET", target)[0] == 413

    @pytest.mark.parametrize("method", ["PUT", "HEAD"])
    def test_other_methods_get_501_with_the_six_headers(self, live_server, method):
        status, headers, body = _request(live_server, method, "/confirm")
        assert status == 501
        expected = dict(defaults_server._response_headers(defaults_server._Response(501, "")))
        for name, value in expected.items():
            assert headers[name] == value, name
        assert (body == "") is (method == "HEAD")

    @pytest.mark.parametrize("data, status", [
        (b"GARBAGE\r\n", 400),                          # a request line it cannot read
        (b"GET /" + b"a" * (65537 - 5), 414),           # over the base class's 65536 bytes
        (b"GET /\r\n\r\n", 403),                        # HTTP/0.9: no Host header
    ])
    def test_a_request_the_http_layer_refuses_still_gets_the_six_headers(
            self, live_server, data, status):
        got, headers, _ = _raw_request(live_server, data)
        assert got == status
        expected = dict(defaults_server._response_headers(defaults_server._Response(got, "")))
        for name, value in expected.items():
            assert headers[name] == value, name

    def test_a_silent_connection_does_not_hold_the_server(self, live_server):
        silent = socket.create_connection(("127.0.0.1", live_server), timeout=10)
        try:
            started = time.monotonic()
            status, _, _ = _request(live_server, "GET", "/")
            assert status == 200
            assert time.monotonic() - started < 5
        finally:
            silent.close()


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    """
    Run defaults_server.main with the server and the browser replaced.

    config.PROJECT_ROOT points at tmp_path, so the log and any dashboard
    live there.

    Args:
        tmp_path (Path): The test's directory.
        monkeypatch (pytest.MonkeyPatch): Replaces the server, the browser
            and PROJECT_ROOT.

    Yields:
        dict: "servers" (each fake server made), "opened" (each address the
            browser was asked to open), "fail" (set it to an exception for
            the bind to raise), "interrupt" (set it to True for
            serve_forever to raise KeyboardInterrupt) and "run" (call it
            with the arguments to run main; see _run_main).
    """
    state = {"servers": [], "opened": [], "fail": None, "interrupt": False}

    class FakeServer:
        """Records how it was made and used; serves nothing."""

        def __init__(self, address, handler):
            """
            Record the address and handler, or fail the bind as the test chose.

            Args:
                address (tuple[str, int]): Where main asked to listen.
                handler (type): The request handler class.

            Raises:
                OSError: state["fail"], when a test set it.
            """
            if state["fail"] is not None:
                raise state["fail"]
            self.address, self.handler = address, handler
            self.served = self.closed = False
            state["servers"].append(self)

        def serve_forever(self):
            """
            Note that main served, and return at once.

            Returns:
                None

            Raises:
                KeyboardInterrupt: When state["interrupt"] is True, as Ctrl-C would.
            """
            self.served = True
            if state["interrupt"]:
                raise KeyboardInterrupt

        def server_close(self):
            """
            Note that main closed the server.

            Returns:
                None
            """
            self.closed = True

    monkeypatch.setattr(defaults_server, "HTTPServer", FakeServer)
    monkeypatch.setattr(defaults_server.webbrowser, "open", state["opened"].append)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    state["run"] = lambda argv: _run_main(argv, tmp_path)
    yield state


def _run_main(argv: list[str], tmp_path) -> str:
    """
    Run defaults_server.main with the root logger's handlers set aside.

    main configures logging with basicConfig, which does nothing while the
    root logger has handlers (pytest adds its own during each test), so they
    are set aside for the call and put back after it, whatever it raises;
    main's own handlers are closed first, so the log file is complete.

    Args:
        argv (list[str]): main's arguments.
        tmp_path (Path): The test's directory (config.PROJECT_ROOT).

    Returns:
        str: The log file's text ("" when nothing was logged).

    Raises:
        BaseException: Whatever main raises (SystemExit on a busy port),
            after the handlers are put back.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        defaults_server.main(argv)
    finally:
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
    log = tmp_path / "kalshi_defaults_server.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


class TestMain:
    """main(): bind, log, open the right page, serve until Ctrl-C."""

    def test_seed_opens_the_seed_page(self, run_main):
        log = run_main["run"](["--seed"])
        url = f"http://127.0.0.1:{PORT}/confirm?{defaults_server._seed_query()}"
        assert run_main["opened"] == [url]
        [server] = run_main["servers"]
        assert server.address == (config.DEFAULTS_SERVER_HOST, PORT)
        assert server.handler is defaults_server._Handler
        assert server.served and server.closed
        assert isinstance(server.defaults_app, defaults_server._App)
        assert server.defaults_app.port == PORT
        assert url in log
        assert "No live defaults are saved" in log

    def test_no_browser_opens_nothing(self, run_main):
        log = run_main["run"](["--seed", "--no-browser"])
        assert run_main["opened"] == []
        assert defaults_server._seed_query() in log

    def test_it_opens_the_dashboard_when_there_is_one(self, run_main, tmp_path):
        log = run_main["run"]([])
        assert run_main["opened"] == []
        dashboard = tmp_path / config.DASHBOARD_FILENAME
        assert f"No backtest dashboard at {dashboard}" in log
        dashboard.write_text("<html></html>", encoding="utf-8")
        run_main["run"]([])
        assert run_main["opened"] == [dashboard.as_uri()]

    def test_it_logs_the_defaults_in_force(self, run_main):
        saved = _save(config.LIVE_DEFAULTS_SEED)
        log = run_main["run"](["--no-browser"])
        assert f"Live defaults in force: {saved.origin}" in log

    def test_it_logs_a_refused_file(self, run_main):
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        log = run_main["run"](["--no-browser"])
        assert "The saved live defaults are refused" in log

    def test_a_busy_port_exits_2_before_logging(self, run_main, tmp_path, capsys):
        run_main["fail"] = OSError(errno.EADDRINUSE, "Address already in use")
        with pytest.raises(SystemExit) as exit_info:
            run_main["run"](["--seed"])
        assert exit_info.value.code == 2
        assert f"port {PORT} is in use" in capsys.readouterr().err
        assert run_main["opened"] == []
        assert not (tmp_path / "kalshi_defaults_server.log").exists()

    def test_another_bind_error_is_raised(self, run_main):
        run_main["fail"] = PermissionError(errno.EACCES, "Permission denied")
        with pytest.raises(PermissionError):
            run_main["run"]([])

    def test_ctrl_c_stops_it_cleanly(self, run_main):
        run_main["interrupt"] = True
        log = run_main["run"](["--no-browser"])
        [server] = run_main["servers"]
        assert server.closed
        assert "Defaults server stopped" in log


class TestIsolation:
    """A person runs the defaults server; no module of the package imports it."""

    def test_no_other_module_imports_it(self):
        names = [m.name for m in pkgutil.iter_modules(kalshi_betting.__path__)]
        assert "defaults_server" in names
        for name in names:
            if name == "defaults_server":
                continue
            module = importlib.import_module(f"kalshi_betting.{name}")
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.Import):
                    imported = {a.name.split(".")[-1] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported = {(node.module or "").split(".")[-1]}
                    imported |= {a.name for a in node.names}
                else:
                    continue
                assert "defaults_server" not in imported, (
                    f"kalshi_betting/{name}.py imports defaults_server")

    def test_it_imports_config_alone_from_the_package(self):
        tree = ast.parse(inspect.getsource(defaults_server))
        project = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or (
                    node.module or "").startswith("kalshi_betting")):
                if node.module:
                    project.add(node.module.split(".")[-1])
                else:
                    project |= {a.name for a in node.names}
        assert project == {"config"}
