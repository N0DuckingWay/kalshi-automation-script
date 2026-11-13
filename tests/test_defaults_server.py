"""
File: test_defaults_server.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for defaults_server.py, the local web server whose pages save the
    live trading defaults and start live trading runs with them. Most tests
    call _App.handle directly with a _Request, so they need no socket: the
    Host check, the index page, the trading page (every changed row
    highlighted, warnings, escaping, its three scripts and their
    Content-Security-Policy hashes, who may frame it, the buttons in their
    fixed order with placeholders for those that do not apply, and its
    dropdown bar: its eleven controls in the backtest page's order and
    words, their choices, the names file its menus read, and a round trip of
    every choice through the server), the page with no address fields (the
    defaults in force, or no buttons when a link cannot carry them as they
    are), every refused proposal, the save and each check in
    front of it (Origin, content type, fingerprint, nonce and token, a stale
    page, a failed write, a save that changes nothing and so writes
    nothing), the redirect of the retired /trade address, the three actions
    (save, trade, dry run) and every check before a run starts (the run's
    blockers, the last real-money run's attention banner and its
    acknowledgement, a nonce used twice), the saved page, the run page (each
    rule of its outcome table, read from hand-written run folders and from
    a stand-in process, and the message it sends a framing page), /checkout,
    the seed, and a saved file an older version wrote (never traded,
    offered in its new form for one save). A POST body is always built from
    the hidden inputs and one button of the page the server rendered, never
    by hand, except where a test signs or crafts a request itself to reach a
    path no rendered page leads to; a dropdown bar request is built as a
    browser builds a GET form from the rendered bar. The process starter is
    a stand-in that records each start and starts nothing (_Starter); no
    test starts main.py. Three classes run the page scripts themselves (the
    arming script, the bar script and the notifier) under node or macOS's
    jsc with a stand-in page, one runs the real handler over a loopback
    socket (skipped only if the sandbox refuses to bind a port), including
    one round trip whose stand-in starts a tiny Python child in place of
    main.py, one runs main() with the server and the browser replaced (which
    page a start opens, and what a start does when its port is taken), one
    runs main() against real loopback listeners holding its port (this
    checkout's server, another checkout's, and listeners that are not a
    defaults server) without ever binding it, and one checks what this
    module imports and where it starts a process.

Dependencies:
    Imports kalshi_betting.config (the saved-defaults helpers and constants),
    kalshi_betting.run_lock (a lock held in-process) and
    kalshi_betting.defaults_server. tests/conftest.py points
    config.LIVE_DEFAULTS_FILE at each test's own tmp_path, so every save here
    lands there, and config.LIVE_RUN_LOCK_FILE and config.LIVE_RUNS_DIR
    there too; an autouse fixture here also points config.PROJECT_ROOT
    there, so the scheduler's state file the attention banner reads is the
    test's own.

Notes:
    The socket tests bind 127.0.0.1 on a free port; under a sandbox that
    forbids local binding they skip, and CI runs them. No test asks the real
    port (DEFAULTS_SERVER_PORT) anything: the tests of a taken port either
    stand in for the /checkout question (run_main) or point the server's
    port at a listener of their own (busy_port). The page scripts' tests
    skip when neither node nor jsc is present.
"""
import ast
import dataclasses
import errno
import fcntl
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
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from base64 import b64encode
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

import kalshi_betting
from kalshi_betting import config, defaults_server, run_lock
from kalshi_betting.config import LiveSettings

PORT = config.DEFAULTS_SERVER_PORT
HOST = f"127.0.0.1:{PORT}"
ORIGIN = f"http://127.0.0.1:{PORT}"
FORM = "application/x-www-form-urlencoded"
KEY = b"k" * 32
# The real process starter, captured before any test could patch it (the
# socket round trip builds its stand-in from it)
_REAL_POPEN = subprocess.Popen
# A source note of the dashboard's shape
DASHBOARD_NOTE = "backtest dashboard for 2025-09-24 to 2026-09-27"
# The eleven rows of a comparison, in their order on the page
LABELS = ["tier floors", "spread band", "k", "per-trade cap",
          "exclude categories", "exclude tags", "rebalance", "min price movement",
          "rebalance until", "sell at", "min days to maturity"]
# A proposal's fields: the seed's values, spelled as the dashboard spells them
_BASE = {"tier_floors": "off", "spread_min": "0", "spread_max": "0.5", "k": "0.8",
         "size_cap": "0.1"}
# What _BASE proposes with nothing saved: the fields it leaves out take the
# seed's values (rebalancing on with no minimum, no selling)
_BASE_SETTINGS = LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                              size_cap=0.1, rebalance=True)
# The ASCII digits as Arabic-Indic and as full-width digits (str.translate tables)
_ARABIC_INDIC = str.maketrans("0123456789", "".join(chr(0x0660 + i) for i in range(10)))
_FULL_WIDTH = str.maketrans("0123456789", "".join(chr(0xFF10 + i) for i in range(10)))
# The trading page's button labels, in page order
_CONFIRM_LABELS = ["Dry run (no orders; defaults unchanged)", "Confirm and save",
                   "Confirm and trade"]
# Each armed button's id and the action it posts
_ID_ACTION = {"confirm-dry-run": "dry_run", "confirm": "save", "confirm-trade": "trade"}
# The three scripts the trading page carries, in page order
_TRADING_SCRIPTS = [defaults_server._CONFIRM_JS, defaults_server._BAR_JS,
                    defaults_server._NOTIFY_JS]


@pytest.fixture(autouse=True)
def _checkout(tmp_path, monkeypatch):
    """
    Point config.PROJECT_ROOT at the test's tmp_path.

    The attention banner reads the scheduler's state file from PROJECT_ROOT,
    and a run starts from it: here both are the test's own, never the
    checkout's.

    Args:
        tmp_path (Path): The test's directory.
        monkeypatch (pytest.MonkeyPatch): Restores the real root afterwards.
    """
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)


class _PageParser(HTMLParser):
    """
    Read what a test needs off a rendered page.

    Collects the forms (method, action), the body's attributes, the text of
    every script, the links, each comparison row's class and cell texts by
    its data-setting label, and, for the form that posts, its hidden inputs
    in page order, its checkboxes and its buttons' attributes and labels in
    page order. The dropdown bar (the form tb-form, which only asks for the
    page again) is read apart, into the bar_* lists, so the posting form's
    fields are exactly what a click on one of its buttons sends.
    """

    def __init__(self) -> None:
        """
        Start with nothing collected, character references decoded in text.

        Returns:
            None
        """
        super().__init__(convert_charrefs=True)
        self.forms: list[tuple[str | None, str | None]] = []
        self.body: dict[str, str | None] = {}
        self.hidden: list[tuple[str, str]] = []
        self.checkboxes: list[dict] = []
        self.buttons: list[dict] = []
        self.labels: list[str] = []
        self.links: list[str] = []
        self.rows: dict[str, str | None] = {}
        self.cells: dict[str, list[str]] = {}
        self.scripts: list[str] = []
        # The bar: its form's attributes, its named controls in page order (each
        # a dict: tag, the element's attributes, and a select's options as
        # dicts of value, text and selected), the labels before its dropdowns
        # and menus, and its buttons
        self.bar_form: dict[str, str | None] | None = None
        self.bar_controls: list[dict] = []
        self.bar_labels: list[str] = []
        self.bar_buttons: list[dict] = []
        self.bar_notes: list[str] = []
        self._row: str | None = None
        self._in_script = False
        self._in_button = False
        self._in_bar = False
        self._option: dict | None = None
        self._label: str | None = None
        self._note: bool = False
        self._in_names = False

    def handle_starttag(self, tag, attrs) -> None:
        """
        Record a form, input, select, option, button, link, row, cell or script as it opens.

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
            self._in_bar = attributes.get("id") == "tb-form"
            if self._in_bar:
                self.bar_form = attributes
        elif tag == "body":
            self.body = attributes
        elif self._in_bar:
            self._bar_starttag(tag, attributes)
        elif tag == "input" and attributes.get("type") == "hidden":
            self.hidden.append((attributes["name"], attributes["value"]))
        elif tag == "input" and attributes.get("type") == "checkbox":
            self.checkboxes.append(attributes)
        elif tag == "button":
            self.buttons.append(attributes)
            self.labels.append("")
            self._in_button = True
        elif tag == "a":
            self.links.append(attributes.get("href"))
        elif tag == "tr" and "data-setting" in attributes:
            self._row = attributes["data-setting"]
            self.rows[self._row] = attributes.get("class")
            self.cells[self._row] = []
        elif tag == "td" and self._row is not None:
            self.cells[self._row].append("")
        elif tag == "script":
            self._in_script = True
            self.scripts.append("")

    def _bar_starttag(self, tag: str, attributes: dict) -> None:
        """
        Record one element of the dropdown bar as it opens.

        Args:
            tag (str): The tag's name, lower-cased.
            attributes (dict): Its attributes, values unescaped.

        Returns:
            None
        """
        if tag in ("select", "input"):
            self.bar_controls.append({"tag": tag, **attributes, "options": []})
            if tag == "select" and self._label is not None:
                self.bar_labels.append(self._label)
                self._label = None
        elif tag == "option":
            self._option = {"value": attributes["value"], "text": "",
                            "selected": "selected" in attributes}
            self.bar_controls[-1]["options"].append(self._option)
        elif tag == "label" and not self._in_names:
            self._label = ""
        elif tag == "span" and attributes.get("class") == "menu":
            self._label = ""
        elif tag == "span" and attributes.get("class") == "note":
            self._note = True
            self.bar_notes.append("")
        elif tag == "details" and self._label is not None:
            self.bar_labels.append(self._label)
            self._label = None
        elif tag == "div" and attributes.get("class") == "names":
            self._in_names = True
        elif tag == "button":
            self.bar_buttons.append(attributes)

    def handle_endtag(self, tag) -> None:
        """
        Close a form, row, button, option or script.

        Args:
            tag (str): The tag's name, lower-cased.

        Returns:
            None
        """
        if tag == "form":
            self._in_bar = False
        elif tag == "div":
            self._in_names = False
        elif tag == "tr":
            self._row = None
        elif tag == "script":
            self._in_script = False
        elif tag == "button":
            self._in_button = False
        elif tag == "option":
            self._option = None
        elif tag == "span":
            self._note = False

    def handle_data(self, data) -> None:
        """
        Add text to the open script, button, table cell, or bar option, label or note.

        Args:
            data (str): The text.

        Returns:
            None
        """
        if self._in_script:
            self.scripts[-1] += data
        elif self._option is not None:
            self._option["text"] += data
        elif self._in_bar and self._label is not None:
            self._label += data
        elif self._in_bar and self._note:
            self.bar_notes[-1] += data
        elif self._in_button:
            self.labels[-1] += data
        elif self._row is not None and self.cells[self._row]:
            self.cells[self._row][-1] += data

    def bar_select(self, name: str) -> dict:
        """
        Find one of the bar's dropdowns by its name (or, for the spread band, its id).

        Args:
            name (str): The select's name, or "tb-band".

        Returns:
            dict: The select: its attributes and its options.
        """
        [select] = [c for c in self.bar_controls if c["tag"] == "select"
                    and (c.get("name") == name or c.get("id") == name)]
        return select

    def bar_boxes(self, name: str) -> list[dict]:
        """
        List the bar's check boxes of one menu, in page order.

        Args:
            name (str): "exclude_category" or "exclude_tag".

        Returns:
            list[dict]: Each box's attributes.
        """
        return [c for c in self.bar_controls
                if c.get("type") == "checkbox" and c.get("name") == name]


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


def _click(page, button_id: str, *, ack: bool = False) -> list[tuple[str, str]]:
    """
    Build the form a browser posts when one of a page's buttons is clicked.

    Args:
        page (str | _PageParser): The rendered page, or its parse.
        button_id (str): The id of the button clicked; it must be a real
            button of the page, with a name and a value.
        ack (bool): Whether the attention box is ticked (it must be on the page).

    Returns:
        list[tuple[str, str]]: The hidden inputs in page order, then the
            ticked box, then the button's (name, value).
    """
    parsed = _parse(page) if isinstance(page, str) else page
    [button] = [b for b in parsed.buttons if b.get("id") == button_id]
    form = list(parsed.hidden)
    if ack:
        [box] = parsed.checkboxes
        form.append((box["name"], box["value"]))
    return form + [(button["name"], button["value"])]


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


class _FakeProcess:
    """A stand-in for a started process: its pid and its return code (None while it runs)."""

    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        """
        Make a process that has or has not ended.

        Args:
            pid (int): Its process id.
            returncode (int | None): Its exit code; None while it runs.
        """
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        """
        Report the return code, as Popen.poll does.

        Returns:
            int | None: The return code, or None while it runs.
        """
        return self.returncode


class _Starter:
    """
    Stands in for subprocess.Popen as a server's process starter: records each start, starts nothing.

    It reads the run's run.json as it is at the moment of the start, so a
    test can check it was written first, and it can write a result file or
    fail the start as a test asks.
    """

    def __init__(self, *, returncode: int | None = None, result: dict | None = None,
                 fail: BaseException | None = None) -> None:
        """
        Set how each start behaves.

        Args:
            returncode (int | None): The return code its processes report;
                None keeps them running.
            result (dict | None): A result record to write at the run's
                --result-file path; None writes none.
            fail (BaseException | None): Raised instead of starting.
        """
        self.returncode = returncode
        self.result = result
        self.fail = fail
        self.calls: list[tuple[list[str], dict]] = []
        self.run_json: list[dict] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, argv, **kwargs) -> _FakeProcess:
        """
        Record one start, as the server makes it.

        Args:
            argv (list[str]): The command line.
            **kwargs: Popen's keyword arguments.

        Returns:
            _FakeProcess: The stand-in process.

        Raises:
            BaseException: self.fail, when set.
        """
        if self.fail is not None:
            raise self.fail
        folder = Path(argv[argv.index("--result-file") + 1]).parent
        self.run_json.append(json.loads((folder / "run.json").read_text(encoding="utf-8")))
        self.calls.append((list(argv), kwargs))
        if self.result is not None:
            (folder / "result.json").write_text(json.dumps(self.result), encoding="utf-8")
        process = _FakeProcess(4242 + len(self.processes), self.returncode)
        self.processes.append(process)
        return process


def _app(key: bytes = KEY, start_process=None) -> defaults_server._App:
    """
    Make the application for the default port.

    Args:
        key (bytes): The token key.
        start_process: Its process starter; None (default) refuses every run.

    Returns:
        defaults_server._App: The application.
    """
    return defaults_server._App(PORT, key=key, start_process=start_process)


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


def _page_form(app, query: str, button_id: str = "confirm", *,
               ack: bool = False) -> list[tuple[str, str]]:
    """
    Open the confirmation page for a query and click one of its buttons.

    Args:
        app (defaults_server._App): The application.
        query (str): The query string.
        button_id (str): The button clicked (Confirm and save by default).
        ack (bool): Whether the attention box is ticked.

    Returns:
        list[tuple[str, str]]: The form the browser posts.
    """
    response = _get(app, f"/confirm?{query}")
    assert response.status == 200, response.body
    return _click(response.body, button_id, ack=ack)


def _home_form(app, button_id: str = "confirm-trade", *,
               ack: bool = False) -> list[tuple[str, str]]:
    """
    Open the trading page with no address fields (the defaults in force) and click one of its buttons.

    Args:
        app (defaults_server._App): The application.
        button_id (str): The button clicked (Confirm and trade by default).
        ack (bool): Whether the attention box is ticked.

    Returns:
        list[tuple[str, str]]: The form the browser posts (to /confirm).
    """
    response = _get(app, "/confirm")
    assert response.status == 200, response.body
    return _click(response.body, button_id, ack=ack)


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


def _run_id(response: defaults_server._Response) -> str:
    """
    Read the run id from a POST's redirect to its run page.

    Args:
        response (defaults_server._Response): A 303 to /runs/<id>.

    Returns:
        str: The id.
    """
    assert response.status == 303, response.body
    assert response.location.startswith("/runs/"), response.location
    return response.location[len("/runs/"):]


def _run_folder(run_id: str) -> Path:
    """
    Find the one folder of a run.

    Args:
        run_id (str): The run's id.

    Returns:
        Path: Its folder under config.LIVE_RUNS_DIR.
    """
    [folder] = [p for p in config.LIVE_RUNS_DIR.iterdir() if p.name.endswith(f"-{run_id}")]
    return folder


def _result(**changes) -> dict:
    """
    Build a run result record, as main.py --result-file writes it, with changes.

    Args:
        **changes: Fields to set.

    Returns:
        dict: A real-money run that ended with exit 0 and no trades, changed.
    """
    record = {
        "format": config.LIVE_RUN_RESULT_FORMAT, "mode": "prod", "dry_run": False,
        "started_at": "2026-09-29T16:00:00Z", "finished_at": "2026-09-29T16:05:00Z",
        "exit_code": 0, "settings": "tier floors off", "defaults": "live_defaults.json",
        "message": "", "balance_before": 100.0, "balance_after": 100.0,
        "portfolio_value_before": 100.0,
        "submission_started": False, "trades": [], "warnings": [], "warnings_dropped": 0,
        "error": None,
    }
    record.update(changes)
    return record


def _trade(status: str, **changes) -> dict:
    """
    Build one pair of a run result, with changes.

    Args:
        status (str): The pair's status.
        **changes: Fields to set.

    Returns:
        dict: A time-series pair of two legs, with the given status.
    """
    record = {
        "status": status, "error": None, "pair_type": "time_series",
        "title": "Will it rain by Friday?",
        "a": {"ticker": "RAIN-A", "market": "Rain by Oct 1", "side": "yes", "count": 30,
              "price": 0.21},
        "b": {"ticker": "RAIN-B", "market": "Rain by Oct 8", "side": "no", "count": 30,
              "price": 0.5},
        "cost_with_fees": 21.9, "profit_if_won": 7.4,
    }
    record.update(changes)
    return record


def _disk_run(run_id: str = "0123456789abcdef", *, started: str = "20260929T160000Z",
              result=None, run_json: dict | None = None, dry_run: bool = False,
              output: str = "") -> Path:
    """
    Write a run's folder by hand, as a server (maybe an earlier one) left it.

    Args:
        run_id (str): The run's id (16 hex digits).
        started (str): The start time at the head of the folder's name.
        result (dict | bytes | None): result.json, as a record or raw bytes; None writes none.
        run_json (dict | None): run.json; None writes the usual record.
        dry_run (bool): The usual record's dry_run.
        output (str): output.log's text.

    Returns:
        Path: The folder.
    """
    folder = config.LIVE_RUNS_DIR / f"{started}-{run_id}"
    folder.mkdir(parents=True)
    if run_json is None:
        run_json = {"run_id": run_id, "dry_run": dry_run, "argv": [],
                    "settings": "tier floors off | k 0.8",
                    "started_at": datetime.strptime(started, "%Y%m%dT%H%M%SZ").strftime(
                        "%Y-%m-%dT%H:%M:%SZ"),
                    "saved_note": "Saved and verified.", "pid": 555}
    (folder / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (folder / "output.log").write_text(output, encoding="utf-8")
    if isinstance(result, bytes):
        (folder / "result.json").write_bytes(result)
    elif result is not None:
        (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return folder


def _memory_run(app, *, returncode: int | None, result=None, dry_run: bool = False,
                output: str = "", run_id: str = "fedcba9876543210",
                started_at: datetime | None = None) -> str:
    """
    Give an application a run it started, with a stand-in process, and write its folder.

    Args:
        app (defaults_server._App): The application.
        returncode (int | None): The process's exit code; None while it runs.
        result (dict | bytes | None): result.json; None writes none.
        dry_run (bool): Whether it is a dry run.
        output (str): output.log's text.
        run_id (str): Its id.
        started_at (datetime | None): When it started; None is now.

    Returns:
        str: The run's id.
    """
    folder = _disk_run(run_id, result=result, dry_run=dry_run, output=output)
    app._runs[run_id] = defaults_server._Run(
        run_id=run_id, folder=folder, process=_FakeProcess(returncode=returncode),
        dry_run=dry_run, settings_text="tier floors off | k 0.8",
        started_at=started_at or datetime.now(UTC), saved_note="Saved and verified.",
        pid=4242)
    return run_id


def _printed_at(folder: Path, when: str) -> Path:
    """
    Set when a run folder's output.log was last written, which places a run with no result in time.

    Args:
        folder (Path): The run's folder.
        when (str): The time, ISO 8601 with its offset.

    Returns:
        Path: The folder.
    """
    stamp = datetime.fromisoformat(when).timestamp()
    os.utime(folder / "output.log", (stamp, stamp))
    return folder


def _attention_result() -> Path:
    """
    Leave a finished real-money run that needed attention (exit 20) in the run folders.

    Returns:
        Path: Its folder.
    """
    return _disk_run("aaaaaaaaaaaaaaaa", result=_result(
        exit_code=config.EXIT_TRADES_NEED_ATTENTION,
        trades=[_trade("manual_review", error="NO leg: unknown fill")]))


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

    @pytest.mark.parametrize("target", ["/checkout", "/trade", "/runs/0123456789abcdef"])
    def test_every_new_route_checks_the_host(self, target):
        response = _get(_app(), target, host="evil.test:8765")
        assert response.status == 403
        assert "project_root" not in response.body


class TestIndexPage:
    """GET / says what the server is for and lists the newest runs, with no form and no script."""

    def test_it_names_where_to_trade(self):
        response = _get(_app(), "/")
        assert response.status == 200
        page = _parse(response.body)
        assert page.forms == [] and page.scripts == [] and page.buttons == []
        assert ("Trade from the live dashboard's Trading tab (<code>./start_dashboard.sh"
                "</code> starts both), or from this server's own trading page below: it opens "
                "on the live defaults in force") in html.unescape(response.body)
        assert f"reads its defaults from <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" \
            in response.body
        assert "This writes" not in response.body
        assert "No run has been started from this checkout's server." in response.body
        # It never names the retired buttons, the backtest file or the /trade address
        assert "Save as live defaults" not in response.body
        assert "Trade using defaults" not in response.body and "/trade" not in page.links

    def test_it_links_the_trading_page_and_the_seed(self):
        # Whatever backtest page is in the checkout, which the server never reads
        (config.PROJECT_ROOT / config.DASHBOARD_FILENAME).write_text("<html></html>",
                                                                     encoding="utf-8")
        response = _get(_app(), "/")
        page = _parse(response.body)
        assert page.links[:2] == ["/confirm", f"/confirm?{defaults_server._seed_query()}"]
        assert ">Trading page</a>" in response.body
        assert "Start from the seed values" in response.body
        # The page a save redirects to is never linked
        assert "/saved" not in page.links

    def test_the_newest_runs_are_listed_first_with_their_state(self):
        for day in range(1, 13):
            _disk_run(f"{day:016x}", started=f"202609{day:02d}T120000Z",
                      result=_result(exit_code=0) if day == 11 else None,
                      dry_run=day == 11)
        # The newest is still going: its output is locked, as its process holds it
        newest = config.LIVE_RUNS_DIR / f"20260912T120000Z-{12:016x}"
        with (newest / "output.log").open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            response = _get(_app(), "/")
        page = _parse(response.body)
        runs = [link for link in page.links if link.startswith("/runs/")]
        assert runs == [f"/runs/{day:016x}" for day in range(12, 2, -1)]
        assert len(runs) == config.DEFAULTS_SERVER_INDEX_RUNS
        text = html.unescape(response.body)
        assert "2026-09-12 12:00:00 UTC</a> — real orders — running" in text
        assert "2026-09-11 12:00:00 UTC</a> — dry run — finished (exit 0)" in text
        assert "2026-09-10 12:00:00 UTC</a> — real orders — ended without a result" in text

    def test_anything_else_is_404(self):
        assert _get(_app(), "/nope").status == 404
        assert _post(_app(), [], target="/").status == 404
        assert _post(_app(), [], target="/saved").status == 404


class TestCheckout:
    """GET /checkout answers which checkout this server serves, and the code it loaded, as JSON."""

    def test_it_names_the_resolved_project_root(self):
        response = _get(_app(), "/checkout")
        assert response.status == 200
        assert json.loads(response.body) == {
            "project_root": str(config.PROJECT_ROOT.resolve()),
            "code": defaults_server._LOADED_CODE}
        headers = dict(defaults_server._response_headers(response))
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert "X-Frame-Options" not in headers
        assert headers["Content-Security-Policy"] == defaults_server._CSP


class TestPortsApart:
    """The live dashboard's two ports are other web origins than this server's."""

    def test_the_live_dashboard_s_tab_page_has_a_port_of_its_own(self):
        assert config.LIVE_DASHBOARD_PORT != config.DEFAULTS_SERVER_PORT

    def test_the_live_dashboard_s_backtest_page_has_a_port_of_its_own(self):
        assert config.LIVE_BACKTEST_PORT != config.DEFAULTS_SERVER_PORT
        assert config.LIVE_BACKTEST_PORT != config.LIVE_DASHBOARD_PORT


class TestCodeFingerprint:
    """_code_fingerprint (a SHA-256 over the package's .py files) and _LOADED_CODE."""

    def test_it_is_the_package_s_python_files_now(self):
        folder = Path(defaults_server.__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(folder.glob("*.py")):
            data = path.read_bytes()
            digest.update(path.name.encode("utf-8") + b"\0")
            digest.update(len(data).to_bytes(8, "big") + data)
        assert defaults_server._code_fingerprint() == digest.hexdigest()

    def test_the_loaded_code_is_the_package_as_this_process_imported_it(self):
        # No test edits the package, so the fingerprint taken at import still matches
        assert defaults_server._LOADED_CODE == defaults_server._code_fingerprint()
        assert re.fullmatch(r"[0-9a-f]{64}", defaults_server._LOADED_CODE)

    def test_a_change_to_a_python_file_changes_it_and_other_files_do_not(self, tmp_path,
                                                                         monkeypatch):
        package = tmp_path / "package"
        package.mkdir()
        (package / "a.py").write_text("x = 1\n", encoding="utf-8")
        (package / "b.py").write_text("y = 2\n", encoding="utf-8")
        monkeypatch.setattr(defaults_server, "__file__", str(package / "defaults_server.py"))
        before = defaults_server._code_fingerprint()
        (package / "notes.md").write_text("not code\n", encoding="utf-8")
        assert defaults_server._code_fingerprint() == before
        (package / "b.py").write_text("y = 3\n", encoding="utf-8")
        assert defaults_server._code_fingerprint() != before

    def test_moving_bytes_between_files_changes_it(self, tmp_path, monkeypatch):
        # Each file is hashed with its name and length, so the same bytes split
        # differently across files do not collide
        package = tmp_path / "package"
        package.mkdir()
        monkeypatch.setattr(defaults_server, "__file__", str(package / "defaults_server.py"))
        (package / "a.py").write_text("xy", encoding="utf-8")
        (package / "b.py").write_text("z", encoding="utf-8")
        first = defaults_server._code_fingerprint()
        (package / "a.py").write_text("x", encoding="utf-8")
        (package / "b.py").write_text("yz", encoding="utf-8")
        assert defaults_server._code_fingerprint() != first

    def test_an_unreadable_file_does_not_raise(self, tmp_path, monkeypatch):
        package = tmp_path / "package"
        package.mkdir()
        # A folder named like a module cannot be read as a file
        (package / "odd.py").mkdir()
        monkeypatch.setattr(defaults_server, "__file__", str(package / "defaults_server.py"))
        assert re.fullmatch(r"[0-9a-f]{64}", defaults_server._code_fingerprint())


class TestConfirmPage:
    """GET /confirm: the defaults in force beside the proposed ones."""

    def test_with_a_file_saved_only_the_changed_rows_are_highlighted(self):
        saved = _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.8,
                                   size_cap=0.1), source=DASHBOARD_NOTE)
        response = _get(_app(), "/confirm?" + _query(source=DASHBOARD_NOTE))
        assert response.status == 200
        page = _parse(response.body)
        assert list(page.rows) == LABELS
        changed = {label for label, css in page.rows.items() if css == "changed"}
        assert changed == {"tier floors", "spread band"}
        assert all(css == "same" for label, css in page.rows.items() if label not in changed)
        assert response.body.count('<span class="tag">changed</span>') == 2
        assert "2 of 11 settings change." in response.body
        assert "Overwrite the live trading defaults?" in response.body
        assert f"This writes <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" \
            in response.body
        assert ("Nothing is saved or traded until you click one of the buttons below; "
                "closing this tab cancels.") in response.body
        assert saved.origin in response.body
        assert f"New defaults from: {DASHBOARD_NOTE}" in response.body
        # Current and new values, in the "Live settings:" line's words
        assert page.cells["tier floors"][:3] == ["tier floors", "on", "off"]
        assert page.cells["spread band"][:3] == ["spread band", "none", "0-0.5"]

    def test_the_changed_value_is_bold(self):
        _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.8,
                           size_cap=0.1))
        body = _get(_app(), "/confirm?" + _query()).body
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
        assert "11 of 11 settings change." in response.body
        # The fields the link leaves out take the seed's values
        assert page.cells["rebalance"][2] == "yes"
        assert page.cells["min price movement"][2] == "any"
        assert page.cells["sell at"][2] == "off"
        assert page.cells["min days to maturity"][2] == "any"

    def test_the_exposure_warning_shows(self):
        response = _get(_app(), "/confirm?" + _query(k="0.4", size_cap="1"))
        warnings = config.live_rule_warnings(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5),
                                                          interval_discount=0.4, size_cap=1.0))
        assert warnings
        for sentence in warnings:
            assert f'<p class="warn">Warning: {sentence}.</p>' in response.body

    def test_no_warning_for_the_seed(self):
        assert 'class="warn"' not in _get(_app(), "/confirm?" + _query()).body

    # Saved defaults that rebalance: k 0.8, a 20% per-trade cap
    _REBALANCE_SAVED = LiveSettings(tier_floors=False, spread_band=(0.0, 0.5),
                                    interval_discount=0.8, size_cap=0.2, rebalance=True)

    @pytest.mark.parametrize("link, named", [
        ({"size_cap": "0.1"}, "the per-trade cap goes from 20% to 10%"),
        ({"k": "0.9", "size_cap": "0.2"},
         "k goes from 0.8 to 0.9, which lowers a time-series pair&#x27;s chance of paying"),
    ], ids=["per-trade cap", "k"])
    def test_with_rebalancing_on_a_lower_cap_or_a_higher_k_is_said_to_sell(self, link, named):
        # These settings can sell more of the held pairs: the page warns in red
        _save(self._REBALANCE_SAVED)
        response = _get(_app(), "/confirm?" + _query(**link))
        assert response.status == 200
        [warning] = re.findall(r'<p class="warn">(.*?)</p>', response.body)
        assert warning == (
            "Warning: with Rebalance yes, these settings can sell more of the held pairs: "
            f"{named}; each exact held time-series pair, or lone leg whose partner has paid "
            "out, worth more than its new Kelly size is sold down to it before anything is "
            "bought, once the wait since its last trade has passed.")

    @pytest.mark.parametrize("saved, link", [
        # Rebalancing saved off, and the link leaves it off
        (dataclasses.replace(_REBALANCE_SAVED, rebalance=False), {"size_cap": "0.1"}),
        (dataclasses.replace(_REBALANCE_SAVED, rebalance=False),
         {"k": "0.9", "size_cap": "0.2"}),
        # Rebalancing turned off by the link itself
        (_REBALANCE_SAVED, {"size_cap": "0.1", "rebalance": "off"}),
        # A cap raised, and k lowered: nothing more is sold
        (_REBALANCE_SAVED, {"size_cap": "0.3"}),
        (_REBALANCE_SAVED, {"k": "0.7", "size_cap": "0.2"}),
    ], ids=["off, cap", "off, k", "link turns it off", "cap raised", "k lowered"])
    def test_no_warning_when_no_more_is_sold(self, saved, link):
        _save(saved)
        response = _get(_app(), "/confirm?" + _query(**link))
        assert response.status == 200
        assert 'class="warn"' not in response.body

    @pytest.mark.parametrize("given, saved_on", [("on", False), ("off", True)])
    def test_a_named_rebalance_is_proposed(self, given, saved_on):
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=saved_on))
        page = _parse(_get(_app(), "/confirm?" + _query(rebalance=given)).body)
        assert page.rows["rebalance"] == "changed"
        assert page.cells["rebalance"][1:3] == ["yes" if saved_on else "no",
                                                "yes" if given == "on" else "no"]

    @pytest.mark.parametrize("saved", [
        {}, {"sell_at": 0.85}, {"sell_at": 0.85, "sell_min_days": 3}])
    def test_missing_sell_settings_keep_the_saved_values(self, saved):
        # A link that names neither (from a dashboard page without the Sell
        # view, or made by hand) leaves selling exactly as saved, shown as no change
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, **saved))
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.rows["sell at"] == page.rows["min days to maturity"] == "same"
        assert page.cells["sell at"][1] == page.cells["sell at"][2]
        assert page.cells["min days to maturity"][1] == page.cells["min days to maturity"][2]
        proposed, _ = defaults_server._proposal(defaults_server._params(_query()),
                                                config.read_saved_live_defaults())
        assert (proposed.sell_at, proposed.sell_min_days) == (
            saved.get("sell_at"), saved.get("sell_min_days"))

    def test_missing_sell_settings_with_none_saved_are_the_seed_s(self):
        settings, _ = defaults_server._proposal(defaults_server._params(_query()), None)
        assert settings.sell_at is config.LIVE_DEFAULTS_SEED.sell_at is None
        assert settings.sell_min_days is config.LIVE_DEFAULTS_SEED.sell_min_days is None

    @pytest.mark.parametrize("given, expected", [
        ({"sell_at": "0.85"}, (0.85, None)),
        ({"sell_at": "0.85", "sell_min_days": "3"}, (0.85, 3)),
        ({"sell_at": "1", "sell_min_days": "1"}, (1.0, 1)),
        ({"sell_at": "0.01", "sell_min_days": "21"}, (0.01, 21)),
        ({"sell_at": "0.29"}, (0.29, None)),
        ({"sell_at": "1e-2"}, (0.01, None)),
        ({"sell_at": "0.85", "sell_min_days": "off"}, (0.85, None)),
        ({"sell_at": "off"}, (None, None)),
        ({"sell_at": "off", "sell_min_days": "off"}, (None, None)),
        ({"sell_at": "0.85", "sell_min_days": "007"}, (0.85, 7)),
    ])
    def test_named_sell_settings_are_proposed(self, given, expected):
        # With nothing saved, so a field a link leaves out is the seed's (off)
        settings, _ = defaults_server._proposal(defaults_server._params(_query(**given)), None)
        assert (settings.sell_at, settings.sell_min_days) == expected
        page = _parse(_get(_app(), "/confirm?" + _query(**given)).body)
        level = "off" if expected[0] is None else f"{round(expected[0] * 100)}% of potential profit"
        days = "any" if expected[1] is None else str(expected[1])
        assert page.cells["sell at"][1:3] == ["—", level]
        assert page.cells["min days to maturity"][1:3] == ["—", days]

    def test_a_named_sell_level_over_a_saved_one_is_a_change(self):
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, sell_at=0.85, sell_min_days=3))
        page = _parse(_get(_app(), "/confirm?" + _query(sell_at="0.9")).body)
        assert page.rows["sell at"] == "changed"
        assert page.cells["sell at"][1:3] == ["85% of potential profit", "90% of potential profit"]
        # The minimum was left out, so it stays as saved
        assert page.rows["min days to maturity"] == "same"
        page = _parse(_get(_app(), "/confirm?" + _query(
            sell_at="off", sell_min_days="off")).body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "sell at", "min days to maturity"]
        assert page.cells["sell at"][1:3] == ["85% of potential profit", "off"]
        assert page.cells["min days to maturity"][1:3] == ["3", "any"]

    def test_turning_the_level_off_but_keeping_a_saved_minimum_is_refused_with_the_reason(self):
        # The minimum was left out, so it stays at the saved 3 with no level
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, sell_at=0.85, sell_min_days=3))
        response = _get(_app(), "/confirm?" + _query(sell_at="off"))
        assert response.status == 400
        assert "sell_min_days needs a sell level" in html.unescape(response.body)
        assert "<form" not in response.body

    @pytest.mark.parametrize("saved", [
        {}, {"rebalance": True}, {"rebalance": True, "min_price_move": 0.25}])
    def test_missing_rebalance_settings_keep_the_saved_values(self, saved):
        # A link that names neither field (the dashboard's save button on a
        # page without the Rebalance view, a hand-made link) keeps both as
        # saved, either way, shown as no change
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, **saved))
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.rows["rebalance"] == page.rows["min price movement"] == "same"
        assert page.cells["rebalance"][1] == page.cells["rebalance"][2] == (
            "yes" if saved.get("rebalance") else "no")
        assert page.cells["min price movement"][1] == page.cells["min price movement"][2] == (
            "25c" if "min_price_move" in saved else "any")
        proposed, _ = defaults_server._proposal(defaults_server._params(_query()),
                                                config.read_saved_live_defaults())
        assert (proposed.rebalance, proposed.min_price_move) == (
            saved.get("rebalance", False), saved.get("min_price_move"))

    def test_missing_rebalance_settings_with_none_saved_are_the_seed_s(self):
        settings, _ = defaults_server._proposal(defaults_server._params(_query()), None)
        assert settings.rebalance is config.LIVE_DEFAULTS_SEED.rebalance is True
        assert settings.min_price_move is config.LIVE_DEFAULTS_SEED.min_price_move is None

    @pytest.mark.parametrize("given, expected", [
        ({"rebalance": "on"}, (True, None)),
        ({"rebalance": "on", "min_price_move": "0.1"}, (True, 0.10)),
        ({"rebalance": "on", "min_price_move": "0.10"}, (True, 0.10)),
        ({"rebalance": "on", "min_price_move": "0.05"}, (True, 0.05)),
        ({"rebalance": "on", "min_price_move": "0.15"}, (True, 0.15)),
        ({"rebalance": "on", "min_price_move": "1"}, (True, 1.0)),
        ({"rebalance": "on", "min_price_move": "1e-1"}, (True, 0.10)),
        ({"rebalance": "on", "min_price_move": "off"}, (True, None)),
        ({"rebalance": "off"}, (False, None)),
        ({"rebalance": "off", "min_price_move": "off"}, (False, None)),
        # The switch left out takes the seed's (on)
        ({"min_price_move": "off"}, (True, None)),
        ({"min_price_move": "0.1"}, (True, 0.10)),
    ])
    def test_named_rebalance_settings_are_proposed(self, given, expected):
        # With nothing saved, so a field a link leaves out is the seed's
        settings, _ = defaults_server._proposal(defaults_server._params(_query(**given)), None)
        assert (settings.rebalance, settings.min_price_move) == expected
        page = _parse(_get(_app(), "/confirm?" + _query(**given)).body)
        move = "any" if expected[1] is None else f"{round(expected[1] * 100)}c"
        assert page.cells["rebalance"][1:3] == ["—", "yes" if expected[0] else "no"]
        assert page.cells["min price movement"][1:3] == ["—", move]

    def test_a_link_s_dollars_and_a_flag_s_cents_are_one_minimum(self):
        # A link's 0.1 (dollars) is the flag's 10 (cents)
        settings, _ = defaults_server._proposal(
            defaults_server._params(_query(rebalance="on", min_price_move="0.1")), None)
        assert settings.min_price_move == 0.10 == 10 / 100
        assert "--min-price-move=10" in config.live_settings_argv(settings)

    def test_a_named_minimum_move_over_a_saved_one_is_a_change(self):
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=True,
                           min_price_move=0.25))
        page = _parse(_get(_app(), "/confirm?" + _query(min_price_move="0.1")).body)
        assert page.rows["min price movement"] == "changed"
        assert page.cells["min price movement"][1:3] == ["25c", "10c"]
        # The switch was left out, so it stays as saved
        assert page.rows["rebalance"] == "same"
        page = _parse(_get(_app(), "/confirm?" + _query(
            rebalance="off", min_price_move="off")).body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "rebalance", "min price movement"]
        assert page.cells["rebalance"][1:3] == ["yes", "no"]
        assert page.cells["min price movement"][1:3] == ["25c", "any"]

    def test_turning_rebalancing_off_but_keeping_a_saved_minimum_is_refused_with_the_reason(
            self):
        # The minimum was left out, so it stays at the saved 25c with rebalancing off
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=True,
                           min_price_move=0.25))
        response = _get(_app(), "/confirm?" + _query(rebalance="off"))
        assert response.status == 400
        assert "min_price_move needs rebalance on" in html.unescape(response.body)
        assert "<form" not in response.body

    @pytest.mark.parametrize("saved_off", [False, True])
    def test_a_minimum_move_with_rebalancing_off_is_refused_with_the_reason(self, saved_off):
        # Named off by the link, or saved off and left out by it
        if saved_off:
            _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                               size_cap=0.1))
            given = {"min_price_move": "0.1"}
        else:
            given = {"rebalance": "off", "min_price_move": "0.1"}
        before = config.LIVE_DEFAULTS_FILE.read_bytes() if saved_off else None
        response = _get(_app(), "/confirm?" + _query(**given))
        assert response.status == 400
        assert "min_price_move needs rebalance on" in html.unescape(response.body)
        assert "<form" not in response.body and "<button" not in response.body
        if saved_off:
            assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        else:
            assert config.read_saved_live_defaults() is None

    def test_a_minimum_move_off_the_grid_is_refused_with_the_reason(self):
        response = _get(_app(), "/confirm?" + _query(rebalance="on", min_price_move="0.07"))
        assert response.status == 400
        assert "min_price_move must be a multiple of 5c from 5c to $1" in html.unescape(
            response.body)

    @pytest.mark.parametrize("saved", [
        {}, {"rebalance": True}, {"rebalance": True, "rebalance_min_days_to_maturity": 1},
        {"rebalance": True, "rebalance_min_days_to_maturity": 7,
         "min_price_move": 0.25}])
    def test_a_missing_rebalance_until_keeps_the_saved_value(self, saved):
        # A link that leaves it out (from a dashboard page without the
        # "Rebalance until" choices, or made by hand) keeps it as saved, shown as
        # no change, exactly as the minimum price movement is kept
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, **saved))
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        words = {None: "maturity", 1: "1 day before maturity", 7: "7 days before maturity"}
        expected = words[saved.get("rebalance_min_days_to_maturity")]
        assert page.rows["rebalance until"] == "same"
        assert page.cells["rebalance until"][1:3] == [expected, expected]
        proposed, _ = defaults_server._proposal(defaults_server._params(_query()),
                                                config.read_saved_live_defaults())
        assert proposed.rebalance_min_days_to_maturity == saved.get(
            "rebalance_min_days_to_maturity")

    def test_a_missing_rebalance_until_with_none_saved_is_the_seed_s(self):
        settings, _ = defaults_server._proposal(defaults_server._params(_query()), None)
        assert settings.rebalance_min_days_to_maturity is (
            config.LIVE_DEFAULTS_SEED.rebalance_min_days_to_maturity) is None

    @pytest.mark.parametrize("given, expected, words", [
        ("0", None, "maturity"),
        ("1", 1, "1 day before maturity"),
        ("7", 7, "7 days before maturity"),
        # Plain ASCII digits as for the minimum of days: leading zeros read as the number
        ("007", 7, "7 days before maturity"),
        ("00", None, "maturity"),
    ])
    def test_a_named_rebalance_until_is_proposed(self, given, expected, words):
        # With nothing saved, so the switch a link leaves out is the seed's (on)
        query = _query(rebalance_min_days_to_maturity=given)
        settings, _ = defaults_server._proposal(defaults_server._params(query), None)
        assert settings.rebalance and settings.rebalance_min_days_to_maturity == expected
        response = _get(_app(), "/confirm?" + query)
        assert response.status == 200
        assert _parse(response.body).cells["rebalance until"][1:3] == ["—", words]

    def test_a_named_rebalance_until_over_a_saved_one_is_the_only_change(self):
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=True,
                           rebalance_min_days_to_maturity=7))
        response = _get(_app(), "/confirm?" + _query(rebalance_min_days_to_maturity="1"))
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "rebalance until"]
        assert page.cells["rebalance until"][1:3] == ["7 days before maturity",
                                                      "1 day before maturity"]
        assert "1 of 11 settings change." in response.body
        # Naming 0 puts it back to "maturity"
        page = _parse(_get(_app(), "/confirm?" + _query(
            rebalance_min_days_to_maturity="0")).body)
        assert page.cells["rebalance until"][1:3] == ["7 days before maturity", "maturity"]

    def test_rebalance_off_without_the_field_clears_a_saved_limit(self):
        # A link that says rebalance=off and min_price_move=off and does not
        # name the limit (a dashboard page without the "Rebalance until"
        # choices sends one, and so may a link made by hand): the limit goes
        # with the switch, so the link is not refused for "needs rebalance on"
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=True,
                           min_price_move=0.25, rebalance_min_days_to_maturity=7))
        query = _query(rebalance="off", min_price_move="off")
        proposed, _ = defaults_server._proposal(defaults_server._params(query),
                                                config.read_saved_live_defaults())
        assert proposed.rebalance is False and proposed.rebalance_min_days_to_maturity is None
        response = _get(_app(), "/confirm?" + query)
        assert response.status == 200
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "rebalance", "min price movement", "rebalance until"]
        assert page.cells["rebalance until"][1:3] == ["7 days before maturity", "maturity"]
        assert "3 of 11 settings change." in response.body

    def test_rebalance_off_without_the_field_clears_it_with_nothing_saved(self):
        settings, _ = defaults_server._proposal(
            defaults_server._params(_query(rebalance="off")), None)
        assert settings.rebalance_min_days_to_maturity is None

    def test_a_saved_limit_is_kept_when_the_link_leaves_out_both(self):
        # The switch is left out as well, so nothing says it goes off
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, rebalance=True,
                           rebalance_min_days_to_maturity=7))
        proposed, _ = defaults_server._proposal(defaults_server._params(_query()),
                                                config.read_saved_live_defaults())
        assert proposed.rebalance and proposed.rebalance_min_days_to_maturity == 7

    def test_rebalance_off_naming_a_limit_is_refused_with_the_reason(self):
        response = _get(_app(), "/confirm?" + _query(
            rebalance="off", min_price_move="off", rebalance_min_days_to_maturity="7"))
        assert response.status == 400
        assert "rebalance_min_days_to_maturity needs rebalance on" in html.unescape(
            response.body)
        assert "<form" not in response.body and "<button" not in response.body
        assert config.read_saved_live_defaults() is None
        # Naming "maturity" (0) with rebalancing off is nothing to refuse
        assert _get(_app(), "/confirm?" + _query(
            rebalance="off", min_price_move="off",
            rebalance_min_days_to_maturity="0")).status == 200

    def test_a_limit_with_rebalancing_saved_off_is_refused_with_the_reason(self):
        # The switch is left out, so it stays at the saved off
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        response = _get(_app(), "/confirm?" + _query(rebalance_min_days_to_maturity="1"))
        assert response.status == 400
        assert "rebalance_min_days_to_maturity needs rebalance on" in html.unescape(
            response.body)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    @pytest.mark.parametrize("given, reason", [
        ("2", "rebalance_min_days_to_maturity must be 0, 1 or 7, got 2"),
        ("14", "rebalance_min_days_to_maturity must be 0, 1 or 7, got 14"),
        ("seven", "rebalance_min_days_to_maturity must be a whole number of at most six "
                  "digits, got 'seven'"),
        ("7.0", "rebalance_min_days_to_maturity must be a whole number of at most six "
                "digits, got '7.0'"),
        ("off", "rebalance_min_days_to_maturity must be a whole number of at most six "
                "digits, got 'off'"),
        ("", "rebalance_min_days_to_maturity is blank"),
    ])
    def test_a_bad_rebalance_until_is_refused_with_the_reason(self, given, reason):
        response = _get(_app(), "/confirm?" + _query(rebalance_min_days_to_maturity=given))
        assert response.status == 400
        assert reason in html.unescape(response.body)
        assert "<form" not in response.body

    def test_rebalance_until_given_twice_is_refused_with_the_reason(self):
        response = _get(_app(), "/confirm?" + _query(rebalance_min_days_to_maturity="7")
                        + "&rebalance_min_days_to_maturity=7")
        assert response.status == 400
        assert "rebalance_min_days_to_maturity is given 2 times" in response.body

    def test_a_missing_exclusion_proposes_none(self):
        _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1, exclude_categories=("Sports",),
                           exclude_tags=("Basketball",)))
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.rows["exclude categories"] == page.rows["exclude tags"] == "changed"
        assert page.cells["exclude categories"][1:3] == ["Sports", "none"]
        assert page.cells["exclude tags"][1:3] == ["Basketball", "none"]

    def test_confirm_is_there_and_disabled_when_something_changes(self):
        # A server built without a process starter: only the save applies.
        # The dropdown bar is a form of its own, which only asks for the page
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.forms == [("get", "/confirm"), ("post", "/confirm")]
        assert page.labels == _CONFIRM_LABELS
        dry, save, trade = page.buttons
        assert save == {"type": "submit", "name": "action", "value": "save", "id": "confirm",
                        "disabled": None}
        for placeholder in (dry, trade):
            assert placeholder == {"type": "button", "disabled": None}
        names = [name for name, _ in page.hidden]
        assert names == ["tier_floors", "spread_min", "spread_max", "k", "size_cap",
                         "fingerprint", "nonce", "token"]

    def test_the_run_buttons_say_why_they_do_not_apply(self):
        body = _get(_app(), "/confirm?" + _query()).body
        assert body.count(html.escape(defaults_server._NOT_STARTED_TO_TRADE)) == 2
        # With a process starter, and nothing saved, a dry run asks for a save first
        body = _get(_app(start_process=_Starter()), "/confirm?" + _query()).body
        assert html.escape(defaults_server._SAVE_FIRST) in body
        page = _parse(body)
        assert [b.get("id") for b in page.buttons] == [None, "confirm", "confirm-trade"]

    def test_nothing_to_change_keeps_the_save_a_real_button(self):
        # Confirm and save always works: with nothing changed it writes nothing
        # and says so (TestSaveUnchanged)
        _save(_BASE_SETTINGS)
        response = _get(_app(start_process=_Starter()), "/confirm?" + _query())
        page = _parse(response.body)
        assert page.labels == _CONFIRM_LABELS
        assert [b.get("id") for b in page.buttons] == ["confirm-dry-run", "confirm",
                                                       "confirm-trade"]
        assert page.buttons[1] == {"type": "submit", "name": "action", "value": "save",
                                   "id": "confirm", "disabled": None}
        assert ("These are already the live defaults: Confirm and save writes nothing, and "
                "Dry run and Confirm and trade run them as they are.") in response.body
        assert "nothing to save" not in response.body.lower()
        assert "0 of 11 settings change." in response.body
        # With nothing to save, the page does not say it writes, and the red
        # line says Confirm and trade runs the saved defaults as they are
        assert "This writes" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body
        assert ('<p class="money">Confirm and trade runs the live bot with these saved '
                "defaults on the production account: it places real orders.</p>") \
            in response.body

    def test_the_red_line_and_the_notes(self):
        body = _get(_app(start_process=_Starter()), "/confirm?" + _query()).body
        assert ('<p class="money">Confirm and trade saves these settings as the live defaults, '
                "then runs the live bot with them on the production account: it places real "
                "orders.</p>") in body
        assert html.escape(defaults_server._DRY_RUN_NOTE) in body
        assert html.escape(defaults_server._ARMING_NOTE) in body
        # The checkout a run starts from, and the machine's lock
        assert str(config.LIVE_RUN_LOCK_FILE.absolute()) in body
        assert f"from <code>{config.PROJECT_ROOT.absolute()}</code>" in body

    def test_each_render_has_a_fresh_nonce(self):
        app = _app()
        first = dict(_parse(_get(app, "/confirm?" + _query()).body).hidden)
        second = dict(_parse(_get(app, "/confirm?" + _query()).body).hidden)
        assert first["nonce"] != second["nonce"]
        assert first["token"] != second["token"]
        assert first["fingerprint"] == second["fingerprint"]

    def test_the_csp_hashes_are_the_three_scripts_on_the_page(self):
        response = _get(_app(), "/confirm?" + _query())
        scripts = _parse(response.body).scripts
        assert scripts == _TRADING_SCRIPTS
        digests = [b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
                   for script in scripts]
        csp = dict(defaults_server._response_headers(response))["Content-Security-Policy"]
        # Exactly these three, in the scripts' order, and nothing else may run
        [script_src] = [part for part in csp.split("; ") if part.startswith("script-src ")]
        assert script_src == "script-src " + " ".join(f"'sha256-{digest}'" for digest in digests)

    def test_the_script_arms_after_the_configured_delay(self):
        js = defaults_server._CONFIRM_JS
        assert f"}}, {config.DEFAULTS_SERVER_CONFIRM_ARM_MS});" in js
        assert "ARM_MS" not in js and "__IDS__" not in js
        assert f"var IDS = {json.dumps(list(defaults_server._ARMED_BUTTONS))};" in js
        for event in ("visibilitychange", "mousemove", "keydown", "submit"):
            assert f"addEventListener('{event}'" in js

    def test_a_hostile_category_is_escaped(self):
        hostile = "<script>alert(1)</script>"
        response = _get(_app(), "/confirm?" + _query(exclude_category=hostile, exclude_tag='"><b>x'))
        assert response.status == 200
        assert "<script>alert" not in response.body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.body
        page = _parse(response.body)
        assert page.scripts == _TRADING_SCRIPTS
        assert ("exclude_category", hostile) in page.hidden
        assert ("exclude_tag", '"><b>x') in page.hidden
        assert page.cells["exclude categories"][2] == hostile
        # The bar's menus list both names, escaped, ticked
        assert [(box["value"], "checked" in box) for box in page.bar_boxes(
            "exclude_category")] == [(hostile, True)]
        assert [box["value"] for box in page.bar_boxes("exclude_tag")] == ['"><b>x']
        # ... and the notifier's message carries them as data, escaped
        message = json.loads(page.body["data-dashboard-message"])
        assert "exclude_category=%3Cscript%3Ealert%281%29%3C%2Fscript%3E" in message["query"]


class TestTradeAddress:
    """/trade, an older version's address for trading the saved defaults: GET is sent to the
    trading page (which opens on them), and POST is no route at all."""

    def test_get_is_sent_to_the_trading_page(self):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        response = _get(app, "/trade")
        assert response.status == 303 and response.location == "/confirm"
        assert "<form" not in response.body
        # Where it is sent, the saved defaults are proposed as they are
        page = _get(app, response.location)
        assert page.status == 200 and "0 of 11 settings change." in page.body
        assert starter.calls == []

    def test_get_is_sent_there_with_nothing_saved_too(self):
        response = _get(_app(), "/trade")
        assert response.status == 303 and response.location == "/confirm"

    def test_a_post_is_not_found_and_starts_nothing(self):
        # An older version's trade page, still open, posts here: no route,
        # whatever it carries, and nothing is saved or started
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        saved = config.read_saved_live_defaults()
        fingerprint, nonce = defaults_server._fingerprint(saved), "5" * 32
        for action in ("trade", "dry_run"):
            form = [("fingerprint", fingerprint), ("nonce", nonce),
                    ("token", app._token("trade", fingerprint, nonce, {})), ("action", action)]
            response = _post(app, form, target="/trade")
            assert response.status == 404
            assert "<form" not in response.body
        assert starter.calls == [] and not config.LIVE_RUNS_DIR.exists()
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        # ... and its nonce stays unused here: a page of this server still works
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))
        assert len(starter.calls) == 1


# macOS's JavaScriptCore shell, the runtime used when node is not installed
_JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")

# A stand-in for the few browser pieces the page script touches: its buttons
# (those whose ids __PRESENT__ lists, each in the one form), a placeholder with
# no id, the page's visibility, event listeners a step fires, and timers that
# run only when a step says so. The script runs inside a function whose
# parameters are these stand-ins, so neither runtime's own globals are
# replaced. The steps follow: ["fire", event], ["show", "visible" | "hidden"]
# (sets the visibility and fires visibilitychange), ["submit"] (fires the
# form's submit listeners, as a click on a button does), ["elapse"] (runs
# every pending timer) and ["snap", label] (records each button's disabled
# flag, the placeholder's, the pending timers' delays and the events the page
# and the form listen for); the records are printed as one JSON line.
_CONFIRM_HARNESS = """
var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __listeners = {};
var __formListeners = {};
var __timers = {};
var __next = 1;
var __form = {
  addEventListener: function(type, fn) {
    (__formListeners[type] = __formListeners[type] || []).push(fn);
  }
};
var __buttons = {};
__PRESENT__.forEach(function(id) { __buttons[id] = {disabled: true, form: __form}; });
var __placeholder = {disabled: true, form: __form};
var __document = {
  visibilityState: __VISIBILITY__,
  getElementById: function(id) {
    return Object.prototype.hasOwnProperty.call(__buttons, id) ? __buttons[id] : null;
  },
  addEventListener: function(type, fn) {
    (__listeners[type] = __listeners[type] || []).push(fn);
  }
};
function __setTimeout(fn, ms) { var id = __next++; __timers[id] = {fn: fn, ms: ms}; return id; }
function __clearTimeout(id) { delete __timers[id]; }
function __fire(table, type) {
  (table[type] || []).forEach(function(fn) { fn({type: type}); });
}
(function(document, setTimeout, clearTimeout) {
__SCRIPT__
})(__document, __setTimeout, __clearTimeout);
var __out = {};
__STEPS__.forEach(function(step) {
  if (step[0] === 'fire') {
    __fire(__listeners, step[1]);
  } else if (step[0] === 'show') {
    __document.visibilityState = step[1];
    __fire(__listeners, 'visibilitychange');
  } else if (step[0] === 'submit') {
    __fire(__formListeners, 'submit');
  } else if (step[0] === 'elapse') {
    Object.keys(__timers).forEach(function(id) {
      var timer = __timers[id];
      delete __timers[id];
      timer.fn();
    });
  } else if (step[0] === 'snap') {
    var disabled = {};
    Object.keys(__buttons).forEach(function(id) { disabled[id] = __buttons[id].disabled; });
    __out[step[1]] = {
      disabled: disabled,
      placeholder: __placeholder.disabled,
      timers: Object.keys(__timers).map(function(id) { return __timers[id].ms; }).sort(),
      listeners: Object.keys(__listeners).sort(),
      form: Object.keys(__formListeners).sort()
    };
  }
});
__emit(JSON.stringify(__out));
"""


def _js_runtime() -> str | None:
    """
    Find a JavaScript runtime for the page script.

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
                        present=defaults_server._ARMED_BUTTONS) -> dict:
    """
    Run the pages' script, byte for byte, through a list of steps.

    Skips the test when no JavaScript runtime is present.

    Args:
        tmp_path (Path): Where to write the program.
        steps (list[list[str]]): The steps (see _CONFIRM_HARNESS).
        visible (bool): Whether the page is visible when the script runs.
        present (tuple[str, ...]): The ids of the page's real buttons.

    Returns:
        dict: Each "snap" step's record, by its label.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("neither node nor macOS's jsc is here to run the page script")
    program = (_CONFIRM_HARNESS
               .replace("__PRESENT__", json.dumps(list(present)))
               .replace("__VISIBILITY__", json.dumps("visible" if visible else "hidden"))
               .replace("__STEPS__", json.dumps(steps))
               .replace("__SCRIPT__", defaults_server._CONFIRM_JS))
    path = tmp_path / "confirm_script.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestConfirmScript:
    """The page script itself, run under node or jsc with a stand-in page."""

    ARM = config.DEFAULTS_SERVER_CONFIRM_ARM_MS
    EVENTS = ["keydown", "mousemove", "visibilitychange"]

    @staticmethod
    def _all(snap, value):
        """
        Whether every real button in a snapshot has one disabled flag.

        Args:
            snap (dict): A snapshot.
            value (bool): The flag.

        Returns:
            bool: True when every button's flag is value (and there is one).
        """
        return bool(snap["disabled"]) and set(snap["disabled"].values()) == {value}

    def test_nothing_arms_it_before_the_delay(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["snap", "load"], ["fire", "mousemove"], ["fire", "keydown"], ["snap", "early"]])
        assert out["load"]["timers"] == [self.ARM]
        assert out["load"]["listeners"] == self.EVENTS and out["load"]["form"] == ["submit"]
        assert self._all(out["load"], True) and self._all(out["early"], True)

    @pytest.mark.parametrize("present", [
        ("confirm-dry-run", "confirm", "confirm-trade"),
        ("confirm-dry-run", "confirm-trade"),
        ("confirm", "confirm-trade"),
    ])
    @pytest.mark.parametrize("event", ["mousemove", "keydown"])
    def test_after_the_delay_it_arms_every_present_button_together(self, tmp_path, present,
                                                                    event):
        out = _run_confirm_script(tmp_path, [
            ["elapse"], ["snap", "ready"], ["fire", event], ["snap", "armed"],
            ["show", "hidden"], ["snap", "hidden"]], present=present)
        # The delay alone arms nothing: it takes the reader's own move
        assert set(out["ready"]["disabled"]) == set(present)
        assert self._all(out["ready"], True)
        assert self._all(out["armed"], False)
        assert self._all(out["hidden"], True)
        # A placeholder has no id, so nothing ever arms it
        assert out["armed"]["placeholder"] is True

    def test_hiding_the_page_disarms_it_and_a_return_needs_the_full_delay_again(
            self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["elapse"], ["fire", "mousemove"], ["snap", "armed"],
            ["show", "hidden"], ["snap", "hidden"], ["fire", "keydown"], ["snap", "moved"],
            ["show", "visible"], ["snap", "back"], ["fire", "mousemove"], ["snap", "early"],
            ["elapse"], ["fire", "keydown"], ["snap", "again"]])
        assert self._all(out["armed"], False)
        assert self._all(out["hidden"], True) and out["hidden"]["timers"] == []
        assert self._all(out["moved"], True)
        assert out["back"]["timers"] == [self.ARM]
        assert self._all(out["early"], True)
        assert self._all(out["again"], False)

    def test_hiding_the_page_before_the_delay_ends_restarts_it(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["show", "hidden"], ["snap", "hidden"], ["show", "visible"], ["snap", "back"],
            ["fire", "mousemove"], ["snap", "early"]])
        assert out["hidden"]["timers"] == [] and out["back"]["timers"] == [self.ARM]
        assert self._all(out["early"], True)

    def test_a_page_opened_hidden_starts_no_clock_until_it_is_shown(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["snap", "load"], ["fire", "mousemove"], ["snap", "moved"],
            ["show", "visible"], ["snap", "shown"]], visible=False)
        assert out["load"]["timers"] == [] and self._all(out["moved"], True)
        assert out["shown"]["timers"] == [self.ARM]

    def test_a_page_without_a_real_button_listens_for_nothing(self, tmp_path):
        out = _run_confirm_script(tmp_path, [["elapse"], ["fire", "mousemove"],
                                             ["snap", "load"]], present=())
        assert out["load"] == {"disabled": {}, "placeholder": True, "timers": [],
                               "listeners": [], "form": []}

    def test_after_a_submit_the_buttons_stay_disabled(self, tmp_path):
        out = _run_confirm_script(tmp_path, [
            ["elapse"], ["fire", "mousemove"], ["snap", "armed"], ["submit"],
            ["snap", "sent"], ["elapse"], ["snap", "after"], ["fire", "mousemove"],
            ["fire", "keydown"], ["snap", "moved"], ["show", "hidden"], ["show", "visible"],
            ["elapse"], ["fire", "mousemove"], ["snap", "again"]])
        assert self._all(out["armed"], False)
        # Still enabled for the submission to read which button was pressed ...
        assert self._all(out["sent"], False) and out["sent"]["timers"] == [0]
        # ... then disabled, and nothing arms them again
        for label in ("after", "moved", "again"):
            assert self._all(out[label], True), label
            assert out[label]["placeholder"] is True


class TestHome:
    """GET /confirm with no address fields: the defaults in force proposed as they are, every
    field named, so each button acts on them; an older version's file's values in their new
    form; the seed values with none saved."""

    @staticmethod
    def _proposed(response) -> tuple[LiveSettings, str]:
        """
        Read what a trading page proposes, from the fields its form posts back.

        Args:
            response (defaults_server._Response): The page.

        Returns:
            tuple[LiveSettings, str]: The proposal and its source note, as the
                server reads the form (with nothing saved to keep a field from).
        """
        fields = {}
        for name, value in _parse(response.body).hidden:
            if name in defaults_server._FIELDS:
                fields.setdefault(name, []).append(value)
        return defaults_server._proposal(fields, None)

    def test_saved_defaults_are_proposed_as_they_are(self):
        saved = _save(LiveSettings(tier_floors=True, spread_band=(0.2, 0.8),
                                   interval_discount=0.85, size_cap=0.15,
                                   exclude_categories=("Sports", "Economics"),
                                   exclude_tags=("Health · Covid",), rebalance=True,
                                   min_price_move=0.1, rebalance_min_days_to_maturity=7,
                                   sell_at=0.85, sell_min_days=3), source=DASHBOARD_NOTE)
        response = _get(_app(start_process=_Starter()), "/confirm")
        assert response.status == 200
        assert "Overwrite the live trading defaults?" in response.body
        assert "0 of 11 settings change." in response.body
        page = _parse(response.body)
        assert set(page.rows.values()) == {"same"}
        # Three real buttons, in their order
        assert [b.get("id") for b in page.buttons] == ["confirm-dry-run", "confirm",
                                                       "confirm-trade"]
        # The form carries every field, exactly as the full link names them,
        # and no source note (the saved note lives only in the origin)
        query = defaults_server._settings_query(saved)
        assert [(n, v) for n, v in page.hidden if n in defaults_server._FIELDS] == [
            (name, value) for name, values in defaults_server._params(query).items()
            for value in values]
        assert self._proposed(response) == (saved, "")
        assert "New defaults from: (no source note)" in response.body

    def test_an_empty_query_is_the_same_page(self):
        _save(_BASE_SETTINGS)
        for target in ("/confirm", "/confirm?"):
            response = _get(_app(), target)
            assert response.status == 200, target
            assert self._proposed(response) == (_BASE_SETTINGS, "")

    def test_a_file_an_older_version_saved_is_proposed_in_its_new_form(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        response = _get(_app(start_process=_Starter()), "/confirm")
        assert response.status == 200
        assert "Save the live trading defaults again?" in response.body
        assert f'<p class="banner">{_OUTDATED_CONFIRM_NOTICE}</p>' in response.body
        assert "11 of 11 settings change." in response.body
        assert self._proposed(response) == (_OLD_VERSION_READS_AS, "")
        # Nothing is in force, so a dry run asks for the save first
        assert [b.get("id") for b in _parse(response.body).buttons] == [
            None, "confirm", "confirm-trade"]
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES

    def test_with_none_saved_the_seed_is_proposed(self):
        response = _get(_app(start_process=_Starter()), "/confirm")
        assert response.status == 200
        assert "Save the first live trading defaults?" in response.body
        assert self._proposed(response) == (config.LIVE_DEFAULTS_SEED,
                                            config.LIVE_DEFAULTS_SEED_SOURCE)
        assert f"New defaults from: {config.LIVE_DEFAULTS_SEED_SOURCE}" in response.body
        assert html.escape(defaults_server._SAVE_FIRST) in response.body
        assert config.read_saved_live_defaults() is None

    def test_a_refused_file_is_409(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        response = _get(_app(), "/confirm")
        assert response.status == 409
        assert "<form" not in response.body

    @staticmethod
    def _uncarried(response) -> _PageParser:
        """
        Check a trading page with no address fields was refused because the saved values cannot be carried.

        Args:
            response (defaults_server._Response): The answer to GET /confirm.

        Returns:
            _PageParser: The refusal page, parsed.
        """
        assert response.status == 409, response.body
        page = _parse(response.body)
        # No form, no button, no script: nothing on it can save or start a run
        assert page.forms == [] and page.buttons == [] and page.scripts == []
        assert "data-dashboard-message" not in page.body
        assert "<title>The saved live defaults cannot be shown here</title>" in response.body
        # Its one way on is the seed values, where a save replaces the file
        assert page.links == ["/confirm?" + defaults_server._seed_query()]
        assert f"Fix or delete <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" in (
            response.body)
        return page

    @pytest.mark.parametrize("names, why", [
        ({"exclude_categories": ("Climate  and Weather",)},
         "exclude_category must not hold two spaces in a row"),
        ({"exclude_categories": ("Sports", "Sports")}, "exclude_category 'Sports' is given twice"),
        ({"exclude_tags": tuple(f"Tag {i:03d}"
                                for i in range(config.DEFAULTS_SERVER_MAX_FILTER_NAMES + 5))},
         f"exclude_tag is given {config.DEFAULTS_SERVER_MAX_FILTER_NAMES + 5} times, more than "
         f"{config.DEFAULTS_SERVER_MAX_FILTER_NAMES}"),
        ({"exclude_categories": ("Sports", "sports")},
         "exclude categories would read Sports here, not Sports, sports"),
    ], ids=["two-spaces", "same-name-twice", "too-many", "two-letter-cases"])
    def test_saved_values_a_link_cannot_carry_get_no_buttons(self, names, why):
        # LiveSettings, and so every live run, takes these names, but the page
        # proposes the saved values as a link: one a link cannot carry would
        # make a button act on other settings (or on nothing at all)
        saved = _save(replace(_BASE_SETTINGS, **names))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        response = _get(app, "/confirm")
        self._uncarried(response)
        assert why in html.unescape(response.body)
        assert ("Every live run still uses them as saved, but this page cannot dry-run, save "
                "or trade them unchanged.") in response.body
        # /trade leads to the same page
        assert _get(app, "/trade").location == "/confirm"
        # Nothing was written or started, and a live run still reads them
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert config.read_saved_live_defaults() == saved
        assert starter.calls == []
        # The seed values it links to open a page whose Confirm and save replaces the file
        seed = _get(app, "/confirm?" + defaults_server._seed_query())
        assert seed.status == 200
        assert [b.get("id") for b in _parse(seed.body).buttons] == [
            "confirm-dry-run", "confirm", "confirm-trade"]

    def test_an_older_version_s_file_a_link_cannot_carry_gets_no_buttons(self):
        from .test_config import _SAME_TITLE_CAP_ERA_BYTES
        config.LIVE_DEFAULTS_FILE.write_bytes(_SAME_TITLE_CAP_ERA_BYTES.replace(
            b'"exclude_categories": null', b'"exclude_categories": ["Climate  and Weather"]'))
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.proposal.exclude_categories == ("Climate  and Weather",)
        response = _get(_app(start_process=_Starter()), "/confirm")
        self._uncarried(response)
        text = html.unescape(response.body)
        assert ("The saved live defaults were saved before the same-title cap was removed, so "
                "every live run refuses them, and their values in their new form hold "
                "something this trading page cannot carry as it is: exclude_category must not "
                "hold two spaces in a row") in text
        assert "Every live run still uses them" not in text

    def test_the_home_query_is_the_full_query_of_what_is_in_force(self):
        saved = _save(replace(_BASE_SETTINGS, exclude_tags=("Sports · Soccer",)))
        assert defaults_server._home_query(saved, None) == defaults_server._settings_query(saved)
        outdated = config.LiveDefaultsOutdated("m", _OLD_VERSION_READS_AS, ("trim_to_kelly",))
        assert defaults_server._home_query(None, outdated) == defaults_server._settings_query(
            _OLD_VERSION_READS_AS)
        unreadable = config.LiveDefaultsOutdated("m", None, ("trim_to_kelly",))
        assert defaults_server._home_query(None, unreadable) == defaults_server._seed_query()
        assert defaults_server._home_query(None, None) == defaults_server._seed_query()


class TestSaveUnchanged:
    """Confirm and save with nothing changed writes nothing and sends the browser to the saved
    page, which says so; with a change it writes once, as always."""

    @staticmethod
    def _counting(monkeypatch) -> list:
        """
        Replace the server's save with one that counts each call, then saves.

        Args:
            monkeypatch (pytest.MonkeyPatch): Puts the real save back afterwards.

        Returns:
            list: The settings of each save, in order.
        """
        saves = []
        real_save = defaults_server.save_live_defaults

        def counting_save(settings, *, source):
            """
            Count one save, then make it.

            Args:
                settings (LiveSettings): The defaults saved.
                source (str): Their source note.

            Returns:
                LiveSettings: What the real save returns.
            """
            saves.append(settings)
            return real_save(settings, source=source)

        monkeypatch.setattr(defaults_server, "save_live_defaults", counting_save)
        return saves

    @pytest.mark.parametrize("opened", ["home", "link"])
    def test_it_writes_nothing_and_says_so(self, monkeypatch, opened):
        _save(_BASE_SETTINGS, source=DASHBOARD_NOTE)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        saves = self._counting(monkeypatch)
        app = _app(start_process=_Starter())
        form = (_home_form(app, "confirm") if opened == "home"
                else _page_form(app, _query(source=DASHBOARD_NOTE)))
        response = _post(app, form)
        assert response.status == 303 and response.location == "/saved?unchanged=1"
        assert saves == []
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        # A resubmit is sent to the same place, still writing nothing
        again = _post(app, form)
        assert again.status == 303 and again.location == "/saved?unchanged=1"
        assert saves == [] and config.LIVE_DEFAULTS_FILE.read_bytes() == before
        # The saved page says nothing changed, and leads back to the trading page
        page = _get(app, response.location)
        assert page.status == 200
        assert ("Nothing changed: these are already the live defaults, so nothing was "
                "written.") in page.body
        assert "Saved and verified" not in page.body
        assert "Live trading defaults unchanged" in page.body
        assert _parse(page.body).links == ["/confirm"]
        assert ">Back to the trading page</a>" in page.body
        assert app._start_process.calls == []

    def test_a_change_still_writes_once_and_goes_to_the_saved_page(self, monkeypatch):
        _save(_BASE_SETTINGS)
        saves = self._counting(monkeypatch)
        app = _app()
        response = _post(app, _page_form(app, _query(k="0.75")))
        assert response.status == 303 and response.location == "/saved"
        assert saves == [replace(_BASE_SETTINGS, interval_discount=0.75)]
        page = _get(app, "/saved")
        assert "Saved and verified" in page.body and "Nothing changed" not in page.body

    @pytest.mark.parametrize("query", ["x=1", "unchanged=0", "unchanged=1&x=1", "unchanged",
                                       "unchanged=1&unchanged=1", "UNCHANGED=1"])
    def test_the_saved_page_takes_no_other_query(self, query):
        _save(_BASE_SETTINGS)
        response = _get(_app(), f"/saved?{query}")
        assert response.status == 400
        assert "<form" not in response.body

    def test_trade_and_dry_run_with_nothing_changed_write_nothing(self, monkeypatch):
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        saves = self._counting(monkeypatch)
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _home_form(app, "confirm-trade")))
        app._runs.clear()
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))
        assert saves == [] and len(starter.calls) == 2
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before


class TestFraming:
    """Only the live dashboard's tab page may frame this server's pages, and only the three
    hash-pinned scripts run; the pages that carry a message say what they show."""

    _ORIGINS = ("http://127.0.0.1:8766", "http://localhost:8766")

    def test_the_tab_page_s_two_origins_are_config_s(self):
        assert config.LIVE_DASHBOARD_ORIGINS == self._ORIGINS
        backtest = f"http://{config.LIVE_DASHBOARD_HOST}:{config.LIVE_BACKTEST_PORT}"
        assert backtest not in config.LIVE_DASHBOARD_ORIGINS
        assert all(str(config.DEFAULTS_SERVER_PORT) not in o for o in self._ORIGINS)

    def test_every_route_allows_only_the_tab_page_to_frame_it(self):
        _save(_BASE_SETTINGS)
        _disk_run(result=_result())
        app = _app(start_process=_Starter())
        answers = [_get(app, target) for target in (
            "/", "/confirm", "/confirm?" + _query(), "/confirm?k", "/saved", "/saved?x=1",
            "/trade", "/runs/0123456789abcdef", "/runs/nope", "/checkout", "/nope")]
        answers += [_get(app, "/", host="evil.test"), _post(app, []),
                    _post(app, [], target="/trade"),
                    _post(app, _page_form(app, _query(k="0.75")))]
        expected_scripts = "script-src " + " ".join(
            f"'{defaults_server._script_hash(s)}'" for s in _TRADING_SCRIPTS)
        for response in answers:
            headers = dict(defaults_server._response_headers(response))
            assert "X-Frame-Options" not in headers
            parts = headers["Content-Security-Policy"].split("; ")
            assert [p for p in parts if p.startswith("frame-ancestors")] == [
                "frame-ancestors " + " ".join(self._ORIGINS)]
            assert [p for p in parts if p.startswith("script-src")] == [expected_scripts]
            assert "form-action 'self'" in parts and "default-src 'none'" in parts

    def test_each_hash_is_its_script_s(self):
        for script in _TRADING_SCRIPTS:
            digest = b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
            assert defaults_server._script_hash(script) == f"sha256-{digest}"
        assert defaults_server._CONFIRM_JS_HASH == defaults_server._script_hash(
            defaults_server._CONFIRM_JS)

    def test_the_trading_page_tells_its_proposal(self):
        for query, source in ((_query(source=DASHBOARD_NOTE, k="0.75"), DASHBOARD_NOTE),
                              (defaults_server._seed_query(), config.LIVE_DEFAULTS_SEED_SOURCE),
                              (_query(), "")):
            response = _get(_app(), f"/confirm?{query}")
            settings, read = defaults_server._proposal(defaults_server._params(query), None)
            assert read == source
            message = json.loads(_parse(response.body).body["data-dashboard-message"])
            assert message == {"type": "kalshi-trading",
                               "query": defaults_server._settings_query(settings, source=source)}
            # The message's query proposes exactly the page's proposal
            assert defaults_server._proposal(defaults_server._params(message["query"]),
                                             None) == (settings, source)

    @pytest.mark.parametrize("running", [True, False])
    def test_a_run_page_tells_its_run_and_carries_only_the_notifier(self, running):
        app = _app()
        run_id = _memory_run(app, returncode=None if running else 0, result=None if running
                             else _result())
        page = _parse(_get(app, f"/runs/{run_id}").body)
        assert json.loads(page.body["data-dashboard-message"]) == {"type": "kalshi-run",
                                                                   "run": run_id}
        assert page.scripts == [defaults_server._NOTIFY_JS]
        assert page.forms == [] and page.buttons == []


def _names_file(categories=(), tags=(), **changes) -> Path:
    """
    Write the backtest page's names file into the test's checkout.

    Args:
        categories: Its categories, in order.
        tags: Its tags ("Category · Tag"), in order.
        **changes: Other keys to set (format, say), or with None remove.

    Returns:
        Path: The file.
    """
    record = {"format": config.BACKTEST_FILTER_NAMES_FORMAT, "categories": list(categories),
              "tags": list(tags), **changes}
    path = config.PROJECT_ROOT / config.BACKTEST_FILTER_NAMES_FILENAME
    path.write_text(json.dumps({k: v for k, v in record.items() if v is not None}),
                    encoding="utf-8")
    return path


def _bar_query(page: _PageParser, choose: dict | None = None,
               ticks: dict | None = None) -> str:
    """
    Build the query a browser sends for the dropdown bar, after the bar script's settle().

    One dropdown may be set first (choose), and a menu's boxes ticked anew
    (ticks). Then settle()'s rules: the spread band's choice goes into
    spread_min and spread_max, and Rebalance no turns Min. price movement to
    "off" and Rebalance until to "0", Sell off turns Min. days to maturity to
    "off" (each a value its dropdown must offer, or a browser would select
    nothing). Then the form's named controls in page order, as a browser
    serialises a GET form: each named dropdown's selected option, each
    ticked box, each hidden field; the unnamed band dropdown and the
    unnamed Apply buttons send nothing.

    Args:
        page (_PageParser): The parsed trading page.
        choose (dict | None): Dropdown name (the band's id, tb-band) -> the
            option value to select.
        ticks (dict | None): Menu field name -> the box values to tick (all
            others unticked); a menu left out keeps the page's ticks.

    Returns:
        str: The query string, encoded as a browser encodes a GET form.
    """
    choose, ticks = choose or {}, ticks or {}
    values, offered = {}, {}
    for control in page.bar_controls:
        if control["tag"] == "select":
            key = control.get("name") or control["id"]
            offered[key] = [option["value"] for option in control["options"]]
            selected = [option["value"] for option in control["options"] if option["selected"]]
            values[key] = choose.get(key, selected[0] if selected else offered[key][0])
            assert values[key] in offered[key], (key, values[key])
    hidden = {c["name"]: c["value"] for c in page.bar_controls if c.get("type") == "hidden"}

    def force(key: str, value: str) -> None:
        """
        Set a dropdown as settle() does, failing when it does not offer the value.

        Args:
            key (str): The dropdown's name.
            value (str): The value settle() sets.
        """
        assert value in offered[key], (key, value, offered[key])
        values[key] = value

    hidden["spread_min"], hidden["spread_max"] = values["tb-band"].split("|")
    if values["rebalance"] == "off":
        force("min_price_move", "off")
        force("rebalance_min_days_to_maturity", "0")
    if values["sell_at"] == "off":
        force("sell_min_days", "off")
    pairs = []
    for control in page.bar_controls:
        name = control.get("name")
        if control["tag"] == "select":
            if name:
                pairs.append((name, values[name]))
        elif control.get("type") == "hidden":
            pairs.append((name, hidden[name]))
        elif control.get("type") == "checkbox":
            ticked = (control["value"] in ticks[name] if name in ticks
                      else "checked" in control)
            if ticked:
                pairs.append((name, control["value"]))
    return urlencode(pairs)


# Saved defaults with every dropdown's full choices: rebalancing and selling on
_FULL_BASE = LiveSettings(tier_floors=True, spread_band=(0.2, 0.8), interval_discount=0.85,
                          size_cap=0.15, exclude_categories=("Sports", "Economics"),
                          exclude_tags=("Health · Covid",), rebalance=True, min_price_move=0.1,
                          rebalance_min_days_to_maturity=7, sell_at=0.85, sell_min_days=3)
# ... and with the narrowed ones: rebalancing and selling off
_NARROW_BASE = LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                            size_cap=0.1)


class TestTradingBar:
    """The trading page's dropdown bar: the backtest page's eleven controls, its choices, the
    names its menus list, and a form of its own that only asks for the page again."""

    @staticmethod
    def _page(settings: LiveSettings | None = None, query: str | None = None) -> _PageParser:
        """
        Open the trading page and parse it.

        Args:
            settings (LiveSettings | None): Defaults to save first; None saves none.
            query (str | None): The page's query; None opens it with none.

        Returns:
            _PageParser: The parsed page.
        """
        if settings is not None:
            _save(settings)
        response = _get(_app(), "/confirm" + ("" if query is None else f"?{query}"))
        assert response.status == 200, response.body[:500]
        return _parse(response.body)

    @staticmethod
    def _offered(page: _PageParser, key: str) -> list[str]:
        """
        List one dropdown's option values.

        Args:
            page (_PageParser): The parsed page.
            key (str): The dropdown's name (or tb-band).

        Returns:
            list[str]: Its option values, in order.
        """
        return [option["value"] for option in page.bar_select(key)["options"]]

    @staticmethod
    def _chosen(page: _PageParser, key: str) -> tuple[str, str]:
        """
        Read one dropdown's selected option.

        Args:
            page (_PageParser): The parsed page.
            key (str): The dropdown's name (or tb-band).

        Returns:
            tuple[str, str]: The selected option's value and text.
        """
        [option] = [o for o in page.bar_select(key)["options"] if o["selected"]]
        return option["value"], option["text"]

    def test_eleven_controls_in_the_backtest_bar_s_order_and_words(self):
        from kalshi_betting import dashboard

        page = self._page()
        labels = [label.strip().removesuffix(":") for label in page.bar_labels]
        # The backtest page's bar, read from its own source: its dropdowns'
        # labels in order, then its two menus' labels
        source = inspect.getsource(dashboard._filter_bar_html)
        expected = re.findall(r"<label>([^<:]+): <select id=\"flt-", source)
        expected += [dashboard._MENU_CAT_LABEL, dashboard._MENU_TAG_LABEL]
        assert len(expected) == 11
        assert labels == expected == [
            "Spread band", "Tier floors", "k", "Size cap", "Rebalance", "Min. price movement",
            "Rebalance until", "Sell", "Min. days to maturity", "Exclude categories",
            "Exclude tags"]
        # The dropdowns in that order, the band by its id, the rest by the field they send
        assert [c.get("name") or c.get("id") for c in page.bar_controls
                if c["tag"] == "select"] == [
            "tb-band", "tier_floors", "k", "size_cap", "rebalance", "min_price_move",
            "rebalance_min_days_to_maturity", "sell_at", "sell_min_days"]

    def test_it_is_a_form_of_its_own_that_only_asks_for_the_page(self):
        page = self._page(_FULL_BASE)
        assert page.bar_form["method"] == "get" and page.bar_form["action"] == "/confirm"
        assert page.forms == [("get", "/confirm"), ("post", "/confirm")]
        # Under the heading, above everything else on the page
        body = _get(_app(), "/confirm").body
        assert body.index("<h1>") < body.index('<form id="tb-form"') < body.index("<table>")
        # Its buttons (each menu's Apply) name nothing, so add no field; no
        # dropdown is ever disabled (a disabled one is not sent); none names
        # a field the posting form signs but a proposal field
        assert [b.get("type") for b in page.bar_buttons] == ["submit", "submit"]
        assert all("name" not in b and "id" not in b for b in page.bar_buttons)
        assert not any("disabled" in c for c in page.bar_controls)
        names = {c.get("name") for c in page.bar_controls} - {None}
        assert names <= set(defaults_server._FIELDS)
        assert not names & {"action", "fingerprint", "nonce", "token", "ack_attention"}
        # The bar's hidden fields are not the posting form's
        assert ("spread_min", "0.2") in page.hidden and len(
            [n for n, _ in page.hidden if n == "spread_min"]) == 1

    def test_each_dropdown_shows_the_proposal_in_the_table_s_words(self):
        page = self._page(_FULL_BASE)
        query = defaults_server._params(defaults_server._settings_query(_FULL_BASE))
        assert self._chosen(page, "tb-band") == ("0.2|0.8", "0.2-0.8")
        for name, _ in defaults_server._BAR_SELECT_LABELS:
            value, _ = self._chosen(page, name)
            assert [value] == query[name], name
        # Each chosen option reads as the table under the bar reads the value
        words = {"tb-band": "spread band", "tier_floors": "tier floors", "k": "k",
                 "size_cap": "per-trade cap", "rebalance": "rebalance",
                 "min_price_move": "min price movement",
                 "rebalance_min_days_to_maturity": "rebalance until", "sell_at": "sell at",
                 "sell_min_days": "min days to maturity"}
        for key, row in words.items():
            assert self._chosen(page, key)[1] == page.cells[row][2], key
        # The hidden band fields hold the proposal's band
        hidden = {c["name"]: c["value"] for c in page.bar_controls if c.get("type") == "hidden"}
        assert (hidden["spread_min"], hidden["spread_max"]) == ("0.2", "0.8")

    @pytest.mark.parametrize("constant", ["DEFAULTS_SERVER_SPREAD_BANDS",
                                          "DEFAULTS_SERVER_SIZE_CAPS"])
    def test_config_s_lists_of_choices_feed_the_bar_alone(self, constant):
        # The live path may not name the backtest's band grids (the AST pin in
        # tests/test_strategy.py); config's copy is a list of dropdown choices,
        # so it feeds this bar and nothing else, here or in any other module
        tree = ast.parse(inspect.getsource(defaults_server))
        uses = [node for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == constant]
        [bar] = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_bar_html"]
        inside = {id(node) for node in ast.walk(bar)}
        assert uses and all(id(node) in inside for node in uses)
        for info in pkgutil.iter_modules(kalshi_betting.__path__):
            if info.name in ("config", "defaults_server"):
                continue
            module = importlib.import_module(f"kalshi_betting.{info.name}")
            assert constant not in inspect.getsource(module), info.name

    def test_the_choices_are_config_s_grids(self):
        page = self._page(_FULL_BASE)
        assert self._offered(page, "tb-band") == [
            f"{lo!r}|{hi!r}" for lo, hi in config.DEFAULTS_SERVER_SPREAD_BANDS]
        assert len(config.DEFAULTS_SERVER_SPREAD_BANDS) == 36
        assert self._offered(page, "tier_floors") == ["on", "off"]
        assert self._offered(page, "k") == [repr(k) for k in config.INTERVAL_DISCOUNT_SWEEP]
        assert self._offered(page, "size_cap") == [
            repr(cap) for cap in config.DEFAULTS_SERVER_SIZE_CAPS]
        assert self._offered(page, "rebalance") == ["off", "on"]
        assert self._offered(page, "min_price_move") == ["off"] + [
            repr(m) for m in config.MIN_PRICE_MOVES]
        assert self._offered(page, "rebalance_min_days_to_maturity") == [
            str(d) for d in config.REBALANCE_MATURITY_DAYS]
        assert self._offered(page, "sell_at") == ["off"] + [
            repr(level) for level in config.TAKE_PROFIT_LEVELS]
        assert self._offered(page, "sell_min_days") == ["off"] + [
            str(d) for d in config.TAKE_PROFIT_MIN_DAYS]
        # Each option's words are config's words for that value
        texts = {o["value"]: o["text"] for o in page.bar_select("min_price_move")["options"]}
        assert texts["off"] == "any" and texts["0.15"] == "15c"
        texts = {o["value"]: o["text"] for o in page.bar_select("size_cap")["options"]}
        assert texts["1.0"] == "100% (no cap)" and texts["0.05"] == "5%"
        texts = {o["value"]: o["text"] for o in page.bar_select(
            "rebalance_min_days_to_maturity")["options"]}
        assert texts == {"0": "maturity", "1": "1 day before maturity",
                         "7": "7 days before maturity"}
        texts = {o["value"]: o["text"] for o in page.bar_select("tb-band")["options"]}
        assert texts["0.0|1.0"] == "none" and texts["0.0|0.5"] == "0-0.5"
        texts = {o["value"]: o["text"] for o in page.bar_select("sell_at")["options"]}
        assert texts["off"] == "off" and texts["0.8"] == "80% of potential profit"
        texts = {o["value"]: o["text"] for o in page.bar_select("rebalance")["options"]}
        assert texts == {"off": "no", "on": "yes"}

    def test_a_value_off_the_grid_is_offered_once_in_its_place(self):
        off_grid = LiveSettings(tier_floors=False, spread_band=(0.1, 0.45),
                                interval_discount=1 / 3, size_cap=0.1, rebalance=True,
                                min_price_move=1.0, sell_at=0.5, sell_min_days=10)
        page = self._page(off_grid)
        for key, value in (("tb-band", "0.1|0.45"), ("k", repr(1 / 3)),
                           ("min_price_move", "1.0"), ("sell_at", "0.5"),
                           ("sell_min_days", "10")):
            offered = self._offered(page, key)
            assert offered.count(value) == 1, key
            assert self._chosen(page, key)[0] == value, key
        # In order: smallest first, "off" or "any" ahead of every number
        assert self._offered(page, "k")[0] == repr(1 / 3)
        assert self._offered(page, "min_price_move")[-1] == "1.0"
        assert self._offered(page, "sell_at")[:2] == ["off", "0.5"]
        assert self._offered(page, "sell_min_days")[-3:] == ["10", "14", "21"]
        assert self._offered(page, "tb-band").index("0.1|0.45") == 6
        # On the grid it is offered once, never twice
        page = self._page(_FULL_BASE)
        for key in ("tb-band", "k", "size_cap", "min_price_move", "sell_at", "sell_min_days"):
            offered = self._offered(page, key)
            assert len(offered) == len(set(offered)), key

    def test_choices_that_do_not_apply_are_not_offered(self):
        page = self._page(_NARROW_BASE)
        # Rebalance no: no minimum price movement, no days limit; Sell off: no minimum of days
        assert self._offered(page, "min_price_move") == ["off"]
        assert [o["text"] for o in page.bar_select("min_price_move")["options"]] == ["any"]
        assert self._offered(page, "rebalance_min_days_to_maturity") == ["0"]
        assert self._offered(page, "sell_min_days") == ["off"]
        # ... and never by disabling a dropdown
        assert not any("disabled" in c for c in page.bar_controls)
        # Rebalance yes with selling off: the rebalance choices open, the days do not
        page = self._page(replace(_NARROW_BASE, rebalance=True))
        assert len(self._offered(page, "min_price_move")) == 1 + len(config.MIN_PRICE_MOVES)
        assert self._offered(page, "sell_min_days") == ["off"]

    @pytest.mark.parametrize("source, kept", [
        (DASHBOARD_NOTE, True), (config.LIVE_DEFAULTS_SEED_SOURCE, False), (None, False)])
    def test_the_source_note_is_kept_but_never_the_seed_s(self, source, kept):
        query = (defaults_server._seed_query() if source == config.LIVE_DEFAULTS_SEED_SOURCE
                 else _query(source=source))
        page = self._page(query=query)
        sources = [c["value"] for c in page.bar_controls
                   if c.get("type") == "hidden" and c["name"] == "source"]
        assert sources == ([source] if kept else [])

    def test_the_menus_list_the_names_file_and_every_excluded_name(self):
        _names_file(categories=("Economics", "Politics", "sports", "Crypto"),
                    tags=("Sports · Soccer", "Health · COVID", "Sports · Basketball"))
        page = self._page(_FULL_BASE)
        # Each menu: the excluded names first, in the proposal's order, ticked;
        # then the file's others, unticked; a second spelling of a listed name
        # in another letter case is left out
        boxes = [(b["value"], "checked" in b) for b in page.bar_boxes("exclude_category")]
        assert boxes == [("Sports", True), ("Economics", True), ("Politics", False),
                         ("Crypto", False)]
        boxes = [(b["value"], "checked" in b) for b in page.bar_boxes("exclude_tag")]
        assert boxes == [("Health · Covid", True), ("Sports · Soccer", False),
                         ("Sports · Basketball", False)]
        assert page.bar_notes == []
        body = _get(_app(), "/confirm").body
        assert "<summary>2 excluded</summary>" in body and "<summary>1 excluded</summary>" in body

    def test_the_names_file_is_read_again_for_every_page(self):
        app = _app()
        _names_file(categories=("Economics",))
        first = _parse(_get(app, "/confirm").body)
        _names_file(categories=("Economics", "Politics"))
        second = _parse(_get(app, "/confirm").body)
        assert [b["value"] for b in first.bar_boxes("exclude_category")] == ["Economics"]
        assert [b["value"] for b in second.bar_boxes("exclude_category")] == [
            "Economics", "Politics"]

    @pytest.mark.parametrize("contents, why", [
        (None, "there is no backtest_filter_names.json beside the backtest dashboard"),
        (b"not json", "cannot be used"),
        (b'{"format": "backtest-filter-names-v1", "categories": [NaN], "tags": []}',
         "which is not JSON"),
        (b'{"format": "backtest-filter-names-v2", "categories": [], "tags": []}',
         "it is not a backtest-filter-names-v1 file"),
        (b'["Sports"]', "it is not a backtest-filter-names-v1 file"),
        (b'{"format": "backtest-filter-names-v1", "categories": ["Sports"], "tags": [], '
         b'"tags": ["X"]}', "it gives a key twice"),
        (b'{"format": "backtest-filter-names-v1", "categories": "Sports", "tags": []}',
         "its categories are not a list of names"),
        (b'{"format": "backtest-filter-names-v1", "categories": [], "tags": [5]}',
         "its tags are not a list of names"),
        (b'{"format": "backtest-filter-names-v1", "categories": ["Sports"]}',
         "its tags are not a list of names"),
        (b'{"format": "backtest-filter-names-v1", "categories": ["any"], "tags": []}',
         "cannot hold the name"),
        ('{"format": "backtest-filter-names-v1", "categories": ["Sports · Soccer"], '
         '"tags": []}'.encode(), "only a whole category can be excluded"),
        (b'{"format": "backtest-filter-names-v1", "categories": [" Sports"], '
         b'"tags": []}', "must be one printable name"),
        ('{"format": "backtest-filter-names-v1", "categories": [], '
         '"tags": ["Sports  · Soccer"]}'.encode(), "must not hold two spaces in a row"),
        ('{"format": "backtest-filter-names-v1", "categories": [], '
         '"tags": ["Sports ·"]}'.encode(), "Category · Tag"),
        (b"\xff\xfe", "cannot be used"),
        (b"[" * 5000, "cannot be used"),
    ], ids=["missing", "not-json", "nan", "other-format", "not-an-object", "key-twice",
            "categories-not-a-list", "tag-not-a-string", "no-tags", "any", "tied-category",
            "spaces-around", "double-space", "half-tag", "not-utf8", "too-deep"])
    def test_a_file_that_cannot_be_used_lists_only_the_excluded_names(self, contents, why):
        path = config.PROJECT_ROOT / config.BACKTEST_FILTER_NAMES_FILENAME
        if contents is not None:
            path.write_bytes(contents)
        names = defaults_server._filter_names()
        assert names.categories == () and names.tags == ()
        assert why in names.note, names.note
        assert names.note.startswith("The Exclude menus list only the names excluded here: ")
        page = self._page(_FULL_BASE)
        assert [b["value"] for b in page.bar_boxes("exclude_category")] == ["Sports",
                                                                            "Economics"]
        assert [b["value"] for b in page.bar_boxes("exclude_tag")] == ["Health · Covid"]
        [note] = page.bar_notes
        assert note == names.note
        assert "Build the backtest dashboard (python3 -m kalshi_betting.backtest)" in note

    def test_a_file_longer_than_the_limit_is_not_read(self, monkeypatch):
        path = _names_file(categories=("Economics",))
        monkeypatch.setattr(defaults_server, "BACKTEST_FILTER_NAMES_MAX_BYTES",
                            len(path.read_bytes()) - 1)
        names = defaults_server._filter_names()
        assert names.categories == () and f"is longer than {len(path.read_bytes()) - 1} bytes" \
            in names.note
        # At exactly the limit it is read
        monkeypatch.setattr(defaults_server, "BACKTEST_FILTER_NAMES_MAX_BYTES",
                            len(path.read_bytes()))
        assert defaults_server._filter_names() == defaults_server._FilterNames(
            ("Economics",), (), None)
        assert config.BACKTEST_FILTER_NAMES_MAX_BYTES == 1_048_576

    def test_a_file_that_cannot_be_opened_never_raises(self, monkeypatch):
        path = _names_file(categories=("Economics",))
        real_open = os.open

        def refuse(file, flags, *args, **kwargs):
            """
            Stand in for os.open, refusing the names file as a file with no read permission would.

            Args:
                file: The path.
                flags (int): The open flags.
                *args: Passed on.
                **kwargs: Passed on.

            Returns:
                int: The real descriptor, for any other file.

            Raises:
                PermissionError: For the names file.
            """
            if Path(file) == path:
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real_open(file, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", refuse)
        names = defaults_server._filter_names()
        assert names.categories == () and "could not be read (PermissionError)" in names.note
        assert _get(_app(), "/confirm").status == 200

    @staticmethod
    def _read_names_quickly() -> defaults_server._FilterNames:
        """
        Read the names file on a thread of its own, failing if the read waits.

        Returns:
            defaults_server._FilterNames: What _filter_names returned.
        """
        done = []
        reader = threading.Thread(target=lambda: done.append(defaults_server._filter_names()),
                                  daemon=True)
        reader.start()
        reader.join(10)
        assert done, "reading the names file waited"
        return done[0]

    @pytest.mark.parametrize("kind", ["folder", "fifo", "link-to-a-device"])
    def test_a_names_file_that_is_not_a_plain_file_is_never_read(self, kind):
        # A FIFO with no writer, or a terminal, would keep an ordinary read
        # waiting, and the server answers one request at a time
        path = config.PROJECT_ROOT / config.BACKTEST_FILTER_NAMES_FILENAME
        if kind == "folder":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(os.devnull)
        names = self._read_names_quickly()
        assert names.categories == () and names.tags == ()
        assert "backtest_filter_names.json is not a plain file" in names.note
        assert _get(_app(), "/confirm").status == 200

    def test_a_fifo_that_slips_past_the_check_is_never_waited_on(self, monkeypatch):
        # Swapped in after the plain-file check: the open and the read return at once
        path = config.PROJECT_ROOT / config.BACKTEST_FILTER_NAMES_FILENAME
        os.mkfifo(path)
        real_is_file = Path.is_file
        monkeypatch.setattr(Path, "is_file", lambda self: self == path or real_is_file(self))
        names = self._read_names_quickly()
        assert names.categories == () and "cannot be used" in names.note

    def test_each_list_keeps_at_most_the_names_a_link_may_name(self, monkeypatch):
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_MAX_FILTER_NAMES", 2)
        _names_file(categories=("A", "B", "C"), tags=("A · x", "B · y"))
        names = defaults_server._filter_names()
        assert names.categories == ("A", "B") and names.tags == ("A · x", "B · y")
        assert "the first 2 categories and the first 2 tags" in names.note
        page = self._page()
        [note] = page.bar_notes
        assert note == names.note

    def test_a_hostile_name_in_the_file_is_escaped(self):
        _names_file(categories=("<script>alert(1)</script>",), tags=('Sports · "><b>x',))
        body = _get(_app(), "/confirm").body
        assert "<script>alert" not in body
        page = _parse(body)
        assert page.scripts == _TRADING_SCRIPTS
        assert [b["value"] for b in page.bar_boxes("exclude_category")] == [
            "<script>alert(1)</script>"]
        assert [b["value"] for b in page.bar_boxes("exclude_tag")] == ['Sports · "><b>x']


class TestBarRoundTrip:
    """Every choice of every dropdown, sent as a browser sends the bar after the bar script's
    settle(), is a proposal the server takes and proposes exactly; so are the menus' ticks."""

    @staticmethod
    def _expected(base: LiveSettings, key: str, value: str) -> LiveSettings:
        """
        Work out what choosing one option should propose, from the option's value alone.

        Args:
            base (LiveSettings): The proposal the bar showed.
            key (str): The dropdown's name (tb-band for the spread band).
            value (str): The option's value.

        Returns:
            LiveSettings: The base with that one choice made, and what settle()
                turns off with it.
        """
        off = value == "off"
        if key == "tb-band":
            lo, hi = value.split("|")
            return replace(base, spread_band=(float(lo), float(hi)))
        if key == "tier_floors":
            return replace(base, tier_floors=value == "on")
        if key == "k":
            return replace(base, interval_discount=float(value))
        if key == "size_cap":
            return replace(base, size_cap=float(value))
        if key == "rebalance":
            return (replace(base, rebalance=True) if value == "on" else replace(
                base, rebalance=False, min_price_move=None,
                rebalance_min_days_to_maturity=None))
        if key == "min_price_move":
            return replace(base, min_price_move=None if off else float(value))
        if key == "rebalance_min_days_to_maturity":
            return replace(base, rebalance_min_days_to_maturity=int(value) or None)
        if key == "sell_at":
            return (replace(base, sell_at=None, sell_min_days=None) if off
                    else replace(base, sell_at=float(value)))
        assert key == "sell_min_days"
        return replace(base, sell_min_days=None if off else int(value))

    @pytest.mark.parametrize("base", [_FULL_BASE, _NARROW_BASE], ids=["full", "narrowed"])
    def test_every_choice_is_proposed_exactly(self, base):
        _save(base)
        app = _app()
        page = _parse(_get(app, "/confirm").body)
        # Unchanged, the bar asks for exactly what it shows: nothing changes
        response = _get(app, "/confirm?" + _bar_query(page))
        assert response.status == 200 and "0 of 11 settings change." in response.body
        tried = 0
        for control in page.bar_controls:
            if control["tag"] != "select":
                continue
            key = control.get("name") or control["id"]
            for option in control["options"]:
                query = _bar_query(page, {key: option["value"]})
                response = _get(app, "/confirm?" + query)
                assert response.status == 200, (key, option["value"], response.body[:400])
                expected = self._expected(base, key, option["value"])
                assert defaults_server._proposal(defaults_server._params(query), base) == (
                    expected, ""), (key, option["value"])
                # The page it opens proposes that, and its bar shows the choice made
                again = _parse(response.body)
                assert json.loads(again.body["data-dashboard-message"])["query"] == (
                    defaults_server._settings_query(expected)), (key, option["value"])
                assert [o["value"] for o in again.bar_select(key)["options"]
                        if o["selected"]] == [option["value"]], (key, option["value"])
                tried += 1
        # Every option of the nine dropdowns was tried
        assert tried == sum(len(c["options"]) for c in page.bar_controls if c["tag"] == "select")
        assert tried > (100 if base is _FULL_BASE else 80)

    def test_the_menus_ticks_are_proposed_in_their_order(self):
        _names_file(categories=("Politics", "Crypto"), tags=("Sports · Soccer",))
        _save(_FULL_BASE)
        app = _app()
        page = _parse(_get(app, "/confirm").body)
        for ticks, categories, tags in (
                ({"exclude_category": {"Sports", "Economics", "Crypto"}},
                 ("Sports", "Economics", "Crypto"), ("Health · Covid",)),
                ({"exclude_category": set(), "exclude_tag": set()}, None, None),
                ({"exclude_tag": {"Health · Covid", "Sports · Soccer"}},
                 ("Sports", "Economics"), ("Health · Covid", "Sports · Soccer"))):
            query = _bar_query(page, ticks=ticks)
            response = _get(app, "/confirm?" + query)
            assert response.status == 200, ticks
            settings, _ = defaults_server._proposal(defaults_server._params(query), None)
            assert settings == replace(_FULL_BASE, exclude_categories=categories,
                                       exclude_tags=tags), ticks

    @staticmethod
    def _each_choice(page: _PageParser):
        """
        List every option of every dropdown of a page's bar.

        Args:
            page (_PageParser): The parsed trading page.

        Returns:
            list[tuple[str, str]]: (the dropdown's name, the band's id for the
                spread band; the option's value), in page order.
        """
        return [(control.get("name") or control["id"], option["value"])
                for control in page.bar_controls if control["tag"] == "select"
                for option in control["options"]]

    @pytest.mark.parametrize("base", [_FULL_BASE, _NARROW_BASE], ids=["full", "narrowed"])
    def test_a_kept_source_note_goes_with_every_choice(self, base):
        app = _app()
        query = defaults_server._settings_query(base, source=DASHBOARD_NOTE)
        page = _parse(_get(app, f"/confirm?{query}").body)
        choices = self._each_choice(page)
        assert len(choices) > (100 if base is _FULL_BASE else 80)
        for key, value in choices:
            sent = _bar_query(page, {key: value})
            assert defaults_server._proposal(defaults_server._params(sent), None) == (
                self._expected(base, key, value), DASHBOARD_NOTE), (key, value)

    def test_the_seed_s_note_goes_with_no_choice(self):
        # The seed's note is dropped, so a changed seed is never refused for it
        app = _app()
        page = _parse(_get(app, "/confirm?" + defaults_server._seed_query()).body)
        for key, value in self._each_choice(page):
            sent = _bar_query(page, {key: value})
            assert defaults_server._proposal(defaults_server._params(sent), None) == (
                self._expected(config.LIVE_DEFAULTS_SEED, key, value), ""), (key, value)
        assert _get(app, "/confirm?" + _bar_query(page, {"k": "0.9"})).status == 200

    def test_the_bar_s_spelling_is_the_query_s(self):
        # One spelling: _settings_query and every bar option use _query_value
        for settings in (_FULL_BASE, _NARROW_BASE, config.LIVE_DEFAULTS_SEED):
            query = defaults_server._params(defaults_server._settings_query(settings))
            assert query["tier_floors"] == [defaults_server._query_value(
                "tier_floors", settings.tier_floors)]
            assert query["rebalance_min_days_to_maturity"] == [defaults_server._query_value(
                "rebalance_min_days_to_maturity", settings.rebalance_min_days_to_maturity)]
            assert query["sell_min_days"] == [defaults_server._query_value(
                "sell_min_days", settings.sell_min_days)]
        assert defaults_server._query_value("k", 1 / 3) == repr(1 / 3)
        assert defaults_server._query_value("min_price_move", None) == "off"
        assert defaults_server._query_value("rebalance_min_days_to_maturity", None) == "0"
        assert defaults_server._query_value("sell_min_days", 7) == "7"


# A stand-in for the few browser pieces the bar script touches: the bar form
# (its elements by name or id, each with a value; its dropdowns, which take
# listeners; its own listeners; and submit(), which records the elements'
# values as a GET would send them, and whether the page was inert then) and the
# body, whose inert switch starts off. The script runs inside a function whose
# parameter is the stand-in document. The steps: ["set", key, value] sets an
# element's value, ["change", key] fires that dropdown's change listeners,
# ["submit"] fires the form's submit listeners (as an Apply button does, without
# submit()), ["snap", label] records every value, each submit() made, whether
# the page was inert at each, whether it is inert now, the form's listeners and
# each dropdown's
_BAR_HARNESS = """
var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __formListeners = {};
var __submitted = [];
var __inertAtSubmit = [];
var __body = {inert: false};
var __elements = {};
var __selects = [];
__CONTROLS__.forEach(function(c) {
  var el = {value: c.value, listeners: {}};
  el.addEventListener = function(type, fn) {
    (el.listeners[type] = el.listeners[type] || []).push(fn);
  };
  __elements[c.key] = el;
  if (c.select) { __selects.push(el); }
});
function __values() {
  var values = {};
  Object.keys(__elements).forEach(function(k) { values[k] = __elements[k].value; });
  return values;
}
var __form = {
  elements: __elements,
  addEventListener: function(type, fn) {
    (__formListeners[type] = __formListeners[type] || []).push(fn);
  },
  getElementsByTagName: function(tag) { return tag === 'select' ? __selects : []; },
  submit: function() { __submitted.push(__values()); __inertAtSubmit.push(__body.inert); }
};
var __document = {
  body: __body,
  getElementById: function(id) { return (__PRESENT__ && id === 'tb-form') ? __form : null; }
};
(function(document) {
__SCRIPT__
})(__document);
var __out = {};
__STEPS__.forEach(function(step) {
  if (step[0] === 'set') {
    __elements[step[1]].value = step[2];
  } else if (step[0] === 'change') {
    (__elements[step[1]].listeners.change || []).forEach(function(fn) { fn({type: 'change'}); });
  } else if (step[0] === 'submit') {
    (__formListeners.submit || []).forEach(function(fn) { fn({type: 'submit'}); });
  } else if (step[0] === 'snap') {
    __out[step[1]] = {
      values: __values(),
      submitted: __submitted.slice(),
      inert_at_submit: __inertAtSubmit.slice(),
      inert: __body.inert,
      form: Object.keys(__formListeners).sort(),
      selects: __selects.map(function(el) { return Object.keys(el.listeners).sort(); })
    };
  }
});
__emit(JSON.stringify(__out));
"""

# The stand-in's dropdowns, in the bar's order, and its two hidden band
# fields, at a proposal that rebalances and sells
_BAR_CONTROLS = [
    {"key": "tb-band", "value": "0.2|0.8", "select": True},
    {"key": "spread_min", "value": "0.2", "select": False},
    {"key": "spread_max", "value": "0.8", "select": False},
    {"key": "tier_floors", "value": "on", "select": True},
    {"key": "k", "value": "0.85", "select": True},
    {"key": "size_cap", "value": "0.15", "select": True},
    {"key": "rebalance", "value": "on", "select": True},
    {"key": "min_price_move", "value": "0.1", "select": True},
    {"key": "rebalance_min_days_to_maturity", "value": "7", "select": True},
    {"key": "sell_at", "value": "0.85", "select": True},
    {"key": "sell_min_days", "value": "3", "select": True},
]


def _run_js(tmp_path, program: str, name: str) -> dict:
    """
    Run a stand-in program under node or jsc and read the one JSON line it prints.

    Skips the test when no JavaScript runtime is present.

    Args:
        tmp_path (Path): Where to write the program.
        program (str): The program's text.
        name (str): The program file's name.

    Returns:
        dict: What it printed, parsed.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("neither node nor macOS's jsc is here to run the page script")
    path = tmp_path / name
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _run_bar_script(tmp_path, steps, *, present: bool = True) -> dict:
    """
    Run the bar script, byte for byte, against the stand-in bar through a list of steps.

    Args:
        tmp_path (Path): Where to write the program.
        steps (list[list[str]]): The steps (see _BAR_HARNESS).
        present (bool): Whether the page has the bar.

    Returns:
        dict: Each "snap" step's record, by its label.
    """
    program = (_BAR_HARNESS
               .replace("__CONTROLS__", json.dumps(_BAR_CONTROLS))
               .replace("__PRESENT__", json.dumps(present))
               .replace("__STEPS__", json.dumps(steps))
               .replace("__SCRIPT__", defaults_server._BAR_JS))
    return _run_js(tmp_path, program, "bar_script.js")


class TestBarScript:
    """The bar script itself, run under node or jsc with a stand-in bar."""

    _START = {c["key"]: c["value"] for c in _BAR_CONTROLS}

    def test_a_change_sends_the_bar_with_the_band_in_its_two_fields(self, tmp_path):
        out = _run_bar_script(tmp_path, [
            ["snap", "load"], ["set", "tb-band", "0.25|0.9"], ["change", "tb-band"],
            ["snap", "sent"]])
        # It listens for each dropdown's change and the form's own submit, nothing else
        assert out["load"]["form"] == ["submit"]
        assert out["load"]["selects"] == [["change"]] * 9
        assert out["load"]["submitted"] == []
        [sent] = out["sent"]["submitted"]
        assert sent == {**self._START, "tb-band": "0.25|0.9", "spread_min": "0.25",
                        "spread_max": "0.9"}

    def test_a_change_elsewhere_keeps_the_rest_as_shown(self, tmp_path):
        out = _run_bar_script(tmp_path, [["set", "k", "0.9"], ["change", "k"], ["snap", "s"]])
        assert out["s"]["submitted"] == [{**self._START, "k": "0.9"}]

    def test_rebalance_no_turns_its_minimum_and_limit_off(self, tmp_path):
        out = _run_bar_script(tmp_path, [
            ["set", "rebalance", "off"], ["change", "rebalance"], ["snap", "s"]])
        assert out["s"]["submitted"] == [{**self._START, "rebalance": "off",
                                          "min_price_move": "off",
                                          "rebalance_min_days_to_maturity": "0"}]

    def test_sell_off_turns_its_minimum_of_days_off(self, tmp_path):
        out = _run_bar_script(tmp_path, [
            ["set", "sell_at", "off"], ["change", "sell_at"], ["snap", "s"]])
        assert out["s"]["submitted"] == [{**self._START, "sell_at": "off",
                                          "sell_min_days": "off"}]

    def test_apply_settles_the_fields_and_leaves_the_sending_to_the_browser(self, tmp_path):
        # A menu's Apply button submits the form itself: the script only settles it
        out = _run_bar_script(tmp_path, [
            ["set", "tb-band", "0.3|1.0"], ["set", "rebalance", "off"], ["submit"],
            ["snap", "s"]])
        assert out["s"]["submitted"] == []
        assert out["s"]["values"] == {**self._START, "tb-band": "0.3|1.0", "spread_min": "0.3",
                                      "spread_max": "1.0", "rebalance": "off",
                                      "min_price_move": "off",
                                      "rebalance_min_days_to_maturity": "0"}

    def test_once_the_bar_is_sent_the_whole_page_is_inert(self, tmp_path):
        # Until the page for the new choice arrives, a button armed on this one
        # would send this page's proposal: from the moment the bar goes, nothing
        # on the page can be clicked or typed into
        out = _run_bar_script(tmp_path, [
            ["snap", "load"], ["set", "k", "0.9"], ["change", "k"], ["snap", "changed"]])
        assert out["load"]["inert"] is False
        # Inert before the bar is sent, not after
        assert out["changed"]["inert_at_submit"] == [True] and out["changed"]["inert"] is True
        # A menu's Apply sends the form itself: the page goes inert as it goes
        out = _run_bar_script(tmp_path, [["snap", "load"], ["submit"], ["snap", "applied"]])
        assert out["load"]["inert"] is False
        assert out["applied"]["inert"] is True and out["applied"]["submitted"] == []

    def test_a_page_without_the_bar_does_nothing(self, tmp_path):
        out = _run_bar_script(tmp_path, [["snap", "s"]], present=False)
        assert out["s"]["form"] == [] and out["s"]["selects"] == [[]] * 9
        assert out["s"]["values"] == self._START
        assert out["s"]["inert"] is False


# A stand-in for the browser pieces the notifier touches: the window (its
# parent, the framing page's window when __FRAMED__, else itself; any listener
# added to it is recorded) and the body's data-dashboard-message attribute
# (__MESSAGE__, or null). It prints every message posted with its target origin,
# and every listener added
_NOTIFY_HARNESS = """
var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };
var __posted = [];
var __listeners = [];
var __window = {addEventListener: function(type) { __listeners.push('window:' + type); }};
__window.parent = __FRAMED__ ? {postMessage: function(message, origin) {
  __posted.push({message: message, origin: origin});
}} : __window;
var __document = {
  addEventListener: function(type) { __listeners.push('document:' + type); },
  body: {getAttribute: function(name) {
    return name === 'data-dashboard-message' ? __MESSAGE__ : null;
  }}
};
(function(window, document) {
__SCRIPT__
})(__window, __document);
__emit(JSON.stringify({posted: __posted, listeners: __listeners}));
"""


def _run_notifier(tmp_path, *, framed: bool, message: str | None) -> dict:
    """
    Run the notifier, byte for byte, in the stand-in window.

    Args:
        tmp_path (Path): Where to write the program.
        framed (bool): Whether the page is shown in a frame.
        message (str | None): The body's data-dashboard-message, as the
            browser reads the attribute; None for none.

    Returns:
        dict: "posted" (each message and its target origin) and "listeners".
    """
    program = (_NOTIFY_HARNESS
               .replace("__FRAMED__", json.dumps(framed))
               .replace("__MESSAGE__", json.dumps(message))
               .replace("__SCRIPT__", defaults_server._NOTIFY_JS))
    return _run_js(tmp_path, program, "notify_script.js")


class TestNotifyScript:
    """The notifier itself, run under node or jsc: it posts only when framed, only to the live
    dashboard's tab page's two origins, and never listens."""

    def test_framed_it_posts_the_page_s_message_to_both_origins(self, tmp_path):
        _save(_BASE_SETTINGS)
        page = _parse(_get(_app(), "/confirm").body)
        text = page.body["data-dashboard-message"]
        out = _run_notifier(tmp_path, framed=True, message=text)
        assert out["posted"] == [{"message": json.loads(text), "origin": origin}
                                 for origin in config.LIVE_DASHBOARD_ORIGINS]
        assert out["listeners"] == []

    def test_a_run_page_s_message_too(self, tmp_path):
        app = _app()
        run_id = _memory_run(app, returncode=None)
        text = _parse(_get(app, f"/runs/{run_id}").body).body["data-dashboard-message"]
        out = _run_notifier(tmp_path, framed=True, message=text)
        assert [p["message"] for p in out["posted"]] == [{"type": "kalshi-run", "run": run_id}] * 2

    def test_not_framed_or_without_a_message_it_posts_nothing(self, tmp_path):
        text = json.dumps({"type": "kalshi-run", "run": "0123456789abcdef"})
        for framed, message in ((False, text), (True, None), (True, "")):
            out = _run_notifier(tmp_path, framed=framed, message=message)
            assert out == {"posted": [], "listeners": []}, (framed, message)

    def test_neither_script_can_press_a_button_or_listen_for_messages(self):
        # The arming script's buttons are named only by it: the bar script and
        # the notifier name none of them, no action, and no way to press,
        # submit the other form or hear another page
        for script in (defaults_server._BAR_JS, defaults_server._NOTIFY_JS):
            for word in ("confirm", "action", "click", "requestSubmit", "dispatchEvent",
                         "addEventListener('message'", 'addEventListener("message"',
                         "onmessage", "forms", "querySelector", "innerHTML", "eval", "fetch"):
                assert word not in script, word
            for button_id in defaults_server._ARMED_BUTTONS:
                assert button_id not in script
        assert "addEventListener" not in defaults_server._NOTIFY_JS
        assert "location" not in defaults_server._NOTIFY_JS
        assert set(re.findall(r"addEventListener\('(\w+)'", defaults_server._BAR_JS)) == {
            "submit", "change"}
        # The bar script reaches its own form only, by its id, and outside it
        # only the body's inert switch, which it turns on and never off
        assert re.findall(r"getElementById\('([\w-]+)'\)", defaults_server._BAR_JS) == [
            "tb-form"]
        assert re.findall(r"document\.([\w.]+)", defaults_server._BAR_JS) == [
            "getElementById", "body.inert"]
        assert "document.body.inert = true;" in defaults_server._BAR_JS
        assert "inert = false" not in defaults_server._BAR_JS
        # The notifier's origins are config's, written in as JSON
        assert json.dumps(list(config.LIVE_DASHBOARD_ORIGINS)) in defaults_server._NOTIFY_JS
        assert "__ORIGINS__" not in defaults_server._NOTIFY_JS


class TestConfirmRefusals:
    """A proposal the server cannot save is refused with 400, naming why."""

    @pytest.mark.parametrize("query", [
        _query(foo="1"),                                  # unknown field
        _query() + "&k=0.8",                              # repeated
        _query(k=""),                                     # blank
        _query(source=""),                                # a blank source
        _query(tier_floors="maybe"),
        # A sell level: a plain decimal share on the 1% grid in (0, 1], or off
        _query(sell_at="maybe"), _query(sell_at="Off"), _query(sell_at="on"),
        _query(sell_at="0"), _query(sell_at="-0.5"), _query(sell_at="1.01"),
        _query(sell_at="85"), _query(sell_at="0.855"), _query(sell_at="0.001"),
        _query(sell_at="nan"), _query(sell_at="inf"), _query(sell_at="0_85"),
        _query(sell_at="０.85"), _query(sell_at="٠.85"), _query(sell_at=" 0.85"),
        _query(sell_at="0x1"), _query(sell_at="1e-400"),
        # A minimum of days: plain ASCII digits (at most six) with a level, or off
        _query(sell_at="0.85", sell_min_days="0"), _query(sell_at="0.85", sell_min_days="-1"),
        _query(sell_at="0.85", sell_min_days="1.5"), _query(sell_at="0.85", sell_min_days="1e1"),
        _query(sell_at="0.85", sell_min_days="+3"), _query(sell_at="0.85", sell_min_days=" 3"),
        _query(sell_at="0.85", sell_min_days="３"), _query(sell_at="0.85", sell_min_days="٣"),
        _query(sell_at="0.85", sell_min_days="1" * 7), _query(sell_at="0.85", sell_min_days="soon"),
        _query(sell_at="0.85", sell_min_days="Off"),
        # A minimum with no level: none given and none saved (the seed's is off),
        # or the level named off
        _query(sell_min_days="3"), _query(sell_at="off", sell_min_days="3"),
        # Rebalancing: on or off, spelled exactly so
        _query(rebalance="maybe"), _query(rebalance="true"), _query(rebalance="yes"),
        _query(rebalance="1"), _query(rebalance="On"), _query(rebalance="OFF"),
        # A minimum price movement: plain dollars on the 5c grid, 5c to $1, or off
        _query(rebalance="on", min_price_move="maybe"),
        _query(rebalance="on", min_price_move="Off"),
        _query(rebalance="on", min_price_move="on"),
        _query(rebalance="on", min_price_move="any"),
        _query(rebalance="on", min_price_move="0"),
        _query(rebalance="on", min_price_move="-0.1"),
        _query(rebalance="on", min_price_move="1.05"),
        _query(rebalance="on", min_price_move="10"),
        _query(rebalance="on", min_price_move="5"),
        _query(rebalance="on", min_price_move="0.07"),
        _query(rebalance="on", min_price_move="0.125"),
        _query(rebalance="on", min_price_move="0.01"),
        _query(rebalance="on", min_price_move="10c"),
        _query(rebalance="on", min_price_move="nan"),
        _query(rebalance="on", min_price_move="inf"),
        _query(rebalance="on", min_price_move="0_1"),
        _query(rebalance="on", min_price_move="０.1"),
        _query(rebalance="on", min_price_move="٠.1"),
        _query(rebalance="on", min_price_move=" 0.1"),
        _query(rebalance="on", min_price_move="0x1"),
        _query(rebalance="on", min_price_move="1e-400"),
        _query(rebalance="on", min_price_move="1e400"),
        # A minimum with rebalancing named off
        _query(rebalance="off", min_price_move="0.1"),
        # A days-to-maturity limit: 0, 1 or 7 as plain ASCII digits
        _query(rebalance_min_days_to_maturity="2"), _query(rebalance_min_days_to_maturity="3"),
        _query(rebalance_min_days_to_maturity="14"),
        _query(rebalance_min_days_to_maturity="-1"),
        _query(rebalance_min_days_to_maturity="+7"),
        _query(rebalance_min_days_to_maturity=" 7"),
        _query(rebalance_min_days_to_maturity="7 "),
        _query(rebalance_min_days_to_maturity="7.0"),
        _query(rebalance_min_days_to_maturity="1e1"),
        _query(rebalance_min_days_to_maturity="0x7"),
        _query(rebalance_min_days_to_maturity="seven"),
        _query(rebalance_min_days_to_maturity="off"),
        _query(rebalance_min_days_to_maturity="maturity"),
        _query(rebalance_min_days_to_maturity=""),
        _query(rebalance_min_days_to_maturity="７"),
        _query(rebalance_min_days_to_maturity="٧"),
        _query(rebalance_min_days_to_maturity="1" * 7),
        # A limit with rebalancing named off, or given twice
        _query(rebalance="off", min_price_move="off", rebalance_min_days_to_maturity="1"),
        _query(rebalance="off", min_price_move="off", rebalance_min_days_to_maturity="7"),
        _query(rebalance_min_days_to_maturity="7") + "&rebalance_min_days_to_maturity=7",
        _query(rebalance_min_days_to_maturity="0") + "&rebalance_min_days_to_maturity=7",
        # Either rebalance field given twice
        _query(rebalance="on") + "&rebalance=on",
        _query(rebalance="on", min_price_move="0.1") + "&min_price_move=0.1",
        # A field a link built before Rebalance carries, which it replaced
        _query(add_to_held_pairs="on"), _query(trim_to_kelly="on"),
        _query(trim_to_kelly="on", trim_min_rise="0.1"),
        _query(rebalance="on", add_to_held_pairs="on"),
        _query(k="nan"), _query(k="inf"), _query(k="0_8"), _query(k="1e-400"),
        _query(k="1e400"), _query(k="０.8"), _query(k=" 0.8"), _query(k="0"),
        _query(k="1.5"), _query(k="0x1"),
        _query(size_cap="0.33"),
        # The field a link built before the same-title cap was removed carries,
        # whatever its value
        _query(same_title_size_cap="0.2"), _query(same_title_size_cap="0"),
        _query(same_title_size_cap=""), _query(rebalance="on", same_title_size_cap="0.2"),
        # A field a link built before the category filter became a list of
        # categories to exclude carries, which exclude_category replaced
        _query(category="Sports"), _query(category="Sports", exclude_category="Politics"),
        _query(exclude_category="any"), _query(exclude_category="Any"),
        _query(exclude_category="none"), _query(exclude_category="None"),
        _query(exclude_category=" Sports"), _query(exclude_category="Sports​"),
        _query(exclude_category="Sports", exclude_tag="Bask‮etball"),
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
        _query() + "&exclude_category=Fútbol",        # not percent-encoded
        _query(nonce="0" * 32), _query(action="trade"),  # not proposal fields
        # More fields than a request may hold
        "k", "&&", _query() + "&" + "&".join(
            f"x{i}=1" for i in range(defaults_server._MAX_FIELDS)),
    ])
    def test_it_is_refused(self, query):
        response = _get(_app(start_process=_Starter()), f"/confirm?{query}")
        assert response.status == 400, query
        assert "<form" not in response.body and "<button" not in response.body
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("field", ["tier_floors", "spread_min", "spread_max", "k",
                                       "size_cap"])
    def test_a_required_field_may_not_be_missing(self, field):
        response = _get(_app(), "/confirm?" + _query(**{field: None}))
        assert response.status == 400
        assert f"{field} is missing" in response.body

    def test_a_link_built_before_exclusions_says_how_to_get_a_new_one(self):
        response = _get(_app(), "/confirm?" + _query(category="Sports"))
        assert response.status == 400
        assert ("this link names category, which exclude_category replaced (the category "
                "filter is now a list of categories NOT to trade): rebuild the backtest "
                "dashboard, or open this server&#x27;s start page") in response.body

    def test_a_link_built_before_tag_exclusions_says_how_to_get_a_new_one(self):
        for query in (_query(tag="Basketball"),
                      _query(exclude_category="Sports") + "&tag=Sports+%C2%B7+Soccer"):
            response = _get(_app(), "/confirm?" + query)
            assert response.status == 400
            assert ("this link names tag, which exclude_tag replaced (the tag filter is "
                    "now a list of tags NOT to trade): rebuild the backtest dashboard, or "
                    "open this server&#x27;s start page") in response.body
            assert "<form" not in response.body

    def test_a_link_naming_both_retired_filter_fields_names_both(self):
        response = _get(_app(), "/confirm?" + _query(category="Sports") + "&tag=Basketball")
        assert response.status == 400
        assert ("this link names category and tag, which exclude_category and exclude_tag "
                "replaced (the category and tag filters are now lists of categories and "
                "tags NOT to trade): rebuild the backtest dashboard, or open this "
                "server&#x27;s start page") in response.body

    def test_a_tied_tag_in_an_excluded_category_is_proposed(self):
        # Excluding a tag of a category that is excluded anyway changes
        # nothing, and is no reason to refuse the link
        response = _get(_app(), "/confirm?" + _filter_query(
            ("Sports",), ("Sports · Basketball",)))
        assert response.status == 200
        page = _parse(response.body)
        assert page.cells["exclude categories"][2] == "Sports"
        assert page.cells["exclude tags"][2] == "Sports · Basketball"

    def test_a_tag_needs_no_category(self):
        # Nothing excluded and one plain tag: every category, only that tag
        response = _get(_app(), "/confirm?" + _query(exclude_tag="Basketball"))
        assert response.status == 200
        page = _parse(response.body)
        assert page.cells["exclude categories"][2] == "none"
        assert page.cells["exclude tags"][2] == "Basketball"

    def test_the_seed_note_labels_the_seed_values_only(self):
        # The seed's values under its note are shown ...
        query = _query(source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert _get(_app(), "/confirm?" + query).status == 200
        # ... and still are over saved defaults whose minimum price movement,
        # rebalance-until days and sell settings all differ from the seed's:
        # under the seed's note, every field the link leaves out takes the
        # seed's value, never the saved one
        saved = _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5),
                                   interval_discount=0.8, size_cap=0.1, rebalance=True,
                                   min_price_move=0.25, rebalance_min_days_to_maturity=7,
                                   sell_at=0.85, sell_min_days=3))
        assert defaults_server._proposal(defaults_server._params(query), saved) == (
            config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)
        assert _get(_app(), "/confirm?" + query).status == 200
        # ... but a value the link names that is not the seed's may not carry its note
        response = _get(_app(), "/confirm?" + _query(
            k="0.75", source=config.LIVE_DEFAULTS_SEED_SOURCE))
        assert response.status == 400
        assert "may label only the seed values" in response.body

    def test_the_seed_note_labels_no_selling_settings(self):
        # The seed sells nothing: its note on a link that sells is refused
        for given in ({"sell_at": "0.85"}, {"sell_at": "0.85", "sell_min_days": "3"}):
            response = _get(_app(), "/confirm?" + _query(
                source=config.LIVE_DEFAULTS_SEED_SOURCE, **given))
            assert response.status == 400, given
            assert "may label only the seed values" in response.body
        # ... and a seed link that names selling off is the seed
        assert _get(_app(), "/confirm?" + _query(
            source=config.LIVE_DEFAULTS_SEED_SOURCE, sell_at="off",
            sell_min_days="off")).status == 200

    def test_the_seed_note_labels_only_the_seed_s_rebalance_settings(self):
        # The seed rebalances with no minimum: its note on a link that turns
        # rebalancing off, or sets a minimum, is refused
        for given in ({"rebalance": "off"},
                      {"rebalance": "on", "min_price_move": "0.1"}):
            response = _get(_app(), "/confirm?" + _query(
                source=config.LIVE_DEFAULTS_SEED_SOURCE, **given))
            assert response.status == 400, given
            assert "may label only the seed values" in response.body
        # ... and a seed link that names the seed's own settings is the seed
        assert _get(_app(), "/confirm?" + _query(
            source=config.LIVE_DEFAULTS_SEED_SOURCE, rebalance="on",
            min_price_move="off")).status == 200

    def test_the_seed_note_labels_only_the_seed_s_rebalance_until(self):
        # The seed rebalances to maturity: its note on a link that stops short
        # of it is refused, and a seed link naming "maturity" (0) is the seed
        for days in ("1", "7"):
            response = _get(_app(), "/confirm?" + _query(
                source=config.LIVE_DEFAULTS_SEED_SOURCE, rebalance="on",
                rebalance_min_days_to_maturity=days))
            assert response.status == 400, days
            assert "may label only the seed values" in response.body
        assert _get(_app(), "/confirm?" + _query(
            source=config.LIVE_DEFAULTS_SEED_SOURCE, rebalance="on",
            rebalance_min_days_to_maturity="0")).status == 200

    @pytest.mark.parametrize("field, value", [
        ("add_to_held_pairs", "on"), ("trim_to_kelly", "off"), ("trim_min_rise", "0.1")])
    def test_a_link_built_before_rebalance_says_how_to_get_a_new_one(self, field, value):
        # The dashboard built before Rebalance sends these; the refusal says what to do
        response = _get(_app(), "/confirm?" + _query(**{field: value}))
        assert response.status == 400
        assert (f"this link was built before Rebalance (it names {field}): rebuild the "
                "backtest dashboard, or open this server&#x27;s start page") in response.body
        assert "<form" not in response.body

    # The refusal of a link that names the same-title cap and no older retired field
    _SAME_TITLE_CAP_REFUSAL = (
        "this link names same_title_size_cap, which is no longer used (the bot trades "
        "time-series pairs only): rebuild the backtest dashboard, or open this "
        "server&#x27;s start page")

    @pytest.mark.parametrize("value", ["0.2", "0.05", "1", "0", "x", ""])
    def test_a_link_naming_the_same_title_cap_says_how_to_get_a_new_one(self, value):
        # A dashboard page built before the cap was removed sends it (when its
        # run recorded one), as an old bookmark may: refused whatever its
        # value, before any other field is read, saving nothing, over saved
        # defaults or none
        for saved in (False, True):
            if saved:
                _save(_BASE_SETTINGS)
            before = config.LIVE_DEFAULTS_FILE.read_bytes() if saved else None
            response = _get(_app(start_process=_Starter()),
                            "/confirm?" + _query(same_title_size_cap=value, k="nan"))
            assert response.status == 400, (value, saved)
            assert self._SAME_TITLE_CAP_REFUSAL in response.body, (value, saved)
            assert "<form" not in response.body and "<button" not in response.body
            if saved:
                assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
            else:
                assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("older, message", [
        # A page built after Rebalance but before the category filter became a
        # list to exclude sent category beside the cap: it is told about the
        # category filter, exactly as a link naming category alone is
        ({"category": "Sports"},
         "this link names category, which exclude_category replaced (the category filter "
         "is now a list of categories NOT to trade)"),
        ({"tag": "Basketball"},
         "this link names tag, which exclude_tag replaced (the tag filter is now a list "
         "of tags NOT to trade)"),
        ({"category": "Sports", "tag": "Basketball"},
         "this link names category and tag, which exclude_category and exclude_tag "
         "replaced (the category and tag filters are now lists of categories and tags NOT "
         "to trade)"),
        # A page built before Rebalance is told about Rebalance
        ({"add_to_held_pairs": "on"}, "this link was built before Rebalance (it names "
                                      "add_to_held_pairs)"),
        ({"trim_to_kelly": "on", "trim_min_rise": "0.1", "category": "Sports"},
         "this link was built before Rebalance (it names trim_to_kelly)"),
    ], ids=["category", "tag", "category and tag", "rebalance", "rebalance and category"])
    def test_a_link_naming_the_cap_beside_an_older_field_is_told_about_the_older_one(
            self, older, message):
        # The cap is the newest retirement, so the older field's message is the
        # one that says what the link predates; the cap is not named in it
        for query in (_query(same_title_size_cap="0.2", **older),
                      _query(**older) + "&same_title_size_cap=0.2"):
            response = _get(_app(), "/confirm?" + query)
            assert response.status == 400, query
            body = html.unescape(response.body)
            assert (message + ": rebuild the backtest dashboard, or open this server's "
                    "start page") in body, query
            assert "same_title_size_cap" not in body, query
            assert "<form" not in response.body
        # ... and without the cap, the very same message
        alone = _get(_app(), "/confirm?" + _query(**older))
        assert alone.status == 400
        assert message in html.unescape(alone.body)

    def test_a_refusal_of_the_cap_is_logged_as_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _get(_app(), "/confirm?" + _query(same_title_size_cap="0.2"))
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.getMessage() == (
            "Refused (400 Bad Request): " + html.unescape(self._SAME_TITLE_CAP_REFUSAL))

    def test_an_invalid_proposal_is_refused_on_the_post_too(self):
        # A form carrying a proposal no page would render, signed as a page would sign it
        app = _app(start_process=_Starter())
        params = {**{name: [value] for name, value in _BASE.items()}, "k": ["1.5"]}
        fingerprint, nonce = defaults_server._fingerprint(None), "1" * 32
        form = [(name, values[0]) for name, values in params.items()]
        form += [("fingerprint", fingerprint), ("nonce", nonce),
                 ("token", app._token("confirm", fingerprint, nonce, params)),
                 ("action", "trade")]
        response = _post(app, form)
        assert response.status == 400
        assert "These settings cannot be saved" in response.body
        assert app._start_process.calls == []


def _filter_query(categories=(), tags=(), **changes) -> str:
    """
    Build a confirmation query that names several excluded categories and tags.

    Args:
        categories: The excluded category names, one exclude_category field
            each, in order.
        tags: The tag names, one field each, in order.
        **changes: Other fields to set or (with None) remove, as _query takes them.

    Returns:
        str: The query string.
    """
    names = [("exclude_category", name) for name in categories] + [("exclude_tag", name) for name in tags]
    return _query(**changes) + ("&" + urlencode(names) if names else "")


# The most names a link may carry: 400 excluded categories, and 400 excluded
# tags each tied to a category
_MOST = config.DEFAULTS_SERVER_MAX_FILTER_NAMES
_MOST_CATEGORIES = tuple(f"Climate and Weather {i:02d}" for i in range(_MOST))
_MOST_TAGS = tuple(f"Climate and Storms {i:02d} · Hurricanes and Storms"
                   for i in range(_MOST))


class TestSeveralCategoriesAndTags:
    """A link may name several excluded categories and tags, one field each:
    the page proposes exactly them, a save writes them and a run gets one flag
    each."""

    _CATEGORIES = ("Economics", "Politics", "Climate and Weather")
    _TAGS = ("Sports · Basketball", "Sports · Pro Football", "science and technology · AI")
    _SETTINGS = replace(_BASE_SETTINGS, exclude_categories=_CATEGORIES, exclude_tags=_TAGS)

    def test_the_page_proposes_every_name_in_order(self):
        query = _filter_query(self._CATEGORIES, self._TAGS)
        assert defaults_server._proposal(defaults_server._params(query), None) == (
            self._SETTINGS, "")
        response = _get(_app(), f"/confirm?{query}")
        assert response.status == 200
        page = _parse(response.body)
        assert page.cells["exclude categories"][:3] == [
            "exclude categories", "—", "Economics, Politics, Climate and Weather"]
        assert page.cells["exclude tags"][:3] == [
            "exclude tags", "—", "Sports · Basketball, Sports · Pro Football, "
            "science and technology · AI"]
        # The form carries one hidden field per name, in the link's order
        assert [v for n, v in page.hidden if n == "exclude_category"] == list(self._CATEGORIES)
        assert [v for n, v in page.hidden if n == "exclude_tag"] == list(self._TAGS)

    def test_the_filter_it_proposes_is_the_rule_s(self):
        settings, _ = defaults_server._proposal(defaults_server._params(
            _filter_query(("Economics", "Politics"), ("Sports · Basketball",))), None)
        keeps = config.trade_filter(settings)
        assert keeps("Sports", "Soccer") and keeps("Climate and Weather", "Hurricanes")
        assert not keeps("Sports", "Basketball") and not keeps("Economics", "Fed")
        assert not keeps("Politics", "Soccer")
        # A tied tag excludes its tag in its own category only
        assert keeps("Climate and Weather", "Basketball")

    def test_a_link_with_one_exclusion_and_one_plain_tag(self):
        for query in (_query(exclude_category="Sports", exclude_tag="Basketball"),
                      _filter_query(("Sports",), ("Basketball",))):
            settings, _ = defaults_server._proposal(defaults_server._params(query), None)
            assert settings == replace(_BASE_SETTINGS, exclude_categories=("Sports",),
                                       exclude_tags=("Basketball",))
            assert _get(_app(), f"/confirm?{query}").status == 200
        # Exclusions alone, and a plain tag beside a tied one
        settings, _ = defaults_server._proposal(defaults_server._params(
            _filter_query(("Sports", "Economics"))), None)
        assert (settings.exclude_categories, settings.exclude_tags) == (("Sports", "Economics"), None)
        settings, _ = defaults_server._proposal(defaults_server._params(
            _filter_query(("Politics", "Economics"), ("Fed", "Sports · Soccer"))), None)
        assert settings.exclude_tags == ("Fed", "Sports · Soccer")

    def test_a_missing_filter_still_means_none_and_any_whatever_is_saved(self):
        _save(self._SETTINGS)
        settings, _ = defaults_server._proposal(defaults_server._params(_query()),
                                                config.read_saved_live_defaults())
        assert settings.exclude_categories is None and settings.exclude_tags is None

    @pytest.mark.parametrize("query, why", [
        # The very same name twice (another letter case is left out instead:
        # test_a_second_spelling_in_another_letter_case_is_left_out)
        (_filter_query(("Sports", "Economics", "Sports")),
         "exclude_category &#x27;Sports&#x27; is given twice"),
        (_filter_query(("Sports",), ("Basketball", "Soccer", "Basketball")),
         "exclude_tag &#x27;Basketball&#x27; is given twice"),
        (_filter_query(("Sports",), ("Sports · Soccer", "Sports · Soccer")), "is given twice"),
        # ... even when it repeats a spelling that was itself left out
        (_filter_query(("Sports", "sports", "sports")),
         "exclude_category &#x27;sports&#x27; is given twice"),
        # A blank value among several
        (_filter_query(("Sports",)) + "&exclude_category=", "exclude_category is blank"),
        (_filter_query(("Sports",), ("Soccer",)) + "&exclude_tag=&exclude_tag=Basketball", "exclude_tag is blank"),
        # More names than a link may carry
        (_filter_query([f"C{i}" for i in range(_MOST + 1)]),
         f"exclude_category is given {_MOST + 1} times, more than {_MOST}"),
        (_filter_query(("Sports",), [f"T{i}" for i in range(_MOST + 1)]),
         f"exclude_tag is given {_MOST + 1} times, more than {_MOST}"),
        # A category written as a tied tag, with tags or without: it would read
        # as "exclude Basketball from Sports" while it excluded nothing
        (_filter_query(("Sports", "Sports · Basketball")),
         "exclude_categories: &#x27;Sports · Basketball&#x27; is a tag tied to a "
         "category; only a whole category can be excluded"),
        (_filter_query(("Politics", "Sports · Basketball"), ("Sports · Basketball",)),
         "is a tag tied to a category; only a whole category can be excluded"),
        (_filter_query(("Sports ·",)), "is a tag tied to a category; only a whole category can be excluded"),
        # Two spaces in a row, which a browser would show as one: the name
        # would look like another category's, or like a tag tied to it
        (_filter_query(("Climate and Weather", "Climate  and Weather"),
                       ("Climate  and Weather · Hurricanes",)),
         "exclude_category must not hold two spaces in a row, "
         "got &#x27;Climate  and Weather&#x27;"),
        (_filter_query(("Politics",), ("Climate  and Weather · Hurricanes",)),
         "exclude_tag must not hold two spaces in a row"),
        (_filter_query(("Politics",), ("Sports ·  Basketball",)),
         "exclude_tag must not hold two spaces in a row"),
        (_filter_query(("Sports",), ("Pro  Football",)),
         "exclude_tag must not hold two spaces in a row"),
        # A tied tag with a blank half is never read as a plain tag
        (_filter_query(("Politics",), ("Sports · ",)), "exclude_tag must be one printable name"),
        (_filter_query(("Politics",), (" · Basketball",)),
         "exclude_tag must be one printable name"),
        (_filter_query(("Politics",), ("Sports ·",)), "must read &#x27;Category · Tag&#x27;"),
        (_filter_query(("Politics",), ("· Basketball",)),
         "must read &#x27;Category · Tag&#x27;"),
        # One bad name refuses the whole link
        (_filter_query(("Sports", "any")), "exclude_categories cannot hold the name"),
        (_filter_query(("Sports", "None")), "exclude_categories cannot hold the name"),
        (_filter_query(("Sports", " Economics")), "exclude_category must be one printable name"),
        (_filter_query(("Sports",), ("Soccer", "Bask‮etball")),
         "exclude_tag must be one printable name"),
        # A single-valued field still may not repeat
        (_filter_query(("Sports",)) + "&k=0.8", "k is given 2 times"),
    ])
    def test_a_bad_list_is_refused_with_the_reason(self, query, why):
        response = _get(_app(start_process=_Starter()), f"/confirm?{query}")
        assert response.status == 400, query
        assert why in response.body, response.body
        assert "<form" not in response.body and "<button" not in response.body
        assert config.read_saved_live_defaults() is None

    def test_a_second_spelling_in_another_letter_case_is_left_out(self):
        # Kalshi spells a few tags two ways under one category, and a page that
        # lists both may send both: the live filter ignores letter case, so the
        # first spelling is kept and the link is not refused
        query = _filter_query(("Health", "Sports", "health"),
                              ("Science · COVID", "Science · Covid", "Basketball", "BASKETBALL"))
        settings, _ = defaults_server._proposal(defaults_server._params(query), None)
        assert settings.exclude_categories == ("Health", "Sports")
        assert settings.exclude_tags == ("Science · COVID", "Basketball")
        app = _app()
        response = _get(app, f"/confirm?{query}")
        assert response.status == 200
        page = _parse(response.body)
        assert page.cells["exclude categories"][:3] == [
            "exclude categories", "—", "Health, Sports"]
        assert page.cells["exclude tags"][:3] == [
            "exclude tags", "—", "Science · COVID, Basketball"]
        # The filter excludes either spelling's category and either spelling's
        # tag
        keeps = config.trade_filter(settings)
        assert not keeps("Science", "Covid") and not keeps("Science", "COVID")
        assert not keeps("Economics", "Basketball") and keeps("Science", "Flu")
        assert not keeps("health", "Flu") and not keeps("SPORTS", "Soccer")
        # Saving the page's own form writes each name once
        assert _post(app, _click(page, "confirm")).status == 303
        saved = config.read_saved_live_defaults()
        assert (saved.exclude_categories, saved.exclude_tags) == (settings.exclude_categories,
                                                          settings.exclude_tags)

    def test_the_cap_counts_the_names_sent(self):
        # 401 spellings of one name are over the cap, though one would be kept
        spellings = [f"{'s' * i}{'S' * (_MOST + 1 - i)}" for i in range(_MOST + 1)]
        assert len({s.casefold() for s in spellings}) == 1 and len(set(spellings)) == _MOST + 1
        response = _get(_app(), "/confirm?" + _filter_query(spellings))
        assert response.status == 400
        assert (f"exclude_category is given {_MOST + 1} times, more than {_MOST}"
                in response.body)
        # ... while 400 of them are one category
        settings, _ = defaults_server._proposal(defaults_server._params(
            _filter_query(spellings[:_MOST])), None)
        assert settings.exclude_categories == (spellings[0],)

    @pytest.mark.parametrize("button_id", ["confirm", "confirm-trade", "confirm-dry-run"])
    def test_a_form_with_several_names_posted_twice_does_its_work_once(self, monkeypatch,
                                                                      button_id):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        saves = []
        real_save = defaults_server.save_live_defaults

        def counting_save(settings, *, source):
            """
            Count each save, then make it.

            Args:
                settings (LiveSettings): The defaults saved.
                source (str): Their source note.

            Returns:
                LiveSettings: What the real save returns.
            """
            saves.append(settings)
            return real_save(settings, source=source)

        monkeypatch.setattr(defaults_server, "save_live_defaults", counting_save)
        form = _page_form(app, _filter_query(self._CATEGORIES, self._TAGS), button_id)
        first = _post(app, form)
        assert first.status == 303
        # The second POST of the same page is sent to what the first produced
        second = _post(app, form)
        assert second.status == 303 and second.location == first.location
        assert defaults_server._STALE_BANNER not in second.body
        dry_run = button_id == "confirm-dry-run"
        assert saves == ([] if dry_run else [self._SETTINGS])
        assert len(starter.calls) == (0 if button_id == "confirm" else 1)
        assert config.read_saved_live_defaults() == (
            _BASE_SETTINGS if dry_run else self._SETTINGS)
        # ... and the one run, if any, got each name once
        for argv, _ in starter.calls:
            assert argv[5:argv.index("--result-file")] == config.live_settings_argv(
                self._SETTINGS)

    def test_the_most_names_a_link_may_carry_fit_one_request(self):
        # 400 excluded categories and 400 excluded tags, each tied to a category
        query = _filter_query(_MOST_CATEGORIES, _MOST_TAGS, source=DASHBOARD_NOTE)
        app = _app()
        target = f"/confirm?{query}"
        response = _get(app, target)
        assert response.status == 200
        form = _click(response.body, "confirm", ack=False)
        body = urlencode(form).encode("ascii")
        # Inside the handler's limit on a request (its path plus its body), and
        # inside the base class's 65,536-byte request line
        assert len(f"GET {target} HTTP/1.1\r\n") <= min(
            65_536, config.DEFAULTS_SERVER_MAX_REQUEST_BYTES)
        assert len("/confirm") + len(body) < config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        assert len(form) <= defaults_server._MAX_FIELDS - 2
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.exclude_categories == _MOST_CATEGORIES and saved.exclude_tags == _MOST_TAGS

    def test_three_hundred_excluded_tags_round_trip(self):
        # A dashboard page that excludes most tags of most categories sends
        # hundreds of "Category · Tag" names, letter-case twins included: the
        # page proposes them all, its form saves them all, in the link's order
        tags = tuple(f"{category} · {tag}"
                     for category in ("Sports", "Politics", "Economics", "Entertainment",
                                      "Climate and Weather", "Science and Technology")
                     for tag in [f"Tag {i:02d}" for i in range(48)] + ["COVID", "Covid"])
        assert len(tags) == 300
        query = _filter_query(("Crypto", "Financials"), tags, source=DASHBOARD_NOTE)
        target = f"/confirm?{query}"
        assert len(f"GET {target} HTTP/1.1\r\n") <= 65_536
        assert len(target) <= config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        app = _app()
        response = _get(app, target)
        assert response.status == 200
        page = _parse(response.body)
        # The form repeats every name the link sent, in its order (the token
        # covers the raw fields); a second spelling in another letter case is
        # left out of what is saved, as for any link
        assert [v for n, v in page.hidden if n == "exclude_tag"] == list(tags)
        kept = tuple(t for t in tags if not t.endswith(" · Covid"))
        form = _click(page, "confirm")
        assert len("/confirm") + len(urlencode(form)) <= config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.exclude_categories == ("Crypto", "Financials")
        assert saved.exclude_tags == kept
        # ... and the saved filter excludes exactly those tags in their categories
        keeps = config.trade_filter(saved)
        assert not keeps("Sports", "Tag 07") and not keeps("Science and Technology", "covid")
        assert keeps("Sports", "Tag 48") and keeps("Health", "Tag 07")
        assert not keeps("Crypto", "Tag 48")

    def test_the_field_limit_covers_the_fullest_form(self):
        # Every proposal field once, the most categories and tags, the three
        # signed fields, the action and the acknowledgement
        fullest = (len(defaults_server._FIELDS) - 2) + 2 * _MOST + 5
        assert fullest <= defaults_server._MAX_FIELDS
        fields = "&".join(f"x{i}=1" for i in range(defaults_server._MAX_FIELDS))
        assert len(defaults_server._params(fields)) == defaults_server._MAX_FIELDS
        with pytest.raises(ValueError):
            defaults_server._params(fields + "&y=1")

    def test_confirm_and_save_writes_every_name(self):
        app = _app()
        form = _page_form(app, _filter_query(self._CATEGORIES, self._TAGS))
        response = _post(app, form)
        assert response.status == 303
        saved = config.read_saved_live_defaults()
        assert saved == self._SETTINGS
        assert (saved.exclude_categories, saved.exclude_tags) == (self._CATEGORIES, self._TAGS)
        stored = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))["settings"]
        assert stored["exclude_categories"] == list(self._CATEGORIES)
        assert stored["exclude_tags"] == list(self._TAGS)
        # The saved page names them
        body = _get(app, "/saved").body
        assert "Economics, Politics, Climate and Weather" in body
        assert "Sports · Basketball, Sports · Pro Football" in body

    @pytest.mark.parametrize("button, dry_run", [("confirm-dry-run", True),
                                                 ("confirm-trade", False)])
    def test_a_run_gets_one_flag_per_name(self, button, dry_run):
        # A dry run needs saved defaults; these exclude no category and name no tag
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _filter_query(self._CATEGORIES, self._TAGS), button)
        _run_id(_post(app, form))
        [(argv, _)] = starter.calls
        assert ("--dry-run" in argv) is dry_run
        assert [a for a in argv if a.startswith(
                ("--exclude-category", "--no-exclude-category", "--exclude-tag",
                 "--no-exclude-tag"))] == [
            "--exclude-category=Economics", "--exclude-category=Politics",
            "--exclude-category=Climate and Weather",
            "--exclude-tag=Sports · Basketball", "--exclude-tag=Sports · Pro Football",
            "--exclude-tag=science and technology · AI"]
        flags = argv[5:argv.index("--result-file")]
        assert flags == config.live_settings_argv(self._SETTINGS)
        # A dry run saves nothing; Confirm and trade saved before it started
        assert config.read_saved_live_defaults() == (
            _BASE_SETTINGS if dry_run else self._SETTINGS)

    def test_the_page_with_no_address_fields_runs_saved_names(self):
        saved = _save(self._SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))
        [(argv, _)] = starter.calls
        assert argv[5:argv.index("--result-file")] == config.live_settings_argv(saved)
        assert "--exclude-tag=Sports · Basketball" in argv and "--exclude-category=Economics" in argv

    def test_the_token_covers_every_name_and_its_place(self):
        app = _app(start_process=_Starter())
        form = _page_form(app, _filter_query(("Economics", "Politics"), ("Sports · Soccer",)))

        def changed(edit):
            """A copy of the form with its (name, value) pairs passed through edit."""
            return edit(list(form))

        def without(pairs, pair):
            pairs.remove(pair)
            return pairs

        def swapped(pairs):
            a = pairs.index(("exclude_category", "Economics"))
            b = pairs.index(("exclude_category", "Politics"))
            pairs[a], pairs[b] = pairs[b], pairs[a]
            return pairs

        for tampered in (
            changed(lambda p: without(p, ("exclude_category", "Economics"))),
            changed(lambda p: without(p, ("exclude_tag", "Sports · Soccer"))),
            changed(lambda p: p + [("exclude_category", "Health")]),
            changed(lambda p: p + [("exclude_tag", "Sports · Basketball")]),
            changed(swapped),
        ):
            assert _post(app, tampered).status == 403, tampered
        assert config.read_saved_live_defaults() is None
        # The page's own form is still good afterwards
        assert _post(app, form).status == 303
        assert config.read_saved_live_defaults().exclude_categories == ("Economics",
                                                                        "Politics")

    def test_a_stale_page_is_shown_again_with_every_name(self):
        app = _app()
        form = _page_form(app, _filter_query(self._CATEGORIES, self._TAGS))
        _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.8,
                           size_cap=0.1))
        response = _post(app, form)
        assert response.status == 409
        page = _parse(response.body)
        assert [v for n, v in page.hidden if n == "exclude_category"] == list(self._CATEGORIES)
        assert [v for n, v in page.hidden if n == "exclude_tag"] == list(self._TAGS)
        # The fresh page's form saves them
        assert _post(app, _click(page, "confirm")).status == 303
        assert config.read_saved_live_defaults().exclude_tags == self._TAGS


class TestRefusedFile:
    """A saved file that cannot be used blocks every read route with 409, no form."""

    def test_every_route_is_refused(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        form = _page_form(app, _query(k="0.75"))
        trade = _home_form(app)
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        for response in (_get(app, "/confirm?" + _query()), _post(app, form),
                         _get(app, "/confirm"), _get(app, "/saved"), _post(app, trade)):
            assert response.status == 409
            assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body
            assert "<form" not in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == b"not json"
        assert app._start_process.calls == []


# A file a version from before Rebalance wrote, in its exact layout: adding to
# held pairs on, the Kelly trim on with a 10c minimum price rise, and the old
# category and tag keys and the same-title cap, which every file saved then
# holds
_OLD_VERSION_BYTES = (
    b'{\n'
    b'  "format": "live-defaults-v1",\n'
    b'  "saved_at": "2026-10-11T18:22:05Z",\n'
    b'  "source": "backtest dashboard for 2025-10-11 to 2026-10-11",\n'
    b'  "settings": {\n'
    b'    "tier_floors": false,\n'
    b'    "spread_band": [0.0, 0.8],\n'
    b'    "interval_discount": 0.7,\n'
    b'    "size_cap": 0.15,\n'
    b'    "same_title_size_cap": 0.2,\n'
    b'    "categories": null,\n'
    b'    "tags": null,\n'
    b'    "add_to_held_pairs": true,\n'
    b'    "trim_to_kelly": true,\n'
    b'    "trim_min_rise": 0.1\n'
    b'  }\n'
    b'}\n')
# What it reads as now: rebalance yes with the old 10c minimum, no category or
# tag excluded, no same-title cap, the rest as saved
_OLD_VERSION_READS_AS = LiveSettings(tier_floors=False, spread_band=(0.0, 0.8),
                                     interval_discount=0.7, size_cap=0.15,
                                     rebalance=True, min_price_move=0.10)
# The notice the confirmation page shows above its table when it proposes them
_OUTDATED_CONFIRM_NOTICE = (
    "The saved live defaults were saved before Rebalance. Below: their values, read as "
    "Rebalance yes and Min. price movement 10c. Confirm and save to replace the file.")


def _old_version_file() -> defaults_server._Response:
    """
    Write the file saved before Rebalance as the saved defaults, and open the index that links it.

    Returns:
        defaults_server._Response: The index page, which carries the link.
    """
    config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
    return _get(_app(), "/")


def _outdated_link() -> str:
    """
    The link every page offers for the file saved before Rebalance: its values in their new form.

    Returns:
        str: /confirm? and the query naming every field, with no source note.
    """
    return "/confirm?" + defaults_server._settings_query(_OLD_VERSION_READS_AS)


class TestOutdatedFile:
    """A file saved before Rebalance (add_to_held_pairs, trim_to_kelly and
    trim_min_rise, beside the later changes' old keys: categories, tags and
    the same-title cap) is never traded: the confirmation page counts nothing
    as in force and offers its values in their new form (rebalance yes, the
    old minimum, no same-title cap) for one Confirm and save, which replaces
    the file; every other page shows a notice linking there, naming only
    Rebalance, the earliest change; and every run stays refused, or keeps
    every blocker, until the save."""

    def test_the_index_shows_the_notice_and_the_link(self):
        response = _old_version_file()
        assert response.status == 200
        assert ("The saved live defaults were saved before Rebalance, so every live run "
                "refuses them until they are saved again.") in response.body
        assert _outdated_link() in _parse(response.body).links
        assert "read as Rebalance yes and Min. price movement 10c" in response.body
        # Only the earliest change is named: the file's null category and tag
        # keys, and its same-title cap, leave what it trades as it was
        assert "same-title" not in response.body
        assert "category filter" not in response.body and "tag filter" not in response.body

    def test_only_the_earliest_change_is_named_while_the_file_can_be_read(self):
        # Every later change's old key sits in the file, but the notice names
        # Rebalance alone; with no reading in the new form it names all four
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.retired == ("add_to_held_pairs", "trim_to_kelly", "trim_min_rise",
                                        "categories", "tags", "same_title_size_cap")
        assert defaults_server._outdated_changes(caught.value) == (True, False, False, False)
        assert defaults_server._outdated_when(caught.value) == "before Rebalance"
        assert defaults_server._outdated_reading(caught.value) == (
            "Rebalance yes and Min. price movement 10c")
        unreadable = config.LiveDefaultsOutdated("m", None, caught.value.retired)
        assert defaults_server._outdated_changes(unreadable) == (True, True, True, True)
        assert defaults_server._outdated_when(unreadable) == (
            "before Rebalance and before the category filter became a list of categories to "
            "exclude and before the tag filter became a list of tags to exclude and before the "
            "same-title cap was removed")

    def test_the_link_names_every_field_and_no_source_note(self):
        params = defaults_server._params(_outdated_link().split("?", 1)[1])
        assert "source" not in params and "same_title_size_cap" not in params
        assert [name for name in defaults_server._FIELDS
                if name not in ("exclude_category", "exclude_tag", "source")] == list(params)
        assert params["rebalance"] == ["on"] and params["min_price_move"] == ["0.1"]
        assert params["rebalance_min_days_to_maturity"] == ["0"]
        # It proposes exactly those values, whatever is saved
        assert defaults_server._proposal(params, None) == (_OLD_VERSION_READS_AS, "")
        assert defaults_server._proposal(params, config.LIVE_DEFAULTS_SEED)[0] == (
            _OLD_VERSION_READS_AS)

    def test_the_confirmation_page_shows_the_notice_and_the_new_values(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        response = _get(_app(), _outdated_link())
        assert response.status == 200
        assert f'<p class="banner">{_OUTDATED_CONFIRM_NOTICE}</p>' in response.body
        # The notice sits above the table
        assert response.body.index(_OUTDATED_CONFIRM_NOTICE) < response.body.index("<table>")
        assert "Save the live trading defaults again?" in response.body
        assert ("Live defaults in force: none — the saved file was saved before Rebalance, "
                "and live runs refuse it until it is saved again") in response.body
        page = _parse(response.body)
        assert list(page.rows) == LABELS
        assert all(cells[1] == "—" for cells in page.cells.values())
        assert page.cells["spread band"][2] == "0-0.8"
        assert page.cells["k"][2] == "0.7"
        assert page.cells["per-trade cap"][2] == "15%"
        assert page.cells["rebalance"][2] == "yes"
        assert page.cells["min price movement"][2] == "10c"
        assert "11 of 11 settings change." in response.body

    def test_a_link_that_leaves_the_limit_out_keeps_the_old_file_s(self):
        # The old file's values in their new form carry no limit, so "maturity"
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        base = defaults_server._kept_base(None, caught.value)
        assert base == _OLD_VERSION_READS_AS
        query = _query(spread_max="0.8", k="0.7", size_cap="0.15")
        proposed, _ = defaults_server._proposal(defaults_server._params(query), base)
        assert proposed.rebalance_min_days_to_maturity is None
        # ... and what the base holds is what a left-out field keeps
        stand_in = SimpleNamespace(proposal=replace(
            _OLD_VERSION_READS_AS, rebalance_min_days_to_maturity=7))
        base = defaults_server._kept_base(None, stand_in)
        proposed, _ = defaults_server._proposal(defaults_server._params(query), base)
        assert proposed.rebalance_min_days_to_maturity == 7
        # With nothing saved and no outdated file, the seed's
        proposed, _ = defaults_server._proposal(
            defaults_server._params(query), defaults_server._kept_base(None, None))
        assert proposed.rebalance_min_days_to_maturity is None

    def test_one_confirm_and_save_replaces_the_file(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        app = _app()
        response = _get(app, _outdated_link())
        posted = _post(app, _click(response.body, "confirm"))
        assert posted.status == 303 and posted.location == "/saved"
        # It now reads cleanly, as the values the page showed, with no old key
        assert config.read_saved_live_defaults() == _OLD_VERSION_READS_AS
        assert config.live_defaults() == _OLD_VERSION_READS_AS
        text = config.LIVE_DEFAULTS_FILE.read_text()
        for retired in ("add_to_held_pairs", "trim_to_kelly", "trim_min_rise",
                        "same_title_size_cap"):
            assert retired not in text
        assert '"rebalance": true' in text and '"min_price_move": 0.1' in text
        # ... and every page reads it as the defaults in force
        assert _get(app, "/saved").status == 200
        home = _get(app, "/confirm")
        assert home.status == 200 and "0 of 11 settings change." in home.body
        assert "saved before Rebalance" not in _get(app, "/").body

    def test_the_saved_page_refuses_with_the_notice_and_the_link(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        response = _get(_app(start_process=_Starter()), "/saved")
        assert response.status == 409
        page = _parse(response.body)
        assert _outdated_link() in page.links
        assert page.forms == [] and page.buttons == []
        assert ("The saved live defaults were saved before Rebalance, so every live run "
                "refuses them until they are saved again.") in response.body
        # Never the advice to delete it: a save replaces it
        assert "Fix or delete" not in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES

    def test_a_dry_run_is_refused_until_it_is_saved(self):
        # A dry run needs saved defaults: the button is a placeholder, and a
        # POST asking for one anyway is refused, saving nothing
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        starter = _Starter()
        app = _app(start_process=starter)
        body = _get(app, _outdated_link()).body
        assert "confirm-dry-run" not in [b.get("id") for b in _parse(body).buttons]
        assert html.escape(defaults_server._SAVE_FIRST) in body
        form = [(n, "dry_run" if n == "action" else v) for n, v in _click(body, "confirm")]
        response = _post(app, form)
        assert response.status == 409
        assert "A dry run needs saved live defaults" in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES
        assert starter.calls == []

    def test_an_old_trade_page_s_form_never_runs_the_old_file(self):
        # A trade form signed as an older version's trade page signed one: no
        # route at /trade, and refused at /confirm (its token signs another
        # purpose); nothing started, the file as it was
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        starter = _Starter()
        app = _app(start_process=starter)
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        for action in ("trade", "dry_run"):
            fingerprint, nonce = defaults_server._fingerprint(None, caught.value), "2" * 32
            form = [("fingerprint", fingerprint), ("nonce", nonce),
                    ("token", app._token("trade", fingerprint, nonce, {})),
                    ("action", action)]
            assert _post(app, form, target="/trade").status == 404
            assert _post(app, form).status == 403
        assert starter.calls == []
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES

    def test_confirm_and_trade_keeps_the_run_lock_blocker(self, held_lock):
        # Another live trading run holds the machine's lock: refused before any save
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        starter = _Starter()
        app = _app(start_process=starter)
        body = _get(app, _outdated_link()).body
        assert "confirm-trade" not in [b.get("id") for b in _parse(body).buttons]
        form = [(n, "trade" if n == "action" else v) for n, v in _click(body, "confirm")]
        response = _post(app, form)
        assert response.status == 409
        assert "Nothing was saved or traded." in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES
        assert starter.calls == []

    def test_confirm_and_trade_keeps_the_attention_acknowledgement(self):
        # The last real-money run needs attention: without the box ticked,
        # nothing is saved or traded, and the page still says the file is old
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        _attention_result()
        starter = _Starter()
        app = _app(start_process=starter)
        response = _post(app, _page_form(app, _outdated_link().split("?", 1)[1],
                                         "confirm-trade"))
        assert response.status == 409
        assert html.escape(defaults_server._ACK_BANNER) in response.body
        assert _OUTDATED_CONFIRM_NOTICE in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES
        assert starter.calls == []

    def _refused_while_a_run_goes(self, app, starter) -> None:
        """
        Check that the outdated file's page, opened while one of this server's
        runs is still going, offers neither run and refuses both POSTs
        before anything is saved: the trade by the run in progress, and the
        dry run by it too, ahead of its own need for saved defaults.

        Args:
            app (defaults_server._App): The application, with a run going.
            starter (_Starter): Its process starter.
        """
        body = _get(app, _outdated_link()).body
        buttons = [b.get("id") for b in _parse(body).buttons]
        assert "confirm-trade" not in buttons and "confirm-dry-run" not in buttons
        assert "A run started from this page is still going" in body
        form = _click(body, "confirm")
        for action in ("trade", "dry_run"):
            response = _post(app, [(n, action if n == "action" else v) for n, v in form])
            assert response.status == 409
            assert ("A run started from this page is still going "
                    "(/runs/") in html.unescape(response.body)
            assert "Nothing was saved or traded." in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES
        assert starter.calls == []

    def test_confirm_and_trade_keeps_the_own_run_blocker(self):
        # A run this server started is still going: neither run starts, and
        # the old file is not replaced
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        starter = _Starter()
        app = _app(start_process=starter)
        _memory_run(app, returncode=None)
        self._refused_while_a_run_goes(app, starter)

    def test_confirm_and_trade_keeps_the_blocker_of_a_run_from_before_a_restart(self):
        # A fresh server finds a run still going by the lock its process holds
        # on output.log, and refuses both runs the same way
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        folder = _disk_run(output="half way\n")
        starter = _Starter()
        app = _app(start_process=starter)
        with (folder / "output.log").open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            self._refused_while_a_run_goes(app, starter)
            response = _post(app, _click(_get(app, _outdated_link()).body, "confirm"))
        # Confirm and save is no run, so it still replaces the file
        assert response.status == 303
        assert config.read_saved_live_defaults() == _OLD_VERSION_READS_AS

    def test_a_page_built_on_a_usable_file_is_stale_once_an_old_file_replaces_it(self):
        # Pages built on a usable file, posted after a file an older version
        # saved has replaced it (a server still running older code, say): each
        # form, a proposal's or the saved defaults' own, is shown again against
        # what is now saved, and nothing is saved or started
        saved = _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        forms = [_page_form(app, _query(k="0.75"), button)
                 for button in ("confirm", "confirm-trade", "confirm-dry-run")]
        forms += [_home_form(app, button)
                  for button in ("confirm", "confirm-trade", "confirm-dry-run")]
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        for form in forms:
            response = _post(app, form)
            assert response.status == 409
            assert html.escape(defaults_server._STALE_BANNER) in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _OLD_VERSION_BYTES
        assert starter.calls == []
        # The old file's fingerprint is neither a usable file's nor no file's
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        outdated = defaults_server._fingerprint(None, caught.value)
        assert outdated != defaults_server._fingerprint(None)
        assert outdated != defaults_server._fingerprint(saved)

    def test_confirm_and_trade_saves_first_then_runs_the_new_settings(self):
        # The one way a run follows: the save replaces the file, and the run
        # trades the confirmed settings as flags, never the old file
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _page_form(app, _outdated_link().split("?", 1)[1],
                                      "confirm-trade")))
        assert config.read_saved_live_defaults() == _OLD_VERSION_READS_AS
        [(argv, _)] = starter.calls
        assert argv[5:-2] == config.live_settings_argv(_OLD_VERSION_READS_AS)
        assert "--rebalance" in argv and "--min-price-move=10" in argv

    def test_a_page_built_on_the_old_file_is_stale_once_it_is_gone(self):
        # The old file counts as nothing in force, but not as no file: a page
        # built on it, posted after the file was deleted, is shown again
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        app = _app()
        form = _click(_get(app, _outdated_link()).body, "confirm")
        config.LIVE_DEFAULTS_FILE.unlink()
        response = _post(app, form)
        assert response.status == 409
        assert html.escape(defaults_server._STALE_BANNER) in response.body
        assert config.read_saved_live_defaults() is None

    def test_another_proposal_over_the_old_file_links_to_its_own_values(self):
        # The dashboard's Save button proposes other values: the page says the
        # save replaces the file with them, and links to the file's own
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        response = _get(_app(), "/confirm?" + _query())
        assert response.status == 200
        assert ("The saved live defaults were saved before Rebalance, so every live run "
                "refuses them. Confirm and save replaces the file with the settings "
                "below;") in response.body
        assert _outdated_link() in _parse(response.body).links
        assert _OUTDATED_CONFIRM_NOTICE not in response.body

    def test_a_lower_cap_over_the_old_file_still_warns_it_sells_more(self):
        # The held pairs were bought and trimmed under the old file's cap, so
        # a save at a lower cap is compared with it (nothing counts as in
        # force, yet the red sell-more warning still shows); the link that
        # proposes exactly the old values shows none
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        app = _app()
        query = _query(spread_max="0.8", k="0.7", size_cap="0.1", rebalance="on",
                       min_price_move="0.1")
        body = html.unescape(_get(app, "/confirm?" + query).body)
        assert "these settings can sell more of the held pairs" in body
        assert "the per-trade cap goes from 15% to 10%" in body
        assert 'class="warn"' not in _get(app, _outdated_link()).body

    def test_a_link_naming_some_fields_keeps_the_old_file_s_others(self):
        # A link that leaves fields out keeps them as the old file had them
        # (its sell rule, Rebalance and minimum), never the seed's, exactly as
        # it keeps a usable saved file's: here an old file that only added to
        # held pairs, which reads as Rebalance no. Its same-title cap is gone
        # whatever it was: the page has no row for one, and the save writes none
        old = (_OLD_VERSION_BYTES
               .replace(b'"same_title_size_cap": 0.2', b'"same_title_size_cap": 0.5')
               .replace(b'"trim_to_kelly": true,\n    "trim_min_rise": 0.1',
                        b'"trim_to_kelly": false,\n    "sell_at": 0.85,\n'
                        b'    "sell_min_days": 7'))
        config.LIVE_DEFAULTS_FILE.write_bytes(old)
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.proposal.rebalance is False
        app = _app()
        query = _query(spread_max="0.8", k="0.7", size_cap="0.15")
        response = _get(app, "/confirm?" + query)
        page = _parse(response.body)
        assert page.cells["rebalance"][2] == "no"
        assert "same-title cap" not in page.rows
        assert page.cells["sell at"][2] != "off"
        posted = _post(app, _click(response.body, "confirm"))
        assert posted.status == 303
        saved = config.read_saved_live_defaults()
        assert saved == caught.value.proposal
        assert (saved.rebalance, saved.min_price_move, saved.sell_at,
                saved.sell_min_days) == (False, None, 0.85, 7)
        assert "same_title_size_cap" not in config.LIVE_DEFAULTS_FILE.read_text()

    def test_an_old_file_whose_other_settings_are_refused_points_at_new_ones(self):
        # The old minimum is off the 5c grid: there is no proposal, so the
        # pages link to the seed values and say why
        config.LIVE_DEFAULTS_FILE.write_bytes(
            _OLD_VERSION_BYTES.replace(b'"trim_min_rise": 0.1', b'"trim_min_rise": 0.07'))
        app = _app()
        saved = _get(app, "/saved")
        assert saved.status == 409
        assert "/confirm?" + defaults_server._seed_query() in _parse(saved.body).links
        assert "its other settings are refused too" in html.unescape(saved.body)
        # The trading page with no address fields proposes the seed values
        home = _get(app, "/confirm")
        assert home.status == 200
        assert defaults_server._proposal(defaults_server._params(
            json.loads(_parse(home.body).body["data-dashboard-message"])["query"]),
            None) == (config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)
        # The index shows the same notice, with the seed link
        index = _get(app, "/")
        assert index.status == 200
        assert "/confirm?" + defaults_server._seed_query() in _parse(index.body).links
        assert "its other settings are refused too" in html.unescape(index.body)
        confirm = _get(app, "/confirm?" + _query())
        assert confirm.status == 200
        assert ("Confirm and save replaces the file with the settings below."
                in html.unescape(confirm.body))
        assert _post(app, _click(confirm.body, "confirm")).status == 303
        assert config.read_saved_live_defaults() == _BASE_SETTINGS

    def test_a_start_logs_where_to_save_it_again(self, run_main):
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        log = run_main["run"](["--no-browser"])
        assert (f"open http://127.0.0.1:{PORT}{_outdated_link()} to review their values "
                "(read as Rebalance yes and Min. price movement 10c) and save them; do not "
                "delete the file") in log


    @staticmethod
    def _categories_file(categories: bytes) -> None:
        """
        Write a file the version before exclusions saved: rebalancing on, the given categories.

        Args:
            categories (bytes): The JSON text of its categories value.
        """
        config.LIVE_DEFAULTS_FILE.write_bytes(
            _OLD_VERSION_BYTES
            .replace(b'    "add_to_held_pairs": true,\n    "trim_to_kelly": true,\n'
                     b'    "trim_min_rise": 0.1\n',
                     b'    "rebalance": true,\n    "min_price_move": 0.1\n')
            .replace(b'"categories": null', b'"categories": ' + categories))

    def test_a_file_with_no_category_filter_reads_as_excluding_none(self):
        self._categories_file(b"null")
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.proposal == replace(_OLD_VERSION_READS_AS, exclude_categories=None)
        app = _app()
        index = _get(app, "/")
        assert index.status == 200
        assert ("The saved live defaults were saved before the category filter became a "
                "list of categories to exclude, so every live run refuses them until they "
                "are saved again.") in index.body
        assert "read as no category excluded" in index.body
        link = "/confirm?" + defaults_server._settings_query(caught.value.proposal)
        assert link in _parse(index.body).links
        confirm = _get(app, link)
        assert confirm.status == 200
        assert ("The saved live defaults were saved before the category filter became a "
                "list of categories to exclude. Below: their values, read as no category "
                "excluded. Confirm and save to replace the file.") in html.unescape(confirm.body)
        assert _post(app, _click(confirm.body, "confirm")).status == 303
        assert config.read_saved_live_defaults() == caught.value.proposal

    def test_a_file_that_listed_categories_offers_its_values_with_a_warning(self):
        # The old file traded only Sports and Economics: its values in their
        # new form exclude no category, so every page that offers them says in
        # red that saving them as they are trades every category
        self._categories_file(b'["Sports", "Economics"]')
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.old_categories == ("Sports", "Economics")
        proposal = caught.value.proposal
        assert proposal == replace(_OLD_VERSION_READS_AS, exclude_categories=None)
        link = "/confirm?" + defaults_server._settings_query(proposal)
        starter = _Starter()
        app = _app(start_process=starter)
        warning = ("The saved defaults traded only these categories: Sports, Economics. "
                   "This version saves the categories to EXCLUDE instead, and the values "
                   "offered here exclude none, so saving them as they are trades every "
                   "category. To keep the old filter, add <code>&exclude_category=<name></code> to the "
                   "address for each category to leave out, or save from the backtest "
                   "dashboard's Save as live defaults… button with those categories ticked "
                   "in its Exclude categories menu.")
        index = _get(app, "/")
        assert index.status == 200
        for response in (index, _get(app, "/saved")):
            body = html.unescape(response.body)
            assert warning in body
            assert "before the category filter became a list of categories to exclude" \
                in body
            # The link offered is the file's own values, never the seed's (the
            # index links the seed page as it always does)
            assert link in _parse(response.body).links
            if response is not index:
                assert ("/confirm?" + defaults_server._seed_query()
                        not in _parse(response.body).links)
        assert _get(app, "/saved").status == 409
        # The trading page with no address fields proposes those values, warning too
        home = _get(app, "/confirm")
        assert home.status == 200 and warning in html.unescape(home.body)
        # The confirmation page the link opens proposes the file's other values
        confirm = _get(app, link)
        assert confirm.status == 200
        assert warning in html.unescape(confirm.body)
        cells = _parse(confirm.body).cells
        assert cells["rebalance"][2] == "yes"
        assert cells["exclude categories"][2] == "none"
        # A page proposing other values over the file still warns
        other = _get(app, "/confirm?" + _query(exclude_category="Politics"))
        assert warning in html.unescape(other.body)
        # An older version's trade form is still refused, nothing started, the
        # file as it was
        old = config.LIVE_DEFAULTS_FILE.read_bytes()
        fingerprint, nonce = defaults_server._fingerprint(None, caught.value), "3" * 32
        form = [("fingerprint", fingerprint), ("nonce", nonce),
                ("token", app._token("trade", fingerprint, nonce, {})), ("action", "trade")]
        assert _post(app, form, target="/trade").status == 404
        assert _post(app, form).status == 403
        assert starter.calls == []
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == old
        # One Confirm and save writes exactly the file's other values
        assert _post(app, _click(confirm.body, "confirm")).status == 303
        saved = config.read_saved_live_defaults()
        assert saved == proposal
        assert (saved.tier_floors, saved.spread_band, saved.interval_discount,
                saved.size_cap, saved.rebalance, saved.min_price_move, saved.sell_at,
                saved.exclude_categories) == (
            False, (0.0, 0.8), 0.7, 0.15, True, 0.1, None, None)

    def test_a_listed_file_s_link_keeps_its_sell_rule_and_tags(self):
        # The offered link carries every one of the file's other settings
        config.LIVE_DEFAULTS_FILE.write_bytes(
            _OLD_VERSION_BYTES
            .replace(b'"categories": null', b'"categories": ["Sports"]')
            .replace(b'"tags": null', b'"tags": ["Sports \xc2\xb7 Basketball"]')
            .replace(b'"trim_min_rise": 0.1\n',
                     b'"trim_min_rise": 0.1,\n    "sell_at": 0.85,\n    "sell_min_days": 7\n'))
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        proposal = caught.value.proposal
        assert (proposal.sell_at, proposal.sell_min_days, proposal.exclude_tags) == (
            0.85, 7, None)
        assert caught.value.old_tags == ("Sports · Basketball",)
        index = _get(_app(), "/")
        assert "/confirm?" + defaults_server._settings_query(proposal) in _parse(index.body).links
        body = html.unescape(index.body)
        assert "traded only these categories: Sports." in body
        # The old tags are named too: their new form excludes no tag
        assert ("limited trading to these tags: Sports · Basketball (a \"Category · Tag\" "
                "limited that category alone).") in body
        assert "before the tag filter became a list of tags to exclude" in body

    @staticmethod
    def _tags_file(tags: bytes) -> None:
        """
        Write a file the version before tag exclusions saved: exclude_categories, the given tags.

        Args:
            tags (bytes): The JSON text of its tags value.
        """
        config.LIVE_DEFAULTS_FILE.write_bytes(
            _OLD_VERSION_BYTES
            .replace(b'    "add_to_held_pairs": true,\n    "trim_to_kelly": true,\n'
                     b'    "trim_min_rise": 0.1\n',
                     b'    "rebalance": true,\n    "min_price_move": 0.1\n')
            .replace(b'"categories": null', b'"exclude_categories": ["Politics"]')
            .replace(b'"tags": null', b'"tags": ' + tags))

    def test_a_file_with_no_tag_filter_reads_as_excluding_none(self):
        self._tags_file(b"null")
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        # The file holds the same-title cap too, a later change than the tag
        # filter's, so only the tag filter's change is named
        assert caught.value.retired == ("tags", "same_title_size_cap")
        assert caught.value.old_tags is None
        assert caught.value.proposal == replace(_OLD_VERSION_READS_AS,
                                                exclude_categories=("Politics",))
        assert defaults_server._outdated_changes(caught.value) == (False, False, True, False)
        app = _app()
        index = _get(app, "/")
        assert ("The saved live defaults were saved before the tag filter became a list "
                "of tags to exclude, so every live run refuses them until they are saved "
                "again.") in index.body
        assert "read as no tag excluded</a>" in index.body
        assert "same-title" not in index.body
        assert "limited trading to these tags" not in html.unescape(index.body)
        link = "/confirm?" + defaults_server._settings_query(caught.value.proposal)
        confirm = _get(app, link)
        assert ("The saved live defaults were saved before the tag filter became a list of "
                "tags to exclude. Below: their values, read as no tag excluded. Confirm and "
                "save to replace the file.") in html.unescape(confirm.body)
        assert _post(app, _click(confirm.body, "confirm")).status == 303
        assert config.read_saved_live_defaults() == caught.value.proposal

    def test_a_file_that_listed_tags_offers_its_values_with_a_warning(self):
        # The old file traded only Basketball, and of Sports only Soccer: its
        # values in their new form exclude no tag, so every page that offers
        # them says in red that saving them as they are trades every tag
        self._tags_file('["Basketball", "Sports · Soccer"]'.encode())
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.old_tags == ("Basketball", "Sports · Soccer")
        proposal = caught.value.proposal
        assert proposal.exclude_tags is None and proposal.exclude_categories == ("Politics",)
        link = "/confirm?" + defaults_server._settings_query(proposal)
        warning = ("The saved defaults limited trading to these tags: Basketball, Sports · "
                   "Soccer (a \"Category · Tag\" limited that category alone). This version saves the tags to EXCLUDE instead, and the values offered "
                   "here exclude none, so saving them as they are trades every tag. To keep "
                   "the old filter, add <code>&exclude_tag=<name></code> to the address for "
                   "each tag to leave out (\"Category · Tag\" for a tag of one category), or "
                   "save from the backtest dashboard's Save as live defaults… button with "
                   "those tags ticked in its Exclude tags menu.")
        app = _app()
        for response in (_get(app, "/"), _get(app, "/confirm"), _get(app, "/saved"),
                         _get(app, link)):
            body = html.unescape(response.body)
            assert warning in body
            assert "before the tag filter became a list of tags to exclude" in body
            assert "traded only these categories" not in body
        assert _get(app, "/saved").status == 409
        # One Confirm and save writes the file's other values, no tag excluded
        assert _post(app, _click(_get(app, link).body, "confirm")).status == 303
        assert config.read_saved_live_defaults() == proposal

    def test_a_file_that_listed_both_names_both(self):
        # A file older still (a list of categories and a list of tags to
        # trade) gets both warnings and both changes named
        config.LIVE_DEFAULTS_FILE.write_bytes(
            _OLD_VERSION_BYTES
            .replace(b'"categories": null', b'"categories": ["Sports"]')
            .replace(b'"tags": null', b'"tags": ["Basketball"]'))
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        when = defaults_server._outdated_when(caught.value)
        assert when == ("before Rebalance and before the category filter became a list of "
                        "categories to exclude and before the tag filter became a list of "
                        "tags to exclude")
        assert defaults_server._outdated_reading(caught.value) == (
            "Rebalance yes and Min. price movement 10c and no category excluded and no tag "
            "excluded")
        body = html.unescape(_get(_app(), "/").body)
        assert "traded only these categories: Sports." in body
        assert "limited trading to these tags: Basketball (" in body

    def test_a_malformed_tags_value_offers_the_seed(self):
        self._tags_file(b'"Basketball"')
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.proposal is None and caught.value.old_tags is None
        saved = _get(_app(), "/saved")
        assert saved.status == 409
        assert "/confirm?" + defaults_server._seed_query() in _parse(saved.body).links
        assert "before the tag filter became a list of tags to exclude" in html.unescape(
            saved.body)
        assert "limited trading to these tags" not in html.unescape(saved.body)

    def test_a_start_logs_the_old_tags(self, run_main):
        self._tags_file(b'["Basketball"]')
        log = run_main["run"](["--no-browser"])
        assert ("They limited trading to the tags Basketball (a \"Category · Tag\" limited "
                "that category alone): saved as offered they trade every tag, so add an exclude_tag= field for each tag to leave out") in log

    def test_a_malformed_categories_value_offers_the_seed(self):
        self._categories_file(b'"Sports"')
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        assert caught.value.proposal is None and caught.value.old_categories is None
        saved = _get(_app(), "/saved")
        assert saved.status == 409
        assert "/confirm?" + defaults_server._seed_query() in _parse(saved.body).links
        assert "traded only these categories" not in html.unescape(saved.body)


# Settings that differ from tests/test_config.py's _SAME_TITLE_CAP_ERA_PROPOSAL
# in every toggle, saved where a test needs other defaults in force
_UNLIKE_THE_CAP_ERA = LiveSettings(
    tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75, size_cap=0.2,
    exclude_categories=("Sports",), exclude_tags=("Basketball",), rebalance=False,
    sell_at=0.9, sell_min_days=5)


class TestSameTitleCapFile:
    """A file saved before the same-title cap was removed: it holds
    same_title_size_cap and no older retired setting, the shape of every file
    saved since the tag filter became a list of tags to exclude, and the shape
    of the account's own saved defaults (tests/test_config.py's
    _SAME_TITLE_CAP_ERA_BYTES). No run trades it. Every page names that one
    change and offers its other settings as saved, with no same-title cap, for
    one Confirm and save; the save writes no same-title cap, and the runs that
    follow read the new file."""

    # The notice /, /trade and /saved show for it, with its link
    @staticmethod
    def _notice(link: str) -> str:
        """
        The red notice the pages that cannot use the file show, as HTML.

        Args:
            link (str): The confirmation link it carries.

        Returns:
            str: The one paragraph.
        """
        return ('<p class="banner">The saved live defaults were saved before the same-title '
                "cap was removed, so every live run refuses them until they are saved again. "
                f'<a href="{html.escape(link)}">Review their values, read as no same-title '
                "cap</a>: one Confirm and save there replaces the file.</p>")

    # The banner the confirmation page shows above its table when it proposes them
    _BANNER = ('<p class="banner">The saved live defaults were saved before the same-title cap '
               "was removed. Below: their values, read as no same-title cap. Confirm and save "
               "to replace the file.</p>")

    @staticmethod
    def _write(data: bytes | None = None) -> tuple[config.LiveDefaultsOutdated, str]:
        """
        Save the account-shaped file (or other bytes) as the live defaults, and read its refusal.

        Args:
            data (bytes | None): The file's bytes; None writes
                _SAME_TITLE_CAP_ERA_BYTES.

        Returns:
            tuple[config.LiveDefaultsOutdated, str]: The refusal, and the
                /confirm? link to its proposal ("" when it has none).
        """
        from .test_config import _SAME_TITLE_CAP_ERA_BYTES
        config.LIVE_DEFAULTS_FILE.write_bytes(_SAME_TITLE_CAP_ERA_BYTES if data is None else data)
        with pytest.raises(config.LiveDefaultsOutdated) as caught:
            config.read_saved_live_defaults()
        proposal = caught.value.proposal
        link = "" if proposal is None else "/confirm?" + defaults_server._settings_query(proposal)
        return caught.value, link

    def test_its_one_change_is_named_and_read_as_no_same_title_cap(self):
        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        outdated, _ = self._write()
        assert outdated.retired == ("same_title_size_cap",)
        assert outdated.proposal == _SAME_TITLE_CAP_ERA_PROPOSAL
        assert defaults_server._outdated_changes(outdated) == (False, False, False, True)
        assert defaults_server._outdated_when(outdated) == "before the same-title cap was removed"
        # Never read as the Rebalance choice, though the file rebalances
        assert defaults_server._outdated_reading(outdated) == "no same-title cap"
        assert "Rebalance" not in defaults_server._outdated_reading(outdated)

    @pytest.mark.parametrize("target, status", [("/", 200), ("/saved", 409)])
    def test_every_page_that_cannot_use_it_shows_the_notice_and_the_link(self, target, status):
        outdated, link = self._write()
        response = _get(_app(start_process=_Starter()), target)
        assert response.status == status
        assert self._notice(link) in response.body
        assert link in _parse(response.body).links
        # No other change is named, nor any widening of the filter
        body = html.unescape(response.body)
        assert "before Rebalance" not in body and "filter became" not in body
        assert "read as Rebalance" not in body
        assert "traded only these categories" not in body
        assert "limited trading to these tags" not in body
        assert "Fix or delete" not in body

    def test_the_link_names_every_field_of_its_values_and_no_same_title_cap(self):
        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        _, link = self._write()
        params = defaults_server._params(link.split("?", 1)[1])
        assert params == {
            "tier_floors": ["off"], "spread_min": ["0.2"], "spread_max": ["0.8"], "k": ["0.85"],
            "size_cap": ["0.1"], "rebalance": ["on"], "min_price_move": ["0.1"],
            "rebalance_min_days_to_maturity": ["7"], "sell_at": ["off"], "sell_min_days": ["off"],
            "exclude_tag": ["Climate and Weather · Ski Resorts", "Entertainment · General"]}
        # It proposes exactly them, whatever is saved
        for saved in (None, _UNLIKE_THE_CAP_ERA, config.LIVE_DEFAULTS_SEED):
            assert defaults_server._proposal(params, saved) == (_SAME_TITLE_CAP_ERA_PROPOSAL, "")

    def test_the_confirmation_page_shows_the_banner_and_eleven_rows(self):
        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        _, link = self._write()
        response = _get(_app(), link)
        assert response.status == 200
        assert self._BANNER in response.body
        # The banner sits above the table
        assert response.body.index(self._BANNER) < response.body.index("<table>")
        assert "Save the live trading defaults again?" in response.body
        assert ("Live defaults in force: none — the saved file was saved before the same-title "
                "cap was removed, and live runs refuse it until it is saved again") in response.body
        page = _parse(response.body)
        assert list(page.rows) == LABELS and len(page.rows) == 11
        assert "same-title cap" not in page.rows
        # Nothing counts as in force: every row is a change from nothing
        assert set(page.rows.values()) == {"changed"}
        assert all(cells[1] == "—" for cells in page.cells.values())
        assert [page.cells[label][2] for label in LABELS] == [
            new for _, _, new, _ in config.live_settings_changes(None, _SAME_TITLE_CAP_ERA_PROPOSAL)]
        assert page.cells["exclude tags"][2] == (
            "Climate and Weather · Ski Resorts, Entertainment · General")
        assert "11 of 11 settings change." in response.body
        # These settings warn of nothing, and the save sells nothing more
        assert config.live_rule_warnings(_SAME_TITLE_CAP_ERA_PROPOSAL) == []
        assert 'class="warn"' not in response.body

    def test_another_proposal_over_it_links_to_its_own_values(self):
        # A dashboard page's Save proposes other values: the page says the save
        # replaces the file with them, and links to the file's own
        _, link = self._write()
        response = _get(_app(), "/confirm?" + _query())
        assert response.status == 200
        assert ("The saved live defaults were saved before the same-title cap was removed, so "
                "every live run refuses them. Confirm and save replaces the file with the "
                f'settings below; <a href="{html.escape(link)}">their own values</a> read as no '
                "same-title cap.</p>") in response.body
        assert self._BANNER not in response.body

    def test_a_start_logs_where_to_save_it_again(self, run_main):
        _, link = self._write()
        log = run_main["run"](["--no-browser"])
        assert ("WARNING  The saved live defaults were saved before the same-title cap was "
                "removed, and every live run refuses them until they are saved again: open "
                f"http://127.0.0.1:{PORT}{link} to review their values (read as no same-title "
                "cap) and save them; do not delete the file\n") in log

    def test_its_fingerprint_is_its_own_and_its_pages_go_stale_once_it_is_replaced(self):
        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        outdated, link = self._write()
        fingerprint = defaults_server._fingerprint(None, outdated)
        # Neither no file's, nor a file saved before Rebalance's, nor its own
        # values' once they are saved
        assert fingerprint != defaults_server._fingerprint(None)
        older, _ = self._write(_OLD_VERSION_BYTES)
        assert fingerprint != defaults_server._fingerprint(None, older)
        assert fingerprint != defaults_server._fingerprint(_SAME_TITLE_CAP_ERA_PROPOSAL)
        # A page built while it was there, posted after another page's save
        # replaced it, is shown again against what is now saved, saving nothing
        self._write()
        app = _app()
        first = _click(_get(app, link).body, "confirm")
        second = _page_form(app, _query())
        assert _post(app, second).status == 303
        replaced = config.LIVE_DEFAULTS_FILE.read_bytes()
        response = _post(app, first)
        assert response.status == 409
        assert html.escape(defaults_server._STALE_BANNER) in response.body
        assert self._BANNER not in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == replaced

    def test_a_cap_the_older_version_refused_points_at_new_ones(self, run_main):
        # A cap off the 5% grid has no reading, so nothing is offered: the
        # pages link to the seed values, and the start-up WARNING to the index
        from .test_config import _SAME_TITLE_CAP_ERA_BYTES
        outdated, link = self._write(_SAME_TITLE_CAP_ERA_BYTES.replace(
            b'"same_title_size_cap": 0.2', b'"same_title_size_cap": 0.07'))
        assert outdated.proposal is None and link == ""
        assert defaults_server._outdated_changes(outdated) == (False, False, False, True)
        assert defaults_server._outdated_when(outdated) == "before the same-title cap was removed"
        saved = _get(_app(), "/saved")
        assert saved.status == 409
        body = html.unescape(saved.body)
        assert ("The saved live defaults were saved before the same-title cap was removed, and "
                "every live run refuses them:") in body
        assert "its other settings are refused too" in body
        assert "/confirm?" + defaults_server._seed_query() in _parse(saved.body).links
        log = run_main["run"](["--no-browser"])
        assert ("WARNING  The saved live defaults were saved before the same-title cap was "
                "removed, and every live run refuses them (") in log
        assert f"): open http://127.0.0.1:{PORT}/ and save new ones; a save replaces the file\n" \
            in log

    def test_the_account_s_file_is_saved_again_in_one_confirm_and_runs_read_it(self, capsys):
        # End to end, from the file exactly as the account has it saved: every
        # live run refuses it; the start page links to its values; that page
        # shows them in eleven rows; one Confirm and save writes them with no
        # same-title cap; and live runs then read the new file
        from kalshi_betting import main

        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        proposal = _SAME_TITLE_CAP_ERA_PROPOSAL
        _, link = self._write()
        parser = main._build_parser()
        with pytest.raises(SystemExit) as exited:
            main._resolve_live_settings(parser.parse_args(["--mode", "prod"]), parser)
        assert exited.value.code == 2
        assert "same_title_size_cap, which is no longer used" in capsys.readouterr().err
        app = _app()
        # The start page: the notice and the link
        index = _get(app, "/")
        assert index.status == 200
        assert self._notice(link) in index.body
        params = defaults_server._params(link.split("?", 1)[1])
        assert "same_title_size_cap" not in params
        assert defaults_server._proposal(params, None) == (proposal, "")
        # The confirmation page the link opens
        confirm = _get(app, link)
        assert confirm.status == 200
        assert self._BANNER in confirm.body
        page = _parse(confirm.body)
        assert list(page.rows) == LABELS and "same-title cap" not in page.rows
        assert "11 of 11 settings change." in confirm.body
        # Confirm and save, through the page's own form
        posted = _post(app, _click(confirm.body, "confirm"))
        assert posted.status == 303 and posted.location == "/saved"
        stored = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))
        assert "same_title_size_cap" not in stored["settings"]
        saved = config.read_saved_live_defaults()
        assert saved == proposal
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(saved, name) == getattr(proposal, name), name
        # The tied tags keep their middle dot
        assert saved.exclude_tags == ("Climate and Weather · Ski Resorts", "Entertainment · General")
        # Every live run now reads it, and its "Live settings:" line names no same-title cap
        settings, reference = main._resolve_live_settings(parser.parse_args(["--mode", "prod"]),
                                                          parser)
        assert settings == reference == proposal
        line = config.describe_live_settings(settings)
        assert "same-title" not in line and "same_title" not in line
        assert "| exclude tags Climate and Weather · Ski Resorts, Entertainment · General |" in line
        # ... and every page reads it as the defaults in force
        assert _get(app, "/saved").status == 200 and _get(app, "/confirm").status == 200
        assert "same-title" not in _get(app, "/").body

    def test_confirm_and_trade_saves_it_then_runs_exactly_it(self):
        # Confirm and trade from the page the notice links to saves the same
        # file, then starts one real-money run whose command names no
        # same-title cap and, read by main.py's own parser, trades exactly
        # these settings, whatever is saved by then
        from kalshi_betting import main

        from .test_config import _SAME_TITLE_CAP_ERA_PROPOSAL
        proposal = _SAME_TITLE_CAP_ERA_PROPOSAL
        _, link = self._write()
        starter = _Starter()
        app = _app(start_process=starter)
        run_id = _run_id(_post(app, _page_form(app, link.split("?", 1)[1], "confirm-trade")))
        # Saved first, exactly as Confirm and save writes it
        assert "same_title_size_cap" not in config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8")
        assert config.read_saved_live_defaults() == proposal
        [(argv, kwargs)] = starter.calls
        folder = _run_folder(run_id)
        assert argv == [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
                        *config.live_settings_argv(proposal), "--result-file",
                        str(folder / "result.json")]
        assert not any(arg.startswith("--same-title") for arg in argv)
        assert kwargs["start_new_session"] is True
        [record] = starter.run_json
        assert record["dry_run"] is False
        assert record["settings"] == config.describe_live_settings(proposal)
        assert "same-title" not in record["settings"]
        # main.py's parser reads the command (after "-m kalshi_betting.main")
        # as exactly the proposal, over saved defaults unlike it in every toggle
        parser = main._build_parser()
        args = parser.parse_args(argv[3:])
        assert args.retired_same_title_size_cap is None
        assert (args.mode, args.dry_run, args.result_file) == (
            "prod", False, folder / "result.json")
        assert all(getattr(_UNLIKE_THE_CAP_ERA, name) != getattr(proposal, name)
                   for name in config.LIVE_TOGGLE_FIELDS)
        config.save_live_defaults(_UNLIKE_THE_CAP_ERA, source="")
        settings, _ = main._resolve_live_settings(args, parser)
        assert settings == proposal
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(settings, name) == getattr(proposal, name), name


class TestSave:
    """POST /confirm with Confirm and save writes exactly what its page showed, after every check."""

    def test_a_first_save_writes_exactly_the_proposal(self):
        app = _app()
        form = _page_form(app, _query(source=DASHBOARD_NOTE))
        response = _post(app, form)
        assert response.status == 303 and response.location == "/saved"
        saved = config.read_saved_live_defaults()
        assert saved == _BASE_SETTINGS
        assert saved.origin.endswith(" from " + DASHBOARD_NOTE)

    def test_a_rebalance_until_is_written_after_the_minimum_and_read_back(self):
        app = _app()
        form = _page_form(app, _query(rebalance="on", min_price_move="0.1",
                                      rebalance_min_days_to_maturity="7"))
        assert ("rebalance_min_days_to_maturity", "7") in form
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.rebalance_min_days_to_maturity == 7
        assert saved == replace(_BASE_SETTINGS, min_price_move=0.10,
                                rebalance_min_days_to_maturity=7)
        text = config.LIVE_DEFAULTS_FILE.read_text()
        assert text.index('"min_price_move"') < text.index('"rebalance_min_days_to_maturity": 7')
        # Saving "maturity" (0) leaves the key out again
        form = _page_form(app, _query(rebalance_min_days_to_maturity="0"))
        assert _post(app, form).status == 303
        assert config.read_saved_live_defaults().rebalance_min_days_to_maturity is None
        assert "rebalance_min_days_to_maturity" not in config.LIVE_DEFAULTS_FILE.read_text()

    def test_it_overwrites_the_defaults_in_force(self):
        _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                           size_cap=0.2))
        app = _app()
        form = _page_form(app, _query(exclude_category="Sports", exclude_tag="Basketball"))
        assert _post(app, form).status == 303
        assert config.read_saved_live_defaults() == LiveSettings(
            tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8, size_cap=0.1,
            exclude_categories=("Sports",), exclude_tags=("Basketball",))

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

    def test_the_live_dashboard_s_tab_page_cannot_save(self):
        # The live dashboard's tab page (its own port, so its own web origin)
        # is refused as any other page is
        app = _app()
        form = _page_form(app, _query())
        origin = f"http://{config.LIVE_DASHBOARD_HOST}:{config.LIVE_DASHBOARD_PORT}"
        assert _post(app, form, origin=origin).status == 403
        assert config.read_saved_live_defaults() is None

    def test_the_backtest_page_in_the_live_dashboard_cannot_save(self):
        # The backtest page the live dashboard serves on its second port is
        # another origin too, so a script on it cannot press Confirm and save
        app = _app()
        form = _page_form(app, _query())
        origin = f"http://{config.LIVE_DASHBOARD_HOST}:{config.LIVE_BACKTEST_PORT}"
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
        # The nonce: missing, repeated, not 32 hex digits, or another one
        lambda f: TestSave._without(f, "nonce"),
        lambda f: f + [pair for pair in f if pair[0] == "nonce"],
        lambda f: TestSave._without(f, "nonce") + [("nonce", dict(f)["nonce"][:31])],
        lambda f: TestSave._without(f, "nonce") + [("nonce", dict(f)["nonce"].upper())],
        lambda f: TestSave._without(f, "nonce") + [("nonce", "0" * 32)],
    ])
    def test_the_fingerprint_nonce_and_token_must_be_this_page_s(self, mangle):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, mangle(form)).status == 403
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("mangle", [
        lambda f: TestSave._without(f, "action"),
        lambda f: f + [("action", "save")],
        lambda f: TestSave._without(f, "action") + [("action", "delete")],
        lambda f: TestSave._without(f, "action") + [("action", "")],
        lambda f: f + [("ack_attention", "yes")],
        lambda f: f + [("ack_attention", "1"), ("ack_attention", "1")],
    ])
    def test_the_action_must_be_one_of_the_page_s(self, mangle):
        app = _app()
        form = _page_form(app, _query())
        assert _post(app, mangle(form)).status == 400
        assert config.read_saved_live_defaults() is None

    def test_a_token_from_another_server_session_is_refused(self):
        form = _page_form(_app(KEY), _query())
        response = _post(_app(b"x" * 32), form)
        assert response.status == 403
        assert "another server session" in response.body
        assert config.read_saved_live_defaults() is None

    @pytest.mark.parametrize("tamper", [
        lambda f: [(n, "0.4" if n == "k" else v) for n, v in f],
        lambda f: f + [("exclude_category", "Economics")],
        lambda f: [(n, v) for n, v in f if n != "exclude_tag"],
        lambda f: [(n, "1" if n == "rebalance_min_days_to_maturity" else v) for n, v in f],
        lambda f: [(n, v) for n, v in f if n != "rebalance_min_days_to_maturity"],
    ])
    def test_a_field_changed_after_signing_is_refused(self, tamper):
        app = _app()
        form = _page_form(app, _query(exclude_category="Sports", exclude_tag="Basketball",
                                      rebalance_min_days_to_maturity="7"))
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
        other = _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                                   size_cap=0.2))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        response = _post(app, form)
        assert response.status == 409
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert defaults_server._STALE_BANNER in response.body
        assert other.origin in response.body
        again = _click(response.body, "confirm")
        assert dict(again)["fingerprint"] != dict(form)["fingerprint"]
        # A fresh nonce too, though the stale POST never used its own
        assert dict(again)["nonce"] != dict(form)["nonce"]
        # The page shown again saves
        assert _post(app, again).status == 303
        assert config.read_saved_live_defaults() == LiveSettings(
            tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8, size_cap=0.1)

    def test_a_failed_save_is_a_500_and_keeps_the_old_file(self, monkeypatch):
        _save(LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                           size_cap=0.2))
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
        assert "no rename today" in response.body and "Nothing was traded." in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_a_zero_floor_and_a_non_ascii_tag_round_trip(self):
        app = _app()
        form = _page_form(app, _query(spread_min="0", exclude_category="Deportes", exclude_tag="Fútbol"))
        assert ("exclude_tag", "Fútbol") in form
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.spread_band == (0.0, 0.5) and saved.exclude_tags == ("Fútbol",)

    def test_nothing_to_save_writes_nothing(self):
        # The page's own Confirm and save, with nothing changed: the file is
        # left as it is, and the browser is sent to the saved page saying so
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        app = _app()
        form = _page_form(app, _query())
        response = _post(app, form)
        assert response.status == 303 and response.location == "/saved?unchanged=1"
        assert "nothing was written" in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        # The nonce is used: the same form is sent to the same place, still writing nothing
        again = _post(app, form)
        assert again.status == 303 and again.location == "/saved?unchanged=1"
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_the_save_is_logged_against_the_defaults_it_replaced(self, caplog):
        _save(LiveSettings(tier_floors=True, spread_band=(0.0, 0.5), interval_discount=0.8,
                           size_cap=0.1))
        app = _app()
        form = _page_form(app, _query())
        with caplog.at_level(logging.INFO):
            assert _post(app, form).status == 303
        [line] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("Saved live defaults to")]
        assert f"to {config.LIVE_DEFAULTS_FILE}: tier floors off (default: on) |" in line

    def test_a_refusal_is_logged_as_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _get(_app(), "/confirm?" + _query(category="Sports"))
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.getMessage() == (
            "Refused (400 Bad Request): this link names category, which exclude_category "
            "replaced (the category filter is now a list of categories NOT to trade): "
            "rebuild the backtest dashboard, or open this server's start page")


class TestSavedPage:
    """GET /saved shows the live defaults in force, saved and verified."""

    def test_it_shows_the_new_defaults(self):
        app = _app()
        assert _post(app, _page_form(app, _query(source=DASHBOARD_NOTE))).status == 303
        response = _get(app, "/saved")
        assert response.status == 200
        saved = config.read_saved_live_defaults()
        assert (f"Saved and verified: {saved.origin}. The file was read back and every "
                "setting matched what was confirmed.") in response.body
        page = _parse(response.body)
        assert list(page.rows) == LABELS
        assert page.cells["tier floors"] == ["tier floors", "off"]
        assert page.cells["per-trade cap"] == ["per-trade cap", "10%"]
        assert page.forms == [] and page.scripts == [] and page.buttons == []
        assert "/confirm" in page.links and ">Back to the trading page</a>" in response.body
        assert "/trade" not in page.links
        # The save is done: the page no longer speaks of writing
        assert "This writes" not in response.body and "click" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body

    def test_a_warning_is_shown_as_a_note_on_what_was_saved(self):
        saved = _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.4,
                                   size_cap=1.0))
        body = _get(_app(), "/saved").body
        sentences = config.live_rule_warnings(saved)
        assert sentences
        for sentence in sentences:
            assert f'<p class="warn">Saved (valid), but note: {sentence}.</p>' in body

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
        saved = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                             size_cap=0.2)
        assert defaults_server._proposal(params, saved)[0] == config.LIVE_DEFAULTS_SEED

    def test_the_seed_query_spells_selling_off(self):
        pairs = dict(defaults_server._params(defaults_server._seed_query()))
        assert pairs["sell_at"] == ["off"] and pairs["sell_min_days"] == ["off"]
        assert config.LIVE_DEFAULTS_SEED.sell_at is None
        assert config.LIVE_DEFAULTS_SEED.sell_min_days is None

    def test_the_seed_query_spells_rebalancing_on_with_no_minimum(self):
        params = defaults_server._params(defaults_server._seed_query())
        assert params["rebalance"] == ["on"] and params["min_price_move"] == ["off"]
        assert config.LIVE_DEFAULTS_SEED.rebalance is True
        assert config.LIVE_DEFAULTS_SEED.min_price_move is None
        # Every field the seed sets, in the server's field order, then its note;
        # no same-title cap, which the server refuses on a link
        assert list(params) == ["tier_floors", "spread_min", "spread_max", "k", "size_cap",
                                "rebalance", "min_price_move",
                                "rebalance_min_days_to_maturity", "sell_at", "sell_min_days",
                                "source"]
        assert defaults_server._proposal(params, None) == (
            config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)

    def test_the_seed_query_spells_maturity_as_zero(self):
        # No limit on how close to maturity a held position is rebalanced
        params = defaults_server._params(defaults_server._seed_query())
        assert params["rebalance_min_days_to_maturity"] == ["0"]
        assert config.LIVE_DEFAULTS_SEED.rebalance_min_days_to_maturity is None

    @pytest.mark.parametrize("saved, row, cells", [
        (replace(config.LIVE_DEFAULTS_SEED, min_price_move=0.25),
         "min price movement", ["25c", "any"]),
        (replace(config.LIVE_DEFAULTS_SEED, rebalance=False), "rebalance", ["no", "yes"]),
    ], ids=["a saved minimum", "rebalancing saved off"])
    def test_the_seed_page_proposes_the_seed_s_rebalance_settings(self, saved, row, cells):
        # Over saved defaults that differ only there, the seed link changes one row
        _save(saved)
        response = _get(_app(), "/confirm?" + defaults_server._seed_query())
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [row]
        assert page.cells[row][1:3] == cells
        assert "1 of 11 settings change." in response.body

    def test_the_seed_page_proposes_the_seed_s_rebalance_until(self):
        # Over saved defaults that stop short of maturity, the seed link changes
        # that one row, and a seed link that leaves the field out does too
        _save(replace(config.LIVE_DEFAULTS_SEED, rebalance_min_days_to_maturity=7))
        stale = "&".join(part for part in defaults_server._seed_query().split("&")
                         if not part.startswith("rebalance_min_days_to_maturity="))
        for query in (defaults_server._seed_query(), stale):
            response = _get(_app(), "/confirm?" + query)
            assert response.status == 200, query
            page = _parse(response.body)
            assert [label for label, css in page.rows.items() if css == "changed"] == [
                "rebalance until"]
            assert page.cells["rebalance until"][1:3] == ["7 days before maturity", "maturity"]
            assert "1 of 11 settings change." in response.body
            assert defaults_server._proposal(
                defaults_server._params(query), config.read_saved_live_defaults()) == (
                config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)

    def test_a_file_with_rebalancing_off_leaves_its_key_out(self):
        # A saved file that leaves rebalance out reads it as off; the writer
        # leaves it out while it is off
        _save(replace(config.LIVE_DEFAULTS_SEED, rebalance=False))
        assert '"rebalance"' not in config.LIVE_DEFAULTS_FILE.read_text()
        assert config.live_defaults().rebalance is False

    @pytest.mark.parametrize("saved, row", [
        (replace(config.LIVE_DEFAULTS_SEED, min_price_move=0.25), "min price movement"),
        (replace(config.LIVE_DEFAULTS_SEED, rebalance=False), "rebalance"),
    ], ids=["a saved minimum", "rebalancing saved off"])
    def test_a_seed_link_that_leaves_rebalancing_out_still_proposes_the_seed(self, saved, row):
        # A seed link with neither rebalance field still proposes the seed: it
        # is not refused for the seed's note on other values, nor (with
        # rebalancing kept on) for a saved minimum the seed does not set
        _save(saved)
        stale = "&".join(part for part in defaults_server._seed_query().split("&")
                         if not part.startswith(("rebalance=", "min_price_move=",
                                                 "rebalance_min_days_to_maturity=")))
        assert "rebalance" not in stale and "source=" in stale
        params = defaults_server._params(stale)
        assert defaults_server._proposal(params, config.read_saved_live_defaults()) == (
            config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)
        app = _app()
        response = _get(app, "/confirm?" + stale)
        assert response.status == 200
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [row]
        assert _post(app, _click(response.body, "confirm")).status == 303
        assert config.live_defaults() == config.LIVE_DEFAULTS_SEED
        # Control: the same link without the seed's note keeps what is saved
        plain = "&".join(part for part in stale.split("&") if not part.startswith("source="))
        kept, _ = defaults_server._proposal(defaults_server._params(plain), saved)
        assert (kept.rebalance, kept.min_price_move) == (saved.rebalance, saved.min_price_move)

    def test_the_seed_page_turns_off_selling_that_is_saved(self):
        # The seed link names selling off, so it proposes exactly the seed over
        # defaults that sell: both rows change
        _save(replace(config.LIVE_DEFAULTS_SEED, sell_at=0.85, sell_min_days=3))
        response = _get(_app(), "/confirm?" + defaults_server._seed_query())
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "sell at", "min days to maturity"]
        assert "2 of 11 settings change." in response.body

    def test_a_seed_link_that_leaves_selling_out_still_proposes_the_seed(self):
        # A seed link from before selling existed (no sell fields), opened over
        # defaults that sell: the seed's off, shown as changes, never refused
        # for putting the seed's note on other values
        _save(replace(config.LIVE_DEFAULTS_SEED, sell_at=0.85, sell_min_days=3))
        stale = "&".join(part for part in defaults_server._seed_query().split("&")
                         if not part.startswith(("sell_at=", "sell_min_days=")))
        assert "sell_at" not in stale and "source=" in stale
        params = defaults_server._params(stale)
        assert defaults_server._proposal(params, config.read_saved_live_defaults()) == (
            config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)
        app = _app()
        response = _get(app, "/confirm?" + stale)
        assert response.status == 200
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "sell at", "min days to maturity"]
        assert _post(app, _click(response.body, "confirm")).status == 303
        assert config.live_defaults() == config.LIVE_DEFAULTS_SEED
        # Control: the same link without the seed's note keeps what is saved
        plain = "&".join(part for part in stale.split("&") if not part.startswith("source="))
        kept, _ = defaults_server._proposal(
            defaults_server._params(plain),
            replace(config.LIVE_DEFAULTS_SEED, sell_at=0.85, sell_min_days=3))
        assert (kept.sell_at, kept.sell_min_days) == (0.85, 3)

    def test_confirming_the_seed_saves_it(self):
        app = _app()
        response = _get(app, "/confirm?" + defaults_server._seed_query())
        assert response.status == 200
        assert "Save the first live trading defaults?" in response.body
        assert 'class="warn"' not in response.body
        assert _post(app, _click(response.body, "confirm")).status == 303
        live = config.live_defaults()
        assert live == config.LIVE_DEFAULTS_SEED
        assert live.origin.endswith(" from " + config.LIVE_DEFAULTS_SEED_SOURCE)


class TestHelpers:
    """The pieces the routes are built from."""

    @pytest.mark.parametrize("settings, written", [
        (LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                      size_cap=0.1, rebalance=True), "0"),
        (LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                      size_cap=0.1, rebalance=True,
                      rebalance_min_days_to_maturity=1), "1"),
        (LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                      size_cap=0.1, rebalance=True, min_price_move=0.1,
                      rebalance_min_days_to_maturity=7), "7"),
        # With rebalancing off there is no limit to write but "maturity"
        (LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                      size_cap=0.1), "0"),
    ])
    def test_the_settings_query_names_the_limit_and_reads_back(self, settings, written):
        query = defaults_server._settings_query(settings)
        params = defaults_server._params(query)
        assert params["rebalance_min_days_to_maturity"] == [written]
        # Right after the minimum price movement, as in _FIELDS
        assert list(params).index("rebalance_min_days_to_maturity") == (
            list(params).index("min_price_move") + 1)
        # It proposes exactly these settings, whatever is saved
        for saved in (None, LiveSettings(tier_floors=True, spread_band=(0.0, 1.0),
                                         interval_discount=0.75, size_cap=0.2,
                                         rebalance=True,
                                         rebalance_min_days_to_maturity=7)):
            assert defaults_server._proposal(params, saved) == (settings, "")

    def test_the_fields_hold_the_limit_once_and_in_the_queries_order(self):
        assert defaults_server._FIELDS.count("rebalance_min_days_to_maturity") == 1
        assert "rebalance_min_days_to_maturity" not in defaults_server._REPEATABLE
        assert len(defaults_server._FIELDS) == 13
        names = list(defaults_server._params(defaults_server._settings_query(
            replace(config.LIVE_DEFAULTS_SEED, exclude_categories=("A",), exclude_tags=("B",)),
            source=config.LIVE_DEFAULTS_SEED_SOURCE)))
        assert names == list(defaults_server._FIELDS)

    def test_the_same_title_cap_is_a_retired_field_and_no_proposal_field(self):
        # A link may not set it (it is refused), and no page or run names it:
        # the eleven toggles are LiveSettings' own, the cap no longer among them
        assert "same_title_size_cap" not in defaults_server._FIELDS
        assert defaults_server._RETIRED_FIELDS[-1] == "same_title_size_cap"
        assert "same_title_size_cap" not in config.LIVE_TOGGLE_FIELDS
        assert len(config.LIVE_TOGGLE_FIELDS) == 11
        # The notice reads the Rebalance retirement from config's one list
        assert defaults_server._REBALANCE_RETIREMENTS == config._RETIREMENTS[0][0] == (
            "add_to_held_pairs", "trim_to_kelly", "trim_min_rise")

    def test_the_fingerprint_follows_the_defaults_and_when_they_were_saved(self):
        seed = config.LIVE_DEFAULTS_SEED
        assert defaults_server._fingerprint(None) == hashlib.sha256(b"none").hexdigest()
        a = replace(seed, origin="live_defaults.json, saved 2026-09-27T21:05:13Z")
        b = replace(seed, origin="live_defaults.json, saved 2026-09-27T21:05:14Z")
        assert defaults_server._fingerprint(a) == defaults_server._fingerprint(replace(a))
        assert defaults_server._fingerprint(a) != defaults_server._fingerprint(b)
        assert defaults_server._fingerprint(a) != defaults_server._fingerprint(
            replace(a, interval_discount=0.75))

    def test_the_signed_text_is_the_purpose_fingerprint_nonce_and_raw_fields_sorted(self):
        params = {"k": ["0.80"], "tier_floors": ["off"], "token": ["x"],
                  "exclude_category": ["A B"], "action": ["trade"], "nonce": ["y"]}
        assert defaults_server._signed_text("confirm", "f" * 64, "n" * 32, params) == (
            "confirm\n" + "f" * 64 + "\n" + "n" * 32
            + "\nexclude_category=A+B&k=0.80&tier_floors=off")
        assert defaults_server._signed_text("trade", "f" * 64, "n" * 32, {}) == (
            "trade\n" + "f" * 64 + "\n" + "n" * 32 + "\n")

    def test_the_signed_text_keeps_a_repeated_field_s_values_in_order(self):
        # Field names are sorted; the categories and tags stay as the page listed them
        params = {"exclude_tag": ["Sports · Soccer", "Fed"], "k": ["0.8"],
                  "exclude_category": ["Sports", "Economics"]}
        assert defaults_server._signed_text("confirm", "f" * 64, "n" * 32, params) == (
            "confirm\n" + "f" * 64 + "\n" + "n" * 32
            + "\nexclude_category=Sports&exclude_category=Economics"
            + "&exclude_tag=Sports+%C2%B7+Soccer&exclude_tag=Fed&k=0.8")
        other = {**params, "exclude_category": ["Economics", "Sports"]}
        assert defaults_server._signed_text("confirm", "f" * 64, "n" * 32, other) != (
            defaults_server._signed_text("confirm", "f" * 64, "n" * 32, params))

    def test_a_name_list_reads_each_name_once(self):
        params = {"exclude_category": ["Sports", "Economics"], "exclude_tag": ["Sports · Soccer"]}
        assert defaults_server._name_list(params, "exclude_category") == ("Sports", "Economics")
        assert defaults_server._name_list(params, "exclude_tag") == ("Sports · Soccer",)
        assert defaults_server._name_list({"exclude_category": ["Sports"]}, "exclude_tag") is None
        # Another letter case of an earlier name is left out; the first is kept
        assert defaults_server._name_list(
            {"exclude_category": ["Sports", "SPORTS", "Economics", "sports"]}, "exclude_category") == (
                "Sports", "Economics")
        assert defaults_server._name_list(
            {"exclude_tag": ["Health · Covid", "Health · COVID"]}, "exclude_tag") == ("Health · Covid",)
        # The very same name twice is refused, a left-out spelling included
        with pytest.raises(ValueError, match="exclude_category 'Sports' is given twice"):
            defaults_server._name_list({"exclude_category": ["Sports", "Sports"]}, "exclude_category")
        with pytest.raises(ValueError, match="exclude_category 'SPORTS' is given twice"):
            defaults_server._name_list({"exclude_category": ["Sports", "SPORTS", "SPORTS"]}, "exclude_category")
        with pytest.raises(ValueError, match="tag must be one printable name"):
            defaults_server._name_list({"exclude_tag": ["Soccer", " Fed"]}, "exclude_tag")
        with pytest.raises(ValueError, match="tag must not hold two spaces in a row"):
            defaults_server._name_list({"exclude_tag": ["Soccer", "Pro  Football"]}, "exclude_tag")

    def test_a_name_may_hold_single_spaces_only(self):
        for name in ("Climate and Weather", "Sports · Pro Football", "Oil & Gas", "A"):
            assert defaults_server._name(name, "exclude_category") == name
        for bad in ("Climate  and Weather", "Sports ·  Basketball", "Sports  · Basketball",
                    "a   b"):
            with pytest.raises(ValueError, match="exclude_category must not hold two spaces in a row"):
                defaults_server._name(bad, "exclude_category")
        # A tab or a no-break space is not printable, so it never gets that far
        for bad in ("a\tb", "a b", "a  b"):
            with pytest.raises(ValueError, match="exclude_category must be one printable name"):
                defaults_server._name(bad, "exclude_category")

    def test_a_number_reads_plainly(self):
        assert defaults_server._number("-0", "k") == 0.0
        assert str(defaults_server._number("-0", "k")) == "0.0"
        assert defaults_server._number("1e-3", "k") == 0.001
        assert defaults_server._number("0.000", "k") == 0.0
        assert defaults_server._number("0e-400", "k") == 0.0
        with pytest.raises(ValueError, match="out of range"):
            defaults_server._number("1e-400", "spread_min")

    def test_log_safe_escapes_what_cannot_print(self):
        text = defaults_server._log_safe("a\nb\x1b[31m‮c\U000e0041 ok")
        assert text == "a\\x0ab\\x1b[31m\\u202ec\\U000e0041 ok"

    def test_the_handler_logs_through_logging_on_one_line(self, caplog):
        handler = object.__new__(defaults_server._Handler)
        handler.client_address = ("127.0.0.1", 5555)
        with caplog.at_level(logging.INFO):
            handler.log_message('"%s" %s %s', "GET /\r\nInjected: yes HTTP/1.0", "200", "-")
        [record] = caplog.records
        assert "\r" not in record.getMessage() and "\n" not in record.getMessage()
        assert "\\x0d\\x0aInjected" in record.getMessage()

    def test_every_response_carries_the_five_headers(self):
        headers = defaults_server._response_headers(defaults_server._Response(200, ""))
        assert [name for name, _ in headers] == [
            "Content-Type", "Cache-Control", "X-Content-Type-Options",
            "Referrer-Policy", "Content-Security-Policy"]
        values = dict(headers)
        assert values["Content-Type"] == "text/html; charset=utf-8"
        assert values["Referrer-Policy"] == "same-origin"
        # Only the live dashboard's tab page may frame a page: no X-Frame-Options,
        # whose DENY would refuse it too
        assert "X-Frame-Options" not in values
        assert "frame-ancestors http://127.0.0.1:8766 http://localhost:8766" in (
            values["Content-Security-Policy"])
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
        # A status line and the five headers, even on a request read as HTTP/0.9
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
def held_lock():
    """
    Hold the (redirected) live-run lock through a real flock, as another run would.

    Yields:
        int: The descriptor holding it.
    """
    fd = run_lock.acquire()
    assert fd is not None
    try:
        yield fd
    finally:
        os.close(fd)


class TestRenderStates:
    """The trading page's buttons, in every state: fixed order, the real-money one last, placeholders inert."""

    @staticmethod
    def _check(body: str, labels: list[str], actions: tuple) -> _PageParser:
        """
        Check a page's buttons: order, placeholders, real buttons, its forms and its three scripts.

        Args:
            body (str): The page.
            labels (list[str]): Its button labels in page order.
            actions (tuple[str, ...]): The actions its real buttons may post.

        Returns:
            _PageParser: The page's parse.
        """
        page = _parse(body)
        assert page.labels == labels
        ids = [b.get("id") for b in page.buttons]
        if "confirm-trade" in ids:
            assert ids.index("confirm-trade") == len(ids) - 1
        # No armed button comes before Dry run's place (placeholders may)
        armed = [i for i, button_id in enumerate(ids) if button_id is not None]
        if "confirm-dry-run" in ids:
            assert armed[0] == ids.index("confirm-dry-run") == 0
        for button in page.buttons:
            if "id" in button:
                assert button["type"] == "submit" and "disabled" in button
                assert button["id"] in defaults_server._ARMED_BUTTONS
                assert button["name"] == "action" and button["value"] in actions
                assert button["value"] == _ID_ACTION[button["id"]]
            else:
                assert button == {"type": "button", "disabled": None}
        assert page.scripts == _TRADING_SCRIPTS
        # Exactly one form that posts, beside the dropdown bar's, which only
        # asks for the page again and has no button that names an action
        assert page.forms == [("get", "/confirm"), ("post", "/confirm")]
        assert page.bar_buttons and all("name" not in b and "id" not in b
                                        for b in page.bar_buttons)
        return page

    @staticmethod
    def _state(app, state: str, monkeypatch):
        """
        Put the application in one of the four states a page can be rendered in.

        Args:
            app (defaults_server._App): The application.
            state (str): "free", "lock" (another run holds the lock), "own"
                (a run it started is still going) or "attention" (the last
                real-money run needed attention).
            monkeypatch (pytest.MonkeyPatch): Makes run_lock.held answer True
                for "lock" (a real flock is held in TestBlockers).
        """
        if state == "lock":
            monkeypatch.setattr(run_lock, "held", lambda: True)
        elif state == "own":
            _memory_run(app, returncode=None)
        elif state == "attention":
            _attention_result()

    @pytest.mark.parametrize("state", ["free", "lock", "own", "attention"])
    @pytest.mark.parametrize("changes, saved", [(True, True), (True, False), (False, True)])
    def test_the_confirm_page(self, monkeypatch, state, changes, saved):
        if saved:
            _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        self._state(app, state, monkeypatch)
        query = _query(k="0.75") if changes and saved else _query()
        response = _get(app, "/confirm?" + query)
        assert response.status == 200
        page = self._check(response.body, _CONFIRM_LABELS, defaults_server._CONFIRM_ACTIONS)
        ids = [b.get("id") for b in page.buttons]
        # Dry run: needs saved defaults and no own run still going
        dry_applies = saved and state != "own"
        assert (ids[0] == "confirm-dry-run") is dry_applies
        # Save: always a real button (with nothing changed it writes nothing);
        # no run ever blocks it
        assert ids[1] == "confirm"
        # Trade: blocked by the lock and by an own run, never by the banner
        assert (ids[2] == "confirm-trade") is (state in ("free", "attention"))
        banner = "ended needing manual attention" in response.body
        assert banner is (state == "attention")
        # The box shows only with the banner and a real trade button
        assert bool(page.checkboxes) is (state == "attention")
        if state == "lock":
            assert "Another live trading run is in progress" in response.body
        if state == "own":
            assert "A run started from this page is still going" in response.body
            assert "/runs/fedcba9876543210" in page.links

    @pytest.mark.parametrize("state", ["free", "lock", "own", "attention"])
    def test_the_page_with_no_address_fields(self, monkeypatch, state):
        # The saved defaults, proposed as they are: the same three buttons
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        self._state(app, state, monkeypatch)
        response = _get(app, "/confirm")
        assert response.status == 200
        assert "0 of 11 settings change." in response.body
        page = self._check(response.body, _CONFIRM_LABELS, defaults_server._CONFIRM_ACTIONS)
        ids = [b.get("id") for b in page.buttons]
        assert (ids[0] == "confirm-dry-run") is (state != "own")
        assert ids[1] == "confirm"
        assert (ids[2] == "confirm-trade") is (state in ("free", "attention"))
        assert bool(page.checkboxes) is (state == "attention")

    def test_a_page_whose_runs_are_refused_still_has_its_save_and_three_scripts(self):
        # A server built without a process starter runs nothing: only the save is real
        _save(_BASE_SETTINGS)
        page = self._check(_get(_app(), "/confirm?" + _query()).body, _CONFIRM_LABELS,
                           defaults_server._CONFIRM_ACTIONS)
        assert [b.get("id") for b in page.buttons] == [None, "confirm", None]

    def test_every_other_page_has_no_form_button_or_script(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        _disk_run(result=_result())
        for target in ("/", "/saved", "/saved?unchanged=1", "/nope",
                       "/confirm?" + _query(k="nan"), "/runs/0000000000000000", "/trade"):
            page = _parse(_get(app, target).body)
            assert page.forms == [] and page.buttons == [] and page.scripts == [], target
            assert "data-dashboard-message" not in page.body, target
        # A run page has no form or button, and only the notifier
        page = _parse(_get(app, "/runs/0123456789abcdef").body)
        assert page.forms == [] and page.buttons == []
        assert page.scripts == [defaults_server._NOTIFY_JS]


class TestTokenAndNonce:
    """Each page's token starts only its own page's actions, and each page's nonce works once."""

    def test_a_token_signed_for_another_purpose_fails(self):
        # The trading page's form is the one form signed: a token for any other
        # purpose (an older version's trade page's) never starts anything, and
        # /trade takes no POST at all
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        saved = config.read_saved_live_defaults()
        fingerprint, nonce = defaults_server._fingerprint(saved), "3" * 32
        params = {name: [value] for name, value in _BASE.items()}
        for purpose in ("trade", "save", ""):
            token = app._token(purpose, fingerprint, nonce, params)
            form = [(name, values[0]) for name, values in params.items()]
            form += [("fingerprint", fingerprint), ("nonce", nonce), ("token", token),
                     ("action", "dry_run")]
            assert _post(app, form).status == 403, purpose
        confirm_token = app._token("confirm", fingerprint, nonce, {})
        form = [("fingerprint", fingerprint), ("nonce", nonce), ("token", confirm_token),
                ("action", "dry_run")]
        assert _post(app, form, target="/trade").status == 404
        assert app._start_process.calls == []
        assert defaults_server._CONFIRM_PURPOSE == "confirm"

    def test_the_action_is_unsigned_so_save_can_become_trade_under_every_trade_check(
            self, monkeypatch):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)

        def as_trade(form):
            """
            Change a form's action to trade, as a hand-edited request would.

            Args:
                form (list[tuple[str, str]]): The form.

            Returns:
                list[tuple[str, str]]: The same form, its action trade.
            """
            return [(n, "trade" if n == "action" else v) for n, v in form]

        # Blocked: the lock is held, so nothing is saved or started
        monkeypatch.setattr(run_lock, "held", lambda: True)
        form = as_trade(_page_form(app, _query(k="0.75")))
        assert _post(app, form).status == 409
        assert config.read_saved_live_defaults() == _BASE_SETTINGS
        # Free: honoured as a trade
        monkeypatch.setattr(run_lock, "held", lambda: False)
        form = as_trade(_page_form(app, _query(k="0.75")))
        _run_id(_post(app, form))
        assert config.read_saved_live_defaults().interval_discount == 0.75
        assert len(starter.calls) == 1

    @pytest.mark.parametrize("button_id, location", [
        ("confirm", "/saved"), ("confirm-trade", None), ("confirm-dry-run", None)])
    def test_the_same_page_posted_twice_does_its_work_once(self, monkeypatch, button_id,
                                                          location):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        saves = []
        real_save = defaults_server.save_live_defaults

        def counting_save(settings, *, source):
            """
            Count each save, then make it.

            Args:
                settings (LiveSettings): The defaults saved.
                source (str): Their source note.

            Returns:
                LiveSettings: What the real save returns.
            """
            saves.append(settings)
            return real_save(settings, source=source)

        monkeypatch.setattr(defaults_server, "save_live_defaults", counting_save)
        form = _page_form(app, _query(k="0.75"), button_id)
        first = _post(app, form)
        assert first.status == 303
        if location is not None:
            assert first.location == location
        second = _post(app, form)
        assert second.status == 303 and second.location == first.location
        assert defaults_server._STALE_BANNER not in second.body
        assert len(saves) == (0 if button_id == "confirm-dry-run" else 1)
        assert len(starter.calls) == (0 if button_id == "confirm" else 1)

    @pytest.mark.parametrize("button_id", ["confirm-dry-run", "confirm-trade"])
    def test_the_page_with_no_address_fields_posted_twice_starts_one_run(self, button_id):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        form = _home_form(app, button_id)
        first = _post(app, form)
        second = _post(app, form)
        assert first.status == 303 and first.location.startswith("/runs/")
        assert second.status == 303 and second.location == first.location
        assert len(starter.calls) == 1

    def test_a_used_nonce_whose_save_failed_is_refused(self, monkeypatch):
        app = _app(start_process=_Starter())
        form = _page_form(app, _query(), "confirm-trade")
        real_replace = os.replace

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
        assert _post(app, form).status == 500
        # The disk works again; the page's button stays used all the same
        monkeypatch.setattr(os, "replace", real_replace)
        response = _post(app, form)
        assert response.status == 409
        assert "This page's button was already used — saving the live defaults failed" \
            in html.unescape(response.body)
        assert config.read_saved_live_defaults() is None
        assert app._start_process.calls == []


class TestActions:
    """What each action does, and what it never does."""

    def test_confirming_selling_saves_it_and_a_run_is_started_with_its_flags(self):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(sell_at="0.85", sell_min_days="3"), "confirm-trade")
        # The page carried both fields, as the link named them
        assert ("sell_at", "0.85") in form and ("sell_min_days", "3") in form
        run_id = _run_id(_post(app, form))
        saved = config.read_saved_live_defaults()
        assert (saved.sell_at, saved.sell_min_days) == (0.85, 3)
        text = config.LIVE_DEFAULTS_FILE.read_text()
        assert '"sell_at": 0.85,' in text and text.rstrip().endswith('"sell_min_days": 3\n  }\n}')
        [(argv, _)] = starter.calls
        assert argv[5:-2] == config.live_settings_argv(saved)
        assert "--sell-at=85" in argv and "--sell-min-days=3" in argv
        assert "--no-sell" not in argv and "--no-sell-min-days" not in argv
        record = json.loads((_run_folder(run_id) / "run.json").read_text())
        assert "sell at 85% of potential profit | min days to maturity 3" in record["settings"]

    def test_a_run_from_defaults_that_do_not_sell_says_so_in_its_flags(self):
        starter = _Starter()
        app = _app(start_process=starter)
        _post(app, _page_form(app, _query(), "confirm-trade"))
        [(argv, _)] = starter.calls
        assert "--no-sell" in argv and "--no-sell-min-days" in argv
        assert not any(arg.startswith(("--sell-at", "--sell-min-days")) for arg in argv)

    def test_confirming_rebalancing_saves_it_and_a_run_is_started_with_its_flags(self):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(rebalance="on", min_price_move="0.1"), "confirm-trade")
        # The page carried both fields, as the link named them
        assert ("rebalance", "on") in form and ("min_price_move", "0.1") in form
        run_id = _run_id(_post(app, form))
        saved = config.read_saved_live_defaults()
        assert (saved.rebalance, saved.min_price_move) == (True, 0.10)
        # With no selling, the two keys are the file's last, in field order
        text = config.LIVE_DEFAULTS_FILE.read_text()
        assert text.rstrip().endswith('"rebalance": true,\n    "min_price_move": 0.1\n  }\n}')
        [(argv, _)] = starter.calls
        assert argv[5:-2] == config.live_settings_argv(saved)
        assert "--rebalance" in argv and "--min-price-move=10" in argv
        assert "--no-rebalance" not in argv and "--no-min-price-move" not in argv
        record = json.loads((_run_folder(run_id) / "run.json").read_text())
        assert ("rebalance yes | min price movement 10c | rebalance until maturity | sell at off"
                in record["settings"])

    def test_confirming_a_rebalance_until_saves_it_and_a_run_is_started_with_its_flag(self):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(rebalance="on", min_price_move="0.1",
                                      rebalance_min_days_to_maturity="7"), "confirm-trade")
        # The page carried the field, as the link named it
        assert ("rebalance_min_days_to_maturity", "7") in form
        run_id = _run_id(_post(app, form))
        saved = config.read_saved_live_defaults()
        assert saved.rebalance_min_days_to_maturity == 7
        [(argv, _)] = starter.calls
        assert argv[5:-2] == config.live_settings_argv(saved)
        assert "--rebalance-min-days-to-maturity=7" in argv
        assert "--no-rebalance-min-days-to-maturity" not in argv
        record = json.loads((_run_folder(run_id) / "run.json").read_text())
        assert "rebalance until 7 days before maturity" in record["settings"]

    def test_the_page_with_no_address_fields_keeps_a_saved_rebalance_until(self):
        # It runs the saved defaults, flags and all
        saved = _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                                   size_cap=0.1, rebalance=True,
                                   rebalance_min_days_to_maturity=1))
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))
        [(argv, _)] = starter.calls
        assert argv[5:argv.index("--result-file")] == config.live_settings_argv(saved)
        assert "--rebalance-min-days-to-maturity=1" in argv

    def test_a_run_from_defaults_that_do_not_rebalance_says_so_in_its_flags(self):
        starter = _Starter()
        app = _app(start_process=starter)
        _post(app, _page_form(app, _query(rebalance="off"), "confirm-trade"))
        [(argv, _)] = starter.calls
        assert "--no-rebalance" in argv and "--no-min-price-move" in argv
        assert "--rebalance" not in argv
        assert not any(arg.startswith("--min-price-move") for arg in argv)
        # ... and the saved file, with rebalancing off, names neither key
        text = config.LIVE_DEFAULTS_FILE.read_text()
        assert "rebalance" not in text and "min_price_move" not in text

    def test_trade_saves_then_starts_exactly_the_page_s_settings(self):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(k="0.75", exclude_category="Sports"), "confirm-trade")
        run_id = _run_id(_post(app, form))
        saved = config.read_saved_live_defaults()
        # Nothing was saved before, so rebalancing is the seed's (on)
        assert saved == LiveSettings(tier_floors=False, spread_band=(0.0, 0.5),
                                     interval_discount=0.75, size_cap=0.1,
                                     exclude_categories=("Sports",), rebalance=True)
        folder = _run_folder(run_id)
        [(argv, kwargs)] = starter.calls
        assert argv == [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
                        *config.live_settings_argv(saved), "--result-file",
                        str(folder / "result.json")]
        # Rebalancing is spelled out, so the saved file never decides it
        assert "--rebalance" in argv
        assert kwargs["cwd"] == config.PROJECT_ROOT
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["stdout"].name == str(folder / "output.log")
        # run.json was on disk, whole, before the start
        [record] = starter.run_json
        assert record == {"run_id": run_id, "dry_run": False, "argv": argv,
                          "settings": config.describe_live_settings(saved),
                          "started_at": record["started_at"],
                          "saved_note": defaults_server._SAVED_NOTE}
        # ... and gains the process id after it
        after = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert after == {**record, "pid": 4242}
        assert folder.name == f"{record['started_at'].replace('-', '').replace(':', '')}-{run_id}"

    def test_trade_with_nothing_to_change_starts_without_saving(self):
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _page_form(app, _query(), "confirm-trade")))
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert starter.run_json[0]["saved_note"] == defaults_server._NOTHING_SAVED_NOTE
        assert "--dry-run" not in starter.calls[0][0]

    def test_dry_run_saves_nothing_and_asks_main_for_a_dry_run(self):
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _page_form(app, _query(k="0.75"), "confirm-dry-run")))
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        [(argv, _)] = starter.calls
        assert argv[-1] == "--dry-run"
        # The proposal's settings, not the saved ones, reach the run
        assert "--interval-discount=0.75" in argv
        assert starter.run_json[0]["dry_run"] is True
        assert starter.run_json[0]["saved_note"] == defaults_server._DRY_RUN_SAVED_NOTE

    def test_the_page_with_no_address_fields_trades_the_saved_defaults(self):
        # Confirm and trade on the saved defaults as they are: the exact argv a
        # trade of them always had, and the file left as it is
        saved = _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                                   size_cap=0.15, exclude_categories=("Sports",),
                                   exclude_tags=("Health · Covid",)), source=DASHBOARD_NOTE)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        run_id = _run_id(_post(app, _home_form(app)))
        [(argv, _)] = starter.calls
        assert argv == [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
                        *config.live_settings_argv(saved), "--result-file",
                        str(_run_folder(run_id) / "result.json")]
        assert starter.run_json[0]["saved_note"] == defaults_server._NOTHING_SAVED_NOTE
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_the_page_with_no_address_fields_dry_runs_the_saved_defaults(self):
        saved = _save(replace(_BASE_SETTINGS, sell_at=0.85, sell_min_days=3))
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))
        [(argv, _)] = starter.calls
        assert argv[5:-3] == config.live_settings_argv(saved) and argv[-1] == "--dry-run"
        assert starter.run_json[0]["saved_note"] == defaults_server._DRY_RUN_SAVED_NOTE
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_the_page_with_no_address_fields_is_signed_like_any_other(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        form = _home_form(app, "confirm-dry-run")
        assert _post(app, form + [("foo", "1")]).status == 400
        assert _post(app, [(n, "0.75" if n == "k" else v) for n, v in form]).status == 403
        assert _post(app, [(n, v) for n, v in form if n != "sell_at"]).status == 403
        assert app._start_process.calls == []

    def test_a_failed_save_starts_nothing(self, monkeypatch):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(), "confirm-trade")

        def refuse(src, dst):
            """
            Stand in for os.replace, failing as a full disk would.

            Args:
                src: The file to rename.
                dst: Where to.

            Raises:
                OSError: Always.
            """
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", refuse)
        response = _post(app, form)
        assert response.status == 500
        assert "Saving the live defaults failed" in response.body
        assert starter.calls == []
        assert not config.LIVE_RUNS_DIR.exists()

    @pytest.mark.parametrize("button_id, saved_sentence", [
        ("confirm-trade", True), ("confirm-dry-run", False)])
    def test_a_start_that_fails_is_a_500_naming_what_was_saved(self, button_id,
                                                               saved_sentence):
        if button_id == "confirm-dry-run":
            _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter(fail=OSError("no fork today")))
        form = _page_form(app, _query(k="0.75"), button_id)
        response = _post(app, form)
        assert response.status == 500
        assert "The run could not be started: no fork today." in response.body
        assert ("The new defaults were saved." in response.body) is saved_sentence
        assert (config.read_saved_live_defaults().interval_discount == 0.75) is saved_sentence
        # The folder says it never started, and its page says nothing ran
        [folder] = list(config.LIVE_RUNS_DIR.iterdir())
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert record["start_error"] == "no fork today"
        page = _get(app, f"/runs/{folder.name[-16:]}").body
        assert "The run could not be started" in page and "no order was sent" in page
        # A second POST of the page says what happened
        again = _post(app, form)
        assert again.status == 409 and "the run could not be started" in again.body

    def test_a_failed_run_json_rewrite_after_the_start_still_redirects(self, monkeypatch):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        real_write = defaults_server._write_json
        writes = []

        def second_fails(path, record):
            """
            Write run.json as the server does, failing the second write (the process id's).

            Args:
                path (Path): The file.
                record (dict): The record.

            Raises:
                OSError: On the second write.
            """
            writes.append(path)
            if len(writes) == 2:
                raise OSError("disk full")
            real_write(path, record)

        monkeypatch.setattr(defaults_server, "_write_json", second_fails)
        run_id = _run_id(_post(app, _page_form(app, _query(), "confirm-dry-run")))
        record = json.loads((_run_folder(run_id) / "run.json").read_text(encoding="utf-8"))
        assert "pid" not in record
        assert _get(app, f"/runs/{run_id}").status == 200

    def test_the_start_is_logged(self, caplog):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        with caplog.at_level(logging.INFO):
            _run_id(_post(app, _page_form(app, _query(), "confirm-dry-run")))
        [line] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("Started a live trading run")]
        assert line.startswith("Started a live trading run (dry run), process 4242: tier floors")

    def test_start_run_refuses_without_a_process_starter(self):
        with pytest.raises(RuntimeError, match="not started to run trades"):
            _app()._start_run(settings=_BASE_SETTINGS, dry_run=True, saved_note="")
        assert not config.LIVE_RUNS_DIR.exists()


class TestBlockers:
    """A run is refused before anything is saved while it could not safely start."""

    def test_an_own_run_still_going_refuses_both_before_any_save(self):
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(k="0.75"), "confirm-trade")
        dry = _page_form(app, _query(k="0.75"), "confirm-dry-run")
        _memory_run(app, returncode=None)
        for posted in (form, dry):
            response = _post(app, posted)
            assert response.status == 409
            assert "A run started from this page is still going" in response.body
            assert "Nothing was saved or traded." in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert starter.calls == []
        # Once it has ended, a fresh page runs
        app._runs["fedcba9876543210"].process.returncode = 0
        _run_id(_post(app, _page_form(app, _query(), "confirm-dry-run")))

    def test_a_run_from_before_a_restart_still_going_refuses_both(self):
        # A fresh server finds it by the lock its process holds on output.log
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        folder = _disk_run(output="half way\n")
        starter = _Starter()
        app = _app(start_process=starter)
        with (folder / "output.log").open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            page = _get(app, "/confirm").body
            assert "A run started from this page is still going" in page
            assert "/runs/0123456789abcdef" in _parse(page).links
            trade = _click(_get(app, "/confirm?" + _query(k="0.75")).body, "confirm")
            for action in ("trade", "dry_run"):
                form = [(n, action if n == "action" else v) for n, v in trade]
                response = _post(app, form)
                assert response.status == 409
                assert ("A run started from this page is still going "
                        "(/runs/0123456789abcdef)") in html.unescape(response.body)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert starter.calls == []
        # Once its process has ended (the lock is free), a fresh page runs
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))

    def test_a_finished_or_unstarted_run_from_disk_blocks_nothing(self):
        # A result written, or a start that failed, is not a run still going,
        # whatever holds its output.log
        _save(_BASE_SETTINGS)
        done = _disk_run(result=_result())
        failed = _disk_run("1111111111111111", run_json={"dry_run": False,
                                                          "start_error": "no such file"})
        app = _app(start_process=_Starter())
        with (done / "output.log").open("rb") as a, (failed / "output.log").open("rb") as b:
            fcntl.flock(a.fileno(), fcntl.LOCK_EX)
            fcntl.flock(b.fileno(), fcntl.LOCK_EX)
            assert app._run_blocker(dry_run=True) is None

    def test_a_real_lock_refuses_a_trade_but_not_a_dry_run(self, held_lock):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        body = _get(app, "/confirm?" + _query(k="0.75")).body
        holder = run_lock.holder().describe()
        assert html.escape(f"Another live trading run is in progress ({holder})") in body
        # The trade button is a placeholder, so the action is crafted (it is unsigned)
        form = _click(body, "confirm-dry-run")
        trade = [(n, "trade" if n == "action" else v) for n, v in form]
        response = _post(app, trade)
        assert response.status == 409 and "Another live trading run" in response.body
        assert config.read_saved_live_defaults() == _BASE_SETTINGS
        _run_id(_post(app, form))
        assert len(starter.calls) == 1 and starter.calls[0][0][-1] == "--dry-run"

    def test_a_lock_held_for_over_the_job_timeout_is_called_possibly_hung(self, held_lock):
        _save(_BASE_SETTINGS)
        old = (datetime.now(UTC) - timedelta(
            seconds=config.SCHEDULER_JOB_TIMEOUT_SECONDS + 60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        config.LIVE_RUN_LOCK_FILE.write_text(json.dumps(
            {"pid": 777, "checkout": "/elsewhere", "started_at": old}), encoding="utf-8")
        body = html.unescape(_get(_app(start_process=_Starter()), "/confirm").body)
        assert "it may be hung; check Kalshi before stopping process 777" in body

    def test_a_server_without_a_process_starter_refuses_every_run(self):
        _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        app = _app()
        page = _parse(_get(app, "/confirm?" + _query(k="0.75")).body)
        base = list(page.hidden)
        for action in ("trade", "dry_run"):
            response = _post(app, base + [("action", action)])
            assert response.status == 409
            assert defaults_server._NOT_STARTED_TO_TRADE in response.body
        home = list(_parse(_get(app, "/confirm").body).hidden)
        for action in ("trade", "dry_run"):
            assert _post(app, home + [("action", action)]).status == 409
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert not config.LIVE_RUNS_DIR.exists()

    def test_a_dry_run_needs_saved_defaults(self):
        starter = _Starter()
        app = _app(start_process=starter)
        page = _parse(_get(app, "/confirm?" + _query()).body)
        response = _post(app, list(page.hidden) + [("action", "dry_run")])
        assert response.status == 409 and "save these first" in response.body
        assert starter.calls == [] and config.read_saved_live_defaults() is None

    def test_a_stale_page_with_no_address_fields_is_shown_again(self):
        # The page proposed the defaults then in force; others were saved since.
        # Shown again, it proposes what it showed (now a change), and nothing runs
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        form = _home_form(app)
        other = _save(LiveSettings(tier_floors=False, spread_band=(0.0, 0.5),
                                   interval_discount=0.75, size_cap=0.1))
        response = _post(app, form)
        assert response.status == 409
        assert defaults_server._STALE_BANNER in response.body
        assert other.origin in response.body
        assert starter.calls == []
        _run_id(_post(app, _click(response.body, "confirm-trade")))
        # That Confirm and trade saved the page's proposal before it ran
        assert config.read_saved_live_defaults() == _BASE_SETTINGS

    def test_a_dry_run_with_none_saved_asks_for_a_save_first(self):
        # With none saved, the page with no address fields proposes the seed,
        # and a dry run (a placeholder there) asked for anyway saves nothing
        app = _app(start_process=_Starter())
        page = _parse(_get(app, "/confirm").body)
        assert [b.get("id") for b in page.buttons] == [None, "confirm", "confirm-trade"]
        response = _post(app, list(page.hidden) + [("action", "dry_run")])
        assert response.status == 409 and "save these first" in response.body
        assert app._start_process.calls == [] and config.read_saved_live_defaults() is None


class TestAttention:
    """After a real-money run that needed attention, a trade needs the box ticked."""

    def test_an_exit_20_result_raises_the_banner(self):
        _save(_BASE_SETTINGS)
        _attention_result()
        app = _app(start_process=_Starter())
        for target in ("/confirm?" + _query(), "/confirm"):
            body = _get(app, target).body
            assert ("The last real-money run (<a href=\"/runs/aaaaaaaaaaaaaaaa\">run "
                    "aaaaaaaaaaaaaaaa</a> of 2026-09-29 16:05 UTC) ended needing manual "
                    "attention (exit 20).") in body
            # After a disproof it names every way a new real-money run starts,
            # in brackets after "stop trading"
            assert ("stop trading (a new real-money run would open another wrong-side "
                    "position, so stop the scheduler daemon if it is running, do not run "
                    "main.py --mode prod, and do not press Confirm and trade)") in body
            # What to undo by hand is what the run's CRITICAL names: a market
            # the account held before the run goes back to what it held
            assert ("then undo by hand in the Kalshi UI what its CRITICAL names: close "
                    "out a market the account did not hold before the run, and put one "
                    "it did hold back to what it held.") in body
            assert html.escape(defaults_server._ACK_LABEL) in body

    def test_an_exit_20_scheduled_run_raises_the_banner(self):
        state = {"last_slot": "2026-09-28T09:00:00", "started_at": "2026-09-28T09:00:01",
                 "finished_at": "2026-09-28T09:03:00", "exit_code": 20, "retries": 0}
        (config.PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME).write_text(
            json.dumps(state), encoding="utf-8")
        notice = _app()._attention()
        assert notice is not None
        # A scheduled run has no run page, so the banner names the log its result is in,
        # and after a disproof it names every way a new real-money run starts
        assert notice.text == (
            "The last real-money run (the scheduled run of 2026-09-28 09:03, logged in "
            "kalshi_arb.log) ended needing manual attention (exit 20). Read its result "
            "first. If it reports a V2 order-mapping disproof, stop trading (a new "
            "real-money run would open another wrong-side position, so stop the scheduler "
            "daemon if it is running, do not run main.py --mode prod, and do not press "
            "Confirm and trade), then undo by hand in the Kalshi UI what its CRITICAL "
            "names: close out a market the account did not hold before the run, and put "
            "one it did hold back to what it held.")

    def test_a_sale_left_for_a_person_raises_the_banner(self):
        # A real-money run whose sale left a pair uneven needs attention,
        # whatever exit code its result records
        _disk_run(result=_result(exit_code=0, sales=[
            {"title": "Pair", "status": "unbalanced", "legs": [], "error": "uneven"}]))
        notice = _app()._attention()
        assert notice is not None
        assert "ended needing manual attention (a sale was left for a person to check)" in (
            notice.text)

    def test_a_newer_clean_run_clears_it(self):
        later = "2099-06-01T00:00:00"
        # A scheduled run that needed attention, then a newer clean server run
        state = {"last_slot": "x", "finished_at": "2026-09-28T09:03:00", "exit_code": 20}
        path = config.PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME
        path.write_text(json.dumps(state), encoding="utf-8")
        _disk_run("bbbbbbbbbbbbbbbb", result=_result(finished_at="2099-01-01T00:00:00Z"))
        assert _app()._attention() is None
        # A server run that needed attention, then a newer clean scheduled run
        _attention_result()
        state.update(finished_at=later, exit_code=0)
        path.write_text(json.dumps(state), encoding="utf-8")
        shutil.rmtree(config.LIVE_RUNS_DIR / "20260929T160000Z-bbbbbbbbbbbbbbbb")
        assert _app()._attention() is None

    @pytest.mark.parametrize("older, newer, banner", [(20, 0, False), (0, 20, True)])
    def test_the_newest_of_several_server_runs_decides(self, older, newer, banner):
        # Folders listed newest first by name; the finish time decides, not the order
        _disk_run("1111111111111111", started="20261001T160000Z", result=_result(
            exit_code=older, finished_at="2026-10-01T16:05:00Z"))
        _disk_run("2222222222222222", started="20260901T160000Z", result=_result(
            exit_code=newer, finished_at="2026-10-02T16:05:00Z"))
        notice = _app()._attention()
        assert (notice is not None) is banner
        if banner:
            assert "run 2222222222222222 of 2026-10-02 16:05 UTC" in notice.text

    def test_a_dry_run_neither_raises_nor_clears_it(self):
        _attention_result()
        _disk_run("cccccccccccccccc", started="20260930T160000Z", result=_result(
            dry_run=True, exit_code=0, finished_at="2099-01-01T00:00:00Z"))
        _disk_run("dddddddddddddddd", started="20261001T160000Z", result=_result(
            dry_run=True, exit_code=20, finished_at="2099-02-01T00:00:00Z"))
        assert _app()._attention().text.startswith(
            "The last real-money run (run aaaaaaaaaaaaaaaa of")

    def test_a_trade_needs_the_box_ticked(self):
        _save(_BASE_SETTINGS)
        _attention_result()
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(k="0.75"), "confirm-trade")
        response = _post(app, form)
        assert response.status == 409
        assert defaults_server._ACK_BANNER in html.unescape(response.body)
        assert config.read_saved_live_defaults() == _BASE_SETTINGS
        assert starter.calls == []
        # The page shown again has a fresh nonce, and ticking its box trades
        again = _click(response.body, "confirm-trade", ack=True)
        assert dict(again)["nonce"] != dict(form)["nonce"]
        _run_id(_post(app, again))
        assert config.read_saved_live_defaults().interval_discount == 0.75
        # The page with no address fields (the defaults now in force) asks the same
        app._runs.clear()
        trade = _home_form(app)
        assert _post(app, trade).status == 409
        _run_id(_post(app, _home_form(app, ack=True)))

    def test_a_dry_run_never_needs_it(self):
        _save(_BASE_SETTINGS)
        _attention_result()
        app = _app(start_process=_Starter())
        _run_id(_post(app, _page_form(app, _query(), "confirm-dry-run")))
        app._runs.clear()
        _run_id(_post(app, _home_form(app, "confirm-dry-run")))

    @staticmethod
    def _state(finished_at, exit_code) -> None:
        """
        Write the scheduler's record of its last run.

        Args:
            finished_at: Its finish time, as recorded (naive local time).
            exit_code: Its exit code, as recorded.
        """
        state = {"last_slot": "2026-09-28T09:00:00", "started_at": "2026-09-28T09:00:01",
                 "finished_at": finished_at, "exit_code": exit_code, "retries": 0}
        (config.PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME).write_text(
            json.dumps(state), encoding="utf-8")

    @pytest.mark.parametrize("state_text", [
        b"not json", b"[]", b'{"finished_at": 5, "exit_code": 20}',
        b'{"finished_at": "yesterday", "exit_code": 20}',
        b'{"finished_at": null, "exit_code": null}',
        b'{"finished_at": "9999-12-31T23:59:59-12:00", "exit_code": 20}'])
    def test_a_state_file_with_no_readable_finish_raises_no_banner_and_no_error(
            self, state_text):
        (config.PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME).write_bytes(state_text)
        _disk_run("eeeeeeeeeeeeeeee", dry_run=True, result=b"\x00garbage")
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        assert app._attention() is None
        response = _get(app, "/confirm")
        assert response.status == 200 and "The last real-money run" not in response.body

    def test_a_state_file_that_cannot_be_read_keeps_a_server_run_s_banner(self):
        # A finish time whose UTC instant is out of range is left out; it
        # never takes the exit-20 run's warning away with it
        _attention_result()
        (config.PROJECT_ROOT / config.SCHEDULER_STATE_FILENAME).write_text(
            '{"finished_at": "9999-12-31T23:59:59-12:00", "exit_code": 0}', encoding="utf-8")
        notice = _app()._attention()
        assert notice is not None and "ended needing manual attention (exit 20)" in notice.text

    @pytest.mark.parametrize("exit_code", [2, 10, 30, 50])
    def test_a_newer_scheduled_run_that_sent_no_order_leaves_it_in_place(self, exit_code):
        _attention_result()
        self._state("2099-01-01T09:03:00", exit_code)
        notice = _app()._attention()
        assert notice is not None
        assert notice.text.startswith("The last real-money run (run aaaaaaaaaaaaaaaa of")
        assert "ended needing manual attention (exit 20)" in notice.text

    @pytest.mark.parametrize("changes", [
        {"exit_code": 10}, {"exit_code": 30}, {"exit_code": 50},
        # An error before its first order: nothing was sent
        {"exit_code": None, "error": "ApiException: HTTP 401", "submission_started": False}])
    def test_a_newer_server_run_that_sent_no_order_leaves_it_in_place(self, changes):
        _attention_result()
        _disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                  result=_result(finished_at="2099-01-01T16:05:00Z", **changes))
        assert "run aaaaaaaaaaaaaaaa of" in _app()._attention().text

    def test_a_newer_run_its_parser_refused_leaves_it_in_place(self):
        _attention_result()
        _printed_at(_disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                              output=TestRunPage._USAGE), "2099-01-01T16:00:01+00:00")
        assert "run aaaaaaaaaaaaaaaa of" in _app()._attention().text

    def test_a_newer_run_that_could_not_be_started_leaves_it_in_place(self):
        _attention_result()
        _printed_at(_disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                              run_json={"dry_run": False,
                                        "start_error": "No such file or directory"}),
                    "2099-01-01T16:00:01+00:00")
        assert "run aaaaaaaaaaaaaaaa of" in _app()._attention().text

    def test_a_newer_run_still_going_leaves_it_in_place(self):
        _attention_result()
        folder = _printed_at(_disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z"),
                             "2099-01-01T16:00:01+00:00")
        with (folder / "output.log").open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            assert "run aaaaaaaaaaaaaaaa of" in _app()._attention().text
        # Once it has ended without a result it decides, and says so
        assert "run bbbbbbbbbbbbbbbb of 2099-01-01 16:00 UTC" in _app()._attention().text

    @pytest.mark.parametrize("exit_code, why", [
        (None, "the scheduler recorded no exit code: it was stopped at its time limit, "
               "could not be started, or its record is damaged"),
        ("20", "the scheduler recorded no exit code"),
        (-9, "stopped by a signal (exit -9)"),
        (1, "exit 1")])
    def test_a_scheduled_run_without_a_clean_result_raises_it(self, exit_code, why):
        # Newer than a clean server run, so it decides
        _disk_run("bbbbbbbbbbbbbbbb", result=_result())
        self._state("2099-01-01T09:03:00", exit_code)
        notice = _app()._attention()
        assert notice is not None
        assert notice.text.startswith(
            "The last real-money run (the scheduled run of 2099-01-01 09:03, logged in "
            "kalshi_arb.log) ended without a clean result (")
        assert why in notice.text
        assert defaults_server._CHECK_POSITIONS in notice.text

    @pytest.mark.parametrize("changes, why", [
        ({"exit_code": None, "error": "ReadTimeoutError: read timed out",
          "submission_started": True}, "it stopped while sending orders"),
        ({"exit_code": None, "error": "OSError: disk full", "submission_started": True,
          "trades": [_trade("executed")]},
         "an error stopped it after it began sending orders")])
    def test_a_server_run_that_stopped_after_sending_raises_it(self, changes, why):
        self._state("2026-09-28T09:03:00", 0)
        _disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                  result=_result(finished_at="2099-01-01T16:05:00Z", **changes))
        notice = _app()._attention()
        assert notice is not None
        assert notice.text == (
            "The last real-money run (run bbbbbbbbbbbbbbbb of 2099-01-01 16:05 UTC) ended "
            f"without a clean result ({why}). {defaults_server._CHECK_POSITIONS}")

    @pytest.mark.parametrize("result, why", [
        (None, "it wrote no result"), (b"\x00garbage", "its result could not be read")])
    def test_a_server_run_without_a_readable_result_raises_it(self, result, why):
        _disk_run("bbbbbbbbbbbbbbbb", result=result, output="Submitting V2 order\n")
        notice = _app()._attention()
        assert notice is not None and f"without a clean result ({why})" in notice.text
        # A dry run in the same state says nothing about the account
        shutil.rmtree(config.LIVE_RUNS_DIR)
        _disk_run("bbbbbbbbbbbbbbbb", result=result, dry_run=True)
        assert _app()._attention() is None

    @pytest.mark.parametrize("returncode, text", [
        (config.EXIT_TRADES_NEED_ATTENTION, "ended needing manual attention (exit 20)"),
        (-15, "ended without a clean result (it wrote no result, exit -15)")])
    def test_a_run_this_server_started_is_judged_by_its_exit_code(self, returncode, text):
        app = _app()
        _memory_run(app, returncode=returncode)
        assert text in app._attention().text
        # The same folder read after a restart has no exit code to go by
        assert "ended without a clean result (it wrote no result)" in _app()._attention().text

    def test_an_unclean_run_needs_the_box_ticked_and_a_newer_clean_run_clears_it(self):
        _save(_BASE_SETTINGS)
        self._state("2026-09-28T09:03:00", None)
        starter = _Starter()
        app = _app(start_process=starter)
        assert _post(app, _home_form(app)).status == 409
        assert starter.calls == []
        _disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                  result=_result(finished_at="2099-01-01T16:05:00Z"))
        assert app._attention() is None
        _run_id(_post(app, _home_form(app)))

    def test_a_failure_to_read_the_records_fails_closed(self, monkeypatch, caplog):
        def broken():
            raise RuntimeError("boom")

        monkeypatch.setattr(defaults_server, "_scheduled_last_run", broken)
        with caplog.at_level(logging.ERROR):
            notice = _app()._attention()
        assert notice is not None and notice.text == defaults_server._OUTCOME_UNREADABLE
        assert "Could not read the last real-money run's outcome" in caplog.text


class TestRunPage:
    """GET /runs/<id>: a run's progress while it runs, and its outcome once it has ended."""

    @staticmethod
    def _headline(body: str) -> str:
        """
        Read a finished run page's headline.

        Args:
            body (str): The page.

        Returns:
            str: The text of its first bold paragraph.
        """
        return html.unescape(re.search(r"<b>(.*?)</b>", body).group(1))

    def test_a_run_still_going(self):
        app = _app()
        run_id = _memory_run(app, returncode=None, output="Scanning markets\n")
        body = _get(app, f"/runs/{run_id}").body
        assert (f'<meta http-equiv="refresh" content="'
                f'{config.DEFAULTS_SERVER_RUN_REFRESH_SECONDS}">') in body
        assert "Trading run in progress — real orders on the production account" in body
        assert "Closing this tab or stopping the server does not stop the run. Keep this " \
               "Mac awake until it finishes." in body
        assert "started before this server restarted" not in body
        assert "Scanning markets" in body and "Settings: tier floors off" in body
        assert "Saved and verified." in body
        assert "may be hung" not in body

    def test_a_dry_run_still_going(self):
        app = _app()
        run_id = _memory_run(app, returncode=None, dry_run=True)
        assert "Dry run in progress — no orders" in _get(app, f"/runs/{run_id}").body

    def test_a_run_going_for_over_the_job_timeout_may_be_hung(self):
        app = _app()
        run_id = _memory_run(app, returncode=None, started_at=datetime.now(UTC) - timedelta(
            seconds=config.SCHEDULER_JOB_TIMEOUT_SECONDS + 5))
        body = _get(app, f"/runs/{run_id}").body
        assert "it may be hung; check Kalshi before stopping process 4242" in body

    def test_a_run_from_disk_whose_output_is_locked_is_still_going(self):
        folder = _disk_run(output="half way\n")
        with (folder / "output.log").open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            body = _get(_app(), "/runs/0123456789abcdef").body
        assert '<meta http-equiv="refresh"' in body
        assert "Trading run in progress" in body
        assert "It was started before this server restarted." in body
        assert "process 555" in body

    def test_a_dead_real_money_run_without_a_result_says_to_check_positions(self):
        _disk_run(output="2026-09-29 INFO Submitting V2 order\n")
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "The run ended without writing a result"
        assert html.escape(defaults_server._CHECK_POSITIONS) in body
        assert "refresh" not in body

    def test_a_dead_dry_run_without_a_result_sent_no_orders(self):
        _disk_run(dry_run=True)
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "The run ended without writing a result"
        assert "It was a dry run: no orders were sent." in body
        assert html.escape(defaults_server._CHECK_POSITIONS) not in body

    def test_a_signal_stops_a_run(self):
        app = _app()
        run_id = _memory_run(app, returncode=-15)
        body = _get(app, f"/runs/{run_id}").body
        assert self._headline(body) == "The run ended without writing a result"
        assert "Exit code: -15 (stopped by a signal)." in body
        assert html.escape(defaults_server._CHECK_POSITIONS) in body

    _USAGE = ("usage: main.py [-h] [--mode {dev,prod}]\n"
              "main.py: error: No live defaults are saved in /x/live_defaults.json\n")

    def test_a_run_its_parser_refused(self):
        app = _app()
        run_id = _memory_run(app, returncode=2, output=self._USAGE)
        body = _get(app, f"/runs/{run_id}").body
        assert self._headline(body) == "The run refused to start"
        assert "main.py: error: No live defaults are saved" in body
        assert html.escape(defaults_server._CHECK_POSITIONS) not in body

    def test_a_run_from_disk_its_parser_refused(self):
        _disk_run(output=self._USAGE)
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "The run refused to start"

    def test_an_error_line_without_usage_is_not_a_refusal(self):
        _disk_run(output="2026-09-29 INFO x: error: y\n")
        assert self._headline(_get(_app(), "/runs/0123456789abcdef").body) == \
            "The run ended without writing a result"

    @pytest.mark.parametrize("raw", [
        b"not json", b"[]", b"{}",
        json.dumps(_result(format="other")).encode(),
        json.dumps(_result(dry_run="no")).encode(),
        json.dumps(_result(exit_code=True)).encode(),
        json.dumps(_result(exit_code="0")).encode()])
    def test_an_unreadable_result_is_said_so(self, raw):
        _disk_run(result=raw)
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "The run's result could not be read"

    def test_badly_typed_fields_are_left_out_not_fatal(self):
        _disk_run(result=_result(
            message=5, warnings=["WARNING: kept", 7], warnings_dropped="x", balance_before=None,
            trades=[_trade("executed", a="junk", b={"ticker": 5, "count": "30",
                                                     "price": float("nan")}), "junk"]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "Trades completed"
        assert "WARNING: kept" in body
        assert "<td>? × ? @ ?</td>" in html.unescape(body)

    # Each rule of the table, on a run read from disk: (result, headline, text on the page)
    _RULES = [
        (_result(exit_code=20, trades=[_trade("executed"), _trade("manual_review")]),
         "Trades or sales need your attention", "Do not start another real-money run"),
        (_result(exit_code=None, submission_started=True, error="KeyboardInterrupt: "),
         "The run stopped while sending orders", "Orders may have been placed"),
        (_result(exit_code=None, trades=[_trade("executed")],
                 error="HTTP 500 Internal Server Error"),
         "Orders were placed, then the run stopped with an error",
         "HTTP 500 Internal Server Error"),
        (_result(exit_code=0, trades=[_trade("executed"), _trade("failed", error="killed")],
                 message="Submitted 1 of 2 trades"),
         "Trades completed", "1 of 2 pairs placed."),
        (_result(exit_code=0, dry_run=True, trades=[_trade("simulated")]),
         "Dry run finished — no orders were sent", "These trades would have been placed:"),
        (_result(exit_code=0, trades=[_trade("failed"), _trade("rolled_back")]),
         "No trade completed", "Each pair's status and error are listed below."),
        (_result(exit_code=0, message="No qualifying pairs found"),
         "No trades to complete", "No qualifying pairs found"),
        (_result(exit_code=40, message="No qualifying pairs found"),
         "No trades to complete", "No time-series pair was searched"),
        (_result(exit_code=40, trades=[_trade("executed")]),
         "Trades completed", "No time-series pair was searched"),
        (_result(exit_code=10, message="Portfolio value $10.00 (cash $10.00) is below minimum "
                                       "$50.00 — skipping run."),
         "Not traded: the portfolio value is below the minimum",
         "Portfolio value $10.00 (cash $10.00) is below minimum"),
        (_result(exit_code=30, message="Every shard is closed"),
         "Not traded: nothing could be scanned (every exchange shard was closed, or no market "
         "was read)", "Every shard is closed"),
        (_result(exit_code=50, message="Another live trading run is in progress"),
         "Not traded: another live trading run was in progress",
         "Another live trading run is in progress"),
        (_result(exit_code=2), "The run refused to start", "No reason was printed."),
        (_result(exit_code=None, error="RuntimeError: secrets.json not found"),
         "The run stopped with an error", "RuntimeError: secrets.json not found"),
        (_result(exit_code=1), "The run stopped with an error", "No error was recorded"),
    ]

    @pytest.mark.parametrize("record, headline, text", _RULES)
    def test_each_rule_of_the_table(self, record, headline, text):
        _disk_run(result=record)
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == headline
        assert html.escape(text) in body

    def test_attention_rows_come_first(self):
        _disk_run(result=_result(exit_code=20, trades=[
            _trade("executed"), _trade("failed"), _trade("rollback_failed", error="orphan")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        order = [body.index(f"<h2>{label}</h2>")
                 for label in ("Needs attention", "Completed", "Not completed")]
        assert order == sorted(order)
        assert "<th>Status</th><th>Type</th><th>Category</th><th>Market A</th><th>Leg A</th>" \
               "<th>Market B</th><th>Leg B</th><th>Cost incl. fees</th><th>Profit if won</th>" \
               "<th>Note</th>" in body
        assert "<td>30 × YES @ 0.2100</td>" in body and "<td>30 × NO @ 0.5000</td>" in body
        assert "<td>Rain by Oct 1 (RAIN-A)</td>" in body
        assert "<td>$21.90</td>" in body and "<td>orphan</td>" in body

    @staticmethod
    def _category_cell(body: str, status: str) -> str:
        """
        Read the Category cell of a pair's row: the cell right after its Type cell.

        Args:
            body (str): The run page.
            status (str): The pair's status, the first cell of its row.

        Returns:
            str: The cell's HTML text, as written on the page.
        """
        found = re.search(rf"<td>{status}</td><td>time_series</td><td>(.*?)</td>"
                          r"<td>Rain by Oct 1 \(RAIN-A\)</td>", body)
        assert found, "the row has no Category cell right after its Type cell"
        return found.group(1)

    def test_each_pair_shows_its_category_and_tag(self):
        _disk_run(result=_result(exit_code=20, trades=[
            _trade("executed", category="Sports", tag="Basketball"),
            _trade("rollback_failed", error="orphan", category="Economics", tag="Fed"),
            _trade("failed", error="killed", category="Politics", tag="Elections")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        for status, text in (("executed", "Sports · Basketball"),
                             ("rollback_failed", "Economics · Fed"),
                             ("failed", "Politics · Elections")):
            assert self._category_cell(body, status) == html.escape(text)
        # Each shows in the table of its own group
        sections = {part.split("</h2>")[0]: part for part in body.split("<h2>")[1:]}
        assert "<td>Sports · Basketball</td>" in sections["Completed"]
        assert "<td>Economics · Fed</td>" in sections["Needs attention"]
        assert "<td>Politics · Elections</td>" in sections["Not completed"]

    def test_a_dry_runs_would_be_trades_show_their_category(self):
        _disk_run(result=_result(exit_code=0, dry_run=True, trades=[
            _trade("simulated", category="Sports", tag="Basketball")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        section = body.split("<h2>Would have traded</h2>")[1]
        assert self._category_cell(section, "simulated") == "Sports · Basketball"

    def test_a_category_with_no_tag_is_shown_alone(self):
        _disk_run(result=_result(exit_code=0, trades=[_trade("executed", category="Sports")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._category_cell(body, "executed") == "Sports"
        assert "Sports ·" not in body

    @pytest.mark.parametrize("changes", [
        {},  # a run from before the column: the keys are not in the result at all
        {"category": None, "tag": None},  # the run could not read Kalshi's /series listing
        {"category": "", "tag": "Basketball"},
        {"category": None, "tag": "Basketball"},  # a tag means nothing without its category
    ])
    def test_a_pair_with_no_recorded_category_shows_a_dash(self, changes):
        _disk_run(result=_result(exit_code=0, trades=[_trade("executed", **changes)]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        # The dash is the Category cell: the Type cell before it says time_series
        assert self._category_cell(body, "executed") == "—"
        assert "Basketball" not in body

    @pytest.mark.parametrize("value", [5, True, 1.5, ["Sports"], {"name": "Sports"}])
    def test_a_badly_typed_category_shows_a_dash(self, value):
        _disk_run(result=_result(exit_code=0, trades=[
            _trade("executed", category=value, tag="Basketball")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "Trades completed"
        assert self._category_cell(body, "executed") == "—"

    @pytest.mark.parametrize("value", [5, True, 1.5, ["Basketball"], {"name": "Basketball"}])
    def test_a_badly_typed_tag_is_left_out(self, value):
        _disk_run(result=_result(exit_code=0, trades=[
            _trade("executed", category="Sports", tag=value)]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "Trades completed"
        assert self._category_cell(body, "executed") == "Sports"

    def test_html_in_a_category_or_tag_is_escaped(self):
        _disk_run(result=_result(exit_code=0, trades=[
            _trade("executed", category="<b>x</b>", tag="<i>y</i>")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._category_cell(body, "executed") == \
            "&lt;b&gt;x&lt;/b&gt; · &lt;i&gt;y&lt;/i&gt;"
        assert "<b>x</b>" not in body and "<i>y</i>" not in body

    def test_a_trade_that_added_to_a_held_pair_says_so_in_its_note(self):
        # The run result's adds_to_held shows in the Note cell, after any
        # error; a trade without it keeps its Note as it was
        _disk_run(result=_result(exit_code=0, trades=[
            _trade("executed", adds_to_held=30.0),
            _trade("rolled_back", error="YES leg FoK not filled", adds_to_held=30.0),
            _trade("failed", error="killed")]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert "<td>(adds to 30 held)</td>" in body
        assert "<td>YES leg FoK not filled (adds to 30 held)</td>" in body
        assert "<td>killed</td>" in body

    @pytest.mark.parametrize("value", [True, "30", 0, -5, None])
    def test_an_unreadable_held_count_adds_no_note(self, value):
        # Only a number above zero is shown; anything else is left out
        _disk_run(result=_result(exit_code=0, trades=[_trade("executed", adds_to_held=value)]))
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert "adds to" not in body

    def test_the_exit_code_a_process_returned_wins_over_its_result(self):
        app = _app()
        run_id = _memory_run(app, returncode=20, result=_result(exit_code=0))
        assert self._headline(_get(app, f"/runs/{run_id}").body) == \
            "Trades or sales need your attention"

    def test_a_runs_sales_are_shown_before_its_trades(self):
        # The Sales table: each position, its status in plain words, what
        # sold on each market and its profit (only for a position that sold
        # in full); the cash the sales left under it
        legs = [{"ticker": "SELL-A", "side": "yes", "held": 30, "sold": 30, "price": 0.55},
                {"ticker": "SELL-B", "side": "no", "held": 30, "sold": 30, "price": 0.42}]
        sale = {"title": "Will S happen by Nov 1, 2026?", "status": "sold", "level": 0.8,
                "days_left": 9, "cost": 18.8, "proceeds": 28.06, "profit": 9.26,
                "realized_percent": 82.68, "legs": legs, "error": None}
        partly = dict(sale, title="Partly", status="partly_sold", error="sold 10 of 30",
                      legs=[dict(legs[0], sold=10), dict(legs[1], sold=None)])
        _disk_run(result=_result(exit_code=0, sales=[sale, partly], cash_after_sales=128.06,
                                 trades=[_trade("executed")]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert body.index("<h2>Sales</h2>") < body.index("<h2>Completed</h2>")
        assert "<th>Position</th><th>Status</th><th>Sold</th><th>Profit</th>" in body
        assert ("<td>sold</td><td>30 of 30 YES on SELL-A, 30 of 30 NO on SELL-B</td>"
                "<td>$9.26</td>") in body
        assert ("<td>partly sold</td><td>10 of 30 YES on SELL-A, ? of 30 NO on SELL-B "
                "(sold 10 of 30)</td><td>—</td>") in body
        assert "Cash after the sales $128.06" in body
        assert self._headline(body) == "Trades completed"

    @pytest.mark.parametrize("status, word", [("unbalanced", "unbalanced"),
                                              ("manual_review", "check")])
    def test_a_sale_left_for_a_person_needs_attention(self, status, word):
        # Even when the recorded exit code says nothing of it
        sale = {"title": "Pair", "status": status, "legs": [], "error": "see the log"}
        _disk_run(result=_result(exit_code=0, sales=[sale]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Trades or sales need your attention"
        assert f"<td>Pair</td><td>{word}</td><td>— (see the log)</td><td>—</td>" in body

    def test_badly_typed_sales_are_left_out_not_fatal(self):
        _disk_run(result=_result(exit_code=0, sales=[
            "junk", {"status": 5, "legs": ["junk", {"ticker": 7, "held": "30"}],
                     "profit": float("nan")}]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "<td>unknown</td><td>? of ? ? on ?</td><td>—</td>" in body

    # One position sold in full, as a run result records it
    _SOLD = {"title": "Pair", "status": "sold", "level": 0.8, "days_left": 9, "cost": 18.8,
             "proceeds": 28.06, "profit": 9.26, "realized_percent": 82.68, "error": None,
             "legs": [{"ticker": "SELL-A", "side": "yes", "held": 30, "sold": 30,
                       "price": 0.55}]}

    def test_a_run_that_only_sold_says_so(self):
        # A live run that sold and then bought nothing: the headline names
        # the sale, and the portfolio value read at the start is not what
        # its buys were sized on, so the line says it came before the sales
        _disk_run(result=_result(exit_code=0, sales=[self._SOLD], cash_after_sales=128.06,
                                 balance_before=100.0, portfolio_value_before=150.0))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Positions sold; no trades to complete"
        assert "Portfolio value $150.00 (cash $100.00) — before the sales" in body
        assert "what Kelly sizes on" not in body
        assert "<p>Cash after the sales $128.06</p>" in body

    def test_a_dry_run_that_would_only_have_sold_says_so(self):
        # Its cash after the sales is an estimate: nothing was sent
        sale = dict(self._SOLD, status="simulated")
        _disk_run(dry_run=True, result=_result(dry_run=True, exit_code=0, sales=[sale],
                                               cash_after_sales=128.06))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Dry run finished — no orders were sent"
        assert "These positions would have been sold:" in body
        assert "Cash after the sales $128.06 (estimated: nothing was sent)" in body

    def test_a_sale_that_sold_nothing_is_no_trade_to_complete(self):
        sale = dict(self._SOLD, status="not_sold", profit=None, error="filled nothing",
                    legs=[dict(self._SOLD["legs"][0], sold=0)])
        _disk_run(result=_result(exit_code=0, sales=[sale]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "No trades to complete"
        assert "<td>not sold</td><td>0 of 30 YES on SELL-A (filled nothing)</td>" in body

    # One Kelly trim in a run result: 13 of the 30 contracts held on each
    # market sold (a trim has no sell level)
    _TRIMMED = {"title": "Trimmed pair", "status": "sold", "kind": "kelly_trim", "level": None,
                "days_left": 9, "cost": 8.15, "proceeds": 12.15, "profit": 4.0,
                "realized_percent": None, "error": None,
                "legs": [{"ticker": "TRIM-A", "side": "yes", "held": 30, "sold": 13,
                          "price": 0.55},
                         {"ticker": "TRIM-B", "side": "no", "held": 30, "sold": 13,
                          "price": 0.42}]}

    def test_a_kelly_trim_is_marked_and_shows_what_sold_of_what_was_held(self):
        # Beside a take-profit sale, whose row carries no trim mark
        _disk_run(result=_result(exit_code=0, sales=[self._SOLD, self._TRIMMED],
                                 trades=[_trade("executed")]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert ("<td>Trimmed pair (Kelly trim)</td><td>sold</td>"
                "<td>13 of 30 YES on TRIM-A, 13 of 30 NO on TRIM-B</td><td>$4.00</td>") in body
        assert ("<td>Pair</td><td>sold</td><td>30 of 30 YES on SELL-A</td>"
                "<td>$9.26</td>") in body
        assert body.count("(Kelly trim)") == 1
        # The note under the table covers both: a trim's profit is on what it sold
        assert defaults_server._SALE_PROFIT_NOTE in body
        assert ("A Kelly trim sells a held pair, or a lone leg whose partner has paid out, "
                "down to its Kelly size: part of it, or all of it when Kelly keeps none.") in body
        assert "for a Kelly trim, less what the part sold cost" in body

    def test_the_profit_note_is_true_of_a_lone_market_sold_with_its_paid_out_partner(self):
        # A lone held market's profit is its proceeds plus its paid-out partner's
        # payout, less what both markets cost (seller._plan_one, reporter.sale_record).
        # The partner was never sold, so the note cannot say "what the contracts sold cost"
        note = defaults_server._SALE_PROFIT_NOTE
        assert "less what the contracts sold cost" not in note
        assert "plus the payout of a partner that has already paid out" in note
        assert "less what the whole position cost" in note
        # A trim's profit is still on the part it sold
        assert "for a Kelly trim, less what the part sold cost" in note
        # A row the note sits under: 30 YES sold return $28.71 after the fee, the
        # partner paid $0, both markets cost $18.80, so the profit is $9.91
        lone = {"title": "Lone", "status": "sold", "level": 0.8, "days_left": 9, "cost": 18.8,
                "proceeds": 28.71, "profit": 9.91, "realized_percent": 88.48, "error": None,
                "legs": [{"ticker": "SELL-A", "side": "yes", "held": 30, "sold": 30,
                          "price": 0.96}]}
        _disk_run(result=_result(exit_code=0, sales=[lone], trades=[_trade("executed")]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "<td>Lone</td><td>sold</td><td>30 of 30 YES on SELL-A</td><td>$9.91</td>" in body
        assert note in body

    def test_a_trim_that_sells_a_whole_pair_is_not_said_to_keep_the_rest(self):
        # The rule sells a pair whole when Kelly keeps none of it (30 of 30)
        whole = dict(self._TRIMMED, status="simulated",
                     legs=[dict(leg, sold=30) for leg in self._TRIMMED["legs"]])
        _disk_run(dry_run=True, result=_result(dry_run=True, exit_code=0, sales=[whole]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "30 of 30 YES on TRIM-A, 30 of 30 NO on TRIM-B" in body
        assert "Part of each of these held pairs would have been sold" not in body
        assert "keeps the rest" not in body
        assert ("These held positions would have been sold down to their Kelly size (a Kelly "
                "trim):") in body

    def test_a_trim_recorded_by_the_reporter_is_read_as_one(self):
        # The kind the reporter writes is the kind this page reads
        reporter = importlib.import_module("kalshi_betting.reporter")
        record = reporter.SaleRecord(
            title="Trimmed pair", status="simulated", level=None, days_left=9, cost=8.15,
            proceeds=12.15, profit=4.0, realized_percent=None, error=None,
            legs=(reporter.SaleLegRecord("TRIM-A", "yes", 30, 13, 0.55),),
            kind=config.SALE_KIND_KELLY_TRIM)
        sale = json.loads(json.dumps(dataclasses.asdict(record)))
        _disk_run(dry_run=True, result=_result(dry_run=True, exit_code=0, sales=[sale]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert ("<td>Trimmed pair (Kelly trim)</td><td>simulated</td>"
                "<td>13 of 30 YES on TRIM-A</td><td>$4.00</td>") in body

    @pytest.mark.parametrize("kind", ["take_profit", None, 5, "KELLY_TRIM", ""])
    def test_only_a_kelly_trim_is_marked(self, kind):
        # A missing or unknown kind reads as a take-profit sale
        _disk_run(result=_result(exit_code=0, sales=[dict(self._TRIMMED, kind=kind)]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "(Kelly trim)" not in body
        assert "<td>Trimmed pair</td><td>sold</td>" in body
        assert self._headline(body) == "Positions sold; no trades to complete"
        no_kind = {key: value for key, value in self._TRIMMED.items() if key != "kind"}
        _disk_run("00000000000000aa", result=_result(exit_code=0, sales=[no_kind]))
        assert "(Kelly trim)" not in _get(_app(), "/runs/00000000000000aa").body

    # One Kelly trim of a lone leg in a run result: 43 of the 129 NO held on LONE-B sold;
    # its paid-out partner is no leg, since it sold nothing
    _LONE_TRIMMED = {"title": "Lone trimmed", "status": "sold", "kind": "kelly_trim",
                     "level": None, "days_left": 9, "cost": 20.1, "proceeds": 36.61,
                     "profit": 16.51, "realized_percent": None, "error": None,
                     "legs": [{"ticker": "LONE-B", "side": "no", "held": 129, "sold": 43,
                               "price": 0.86}]}

    def test_a_lone_leg_trim_shows_what_sold_of_what_was_held(self):
        _disk_run(result=_result(exit_code=0, sales=[self._LONE_TRIMMED]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert ("<td>Lone trimmed (Kelly trim)</td><td>sold</td>"
                "<td>43 of 129 NO on LONE-B</td><td>$16.51</td>") in body
        # The note under the table covers a lone leg too
        assert "a lone leg whose partner has paid out, down to its Kelly size" in body

    def test_a_run_that_only_trimmed_lone_legs_says_held_positions_were_trimmed(self):
        _disk_run(result=_result(exit_code=0, sales=[self._LONE_TRIMMED],
                                 cash_after_sales=130.61))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Held positions trimmed; no trades to complete"
        # Beside a pair's trim: still only trims
        _disk_run("00000000000000dd", result=_result(
            exit_code=0, sales=[self._LONE_TRIMMED, self._TRIMMED]))
        assert self._headline(_get(_app(), "/runs/00000000000000dd").body) == (
            "Held positions trimmed; no trades to complete")

    def test_a_dry_run_that_would_only_have_trimmed_a_lone_leg_says_so(self):
        trim = dict(self._LONE_TRIMMED, status="simulated")
        _disk_run(dry_run=True, result=_result(dry_run=True, exit_code=0, sales=[trim],
                                               cash_after_sales=130.61))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Dry run finished — no orders were sent"
        assert ("These held positions would have been sold down to their Kelly size (a Kelly "
                "trim):") in body
        assert "<td>43 of 129 NO on LONE-B</td>" in body

    @pytest.mark.parametrize("status", ["sold", "partly_sold"])
    def test_a_run_that_only_trimmed_says_so(self, status):
        # A trim usually sells only part of a pair, so the headline says
        # "trimmed", not "positions sold"
        trim = dict(self._TRIMMED, status=status)
        _disk_run(result=_result(exit_code=0, sales=[trim], cash_after_sales=112.15,
                                 balance_before=100.0, portfolio_value_before=150.0))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Held positions trimmed; no trades to complete"
        assert "Portfolio value $150.00 (cash $100.00) — before the sales" in body

    def test_a_run_that_sold_a_position_and_trimmed_a_pair_says_positions_sold(self):
        _disk_run(result=_result(exit_code=0, sales=[self._TRIMMED, self._SOLD]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Positions sold; no trades to complete"
        # A trim that filled nothing beside a take-profit sale that did: the same
        _disk_run("00000000000000bb", result=_result(exit_code=0, sales=[
            dict(self._TRIMMED, status="not_sold", profit=None), self._SOLD]))
        assert self._headline(_get(_app(), "/runs/00000000000000bb").body) == \
            "Positions sold; no trades to complete"

    def test_a_dry_run_that_would_only_have_trimmed_says_so(self):
        trim = dict(self._TRIMMED, status="simulated")
        _disk_run(dry_run=True, result=_result(dry_run=True, exit_code=0, sales=[trim],
                                               cash_after_sales=112.15))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Dry run finished — no orders were sent"
        assert ("These held positions would have been sold down to their Kelly size (a Kelly "
                "trim):") in body
        assert "These positions would have been sold:" not in body
        # With a take-profit sale beside it, the positions line stands
        sold = dict(self._SOLD, status="simulated")
        _disk_run("00000000000000cc", dry_run=True,
                  result=_result(dry_run=True, exit_code=0, sales=[trim, sold]))
        body = html.unescape(_get(_app(), "/runs/00000000000000cc").body)
        assert "These positions would have been sold:" in body
        assert "would have been sold down to their Kelly size" not in body

    def test_a_trim_left_for_a_person_needs_attention(self):
        trim = dict(self._TRIMMED, status="unbalanced", profit=None, error="uneven",
                    legs=[dict(self._TRIMMED["legs"][0], sold=13),
                          dict(self._TRIMMED["legs"][1], sold=9)])
        _disk_run(result=_result(exit_code=0, sales=[trim]))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert self._headline(body) == "Trades or sales need your attention"
        assert ("<td>Trimmed pair (Kelly trim)</td><td>unbalanced</td>"
                "<td>13 of 30 YES on TRIM-A, 9 of 30 NO on TRIM-B (uneven)</td><td>—</td>"
                ) in body

    def test_every_exit_code_has_a_rule_of_its_own(self):
        codes = {name: getattr(config, name) for name in dir(config)
                 if name.startswith("EXIT_")}
        assert set(codes.values()) == {0, 10, 20, 30, 40, 50}
        for number, (name, code) in enumerate(sorted(codes.items())):
            run_id = f"{number + 1:016x}"
            _disk_run(run_id, result=_result(exit_code=code))
            headline = self._headline(_get(_app(), f"/runs/{run_id}").body)
            assert headline != "The run stopped with an error", name

    def test_balances_and_warnings(self):
        _disk_run(result=_result(balance_before=1000.5, balance_after=990.25,
                                 warnings=["WARNING: one", "CRITICAL: two"],
                                 warnings_dropped=3))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "Balance before $1,000.50 → after $990.25" in body
        assert "<li>WARNING: one</li><li>CRITICAL: two</li>" in body
        assert "3 more warning lines were not kept." in body
        assert "Finished 2026-09-29 16:05:00 UTC, exit code 0." in body

    def test_the_portfolio_value_kelly_sizes_on_is_shown_with_its_cash(self):
        _disk_run(result=_result(balance_before=1000.5, balance_after=990.25,
                                 portfolio_value_before=1850.75))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "Portfolio value $1,850.75 (cash $1,000.50) — what Kelly sizes on" in body
        # Beside the cash before and after, not in place of it
        assert "Balance before $1,000.50 → after $990.25" in body

    def test_a_skipped_run_still_shows_the_portfolio_value_it_gated_on(self):
        # A run stopped by the minimum never reads the balance after trading,
        # but it read the one it gated on; the line claims no sizing took place
        _disk_run(result=_result(exit_code=10, balance_before=10.0, balance_after=None,
                                 portfolio_value_before=40.0))
        body = html.unescape(_get(_app(), "/runs/0123456789abcdef").body)
        assert "Portfolio value $40.00 (cash $10.00) — what Kelly sizes on" in body
        assert "Sized on" not in body
        assert "Balance before" not in body

    @pytest.mark.parametrize("value", [None, "1850.75", True, float("nan")],
                             ids=["null", "text", "bool", "nan"])
    def test_a_portfolio_value_that_is_not_a_number_is_left_out(self, value):
        # A result without the key (a run started before it existed) or with
        # a damaged one shows no line rather than a wrong one
        _disk_run(result=_result(portfolio_value_before=value))
        body = _get(_app(), "/runs/0123456789abcdef").body
        # The rest of the record is still read
        assert self._headline(body) == "No trades to complete"
        assert "what Kelly sizes on" not in body

    def test_a_result_without_the_key_is_still_read(self):
        record = _result()
        del record["portfolio_value_before"]
        _disk_run(result=record)
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert self._headline(body) == "No trades to complete"
        assert "what Kelly sizes on" not in body

    def test_one_balance_alone_is_not_shown(self):
        _disk_run(result=_result(balance_after=None))
        assert "Balance before" not in _get(_app(), "/runs/0123456789abcdef").body

    def test_html_in_a_result_or_a_log_line_is_escaped(self):
        _disk_run(result=_result(
            message="<u>message</u>", warnings=["<s>warning</s>"],
            trades=[_trade("failed", error="<b>error</b>", a={
                "ticker": "<script>", "market": "<i>m</i>", "side": "yes", "count": 1,
                "price": 0.5})]), output="<script>alert(1)</script>\n")
        body = _get(_app(), "/runs/0123456789abcdef").body
        # The page's one script is the notifier; nothing else opens a script
        assert body.count(f"<script>{defaults_server._NOTIFY_JS}</script>") == 1
        assert "<script>" not in body.replace(
            f"<script>{defaults_server._NOTIFY_JS}</script>", "")
        for text in ("<script>alert(1)</script>", "<i>m</i>", "<b>error</b>",
                     "<u>message</u>", "<s>warning</s>"):
            assert text not in body and html.escape(text) in body, text

    def test_the_tail_reads_only_the_end_of_a_large_file(self, monkeypatch):
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_RUN_LOG_TAIL_BYTES", 4096)
        lines = ["FIRST LINE MARKER"] + [f"line {i:05d}" for i in range(2000)]
        folder = _disk_run(result=_result())
        (folder / "output.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tail = defaults_server._log_tail(folder)
        assert tail == [f"line {i:05d}" for i in range(2000 - 25, 2000)]
        assert len(tail) == config.DEFAULTS_SERVER_RUN_LOG_TAIL_LINES
        body = _get(_app(), "/runs/0123456789abcdef").body
        assert "FIRST LINE MARKER" not in body and "line 01999" in body
        read = []
        real_open = Path.open

        class Spy:
            """Wraps an open file, recording how much each read asks for."""

            def __init__(self, handle):
                """
                Wrap one open file.

                Args:
                    handle: The file.
                """
                self._handle = handle

            def __enter__(self):
                """
                Enter the with block.

                Returns:
                    Spy: This wrapper.
                """
                return self

            def __exit__(self, *exc):
                """
                Close the file.

                Args:
                    *exc: The with block's exception, if any.
                """
                self._handle.close()

            def seek(self, *args):
                """
                Seek, as the file does.

                Args:
                    *args: seek's arguments.

                Returns:
                    int: The new position.
                """
                return self._handle.seek(*args)

            def read(self, size=-1):
                """
                Record the size asked for, then read it.

                Args:
                    size (int): How many bytes to read (-1: all).

                Returns:
                    bytes: What was read.
                """
                read.append(size)
                return self._handle.read(size)

        def spying_open(self, *args, **kwargs):
            """
            Open a file as Path.open does, wrapping output.log in a Spy.

            Args:
                *args: Path.open's arguments.
                **kwargs: Path.open's keyword arguments.

            Returns:
                The file, or a Spy of it.
            """
            handle = real_open(self, *args, **kwargs)
            return Spy(handle) if self.name == "output.log" else handle

        monkeypatch.setattr(Path, "open", spying_open)
        assert defaults_server._log_tail(folder)[-1] == "line 01999"
        # One read, of at most the configured bytes: never the whole file
        assert read == [4096]

    @pytest.mark.parametrize("run_id", [
        "0000000000000000", "0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0",
        "../../etc", "0123456789abcdef/x"])
    def test_an_unknown_or_malformed_id_is_404(self, run_id):
        _disk_run(result=_result())
        assert _get(_app(), f"/runs/{run_id}").status == 404

    def test_two_folders_with_one_id_are_404(self):
        _disk_run(result=_result())
        _disk_run(started="20260930T160000Z", result=_result())
        assert _get(_app(), "/runs/0123456789abcdef").status == 404


class TestLoadRun:
    """A run's folder alone describes it, and a damaged run.json errs toward real money."""

    def test_a_damaged_run_json_is_a_real_money_run(self):
        folder = _disk_run(run_json={"dry_run": "yes", "pid": "1"})
        (folder / "run.json").write_bytes(b"\xff")
        run = defaults_server._load_run(folder)
        assert run.dry_run is False and run.pid is None
        assert run.started_at == datetime(2026, 9, 29, 16, 0, tzinfo=UTC)
        assert run.settings_text == "not recorded"


@pytest.fixture
def live_server(monkeypatch):
    """
    Run the real handler on a free loopback port in a thread.

    The silent-connection timeout is lowered so a test does not wait the
    configured seconds. Skipped when the sandbox refuses to bind a port.

    Args:
        monkeypatch (pytest.MonkeyPatch): Lowers the handler's timeout.

    Yields:
        HTTPServer: The server; its port is server.server_address[1], and a
            test may set its defaults_app before its first request.
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
        yield server
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


# The stand-in for main.py a round trip runs: writes a dry run's result, with
# no trades, at the path it is given, and exits 0
_CHILD = """
import json, sys
json.dump({"format": __FORMAT__, "mode": "prod", "dry_run": True,
           "started_at": "2026-09-29T16:00:00Z", "finished_at": "2026-09-29T16:00:01Z",
           "exit_code": 0, "settings": "", "defaults": "", "message": "No qualifying pairs",
           "balance_before": None, "balance_after": None, "submission_started": False,
           "trades": [], "warnings": [], "warnings_dropped": 0, "error": None},
          open(sys.argv[1], "w"))
print("stand-in for main.py done")
""".replace("__FORMAT__", repr(config.LIVE_RUN_RESULT_FORMAT))


class TestOverASocket:
    """The real handler, over a loopback socket."""

    def test_a_confirm_page_and_its_save(self, live_server):
        port = live_server.server_address[1]
        status, _, body = _request(port, "GET", "/confirm?" + _query(source=DASHBOARD_NOTE))
        assert status == 200
        form = urlencode(_click(body, "confirm")).encode("ascii")
        status, headers, _ = _request(port, "POST", "/confirm", body=form, headers={
            "Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM})
        assert status == 303 and headers["Location"] == "/saved"
        status, _, body = _request(port, "GET", "/saved")
        assert status == 200 and "Saved and verified:" in body
        assert config.read_saved_live_defaults() == _BASE_SETTINGS

    def test_the_most_categories_and_tags_fit_a_page_and_its_save(self, live_server):
        # 40 excluded categories and 40 tied tags, through the handler's request-size limit
        port = live_server.server_address[1]
        query = _filter_query(_MOST_CATEGORIES, _MOST_TAGS, source=DASHBOARD_NOTE)
        status, _, body = _request(port, "GET", f"/confirm?{query}")
        assert status == 200
        form = urlencode(_click(body, "confirm")).encode("ascii")
        status, headers, _ = _request(port, "POST", "/confirm", body=form, headers={
            "Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM})
        assert status == 303 and headers["Location"] == "/saved"
        saved = config.read_saved_live_defaults()
        assert saved.exclude_categories == _MOST_CATEGORIES and saved.exclude_tags == _MOST_TAGS

    def test_a_dry_run_round_trip_starts_a_process_and_shows_its_result(self, live_server):
        port = live_server.server_address[1]
        _save(_BASE_SETTINGS)
        built = []

        def stand_in(argv, **kwargs):
            """
            Start a tiny Python child in place of main.py, with the server's own arguments.

            Args:
                argv (list[str]): The command line the server built.
                **kwargs: Popen's keyword arguments, as the server passes them.

            Returns:
                subprocess.Popen: The child.
            """
            built.append(list(argv))
            result = argv[argv.index("--result-file") + 1]
            return _REAL_POPEN([sys.executable, "-c", _CHILD, result], **kwargs)

        live_server.defaults_app = defaults_server._App(port, start_process=stand_in)
        # The retired /trade address sends the browser to the trading page,
        # which opens on the saved defaults
        status, headers, _ = _request(port, "GET", "/trade")
        assert status == 303 and headers["Location"] == "/confirm"
        status, _, body = _request(port, "GET", "/confirm")
        assert status == 200
        page = _parse(body)
        [button] = [b for b in page.buttons if b.get("id") == "confirm-dry-run"]
        form = urlencode(page.hidden + [(button["name"], button["value"])]).encode("ascii")
        status, headers, _ = _request(port, "POST", "/confirm", body=form, headers={
            "Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM})
        assert status == 303 and headers["Location"].startswith("/runs/")
        [argv] = built
        assert argv[-1] == "--dry-run"
        assert argv[:5] == [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod"]
        run_id = headers["Location"][len("/runs/"):]
        live_server.defaults_app._runs[run_id].process.wait(timeout=30)
        status, _, body = _request(port, "GET", headers["Location"])
        assert status == 200
        assert "<b>No trades to complete</b>" in body
        assert "No qualifying pairs" in body and "stand-in for main.py done" in body

    def test_every_route_sends_the_five_headers(self, live_server):
        port = live_server.server_address[1]
        expected = dict(defaults_server._response_headers(defaults_server._Response(200, "")))
        answers = [
            _request(port, "GET", "/"),
            _request(port, "GET", "/confirm?" + _query()),
            _request(port, "GET", "/confirm"),
            _request(port, "GET", "/confirm?k"),
            _request(port, "GET", "/saved"),
            _request(port, "GET", "/saved?x=1"),
            _request(port, "GET", "/trade"),
            _request(port, "GET", "/runs/0123456789abcdef"),
            _request(port, "GET", "/nope"),
            _request(port, "GET", "/", host="evil.test"),
            _request(port, "POST", "/confirm", body=b"", headers={"Content-Type": FORM}),
            _request(port, "POST", "/trade", body=b"", headers={"Content-Type": FORM}),
        ]
        assert [status for status, _, _ in answers] == [200, 200, 200, 400, 404, 400, 303,
                                                        404, 404, 403, 403, 404]
        for _, headers, _ in answers:
            # Framed only by the live dashboard's tab page: no X-Frame-Options
            assert "X-Frame-Options" not in headers
            for name, value in expected.items():
                assert headers[name] == value, name

    def test_checkout_answers_json_with_the_other_four_headers(self, live_server):
        port = live_server.server_address[1]
        status, headers, body = _request(port, "GET", "/checkout")
        assert status == 200
        assert json.loads(body) == {"project_root": str(config.PROJECT_ROOT.resolve()),
                                    "code": defaults_server._LOADED_CODE}
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        expected = dict(defaults_server._response_headers(defaults_server._Response(200, "")))
        for name in ("Cache-Control", "X-Content-Type-Options", "Referrer-Policy",
                     "Content-Security-Policy"):
            assert headers[name] == expected[name]
        assert "X-Frame-Options" not in headers
        assert _request(port, "GET", "/checkout", host="evil.test")[0] == 403

    @pytest.mark.parametrize("length, status", [
        (None, 411), ("abc", 400), ("-1", 400), ("+5", 400),
        (str(config.DEFAULTS_SERVER_MAX_REQUEST_BYTES + 1), 413),
        # More digits than int() reads from a string
        ("9" * 5000, 413),
    ])
    def test_a_bad_content_length_is_refused(self, live_server, length, status):
        port = live_server.server_address[1]
        headers = {"Origin": f"http://127.0.0.1:{port}", "Content-Type": FORM}
        if length is not None:
            headers["Content-Length"] = length
        got, answer_headers, _ = _request(port, "POST", "/confirm", headers=headers)
        assert got == status
        assert answer_headers["Content-Security-Policy"] == defaults_server._CSP
        assert "X-Frame-Options" not in answer_headers

    def test_two_content_lengths_are_refused(self, live_server):
        headers = {"Content-Type": FORM, "Content-Length": "0"}
        status, _, _ = _request(live_server.server_address[1], "POST", "/confirm", body=b"",
                                headers=headers)
        assert status == 400

    def test_a_path_over_the_limit_is_refused(self, live_server):
        target = "/confirm?" + "k=" + "1" * config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        # The base class reads a request line of at most 65,536 bytes and
        # refuses a longer one with 414 before this server sees it; at a limit
        # that large it is the tighter of the two, and this server's own 413
        # answers a path over a smaller limit
        line = len(f"GET {target} HTTP/1.1\r\n")
        expected = 414 if line > 65_536 else 413
        assert _request(live_server.server_address[1], "GET", target)[0] == expected

    @pytest.mark.parametrize("method", ["PUT", "HEAD"])
    def test_other_methods_get_501_with_the_five_headers(self, live_server, method):
        status, headers, body = _request(live_server.server_address[1], method, "/confirm")
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
    def test_a_request_the_http_layer_refuses_still_gets_the_five_headers(
            self, live_server, data, status):
        got, headers, _ = _raw_request(live_server.server_address[1], data)
        assert got == status
        expected = dict(defaults_server._response_headers(defaults_server._Response(got, "")))
        for name, value in expected.items():
            assert headers[name] == value, name

    def test_a_silent_connection_does_not_hold_the_server(self, live_server):
        port = live_server.server_address[1]
        silent = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            started = time.monotonic()
            status, _, _ = _request(port, "GET", "/")
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
            serve_forever to raise KeyboardInterrupt), "during" (set it to a
            function of the server, called while it serves), "checkout"
            (what the stand-in for _running_checkout answers a busy port
            with: a root, or None, the default), "code" (the code
            fingerprint that answer carries: this checkout's current one
            by default), "asked" (each address it was asked about) and
            "run" (call it with the arguments to run main; see _run_main).
            The stand-in means no test here asks a real listener on the
            real port anything.
    """
    state = {"servers": [], "opened": [], "fail": None, "interrupt": False, "during": None,
             "checkout": None, "code": defaults_server._code_fingerprint(), "asked": []}

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
            Note that main served, run the test's hook, and return at once.

            Returns:
                None

            Raises:
                KeyboardInterrupt: When state["interrupt"] is True, as Ctrl-C would.
            """
            self.served = True
            if state["during"] is not None:
                state["during"](self)
            if state["interrupt"]:
                raise KeyboardInterrupt

        def server_close(self):
            """
            Note that main closed the server.

            Returns:
                None
            """
            self.closed = True

    def running_checkout(base):
        """
        Stand in for _running_checkout: record the address, answer state["checkout"].

        Args:
            base (str): The address main asked about.

        Returns:
            defaults_server._RunningServer | None: state["checkout"] with
                state["code"], or None when state["checkout"] is None.
        """
        state["asked"].append(base)
        if state["checkout"] is None:
            return None
        return defaults_server._RunningServer(project_root=state["checkout"],
                                              code=state["code"])

    monkeypatch.setattr(defaults_server, "HTTPServer", FakeServer)
    monkeypatch.setattr(defaults_server.webbrowser, "open", state["opened"].append)
    monkeypatch.setattr(defaults_server, "_running_checkout", running_checkout)
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
        assert f"trades them at http://127.0.0.1:{PORT}/confirm" in log
        assert "Ctrl-C stops it; stop it when you are done." in log
        # A free port: nothing asked who holds it
        assert run_main["asked"] == []

    def test_its_app_can_start_runs_through_popen(self, run_main):
        run_main["run"](["--no-browser"])
        [server] = run_main["servers"]
        assert server.defaults_app._start_process is subprocess.Popen

    def test_a_run_still_going_at_shutdown_is_named(self, run_main, tmp_path):
        def during(server):
            """
            Give the server's app one run still going and one that has ended.

            Args:
                server: The fake server, its defaults_app set by main().
            """
            server.defaults_app._runs["fedcba9876543210"] = defaults_server._Run(
                run_id="fedcba9876543210", folder=tmp_path / "live_runs" / "r",
                process=_FakeProcess(4343, None), dry_run=False, settings_text="",
                started_at=None, saved_note="")
            server.defaults_app._runs["0123456789abcdef"] = defaults_server._Run(
                run_id="0123456789abcdef", folder=tmp_path / "done",
                process=_FakeProcess(4444, 0), dry_run=False, settings_text="",
                started_at=None, saved_note="")

        run_main["during"] = during
        run_main["interrupt"] = True
        log = run_main["run"](["--no-browser"])
        assert ("A live trading run started here is still running (process 4343); it keeps "
                "running on its own — its result will be in "
                f"{tmp_path / 'live_runs' / 'r'}") in log
        assert "process 4444" not in log

    def test_no_browser_opens_nothing(self, run_main):
        log = run_main["run"](["--seed", "--no-browser"])
        assert run_main["opened"] == []
        assert defaults_server._seed_query() in log

    def test_it_opens_the_trading_page(self, run_main, tmp_path):
        url = f"http://127.0.0.1:{PORT}/confirm"
        log = run_main["run"]([])
        assert run_main["opened"] == [url]
        assert (f"Trading page: open {url} to review the live defaults and trade from "
                "them") in log
        # Whatever backtest page is in the checkout, which the server never reads
        (tmp_path / config.DASHBOARD_FILENAME).write_text("<html></html>", encoding="utf-8")
        log = run_main["run"]([])
        assert run_main["opened"] == [url, url]
        assert "WARNING" not in log

    def test_it_logs_the_defaults_in_force(self, run_main):
        saved = _save(config.LIVE_DEFAULTS_SEED)
        log = run_main["run"](["--no-browser"])
        assert f"Live defaults in force: {saved.origin}" in log

    def test_it_logs_a_refused_file(self, run_main):
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        log = run_main["run"](["--no-browser"])
        assert "The saved live defaults are refused" in log

    def test_a_busy_port_exits_2_before_logging(self, run_main, tmp_path, capsys):
        # The listener gives no /checkout answer (run_main's stand-in says None)
        run_main["fail"] = OSError(errno.EADDRINUSE, "Address already in use")
        with pytest.raises(SystemExit) as exit_info:
            run_main["run"](["--seed"])
        assert exit_info.value.code == 2
        assert (f"port {PORT} is in use — stop the other server, or change "
                "config.DEFAULTS_SERVER_PORT and rebuild the dashboard") in capsys.readouterr().err
        assert run_main["asked"] == [f"http://127.0.0.1:{PORT}"]
        assert run_main["opened"] == [] and run_main["servers"] == []
        assert not (tmp_path / "kalshi_defaults_server.log").exists()

    @pytest.mark.parametrize("argv, opened", [
        (["--seed"], lambda base: [f"{base}/confirm?{defaults_server._seed_query()}"]),
        ([], lambda base: [f"{base}/confirm"]),
        (["--no-browser"], lambda base: []),
        (["--seed", "--no-browser"], lambda base: []),
    ])
    def test_this_checkout_s_running_server_has_its_page_opened_again(
            self, run_main, tmp_path, capsys, argv, opened):
        run_main["fail"] = OSError(errno.EADDRINUSE, "Address already in use")
        run_main["checkout"] = str(tmp_path.resolve())
        base = f"http://127.0.0.1:{PORT}"
        # It returns (exit 0): nothing bound, no second server, no log file of its own
        assert run_main["run"](argv) == ""
        assert run_main["opened"] == opened(base)
        assert run_main["servers"] == []
        assert not (tmp_path / "kalshi_defaults_server.log").exists()
        err = capsys.readouterr().err
        assert f"This checkout's defaults server is already running at {base}/" in err
        if "--seed" in argv:
            assert f"Seed values: open {base}/confirm?" in err

    def test_another_checkout_s_server_on_the_port_is_refused(self, run_main, tmp_path,
                                                            capsys):
        run_main["fail"] = OSError(errno.EADDRINUSE, "Address already in use")
        run_main["checkout"] = "/elsewhere/other checkout\x1b[31m"
        with pytest.raises(SystemExit) as exit_info:
            run_main["run"]([])
        assert exit_info.value.code == 2
        # The other root is named, its control characters written as escapes
        assert (f"port {PORT} is served by the defaults server of /elsewhere/other "
                "checkout\\x1b[31m — stop it (Ctrl-C in its terminal) before starting this "
                "checkout's") in capsys.readouterr().err
        assert run_main["opened"] == [] and run_main["servers"] == []
        assert not (tmp_path / "kalshi_defaults_server.log").exists()

    @pytest.mark.parametrize("code", ["0" * 64, None], ids=["other-code", "no-code"])
    def test_this_checkout_s_server_running_other_code_is_refused(self, run_main, tmp_path,
                                                                  capsys, code):
        # The running server loaded code that is not this checkout's code now
        # (a pull or an edit since it started), or its answer carries none
        run_main["fail"] = OSError(errno.EADDRINUSE, "Address already in use")
        run_main["checkout"] = str(tmp_path.resolve())
        run_main["code"] = code
        with pytest.raises(SystemExit) as exit_info:
            run_main["run"](["--seed"])
        assert exit_info.value.code == 2
        assert (f"port {PORT} is served by this checkout's defaults server, but it is running "
                "code from before a change to this checkout — stop it (Ctrl-C in its "
                "terminal) and start it again") in capsys.readouterr().err
        assert run_main["opened"] == [] and run_main["servers"] == []
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

    def test_it_turns_ctrl_c_back_on(self, run_main, monkeypatch):
        # start_dashboard.sh starts it in the background, where a job begins
        # with Ctrl-C ignored: main makes Ctrl-C stop it again, first of all
        calls = []
        monkeypatch.setattr(defaults_server.signal, "signal",
                            lambda signum, handler: calls.append((signum, handler)))
        run_main["run"](["--no-browser"])
        assert calls == [(signal.SIGINT, signal.default_int_handler)]


@pytest.fixture
def busy_port(tmp_path, monkeypatch):
    """
    Run defaults_server.main on a port a real loopback listener holds, never binding it.

    The server's port (defaults_server.DEFAULTS_SERVER_PORT) is patched to
    the listener's; main's own bind is a guard that fails the test if it
    ever succeeds; the browser is replaced.

    Args:
        tmp_path (Path): The test's directory (config.PROJECT_ROOT, by the
            module's autouse fixture).
        monkeypatch (pytest.MonkeyPatch): Patches the port, the bind and the
            browser.

    Yields:
        dict: "hold" (call it with a listening server or socket to point the
            port at it), "opened" (each address the browser was asked to
            open) and "run" (call it with main's arguments; see _run_main).
    """
    state = {"opened": []}

    def guarded_bind(address, handler):
        """
        Bind as main would; a bind that succeeds means the port was not held after all.

        Args:
            address (tuple[str, int]): Where main asked to listen.
            handler (type): The request handler class.

        Raises:
            OSError: The real bind's error (EADDRINUSE while the port is held).
            AssertionError: When the bind succeeded.
        """
        server = HTTPServer(address, handler)
        server.server_close()
        raise AssertionError(f"main bound {address} although the port was held")

    def hold(port: int) -> None:
        """
        Point the server's port at a port something already listens on.

        Args:
            port (int): The held port.
        """
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_PORT", port)

    monkeypatch.setattr(defaults_server, "HTTPServer", guarded_bind)
    monkeypatch.setattr(defaults_server.webbrowser, "open", state["opened"].append)
    # The listener answers on its own thread while _run_main has set the root
    # logger's handlers aside, and logging.info() on a root with no handler
    # configures logging itself (at WARNING), before main can: the listener's
    # request line is not logged here, so main's own configuration stands
    monkeypatch.setattr(defaults_server._Handler, "log_message", lambda self, *args: None)
    state["hold"] = hold
    state["run"] = lambda argv: _run_main(argv, tmp_path)
    yield state


@pytest.fixture
def stub_listener():
    """
    Start stand-in HTTP listeners on free loopback ports, each answering from a table.

    Skipped when the sandbox refuses to bind a port.

    Yields:
        Callable: start(replies) takes {path: (status, headers, body)} (a
            path it lacks gets 404) and returns (port, requests), requests
            being the list of (path, Host header) each request carried.
    """
    started = []

    def start(replies: dict) -> tuple[int, list]:
        """
        Start one stand-in listener.

        Args:
            replies (dict): Each path's (status, headers, body bytes).

        Returns:
            tuple[int, list]: Its port, and the list its requests are recorded in.
        """
        requests = []

        class Handler(BaseHTTPRequestHandler):
            """Answers each GET from the table and records it."""

            def do_GET(self):
                """
                Record the request and send its reply from the table.

                Returns:
                    None
                """
                requests.append((self.path, self.headers.get("Host")))
                status, headers, body = replies.get(self.path, (404, {}, b"not here"))
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                """
                Log nothing.

                Args:
                    format (str): Ignored.
                    *args: Ignored.
                """

        try:
            server = HTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError as exc:
            pytest.skip(f"this sandbox does not allow binding a local port: {exc}")
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        started.append((server, thread))
        return server.server_address[1], requests

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestBusyPortOverASocket:
    """A start that finds the port held, against real listeners: reuse this
    checkout's server, refuse anything else, and never bind or start anything."""

    def test_this_checkout_s_server_has_the_seed_page_opened_again(self, live_server,
                                                                   busy_port, capsys):
        port = live_server.server_address[1]
        live_server.defaults_app = defaults_server._App(port)
        busy_port["hold"](port)
        assert busy_port["run"](["--seed"]) == ""
        base = f"http://127.0.0.1:{port}"
        assert busy_port["opened"] == [f"{base}/confirm?{defaults_server._seed_query()}"]
        assert (f"This checkout's defaults server is already running at {base}/"
                in capsys.readouterr().err)
        # Nothing started: no run, no run folder, no log file of its own
        assert live_server.defaults_app._runs == {}
        assert not config.LIVE_RUNS_DIR.exists()
        assert not (config.PROJECT_ROOT / "kalshi_defaults_server.log").exists()

    def test_without_seed_it_opens_the_trading_page(self, live_server, busy_port):
        port = live_server.server_address[1]
        live_server.defaults_app = defaults_server._App(port)
        busy_port["hold"](port)
        busy_port["run"]([])
        # Whatever backtest file is in the checkout
        (config.PROJECT_ROOT / config.DASHBOARD_FILENAME).write_text("<html></html>",
                                                                     encoding="utf-8")
        busy_port["run"]([])
        trading = f"http://127.0.0.1:{port}/confirm"
        assert busy_port["opened"] == [trading, trading]

    def test_no_browser_opens_nothing(self, live_server, busy_port, capsys):
        port = live_server.server_address[1]
        live_server.defaults_app = defaults_server._App(port)
        busy_port["hold"](port)
        busy_port["run"](["--seed", "--no-browser"])
        assert busy_port["opened"] == []
        assert defaults_server._seed_query() in capsys.readouterr().err

    def test_another_checkout_s_server_is_refused(self, stub_listener, busy_port, capsys):
        other = "/somewhere/else/other checkout"
        port, requests = stub_listener({"/checkout": (
            200, {"Content-Type": "application/json"},
            json.dumps({"project_root": other}).encode("utf-8"))})
        busy_port["hold"](port)
        with pytest.raises(SystemExit) as exit_info:
            busy_port["run"](["--seed"])
        assert exit_info.value.code == 2
        assert (f"port {port} is served by the defaults server of {other} — stop it (Ctrl-C "
                "in its terminal) before starting this checkout's") in capsys.readouterr().err
        assert busy_port["opened"] == []
        # It asked once, naming the server as its Host check needs
        assert requests == [("/checkout", f"127.0.0.1:{port}")]

    @pytest.mark.parametrize("replies", [
        {},                                                        # 404 everywhere
        {"/checkout": (200, {}, b"hello")},                        # not JSON
        {"/checkout": (200, {}, b'["/x"]')},                       # not a JSON object
        {"/checkout": (200, {}, b'{"root": "/x"}')},               # no project_root
        {"/checkout": (200, {}, b'{"project_root": ""}')},         # an empty one
        {"/checkout": (200, {}, b'{"project_root": 5}')},          # not a string
        {"/checkout": (500, {}, b"broken")},                       # an error status
        {"/checkout": (200, {}, b"[" * 60_000)},                   # nested too deeply to parse
    ], ids=["404", "not-json", "list", "no-root", "empty-root", "number-root", "500",
            "deeply-nested"])
    def test_a_listener_that_is_not_a_defaults_server_gives_the_port_in_use_error(
            self, stub_listener, busy_port, capsys, replies):
        port, requests = stub_listener(replies)
        busy_port["hold"](port)
        with pytest.raises(SystemExit) as exit_info:
            busy_port["run"]([])
        assert exit_info.value.code == 2
        assert f"port {port} is in use — stop the other server" in capsys.readouterr().err
        assert busy_port["opened"] == []
        assert [path for path, _ in requests] == ["/checkout"]

    @pytest.mark.parametrize("answer", [
        {"code": "0" * 64},                                        # other code
        {},                                                        # no code at all
        {"code": 5},                                               # not a fingerprint
    ], ids=["other-code", "no-code", "number-code"])
    def test_this_checkout_s_server_running_other_code_is_refused(
            self, stub_listener, busy_port, capsys, answer):
        body = json.dumps({"project_root": str(config.PROJECT_ROOT.resolve()), **answer})
        port, requests = stub_listener({"/checkout": (200, {}, body.encode("utf-8"))})
        busy_port["hold"](port)
        with pytest.raises(SystemExit) as exit_info:
            busy_port["run"](["--seed"])
        assert exit_info.value.code == 2
        assert (f"port {port} is served by this checkout's defaults server, but it is "
                "running code from before a change to this checkout — stop it (Ctrl-C in "
                "its terminal) and start it again") in capsys.readouterr().err
        assert busy_port["opened"] == []
        assert [path for path, _ in requests] == ["/checkout"]

    def test_a_browser_s_idle_connection_does_not_hide_this_checkout_s_server(
            self, live_server, busy_port, capsys, monkeypatch):
        # The running server answers one connection at a time, and drops a
        # connection that sends nothing only after its handler's timeout, so
        # the question waits behind a browser's idle connection. The wait is
        # scaled from config's ratio of the question's timeout to the
        # server's, so this fails if that ratio leaves no room for the wait
        port = live_server.server_address[1]
        live_server.defaults_app = defaults_server._App(port)
        busy_port["hold"](port)
        handler_timeout = defaults_server._Handler.timeout
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS",
                            handler_timeout * config.DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS
                            / config.DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS)
        idle = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            started = time.monotonic()
            assert busy_port["run"](["--no-browser"]) == ""
            waited = time.monotonic() - started
        finally:
            idle.close()
        assert (f"This checkout's defaults server is already running at "
                f"http://127.0.0.1:{port}/") in capsys.readouterr().err
        assert busy_port["opened"] == []
        # The answer came only once the idle connection was dropped
        assert waited >= handler_timeout * 0.8

    def test_a_redirect_is_not_followed(self, stub_listener, busy_port, capsys):
        # Followed, the redirect would reach an answer naming this checkout
        mine = json.dumps({"project_root": str(config.PROJECT_ROOT.resolve()),
                           "code": defaults_server._code_fingerprint()}).encode()
        port, requests = stub_listener({
            "/checkout": (302, {"Location": "/moved"}, b""),
            "/moved": (200, {}, mine)})
        busy_port["hold"](port)
        with pytest.raises(SystemExit) as exit_info:
            busy_port["run"]([])
        assert exit_info.value.code == 2
        assert f"port {port} is in use" in capsys.readouterr().err
        assert [path for path, _ in requests] == ["/checkout"]
        assert busy_port["opened"] == []

    def test_an_answer_over_the_limit_is_not_read_as_one(self, stub_listener, busy_port,
                                                         capsys, monkeypatch):
        mine = json.dumps({"project_root": str(config.PROJECT_ROOT.resolve()),
                           "code": defaults_server._code_fingerprint()}).encode()
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_CHECKOUT_MAX_BYTES",
                            len(mine) - 1)
        port, _ = stub_listener({"/checkout": (200, {}, mine)})
        busy_port["hold"](port)
        with pytest.raises(SystemExit) as exit_info:
            busy_port["run"]([])
        assert exit_info.value.code == 2
        assert f"port {port} is in use" in capsys.readouterr().err
        # At exactly the limit it is an answer
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_CHECKOUT_MAX_BYTES", len(mine))
        busy_port["run"](["--no-browser"])
        assert busy_port["opened"] == []

    def test_a_silent_listener_gives_the_port_in_use_error(self, busy_port, capsys,
                                                           monkeypatch):
        monkeypatch.setattr(defaults_server, "DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS", 0.2)
        silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                silent.bind(("127.0.0.1", 0))
            except PermissionError as exc:
                pytest.skip(f"this sandbox does not allow binding a local port: {exc}")
            silent.listen(1)
            busy_port["hold"](silent.getsockname()[1])
            started = time.monotonic()
            with pytest.raises(SystemExit) as exit_info:
                busy_port["run"]([])
            assert time.monotonic() - started < 10
        finally:
            silent.close()
        assert exit_info.value.code == 2
        assert "is in use — stop the other server" in capsys.readouterr().err
        assert busy_port["opened"] == []


class TestIsolation:
    """A person runs the defaults server; nothing imports it, and it starts processes in one place only."""

    TREE = ast.parse(inspect.getsource(defaults_server))

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

    def test_its_copies_of_the_sale_tables_match_their_sources(self):
        # It may not import reporter or main, so it keeps its own copies of
        # the sale status words and of the sale statuses that need a person;
        # they must say exactly what the originals say
        reporter = importlib.import_module("kalshi_betting.reporter")
        main = importlib.import_module("kalshi_betting.main")
        assert defaults_server._SALE_STATUS_WORDS == reporter._SALE_STATUS_WORDS
        assert defaults_server._SALE_ATTENTION_STATUSES == main._SALE_ATTENTION_STATUSES

    def test_it_imports_the_standard_library_config_and_run_lock_only(self):
        project, other = set(), set()
        for node in ast.walk(self.TREE):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.module:
                    project.add(node.module)
                else:
                    project |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                other.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                other |= {a.name.split(".")[0] for a in node.names}
        assert project == {"config", "run_lock"}
        assert other and other <= set(sys.stdlib_module_names), other - set(
            sys.stdlib_module_names)

    # The standard library modules the server may import: the ones it needs,
    # and none that starts a process another way (pty, multiprocessing,
    # asyncio, concurrent.futures) or loads code by name (importlib, runpy,
    # ctypes); _import_problems holds it to them
    _STDLIB_ALLOWED = frozenset({
        "argparse", "base64", "collections", "dataclasses", "datetime", "errno", "fcntl",
        "hashlib", "hmac", "html", "http", "json", "logging", "math", "os", "pathlib", "re",
        "secrets", "signal", "subprocess", "sys", "urllib", "webbrowser"})

    @classmethod
    def _import_problems(cls, tree) -> list[str]:
        """
        List every way a module's imports could reach a process starter the other checks miss.

        The other checks read uses spelled os.X or subprocess.X, so these
        must hold: every module imported is one of _STDLIB_ALLOWED (or of the
        package), os and subprocess are imported whole and never renamed,
        and Popen is never a bare name outside an annotation.

        Args:
            tree (ast.AST): The module's tree.

        Returns:
            list[str]: One line per problem; [] when there is none.
        """
        problems = []
        annotations = cls._annotations_of(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                top = node.module.split(".")[0]
                if top not in cls._STDLIB_ALLOWED or top in {"os", "subprocess"}:
                    problems.append(ast.unparse(node))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    renamed = top in {"os", "subprocess"} and alias.asname is not None
                    if top not in cls._STDLIB_ALLOWED or renamed:
                        problems.append(ast.unparse(node))
            elif isinstance(node, ast.alias) and "Popen" in (node.name, node.asname):
                problems.append(f"alias {node.name}")
            elif (isinstance(node, ast.Name) and node.id == "Popen"
                  and id(node) not in annotations):
                problems.append("bare Popen")
        return problems

    def test_no_process_starter_is_imported_under_another_name(self):
        assert self._import_problems(self.TREE) == []

    @pytest.mark.parametrize("snippet", [
        "from subprocess import Popen\nPopen(['x'])",
        "from subprocess import run as go\ngo(['x'])",
        "import subprocess as sp\nsp.run(['x'])",
        "from os import system\nsystem('x')",
        "import pty\npty.spawn(['x'])",
        "import multiprocessing",
        "import concurrent.futures",
        "from asyncio import create_subprocess_exec",
        "import importlib",
        "Popen = subprocess.Popen\nPopen(['x'])"])
    def test_the_import_check_refuses_a_renamed_process_starter(self, snippet):
        assert self._import_problems(ast.parse(snippet)), snippet

    @staticmethod
    def _parents(tree) -> dict:
        """
        Map each node of a tree to its parent.

        Args:
            tree (ast.AST): The tree.

        Returns:
            dict: id(node) -> its parent node.
        """
        return {id(child): parent for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)}

    def _enclosing(self, node, parents) -> str | None:
        """
        Name the function a node sits in, a method with its class.

        Args:
            node (ast.AST): The node.
            parents (dict): _parents of its tree.

        Returns:
            str | None: "Class.method", "function", or None at module level.
        """
        cur = parents.get(id(node))
        while cur is not None and not isinstance(cur, ast.FunctionDef):
            cur = parents.get(id(cur))
        if cur is None:
            return None
        owner = parents.get(id(cur))
        return f"{owner.name}.{cur.name}" if isinstance(owner, ast.ClassDef) else cur.name

    def _annotations(self) -> set:
        """
        List every node inside an annotation of this module, which may name Popen as a type.

        Returns:
            set: The ids of every node of every annotation in the module.
        """
        return self._annotations_of(self.TREE)

    @staticmethod
    def _annotations_of(tree) -> set:
        """
        List every node inside an annotation of a tree.

        Args:
            tree (ast.AST): The tree.

        Returns:
            set: The ids of every node of every annotation in it.
        """
        inside = set()
        for node in ast.walk(tree):
            roots = []
            if isinstance(node, ast.arg) and node.annotation is not None:
                roots.append(node.annotation)
            elif isinstance(node, ast.AnnAssign):
                roots.append(node.annotation)
            elif isinstance(node, ast.FunctionDef) and node.returns is not None:
                roots.append(node.returns)
            for root in roots:
                inside |= {id(n) for n in ast.walk(root)}
        return inside

    def test_the_only_process_start_is_start_run_s(self):
        parents = self._parents(self.TREE)
        starts = [node for node in ast.walk(self.TREE)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "_start_process"]
        assert [self._enclosing(node, parents) for node in starts] == ["_App._start_run"]
        [start] = starts
        assert isinstance(start.func.value, ast.Name) and start.func.value.id == "self"
        keywords = {kw.arg: kw.value for kw in start.keywords}
        assert isinstance(keywords["start_new_session"], ast.Constant)
        assert keywords["start_new_session"].value is True

    def test_popen_is_named_only_as_main_s_start_process(self):
        parents = self._parents(self.TREE)
        annotations = self._annotations()
        uses = [node for node in ast.walk(self.TREE)
                if isinstance(node, ast.Attribute) and node.attr == "Popen"
                and id(node) not in annotations]
        assert len(uses) == 1
        [use] = uses
        keyword = parents[id(use)]
        assert isinstance(keyword, ast.keyword) and keyword.arg == "start_process"
        assert self._enclosing(use, parents) == "main"
        # The constructor's default is None: nothing built without it can start a run
        init = next(n for n in ast.walk(self.TREE)
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    and self._enclosing(n, parents) is None
                    and getattr(parents[id(n)], "name", None) == "_App")
        [default] = [d for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
                     if a.arg == "start_process"]
        assert isinstance(default, ast.Constant) and default.value is None

    def test_nothing_else_runs_a_program(self):
        forbidden = {("subprocess", name) for name in (
            "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")}
        forbidden |= {("os", name) for name in ("system", "popen", "startfile")}
        for node in ast.walk(self.TREE):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
                continue
            pair = (node.value.id, node.attr)
            assert pair not in forbidden, pair
            assert not (node.value.id == "os" and node.attr.startswith(
                ("exec", "spawn", "posix_spawn", "fork"))), pair

    def test_the_run_s_command_line_starts_main_in_production(self):
        start_run = next(n for n in ast.walk(self.TREE)
                         if isinstance(n, ast.FunctionDef) and n.name == "_start_run")
        [assign] = [n for n in ast.walk(start_run) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "argv" for t in n.targets)]
        head = assign.value.elts[:5]
        assert isinstance(head[0], ast.Attribute) and head[0].attr == "executable"
        assert head[0].value.id == "sys"
        assert [c.value for c in head[1:]] == ["-m", "kalshi_betting.main", "--mode", "prod"]
