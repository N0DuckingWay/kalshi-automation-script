"""
File: test_defaults_server.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for defaults_server.py, the local web server whose pages save the
    live trading defaults and start live trading runs with them. Most tests
    call _App.handle directly with a _Request, so they need no socket: the
    Host check, the index page, the confirmation page (every changed row
    highlighted, warnings, escaping, the script and its Content-Security-
    Policy hash, the buttons in their fixed order with placeholders for
    those that do not apply), every refused proposal, the save and each
    check in front of it (Origin, content type, fingerprint, nonce and
    token, a stale page, a failed write), the trade page, the three actions
    (save, trade, dry run) and every check before a run starts (the run's
    blockers, the last real-money run's attention banner and its
    acknowledgement, a nonce used twice), the saved page, the run page (each
    rule of its outcome table, read from hand-written run folders and from
    a stand-in process), /checkout and the seed. A POST body is always built
    from the hidden inputs and one button of the page the server rendered,
    never by hand, except where a test signs or crafts a request itself to
    reach a path no rendered page leads to. The process starter is a
    stand-in that records each start and starts nothing (_Starter); no test
    starts main.py. One class runs the page script itself under node or
    macOS's jsc with a stand-in page, one runs the real handler over a
    loopback socket (skipped only if the sandbox refuses to bind a port),
    including one round trip whose stand-in starts a tiny Python child in
    place of main.py, one runs main() with the server and the browser
    replaced (which page a start opens, the dashboard's Save/Trade marker
    check, and what a start does when its port is taken), one runs main()
    against real loopback listeners holding its port (this checkout's
    server, another checkout's, and listeners that are not a defaults
    server) without ever binding it, and one checks what this module imports
    and where it starts a process.

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
    port at a listener of their own (busy_port). The page script's tests
    skip when neither node nor jsc is present.
"""
import ast
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
# The eight rows of a comparison, in their order on the page
LABELS = ["tier floors", "spread band", "k", "per-trade cap", "same-title cap",
          "categories", "tags", "add to held pairs"]
# A proposal's fields: the seed's values, spelled as the dashboard spells them
_BASE = {"tier_floors": "off", "spread_min": "0", "spread_max": "0.5", "k": "0.8",
         "size_cap": "0.1"}
# The settings _BASE proposes when nothing is saved: the seed's (its same-title
# cap, and adding to held pairs on, the two fields a proposal may leave out)
_BASE_SETTINGS = LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2, add_to_held_pairs=True)
# The ASCII digits as Arabic-Indic and as full-width digits (str.translate tables)
_ARABIC_INDIC = str.maketrans("0123456789", "".join(chr(0x0660 + i) for i in range(10)))
_FULL_WIDTH = str.maketrans("0123456789", "".join(chr(0xFF10 + i) for i in range(10)))
# The button labels of each page, in page order
_CONFIRM_LABELS = ["Dry run (no orders; defaults unchanged)", "Confirm and save",
                   "Confirm and trade"]
_TRADE_LABELS = ["Dry run (no orders)", "Confirm and trade"]
# Each armed button's id and the action it posts
_ID_ACTION = {"confirm-dry-run": "dry_run", "confirm": "save", "confirm-trade": "trade"}
# A dashboard page with the filter bar's Save and Trade buttons (the Trade
# link's id is what the server looks for), and one built before them
_DASHBOARD_WITH_BUTTONS = ('<html><body><div id="flt-bar"><button id="flt-save">Save</button>'
                           '<a id="flt-trade" href="x">Trade</a></div></body></html>')
_DASHBOARD_BEFORE_BUTTONS = '<html><body><div id="flt-bar"></div></body></html>'


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

    Collects the forms (method, action), the hidden inputs in page order,
    the checkboxes, the buttons' attributes and labels in page order, the
    links, each comparison row's class and cell texts by its data-setting
    label, and the text of every script.
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
        self.checkboxes: list[dict] = []
        self.buttons: list[dict] = []
        self.labels: list[str] = []
        self.links: list[str] = []
        self.rows: dict[str, str | None] = {}
        self.cells: dict[str, list[str]] = {}
        self.scripts: list[str] = []
        self._row: str | None = None
        self._in_script = False
        self._in_button = False

    def handle_starttag(self, tag, attrs) -> None:
        """
        Record a form, input, button, link, row, cell or script as it opens.

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

    def handle_endtag(self, tag) -> None:
        """
        Close a row, a button or a script.

        Args:
            tag (str): The tag's name, lower-cased.

        Returns:
            None
        """
        if tag == "tr":
            self._row = None
        elif tag == "script":
            self._in_script = False
        elif tag == "button":
            self._in_button = False

    def handle_data(self, data) -> None:
        """
        Add text to the open script, button or table cell.

        Args:
            data (str): The text.

        Returns:
            None
        """
        if self._in_script:
            self.scripts[-1] += data
        elif self._in_button:
            self.labels[-1] += data
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


def _trade_form(app, button_id: str = "confirm-trade", *,
                ack: bool = False) -> list[tuple[str, str]]:
    """
    Open the trade page and click one of its buttons.

    Args:
        app (defaults_server._App): The application.
        button_id (str): The button clicked (Confirm and trade by default).
        ack (bool): Whether the attention box is ticked.

    Returns:
        list[tuple[str, str]]: The form the browser posts.
    """
    response = _get(app, "/trade")
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

    def test_without_a_dashboard(self):
        response = _get(_app(), "/")
        assert response.status == 200
        page = _parse(response.body)
        assert page.forms == [] and page.scripts == [] and page.buttons == []
        assert ("No backtest dashboard yet: run <code>python3 -m kalshi_betting.backtest"
                "</code> to build one") in response.body
        assert f"reads its defaults from <code>{config.LIVE_DEFAULTS_FILE.absolute()}</code>" \
            in response.body
        assert "This writes" not in response.body
        assert "No run has been started from this checkout's server." in response.body

    def test_with_a_dashboard_and_its_links(self):
        dashboard = config.PROJECT_ROOT / config.DASHBOARD_FILENAME
        dashboard.write_text(_DASHBOARD_WITH_BUTTONS, encoding="utf-8")
        response = _get(_app(), "/")
        page = _parse(response.body)
        assert str(dashboard.absolute()) in response.body
        assert ("use its filter bar's Save as live defaults… button to save a scenario, or "
                "its Trade using defaults… button to trade the saved ones") in response.body
        assert page.links[:2] == [f"/confirm?{defaults_server._seed_query()}", "/trade"]
        assert "Start from the seed values" in response.body
        assert "Trade using defaults" in response.body
        # The page a save redirects to is never linked
        assert "/saved" not in page.links

    def test_a_dashboard_built_before_the_buttons_says_to_rebuild_it(self):
        dashboard = config.PROJECT_ROOT / config.DASHBOARD_FILENAME
        dashboard.write_text(_DASHBOARD_BEFORE_BUTTONS, encoding="utf-8")
        response = _get(_app(), "/")
        assert response.status == 200
        assert (f"The backtest dashboard at <code>{dashboard.absolute()}</code> was built "
                "before its Save as live defaults… and Trade using defaults… buttons: rebuild "
                "it with <code>python3 -m kalshi_betting.backtest</code> and the "
                "<code>--start-date</code> you built it from. Until then, use the links "
                "below.") in response.body
        assert _parse(response.body).links[:2] == [
            f"/confirm?{defaults_server._seed_query()}", "/trade"]

    def test_a_dashboard_that_cannot_be_read_says_why(self):
        # A folder where the file should be: it exists but cannot be read as one
        dashboard = config.PROJECT_ROOT / config.DASHBOARD_FILENAME
        dashboard.mkdir()
        response = _get(_app(), "/")
        assert response.status == 200
        assert f"The backtest dashboard at <code>{dashboard.absolute()}</code> could not be " \
               "read (IsADirectoryError" in response.body
        assert _parse(response.body).links[:2] == [
            f"/confirm?{defaults_server._seed_query()}", "/trade"]

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
        assert headers["X-Frame-Options"] == "DENY"


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
        assert "2 of 8 settings change." in response.body
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
        assert "8 of 8 settings change." in response.body
        # The same-title cap and adding to held pairs fall back to the seed's
        assert page.cells["same-title cap"][2] == "20%"
        assert page.cells["add to held pairs"][2] == "on"

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

    @pytest.mark.parametrize("saved_on", [False, True])
    def test_a_missing_add_to_held_pairs_keeps_the_saved_value(self, saved_on):
        # A link that does not name it (the dashboard's save button, on a page
        # that does not show the choice) leaves it as saved, either way, shown
        # as no change
        saved = LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2, add_to_held_pairs=saved_on)
        _save(saved)
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.rows["add to held pairs"] == "same"
        word = "on" if saved_on else "off"
        assert page.cells["add to held pairs"][1:3] == [word, word]
        assert defaults_server._proposal(defaults_server._params(_query()),
                                         config.read_saved_live_defaults())[0] == saved

    @pytest.mark.parametrize("given, saved_on", [("on", False), ("off", True)])
    def test_a_named_add_to_held_pairs_is_proposed(self, given, saved_on):
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2, add_to_held_pairs=saved_on))
        page = _parse(_get(_app(), "/confirm?" + _query(add_to_held_pairs=given)).body)
        assert page.rows["add to held pairs"] == "changed"
        assert page.cells["add to held pairs"][1:3] == ["on" if saved_on else "off", given]

    def test_a_missing_add_to_held_pairs_with_none_saved_is_the_seed_s(self):
        settings, _ = defaults_server._proposal(defaults_server._params(_query()), None)
        assert settings.add_to_held_pairs is config.LIVE_DEFAULTS_SEED.add_to_held_pairs

    def test_a_missing_category_or_tag_proposes_any(self):
        _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.1, 0.2, ("Sports",), ("Basketball",)))
        page = _parse(_get(_app(), "/confirm?" + _query(same_title_size_cap="0.2")).body)
        assert page.rows["categories"] == page.rows["tags"] == "changed"
        assert page.cells["categories"][1:3] == ["Sports", "any"]
        assert page.cells["tags"][1:3] == ["Basketball", "any"]

    def test_confirm_is_there_and_disabled_when_something_changes(self):
        # A server built without a process starter: only the save applies
        page = _parse(_get(_app(), "/confirm?" + _query()).body)
        assert page.forms == [("post", "/confirm")]
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

    def test_nothing_to_change_leaves_the_save_a_placeholder(self):
        _save(_BASE_SETTINGS)
        response = _get(_app(start_process=_Starter()), "/confirm?" + _query())
        page = _parse(response.body)
        assert page.labels == _CONFIRM_LABELS
        assert [b.get("id") for b in page.buttons] == ["confirm-dry-run", None, "confirm-trade"]
        assert page.buttons[1] == {"type": "button", "disabled": None}
        assert html.escape(defaults_server._NOTHING_TO_SAVE) in response.body
        assert "These are already the live defaults — nothing to save." in response.body
        assert "0 of 8 settings change." in response.body
        # With nothing to save, the page does not say it writes
        assert "This writes" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body

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
        assert "ARM_MS" not in js and "__IDS__" not in js
        assert f"var IDS = {json.dumps(list(defaults_server._ARMED_BUTTONS))};" in js
        for event in ("visibilitychange", "mousemove", "keydown", "submit"):
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


class TestTradePage:
    """GET /trade: the saved live defaults, and the buttons that run the bot with them."""

    def test_it_shows_the_saved_defaults(self):
        saved = _save(LiveSettings(False, (0.0, 0.5), 0.4, 1.0, 0.2), source=DASHBOARD_NOTE)
        response = _get(_app(start_process=_Starter()), "/trade")
        assert response.status == 200
        page = _parse(response.body)
        assert "Trade with the saved live defaults?" in response.body
        assert f"Saved defaults: {saved.origin}" in response.body
        assert list(page.rows) == LABELS
        assert page.cells["k"] == ["k", "0.4"]
        for sentence in config.live_rule_warnings(saved):
            assert f'<p class="warn">Warning: {sentence}.</p>' in response.body
        assert page.forms == [("post", "/trade")]
        assert page.labels == _TRADE_LABELS
        assert [name for name, _ in page.hidden] == ["fingerprint", "nonce", "token"]
        assert ('<p class="money">Confirm and trade runs the live bot with these saved '
                "defaults on the production account: it places real orders.</p>") \
            in response.body
        assert str(config.LIVE_RUN_LOCK_FILE.absolute()) in response.body

    def test_no_defaults_saved_is_404_naming_how_to_save_them(self):
        response = _get(_app(), "/trade")
        assert response.status == 404
        assert "Save as live defaults…" in response.body
        assert "./start_dashboard.sh --seed in this checkout" in response.body
        assert "<form" not in response.body


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


class TestConfirmRefusals:
    """A proposal the server cannot save is refused with 400, naming why."""

    @pytest.mark.parametrize("query", [
        _query(foo="1"),                                  # unknown field
        _query() + "&k=0.8",                              # repeated
        _query(k=""),                                     # blank
        _query(source=""),                                # a blank source
        _query(tier_floors="maybe"),
        _query(add_to_held_pairs="maybe"), _query(add_to_held_pairs="true"),
        _query(add_to_held_pairs="1"), _query(add_to_held_pairs="On"),
        _query(k="nan"), _query(k="inf"), _query(k="0_8"), _query(k="1e-400"),
        _query(k="1e400"), _query(k="０.8"), _query(k=" 0.8"), _query(k="0"),
        _query(k="1.5"), _query(k="0x1"),
        _query(size_cap="0.33"), _query(same_title_size_cap="0"),
        _query(tag="Basketball"),                         # a tag without its category
        _query(category="any"), _query(category="Any"), _query(category=" Sports"),
        _query(category="Sports​"), _query(category="Sports", tag="Bask‮etball"),
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
        _query(nonce="0" * 32), _query(action="trade"),  # not proposal fields
        "k", "&&", _query() + "&" + "&".join(f"x{i}=1" for i in range(20)),
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


class TestRefusedFile:
    """A saved file that cannot be used blocks every read route with 409, no form."""

    def test_every_route_is_refused(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        form = _page_form(app, _query(k="0.75"))
        trade = _trade_form(app)
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        for response in (_get(app, "/confirm?" + _query()), _post(app, form),
                         _get(app, "/saved"), _get(app, "/trade"),
                         _post(app, trade, target="/trade")):
            assert response.status == 409
            assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body
            assert "<form" not in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == b"not json"
        assert app._start_process.calls == []


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
        again = _click(response.body, "confirm")
        assert dict(again)["fingerprint"] != dict(form)["fingerprint"]
        # A fresh nonce too, though the stale POST never used its own
        assert dict(again)["nonce"] != dict(form)["nonce"]
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
        assert "no rename today" in response.body and "Nothing was traded." in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_a_zero_floor_and_a_non_ascii_tag_round_trip(self):
        app = _app()
        form = _page_form(app, _query(spread_min="0", category="Deportes", tag="Fútbol"))
        assert ("tag", "Fútbol") in form
        assert _post(app, form).status == 303
        saved = config.read_saved_live_defaults()
        assert saved.spread_band == (0.0, 0.5) and saved.tags == ("Fútbol",)

    def test_nothing_to_save_writes_nothing(self):
        # No rendered page offers this (its save button is a placeholder when
        # nothing changes), so the test signs the request itself
        saved = _save(_BASE_SETTINGS)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        app = _app()
        params = {name: [value] for name, value in _BASE.items()}
        fingerprint, nonce = defaults_server._fingerprint(saved), "2" * 32
        form = [(name, values[0]) for name, values in params.items()]
        form += [("fingerprint", fingerprint), ("nonce", nonce),
                 ("token", app._token("confirm", fingerprint, nonce, params)),
                 ("action", "save")]
        response = _post(app, form)
        assert response.status == 200
        assert "nothing was written" in response.body
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        # The nonce stays unused: the same form answers the same way
        assert _post(app, form).status == 200

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
        assert "/trade" in page.links and "Trade with these defaults →" in response.body
        # The save is done: the page no longer speaks of writing
        assert "This writes" not in response.body and "click" not in response.body
        assert str(config.LIVE_DEFAULTS_FILE.absolute()) in response.body

    def test_a_warning_is_shown_as_a_note_on_what_was_saved(self):
        saved = _save(LiveSettings(False, (0.0, 0.5), 0.4, 1.0, 0.2))
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
        saved = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        assert defaults_server._proposal(params, saved)[0] == config.LIVE_DEFAULTS_SEED

    def test_a_seven_toggle_file_is_proposed_the_seed_with_adding_on(self):
        # A saved file that leaves the toggle out reads it as off; the seed
        # page proposes it on, a change shown in its row
        _save(replace(config.LIVE_DEFAULTS_SEED, add_to_held_pairs=False))
        assert '"add_to_held_pairs"' not in config.LIVE_DEFAULTS_FILE.read_text()
        response = _get(_app(), "/confirm?" + defaults_server._seed_query())
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "add to held pairs"]
        assert page.cells["add to held pairs"][1:3] == ["off", "on"]
        assert "1 of 8 settings change." in response.body

    def test_a_seed_link_that_leaves_out_adding_to_held_pairs_still_proposes_the_seed(self):
        # A seed link without the add_to_held_pairs field, opened while a file
        # with it off is saved: it proposes exactly the seed (the choice on),
        # shown as a change, and saves it; it is never refused for putting the
        # seed's note on other values
        _save(replace(config.LIVE_DEFAULTS_SEED, add_to_held_pairs=False))
        query = defaults_server._seed_query()
        stale = "&".join(part for part in query.split("&")
                         if not part.startswith("add_to_held_pairs="))
        assert stale != query
        params = defaults_server._params(stale)
        assert defaults_server._proposal(params, config.read_saved_live_defaults()) == (
            config.LIVE_DEFAULTS_SEED, config.LIVE_DEFAULTS_SEED_SOURCE)
        app = _app()
        response = _get(app, "/confirm?" + stale)
        assert response.status == 200
        page = _parse(response.body)
        assert [label for label, css in page.rows.items() if css == "changed"] == [
            "add to held pairs"]
        assert page.cells["add to held pairs"][1:3] == ["off", "on"]
        assert _post(app, _click(response.body, "confirm")).status == 303
        assert config.live_defaults() == config.LIVE_DEFAULTS_SEED
        # Control: the same link without the seed's note keeps what is saved
        plain = "&".join(part for part in stale.split("&") if not part.startswith("source="))
        assert defaults_server._proposal(
            defaults_server._params(plain),
            replace(config.LIVE_DEFAULTS_SEED, add_to_held_pairs=False),
        )[0].add_to_held_pairs is False

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
        params = {"k": ["0.80"], "tier_floors": ["off"], "token": ["x"], "category": ["A B"],
                  "action": ["trade"], "nonce": ["y"]}
        assert defaults_server._signed_text("confirm", "f" * 64, "n" * 32, params) == (
            "confirm\n" + "f" * 64 + "\n" + "n" * 32 + "\ncategory=A+B&k=0.80&tier_floors=off")
        assert defaults_server._signed_text("trade", "f" * 64, "n" * 32, {}) == (
            "trade\n" + "f" * 64 + "\n" + "n" * 32 + "\n")

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

    def test_every_response_carries_the_six_headers(self):
        headers = defaults_server._response_headers(defaults_server._Response(200, ""))
        assert [name for name, _ in headers] == [
            "Content-Type", "Cache-Control", "X-Content-Type-Options", "X-Frame-Options",
            "Referrer-Policy", "Content-Security-Policy"]
        values = dict(headers)
        assert values["Content-Type"] == "text/html; charset=utf-8"
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
    """The buttons of every page, in every state: fixed order, the real-money one last, placeholders inert."""

    @staticmethod
    def _check(body: str, labels: list[str], actions: tuple) -> _PageParser:
        """
        Check a page's buttons: order, placeholders, real buttons, and its one script.

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
        assert page.scripts == [defaults_server._CONFIRM_JS]
        assert len(page.forms) == 1
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
        # Save: only when something changes; no run ever blocks it
        assert (ids[1] == "confirm") is changes
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
    def test_the_trade_page(self, monkeypatch, state):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        self._state(app, state, monkeypatch)
        response = _get(app, "/trade")
        assert response.status == 200
        page = self._check(response.body, _TRADE_LABELS, defaults_server._TRADE_ACTIONS)
        ids = [b.get("id") for b in page.buttons]
        assert (ids[0] == "confirm-dry-run") is (state != "own")
        assert (ids[1] == "confirm-trade") is (state in ("free", "attention"))
        assert bool(page.checkboxes) is (state == "attention")

    def test_a_page_with_no_real_button_still_has_its_one_script(self):
        _save(_BASE_SETTINGS)
        page = self._check(_get(_app(), "/confirm?" + _query()).body, _CONFIRM_LABELS,
                           defaults_server._CONFIRM_ACTIONS)
        assert all("id" not in button for button in page.buttons)

    def test_every_other_page_has_no_form_button_or_script(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        _disk_run(result=_result())
        for target in ("/", "/saved", "/runs/0123456789abcdef", "/nope",
                       "/confirm?" + _query(k="nan"), "/runs/0000000000000000"):
            page = _parse(_get(app, target).body)
            assert page.forms == [] and page.buttons == [] and page.scripts == [], target


class TestTokenAndNonce:
    """Each page's token starts only its own page's actions, and each page's nonce works once."""

    def test_a_confirm_token_fails_on_the_trade_page_and_the_reverse(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        saved = config.read_saved_live_defaults()
        fingerprint, nonce = defaults_server._fingerprint(saved), "3" * 32
        confirm_token = app._token("confirm", fingerprint, nonce, {})
        form = [("fingerprint", fingerprint), ("nonce", nonce), ("token", confirm_token),
                ("action", "dry_run")]
        assert _post(app, form, target="/trade").status == 403
        params = {name: [value] for name, value in _BASE.items()}
        trade_token = app._token("trade", fingerprint, nonce, params)
        form = [(name, values[0]) for name, values in params.items()]
        form += [("fingerprint", fingerprint), ("nonce", nonce), ("token", trade_token),
                 ("action", "dry_run")]
        assert _post(app, form).status == 403
        assert app._start_process.calls == []

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

    def test_the_trade_page_posted_twice_starts_one_run(self):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        form = _trade_form(app, "confirm-dry-run")
        first = _post(app, form, target="/trade")
        second = _post(app, form, target="/trade")
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

    def test_trade_saves_then_starts_exactly_the_page_s_settings(self):
        starter = _Starter()
        app = _app(start_process=starter)
        form = _page_form(app, _query(k="0.75", category="Sports"), "confirm-trade")
        run_id = _run_id(_post(app, form))
        saved = config.read_saved_live_defaults()
        # Nothing was saved before, so adding to held pairs is the seed's (on)
        assert saved == LiveSettings(False, (0.0, 0.5), 0.75, 0.1, 0.2, ("Sports",),
                                     add_to_held_pairs=True)
        folder = _run_folder(run_id)
        [(argv, kwargs)] = starter.calls
        assert argv == [sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
                        *config.live_settings_argv(saved), "--result-file",
                        str(folder / "result.json")]
        # Adding to held pairs is spelled out, so the saved file never decides it
        assert "--add-to-held-pairs" in argv
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

    def test_the_trade_page_starts_the_saved_defaults(self):
        saved = _save(LiveSettings(False, (0.0, 0.5), 0.8, 0.15, 0.2), source=DASHBOARD_NOTE)
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        starter = _Starter()
        app = _app(start_process=starter)
        _run_id(_post(app, _trade_form(app), target="/trade"))
        [(argv, _)] = starter.calls
        assert argv[5:-2] == config.live_settings_argv(saved)
        assert "--dry-run" not in argv
        assert saved.origin in starter.run_json[0]["saved_note"]
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_the_trade_page_refuses_any_other_field(self):
        _save(_BASE_SETTINGS)
        app = _app(start_process=_Starter())
        form = _trade_form(app, "confirm-dry-run")
        assert _post(app, form + [("k", "0.75")], target="/trade").status == 400
        assert _post(app, form + [("action", "save")], target="/trade").status == 400
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
            page = _get(app, "/trade").body
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
        _run_id(_post(app, _trade_form(app, "confirm-dry-run"), target="/trade"))

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
        body = html.unescape(_get(_app(start_process=_Starter()), "/trade").body)
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
        trade = list(_parse(_get(app, "/trade").body).hidden)
        assert _post(app, trade + [("action", "trade")], target="/trade").status == 409
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert not config.LIVE_RUNS_DIR.exists()

    def test_a_dry_run_needs_saved_defaults(self):
        starter = _Starter()
        app = _app(start_process=starter)
        page = _parse(_get(app, "/confirm?" + _query()).body)
        response = _post(app, list(page.hidden) + [("action", "dry_run")])
        assert response.status == 409 and "save these first" in response.body
        assert starter.calls == [] and config.read_saved_live_defaults() is None

    def test_a_stale_trade_page_is_shown_again(self):
        _save(_BASE_SETTINGS)
        starter = _Starter()
        app = _app(start_process=starter)
        form = _trade_form(app)
        other = _save(LiveSettings(False, (0.0, 0.5), 0.75, 0.1, 0.2))
        response = _post(app, form, target="/trade")
        assert response.status == 409
        assert defaults_server._STALE_BANNER in response.body
        assert other.origin in response.body
        assert starter.calls == []
        _run_id(_post(app, _click(response.body, "confirm-trade"), target="/trade"))

    def test_the_trade_page_needs_saved_defaults(self):
        app = _app(start_process=_Starter())
        fingerprint, nonce = defaults_server._fingerprint(None), "4" * 32
        form = [("fingerprint", fingerprint), ("nonce", nonce),
                ("token", app._token("trade", fingerprint, nonce, {})), ("action", "dry_run")]
        response = _post(app, form, target="/trade")
        assert response.status == 409 and "No live defaults are saved" in response.body
        assert app._start_process.calls == []


class TestAttention:
    """After a real-money run that needed attention, a trade needs the box ticked."""

    def test_an_exit_20_result_raises_the_banner(self):
        _save(_BASE_SETTINGS)
        _attention_result()
        app = _app(start_process=_Starter())
        for target in ("/confirm?" + _query(), "/trade"):
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
        # The trade page asks the same
        app._runs.clear()
        trade = _trade_form(app)
        assert _post(app, trade, target="/trade").status == 409
        _run_id(_post(app, _trade_form(app, ack=True), target="/trade"))

    def test_a_dry_run_never_needs_it(self):
        _save(_BASE_SETTINGS)
        _attention_result()
        app = _app(start_process=_Starter())
        _run_id(_post(app, _page_form(app, _query(), "confirm-dry-run")))
        app._runs.clear()
        _run_id(_post(app, _trade_form(app, "confirm-dry-run"), target="/trade"))

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
        response = _get(app, "/trade")
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
        assert _post(app, _trade_form(app), target="/trade").status == 409
        assert starter.calls == []
        _disk_run("bbbbbbbbbbbbbbbb", started="20990101T160000Z",
                  result=_result(finished_at="2099-01-01T16:05:00Z"))
        assert app._attention() is None
        _run_id(_post(app, _trade_form(app), target="/trade"))

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
         "Trades need your attention", "Do not start another real-money run"),
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
        assert "<th>Status</th><th>Type</th><th>Market A</th><th>Leg A</th><th>Market B</th>" \
               "<th>Leg B</th><th>Cost incl. fees</th><th>Profit if won</th><th>Note</th>" in body
        assert "<td>30 × YES @ 0.2100</td>" in body and "<td>30 × NO @ 0.5000</td>" in body
        assert "<td>Rain by Oct 1 (RAIN-A)</td>" in body
        assert "<td>$21.90</td>" in body and "<td>orphan</td>" in body

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
            "Trades need your attention"

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
        assert "<script>" not in body.replace(
            f"<script>{defaults_server._CONFIRM_JS}</script>", "")
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
        status, _, body = _request(port, "GET", "/trade")
        assert status == 200
        page = _parse(body)
        [button] = [b for b in page.buttons if b.get("id") == "confirm-dry-run"]
        form = urlencode(page.hidden + [(button["name"], button["value"])]).encode("ascii")
        status, headers, _ = _request(port, "POST", "/trade", body=form, headers={
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

    def test_every_route_sends_the_six_headers(self, live_server):
        port = live_server.server_address[1]
        expected = dict(defaults_server._response_headers(defaults_server._Response(200, "")))
        answers = [
            _request(port, "GET", "/"),
            _request(port, "GET", "/confirm?" + _query()),
            _request(port, "GET", "/confirm?k"),
            _request(port, "GET", "/saved"),
            _request(port, "GET", "/trade"),
            _request(port, "GET", "/runs/0123456789abcdef"),
            _request(port, "GET", "/nope"),
            _request(port, "GET", "/", host="evil.test"),
            _request(port, "POST", "/confirm", body=b"", headers={"Content-Type": FORM}),
            _request(port, "POST", "/trade", body=b"", headers={"Content-Type": FORM}),
        ]
        assert [status for status, _, _ in answers] == [200, 200, 400, 404, 404, 404, 404,
                                                        403, 403, 403]
        for _, headers, _ in answers:
            for name, value in expected.items():
                assert headers[name] == value, name

    def test_checkout_answers_json_with_the_other_five_headers(self, live_server):
        port = live_server.server_address[1]
        status, headers, body = _request(port, "GET", "/checkout")
        assert status == 200
        assert json.loads(body) == {"project_root": str(config.PROJECT_ROOT.resolve()),
                                    "code": defaults_server._LOADED_CODE}
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        expected = dict(defaults_server._response_headers(defaults_server._Response(200, "")))
        for name in ("Cache-Control", "X-Content-Type-Options", "X-Frame-Options",
                     "Referrer-Policy", "Content-Security-Policy"):
            assert headers[name] == expected[name]
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
        assert answer_headers["X-Frame-Options"] == "DENY"

    def test_two_content_lengths_are_refused(self, live_server):
        headers = {"Content-Type": FORM, "Content-Length": "0"}
        status, _, _ = _request(live_server.server_address[1], "POST", "/confirm", body=b"",
                                headers=headers)
        assert status == 400

    def test_a_path_over_the_limit_is_refused(self, live_server):
        target = "/confirm?" + "k=" + "1" * config.DEFAULTS_SERVER_MAX_REQUEST_BYTES
        assert _request(live_server.server_address[1], "GET", target)[0] == 413

    @pytest.mark.parametrize("method", ["PUT", "HEAD"])
    def test_other_methods_get_501_with_the_six_headers(self, live_server, method):
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
    def test_a_request_the_http_layer_refuses_still_gets_the_six_headers(
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
        assert f"trades them at http://127.0.0.1:{PORT}/trade" in log
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

    def test_it_opens_the_dashboard_when_there_is_one(self, run_main, tmp_path):
        index = f"http://127.0.0.1:{PORT}/"
        log = run_main["run"]([])
        # No dashboard: the server's own index
        assert run_main["opened"] == [index]
        dashboard = tmp_path / config.DASHBOARD_FILENAME
        assert (f"No backtest dashboard at {dashboard}: run python3 -m "
                "kalshi_betting.backtest to build one, or start from the seed values with "
                f"--seed; opening {index}") in log
        dashboard.write_text(_DASHBOARD_WITH_BUTTONS, encoding="utf-8")
        log = run_main["run"]([])
        assert run_main["opened"] == [index, dashboard.as_uri()]
        assert (f"Open {dashboard} and use its filter bar's Save as live defaults… or Trade "
                "using defaults… button") in log
        assert "WARNING" not in log

    def test_a_dashboard_built_before_the_buttons_opens_the_index_with_a_warning(
            self, run_main, tmp_path):
        dashboard = tmp_path / config.DASHBOARD_FILENAME
        dashboard.write_text(_DASHBOARD_BEFORE_BUTTONS, encoding="utf-8")
        log = run_main["run"]([])
        base = f"http://127.0.0.1:{PORT}"
        assert run_main["opened"] == [f"{base}/"]
        [line] = [line for line in log.splitlines() if "WARNING" in line]
        assert line.endswith(
            "This dashboard was built before the Save/Trade buttons: rebuild it with "
            "python3 -m kalshi_betting.backtest and the --start-date you built it from; "
            f"until then use {base}/trade (opening {base}/)")

    def test_the_buttons_count_only_within_the_scanned_part(self, run_main, tmp_path,
                                                             monkeypatch):
        # Only the first DASHBOARD_MARKER_SCAN_BYTES are read: the Trade link's
        # id past them reads as a page without the buttons
        monkeypatch.setattr(defaults_server, "DASHBOARD_MARKER_SCAN_BYTES", 64)
        dashboard = tmp_path / config.DASHBOARD_FILENAME
        index = f"http://127.0.0.1:{PORT}/"
        marker = defaults_server._DASHBOARD_MARKER
        dashboard.write_bytes(b"x" * (64 - len(marker)) + marker)
        run_main["run"]([])
        dashboard.write_bytes(b"x" * (65 - len(marker)) + marker)
        run_main["run"]([])
        assert run_main["opened"] == [dashboard.as_uri(), index]

    def test_a_dashboard_that_cannot_be_read_opens_the_index_with_a_warning(
            self, run_main, tmp_path):
        dashboard = tmp_path / config.DASHBOARD_FILENAME
        dashboard.mkdir()
        log = run_main["run"]([])
        assert run_main["opened"] == [f"http://127.0.0.1:{PORT}/"]
        [line] = [line for line in log.splitlines() if "WARNING" in line]
        assert f"The backtest dashboard at {dashboard} could not be read (IsADirectoryError" \
            in line

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
        ([], lambda base: [f"{base}/"]),
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

    def test_without_seed_it_opens_the_dashboard_or_the_index(self, live_server, busy_port):
        port = live_server.server_address[1]
        live_server.defaults_app = defaults_server._App(port)
        busy_port["hold"](port)
        busy_port["run"]([])
        dashboard = config.PROJECT_ROOT / config.DASHBOARD_FILENAME
        dashboard.write_text(_DASHBOARD_WITH_BUTTONS, encoding="utf-8")
        busy_port["run"]([])
        dashboard.write_text(_DASHBOARD_BEFORE_BUTTONS, encoding="utf-8")
        busy_port["run"]([])
        index = f"http://127.0.0.1:{port}/"
        assert busy_port["opened"] == [index, dashboard.as_uri(), index]

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
        "secrets", "subprocess", "sys", "urllib", "webbrowser"})

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
