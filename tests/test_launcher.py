"""
File: test_launcher.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for start_dashboard.sh, the script at the repo root that starts the
    dashboard's two servers. It parses as bash and git records it as
    executable. It takes only --seed, --no-browser, -h and --help: help is
    printed and anything else refused (exit 2) before anything starts. It
    runs from its own folder (a folder whose path has a space included,
    whatever folder it is started from, and whatever CDPATH says), checks
    that its Python can import the live bot and the live dashboard from that
    folder, then starts `<python> -m kalshi_betting.live_dashboard` (with
    --no-browser when given --seed or --no-browser) and, once that has run
    for 2 s, `<python> -m kalshi_betting.defaults_server` with the flags it
    was given, plus --no-browser unless one of them says otherwise, since
    the live dashboard opens the page. Both run in the background.

    A live dashboard that stops with an error within those 2 s leaves the
    defaults server to open its own page, with a warning that fits the
    flags; one that stops with 0 (this checkout's was already running) adds
    no warning and the server still gets --no-browser; one that stops with
    an error later is named in a warning while the server keeps running. A
    defaults server that fails ends the script with its exit code and stops
    the live dashboard with a Ctrl-C; one that returns 0 at once leaves the
    script running the live dashboard. Ctrl-C, SIGTERM and SIGHUP to the
    script's process group stop both (exit 130, 143, 129), each server
    stopping on its own; the same signals sent to the script alone stop both
    too, through a Ctrl-C the script sends each; a Ctrl-C before the
    defaults server starts stops the live dashboard even while it still
    ignores Ctrl-C; and the script signals no server that has already ended,
    so a process number reused since can never be signalled. A Python that
    is missing, that cannot import the bot, or that imports it from another
    folder (a link to the script, PYTHONSAFEPATH) stops it with exit 1 and a
    message saying what to do. It names no user's path.

Dependencies:
    Imports nothing from the package: the script is run as a program. Each
    test copies it into its own tmp_path and runs it with KALSHI_PYTHON (or,
    once, the PATH) pointing at a stand-in for Python, a small sh script that
    records each call and starts neither server. As a server that keeps
    running it becomes a small program under the Python running the tests
    that acts as both servers do with Ctrl-C (it turns Ctrl-C back on,
    since a background job starts with it ignored, and exits 0 on it) and
    notes how it stopped. One test runs the script's own import check under
    the Python running the tests, from the repo root, which imports the live
    bot and the live dashboard but starts nothing.

Notes:
    Skipped where there is no bash. macOS ships bash 3.2, and these tests
    run under whatever bash is first on the PATH, which is that one on a
    stock Mac. The script gives the live dashboard 2 s before it starts the
    defaults server, so each test that gets that far takes at least 2 s.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

# The repo root, where the script lives
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "start_dashboard.sh"
_BASH = shutil.which("bash")

# The Python code of the script's import check, as the script spells it
_CHECK = re.search(r"^check='(.*)'$", _SCRIPT.read_text(encoding="utf-8"), re.M).group(1)

pytestmark = pytest.mark.skipif(_BASH is None, reason="no bash on this machine")

# The exit code the stand-in gives as the defaults server unless a test says
# otherwise, so a test can tell its exit code reached the caller (the script
# ends with it when it is not 0)
_STUB_SERVER_EXIT = 7

# The two servers the script starts, as `python -m` names them
_LIVE = "kalshi_betting.live_dashboard"
_SERVER = "kalshi_betting.defaults_server"

# The warnings the script prints when the live dashboard stopped with an
# error: at once (by the flags given), and later
_LIVE_FAILED = "start_dashboard.sh: the live dashboard did not start (its message is above); "
_LIVE_FAILED_INSTEAD = {
    (): "only the defaults server runs, and it opens the backtest page itself",
    ("--seed",): "only the defaults server runs (it opens the seed values' confirmation page)",
    ("--no-browser",): "only the defaults server runs",
}
_LIVE_STOPPED = "start_dashboard.sh: the live dashboard stopped with exit status {status} " \
                "(its message is above)"
_KEEPS = "; the defaults server keeps running (its address is in its log above)"

# A server that keeps running, as the stand-in plays it: a program under the
# Python running the tests that runs for argv[1] seconds. Like both servers
# it turns Ctrl-C back on (a background job starts with it ignored), argv[2]
# seconds after it starts (a server still importing ignores it until then),
# and exits 0 on it. It notes in the file argv[3], one word per line, READY
# once Ctrl-C would stop it, then INT when Ctrl-C stopped it or TERM when
# SIGTERM did (it then exits 143), or END when its time ran out.
_SERVE = """import signal, sys, time
seconds, ignored, stops = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]

def note(word):
    with open(stops, "a", encoding="utf-8") as out:
        out.write(word + "\\n")

def on_term(signum, frame):
    note("TERM")
    sys.exit(143)

signal.signal(signal.SIGTERM, on_term)
time.sleep(ignored)
signal.signal(signal.SIGINT, signal.default_int_handler)
note("READY")
try:
    time.sleep(seconds)
except KeyboardInterrupt:
    note("INT")
    sys.exit(0)
note("END")
"""

# The stand-in for Python. Every call first records itself as one line in a
# file of its own, "$STUB_RECORD.<its process id>": its process id, its
# folder and each argument, each followed by a NUL byte. Called with -c (the
# script's import check) it then, like Python, reports where it imported the
# package from: the current folder when that folder holds a kalshi_betting
# package and PYTHONSAFEPATH is unset (the current folder comes first on
# Python's import path), else the installed copy STUB_INSTALLED_ROOT names.
# When STUB_IMPORT_ERROR is set it prints a Python-style error and fails
# instead. Called as the live dashboard it waits STUB_LIVE_DELAY seconds (0)
# and exits STUB_LIVE_EXIT when that is set; else it becomes _SERVE (under
# STUB_PYTHON) for STUB_LIVE_SLEEP seconds (30), ignoring Ctrl-C for its first
# STUB_LIVE_IMPORT seconds (0). Called as the defaults server it becomes
# _SERVE for STUB_SERVER_SLEEP seconds when that is set, else exits
# STUB_SERVER_EXIT (0). _SERVE notes how it stopped in
# "$STUB_RECORD-stops.<its process id>". It writes only to those files.
_STUB = """#!/bin/sh
{ printf '%s\\000' "$$" "$(pwd -P)" "$@"; printf '\\n'; } >> "$STUB_RECORD.$$"
if [ "$1" = "-c" ]; then
  if [ -n "${STUB_IMPORT_ERROR:-}" ]; then
    echo "Traceback (most recent call last):" >&2
    echo "$STUB_IMPORT_ERROR" >&2
    exit 1
  fi
  echo "a warning on stderr" >&2
  if [ -z "${PYTHONSAFEPATH:-}" ] && [ -d kalshi_betting ]; then
    echo "kalshi_betting imported from $(pwd -P)"
  else
    echo "kalshi_betting imported from $STUB_INSTALLED_ROOT"
  fi
  exit 0
fi
case "$2" in
  kalshi_betting.live_dashboard)
    if [ -n "${STUB_LIVE_EXIT:-}" ]; then
      sleep "${STUB_LIVE_DELAY:-0}"
      exit "$STUB_LIVE_EXIT"
    fi
    exec "$STUB_PYTHON" -c "$STUB_SERVE" "${STUB_LIVE_SLEEP:-30}" "${STUB_LIVE_IMPORT:-0}" \\
      "$STUB_RECORD-stops.$$"
    ;;
  kalshi_betting.defaults_server)
    if [ -n "${STUB_SERVER_SLEEP:-}" ]; then
      exec "$STUB_PYTHON" -c "$STUB_SERVE" "$STUB_SERVER_SLEEP" 0 "$STUB_RECORD-stops.$$"
    fi
    exit "${STUB_SERVER_EXIT:-0}"
    ;;
esac
exit 99
"""


@dataclass(frozen=True)
class _Call:
    """
    One call of the stand-in for Python.

    Attributes:
        pid (int): Its process id (the same after it became _SERVE).
        cwd (str): The folder it ran in, links resolved.
        args (tuple[str, ...]): Its arguments, as given.
    """
    pid: int
    cwd: str
    args: tuple[str, ...]


def _write_stub(folder: Path, name: str = "fake-python") -> Path:
    """
    Write the stand-in for Python into a folder, executable.

    Args:
        folder (Path): Where to write it (created if missing).
        name (str): Its file name.

    Returns:
        Path: The stand-in.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stub = folder / name
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _checkout(tmp_path: Path) -> Path:
    """
    Copy the script into a stand-in checkout whose path has a space in it.

    The checkout gets an empty kalshi_betting folder, which the stand-in for
    Python reads as the package it imports from the current folder.

    Args:
        tmp_path (Path): The test's directory.

    Returns:
        Path: The copied script, executable, in tmp_path / "my checkout".
    """
    folder = tmp_path / "my checkout"
    folder.mkdir()
    (folder / "kalshi_betting").mkdir()
    script = folder / _SCRIPT.name
    shutil.copy2(_SCRIPT, script)
    script.chmod(0o755)
    return script


def _start(command: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    """
    Start the script in a session of its own, so a test can signal its whole process group.

    Args:
        command (list[str]): The command line.
        cwd (Path): The folder to start it from.
        env (dict): Its environment.

    Returns:
        subprocess.Popen: The running script; its stdout and stderr are pipes.
    """
    return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)


def _finish(process: subprocess.Popen, timeout: float = 60) -> str:
    """
    Wait for a started script to end, killing its process group if it does not.

    Args:
        process (subprocess.Popen): The script, from _start.
        timeout (float): The seconds to wait.

    Returns:
        str: What it wrote to stderr.

    Raises:
        subprocess.TimeoutExpired: When it had not ended in time (it is then
            killed, with every process in its group).
    """
    try:
        _, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        process.communicate()
        raise
    return err


def _kill_group(process: subprocess.Popen) -> None:
    """
    Kill the script's process group, the stand-ins in it included, if any of it is left.

    Args:
        process (subprocess.Popen): The script, from _start.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run(command: list[str], cwd: Path, env: dict, timeout: float = 60
         ) -> tuple[subprocess.Popen, str]:
    """
    Run the script and wait for it.

    Args:
        command (list[str]): The command line.
        cwd (Path): The folder to start it from.
        env (dict): Its environment.
        timeout (float): The seconds to wait (_finish).

    Returns:
        tuple[subprocess.Popen, str]: The finished process (its returncode)
            and what it wrote to stderr.
    """
    process = _start(command, cwd, env)
    return process, _finish(process, timeout)


def _env(record: Path, **extra) -> dict:
    """
    The script's environment: this one's, less the variables that steer it, plus the stand-in's.

    KALSHI_PYTHON, CDPATH, PYTHONSAFEPATH and the stand-in's own variables
    are left out, so a test sets only the ones it means to; the stand-in's
    defaults server exits _STUB_SERVER_EXIT and its live dashboard runs 300 s
    unless a test says otherwise. A live dashboard the script fails to stop
    therefore holds the script's output open well past any test's wait.

    Args:
        record (Path): The stand-in's record files are this path plus "."
            and a process id; its installed copy of the package is said to
            live beside it, in "installed checkout".
        **extra: More variables to set (they win over the defaults here).

    Returns:
        dict: The environment.
    """
    env = {name: value for name, value in os.environ.items()
           if name not in ("KALSHI_PYTHON", "CDPATH", "PYTHONSAFEPATH")
           and not name.startswith("STUB_")}
    env["STUB_RECORD"] = str(record)
    env["STUB_INSTALLED_ROOT"] = str(record.parent / "installed checkout")
    env["STUB_PYTHON"] = sys.executable
    env["STUB_SERVE"] = _SERVE
    env["STUB_SERVER_EXIT"] = str(_STUB_SERVER_EXIT)
    env["STUB_LIVE_SLEEP"] = "300"
    env.update(extra)
    return env


def _calls(record: Path) -> list[_Call]:
    """
    Read every call the stand-in recorded.

    Args:
        record (Path): The path the stand-in's record files start with.

    Returns:
        list[_Call]: One per call, in no particular order; [] when there was none.
    """
    out: list[_Call] = []
    for path in record.parent.glob(f"{record.name}.*"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\0")
            assert fields[-1] == "", line                   # each field ends with a NUL
            pid, cwd, *args = fields[:-1]
            out.append(_Call(int(pid), cwd, tuple(args)))
    return out


def _seen(record: Path) -> set[tuple[str, tuple[str, ...]]]:
    """
    The calls the stand-in recorded, as a set of (folder, arguments).

    Args:
        record (Path): The path the stand-in's record files start with.

    Returns:
        set[tuple[str, tuple[str, ...]]]: The calls.
    """
    return {(call.cwd, call.args) for call in _calls(record)}


def _expected(folder: Path, live: list[str] | None, server: list[str] | None
              ) -> set[tuple[str, tuple[str, ...]]]:
    """
    The calls a run should make: the import check, then the live dashboard and the defaults server.

    Args:
        folder (Path): The folder every call runs in.
        live (list[str] | None): The live dashboard's arguments, or None when not started.
        server (list[str] | None): The defaults server's arguments, or None when not started.

    Returns:
        set[tuple[str, tuple[str, ...]]]: The calls, as _seen gives them.
    """
    here = str(folder.resolve())
    out = {(here, ("-c", _CHECK))}
    if live is not None:
        out.add((here, ("-m", _LIVE, *live)))
    if server is not None:
        out.add((here, ("-m", _SERVER, *server)))
    return out


def _pid_of(record: Path, module: str) -> int:
    """
    The process id of the stand-in's call as one server.

    Args:
        record (Path): The path the stand-in's record files start with.
        module (str): _LIVE or _SERVER.

    Returns:
        int: Its process id.
    """
    pids = [call.pid for call in _calls(record) if call.args[:2] == ("-m", module)]
    assert len(pids) == 1, pids
    return pids[0]


def _stops(record: Path, pid: int) -> list[str]:
    """
    What a server the stand-in played noted: READY, then how it stopped.

    Args:
        record (Path): The path the stand-in's record files start with.
        pid (int): The server's process id.

    Returns:
        list[str]: Its notes in order; [] when it noted nothing.
    """
    path = record.parent / f"{record.name}-stops.{pid}"
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def _wait_until(condition, timeout: float = 20) -> bool:
    """
    Wait for a condition to hold, checking every 50 ms.

    Args:
        condition (Callable[[], bool]): The condition.
        timeout (float): The seconds to wait.

    Returns:
        bool: Whether it held in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _gone(pid: int) -> bool:
    """
    Whether a process no longer exists.

    Args:
        pid (int): The process.

    Returns:
        bool: True once no process has that id.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _started(record: Path, module: str) -> bool:
    """
    Whether the stand-in has recorded a call as one server.

    Args:
        record (Path): The path the stand-in's record files start with.
        module (str): _LIVE or _SERVER.

    Returns:
        bool: True once it has.
    """
    return any(call.args[:2] == ("-m", module) for call in _calls(record))


def _ready(record: Path, module: str) -> bool:
    """
    Whether one server the stand-in plays has started and would now stop on Ctrl-C.

    Args:
        record (Path): The path the stand-in's record files start with.
        module (str): _LIVE or _SERVER.

    Returns:
        bool: True once it noted READY.
    """
    return _started(record, module) and "READY" in _stops(record, _pid_of(record, module))


def test_the_script_parses_as_bash():
    result = subprocess.run([_BASH, "-n", str(_SCRIPT)], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stderr


def test_git_records_it_as_executable():
    assert os.access(_SCRIPT, os.X_OK)
    git = shutil.which("git")
    if git is None:
        pytest.skip("no git on this machine")
    listed = subprocess.run([git, "ls-files", "-s", "--", _SCRIPT.name], cwd=_ROOT,
                            capture_output=True, text=True, timeout=30)
    if listed.returncode != 0 or not listed.stdout.strip():
        pytest.skip("not a git checkout, or the script is not tracked here")
    assert listed.stdout.split()[0] == "100755"


# How the script is started: by its full path, by a path relative to the
# folder above its checkout, and through bash by that relative path
_INVOCATIONS = {
    "full path": lambda script: (str(script), script.parent.parent / "elsewhere"),
    "relative path": lambda script: (f"{script.parent.name}/{script.name}",
                                     script.parent.parent),
    "through bash": lambda script: ([_BASH, f"{script.parent.name}/{script.name}"],
                                    script.parent.parent),
}


@pytest.mark.parametrize("how", list(_INVOCATIONS))
def test_it_starts_both_servers_from_its_own_folder(tmp_path, how):
    script = _checkout(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    start, cwd = _INVOCATIONS[how](script)
    command = start if isinstance(start, list) else [start]
    arguments = ["--seed", "--no-browser"]
    process, err = _run(command + arguments, cwd, _env(record, KALSHI_PYTHON=str(stub)))
    # The defaults server's exit code (not 0) is the script's
    assert process.returncode == _STUB_SERVER_EXIT, err
    # The import check, the live dashboard (--no-browser, once) and the
    # defaults server with the flags as given (nothing added), all from the
    # script's own folder
    assert _seen(record) == _expected(script.parent, ["--no-browser"], arguments)
    # Each in a process of its own, none of them the script itself
    pids = [call.pid for call in _calls(record)]
    assert len(set(pids)) == 3 and process.pid not in pids


def test_with_no_arguments_the_server_gets_no_browser_and_the_live_dashboard_nothing(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script)], tmp_path, _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == _STUB_SERVER_EXIT, err
    # The live dashboard opens the page, so the defaults server opens nothing
    assert _seen(record) == _expected(script.parent, [], ["--no-browser"])
    assert _LIVE_FAILED not in " ".join(err.split())


def test_with_seed_only_the_server_opens_a_page(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script), "--seed"], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == _STUB_SERVER_EXIT, err
    # The seed confirmation page is what opens: the live dashboard opens nothing
    assert _seen(record) == _expected(script.parent, ["--no-browser"], ["--seed"])


def test_with_no_browser_neither_opens_a_page(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script), "--no-browser"], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == _STUB_SERVER_EXIT, err
    assert _seen(record) == _expected(script.parent, ["--no-browser"], ["--no-browser"])


@pytest.mark.parametrize("arguments", [["-h"], ["--help"], ["--seed", "--help"]])
def test_help_is_printed_and_nothing_starts(tmp_path, arguments):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    result = subprocess.run([str(script), *arguments], cwd=tmp_path, capture_output=True,
                            text=True, timeout=30, env=_env(record, KALSHI_PYTHON=str(stub)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: start_dashboard.sh [--seed] [--no-browser]\n")
    assert "--seed" in result.stdout and "--no-browser" in result.stdout
    assert result.stderr == ""
    # Not even the import check ran
    assert _calls(record) == []


# Arguments the servers would read some other way, or refuse only after the
# live dashboard had already opened a page: each is refused before anything
# starts. --se and --no-b are abbreviations Python's argparse would accept.
@pytest.mark.parametrize("argument", ["--se", "--no-b", "--seed=1", "-s", "--help=1",
                                      "two words", "--x=$HOME", "*", ""])
def test_any_other_argument_is_refused_before_anything_starts(tmp_path, argument):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script), "--seed", argument], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == 2
    assert err == (f"start_dashboard.sh: unknown argument '{argument}' — use --seed, "
                   "--no-browser or --help\n")
    assert _calls(record) == []


def test_without_kalshi_python_it_runs_python3(tmp_path):
    script = _checkout(tmp_path)
    stubs = tmp_path / "bin"
    _write_stub(stubs, "python3")
    record = tmp_path / "record.txt"
    env = _env(record)
    # The stand-in named python3 comes first on the PATH
    env["PATH"] = f"{stubs}{os.pathsep}{env.get('PATH', '')}"
    process, err = _run([str(script), "--seed"], tmp_path, env)
    assert process.returncode == _STUB_SERVER_EXIT, err
    assert _seen(record) == _expected(script.parent, ["--no-browser"], ["--seed"])


@pytest.mark.parametrize("arguments", [[], ["--seed"], ["--no-browser"]])
def test_a_live_dashboard_that_fails_at_once_leaves_the_page_to_the_server(tmp_path,
                                                                            arguments):
    # Its port held by another program: it exits 2 at once, so the defaults
    # server is given only the flags and opens its own start page, if any
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    started = time.monotonic()
    process, err = _run([str(script), *arguments], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), STUB_LIVE_EXIT="2",
                             STUB_SERVER_EXIT="0"))
    assert process.returncode == 0, err
    live = ["--no-browser"] if arguments else []
    assert _seen(record) == _expected(script.parent, live, arguments)
    assert _LIVE_FAILED + _LIVE_FAILED_INSTEAD[tuple(arguments)] in " ".join(err.split())
    # It did not wait out the 2 s once the live dashboard had stopped
    assert time.monotonic() - started < 15
    # The live dashboard, already waited for, is not waited for again out loud
    assert "not a child" not in err


def test_a_live_dashboard_already_running_keeps_the_server_quiet(tmp_path):
    # This checkout's live dashboard already runs: a second one opens its
    # page and exits 0 at once, so the defaults server still opens nothing
    # and no warning is printed
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script)], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), STUB_LIVE_EXIT="0",
                             STUB_SERVER_EXIT="0"))
    assert process.returncode == 0, err
    assert _seen(record) == _expected(script.parent, [], ["--no-browser"])
    assert "start_dashboard.sh: the live dashboard" not in err
    assert "not a child" not in err


def test_a_live_dashboard_that_fails_later_is_named_and_the_server_keeps_running(tmp_path):
    # It stops with an error after the 2 s (a port holder that does not
    # answer, say): the defaults server, already told to open nothing, keeps
    # running, and a warning names the live dashboard's exit status
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_LIVE_EXIT="2",
                          STUB_LIVE_DELAY="3", STUB_SERVER_SLEEP="60"))
    try:
        assert _wait_until(lambda: _ready(record, _SERVER))
        live, server = _pid_of(record, _LIVE), _pid_of(record, _SERVER)
        assert _wait_until(lambda: _gone(live))
        time.sleep(1.5)                        # the script has looked again
        assert process.poll() is None and not _gone(server)
        os.killpg(process.pid, signal.SIGINT)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == 130, err
    assert _seen(record) == _expected(script.parent, [], ["--no-browser"])
    assert _LIVE_STOPPED.format(status=2) + _KEEPS in " ".join(err.split())
    assert _LIVE_FAILED not in " ".join(err.split())
    assert _stops(record, server) == ["READY", "INT"]


def test_a_server_that_returns_at_once_leaves_it_running_the_live_dashboard(tmp_path):
    # This checkout's defaults server already runs, so the new one returns 0
    # at once: the script keeps the live dashboard going, and ends when it ends
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_SERVER_EXIT="0"))
    try:
        assert _wait_until(lambda: _started(record, _SERVER))
        time.sleep(1.5)
        assert process.poll() is None
        live = _pid_of(record, _LIVE)
        assert not _gone(live)
        os.kill(live, signal.SIGTERM)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == 0, err
    assert _seen(record) == _expected(script.parent, [], ["--no-browser"])
    # Stopped by something else, the live dashboard is named; no server runs any more
    flat = " ".join(err.split())
    assert _LIVE_STOPPED.format(status=143) in flat and _KEEPS not in flat


def test_a_failing_server_ends_it_with_its_code_and_stops_the_live_dashboard(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script)], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), STUB_SERVER_EXIT="3"),
                        timeout=20)
    assert process.returncode == 3, err
    # The live dashboard (300 s, holding the script's output open) was
    # stopped on the way out, by a Ctrl-C it turned into a clean stop
    live = _pid_of(record, _LIVE)
    assert _wait_until(lambda: _gone(live), timeout=10)
    assert _stops(record, live) == ["READY", "INT"]


def _both_serving(tmp_path: Path) -> tuple[subprocess.Popen, Path, int, int]:
    """
    Start the script with both servers serving, and wait until each would stop on Ctrl-C.

    Args:
        tmp_path (Path): The test's directory.

    Returns:
        tuple[subprocess.Popen, Path, int, int]: The script, the stand-in's
            record path, and the live dashboard's and the defaults server's
            process ids.
    """
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_SERVER_SLEEP="60"))
    try:
        assert _wait_until(lambda: _ready(record, _LIVE) and _ready(record, _SERVER))
        live, server = _pid_of(record, _LIVE), _pid_of(record, _SERVER)
        assert process.poll() is None and not _gone(live) and not _gone(server)
    except BaseException:
        _kill_group(process)
        raise
    return process, record, live, server


def test_ctrl_c_stops_both(tmp_path):
    # Ctrl-C reaches the whole process group: each server stops on its own,
    # and the script, after giving them time to, sends neither anything
    process, record, live, server = _both_serving(tmp_path)
    try:
        os.killpg(process.pid, signal.SIGINT)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == 130, err
    assert _wait_until(lambda: _gone(live) and _gone(server), timeout=10)
    assert _stops(record, live) == ["READY", "INT"]
    assert _stops(record, server) == ["READY", "INT"]


@pytest.mark.parametrize("signum, code", [(signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_sigterm_or_sighup_to_the_group_stops_both(tmp_path, signum, code):
    # A closed terminal sends SIGHUP to the whole group; `kill -- -<group>`
    # SIGTERM: the servers stop by it, and the script exits 128 + its number
    process, _, live, server = _both_serving(tmp_path)
    try:
        os.killpg(process.pid, signum)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == code, err
    assert _wait_until(lambda: _gone(live) and _gone(server), timeout=10)


@pytest.mark.parametrize("signum, code",
                         [(signal.SIGINT, 130), (signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_a_signal_to_the_script_alone_stops_both(tmp_path, signum, code):
    # `kill <script's pid>`: the servers get nothing from it, so the script
    # sends each a Ctrl-C, and both stop cleanly
    process, record, live, server = _both_serving(tmp_path)
    try:
        os.kill(process.pid, signum)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == code, err
    assert _wait_until(lambda: _gone(live) and _gone(server), timeout=10)
    assert _stops(record, live) == ["READY", "INT"]
    assert _stops(record, server) == ["READY", "INT"]


def test_ctrl_c_before_the_server_starts_stops_the_live_dashboard(tmp_path):
    # Ctrl-C in the first 2 s, while the live dashboard still ignores it (a
    # background job starts with Ctrl-C ignored, and the real one turns it
    # back on only once imported): the defaults server never starts, and the
    # script stops the live dashboard itself
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_LIVE_IMPORT="2.5"))
    try:
        assert _wait_until(lambda: _started(record, _LIVE))
        live = _pid_of(record, _LIVE)
        time.sleep(0.5)
        assert _stops(record, live) == []
        os.killpg(process.pid, signal.SIGINT)
        err = _finish(process, timeout=30)
    finally:
        _kill_group(process)
    assert process.returncode == 130, err
    assert not _started(record, _SERVER)
    assert _wait_until(lambda: _gone(live), timeout=10)


def test_ctrl_c_with_only_the_live_dashboard_left_lets_it_stop_itself(tmp_path):
    # The defaults server returned 0 at once: Ctrl-C reaches the live
    # dashboard, which stops on its own, and the script sends it nothing
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_SERVER_EXIT="0"))
    try:
        assert _wait_until(lambda: _ready(record, _LIVE) and _started(record, _SERVER))
        time.sleep(1.5)                        # the script has seen the server end
        live = _pid_of(record, _LIVE)
        assert process.poll() is None and not _gone(live)
        os.killpg(process.pid, signal.SIGINT)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == 130, err
    assert _wait_until(lambda: _gone(live), timeout=10)
    assert _stops(record, live) == ["READY", "INT"]


def test_a_live_dashboard_that_has_ended_is_never_signalled(tmp_path):
    # It ended at once (this checkout's was already running), so its process
    # number may since belong to another program: when a SIGTERM to the
    # script alone stops it, it signals the defaults server and nothing that
    # names the live dashboard. Traced with bash -x, every command the script
    # runs is in its stderr.
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process = _start([_BASH, "-x", str(script)], tmp_path,
                     _env(record, KALSHI_PYTHON=str(stub), STUB_LIVE_EXIT="0",
                          STUB_SERVER_SLEEP="60"))
    try:
        assert _wait_until(lambda: _ready(record, _SERVER))
        live, server = _pid_of(record, _LIVE), _pid_of(record, _SERVER)
        assert _gone(live)
        os.kill(process.pid, signal.SIGTERM)
        err = _finish(process, timeout=20)
    finally:
        _kill_group(process)
    assert process.returncode == 143, err
    kills = [line for line in err.splitlines() if re.match(r"^\++ kill\b", line)]
    assert kills and all(re.search(rf"\b{server}\b", line) for line in kills), kills
    assert not [line for line in kills if re.search(rf"\b{live}\b", line)], kills
    assert _wait_until(lambda: _gone(server), timeout=10)
    assert _stops(record, server) == ["READY", "INT"]


def test_a_python_that_cannot_import_the_bot_stops_it(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    error = "ModuleNotFoundError: No module named 'kalshi_python_sync'"
    process, err = _run([str(script), "--seed"], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), STUB_IMPORT_ERROR=error))
    assert process.returncode == 1
    assert (f"start_dashboard.sh: {stub} cannot import the live bot and the live dashboard "
            "(the Kalshi SDK, tabulate, openpyxl, pandas, numpy, plotly, yfinance)"
            in " ".join(err.split()))
    assert 'pip install -e ".[dev]"' in err and "KALSHI_PYTHON" in err
    # The error's last line is shown, so the missing piece is named
    assert err.rstrip().endswith(error)
    # Neither server was started
    assert _seen(record) == _expected(script.parent, None, None)


def test_a_missing_python_stops_it(tmp_path):
    script = _checkout(tmp_path)
    record = tmp_path / "record.txt"
    missing = tmp_path / "no such python"
    process, err = _run([str(script)], tmp_path, _env(record, KALSHI_PYTHON=str(missing)))
    assert process.returncode == 1
    assert (f"start_dashboard.sh: {missing} not found — set KALSHI_PYTHON to the Python "
            "this package is installed in") in err
    assert _calls(record) == []


def test_cdpath_does_not_move_it_to_a_folder_of_the_same_name(tmp_path):
    # bash's cd looks a relative folder up in CDPATH first, so without the
    # script's `unset CDPATH` it would land in the decoy
    script = _checkout(tmp_path)
    decoy = tmp_path / "decoy" / script.parent.name
    (decoy / "kalshi_betting").mkdir(parents=True)
    shutil.copy2(script, decoy / script.name)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    command = [_BASH, f"{script.parent.name}/{script.name}", "--seed"]
    process, err = _run(command, tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), CDPATH=str(tmp_path / "decoy")))
    assert process.returncode == _STUB_SERVER_EXIT, err
    assert _seen(record) == _expected(script.parent, ["--no-browser"], ["--seed"])


@pytest.mark.parametrize("how", ["link", "PYTHONSAFEPATH"])
def test_a_package_imported_from_another_folder_stops_it(tmp_path, how):
    # Through a link the script runs from the link's folder, which holds no
    # package, and with PYTHONSAFEPATH Python skips the current folder: either
    # way the installed copy is imported, and the script refuses to start it
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    env = _env(record, KALSHI_PYTHON=str(stub))
    if how == "link":
        links = tmp_path / "links"
        links.mkdir()
        (links / script.name).symlink_to(script)
        command, here = [str(links / script.name)], links.resolve()
    else:
        env["PYTHONSAFEPATH"] = "1"
        command, here = [str(script)], script.parent.resolve()
    process, err = _run(command + ["--seed"], tmp_path, env)
    assert process.returncode == 1
    installed = record.parent / "installed checkout"
    assert (f"start_dashboard.sh: {stub} imports kalshi_betting from {installed}, not from "
            f"{here} — run this script by its own path inside its checkout (not through a "
            "link), with PYTHONSAFEPATH unset") in " ".join(err.split())
    # Neither server was started
    assert _seen(record) == _expected(here, None, None)


def test_the_import_check_names_this_checkout_under_the_real_python():
    # The script's own check code, run as the script runs it from the repo
    # root: it imports the live bot and the live dashboard (starting nothing)
    # and names this checkout
    assert "kalshi_betting.live_dashboard" in _CHECK and "kalshi_betting.main" in _CHECK
    env = {name: value for name, value in os.environ.items() if name != "PYTHONSAFEPATH"}
    result = subprocess.run([sys.executable, "-c", _CHECK], cwd=_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == f"kalshi_betting imported from {_ROOT.resolve()}"


def test_it_names_no_user_s_path():
    text = _SCRIPT.read_text(encoding="utf-8")
    for fragment in ("/Users/", "/home/", "Coding Projects", "miniconda", "/opt/"):
        assert fragment not in text, fragment
    # It finds its folder from its own path, with CDPATH cleared first, and its
    # Python from the environment
    assert 'unset CDPATH\ncd -- "$(dirname -- "$0")"' in text
    assert 'python="${KALSHI_PYTHON:-python3}"' in text
    # It never replaces itself (it stays to stop both servers), and it starts
    # both in the background with that Python, its traps set before the first
    assert not re.search(r"^\s*exec\b", text, re.M)
    live = '\n"$python" -m kalshi_betting.live_dashboard ${live_args[@]+"${live_args[@]}"} &\n'
    server = ('\n"$python" -m kalshi_betting.defaults_server "$@" '
              '${server_args[@]+"${server_args[@]}"} &\n')
    assert live in text and server in text
    traps = ["trap stop_servers EXIT", "trap 'signalled=1; exit 130' INT",
             "trap 'signalled=1; exit 143' TERM", "trap 'signalled=1; exit 129' HUP"]
    for trap in traps:
        assert f"\n{trap}\n" in text and text.index(trap) < text.index(live), trap
    # A server is signalled only while bash still lists it as running
    assert "kill -INT $pids" in text and "kill -TERM $pids" in text
    assert len(re.findall(r"\bkill\b", text)) == 2
