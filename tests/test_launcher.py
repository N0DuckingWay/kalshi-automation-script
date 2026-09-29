"""
File: test_launcher.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for start_dashboard.sh, the script at the repo root that starts the
    live-defaults server: it parses as bash, git records it as executable,
    it runs from its own folder (a folder whose path has a space included,
    whatever folder it is started from, and whatever CDPATH says), checks
    that its Python can import the live bot from that folder, and then
    replaces itself with `<python> -m kalshi_betting.defaults_server`,
    passing every argument through. A Python that is missing, that cannot
    import the bot, or that imports it from another folder (a link to the
    script, PYTHONSAFEPATH) stops it with exit 1 and a message saying what to
    do. It names no user's path.

Dependencies:
    Imports nothing from the package: the script is run as a program. Each
    test copies it into its own tmp_path and runs it with KALSHI_PYTHON (or,
    once, the PATH) pointing at a stand-in for Python, a small sh script that
    records how it was called and starts nothing, so no test starts the real
    server. One test runs the script's own import check under the Python
    running the tests, from the repo root, which imports the live bot but
    starts nothing.

Notes:
    Skipped where there is no bash. macOS ships bash 3.2, and these tests
    run under whatever bash is first on the PATH, which is that one on a
    stock Mac.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The repo root, where the script lives
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "start_dashboard.sh"
_BASH = shutil.which("bash")

# The Python code of the script's import check, as the script spells it
_CHECK = re.search(r"^check='(.*)'$", _SCRIPT.read_text(encoding="utf-8"), re.M).group(1)

pytestmark = pytest.mark.skipif(_BASH is None, reason="no bash on this machine")

# The exit code the stand-in gives when it is run as the server, so a test can
# tell its exit code reached the caller (the script replaced itself with it)
_STUB_SERVER_EXIT = 7

# The stand-in for Python. Called with -c (the script's import check) it
# records the code and, like Python, reports where it imported the package
# from: the current folder when that folder holds a kalshi_betting package and
# PYTHONSAFEPATH is unset (the current folder comes first on Python's import
# path), else the installed copy STUB_INSTALLED_ROOT names. When
# STUB_IMPORT_ERROR is set it prints a Python-style error and fails instead.
# Called any other way (as the server) it records its process id, its folder
# and each argument, and exits _STUB_SERVER_EXIT. It writes only to the file
# STUB_RECORD names.
_STUB = f"""#!/bin/sh
if [ "$1" = "-c" ]; then
  printf 'check %s\\n' "$2" >> "$STUB_RECORD"
  if [ -n "${{STUB_IMPORT_ERROR:-}}" ]; then
    echo "Traceback (most recent call last):" >&2
    echo "$STUB_IMPORT_ERROR" >&2
    exit 1
  fi
  echo "a warning on stderr" >&2
  if [ -z "${{PYTHONSAFEPATH:-}}" ] && [ -d kalshi_betting ]; then
    echo "kalshi_betting imported from $(pwd -P)"
  else
    echo "kalshi_betting imported from $STUB_INSTALLED_ROOT"
  fi
  exit 0
fi
printf 'pid %s\\n' "$$" >> "$STUB_RECORD"
printf 'cwd %s\\n' "$(pwd -P)" >> "$STUB_RECORD"
for arg in "$@"; do
  printf 'arg %s\\n' "$arg" >> "$STUB_RECORD"
done
exit {_STUB_SERVER_EXIT}
"""


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


def _run(command: list[str], cwd: Path, env: dict) -> tuple[subprocess.Popen, str]:
    """
    Run the script and wait for it.

    Args:
        command (list[str]): The command line.
        cwd (Path): The folder to start it from.
        env (dict): Its environment.

    Returns:
        tuple[subprocess.Popen, str]: The finished process (its pid and
            returncode) and what it wrote to stderr.
    """
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = process.communicate(timeout=60)
    return process, err


def _env(record: Path, **extra) -> dict:
    """
    The script's environment: this one's, less the variables that steer it, plus the stand-in's.

    KALSHI_PYTHON, CDPATH and PYTHONSAFEPATH are left out, so a test sets
    only the ones it means to.

    Args:
        record (Path): The file the stand-in writes to; the stand-in's
            installed copy of the package is said to live beside it, in
            "installed checkout".
        **extra: More variables to set.

    Returns:
        dict: The environment.
    """
    env = {name: value for name, value in os.environ.items()
           if name not in ("KALSHI_PYTHON", "CDPATH", "PYTHONSAFEPATH")}
    env["STUB_RECORD"] = str(record)
    env["STUB_INSTALLED_ROOT"] = str(record.parent / "installed checkout")
    env.update(extra)
    return env


def _record(record: Path) -> list[tuple[str, str]]:
    """
    Read what the stand-in recorded, as (kind, text) pairs in order.

    Args:
        record (Path): The stand-in's record file.

    Returns:
        list[tuple[str, str]]: One pair per line ("check", "pid", "cwd" or
            "arg", and the rest of the line); [] when nothing was recorded.
    """
    if not record.exists():
        return []
    return [tuple(line.split(" ", 1)) if " " in line else (line, "")
            for line in record.read_text(encoding="utf-8").splitlines()]


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
def test_it_replaces_itself_with_the_server_from_its_own_folder(tmp_path, how):
    script = _checkout(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    start, cwd = _INVOCATIONS[how](script)
    command = start if isinstance(start, list) else [start]
    arguments = ["--seed", "--no-browser", "two words", "--x=$HOME", "*", ""]
    process, err = _run(command + arguments, cwd, _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == _STUB_SERVER_EXIT, err
    lines = _record(record)
    # The import check first, then the server: the same process (it replaced
    # itself), started from the script's own folder, with every argument as given
    assert lines[0] == ("check", _CHECK)
    assert lines[1] == ("pid", str(process.pid))
    assert lines[2] == ("cwd", str(script.parent.resolve()))
    assert [text for kind, text in lines[3:]] == [
        "-m", "kalshi_betting.defaults_server", *arguments]
    assert {kind for kind, _ in lines[3:]} == {"arg"}


def test_with_no_arguments_the_server_gets_none(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    process, err = _run([str(script)], tmp_path, _env(record, KALSHI_PYTHON=str(stub)))
    assert process.returncode == _STUB_SERVER_EXIT, err
    assert [text for kind, text in _record(record) if kind == "arg"] == [
        "-m", "kalshi_betting.defaults_server"]


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
    assert [text for kind, text in _record(record) if kind == "arg"] == [
        "-m", "kalshi_betting.defaults_server", "--seed"]


def test_a_python_that_cannot_import_the_bot_stops_it(tmp_path):
    script = _checkout(tmp_path)
    stub = _write_stub(tmp_path / "bin")
    record = tmp_path / "record.txt"
    error = "ModuleNotFoundError: No module named 'kalshi_python_sync'"
    process, err = _run([str(script), "--seed"], tmp_path,
                        _env(record, KALSHI_PYTHON=str(stub), STUB_IMPORT_ERROR=error))
    assert process.returncode == 1
    assert f"start_dashboard.sh: {stub} cannot import the live bot" in err
    assert 'pip install -e ".[dev]"' in err and "KALSHI_PYTHON" in err
    # The error's last line is shown, so the missing piece is named
    assert err.rstrip().endswith(error)
    # The server was never started
    assert _record(record) == [("check", _CHECK)]


def test_a_missing_python_stops_it(tmp_path):
    script = _checkout(tmp_path)
    record = tmp_path / "record.txt"
    missing = tmp_path / "no such python"
    process, err = _run([str(script)], tmp_path, _env(record, KALSHI_PYTHON=str(missing)))
    assert process.returncode == 1
    assert (f"start_dashboard.sh: {missing} not found — set KALSHI_PYTHON to the Python "
            "this package is installed in") in err
    assert _record(record) == []


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
    assert ("cwd", str(script.parent.resolve())) in _record(record)


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
    # The server was never started
    assert _record(record) == [("check", _CHECK)]


def test_the_import_check_names_this_checkout_under_the_real_python():
    # The script's own check code, run as the script runs it from the repo
    # root: it imports the live bot (starting nothing) and names this checkout
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
    assert text.rstrip().endswith('exec "$python" -m kalshi_betting.defaults_server "$@"')
