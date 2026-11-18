"""
File: dashboard_chunks_golden.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The cases, the page builder and the digests behind
    tests/test_dashboard.py::TestParkedChunks — the gate proving that parking
    the dashboard's scenario chunks on disk while the page's grid is walked
    (dashboard._ChunkParking) changed no byte of what a build writes. A
    "chunk" is one packed block of figures for one scenario trade list: the
    page holds the run's own scenario's, and every other is a file in the
    build's folder beside the page, which the page loads when a reader
    chooses that scenario. Each case builds one whole page with
    dashboard.generate_dashboard, offline and with every name and date the
    page prints pinned, and records the SHA-256 of the page, of every file
    in the folder of chunk files beside it (config.DASHBOARD_FILES_DIRNAME)
    and the list of that folder's entries. The cases cover a plain size-cap
    sweep, the tier-floors-off view, the Rebalance, minimum price movement,
    "Rebalance until" and Sell views, and every path that drops chunks the
    walk already packed or rewrites them: a Rebalance, price-movement or
    "Rebalance until" cell that raises, the tier-floors-off size-cap
    fallback, the size-cap fallback walk, a tier-floors-off cell that fails
    with no fallback after the off cells were packed (the whole grid walked
    again, handing out ids the first walk parked), a chunk that cannot be
    built (a page without the filter bar), a primary scenario that is not
    the first chunk, a build folder that cannot be made (every chunk in the
    page), a Sell build that fails, a page whose one scenario chunk is its
    own (the build folder holds the Sell select's files alone), and a page
    of one chunk.

    It lives in its own module so that the SAME code both CAPTURES the
    digests (run as a script against a copy of the commit before parking)
    and REPLAYS them (imported by the test against the tree under test). A
    digest of the tree under test against itself would prove nothing.

Dependencies:
    kalshi_betting.config and kalshi_betting.dashboard, and the page
    fixtures of tests/test_dashboard.py (handed in as `fx`) — from whichever
    tree is first on sys.path. Imported by tests/test_dashboard.py; run
    directly to capture.

Notes:
    A digest is a property of the bytes, so it moves with anything that
    changes how the page is serialised: plotly (its figure JSON and default
    template), pandas and numpy (number and date spellings), the zlib
    library (the packed blocks are gzip), the Python version and whether
    orjson is importable (plotly serialises through it when it is). The
    golden records that rendering environment, and the test skips — naming
    both environments — rather than fails when a digest diverges in a
    DIFFERENT one. Within one environment two builds of a case write the
    same bytes (no random id is left in the page once the build folder's
    name is pinned), so no normaliser is applied.

    Re-capture (only when the oracle commit or the rendering environment
    changes):
        git worktree add --detach <dir> <commit whose output is the oracle>
        cd <dir>
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<dir> \\
            python <this tree>/tests/dashboard_chunks_golden.py [--source-sha <sha>] \\
            > <this tree>/tests/fixtures/dashboard_chunks_golden.json
    Run it from inside <dir>, with PYTHONPATH naming <dir>: the package is an
    editable install that otherwise imports the primary checkout, and the
    page fixtures must come from <dir>'s tests too. The script records the
    tree it imported and that tree's commit (git's, or --source-sha for a
    copy without git metadata) in the JSON.
"""
import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import tempfile
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import pytest

from kalshi_betting import config, dashboard

# The window's last day every case's page names (its Period line and the
# source note its bar proposes), so a digest does not move with the clock
PERIOD_END = date(2026, 1, 31)
# The name every case's build folder is given, in place of the time, process
# and random suffix a build names it by
BUILD_NAME = "build"


def _offline(*_args, **_kwargs):
    """A yfinance download that fails, as it does with no network."""
    raise RuntimeError("offline")


def _sweep_case(make):
    """A case whose page is a sweep's: `make(fx, mp)` returns the sweep."""
    def case(fx, mp):
        sweep = make(fx, mp)
        return sweep, sweep.primary.trades, sweep.primary.equity_df
    return case


def _nth_call_raises(mp, name: str, n: int, error: BaseException) -> None:
    """Make dashboard.<name> raise `error` on its n-th call (1 for the first)."""
    real = getattr(dashboard, name)
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(None)
        if len(calls) == n:
            raise error
        return real(*args, **kwargs)

    mp.setattr(dashboard, name, wrapped)


def _chunk_fails(fx, mp):
    """The fourth chunk packed cannot be built: the page loses its filter bar."""
    _nth_call_raises(mp, "_list_payload", 4, RuntimeError("a chunk could not be built"))
    return fx._kc_sweep_tiers()


def _primary_not_first(fx, mp):
    """The walk files the primary scenario under chunk 1: the page loses its bar."""
    real = dashboard._build_filter_grid

    def walked_wrongly(*args, **kwargs):
        source, chunker, kd = real(*args, **kwargs)
        pb, pk, pc = source.primary
        chunker.grid[pb][pk][pc] = 1
        return source, chunker, kd

    mp.setattr(dashboard, "_build_filter_grid", walked_wrongly)
    return fx._kc_sweep()


def _no_folder(fx, mp):
    """The build folder cannot be made: every chunk stays in the page."""
    def refuse(_page):
        raise OSError(30, "Read-only file system")

    mp.setattr(dashboard, "_new_build_folder", refuse)
    return fx._kc_sweep_until(fx._kc_sweep_tiers())


def _off_fails_late(fx, mp):
    """The tier-floors-off cells are packed, then one reports a failure no fallback recovers.

    The whole grid is then walked again from the eager points (_walk_grid),
    so ids the first walk packed (and parked) are handed out again.
    """
    real = dashboard._walk_off_cells
    calls = []

    def walked(*args, **kwargs):
        calls.append(None)
        error = real(*args, **kwargs)
        if error is None and len(calls) == 1:
            return RuntimeError("a tier-floors-off cell could not be simulated")
        return error

    mp.setattr(dashboard, "_walk_off_cells", walked)
    return fx._kc_sweep_until(fx._kc_sweep_tiers())


def _sell_only(fx, mp):
    """Every scenario packs to the page's own chunk: the build folder holds only Sell files."""
    def one_key(self, bi, ki, k, listed, *, phase="on"):
        return "every scenario"

    mp.setattr(dashboard._ChunkVisitor, "_key", one_key)
    return fx._kc_sweep_sell()


def _sell_fails(fx, mp):
    """The Sell build fails after one of its tasks wrote files: the select stays shut."""
    def fail(tasks, workers):
        (tasks[0].out_dir / "chunk-9.js").write_text("x", encoding="utf-8")
        (tasks[0].out_dir / "left-over.js").write_text("x", encoding="utf-8")
        raise RuntimeError("the pool could not start")

    mp.setattr(dashboard, "_run_sell_tasks", fail)
    return fx._kc_sweep_sell()


def _one_chunk(fx, _mp):
    """No sweep: the page's own run is its one chunk, and no folder is made."""
    trades = fx._flt_trades()
    return None, trades, fx.backtester._build_equity_curve(trades, fx._FLT_START, 1000.0)


# Every case: name -> (fx, mp) -> (sweep or None, the page's trades, their curve)
CASES = {
    "size_cap_sweep": _sweep_case(lambda fx, mp: fx._kc_sweep()),
    "tier_floors_off": _sweep_case(lambda fx, mp: fx._kc_sweep_tiers()),
    "tier_floors_off_every_cap": _sweep_case(lambda fx, mp: fx._kc_sweep_tiers_capped()),
    "rebalance": _sweep_case(lambda fx, mp: fx._kc_sweep_rebalance(fx._kc_sweep_tiers())),
    "price_movement": _sweep_case(lambda fx, mp: fx._kc_sweep_move(fx._kc_sweep_tiers())),
    "rebalance_until": _sweep_case(lambda fx, mp: fx._kc_sweep_until(fx._kc_sweep_tiers())),
    "sell": _sweep_case(lambda fx, mp: fx._kc_sweep_sell()),
    "sell_and_every_view": _sweep_case(
        lambda fx, mp: fx._kc_sweep_sell(fx._kc_sweep_until(fx._kc_sweep_tiers()))),
    # A Rebalance cell raises after earlier Rebalance cells were packed
    "rebalance_cell_raises": _sweep_case(lambda fx, mp: fx._kc_sweep_rebalance(
        fx._kc_sweep_tiers(), raise_on=(fx._KC_B0, 0.75))),
    # A minimum price movement raises mid-cell, after one of its runs was packed
    "price_movement_cell_raises": _sweep_case(lambda fx, mp: fx._kc_sweep_move(
        fx._kc_sweep_tiers(), move_raise=((False, fx._KC_B0, 0.75), 2))),
    # A "Rebalance until" choice raises mid-cell, after one of its runs was packed
    "rebalance_until_cell_raises": _sweep_case(lambda fx, mp: fx._kc_sweep_until(
        fx._kc_sweep_tiers(), maturity_raise=((False, fx._KC_B0, 0.75), 6))),
    # The tier-floors-off size-cap sweep's second cell raises: the off cells are
    # walked again from the family's own points at the run's own cap
    "tier_floors_off_fallback": _sweep_case(
        lambda fx, mp: fx._kc_sweep_tiers_capped(raise_on=(fx._KC_B0, 0.6))),
    # The size-cap sweep's last cell raises: the whole grid is walked again
    # from the eager points (the Rebalance family is lost with the cap axis)
    "size_cap_fallback": _sweep_case(lambda fx, mp: fx._kc_sweep_until(
        fx._kc_sweep(raise_on=(fx._KC_B1, 0.75)))),
    # A tier-floors-off cell fails with no fallback after the off cells were
    # packed: the whole grid is walked again from the eager points, handing
    # out ids the first walk parked
    "off_cell_fails_without_fallback": _sweep_case(_off_fails_late),
    "chunk_cannot_be_built": _sweep_case(_chunk_fails),
    "primary_not_the_first_chunk": _sweep_case(_primary_not_first),
    "folder_cannot_be_made": _sweep_case(_no_folder),
    "sell_build_fails": _sweep_case(_sell_fails),
    # The page's own chunk is its only scenario chunk, so its build folder
    # holds the Sell select's files alone
    "sell_view_only": _sweep_case(_sell_only),
    "one_chunk": _one_chunk,
}


def _sha(data: bytes) -> str:
    """SHA-256 of some bytes, in hex."""
    return hashlib.sha256(data).hexdigest()


def digest(out: Path) -> dict:
    """
    What one build wrote: the page's digest, and every entry and file beside it.

    Args:
        out (Path): The page generate_dashboard returned.

    Returns:
        dict: {"page": SHA-256 of the page's bytes, "entries": every path
            under the folder of chunk files (relative to it, sorted; None
            when the folder does not exist), "files": each file's path ->
            SHA-256 of its bytes}.
    """
    root = out.parent / config.DASHBOARD_FILES_DIRNAME
    if not root.exists():
        return {"page": _sha(out.read_bytes()), "entries": None, "files": {}}
    paths = sorted(root.rglob("*"))
    return {"page": _sha(out.read_bytes()),
            "entries": [path.relative_to(root).as_posix() for path in paths],
            "files": {path.relative_to(root).as_posix(): _sha(path.read_bytes())
                      for path in paths if path.is_file()}}


def build(fx, name: str, folder: Path, *, in_memory: bool = False) -> Path:
    """
    Build one case's whole page into `folder`, offline, its names and dates pinned.

    Args:
        fx: The module of page fixtures (tests/test_dashboard.py).
        name (str): The case (a key of CASES).
        folder (Path): An empty folder, used as PROJECT_ROOT.
        in_memory (bool): Keyword-only. Hold every chunk in memory during the
            walk instead of parking it on disk (the walk is handed no
            parking folder). False (default) builds as production does.
            Only for the tree under test: the oracle commit has no parking.

    Returns:
        Path: The page written.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dashboard, "PROJECT_ROOT", folder)
        mp.setattr(dashboard.yf, "download", _offline)
        made = dashboard._new_build_folder

        def fixed_build_folder(page):
            new = made(page)
            return new.rename(new.with_name(BUILD_NAME))

        mp.setattr(dashboard, "_new_build_folder", fixed_build_folder)
        if in_memory:
            walk = dashboard._build_filter_grid

            def walk_in_memory(*args, parking=None, **kwargs):
                return walk(*args, **kwargs)

            mp.setattr(dashboard, "_build_filter_grid", walk_in_memory)
        sweep, trades, curve = CASES[name](fx, mp)
        return dashboard.generate_dashboard(
            trades, curve, fx._FLT_START, 1000.0, sweep=sweep,
            interval_discount=None if sweep is None else 0.75,
            series_categories=fx._FLT_SERIES_TIERS, sell_workers=1, period_end=PERIOD_END)


def rendering_environment() -> dict[str, str | None]:
    """The library versions a digest depends on beyond this repository's code."""
    try:
        import orjson
        orjson_version = orjson.__version__
    except ImportError:
        orjson_version = None
    return {"plotly": plotly.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "orjson": orjson_version}


def _source_commit(tree: Path) -> str:
    """The commit of the tree the package was imported from, or "unknown"."""
    try:
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Capture: print the golden JSON for the tree currently on sys.path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source-sha", help="the tree's commit, when it has no git metadata")
    args = parser.parse_args()
    fx = importlib.import_module("tests.test_dashboard")
    tree = Path(dashboard.__file__).resolve().parent.parent
    print(f"capturing from {tree} (fixtures {fx.__file__})", file=sys.stderr)
    cases = {}
    for name in CASES:
        with tempfile.TemporaryDirectory() as folder:
            cases[name] = digest(build(fx, name, Path(folder)))
    golden = {
        "note": (
            "SHA-256 of every byte a dashboard build writes, per case of "
            "tests/dashboard_chunks_golden.py::CASES: the page, every file in the folder of "
            "chunk files beside it, and that folder's entries — captured by running that "
            "module as a script against source_tree at source_sha, the commit before the "
            "walk parked chunks on disk (the recipe is in its module docstring). "
            "tests/test_dashboard.py::TestParkedChunks replays the same code against the "
            "tree under test; every digest must match, and one that diverges in an "
            "environment other than rendering_env skips rather than fails."
        ),
        "source_sha": args.source_sha or _source_commit(tree),
        "source_tree": tree.name,
        "rendering_env": rendering_environment(),
        "cases": cases,
    }
    json.dump(golden, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
