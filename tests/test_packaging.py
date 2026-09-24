"""
File: test_packaging.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline tests that pin pyproject.toml's explicit setuptools package
    discovery. Without it, setuptools' flat-layout auto-discovery treats every
    top-level directory whose name is a Python identifier as a candidate
    package, and a checkout that has run a backtest also holds backtest_cache/
    at the root — so the documented `pip install -e ".[dev]"` refused to build
    ("Multiple top-level packages discovered in a flat-layout") on exactly the
    machines that run backtests, while CI, whose fresh checkout has no cache,
    never saw it.

Dependencies:
    Reads pyproject.toml with the stdlib tomllib and exercises
    setuptools.find_packages when setuptools is importable. Imports nothing
    from kalshi_betting.

Notes:
    The discovery test builds a synthetic flat layout under tmp_path rather
    than scanning the real checkout, so it does not depend on whether this
    machine has a backtest_cache/ (CI never does). It gives that fake cache an
    __init__.py on purpose: the include pattern must exclude it even then.
"""
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _find_config() -> dict:
    """
    Return pyproject.toml's [tool.setuptools.packages.find] table.

    Returns:
        dict: The table, or {} when pyproject.toml declares none.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})


class TestExplicitPackageDiscovery:
    """pyproject.toml names the one package to ship instead of auto-discovering."""

    def test_discovery_is_restricted_to_kalshi_betting(self):
        assert _find_config().get("include") == ["kalshi_betting*"]

    def test_a_backtest_cache_at_the_root_is_never_discovered(self, tmp_path):
        setuptools = pytest.importorskip("setuptools")
        (tmp_path / "kalshi_betting").mkdir()
        (tmp_path / "kalshi_betting" / "__init__.py").write_text("")
        (tmp_path / "kalshi_betting" / "sub").mkdir()
        (tmp_path / "kalshi_betting" / "sub" / "__init__.py").write_text("")
        # The directory a backtest creates, made to look as much like a
        # package as possible, plus the tests directory beside it.
        (tmp_path / "backtest_cache" / "live_days").mkdir(parents=True)
        (tmp_path / "backtest_cache" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "__init__.py").write_text("")

        found = setuptools.find_packages(
            where=str(tmp_path), include=_find_config()["include"]
        )

        assert sorted(found) == ["kalshi_betting", "kalshi_betting.sub"]
