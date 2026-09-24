"""
File: dashboard_golden.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The fixture, the renderer and the normaliser behind
    tests/test_dashboard.py::TestGoldenSections — the gate proving that the
    scenario-explorer commit (PB5) left the seven pre-existing dashboard
    sections rendering exactly as they did on main. It lives in its own
    module, importing only names main @ fe0a758 already had, so that the SAME
    code both CAPTURES the golden (run as a script against a checkout of
    main) and REPLAYS it (imported by the test against the tree under test).
    A golden rendered from the tree under test would be tautological, and a
    fixture copied between a capture script and a test could drift.

    Two sections are rendered twice, to reach branches a single fixture
    cannot: _section_interval_discount with an outcome-label census below
    the floor (the red banner and the recoloured k-hat card) and above it
    (the healthy coverage line); and _section_benchmark with a deterministic
    ^GSPC frame (the S&P row and its trading-day Sharpe) and with a failing
    download (the strategy-only degradation).

Dependencies:
    kalshi_betting.backtester, kalshi_betting.config and
    kalshi_betting.dashboard — from whichever tree is first on sys.path.
    Imported by tests/test_dashboard.py; run directly to capture.

Notes:
    The normaliser makes the digest a property of the RENDERED CONTENT, not
    of the serialiser that happened to produce it. Plotly serialises through
    orjson when it is importable (it is in the optional `perf` extra, which
    CI does not install) and through the stdlib json module otherwise, and
    the two spell non-ASCII text differently ("20–40¢" vs "20\\u201340\\u00a2");
    plotly 6 encodes numpy arrays as base64 typed arrays where plotly 5 wrote
    plain lists; and every figure embeds plotly's version-specific default
    template. So every Plotly.newPlot(...) argument list is parsed, typed
    arrays are decoded to plain lists, layout.template is dropped, and the
    result is re-serialised canonically (sorted keys, ASCII escapes); Plotly's
    per-render UUID div ids become "UUID"; and runs of whitespace collapse to
    one space. What is left can still move with a plotly or pandas release
    (a changed default attribute, a different date spelling), which is why
    the golden records the rendering environment it was captured in and the
    test skips — naming both environments — rather than fails when a digest
    diverges in a DIFFERENT environment.

    Re-capture (only when a pre-existing section is meant to change, or the
    rendering environment moved):
        git worktree add --detach <dir> <commit whose render is the oracle>
        cd <dir>
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<dir> \\
            python <this tree>/tests/dashboard_golden.py \\
            > <this tree>/tests/fixtures/dashboard_golden_main.json
    Run it from inside <dir>, with PYTHONPATH naming <dir>: the package is an
    editable install that otherwise imports the primary checkout. The script
    prints the tree it actually imported and that tree's commit into the
    JSON, so a capture from the wrong tree is visible in the fixture itself.
"""
import base64
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly

from kalshi_betting import dashboard
from kalshi_betting.backtester import (
    BacktestSweep,
    BacktestTrade,
    IntervalCalibration,
    IntervalCalibrationBucket,
    OutcomeLabelCoverage,
    SweepPoint,
)
from kalshi_betting.config import BACKTEST_OUTCOME_LABEL_WARN_FRACTION, fee_leg_exact

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_NEWPLOT = "Plotly.newPlot("
_WHITESPACE = re.compile(r"\s+")


# ─── Fixture ──────────────────────────────────────────────────────────────────

def _trade(title_a: str = "Will BTC exceed $80k?", profit: float | None = None,
           deadline_gap_days: int | None = None) -> BacktestTrade:
    """A coherent time-series trade (YES 0.30 on A, NO 0.40 on B, n=5), in
    tests/test_dashboard.py::make_trade's shape, built only from the fields
    main's BacktestTrade already had."""
    n = 5
    entry_pA, entry_pB, entry_nA, entry_nB = 0.30, 0.60, 0.70, 0.40
    total_cost = n * (entry_pA + entry_nB)
    fees = fee_leg_exact(n, entry_pA) + fee_leg_exact(n, entry_nB)
    expected_payoff = n * (1.0 - entry_pA - entry_nB) - fees
    if profit is None:
        profit = expected_payoff
    return BacktestTrade(
        pair_type="time_series", ticker_a="TICK-A", ticker_b="TICK-B",
        title_a=title_a, title_b="Will BTC exceed $90k?", category="Crypto",
        entry_date=date(2026, 1, 5), exit_date=date(2026, 1, 12),
        entry_pA=entry_pA, entry_pB=entry_pB, entry_nA=entry_nA, entry_nB=entry_nB,
        n=n, total_cost=total_cost, fees=fees, outcome_a="yes", outcome_b="yes",
        actual_payoff=float(n), profit=profit,
        profit_ratio=profit / (total_cost + fees), monthly_profit_ratio=0.1,
        kelly_fraction=0.1, expected_payoff=expected_payoff,
        slippage=profit - expected_payoff, holding_days=7, balance_at_entry=1000.0,
        deadline_gap_days=deadline_gap_days,
    )


def _equity(values: list[float], start: date = date(2026, 1, 5)) -> pd.DataFrame:
    """An equity curve in _build_equity_curve's column shape."""
    df = pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(len(values))],
        "portfolio_value": [float(v) for v in values],
    })
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df


def _calibration() -> IntervalCalibration:
    """One gap band plus the POOLED row."""
    return IntervalCalibration(
        pooled=IntervalCalibrationBucket(label="POOLED", tier=0.0, n=10,
                                         realised_rate=0.12, mean_implied=0.20,
                                         empirical_k=0.60),
        buckets=[IntervalCalibrationBucket(label="0-7d", tier=0.15, n=6,
                                           realised_rate=0.10, mean_implied=0.18,
                                           empirical_k=None)],
        excluded_premise_violations=3,
    )


def _coverage(with_subtitle: int, total: int = 100) -> OutcomeLabelCoverage:
    """A census shaped as the backtester produces one, below_floor computed
    the way the census computes it."""
    fraction = with_subtitle / total
    return OutcomeLabelCoverage(
        total=total, with_subtitle=with_subtitle, with_event_title=total,
        subtitle_fraction=fraction, event_title_fraction=1.0,
        below_floor=fraction < BACKTEST_OUTCOME_LABEL_WARN_FRACTION,
        cumulative_markets=7, snapshot_markets=11, unknown_deadline_markets=82,
    )


def _sp500_frame(*_args, **_kwargs) -> pd.DataFrame:
    """A deterministic stand-in for yfinance's ^GSPC download."""
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07",
                          "2026-01-08", "2026-01-09"])
    return pd.DataFrame({"Close": [100.0, 101.0, 99.5, 102.0, 101.5]}, index=idx)


def _offline(*_args, **_kwargs):
    """A yfinance download that fails, as it does with no network."""
    raise RuntimeError("offline")


def render_sections() -> dict[str, str]:
    """
    Render the seven pre-existing sections on the fixed fixture.

    yfinance's download is stubbed for the duration and restored afterwards,
    so this never touches the network.

    Returns:
        dict[str, str]: Section name -> raw rendered HTML (not normalised).
    """
    trades = [
        _trade(),
        _trade(title_a="Will ETH exceed $5k?", profit=50.0, deadline_gap_days=10),
        _trade(title_a="Will SOL exceed $500?", profit=-20.0, deadline_gap_days=25),
    ]
    equity_df = _equity([1000.0, 1010.0, 1005.0, 1020.0, 995.0, 1030.0])
    start_date = date(2026, 1, 5)
    initial_balance = 1000.0
    points = [
        SweepPoint(k=k, trades=[_trade()] * i,
                   equity_df=_equity([1000.0, 1000.0 + 10 * i, 1000.0 + 5 * i]))
        for i, k in enumerate([0.60, 0.75, 0.90], start=1)
    ]

    def sweep(coverage: OutcomeLabelCoverage) -> BacktestSweep:
        return BacktestSweep(primary=points[1], points=points,
                             calibration=_calibration(), label_coverage=coverage)

    real_download = dashboard.yf.download
    try:
        dashboard.yf.download = _sp500_frame
        benchmark = dashboard._section_benchmark(equity_df, start_date, initial_balance)
        dashboard.yf.download = _offline
        benchmark_offline = dashboard._section_benchmark(
            equity_df, start_date, initial_balance)
    finally:
        dashboard.yf.download = real_download

    return {
        "performance": dashboard._section_performance(
            equity_df, trades, start_date, initial_balance),
        "decomposition": dashboard._section_decomposition(trades),
        "calibration": dashboard._section_calibration(trades),
        "interval_discount_below_floor": dashboard._section_interval_discount(
            sweep(_coverage(2))),
        "interval_discount_healthy": dashboard._section_interval_discount(
            sweep(_coverage(97))),
        "diagnostics": dashboard._section_diagnostics(trades),
        "risk": dashboard._section_risk(trades, equity_df, initial_balance, k=0.75),
        "benchmark": benchmark,
        "benchmark_offline": benchmark_offline,
    }


# ─── Normalisation ────────────────────────────────────────────────────────────

def _decode_typed_arrays(obj):
    """Replace every plotly base64 typed array ({dtype, bdata[, shape]}) with
    the plain nested list plotly 5 would have written in its place."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj and set(obj) <= {"bdata", "dtype", "shape"}:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]),
                                dtype=np.dtype("<" + obj["dtype"]))
            shape = [int(s) for s in str(obj.get("shape") or "").split(",") if s.strip()]
            if len(shape) > 1:
                arr = arr.reshape(shape)
            return arr.tolist()
        return {k: _decode_typed_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_typed_arrays(v) for v in obj]
    return obj


def _skip_whitespace(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def normalize(fragment: str) -> str:
    """
    Reduce a rendered section to what it SHOWS, independent of how plotly
    serialised it.

    Every Plotly.newPlot(...) argument list is parsed as JSON, its typed
    arrays decoded, the layout's default template dropped, and the whole
    list re-serialised with sorted keys and ASCII escapes; Plotly's random
    UUID div ids become "UUID"; runs of whitespace become one space.

    Args:
        fragment (str): One section's rendered HTML.

    Returns:
        str: The canonical form the golden digests are taken over.

    Raises:
        json.JSONDecodeError: If a Plotly.newPlot call's arguments are not a
            comma-separated run of JSON values — a changed embedding the
            golden must not silently skip over.
    """
    text = _UUID_RE.sub("UUID", fragment)
    decoder = json.JSONDecoder()
    out: list[str] = []
    pos = 0
    while (start := text.find(_NEWPLOT, pos)) >= 0:
        i = start + len(_NEWPLOT)
        out.append(text[pos:i])
        args = []
        while True:
            i = _skip_whitespace(text, i)
            if text[i] == ")":
                break
            value, i = decoder.raw_decode(text, i)
            args.append(value)
            i = _skip_whitespace(text, i)
            if text[i] == ",":
                i += 1
        if len(args) >= 3 and isinstance(args[2], dict):
            args[2].pop("template", None)
        out.append(json.dumps(_decode_typed_arrays(args), sort_keys=True,
                              ensure_ascii=True, separators=(",", ":")))
        pos = i
    out.append(text[pos:])
    return _WHITESPACE.sub(" ", "".join(out))


def section_digests(sections: dict[str, str]) -> dict[str, str]:
    """SHA-256 of each section's normalised HTML."""
    return {name: hashlib.sha256(normalize(html).encode("utf-8")).hexdigest()
            for name, html in sections.items()}


def rendering_environment() -> dict[str, str]:
    """The library versions a digest depends on beyond this repository's code."""
    return {"plotly": plotly.__version__, "pandas": pd.__version__,
            "numpy": np.__version__}


def _source_commit(tree: Path) -> str:
    """The commit of the tree the package was imported from, or "unknown"."""
    try:
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Capture: print the golden JSON for the tree currently on sys.path."""
    tree = Path(dashboard.__file__).resolve().parent.parent
    golden = {
        "note": (
            "SHA-256 of each pre-existing dashboard section, rendered by "
            "tests/dashboard_golden.py::render_sections() and normalised by its "
            "normalize(), captured by running that module as a script against "
            "source_tree at source_sha (the recipe is in its module docstring). "
            "tests/test_dashboard.py::TestGoldenSections replays the same code "
            "against the tree under test; every digest must match, and a digest "
            "that diverges in an environment other than rendering_env skips "
            "rather than fails."
        ),
        "source_sha": _source_commit(tree),
        "source_tree": tree.name,
        "rendering_env": rendering_environment(),
        "hashes": section_digests(render_sections()),
    }
    json.dump(golden, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
