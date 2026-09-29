#!/usr/bin/env bash
# Start the live-defaults server and open the backtest dashboard.
#
# The dashboard's "Save as live defaults…" and "Trade using defaults…" buttons
# open pages served by kalshi_betting.defaults_server on this machine. This
# script starts that server from this checkout; the server then opens
# backtest_dashboard.html (with --seed, the page proposing the seed values).
# If this checkout's server is already running, it just opens the page again.
# Ctrl-C stops the server; stop it when you are done. A trading run it started
# keeps going until it finishes.
#
#   ./start_dashboard.sh            start the server and open the dashboard
#   ./start_dashboard.sh --seed     start it and open the seed values' confirmation page
#
# Every argument is passed to the server (--no-browser opens nothing and only
# logs the address). It can be run from any directory.
#
# KALSHI_PYTHON picks the Python this package is installed in (default: python3).
set -euo pipefail

# Run from this checkout's folder: python -m then imports this checkout's
# package (the current folder comes first on Python's import path), so the
# server saves and trades this checkout's live defaults with this checkout's
# code. CDPATH is cleared so cd cannot land in a same-named folder elsewhere.
unset CDPATH
cd -- "$(dirname -- "$0")"
here=$(pwd -P)
python="${KALSHI_PYTHON:-python3}"

if ! command -v "$python" >/dev/null 2>&1; then
  echo "start_dashboard.sh: $python not found — set KALSHI_PYTHON to the Python this package is installed in" >&2
  exit 1
fi
# The runs the server starts import the whole live bot (the Kalshi SDK, tabulate,
# openpyxl). The check also prints the folder the package was imported from.
check='import pathlib, kalshi_betting.main as bot; print("kalshi_betting imported from", pathlib.Path(bot.__file__).resolve().parent.parent)'
if ! output=$("$python" -c "$check" 2>&1); then
  echo "start_dashboard.sh: $python cannot import the live bot — activate the environment" \
       "you ran pip install -e \".[dev]\" in, or set KALSHI_PYTHON" >&2
  if [ -n "$output" ]; then
    printf '%s\n' "$output" | tail -n 1 >&2
  fi
  exit 1
fi
# A link to this script, or PYTHONSAFEPATH (which drops the current folder from
# the import path), would have the server run another checkout's code and
# trade that checkout's live defaults: refuse rather than do that
found=$(printf '%s\n' "$output" | sed -n 's/^kalshi_betting imported from //p' | tail -n 1)
if [ "$found" != "$here" ]; then
  echo "start_dashboard.sh: $python imports kalshi_betting from ${found:-an unknown folder}," \
       "not from $here — run this script by its own path inside its checkout (not" \
       "through a link), with PYTHONSAFEPATH unset" >&2
  exit 1
fi
exec "$python" -m kalshi_betting.defaults_server "$@"
