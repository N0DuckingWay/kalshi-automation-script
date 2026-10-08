#!/usr/bin/env bash
# Start the dashboard: the live dashboard and the live-defaults server, then
# open the page on its Live trading tab.
#
# Two servers start from this checkout, each in the background. The live
# dashboard (kalshi_betting.live_dashboard, read-only) serves the page's two
# tabs: Live trading, which reads the account from Kalshi on every load, and
# Backtest, backtest_dashboard.html as it is. It opens the page in the
# browser. The live-defaults server (kalshi_betting.defaults_server) serves
# the pages the Backtest tab's "Save as live defaults…" and "Trade using
# defaults…" buttons open; it opens nothing itself, except with --seed. If
# this checkout's servers are already running, the page just opens again.
# Ctrl-C stops both, and so does a SIGTERM or SIGHUP sent to this script
# alone; stop them when you are done. A trading run the defaults server
# started keeps going until it finishes.
#
#   ./start_dashboard.sh                start both and open the Live trading tab
#   ./start_dashboard.sh --seed         start both and open the seed values'
#                                       confirmation page instead
#   ./start_dashboard.sh --no-browser   start both, open nothing, only log the addresses
#   ./start_dashboard.sh --help         say this, and start nothing
#
# It takes only those flags, spelled in full; anything else is refused (exit 2)
# before anything starts. The flags are passed to the defaults server, and the
# live dashboard gets --no-browser when either is given. If the live dashboard
# stops with an error within 2 s of starting (its port held by another
# program, say), the defaults server opens its own start page instead (the
# backtest page; the seed page with --seed; nothing with --no-browser) and a
# warning says so. If it stops later, a warning says so and the defaults
# server keeps running. It can be run from any directory.
#
# KALSHI_PYTHON picks the Python this package is installed in (default: python3).
set -euo pipefail

usage() {
  cat <<'EOF'
usage: start_dashboard.sh [--seed] [--no-browser]

Start the live dashboard (the Live trading and Backtest tabs, read-only) and the
live-defaults server from this checkout, and open the page on its Live trading tab.

  --seed        open the seed values' confirmation page instead
  --no-browser  open nothing; only log the addresses
  -h, --help    show this message and start nothing
EOF
}

# The flags, checked before anything starts. Only these, spelled in full: the
# servers would read an abbreviation such as --se as --seed, and the live
# dashboard, not told, would open the page too.
seed=0
no_browser=0
for arg in "$@"; do
  case "$arg" in
    --seed) seed=1 ;;
    --no-browser) no_browser=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf "start_dashboard.sh: unknown argument '%s' — use --seed, --no-browser or --help\n" \
             "$arg" >&2
      exit 2
      ;;
  esac
done

# Run from this checkout's folder: python -m then imports this checkout's
# package (the current folder comes first on Python's import path), so the
# servers save and trade this checkout's live defaults with this checkout's
# code. CDPATH is cleared so cd cannot land in a same-named folder elsewhere.
unset CDPATH
cd -- "$(dirname -- "$0")"
here=$(pwd -P)
python="${KALSHI_PYTHON:-python3}"

if ! command -v "$python" >/dev/null 2>&1; then
  echo "start_dashboard.sh: $python not found — set KALSHI_PYTHON to the Python this package is installed in" >&2
  exit 1
fi
# The runs the defaults server starts import the whole live bot (the Kalshi
# SDK, tabulate, openpyxl), and the live dashboard needs pandas, numpy, plotly
# and yfinance too. The check also prints the folder the package was imported from.
check='import pathlib, kalshi_betting.live_dashboard, kalshi_betting.main as bot; print("kalshi_betting imported from", pathlib.Path(bot.__file__).resolve().parent.parent)'
if ! output=$("$python" -c "$check" 2>&1); then
  echo "start_dashboard.sh: $python cannot import the live bot and the live dashboard (the" \
       "Kalshi SDK, tabulate, openpyxl, pandas, numpy, plotly, yfinance) — activate the" \
       "environment you ran pip install -e \".[dev]\" in, or set KALSHI_PYTHON" >&2
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

live_pid=""     # the live dashboard while it runs
server_pid=""   # the defaults server while it runs
nap_pid=""      # the sleep of the latest pause
signalled=0     # 1 once a signal is ending this script
hurry=0         # 1 once another signal comes while the servers are being stopped

# Whether a background job of this script is still running. bash's own list
# of its jobs is asked, never the process table, so a process number the
# system has given to another program since the job ended never counts.
running() {
  [ -n "$1" ] || return 1
  case " $(jobs -rp | tr '\n' ' ') " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

# The given jobs that are still running, one per line
still_running() {
  local pid
  for pid in "$@"; do
    if running "$pid"; then
      printf '%s\n' "$pid"
    fi
  done
}

# Wait $1 seconds. A signal ends the wait at once: its trap runs without
# waiting for the sleep to finish.
pause() {
  sleep "$1" &
  nap_pid=$!
  wait "$nap_pid" || true
}

# Wait up to $1 tenths of a second for both servers to end, or for another signal
settle() {
  local tries=$1
  while [ "$tries" -gt 0 ] && [ "$hurry" = 0 ] &&
        [ -n "$(still_running "$live_pid" "$server_pid")" ]; do
    pause 0.1
    tries=$((tries - 1))
  done
}

# On every exit, stop the servers still running. After a signal, give them 2 s
# to stop on their own (Ctrl-C reaches them too). Then send each a Ctrl-C,
# which both turn into a clean stop (the defaults server then names any
# trading run still going), and after 5 s SIGTERM whatever is left. Another
# signal meanwhile goes straight to the SIGTERM.
stop_servers() {
  local pids
  trap 'hurry=1' INT TERM HUP
  if [ "$signalled" = 1 ]; then
    settle 20
  fi
  pids=$(still_running "$live_pid" "$server_pid")
  if [ -n "$pids" ] && [ "$hurry" = 0 ]; then
    kill -INT $pids 2>/dev/null || true
    settle 50
  fi
  pids=$(still_running "$live_pid" "$server_pid" "$nap_pid")
  if [ -n "$pids" ]; then
    kill -TERM $pids 2>/dev/null || true
  fi
}

# Set before anything starts, so no signal can leave a server running
trap stop_servers EXIT
trap 'signalled=1; exit 130' INT
trap 'signalled=1; exit 143' TERM
trap 'signalled=1; exit 129' HUP

# The live dashboard (both tabs, read-only) opens the page itself, so the
# defaults server opens nothing, except with --seed, when the seed
# confirmation page is what opens
live_args=()
if [ "$seed" = 1 ] || [ "$no_browser" = 1 ]; then
  live_args=(--no-browser)
fi
# ${arr[@]+"${arr[@]}"}: bash 3.2 (macOS) with set -u refuses an empty "${arr[@]}"
"$python" -m kalshi_betting.live_dashboard ${live_args[@]+"${live_args[@]}"} &
live_pid=$!

# Up to 2 s for it to stop at once. With an error (its port held by another
# program, say) the defaults server opens its own start page instead; with 0,
# this checkout's live dashboard was already running and has opened its page.
tries=20
while [ "$tries" -gt 0 ] && running "$live_pid"; do
  pause 0.1
  tries=$((tries - 1))
done
opened=1
if ! running "$live_pid"; then
  status=0
  wait "$live_pid" 2>/dev/null || status=$?
  live_pid=""
  if [ "$status" != 0 ]; then
    opened=0
    if [ "$no_browser" = 1 ]; then
      instead="only the defaults server runs"
    elif [ "$seed" = 1 ]; then
      instead="only the defaults server runs (it opens the seed values' confirmation page)"
    else
      instead="only the defaults server runs, and it opens the backtest page itself"
    fi
    echo "start_dashboard.sh: the live dashboard did not start (its message is above); $instead" >&2
  fi
fi
server_args=()
if [ "$opened" = 1 ] && [ "$seed" = 0 ] && [ "$no_browser" = 0 ]; then
  server_args=(--no-browser)
fi
"$python" -m kalshi_betting.defaults_server "$@" ${server_args[@]+"${server_args[@]}"} &
server_pid=$!

# Until both have ended. The defaults server's exit code, when not 0, is this
# script's own; a 0 (this checkout's was already running) leaves the script
# running until the live dashboard ends. A live dashboard that stops with an
# error is named, and the defaults server keeps running.
while [ -n "$live_pid" ] || [ -n "$server_pid" ]; do
  if [ -n "$server_pid" ] && ! running "$server_pid"; then
    status=0
    wait "$server_pid" 2>/dev/null || status=$?
    server_pid=""
    if [ "$status" != 0 ]; then
      exit "$status"
    fi
  fi
  if [ -n "$live_pid" ] && ! running "$live_pid"; then
    status=0
    wait "$live_pid" 2>/dev/null || status=$?
    live_pid=""
    if [ "$status" != 0 ]; then
      keeps=""
      if running "$server_pid"; then
        keeps="; the defaults server keeps running (its address is in its log above)"
      fi
      echo "start_dashboard.sh: the live dashboard stopped with exit status $status" \
           "(its message is above)$keeps" >&2
    fi
  fi
  if [ -n "$live_pid" ] || [ -n "$server_pid" ]; then
    pause 1
  fi
done
