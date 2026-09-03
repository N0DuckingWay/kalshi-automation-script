# Kalshi Arbitrage Bot

An automated arbitrage trading bot for the [Kalshi](https://kalshi.com) prediction market platform. The bot finds pairs of correlated prediction market contracts where one is mispriced relative to the other, sizes positions using the Kelly criterion, and submits fill-or-kill orders leg-by-leg to lock in a risk-free profit — with automatic rollback if the second leg doesn't fill. A separate backtesting pipeline replays the same strategy on the full history of settled Kalshi markets and generates an interactive HTML performance dashboard.

---

## How It Profits

Kalshi markets are binary contracts that pay $1 if a question resolves YES and $0 if it resolves NO. The bot exploits two specific pricing anomalies:

**Time-series pairs:** Two contracts asking the same question at different deadlines (e.g. "Will BTC exceed $80k by March 2025?" and "Will BTC exceed $80k by June 2025?") should satisfy `P(A) <= P(B)` because the later deadline gives more time for the event to occur. When the earlier contract is priced *higher* than the later one by at least a required margin, the market is mispriced. The bot buys NO on the expensive (earlier) contract and YES on the cheap (later) contract. All three resolution scenarios are profitable:

- A=YES before B resolves: B likely resolves YES too — both pay out
- Both resolve YES: YES on B pays out, covering the NO on A cost
- Both resolve NO: NO on A pays out, covering the YES on B cost

**Same-title pairs:** Two contracts on different event tickers but with the *identical* title and subtitle (i.e. asking exactly the same question). If their prices diverge by more than 5%, the bot buys NO on the expensive one and YES on the cheap one. Since both contracts should co-resolve, the trade is essentially risk-free. The subtitle here is the outcome label that distinguishes markets sharing one question title (e.g. two candidate names under "Who will the next Pope be?"); the API stopped sending a `subtitle` field in 2026-08, so ingest now sources it from `yes_sub_title` — without that discriminator, two *different* outcomes would be paired as if they were the same contract.

The required price gap for time-series pairs is tiered by how far apart the two deadlines are: 15% for deadlines ≤ 15 days apart, 30% for 16–30 days — wider gaps need a bigger edge because the correlation between the two dates is weaker. Deadlines more than 30 days apart are never considered. See `min_price_diff_for_gap()` in `config.py` for the exact thresholds.

In both cases, Kalshi charges a taker fee per contract leg. The bot only executes trades where the profit margin exceeds all fees after applying order book depth to confirm the gap exists in real liquidity.

---

## Architecture

### Module Dependency Graph

```
secrets.json + PEM key
        |
    auth.py  ←───────────────────────┐
        |                            |
    config.py (constants)            |
        |                            |
    _http.py (retry + raw-response fetch, used by auth/scanner/historical/trader)
        |                            |
        ↓                            |
    scanner.py ──→ strategy.py ──→ trader.py
        |                |              |
        |            reporter.py ←──────┘
        |
    main.py  (orchestrates live trading pipeline)
    scheduler.py (weekly daemon → calls main.py)


    historical.py ──→ backtester.py ──→ dashboard.py
                            |
                    backtest.py (CLI entry)

    (historical.py also imports auth.py's build_client for its own client
     builders; backtester.py also imports scanner.py's normalize_title for
     title-based pair grouping)
```

### Live Trading Data Flow

```
main.py
  ├─ auth.build_client()           — authenticate with Kalshi API
  ├─ auth.verify_auth()            — read the per-shard balance breakdown (prod only; gate and size on the sum)
  ├─ scanner.fetch_shard_statuses() — read GET /exchange/status per-shard trading/transfer flags (fail-soft)
  ├─ scanner.get_held_tickers()    — fetch currently-held positions (prod only) so we skip re-entering them
  ├─ scanner.fetch_open_events_with_markets() — fetch open events + their markets from EVERY exchange shard, tagged (attaches event titles for MVE grouping; drops only markets on trading-inactive shards)
  ├─ main._log_shard_coverage()    — audit advertised shards vs ingested markets/funds (reports, never aborts)
  ├─ scanner.filter_markets_within_horizon() — optional --max-horizon-days cap (no-op if unset)
  ├─ scanner.find_time_series_pairs()   — time-series pair detection
  ├─ scanner.find_same_title_pairs()    — same-title pair detection
  ├─ main._dedup_pairs()            — merge both lists, preferring same-title on overlap
  ├─ scanner.enrich_with_orderbook_prices() — validate depth & real fills
  ├─ strategy.compute_trade()      — Kelly sizing per pair
  ├─ strategy.select_portfolio()   — greedy portfolio selection
  ├─ trader.pre_execution_check()  — re-fetch order books, drop pairs whose prices moved
  ├─ trader.ensure_shard_collateral() — move funds onto the shards the selected legs settle against (prod; dry-run only plans)
  ├─ trader.execute_trades()       — submit fill-or-kill orders leg-by-leg to the V2 order endpoint, each leg routed to its own market's shard (parallel across pairs, rollback on partial fill)
  └─ reporter.append_to_prod_log() — write results to trade_log.xlsx
```

### Backtest Data Flow

```
backtest.py (CLI)
  ├─ historical.build_historical_client()    — prod API client for archives
  ├─ historical.build_prod_live_client()     — prod API client for recent data
  ├─ backtester.run_backtest()
  │    ├─ historical.fetch_all_settled_markets() — market metadata
  │    │     └─ prefilter=_can_ever_enter        — drop never-tradeable markets during assembly
  │    ├─ historical.fetch_candlesticks()        — hourly price series per ticker (parallel across tickers)
  │    ├─ _find_entry()                          — first tradeable Monday per pair
  │    ├─ Kelly sizing + P&L from outcomes
  │    └─ _build_equity_curve()                 — daily portfolio value
  └─ dashboard.generate_dashboard()         — write HTML report
```

---

## Module Descriptions

| Module | Description |
|--------|-------------|
| `__init__.py` | Package initializer. No exports; marks the directory as the `kalshi_betting` package. |
| `config.py` | All tunable constants (price thresholds, Kelly cap, fee rates, API URLs, file paths) and the two fee helper functions used throughout the codebase. |
| `auth.py` | Reads RSA credentials from `secrets.json` and the PEM key file, constructs an authenticated `KalshiClient`, and provides `verify_auth()` to confirm credentials and read the live account balance per exchange shard (`{exchange_index: cents}`; callers sum for sizing). |
| `_http.py` | Shared HTTP helpers used across the package: `api_call_with_retry()` (exponential backoff on 429/5xx for market-data calls) and `fetch_json_page()` (parses the SDK's raw `*_without_preload_content` responses, re-raising non-2xx as `ApiException`), and `signed_request_json()` (signed GET/POST against an arbitrary API path for routes the pinned SDK has no method for — retry-free, since order submission calls it directly). |
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles to group time-series pairs, detects same-title pairs via exact match, and enriches tradeable pairs with live order book depth to compute real fill prices. |
| `strategy.py` | Applies the Kelly criterion to size each trade, computes minimum guaranteed profit and monthly-normalized return, and greedily selects a portfolio that fits within the available balance. |
| `trader.py` | Converts `TradeSpec` objects into orders and submits each pair's two legs sequentially (fill-or-kill, NO leg then YES leg) via the Kalshi API, with automatic rollback of a filled leg A if leg B doesn't fill. Multiple pairs execute concurrently. Submission goes to the V2 order endpoint by default and to the retained legacy endpoint when `config.ORDER_API_VERSION` is flipped — see "Order API version" below. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`. |
| `scheduler.py` | Long-running daemon that fires the production bot every Monday at 09:00 using the `schedule` library. Also prints the equivalent cron job command. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints, sharded into parallel per-day slices that are cached individually so interrupted or repeated fetches resume instead of re-walking months of history) and hourly candlestick price series needed by the backtester (candlesticks are fetched in parallel across tickers and cached per ticker, so workers never share a cache file and a repeat run re-reads them from disk). |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs, scans weekly Monday snapshots for the first tradeable entry, applies Kelly sizing, records actual P&L from settlement outcomes, and builds a daily equity curve. |
| `dashboard.py` | Generates a self-contained HTML performance report from backtest results, including equity curve, Sharpe/Sortino/drawdown KPIs, calibration analysis, trade diagnostics, and an S&P 500 benchmark comparison. |
| `backtest.py` | CLI entry point for the backtest pipeline. Parses arguments, builds the historical API clients, calls `backtester.run_backtest()` then `dashboard.generate_dashboard()`, and logs a summary. |

### Order API version

`config.ORDER_API_VERSION` selects which Kalshi create-order endpoint `trader.py` submits through. The default `"v2"` posts to `/portfolio/events/orders`: a fill-or-kill **limit** order with a dollar-string price, a fixed-point contract count, a `bid`/`ask` side on the market's single YES book, and an explicit `exchange_index`. V2 has no "market" order type, so the limit price is itself the price protection — the scanned price rounded up onto the market's own tick grid plus `BUY_SLIPPAGE_TICKS` ticks, which is a cap the older integer-cent `buy_max_cost` field could not express once MVE/combo markets moved to sub-cent ticks.

Setting it to `"legacy"` restores the original `/portfolio/orders` path (`CreateOrderRequest`, `type="market"`, integer-cent `buy_max_cost` via `BUY_MAX_COST_SLIPPAGE_CENTS`), which is retained unmodified purely as an instant rollback if the V2 request/response mapping misbehaves. Both paths share the same leg ordering, rollback logic, and result-status vocabulary, and neither ever retries a submission.

---

## Setup

### Dependencies

Requires Python >= 3.11.

```bash
pip install -e ".[dev]"
```

If you plan to run backtests, install the optional `perf` extra as well:

```bash
pip install -e ".[dev,perf]"
```

Dependencies are declared in `pyproject.toml`: `kalshi-python-sync` (pinned to `3.2.0` — do not bump, see `CLAUDE.md`), `schedule`, `tabulate`, `cryptography`, `python-dateutil`, `openpyxl`, `plotly`, `pandas`, `numpy`, `scipy`, `yfinance`. The `[dev]` extra adds `pytest` and `ruff`. The `[perf]` extra adds `orjson`, which speeds up the backtest's settled-market fetch — that fetch parses tens of millions of JSON records and is CPU-bound on JSON decoding. It is entirely optional: without it the code falls back to the stdlib `json` module, and because `orjson` emits plain JSON the on-disk cache format is identical either way, so installing or removing it never invalidates a cache.

### Credentials

Create `secrets.json` in the project root with the following structure:

```json
{
  "Kalshi-api-key": "your-production-api-key-id",
  "dev_api_key": "your-sandbox-api-key-id"
}
```

- `Kalshi-api-key` is used for all production API calls.
- `dev_api_key` is optional. If omitted, the sandbox falls back to `Kalshi-api-key` (which will return 401 — see sandbox note below).

Place your RSA private key at:

```
kalshi_private_key.pem
```

(Same directory as `secrets.json`, i.e. the project root defined by `PROJECT_ROOT` in `config.py`.)

### Key files at a glance

```
<project root>/
  secrets.json              ← API key IDs
  kalshi_private_key.pem    ← RSA private key for request signing
  trade_log.xlsx            ← Persistent production trade log (auto-created)
  kalshi_arb.log            ← Live bot log file (auto-created)
  kalshi_backtest.log       ← Backtest log file (auto-created)
  backtest_cache/           ← Disk cache for historical data
    settled_markets_*.json  ← Assembled market list, keyed by start date (and by
                              eligibility-filter tag when the backtester filters
                              during assembly, so subsets never mix with full lists)
    archive_days/           ← Per-created-day archive slices (incremental/resumable)
    live_days/              ← Per-settled-day recent-market slices (incremental/resumable)
    candlesticks/           ← Per-ticker hourly price series
  kalshi_betting/           ← Python package
```

---

## Run Commands

CLI runs now echo log output to the terminal as well as `kalshi_arb.log`.

### Live V2 order-mapping probe (~1 cent of real money)

```bash
python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step no-mapping|unfillable-ask|transfer] [--yes]
```

Human-run verification of the V2 order path's NO-leg mapping (an `ask` must open a NO
position and a reduce-only `bid` must close it), fill-or-kill kill semantics, and the
inter-shard transfer's centicent unit — against the production account, for roughly one
cent of worst-case exposure. Never wired into the pipeline; run it before trusting the
V2 path unsupervised.

### Dev dry-run (sandbox simulation, no real orders)

```bash
python3 -m kalshi_betting.main --mode dev --dry-run
```

Scans the Kalshi sandbox markets with a virtual $1,000 balance and writes a simulation Excel file.
Dev mode never submits real orders regardless of `--dry-run` — the flag is a no-op
here (`main.py` logs `"--dry-run is inert in dev mode"` if you pass it) and is
only meaningful in `--mode prod`; the heading above names the plain `--mode dev`
behavior, not something `--dry-run` changes.

Specify a different virtual balance:

```bash
python3 -m kalshi_betting.main --mode dev --sandbox-balance 5000
```

### Production (real money)

```bash
python3 -m kalshi_betting.main --mode prod
```

Fetches the live account balance, scans real markets, submits batch orders, and appends results to `trade_log.xlsx`.

### Production dry-run (discover trades but don't submit)

```bash
python3 -m kalshi_betting.main --mode prod --dry-run
```

Uses the real account balance and real markets (read-only API calls only — no
orders are submitted), but still writes a simulated row per discovered trade
to `trade_log.xlsx` (status `"simulated"`), same as prod's real-order rows
just without a live fill. Confirmed live behavior — don't assume `--dry-run`
leaves `trade_log.xlsx` untouched.

### Limit how far out a contract's deadline can be

```bash
python3 -m kalshi_betting.main --mode dev --max-horizon-days 14
```

Optional in both dev and prod. Only markets closing within the given number of
days from the moment the bot runs are considered — applies to both time-series
and same-title pairs. Omit the flag (the default) to consider all otherwise-eligible
markets regardless of deadline.

### Backtest

```bash
python3 -m kalshi_betting.backtest
```

Runs from 2024-01-01 with a $10,000 simulated balance. Results are cached in `backtest_cache/`.

Options:

```bash
python3 -m kalshi_betting.backtest --start-date 2023-01-01 --balance 50000
python3 -m kalshi_betting.backtest --no-cache   # rebuild the assembled market list
python3 -m kalshi_betting.backtest --max-horizon-days 14
```

`--start-date` should predate the Kalshi archive cutoff. Markets that settled
after the cutoff have no historical candlestick data, so a window starting after
it produces no trades regardless of how many pairs it finds.

**Feasibility pre-check (BS-11).** Before any network call, `run_backtest()`
checks whether the `[--start-date, yesterday]` window contains at least one
Monday-09:00-UTC entry checkpoint (the only time the replay ever enters a
trade). If not, it logs a warning and returns the same empty result the
zero-trade path already produces — instead of spending minutes fetching
millions of settled-market records into a cache that was always going to
produce zero trades. `historical.py` separately warns (without aborting) when
`--start-date` is on or after the archive cutoff, since that also makes the
window structurally 0-trade — see the paragraph above.

`--max-horizon-days` only enters trades where the later-closing leg is within the
given number of days of the *simulated* entry checkpoint (each Monday evaluated
during the replay), not today's real date. Optional; omit for no limit.

The settled-market fetch is sharded into one slice per UTC day and fetched with
`SETTLED_FETCH_MAX_WORKERS` (default 8) parallel workers; each worker writes its
own completed slice to `backtest_cache/archive_days/` / `backtest_cache/live_days/`
and then releases it, so memory use stays flat no matter how many days the run
spans — the final record list is streamed back off disk once every slice is
present. An interrupted fetch resumes at day granularity, and `--no-cache`
reuses the day slices (they cannot go stale — see CLAUDE.md), so a refresh only
fetches the current day plus any days not yet on disk.

**One-time cache rebuild (BS-02).** Assembled `settled_markets_*.json` files
written before the archive stop rule was fixed can be missing *long-lived*
markets — ones created before `--start-date` that settled inside the window.
The archive is ordered by creation time, and the old walk stopped too early to
reach them. Run the backtest once with `--no-cache` to rebuild those assembled
files; the per-day slice files under `archive_days/` and `live_days/` are
unaffected and are reused, so the rebuild re-pays only the tail walk. That tail
is *not* free: it is never slice-cached, so it is a sequential, one-page-at-a-
time walk down created-time history that is re-paid on **every** run, rebuild or
not. It stops after `ARCHIVE_MAX_BARREN_PAGES` (50) consecutive pages with no
in-window settlement, and — because a single long-dated settler resets that
counter — is hard-capped at `ARCHIVE_TAIL_MAX_PAGES` (2000) pages total, which
logs a WARNING when hit (markets created deeper than that may be missed; raise
the constant if a run needs them).

Candlesticks are then fetched with `CANDLESTICK_FETCH_MAX_WORKERS` (default 8)
parallel workers, one independent request per ticker. Cache files are keyed per
ticker, so workers never contend for a path and any ticker already on disk is
skipped. Fetched sequentially this step ran at roughly 4 tickers/sec, which made
it the single largest cost of a backtest.

Note that `--start-date` must fall before the Kalshi archive cutoff: candlestick
history only exists for archived markets, so a window that starts after the
cutoff has no price data for any of its tickers and produces no trades.

### Weekly scheduler daemon

```bash
python3 -m kalshi_betting.scheduler
```

Runs the production bot every Monday at 09:00 (local time) in a blocking loop, each run spawned as a `python3 -m kalshi_betting.main --mode prod` subprocess and killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` (3600s). The log also prints the equivalent `crontab` entry if you prefer cron.

**Slot record and startup catch-up.** Each run claims its Monday-09:00 slot in `scheduler_state.json` (repo root) *before* spawning the subprocess and finalizes the record — `finished_at`, `exit_code` — on every exit path, including timeout and spawn failure. On startup, the daemon compares the most recent Monday-09:00 slot against that record: if the slot has **no** recorded attempt (daemon not running when it came around — never started, crashed, host rebooted, mid-deploy), it runs a catch-up job immediately rather than waiting up to a week for the next Monday. A slot whose recorded attempt merely *failed* is not retried; only a slot with no attempt at all triggers catch-up.

> ⚠️ **The first daemon start after this upgrade immediately runs a live production trade.** `scheduler_state.json` does not exist yet, so the startup check sees no record for the most recent Monday slot and fires a real `--mode prod` run right away — not at the next Monday 09:00. The same applies to **any** restart after a Monday 09:00 slot passed while the daemon was down. Start the daemon only when you are prepared for it to trade immediately; if you are not, run `python3 -m kalshi_betting.main --mode prod --dry-run` first to confirm what it would do.

**Process exit codes.** `main.py`'s exit code is the only signal the scheduler has for what happened inside a run:

| Code | Meaning |
|------|---------|
| `0`  | `EXIT_OK` — run completed (including a clean run that found no qualifying pairs) |
| `10` | `EXIT_SKIPPED_LOW_BALANCE` — run skipped because the account balance was below the minimum |
| `20` | `EXIT_TRADES_NEED_ATTENTION` — at least one trade came back `rollback_failed` or `manual_review`; **a human must check the account and trade log** |
| `1`  | Unhandled exception — the interpreter's default for a crash; not part of the contract above |

The constants live in `config.py` (`EXIT_OK` / `EXIT_SKIPPED_LOW_BALANCE` / `EXIT_TRADES_NEED_ATTENTION`) and the scheduler maps each to a distinct log level and message, so a low-balance skip or a manual-review run is never logged as "completed successfully".

---

## Testing

```bash
python3 -m pytest tests/ -v      # run the test suite
python3 -m ruff check kalshi_betting/   # lint check
```

Tests run fully offline against `unittest.mock.MagicMock` clients — no real Kalshi API calls. `.github/workflows/ci.yml` runs both commands on every push and pull request to `main`; both must pass before merging.

---

## Sandbox Note

The Kalshi sandbox endpoint (`https://demo-api.kalshi.co`) requires a **completely separate account** registered at [demo.kalshi.co](https://demo.kalshi.co). Your production API key will return `401 Unauthorized` on the sandbox endpoint — this is intentional by Kalshi.

To use dev mode with real sandbox authentication, register a sandbox account, generate its API key, and add it as `"dev_api_key"` in `secrets.json`. Without a sandbox key, dev mode still fetches real sandbox market data (for scanning) but skips the held-positions and balance checks that require authentication.
