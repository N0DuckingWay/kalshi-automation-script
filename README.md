# Kalshi Arbitrage Bot

An automated arbitrage trading bot for the [Kalshi](https://kalshi.com) prediction market platform. The bot finds pairs of correlated prediction market contracts where one is mispriced relative to the other, sizes positions using the Kelly criterion, and submits fill-or-kill orders leg-by-leg to lock in a risk-free profit — with automatic rollback if the second leg doesn't fill. A separate backtesting pipeline replays the same strategy on the full history of settled Kalshi markets and generates an interactive HTML performance dashboard.

---

## How It Profits

Kalshi markets are binary contracts that pay $1 if a question resolves YES and $0 if it resolves NO. The bot exploits two specific pricing anomalies:

**Time-series pairs:** Two contracts asking the same question at different deadlines (e.g. "Will BTC exceed $80k by March 2025?" and "Will BTC exceed $80k by June 2025?") should satisfy `P(A) <= P(B)` because the later deadline gives more time for the event to occur. When the earlier contract is priced *higher* than the later one by at least a required margin, the market is mispriced. The bot buys NO on the expensive (earlier) contract and YES on the cheap (later) contract. All three resolution scenarios are profitable:

- A=YES before B resolves: B likely resolves YES too — both pay out
- Both resolve YES: YES on B pays out, covering the NO on A cost
- Both resolve NO: NO on A pays out, covering the YES on B cost

**Same-title pairs:** Two contracts on different event tickers but with the *identical* title and subtitle (i.e. asking exactly the same question). If their prices diverge by more than 5%, the bot buys NO on the expensive one and YES on the cheap one. Since both contracts should co-resolve, the trade is essentially risk-free.

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
```

### Live Trading Data Flow

```
main.py
  ├─ auth.build_client()           — authenticate with Kalshi API
  ├─ scanner.get_held_tickers()    — fetch currently-held positions (prod only) so we skip re-entering them
  ├─ scanner.fetch_open_events_with_markets() — fetch open events + their markets (attaches event titles for MVE grouping)
  ├─ scanner.filter_markets_within_horizon() — optional --max-horizon-days cap (no-op if unset)
  ├─ scanner.find_time_series_pairs()   — time-series pair detection
  ├─ scanner.find_same_title_pairs()    — same-title pair detection
  ├─ main._dedup_pairs()            — merge both lists, preferring same-title on overlap
  ├─ scanner.enrich_with_orderbook_prices() — validate depth & real fills
  ├─ strategy.compute_trade()      — Kelly sizing per pair
  ├─ strategy.select_portfolio()   — greedy portfolio selection
  ├─ trader.pre_execution_check()  — re-fetch order books, drop pairs whose prices moved
  ├─ trader.execute_trades()       — submit fill-or-kill orders leg-by-leg (parallel across pairs, rollback on partial fill)
  └─ reporter.append_to_prod_log() — write results to trade_log.xlsx
```

### Backtest Data Flow

```
backtest.py (CLI)
  ├─ historical.build_historical_client()    — prod API client for archives
  ├─ historical.build_prod_live_client()     — prod API client for recent data
  ├─ backtester.run_backtest()
  │    ├─ historical.fetch_all_settled_markets() — market metadata
  │    ├─ historical.fetch_candlesticks()        — hourly price series per ticker
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
| `auth.py` | Reads RSA credentials from `secrets.json` and the PEM key file, constructs an authenticated `KalshiClient`, and provides `verify_auth()` to confirm credentials and read the live account balance. |
| `_http.py` | Shared HTTP helpers used across the package: `api_call_with_retry()` (exponential backoff on 429/5xx for market-data calls) and `fetch_json_page()` (parses the SDK's raw `*_without_preload_content` responses, re-raising non-2xx as `ApiException`). |
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles to group time-series pairs, detects same-title pairs via exact match, and enriches tradeable pairs with live order book depth to compute real fill prices. |
| `strategy.py` | Applies the Kelly criterion to size each trade, computes minimum guaranteed profit and monthly-normalized return, and greedily selects a portfolio that fits within the available balance. |
| `trader.py` | Converts `TradeSpec` objects into orders and submits each pair's two legs sequentially (fill-or-kill, NO leg then YES leg) via the Kalshi API, with automatic rollback of a filled leg A if leg B doesn't fill. Multiple pairs execute concurrently. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`. |
| `scheduler.py` | Long-running daemon that fires the production bot every Monday at 09:00 using the `schedule` library. Also prints the equivalent cron job command. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints, sharded into parallel per-day slices that are cached individually so interrupted or repeated fetches resume instead of re-walking months of history) and hourly candlestick price series needed by the backtester. |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs, scans weekly Monday snapshots for the first tradeable entry, applies Kelly sizing, records actual P&L from settlement outcomes, and builds a daily equity curve. |
| `dashboard.py` | Generates a self-contained HTML performance report from backtest results, including equity curve, Sharpe/Sortino/drawdown KPIs, calibration analysis, trade diagnostics, and an S&P 500 benchmark comparison. |
| `backtest.py` | CLI entry point for the backtest pipeline. Parses arguments, builds the historical API clients, calls `backtester.run_backtest()` then `dashboard.generate_dashboard()`, and logs a summary. |

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
    settled_markets_*.json  ← Assembled per-start-date market list
    archive_days/           ← Per-created-day archive slices (incremental/resumable)
    live_days/              ← Per-settled-day recent-market slices (incremental/resumable)
    candlesticks/           ← Per-ticker hourly price series
  kalshi_betting/           ← Python package
```

---

## Run Commands

### Dev dry-run (sandbox simulation, no real orders)

```bash
python3 -m kalshi_betting.main --mode dev --dry-run
```

Scans the Kalshi sandbox markets with a virtual $1,000 balance and writes a simulation Excel file.

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

`--max-horizon-days` only enters trades where the later-closing leg is within the
given number of days of the *simulated* entry checkpoint (each Monday evaluated
during the replay), not today's real date. Optional; omit for no limit.

The settled-market fetch is sharded into one slice per UTC day and fetched with
`SETTLED_FETCH_MAX_WORKERS` (default 8) parallel workers; each completed slice
is cached in `backtest_cache/archive_days/` / `backtest_cache/live_days/`. An
interrupted fetch resumes at day granularity, and `--no-cache` reuses the day
slices (they cannot go stale — see CLAUDE.md), so a refresh only fetches the
current day plus any days not yet on disk.

### Weekly scheduler daemon

```bash
python3 -m kalshi_betting.scheduler
```

Runs the production bot every Monday at 09:00 in a blocking loop. The log also prints the equivalent `crontab` entry if you prefer cron.

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
