# Kalshi Arbitrage Bot

An automated arbitrage trading bot for the [Kalshi](https://kalshi.com) prediction market platform. The bot finds pairs of correlated prediction market contracts where one is mispriced relative to the other, sizes positions using the Kelly criterion, and submits atomic batch orders to lock in a risk-free profit. A separate backtesting pipeline replays the same strategy on the full history of settled Kalshi markets and generates an interactive HTML performance dashboard.

---

## How It Profits

Kalshi markets are binary contracts that pay $1 if a question resolves YES and $0 if it resolves NO. The bot exploits two specific pricing anomalies:

**Time-series pairs:** Two contracts asking the same question at different deadlines (e.g. "Will BTC exceed $80k by March 2025?" and "Will BTC exceed $80k by June 2025?") should satisfy `P(A) <= P(B)` because the later deadline gives more time for the event to occur. When the earlier contract is priced *higher* than the later one, the market is mispriced. The bot buys NO on the expensive (earlier) contract and YES on the cheap (later) contract. All three resolution scenarios are profitable:

- A=YES before B resolves: B likely resolves YES too — both pay out
- Both resolve YES: YES on B pays out, covering the NO on A cost
- Both resolve NO: NO on A pays out, covering the YES on B cost

**Same-title pairs:** Two contracts on different event tickers but with the *identical* title and subtitle (i.e. asking exactly the same question). If their prices diverge by more than 5%, the bot buys NO on the expensive one and YES on the cheap one. Since both contracts should co-resolve, the trade is essentially risk-free.

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
  ├─ scanner.fetch_open_markets()  — fetch all open markets
  ├─ scanner.find_candidate_pairs()     — time-series pair detection
  ├─ scanner.find_same_title_pairs()    — same-title pair detection
  ├─ scanner.enrich_with_orderbook_prices() — validate depth & real fills
  ├─ strategy.compute_trade()      — Kelly sizing per pair
  ├─ strategy.select_portfolio()   — greedy portfolio selection
  ├─ trader.execute_trades()       — submit batch orders to Kalshi API
  └─ reporter.append_to_prod_log() — write results to trade_log.xlsx
```

### Backtest Data Flow

```
backtest.py (CLI)
  ├─ historical.build_historical_client()    — prod API client for archives
  ├─ historical.build_prod_live_client()     — prod API client for recent data
  ├─ backtester.run_backtest()
  │    ├─ historical.fetch_all_settled_markets() — market metadata
  │    ├─ historical.fetch_daily_candlesticks()  — price series per ticker
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
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles to group time-series pairs, detects same-title pairs via exact match, and enriches tradeable pairs with live order book depth to compute real fill prices. |
| `strategy.py` | Applies the Kelly criterion to size each trade, computes minimum guaranteed profit and monthly-normalized return, and greedily selects a portfolio that fits within the available balance. |
| `trader.py` | Converts `TradeSpec` objects into `CreateOrderRequest` pairs and submits them to the Kalshi API as atomic `batch_create_orders` calls with fill-or-kill semantics. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`. |
| `scheduler.py` | Long-running daemon that fires the production bot every Monday at 09:00 using the `schedule` library. Also prints the equivalent cron job command. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints) and daily candlestick price series needed by the backtester. |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs, scans weekly Monday snapshots for the first tradeable entry, applies Kelly sizing, records actual P&L from settlement outcomes, and builds a daily equity curve. |
| `dashboard.py` | Generates a self-contained HTML performance report from backtest results, including equity curve, Sharpe/Sortino/drawdown KPIs, calibration analysis, trade diagnostics, and an S&P 500 benchmark comparison. |

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `kalshi-python-sync`, `openpyxl`, `tabulate`, `schedule`, `plotly`, `pandas`, `numpy`, `yfinance`.

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

### Backtest

```bash
python3 -m kalshi_betting.backtest
```

Runs from 2024-01-01 with a $10,000 simulated balance. Results are cached in `backtest_cache/`.

Options:

```bash
python3 -m kalshi_betting.backtest --start-date 2023-01-01 --balance 50000
python3 -m kalshi_betting.backtest --no-cache   # force fresh API fetch
```

### Weekly scheduler daemon

```bash
python3 -m kalshi_betting.scheduler
```

Runs the production bot every Monday at 09:00 in a blocking loop. The log also prints the equivalent `crontab` entry if you prefer cron.

---

## Sandbox Note

The Kalshi sandbox endpoint (`https://demo-api.kalshi.co`) requires a **completely separate account** registered at [demo.kalshi.co](https://demo.kalshi.co). Your production API key will return `401 Unauthorized` on the sandbox endpoint — this is intentional by Kalshi.

To use dev mode with real sandbox authentication, register a sandbox account, generate its API key, and add it as `"dev_api_key"` in `secrets.json`. Without a sandbox key, dev mode still fetches real sandbox market data (for scanning) but skips the held-positions and balance checks that require authentication.
