# Kalshi Arbitrage Bot — Claude Context

This is a **production prediction-market arbitrage bot** that trades real money via RSA-signed Kalshi REST API calls. Read this file in full before making any changes to trading, fee, or order-submission logic.

---

## What This Is

The bot finds two types of mispriced binary contract pairs on Kalshi, sizes positions using the Kelly criterion, and submits fill-or-kill batch orders to lock in a risk-free profit. A separate backtesting pipeline replays the same strategy on the full history of settled markets.

**Time-series pairs:** Two contracts asking the same question at different deadlines (e.g. "Will BTC exceed $80k by March?" vs. "by June?"). The later deadline must have a higher or equal probability. When the earlier is priced higher, the bot buys NO on the expensive one and YES on the cheap one — all three resolution scenarios are profitable.

**Same-title pairs:** Two contracts with identical title + subtitle on different event tickers. Any divergence >5% is anomalous; the bot exploits it assuming 95% co-resolution probability.

**Multivariate (MVE) markets:** Both pair types include MVE markets (option markets within a multi-choice event, e.g. "Trump" inside "2024 Election Winner") when `INCLUDE_MVE_MARKETS=True` in `config.py`. Pairs are grouped by `event_title + market_title` (see `scanner.pair_key`) so option labels like "Trump" or "Above $80k" can't false-positive across unrelated events. Set `INCLUDE_MVE_MARKETS=False` to fall back to binary-only behaviour.

---

## Module Map

| Module | Role | Key exports |
|--------|------|-------------|
| `config.py` | All constants and fee helpers — imported by everyone | `PROJECT_ROOT`, `BUDGET_FRACTION`, `fee_per_pair_approx()`, `fee_leg_exact()` |
| `auth.py` | Builds authenticated `KalshiClient` | `build_client(mode)`, `verify_auth(client)` |
| `_http.py` | Shared HTTP retry helper for market-data API calls | `api_call_with_retry()` |
| `scanner.py` | Fetches markets, detects pairs, validates orderbook depth | `CandidatePair`, `normalize_title()`, `pair_key()`, `display_title()`, `fetch_open_events_with_markets()`, `find_time_series_pairs()`, `find_same_title_pairs()`, `enrich_with_orderbook_prices()` |
| `strategy.py` | Kelly sizing, portfolio selection | `TradeSpec`, `compute_trade()`, `select_portfolio()` |
| `trader.py` | Order submission with atomic rollback | `execute_trades()`, `pre_execution_check()` |
| `reporter.py` | Excel logging and dev simulation output | `TradeResult`, `append_to_prod_log()`, `write_dev_simulation()` |
| `main.py` | Orchestrator for live trading pipeline — no business logic | `_run_dev()`, `_run_prod()` |
| `scheduler.py` | Weekly daemon — runs main.py as a subprocess every Monday 09:00, killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` | `main()` |
| `historical.py` | Fetches and disk-caches historical market data | `fetch_all_settled_markets()`, `fetch_daily_candlesticks()` |
| `backtester.py` | Replays strategy on settled market history | `run_backtest()` |
| `dashboard.py` | Generates self-contained HTML performance report | `generate_dashboard()` |
| `backtest.py` | CLI entry point for the backtest pipeline | — |

**Dependency order (no circular imports):**
```
config.py, _http.py
  └─ auth.py, scanner.py
       └─ strategy.py
            └─ trader.py, reporter.py
                  └─ main.py, scheduler.py

historical.py → backtester.py → dashboard.py → backtest.py
```

---

## Run Commands

All commands must be run from the **repo root**:

```bash
# One-time setup (Python >= 3.11)
pip install -e ".[dev]"

# Sandbox simulation with virtual $1k balance
python3 -m kalshi_betting.main --mode dev --sandbox-balance 1000

# Sandbox simulation, dry-run (no orders submitted)
python3 -m kalshi_betting.main --mode dev --dry-run

# Production live trading
python3 -m kalshi_betting.main --mode prod

# Production discovery only, no orders
python3 -m kalshi_betting.main --mode prod --dry-run

# Backtest from 2024-01-01 with $10k starting balance
python3 -m kalshi_betting.backtest --start-date 2024-01-01 --balance 10000

# Backtest forcing fresh API fetch (ignores disk cache)
python3 -m kalshi_betting.backtest --no-cache

# Weekly scheduler daemon (runs prod every Monday 09:00)
python3 -m kalshi_betting.scheduler

# Run tests
python3 -m pytest tests/ -v

# Lint check
python3 -m ruff check kalshi_betting/
```

---

## CI and Packaging

- `.github/workflows/ci.yml` runs two jobs on every push/PR to `main`: **Lint (ruff)** and **Test (pytest)**. Both must pass before merging — run them locally first.
- `pyproject.toml` declares `requires-python = ">=3.11"` and pins `kalshi-python-sync==3.2.0`. Do NOT bump the SDK pin to 3.13.0 — that version's own metadata requires Python >= 3.13 and breaks `pip install -e ".[dev]"` on 3.11 (i.e. breaks CI).
- Tests live in `tests/` (`test_backtester.py`, `test_config.py`, `test_historical.py`, `test_scanner.py`, `test_strategy.py`, `test_trader.py`) and run fully offline against `MagicMock` clients.

---

## Credential Setup

Two files must exist in the **repo root** (not committed — in `.gitignore`):

**`secrets.json`** — API key IDs:
```json
{
  "Kalshi-api-key": "your-prod-key-id-here",
  "dev_api_key": "your-sandbox-key-id-here"
}
```
The `dev_api_key` field is optional; if absent, `auth.py` falls back to `Kalshi-api-key` for sandbox mode.

**`kalshi_private_key.pem`** — RSA private key for production signing.
**`kalshi_demo_private_key.pem`** (optional) — RSA private key for sandbox. Falls back to the prod key if absent.

`PROJECT_ROOT` in `config.py` is derived from `__file__` and always resolves to the repo root, so credentials must live there.

---

## Key Patterns — Always Follow These

**Constants live in `config.py` exclusively.** Never hardcode `0.07` (fee rate), `0.20` (Kelly cap), `0.15` or `0.05` (price thresholds), `5000` (min balance cents), `1` (buy_max_cost slippage cents), or `3600` (scheduler job timeout) inline. Always import from `config.py`.

**Use `@dataclass` for all data transfer objects.** Existing: `CandidatePair`, `TradeSpec`, `TradeResult`, `BacktestTrade`. Never use a plain dict when a dataclass fits.

**Return `None` for validation failures; don't raise.** `compute_trade()` returns `None` when there's no edge. `find_time_series_pairs()` skips bad markets silently. Only raise for truly unexpected errors.

**Two-stage fee calculation:**
1. `fee_per_pair_approx(nA, pB)` — continuous approximation; use during pair *filtering* before the integer contract count `n` is known.
2. `fee_leg_exact(n, p)` — ceiling-rounded exact fee; use for *final validation* once `n` is determined.
Never swap these — the approximation underestimates and will let bad trades through if used for final validation.

**Retry pattern:** Use `api_call_with_retry()` from `_http.py` for all new market-data API calls (this includes read-only portfolio calls like `get_balance` and `get_held_tickers`). It handles HTTP 429 and 5xx with exponential backoff doubling 2s → 32s across 6 attempts (`_MAX_DELAY = 60s` is only a defensive ceiling). Do NOT add retry logic to order submission in `trader.py` — a failed leg means price moved, not a transient error.

**Pair dedup prefers same-title.** When both scanners detect the same ticker pair, `main._dedup_pairs(same_title_pairs, time_series_pairs)` keeps the same-title entry — its co-resolution model is simpler and doesn't depend on the time-series independence assumption. This matches `strategy.select_portfolio()`'s same_title > time_series tie-breaker. Don't flip the argument order.

**Directional price-gap filter.** `find_time_series_pairs()` requires `pA - pB >= MIN_PRICE_DIFF` (earlier-deadline leg priced HIGHER) — not `abs(pA - pB)`. A pricier later-closing contract is normal term structure, not an arbitrage; reintroducing `abs()` would create losing candidates.

**Budget fitting in `compute_trade()`.** After Kelly sizing, `n` is shrunk until the fee-inclusive total cost fits within the Kelly-capped budget. Don't remove the shrink loop — sizing on price alone systematically overshoots `BUDGET_FRACTION`.

**Use `pathlib.Path` for all file paths.** Never `os.path` or string concatenation.

**Monetary values:** The Kalshi API returns balances in **cents** (int). Contract prices are in **dollars** (float, range 0–1). The conversion is always `/ 100`. Don't mix units.

---

## Known Gotchas

**`KalshiClient` monkey-patch pattern:** The SDK does NOT accept `api_key_id` or `private_key_pem` as constructor parameters. They must be set as attributes on the `Configuration` object:
```python
cfg = Configuration(host=url)
cfg.api_key_id = key_id        # detected by hasattr() inside SDK
cfg.private_key_pem = pem_text
client = KalshiClient(configuration=cfg)
```
Do not "fix" this by moving them to the constructor — it will break silently.

**Fee ceiling rounding at small `n`:** `fee_leg_exact(1, 0.5)` = `ceil(0.07 * 1 * 0.5 * 0.5 * 100) / 100` = `ceil(1.75) / 100` = `$0.02`. At `n=1`, fees round up aggressively. The exact payoff check `min_payoff > 0` in `compute_trade()` catches this and returns `None` — this is intentional, not a bug. Also: `fee_leg_exact()` rounds to 6 decimals *before* the ceiling so binary float noise (e.g. `175.00000000000003`) can't bump an exact-cent fee up an extra cent — don't remove the `round()`.

**`TradeResult.status` vocabulary:** `"executed"` (both legs filled), `"simulated"` (dry run), `"failed"` (leg A never filled — nothing to unwind), `"rolled_back"` (leg B failed, leg A unwind FoK filled), `"rollback_failed"` (unwind did not fill — orphaned position), `"manual_review"` (leg B errored AND the position lookup also failed, so leg B's fill state is unknown — the bot deliberately does NOT auto-rollback, since unwinding could reverse a real fill; a human must check the account). Don't "simplify" `manual_review` into an automatic rollback.

**FoK price protection:** Both buy legs carry `buy_max_cost` = `count × (scanned price + BUY_MAX_COST_SLIPPAGE_CENTS)`, so an order fills at or below scanned-price-plus-slippage or not at all. Any new order-building code must set this cap.

**Backtest cache semantics:** `fetch_all_settled_markets()` and `fetch_daily_candlesticks()` always persist a fresh fetch to disk even when `use_cache=False`, so `--no-cache` refreshes what the next default run loads. Candlestick cache files are tagged with the `[open_ts, close_ts]` window they cover — a cache built for one `--start-date` is not reused for an earlier one.

**`secrets.json` may be missing outer braces.** `auth.py` wraps it if needed. New code must call `auth.build_client()` rather than reading `secrets.json` directly.

**`CandidatePair.max_contracts = 0` means two different things** depending on where you are in the pipeline:
- Before `enrich_with_orderbook_prices()`: not yet computed (skip depth cap).
- After enrichment with `tradeable=False`: no qualifying depth found.
Check `pair.tradeable` to distinguish.

**Backtest uses prod credentials**, not sandbox. The Kalshi historical API endpoint doesn't exist on the sandbox. `historical.py` always calls `build_prod_live_client()`.

**Dev mode skips `get_held_tickers()`** because the sandbox account is separate and has no positions from production. Don't add held-ticker filtering to `_run_dev()`.

**`normalize_title()` pattern order matters.** `_DATE_PATTERNS` in `scanner.py` applies patterns sequentially. Longer/more specific patterns must come before shorter ones to avoid partial matches leaving date fragments. If adding new patterns, prepend them before similar shorter ones.

**MVE event-title plumbing.** The live scanner attaches an `_event_title` attribute to each `Market` object inside `fetch_open_events_with_markets()`. The backtest stores it as the `event_title` key in cached market dicts. Both pair-finder paths (`find_time_series_pairs`, `find_same_title_pairs`) and their backtester counterparts (`_group_by_normalized_title`, `_group_by_exact_title`) require this to be set for MVE markets — without it they fall back to grouping by market title alone, which would let cross-event option-label collisions through. If you add a new entry point that produces markets, attach `_event_title` (or `event_title` in dict form) before passing into the grouping functions. Old backtest caches written before this change have no `event_title` field; re-run with `--no-cache` to refresh.

---

## Documentation Standards

**Every file must have a header comment block** at the top (after the shebang/encoding if any) following the pattern already established in every module:
```python
"""
File: filename.py
Author: <name>
Last edited by: <name>

Purpose:
    One paragraph describing what this module does and why it exists.

Dependencies:
    What it imports from the project and what imports from it.

Notes:
    Any non-obvious constraints, SDK quirks, or gotchas specific to this file.
"""
```

**Every function must have a Google-style docstring** matching the existing style:
```python
def my_function(arg: type) -> ReturnType:
    """
    One-sentence summary of what the function does.

    Longer explanation if the logic is non-trivial. Include the invariants
    this function assumes and what it guarantees.

    Args:
        arg (type): Description. Include valid range if numeric.

    Returns:
        type: Description of what's returned and when.

    Raises:
        ExceptionType: When and why this is raised.
    """
```
Functions that return `None` only on failure should document that: `Returns None if <condition>.`

**Every cross-module function call must have an inline comment** explaining the *why*, not just the *what*. The existing codebase already follows this for critical calls (see `main.py` and `strategy.py`):
```python
# Validate depth & replace best-ask with depth-weighted fill price
pairs = scanner.enrich_with_orderbook_prices(client, pairs)
```
Single-line calls to stdlib or well-known helpers (e.g. `logging.info(...)`, `sorted(...)`) don't need comments unless the arguments are surprising.

**Update the README and all relevant comments whenever a change affects behavior.** Specifically:
- If you add, remove, or rename a module: update the Module Dependency Graph in `README.md` and the Module Map in `CLAUDE.md`.
- If you change a constant in `config.py`: update the comment explaining the constant's purpose and rationale.
- If you add a new CLI argument to `main.py` or `backtest.py`: update the Run Commands section in `README.md` and in `CLAUDE.md`.
- If you change the data flow of the live or backtest pipeline: update the corresponding ASCII diagram in `README.md`.
- If you change function signatures or behavior: update the docstring in the same commit, not as a follow-up.

---

## What NOT To Do

- **Do not bypass the rollback logic in `trader._execute_one()`**. If leg B is confirmed unfilled after leg A fills, an unwind must be attempted immediately — an orphaned leg-A NO position is an unhedged directional bet. But if leg B's fill state is *unknown* (order errored and the position lookup also failed), the correct behavior is `status="manual_review"`, NOT an automatic rollback that could reverse a real fill.
- **Do not add retry logic to order submission**. One retry after a fill-or-kill rejection could submit leg A twice at different prices, creating an unhedged position.
- **Do not import `trader.py` from `scanner.py` or `strategy.py`** — circular dependency.
- **Do not write output files outside `PROJECT_ROOT`**. All logs, Excel files, and HTML dashboards write relative to `PROJECT_ROOT`.
- **Do not modify `_DATE_PATTERNS` without running `test_normalize_title`** — a regression silently causes missed pairs or false positives with no error.
- **Do not add tests that call the real Kalshi API**. Use `unittest.mock.MagicMock` for `KalshiClient`. Tests should be runnable offline.
- **Do not change the fee formula without updating both** `fee_per_pair_approx()` and `fee_leg_exact()` in `config.py`. The backtester also uses both.
