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
| `config.py` | All constants and fee/threshold helpers — imported by everyone | `PROJECT_ROOT`, `BUDGET_FRACTION`, `fee_per_pair_approx()`, `fee_leg_exact()`, `min_price_diff_for_gap()` |
| `auth.py` | Builds authenticated `KalshiClient` | `build_client(mode)`, `verify_auth(client)` |
| `_http.py` | Shared HTTP retry + raw-response JSON fetch helpers | `api_call_with_retry()`, `fetch_json_page()` |
| `scanner.py` | Fetches markets, detects pairs, validates orderbook depth | `CandidatePair`, `ApiMarket`, `normalize_title()`, `pair_key()`, `display_title()`, `fetch_open_events_with_markets()`, `find_time_series_pairs()`, `find_same_title_pairs()`, `enrich_with_orderbook_prices()`, `filter_markets_within_horizon()`, `get_held_tickers()` |
| `strategy.py` | Kelly sizing, portfolio selection | `TradeSpec`, `compute_trade()`, `select_portfolio()` |
| `trader.py` | Order submission with atomic rollback | `execute_trades()`, `pre_execution_check()` |
| `reporter.py` | Excel logging and dev simulation output | `TradeResult`, `append_to_prod_log()`, `write_dev_simulation()` |
| `main.py` | Orchestrator for live trading pipeline — no business logic | `_run_dev()`, `_run_prod()` |
| `scheduler.py` | Weekly daemon — runs main.py as a subprocess every Monday 09:00, killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` | `main()` |
| `historical.py` | Fetches and disk-caches historical market data | `fetch_all_settled_markets()`, `fetch_candlesticks()` |
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

# Live trading (either mode), only consider markets closing within 14 days from now
python3 -m kalshi_betting.main --mode dev --max-horizon-days 14

# Backtest from 2024-01-01 with $10k starting balance
python3 -m kalshi_betting.backtest --start-date 2024-01-01 --balance 10000

# Backtest forcing fresh API fetch (ignores disk cache)
python3 -m kalshi_betting.backtest --no-cache

# Backtest, only enter trades whose later-closing leg is within 14 days of the simulated entry date
python3 -m kalshi_betting.backtest --max-horizon-days 14

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

**Constants live in `config.py` exclusively.** Never hardcode `0.07` (fee rate), `0.20` (Kelly cap), `0.15`/`0.30` (tiered time-series price thresholds — always go through `min_price_diff_for_gap()`), `0.05` (same-title price threshold), `5000` (min balance cents), `1` (buy_max_cost slippage cents), or `3600` (scheduler job timeout) inline. Always import from `config.py`.

**Use `@dataclass` for all data transfer objects.** Existing: `CandidatePair`, `ApiMarket`, `TradeSpec`, `TradeResult`, `BacktestTrade`. Never use a plain dict when a dataclass fits.

**Return `None` for validation failures; don't raise.** `compute_trade()` returns `None` when there's no edge. `find_time_series_pairs()` skips bad markets silently. Only raise for truly unexpected errors.

**Two-stage fee calculation:**
1. `fee_per_pair_approx(nA, pB)` — continuous approximation; use during pair *filtering* before the integer contract count `n` is known.
2. `fee_leg_exact(n, p)` — ceiling-rounded exact fee; use for *final validation* once `n` is determined.
Never swap these — the approximation underestimates and will let bad trades through if used for final validation.

**Retry pattern:** Use `api_call_with_retry()` from `_http.py` for all new market-data API calls (this includes read-only portfolio calls like `get_balance` and `get_held_tickers`). It handles HTTP 429 and 5xx with exponential backoff doubling 2s → 32s across 6 attempts (`_MAX_DELAY = 60s` is only a defensive ceiling). Do NOT add retry logic to order submission in `trader.py` — a failed leg means price moved, not a transient error.

**Pair dedup prefers same-title.** When both scanners detect the same ticker pair, `main._dedup_pairs(same_title_pairs, time_series_pairs)` keeps the same-title entry — its co-resolution model is simpler and doesn't depend on the time-series independence assumption. This matches `strategy.select_portfolio()`'s same_title > time_series tie-breaker. Don't flip the argument order.

**Directional, deadline-gap-tiered price filter.** `find_time_series_pairs()` requires `pA - pB >= min_price_diff_for_gap(gap_days)` (earlier-deadline leg priced HIGHER) — not `abs(pA - pB)`. A pricier later-closing contract is normal term structure, not an arbitrage; reintroducing `abs()` would create losing candidates. The threshold is tiered by the deadline gap between the two legs: 15% (`MIN_PRICE_DIFF_SHORT_GAP`) when the deadlines are ≤ 15 days apart (`SHORT_DEADLINE_GAP_DAYS`, inclusive), 30% (`MIN_PRICE_DIFF_LONG_GAP`) for 16–30 days; gaps over `MAX_DEADLINE_GAP_DAYS` (30) are never candidates. The same tier drives the orderbook price-sum ceiling (`1 - threshold`, i.e. 0.85 / 0.70) via `scanner._pair_max_sum()` in both `enrich_with_orderbook_prices()` and `validate_pair_price()`, and the backtester's `_find_entry` mirrors all of it. Never bypass `min_price_diff_for_gap()` with a flat threshold — a flat 15% admits weakly-correlated wide-gap pairs the strategy was retuned to exclude.

**Budget fitting in `compute_trade()`.** After Kelly sizing, `n` is shrunk until the fee-inclusive total cost fits within the Kelly-capped budget. Don't remove the shrink loop — sizing on price alone systematically overshoots `BUDGET_FRACTION`.

**Use `pathlib.Path` for all file paths.** Never `os.path` or string concatenation.

**Monetary values:** The Kalshi API returns balances in **cents** (int). Contract prices are in **dollars** (float, range 0–1). The conversion is always `/ 100`. Don't mix units.

---

## Known Gotchas

**Live fetching bypasses the SDK's response models (2026-07 API drift).** The Kalshi API (a) rejects events page sizes above 200 with HTTP 400 (`MARKET_PAGE_SIZE` is now 200 and must stay ≤ 200), and (b) no longer populates the legacy integer fields the pinned SDK's response models type as required: markets lost `yes_ask`/`no_bid`/… (prices now only in `*_dollars` strings) and positions lost `position`/`market_exposure`/… (count now in the `position_fp` string) — so modeled calls that deserialize nested markets, any non-empty positions page, or any order (`Order` lost `yes_price`/`fill_count`/…) raise pydantic `ValidationError`. Therefore `scanner.fetch_open_events_with_markets()`, `scanner.get_held_tickers()`, `scanner._fetch_orderbook()`, `trader._position_count()`, and `trader._submit_order()` call the SDK's `*_without_preload_content` raw-response variants and parse the JSON themselves (markets into `scanner.ApiMarket` dataclasses). Order submission is the highest-stakes case: the modeled `create_order` raises AFTER the order is placed, which would shove every real fill into the ambiguous-exception path and unwind successfully filled legs. Two rules: (1) the raw variants do NOT raise on 4xx/5xx, so every raw call must go through `_http.fetch_json_page()`, which re-raises non-2xx as `ApiException` to keep prior error semantics (`api_call_with_retry`'s 429/5xx backoff for market data; order submission stays retry-free per the rule above); (2) don't "simplify" back to the modeled calls (`client.get_events(...)`, `client.get_positions(...)`, `client.create_order(...)`) — they crash on live responses. Events fetched WITHOUT nested markets (e.g. `historical._load_or_build_event_titles`'s bulk listings) still deserialize fine through the modeled calls, but the modeled `get_event` (singular) does NOT — it embeds nested `Market` models — so the per-ticker title fallback uses a raw GET. Related drift: the MVE events listing is effectively unbounded (hundreds of thousands of auto-generated collection events) and currently returns zero nested markets regardless of `with_nested_markets` — the MVE pull in `fetch_open_events_with_markets()` bails out after `MVE_MAX_EMPTY_PAGES` (25) consecutive marketless pages, and the backtester's title lookup caps its MVE scan at `MVE_TITLE_LOOKUP_MAX_PAGES` (500); without those caps the scan hangs for hours (this is likely why the 2026-07-06 prod run logged nothing after auth). Don't remove the bail-outs.

**Orderbook key rename is a live, recurring hazard — the API has already done it once.** `scanner._fetch_orderbook()` resolves the book via `data.get("orderbook_fp") or data.get("orderbook")` (the field was renamed `orderbook` → `orderbook_fp` in the 2026-07 drift) and each side's levels via `ob.get("yes_dollars") or ob.get("yes")` / `ob.get("no_dollars") or ob.get("no")`. If the API renames either key again, both `or` fallbacks silently resolve to `None`/`[]`, `enrich_with_orderbook_prices()` marks every pair `tradeable=False`, and the bot produces a 0-trade run whose only symptom is a generic "Orderbook unavailable" warning per pair — indistinguishable from markets that genuinely have no depth. PR #30 added a `logging.warning` that fires when neither orderbook key resolves, logging the response's actual top-level keys so this failure mode is diagnosable instead of looking like universally thin liquidity. Don't remove that log line, and if you add a new field-name fallback here, keep the same "log what we actually got" pattern.

**Backtest fetch uses direct signed GETs — the pinned SDK has NO historical_api module.** `historical.build_historical_client()` returns a plain prod `KalshiClient`; `/historical/cutoff`, `/historical/markets`, `/historical/markets/{ticker}/candlesticks`, and `/events/{ticker}` are reached via `historical._signed_raw_get()` (KalshiAuth signs timestamp+method+path, the SDK's rest client executes) composed with `_http.fetch_json_page()` for status checks and `api_call_with_retry` for backoff. Hard-won facts about the archive endpoint (all live-verified 2026-07-13): (1) it IGNORES every server-side time filter (`min_settled_ts`, `max_settled_ts`, etc. all return the identical page); (2) its page size is hard-capped at 1000 (larger → HTTP 400); (3) it is ordered by **`created_time` DESC, ticker DESC — NOT by settlement time** (settlements interleave hours out of order, and long-lived markets created before a date can settle long after it); (4) its pagination cursor is a urlsafe-base64 **protobuf of the keyset position** (field 1 = Timestamp of `created_time`, field 2 = ticker), which `historical._encode_archive_cursor()` reproduces byte-for-byte — this is what makes sharded parallel fetching possible. The live `/markets` endpoint honors `min_settled_ts` AND `max_settled_ts` server-side (page size also capped at 1000), but does NOT serve pre-cutoff settlements — once markets migrate into the archive they disappear from `/markets` entirely. Candlestick candles now arrive with the dollar string in `yes_ask.close` (the old `close_dollars` field is gone); the parser prefers `close_dollars` when present so both formats read as dollars, never cents.

**`fetch_all_settled_markets()` is sharded, parallel, and incrementally cached — don't serialize it again.** A sequential walk of the settled-market history is tens of millions of records at 1000/page (live-measured 2026-07-13: 95+ minutes for 8,300 pages without finishing). The fetch therefore: (a) runtime-verifies archive cursor synthesis (`_archive_cursor_synthesis_ok` re-encodes page 1's last record and byte-compares against the server cursor); (b) fetches one slice per UTC **created-day** via synthesized cursors across `SETTLED_FETCH_MAX_WORKERS` (config.py, 8) threads, persisting each completed slice to `backtest_cache/archive_days/<day>.json.gz`; (c) walks a sequential **tail** below `created_time == start_date` with the original settlement-based early-stop rule — the archive is created-ordered, so markets created before start_date can still settle in-window; the old single walk dropped ~28k such markets (verified live) as a page-alignment artifact, and the tail now catches them deterministically (new output is a verified strict superset of the old: zero markets lost, extras are all genuinely in-window settlements); (d) sweeps the post-cutoff range as parallel per-**settled-day** windows via `min/max_settled_ts`, persisting fully-elapsed days to `backtest_cache/live_days/`. Any runtime-assumption failure raises `_ShardedFetchUnsupported` and falls back to the original sequential walks (which rely only on documented behavior) — never delete the fallbacks or the runtime checks. Slice-reuse rules are load-bearing: archive day files are stamped with the `cutoff_ts` they were fetched under and are ONLY reused while it matches the current cutoff (a cutoff advance migrates markets archive-ward and OFF the live endpoint, so stale slices would silently miss them); live day files cover immutable fully-elapsed UTC days and are reused unconditionally; the frontier (current) day is always refetched and never persisted. Day slices are deliberately reused even under `--no-cache` (they cannot go stale by construction) — that's what makes refresh runs cheap, and an interrupted fetch resumes at day granularity.

**The old fetch's multi-minute stalls were client-side memory/GC, not API throttling.** The 2026-07-13 live run showed growing pauses (up to ~14 minutes between page-100 log lines, growing roughly linearly with markets held) with ZERO retry warnings logged and consistent ~0.1–0.3s server latency — the cause was accumulating millions of RAW market payloads (30+ fields plus nested `mve_selected_legs` lists) in one Python list. The rewritten fetch normalizes every record to the compact `_market_to_dict()` form at keep-time and streams day slices to disk. Don't "optimize" by deferring normalization or accumulating raw pages again.

**The live-sweep `min_settled_ts` must track `start_date`, not just `cutoff_ts` (fixed 2026-07-13).** `fetch_all_settled_markets()`'s live phase used to always pass `min_settled_ts=cutoff_ts` — correct for the common case (`start_date` far in the past, e.g. the default `2024-01-01`, where the archive already covers everything before `cutoff_ts` anyway), but catastrophic for a narrow recent `start_date`: live-verified 2026-07-13, requesting the last 7 days scanned 20k+ pages (20M+ records) over ~5 hours because the server was asked for everything since `cutoff_ts` (which can trail `now` by a long way) and the existing client-side `settle_epoch < start_ts` check just silently discarded almost every page. Since the live endpoint DOES honor `min_settled_ts` server-side, the fix is `min_settled_ts=max(cutoff_ts, start_ts)` — strictly narrows the server-side window (never widens it, so it can't skip markets the client-side filter wasn't already going to drop) and collapses a multi-hour scan to a bounded one for recent windows. Don't revert to a bare `cutoff_ts` — that reintroduces the unbounded scan for any caller that passes a recent `start_date` (e.g. a quick verification run or a future `--recent-only` mode). In the windowed live sweep this bound survives as the floor of the settled-day windows: no window's `min_settled_ts` may go below `max(cutoff_ts, start_ts)` (tests assert the minimum across all window calls).

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

**Backtest cache semantics:** `fetch_all_settled_markets()` and `fetch_candlesticks()` always persist a fresh fetch to disk even when `use_cache=False`, so `--no-cache` refreshes what the next default run loads. `use_cache` gates only the assembled per-start_date `settled_markets_*.json` file; the per-day slice stores under `backtest_cache/archive_days/` and `backtest_cache/live_days/` are consulted regardless (see the sharded-fetch gotcha for why they can't go stale). Candlestick cache files are tagged with the `[open_ts, close_ts]` window AND the `period_interval` they cover — a cache built for one `--start-date` is not reused for an earlier one, and a cache built under a different granularity (e.g. an old pre-hourly daily cache) is not silently reused as if it matched the current `CANDLESTICK_PERIOD_INTERVAL_MINUTES`. Old caches from before this tag existed migrate automatically on the next fetch, same as the legacy bare-list format.

**Daily candlesticks silently produced zero data for most markets — use hourly.** `historical.fetch_candlesticks()` requests `period_interval=CANDLESTICK_PERIOD_INTERVAL_MINUTES` (60, hourly), not daily (1440). Confirmed 2026-07 by direct API testing: `/historical/markets/{ticker}/candlesticks` with `period_interval=1440` only emits a bar for a market whose lifespan crosses a UTC midnight boundary — a market open for 2 hours entirely within one day returned 0 daily candles but 2 hourly candles for the same window. Most Kalshi markets are single-game/few-hour windows within one calendar day, so daily granularity structurally produced zero price data for most markets regardless of liquidity or age, which is why backtest runs kept finding 0 trades even after the pair-extraction logic itself was verified correct. `period_interval=1` (minute) returns HTTP 400 — hourly is the finest granularity actually available. `_candle_at_or_before()` and `_monday_timestamps()` in `backtester.py` are granularity-agnostic (they just scan sorted candles for the latest one at-or-before a timestamp), so switching interval required no change to the entry-detection logic — only the fetch call and the cache tag.

**Backtester pair-extraction eligibility prefilter — don't remove it.** At current Kalshi volumes, normalized-title groups can collapse tens of thousands of hourly/intraday markets into one group (a live measurement found a 53k-member hourly-crypto group in a 3-month window), making the naive O(n²) pairing in `_extract_pairs()` infeasible (500B+ iterations observed, hung for over an hour). `run_backtest()` filters `markets` through `backtester._can_ever_enter(m, start_date)` before either grouping call: `_find_entry()` can only open a trade at a Monday-09:00-UTC checkpoint in `[start_date, min(close_a, close_b) − 1 day]` and needs a daily candle at-or-before that Monday for both legs (i.e. each market must have opened by then) — so a market whose own `[open_time, close_time − 1 day]` window contains no Monday on/after `start_date` can never appear in any entered pair, as either leg, in either pair type, and can be safely dropped before grouping. `open_time` is carried through `historical._market_to_dict()`; **cache files written before this field existed have no `open_time` and read back as `None`**, and `_can_ever_enter()` treats a missing/unparseable `open_time` or `close_time` as "can't prove ineligibility" (keeps the market, no speedup) rather than risking an incorrect drop — re-run with `--no-cache` to unlock the speedup on old caches.

**Time-series pair extraction is close-time-windowed, not naive O(n²).** For string-keyed (time-series) groups, `_extract_pairs()` sorts members by `close_time` and sweeps a two-pointer window bounded by `MAX_DEADLINE_GAP_DAYS + 1` days of margin, since `_find_entry()` unconditionally rejects any time-series pair whose close dates differ by more than `MAX_DEADLINE_GAP_DAYS` — pairs outside that window are skipped without ever being materialized as a candidate. The `+1` day is slack only (it can only add harmless boundary candidates that `_find_entry`'s exact cutoff then rejects, never drop a legitimate one). Same-title (3-tuple-keyed) groups have no deadline-gap concept and stay naive — the eligibility prefilter above is what keeps those groups small. `config.LARGE_GROUP_WARN_THRESHOLD` (1000) logs a warning (not a cap) whenever a post-filter group is still large, as a canary that the prefilter/windowing are working as expected.

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
