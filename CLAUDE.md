# Kalshi Arbitrage Bot — Claude Context

This is a **production prediction-market arbitrage bot** that trades real money via RSA-signed Kalshi REST API calls. Read this file in full before making any changes to trading, fee, or order-submission logic.

---

## What This Is

The bot finds two types of mispriced binary contract pairs on Kalshi, sizes positions using the Kelly criterion, and submits fill-or-kill batch orders to lock in a risk-free profit. A separate backtesting pipeline replays the same strategy on the full history of settled markets.

**Time-series pairs:** Two contracts asking the same question at different deadlines (e.g. "Will BTC exceed $80k by March?" vs. "by June?"). The later deadline must have a higher or equal probability. When the earlier is priced higher, the bot buys NO on the expensive one and YES on the cheap one — all three resolution scenarios are profitable.

**Same-title pairs:** Two contracts with identical title + subtitle on different event tickers. Any divergence >5% is anomalous; the bot exploits it assuming 95% co-resolution probability. The subtitle is the intra-title outcome discriminator and is sourced from the API's `yes_sub_title` field (see the subtitle-drift gotcha below) — it is load-bearing, not cosmetic.

**Multivariate (MVE) markets:** Both pair types include MVE markets (option markets within a multi-choice event, e.g. "Trump" inside "2024 Election Winner") when `INCLUDE_MVE_MARKETS=True` in `config.py`. Pairs are grouped by `event_title + market_title` (see `scanner.pair_key`) so option labels like "Trump" or "Above $80k" can't false-positive across unrelated events. Set `INCLUDE_MVE_MARKETS=False` to fall back to binary-only behaviour.

---

## Module Map

| Module | Role | Key exports |
|--------|------|-------------|
| `config.py` | All constants and fee/threshold helpers — imported by everyone | `PROJECT_ROOT`, `BUDGET_FRACTION`, `fee_per_pair_approx()`, `fee_leg_exact()`, `min_price_diff_for_gap()` |
| `auth.py` | Builds authenticated `KalshiClient` | `build_client(mode)`, `verify_auth(client)` |
| `_http.py` | Shared HTTP retry + raw-response JSON fetch helpers | `api_call_with_retry()`, `fetch_json_page()`, `signed_request_json()` |
| `scanner.py` | Fetches markets, detects pairs, validates orderbook depth | `CandidatePair`, `ApiMarket`, `normalize_title()`, `pair_key()`, `display_title()`, `fetch_open_events_with_markets()`, `find_time_series_pairs()`, `find_same_title_pairs()`, `enrich_with_orderbook_prices()`, `filter_markets_within_horizon()`, `get_held_tickers()`, `tick_size_for_price()` |
| `strategy.py` | Kelly sizing, portfolio selection | `TradeSpec`, `compute_trade()`, `select_portfolio()` |
| `trader.py` | Order submission (V2 endpoint by default, legacy retained behind `ORDER_API_VERSION`) with atomic rollback | `execute_trades()`, `pre_execution_check()` |
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
- Tests live in `tests/` (`test_auth.py`, `test_backtester.py`, `test_config.py`, `test_historical.py`, `test_http.py`, `test_main.py`, `test_scanner.py`, `test_scheduler.py`, `test_strategy.py`, `test_trader.py`) and run fully offline against `MagicMock` clients.

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

**Use `@dataclass` for all data transfer objects.** Existing: `CandidatePair`, `ApiMarket`, `PriceRange`, `TradeSpec`, `TradeResult`, `BacktestTrade`. Never use a plain dict when a dataclass fits.

**Return `None` for validation failures; don't raise.** `compute_trade()` returns `None` when there's no edge. `find_time_series_pairs()` skips bad markets silently. Only raise for truly unexpected errors.

**Two-stage fee calculation:**
1. `fee_per_pair_approx(nA, pB)` — continuous approximation; use during pair *filtering* before the integer contract count `n` is known.
2. `fee_leg_exact(n, p)` — ceiling-rounded exact fee; use for *final validation* once `n` is determined.
Never swap these — the approximation underestimates and will let bad trades through if used for final validation.

**Retry pattern:** Use `api_call_with_retry()` from `_http.py` for all new market-data API calls (this includes read-only portfolio calls like the balance read in `auth.verify_auth()` and `get_held_tickers`; both now go through `fetch_json_page` on the SDK's raw `*_without_preload_content` variant rather than the modeled call — see the shard-balance gotcha). It handles HTTP 429 and 5xx with exponential backoff doubling 2s → 32s across 6 attempts (`_MAX_DELAY = 60s` is only a defensive ceiling). It ALSO retries status-less transport failures — `urllib3.ProtocolError`, `ReadTimeoutError`, `http.client.IncompleteRead`, and the builtin `ConnectionError`/`TimeoutError` — matched through a bounded `__cause__`/`__context__` walk (`_is_transient_network_error`), because urllib3 wraps the real error (a `ProtocolError: Connection broken: IncompleteRead` from `resp.read()` killed a multi-hour backtest fetch on 2026-08-03; it carries no `.status`, so the old status-only classifier treated it as fatal). Retrying these is safe *because* every caller of this wrapper is a read-only GET. Do NOT add retry logic to order submission in `trader.py`, on either order path — `_submit_order` (legacy) deliberately calls `fetch_json_page` directly and `_submit_order_v2` deliberately calls `_http.signed_request_json` directly (which itself contains no retry logic, precisely so that call site stays single-shot); a failed leg means the price moved, not a transient error.

**Pair dedup prefers same-title.** When both scanners detect the same ticker pair, `main._dedup_pairs(same_title_pairs, time_series_pairs)` keeps the same-title entry — its co-resolution model is simpler and doesn't depend on the time-series independence assumption. This matches `strategy.select_portfolio()`'s same_title > time_series tie-breaker. Don't flip the argument order.

**Directional, deadline-gap-tiered price filter.** `find_time_series_pairs()` requires `pA - pB >= min_price_diff_for_gap(gap_days)` (earlier-deadline leg priced HIGHER) — not `abs(pA - pB)`. A pricier later-closing contract is normal term structure, not an arbitrage; reintroducing `abs()` would create losing candidates. The threshold is tiered by the deadline gap between the two legs: 15% (`MIN_PRICE_DIFF_SHORT_GAP`) when the deadlines are ≤ 15 days apart (`SHORT_DEADLINE_GAP_DAYS`, inclusive), 30% (`MIN_PRICE_DIFF_LONG_GAP`) for 16–30 days; gaps over `MAX_DEADLINE_GAP_DAYS` (30) are never candidates. The same tier drives the orderbook price-sum ceiling (`1 - threshold`, i.e. 0.85 / 0.70) via `scanner._pair_max_sum()` in both `enrich_with_orderbook_prices()` and `validate_pair_price()`, and the backtester's `_find_entry` mirrors all of it. Never bypass `min_price_diff_for_gap()` with a flat threshold — a flat 15% admits weakly-correlated wide-gap pairs the strategy was retuned to exclude.

**Budget fitting in `compute_trade()`.** After Kelly sizing, `n` is shrunk until the fee-inclusive total cost fits within the Kelly-capped budget. Don't remove the shrink loop — sizing on price alone systematically overshoots `BUDGET_FRACTION`.

**Use `pathlib.Path` for all file paths.** Never `os.path` or string concatenation.

**Monetary values:** The Kalshi API returns balances in **cents** (int). Contract prices are in **dollars** (float, range 0–1). The conversion is always `/ 100`. Don't mix units.

---

## Known Gotchas


**Live fetching bypasses the SDK's response models (2026-07 API drift).** The Kalshi API (a) rejects events page sizes above 200 with HTTP 400 (`MARKET_PAGE_SIZE` is now 200 and must stay ≤ 200), and (b) no longer populates the legacy integer fields the pinned SDK's response models type as required: markets lost `yes_ask`/`no_bid`/… (prices now only in `*_dollars` strings) and positions lost `position`/`market_exposure`/… (count now in the `position_fp` string) — so modeled calls that deserialize nested markets, any non-empty positions page, or any order (`Order` lost `yes_price`/`fill_count`/…) raise pydantic `ValidationError`. Therefore `scanner.fetch_open_events_with_markets()`, `scanner.get_held_tickers()`, `trader._position_count()`, and `trader._submit_order()` call the SDK's `*_without_preload_content` raw-response variants and parse the JSON themselves (markets into `scanner.ApiMarket` dataclasses). Order submission is the highest-stakes case: the modeled `create_order` raises AFTER the order is placed, which would shove every real fill into the ambiguous-exception path and unwind successfully filled legs. Two rules: (1) the raw variants do NOT raise on 4xx/5xx, so every raw call must go through `_http.fetch_json_page()`, which re-raises non-2xx as `ApiException` to keep prior error semantics (`api_call_with_retry`'s 429/5xx backoff for market data; order submission stays retry-free per the rule above); (2) don't "simplify" back to the modeled calls (`client.get_events(...)`, `client.get_positions(...)`, `client.create_order(...)`) — they crash on live responses. **Event listings drifted too (2026-08).** They were previously safe through the modeled calls (no nested markets), but the live API now sends `category: null` on some events while the pinned SDK types `EventData.category` as a required `str` — so `get_events` and `get_multivariate_events` raise pydantic `ValidationError` mid-listing. Verified 2026-08-03: this killed a backtest *after* a successful 28-minute fetch. `historical._load_or_build_event_titles` therefore uses `get_events_without_preload_content` / `get_multivariate_events_without_preload_content` through `fetch_json_page` for BOTH bulk listings; the modeled `get_event` (singular) was already unusable — it embeds nested `Market` models — so the per-ticker fallback continues to use a raw GET. There is now no modeled events call left in the codebase; don't reintroduce one. Related drift: the MVE events listing is effectively unbounded (hundreds of thousands of auto-generated collection events) and currently returns zero nested markets regardless of `with_nested_markets` — the MVE pull in `fetch_open_events_with_markets()` bails out after `MVE_MAX_EMPTY_PAGES` (25) consecutive marketless pages, and the backtester's title lookup caps its MVE scan at `MVE_TITLE_LOOKUP_MAX_PAGES` (500); without those caps the scan hangs for hours (this is likely why the 2026-07-06 prod run logged nothing after auth). Don't remove the bail-outs. Same drift hits the orderbook: `scanner._fetch_orderbook()` also uses a raw GET, and exactly ONE **matched** key generation is supported, tracked by `scanner._ORDERBOOK_SIDE_KEYS` — current `orderbook_fp` ⇒ `yes_dollars`/`no_dollars` (dollar-string bid arrays). The container key strictly selects its own side keys. `_fetch_orderbook` picks the container by key *presence* (not truthiness, so an empty book still selects its own generation), then requires that container's matched side keys to exist (empty/`null` = a side with no resting bids, which is fine); a container whose matched side keys are missing, or which isn't a dict, is logged as a **potential key mismatch** and returns `None` (fail-closed) rather than reading whatever side keys happen to be present. The legacy `orderbook` container (integer-cent `yes`/`no` bid arrays) is deliberately **unsupported** — Kalshi removed every legacy integer price field on 2026-03-12 — so a legacy-shaped (or any unknown-container) response is not recognized at all and falls to the loud `No usable orderbook for %s (response keys: ...)` warning ⇒ `None` ⇒ pair untradeable. That is the point: `_bids_to_ask_levels` parses entries as dollars unconditionally, so a legacy cents bid of `35` becomes an ask of `-34.0` and is silently dropped by the price-range check, yielding an empty book with no warning (the pre-fix bug). Don't re-add the legacy generation, and don't collapse this back to a flat `data.get("orderbook_fp") or data.get("orderbook")` with an independent `yes_dollars or yes` side lookup: that decoupling is exactly the cross-generation read this guards against.

**Exchange sharding (2026-08-14) — non-shard-0 markets are dropped at ingest, fail-safe.** Kalshi has partitioned the exchange into parallel shards identified by an integer `exchange_index` on market payloads and in the account balance breakdown (shard 0 and shard 1 both observed live 2026-08-14). Each shard has its own markets, order books, and slice of account funds. The legacy create-order endpoint the bot submits through has no shard-routing parameter, so an order for a market on a non-zero shard would fail or misroute at submission time — there is currently no way to trade a non-shard-0 market safely. `config.DEFAULT_EXCHANGE_INDEX` (0) is the single source of truth for which shard our order path can reach; `scanner._on_routable_shard()` checks a raw market dict against it and `scanner.fetch_open_events_with_markets()` calls it in both the standard-events loop and the MVE loop, before a market is parsed into `ApiMarket` and before it can become a pair candidate. A missing, null, or unparseable `exchange_index` is treated as shard 0 (fail-safe — absence of the field must never empty the market list); a non-zero skip count logs a `logging.warning` naming how many markets were dropped, silent when the count is zero. This is currently a pure no-op: every Kalshi market is on shard 0 today, and the whole guard exists for the day Kalshi starts placing markets on other shards. Don't remove it — without it, a shard-1 market could become a live trade candidate that fails or misroutes at the order-submission step. The backtest path stores but does not filter: `historical._market_to_dict` passes `exchange_index` through raw into cache records (pre-existing caches read back `None`) so backtest data can distinguish shards, but no historical/backtester code drops non-shard-0 markets — all historical markets are shard 0, and filtering would silently change backtest results with no live-safety gain.

**The balance read is raw and shard-aware (2026-08-14).** `auth.verify_auth()` no longer calls the modeled `client.get_balance()` — that deserializes through the pinned SDK's strict pydantic model (`balance`/`portfolio_value`/`updated_ts` all required ints), the exact drift-fragile pattern that already broke events, orders, and positions. It now calls `get_balance_without_preload_content` through `_http.fetch_json_page()` (wrapped in `api_call_with_retry`, since it's a read-only GET), so non-2xx still raises `ApiException` and bad credentials still fail loudly. Don't "simplify" it back to the modeled call. `auth._balance_cents_from_payload()` parses the body with a three-tier fallback: (1) the `balance_breakdown` entry whose `exchange_index` equals `config.DEFAULT_EXCHANGE_INDEX` — funds parked on a shard our order path cannot reach must never inflate Kelly sizing; (2) top-level `balance_dollars`, the cross-shard aggregate; (3) legacy top-level integer `balance`, which is the sandbox shape. A breakdown that exists but has no parseable shard-0 entry logs a WARNING and falls through to the aggregate rather than reading `$0` (a false `MIN_BALANCE_CENTS` abort), and a malformed or non-dict breakdown entry is skipped rather than raising. If nothing parses, it raises `ValueError` — a deliberate loud-failure carve-out to the return-`None`-on-validation-failure convention, because sizing real trades against an unknown balance is worse than aborting the run. **Same-key-different-units trap:** inside a `balance_breakdown` entry, `"balance"` is a fixed-point DOLLAR STRING (live: `"1.1407"`); at the TOP level, `"balance"` is legacy INTEGER CENTS (live: `114`). Never route the top-level field through `auth._dollar_str_to_cents()` and never return an entry's field as-is. Dollar strings are converted with `Decimal` and **floored, not rounded** (`ROUND_FLOOR`), so the bot is never told it holds a sub-cent it cannot spend; float parsing would reintroduce the truncation noise the `*_dollars` fields exist to avoid. `verify_auth()` still returns plain `int` cents, so `main.py` and `strategy.py` are unaffected.


**`subtitle` is gone from every market payload — ingest normalizes `yes_sub_title` back into it (2026-08 drift).** Live-verified 2026-08-14: the live `/markets` endpoint, `/events?with_nested_markets=true`, and the signed `/historical/markets` archive have ALL stopped sending a `subtitle` key; the distinguishing outcome label now arrives as `yes_sub_title` (e.g. the candidate names under one shared `title` of "Who will the next Pope be?"), with `no_sub_title` carrying the negated phrasing. This is not cosmetic: `subtitle` is the only intra-title discriminator in the same-title grouping key `(event_title, title, subtitle)` used by `scanner.find_same_title_pairs()` and `backtester._group_by_exact_title()`. With every market parsing to `subtitle=""` the key collapsed to `(event_title, title)`, so two **different** outcomes sharing one question title on different event tickers grouped as the *same* contract and were traded under the 95% co-resolution assumption — a real-money false positive (reproduced offline: two distinct papal candidates paired with `tradeable=True`). The fix is at ingest only: `scanner._market_from_dict()` and `historical._market_to_dict()` read `subtitle` with a fallback to `yes_sub_title`, so the `ApiMarket.subtitle` attribute and the `"subtitle"` cache key keep their existing names and shape and no downstream code changed. Use `yes_sub_title`, never `no_sub_title` — the negated phrasing would make the grouping key asymmetric between the YES and NO framings of one outcome. Display fallbacks (`scanner.market_title`, `backtester._pair_key`, the backtester's trade labels) read the now-populated subtitle and silently improve: a title-less market falls back to its outcome label instead of the opaque ticker. **Cache caveat:** backtest day slices written before this fix carry `subtitle=None` and therefore reproduce the pre-fix weakened grouping — this is backtest fidelity only, with no live-money exposure — and `--no-cache` does NOT refresh them (day slices are deliberately reused unconditionally, see the sharded-fetch gotcha). A genuine refresh requires deleting `backtest_cache/archive_days/` and `backtest_cache/live_days/` **and** re-running with `--no-cache` (equivalently, also deleting the assembled `backtest_cache/settled_markets_*.json`) — on a default run the assembled per-`start_date` cache is loaded first and returns before the day slices are ever consulted, so deleting the slices alone changes nothing.

**Backtest fetch uses direct signed GETs — the pinned SDK has NO historical_api module.** `historical.build_historical_client()` returns a plain prod `KalshiClient`; `/historical/cutoff`, `/historical/markets`, `/historical/markets/{ticker}/candlesticks`, and `/events/{ticker}` are reached via `historical._signed_raw_get()` (KalshiAuth signs timestamp+method+path, the SDK's rest client executes) composed with `_http.fetch_json_page()` for status checks and `api_call_with_retry` for backoff. Hard-won facts about the archive endpoint (all live-verified 2026-07-13): (1) it IGNORES every server-side time filter (`min_settled_ts`, `max_settled_ts`, etc. all return the identical page); (2) its page size is hard-capped at 1000 (larger → HTTP 400); (3) it is ordered by **`created_time` DESC, ticker DESC — NOT by settlement time** (settlements interleave hours out of order, and long-lived markets created before a date can settle long after it); (4) its pagination cursor is a urlsafe-base64 **protobuf of the keyset position** (field 1 = Timestamp of `created_time`, field 2 = ticker), which `historical._encode_archive_cursor()` reproduces byte-for-byte — this is what makes sharded parallel fetching possible. The live `/markets` endpoint honors `min_settled_ts` AND `max_settled_ts` server-side (page size also capped at 1000), but does NOT serve pre-cutoff settlements — once markets migrate into the archive they disappear from `/markets` entirely. Candlestick candles now arrive with the dollar string in `yes_ask.close` (the old `close_dollars` field is gone); the parser prefers `close_dollars` when present so both formats read as dollars, never cents.

**`fetch_all_settled_markets()` is sharded, parallel, and incrementally cached — don't serialize it again.** A sequential walk of the settled-market history is tens of millions of records at 1000/page (live-measured 2026-07-13: 95+ minutes for 8,300 pages without finishing). The fetch therefore: (a) runtime-verifies archive cursor synthesis (`_archive_cursor_synthesis_ok` re-encodes page 1's last record and byte-compares against the server cursor); (b) fetches one slice per UTC **created-day** via synthesized cursors across `SETTLED_FETCH_MAX_WORKERS` (config.py, 8) threads, persisting each completed slice to `backtest_cache/archive_days/<day>.json.gz`; (c) walks a sequential **tail** below `created_time == start_date` with the original settlement-based early-stop rule — the archive is created-ordered, so markets created before start_date can still settle in-window; the old single walk dropped ~28k such markets (verified live) as a page-alignment artifact, and the tail now catches them deterministically (new output is a verified strict superset of the old: zero markets lost, extras are all genuinely in-window settlements); (d) sweeps the post-cutoff range as parallel per-**settled-day** windows via `min/max_settled_ts`, persisting fully-elapsed days to `backtest_cache/live_days/`. Any runtime-assumption failure raises `_ShardedFetchUnsupported` and falls back to the original sequential walks (which rely only on documented behavior) — never delete the fallbacks or the runtime checks. The cursor-synthesis probe issues a real request, so **any** unexpected exception from it (auth, outage, unparseable body) is wrapped into `_ShardedFetchUnsupported` too — it used to escape the phase and kill the run with no warning logged. Both pool loops call `pool.shutdown(wait=False, cancel_futures=True)` when a worker raises: the executor's `__exit__` would otherwise drain every still-queued day (hundreds of them — hours) before the fallback is even reached. **Progress-line labels are load-bearing for diagnosis:** the sharded and sequential paths emit the same line shape, so they are suffixed `[sharded]`/`[windowed]` vs `[sequential]` — with identical labels (the pre-2026-08-03 state) a log cannot tell you which path a run actually took, and a live run was misdiagnosed exactly that way. Each completed slice also logs `N/M complete` with a rate and ETA; don't drop it, it's the only signal of how far a multi-hour fetch has actually got. Slice-reuse rules are load-bearing: archive day files are stamped with the `cutoff_ts` they were fetched under and are ONLY reused while it matches the current cutoff (a cutoff advance migrates markets archive-ward and OFF the live endpoint, so stale slices would silently miss them); live day files cover immutable fully-elapsed UTC days and are reused unconditionally; the frontier (current) day is always refetched and never persisted. Day slices are deliberately reused even under `--no-cache` (they cannot go stale by construction) — that's what makes refresh runs cheap, and an interrupted fetch resumes at day granularity. **Slice records are never accumulated in memory.** Each pool worker (`_fetch_and_store_archive_day` / `_fetch_and_store_live_window`) persists its own slice and returns only a count, and each phase then rebuilds its record list by streaming the slices back off disk via `_assemble_day_slices()` — newest day first, which is the ordering the caller's first-wins ticker dedup depends on. Don't "simplify" by having workers return their records and flattening them: that is exactly what OOM'd a full-history run (2.7 GB RSS only 17% through, growing superlinearly with GC pressure), and it also drags gzip/JSON serialization back onto the main thread where it can't overlap. The prescan deliberately loads-and-discards to test slice validity, so reused days are decoded twice — minutes against a multi-hour fetch, and the alternative (partial-JSON meta peeking) is far more fragile. **The backtester's candlestick fetch (`backtester._fetch_candles_parallel`) is parallel for the same reasons**, across `CANDLESTICK_FETCH_MAX_WORKERS` (config.py, 8) threads, and uses the identical `pool.shutdown(wait=False, cancel_futures=True)` tear-down. It is safe because candle cache paths are strictly **per ticker**, so two workers can never target one path — `_save_json_cache` is a NON-atomic direct write, so a shared path would corrupt the cache, not just race it; never introduce a fetch whose cache path is shared across workers. Worker exceptions must **propagate** out of `run_backtest`: `fetch_candlesticks` already fail-softs network errors to `[]` internally, so anything that still raises is a real defect (e.g. a market that should have been prefiltered out — the run_backtest tests rely on exactly that KeyError surfacing) and must not be degraded into "this ticker has no prices".

**The old fetch's multi-minute stalls were client-side memory/GC, not API throttling.** The 2026-07-13 live run showed growing pauses (up to ~14 minutes between page-100 log lines, growing roughly linearly with markets held) with ZERO retry warnings logged and consistent ~0.1–0.3s server latency — the cause was accumulating millions of RAW market payloads (30+ fields plus nested `mve_selected_legs` lists) in one Python list. The rewritten fetch normalizes every record to the compact `_market_to_dict()` form at keep-time and streams day slices to disk. Don't "optimize" by deferring normalization or accumulating raw pages again. The same failure recurred on 2026-08-03 in compact form — the phase functions still held every completed slice in a `results` dict *and* flattened it into a second list — so slices are now dropped as soon as they're persisted and re-read at assembly (see the sharded-fetch gotcha above). If a future run stalls again, check RSS before suspecting the API.

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

**FoK price protection:** Every buy leg must be capped so it fills at or below the scanned price plus a small allowance, or not at all. On the **V2 path (default)** there is no "market" order type: a taker order is a marketable fill-or-kill LIMIT, and **the limit price IS the price protection** — `trader._v2_limit_price()` computes it as the scanned price ceiled onto the market's own tick grid (`scanner.tick_size_for_price()`) plus `BUY_SLIPPAGE_TICKS` ticks, and the buy-NO leg sends the complement `1 − cap` as an ask. The final price is then **re-quantized (ceiling) onto the grid of the band that actually contains it** — the cap can step up into a coarser band, or be mirrored across the book by the complement, where the scanned band's grid is no longer valid — and clamped one tick inside the open unit interval on this market's own grid (0.99/0.01 on a cent market, not the finest-grid 0.9999/0.0001). Ceiling is the protective direction throughout: the worst outcome is a killed FoK, never a worse-than-intended fill. On the **legacy path** (`ORDER_API_VERSION="legacy"`) the cap is `buy_max_cost` = `count × (scanned price + BUY_MAX_COST_SLIPPAGE_CENTS)`; that constant applies to the legacy path only. Any new order-building code must carry the cap appropriate to its path.

**Backtest cache semantics:** `fetch_all_settled_markets()` and `fetch_candlesticks()` always persist a fresh fetch to disk even when `use_cache=False`, so `--no-cache` refreshes what the next default run loads. `use_cache` gates only the assembled per-start_date `settled_markets_*.json` file; the per-day slice stores under `backtest_cache/archive_days/` and `backtest_cache/live_days/` are consulted regardless (see the sharded-fetch gotcha for why they can't go stale). When a `prefilter` is passed, the assembled file is named `settled_markets_<start_date>_<prefilter_tag>.json` — a prefiltered result is a strict subset, so it must never be served to an unfiltered caller or to one using different filter semantics. `run_backtest()` passes `config.SETTLED_PREFILTER_CACHE_TAG`, which names `backtester._can_ever_enter`'s semantics: **bump the tag whenever that predicate changes**, or a stale prefiltered cache is silently reused. `prefilter` and `prefilter_tag` must be passed together (`ValueError` otherwise) precisely so an untagged filtered cache can't exist. Candlestick cache files are tagged with the `[open_ts, close_ts]` window AND the `period_interval` they cover — a cache built for one `--start-date` is not reused for an earlier one, and a cache built under a different granularity (e.g. an old pre-hourly daily cache) is not silently reused as if it matched the current `CANDLESTICK_PERIOD_INTERVAL_MINUTES`. Old caches from before this tag existed migrate automatically on the next fetch, same as the legacy bare-list format.

**Backtest windows must start BEFORE the archive cutoff or they find zero trades by construction.** Live-era (post-cutoff) markets 404 on `/historical/markets/{ticker}/candlesticks` — verified live 2026-08-03: a `--start-date` after the cutoff (then 2026-06-04) produced 404s for every one of its pair tickers, so `_find_entry()` saw no candles for either leg and the run could never have entered a trade, no matter how many pairs were extracted. A near-zero trade count on a recent window is this, not a strategy result. The window always runs from `--start-date` to today, so even a correctly-chosen pre-cutoff start still has a post-cutoff tail whose 404s are unavoidable; those failures are deliberately NOT cached (a poisoned cache would silence a ticker forever, including after its data migrates into the archive), so they are re-paid on every run — just in parallel now.

**Daily candlesticks silently produced zero data for most markets — use hourly.** `historical.fetch_candlesticks()` requests `period_interval=CANDLESTICK_PERIOD_INTERVAL_MINUTES` (60, hourly), not daily (1440). Confirmed 2026-07 by direct API testing: `/historical/markets/{ticker}/candlesticks` with `period_interval=1440` only emits a bar for a market whose lifespan crosses a UTC midnight boundary — a market open for 2 hours entirely within one day returned 0 daily candles but 2 hourly candles for the same window. Most Kalshi markets are single-game/few-hour windows within one calendar day, so daily granularity structurally produced zero price data for most markets regardless of liquidity or age, which is why backtest runs kept finding 0 trades even after the pair-extraction logic itself was verified correct. `period_interval=1` (minute) returns HTTP 400 — hourly is the finest granularity actually available. `_candle_at_or_before()` and `_monday_timestamps()` in `backtester.py` are granularity-agnostic (they just scan sorted candles for the latest one at-or-before a timestamp), so switching interval required no change to the entry-detection logic — only the fetch call and the cache tag. Candles are fetched in parallel across tickers (`backtester._fetch_candles_parallel`, `CANDLESTICK_FETCH_MAX_WORKERS`); each worker still paces its own pages with `fetch_candlesticks`'s unchanged `rate_limit_sleep` default (0.15s), so the per-connection rate is the same as before — only the number of connections changed.

**Backtester pair-extraction eligibility prefilter — don't remove it.** At current Kalshi volumes, normalized-title groups can collapse tens of thousands of hourly/intraday markets into one group (a live measurement found a 53k-member hourly-crypto group in a 3-month window), making the naive O(n²) pairing in `_extract_pairs()` infeasible (500B+ iterations observed, hung for over an hour). `run_backtest()` filters `markets` through `backtester._can_ever_enter(m, start_date)` before either grouping call: `_find_entry()` can only open a trade at a Monday-09:00-UTC checkpoint in `[start_date, min(close_a, close_b) − 1 day]` and needs a daily candle at-or-before that Monday for both legs (i.e. each market must have opened by then) — so a market whose own `[open_time, close_time − 1 day]` window contains no Monday on/after `start_date` can never appear in any entered pair, as either leg, in either pair type, and can be safely dropped before grouping. `open_time` is carried through `historical._market_to_dict()`; **cache files written before this field existed have no `open_time` and read back as `None`**, and `_can_ever_enter()` treats a missing/unparseable `open_time` or `close_time` as "can't prove ineligibility" (keeps the market, no speedup) rather than risking an incorrect drop — re-run with `--no-cache` to unlock the speedup on old caches. The same predicate is ALSO handed to `fetch_all_settled_markets(prefilter=...)` so ineligible records are dropped during assembly instead of being materialized and cached first — at full-MVE volumes the unfiltered list is the difference between a working run and an OOM, and it also shrinks the `_load_or_build_event_titles` ticker set from runaway to bounded. This is result-neutral by construction (filtering during an order-preserving merge yields the same subsequence as filtering afterwards; `test_prefilter_assembly_equals_postfilter` asserts exact list equality), and the day-slice FILES stay complete and unfiltered because they're shared across start dates. **Keep `run_backtest`'s own post-fetch filter anyway** — it's idempotent, costs one pass, and keeps the guarantee local to the O(n²) pairing that depends on it.

**Time-series pair extraction is close-time-windowed, not naive O(n²).** For string-keyed (time-series) groups, `_extract_pairs()` sorts members by `close_time` and sweeps a two-pointer window bounded by `MAX_DEADLINE_GAP_DAYS + 1` days of margin, since `_find_entry()` unconditionally rejects any time-series pair whose close dates differ by more than `MAX_DEADLINE_GAP_DAYS` — pairs outside that window are skipped without ever being materialized as a candidate. The `+1` day is slack only (it can only add harmless boundary candidates that `_find_entry`'s exact cutoff then rejects, never drop a legitimate one). Same-title (3-tuple-keyed) groups have no deadline-gap concept and stay naive — the eligibility prefilter above is what keeps those groups small. `config.LARGE_GROUP_WARN_THRESHOLD` (1000) logs a warning (not a cap) whenever a post-filter group is still large, as a canary that the prefilter/windowing are working as expected.

**`secrets.json` may be missing outer braces.** `auth.py` wraps it if needed. New code must call `auth.build_client()` rather than reading `secrets.json` directly.

**`CandidatePair.max_contracts = 0` means two different things** depending on where you are in the pipeline:
- Before `enrich_with_orderbook_prices()`: not yet computed (skip depth cap).
- After enrichment with `tradeable=False`: no qualifying depth found.
Check `pair.tradeable` to distinguish.

**Backtest uses prod credentials**, not sandbox. The Kalshi historical API endpoint doesn't exist on the sandbox. `historical.py` always calls `build_prod_live_client()`.

**Dev mode skips `get_held_tickers()`** because the sandbox account is separate and has no positions from production. Don't add held-ticker filtering to `_run_dev()`.

**`normalize_title()` pattern order matters.** `_DATE_PATTERNS` in `scanner.py` applies patterns sequentially. Longer/more specific patterns must come before shorter ones to avoid partial matches leaving date fragments. If adding new patterns, prepend them before similar shorter ones.

**Event-title resolution is bounded, parallel, and must stay loud.** `_load_or_build_event_titles` resolves `event_ticker` → title in three phases: bulk `get_events` listings (settled/closed/open), the MVE listing (capped at `MVE_TITLE_LOOKUP_MAX_PAGES`), then a per-ticker `/events/{ticker}` fallback. That fallback costs **one HTTP round trip per ticker** and was written for a handful of stragglers — but a 21-day window measured **289,235 unresolved tickers** live (2026-08-03), where the original sequential, uncapped loop ground for hours emitting nothing, which is indistinguishable from a hang. It is now capped at `EVENT_TITLE_FALLBACK_MAX_LOOKUPS` (5000), run across `EVENT_TITLE_FALLBACK_MAX_WORKERS` (8) threads, and selected from a **sorted** miss list so the cap keeps a deterministic subset across runs. Tickers past the cap are stored as `""` — the exact poison pill a failed lookup already produced, so the caller's behaviour is unchanged (those markets group by market title alone). Every phase logs progress and the cap logs a WARNING naming how many were skipped: **never make this phase silent again, and never remove the cap** without replacing it with something else bounded. The bulk listings are bounded too, by a productivity bail-out (`EVENT_TITLE_LISTING_MAX_BARREN_PAGES`, 50 consecutive pages resolving nothing → move on), the same idiom as `MVE_MAX_EMPTY_PAGES`. That matters because a listing is an O(all events) scan for a specific ticker set, and `get_events` **excludes MVE events by API design** — so for an MVE-heavy miss set it can never resolve them no matter how long it pages. Live-measured 2026-08-03: the `settled` listing resolved 8,696 of 289,235 tickers in its first ~500 pages, then 9 more across the next 400.

**MVE event-title plumbing.** The live scanner attaches an `_event_title` attribute to each `Market` object inside `fetch_open_events_with_markets()`. The backtest stores it as the `event_title` key in cached market dicts. Both pair-finder paths (`find_time_series_pairs`, `find_same_title_pairs`) and their backtester counterparts (`_group_by_normalized_title`, `_group_by_exact_title`) require this to be set for MVE markets — without it they fall back to grouping by market title alone, which would let cross-event option-label collisions through. If you add a new entry point that produces markets, attach `_event_title` (or `event_title` in dict form) before passing into the grouping functions. Old backtest caches written before this change have no `event_title` field; re-run with `--no-cache` to refresh.

**Tick structure drives the V2 order price cap (2026-08).** Live market payloads now carry `price_level_structure` (a string naming the tick regime — observed: `"linear_cent"` = 1c ticks, `"tapered_deci_cent"`, `"deci_cent"` = $0.001 ticks, and `"center_deci_edge_centi_cent"` = the regime all MVE/combo markets migrated to on 2026-08-17, with $0.0001 ticks below $0.01 and above $0.99 and $0.001 ticks in between) and `price_ranges` (an array of dollar-string bands `{"start", "end", "step"}` — a tapered market has multiple bands, e.g. finer ticks near 0/1 and coarser in the middle). `scanner.ApiMarket` now carries both (`price_level_structure: str = ""`, `price_ranges: list[PriceRange] | None = None`, parsed by `scanner._parse_price_ranges`, fail-soft to `None` on anything malformed), and `historical._market_to_dict` passes both through **raw** (not parsed) into the backtest cache, since that dict is JSON-serialized directly. **`scanner.tick_size_for_price(market, price_dollars)` now reads both** to derive the tick grid a V2 order price cap must land on: it returns the `step` of the first `price_ranges` band containing the price as a `Decimal`, short-circuiting to `Decimal(config.DEFAULT_TICK_SIZE_DOLLARS)` (`"0.01"`) when `price_level_structure` is `""`/`"linear_cent"` or the bands are absent, and falling back to that same default with a `logging.warning` when no band contains the price or the matched step is nonpositive. The fallback is always safe because the grids are nested ($0.01 ⊂ $0.001 ⊂ $0.0001) — a coarse-grid price is valid on any finer grid. The reverse is NOT true: a fine-grid cap that steps up into a coarser band (or is complemented across the book) can land off that band's grid, which is why `_v2_limit_price` re-quantizes the final price onto the grid of the band containing it. Old backtest cache records (written before this change) lack both keys entirely and read back as `None`; every reader goes through `.get()`, so this is not a crash risk. **On 2026-08-17, all MVE/combo markets migrated from $0.001 to the center_deci_edge_centi_cent regime described above.** That is what made the legacy cap untenable: `trader._buy_max_cost_cents`'s FoK price protection (`BUY_MAX_COST_SLIPPAGE_CENTS`, config.py) is denominated in whole cents — 1c of tolerance, roughly "one tick" back on a $0.001 grid, now means roughly 10-100 ticks of permitted drift depending on where in the band the price sits. **Resolved by the V2 migration:** order submission now defaults to the V2 endpoint `/portfolio/events/orders` (`config.V2_ORDER_PATH`, `ORDER_API_VERSION="v2"`) — dollar-string fill-or-kill LIMIT prices, fixed-point counts, bid/ask sides on the single YES book, and an explicit `exchange_index` — where the tick-aware cap `_v2_limit_price()` (scanned price ceiled to `tick_size_for_price()` plus `BUY_SLIPPAGE_TICKS` ticks) is finally expressible. The legacy `/portfolio/orders` path (deprecated June 2026, 5x rate-limit tokens, integer-cent `buy_max_cost`) is retained **unmodified** behind `ORDER_API_VERSION="legacy"` as the instant rollback; the side/price mapping lives in `trader._V2_LEG_SIDE`. Do not attempt a cents-based tick-aware cap on the legacy endpoint — it cannot express one. Fractional-contract sizing remains a deferred follow-up (`_format_count` always emits `.00`).

**V2 order semantics — assumptions to verify at the first live submission.** The V2 mapping was built from the docs, not from an observed live order, and every piece of it is deliberately arranged so a correction is a one-line change:
- **Side mapping** (`trader._V2_LEG_SIDE`, the single point of correction): buy YES = `bid` at the capped YES price; buy NO = `ask` at `1 − cap` (selling YES you don't hold IS a long NO); the rollback closes a held NO by buying the YES short back, i.e. a **reduce-only `bid`** at the highest tradeable level of the market's own grid (`trader._v2_rollback_price()`: the finest-grid target `V2_ROLLBACK_BID_PRICE_DOLLARS` = "0.9999" floored onto the market's grid — 0.99 on linear-cent, 0.999 on deci-cent, 0.9999 on a centi-cent edge band — emulating the legacy capless `type="market"` unwind; a flat 0.99 would fail to cross asks resting in (0.99, 1) on sub-cent regimes, and $1.00 is not a tradeable level). If the exchange reads a leg the other way round, flip the value in that dict — no builder logic moves.
- **Runtime backstop on the NO-leg mapping** (`trader._confirm_v2_no_mapping()`): the side mapping above cannot be proven offline, so the **first V2 leg-A fill of each process** is sign-checked against the account's own signed positions ledger (`_position_count` on market A) before leg B is submitted. A **negative** position confirms the hypothesis (an `ask` really did open a NO position) and latches the module-level `_V2_NO_MAPPING_CONFIRMED`, so the check costs one extra positions read per **process**, not per trade — the mapping is a property of the exchange, and one confirmed negative proves it for every later trade in the run; the latch is deliberately not persisted, so a fresh process re-verifies. A **non-negative** position (zero included) means the mapping is DISPROVEN live: the pair stops at `status="manual_review"` with **leg B unsubmitted** (it would hedge a position we don't hold) and **leg A deliberately left in place** — the unwind is a `bid` resting on the same disproven hypothesis, so an automated unwind could double the error rather than reverse it (`reduce_only` would make a wrong unwind fail safe into `rollback_failed`, but a human must decide); a `logging.critical` names the remedy, `ORDER_API_VERSION="legacy"`. A **failed lookup** is unknown, not disproven: it warns, proceeds to leg B, and does NOT latch, so the check re-arms on the next NO fill — one flaky positions read must never stall trading. Don't make this check unconditional per-trade (it is a per-process property) and don't turn the disproven branch into an automatic rollback.
- **Fill classification:** a full FoK fill is assumed to report `fill_count == count` and a kill to report **2xx with `fill_count == 0`**. `_v2_fill_status()` maps only those two to the legacy `"executed"`/`"canceled"` vocabulary; a partial fill (a FoK invariant violation), a missing `fill_count`, or anything else logs `logging.critical` and **raises**. That is the design, not an oversight: raising routes into `_execute_one`'s existing ambiguous-exception path, which consults `_position_count` (the account's ground truth), so a mispredicted response shape degrades safely into `failed`/`rolled_back`/`manual_review` instead of being misread as a clean fill or a clean kill. Never "simplify" it into returning a guessed status.
- **Response shape:** the order object may arrive wrapped under an `"order"` key or flat, and counts as either `fill_count` or the fixed-point string `fill_count_fp` — all four combinations are accepted (`_parse_fixed_point` prefers the `_fp` variant).
- **Sandbox V2 support is unknown.** Dev mode never submits orders, so it cannot verify this; the smallest possible prod run is the real check. If it misbehaves, set `ORDER_API_VERSION="legacy"` — that is the whole reason the legacy path is retained intact.
- **Omitted optional fields rely on server defaults:** `self_trade_prevention_type`, `subaccount`, and `cancel_order_on_pause` are deliberately not sent.
- Other fixed choices: prices are 4-decimal dollar strings (`_format_price`, exactly representing every grid point of every regime); counts are fixed-point strings (`_format_count`); each order body's `exchange_index` is that leg's OWN market's `exchange_index` (leg A and the rollback from `market_a`, leg B from `market_b` — a pair's legs can sit on different shards) and never `-1` auto-route, so a wrong shard model is rejected loudly by the exchange instead of silently routed, and an explicit-shard write bills only that shard's rate-limit bucket where auto-route bills every nonzero shard's. The legacy endpoint has no shard-routing parameter at all, so while `ORDER_API_VERSION` is not `"v2"`, `trader._legacy_routable()` refuses any spec with a leg off `DEFAULT_EXCHANGE_INDEX` at the top of `_execute_one()` — `status="failed"`, since nothing was submitted and there is nothing to unwind; `post_only` is always `False` (we are deliberate takers) and `reduce_only` is `True` only on the rollback.

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
- **Do not add retry logic to order submission** — this covers `trader._submit_order` (legacy, via `fetch_json_page`) and `trader._submit_order_v2` (V2, via `_http.signed_request_json`) equally. One retry after a fill-or-kill rejection could submit leg A twice at different prices, creating an unhedged position.
- **Do not import `trader.py` from `scanner.py` or `strategy.py`** — circular dependency.
- **Do not write output files outside `PROJECT_ROOT`**. All logs, Excel files, and HTML dashboards write relative to `PROJECT_ROOT`.
- **Do not modify `_DATE_PATTERNS` without running `test_normalize_title`** — a regression silently causes missed pairs or false positives with no error.
- **Do not add tests that call the real Kalshi API**. Use `unittest.mock.MagicMock` for `KalshiClient`. Tests should be runnable offline.
- **Do not change the fee formula without updating both** `fee_per_pair_approx()` and `fee_leg_exact()` in `config.py`. The backtester also uses both.
