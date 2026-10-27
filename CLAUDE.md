# Kalshi Arbitrage Bot — Claude Context

This is a **production prediction-market pair-trading bot** (the "Kalshi Arbitrage Bot" is its product name; the same-title strategy is a near-arbitrage, the time-series strategy is a directional bet) that trades real money via RSA-signed Kalshi REST API calls.

This file is the short part of the project's notes: what the bot is, the rules no change may break, the run commands and an index of the rest. The rest is in `docs/claude/`, moved there word for word from this file on 2026-10-26. **Before changing trading, fee, sizing or order-submission code, read `docs/claude/what-not-to-do.md` in full and the notes the index names for the code you are changing.**

**Keep this file short.** Claude Code sends it with every request, and when it changes on disk during a session it sends the whole new file again beside the old copy (open bug anthropics/claude-code#98553). At 1.28 MB it took about 400,000 of a session's 1,000,000 tokens, and two long sessions were lost when it changed under them. New detail goes into `docs/claude/` (see Documentation Standards), and `tests/test_claude_docs.py` fails if this file grows past 40,000 bytes. Do not `@`-import the notes: an imported file loads with this one and costs the same.
---

## What This Is

The bot finds two types of mispriced binary contract pairs on Kalshi, sizes each with the Kelly criterion, and submits fill-or-kill orders leg by leg (there is no batch order endpoint), unwinding the first leg if the second does not fill. With a sell level saved (`LiveSettings.sell_at`), a production run first sells the held positions that have reached it, before it buys anything; with rebalancing on (`LiveSettings.rebalance`), it also sells, in that same step, part of each exact held pair (and of each lone leg whose earlier market paid out $0) that has grown past its Kelly share, and may then add to the pairs the account holds (DR-77). A separate backtest replays the same strategy on the history of settled markets.

- **Time-series pairs:** one question at two deadlines ("BTC above $80k by March?" and "… by June?"); `market_a` is the earlier contract. When `pB − pA` passes the run's live spread rule (`config.time_series_spread_refusal`), the bot buys YES on the earlier contract and NO on the later. It is a directional bet, not an arbitrage: of the three settlement cells, the event happening between the two deadlines loses the whole stake. Kelly's chance of profit is `1 − k × mid spread` (`config.time_series_profit_prob`, DR-78). Both finders require both legs to be worded as cumulative "by <date>" deadlines that differ (DR-67), group on title plus outcome label (`scanner.time_series_group_key`, DR-01), refuse identical wording on two events of one series (DR-02/54), and pair two rungs of one event's deadline ladder (`config.TIME_SERIES_SAME_EVENT_LADDERS`, on since 2026-09-26, DR-73).
- **Same-title pairs:** identical title and subtitle on two events of DIFFERENT series whose markets close within `config.SAME_TITLE_MAX_CLOSE_GAP_SECONDS` (1 hour, DR-74), filed under one Kalshi category and first tag and naming one competition, or none (DR-79). A price gap of 5% or more (`>=`) is traded on a 95% co-resolution prior. The subtitle, the outcome label, is read from the API's `yes_sub_title`.
- **MVE (combo) markets** are included while `INCLUDE_MVE_MARKETS=True`, grouped by event title plus market title; no same-title pair is ever formed between two `KXMVE*` combo events (DR-54/55).

The full description is in `docs/claude/overview.md`.

---

## Where the Detail Is

Every file here was moved word for word from this file, and most start with a list of their paragraphs. Read the one for the code you are changing.

| File (under `docs/claude/`) | What it holds |
|---|---|
| `overview.md` | The full "What This Is": both pair types, the three settlement cells, MVE markets |
| `what-not-to-do.md` | The full "What NOT To Do" rules (their short forms are below) |
| `run-commands.md` | Every run command with its notes: live and dry runs, one-run overrides, every backtest flag, the dashboards, the probes |
| `ci-and-packaging.md` | CI, packaging, every test file and fixture, `conftest.py`'s autouse fixtures, the golden dashboard digests, the tests that read these notes |
| `module-map/live-modules.md` | Module Map rows for `__init__`, `config`, `auth`, `_http`, `scanner`, `strategy`, `trader`, `reporter`, `seller`, `main`, `scheduler`, `run_lock` |
| `module-map/backtest-modules.md` | Module Map rows for `historical`, `treasury`, `depth_model`, `eda`, `glossary`, `market_details`, `backtester`, `dashboard`, `backtest` |
| `module-map/tools-modules.md` | Module Map rows for `v2_probe`, `defaults_server`, `live_portfolio`, `live_dashboard` |
| `module-map/dependencies.md` | The dependency order: which module imports which |
| `key-patterns/live-defaults-and-dashboard.md` | The saved live defaults, trading from the defaults server, the Live trading tab, the 2026-09-27 decision record and its addendum |
| `key-patterns/code-rules.md` | The constants rule and its list of values never to type, dataclasses, returning `None`, the two fee stages and fee multipliers, the retry rule, order and transfer pacing, `PRICE_EPSILON`, `pathlib`, money units, the exit-code contract |
| `key-patterns/selling-and-rebalancing.md` | Live selling and its sale orders, live Kelly trimming, the first-run orders for selling and for rebalancing |
| `key-patterns/pair-finding.md` | Dedup, the live category/tag filter, the live spread rule and its residual, the time-series legs and submission order |
| `key-patterns/sizing.md` | Kelly's denominator (DR-62), sizing on portfolio value, budget fitting, `select_portfolio`, marginal fill pricing, reachable depth (TS-08) |
| `known-gotchas/orders-and-v2.md` | Fee rounding, position deltas, `TradeResult.status`, the FoK price cap, order-book levels, the rollback, tick structure, V2 order semantics, the retired legacy order path |
| `known-gotchas/kalshi-api-and-shards.md` | SDK response-model drift and the order book's keys, exchange shards, coverage, cross-shard sizing, per-leg order routing (`exchange_index`, never `-1`), the balance read, collateral transfers |
| `known-gotchas/scheduler-and-credentials.md` | The scheduler's job guard (DR-59), the `KalshiClient` patch, `secrets.json`, the backtest's prod credentials, dev mode |
| `known-gotchas/time-series-pairing.md` | The inverted strategy, the cumulative-deadline wording rule (DR-67 to DR-72), same-event ladders (DR-73), the outcome label (DR-01), `_DATE_PATTERNS` order |
| `known-gotchas/same-title-pairing.md` | One series (DR-02/54/55), the close gate (DR-74), the sport and competition check (DR-79), `yes_sub_title`, MVE event titles |
| `known-gotchas/ladders-and-held-pairs.md` | One open time-series trade per ladder (DR-76, live and backtest), adding to held pairs and lone legs (DR-77) |
| `known-gotchas/forecast.md` | The mid-spread forecast (DR-78) |
| `known-gotchas/backtest-sweeps.md` | Interval-discount calibration and `k̂`, the band × `k` sweep, the tier-floors-off family, the filter bar and scenario explorer, `BacktestSweep`'s fields, the size-cap sweep |
| `known-gotchas/backtest-selling-and-rebalancing.md` | The Rebalance family and its selects, the backtest's take-profit rule, the Sell select's workers and chunk files, rebuilding from the saved sweep, the backtest Kelly trim and its wait, the live trim against it |
| `known-gotchas/backtest-valuation.md` | The equity curve, open trades at market, the modeled order book, Sharpe and Sortino net of the T-bill |
| `known-gotchas/backtest-data.md` | The settled-market fetch and its caches, memory, `min_settled_ts`, cache semantics, the UTC feasibility check, candles through today, hourly candles, the eligibility prefilter, event-title resolution |
| `known-gotchas/backtest-entries.md` | The entry checkpoint (the scheduler's run instant), the pair-extraction window, the earliest Kelly-passing Monday (DR-75) |

Where the code's comments, README.md or older plans point at a section of this file, the section now lives here: "What This Is" is `overview.md`, "Module Map" is `module-map/`, "Run Commands" is `run-commands.md`, "CI and Packaging" is `ci-and-packaging.md`, "Key Patterns" is `key-patterns/`, "Known Gotchas" is `known-gotchas/` and "What NOT To Do" is `what-not-to-do.md`, all under `docs/claude/`. A tag such as DR-74, TS-08 or BS-14 names a recorded finding: `grep -rln "DR-74" docs/claude` finds its file.

Each paragraph is one line, and some lines are long (the longest, in `backtest-sweeps.md`, is about 98 KB). Find a paragraph with `grep -n 'some words' docs/claude/<file> | cut -c1-200`, then read just that line by its number.

---

## Rules No Change May Break

Short forms, so every session has them. The full text, with the reasons and the tests that pin each rule, is in the file named beside it.

**Orders and money** (`known-gotchas/orders-and-v2.md`, `key-patterns/code-rules.md`, `key-patterns/sizing.md`):

- V2 (`/portfolio/events/orders`) is the only order path: `main.py` and `v2_probe` exit 2 on any `config.ORDER_API_VERSION` but `"v2"`. Kalshi retired the legacy order endpoints in June 2026, so there is nothing to fall back on. If the V2 path misbehaves, stop trading (stop the scheduler daemon, do not run `main.py --mode prod`, stop the defaults server and do not press Confirm and trade), then in the Kalshi UI close out a market the account did not hold before the run and put one it did hold back to what it held.
- Every order, sale and collateral-transfer POST is sent exactly once: one place on the write pacer (`trader._ORDER_WRITE_PACER`), then one `_http.signed_request_json` call, never `api_call_with_retry`. A retried order can fill twice and a retried transfer moves the money twice. The read-only position and balance reads ARE retried, except `trader._position_count_once` inside the window where a NO leg may be filled and is unhedged, and the transfer-settlement poll's `auth.read_shard_balances`.
- Which side each leg buys, and at what price, comes only from `scanner.leg_sides`, `scanner.leg_prices` and `trader._ordered_legs`. The NO leg is submitted first and is the leg unwound: `market_a` for same-title, `market_b` for time-series. Never hardcode `market_a` as the NO leg.
- A leg whose outcome is unclear is judged by the position DELTA across that order, never by what the account holds; a delta of 0 is read again once after a pause; an outcome that cannot be attributed is `manual_review`, and nothing is sent (no automatic unwind).
- The unwind is a priced, `reduce_only`, `immediate_or_cancel` YES bid capped at the loss floor (`config.ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT`), for exactly this pair's `no_leg.count`, sent only after this order's NO fill is confirmed: never an unpriced order, never sized from a position read. A part-filled unwind is `rollback_failed`, never reported as closed.
- The first V2 NO-leg fill of each process is checked against the positions ledger (`trader._confirm_v2_no_mapping`), with single-shot reads, once per process; a disproof is `manual_review` with no unwind. Keep the `_V2_NO_MAPPING_DISPROVEN` latch and `execute_trades`' one-pair-at-a-time phase with its `config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS` budget.
- Every buy leg is a fill-or-kill limit capped by `trader._v2_limit_price` (the scanned price ceiled onto the market's tick grid, plus `config.BUY_SLIPPAGE_TICKS` ticks of that grid). The exchange's HTTP 409 `fill_or_kill_insufficient_resting_volume` reply is a kill (`trader._is_fok_kill`).
- Money has three units: contract prices in dollars; balances and order costs in integer cents (but a `balance_breakdown` entry's `balance` is a dollar string, floored); a collateral transfer's `amount` in centicents, converted only by `trader._cents_to_centicents`. Kalshi's `portfolio_value` is the open positions alone, in cents.
- Kelly sizes on the portfolio value (the cash plus Kalshi's value of the open positions, kept only when `main._checked_positions_value` accepts it) and spends only the cash (`config.kelly_budget`). Kelly's `b` divides by the contract cost PLUS the fee (DR-62), in all three sizers. Use `fee_per_pair_approx` to filter and `fee_leg_exact` to validate, never the other way round, and charge each leg its own market's confirmed fee multiplier, named at every fee call.
- The time-series forecast is `config.time_series_profit_prob` of the MID spread (DR-78), never of the fills or the YES-ask gap; the entry rules still read the YES-ask gap.
- The live toggles reach live code only through the run's one `config.LiveSettings`, built from the saved `live_defaults.json` and `main.py`'s toggle flags. No saved file, a refused one, or one an older version saved means every live run exits 2; there is no fallback to config.py's toggle constants. Only `config.save_live_defaults`, called from the defaults server's confirmation page, writes that file.
- The exit code is the scheduler's only signal: 0, 10, 20, 30, 40 and 50 (`config.EXIT_*`) each mean something different. Never collapse one into another.

**Code rules** (`key-patterns/code-rules.md`):

- Constants live in `config.py` only; the one named exception is `backtester.SIZE_CAP_SWEEP`. Never type a value config.py defines (the fee rate, the caps, the tier floors, `k`, `PRICE_EPSILON`, the exit codes, the scheduled run time, …); `code-rules.md` lists the values easily confused with one another. Where a module binds a constant by value (for example `scanner.SAME_TITLE_MAX_CLOSE_GAP_SECONDS`, `backtester.SCHEDULED_RUN`, `seller.KELLY_TRIM_MAX_BID_DROP`), a test patches that module's binding, not `config`'s.
- Use `@dataclass` for data transfer objects; return `None` for a validation failure and raise only for the unexpected; use `pathlib.Path` for paths; compare prices with `config.PRICE_EPSILON`, never bare.
- New read-only market-data calls go through `_http.api_call_with_retry`, and raw SDK calls (the `*_without_preload_content` variants) through `_http.fetch_json_page`. Make no modeled SDK call: the pinned SDK's response models crash on live payloads.
- A rule the live bot and the backtest share has one definition that both call (the pairing rules, the sell and trim arithmetic, the ladder labels, the fee functions); AST pins in `tests/test_strategy.py` check this. Never apply such a rule on one path only. The time-series spread rule is the exception: the live `config.time_series_spread_refusal` and the backtest's `_find_entry` share its pieces and are held equal by value (`tests/test_backtester.py::TestLiveBacktestSpreadParity`).

**What NOT To Do** (short forms; the full rules are in `what-not-to-do.md`):

- Keep the rollback logic in `trader._execute_one()`: a YES leg confirmed unfilled after the NO leg fills is unwound at once; an unattributable outcome is `manual_review` with nothing sent; the unwind's count is `no_leg.count`; keep `trader._add_on_mismatch`.
- Add no retry to `trader._submit_order_v2`, `_submit_sale_v2` or `_execute_transfer`; keep the retries on the read-only `_position_count`; the pacer is one place before one POST, and a held write counts against its bucket until sent or released.
- Never unwind the NO leg with an unpriced order; the unwind is a floored immediate-or-cancel bid (IOC is the only time in force `reduce_only` allows). `v2_probe`'s top-of-grid closing bid is deliberate and is not a trade unwind (DR-04): leave it.
- Never hardcode `spec.pair.market_a` as the NO leg.
- Never describe, price or pay a fourth time-series settlement cell: A=YES with B=NO is impossible and is excluded as a premise violation.
- Apply the one-series rule in both finders (`scanner._same_series` in `find_same_title_pairs`, conjoined with `_identical_wording` in `find_time_series_pairs`) and on both backtester branches; a close-time gate is no substitute for it.
- Apply the same-title close gate (`scanner.closes_apart`, DR-74) on both paths, same-title only, failing closed, after the series test.
- Apply the same-title sport and competition check (DR-79) in `find_same_title_pairs`, `held_pair_kind` and the backtest, same-title only, failing closed, after the close gate. A one-sided competition is a refusal, and `main._attach_series_labels` files every market before the sale block and both finders.
- Apply the cumulative-deadline rule (DR-67) on both paths' time-series branch, failing closed; never move it into `_can_ever_enter`, never merge its tables into `_DATE_PATTERNS`.
- Order and tier a same-event ladder (DR-73) on its stated deadlines, never `close_time`, on both paths behind `TIME_SERIES_SAME_EVENT_LADDERS`; read its gap downstream only through `scanner.pair_gap_days`; fail closed at every step; branch on `is None` / `is SAME_DAY`, never truthiness.
- Keep one open time-series trade per ladder (DR-76) failing closed: a held market that cannot be identified means no time-series pair that run. Hand `held_ladders` to both `find_time_series_pairs` and `select_portfolio` (both default to an empty set, so a dropped keyword refuses nothing), resolved from the market list before held markets are dropped.
- Keep adding to held pairs (DR-77) failing closed: a `HeldPair` comes only from `scanner.held_pairs`, add only to pairs `seller.plan_add_ons` passed, and count the fees paid in a held pair's stake.
- Size on Kalshi's positions value only when the contracts held can back it (`main._checked_positions_value`).
- Decide the category/tag filter only in `config.trade_filter_for`; a tied tag excludes in its own category only and a plain tag under every category, and while anything is excluded a pair whose series the /series listing lacks is left out.
- Keep live selling failing closed: a sale's count is the plan's, never a position read; its reply is read by `_sale_fill_count`, never `_v2_fill_status`; the order is sent once; `_sale_limit` rounds the YES ask up and the NO's YES bid down; the rule's arithmetic stays in the shared functions; sales come before purchases, and the markets sold stay out of every purchase.
- Keep live trimming failing closed: size a trim on the checked positions value, take its count from the plan, and trim nothing while an order rests on the account.
- Count the rebalance wait only with `config.rebalance_rested`, from one last-trade day per path; a missing fill never reads as rested, and a price move (up or down alike) is tested only with `config.price_move_reached`.
- Do not import `trader.py` from `scanner.py` or `strategy.py`.
- Write output files only under `PROJECT_ROOT` (the one exception is the machine-wide live-run lock, `~/.kalshi_betting/live_run.lock`, which must never move under `PROJECT_ROOT`: every checkout shares it, or two real-money runs could trade at once).
- Run `tests/test_scanner.py::TestNormalizeTitle` after any change to `_DATE_PATTERNS`.
- Import `defaults_server` from nothing and never let it serve the dashboard; keep every check in front of a save or a run.
- Let the defaults server import only `config` and `run_lock`, start a process only in `_App._start_run`, and never stop a run.
- Write `live_defaults.json` only through `config.save_live_defaults`; add no config.py fallback to `main._resolve_live_settings`.
- Only a real-money prod run takes the live-run lock, and it holds it until the run ends; a lock file that cannot be made or opened is exit 1, never 50.
- Spawn the dashboard's Sell workers only under a `__main__` guard (never fork), and publish the page with its chunk folder.
- Import `v2_probe` from nothing in the pipeline.
- Add no test that calls the real Kalshi API.
- Change the fee formula only in both `fee_per_pair_approx()` and `fee_leg_exact()`.
- Never charge a leg the other leg's fee multiplier, and never leave a fee call to the multiplier's default; add a multiplier to `config.CONFIRMED_FEE_MULTIPLIERS` only on fills Kalshi charged that match it to the cent.
- Keep the fee in Kelly's denominator, apart from the reported `profit_ratio`.
- Read a live toggle only through `config.LiveSettings`, and give no internal helper a default `settings`; every live call passes `k` and the Kelly fraction explicitly (`time_series_profit_prob(k=None)` and `max_affordable_pairs(fraction=None)` read config.py), and a one-run override is a `main.py` flag, never a patched constant.
- Price the time-series forecast at the mid spread only, and keep `compute_trade` refusing a time-series pair with no usable mid spread.
- Keep enrichment's fail-closed reference, both crossed-book guards, the backtest's crossed-Monday skip, the fee truncation (`scanner._levels_with_edge_after_fee`) in both enrichment and `validate_pair_price`, and the `round(1 − k, 12)` in `config.max_kelly_fraction`.

---

## Run Commands

All commands run from the **repo root**. `docs/claude/run-commands.md` has every command, every one-run override and what each run logs.

```bash
# One-time setup (Python >= 3.11)
pip install -e ".[dev]"

# Tests and lint (CI runs both on every push to main and every pull request)
python3 -m pytest tests/
python3 -m ruff check kalshi_betting/

# Sandbox simulation with a virtual $1k balance (dev never submits orders; --dry-run matters only in prod)
python3 -m kalshi_betting.main --mode dev --sandbox-balance 1000
python3 -m kalshi_betting.main --mode dev --dry-run

# Production trades real money, from the saved live defaults (live_defaults.json): with none
# saved, a refused file or one an older version saved, every live run exits 2. One run that
# sends orders at a time on this machine (another holding the live-run lock: exit 50)
python3 -m kalshi_betting.main --mode prod
# Production discovery only, no orders
python3 -m kalshi_betting.main --mode prod --dry-run

# Backtest from a start date, starting from the account's value now (or --balance 10000)
python3 -m kalshi_betting.backtest --start-date 2024-01-01
# Rebuild the backtest dashboard from the last run's saved sweep, without a backtest
python3 -m kalshi_betting.backtest --dashboard-only
# Rebuild the Data page from the cache
python3 -m kalshi_betting.eda

# Weekly scheduler daemon (prod every Monday 09:00 on the host's clock, config.SCHEDULED_RUN)
python3 -m kalshi_betting.scheduler

# The dashboard (Live trading, Backtest and Data tabs) and the defaults server, whose pages save
# the live defaults every live run trades and can start real-money runs
./start_dashboard.sh [--seed] [--no-browser]

# Live check of the V2 order mapping (about 1c of real money; the user runs it). --step yes-close
# must PASS before selling or rebalancing is turned on
python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step no-mapping|unfillable-ask|yes-close|transfer] [--dest-shard N] [--yes]
```

---

## CI and Tests

CI (`.github/workflows/ci.yml`) runs ruff and pytest on every push to `main` and on every pull request; both must pass. Tests run offline against `MagicMock` clients, and `tests/conftest.py`'s autouse fixtures keep every test off the real live defaults, caches, logs and Kalshi. Do not bump the `kalshi-python-sync==3.2.0` pin (3.13.0 needs Python 3.13 and breaks CI), and keep `pyproject.toml`'s package-discovery table. Some tests read these notes, so editing a sentence can fail them; they read this file and every file under `docs/claude/` as one text (`tests/claude_docs.py`), so moving a paragraph between those files does not. Details are in `docs/claude/ci-and-packaging.md`.

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

**Update the README, the notes in `docs/claude/` and all relevant comments whenever a change affects behavior.** Specifically:
- If you add, remove, or rename a module: update the Module Dependency Graph in `README.md`, the module's row in the `docs/claude/module-map/` file for its area, and `docs/claude/module-map/dependencies.md`.
- If you change a constant in `config.py`: update the comment explaining the constant's purpose and rationale.
- If you add a new CLI argument to `main.py` or `backtest.py`: update the Run Commands section in `README.md` and `docs/claude/run-commands.md` (and this file's short list only when every session needs the command).
- If you change the data flow of the live or backtest pipeline: update the corresponding ASCII diagram in `README.md`.
- If you change function signatures or behavior: update the docstring in the same commit, not as a follow-up.

**Record detail in `docs/claude/`, never in this file.** A new gotcha, decision record, measured figure or recorded residual goes into the `docs/claude/` file for its area, one paragraph per line, and a new file there goes into this file's index. Change this file only when a rule every session needs changes, and then change its short form here and its full text in `docs/claude/` in the same commit. Never name a notes file `CLAUDE.md` and never put one under `.claude/rules/` (either is loaded with every request, which is the cost this split removed), and never `@`-import one. `tests/test_claude_docs.py` fails when this file passes 40,000 bytes, when a file under `docs/claude/` is missing from the index, or when the index names a file that does not exist.

A branch made before this split ports its edits of the old single-file CLAUDE.md into these notes with `scripts/port_claude_md_change.py`; its docstring gives the one-command merge that never leaves the old 1.28 MB file on disk.
