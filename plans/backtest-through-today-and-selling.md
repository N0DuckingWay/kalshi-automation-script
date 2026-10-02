# Plan: backtests through today, and selling at a share of potential profit

## Context

There are two asks. Both touch only the backtest and the dashboard; live trading is not changed.

**1. Run every backtest through the most recent available date.** Three things stop a backtest short of today, even though its "Period" line ends today:
- **Candles.** Candles come only from `/historical/markets/{ticker}/candlesticks` (`historical.fetch_candlesticks`). That endpoint returns 404 for any market that settled after Kalshi's archive cutoff, which lags about two months (2026-07-29 when read on 2026-09-28). So no trade can enter on those markets. Kalshi's docs send such markets to `GET /series/{series_ticker}/markets/{ticker}/candlesticks`, which nothing in the repo calls.
- **Cache hits.** An assembled-cache hit serves the market list exactly as it was built, at any age; it only announces how old it is.
- **Settled markets only.** The corpus holds settled markets only (see "Not in this change").

**2. A dashboard drop-down that sells a whole position early.** Its options are "no selling", then 5%, 10%, … 100%. A position is sold once its realized profit (current market value − total cost) reaches that share of its potential profit (potential total return − total cost).

Your corrections to the first draft:
- **Coverage.** The sell levels must cover every scenario the filter bar can show.
- **Combining.** Selling and adding to held pairs must combine as one strategy, while never selling and adding to a position at the same time.

## Decisions

Decisions 1–5 and 8 are my picks; the questions were declined. Decisions 6 and 7 follow your corrections. Change any of them at approval.

1. **Item 1 covers settled markets through today.** That means live-endpoint candles for recently settled markets, and the assembled cache extended automatically on each new UTC day. Positions still open today are not included (see "Not in this change").
2. **Current market value is what selling would return.** Each leg counts at the **bid** of the side it holds, minus Kalshi's taker fee on the sale (`config.fee_leg_exact`). A leg whose market has already paid out counts at its payout, with no fee. The trigger and the sale proceeds use the same number. The equity curve still values open positions at the ask, as today.
3. **Total cost and potential total return.** Total cost = contract cost + entry fees, summed over the position. Potential total return = contracts × $1 (`config.CONTRACT_PAYOUT_DOLLARS`), which is the payout in a win cell.
4. **Sales are checked weekly, at the scheduled-run checkpoint** (Monday 09:00 Los Angeles). That is when the live bot runs. A position that pays out between checks is held to settlement.
5. **A sold pair can be bought again, but only at a later checkpoint.** For time-series pairs this happens through their later Kelly-passing Mondays. Same-title pairs are only ever tried on their first passing Monday, so they are never bought again (an existing residual).
6. **Coverage is comprehensive.** All 20 levels are simulated for every scenario the bar shows: Spread band × Tier floors × k × size cap × Add to held pairs. Category and tag are slices of each scenario, as today.
7. **Selling and adding combine.** A **position** is a pair together with everything added to it: the open trades linked by a shared market, which covers lone-leg add-ons too. At each checkpoint the order is:
   - pay-outs;
   - then **sales**: any position at its level is sold whole;
   - then valuation;
   - then new trades and add-ons.

   A position sold at a checkpoint is not bought or added to at that checkpoint. A position not sold may be added to, exactly as today.
8. **"Save as live defaults…" is disabled while a sell level is shown.** A note says live trading does not sell positions; there is no live setting to carry the choice.

## Part 1 — Through the most recent date

### 1a. Candles for recently settled markets

**`historical.fetch_candlesticks(..., *, series=None, live_first=False)`**
- When `series` is given, the live path `/series/{series}/markets/{ticker}/candlesticks` is also used, through the same signed `_historical_get`.
- The live path is tried first when `live_first`; otherwise the historical path is tried first.
- A 404 retries the whole window on the other path.
- Any other failure behaves as today: one line, `[]`, nothing cached.
- Paging, parsing (`_candle_close` already reads `close_dollars`) and caching are unchanged.
- The live endpoint's per-request cap is assumed to be the same 5,000 candles. That assumption is noted as unverified.

**`backtester._fetch_candles_parallel(..., *, cutoff_ts=None)`**
- Per market: `series = historical.series_ticker(event_ticker)` (the literal prefix) and `live_first = settlement_ts >= cutoff_ts`.
- The cutoff comes from `_Candidates.corpus_provenance.archive_cutoff`.
- Routing only saves requests: the 404 fallback makes it result-neutral.
- The "N of M tickers returned no candles" WARNING is reworded.

### 1b. Retire the "structurally 0-trade" verdict (now false)

Remove these pieces:
- historical.py: `_starts_at_or_after_cutoff`, `_warn_post_cutoff`, `CorpusProvenance.post_cutoff`, and its repeat in `_announce_cache_hit`
- backtest.py: the verdict branches of `_log_corpus_provenance`
- dashboard.py: the red banner and amber notice in `_corpus_provenance_html`
- backtester.py: `max_trades_simulated`, whose only use was this verdict
- dashboard.py: `_MaxTrades`, whose only use was this verdict

The archive cutoff is still reported, as information. DR-50 becomes moot.

### 1c. Extend a stale assembled cache automatically (historical.py)

**When it extends**
- A streamed cache assembled on today's UTC date is served as today, with zero network calls.
- An older cache is extended when three things hold:
  - its meta records `assembled_at` and `archive_cutoff_ts`;
  - one `/historical/cutoff` read returns the current cutoff;
  - that cutoff is at or before the start of the cache's assembly day, D0.
- Otherwise it gets a full re-assembly (the existing miss path), with an INFO line giving the reason. That covers legacy `.json` caches, caches without a recorded cutoff, and a cutoff that has moved past D0.
- This replaces DR-13's empty-cache age rule; remove `EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS`.

**How it extends: new `_extend_assembled_cache`**
1. Call `_prune_stale_live_days(cutoff)`, then `_fetch_live_phase(live_client, max(D0_start, start_ts, cutoff_ts), now, …)`. This reuses the persisted `live_days/` slices, fetches the missing ones, and fetches today's frontier.
2. Run the existing two-walk assembly (`_count_assembled` / `_assembled_records` / identity check) over the sources in fresh-assembly order:
   - the old archive part (settle < old cutoff);
   - the new live records;
   - the old live part (old cutoff ≤ settle < D0_start).

   This drops the old partial frontier day, which the complete D0 slice replaces. `_assembled_records` gains a per-source lower bound.
3. Titles are resolved for the new records only.
4. Write atomically with `_DayStreamWriter`. The meta carries `assembled_at=now`, the current `archive_cutoff_ts`, and `extended_from`. It omits `assembly_counts`, since the dropped frontier day's counts can't be separated; the extension's own counts are logged instead.
5. Return a `SettledCorpus`. `CorpusProvenance.extended_from` feeds the dashboard corpus line and the closing log line.

**What to expect**
- With the cutoff unchanged, the result equals a fresh assembly, record for record (a test pins this).
- With a moved cutoff, the set of markets is the same.
- A network failure is a WARNING, and the cache is served as assembled.
- **Cost:** each newly completed settled day is fetched once and then shared by every window. At current combo volumes that is about 1–2 h of paging per full day, with days fetched in parallel (the 7-day window measured 1 h 47 min). Today's partial day comes on top of that.

## Part 2 — Sell at a share of potential profit (backtest)

### 2a. Constants

- `config.TAKE_PROFIT_LEVELS = (0.05, …, 1.00)` (20 literals).
- `config.DASHBOARD_SELL_MAX_WORKERS = 8`.

### 2b. Bids at each checkpoint (`LegQuotes`, `_leg_quotes`)

- **New arrays.** `yes_bid_checkpoints` and `no_bid_checkpoints`, **not carried forward**:
  - YES bid = 1 − NO ask, and NO bid = 1 − YES ask;
  - read from the latest candle ending at or before the checkpoint, only when that candle ends within one candle period of it and the opposite ask is usable (`_usable_ask`);
  - otherwise NaN, meaning no bid, so that leg cannot be sold.
- **Paid-out marker.** One per checkpoint, taken from the exact settlement time.
- **Methods.** `bid_at_checkpoint` and `paid_at_checkpoint`.
- **Fingerprint and pickling.** Both go into the fingerprint. An explicit `__reduce__` lets worker processes receive the object intact.

### 2c. The cash walk: `_simulate_at_discount(..., *, sell_at=None)`

**With `sell_at=None`:** the results are byte-identical, pinned on the golden fixture and on the add-on fixtures.

**With `sell_at` set:**
- Pass 2 also visits checkpoints that have no candidate, and keeps going after the last candidate until the last open trade pays out.
- At each checkpoint: release pay-outs (the existing block, factored into one local function), then sell.
- **Positions** are the open trades grouped by shared market (union-find on tickers), sold in entry order. One helper, `_position_sale_value`, returns the value per trade, the sale fees and the sale prices. It returns None when an unpaid leg has no bid or the position has no quotes, so trades without quotes are never sold.
- **Sell rule.** Sell when potential > 0 and realized ≥ sell_at × potential − `PRICE_EPSILON`.
- **A sale does the following:**
  - cash += value;
  - each trade of the position is replaced by its sold copy (2d);
  - its pending exit, tickers and ladder labels are released;
  - its `open_pairs` and `open_legs` entries are dropped;
  - its `pair_id` is freed for a later checkpoint only.
- **Linked records.** `pending_exits`, `active_until` and `ladders_until` gain a trade link. Their order is unchanged, so float sums are unchanged.
- **Peak Kelly fraction.** `peak_kelly_fraction` is unchanged: every Monday a pair may trade on is already scored. So `CapSweep`'s reuse above the peak stays exact, and a test pins it.
- **Logging.**
  - The run label ends ", selling at 25% of potential profit" (after ", adding to held pairs").
  - Two new lines, silent at zero: positions sold, and pairs bought again after a sale.

**Exact shortcut: `_highest_sale_ratio(point)`.** This reads a no-selling run. It returns the highest realized/potential ratio any open position reached at any checkpoint, using the same helper and the same "open at a checkpoint" rule. Any level above that ratio cannot change the run, by induction over checkpoints: the walks match until the first sale, and the no-selling run never reaches the level. A test pins this for every level on the fixtures.

### 2d. `BacktestTrade` new fields

All are defaulted, so every existing construction still builds: `sold: bool = False`, `sale_price_a/b: float | None = None`, `sale_fees: float = 0.0`.

For a sold trade:
- `exit_date` is the sale day.
- `actual_payoff` is the sale value net of `sale_fees`, so `profit = actual_payoff − total_cost − fees` still holds and the equity curve needs no change.
- `holding_days`, the ratios and `slippage` are recomputed.
- `outcome_*` and `settled_date_*` keep the markets' real results, so calibration still measures the model.

### 2e. The sell family (backtester)

**`SweepPoint` and `CapSweep`**
- `SweepPoint` gains `sell_at`.
- `CapSweep` gains `sell_at`, forwarded through `_sim_options(..., sell_at=None)`, which forwards it only when it is set, so the fixed-signature spies keep passing.
- With `sell_at`, `CapSweep` requires `eager={}` and no same-title seed, as `add_to_held` does. `sell_at` combines with `add_to_held`.
- `_half_split` and `_ex_top_event` forward `point.sell_at`.

**`run_backtest_sweep(..., sell_sweep=False)`**
- Returns `BacktestSweep.sell_sweep`, a new frozen `SellSweep`. It holds:
  - the levels;
  - the tier-on and tier-off entry maps, the very objects the size-cap and add-on families already hold, so there is no new retention;
  - the bands, ks and caps axes;
  - the end-date maps.
- `SellSweep.cell(level, band, k, *, tier_floors, add_to_held)` builds the `CapSweep(sell_at=level, …)` and reads its cell.
- Nothing is simulated during the run: one setting line and one summary line.
- It is skipped on the Monday-infeasible window.

**`backtest.py`**
- `--no-sell-sweep`: the sell family is on by default.
- `--sell-workers N`: default min(CPU count − 1, 8).
- Both are echoed in the pre-fetch summary.

### 2f. Dashboard

**The select**
- `flt-sell` sits between `flt-add` and `flt-cat`, with options "no selling", then "sell at 5% of potential profit" … "100%".
- It is rendered disabled. Its title is worded in Python and explains the rule.
- A note `flt-sell-note` reads "(not simulated in this backtest)" or "(not available on this page; see the log)".

**Grids (base block)**
- `grid_sell[level][tier][add]` → `[band][k][cap]` chunk ids, covering every non-null cell of the page's four grids (on, off, add, add_off). Binding-band rules alias as `off_grid()` does.
- `sell_levels`, `sell_state`, `inline_chunks` (how many chunks are inline) and `sidecar_dir`.

**Parallel build** (new `_build_sell_grid`, after the existing walk)
- One task per (tier setting, band), carrying:
  - that band's entries;
  - the walked grid's ks, caps and base cells;
  - the levels;
  - the page axis, labels (`cat_index`, `sub_index`), series categories and `risk_free`;
  - the parent's resolved `SAME_TITLE_SIZE_CAP` and `SCHEDULED_RUN`, which the worker installs.
- For each cell and add-on setting, a worker:
  1. simulates the no-selling run;
  2. takes `_highest_sale_ratio`;
  3. marks every level above it "same as base";
  4. simulates the rest through `SellSweep.cell`, so caps at or above the peak share one run;
  5. builds `_list_payload` with a chunk-local row-head table;
  6. writes each chunk whose `_list_key` is new and not already inline as a sidecar file.
- The parent maps "same as base" to the base grid's id and inline-key matches to the inline id, then numbers the sidecar files.
- Pool: `ProcessPoolExecutor` with the spawn context (safe on macOS). With `--sell-workers 1`, which tests use, it runs in-process.
- A failed task is a WARNING naming its band; that band's cells stay null. Progress is logged per band.

**Sidecar files**
- Chunks go in `backtest_dashboard_files/<build id>/chunk-<id>.js`, each holding `window.__dashChunk(document.currentScript, "<gzip+base64>")`.
- The folder is written first. The page is then replaced atomically, and older build folders are deleted.
- The HTML stays the size it is today, plus about 1 MB for the sell grids (estimate).
- Add the folder to `.gitignore`.

**`_FILTER_JS`**
- `SHOWN[7]` holds the level.
- `gridAt(t, a, s)` picks the sell grid.
- `load(id)` reads inline chunks as today and loads an id ≥ `D.inline_chunks` through a `<script src>` element. The text is inflated by the same code (`inflate` split so the text can come from either place).
- Load failures are worded from `D.text`.
- Row heads come from `C.heads` when present.
- `isPrimary` requires no selling, and `saveHref` returns null while selling.
- On load the select resets to "no selling"; it is enabled only when `D.grid_sell` is present.
- The explorer call keeps 4 arguments.

**Text and pages**
- `scenarioAt` gains the sell phrase. A `sell_note` says each sale is its trade's exit and its slippage is what it made below the win-scenario profit.
- `_bar_reach` gains `_SELL_REACH`: the trade sections only; never the k̂ figures, the explorer or the Interval Discount section.
- `_trade_row_head` words a sold trade's legs "sold on <day> at $<bid> (settled <X> on <day>)".
- No golden-pinned section text changes.

### 2g. Cost

All three figures are estimates from the size-cap sweep's cost, not measurements; the build logs the real figures.
- **Work:** up to 20× today's size-cap and add-on walks. The exact shortcut and cap sharing remove part of it, and the workers divide the rest.
- **Time:** for the 365-day run, roughly 30–60 minutes of extra build with 8 workers.
- **Disk:** roughly 1–5 GB of sidecar files.

`--no-sell-sweep` skips the whole family.

## Docs

- **README:** backtest options (`--no-sell-sweep`, `--sell-workers`); remove "start-date should predate the cutoff"; rewrite "Cached runs say what they cover"; a new "Selling at a share of potential profit" section; and a note that the dashboard is now the HTML plus its `_files` folder, which must be kept together.
- **CLAUDE.md:** Module Map rows; constants (`TAKE_PROFIT_LEVELS` and `DASHBOARD_SELL_MAX_WORKERS` added, `EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS` removed); Run Commands; rewrites of the archive-cutoff, DR-13/P2 and DR-50 passages; new gotchas for the extension, the sell rule, positions with add-ons, the exact shortcut, workers and sidecars.
- **todo.md:** drop DR-50 and add the residuals below.

## Tests

- **`tests/test_historical.py`**
  - candle routing and the 404 fallback in both directions, plus paging on the live path;
  - a same-day hit makes zero calls;
  - a next-day hit extends the cache and equals a fresh assembly;
  - a moved cutoff, a legacy cache, or a cache with no cutoff each rebuilds;
  - an extension failure warns and serves the old cache;
  - the verdict tests are updated.
- **`tests/test_backtester.py`**
  - bids and paid-out markers;
  - the sell rule: positions with add-ons and lone legs; fees; no bid; no quotes; checkpoints with no candidate; checkpoints after the last candidate;
  - buying again later but not at the same checkpoint;
  - no add at the checkpoint of a sale; releases;
  - the curve sums to profit;
  - `sell_at=None` is identical to today;
  - every cap of sell `CapSweep`s (adding on and off, tiers on and off) equals a fresh simulation;
  - `_highest_sale_ratio`'s shortcut is exact;
  - `sell_sweep` adds no simulation.
- **`tests/test_dashboard.py`**
  - TestSellGrid: grid shape, aliasing, the "same as base" mapping, and failure isolation;
  - in-process versus a 2-worker spawn pool give identical grids and files;
  - TestSellScript: the node harness gains `<script src>` loading with deferred resolve and reject, sidecar loading, the summary, the disabled Save, reset on load, and the explorer call;
  - TestSellEndToEnd: a real sweep's sidecar KPIs equal fresh simulations;
  - sidecar atomicity and cleanup;
  - the sold-trade row head;
  - the header without the verdict.
- **`tests/test_backtest.py`:** the flags and the closing line.

## Verification

1. `pip install -e ".[dev]"`, then `python3 -m ruff check kalshi_betting/` and `python3 -m pytest tests/ -q`. Node is present, so the JS harness tests run.
2. Build a dashboard from the golden fixture with the sell family and `--sell-workers 2`, then check the select, sidecar loading, a sold trade's row and the disabled Save.
3. Run `/code-review high` on the diff and fix its findings before pushing.
4. **To run on your machine** (Kalshi is unreachable from this container):
   - a backtest starting two weeks ago: the no-candles count drops and trades appear;
   - two runs on consecutive UTC days: the second extends the cache through today;
   - the 365-day build: its logged sell-family time and the size of the sidecar folder, against 2g.

## Branch and commits

Work happens on `claude/backtesting-position-auto-sell-hsj4d4`, which is already at `origin/main` 0830eec. The commits:
1. Live candles and verdict removal
2. Cache extension
3. Sell rule, `SellSweep` and CLI
4. Dashboard grids, workers and sidecars
5. Docs

Push with `git push -u origin claude/backtesting-position-auto-sell-hsj4d4`. No PR unless you ask.

## Not in this change (recorded residuals)

- **Positions still open today.** The corpus is settled-only, so the last weeks of the window hold only pairs that have already paid out. A pair sold before its markets settled is also missing until they settle.
- **Fill depth.** Sales fill at the candle's bid in any size, as entries do; the backtest has no order-book depth.
- **Sale fees.** Fees are charged per trade per leg, so a position with add-ons can pay up to 1¢ per extra trade per market more than one combined order would.
- **Live selling.** Live trading never sells. Building a live sell path would be a separate change.
- **Combo fetch cost.** Skipping MVE combo markets in the backtest fetch is a separate decision. They form no pairs (measured) but dominate every extension's cost.
