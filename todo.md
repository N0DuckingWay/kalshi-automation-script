# To-do list

Compiled 2026-09-29 against `main` @ `b5c2197`. Every item below was checked against the current
code, not just copied from `CLAUDE.md`/`BUG_SWEEP_FINDINGS.md`; line numbers are as of that commit
and will drift. Items already fixed are listed at the bottom so nobody re-opens them.

Legend: **[live]** touches real-money trading · **[bt]** backtest fidelity · **[ops]** needs live
data or the operator's machine · **[decision]** needs an operator call before any code

---

## 1. Live trading path

- [ ] **[live] Order POSTs have no request timeout.** `_http.signed_request_json` calls
  `client.rest_client.request(...)` with no timeout (`_http.py:470`), so a stuck order POST blocks
  `execute_trades` indefinitely (`trader.py:2800-2803`, `config.py:757-759`). Needs a timeout whose
  expiry is routed into the existing ambiguous-submission path (never a retry).
- [ ] **[live] Scheduler loses the exit code on non-UTF-8 child output.** `subprocess.run(...,
  text=True)` decodes strictly (`scheduler.py:591-599`); one bad byte raises `UnicodeDecodeError`
  *after* the prod run finished, so an exit 20 (trades need attention) is never logged or finalized.
  Fix: `encoding="utf-8", errors="replace"`.
- [ ] **[live] Buy-leg price cap can loosen by one coarser-band tick.** Ceiling re-quantization in
  `scanner.v2_limit_price` (`scanner.py:644-647, 680-683`) and the rollback cap (`trader.py:831-834`)
  can land up to one destination-band tick looser when the cap crosses into a coarser band. Recorded
  in CLAUDE.md as "its own open finding"; no fix designed.
- [ ] **[live] Book can move after `validate_pair_price`.** With tier floors off (shipped), a
  thin-edge spec can still fill within the FoK tick of slippage and lose in the win cells
  (`config.py:455-458`, `scanner.py:5271-5273`). No check between validation and submission.
- [ ] **[live] Disproof latch doesn't survive the process.** `_V2_NO_MAPPING_DISPROVEN` is
  in-memory (`trader.py:343-354`); after a genuine disproof every later scheduled Monday run opens
  one more wrong-side position until a human stops the daemon. Consider a persisted stop file the
  scheduler/main refuse to run past.
- [ ] **[live] Write pacer is per-process.** A manual run overlapping the scheduled one (or
  `v2_probe --step transfer`) has its own full-rate pacer (`trader.py:705`). Consider a cross-process
  lock or refusing concurrent prod runs.
- [ ] **[live] Ladder quality screens not built.** No intervening-rung staleness guard
  (`config.py:306-310`; 1 of 48 emitted pairs on the 2026-09-22 snapshot has an out-of-order rung),
  and no spread/liquidity screen for ladders, whose pairs are all market-EV-negative by the later
  leg's bid-ask spread (`config.py:273-279`).
- [ ] **[live] Ladders the stated-deadline reader can't place are missed.** Year-less spans
  (69 of 4,161 live rungs, per the comment) and numeric m/d dates are refused by `_span_deadline`
  (`scanner.py:1626-1633, 1733-1738`). Fails closed.

## 2. Verify on live data  [ops]

- [ ] **Paid-out positions leave `count_filter="position"`.** DR-76's held-ladder resolver assumes
  so (`scanner.py:2394`). If not, a paid-out market 404s on lookup, `resolve_held_ladders` returns
  None, and **every** run skips time-series and exits 40 (`scanner.py:2586-2674`, `main.py:1005-1022`).
  Check on the first live positions after they settle.
- [ ] **HTTP 409 FoK kill ⇒ nothing filled.** Rests on one observation of an order with no crossing
  depth (`trader.py:1108-1139`). Not seen for a FoK that crosses *some* depth.
- [ ] **Immediate reduce-only unwind after a killed YES leg.** Sent ~one round trip after the NO
  fill; never observed live. If the exchange checks `reduce_only` against a lagging position it ends
  `rollback_failed` (`trader.py:1227-1239, 1491-1498`).
- [ ] **Self-trade-prevention cancel response shape** has never been observed (`config.py:676-681`).
- [ ] **`fill_count` vs `fill_count_fp`** — which one production sends is "not yet verified live"
  per `trader.py:1031-1034` (the 2026-09-28 responses may already settle it; update the comment).
- [ ] **`TRADER_MAX_WORKERS` never exercised at scale** against the live API (`config.py:1351-1358`).
- [ ] **Existing `trade_log.xlsx` keeps its old header row** (no migration; `reporter.py:213-216,
  432-434`). Decide whether to rewrite headers once.
- [ ] **Reduce-only unwind and NO buy against an existing, larger NO position.** Never observed live
  (the probe only closes from flat); watch the first add-on rollback, and the case where a
  process's first NO fill is an add-on (`trader.py`, `_add_on_mismatch`/`_rollback_no_leg`).
- [ ] **First `--add-to-held-pairs` dry run.** Before adding to held pairs goes on for every run,
  follow the first-run order (README "Save the live defaults"; the deploy steps in `CLAUDE.md`'s
  saved-live-defaults paragraph): save the seed values with adding off, run
  `python3 -m kalshi_betting.main --mode prod --dry-run --add-to-held-pairs`, check that its
  `Sizing on portfolio value` line shows the open positions as a dollar figure (not `not read`) with
  no WARNING refusing them, check each `Held pair to add to: …` line's cost and fees against its fills
  in the Kalshi UI and its `worth $W at today's prices` against the asks there, then turn adding on. That
  `market_exposure_dollars` is the cost without fees, positive for a NO position, and falls with a
  partial close is inferred from the API reference's one-line description, not observed
  (`scanner.get_held_positions`, `scanner.held_pairs`; DR-77 in `CLAUDE.md`). Nor has a live
  `portfolio_value` been checked against `main._checked_positions_value`'s $1-a-contract bound, or
  compared with the held pairs' worth at the asks (Kalshi's own valuation; the two need not agree).

## 3. Pending operator decisions  [decision]

- [ ] **DR-50: short-circuit post-cutoff backtest windows** (structurally 0-trade). Today only a
  WARNING (`historical.py:4104-4125, 5122-5134`). Must go on the cache-miss path after the cutoff
  read, never on a hit.
- [ ] **DR-75 follow-up: make the same-title group contest causal.** `best_by_group` still picks the
  largest `entry_monthly_ratio` across different Mondays — look-ahead (`backtester.py:6141-6157`) —
  and the ratio's horizon uses the *realized* close (`backtester.py:6064-6069`). Options: earliest
  Kelly-passing candidate wins, and/or horizon to the stated deadline.
- [ ] **k vs measured k̂.** Live k 0.80 sits below the measured k̂ 0.87/0.92 in the >0.60 band; only
  the 0.5 band ceiling refuses those spreads (`config.py:330-333`).
- [ ] **Same-title close-gate bound: 1 h vs 15 min.** 15 min gives the same result on the 365-day
  corpus with ~1 h more margin (CLAUDE.md DR-74 table; `config.py:219`).
- [ ] **Widen market-eligibility bounds (0.01/0.99)?** Deliberately held (`scanner.py:458-464`).
- [ ] **Same-title stacking across weeks** is out of scope of DR-76: its held-ladder rule refuses
  time-series pairs only (in `find_time_series_pairs` and `select_portfolio`), so a later run can
  trade another same-title pair of a question it already holds one on; and with `add_to_held_pairs` on it may also add to the
  exact same-title pair it holds (DR-77). Confirm that's still the intent.

## 4. Backtest fidelity  [bt]

- [ ] **Re-measure the headline numbers after DR-75/DR-76 and the v3 checkpoint prefilter.** Ladder
  backtest figures and RAM-warning eligible counts in `config.py` are marked "not re-measured"
  (`config.py:352-355, 1176-1179`).
- [ ] **Re-measure the add-on figures now that a held pair's stake counts its fees** (operator
  decision 2026-09-30) and trades are sized on the portfolio value: the Kelly chart's smallest
  add-on y/x (0.002, CLAUDE.md's add-on select paragraph), and the add-on family's times and page
  sizes (CLAUDE.md's add-on family paragraph, README's "Cost of the family"). The golden fixture's
  add-on counts and sizes were re-measured on 2026-09-30 (CLAUDE.md's backtest add-on paragraph).
- [ ] **No candle-staleness bound and no crossed-book guard in `_find_entry`**, while live fails
  closed on both (`backtester.py:3826-3834, 3852-3945`; `scanner.py:5111-5115`). Will move every
  result when fixed.
- [ ] **Same-title pairs Pass 2 can't fund are never retried** (time-series are, since DR-76)
  (`backtester.py:5800-5804, 6253-6308`).
- [ ] **Cross-event time-series pairs ordered on realized close**, live on scheduled close
  (`backtester.py:3725-3731, 3765-3795`). Needs a scheduled-close field (refetch) or span-date
  parsing, as ladders already have.
- [ ] **No order-book depth in the backtest**: fills assumed at unlimited size at the candle close;
  no FoK reachability or depth-weighted pricing (`backtester.py:6280-6285`).
- [ ] **`no_ask_close` clamped to [0.01, 0.99]** in `historical.fetch_candlesticks`
  (`historical.py:5588-5592`): a dead NO leg reads as tradeable, sub-cent books are distorted, and
  `yes_ask_close` is *not* clamped despite its docstring.
- [ ] **DR-76 residuals** (`backtester.py`): voided/premise-violating pairs never hold their ladder
  (proposal: claim-only candidates, 5809-5811, 5999-6028); ladders freed at start of payout day
  (6233-6244); blank cached `event_title` widens the question label (2342-2344); an H1 pair only
  takeable after the split trades in neither half (6949-6953).
- [ ] **Same-day cash release**: a receipt settling later on the entry day funds that morning's
  checkpoint entries (`backtester.py:6218-6220, 6246-6251`).
- [ ] **Split-half gaps unwarned outside the primary band** — empty H1 when half the entries share
  the earliest date (`backtester.py:6918-6922, 7682-7696`).
- [ ] **Subtitle-blank corpora change which ladders exist** (4 admitted that live refuses, 136
  missed on the measured snapshot; `backtester.py:2228-2261`). The k̂ gating the ladder switch was
  measured on such a corpus — re-measure on a subtitle-complete corpus.
- [ ] **Mark-to-market equity curve** (drawdown is realized-only today; `backtester.py:8220-8238`).
  Deferred as a data-threading change, not a new fetch.
- [ ] **`_live_rule_view` duplicates `dashboard._tier_off_binds`'s rule inline**
  (`backtester.py:5344-5377`) — extract one definition.
- [ ] **Tier-off calibrations are never logged**, only shown on the page (`backtester.py:6576-6579`).
- [ ] **A malformed candle anywhere in the scan window now raises** out of `_find_entry` (since
  DR-75 reads past the first hit; `backtester.py:3514-3516`). Cache holds none today.

## 5. Data fetch and cache robustness

- [ ] **Unbounded cursor loops in the settled-market fetch.** `_fetch_archive_day`
  (`historical.py:2567-2610`), `_fetch_live_window` (3172-3211) and `_fetch_live_sequential`
  (3271-3302) have no seen-cursor guard and no page cap (unlike scanner's TS-05 loops); a cycling
  cursor on an in-window page loops forever. Since 2026-10-01 `_fetch_archive_day` runs once per
  created-day from `ARCHIVE_FIRST_CREATED_DATE` (~1,900 days) rather than from `--start-date`.
- [ ] **`/series` cache write collision** between a concurrent live run and backtest can leave an
  unreadable file (tmp name derives from the destination; `historical.py:398-409, 678-687`). Same
  for `treasury_bill_rates.json`. Use a unique tmp name.
- [ ] **Legacy `settled_markets_*.json` hits still load whole into memory** (`historical.py:5086-5110`)
  and the RAM warning doesn't count that list (`backtester.py:5054-5076`). Simplest fix: delete the
  legacy files (see §11).
- [ ] **Pruning cached market dicts to the keys the backtester reads** — specified, unlanded, gated
  on profiling (peak RSS ≤ 3 GB, identical `Potential pairs` counts).
- [ ] **Candle cache dict with `open_ts` but no `close_ts`** raises `KeyError` out of the worker
  (`historical.py:5528-5532`); only via a hand edit.
- [ ] **DR-51 standing cost**: a cache-miss re-fetch of an already-seen window pages the bulk
  event-title listings to their barren bail-out (`historical.py:1088-1153`). Accepted; revisit if
  it gets slow.

## 6. `v2_probe` residuals

- [ ] `_step_unfillable_ask`: the unreadable-fill-counts branch FAILs with no position read
  (`v2_probe.py:1352-1355`); the `not killed` branch uses one un-refreshed read and never says
  FLATTEN (1360-1367).
- [ ] Both steps' non-kill exception handlers skip `_recheck_and_report_position`
  (`v2_probe.py:1008-1017, 1318-1321`); `_step_no_mapping`'s `if after_err:` treats a failed lookup
  as flat.
- [ ] Closing order raising prints "may still be OPEN" without reading the position
  (`v2_probe.py:1190-1197`).
- [ ] Fill reported but first position read is `None`: no re-read, no FLATTEN (`v2_probe.py:1149-1151`).
- [ ] A 2xx kill with a non-zero/`None` position is judged on one read; the 409 path re-reads
  (`v2_probe.py:1144-1148, 1368-1373`) — make them symmetric.
- [ ] A first read of exactly 0 after a kill is judged at once; the close is judged on one 1 s
  re-read, while `trader` uses 1/2/4 s (`v2_probe.py:842-846, 1103-1104, 1200-1209`). Align with
  `V2_MAPPING_ZERO_RECHECK_DELAYS_SECONDS`.
- [ ] Exceptions printed whole (multi-line SDK text) at `v2_probe.py:1009, 1194, 1319, 1404, 1668`;
  use `_http.api_error_summary`.

## 7. Observability and logging

- [ ] Multi-line exception logs remain at `scanner.fetch_shard_statuses` (`scanner.py:2945-2949`),
  `main._run_prod`'s post-trade balance read (`main.py:1093-1096`) and the retry WARNING
  (`_http.py:516`).
- [ ] Scheduler's catch-all failure branch dumps the child's whole stderr as one entry
  (`scheduler.py:674-675`).
- [ ] Rename the fetch's "N markets kept so far" progress lines, which count pre-prefilter records,
  not eligible markets (`historical.py:1606-1618, 2839-2842, 3292-3295`).
- [ ] Uncounted refusals: price-parse, non-positive-spread and the same-title 5% gate
  (`scanner.py:3585-3587, 3831-3843, 4129-4131, 4262-4281`); the `seen` guard in `_extract_pairs`
  (`backtester.py:2592-2596`).
- [ ] Consolidate the three one-line error describers (`historical._exception_summary`,
  `scanner._error_text`, `_http.api_error_summary`); `treasury.py:47` also imports historical
  privates.
- [ ] Ladder rows render `close_time` as "A/B Deadline", not the stated deadline the pair was
  ordered and tiered on (`main.py:480-481, 498`, reporter, dev simulation).

## 8. Wording-classifier residuals (`scanner.py`)

All currently produce false cumulative verdicts/spans or false refusals; none has a live
time-series partner today (per CLAUDE.md's measurements). Confirmed by probe on 2026-09-29.

- [ ] Season ranges ("2018-19 through 2025-26" → `through 2025`), 2020–2099 quantities
  ("increase by 2030 units"), fractions ("by 1/2 point"), month-named proper nouns ("by April
  Ryan", "Mar-a-Lago"), "by the end of the weekend" → `by the end of the week`
  (`scanner.py:329-358`).
- [ ] Weekday + numeric date ("by Friday, 9/19/2026") truncates to `by friday` → false refusal
  (`scanner.py:331-339`).
- [ ] "Level at one instant" ("SOFR above X by end of <period>") reads cumulative; needs a predicate
  classifier (not designed). Held back only by the 30-day gap cap.
- [ ] Non-monotone predicates ("exactly N by <date>") pass the ladder branch with no predicate check
  (`scanner.py:3653-3760`); unmeasured on ladders.
- [ ] DR-26: bare 4-digit strike inside a *title* is eaten by the year pattern (`scanner.py:222-223`).
- [ ] Unanchored explicit-date pattern (index 12): a bare "Sep 14" outcome label normalizes to ""
  (`scanner.py:245-247`).
- [ ] Same-title close gate: family-wide placeholder closes (e.g. `KXOSCAR*`) pass live at 0 s; a
  naive/aware close mix is counted on a line whose wording is then false (`scanner.py:1178-1185`).
- [ ] Rolling "before <date>" windows pass the screen — measured harmless; revisit only if a series
  co-lists mid-window.

## 9. Dashboard

- [ ] Plotly.js is pinned to `plotly-2.30.0` on the CDN (`dashboard.py:9770`) while plotly.py is
  `>=5.18` and goldens were captured on 6.9.0 — check the pairing (plotly.py 6.x targets plotly.js
  3.x, to my knowledge; unverified here).
- [ ] Size-cap option/summary labels ("off (full Kelly)", "no per-trade cap") don't say same-title
  stays at 20% (`dashboard.py:6222-6261`).
- [ ] `_performance_kpis` / `_strategy_row` `iloc[-1]` on an empty equity frame (`dashboard.py:1149,
  1224`; unreachable today).
- [ ] Category/tag names reach Plotly axis labels unescaped (Plotly's limited HTML subset; low risk)
  (`dashboard.py:1511-1522, 2902, 8072-8075, 9086`).
- [ ] Page-size figures in `_packed_json_script`'s docstring mix designs and are not re-measured
  (`dashboard.py:8594-8647`).
- [ ] An `interval_discount` override ≠ `sweep.primary.k` is unsupported (WARNING only;
  `dashboard.py:9567-9578`). Never hit in production.

## 10. Code hygiene and packaging

- [ ] Fractional-contract sizing: `_format_count` always emits `.00` (`trader.py:874-889`); fill
  classification also truncates via `int(Decimal(count))` (`trader.py:1214`).
- [ ] Literals outside `config.py`: page cap `1000` (`historical.py:3173, 3272, 5143`),
  `rate_limit_sleep=0.15` (5452), `_exception_summary limit=120` (462).
- [ ] Move `SIZE_CAP_SWEEP` from `backtester.py` to `config.py` (one line).
- [ ] Remove dead `markets=None` fallback fetch in `find_time_series_pairs` (`scanner.py:3446-3486`).
- [ ] `pyproject.toml`: plotly/pandas/numpy unpinned so the dashboard golden test *skips* in other
  environments (CI has nothing turning that skip into a failure); `scipy` is a runtime dep only
  tests use; `[tool.mypy]` configured but mypy isn't installed or run; no `[build-system]` table.
- [ ] `scheduler.py:173-192` suggests a crontab line writing to `/tmp/kalshi_arb.log` (outside
  `PROJECT_ROOT`) and bypassing the daemon's timeout/catch-up/retry.
- [ ] `backtest.py --balance` is not validated (0 or negative accepted; `backtest.py:286-289`).
- [ ] `display_title` renders the name twice when the title ends with the subtitle
  (`scanner.py:2226-2232`; cosmetic).

## 11. Documentation drift

- [ ] **README.md:17 says the bot sizes on "75%"** of the in-between mass; `config.py:427` is 0.80.
- [ ] README.md:11, :33 still describe the tier margin; the shipped rule is any positive spread ≤ 0.5.
- [ ] `kalshi_bot_flowchart.pdf` (last touched 2026-08-03) shows `|pA−pB|`, `pA > pB` and the wrong
  dedup order — regenerate or delete.
- [ ] Mark `BUG_SWEEP_FINDINGS.md` historical (all BS-01..32 fixed in #37 `e280cc4`; line refs stale).
- [ ] Stale "unverified hypothesis" wording for the V2 NO-leg mapping, confirmed live 2026-09-28:
  `v2_probe.py:14-20`, `trader.py:234-236, 900-901, 946-948`.
- [ ] `trader.py:1541` error string says "rollback FoK" — the unwind is IOC. Same in `__init__.py:12-18`,
  which also still says the later leg must be "well above".
- [ ] `scanner.py:681-682` ("ceiling is protective") and `scanner.py:537-542` contradict the
  one-tick loosening above.
- [ ] `config.py:87-95` (`SIZE_SOLVE_MAX_ITERATIONS` describes a descent, code bisects),
  `config.py:1525-1531` (`MVE_MAX_EMPTY_PAGES` wording), `config.py:1792-1794`
  (`run_backtest` → `_simulate_at_discount`).
- [ ] `backtester.py` docstrings: 2525-2559 (wrong caller names, "two-pointer" window, dedup
  location), 99, 834-836, 2433-2447, 4927-4931; `backtest.py:557-558` ("six sections" → nine).
- [ ] `historical.py`: 618-623 ("all historical markets are shard 0", "guarded at scanner ingest"),
  576-615 (tick fields "groundwork" — shipped live), 5499-5500, 5512, 636-640, 678-684.
- [ ] `reporter.py:33` points to BS-18 "in CLAUDE.md" (it's in `BUG_SWEEP_FINDINGS.md`);
  `auth.py:18-19` caller attribution; `v2_probe.py:699-700` call-site count;
  `_http.py:485-487` ("only market-data calls" — also portfolio reads);
  `scheduler.py:406, 520, 570` hardcode "Monday slot".
- [ ] Missing dependency edges in CLAUDE.md/README graphs: `dashboard → scanner`,
  `backtest → historical`, `historical → _http` (CLAUDE.md only), `reporter → scanner`.
- [ ] README.md:316 (check "below" is actually above), :1041 (undated "this upgrade"), :1075
  (dev mode skips held positions regardless of sandbox key).

## 12. Housekeeping on the operator's machine  [ops]

Not checkable in the cloud container (no `backtest_cache/`); nothing in the code does these.

- [ ] Delete orphaned assembled caches tagged `monday-eligibility-v1` (~8.7 GB, per CLAUDE.md) and
  `monday-checkpoint-v2` (~1.35 GB) once no checkout still uses those tags.
- [ ] Delete the strike-blind `live_days/` slices for 2026-07-25..08-02, 08-29 and 08-30 (they're
  reused unconditionally until the cutoff passes them), then re-run the affected windows with
  `--no-cache`.
- [ ] Delete any lingering legacy `event_titles.json` (warned on every run).
- [ ] Refetch the 115 DR-73 calibration-corpus legs missing candles through the paged
  `fetch_candlesticks`.

---

## Accepted residuals (recorded; revisit only if the stated trigger appears)

Kept here so they aren't mistaken for forgotten work. Source: code comments and CLAUDE.md.

- YES-leg retried re-read can hold the unwind ~62 s (`trader.py:2660-2693`).
- Transfer "landed" inferred from a balance threshold; an unrelated credit could hide a stuck
  transfer (`trader.py:2142-2151`).
- Unverified NO leg can wait up to 7 s unhedged when the ledger lags (`trader.py:2251-2258`); after
  the serial budget a late disproof only stops pairs that start after it.
- Reporter lock-timeout fallback file must be merged by hand (`reporter.py:466-469`).
- Bisection can under-size across reachability holes; exact-fee rounding can leave a boundary spec
  slightly EV-negative at small n (`strategy.py:299-301, 341-375`).
- Live picks one pair per group by widest spread before Kelly; no runner-up after
  `select_portfolio` refuses a ladder-mate (`scanner.py:3929-3930`, `strategy.py:605-624`).
- Scheduler: a fault at 09:00 costs the Monday slot (DR-59); Treasury download can take ~4 min
  before fallback on a black-holed host.
- Prefilter relies on observed (not contracted) candle-timing behaviour (`backtester.py:1975-2008`).
- Empty-candle-cache future-mtime asymmetry vs DR-13 (`historical.py:4249-4256`).
- The sequential archive fallback still stops after `ARCHIVE_MAX_BARREN_PAGES` empty pages (it can
  miss long-lived markets; the sharded path reads every created-day instead); sequential fallbacks
  unbounded in memory; identity check is a hash (`historical.py` `_fetch_archive_sequential`,
  `_assembled_records`).
- `_ex_top_event` / `CapSweep.entry_events` deliberately wider than the trades; `max_trades_simulated`
  counts eager points only; `CapSweep` has no memo; O(B²) ladder sub-pass has only a per-bucket canary.
- Dashboard: best/worst-5 tables overlap below 11 trades (WONTFIX 2026-09-13); benchmark can carry
  one extra bar; `_deployed_on_days` pre-curve entry (unreachable).
- `scheduler.py` "arbitrage" wording left deliberately.

## Already fixed (checked 2026-09-29 — don't re-open)

BS-01…BS-32 from `BUG_SWEEP_FINDINGS.md`, including specifically re-checked: BS-02 archive barren
stop, BS-06 backtest shrink loop, BS-07 close-time parsing, BS-08 atomic caches, BS-09 event-title
accumulator, BS-12/28/29 orderbook/cursor guards, BS-13 scanner progress logs, BS-16/17/31 scheduler,
BS-18 trade-log locking, BS-19 dev key lookup, BS-20 HTML escaping, BS-21 worker-count comments,
BS-22/23/27 historical, BS-24 ticker release, BS-25 log rotation, BS-26 duplicate log lines, BS-30
empty drawdown; plus orphaned pre-cutoff `live_days/` pruning, CLAUDE.md test list, and README/CLAUDE
CLI flag coverage.
