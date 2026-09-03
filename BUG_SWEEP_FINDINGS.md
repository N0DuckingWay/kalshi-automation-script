# Bug Sweep Findings — New Endpoint Code (PRs #23–#33)

Sweep date: 2026-08-31. **Bug identification only — fixes come in a separate plan.**

Method: 6-dimension static review (Sonnet agents) with an adversarial skeptic verification pass,
plus live small-batch test runs: dev sandbox (full run), prod dry-run (full live pipeline, no
orders), fresh 2-day backtest (full fetch, 60 min), cached 2026-05-28 backtest (60-min box), and
harness micro-runs of scheduler / dashboard / reporter. Baseline: 226/226 tests pass, ruff clean.
Verdicts: CONFIRMED = reproduced or proven by direct code/SDK trace; PLAUSIBLE = mechanism certain
but trigger unobserved.

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| High     | 11    | BS-01 … BS-11 |
| Medium   | 12    | BS-12 … BS-23 |
| Low      | 9     | BS-24 … BS-32 |

**Top 5 by stakes:**
1. **BS-01** — ambiguous-order handling tests the account's *absolute* position, not the order's
   delta: a coincident external position can mark an unhedged trade "executed" or liquidate an
   unrelated holding (live prod money path).
2. **BS-02** — archive tail early-stop checks settlement time on a created-time-ordered walk;
   long-lived in-window settlements are silently dropped, and the test oracle shares the bug, so
   parity tests can't catch it (backtest data completeness).
3. **BS-03** — the orderbook legacy key mapping contradicts the pinned SDK's own model; if the API
   ever serves the SDK-documented shape, every book fails closed → silent zero-trade runs (latent;
   live API currently serves `orderbook_fp`, confirmed by the prod dry-run).
4. **BS-06** — the backtester skips the Kelly-budget shrink loop, systematically simulating larger
   positions than the live bot would take (strategy-validation integrity).
5. **BS-08/BS-09** — cache-layer fragility: non-atomic JSON writes that crash the next run when
   truncated, and `--no-cache` silently destroying the cross-run event-title cache.

Refuted during verification (not bugs): `dashboard._log_loss` log(0) (probabilities are clipped);
`reporter._result_to_row` round(None) crash (all 6 statuses serialize cleanly — zero reporter
defects found); prior-run orphans as a BS-01 vector (the held-tickers filter excludes them); dev
mode as a BS-01 surface (dev never submits); `pre_execution_check`'s uncached re-fetch (intentional
race protection — observed correctly dropping a stale candidate live).

---

## High findings

### BS-01 — Ambiguous-order check tests absolute position, not order delta
- Severity: high (downgraded from critical by adversarial verification) — Verdict: CONFIRMED-NARROWED
- Location: [trader.py:144-182](kalshi_betting/trader.py:144) (`_position_count`),
  [trader.py:403-433](kalshi_betting/trader.py:403) (leg B), [trader.py:374-390](kalshi_betting/trader.py:374)
  (leg A); snapshot timing [main.py:324](kalshi_betting/main.py:324)
- Symptom: `_position_count` returns the account's total signed position, not this order's fill
  delta. (a) Leg-B ambiguous + any non-zero count → `status="executed"`, no rollback — a genuinely
  unfilled leg B leaves leg A unhedged while the log reports success. (b) Leg-A ambiguous +
  non-zero/None → reduce_only FoK NO-sell of `spec.x` off whatever position exists.
- Verified preconditions: live prod only (dev never calls `_execute_one`; `--dry-run`
  short-circuits); requires an out-of-band trade (manual UI, concurrent invocation) on that exact
  ticker inside the unguarded snapshot→execution window (nothing re-snapshots held_tickers during
  the multi-minute scan→submit chain; `pre_execution_check` only re-validates price). Prior-run
  orphans are refuted as a vector. Leg-A liquidation damage additionally needs a same-side NO
  collision with quantity ≥ spec.x; opposite-side collisions fail safe as a false rollback_failed.
- Found by: static-review:money; narrowed by skeptic pass.

### BS-02 — Archive tail/sequential early-stop checks settlement on a created-ordered walk
- Severity: high (backtest data loss; typical-case, not edge-case) — Verdict: CONFIRMED (skeptic pass)
- Location: [historical.py:1125-1134](kalshi_betting/historical.py:1125) (`_fetch_archive_tail`),
  [historical.py:1195-1201](kalshi_betting/historical.py:1195) (`_fetch_archive_sequential`)
- Symptom: both walks stop the moment `page[0].settlement_ts < start_ts`, but `page[0]` is merely
  the newest-CREATED record on the page — deeper pages can hold long-lived markets settling inside
  the window. Skeptic verified: no whole-page/N-consecutive/overshoot mitigation; no other phase
  covers created<start ∧ settled-in-window (exclusively the tail's job); the sequential fallback
  ("always correct" per docstring) has identical logic; the test oracle `_old_semantics_expected`
  implements the same flawed rule so parity tests structurally cannot detect it (the LONGLIVED
  fixture is arranged too favorably). Since most Kalshi markets are short-lived, a shallow page's
  page[0] has typically already settled pre-window — terminating the tail after ~1–2 pages.
- Found by: static-review:fetch; strengthened by skeptic pass.

### BS-03 — `_ORDERBOOK_SIDE_KEYS` legacy mapping contradicts the pinned SDK's own model
- Severity: high (latent fail-closed; fails toward not-trading, not bad fills) — Verdict: CONFIRMED
  vs SDK source; trigger unobserved live
- Location: [scanner.py:851-854](kalshi_betting/scanner.py:851); load-bearing TODO at
  [scanner.py:908](kalshi_betting/scanner.py:908)
- Symptom: the SDK's `Orderbook` model for container `"orderbook"` REQUIRES `yes_dollars`/`no_dollars`
  and aliases its legacy int arrays as `"true"`/`"false"` — never `"yes"`/`"no"` as the code assumes.
  `orderbook_fp` appears nowhere in the SDK (live-observed only). If the API ever sends the
  SDK-modeled shape, every orderbook logs "Potential orderbook key mismatch" and fails closed →
  silent zero-trade runs. The prod dry-run confirmed the live API still serves `orderbook_fp`
  (zero mismatch warnings), so this is latent — but the TODO's requested validation against captured
  live data is genuinely unresolved.
- Found by: static-review:endpoints (seed #1); independently re-derived by consistency reviewer.

### BS-04 — `trader._position_count` bypasses `api_call_with_retry`
- Severity: high — Verdict: CONFIRMED (doc-vs-code; sibling `get_held_tickers` wraps the identical
  endpoint at [scanner.py:312-315](kalshi_betting/scanner.py:312))
- Location: [trader.py:170](kalshi_betting/trader.py:170)
- Symptom: one transient 429/5xx/transport error during position confirmation returns None
  ("unknown") → leg A escalates to an unconditional rollback, leg B to manual_review, where a retry
  would have resolved it. This is a read-only call — CLAUDE.md's no-retry rule covers only order
  submission. Directly amplifies BS-01's None branches.
- Found by: static-review:money (seed #4); independently re-found by consistency reviewer.

### BS-05 — Rollback market FoK sell has no price floor
- Severity: high — Verdict: CONFIRMED mechanism (SDK `CreateOrderRequest` has no sell-side proceeds
  protection; `yes_price`/`no_price` apply only to limit orders)
- Location: [trader.py:238-248](kalshi_betting/trader.py:238)
- Symptom: the unwind fills at whatever the NO-side best bid is. Leg-B failures correlate with the
  paired book moving, so the rollback can realize losses well beyond the fee/slippage assumptions in
  `compute_trade`'s sizing.
- Found by: static-review:money.

### BS-06 — Backtester Pass 2 lacks the Kelly-budget shrink loop
- Severity: high (strategy-validation integrity) — Verdict: CONFIRMED (mechanically verified diff)
- Location: [backtester.py:951-966](kalshi_betting/backtester.py:951) vs
  [strategy.py:181-184](kalshi_betting/strategy.py:181)
- Symptom: `n = int(budget/(nA+pB))` is only rejected when cost+fees > cash (full balance), never
  shrunk to fit the Kelly-capped budget — the exact "systematic overshoot" CLAUDE.md forbids
  removing live. The backtest simulates larger positions than live `compute_trade` would take for
  identical inputs, on essentially every trade.
- Found by: static-review:backtester.

### BS-07 — Unguarded `fromisoformat(close_time)` can crash a whole backtest on one malformed record
- Severity: high — Verdict: CONFIRMED path (manifestation depends on archive data quality; the
  cached run's 1.85M records happened to parse clean)
- Location: [backtester.py:678](kalshi_betting/backtester.py:678) (main thread, pre-pool); same
  pattern at :505/:509
- Symptom: the same-title `_extract_pairs` branch (unlike the time-series sweep at :364-365) never
  filters malformed `close_time`, so a bad record reaches `_fetch_candles_parallel`'s unguarded
  parse on the main thread → ValueError kills the run. Breaks the file's own "can't parse = unknown,
  not error" convention (`_parse_iso_date`). Not covered by the "worker exceptions must propagate"
  rationale (this is pre-pool).
- Found by: static-review:backtester.

### BS-08 — `_save_json_cache` is non-atomic; a corrupted cache file crashes the next run
- Severity: high — Verdict: CONFIRMED (contrast with `_day_store_save`'s tmp+replace idiom)
- Location: [historical.py:279-306](kalshi_betting/historical.py:279); inline read at :1873, write
  at :1934
- Symptom: an interrupted write (OOM/SIGKILL — both documented past events for these multi-hour
  fetches) leaves truncated JSON; the unguarded `json.loads` on the next run raises an uncaught
  JSONDecodeError instead of treating it as a cache miss. Affects the assembled
  `settled_markets_*.json`, `event_titles.json`, and every per-ticker candlestick cache. (Seed #8
  confirmed and widened.)
- Found by: static-review:fetch.

### BS-09 — `--no-cache` silently wipes the cross-run `event_titles.json` accumulator
- Severity: high — Verdict: CONFIRMED (mechanically verified)
- Location: [historical.py:355](kalshi_betting/historical.py:355), :511; passed through at :1797
- Symptom: with `use_cache=False`, resolution starts from `{}` and the save unconditionally
  overwrites the file with only this run's tickers — destroying potentially hours of accumulated
  resolution work (289k-scale fallback lookups documented) that the next default run silently
  re-pays.
- Found by: static-review:fetch.

### BS-10 — `auth.verify_auth` is the last modeled SDK call — drift single point of failure
- Severity: high (latent) — Verdict: PLAUSIBLE (pattern certain; parsed clean in the live prod
  dry-run on 2026-08-31)
- Location: [auth.py:113-119](kalshi_betting/auth.py:113)
- Symptom: `client.get_balance()` deserializes via pydantic `GetBalanceResponse` requiring StrictInt
  `balance`/`portfolio_value`/`updated_ts` — the same legacy-int field class that already drifted
  for markets, positions, and orders. Gates every prod run pre-trade (hard crash, nothing logged);
  the post-trade read degrades gracefully via a broad except.
- Found by: static-review:untested (seed #11).

### BS-11 — Recent/narrow backtest windows burn ~an hour fetching markets the prefilter discards
- Severity: high (cost/usability; correctness intact) — Verdict: CONFIRMED (live run: 9,228,720
  records fetched over ~59 min, ALL dropped by the Monday-eligibility prefilter → 2-byte assembled
  cache, 0 markets analyzed)
- Location: `fetch_all_settled_markets` live sweep + `backtester._can_ever_enter`; related missing
  diagnostic: no `start_ts >= cutoff_ts` warning at [historical.py:1729](kalshi_betting/historical.py:1729)
  (seed #2, confirmed: the 0-trade symptom itself arises downstream via candlestick 404s)
- Symptom: whether any Monday checkpoint fits in `[start_date, today]` is knowable before a single
  API call; instead the full MVE-inclusive fetch runs. A cheap pre-fetch feasibility check (plus the
  cutoff warning) would prevent both wasted-hour classes.
- Found by: test-run:backtest-fresh.

---

## Medium findings

### BS-12 — Legacy-cents orderbook levels silently parse to an empty book
- Severity: medium-high (latent; compounds BS-03) — Verdict: CONFIRMED arithmetic
- Location: [scanner.py:818-840](kalshi_betting/scanner.py:818) (esp. :834)
- Symptom: an integer-cent bid (45) computes ask 1.0−45.0 = −44.0 → dropped by the [0.01,0.99]
  guard; every level of a cents book vanishes into an "empty book" with no warning naming the cause
  — even on the legacy generation's own happy path, no cents→dollars conversion exists anywhere.
  Indistinguishable from genuine no-liquidity. (Seeds #3/#7; independently derived twice.)

### BS-13 — Silent multi-minute fetch phase in `fetch_open_events_with_markets`
- Severity: medium (observability; CLAUDE.md's documented misdiagnosis class) — Verdict: CONFIRMED
  (dev run: 13m27s with zero log lines while fetching 125,538 sandbox markets)
- Location: [scanner.py:448-479](kalshi_betting/scanner.py:448) and :491-526 — no per-page progress
  logging; single end-of-fetch summary at :528
- Symptom: kalshi_arb.log (the sole output channel) cannot distinguish "paginating 125k markets"
  from "hung" — the exact failure class the sharded fetcher's labels exist to prevent.

### BS-14 — Low-balance skip exits 0; scheduler logs it as success
- Severity: medium — Verdict: CONFIRMED (code trace, prod-run report)
- Location: [main.py:314-321](kalshi_betting/main.py:314) (bare return), :466-469 (no exit-code
  propagation); [scheduler.py:70-73](kalshi_betting/scheduler.py:70) (returncode-only check)
- Symptom: a weekly run skipped for insufficient funds logs "Job completed successfully." — the only
  distinguishing signal is a WARNING inside a log the scheduler never reads.

### BS-15 — Per-day-window fetch RSS peaks ~7.9GB on a 16GB host
- Severity: medium (escalating OOM risk) — Verdict: CONFIRMED (live telemetry: sawtooth
  3.4→4.3→7.89GB before each `_day_store_save`; ~62MB free at peak)
- Location: `historical._fetch_live_window` — accumulates a full UTC day in one list, not chunked
  below day granularity; 2026-08 MVE cardinality is ~4.4–4.8M markets/day, materially above what
  CLAUDE.md's 2026-07/08 narrative implies.

### BS-16 — Scheduler `TimeoutExpired` handler logs bytes-repr and drops stderr
- Severity: medium — Verdict: CONFIRMED (reproduced: `TimeoutExpired.stdout` is bytes despite
  text=True)
- Location: [scheduler.py:62-67](kalshi_betting/scheduler.py:62)
- Symptom: on timeout the log shows `b'...'` for stdout and never logs `exc.stderr` — the hang's
  traceback is silently dropped, contradicting the module docstring's traceability claim.

### BS-17 — Scheduler has no missed-run detection across daemon restarts
- Severity: medium — Verdict: PLAUSIBLE (schedule-library semantics)
- Location: [scheduler.py:94](kalshi_betting/scheduler.py:94)
- Symptom: a restart at/after Monday 09:00 silently schedules for next week — one missed weekly run
  with no log line distinguishing "skipped" from "not due".

### BS-18 — `append_to_prod_log` has no file locking
- Severity: medium — Verdict: PLAUSIBLE (needs concurrent invocation, e.g. manual + scheduled overlap)
- Location: [reporter.py:215-252](kalshi_betting/reporter.py:215)
- Symptom: load→append→save with no lock/atomic rename; the second saver silently clobbers the
  first run's real trade rows.

### BS-19 — `build_client("dev")` crashes when secrets.json has only `dev_api_key`
- Severity: medium — Verdict: CONFIRMED (repro: dict `.get(k, d[other])` evaluates the default
  eagerly)
- Location: [auth.py:77](kalshi_betting/auth.py:77)
- Symptom: `secrets.get("dev_api_key", secrets["Kalshi-api-key"])` raises KeyError even when
  dev_api_key exists — sandbox-only setups are silently coupled to having a prod key, contradicting
  the documented "optional, falls back" behavior.

### BS-20 — Dashboard interpolates raw market titles into HTML with no escaping
- Severity: medium — Verdict: CONFIRMED (no escape call exists in the file)
- Location: [dashboard.py:523-543](kalshi_betting/dashboard.py:523) (`_trow`); titles from
  [backtester.py:884](kalshi_betting/backtester.py:884)
- Symptom: Kalshi-controlled title/subtitle strings containing markup break the report tables;
  script-bearing titles would execute when the self-contained dashboard is opened in a browser.

### BS-21 — Worker-count comments claim live-verified "12 workers"; both constants are 8
- Severity: medium (uncertainty about what was actually tested) — Verdict: CONFIRMED
- Location: [config.py:140-144](kalshi_betting/config.py:140), :150-156

### BS-22 — `historical.py` hardcodes `limit: 200` instead of `config.MARKET_PAGE_SIZE`
- Severity: medium — Verdict: CONFIRMED (literal at :377/:423; constant not imported)
- Symptom: violates the constants-in-config rule; a page-size change silently desyncs the
  event-title listings.

### BS-23 — `fetch_candlesticks` drops malformed candles with zero logging
- Severity: medium — Verdict: CONFIRMED (bare `pass`, no counter/log at
  [historical.py:1923-1924](kalshi_betting/historical.py:1923))
- Symptom: silent thinning of a candle series quietly shifts backtest entries — the module's
  previously-diagnosed silent-data-loss class. (Seed #14.)

---

## Low findings

### BS-24 — Backtester ticker-conflict filter is permanent, unlike live's open-positions-only rule
- [backtester.py:1012-1013](kalshi_betting/backtester.py:1012); conservative-direction parity gap;
  docstring claims it "mirrors" the live rule. (CONFIRMED)

### BS-25 — `kalshi_backtest.log` has no rotation (398 MB after this sweep's runs)
- [backtest.py:73-80](kalshi_betting/backtest.py:73), plain FileHandler. (CONFIRMED)

### BS-26 — Duplicate completion log lines ("Dev simulation written", "Dashboard written")
- [reporter.py:364](kalshi_betting/reporter.py:364)+[main.py:290](kalshi_betting/main.py:290);
  [dashboard.py:821](kalshi_betting/dashboard.py:821)+[backtest.py:122](kalshi_betting/backtest.py:122).
  Both reproduced live; misleads log-grep metrics. (CONFIRMED)

### BS-27 — Progress line reports "0.0 slices/min" alongside a finite ETA
- `historical._log_slice_progress`; observed live. CLAUDE.md calls these lines load-bearing. (CONFIRMED)

### BS-28 — Null orderbook container classified as "key mismatch" rather than empty book
- [scanner.py:906-925](kalshi_betting/scanner.py:906); gap certain, live behavior unverified.
  (PLAUSIBLE; seed #6)

### BS-29 — Pagination loops unprotected against a non-advancing cursor
- [scanner.py:307-334](kalshi_betting/scanner.py:307), :447-479, :484-526; requires an API-side
  pagination bug; inconsistent with the codebase's bounded-scan idiom. (PLAUSIBLE)

### BS-30 — `_max_drawdown` crashes on an empty equity series (defensive only)
- [dashboard.py:96-101](kalshi_betting/dashboard.py:96); harness-reproduced ValueError, but
  `_build_equity_curve` always emits ≥1 row and the live 0-trade run rendered clean — unreachable
  today from `run_backtest`. (CONFIRMED, defensive hardening)

### BS-31 — Scheduler `run_job` doesn't catch OSError from subprocess.run
- [scheduler.py:53-74](kalshi_betting/scheduler.py:53); harness-reproduced; degrades to main()'s
  generic "Scheduler tick raised" losing the specific cause. Otherwise scheduler passed 11/11
  behavior checks. (CONFIRMED)

### BS-32 — Doc drift bundle
- CLAUDE.md test list omits `tests/test_http.py`; dependency diagrams omit
  historical→auth and backtester→scanner edges; README's prod dry-run section omits the
  trade_log.xlsx side effect (main.py docstring documents it — behavior confirmed live: this
  sweep's dry-run created the file with 5 simulated rows); `--mode dev --dry-run` presented as if
  the flag suppresses orders when it is inert (help says "prod only", `_run_dev` never reads it —
  seed #12); `TradeSpec.profit_ratio` docstring omits the fee netting the code performs
  ([strategy.py:47](kalshi_betting/strategy.py:47) vs :139-144); `_buy_max_cost_cents` lacks the
  round-before-ceil float-noise guard `fee_leg_exact` documents as load-bearing
  ([trader.py:61-77](kalshi_betting/trader.py:61)); trader.py hardcodes thread-pool size 8
  (:301/:495) against the named-constant pattern; the "always correct" docstring on
  `_fetch_archive_sequential` is falsified by BS-02; orphaned pre-cutoff `live_days/` slice files
  are never pruned after a cutoff advance (no correctness impact — verified no dup/loss). (All CONFIRMED)

---

## Coverage ledger

- Static dimensions (6/6 reported): endpoints, money paths, sharded fetch, backtester, untested
  modules, cross-cutting consistency. Skeptic verification ran on the critical/subtle claims
  (absolute-position, archive tail) and a 10-point mechanical line-check (9 verified exactly,
  1 verified-with-nuance).
- Test-run targets (7/7 ran): dev sandbox (full clean run, 9 simulated trades, xlsx valid); prod
  dry-run (full live pipeline clean, no orders POSTed); fresh 2-day backtest (clean 60-min run,
  0 trades via prefilter, [windowed] labels correct, no sequential fallback under the advanced
  cutoff); cached 2026-05-28 backtest (killed at 60-min cap 24% through candlesticks — cache load,
  prefilter, pair extraction over 1.85M markets, 404 fail-soft and 429 backoff all clean;
  entry/portfolio/dashboard-on-real-trades not reached, covered by static parity review + offline
  tests); scheduler harness (11 checks + 2 findings); dashboard stats harness (41 tests, math
  hand-verified); reporter harness (zero defects).
- All 14 planning seeds adjudicated: confirmed (1,2,3,4,6,7,8,11,12,14 → BS-03/11/12/04/28/12/08/
  10/32/23), folded-and-narrowed (5 → BS-01), refuted (9, 10, 13).
- Environment notes: live archive cutoff advanced 2026-06-04 → 2026-07-02 during this period
  (all 228 cached archive day slices are now stale and will refetch on the next pre-cutoff-window
  run); prod balance is $256.77 (the old $1.14 log line came from unmerged branch
  feat/multi-shard-support); `timeout`/`gtimeout` are not installed on this host (drivers enforced
  caps manually); this sweep's own runs appended to kalshi_arb.log / kalshi_backtest.log, created
  trade_log.xlsx (simulated rows only), wrote ~600MB of new live_days/candlestick cache, and
  produced dev_simulation_2026-08-31_162902.xlsx; two fixture xlsx files leaked into the repo root
  by a sweep harness were verified as synthetic and moved to the session scratchpad. Pre-run
  backups of both logs are in the scratchpad.
