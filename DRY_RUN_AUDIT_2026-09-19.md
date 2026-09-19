# Dry-Run Audit — 2026-09-19

A dated, frozen record of a full dry-run sweep of every module, in the style of
`BUG_SWEEP_FINDINGS.md`. **Nothing in this document has been fixed.** No trades were
executed and no source file was modified: the `kalshi_betting/` + `tests/` tree hashed
identically (`6b9fff06eb8ccffb7c3b2f1dfa7f2c35`) before and after the sweep.

Baseline at audit time: **1385 tests pass, 1 skipped**; `ruff check kalshi_betting/` clean;
all 14 modules import.

---

## How this was produced

**Phase 1 — 27 parallel dry-run agents** (opus on every money path; sonnet on
scheduler / dashboard / reporter / concurrency). Sixteen covered one module area each;
eleven were cross-cutting (leg sides, money units, live-vs-backtest parity, retry policy
and API drift, exit codes and logging, constants discipline, docstring truth, CLAUDE.md
claims A and B, concurrency and resources, and an adversarial money-loss red team).
Each agent wrote and **ran** harnesses against the real modules with `MagicMock` clients —
314 harness scripts in total — rather than reading code.

**Phase 1b — mutation testing.** The parity agent additionally broke each live/backtest
mirror on a *copy* of the repo and ran the full 1385-test suite against each (29 mutations,
one full run each) to measure which invariants the suite actually protects. Results in the
appendix — **8 mutations survive**, and they are precisely where silent divergence can live.

**Phase 2 — 4 adversarial verifiers**, each tasked with *refuting* the 14 highest-impact
findings and writing its own repro from scratch rather than re-running the originals.
**This round overturned or downgraded 6 of the 14.** Where a verifier corrected a
reporter, the verifier's version is what appears below.

Confidence notation used throughout:
- **[Vn]** — independently reached by _n_ separate phase-1 agents.
- **[VERIFIED]** — survived a phase-2 adversarial refutation attempt.
- **[CORRECTED]** — survived, but the mechanism or severity was wrong as first reported.

---

## Summary

| Tier | Count | Theme |
|---|---|---|
| P1 | 4 | Real money / unhedged position / money-visibility |
| P2 | 12 | Correctness, run-killing crashes, availability |
| P2b | 4 | Live-vs-backtest parity (backtest fidelity) |
| P3 | 9 | Silent drops and observability gaps |
| P4 | 14 | Documentation defects with a concrete mis-edit hazard |
| — | 6 | Refuted or downgraded in phase 2 — **do not act on these** |

The money paths that matter most came back **clean**. Worth recording, because it bounds
where the risk is *not*:

- **Leg-side single source of truth**: 4,000 randomized pairs + a 122-assertion
  differential harness through the whole chain — **0 violations**. A time-series pair
  never submits or unwinds `market_a`.
- **TS-08 reachability**: 36,292 + 1,181 sized specs across all four tick regimes —
  **0** specs returned an `n` the V2 FoK cap could not reach.
- **Kelly / DR-62**: the fee is in the denominator in all three sizers; every published
  figure in CLAUDE.md reproduces to the digit; 61,305 live-vs-backtest parity points and
  200 whole-checkpoint portfolio comparisons — **0 divergences**.
- **`trader._execute_one` status matrix**: every branch driven on both pair types, both
  order paths, plus a 144-combination ground-truth fuzz — **0 unescalated unhedged
  states, 0 misdescribing statuses, 0 exceptions escaping**.
- **Constants discipline**: 0 inlined literals, 0 dead constants, and every shared-value
  collision group proven to read the right constant by runtime sentinel patching.
- **No modeled SDK call remains anywhere** (exhaustively confirmed).

---

## P1 — Money, unhedged positions, money-visibility

### DRA-01 — Collateral is funded at the SCANNED price while the order may pay the CAPPED price
**`trader.py:1387-1399` (`_required_cents_by_shard`)** · severity **medium-high** ·
money-loss · **[V2] [VERIFIED] [CORRECTED]**

`_required_cents_by_shard` funds `ceil(n·p_scanned + fee_leg_exact(n, p_scanned))`, but the
order carries price protection permitting a fill at `v2_effective_cap` = scanned +
`BUY_SLIPPAGE_TICKS` ticks (legacy: `buy_max_cost` caps contract cost and the taker fee is
charged **on top**). `_plan_transfers` moves exactly `required − available`, so a
transferred-to shard lands at precisely `required` — **measured headroom 0¢**. The per-leg
`math.ceil` contributes < 1¢, orders of magnitude short of the gap.

**Correction from verification:** the two reporting agents compared `n × cap` against
funded, which gives a *negative* gap (the fee inside `cost_with_fees` covers one tick of
contract cost). The correct comparison adds the taker fee charged *at the cap price*.
Re-measured: **1,898 of 1,900 legs (99.9%) under-funded on `linear_cent`**, 1,886/1,900
(99.3%) on `center_deci_edge_centi_cent`; worst shortfall **$21.36**.

This does **not** depend on the unverifiable "does the exchange reserve at the limit"
premise. If a leg fills anywhere above the scanned price — exactly what the cap exists to
permit when the book moves between `validate_pair_price` and submission — the cash
consumed exceeds the funded cents. On the **YES** leg that is a filled NO leg with no
hedge: loss-floored rollback at up to 12¢/contract, or `rollback_failed`.

Unrecorded in CLAUDE.md (its collateral paragraph justifies the ceiling only at sub-cent
granularity) and untested (`tests/test_trader.py:1085-1152` pins the exact-deficit planner;
nothing pins funding-vs-cap).

**Fix.** In `_required_cents_by_shard`, resolve each leg through `scanner.leg_sides` /
`leg_prices` and fund from the cap: `n × float(scanner.v2_effective_cap(leg_kind, price,
market)) + config.fee_leg_exact(n, cap)` on V2, `_buy_max_cost_cents(n, price)/100 + fee at
that price` on legacy. Keep the existing `math.ceil(round(·, 6))` and the
`cost_with_fees_a ↔ market_a` attribution. It only ever over-funds, which the existing
comment says costs nothing. **Add a test** pinning funded ≥ at-cap spend per leg.

### DRA-02 — `_await_transfer_settlement` discards confirmed arrivals when the FINAL read fails
**`trader.py:1682-1695`** · severity **medium-high** · correctness ·
**[VERIFIED] [CORRECTED — worse than reported]**

One `balances` local serves both the per-iteration funded check and the return value, and
is overwritten with `{}` in the `except`. If the **last** read raises, the function returns
`{}` even though an earlier read already observed money land.

The tempting refutation — "the loop only continues when still short, so nothing is lost" —
holds only for a single shard. With a multi-shard `awaitable`, a read where shard 1 landed
and shard 2 is still short is still `unfunded`, so the loop continues, the next read
raises, and **shard 1's confirmed arrival is destroyed**.

**Worse than first reported:** `_unfunded_shards(required, confirmed)` is called with the
**full** required map, so `confirmed == {}` marks *every* shard unfunded and
`_partition_by_funding` drops the **entire portfolio** — including trades on a rich shard
that was funded from the start and never appeared in the plan. DR-65's threshold partition
does not save it: `confirmed.get(s, 0) < awaitable.get(s, 0)` reads `0 < 5000` and fires the
critical anyway.

The docstring ("possibly empty if every read failed") is materially wrong — empty is
returned whenever the *last* read fails. `TestAwaitTransferSettlement` covers only
single-shard, and its failed-read test sets the timeout to 0 so exactly one read happens.

**Fix.** Merge instead of overwrite, and never treat a failed read as an observation:
```python
observed: dict[int, int] = {}
while True:
    try:
        observed.update(read_shard_balances(client))   # last-wins per shard, never reset
    except Exception as exc:
        logging.warning("Balance re-read failed while awaiting transfers: %s", exc)
    if not _unfunded_shards(required, observed): return observed
    if time.monotonic() >= deadline:             return observed
    time.sleep(TRANSFER_POLL_INTERVAL_SECONDS)
```
A later genuinely-lower reading still wins, so a real debit is not masked. Correct the
docstring to "possibly empty if no read ever succeeded". Separately, in
`ensure_shard_collateral`, fall back to the run's opening `shard_balances` for shards absent
from `confirmed`, so a total read outage cannot drop trades on already-funded shards.
**Add a multi-shard test** where the final read raises.

### DRA-03 — A `MONEY IS IN FLIGHT` critical cannot reach the process exit code
**`main.py:764-772`, `:846`; `trader.py:1874`, `:1592`, `:1608`** · severity **high** ·
observability of a money event · **[VERIFIED] [CORRECTED — framing]**

When `ensure_shard_collateral` POSTs an accepted, non-idempotent transfer that is never
observed to settle, it logs `logging.critical(... MONEY IS IN FLIGHT, CHECK THE ACCOUNT ...)`
and drops the specs. `_run_prod` returns `EXIT_OK` on **both** the all-dropped branch
(`main.py:772`) and the normal tail (`main.py:846`), so `scheduler.run_job` logs
`Job completed successfully.` and finalizes the Monday slot with `exit_code=0`.

All ten `_run_prod` return paths were traced. Exactly one returns non-OK, gated on
`n_orphaned or n_unknown` counted off `execute_trades`' results. The collateral outcome is
read nowhere. The scheduler branches solely on `returncode`.

**Worse:** `execute_trades` runs over the **kept** list, so a collateral-dropped spec
produces no `TradeResult` and therefore **no Excel row at all**. `kalshi_arb.log` — the file
CLAUDE.md says the daemon never reads — is the only trace anywhere.

**Framing correction:** this is *not* a code-vs-doc divergence. BS-14 defines
`EXIT_TRADES_NEED_ATTENTION` as "any `execute_trades()` result with status `rollback_failed`
or `manual_review`", and the code matches that exactly. It is the **same failure shape BS-14
was created to fix**, left uncovered — BS-14's own text says the low-balance short-circuit
"used to `return` and exit 0 just like a clean run, silently burying the reason nothing
happened." A `MONEY IS IN FLIGHT` critical is strictly more urgent and is the only alarm in
the pipeline with no exit-code representation.

**Fix.** Have `ensure_shard_collateral` return the in-flight verdict (e.g.
`(kept, in_flight_shards)`), and in `_run_prod` return `EXIT_TRADES_NEED_ATTENTION` when it
is non-empty — on the all-dropped branch **and** OR'd into the existing
`n_orphaned or n_unknown` test at the tail. Extend `_run_prod`'s docstring return contract
and `scheduler.run_job`'s message. Secondary: emit a rescue line per collateral-dropped spec
so it leaves a record outside `kalshi_arb.log`.

### DRA-04 — `_read_position` reports "confirmed flat" for a payload it cannot read
**`trader.py:1056-1064`** · severity **high (mechanism) / trigger unobserved** ·
correctness · **[VERIFIED] [CORRECTED — severity split]**

`data.get("market_positions") or []` yields `[]` identically for an **absent key**, a
**renamed key** and a **null**, then falls through to an unconditional `return 0.0` — which
every caller reads as *confirmed no position*. It is the only raw reader in the codebase
with no container-key validation, and its answer decides whether an unhedged leg is
unwound. Nothing upstream validates: `_check_and_parse` rejects only an *empty* or
*non-JSON* 2xx; a JSON object with the wrong keys passes through.

The DR-64 lag re-read cannot help — a drifted ledger reads `0.0` on *every* read, so the
delta is structurally 0 however many times it is re-read.

**Severity correction — the two arms are not equivalent:**
- **V2 (the default)** needs only the drift, and is **loud**: `_confirm_v2_no_mapping` sees
  delta 0 twice, fires `CRITICAL V2 NO-LEG MAPPING DISPROVEN` on a mapping that is in fact
  *correct*, and returns `manual_review` → `EXIT_TRADES_NEED_ATTENTION`. Bad, but
  fail-visible. Because the latch never sets, **every pair in the run repeats it** — 3 pairs
  measured as 3 NO legs submitted, 0 YES legs, 0 unwinds, each its own unhedged position.
- **Legacy** is the silent arm and needs a *conjunction* (drifted ledger **and** a transport
  error landing after a real fill): `status="failed"`, no unwind, not in the attention set,
  **run exits 0**.

`scanner.get_held_tickers` returns `set()` on the same payload, silently disabling the
duplicate-position filter.

**Honest caveat.** Container-key drift on `/portfolio/positions` has **not** been observed.
Field-level drift on that same endpoint (`position` → `position_fp`) has, as has container
drift on markets, orders, events and balance. Treat this as a missing fail-closed guard in a
documented drift class, not a bug firing today. `tests/test_http.py:304`'s own rationale
condemns it: "Returning `{}` here … `trader._read_position` would answer 'the account is
flat' when the truth is unknown." The invariant is guarded for an *empty* body and unguarded
for a structurally-present-but-unreadable object.

**Fix.** In `_read_position`, before the loop:
```python
if not isinstance(data, dict) or "market_positions" not in data:
    raise ValueError(f"positions payload for {ticker} has no 'market_positions' key: ...")
```
`_position_count` / `_position_count_once` already translate a raise into `None` = *state
unknown*, which routes to `manual_review` instead of a false confirmed non-fill, and routes
`_confirm_v2_no_mapping` to *proceed unlatched* instead of a false disproof. Add the same
presence check to `scanner.get_held_tickers` (loud there — an empty held set silently
re-trades held tickers). Separately, drop the unverified single-page assumption in
`_read_position`: follow `data.get("cursor")`, or assert every returned row matches the
requested ticker.

---

## P2 — Correctness, run-killing crashes, availability

### DRA-05 — A dict-shaped bid level raises an uncaught `KeyError` that kills the whole scan
**`scanner.py:2468` / `:2475`** · severity **medium-high** · crash · **[V2] [VERIFIED]**

`bid_price = float(entry[0])` sits inside `except (ValueError, TypeError, IndexError)` — no
`KeyError`. A level arriving as a JSON **object** (`{"price": "0.60", "quantity": "400"}`)
raises `KeyError: 0` out of `_bids_to_ask_levels` → `_leg_ask_levels` →
`enrich_with_orderbook_prices` → `main`, aborting the entire run after the multi-minute
ingest. A dict is the **unique** uncaught shape: `str`, `int`, `None` and short-list levels
all raise caught types.

Verified there is **no handler anywhere up the chain**: `_run_dev` has 0 `try:` blocks,
`main()` has 0, and `_run_prod`'s two are the post-trade balance read and the Excel write —
both *after* execution. `_fetch_orderbook`'s broad `except Exception` covers only the fetch
and container/side-key selection; level parsing is deferred outside that `try`, so the dict
passes straight through.

Two call sites of one parser behave oppositely: under `trader.pre_execution_check` the same
drift is caught per-spec and merely drops the spec. And the sibling parser
`_cents_bids_to_dollar_bids` **deliberately catches `KeyError`**. The docstring promises the
level is dropped with a summary WARNING.

Downstream: exit 1 → `scheduler.run_job`'s generic failure branch → the Monday slot is
consumed and `_maybe_catch_up` re-runs only `EXIT_NO_TRADEABLE_SHARDS`, so **nothing retries
for a week**.

**Fix.** `except (ValueError, TypeError, IndexError, KeyError):` at `scanner.py:2475`,
matching the sibling. The existing drop-counter WARNING then reports it. **Add a test**
supplying a mapping level.

### DRA-06 — `pair_key`'s `" | "` separator defeats the empty-title guard
**`scanner.py:707-710`, `:895-898`, `:2017-2020`** · severity **medium** ·
false-premise trade · **[VERIFIED]**

`time_series_group_key` promises `""` (the caller drops the market) when the title
normalizes away, but `pair_key` joins event title and market title with a literal `" | "`
that no `_DATE_PATTERNS` entry strips. A market whose event title **and** market title are
both pure date tokens normalizes to `"|"` — truthy — so the `if norm:` guard at
`scanner.py:2018` passes and every such market lands in one bucket.

Reproduced end-to-end through the real finder: two unrelated questions on different series
(`"Apr 02"` / `"Q1 2026"` at 0.20 and `"H0650"` / `"Q4 2027"` at 0.60) emit a
`tradeable=True` time-series pair with `canonical_title='|'`. The decisive control: the
*same two markets* with `_event_title=""` key to `""` and are correctly dropped — the
separator is the sole cause.

Every other guard passes: `_filter_active_markets` keeps them, the one-series rule passes
(`KXAAA` ≠ `KXBBB`), the gap is 10 ≤ 30 days, the tier passes (0.40 ≥ 0.15). An amplifier
was checked and ruled out — real tickers do **not** normalize away, so the
`market_title` → `.ticker` fallback is safe. Reachability requires a date-only market title
*and* a date-only event title; `_DATE_PATTERNS`' own comment records date-only titles
("Apr 02", "Mar 21", "H0650") as observed in Kalshi sandbox.

`TestTimeSeriesGroupKey::test_empty_title_yields_the_empty_key` pins only the literally-empty
input — it pins the *intent* this defeats.

**Fix.** Make the emptiness test structural rather than string-truthy: after
`base = normalize_title(combined_title)`, return `""` when `base` has no alphanumeric
content (`if not re.search(r"[0-9a-z]", base): return ""` — `base` is already lowercased).
`base == "|"` is a band-aid that breaks if a real title contains a pipe.

### DRA-07 — Settlement data decides which candidate wins its title group (look-ahead bias)
**`backtester.py:2036`, `:2046`, `:2090`, `:2135-2141`** · severity **high** ·
correctness · **[VERIFIED — upgraded]**

`_simulate_at_discount` drops candidates on **settlement-derived** grounds (the
outcome-validity check at `:2036` and the premise-violation check at `:2046` — **two**
filters, not one) inside the Pass-1 loop, and the one-pair-per-group dedup
(`best_by_group`) runs **afterwards** at `:2135`. Removing a candidate for a reason knowable
only at settlement therefore **promotes that group's runner-up**.

Verified with the live counterpart: the real `scanner.find_time_series_pairs` over the same
four markets emits **one** pair — C1 (gap 0.20), ranked by `(tradeable, pB−pA)` — and never
proposes C2 (gap 0.18). So the promoted pair is definitively one the live bot would not have
offered. With entry-time data identical across runs and only C1's settlement result changed:

```
RUN A (C1 settles validly):           trades [('C1-A','C1-B', +765.62)]  final $10,765.62
RUN B (C1 settles premise-violating): trades [('C2-A','C2-B', +490.27)]  final $10,490.27
```

That phantom trade flows into total return, the equity curve, drawdown, Sharpe/Sortino and
the per-`k` sweep table. The module guards against precisely this class twenty lines above
(`backtester.py:2073-2075`: "Sorting Pass 2 by REALIZED returns would leak settlement
outcomes into trade selection (look-ahead bias)") — the *ranking* is look-ahead free, the
candidate **set** is not. CLAUDE.md's load-bearing-statement-order note protects the Kelly
gate's position only and is silent on the settlement filters. Not pinned:
`tests/test_backtester.py:3227` uses a single candidate; `grep best_by_group tests/` → no hits.

**Fix.** Split Pass 1: (1a) Kelly gate + `best_by_group` over entry-data-qualified
candidates, (1b) settlement filters applied to the surviving group winners. A group whose
winner is a premise violation then contributes **nothing** — still "never traded, never
paid" — instead of promoting a pair live would not have offered. This changes the
`premise_violations` count, so
`TestRunBacktestTimeSeriesFlow::test_premise_violation_is_excluded_and_warned` and
`::test_premise_count_is_larger_than_the_k_dependent_warning` need re-pinning. **Every
pre-existing backtest baseline becomes incomparable.**

### DRA-08 — Four `historical.py` cursor loops are unbounded in pages, memory and disk
**`historical.py:1596` (`_fetch_archive_day`), `:1798` (`_fetch_archive_sequential`),
`:2122` (`_fetch_live_window`), `:2170` (`_fetch_live_sequential`)** · severity **medium** ·
crash (hang / OOM / disk fill) · **[V2] [VERIFIED]**

TS-05 gave `scanner.py`'s cursor loops a `seen_cursors` set **and** an unconditional
`SCANNER_MAX_PAGES` ceiling, and `_fetch_archive_tail` carries `ARCHIVE_TAIL_MAX_PAGES` /
`ARCHIVE_TAIL_MAX_RECORDS`. None of these four got either. Driven with a cycling
`A,B,A,B…` cursor (which never repeats *consecutively*, so a `new_cursor == cursor` guard
misses it), all four ran to a 3,000-page hard stop still going.

`_fetch_live_sequential` is worst: its **only** exit is `if not cursor: return kept` — it
lacks even the `or not page` disjunct its own sibling `_fetch_live_window` has, so an empty
page beside a non-null cursor loops forever, logging "0 markets kept so far" every 100 pages.

**Memory and disk.** `_fetch_archive_sequential` accumulates into `selected` with no sink
and no record cap; on a healthy archive the barren counter resets on any in-window
settlement and bounds nothing. CLAUDE.md is wrong twice in one paragraph: "The tail walk is
the **ONE** fetch path with no chunked `emit` sink" (false — `_fetch_archive_sequential` and
`_fetch_live_sequential` both are) and "the sequential fallbacks … are the remaining
**bounded** in-memory cases" (false). Same OOM shape BS-15/TS-15 fixed for the sharded
workers — and reached via the path taken *when the archive cursor protobuf drifts*, i.e.
exactly when someone re-runs a backtest after an API change. Additionally,
`_fetch_archive_day`'s `emit` sink caps RAM but **not disk**: a looping worker streams
chunks into `<day>.json.gz.tmp` until the filesystem fills.

No outer bound exists — `backtest.py` has no deadline, and `SCHEDULER_JOB_TIMEOUT_SECONDS`
applies to `main.py`, not the backtest CLI. Not a TS-05 re-report: TS-05's own scope
sentence is "the last unbounded scans in the **ingest** path".

**Fix.** Carry the TS-05 idiom to all four: a `seen_cursors: set[str]`, a break with a
WARNING containing `cursor did not advance`, and an unconditional page ceiling (reuse
`SCANNER_MAX_PAGES` or add `HISTORICAL_MAX_PAGES`). Add `or not page` to
`_fetch_live_sequential`. Give both sequential fallbacks a record ceiling mirroring
`ARCHIVE_TAIL_MAX_RECORDS`. Correct the two CLAUDE.md sentences.

### DRA-09 — The one-series rule is bypassed when the deadline lives in the EVENT TITLE
**`scanner.py:2048`, `:767`** · severity **medium** · false-premise trade

Two events of ONE series with identical market title and subtitle, whose **event titles**
differ only by a date token, are refused by `find_same_title_pairs` but **paired by**
`find_time_series_pairs`. `_identical_wording` compares the RAW event title (different →
`False`, so the DR-02 conjunct does not fire) while the time-series group key normalizes that
same date away and puts them in one group. `main._dedup_pairs` cannot help — there is no
same-title copy to prefer.

This is the exact DR-02 shape (two different ball games priced on a cumulative-deadline
premise that does not exist), reached through the one string a grouping key *does* read.
CLAUDE.md's DR-02 text asserts "The date lives only in the event ticker, the event
sub-title, the close time and the rules text — **none of which any grouping key reads**."

**Fix.** In `find_time_series_pairs` only, test wording identity on the MARKET wording:
skip when `_same_series(mA, mB)` and `(mA.title, mA.subtitle) == (mB.title, mB.subtitle)`.
This preserves every case CLAUDE.md lists under "what still pairs" (`KXSOLD-26SEP14/18` and
the FIDE Oct/Nov markets differ in the market title) and extends the already-accepted cost
from "deadline only in the event ticker" to "…or only in the event title". Leave
`find_same_title_pairs` and `_identical_wording`'s contract unchanged.

### DRA-10 — A NaN leg price passes the `(0,1)` gate and crashes the sizer
**`strategy.py:375` → `config.py:973`** · severity **medium** · crash

The gate is written in the NEGATED form `price <= 0.0 or price >= 1.0`, which **NaN
evaluates `False` for**, so a NaN leg price reaches `max_affordable_pairs` and `int(nan)`
raises an uncaught `ValueError` out of `compute_trade` → `main._compute_trade_specs`,
killing a prod run after auth, ingest and pairing. `inf` is correctly caught
(`inf >= 1.0`); only NaN slips.

The sibling guard `scanner._filter_active_markets` uses the INCLUSIVE form
(`_MIN <= ya <= _MAX`) and correctly rejects NaN — but it range-checks **only**
`yes_ask_dollars`. **No scanner filter range-checks `no_ask_dollars`**, which is a traded
price for both pair types. The stdlib `json` parser in use (orjson is not installed) accepts
bare `NaN`.

**Fix.** `if not (0.0 < price_a < 1.0 and 0.0 < price_b < 1.0): return None` — behaviour-
identical for every finite input, rejects NaN. Also range-check `no_ask_dollars` in
`_filter_active_markets`.

### DRA-11 — A naive/aware `close_time` mix raises `TypeError` out of `find_time_series_pairs`
**`scanner.py:2030`, `:621`, `:934`** · severity **medium** · crash

`_filter_active_markets` drops only a `None` `close_time`, so a tz-naive datetime (from a
payload whose `close_time` lost its trailing `Z`; `datetime.fromisoformat` accepts it)
reaches the finder. `sorted(members, key=close_time)` then raises
`TypeError: can't compare offset-naive and offset-aware datetimes` out of `main`, killing
the run → exit 1 → the Monday slot is consumed.

The module already treats this as an expected input elsewhere: `filter_markets_within_horizon`
explicitly documents dropping naive datetimes — but it is opt-in (`--max-horizon-days`) and
off by default. `backtester._extract_pairs` keeps a DATE sort precisely because "a datetime
sort would raise on a naive/aware mix", and `_find_entry` wraps its subtraction in
`try/except TypeError`. `find_same_title_pairs` never sorts by close time, so the two finders
behave differently on one list.

**Fix.** Drop tz-naive close times in `_filter_active_markets` beside the `None` case
(`m.close_time is None or m.close_time.tzinfo is None or m.close_time.utcoffset() is None`)
and fold them into the same summary WARNING, renamed "missing/unparseable/naive close_time".

### DRA-12 — A non-UTF-8 error body turns a retryable 429/5xx into a fatal `UnicodeDecodeError`
**`_http.py:219-224`** · severity **medium** · correctness · **[V2]**

`_check_and_parse` decodes the error body as *strict* UTF-8 while constructing the
`ApiException`. The decode is an argument expression evaluated *before* `from_response`
runs, so on an undecodable body the `ApiException` is never built, the status never
surfaces, `_extract_status` → `None`, `_is_transient_network_error` → `False`, and
`api_call_with_retry` re-raises on attempt 1. The whole backoff policy silently does not
apply to that response.

Measured: a 502 with a UTF-8 body retries 6 times with sleeps `[2,4,8,16,32]`; the same 502
with a Latin-1 body dies after 1 call. The SDK's own `ApiException.__init__` already does
this decode defensively inside `try/except`; `_check_and_parse` added the unguarded copy.

**Fix.** `body.decode("utf-8", errors="replace")`. The value is only ever the exception's
human-readable body.

### DRA-13 — `api_call_with_retry` retries 5 hand-picked statuses while its docstring says "5xx"
**`_http.py:85` vs `:365`** · severity **medium** · spec-divergence

`_RETRYABLE_STATUS = {429, 500, 502, 503, 504}`, so 501, 507, 508, 408 and the whole
Cloudflare 520–530 family are fatal on first occurrence. The pinned SDK collapses the range
into one class (`if 500 <= status <= 599: raise ServiceException`), so a 520 arrives as the
*same exception type* as a 503 — only this literal separates them. Kalshi is CDN-fronted, so
one `524` in a multi-hour `fetch_all_settled_markets` walk kills the run, where a 503 in the
same position costs 62s and survives.

**Fix.** `retryable = (status is not None and (status == 429 or 500 <= status <= 599)) or transient`.
Safe at every call site because every caller of this wrapper is a read-only GET. Nothing pins
`_RETRYABLE_STATUS` today — pin it when fixing. Whether to add 408 is a separate call; state
it either way.

### DRA-14 — `_check_and_parse` reads the body BEFORE checking the status
**`_http.py:216-225`** · severity **medium** · money-visibility

The body read happens before the `200 <= status < 300` test, so a transport failure during
the read discards the one fact distinguishing "the request never landed" from "the exchange
answered 2xx and the money moved". The SDK's `RESTResponse` sets `self.data = None` and
fills it only in `read()`, so the read branch is the *normal* path; `resp.status` is already
known from the status line at that point, i.e. the information exists and is thrown away.

`trader._execute_transfer`'s TS-17/DR-05 guards catch `JSONDecodeError` / `TypeError` /
non-dict, but not a `ProtocolError` from the read — so a $200 transfer the exchange
**accepted** is logged as a FAILED POST, the shard is left out of `accepted_cents`, no
settlement poll runs, and no in-flight critical fires. On read-only GETs the same ordering is
harmless (the `ProtocolError` is correctly classified transient and retried), so this is
specific to the money path.

CLAUDE.md's claim "`_check_and_parse` validates the status before parsing, so a
`JSONDecodeError` out of that call proves a 2xx" is true of the *parse* but not of the
*read*, which sits upstream.

**Fix.** Compute `ok = 200 <= resp.status < 300` first; wrap the read in `try/except` and,
when `ok`, raise a narrow `_AcceptedBodyUnreadable(resp.status) from exc`; otherwise
re-raise. Add that type to `_execute_transfer`'s in-flight `except` tuple. Read-only GET
callers keep today's behaviour as long as the new type chains via `from exc`.

### DRA-15 — A `/exchange/status` failure disables the collateral-transfer gate entirely
**`trader.py:1501-1503` ← `scanner.py:1522-1527`** · severity **medium** · money-loss

`fetch_shard_statuses` returns `None` on *any* failure — a renamed key, a 5xx outliving the
retry budget, a non-dict body — and `_transfers_active(None, shard)` returns `True` for
**every** shard. On a live sharded account that turns off the "never move money to or from a
shard whose transfers the exchange disabled" gate and fires a real, non-idempotent,
never-retried POST at a disabled shard.

Demonstrated: with a healthy status the planner logs "Intra-exchange transfers are not active
on shard 0 and/or 1 — NOT moving $19.00" and sends **0** POSTs; with
`exchange_index_statuses` renamed (the same rename class as `orderbook` → `orderbook_fp`) it
POSTs 190,000 centicents to the disabled shard.

The producer conflates three states onto one `None`. That is right for `trading_active`
(TS-04: unknown must keep the shard in the ingest) but inverts the polarity for the money
flag — and the module is otherwise careful here, storing the same flag as
`_status_flag(...) is True` so an unreadable *value* fails closed (TS-04b). Note the suite
pins each half in isolation but nothing pins their composition, and the pinned test's own
comment describes the "sandbox / pre-sharding shape", not a read failure on a live 4-shard
account.

**Fix.** Do not decide a money question from a value that also means "read failed". In
`ensure_shard_collateral`, refuse to transfer when `shard_statuses is None` **and** the
account is demonstrably sharded (`len(shard_balances) > 1`, or any selected leg off
`DEFAULT_EXCHANGE_INDEX`), logging the same "move the funds manually" warning and dropping
those specs. `_transfers_active`'s `None → True` stays, so sandbox/single-shard accounts and
the pinned test are unaffected.

### DRA-16 — The per-leg cent ceiling can exceed the balance `select_portfolio` approved
**`trader.py:1398` vs `strategy.py:854`** · severity **medium** · correctness · **[V2]**

`select_portfolio` gates on the un-rounded float `total_cost_with_fees`;
`_required_cents_by_shard` re-totals with `math.ceil` applied **per leg**. With
`a*100 = A+fa`, `b*100 = B+fb` and `0 < fa, fb < 1`, the requirement is `A+B+2` while the
approved budget can be `A+B+1`. On a single-shard account there is no surplus to cover it,
`_plan_transfers` returns `[]`, and `_partition_by_funding` drops **every** spec on that
shard — the whole run's trading — over a rounding cent, warning that the shard is
"under-funded" when it is not.

Two independent measurements, differing by sampling: **65.9%** of runs when the balance is
within 3¢ of the portfolio cost, and **0.5%** (21/4000) over random near-full-balance
portfolios. Depth-weighted leg prices carry sub-cent precision by design, so the fractions are
non-zero in the normal case.

**Fix.** Make the budget gate use the planner's arithmetic. In `select_portfolio`, compare
`(ceil(round(cost_a*100,6)) + ceil(round(cost_b*100,6))) / 100` against `available`,
decrementing by the same quantity. (Ceiling once per spec in `_required_cents_by_shard` would
break the per-leg cross-shard split, so `select_portfolio` is the right place.) Note this
interacts with DRA-01 — fix them together.

---

## P3 — Silent drops and observability gaps

Each of these makes a real condition indistinguishable from a healthy run. The codebase's
own convention — one summary WARNING carrying a count, silent at zero — is the fix for most.

### DRA-17 — A funded shard whose `exchange_index` is unparseable is dropped in COMPLETE silence
**`auth.py:210-217`** · severity **high** · silent-drop

`_balance_cents_by_shard` warns loudly ("that shard's funds are NOT counted") when an
entry's *balance* is unparseable, but `continue`s with **no log at all** when its
*exchange_index* is — while the money disappears from sizing, the coverage audit and the
transfer planner identically. `int(None)` raises `TypeError` straight into that `continue`,
and the live API already started sending `category: null` on a field the SDK types as
required.

Measured consequences on an account holding $1 on shard 0 and $500 on shard 2: `verify_auth`
returns `{0: 100}` and logs `Auth OK … (total $1.00)` with no warning; `_run_prod` then
returns `EXIT_SKIPPED_LOW_BALANCE` on an account holding $501; `check_shard_coverage`
downgrades shard 2's blind spot from **CRITICAL** to a warning because it appears unfunded;
and `ensure_shard_collateral` either drops every trade settling there or plans a real POST
that was never needed.

**Fix.** Keep the skip; add a WARNING before the `continue` mirroring the balance branch,
and count the non-dict (`AttributeError`) arm separately so a wholesale shape change is
distinguishable from one bad row.

### DRA-18 — A missing/renamed `yes_ask_dollars` empties the scan and the run reports success
**`scanner.py:988-994`** · severity **medium** · silent-drop

In the same function, a missing `close_time` is counted and reported; a missing or
unparseable `yes_ask_dollars` is swallowed by a bare `except (ValueError, TypeError): pass`
with no counter and no log line. If Kalshi renames or re-types that field, **every** market
is discarded inside both finders while ingest, the shard census and `check_shard_coverage`
report a completely healthy run — `_blind_run_reason` is evaluated on the RAW ingest (by
design, TS-01/VI-02), which is non-empty, so it returns `None` and the process exits 0.
At prod scale that is ~135,000 markets discarded in silence every Monday.

**Fix.** A second counter in the `except` arm and one summary WARNING, silent at zero, in the
same `warn_missing_close`-gated block. Do **not** conflate it with the 1¢–99¢ eligibility
drop, which is a legitimate business filter.

### DRA-19 — `scanner._fetch_orderbook` logs the whole SDK exception once per ticker (TS-02 class)
**`scanner.py:2812`** · severity **medium** · log flood

`logging.warning("Orderbook fetch failed for %s: %s", ticker, exc)` renders
`ApiException.__str__` in full — status, reason, the entire response header dict and the
body, across five lines and ~615 bytes — once per market whose book fetch fails, and the
exception has already exhausted six retries, so a book-wide 5xx puts *every* call through it.
Measured: 200 book fetches = **123,000 bytes / 1,000 lines**; the one-line idiom costs
~20,000. `historical._exception_summary`, built for exactly this, is not used here.

TS-02's rule is stated for *any* per-ticker path, not just the backtester's — and this one
writes into the live `kalshi_arb.log` that must also hold the shard census, the pairs table
and any ORPHANED POSITION dump from the same run.

**Fix.** `scanner.py` must not import `historical.py` (dependency order), so either lift
`_exception_summary` into `_http.py` or inline the same one-line read. Then count failures in
`enrich_with_orderbook_prices` and emit one `N of M orderbooks unavailable` summary WARNING,
silent at zero.

### DRA-20 — `tick_size_for_price`'s fallback WARNING can fire ~40–90× per candidate pair
**`scanner.py:313-320`** · severity **low-medium** · log flood · **[V2]**

The comment says it is "called once per order leg at build time, so a warning here cannot
spam the log". Measured: `v2_limit_price` calls it **four** times (scanned price, final
price, and two clamp probes at 0.0001 / 0.9999), and it is reached from
`strategy._reachable_contracts` and `scanner.validate_pair_price` via `v2_effective_cap`
once per leg per candidate size in the bisection — **40 warnings from one `compute_trade`**
in one measurement, 88 tick lookups in another. A market whose `price_ranges` cover only the
tradeable range (a single `[0.01, 0.99]` band) has no band containing either clamp probe, so
every call emits two. ~1,000 candidate pairs ⇒ ~40,000 lines per run.

**Fix.** Correct the comment; and either hoist the two clamp-bound lookups to once per market
(they do not depend on the scanned price) or emit the WARNING through a per-(ticker,
structure) `seen` set.

### DRA-21 — `fetch_shard_statuses` silently last-wins on a repeated `exchange_index`
**`scanner.py:1490`** · severity **low** · parity

Two `exchange_index_statuses` entries with the same index silently overwrite, last one
winning, with no warning — while `auth._balance_cents_by_shard` treats the identical payload
shape as worth a WARNING by explicit design. Here the consequence is larger: a later
duplicate with `trading_active: true` un-halts a shard the exchange just reported halted, and
one with `intra_exchange_transfers_active: true` re-enables a real collateral POST.

**Fix.** Count repeats and emit one summary WARNING, silent at zero. For `trading_active`,
prefer the safe direction on conflict: if any entry for a shard says `False`, keep `False`
regardless of order.

### DRA-22 — `_extract_pairs` drops unusable-`close_time` group members silently
**`backtester.py:994-995`** · severity **low** · silent-drop

The time-series branch filters out members whose `close_time` is missing or unparseable with
no counter and no log, so a corpus that lost that field reports a reduced (or zero)
`Potential pairs` count indistinguishable from a window that genuinely had no pairable
markets. The same-title branch deliberately does *not* drop them, so the two counts move
differently for one corpus — reading as a pairing result rather than a data defect. Every
sibling drop path in the codebase reports itself once with a count.

**Fix.** One function-level counter, one summary WARNING after the group loop, silent at
zero. Do not log per member (TS-02) and do not change which members are dropped.

### DRA-23 — The emergency rescue dump omits the pair type and each leg's bought side
**`main.py:801-811`** · severity **medium** · spec-divergence

When `append_to_prod_log` fails, `_run_prod` dumps every `TradeResult` to the log as the sole
surviving record of real fills — but prints only `A=<ticker> B=<ticker> | x=<n> y=<n>`, with
no `pair_type` and no side. Since which side a count represents is a pure function of the
pair type, an operator holding an orphaned position cannot tell whether `x=40` on market B is
YES or NO. `main.py`'s own module docstring promises the opposite: "the pairs table,
`_print_portfolio` **and the rescue dump** render legs in MARKET order … **with the bought
side next to each count**." Zero test coverage (`grep RESCUE tests/` → nothing).

**Fix.** Add `side_a, side_b = leg_sides(r.spec.pair.pair_type)` and render
`%d× %s(A) + %d× %s(B)`, mirroring `_print_portfolio`. Add a test driving `_run_prod` with
`append_to_prod_log` raising.

### DRA-24 — The rescue dump's `|`-delimited format is broken by every time-series pair
**`main.py:802`** · severity **low** · correctness

The rescue line is a `|`-delimited five-field record that embeds `spec.pair.canonical_title`
unescaped. Since DR-01 a time-series canonical title is `"<event> | <market> | <subtitle>"`,
so every time-series rescue row emits **seven** `|`-separated fields where the format defines
five, shifting every later field. Same-title rows are unaffected — the malformed rows are
exactly the directional-bet rows.

**Fix.** Quote the title (`| %r |`). Natural to fix together with DRA-23.

### DRA-25 — `_print_portfolio` prints a fee-LESS return % beside a fee-INCLUSIVE cost
**`main.py:177-190`** · severity **low** · correctness

The per-trade line renders `cost $X incl. fees, profit if won $Y (Z% return)` where `X` is
`total_cost_with_fees` but `Z` is `profit_ratio` — deliberately fee-less (DR-62: it is the
ranking key and must not change). So `Y/X` never equals `Z`, on the same line, with nothing
signalling the denominators differ. Measured: `cost $188.40 … profit $111.60 (62.0% return)`
where `111.60/188.40 = 59.24%`. This is the same confusion TS-12 fixed one field over.

**Fix.** Do **not** change `profit_ratio`. Relabel to `(%.1f%% return on contract cost)`, or
print both denominators.

---

## P4 — Documentation defects

These change no behaviour today. They are listed because each one, read literally, would
lead a future editor to write a real-money regression — which is the bar CLAUDE.md itself
sets for recording a doc defect.

### DRA-26 — `SIZE_SOLVE_MAX_ITERATIONS`'s comment describes a descent that no longer exists
**`config.py:70-78`** · **[V3]** — reached independently by three agents.

The comment says the constant bounds "`strategy.compute_trade`'s marginal-price **descent**",
that "the sequence is strictly decreasing", that "convergence takes **1-3 passes** in
practice", and that exhaustion yields "one WARNING and **a skipped pair**". The shipped
`strategy._solve_marginal_size` is an ascending **binary search** taking ~log2(max_contracts)
passes (measured 10 / 14 / 20 for 600 / 10,000 / 1,000,000 contracts), and on exhaustion it
returns the largest verified count and **the pair is traded at that size**.

Concrete hazard, measured: an editor who lowers 64 to 3 on the strength of "1-3 passes"
silently sizes **n=875,000 instead of 1,000,000 and trades it** — a 12.5% under-size, not a
skip. CLAUDE.md and the function's own docstring are both correct; `config.py` is the stale
one, and nothing pins it (`grep SIZE_SOLVE tests/` → nothing).

**Fix.** Rewrite to name `_solve_marginal_size`, describe the bisection, state the real
exhaustion behaviour, and state the floor an editor actually needs:
`SIZE_SOLVE_MAX_ITERATIONS` must stay ≥ log2(max_contracts).

### DRA-27 — `MVE_MAX_EMPTY_PAGES`'s comment describes the pre-2026-08 DEFECTIVE reset rule
**`config.py:727-734`**

The comment says the pull stops after "this many CONSECUTIVE pages that contain no nested
markets" and that "Pages that DO contain markets reset the counter" — precisely the rule
CLAUDE.md records as **the bug fixed in 2026-08**. The code resets only on a page that
appends at least one market with `status == "active"`. `config.py` is where the
documentation standards say a constant's purpose lives, and this text is a precise
description of the defective behaviour with no staleness marker, so an editor reconciling it
against `scanner.py` would "fix" the code backwards and reinstate the 4,600-page sandbox hang.

**Fix.** Replace with the active-market rule, matching `scanner.py:1800-1804`.

### DRA-28 — CLAUDE.md's TS-34 direction-guard paragraph has never matched the code
**CLAUDE.md:156** · **[V4]** — reached independently by four agents, and confirmed via
`git show 41c4f71` (the TS-34 commit itself) that the doc was wrong from the start.

It says the guard "is ONE expression for both pair types — `reference > avg_yes`" and
"Keep the strict `>` rather than re-asserting the full tier". The code applies it to
**time-series only** (`scanner.py:3031`, with a ten-line comment explaining why same-title is
deliberately excluded) and its `ref_yes is None` fallback **does** re-assert the full tier
with `PRICE_EPSILON` (`scanner.py:3046`). CLAUDE.md contradicts itself: its own TS-09
paragraph enumerates "the `ref_yes is None` fallback's tier test" as a `PRICE_EPSILON` site.

Hazard: an editor following it either extends the guard to same-title (dropping sound
near-arbitrages, which the code comment explicitly warns against) or "restores" the strict
`>` in the fallback — and that branch's own comment states the consequence: "a mixed-snapshot
gap of a thousandth would otherwise pass, and as the gap shrinks `time_series_profit_prob`
rises toward 1.0 and Kelly sizes toward the `BUDGET_FRACTION` cap." No test pins the
fallback's tier magnitude, so that edit ships green.

**Fix.** State that the guard is time-series only and has two arms — strict `>` on a
refreshed reference, full tier on the scan-time fallback. Note that same-title's `pA` is
still refreshed, for the reported Price Diff only.

### DRA-29 — `strategy.py`'s bisection comment asserts an invariant that is measurably false
**`strategy.py:675-684`; CLAUDE.md:168, :177**

The post-shrink-snap comment asserts `_solve_marginal_size` "converges to the top of the
LOWER island and **cannot land in the upper one at all**", hence "**every** count it can
return sits in a region where the cap covers the whole reachable prefix and shrinking keeps
that true". Measured: under enrichment's own `max_contracts` the bisection lands **above a
hole in 29,203 of 106,800 solved pairs (27%)**. CLAUDE.md's supporting evidence — "118,726
randomized **two-level** ladders" — could never have exercised the shape, which needs 3+
levels.

The snap catches it (0/106,800 fired), so this is a doc defect — but the *reason given for
keeping the snap is wrong*, and a reader trusting the comment concludes the snap is dead code
and that a non-bisecting solver is interchangeable.

**Fix.** Replace the false invariant with what is measurably true: the bisection *can* land
above a hole; its empirical non-firing comes from enrichment's affordability cap keeping `hi`
low, **not** from any property of bisection. Keep "do not delete it as dead code". Same edit
at CLAUDE.md:168 and :177.

### DRA-30 — `strategy.py`'s module docstring calls `leg_sides()` cosmetic
**`strategy.py:24-25`**

"`leg_sides()` only names the sides in the log line." It is load-bearing twice:
`_reachable_contracts` builds each leg's FoK cap from it, and `compute_trade` uses it to
choose **which two of the pair's four price fields** the solved marginal prices are written
back to — i.e. what `leg_prices(spec.pair)` returns and therefore what `_ordered_legs`,
`_v2_limit_price`, `_buy_max_cost_cents` and `_rollback_floor_cents` submit at. Demonstrated:
enrichment wrote `pA=0.31333`, `compute_trade` replaced it with `0.30`.

Hazard: an editor treating the call as log-only routes a time-series pair's solved prices
into `nA`/`pB` — reporting-only for that pair type — so the trade submits at the price of a
different quantity than it sized, and the rollback floor derives from the wrong price.

**Fix.** Say it is load-bearing, and refresh the same block's stale import list
(`ORDER_API_VERSION`, `PRICE_EPSILON`, `SIZE_SOLVE_MAX_ITERATIONS`, `max_affordable_pairs`,
`prefix_fill_prices`, `v2_effective_cap` are all imported and unlisted).

### DRA-31 — `_interval_calibration`'s docstring claims a population it does not measure
**`backtester.py:2416-2420`**

"…it is exactly the conditional distribution the live sizer faces, so `k_hat` is the right
number to compare `TIME_SERIES_INTERVAL_PROB_DISCOUNT` against." It is not: it reads
`raw_entries` directly, including the many overlapping pairs of one title group that the
one-pair-per-group rule discards and the time-series copies `_drop_cross_type_duplicates`
drops. `_extract_pairs` emits every windowed pair of a group, so a many-deadline family
contributes O(n·window) overlapping observations sharing legs, and `k̂` is a family-weighted
ratio-of-means. Measured: 4 pairs in the denominator where the live rules leave the sizer
facing one.

This matters because `k̂` is what the dashboard renders as a recommendation for the
real-money discount constant. The k-independence the rest of the docstring defends is
correct and must not change — only the characterisation is wrong.

**Fix.** Name the population precisely, and mirror the correction in CLAUDE.md.

### DRA-32 — `Raises: KeyError` is documented on all three backtest entry points as a production guarantee
**`backtester.py:1710`, `:2629`, `:2737`**

All three say a mis-prefiltered ticker surfaces as a `KeyError` and "is treated as a real
defect … not degraded into 'no price history'". In production that cannot happen:
`historical.fetch_candlesticks` catches `except Exception` and returns `[]`, and the consumer
side is `.get`-guarded. The `KeyError` the suite asserts comes from a **monkeypatched**
fixture. So a prefilter regression yields a smaller trade count plus one aggregate WARNING —
indistinguishable from a post-cutoff window — while the operator believes a loud guard exists.

**Fix.** Reword all three: the pool *is* transparent to worker exceptions (keep that note,
it is load-bearing), but `fetch_candlesticks` fail-softs to `[]`, so the `KeyError` is what
the tests inject to pin the transparency. Do **not** narrow `fetch_candlesticks`'s catch —
post-cutoff 404s are expected.

### DRA-33 — Backtest Pass 2 claims to mirror `_run_prod` "exactly" but has no `MIN_BALANCE_CENTS` gate
**`backtester.py:2159-2166`, `:2227`**

A live run below `MIN_BALANCE_CENTS` ($50) short-circuits with `EXIT_SKIPPED_LOW_BALANCE` and
trades nothing; the backtest keeps entering trades down to a $5 balance (measured: $49.99 →
14 contracts; $5.00 → 1 contract). It biases exactly the ruin tail the dashboard's drawdown,
Sharpe, Sortino and per-`k` sweep table report on — the numbers an operator reads when
choosing the discount constant.

**Fix.** Skip the checkpoint when `checkpoint_cash * 100 < config.MIN_BALANCE_CENTS`, or drop
the word "exactly" and record the omission.

### DRA-34 — `_extract_status`'s docstring states the pre-fix retry policy as the caller contract
**`_http.py:134` vs `:394-395`**

"Returns None when no status can be found — **the caller should treat that as
non-retryable**." Its only caller does the opposite (`transient = status is None and
_is_transient_network_error(exc)`). The sentence states the policy CLAUDE.md records as
having killed a multi-hour backtest fetch on 2026-08-03, and sits in the function a
maintainer reads first when reasoning about retry classification — reading as licence to
delete the `transient` disjunct "for consistency". `tests/test_http.py` pins the return
values only, so such an edit passes.

### DRA-35 — Stale symbol and count references across CLAUDE.md
**Various** · **[V3]**

- **`trader._SCANNED_PRICE_QUANTUM` does not exist.** It is `scanner._SCANNED_PRICE_QUANTUM`
  (`0.000001`); `trader` has only the unrelated `_V2_PRICE_QUANTUM` (`0.0001`, the 4-dp wire
  quantum). An editor "fixing" the near-miss to 6 dp would emit off-grid price strings.
  Relatedly, "`ceil_to_tick`, `v2_limit_price`, `v2_effective_cap` … only re-exported by
  `trader.py`" over-claims — only the first two have trader aliases.
- **"Five scanner sites carry `PRICE_EPSILON`"** — there are six in `scanner.py` (2088, 2256,
  2948, 3046, 3151, **3186**) plus a seventh in `strategy.py:333`. *Agents disagreed on
  whether this is an error*: the uncounted sites are TS-08 reachability-cap comparisons, a
  different class from the TS-09 threshold tests the paragraph enumerates. Recorded as
  editorial, because the paragraph ends "the AST parity pins do NOT check this constant, so a
  missed mirror is silent divergence" — i.e. the enumeration is meant to be the checklist.
- **`tests/test_strategy.py::test_ast_backtester_prices_through_helper`** is a method of
  `TestTimeSeriesKellyParity`; the quoted node ID collects nothing.
- **`config.time_series_profit_prob`'s docstring names `backtester.run_backtest`** as one of
  three callers. Since the k-split the priced call lives in `_simulate_at_discount`.
- **`_recheck_and_report_position`'s `Returns:`** still says "_FAIL at **both** of today's
  call sites"; there are three.
- **The Notes-prefix wording** is given three ways; two omit the ` fees=$x.xx` suffix the
  code emits — the suffix TS-12 added precisely because an existing workbook keeps its old
  "Total Cost ($)" header above a now fee-inclusive column.
- **CLAUDE.md's label rule** says "the reporter rows list MARKET order … **with the side
  rendered next to each count**". The reporter deliberately does the opposite (side-neutral
  `"x — A leg"` headers; sides only in the Notes prefix), and says so in its own comment.
  **[V2]**
- **CLAUDE.md's EV-residual caveat** says "at single-digit `n`"; it was measured at n=30 (its
  own example), n=58 and n=105. The magnitude bound (≤ $0.02/trade) is right; the framing is
  not.
- **`README.md`'s live data-flow diagram** omits `main._blind_run_reason()` and the
  held-ticker filter, placing the horizon filter exactly where the blind check must go — so
  an editor following it would land a new filter above the blind check and reintroduce
  TS-01/VI-02.
- **CLAUDE.md:244** says the position baselines are read "immediately **before each
  submission**"; both are read up front before either order is submitted, deliberately, so no
  retryable lookup sits in the unhedged window.
- **`trader.py:2196-2205`** says the mapping read "costs at most one round trip"; on a zero
  first delta it is two reads plus a 1.0s sleep, inside the unhedged window.
- **`config.py`'s `INTERVAL_DISCOUNT_SWEEP` comment** says at `k = 1.00` "nothing trades".
  Same-title pairs price on the fixed `SAME_TITLE_CO_RESOLVE_PROB` and are k-independent, so
  that point routinely trades — contradicted by CLAUDE.md's own DR-61 record of a `k = 1.00`
  point with 3 profitable trades. (CLAUDE.md scopes it correctly to time-series; `config.py`
  does not, in either of two places.)
- **`_fetch_archive_sequential` memory claims** — see DRA-08.
- **`_await_transfer_settlement`'s `Returns:`** — see DRA-02.
- **`_solve_marginal_size`'s non-convergence WARNING** claims "using the largest count
  verified so far" but returns `None` and drops the pair when nothing was ever verified
  (unreachable today — needs `max_contracts ≥ 2^64` — but CLAUDE.md:168 repeats the claim).

### DRA-36 — BS-17's catch-up docstring promises coverage `_maybe_catch_up` does not deliver
**`scheduler.py:80-104`, `:649-669`**

`_maybe_catch_up`'s only triggers are `last_slot < slot` and
`exit_code == EXIT_NO_TRADEABLE_SHARDS`; neither reads `finished_at`. A slot **claimed but
never finalized** — the claim `_save_state` landed, then `subprocess.run` died (host reboot,
or the `UnicodeDecodeError` this file already anticipates) — is silently treated as satisfied
for the rest of the week. The module docstring reads as a promise of exactly that coverage
("guards against a missed run when the daemon was offline (host down, mid-deploy)").

**This is filed as a doc defect, not a code bug:** `TestCatchUp::test_current_slot_recorded_skips_catch_up`
pins the behaviour as intentional, and there is a plausible reason (an unfinalized claim is
ambiguous — the subprocess may have placed real orders moments before the write was
interrupted, so blindly retrying risks a second real-money run).

**Fix.** Either state plainly that catch-up recovers only a slot with *no* claim or one that
recorded `EXIT_NO_TRADEABLE_SHARDS`, and that a mid-run crash is deliberately left unretried
until next Monday to avoid double-submitting — or, if full coverage was intended, add a
bounded `finished_at is None` branch, update the contradicted test, and log a distinct
message for "recovering an interrupted run".

### DRA-37 — `infer_category`'s prefix table predates the `KX…` ticker convention
**`historical.py:137-146`**

Nine of 23 prefixes (`NFL`, `NBA`, `MLB`, `NHL`, `NCAA`, `EPL`, `PRES`, `SENATE`, `HOUSE`)
are bare and can never match a `KX`-prefixed ticker, and no combo/sports/soccer `KX…` prefix
is listed at all — so the dashboard's "P&L by market category" chart degenerates to a single
"Other" bar (~99.7% of the corpus is `KXMVE*`). Separately the 4-char `KXSP` entry claims any
`KXSP*` series for "Finance", so `KXSPEAKERHOUSE-26` renders as Finance.

**Fix.** Add `KX`-prefixed spellings for the families actually present, ordered
longest-prefix-first so `KXSPEAKER` is tested before `KXSP` — the same rule CLAUDE.md
already states for `_DATE_PATTERNS`.

### DRA-38 — Unvalidated CLI numerics (four independent reports)
**`main.py:924-927`, `:507`; `backtest.py:71-74`** · **[V4]**

`--max-horizon-days` and `--interval-discount` are both range-checked via `parser.error`
before any work runs. `--sandbox-balance` and `--balance` are not:

- `--balance nan` / `inf` parse fine, survive the (potentially multi-hour, real-API) fetch
  prologue, and crash inside `_simulate_at_discount`'s `int(budget / …)` with a raw traceback
  giving no hint the cause was the CLI. (`--balance 0` / negative are genuinely safe — the
  `n < 1` guard skips every candidate.)
- `--balance 0` separately crashes `dashboard._section_performance` with `ZeroDivisionError`
  before any HTML is written, while two sibling divisions in `_section_benchmark` render
  `"Total Return: +inf%"` instead — four "divide by opening balance" sites, inconsistently
  guarded.
- `--sandbox-balance -250` runs the whole scan and reports "no executable trades" in the same
  words as a genuine no-edge run; `nan`/`inf` raise uncaught out of `int(x * 100)`.
- `main.py:507` uses a bare `int(args.sandbox_balance * 100)`, omitting the round-before-int
  idiom the four money helpers all apply — `1234.35 * 100 == 123434.99999999999` loses a cent.

**Fix.** Add `parser.error` guards for finiteness and positivity on both flags, alongside the
existing checks; use `int(round(x * 100))`; and guard `_section_performance` and both
`_section_benchmark` divisions the way `_srow` / `_section_risk` already do.

### DRA-39 — Smaller confirmed defects
- **`v2_probe`'s NO-close exception handler is an undocumented FIFTH branch** that skips the
  shared `_recheck_and_report_position` (`v2_probe.py:919-926`). It is the only failure path
  in the module doing **zero** position reads — it prints the pre-close value as "may still be
  OPEN". CLAUDE.md and the helper's own docstring both enumerate "FOUR OTHER BRANCHES",
  undercounting because `_step_no_mapping` has two submission sites. **Fix:** route it through
  the shared helper and correct the enumeration.
- **`_day_store_load` applies the `keep` predicate inside the `except ValueError` guard on the
  JSONL branch and outside it on the legacy branch** (`historical.py:1219` vs `:1239`). A
  raising predicate is therefore misreported as slice corruption — only in the format every
  production worker writes — which raises `_ShardedFetchUnsupported`, abandons the whole
  parallel archive phase, blames a healthy file, and drops the run onto DRA-08's unbounded
  walk. Latent (`_can_ever_enter` is raise-free today). **Fix:** narrow the guarded region to
  the JSON parse only.
- **`_load_json_cache`'s "corrupt reads as a MISS" contract covers parse failures, not shape**:
  `dict(disk_titles)` on a JSON array raises `ValueError` (`historical.py:512`), and
  `cached["close_ts"]` is read after gating only on `open_ts` (`historical.py:2667`) — the
  latter inside a pool worker, so one shape-damaged per-ticker cache kills a whole backtest.
  **Fix:** `isinstance` guard on the accumulator; extend the candlestick gate to `close_ts`.
- **`ensure_shard_collateral` POSTs real transfers for a spec it can already prove will be
  dropped** (`trader.py:1801-1826`): the plan is per-deficit-shard and the execution loop
  gates per-transfer, while `_partition_by_funding` (which correctly requires BOTH legs
  payable) runs only after every POST. Measured: $40 really moved to shard 2, then the spec
  dropped because shard 3's transfers are inactive — $40 stranded, a second POST needed to
  return it. This is the failure `drop_legacy_unroutable` exists to prevent, transplanted to
  V2. **Note:** the closely-related *insufficient total surplus* variant IS pinned as
  deliberate; only the inactive-transfer variant is unpinned. **Fix:** drop provably-unfundable
  specs before building the plan.
- **`_rollback_floor_cents` uses `round()` where `ceil` is needed** (`trader.py:404`): the NO
  leg's price is a depth-weighted average, not a cent-quantized quote, so rounding down
  loosens the bound — realized loss up to 12.5¢/contract against a constant documented as a
  12¢ maximum. CLAUDE.md:252's stated justification ("that price is a cent-quantized book
  price") is contradicted by `trader.py`'s own adjacent comment. **Fix:**
  `math.ceil(round(price * 100, 6))`, the idiom already used twice in the module.
- **`_dollar_str_to_cents` lets `OverflowError` / `decimal.Overflow` escape** (`auth.py:160-165`),
  contradicting its "unparseable → None" contract; `Decimal("Infinity")` is a *valid* Decimal
  so neither construction nor `to_integral_value` raises, and the failure lands in `int()`.
  Reaches `main._run_prod` unhandled → exit 1 rather than a documented `EXIT_*` code.
  **Fix:** `except (ArithmeticError, ValueError, TypeError)`, or `if not dec.is_finite(): return None`.
- **A boolean top-level `balance` is accepted as integer cents** (`auth.py:265`):
  `isinstance(True, int)` is `True`, so `"balance": false` reports $0.00 and the run exits
  `EXIT_SKIPPED_LOW_BALANCE` instead of raising the documented `ValueError`. The sibling
  `scanner._coerce_int_cents` guards this explicitly. **Fix:** add `and not isinstance(legacy, bool)`.
- **`_bids_to_ask_levels` drops the extreme $0.0001 ask** (`scanner.py:2471`) · **[V5]** —
  five agents. `1.0 - 0.9999 == 9.999999999998899e-05`, one ULP below
  `MIN_ACTIVE_PRICE_DOLLARS`, so a bid at the top tradeable level of a centi-cent market has
  its complement — the cheapest possible NO ask — discarded and counted into the TS-14 drift
  WARNING. The other end is safe only by luck (`1.0 - 0.0001` is exact). **Fix:**
  `MIN_ACTIVE_PRICE_DOLLARS - PRICE_EPSILON <= ask_price <= MAX_ACTIVE_PRICE_DOLLARS + PRICE_EPSILON`.
- **`prefix_fill_prices` returns `None` on an exactly-sufficient fractional book**
  (`scanner.py:2409-2412`): `remaining` is decremented in floating point, so 240 of 864
  exactly-sufficient three-level fractional books read as short. Safe direction today (integer
  books unaffected); latent for the deferred fractional-sizing work. The adjacent comment
  ("No epsilon is needed") is the wrong half. **Fix:** `if remaining > config.PRICE_EPSILON`.
- **A null `orderbook_fp` beside a populated legacy `orderbook` reports an empty book**
  (`scanner.py:2747`, `:2766`): container selection is by key presence and stops at the first
  match, and a present-but-null container is then read as "no resting bids" — so the mismatch
  WARNING that exists to make container drift diagnosable cannot fire. **Fix:** prefer the
  first container whose value is a non-empty dict, falling back to the null/`{}` empty-book
  verdict only when none is usable.
- **`_max_drawdown` attaches a trough date when there was no drawdown**
  (`dashboard.py:165-194`): a monotonically non-declining curve yields an all-zero `dd` series
  that passes the NaN guards, and `dd.idxmin()` returns the *first* index on a full tie — so
  the page renders "Max Drawdown: 0.0% (2026-01-05)", implying a decline on a specific date.
  The docstring promises `(0.0, None)`. **Fix:** short-circuit when `(dd == 0).all()`.
- **`_depth_levels` type-checks the container but not its elements** (`strategy.py:174-175`):
  a pair carrying non-3-tuple entries raises `ValueError` out of `compute_trade`, the opposite
  of its documented "price on the scalars" fail-safe. No live producer today. **Fix:** require
  every entry to be a length-3 sequence of numbers.
- **`_save_json_cache`'s tmp filename has no PID/uuid** (`historical.py:410-432`) ·
  **[V2] [CORRECTED — narrower]**. Cross-process only; nothing in the repo runs two backtests
  at once, so severity is **low**. But the consequence is sharper than a crash: `tmp.replace()`
  by one process publishes the *other* process's still-open half-written inode, so the
  atomicity contract is **void**, not merely racy — measured **402 of 16,143 reads saw a torn
  file**, which `_load_json_cache` swallows as a cache MISS, silently losing the
  `event_titles.json` accumulator (the TS-11 shape). **Fix:** PID+uuid tmp name; scope the
  docstring's invariant to "per process"; an advisory lock around the read-merge-write would
  be the complete fix.
- **Backtest checkpoint timezone vs the scheduler** (`backtester.py:1063` vs
  `scheduler.py:773`) · **[VERIFIED] [CORRECTED]**. `_monday_timestamps` hardcodes 09:00
  **UTC**; `schedule.every().monday.at("09:00")` is passed no `tz`, and the installed library
  uses `datetime.now(None)` — naive **local**. Corrections: it is **7** candles on US/Pacific
  in DST (8 only in standard time), **zero on a UTC host**, and it *reverses* east of UTC
  (Tokyo fires 9h earlier). Both halves are separately documented (`scheduler.py:50-51`,
  `README.md:426` "local time"; `README.md:374` / `CLAUDE.md:266` "Monday-09:00-UTC") —
  **nothing says they must agree**, so the defect is the undocumented parity gap, not an
  unknown scheduler bug. Materiality demonstrated: the hour flips admission outright (ENTER at
  09:00 UTC, NO ENTRY at 16:00/17:00 UTC on the same fixture), and the admitted `pB − pA`
  feeds `_interval_calibration`'s `mean_implied` and hence the pooled `k̂`. **Fix:** one
  `config.ENTRY_CHECKPOINT_TZ` read by both `.at("09:00", tz=…)` and `_monday_timestamps`; or,
  if local firing stays, a CLAUDE.md sentence plus a startup INFO naming the resolved UTC fire
  time.
- **`manual_review` CRITICALs identify the leg by title, not ticker** (`trader.py:2291`,
  `:2398`): for a same-title pair both markets carry an identical title by construction, so
  neither the CRITICAL nor `TradeResult.error` says which ticker to flatten — in the two
  branches where a real position may be open and the bot deliberately submits nothing. Every
  other urgent alert in the module does name the market. **Fix:** interpolate
  `no_leg.market.ticker` / `yes_leg.market.ticker`; purely additive, tests pin the status not
  the wording.
- **`_load_state` validates that `last_slot` PARSES but not that it is naive**
  (`scheduler.py:280-288`, `:652`): a hand-edited tz-aware value raises `TypeError` out of
  `_maybe_catch_up`. `_startup_catch_up` catches it so the daemon survives, but the BS-17
  catch-up is skipped — a full week with no production run. **Fix:** normalise or reject in
  `_load_state`; don't leave it to the backstop, which keeps the daemon alive but loses the
  feature.
- **The sweep counter skips the primary's slot** (`backtester.py:2812-2820`): a default run
  prints `1/13 … 7/13, 9/13 … 13/13` — 8/13 missing — while the adjacent comment says "The
  index is filled in after the grid is known, below". TS-21's own symptom is genuinely gone;
  this is the residual. **Fix:** log the slot in the reuse branch before `continue`.

---

## Refuted or downgraded in phase 2 — DO NOT ACT ON THESE

Recorded so they are not re-reported by a future sweep, and because a first-pass agent
reported each of them as a genuine finding. This is the value of the refutation round.

### R-1 — "The crossed-book direction guard destroys the tier and sizes at the cap" — **DOWNGRADED to low**
Two phase-1 agents reported this as high-severity money loss (one quoting $19,999.69 of a
$100,000 balance, modelled EV +$250.44 vs a true −$242.64). A verifier independently
re-derived CLAUDE.md's implication chain — confirming it is **sound**, and that crossing
breaks exactly one step (`reference = 1 − max(NO bid) >= max(YES bid)`) — then tested the
*consequences*, where the reports overreached:
- **"The deadline-gap tier is destroyed" is false.** The qualifying-level ceiling
  independently guarantees the **executable** gap `(1 − nB) − pA >= tier`. Of 540 below-tier
  survivors, **0** violated it (`min(g_exec − tier) = -0.000000`, float boundary only). Only
  the *reference-ask* gap collapses. CLAUDE.md already says the ceiling "already implies an
  executable gap of at least the tier".
- **`p = 1.0` is unreachable.** The strict `>` keeps the gap strictly positive; worst float
  case `time_series_profit_prob(0.45, 0.45+1e-16) = 0.9999999999999999`. The TS-34 riskless
  clamp is genuinely closed.
- **"Kelly saturates at `BUDGET_FRACTION`" is true but not distinctive.** Re-priced with the
  reference placed where an uncrossed book would put it, **307 of 540 size identically**
  (both hit the 0.20 cap); 233 oversized, **0 undersized**; worst `f*` 0.2000 vs 0.1303.
- CLAUDE.md explicitly documents "Keep the strict `>` rather than re-asserting the full
  tier" as deliberate, so the accepted-trade-off exclusion applies.

**Residual worth a doc note only:** on a crossed book the deliberately-conservative model
input inverts, oversizing ~43% of that population by up to ~35% relative. A crossed
single-market snapshot is an arbitrage Kalshi's matching engine crosses instantly, so the
live domain is a torn read. Worth noting that the *stated* rationale for the bare `>` (a
float-boundary pair at exactly the tier) is already solved by the sibling branch's
`>= tier - PRICE_EPSILON`, so if anyone does tighten it, that is the precedent.

### R-2 — "`tick_size_for_price`'s short-circuit loosens the cap 10x-150x" — **DOWNGRADED to a one-sentence doc fix**
Two agents reported the two arms of `if structure in ("", "linear_cent") or not bands`. The
arithmetic is right (1 intended tick → 10 ticks at $0.50, 150 at $0.005), but:
- **Arm 1 is pinned as intentional.** `tests/test_scanner.py:3689`
  (`test_none_ranges_falls_back`) asserts the coarse default with **no** `caplog` assertion,
  while its two sibling tests *do* assert warnings. The author distinguished the three cases
  and chose silence for this one.
- **CLAUDE.md specifies both arms and the silence verbatim.**
- **The alleged `config.py` contradiction does not hold.** `config.py:283-284` says "unknown
  **or** uniform-cent"; `structure == ""` **is** unknown. The parenthetical enumerates
  examples, it is not an exhaustive trigger list.
- **It is internally self-consistent:** `strategy._reachable_contracts` derives its cap
  through the same `v2_limit_price`, so sizing and the submitted order carry the same (loose)
  cap — no TS-08-style disagreement, and a looser cap makes an order *more* fillable.

**Residual:** one false sentence at `scanner.py:252-253` ("the structure name is only used to
short-circuit the uniform-cent case"), which is untrue for the bands-present/name-absent arm.

### R-3 — "The NO leg's re-quantization floors the cap, making legit orders unfillable" — **NOT REPRODUCED**
The *sign* observation is correct: `ceil_to_tick` on the wire ask is a floor on the NO-side
cap, and the docstring's justification is a bid-side argument applied unexamined to the ask
side. But **all four documented Kalshi regimes are symmetric about 0.5**, so the complement
of an on-grid cap is itself on-grid. Measured **0 unfillable caps in 300,000 samples × 4
regimes × both leg kinds**; 304/400 synthetic *asymmetric* layouts show it, 0/400 symmetric.
The worst real effect on a live regime is the NO leg's 1-tick slippage allowance being
silently consumed (delta = 0.000000), **never** a cap below the scanned price. Reachable only
via band-drop drift, and even then the outcome is a killed FoK — a missed trade, not a loss.
The tightening direction on `buy_no` is genuinely unrecorded (both recorded examples are
`buy_yes`), so a one-line docstring note is the whole fix.

### R-4 — "The V2 mapping backstop does not bound blast radius to one pair" — **INTENDED BEHAVIOUR**
The factual mechanism is confirmed (8 workers, 8 NO legs submitted, latch still `False`,
8 × `manual_review`), but the "contradiction" is a misreading. The docstring sentence is
"…caught with exactly one wrong-side position outstanding **rather than a completed pair**" —
the contrast clause makes it a **per-pair** statement, not a process-wide bound. And the race
is named explicitly 45 lines below in the same docstring: "two trades can reach this before
either latches — that costs a duplicate positions read **and, in the disproven case, sends
both to manual_review**." The reporting agent quoted only the first half.

**Residual (doc-precision, low):** "Both outcomes are harmless" understates it — N unhedged
wrong-side positions need N manual flattenings; and "one extra positions read per process"
is up to `TRADER_MAX_WORKERS` reads under the race, each single-shot and in an unhedged
window.

### R-5 — "Entries are priced after a leg has settled" — **REFUTED** (the staleness half stands)
"No age bound" is **true** — `_candle_at_or_before` never compares ages, and because
`open_ts` is one shared `start_date` for every ticker, staleness can legitimately reach ~365
days. But **"including after a leg has settled" is refuted twice over**:
`scan_end = min(close_a, close_b) − 1 day` puts every checkpoint at least a full day before
the *earlier* close, and `_candle_at_or_before` can only return a candle at-or-before the
checkpoint. Verified empirically (no post-close candle was ever returned).

The "32 days apart" figure is **unverifiable offline** (`backtest_cache/candlesticks/` is
empty, no network), and two readings compete: CLAUDE.md's "a 2-hour market returned 2 hourly
candles" suggests a dense grid, while `fetch_candlesticks` *drops* candles with an absent
`yes_ask.close` — exactly what an illiquid hour produces. **Recommendation: do not fix
blind.** Add `MAX_CANDLE_STALENESS_SECONDS` with a generous default (48h) and a counted
summary WARNING to *measure* the real rate first.

### R-6 — "`_save_json_cache` tmp collision is high severity" — **DOWNGRADED to low**
See DRA-39. Cross-process only; nothing in the repo runs, documents or forbids two concurrent
backtests. Kept because the consequence found during verification (a **void** atomicity
contract, 402/16,143 torn reads) is sharper than the crash originally reported.

---

## Appendix — negative evidence

Recorded so a future sweep does not re-derive it. All figures are from executed harnesses.

| Area | Evidence |
|---|---|
| Leg-side single source of truth | 4,000 randomized pairs (both types, random shards/regimes) → **0 violations**; a 122-assertion differential harness through finder → enrichment → sizing → all builders → collateral → reporter → settlement |
| TS-08 reachability | 36,292 + 1,181 + 97,658 sized specs across 4 regimes → **0** unreachable `n` |
| Kelly / DR-62 | fee in the denominator in all three sizers; 61,305 parity points and 200 whole-checkpoint portfolio comparisons → **0 divergences**; every CLAUDE.md figure reproduces to the digit |
| `_execute_one` status matrix | every branch on both pair types and both order paths, + 144-combination ground-truth fuzz → **0** unescalated unhedged states, **0** misdescribing statuses, **0** escaping exceptions |
| Transfer planner | 60,010 randomized surplus/deficit maps → never self-directed, negative, over-funding or draining a source below its own requirement; 10,000 end-to-end DR-65 runs → no shard ever reported both in-flight and settled-short |
| Constants | 0 inlined literals from CLAUDE.md's list; 0 dead constants; all 9 shared-value collision groups proven correct by runtime sentinel patching |
| Retry policy | 46 call sites inventoried with *measured* attempt counts: 6 for retried GETs, **1** for all four submission/transfer sites |
| SDK surface | **zero** modeled SDK calls remain; every kwarg validated against the pinned SDK's real signatures |
| Backtester prologue | 6,000-trial `_find_entry` fuzz vs an independent reference model → **0 mismatches**; 6,336-combination exhaustive window check → **0** legitimate pairs dropped; corpus-level test selects the **identical** pair set as the live scanner |
| Docstrings | 261 functions mechanically compared; arg types, tuple arity, dataclass fields, `Dependencies:` blocks and defaults all clean; **no DR-66b sibling exists** |
| CLAUDE.md counts | 18 Excel columns, 13 dev headers, two scheduler registration sites, exactly two overridden keys in `_no_close_body`, three `mve_filter` sites, 13-point sweep with `primary is points[7]` — all verified; all 21 quoted log strings exist verbatim; every "What NOT To Do" bullet still honoured |
| Enrichment | 4,000-trial + 40,000-trial fuzz → `tradeable ⇒ max_contracts ≥ 1`, `depth_levels` ascending, `n ≤ max_contracts`, and the enriched price never optimistic relative to the traded price |
| Price grid | 117,220 + 79,992 combinations → 0 off-grid wire prices, 0 caps looser than intended + one destination-band tick, 0 round-trip failures through the 4-dp wire string |
| Scheduler | `_guarded_job` forwarding, `on_error`/`CancelJob`, `KeyboardInterrupt` passthrough, every `_load_state` malformed shape, all 4 exit-code mappings + unknown, retry-cap arithmetic 0→4, log isolation — all correct; **no** scheduler-internal timezone mismatch |
| Log ownership | importing all 14 modules leaves `root.handlers == []`; no module reconfigures another's handler |

---

## Suggested sequencing

1. **DRA-01 + DRA-16 together** — both are the funding-vs-approval arithmetic; fixing one
   without the other re-opens the drop path.
2. **DRA-02, DRA-03** — both in `ensure_shard_collateral`'s result handling; one commit.
3. **DRA-04, DRA-05, DRA-10, DRA-11, DRA-12** — five small fail-closed/except-tuple fixes,
   each independently testable.
4. **DRA-06, DRA-09** — both grouping-key correctness; land with new finder tests.
5. **DRA-07** — backtest look-ahead. Isolated, but **invalidates every existing backtest
   baseline**, so land it alone and re-run the reference backtest immediately after.
6. **DRA-08** — `historical.py` bounds; mechanical, four sites, one idiom.
7. **P3 observability** — mostly one-counter-plus-one-WARNING each; cheap, batchable.
8. **P4 docs** — batch into one commit per file. DRA-26, DRA-27, DRA-28, DRA-29 and DRA-30
   are the ones with a real mis-edit hazard; the rest are hygiene.

**Testing note.** Almost every finding above is unpinned — that is *why* it survived 1385
tests. Each fix should land with the test named in its entry, or the next sweep will find it
again.

---

## P2b — Live-vs-backtest parity

Baseline parity is good and was measured, not assumed: **24,576 exhaustive whole-cent grid
cases** (6 pair-type/gap combinations) plus 3,000 randomized cases, pushing ONE synthetic
universe through **both** pipelines end to end with the identical float prices, produced
**zero** admit/reject or sizing disagreements. The tier test, `min_price_diff_for_gap`, the
`1 - tier` ceiling, `MAX_DEADLINE_GAP_DAYS` and its band edges, both grouping keys, the
one-series rule, the leg mapping, Kelly `p`, Kelly's fee-inclusive `b`, the fee-less
`profit_ratio`, the budget shrink and `SAME_TITLE_CO_RESOLVE_PROB` all agree exactly.

The four divergences that remain all bias the **same way**: the backtest is never stricter
than live, so every backtest return figure is optimistic by their combined amount.

### DRA-40 — The one-best-pair-per-group rule sits on OPPOSITE sides of the price/Kelly gates
**`scanner.py:2117`, `:2310` vs `backtester.py:2117-2124`** · severity **medium** · parity

Live picks one pair per group **inside the finder**, where the only gate applied is the
continuous-fee `tradeable` check — the `1 - tier` leg-price ceiling lives downstream in
`enrich_with_orderbook_prices` and the Kelly gate in `strategy._evaluate_size`. Nothing
re-opens the group when the chosen pair dies there, so a group whose widest-gap candidate
later fails either gate yields **nothing**. The backtester's `best_by_group` runs **after**
`_find_entry` (which applies the ceiling) and after `kelly_f <= 0: continue`, so the
runner-up is promoted and a trade is booked on tickers the live pipeline never had a
candidate for.

Measured: over 1,200 randomized 3-market groups, **19 time-series and 18 same-title groups**
produce a backtest trade where live produces none — **11.4% of all backtest-trading
time-series groups** in that population. Worked example: live emits only (M0,M1) (widest
gap), enrichment finds no qualifying level, the run trades nothing; the backtest rejects
(M0,M1), accepts (M0,M2), and books **587 contracts for $475.47**.

`backtester.py:2110` records the *tie-break* difference ("mirrors the live scanners' CONCEPT
… not their tie-break rule") but not this far larger one. CLAUDE.md's DR-54 note
("removing a group's top candidate PROMOTES the runner-up") describes promotion for a skip
**inside** the finder loop; a ceiling/Kelly rejection happens outside it, so live never
promotes.

**This is the same defect family as DRA-07 and should be fixed in one change.** DRA-07 is
the settlement filters sitting before the dedup; this is the ceiling and Kelly gates sitting
before it. Option (a) below fixes both.

**Fix.** (a) *Backtest-side, cheapest and preserves live behaviour:* build `best_by_group`
from the pre-Kelly `_extract_pairs` output using the live tie-break (`tradeable` first, then
the signed price gap read off the entry), so a group that loses its top candidate downstream
yields nothing, exactly as live does. (b) *Live-side:* keep all of a group's tradeable
candidates through enrichment and apply the one-pair-per-group rule in
`strategy.select_portfolio`, after the ceiling and Kelly gates — this changes real-money
behaviour and needs its own review. Say which in CLAUDE.md.

### DRA-41 — TS-08 reachability has NO backtest mirror, and CLAUDE.md never records the asymmetry
**`strategy.py:290`, `:380`, `:690` vs `backtester.py:2237`** · severity **medium** ·
parity / doc-defect

`grep -c reachab kalshi_betting/backtester.py` finds no reachability logic at all. The
backtest sizes against the candle's top-of-book quote with no depth cap and no FoK-limit cap
(`n = int(budget / (price_a + price_b))`, capped by nothing), so it routinely books sizes the
live fill-or-kill order could never fill. Because a candle quote is the TOP of book and any
live prefix average is ≥ it, **the bias is one-sided for every pair**: backtest price ≤ live
price ⇒ backtest `net_spread`, Kelly fraction and size are all ≥ live's.

Measured on one pair: flat deep book → both size 3,802. Put a two-level ladder on either leg
and live returns **300** while the backtest still books **3,802** — a **12.7× oversize**.
With 5 contracts of top-of-book depth, live returns **5** against 3,802.

CLAUDE.md's TS-08 section names three application sites, all live, and never says the
backtester mirrors none of it — in a file that elsewhere says "the backtester's `_find_entry`
mirrors all of it" (of the tiered filter) and "a missed mirror is silent divergence". The
silence reads as "mirrored".

**Fix — doc-only, and it is the cheap half.** Add one sentence to CLAUDE.md's TS-08
paragraph: the backtester mirrors NONE of it (no book exists in candle data), backtest sizes
are therefore an **upper bound** on live sizes, and a backtest return figure is optimistic by
whatever the live depth/FoK cap removes. If fidelity is ever wanted, note that a candle
carries no depth, so a real mirror needs a depth source that does not exist today — say that
too, so nobody re-derives it.

### DRA-42 — `_find_entry` records only the FIRST qualifying Monday, defeating the ticker-release parity Pass 2 claims
**`backtester.py:1099` vs `:2196-2206`** · severity **low** · parity

Pass 2 releases a ticker on its trade's exit date precisely so a later candidate can reuse
it, "mirror[ing] the live bot's rule precisely". But `_find_entry` `return`s inside the
`_monday_timestamps` loop, so every candidate carries exactly ONE entry date — a candidate
blocked on that Monday is dropped permanently, where a live weekly run would re-scan and
enter it once `get_held_tickers` released the ticker. The release happens; the re-entry it
exists to enable cannot.

Measured: X/Y and X/Z both first qualify 2026-03-02; X/Y wins and blocks ticker X until it
settles 2026-03-10. X/Z is never reconsidered. Fed the same candidate at 2026-03-16 — what
live would do — it produces a **2,915-contract trade** the backtest reports as never having
existed.

**Fix.** Either have `_find_entry` return every qualifying checkpoint and let Pass 2 take the
first unblocked one, or — much cheaper — record the scan window on the entry and, when Pass 2
skips for a ticker conflict, re-try at the first Monday on/after the blocking trade's exit
date still inside that window. If neither, amend the Pass-2 comment: the mirror is partial
and the backtest under-counts re-entries.

### DRA-43 — `--max-horizon-days` is a DATE span in the backtest and a DATETIME span live
**`backtester.py:1269` vs `scanner.py:1029`** · severity **low** · parity

Live compares tz-aware close *datetimes* against `now + N days`; the backtester compares
close *dates* against the checkpoint *date*, discarding the 09:00 time of day on both sides.
The backtest's window therefore runs to 23:59 on day N — up to ~15 hours wider — and the skew
is one-sided: **the backtest is never stricter**.

Measured: `--max-horizon-days 14`, checkpoint 2026-03-02 09:00 UTC, later leg closing
2026-03-16 23:00 UTC (true span 14d 14h). Live's cutoff is 2026-03-16 09:00 UTC so the market
is dropped before pairing and the pair never exists; `_find_entry` measures 14 days by date
and enters.

The docstring's stated difference is the reference *point* ("relative to each simulated
checkpoint rather than real-world now"), not the resolution.

**Fix.** Compare datetimes in `_find_entry` — the time-series branch already parses them —
with the current date arithmetic as the naive/aware `TypeError` fallback, exactly as the gap
computation beside it does. Same-title needs the two datetimes parsed there too.

---

## Appendix B — What the test suite actually protects (mutation testing)

The AST parity pins verify **only that a name is called somewhere in a function body**.
`tests/test_strategy.py::_function_calls` finds the named `FunctionDef` and returns True if
any `ast.Call` inside it names the callee — it checks no arguments, branch, polarity or
conjunction. Two concrete gaps: a swapped `pA`/`pB` passes the Kelly-helper pin, and a `""`
second argument passes the group-key pin (the subtitle is the whole point of DR-01).

To find out whether the semantic gaps are covered *elsewhere*, each mirror was broken on a
**copy** of the repo and the full 1385-test suite run against it — 29 mutations, one full run
each. **15 of 19 backtester mutations and 6 of 10 live mutations are caught.** The 8
survivors are where silent divergence can actually live:

| Mutation (on a repo COPY) | Suite result |
|---|---|
| backtester: `gap < threshold - PRICE_EPSILON` → `gap < threshold` | **1385 passed** |
| backtester: ceiling `+ PRICE_EPSILON` removed | **1385 passed** |
| backtester: `profit_ratio_entry` given Kelly's fee-INCLUSIVE denominator (DR-62's "do not collapse the two") | **1385 passed** |
| backtester: `_find_entry`'s fee/profitability gate disabled | **1385 passed** |
| scanner: enrichment ceiling `+ PRICE_EPSILON` removed | **1385 passed** |
| scanner: `validate_pair_price` ceiling `+ PRICE_EPSILON` removed | **1385 passed** |
| scanner: `ref_yes is None` fallback tier `- PRICE_EPSILON` removed | **1385 passed** |
| strategy: `_reachable_contracts` cap `+ PRICE_EPSILON` removed | **1385 passed** |

So **2 of the 7 `PRICE_EPSILON` sites CLAUDE.md enumerates are pinned** (the two finder tier
tests, by `TestPriceEpsilonThresholds`); the other five, plus the two reachability-cap sites,
are not. The price point they would break at is the one CLAUDE.md already documents:
`0.35 - 0.20 == 0.14999999999999997`, a hair under the 15% tier.

Note the fourth survivor is **not** redundant: disabling `_find_entry`'s fee gate changes no
trade (the Kelly gate re-rejects them) but *does* change `_interval_calibration`, whose k̂
population is `_find_entry`'s output — and k̂ is the recommendation for the real-money
discount constant.

This is a **test-coverage gap, not a code defect** — the constant is present and correct at
all seven sites, and the 24,576-case differential shows live and backtest agreeing. It is
recorded so the fixes above land with pins that would actually catch a regression.

Everything CLAUDE.md claims is pinned **by value** genuinely is: removing the fee from the
backtester's `kelly_b_entry` fails `TestKellyRiskIncludesFees::test_all_three_sizers_agree_by_value`,
`::test_the_backtester_also_rejects_the_headline_fixture`, and three `TestRunBacktestTimeSeriesFlow`
tests.

**One more negative result worth keeping.** `main._dedup_pairs` vs
`_drop_cross_type_duplicates`: the whole whole-cent admissible region was brute-forced
(5 gap tiers × 99³ price points) looking for a point where the same-title copy fails Kelly
but the time-series copy of the same two tickers passes — which the position of
`_drop_cross_type_duplicates` after the Kelly gate would resurrect. **0 such points exist**:
when both copies exist the leg prices are the same two numbers and `p = 0.95` dominates.
