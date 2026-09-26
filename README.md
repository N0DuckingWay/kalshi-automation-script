# Kalshi Arbitrage Bot

An automated pair-trading bot for the [Kalshi](https://kalshi.com) prediction market platform. The bot finds pairs of related prediction market contracts where one is mispriced relative to the other, sizes positions using the Kelly criterion, and submits fill-or-kill orders leg-by-leg — with automatic rollback if the second leg doesn't fill. A separate backtesting pipeline replays the same strategy on the full history of settled Kalshi markets and generates an interactive HTML performance dashboard.

---

## How It Profits

Kalshi markets are binary contracts that pay $1 if a question resolves YES and $0 if it resolves NO. The bot exploits two specific pricing anomalies:

**Time-series pairs:** Two contracts asking the same question about the same outcome at two different **deadlines** (e.g. "Will BTC exceed $80k by March 2025?" and "Will BTC exceed $80k by June 2025?"). The "by" matters: the strategy only works between *cumulative-deadline* markets, where the earlier deadline's event is contained in the later one's, and the bot refuses a pair whose wording does not say so (see below). "The same question" means both the date-stripped title *and* the outcome label (the subtitle) match: a daily price family lists dozens of strikes under one shared title, and keying on the title alone paired the highest earlier strike against the lowest later one — two different questions traded as one. The later deadline gives more time for the event to occur, so the later contract's YES price normally sits above the earlier one's — and the gap between the two YES prices is the market's implied probability that the event first happens *between* the two deadlines. When the later contract is priced *higher* than the earlier one by at least a required margin, the bot disputes that in-between probability: it buys YES on the earlier contract and NO on the later one. There are exactly three ways such a pair can settle:

- The event happens by the earlier deadline (both resolve YES): the YES on the earlier contract pays out — **win**
- The event never happens by the later deadline (both resolve NO): the NO on the later contract pays out — **win**
- The event happens in between (earlier NO, later YES): both legs expire worthless — the full stake is lost — **loss**

The earlier contract resolving YES while the later resolves NO is impossible for a genuine cumulative-deadline pair ("by March" YES implies "by June" YES). Both legs must therefore be *cumulative-deadline* markets, and the bot now checks that rather than assuming it: a pair is only formed when both legs' wording states a "by \<date\>" deadline and the two deadlines differ — compared as normalized text, not parsed calendar dates, so one deadline spelled two ways reads as two, and a truncated spelling can read two deadlines as one (mostly fixed; a few residuals remain, see `CLAUDE.md`). Kalshi also lists **snapshot** markets — "what is X **on** \<date\>", "X **in** \<month\>" — whose probabilities do not nest (SOL ≥ $180 on Sep 14 does not imply SOL ≥ $180 on Sep 18), and stripping dates from titles collapsed such a family into a single time-series group, so it used to be eligible to trade. The backtester still treats a pair that settled earlier-YES/later-NO as a premise violation and excludes it with a counted warning rather than paying it — that counter is now a backstop behind the wording check, not the only thing watching. This trade is **not risk-free**: at market prices its expected value is zero minus fees, and it profits only if the market systematically overstates the in-between probability. The bot sizes it on the operator's estimate that 75% of the market-implied in-between probability is genuine (`TIME_SERIES_INTERVAL_PROB_DISCOUNT` in `config.py` — a hand-set estimate, not a measured quantity).

**Same-title pairs:** Two contracts on different event tickers but with the *identical* title and subtitle (i.e. asking exactly the same question), whose two markets close within an hour of each other (see below for why both conditions are needed). If their prices diverge by 5% or more, the bot buys NO on the expensive one and YES on the cheap one. Both contracts should co-resolve, so the trade is priced on a 95% co-resolution assumption — high, but **not risk-free**: the 5% that does not co-resolve is a total loss on one leg. The subtitle here is the outcome label that distinguishes markets sharing one question title (e.g. two candidate names under "Who will the next Pope be?"); the API stopped sending a `subtitle` field in 2026-08, so ingest now sources it from `yes_sub_title` — without that discriminator, two *different* outcomes would be paired as if they were the same contract.

**Time-series pairs** must also sit on **different event tickers**, with one exception: a same-event cumulative deadline **ladder**, where Kalshi lists one question's several deadlines as separate markets inside a single event ("Will SpaceX launch another Starship before Sep 23, 2026?" and "… by Oct 16, 2026?" are both `KXSPACEXSTARSHIP-14`). Two such rungs are the time-series premise itself, and the time-series finder will pair them when `TIME_SERIES_SAME_EVENT_LADDERS` in `config.py` is turned on — ordering the legs and choosing the price tier by the two **stated** deadlines rather than by the exchange close times, which a settled event gives every rung alike. It ships **off**: read the constant's comment first, which records both why ladders nest (the impossible earlier-YES/later-NO settlement never occurred in 1,821 archive ladder pairs, against 13% of cross-event ones) and what turning it on would deploy (98% of a $10,000 balance into six trades, at a market-implied expected value of −31% — which is the later leg's own bid-ask spread plus taker fees, not the strategy's disbelief in the market). **Both paths implement ladders**: the backtester forms the same pairs from a per-event sub-pass and orders and gaps them on the same stated deadlines, so a ladder-enabled backtest measures the strategy a ladder-enabled live run would trade — with one standing caveat, that the two paths keep one pair per group by different rules (the live finder ranks tradeable-first then largest price gap, the backtest by largest entry-time monthly ratio), so they can replay different rungs of the same ladder. The setting reaches the backtest log *and* the dashboard's own page header (`Primary spread band: … | same-event ladders: on / off / not recorded`, a recorded on/off followed by whether it matches `config.py`'s switch, printed above every section), so a ladder-enabled run's dashboard is no longer indistinguishable from a switch-off one. When a run's setting departs from `config.py`'s switch, the header and the log say the run does not replay the live rule. `backtest.py --same-event-ladders` / `--no-same-event-ladders` overrides the switch for one backtest run, which is how the calibration that gates the flip gets measured without flipping it — the live finder binds the constant at import, so a backtest override can never reach it. Everything below is unchanged by that switch.

The two event tickers must also belong to **different event series** (a same-event ladder's two rungs share one event and therefore one series; the series rule bites on a time-series pair only when the wording is identical across that series, which a dated ladder's is not). A Kalshi event ticker is a series prefix followed by an instance stamp, so two events of one series are two instances of one recurring fixture — two ball games, two 15-minute price windows, two multi-leg combos of different games. Multi-leg combos are the one exception to "the prefix is the series": Kalshi lists them under several prefixes that all begin `KXMVE` (`KXMVECROSSCATEGORY`, `KXMVESPORTSMULTIGAMEEXTENDED`, `KXMVECROSSCATEGORY0`, `KXMVENBASINGLEGAME`), and the whole family is treated as **one** series, so two combos under two *different* `KXMVE*` prefixes are still refused — a combo ticket's wording names its legs but never its date, so identical wording across any two combo tickets is two different tickets. Reading the literal prefix missed exactly those cross-prefix pairs: in one day's settled history they co-resolved only 68% of the time (an unconditional rate over every record in each cross-prefix wording class, with no price or liquidity filter — the measured reason the family rule exists, not a co-resolution estimate for pairs that would actually be traded). Identical wording across two events of one series is the same question asked about two *different* events, and the co-resolution assumption simply does not hold: one such pair of consecutive baseball game days was quoted 0.97 and 0.01 at the same moment. The time-series finder applies the same rule (when the wording is identical across one series, the deadline is not in the wording and there is no cumulative-deadline pair either), so the trade cannot reappear relabelled. A second, independent rule decides whether a time-series pair is a *deadline* pair at all: both legs' wording must state a cumulative "by \<date\>" deadline, and the two deadlines must differ. It reads the outcome label first, then the market title, then the event title, so a multi-choice market whose sub-contract is labelled "$80,000 by June 30" qualifies on that label alone — the rule turns on wording, never on a market's type. It is applied in the live scanner and the backtester through one shared helper for the same reason the series rule is: gating one path only would leave the other replaying the defect. Different series is a *necessary* condition, not a sufficient one: the same prop listed for two different games in two *different* leagues shares its wording too (`KXBRASILEIROB1HTOTAL-…` vs `KXUCL1HTOTAL-…`, both "Over 0.5 1H goals scored", settled no and yes on 2026-09-10), and so do a men's and a women's college basketball game between the same two schools (`KXNCAAMBGAME-…` vs `KXNCAAWBGAME-…`, identical title, subtitle and event title) or two competitions' fixtures of one matchup (Champions League and La Liga). So a same-title pair must also **close at the same moment**: its two markets' close times may be at most one hour apart (`SAME_TITLE_MAX_CLOSE_GAP_SECONDS` in `config.py`), and a close time that cannot be compared refuses the pair (DR-74). Two games close hours or days apart; one question listed by two series closes at one instant. On the 365-day backtest (start 2025-09-24), 17 of its 21 trades were such men's/women's pairs, and every same-title candidate closed either at the identical instant (245 pairs, 99.2% settled the same way) or at least 1.17 hours apart (1,661 pairs, ~60.7%); the rule leaves 3 trades, for +4.8% at a −1.9% max drawdown, against 21 trades, +4.7% and −48.0% before it. It is applied in the live scanner and the backtester through one shared helper (`scanner.closes_apart`). The time-series finder deliberately carries no twin: identical wording can never form a time-series pair (see the deadline rule above), so a pair the close rule refuses cannot reappear relabelled as one. Known residuals — for example, futures that can be decided early carry a family-wide placeholder close time on the live exchange, so two such listings pass live at a zero gap — are recorded in `CLAUDE.md`.

The required price gap for time-series pairs is tiered by how far apart the two deadlines are: 15% for deadlines ≤ 15 days apart, 30% for 16–30 days — wider gaps leave more room for the event to genuinely land between the two deadlines, so more of the market-implied in-between probability is real rather than mispricing, and a bigger gap must be demanded before disputing it. Deadlines more than 30 days apart are never considered. See `min_price_diff_for_gap()` in `config.py` for the exact thresholds.

In both cases, Kalshi charges a taker fee per contract leg. The bot only executes trades where the profit margin exceeds all fees after applying order book depth to confirm the gap exists in real liquidity. Depth is priced at the **margin**: a single pair can never consume more than `BUDGET_FRACTION` of the balance, so averaging a liquid market's entire book would price every trade against levels it could never reach. The scanner bounds its average at the most contracts the balance could ever buy, and the sizer then searches the book for the largest contract count whose own fill price still justifies it.

### Strategy change (2026-09)

Until September 2026 the time-series strategy traded the opposite way round: it fired when the *earlier* contract was priced higher than the later one, bought NO on the earlier and YES on the later, and described the result as a risk-free position with three profitable outcomes. That direction has been inverted, and the time-series bet is now a **directional trade, not an arbitrage**: the bot buys YES on the earlier contract and NO on the later one when the later is priced at least the required margin above the earlier, wins if the event happens by the earlier deadline or never happens by the later one, and loses the whole stake if it happens in between (see the three outcomes above). Same-title pairs are unchanged.

What that means for anyone reading the outputs:

- **Sizing.** The Kelly sizer models the probability of profit as `1 − k × (later YES price − earlier YES price)`, with `k = TIME_SERIES_INTERVAL_PROB_DISCOUNT = 0.75` in `config.py`: the bot believes 75% of the market-implied in-between probability ("prices converge by 25%"). At `k = 1` — taking the market at face value — or under an independence model, Kelly is zero or negative for every pair and the strategy never trades, so `k` is what makes it fire at all; it is a hand-set operator estimate. With `k = 0.75` the Kelly fraction sits well below the `BUDGET_FRACTION` cap (20%) for typical gaps (reading off an earlier contract at 0.30 on a tight book: about 5% at the minimum short-tier gap, about 16% at a 0.30 gap — the figure is strongly dependent on that earlier price, running to about 14% at the same tier with the earlier contract at 0.10), so Kelly itself sizes and differentiates pairs and the cap only binds for gaps of roughly 0.45 or more; a wide order book drives Kelly negative and the pair is skipped. Kelly's payoff-per-dollar term divides by the cash actually at risk — both legs' cost **plus** the taker fees, since a losing pair forfeits the fees too — so a positive Kelly fraction means positive expected value under the continuous fee approximation the gate prices with. That approximation sits slightly below the ceiling-rounded fee the trade is actually charged, so a pair right on the boundary can still be a few tenths of a cent EV-negative at single-digit contract counts; the separate "profit if won must be positive" check bounds that residual. Because a pricier later contract is normal term structure, far more time-series candidates qualify than before. `k` is no longer only a hand-set guess — it is now *measurable* against settled history: see [Backtest](#backtest) for `--interval-discount` and the dashboard's empirical-`k` report. The live bot still only ever reads the `config.py` constant; nothing writes the measured value back.
- **Trade log.** `trade_log.xlsx` keeps its 18 columns, but new workbooks head the count columns "x — A leg" / "y — B leg", the cost column "Total Cost incl. fees ($)" (the value is fee-inclusive in every workbook — only the header is new, and every row's Notes carries `fees=$x.xx` so a row under an old header is still readable) and the profit column "Profit if won ($)" (an existing workbook keeps its old header row) and every row's Notes cell is prefixed `[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=…]]` so the side traded on each market is explicit. For time-series rows the "nA (NO ask)" column is the earlier contract's best NO ask for reference only — the traded NO price is the `nB` in the Notes prefix. The dev-simulation candidates sheet gains an "nB (NO ask)" column, and the live pairs table logged by `main.py` gains an "nB (NO)" column and labels its profit column "Profit (win)". The Excel log's market cells now carry each leg's outcome label alongside the title, and the pairs table shows that label in its own "Outcome A"/"Outcome B" columns — its market cells are cut at 40 characters, so on any title that runs past that — which the daily families this exists for all do — the appended label is the first thing lost, and two strikes of one family share a title and would otherwise render identically.
- **Backtest.** Each `BacktestTrade` records `entry_nB`; a time-series trade's profit is negative only in the in-between outcome. Any candidate whose settlement violates the cumulative-deadline premise (earlier YES, later NO) is skipped rather than paid, and the run logs one warning with the count. Since the wording check landed, that counter is a backstop rather than the only signal — the backtester refuses a snapshot pair up front through the same helper the live scanner uses — so expect it at or near zero. DR-72 widened what a non-zero count is read as: most likely a wording false negative (a snapshot or "before \<date\>"-worded recurring window read as cumulative), but also possibly legs ordered on an early REALIZED close (cross-event pairs only — a same-event ladder is ordered on its stated deadlines), or strike-blind grouping on a cache without subtitles — the warning names all three rather than pointing at one. Each run also logs, and the dashboard renders, how its eligible markets are worded: how many are worded as a cumulative "by \<date\>" deadline, how many are snapshots, and how many carry no deadline wording the classifier recognises — reported as a verdict only when the three counts actually sum to the corpus, otherwise as raw counts with no claim attached. Existing backtest caches need no refresh.
- **Not updated.** `kalshi_bot_flowchart.pdf` predates this change (it shows the old `|pA − pB|` filter) and has not been regenerated; `BUG_SWEEP_FINDINGS.md` is a dated record and is left as-is.
- **Calibrating `k` from settled history (shipped 2026-09).** `backtest.py --interval-discount K` overrides `k` for one backtest run (the live bot is untouched — it never passes an override, so it always reads the `config.py` constant); the backtest also measures the *empirical* `k` — the realised in-between rate divided by the mean market-implied gap, pooled and per deadline-gap bucket — logs it, and reports it in the dashboard's new "Interval Discount (k) Calibration" section, which also lets you switch the equity curve between every `k` in the swept grid (`--no-sweep` to skip the extra re-simulation passes). The section after it breaks the same `k̂` down by Kalshi category, by tag or by spread band, following the page's filter bar. This is a recommendation only: nothing writes the measured value back to `config.py`. That same section now opens with the run's **outcome-label coverage** — how many of the run's eligible markets carried the `subtitle` both grouping keys are built from. It is shown on every run, so a clean run is confirmable rather than merely unwarned; below `config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION` it becomes a red banner beside the `k̂` card (and a one-line notice at the top of the page), because a cache written before the 2026-08-14 outcome-label ingest fix groups strike-blind and makes every figure on the page describe a different strategy from the shipped one. The remedy it names: delete `backtest_cache/archive_days/` and `backtest_cache/live_days/`, then re-run with `--no-cache` — `--no-cache` alone does not refresh the day slices. Fractional-contract sizing remains deferred. Note that since the cumulative-deadline wording rule shipped, the time-series leg is dormant by default (same-event deadline ladders, the one population that is not, ship behind an off switch — see above): it forms 0 pairs live, and a backtest's surviving time-series candidates are mostly recurring windows the entry rule can never enter, so a pooled `k̂` measured on a pre-rule cache should not be trusted (see `CLAUDE.md`'s cumulative-deadline gotcha).

---

## Architecture

### Module Dependency Graph

```
secrets.json + PEM key
        |
        ↓
config.py (constants), _http.py (retry + raw-response fetch)
  — both leaves: neither imports another project module —
        |
        ↓
    auth.py, scanner.py
        |
        ↓
    strategy.py
        |
        ↓
    reporter.py
        |
        ↓
    trader.py ──→ auth.py (also imports read_shard_balances directly)
        |
        ↓
    main.py  (orchestrates live trading pipeline)
    scheduler.py (weekly daemon → calls main.py as a subprocess)


    historical.py ──→ backtester.py ──→ dashboard.py
                            |
                    backtest.py (CLI entry)

    (historical.py also imports auth.py's build_client for its own client
     builders, _http.py directly for its raw signed GETs, and scanner.py's
     event_series so the event-title lookup budget's combo test agrees with
     the one-series rule; backtester.py
     also imports scanner.py's time_series_group_key — the one definition of
     the time-series grouping key, title plus outcome label, shared with the
     live scanner — event_series, the one definition of an event's series
     identity so the one-series rule can never disagree, plus
     deadline_profile/cumulative_deadline_pair, the one
     definition of what makes a pair a cumulative-deadline pair,
     deadline_pair_refusal and its REFUSED_SNAPSHOT/REFUSED_NO_STATED_DEADLINE/
     REFUSED_SAME_DEADLINE constants (the one definition of WHY a candidate is
     refused, DR-72), closes_apart (the one definition of the same-title
     close gate, DR-74) with close_gap_bound_text (the bound its refusal
     line prints), and leg_sides so settlement pays by the side each leg
     bought)

    v2_probe.py — standalone, human-run verification CLI; imports
    auth.py/config.py/_http.py/scanner.py/trader.py, imported by NOTHING
    in the pipeline
```

### Live Trading Data Flow

Step order below is `_run_prod`'s; `_run_dev` skips `get_held_tickers` entirely (no
sandbox positions) and so calls `fetch_shard_statuses` first instead.

```
main.py
  ├─ auth.build_client()           — authenticate with Kalshi API
  ├─ auth.verify_auth()            — read the per-shard balance breakdown (prod only; gate and size on the sum)
  ├─ scanner.get_held_tickers()    — fetch currently-held positions (prod only) so we skip re-entering them
  ├─ scanner.fetch_shard_statuses() — read GET /exchange/status per-shard trading/transfer flags (fail-soft)
  ├─ scanner.inactive_shard_indexes() — derive the trading-inactive shard set from the statuses above
  ├─ scanner.fetch_open_events_with_markets() — fetch open events + their markets from EVERY exchange shard, tagged (attaches event titles for MVE grouping; drops only markets on trading-inactive shards)
  ├─ main._log_shard_coverage()    — audit advertised shards vs ingested markets/funds (reports, never aborts)
  ├─ scanner.filter_markets_within_horizon() — optional --max-horizon-days cap (no-op if unset)
  ├─ scanner.find_time_series_pairs()   — time-series pair detection
  │                                        (both legs must be cumulative "by <date>"
  │                                         deadlines at two different dates, compared
  │                                         as normalized strings — see the note above)
  ├─ scanner.find_same_title_pairs()    — same-title pair detection
  │                                        (two different event series, and both
  │                                         markets closing within one hour of each
  │                                         other — scanner.closes_apart, DR-74)
  ├─ main._dedup_pairs()            — merge both lists, preferring same-title on overlap
  ├─ scanner.enrich_with_orderbook_prices() — validate depth; price each pair over the depth this balance could actually buy
  ├─ strategy.compute_trade()      — Kelly sizing per pair, at the marginal fill price of the size it settles on, and only over depth the fill-or-kill limit can actually reach
  ├─ strategy.select_portfolio()   — greedy portfolio selection
  ├─ trader.pre_execution_check()  — re-fetch order books, drop pairs whose prices moved or whose depth is no longer reachable at the limit about to be submitted
  ├─ trader.drop_legacy_unroutable() — legacy path only: drop specs with a leg off the default shard BEFORE any money moves
  ├─ trader.ensure_shard_collateral() — move funds onto the shards the selected legs settle against (prod; dry-run only plans)
  ├─ trader.execute_trades()       — submit fill-or-kill orders leg-by-leg to the V2 order endpoint, each leg routed to its own market's shard (parallel across pairs, rollback on partial fill)
  ├─ auth.verify_auth()            — re-read the post-trade balance for the Excel log (falls back to the pre-trade balance if this read fails)
  └─ reporter.append_to_prod_log() — write results to trade_log.xlsx
```

### Backtest Data Flow

```
backtest.py (CLI)
  ├─ historical.build_historical_client()    — prod API client for archives
  ├─ historical.build_prod_live_client()     — prod API client for recent data
  ├─ backtester.run_backtest_sweep()         — run_backtest() is the plain two-tuple wrapper other callers use
  │    ├─ _prepare_candidates()                   — band- AND k-independent; runs once no matter how many
  │    │    │                                        bands or k's are simulated
  │    │    ├─ historical.fetch_all_settled_markets() — market metadata, returned as a SettledCorpus
  │    │    │     │                                    (streams settled_markets_*.jsonl.gz on every walk;
  │    │    │     │                                    a legacy .json cache hit is still one list)
  │    │    │     ├─ prefilter=_can_ever_enter        — drop never-tradeable markets during assembly;
  │    │    │     │                                    the log reports the window's settled records
  │    │    │     │                                    beside the eligible ones it kept
  │    │    │     └─ assembly walk A / walk B         — both stream the day slices (unfiltered) and
  │    │    │                                          the current day (spooled to an anonymous temp
  │    │    │                                          file, prefiltered as it arrived) lazily
  │    │    │                                          through one merge generator: A counts — the
  │    │    │                                          kept markets, and the settled records the
  │    │    │                                          prefilter or the dedup removed — and
  │    │    │                                          collects event tickers for title resolution,
  │    │    │                                          B patches titles and writes the cache, the
  │    │    │                                          counts in its meta — the corpus is never held
  │    │    ├─ _index_eligible_keys()                 — walk 1: count, re-check the prefilter (a no-op on a
  │    │    │                                           fetched corpus, which says so), census, hash both
  │    │    │                                           grouping keys
  │    │    ├─ _materialize_groupable()               — walk 2: keep only eligible markets sharing a key with
  │    │    │                                           another (the rest form single-member groups, which both
  │    │    │                                           groupings drop) — then group and extract pairs on those
  │    │    └─ historical.fetch_candlesticks()        — hourly price series per ticker (parallel across tickers;
  │    │                                               each from the later of --start-date and the market's own
  │    │                                               open; a window over the 5,000-candle cap is paged)
  │    └─ _sweep_from_candidates()
  │         ├─ _entries_for_band()            — k-independent; one time-series _find_entry() pass per band:
  │         │                                    every band of SPREAD_BAND_SWEEP_FLOORS x CEILINGS (plus the
  │         │                                    primary if it is off-grid) by default — only the no-band pass
  │         │                                    scans every pair, the rest rescan the pairs that entered
  │         │                                    there — or the primary band alone with --no-band-sweep;
  │         │                                    plus one same-title pass
  │         ├─ _interval_calibration()        — empirical k_hat per band, from that band's k-independent entries
  │         └─ _simulate_at_discount()  — once per (band, k[, population]): Kelly gate, dedup, P&L from
  │                                        outcomes, _build_equity_curve() — the scenario explorer's
  │                                        band x k x {all, time_series, ladder, cross} grid, plus
  │                                        one same-title simulation, come from repeating this call,
  │                                        never from slicing a joint run
  └─ dashboard.generate_dashboard()          — write HTML report: k-selector section, a k̂ breakdown by
                                                category / tag / band (regrouped from each band's
                                                calibration observations), the scenario-explorer section
                                                (band x k heatmap, per-population KPIs), and a sticky
                                                filter bar whose every band x Kalshi category x tag view
                                                is computed here, packed into the page and swapped in by
                                                a small script
```

`run_backtest()` composes `_prepare_candidates()` with a single `_entries_for_band()` pass at the default (no-op) band (the two together are `_prepare_entries()`) and one `_simulate_at_discount()` call, so every pre-existing caller of the plain two-tuple entry point is unaffected by the band sweep above.

---

## Module Descriptions

| Module | Description |
|--------|-------------|
| `__init__.py` | Package initializer. No exports; marks the directory as the `kalshi_betting` package. |
| `config.py` | All tunable constants (price thresholds, Kelly cap, fee rates, API URLs, file paths), the two fee helper functions, `max_affordable_pairs()` — the single budget-to-contracts definition shared by the scanner's depth bound and the sizer — and the backtest-only time-series spread-band helpers `time_series_spread_band()` / `time_series_spread_too_wide()` plus the `SPREAD_BAND_SWEEP_FLOORS` / `SPREAD_BAND_SWEEP_CEILINGS` grid the scenario explorer sweeps; nothing on the live path reads a band value. |
| `auth.py` | Reads RSA credentials from `secrets.json` and the PEM key file, constructs an authenticated `KalshiClient`, and provides `verify_auth()` to confirm credentials and read the live account balance per exchange shard (`{exchange_index: cents}`; callers sum for sizing). |
| `_http.py` | Shared HTTP helpers used across the package (auth, scanner, historical, trader, and v2_probe): `api_call_with_retry()` (exponential backoff on 429/5xx for market-data calls) and `fetch_json_page()` (parses the SDK's raw `*_without_preload_content` responses, re-raising non-2xx as `ApiException`), and `signed_request_json()` (signed GET/POST against an arbitrary API path for routes the pinned SDK has no method for — retry-free, since order submission and the collateral transfer call it directly). |
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles and appends each market's outcome label (subtitle) to group time-series pairs, detects same-title pairs via exact match, refuses a same-title pair between two events of one series or whose two markets close more than an hour apart (`closes_apart()`, the one definition of that close gate, failing closed on a close time it cannot compare, with its own silent-at-zero refusal count whose printed bound comes from `close_gap_bound_text()` on both paths — DR-74), a time-series pair whose wording is identical across one series (two instances of one recurring fixture — a genuine two-deadline family of one series spells its deadline in the wording and can still pair, if both legs are worded as cumulative deadlines; `event_series()` reads the prefix before the first hyphen, except that every `KXMVE*` combo prefix collapses onto one family so two combos listed under two different combo series are still refused), a time-series pair between two markets of ONE event unless `TIME_SERIES_SAME_EVENT_LADDERS` is on and they are two dated rungs of that event's deadline ladder (`stated_deadline()` / `same_event_ladder()`, which order the legs and measure the gap on the STATED deadlines; `pair_gap_days()` is the one place anything downstream reads that gap), and a time-series pair whose legs are not both worded as cumulative "by \<date\>" deadlines at two different dates (compared as normalized text, not parsed calendar dates — see the note above; `deadline_phrasing()` — a snapshot family such as "Solana price on Sep 14/18, 2026?" still groups but no longer pairs; `deadline_pair_refusal()` names WHY a refused pair was refused — snapshot wording, no stated deadline, or the same deadline stated twice — feeding three separate, honestly-labelled skip counts instead of one folded one, DR-72; every one-series refusal, in both finders, and every same-title same-event skip is counted on its own silent-at-zero line too, M10), and enriches tradeable pairs with live order book depth to compute real fill prices — averaged over the contracts the balance could actually buy, not the whole book, with the qualifying levels kept on the pair (`depth_levels`) for the sizer to re-price against via `prefix_fill_prices()`. Also home to `leg_sides()` / `leg_prices()`, the single mapping from a pair's type to the side and price each leg actually trades, and to the V2 order-price grid arithmetic (`tick_size_for_price()`, `ceil_to_tick()`, `v2_limit_price()`, `v2_effective_cap()`) — it lives here, not in `trader.py`, so the sizer can test a candidate size against the very limit the trader will submit without importing it. |
| `strategy.py` | Solves size and price together — binary-searching the book for the largest contract count whose own marginal fill price still justifies it AND that the resulting fill-or-kill limit can actually buy — then applies the Kelly criterion to size each trade, computes the profit floor for same-title pairs / the win-scenario profit for time-series pairs and the monthly-normalized return, and greedily selects a portfolio that fits within the available balance. |
| `trader.py` | Converts `TradeSpec` objects into orders and submits each pair's two legs sequentially (fill-or-kill, NO leg then YES leg — the NO leg is `market_a` for a same-title pair and `market_b`, the later contract, for a time-series pair) via the Kalshi API, with automatic rollback of the filled NO leg if the YES leg doesn't fill. Multiple pairs execute concurrently. Submission goes to the V2 order endpoint by default and to the retained legacy endpoint when `config.ORDER_API_VERSION` is flipped — see "Order API version" below. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). Market cells are rendered by `scanner.display_title`, so they carry the event title and the outcome label alongside the market title. |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`. |
| `scheduler.py` | Long-running daemon that fires the production bot every Monday at 09:00 using the `schedule` library. Also prints the equivalent cron job command. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints, sharded into parallel per-day slices that are cached individually so interrupted or repeated fetches resume instead of re-walking months of history) and hourly candlestick price series needed by the backtester (candlesticks are fetched in parallel across tickers and cached per ticker, so workers never share a cache file and a repeat run re-reads them from disk). |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs (including, behind `TIME_SERIES_SAME_EVENT_LADDERS`, two dated rungs of one event's deadline ladder — formed by a separate, deliberately unwindowed per-event sub-pass and ordered and gapped on their stated deadlines through the same `scanner.stated_deadline()` / `same_event_ladder()` the live finder uses), scans weekly Monday snapshots for the first tradeable entry — at a BACKTEST-only time-series spread band (`_entries_for_band()`, `_find_entry()`) that live trading never reads — applies Kelly sizing, records actual P&L from settlement outcomes, and builds a daily equity curve that opens one day before the start date at the untouched initial balance, so a trade entering on the first day of the window shows its day-0 charges as a real daily return and a real drawdown. The curve is a portfolio value, not a cash balance: an open position is carried at its cost basis for its whole holding period, so committing capital does not move the curve and the drawdown/Sharpe/Sortino figures derived from it measure realized loss rather than peak deployment. The work is split at the band and the interval discount `k`: `_prepare_candidates()` (fetch through candlesticks) depends on neither, `_entries_for_band()` depends only on the band, and `_simulate_at_discount()` — the Kelly gate, dedup, P&L, equity curve — depends on `k`; `_sweep_from_candidates()` composes all three into the band x `k` x population scenario grid (`SweepPoint`, `HalfSplit`, `BacktestSweep`) the dashboard's scenario explorer renders. Pair extraction applies the live scanner's same-title close gate through the same `scanner.closes_apart()` (DR-74), and accounts for every pair of every group's members, each on its own silent-at-zero line — the three wording reasons, the one-series rule, the same-event skip, the same-title close gate (plus a backtest-only line for a same-title pair whose close time cannot be read), each ladder reason and, for time-series groups, the pairs its close-date window never visits and the members with no readable close time — and the size of each grouping is logged on every run, zero included, so a run that forms no pairs still logs why (M10). |
| `dashboard.py` | Generates a self-contained HTML performance report from backtest results — nine sections: cumulative return lines (total plus one per trade type) / Sharpe/Sortino/drawdown KPIs plus mean and median return per trade and the median monthly return, returns decomposition (P&L by Kalshi's official category and by category · tag, with a per-group table), price calibration analysis, an interval-discount (`k`) calibration section with a dropdown that switches the equity curve between every swept `k`, an empirical `k̂` breakdown (a bar chart and table of `k̂` by Kalshi category, by tag or by spread band, each bar naming its entries and distinct events, with a Group-by `<select>` of its own and a dashed line at the `k` the run was sized at), a scenario-explorer section (a fragility banner, a spread-band x `k` heatmap whose metric menu includes median per trade and each band's empirical `k̂`, a one-row-per-band `k̂` table, and a per-population KPI table over the band sweep, with its own band/`k` `<select>`s), trade diagnostics (best and worst five trades with each leg's side, price, close date and settlement), risk metrics, and an S&P 500 benchmark comparison whose download window opens on the same date as the equity curve's leading initial-balance row. The page header names the run's primary spread band and same-event-ladder setting. A sticky filter bar at the top — Spread band, Category, Tag — re-scopes every trade-derived section (performance, decomposition, calibration, diagnostics, risk, the benchmark's strategy row) to another spread band's own run at the primary `k`, and/or to one Kalshi category or category · tag (the series' first tag, so breakdowns partition) of that run, whose return, drawdown, Sharpe, Sortino, median monthly return and benchmark row are then its contribution — the starting balance plus its trades' P&L as the run booked them — not a standalone simulation. The header's trade count follows the selection too, and the `k̂` breakdown moves to the same band and selection — that chart regroups the band's own `k̂` population (every time-series candidate entry, measured before the Kelly gate), not the run's trades. Every view is computed in Python by the helpers the sections render with and shipped gzip-packed; the Interval Discount (k) and Scenario Explorer sections keep their own controls and are not filtered. The bar's selects start disabled and are enabled once the page has unpacked its data, and each chart is redrawn from its layout as first drawn, so a zoom never carries into another selection. If the filter's data cannot be built, the page is still written, without the bar and with a notice in its place. Every run overwrites the one `backtest_dashboard.html` in the repo root. |
| `backtest.py` | CLI entry point for the backtest pipeline. Parses arguments (including `--interval-discount`, `--no-sweep`, `--same-event-ladders` / `--no-same-event-ladders`, and the backtest-only `--spread-min` / `--spread-max` / `--no-band-sweep`), builds the historical API clients, calls `backtester.run_backtest_sweep()` then `dashboard.generate_dashboard()`, and logs a summary of the primary result. |
| `v2_probe.py` | Human-run CLI that verifies the V2 order path's NO-leg mapping, fill-or-kill kill semantics, and the inter-shard transfer's centicent unit against the production account for roughly one cent of exposure. Its closing reduce-only bid is priced at the top of the market's own grid (0.99 / 0.999 / 0.9999 by tick regime), not at the rollback builder's loss floor, so that floor can no longer cause a FAIL unrelated to the mapping (a book with no reachable resting YES ask still can); `reduce_only` is what bounds that bid. A 2xx order body that is not a JSON object, in either step that reads one (DR-58), and — in the NO-buy step only — an object whose fill counts are unreadable (DR-60), are a clean FAIL that still reads the position, re-reads it once when that first read is `None` or `0`, and reports lookup-failed, position-open and genuinely-flat as three distinct outcomes, never a traceback out of the fill readers. Two of the unfillable-ask step's branches are recorded residuals — its unreadable-fill-counts branch and its `not killed` branch both FAIL without re-reading the account. The NO-buy step classifies readable fill counts into three outcomes, not two — a complete fill, a true kill, and a fill-or-kill invariant violation — so a partial fill FAILs naming the counts and re-reading the account rather than being reported as a clean kill with the account "still flat" (DR-20); the unfillable-ask step always read a partial that way. Both steps judge on `fill_count` AND `remaining_count`, which is deliberately stricter than the live `trader._v2_fill_status`, whose contract is `fill_count` alone. `--dest-shard` equal to the source shard is refused with a NEUTRAL at the top of the transfer step, before any transfer I/O, so it can no longer POST a net-zero self-transfer and then report a false in-flight FAIL (DR-22). Never imported by the pipeline. |

### Order API version

`config.ORDER_API_VERSION` selects which Kalshi create-order endpoint `trader.py` submits through. The default `"v2"` posts to `/portfolio/events/orders`: a fill-or-kill **limit** order with a dollar-string price, a fixed-point contract count, a `bid`/`ask` side on the market's single YES book, and an explicit `exchange_index`. V2 has no "market" order type, so the limit price is itself the price protection — the scanned price rounded up onto the market's own tick grid plus `BUY_SLIPPAGE_TICKS` ticks, which is a cap the older integer-cent `buy_max_cost` field could not express once MVE/combo markets moved to sub-cent ticks. A price sitting exactly on the boundary between two tick bands belongs to both, and the **finest** of them wins: taking the first match instead made the allowance ten times coarser at a band's upper edge, loosening a cap that is a bid. Because that limit applies per contract while the scanned price is an average over several book levels, `strategy.py` checks a candidate size against it before committing — otherwise the order asks for depth priced above its own limit and the whole fill-or-kill is killed.

The arithmetic itself lives in `scanner.py` (`v2_limit_price()` builds the wire price, `v2_effective_cap()` states it in the leg's own side terms) and `trader.py` re-exports it. That is deliberate: `trader.py` imports both `scanner.py` and `strategy.py`, so neither can import it back, and a second copy of the formula is exactly how the size and the limit drift apart again.

Setting it to `"legacy"` restores the original `/portfolio/orders` path (`CreateOrderRequest`, `type="market"`, integer-cent `buy_max_cost` via `BUY_MAX_COST_SLIPPAGE_CENTS`), which is retained unmodified purely as an instant rollback if the V2 request/response mapping misbehaves. Both paths share the same leg ordering (the NO leg is always submitted first and is the leg that gets unwound — `trader._ordered_legs` decides which market that is for the pair type), rollback logic, and result-status vocabulary, and neither ever retries a submission.

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

The package to install is named explicitly (`[tool.setuptools.packages.find]` includes only `kalshi_betting*`). Otherwise the `backtest_cache/` directory a backtest leaves at the repo root is auto-discovered as a second package, and `pip install -e` refuses to build.

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

Optionally, place a separate sandbox private key at `kalshi_demo_private_key.pem`
in the same directory. `auth.py` uses it for dev-mode signing when present and
falls back to `kalshi_private_key.pem` when it's absent.

### Key files at a glance

```
<project root>/
  secrets.json                    ← API key IDs
  kalshi_private_key.pem          ← RSA private key for request signing
  kalshi_demo_private_key.pem     ← Optional sandbox RSA private key
  trade_log.xlsx                  ← Persistent production trade log (auto-created)
  kalshi_arb.log                  ← Live bot log file (auto-created)
  kalshi_scheduler.log            ← Weekly daemon's own log (auto-created; rotates 5 MB x 3)
  kalshi_backtest.log             ← Backtest log file (auto-created)
  scheduler_state.json            ← Scheduler's claimed-slot record (auto-created)
  backtest_dashboard.html         ← Backtest HTML dashboard (rewritten by every run)
  backtest_cache/                ← Disk cache for historical data
    series_categories.json        ← Kalshi's category + tags per series (dashboard breakdown; refreshed weekly)
    settled_markets_*.jsonl.gz    ← Assembled market corpus (gzipped JSON lines, streamed —
                                    never loaded whole), keyed by start date (and by
                                    eligibility-filter tag when the backtester filters
                                    during assembly, so subsets never mix with full lists,
                                    and by a trailing `_nomve` when INCLUDE_MVE_MARKETS is
                                    False, since that flag changes which markets are
                                    fetched at all — the default True keeps the unmarked
                                    name). Legacy `settled_markets_*.json` files of the
                                    same name stem are still read (whole) when no
                                    .jsonl.gz exists at all, are never written any more,
                                    and are deleted once a rebuild of the same name stem
                                    has written its .jsonl.gz
    event_titles_v2.json          ← Cross-run event-title accumulator (merged, not overwritten).
                                    Stores genuine answers only — a title, or "" for a
                                    lookup that failed — never a ticker the lookup cap
                                    deferred. A legacy event_titles.json (which also
                                    stored "" for every deferred ticker, millions per
                                    bulk window) is migrated into it once, keeping only
                                    its titled entries, and then deleted; one found
                                    beside it later is never read, and is named in a
                                    WARNING on every run until it is removed
    archive_days/                 ← Per-created-day archive slices (incremental/resumable)
    live_days/                    ← Per-settled-day recent-market slices (incremental/resumable)
    candlesticks/                 ← Per-ticker hourly price series
  kalshi_betting/                 ← Python package
```

---

## Run Commands

`main.py` and `scheduler.py` echo log output to the terminal as well as their log
file. `backtest.py` does not: it installs only a `RotatingFileHandler` on
`kalshi_backtest.log`, so a backtest run's progress is visible only by tailing
that file, not in the terminal that launched it.

### Live V2 order-mapping probe (~1 cent of real money)

```bash
python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step no-mapping|unfillable-ask|transfer] [--dest-shard N] [--yes]
```

Human-run verification of the V2 order path's NO-leg mapping (an `ask` must open a NO
position and a reduce-only `bid` must close it), fill-or-kill kill semantics, and the
inter-shard transfer's centicent unit — against the production account, for roughly one
cent of worst-case exposure. `--dest-shard N` picks the transfer step's destination and
must not name the source shard (0) — a self-transfer is refused with a NEUTRAL before any
POST. Never wired into the pipeline; run it before trusting the V2 path unsupervised.

What is being verified is a *side* mapping, not a price, so the closing bid is priced at
the top of the market's own grid rather than at the rollback builder's loss floor. That
floor is derived from the probe's placeholder entry and comes out at $0.62 on every
market — a bid structurally killed on any market whose YES ask is above 62c, i.e. a FAIL
unrelated to the mapping with a real position left open. `reduce_only` bounds the
top-of-grid bid to the 0.01 contracts just opened. For the same reason, a position that
reads exactly 0 straight after a reported full fill is re-read once, one second later,
before the probe judges the sign: an unmoved ledger is usually read-after-write lag, not
disproof.

A 2xx response body that is not a JSON object at all (`"accepted"`, `[]`, `123`, `true`,
`null`) is a FAIL, not a crash. Both fill readers call `.get()` on the body, so such a
response used to raise an uncaught `AttributeError` immediately after a real order had
been submitted — no position read, no warning, and a real position left open with nothing
to tell the operator it existed. Both submission steps now print a FAIL naming the body
type, read the position (re-reading once when the first read is flat *or* unreadable, so
"flat", "open" and "could not read" stay distinguishable), and say exactly what to flatten
in the Kalshi UI. The reduce-only close is deliberately *not* submitted: it rests on the
mapping the unreadable body proves nothing about, so the position is left for a human,
exactly as a disproven mapping leaves it.

### Dev dry-run (sandbox simulation, no real orders)

```bash
python3 -m kalshi_betting.main --mode dev --dry-run
```

Scans the Kalshi sandbox markets with a virtual $1,000 balance and writes a simulation Excel file.
Dev mode never submits real orders regardless of `--dry-run` — the flag is a no-op
here (`main.py` logs `"--dry-run is inert in dev mode"` if you pass it) and is
only meaningful in `--mode prod`; the heading above names the plain `--mode dev`
behavior, not something `--dry-run` changes.

Specify a different virtual balance:

```bash
python3 -m kalshi_betting.main --mode dev --sandbox-balance 5000
```

`--sandbox-balance` is the mirror image of `--dry-run`: meaningful only in dev,
and inert in prod, where sizing always uses the real per-shard account balance.
Passing it to `--mode prod` logs `"--sandbox-balance is inert in prod mode"`
rather than silently ignoring it — it is not a way to cap your exposure on a
live run, and someone using it as one would get full-size real orders. Use
`--dry-run` to avoid submitting.

### Production (real money)

```bash
python3 -m kalshi_betting.main --mode prod
```

Fetches the live account balance, scans real markets, submits fill-or-kill orders leg-by-leg, and appends results to `trade_log.xlsx`.

### Production dry-run (discover trades but don't submit)

```bash
python3 -m kalshi_betting.main --mode prod --dry-run
```

Uses the real account balance and real markets (read-only API calls only — no
orders are submitted), but still writes a simulated row per discovered trade
to `trade_log.xlsx` (status `"simulated"`), same as prod's real-order rows
just without a live fill. Confirmed live behavior — don't assume `--dry-run`
leaves `trade_log.xlsx` untouched.

### Limit how far out a contract's deadline can be

```bash
python3 -m kalshi_betting.main --mode dev --max-horizon-days 14
```

Optional in both dev and prod. Only markets closing within the given number of
days from the moment the bot runs are considered — applies to both time-series
and same-title pairs. Omit the flag (the default) to consider all otherwise-eligible
markets regardless of deadline.

When the flag is set, the run logs one `Horizon filter: kept N of M markets
closing on or before <cutoff>` line at INFO, so a run whose pair counts differ
from the previous one can be attributed to the horizon rather than to the
market. The cutoff is *now + N days* — a time of day, not a midnight boundary —
and is printed to the second. Nothing is logged when the flag is absent.

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
python3 -m kalshi_betting.backtest --interval-discount 0.60   # override k for this run only
python3 -m kalshi_betting.backtest --no-sweep   # skip the k-grid re-simulation (single-point dashboard selector; one k column in the scenario explorer)
python3 -m kalshi_betting.backtest --same-event-ladders     # force same-event deadline ladders ON for this run
python3 -m kalshi_betting.backtest --no-same-event-ladders  # force them OFF for this run
python3 -m kalshi_betting.backtest --spread-min 0.30 --spread-max 0.60   # primary scenario's time-series spread band (backtest only)
python3 -m kalshi_betting.backtest --no-band-sweep           # skip the spread-band grid; the dashboard's scenario explorer has nothing to show
```

`--start-date` should predate the Kalshi archive cutoff. Markets that settled
after the cutoff have no historical candlestick data, so a window starting after
it produces no trades regardless of how many pairs it finds. A run that fetches
the settled markets says so in three places: a WARNING when they are fetched, a
warning line at the end of the run's log summary, and a red banner under the
dashboard's "Period:" line. A cached re-run repeats the verdict "as of" the
cache's assembly, without re-reading the cutoff, but only for a cache assembled
since this was added: a legacy `settled_markets_*.json`, or a
`settled_markets_*.jsonl.gz` written before it (the 2026-09-17 one on disk),
recorded no cutoff, so its re-run says the cutoff was "not recorded" and shows
no verdict at all. Run once with `--no-cache` to re-check the cutoff and stamp
it. If a cached run nevertheless enters trades under a post-cutoff verdict, the
cutoff has moved since the cache was assembled, and the log and dashboard say
the verdict is stale instead of repeating it.

**Cached runs say what they cover.** The assembled market list
(`backtest_cache/settled_markets_*.jsonl.gz`, or a legacy `.json`) is a
snapshot: it holds no market that settled after it was assembled, while the
report's period always runs to today. A run that reuses it logs when it was
assembled (a legacy file's modification time) and how long ago, and the
dashboard shows the same under the "Period:" line. Pass `--no-cache` to extend
it, and expect it to cost close to a full fetch for any window that reaches back
before the archive cutoff: it re-assembles the whole corpus, reusing a stored day
slice only while it is still valid (an archive day slice goes stale whenever the
cutoff advances), and it also re-fetches every pair's candlesticks and
re-resolves event titles. A non-empty cache is never expired automatically. An
**empty** one is reused only while it is less than a day old
(`EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS`), and after that the next run with that
`--start-date` rebuilds it — a full assembly of the window, not a quick check.

`--interval-discount K` (`0 <= K <= 1`) overrides the time-series interval
discount `k` for this backtest run only — it never reaches live trading, which
always reads `config.TIME_SERIES_INTERVAL_PROB_DISCOUNT`. Omit it (the
default) to run at the configured value. `--no-sweep` skips the extra
re-simulation across `config.INTERVAL_DISCOUNT_SWEEP`; with the band sweep on
(the default), each band is simulated at the primary `k` only, so the scenario
explorer's heatmap has a single `k` column. The empirical-`k`
calibration measurement and its log/dashboard report are unaffected by either
flag — see the sizing bullet under [Strategy change (2026-09)](#strategy-change-2026-09)
and CLAUDE.md for the full mechanism.

`--same-event-ladders` / `--no-same-event-ladders` overrides
`config.TIME_SERIES_SAME_EVENT_LADDERS` for this backtest run only. Omit both
(the default) to run at the configured value. Unlike `--interval-discount`, it
changes *which pairs exist* rather than how they are priced, so it applies
identically to every swept `k` and a run with it on is **not comparable** to a
baseline taken without it. It never reaches the live finder, which binds that
constant at import.

**`--spread-min` / `--spread-max` / `--no-band-sweep` — the scenario explorer
(backtest only; live trading reads no band value at all).** By default the
time-series finder's own directional price filter is the only gate on a
pair's YES-price gap (`pB − pA`): a floor tiered on the deadline gap, and no
ceiling. `--spread-min X` / `--spread-max Y` (each in `[0, 1]`) layer a
BACKTEST-only spread *band* on top of that for the **primary** scenario —
the floor is `max(deadline-gap tier, X)` and the ceiling is `Y`. The raised
floor also tightens the leg-price-sum ceiling to `1 − max(tier, X)`, exactly
as the live tier sets `1 − tier`, so a band is stricter than a YES-gap floor
alone. Either flag may be given alone, and omitting both runs the configured
default band (`config.BACKTEST_DEFAULT_SPREAD_BAND`, `(0.0, 1.0)` — no band,
i.e. exactly the live rule). The resolved floor must be strictly below the
ceiling (a violation is rejected before anything is logged), and a ceiling at
or below a deadline-gap tier logs a WARNING that it empties that tier. The
band sweep is **on by default**, so a default run also re-simulates every
band of `config.SPREAD_BAND_SWEEP_FLOORS` x `SPREAD_BAND_SWEEP_CEILINGS` (36
bands) crossed with every swept `k` (13 values) — 468 scenarios, plus a
standalone time-series (ladders and cross-event together), same-event-ladder
and cross-event simulation per non-empty cell, a split-half out-of-sample
check, and, for every cell that traded at least one event, a re-simulation
excluding the single most concentrated event — computed over the *same*
fetch, pair extraction and candlestick set as the primary scenario (only the
entry-detection pass and the simulation are repeated per band, and only the
no-band band's entry pass scans every pair: every other band rescans just
the pairs that entered there, which cannot change its entries because a band
only ever tightens the entry tests). An "All" cell is every entry at its
band, same-title included (it is the run's actual result); a "Time-series"
cell is every time-series entry, same-title excluded — and it is the one the
explorer's heatmap, banner and equity curve show, because the band and `k`
act on time-series pairs only and a same-title result would dilute the
comparison. "Ladders" and "Cross-event" split the time-series entries
further, and "Same-title" is simulated once, independent of band and `k`;
every population is labelled on the page. `--no-band-sweep` skips that
grid: the primary scenario still runs, but the dashboard's "Scenario
Explorer" section has no scenarios to show. The explorer's fragility banner
reports how many band x `k` cells were computed, what share of the
time-series cells had a positive return, and the split-half rank correlation
of their returns — the point being that the best of many correlated cells
overstates what you should expect, so read any cell with that banner on
screen rather than from one flattering cell (the banner's "best of N"
counts the cells that have a time-series entry, not the whole grid). A
split-half half with no entries reads "—" rather than a 0% return and is
left out of that correlation. One split date serves every cell — the median
entry date of the primary band's time-series entries (the lower of the two
middle dates on an even count) — so a half is empty
either when at least half of those entries share the first entry date (the
backtest log warns when that happens) or, at any other band or on the "All"
population, when all of that cell's own entries fall on one side of it.

Kalshi serves at most 5,000 hourly candles (~208 days) per candlestick
request and rejects a longer one with HTTP 400, so a ticker's price series —
requested from the later of `--start-date` and the market's own open — is
fetched in as many requests as its window needs and merged; a long window no
longer loses the markets that close late in it. (Until 2026-09-23 it did:
each window was one request opened at `--start-date`, so on the default
`--start-date 2024-01-01` every ticker closing more than ~207 days after the
start got no candles and could never enter.) One caveat: **Picking a
scenario changes nothing the live bot does:** live
trading has no band setting at all, and `k` and the ladder switch stay
whatever `config.py` says. Applying a chosen band and `k` live is a separate
change that must add live enforcement first. This is entirely a backtest
reporting feature: `min_price_diff_for_gap()` gains the `spread_min` keyword
the band's floor is layered through, but no live caller (`scanner.py`,
`strategy.py`, `trader.py`, `main.py`) ever passes it or reads a band
constant — see `CLAUDE.md`'s `test_ast_live_path_reads_no_band`.

**Feasibility pre-check (BS-11).** Before any network call, `_prepare_candidates()`
— the band- and `k`-independent preparation step `run_backtest()` and
`run_backtest_sweep()` both build on —
checks whether the `[--start-date, today]` window contains at least one
Monday-09:00-UTC entry checkpoint (the only time the replay ever enters a
trade). If not, it logs a warning and the run returns the same empty result the
zero-trade path already produces — instead of spending minutes fetching
millions of settled-market records into a cache that was always going to
produce zero trades. `historical.py` separately warns (without aborting) when
`--start-date` is on or after the archive cutoff, since that also makes the
window structurally 0-trade. The warning also appears on the dashboard and, for
a cache assembled since this was added, on a cached re-run "as of" the cache's
assembly — see the paragraph above for which caches record no cutoff.

`--max-horizon-days` only enters trades where the later-closing leg is within the
given number of days of the *simulated* entry checkpoint (each Monday evaluated
during the replay), not today's real date. Optional; omit for no limit.

The settled-market fetch is sharded into one slice per UTC day and fetched with
`SETTLED_FETCH_MAX_WORKERS` (default 8) parallel workers; each worker writes its
own completed slice to `backtest_cache/archive_days/` / `backtest_cache/live_days/`
and then releases it, so the day slices cost no memory however many days the
run spans. The current (partial) UTC day, which is never saved as a slice, is
filtered as it arrives and written to a private temporary file that disappears
with the run. The market records are never assembled in memory either: once
every slice is present, the slices are streamed off disk twice (once to count
the markets and collect the event tickers whose titles are resolved, once to
write every market into the assembled `settled_markets_*.jsonl.gz` cache), and
the backtester then streams that file on each of its own walks. The first walk
also counts every record that settled in the window and how many of them the
backtester's eligibility prefilter rejected, so the log reads "N eligible
markets of M records settled ..." instead of calling the survivors settled
markets; those counts are stored in the assembled cache, repeated when a later
run is served from it, and quoted on the backtester's own "Markets to analyze"
line, whose prefilter re-check then reports the zero it finds as expected
rather than as "skipping 0". What still
grows with the run is much smaller, because it holds strings rather than whole
records: the set of market tickers each of those two walks keeps to drop
duplicates, and the event tickers and titles being resolved. Three record lists
also remain: the archive tail (capped by `ARCHIVE_TAIL_MAX_RECORDS`), a
sequential fallback's whole result if the sharded fetch ever falls back to one,
and a legacy `settled_markets_*.json` cache, which is read whole when it is
served. An interrupted fetch resumes at day granularity, and `--no-cache`
reuses the day slices (they cannot go stale — see CLAUDE.md), so a refresh only
fetches the current day plus any days not yet on disk. If a day slice disappears or is damaged while it is being streamed, the
run stops with an error naming the day rather than continuing with a short
corpus; re-running refetches that day.

**One-time cache rebuild (BS-02).** Assembled `settled_markets_*.json` files
written before the archive stop rule was fixed can be missing *long-lived*
markets — ones created before `--start-date` that settled inside the window.
The archive is ordered by creation time, and the old walk stopped too early to
reach them. Run the backtest once with `--no-cache` to rebuild those assembled
files (the rebuild is written in the streamed `.jsonl.gz` format and, once it
is written, deletes the old `.json` of the same name, just as a rebuild used to
overwrite it); the per-day slice files under
`archive_days/` and `live_days/` are unaffected and are reused, so the rebuild
re-pays only the tail walk. That tail
is *not* free: it is never slice-cached, so it is a sequential, one-page-at-a-
time walk down created-time history that is re-paid on **every** run, rebuild or
not. It stops after `ARCHIVE_MAX_BARREN_PAGES` (50) consecutive pages with no
in-window settlement, and — because a single long-dated settler resets that
counter — is hard-capped at `ARCHIVE_TAIL_MAX_PAGES` (2000) pages total, which
logs a WARNING when hit (markets created deeper than that may be missed; raise
the constant if a run needs them).

**Prefilter tag bump (P5): delete the old assembled caches by hand.** The
backtester's eligibility prefilter now drops a market that opened at or after
every Monday-09:00-UTC checkpoint it could be entered at — those dated from
`--start-date` to the day before its close. It used to compare the opening
DATE, so it kept markets opened later on a checkpoint Monday, which that
Monday's checkpoint can never enter: 30% of the 2026-09-17 window's cache, 59%
of the 2026-07-13 one. For the same settled records, trades and returns are
unchanged and only the eligible-market counts shrink; a re-assembled corpus can
still differ from an old one, as any two assemblies made at different times do
(see the prefilter gotcha in `CLAUDE.md`). Because the predicate names the
assembled cache, its tag moved from `monday-eligibility-v1` to
`monday-checkpoint-v2`, so every
`backtest_cache/settled_markets_*_monday-eligibility-v1.*` file is never read
again and is not deleted by any rebuild (a rebuild only replaces a legacy file
of its own name). None of them held a usable result: the three that start
before the archive cutoff (2026-05-01, 05-28, 07-13) were assembled on
2026-08-03, before the subtitle fix, so they group strike-blind; the one for
2026-08-29 is empty and more than a day old, which the cache lookup already
treats as a miss; and the other four start after the cutoff, so they are
0-trade by construction. Take any before/after baseline you want from them with
a checkout from before this change, then delete them. The next run of each
start date re-assembles its corpus from the day slices, which are unaffected.
For a window that starts after the cutoff that is one extra assembly, plus any
live day not on disk (the 7-day 2026-09-17 window: 2,382 s fresh against
1,040 s from its cache). For one that starts before it, it is close to a full fetch: on
2026-09-24 none of the archive day slices such a window needs was valid under
the current cutoff, and 35 of its 61 past live days were not on disk. That is
the rebuild the strike-blind caches needed anyway. It is not a complete fix
for them, though: the live slices for 2026-07-25..08-02, 08-29 and 08-30 were
also written before the subtitle fix and are reused unless you delete them
(see the subtitle-drift gotcha in `CLAUDE.md`).

Candlesticks are then fetched with `CANDLESTICK_FETCH_MAX_WORKERS` (default 8)
parallel workers, one independent request per ticker. Cache files are keyed per
ticker, so workers never contend for a path and any ticker already on disk is
skipped. Fetched sequentially this step ran at roughly 4.3 tickers/sec, which made
it the single largest cost of a backtest.

Note that `--start-date` must fall before the Kalshi archive cutoff: candlestick
history only exists for archived markets, so a window that starts after the
cutoff has no price data for any of its tickers and produces no trades.

### Weekly scheduler daemon

```bash
python3 -m kalshi_betting.scheduler
```

Runs the production bot every Monday at 09:00 (local time) in a blocking loop, each run spawned as a `python3 -m kalshi_betting.main --mode prod` subprocess and killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` (3600s). The log also prints the equivalent `crontab` entry if you prefer cron.

The daemon logs to its own `kalshi_scheduler.log` (and the console), deliberately separate from `kalshi_arb.log`, which the spawned run writes and rotates — a second process holding an open handle on a rotated file would keep writing into the renamed backup.

**Slot record and startup catch-up.** Each run claims its Monday-09:00 slot in `scheduler_state.json` (repo root) *before* spawning the subprocess and finalizes the record — `finished_at`, `exit_code` — on every exit path, including timeout and spawn failure. On startup, the daemon compares the most recent Monday-09:00 slot against that record: if the slot has **no** recorded attempt (daemon not running when it came around — never started, crashed, host rebooted, mid-deploy), it runs a catch-up job immediately rather than waiting up to a week for the next Monday. A slot whose recorded attempt merely *failed* is not retried; only a slot with no attempt at all triggers catch-up. A hand-edited or partially-written state file with a non-integer `retries` or `exit_code` degrades to an unknown, typed reading (`0` / `None`) with a WARNING. For `retries` that is a real crash fix: a string or `null` there used to raise `TypeError` out of the daemon at startup, before the weekly job was ever registered, silently stopping the bot from trading at all. For `exit_code` it is typing hygiene only — a non-integer value already compared unequal to `30` and so already left the slot unretried; it is now named in a WARNING instead of failing silently.

**Blind-run retry.** There is one exception to "a failed attempt satisfies the slot": a run that exits `30` (`EXIT_NO_TRADEABLE_SHARDS`) scanned **nothing at all** — every advertised exchange shard trading-inactive (a Kalshi maintenance window overlapping the 09:00 fire), or an ingest that came back empty for a cause `/exchange/status` could not name. The scheduler sees only the exit code, so its own WARNING/ERROR name both possibilities; the run's own `kalshi_arb.log` carries the line saying which one fired. Since the bot only trades on that weekly fire, such a slot used to cost the entire week while both logs reported success. `run_job` now registers a one-shot retry `SCHEDULER_BLIND_RETRY_SECONDS` (3600s) later, at most `SCHEDULER_BLIND_MAX_RETRIES` (4) times per slot — hourly × 4 covers a typical outage while keeping the scan near its intended Monday morning — and the startup catch-up check re-runs a slot whose recorded attempt exited `30`. The attempt count is persisted as an optional `retries` key in `scheduler_state.json`; a state file written before this feature has no such key and counts as 0. Once the cap is reached the daemon logs an ERROR and gives up on that slot rather than retrying through a multi-day outage.

**Registration guard.** The `schedule` library reschedules a job only *after* its function **returns**, so an exception escaping a job leaves its next fire time in the past. The daemon's poll loop catches the exception and survives, but the job is still overdue — so it is re-entered on the very next 60-second poll tick, which would turn the weekly production run into a once-a-minute one for as long as the fault lasts. Every job is therefore registered through `scheduler._guarded_job`, which catches at that boundary, logs the traceback once, and lets the library reschedule normally. On a healthy host nothing changes; on a fault the daemon abandons that slot and waits for the next scheduled fire. The blind retry additionally passes `on_error=schedule.CancelJob`, so a retry that *raises* still deregisters itself instead of becoming a recurring job that never advances the retry cap — which also means a raising retry **ends the retry chain for that slot**: the slot is then recoverable only by the startup catch-up check on a daemon restart, never while the daemon keeps polling. That is the deliberate trade: the alternative is a real production trading run every 60 seconds for as long as the fault lasts.

> ⚠️ **The first daemon start after this upgrade immediately runs a live production trade.** `scheduler_state.json` does not exist yet, so the startup check sees no record for the most recent Monday slot and fires a real `--mode prod` run right away — not at the next Monday 09:00. The same applies to **any** restart after a Monday 09:00 slot passed while the daemon was down. Start the daemon only when you are prepared for it to trade immediately; if you are not, run `python3 -m kalshi_betting.main --mode prod --dry-run` first to confirm what it would do. That pre-flight exits `0` on a normal exchange and `30` if it could not scan anything (halt, or an empty ingest) — a `30` from the pre-flight means the check itself saw nothing, not that the bot found no edge.

**Process exit codes.** `main.py`'s exit code is the only signal the scheduler has for what happened inside a run:

| Code | Meaning |
|------|---------|
| `0`  | `EXIT_OK` — run completed (including a clean run that found no qualifying pairs) |
| `10` | `EXIT_SKIPPED_LOW_BALANCE` — run skipped because the account balance was below the minimum |
| `20` | `EXIT_TRADES_NEED_ATTENTION` — at least one trade came back `rollback_failed` or `manual_review`; **a human must check the account and trade log** |
| `30` | `EXIT_NO_TRADEABLE_SHARDS` — **nothing was scanned**: either every advertised exchange shard reported `trading_active=false` (an exchange-wide halt or maintenance window), so ingest dropped every market, or ingest produced **zero markets** for a reason `/exchange/status` could not name (`fetch_shard_statuses` is fail-soft and returns `None` on any internal failure, so the all-halted test is unevaluable exactly when something went wrong). Deliberately distinct from `0`'s "scanned everything, found no edge" |
| `1`  | Unhandled exception — the interpreter's default for a crash; not part of the contract above |

The constants live in `config.py` (`EXIT_OK` / `EXIT_SKIPPED_LOW_BALANCE` / `EXIT_TRADES_NEED_ATTENTION` / `EXIT_NO_TRADEABLE_SHARDS`) and the scheduler maps each to a distinct log level and message, so a low-balance skip or a manual-review run is never logged as "completed successfully". Exit `30` additionally means the weekly slot was **not** satisfied: nothing was scanned, so the run must never count as the week's scan — see the blind-run retry above.

---

## Testing

```bash
python3 -m pytest tests/ -v      # run the test suite
python3 -m ruff check kalshi_betting/   # lint check
```

Tests run fully offline against `unittest.mock.MagicMock` clients — no real Kalshi API calls. `.github/workflows/ci.yml` runs both commands on every push to `main` and on EVERY pull request, whatever its base branch — the `pull_request:` trigger carries no branch filter. Both must pass before merging.

---

## Sandbox Note

The Kalshi sandbox endpoint (`https://demo-api.kalshi.co`) requires a **completely separate account** registered at [demo.kalshi.co](https://demo.kalshi.co). Your production API key will return `401 Unauthorized` on the sandbox endpoint — this is intentional by Kalshi.

To use dev mode with real sandbox authentication, register a sandbox account, generate its API key, and add it as `"dev_api_key"` in `secrets.json`. Without a sandbox key, dev mode still fetches real sandbox market data (for scanning) but skips the held-positions and balance checks that require authentication.
