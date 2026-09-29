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

**Time-series pairs** must also sit on **different event tickers**, with one exception: a same-event cumulative deadline **ladder**, where Kalshi lists one question's several deadlines as separate markets inside a single event ("Will SpaceX launch another Starship before Sep 23, 2026?" and "… by Oct 16, 2026?" are both `KXSPACEXSTARSHIP-14`). Two such rungs are the time-series premise itself, and the time-series finder pairs them while `TIME_SERIES_SAME_EVENT_LADDERS` in `config.py` is on, as it is by operator decision of 2026-09-26 — ordering the legs and choosing the price tier by the two **stated** deadlines rather than by the exchange close times, which a settled event gives every rung alike. It shipped **off** and was turned **on** by operator decision on 2026-09-26: read the constant's comment, which records why ladders nest (the impossible earlier-YES/later-NO settlement never occurred in 1,821 archive ladder pairs, against 13% of cross-event ones), what it deployed on the 2026-09-22 snapshot at the live toggles then shipped (98% of a $10,000 balance into six trades, at a market-implied expected value of −31% — which is the later leg's own bid-ask spread plus taker fees, not the strategy's disbelief in the market), the decision rule it shipped behind and its unmet result (k̂ 0.87 in the widest-spread band, where the rule asked for materially below 1; live sizing assumed k = 0.75 then, 0.80 since the operator decision of 2026-09-27), and the 365-day ladder backtest (+111.8% at k 0.75, resting on one trade: the run re-simulated without its event returns −92.8%). **Both paths implement ladders**: the backtester forms the same pairs from a per-event sub-pass and orders and gaps them on the same stated deadlines, so a ladder-enabled backtest measures the strategy a ladder-enabled live run would trade — with one standing caveat: both take at most one open time-series trade per ladder but pick the rung differently (each live run keeps one pair per group, ranked tradeable-first then by largest price gap before sizing; the backtest takes, Monday by Monday, the Kelly-passing pair with the largest entry-time monthly ratio whose ladder is free), so they can replay different rungs of the same ladder. The setting reaches the backtest log *and* the dashboard's own page header (`Primary spread band: … | same-event ladders: on / off / not recorded`, a recorded on/off followed by whether it matches `config.py`'s switch, printed above every section), so a ladder-enabled run's dashboard is no longer indistinguishable from a switch-off one. When a run's setting departs from `config.py`'s switch, the header and the log say the run does not replay the live rule. A second header line, `Live rule (saved live defaults): …`, names the LIVE time-series rule itself — the saved live defaults' tier floors and spread band, plus their category/tag filter when one is set, read once before the run's fetch (never `main.py`'s per-run overrides, which a backtest cannot see; with no usable live defaults saved it reads `Live rule: none recorded — …`) and whether the primary scenario already is it, the filter bar can show it (naming the bar's own Spread band, Tier floors and, for a filter, Category or Tag options that select it — the bar shows one Category or Tag at a time, so a filter naming several is read one slice at a time), or it sits off this run's own grid — plus the same-title cap: at every size cap the page offers, same-title rows are capped at the lower of that cap and `config.SAME_TITLE_SIZE_CAP`, and the line gives the figure at the run's own cap. When the saved defaults' `k` or caps differ from the ones this backtest sized at (`config.py`'s), the line says so and, where the page shows the live rule, names the filter bar's `k` and Size cap options that show it at that sizing, or what the bar cannot show. The backtest log states the same verdict on its `Live time-series rule (saved live defaults): …` line. `backtest.py --same-event-ladders` / `--no-same-event-ladders` overrides the switch for one backtest run — `--no-same-event-ladders` replays the rule as it stood before the switch was turned on — and the live finder binds the constant at import, so a backtest override can never reach it. The pairing rules below are unchanged by that switch; the backtest figures quoted with them were measured with it off.

The two event tickers must also belong to **different event series** (a same-event ladder's two rungs share one event and therefore one series; the series rule bites on a time-series pair only when the wording is identical across that series, which a dated ladder's is not). A Kalshi event ticker is a series prefix followed by an instance stamp, so two events of one series are two instances of one recurring fixture — two ball games, two 15-minute price windows, two multi-leg combos of different games. Multi-leg combos are the one exception to "the prefix is the series": Kalshi lists them under several prefixes that all begin `KXMVE` (`KXMVECROSSCATEGORY`, `KXMVESPORTSMULTIGAMEEXTENDED`, `KXMVECROSSCATEGORY0`, `KXMVENBASINGLEGAME`), and the whole family is treated as **one** series, so two combos under two *different* `KXMVE*` prefixes are still refused — a combo ticket's wording names its legs but never its date, so identical wording across any two combo tickets is two different tickets. Reading the literal prefix missed exactly those cross-prefix pairs: in one day's settled history they co-resolved only 68% of the time (an unconditional rate over every record in each cross-prefix wording class, with no price or liquidity filter — the measured reason the family rule exists, not a co-resolution estimate for pairs that would actually be traded). Identical wording across two events of one series is the same question asked about two *different* events, and the co-resolution assumption simply does not hold: one such pair of consecutive baseball game days was quoted 0.97 and 0.01 at the same moment. The time-series finder applies the same rule (when the wording is identical across one series, the deadline is not in the wording and there is no cumulative-deadline pair either), so the trade cannot reappear relabelled. A second, independent rule decides whether a time-series pair is a *deadline* pair at all: both legs' wording must state a cumulative "by \<date\>" deadline, and the two deadlines must differ. It reads the outcome label first, then the market title, then the event title, so a multi-choice market whose sub-contract is labelled "$80,000 by June 30" qualifies on that label alone — the rule turns on wording, never on a market's type. It is applied in the live scanner and the backtester through one shared helper for the same reason the series rule is: gating one path only would leave the other replaying the defect. Different series is a *necessary* condition, not a sufficient one: the same prop listed for two different games in two *different* leagues shares its wording too (`KXBRASILEIROB1HTOTAL-…` vs `KXUCL1HTOTAL-…`, both "Over 0.5 1H goals scored", settled no and yes on 2026-09-10), and so do a men's and a women's college basketball game between the same two schools (`KXNCAAMBGAME-…` vs `KXNCAAWBGAME-…`, identical title, subtitle and event title) or two competitions' fixtures of one matchup (Champions League and La Liga). So a same-title pair must also **close at the same moment**: its two markets' close times may be at most one hour apart (`SAME_TITLE_MAX_CLOSE_GAP_SECONDS` in `config.py`), and a close time that cannot be compared refuses the pair (DR-74). Two games close hours or days apart; one question listed by two series closes at one instant. On the 365-day backtest (start 2025-09-24, with same-event ladders off — the switch's value until 2026-09-26), 17 of its 21 trades were such men's/women's pairs, and every same-title candidate closed either at the identical instant (245 pairs, 99.2% settled the same way) or at least 1.17 hours apart (1,661 pairs, ~60.7%); the rule leaves 3 trades, for +4.8% at a −1.9% max drawdown, against 21 trades, +4.7% and −48.0% before it. It is applied in the live scanner and the backtester through one shared helper (`scanner.closes_apart`). The time-series finder deliberately carries no twin: identical wording can never form a time-series pair (see the deadline rule above), so a pair the close rule refuses cannot reappear relabelled as one. Known residuals — for example, futures that can be decided early carry a family-wide placeholder close time on the live exchange, so two such listings pass live at a zero gap — are recorded in `CLAUDE.md`.

**At most one open time-series trade per ladder.** A *ladder* here is one question asked at several deadlines: two markets are on one ladder when they share an event ticker, or ask the same question once the dates are removed (the time-series grouping key — event title, market title and outcome label — at any deadline, across events, compared without letter case, articles, quote marks or stray punctuation, so one question listed twice with slightly different wording is still one ladder). The weekly live run already skipped the markets it holds, but nothing stopped a later week opening a *different* pair on the *same* ladder while the first trade was still open, and a week-by-week replay of the live rules found exactly that: 18 trades entered while their group was still open, and $3,990 lost when stacked rungs of two ladders lost together — the second trade's in-between window sat inside the first's, so one outcome sank both. A production run now first finds the ladders its open positions are on — from the run's own market list, or by looking the market up when the list lacks it (one that has closed but not yet paid out, say) — and the time-series finder refuses any candidate with a market on one of them, before it chooses each group's best pair, so a runner-up off those ladders can still trade. The portfolio step then takes at most one time-series trade per ladder, counting the ladders of every trade picked earlier in the same run, same-title ones included: a same-title pair is never refused by this rule, but its markets' ladders count. A refused pair spends no cash, so a lower-ranked pair of either kind may then fit in its place — which can in turn leave too little cash for a later same-title pair. If a held market cannot be identified, the run makes no time-series trade at all (an ERROR says so, and so does the closing "no qualifying pairs" line when nothing else is found) while same-title trades go ahead, and the run exits with code `40` so the scheduler's own log says time-series trading was skipped (see the exit codes under [Weekly scheduler daemon](#weekly-scheduler-daemon)). A ladder is free again once the account no longer holds a position on it. Dev runs hold nothing, so only the within-run rule applies there. The backtester applies the same rule as it walks the simulated Mondays — it never sells, so on each Monday it enters a time-series pair only if no trade it still holds, of either kind, has a market on that pair's ladders, and a ladder frees up on the day its market pays out. A time-series pair it skips on one Monday (its ladder busy, or too little cash) is tried again on its next Monday that passes the Kelly gate, as the weekly live run would; same-title pairs keep their one-pair-per-group rule and are not tried again. Every backtest time-series figure recorded before this rule is therefore not comparable (DR-76 in `CLAUDE.md`).

The required price gap for time-series pairs is tiered by how far apart the two deadlines are: 15% for deadlines ≤ 15 days apart, 30% for 16–30 days — wider gaps leave more room for the event to genuinely land between the two deadlines, so more of the market-implied in-between probability is real rather than mispricing, and a bigger gap must be demanded before disputing it. Deadlines more than 30 days apart are never considered. See `min_price_diff_for_gap()` in `config.py` for the exact thresholds. Those tiers were the live default until the operator decision of 2026-09-27 and are not a fixed rule: the saved live defaults' `tier_floors` turns them off (the gap must then only be positive, and clear the spread band's floor), and their `spread_band` adds a band on the gap — a floor layered on the tier and a ceiling — which the scanner applies to the scanned prices, again to the fresh order book, and once more just before submission, through one definition (`config.time_series_spread_refusal`). The seed saves the tiers **off** and the band **0–0.5**, the values `config.py`'s `TIME_SERIES_TIER_FLOORS` / `TIME_SERIES_SPREAD_BAND` have held since that decision: any positive gap up to 0.5 qualifies, at any deadline gap up to 30 days (see [Live trading toggles](#live-trading-toggles), and the 2026-09-27 decision record in `CLAUDE.md`, before a live run). A time-series pair whose later market has no YES ask on its book right now, or whose later book is crossed, is dropped rather than sized.

In both cases, Kalshi charges a taker fee per contract leg. The bot only executes trades where the profit margin exceeds all fees after applying order book depth to confirm the gap exists in real liquidity. Depth is priced at the **margin**: a single pair can never consume more than its per-trade cap of the balance (`BUDGET_FRACTION`, and for a same-title pair `SAME_TITLE_SIZE_CAP` too), nor, for a time-series pair, more than `1 − k` — 20% of the balance either way at the toggles shipped since 2026-09-27, when `BUDGET_FRACTION` is lifted to no cap — so averaging a liquid market's entire book would price every trade against levels it could never reach. The scanner bounds its average at the most contracts the balance could ever buy — the largest Kelly fraction the sizer can reach for that pair type under the run's settings (the same `k` and caps the sizer is handed), over the cheapest level — and first cuts the book at the first level with no edge left after fees (the pre-execution check cuts the freshly fetched book the same way before counting reachable depth); the sizer then searches the book for the largest contract count whose own fill price still justifies it.

### Strategy change (2026-09)

Until September 2026 the time-series strategy traded the opposite way round: it fired when the *earlier* contract was priced higher than the later one, bought NO on the earlier and YES on the later, and described the result as a risk-free position with three profitable outcomes. That direction has been inverted, and the time-series bet is now a **directional trade, not an arbitrage**: the bot buys YES on the earlier contract and NO on the later one when the later is priced at least the required margin above the earlier, wins if the event happens by the earlier deadline or never happens by the later one, and loses the whole stake if it happens in between (see the three outcomes above). Same-title pairs are unchanged.

What that means for anyone reading the outputs:

- **Sizing.** The Kelly sizer models the probability of profit as `1 − k × (later YES price − earlier YES price)`, with `k` read from the saved live defaults (the seed saves 0.80; `config.py`'s `TIME_SERIES_INTERVAL_PROB_DISCOUNT` holds 0.80 since the operator decision of 2026-09-27, 0.75 before it): the bot believes 80% of the market-implied in-between probability ("prices converge by 20%"; 75% and 25% at 0.75). At `k = 1` — taking the market at face value — or under an independence model, Kelly is zero or negative for every pair and the strategy never trades, so `k` is what makes it fire at all; it is a hand-set operator estimate. With `k = 0.75` — the value these figures were measured at, under the 20% `BUDGET_FRACTION` cap shipped until 2026-09-27 — the Kelly fraction sat well below that cap for typical gaps (reading off an earlier contract at 0.30 on a tight book: about 5% at the minimum short-tier gap, about 16% at a 0.30 gap — the figure is strongly dependent on that earlier price, running to about 14% at the same tier with the earlier contract at 0.10), so Kelly itself sized and differentiated pairs and the cap only bound for gaps of roughly 0.45 or more. Since that decision `config.py` ships no per-trade cap, and `1 − k` (0.20 at `k` 0.80) bounds every time-series trade instead (the seed adds a 10% per-trade cap); a wide order book drives Kelly negative and the pair is skipped. Kelly's payoff-per-dollar term divides by the cash actually at risk — both legs' cost **plus** the taker fees, since a losing pair forfeits the fees too — so a positive Kelly fraction means positive expected value under the continuous fee approximation the gate prices with. That approximation sits slightly below the ceiling-rounded fee the trade is actually charged, so a pair right on the boundary can still be a few tenths of a cent EV-negative at single-digit contract counts; the separate "profit if won must be positive" check bounds that residual. Because a pricier later contract is normal term structure, far more time-series candidates qualify than before. `k` is no longer only a hand-set guess — it is now *measurable* against settled history: see [Backtest](#backtest) for `--interval-discount` and the dashboard's empirical-`k` report. The live bot reads `k` from the saved live defaults (through the run's one `config.LiveSettings`, which hands the sizer and the scanner's depth bound the same `k`) unless `main.py`'s own `--interval-discount` overrides it for one run — see [Live trading toggles](#live-trading-toggles); nothing writes the measured value back on its own.
- **Trade log.** `trade_log.xlsx` keeps its 18 columns, but new workbooks head the count columns "x — A leg" / "y — B leg", the cost column "Total Cost incl. fees ($)" (the value is fee-inclusive in every workbook — only the header is new, and every row's Notes carries `fees=$x.xx` so a row under an old header is still readable) and the profit column "Profit if won ($)" (an existing workbook keeps its old header row) and every row's Notes cell is prefixed `[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=…]]` so the side traded on each market is explicit. For time-series rows the "nA (NO ask)" column is the earlier contract's best NO ask for reference only — the traded NO price is the `nB` in the Notes prefix. The dev-simulation candidates sheet gains an "nB (NO ask)" column, and the live pairs table logged by `main.py` gains an "nB (NO)" column and labels its profit column "Profit (win)". The Excel log's market cells now carry each leg's outcome label alongside the title, and the pairs table shows that label in its own "Outcome A"/"Outcome B" columns — its market cells are cut at 40 characters, so on any title that runs past that — which the daily families this exists for all do — the appended label is the first thing lost, and two strikes of one family share a title and would otherwise render identically.
- **Backtest.** Each `BacktestTrade` records `entry_nB`; a time-series trade's profit is negative only in the in-between outcome. Any candidate whose settlement violates the cumulative-deadline premise (earlier YES, later NO) is skipped rather than paid, and the run logs one warning with the count. Since the wording check landed, that counter is a backstop rather than the only signal — the backtester refuses a snapshot pair up front through the same helper the live scanner uses — so expect it at or near zero. DR-72 widened what a non-zero count is read as: most likely a wording false negative (a snapshot or "before \<date\>"-worded recurring window read as cumulative), but also possibly legs ordered on an early REALIZED close (cross-event pairs only — a same-event ladder is ordered on its stated deadlines), or strike-blind grouping on a cache without subtitles — the warning names all three rather than pointing at one. Each run also logs, and the dashboard renders, how its eligible markets are worded: how many are worded as a cumulative "by \<date\>" deadline, how many are snapshots, and how many carry no deadline wording the classifier recognises — reported as a verdict only when the three counts actually sum to the corpus, otherwise as raw counts with no claim attached. Existing backtest caches need no refresh.
- **Not updated.** `kalshi_bot_flowchart.pdf` predates this change (it shows the old `|pA − pB|` filter) and has not been regenerated; `BUG_SWEEP_FINDINGS.md` is a dated record and is left as-is.
- **Calibrating `k` from settled history (shipped 2026-09).** `backtest.py --interval-discount K` overrides `k` for one backtest run (the live bot is untouched — it prices at its run's `config.LiveSettings`, built from the saved live defaults, or `main.py`'s own `--interval-discount` for one live run, neither of which a backtest can reach); the backtest also measures the *empirical* `k` — the realised in-between rate divided by the mean market-implied gap, pooled and per deadline-gap bucket — logs it, and reports it in the dashboard's "Interval Discount (k) Calibration" section, whose equity curve and per-`k` table follow the page's filter bar: its `k` select switches them between every `k` in the swept grid (`--no-sweep` to skip the extra re-simulation passes), and its size cap between every per-trade cap, always at the primary spread band and with the tier floors on. The section after it breaks the same `k̂` down by Kalshi category, by tag or by spread band, following the page's filter bar, and two cards in the Portfolio Performance section show the `k̂` of the band, category or tag selected and `k̂ − k` against the `k` selected (red when the sizer sized too big). This is a recommendation only: nothing writes the measured value back to `config.py`. That same section now opens with the run's **outcome-label coverage** — how many of the run's eligible markets carried the `subtitle` both grouping keys are built from. It is shown on every run, so a clean run is confirmable rather than merely unwarned; below `config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION` it becomes a red banner beside the `k̂` card (and a one-line notice at the top of the page), because a cache written before the 2026-08-14 outcome-label ingest fix groups strike-blind and makes every figure on the page describe a different strategy from the shipped one. The remedy it names: delete `backtest_cache/archive_days/` and `backtest_cache/live_days/`, then re-run with `--no-cache` — `--no-cache` alone does not refresh the day slices. Fractional-contract sizing remains deferred. Note that since the cumulative-deadline wording rule shipped, the **cross-event** time-series leg is dormant (same-event deadline ladders, the one population that is not, are on by the 2026-09-26 decision — see above): it forms 0 cross-event pairs live, and a backtest's surviving cross-event candidates are mostly recurring windows the entry rule can never enter, so a pooled `k̂` measured on a pre-rule cache should not be trusted (see `CLAUDE.md`'s cumulative-deadline gotcha).

---

## Architecture

### Module Dependency Graph

```
secrets.json + PEM key
        |
        ↓
config.py (constants), _http.py (retry + raw-response fetch + one-line error text)
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

    historical.py ──→ treasury.py  (downloads the 8-week Treasury bill's auction
                                     yields for the backtest dashboard's Sharpe/
                                     Sortino risk-free rate; reuses historical.py's
                                     CACHE_DIR/_load_json_cache/_save_json_cache/
                                     _exception_summary idioms. Reporting only —
                                     the live bot never imports it.
                                     treasury.py → dashboard.py: every
                                     Sharpe/Sortino subtracts its per-day
                                     yield — a strategy curve's on its capital
                                     in open trades only; treasury.py →
                                     backtest.py: load_risk_free_rates() hands
                                     the yields to generate_dashboard)

    (historical.py also imports auth.py's build_client for its own client
     builders, _http.py directly for its raw signed GETs, and scanner.py's
     event_series so the event-title lookup budget's combo test agrees with
     the one-series rule; main.py ──→ historical.py for load_series_categories,
     series_labels and infer_category, and dashboard.py ──→ historical.py for
     series_labels — the one rule that files a pair under a Kalshi category
     and tag, so the live category/tag filter files a pair where the
     dashboard's Category and Tag selects file a trade of the same event;
     backtester.py
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

    defaults_server.py — standalone, human-run; imports config.py only,
    imported by NOTHING (the local confirmation page that saves
    live_defaults.json)
```

### Live Trading Data Flow

Step order below is `_run_prod`'s; `_run_dev` skips `get_held_tickers` and
`resolve_held_ladders` entirely (no sandbox positions) and so calls `fetch_shard_statuses`
first instead.

```
main.py
  ├─ config.order_api_version_error() — refuse any ORDER_API_VERSION but "v2" (exit 2, right after parsing its arguments)
  ├─ main._resolve_live_settings() — the run's one config.LiveSettings: live_defaults() reads
  │                                   live_defaults.json (the saved live defaults, the run's one read,
  │                                   no config.py fallback), each toggle overridden by its flag when
  │                                   given; the saved defaults are the reference. No file, a refused
  │                                   file or a bad flag exits 2 before logging, the client or any request
  ├─ auth.build_client()           — authenticate with Kalshi API
  ├─ main._log_live_settings()     — log "Live defaults: <origin>" (the saved file, when and from what),
  │                                   the run's toggles with each departure marked "(default: X)" (a
  │                                   WARNING when a production run submits orders under one), and every
  │                                   live_rule_warnings sentence
  ├─ auth.verify_auth()            — read the per-shard balance breakdown (prod only; gate and size on the sum)
  ├─ scanner.get_held_tickers()    — fetch currently-held positions (prod only) so we skip re-entering them
  ├─ scanner.fetch_shard_statuses() — read GET /exchange/status per-shard trading/transfer flags (fail-soft)
  ├─ scanner.inactive_shard_indexes() — derive the trading-inactive shard set from the statuses above
  ├─ scanner.fetch_open_events_with_markets() — fetch open events + their markets from EVERY exchange shard, tagged (attaches event titles for MVE grouping; drops only markets on trading-inactive shards)
  ├─ main._log_shard_coverage()    — audit advertised shards vs ingested markets/funds (reports, never aborts)
  ├─ scanner.resolve_held_ladders() — the ladders (events, and questions at any deadline) the held positions
  │                                   are on (prod only): read from the full market list before held markets
  │                                   are dropped, a held market missing from it looked up; one that cannot
  │                                   be identified means no time-series pair this run (same-title still runs,
  │                                   and the run exits 40)
  ├─ scanner.filter_markets_within_horizon() — optional --max-horizon-days cap (no-op if unset)
  ├─ scanner.find_time_series_pairs()   — time-series pair detection (first refusing any
  │                                        candidate with a market on a held ladder, before
  │                                        the one-best-pair-per-group choice), under the run's
  │                                        config.LiveSettings (the spread rule:
  │                                         pB − pA positive, at least the entry floor,
  │                                         within the band's ceiling; the rule is logged)
  │                                        (both legs must be cumulative "by <date>"
  │                                         deadlines at two different dates, compared
  │                                         as normalized strings — see the note above;
  │                                         with TIME_SERIES_SAME_EVENT_LADDERS on, the
  │                                         default by the 2026-09-26 decision, also
  │                                         two dated rungs of ONE event, ordered and
  │                                         gapped on their stated calendar deadlines)
  ├─ scanner.find_same_title_pairs()    — same-title pair detection
  │                                        (two different event series, and both
  │                                         markets closing within one hour of each
  │                                         other — scanner.closes_apart, DR-74)
  ├─ main._dedup_pairs()            — merge both lists, preferring same-title on overlap
  ├─ main._filter_by_category()     — keep only pairs filed under the run's Kalshi categories and tags
  │                                   (the dashboard's rule: market A's literal series, its category and
  │                                    first tag, via historical.series_labels); a no-op, with no request,
  │                                    when neither is set; prod refreshes a stale /series listing, dev
  │                                    reads the cached copy only; no listing -> no pair (fail closed)
  ├─ scanner.enrich_with_orderbook_prices() — validate depth; price each pair over the depth this balance could actually buy; drop a time-series pair with no later YES ask, a crossed later book, or a fresh spread the run's rule refuses
  ├─ strategy.compute_trade()      — Kelly sizing per pair, under the run's config.LiveSettings (k and the per-trade caps, the same object enrichment's depth bound read), at the marginal fill price of the size it settles on, and only over depth the fill-or-kill limit can actually reach
  ├─ strategy.select_portfolio()   — greedy portfolio selection, at most one time-series trade per ladder (none on
  │                                   a held ladder, and none on the ladder of a trade picked earlier in the run)
  ├─ trader.pre_execution_check()  — re-fetch order books, drop pairs whose prices moved or whose depth is no longer reachable at the limit about to be submitted, counting only the fresh levels that still keep an edge after the fee (a time-series pair also when its later book has no YES ask now, or its fresh spread exceeds the band's ceiling)
  ├─ trader.ensure_shard_collateral() — move funds onto the shards the selected legs settle against (prod; dry-run only plans)
  ├─ trader.execute_trades()       — submit fill-or-kill orders leg-by-leg to the V2 order endpoint, each leg routed to its own market's shard (parallel across pairs once the first NO fill confirms the V2 order mapping, one at a time before that; rollback on partial fill; a disproven mapping stops the rest of the run)
  ├─ auth.verify_auth()            — re-read the post-trade balance for the Excel log (falls back to the pre-trade balance if this read fails)
  └─ reporter.append_to_prod_log() — write results to trade_log.xlsx, the run's live toggles on its separator row
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
  │         ├─ _entries_for_band()            — k-independent; records every qualifying Monday of each
  │         │                                    pair; one time-series _find_entry() pass per band:
  │         │                                    every band of SPREAD_BAND_SWEEP_FLOORS x CEILINGS (plus the
  │         │                                    primary if it is off-grid) by default — only the no-band pass
  │         │                                    scans every pair, the rest rescan the pairs that entered
  │         │                                    there — or the primary band alone with --no-band-sweep;
  │         │                                    plus one same-title pass; with the band sweep, again with
  │         │                                    the tier floors off (tier_floors=False) for every band a
  │         │                                    deadline-gap tier binds at (floors 0 / 0.20 / 0.25), from
  │         │                                    its own no-band pre-pass
  │         ├─ _interval_calibration()        — empirical k_hat per band, from that band's k-independent entries
  │         │                                    (and per binding band with the tier floors off)
  │         ├─ _simulate_at_discount()  — once per (band, k[, population]): Kelly gate (a time-
  │         │                              series pair keeps every Monday it passes on, a
  │         │                              same-title pair its first), one same-title pair per
  │         │                              group, dedup, a Monday-by-Monday cash walk that trades
  │         │                              each pair at most once and holds at most one open
  │         │                              time-series trade per ladder (a ladder frees up the day
  │         │                              its market pays out; a time-series pair skipped one
  │         │                              Monday is tried again on its next passing one), P&L from
  │         │                              outcomes, _build_equity_curve() — the scenario explorer's
  │         │                              band x k x {all, time_series, ladder, cross} grid, its
  │         │                              tier-floors-off twin at the binding bands, plus
  │         │                              one same-title simulation, come from repeating this call,
  │         │                              never from slicing a joint run; every one of them at the
  │         │                              run's own per-trade size cap (config.BUDGET_FRACTION), a
  │         │                              same-title candidate also under config.SAME_TITLE_SIZE_CAP
  │         │                              (config.pair_size_cap, as live sizing caps it)
  │         └─ CapSweep (default; not with --no-cap-sweep) — every other per-trade size cap of
  │                                        SIZE_CAP_SWEEP over the tier-on grid, and a second
  │                                        one (tier_floors=False) over the tier-floors-off
  │                                        twin, each seeded from its own grid's points,
  │                                        none of them simulated here: each (band, k) cell
  │                                        is simulated through
  │                                        _simulate_at_discount(size_cap=) only when a report
  │                                        reads it, and caps at or above a point's peak Kelly
  │                                        fraction share one simulation
  ├─ treasury.load_risk_free_rates()         — 8-week T-bill auction yields from Treasury Fiscal
  │                                             Data (one open, no-key GET that never raises —
  │                                             falls back to the last saved download, then to
  │                                             "unavailable"; a host that swallows packets
  │                                             costs each of 6 attempts at least the 30 s
  │                                             timeout — per resolved address, since it bounds
  │                                             each socket operation, not the request — plus
  │                                             62 s of backoff: ~4 min for a single-address
  │                                             host); passed to generate_dashboard(risk_free=)
  └─ dashboard.generate_dashboard()          — write HTML report: a k section and two k̂ cards
                                                that follow the filter bar, a k̂ breakdown by
                                                category / tag / band (regrouped from each band's
                                                calibration observations), the scenario-explorer section
                                                (band x k heatmap per size cap, nine metrics,
                                                per-population KPIs, and a Tier floors select over its
                                                tier-floors-off grid at every cap), and a sticky
                                                filter bar whose every band x tier floors x k x size cap
                                                x Kalshi category x tag view is computed here: one walk
                                                over the scenario grid reads each CapSweep cell
                                                (simulating it) and then each binding band's
                                                tier-floors-off CapSweep cell, packs one chunk per distinct
                                                trade list, keeps the k section's rows and curves at
                                                every k and cap and the explorer's figures for every
                                                band x k x cap (a block per cap, and a tier-off block
                                                per cap), streamed into the page beside
                                                base blocks and unpacked by small scripts only when
                                                chosen; the bar's "Save as live
                                                defaults…" button opens the defaults
                                                server's confirmation page for the
                                                scenario on screen (the page itself
                                                writes nothing)
```

`run_backtest()` composes `_prepare_candidates()` with a single `_entries_for_band()` pass at the default (no-op) band (the two together are `_prepare_entries()`) and one `_simulate_at_discount()` call, so every pre-existing caller of the plain two-tuple entry point is unaffected by the band sweep above.

---

## Module Descriptions

| Module | Description |
|--------|-------------|
| `__init__.py` | Package initializer. No exports; marks the directory as the `kalshi_betting` package. |
| `config.py` | All tunable constants (price thresholds, Kelly cap, fee rates, API URLs, file paths — bar one backtest-only exception, the size-cap grid `backtester.SIZE_CAP_SWEEP`, kept in `backtester.py` by the operator's instruction for that change), the two fee helper functions, `max_affordable_pairs()` — the single budget-to-contracts definition shared by the scanner's depth bound and the sizer — and the time-series spread-band helpers `time_series_spread_band()` / `time_series_spread_too_wide()` plus the `SPREAD_BAND_SWEEP_FLOORS` / `SPREAD_BAND_SWEEP_CEILINGS` grid the scenario explorer sweeps, and the live trading toggles' constants — `TIME_SERIES_TIER_FLOORS`, `TIME_SERIES_SPREAD_BAND`, `SAME_TITLE_SIZE_CAP` (an extra per-trade cap on same-title pairs; 0.20 since the operator decision of 2026-09-27, 1.0 — no extra cap — before it) and `TRADE_CATEGORIES` / `TRADE_TAGS` (the category/tag filter; `None`, any, today), which with `k` and the per-trade cap are the backtest's values and, through `live_settings()`, the fallback for a caller that hands no settings — never a live run's defaults. A live run starts only from the saved live defaults, `live_defaults.json` (`LIVE_DEFAULTS_FILE`), read into one frozen `LiveSettings` per run by `live_defaults()` (over which `main.py`'s toggle flags may lay a value for one run), written only by `save_live_defaults()` (on `defaults_server.py`'s Confirm, whose address and limits are the `DEFAULTS_SERVER_*` constants), with `LIVE_DEFAULTS_SEED` the starting values `defaults_server --seed` offers for a first save; the helpers that name them (`describe_live_settings()`, the run's `Live settings:` line, `describe_trade_filter()`, the category/tag filter in that line's words, and `live_rule_warnings()`, every setting that empties part of the strategy or lets one pair stake more than `LIVE_EXPOSURE_WARN_FRACTION`), and the helpers that apply them (`live_time_series_floor()`, `time_series_spread_refusal()`, the one definition of the live time-series spread rule, `pair_size_cap()`, the one definition of a pair's per-trade cap, shared by the live sizer and the backtester, and `max_kelly_fraction()`, the scanner's affordability bound), and `order_api_version_error()`, the startup check `main.py` and `v2_probe.py` run to refuse any order path but `"v2"`. No live module reads a toggle constant or a backtest band helper directly: the finder, the category/tag filter (`main._filter_by_category`), the scanner's enrichment and pre-execution check, and live sizing (`strategy.compute_trade`) all read the run's one `LiveSettings`. Also the weekly live run's schedule, `SCHEDULED_RUN` — a frozen `ScheduledRun` holding its weekday, time and IANA zone (Monday 09:00 America/Los_Angeles), which the scheduler reads and the backtest enters every trade at. |
| `auth.py` | Reads RSA credentials from `secrets.json` and the PEM key file, constructs an authenticated `KalshiClient`, and provides `verify_auth()` to confirm credentials and read the live account balance per exchange shard (`{exchange_index: cents}`; callers sum for sizing). |
| `_http.py` | Shared HTTP helpers used across the package (auth, scanner, historical, trader, and v2_probe): `api_call_with_retry()` (exponential backoff on 429/5xx for market-data calls) and `fetch_json_page()` (parses the SDK's raw `*_without_preload_content` responses, re-raising non-2xx as `ApiException`), and `signed_request_json()` (signed GET/POST against an arbitrary API path for routes the pinned SDK has no method for — retry-free, since order submission and the collateral transfer call it directly); plus `api_error_payload()` (reads the exchange's JSON error object — `code`, `message`, `details` — out of a failed request) and `api_error_summary()` (one line per failed request: `HTTP 400 Bad Request — missing_parameters: missing parameters (…)`, or the body itself on one line when it is not that object, never the SDK's multi-line exception text with every response header; `trader.py` logs and records every failed order, unwind, position read and transfer this way — `TradeResult.error`, and so the trade log's Notes cell, carries the same line — and `scanner.py` logs a failed order-book read this way). |
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles and appends each market's outcome label (subtitle) to group time-series pairs, detects same-title pairs via exact match, refuses a same-title pair between two events of one series or whose two markets close more than an hour apart (`closes_apart()`, the one definition of that close gate, failing closed on a close time it cannot compare, with its own silent-at-zero refusal count whose printed bound comes from `close_gap_bound_text()` on both paths — DR-74), a time-series pair whose wording is identical across one series (two instances of one recurring fixture — a genuine two-deadline family of one series spells its deadline in the wording and can still pair, if both legs are worded as cumulative deadlines; `event_series()` reads the prefix before the first hyphen, except that every `KXMVE*` combo prefix collapses onto one family so two combos listed under two different combo series are still refused), a time-series pair between two markets of ONE event unless `TIME_SERIES_SAME_EVENT_LADDERS` is on and they are two dated rungs of that event's deadline ladder (`stated_deadline()` / `same_event_ladder()`, which order the legs and measure the gap on the STATED deadlines; `pair_gap_days()` is the one place anything downstream reads that gap), and a time-series pair whose legs are not both worded as cumulative "by \<date\>" deadlines at two different dates (compared as normalized text, not parsed calendar dates — see the note above; `deadline_phrasing()` — a snapshot family such as "Solana price on Sep 14/18, 2026?" still groups but no longer pairs; `deadline_pair_refusal()` names WHY a refused pair was refused — snapshot wording, no stated deadline, or the same deadline stated twice — feeding three separate, honestly-labelled skip counts instead of one folded one, DR-72; every one-series refusal, in both finders, and every same-title same-event skip is counted on its own silent-at-zero line too, M10), refuses — first, before every other check — a time-series candidate with a market on a ladder the account holds (`ladder_keys()` / `market_ladder_keys()` / `pair_ladder_keys()` name the ladders a market is on — its event, and its question with the dates removed — and `resolve_held_ladders()` finds the held positions' ladders, looking up a held market the run's list lacks, and failing closed when one cannot be identified), and enriches tradeable pairs with live order book depth to compute real fill prices — averaged over the contracts the balance could actually buy, not the whole book, with the qualifying levels kept on the pair (`depth_levels`) for the sizer to re-price against via `prefix_fill_prices()`. Also home to `leg_sides()` / `leg_prices()`, the single mapping from a pair's type to the side and price each leg actually trades, and to the V2 order-price grid arithmetic (`tick_size_for_price()`, `ceil_to_tick()`, `v2_limit_price()`, `v2_effective_cap()`) — it lives here, not in `trader.py`, so the sizer can test a candidate size against the very limit the trader will submit without importing it. |
| `strategy.py` | Solves size and price together — binary-searching the book for the largest contract count whose own marginal fill price still justifies it AND that the resulting fill-or-kill limit can actually buy — then applies the Kelly criterion to size each trade — at the run's `k` and under its per-pair cap (`config.pair_size_cap`: the per-trade cap, and for a same-title pair `SAME_TITLE_SIZE_CAP` too), both read from the run's one `config.LiveSettings` — computes the profit floor for same-title pairs / the win-scenario profit for time-series pairs and the monthly-normalized return, and greedily selects a portfolio that fits within the available balance, with at most one time-series trade per ladder (none on a ladder the account holds, and none on the ladder of a trade picked earlier in the run). |
| `trader.py` | Converts `TradeSpec` objects into orders and submits each pair's two legs sequentially (fill-or-kill, NO leg then YES leg — the NO leg is `market_a` for a same-title pair and `market_b`, the later contract, for a time-series pair) via the Kalshi API, with automatic rollback of the filled NO leg if the YES leg doesn't fill. Multiple pairs execute concurrently — except until the process's first NO-leg fill has confirmed or disproven the order-side mapping, when they run one at a time (for at most `config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS`, 300 s); a disproof stops every pair that starts after it before anything is sent (`failed`) — and every order and collateral-transfer POST from every pair takes a place on one shared pacer (`config.ORDER_WRITES_PER_SECOND`, in bursts of at most `config.ORDER_WRITE_BURST`) so they stay under the account's write limit; a pair's NO leg holds a place for its YES leg, and unwinds go ahead of waiting NO legs — see "Order API version" below. Submission goes to the V2 order endpoint, the only order path the bot has — see "Order API version" below. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`, each run's separator row naming the live toggles the run traded under. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). Market cells are rendered by `scanner.display_title`, so they carry the event title and the outcome label alongside the market title. |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Refuses to start (exit 2) on any `config.ORDER_API_VERSION` but `"v2"` (see [Order API version](#order-api-version)), then resolves the run's one `config.LiveSettings` — the saved live defaults (`live_defaults.json`; with none saved, or a refused file, the run exits 2), each overridden for this run only by its flag when given (see [Live trading toggles](#live-trading-toggles)) — before logging is configured, then dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`, handing it that object and the saved defaults it was built from; the run logs where the defaults came from and its settings (marking every departure from the saved defaults `(default: X)`, and warning when a production run submits orders under one) and every live site reads that one object. Beyond orchestration it holds two pair-list filters: `_dedup_pairs()` (a same-title pair wins a ticker-pair collision) and `_filter_by_category()` (keep only the pairs filed under the run's Kalshi categories and tags, by the backtest dashboard's own rule — a no-op when neither is set). A production run also finds the ladders its open positions are on (`scanner.resolve_held_ladders()`) and hands them to the time-series finder and the portfolio step; a held market it cannot identify means no time-series pair that run. |
| `scheduler.py` | Long-running daemon that fires the production bot once a week at `config.SCHEDULED_RUN`'s weekday and time (Monday 09:00) on the host's clock, using the `schedule` library, with no live toggle flag, so a scheduled run trades exactly the saved live defaults (with none saved, or a refused file, every run exits 2, and the daemon logs an ERROR saying so when it starts), and checks at startup that the host's clock keeps the schedule's zone (America/Los_Angeles), logging CRITICAL if it does not. Also prints the equivalent cron job command. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints, sharded into parallel per-day slices that are cached individually so interrupted or repeated fetches resume instead of re-walking months of history) and hourly candlestick price series needed by the backtester (candlesticks are fetched in parallel across tickers and cached per ticker, so workers never share a cache file and a repeat run re-reads them from disk). Also caches Kalshi's /series listing (`load_series_categories()`, each series' category and tags, refreshed weekly) and owns the one rule that files an event under a category and FIRST tag (`series_ticker()` / `series_labels()`), which both the backtest dashboard and the live category/tag filter (`main._filter_by_category`) use. |
| `treasury.py` | Downloads the 8-week U.S. Treasury bill's auction yields from the Treasury's Fiscal Data API (one open, no-key, read-only GET, retried like every other market-data read) for the backtest dashboard's Sharpe and Sortino ratios to subtract — on each day of a curve, the yield of the most recent auction on or before that day, not one fixed hurdle. Every successful download is saved under `backtest_cache/treasury_bill_rates.json`; the loader never raises — on any failure it falls back to the last saved copy, and with none to "unavailable" (every ratio then subtracts 0%, and the dashboard's header says so). Reporting only: nothing sizes, prices or settles on it, and no live-trading module may import it. Every Sharpe and Sortino on the dashboard subtracts it — the performance cards, both benchmark rows, the per-`k` table, the scenario explorer and every filter-bar view — through `dashboard.generate_dashboard(risk_free=...)`, whose header line names the rate (or says none was supplied). A strategy curve is charged the yield only on its capital in open trades (each day, the previous close's share of the portfolio held in open trades at cost — without fees, the cost basis the portfolio value itself carries — so a trade is charged for the days it is held); idle cash is taken to earn the same yield, since the backtester books it at 0%. The S&P 500 row is fully invested and is charged the whole yield. `backtest.py` calls `load_risk_free_rates()` beside its series-category read and hands the result in. |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs (including, behind `TIME_SERIES_SAME_EVENT_LADDERS`, two dated rungs of one event's deadline ladder — formed by a separate, deliberately unwindowed per-event sub-pass and ordered and gapped on their stated deadlines through the same `scanner.stated_deadline()` / `same_event_ladder()` the live finder uses), scans weekly snapshots at the live run's own instant — `config.SCHEDULED_RUN`, Monday 09:00 America/Los_Angeles (16:00 UTC under daylight time, 17:00 UTC under standard time) — for every Monday each pair was tradeable, refusing before any fetch a schedule whose run time falls on another date in UTC or on a clock change — at a BACKTEST time-series spread band (`_entries_for_band()`, `_find_entry()`) that never reaches live trading, which reads its own band, the saved live defaults' (or `main.py`'s own `--spread-min` / `--spread-max` for one run) — the two apply the same spread tests, pinned to one verdict by `tests/test_backtester.py::TestLiveBacktestSpreadParity` — enters each time-series pair on the earliest of those Mondays whose Kelly fraction is positive at the simulated `k` and on which it can be taken, and each same-title pair on its first such Monday or not at all, holding at most one open time-series trade per ladder — a time-series pair skipped one Monday for a busy ladder or short cash is tried again on its next such Monday (as the weekly live run would), while a same-title pair keeps the one-per-group rule and is not tried again — applies Kelly sizing, records actual P&L from settlement outcomes, and builds a daily equity curve that opens one day before the start date at the untouched initial balance, so a trade entering on the first day of the window shows its day-0 charges as a real daily return and a real drawdown. The curve is a portfolio value, not a cash balance: an open position is carried at its cost basis for its whole holding period, so committing capital does not move the curve and the drawdown/Sharpe/Sortino figures derived from it measure realized loss rather than peak deployment. The work is split at the band and the interval discount `k`: `_prepare_candidates()` (fetch through candlesticks) depends on neither, `_entries_for_band()` depends only on the band, and `_simulate_at_discount()` — the Kelly gate, dedup, P&L, equity curve — depends on `k`; `_sweep_from_candidates()` composes all three into the band x `k` x population scenario grid (`SweepPoint`, `HalfSplit`, `BacktestSweep`) the dashboard's scenario explorer renders — and, with `tier_off_sweep` (which the CLI turns on with the band sweep), re-runs every band a deadline-gap tier floor binds at with the tier floors off, so that band's floor alone gates the spread (`BacktestSweep.tier_off_scenarios`, backtest only; every tier-on figure unchanged). Pair extraction applies the live scanner's same-title close gate through the same `scanner.closes_apart()` (DR-74), and accounts for every pair of every group's members, each on its own silent-at-zero line — the three wording reasons, the one-series rule, the same-event skip, the same-title close gate (plus a backtest-only line for a same-title pair whose close time cannot be read), each ladder reason and, for time-series groups, the pairs its close-date window never visits and the members with no readable close time — and the size of each grouping is logged on every run, zero included, so a run that forms no pairs still logs why (M10). The per-trade Kelly size cap is a simulation parameter too (`_simulate_at_discount(size_cap=)`, default `config.BUDGET_FRACTION`; a same-title candidate also stays under `config.SAME_TITLE_SIZE_CAP` at every cap, through `config.pair_size_cap`, as live sizing caps it), and `run_backtest_sweep(cap_sweep=True)` — on by default in the CLI — adds a lazy `CapSweep` over every other cap of `SIZE_CAP_SWEEP` (5% to 95%, and 1.0, shown as off) for the tier-on grid, and a second one over the tier-floors-off runs (`BacktestSweep.tier_off_cap_sweep`, seeded from their own points) when those ran too, each simulated one (band, `k`) cell at a time when a report reads it. |
| `dashboard.py` | Generates a self-contained HTML performance report from backtest results — nine sections: cumulative return lines (total plus one per trade type) / Sharpe/Sortino/drawdown KPIs plus mean and median return per trade and the median monthly return, returns decomposition (P&L by Kalshi's official category and by category · tag, with a per-group table), time-series spread calibration (each traded pair's entry spread `pB − pA` — the market-implied probability that the event lands between the two deadlines — against the rate the pairs settled A = NO, B = YES, with Brier score and log loss; same-title trades are not shown), an interval-discount (`k`) calibration section (the pooled empirical `k̂`, one equity curve and a per-`k` table at the primary spread band with the tier floors on, which the filter bar's `k` and size cap move), two Portfolio Performance cards with the selection's empirical `k̂` and `k̂ − k` (the `k̂ − k` red when positive, i.e. the sizer sized too big), an empirical `k̂` breakdown (a bar chart and table of `k̂` by Kalshi category, by tag or by spread band, each bar naming its entries and distinct events, with a Group-by `<select>` of its own and a dashed line at the page's `k` — the run's own as rendered, the filter bar's once one is chosen), a scenario-explorer section over every spread band x `k` x per-trade size cap of the band sweep (a fragility banner for the cap shown, a spread-band x `k` heatmap whose metric menu offers mean, median and total return, Sharpe ratio (blank on a cell with no trade), H1/H2 returns, trade count, each band's empirical `k̂` and `k̂ − k` — both red where `k̂` exceeds `k`, as on the cards — a one-row-per-band `k̂` table, and a per-population KPI table, with band, `k` and size-cap `<select>`s of its own — and a Tier floors `<select>` when the run carries the band sweep's tier-floors-off runs — that the filter bar also moves; its data comes from the same one walk as the bar's and ships as one gzip-packed block per size cap, plus a tier-floors-off block per size cap, each unpacked only when chosen), trade diagnostics (best and worst five trades with each leg's side, price, close date and settlement), risk metrics, and an S&P 500 benchmark comparison whose download window opens on the same date as the equity curve's leading initial-balance row. The page header names the run's primary spread band, same-event-ladder setting and per-trade size cap (and whether the size-cap sweep ran), and, on the line under them, the entry checkpoint every trade was entered at — the live scheduler's weekly run time, `config.SCHEDULED_RUN`. A sticky filter bar at the top — Spread band, Tier floors, k, Size cap, Category, Tag — re-scopes every trade-derived section (performance, decomposition, calibration, diagnostics, risk, the benchmark's strategy row) to another scenario's own run — a spread band at a `k` and a per-trade size cap (5% to 95%, or off, when the size-cap sweep ran; the run's own cap otherwise), each a standalone simulation, the caps other than the run's own simulated as the page is built — and/or to one Kalshi category or category · tag (the series' first tag, so breakdowns partition) of that run, whose return, drawdown, Sharpe, Sortino, median monthly return and benchmark row are then its contribution — the starting balance plus its trades' P&L as the run booked them — not a standalone simulation. Its Tier floors choice switches every band between the run as simulated, with the 0.15/0.30 deadline-gap tier floors applied, and the band's run with them off — its own floor alone, from the tier-floors-off runs the CLI's band sweep also simulates (backtest only, at every size cap: the run's own cap from those runs themselves, every other from the lazy tier-floors-off size-cap sweep, simulated as the page is built — a run or a page without that sweep has a band the tiers bind at with the tier floors off at the run's own cap only, and the bar says so at any other); a band whose floor sits at or above both tiers was never simulated again, because the tiers never bind there, so its off view is its tier-on run, at every size cap, and says so, and a run without those runs (`--no-band-sweep`) keeps the select disabled with a "(not simulated for this run)" note. The header's trade count follows the selection too, the `k̂` breakdown and the two `k̂` cards move to the same band, tier setting and selection (the breakdown's dashed reference line, and the `k̂ − k` card, to the chosen `k`), and the Interval Discount (k) section to the chosen `k` and size cap (still at the primary spread band, with the tier floors on — it never follows the Tier floors choice) — that chart regroups the band's own `k̂` population (every time-series candidate entry, measured before the Kelly gate), not the run's trades. Every view is computed in Python by the helpers the sections render with and shipped gzip-packed — one small base block, and one chunk per distinct scenario trade list (scenarios that traded equal lists at one `k` share it), which the page unpacks only when that scenario is chosen, so a large grid does not slow the page's load; the page is streamed to disk piece by piece, never joined into one string (each packed chunk is held until it is written). The bar's band, Tier floors choice, `k` and size cap also move the Scenario Explorer's own selects to the same scenario — each only when the bar's choice on it changes, so a choice made in the explorer survives a category or tag change (they stay usable on their own), and a move the explorer has to refuse (a size cap it holds no tier-floors-off data for — none when the tier-floors-off size-cap sweep could be read; otherwise every cap but the run's own, whatever the band, unlike the bar — or a block it cannot unpack) is named on its status line and applied again on the bar's next change; category and tag reach neither it nor the Interval Discount (k) section. Set to off, the explorer's banner, heatmap and `k̂`-by-band table switch to the tier-floors-off grid and its KPI table, calibration table and equity curve read the tier-off cells — a band the tiers never bind at shows its one run, and the Same-title row is the same either way — and its equity curve re-autoranges on every redraw, so a zoom never carries into another band, `k`, cap or setting; the bar's summary line says what it reaches on that page. The `k̂` cards are computed from the primary calibration itself, so they render even when the bar cannot be built. The bar's selects start disabled and are enabled once the page has unpacked its base block and the run's own scenario (the Tier floors select only when the run carries its tier-floors-off runs), and each chart is redrawn from its layout as first drawn, so a zoom never carries into another selection; a later choice supersedes one still loading, and a scenario the run never simulated says so. If the filter's data cannot be built, the page is still written, without the bar and with a notice in its place; if the size-cap sweep cannot be simulated, the bar offers the run's own cap only and the header says the sweep could not be used; if only its tier-floors-off half cannot be, the bar keeps every cap with the tier floors on, offers the run's own cap with them off, and the header says so. After the Tag select the bar carries a "Save as live defaults…" button, rendered disabled and enabled once the page has loaded the scenario on screen, whenever that scenario can become the live settings (simulated, its band recorded, its `k` and size cap recorded and above zero, and a category or tag only on a page built with Kalshi's series listing); a click opens the defaults server's confirmation page (`python3 -m kalshi_betting.defaults_server`, on `127.0.0.1:8765`) in a new tab for the bar's scenario on screen — never the Scenario Explorer's own selects — with the run's same-title cap, and that page saves it only on its own Confirm. A page whose filter bar could not be built has no button. Every run overwrites the one `backtest_dashboard.html` in the repo root. |
| `backtest.py` | CLI entry point for the backtest pipeline. Parses arguments (including `--interval-discount`, `--no-sweep`, `--same-event-ladders` / `--no-same-event-ladders`, the backtest-only `--spread-min` / `--spread-max` / `--no-band-sweep`, and `--no-cap-sweep`), builds the historical API clients, calls `backtester.run_backtest_sweep()` then `dashboard.generate_dashboard()`, and logs a summary of the primary result. |
| `defaults_server.py` | Human-run local web server (`python3 -m kalshi_betting.defaults_server [--seed] [--no-browser]`, on `127.0.0.1:8765`) whose one confirmation page saves the live defaults, `live_defaults.json`. The page is opened with the proposed settings in its address (by `--seed`, which proposes the seed values, or by the backtest dashboard), shows the defaults in force beside the proposed ones with every change highlighted and, in red, every warning a live run would log about the proposed settings, and writes the file (through `config.save_live_defaults`) only on its Confirm button; closing the tab cancels. It never serves the dashboard, listens on loopback only, answers one request at a time, and refuses a save unless the request names this server (Host), comes from its own page (Origin), carries the token that page was built with, and finds the same defaults still in force (a page gone stale is shown again against the defaults now in force). Its Confirm button is enabled only after the page has been visible for a second and the mouse moves or a key is pressed. Logs to `kalshi_defaults_server.log`. Imports `config.py` only; nothing imports it. |
| `v2_probe.py` | Human-run CLI that verifies the V2 order path's NO-leg mapping, fill-or-kill kill semantics, and the inter-shard transfer's centicent unit against the production account for roughly one cent of exposure. Its closing reduce-only bid is priced at the top of the market's own grid (0.99 / 0.999 / 0.9999 by tick regime), not at the rollback builder's loss floor, so that floor can no longer cause a FAIL unrelated to the mapping (a book with no reachable resting YES ask still can); `reduce_only` is what bounds that bid. A 2xx order body that is not a JSON object, in either step that reads one (DR-58), and — in the NO-buy step only — an object whose fill counts are unreadable (DR-60), are a clean FAIL that still reads the position, re-reads it once when that first read is `None` or `0`, and reports lookup-failed, position-open and genuinely-flat as three distinct outcomes, never a traceback out of the fill readers. Two of the unfillable-ask step's branches are recorded residuals — its unreadable-fill-counts branch and its `not killed` branch both FAIL without re-reading the account. The NO-buy step classifies readable fill counts into three outcomes, not two — a complete fill, a true kill, and a fill-or-kill invariant violation — so a partial fill FAILs naming the counts and re-reading the account rather than being reported as a clean kill with the account "still flat" (DR-20); the unfillable-ask step always read a partial that way. Both steps judge a 2xx on `fill_count` AND `remaining_count`, which is deliberately stricter than the live `trader._v2_fill_status`, whose contract is `fill_count` alone. The exchange's HTTP 409 kill response to a fill-or-kill (`trader._is_fok_kill`) is read as a kill — PASS on the unfillable ask, NEUTRAL on the NO buy, when the account reads flat — and the close's verdict is judged on one re-read when the read after it is not exactly 0. `--dest-shard` equal to the source shard is refused with a NEUTRAL at the top of the transfer step, before any transfer I/O, so it can no longer POST a net-zero self-transfer and then report a false in-flight FAIL (DR-22). It refuses to start (exit 2) on any `config.ORDER_API_VERSION` but `"v2"`. The FAIL lines that doubt the V2 path after a submission on an account the probe checked was flat tell the operator to stop trading and flatten any position on the probed ticker by hand in the Kalshi UI. The closing line depends on the result: after a FAIL it says to stop trading — and, for an order step, to act only on the position warnings printed above, never by itself to flatten, since a FAIL before anything was submitted opened nothing and the ticker may carry the bot's own position; for the transfer step, to check each shard's balance — and after a NEUTRAL it says nothing needs doing. Its informational fee check compares the exchange's `average_fee_paid` — per contract, by Kalshi's API reference, and including Kalshi's rounding of the order's total fee up to the account's balance precision — with the fee model per contract at the fill price: before rounding, as `config.fee_leg_exact(1, p)` for one whole contract, and rounded as Kalshi rounds the probe's order, which is the figure a 0.01-contract charge should match. Never imported by the pipeline. |

### Order API version

`config.ORDER_API_VERSION` names the bot's order path, and `"v2"` is the only value it accepts: `main.py` and `v2_probe.py` call `config.order_api_version_error()` right after parsing their arguments and exit 2, with the reason on stderr, before logging is configured or any request is made, on anything else (another spelling such as `"V2"`, an empty value, `"legacy"`, …). The V2 path posts to `/portfolio/events/orders`: a fill-or-kill **limit** order with a dollar-string price, a fixed-point contract count, a `bid`/`ask` side on the market's single YES book, and an explicit `exchange_index`. V2 has no "market" order type, so the limit price is itself the price protection — the scanned price rounded up onto the market's own tick grid plus `BUY_SLIPPAGE_TICKS` ticks, which is a cap the older integer-cent `buy_max_cost` field could not express once MVE/combo markets moved to sub-cent ticks. A price sitting exactly on the boundary between two tick bands belongs to both, and the **finest** of them wins: taking the first match instead made the allowance ten times coarser at a band's upper edge, loosening a cap that is a bid. Because that limit applies per contract while the scanned price is an average over several book levels, `strategy.py` checks a candidate size against it before committing — otherwise the order asks for depth priced above its own limit and the whole fill-or-kill is killed.

Every V2 body also carries `self_trade_prevention_type`, a field the endpoint requires (it rejects a body without it with HTTP 400). The bot sends `config.V2_SELF_TRADE_PREVENTION_TYPE`, `"taker_at_cross"`: an order that would trade against another order on the same account is cancelled at that point. The two buy legs are fill-or-kill. The exchange kills one that cannot fill in full with an HTTP 409 error, code `fill_or_kill_insufficient_resting_volume` (`config.V2_FOK_KILL_HTTP_STATUS` / `V2_FOK_KILL_ERROR_CODE`), and the trader reads exactly that response as a clean non-fill — the pair fails, or rolls back, at once; any other error still goes to the position check. The rollback that unwinds a filled NO leg is not fill-or-kill. It is `reduce_only`, which the endpoint accepts only with `immediate_or_cancel`, so it buys back what rests at or under its loss-floored cap and cancels the rest. A close of only part of the position is reported as `rollback_failed` for manual review, never as flat, and no second order is sent; its alert names how many NO contracts are still open — the NO leg's count minus the fill count the rollback's own response reports — or says "up to" the NO leg's count when there is no usable count (an error response, a transport error, or a body with no readable count).

The V2 side mapping — an `ask` on the YES book opens a NO position — comes from Kalshi's docs, so the first NO leg that fills in each process is checked against the account's positions (`trader._confirm_v2_no_mapping`): the position must move by exactly minus the contracts bought. Until that check gives a verdict, the run's pairs execute one at a time, for at most `config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS` (300 s; a killed NO leg or an unreadable account gives no verdict, and the next pair runs alone in turn). If the position moved any other way — an unchanged position is re-read after 1, 2 and 4 s first, since the ledger lags a fill by about a second — the mapping is disproven: that pair stops with its YES leg unsent and its NO leg left for a human (`manual_review`), a CRITICAL says so and names any earlier pair that went ahead unchecked, and every pair that starts after it is stopped before anything is sent (`failed`) — within the 300 s that is every later pair; after it, pairs already running still send their NO legs. The run exits 20. The stop lasts only for the process, so each scheduled run checks again: after a disproof, stop the scheduler and flatten the positions by hand in the Kalshi UI.

The arithmetic itself lives in `scanner.py` (`v2_limit_price()` builds the wire price, `v2_effective_cap()` states it in the leg's own side terms) and `trader.py` re-exports it. That is deliberate: `trader.py` imports both `scanner.py` and `strategy.py`, so neither can import it back, and a second copy of the formula is exactly how the size and the limit drift apart again.

There is no other order path to switch to. Kalshi deprecated its legacy `/portfolio/orders` order endpoints in June 2026 (its changelog entry of June 18, 2026: once deprecated, a call returns `Please switch to the V2 endpoints`), and the V2 side mapping was confirmed live on 2026-09-28 (a `v2_probe` PASS and the first live trades). If the V2 path ever misbehaves — `trader.py`'s NO-leg mapping check logs a CRITICAL, or a `v2_probe` step FAILs — stop trading (stop the scheduler daemon if it is running, and do not run `main.py --mode prod`) and flatten any open position by hand in the Kalshi UI. The CRITICAL says exactly that, and so do the probe's FAIL lines that follow a submission on an account it had checked was flat; after any other probe FAIL, act only on the position warnings it printed, since a position on the ticker may be the bot's own. Every pair's legs are submitted in one order (the NO leg first — it is the leg that gets unwound, and `trader._ordered_legs` decides which market that is for the pair type), and no submission is ever retried.

Kalshi limits how fast an account may write: a token bucket per account, which on the Basic tier refills 100 tokens a second into a 100-token bucket, with an order POST costing 10 tokens — so 10 orders a second, and 10 back to back. A request over the limit is answered HTTP 429 and not processed. The pairs of one run execute concurrently (once the order-mapping check below has given a verdict), so `trader.py` sends every order and collateral-transfer POST through one shared pacer set to `config.ORDER_WRITES_PER_SECOND` (8) in bursts of at most `config.ORDER_WRITE_BURST` (8), leaving 20% of the Basic budget for writes the bot does not see. A higher usage tier allows more (`GET /account/limits` names the account's); the two constants can be raised only up to that tier's write budget divided by the order's token cost (10). Pacing only delays a request: each is still sent exactly once, and a 429 that comes back anyway is handled like any other error response. The first live run (2026-09-28) sent 14 orders inside one second without it, and two YES legs were rejected with 429 and rolled back.

Pacing must not stretch the moment a pair is exposed — its NO leg filled, its YES leg not yet sent — so a pair's hedge writes go ahead of other pairs' opening NO legs. A pair's NO leg waits until two tokens are free together, sends one and holds the other, and its YES leg is sent on the held place with no wait. An unwind takes a place in the pacer's hedge lane, served before every waiting NO leg, so it waits only for the refill of one token (1/8 s) for itself and one more per unwind already ahead of it. A held place counts against the bucket until it is sent or given back, which keeps a YES leg sent late inside the limit; it is given back at once if the NO leg's POST raises, since working out whether that leg filled can take a minute of retried reads.

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

Dependencies are declared in `pyproject.toml`: `kalshi-python-sync` (pinned to `3.2.0` — do not bump, see `CLAUDE.md`), `schedule`, `tabulate`, `cryptography`, `python-dateutil`, `openpyxl`, `plotly`, `pandas`, `numpy`, `scipy`, `yfinance`, `tzdata` (the IANA time-zone database the stdlib `zoneinfo` falls back to on a host without one of its own; it resolves `config.SCHEDULED_RUN`'s zone). The `[dev]` extra adds `pytest` and `ruff`. The `[perf]` extra adds `orjson`, which speeds up the backtest's settled-market fetch — that fetch parses tens of millions of JSON records and is CPU-bound on JSON decoding. It is entirely optional: without it the code falls back to the stdlib `json` module, and because `orjson` emits plain JSON the on-disk cache format is identical either way, so installing or removing it never invalidates a cache.

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
  live_defaults.json              ← The saved live defaults every live run starts from (gitignored;
                                    written only by defaults_server's Confirm; no file, no live
                                    run — see Live trading toggles)
  kalshi_defaults_server.log      ← The defaults server's log (auto-created; rotates 5 MB x 3)
  backtest_dashboard.html         ← Backtest HTML dashboard (rewritten by every run)
  backtest_cache/                ← Disk cache for historical data
    series_categories.json        ← Kalshi's category + tags per series (dashboard breakdowns and the live
                                    category/tag filter; refreshed weekly)
    treasury_bill_rates.json      ← The 8-week T-bill's auction yields (Treasury Fiscal Data), the
                                    dashboard's Sharpe/Sortino risk-free rate; rewritten by every
                                    successful download and read only when the next one fails
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

### Save the live defaults (do this first after upgrading)

```bash
python3 -m kalshi_betting.defaults_server --seed
```

Every live run starts from the saved live defaults, `live_defaults.json`, and
refuses to start (exit `2`) while none are saved — the Monday scheduled run
included. Run this in the checkout the scheduler runs from: it starts a local
server on `127.0.0.1:8765` and opens a confirmation page proposing the seed
values (tier floors off, spread band 0–0.5, `k` 0.80, a 10% per-trade cap, a
20% same-title cap, any category or tag). Click Confirm to save them; closing
the tab saves nothing. Ctrl-C stops the server. Without `--seed` the server
opens the backtest dashboard, when one has been built: choose a scenario in its
filter bar (spread band, tier floors, `k`, size cap, and a category or tag if
you want one) and click the bar's "Save as live defaults…" button, which opens
the same confirmation page for that scenario, with the backtest run's own
same-title cap, in a new tab (the Scenario Explorer's own selects are not what
it saves). The button stays disabled until the page has loaded the scenario on
screen, and for a scenario the run never simulated; a category or tag can be
saved only from a dashboard built with Kalshi's series listing, which is how
the live category filter files pairs. The dashboard itself writes nothing, and
the server must be running for the button's page to open. `--no-browser` only
logs the address to open. The page shows the defaults in force beside the proposed ones with every
change highlighted, and shows in red each warning a live run would log about
the proposed settings (for example, a deadline-gap range in which no pair can
trade, or one pair staking more than 20% of the balance). See
[Live trading toggles](#live-trading-toggles).

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
disproof. A position that is not exactly 0 straight after the close is re-read the same
way before the close is judged, and so is one after the exchange's HTTP 409 kill response
(`fill_or_kill_insufficient_resting_volume`), which both order steps read as a kill: the
unfillable-ask step passes on it, and the NO buy is neutral, when the account is flat.

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

After the NO buy the probe also prints an informational fee check, with no verdict:
the exchange's `average_fee_paid`, which Kalshi's API reference defines as the average
fee paid **per contract** for the order's fills, next to the bot's fee model per
contract at the order's average fill price (the limit price when the response carries
none). The model is shown three ways: before rounding (`TAKER_FEE_RATE × p × (1 − p)`);
as `config.fee_leg_exact(1, p)`, one whole contract rounded up to the cent as trade
sizing charges it; and rounded the way Kalshi rounds an order — its total fee up to
the account's balance precision, $0.0001 (or $0.01 on some accounts), with the excess
rebated later. The exchange's figure includes that rounding, which on the probe's
0.01-contract order is a large share of it: a 0.58 fill was charged $0.0200 per
contract, against $0.0171 before rounding and exactly the $0.0200 the rounded model
gives. Nothing on the line is to be scaled by the contract count. It is there to catch
an order-of-magnitude surprise in the V2 fee, not to pass or fail the run.

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

Fetches the live account balance, scans real markets, submits fill-or-kill orders leg-by-leg, and appends results to `trade_log.xlsx`, all at the saved live defaults (`live_defaults.json`) unless a flag overrides one for this run (see [Live trading toggles](#live-trading-toggles)). **With no live defaults saved, or a saved file that cannot be used, every live run — this one, dev, and the weekly scheduled run — exits `2` before it logs, connects or requests anything**; it never falls back to `config.py`. Read the 2026-09-27 decision record in `CLAUDE.md`, with its addendum, first (V0's live-contest replay of the rule `config.py` ships said STOP on drawdown, and the backtest evidence for it is anchor-fragile; the operator shipped it anyway). By the 2026-09-26 decision this also pairs and sizes same-event deadline ladders (`config.TIME_SERIES_SAME_EVENT_LADDERS`); read that constant's comment and run the dry run below first.

### Production dry-run (discover trades but don't submit)

```bash
python3 -m kalshi_betting.main --mode prod --dry-run
```

Uses the real account balance and real markets (read-only API calls only — no
orders are submitted), but still writes a simulated row per discovered trade
to `trade_log.xlsx` (status `"simulated"`), same as prod's real-order rows
just without a live fill. Confirmed live behavior — don't assume `--dry-run`
leaves `trade_log.xlsx` untouched.

### Live trading toggles

```bash
# Production dry run replaying the live rule as it stood before the 2026-09-27
# decision, for this run only (each flag that departs from the saved live
# defaults is marked "(default: X)")
python3 -m kalshi_betting.main --mode prod --dry-run --tier-floors --spread-max 1 --interval-discount 0.75 --size-cap 20 --same-title-size-cap 100

# Trade only one Kalshi category this run
python3 -m kalshi_betting.main --mode prod --dry-run --category Economics
```

The live strategy has seven toggles: whether the time-series deadline-gap tier
floors apply, the time-series spread band on `pB − pA`, the interval discount
`k`, the per-trade Kelly cap for every pair, the extra cap on same-title pairs,
and the Kalshi categories and tags a pair may trade in (`null` for any). **A
live run reads them only from the saved live defaults, `live_defaults.json` in
the repo root** (gitignored operator state, like `scheduler_state.json`, in the
checkout the scheduler runs from). With no file saved, or a file that cannot be
used — unreadable, malformed, or holding a value the bot refuses — every live
run, prod and dev, scheduled or by hand, exits `2` before it logs, connects or
requests anything, and says how to save one: from the backtest dashboard's
"Save as live defaults…" button, or from the seed values with
`python3 -m kalshi_betting.defaults_server --seed` (tier floors off, spread band
0–0.5, `k` 0.80, a 10% per-trade cap, a 20% same-title cap, any category or
tag). **After this change is deployed, save them in the scheduler's checkout
before the next Monday run**, or it exits `2` and its slot is spent (the
scheduler logs an ERROR when it starts while none are saved). A live run never
falls back to `config.py`'s toggle constants (`TIME_SERIES_TIER_FLOORS`,
`TIME_SERIES_SPREAD_BAND`, `TIME_SERIES_INTERVAL_PROB_DISCOUNT`,
`BUDGET_FRACTION`, `SAME_TITLE_SIZE_CAP`, `TRADE_CATEGORIES`, `TRADE_TAGS`):
those are the backtest's `k` and caps, and the fallback for library callers
that hand no settings.

**Since the operator decision of 2026-09-27 `config.py` ships tier floors off,
spread band 0–0.5, `k` 0.80, no per-trade cap (`BUDGET_FRACTION` 1.0) and a 20%
same-title cap, with no category or tag filter** (the seed differs in one field:
a 10% per-trade cap) — before it, tier floors on,
no band, `k` 0.75, a 20% cap for every pair and no extra same-title cap. One
pair still stakes at most 20% of the balance: a time-series trade sizes under
`1 − k` = 0.20, a same-title trade at most at its 20% cap. Read the decision
record in `CLAUDE.md` ("The live defaults of 2026-09-27 — decision record")
before a live run. It records the evidence the operator shipped these values
against, verbatim: the 365-day backtest dashboard's cell for this rule (33
trades, +78.4%, max drawdown −9.3%); V0, a weekly live-contest replay of the
rule (77 trades, +50.8%, max drawdown −28.4% against a −20% stop limit —
verdict STOP on drawdown); and the finding that the dashboard's evidence for
this rule is anchor-fragile (the backtest enters at Monday 09:00 UTC while the
scheduler runs at Monday 09:00 host-local; anchored at 09:00 PT the cell
returns +26.3% with a −24.2% max drawdown, and across 13 anchors the live
replay spans −56% to +51%). The operator shipped them despite V0's STOP and the
anchor finding.
The backtest now enters at the scheduler's own instant (see the v3 prefilter
note below); measured exactly over one corpus assembled for it, this rule's
cell returns +1.9% with a −26.4% max drawdown at 09:00 Los Angeles, against
+58.9% / −13.2% at 09:00 UTC on the same corpus.

Each toggle has a flag that
overrides it for **this run only**, in either mode: `--tier-floors` /
`--no-tier-floors`, `--spread-min X` / `--spread-max Y` (either alone keeps
the saved other bound), `--interval-discount K` (in `(0, 1]`),
`--size-cap PCT` / `--same-title-size-cap PCT` (whole percent, in 5% steps;
100 = no cap — for `--same-title-size-cap`, no cap beyond `--size-cap`, which
still binds same-title pairs), and `--category NAME` / `--tag NAME`
(repeatable; `--any-category` / `--any-tag` clear a filter the saved defaults
set). A flag not given keeps the saved value. The values are validated exactly
as the saved ones are, before anything is logged or requested — a cap off the
5% grid, `k` outside `(0, 1]`, a band floor at or above its ceiling, an empty
category or tag name, or the name `any` (which would match nothing yet read as
"any" on the `Live settings:` line — use `--any-category` / `--any-tag`, or
`null` in `live_defaults.json`) exits `2` with the reason.

A pair is filed under a category and tag exactly as the backtest dashboard's
Category and Tag selects file a trade: by market A's own series, looked up in
Kalshi's /series listing (its category, and the series' FIRST tag), or, for a
series the listing lacks, the ticker-prefix label and `General` — so
`--category Sports --tag Basketball` trades the dashboard's "Sports ·
Basketball" slice. Names match case-insensitively, and categories and tags
combine by AND. Two differences from the dashboard's options follow from
that: a `--tag` is matched under **every** category, where the dashboard's Tag
options are scoped to one (`--tag Soccer` alone also trades the Economics and
Entertainment series whose first tag is Soccer — 62 of the 210 first tags on
Kalshi's 2026-09-25 listing sit under more than one category), and a first tag
Kalshi spells in two casings under one category ("Anime Awards" / "Anime
awards") is one tag here but two options there. Several names on one axis
trade their union. The filter runs after the two pair finders and before any
order book is read; with neither set it does nothing and makes no request. A
production run refreshes the listing when its cached copy
(`backtest_cache/series_categories.json`) is over a week old; a dev run reads
the cached copy only. With no listing at all it keeps no pair (it never
guesses a category) — the run then finds no pair and exits `0`, like any run
that finds none, so the filter's WARNING is the line that says why — and a
name no series carries is logged as a likely typo.

Every run logs a `Live defaults:` line naming the saved file, when and from
what it was saved, then one `Live settings:` line naming all seven, each one a
flag moved away from the saved defaults marked `(default: X)`. A production run
that will submit orders under any such departure also logs a WARNING saying its
trades follow the flags, not the saved defaults (a dry run and a dev run do
not).
A WARNING also names any setting that empties part of the strategy — a band
ceiling below, or within float noise of, the time-series entry floor (with the
tier floors on, the larger of a deadline-gap tier and the band's floor; with
them off, the band's floor alone) — or
lets one pair stake more than 20% of the balance
(`LIVE_EXPOSURE_WARN_FRACTION`), and `k = 1`, at which no time-series trade can
size. The weekly scheduler passes none of these flags, so a scheduled run trades
exactly the saved live defaults and its line carries no mark. A production run's
separator row in `trade_log.xlsx` carries the same line (`settings: …`), with
the same `(default: X)` marks, ending `| defaults: <the saved file, when and
from what>`, so rows traded under a flag, and rows traded under one set of
saved defaults or the next, can be told apart in the workbook itself.

These flags never reach a backtest, and `backtest.py`'s own `--interval-discount`
/ `--spread-min` / `--spread-max` never reach live trading.

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

Runs from 2024-01-01 with a $10,000 simulated balance, pairing same-event ladders at the configured switch (on by the 2026-09-26 decision). Results are cached in `backtest_cache/`. The run ends by pointing at `backtest_dashboard.html` and at how a scenario on it becomes the live defaults: start `python3 -m kalshi_betting.defaults_server`, then click the page's "Save as live defaults…" button (see [Save the live defaults](#save-the-live-defaults-do-this-first-after-upgrading)).

Options:

```bash
python3 -m kalshi_betting.backtest --start-date 2023-01-01 --balance 50000
python3 -m kalshi_betting.backtest --no-cache   # rebuild the assembled market list
python3 -m kalshi_betting.backtest --max-horizon-days 14
python3 -m kalshi_betting.backtest --interval-discount 0.60   # override k for this run only
python3 -m kalshi_betting.backtest --no-sweep   # skip the k-grid re-simulation (the dashboard filter bar's k select offers one point; one k column in the scenario explorer)
python3 -m kalshi_betting.backtest --same-event-ladders     # force same-event deadline ladders ON for this run (the configured default)
python3 -m kalshi_betting.backtest --no-same-event-ladders  # force them OFF for this run (the rule before the 2026-09-26 decision)
python3 -m kalshi_betting.backtest --spread-min 0.30 --spread-max 0.60   # primary scenario's time-series spread band (backtest only)
python3 -m kalshi_betting.backtest --no-band-sweep           # skip the spread-band grid (and its tier-floors-off runs); the dashboard's scenario explorer has nothing to show, and its filter bar offers the primary band only, with the Tier floors select disabled
python3 -m kalshi_betting.backtest --no-cap-sweep            # skip the per-trade size-cap sweep: the dashboard offers config.BUDGET_FRACTION only (smaller, faster page)
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
reads the saved live defaults' `k` unless `main.py`'s own `--interval-discount`
(a separate flag of a separate command) overrides it for one live run. Omit it
(the default) to run at the configured value,
`config.TIME_SERIES_INTERVAL_PROB_DISCOUNT`. `--no-sweep` skips the extra
re-simulation across `config.INTERVAL_DISCOUNT_SWEEP`; with the band sweep on
(the default), each band is simulated at the primary `k` only, so the scenario
explorer's heatmap has a single `k` column. The empirical-`k`
calibration measurement and its log/dashboard report are unaffected by either
flag — see the sizing bullet under [Strategy change (2026-09)](#strategy-change-2026-09)
and CLAUDE.md for the full mechanism.

`--same-event-ladders` / `--no-same-event-ladders` overrides
`config.TIME_SERIES_SAME_EVENT_LADDERS` for this backtest run only. Omit both
(the default) to run at the configured value — on by the 2026-09-26 decision. Unlike
`--interval-discount`, it changes *which pairs exist* rather than how they are
priced, so it applies identically to every swept `k` and a run with it on is
**not comparable** to a baseline taken without it. It never reaches the live
finder, which binds that constant at import.

**`--spread-min` / `--spread-max` / `--no-band-sweep` — the scenario explorer
(backtest only; these flags never reach live trading, whose own band is the
saved live defaults', or `main.py`'s own `--spread-min` / `--spread-max` for
one live run).** By default — and on the live path until
the 2026-09-27 decision turned the tier floors off and set a 0–0.5 band — the
time-series finder's own directional price filter is the only gate on a
pair's YES-price gap (`pB − pA`): a floor tiered on the deadline gap, and no
ceiling. `--spread-min X` / `--spread-max Y` (each in `[0, 1]`) layer a
BACKTEST-only spread *band* on top of that for the **primary** scenario —
the floor is `max(deadline-gap tier, X)` and the ceiling is `Y`. The raised
floor also tightens the leg-price-sum ceiling to `1 − max(tier, X)`, exactly
as the live rule's entry floor sets `1 − floor` (`1 − tier` with the tier
floors on; `1.0` at the shipped floor of 0 with them off), so a band is
stricter than a YES-gap floor alone. Either flag may be given alone, and omitting both runs the configured
default band (`config.BACKTEST_DEFAULT_SPREAD_BAND`, `(0.0, 1.0)` — no band,
with the tier floors on: exactly the live rule until the 2026-09-27 decision,
and no longer the live rule since — the run's `Live time-series rule (saved
live defaults)` log line and the dashboard's `Live rule (saved live defaults)`
header line name the cell of the band sweep's grid that holds the saved rule:
on the seed values, band 0–0.5 with the tier floors off. Both lines also say
when the saved `k` or caps differ from the ones this backtest sized at, which
are `config.py`'s, and with no usable live defaults saved they say none are
recorded). The resolved floor must be strictly below the
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
no-band band's entry pass scans every pair — once per tier setting, since
the tier-floors-off runs below have a no-band pass of their own: every other
band rescans just the pairs that entered there at its own tier setting,
which cannot change its entries because a band only ever tightens the entry
tests). An "All" cell is every entry at its
band, same-title included (it is the run's actual result); a "Time-series"
cell is every time-series entry, same-title excluded — and it is the one the
explorer's heatmap, banner and equity curve show, because the band and `k`
act on time-series pairs only and a same-title result would dilute the
comparison. "Ladders" and "Cross-event" split the time-series entries
further, and "Same-title" is simulated once, independent of band, `k` and
the tier floors below; every population is labelled on the page.
`--no-band-sweep` skips that grid, and the tier-floors-off runs below with
it: the primary scenario still runs, but the dashboard's "Scenario
Explorer" section has no scenarios to show, and its filter bar offers the
primary band only, with the Tier floors select disabled. The heatmap's
metric menu always offers nine metrics, among them the Sharpe ratio and
`k̂ − k` (each band's pooled empirical `k̂` minus the column's `k`; it and
the `k̂` view are both red where `k̂` exceeds `k`, as on the cards). With
the size-cap sweep on (the default, `--no-cap-sweep` the opt-out) the
explorer also offers every per-trade size cap, each cap's figures simulated
as the dashboard is built and shipped in a block of its own. The explorer's
fragility banner — one per size cap — reports how many band x `k` cells were computed, what share of the
time-series cells had a positive return, and the split-half rank correlation
of their returns — the point being that the best of many correlated cells
overstates what you should expect, so read any cell with that banner on
screen rather than from one flattering cell (the banner's "best of N"
counts the cells that have a time-series entry, not the whole grid). A
split-half half with no entries reads "—" rather than a 0% return and is
left out of that correlation. One split date serves every cell — the median
entry date of the primary band's time-series entries (each pair's first
qualifying Monday, which no `k` moves; the lower of the two middle dates on
an even count) — so a half is empty
either when at least half of those entries share the first entry date (the
backtest log warns when that happens) or, at any other band or on the "All"
population, when all of that cell's own entries fall on one side of it. A
pair stays in the half its first qualifying Monday falls in, and a
first-half pair is simulated on its Mondays before the split only, so no
first-half trade is dated in the second period; a first-half pair whose
only Kelly-passing Mondays fall on or after the split trades in neither half.

**Tier floors off (backtest only).** With the band sweep on, every band
where a deadline-gap tier floor binds — a floor below a tier, which on the
shipped grid is floors 0, 0.20 and 0.25, 18 of the 36 bands — is entered and
simulated a second time with the two tier floors (0.15 for deadlines up to
15 days apart, 0.30 for 16–30) not applied. "Off" means the band's own floor
is the only floor: `pB − pA ≥ floor` and `pA + nB ≤ 1 − floor`, the
leg-price-sum ceiling staying tied to the floor exactly as it is for a band.
Nothing else moves: the 30-day deadline-gap cap, the band ceiling, the
live-quote and fee checks, the Kelly gate, same-event ladders, same-title
pairs and the primary scenario are the same, and so is every tier-on figure
the run reports. At a floor of 0 both tiers go, leaving `pB > pA` (a Monday
whose spread is not strictly positive has no in-between mass to dispute and
is refused, a check that never bites with the tiers on) and the fee check;
at 0.20 and 0.25 only 16–30-day pairs can change; a band whose floor
is 0.30 or more enters the same pairs either way, so it is not run twice. The
tier-off cells come back beside the tier-on grid, same shape and same split
date, with their own `k̂` per band, at the run's own per-trade size cap —
and, with the size-cap sweep below, at every other cap too (a second, lazy
size-cap sweep over them, seeded from their own results). The dashboard's
filter bar shows them through its Tier floors select: set to off, each band
a tier binds at shows its tier-off "all" run at the `k` chosen — the trade
sections, the header's trade count, the `k̂` chart and the `k̂` cards all
switch to it, at every size cap (without the size-cap sweep, or if its
tier-floors-off half cannot be simulated, at the run's own cap only, and the
bar says any other was never simulated) — and the scenario explorer's own
Tier floors select follows it, switching to the tier-off cells at every `k`
and cap (its Same-title row is the same either way; with the tier floors off
it refuses only a cap it holds no tier-off data for, on its status line) —
while in the bar a band whose floor is
0.30 or more shows its one run under both settings, at every size cap. They have no flag of their own:
`--no-band-sweep` skips them with the grid. A tier-off result is the model's
own extrapolation below the tier thresholds — nothing about it reaches the
live bot, whose own tier floors are the saved live defaults' (off in the seed,
so on the seed values its band 0–0.5 cell here IS the live rule's entry rule;
`main.py --tier-floors` restores them for one live run).

Kalshi serves at most 5,000 hourly candles (~208 days) per candlestick
request and rejects a longer one with HTTP 400, so a ticker's price series —
requested from the later of `--start-date` and the market's own open — is
fetched in as many requests as its window needs and merged; a long window no
longer loses the markets that close late in it. (Until 2026-09-23 it did:
each window was one request opened at `--start-date`, so on the default
`--start-date 2024-01-01` every ticker closing more than ~207 days after the
start got no candles and could never enter.) One caveat: **Picking a
scenario changes nothing the live bot does by itself:** live trading reads its
own toggles — the tier floors and the spread band, `k` and the two per-trade
caps, and the category/tag filter — from the saved live defaults
(`live_defaults.json`), and the ladder switch from `config.py`, never from a
backtest run, so applying a chosen scenario live is saving it as the live
defaults (the filter bar's "Save as live defaults…" button, with
`python3 -m kalshi_betting.defaults_server` running, then Confirm on the page
it opens), or, for one run, `main.py`'s matching flags (see
[Live trading toggles](#live-trading-toggles); the filter bar's Tag option
"C · T" is `--category C --tag T`). The scenario explorer itself is a backtest reporting feature:
`min_price_diff_for_gap()` layers the band's floor through its `spread_min`
keyword and drops the tiers only on an explicit `tier_floors=False`, and no
live module (`scanner.py`, `strategy.py`, `trader.py`, `main.py`) calls it,
a band helper or a toggle constant directly — the live spread rule reads
those toggles through one `config.LiveSettings` per run (see `CLAUDE.md`'s
`test_ast_live_path_reads_toggles_only_through_live_settings`; live sizing
and enrichment's affordability bound read `k` and the per-trade caps from
that same object), and its
spread tests agree with the backtest's `_find_entry` on every spread
(`TestLiveBacktestSpreadParity`).

**`--no-cap-sweep` — the per-trade size-cap sweep (backtest only; live
sizing reads the saved live defaults' caps, through its run's `LiveSettings`,
unless `main.py --size-cap` overrides one for one run).** Every simulation above sizes a trade at most at the run's
own per-trade Kelly cap, `config.BUDGET_FRACTION` (a share of the checkpoint's
opening balance: 1.0, no cap, since the 2026-09-27 decision, 20% before it),
and a same-title trade at most at `config.SAME_TITLE_SIZE_CAP` as well (20%
since that decision, 1.0 — adding no cap — before it) — the same per-pair cap live sizing
applies, through the one `config.pair_size_cap`. Each cap below is the cap
for every pair; a same-title trade stays under `SAME_TITLE_SIZE_CAP` at all
of them, so "no cap" leaves same-title capped whenever that constant is
below 1. The cap sweep is **on by default** and offers every other cap of
`backtester.SIZE_CAP_SWEEP` — 5% to 95% in 5% steps, plus "no cap" (full
Kelly; 100% is the same run, since Kelly's fraction never exceeds 1) — but it
simulates **nothing** during the run: the result carries a lazy
`backtester.CapSweep`, and each (band, `k`) cell is simulated at every cap
only when a report reads it, because keeping every band x `k` x cap x
population result would take gigabytes. Every cap at or above a result's
largest uncapped Kelly fraction (taken over every Monday a pair may be traded
on, not only the one it trades on) sizes its trades identically (a time-series
trade at its Kelly fraction, a same-title one at the smaller of that and
`SAME_TITLE_SIZE_CAP`, whatever the cap), so those caps
share one simulation — the run's own result itself when its own cap is at or
above that fraction (always, at the shipped no-cap `BUDGET_FRACTION`), or else
the first larger cap's fresh simulation (a
result sized under a cap BELOW its own peak does not size as the larger caps
do). The dashboard is that report: its filter bar's Size cap select offers
every cap, so building the page reads — and simulates — every cell, which
adds to the dashboard phase's time and page size. The tier-floors-off runs
above get a size-cap sweep of their own, seeded from their own results and
read the same way (one never seeds the other's cells), so every tier x band x
`k` x cap scenario the page offers is a real simulation; reading every
tier-off cell of the test suite's golden fixture at every cap took about
15 s (7,134 cap results simulated, 10,650 shared; measured 2026-09-27,
under the one-open-trade-per-ladder rule). `--no-cap-sweep` returns no
cap sweep at all, and the dashboard then offers the run's own cap only (the
header says which). The grid lives in `backtester.py` rather than `config.py`
by the operator's instruction for this change.

**Feasibility pre-check (BS-11).** Before any network call, `_prepare_candidates()`
— the band- and `k`-independent preparation step `run_backtest()` and
`run_backtest_sweep()` both build on — first refuses, with an error, a
`config.SCHEDULED_RUN` that cannot serve as the entry checkpoint (its run time
falls on another date in UTC, or a clock change skips or repeats it, anywhere
from `--start-date` to ten years past today), then
checks whether the `[--start-date, today]` window contains at least one
entry checkpoint — a Monday, the run's weekday (the only day the replay ever
enters a trade). If not, it logs a warning and the run returns the same empty result the
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

**Prefilter tag bump (P5): delete the old assembled caches by hand.** P5's
eligibility prefilter dropped a market that opened at or after
every Monday-09:00-UTC checkpoint it could be entered at (the checkpoint
before the v3 change below) — those dated from
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

**Prefilter tag bump (v3): the backtest enters at the live run's time.** The
backtest used to enter every trade at Monday 09:00 UTC; it now enters at
`config.SCHEDULED_RUN`'s instant, the one the weekly scheduler runs at —
Monday 09:00 America/Los_Angeles, 16:00 UTC under daylight time and 17:00 UTC
under standard time. Two replays of the 365-day window on 2026-09-27 (band
0–0.5, tier floors off, k 0.80, cap 0.20) put the difference at roughly +78.4%
at 09:00 UTC against +26.3% at 09:00 Los Angeles. Measured exactly the same day
over one corpus assembled under the new prefilter, that cell went from +58.9%
(09:00 UTC) to +1.9% (09:00 Los Angeles), the primary cell from +116.4% to
+136.3%, and the pooled empirical k̂ from 0.891 to 0.958 (see the entry-checkpoint
gotcha in `CLAUDE.md`). No backtest result or dashboard from before this change is
comparable with one after it. The
eligibility prefilter now keeps a market only if it opened before the start of
some reachable checkpoint's candle hour, and the assembled cache is keyed by
the prefilter's version and the schedule together
(`checkpoint-v3-mon0900-America-Los_Angeles`), so editing `config.SCHEDULED_RUN`
re-keys it by itself. The backtest log names the checkpoint and the UTC times
it fell at over the window (`Entry checkpoint (backtest): Monday 09:00
America/Los_Angeles ...: 16:00/17:00 UTC in this window`), and the dashboard
header names it too. Two v2 caches are orphaned:
`backtest_cache/settled_markets_2025-09-24_monday-checkpoint-v2.jsonl.gz`
(841,311,284 bytes) and `settled_markets_2026-09-17_monday-checkpoint-v2.jsonl.gz`
(511,174,809 bytes). Other checkouts still on the v2 tag read them, so delete
them by hand only once none does. The new prefilter keeps everything v2 kept
and more (markets opened on a Monday between 09:00 UTC and the run), so the
next run of each start date re-assembles its corpus from the day slices, which
are unaffected, and fetches candles for the newly eligible markets.

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

Runs the production bot once a week in a blocking loop, at `config.SCHEDULED_RUN`'s weekday and time — Monday 09:00 — on the **host's clock**, each run spawned as a `python3 -m kalshi_betting.main --mode prod` subprocess and killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` (3600s). The log also prints the equivalent `crontab` entry if you prefer cron (cron fires on the host's clock too).

**Run time and zone.** `config.SCHEDULED_RUN` also names the run's zone, America/Los_Angeles, where 09:00 is 16:00 UTC under daylight time and 17:00 UTC under standard time. The daemon still fires on the host's clock rather than converting: the `schedule` library's own zone support goes through `pytz`, whose America/Los_Angeles table has no daylight time after 2037, so from 2038 it would fire at 10:00 all summer. Instead, at startup the daemon checks that the host's clock places the next 104 weekly fires at the schedule's own UTC instants. **Keep the host's time zone set to America/Los_Angeles.** On a host whose clock keeps different UTC offsets over those two years (a UTC host, or a zone without the same daylight-saving rules), the daemon logs CRITICAL, naming the first mismatched date, and keeps firing at 09:00 on the host's clock; a zone with the same rules under another name (`US/Pacific`, `PST8PDT`) passes. The backtest enters every trade at the same instants, so on such a host it no longer replays the daemon's runs, and the CRITICAL says so. To move the run, edit `config.SCHEDULED_RUN` — its weekday, hour, minute and zone form one value, and the edit also moves every backtest entry and re-keys the backtest's assembled cache; the backtest refuses a schedule whose run time falls on another date in UTC or on a clock change. A time the zone skips at a clock change (02:30 on a spring-forward Sunday, say) cannot fire at one UTC instant on any host, and the startup check logs that CRITICAL against the schedule, not the host.

Every scheduled run trades exactly the saved live defaults (`live_defaults.json` in the checkout the daemon runs from; see [Live trading toggles](#live-trading-toggles)). With none saved, or a saved file that cannot be used, each run exits `2` (`Job failed (exit 2)`) and its slot is spent, so the daemon logs an ERROR as soon as it starts while that is so, naming how to save them.

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
| `40` | `EXIT_TIME_SERIES_SKIPPED` — production only: a held market could not be identified, so the run could not tell which ladders it already holds and made **no time-series trade**; it still searched for and traded same-title pairs. `20` wins over it when a trade also needs a human. The scheduler logs it as an ERROR (pointing at the ERROR in `kalshi_arb.log` that names the market) and counts the weekly slot as done — a retry an hour later would most likely fail the same lookup |
| `2`  | Usage error — argparse's code for an invalid flag, a `config.ORDER_API_VERSION` other than `"v2"`, or no live defaults saved, or a saved `live_defaults.json` that cannot be used, refused before logging is configured or any request is made. A scheduled run passes no toggle flag, so a `2` there means `config.ORDER_API_VERSION` is not `"v2"`, or no live defaults are saved or the saved file is refused: `kalshi_arb.log` records nothing, and the reason is on stderr, which the scheduler logs under `Job failed (exit 2)` (the scheduler also logs an ERROR at start while none are usable). Like `1`, it satisfies the weekly slot |
| `1`  | Unhandled exception — the interpreter's default for a crash; not part of the contract above |

The constants live in `config.py` (`EXIT_OK` / `EXIT_SKIPPED_LOW_BALANCE` / `EXIT_TRADES_NEED_ATTENTION` / `EXIT_NO_TRADEABLE_SHARDS` / `EXIT_TIME_SERIES_SKIPPED`) and the scheduler maps each to a distinct log level and message, so a low-balance skip or a manual-review run is never logged as "completed successfully". Exit `30` additionally means the weekly slot was **not** satisfied: nothing was scanned, so the run must never count as the week's scan — see the blind-run retry above.

---

## Testing

```bash
python3 -m pytest tests/ -v      # run the test suite
python3 -m ruff check kalshi_betting/   # lint check
```

Tests run fully offline against `unittest.mock.MagicMock` clients — no real Kalshi API calls. The defaults server's socket tests (`tests/test_defaults_server.py::TestOverASocket`) bind a loopback port and skip where a sandbox refuses that; CI binds. Its Confirm-script tests (`::TestConfirmScript`) run the page's script under node or macOS's `jsc`, and skip when neither is present. `.github/workflows/ci.yml` runs both commands on every push to `main` and on EVERY pull request, whatever its base branch — the `pull_request:` trigger carries no branch filter. Both must pass before merging.

---

## Sandbox Note

The Kalshi sandbox endpoint (`https://demo-api.kalshi.co`) requires a **completely separate account** registered at [demo.kalshi.co](https://demo.kalshi.co). Your production API key will return `401 Unauthorized` on the sandbox endpoint — this is intentional by Kalshi.

To use dev mode with real sandbox authentication, register a sandbox account, generate its API key, and add it as `"dev_api_key"` in `secrets.json`. Without a sandbox key, dev mode still fetches real sandbox market data (for scanning) but skips the held-positions and balance checks that require authentication.
