# Kalshi Arbitrage Bot

An automated pair-trading bot for the [Kalshi](https://kalshi.com) prediction market platform. The bot finds pairs of related prediction market contracts where one is mispriced relative to the other, sizes positions using the Kelly criterion, and submits fill-or-kill orders leg-by-leg — with automatic rollback if the second leg doesn't fill. A separate backtesting pipeline replays the same strategy on the full history of settled Kalshi markets and generates an interactive HTML performance dashboard.

---

## How It Profits

Kalshi markets are binary contracts that pay $1 if a question resolves YES and $0 if it resolves NO. The bot exploits two specific pricing anomalies:

**Time-series pairs:** Two contracts asking the same question about the same outcome at two different **deadlines** (e.g. "Will BTC exceed $80k by March 2025?" and "Will BTC exceed $80k by June 2025?"). The "by" matters: the strategy only works between *cumulative-deadline* markets, where the earlier deadline's event is contained in the later one's, and the bot refuses a pair whose wording does not say so (see below). "The same question" means both the date-stripped title *and* the outcome label (the subtitle) match: a daily price family lists dozens of strikes under one shared title, and keying on the title alone paired the highest earlier strike against the lowest later one — two different questions traded as one. The later deadline gives more time for the event to occur, so the later contract's YES price normally sits above the earlier one's — and the gap between the two YES prices roughly measures the market's implied probability that the event first happens *between* the two deadlines (the bot's forecast reads that probability at the two markets' midpoints instead; see **Sizing** below). When the later contract is priced *higher* than the earlier one by at least a required margin, the bot disputes that in-between probability: it buys YES on the earlier contract and NO on the later one. There are exactly three ways such a pair can settle:

- The event happens by the earlier deadline (both resolve YES): the YES on the earlier contract pays out — **win**
- The event never happens by the later deadline (both resolve NO): the NO on the later contract pays out — **win**
- The event happens in between (earlier NO, later YES): both legs expire worthless — the full stake is lost — **loss**

The earlier contract resolving YES while the later resolves NO is impossible for a genuine cumulative-deadline pair ("by March" YES implies "by June" YES). Both legs must therefore be *cumulative-deadline* markets, and the bot now checks that rather than assuming it: a pair is only formed when both legs' wording states a "by \<date\>" deadline and the two deadlines differ — compared as normalized text, not parsed calendar dates, so one deadline spelled two ways reads as two, and a truncated spelling can read two deadlines as one (mostly fixed; a few residuals remain, see `CLAUDE.md`). Kalshi also lists **snapshot** markets — "what is X **on** \<date\>", "X **in** \<month\>" — whose probabilities do not nest (SOL ≥ $180 on Sep 14 does not imply SOL ≥ $180 on Sep 18), and stripping dates from titles collapsed such a family into a single time-series group, so it used to be eligible to trade. The backtester still treats a pair that settled earlier-YES/later-NO as a premise violation and excludes it with a counted warning rather than paying it — that counter is now a backstop behind the wording check, not the only thing watching. This trade is **not risk-free**: at market prices its expected value is zero minus fees, and it profits only if the market systematically overstates the in-between probability. The bot sizes it on the operator's estimate `k` of how much of the market-implied in-between probability is genuine, read from the saved live defaults (0.80, i.e. 80%, on the seed values; `TIME_SERIES_INTERVAL_PROB_DISCOUNT` in `config.py` is the backtest's `k`) — a hand-set estimate, not a measured quantity.

**Same-title pairs:** Two contracts on different event tickers but with the *identical* title and subtitle (i.e. asking exactly the same question), whose two markets close within an hour of each other (see below for why both conditions are needed). If their prices diverge by 5% or more, the bot buys NO on the expensive one and YES on the cheap one. Both contracts should co-resolve, so the trade is priced on a 95% co-resolution assumption — high, but **not risk-free**: the 5% that does not co-resolve is a total loss on one leg. The subtitle here is the outcome label that distinguishes markets sharing one question title (e.g. two candidate names under "Who will the next Pope be?"); the API stopped sending a `subtitle` field in 2026-08, so ingest now sources it from `yes_sub_title` — without that discriminator, two *different* outcomes would be paired as if they were the same contract.

**Time-series pairs** must also sit on **different event tickers**, with one exception: a same-event cumulative deadline **ladder**, where Kalshi lists one question's several deadlines as separate markets inside a single event ("Will SpaceX launch another Starship before Sep 23, 2026?" and "… by Oct 16, 2026?" are both `KXSPACEXSTARSHIP-14`). Two such rungs are the time-series premise itself, and the time-series finder pairs them while `TIME_SERIES_SAME_EVENT_LADDERS` in `config.py` is on, as it is by operator decision of 2026-09-26 — ordering the legs and choosing the price tier by the two **stated** deadlines rather than by the exchange close times, which a settled event gives every rung alike. It shipped **off** and was turned **on** by operator decision on 2026-09-26: read the constant's comment, which records why ladders nest (the impossible earlier-YES/later-NO settlement never occurred in 1,821 archive ladder pairs, against 13% of cross-event ones), what it deployed on the 2026-09-22 snapshot at the live toggles then shipped (98% of a $10,000 balance into six trades, at a market-implied expected value of −31% — which is the later leg's own bid-ask spread plus taker fees, not the strategy's disbelief in the market), the decision rule it shipped behind and its unmet result (k̂ 0.87 in the widest-spread band, measured on the YES-ask gap before DR-78, where the rule asked for materially below 1; live sizing assumed k = 0.75 then, 0.80 since the operator decision of 2026-09-27), and the 365-day ladder backtest (+111.8% at k 0.75, resting on one trade: the run re-simulated without its event returns −92.8%). **Both paths implement ladders**: the backtester forms the same pairs from a per-event sub-pass and orders and gaps them on the same stated deadlines, so a ladder-enabled backtest measures the strategy a ladder-enabled live run would trade — with one standing caveat: both hold at most one open time-series pair per ladder but pick the rung differently (each live run keeps one pair per group, ranked tradeable-first then by largest price gap before sizing; the backtest takes, Monday by Monday, the Kelly-passing pair with the largest entry-time monthly ratio whose ladder is free), so they can replay different rungs of the same ladder. The setting reaches the backtest log *and* the dashboard's own page header (`Primary spread band: … | same-event ladders: on / off / not recorded`, a recorded on/off followed by whether it matches `config.py`'s switch, printed above every section), so a ladder-enabled run's dashboard is no longer indistinguishable from a switch-off one. When a run's setting departs from `config.py`'s switch, the header and the log say the run does not replay the live rule. A second header line, `Live rule (saved live defaults): …`, names the LIVE time-series rule itself — the saved live defaults' tier floors and spread band, plus their category/tag filter when one is set, read once before the run's fetch (never `main.py`'s per-run overrides, which a backtest cannot see; with no usable live defaults saved it reads `Live rule: none recorded — …`) and whether the primary scenario already is it, the filter bar can show it (naming the bar's own Spread band and Tier floors options and, for a category/tag filter, what to tick in the Category and Tag menus, such as "Category Economics, Sports and Tag Sports · Basketball" — or saying that no scenario of this run filed a pair under the filter), or it sits off this run's own grid — plus the same-title cap: at every size cap the page offers, same-title rows are capped at the lower of that cap and `config.SAME_TITLE_SIZE_CAP`, and the line gives the figure at the run's own cap. When the saved defaults' `k` or caps differ from the ones this backtest sized at (`config.py`'s), the line says so and, where the page shows the live rule, names the filter bar's `k` and Size cap options that show it at that sizing, or what the bar cannot show. The backtest log states the same verdict on its `Live time-series rule (saved live defaults): …` line, which also says when the saved defaults add to held pairs, something this run's primary never does (the dashboard's Add to held pairs views come from the separate add-on family below). `backtest.py --same-event-ladders` / `--no-same-event-ladders` overrides the switch for one backtest run — `--no-same-event-ladders` replays the rule as it stood before the switch was turned on — and the live finder binds the constant at import, so a backtest override can never reach it. The pairing rules below are unchanged by that switch; the backtest figures quoted with them were measured with it off.

The two event tickers must also belong to **different event series** (a same-event ladder's two rungs share one event and therefore one series; the series rule bites on a time-series pair only when the wording is identical across that series, which a dated ladder's is not). A Kalshi event ticker is a series prefix followed by an instance stamp, so two events of one series are two instances of one recurring fixture — two ball games, two 15-minute price windows, two multi-leg combos of different games. Multi-leg combos are the one exception to "the prefix is the series": Kalshi lists them under several prefixes that all begin `KXMVE` (`KXMVECROSSCATEGORY`, `KXMVESPORTSMULTIGAMEEXTENDED`, `KXMVECROSSCATEGORY0`, `KXMVENBASINGLEGAME`), and the whole family is treated as **one** series, so two combos under two *different* `KXMVE*` prefixes are still refused — a combo ticket's wording names its legs but never its date, so identical wording across any two combo tickets is two different tickets. Reading the literal prefix missed exactly those cross-prefix pairs: in one day's settled history they co-resolved only 68% of the time (an unconditional rate over every record in each cross-prefix wording class, with no price or liquidity filter — the measured reason the family rule exists, not a co-resolution estimate for pairs that would actually be traded). Identical wording across two events of one series is the same question asked about two *different* events, and the co-resolution assumption simply does not hold: one such pair of consecutive baseball game days was quoted 0.97 and 0.01 at the same moment. The time-series finder applies the same rule (when the wording is identical across one series, the deadline is not in the wording and there is no cumulative-deadline pair either), so the trade cannot reappear relabelled. A second, independent rule decides whether a time-series pair is a *deadline* pair at all: both legs' wording must state a cumulative "by \<date\>" deadline, and the two deadlines must differ. It reads the outcome label first, then the market title, then the event title, so a multi-choice market whose sub-contract is labelled "$80,000 by June 30" qualifies on that label alone — the rule turns on wording, never on a market's type. It is applied in the live scanner and the backtester through one shared helper for the same reason the series rule is: gating one path only would leave the other replaying the defect. Different series is a *necessary* condition, not a sufficient one: the same prop listed for two different games in two *different* leagues shares its wording too (`KXBRASILEIROB1HTOTAL-…` vs `KXUCL1HTOTAL-…`, both "Over 0.5 1H goals scored", settled no and yes on 2026-09-10), and so do a men's and a women's college basketball game between the same two schools (`KXNCAAMBGAME-…` vs `KXNCAAWBGAME-…`, identical title, subtitle and event title) or two competitions' fixtures of one matchup (Champions League and La Liga). So a same-title pair must also **close at the same moment**: its two markets' close times may be at most one hour apart (`SAME_TITLE_MAX_CLOSE_GAP_SECONDS` in `config.py`), and a close time that cannot be compared refuses the pair (DR-74). Two games close hours or days apart; one question listed by two series closes at one instant. On the 365-day backtest (start 2025-09-24, with same-event ladders off — the switch's value until 2026-09-26), 17 of its 21 trades were such men's/women's pairs, and every same-title candidate closed either at the identical instant (245 pairs, 99.2% settled the same way) or at least 1.17 hours apart (1,661 pairs, ~60.7%); the rule leaves 3 trades, for +4.8% at a −1.9% max drawdown, against 21 trades, +4.7% and −48.0% before it. It is applied in the live scanner and the backtester through one shared helper (`scanner.closes_apart`). The time-series finder deliberately carries no twin: identical wording can never form a time-series pair (see the deadline rule above), so a pair the close rule refuses cannot reappear relabelled as one. Known residuals — for example, futures that can be decided early carry a family-wide placeholder close time on the live exchange, so two such listings pass live at a zero gap — are recorded in `CLAUDE.md`.

**At most one open time-series trade per ladder.** A *ladder* here is one question asked at several deadlines: two markets are on one ladder when they share an event ticker, or ask the same question once the dates are removed (the time-series grouping key — event title, market title and outcome label — at any deadline, across events, compared without letter case, articles, quote marks or stray punctuation, so one question listed twice with slightly different wording is still one ladder). The weekly live run never trades a market it holds (bar adding to an exact held pair, below), but nothing stopped a later week opening a *different* pair on the *same* ladder while the first trade was still open, and a week-by-week replay of the live rules found exactly that: 18 trades entered while their group was still open, and $3,990 lost when stacked rungs of two ladders lost together — the second trade's in-between window sat inside the first's, so one outcome sank both. A production run now first finds the ladders its open positions are on — from the run's own market list, or by looking the market up when the list lacks it (one that has closed but not yet paid out, say) — and the time-series finder refuses any candidate with a market on one of them (except the exact held pair a run may add to, below), before it chooses each group's best pair, so a runner-up off those ladders can still trade. The portfolio step then takes at most one time-series trade per ladder, counting the ladders of every trade picked earlier in the same run, same-title ones included: a same-title pair is never refused by this rule, but its markets' ladders count. A refused pair spends no cash, so a lower-ranked pair of either kind may then fit in its place — which can in turn leave too little cash for a later same-title pair. If a held market cannot be identified, the run makes no time-series trade at all (an ERROR says so, and so does the closing "no qualifying pairs" line when nothing else is found) while same-title trades go ahead, and the run exits with code `40` so the scheduler's own log says time-series trading was skipped (see the exit codes under [Weekly scheduler daemon](#weekly-scheduler-daemon)). A ladder is free again once the account no longer holds a position on it. Dev runs hold nothing, so only the within-run rule applies there. The backtester applies the same rule as it walks the simulated Mondays — its own runs never sell (only the dashboard's Sell select does), so on each Monday it enters a time-series pair only if no trade it still holds, of either kind, has a market on that pair's ladders, and a ladder frees up on the day its market pays out. A time-series pair it skips on one Monday (its ladder busy, or too little cash left for even one contract pair — a trade the cash left can buy only in part is shrunk to it instead) is tried again on its next Monday that passes the Kelly gate, as the weekly live run would; same-title pairs keep their one-pair-per-group rule and are not tried again. Every backtest time-series figure recorded before this rule is therefore not comparable (DR-76 in `CLAUDE.md`).

**Adding to a pair the account already holds.** With the live toggle `add_to_held_pairs` on (see [Live trading toggles](#live-trading-toggles)), a production run may buy more of a pair the account already holds: exactly the same two markets, the same side on each. It first reads every held position with its side and cost, and only when that listing was read to its end, every held market was identified and Kalshi's value of the open positions was read and kept by its check does it look for *exact* held pairs — two held markets alone on one ladder, one held YES and one held NO, of equal size, with both costs reported. When one market of a held pair has paid out and the other is still open, the open one is still a held market (it keeps blocking its ladder) and, with the setting on, a *lone leg* the run may add to: a new pair that buys that market on the side held there, beside a market the account does not hold that sits on no other held ladder, staked at that one market's worth today plus its own fees (the paid-out market's payout is already cash). The log names it as `Held market to add to (its partner has paid out): …`. Anything else (unequal counts after a partial unwind, a third held market on the ladder, a cost the listing does not report) is never added to and stays blocked as any held market is, and a listing cut short, a held market that cannot be identified, or, while anything is held, Kalshi's value of the open positions unread or refused (the portfolio value would then count only the cash) means no add-on that run, with a WARNING saying which. The finders let an exact held pair through, and only as that same pair buying the same sides; every other candidate touching one of its markets is refused. The add-on is judged by every rule any other trade is, and Kelly sizes the **whole position**: the held pair, valued at today's prices (each market's contracts at the ask of the side held there, or at what they cost when a market has no usable ask) plus the fees paid for it, and the new contracts with their fees together stake at most `min(f*, cap)` of the portfolio value, and like every trade the add-on spends at most the cash — so no add-on stakes more than a new pair would, and the whole position stays within that share of the portfolio value. The fees count because the portfolio value has already lost them: at the prices it was bought at, a pair bought at its full share adds nothing. A held pair that has gained value gets a smaller add-on, one that has lost value a larger one. A held pair with no room left even at the per-trade cap is left out, so its markets stay blocked. Just before the order is sent the trader checks again that both markets still hold exactly that pair, and sends nothing if not; any rollback unwinds only the new contracts, and every alert about such a market names what it held before the pair. The log names each held pair it may add to, with its cost, the fees in that cost and its worth at today's prices, and marks every add-on "adds to N held". A saved file without the setting reads it as **off**, though `config.py` and the seed ship it on. To turn it on for every run, follow the order under [Save the live defaults](#save-the-live-defaults-do-this-first-after-upgrading), which checks a dry run with `--add-to-held-pairs` against the fills in the Kalshi UI before adding goes on; the dashboard's save button sends the choice only from a page that shows it, and otherwise keeps what is saved.

**Selling a held position at a share of its potential profit.** With the live toggle `sell_at` set (off unless saved; see [Live trading toggles](#live-trading-toggles)), a production run first sells every held position that has stayed at or above that share of its potential profit at each of three daily checks, and only then looks for pairs to buy. It is the rule the backtest's Sell select simulates (see [Selling at a share of potential profit](#selling-at-a-share-of-potential-profit)), applied by `seller.py` to the real account. A position is an exact held pair, or one held market whose partner has already paid out (found among the account's settlements: a market the account no longer holds, held on the other side in the same count and asking the same question, that paid out within the last 60 days). Anything else (three or more markets on one ladder, unequal counts, a missing or ambiguous partner, a cost or ladder the bot cannot read) is never sold, and the log says why. Realized profit is what selling the held markets at the bids would return after the sale's fees (plus the partner's payout) less what the whole position cost; potential profit is what it pays if it wins less that cost. The check at the run's moment walks each market's real order book for its count; each of the two earlier days reads the market's last hourly candle bid in the 24 hours before that check, in any size. With `sell_min_days` set, a position with fewer days left before its last market stops trading is not sold. Each held market is sold by one reduce-only, immediate-or-cancel order for the contracts held there (a held YES by an ask, a held NO by a YES bid), priced no worse than the walk reached; a pair is two orders, the thinner book first and then exactly as many contracts of the other as the first sold (at most one tick below its walked price), so what is left is still an exact pair. When an order's reply does not say how many it sold, the account's position decides, read again after pauses of 1, 2 and 4 seconds while it shows less than the whole order, and the run then checks its positions listing, read again after the same pauses while such a count came up short: a listing that shows more sold than the run recorded is a CRITICAL. The listing reads the same ledger, so one that still trails after the last pause leaves the short count standing, unflagged (a recorded residual; see `CLAUDE.md`). A sale the bot cannot account for, that leaves a pair uneven, or that it recorded short, logs a CRITICAL and makes the run exit `20`. The sales go into `trade_log.xlsx` before anything is bought (a live run writes them once it has read its positions and cash back); the cash they free funds the same run's buys, the markets sold are not bought back in it, and no position picked for sale is added to in it, sold or not. The dashboard's Live trading tab reads those rows as the run's sales, never as purchases, and the sale orders as sales by hand (see [Live trading tab](#live-trading-tab)). A dry run sends no order: it adds each sale's estimated proceeds to the cash its buys are sized on (the portfolio value never below that cash) and says what it would have sold. The rule differs from the backtest's in a few recorded ways (the earlier days' checks read one candle bid in any size, the days rule reads each market's scheduled close, the fee is charged once per market, and a ladder with three or more held markets is never sold; see the live-selling paragraph in `CLAUDE.md`). **Before turning it on, run `python3 -m kalshi_betting.v2_probe --ticker <TICKER> --step yes-close` and see it PASS**, then follow the order under [Turn on live selling](#turn-on-live-selling).

The required price gap for time-series pairs is tiered by how far apart the two deadlines are: 15% for deadlines ≤ 15 days apart, 30% for 16–30 days — wider gaps leave more room for the event to genuinely land between the two deadlines, so more of the market-implied in-between probability is real rather than mispricing, and a bigger gap must be demanded before disputing it. Deadlines more than 30 days apart are never considered. See `min_price_diff_for_gap()` in `config.py` for the exact thresholds. Those tiers were the live default until the operator decision of 2026-09-27 and are not a fixed rule: the saved live defaults' `tier_floors` turns them off (the gap must then only be positive, and clear the spread band's floor), and their `spread_band` adds a band on the gap — a floor layered on the tier and a ceiling — which the scanner applies to the scanned prices, again to the fresh order book, and once more just before submission, through one definition (`config.time_series_spread_refusal`). The seed saves the tiers **off** and the band **0–0.5**, the values `config.py`'s `TIME_SERIES_TIER_FLOORS` / `TIME_SERIES_SPREAD_BAND` have held since that decision: any positive gap up to 0.5 qualifies, at any deadline gap up to 30 days (see [Live trading toggles](#live-trading-toggles), and the 2026-09-27 decision record in `CLAUDE.md`, before a live run). A time-series pair whose later market has no YES ask on its book right now, or either of whose books is crossed (its YES ask below its own YES bid), is dropped rather than sized.

In both cases, Kalshi charges a taker fee per contract leg. The bot only executes trades where the profit margin exceeds all fees after applying order book depth to confirm the gap exists in real liquidity. Depth is priced at the **margin**: a single pair can never consume more than its per-trade cap of the portfolio value Kelly sizes on (the saved live defaults' per-trade cap, and for a same-title pair their same-title cap too), nor, for a time-series pair, more than `1 − k` of it (a bound that holds because the scanner drops a pair either of whose books is crossed), nor ever more than the cash on hand (`config.kelly_budget`), so averaging a liquid market's entire book would price every trade against levels it could never reach. On the seed values the per-trade cap binds first, so one pair of either type stakes at most 10% of the portfolio value (`1 − k` is 20% there); at `config.py`'s `k` and caps, which a backtest sizes at (no per-trade cap since 2026-09-27, and a 20% same-title cap), the limit is 20% either way. The scanner bounds its average at the most contracts one trade's budget could ever buy — the largest Kelly fraction the sizer can reach for that pair type under the run's settings (the same `k` and caps, portfolio value and cash the sizer is handed), over the cheapest level — and first cuts the book at the first level with no edge left after fees (the pre-execution check cuts the freshly fetched book the same way before counting reachable depth); the sizer then searches the book for the largest contract count whose own fill price still justifies it. The portfolio step then spends the cash itself, in whole cents, at what each trade's fill-or-kill orders can draw at their limit prices, and a trade that no longer fits the cash left is shrunk to the largest size that does rather than dropped (dropped only when not one contract pair fits).

### Strategy change (2026-09)

Until September 2026 the time-series strategy traded the opposite way round: it fired when the *earlier* contract was priced higher than the later one, bought NO on the earlier and YES on the later, and described the result as a risk-free position with three profitable outcomes. That direction has been inverted, and the time-series bet is now a **directional trade, not an arbitrage**: the bot buys YES on the earlier contract and NO on the later one when the later is priced at least the required margin above the earlier, wins if the event happens by the earlier deadline or never happens by the later one, and loses the whole stake if it happens in between (see the three outcomes above). Same-title pairs are unchanged.

What that means for anyone reading the outputs:

- **Sizing.** The Kelly sizer models the probability of profit as `1 − k × the mid spread`: the later market's midpoint minus the earlier one's, a midpoint being halfway between a market's YES ask and its YES bid (DR-78 in `CLAUDE.md`). The rules that pick which pairs qualify read the YES-price gap (later YES ask − earlier YES ask) instead, which is the mid spread plus half the later book's bid-ask width less half the earlier book's — a cost of trading rather than a view of the odds. The chance of profit is the same at every order size: the prices the orders pay set the cost, the fees and Kelly's payoff term, never the chance. `k` is read from the saved live defaults (the seed saves 0.80; `config.py`'s `TIME_SERIES_INTERVAL_PROB_DISCOUNT` holds 0.80 since the operator decision of 2026-09-27, 0.75 before it): the bot believes 80% of the market-implied in-between probability ("prices converge by 20%"; 75% and 25% at 0.75). At `k = 1` — taking the market at face value — or under an independence model, Kelly is zero or negative for every pair and the strategy never trades, so `k` is what makes it fire at all; it is a hand-set operator estimate. With `k = 0.75` — the value these figures were measured at, under the 20% `BUDGET_FRACTION` cap shipped until 2026-09-27 — the Kelly fraction sat well below that cap for typical gaps (reading off an earlier contract at 0.30 on a tight book: about 5% at the minimum short-tier gap, about 16% at a 0.30 gap — the figure is strongly dependent on that earlier price, running to about 14% at the same tier with the earlier contract at 0.10), so Kelly itself sized and differentiated pairs and the cap only bound for gaps of roughly 0.45 or more. Since that decision `config.py` ships no per-trade cap, and `1 − k` (0.20 at `k` 0.80) bounds every time-series trade instead (the seed adds a 10% per-trade cap); a wide order book drives Kelly negative and the pair is skipped. Kelly's payoff-per-dollar term divides by the cash actually at risk — both legs' cost **plus** the taker fees, since a losing pair forfeits the fees too — so a positive Kelly fraction means positive expected value under the continuous fee approximation the gate prices with. That approximation sits slightly below the ceiling-rounded fee the trade is actually charged, so a pair right on the boundary can still be a few tenths of a cent EV-negative at single-digit contract counts; the separate "profit if won must be positive" check bounds that residual. Because a pricier later contract is normal term structure, far more time-series candidates qualify than before. `k` is no longer only a hand-set guess — it is now *measurable* against settled history: see [Backtest](#backtest) for `--interval-discount` and the dashboard's empirical-`k` report. The live bot reads `k` from the saved live defaults (through the run's one `config.LiveSettings`, which hands the sizer and the scanner's depth bound the same `k`) unless `main.py`'s own `--interval-discount` overrides it for one run — see [Live trading toggles](#live-trading-toggles); nothing writes the measured value back on its own.
- **Trade log.** `trade_log.xlsx` keeps its 18 columns, but new workbooks head the count columns "x — A leg" / "y — B leg", the cost column "Total Cost incl. fees ($)" (the value is fee-inclusive in every workbook — only the header is new, and every row's Notes carries `fees=$x.xx` so a row under an old header is still readable) and the profit column "Profit if won ($)" (an existing workbook keeps its old header row) and every row's Notes cell is prefixed `[<pair_type>: <SIDE_A> A / <SIDE_B> B[ nB=…] fees=$x.xx[ adds to N held]]` so the side traded on each market is explicit (the last part only on a trade that adds to a pair the account already holds, which the pairs table marks "(adds to N held)" too). For time-series rows the "nA (NO ask)" column is the earlier contract's best NO ask as the scan read it, for reference only — the traded NO price is the `nB` in the Notes prefix, and the column is not the book quote the forecast reads (only the "Trade computed" log line prints the mid spread). The dev-simulation candidates sheet gains an "nB (NO ask)" column, and the live pairs table logged by `main.py` gains an "nB (NO)" column and labels its profit column "Profit (win)". The Excel log's market cells now carry each leg's outcome label alongside the title, and the pairs table shows that label in its own "Outcome A"/"Outcome B" columns — its market cells are cut at 40 characters, so on any title that runs past that — which the daily families this exists for all do — the appended label is the first thing lost, and two strikes of one family share a title and would otherwise render identically.
- **Backtest.** Each `BacktestTrade` records `entry_nB`; a time-series trade's profit is negative only in the in-between outcome. Any candidate whose settlement violates the cumulative-deadline premise (earlier YES, later NO) is skipped rather than paid, and the run logs one warning with the count. Since the wording check landed, that counter is a backstop rather than the only signal — the backtester refuses a snapshot pair up front through the same helper the live scanner uses — so expect it at or near zero. DR-72 widened what a non-zero count is read as: most likely a wording false negative (a snapshot or "before \<date\>"-worded recurring window read as cumulative), but also possibly legs ordered on an early REALIZED close (cross-event pairs only — a same-event ladder is ordered on its stated deadlines), or strike-blind grouping on a cache without subtitles — the warning names all three rather than pointing at one. Each run also logs, and the dashboard renders, how its eligible markets are worded: how many are worded as a cumulative "by \<date\>" deadline, how many are snapshots, and how many carry no deadline wording the classifier recognises — reported as a verdict only when the three counts actually sum to the corpus, otherwise as raw counts with no claim attached. Existing backtest caches need no refresh.
- **Not updated.** `kalshi_bot_flowchart.pdf` predates this change (it shows the old `|pA − pB|` filter) and has not been regenerated; `BUG_SWEEP_FINDINGS.md` is a dated record and is left as-is.
- **Calibrating `k` from settled history (shipped 2026-09).** `backtest.py --interval-discount K` overrides `k` for one backtest run (the live bot is untouched — it prices at its run's `config.LiveSettings`, built from the saved live defaults, or `main.py`'s own `--interval-discount` for one live run, neither of which a backtest can reach); the backtest also measures the *empirical* `k` — the realised in-between rate divided by the mean market-implied gap at the midpoints (the mid spread, the same quantity the forecast multiplies by `k`, so the two compare directly; a `k̂` recorded before DR-78 divided by the YES-ask gap `pB − pA` instead, and is not comparable), pooled and per deadline-gap bucket — logs it, and reports it in the dashboard's "Interval Discount (k) Calibration" section, whose equity curve and per-`k` table follow the page's filter bar: its `k` select switches them between every `k` in the swept grid (`--no-sweep` to skip the extra re-simulation passes), and its size cap between every per-trade cap, always at the primary spread band and with the tier floors on. The section after it breaks the same `k̂` down by Kalshi category, by tag or by spread band, following the page's filter bar, and two cards in the Portfolio Performance section show the `k̂` of the band, category or tag selected and `k̂ − k` against the `k` selected (red when the sizer sized too big). This is a recommendation only: nothing writes the measured value back to `config.py`. That same section now opens with the run's **outcome-label coverage** — how many of the run's eligible markets carried the `subtitle` both grouping keys are built from. It is shown on every run, so a clean run is confirmable rather than merely unwarned; below `config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION` it becomes a red banner beside the `k̂` card (and a one-line notice at the top of the page), because a cache written before the 2026-08-14 outcome-label ingest fix groups strike-blind and makes every figure on the page describe a different strategy from the shipped one. The remedy it names: delete `backtest_cache/archive_days/` and `backtest_cache/live_days/`, then re-run with `--no-cache` — `--no-cache` alone does not refresh the day slices. Fractional-contract sizing remains deferred. Note that since the cumulative-deadline wording rule shipped, the **cross-event** time-series leg is dormant (same-event deadline ladders, the one population that is not, are on by the 2026-09-26 decision — see above): it forms 0 cross-event pairs live, and a backtest's surviving cross-event candidates are mostly recurring windows the entry rule can never enter, so a pooled `k̂` measured on a pre-rule cache should not be trusted (see `CLAUDE.md`'s cumulative-deadline gotcha).

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

    run_lock.py — imports config.py only; main.py takes its machine-wide
    lock for a production run that sends orders, scheduler.py reads its
    record of the run holding it, and defaults_server.py checks it before
    it offers a real-money run


    historical.py ──→ backtester.py ──→ dashboard.py
                            |
                    backtest.py (CLI entry) ──→ auth.py (read_account_balance:
                                                 the starting balance when
                                                 --balance is not given)

    historical.py ──→ treasury.py  (downloads the 8-week Treasury bill's auction
                                     yields for the backtest dashboard's Sharpe/
                                     Sortino risk-free rate; reuses historical.py's
                                     CACHE_DIR/_load_json_cache/_save_json_cache/
                                     _exception_summary idioms. Reporting only —
                                     no order-path module imports it;
                                     live_portfolio.py and live_dashboard.py,
                                     which only read, do: the Live trading
                                     tab's Sharpe/Sortino subtract it too.
                                     treasury.py → dashboard.py: every
                                     Sharpe/Sortino subtracts its per-day
                                     yield — a strategy curve's on its capital
                                     in open trades only; treasury.py →
                                     backtest.py: load_risk_free_rates() hands
                                     the yields to generate_dashboard)

    historical.py ──→ depth_model.py  (the backtest's order-book depth model: it
    scanner.py ──→ depth_model.py      saves snapshots of live books, fits a table
                                       of how many contracts rest near the best
                                       bid and builds a synthetic book from it;
                                       it also imports config.py and _http.py,
                                       reads with GETs only and never places an
                                       order. Backtest only: no live module
                                       imports it; backtester.py and
                                       backtest.py do)

    (historical.py also imports auth.py's build_client for its own client
     builders, _http.py directly for its raw signed GETs, and scanner.py's
     event_series so the event-title lookup budget's combo test agrees with
     the one-series rule; main.py ──→ historical.py for load_series_categories,
     series_labels and infer_category, and dashboard.py ──→ historical.py for
     series_labels — the one rule that files a pair under a Kalshi category
     and tag, so the live category/tag filter, and the run result's category
     and tag, file a pair where the dashboard's Category and Tag menus file
     a trade of the same event;
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
     bought; and, to size each trade, scanner.py's _enrich_pair and
     strategy.py's compute_trade — the live sizing code — over a synthetic
     book that depth_model.py's book builds, and a sale walks the bid
     ladder its bid_ladder builds; backtest.py ──→ depth_model.py
     for load_depth_model)

    seller.py — imports config.py, scanner.py and historical.py only (never
    backtester.py, depth_model.py, trader.py, strategy.py, reporter.py or
    main.py); main.py alone imports it (plan_sales: which held positions the
    take-profit rule sells, before any purchase), and trader.py and
    reporter.py read its plans by their fields without importing it

    v2_probe.py — standalone, human-run verification CLI; imports
    auth.py/config.py/_http.py/scanner.py/trader.py, imported by NOTHING
    in the pipeline

    defaults_server.py — standalone, human-run (started by
    start_dashboard.sh at the repo root, beside live_dashboard.py); imports
    config.py and run_lock.py only, imported by NOTHING (the local pages that
    save live_defaults.json and start live trading runs — main.py, each as
    its own process)

    live_portfolio.py — reporting only, for the Live trading tab: reads the
    account with read-only GETs (through historical.py's _historical_get) and
    the bot's trade log (reporter.py's PROD_LOG_PATH), and works out values
    and statistics (dashboard.py's _sharpe/_sortino, treasury.py's yields);
    also imports config.py, auth.py and _http.py; never the order path
    (trader, main, scheduler, run_lock, defaults_server, v2_probe)

    live_dashboard.py — standalone, human-run, read-only (started by
    start_dashboard.sh): serves the dashboard's two tabs, the Live trading
    tab on 127.0.0.1:8766 and the backtest page on 127.0.0.1:8767; imports
    config.py, live_portfolio.py, historical.py, treasury.py and _http.py,
    imported by NOTHING
```

### Live Trading Data Flow

Step order below is `_run_prod`'s; `_run_dev` skips `get_held_positions`,
`resolve_held_ladders`, `plan_sales`, `sell_positions` (and the sale log and read-back) and `held_pairs`
entirely (no sandbox positions to sell or add to) and so calls `fetch_shard_statuses` first instead.

```
main.py
  ├─ config.order_api_version_error() — refuse any ORDER_API_VERSION but "v2" (exit 2, right after parsing its arguments)
  ├─ main._resolve_live_settings() — the run's one config.LiveSettings: live_defaults() reads
  │                                   live_defaults.json (the saved live defaults, the run's one read,
  │                                   no config.py fallback), each toggle overridden by its flag when
  │                                   given; the saved defaults are the reference. No file, a refused
  │                                   file or a bad flag exits 2 before logging, the client or any request
  ├─ run_lock.acquire()            — production without --dry-run only: take the machine-wide live-run lock
  │                                   (~/.kalshi_betting/live_run.lock), held until main() ends; still held by
  │                                   another run after 2 s -> exit 50, before the client or any request
  ├─ auth.build_client()           — authenticate with Kalshi API
  ├─ main._log_live_settings()     — log "Live defaults: <origin>" (the saved file, when and from what),
  │                                   the run's toggles with each departure marked "(default: X)" (a
  │                                   WARNING when a production run submits orders under one), and every
  │                                   live_rule_warnings sentence
  ├─ auth.read_account_balance()   — read each shard's cash and Kalshi's value of the open positions (prod only);
  │                                   size on the portfolio value (cash + positions; cash alone, with a WARNING,
  │                                   when that value cannot be read), spend only the cash, and stop below $50
  │                                   of portfolio value (exit 10); logs "Sizing on portfolio value $X = cash $Y
  │                                   + open positions $Z"
  ├─ scanner.get_held_positions()  — fetch currently-held positions with their sides and costs (prod only);
  │                                   held markets are skipped, bar an exact held pair the run adds to
  ├─ main._checked_positions_value() — keep Kalshi's value of the open positions only if the contracts held
  │                                   can back it (prod only; $1 a contract, over a positions listing read to
  │                                   its end with every count readable; a value of 0 is always kept); else
  │                                   one WARNING: size on the cash alone, and stop if that is below $50
  │                                   (exit 10)
  ├─ scanner.fetch_shard_statuses() — read GET /exchange/status per-shard trading/transfer flags (fail-soft)
  ├─ scanner.inactive_shard_indexes() — derive the trading-inactive shard set from the statuses above
  ├─ scanner.fetch_open_events_with_markets() — fetch open events + their markets from EVERY exchange shard, tagged (attaches event titles for MVE grouping; drops only markets on trading-inactive shards)
  ├─ main._log_shard_coverage()    — audit advertised shards vs ingested markets/funds (reports, never aborts)
  ├─ scanner.resolve_held_ladders() — the ladders (events, and questions at any deadline) the held positions
  │                                   are on (prod only): read from the full market list before held markets
  │                                   are dropped, a held market missing from it looked up; one that cannot
  │                                   be identified means no time-series pair this run (same-title still runs,
  │                                   and the run exits 40)
  ├─ seller.plan_sales()           — with sell_at set (and every held market identified and the positions listing
  │                                   complete; else one WARNING and no sale): judge each exact held pair, and each lone held market whose
  │                                   partner has paid out, by the take-profit rule. The share of potential profit selling it would realize
  │                                   must reach the level at each of three daily checks 24 hours apart: now on the real order book walked
  │                                   for its count, and on each earlier day on the market's last hourly candle bid (any size); with
  │                                   sell_min_days, enough days must also be left before its last market stops trading. One log line per
  │                                   position, and a summary
  ├─ trader.sell_positions()       — sell what was planned before anything is bought, one position at a time: one
  │                                   reduce-only, immediate-or-cancel order per held market (a pair's second order sized to what the first
  │                                   sold), each sent once and never retried; a dry run sends nothing
  ├─ main._positions_after_sales() / auth.read_account_balance() / main._value_after_sales() — a live run that
  │                                   sold reads its positions, its cash and Kalshi's value of what is left again, so its buys are sized on
  │                                   what the sales left (a dry run adds each sale's estimated proceeds to the cash instead). The markets
  │                                   sold stay out of every purchase, and a sold position's ladders are free again
  ├─ reporter.append_to_prod_log() — write the sales to trade_log.xlsx behind a banner of their own, before anything
  │                                   is bought (a dry run: right after the sales; a live run: once the read-back above
  │                                   is done, or at once if it raises)
  ├─ scanner.held_pairs()          — with add_to_held_pairs on, a complete positions listing, every held
  │                                   market identified and Kalshi's value of the open positions kept (else
  │                                   one WARNING, no add-on): the exact held pairs this run may add to (two
  │                                   held markets alone on one ladder, one YES and one NO of equal size, costs
  │                                   reported) and lone legs (one held market alone on its ladder, its
  │                                   partner paid out), each valued at today's prices and staked at that worth plus
  │                                   the fees paid for it, less any with no room left even at the per-trade
  │                                   cap; their markets stay in, every other held one is dropped
  ├─ scanner.filter_markets_within_horizon() — optional --max-horizon-days cap (no-op if unset)
  ├─ scanner.find_time_series_pairs()   — time-series pair detection (first refusing any
  │                                        candidate with a market on a held ladder, but
  │                                        the exact held pair it adds to, before
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
  ├─ scanner.find_same_title_pairs()    — same-title pair detection (a held
  │                                        pair's markets pair only with each other)
  │                                        (two different event series, and both
  │                                         markets closing within one hour of each
  │                                         other — scanner.closes_apart, DR-74)
  ├─ main._dedup_pairs()            — merge both lists, preferring same-title on overlap
  ├─ historical.load_series_categories() — with --result-file and candidate pairs only: read Kalshi's
  │                                   /series listing once, to file each trade in the run result
  │                                   (main._pair_labels) and hand the filter below
  ├─ main._filter_by_category()     — keep only pairs filed under the run's Kalshi categories and tags
  │                                   (the dashboard's rule: market A's literal series, its category and
  │                                    first tag, via historical.series_labels); a no-op, with no request,
  │                                    when neither is set; prod refreshes a stale /series listing, dev
  │                                    reads the cached copy only; no listing -> no pair (fail closed)
  ├─ scanner.enrich_with_orderbook_prices() — validate depth; price each pair over the depth one trade's budget (its Kelly share of the portfolio value, never more than the cash) could actually buy; drop a time-series pair with no later YES ask, a crossed book on either market, or a fresh spread the run's rule refuses; record a time-series pair's mid spread from the tops of both books
  ├─ strategy.compute_trade()      — Kelly sizing per pair, under the run's config.LiveSettings (k and the per-trade caps, the same object enrichment's depth bound read), a time-series pair's chance of profit read from its mid spread (the same at every size; no mid spread, no trade), a fraction of the portfolio value and never more than the cash, at the marginal fill price of the size it settles on, and only over depth the fill-or-kill limit can actually reach; an add-on to a held pair on its whole position (config.held_pair_fraction, the held pair at its worth today plus the fees paid for it)
  ├─ strategy.select_portfolio()   — greedy portfolio selection spending the cash in whole cents at what each trade's
  │                                   orders can draw at their limit prices; a trade that no longer fits the cash left
  │                                   is shrunk to the largest size that does (skipped only when not one contract pair
  │                                   fits); at most one time-series trade per ladder (none on
  │                                   a held ladder, and none on the ladder of a trade picked earlier in the run;
  │                                   an add-on to a held pair is blocked only by the run's earlier picks)
  ├─ trader.pre_execution_check()  — re-fetch order books, drop pairs whose prices moved or whose depth is no longer reachable at the limit about to be submitted, counting only the fresh levels that still keep an edge after the fee (a time-series pair also when its later book has no YES ask now, or its fresh spread exceeds the band's ceiling)
  ├─ trader.ensure_shard_collateral() — move funds onto the shards the selected legs settle against (prod; dry-run only plans)
  ├─ trader.execute_trades()       — submit fill-or-kill orders leg-by-leg to the V2 order endpoint, each leg routed to its own market's shard (parallel across pairs once the first NO fill confirms the V2 order mapping, one at a time before that; rollback on partial fill; a disproven mapping stops the rest of the run)
  ├─ auth.read_account_balance()   — re-read the post-trade cash for the Excel log (falls back to the pre-trade cash if this read fails)
  ├─ reporter.append_to_prod_log() — write results to trade_log.xlsx, the run's live toggles on its separator row
  └─ reporter.write_run_report()   — with --result-file (production only): write what the run did as JSON, in
                                      main()'s finally, however the run ended once logging was set up (exit code,
                                      the line it stopped or finished on, its cash before and after and the
                                      portfolio value it sized on, whether it began sending orders,
                                      each sale's outcome, each pair's outcome and Kalshi category and
                                      tag, warnings, any error), before the live-run lock is released
```

### Backtest Data Flow

```
backtest.py (CLI)
  ├─ historical.build_historical_client()    — prod API client for archives
  ├─ historical.build_prod_live_client()     — prod API client for recent data
  ├─ auth.read_account_balance()             — without --balance: the starting balance is the
  │                                             account's value now (cash on every shard + Kalshi's
  │                                             value of the open positions; the cash alone, with a
  │                                             WARNING, when that value is unreadable). One
  │                                             read-only GET; a read that fails or comes to
  │                                             nothing stops the run (exit 1) before the fetch
  ├─ depth_model.load_depth_model()          — fits the depth table from every saved order-book snapshot
  │                                             (python3 -m kalshi_betting.depth_model snapshot saves
  │                                             one: read-only GETs); never raises; with none usable it
  │                                             returns None and every trade fills at the top of the
  │                                             book. Handed to run_backtest_sweep(depth_model=)
  ├─ backtester.run_backtest_sweep()         — run_backtest() is the plain two-tuple wrapper other callers use
  │    ├─ _prepare_candidates()                   — band- AND k-independent; runs once no matter how many
  │    │    │                                        bands or k's are simulated
  │    │    ├─ historical.fetch_all_settled_markets() — market metadata, returned as a SettledCorpus
  │    │    │     │                                    (streams settled_markets_*.jsonl.gz on every walk;
  │    │    │     │                                    a cache from an earlier UTC day is extended
  │    │    │     │                                    through today; a legacy .json is re-assembled)
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
  │    │                                               open; a window over the 5,000-candle cap is paged; a
  │    │                                               market settled after the archive cutoff from Kalshi's
  │    │                                               live endpoint, any 404 retried on the other endpoint)
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
  │         │                                    (in-between rate / mean mid spread at each pair's first
  │         │                                    qualifying Monday; and per binding band with the tier
  │         │                                    floors off)
  │         ├─ _simulate_at_discount()  — once per (band, k[, population]): Kelly gate (a time-
  │         │                              series Monday forecast at its mid spread, and skipped
  │         │                              when either market's quote is crossed; a time-series
  │         │                              pair keeps every Monday it passes on, a
  │         │                              same-title pair its first), one same-title pair per
  │         │                              group, dedup, a Monday-by-Monday cash walk that sizes
  │         │                              each Monday on its portfolio value (cash + open trades
  │         │                              at market) and the cash left, through _size_trade: the
  │         │                              live scanner._enrich_pair and strategy.compute_trade
  │         │                              over a modeled order book (a market with no volume data
  │         │                              that Monday, or no depth model, fills at the top of the
  │         │                              book), so a trade pays the average price over the levels
  │         │                              it walks and is shrunk to the cash left, trades each
  │         │                              pair at most once
  │         │                              (only the add-on family below adds to a pair it still
  │         │                              holds) and holds at most one open time-series trade per
  │         │                              ladder (a ladder frees up the day its market pays out; a
  │         │                              time-series pair skipped one Monday is tried again on
  │         │                              its next passing one), P&L from outcomes,
  │         │                              _build_equity_curve() — the scenario explorer's
  │         │                              band x k x {all, time_series, ladder, cross} grid, its
  │         │                              tier-floors-off twin at the binding bands, plus
  │         │                              one same-title simulation, come from repeating this call,
  │         │                              never from slicing a joint run; every one of them at the
  │         │                              run's own per-trade size cap (config.BUDGET_FRACTION), a
  │         │                              same-title candidate also under config.SAME_TITLE_SIZE_CAP
  │         │                              (config.pair_size_cap, as live sizing caps it)
  │         ├─ CapSweep (default; not with --no-cap-sweep) — every other per-trade size cap of
  │         │                              SIZE_CAP_SWEEP over the tier-on grid, and a second
  │         │                              one (tier_floors=False) over the tier-floors-off
  │         │                              twin, each seeded from its own grid's points,
  │         │                              none of them simulated here: each (band, k) cell
  │         │                              is simulated through
  │         │                              _simulate_at_discount(size_cap=) only when a report
  │         │                              reads it, and caps at or above a point's peak Kelly
  │         │                              fraction (over a walked book, and at or above its
  │         │                              cap_free_from too) share one simulation
  │         ├─ CapSweep (add-on family; default, not with --no-add-on-sweep) — the same grid
  │         │                              again, and over the tier-floors-off twin, with every
  │         │                              simulation adding to held pairs
  │         │                              (_simulate_at_discount(add_to_held=True)), the "all"
  │         │                              population only; no eager point exists, so each cell
  │         │                              is simulated only when a report reads it, ending its
  │         │                              curves on the day its eager twin's ended
  │         └─ SellSweep (sell family; default, not with --no-sell-sweep) — every level of
  │                                        config.TAKE_PROFIT_LEVELS (80% to 100% in 1% steps),
  │                                        each with every minimum of TAKE_PROFIT_MIN_DAYS
  │                                        (1-7, 14 and 21 days), over the same grid, tier
  │                                        floors on and off, adding to held pairs or not
  │                                        (_simulate_at_discount(sell_at=, sell_min_days=)):
  │                                        a position is sold whole at the first weekly
  │                                        checkpoint where its realized profit has stayed at
  │                                        or above that share of its potential profit for
  │                                        TAKE_PROFIT_HOLD_DAYS (3) days in a row, and only
  │                                        while at least that minimum of days remains before
  │                                        its last market stops trading; nothing simulated here
  │                                        (sold_grid later simulates only the settings
  │                                        that differ)
  ├─ treasury.load_risk_free_rates()         — 8-week T-bill auction yields from Treasury Fiscal
  │                                             Data (one open, no-key GET that never raises —
  │                                             falls back to the last saved download, then to
  │                                             "unavailable"; a host that swallows packets
  │                                             costs each of 6 attempts at least the 30 s
  │                                             timeout — per resolved address, since it bounds
  │                                             each socket operation, not the request — plus
  │                                             62 s of backoff: ~4 min for a single-address
  │                                             host); passed to generate_dashboard(risk_free=)
  └─ dashboard.generate_dashboard()          — write HTML report: a header line saying how many
                                                trades walked a modeled order book, a k section and two k̂ cards
                                                that follow the filter bar, a k̂ breakdown by
                                                category / tag / band (regrouped from each band's
                                                calibration observations), the scenario-explorer section
                                                (band x k heatmap per size cap, nine metrics,
                                                per-population KPIs, and a Tier floors select over its
                                                tier-floors-off grid at every cap), and a sticky
                                                filter bar whose every band x tier floors x k x size cap
                                                x add to held pairs x Kalshi category x tag view is
                                                computed here: one walk
                                                over the scenario grid reads each CapSweep cell
                                                (simulating it) and then each binding band's
                                                tier-floors-off CapSweep cell, then the add-on
                                                family's cells, packs one chunk per distinct
                                                trade list, keeps the k section's rows and curves at
                                                every k and cap and the explorer's figures for every
                                                band x k x cap (a block per cap, and a tier-off block
                                                per cap), streamed into the page beside
                                                base blocks and unpacked by small scripts only when
                                                chosen; then the Sell and Min. days to
                                                maturity selects: every level and
                                                minimum of days of every scenario the
                                                bar shows, read through sold_grid in
                                                --sell-workers worker processes (a
                                                setting no position of a scenario
                                                reaches is that scenario's own run;
                                                one that makes an earlier run's sales
                                                is that run), each new trade list
                                                written as a file in
                                                backtest_dashboard_files/<build>/
                                                beside the page, and each band's chunk
                                                ids under each Tier floors setting as
                                                one block in the page, loaded by the
                                                page's script when chosen; the bar's "Save as live
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
| `config.py` | All tunable constants (price thresholds, Kelly cap, fee rates, API URLs, file paths — bar one backtest-only exception, the size-cap grid `backtester.SIZE_CAP_SWEEP`, kept in `backtester.py` by the operator's instruction for that change), the two fee helper functions, `max_affordable_pairs()` — the single budget-to-contracts definition shared by the scanner's depth bound and the sizer — with `kelly_budget()`, the one rule for what a trade may spend (its Kelly fraction of the portfolio value, never more than the cash on hand), and `leg_cash_cents()`, the one rounding of an order leg's cost up to the whole cent, shared by the portfolio step and the shard funder — and the time-series spread-band helpers `time_series_spread_band()` / `time_series_spread_too_wide()` plus the `SPREAD_BAND_SWEEP_FLOORS` / `SPREAD_BAND_SWEEP_CEILINGS` grid the scenario explorer sweeps, and the live trading toggles' constants — `TIME_SERIES_TIER_FLOORS`, `TIME_SERIES_SPREAD_BAND`, `SAME_TITLE_SIZE_CAP` (an extra per-trade cap on same-title pairs; 0.20 since the operator decision of 2026-09-27, 1.0 — no extra cap — before it) `TRADE_CATEGORIES` / `TRADE_TAGS` (the category/tag filter; `None`, any, today), `ADD_TO_HELD_PAIRS` (whether a production run may add to an exact pair the account already holds; `True`) and `SELL_AT` / `SELL_MIN_DAYS` (the share of potential profit a held position is sold at, and the fewest days before its last market stops trading it may be sold at; both `None`, which never sells), which with `k` and the per-trade cap are the backtest's values and, through `live_settings()`, the fallback for a caller that hands no settings — never a live run's defaults. A live run starts only from the saved live defaults, `live_defaults.json` (`LIVE_DEFAULTS_FILE`), read into one frozen `LiveSettings` per run by `live_defaults()` (over which `main.py`'s toggle flags may lay a value for one run), written only by `save_live_defaults()` (on `defaults_server.py`'s Confirm and save, or its Confirm and trade, which saves before it runs; the server's address and limits are the `DEFAULTS_SERVER_*` constants), with `LIVE_DEFAULTS_SEED` the starting values `defaults_server --seed` offers for a first save; the helpers that name them (`describe_live_settings()`, the run's `Live settings:` line, `describe_trade_filter()`, the category/tag filter in that line's words, `trade_filter()` and its two-field twin `trade_filter_for()`, that filter's one rule (see [Live trading toggles](#live-trading-toggles); the dashboard asks the twin to word what to tick in its menus), and `live_rule_warnings()`, every setting that empties part of the strategy or lets one pair stake more than `LIVE_EXPOSURE_WARN_FRACTION`), and the helpers that apply them (`live_time_series_floor()`, `time_series_spread_refusal()`, the one definition of the live time-series spread rule, `pair_size_cap()`, the one definition of a pair's per-trade cap, shared by the live sizer and the backtester, `held_pair_fraction()`, the one definition of an add-on's size — what a held pair, at its current worth plus the fees paid for it, is missing of its Kelly share of the portfolio value, never more than a new pair would get — and `max_kelly_fraction()`, the scanner's affordability bound), and `order_api_version_error()`, the startup check `main.py` and `v2_probe.py` run to refuse any order path but `"v2"`. The sell rule's tests live here too, where live code may import them: `take_profit_reached()`, `reached_every_check()` and `days_to_maturity()` (the backtest and `seller.py` decide with them), beside the live sale's own settings (`SELL_AT_STEP`, `SALE_PARTNER_MAX_LOOKUPS`, `SALE_PARTNER_MAX_AGE_DAYS`, `SALE_HEDGE_SLIPPAGE_TICKS`, `SALE_READ_BACK_RECHECK_SECONDS`) and `TAKE_PROFIT_HOLD_DAYS`, the days in a row a position must hold its level, which the backtest and live selling share. No live module reads a toggle constant or a backtest band helper directly: the finder, the category/tag filter (`main._filter_by_category`), the scanner's enrichment and pre-execution check, and live sizing (`strategy.compute_trade`) all read the run's one `LiveSettings`. Also `CONTRACT_PAYOUT_DOLLARS`, what one contract pays if it wins ($1): `main.py` refuses Kalshi's value of the open positions when it is above that times the contracts held. Also the weekly live run's schedule, `SCHEDULED_RUN` — a frozen `ScheduledRun` holding its weekday, time and IANA zone (Monday 09:00 America/Los_Angeles), which the scheduler reads and the backtest enters every trade at. |
| `auth.py` | Reads RSA credentials from `secrets.json` and the PEM key file, constructs an authenticated `KalshiClient`, and reads the live account balance: `read_account_balance()` returns each exchange shard's cash (`{exchange_index: cents}`) and Kalshi's value of the open positions (the reply's `portfolio_value`, integer cents, which does not include cash; `None` when unreadable) as an `AccountBalance`, and `verify_auth()` makes the same read and returns only the cash per shard (the `v2_probe` verification tool uses it). `main.py` reads `read_account_balance()`: it spends the cash summed over the shards and sizes on that sum plus the positions' value. |
| `_http.py` | Shared HTTP helpers used across the package (auth, scanner, historical, trader, and v2_probe): `api_call_with_retry()` (exponential backoff on 429/5xx for market-data calls) and `fetch_json_page()` (parses the SDK's raw `*_without_preload_content` responses, re-raising non-2xx as `ApiException`), and `signed_request_json()` (signed GET/POST against an arbitrary API path for routes the pinned SDK has no method for — retry-free, since order submission and the collateral transfer call it directly); plus `api_error_payload()` (reads the exchange's JSON error object — `code`, `message`, `details` — out of a failed request) and `api_error_summary()` (one line per failed request: `HTTP 400 Bad Request — missing_parameters: missing parameters (…)`, or the body itself on one line when it is not that object, never the SDK's multi-line exception text with every response header; `trader.py` logs and records every failed order, unwind, position read and transfer this way — `TradeResult.error`, and so the trade log's Notes cell, carries the same line — and `scanner.py` logs a failed order-book read this way; `backtest.py` stops with this line when the account balance it starts from cannot be read). |
| `scanner.py` | Fetches all open Kalshi markets, strips date tokens from titles and appends each market's outcome label (subtitle) to group time-series pairs, detects same-title pairs via exact match, refuses a same-title pair between two events of one series or whose two markets close more than an hour apart (`closes_apart()`, the one definition of that close gate, failing closed on a close time it cannot compare, with its own silent-at-zero refusal count whose printed bound comes from `close_gap_bound_text()` on both paths — DR-74), a time-series pair whose wording is identical across one series (two instances of one recurring fixture — a genuine two-deadline family of one series spells its deadline in the wording and can still pair, if both legs are worded as cumulative deadlines; `event_series()` reads the prefix before the first hyphen, except that every `KXMVE*` combo prefix collapses onto one family so two combos listed under two different combo series are still refused), a time-series pair between two markets of ONE event unless `TIME_SERIES_SAME_EVENT_LADDERS` is on and they are two dated rungs of that event's deadline ladder (`stated_deadline()` / `same_event_ladder()`, which order the legs and measure the gap on the STATED deadlines; `pair_gap_days()` is the one place anything downstream reads that gap), and a time-series pair whose legs are not both worded as cumulative "by \<date\>" deadlines at two different dates (compared as normalized text, not parsed calendar dates — see the note above; `deadline_phrasing()` — a snapshot family such as "Solana price on Sep 14/18, 2026?" still groups but no longer pairs; `deadline_pair_refusal()` names WHY a refused pair was refused — snapshot wording, no stated deadline, or the same deadline stated twice — feeding three separate, honestly-labelled skip counts instead of one folded one, DR-72; every one-series refusal, in both finders, and every same-title same-event skip is counted on its own silent-at-zero line too, M10), refuses — first, before every other check — a time-series candidate with a market on a ladder the account holds (`ladder_keys()` / `market_ladder_keys()` / `pair_ladder_keys()` name the ladders a market is on — its event, and its question with the dates removed — and `resolve_held_ladders()` finds the held positions' ladders, looking up a held market the run's list lacks, and failing closed when one cannot be identified) — except, on a run that adds to held pairs, the exact held pair itself, which pairs only with its own partner and only buying the sides already held (`get_held_positions()` reads each held position's side and cost, `get_held_tickers()` is its tickers, and `held_pairs()` finds what a run may add to: exact held pairs — two held markets alone on one ladder, one YES and one NO of equal size, both costs reported — and lone legs, one held market alone on its ladder whose partner has paid out, each valued at today's prices from the run's market list (its stake, which sizing reads, adds the fees paid for it), and none when any held market's ladder is unknown), and enriches tradeable pairs with live order book depth to compute real fill prices — averaged over the contracts one trade's budget (its Kelly share of the portfolio value, never more than the cash) could actually buy, not the whole book, with the qualifying levels kept on the pair (`depth_levels`) for the sizer to re-price against via `prefix_fill_prices()`. Also home to `leg_sides()` / `leg_prices()`, the single mapping from a pair's type to the side and price each leg actually trades, and to the V2 order-price grid arithmetic (`tick_size_for_price()`, `ceil_to_tick()`, `v2_limit_price()`, `v2_effective_cap()`) — it lives here, not in `trader.py`, so the sizer can test a candidate size against the very limit the trader will submit without importing it. It also holds what live selling reads: `walk_bids()` (sell n contracts down a bid ladder: their average price and the lowest price reached) and `bid_ladder()` (one side's usable resting bids of an order book, best first), `floor_to_tick()` and `v2_bottom_of_grid_price()` (the price math a sale order needs: a price rounded down onto a grid, and a grid's lowest level), `get_settlements()` (the markets the account held when they paid out, with their counts, cost, fees and revenue), `market_for_labels()` (a market's ladder labels, looked up as `resolve_held_ladders()` looks one up) and `held_pairs(..., log=False)`, which finds the positions it may sell without logging what a run may add to. |
| `strategy.py` | Solves size and price together — binary-searching the book for the largest contract count whose own marginal fill price still justifies it AND that the resulting fill-or-kill limit can actually buy — then applies the Kelly criterion to size each trade — a time-series pair's chance of profit read from the mid spread enrichment wrote on it (`config.time_series_profit_prob`, the same at every size; a time-series pair with none is not traded), a fraction of the portfolio value, never more than the cash (`config.kelly_budget`), at the run's `k` and under its per-pair cap (`config.pair_size_cap`: the per-trade cap, and for a same-title pair `SAME_TITLE_SIZE_CAP` too), both read from the run's one `config.LiveSettings` — computes the profit floor for same-title pairs / the win-scenario profit for time-series pairs and the monthly-normalized return, and greedily selects a portfolio the cash buys — spending it in whole cents at what each trade's fill-or-kill orders can draw at their limit prices (`TradeSpec.cash_need_cents`), never less than each exchange shard is later funded with, and shrinking a trade that no longer fits the cash left to the largest size whose own fill, reachability, win payoff and expected value after the exact fees check out, rather than dropping it — with at most one time-series trade per ladder (none on a ladder the account holds, and none on the ladder of a trade picked earlier in the run). An add-on to a held pair is sized on its whole position (`config.held_pair_fraction`, the held pair at its worth today plus the fees paid for it) and blocked only by the run's earlier picks; one shrunk to the cash left is still an add-on. |
| `trader.py` | Converts `TradeSpec` objects into orders and submits each pair's two legs sequentially (fill-or-kill, NO leg then YES leg — the NO leg is `market_a` for a same-title pair and `market_b`, the later contract, for a time-series pair) via the Kalshi API, with automatic rollback of the filled NO leg if the YES leg doesn't fill (the rollback sells back only this pair's NO contracts, so a position the account already held on that market is left as it was; a pair that adds to a held pair is sent only while both markets still hold exactly that pair). Multiple pairs execute concurrently — except until the process's first NO-leg fill has confirmed or disproven the order-side mapping, when they run one at a time (for at most `config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS`, 300 s); a disproof stops every pair that starts after it before anything is sent (`failed`) — and every order and collateral-transfer POST from every pair takes a place on one shared pacer (`config.ORDER_WRITES_PER_SECOND`, in bursts of at most `config.ORDER_WRITE_BURST`) so they stay under the account's write limit; a pair's NO leg holds a place for its YES leg, and unwinds go ahead of waiting NO legs — see "Order API version" below. Submission goes to the V2 order endpoint, the only order path the bot has — see "Order API version" below. `sell_positions()` also sells the held positions `seller.py` picked, before anything is bought: one reduce-only, immediate-or-cancel order per held market for the contracts held there (a held YES by an ask, a held NO by a YES bid), priced from the lowest bid the plan's walk reached and rounded toward the safe side, a pair's second order sized to what the first sold; each order is sent once, after one place on the same write pacer, and never retried, and an unclear reply is settled by how the account's position moved. |
| `reporter.py` | Writes trade results to Excel. In production, appends to a persistent `trade_log.xlsx`, each run's separator row naming the live toggles the run traded under. In dev mode, writes a fresh timestamped simulation file with two sheets (trades + all candidates). Market cells are rendered by `scanner.display_title`, so they carry the event title and the outcome label alongside the market title. For a production run started with `--result-file`, it also holds the run result (`RunReport`: the cash before and after trading, the portfolio value the run sized on, one `TradeRecord` per pair with the Kalshi category and tag it is filed under, the run's warnings) and writes it as JSON when the run ends (`write_run_report`, which never raises). A production run's sales are written to the same log, behind a banner of their own and before any trade (`append_to_prod_log(..., sales=)`, one row per sale), and the run result gains one `SaleRecord` per position and `cash_after_sales`. |
| `seller.py` | Decides which held positions the take-profit rule sells this run, and plans each sale; it sends no order. `plan_sales()` judges every held position (an exact held pair, or a lone held market with its one paid-out partner among the account's settlements) by the share of its potential profit that selling it would realize, at each of `config.TAKE_PROFIT_HOLD_DAYS` (3) checks 24 hours apart: now on the held markets' real order books (walked for the position's count, fees included), and on each earlier day on their last hourly candle bids (in any size), with `sell_min_days` as a days-to-maturity rule checked first. The tests are `config.take_profit_reached()` and `config.reached_every_check()`, the ones the backtest decides with. It logs a `Take-profit check` line per position and a summary, a `Not selling` line with the reason for each held market or position it will not sell, and returns the plans (`SalePlan`: the legs, count, cost, the bid ladders and walks the orders are priced from, the profit at each check) that `trader.sell_positions()` sends. It imports `config.py`, `scanner.py` and `historical.py` only and is imported by `main.py` alone; a test builds one position as a backtest trade and as live positions and compares the two at every check. |
| `main.py` | Top-level CLI orchestrator for the live trading pipeline. Refuses to start (exit 2) on any `config.ORDER_API_VERSION` but `"v2"` (see [Order API version](#order-api-version)), then resolves the run's one `config.LiveSettings` — the saved live defaults (`live_defaults.json`; with none saved, or a refused file, the run exits 2), each overridden for this run only by its flag when given (see [Live trading toggles](#live-trading-toggles)) — before logging is configured, then dispatches to `_run_dev()` (sandbox simulation) or `_run_prod()` (real-money trading) based on `--mode`, handing it that object and the saved defaults it was built from; the run logs where the defaults came from and its settings (marking every departure from the saved defaults `(default: X)`, and warning when a production run submits orders under one) and every live site reads that one object. Beyond orchestration it holds two pair-list filters: `_dedup_pairs()` (a same-title pair wins a ticker-pair collision) and `_filter_by_category()` (keep only the pairs filed under the run's Kalshi categories and tags — filed by the backtest dashboard's own rule, kept by `config.trade_filter` — a no-op when neither is set). A production run also reads its open positions with their sides and costs (`scanner.get_held_positions()`), finds the ladders they are on (`scanner.resolve_held_ladders()`) and hands them to the time-series finder and the portfolio step; a held market it cannot identify means no time-series pair that run. With the run's `add_to_held_pairs` on, a complete positions listing, every held market identified and, while anything is held, Kalshi's value of the open positions read and kept, it also finds the exact held pairs it may add to (`scanner.held_pairs()`, each valued at today's prices), leaves out any whose worth plus the fees paid for it already fill its per-trade cap of the portfolio value, and hands the rest to both finders; otherwise it logs one WARNING saying why and adds to nothing. A production run sizes every Kelly fraction on the account's portfolio value — the cash on every shard plus Kalshi's value of the open positions (the cash alone, with a WARNING, when that value cannot be read) — and spends only the cash; Kalshi's value of the open positions counts only when the contracts held can back it (at most $1 a contract, `config.CONTRACT_PAYOUT_DOLLARS`, over a positions listing read to its end whose every count can be read; a value of 0 is always kept, since it adds nothing), and a larger value, or one above 0 it cannot check, is refused with a WARNING, the run then sizing on the cash alone; it stops below $50 of portfolio value (exit `10`), checked again on the cash after a refusal, and cash alone below $50 only draws a WARNING. Its pairs table shows each selected trade at the size that trades, one shrunk to fit the cash included. A dev run spends its virtual `--sandbox-balance` as both. With the run's `sell_at` set, and every held market identified and the positions listing complete, a production run sells before it buys: `seller.plan_sales` picks the held positions that have reached that share of their potential profit, `trader.sell_positions` sells them (a dry run sends nothing) and the sales go into the trade log before anything is bought; a live run reads its positions and cash again first, so its buys are sized on what the sales left, while a dry run adds each sale's estimated proceeds to the cash its buys are sized on. The markets sold are not bought or added to in the same run, and a sale left uneven, or whose outcome is unknown, makes the run exit `20`. The flags `--sell-at PCT` / `--no-sell` and `--sell-min-days N` / `--no-sell-min-days` override the saved sell level and minimum of days for one run. A production run that sends orders (not `--dry-run`) first takes the machine-wide live-run lock (`run_lock.py`) and holds it until the run ends; when another run holds it, this one exits `50` before building a client. With `--result-file PATH` (production only) it writes what the run did to `PATH` as JSON when the run ends, however it ends once logging is set up, each trade filed under its Kalshi category and tag (`_pair_labels()`, the filter's own rule, over Kalshi's /series listing, read once right after dedup when the run has candidate pairs); a usage error that exits 2 before that, or a kill signal, leaves no file (see [Run result file](#run-result-file)). |
| `scheduler.py` | Long-running daemon that fires the production bot once a week at `config.SCHEDULED_RUN`'s weekday and time (Monday 09:00) on the host's clock, using the `schedule` library, with no live toggle flag, so a scheduled run trades exactly the saved live defaults (with none saved, or a refused file, every run exits 2, and the daemon logs an ERROR saying so when it starts), and checks at startup that the host's clock keeps the schedule's zone (America/Los_Angeles), logging CRITICAL if it does not. A run stopped because another live trading run held the run lock (exit 50) counts as the week's slot done and is not retried; the daemon logs a WARNING naming that run, or an ERROR when that run has held the lock for over an hour or its start is not recorded. Also prints the equivalent cron job command. |
| `run_lock.py` | The machine-wide lock that lets one live trading run place orders at a time: an flock on `~/.kalshi_betting/live_run.lock` (`config.LIVE_RUN_LOCK_FILE`), shared by every checkout and worktree, since they all trade one account. `main.py` takes it (`acquire()`) for a production run that sends orders, before it builds a client, and releases it when the run ends; a second such run retries for `config.LIVE_RUN_LOCK_WAIT_SECONDS` (2 s) and then exits `50` without contacting Kalshi. The operating system drops the lock however the holder ends, so a crash cannot leave it stuck. The file also records the holder's process id, checkout and start time (`holder()`), which the refusal and the scheduler's message print; `held()` checks the lock without keeping it. |
| `historical.py` | Fetches and disk-caches historical settled market metadata (from two API endpoints, sharded into parallel per-day slices that are cached individually so interrupted or repeated fetches resume instead of re-walking months of history) and hourly candlestick price series needed by the backtester (candlesticks are fetched in parallel across tickers and cached per ticker, so workers never share a cache file and a repeat run re-reads them from disk; each candle also carries the contracts traded in its hour, and a cached file is reused only under the current candle-fields version). Also caches Kalshi's /series listing (`load_series_categories()`, each series' category and tags, refreshed weekly) and owns the one rule that files an event under a category and FIRST tag (`series_ticker()` / `series_labels()`), which both the backtest dashboard and the live category/tag filter (`main._filter_by_category`) use. It also holds the candle bid rule a sale reads (`usable_candle_ask()` and `candle_sale_bids()`, which the backtest and live selling both use, and `bid_before()`, which only live selling uses, to read a recent candle's bid) and `recent_candles()`, which fetches a market's last few days of hourly candles for a live run's earlier checks without reading or writing the candle cache. |
| `treasury.py` | Downloads the 8-week U.S. Treasury bill's auction yields from the Treasury's Fiscal Data API (one open, no-key, read-only GET, retried like every other market-data read) for the backtest dashboard's Sharpe and Sortino ratios to subtract — on each day of a curve, the yield of the most recent auction on or before that day, not one fixed hurdle. Every successful download is saved under `backtest_cache/treasury_bill_rates.json`; the loader never raises — on any failure it falls back to the last saved copy, and with none to "unavailable" (every ratio then subtracts 0%, and the dashboard's header says so). Reporting only: nothing sizes, prices or settles on it, and no order-path module may import it (`live_portfolio.py` does, for the Live trading tab's ratios, and `live_dashboard.py` downloads the yields; both only read). Every Sharpe and Sortino on the dashboard subtracts it — the performance cards, both benchmark rows, the per-`k` table, the scenario explorer and every filter-bar view — through `dashboard.generate_dashboard(risk_free=...)`, whose header line names the rate (or says none was supplied). A strategy curve is charged the yield only on its capital in open trades (each day, the previous close's share of the portfolio held in open trades, valued as the curve values them — at market, or at cost without fees for a trade with no quotes — so a trade is charged for the days it is held); idle cash is taken to earn the same yield, since the backtester books it at 0%. The S&P 500 row is fully invested and is charged the whole yield. `backtest.py` calls `load_risk_free_rates()` beside its series-category read and hands the result in. |
| `depth_model.py` | The backtest's model of order-book depth. Backtest only: no live module imports it; `backtester.py` and `backtest.py` do. Kalshi keeps no historical order books, so this saves snapshots of live books, fits a table of how many contracts typically rest near the best bid (the median over every saved ladder, by 24-hour volume and best-bid price; a cell with too few ladders reads its volume row, then the whole table), and builds a synthetic book from it for any market and moment (`book()`), whose top levels reproduce the quoted prices and which the backtest sizes each trade over; a backtest sale walks one side of it (`bid_ladder()`, for a bid from 1c to 99c). `volume_24h()` is the one definition of a market's trailing volume, read from the candles at every checkpoint and at each daily check before a sale. `python3 -m kalshi_betting.depth_model snapshot [--markets N]` saves a snapshot (read-only requests, no orders; about 2,000 open non-combo markets, under `backtest_cache/depth_snapshots/`); `load_depth_model()` fits every saved snapshot and returns `None`, with a WARNING, when none is usable; `backtest.py` calls it before the fetch. |
| `backtester.py` | Replays the strategy on settled markets: groups them into candidate pairs (including, behind `TIME_SERIES_SAME_EVENT_LADDERS`, two dated rungs of one event's deadline ladder — formed by a separate, deliberately unwindowed per-event sub-pass and ordered and gapped on their stated deadlines through the same `scanner.stated_deadline()` / `same_event_ladder()` the live finder uses), scans weekly snapshots at the live run's own instant — `config.SCHEDULED_RUN`, Monday 09:00 America/Los_Angeles (16:00 UTC under daylight time, 17:00 UTC under standard time) — for every Monday each pair was tradeable, refusing before any fetch a schedule whose run time falls on another date in UTC or on a clock change — at a BACKTEST time-series spread band (`_entries_for_band()`, `_find_entry()`) that never reaches live trading, which reads its own band, the saved live defaults' (or `main.py`'s own `--spread-min` / `--spread-max` for one run) — the two apply the same spread tests, pinned to one verdict by `tests/test_backtester.py::TestLiveBacktestSpreadParity` — enters each time-series pair on the earliest of those Mondays whose Kelly fraction is positive at the simulated `k` and on which it can be taken, and each same-title pair on its first such Monday or not at all, holding at most one open time-series pair per ladder — a time-series pair skipped one Monday for a busy ladder or for cash that cannot buy one contract pair is tried again on its next such Monday (as the weekly live run would), while a same-title pair keeps the one-per-group rule and is not tried again — applies Kelly sizing on each Monday's portfolio value (the cash plus every open trade at market: each leg at the latest usable ask of the side it holds at the checkpoint, at its payout once its market has paid out, and the whole trade at cost when it has no quotes) and on the cash left, through the live sizing code (`scanner._enrich_pair` walks a modeled order book built from the depth model and each market's 24-hour volume, then `strategy.compute_trade` picks the count; with no depth model, or no volume data for a market that Monday, the trade is sized at the candle prices, the top of the book, and counted), never spending more than the cash left, so a trade the cash left cannot buy in full is shrunk to what it can buy (as a live run sizes on its cash plus Kalshi's value of its positions; the two valuations of open positions can differ either way), records the average price each trade paid on each leg (`BacktestTrade.fill_price_a`/`_b`) and its actual P&L from settlement outcomes, and builds a daily equity curve that opens one day before the start date at the untouched initial balance, so a trade entering on the first day of the window shows its day-0 charges as a real daily return and a real drawdown. The curve is a portfolio value, not a cash balance: an open position is valued at market at the end of each day it is held (each leg at the latest usable ask of the side it holds, from the backtest's own hourly candles — an empty book, an unreadable candle or a NO ask at the fetch's 0.99 clamp keeps the last usable one — the price it paid before its side has had a usable ask, its payout once its market has paid out; the whole trade at cost when it has no quotes), so committing capital does not move the curve, a swing while a trade is open does, and every curve still ends at the starting balance plus every trade's profit. The quotes are sampled once per market from the candles already fetched (`_attach_leg_quotes`) and carried on each trade as `BacktestTrade.marks`; open trades were valued at cost until 2026-09-30, so results recorded before then are not comparable. The work is split at the band and the interval discount `k`: `_prepare_candidates()` (fetch through candlesticks) depends on neither, `_entries_for_band()` depends only on the band, and `_simulate_at_discount()` — the Kelly gate, dedup, P&L, equity curve — depends on `k`; `_sweep_from_candidates()` composes all three into the band x `k` x population scenario grid (`SweepPoint`, `HalfSplit`, `BacktestSweep`) the dashboard's scenario explorer renders — and, with `tier_off_sweep` (which the CLI turns on with the band sweep), re-runs every band a deadline-gap tier floor binds at with the tier floors off, so that band's floor alone gates the spread (`BacktestSweep.tier_off_scenarios`, backtest only; every tier-on figure unchanged). Pair extraction applies the live scanner's same-title close gate through the same `scanner.closes_apart()` (DR-74), and accounts for every pair of every group's members, each on its own silent-at-zero line — the three wording reasons, the one-series rule, the same-event skip, the same-title close gate (plus a backtest-only line for a same-title pair whose close time cannot be read), each ladder reason and, for time-series groups, the pairs its close-date window never visits and the members with no readable close time — and the size of each grouping is logged on every run, zero included, so a run that forms no pairs still logs why (M10). The per-trade Kelly size cap is a simulation parameter too (`_simulate_at_discount(size_cap=)`, default `config.BUDGET_FRACTION`; a same-title candidate also stays under `config.SAME_TITLE_SIZE_CAP` at every cap, through `config.pair_size_cap`, as live sizing caps it), and `run_backtest_sweep(cap_sweep=True)` — on by default in the CLI — adds a lazy `CapSweep` over every other cap of `SIZE_CAP_SWEEP` (5% to 95%, and 1.0, shown as off) for the tier-on grid, and a second one over the tier-floors-off runs (`BacktestSweep.tier_off_cap_sweep`, seeded from their own points) when those ran too, each simulated one (band, `k`) cell at a time when a report reads it. `run_backtest_sweep(add_on_sweep=True)` — also on by default in the CLI — adds the dashboard's "Add to held pairs" family the same lazy way: `BacktestSweep.add_on_cap_sweep` (and `add_on_tier_off_cap_sweep` over the tier-floors-off runs), whose every simulation may add to a pair it still holds (`_simulate_at_discount(add_to_held=True)`, an add-on sized on the same portfolio value, the pair's stake its open trades at market plus the fees paid for them), for the "all" population only; without it a pair trades at most once. |
| `dashboard.py` | Generates an HTML performance report from backtest results, self-contained but for the Sell select's chunk files (`backtest_dashboard_files/` beside it) — nine sections: cumulative return lines (total plus one per trade type) / Sharpe/Sortino/drawdown KPIs plus mean and median return per trade and the median monthly return, returns decomposition (P&L by Kalshi's official category and by category · tag, with a per-group table), time-series spread calibration (each traded pair's entry mid spread — the later market's midpoint minus the earlier one's, a midpoint being halfway between a market's YES ask and its YES bid: the market-implied probability that the event lands between the two deadlines, and the input the forecast reads — against the rate the pairs settled A = NO, B = YES, with Brier score and log loss; same-title trades are not shown), an interval-discount (`k`) calibration section (the pooled empirical `k̂`, one equity curve and a per-`k` table at the primary spread band with the tier floors on, which the filter bar's `k` and size cap move), two Portfolio Performance cards with the selection's empirical `k̂` and `k̂ − k` (the `k̂ − k` red when positive, i.e. the sizer sized too big), an empirical `k̂` breakdown (a bar chart and table of `k̂` by Kalshi category, by tag or by spread band, each bar naming its entries and distinct events, with a Group-by `<select>` of its own and a dashed line at the page's `k` — the run's own as rendered, the filter bar's once one is chosen), a scenario-explorer section over every spread band x `k` x per-trade size cap of the band sweep (a fragility banner for the cap shown, a spread-band x `k` heatmap whose metric menu offers mean, median and total return, Sharpe ratio (blank on a cell with no trade), H1/H2 returns, trade count, each band's empirical `k̂` and `k̂ − k` — both red where `k̂` exceeds `k`, as on the cards — a one-row-per-band `k̂` table, and a per-population KPI table, with band, `k` and size-cap `<select>`s of its own — and a Tier floors `<select>` when the run carries the band sweep's tier-floors-off runs — that the filter bar also moves; its data comes from the same one walk as the bar's and ships as one gzip-packed block per size cap, plus a tier-floors-off block per size cap, each unpacked only when chosen), trade diagnostics (best and worst five trades with each leg's side, price, close date and settlement), risk metrics, and an S&P 500 benchmark comparison whose download window opens on the same date as the equity curve's leading initial-balance row. The page header names the run's primary spread band, same-event-ladder setting and per-trade size cap (and whether the size-cap sweep ran), and, on the lines under them, the entry checkpoint every trade was entered at — the live scheduler's weekly run time, `config.SCHEDULED_RUN` — and how the primary scenario's trades filled: how many walked a modeled order book (and from which depth snapshots) and how many filled at the top of the book, or, in red, that no depth snapshot was usable so every trade did. The trade rows show the average price each leg paid, and the Kelly scatter prices the legs at it, with a time-series trade's chance of profit read from the mid spread of its entry quotes, as the live sizer reads it. A sticky filter bar at the top — Spread band, Tier floors, k, Size cap, Add to held pairs, Sell, Min. days to maturity, Category, Tag — re-scopes every trade-derived section (performance, decomposition, calibration, diagnostics, risk, the benchmark's strategy row) to another scenario's own run — a spread band at a `k` and a per-trade size cap (5% to 95%, or off, when the size-cap sweep ran; the run's own cap otherwise), each a standalone simulation, the caps other than the run's own simulated as the page is built — and/or to any set of its Kalshi categories and category · tags (the series' first tag, so breakdowns partition; Category and Tag are check-box menus, and a ticked category counts in full unless some of its tags are ticked, then only those, as the live filter reads them) of that run, whose return, drawdown, Sharpe, Sortino, median monthly return and benchmark row are then its contribution — the starting balance plus its trades' P&L as the run booked them — not a standalone simulation. One category or one tag shows figures computed when the page was built; several together are combined by the page itself from each tag's own figures, and the summary line says so. Its Tier floors choice switches every band between the run as simulated, with the 0.15/0.30 deadline-gap tier floors applied, and the band's run with them off — its own floor alone, from the tier-floors-off runs the CLI's band sweep also simulates (backtest only, at every size cap: the run's own cap from those runs themselves, every other from the lazy tier-floors-off size-cap sweep, simulated as the page is built — a run or a page without that sweep has a band the tiers bind at with the tier floors off at the run's own cap only, and the bar says so at any other); a band whose floor sits at or above both tiers was never simulated again, because the tiers never bind there, so its off view is its tier-on run, at every size cap, and says so, and a run without those runs (`--no-band-sweep`) keeps the select disabled with a "(not simulated for this run)" note. The header's trade count follows the selection too, the `k̂` breakdown and the two `k̂` cards move to the same band, tier setting and selection (the breakdown's dashed reference line, and the `k̂ − k` card, to the chosen `k`), and the Interval Discount (k) section to the chosen `k` and size cap (still at the primary spread band, with the tier floors on — it never follows the Tier floors choice) — that chart regroups the band's own `k̂` population (every time-series candidate entry, measured before the Kelly gate), not the run's trades. Every view is computed in Python by the helpers the sections render with and shipped gzip-packed — one small base block, and one chunk per distinct scenario trade list (scenarios that traded equal lists at one `k` share it), which the page unpacks only when that scenario is chosen, so a large grid does not slow the page's load; the page is streamed to disk piece by piece, never joined into one string (each packed chunk is held until it is written). The bar's band, Tier floors choice, `k` and size cap also move the Scenario Explorer's own selects to the same scenario — each only when the bar's choice on it changes, so a choice made in the explorer survives a category or tag change (they stay usable on their own), and a move the explorer has to refuse (a size cap it holds no tier-floors-off data for — none when the tier-floors-off size-cap sweep could be read; otherwise every cap but the run's own, whatever the band, unlike the bar — or a block it cannot unpack) is named on its status line and applied again on the bar's next change; category and tag reach neither it nor the Interval Discount (k) section. Set to off, the explorer's banner, heatmap and `k̂`-by-band table switch to the tier-floors-off grid and its KPI table, calibration table and equity curve read the tier-off cells — a band the tiers never bind at shows its one run, and the Same-title row is the same either way — and its equity curve re-autoranges on every redraw, so a zoom never carries into another band, `k`, cap or setting; the bar's summary line says what it reaches on that page. Its Add to held pairs choice — off, the run as simulated, or on (up to the size cap) — shows the same scenario re-simulated so a pair the run still holds can be bought again on a later Monday, sized so old and new together, with the fees paid, stay within the size cap's share of the portfolio value. Like the size cap it re-scopes the trade sections, the header's trade count and a category or tag slice, and the button below then sends `add_to_held_pairs` for it (when the saved live defaults add to held pairs, a grey note beside the button says that saving with the choice off turns adding off); the `k̂` figures do not depend on it, and the Scenario Explorer and the Interval Discount (k) section never follow it and always show it off, which the summary line says. While adding is on the summary line also says that each purchase that adds to a held pair counts as a trade of its own and, in the Risk section's Kelly chart, shows only what it added. The select stays disabled with a short note where the page cannot show it: "(not simulated in this backtest)" for a run without the family (`--no-add-on-sweep`), "(not available on this page; see the log)" for one the page could not use. With the tier floors off it is shown at a band the tiers never bind at (off and on are one run there) and at each band the run's tier-floors-off add-on runs cover; at any other band it reads as not simulated. Its Sell choice — no selling, or sell at 80% to 100% of potential profit — shows the same scenario with each position sold whole once it has held that share of its potential profit for 3 days in a row, and its Min. days to maturity choice (1 to 7, 14 or 21 days; shut under no selling) holds a sale back unless at least that many days remain before the position's last market stops trading (see [Selling at a share of potential profit](#selling-at-a-share-of-potential-profit)); every level and minimum of every scenario is read after the walk, in worker processes, simulating only the settings that differ, each new trade list is a file in `backtest_dashboard_files/<build>/` beside the page, and each band's chunk ids under each Tier floors setting are one block in the page (a band the tier floors never bind at uses its tier-on block for both), unpacked the first time a level is chosen there — both loaded only when chosen. Like Add to held pairs it re-scopes the trade sections only, and Save as live defaults… sends the level and minimum shown as the live sell level and minimum of days (or `off` for both under no selling), so a live run then sells at them. The `k̂` cards are computed from the primary calibration itself, so they render even when the bar cannot be built. The bar's selects start disabled and are enabled once the page has unpacked its base block and the run's own scenario (the Tier floors select only when the run carries its tier-floors-off runs, the Add to held pairs select only when it carries the add-on family, the Sell select only when it carries the sell family, and the Min. days to maturity select only while Sell names a level), and each chart is redrawn from its layout as first drawn, so a zoom never carries into another selection; a later choice supersedes one still loading, and a scenario the run never simulated says so. If the filter's data cannot be built, the page is still written, without the bar and with a notice in its place; if the size-cap sweep cannot be simulated, the bar offers the run's own cap only and the header says the sweep could not be used; if only its tier-floors-off half cannot be, the bar keeps every cap with the tier floors on, offers the run's own cap with them off, and the header says so. After the Tag menu the bar carries a "Save as live defaults…" button, rendered disabled and enabled once the page has loaded the scenario on screen, whenever that scenario can become the live settings (simulated, its band recorded, its `k` and size cap recorded and above zero, and a category or tag only on a page built with Kalshi's series listing; each ticked category is saved by name and each ticked tag under its own category); a click opens the defaults server's confirmation page (started by `./start_dashboard.sh`, on `127.0.0.1:8765`) in a new tab for the bar's scenario on screen — never the Scenario Explorer's own selects — with the run's same-title cap, and that page saves it only on its own Confirm and save (or Confirm and trade, which saves before it runs). Beside it a plain "Trade using defaults…" link, which needs no script, opens the server's trade page (the saved defaults, with Dry run and Confirm and trade) in a new tab. A page whose filter bar could not be built has no save button, but keeps the trade link under the notice in the bar's place; the link's id is how the server tells a new page from one built before these buttons. Every run overwrites the one `backtest_dashboard.html` in the repo root and replaces its `backtest_dashboard_files/` folder. |
| `backtest.py` | CLI entry point for the backtest pipeline. Parses arguments (including `--interval-discount`, `--no-sweep`, `--same-event-ladders` / `--no-same-event-ladders`, the backtest-only `--spread-min` / `--spread-max` / `--no-band-sweep`, `--no-cap-sweep`, `--no-add-on-sweep`, `--no-sell-sweep` and `--sell-workers`), builds the historical API clients, takes the starting balance from `--balance` or else reads what the Kalshi account is worth now (`auth.read_account_balance`: its cash plus Kalshi's value of its open positions), fits the depth model from the saved order-book snapshots before the fetch (`depth_model.load_depth_model()`; its config echo ends with `fills=walked book (...)` or `fills=top of book (no usable depth snapshot)`), calls `backtester.run_backtest_sweep()` then `dashboard.generate_dashboard()` (whose header says where the starting balance came from), and logs a summary of the primary result. |
| `defaults_server.py` | Human-run local web server (`./start_dashboard.sh [--seed]`, which runs `python3 -m kalshi_betting.defaults_server [--seed] [--no-browser]` from its own checkout, in the background beside the live dashboard, on `127.0.0.1:8765`) whose pages save the live defaults, `live_defaults.json`, and start live trading runs with them (see [Trade from the browser](#trade-from-the-browser)). Its confirmation page is opened with the proposed settings in its address (by `--seed`, which proposes the seed values, or by the backtest dashboard) and shows the defaults in force beside the proposed ones with every change highlighted and, in red, every warning a live run would log about the proposed settings; its trade page (`/trade`) shows the saved defaults. Their buttons, always in one order — Dry run, Confirm and save (the confirmation page only), then, set apart below a red line, Confirm and trade — save the file (through `config.save_live_defaults`) or start `python -m kalshi_betting.main --mode prod` with every toggle as a flag, in its own session, its folder under `live_runs/`, and send the browser to the run's page (`/runs/<id>`), which follows the run and then sums up its outcome from the result file the run writes (with a Sales table when the run sold). Closing a tab cancels; a button that does not apply is shown disabled with its reason. It never serves the dashboard, listens on loopback only, answers one request at a time, and refuses a save or a run unless the request names this server (Host), comes from its own page (Origin), carries the token and the one-time nonce that page was built with, and finds the same defaults still in force (a page gone stale is shown again against the defaults now in force); a page posted twice does its work once. A run is refused while one it started is still going, a real-money run while another live trading run holds the machine's lock, and a real-money run after one that needed attention until a box is ticked. Its buttons are enabled only after the page has been visible for a second and the mouse moves or a key is pressed, and disabled once one is clicked. It never places an order itself and never stops a run. On start it opens the page asked for (the launcher passes it `--no-browser` unless `--seed`, since the live dashboard opens the page): with `--seed` the seed values' confirmation page; otherwise the backtest dashboard, or its own index when there is none, the dashboard was built before its Save and Trade buttons, or it cannot be read (it reads the first 1 MiB of the file for the Trade link's id, and logs a WARNING naming the rebuild, or the reason it could not read it). When its port is already taken, it binds nothing and starts nothing: if the listener's `/checkout` names this checkout and the code this checkout has now, it opens that page from the running server and exits 0; if it names another checkout, or this one with code from before a change (a pull or an edit since that server started — stop it and start it again), or the listener is not a defaults server, it exits 2. Logs to `kalshi_defaults_server.log` (a start that finds its own server running logs to the terminal only). Imports `config.py` and `run_lock.py` only; nothing imports it. |
| `live_portfolio.py` | Works out how the live account is doing, for the Live trading tab (reporting only: nothing trades, sizes or saves on it). Reads the account from Kalshi with read-only GETs — every fill, settlement, deposit and withdrawal, then the balance and the positions — and the bot's trade log; matches each of the bot's purchases to the Kalshi order that made it (every other fill is "Other bets", the bot's sale orders included, which are read as sales by hand; a sale row in the log is never a purchase); replays every fill and payout, oldest first, into a ledger of each owner's contracts and cash (a fill's direction from its `book_side` alone); values the account over time (each holding at its midpoint) just before the bot's first trade, at each of Kalshi's daily closes, at each deposit or withdrawal and now, by category; and works out each period's statistics (total return, profit, Sharpe and Sortino, mean and median trade, each category's return), the holdings now and whether the cash each run logged (above its sales, when it sold, and above its purchases) matches Kalshi's records. Each read adds one JSON line to `live_portfolio_log.jsonl`. See [Live trading tab](#live-trading-tab). |
| `live_dashboard.py` | Human-run, read-only local web server for the dashboard's two tabs (`python3 -m kalshi_betting.live_dashboard [--no-browser]`, or started by `./start_dashboard.sh`): the Live trading tab and its data on `127.0.0.1:8766` (the account read from Kalshi through `live_portfolio.py` on every page load), and the backtest page and its chunk files on `127.0.0.1:8767`, which the Backtest tab shows in a frame. Two ports are two web origins, so nothing on the backtest page can read the account. It formats every number and sentence the page shows, reads one account at a time, sends Kalshi GET requests only, never loads the order path, and refuses requests from other web sites. A second start reuses this checkout's running dashboard (it opens the page and exits); a port held by anything else is refused (exit 2). Logs to `kalshi_live_dashboard.log`. Imports `config.py`, `live_portfolio.py`, `historical.py`, `treasury.py` and `_http.py`; nothing imports it. |
| `v2_probe.py` | Human-run CLI that verifies the V2 order path's NO-leg mapping, fill-or-kill kill semantics, and the inter-shard transfer's centicent unit against the production account for roughly one cent of exposure. Its `--step yes-close` also verifies the one order shape live selling sends that no other step does, a reduce-only ask that sells a held YES: it buys 0.01 YES, sells it back, and PASSes only when the account reads flat and the sale's reply reports the 0.01 sold. Its closing reduce-only bid is priced at the top of the market's own grid (0.99 / 0.999 / 0.9999 by tick regime), not at the rollback builder's loss floor, so that floor can no longer cause a FAIL unrelated to the mapping (a book with no reachable resting YES ask still can); `reduce_only` is what bounds that bid. A 2xx order body that is not a JSON object, in either step that reads one (DR-58), and — in the NO-buy step only — an object whose fill counts are unreadable (DR-60), are a clean FAIL that still reads the position, re-reads it once when that first read is `None` or `0`, and reports lookup-failed, position-open and genuinely-flat as three distinct outcomes, never a traceback out of the fill readers. Two of the unfillable-ask step's branches are recorded residuals — its unreadable-fill-counts branch and its `not killed` branch both FAIL without re-reading the account. The NO-buy step classifies readable fill counts into three outcomes, not two — a complete fill, a true kill, and a fill-or-kill invariant violation — so a partial fill FAILs naming the counts and re-reading the account rather than being reported as a clean kill with the account "still flat" (DR-20); the unfillable-ask step always read a partial that way. Both steps judge a 2xx on `fill_count` AND `remaining_count`, which is deliberately stricter than the live `trader._v2_fill_status`, whose contract is `fill_count` alone. The exchange's HTTP 409 kill response to a fill-or-kill (`trader._is_fok_kill`) is read as a kill — PASS on the unfillable ask, NEUTRAL on the NO buy, when the account reads flat — and the close's verdict is judged on one re-read when the read after it is not exactly 0. `--dest-shard` equal to the source shard is refused with a NEUTRAL at the top of the transfer step, before any transfer I/O, so it can no longer POST a net-zero self-transfer and then report a false in-flight FAIL (DR-22). It refuses to start (exit 2) on any `config.ORDER_API_VERSION` but `"v2"`. The FAIL lines that doubt the V2 path after a submission on an account the probe checked was flat tell the operator to stop trading and flatten any position on the probed ticker by hand in the Kalshi UI. The closing line depends on the result: after a FAIL it says to stop trading — and, for an order step, to act only on the position warnings printed above, never by itself to flatten, since a FAIL before anything was submitted opened nothing and the ticker may carry the bot's own position; for the transfer step, to check each shard's balance — and after a NEUTRAL it says nothing needs doing. Its informational fee check compares the exchange's `average_fee_paid` — per contract, by Kalshi's API reference, and including Kalshi's rounding of the order's total fee up to the account's balance precision — with the fee model per contract at the fill price: before rounding, as `config.fee_leg_exact(1, p)` for one whole contract, and rounded as Kalshi rounds the probe's order, which is the figure a 0.01-contract charge should match. Never imported by the pipeline. |

### Order API version

`config.ORDER_API_VERSION` names the bot's order path, and `"v2"` is the only value it accepts: `main.py` and `v2_probe.py` call `config.order_api_version_error()` right after parsing their arguments and exit 2, with the reason on stderr, before logging is configured or any request is made, on anything else (another spelling such as `"V2"`, an empty value, `"legacy"`, …). The V2 path posts to `/portfolio/events/orders`: a fill-or-kill **limit** order with a dollar-string price, a fixed-point contract count, a `bid`/`ask` side on the market's single YES book, and an explicit `exchange_index`. V2 has no "market" order type, so the limit price is itself the price protection — the scanned price rounded up onto the market's own tick grid plus `BUY_SLIPPAGE_TICKS` ticks, which is a cap the older integer-cent `buy_max_cost` field could not express once MVE/combo markets moved to sub-cent ticks. A price sitting exactly on the boundary between two tick bands belongs to both, and the **finest** of them wins: taking the first match instead made the allowance ten times coarser at a band's upper edge, loosening a cap that is a bid. Because that limit applies per contract while the scanned price is an average over several book levels, `strategy.py` checks a candidate size against it before committing — otherwise the order asks for depth priced above its own limit and the whole fill-or-kill is killed.

Every V2 body also carries `self_trade_prevention_type`, a field the endpoint requires (it rejects a body without it with HTTP 400). The bot sends `config.V2_SELF_TRADE_PREVENTION_TYPE`, `"taker_at_cross"`: an order that would trade against another order on the same account is cancelled at that point. The two buy legs are fill-or-kill. The exchange kills one that cannot fill in full with an HTTP 409 error, code `fill_or_kill_insufficient_resting_volume` (`config.V2_FOK_KILL_HTTP_STATUS` / `V2_FOK_KILL_ERROR_CODE`), and the trader reads exactly that response as a clean non-fill — the pair fails, or rolls back, at once; any other error still goes to the position check. The rollback that unwinds a filled NO leg is not fill-or-kill. It is `reduce_only`, which the endpoint accepts only with `immediate_or_cancel`, so it buys back what rests at or under its loss-floored cap and cancels the rest. Its count is always this pair's NO count, and it is sent only once this pair's NO order is known to have filled, so on a market the account already held it closes only this pair's contracts. A close of only part of them is reported as `rollback_failed` for manual review, never as closed, and no second order is sent; its alert names how many NO contracts are still open — the NO leg's count minus the fill count the rollback's own response reports — or says "up to" the NO leg's count when there is no usable count (an error response, a transport error, or a body with no readable count).

The V2 side mapping — an `ask` on the YES book opens a NO position — comes from Kalshi's docs, so the first NO leg that fills in each process is checked against the account's positions (`trader._confirm_v2_no_mapping`): the position must move by exactly minus the contracts bought. Until that check gives a verdict, the run's pairs execute one at a time, for at most `config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS` (300 s; a killed NO leg or an unreadable account gives no verdict, and the next pair runs alone in turn). If the position moved any other way — an unchanged position is re-read after 1, 2 and 4 s first, since the ledger lags a fill by about a second — the mapping is disproven: that pair stops with its YES leg unsent and its NO leg left for a human (`manual_review`), a CRITICAL says so and names any earlier pair that went ahead unchecked, and every pair that starts after it is stopped before anything is sent (`failed`) — within the 300 s that is every later pair; after it, pairs already running still send their NO legs. The run exits 20. The stop lasts only for the process, so each scheduled run checks again: after a disproof, stop the scheduler and undo by hand in the Kalshi UI what the CRITICAL names — flatten a market the account did not hold before the pair, and put one it did hold (a pair it added to, or an earlier trade) back to what it held, which the CRITICAL states in plain words (for example "30 NO contracts (-30)"), since closing it out would close those older contracts too. Each run the defaults server's Confirm and trade starts is a new process too, so stop that server as well if it is running (Ctrl-C in the terminal running `./start_dashboard.sh` or `python3 -m kalshi_betting.defaults_server`), and do not press Confirm and trade.

The arithmetic itself lives in `scanner.py` (`v2_limit_price()` builds the wire price, `v2_effective_cap()` states it in the leg's own side terms) and `trader.py` re-exports it. That is deliberate: `trader.py` imports both `scanner.py` and `strategy.py`, so neither can import it back, and a second copy of the formula is exactly how the size and the limit drift apart again.

There is no other order path to switch to. Kalshi deprecated its legacy `/portfolio/orders` order endpoints in June 2026 (its changelog entry of June 18, 2026: once deprecated, a call returns `Please switch to the V2 endpoints`), and the V2 side mapping was confirmed live on 2026-09-28 (a `v2_probe` PASS and the first live trades). If the V2 path ever misbehaves — `trader.py`'s NO-leg mapping check logs a CRITICAL, or a `v2_probe` step FAILs — stop trading (stop the scheduler daemon if it is running, and do not run `main.py --mode prod`; if the defaults server is running, stop it with Ctrl-C in the terminal running `./start_dashboard.sh` or `python3 -m kalshi_betting.defaults_server`, and do not press Confirm and trade) and flatten any open position by hand in the Kalshi UI — except on a market the account held before the pair, which goes back to what it held. The CRITICAL says exactly that, naming that holding when there is one; the probe's FAIL lines that follow a submission on an account it had checked was flat say to flatten, since a position on the probed ticker can only be the probe's own. After any other probe FAIL, act only on the position warnings it printed, since a position on the ticker may be the bot's own. Every pair's legs are submitted in one order (the NO leg first — it is the leg that gets unwound, and `trader._ordered_legs` decides which market that is for the pair type), and no submission is ever retried.

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
  start_dashboard.sh              ← Starts the live dashboard and the defaults server and opens the
                                    page on its Live trading tab (./start_dashboard.sh [--seed]
                                    [--no-browser]; KALSHI_PYTHON picks the Python)
  kalshi_live_dashboard.log       ← The live dashboard's log (auto-created; rotates 5 MB x 3)
  live_portfolio_log.jsonl        ← One line per load of the Live trading tab: the cash and every
                                    holding with its value (gitignored)
  live_defaults.json              ← The saved live defaults every live run starts from (gitignored;
                                    written only by defaults_server's Confirm and save or Confirm
                                    and trade; no file, no live run — see Live trading toggles)
  kalshi_defaults_server.log      ← The defaults server's log (auto-created; rotates 5 MB x 3)
  live_runs/                      ← One folder per run the defaults server starts (gitignored):
                                    run.json (what the run is), output.log (everything it
                                    printed) and result.json (what it did)
  backtest_dashboard.html         ← Backtest HTML dashboard (rewritten by every run)
  backtest_dashboard_files/       ← The data the dashboard's Sell select loads, one folder per
                                    build (keep it beside the page; each build deletes the last's)
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
    candlesticks/                 ← Per-ticker hourly price series (each hour's traded volume too)
    depth_snapshots/              ← Saved live order books for the depth model, one file per
                                    `depth_model snapshot` run
    live_marks/                   ← The Live trading tab's daily prices of finalized markets read
                                    from Kalshi's archive (they never change), one file per market
  kalshi_betting/                 ← Python package

~/.kalshi_betting/                ← In your home folder, shared by every checkout and worktree
  live_run.lock                   ← Locked by a production run that sends orders while it runs,
                                    and records who holds it; one real-money run at a time on
                                    this machine (auto-created)
```

---

## Run Commands

`main.py` and `scheduler.py` echo log output to the terminal as well as their log
file. `backtest.py` does not: it installs only a `RotatingFileHandler` on
`kalshi_backtest.log`, so a backtest run's progress is visible only by tailing
that file, not in the terminal that launched it.

### Save the live defaults (do this first after upgrading)

```bash
./start_dashboard.sh                                                       # step 1, then open its link below
python3 -m kalshi_betting.main --mode prod --dry-run --add-to-held-pairs   # step 2
./start_dashboard.sh --seed                                                # step 3, once step 2 checks out
```

Every live run starts from the saved live defaults, `live_defaults.json`, and
refuses to start (exit `2`) while none are saved — the Monday scheduled run
included, and hand runs (dry runs, and runs with toggle flags) too. The seed
values that `./start_dashboard.sh --seed` proposes turn **adding to held pairs
on**, so do not start with them: save them with adding off, check the held
pairs with a dry run, and only then turn adding on. After pulling this change
into the checkout the scheduler runs from, in that checkout:

1. **Save the seed values with adding to held pairs off.** Start the defaults
   server (`./start_dashboard.sh`) and open this link in the browser:
   `http://127.0.0.1:8765/confirm?tier_floors=off&spread_min=0.0&spread_max=0.5&k=0.8&size_cap=0.1&same_title_size_cap=0.2&add_to_held_pairs=off`.
   Review the table, click Confirm and save, and check that the page it lands
   on (`/saved`) names that checkout's `live_defaults.json`. The link carries
   no `source` note on purpose: the seed's note may label only the seed values
   themselves, and the server refuses it on any others. From then on every
   live run, the Monday one included, trades these values and adds to
   nothing. The seed caps each trade at 10% of the portfolio value; a `main.py` flag
   still overrides any toggle for one run. A checkout whose defaults were saved
   before this change starts at step 2: that file reads adding as off.
2. **Check the held pairs with a dry run.** Run
   `python3 -m kalshi_betting.main --mode prod --dry-run --add-to-held-pairs`
   (it sends no orders and saves nothing). In `kalshi_arb.log`, check first
   that the `Sizing on portfolio value $X = cash $Y + open positions $Z` line
   shows the open positions as a dollar figure (not `not read`), with no
   `Kalshi's value of the open positions (…) is not used` WARNING after it:
   either one means the run adds to no held pair, and it says so
   (`Not adding to held pairs this run: …`). Then check each
   `Held pair to add to: …` line (and each `Held market to add to (its
   partner has paid out): …` line, the same way, for its one market): its `cost $C (fees $F)` against that pair's
   fills in the Kalshi UI (count × price plus fees for the cost, the fees
   alone for `$F`, both legs together), and its `worth $W at today's prices`
   against the asks the Kalshi UI shows (count × the ask of the side held,
   both legs together). If the value is missing or refused, or a cost or its
   fees are wrong, stop here and leave adding off.
3. **Only then turn adding to held pairs on for every run.** Run
   `./start_dashboard.sh --seed` and click Confirm and save (after step 1 the
   page shows 1 of 10 settings change), or open the step 1 link with
   `add_to_held_pairs=on`; either also sets every other toggle to the seed's.
   Check that the next run's `Live settings:` line ends `add to held pairs on`.

   A shorter order, if you accept adding on until the check: save the seed at
   step 1 instead (`./start_dashboard.sh --seed`, Confirm and save), run step
   2's dry run without the flag before the next scheduled run or any Confirm
   and trade, and if a cost is wrong, save again through the step 1 link.
4. Rebuild the dashboard with the `--start-date` it was built from — for the
   365-day window, `python3 -m kalshi_betting.backtest --start-date 2025-09-24`
   (about 23 minutes and 2.1 GB of memory when that window's corpus is already
   cached; run nothing else heavy beside it). Without `--start-date` the
   backtest runs from 2024-01-01, a far longer fetch. Until then the old page
   has no Save or Trade buttons: the defaults server says so in its log (run
   on its own it opens its own index instead; through `./start_dashboard.sh`
   it opens nothing, and the Backtest tab shows the old page), and
   `http://127.0.0.1:8765/trade` trades the saved defaults. A dashboard
   scenario can be saved only from a rebuilt page.
5. If the scheduler daemon is running, restart it after the pull, so it runs
   the new code and names a run stopped by the live-run lock (exit `50`).
   Do step 1 before you start it: on start it can run a missed Monday slot at
   once.

Live selling is off through all of this (the seed saves no sell level); it is turned on in a later step, after the probe's `yes-close` step passes: see [Turn on live selling](#turn-on-live-selling).

`./start_dashboard.sh` (at the repo root; it runs from its own folder, so it
can be started from anywhere) checks that its Python — `python3`, or the one
`KALSHI_PYTHON` names — can import the live bot and the live dashboard from this checkout (run it by
its own path, not through a link, and without `PYTHONSAFEPATH` set; it
refuses otherwise, since the server would run another checkout's code), then
starts the live dashboard (the page with the Live trading and Backtest tabs,
on `127.0.0.1:8766`; see [Live trading tab](#live-trading-tab)) and the
defaults server, which listens on `127.0.0.1:8765` and, with
`--seed`, opens a confirmation page proposing the seed values (tier floors
off, spread band 0–0.5, `k` 0.80, a 10% per-trade cap, a 20% same-title cap,
any category or tag, adding to held pairs on, selling off). Click Confirm and save to save
them; closing the tab saves nothing. On a first save, follow the order above.
Ctrl-C stops both servers (so does a SIGTERM or SIGHUP sent to the script);
stop them when you are done. Running the script
again while this checkout's servers run starts nothing: it opens the page
again and exits. If the checkout's code has changed since that server started (a pull,
say), it is refused instead (exit `2` for the defaults server; for the live
dashboard, a warning that it did not start): stop the servers and start them
again, so the pages run the new code. A server of another checkout on a port
is refused too — stop it first. The script takes only `--seed`,
`--no-browser` and `--help`, spelled in full, and refuses anything else before
it starts anything; it passes the two flags to
`python3 -m kalshi_betting.defaults_server`, which can also be run directly
(it then opens the backtest dashboard itself, or its own index,
`http://127.0.0.1:8765/`, when there is none or it was built before the Save
and Trade buttons). The script lets the defaults server do that too when the
live dashboard stops with an error within 2 s of starting, and says so; if
the live dashboard stops later, the script says so and the defaults server
keeps running without opening anything. Without `--seed` the page opens on its Live trading tab;
its Backtest tab shows the backtest dashboard: choose a scenario in its
filter bar (spread band, tier floors, `k`, size cap, add to held pairs, Sell level and
Min. days to maturity, and the categories and tags you want ticked in its two
check-box menus) and click the bar's
"Save as live defaults…" button, which opens
the same confirmation page for that scenario, with the backtest run's own
same-title cap, in a new tab (the Scenario Explorer's own selects are not what
it saves; the choice to add to held pairs, and the Sell level with its minimum
of days, are each sent only from a page that shows them — there "no selling"
turns live selling off — and otherwise the saved choice is kept). The button stays disabled until the page has loaded the scenario on
screen, and for a scenario the run never simulated; categories and tags can be
saved only from a dashboard built with Kalshi's series listing, which is how
the live category filter files pairs. Each ticked category is sent as a
`category` and each ticked tag as a `tag` written "Category · Tag"; the button
stays disabled past 40 of either. The dashboard itself writes nothing, and
the server must be running for the button's page to open. `--no-browser` only
logs the addresses to open. The page shows the defaults in force beside the proposed ones with every
change highlighted, and shows in red each warning a live run would log about
the proposed settings (for example, a deadline-gap range in which no pair can
trade, or one pair staking more than 20% of the portfolio value). See
[Live trading toggles](#live-trading-toggles).

### Live trading tab

```bash
./start_dashboard.sh                                      # opens http://127.0.0.1:8766/ on the Live trading tab
python3 -m kalshi_betting.live_dashboard [--no-browser]   # the live dashboard alone (both tabs, read-only)
```

The page has two tabs. **Backtest** is `backtest_dashboard.html` as it is
(loaded the first time the tab is clicked). **Live trading** reads the account
from Kalshi each time the page loads or Refresh is clicked, and shows, from
the bot's first live trade (Sep 28, 2026):

- Total return, Profit, Sharpe, Sortino, Mean trade and Median trade, for
  All, 1Y, 6M, 3M and 1M.
- The account's value over time, stacked by category: Cash, up to six
  categories in colors of their own, "More categories" for the rest (a lone
  seventh keeps its own name, in gray), and "Other bets" (anything the bot did
  not buy — your own trades).
- The return of each category over the period.
- The holdings now, with Cash and the totals, and notes and warnings.

How the figures are worked out:

- **The bot's purchases** come from the trade log; each is matched to the one
  Kalshi order that made it, inside the run's own time window (the previous
  run's log time, or 6 hours before its own if that is later, to its own),
  then up to two minutes past it.
- **The bot's sales** (a run with a sell level saved sells before it buys)
  are rows of their own in the trade log, under a banner of their own; they
  are never counted as purchases or warned about. The sale orders are read
  as sales by hand: a pair's two sale orders close that pair's purchase
  first, so what they bring in goes to it, and a lone held market's sale
  closes your own contracts in that market first, then the bot's. The cash
  logged above the sales is checked against the cash rebuilt just before
  the run's first fill on the markets its sale orders may have filled on
  (never a paid-out partner, a market where nothing sold, or a fill one of
  the bot's purchases made).
- **Holdings** are valued at the midpoint of the best bid and ask (an empty
  side counts at 0 or 1; with both sides empty, the last trade); a decided
  market at its payout.
- **Total return** is time-weighted: a deposit or withdrawal is taken out at
  the moment it lands. **Sharpe and Sortino** use whole days only (one daily
  close, midnight New York, to the next), less the 8-week T-bill yield on the
  money held in positions; with fewer than two whole days they show "—".
- **Mean and median trade** count each contract pair once (open pairs at
  today's value). A category's return is its profit over what it put in (its
  value at the start plus the most cash it had out since), so a sale counts as
  cash back, never as a loss.
- **Colors** stay the same from one load to the next: each category keeps the
  color it got when the bot first bought in it, as long as the bot's purchases
  and their categories stay the same (Kalshi filing a series under another
  category, or an older trade log turning up, can move them).

It is read-only: it sends Kalshi GET requests only and never loads the order
path. The tab page and its data are on `127.0.0.1:8766` and the backtest page
on `127.0.0.1:8767`, two separate web origins, so nothing on the backtest page
can read the account. Each load adds one line to `live_portfolio_log.jsonl`.
Known limits: how deposit and withdrawal fees are charged is assumed, not checked;
interest and fee rebates are not seen; a run whose trade-log write failed
counts as Other bets; the trade log's times are this computer's local time;
a sale of a bot pair, your own or the bot's, is recognised only when both
markets are sold within 10 minutes; a read has no overall time limit; and with a short history
Sharpe and Sortino are noisy.

### Dashboard buttons

```bash
./start_dashboard.sh           # starts both servers; the Backtest tab shows backtest_dashboard.html
```

The backtest dashboard's filter bar (on the Backtest tab, or the page opened
on its own) carries two buttons, and both open a page of the
defaults server in a new tab, so the server must be running
(`./start_dashboard.sh`); the dashboard itself writes nothing and starts
nothing.

- **Save as live defaults…** opens the confirmation page for the scenario the
  filter bar shows (spread band, tier floors, `k`, size cap, add to held pairs, Sell level and minimum days, and
  the categories and tags ticked), beside the live defaults in force with every change
  highlighted. It sends the Sell level and Min. days to maturity shown (or `off` for
  both), so a live run sells at them once they are saved. There:
  - **Dry run** runs the live bot with the proposed settings on the
    production account but sends no orders and saves nothing — a preview of
    what they would trade. It needs some defaults saved first.
  - **Confirm and save** saves the proposal as the live defaults.
  - **Confirm and trade**, last and set apart below a red line, saves the
    proposal and then runs the live bot with it, placing real orders.
- **Trade using defaults…** opens the trade page, which shows the saved live
  defaults. There:
  - **Dry run** runs the live bot with them but sends no orders.
  - **Confirm and trade** runs the live bot with them, placing real orders.

Either run lands on its own page, which follows the run and then says what
it did (see below). A dashboard built before these buttons existed has
neither; the defaults server says so when it starts (run on its own, it opens
its own index, which links the seed values and the trade page) — rebuild the dashboard with
`python3 -m kalshi_betting.backtest` and the `--start-date` it was built from.
A page whose filter bar could not be built keeps the Trade using defaults…
link, under the notice that replaces the bar.

### Trade from the browser

```bash
./start_dashboard.sh           # then use the Backtest tab's "Trade using defaults…", or open http://127.0.0.1:8765/trade
```

The defaults server also runs the live bot. Its confirmation page (from the
dashboard's "Save as live defaults…" button, or `--seed`) and its trade page,
`http://127.0.0.1:8765/trade` (the saved defaults), each end with the same
buttons, in one fixed order:

- **Dry run** runs the same search and sizing but sends no orders and saves
  nothing (on the confirmation page it runs the proposed settings). It adds
  "simulated" rows to `trade_log.xlsx`, and needs saved defaults to exist.
- **Confirm and save** (confirmation page only) saves the proposed settings.
- **Confirm and trade**, last and set apart below a red line, places real
  orders: on the confirmation page it first saves the proposed settings, and
  starts the run only once the file has been written and read back.

```
./start_dashboard.sh ─► the live dashboard on 127.0.0.1:8766 (Live trading and Backtest tabs;
                         it opens the page) and the defaults server on 127.0.0.1:8765

Backtest tab's filter bar (each button opens a new tab):
  ├─ [Save as live defaults…] ─► /confirm?<scenario>
  │     [Dry run] ────────────────────────────────────────────┐
  │     [Confirm and save] ─► save ─► /saved                  │
  │     [Confirm and trade] ─► save, read back ───────────────┤
  └─ [Trade using defaults…] ─► /trade (the saved defaults)   │
        [Dry run] ────────────────────────────────────────────┤
        [Confirm and trade] ──────────────────────────────────┴─► start the run ─► /runs/<id>

start the run = python -m kalshi_betting.main --mode prod <all ten toggle flags>
                --result-file live_runs/<id>/result.json [--dry-run]
                (its own session, from this checkout; everything it prints → live_runs/<id>/output.log)
/runs/<id>    = reloads every 2 s while it runs, then the outcome: trades completed, a dry
                run's would-be trades, no trades to complete, not traded (and why), or an
                error — with every sale, every pair and its Kalshi category, the warnings, the
                balances, the portfolio value Kelly sizes on and the end of the log
```

A button that does not apply is shown disabled, in its place, with the reason
beside it: nothing to save, no saved defaults for a dry run, a run started from
this server still going (one started before a restart of the server included),
or — for Confirm and trade only — another live trading run in progress on this
machine (the lock described under [Production (real
money)](#production-real-money)). When the last finished real-money run (one
started here, or the scheduler's) ended needing attention (exit `20`), or ended
without a clean result (it wrote no result, was stopped by a signal or at the
scheduler's time limit, or hit an error after it began sending orders), both
pages show a red warning and Confirm and trade needs a box ticked first; read
that run's result (a scheduled run has no run page: the warning names
`kalshi_arb.log`, where its result is), check your positions in the Kalshi UI,
and if it reports a V2 order-mapping disproof, stop trading (stop the
scheduler daemon if it is running, do not run `main.py --mode prod`, and do
not press Confirm and trade), then undo by hand there what its CRITICAL names:
close out a market the account did not hold before the run, and put one it did
hold back to what it held. A run that
stopped before it could send an order (exit `2`, `10`, `30` or `50`) and a dry
run are passed over, so they neither raise the warning nor clear it; only a
newer run that ended cleanly (exit `0` or `40`) clears it. Every page's buttons
are enabled a moment after it is shown, once the mouse moves or a key is
pressed, and a second click (or a resubmitted page) does nothing twice: it
lands on the page the first one produced.

Each run is its own `main.py` process in a new session, so Ctrl-C on the
server, or closing its terminal, never stops it; the run's page keeps working
after a restart of the server (the index, `http://127.0.0.1:8765/`, lists the
newest runs). The server never stops a run: keep the Mac awake until it
finishes, and check Kalshi before stopping one by hand that the page calls
possibly hung (over an hour). A run stopped by a kill signal writes no result;
its page says so and, for real money, says to check your positions. The pages
check that a request comes from their own page; they cannot tell one local
program from another, so anything on this machine that can reach
`127.0.0.1:8765` could start a run while the server runs — stop it when you are
done. A link can also open the confirmation page with settings someone else
chose (valid ones, each change highlighted), and one press of Confirm and trade
would save and trade them, so read the highlighted changes before you press it.

### Live V2 order-mapping probe (~1 cent of real money)

```bash
python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step no-mapping|unfillable-ask|yes-close|transfer] [--dest-shard N] [--yes]
```

Human-run verification of the V2 order path's NO-leg mapping (an `ask` must open a NO
position and a reduce-only `bid` must close it), fill-or-kill kill semantics, and the
inter-shard transfer's centicent unit — against the production account, for roughly one
cent of worst-case exposure. `--dest-shard N` picks the transfer step's destination and
must not name the source shard (0) — a self-transfer is refused with a NEUTRAL before any
POST. Never wired into the pipeline; run it before trusting the V2 path unsupervised. `--step yes-close` buys 0.01 YES and sells it back with the reduce-only ask live selling sends (an immediate-or-cancel ask at the lowest level of the market's grid), and PASSes only when the account then reads exactly flat and the sale's reply reports the 0.01 sold with nothing remaining; run it, and see it PASS, before saving a sell level. The probe's closing line after a PASS, a NEUTRAL or an order step's FAIL now says so.

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

`--sandbox-balance` is the mirror image of `--dry-run`: meaningful only in dev
(where the virtual balance is both the portfolio value Kelly sizes on and the
cash the portfolio spends), and inert in prod, where sizing always uses the
real account: its portfolio value (the cash on every shard plus Kalshi's value
of the open positions), spending only its cash.
Passing it to `--mode prod` logs `"--sandbox-balance is inert in prod mode"`
rather than silently ignoring it — it is not a way to cap your exposure on a
live run, and someone using it as one would get full-size real orders. Use
`--dry-run` to avoid submitting.

### Production (real money)

```bash
python3 -m kalshi_betting.main --mode prod
```

Fetches the live account balance, scans real markets, submits fill-or-kill orders leg-by-leg, and appends results to `trade_log.xlsx`, all at the saved live defaults (`live_defaults.json`) unless a flag overrides one for this run (see [Live trading toggles](#live-trading-toggles)). **With no live defaults saved, or a saved file that cannot be used, every live run — this one, dev, and the weekly scheduled run — exits `2` before it logs, connects or requests anything**; it never falls back to `config.py`. Only one production run that sends orders (`--mode prod` without `--dry-run`) trades at a time on this machine: while another holds the live-run lock (a scheduled run, say), this one stops with exit `50` before contacting Kalshi (see [One real-money run at a time](#weekly-scheduler-daemon)). Read the 2026-09-27 decision record in `CLAUDE.md`, with its addendum, first (V0's live-contest replay of the rule `config.py` ships said STOP on drawdown, and the backtest evidence for it is anchor-fragile; the operator shipped it anyway). By the 2026-09-26 decision this also pairs and sizes same-event deadline ladders (`config.TIME_SERIES_SAME_EVENT_LADDERS`); read that constant's comment and run the dry run below first.

### Production dry-run (discover trades but don't submit)

```bash
python3 -m kalshi_betting.main --mode prod --dry-run
```

Uses the real account balance and real markets (read-only API calls only — no
orders are submitted), but still writes a simulated row per discovered trade
to `trade_log.xlsx` (status `"simulated"`), same as prod's real-order rows
just without a live fill. Confirmed live behavior — don't assume `--dry-run`
leaves `trade_log.xlsx` untouched.

### Turn on live selling

```bash
python3 -m kalshi_betting.v2_probe --ticker <TICKER> --step yes-close                 # step 1 (about 1 cent of real money; you run it)
python3 -m kalshi_betting.main --mode prod --dry-run --sell-at 80 --sell-min-days 7   # step 2 (no orders)
```

A live run sells only when a sell level is saved (or `--sell-at PCT` is given for one run); the seed saves none. Once one is saved, every run sells before it buys, the scheduled Monday run included. How a sale is judged and sent is described under [Selling a held position at a share of its potential profit](#how-it-profits). Turn it on in this order:

1. **Verify the sale order.** Run the probe's `yes-close` step on a liquid, cheap market the account holds nothing on. It buys 0.01 YES, sells it with the reduce-only ask live selling sends, and PASSes only when the account reads flat and the sale's reply reports the 0.01 sold. A held NO is sold by the same closing bid the `no-mapping` step already verified; no other step has sent the ask that sells a held YES.
2. **Dry-run the rule.** `python3 -m kalshi_betting.main --mode prod --dry-run --sell-at 80` sends no orders and saves nothing (add `--sell-min-days 7` to leave out positions near maturity). In `kalshi_arb.log` it logs one `Take-profit check (sell at 80%): …` line per held position (`-> sell`, or `-> keep` with the share of potential profit it reached at each check read, which stops at the first one below the level), a `Not selling …` line for each position it cannot judge, with the reason, `Positions to sell this run: N of M` and, for each sale, `[DRY RUN] Would sell …` with the orders. When something would sell, the estimated proceeds go onto the cash the buys are sized on (`Dry run: sizing as if the sales filled — cash $X (+$Y)`). Check a sale against the Kalshi UI: the walked prices against the book, the cash against the position's cost. A low level such as `--sell-at 5` is likelier to show a sale. `--no-sell` sells nothing this run, whatever is saved.
3. **Save the level.** In the dashboard's filter bar choose a Sell level (and a Min. days to maturity) and click "Save as live defaults…"; the confirmation page shows the change, and Confirm and save saves it. Or open `/confirm?tier_floors=…&spread_min=…&spread_max=…&k=…&size_cap=…&sell_at=0.85` (add `&sell_min_days=7`), keeping the five required fields at the values you want (and `category`, with `tag`, when a filter is saved: a link without them proposes no filter; each may be given more than once, one name per field and at most 40 of each, and a tag may be written `Category · Tag` to narrow that one category), and check that the next run's `Live settings:` line reads `sell at 85% of potential profit`.

The dry runs made on the live account before selling was turned on are recorded in `CLAUDE.md` (the live-selling paragraph under Key Patterns).

### Run result file

```bash
python3 -m kalshi_betting.main --mode prod --dry-run --result-file /tmp/run_result.json
```

A production run, with or without `--dry-run`, given `--result-file PATH`
writes what it did to `PATH` as one JSON object when it ends (`--mode dev`
refuses the flag and exits `2`, right after the order-path check). Any file
already at `PATH` is removed first, and the result is written however the run
ends once logging is set up. Two endings leave no file: a usage error that exits
`2` before logging (a bad flag, no or refused saved live defaults, a non-`v2`
order path), whose reason is on stderr, and a kill signal — so read the exit
code first. The file holds balances and trades, so the example writes it outside
the checkout; `run_result.json` and its staging file are gitignored as well.
It holds the `format` (`live-run-result-v1`), `mode`, `dry_run`, `started_at`
and `finished_at` (UTC), the `exit_code` (`null` when an exception stopped the
run, with the exception on one line in `error`), the run's `settings` in its
"Live settings:" line's words, where its `defaults` came from, the `message`
(the line the run logged when it stopped without trading, or its closing
"Submitted …" / "[DRY RUN] …" line), `balance_before` and `balance_after` in
dollars — the cash on every shard before and after trading (`null` when not
read) — `portfolio_value_before`, the portfolio value read before trading,
which Kelly sizes on and the $50 minimum reads (that cash plus Kalshi's value
of the open positions, or the cash alone when that value could not be read or
was refused; `null` when the balance was not read), `submission_started`
(`true` once a run that is not a dry run began sending orders — with no
`trades` listed, the run stopped while sending and orders may have been
placed), one entry per pair in `trades`
(its status and error, its title, and for market A and B the ticker, market,
side it buys, and the count and price it was sized at — what filled depends on
the status — then its cost with fees, profit if it wins, and `adds_to_held`:
for a trade that adds to a pair the account already holds, the contracts held
on each market, else `null`, and `category` and `tag`: the Kalshi category and
first tag market A's series is filed under, the backtest dashboard's rule, or
`null` when the run had no copy of Kalshi's /series listing), one entry per held position it sold or tried to sell in `sales` (its title, status, level, days left, cost, proceeds, the profit when all of it sold, the share of potential profit it had reached and, for each held market, its side, the contracts held, the contracts sold and the price) with `cash_after_sales` (the cash after the sales: read back from Kalshi in a live run that sold, an estimate in a dry run, `null` when no sale was tried), and every
WARNING-or-worse line it logged in `warnings`: every ERROR and CRITICAL line
whole, and the first 50 WARNING lines, each cut at 500 characters (ending in
"…"), with `warnings_dropped` counting the WARNING lines left out. The file is
replaced whole and written before the live-run lock is released. Writing it can
never stop a run or change its exit code: a failure is logged as an error
instead. The defaults server passes `--result-file` for every run it starts
(see [Trade from the browser](#trade-from-the-browser)), and its run page reads
the file.

### Live trading toggles

```bash
# Production dry run replaying the live rule as it stood before the 2026-09-27
# decision, for this run only. It gives all ten toggles, because a toggle no
# flag gives keeps its saved value (--no-add-to-held-pairs and --no-sell too, so a saved "on"
# or a saved sell level does not leak into a replay of a rule that never added to held pairs or sold); each
# flag that departs from the saved live defaults is marked "(default: X)"
python3 -m kalshi_betting.main --mode prod --dry-run --tier-floors --spread-min 0 --spread-max 1 --interval-discount 0.75 --size-cap 20 --same-title-size-cap 100 --no-add-to-held-pairs --no-sell --any-category --any-tag

# Let this run add to the pairs the account already holds (exactly the same two
# markets, the same side on each); the log names each held pair, with its cost,
# the fees in that cost and its worth at today's prices.
# This is step 2 of the first-run order under "Save the live defaults" above
python3 -m kalshi_betting.main --mode prod --dry-run --add-to-held-pairs

# Let this run sell the held positions that have reached 80% of their potential profit
# (here only while 7 or more days remain before their last market stops trading); no
# orders are sent. This is step 2 of "Turn on live selling" above
python3 -m kalshi_betting.main --mode prod --dry-run --sell-at 80 --sell-min-days 7

# Sell nothing this run, whatever the saved defaults say (also clears the saved minimum of days)
python3 -m kalshi_betting.main --mode prod --dry-run --no-sell

# Trade only one Kalshi category this run (exits 2 if the saved defaults tie a tag to
# another category: add --any-tag, or give this run's own --tag)
python3 -m kalshi_betting.main --mode prod --dry-run --category Economics

# All of Economics, and of Sports only Basketball (a tag written "Category · Tag"
# narrows that one listed category)
python3 -m kalshi_betting.main --mode prod --dry-run --category Economics --category Sports --tag "Sports · Basketball"
```

The live strategy has ten toggles: whether the time-series deadline-gap tier
floors apply, the time-series spread band on `pB − pA`, the interval discount
`k`, the per-trade Kelly cap for every pair, the extra cap on same-title pairs,
the Kalshi categories and tags a pair may trade in (`null` for any),
whether a production run may add to a pair the account already holds (see
[Adding to a pair the account already holds](#how-it-profits) above), the share
of potential profit at which it sells a held position (`sell_at`, `null` for
never; see [Selling a held position](#how-it-profits) above) and the fewest
days before maturity it may sell at (`sell_min_days`, `null` for no minimum). **A
live run reads them only from the saved live defaults, `live_defaults.json` in
the repo root** (gitignored operator state, like `scheduler_state.json`, in the
checkout the scheduler runs from). With no file saved, every live run, prod and
dev, scheduled or by hand, exits `2` before it logs, connects or requests
anything, and says how to save one: from the backtest dashboard's
"Save as live defaults…" button, or from the seed values with
`./start_dashboard.sh --seed` (tier floors off, spread band
0–0.5, `k` 0.80, a 10% per-trade cap, a 20% same-title cap, any category or
tag, adding to held pairs on, selling off). A saved file that leaves the
adding-to-held-pairs toggle out reads it as **off**: a file saved before the
toggle existed (nobody confirmed a value for it), or one saved with it off,
which the writer leaves out. To turn it on for every run, follow the order under
[Save the live defaults](#save-the-live-defaults-do-this-first-after-upgrading),
which checks a dry run first; the dashboard's save button sends
the choice only from a page that shows it. A file saved with it off holds only
the other seven toggles, so code from before this toggle still reads it; one
saved with it on is refused by that code. A saved file that leaves `sell_at` or
`sell_min_days` out reads each as **off** too (never selling, no minimum), and
the writer leaves each out while it is off, so a file with adding and selling
all off still holds only the other seven toggles; one that sells carries them
after `add_to_held_pairs`, and code from before them refuses it. A saved file that cannot be used — unreadable, malformed, or holding a
value the bot refuses — stops every live run the same way, naming the file and
the reason. Fix it, or delete it and then save new defaults: the defaults
server will not save over a file it refuses (its page answers 409 until the
file is fixed or gone). **After this change is deployed, save them in the scheduler's checkout
before the next Monday run**, in the order under
[Save the live defaults](#save-the-live-defaults-do-this-first-after-upgrading), or it exits `2` and its slot is spent (the
scheduler logs an ERROR when it starts while none are saved). A live run never
falls back to `config.py`'s toggle constants (`TIME_SERIES_TIER_FLOORS`,
`TIME_SERIES_SPREAD_BAND`, `TIME_SERIES_INTERVAL_PROB_DISCOUNT`,
`BUDGET_FRACTION`, `SAME_TITLE_SIZE_CAP`, `TRADE_CATEGORIES`, `TRADE_TAGS`,
`ADD_TO_HELD_PAIRS`, `SELL_AT`, `SELL_MIN_DAYS`): those are the backtest's `k` and caps, and the fallback for library callers
that hand no settings.

**Since the operator decision of 2026-09-27 `config.py` ships tier floors off,
spread band 0–0.5, `k` 0.80, no per-trade cap (`BUDGET_FRACTION` 1.0) and a 20%
same-title cap, with no category or tag filter** (and adding to held pairs on,
a later toggle; the seed differs in one field: a 10% per-trade cap) — before
it, tier floors on,
no band, `k` 0.75, a 20% cap for every pair and no extra same-title cap. One
pair still stakes at most 20% of the portfolio value: a time-series trade sizes under
`1 − k` = 0.20 (on books neither of which is crossed, which the scanner
requires), a same-title trade at most at its 20% cap. Read the decision
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
still binds same-title pairs), `--add-to-held-pairs` /
`--no-add-to-held-pairs` (production runs only: a dev run holds nothing),
`--sell-at PCT` / `--no-sell` (a whole percent from 1 to 100; production runs
only; `--no-sell` also clears the minimum of days), `--sell-min-days N` /
`--no-sell-min-days` (needs a sell level, saved or given) and
`--category NAME` / `--tag NAME` (repeatable; a tag may be written
`"Category · Tag"` to narrow one listed category; `--any-category` / `--any-tag`
clear a filter the saved defaults set). A flag not given keeps the saved value.
`--category` replaces every saved category for the run and `--tag` every saved
tag, so a category the saved tags narrowed trades in full unless this run's own
tags narrow it; there is no extra warning, but the `Live settings:` line marks
the field `(default: …)`. The values are validated exactly
as the saved ones are, before anything is logged or requested — a cap off the
5% grid, `k` outside `(0, 1]`, a band floor at or above its ceiling, an empty
category or tag name, the name `any` (which would match nothing yet read as
"any" on the `Live settings:` line — use `--any-category` / `--any-tag`, or
`null` in `live_defaults.json`), a tag tied to a category the run does not
list, a tied tag whose tag half is `any` or whose category or tag half is blank,
or a category written as a tied tag (`--category Sports --category "Sports ·
Basketball" --any-tag`) exits `2` with the reason. A `--category` list (or
`--any-category`) that drops the category of a tag tied in the saved defaults
exits `2` too, and the message ends with the two ways out: add `--any-tag`, or
give this run's own tags with `--tag`. A dot with no space beside it
("·Basketball", "Sports·") is an ordinary character of a plain tag, not a tie.

A pair is filed under a category and tag exactly as the backtest dashboard's
Category and Tag menus file a trade: by market A's own series, looked up in
Kalshi's /series listing (its category, and the series' FIRST tag), or, for a
series the listing lacks, the ticker-prefix label and `General`.

The rule: a pair trades when its category is listed (or no category is listed)
and, of the tags that apply to that category, either none applies or the
pair's tag is one of them. A plain tag (`--tag Basketball`) applies to every
listed category; a tag tied to a category (`--tag "Sports · Basketball"`: a
space, a middle dot, a space) applies to that category alone, and its category
must be listed. So a listed category trades in full unless a tag narrows it:

```bash
# All of Economics, and of Sports only Basketball
python3 -m kalshi_betting.main --mode prod --dry-run --category Economics --category Sports --tag "Sports · Basketball"
```

With one category listed, `--category Sports --tag Basketball` trades the same
pairs as `--category Sports --tag "Sports · Basketball"`. Names match
case-insensitively, a tied tag's two halves included. A plain tag differs from
the dashboard's Tag menu in one way: it is matched under **every** listed
category, where the menu's tags are scoped to one (`--tag Soccer` alone also
trades the Economics and Entertainment series whose first tag is Soccer — 62 of
the 210 first tags on Kalshi's 2026-09-25 listing sit under more than one
category). And a first tag Kalshi spells in two casings under one category
("Anime Awards" / "Anime awards") is one tag here but two boxes in the menu.

The dashboard's Category and Tag menus build exactly this filter. Suppose the
Category menu lists Economics and Sports: tick **Economics** and **Sports**,
then tick **Sports · Basketball** in the Tag menu (ticking a tag ticks its
category too, and a ticked category counts in full unless some of its tags are
ticked). The page now shows all of Economics and only the
Basketball part of Sports, and "Save as live defaults…" sends
`category=Economics&category=Sports&tag=Sports · Basketball` to the confirmation
page — the very flags `--category Economics --category Sports --tag "Sports ·
Basketball"` above. A confirmation link may name up to 40 categories and 40 tags;
it refuses the very same name twice, a name with two spaces in a row, a tag with
no category and a tied tag whose category is not named, and it leaves out a
second spelling of a name in another letter case. Names are checked for form
only, so a hand-made link can name a look-alike of a real category (say with a
Cyrillic letter) beside the real one: read the page's table before you confirm. The page's own "Live rule" line says what to tick to see the
filter your saved defaults hold. The menus live on the dashboard's filter bar
only: the confirmation and trade pages show every tick but have no menus, and
`backtest.py` has no `--category` (it simulates everything once and the page
shows the ticked share).

A server or run still on code from before tied tags reads a tied tag as a plain
tag that matches nothing, so such a filter trades no pair (safe, and the typo
warning says why). Restart the scheduler daemon and `./start_dashboard.sh`, and
pull in every checkout, after updating.

The filter runs after the two pair finders and before any
order book is read; with neither set it does nothing and makes no request. A
production run started with `--result-file` (every run the defaults server
starts) that finds candidate pairs reads the listing once anyway, at that same
point, to record each trade's category and tag in its result, and the filter
uses that copy. A production run refreshes the listing when its cached copy
(`backtest_cache/series_categories.json`) is over a week old; a dev run reads
the cached copy only. With no listing at all it keeps no pair (it never
guesses a category) — the run then finds no pair and exits `0`, like any run
that finds none, so the filter's WARNING is the line that says why — and a
name no series carries (for a tied tag, a category and tag no series carries
together) is logged as a likely typo.

Every run logs a `Live defaults:` line naming the saved file, when and from
what it was saved, then one `Live settings:` line naming all ten, each one a
flag moved away from the saved defaults marked `(default: X)`. A production run
that will submit orders under any such departure also logs a WARNING saying its
trades follow the flags, not the saved defaults (a dry run and a dev run do
not).
A WARNING also names any setting that empties part of the strategy — a band
ceiling below, or within float noise of, the time-series entry floor (with the
tier floors on, the larger of a deadline-gap tier and the band's floor; with
them off, the band's floor alone) — or
lets one pair stake more than 20% of the portfolio value
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

Runs from 2024-01-01, starting from what your Kalshi account is worth now, pairing same-event ladders at the configured switch (on by the 2026-09-26 decision). The starting balance is the account's portfolio value — its cash on every shard plus what Kalshi says its open positions are worth, the figure live runs size their trades on — read once from the production account before anything is fetched; the simulation holds it all as cash. When Kalshi's reply has no readable positions value the cash alone is used, with a WARNING. If the balance cannot be read (no network, bad credentials), or what it reads comes to nothing (an account worth $0, or no cash beside a positions value that could not be read), the run stops before the fetch with the reason on the terminal (exit 1); `--balance DOLLARS` starts from an amount of your choosing and makes no balance read. A starting balance below the $50 a live run needs before it trades (`config.MIN_BALANCE_CENTS`) draws a WARNING, since the backtest trades from it anyway. The log's `Starting balance` line and the dashboard's header say where the amount came from, e.g. `Starting balance: $211.42 (the account's value at 2026-10-02 21:30 UTC: cash $116.15 + open positions $95.27)`. Unlike a live run, the backtest does not check the positions value against the contracts held: it risks no money, and both parts are shown. Results are cached in `backtest_cache/`. The run ends by pointing at `backtest_dashboard.html` and at how a scenario on it becomes the live defaults, or is traded: run `./start_dashboard.sh` (it starts the live dashboard, whose Backtest tab is the page, and the defaults server), then use the filter bar's "Save as live defaults…" or "Trade using defaults…" button (see [Dashboard buttons](#dashboard-buttons)).

Options:

```bash
python3 -m kalshi_betting.backtest --start-date 2023-01-01 --balance 50000   # start from $50,000 instead of the account's value (no balance read)
python3 -m kalshi_betting.backtest --no-cache   # rebuild the assembled market list
python3 -m kalshi_betting.backtest --max-horizon-days 14
python3 -m kalshi_betting.backtest --interval-discount 0.60   # override k for this run only
python3 -m kalshi_betting.backtest --no-sweep   # skip the k-grid re-simulation (the dashboard filter bar's k select offers one point; one k column in the scenario explorer)
python3 -m kalshi_betting.backtest --same-event-ladders     # force same-event deadline ladders ON for this run (the configured default)
python3 -m kalshi_betting.backtest --no-same-event-ladders  # force them OFF for this run (the rule before the 2026-09-26 decision)
python3 -m kalshi_betting.backtest --spread-min 0.30 --spread-max 0.60   # primary scenario's time-series spread band (backtest only)
python3 -m kalshi_betting.backtest --no-band-sweep           # skip the spread-band grid (and its tier-floors-off runs); the dashboard's scenario explorer has nothing to show, and its filter bar offers the primary band only, with the Tier floors select disabled
python3 -m kalshi_betting.backtest --no-cap-sweep            # skip the per-trade size-cap sweep: the dashboard offers config.BUDGET_FRACTION only (smaller, faster page)
python3 -m kalshi_betting.backtest --no-add-on-sweep         # skip the dashboard's "Add to held pairs" family (its simulations run only while the dashboard is built, so this makes that step faster and leaves the filter bar's select disabled)
python3 -m kalshi_betting.backtest --no-sell-sweep           # skip the dashboard's Sell and Min. days to maturity selects (their simulations run only while the dashboard is built, so this makes that step much faster and leaves both selects disabled)
python3 -m kalshi_betting.backtest --sell-workers 4          # simulate the Sell select in 4 worker processes (default: one less than the CPU count, at most 8; 1 runs them in the main process)
```

**How trades fill.** Kalshi keeps no historical order books, so the backtest builds
one. It fits a table of how many contracts typically rest near the best bid from the
snapshots that `python3 -m kalshi_betting.depth_model snapshot` saves (see
[Depth snapshot](#depth-snapshot-for-the-backtest)), then, for each trade, builds a book
for each market from that table, the market's 24-hour volume and the candle's own prices,
so the top of the book is the quoted price. The live sizing code (`scanner._enrich_pair`,
then `strategy.compute_trade`) walks it: a bigger trade pays a worse average price, and
the live caps, edge cut and fill-or-kill reach all apply. A sale walks the modeled bid
ladder the same way, at the checkpoint and at each daily check before it (each built from
that check's own 24-hour volume), and a position whose ladder at any of them holds fewer
contracts than it does is not sold that Monday. With no usable snapshot, or no volume data
for a market at that moment, a trade fills at the candle's top-of-book price, in any size,
and a sale at the bid, in any size (so does a sale at a bid below 1c or above 99c, which the table has no ladders
for); the log counts the trades bought that way, and the dashboard's header line says how many of the primary scenario's trades
walked a book (in red when no snapshot was usable at all). Results recorded before this
change filled every trade at one candle price, so none is comparable. As the live sizer
requires, `--interval-discount` must be above 0 and each size cap on the 5% grid.

**Every window runs through today.** A market that settled after Kalshi's
archive cutoff (which trailed today by about two months when this was written)
has no history on the archive's candlestick endpoint, so the backtest asks
Kalshi's live endpoint (`/series/{series}/markets/{ticker}/candlesticks`) for it:
first for a market that settled after the cutoff, and as the fallback after a 404
for any other. A window may therefore start at any date, before or after the
cutoff. The live endpoint's per-request limit is assumed to be the same
5,000 candles as the archive's (not verified), and the archive cutoff is still
reported, as information, in the log and under the dashboard's "Period:" line.

**Cached runs are brought up to date.** The assembled market list
(`backtest_cache/settled_markets_*.jsonl.gz`) holds no market that settled after
it was assembled, while the report's period always runs to today. A run on the
UTC day the cache was assembled (or last extended) reuses it with no network
call. A run on a later day extends it instead: one read of the archive cutoff,
then the live endpoint's settled days from the cache's last, partial day through
today are fetched — reusing every day slice already on disk — and spliced in, in
the order a fresh assembly gives, and only the new markets' event titles are
resolved. A cache whose archive cutoff has since moved past its last day, one
that records no cutoff (assembled before that was recorded) and a legacy
`settled_markets_*.json` are re-assembled in full; a failed cutoff read or a
failed extension is a WARNING, and the cache is then served as it was. The log's
closing lines and the dashboard's header say whether the corpus was assembled,
extended, reused from earlier the same day, or served as an earlier day's cache
that could not be brought up to date. `--no-cache` still forces a full
re-assembly. An extension costs every newly completed settled day once (an
estimate: an hour or two of paging per full day of today's combo-heavy listings,
with days fetched in parallel — the 7-day window's seven days took 1 h 47 min),
plus today's partial day; each completed day is then stored and shared by every
window.

`--interval-discount K` (`0 < K <= 1`) overrides the time-series interval
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
`./start_dashboard.sh` running, then Confirm and save on
the page it opens), or, for one run, `main.py`'s matching flags (see
[Live trading toggles](#live-trading-toggles); ticking Economics, Sports and
Sports · Basketball in the filter bar's menus is `--category Economics --category
Sports --tag "Sports · Basketball"`). The scenario explorer itself is a backtest reporting feature:
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
portfolio value, its cash plus every open trade at cost, and never more than
the cash left: 1.0, no cap, since the 2026-09-27 decision, 20% before it),
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
do). Over a walked book (see [Backtest](#backtest)) the cap also bounds how deep
the live code averages the book (so a higher cap can even give a smaller trade), so
each point also records `cap_free_from`, and only caps at or above the larger of that
and the peak share a simulation. The
dashboard is that report: its filter bar's Size cap select offers
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

**`--no-add-on-sweep` — the "Add to held pairs" family (backtest only; a live
run adds to held pairs when its saved live defaults say so, or per
`main.py --add-to-held-pairs` / `--no-add-to-held-pairs` for one run).** No
eager run of the backtest adds to a pair it still holds, so the dashboard's
"Add to held pairs: on" views have no finished run to start from. The add-on sweep is **on
by default** and, like the cap sweep, simulates **nothing** during the run:
the result carries two lazy `backtester.CapSweep`s,
`BacktestSweep.add_on_cap_sweep` over the run's band x `k` x size-cap grid
(the run's own cap alone when the cap sweep is off) and
`add_on_tier_off_cap_sweep` over the tier-floors-off runs' binding bands. Every
simulation in them passes `add_to_held=True`, so a pair with an open trade may
trade again on a later passing Monday, sized as live sizes an add-on (Kelly on
the whole position, as a share of the portfolio value, the pair's open trades
at their cost plus the fees paid for them); only the "all" population is read, with no split-half or
top-event checks. Each cell is simulated when the dashboard is built, and ends
its curves on the day the run's own curve for that cell ended. The run's
figures, simulations and log lines are the same with the flag on or off,
apart from one setting line (and, with the sweep on, one summary line per
sweep). The dashboard's filter bar reads the family through its Add to held
pairs select (off, or on up to the size cap): choosing on shows the same
scenario re-simulated with adding, in the trade sections, the header's trade
count and any selection of categories and tags, and the Save as live defaults… button then
sends `add_to_held_pairs` for it; the Scenario Explorer and the Interval
Discount (k) section never follow it. `--no-add-on-sweep` skips the family,
which makes the dashboard step faster and leaves that select disabled.

**Cost of the family.** Every add-on cell is simulated when the dashboard is
built, cap by cap. Measured on the test suite's golden fixture (synthetic
data), before trades were sized on the portfolio value (the simulation counts
do not depend on that; the times and page sizes may): over its full
36 x 13 x 20 grid, reading every tier-on add-on cell took
about 8.1 s (468 band x `k` cells, 7,488 simulations) and every tier-off one
about 4.7 s (234 cells, 3,744), against 28.4 s for the size-cap sweep (468
cells, 13,333 simulations); a real `run_backtest_sweep(add_on_sweep=True)`
page over that fixture at 6 bands x 3 `k` x 20 caps grew from 665,339 to
950,892 bytes (+43%), with `generate_dashboard` taking 8.95 s against 5.75 s
(+56%). **A real 365-day page has not been measured with the family:** expect
it to grow by a similar share and its dashboard step to take longer.

#### Selling at a share of potential profit

**`--no-sell-sweep` — the dashboard's Sell and Min. days to maturity selects
(the simulation is backtest only; a live run sells at its own saved level by the same rule, see [Turn on live selling](#turn-on-live-selling)).** The filter bar's Sell
select offers "no selling" —
every position held until its markets pay out, as the run simulates it — and a
level from 80% to 100% in steps of 1% (`config.TAKE_PROFIT_LEVELS`; with the
Sell select on, a backtest refuses, before it fetches any market data, levels
that are not distinct shares above 0 and at most 1). At a level,
a position (a pair, with every trade that added to it) is sold whole at the
first weekly checkpoint — the live run's time — where its **realized profit**
has stayed at or above that share of its **potential profit** for 3 days in a
row (`config.TAKE_PROFIT_HOLD_DAYS`):

- realized profit = what selling it then would return − what it cost: each
  market's contracts sold down a modeled bid ladder that starts at the bid of the
  side held (one minus the other side's ask, on an hourly candle that ended within
  the hour before the checkpoint), at the average price reached, less Kalshi's
  taker fee on that sale (with no depth snapshot, no volume data for the market
  at that check, or a bid below 1c or above 99c, the sale is that one bid, in
  any size) — a leg whose market has already paid out counts at its payout, with no
  fee — less the contracts' cost and the entry fees;
- potential profit = what it pays in a win (its contract pairs at $1 each) −
  what it cost.

**Days in a row.** The position is checked once a day, 24 hours apart, the
last check at the checkpoint where it is sold. That check reads the bids there,
as above; each earlier check reads the last quote of the 24 hours before it,
and walks a ladder built from the trading volume of those 24 hours (a leg whose
market has paid out by then counts at its payout). A level reached at the
checkpoint alone, a day below it, or a day with no quote holds the sale back to
a later checkpoint, so a price that jumps for a moment does not trigger a sale.
`TAKE_PROFIT_HOLD_DAYS` is a whole number from 1 (the checkpoint alone) to 7 (live selling reads the same setting, counting the days back from the run's own moment, and refuses any other value too);
with the Sell select on, a backtest refuses any other value before it fetches
any market data.

**A minimum of days before maturity.** The backtester can also hold back a
sale that comes too close to the end: `_simulate_at_discount(sell_at=…,
sell_min_days=N)` sells a position that has reached its level only while at
least N days remain before it **matures** — the day its last market stops
trading, i.e. the latest close date among its markets, counted in calendar
days from the checkpoint's date (the close dates are UTC on every Kalshi
timestamp, and the checkpoint's date is its UTC date). Nearer than that, the
position is not sold; since its days left only shrink, it is then held until
it pays out (unless, with Add to held pairs, an add-on brings in a market
that closes later). N is a whole number of at least 1 and needs a sell level;
without it (the default) there is no minimum. A minimum of 1 day is almost
the same as none: the two differ only when the position's last market closes
on the checkpoint's own UTC date or earlier. Every selling run records, for
each position it sells, the days it had left and its profit at each check
(`SweepPoint.sales`). The sell family carries every option of
`config.TAKE_PROFIT_MIN_DAYS` (1 to 7, 14 and 21 days), which a backtest
checks before it fetches any market data, and the dashboard offers them as
its **Min. days to maturity** select, right after Sell: "1 day" by default,
and shut while Sell says "no selling", since there is then no sale to hold
back. Its hover text gives the rule in full, and the summary line names the
minimum after the level ("… at 85% of its potential profit and at least 3
days before its last market closes").
One look-ahead to know about: the backtest's close date is the date a
settled market actually closed, so for an event decided early the backtest
sees maturity coming sooner than a live trader, reading the scheduled close,
would have.

**Sales come before purchases.** At each checkpoint the backtest first pays
out what settled, then makes its sales, then values the portfolio, and only
then buys — new pairs and add-ons to held pairs alike. So a sale's money is
there for that checkpoint's purchases, and every purchase there is sized on
the portfolio value after the sales.

A leg with no fresh bid at a checkpoint, or whose ladder at any of the checks
holds fewer contracts than the position, cannot be sold there, so its position
is held until the next one. A sold trade ends on the sale day at the sale's
value, and its cash is available from then. A position sold at a checkpoint is
not bought again or added to there; a time-series pair can be bought again on a
later Monday it qualifies on. The checkpoint's check and the sale use the same
value; the equity curve still values an open position at the ask, as before.
Selling and Add to held pairs combine: with both on, a position is sold whole,
its add-ons included, and a position not sold may still be added to.

Like the add-on family, the sell family simulates nothing during the run.
`BacktestSweep.sell_sweep` (a `backtester.SellSweep`) holds the same entries,
and the dashboard reads every level and minimum of days of every scenario the
bar shows (band, Tier floors choice, `k`, size cap and Add to held pairs
choice) after its walk, in worker processes: `--sell-workers N`, by default one
less than the CPU count and at most 8 (`config.DASHBOARD_SELL_MAX_WORKERS`); 1
runs them in the main process. It simulates only the settings that differ
(`SellSweep.sold_grid`): a setting at which no position of a scenario's run
would have sold — its level never reached, or reached only nearer to maturity
than the minimum — is that run itself and shows the run's own data, and a
setting at least as strict as one already simulated, when every sale of that
run also meets it, is that run. Each new trade list is written as a file in
`backtest_dashboard_files/<build>/` beside the page, which the page loads when
that setting is chosen, so **keep the folder beside `backtest_dashboard.html`**:
copy or move the two together. Which file each setting shows is kept in the
page, one block per band and Tier floors setting (a band the tier floors never
bind at uses its tier-on block for both), which the page unpacks the first time
a level is chosen there. Each build deletes the folders of earlier builds once it
has replaced the page (never one another backtest is still writing). The
Scenario Explorer and the Interval Discount (k) section never follow either
select, the `k̂` figures do not depend on them, and Save as live defaults… sends
the level and minimum shown as the live sell level and minimum of days (or `off`
for both under no selling). `--no-sell-sweep` skips the family and leaves
both selects disabled.

**Cost of the selects.** With 21 levels and 9 minimums the selects cover 189
settings per scenario, so the Sell part of a dashboard build — its time and
its folder of files — grows with the levels and minimums offered. Three
savings keep it below 189 simulations per scenario: settings no
position reaches, settings that repeat a run already simulated, and size caps
at or above a cell's sharing floor (its peak Kelly fraction, or over a walked
book its `cap_free_from` if larger) sharing one simulation. The build's closing
log line gives its real time, the simulations it ran, the ones it shared across
caps, the settings that repeated an earlier run, those that sold nothing, and
the files' total size. `--no-sell-sweep` skips all of it.

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
produce zero trades.

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
duplicates, and the event tickers and titles being resolved. Two record lists
also remain: a sequential fallback's whole result if the sharded fetch ever
falls back to one, and a legacy `settled_markets_*.json` cache, which is read
whole when it is served. An interrupted fetch resumes at day granularity, and `--no-cache`
reuses the day slices (they cannot go stale — see CLAUDE.md), so a refresh only
fetches the current day plus any days not yet on disk. If a day slice disappears or is damaged while it is being streamed, the
run stops with an error naming the day rather than continuing with a short
corpus; re-running refetches that day.

**Long-lived markets: every created-day is read.** Some markets are created
long before `--start-date` and settle inside the window ("Will GTA 6 be released
by Dec 31, 2025?", created 2023-11). The archive is ordered by creation time and
cannot be filtered by time, so the fetch reads one archive day slice per
created-day from `ARCHIVE_FIRST_CREATED_DATE` (2021-06-01, a month before the
archive's first market) up to the cutoff, and keeps the markets that settled in
the window. The days before a 2025-10-01 start are small — about 3,000 pages,
58 MB of slices, fetched in under 3 minutes on 2026-10-01 — and are reused
from disk like every archive slice until the cutoff moves. Each run also asks the archive once
whether it holds a market created before that first day, and logs a WARNING if
it does. A one-page-at-a-time walk below `--start-date` did this before
2026-10-01; it stopped after 50 pages in a row with nothing in the window and
missed long-dated markets, so assembled `settled_markets_*` caches built before
then can lack them: run the backtest once with `--no-cache` to rebuild them (the
rebuild is written in the streamed `.jsonl.gz` format and, once it is written,
deletes the old `.json` of the same name). The sequential fallback, used only
when cursor synthesis fails, still stops after `ARCHIVE_MAX_BARREN_PAGES` (50)
empty pages.

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
2026-08-29 is empty and more than a day old, which the cache lookup of that
day treated as a miss; and the other four start after the cutoff, so they were
0-trade by construction (until 2026-10-02 the backtest had no candles for a
market settled after the cutoff). Take any before/after baseline you want from them with
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
+136.3%, and the pooled empirical k̂ (then measured on the YES-ask gap) from
0.891 to 0.958 (see the entry-checkpoint gotcha in `CLAUDE.md`). No backtest
result or dashboard from before this change is
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

A ticker that settled after the archive cutoff is asked of Kalshi's live
candlestick endpoint first (see "Every window runs through today" above); the
log's `N of M tickers returned no candles` line counts the tickers neither
endpoint had candles for.

### Depth snapshot (for the backtest)

```bash
python3 -m kalshi_betting.depth_model snapshot [--markets N]
```

Saves live order books for the backtest's depth model, a table of how many contracts typically rest near the best bid. It sends read-only requests and places no orders. It samples about 2,000 open markets (`--markets N` for another number) and saves them under `backtest_cache/depth_snapshots/`, then logs how many ladders each cell of the table holds. The model is fitted to every saved snapshot, so more snapshots, especially ones taken near Monday 09:00 PT, make the table better. It reads with your production credentials. A backtest fits every saved snapshot before its fetch; with none usable it fills every trade at the top of the book (see [Backtest](#backtest)).

### Weekly scheduler daemon

```bash
python3 -m kalshi_betting.scheduler
```

Runs the production bot once a week in a blocking loop, at `config.SCHEDULED_RUN`'s weekday and time — Monday 09:00 — on the **host's clock**, each run spawned as a `python3 -m kalshi_betting.main --mode prod` subprocess and killed after `SCHEDULER_JOB_TIMEOUT_SECONDS` (3600s). The log also prints the equivalent `crontab` entry if you prefer cron (cron fires on the host's clock too).

**Run time and zone.** `config.SCHEDULED_RUN` also names the run's zone, America/Los_Angeles, where 09:00 is 16:00 UTC under daylight time and 17:00 UTC under standard time. The daemon still fires on the host's clock rather than converting: the `schedule` library's own zone support goes through `pytz`, whose America/Los_Angeles table has no daylight time after 2037, so from 2038 it would fire at 10:00 all summer. Instead, at startup the daemon checks that the host's clock places the next 104 weekly fires at the schedule's own UTC instants. **Keep the host's time zone set to America/Los_Angeles.** On a host whose clock keeps different UTC offsets over those two years (a UTC host, or a zone without the same daylight-saving rules), the daemon logs CRITICAL, naming the first mismatched date, and keeps firing at 09:00 on the host's clock; a zone with the same rules under another name (`US/Pacific`, `PST8PDT`) passes. The backtest enters every trade at the same instants, so on such a host it no longer replays the daemon's runs, and the CRITICAL says so. To move the run, edit `config.SCHEDULED_RUN` — its weekday, hour, minute and zone form one value, and the edit also moves every backtest entry and re-keys the backtest's assembled cache; the backtest refuses a schedule whose run time falls on another date in UTC or on a clock change. A time the zone skips at a clock change (02:30 on a spring-forward Sunday, say) cannot fire at one UTC instant on any host, and the startup check logs that CRITICAL against the schedule, not the host.

Every scheduled run trades exactly the saved live defaults (`live_defaults.json` in the checkout the daemon runs from; see [Live trading toggles](#live-trading-toggles)). With none saved, or a saved file that cannot be used, each run exits `2` (`Job failed (exit 2)`) and its slot is spent, so the daemon logs an ERROR as soon as it starts while that is so, naming the fix: with none saved, the checkout they must be saved in and how to save them (run the save from that directory: `python3 -m` saves into the checkout whose root it is run from, and from any other directory into the checkout pip installed, so a save from another checkout, a worktree say, does not count; the defaults server's first log line names the file it saves to); with an unusable file, that it must be fixed, or deleted before new defaults are saved.

The daemon logs to its own `kalshi_scheduler.log` (and the console), deliberately separate from `kalshi_arb.log`, which the spawned run writes and rotates — a second process holding an open handle on a rotated file would keep writing into the renamed backup.

**Slot record and startup catch-up.** Each run claims its Monday-09:00 slot in `scheduler_state.json` (repo root) *before* spawning the subprocess and finalizes the record — `finished_at`, `exit_code` — on every exit path, including timeout and spawn failure. On startup, the daemon compares the most recent Monday-09:00 slot against that record: if the slot has **no** recorded attempt (daemon not running when it came around — never started, crashed, host rebooted, mid-deploy), it runs a catch-up job immediately rather than waiting up to a week for the next Monday. A slot whose recorded attempt merely *failed* is not retried; only a slot with no attempt at all triggers catch-up. A hand-edited or partially-written state file with a non-integer `retries` or `exit_code` degrades to an unknown, typed reading (`0` / `None`) with a WARNING. For `retries` that is a real crash fix: a string or `null` there used to raise `TypeError` out of the daemon at startup, before the weekly job was ever registered, silently stopping the bot from trading at all. For `exit_code` it is typing hygiene only — a non-integer value already compared unequal to `30` and so already left the slot unretried; it is now named in a WARNING instead of failing silently.

**Blind-run retry.** There is one exception to "a failed attempt satisfies the slot": a run that exits `30` (`EXIT_NO_TRADEABLE_SHARDS`) scanned **nothing at all** — every advertised exchange shard trading-inactive (a Kalshi maintenance window overlapping the 09:00 fire), or an ingest that came back empty for a cause `/exchange/status` could not name. The scheduler sees only the exit code, so its own WARNING/ERROR name both possibilities; the run's own `kalshi_arb.log` carries the line saying which one fired. Since the bot only trades on that weekly fire, such a slot used to cost the entire week while both logs reported success. `run_job` now registers a one-shot retry `SCHEDULER_BLIND_RETRY_SECONDS` (3600s) later, at most `SCHEDULER_BLIND_MAX_RETRIES` (4) times per slot — hourly × 4 covers a typical outage while keeping the scan near its intended Monday morning — and the startup catch-up check re-runs a slot whose recorded attempt exited `30`. The attempt count is persisted as an optional `retries` key in `scheduler_state.json`; a state file written before this feature has no such key and counts as 0. Once the cap is reached the daemon logs an ERROR and gives up on that slot rather than retrying through a multi-day outage.

**One real-money run at a time.** A production run that sends orders — the weekly scheduled run, one started by hand, or one started from the defaults server's pages, from any checkout or worktree — takes a machine-wide lock, `~/.kalshi_betting/live_run.lock` in your home folder, before it contacts Kalshi, and keeps it until it ends; every checkout and worktree shares it, since they all trade the same account. Two such runs at once would each size on the whole balance, could pick the same pair, and would confuse each other's reading of whether a leg filled. So a run that finds the lock taken waits up to 2 s and then stops without contacting Kalshi, with exit `50`. The operating system drops the lock however its holder ends (it finishes, crashes, is killed or the Mac restarts), so only a run that is still alive but hung can hold it for long. When the Monday run stops this way, the scheduler logs `Job did not trade (exit 50): another live trading run on this machine was in progress (process …, since …)` and counts the week's slot as done, with no retry an hour later, because a real-money run was already trading. If that run started over an hour ago (`SCHEDULER_JOB_TIMEOUT_SECONDS`), or its start time is not recorded, the scheduler logs an ERROR instead: the run may be hung or stopped, no trade was made this week, and every scheduled run will stop the same way until it ends. A dry run (`--dry-run`, or `--mode dev`) sends no orders, so it neither takes the lock nor stops for it and never blocks the Monday run. While another run holds the lock, the defaults server shows its Confirm and trade button disabled, naming that run (see [Trade from the browser](#trade-from-the-browser)). `v2_probe` and runs on another computer are not covered.

**Registration guard.** The `schedule` library reschedules a job only *after* its function **returns**, so an exception escaping a job leaves its next fire time in the past. The daemon's poll loop catches the exception and survives, but the job is still overdue — so it is re-entered on the very next 60-second poll tick, which would turn the weekly production run into a once-a-minute one for as long as the fault lasts. Every job is therefore registered through `scheduler._guarded_job`, which catches at that boundary, logs the traceback once, and lets the library reschedule normally. On a healthy host nothing changes; on a fault the daemon abandons that slot and waits for the next scheduled fire. The blind retry additionally passes `on_error=schedule.CancelJob`, so a retry that *raises* still deregisters itself instead of becoming a recurring job that never advances the retry cap — which also means a raising retry **ends the retry chain for that slot**: the slot is then recoverable only by the startup catch-up check on a daemon restart, never while the daemon keeps polling. That is the deliberate trade: the alternative is a real production trading run every 60 seconds for as long as the fault lasts.

> ⚠️ **The first daemon start after this upgrade immediately runs a live production trade.** `scheduler_state.json` does not exist yet, so the startup check sees no record for the most recent Monday slot and fires a real `--mode prod` run right away — not at the next Monday 09:00. The same applies to **any** restart after a Monday 09:00 slot passed while the daemon was down. Save the live defaults first, in this checkout (see [Save the live defaults](#save-the-live-defaults-do-this-first-after-upgrading)): without them the catch-up run exits `2` and spends that Monday's slot, and so does every scheduled run after it. Start the daemon only when you are prepared for it to trade immediately; if you are not, run `python3 -m kalshi_betting.main --mode prod --dry-run` first to confirm what it would do. That pre-flight exits `0` on a normal exchange, `30` if it could not scan anything (halt, or an empty ingest) — a `30` from the pre-flight means the check itself saw nothing, not that the bot found no edge — and `2`, before scanning anything, while no usable live defaults are saved. The table below lists every exit code.

**Process exit codes.** `main.py`'s exit code is the only signal the scheduler has for what happened inside a run:

| Code | Meaning |
|------|---------|
| `0`  | `EXIT_OK` — run completed (including a clean run that found no qualifying pairs) |
| `10` | `EXIT_SKIPPED_LOW_BALANCE` — run skipped because the portfolio value (cash on every shard plus Kalshi's value of the open positions, or the cash alone when that value is unread or refused) was below the $50 minimum; cash alone below it only logs a WARNING, and the run goes on |
| `20` | `EXIT_TRADES_NEED_ATTENTION` — at least one trade came back `rollback_failed` or `manual_review`, or a sale was left unbalanced or of unknown outcome (`unbalanced`, `manual_review`); **a human must check the account and trade log** |
| `30` | `EXIT_NO_TRADEABLE_SHARDS` — **nothing was scanned**: either every advertised exchange shard reported `trading_active=false` (an exchange-wide halt or maintenance window), so ingest dropped every market, or ingest produced **zero markets** for a reason `/exchange/status` could not name (`fetch_shard_statuses` is fail-soft and returns `None` on any internal failure, so the all-halted test is unevaluable exactly when something went wrong). Deliberately distinct from `0`'s "scanned everything, found no edge" |
| `40` | `EXIT_TIME_SERIES_SKIPPED` — production only: a held market could not be identified, so the run could not tell which ladders it already holds and made **no time-series trade**; it still searched for and traded same-title pairs. `20` wins over it when a trade also needs a human. The scheduler logs it as an ERROR (pointing at the ERROR in `kalshi_arb.log` that names the market) and counts the weekly slot as done — a retry an hour later would most likely fail the same lookup |
| `50` | `EXIT_RUN_IN_PROGRESS` — production without `--dry-run` only: another live trading run on this machine held the live-run lock (`~/.kalshi_betting/live_run.lock`), so this run stopped after a 2 s wait, before building a client or contacting Kalshi, and sent nothing. Its WARNING names the run in the way (process, checkout and start time). The scheduler counts the weekly slot as done and does not retry it, since a real-money run was already trading: a WARNING naming that run, or an ERROR when it has held the lock for over `SCHEDULER_JOB_TIMEOUT_SECONDS` (an hour) or its start is not recorded — it may be hung, and every scheduled run stops the same way until it ends |
| `2`  | Usage error — argparse's code for an invalid flag, a `config.ORDER_API_VERSION` other than `"v2"`, or no live defaults saved, or a saved `live_defaults.json` that cannot be used, refused before logging is configured or any request is made. A scheduled run passes no toggle flag, so a `2` there means `config.ORDER_API_VERSION` is not `"v2"`, or no live defaults are saved or the saved file is refused: `kalshi_arb.log` records nothing, and the reason is on stderr, which the scheduler logs under `Job failed (exit 2)` (the scheduler also logs an ERROR at start while none are usable). Like `1`, it satisfies the weekly slot |
| `1`  | Unhandled exception — the interpreter's default for a crash; not part of the contract above |

The constants live in `config.py` (`EXIT_OK` / `EXIT_SKIPPED_LOW_BALANCE` / `EXIT_TRADES_NEED_ATTENTION` / `EXIT_NO_TRADEABLE_SHARDS` / `EXIT_TIME_SERIES_SKIPPED` / `EXIT_RUN_IN_PROGRESS`) and the scheduler maps each to a distinct log level and message, so a low-balance skip or a manual-review run is never logged as "completed successfully". Exit `30` additionally means the weekly slot was **not** satisfied: nothing was scanned, so the run must never count as the week's scan — see the blind-run retry above.

---

## Testing

```bash
python3 -m pytest tests/ -v      # run the test suite
python3 -m ruff check kalshi_betting/   # lint check
```

Tests run fully offline against `unittest.mock.MagicMock` clients — no real Kalshi API calls. The defaults server's socket tests (`tests/test_defaults_server.py::TestOverASocket` and `::TestBusyPortOverASocket`, which points the server's port at a listener of its own and never asks the real port anything) bind a loopback port and skip where a sandbox refuses that; CI binds. One of them starts a tiny Python child through the real `subprocess.Popen` in place of `main.py` (a dry run's round trip); every other test that starts a run hands the server a stand-in that starts nothing, and no test starts `main.py`. Its page-script tests (`::TestConfirmScript`) run the page's script under node or macOS's `jsc`, and skip when neither is present. `tests/test_launcher.py` runs `start_dashboard.sh` with a stand-in for Python that records each call and starts neither server (each server it plays either exits, or keeps running and, like the real ones, stops cleanly on Ctrl-C), checking the flags it refuses before starting anything, which arguments each server gets, the warnings when the live dashboard stops early or late, the exit code, that Ctrl-C, SIGTERM or SIGHUP to the script or its process group, or a failing defaults server, stops both servers, and that a server that has already ended is never signalled (and runs the script's own import check once under the real Python, which imports the live bot and the live dashboard and starts nothing); it skips where there is no bash. The Live trading tab's tests (`tests/test_live_portfolio.py`, `tests/test_live_dashboard.py`) read stand-ins for Kalshi's replies, never the network; the live dashboard's socket tests skip where a sandbox refuses binding a port, and its page-script tests run under node or `jsc` and skip when neither is present. `.github/workflows/ci.yml` runs both commands on every push to `main` and on EVERY pull request, whatever its base branch — the `pull_request:` trigger carries no branch filter. Both must pass before merging.

---

## Sandbox Note

The Kalshi sandbox endpoint (`https://demo-api.kalshi.co`) requires a **completely separate account** registered at [demo.kalshi.co](https://demo.kalshi.co). Your production API key will return `401 Unauthorized` on the sandbox endpoint — this is intentional by Kalshi.

To use dev mode with real sandbox authentication, register a sandbox account, generate its API key, and add it as `"dev_api_key"` in `secrets.json`. Without a sandbox key, dev mode still fetches real sandbox market data (for scanning) but skips the held-positions and balance checks that require authentication.
