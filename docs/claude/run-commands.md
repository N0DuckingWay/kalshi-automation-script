# Run Commands — every command, with its notes

Moved word for word from the "Run Commands" section of the project's `CLAUDE.md` on 2026-10-26, when that file was cut down to a short root that indexes these notes. Where the text below says "this file", it means the project's notes as a whole: `CLAUDE.md` and the files in `docs/claude/`. Section names the code and README use ("Key Patterns", "Known Gotchas", "Module Map") name the folders of `docs/claude/`. Each paragraph is one line, some of them long: find one with `grep -n`, then read that line by number.

All commands must be run from the **repo root**:

```bash
# One-time setup (Python >= 3.11)
pip install -e ".[dev]"

# Sandbox simulation with virtual $1k balance
python3 -m kalshi_betting.main --mode dev --sandbox-balance 1000

# Sandbox simulation, dry-run (no orders submitted)
python3 -m kalshi_betting.main --mode dev --dry-run

# Production live trading, from the saved live defaults (live_defaults.json in the
# repo root: tier floors, spread band, k, per-trade and same-title caps, category/tag
# filter, rebalancing and its minimum price movement (with it on, a production run trims
# held pairs before it buys and may add to them), and the sell level and its minimum of
# days — each overridable for one run by the flags below, which the run marks
# "(default: X)" on its "Live settings:" line and, when it submits orders under one,
# names in a WARNING). With no saved file, a refused one, or one an older version saved
# (holding add_to_held_pairs, trim_to_kelly or trim_min_rise, which rebalance and
# min_price_move replaced, or categories, the old list of categories to trade, which
# exclude_categories replaced — every file saved before that change holds it: save it
# again through the defaults server, do not delete it),
# every live run (prod and
# dev, scheduled or by hand) exits 2 before anything is logged or requested; it never
# falls back to config.py's toggles — read the saved-live-defaults paragraph under Key
# Patterns first, and the 2026-09-27 decision record with its addendum (V0's
# live-contest replay of the rule config.py ships said STOP on drawdown; the operator
# shipped it anyway). Pairs and sizes same-event ladders since the 2026-09-26
# decision — read config.TIME_SERIES_SAME_EVENT_LADDERS too
# Only one production run that sends orders (no --dry-run) trades at a time on this
# machine: while another holds ~/.kalshi_betting/live_run.lock (run_lock.py), this one
# exits 50 before contacting Kalshi
python3 -m kalshi_betting.main --mode prod

# Production discovery only, no orders
python3 -m kalshi_betting.main --mode prod --dry-run

# Either production run can also write what it did to a JSON file when it ends
# (--mode prod only; exit 2 in dev): the exit code, the line it stopped or finished
# on, the balances, whether it began sending orders, each pair's outcome, its
# WARNING-or-worse lines and any error. Any file already at the path is removed
# first; the result is written however the run ends once logging is set up (for a
# run that sends orders, before the live-run lock is released). A usage error
# (exit 2, before logging) or a kill signal leaves no file, so read the exit code
# first. The file holds balances and trades, so this example writes it outside the
# checkout (run_result.json and its staging file are gitignored as well)
python3 -m kalshi_betting.main --mode prod --dry-run --result-file /tmp/run_result.json

# Production dry run replaying the live rule as it stood before the 2026-09-27
# decision, for this run only. It gives all eleven toggles, because a toggle no flag
# gives keeps its saved value (--no-rebalance and --no-sell too, so a saved "yes" or a
# saved sell level does not leak into a replay of a rule that never rebalanced or sold;
# --no-rebalance clears the saved minimum price movement as well); each flag that
# departs from the saved live defaults is marked "(default: X)" on the "Live
# settings:" line
python3 -m kalshi_betting.main --mode prod --dry-run --tier-floors --spread-min 0 --spread-max 1 --interval-discount 0.75 --size-cap 20 --same-title-size-cap 100 --no-rebalance --no-sell --no-exclude-category --no-exclude-tag

# Production dry run that rebalances (no orders, nothing saved): before buying, sell part of each
# exact held pair that has outgrown its Kelly share of the portfolio value, by the backtest's rule
# (config.kelly_trim_count), once 7 calendar days (config.REBALANCE_MIN_DAYS_SINCE_TRADE) have
# passed since the last fill on either of its markets, a fill by hand included, and only if
# Kalshi's list of the account's fills reaches back that far; and add to the pairs the account
# already holds (exactly the same two markets, the same side on each, or a lone held market whose
# partner has paid out), after the same wait since the last fill on their markets. It trims
# nothing while an order rests on the account (Kalshi holds its cash outside the balance and the
# positions' value a trim is sized on). The log has the
# portfolio value it sizes on (its "Sizing on portfolio value $X = cash $Y + open positions $Z" line
# must show the open positions, with no WARNING refusing them, or the run adds to no held pair and
# trims nothing), each held pair it may add to with its cost, the fees in it and its worth at
# today's prices (`Held pair to add to: ...`), one `Add-on check: ... -> may add` line per held pair
# that has waited long enough (or a `Not adding to ...: <reason>` line for one held back),
# `Held pairs that may be added to this run: N of M`, each add-on marked "adds to N held",
# `Resting orders on the account: N` (read before the fills),
# `Fills read: N on M market(s), <oldest day> to <newest day> (UTC)` (the first time a trading run
# reads that list: check how far back it goes),
# `Kelly trim sizes on $X = cash $Y + open positions $Z (Kalshi's value $A; $B at this run's bids)`
# (a trim is sized on the larger of the two), one `Kelly trim check: ... -> sell N` (or `-> keep`)
# line per pair, a `Not trimming ...: <reason>` line for a pair it leaves alone,
# `Pairs to trim this run: N of M`, a `[DRY RUN] Would sell ...` line per trim (`(of N held)`
# follows each order's count) and, when something would sell,
# `Dry run: sizing as if the sales filled — cash $X (+$Y)`: the buys are then sized on the cash
# those trims would free, and the pairs trimmed stay held and are not added to. A saved file
# without the setting reads it as off. Run `v2_probe --step yes-close` (below) and see it PASS
# before saving rebalance on: with it saved, every live run, the Monday one included, trims before
# it buys. First-run order for rebalancing: the paragraph after the live-trimming one under Key
# Patterns
python3 -m kalshi_betting.main --mode prod --dry-run --rebalance
# The same with a minimum price movement of 10c: a pair is trimmed or added to only once its total
# price (the asks of the sides held, added; a lone leg's one market's ask) has moved that far, up or
# down, since its last fill, and its check lines then read `price $0.74 then, $0.87 now (+13c,
# minimum 10c)` (a fall shows as -12c, say). --no-rebalance neither trims nor adds to a held pair
# this run whatever is saved (and clears the saved minimum price movement too)
python3 -m kalshi_betting.main --mode prod --dry-run --rebalance --min-price-move 10
python3 -m kalshi_betting.main --mode prod --dry-run --no-rebalance

# Production dry run that sells (no orders, nothing saved): judge every held position by the
# take-profit rule at 80% of its potential profit, and only while at least 7 days remain
# before its last market stops trading (--sell-min-days is optional). The log has one
# `Take-profit check (sell at 80%): ...` line per position (`-> sell`, or `-> keep` with the
# share it reached at each check read, up to the first one below the level), a `[DRY RUN] Would sell ...` line per sale,
# `Positions to sell this run: N of M` and, when something would sell, `Dry run: sizing as
# if the sales filled — cash $X (+$Y)`: the buys are then sized on the cash those sales
# would free, and the markets sold stay out of them. Run `v2_probe --step yes-close` (below)
# and see it PASS before saving a sell level: with one saved, every live run, the Monday one
# included, sells before it buys. --no-sell sells nothing this run whatever is saved (and
# clears the saved minimum of days too). First-run order for selling: the live-selling
# paragraph under Key Patterns
python3 -m kalshi_betting.main --mode prod --dry-run --sell-at 80 --sell-min-days 7
python3 -m kalshi_betting.main --mode prod --dry-run --no-sell

# Exclude one Kalshi category this run: every other category trades, one never seen
# included (named as the backtest dashboard's Exclude categories menu names its categories;
# repeatable, and --exclude-tag NAME excludes a series' first tag, a plain name under every
# category; --no-exclude-category / --no-exclude-tag clear an exclusion the saved live
# defaults set). An --exclude-category list replaces every saved excluded category and an
# --exclude-tag list every saved excluded tag. While any category or tag is excluded, a
# pair whose series Kalshi's /series listing lacks is not traded (its category and tag are
# not known), with one WARNING. --category, --any-category, --tag and --any-tag, the old
# include-list flags, exit 2 naming their replacement
python3 -m kalshi_betting.main --mode prod --dry-run --exclude-category Politics

# Every category but Politics, and of Sports every tag but Hockey: a tag written
# "Category · Tag" is excluded in that one category alone (a plain --exclude-tag Hockey
# would exclude Hockey under every category). The dashboard's menus show this filter as
# Politics ticked in the Exclude categories menu and Sports · Hockey ticked in the Exclude
# tags menu (the page's Live rule line names these boxes)
python3 -m kalshi_betting.main --mode prod --dry-run --exclude-category Politics --exclude-tag "Sports · Hockey"

# Live trading (either mode), only consider markets closing within 14 days from now
python3 -m kalshi_betting.main --mode dev --max-horizon-days 14

# Backtest from 2024-01-01, starting from what the account is worth now: its cash on
# every shard plus Kalshi's value of its open positions (the cash alone, with a
# WARNING, when that value is unreadable), read once from the production account
# before the fetch. A read that fails or comes to nothing stops the run (exit 1).
# It also fits the depth model from the saved order-book snapshots (the snapshot command
# below): each trade is then sized over a modeled order book, and the config echo ends
# `fills=walked book (...)`; with none usable every trade fills at the top of the book,
# in any size (`fills=top of book (no usable depth snapshot)`, and a red line on the page)
python3 -m kalshi_betting.backtest --start-date 2024-01-01

# The same from a $10k starting balance of your choosing (no balance read)
python3 -m kalshi_betting.backtest --start-date 2024-01-01 --balance 10000

# Backtest forcing fresh API fetch (ignores disk cache)
python3 -m kalshi_betting.backtest --no-cache

# Backtest, only enter trades whose later-closing leg is within 14 days of the simulated entry date
python3 -m kalshi_betting.backtest --max-horizon-days 14

# Backtest at a specific time-series interval discount k, above 0 and at most 1 (this backtest run only; live
# sizing reads the saved live defaults' k, or main.py's own --interval-discount for one live run)
python3 -m kalshi_betting.backtest --interval-discount 0.60

# Backtest without the k-grid re-simulation (dashboard's k selector offers one point; the empirical-k calibration report still runs;
# with the band sweep on — the default — each band is simulated at the primary k only, so the scenario explorer's heatmap has a single k column)
python3 -m kalshi_betting.backtest --no-sweep

# Backtest with same-event deadline ladders forced on/off for THIS run only
# (default: config.TIME_SERIES_SAME_EVENT_LADDERS — ON by the 2026-09-26 decision, so
#  --no-same-event-ladders is the one that departs from a default run; never
#  reaches the live finder, which binds that constant at import)
python3 -m kalshi_betting.backtest --same-event-ladders
python3 -m kalshi_betting.backtest --no-same-event-ladders

# Backtest with each settled market's close read as its realized close for THIS run
# (default: config.BACKTEST_SCHEDULED_CLOSE, on, which reads the scheduled close a
# live run saw wherever a rule learned per series from backtest_cache/scheduled_closes/
# gives one; a series with no rule keeps the realized close either way)
python3 -m kalshi_betting.backtest --no-scheduled-close
python3 -m kalshi_betting.backtest --scheduled-close

# Backtest with the stale-quote check at 48 hours, or off, for THIS run only (default:
# config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS, 24 hours — an entry, a sale or a Kelly
# trim at a checkpoint needs each of its markets to have traded, or moved its quote,
# within that long before it; backtest only)
python3 -m kalshi_betting.backtest --stale-quote-hours 48
python3 -m kalshi_betting.backtest --no-stale-quote-check

# Backtest with the volume cap at 5% of each market's next-day traded contracts, or off,
# for THIS run only (default: config.BACKTEST_MAX_VOLUME_SHARE, 10%; it reads hours a live
# run cannot see and only ever makes a trade smaller; backtest only)
python3 -m kalshi_betting.backtest --max-volume-share 5
python3 -m kalshi_betting.backtest --no-volume-cap

# Backtest at a specific PRIMARY-scenario time-series spread band on pB - pA
# (backtest only; the live band is the saved live defaults', or main.py's own
#  --spread-min/--spread-max for one live run, read only through config.LiveSettings — no module outside config/backtester/backtest/dashboard calls
#  config.min_price_diff_for_gap at all)
python3 -m kalshi_betting.backtest --spread-min 0.30 --spread-max 0.60

# Backtest without the band x k x population scenario sweep AND without its
# tier-floors-off re-runs of every band a deadline-gap tier binds at (both are
# ON by default in the CLI; the re-runs have no flag of their own, so this skips
# both); the primary scenario still runs, but the dashboard's Scenario Explorer
# section has no grid to show, and its filter bar offers the primary band only,
# with the Tier floors select disabled
python3 -m kalshi_betting.backtest --no-band-sweep

# Backtest without the per-trade size-cap sweep (the cap sweep is ON by default in
# the CLI but simulates nothing during the run: BacktestSweep.cap_sweep simulates each
# cap other than config.BUDGET_FRACTION lazily, one (band, k) cell at a time, as the
# dashboard's one grid walk reads it for the filter bar, the Interval Discount section
# and the scenario explorer — skipping it gives a far smaller, faster page whose
# Size cap selects offer the run's own cap only — with the tier floors off too, since
# the tier-floors-off runs then have no size-cap sweep of their own either; backtest
# only — live sizing reads its caps from the saved live defaults, or main.py's --size-cap /
# --same-title-size-cap for one live run, through the run's LiveSettings)
python3 -m kalshi_betting.backtest --no-cap-sweep

# Backtest without the dashboard's "Rebalance" family (ON by default in the CLI, and lazy like
# the cap sweep: the run simulates nothing — BacktestSweep.rebalance_sweep simulates every
# band x k x cap cell with held pairs both added to and trimmed (add_to_held and trim_to_kelly
# together) when the dashboard reads it, the tier floors on and off, the "all" population
# only). Skipping it makes that dashboard step faster and leaves the filter bar's Rebalance
# and Min. price movement selects disabled; backtest only — a live run adds to and trims held
# pairs by its own saved settings
python3 -m kalshi_betting.backtest --no-rebalance-sweep

# Backtest with the "Rebalance" family but without its minimum price movements (ON by default
# with that family, and lazy the same way: BacktestSweep.rebalance_sweep.min_moves holds every
# value of config.MIN_PRICE_MOVES, 5c to 95c, where a held pair is added to or trimmed only
# once its total price has moved that far, up or down, since the last trade on it; each is
# simulated only when a report reads it, and only where it changes a run). Skipping them skips
# the dashboard's Min. price movement select; Rebalance itself is still built, and
# --no-rebalance-sweep skips both. Backtest only
python3 -m kalshi_betting.backtest --no-move-sweep

# Backtest without the dashboard's "Sell" family (ON by default in the CLI, and lazy like the
# cap sweep: the run simulates nothing — the dashboard step then reads every sell level and
# minimum of days to maturity of every scenario the filter bar shows that does not rebalance
# (Sell is shut while Rebalance is yes), in worker processes,
# simulating only the settings that differ, and writes the new trade lists as sidecar chunk files
# in backtest_dashboard_files/<build>/ beside the page, which must be kept with it). Skipping it
# makes that step much faster and leaves the filter bar's Sell and Min. days to maturity selects
# disabled; backtest only — a live run sells only at its own saved sell_at level (--sell-at for one run), not at the dashboard's grid
python3 -m kalshi_betting.backtest --no-sell-sweep

# Backtest building the dashboard's Sell select in 4 worker processes (default: one less than
# this machine's CPU count, at most config.DASHBOARD_SELL_MAX_WORKERS = 8; 1 runs it in the main
# process). Each worker holds one spread band's entries, so memory grows with the count
python3 -m kalshi_betting.backtest --sell-workers 4

# Rebuild backtest_dashboard.html without running the backtest again: every full run saves what
# it hands the dashboard to backtest_sweep.pkl.gz in the repo root (gitignored) right after the
# sweep, and this loads it, reads the T-bill yields again and builds the page with the same
# arguments — no fetch, no Kalshi request. The page's lazy cells (size caps, Sell, Rebalance)
# are still simulated while it is built, so it takes as long as the run's own
# dashboard step. --sell-workers applies. The flags that shape a run (--start-date, --balance,
# --no-cache, --max-horizon-days, --interval-discount, --no-sweep, the ladder flags, --spread-min,
# --spread-max, --no-band-sweep) are refused with it (exit 2). A missing file exits 2, a damaged one 1
# (a file saved in an earlier format, before the one Rebalance family or before the live category filter became a list of categories to exclude, counts as damaged: run the
# backtest again)
python3 -m kalshi_betting.backtest --dashboard-only

# The same, quickly, to check a small dashboard change: each of --no-sell-sweep,
# --no-rebalance-sweep, --no-move-sweep and --no-cap-sweep drops that family from the saved run,
# giving the page a run with that flag would have built (the Sell select alone took 1.4 to 4.6 h on
# the last 365-day runs)
python3 -m kalshi_betting.backtest --dashboard-only --no-sell-sweep --no-rebalance-sweep

# A rebuild is refused (exit 2) when the simulator's code (backtester, strategy, scanner,
# depth_model, historical, or the parts of config.py they read) changed since the run was saved:
# the lazy cells would then come from the new code and the run's other figures from the old.
# This builds the page anyway, with a WARNING naming the files that changed
python3 -m kalshi_betting.backtest --dashboard-only --accept-changed-simulator

# A backtest (or a --dashboard-only rebuild) without rebuilding the Data page at its end. By
# default the Data page is rebuilt after the backtest page, with at most config.EDA_MAX_WORKERS
# workers: the first build (or one after many new day files) reads each cached day file once
# and can take several minutes (about 10 on a full cache); later builds take seconds. A failure
# there is one WARNING and changes neither the exit code nor the backtest page
python3 -m kalshi_betting.backtest --no-eda
python3 -m kalshi_betting.backtest --dashboard-only --no-eda

# Rebuild the Data page (eda_dashboard.html: what backtest_cache/ holds, price calibration,
# deadline ladders and k, data quality, and a Terms dictionary; the live dashboard's Data tab)
# from the cache, without a backtest. Every backtest, and every --dashboard-only rebuild,
# already rebuilds it after its own page, with that run's k figures (--no-eda skips that). Reads only the day files that are new or changed since the last build, reuses
# the k figures the last backtest saved, makes no request to Kalshi, and prints the page's
# path; --workers defaults to one less than the CPU count, at most config.EDA_MAX_WORKERS
python3 -m kalshi_betting.eda [--workers N]

# Save live order books for the backtest's depth model: read-only, no orders, about 2,000 open markets, saved under backtest_cache/depth_snapshots/ (more snapshots near Monday 09:00 PT make its table better; a backtest fits every saved snapshot before its fetch)
python3 -m kalshi_betting.depth_model snapshot [--markets N]

# Record every open market's live close_time for the backtest's scheduled-close rule: read-only, no orders,
# saved under backtest_cache/scheduled_closes/<UTC stamp>.json.gz (a few MB; run it weekly so the rule keeps learning)
python3 -m kalshi_betting.market_details capture-closes

# Audit the backtest's side caches (market details, trades) and the full hourly candle cache: candle volumes against
# each market's volume_fp against its trades, the order inside each candle, trade prices, settlement values and
# timing fields. Read-only, prints the counts and a few examples of each failing check, exits 1 on any failure
python3 -m kalshi_betting.market_details check

# Weekly scheduler daemon (runs prod every Monday 09:00 on the host's clock —
# config.SCHEDULED_RUN, from the saved live defaults; logs CRITICAL at startup unless the
# host keeps America/Los_Angeles, and an ERROR when no live defaults are saved or the file
# is refused, since every run would exit 2)
python3 -m kalshi_betting.scheduler

# The dashboard, the live defaults and trading from a browser. ./start_dashboard.sh starts two
# servers from this checkout and opens http://127.0.0.1:8766/ on the Live trading tab. The
# live dashboard (127.0.0.1:8766 and :8767, read-only) serves the page's three tabs: Live
# trading, the account read from Kalshi on every load (value by category over time, returns
# by category, Total return, Profit, Sharpe, Sortino, Mean and Median trade for All, 1Y, 6M,
# 3M and 1M, and the holdings now; see "The Live trading tab" under Key Patterns),
# Backtest, backtest_dashboard.html as it is, and Data, eda_dashboard.html (what
# backtest_cache/ holds, rebuilt at the end of each backtest). The defaults server (127.0.0.1:8765) serves the
# pages the Backtest tab's "Save as live defaults…" button (the filter bar's scenario on
# screen) and "Trade using defaults…" link (/trade, the saved defaults) open. The launcher
# takes only --seed, --no-browser and -h/--help, spelled in full (anything else exits 2
# before anything starts), runs from its own folder (so from any directory), checks that its
# Python (KALSHI_PYTHON, else python3) can import the live bot and the live dashboard from
# that folder (it refuses a link to the script, or PYTHONSAFEPATH, which would import
# another checkout's copy), then starts the live dashboard and, 2 s later, the defaults
# server, both in the background. --seed opens the defaults server's confirmation page for
# the seed values instead of the Live trading tab; --no-browser opens nothing and only logs
# the addresses. When the live dashboard cannot start (its port held by another program, or
# by this checkout's dashboard running older code) and stops within those 2 s, a warning says
# so and the defaults server opens the backtest dashboard itself (or its own index when there
# is no dashboard, or one built before its Save and Trade buttons); when it stops later (a
# port holder that never answers, say), a warning names its exit status and the defaults
# server keeps running, opening nothing. Run
# it in the scheduler's checkout before the first live run, in the first-run order of the
# saved-live-defaults paragraph under Key Patterns (--seed proposes Rebalance yes): with no
# saved file, or one saved before Rebalance, every live run exits 2. Confirm and trade (on /confirm it saves
# first; on /trade it trades the saved defaults) and Dry run (no orders, nothing saved) start
# `python -m kalshi_betting.main --mode prod` with all eleven toggles as flags, in its own
# session; each run's folder is under live_runs/, and its page is /runs/<id>. Run it again
# while this checkout's servers run and it only opens the page again (nothing started); a
# defaults-server port held by that server running code from before a change to the
# checkout (stop it and start it again), by another checkout's server, or by anything else,
# exits 2 and stops the live dashboard just started. Ctrl-C stops both servers, and so does a
# SIGTERM or SIGHUP sent to the script alone
./start_dashboard.sh [--seed] [--no-browser]
python3 -m kalshi_betting.live_dashboard [--no-browser]             # the live dashboard alone (its three tabs, read-only), without the launcher's import check
python3 -m kalshi_betting.defaults_server [--seed] [--no-browser]   # the defaults server alone (it opens the backtest page itself), without the import check

# Live verification of the V2 order mapping + transfer unit (~1c of real money, prod account)
python3 -m kalshi_betting.v2_probe --ticker <TICKER> [--step no-mapping|unfillable-ask|yes-close|transfer] [--dest-shard N] [--yes]

# Live verification of the reduce-only ask that sells a held YES (about 1c of real money; you run
# it, on a liquid, cheap market you hold nothing on): buys 0.01 YES, sells it back with the order
# live selling and live trimming send, and PASSes only when the account reads flat and the sale's
# reply reports the 0.01 sold. Run it, and see it PASS, before saving a sell level or turning
# rebalancing on
python3 -m kalshi_betting.v2_probe --ticker <TICKER> --step yes-close

# Run tests
python3 -m pytest tests/ -v

# Lint check
python3 -m ruff check kalshi_betting/
```
