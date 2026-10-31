# Writing for people — code comments, docstrings, commit messages, pull requests and GitHub replies

How to write anything a person reads about this project: code comments and docstrings, commit messages, pull request titles and descriptions, and comments or replies on GitHub. The root `CLAUDE.md` has the short form of these rules (under Documentation Standards); this file has the full text, a list of plain meanings for the project's terms, and before-and-after examples. The pull request template is `.github/pull_request_template.md`.

In this file:

- Who the reader is
- Plain words first
- Define each project term once
- Name code only when the reader needs it
- Numbers carry units and meaning
- Say first whether real money is affected
- Code comments and docstrings
- Commit messages
- Pull request titles
- Pull request descriptions
- GitHub comments and review replies
- Tables
- Plain meanings for this project's terms
- Before and after
- A check before you post

**Who the reader is.** Write for someone smart who has not read this code. They know that a Kalshi contract pays $1 if its question is decided Yes and nothing if No, but they do not know this bot's shorthand, its history or its function names. After the first section of a pull request, or the first sentence of a comment, they should know what changed and whether it touches real money. "Just technical enough" means keeping the facts a reviewer needs (which setting, its default, what happens when something fails) and dropping the plumbing (private function names, data layouts, call order) unless the reviewer has to check it. It does not mean leaving facts out: say less about how the code is arranged and more about what it does.

**Plain words first.** Use short sentences, everyday words and the active voice: "the bot sells", not "a sale is submitted". Say what happens, not how the code is arranged: "the bot no longer tops up a pair in its last week" beats "`plan_add_ons` now gates on `days_left_reached`". One idea per sentence. Spell out every abbreviation the first time (FoK, IOC, TS, ST, MVE, DD), or use the plain phrase instead ("a fill-or-kill order, which buys every contract asked for at once or none").

**Define each project term once.** The first time a piece of writing uses a project term, give its plain meaning in a few words; the list under "Plain meanings for this project's terms" has short ones. For a figure the dashboards show, use the wording in `kalshi_betting/glossary.py`, so the writing matches the page. After that, use the same word every time: do not switch between "trim", "sell down" and "reduce" for one thing.

**Name code only when the reader needs it.** Put a function, constant, flag, setting or file name in backticks, and use one only where a reviewer would look for it or a user would type or see it (a command-line flag, a saved setting, a log line). Give its job in words the first time: "`seller.plan_trims`, which picks the held pairs to sell part of". If you deleted every backticked name, each sentence should still say what happens. Private names (a leading underscore), internal class names and the layout of data passed between functions belong in the diff and the docstrings, not in the opening sections of a pull request.

**Numbers carry units and meaning.** Give each number its unit: cents or dollars for prices ("45¢"), dollars for money, contracts for counts, calendar days for waits, and say when a share is of the portfolio value or of the cash. Round to what matters ("+4.7% over the year", not "0.04712"). Say what a number means for the reader: "the backtest returned 4.7%, but lost 37.5% without its single best event, so the gain rests on that one event".

**Say first whether real money is affected.** This bot trades real money. Any change that could alter what it buys or sells, how much, or how it sends orders says so near the top, in plain words: what could go wrong, roughly how much it could cost, and what the owner must do before the next live run (for example a production dry run, or the V2 probe). A change that cannot touch a live order says "No live trading code changed." A reader should never have to read the diff to learn whether their money is affected.

**Code comments and docstrings.** Say what the code does now and how it fits the rest of the codebase: who calls it, what it feeds and why it exists. Leave out history: no bug stories, no "until 2026-09-25", no "used to" or "has always", and no dated measurements; those go in the commit message and the pull request. A safety rule stays at the code it protects, stated as a rule with its reason ("sent exactly once and never retried: a retried order can fill twice"). A finding tag such as DR-74 may point a reader to these notes, but always beside a few plain words saying what it is ("the one-hour close rule (DR-74)"). A test's docstring says what the test checks and why that matters. When you edit a comment that tells a story, rewrite it to describe the code as it is. Plain rewrites drift from the code easily ("entered" where the code means "picked by the sizing rule", "would" where only "could" is true), so re-read every rewritten sentence against the code before committing.

**Commit messages.** The subject is one plain line of about 70 characters or fewer, saying what changes as a user would see it: "Stop rebalancing a pair in its last days before it closes", not "Add days_left_reached gate to plan_add_ons". The body says what and why in plain sentences, then any detail a later reader needs: measured figures, the tests added, what was checked. History and measurements belong here, not in code comments. End with the attribution lines the session gives.

**Pull request titles.** A plain sentence about the effect: what the bot, the backtest or the dashboard does differently. No function names. Put series bookkeeping ("PR 4b of rebalance lone legs") at the end in brackets, if anywhere.

**Pull request descriptions.** Use the sections of `.github/pull_request_template.md`, in its order: What changes, Why, Real money, How it works, Checked. The first three are for anyone: plain words, no function names, and short enough to read in a minute. "How it works" is for a reviewer: a few short bullets, each naming a file, function or setting only where the reviewer would look, with what it does. "Checked" says in plain words what was tested and what it showed (the test suite, ruff, a dry run, a backtest). Detail most readers can skip (each new field, each log line, each edge case) goes at the end inside `<details><summary>More detail</summary> … </details>`, which GitHub shows closed; it is still written in sentences. Keep the part shown open to about one screen, and do not retell the diff line by line. Link another pull request by its full URL. End with the attribution line the session gives.

**GitHub comments and review replies.** Answer in the first sentence ("Yes, fixed in abc1234: the bot now …", or "No: the bot never …, because …"), then explain. Point at code by quoting the line or linking it, rather than by naming a function the reader must look up. Keep one comment to one point. A reply posted with `gh` from a local session appears under the owner's account, so start it with "🤖 Claude:" so readers know who wrote it.

**Tables.** Short, plain column headers ("What it does", "Before", "After", "Files"), one short phrase per cell. Put identifiers, test names and caveats in a list or a sentence under the table.

**Plain meanings for this project's terms.** Use these, or the `glossary.py` wording where it has the term, the first time a term appears:

| Term | Plain meaning |
|---|---|
| Pair | Two contracts bought together as one bet: Yes on one market, No on another |
| Leg | One of the two contracts in a pair |
| Time-series pair | One question at two deadlines; the bot buys Yes on the earlier and No on the later. It loses if the event happens between the deadlines |
| Same-title pair | Two identically worded markets, listed separately, whose Yes prices differ by 5¢ or more; the bot buys No on the pricier and Yes on the cheaper |
| Ladder | One event that lists the same question at several deadlines |
| Kelly size | How much the sizing rule says to put on a trade, from its chance of paying and its price after fees, held to the size cap |
| Portfolio value | Cash plus what the open positions are worth |
| Trim / add-on | Selling part of a held pair that has grown past its Kelly size / buying more of one that has room below it |
| Rebalance | Trimming and adding on, so each held pair stays near its Kelly size |
| Fill-or-kill order | An order that buys every contract asked for at once, or none |
| Unwind | Selling back a pair's first contract when the second could not be bought, so the account is not left holding half a pair |
| Settle | A market's question is decided, and each contract pays $1 or nothing |
| Prod / dev | Prod is the real account and real money; dev is Kalshi's demo site, where the bot never sends orders |
| Dry run | A production run that finds and sizes trades but sends no orders |
| Backtest | Replaying the strategy on markets that have already settled, to see how it would have done |
| Sweep / cell | The backtest run once per combination of settings; one combination is a cell (a scenario on the dashboard) |
| V2 | Kalshi's current order system, the only one the bot uses |
| DR-74, TS-08 | Tags naming a recorded finding in these notes; always say in words what the finding is |

**Before and after.** Each pair below says the same thing; the second can be read without the code.

A pull request bullet. Before: "`_ChunkVisitor.maturity` files a simulated run as a chunk, and resolves each `SameMaturity` to a chunk already filed: the movement view's chunk for the maturity run, or the id it stored for an earlier choice or minimum." After: "Each new backtest run is stored as a separate file, which the page loads only when that choice is picked. A choice whose run is identical to one already stored reuses that file." (The class and key names, if a reviewer needs them, go under More detail.)

A title. Before: "Backtest "Rebalance until": the Rebalance family covers the three choices (rebalance lone legs, PR 4b)". After: "Backtest also runs "stop rebalancing 1 or 7 days before the close", ready for the dashboard (PR 4b of rebalance lone legs)".

A code comment. Before: "# DR-74 gate (added 2026-09-25 after the NCAA mispair drove a −48% drawdown); fail closed on None". After: "# Same-title pairs must close within an hour of each other (DR-74), so a men's and a women's game with the same title are never paired. An unknown close time refuses the pair."

The opening of a pull request description:

```markdown
## What changes

A new live setting, **Rebalance until**. With rebalancing on, each weekly run tops up
or sells part of the pairs the account holds, to keep each near the size the sizing
rule wants. This setting stops that in the last 1 or 7 days before a pair's later
market closes. The default, "maturity", rebalances right up to the close, as now.

## Real money

Yes, once turned on: the bot makes fewer trims and top-ups near a pair's close. It
stays off until 1 or 7 is saved on the dashboard; a saved file without the setting
trades exactly as before.
```

**A check before you post.** Could someone who has never seen the code say, after the first section, what changed and whether it touches real money? Does every function, constant or file name have its job in words beside it, and is each one needed? Is every abbreviation spelled out the first time? Does every number have a unit? Is anything in a code comment only history, which belongs in the commit message instead?
