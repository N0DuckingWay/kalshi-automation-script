<!--
Write for someone who has not read the code: plain words, short sentences,
every abbreviation spelled out the first time, and a file or function name
only where a reviewer would look for it, with its job in words beside it.
The full guide, with examples, is docs/claude/writing-for-people.md.
These comments do not show on the pull request.
-->

## What changes

<!-- Two to four plain sentences: what the bot, the backtest or the dashboard
does differently now, as the person using it would notice. No function names. -->

## Why

<!-- The problem this solves, or what it makes possible, in plain words. -->

## Real money

<!-- Could this change what the bot buys or sells, how much, or how it sends
orders? If yes: what could go wrong, roughly how much it could cost, and what
to run before the next live run (for example a production dry run:
python3 -m kalshi_betting.main --mode prod --dry-run).
If no, write: No live trading code changed. -->

## How it works

<!-- A few short bullets for a reviewer. Name a file, function or setting only
where they would look for it, and say what it does. -->

## Checked

<!-- What was tested, in plain words, and what it showed: the test suite,
ruff, a dry run, a backtest. -->

<!-- Optional, for detail most readers can skip (each new field, log line or
edge case). GitHub shows it closed. Copy it out of this comment to use it.

<details><summary>More detail</summary>

</details>
-->
