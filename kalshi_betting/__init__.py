"""
File: __init__.py
Author: Zachary Hoffman
Last edited by: Claude

Purpose:
    Package initializer for the kalshi_betting bot. This package implements
    one pair strategy on the Kalshi prediction market platform: time-series
    pairs, a directional bet on the interval between two deadlines. When the
    later contract is priced well above the earlier one, the bot buys YES on
    the earlier and NO on the later, winning if the event happens by the
    earlier deadline or never happens by the later one, and losing the stake
    if it first happens in between. (The bot once also traded same-title
    pairs, one question listed twice by two series; it no longer does, and
    every part of it that takes a pair refuses any other kind.) Each pair is
    sized with the Kelly criterion, submitted as fill-or-kill orders
    leg-by-leg to the Kalshi REST API (there is no batch order endpoint), and
    logged to Excel. A separate backtest pipeline replays the same strategy
    on historical settled markets and generates an interactive HTML
    performance dashboard.

Dependencies:
    This file has no imports. All public modules are accessible as
    kalshi_betting.<module> after this package is imported.

Notes:
    Deliberately empty of re-exports. Importing submodules here would make
    `import kalshi_betting` pull in the whole package — including trader.py,
    which submits real orders — and would give the dependency graph in
    CLAUDE.md a cycle through the package root that the modules themselves do
    not have. Callers import the module they need by name.
"""
