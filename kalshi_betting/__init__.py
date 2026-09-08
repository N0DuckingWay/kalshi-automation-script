"""
File: __init__.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Package initializer for the kalshi_betting bot. This package implements
    two pair strategies on the Kalshi prediction market platform. Same-title
    pairs are a near-arbitrage: one question listed twice at divergent prices,
    where buying NO on the pricier listing and YES on the cheaper one profits
    whenever the two co-resolve. Time-series pairs are a directional bet on
    the interval between two deadlines: when the later-closing contract is
    priced well above the earlier one, the bot buys YES on the earlier and NO
    on the later, winning if the event happens by the earlier deadline or
    never happens by the later one, and losing the stake if it first happens
    in between. Both are sized with the Kelly criterion, submitted as
    fill-or-kill orders leg-by-leg to the Kalshi REST API (there is no batch
    order endpoint), and logged to Excel. A separate backtest pipeline replays
    the same strategies on historical settled markets and generates an
    interactive HTML performance dashboard.

Dependencies:
    This file has no imports. All public modules are accessible as
    kalshi_betting.<module> after this package is imported.
"""
