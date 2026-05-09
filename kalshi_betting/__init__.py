"""
File: __init__.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Package initializer for the kalshi_betting arbitrage bot. This package
    implements a two-path arbitrage strategy on the Kalshi prediction market
    platform: it finds pairs of correlated contracts where one leg is mispriced
    relative to the other, sizes trades using the Kelly criterion, submits
    batch orders to the Kalshi REST API, and logs results to Excel. A separate
    backtest pipeline replays the same strategy on historical settled markets
    and generates an interactive HTML performance dashboard.

Dependencies:
    This file has no imports. All public modules are accessible as
    kalshi_betting.<module> after this package is imported.
"""
