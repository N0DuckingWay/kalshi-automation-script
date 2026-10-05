# To-do list

- [ ] **[bt] Profitability of different selling thresholds that depend on days before maturity.**
  Test a sell threshold of (100 − x)% of potential profit, where x is the number of days left
  before the position's markets pay out (for example, sell at 95% of potential profit with 5 days
  to go, 90% with 10 days to go), instead of the single fixed level the Sell select uses now.
  Compare the total portfolio return against never selling and against the fixed levels, for the
  live-defaults scenario (tiers off, band 0–0.8, 15% cap, 20% same-title cap, add to held pairs
  on) and for all scenarios in the band.

  Starting point (2026-10-05): with fixed levels, selling early lowered whole-run return nearly
  everywhere, because sale proceeds were reinvested into trades that lose about half the time;
  only the 95% level was close to never selling. Harness, per-trade data and per-scenario returns
  are in `.git/sell-days/` (`sd_run.py`, `sold_trades.csv`, `scenario_total_returns.csv`).
  Open question to settle first: whether to also test holding the proceeds in cash instead of
  reinvesting, to separate the reinvestment effect from the capping-the-winner effect.
