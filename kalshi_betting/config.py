"""Configuration constants for the Kalshi arbitrage bot."""
import math
import pathlib
import sys

PROD_URL    = "https://api.elections.kalshi.com/trade-api/v2"
SANDBOX_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Absolute path to project root (where secrets.json and the PEM key live).
# This is fixed regardless of which git worktree the package runs from.
PROJECT_ROOT = pathlib.Path("/Users/zdhoffman/Documents/Coding Projects/Kalshi Betting App")
SECRETS_FILE = PROJECT_ROOT / "secrets.json"
PEM_FILE     = PROJECT_ROOT / "kalshi_private_key.pem"

# Trading parameters
BUDGET_FRACTION               = 0.20  # hard cap: max Kelly fraction per trade (20% of balance)
MIN_PRICE_DIFF                = 0.15  # min YES price gap for time-series pairs
SAME_TITLE_MIN_PRICE_DIFF     = 0.05  # min YES price gap for exact same-title pairs
SAME_TITLE_CO_RESOLVE_PROB    = 0.95  # assumed probability that same-title pairs co-resolve
MAX_DEADLINE_GAP_DAYS         = 30    # max days between pair deadlines
MIN_BALANCE_CENTS             = 500   # skip run if balance below $5
TAKER_FEE_RATE                = 0.07  # Kalshi taker fee: ceil(TAKER_FEE_RATE × C × P × (1−P))

# API pagination
MARKET_PAGE_SIZE   = 1000
POSITION_PAGE_SIZE = 500


def fee_per_pair_approx(nA: float, pB: float) -> float:
    """Continuous approximation of total Kalshi taker fee for a NO(A) + YES(B) pair."""
    return TAKER_FEE_RATE * (nA * (1.0 - nA) + pB * (1.0 - pB))


def fee_leg_exact(n: int, p: float) -> float:
    """Ceiling-rounded Kalshi taker fee for n contracts of one order leg at price p."""
    return math.ceil(TAKER_FEE_RATE * n * p * (1.0 - p) * 100) / 100


def parse_mode() -> str:
    """Return 'prod' or 'dev' based on --mode CLI arg. Defaults to 'dev'."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--mode" and i < len(sys.argv):
            m = sys.argv[i + 1].lower()
            if m in ("prod", "dev"):
                return m
    return "dev"


def is_dry_run() -> bool:
    return "--dry-run" in sys.argv
