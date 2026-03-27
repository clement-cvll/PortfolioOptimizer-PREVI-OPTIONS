"""Centralised configuration for the portfolio optimiser."""

import os

# ── Database ──────────────────────────────────────────────────────────────────
DB_URL: str = os.getenv(
    "PORTFOLIO_DB_URL",
    "postgresql+psycopg2://postgres:password@localhost/postgres",
)

# ── Universe & History ────────────────────────────────────────────────────────
YEARS: int = 6
ANNUAL_FACTOR: int = 252  # trading days per year
MIN_DATA_FILL_RATIO: float = 0.95  # drop assets with less coverage

# ── Optimisation ──────────────────────────────────────────────────────────────
MAX_WEIGHT: float = 1.0  # upper bound per asset (1.0 = unconstrained)
RISK_FREE_ANNUAL: float = 0.0193  # €STR (ECB overnight rate, Mar 2026)
WEIGHT_THRESHOLD: float = 0.01  # hide weights below 1 % in output

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_SAMPLES: int = 500_000
MC_SEED: int = 42

# ── Walk-Forward Backtest ─────────────────────────────────────────────────────
REBAL_DAYS: int = 126  # rebalance every rebalance period
MIN_TRAIN_DAYS: int = 504  # minimum days before first rebalance
TRANSACTION_COST: float = 0.01  # 1% proportional cost on turnover

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR: str = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR: str = os.path.join(SRC_DIR, "figures")
TICKERS_CSV: str = os.path.join(SRC_DIR, "tickers.csv")
