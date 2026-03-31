"""Centralised configuration for the portfolio optimiser."""

import os

# ── Storage (DuckDB + Parquet) ────────────────────────────────────────────────
# Canonical store: partitioned Parquet dataset under PARQUET_DIR
DATA_DIR: str = os.getenv(
    "PORTFOLIO_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")
)
PARQUET_DIR: str = os.getenv(
    "PORTFOLIO_PARQUET_DIR", os.path.join(DATA_DIR, "opcvm_parquet")
)
TICKER_META_PATH: str = os.getenv(
    "PORTFOLIO_TICKER_META_PATH", os.path.join(DATA_DIR, "ticker_meta.parquet")
)
LAST_DATES_PATH: str = os.getenv(
    "PORTFOLIO_LAST_DATES_PATH", os.path.join(DATA_DIR, "last_dates.parquet")
)

# ── Legacy DB (optional) ───────────────────────────────────────────────────────
# Only used if you explicitly set PORTFOLIO_DB_URL and run in legacy mode.
DB_URL: str = os.getenv("PORTFOLIO_DB_URL", "")

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
TURNOVER_PENALTY: float = 0.05  # soft penalty on |w - w_prev| during optimisation

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR: str = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR: str = os.path.join(SRC_DIR, "figures")
TICKERS_CSV: str = os.path.join(SRC_DIR, "tickers.csv")
