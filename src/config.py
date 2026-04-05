"""Centralised configuration for the portfolio optimiser."""

import os

_SRC = os.path.dirname(os.path.abspath(__file__))

# Storage (partitioned Parquet under DATA_DIR)
DATA_DIR: str = os.path.join(_SRC, "data")
PARQUET_DIR: str = os.path.join(DATA_DIR, "opcvm_parquet")
TICKER_META_PATH: str = os.path.join(DATA_DIR, "ticker_meta.parquet")
LAST_DATES_PATH: str = os.path.join(DATA_DIR, "last_dates.parquet")

# Universe & history
YEARS: int = 6
ANNUAL_FACTOR: int = 252
MIN_DATA_FILL_RATIO: float = 0.7

# Optimisation
MAX_WEIGHT: float = 1.0
RISK_FREE_ANNUAL: float = 0.0193  # €STR (ECB overnight rate, Mar 2026)
WEIGHT_THRESHOLD: float = 0.01

# Walk-forward backtest
REBAL_DAYS: int = 126
MIN_TRAIN_DAYS: int = 504
TRANSACTION_COST: float = 0.005
TURNOVER_PENALTY: float = 0.00

# Output paths
FIGURES_DIR: str = os.path.join(_SRC, "figures")
