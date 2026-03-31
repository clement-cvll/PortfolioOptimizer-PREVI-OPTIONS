"""Data access layer (DuckDB + Parquet)."""

from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd


def load_prices_parquet(
    *,
    parquet_dir: str,
    ticker_meta_path: str | None,
    years: int,
    annual_factor: int = 252,
    fill_ratio: float = 0.94,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load close prices from a partitioned Parquet dataset.

    Expected layout: hive partitions under parquet_dir, e.g.
    ticker=.../year=.../*.parquet
    Returns (prices DataFrame, ticker→name Series).
    """
    if not parquet_dir or not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"parquet_dir not found: {parquet_dir!r}")

    glob = os.path.join(parquet_dir, "**", "*.parquet")

    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(
            """
            WITH d AS (
                SELECT date, ticker, close
                FROM read_parquet(?, hive_partitioning=1)
            )
            SELECT date, ticker, close
            FROM d
            WHERE date >= (SELECT max(date) FROM d) - (? * INTERVAL '1 year')
            """,
            [glob, years],
        ).df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"No data found under parquet_dir={parquet_dir!r}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).astype("category")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)

    price_df = df.pivot(index="date", columns="ticker", values="close")
    min_obs = annual_factor * years
    prices = price_df.sort_index().tail(min_obs)
    prices = prices.dropna(axis=1, thresh=int(fill_ratio * min_obs)).dropna(axis=0)

    if not ticker_meta_path or not os.path.exists(ticker_meta_path):
        ticker_names = pd.Series(dtype=object)
    else:
        meta = pd.read_parquet(ticker_meta_path)
        ticker_names = meta.set_index("ticker")["name"]

    return prices, ticker_names
