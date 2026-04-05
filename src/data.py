"""Data access layer — load prices from hive-partitioned Parquet."""

import os

import numpy as np
import pandas as pd
import pyarrow.dataset as pds


def load_prices_parquet(
    *,
    parquet_dir: str,
    ticker_meta_path: str | None,
    years: int,
    annual_factor: int = 252,
    fill_ratio: float = 0.94,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load close prices for the last *years* of trading.

    Returns (prices DataFrame indexed by date, ticker→name Series).
    Assets with too many NaNs (below *fill_ratio*) are dropped.
    """
    if not parquet_dir or not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"parquet_dir not found: {parquet_dir!r}")

    dataset = pds.dataset(parquet_dir, format="parquet", partitioning="hive")
    df = dataset.to_table(columns=["date", "ticker", "close"]).to_pandas()

    if df.empty:
        raise RuntimeError(
            f"No data found under parquet_dir={parquet_dir!r}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).astype("category")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(
        np.float64
    )

    cutoff = df["date"].max() - pd.DateOffset(years=years)
    df = df[df["date"] >= cutoff]

    prices = df.pivot(index="date", columns="ticker", values="close")
    prices = prices.sort_index().tail(annual_factor * years)
    thresh = max(1, int(fill_ratio * len(prices)))
    prices = prices.dropna(axis=1, thresh=thresh).dropna(axis=0)

    if ticker_meta_path and os.path.exists(ticker_meta_path):
        ticker_names = pd.read_parquet(
            ticker_meta_path
        ).set_index("ticker")["name"]
    else:
        ticker_names = pd.Series(dtype=object)

    return prices, ticker_names
