import os

import numpy as np
import pandas as pd

from data import load_prices_parquet


def _write_partitioned_parquet(base_dir: str, df: pd.DataFrame, *, ts: int = 1) -> None:
    for (ticker, year), g in df.groupby(["ticker", "year"], sort=False):
        part_dir = os.path.join(base_dir, f"ticker={ticker}", f"year={int(year)}")
        os.makedirs(part_dir, exist_ok=True)
        path = os.path.join(part_dir, f"part-{ts}.parquet")
        g.drop(columns=["ticker", "year"]).to_parquet(path, index=False)


def test_load_prices_from_parquet_basic(tmp_path) -> None:
    parquet_dir = tmp_path / "opcvm_parquet"
    meta_path = tmp_path / "ticker_meta.parquet"
    os.makedirs(parquet_dir, exist_ok=True)

    dates = pd.bdate_range("2024-01-01", periods=40)
    long = pd.DataFrame(
        {
            "date": np.tile(dates.date, 2),
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": np.concatenate(
                [100 + np.arange(len(dates)), 200 + np.arange(len(dates))]
            ).astype(float),
        }
    )
    long["year"] = pd.to_datetime(long["date"]).dt.year.astype(int)
    _write_partitioned_parquet(str(parquet_dir), long, ts=1)

    meta = pd.DataFrame({"ticker": ["AAA", "BBB"], "name": ["Fund A", "Fund B"]})
    meta.to_parquet(meta_path, index=False)

    prices, ticker_names = load_prices_parquet(
        years=1,
        annual_factor=20,
        fill_ratio=0.9,
        ticker_meta_path=str(meta_path),
        parquet_dir=str(parquet_dir),
    )

    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.index.is_monotonic_increasing
    assert set(ticker_names.index) == {"AAA", "BBB"}
    assert ticker_names["AAA"] == "Fund A"


def test_load_prices_fill_ratio_drops_sparse(tmp_path) -> None:
    parquet_dir = tmp_path / "opcvm_parquet"
    meta_path = tmp_path / "ticker_meta.parquet"
    os.makedirs(parquet_dir, exist_ok=True)

    dates = pd.bdate_range("2024-01-01", periods=30)
    long = pd.DataFrame(
        {
            "date": np.tile(dates.date, 2),
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": np.concatenate(
                [
                    100 + np.arange(len(dates)),
                    np.where(
                        np.arange(len(dates)) < 10, np.nan, 200 + np.arange(len(dates))
                    ),
                ]
            ).astype(float),
        }
    )
    long["year"] = pd.to_datetime(long["date"]).dt.year.astype(int)
    _write_partitioned_parquet(str(parquet_dir), long, ts=2)

    meta = pd.DataFrame({"ticker": ["AAA", "BBB"], "name": ["Fund A", "Fund B"]})
    meta.to_parquet(meta_path, index=False)

    prices, _ = load_prices_parquet(
        years=1,
        annual_factor=30,
        fill_ratio=0.9,
        ticker_meta_path=str(meta_path),
        parquet_dir=str(parquet_dir),
    )

    assert "AAA" in prices.columns
    assert "BBB" not in prices.columns


def test_load_prices_fill_ratio_vs_shorter_history_than_window(tmp_path) -> None:
    """fill_ratio must apply to actual rows after tail(), not requested min_obs."""
    parquet_dir = tmp_path / "opcvm_parquet"
    os.makedirs(parquet_dir, exist_ok=True)

    dates = pd.bdate_range("2020-01-01", periods=100)
    long = pd.DataFrame(
        {
            "date": np.tile(dates.date, 2),
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": np.concatenate(
                [100 + np.arange(len(dates)), 200 + np.arange(len(dates))]
            ).astype(float),
        }
    )
    long["year"] = pd.to_datetime(long["date"]).dt.year.astype(int)
    _write_partitioned_parquet(str(parquet_dir), long, ts=3)

    prices, _ = load_prices_parquet(
        years=10,
        annual_factor=252,
        fill_ratio=0.9,
        ticker_meta_path=None,
        parquet_dir=str(parquet_dir),
    )

    assert list(prices.columns) == ["AAA", "BBB"]
    assert len(prices) == 100
