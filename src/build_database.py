"""build_database.py — Scrape OPCVM tickers and persist them as Parquet.

Usage:
    python build_database.py           # incremental update
    python build_database.py --rebuild # full re-scrape (rebuild Parquet from scratch)
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import (
    LAST_DATES_PATH,
    PARQUET_DIR,
    TICKER_META_PATH,
    TICKERS_CSV,
)

MAX_WORKERS = 10
REQUEST_TIMEOUT = 15  # seconds

PREVI_URL = "https://www.previ-direct.com/web/eclient-suravenir/perf-uc-previ-options"


def fetch_tickers() -> pd.DataFrame:
    """Scrape Previ-Options and return a validated {ticker → name} DataFrame."""
    response = requests.get(PREVI_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    units = [
        {"unit_isin": a1.text.strip(), "unit_name": a2.text.strip()}
        for row in soup.find_all("tr", class_="portlet-section-alternate results-row")
        if len(tds := row.find_all("td")) >= 2
        and (a1 := tds[0].find("a"))
        and (a2 := tds[1].find("a"))
    ]

    def lookup(unit: dict[str, str]) -> list[str]:
        try:
            return yf.Lookup(unit["unit_isin"]).all.index.tolist()
        except Exception:
            return []

    def filter_ticker(ticker: str) -> tuple[str, dict[str, str | int]] | None:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info.get("currency") != "EUR":
                return None
            hist = t.history(period="10y")
            if len(hist) < 252:
                return None
            name = info.get("longName", info.get("shortName", "")).replace('"', "")
            return ticker, {"name": name, "n_quotes": len(hist)}
        except Exception:
            return None

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        candidates = set(
            t
            for ts in tqdm(
                pool.map(lookup, units), total=len(units), desc="Looking up tickers"
            )
            for t in ts
        )
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(filter_ticker, t): t for t in candidates}
        results = [
            f.result()
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="Filtering tickers"
            )
        ]

    df = pd.DataFrame.from_dict(
        {t: m for t, m in (r for r in results if r)}, orient="index"
    )
    df = df.loc[df.groupby("name")["n_quotes"].idxmax()].drop(columns=["n_quotes"])
    return df


def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(TICKER_META_PATH), exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)


def _load_last_dates() -> dict[str, object]:
    """Load {ticker: last_date} from LAST_DATES_PATH if present."""
    if not os.path.exists(LAST_DATES_PATH):
        return {}
    df = pd.read_parquet(LAST_DATES_PATH)
    if df.empty:
        return {}
    df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
    return dict(zip(df["ticker"].astype(str), df["last_date"], strict=False))


def _save_last_dates(last_dates: dict[str, object]) -> None:
    out = (
        pd.DataFrame(
            {"ticker": list(last_dates.keys()), "last_date": list(last_dates.values())}
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    out.to_parquet(LAST_DATES_PATH, index=False)


def _to_records(ticker: str, name: str, hist: pd.DataFrame, last_date) -> list[tuple]:
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    if last_date is not None:
        hist = hist[hist["Date"].dt.date > last_date]
    return [
        (
            row.Date.date(),
            None if pd.isna(row.Open) else float(row.Open),
            None if pd.isna(row.High) else float(row.High),
            None if pd.isna(row.Low) else float(row.Low),
            None if pd.isna(row.Close) else float(row.Close),
            ticker,
            name,
        )
        for row in hist.itertuples(index=False)
    ]


def _records_to_frame(records: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        records, columns=["date", "open", "high", "low", "close", "ticker", "name"]
    )
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year.astype(int)
    return df


def _download_and_prepare(
    tickers: list[str],
    tickers_df: pd.DataFrame,
    last_dates: dict,
    **dl_kwargs,
) -> list[tuple]:
    if not tickers:
        return []
    raw = yf.download(
        tickers,
        group_by="ticker",
        auto_adjust=False,
        progress=True,
        threads=True,
        **dl_kwargs,
    )
    records: list[tuple] = []
    for t in tqdm(tickers, desc="Preparing"):
        try:
            hist = (raw[t] if len(tickers) > 1 else raw).dropna(how="all")
            records += _to_records(
                t, tickers_df.loc[t, "name"], hist, last_dates.get(t)
            )
        except Exception as exc:
            print(f"  [WARN] {t}: {exc}", file=sys.stderr)
    return records


def _write_parquet_partitioned(df: pd.DataFrame) -> int:
    """Append by writing new Parquet files into hive partitions."""
    if df.empty:
        return 0
    # Keep fact table lean: name is stored separately as metadata.
    fact = df.drop(columns=["name"])
    # Write each (ticker, year) group as its own file to avoid expensive merges.
    ts = int(time.time() * 1000)
    rows = 0
    for (ticker, year), g in fact.groupby(["ticker", "year"], sort=False):
        part_dir = os.path.join(PARQUET_DIR, f"ticker={ticker}", f"year={int(year)}")
        os.makedirs(part_dir, exist_ok=True)
        path = os.path.join(part_dir, f"part-{ts}.parquet")
        g.drop(columns=["year"]).to_parquet(path, index=False)
        rows += len(g)
    return rows


def ingest_all(tickers_df: pd.DataFrame, last_dates: dict) -> int:
    """Download and persist all ticker data. Returns the number of rows written."""
    tickers = list(tickers_df.index)
    new = [t for t in tickers if t not in last_dates]
    existing = [t for t in tickers if t in last_dates]
    records = _download_and_prepare(
        new, tickers_df, {}, period="max"
    ) + _download_and_prepare(
        existing,
        tickers_df,
        last_dates,
        start=str(min(last_dates[t] for t in existing)) if existing else None,
    )
    df = _records_to_frame(records)
    written = _write_parquet_partitioned(df)
    if written:
        # Update last_dates from the *newly written* batch.
        maxes = df.groupby("ticker")["date"].max().to_dict() if not df.empty else {}
        for t, d in maxes.items():
            prev = last_dates.get(t)
            if prev is None or d > prev:
                last_dates[t] = d
        _save_last_dates(last_dates)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-scrape tickers and rebuild Parquet from scratch",
    )
    args = parser.parse_args()
    _ensure_dirs()

    if args.rebuild or not os.path.exists(TICKERS_CSV):
        tickers_df = fetch_tickers()
        tickers_df.to_csv(TICKERS_CSV, index=True, index_label="ticker")
        print(f"Saved {len(tickers_df)} tickers to {TICKERS_CSV}")
    else:
        tickers_df = pd.read_csv(TICKERS_CSV, index_col=0)

    if args.rebuild:
        if os.path.exists(PARQUET_DIR):
            shutil.rmtree(PARQUET_DIR)
        if os.path.exists(TICKER_META_PATH):
            os.remove(TICKER_META_PATH)
        if os.path.exists(LAST_DATES_PATH):
            os.remove(LAST_DATES_PATH)
        os.makedirs(PARQUET_DIR, exist_ok=True)

    # Write ticker metadata (ticker -> name)
    meta = (
        tickers_df.reset_index(names="ticker")[["ticker", "name"]]
        .astype({"ticker": str, "name": str})
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    meta.to_parquet(TICKER_META_PATH, index=False)

    last_dates = {} if args.rebuild else _load_last_dates()
    total = ingest_all(tickers_df, last_dates)
    print(f"Done — {total} rows written to Parquet.")


if __name__ == "__main__":
    main()
