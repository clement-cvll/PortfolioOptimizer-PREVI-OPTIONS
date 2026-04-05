"""Scrape OPCVM tickers from Previ-Options and persist them as Parquet.

Usage (standalone):
    python build_database.py           # incremental update
    python build_database.py --rebuild # full re-scrape
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

from config import LAST_DATES_PATH, PARQUET_DIR, TICKER_META_PATH

MAX_WORKERS = 10
_HTTP_TIMEOUT = (15.0, 120.0)  # (connect, read) seconds
_HTTP_RETRIES = 3

PREVI_URL = (
    "https://www.previ-direct.com/web/eclient-suravenir/perf-uc-previ-options"
)


# ── Scraping ─────────────────────────────────────────────────────────────────
def _fetch_previ_html() -> bytes:
    """GET the Previ performance page with exponential-backoff retries."""
    for attempt in range(_HTTP_RETRIES):
        try:
            r = requests.get(PREVI_URL, timeout=_HTTP_TIMEOUT)
            r.raise_for_status()
            return r.content
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == _HTTP_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"Previ request failed ({type(exc).__name__}), "
                  f"retry in {wait}s…", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_tickers() -> pd.DataFrame:
    """Scrape Previ-Options → ISIN lookup → yfinance validation → DataFrame.

    Filters: EUR-denominated, ≥ 1 year of history. Deduplicates by fund name,
    keeping the ticker with the most quotes.
    """
    soup = BeautifulSoup(_fetch_previ_html(), "html.parser")

    units = [
        {"unit_isin": a1.text.strip(), "unit_name": a2.text.strip()}
        for row in soup.find_all(
            "tr", class_="portlet-section-alternate results-row"
        )
        if len(tds := row.find_all("td")) >= 2
        and (a1 := tds[0].find("a"))
        and (a2 := tds[1].find("a"))
    ]

    def lookup(unit: dict[str, str]) -> list[str]:
        try:
            return yf.Lookup(unit["unit_isin"]).all.index.tolist()
        except Exception:
            return []

    def filter_ticker(ticker: str) -> tuple[str, dict] | None:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info.get("currency") != "EUR":
                return None
            hist = t.history(period="8y")
            if len(hist) < 252:
                return None
            name = info.get("longName", info.get("shortName", ""))
            return ticker, {"name": name.replace('"', ""), "n_quotes": len(hist)}
        except Exception:
            return None

    # Resolve ISINs → candidate tickers (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        candidates = {
            t
            for ts in tqdm(
                pool.map(lookup, units), total=len(units),
                desc="Looking up tickers",
            )
            for t in ts
        }
    # Validate candidates against yfinance (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(filter_ticker, t): t for t in candidates}
        results = [
            f.result()
            for f in tqdm(
                as_completed(futures), total=len(futures),
                desc="Filtering tickers",
            )
        ]

    df = pd.DataFrame.from_dict(
        {t: m for t, m in (r for r in results if r)}, orient="index"
    )
    # Keep the ticker with the most history per fund name
    return df.loc[df.groupby("name")["n_quotes"].idxmax()].drop(
        columns=["n_quotes"]
    )


# ── Parquet persistence ──────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(TICKER_META_PATH), exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)


def _load_last_dates() -> dict[str, object]:
    """Load {ticker: last_date} for incremental downloads."""
    if not os.path.exists(LAST_DATES_PATH):
        return {}
    df = pd.read_parquet(LAST_DATES_PATH)
    if df.empty:
        return {}
    df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
    return dict(zip(df["ticker"].astype(str), df["last_date"], strict=False))


def _save_last_dates(last_dates: dict[str, object]) -> None:
    pd.DataFrame({
        "ticker": list(last_dates.keys()),
        "last_date": list(last_dates.values()),
    }).sort_values("ticker").reset_index(drop=True).to_parquet(
        LAST_DATES_PATH, index=False
    )


def _to_records(
    ticker: str, name: str, hist: pd.DataFrame, last_date
) -> list[tuple]:
    """Convert yfinance history to flat (date, OHLC, ticker, name) tuples."""
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    if last_date is not None:
        hist = hist[hist["Date"].dt.date > last_date]
    cols = ["Open", "High", "Low", "Close"]

    def _val(row, c):
        v = getattr(row, c)
        return None if pd.isna(v) else float(v)

    return [
        (row.Date.date(), *(_val(row, c) for c in cols), ticker, name)
        for row in hist.itertuples(index=False)
    ]


def _records_to_frame(records: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        records,
        columns=["date", "open", "high", "low", "close", "ticker", "name"],
    )
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year.astype(int)
    return df


def _download_and_prepare(
    tickers: list[str], tickers_df: pd.DataFrame,
    last_dates: dict, **dl_kwargs,
) -> list[tuple]:
    if not tickers:
        return []
    raw = yf.download(
        tickers, group_by="ticker", auto_adjust=False,
        progress=True, threads=True, **dl_kwargs,
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
    """Append new rows as hive-partitioned Parquet files (one per ticker/year)."""
    if df.empty:
        return 0
    fact = df.drop(columns=["name"])
    ts = int(time.time() * 1000)
    rows = 0
    for (ticker, year), g in fact.groupby(["ticker", "year"], sort=False):
        part_dir = os.path.join(
            PARQUET_DIR, f"ticker={ticker}", f"year={int(year)}"
        )
        os.makedirs(part_dir, exist_ok=True)
        g.drop(columns=["ticker", "year"]).to_parquet(
            os.path.join(part_dir, f"part-{ts}.parquet"), index=False
        )
        rows += len(g)
    return rows


def ingest_all(tickers_df: pd.DataFrame, last_dates: dict) -> int:
    """Download and persist all ticker data. Returns the number of rows written."""
    tickers = list(tickers_df.index)
    new = [t for t in tickers if t not in last_dates]
    existing = [t for t in tickers if t in last_dates]

    # New tickers: full history; existing: incremental from last known date
    records = _download_and_prepare(
        new, tickers_df, {}, period="max"
    ) + _download_and_prepare(
        existing, tickers_df, last_dates,
        start=(
            str(min(last_dates[t] for t in existing)) if existing else None
        ),
    )
    df = _records_to_frame(records)
    written = _write_parquet_partitioned(df)
    if written and not df.empty:
        maxes = df.groupby("ticker")["date"].max().to_dict()
        for t, d in maxes.items():
            if last_dates.get(t) is None or d > last_dates[t]:
                last_dates[t] = d
        _save_last_dates(last_dates)
    return written


def _save_ticker_meta(tickers_df: pd.DataFrame) -> None:
    """Write the ticker→name mapping to TICKER_META_PATH (Parquet)."""
    (
        tickers_df.reset_index(names="ticker")[["ticker", "name"]]
        .astype({"ticker": str, "name": str})
        .sort_values("ticker")
        .reset_index(drop=True)
        .to_parquet(TICKER_META_PATH, index=False)
    )


def _load_ticker_meta() -> pd.DataFrame:
    """Read TICKER_META_PATH back as a ticker-indexed DataFrame."""
    return pd.read_parquet(TICKER_META_PATH).set_index("ticker")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(*, rebuild: bool = False) -> None:
    _ensure_dirs()

    if rebuild or not os.path.exists(TICKER_META_PATH):
        tickers_df = fetch_tickers()
        _save_ticker_meta(tickers_df)
        print(f"Saved {len(tickers_df)} tickers to {TICKER_META_PATH}")
    else:
        tickers_df = _load_ticker_meta()

    if rebuild:
        for p in (PARQUET_DIR, LAST_DATES_PATH):
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        os.makedirs(PARQUET_DIR, exist_ok=True)

    last_dates = {} if rebuild else _load_last_dates()
    total = ingest_all(tickers_df, last_dates)
    print(f"Done — {total} rows written to Parquet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    main(rebuild=parser.parse_args().rebuild)
