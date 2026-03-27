"""build_database.py — Scrape OPCVM tickers and populate the TimescaleDB database.

Usage:
    python build_database.py           # incremental update
    python build_database.py --rebuild # re-scrape tickers, drop table, rebuild from scratch
"""

import os
import sys
import requests
import argparse
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DB_URL, TICKERS_CSV

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


def init_db(engine, rebuild: bool = False) -> None:
    """Create the opcvm_data table (and hypertable) if it doesn't exist."""
    with engine.begin() as conn:
        if rebuild:
            conn.execute(text("DROP TABLE IF EXISTS opcvm_data"))
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS opcvm_data (
                date DATE, open FLOAT, high FLOAT, low FLOAT,
                close FLOAT, ticker TEXT, name TEXT
            )""")
        )
        conn.execute(
            text(
                "SELECT create_hypertable('opcvm_data', 'date', if_not_exists => TRUE)"
            )
        )


def get_last_dates(engine) -> dict:
    """Return {ticker: max_date} for all tickers in the database."""
    with engine.connect() as conn:
        return {
            r[0]: r[1]
            for r in conn.execute(
                text("SELECT ticker, MAX(date) FROM opcvm_data GROUP BY ticker")
            )
        }


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


def _bulk_insert(engine, records: list[tuple]) -> None:
    """Insert records into opcvm_data using psycopg2 execute_values."""
    if not records:
        return
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO opcvm_data (date, open, high, low, close, ticker, name) VALUES %s",
                records,
                page_size=2000,
            )
        conn.commit()
    finally:
        conn.close()


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


def ingest_all(engine, tickers_df: pd.DataFrame, last_dates: dict) -> int:
    """Download and insert all ticker data. Returns the number of rows inserted."""
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
    _bulk_insert(engine, records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-scrape tickers, drop the table, and rebuild from scratch",
    )
    args = parser.parse_args()
    engine = create_engine(DB_URL)

    if args.rebuild or not os.path.exists(TICKERS_CSV):
        tickers_df = fetch_tickers()
        tickers_df.to_csv(TICKERS_CSV, index=True, index_label="ticker")
        print(f"Saved {len(tickers_df)} tickers to {TICKERS_CSV}")
    else:
        tickers_df = pd.read_csv(TICKERS_CSV, index_col=0)

    init_db(engine, rebuild=args.rebuild)
    last_dates = {} if args.rebuild else get_last_dates(engine)
    total = ingest_all(engine, tickers_df, last_dates)
    print(f"Done — {total} rows inserted.")


if __name__ == "__main__":
    main()
