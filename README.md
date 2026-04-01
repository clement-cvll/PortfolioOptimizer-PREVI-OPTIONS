# Portfolio Optimizer — PREVI-OPTIONS

This is a **personal project**: a small Python playground around Markowitz-style portfolio optimisation on OPCVM funds listed in Previ-Options. I built it to learn and experiment with real data.

It pulls historical prices into a **local Parquet** store (read with **DuckDB**), fits **max Sharpe** and **min variance** portfolios, and tests them with an **out-of-sample** walk-forward backtest. Everything runs locally.

## What it does

- **Data:** Scrapes the Previ-Options universe, resolves tickers with `yfinance`, and saves partitioned Parquet under `src/data/` (metadata and incremental updates included).
- **Prep:** You can easily adjust how much price history to include, filter out assets with too little data, calculate log-returns, and apply a Ledoit–Wolf covariance shrinkage. All of these options are configurable in [`src/config.py`](src/config.py).
- **Optimisation:** Tangency (max Sharpe) and minimum variance, plus a Monte Carlo cloud for the frontier chart.
- **Backtest:** Expanding window, with optional **turnover penalty** and **transaction costs** so results aren’t totally naive.
- **Plots:** One combined figure (`portfolio_report.png`): frontier, equity curves, and per-period Sharpe bars.

![Efficient frontier, OOS equity curves, metrics, and per-period Sharpe bars](src/figures/portfolio_report.png)

## Quick start (everything in one go)

From the **repository root** (after [uv](https://docs.astral.sh/uv/) is installed):

```bash
git clone https://github.com/clement-cvll/PortfolioOptimizer-PREVI-OPTIONS.git
cd PortfolioOptimizer-PREVI-OPTIONS
uv sync
uv run main.py
```

That installs dependencies, **updates or builds** the local Parquet dataset (network needed the first time), then runs optimisation, backtests, and saves figures. Equivalent commands: `uv run python main.py` or `uv run previ-options`.

**Flags:**

| Command | What it does |
|---------|----------------|
| `uv run main.py` | Incremental data update (if you already have data), then full analysis |
| `uv run main.py --rebuild` | Re-scrape tickers and rebuild Parquet from scratch, then analysis |

**Outputs:** `src/figures/portfolio_report.png` and a short summary in the terminal.

## Configuration

Everything lives in [`src/config.py`](src/config.py): where data and figures go (`src/data/`, `src/figures/` by default), risk-free rate, rebalance cadence, Monte Carlo size, and so on. Edit the file directly—no extra env setup required.

### Key parameters (defaults in `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YEARS` | 6 | History window (years of trading days) |
| `MIN_DATA_FILL_RATIO` | 0.95 | Minimum coverage per asset |
| `MAX_WEIGHT` | 1.0 | Upper bound per asset |
| `RISK_FREE_ANNUAL` | 1.93% | Risk-free (€STR) |
| `MC_SAMPLES` | 500,000 | Monte Carlo samples for frontier cloud |
| `REBAL_DAYS` | 126 | Days between rebalances |
| `MIN_TRAIN_DAYS` | 504 | Minimum training length before first rebalance |
| `TRANSACTION_COST` | 1.0% | Cost applied to turnover at rebalance |
| `TURNOVER_PENALTY` | 0.0 | Soft penalty on L1 turnover vs previous weights in the optimiser |

## Project layout

```
.
├── assets/
│   └── portfolio_report.png   # Sample figure for this README
├── main.py                 # Root entry: ingest + run_analysis
├── pyproject.toml
├── README.md
├── src/
│   ├── config.py
│   ├── data.py             # DuckDB + Parquet price load
│   ├── markowitz.py        # Returns, optimisation, walk-forward backtest
│   ├── plots.py            # Combined report figure
│   ├── run_analysis.py     # Pipeline CLI
│   ├── build_database.py   # Scrape + Parquet ingest
│   ├── tickers.csv         # Cached universe (optional)
│   ├── data/               # Generated Parquet + metadata (gitignored)
│   └── figures/            # Generated plots (gitignored)
└── tests/
    ├── test_markowitz.py
    ├── test_data_layer.py
    └── test_turnover_penalty.py
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
