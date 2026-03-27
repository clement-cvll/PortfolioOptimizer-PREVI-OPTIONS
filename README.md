# Portfolio Optimizer - PREVI-OPTIONS

A Python portfolio optimisation tool based on Modern Portfolio Theory (Markowitz MPT). Fetches historical OPCVM fund data from a TimescaleDB database, estimates optimal portfolio weights (Max Sharpe and Min Variance), and validates the strategies with a walk-forward out-of-sample backtest.

## Features

*   **Data Ingestion:** Scrapes available OPCVM tickers from Previ-Options, resolves them via `yfinance`, and stores OHLC history in TimescaleDB.
*   **Data Preparation:**
    *   Filters the universe to a configurable historical window (default: 5-10 years).
    *   Drops assets with insufficient data coverage.
    *   Computes daily log-returns and a Ledoit-Wolf shrinkage covariance matrix.
*   **Markowitz Optimisation:**
    *   Maximises the Sharpe ratio (Tangency portfolio) and minimises portfolio variance.
    *   Monte Carlo simulation for frontier visualisation.
*   **Walk-Forward Backtest:**
    *   Expanding-window re-optimisation with proportional transaction costs.
    *   Compares Max Sharpe vs Min Variance strategies out-of-sample.
*   **Visualisation:**
    *   Efficient frontier with Capital Market Line and Tangency/Min-Var points.
    *   Strategy comparison module showing multiple equity curves and grouped Sharpe bar charts.

## Results

![Portfolio frontier](https://github.com/clement-cvll/PortfolioOptimizer-PREVI-OPTIONS/blob/main/src/figures/markowitz_portfolio_frontier.png)

![Strategy Comparison](https://github.com/clement-cvll/PortfolioOptimizer-PREVI-OPTIONS/blob/main/src/figures/strategy_comparison.png)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/clement-cvll/PortfolioOptimizer-PREVI-OPTIONS.git
    cd PortfolioOptimizer-PREVI-OPTIONS
    ```
2.  **Set up the project:**
    ```bash
    uv sync
    ```
3.  **Database Setup:**
    *   **Start TimescaleDB (PostgreSQL) using Docker:**
        ```bash
        sh src/database/init.sh
        sh src/database/start.sh
        ```
        This starts a PostgreSQL instance with TimescaleDB on port `5432` (user `postgres`, password `password`).
    *   **Populate the database:**
        ```bash
        cd src
        uv run python build_database.py          # incremental update
        uv run python build_database.py --rebuild # full re-scrape
        ```

## Usage

```bash
cd src
uv run python run_analysis.py
```

This runs the full pipeline: data loading → optimisation → backtest → plots saved to `src/figures/`.

## Project Structure

```
.
├── pyproject.toml       # Python dependencies (managed by uv)
├── README.md            # This documentation file
├── src/
│   ├── config.py             # Centralised constants and paths
│   ├── markowitz.py          # Data loading, transforms, optimisation, backtest
│   ├── plots.py              # Frontier and backtest visualisation
│   ├── run_analysis.py       # Main CLI pipeline entrypoint
│   ├── build_database.py     # Ticker scraping and database population
│   ├── tickers.csv           # Cached ticker list
│   ├── database/             # PostgreSQL/TimescaleDB setup scripts
│   │   ├── init.sh           # Docker image pull
│   │   └── start.sh          # Docker container start
│   └── figures/              # Generated plot outputs (gitignored)
└── tests/
    └── test_markowitz.py     # Unit tests (synthetic data, no DB needed)
```

## Key Parameters

All configurable in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `YEARS` | 6 | Historical data window (years) |
| `MIN_DATA_FILL_RATIO` | 0.95 | Minimum fraction of data coverage |
| `MAX_WEIGHT` | 1.0 | Maximum weight per asset |
| `RISK_FREE_ANNUAL` | 1.93% | Risk-free rate (€STR) |
| `MC_SAMPLES` | 200,000 | Monte Carlo portfolio samples |
| `REBAL_DAYS` | 126 | Trading days between rebalances |
| `MIN_TRAIN_DAYS` | 504 | Minimum training window (~2 years) |
| `TRANSACTION_COST` | 1.0% | Proportional cost on portfolio turnover |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
