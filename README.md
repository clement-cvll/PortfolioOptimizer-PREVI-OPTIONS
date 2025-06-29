# Portfolio Optimizer - PREVI-OPTIONS

This repository contains a Python-based portfolio optimization tool leveraging the Markowitz Modern Portfolio Theory (MPT). It connects to a PostgreSQL database to fetch historical asset data, performs data cleaning and transformation, and then optimizes portfolio weights to maximize the Sharpe Ratio under specified constraints.

## Features

*   **Data Ingestion:** Connects to a PostgreSQL database to retrieve historical `close` prices and `ticker` information for various assets.
*   **Data Preparation:**
    *   Filters data to a specified historical period (e.g., last X years).
    *   Handles missing values by dropping assets or dates with insufficient data.
    *   Calculates daily returns, mean returns, and the covariance matrix.
    *   Applies regularization to the covariance matrix for numerical stability.
*   **Markowitz Optimization:**
    *   Implements the Markowitz MPT to find optimal asset weights.
    *   Maximizes the Sharpe Ratio (risk-adjusted return).
    *   Supports "long-only" constraints (weights between 0 and 1) and a sum-to-one constraint.
    *   Filters out assets with insignificant optimal weights (e.g., < 1%).
*   **Performance Analysis:**
    *   Calculates and displays the portfolio's annualized geometric return over the specified period.
    *   Computes the portfolio's annualized Sharpe Ratio, annualized return, and volatility.
*   **Visualization:** Plots the normalized portfolio value over time and displays asset allocation in a pie chart.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PortfolioOptimizer-PREVI-OPTIONS.git
    cd PortfolioOptimizer-PREVI-OPTIONS
    ```
2.  **Set up a uv project:**
    ```bash
    uv sync
    ```
4.  **Database Setup:**
    *   **Start TimescaleDB (PostgreSQL) using Docker:**
        ```bash
        # Pull the TimescaleDB Docker image (if not already pulled)
        ./src/database/init.sh
        # Start the Docker container
        ./src/database/start.sh
        ```
        This will start a PostgreSQL instance with TimescaleDB extension on port `5432` with user `postgres` and password `password`.
    *   **Populate the database:**
        Run the `fetch_tickers.py` and `build_database.py` script to create the `opcvm_data` table and insert historical asset data (from `tickers.csv` via `yfinance`).
        ```bash
        cd src
        uv run fetch_tickers.py
        uv run build_database.py
        ```

## Usage

The core logic is implemented in a Jupyter Notebook, typically `src/notebook/1-markowitz.ipynb`.

The notebook will guide you through data loading, cleaning, optimization, and performance analysis.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
