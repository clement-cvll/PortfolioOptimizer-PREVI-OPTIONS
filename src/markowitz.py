"""Markowitz portfolio optimisation — data loading, transforms, and solvers."""

import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sqlalchemy import Engine

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptimResult:
    """Result of a portfolio optimisation."""

    weights: np.ndarray
    ret: float  # annualised return
    vol: float  # annualised volatility
    sharpe: float


@dataclass(frozen=True)
class BacktestResult:
    """Result of a walk-forward backtest."""

    portfolio_value: pd.Series
    oos_returns: pd.Series
    period_sharpes: list[float]
    rebal_dates: list[pd.Timestamp]


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_prices(
    engine: Engine | None,
    *,
    years: int,
    annual_factor: int = 252,
    fill_ratio: float = 0.94,
    parquet_dir: str | None = None,
    ticker_meta_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load close prices, pivot, and filter the universe.

    Returns (prices DataFrame, ticker→name Series).
    """
    df: pd.DataFrame
    ticker_names: pd.Series

    if parquet_dir and os.path.exists(parquet_dir):
        glob = os.path.join(parquet_dir, "**", "*.parquet")
        con = duckdb.connect(database=":memory:")
        try:
            max_date = con.execute(
                "SELECT max(date) FROM read_parquet(?, hive_partitioning=1)",
                [glob],
            ).fetchone()[0]
            if max_date is None:
                raise RuntimeError(f"No data found under parquet_dir={parquet_dir!r}")

            df = con.execute(
                """
                SELECT date, ticker, close
                FROM read_parquet(?, hive_partitioning=1)
                WHERE date >= (?::DATE - (? * INTERVAL '1 year'))
                """,
                [glob, max_date, years],
            ).df()
        finally:
            con.close()

        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).astype("category")
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)

        if not ticker_meta_path or not os.path.exists(ticker_meta_path):
            ticker_names = pd.Series(dtype=object)
        else:
            meta = pd.read_parquet(ticker_meta_path)
            ticker_names = meta.set_index("ticker")["name"]

    else:
        if engine is None:
            raise ValueError(
                "Provide parquet_dir (preferred) or pass a SQLAlchemy engine "
                "for legacy DB mode."
            )
        df = pd.read_sql(
            "SELECT date, close, ticker, name FROM opcvm_data ORDER BY date ASC",
            engine,
        )
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
        ticker_names = df.groupby("ticker")["name"].first()

    price_df = df.pivot(index="date", columns="ticker", values="close")
    min_obs = annual_factor * years
    prices = price_df.sort_index().tail(min_obs)
    prices = prices.dropna(axis=1, thresh=int(fill_ratio * min_obs)).dropna(axis=0)
    return prices, ticker_names


# ── Transforms ────────────────────────────────────────────────────────────────


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns, first NaN row dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def shrink_covariance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance, returned as a ticker-indexed DataFrame."""
    lw = LedoitWolf().fit(log_returns.values)
    return pd.DataFrame(
        lw.covariance_, index=log_returns.columns, columns=log_returns.columns
    )


# ── Portfolio helpers ─────────────────────────────────────────────────────────


def _optim_setup(n: int, max_weight: float = 1.0):
    """Initial guess, bounds, and sum-to-one constraint for n assets."""
    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1}
    return x0, bounds, constraint


def portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
) -> tuple[float, float, float]:
    """Annualised (return, volatility, Sharpe) using log-return convention."""
    ret = float(mu @ weights * annual_factor)
    vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(annual_factor))
    sharpe = (ret - risk_free) / (vol + 1e-12)
    return ret, vol, sharpe


def format_weights(
    weights: np.ndarray,
    tickers: pd.Index,
    ticker_names: pd.Series,
    *,
    threshold: float = 0.01,
) -> str:
    """Human-readable string of non-negligible weights."""
    return "\n".join(
        f"  {ticker_names.get(t, t)}: {w * 100:.2f}%"
        for t, w in zip(tickers, weights, strict=False)
        if w > threshold
    )


# ── Optimisation ──────────────────────────────────────────────────────────────


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    max_weight: float = 1.0,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
) -> OptimResult:
    """Tangency (max Sharpe) portfolio via SLSQP, long-only."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)
    res = minimize(
        lambda w: (
            -portfolio_stats(
                w, mu, cov, annual_factor=annual_factor, risk_free=risk_free
            )[2]
        ),
        x0,
        method="SLSQP",
        constraints=[eq],
        bounds=bounds,
    )
    ret, vol, sharpe = portfolio_stats(
        res.x, mu, cov, annual_factor=annual_factor, risk_free=risk_free
    )
    return OptimResult(weights=res.x, ret=ret, vol=vol, sharpe=sharpe)


def min_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    max_weight: float = 1.0,
    annual_factor: int = 252,
) -> np.ndarray:
    """Minimum-variance portfolio (ignoring expected returns)."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)
    res = minimize(
        lambda w: np.sqrt(w @ cov @ w) * np.sqrt(annual_factor),
        x0,
        method="SLSQP",
        constraints=[eq],
        bounds=bounds,
    )
    return res.x


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    max_weight: float = 1.0,
    annual_factor: int = 252,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Upper efficient frontier. Returns (vols, rets) arrays."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)

    mv_weights = min_variance(
        mu, cov, n_assets=n_assets, max_weight=max_weight, annual_factor=annual_factor
    )
    mv_ret = float(mu @ mv_weights * annual_factor)
    max_ret = float(mu.max() * annual_factor)

    pairs: list[tuple[float, float]] = []
    for target in np.linspace(mv_ret, max_ret, n_points):
        res = minimize(
            lambda w: np.sqrt(w @ cov @ w) * np.sqrt(annual_factor),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                eq,
                {"type": "eq", "fun": lambda w, t=target: mu @ w * annual_factor - t},
            ],
        )
        if res.success:
            vol = float(np.sqrt(res.x @ cov @ res.x) * np.sqrt(annual_factor))
            pairs.append((vol, target))

    arr = np.array(pairs)
    return arr[:, 0], arr[:, 1]


def monte_carlo(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    n_samples: int = 200_000,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Dirichlet portfolios. Returns (vols, returns, sharpes)."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(n_assets), size=n_samples)

    mc_returns = (W @ mu) * annual_factor
    mc_vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W)) * np.sqrt(annual_factor)
    mc_sharpes = np.where(mc_vols > 0, (mc_returns - risk_free) / mc_vols, -np.inf)
    return mc_vols, mc_returns, mc_sharpes


# ── Walk-Forward Backtest ─────────────────────────────────────────────────────


def walk_forward_backtest(
    log_returns: pd.DataFrame,
    *,
    strategy: str = "min_variance",
    max_weight: float = 1.0,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
    rebal_days: int = 126,
    min_train_days: int = 504,
    transaction_cost: float = 0.0,
) -> BacktestResult:
    """Expanding-window backtest. strategy can be 'max_sharpe' or 'min_variance'."""
    n_assets = log_returns.shape[1]
    x0, bounds, eq = _optim_setup(n_assets, max_weight)
    prev_weights = np.zeros(n_assets)

    period_returns: list[pd.Series] = []
    rebal_dates: list[pd.Timestamp] = []

    for rebal in range(min_train_days, len(log_returns), rebal_days):
        train_lr = log_returns.iloc[:rebal]
        test_lr = log_returns.iloc[rebal : rebal + rebal_days]
        if test_lr.empty:
            break

        mu_wf = train_lr.mean().values
        cov_wf = LedoitWolf().fit(train_lr.values).covariance_

        if strategy == "max_sharpe":

            def obj(
                w: np.ndarray, m: np.ndarray = mu_wf, c: np.ndarray = cov_wf
            ) -> float:
                num = m @ w * annual_factor - risk_free
                den = np.sqrt(w @ c @ w) * np.sqrt(annual_factor) + 1e-12
                return float(-(num / den))
        else:

            def obj(w: np.ndarray, c: np.ndarray = cov_wf) -> float:
                return float(np.sqrt(w @ c @ w) * np.sqrt(annual_factor))

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            constraints=[eq],
            bounds=bounds,
        )
        if not res.success:
            continue

        # log-returns → simple returns for P&L accumulation
        period_ret = (np.exp(test_lr) - 1) @ res.x

        # Deduct transaction cost (proportional to turnover) on first day
        turnover = np.abs(res.x - prev_weights).sum()
        period_ret.iloc[0] -= transaction_cost * turnover
        prev_weights = res.x.copy()

        period_returns.append(period_ret)
        rebal_dates.append(test_lr.index[0])

    oos_returns = pd.concat(period_returns)
    portfolio_value = (1 + oos_returns).cumprod()

    period_sharpes: list[float] = []
    for pr in period_returns:
        if len(pr) < 21:
            period_sharpes.append(np.nan)
        else:
            ann_ret = (1 + pr).prod() ** (annual_factor / len(pr)) - 1
            ann_vol = pr.std() * np.sqrt(annual_factor)
            period_sharpes.append((ann_ret - risk_free) / (ann_vol + 1e-12))

    return BacktestResult(
        portfolio_value=portfolio_value,
        oos_returns=oos_returns,
        period_sharpes=period_sharpes,
        rebal_dates=rebal_dates,
    )
