"""Markowitz portfolio optimisation — transforms, solvers, and backtests."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class OptimResult:
    weights: np.ndarray
    ret: float  # annualised
    vol: float  # annualised
    sharpe: float


@dataclass(frozen=True)
class BacktestResult:
    portfolio_value: pd.Series  # cumulative equity curve
    oos_returns: pd.Series  # daily simple returns
    period_sharpes: list[float]  # one Sharpe per rebalance window
    rebal_dates: list[pd.Timestamp]


# ── Transforms ───────────────────────────────────────────────────────────────


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def shrink_covariance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage — better conditioned than the sample covariance."""
    lw = LedoitWolf().fit(log_returns.values)
    return pd.DataFrame(
        lw.covariance_, index=log_returns.columns, columns=log_returns.columns
    )


# ── Portfolio helpers ────────────────────────────────────────────────────────


def portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
) -> tuple[float, float, float]:
    """Return (annualised return, annualised vol, Sharpe) from daily log mu/cov."""
    ret = float(mu @ weights * annual_factor)
    vol = float(np.sqrt(weights @ cov @ weights * annual_factor))
    sharpe = (ret - risk_free) / (vol + 1e-12)  # guard against zero vol
    return ret, vol, sharpe


def risk_contributions(
    weights: np.ndarray, cov: np.ndarray, *, annual_factor: int = 252
) -> np.ndarray:
    """Marginal contribution to risk for each asset (sums to 1)."""
    sigma_p = np.sqrt(weights @ cov @ weights * annual_factor)
    if sigma_p < 1e-14:
        return np.full_like(weights, 1.0 / len(weights))
    mcr = weights * (cov @ weights * annual_factor) / sigma_p
    return mcr / mcr.sum()


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


def _optim_setup(n: int, max_weight: float = 1.0):
    """Equal-weight start, long-only bounds, and sum-to-one constraint."""
    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1}
    return x0, bounds, constraint


# ── Optimisation ─────────────────────────────────────────────────────────────


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    max_weight: float = 1.0,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> OptimResult:
    """Tangency portfolio via SLSQP. Minimises −Sharpe + λ·turnover."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)
    prev = np.zeros(n_assets) if prev_weights is None else prev_weights
    lam = float(turnover_penalty)

    def obj(w: np.ndarray) -> float:
        sharpe = portfolio_stats(
            w, mu, cov, annual_factor=annual_factor, risk_free=risk_free
        )[2]
        return float(-sharpe + lam * np.abs(w - prev).sum())

    res = minimize(obj, x0, method="SLSQP", constraints=[eq], bounds=bounds)
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
    risk_free: float = 0.0193,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> OptimResult:
    """Global minimum-variance portfolio (ignores expected returns)."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)
    prev = np.zeros(n_assets) if prev_weights is None else prev_weights
    lam = float(turnover_penalty)

    def obj(w: np.ndarray) -> float:
        vol = np.sqrt(w @ cov @ w * annual_factor)
        return float(vol + lam * np.abs(w - prev).sum())

    res = minimize(obj, x0, method="SLSQP", constraints=[eq], bounds=bounds)
    ret, vol, sharpe = portfolio_stats(
        res.x, mu, cov, annual_factor=annual_factor, risk_free=risk_free
    )
    return OptimResult(weights=res.x, ret=ret, vol=vol, sharpe=sharpe)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_assets: int,
    max_weight: float = 1.0,
    annual_factor: int = 252,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep target volatilities and maximise return at each level."""
    x0, bounds, eq = _optim_setup(n_assets, max_weight)

    mv = min_variance(
        mu, cov, n_assets=n_assets, max_weight=max_weight,
        annual_factor=annual_factor,
    )
    # Upper bound: max individual-asset vol (extends frontier past best-return corner)
    max_vol = float(
        np.sqrt(np.maximum(np.diag(cov), 0.0) * annual_factor).max()
    )

    pairs: list[tuple[float, float]] = []
    for target_vol in np.linspace(mv.vol, max_vol, n_points):
        tv2 = target_vol**2
        res = minimize(
            lambda w: -(mu @ w * annual_factor),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                eq,
                {
                    "type": "ineq",
                    "fun": lambda w, t=tv2: t - w @ cov @ w * annual_factor,
                },
            ],
        )
        if res.success:
            ret = float(mu @ res.x * annual_factor)
            vol = float(np.sqrt(res.x @ cov @ res.x * annual_factor))
            pairs.append((vol, ret))

    arr = np.array(pairs)
    return arr[:, 0], arr[:, 1]


# ── Walk-forward backtest ────────────────────────────────────────────────────


def _each_oos_period(
    log_returns: pd.DataFrame, rebal_days: int, min_train_days: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (train_end_idx, test_window) for each walk-forward step."""
    for rebal in range(min_train_days, len(log_returns), rebal_days):
        test_lr = log_returns.iloc[rebal : rebal + rebal_days]
        if test_lr.empty:
            break
        yield rebal, test_lr


def _period_sharpe(
    simple_returns: pd.Series, *, annual_factor: int, risk_free: float
) -> float:
    """Annualised Sharpe for one rebalance window (NaN if < 21 days)."""
    n = len(simple_returns)
    if n < 21:
        return float("nan")
    ann_ret = (1 + simple_returns).prod() ** (annual_factor / n) - 1
    ann_vol = simple_returns.std() * np.sqrt(annual_factor)
    return float((ann_ret - risk_free) / (ann_vol + 1e-12))


def walk_forward_backtest(
    log_returns: pd.DataFrame,
    *,
    strategy: str = "min_variance",
    max_weight: float = 1.0,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
    turnover_penalty: float = 0.0,
    rebal_days: int = 126,
    min_train_days: int = 504,
    transaction_cost: float = 0.0,
) -> BacktestResult:
    """Expanding-window backtest with optional turnover penalty and costs."""
    n_assets = log_returns.shape[1]
    prev_weights = np.zeros(n_assets)

    period_returns: list[pd.Series] = []
    rebal_dates: list[pd.Timestamp] = []

    solver = max_sharpe if strategy == "max_sharpe" else min_variance
    opt_kw = dict(
        n_assets=n_assets,
        max_weight=max_weight,
        annual_factor=annual_factor,
        turnover_penalty=turnover_penalty,
    )
    if strategy == "max_sharpe":
        opt_kw["risk_free"] = risk_free

    for rebal, test_lr in _each_oos_period(
        log_returns, rebal_days, min_train_days,
    ):
        train_lr = log_returns.iloc[:rebal]

        mu_wf = train_lr.mean().values
        cov_wf = LedoitWolf().fit(train_lr.values).covariance_
        weights = solver(
            mu_wf, cov_wf, prev_weights=prev_weights, **opt_kw
        ).weights

        # Log → simple returns for P&L, then deduct costs on day 1
        period_ret = (np.exp(test_lr) - 1) @ weights
        turnover = np.abs(weights - prev_weights).sum()
        period_ret.iloc[0] -= transaction_cost * turnover
        prev_weights = weights.copy()

        period_returns.append(period_ret)
        rebal_dates.append(test_lr.index[0])

    oos_returns = pd.concat(period_returns)
    portfolio_value = (1 + oos_returns).cumprod()
    period_sharpes = [
        _period_sharpe(pr, annual_factor=annual_factor, risk_free=risk_free)
        for pr in period_returns
    ]

    return BacktestResult(
        portfolio_value=portfolio_value,
        oos_returns=oos_returns,
        period_sharpes=period_sharpes,
        rebal_dates=rebal_dates,
    )


def equal_weight_backtest(
    log_returns: pd.DataFrame,
    *,
    annual_factor: int = 252,
    risk_free: float = 0.0193,
    rebal_days: int = 126,
    min_train_days: int = 504,
) -> BacktestResult:
    """1/N benchmark: equal weights, no optimisation, no costs."""
    n_assets = log_returns.shape[1]
    w = np.ones(n_assets) / n_assets

    period_returns: list[pd.Series] = []
    rebal_dates: list[pd.Timestamp] = []

    for _, test_lr in _each_oos_period(
        log_returns, rebal_days, min_train_days,
    ):
        period_returns.append((np.exp(test_lr) - 1) @ w)
        rebal_dates.append(test_lr.index[0])

    oos_returns = pd.concat(period_returns)
    portfolio_value = (1 + oos_returns).cumprod()
    period_sharpes = [
        _period_sharpe(pr, annual_factor=annual_factor, risk_free=risk_free)
        for pr in period_returns
    ]

    return BacktestResult(
        portfolio_value=portfolio_value,
        oos_returns=oos_returns,
        period_sharpes=period_sharpes,
        rebal_dates=rebal_dates,
    )
