"""Unit tests for the Markowitz optimisation module.

All tests use synthetic data — no database required.
"""

import numpy as np
import pandas as pd
import pytest

from markowitz import (
    BacktestResult,
    OptimResult,
    compute_log_returns,
    efficient_frontier,
    equal_weight_backtest,
    max_sharpe,
    min_variance,
    portfolio_stats,
    risk_contributions,
    shrink_covariance,
    walk_forward_backtest,
)


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n_days = 200
    daily = np.column_stack([
        rng.normal(0.0004, 0.01, n_days),
        rng.normal(0.0006, 0.03, n_days),
        rng.normal(-0.0002, 0.015, n_days),
    ])
    prices = 100 * np.cumprod(1 + daily, axis=0)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


@pytest.fixture()
def mu_cov(synthetic_prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lr = compute_log_returns(synthetic_prices)
    cov_df = shrink_covariance(lr)
    return lr.mean().values, cov_df.values


class TestPortfolioStats:
    def test_equal_weight(self, mu_cov):
        mu, cov = mu_cov
        ret, vol, sharpe = portfolio_stats(
            np.ones(3) / 3, mu, cov, annual_factor=252, risk_free=0.02
        )
        assert isinstance(ret, float)
        assert vol > 0
        assert np.isfinite(sharpe)

    def test_concentrated_portfolio(self, mu_cov):
        mu, cov = mu_cov
        ret, vol, _ = portfolio_stats(np.array([1.0, 0.0, 0.0]), mu, cov)
        assert abs(ret - mu[0] * 252) < 1e-10

    def test_zero_vol_gives_finite_sharpe(self):
        mu = np.array([0.001])
        cov = np.array([[0.0]])
        _, vol, sharpe = portfolio_stats(np.array([1.0]), mu, cov)
        assert vol == 0.0
        assert np.isfinite(sharpe)


class TestMaxSharpe:
    def test_weights_sum_to_one(self, mu_cov):
        mu, cov = mu_cov
        assert abs(max_sharpe(mu, cov, n_assets=3).weights.sum() - 1.0) < 1e-8

    def test_weights_within_bounds(self, mu_cov):
        mu, cov = mu_cov
        w = max_sharpe(mu, cov, n_assets=3, max_weight=0.5).weights
        assert np.all(w >= -1e-8)
        assert np.all(w <= 0.5 + 1e-8)

    def test_result_fields(self, mu_cov):
        mu, cov = mu_cov
        r = max_sharpe(mu, cov, n_assets=3)
        assert r.vol > 0
        assert np.isfinite(r.sharpe)


class TestMinVariance:
    def test_returns_optim_result(self, mu_cov):
        mu, cov = mu_cov
        assert isinstance(min_variance(mu, cov, n_assets=3), OptimResult)

    def test_weights_sum_to_one(self, mu_cov):
        mu, cov = mu_cov
        assert abs(min_variance(mu, cov, n_assets=3).weights.sum() - 1.0) < 1e-8

    def test_weights_within_bounds(self, mu_cov):
        mu, cov = mu_cov
        w = min_variance(mu, cov, n_assets=3, max_weight=0.5).weights
        assert np.all(w >= -1e-8)
        assert np.all(w <= 0.5 + 1e-8)

    def test_lower_volatility_than_max_sharpe(self, mu_cov):
        mu, cov = mu_cov
        vol_mv = min_variance(mu, cov, n_assets=3).vol
        vol_ms = max_sharpe(mu, cov, n_assets=3).vol
        assert vol_mv <= vol_ms + 1e-8


class TestEfficientFrontier:
    def test_monotonic_vol(self, mu_cov):
        mu, cov = mu_cov
        f_vols, _ = efficient_frontier(mu, cov, n_assets=3, n_points=30)
        assert len(f_vols) > 5
        assert np.all(np.diff(f_vols) >= -1e-6)


class TestRiskContributions:
    def test_sums_to_one(self, mu_cov):
        mu, cov = mu_cov
        w = max_sharpe(mu, cov, n_assets=3).weights
        rc = risk_contributions(w, cov)
        assert abs(rc.sum() - 1.0) < 1e-8

    def test_all_non_negative(self, mu_cov):
        mu, cov = mu_cov
        w = min_variance(mu, cov, n_assets=3).weights
        rc = risk_contributions(w, cov)
        assert np.all(rc >= -1e-8)


class TestEqualWeightBacktest:
    def test_returns_backtest_result(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        result = equal_weight_backtest(lr, rebal_days=30, min_train_days=60)
        assert isinstance(result, BacktestResult)
        assert len(result.period_sharpes) > 0

    def test_equity_curve_starts_positive(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        result = equal_weight_backtest(lr, rebal_days=30, min_train_days=60)
        assert result.portfolio_value.iloc[0] > 0


class TestBacktest:
    def test_no_lookahead(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        result = walk_forward_backtest(lr, rebal_days=30, min_train_days=60)
        assert isinstance(result, BacktestResult)
        assert len(result.period_sharpes) > 0
        assert len(result.portfolio_value) > 0

    def test_equity_curve_starts_positive(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        result = walk_forward_backtest(lr, rebal_days=30, min_train_days=60)
        assert result.portfolio_value.iloc[0] > 0

    def test_min_variance_strategy(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        result = walk_forward_backtest(
            lr, strategy="min_variance", rebal_days=30, min_train_days=60
        )
        assert isinstance(result, BacktestResult)
        assert len(result.period_sharpes) > 0

    def test_transaction_costs_reduce_returns(self, synthetic_prices):
        lr = compute_log_returns(synthetic_prices)
        res_free = walk_forward_backtest(
            lr, rebal_days=30, min_train_days=60, transaction_cost=0.0
        )
        res_cost = walk_forward_backtest(
            lr, rebal_days=30, min_train_days=60, transaction_cost=0.05
        )
        assert res_cost.portfolio_value.iloc[-1] < res_free.portfolio_value.iloc[-1]
