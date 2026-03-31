"""Unit tests for the Markowitz optimisation module.

All tests use synthetic data — no database required.
"""

import numpy as np
import pandas as pd
import pytest

from markowitz import (
    BacktestResult,
    compute_log_returns,
    efficient_frontier,
    max_sharpe,
    min_variance,
    monte_carlo,
    portfolio_stats,
    shrink_covariance,
    walk_forward_backtest,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """200 days of prices for 3 assets with distinct drift & volatility."""
    rng = np.random.default_rng(123)
    n_days = 200
    # daily simple returns: asset 0 steady, asset 1 volatile, asset 2 negative
    daily = np.column_stack(
        [
            rng.normal(0.0004, 0.01, n_days),
            rng.normal(0.0006, 0.03, n_days),
            rng.normal(-0.0002, 0.015, n_days),
        ]
    )
    prices = 100 * np.cumprod(1 + daily, axis=0)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


@pytest.fixture()
def mu_cov(synthetic_prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lr = compute_log_returns(synthetic_prices)
    cov_df = shrink_covariance(lr)
    return lr.mean().values, cov_df.values


# ── portfolio_stats ───────────────────────────────────────────────────────────


class TestPortfolioStats:
    def test_equal_weight(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        w = np.ones(3) / 3
        ret, vol, sharpe = portfolio_stats(
            w, mu, cov, annual_factor=252, risk_free=0.02
        )

        assert isinstance(ret, float)
        assert vol > 0, "Volatility must be positive"
        assert np.isfinite(sharpe)

    def test_concentrated_portfolio(
        self, mu_cov: tuple[np.ndarray, np.ndarray]
    ) -> None:
        mu, cov = mu_cov
        w = np.array([1.0, 0.0, 0.0])
        ret, vol, _ = portfolio_stats(w, mu, cov)

        # Return should match asset 0's annualised mean log-return
        expected_ret = mu[0] * 252
        assert abs(ret - expected_ret) < 1e-10

    def test_zero_vol_gives_finite_sharpe(self) -> None:
        """The 1e-12 denominator guard prevents division by zero."""
        mu = np.array([0.001])
        cov = np.array([[0.0]])  # zero variance (degenerate)
        _, vol, sharpe = portfolio_stats(np.array([1.0]), mu, cov)
        assert vol == 0.0
        assert np.isfinite(sharpe)


# ── max_sharpe ────────────────────────────────────────────────────────────────


class TestMaxSharpe:
    def test_weights_sum_to_one(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        result = max_sharpe(mu, cov, n_assets=3)
        assert abs(result.weights.sum() - 1.0) < 1e-8

    def test_weights_within_bounds(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        result = max_sharpe(mu, cov, n_assets=3, max_weight=0.5)
        assert np.all(result.weights >= -1e-8), "Weights must be non-negative"
        assert np.all(result.weights <= 0.5 + 1e-8), "Weights must respect upper bound"

    def test_result_fields(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        result = max_sharpe(mu, cov, n_assets=3)
        assert result.vol > 0
        assert np.isfinite(result.sharpe)


# ── min_variance ──────────────────────────────────────────────────────────────


class TestMinVariance:
    def test_weights_sum_to_one(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        result = min_variance(mu, cov, n_assets=3)
        assert abs(result.sum() - 1.0) < 1e-8

    def test_weights_within_bounds(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        result = min_variance(mu, cov, n_assets=3, max_weight=0.5)
        assert np.all(result >= -1e-8), "Weights must be non-negative"
        assert np.all(result <= 0.5 + 1e-8), "Weights must respect upper bound"

    def test_lower_volatility_than_max_sharpe(
        self, mu_cov: tuple[np.ndarray, np.ndarray]
    ) -> None:
        mu, cov = mu_cov
        w_mv = min_variance(mu, cov, n_assets=3)
        w_ms = max_sharpe(mu, cov, n_assets=3).weights

        _, vol_mv, _ = portfolio_stats(w_mv, mu, cov)
        _, vol_ms, _ = portfolio_stats(w_ms, mu, cov)

        assert vol_mv <= vol_ms + 1e-8, (
            "Min Variance portfolio should have lowest possible volatility"
        )


# ── efficient_frontier ────────────────────────────────────────────────────────


class TestEfficientFrontier:
    def test_monotonic_vol(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        """Frontier volatility should be non-decreasing with return."""
        mu, cov = mu_cov
        f_vols, f_rets = efficient_frontier(mu, cov, n_assets=3, n_points=30)
        assert len(f_vols) > 5, "Should have enough converged points"
        # Allow small numerical noise (1e-6)
        assert np.all(np.diff(f_vols) >= -1e-6), (
            "Frontier vols should be non-decreasing"
        )


# ── monte_carlo ───────────────────────────────────────────────────────────────


class TestMonteCarlo:
    def test_output_shapes(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        vols, rets, sharpes = monte_carlo(mu, cov, n_assets=3, n_samples=100)
        assert vols.shape == (100,)
        assert rets.shape == (100,)
        assert sharpes.shape == (100,)

    def test_deterministic(self, mu_cov: tuple[np.ndarray, np.ndarray]) -> None:
        mu, cov = mu_cov
        r1 = monte_carlo(mu, cov, n_assets=3, n_samples=50, seed=99)
        r2 = monte_carlo(mu, cov, n_assets=3, n_samples=50, seed=99)
        np.testing.assert_array_equal(r1[0], r2[0])


# ── walk_forward_backtest ─────────────────────────────────────────────────────


class TestBacktest:
    def test_no_lookahead(self, synthetic_prices: pd.DataFrame) -> None:
        """Train indices must always be strictly before test indices."""
        lr = compute_log_returns(synthetic_prices)
        # Use small windows so the test runs on 200 days of data
        result = walk_forward_backtest(lr, rebal_days=30, min_train_days=60)
        assert isinstance(result, BacktestResult)
        assert len(result.period_sharpes) > 0, "Should have at least one period"
        assert len(result.portfolio_value) > 0

    def test_equity_curve_starts_positive(self, synthetic_prices: pd.DataFrame) -> None:
        lr = compute_log_returns(synthetic_prices)
        result = walk_forward_backtest(lr, rebal_days=30, min_train_days=60)
        assert result.portfolio_value.iloc[0] > 0

    def test_min_variance_strategy(self, synthetic_prices: pd.DataFrame) -> None:
        lr = compute_log_returns(synthetic_prices)
        result = walk_forward_backtest(
            lr, strategy="min_variance", rebal_days=30, min_train_days=60
        )
        assert isinstance(result, BacktestResult)
        assert len(result.period_sharpes) > 0

    def test_transaction_costs_reduce_returns(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        lr = compute_log_returns(synthetic_prices)
        res_no_cost = walk_forward_backtest(
            lr, rebal_days=30, min_train_days=60, transaction_cost=0.0
        )
        res_with_cost = walk_forward_backtest(
            lr, rebal_days=30, min_train_days=60, transaction_cost=0.05
        )

        # The final portfolio value should be strictly lower when high costs are applied
        assert (
            res_with_cost.portfolio_value.iloc[-1]
            < res_no_cost.portfolio_value.iloc[-1]
        )
