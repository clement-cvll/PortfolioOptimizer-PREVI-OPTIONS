import numpy as np
import pandas as pd

from markowitz import compute_log_returns, walk_forward_backtest


def _synthetic_prices(n_days: int = 260, n_assets: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    daily = rng.normal(0.0002, 0.015, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + daily, axis=0)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def test_turnover_penalty_changes_backtest_path() -> None:
    prices = _synthetic_prices()
    lr = compute_log_returns(prices)

    res0 = walk_forward_backtest(
        lr,
        strategy="max_sharpe",
        rebal_days=21,
        min_train_days=63,
        transaction_cost=0.0,
        turnover_penalty=0.0,
    )
    res1 = walk_forward_backtest(
        lr,
        strategy="max_sharpe",
        rebal_days=21,
        min_train_days=63,
        transaction_cost=0.0,
        turnover_penalty=0.5,
    )

    # With a penalty, the optimized path should generally change.
    assert not res0.portfolio_value.equals(res1.portfolio_value)

    # Sanity: both runs produce finite equity curves.
    assert np.isfinite(res0.portfolio_value.iloc[-1])
    assert np.isfinite(res1.portfolio_value.iloc[-1])
