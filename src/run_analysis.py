"""Run the full Markowitz portfolio analysis pipeline.

Usage:
    cd src/
    uv run python run_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from data import load_prices_parquet
from markowitz import (
    OptimResult,
    compute_log_returns,
    efficient_frontier,
    format_weights,
    max_sharpe,
    min_variance,
    monte_carlo,
    portfolio_stats,
    shrink_covariance,
    walk_forward_backtest,
)
from plots import plot_report


def _print_portfolio(name: str, result: OptimResult, prices, ticker_names) -> None:
    print(f"{name} portfolio:")
    print(
        format_weights(
            result.weights,
            prices.columns,
            ticker_names,
            threshold=cfg.WEIGHT_THRESHOLD,
        )
    )
    print(
        f"  Return {result.ret:.2%}  Vol {result.vol:.2%}  Sharpe {result.sharpe:.2f}\n"
    )


def main() -> None:
    prices, ticker_names = load_prices_parquet(
        parquet_dir=cfg.PARQUET_DIR,
        ticker_meta_path=cfg.TICKER_META_PATH,
        years=cfg.YEARS,
        annual_factor=cfg.ANNUAL_FACTOR,
        fill_ratio=cfg.MIN_DATA_FILL_RATIO,
    )
    n_assets = prices.shape[1]
    print(f"Universe: {n_assets} assets, {len(prices)} trading days")

    log_returns = compute_log_returns(prices)
    cov_df = shrink_covariance(log_returns)
    mu = log_returns.mean().values
    cov = cov_df.values

    tangency = max_sharpe(
        mu,
        cov,
        n_assets=n_assets,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
    )
    _print_portfolio("Tangency", tangency, prices, ticker_names)

    mv_weights = min_variance(
        mu,
        cov,
        n_assets=n_assets,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
    )
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(
        mv_weights,
        mu,
        cov,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
    )
    min_var_port = OptimResult(
        weights=mv_weights, ret=mv_ret, vol=mv_vol, sharpe=mv_sharpe
    )

    _print_portfolio("Minimum Variance", min_var_port, prices, ticker_names)

    mc_vols, mc_rets, mc_sharpes = monte_carlo(
        mu,
        cov,
        n_assets=n_assets,
        n_samples=cfg.MC_SAMPLES,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        seed=cfg.MC_SEED,
    )

    front_vols, front_rets = efficient_frontier(
        mu,
        cov,
        n_assets=n_assets,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
    )

    bt_kw = dict(
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        turnover_penalty=cfg.TURNOVER_PENALTY,
        rebal_days=cfg.REBAL_DAYS,
        min_train_days=cfg.MIN_TRAIN_DAYS,
        transaction_cost=cfg.TRANSACTION_COST,
    )

    bt_ms = walk_forward_backtest(log_returns, strategy="max_sharpe", **bt_kw)
    bt_mv = walk_forward_backtest(log_returns, strategy="min_variance", **bt_kw)

    print(
        f"Max Sharpe Backtest: {len(bt_ms.period_sharpes)} periods, "
        f"mean OOS Sharpe {np.nanmean(bt_ms.period_sharpes):.2f}"
    )
    print(
        f"Min Variance Backtest: {len(bt_mv.period_sharpes)} periods, "
        f"mean OOS Sharpe {np.nanmean(bt_mv.period_sharpes):.2f}"
    )

    plot_report(
        mc_vols=mc_vols,
        mc_returns=mc_rets,
        mc_sharpes=mc_sharpes,
        frontier_vols=front_vols,
        frontier_rets=front_rets,
        tangency=tangency,
        min_var=min_var_port,
        backtests={"Max Sharpe": bt_ms, "Min Variance": bt_mv},
        risk_free=cfg.RISK_FREE_ANNUAL,
        figures_dir=cfg.FIGURES_DIR,
    )
    plt.show()


if __name__ == "__main__":
    main()
