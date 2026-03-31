"""Run the full Markowitz portfolio analysis pipeline.

Usage:
    cd src/
    uv run python run_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from markowitz import (
    OptimResult,
    compute_log_returns,
    efficient_frontier,
    format_weights,
    load_prices,
    max_sharpe,
    min_variance,
    monte_carlo,
    portfolio_stats,
    shrink_covariance,
    walk_forward_backtest,
)
from plots import plot_frontier, plot_strategy_comparison


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────
    prices, ticker_names = load_prices(
        None,
        years=cfg.YEARS,
        annual_factor=cfg.ANNUAL_FACTOR,
        fill_ratio=cfg.MIN_DATA_FILL_RATIO,
        parquet_dir=cfg.PARQUET_DIR,
        ticker_meta_path=cfg.TICKER_META_PATH,
    )
    n_assets = prices.shape[1]
    print(f"Universe: {n_assets} assets, {len(prices)} trading days")

    # ── 2. Returns & covariance ───────────────────────────────────────────
    log_returns = compute_log_returns(prices)
    cov_df = shrink_covariance(log_returns)
    mu = log_returns.mean().values
    cov = cov_df.values

    # ── 3. Max-Sharpe optimisation ────────────────────────────────────────
    tangency = max_sharpe(
        mu,
        cov,
        n_assets=n_assets,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
    )
    print("Tangency portfolio:")
    print(
        format_weights(
            tangency.weights,
            prices.columns,
            ticker_names,
            threshold=cfg.WEIGHT_THRESHOLD,
        )
    )
    print(
        f"  Return {tangency.ret:.2%}  Vol {tangency.vol:.2%}  "
        f"Sharpe {tangency.sharpe:.2f}\n"
    )

    # ── 3.5 Min-Variance optimisation ──────────────────────────────────────
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

    print("Minimum Variance portfolio:")
    print(
        format_weights(
            mv_weights,
            prices.columns,
            ticker_names,
            threshold=cfg.WEIGHT_THRESHOLD,
        )
    )
    print(f"  Return {mv_ret:.2%}  Vol {mv_vol:.2%}  Sharpe {mv_sharpe:.2f}\n")

    # ── 4. Monte Carlo simulation ─────────────────────────────────────────
    mc_vols, mc_rets, mc_sharpes = monte_carlo(
        mu,
        cov,
        n_assets=n_assets,
        n_samples=cfg.MC_SAMPLES,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        seed=cfg.MC_SEED,
    )

    # ── 5. Efficient frontier ──────────────────────────────────────────────
    front_vols, front_rets = efficient_frontier(
        mu,
        cov,
        n_assets=n_assets,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
    )

    # ── 6. Frontier plot ──────────────────────────────────────────────────
    plot_frontier(
        mc_vols,
        mc_rets,
        mc_sharpes,
        front_vols,
        front_rets,
        tangency,
        min_var_port,
        risk_free=cfg.RISK_FREE_ANNUAL,
        figures_dir=cfg.FIGURES_DIR,
    )

    # ── 7. Walk-forward backtests ─────────────────────────────────────────
    bt_ms = walk_forward_backtest(
        log_returns,
        strategy="max_sharpe",
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        rebal_days=cfg.REBAL_DAYS,
        min_train_days=cfg.MIN_TRAIN_DAYS,
        transaction_cost=cfg.TRANSACTION_COST,
    )
    bt_mv = walk_forward_backtest(
        log_returns,
        strategy="min_variance",
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        rebal_days=cfg.REBAL_DAYS,
        min_train_days=cfg.MIN_TRAIN_DAYS,
        transaction_cost=cfg.TRANSACTION_COST,
    )

    print(
        f"Max Sharpe Backtest: {len(bt_ms.period_sharpes)} periods, "
        f"mean OOS Sharpe {np.nanmean(bt_ms.period_sharpes):.2f}"
    )
    print(
        f"Min Variance Backtest: {len(bt_mv.period_sharpes)} periods, "
        f"mean OOS Sharpe {np.nanmean(bt_mv.period_sharpes):.2f}"
    )

    # ── 8. Strategy comparison plot ───────────────────────────────────────
    plot_strategy_comparison(
        {"Max Sharpe": bt_ms, "Min Variance": bt_mv}, figures_dir=cfg.FIGURES_DIR
    )
    plt.show()


if __name__ == "__main__":
    main()
