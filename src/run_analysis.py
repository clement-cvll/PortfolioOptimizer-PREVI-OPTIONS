"""Markowitz analysis: load data → optimise → backtest → report."""

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from data import load_prices_parquet
from markowitz import (
    compute_log_returns,
    efficient_frontier,
    equal_weight_backtest,
    format_weights,
    max_sharpe,
    min_variance,
    risk_contributions,
    shrink_covariance,
    walk_forward_backtest,
)
from plots import plot_report


def _print_portfolio(name: str, result, prices, ticker_names) -> None:
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
        f"  Return {result.ret:.2%}  Vol {result.vol:.2%}  "
        f"Sharpe {result.sharpe:.2f}\n"
    )


def main() -> None:
    prices, ticker_names = load_prices_parquet(
        parquet_dir=cfg.PARQUET_DIR,
        ticker_meta_path=cfg.TICKER_META_PATH,
        years=cfg.YEARS,
        annual_factor=cfg.ANNUAL_FACTOR,
        fill_ratio=cfg.MIN_DATA_FILL_RATIO,
    )
    n = prices.shape[1]
    print(f"Universe: {n} assets, {len(prices)} trading days")
    if n == 0:
        raise RuntimeError(
            "No assets survived filtering. "
            "Lower MIN_DATA_FILL_RATIO or shorten YEARS in config.py."
        )

    lr = compute_log_returns(prices)
    mu, cov_df = lr.mean().values, shrink_covariance(lr)
    cov = cov_df.values

    opt = dict(
        n_assets=n,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
    )
    tangency, min_var = max_sharpe(mu, cov, **opt), min_variance(mu, cov, **opt)
    _print_portfolio("Tangency", tangency, prices, ticker_names)
    _print_portfolio("Minimum Variance", min_var, prices, ticker_names)

    front_vols, front_rets = efficient_frontier(
        mu,
        cov,
        n_assets=n,
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
    )

    rc = dict(cov=cov, annual_factor=cfg.ANNUAL_FACTOR)
    risk_contribs = {
        "Tangency": (risk_contributions(tangency.weights, **rc), prices.columns),
        "Min Var": (risk_contributions(min_var.weights, **rc), prices.columns),
    }

    bt_kw = dict(
        max_weight=cfg.MAX_WEIGHT,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        turnover_penalty=cfg.TURNOVER_PENALTY,
        rebal_days=cfg.REBAL_DAYS,
        min_train_days=cfg.MIN_TRAIN_DAYS,
        transaction_cost=cfg.TRANSACTION_COST,
    )
    bt_ms = walk_forward_backtest(lr, strategy="max_sharpe", **bt_kw)
    bt_mv = walk_forward_backtest(lr, strategy="min_variance", **bt_kw)
    bt_ew = equal_weight_backtest(
        lr,
        annual_factor=cfg.ANNUAL_FACTOR,
        risk_free=cfg.RISK_FREE_ANNUAL,
        rebal_days=cfg.REBAL_DAYS,
        min_train_days=cfg.MIN_TRAIN_DAYS,
    )

    for label, bt in (
        ("Max Sharpe", bt_ms),
        ("Min Variance", bt_mv),
        ("Equal Weight", bt_ew),
    ):
        m = np.nanmean(bt.period_sharpes)
        print(
            f"{label} Backtest: {len(bt.period_sharpes)} periods, "
            f"mean OOS Sharpe {m:.2f}"
        )

    plot_report(
        frontier_vols=front_vols,
        frontier_rets=front_rets,
        tangency=tangency,
        min_var=min_var,
        backtests={
            "Max Sharpe": bt_ms,
            "Min Variance": bt_mv,
            "Equal Weight (1/N)": bt_ew,
        },
        cov_df=cov_df,
        risk_contribs=risk_contribs,
        ticker_names=ticker_names,
        risk_free=cfg.RISK_FREE_ANNUAL,
        annual_factor=cfg.ANNUAL_FACTOR,
        figures_dir=cfg.FIGURES_DIR,
    )
    plt.show()


if __name__ == "__main__":
    main()
