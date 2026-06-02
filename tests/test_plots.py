"""Smoke tests for plot_report figure output."""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from markowitz import BacktestResult, OptimResult, shrink_covariance
from plots import plot_report


@pytest.fixture()
def minimal_inputs():
    rng = np.random.default_rng(42)
    n, d = 120, 4
    lr = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n, d)),
        index=pd.bdate_range("2022-01-01", periods=n),
        columns=[f"F{i}" for i in range(d)],
    )
    cov_df = shrink_covariance(lr)
    mu = lr.mean().values
    cov = cov_df.values
    w = np.ones(d) / d
    ret, vol = float(mu @ w * 252), float(np.sqrt(w @ cov @ w * 252))
    tangency = OptimResult(weights=w, ret=ret, vol=vol, sharpe=ret / (vol + 1e-12))
    idx = lr.index[60:]
    oos = pd.Series(rng.normal(0.0002, 0.008, len(idx)), index=idx)
    bt = BacktestResult(
        portfolio_value=(1 + oos).cumprod(),
        oos_returns=oos,
        period_sharpes=[0.5],
        rebal_dates=[idx[0]],
    )
    return dict(
        frontier_vols=np.array([0.08, 0.10, 0.12]),
        frontier_rets=np.array([0.05, 0.06, 0.07]),
        tangency=tangency,
        min_var=tangency,
        backtests={"Bench": bt},
        cov_df=cov_df,
        risk_contribs={
            "Max Sharpe": (np.array([0.5, 0.3, 0.15, 0.05]), lr.columns, w),
            "Min variance": (np.array([0.25, 0.25, 0.25, 0.25]), lr.columns, w),
        },
        ticker_names=pd.Series({f"F{i}": f"Fund {i}" for i in range(d)}),
        risk_free=0.02,
    )


def test_plot_report_writes_four_pngs(tmp_path, minimal_inputs):
    filenames = {
        "efficient_frontier": "efficient_frontier.png",
        "oos_equity": "oos_equity.png",
        "correlation": "correlation.png",
        "risk_contributions": "risk_contributions.png",
    }
    saved = plot_report(
        **minimal_inputs,
        figures_dir=str(tmp_path),
        figure_filenames=filenames,
    )
    assert set(saved.keys()) == set(filenames.keys())
    for path in saved.values():
        assert path.endswith(".png")
        assert os.path.isfile(path)
