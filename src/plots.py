"""Visualisation helpers for the Markowitz portfolio optimiser."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import leaves_list, linkage

from markowitz import BacktestResult, OptimResult

_COLORS = {
    "tangency": "#e63946",
    "minvar": "#2196F3",
    "equal": "#4CAF50",
    "cml": "#e63946",
    "frontier": "#1a1a2e",
}
_FRONTIER_XLIM = (0.0, 0.20)  # vol axis (fraction → %)
_FRONTIER_YLIM = (0.0, 0.18)  # return axis
_REPORT_FIGSIZE = (16, 10)
_GS_MAIN = dict(hspace=0.34, wspace=0.42)
_GS_CORR = dict(width_ratios=[0.68, 0.32], wspace=0.02)


# ── Style ─────────────────────────────────────────────────────────────────────


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titlepad": 8,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── OOS metrics ──────────────────────────────────────────────────────────────


def _oos_metrics(
    bt: BacktestResult, *, annual_factor: int, risk_free: float
) -> dict[str, float]:
    r = bt.oos_returns
    n = len(r)
    nan = float("nan")
    if n == 0:
        return dict(ann_ret=nan, ann_vol=nan, sharpe=nan, max_dd=nan)
    total_ret = float((1 + r).prod() - 1)
    ann_ret = (1 + total_ret) ** (annual_factor / n) - 1
    ann_vol = float(r.std(ddof=1) * np.sqrt(annual_factor)) if n > 1 else nan
    sharpe = (
        (ann_ret - risk_free) / (ann_vol + 1e-12)
        if np.isfinite(ann_vol) else nan
    )
    max_dd = float((bt.portfolio_value / bt.portfolio_value.cummax() - 1).min())
    return dict(
        ann_ret=float(ann_ret), ann_vol=ann_vol, sharpe=float(sharpe), max_dd=max_dd,
    )


# ── Per-axis helpers ─────────────────────────────────────────────────────────


def _display_name(ticker: str, ticker_names: pd.Series) -> str:
    return str(ticker_names.get(ticker, ticker))


def _plot_frontier(
    ax,
    *,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None,
    risk_free: float,
):
    """Efficient frontier line with CML, tangency, and min-var markers."""
    ax.plot(
        frontier_vols, frontier_rets,
        color=_COLORS["frontier"], linewidth=2.5, zorder=4,
        label="Efficient frontier",
    )

    x_hi = _FRONTIER_XLIM[1]
    cml_x = np.linspace(0.0, x_hi, 50)
    ax.plot(
        cml_x, risk_free + tangency.sharpe * cml_x,
        color=_COLORS["cml"], linewidth=1.8, linestyle="--",
        zorder=4, label="Capital Market Line",
    )
    ax.scatter(0, risk_free, color=_COLORS["cml"], s=55, zorder=6)
    ax.scatter(
        tangency.vol, tangency.ret,
        color=_COLORS["tangency"], s=240, marker="*", zorder=6,
        label=f"Tangency (SR {tangency.sharpe:.2f})",
    )

    if min_var:
        ax.scatter(
            min_var.vol, min_var.ret,
            color=_COLORS["minvar"], s=160, marker="o",
            edgecolor="white", zorder=6, label="Min Var",
        )

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set(xlabel="Annualised Volatility", ylabel="Annualised Return")
    ax.set_xlim(*_FRONTIER_XLIM)
    ax.set_ylim(*_FRONTIER_YLIM)
    ax.set_title("Efficient Frontier")
    ax.legend(framealpha=0.95, loc="upper left")


def _plot_equity(
    ax,
    backtests: dict[str, BacktestResult],
    *,
    annual_factor: int,
    risk_free: float,
):
    """OOS equity curves with key stats in the legend."""
    if not backtests:
        ax.set_axis_off()
        ax.text(
            0.5, 0.5, "No walk-forward backtests",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
        )
        return

    color_cycle = [
        _COLORS["tangency"], _COLORS["minvar"], _COLORS["equal"],
    ]
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.4)

    for i, (name, bt) in enumerate(backtests.items()):
        m = _oos_metrics(
            bt, annual_factor=annual_factor, risk_free=risk_free
        )
        label = (
            f"{name}  SR {m['sharpe']:.2f}  "
            f"DD {m['max_dd']:.0%}"
        )
        c = color_cycle[i % len(color_cycle)]
        ax.plot(
            bt.portfolio_value, color=c, linewidth=2,
            zorder=3, label=label,
        )

    ax.set_title("Out-of-Sample Equity Curves")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8)


def _plot_correlation(
    ax, cov_df: pd.DataFrame, ticker_names: pd.Series,
):
    """Clustered Ledoit-Wolf correlation heatmap (labels = fund names)."""
    std = np.sqrt(np.diag(cov_df.values))
    std[std == 0] = 1.0
    corr = cov_df.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    Z = linkage(1 - corr, method="ward")
    order = leaves_list(Z)
    corr = corr[np.ix_(order, order)]
    tickers = cov_df.columns[order]
    labels = [_display_name(str(t), ticker_names) for t in tickers]

    im = ax.imshow(
        corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    n = len(labels)
    if n <= 25:
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_title("Correlation (Ledoit-Wolf, clustered)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.2%", pad=0.06)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    ax.set_anchor("W")


def _plot_risk_contributions(
    ax,
    risk_contribs: dict[str, tuple[np.ndarray, pd.Index]],
    ticker_names: pd.Series,
):
    """Side-by-side horizontal bar chart of marginal risk contributions."""
    if not risk_contribs:
        ax.set_axis_off()
        return

    name_list = list(risk_contribs.keys())
    colors = [_COLORS.get("tangency"), _COLORS.get("minvar")]
    rc0, tickers0 = risk_contribs[name_list[0]]
    order = np.argsort(rc0)[::-1]
    top = min(15, len(order))
    idx = order[:top]

    y = np.arange(top)
    bar_h = 0.35

    for k, name in enumerate(name_list):
        rc, tickers = risk_contribs[name]
        ax.barh(
            y + k * bar_h, rc[idx], height=bar_h,
            color=colors[k % len(colors)], alpha=0.85,
            label=name,
        )

    ax.set_yticks(y + bar_h * (len(name_list) - 1) / 2)
    ax.set_yticklabels(
        [_display_name(str(t), ticker_names) for t in tickers0[idx]],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Risk Contribution")
    ax.set_title("Marginal Risk Contributions (top assets)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)


# ── Public entry point ───────────────────────────────────────────────────────


def plot_report(
    *,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None,
    backtests: dict[str, BacktestResult],
    cov_df: pd.DataFrame,
    risk_contribs: dict[str, tuple[np.ndarray, pd.Index]],
    ticker_names: pd.Series,
    risk_free: float = 0.0193,
    annual_factor: int = 252,
    figures_dir: str | None = None,
) -> Figure:
    """2×2 figure: frontier, equity, correlation (+ colorbar), risk contributions."""
    _apply_style()
    fig = plt.figure(figsize=_REPORT_FIGSIZE)
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1, 1], height_ratios=[1, 1], **_GS_MAIN,
    )

    _plot_frontier(
        fig.add_subplot(gs[0, 0]),
        frontier_vols=frontier_vols,
        frontier_rets=frontier_rets,
        tangency=tangency,
        min_var=min_var,
        risk_free=risk_free,
    )
    _plot_equity(
        fig.add_subplot(gs[0, 1]),
        backtests,
        annual_factor=annual_factor,
        risk_free=risk_free,
    )

    gsc = gs[1, 0].subgridspec(1, 2, **_GS_CORR)
    ax_c = fig.add_subplot(gsc[0, 0])
    fig.add_subplot(gsc[0, 1]).set_axis_off()
    _plot_correlation(ax_c, cov_df, ticker_names)
    _plot_risk_contributions(fig.add_subplot(gs[1, 1]), risk_contribs, ticker_names)

    fig.suptitle("Portfolio Optimizer Report", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "portfolio_report.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig
