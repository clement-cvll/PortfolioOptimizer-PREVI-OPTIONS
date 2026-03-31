"""Visualisation helpers for the Markowitz portfolio optimiser."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from markowitz import BacktestResult, OptimResult


def _apply_professional_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_frontier_ax(
    ax,
    *,
    mc_vols: np.ndarray,
    mc_returns: np.ndarray,
    mc_sharpes: np.ndarray,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None,
    risk_free: float,
):
    mask = (mc_returns > -0.02) & (mc_vols < frontier_vols.max() * 1.2)
    visible = np.where(mask)[0]
    if len(visible) == 0:
        raise ValueError(
            "No Monte Carlo points are within the visible plotting window."
        )

    s_min = float(mc_sharpes[visible].min())
    s_max = float(mc_sharpes[visible].max())
    norm = mcolors.TwoSlopeNorm(
        vmin=min(s_min, -0.001),
        vcenter=0,
        vmax=max(s_max, 0.001),
    )

    v_vols = mc_vols[visible]
    v_rets = mc_returns[visible]
    v_sharpes = mc_sharpes[visible]
    if len(visible) >= 150_000:
        mappable = ax.hexbin(
            v_vols,
            v_rets,
            C=v_sharpes,
            reduce_C_function=np.mean,
            gridsize=200,
            cmap="Greens",
            norm=norm,
            mincnt=1,
            linewidths=0,
            alpha=0.95,
            zorder=2,
        )
    else:
        mappable = ax.scatter(
            v_vols,
            v_rets,
            c=v_sharpes,
            cmap="Greens",
            norm=norm,
            s=5,
            alpha=0.25,
            rasterized=True,
            zorder=2,
        )

    ax.plot(
        frontier_vols,
        frontier_rets,
        color="#1a1a2e",
        linewidth=2.5,
        zorder=4,
        label="Efficient frontier",
    )

    cml_vols = np.array([0, tangency.vol * 1.05])
    ax.plot(
        cml_vols,
        risk_free + tangency.sharpe * cml_vols,
        color="#e63946",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
        label="Capital Market Line",
    )
    ax.scatter(0, risk_free, color="#e63946", s=55, zorder=6)
    ax.scatter(
        tangency.vol,
        tangency.ret,
        color="#e63946",
        s=240,
        marker="*",
        zorder=6,
        label=f"Tangency (SR {tangency.sharpe:.2f})",
    )

    if min_var:
        ax.scatter(
            min_var.vol,
            min_var.ret,
            color="#2196F3",
            s=160,
            marker="o",
            edgecolor="white",
            zorder=6,
            label="Min Var",
        )

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set(xlabel="Annualised Volatility", ylabel="Annualised Return")
    ax.set_xlim(left=0)
    ax.set_title("Efficient Frontier")
    ax.legend(framealpha=0.95, loc="upper left")
    return mappable


def _plot_strategy_comparison_axes(
    ax_eq, ax_sh, backtests: dict[str, BacktestResult]
) -> None:
    colors = ["#e63946", "#2196F3", "#4CAF50", "#FF9800"]
    first_bt = next(iter(backtests.values()))

    ax_eq.axhline(1, color="black", linewidth=0.8, alpha=0.4)
    if len(first_bt.rebal_dates) <= 60:
        for d in first_bt.rebal_dates:
            ax_eq.axvline(d, color="#9E9E9E", linewidth=0.8, linestyle=":", alpha=0.7)

    for idx, (name, bt) in enumerate(backtests.items()):
        color = colors[idx % len(colors)]
        ax_eq.plot(bt.portfolio_value, color=color, linewidth=2, zorder=3, label=name)

    ax_eq.set_title("Out-of-Sample Equity Curves")
    ax_eq.set_ylabel("Portfolio Value")
    ax_eq.legend(loc="upper left", framealpha=0.95)

    n_strats = len(backtests)
    width = 0.8 / n_strats
    x = np.arange(len(first_bt.period_sharpes))

    for idx, (name, bt) in enumerate(backtests.items()):
        color = colors[idx % len(colors)]
        offset = (idx - n_strats / 2 + 0.5) * width
        ax_sh.bar(
            x + offset,
            bt.period_sharpes,
            width=width,
            color=color,
            alpha=0.85,
            label=name,
        )

    ax_sh.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_sh.set_xticks(x)
    ax_sh.set_xticklabels(
        [d.strftime("%b %Y") for d in first_bt.rebal_dates],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax_sh.set_title("Out-of-Sample Sharpe per Period")
    ax_sh.set_ylabel("Sharpe")


def plot_report(
    *,
    mc_vols: np.ndarray,
    mc_returns: np.ndarray,
    mc_sharpes: np.ndarray,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None,
    backtests: dict[str, BacktestResult],
    risk_free: float = 0.0193,
    figures_dir: str | None = None,
) -> Figure:
    """Single professional report: frontier + equity curves + Sharpe bars."""
    _apply_professional_style()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])

    ax_frontier = fig.add_subplot(gs[:, 0])
    ax_eq = fig.add_subplot(gs[0, 1])
    ax_sh = fig.add_subplot(gs[1, 1])

    mappable = _plot_frontier_ax(
        ax_frontier,
        mc_vols=mc_vols,
        mc_returns=mc_returns,
        mc_sharpes=mc_sharpes,
        frontier_vols=frontier_vols,
        frontier_rets=frontier_rets,
        tangency=tangency,
        min_var=min_var,
        risk_free=risk_free,
    )
    fig.colorbar(mappable, ax=ax_frontier, pad=0.02, fraction=0.035).set_label(
        "Sharpe Ratio"
    )

    _plot_strategy_comparison_axes(ax_eq, ax_sh, backtests)

    fig.suptitle("Portfolio Optimizer Report", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "portfolio_report.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig


def plot_frontier(
    mc_vols: np.ndarray,
    mc_returns: np.ndarray,
    mc_sharpes: np.ndarray,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None = None,
    *,
    risk_free: float = 0.0193,
    figures_dir: str | None = None,
) -> Figure:
    """Deprecated: use plot_report()."""
    return plot_report(
        mc_vols=mc_vols,
        mc_returns=mc_returns,
        mc_sharpes=mc_sharpes,
        frontier_vols=frontier_vols,
        frontier_rets=frontier_rets,
        tangency=tangency,
        min_var=min_var,
        backtests={},
        risk_free=risk_free,
        figures_dir=figures_dir,
    )


def plot_backtest(
    result: BacktestResult,
    in_sample_sharpe: float,
    *,
    figures_dir: str | None = None,
) -> Figure:
    """Deprecated: use plot_report() with a single backtest."""
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]}
    )
    pv = result.portfolio_value
    sharpes = result.period_sharpes

    # ── Equity curve ──────────────────────────────────────────────────────
    ax0.plot(pv, color="#2196F3", linewidth=2, zorder=3)
    ax0.fill_between(
        pv.index, 1, pv, where=pv >= 1, color="#2196F3", alpha=0.12, zorder=2
    )
    ax0.fill_between(
        pv.index, 1, pv, where=pv < 1, color="#e63946", alpha=0.12, zorder=2
    )
    ax0.axhline(1, color="black", linewidth=0.8, alpha=0.4)
    # Too many vertical lines slows rendering; cap for readability/perf.
    if len(result.rebal_dates) <= 60:
        for d in result.rebal_dates:
            ax0.axvline(d, color="#9E9E9E", linewidth=0.8, linestyle=":", alpha=0.7)
    ax0.set_title(
        "Walk-Forward Backtest — Out-of-Sample Equity Curve",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax0.set_ylabel("Portfolio Value", fontsize=11)

    # ── Per-period Sharpe bars ────────────────────────────────────────────
    bar_colors = ["#4CAF50" if s > 0 else "#e63946" for s in sharpes]
    ax1.bar(
        range(len(sharpes)),
        sharpes,
        color=bar_colors,
        edgecolor="white",
        width=0.7,
        zorder=3,
    )
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.axhline(
        in_sample_sharpe,
        color="#FF5722",
        ls="--",
        lw=1.5,
        zorder=4,
        label=f"In-sample: {in_sample_sharpe:.2f}",
    )
    mean_oos = float(np.mean(sharpes))
    ax1.axhline(
        mean_oos,
        color="#FF9800",
        ls="--",
        lw=1.5,
        zorder=4,
        label=f"Mean OOS: {mean_oos:.2f}",
    )
    ax1.set_xticks(range(len(result.rebal_dates)))
    ax1.set_xticklabels(
        [d.strftime("%b %Y") for d in result.rebal_dates],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    ax1.set_title("OOS Sharpe Ratio per Period", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.legend(fontsize=10, framealpha=0.95)

    fig.tight_layout(h_pad=3)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "markowitz_walkforward_backtest.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig


def plot_strategy_comparison(
    backtests: dict[str, BacktestResult],
    *,
    figures_dir: str | None = None,
) -> Figure:
    """Deprecated: use plot_report()."""
    _apply_professional_style()
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]}
    )
    _plot_strategy_comparison_axes(ax0, ax1, backtests)
    fig.tight_layout(h_pad=3)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "strategy_comparison.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig
