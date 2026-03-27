"""Visualisation helpers for the Markowitz portfolio optimiser."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from markowitz import BacktestResult, OptimResult


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
    """MC scatter + efficient frontier + CML + tangency & min-var points."""
    plt.style.use("seaborn-v0_8-whitegrid")

    visible = np.where((mc_returns > -0.02) & (mc_vols < frontier_vols.max() * 1.2))[0]
    
    s_min = float(mc_sharpes[visible].min())
    s_max = float(mc_sharpes[visible].max())
    
    norm = mcolors.TwoSlopeNorm(
        vmin=min(s_min, -0.001),
        vcenter=0,
        vmax=max(s_max, 0.001),
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sc = ax.scatter(
        mc_vols[visible],
        mc_returns[visible],
        c=mc_sharpes[visible],
        cmap="Greens",
        norm=norm,
        s=4,
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

    # Capital Market Line
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
    ax.scatter(0, risk_free, color="#e63946", s=60, zorder=6)

    ax.scatter(
        tangency.vol,
        tangency.ret,
        color="#e63946",
        s=280,
        marker="*",
        zorder=6,
        label=f"Tangency  |  {tangency.ret:.1%} ret  |  {tangency.vol:.1%} vol  |  SR {tangency.sharpe:.2f}",
    )

    if min_var:
        ax.scatter(
            min_var.vol,
            min_var.ret,
            color="#2196F3",
            s=280,
            marker="o",
            edgecolor="white",
            zorder=6,
            label=f"Min Var  |  {min_var.ret:.1%} ret  |  {min_var.vol:.1%} vol  |  SR {min_var.sharpe:.2f}",
        )

    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03).set_label(
        "Sharpe Ratio", fontsize=11
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set(xlabel="Annualised Volatility", ylabel="Annualised Return")
    ax.set_xlim(left=0)
    ax.set_title(
        "Markowitz Portfolio Optimisation", fontsize=15, fontweight="bold", pad=12
    )
    ax.legend(fontsize=10, framealpha=0.95, loc="upper left")
    fig.tight_layout()

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "markowitz_portfolio_frontier.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig


def plot_backtest(
    result: BacktestResult,
    in_sample_sharpe: float,
    *,
    figures_dir: str | None = None,
) -> Figure:
    """Equity curve (top) and per-period Sharpe bars (bottom)."""
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
    """Compare multiple walk-forward backtests (equity curves & Sharpe ratios)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]}
    )
    colors = ["#e63946", "#2196F3", "#4CAF50", "#FF9800"]
    first_bt = next(iter(backtests.values()))

    # ── Equity curves ─────────────────────────────────────────────────────
    ax0.axhline(1, color="black", linewidth=0.8, alpha=0.4)
    for d in first_bt.rebal_dates:
        ax0.axvline(d, color="#9E9E9E", linewidth=0.8, linestyle=":", alpha=0.7)

    for idx, (name, bt) in enumerate(backtests.items()):
        color = colors[idx % len(colors)]
        ax0.plot(bt.portfolio_value, color=color, linewidth=2, zorder=3, label=name)

    ax0.set_title(
        "Walk-Forward Backtest — Out-of-Sample Equity Curves",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax0.set_ylabel("Portfolio Value", fontsize=11)
    ax0.legend(loc="upper left", framealpha=0.95)

    # ── Per-period Sharpe grouped bars ────────────────────────────────────
    n_strats = len(backtests)
    width = 0.8 / n_strats
    x = np.arange(len(first_bt.period_sharpes))

    for idx, (name, bt) in enumerate(backtests.items()):
        color = colors[idx % len(colors)]
        offset = (idx - n_strats / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            bt.period_sharpes,
            width=width,
            color=color,
            alpha=0.85,
            label=name,
        )

        mean_oos = float(np.nanmean(bt.period_sharpes))
        ax1.axhline(
            mean_oos,
            color=color,
            ls="--",
            lw=1.5,
            zorder=4,
            label=f"{name} Mean OOS: {mean_oos:.2f}",
        )

    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [d.strftime("%b %Y") for d in first_bt.rebal_dates],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    ax1.set_title("OOS Sharpe Ratio per Period", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.legend(fontsize=9, framealpha=0.95, ncol=2)

    fig.tight_layout(h_pad=3)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(
            os.path.join(figures_dir, "strategy_comparison.png"),
            dpi=200,
            bbox_inches="tight",
        )
    return fig
