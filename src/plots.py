"""Visualisation helpers for the Markowitz portfolio optimiser."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import config as cfg
from markowitz import BacktestResult, OptimResult, oos_metrics

_COLORS = {
    "tangency": "#e63946",
    "minvar": "#2196F3",
    "equal": "#4CAF50",
    "cml": "#e63946",
    "frontier": "#1a1a2e",
}
_FRONTIER_XLIM = (0.0, 0.20)
_FRONTIER_YLIM = (0.0, 0.18)
_PANEL_FIGSIZE = (8.0, 5.0)
_CORR_FIGSIZE = (9.0, 7.0)
_RISK_RC_FIGSIZE = (14.0, 6.5)
_TOP_RISK_BARS = 8
_MIN_OTHER_SHARE = 0.005
_BAR_LABEL_MIN = 0.04
_SAVE_DPI = 300

RiskEntry = (
    tuple[np.ndarray, pd.Index]
    | tuple[np.ndarray, pd.Index, np.ndarray]
)


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.titlepad": 9,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.1,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.antialiased": True,
            "patch.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=_SAVE_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


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
    ax.plot(
        frontier_vols,
        frontier_rets,
        color=_COLORS["frontier"],
        linewidth=2.35,
        solid_capstyle="round",
        zorder=4,
        label="Efficient frontier",
    )

    x_hi = _FRONTIER_XLIM[1]
    cml_x = np.linspace(0.0, x_hi, 96)
    ax.plot(
        cml_x,
        risk_free + tangency.sharpe * cml_x,
        color=_COLORS["cml"],
        linewidth=2.0,
        linestyle="--",
        dash_capstyle="round",
        zorder=4,
        label="Capital Market Line",
    )
    ax.scatter(
        0,
        risk_free,
        color=_COLORS["cml"],
        s=70,
        zorder=6,
        linewidths=1.0,
        edgecolors="white",
    )
    ax.scatter(
        tangency.vol,
        tangency.ret,
        color=_COLORS["tangency"],
        s=260,
        marker="*",
        zorder=6,
        linewidths=1.0,
        edgecolors="white",
        label=f"Tangency (SR {tangency.sharpe:.2f})",
    )

    if min_var:
        ax.scatter(
            min_var.vol,
            min_var.ret,
            color=_COLORS["minvar"],
            s=190,
            marker="o",
            linewidths=1.1,
            edgecolors="white",
            zorder=6,
            label="Min Var",
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
    if not backtests:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No walk-forward backtests",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        return

    color_cycle = [_COLORS["tangency"], _COLORS["minvar"], _COLORS["equal"]]
    ax.axhline(1, color="black", linewidth=0.75, alpha=0.35)

    for i, (name, bt) in enumerate(backtests.items()):
        m = oos_metrics(bt, annual_factor=annual_factor, risk_free=risk_free)
        label = f"{name}  SR {m['sharpe']:.2f}  DD {m['max_dd']:.0%}"
        ax.plot(
            bt.portfolio_value,
            color=color_cycle[i % len(color_cycle)],
            linewidth=1.35,
            solid_capstyle="round",
            alpha=0.92,
            zorder=3 + i,
            label=label,
        )

    ax.set_title("Out-of-Sample Equity Curves")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)


def _plot_correlation(ax, cov_df: pd.DataFrame, ticker_names: pd.Series):
    std = np.sqrt(np.diag(cov_df.values))
    std[std == 0] = 1.0
    corr = cov_df.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    corr = corr[np.ix_(order, order)]
    labels = [_display_name(str(t), ticker_names) for t in cov_df.columns[order]]

    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([])
    if len(labels) <= 25:
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_title("Correlation (Ledoit-Wolf, clustered)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.2%", pad=0.06)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)


def _risk_entry(entry: RiskEntry) -> tuple[np.ndarray, pd.Index, np.ndarray | None]:
    if len(entry) == 3:
        return entry[0], entry[1], entry[2]
    return entry[0], entry[1], None


def _truncate_label(text: str, max_len: int = 36) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _attribution_rows(
    rc: np.ndarray,
    tickers: pd.Index,
    ticker_names: pd.Series,
    weights: np.ndarray | None,
    *,
    top_n: int = _TOP_RISK_BARS,
) -> list[dict[str, float | str | None]]:
    """Top contributors plus optional Other bucket."""
    order = np.argsort(rc)[::-1]
    significant = order[rc[order] >= _MIN_OTHER_SHARE]
    rows: list[dict[str, float | str | None]] = []
    for i in significant[:top_n]:
        rows.append({
            "label": _truncate_label(_display_name(str(tickers[i]), ticker_names)),
            "risk": float(rc[i]),
            "weight": float(weights[i]) if weights is not None else None,
        })
    if len(significant) > top_n:
        tail = significant[top_n:]
        other_risk = float(rc[tail].sum())
        if other_risk >= _MIN_OTHER_SHARE:
            other_weight = float(weights[tail].sum()) if weights is not None else None
            rows.append({
                "label": f"Other ({len(tail)} funds)",
                "risk": other_risk,
                "weight": other_weight,
            })
    return rows


def _risk_ylabel(row: dict[str, float | str | None]) -> str:
    label = str(row["label"])
    weight, risk = row["weight"], row["risk"]
    if weight is None:
        return label
    return f"{label}\n{weight:.0%} weight  ·  {risk:.0%} of risk"


def _plot_risk_attribution_panel(
    ax,
    portfolio_name: str,
    rc: np.ndarray,
    tickers: pd.Index,
    ticker_names: pd.Series,
    *,
    weights: np.ndarray | None,
    color: str,
) -> None:
    rows = _attribution_rows(rc, tickers, ticker_names, weights)
    if not rows:
        ax.set_axis_off()
        return

    values = np.array([r["risk"] for r in rows])
    y = np.arange(len(rows))
    bars = ax.barh(
        y, values, height=0.72, color=color, alpha=0.92,
        edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(
        bars,
        labels=[f"{v:.0%}" if v >= _BAR_LABEL_MIN else "" for v in values],
        padding=4,
        fontsize=9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([_risk_ylabel(r) for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.05, max(values) * 1.18 + 0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Share of portfolio volatility", fontsize=10)
    ax.set_title(portfolio_name, fontsize=12, pad=8)

    top3 = float(np.sort(rc)[::-1][:3].sum())
    ax.text(
        0.98, 0.04, f"Top 3 funds: {top3:.0%} of risk",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", alpha=0.85, edgecolor="#cccccc",
        ),
    )


def _plot_risk_contributions(
    fig: plt.Figure,
    risk_contribs: dict[str, RiskEntry],
    ticker_names: pd.Series,
) -> None:
    if not risk_contribs:
        fig.text(0.5, 0.5, "No risk attribution data", ha="center", va="center")
        return

    n = len(risk_contribs)
    axes = np.atleast_1d(fig.subplots(1, n, squeeze=False)).flatten()
    panel_colors = [_COLORS["tangency"], _COLORS["minvar"]]

    for ax, (name, entry), color in zip(
        axes, risk_contribs.items(), panel_colors, strict=False,
    ):
        rc, tickers, weights = _risk_entry(entry)
        _plot_risk_attribution_panel(
            ax, name, rc, tickers, ticker_names, weights=weights, color=color,
        )

    fig.suptitle(
        "Which funds drive portfolio risk?",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.supxlabel(
        "Each bar is that fund's share of total portfolio volatility "
        "(Euler decomposition; all bars sum to 100% per portfolio).",
        fontsize=10, color="#555555",
    )


def plot_report(
    *,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency: OptimResult,
    min_var: OptimResult | None,
    backtests: dict[str, BacktestResult],
    cov_df: pd.DataFrame,
    risk_contribs: dict[str, RiskEntry],
    ticker_names: pd.Series,
    risk_free: float,
    annual_factor: int = 252,
    figures_dir: str | None = None,
    figure_filenames: dict[str, str] | None = None,
) -> dict[str, str]:
    """Save frontier, equity, correlation, and risk-contribution PNGs."""
    _apply_style()
    names = figure_filenames or cfg.FIGURE_FILENAMES
    if figures_dir is None:
        return {}

    saved: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE, layout="constrained")
    _plot_frontier(
        ax,
        frontier_vols=frontier_vols,
        frontier_rets=frontier_rets,
        tangency=tangency,
        min_var=min_var,
        risk_free=risk_free,
    )
    key = "efficient_frontier"
    path = os.path.join(figures_dir, names[key])
    _save_fig(fig, path)
    saved[key] = path

    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE, layout="constrained")
    _plot_equity(ax, backtests, annual_factor=annual_factor, risk_free=risk_free)
    key = "oos_equity"
    path = os.path.join(figures_dir, names[key])
    _save_fig(fig, path)
    saved[key] = path

    fig, ax = plt.subplots(figsize=_CORR_FIGSIZE, layout="constrained")
    _plot_correlation(ax, cov_df, ticker_names)
    key = "correlation"
    path = os.path.join(figures_dir, names[key])
    _save_fig(fig, path)
    saved[key] = path

    fig = plt.figure(figsize=_RISK_RC_FIGSIZE, layout="constrained")
    _plot_risk_contributions(fig, risk_contribs, ticker_names)
    key = "risk_contributions"
    path = os.path.join(figures_dir, names[key])
    _save_fig(fig, path)
    saved[key] = path

    return saved
