"""把修订分析画成小图。图中的线来自同一轮已经写出的表，不另估参数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .reporting import figure_directories


def write_figures(
    *,
    male: pd.DataFrame,
    female: pd.DataFrame,
    bins: pd.DataFrame,
    curves2: pd.DataFrame,
    curves3: pd.DataFrame,
    groups3: pd.DataFrame,
    folders=None,
) -> list[str]:
    plt = _pyplot()
    targets = list(folders) if folders is not None else figure_directories()
    written = []
    specs = (
        ("y_vs_ga_bmi.png", lambda: _figure_concentration(plt, male, bins)),
        ("problem2_risk_curves.png", lambda: _figure_problem2(plt, curves2)),
        ("problem3_timing.png", lambda: _figure_problem3(plt, groups3, curves3)),
        ("problem4_z_by_ab.png", lambda: _figure_problem4(plt, female)),
    )
    for name, builder in specs:
        figure = builder()
        if figure is None:
            continue
        for folder in targets:
            path = Path(folder) / name
            figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
            written.append(str(path))
        plt.close(figure)
    return written


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    chosen = None
    for path in font_manager.findSystemFonts():
        lower = path.lower()
        if "wqy-microhei" in lower or "notosanscjk" in lower or "sourcehansans" in lower:
            chosen = path
            break
    if chosen is not None:
        font_manager.fontManager.addfont(chosen)
        name = font_manager.FontProperties(fname=chosen).get_name()
        plt.rcParams["font.family"] = name
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 0.6
    return plt


def _figure_concentration(plt, male: pd.DataFrame, bins: pd.DataFrame):
    if male is None or male.empty or "y_frac" not in male.columns:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    _scatter_with_bins(axes[0], male["ga_weeks"], male["y_frac"], bins, "ga", "孕周（周）")
    _scatter_with_bins(axes[1], male["bmi"], male["y_frac"], bins, "bmi", "当次 BMI")
    axes[0].set_ylabel("Y 染色体浓度")
    figure.tight_layout()
    return figure


def _scatter_with_bins(ax, x, y, bins: pd.DataFrame, variable: str, xlabel: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.scatter(x, y, s=9, c="#9aa7b5", alpha=0.45, linewidths=0, rasterized=True)
    if bins is not None and not bins.empty and "variable" in bins.columns:
        part = bins.loc[(bins["variable"] == variable) & (bins["n"] >= 8)].sort_values("mean_x")
        if not part.empty:
            ax.plot(part["mean_x"], part["mean_y"], color="#1f4e79", marker="o", markersize=3.5, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.axhline(0.04, color="#b23a48", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _figure_problem2(plt, curves: pd.DataFrame):
    if curves is None or curves.empty:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    colors = ["#1f4e79", "#c47b2b", "#2f6f4e", "#6b4c9a"]
    for index, (interval, part) in enumerate(curves.groupby("interval", sort=False)):
        color = colors[index % len(colors)]
        part = part.sort_values("t")
        axes[0].plot(part["t"], part["pi"], color=color, linewidth=1.4, label=str(interval))
        axes[1].plot(part["t"], part["objective"], color=color, linewidth=1.4, label=str(interval))
        t_star = float(part["t_star"].iloc[0])
        for axis, column in ((axes[0], "pi"), (axes[1], "objective")):
            axis.axvline(t_star, color=color, linewidth=0.7, linestyle=":")
            nearest = part.iloc[(part["t"] - t_star).abs().argmin()]
            axis.scatter([t_star], [nearest[column]], color=color, s=16, zorder=3)
    axes[0].set_ylabel("达标概率")
    axes[1].set_ylabel("目标函数")
    for axis in axes:
        axis.set_xlabel("孕周（周）")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def _figure_problem3(plt, groups: pd.DataFrame, curves: pd.DataFrame):
    if groups is None or groups.empty or "t_star" not in groups.columns:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    labels = [f"{interval}\nn={int(n)}" for interval, n in zip(groups["interval"], groups["n_women"])]
    x = np.arange(len(groups))
    y = groups["t_star"].to_numpy(dtype=float)
    if {"t_boot_low", "t_boot_high"}.issubset(groups.columns):
        low = y - groups["t_boot_low"].to_numpy(dtype=float)
        high = groups["t_boot_high"].to_numpy(dtype=float) - y
        yerr = np.vstack([low, high])
    else:
        yerr = None
    axes[0].errorbar(x, y, yerr=yerr, fmt="o", color="#1f4e79", ecolor="#7f93a8", capsize=3, markersize=4.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6.5)
    axes[0].set_ylabel("检测时点（周）")
    axes[0].set_xlabel("首次 BMI 区间")
    colors = ["#1f4e79", "#4c78a8", "#f2c14e", "#e07a3d", "#b23a48"]
    if curves is not None and not curves.empty:
        for index, (interval, part) in enumerate(curves.groupby("interval", sort=False)):
            part = part.sort_values("t")
            color = colors[index % len(colors)]
            axes[1].plot(part["t"], part["pi"], color=color, linewidth=1.3, label=str(interval))
            t_star = float(part["t_star"].iloc[0])
            axes[1].axvline(t_star, color=color, linewidth=0.6, linestyle=":")
        axes[1].legend(frameon=False, fontsize=6, loc="lower right")
    axes[1].set_xlabel("孕周（周）")
    axes[1].set_ylabel("组内平均达标概率")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _figure_problem4(plt, female: pd.DataFrame):
    needed = {"z13", "z18", "z21", "ab_t13", "ab_t18", "ab_t21"}
    if female is None or female.empty or not needed.issubset(female.columns):
        return None
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    specs = (("13 号", "z13", "ab_t13"), ("18 号", "z18", "ab_t18"), ("21 号", "z21", "ab_t21"))
    for axis, (title, z_column, label) in zip(axes, specs):
        z = female[z_column].to_numpy(dtype=float)
        y = female[label].to_numpy(dtype=int)
        negative = z[(y == 0) & np.isfinite(z)]
        positive = z[(y == 1) & np.isfinite(z)]
        axis.boxplot(
            [negative, positive],
            positions=[1, 2],
            widths=0.55,
            showfliers=True,
            medianprops={"color": "#1f4e79"},
            boxprops={"color": "#1f4e79"},
            whiskerprops={"color": "#1f4e79"},
            capprops={"color": "#1f4e79"},
        )
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["AB 阴性", "AB 阳性"])
        rng = np.random.default_rng(2025)
        if positive.size:
            axis.scatter(
                2 + rng.uniform(-0.08, 0.08, size=positive.size),
                positive,
                s=12,
                c="#b23a48",
                alpha=0.75,
                linewidths=0,
                zorder=3,
            )
        axis.axhline(3, color="#b23a48", linewidth=0.7, linestyle="--")
        axis.axhline(-3, color="#b23a48", linewidth=0.7, linestyle="--")
        axis.set_title(title, fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Z 值")
    figure.tight_layout()
    return figure
