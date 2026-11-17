"""只在 BMI 轴上做有序分割。不使用 K-means，也不把达标时间放进聚类特征。"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .constants import MAX_BMI_CUTS, N_MIN_GROUP, OFFICIAL_BMI_CUTS


def boundary_indices(bmi: np.ndarray, n_min: int = N_MIN_GROUP, max_interior: int = MAX_BMI_CUTS) -> list[int]:
    """候选切点只落在 BMI 发生变化的位置，并且每组至少 ``n_min`` 人。"""
    values = np.asarray(bmi, dtype=float)
    n = len(values)
    change = [index for index in range(1, n) if values[index] != values[index - 1]]
    feasible = [index for index in change if n_min <= index <= n - n_min]
    if len(feasible) > max_interior:
        chosen = np.linspace(0, len(feasible) - 1, max_interior)
        keep = sorted({int(round(item)) for item in chosen})
        feasible = [feasible[item] for item in keep]
    return [0, *feasible, n]


def best_ordered_partition(boundary: list[int], segment_cost: Callable[[int, int], float], k: int, n_min: int):
    """动态规划。``segment_cost(start, end)`` 用的是排序后的孕妇下标切片。"""
    m = len(boundary)
    inf = 1e300
    dp = [[inf] * m for _ in range(k + 1)]
    prev = [[-1] * m for _ in range(k + 1)]
    dp[0][0] = 0.0
    for groups in range(1, k + 1):
        for end_pos in range(1, m):
            best = inf
            best_start = -1
            end = boundary[end_pos]
            for start_pos in range(groups - 1, end_pos):
                start = boundary[start_pos]
                if end - start < n_min or dp[groups - 1][start_pos] >= inf / 2:
                    continue
                value = dp[groups - 1][start_pos] + float(segment_cost(start, end))
                if value < best:
                    best = value
                    best_start = start_pos
            dp[groups][end_pos] = best
            prev[groups][end_pos] = best_start
    if dp[k][m - 1] >= inf / 2:
        return None, inf
    nodes = [m - 1]
    cursor = m - 1
    for groups in range(k, 0, -1):
        cursor = prev[groups][cursor]
        nodes.append(cursor)
    nodes = nodes[::-1]
    edges = [boundary[item] for item in nodes]
    return edges, dp[k][m - 1]


def cuts_from_edges(bmi_sorted: np.ndarray, edges: list[int]) -> list[float]:
    """下一组第一位孕妇的 BMI，作为左闭区间的切点。"""
    return [float(bmi_sorted[edge]) for edge in edges[1:-1]]


def assign_groups(bmi, cuts) -> np.ndarray:
    """``cuts`` 是各组（除第一组外）的 BMI 下界。等于切点的人进入右侧组。"""
    values = np.asarray(bmi, dtype=float)
    if len(cuts) == 0:
        return np.zeros(len(values), dtype=int)
    return np.searchsorted(np.asarray(cuts, dtype=float), values, side="right").astype(int)


def official_bmi_group(bmi, cuts=OFFICIAL_BMI_CUTS) -> np.ndarray:
    return assign_groups(bmi, cuts)


def official_labels(cuts=OFFICIAL_BMI_CUTS) -> list[str]:
    labels = [f"[{_format_cut(20)}, {_format_cut(cuts[0])})"]
    for left, right in zip(cuts[:-1], cuts[1:]):
        labels.append(f"[{_format_cut(left)}, {_format_cut(right)})")
    labels.append(f"≥{_format_cut(cuts[-1])}")
    return labels


def interval_label(cuts: list[float], group: int, observed_bmi: np.ndarray) -> str:
    observed = np.asarray(observed_bmi, dtype=float)
    low = float(np.min(observed))
    high = float(np.max(observed))
    if len(cuts) == 0:
        return f"[{low:.2f}, {high:.2f}]"
    if group == 0:
        return f"[{low:.2f}, {cuts[0]:.2f})"
    if group == len(cuts):
        return f"[{cuts[-1]:.2f}, {high:.2f}]"
    return f"[{cuts[group - 1]:.2f}, {cuts[group]:.2f})"


def bic(nll: float, n: int, n_params: int) -> float:
    if n <= 1 or not np.isfinite(nll):
        return float("inf")
    return float(2.0 * nll + n_params * np.log(n))


def _format_cut(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
