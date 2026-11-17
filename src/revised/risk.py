"""时间风险。主设定是组委会讲评中的分段线性累积，斜率 1:3:6，再归一化到 [0, 1]。

未达标不另设第四个官方权重，而是进入目标函数里的「尚未达标」项。
"""

from __future__ import annotations

import numpy as np

from .constants import (
    PRIMARY_LAMBDA,
    PRIMARY_SLOPES,
    RISK_NORM_HIGH,
    RISK_NORM_LOW,
    T_MAX,
    T_MIN,
    T_STEP,
)


def exposure_pieces(time, early_end: float = 12.0, late_start: float = 28.0):
    """早期、中期、晚期三段上已经等待的时间长度。"""
    t = np.asarray(time, dtype=float)
    early = np.minimum(t, early_end)
    mid = np.maximum(np.minimum(t, late_start) - early_end, 0.0)
    late = np.maximum(t - late_start, 0.0)
    return early, mid, late


def raw_time_risk(time, slopes=PRIMARY_SLOPES, early_end: float = 12.0, late_start: float = 28.0):
    early, mid, late = exposure_pieces(time, early_end=early_end, late_start=late_start)
    slope_early, slope_mid, slope_late = slopes
    return slope_early * early + slope_mid * mid + slope_late * late


def time_risk(
    time,
    slopes=PRIMARY_SLOPES,
    t0: float = RISK_NORM_LOW,
    t1: float = RISK_NORM_HIGH,
):
    """归一化时间风险。``t0`` 处为 0，``t1`` 处为 1，中间单调上升。"""
    raw = raw_time_risk(time, slopes=slopes)
    raw0 = float(np.asarray(raw_time_risk(t0, slopes=slopes)).reshape(-1)[0])
    raw1 = float(np.asarray(raw_time_risk(t1, slopes=slopes)).reshape(-1)[0])
    scale = raw1 - raw0
    if scale <= 0:
        raise ValueError("风险归一化的两个端点必须使 r(t) 严格增加")
    return np.clip((raw - raw0) / scale, 0.0, 1.0)


def decision_objective(time, success_probability, lam: float = PRIMARY_LAMBDA, slopes=PRIMARY_SLOPES):
    """J(t) = -log(π(t)) + λ Risk(t)。π 相同时，更早的时点时间风险更低。"""
    pi = np.clip(np.asarray(success_probability, dtype=float), 1e-6, 1.0)
    return -np.log(pi) + lam * time_risk(time, slopes=slopes)


def time_grid(t_min: float = T_MIN, t_max: float = T_MAX, step: float = T_STEP) -> np.ndarray:
    count = int(round((t_max - t_min) / step))
    return np.round(t_min + step * np.arange(count + 1), 10)


def minimize_objective(success_on_grid, grid=None, lam: float = PRIMARY_LAMBDA, slopes=PRIMARY_SLOPES):
    """在给定网格上取 J 的最小点。返回 (时点, 目标值, 该点达标概率)。"""
    grid = time_grid() if grid is None else np.asarray(grid, dtype=float)
    values = np.asarray(decision_objective(grid, success_on_grid, lam=lam, slopes=slopes), dtype=float)
    index = int(np.nanargmin(values))
    pi = np.asarray(success_on_grid, dtype=float)
    return float(grid[index]), float(values[index]), float(pi[index])


def sensitivity_settings():
    """预先登记的小敏感性网格：三组斜率比 × 三个 λ。主设定是其中的 1:3:6 与 λ=1。"""
    slopes = ((1.0, 2.0, 4.0), (1.0, 3.0, 6.0), (1.0, 4.0, 8.0))
    lambdas = (0.5, 1.0, 2.0)
    for slope in slopes:
        for lam in lambdas:
            yield slope, lam
