"""把已经拟合的达标概率写成孕周网格，供图和表使用。

这里不再重新估计参数。问题二用组内对数正态的 mu、sigma；
问题三的曲线在 problem3 里按个体概率平均后写入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import decision_objective, time_grid, time_risk
from .survival import success_probability


def lognormal_curves(groups: list[dict]) -> pd.DataFrame:
    grid = time_grid()
    risk = np.asarray(time_risk(grid), dtype=float)
    rows = []
    for group in groups:
        mu = group.get("mu")
        sigma = group.get("sigma")
        if not group.get("success") or mu is None or sigma is None:
            continue
        if not np.isfinite(mu) or not np.isfinite(sigma):
            continue
        pi = np.asarray(success_probability(mu, sigma, grid), dtype=float)
        objective = np.asarray(decision_objective(grid, pi), dtype=float)
        for t, probability, time_penalty, value in zip(grid, pi, risk, objective):
            rows.append(
                {
                    "group": int(group["group"]),
                    "interval": group.get("interval", ""),
                    "t": float(t),
                    "pi": float(probability),
                    "risk": float(time_penalty),
                    "objective": float(value),
                    "t_star": float(group.get("t_star", np.nan)),
                }
            )
    return pd.DataFrame(rows)
