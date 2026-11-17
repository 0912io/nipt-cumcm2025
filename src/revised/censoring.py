"""最早达标时间的左删失、区间删失与右删失。

同一抽血次数的多行视为同一次抽血的重复测序，先汇总再构造删失区间。
从未达标的孕妇保留为右删失，不删除，也不把左删失写成精确达标时间。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .constants import Y_THRESHOLD


def draw_group_key(visits: pd.DataFrame) -> pd.Series:
    """同孕妇、同检测抽血次数为一次抽血。缺失抽血次数时退回检测日期，再退回行号。"""
    keys = []
    draws = visits["draw"].to_numpy()
    dates = visits["exam_date"].to_numpy() if "exam_date" in visits.columns else np.array([pd.NaT] * len(visits))
    for index, (draw, date) in enumerate(zip(draws, dates)):
        if _is_present(draw):
            keys.append(f"draw:{int(float(draw))}")
        elif _is_present(date):
            keys.append(f"date:{pd.Timestamp(date).isoformat()}")
        else:
            keys.append(f"row:{index}")
    return pd.Series(keys, index=visits.index)


def collapse_draws(visits: pd.DataFrame, threshold: float = Y_THRESHOLD) -> pd.DataFrame:
    """把重复测序收成一次抽血。浓度取中位数，孕周取中位数。"""
    if visits.empty:
        return pd.DataFrame(columns=["woman_id", "draw_key", "ga_weeks", "y_median", "n_reps", "exam_date", "draw"])
    frame = visits.copy()
    frame["_draw_key"] = draw_group_key(frame)
    rows = []
    for (woman_id, draw_key), group in frame.groupby(["woman_id", "_draw_key"], sort=False):
        y_values = pd.to_numeric(group["y_frac"], errors="coerce")
        ga_values = pd.to_numeric(group["ga_weeks"], errors="coerce")
        if y_values.notna().sum() == 0 or ga_values.notna().sum() == 0:
            continue
        y_median = float(np.nanmedian(y_values.to_numpy(dtype=float)))
        ga_median = float(np.nanmedian(ga_values.to_numpy(dtype=float)))
        dates = group["exam_date"] if "exam_date" in group.columns else pd.Series(pd.NaT, index=group.index)
        draw_value = group["draw"].dropna()
        rows.append(
            {
                "woman_id": woman_id,
                "draw_key": draw_key,
                "ga_weeks": ga_median,
                "y_median": y_median,
                "reached": bool(y_median >= threshold),
                "n_reps": int(y_values.notna().sum()),
                "exam_date": dates.min() if dates.notna().any() else pd.NaT,
                "draw": float(draw_value.iloc[0]) if len(draw_value) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["woman_id", "draw_key", "ga_weeks", "y_median", "n_reps", "exam_date", "draw"])
    return pd.DataFrame(rows)


def hitting_from_draws(draws: pd.DataFrame, threshold: float = Y_THRESHOLD) -> Optional[dict]:
    """按孕周排序后的抽血序列构造首次达到阈值的删失区间。"""
    ordered = draws.dropna(subset=["ga_weeks", "y_median"]).copy()
    if ordered.empty:
        return None
    ordered = ordered.sort_values(["ga_weeks", "exam_date", "draw"], kind="mergesort")
    ga = ordered["ga_weeks"].to_numpy(dtype=float)
    y = ordered["y_median"].to_numpy(dtype=float)
    hit_at = np.flatnonzero(y >= threshold)
    if hit_at.size == 0:
        record = {
            "kind": "right",
            "left": float(ga[-1]),
            "right": float("inf"),
            "exact": False,
        }
    else:
        first = int(hit_at[0])
        if first == 0:
            record = {
                "kind": "left",
                "left": 0.0,
                "right": float(ga[0]),
                "exact": False,
            }
        else:
            left = float(ga[first - 1])
            right = float(ga[first])
            if right <= left:
                right = left + 1e-3
            record = {"kind": "interval", "left": left, "right": right, "exact": False}
    later = y[hit_at[0] + 1 :] if hit_at.size else np.array([])
    record.update(
        {
            "n_draws": int(len(ordered)),
            "ga_first": float(ga[0]),
            "ga_last": float(ga[-1]),
            "y_first": float(y[0]),
            "reverted": bool(hit_at.size and np.any(later < threshold)),
        }
    )
    return record


def build_woman_hits(visits: pd.DataFrame, threshold: float = Y_THRESHOLD) -> pd.DataFrame:
    """每位孕妇一行。右删失样本保留。"""
    if visits.empty:
        return pd.DataFrame()
    draws = collapse_draws(visits, threshold=threshold)
    first = _first_rows(visits)
    first_by_id = first.set_index("woman_id", drop=False)
    rows = []
    if draws.empty:
        return pd.DataFrame()
    for woman_id, group in draws.groupby("woman_id", sort=False):
        hit = hitting_from_draws(group, threshold=threshold)
        if hit is None:
            continue
        covariates = _covariates(first_by_id, woman_id)
        covariates.update(hit)
        covariates["woman_id"] = woman_id
        rows.append(covariates)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def replicate_sigma(visits: pd.DataFrame) -> float:
    """同一次抽血多次测序的 Y 浓度合并标准差。没有重复测序时返回 NaN。"""
    if "y_frac" not in visits.columns or visits.empty:
        return float("nan")
    frame = visits.copy()
    frame["_draw_key"] = draw_group_key(frame)
    variances = []
    weights = []
    for _, group in frame.groupby(["woman_id", "_draw_key"], sort=False):
        y = pd.to_numeric(group["y_frac"], errors="coerce").dropna().to_numpy(dtype=float)
        if y.size < 2:
            continue
        variances.append(float(np.var(y, ddof=1)))
        weights.append(y.size - 1)
    if not weights:
        return float("nan")
    return float(np.sqrt(np.average(np.asarray(variances), weights=np.asarray(weights, dtype=float))))


def _first_rows(visits: pd.DataFrame) -> pd.DataFrame:
    ordered = visits.sort_values(["woman_id", "exam_date", "draw"], kind="mergesort", na_position="last")
    return ordered.groupby("woman_id", sort=False, as_index=False).head(1).copy()


def _covariates(first_by_id: pd.DataFrame, woman_id) -> dict:
    if woman_id not in first_by_id.index:
        return {"bmi_first": np.nan, "age": np.nan, "height_cm": np.nan, "weight_first": np.nan}
    row = first_by_id.loc[woman_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    bmi = _scalar(row.get("bmi", np.nan))
    height = _scalar(row.get("height_cm", np.nan))
    weight = _scalar(row.get("weight_kg", np.nan))
    if not np.isfinite(bmi) and np.isfinite(height) and np.isfinite(weight) and height > 0:
        bmi = weight / (height / 100.0) ** 2
    return {
        "bmi_first": bmi,
        "age": _scalar(row.get("age", np.nan)),
        "height_cm": height,
        "weight_first": weight,
    }


def _scalar(value):
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_present(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        return True
    return True
