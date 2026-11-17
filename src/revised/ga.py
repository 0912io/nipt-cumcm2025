"""孕周解析。统一为「周 + 天/7」，不把 ``11w+6`` 写成 83 天。"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

_GA_PATTERN = re.compile(r"^\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+)\s*)?\s*$")


def parse_gestational_age(value: Any) -> float:
    """把孕周解析为以周为单位的连续值。

    ``11w+6`` 为 ``11 + 6/7``。纯数字只在 (0, 45] 内被当作已经是「周」；
    历史脚本使用的天数编码（例如 83）会返回 NaN，避免两种单位混用。
    """
    if value is None:
        return float("nan")
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return float("nan")
    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
        weeks = float(value)
        if 0.0 < weeks <= 45.0:
            return weeks
        return float("nan")
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "nat"}:
        return float("nan")
    matched = _GA_PATTERN.match(text)
    if matched is None:
        return float("nan")
    weeks = int(matched.group(1))
    days = int(matched.group(2) or 0)
    if days < 0 or days >= 7:
        return float("nan")
    return weeks + days / 7.0


def parse_exam_date(value: Any) -> pd.Timestamp:
    """解析检测日期。兼容 Excel 时间戳与整数 YYYYMMDD。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, (int, np.integer)):
        return _from_yyyymmdd(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number >= 1e7:
            return _from_yyyymmdd(int(number))
        return pd.to_datetime(number, errors="coerce")
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return _from_yyyymmdd(int(text))
    return pd.to_datetime(text, errors="coerce")


def _from_yyyymmdd(value: int) -> pd.Timestamp:
    if 19000101 <= value <= 21001231:
        return pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")
    return pd.NaT


def parse_pregnancy_count(value: Any) -> float:
    """怀孕次数。字符串 ``≥3`` 记为下界 3，只用于描述，不进入主回归。"""
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return float("nan")
        if "≥" in text or ">=" in text:
            digits = re.sub(r"[^\d]", "", text)
            return float(digits) if digits else float("nan")
        try:
            return float(text)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
