"""工作表与列名。按表名区分男胎/女胎，不用 Y≤0.08。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

MALE_SHEET_NAMES = ("男胎检测数据",)
FEMALE_SHEET_NAMES = ("女胎检测数据",)

# 题面附录字母对应的实际表头（读取前会去掉首尾空格）。
COLUMN_MAP = {
    "序号": "row_id",
    "孕妇代码": "woman_id",
    "年龄": "age",
    "身高": "height_cm",
    "体重": "weight_kg",
    "末次月经": "lmp",
    "IVF妊娠": "ivf_raw",
    "检测日期": "exam_date_raw",
    "检测抽血次数": "draw",
    "检测孕周": "ga_raw",
    "孕妇BMI": "bmi",
    "原始读段数": "reads",
    "在参考基因组上比对的比例": "map_ratio",
    "重复读段的比例": "dup_ratio",
    "唯一比对的读段数": "unique_reads",
    "GC含量": "gc",
    "13号染色体的Z值": "z13",
    "18号染色体的Z值": "z18",
    "21号染色体的Z值": "z21",
    "X染色体的Z值": "zx",
    "Y染色体的Z值": "zy",
    "Y染色体浓度": "y_frac",
    "X染色体浓度": "x_frac",
    "13号染色体的GC含量": "gc13",
    "18号染色体的GC含量": "gc18",
    "21号染色体的GC含量": "gc21",
    "被过滤掉读段数的比例": "filter_ratio",
    "染色体的非整倍体": "aneuploidy",
    "怀孕次数": "pregnancy_raw",
    "生产次数": "births",
    "胎儿是否健康": "birth_healthy",
}

FEMALE_Y_Z_INDEX = 20
FEMALE_Y_FRACTION_INDEX = 21

NUMERIC_COLUMNS = (
    "age",
    "height_cm",
    "weight_kg",
    "draw",
    "bmi",
    "reads",
    "map_ratio",
    "dup_ratio",
    "unique_reads",
    "gc",
    "z13",
    "z18",
    "z21",
    "zx",
    "zy",
    "y_frac",
    "x_frac",
    "gc13",
    "gc18",
    "gc21",
    "filter_ratio",
    "births",
)


def sex_from_sheet_name(sheet_name: str) -> Optional[str]:
    """只看工作表名称。男胎表即使含有较低的 Y 浓度，也不会被改判为女胎。"""
    text = str(sheet_name).strip().lower()
    if text in {name.lower() for name in MALE_SHEET_NAMES} or "男胎" in str(sheet_name):
        return "male"
    if text in {name.lower() for name in FEMALE_SHEET_NAMES} or "女胎" in str(sheet_name):
        return "female"
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return None


def infer_sex(sheet_name: str, columns: Any, y_values: Any = None) -> str:
    """表名优先；表名无法识别时才看列结构。

    列结构规则看的是 Y 浓度列是否「整列为空」，不是 ``Y<=0.08``。
    因此男胎里偏低但非空的 Y 仍属于男胎。
    """
    named = sex_from_sheet_name(sheet_name)
    if named is not None:
        return named
    stripped = [_clean_header(column) for column in list(columns)]
    y_name_present = "Y染色体浓度" in stripped or "y_frac" in stripped
    if y_values is not None:
        y_array = np.asarray(y_values, dtype=float)
        y_all_missing = y_array.size == 0 or np.all(np.isnan(y_array))
    else:
        y_all_missing = not y_name_present
    if y_name_present and not y_all_missing:
        return "male"
    if y_all_missing:
        return "female"
    return "male"


def _clean_header(column: Any) -> str:
    if column is None:
        return ""
    text = str(column).strip()
    if text.startswith("Unnamed"):
        return ""
    return text


def standardize_frame(frame: pd.DataFrame, sex: str) -> pd.DataFrame:
    """统一列名。女胎表位置 20/21 的空表头记为 Y 的 Z 值与 Y 浓度（取值仍为空）。"""
    renamed = frame.copy()
    headers = [_clean_header(column) for column in renamed.columns]
    if sex == "female":
        if len(headers) > FEMALE_Y_Z_INDEX and headers[FEMALE_Y_Z_INDEX] == "":
            headers[FEMALE_Y_Z_INDEX] = "Y染色体的Z值"
        if len(headers) > FEMALE_Y_FRACTION_INDEX and headers[FEMALE_Y_FRACTION_INDEX] == "":
            headers[FEMALE_Y_FRACTION_INDEX] = "Y染色体浓度"
    mapped = []
    seen: dict[str, int] = {}
    for header in headers:
        name = COLUMN_MAP.get(header, header if header else "blank")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        mapped.append(name)
    renamed.columns = mapped
    renamed["sex"] = sex
    return renamed
