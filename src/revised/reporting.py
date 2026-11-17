"""把修订分析的小表写到 results/revised（不入库）和 paper/tables（可入库）。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .load import repo_root


def output_directories():
    results = repo_root() / "results" / "revised"
    tables = repo_root() / "paper" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return results, tables


def figure_directories():
    results, tables = output_directories()
    folders = [results / "figures", tables.parent / "figures"]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def write_frame(frame: pd.DataFrame, name: str) -> None:
    results, tables = output_directories()
    for folder in (results, tables):
        frame.to_csv(folder / name, index=False, encoding="utf-8-sig")


def write_json(payload, name: str) -> None:
    results, tables = output_directories()
    text = json.dumps(to_builtin(payload), ensure_ascii=False, indent=2)
    for folder in (results, tables):
        (folder / name).write_text(text, encoding="utf-8")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def read_json(name: str):
    path = Path(output_directories()[1]) / name
    return json.loads(path.read_text(encoding="utf-8"))
