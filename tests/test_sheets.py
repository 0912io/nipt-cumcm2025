from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.revised.errors import WorkbookNotFoundError
from src.revised.load import load_workbook
from src.revised.run_analysis import main
from src.revised.schema import infer_sex, standardize_frame


HEADERS = [
    "序号",
    "孕妇代码",
    "年龄",
    "身高",
    "体重",
    "末次月经",
    "IVF妊娠",
    "检测日期",
    "检测抽血次数",
    "检测孕周",
    "孕妇BMI",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
    "Y染色体浓度",
    "X染色体浓度",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
    "染色体的非整倍体",
    "怀孕次数",
    "生产次数",
    "胎儿是否健康",
]


def _row(code, ga, y_value, pregnancy, aneuploidy=""):
    bmi = 70 / (1.65 ** 2)
    return [
        1,
        code,
        28,
        165,
        70,
        "2023-01-01",
        "自然受孕",
        20230401,
        1,
        ga,
        bmi,
        5000000,
        0.8,
        0.03,
        4000000,
        0.4,
        0.2,
        0.1,
        0.3,
        0.0,
        0.5 if y_value is not None else None,
        y_value,
        -0.01,
        0.4,
        0.4,
        0.4,
        0.02,
        aneuploidy,
        pregnancy,
        0,
        "是",
    ]


def test_sheet_order_trailing_space_and_low_y_males(tmp_path):
    male = pd.DataFrame([_row("A001", "11w+6", 0.02, 1), _row("A001", "16w", 0.03, "≥3", "T21")], columns=HEADERS)
    male = male.rename(columns={"唯一比对的读段数": "唯一比对的读段数  "})
    female = pd.DataFrame([_row("B001", "13w+1", None, 2), _row("B002", "15w", None, "≥3", "T18")], columns=HEADERS)
    path = tmp_path / "附件.xlsx"
    with pd.ExcelWriter(path) as writer:
        female.to_excel(writer, sheet_name="女胎检测数据", index=False)
        male.to_excel(writer, sheet_name="男胎检测数据", index=False)
    frames = load_workbook(path)
    assert list(frames) == ["female", "male"] or set(frames) == {"female", "male"}
    assert len(frames["female"]) == 2
    assert frames["female"]["y_frac"].isna().all()
    assert len(frames["male"]) == 2
    assert set(frames["male"]["y_frac"]) == {0.02, 0.03}
    assert (frames["male"]["y_frac"] <= 0.08).all()
    assert frames["male"]["sex"].eq("male").all()
    assert "unique_reads" in frames["male"].columns
    assert frames["male"]["ga_weeks"].iloc[0] == pytest.approx(11 + 6 / 7)
    assert set(frames["male"]["pregnancy_raw"].astype(str)) >= {"1", "≥3"} or "≥3" in set(frames["male"]["pregnancy_raw"].map(lambda v: str(v).strip()))


def test_empty_y_header_is_not_a_male_filter():
    columns = [f"c{i}" for i in range(31)]
    columns[20] = "Unnamed: 20"
    columns[21] = "Unnamed: 21"
    frame = pd.DataFrame([[None] * 31, [None] * 31], columns=columns)
    standardized = standardize_frame(frame, sex="female")
    assert standardized["y_frac"].isna().all()
    assert len(standardized) == 2
    low_y = np.array([0.01, 0.02, 0.05])
    assert infer_sex("男胎检测数据", ["Y染色体浓度"], low_y) == "male"
    assert infer_sex("女胎检测数据", ["Y染色体浓度"], np.array([np.nan, np.nan])) == "female"
    assert infer_sex("未命名", ["Y染色体浓度"], low_y) == "male"
    assert infer_sex("未命名", ["Unnamed: 21"], np.array([np.nan, np.nan])) == "female"


def test_missing_workbook_has_a_clear_error(tmp_path):
    missing = tmp_path / "附件.xlsx"
    with pytest.raises(WorkbookNotFoundError, match="附件.xlsx") as raised:
        load_workbook(missing)
    message = str(raised.value)
    assert "不要提交" in message
    assert main(["--workbook", str(missing)]) == 2


def test_gitignore_blocks_workbook():
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "*.xlsx" in text
    assert "附件.xlsx" in text
