"""读取两张工作表，并整理成男胎就诊、女胎就诊和删失达标记录。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .censoring import build_woman_hits, replicate_sigma
from .constants import Y_THRESHOLD
from .errors import WorkbookNotFoundError, missing_workbook_message
from .ga import parse_exam_date, parse_gestational_age, parse_pregnancy_count
from .schema import infer_sex, standardize_frame

PathLike = Union[str, Path]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_workbook(path: Optional[PathLike] = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise WorkbookNotFoundError(missing_workbook_message(candidate))
        return candidate
    candidates = [repo_root() / "附件.xlsx", Path.cwd() / "附件.xlsx"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise WorkbookNotFoundError(missing_workbook_message(candidates[0]))


def load_workbook(path: Optional[PathLike] = None) -> dict[str, pd.DataFrame]:
    """按表名读取男胎表和女胎表。文件不存在时抛出 ``WorkbookNotFoundError``。"""
    workbook = resolve_workbook(path)
    excel = pd.ExcelFile(workbook)
    frames = {}
    for sheet_name in excel.sheet_names:
        raw = pd.read_excel(excel, sheet_name=sheet_name)
        sex = infer_sex(sheet_name, raw.columns, _y_preview(raw))
        frames[sex] = _enrich(standardize_frame(raw, sex=sex), sex=sex, sheet_name=sheet_name)
    if "male" not in frames or "female" not in frames:
        found = ", ".join(excel.sheet_names)
        raise ValueError(f"工作簿需要男胎与女胎两张表，当前读到：{found}")
    return frames


def prepare(path: Optional[PathLike] = None) -> dict:
    frames = load_workbook(path)
    male = frames["male"]
    female = frames["female"]
    hits = build_woman_hits(male, threshold=Y_THRESHOLD)
    notes = describe(male, female, hits)
    return {"male": male, "female": female, "hits": hits, "notes": notes, "path": str(resolve_workbook(path))}


def describe(male: pd.DataFrame, female: pd.DataFrame, hits: pd.DataFrame) -> dict:
    """只保存计数与范围，不保存原始行。"""
    male_ids = set(male["woman_id"].astype(str))
    female_ids = set(female["woman_id"].astype(str))
    return {
        "male_rows": int(len(male)),
        "female_rows": int(len(female)),
        "male_women": int(len(male_ids)),
        "female_women": int(len(female_ids)),
        "id_overlap": int(len(male_ids & female_ids)),
        "male_ga_parse_failures": _parse_failures(male),
        "female_ga_parse_failures": _parse_failures(female),
        "male_y": _numeric_summary(male["y_frac"]),
        "male_y_below_threshold": int((male["y_frac"] < Y_THRESHOLD).sum()),
        "male_y_at_or_above_threshold": int((male["y_frac"] >= Y_THRESHOLD).sum()),
        "male_x_negative": int((male["x_frac"] < 0).sum()),
        "female_x_negative": int((female["x_frac"] < 0).sum()),
        "female_y_all_missing": bool(male_column_all_missing(female, "y_frac")),
        "bmi_formula_max_abs_diff_male": _bmi_formula_gap(male),
        "bmi_formula_max_abs_diff_female": _bmi_formula_gap(female),
        "male_women_bmi_changes": _women_with_variation(male, "bmi"),
        "male_women_height_changes": _women_with_variation(male, "height_cm"),
        "male_age_below_20_rows": int((male["age"] < 20).sum()),
        "female_age_below_20_rows": int((female["age"] < 20).sum()),
        "replicate_draw_groups_male": _replicate_groups(male),
        "replicate_sigma_y": replicate_sigma(male),
        "ivf_male": _value_counts(male["ivf_raw"]),
        "ivf_female": _value_counts(female["ivf_raw"]),
        "pregnancy_male": _value_counts(male["pregnancy_raw"]),
        "pregnancy_female": _value_counts(female["pregnancy_raw"]),
        "ae_male": _value_counts(male["birth_healthy"]),
        "ae_female": _value_counts(female["birth_healthy"]),
        "ab_male": _ab_summary(male),
        "ab_female": _ab_summary(female),
        "censoring": _censoring_summary(hits),
        "hits_women": int(hits["woman_id"].nunique()) if len(hits) else 0,
    }


def male_column_all_missing(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return True
    return bool(frame[column].isna().all())


def _enrich(frame: pd.DataFrame, sex: str, sheet_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["sheet_name"] = sheet_name
    out["woman_id"] = out["woman_id"].astype(str).str.strip()
    for column in (
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
    ):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["ga_weeks"] = out["ga_raw"].map(parse_gestational_age) if "ga_raw" in out.columns else np.nan
    out["exam_date"] = out["exam_date_raw"].map(parse_exam_date) if "exam_date_raw" in out.columns else pd.NaT
    out["ivf_assisted"] = out["ivf_raw"].map(_ivf_assisted) if "ivf_raw" in out.columns else np.nan
    out["pregnancy_num"] = out["pregnancy_raw"].map(parse_pregnancy_count) if "pregnancy_raw" in out.columns else np.nan
    out["bmi_filled_by_formula"] = False
    missing_bmi = out["bmi"].isna() & out["height_cm"].notna() & out["weight_kg"].notna() & (out["height_cm"] > 0)
    out.loc[missing_bmi, "bmi"] = out.loc[missing_bmi, "weight_kg"] / (out.loc[missing_bmi, "height_cm"] / 100.0) ** 2
    out.loc[missing_bmi, "bmi_filled_by_formula"] = True
    flags = out["aneuploidy"].map(_ab_flags) if "aneuploidy" in out.columns else [{}] * len(out)
    flag_frame = pd.DataFrame(list(flags), index=out.index)
    for column in ("ab_any", "ab_t13", "ab_t18", "ab_t21"):
        out[column] = flag_frame[column] if column in flag_frame.columns else 0
    out["log_reads"] = np.where(out["reads"] > 0, np.log(out["reads"]), np.nan)
    return out


def _y_preview(raw: pd.DataFrame):
    for column in raw.columns:
        if str(column).strip() == "Y染色体浓度":
            return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
    if raw.shape[1] > 21 and str(raw.columns[21]).startswith("Unnamed"):
        return np.full(len(raw), np.nan)
    return None


def _ivf_assisted(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    text = str(value).strip()
    if text == "自然受孕":
        return 0.0
    if any(token in text for token in ("IVF", "IUI", "试管", "人工")):
        return 1.0
    return float("nan")


def _ab_flags(value) -> dict:
    if value is None:
        return {"ab_any": 0, "ab_t13": 0, "ab_t18": 0, "ab_t21": 0}
    try:
        if pd.isna(value):
            return {"ab_any": 0, "ab_t13": 0, "ab_t18": 0, "ab_t21": 0}
    except TypeError:
        pass
    text = str(value).strip().upper()
    if text == "" or text == "NAN":
        return {"ab_any": 0, "ab_t13": 0, "ab_t18": 0, "ab_t21": 0}
    return {
        "ab_any": 1,
        "ab_t13": int("T13" in text),
        "ab_t18": int("T18" in text),
        "ab_t21": int("T21" in text),
    }


def _parse_failures(frame: pd.DataFrame) -> int:
    raw = frame["ga_raw"]
    parsed = frame["ga_weeks"]
    present = raw.notna() & raw.astype(str).str.strip().ne("")
    return int((present & parsed.isna()).sum())


def _numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _bmi_formula_gap(frame: pd.DataFrame) -> float:
    comparable = frame["bmi"].notna() & frame["height_cm"].notna() & frame["weight_kg"].notna()
    comparable = comparable & ~frame["bmi_filled_by_formula"] & (frame["height_cm"] > 0)
    if not comparable.any():
        return float("nan")
    formula = frame.loc[comparable, "weight_kg"] / (frame.loc[comparable, "height_cm"] / 100.0) ** 2
    return float(np.max(np.abs(formula.to_numpy(dtype=float) - frame.loc[comparable, "bmi"].to_numpy(dtype=float))))


def _women_with_variation(frame: pd.DataFrame, column: str) -> int:
    rounded = frame[column].round(3)
    nunq = rounded.groupby(frame["woman_id"]).nunique(dropna=True)
    return int((nunq > 1).sum())


def _replicate_groups(frame: pd.DataFrame) -> int:
    from .censoring import draw_group_key

    keys = draw_group_key(frame)
    sizes = frame.groupby([frame["woman_id"], keys], sort=False).size()
    return int((sizes > 1).sum())


def _value_counts(series: pd.Series) -> dict:
    counts = series.fillna("空白").astype(str).str.strip().replace({"": "空白"}).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _ab_summary(frame: pd.DataFrame) -> dict:
    return {
        "blank_rows": int((frame["ab_any"] == 0).sum()),
        "any_rows": int((frame["ab_any"] == 1).sum()),
        "t13_rows": int(frame["ab_t13"].sum()),
        "t18_rows": int(frame["ab_t18"].sum()),
        "t21_rows": int(frame["ab_t21"].sum()),
        "women_ever_any": int(frame.loc[frame["ab_any"] == 1, "woman_id"].nunique()),
        "women_never": int(frame.loc[~frame["woman_id"].isin(frame.loc[frame["ab_any"] == 1, "woman_id"]), "woman_id"].nunique()),
        "women_ab_changes": _women_with_variation(frame.assign(ab_any_num=frame["ab_any"]), "ab_any_num"),
        "ae_cross": _ae_cross(frame),
    }


def _ae_cross(frame: pd.DataFrame) -> dict:
    ae = frame["birth_healthy"].fillna("空白").astype(str).str.strip()
    positive = frame["ab_any"].astype(int)
    table = pd.crosstab(positive, ae)
    return {f"ab{int(ab)}|{column}": int(table.loc[ab, column]) for ab in table.index for column in table.columns}


def _censoring_summary(hits: pd.DataFrame) -> dict:
    if hits is None or len(hits) == 0:
        return {}
    counts = hits["kind"].value_counts().to_dict()
    return {
        "left": int(counts.get("left", 0)),
        "interval": int(counts.get("interval", 0)),
        "right": int(counts.get("right", 0)),
        "reverted_after_hit": int(hits["reverted"].sum()) if "reverted" in hits.columns else 0,
    }
