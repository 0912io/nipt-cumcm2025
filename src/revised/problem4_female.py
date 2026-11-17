"""问题四：只使用女胎表，以 AB 列按染色体作标签。

经典基线是 |Z|>=3。主模型是逻辑回归。标准化、阈值和质量分层都只在训练折上完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .models import design_vif
from .validation import (
    binary_rates,
    choose_threshold,
    confusion_counts,
    fit_transform_train_only,
    iter_group_kfold,
    negative_score_thresholds,
    zscore_calls,
)

CHROMOSOMES = {
    "T13": {"label": "ab_t13", "z": "z13", "gc": "gc13"},
    "T18": {"label": "ab_t18", "z": "z18", "gc": "gc18"},
    "T21": {"label": "ab_t21", "z": "z21", "gc": "gc21"},
}
SHARED_FEATURES = ["gc", "log_reads", "map_ratio", "dup_ratio", "unique_reads", "filter_ratio", "bmi", "ga_weeks", "zx", "x_frac"]


def run_problem4(visits: pd.DataFrame, seed: int = 2025) -> dict:
    """女胎就诊记录。标签是本次检测的 AB，不是 AE。"""
    if "ab_any" not in visits.columns:
        raise ValueError("女胎数据缺少 AB 标签")
    per_chromosome = {}
    for name, spec in CHROMOSOMES.items():
        per_chromosome[name] = _run_chromosome(visits, name, spec["label"], spec["z"], spec["gc"], seed)
    per_chromosome["ANY"] = _run_any(visits, seed)
    ae_positive = int((visits["birth_healthy"].astype(str).str.strip() != "是").sum()) if "birth_healthy" in visits.columns else None
    return {
        "n_rows": int(len(visits)),
        "n_women": int(visits["woman_id"].nunique()),
        "ae_non_yes_rows": ae_positive,
        "z_by_ab": z_by_ab(visits),
        "chromosomes": per_chromosome,
        "label": "AB 列，按染色体拆开；空白为该染色体阴性",
        "female_rule": "工作表名称，不用 Y<=0.08",
    }


def _run_chromosome(visits, name, label, z_column, gc_column, seed):
    features = [z_column, gc_column, *SHARED_FEATURES]
    model_frame = _complete_case(visits, features, label)
    baseline_all = _z_baseline(visits[z_column], visits[label], f"{name} 全部有 Z 值的行")
    baseline_sample = _z_baseline(model_frame[z_column], model_frame[label], f"{name} 与逻辑回归相同的完整样本")
    cv = _cv_logistic(model_frame, features, label, z_column, gc_column, seed)
    heldout = _illustrative_split(model_frame, features, label, z_column, gc_column, seed)
    inference = _inferential_logit(model_frame, features, label)
    return {
        "n_rows": int(len(model_frame)),
        "n_women": int(model_frame["woman_id"].nunique()) if len(model_frame) else 0,
        "n_positive_rows": int(model_frame[label].sum()) if len(model_frame) else 0,
        "n_positive_women": int(model_frame.loc[model_frame[label] == 1, "woman_id"].nunique()) if len(model_frame) else 0,
        "baseline_all_finite_z": baseline_all,
        "baseline_model_sample": baseline_sample,
        "cv": cv,
        "illustrative_split": heldout,
        "inference": inference,
    }


def _run_any(visits, seed):
    features = ["z13", "z18", "z21", *SHARED_FEATURES]
    model_frame = _complete_case(visits, features, "ab_any")
    z_any = _any_z_call(visits)
    baseline_all = _attach_rates(visits["ab_any"].to_numpy(dtype=int), z_any, "任一 |Z|>=3，全部行")
    z_sample = _any_z_call(model_frame)
    baseline_sample = _attach_rates(model_frame["ab_any"].to_numpy(dtype=int), z_sample, "任一 |Z|>=3，完整样本")
    cv = _cv_logistic(model_frame, features, "ab_any", "z21", "gc", seed, zone_z=None)
    inference = _inferential_logit(model_frame, features, "ab_any")
    return {
        "n_rows": int(len(model_frame)),
        "n_women": int(model_frame["woman_id"].nunique()) if len(model_frame) else 0,
        "n_positive_rows": int(model_frame["ab_any"].sum()) if len(model_frame) else 0,
        "baseline_all_finite_z": baseline_all,
        "baseline_model_sample": baseline_sample,
        "cv": cv,
        "inference": inference,
        "note": "总异常模型同时放入三条染色体的 Z 值。分区阈值仍按各染色体分别报告。",
    }


def _complete_case(visits, features, label) -> pd.DataFrame:
    columns = ["woman_id", label, *features]
    frame = visits[columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    frame[label] = frame[label].astype(int)
    return frame


def _z_baseline(z_values, labels, note: str) -> dict:
    z = np.asarray(z_values, dtype=float)
    y = np.asarray(labels, dtype=int)
    finite = np.isfinite(z)
    return _attach_rates(y[finite], zscore_calls(z[finite], 3.0), note)


def _any_z_call(frame: pd.DataFrame) -> np.ndarray:
    calls = np.zeros(len(frame), dtype=int)
    for column in ("z13", "z18", "z21"):
        calls = np.maximum(calls, zscore_calls(frame[column].to_numpy(dtype=float), 3.0))
    return calls


def _attach_rates(y_true, y_pred, note: str) -> dict:
    counts = confusion_counts(y_true, y_pred)
    rates = binary_rates(counts)
    rates.update(counts)
    rates["note"] = note
    rates["n"] = int(len(np.asarray(y_true)))
    return rates


def _cv_logistic(frame, features, label, z_column, gc_column, seed, zone_z="use"):
    if frame["woman_id"].nunique() < 4 or frame[label].nunique() < 2:
        return {"error": "分组交叉验证样本或阳性类别不足"}
    splits = iter_group_kfold(frame["woman_id"].to_numpy(), n_splits=5)
    pooled_y = []
    pooled_pred = []
    pooled_proba = []
    fold_auc = []
    zone_positive_pred = []
    zone_flag_pred = []
    zone_y = []
    for train_index, test_index in splits:
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        if train[label].nunique() < 2 or test[label].nunique() < 1:
            continue
        x_train, x_test, _ = fit_transform_train_only(train[features], test[features])
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
        classifier.fit(x_train, train[label].to_numpy(dtype=int))
        train_proba = classifier.predict_proba(x_train)[:, 1]
        test_proba = classifier.predict_proba(x_test)[:, 1]
        threshold = choose_threshold(train[label].to_numpy(dtype=int), train_proba, target_sensitivity=0.95)
        prediction = (test_proba >= threshold).astype(int)
        pooled_y.append(test[label].to_numpy(dtype=int))
        pooled_pred.append(prediction)
        pooled_proba.append(test_proba)
        if test[label].nunique() == 2:
            fold_auc.append(float(roc_auc_score(test[label], test_proba)))
        if zone_z == "use" and z_column in train.columns:
            z_pred_pos, z_pred_flag, y_zone = _zone_predictions(train, test, z_column, gc_column, label)
            zone_positive_pred.append(z_pred_pos)
            zone_flag_pred.append(z_pred_flag)
            zone_y.append(y_zone)
    if not pooled_y:
        return {"error": "没有可用的训练折"}
    y = np.concatenate(pooled_y)
    pred = np.concatenate(pooled_pred)
    proba = np.concatenate(pooled_proba)
    summary = _attach_rates(y, pred, "分组交叉验证合并测试折；阈值在训练折上按目标敏感度 0.95 选择")
    summary["auc"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan")
    summary["auc_fold_std"] = float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else float("nan")
    summary["brier"] = float(brier_score_loss(y, proba))
    summary["n_folds_with_auc"] = int(len(fold_auc))
    if zone_y:
        y_zone = np.concatenate(zone_y)
        summary["zone_positive"] = _attach_rates(y_zone, np.concatenate(zone_positive_pred), "训练折特异度 99.5% 的阳性区")
        summary["zone_flag"] = _attach_rates(y_zone, np.concatenate(zone_flag_pred), "训练折特异度 99% 以上的阳性区或观察区")
    return summary


def _zone_predictions(train, test, z_column, gc_column, label):
    qi_train = quality_index(train, train, gc_column)
    qi_test = quality_index(train, test, gc_column)
    q20, q80 = np.quantile(qi_train, [0.2, 0.8])
    train_strata = _strata(qi_train, q20, q80)
    test_strata = _strata(qi_test, q20, q80)
    positive = np.zeros(len(test), dtype=int)
    flag = np.zeros(len(test), dtype=int)
    y_train = train[label].to_numpy(dtype=int)
    z_train = train[z_column].to_numpy(dtype=float)
    z_test = test[z_column].to_numpy(dtype=float)
    for stratum in ("marginal", "typical", "good"):
        train_mask = train_strata == stratum
        negatives = int(np.sum((y_train == 0) & train_mask & np.isfinite(z_train)))
        if negatives >= 20:
            z_low, z_high = negative_score_thresholds(z_train[train_mask], y_train[train_mask])
        else:
            z_low, z_high = negative_score_thresholds(z_train, y_train)
        test_mask = test_strata == stratum
        positive[test_mask] = (z_test[test_mask] >= z_high).astype(int)
        flag[test_mask] = (z_test[test_mask] >= z_low).astype(int)
    return positive, flag, test[label].to_numpy(dtype=int)


def _illustrative_split(frame, features, label, z_column, gc_column, seed):
    if frame["woman_id"].nunique() < 8 or frame[label].nunique() < 2:
        return {"error": "留出样本不足"}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_index, test_index = next(splitter.split(frame, frame[label], frame["woman_id"]))
    train = frame.iloc[train_index]
    test = frame.iloc[test_index]
    if train[label].nunique() < 2:
        return {"error": "训练孕妇中没有两类标签"}
    thresholds = []
    qi_train = quality_index(train, train, gc_column)
    qi_test = quality_index(train, test, gc_column)
    q20, q80 = np.quantile(qi_train, [0.2, 0.8])
    train_strata = _strata(qi_train, q20, q80)
    test_strata = _strata(qi_test, q20, q80)
    y_train = train[label].to_numpy(dtype=int)
    z_train = train[z_column].to_numpy(dtype=float)
    z_test = test[z_column].to_numpy(dtype=float)
    positive = np.zeros(len(test), dtype=int)
    flag = np.zeros(len(test), dtype=int)
    for stratum in ("marginal", "typical", "good"):
        train_mask = train_strata == stratum
        negatives = int(np.sum((y_train == 0) & train_mask & np.isfinite(z_train)))
        fallback = negatives < 20
        if fallback:
            z_low, z_high = negative_score_thresholds(z_train, y_train)
        else:
            z_low, z_high = negative_score_thresholds(z_train[train_mask], y_train[train_mask])
        test_mask = test_strata == stratum
        positive[test_mask] = (z_test[test_mask] >= z_high).astype(int)
        flag[test_mask] = (z_test[test_mask] >= z_low).astype(int)
        thresholds.append(
            {
                "stratum": stratum,
                "z_low_specificity_99": z_low,
                "z_high_specificity_995": z_high,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "used_overall_negatives": bool(fallback),
            }
        )
    x_train, x_test, _ = fit_transform_train_only(train[features], test[features])
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    classifier.fit(x_train, y_train)
    train_proba = classifier.predict_proba(x_train)[:, 1]
    test_proba = classifier.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_train, train_proba, 0.95)
    prediction = (test_proba >= threshold).astype(int)
    return {
        "probability_threshold_train": threshold,
        "test_logistic": _attach_rates(test[label].to_numpy(dtype=int), prediction, "单次分组留出的测试孕妇"),
        "test_zone_positive": _attach_rates(test[label].to_numpy(dtype=int), positive, "单次分组留出，阳性区"),
        "test_zone_flag": _attach_rates(test[label].to_numpy(dtype=int), flag, "单次分组留出，阳性区或观察区"),
        "thresholds": thresholds,
        "note": "阈值来自训练孕妇。这一次留出只说明规则长什么样，性能以分组交叉验证为准。",
    }


def _inferential_logit(frame, features, label) -> dict:
    if len(frame) < 20 or frame[label].nunique() < 2:
        return {"error": "无法拟合逻辑回归"}
    y = frame[label].to_numpy(dtype=int)
    x = sm.add_constant(frame[features].astype(float), has_constant="add")
    codes, _ = pd.factorize(frame["woman_id"])
    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": codes})
    except Exception as exc:
        return {"error": str(exc)}
    conf = result.conf_int()
    rows = []
    for name in result.params.index:
        rows.append(
            {
                "term": str(name),
                "estimate": float(result.params[name]),
                "std_error": float(result.bse[name]),
                "p_value": float(result.pvalues[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
                "odds_ratio": float(np.exp(result.params[name])),
            }
        )
    return {
        "coefficients": rows,
        "vif": design_vif(frame, features),
        "n_rows": int(len(frame)),
        "n_women": int(frame["woman_id"].nunique()),
        "note": "全样本、未标准化的聚类稳健逻辑回归，只用于系数解释，不用于挑选阈值或报告测试性能。",
    }


def quality_index(reference: pd.DataFrame, target: pd.DataFrame, gc_column: str) -> np.ndarray:
    """质量指数只用参考样本（训练折）的中位数和 MAD。"""
    pieces = []
    for column, sign in (
        ("map_ratio", 1.0),
        ("unique_reads", 1.0),
        ("log_reads", 1.0),
        ("dup_ratio", -1.0),
        ("filter_ratio", -1.0),
    ):
        pieces.append(sign * _robust_z_from(reference[column].to_numpy(dtype=float), target[column].to_numpy(dtype=float)))
    gc_reference = reference[gc_column].to_numpy(dtype=float)
    gc_target = target[gc_column].to_numpy(dtype=float)
    center = float(np.median(gc_reference))
    pieces.append(-_robust_z_from(np.abs(gc_reference - center), np.abs(gc_target - center)))
    return np.sum(np.vstack(pieces), axis=0)


def _robust_z_from(reference, target) -> np.ndarray:
    med = float(np.median(reference))
    mad = float(np.median(np.abs(reference - med)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        return np.zeros(len(target), dtype=float)
    return (np.asarray(target, dtype=float) - med) / scale


def z_by_ab(visits: pd.DataFrame) -> list[dict]:
    """女胎 Z 值按本次 AB 标签拆开的分布。只描述，不另设阈值。"""
    rows = []
    specs = (("T13", "z13", "ab_t13"), ("T18", "z18", "ab_t18"), ("T21", "z21", "ab_t21"))
    for name, z_column, label in specs:
        if z_column not in visits.columns or label not in visits.columns:
            continue
        z = visits[z_column].to_numpy(dtype=float)
        y = visits[label].to_numpy(dtype=int)
        for flag, text in ((0, "阴性"), (1, "阳性")):
            values = z[(y == flag) & np.isfinite(z)]
            rows.append(_z_row(name, text, values))
    return rows


def _z_row(chromosome: str, ab: str, values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "chromosome": chromosome,
            "ab": ab,
            "n": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "n_abs_ge_3": 0,
        }
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "chromosome": chromosome,
        "ab": ab,
        "n": int(values.size),
        "min": float(np.min(values)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(np.max(values)),
        "n_abs_ge_3": int(np.sum(np.abs(values) >= 3)),
    }


def _strata(qi, q20, q80) -> np.ndarray:
    labels = np.array(["typical"] * len(qi), dtype=object)
    labels[qi <= q20] = "marginal"
    labels[qi >= q80] = "good"
    return labels
