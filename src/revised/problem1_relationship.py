"""问题一：Y 浓度与孕周、BMI 的关系。

主结论是聚类稳健的系数和混合模型的方差分量。岭回归只作为敏感性分析。
就诊级 BMI（当次官方 K 列）用于浓度模型；分组用的首次 BMI 不在这里替换它。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from .models import (
    coefficient_table,
    design_vif,
    fit_cluster_ols,
    fit_mixed_ml,
    fit_mixed_model,
    likelihood_ratio,
    variance_components,
)
from .validation import fit_transform_train_only, iter_group_kfold

MAIN_FORMULAS = {
    "M0_仅孕周": "y ~ ga_c",
    "M1_孕周加BMI": "y ~ ga_c + bmi_c",
    "M2_含交互": "y ~ ga_c + bmi_c + ga_bmi",
    "M3_主模型": "y ~ ga_c + bmi_c + ga_bmi + age_c + ivf_assisted + gc_c + map_c",
}
RIDGE_FEATURES = ["ga_c", "bmi_c", "ga_bmi", "age_c", "ivf_assisted", "gc_c", "map_c"]
CENTER_COLUMNS = ("ga", "bmi", "age", "gc", "map_ratio")


def build_concentration_frame(visits: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "woman_id",
        "y_frac",
        "ga_weeks",
        "bmi",
        "age",
        "ivf_assisted",
        "gc",
        "map_ratio",
        "height_cm",
        "weight_kg",
    ]
    frame = visits[columns].copy()
    frame = frame.rename(columns={"y_frac": "y", "ga_weeks": "ga"})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.reset_index(drop=True)


def run_problem1(visits: pd.DataFrame, n_boot: int = 999, seed: int = 2025) -> dict:
    raw = build_concentration_frame(visits)
    needed = ["y", "ga", "bmi", "age", "ivf_assisted", "gc", "map_ratio", "woman_id"]
    sample = raw.dropna(subset=needed).reset_index(drop=True)
    centered, centers = center_frame(sample)
    correlations = correlation_block(sample, n_boot=n_boot, seed=seed)
    ols_tables = []
    cv_rows = []
    fitted = {}
    for name, formula in MAIN_FORMULAS.items():
        result = fit_cluster_ols(formula, centered)
        fitted[name] = result
        for row in coefficient_table(result, "ols"):
            row["model"] = name
            row["fit"] = "cluster_ols"
            ols_tables.append(row)
        scores = cv_ols_r2(sample, formula)
        cv_rows.append(
            {
                "model": name,
                "fit": "OLS固定效应",
                "in_sample_r2": float(result.rsquared),
                "group_cv_r2_mean": float(np.mean(scores)),
                "group_cv_r2_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
                "n_rows": int(len(sample)),
                "n_women": int(sample["woman_id"].nunique()),
            }
        )
    mixed = fit_mixed_models(centered)
    ridge_scores = cv_ridge(sample, seed=seed)
    cv_rows.append(
        {
            "model": "岭回归敏感性",
            "fit": "RidgeCV_仅敏感性",
            "in_sample_r2": float("nan"),
            "group_cv_r2_mean": float(np.mean(ridge_scores)) if ridge_scores else float("nan"),
            "group_cv_r2_std": float(np.std(ridge_scores, ddof=1)) if len(ridge_scores) > 1 else float("nan"),
            "n_rows": int(len(sample)),
            "n_women": int(sample["woman_id"].nunique()),
        }
    )
    vif_columns = ["ga_c", "bmi_c", "age_c", "ivf_assisted", "gc_c", "map_c"]
    return {
        "n_rows_input": int(len(raw)),
        "n_rows": int(len(sample)),
        "n_women": int(sample["woman_id"].nunique()),
        "dropped_rows": int(len(raw) - len(sample)),
        "centering": centers,
        "correlations": correlations,
        "coefficients": ols_tables,
        "cv": cv_rows,
        "vif_additive": design_vif(centered, vif_columns),
        "mixed": mixed,
        "bmi_definition": "就诊级官方BMI（K列）；缺失时才用当次身高体重公式补齐",
    }


def center_frame(frame: pd.DataFrame, centers: dict = None):
    out = frame.copy()
    known = dict(centers or {})
    for column in CENTER_COLUMNS:
        if column not in out.columns:
            continue
        center = known.get(column)
        if center is None:
            center = float(out[column].mean())
            known[column] = center
        out[f"{column}_c"] = out[column] - center
    out["ga_bmi"] = out["ga_c"] * out["bmi_c"]
    # map_ratio 中心化后的列名与公式一致。
    if "map_ratio_c" in out.columns:
        out["map_c"] = out["map_ratio_c"]
    return out, known


def cv_ols_r2(sample: pd.DataFrame, formula: str) -> list[float]:
    scores = []
    for train_index, test_index in iter_group_kfold(sample["woman_id"].to_numpy(), n_splits=5):
        train, train_center = center_frame(sample.iloc[train_index])
        test, _ = center_frame(sample.iloc[test_index], train_center)
        result = fit_cluster_ols(formula, train)
        predicted = result.predict(test)
        scores.append(float(r2_score(test["y"], predicted)))
    return scores


def cv_ridge(sample: pd.DataFrame, seed: int) -> list[float]:
    del seed
    scores = []
    alphas = np.logspace(-3, 3, 25)
    for train_index, test_index in iter_group_kfold(sample["woman_id"].to_numpy(), n_splits=5):
        train, train_center = center_frame(sample.iloc[train_index])
        test, _ = center_frame(sample.iloc[test_index], train_center)
        x_train, x_test, _ = fit_transform_train_only(train[RIDGE_FEATURES], test[RIDGE_FEATURES])
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(x_train, train["y"].to_numpy(dtype=float))
        predicted = model.predict(x_test)
        scores.append(float(r2_score(test["y"], predicted)))
    return scores


def fit_mixed_models(centered: pd.DataFrame) -> dict:
    output = {"models": {}, "lrt": {}, "random_slope": {}}
    ml_fits = {}
    for name in ("M1_孕周加BMI", "M2_含交互", "M3_主模型"):
        formula = MAIN_FORMULAS[name]
        try:
            reml = fit_mixed_model(formula, centered)
            output["models"][name] = {
                "coefficients": coefficient_table(reml, "mixed"),
                "variance": variance_components(reml),
                "marginal_r2": marginal_r2(reml, centered["y"].to_numpy(dtype=float)),
            }
        except Exception as exc:
            output["models"][name] = {"error": str(exc)}
        try:
            ml_fits[name] = fit_mixed_ml(formula, centered)
        except Exception as exc:
            ml_fits[name] = None
            output["lrt"][name] = {"error": str(exc)}
    if ml_fits.get("M1_孕周加BMI") is not None and ml_fits.get("M2_含交互") is not None:
        output["lrt"]["M2_vs_M1"] = likelihood_ratio(ml_fits["M2_含交互"], ml_fits["M1_孕周加BMI"])
    if ml_fits.get("M2_含交互") is not None and ml_fits.get("M3_主模型") is not None:
        output["lrt"]["M3_vs_M2"] = likelihood_ratio(ml_fits["M3_主模型"], ml_fits["M2_含交互"])
    try:
        slope = fit_mixed_model(MAIN_FORMULAS["M3_主模型"], centered, random_slope="ga_c")
        output["random_slope"] = {
            "coefficients": coefficient_table(slope, "mixed"),
            "variance": variance_components(slope),
        }
    except Exception as exc:
        output["random_slope"] = {"converged": False, "message": str(exc)}
    return output


def marginal_r2(result, y) -> float:
    exog = np.asarray(result.model.exog, dtype=float)
    fitted = exog @ np.asarray(result.fe_params, dtype=float)
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    if total <= 0:
        return float("nan")
    return float(1.0 - residual / total)


def correlation_block(sample: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    pairs = [("y", "ga"), ("y", "bmi"), ("y", "age"), ("ga", "bmi")]
    rows = []
    for left, right in pairs:
        row = {"x": right, "y": left}
        for method in ("pearson", "spearman", "kendall"):
            estimate, low, high = cluster_bootstrap_corr(sample, left, right, method, n_boot, seed)
            row[f"{method}"] = estimate
            row[f"{method}_ci_low"] = low
            row[f"{method}_ci_high"] = high
        rows.append(row)
    woman = sample.groupby("woman_id", sort=False)[["y", "ga", "bmi", "age"]].mean()
    for left, right in (("y", "ga"), ("y", "bmi"), ("y", "age")):
        estimate, p_value = permutation_spearman(woman[right].to_numpy(), woman[left].to_numpy(), n_boot, seed)
        rows.append(
            {
                "x": right,
                "y": f"woman_mean_{left}",
                "spearman": estimate,
                "permutation_p": p_value,
                "n_women": int(len(woman)),
            }
        )
    return rows


def cluster_bootstrap_corr(frame, y_col, x_col, method, n_boot, seed):
    indexed = frame.reset_index(drop=True)
    buckets = indexed.groupby("woman_id", sort=False).indices
    ids = list(buckets)
    y = indexed[y_col].to_numpy(dtype=float)
    x = indexed[x_col].to_numpy(dtype=float)
    observed = _corr(x, y, method)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        chosen = rng.integers(0, len(ids), size=len(ids))
        positions = np.concatenate([buckets[ids[item]] for item in chosen])
        draws[index] = _corr(x[positions], y[positions], method)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(low), float(high)


def permutation_spearman(x, y, n_perm, seed):
    observed = _corr(x, y, "spearman")
    rng = np.random.default_rng(seed + 17)
    count = 0
    for _ in range(n_perm):
        statistic = _corr(rng.permutation(x), y, "spearman")
        if abs(statistic) >= abs(observed) - 1e-15:
            count += 1
    return float(observed), float((count + 1) / (n_perm + 1))


def _corr(x, y, method: str) -> float:
    from scipy import stats

    if method == "pearson":
        result = stats.pearsonr(x, y)
    elif method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        result = stats.kendalltau(x, y)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])
