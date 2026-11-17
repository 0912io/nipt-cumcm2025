"""线性模型、聚类稳健标准误与随机截距混合模型。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .constants import BANNED_Y_PREDICTORS


def assert_formula_allowed(formula: str) -> None:
    for token in BANNED_Y_PREDICTORS:
        if token in formula:
            raise ValueError(f"主模型不能把 {token} 当作 Y 浓度的解释变量")


def fit_cluster_ols(formula: str, data: pd.DataFrame, group_column: str = "woman_id"):
    assert_formula_allowed(formula)
    model = smf.ols(formula, data=data)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": data[group_column]})
    return result


def fit_mixed_model(formula: str, data: pd.DataFrame, group_column: str = "woman_id", random_slope: str = None):
    assert_formula_allowed(formula)
    kwargs = {"groups": data[group_column]}
    if random_slope:
        kwargs["re_formula"] = f"~{random_slope}"
    model = smf.mixedlm(formula, data=data, **kwargs)
    result = model.fit(method="lbfgs", reml=True, maxiter=300, disp=False)
    return result


def fit_mixed_ml(formula: str, data: pd.DataFrame, group_column: str = "woman_id"):
    assert_formula_allowed(formula)
    model = smf.mixedlm(formula, data=data, groups=data[group_column])
    return model.fit(method="lbfgs", reml=False, maxiter=300, disp=False)


def likelihood_ratio(full, reduced) -> dict:
    statistic = float(2.0 * (full.llf - reduced.llf))
    df = int(len(full.fe_params) - len(reduced.fe_params))
    if df <= 0:
        return {"lr": statistic, "df": df, "p_value": float("nan")}
    return {"lr": statistic, "df": df, "p_value": float(stats.chi2.sf(statistic, df))}


def coefficient_table(result, kind: str) -> list[dict]:
    if kind == "ols":
        names = list(result.params.index)
        params = result.params
        bse = result.bse
        pvalues = result.pvalues
        conf = result.conf_int()
        rows = []
        for name in names:
            rows.append(
                {
                    "term": str(name),
                    "estimate": float(params[name]),
                    "std_error": float(bse[name]),
                    "p_value": float(pvalues[name]),
                    "ci_low": float(conf.loc[name, 0]),
                    "ci_high": float(conf.loc[name, 1]),
                }
            )
        return rows
    names = list(result.fe_params.index)
    conf = result.conf_int().loc[names]
    rows = []
    for name in names:
        rows.append(
            {
                "term": str(name),
                "estimate": float(result.fe_params[name]),
                "std_error": float(result.bse_fe[name]),
                "p_value": float(result.pvalues[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
            }
        )
    return rows


def variance_components(result) -> dict:
    random_sd = np.sqrt(np.asarray(result.cov_re, dtype=float))
    residual = float(np.sqrt(result.scale))
    return {
        "random_effect_sd": np.atleast_1d(np.diag(random_sd) if random_sd.ndim == 2 else random_sd).astype(float).tolist(),
        "random_effect_covariance": np.asarray(result.cov_re, dtype=float).tolist(),
        "residual_sd": residual,
        "converged": bool(result.converged),
    }


def design_vif(data: pd.DataFrame, columns: list[str]) -> dict:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    matrix = data[columns].astype(float).copy()
    matrix.insert(0, "const", 1.0)
    values = matrix.to_numpy(dtype=float)
    output = {}
    for index, name in enumerate(matrix.columns):
        if name == "const":
            continue
        try:
            output[name] = float(variance_inflation_factor(values, index))
        except Exception:
            output[name] = float("inf")
    return output
