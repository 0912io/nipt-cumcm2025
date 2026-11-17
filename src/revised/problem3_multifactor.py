"""问题三：在 BMI 分组中纳入年龄、身高，并用个体首次达标概率定义组达标比例。

预先登记：主模型使用 BMI + 年龄 + 身高，不同时放入体重。
敏感性模型改为身高 + 体重 + 年龄，不再放入 BMI。
分组仍是连续的 BMI 区间，每组一个时点。这不是复检策略。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .censoring import build_woman_hits
from .constants import K_CANDIDATES, N_BOOT_TIME, N_MIN_GROUP, N_NOISE, PRIMARY_LAMBDA, RANDOM_SEED, Y_THRESHOLD
from .grouping import assign_groups, best_ordered_partition, boundary_indices, cuts_from_edges, interval_label
from .models import coefficient_table, design_vif, fit_mixed_model, variance_components
from .problem1_relationship import build_concentration_frame, center_frame
from .problem2_timing import _analysis_frame, _earliest_observed
from .risk import decision_objective, minimize_objective, time_grid, time_risk
from .survival import fit_interval_aft, success_probability
from .validation import iter_group_kfold

PRIMARY_COVARIATES = ["bmi_first", "age", "height_cm"]
SENSITIVITY_COVARIATES = ["age", "height_cm", "weight_first"]


def run_problem3(
    visits: pd.DataFrame,
    hits: pd.DataFrame = None,
    n_boot: int = N_BOOT_TIME,
    n_noise: int = N_NOISE,
    seed: int = RANDOM_SEED,
    n_min: int = N_MIN_GROUP,
) -> dict:
    records = hits if hits is not None else build_woman_hits(visits, threshold=Y_THRESHOLD)
    data = _analysis_frame(records).dropna(subset=PRIMARY_COVARIATES + SENSITIVITY_COVARIATES).reset_index(drop=True)
    data = data.sort_values(["bmi_first", "woman_id"], kind="mergesort").reset_index(drop=True)
    primary = fit_interval_aft(
        data[PRIMARY_COVARIATES].to_numpy(dtype=float),
        data["left"],
        data["right"],
        data["kind"].to_numpy(),
        PRIMARY_COVARIATES,
    )
    sensitivity = fit_interval_aft(
        data[SENSITIVITY_COVARIATES].to_numpy(dtype=float),
        data["left"],
        data["right"],
        data["kind"].to_numpy(),
        SENSITIVITY_COVARIATES,
    )
    chosen_k, cv_scores = _select_k(data, PRIMARY_COVARIATES, n_min)
    alternatives = []
    groups = []
    if primary.success:
        for k in K_CANDIDATES:
            groups_k = _groups_from_fit(data, primary, k, n_min)
            alternatives.append({"k": int(k), "groups": [{key: value for key, value in group.items() if key != "_mask"} for group in groups_k]})
            if int(k) == int(chosen_k):
                groups = groups_k
    _bootstrap_groups(data, primary, groups, n_boot, seed)
    _noise_groups(visits, data, groups, n_noise, seed + 1, n_min)
    lmm = _lmm_sensitivity(visits, data, groups)
    curves = _curve_rows(data, primary, groups)
    return {
        "n_women": int(len(data)),
        "bmi_definition": "分组轴是首次检测官方BMI。达标时间模型主设定为 BMI + 年龄 + 身高，不含体重。",
        "chosen_k": int(chosen_k),
        "cv_scores": cv_scores,
        "alternatives": alternatives,
        "primary_aft": _aft_record(primary),
        "sensitivity_aft": _aft_record(sensitivity),
        "primary_aic": _aic(primary),
        "sensitivity_aic": _aic(sensitivity),
        "vif_primary": _aft_vif(data, PRIMARY_COVARIATES),
        "vif_sensitivity": _aft_vif(data, SENSITIVITY_COVARIATES),
        "groups": groups,
        "curves": curves,
        "lmm": lmm,
        "not_a_retest_policy": True,
    }


def _curve_rows(data: pd.DataFrame, fit, groups: list[dict]) -> list[dict]:
    if not fit.success or not groups:
        return []
    grid = time_grid()
    mu = fit.mean_log_time(data[PRIMARY_COVARIATES].to_numpy(dtype=float))
    pi = success_probability(mu, fit.sigma, grid)
    risk = np.asarray(time_risk(grid), dtype=float)
    rows = []
    for group in groups:
        pi_group = np.asarray(pi[group["_mask"]], dtype=float).mean(axis=0)
        objective = np.asarray(decision_objective(grid, pi_group), dtype=float)
        for t, probability, time_penalty, value in zip(grid, pi_group, risk, objective):
            rows.append(
                {
                    "group": int(group["group"]),
                    "interval": group["interval"],
                    "t": float(t),
                    "pi": float(probability),
                    "risk": float(time_penalty),
                    "objective": float(value),
                    "t_star": float(group["t_star"]),
                }
            )
    return rows


def _select_k(data: pd.DataFrame, covariates: list[str], n_min: int):
    scores = {k: [] for k in K_CANDIDATES}
    try:
        splits = iter_group_kfold(data["woman_id"].to_numpy(), n_splits=5)
    except ValueError:
        return K_CANDIDATES[0], {k: [] for k in K_CANDIDATES}
    for train_index, test_index in splits:
        train = data.iloc[train_index].sort_values(["bmi_first", "woman_id"], kind="mergesort").reset_index(drop=True)
        test = data.iloc[test_index]
        fit = _fit_frame(train, covariates)
        if not fit.success:
            continue
        grid = time_grid()
        pi = success_probability(fit.mean_log_time(train[covariates].to_numpy(dtype=float)), fit.sigma, grid)
        mu_test = fit.mean_log_time(test[covariates].to_numpy(dtype=float))
        for k in K_CANDIDATES:
            edges = _risk_edges(train["bmi_first"].to_numpy(dtype=float), pi, k, n_min, grid)
            if edges is None:
                continue
            cuts = cuts_from_edges(train["bmi_first"].to_numpy(dtype=float), edges)
            train_labels = assign_groups(train["bmi_first"].to_numpy(dtype=float), cuts)
            time_of = {}
            for group in range(k):
                mask = train_labels == group
                if not np.any(mask):
                    continue
                time_of[group] = minimize_objective(pi[mask].mean(axis=0), grid)[0]
            test_labels = assign_groups(test["bmi_first"].to_numpy(dtype=float), cuts)
            losses = []
            for index, group in enumerate(test_labels):
                if int(group) not in time_of:
                    continue
                chosen_time = time_of[int(group)]
                pi_i = float(success_probability(mu_test[index], fit.sigma, chosen_time))
                losses.append(float(decision_objective(chosen_time, pi_i)))
            if losses:
                scores[k].append(float(np.mean(losses)))
    chosen = min(K_CANDIDATES, key=lambda k: float(np.mean(scores[k])) if scores[k] else float("inf"))
    summary = {
        str(k): {
            "mean_heldout_risk": float(np.mean(values)) if values else float("nan"),
            "n_folds": int(len(values)),
        }
        for k, values in scores.items()
    }
    return int(chosen), summary


def _groups_from_fit(data: pd.DataFrame, fit, k: int, n_min: int) -> list[dict]:
    if not fit.success:
        return []
    grid = time_grid()
    mu = fit.mean_log_time(data[PRIMARY_COVARIATES].to_numpy(dtype=float))
    pi = success_probability(mu, fit.sigma, grid)
    bmi = data["bmi_first"].to_numpy(dtype=float)
    edges = _risk_edges(bmi, pi, k, n_min, grid)
    if edges is None:
        edges = [0, len(data)]
        k = 1
    cuts = cuts_from_edges(bmi, edges)
    labels = assign_groups(bmi, cuts)
    groups = []
    for group in range(int(labels.max()) + 1 if len(labels) else 0):
        mask = labels == group
        if not np.any(mask):
            continue
        pi_group = pi[mask].mean(axis=0)
        t_star, objective, pi_star = minimize_objective(pi_group, grid)
        member = data.loc[mask]
        support = _earliest_observed(member)
        supported = grid[grid + 1e-9 >= support]
        if supported.size == 0:
            supported = grid
        pi_supported = success_probability(mu[mask], fit.sigma, supported).mean(axis=0)
        t_supported, _, _ = minimize_objective(pi_supported, supported)
        individual = success_probability(mu[mask], fit.sigma, t_star)
        groups.append(
            {
                "group": int(group),
                "interval": interval_label(cuts, int(group), bmi[mask]),
                "n_women": int(mask.sum()),
                "bmi_min": float(np.min(bmi[mask])),
                "bmi_max": float(np.max(bmi[mask])),
                "t_star": t_star,
                "t_star_supported": t_supported,
                "pi_star": pi_star,
                "mean_individual_pi": float(np.mean(individual)),
                "objective": objective,
                "mean_age": float(member["age"].mean()),
                "mean_height": float(member["height_cm"].mean()),
                "n_left": int((member["kind"] == "left").sum()),
                "n_interval": int((member["kind"] == "interval").sum()),
                "n_right": int((member["kind"] == "right").sum()),
                "_mask": mask,
            }
        )
    return groups


def _risk_edges(bmi, pi, k: int, n_min: int, grid) -> list[int] | None:
    if k * n_min > len(bmi):
        return None
    boundary = boundary_indices(bmi, n_min=n_min)

    def cost(start: int, end: int) -> float:
        pi_group = np.asarray(pi[start:end], dtype=float).mean(axis=0)
        _, objective, _ = minimize_objective(pi_group, grid)
        return float(objective) * (end - start)

    edges, _ = best_ordered_partition(boundary, cost, k, n_min)
    return edges


def _bootstrap_groups(data, fit, groups, n_boot, seed):
    if n_boot <= 0 or not groups or not fit.success:
        return
    rng = np.random.default_rng(seed)
    slope_draws = []
    time_draws = {group["group"]: [] for group in groups}
    n = len(data)
    covariates = data[PRIMARY_COVARIATES].to_numpy(dtype=float)
    grid = time_grid()
    for _ in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        sample = data.iloc[chosen]
        refit = _fit_frame(sample, PRIMARY_COVARIATES)
        if not refit.success:
            continue
        slope_draws.append(np.concatenate([refit.beta[1:], [refit.sigma]]))
        mu = refit.mean_log_time(covariates)
        pi = success_probability(mu, refit.sigma, grid)
        for group in groups:
            pi_group = pi[group["_mask"]].mean(axis=0)
            time_draws[group["group"]].append(minimize_objective(pi_group, grid)[0])
    names = PRIMARY_COVARIATES + ["sigma"]
    if slope_draws:
        stacked = np.vstack(slope_draws)
        fit.bootstrap_ci = {
            name: {
                "low": float(np.quantile(stacked[:, index], 0.025)),
                "high": float(np.quantile(stacked[:, index], 0.975)),
            }
            for index, name in enumerate(names)
        }
        fit.n_boot = int(len(slope_draws))
    for group in groups:
        draws = time_draws[group["group"]]
        group["n_boot"] = int(len(draws))
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            group["t_boot_low"] = float(low)
            group["t_boot_high"] = float(high)
        else:
            group["t_boot_low"] = float("nan")
            group["t_boot_high"] = float("nan")


def _noise_groups(visits, data, groups, n_noise, seed, n_min):
    del n_min
    for group in groups:
        group["t_noise_low"] = float("nan")
        group["t_noise_high"] = float("nan")
        group["n_noise"] = 0
    if n_noise <= 0 or not groups:
        return
    from .censoring import replicate_sigma

    sigma = replicate_sigma(visits)
    if not np.isfinite(sigma) or sigma <= 0:
        return
    rng = np.random.default_rng(seed)
    collected = {group["group"]: [] for group in groups}
    membership = data["woman_id"].astype(str).to_numpy()
    group_of = {}
    for group in groups:
        for woman in membership[group["_mask"]]:
            group_of[woman] = group["group"]
    grid = time_grid()
    for _ in range(n_noise):
        perturbed = visits.copy()
        perturbed["y_frac"] = perturbed["y_frac"].to_numpy(dtype=float) + rng.normal(0.0, sigma, size=len(perturbed))
        hits = _analysis_frame(build_woman_hits(perturbed)).dropna(subset=PRIMARY_COVARIATES)
        if len(hits) < 30:
            continue
        refit = _fit_frame(hits, PRIMARY_COVARIATES)
        if not refit.success:
            continue
        hits = hits.copy()
        hits["group"] = hits["woman_id"].astype(str).map(group_of)
        mu = refit.mean_log_time(hits[PRIMARY_COVARIATES].to_numpy(dtype=float))
        for group_id, member_index in hits.dropna(subset=["group"]).groupby("group").indices.items():
            pi_group = success_probability(mu[member_index], refit.sigma, grid).mean(axis=0)
            collected[int(group_id)].append(minimize_objective(pi_group, grid)[0])
    for group in groups:
        draws = collected.get(group["group"], [])
        group["n_noise"] = int(len(draws))
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            group["t_noise_low"] = float(low)
            group["t_noise_high"] = float(high)


def _lmm_sensitivity(visits, data, groups) -> dict:
    frame = build_concentration_frame(visits)
    sample = frame.dropna(subset=["y", "ga", "bmi", "age", "height_cm", "woman_id"]).reset_index(drop=True)
    if sample["woman_id"].nunique() < 10:
        return {"error": "浓度模型样本不足"}
    centered, centers = center_frame(sample)
    height_center = float(sample["height_cm"].mean())
    centered["height_c"] = sample["height_cm"] - height_center
    formula = "y ~ ga_c + bmi_c + ga_bmi + age_c + height_c"
    try:
        result = fit_mixed_model(formula, centered)
    except Exception as exc:
        return {"error": str(exc)}
    variance = variance_components(result)
    random_var = float(np.asarray(result.cov_re, dtype=float).reshape(-1)[0])
    sigma = float(np.sqrt(max(random_var, 0.0) + float(result.scale)))
    params = {str(name): float(value) for name, value in result.fe_params.items()}
    grid = time_grid()
    by_woman = data.set_index(data["woman_id"].astype(str), drop=False)
    if not groups:
        return {
            "coefficients": coefficient_table(result, "mixed"),
            "variance": variance,
            "sigma_predictive": sigma,
            "vif": design_vif(centered, ["ga_c", "bmi_c", "age_c", "height_c"]),
            "groups": [],
        }
    lmm_rows = []
    for group in groups:
        women = by_woman.loc[by_woman.index.intersection(data.loc[group["_mask"], "woman_id"].astype(str))]
        if isinstance(women, pd.Series):
            women = women.to_frame().T
        pi_rows = []
        for _, woman in women.iterrows():
            mu = _lmm_mean(grid, woman, params, centers, height_center)
            pi_rows.append(1.0 - _normal_cdf((Y_THRESHOLD - mu) / sigma))
        if not pi_rows:
            continue
        pi_group = np.mean(np.vstack(pi_rows), axis=0)
        t_star, objective, pi_star = minimize_objective(pi_group, grid)
        lmm_rows.append(
            {
                "group": group["group"],
                "t_star_lmm": t_star,
                "pi_star_lmm": pi_star,
                "objective_lmm": objective,
            }
        )
        group["t_star_lmm"] = t_star
    return {
        "coefficients": coefficient_table(result, "mixed"),
        "variance": variance,
        "sigma_predictive": sigma,
        "height_center": height_center,
        "centering": centers,
        "vif": design_vif(centered, ["ga_c", "bmi_c", "age_c", "height_c"]),
        "groups": lmm_rows,
        "note": "这是浓度模型给出的同期超过 0.04 的概率，用作敏感性；主时点来自区间删失首次达标模型。",
    }


def _lmm_mean(grid, woman, params, centers, height_center):
    ga_c = grid - centers["ga"]
    bmi_c = float(woman["bmi_first"]) - centers["bmi"]
    age_c = float(woman["age"]) - centers["age"]
    height_c = float(woman["height_cm"]) - height_center
    return (
        params.get("Intercept", 0.0)
        + params.get("ga_c", 0.0) * ga_c
        + params.get("bmi_c", 0.0) * bmi_c
        + params.get("ga_bmi", 0.0) * ga_c * bmi_c
        + params.get("age_c", 0.0) * age_c
        + params.get("height_c", 0.0) * height_c
    )


def _normal_cdf(z):
    from scipy.stats import norm

    return norm.cdf(z)


def _fit_frame(frame: pd.DataFrame, covariates: list[str]):
    return fit_interval_aft(
        frame[covariates].to_numpy(dtype=float),
        frame["left"].to_numpy(dtype=float),
        frame["right"].to_numpy(dtype=float),
        frame["kind"].to_numpy(),
        covariates,
    )


def _aic(fit) -> float:
    if not fit.success:
        return float("inf")
    k = int(len(fit.beta) + 1)
    return float(2 * k + 2 * fit.nll)


def _aft_record(fit) -> dict:
    rows = [{"term": "Intercept", "estimate": float(fit.beta[0]) if fit.success else float("nan"), "scale": "均值处截距"}]
    if fit.success:
        for name, estimate, center in zip(fit.covariate_names, fit.beta[1:], fit.center):
            rows.append({"term": name, "estimate": float(estimate), "center": float(center), "scale": "每单位斜率，协变量已中心化"})
        rows.append({"term": "sigma", "estimate": float(fit.sigma), "scale": "log时间的尺度"})
    return {
        "success": bool(fit.success),
        "n": int(fit.n),
        "nll": float(fit.nll) if np.isfinite(fit.nll) else None,
        "boundary": bool(fit.boundary),
        "message": fit.message,
        "coefficients": rows,
        "bootstrap_ci": getattr(fit, "bootstrap_ci", {}),
        "n_boot": int(getattr(fit, "n_boot", 0)),
    }


def _aft_vif(data: pd.DataFrame, covariates: list[str]) -> dict:
    centered = data[covariates].astype(float).copy()
    for column in covariates:
        centered[column] = centered[column] - centered[column].mean()
    return design_vif(centered, covariates)


def public_groups(groups: list[dict]) -> pd.DataFrame:
    hidden = {"_mask"}
    return pd.DataFrame([{key: value for key, value in group.items() if key not in hidden} for group in groups])
