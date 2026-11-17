"""问题二：按首次就诊 BMI 做有序分组，并在各组上最小化预先登记的风险。

官方五档只作为基线。检测误差分析只扰动 Y，不重新分组。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .censoring import build_woman_hits, replicate_sigma
from .constants import (
    K_CANDIDATES,
    N_BOOT_TIME,
    N_MIN_GROUP,
    N_NOISE,
    OFFICIAL_BMI_CUTS,
    PRIMARY_LAMBDA,
    PRIMARY_SLOPES,
    RANDOM_SEED,
    Y_THRESHOLD,
)
from .grouping import (
    assign_groups,
    best_ordered_partition,
    bic,
    boundary_indices,
    cuts_from_edges,
    interval_label,
    official_bmi_group,
    official_labels,
)
from .risk import minimize_objective, sensitivity_settings, time_grid
from .survival import fit_interval_lognormal, success_probability


def run_problem2(
    visits: pd.DataFrame,
    hits: pd.DataFrame = None,
    n_boot: int = N_BOOT_TIME,
    n_noise: int = N_NOISE,
    seed: int = RANDOM_SEED,
    n_min: int = N_MIN_GROUP,
) -> dict:
    records = hits if hits is not None else build_woman_hits(visits, threshold=Y_THRESHOLD)
    data = _analysis_frame(records)
    official = _fixed_partition(data, official_bmi_group(data["bmi_first"].to_numpy()), list(OFFICIAL_BMI_CUTS), "官方示例五档")
    if (data["bmi_first"] < 20).any():
        for row in official["groups"]:
            if row["group"] == 0:
                row["interval"] = "<28"
    else:
        labels = official_labels()
        for row in official["groups"]:
            row["nominal_interval"] = labels[row["group"]]
    selected, selection_table, cache_size = _select_partition(data, n_min=n_min)
    driven = _fixed_partition(data, selected["labels"], selected["cuts"], "有序统计分割")
    for row, edge_text in zip(driven["groups"], selected["interval_texts"]):
        row["interval"] = edge_text
    sigma = replicate_sigma(visits)
    rng = np.random.default_rng(seed)
    _attach_bootstrap(data, driven["groups"], n_boot, rng)
    _attach_bootstrap(data, official["groups"], n_boot, rng)
    noise_rng = np.random.default_rng(seed + 1)
    _attach_noise(visits, driven["groups"], sigma, n_noise, noise_rng)
    sensitivity = _sensitivity_table(driven["groups"])
    return {
        "n_women": int(len(data)),
        "dropped_women": int(len(records) - len(data)) if records is not None else 0,
        "bmi_definition": "首次检测的官方BMI；缺失时用首次身高体重计算",
        "selection": selection_table,
        "chosen_k": int(selected["k"]),
        "segment_fits": int(cache_size),
        "official_nll": official["nll"],
        "official_bic_fixed_cuts": official["bic"],
        "driven": driven["groups"],
        "official": official["groups"],
        "sensitivity": sensitivity,
        "replicate_sigma_y": sigma,
        "primary_slopes": PRIMARY_SLOPES,
        "primary_lambda": PRIMARY_LAMBDA,
    }


def _analysis_frame(hits: pd.DataFrame) -> pd.DataFrame:
    frame = hits.dropna(subset=["woman_id", "bmi_first", "kind"]).copy()
    mask = _valid_censoring(frame)
    frame = frame.loc[mask].sort_values(["bmi_first", "woman_id"], kind="mergesort").reset_index(drop=True)
    return frame


def _valid_censoring(frame: pd.DataFrame) -> pd.Series:
    kind = frame["kind"].astype(str)
    left_ok = (kind != "left") | (np.isfinite(frame["right"]) & (frame["right"] > 0))
    right_ok = (kind != "right") | (np.isfinite(frame["left"]) & (frame["left"] > 0))
    interval_ok = (kind != "interval") | (
        np.isfinite(frame["left"]) & np.isfinite(frame["right"]) & (frame["right"] > frame["left"]) & (frame["left"] > 0)
    )
    return left_ok & right_ok & interval_ok


def _select_partition(data: pd.DataFrame, n_min: int):
    bmi = data["bmi_first"].to_numpy(dtype=float)
    boundary = boundary_indices(bmi, n_min=n_min)
    cache: dict[tuple[int, int], float] = {}

    def segment_cost(start: int, end: int) -> float:
        key = (start, end)
        if key not in cache:
            fit = fit_interval_lognormal(data["left"].iloc[start:end], data["right"].iloc[start:end], data["kind"].iloc[start:end])
            cache[key] = float(fit.nll) if fit.success else 1e6 + 100.0 * (end - start)
        return cache[key]

    rows = []
    best = None
    n = len(data)
    for k in K_CANDIDATES:
        if k * n_min > n:
            continue
        edges, total = best_ordered_partition(boundary, segment_cost, k, n_min)
        if edges is None:
            continue
        penalty = bic(total, n, 2 * k + (k - 1))
        cuts = cuts_from_edges(bmi, edges)
        labels = assign_groups(bmi, cuts)
        row = {"k": int(k), "nll": float(total), "bic": float(penalty), "n_params": int(2 * k + (k - 1))}
        rows.append(row)
        if best is None or penalty < best["bic"]:
            texts = []
            for group in range(k):
                member = bmi[labels == group]
                texts.append(interval_label(cuts, group, member))
            best = {"k": k, "bic": penalty, "cuts": cuts, "labels": labels, "edges": edges, "interval_texts": texts}
    if best is None:
        labels = np.zeros(n, dtype=int)
        best = {"k": 1, "bic": float("inf"), "cuts": [], "labels": labels, "edges": [0, n], "interval_texts": [interval_label([], 0, bmi)]}
    return best, rows, len(cache)


def _fixed_partition(data: pd.DataFrame, labels: np.ndarray, cuts: list[float], source: str) -> dict:
    groups = []
    total_nll = 0.0
    occupied = 0
    for group in sorted(np.unique(labels)):
        member = data.loc[labels == group].copy()
        record = _fit_group(member, int(group), cuts)
        record["source"] = source
        if record["success"]:
            total_nll += record["nll"]
            occupied += 1
        groups.append(record)
    n = len(data)
    return {"groups": groups, "nll": float(total_nll), "bic": bic(total_nll, n, 2 * max(occupied, 1))}


def _fit_group(member: pd.DataFrame, group: int, cuts: list[float]) -> dict:
    bmi = member["bmi_first"].to_numpy(dtype=float)
    fit = fit_interval_lognormal(member["left"], member["right"], member["kind"].to_numpy())
    grid = time_grid()
    record = {
        "group": group,
        "interval": interval_label(list(cuts), group, bmi) if len(bmi) else "",
        "n_women": int(len(member)),
        "bmi_min": float(np.min(bmi)) if len(bmi) else float("nan"),
        "bmi_max": float(np.max(bmi)) if len(bmi) else float("nan"),
        "n_left": int((member["kind"] == "left").sum()),
        "n_interval": int((member["kind"] == "interval").sum()),
        "n_right": int((member["kind"] == "right").sum()),
        "success": bool(fit.success),
        "boundary": bool(fit.boundary),
        "mu": fit.mu,
        "sigma": fit.sigma,
        "nll": fit.nll,
        "message": fit.message,
    }
    if not fit.success:
        record.update({"t_star": float("nan"), "t_star_supported": float("nan"), "pi_star": float("nan"), "objective": float("nan")})
        return record
    pi = success_probability(fit.mu, fit.sigma, grid)
    t_star, objective, pi_star = minimize_objective(pi, grid)
    support = _earliest_observed(member)
    supported_grid = grid[grid + 1e-9 >= support]
    if supported_grid.size == 0:
        supported_grid = grid
    pi_supported = success_probability(fit.mu, fit.sigma, supported_grid)
    t_supported, _, _ = minimize_objective(pi_supported, supported_grid)
    record.update(
        {
            "t_star": t_star,
            "t_star_supported": t_supported,
            "support_start": float(support),
            "pi_star": pi_star,
            "objective": objective,
            "pi_12": float(success_probability(fit.mu, fit.sigma, 12.0)),
            "pi_16": float(success_probability(fit.mu, fit.sigma, 16.0)),
            "pi_20": float(success_probability(fit.mu, fit.sigma, 20.0)),
            "pi_24": float(success_probability(fit.mu, fit.sigma, 24.0)),
            "_women": set(member["woman_id"].astype(str)),
        }
    )
    return record


def _earliest_observed(member: pd.DataFrame) -> float:
    times = []
    for kind, left, right in zip(member["kind"], member["left"], member["right"]):
        if kind == "left":
            times.append(float(right))
        elif kind == "right":
            times.append(float(left))
        else:
            times.append(float(left))
    return float(np.min(times)) if times else 10.0


def _attach_bootstrap(data: pd.DataFrame, groups: list[dict], n_boot: int, rng: np.random.Generator) -> None:
    if n_boot <= 0:
        return
    for group in groups:
        women = group.get("_women")
        if not women or not group.get("success"):
            group["t_boot_low"] = float("nan")
            group["t_boot_high"] = float("nan")
            group["n_boot"] = 0
            continue
        member = data.loc[data["woman_id"].astype(str).isin(women)].reset_index(drop=True)
        draws = []
        for _ in range(n_boot):
            chosen = rng.integers(0, len(member), size=len(member))
            sample = member.iloc[chosen]
            fit = fit_interval_lognormal(sample["left"], sample["right"], sample["kind"].to_numpy())
            if not fit.success:
                continue
            grid = time_grid()
            pi = success_probability(fit.mu, fit.sigma, grid)
            draws.append(minimize_objective(pi, grid)[0])
        group["n_boot"] = int(len(draws))
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            group["t_boot_low"] = float(low)
            group["t_boot_high"] = float(high)
        else:
            group["t_boot_low"] = float("nan")
            group["t_boot_high"] = float("nan")


def _attach_noise(visits, groups, sigma, n_noise, rng) -> None:
    for group in groups:
        group["t_noise_low"] = float("nan")
        group["t_noise_high"] = float("nan")
        group["n_noise"] = 0
    if n_noise <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return
    membership = []
    for group in groups:
        for woman in group.get("_women", ()):
            membership.append((woman, group["group"]))
    if not membership:
        return
    lookup = {woman: group_id for woman, group_id in membership}
    collected = {group["group"]: [] for group in groups}
    for _ in range(n_noise):
        perturbed = visits.copy()
        perturbed["y_frac"] = perturbed["y_frac"].to_numpy(dtype=float) + rng.normal(0.0, sigma, size=len(perturbed))
        hits = build_woman_hits(perturbed)
        if hits.empty:
            continue
        hits = _analysis_frame(hits)
        hits["group"] = hits["woman_id"].astype(str).map(lookup)
        for group_id, member in hits.dropna(subset=["group"]).groupby("group"):
            fit = fit_interval_lognormal(member["left"], member["right"], member["kind"].to_numpy())
            if not fit.success:
                continue
            grid = time_grid()
            pi = success_probability(fit.mu, fit.sigma, grid)
            collected[int(group_id)].append(minimize_objective(pi, grid)[0])
    for group in groups:
        draws = collected.get(group["group"], [])
        group["n_noise"] = int(len(draws))
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            group["t_noise_low"] = float(low)
            group["t_noise_high"] = float(high)


def _sensitivity_table(groups: list[dict]) -> list[dict]:
    rows = []
    grid = time_grid()
    for group in groups:
        if not group.get("success"):
            continue
        pi = success_probability(group["mu"], group["sigma"], grid)
        for slopes, lam in sensitivity_settings():
            t_star, objective, pi_star = minimize_objective(pi, grid, lam=lam, slopes=slopes)
            rows.append(
                {
                    "group": group["group"],
                    "interval": group["interval"],
                    "slopes": ":".join(str(int(item)) for item in slopes),
                    "lambda": lam,
                    "t_star": t_star,
                    "pi_star": pi_star,
                    "objective": objective,
                    "primary": bool(slopes == PRIMARY_SLOPES and lam == PRIMARY_LAMBDA),
                }
            )
    return rows


def public_groups(groups: list[dict]) -> pd.DataFrame:
    hidden = {"_women"}
    rows = [{key: value for key, value in group.items() if key not in hidden} for group in groups]
    return pd.DataFrame(rows)
