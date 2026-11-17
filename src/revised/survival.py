"""区间删失的对数正态模型。

中点填补只用来给优化器一个起点，不作为达标时间的估计结果输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

_NORMAL = stats.norm


@dataclass
class LogNormalFit:
    mu: float
    sigma: float
    nll: float
    success: bool
    n: int
    boundary: bool = False
    message: str = ""


@dataclass
class AFTFit:
    beta: np.ndarray
    sigma: float
    nll: float
    success: bool
    n: int
    covariate_names: list[str] = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    boundary: bool = False
    message: str = ""

    def mean_log_time(self, covariates: np.ndarray) -> np.ndarray:
        raw = np.asarray(covariates, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        centered = raw - self.center
        design = np.column_stack([np.ones(len(centered)), centered])
        return design @ self.beta


def interval_nll(mu, sigma, left, right, kind) -> float:
    """左删失、区间删失、右删失观测的负对数似然。``mu`` 可以是标量或逐人向量。"""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return 1e12
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    kind = np.asarray(kind)
    mu_vector = np.broadcast_to(np.asarray(mu, dtype=float), kind.shape).copy()
    loglik = np.zeros(kind.shape[0], dtype=float)

    left_mask = kind == "left"
    if np.any(left_mask):
        z_right = (np.log(right[left_mask]) - mu_vector[left_mask]) / sigma
        loglik[left_mask] = _NORMAL.logcdf(z_right)

    right_mask = kind == "right"
    if np.any(right_mask):
        z_left = (np.log(left[right_mask]) - mu_vector[right_mask]) / sigma
        loglik[right_mask] = _NORMAL.logsf(z_left)

    interval_mask = kind == "interval"
    if np.any(interval_mask):
        z_left = (np.log(left[interval_mask]) - mu_vector[interval_mask]) / sigma
        z_right = (np.log(right[interval_mask]) - mu_vector[interval_mask]) / sigma
        loglik[interval_mask] = _log_cdf_diff(z_left, z_right)

    unknown = ~(left_mask | right_mask | interval_mask)
    if np.any(unknown):
        return 1e12
    if not np.all(np.isfinite(loglik)):
        return 1e12
    return float(-np.sum(loglik))


def fit_interval_lognormal(left, right, kind) -> LogNormalFit:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    kind = np.asarray(kind)
    n = int(kind.shape[0])
    if n < 3:
        return LogNormalFit(mu=np.nan, sigma=np.nan, nll=np.inf, success=False, n=n, message="样本少于 3")
    start_mu, start_sigma = _start_from_imputed(left, right, kind)
    bounds = [(np.log(6.0), np.log(40.0)), (np.log(0.05), np.log(1.2))]

    def objective(theta):
        return interval_nll(theta[0], np.exp(theta[1]), left, right, kind)

    result = optimize.minimize(
        objective,
        x0=np.array([start_mu, np.log(start_sigma)]),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-10},
    )
    mu = float(result.x[0])
    sigma = float(np.exp(result.x[1]))
    nll = float(result.fun) if np.isfinite(result.fun) else float(objective(result.x))
    on_bound = _on_bound(mu, bounds[0]) or _on_bound(np.log(sigma), bounds[1])
    success = bool(np.isfinite(nll) and sigma > 0)
    return LogNormalFit(
        mu=mu,
        sigma=sigma,
        nll=nll,
        success=success,
        n=n,
        boundary=on_bound,
        message="" if result.success else "优化未报告收敛",
    )


def fit_interval_aft(covariates, left, right, kind, covariate_names=None) -> AFTFit:
    raw = np.asarray(covariates, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    kind = np.asarray(kind)
    n, p = raw.shape
    names = list(covariate_names) if covariate_names is not None else [f"x{i}" for i in range(p)]
    if n < max(8, p + 3) or not np.all(np.isfinite(raw)):
        return AFTFit(
            beta=np.full(p + 1, np.nan),
            sigma=np.nan,
            nll=np.inf,
            success=False,
            n=n,
            covariate_names=names,
            message="协变量不完整或样本过少",
        )
    center = raw.mean(axis=0)
    centered = raw - center
    design = np.column_stack([np.ones(n), centered])
    start_log_t = _imputed_log_time(left, right, kind)
    beta0, *_ = np.linalg.lstsq(design, start_log_t, rcond=None)
    residual = start_log_t - design @ beta0
    sigma0 = float(np.std(residual, ddof=1)) if n > 2 else 0.25
    sigma0 = float(np.clip(sigma0, 0.08, 0.8))
    theta0 = np.concatenate([beta0, [np.log(sigma0)]])
    bounds = [(-8.0, 8.0)] * (p + 1) + [(np.log(0.05), np.log(1.2))]

    def objective(theta):
        mu = design @ theta[:-1]
        return interval_nll(mu, np.exp(theta[-1]), left, right, kind)

    result = optimize.minimize(
        objective,
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10},
    )
    beta = np.asarray(result.x[:-1], dtype=float)
    sigma = float(np.exp(result.x[-1]))
    nll = float(result.fun) if np.isfinite(result.fun) else float(objective(result.x))
    on_bound = any(_on_bound(value, bound) for value, bound in zip(result.x, bounds))
    return AFTFit(
        beta=beta,
        sigma=sigma,
        nll=nll,
        success=bool(np.isfinite(nll) and sigma > 0),
        n=n,
        covariate_names=names,
        center=center,
        boundary=on_bound,
        message="" if result.success else "优化未报告收敛",
    )


def success_probability(mu, sigma, times):
    """对数正态下 P(T <= t)。

    ``mu`` 可以是标量或形状 (n,)；``times`` 可以是标量、形状 (m,) 或与 ``mu`` 相同的 (n,)。
    """
    sigma = float(sigma)
    location = np.asarray(mu, dtype=float)
    t = np.asarray(times, dtype=float)
    if location.ndim == 0 and t.ndim == 0:
        return float(_NORMAL.cdf((math_log(t) - float(location)) / sigma))
    if location.ndim == 0:
        return _NORMAL.cdf((np.log(t) - float(location)) / sigma)
    if t.ndim == 0:
        return _NORMAL.cdf((math_log(t) - location) / sigma)
    if location.shape == t.shape:
        return _NORMAL.cdf((np.log(t) - location) / sigma)
    return _NORMAL.cdf((np.log(t)[None, :] - location[:, None]) / sigma)


def math_log(value: float) -> float:
    return float(np.log(value))


def _log_cdf_diff(z_left, z_right) -> np.ndarray:
    """log(Φ(z_right) - Φ(z_left))，避免两端过近时 log1p(-1) 的数值问题。"""
    a = _NORMAL.logcdf(z_right)
    b = _NORMAL.logcdf(z_left)
    gap = np.clip(b - a, -80.0, 0.0)
    close = gap > -1e-6
    diff = np.empty(np.shape(a), dtype=float)
    if np.any(~close):
        diff[~close] = a[~close] + np.log1p(-np.exp(gap[~close]))
    if np.any(close):
        midpoint = 0.5 * (np.asarray(z_left)[close] + np.asarray(z_right)[close])
        width = np.maximum(np.asarray(z_right)[close] - np.asarray(z_left)[close], 1e-12)
        diff[close] = _NORMAL.logpdf(midpoint) + np.log(width)
    return np.where(np.asarray(z_right) > np.asarray(z_left), diff, -1e6)


def _imputed_log_time(left, right, kind) -> np.ndarray:
    """仅优化起点。左删失用右端点，右删失把左端点稍向后推，区间用几何平均。"""
    out = np.empty(len(kind), dtype=float)
    for index, label in enumerate(kind):
        if label == "left":
            out[index] = np.log(max(right[index], 1e-3))
        elif label == "right":
            out[index] = np.log(max(left[index], 1e-3)) + 0.15
        else:
            out[index] = 0.5 * (np.log(max(left[index], 1e-3)) + np.log(max(right[index], 1e-3)))
    return out


def _start_from_imputed(left, right, kind):
    imputed = _imputed_log_time(left, right, kind)
    mu = float(np.mean(imputed))
    sigma = float(np.std(imputed, ddof=1)) if len(imputed) > 2 else 0.25
    if not np.isfinite(sigma) or sigma < 0.08:
        sigma = 0.25
    return float(np.clip(mu, np.log(6.0), np.log(40.0))), float(np.clip(sigma, 0.08, 0.8))


def _on_bound(value: float, bound, tol: float = 1e-4) -> bool:
    low, high = bound
    return abs(value - low) <= tol or abs(value - high) <= tol
