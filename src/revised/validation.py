"""分组交叉验证、只在训练集上拟合的标准化，以及阈值选择。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def iter_group_kfold(groups, n_splits: int = 5, y=None):
    """按孕妇分组。同一 ``groups`` 值不会同时出现在训练折与测试折。

    ``y`` 不参与划分，保留这个参数只是为了让调用处明确标签没有被用来切分。
    """
    del y
    group_array = np.asarray(groups)
    n_groups = len(np.unique(group_array))
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        raise ValueError("分组交叉验证至少需要两名孕妇")
    folder = GroupKFold(n_splits=splits)
    dummy_x = np.zeros((len(group_array), 1))
    dummy_y = np.zeros(len(group_array))
    return list(folder.split(dummy_x, dummy_y, group_array))


def assert_no_group_leakage(splits: Iterable[tuple], groups) -> None:
    group_array = np.asarray(groups)
    for train_index, test_index in splits:
        train_ids = set(group_array[train_index])
        test_ids = set(group_array[test_index])
        if not train_ids.isdisjoint(test_ids):
            raise AssertionError("同一孕妇同时出现在训练折和测试折")


def fit_transform_train_only(x_train, x_test):
    """标准化只在训练折上 ``fit``。测试折只 ``transform``。"""
    scaler = StandardScaler()
    train_matrix = np.asarray(x_train, dtype=float)
    test_matrix = np.asarray(x_test, dtype=float)
    if train_matrix.ndim == 1:
        train_matrix = train_matrix.reshape(-1, 1)
        test_matrix = test_matrix.reshape(-1, 1)
    transformed_train = scaler.fit_transform(train_matrix)
    transformed_test = scaler.transform(test_matrix)
    return transformed_train, transformed_test, scaler


def choose_threshold(y_true, scores, target_sensitivity: float = 0.95) -> float:
    """只根据给定样本选择阈值。调用方必须传入训练折。

    在敏感度达到目标的阈值中，取特异度最高者；若达不到目标，则在能达到的
    最高敏感度上再取特异度最高的阈值。预测规则是 score >= 阈值。
    """
    y = np.asarray(y_true).astype(int)
    score = np.asarray(scores, dtype=float)
    if y.size == 0:
        raise ValueError("没有样本，不能选择阈值")
    candidates = np.unique(score)
    best_feasible = None
    best_fallback = None
    for threshold in candidates:
        sensitivity, specificity = _sens_spec(y, score >= threshold)
        if sensitivity + 1e-12 >= target_sensitivity:
            key = (specificity, float(threshold))
            if best_feasible is None or key > best_feasible[0]:
                best_feasible = (key, float(threshold))
        fallback_key = (sensitivity, specificity, float(threshold))
        if best_fallback is None or fallback_key > best_fallback[0]:
            best_fallback = (fallback_key, float(threshold))
    if best_feasible is not None:
        return best_feasible[1]
    return best_fallback[1]


def negative_score_thresholds(scores, y_true, specificities=(0.99, 0.995)):
    """在训练折的阴性样本上，按特异度取高分端阈值。返回 (较低阈值, 较高阈值)。"""
    y = np.asarray(y_true).astype(int)
    score = np.asarray(scores, dtype=float)
    negative = score[y == 0]
    negative = negative[np.isfinite(negative)]
    if negative.size == 0:
        raise ValueError("训练集没有阴性样本，不能估计特异度阈值")
    low = float(np.quantile(negative, specificities[0]))
    high = float(np.quantile(negative, specificities[1]))
    if high < low:
        low, high = high, low
    return low, high


def zscore_calls(z_values, threshold: float = 3.0) -> np.ndarray:
    """经典规则：|Z| >= 阈值判为阳性。缺失 Z 不判阳性。"""
    z = np.asarray(z_values, dtype=float)
    calls = np.zeros(z.shape, dtype=int)
    finite = np.isfinite(z)
    calls[finite] = (np.abs(z[finite]) >= threshold).astype(int)
    return calls


def confusion_counts(y_true, y_pred) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y == 1) & (pred == 1))),
        "tn": int(np.sum((y == 0) & (pred == 0))),
        "fp": int(np.sum((y == 0) & (pred == 1))),
        "fn": int(np.sum((y == 1) & (pred == 0))),
    }


def binary_rates(counts: dict) -> dict:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
    }


def _sens_spec(y, pred):
    counts = confusion_counts(y, pred.astype(int))
    rates = binary_rates(counts)
    sensitivity = 0.0 if rates["sensitivity"] != rates["sensitivity"] else rates["sensitivity"]
    specificity = 0.0 if rates["specificity"] != rates["specificity"] else rates["specificity"]
    return sensitivity, specificity


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)
