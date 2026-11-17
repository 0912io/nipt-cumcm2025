import numpy as np
import pytest

from src.revised.problem1_relationship import MAIN_FORMULAS, RIDGE_FEATURES
from src.revised.validation import (
    assert_no_group_leakage,
    binary_rates,
    choose_threshold,
    confusion_counts,
    fit_transform_train_only,
    iter_group_kfold,
    zscore_calls,
)


def test_group_kfold_does_not_leak_women():
    groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F"])
    splits = iter_group_kfold(groups, n_splits=5)
    assert_no_group_leakage(splits, groups)
    covered = np.sort(np.concatenate([test for _, test in splits]))
    assert np.array_equal(covered, np.arange(len(groups)))


def test_scaler_uses_training_rows_only():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(30, 2)) + np.array([5.0, -3.0])
    x_test = rng.normal(size=(10, 2)) + 100
    _, _, scaler = fit_transform_train_only(x_train, x_test)
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))
    pooled = np.vstack([x_train, x_test]).mean(axis=0)
    assert not np.allclose(scaler.mean_, pooled)
    _, _, scaler_again = fit_transform_train_only(x_train, np.zeros_like(x_test))
    assert np.allclose(scaler.mean_, scaler_again.mean_)


def test_threshold_is_a_function_of_the_supplied_sample_only():
    y_train = np.array([0, 0, 0, 1, 1])
    scores_train = np.array([0.1, 0.2, 0.4, 0.6, 0.9])
    y_extra = np.array([0, 1])
    scores_extra = np.array([0.95, 0.05])
    train_only = choose_threshold(y_train, scores_train, target_sensitivity=1.0)
    pooled = choose_threshold(np.r_[y_train, y_extra], np.r_[scores_train, scores_extra], target_sensitivity=1.0)
    assert train_only == pytest.approx(0.6)
    assert pooled != pytest.approx(train_only)


def test_zscore_baseline_confusion():
    calls = zscore_calls([0.2, 3, -3.1, 2.9, np.nan], threshold=3)
    assert calls.tolist() == [0, 1, 1, 0, 0]
    counts = confusion_counts([1, 0, 1, 0], [1, 1, 0, 0])
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    rates = binary_rates(counts)
    assert rates["sensitivity"] == pytest.approx(0.5)
    assert rates["specificity"] == pytest.approx(0.5)


def test_problem1_formulas_exclude_banned_predictors():
    banned = ("zy", "x_frac", "aneuploidy", "birth_healthy")
    for formula in MAIN_FORMULAS.values():
        for token in banned:
            assert token not in formula
    for token in banned:
        assert token not in RIDGE_FEATURES
