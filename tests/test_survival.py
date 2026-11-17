import numpy as np

from src.revised.grouping import assign_groups, best_ordered_partition, boundary_indices, cuts_from_edges
from src.revised.survival import fit_interval_lognormal


def test_lognormal_recovers_narrow_intervals():
    rng = np.random.default_rng(1)
    times = rng.lognormal(mean=np.log(16), sigma=0.15, size=80)
    fit = fit_interval_lognormal(times * 0.98, times * 1.02, np.array(["interval"] * 80))
    assert fit.success
    assert fit.mu == np.log(16) or abs(fit.mu - np.log(16)) < 0.05
    assert abs(fit.sigma - 0.15) < 0.05


def test_right_censoring_moves_the_distribution_later():
    early = fit_interval_lognormal(
        np.array([10, 10, 12, 12]),
        np.array([12, 12, 14, 14]),
        np.array(["interval"] * 4),
    )
    late = fit_interval_lognormal(
        np.array([18, 18, 20, 20]),
        np.array([np.inf, np.inf, np.inf, np.inf]),
        np.array(["right"] * 4),
    )
    assert early.success and late.success
    assert late.mu > early.mu


def test_ordered_split_is_contiguous_and_covers_everyone():
    bmi = np.linspace(22, 40, 40)
    boundary = boundary_indices(bmi, n_min=8, max_interior=10)

    def cost(start, end):
        values = bmi[start:end]
        return float(np.sum((values - values.mean()) ** 2))

    edges, _ = best_ordered_partition(boundary, cost, k=3, n_min=8)
    assert edges[0] == 0
    assert edges[-1] == len(bmi)
    assert edges == sorted(edges)
    cuts = cuts_from_edges(bmi, edges)
    labels = assign_groups(bmi, cuts)
    assert set(labels) == {0, 1, 2}
    assert list(labels) == sorted(labels)
    for group in range(3):
        assert int(np.sum(labels == group)) >= 8
