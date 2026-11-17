import numpy as np
import pandas as pd

from src.revised.curves import lognormal_curves
from src.revised.figures import write_figures
from src.revised.problem1_relationship import binned_profile, center_frame, compare_optimizers, quadratic_check
from src.revised.survival import success_probability


def _concentration_frame(curved: bool):
    rng = np.random.default_rng(0)
    rows = []
    for woman in range(36):
        ga = np.linspace(12, 24, 4) + rng.normal(0, 0.05, size=4)
        bmi = np.full(4, 28 + woman % 8) 
        shift = 0.004 * (ga - 18) ** 2 if curved else 0
        y = 0.08 + 0.001 * (ga - 18) - 0.001 * (bmi - 30) + shift + rng.normal(0, 0.004, size=4)
        for index in range(4):
            rows.append({"woman_id": f"W{woman:03d}", "y": float(y[index]), "ga": float(ga[index]), "bmi": float(bmi[index]), "age": 30.0, "ivf_assisted": 0.0, "gc": 0.4, "map_ratio": 0.8})
    return pd.DataFrame(rows)


def test_lognormal_curve_is_increasing_and_matches_cdf():
    groups = [{"group": 0, "interval": "[20, 30)", "success": True, "mu": 2.5, "sigma": 0.4, "t_star": 16.0}]
    frame = lognormal_curves(groups)
    assert frame["pi"].is_monotonic_increasing
    at_star = frame.loc[np.isclose(frame["t"], 16.0), "pi"].iloc[0]
    assert abs(at_star - float(success_probability(2.5, 0.4, 16.0))) < 1e-9
    assert float(frame["risk"].iloc[-1]) > float(frame["risk"].iloc[0])


def test_quadratic_check_detects_curvature():
    centered, _ = center_frame(_concentration_frame(curved=True))
    result = quadratic_check(centered)
    assert result["wald_df"] == 2
    assert result["wald_p"] < 0.01
    assert any(row["term"] == "ga_sq" for row in result["coefficients"])


def test_binned_profile_counts_every_finite_row():
    frame = _concentration_frame(curved=False)
    rows = binned_profile(frame, width=2.0)
    ga_rows = [row for row in rows if row["variable"] == "ga"]
    assert sum(row["n"] for row in ga_rows) == len(frame)
    assert all(row["bin_right"] > row["bin_left"] for row in rows)


def test_optimizer_comparison_reports_random_intercept():
    rng = np.random.default_rng(1)
    rows = []
    for woman in range(30):
        effect = float(rng.normal(0, 0.02))
        ga = np.linspace(12, 24, 4)
        bmi = np.full(4, 28 + woman % 6)
        y = 0.08 + 0.001 * (ga - 18) + effect + rng.normal(0, 0.004, size=4)
        for index in range(4):
            rows.append({"woman_id": f"W{woman:03d}", "y": float(y[index]), "ga": float(ga[index]), "bmi": float(bmi[index]), "age": 30.0, "ivf_assisted": 0.0, "gc": 0.4, "map_ratio": 0.8})
    centered, _ = center_frame(pd.DataFrame(rows))
    result = compare_optimizers(centered, "y ~ ga_c + bmi_c", methods=("lbfgs",))
    fit = result["fits"][0]
    assert fit["method"] == "lbfgs"
    assert np.isfinite(fit["random_intercept_sd"])
    assert fit["random_intercept_sd"] > 0
    assert isinstance(fit["warnings"], list)


def test_figures_write_png(tmp_path):
    male = pd.DataFrame(
        {
            "ga_weeks": np.linspace(12, 22, 40),
            "bmi": np.linspace(22, 40, 40),
            "y_frac": np.linspace(0.03, 0.1, 40),
        }
    )
    bins = pd.DataFrame(binned_profile(pd.DataFrame({"ga": male["ga_weeks"], "bmi": male["bmi"], "y": male["y_frac"]}), width=2))
    bins = bins.rename(columns={})
    curves = lognormal_curves(
        [
            {"group": 0, "interval": "[20, 32)", "success": True, "mu": 2.4, "sigma": 0.45, "t_star": 14.0},
            {"group": 1, "interval": "[32, 45]", "success": True, "mu": 2.8, "sigma": 0.35, "t_star": 20.0},
        ]
    )
    groups = pd.DataFrame(
        {
            "interval": ["[20, 32)", "[32, 45]"],
            "n_women": [30, 20],
            "t_star": [14.0, 20.0],
            "t_boot_low": [13.0, 17.0],
            "t_boot_high": [15.0, 22.0],
        }
    )
    female = pd.DataFrame(
        {
            "z13": np.linspace(-2, 4, 30),
            "z18": np.linspace(-1, 3, 30),
            "z21": np.linspace(-2, 2, 30),
            "ab_t13": [0] * 24 + [1] * 6,
            "ab_t18": [0] * 20 + [1] * 10,
            "ab_t21": [0] * 27 + [1] * 3,
        }
    )
    written = write_figures(
        male=male,
        female=female,
        bins=bins,
        curves2=curves,
        curves3=curves,
        groups3=groups,
        folders=[tmp_path],
    )
    names = {path.rsplit("/", 1)[-1] for path in written}
    assert names == {"y_vs_ga_bmi.png", "problem2_risk_curves.png", "problem3_timing.png", "problem4_z_by_ab.png"}
    for name in names:
        assert (tmp_path / name).stat().st_size > 2000
