"""运行四问修订分析。缺少附件时返回退出码 2，不写入编造的数值。"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .constants import N_BOOT_TIME, N_NOISE, RANDOM_SEED
from .errors import WorkbookNotFoundError
from .load import prepare
from .problem1_relationship import run_problem1
from .problem2_timing import public_groups as public_problem2
from .problem2_timing import run_problem2
from .problem3_multifactor import public_groups as public_problem3
from .problem3_multifactor import run_problem3
from .figures import write_figures
from .problem4_female import run_problem4
from .reporting import write_frame, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NIPT 赛后修订分析（需要本地附件.xlsx，不会把该文件写入 Git）")
    parser.add_argument("--workbook", default=None, help="官方附件路径，默认是仓库根目录的附件.xlsx")
    parser.add_argument("--boot", type=int, default=N_BOOT_TIME)
    parser.add_argument("--noise", type=int, default=N_NOISE)
    parser.add_argument("--corr-boot", type=int, default=999)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.workbook)
    except WorkbookNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("已读取本地附件。开始问题一。", flush=True)
    problem1 = run_problem1(prepared["male"], n_boot=args.corr_boot, seed=args.seed)
    print("开始问题二。", flush=True)
    problem2 = run_problem2(
        prepared["male"],
        hits=prepared["hits"],
        n_boot=args.boot,
        n_noise=args.noise,
        seed=args.seed,
    )
    print("开始问题三。", flush=True)
    problem3 = run_problem3(
        prepared["male"],
        hits=prepared["hits"],
        n_boot=args.boot,
        n_noise=args.noise,
        seed=args.seed,
    )
    print("开始问题四。", flush=True)
    problem4 = run_problem4(prepared["female"], seed=args.seed)
    _write_outputs(prepared["notes"], problem1, problem2, problem3, problem4)
    written = write_figures(
        male=prepared["male"],
        female=prepared["female"],
        bins=pd.DataFrame(problem1.get("bins", [])),
        curves2=problem2.get("curves", pd.DataFrame()),
        curves3=pd.DataFrame(problem3.get("curves", [])),
        groups3=public_problem3(problem3["groups"]),
    )
    print("修订分析已写到 results/revised、paper/tables 与 paper/figures。", flush=True)
    if written:
        print("图：", ", ".join(sorted({path.rsplit('/', 1)[-1] for path in written})), flush=True)
    return 0


def _write_outputs(notes, problem1, problem2, problem3, problem4) -> None:
    write_json(notes, "sample_notes.json")
    write_frame(pd.DataFrame(problem1["correlations"]), "problem1_correlations.csv")
    write_frame(pd.DataFrame(problem1["coefficients"]), "problem1_coefficients.csv")
    write_frame(pd.DataFrame(problem1["cv"]), "problem1_cv.csv")
    write_json(
        {
            "n_rows": problem1["n_rows"],
            "n_women": problem1["n_women"],
            "dropped_rows": problem1["dropped_rows"],
            "centering": problem1["centering"],
            "vif_additive": problem1["vif_additive"],
            "mixed": problem1["mixed"],
            "nonlinearity": problem1.get("nonlinearity"),
            "optimizers": problem1.get("optimizers"),
            "bmi_definition": problem1["bmi_definition"],
        },
        "problem1_models.json",
    )
    if problem1.get("bins"):
        write_frame(pd.DataFrame(problem1["bins"]), "problem1_bins.csv")
    curves2 = problem2.get("curves")
    if isinstance(curves2, pd.DataFrame) and not curves2.empty:
        write_frame(curves2, "problem2_curves.csv")
    if problem3.get("curves"):
        write_frame(pd.DataFrame(problem3["curves"]), "problem3_curves.csv")
    write_frame(public_problem2(problem2["driven"]), "problem2_groups.csv")
    write_frame(public_problem2(problem2["official"]), "problem2_official_bins.csv")
    write_frame(pd.DataFrame(problem2["selection"]), "problem2_selection.csv")
    write_frame(pd.DataFrame(problem2["sensitivity"]), "problem2_sensitivity.csv")
    write_json(
        {
            "n_women": problem2["n_women"],
            "chosen_k": problem2["chosen_k"],
            "bmi_definition": problem2["bmi_definition"],
            "replicate_sigma_y": problem2["replicate_sigma_y"],
            "primary_slopes": problem2["primary_slopes"],
            "primary_lambda": problem2["primary_lambda"],
            "official_nll": problem2["official_nll"],
        },
        "problem2_summary.json",
    )
    write_frame(public_problem3(problem3["groups"]), "problem3_groups.csv")
    write_json(
        {
            "n_women": problem3["n_women"],
            "chosen_k": problem3["chosen_k"],
            "cv_scores": problem3["cv_scores"],
            "alternatives": problem3["alternatives"],
            "bmi_definition": problem3["bmi_definition"],
            "primary_aft": problem3["primary_aft"],
            "sensitivity_aft": problem3["sensitivity_aft"],
            "primary_aic": problem3["primary_aic"],
            "sensitivity_aic": problem3["sensitivity_aic"],
            "vif_primary": problem3["vif_primary"],
            "vif_sensitivity": problem3["vif_sensitivity"],
            "lmm": problem3["lmm"],
        },
        "problem3_models.json",
    )
    baseline_rows = []
    cv_rows = []
    inference_rows = []
    threshold_rows = []
    for name, result in problem4["chromosomes"].items():
        for key in ("baseline_all_finite_z", "baseline_model_sample"):
            if key in result:
                row = dict(result[key])
                row["chromosome"] = name
                row["which"] = key
                baseline_rows.append(row)
        cv = result.get("cv", {})
        if "error" not in cv:
            flat = {key: value for key, value in cv.items() if not isinstance(value, dict)}
            flat["chromosome"] = name
            cv_rows.append(flat)
            for zone_name in ("zone_positive", "zone_flag"):
                if zone_name in cv:
                    zone = dict(cv[zone_name])
                    zone["chromosome"] = name
                    zone["which"] = zone_name
                    baseline_rows.append(zone)
        for row in result.get("inference", {}).get("coefficients", []):
            copied = dict(row)
            copied["chromosome"] = name
            inference_rows.append(copied)
        for row in result.get("illustrative_split", {}).get("thresholds", []):
            copied = dict(row)
            copied["chromosome"] = name
            threshold_rows.append(copied)
    write_frame(pd.DataFrame(baseline_rows), "problem4_baseline.csv")
    write_frame(pd.DataFrame(cv_rows), "problem4_cv.csv")
    write_frame(pd.DataFrame(inference_rows), "problem4_coefficients.csv")
    if threshold_rows:
        write_frame(pd.DataFrame(threshold_rows), "problem4_zone_thresholds.csv")
    if problem4.get("z_by_ab"):
        write_frame(pd.DataFrame(problem4["z_by_ab"]), "problem4_z_by_ab.csv")
    summary = {key: value for key, value in problem4.items() if key != "z_by_ab"}
    summary["z_by_ab_table"] = "problem4_z_by_ab.csv"
    write_json(summary, "problem4_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
