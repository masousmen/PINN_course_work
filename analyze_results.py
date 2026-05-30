from pathlib import Path
import runpy
import shutil

import pandas as pd


root = Path(__file__).resolve().parent
clean_dir = root / "report_results_clean"
out_dir = root / "report_results"
table_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
selected_dir = out_dir / "selected_runs"
rerun_dir = out_dir / "rerun_plan"


def copy_file(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def fmt_num(x):
    if pd.isna(x):
        return "nan"
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def result_text(row, dtype):
    n = row.get(f"n_{dtype}", row.get(f"{dtype}_n_valid", ""))
    med = row.get(f"{dtype}_median_best_l2", float("nan"))
    bad = row.get(f"{dtype}_bad_rate", float("nan"))
    return f"n={n}, median best L2={fmt_num(med)}, bad rate={fmt_num(bad)}"


def public_conclusion(x):
    if x in ["single_seed_hard_case", "seed_sensitive"]:
        return "unstable_or_seed_sensitive"
    return x


def make_selected_cases():
    src = clean_dir / "tables" / "report_cases.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "task_name": row["task_name"],
            "parameter": row["parameter"],
            "case_title": row.get("case_title", row["case_id"]),
            "variant": row["variant"],
            "why_selected": row["why_selected"],
            "fp32_result": "" if row["task_name"] == "fp16" else result_text(row, "fp32"),
            "fp64_result": "" if row["task_name"] == "fp16" else result_text(row, "fp64"),
            "conclusion": public_conclusion(row["conclusion"]),
            "confidence_label": row["confidence_label"],
            "source_runs": row.get("source_paths", ""),
            "main_parameter_name": row.get("main_parameter_name", ""),
            "main_parameter_value": row.get("main_parameter_value", ""),
        })
    selected = pd.DataFrame(rows)
    selected.to_csv(table_dir / "selected_cases.csv", index=False)


def write_report_readme():
    runs = pd.read_csv(clean_dir / "tables" / "all_runs.csv")
    selected = pd.read_csv(table_dir / "selected_cases.csv")
    fp16 = pd.read_csv(clean_dir / "tables" / "fp16_summary.csv")
    valid = int(runs["is_valid"].sum())
    bad = int(runs["is_bad"].sum())
    invalid = int((~runs["is_valid"]).sum())

    lines = [
        "# Report results",
        "",
        "These files are the cleaned output layer for the report.",
        "They are rebuilt from real run folders with `summary.json` and `metrics.csv`; old aggregate CSV files are not used as run sources.",
        "",
        f"Unique run folders found: {len(runs)}.",
        f"Valid runs: {valid}.",
        f"Invalid runs: {invalid}.",
        f"Bad or unstable by threshold: {bad}.",
        "",
        "`bad_runs.csv` includes invalid runs and valid runs with high error, so bad and valid counts can overlap.",
        "",
        "## Main tables",
        "",
        "- `tables/selected_cases.csv`",
        "- `tables/grouped_by_dtype.csv`",
        "- `tables/fp32_fp64_comparison.csv`",
        "- `tables/fp16_summary.csv`",
        "- `tables/run_quality.csv`",
        "",
        "## Main figures",
        "",
        "- `figures/report_best_l2_by_dtype.png`",
        "- `figures/report_fp64_fp32_ratio.png`",
        "- `figures/report_seed_scatter.png`",
        "- `figures/report_convection_beta50_curves.png`",
        "- `figures/report_helmholtz_m12_curves.png`",
        "",
        "## Selected cases",
        "",
    ]
    for _, row in selected.iterrows():
        lines.append(f"- `{row['case_title']}`: {row['conclusion']} ({row['confidence_label']})")
    lines.extend([
        "",
        "## Report wording",
        "",
        "Can say:",
        "- FP64 improves selected hard cases.",
        "- FP64 is not uniformly better than FP32.",
        "- FP16 is mostly unstable in these runs.",
        "- Some cases are seed-sensitive.",
        "",
        "Should not say:",
        "- FP64 always wins.",
        "- FP32 fails everywhere.",
        "- FP16 was fully evaluated as a stable baseline.",
        "",
        f"FP16 groups in the separate summary: {len(fp16)}.",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_missing_artifacts():
    lines = [
        "# Missing artifacts",
        "",
        "## Already available",
        "",
        "- tables with best/final relative L2",
        "- loss/L2 curves for selected cases",
        "- FP32/FP64 comparison",
        "- FP16 summary",
        "- selected cases for the report",
        "",
        "## Still missing for a polished report",
        "",
        "- MAE/RMSE for selected convection cases, if dense-grid metrics are needed",
        "- exact/prediction/error maps for convection beta=50",
        "- one FP16 selected check on convection beta=50, if a failure illustration is needed",
        "",
        "## Large rerun",
        "",
        "Large rerun is not needed.",
        "",
        "## Minimal selected checks",
        "",
        "- convection_beta50_fp32_seed0",
        "- convection_beta50_fp64_seed0",
        "- convection_beta50_fp16_seed0",
        "",
        "Do not rerun all 33 final runs for the report.",
    ]
    (rerun_dir / "missing_artifacts.md").write_text("\n".join(lines) + "\n")


def sync_outputs():
    for p in [table_dir, fig_dir, selected_dir, rerun_dir]:
        p.mkdir(parents=True, exist_ok=True)

    copies = [
        ("all_runs.csv", "all_runs_normalized.csv"),
        ("run_quality.csv", "run_quality.csv"),
        ("grouped_by_dtype.csv", "grouped_by_dtype.csv"),
        ("fp32_fp64_comparison.csv", "fp32_fp64_comparison.csv"),
        ("fp16_summary.csv", "fp16_summary.csv"),
        ("bad_runs.csv", "bad_runs.csv"),
        ("valid_runs.csv", "valid_runs.csv"),
    ]
    for src_name, dst_name in copies:
        copy_file(clean_dir / "tables" / src_name, table_dir / dst_name)

    copy_file(clean_dir / "selected_runs" / "selected_run_paths.csv", selected_dir / "selected_run_paths.csv")
    make_selected_cases()

    fig_copies = [
        ("report_best_l2.png", "report_best_l2_by_dtype.png"),
        ("report_ratio.png", "report_fp64_fp32_ratio.png"),
        ("report_seed_scatter.png", "report_seed_scatter.png"),
        ("curves_convection_beta50.png", "report_convection_beta50_curves.png"),
        ("curves_helmholtz_m12.png", "report_helmholtz_m12_curves.png"),
        ("report_best_l2.png", "fp32_fp64_median_best_l2.png"),
        ("report_ratio.png", "fp64_over_fp32_ratio.png"),
        ("report_seed_scatter.png", "seed_scatter_best_l2.png"),
    ]
    for src_name, dst_name in fig_copies:
        copy_file(clean_dir / "figures" / src_name, fig_dir / dst_name)

    write_report_readme()
    write_missing_artifacts()


def main():
    ns = runpy.run_path(str(root / "scripts" / "build_report_results.py"))
    ns["main"]()
    sync_outputs()
    print("report_results updated")


if __name__ == "__main__":
    main()
