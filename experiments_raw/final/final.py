import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import json
import sys
import time

model_paths = [
    "",
    "/kaggle/input/models/leonidtikhanov/pinn-model/pytorch/default/8",
    "/kaggle/input/models/leonidtikhanov/pinn-model/pytorch/default/7",
]

for p in model_paths:
    if p == "" and Path("pinn_model.py").exists():
        sys.path.insert(0, p)
        break
    if p != "" and Path(p).exists():
        sys.path.insert(0, p)
        break

import pinn_model
pinn_model = importlib.reload(pinn_model)

print(pinn_model.__file__)
print("run_experiment:", hasattr(pinn_model, "run_experiment"))
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = "cuda"
    print("gpu:", torch.cuda.get_device_name(0))
else:
    device = "cpu"

work_dir = Path("/kaggle/working")
if not work_dir.exists():
    work_dir = Path(".")

out_dir = work_dir / "final"
out_dir.mkdir(parents=True, exist_ok=True)

print("device:", device)
print("work_dir:", work_dir)
print("out_dir:", out_dir)

dtype_values = ["fp32", "fp64"]
if device != "cpu":
    try:
        pinn_model.get_dtype("fp16")
        dtype_values.append("fp16")
    except Exception:
        print("fp16 is not supported by this pinn_model, skipping fp16")

base_config = {
    "device": device,
    "use_adam": True,
    "use_lbfgs": True,
}

runs = []
i = 1

for nu in [0.001, 0.002]:
    for dtype in dtype_values:
        for seed in [0, 1]:
            row = {
                "run_id": i,
                "task_name": "burgers1d",
                "variant": "burgers_more_points",
                "case_name": "nu",
                "case_value": nu,
                "nu": nu,
                "dtype": dtype,
                "seed": seed,
                "hid_size": 256,
                "num_layers": 4,
                "init_gain": 1.0,
                "n_collocation": 10000,
                "n_ic": 800,
                "n_bc": 800,
                "adam_steps": 3000,
                "lr_adam": 5e-5,
                "lbfgs_steps": 2000,
                "lbfgs_max_iter": 5,
                "lbfgs_max_eval": None,
                "lbfgs_lr": 0.5,
                "lbfgs_history_size": 100,
                "lbfgs_tolerance_grad": 1e-8,
                "lbfgs_tolerance_change": 1e-10,
                "lbfgs_line_search_fn": "strong_wolfe",
                "resample_every": 200,
            }
            runs.append(row)
            i += 1

for m in [8, 12]:
    for dtype in dtype_values:
        for seed in [0, 1]:
            row = {
                "run_id": i,
                "task_name": "helmholtz1d",
                "variant": "helmholtz_resample_long",
                "case_name": "m",
                "case_value": m,
                "m": m,
                "lambda_val": 1.0,
                "dtype": dtype,
                "seed": seed,
                "hid_size": 128,
                "num_layers": 4,
                "init_gain": None,
                "n_collocation": 5000,
                "n_ic": 0,
                "n_bc": 2,
                "adam_steps": 10000,
                "lr_adam": 5e-4,
                "lbfgs_steps": 1000,
                "lbfgs_max_iter": 1,
                "lbfgs_max_eval": None,
                "lbfgs_lr": 1.0,
                "lbfgs_history_size": 50,
                "lbfgs_tolerance_grad": 1e-8,
                "lbfgs_tolerance_change": 1e-9,
                "lbfgs_line_search_fn": "strong_wolfe",
                "resample_every": 200,
            }
            runs.append(row)
            i += 1

for beta in [30.0]:
    for dtype in dtype_values:
        for seed in [0, 1]:
            row = {
                "run_id": i,
                "task_name": "convection1d",
                "variant": "convection_beta30_lbfgs_grid",
                "case_name": "beta",
                "case_value": beta,
                "beta": beta,
                "dtype": dtype,
                "seed": seed,
                "hid_size": 256,
                "num_layers": 3,
                "init_gain": 1.0,
                "point_mode": "grid",
                "n_grid": 65,
                "n_collocation": 4225,
                "n_ic": 65,
                "n_bc": 65,
                "adam_steps": 0,
                "lr_adam": 1e-4,
                "lbfgs_steps": 4000,
                "lbfgs_max_iter": 10,
                "lbfgs_max_eval": None,
                "lbfgs_lr": 1.0,
                "lbfgs_history_size": 50,
                "lbfgs_tolerance_grad": 1e-8,
                "lbfgs_tolerance_change": 1e-9,
                "lbfgs_line_search_fn": "strong_wolfe",
                "resample_every": 0,
            }
            runs.append(row)
            i += 1

for beta in [50.0]:
    for dtype in dtype_values:
        for seed in [0]:
            row = {
                "run_id": i,
                "task_name": "convection1d",
                "variant": "convection_beta50_wide_lbfgs",
                "case_name": "beta",
                "case_value": beta,
                "beta": beta,
                "dtype": dtype,
                "seed": seed,
                "hid_size": 512,
                "num_layers": 3,
                "init_gain": 1.0,
                "point_mode": "grid",
                "n_grid": 101,
                "n_collocation": 10201,
                "n_ic": 101,
                "n_bc": 101,
                "adam_steps": 0,
                "lr_adam": 5e-5,
                "lbfgs_steps": 5000,
                "lbfgs_max_iter": 10,
                "lbfgs_max_eval": None,
                "lbfgs_lr": 0.5,
                "lbfgs_history_size": 100,
                "lbfgs_tolerance_grad": 1e-8,
                "lbfgs_tolerance_change": 1e-10,
                "lbfgs_line_search_fn": "strong_wolfe",
                "resample_every": 0,
            }
            runs.append(row)
            i += 1

run_plan = pd.DataFrame(runs)
run_plan.to_csv(out_dir / "final_run_plan.csv", index=False)
print("runs:", len(runs))
print(run_plan[["run_id", "task_name", "variant", "case_name", "case_value", "dtype", "seed"]].to_string(index=False))

all_summaries = []
all_histories = {}

def tag_value(x):
    return str(x).replace(".", "p")

for run in runs:
    if device == "cpu" and run["dtype"] == "fp16":
        continue

    config = base_config.copy()
    config.update(run)
    config["use_adam"] = config["adam_steps"] > 0
    config["use_lbfgs"] = config["lbfgs_steps"] > 0
    case_tag = f"{run['case_name']}{tag_value(run['case_value'])}"
    name = f"run{run['run_id']:03d}_{run['task_name']}_{run['variant']}_{case_tag}_{run['dtype']}_s{run['seed']}"
    config["log_dir"] = str(out_dir / "runs" / name)

    run_dir = Path(config["log_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics.csv"
    error_text = None
    start = time.time()

    if summary_file.exists() and metrics_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        history = pd.read_csv(metrics_file)
    else:
        try:
            history, summary = pinn_model.run_experiment(config)
        except Exception as err:
            error_text = str(err)
            history = pd.DataFrame([{
                "step": 0,
                "opt": "error",
                "total_loss": np.nan,
                "pde_loss": np.nan,
                "ic_loss": np.nan,
                "bc_loss": np.nan,
                "l2_error": np.nan,
                "time_sec": time.time() - start,
            }])
            history.to_csv(metrics_file, index=False)
            summary = {
                "task_name": run["task_name"],
                "dtype": run["dtype"],
                "seed": run["seed"],
                "final_loss": None,
                "final_l2_error": None,
                "time_sec": float(history["time_sec"].iloc[-1]),
                "log_dir": str(run_dir),
                "error": error_text,
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

    if "l2_error" in history.columns and history["l2_error"].notna().any():
        best_row = history.loc[history["l2_error"].idxmin()]
        best_l2 = float(best_row["l2_error"])
        best_step = int(best_row["step"])
        final_l2 = float(history["l2_error"].iloc[-1])
        elapsed = float(history["time_sec"].iloc[-1])
    else:
        best_l2 = np.nan
        best_step = np.nan
        final_l2 = np.nan
        elapsed = float(history["time_sec"].iloc[-1]) if "time_sec" in history.columns else np.nan

    summary["run_id"] = run["run_id"]
    summary["run_name"] = name
    summary["variant"] = run["variant"]
    summary["case_name"] = run["case_name"]
    summary["case_value"] = run["case_value"]
    summary["dtype"] = run["dtype"]
    summary["seed"] = run["seed"]
    summary["hid_size"] = run["hid_size"]
    summary["num_layers"] = run["num_layers"]
    summary["init_gain"] = run["init_gain"]
    summary["n_collocation"] = run["n_collocation"]
    summary["n_ic"] = run["n_ic"]
    summary["n_bc"] = run["n_bc"]
    summary["adam_steps"] = run["adam_steps"]
    summary["lr_adam"] = run["lr_adam"]
    summary["lbfgs_steps"] = run["lbfgs_steps"]
    summary["lbfgs_max_iter"] = run["lbfgs_max_iter"]
    summary["lbfgs_max_eval"] = run["lbfgs_max_eval"]
    summary["lbfgs_lr"] = run["lbfgs_lr"]
    summary["lbfgs_history_size"] = run["lbfgs_history_size"]
    summary["lbfgs_tolerance_grad"] = run["lbfgs_tolerance_grad"]
    summary["lbfgs_tolerance_change"] = run["lbfgs_tolerance_change"]
    summary["lbfgs_line_search_fn"] = run["lbfgs_line_search_fn"]
    summary["resample_every"] = run["resample_every"]
    summary["best_l2_error"] = best_l2
    summary["best_step"] = best_step
    summary["final_l2_error"] = final_l2
    summary["elapsed_time"] = elapsed
    if "nu" in run:
        summary["nu"] = run["nu"]
    if "m" in run:
        summary["m"] = run["m"]
    if "beta" in run:
        summary["beta"] = run["beta"]
    if "point_mode" in run:
        summary["point_mode"] = run["point_mode"]
    if "n_grid" in run:
        summary["n_grid"] = run["n_grid"]
    if error_text is not None:
        summary["error"] = error_text

    all_summaries.append(summary)
    all_histories[name] = history
    print(run["run_id"], run["task_name"], run["variant"], case_tag, run["dtype"], "seed", run["seed"], "final", final_l2, "best", best_l2, "time", elapsed)

summary_df = pd.DataFrame(all_summaries)
summary_path = out_dir / "final_summary.csv"
summary_df.to_csv(summary_path, index=False)

grouped = summary_df.groupby(["task_name", "variant", "case_name", "case_value", "dtype"])[["best_l2_error", "final_l2_error", "elapsed_time"]].agg(["mean", "std", "min", "max"])
grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
grouped = grouped.reset_index()
grouped_path = out_dir / "final_grouped.csv"
grouped.to_csv(grouped_path, index=False)

best_ratio = grouped.pivot_table(index=["task_name", "variant", "case_name", "case_value"], columns="dtype", values="best_l2_error_mean").reset_index()
best_ratio = best_ratio.rename(columns={c: f"{c}_best_l2_error_mean" for c in ["fp32", "fp64", "fp16"] if c in best_ratio.columns})
final_ratio = grouped.pivot_table(index=["task_name", "variant", "case_name", "case_value"], columns="dtype", values="final_l2_error_mean").reset_index()
final_ratio = final_ratio.rename(columns={c: f"{c}_final_l2_error_mean" for c in ["fp32", "fp64", "fp16"] if c in final_ratio.columns})
ratio = best_ratio.merge(final_ratio, on=["task_name", "variant", "case_name", "case_value"], how="outer")
if "fp32_best_l2_error_mean" in ratio.columns and "fp64_best_l2_error_mean" in ratio.columns:
    ratio["fp64_over_fp32_best"] = ratio["fp64_best_l2_error_mean"] / ratio["fp32_best_l2_error_mean"]
if "fp32_best_l2_error_mean" in ratio.columns and "fp16_best_l2_error_mean" in ratio.columns:
    ratio["fp16_over_fp32_best"] = ratio["fp16_best_l2_error_mean"] / ratio["fp32_best_l2_error_mean"]
if "fp32_final_l2_error_mean" in ratio.columns and "fp64_final_l2_error_mean" in ratio.columns:
    ratio["fp64_over_fp32_final"] = ratio["fp64_final_l2_error_mean"] / ratio["fp32_final_l2_error_mean"]
if "fp32_final_l2_error_mean" in ratio.columns and "fp16_final_l2_error_mean" in ratio.columns:
    ratio["fp16_over_fp32_final"] = ratio["fp16_final_l2_error_mean"] / ratio["fp32_final_l2_error_mean"]
ratio_path = out_dir / "final_ratio.csv"
ratio.to_csv(ratio_path, index=False)

bad_mask = summary_df["best_l2_error"].isna() | (summary_df["best_l2_error"] > 0.2) | (summary_df["final_l2_error"] > 0.5)
bad = summary_df[bad_mask].copy()
bad_path = out_dir / "final_bad_runs.csv"
bad.to_csv(bad_path, index=False)

for task_name in summary_df["task_name"].dropna().unique():
    task_df = summary_df[summary_df["task_name"] == task_name]
    for variant in task_df["variant"].dropna().unique():
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        cur = task_df[task_df["variant"] == variant]
        for row in cur.sort_values(["case_value", "dtype", "seed"]).itertuples():
            hist = all_histories.get(row.run_name)
            if hist is None or len(hist) == 0:
                continue
            if "total_loss" not in hist.columns or "l2_error" not in hist.columns:
                continue
            label = f"{row.case_name}={row.case_value} {row.dtype} s{row.seed}"
            ax[0].plot(hist["step"], hist["total_loss"], label=label)
            ax[1].plot(hist["step"], hist["l2_error"], label=label)
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].set_xlabel("step")
        ax[1].set_xlabel("step")
        ax[0].set_ylabel("total_loss")
        ax[1].set_ylabel("l2_error")
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].legend(fontsize=7)
        ax[1].legend(fontsize=7)
        fig.suptitle(f"{task_name} {variant}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{task_name}_{variant}_loss_l2.png", dpi=150)
        plt.close(fig)

print("saved:", summary_path)
print("saved:", grouped_path)
print("saved:", ratio_path)
print("saved:", bad_path)
