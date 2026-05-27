import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import json
import sys

try:
    from IPython.display import Image, display
except Exception:
    Image = None
    display = None

model_paths = [
    "/kaggle/input/models/leonidtikhanov/pinn-model/pytorch/default/7",
    "",
]

for p in model_paths:
    if Path(p).exists():
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

out_dir = work_dir / "results_exp_15_helmholtz1d_lr_check"
out_dir.mkdir(parents=True, exist_ok=True)

print("device:", device)
print("work_dir:", work_dir)
print("out_dir:", out_dir)

show_run_plots = True

m_values = [8, 10, 12]
dtype_values = ["fp32", "fp64"]
seed_values = [0]

variants = [
    {
        "variant": "adam5e4_lbfgs1_inner1_resample200",
        "lr_adam": 5e-4,
        "lbfgs_lr": 1.0,
    },
    {
        "variant": "adam2e4_lbfgs1_inner1_resample200",
        "lr_adam": 2e-4,
        "lbfgs_lr": 1.0,
    },
    {
        "variant": "adam2e4_lbfgs05_inner1_resample200",
        "lr_adam": 2e-4,
        "lbfgs_lr": 0.5,
    },
]

runs = []
i = 1
for m in m_values:
    for variant in variants:
        for dtype in dtype_values:
            for seed in seed_values:
                run = variant.copy()
                run["run_id"] = i
                run["m"] = m
                run["dtype"] = dtype
                run["seed"] = seed
                runs.append(run)
                i += 1

print("runs:", len(runs))
print(pd.DataFrame(runs)[["run_id", "m", "variant", "dtype", "seed", "lr_adam", "lbfgs_lr"]])

base_config = {
    "task_name": "helmholtz1d",
    "dtype": "fp32",
    "seed": 0,
    "device": device,
    "m": 8,
    "lambda_val": 1.0,
    "hid_size": 128,
    "num_layers": 4,
    "init_gain": None,
    "n_collocation": 5000,
    "point_mode": None,
    "n_bc": 2,
    "adam_steps": 10000,
    "lbfgs_steps": 1500,
    "lr_adam": 5e-4,
    "resample_every": 200,
    "use_adam": True,
    "use_lbfgs": True,
    "lbfgs_tolerance_grad": 1e-8,
    "lbfgs_tolerance_change": 1e-9,
    "lbfgs_history_size": 50,
    "lbfgs_lr": 1.0,
    "lbfgs_max_iter": 1,
    "lbfgs_max_eval": None,
    "lbfgs_line_search_fn": "strong_wolfe",
    "hard_bc": False,
    "bc_weight": 1.0,
    "log_dir": str(out_dir / "runs" / "helmholtz_tmp"),
}

all_summaries = []
all_histories = {}

for run in runs:
    config = base_config.copy()
    config.update(run)
    config["use_adam"] = config["adam_steps"] > 0
    config["use_lbfgs"] = config["lbfgs_steps"] > 0
    name = f"exp15_r{run['run_id']:03d}_m{run['m']}_{run['variant']}_{run['dtype']}_s{run['seed']}"
    config["log_dir"] = str(out_dir / "runs" / name)

    run_dir = Path(config["log_dir"])
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics.csv"

    if summary_file.exists() and metrics_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        history = pd.read_csv(metrics_file)
    else:
        history, summary = pinn_model.run_experiment(config)

    summary["run_id"] = run["run_id"]
    summary["run_name"] = name
    summary["variant"] = run["variant"]
    summary["m"] = run["m"]
    summary["dtype"] = run["dtype"]
    summary["seed"] = run["seed"]
    summary["lr_adam"] = run["lr_adam"]
    summary["lbfgs_lr"] = run["lbfgs_lr"]
    summary["best_l2_error"] = float(history["l2_error"].min())
    summary["final_l2_error"] = float(history["l2_error"].iloc[-1])
    summary["elapsed_time"] = float(history["time_sec"].iloc[-1])
    all_summaries.append(summary)
    all_histories[name] = history

    print(
        run["run_id"],
        run["variant"],
        "m",
        run["m"],
        run["dtype"],
        "seed",
        run["seed"],
        "final",
        summary["final_l2_error"],
        "best",
        summary["best_l2_error"],
        "time",
        summary["elapsed_time"],
    )

summary_df = pd.DataFrame(all_summaries)
summary_path = out_dir / "exp_15_helmholtz1d_lr_check_summary.csv"
summary_df.to_csv(summary_path, index=False)

cols = [
    "run_id",
    "variant",
    "m",
    "dtype",
    "seed",
    "lr_adam",
    "lbfgs_lr",
    "best_l2_error",
    "final_l2_error",
    "elapsed_time",
]

print(summary_df[cols].sort_values(["m", "variant", "dtype", "seed"]).to_string(index=False))

grouped = summary_df.groupby(["variant", "m", "dtype"])[["best_l2_error", "final_l2_error", "elapsed_time"]].agg(["mean", "std", "min", "max"])
grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
grouped = grouped.reset_index()
grouped_path = out_dir / "exp_15_helmholtz1d_lr_check_grouped.csv"
grouped.to_csv(grouped_path, index=False)

ratio = grouped.pivot_table(index=["variant", "m"], columns="dtype", values="best_l2_error_mean").reset_index()
if "fp32" in ratio.columns and "fp64" in ratio.columns:
    ratio["fp64_over_fp32_best"] = ratio["fp64"] / ratio["fp32"]
ratio_path = out_dir / "exp_15_helmholtz1d_lr_check_fp64_ratio.csv"
ratio.to_csv(ratio_path, index=False)

print(ratio.to_string(index=False))

for variant in summary_df["variant"].unique():
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for name, hist in all_histories.items():
        row = summary_df[summary_df["run_name"] == name].iloc[0]
        if row["variant"] != variant:
            continue
        label = f"m={row['m']} {row['dtype']} s{row['seed']}"
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
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.suptitle(variant)
    fig.tight_layout()
    fig.savefig(out_dir / f"{variant}_curves.png", dpi=150)
    plt.show()

if show_run_plots and display is not None:
    for row in summary_df.sort_values(["m", "variant", "dtype", "seed"]).itertuples():
        p = Path(row.log_dir) / "curves.png"
        if p.exists():
            print(row.run_name)
            display(Image(filename=str(p)))

bad = summary_df[summary_df["best_l2_error"] > 0.5]
bad_path = out_dir / "exp_15_helmholtz1d_lr_check_bad_runs.csv"
bad[cols].to_csv(bad_path, index=False)

print("saved:", summary_path)
print("saved:", grouped_path)
print("saved:", ratio_path)
print("saved:", bad_path)
