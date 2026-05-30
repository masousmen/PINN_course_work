import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import pinn_model


out_dir = root / "results_selected_checks"
out_dir.mkdir(exist_ok=True)


def dense_maps(run_dir, dtype_name, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pinn_model.get_dtype(dtype_name)
    torch.set_default_dtype(dtype)

    model = pinn_model.PINN(input_dim=2, hid_size=256, num_layers=3).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    n = 200
    x = torch.linspace(0, 2 * torch.pi, n, device=device).reshape(-1, 1)
    t = torch.linspace(0, 1, n, device=device).reshape(-1, 1)
    xx, tt = torch.meshgrid(x.reshape(-1), t.reshape(-1), indexing="ij")
    inp_x = xx.reshape(-1, 1)
    inp_t = tt.reshape(-1, 1)

    with torch.no_grad():
        pred = model(inp_x, inp_t)
        exact = pinn_model.convection_1d_solution(inp_x, inp_t, 50.0)

    diff = pred - exact
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    rel_mae = mae / torch.mean(torch.abs(exact)).item()
    rel_rmse = rmse / torch.sqrt(torch.mean(exact ** 2)).item()
    rel_l2 = torch.linalg.norm(diff).item() / torch.linalg.norm(exact).item()

    extra = {
        "mae": mae,
        "rmse": rmse,
        "relative_mae": rel_mae,
        "relative_rmse": rel_rmse,
        "relative_l2": rel_l2,
    }
    (run_dir / "metrics_extra.json").write_text(json.dumps(extra, indent=2))

    pred_np = pred.detach().cpu().numpy().reshape(n, n)
    exact_np = exact.detach().cpu().numpy().reshape(n, n)
    err_np = np.abs(pred_np - exact_np)

    for arr, name, title in [
        (pred_np, "solution_map.png", f"prediction {dtype_name} seed={seed}"),
        (err_np, "error_map.png", f"absolute error {dtype_name} seed={seed}"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=[0, 2 * np.pi, 0, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / name, dpi=160)
        plt.close(fig)


def run_one(dtype_name, seed):
    if dtype_name == "fp16" and not torch.cuda.is_available():
        return

    run_name = f"convection_beta50_{dtype_name}_seed{seed}"
    log_dir = out_dir / run_name
    if (log_dir / "summary.json").exists() and (log_dir / "metrics.csv").exists():
        if (log_dir / "model.pt").exists() and not (log_dir / "metrics_extra.json").exists():
            dense_maps(log_dir, dtype_name, seed)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "task_name": "convection1d",
        "beta": 50.0,
        "dtype": dtype_name,
        "seed": seed,
        "device": device,
        "hid_size": 256,
        "num_layers": 3,
        "init_gain": 1.0,
        "point_mode": "grid",
        "n_grid": 81,
        "n_collocation": 6561,
        "n_ic": 81,
        "n_bc": 81,
        "use_adam": True,
        "adam_steps": 2000,
        "lr_adam": 5e-5,
        "use_lbfgs": True,
        "lbfgs_steps": 3000,
        "lbfgs_max_iter": 5,
        "lbfgs_max_eval": None,
        "lbfgs_lr": 0.2,
        "lbfgs_history_size": 100,
        "lbfgs_tolerance_grad": 1e-8,
        "lbfgs_tolerance_change": 1e-10,
        "lbfgs_line_search_fn": "strong_wolfe",
        "resample_every": 0,
        "log_dir": str(log_dir),
    }
    pinn_model.run_experiment(config)
    if (log_dir / "model.pt").exists():
        dense_maps(log_dir, dtype_name, seed)


def main():
    for dtype_name in ["fp32", "fp64"]:
        for seed in [1, 2]:
            run_one(dtype_name, seed)


if __name__ == "__main__":
    main()
