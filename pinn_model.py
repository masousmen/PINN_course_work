import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import time


class PINN(nn.Module):
    def __init__(self, input_dim=2, num_layers=2, hid_size=32):
        super().__init__()
        layers = [nn.Linear(input_dim, hid_size), nn.Tanh()]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hid_size, hid_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hid_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, *xs):
        if len(xs) == 1:
            inp = xs[0]
        else:
            inp = torch.cat(xs, dim=1)
        return self.model(inp)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(type_name):
    if type_name == 'fp32' or type_name == 'float32':
        return torch.float32
    if type_name == 'fp64' or type_name == 'float64':
        return torch.float64
    if type_name == 'bf16':
        return torch.bfloat16
    raise ValueError("Unsupported dtype")


def heat_1d_solution(x, t, alpha):
    return torch.exp(-torch.pi * torch.pi * alpha * t) * torch.sin(torch.pi * x)

def burgers_1d_initial(x):
    return -torch.sin(torch.pi * x)


burgers_cache = {}


def burgers_1d_reference(n, nu):
    key = (n, round(float(nu), 12))
    if key in burgers_cache:
        return burgers_cache[key]

    x = np.linspace(-1, 1, n)
    ts = np.linspace(0, 1, n)
    dx = x[1] - x[0]

    u = -np.sin(np.pi * x)
    u[0] = 0
    u[-1] = 0

    vals = np.zeros((n, n))
    vals[:, 0] = u

    dt = min(0.0005, 0.2 * dx * dx / max(float(nu), 1e-8))
    steps = int(np.ceil(1.0 / dt))
    dt = 1.0 / steps
    k = 1

    for step in range(1, steps + 1):
        old = u.copy()
        ux_left = (old[1:-1] - old[:-2]) / dx
        ux_right = (old[2:] - old[1:-1]) / dx
        ux = np.where(old[1:-1] >= 0, ux_left, ux_right)
        uxx = (old[2:] - 2 * old[1:-1] + old[:-2]) / (dx * dx)
        u[1:-1] = old[1:-1] - dt * old[1:-1] * ux + nu * dt * uxx
        u[0] = 0
        u[-1] = 0

        cur_t = step * dt
        while k < n and cur_t >= ts[k]:
            vals[:, k] = u
            k += 1

    burgers_cache[key] = (x, ts, vals)
    return burgers_cache[key]


def make_points(n_collocation, n_ic, n_bc, task):
    if task == 'heat1d':
        x_pde = torch.rand(n_collocation, 1)
        t_pde = torch.rand(n_collocation, 1)
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        x_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)

        t_bc = torch.rand(n_bc, 1)
        x0 = torch.zeros(n_bc, 1)
        x1 = torch.ones(n_bc, 1)

        points = {
            "x_pde": x_pde,
            "t_pde": t_pde,
            "x_ic": x_ic,
            "t_ic": t_ic,
            "t_bc": t_bc,
            "x0": x0,
            "x1": x1,
        }
        return points

    if task == 'burgers1d':
        x_pde = 2 * torch.rand(n_collocation, 1) - 1
        t_pde = torch.rand(n_collocation, 1)
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        x_ic = 2 * torch.rand(n_ic, 1) - 1
        t_ic = torch.zeros(n_ic, 1)

        t_bc = torch.rand(n_bc, 1)
        x0 = -torch.ones(n_bc, 1)
        x1 = torch.ones(n_bc, 1)

        points = {
            "x_pde": x_pde,
            "t_pde": t_pde,
            "x_ic": x_ic,
            "t_ic": t_ic,
            "t_bc": t_bc,
            "x0": x0,
            "x1": x1,
        }
        return points

    raise ValueError("Unsupported task")


def pinn_loss(model, points, alpha, task):
    if task == 'heat1d':
        x_pde = points['x_pde']
        t_pde = points['t_pde']

        u = model(x_pde, t_pde)
        ones = torch.ones_like(u)

        u_t = torch.autograd.grad(u, t_pde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x_pde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        res = u_t - alpha * u_xx
        pde_loss = torch.mean(res ** 2)

        u_ic = model(points["x_ic"], points["t_ic"])
        u_ic_true = heat_1d_solution(points["x_ic"], points["t_ic"], alpha)
        ic_loss = torch.mean((u_ic - u_ic_true) ** 2)

        u0 = model(points["x0"], points["t_bc"])
        u1 = model(points["x1"], points["t_bc"])
        bc_loss = torch.mean(u0 ** 2) + torch.mean(u1 ** 2)

        loss = pde_loss + ic_loss + bc_loss
        return loss, pde_loss, ic_loss, bc_loss

    if task == 'burgers1d':
        x_pde = points['x_pde']
        t_pde = points['t_pde']
        u = model(x_pde, t_pde)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t_pde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x_pde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        res = u_t + u * u_x - alpha * u_xx
        pde_loss = torch.mean(res ** 2)

        u_ic = model(points["x_ic"], points["t_ic"])
        u_ic_true = burgers_1d_initial(points["x_ic"])
        ic_loss = torch.mean((u_ic - u_ic_true) ** 2)

        u0 = model(points["x0"], points["t_bc"])
        u1 = model(points["x1"], points["t_bc"])
        bc_loss = torch.mean(u0 ** 2) + torch.mean(u1 ** 2)

        loss = pde_loss + ic_loss + bc_loss
        return loss, pde_loss, ic_loss, bc_loss

    raise ValueError("Unsupported task")


def l2_error(model, alpha, task):
    n = 100
    if task == 'heat1d':
        x = torch.linspace(0, 1, n)
    elif task == 'burgers1d':
        x = torch.linspace(-1, 1, n)
    else:
        raise ValueError("Unsupported task")

    t = torch.linspace(0, 1, n)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xv = xx.reshape(-1, 1)
    tv = tt.reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        pred = model(xv, tv)
        if task == 'heat1d':
            true = heat_1d_solution(xv, tv, alpha)
        elif task == 'burgers1d':
            ref_x, ref_t, vals = burgers_1d_reference(n, alpha)
            true = torch.tensor(vals.reshape(-1, 1), dtype=pred.dtype, device=pred.device)
        else:
            raise ValueError("Unsupported task")
        err = torch.sqrt(torch.mean((pred - true) ** 2)) / torch.sqrt(torch.mean(true ** 2))
    model.train()
    return float(err.detach().cpu())


def get_input_dim(task):
    if task == 'heat1d':
        return 2
    if task == 'burgers1d':
        return 2
    raise ValueError("Unsupported task")


def run_experiment(config):
    dtype_name = config.get('dtype', config.get('dtype_name', 'fp32'))
    dtype = get_dtype(dtype_name)
    torch.set_default_dtype(dtype)

    if 'device' in config:
        device = torch.device(config['device'])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    set_seed(config.get('seed', 0))

    log_dir = Path(config.get('log_dir', config.get('out_dir', 'runs/tmp')))
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg_save = dict(config)
    cfg_save['device'] = str(device)
    with open(log_dir / "config.json", 'w') as file:
        json.dump(cfg_save, file, indent=2)

    task_name = config.get('task_name', 'heat1d')
    input_dim = config.get('input_dim', get_input_dim(task_name))
    hid_size = config.get('hid_size', config.get('hidden_dim', 64))
    num_layers = config.get('num_layers', 4)
    if task_name == 'burgers1d':
        alpha = config.get('nu', config.get('alpha', 0.01 / np.pi))
    else:
        alpha = config.get('alpha', 0.1)

    points = make_points(config['n_collocation'], config['n_ic'], config['n_bc'], task_name)
    model = PINN(input_dim=input_dim, hid_size=hid_size, num_layers=num_layers).to(device)

    history = []
    start_time = time.time()

    def save_error(step, loss, pde_loss, ic_loss, bc_loss, opt):
        l2 = l2_error(model, alpha, task_name)
        history.append({
            "step": step,
            "opt": opt,
            "total_loss": float(loss.detach().cpu()),
            "pde_loss": float(pde_loss.detach().cpu()),
            "ic_loss": float(ic_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "l2_error": l2,
            "time_sec": time.time() - start_time,
        })

    use_adam = config.get('use_adam', True)
    adam_steps = config.get('adam_steps', 3000)
    lr_adam = config.get('lr_adam', config.get('lr', 1e-3))

    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
        for step in range(1, adam_steps + 1):
            optimizer.zero_grad()
            loss, pde_loss, ic_loss, bc_loss = pinn_loss(model, points, alpha, task_name)
            loss.backward()
            optimizer.step()

            if step == 1 or step % 100 == 0 or step == adam_steps:
                loss, pde_loss, ic_loss, bc_loss = pinn_loss(model, points, alpha, task_name)
                save_error(step, loss, pde_loss, ic_loss, bc_loss, 'adam')

    use_lbfgs = config.get('use_lbfgs', False)
    lbfgs_steps = config.get('lbfgs_steps', 300)

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, history_size=50)

        for i in range(1, lbfgs_steps + 1):
            def closure():
                optimizer.zero_grad()
                loss, pde_loss, ic_loss, bc_loss = pinn_loss(model, points, alpha, task_name)
                loss.backward()
                return loss

            optimizer.step(closure)

            if i == 1 or i % 100 == 0 or i == lbfgs_steps:
                loss, pde_loss, ic_loss, bc_loss = pinn_loss(model, points, alpha, task_name)
                save_error(adam_steps + i, loss, pde_loss, ic_loss, bc_loss, 'lbfgs')

    if len(history) == 0:
        loss, pde_loss, ic_loss, bc_loss = pinn_loss(model, points, alpha, task_name)
        save_error(0, loss, pde_loss, ic_loss, bc_loss, 'none')

    hist = pd.DataFrame(history)
    hist.to_csv(log_dir / "metrics.csv", index=False)

    last = history[-1]
    summary = {
        "task_name": task_name,
        "dtype": dtype_name,
        "seed": config.get('seed', 0),
        "alpha": alpha,
        "hid_size": hid_size,
        "num_layers": num_layers,
        "n_collocation": config['n_collocation'],
        "n_ic": config['n_ic'],
        "n_bc": config['n_bc'],
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "final_loss": last["total_loss"],
        "final_l2_error": last["l2_error"],
        "time_sec": last["time_sec"],
        "log_dir": str(log_dir),
    }
    if is_burgers(task_name):
        summary["nu"] = alpha

    with open(log_dir / "summary.json", 'w') as file:
        json.dump(summary, file, indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["step"], hist["total_loss"])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("total_loss")
    ax[0].grid(True)
    ax[1].plot(hist["step"], hist["l2_error"])
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("l2_error")
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(log_dir / "curves.png", dpi=150)
    plt.close(fig)

    if task_name == 'heat1d':
        x = torch.linspace(0, 1, 200).reshape(-1, 1)
    elif is_burgers(task_name):
        x = torch.linspace(-1, 1, 200).reshape(-1, 1)
    else:
        raise ValueError("Unsupported task")

    t = torch.ones_like(x)
    model.eval()
    with torch.no_grad():
        pred = model(x, t).detach().cpu().numpy().reshape(-1)
        if task_name == 'heat1d':
            true = heat_1d_solution(x, t, alpha).detach().cpu().numpy().reshape(-1)
            true_name = "exact"
        else:
            ref_x, ref_t, vals = burgers_1d_reference(200, alpha)
            true = vals[:, -1]
            true_name = "reference"
        x_np = x.detach().cpu().numpy().reshape(-1)
    model.train()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_np, true, label=true_name)
    ax.plot(x_np, pred, label="pinn")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, 1)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(log_dir / "solution_t1.png", dpi=150)
    plt.close(fig)
    return hist, summary
