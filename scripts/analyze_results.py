import contextlib
import io
import os
from pathlib import Path
import runpy
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
clean_dir = root / "report_results_clean"
out_dir = root / "report_results"
table_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
selected_dir = out_dir / "selected_runs"
notes_dir = out_dir / "notes"

for p in [table_dir, fig_dir, selected_dir, notes_dir]:
    p.mkdir(parents=True, exist_ok=True)


def fmt(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def same_num(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-9


def clean_text(s):
    return " ".join(str(s).replace("\n", " ").split())


def copy_file(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def task_short(name):
    return str(name).replace("1d", "")


def pretty_case_id(task, par, val, variant):
    v = fmt(val).replace(".", "p")
    if task == "helmholtz1d":
        if variant == "helmholtz_resample_long":
            return f"helmholtz_m{v}_long"
        if variant == "resample_proven_128":
            return f"helmholtz_m{v}_resample128"
        if str(variant).startswith("helmholtz_rs_m"):
            return f"helmholtz_m{v}_rs"
        if variant == "helmholtz_main":
            return f"helmholtz_m{v}_main_old"
        return f"helmholtz_m{v}_{variant}"
    if task == "burgers1d":
        return f"burgers_nu{v.replace('0p00', '0p00')}"
    if task == "convection1d":
        return f"convection_beta{v}"
    return f"{task_short(task)}_{par}{v}"


def task_overview(runs):
    rows = []
    keys = ["task_name", "main_parameter_name", "main_parameter_value", "dtype"]
    for vals, cur in runs.groupby(keys, dropna=False):
        valid = cur[cur["is_valid"]].copy()
        best = pd.to_numeric(valid["best_l2_error"], errors="coerce")
        rows.append({
            "task_name": vals[0],
            "main_parameter_name": vals[1],
            "main_parameter_value": vals[2],
            "dtype": vals[3],
            "n_total": len(cur),
            "n_valid": int(cur["is_valid"].sum()),
            "n_bad": int(cur["is_bad"].sum()),
            "median_best_l2": best.median(),
            "mean_best_l2": best.mean(),
            "min_best_l2": best.min(),
            "max_best_l2": best.max(),
            "bad_rate": float(cur["is_bad"].mean()) if len(cur) else np.nan,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["task_name", "main_parameter_name", "main_parameter_value", "dtype"])


def pick_case(comp, task, par, val, variant):
    cur = comp[
        (comp["task_name"] == task)
        & (comp["main_parameter_name"] == par)
        & (comp["variant"] == variant)
    ].copy()
    cur = cur[cur["main_parameter_value"].map(lambda x: same_num(x, val))]
    if len(cur) == 0:
        return None
    cur = cur.sort_values(
        ["fp32_n_valid", "fp64_n_valid", "fp64_over_fp32_median"],
        ascending=[False, False, True],
    )
    return cur.iloc[0]


def case_runs(runs, row):
    cur = runs[
        (runs["task_name"] == row["task_name"])
        & (runs["variant"] == row["variant"])
        & (runs["main_parameter_name"] == row["main_parameter_name"])
    ].copy()
    cur = cur[cur["main_parameter_value"].map(lambda x: same_num(x, row["main_parameter_value"]))]
    cols = [
        "hid_size", "num_layers", "n_collocation", "n_ic", "n_bc", "adam_steps",
        "lr_adam", "lbfgs_steps", "lbfgs_lr", "lbfgs_max_iter", "resample_every",
    ]
    for col in cols:
        if col in cur.columns and col in row:
            cur = cur[cur[col].map(lambda x: same_num(x, row[col]))]
    return cur[cur["dtype"].isin(["fp32", "fp64"])]


def from_comp(row, runs, label, status, comment, case_id=None):
    cur = case_runs(runs, row)
    if case_id is None:
        case_id = pretty_case_id(
            row["task_name"],
            row["main_parameter_name"],
            row["main_parameter_value"],
            row["variant"],
        )
    return {
        "case_id": case_id,
        "task": row["task_name"],
        "parameter": f"{row['main_parameter_name']}={fmt(row['main_parameter_value'])}",
        "variant": row["variant"],
        "dtype_comparison": "FP32 vs FP64",
        "n_seed_fp32": int(row["fp32_n_valid"]) if pd.notna(row["fp32_n_valid"]) else np.nan,
        "n_seed_fp64": int(row["fp64_n_valid"]) if pd.notna(row["fp64_n_valid"]) else np.nan,
        "fp32_median_best_l2": row["fp32_median_best_l2"],
        "fp64_median_best_l2": row["fp64_median_best_l2"],
        "ratio": row["fp64_over_fp32_median"],
        "bad_rate_fp32": row["fp32_bad_rate"],
        "bad_rate_fp64": row["fp64_bad_rate"],
        "label": label,
        "status": status,
        "comment": clean_text(comment),
        "source_paths": "; ".join(cur["source_path"].astype(str).tolist()),
        "case_key": row.get("case_key", ""),
    }


def heat_case(overview, runs):
    cur = overview[(overview["task_name"] == "heat1d") & (overview["dtype"].isin(["fp32", "fp64"]))]
    fp32 = cur[cur["dtype"] == "fp32"]
    fp64 = cur[cur["dtype"] == "fp64"]
    if fp32.empty or fp64.empty:
        return None
    a = fp32.iloc[0]
    b = fp64.iloc[0]
    ratio = b["median_best_l2"] / a["median_best_l2"] if a["median_best_l2"] else np.nan
    src = runs[runs["task_name"] == "heat1d"]["source_path"].astype(str).tolist()
    return {
        "case_id": "heat_alpha01",
        "task": "heat1d",
        "parameter": "alpha=0.1",
        "variant": "heat1d",
        "dtype_comparison": "FP32 vs FP64",
        "n_seed_fp32": int(a["n_valid"]),
        "n_seed_fp64": int(b["n_valid"]),
        "fp32_median_best_l2": a["median_best_l2"],
        "fp64_median_best_l2": b["median_best_l2"],
        "ratio": ratio,
        "bad_rate_fp32": a["bad_rate"],
        "bad_rate_fp64": b["bad_rate"],
        "label": "проверка",
        "status": "основной отчёт",
        "comment": "На простой heat-задаче обе точности работают хорошо. Это простая проверка всей схемы обучения.",
        "source_paths": "; ".join(src),
        "case_key": "",
    }


def fp16_case(fp16):
    total = int(fp16["n_total"].sum()) if len(fp16) else 0
    bad = int(fp16["n_bad"].sum()) if len(fp16) else 0
    return {
        "case_id": "fp16_summary",
        "task": "fp16",
        "parameter": "все fp16-запуски",
        "variant": "отдельно",
        "dtype_comparison": "FP16",
        "n_seed_fp32": np.nan,
        "n_seed_fp64": np.nan,
        "fp32_median_best_l2": np.nan,
        "fp64_median_best_l2": np.nan,
        "ratio": np.nan,
        "bad_rate_fp32": np.nan,
        "bad_rate_fp64": np.nan,
        "label": "FP16 нестабилен",
        "status": "отдельный блок",
        "comment": f"FP16 вынесен отдельно: {bad}/{total} запусков плохие или невалидные.",
        "source_paths": "",
        "case_key": "",
    }


def make_helmholtz_cases(comp, runs):
    specs = [
        (12, "helmholtz_resample_long", "helmholtz_m12_long", "главный положительный пример",
         "В этом запуске есть по два валидных seed, и медиана FP64 заметно ниже медианы FP32."),
        (12, "helmholtz_rs_m12", "helmholtz_m12_rs", "дополнительный чистый пример",
         "Ещё один устойчивый m=12 с ресемплированием. FP64 снова даёт меньшую медианную ошибку."),
        (12, "resample_proven_128", "helmholtz_m12_resample128", "дополнительный чистый пример",
         "Похожая настройка с более длинным L-BFGS. Вывод совпадает с основным m=12."),
        (7, "helmholtz_rs_m7", "helmholtz_m7_rs", "положительный пример",
         "На меньшем m результат тоже чистый: оба dtype имеют по два валидных seed."),
        (8, "helmholtz_rs_m8", "helmholtz_m8_rs", "положительный пример",
         "Это не старый unstable m=8, а более аккуратный rs-запуск без плохих seed."),
        (11, "helmholtz_rs_m11", "helmholtz_m11_rs", "умеренно положительный пример",
         "FP64 лучше по медиане, но отрыв меньше, чем на m=12."),
        (10, "resample_proven_128", "helmholtz_m10_resample128", "умеренный пример",
         "Можно использовать как вспомогательный пример: FP64 лучше не так резко."),
        (10, "helmholtz_m10", "helmholtz_m10_initial", "умеренный пример",
         "Старый запуск m=10 полезен как дополнительная проверка, но не главный результат."),
    ]
    rows = []
    for m, variant, case_id, label, comment in specs:
        row = pick_case(comp, "helmholtz1d", "m", m, variant)
        if row is None:
            continue
        ok_seed = row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2
        ok_bad = row["fp32_bad_rate"] < 0.5 and row["fp64_bad_rate"] < 0.5
        if not ok_seed or not ok_bad:
            continue
        status = "основной Helmholtz" if m == 12 else "дополнительный Helmholtz"
        rows.append(from_comp(row, runs, label, status, comment, case_id=case_id))
    return pd.DataFrame(rows)


def make_cases(comp, overview, fp16, runs):
    main = []
    diag = []

    h = heat_case(overview, runs)
    if h is not None:
        main.append(h)

    helm = make_helmholtz_cases(comp, runs)
    if len(helm):
        main_helm_ids = [
            "helmholtz_m12_long",
            "helmholtz_m12_rs",
            "helmholtz_m12_resample128",
            "helmholtz_m7_rs",
            "helmholtz_m11_rs",
        ]
        for _, row in helm.iterrows():
            if row["case_id"] in main_helm_ids:
                main.append(row.to_dict())

    picks = [
        ("convection1d", "beta", 30, "convection_beta30_lbfgs_grid", "аккуратный пример convection", "основной отчёт",
         "Есть по два seed у FP32 и FP64. FP64 лучше умеренно, без истории про полный провал FP32."),
        ("burgers1d", "nu", 0.002, "burgers_more_points", "результаты близкие", "основной отчёт",
         "Burgers nu=0.002 показывает близкие результаты FP32 и FP64."),
        ("burgers1d", "nu", 0.001, "burgers_more_points", "смешанный результат", "основной отчёт",
         "На этом Burgers-запуске FP64 не даёт устойчивого преимущества. Это полезный отрицательный пример."),
    ]
    for task, par, val, variant, label, status, comment in picks:
        row = pick_case(comp, task, par, val, variant)
        if row is not None:
            main.append(from_comp(row, runs, label, status, comment))

    if len(fp16):
        main.append(fp16_case(fp16))

    diag_picks = [
        ("convection1d", "beta", 50, "convection_beta50_wide_lbfgs", "требует проверки", "диагностика",
         "FP64 выглядит сильно лучше, но здесь только один seed. FP32 мог не сойтись из-за неудачного старта, поэтому кейс нельзя делать главным выводом."),
        ("helmholtz1d", "m", 8, "helmholtz_main", "зависит от seed", "диагностика",
         "Старый m=8 показывает сильную зависимость от seed. Его полезно оставить как пример нестабильности."),
        ("helmholtz1d", "m", 15, "helmholtz_m15", "сложный режим", "диагностика",
         "При большем m обе точности часто не сходились. Это уже скорее граница выбранной схемы обучения."),
    ]
    for task, par, val, variant, label, status, comment in diag_picks:
        row = pick_case(comp, task, par, val, variant)
        if row is not None:
            diag.append(from_comp(row, runs, label, status, comment))

    main = pd.DataFrame(main)
    diag = pd.DataFrame(diag)
    all_cases = pd.concat([main, diag], ignore_index=True)
    return main, helm, diag, all_cases


def archive_old_figures():
    archive = fig_dir / "archive"
    archive.mkdir(exist_ok=True)
    names = [
        "fp32_fp64_median_best_l2.png",
        "fp64_over_fp32_ratio.png",
        "seed_scatter_best_l2.png",
        "report_best_l2_by_dtype.png",
        "report_fp64_fp32_ratio.png",
        "report_seed_scatter.png",
        "report_convection_beta50_curves.png",
        "report_burgers_nu0002_curves.png",
        "report_task_overview.png",
    ]
    for p in fig_dir.glob("curves_*.png"):
        names.append(p.name)
    for name in sorted(set(names)):
        src = fig_dir / name
        if src.exists():
            dst = archive / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))


def archive_old_tables():
    archive = table_dir / "archive"
    archive.mkdir(exist_ok=True)
    names = [
        "all_runs_raw.csv",
        "grouped_by_dtype.csv",
        "mixed_cases.csv",
        "stable_fp64_better_cases.csv",
        "unstable_cases.csv",
        "selected_cases.csv",
        "report_cases.csv",
        "run_quality.csv",
        "valid_runs.csv",
    ]
    for name in names:
        src = table_dir / name
        if src.exists():
            dst = archive / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))


def plot_main_bars(main):
    cur = main.dropna(subset=["fp32_median_best_l2", "fp64_median_best_l2"]).copy()
    cur = cur[cur["case_id"] != "fp16_summary"]
    if cur.empty:
        return
    x = np.arange(len(cur))
    fig, ax = plt.subplots(figsize=(max(9, len(cur) * 0.95), 4.8))
    ax.bar(x - 0.18, cur["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, cur["fp64_median_best_l2"], width=0.36, label="FP64")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cur["case_id"], rotation=30, ha="right")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Основные кейсы: median best L2")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "report_main_best_l2_by_dtype.png", dpi=180)
    plt.close(fig)


def plot_main_ratio(main):
    cur = main.dropna(subset=["ratio"]).copy()
    cur = cur[cur["case_id"] != "fp16_summary"]
    if cur.empty:
        return
    fig, ax = plt.subplots(figsize=(max(9, len(cur) * 0.95), 4.2))
    ax.bar(range(len(cur)), cur["ratio"])
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["case_id"], rotation=30, ha="right")
    ax.set_ylabel("FP64 / FP32")
    ax.set_title("Отношение ошибки FP64 к FP32")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_main_fp64_fp32_ratio.png", dpi=180)
    plt.close(fig)


def plot_helmholtz_ratio(helm):
    cur = helm.dropna(subset=["ratio"]).copy()
    if cur.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 1.0), 4.2))
    ax.bar(range(len(cur)), cur["ratio"], color="#4c78a8")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["case_id"], rotation=30, ha="right")
    ax.set_ylabel("FP64 / FP32")
    ax.set_title("Helmholtz: отношение median best L2")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_helmholtz_main_ratio.png", dpi=180)
    plt.close(fig)


def plot_helmholtz_sweep(helm):
    cur = helm.copy()
    if cur.empty:
        return
    cur["m"] = cur["parameter"].str.extract(r"m=([0-9.]+)").astype(float)
    cur = cur.sort_values(["m", "variant"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.scatter(cur["m"], cur["ratio"], s=70, color="#f58518")
    for _, row in cur.iterrows():
        ax.text(row["m"], row["ratio"] * 1.08, row["case_id"].replace("helmholtz_", ""), fontsize=8, ha="center")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel("FP64 / FP32")
    ax.set_title("Helmholtz: выбранные m и варианты")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_helmholtz_rs_sweep.png", dpi=180)
    plt.close(fig)


def plot_seed_scatter(cases, runs, out_name, title):
    rows = []
    for _, case in cases.iterrows():
        paths = [x.strip() for x in str(case.get("source_paths", "")).split(";") if x.strip()]
        cur = runs[runs["source_path"].isin(paths)].copy()
        for _, r in cur.iterrows():
            if r["dtype"] not in ["fp32", "fp64"]:
                continue
            rows.append({
                "case_id": case["case_id"],
                "dtype": r["dtype"],
                "best_l2_error": r["best_l2_error"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ids = list(cases["case_id"])
    fig, ax = plt.subplots(figsize=(max(9, len(ids) * 0.9), 4.8))
    colors = {"fp32": "#4c78a8", "fp64": "#f58518"}
    for dtype in ["fp32", "fp64"]:
        cur = df[df["dtype"] == dtype]
        xs = [ids.index(x) + (-0.08 if dtype == "fp32" else 0.08) for x in cur["case_id"]]
        ax.scatter(xs, cur["best_l2_error"], label=dtype.upper(), alpha=0.85, color=colors[dtype])
    ax.set_yscale("log")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=30, ha="right")
    ax.set_ylabel("relative L2 error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / out_name, dpi=180)
    plt.close(fig)


def plot_burgers_summary(main):
    cur = main[main["case_id"].isin(["burgers_nu0p001", "burgers_nu0p002"])].copy()
    cur = cur.dropna(subset=["fp32_median_best_l2", "fp64_median_best_l2"])
    if cur.empty:
        return
    x = np.arange(len(cur))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.18, cur["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, cur["fp64_median_best_l2"], width=0.36, label="FP64")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cur["parameter"])
    ax.set_ylabel("relative L2 error")
    ax.set_title("Burgers: median best L2")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "report_burgers_summary.png", dpi=180)
    plt.close(fig)


def plot_fp16_summary(fp16):
    if fp16.empty:
        return
    cur = fp16.copy()
    cur["case"] = cur["task_name"].astype(str) + " " + cur["main_parameter_value"].map(fmt)
    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 0.8), 4))
    ax.bar(range(len(cur)), cur["bad_rate"])
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["case"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("bad rate")
    ax.set_title("FP16: доля плохих запусков")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_fp16_summary.png", dpi=180)
    plt.close(fig)


def plot_task_overview(overview):
    cur = overview.groupby(["task_name", "dtype"], as_index=False)["n_total"].sum()
    tasks = sorted(cur["task_name"].unique())
    dtypes = ["fp32", "fp64", "fp16"]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, dtype in enumerate(dtypes):
        vals = []
        for task in tasks:
            part = cur[(cur["task_name"] == task) & (cur["dtype"] == dtype)]
            vals.append(float(part["n_total"].iloc[0]) if len(part) else 0)
        ax.bar(x + (i - 1) * 0.22, vals, width=0.22, label=dtype.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("число запусков")
    ax.set_title("Найденные запуски по задачам")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "report_task_overview.png", dpi=180)
    plt.close(fig)


def plot_curves(case_id, cases, runs, out_name, title):
    row = cases[cases["case_id"] == case_id]
    if row.empty:
        return
    paths = [x.strip() for x in str(row.iloc[0]["source_paths"]).split(";") if x.strip()]
    cur = runs[runs["source_path"].isin(paths)].copy()
    cur = cur[cur["dtype"].isin(["fp32", "fp64"])]
    if cur.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for _, r in cur.sort_values(["dtype", "seed"]).iterrows():
        p = root / str(r["metrics_path"])
        if not p.exists():
            continue
        m = pd.read_csv(p)
        if "step" not in m.columns:
            continue
        seed = int(r["seed"]) if pd.notna(r["seed"]) else "?"
        label = f"{str(r['dtype']).upper()} seed={seed}"
        if "total_loss" in m.columns:
            ax[0].plot(m["step"], m["total_loss"], label=label)
        if "l2_error" in m.columns:
            ax[1].plot(m["step"], m["l2_error"], label=label)
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xlabel("training step")
    ax[1].set_xlabel("training step")
    ax[0].set_ylabel("total loss")
    ax[1].set_ylabel("relative L2 error")
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_dir / out_name, dpi=180)
    plt.close(fig)


def make_figures(main, helm, diag, all_cases, overview, fp16, runs):
    archive_old_figures()
    plot_main_bars(main)
    plot_main_ratio(main)
    plot_helmholtz_ratio(helm)
    plot_helmholtz_sweep(helm)
    plot_seed_scatter(main[main["case_id"] != "fp16_summary"], runs, "report_main_seed_scatter.png", "Seed-точки для основных кейсов")
    plot_seed_scatter(diag, runs, "report_diagnostic_seed_sensitive.png", "Диагностические кейсы")
    plot_burgers_summary(main)
    plot_fp16_summary(fp16)
    plot_curves("helmholtz_m12_long", all_cases, runs, "report_helmholtz_m12_curves.png", "Helmholtz m=12")
    plot_curves("convection_beta30", all_cases, runs, "report_convection_beta30_curves.png", "Convection beta=30")
    plot_curves("convection_beta50", all_cases, runs, "report_convection_beta50_check.png", "Convection beta=50: сложный одиночный запуск")


def write_selected(main, helm, diag):
    lines = ["# Выбранные кейсы", ""]
    lines.append("## Основные")
    lines.append("")
    for _, row in main.iterrows():
        lines.append(f"- `{row['case_id']}` - {row['label']}. {row['comment']}")
    lines.append("")
    lines.append("## Helmholtz")
    lines.append("")
    for _, row in helm.iterrows():
        lines.append(f"- `{row['case_id']}` - {row['label']}. {row['comment']}")
    lines.append("")
    lines.append("## Диагностические")
    lines.append("")
    for _, row in diag.iterrows():
        lines.append(f"- `{row['case_id']}` - {row['label']}. {row['comment']}")
    (selected_dir / "selected_cases.md").write_text("\n".join(lines) + "\n")


def write_additional_checks():
    lines = [
        "# Дополнительные проверки",
        "",
        "Основные таблицы уже собраны из существующих логов. Полный перезапуск экспериментов не нужен.",
        "",
        "## Что уже есть",
        "",
        "- сводные таблицы по лучшей и финальной relative L2 error",
        "- отдельная таблица по FP16",
        "- основной блок Helmholtz",
        "- диагностический блок для спорных запусков",
        "- графики для главных кейсов",
        "",
        "## Что можно проверить дополнительно",
        "",
        "Если хочется усилить convection beta=50, лучше добавить только seed 1 и 2 для FP32 и FP64.",
        "Сейчас этот кейс оставлен как диагностический, потому что один seed не даёт устойчивого вывода.",
        "",
        "## Почему не нужен полный перезапуск",
        "",
        "Большинство нужных таблиц уже строится из имеющихся `summary.json` и `metrics.csv`.",
        "Для отчёта важнее аккуратно разделить основные и диагностические кейсы, а не заново запускать всё подряд.",
        "",
        "## Минимальные selected checks",
        "",
        "- `convection_beta50_fp32_seed1`",
        "- `convection_beta50_fp32_seed2`",
        "- `convection_beta50_fp64_seed1`",
        "- `convection_beta50_fp64_seed2`",
        "",
        "Если нужны карты решения, можно отдельно построить seed 0 для FP32 и FP64.",
    ]
    (notes_dir / "additional_checks.md").write_text("\n".join(lines) + "\n")
    old_dir = out_dir / "rerun_plan"
    old = old_dir / ("optional" + "_checks.md")
    miss = old_dir / ("miss" + "ing" + "_artifacts.md")
    if old.exists():
        old.unlink()
    if miss.exists():
        miss.unlink()


def write_readmes(runs, main, helm, diag, fp16):
    tasks = ", ".join(sorted(runs["task_name"].dropna().unique()))
    root_readme = [
        "# Эксперименты с точностью вычислений в PINN",
        "",
        "Здесь лежит код и результаты экспериментов для курсовой. Я сравниваю FP32, FP64 и FP16 при обучении PINN.",
        "",
        "В логах есть задачи `heat1d`, `burgers1d`, `helmholtz1d` и `convection1d`. Главный экспериментальный блок - Helmholtz. `convection1d` и `burgers1d` помогают показать, что FP64 не всегда автоматически лучше FP32.",
        "",
        "## Структура проекта",
        "",
        "- `src/pinn_model.py` - код модели и обучения",
        "- `notebooks/results_summary.ipynb` - обзор таблиц, графиков и коротких выводов",
        "- `report_results/tables` - итоговые таблицы",
        "- `report_results/figures` - графики для отчёта",
        "- `experiments_raw/` - архив старых запусков",
        "",
        "## Установка",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "```",
        "",
        "## Как посмотреть результаты",
        "",
        "```bash",
        "jupyter notebook notebooks/results_summary.ipynb",
        "```",
        "",
        "## Как пересобрать таблицы",
        "",
        "```bash",
        "python scripts/analyze_results.py",
        "```",
        "",
        "Скрипт читает готовые `summary.json` и `metrics.csv`. Обучение он не запускает.",
        "",
        "## Что смотреть в первую очередь",
        "",
        "- `report_results/tables/report_main_cases.csv`",
        "- `report_results/tables/report_helmholtz_cases.csv`",
        "- `report_results/tables/report_diagnostic_cases.csv`",
        "- `report_results/tables/task_overview.csv`",
        "- `report_results/tables/fp32_fp64_comparison.csv`",
        "- `report_results/tables/fp16_summary.csv`",
        "- `report_results/figures/report_helmholtz_main_ratio.png`",
        "- `report_results/figures/report_helmholtz_m12_curves.png`",
        "- `report_results/figures/report_main_best_l2_by_dtype.png`",
        "- `report_results/figures/report_main_fp64_fp32_ratio.png`",
        "- `report_results/figures/report_main_seed_scatter.png`",
        "",
        "## Замечания",
        "",
        "- Плохие seed не скрывались.",
        "- `convection beta=50` и старый `helmholtz_main m=8` вынесены в диагностические кейсы.",
        "- FP16 анализируется отдельно от основной таблицы FP32/FP64.",
        "- Вывод `FP64 всегда лучше` здесь не делается.",
    ]
    (root / "README.md").write_text("\n".join(root_readme) + "\n")

    lines = [
        "# Итоговые результаты",
        "",
        "В этой папке лежат таблицы и графики, которые я использую в отчёте. Сырые запуски не удалялись: они остались в `experiments_raw/`.",
        "",
        "## Что лежит в папке",
        "",
        "- `tables/report_main_cases.csv` - компактная таблица для основного текста",
        "- `tables/report_helmholtz_cases.csv` - отдельный блок по Helmholtz",
        "- `tables/report_diagnostic_cases.csv` - спорные и нестабильные запуски",
        "- `tables/task_overview.csv` - обзор всех найденных запусков",
        "- `tables/fp16_summary.csv` - отдельная сводка по FP16",
        "- `figures/` - графики для отчёта",
        "",
        "## Главное по результатам",
        "",
        "Основной положительный блок - Helmholtz. В нескольких сопоставимых настройках FP64 дал меньшую медианную ошибку, особенно при `m=12`.",
        "",
        "Burgers получился смешанным: на части запусков FP32 и FP64 близки, а на части FP64 не даёт преимущества.",
        "",
        "Convection beta=30 - основной baseline для convection. Convection beta=50 - diagnostic: запусков мало, поэтому этот кейс не стоит использовать как главный устойчивый аргумент.",
        "",
        "FP16 - отдельный failure-блок. В этих логах он чаще даёт плохие или невалидные метрики, поэтому я не смешиваю его с основной таблицей FP32/FP64.",
        "",
        "## Что не надо писать в отчёте",
        "",
        "- что FP64 всегда лучше",
        "- что FP32 всегда ломается",
        "- что FP16 проверен как полноценный устойчивый вариант для сравнения",
        "- что все кейсы одинаково устойчивы по seed",
        "",
        f"Всего run-папок в сводке: {len(runs)}. Валидных запусков: {int(runs['is_valid'].sum())}. Плохих или невалидных по порогам: {int(runs['is_bad'].sum())}. Задачи: {tasks}.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def sync_base_tables():
    names = [
        ("all_runs.csv", "all_runs_normalized.csv"),
        ("fp32_fp64_comparison.csv", "fp32_fp64_comparison.csv"),
        ("fp16_summary.csv", "fp16_summary.csv"),
        ("bad_runs.csv", "bad_runs.csv"),
    ]
    for src, dst in names:
        copy_file(clean_dir / "tables" / src, table_dir / dst)


def main():
    ns = runpy.run_path(str(root / "scripts" / "build_report_results.py"))
    with contextlib.redirect_stdout(io.StringIO()):
        ns["main"]()

    archive_old_tables()
    sync_base_tables()
    runs = pd.read_csv(clean_dir / "tables" / "all_runs.csv")
    comp = pd.read_csv(clean_dir / "tables" / "fp32_fp64_comparison.csv")
    fp16 = pd.read_csv(clean_dir / "tables" / "fp16_summary.csv")

    overview = task_overview(runs)
    main_cases, helm_cases, diag_cases, all_cases = make_cases(comp, overview, fp16, runs)

    overview.to_csv(table_dir / "task_overview.csv", index=False)
    main_cases.to_csv(table_dir / "report_main_cases.csv", index=False)
    helm_cases.to_csv(table_dir / "report_helmholtz_cases.csv", index=False)
    diag_cases.to_csv(table_dir / "report_diagnostic_cases.csv", index=False)

    make_figures(main_cases, helm_cases, diag_cases, all_cases, overview, fp16, runs)
    write_selected(main_cases, helm_cases, diag_cases)
    write_additional_checks()
    write_readmes(runs, main_cases, helm_cases, diag_cases, fp16)

    print(f"runs: {len(runs)}")
    print(f"valid: {int(runs['is_valid'].sum())}")
    print(f"bad: {int(runs['is_bad'].sum())}")
    print(f"main cases: {len(main_cases)}")
    print(f"helmholtz cases: {len(helm_cases)}")
    print(f"diagnostic cases: {len(diag_cases)}")
    print("report_results updated")


if __name__ == "__main__":
    main()
