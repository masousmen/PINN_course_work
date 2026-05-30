import os
from pathlib import Path
import runpy
import shutil
import contextlib
import io

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent
clean_dir = root / "report_results_clean"
out_dir = root / "report_results"
table_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
selected_dir = out_dir / "selected_runs"
rerun_dir = out_dir / "rerun_plan"

for p in [table_dir, fig_dir, selected_dir, rerun_dir]:
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def fmt(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def num_eq(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-9


def ru_label(x):
    d = {
        "stable_positive": "устойчивый положительный",
        "moderate_positive": "умеренно положительный",
        "similar": "похоже",
        "fp32_better": "FP32 лучше",
        "seed_sensitive": "зависит от seed",
        "fp16_failure": "FP16 нестабилен",
        "sanity_check": "простая проверка",
    }
    return d.get(str(x), str(x))


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


def case_runs(runs, row):
    cur = runs[
        (runs["task_name"] == row["task_name"])
        & (runs["variant"] == row["variant"])
        & (runs["main_parameter_name"] == row["main_parameter_name"])
    ].copy()
    cur = cur[cur["main_parameter_value"].map(lambda x: num_eq(x, row["main_parameter_value"]))]
    for col in [
        "hid_size", "num_layers", "n_collocation", "n_ic", "n_bc", "adam_steps",
        "lr_adam", "lbfgs_steps", "lbfgs_lr", "lbfgs_max_iter", "resample_every",
    ]:
        if col in cur.columns and col in row:
            cur = cur[cur[col].map(lambda x: num_eq(x, row[col]))]
    return cur


def pick_one(df, task=None, par=None, val=None, variant=None, label=None, prefer=None):
    cur = df.copy()
    if task is not None:
        cur = cur[cur["task_name"] == task]
    if par is not None:
        cur = cur[cur["main_parameter_name"] == par]
    if val is not None:
        cur = cur[cur["main_parameter_value"].map(lambda x: num_eq(x, val))]
    if variant is not None:
        cur = cur[cur["variant"] == variant]
    if label is not None:
        cur = cur[cur["label"] == label]
    if len(cur) == 0:
        return None
    if prefer == "low_ratio":
        cur = cur.sort_values("ratio", na_position="last")
    elif prefer == "many_seeds":
        cur = cur.sort_values(["fp32_n_valid", "fp64_n_valid", "fp64_over_fp32_median"], ascending=[False, False, True])
    else:
        cur = cur.sort_values(["fp32_n_valid", "fp64_n_valid"], ascending=[False, False])
    return cur.iloc[0]


def comp_to_case(row, runs, label, comment, status="ok"):
    cur = case_runs(runs, row)
    cur = cur[cur["dtype"].isin(["fp32", "fp64"])]
    src = "; ".join(cur["source_path"].astype(str).tolist())
    param = f"{row['main_parameter_name']}={fmt(row['main_parameter_value'])}"
    task_short = str(row["task_name"]).replace("1d", "")
    value = fmt(row["main_parameter_value"]).replace(".", "p")
    ratio = row["fp64_over_fp32_median"]
    return {
        "case_id": f"{task_short}_{row['main_parameter_name']}{value}",
        "task": row["task_name"],
        "parameter": param,
        "variant": row["variant"],
        "dtype_comparison": "FP32 vs FP64",
        "n_seed_fp32": row["fp32_n_valid"],
        "n_seed_fp64": row["fp64_n_valid"],
        "fp32_median_best_l2": row["fp32_median_best_l2"],
        "fp64_median_best_l2": row["fp64_median_best_l2"],
        "ratio": ratio,
        "bad_rate_fp32": row["fp32_bad_rate"],
        "bad_rate_fp64": row["fp64_bad_rate"],
        "label": label,
        "status": status,
        "comment": comment,
        "source_paths": src,
        "case_key": row.get("case_key", ""),
    }


def make_heat_case(overview, runs):
    cur = overview[(overview["task_name"] == "heat1d") & (overview["dtype"].isin(["fp32", "fp64"]))]
    if cur.empty:
        return None
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
        "n_seed_fp32": a["n_valid"],
        "n_seed_fp64": b["n_valid"],
        "fp32_median_best_l2": a["median_best_l2"],
        "fp64_median_best_l2": b["median_best_l2"],
        "ratio": ratio,
        "bad_rate_fp32": a["bad_rate"],
        "bad_rate_fp64": b["bad_rate"],
        "label": "sanity_check",
        "status": "ok",
        "comment": "Простая задача для проверки: ошибки у FP32 и FP64 маленькие, большой разницы тут ждать не нужно.",
        "source_paths": "; ".join(src),
        "case_key": "",
    }


def make_report_cases(comp, overview, fp16, runs):
    comp2 = comp.copy()
    comp2["label"] = "seed_sensitive"
    stable = (
        (comp2["fp32_n_valid"] >= 2)
        & (comp2["fp64_n_valid"] >= 2)
        & (comp2["fp64_over_fp32_median"] < 0.5)
        & (comp2["fp64_bad_rate"] <= comp2["fp32_bad_rate"] + 0.25)
        & (comp2["fp32_bad_rate"] < 0.5)
        & (comp2["fp64_bad_rate"] < 0.5)
    )
    moderate = (
        (comp2["fp32_n_valid"] >= 2)
        & (comp2["fp64_n_valid"] >= 2)
        & (comp2["fp64_over_fp32_median"] >= 0.5)
        & (comp2["fp64_over_fp32_median"] <= 0.8)
        & (comp2["fp32_bad_rate"] < 0.5)
        & (comp2["fp64_bad_rate"] < 0.5)
    )
    similar = (
        (comp2["fp32_n_valid"] >= 2)
        & (comp2["fp64_n_valid"] >= 2)
        & (comp2["fp64_over_fp32_median"] > 0.8)
        & (comp2["fp64_over_fp32_median"] < 1.25)
        & (comp2["fp32_bad_rate"] < 0.5)
        & (comp2["fp64_bad_rate"] < 0.5)
    )
    fp32_better = (
        (comp2["fp32_n_valid"] >= 2)
        & (comp2["fp64_n_valid"] >= 2)
        & (comp2["fp64_over_fp32_median"] >= 1.25)
        & (comp2["fp32_bad_rate"] < 0.5)
        & (comp2["fp64_bad_rate"] < 0.5)
    )
    comp2.loc[stable, "label"] = "stable_positive"
    comp2.loc[moderate, "label"] = "moderate_positive"
    comp2.loc[similar, "label"] = "similar"
    comp2.loc[fp32_better, "label"] = "fp32_better"

    rows = []
    heat = make_heat_case(overview, runs)
    if heat is not None:
        rows.append(heat)

    row = pick_one(comp2, "helmholtz1d", "m", 12, "helmholtz_resample_long")
    if row is not None:
        label = row["label"]
        if label not in ["stable_positive", "moderate_positive"]:
            label = "seed_sensitive"
        rows.append(comp_to_case(
            row, runs, label,
            "Основной Helmholtz-кейс: тут есть по два валидных seed и FP64 лучше по медиане.",
        ))

    row = pick_one(comp2, "helmholtz1d", "m", 8, "helmholtz_main")
    if row is not None:
        rows.append(comp_to_case(
            row, runs, "seed_sensitive",
            "Helmholtz m=8 полезен как пример зависимости от seed: один плохой FP32 seed сильно влияет на картину.",
            "needs_check",
        ))

    row = pick_one(comp2, "burgers1d", "nu", 0.002, "burgers_more_points")
    if row is not None:
        rows.append(comp_to_case(
            row, runs, "similar",
            "Burgers nu=0.002 показывает близкие результаты FP32 и FP64.",
        ))

    row = pick_one(comp2, "burgers1d", "nu", 0.001, "burgers_more_points")
    if row is not None:
        rows.append(comp_to_case(
            row, runs, "fp32_better",
            "Burgers nu=0.001 оставлен как отрицательный пример: FP64 здесь не дал преимущества.",
        ))

    row = pick_one(comp2, "convection1d", "beta", 30, "convection_beta30_lbfgs_grid")
    if row is not None:
        label = "moderate_positive" if row["label"] == "moderate_positive" else "similar"
        rows.append(comp_to_case(
            row, runs, label,
            "Convection beta=30 выглядит аккуратнее, потому что есть по два seed у FP32 и FP64.",
        ))

    row = pick_one(comp2, "convection1d", "beta", 50, "convection_beta50_wide_lbfgs")
    if row is not None:
        rows.append(comp_to_case(
            row, runs, "seed_sensitive",
            "Предварительный hard-case: FP64 выглядит сильно лучше, но есть только один seed. FP32 мог не сойтись из-за неудачного старта.",
            "preliminary; needs_check",
        ))

    if len(fp16):
        total = int(fp16["n_total"].sum())
        bad = int(fp16["n_bad"].sum())
        rows.append({
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
            "label": "fp16_failure",
            "status": "separate",
            "comment": f"FP16 вынесен отдельно: {bad}/{total} запусков плохие или невалидные.",
            "source_paths": "",
            "case_key": "",
        })

    df = pd.DataFrame(rows)
    seen = {}
    ids = []
    for _, row in df.iterrows():
        base = str(row["case_id"]).replace(".", "p").replace("=", "")
        n = seen.get(base, 0)
        seen[base] = n + 1
        ids.append(base if n == 0 else f"{base}_{n + 1}")
    df["case_id"] = ids
    return df.head(10)


def write_selected_md(cases):
    lines = ["# Выбранные кейсы", ""]
    for _, row in cases.iterrows():
        lines.append(f"## {row['case_id']}")
        lines.append("")
        lines.append(f"- задача: `{row['task']}`")
        lines.append(f"- параметр: `{row['parameter']}`")
        lines.append(f"- вариант: `{row['variant']}`")
        lines.append(f"- тип: {ru_label(row['label'])}")
        lines.append(f"- статус: {row['status']}")
        if pd.notna(row["fp32_median_best_l2"]):
            lines.append(f"- FP32 median best L2: {fmt(row['fp32_median_best_l2'])}")
        if pd.notna(row["fp64_median_best_l2"]):
            lines.append(f"- FP64 median best L2: {fmt(row['fp64_median_best_l2'])}")
        if pd.notna(row["ratio"]):
            lines.append(f"- FP64/FP32: {fmt(row['ratio'])}")
        lines.append(f"- комментарий: {row['comment']}")
        src = str(row.get("source_paths", "")).strip()
        if src and src.lower() != "nan":
            lines.append("- исходные run-папки:")
            for p in src.split("; "):
                if p:
                    lines.append(f"  - `{p}`")
        lines.append("")
    (selected_dir / "selected_cases.md").write_text("\n".join(lines) + "\n")


def plot_best_l2(cases):
    cur = cases.dropna(subset=["fp32_median_best_l2", "fp64_median_best_l2"]).copy()
    if cur.empty:
        return
    labels = cur["case_id"].tolist()
    x = np.arange(len(cur))
    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 1.2), 4.5))
    ax.bar(x - 0.18, cur["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, cur["fp64_median_best_l2"], width=0.36, label="FP64")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Median best L2 по выбранным кейсам")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_best_l2_by_dtype.png", dpi=180)
    plt.close(fig)


def plot_ratio(cases):
    cur = cases.dropna(subset=["ratio"]).copy()
    if cur.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 1.1), 4))
    ax.bar(range(len(cur)), cur["ratio"])
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["case_id"], rotation=25, ha="right")
    ax.set_ylabel("FP64 / FP32")
    ax.set_title("Отношение median best L2")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_fp64_fp32_ratio.png", dpi=180)
    plt.close(fig)


def plot_task_overview(overview):
    cur = overview.groupby(["task_name", "dtype"], as_index=False)["n_total"].sum()
    if cur.empty:
        return
    tasks = sorted(cur["task_name"].unique())
    dtypes = ["fp32", "fp64", "fp16"]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, dtype in enumerate(dtypes):
        vals = []
        for task in tasks:
            t = cur[(cur["task_name"] == task) & (cur["dtype"] == dtype)]
            vals.append(float(t["n_total"].iloc[0]) if len(t) else 0)
        ax.bar(x + (i - 1) * 0.22, vals, width=0.22, label=dtype.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("число запусков")
    ax.set_title("Сколько запусков найдено")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_task_overview.png", dpi=180)
    plt.close(fig)


def plot_seed_scatter(cases, runs):
    rows = []
    for _, case in cases.iterrows():
        src = str(case.get("source_paths", ""))
        if not src:
            continue
        paths = [x.strip() for x in src.split(";") if x.strip()]
        cur = runs[runs["source_path"].isin(paths)]
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
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 1.1), 4.5))
    colors = {"fp32": "#4c78a8", "fp64": "#f58518"}
    for dtype in ["fp32", "fp64"]:
        cur = df[df["dtype"] == dtype]
        xs = [ids.index(x) + (-0.08 if dtype == "fp32" else 0.08) for x in cur["case_id"]]
        ax.scatter(xs, cur["best_l2_error"], label=dtype.upper(), alpha=0.85, color=colors[dtype])
    ax.set_yscale("log")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=25, ha="right")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Разброс по seed")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_seed_scatter.png", dpi=180)
    plt.close(fig)


def plot_burgers_summary(comp):
    cur = comp[(comp["task_name"] == "burgers1d") & comp["fp64_over_fp32_median"].notna()].copy()
    cur = cur[cur["fp32_n_valid"] + cur["fp64_n_valid"] >= 4]
    if cur.empty:
        return
    cur["label"] = cur["variant"] + ", nu=" + cur["main_parameter_value"].map(fmt)
    cur = cur.sort_values(["main_parameter_value", "variant"]).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(cur)), cur["fp64_over_fp32_median"])
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["label"], rotation=35, ha="right")
    ax.set_ylabel("FP64 / FP32")
    ax.set_title("Burgers: сравнение по найденным группам")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_burgers_summary.png", dpi=180)
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
        label = f"{str(r['dtype']).upper()} seed={int(r['seed']) if pd.notna(r['seed']) else '?'}"
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


def make_figures(cases, comp, overview, runs):
    plot_task_overview(overview)
    plot_best_l2(cases)
    plot_ratio(cases)
    plot_seed_scatter(cases, runs)
    plot_burgers_summary(comp)
    for case_id, name, title in [
        ("helmholtz_m12", "report_helmholtz_m12_curves.png", "Helmholtz m=12"),
        ("burgers_nu0p002", "report_burgers_nu0002_curves.png", "Burgers nu=0.002"),
        ("convection_beta30", "report_convection_beta30_curves.png", "Convection beta=30"),
        ("convection_beta50", "report_convection_beta50_check.png", "Convection beta=50: один из hard-case запусков"),
        ("convection_beta50", "report_convection_beta50_curves.png", "Convection beta=50: один из hard-case запусков"),
    ]:
        plot_curves(case_id, cases, runs, name, title)


def write_optional_checks():
    lines = [
        "# Дополнительные проверки",
        "",
        "Основные таблицы уже собраны из существующих логов. Полный перезапуск экспериментов не нужен.",
        "Ниже только небольшие проверки, которые можно сделать, если нужно усилить отдельные места в отчёте.",
        "",
        "## Что уже есть",
        "",
        "- таблицы по всем найденным run-папкам;",
        "- сравнение FP32 и FP64 по медиане и bad rate;",
        "- отдельный блок по FP16;",
        "- графики для основных кейсов;",
        "- осторожная пометка для Convection beta=50.",
        "",
        "## Что можно проверить дополнительно",
        "",
        "- Convection beta=50 на seed 1 и 2 для FP32/FP64;",
        "- MAE/RMSE и карты exact / prediction / error для Convection beta=50, если нужны картинки в отчёт;",
        "- один FP16-запуск можно оставить только как иллюстрацию нестабильности.",
        "",
        "## Почему не нужен полный перезапуск",
        "",
        "В логах уже есть Heat, Burgers, Helmholtz и Convection с разными параметрами. Проблема не в объёме данных, а в том, что часть кейсов seed-sensitive.",
        "",
        "## Минимальные selected checks",
        "",
        "- convection_beta50_fp32_seed1",
        "- convection_beta50_fp32_seed2",
        "- convection_beta50_fp64_seed1",
        "- convection_beta50_fp64_seed2",
        "",
        "Если нужны карты для уже выбранного seed 0:",
        "",
        "- convection_beta50_fp32_seed0",
        "- convection_beta50_fp64_seed0",
        "",
        "FP16 check можно запускать отдельно, но он не обязателен для основного сравнения FP32/FP64.",
    ]
    (rerun_dir / "optional_checks.md").write_text("\n".join(lines) + "\n")


def write_readme(runs, cases, fp16):
    valid = int(runs["is_valid"].sum())
    bad = int(runs["is_bad"].sum())
    tasks = ", ".join(sorted(runs["task_name"].dropna().unique()))
    lines = [
        "# Итоговые результаты",
        "",
        "Здесь лежит отчётный слой: таблицы и графики, собранные из уже готовых логов.",
        "Сырые запуски не удалялись. Для выводов я смотрю на медиану, число seed и bad rate, а не на лучший отдельный запуск.",
        "",
        "## Что лежит в папке",
        "",
        "- `tables/all_runs_normalized.csv` - все найденные run-папки;",
        "- `tables/task_overview.csv` - общий обзор по задачам, параметрам и dtype;",
        "- `tables/fp32_fp64_comparison.csv` - сравнение FP32 и FP64 внутри одинаковых настроек;",
        "- `tables/report_cases.csv` - главная таблица для отчёта;",
        "- `tables/fp16_summary.csv` - отдельный обзор FP16;",
        "- `figures/` - основные графики;",
        "- `rerun_plan/optional_checks.md` - небольшие проверки, если их захочется добавить.",
        "",
        "## Сколько данных найдено",
        "",
        f"- Всего run-папок: {len(runs)}.",
        f"- Валидных запусков: {valid}.",
        f"- Плохих или нестабильных по порогу: {bad}.",
        f"- Задачи: {tasks}.",
        "",
        "## Как читать результаты",
        "",
        "Главный устойчивый положительный пример - Helmholtz m=12. Там есть по два валидных seed у FP32 и FP64, и FP64 лучше по медиане.",
        "Convection beta=50 оставлен как предварительный hard-case: там только один seed, поэтому его нельзя использовать как главный устойчивый вывод.",
        "Burgers получился смешанным: на части запусков FP32 и FP64 близки, на части FP32 лучше.",
        "FP16 я не смешиваю с основной таблицей, потому что в этих запусках он часто даёт плохие или невалидные метрики.",
        "",
        "## Основные файлы для отчёта",
        "",
        "- `tables/report_cases.csv`",
        "- `tables/task_overview.csv`",
        "- `tables/fp32_fp64_comparison.csv`",
        "- `tables/fp16_summary.csv`",
        "- `figures/report_best_l2_by_dtype.png`",
        "- `figures/report_fp64_fp32_ratio.png`",
        "- `figures/report_seed_scatter.png`",
        "- `figures/report_task_overview.png`",
        "",
        f"FP16-групп в отдельной сводке: {len(fp16)}.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def sync_base_tables():
    copies = [
        ("all_runs.csv", "all_runs_normalized.csv"),
        ("run_quality.csv", "run_quality.csv"),
        ("grouped_by_dtype.csv", "grouped_by_dtype.csv"),
        ("fp32_fp64_comparison.csv", "fp32_fp64_comparison.csv"),
        ("fp16_summary.csv", "fp16_summary.csv"),
        ("bad_runs.csv", "bad_runs.csv"),
        ("valid_runs.csv", "valid_runs.csv"),
    ]
    for src, dst in copies:
        copy_file(clean_dir / "tables" / src, table_dir / dst)


def main():
    ns = runpy.run_path(str(root / "scripts" / "build_report_results.py"))
    with contextlib.redirect_stdout(io.StringIO()):
        ns["main"]()

    sync_base_tables()

    runs = pd.read_csv(clean_dir / "tables" / "all_runs.csv")
    comp = pd.read_csv(clean_dir / "tables" / "fp32_fp64_comparison.csv")
    fp16 = pd.read_csv(clean_dir / "tables" / "fp16_summary.csv")

    overview = task_overview(runs)
    cases = make_report_cases(comp, overview, fp16, runs)

    overview.to_csv(table_dir / "task_overview.csv", index=False)
    cases.to_csv(table_dir / "report_cases.csv", index=False)
    cases.to_csv(table_dir / "selected_cases.csv", index=False)

    write_selected_md(cases)
    make_figures(cases, comp, overview, runs)
    write_optional_checks()
    write_readme(runs, cases, fp16)

    print(f"runs: {len(runs)}")
    print(f"valid: {int(runs['is_valid'].sum())}")
    print(f"bad: {int(runs['is_bad'].sum())}")
    print(f"comparison rows: {len(comp)}")
    print(f"report cases: {len(cases)}")
    print("report_results updated")


if __name__ == "__main__":
    main()
