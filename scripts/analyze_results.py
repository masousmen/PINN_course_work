from pathlib import Path
import json
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "experiments_raw"
OUT_DIR = ROOT / "report_results"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

BAD_L2_THRESHOLD = 0.8

CASE_NAMES = {
    "heat_alpha01": "Heat, α=0.1",
    "helmholtz_m12_long": "Helmholtz, m=12",
    "helmholtz_m12_rs": "Helmholtz, m=12, resampling",
    "helmholtz_m12_resample128": "Helmholtz, m=12, long run",
    "helmholtz_m7_rs": "Helmholtz, m=7",
    "helmholtz_m8_rs": "Helmholtz, m=8",
    "helmholtz_m10_resample128": "Helmholtz, m=10",
    "helmholtz_m11_rs": "Helmholtz, m=11",
    "convection_beta30": "Convection, β=30",
    "convection_beta50": "Convection, β=50, diagnostic",
    "burgers_nu0p002": "Burgers, ν=0.002",
    "burgers_nu0p001": "Burgers, ν=0.001",
    "fp16_summary": "FP16 failure cases",
}

HELMHOLTZ_NAMES = {
    "helmholtz_m12_long": "m=12",
    "helmholtz_m12_rs": "m=12, rs",
    "helmholtz_m12_resample128": "m=12, long",
    "helmholtz_m7_rs": "m=7",
    "helmholtz_m8_rs": "m=8",
    "helmholtz_m10_resample128": "m=10",
    "helmholtz_m11_rs": "m=11",
}

MAIN_HELMHOLTZ_IDS = {
    "helmholtz_m12_long",
    "helmholtz_m12_rs",
    "helmholtz_m12_resample128",
    "helmholtz_m7_rs",
    "helmholtz_m11_rs",
}

TARGET_TABLES = {
    "all_runs_normalized.csv",
    "bad_runs.csv",
    "task_overview.csv",
    "fp32_fp64_comparison.csv",
    "fp16_summary.csv",
    "report_main_cases.csv",
    "report_helmholtz_cases.csv",
    "report_diagnostic_cases.csv",
}

TARGET_FIGURES = {
    "report_main_best_l2_by_dtype.png",
    "report_main_fp64_fp32_ratio.png",
    "report_main_seed_scatter.png",
    "report_helmholtz_main_ratio.png",
    "report_helmholtz_rs_sweep.png",
    "report_helmholtz_m12_curves.png",
    "report_convection_beta30_curves.png",
    "report_burgers_summary.png",
    "report_fp16_summary.png",
    "report_diagnostic_seed_sensitive.png",
}

def clean_text(text):
    return " ".join(str(text).replace("\n", " ").split())

def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.lower() in {"none", "nan", "null"}:
            return np.nan
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return np.nan

def to_int(x):
    val = to_float(x)
    if pd.isna(val):
        return np.nan
    return int(val)

def fmt_num(x):
    val = to_float(x)
    if pd.isna(val):
        return ""
    if abs(val - round(val)) < 1e-10:
        return str(int(round(val)))
    return f"{val:.6g}"

def same_num(a, b, eps=1e-9):
    a = to_float(a)
    b = to_float(b)
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < eps

def flatten_dict(d, prefix=""):
    out = {}

    if not isinstance(d, dict):
        return out

    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value

    return out

def get_first(data, names, default=None):
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_number(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return np.nan

    raw = match.group(1)
    raw = raw.replace("p", ".")
    return to_float(raw)

def infer_dtype(text):
    text = text.lower()
    match = re.search(r"fp(16|32|64)", text)
    if match:
        return "fp" + match.group(1)
    return ""

def infer_seed(text):
    text = text.lower()

    patterns = [
        r"(?:seed|s)[_-]?(\d+)",
        r"fp(?:16|32|64)[_-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return np.nan

def infer_task(text):
    text = text.lower()

    if "helmholtz" in text:
        return "helmholtz1d"
    if "burgers" in text:
        return "burgers1d"
    if "convection" in text:
        return "convection1d"
    if "heat" in text:
        return "heat1d"

    return ""

def infer_parameter(task_name, text):
    text = text.lower()

    if task_name == "helmholtz1d":
        value = parse_number(r"(?:^|[_/\-])m([0-9]+(?:p[0-9]+|\.[0-9]+)?)", text)
        return "m", value

    if task_name == "burgers1d":
        value = parse_number(r"nu[_=\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", text)
        return "nu", value

    if task_name == "convection1d":
        value = parse_number(r"beta[_=\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", text)
        return "beta", value

    if task_name == "heat1d":
        value = parse_number(r"alpha[_=\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", text)
        if pd.isna(value):
            value = 0.1
        return "alpha", value

    return "", np.nan

def infer_variant(task_name, parameter_value, text):
    text = text.lower()

    if task_name == "heat1d":
        return "heat1d"

    if task_name == "helmholtz1d":
        m = int(parameter_value) if pd.notna(parameter_value) else None

        if "helmholtz_resample_long" in text:
            return "helmholtz_resample_long"
        if "resample_proven_128" in text:
            return "resample_proven_128"
        if "helmholtz_rs_m" in text and m is not None:
            return f"helmholtz_rs_m{m}"
        if "helmholtz_main" in text:
            return "helmholtz_main"
        if m is not None:
            return f"helmholtz_m{m}"

    if task_name == "burgers1d":
        if "burgers_more_points" in text:
            return "burgers_more_points"
        if "low_nu" in text:
            return "burgers_low_nu"
        return "burgers"

    if task_name == "convection1d":
        if "convection_beta30_lbfgs_grid" in text:
            return "convection_beta30_lbfgs_grid"
        if "convection_beta50_wide_lbfgs" in text:
            return "convection_beta50_wide_lbfgs"
        return "convection"

    return "unknown"

def read_metrics(run_dir):
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None, None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None, path

    return df, path

def find_metric_col(df, words):
    if df is None or df.empty:
        return None

    for col in df.columns:
        low = str(col).lower()
        if low in words:
            return col

    for col in df.columns:
        low = str(col).lower()
        for word in words:
            if word in low:
                return col

    return None

def metric_min(df, words):
    col = find_metric_col(df, words)
    if col is None:
        return np.nan

    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return np.nan

    return vals.min()

def metric_last(df, words):
    col = find_metric_col(df, words)
    if col is None:
        return np.nan

    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan

    return vals.iloc[-1]

def get_setting(flat, names):
    return get_first(flat, names, np.nan)

def run_from_summary(path):
    data = read_json(path)
    if data is None:
        return None

    run_dir = path.parent
    rel_run = run_dir.relative_to(ROOT)
    text = str(rel_run).lower()

    flat = flatten_dict(data)

    task_name = get_first(
        flat,
        ["task_name", "config.task_name", "params.task_name", "task"],
        "",
    )
    if not task_name:
        task_name = infer_task(text)

    dtype = get_first(
        flat,
        ["dtype", "config.dtype", "params.dtype"],
        "",
    )
    if not dtype:
        dtype = infer_dtype(text)
    dtype = str(dtype).lower()

    seed = get_first(
        flat,
        ["seed", "config.seed", "params.seed"],
        np.nan,
    )
    seed = to_int(seed)
    if pd.isna(seed):
        seed = infer_seed(text)

    if task_name == "helmholtz1d":
        par_name = "m"
        par_value = get_first(flat, ["m", "config.m", "params.m"], np.nan)
    elif task_name == "burgers1d":
        par_name = "nu"
        par_value = get_first(flat, ["nu", "config.nu", "params.nu"], np.nan)
    elif task_name == "convection1d":
        par_name = "beta"
        par_value = get_first(flat, ["beta", "config.beta", "params.beta"], np.nan)
    elif task_name == "heat1d":
        par_name = "alpha"
        par_value = get_first(flat, ["alpha", "config.alpha", "params.alpha"], np.nan)
    else:
        par_name = ""
        par_value = np.nan

    par_value = to_float(par_value)
    if not par_name or pd.isna(par_value):
        par_name, par_value = infer_parameter(task_name, text)

    variant = get_first(
        flat,
        ["variant", "config.variant", "experiment", "experiment_name", "run_name"],
        "",
    )
    if not variant:
        variant = infer_variant(task_name, par_value, text)

    metrics, metrics_path = read_metrics(run_dir)

    best_l2 = get_first(
        flat,
        [
            "best_l2_error",
            "best_l2",
            "min_l2_error",
            "best_relative_l2",
            "metrics.best_l2_error",
            "result.best_l2_error",
        ],
        np.nan,
    )
    best_l2 = to_float(best_l2)
    if pd.isna(best_l2):
        best_l2 = metric_min(metrics, {"l2_error", "relative_l2_error", "rel_l2_error"})

    final_l2 = get_first(
        flat,
        [
            "final_l2_error",
            "l2_error",
            "relative_l2_error",
            "metrics.final_l2_error",
            "result.final_l2_error",
        ],
        np.nan,
    )
    final_l2 = to_float(final_l2)
    if pd.isna(final_l2):
        final_l2 = metric_last(metrics, {"l2_error", "relative_l2_error", "rel_l2_error"})

    final_loss = get_first(
        flat,
        ["final_loss", "loss", "metrics.final_loss", "result.final_loss"],
        np.nan,
    )
    final_loss = to_float(final_loss)
    if pd.isna(final_loss):
        final_loss = metric_last(metrics, {"total_loss", "loss"})

    row = {
        "source_path": str(rel_run),
        "summary_path": str(path.relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)) if metrics_path else "",
        "task_name": task_name,
        "variant": str(variant),
        "dtype": dtype,
        "seed": seed,
        "main_parameter_name": par_name,
        "main_parameter_value": par_value,
        "best_l2_error": best_l2,
        "final_l2_error": final_l2,
        "final_loss": final_loss,
        "hid_size": to_float(get_setting(flat, ["hid_size", "config.hid_size", "hidden_size", "config.hidden_size"])),
        "num_layers": to_float(get_setting(flat, ["num_layers", "config.num_layers"])),
        "n_collocation": to_float(get_setting(flat, ["n_collocation", "config.n_collocation"])),
        "n_ic": to_float(get_setting(flat, ["n_ic", "config.n_ic"])),
        "n_bc": to_float(get_setting(flat, ["n_bc", "config.n_bc"])),
        "adam_steps": to_float(get_setting(flat, ["adam_steps", "config.adam_steps"])),
        "lbfgs_steps": to_float(get_setting(flat, ["lbfgs_steps", "config.lbfgs_steps"])),
        "lr_adam": to_float(get_setting(flat, ["lr_adam", "config.lr_adam"])),
        "lbfgs_lr": to_float(get_setting(flat, ["lbfgs_lr", "config.lbfgs_lr"])),
        "resample_every": to_float(get_setting(flat, ["resample_every", "config.resample_every"])),
    }

    row["is_valid"] = bool(np.isfinite(row["best_l2_error"]))
    row["is_bad"] = bool((not row["is_valid"]) or row["best_l2_error"] > BAD_L2_THRESHOLD)

    return row

def read_runs():
    files = list(RAW_DIR.rglob("summary.json"))

    if not files:
        files = [p for p in RAW_DIR.rglob("*.json") if "summary" in p.name.lower()]

    rows = []
    for path in files:
        row = run_from_summary(path)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(["task_name", "main_parameter_value", "variant", "dtype", "seed"])
    return df.reset_index(drop=True)

def make_case_key(row):
    keys = [
        "task_name",
        "variant",
        "main_parameter_name",
        "main_parameter_value",
        "hid_size",
        "num_layers",
        "n_collocation",
        "n_ic",
        "n_bc",
        "adam_steps",
        "lbfgs_steps",
        "lr_adam",
        "lbfgs_lr",
        "resample_every",
    ]

    parts = []
    for key in keys:
        value = row.get(key, np.nan)
        if isinstance(value, float):
            value = fmt_num(value)
        parts.append(str(value))

    return "|".join(parts)

def make_task_overview(runs):
    if runs.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["task_name", "main_parameter_name", "main_parameter_value", "dtype"]

    for values, cur in runs.groupby(group_cols, dropna=False):
        valid = cur[cur["is_valid"]].copy()
        best = pd.to_numeric(valid["best_l2_error"], errors="coerce")

        rows.append({
            "task_name": values[0],
            "main_parameter_name": values[1],
            "main_parameter_value": values[2],
            "dtype": values[3],
            "n_total": len(cur),
            "n_valid": int(cur["is_valid"].sum()),
            "n_bad": int(cur["is_bad"].sum()),
            "bad_rate": float(cur["is_bad"].mean()) if len(cur) else np.nan,
            "median_best_l2": best.median(),
            "mean_best_l2": best.mean(),
            "min_best_l2": best.min(),
            "max_best_l2": best.max(),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["task_name", "main_parameter_value", "dtype"])

def make_fp32_fp64_comparison(runs):
    if runs.empty:
        return pd.DataFrame()

    cur = runs[runs["dtype"].isin(["fp32", "fp64"])].copy()
    if cur.empty:
        return pd.DataFrame()

    cur["case_key"] = cur.apply(make_case_key, axis=1)

    rows = []
    for case_key, part in cur.groupby("case_key"):
        base = part.iloc[0]

        row = {
            "case_key": case_key,
            "task_name": base["task_name"],
            "variant": base["variant"],
            "main_parameter_name": base["main_parameter_name"],
            "main_parameter_value": base["main_parameter_value"],
            "hid_size": base["hid_size"],
            "num_layers": base["num_layers"],
            "n_collocation": base["n_collocation"],
            "n_ic": base["n_ic"],
            "n_bc": base["n_bc"],
            "adam_steps": base["adam_steps"],
            "lbfgs_steps": base["lbfgs_steps"],
            "lr_adam": base["lr_adam"],
            "lbfgs_lr": base["lbfgs_lr"],
            "resample_every": base["resample_every"],
        }

        for dtype in ["fp32", "fp64"]:
            d = part[part["dtype"] == dtype]
            valid = d[d["is_valid"]]
            best = pd.to_numeric(valid["best_l2_error"], errors="coerce")

            row[f"{dtype}_n_total"] = len(d)
            row[f"{dtype}_n_valid"] = int(d["is_valid"].sum())
            row[f"{dtype}_n_bad"] = int(d["is_bad"].sum())
            row[f"{dtype}_bad_rate"] = float(d["is_bad"].mean()) if len(d) else np.nan
            row[f"{dtype}_median_best_l2"] = best.median()
            row[f"{dtype}_min_best_l2"] = best.min()
            row[f"{dtype}_max_best_l2"] = best.max()
            row[f"{dtype}_source_paths"] = "; ".join(d["source_path"].astype(str))

        a = row.get("fp32_median_best_l2", np.nan)
        b = row.get("fp64_median_best_l2", np.nan)
        row["fp64_over_fp32_median"] = b / a if pd.notna(a) and a != 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["task_name", "main_parameter_value", "variant"])

def pick_case(comp, task, par_name, par_value, variant=None):
    if comp.empty:
        return None

    cur = comp[
        (comp["task_name"] == task)
        & (comp["main_parameter_name"] == par_name)
    ].copy()

    cur = cur[cur["main_parameter_value"].map(lambda x: same_num(x, par_value))]

    if variant is not None:
        exact = cur[cur["variant"] == variant].copy()
        if not exact.empty:
            cur = exact

    if cur.empty:
        return None

    cur = cur.sort_values(
        [
            "fp32_n_valid",
            "fp64_n_valid",
            "fp32_bad_rate",
            "fp64_bad_rate",
            "fp64_over_fp32_median",
        ],
        ascending=[False, False, True, True, True],
    )

    return cur.iloc[0]

def case_id_from_row(row):
    task = row["task_name"]
    value = row["main_parameter_value"]
    variant = row["variant"]

    if task == "heat1d":
        return "heat_alpha01"

    if task == "helmholtz1d":
        m = fmt_num(value)

        if variant == "helmholtz_resample_long":
            return f"helmholtz_m{m}_long"
        if variant == "resample_proven_128":
            return f"helmholtz_m{m}_resample128"
        if str(variant).startswith("helmholtz_rs_m"):
            return f"helmholtz_m{m}_rs"
        if variant == "helmholtz_main":
            return f"helmholtz_m{m}_main_old"

        return f"helmholtz_m{m}"

    if task == "convection1d":
        return f"convection_beta{fmt_num(value)}"

    if task == "burgers1d":
        value = fmt_num(value).replace(".", "p")
        return f"burgers_nu{value}"

    return f"{task}_{fmt_num(value)}"

def row_to_report(row, label, status, comment, case_id=None):
    if case_id is None:
        case_id = case_id_from_row(row)

    src = []
    for dtype in ["fp32", "fp64"]:
        text = row.get(f"{dtype}_source_paths", "")
        if isinstance(text, str) and text:
            src.append(text)

    return {
        "case_id": case_id,
        "task": row["task_name"],
        "parameter": f"{row['main_parameter_name']}={fmt_num(row['main_parameter_value'])}",
        "variant": row["variant"],
        "dtype_comparison": "FP32 vs FP64",
        "n_seed_fp32": row.get("fp32_n_valid", np.nan),
        "n_seed_fp64": row.get("fp64_n_valid", np.nan),
        "fp32_median_best_l2": row.get("fp32_median_best_l2", np.nan),
        "fp64_median_best_l2": row.get("fp64_median_best_l2", np.nan),
        "ratio": row.get("fp64_over_fp32_median", np.nan),
        "bad_rate_fp32": row.get("fp32_bad_rate", np.nan),
        "bad_rate_fp64": row.get("fp64_bad_rate", np.nan),
        "label": label,
        "status": status,
        "comment": clean_text(comment),
        "source_paths": "; ".join(src),
        "case_key": row.get("case_key", ""),
    }

def make_heat_case(overview, runs):
    cur = overview[
        (overview["task_name"] == "heat1d")
        & (overview["dtype"].isin(["fp32", "fp64"]))
    ]

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
        "comment": "На простой heat-задаче обе точности работают хорошо; это проверка всей схемы обучения.",
        "source_paths": "; ".join(src),
        "case_key": "",
    }

def make_fp16_table(overview):
    if overview.empty:
        return pd.DataFrame()

    fp16 = overview[overview["dtype"] == "fp16"].copy()
    if fp16.empty:
        return pd.DataFrame()

    cols = [
        "task_name",
        "main_parameter_name",
        "main_parameter_value",
        "n_total",
        "n_valid",
        "n_bad",
        "bad_rate",
        "median_best_l2",
        "min_best_l2",
        "max_best_l2",
    ]

    return fp16[cols].sort_values(["task_name", "main_parameter_value"])

def make_fp16_case(fp16_table):
    if fp16_table.empty:
        return None

    total = int(fp16_table["n_total"].sum())
    bad = int(fp16_table["n_bad"].sum())

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

def make_helmholtz_cases(comp):
    specs = [
        (12, "helmholtz_resample_long", "helmholtz_m12_long", "главный положительный пример", "основной Helmholtz", "В этом запуске есть по два валидных seed; медиана FP64 заметно ниже медианы FP32."),
        (12, "helmholtz_rs_m12", "helmholtz_m12_rs", "дополнительный чистый пример", "основной Helmholtz", "Ещё один устойчивый m=12 с ресемплированием; FP64 снова даёт меньшую медианную ошибку."),
        (12, "resample_proven_128", "helmholtz_m12_resample128", "дополнительный чистый пример", "основной Helmholtz", "Похожая настройка с более длинным L-BFGS; вывод совпадает с основным m=12."),
        (7, "helmholtz_rs_m7", "helmholtz_m7_rs", "положительный пример", "дополнительный Helmholtz", "На меньшем m результат тоже чистый: оба dtype имеют по два валидных seed."),
        (8, "helmholtz_rs_m8", "helmholtz_m8_rs", "положительный пример", "дополнительный Helmholtz", "Это не старый unstable m=8, а более аккуратный rs-запуск без плохих seed."),
        (10, "resample_proven_128", "helmholtz_m10_resample128", "умеренный пример", "дополнительный Helmholtz", "Вспомогательный Helmholtz-кейс: FP64 лучше, но не так резко, как на m=12."),
        (11, "helmholtz_rs_m11", "helmholtz_m11_rs", "умеренно положительный пример", "дополнительный Helmholtz", "FP64 лучше по медиане, но отрыв меньше, чем на m=12."),
    ]

    rows = []

    for m, variant, case_id, label, status, comment in specs:
        row = pick_case(comp, "helmholtz1d", "m", m, variant)
        if row is None:
            continue

        enough_seeds = row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2
        clean_bad_rate = row["fp32_bad_rate"] < 0.5 and row["fp64_bad_rate"] < 0.5

        if not enough_seeds or not clean_bad_rate:
            continue

        rows.append(row_to_report(row, label, status, comment, case_id=case_id))

    return pd.DataFrame(rows)

def make_report_tables(comp, overview, runs):
    main_rows = []
    diag_rows = []

    heat = make_heat_case(overview, runs)
    if heat is not None:
        main_rows.append(heat)

    helm = make_helmholtz_cases(comp)
    if not helm.empty:
        for _, row in helm.iterrows():
            if row["case_id"] in MAIN_HELMHOLTZ_IDS:
                main_rows.append(row.to_dict())

    main_specs = [
        ("convection1d", "beta", 30, "convection_beta30_lbfgs_grid", "convection_beta30", "аккуратный пример convection", "основной отчёт", "Есть по два seed у FP32 и FP64; FP64 лучше умеренно, без истории про полный провал FP32."),
        ("burgers1d", "nu", 0.002, "burgers_more_points", "burgers_nu0p002", "результаты близкие", "основной отчёт", "Burgers nu=0.002 показывает близкие результаты FP32 и FP64."),
        ("burgers1d", "nu", 0.001, "burgers_more_points", "burgers_nu0p001", "смешанный результат", "основной отчёт", "На этом Burgers-запуске FP64 не даёт устойчивого преимущества; это полезный отрицательный пример."),
    ]

    for task, par, value, variant, case_id, label, status, comment in main_specs:
        row = pick_case(comp, task, par, value, variant)
        if row is not None:
            main_rows.append(row_to_report(row, label, status, comment, case_id=case_id))

    fp16_table = make_fp16_table(overview)
    fp16_case = make_fp16_case(fp16_table)
    if fp16_case is not None:
        main_rows.append(fp16_case)

    diag_specs = [
        ("convection1d", "beta", 50, "convection_beta50_wide_lbfgs", "convection_beta50", "требует проверки", "диагностика", "FP64 выглядит сильно лучше, но здесь только один seed; FP32 мог не сойтись из-за неудачного старта, поэтому кейс нельзя делать главным выводом."),
        ("helmholtz1d", "m", 8, "helmholtz_main", "helmholtz_m8_main_old", "зависит от seed", "диагностика", "Старый m=8 показывает сильную зависимость от seed; его лучше обсуждать отдельно от основного Helmholtz-блока."),
        ("helmholtz1d", "m", 15, "helmholtz_m15", "helmholtz_m15_hard", "сложный режим", "диагностика", "При большем m обе точности часто не сходились; это скорее граница текущей схемы обучения."),
    ]

    for task, par, value, variant, case_id, label, status, comment in diag_specs:
        row = pick_case(comp, task, par, value, variant)
        if row is not None:
            diag_rows.append(row_to_report(row, label, status, comment, case_id=case_id))

    main = pd.DataFrame(main_rows)
    diag = pd.DataFrame(diag_rows)

    return main, helm, diag, fp16_table

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def archive_old_outputs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    table_archive = TABLE_DIR / "archive"
    fig_archive = FIG_DIR / "archive"
    table_archive.mkdir(exist_ok=True)
    fig_archive.mkdir(exist_ok=True)

    for path in TABLE_DIR.glob("*.csv"):
        if path.name not in TARGET_TABLES:
            dst = table_archive / path.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(path), str(dst))

    for path in FIG_DIR.glob("*.png"):
        if path.name not in TARGET_FIGURES:
            dst = fig_archive / path.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(path), str(dst))

def add_plot_names(df, mapping=None):
    if df.empty:
        return df

    df = df.copy()
    if mapping is None:
        mapping = CASE_NAMES

    df["plot_name"] = df["case_id"].map(mapping).fillna(df["case_id"])
    return df

def plot_main_bars(main):
    cur = main.dropna(subset=["fp32_median_best_l2", "fp64_median_best_l2"]).copy()
    cur = cur[cur["case_id"] != "fp16_summary"]
    cur = add_plot_names(cur)

    if cur.empty:
        return

    x = np.arange(len(cur))

    fig, ax = plt.subplots(figsize=(max(9, len(cur) * 1.15), 4.8))
    ax.bar(x - 0.18, cur["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, cur["fp64_median_best_l2"], width=0.36, label="FP64")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cur["plot_name"], rotation=30, ha="right")
    ax.set_ylabel("Best L2 error")
    ax.set_title("Main cases: median best L2")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_main_best_l2_by_dtype.png", dpi=180)
    plt.close(fig)

def plot_main_ratio(main):
    cur = main.dropna(subset=["ratio"]).copy()
    cur = cur[cur["case_id"] != "fp16_summary"]
    cur = add_plot_names(cur)

    if cur.empty:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(cur) * 1.15), 4.3))
    ax.bar(range(len(cur)), cur["ratio"])
    ax.axhline(1.0, linewidth=1)

    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["plot_name"], rotation=30, ha="right")
    ax.set_ylabel("FP64 / FP32 median best L2")
    ax.set_title("Main cases: FP64 to FP32 error ratio")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_main_fp64_fp32_ratio.png", dpi=180)
    plt.close(fig)

def plot_helmholtz_ratio(helm):
    cur = helm.dropna(subset=["ratio"]).copy()
    cur = add_plot_names(cur, HELMHOLTZ_NAMES)

    if cur.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 1.0), 4.3))
    ax.bar(range(len(cur)), cur["ratio"])
    ax.axhline(1.0, linewidth=1)

    ax.set_yscale("log")
    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["plot_name"], rotation=25, ha="right")
    ax.set_ylabel("FP64 / FP32 median best L2")
    ax.set_title("Helmholtz selected runs")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_helmholtz_main_ratio.png", dpi=180)
    plt.close(fig)

def plot_helmholtz_sweep(helm):
    cur = helm.dropna(subset=["ratio"]).copy()
    if cur.empty:
        return

    cur["m"] = cur["parameter"].str.extract(r"m=([0-9.]+)").astype(float)
    cur = cur.sort_values(["m", "case_id"])
    cur = add_plot_names(cur, HELMHOLTZ_NAMES)

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.scatter(cur["m"], cur["ratio"], s=70)

    for _, row in cur.iterrows():
        ax.text(row["m"], row["ratio"] * 1.08, row["plot_name"], fontsize=8, ha="center")

    ax.axhline(1.0, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel("FP64 / FP32 median best L2")
    ax.set_title("Helmholtz: selected comparable runs")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_helmholtz_rs_sweep.png", dpi=180)
    plt.close(fig)

def split_sources(text):
    if not isinstance(text, str):
        return []
    return [x.strip() for x in text.split(";") if x.strip()]

def runs_for_case(case, runs):
    paths = split_sources(case.get("source_paths", ""))
    if not paths:
        return pd.DataFrame()

    return runs[runs["source_path"].isin(paths)].copy()

def plot_seed_scatter(cases, runs, file_name, title):
    if cases.empty:
        return

    rows = []
    cases = add_plot_names(cases)

    for _, case in cases.iterrows():
        cur = runs_for_case(case, runs)

        for _, run in cur.iterrows():
            if run["dtype"] not in {"fp32", "fp64"}:
                continue

            rows.append({
                "case_id": case["case_id"],
                "plot_name": case["plot_name"],
                "dtype": run["dtype"],
                "best_l2_error": run["best_l2_error"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    names = list(cases["plot_name"])
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.1), 4.8))

    for dtype, shift in [("fp32", -0.08), ("fp64", 0.08)]:
        cur = df[df["dtype"] == dtype]
        xs = [names.index(name) + shift for name in cur["plot_name"]]
        ax.scatter(xs, cur["best_l2_error"], label=dtype.upper(), alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Best L2 error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / file_name, dpi=180)
    plt.close(fig)

def plot_burgers_summary(main):
    cur = main[main["case_id"].isin(["burgers_nu0p001", "burgers_nu0p002"])].copy()
    cur = cur.dropna(subset=["fp32_median_best_l2", "fp64_median_best_l2"])
    cur = add_plot_names(cur)

    if cur.empty:
        return

    x = np.arange(len(cur))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - 0.18, cur["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, cur["fp64_median_best_l2"], width=0.36, label="FP64")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cur["plot_name"])
    ax.set_ylabel("Best L2 error")
    ax.set_title("Burgers selected runs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_burgers_summary.png", dpi=180)
    plt.close(fig)

def plot_fp16_summary(fp16_table):
    if fp16_table.empty:
        return

    cur = fp16_table.copy()
    cur["name"] = cur["task_name"].astype(str) + ", " + cur["main_parameter_name"].astype(str) + "=" + cur["main_parameter_value"].map(fmt_num)

    fig, ax = plt.subplots(figsize=(max(8, len(cur) * 0.9), 4.0))
    ax.bar(range(len(cur)), cur["bad_rate"])

    ax.set_xticks(range(len(cur)))
    ax.set_xticklabels(cur["name"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Bad rate")
    ax.set_title("FP16 runs")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_fp16_summary.png", dpi=180)
    plt.close(fig)

def metric_columns_for_plot(df):
    step_col = find_metric_col(df, {"step", "epoch", "iter", "iteration"})
    loss_col = find_metric_col(df, {"total_loss", "loss"})
    l2_col = find_metric_col(df, {"l2_error", "relative_l2_error", "rel_l2_error"})

    return step_col, loss_col, l2_col

def plot_curves(case_id, cases, runs, file_name, title):
    row = cases[cases["case_id"] == case_id]
    if row.empty:
        return

    case = row.iloc[0]
    cur_runs = runs_for_case(case, runs)
    cur_runs = cur_runs[cur_runs["dtype"].isin(["fp32", "fp64"])]

    if cur_runs.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    any_loss = False
    any_l2 = False

    for _, run in cur_runs.sort_values(["dtype", "seed"]).iterrows():
        metrics_path = run.get("metrics_path", "")
        if not metrics_path:
            continue

        path = ROOT / metrics_path
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        step_col, loss_col, l2_col = metric_columns_for_plot(df)

        if step_col is None:
            x = np.arange(len(df))
        else:
            x = pd.to_numeric(df[step_col], errors="coerce")

        seed = run["seed"]
        seed = int(seed) if pd.notna(seed) else "?"
        label = f"{run['dtype'].upper()} seed={seed}"

        if loss_col is not None:
            y = pd.to_numeric(df[loss_col], errors="coerce")
            axes[0].plot(x, y, label=label)
            any_loss = True

        if l2_col is not None:
            y = pd.to_numeric(df[l2_col], errors="coerce")
            axes[1].plot(x, y, label=label)
            any_l2 = True

    axes[0].set_title("Loss")
    axes[1].set_title("Relative L2 error")

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    if any_loss:
        axes[0].legend(fontsize=8)
    if any_l2:
        axes[1].legend(fontsize=8)

    if not any_loss:
        axes[0].text(0.5, 0.5, "no loss column", ha="center", va="center", transform=axes[0].transAxes)
    if not any_l2:
        axes[1].text(0.5, 0.5, "no L2 column", ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / file_name, dpi=180)
    plt.close(fig)

def make_figures(main, helm, diag, fp16_table, runs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    main_no_fp16 = main[main["case_id"] != "fp16_summary"].copy()

    plot_main_bars(main)
    plot_main_ratio(main)
    plot_seed_scatter(main_no_fp16, runs, "report_main_seed_scatter.png", "Main cases: seed-level errors")

    plot_helmholtz_ratio(helm)
    plot_helmholtz_sweep(helm)

    plot_curves(
        "helmholtz_m12_long",
        pd.concat([main, helm, diag], ignore_index=True),
        runs,
        "report_helmholtz_m12_curves.png",
        "Helmholtz, m=12",
    )

    plot_curves(
        "convection_beta30",
        main,
        runs,
        "report_convection_beta30_curves.png",
        "Convection, β=30",
    )

    plot_burgers_summary(main)
    plot_fp16_summary(fp16_table)

    if not diag.empty:
        plot_seed_scatter(diag, runs, "report_diagnostic_seed_sensitive.png", "Diagnostic cases")

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = read_runs()

    if runs.empty:
        print("No runs found. Check experiments_raw/.")
        return

    archive_old_outputs()

    overview = make_task_overview(runs)
    comp = make_fp32_fp64_comparison(runs)
    main_cases, helm_cases, diag_cases, fp16_table = make_report_tables(comp, overview, runs)

    save_csv(runs, TABLE_DIR / "all_runs_normalized.csv")
    save_csv(runs[runs["is_bad"]], TABLE_DIR / "bad_runs.csv")
    save_csv(overview, TABLE_DIR / "task_overview.csv")
    save_csv(comp, TABLE_DIR / "fp32_fp64_comparison.csv")
    save_csv(fp16_table, TABLE_DIR / "fp16_summary.csv")
    save_csv(main_cases, TABLE_DIR / "report_main_cases.csv")
    save_csv(helm_cases, TABLE_DIR / "report_helmholtz_cases.csv")
    save_csv(diag_cases, TABLE_DIR / "report_diagnostic_cases.csv")

    make_figures(main_cases, helm_cases, diag_cases, fp16_table, runs)

    print("Done.")
    print(f"Runs: {len(runs)}")
    print(f"Valid runs: {int(runs['is_valid'].sum())}")
    print(f"Bad runs: {int(runs['is_bad'].sum())}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")

if __name__ == "__main__":
    main()