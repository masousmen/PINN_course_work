import ast
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
out_dir = root / "report_results_clean"
table_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
notes_dir = out_dir / "notes"
selected_dir = out_dir / "selected_runs"
rerun_dir = out_dir / "rerun_plan"

for p in [table_dir, fig_dir, notes_dir, selected_dir, rerun_dir]:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def write_text(p, lines):
    Path(p).write_text("\n".join(lines) + "\n")


def ru_conclusion(x):
    d = {
        "stable_fp64_better": "FP64 заметно лучше",
        "moderate_fp64_better": "FP64 немного лучше",
        "similar": "FP32 и FP64 близки",
        "fp32_better": "FP32 лучше",
        "single_seed_hard_case": "один сильный seed, нужна проверка",
        "seed_sensitive": "зависит от seed",
        "insufficient_data": "не хватает данных",
        "fp16 failed or unstable": "FP16 нестабилен",
    }
    return d.get(str(x), str(x))


def ru_confidence(x):
    d = {
        "strong": "сильная",
        "medium": "средняя",
        "needs_check": "нужна проверка",
    }
    return d.get(str(x), str(x))


def to_num(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ["nan", "none", "null"]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def norm_dtype(x):
    s = str(x).lower()
    if s in ["float32", "torch.float32"]:
        return "fp32"
    if s in ["float64", "torch.float64"]:
        return "fp64"
    if s in ["float16", "torch.float16"]:
        return "fp16"
    return s


def parse_alpha(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}
    return {}


def get_value(summary, config, name, default=np.nan):
    if name in summary and summary.get(name) is not None:
        return summary.get(name)
    if name in config and config.get(name) is not None:
        return config.get(name)
    return default


def rel(p):
    try:
        return Path(p).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(p)


def source_rank(p):
    s = rel(p)
    if s.startswith("results_exp_"):
        return 0
    if s.startswith("final/final 2/"):
        return 1
    if s.startswith("final/final/"):
        return 2
    if s.startswith("experiments_raw/"):
        return 3
    return 4


def infer_variant(run_name, task_name):
    s = str(run_name)
    s = re.sub(r"^run\d+_", "", s)
    s = re.sub(r"^exp\d+_r\d+_", "", s)
    s = re.sub(r"^overnight_r\d+_", "", s)
    if task_name and s.startswith(str(task_name) + "_"):
        s = s[len(str(task_name)) + 1:]
    s = re.sub(r"_(fp16|fp32|fp64|float16|float32|float64)_s?\d+$", "", s)
    s = re.sub(r"_(fp16|fp32|fp64|float16|float32|float64)_\d+$", "", s)
    s = re.sub(r"^(m\d+|nu\d+p?\d*|beta\d+p?\d*|alpha\d+p?\d*)_", "", s)
    if s.strip() == "":
        return "base"
    return s


def main_parameter(summary, config):
    task = get_value(summary, config, "task_name", "")
    alpha = get_value(summary, config, "alpha", np.nan)
    alpha_dict = parse_alpha(alpha)

    beta = get_value(summary, config, "beta", alpha_dict.get("beta", np.nan))
    nu = get_value(summary, config, "nu", alpha_dict.get("nu", np.nan))
    m = get_value(summary, config, "m", alpha_dict.get("m", np.nan))

    if isinstance(alpha, dict):
        alpha_val = alpha.get("alpha", np.nan)
    else:
        alpha_val = alpha_dict.get("alpha", alpha)

    if task == "burgers1d":
        return "nu", to_num(nu)
    if task == "helmholtz1d":
        return "m", to_num(m)
    if task == "convection1d":
        return "beta", to_num(beta)
    if task == "heat1d":
        val = to_num(alpha_val)
        if finite(val):
            return "alpha", val
        return "task", 0.0
    return "task", 0.0


def metrics_stats(p):
    res = {
        "best_l2_error": np.nan,
        "final_l2_error": np.nan,
        "best_step": np.nan,
        "final_step": np.nan,
        "elapsed_time": np.nan,
        "missing_metrics": False,
        "metrics_has_l2": False,
    }
    if not p.exists():
        res["missing_metrics"] = True
        return res
    try:
        h = pd.read_csv(p)
    except Exception:
        res["missing_metrics"] = True
        return res
    if len(h) == 0 or "l2_error" not in h.columns:
        return res
    vals = pd.to_numeric(h["l2_error"], errors="coerce")
    res["metrics_has_l2"] = True
    if vals.notna().any():
        idx = vals.idxmin()
        res["best_l2_error"] = float(vals.loc[idx])
        res["final_l2_error"] = float(vals.iloc[-1])
        if "step" in h.columns:
            steps = pd.to_numeric(h["step"], errors="coerce")
            res["best_step"] = float(steps.loc[idx])
            res["final_step"] = float(steps.iloc[-1])
    if "time_sec" in h.columns:
        t = pd.to_numeric(h["time_sec"], errors="coerce")
        if t.notna().any():
            res["elapsed_time"] = float(t.iloc[-1])
    return res


def scan_summary_files():
    folders = []
    for p in root.glob("results_exp_*"):
        if p.is_dir():
            folders.append(p)
    for p in [root / "final" / "final", root / "final" / "final 2", root / "experiments_raw"]:
        if p.exists():
            folders.append(p)

    files = []
    for folder in folders:
        for p in folder.rglob("summary.json"):
            parts = set(p.parts)
            if "report_results" in parts or "report_results_clean" in parts:
                continue
            if "__pycache__" in parts or ".git" in parts:
                continue
            if (p.parent / "metrics.csv").exists():
                files.append(p)
    return sorted(files)


def collect_runs():
    rows = []
    for p in scan_summary_files():
        summary = read_json(p)
        config = read_json(p.parent / "config.json")
        mpath = p.parent / "metrics.csv"
        mst = metrics_stats(mpath)

        task = get_value(summary, config, "task_name", "")
        dtype = norm_dtype(get_value(summary, config, "dtype", get_value(summary, config, "dtype_name", "")))
        seed = to_num(get_value(summary, config, "seed", np.nan))
        param_name, param_value = main_parameter(summary, config)
        run_name = p.parent.name
        variant = get_value(summary, config, "variant", infer_variant(run_name, task))

        best = to_num(get_value(summary, config, "best_l2_error", mst["best_l2_error"]))
        final = to_num(get_value(summary, config, "final_l2_error", mst["final_l2_error"]))
        if not finite(best):
            best = mst["best_l2_error"]
        if not finite(final):
            final = mst["final_l2_error"]

        row = {
            "source_path": rel(p.parent),
            "run_name": run_name,
            "task_name": task,
            "dtype": dtype,
            "seed": seed,
            "variant": variant,
            "beta": np.nan,
            "nu": np.nan,
            "m": np.nan,
            "alpha": np.nan,
            "hid_size": to_num(get_value(summary, config, "hid_size", get_value(summary, config, "hidden_dim", np.nan))),
            "num_layers": to_num(get_value(summary, config, "num_layers", np.nan)),
            "n_collocation": to_num(get_value(summary, config, "n_collocation", np.nan)),
            "n_ic": to_num(get_value(summary, config, "n_ic", np.nan)),
            "n_bc": to_num(get_value(summary, config, "n_bc", np.nan)),
            "adam_steps": to_num(get_value(summary, config, "adam_steps", np.nan)),
            "lr_adam": to_num(get_value(summary, config, "lr_adam", np.nan)),
            "lbfgs_steps": to_num(get_value(summary, config, "lbfgs_steps", np.nan)),
            "lbfgs_lr": to_num(get_value(summary, config, "lbfgs_lr", np.nan)),
            "lbfgs_max_iter": to_num(get_value(summary, config, "lbfgs_max_iter", np.nan)),
            "lbfgs_tolerance_grad": to_num(get_value(summary, config, "lbfgs_tolerance_grad", np.nan)),
            "lbfgs_tolerance_change": to_num(get_value(summary, config, "lbfgs_tolerance_change", np.nan)),
            "lbfgs_line_search_fn": get_value(summary, config, "lbfgs_line_search_fn", ""),
            "resample_every": to_num(get_value(summary, config, "resample_every", 0)),
            "best_l2_error": best,
            "final_l2_error": final,
            "best_step": to_num(get_value(summary, config, "best_step", mst["best_step"])),
            "final_step": to_num(get_value(summary, config, "final_step", mst["final_step"])),
            "elapsed_time": to_num(get_value(summary, config, "elapsed_time", get_value(summary, config, "time_sec", mst["elapsed_time"]))),
            "error": str(get_value(summary, config, "error", get_value(summary, config, "exception", ""))),
            "metrics_path": rel(mpath),
            "summary_path": rel(p),
            "main_parameter_name": param_name,
            "main_parameter_value": param_value,
            "source_rank": source_rank(p),
            "missing_metrics": bool(mst["missing_metrics"]),
            "metrics_has_l2": bool(mst["metrics_has_l2"]),
        }
        if param_name == "beta":
            row["beta"] = param_value
        elif param_name == "nu":
            row["nu"] = param_value
        elif param_name == "m":
            row["m"] = param_value
        elif param_name == "alpha":
            row["alpha"] = param_value
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    def key_float(x):
        if finite(x):
            return round(float(x), 12)
        return None

    df["dedupe_key"] = df.apply(
        lambda r: (
            r["task_name"], r["dtype"], r["seed"], r["variant"],
            r["main_parameter_name"], key_float(r["main_parameter_value"]),
            key_float(r["best_l2_error"]), key_float(r["final_l2_error"]),
        ),
        axis=1,
    )
    df = df.sort_values(["source_rank", "source_path"]).drop_duplicates("dedupe_key", keep="first")
    df = df.drop(columns=["dedupe_key", "source_rank"])
    df = df.sort_values(["task_name", "main_parameter_value", "variant", "dtype", "seed", "source_path"])
    return df.reset_index(drop=True)


def mark_quality(df):
    if len(df) == 0:
        return df
    reasons = []
    valid = []
    bad = []
    labels = []
    for _, r in df.iterrows():
        err = str(r.get("error", ""))
        best = r.get("best_l2_error", np.nan)
        final = r.get("final_l2_error", np.nan)
        missing = bool(r.get("missing_metrics", False)) or not bool(r.get("metrics_has_l2", False))

        if err and err.lower() not in ["nan", "none"]:
            reason = "runtime_error"
        elif missing:
            reason = "missing_metrics"
        elif pd.isna(best) or pd.isna(final):
            reason = "nan_metric"
        elif not finite(best) or not finite(final):
            reason = "inf_metric"
        elif best > 0.2:
            reason = "high_best_l2"
        elif final > 0.5:
            reason = "high_final_l2"
        else:
            reason = "ok"

        is_valid = reason not in ["runtime_error", "missing_metrics", "nan_metric", "inf_metric"]
        is_bad = reason != "ok"

        if not is_valid or is_bad:
            label = "bad"
        elif best <= 0.02 and final <= 0.05:
            label = "good"
        elif best <= 0.1 and final <= 0.2:
            label = "acceptable"
        elif best <= 0.1 and final > 0.2:
            label = "unstable"
        else:
            label = "bad"

        reasons.append(reason)
        valid.append(is_valid)
        bad.append(is_bad)
        labels.append(label)

    df = df.copy()
    df["is_valid"] = valid
    df["is_bad"] = bad
    df["bad_reason"] = reasons
    df["quality_label"] = labels
    return df


def seed_list(s):
    vals = []
    for x in s.dropna().unique():
        if finite(x):
            vals.append(int(x))
    return ",".join(str(x) for x in sorted(vals))


def grouped_by_dtype(runs):
    keys = ["task_name", "variant", "main_parameter_name", "main_parameter_value", "dtype"]
    all_part = runs.groupby(keys, dropna=False).agg(
        n_total=("run_name", "count"),
        n_valid=("is_valid", "sum"),
        n_bad=("is_bad", "sum"),
        seed_list=("seed", seed_list),
    ).reset_index()
    good = runs[runs["is_valid"]].copy()
    val_part = good.groupby(keys, dropna=False).agg(
        best_l2_mean=("best_l2_error", "mean"),
        best_l2_median=("best_l2_error", "median"),
        best_l2_std=("best_l2_error", "std"),
        best_l2_min=("best_l2_error", "min"),
        best_l2_max=("best_l2_error", "max"),
        final_l2_mean=("final_l2_error", "mean"),
        final_l2_median=("final_l2_error", "median"),
        final_l2_std=("final_l2_error", "std"),
        final_l2_min=("final_l2_error", "min"),
        final_l2_max=("final_l2_error", "max"),
    ).reset_index()
    grouped = all_part.merge(val_part, on=keys, how="left")
    grouped["bad_rate"] = grouped["n_bad"] / grouped["n_total"]
    return grouped


compare_cols = [
    "task_name", "variant", "main_parameter_name", "main_parameter_value",
    "hid_size", "num_layers", "n_collocation", "n_ic", "n_bc",
    "adam_steps", "lr_adam", "lbfgs_steps", "lbfgs_lr", "lbfgs_max_iter",
    "resample_every",
]


def spread_large(vals):
    vals = [float(x) for x in vals if finite(x) and float(x) > 0]
    if len(vals) < 2:
        return False
    return max(vals) / min(vals) > 20


def comparison_table(runs):
    rows = []
    src = runs[runs["dtype"].isin(["fp32", "fp64"])].copy()
    for key, tmp in src.groupby(compare_cols, dropna=False):
        row = dict(zip(compare_cols, key))
        row["case_key"] = "|".join(str(x) for x in key)
        for dtype in ["fp32", "fp64"]:
            cur = tmp[tmp["dtype"] == dtype]
            val = cur[cur["is_valid"]]
            row[f"{dtype}_n_total"] = len(cur)
            row[f"{dtype}_n_valid"] = len(val)
            row[f"{dtype}_seed_list"] = seed_list(val["seed"])
            row[f"{dtype}_bad_rate"] = float(cur["is_bad"].sum() / len(cur)) if len(cur) else np.nan
            for stat, func in [
                ("median", "median"),
                ("mean", "mean"),
                ("min", "min"),
                ("max", "max"),
            ]:
                col = f"{dtype}_{stat}_best_l2"
                if len(val):
                    row[col] = getattr(val["best_l2_error"], func)()
                else:
                    row[col] = np.nan

        if finite(row["fp32_median_best_l2"]) and finite(row["fp64_median_best_l2"]) and row["fp32_median_best_l2"] > 0:
            row["fp64_over_fp32_median"] = row["fp64_median_best_l2"] / row["fp32_median_best_l2"]
        else:
            row["fp64_over_fp32_median"] = np.nan
        if finite(row["fp32_mean_best_l2"]) and finite(row["fp64_mean_best_l2"]) and row["fp32_mean_best_l2"] > 0:
            row["fp64_over_fp32_mean"] = row["fp64_mean_best_l2"] / row["fp32_mean_best_l2"]
        else:
            row["fp64_over_fp32_mean"] = np.nan

        fp32_valid = tmp[(tmp["dtype"] == "fp32") & (tmp["is_valid"])]
        fp64_valid = tmp[(tmp["dtype"] == "fp64") & (tmp["is_valid"])]
        s32 = set(int(x) for x in fp32_valid["seed"].dropna())
        s64 = set(int(x) for x in fp64_valid["seed"].dropna())
        common = sorted(s32 & s64)
        row["common_seed_list"] = ",".join(str(x) for x in common)
        wins = []
        for seed in common:
            a = fp32_valid[fp32_valid["seed"] == seed]["best_l2_error"].median()
            b = fp64_valid[fp64_valid["seed"] == seed]["best_l2_error"].median()
            if finite(a) and finite(b):
                wins.append(float(b) < float(a))
        row["seed_win_rate_fp64"] = float(np.mean(wins)) if wins else np.nan

        ratio = row["fp64_over_fp32_median"]
        seed_sensitive = False
        seed_sensitive = seed_sensitive or spread_large(fp32_valid["best_l2_error"])
        seed_sensitive = seed_sensitive or spread_large(fp64_valid["best_l2_error"])
        seed_sensitive = seed_sensitive or row["fp32_bad_rate"] >= 0.5
        seed_sensitive = seed_sensitive or row["fp64_bad_rate"] >= 0.5
        seed_sensitive = seed_sensitive or row["fp32_n_valid"] < 2
        seed_sensitive = seed_sensitive or row["fp64_n_valid"] < 2
        if finite(ratio) and ratio <= 0.5 and finite(row["seed_win_rate_fp64"]) and row["seed_win_rate_fp64"] < 0.5:
            seed_sensitive = True

        if row["fp32_n_valid"] == 0 or row["fp64_n_valid"] == 0:
            conclusion = "insufficient_data"
        elif row["fp32_n_valid"] == 1 and row["fp64_n_valid"] == 1 and finite(ratio) and ratio <= 0.1 and row["fp32_bad_rate"] > row["fp64_bad_rate"]:
            conclusion = "single_seed_hard_case"
        elif seed_sensitive:
            conclusion = "seed_sensitive"
        elif row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2 and finite(ratio) and ratio <= 0.5 and row["fp64_bad_rate"] <= row["fp32_bad_rate"] + 0.25:
            win_ok = pd.isna(row["seed_win_rate_fp64"]) or row["seed_win_rate_fp64"] >= 0.5
            conclusion = "stable_fp64_better" if win_ok else "seed_sensitive"
        elif row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2 and finite(ratio) and 0.5 < ratio <= 0.8:
            conclusion = "moderate_fp64_better"
        elif row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2 and finite(ratio) and 0.8 < ratio < 1.25:
            conclusion = "similar"
        elif row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2 and finite(ratio) and ratio >= 1.25:
            conclusion = "fp32_better"
        else:
            conclusion = "insufficient_data"
        row["conclusion"] = conclusion
        rows.append(row)
    return pd.DataFrame(rows)


def fp16_table(runs):
    src = runs[runs["dtype"] == "fp16"].copy()
    if len(src) == 0:
        return pd.DataFrame(columns=["case_key", "n_total", "n_valid", "n_bad", "bad_rate", "best_l2_median", "final_l2_median", "typical_failure_reason"])
    rows = []
    for key, tmp in src.groupby(compare_cols, dropna=False):
        valid = tmp[tmp["is_valid"]]
        reason = ""
        if len(tmp):
            mode = tmp["bad_reason"].mode()
            reason = mode.iloc[0] if len(mode) else ""
        row = dict(zip(compare_cols, key))
        row["case_key"] = "|".join(str(x) for x in key)
        row["n_total"] = len(tmp)
        row["n_valid"] = len(valid)
        row["n_bad"] = int(tmp["is_bad"].sum())
        row["bad_rate"] = row["n_bad"] / row["n_total"]
        row["best_l2_median"] = valid["best_l2_error"].median() if len(valid) else np.nan
        row["final_l2_median"] = valid["final_l2_error"].median() if len(valid) else np.nan
        row["typical_failure_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_value(x):
    if not finite(x):
        return "unknown"
    v = float(x)
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    return f"{v:g}"


def case_title(row):
    task = row["task_name"]
    name = row["main_parameter_name"]
    value = fmt_value(row["main_parameter_value"])
    if task == "helmholtz1d":
        return f"Helmholtz, {name}={value}"
    if task == "burgers1d":
        return f"Burgers, {name}={value}"
    if task == "convection1d":
        return f"Convection, {name}={value}"
    if task == "heat1d":
        return f"Heat, {name}={value}"
    return f"{task}, {name}={value}"


def base_case_id(row):
    task = row["task_name"]
    value = fmt_value(row["main_parameter_value"]).replace(".", "p").replace("-", "m")
    if task == "helmholtz1d":
        return f"helmholtz_m{value}"
    if task == "burgers1d":
        return f"burgers_nu{value.replace('0p', '0')}"
    if task == "convection1d":
        return f"convection_beta{value}"
    if task == "heat1d":
        return f"heat_alpha{value}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", f"{task}_{value}").strip("_")


def add_selected(selected, row, why):
    if row is None or len(row) == 0:
        return
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    title = case_title(row)
    for old in selected:
        if old["task_name"] == row["task_name"] and old["main_parameter_name"] == row["main_parameter_name"] and old["main_parameter_value"] == row["main_parameter_value"]:
            return
    conf = "needs_check"
    if row["fp32_n_valid"] >= 2 and row["fp64_n_valid"] >= 2:
        if row["conclusion"] in ["stable_fp64_better", "similar", "fp32_better"]:
            conf = "strong"
        else:
            conf = "medium"
    selected.append({
        "case_id": base_case_id(row),
        "task_name": row["task_name"],
        "main_parameter_name": row["main_parameter_name"],
        "main_parameter_value": row["main_parameter_value"],
        "parameter": f"{row['main_parameter_name']}={fmt_value(row['main_parameter_value'])}",
        "variant": row["variant"],
        "n_fp32": row["fp32_n_valid"],
        "n_fp64": row["fp64_n_valid"],
        "fp32_median_best_l2": row["fp32_median_best_l2"],
        "fp64_median_best_l2": row["fp64_median_best_l2"],
        "ratio_median": row["fp64_over_fp32_median"],
        "fp32_bad_rate": row["fp32_bad_rate"],
        "fp64_bad_rate": row["fp64_bad_rate"],
        "conclusion": row["conclusion"],
        "confidence_label": conf,
        "why_selected": why,
        "source_paths": "",
        "case_title": title,
        "case_key": row["case_key"],
    })


def pick_report_cases(comp, runs, fp16):
    selected = []

    stable = comp[comp["conclusion"] == "stable_fp64_better"].copy()
    m12 = stable[(stable["task_name"] == "helmholtz1d") & (stable["main_parameter_name"] == "m") & (stable["main_parameter_value"] == 12)]
    if len(m12):
        add_selected(selected, m12.sort_values("fp64_over_fp32_median"), "основной устойчивый пример на Helmholtz")
    elif len(stable):
        add_selected(selected, stable.sort_values("fp64_over_fp32_median"), "stable fp64 better case")

    hard = comp[
        (comp["task_name"] == "convection1d")
        & (comp["main_parameter_name"] == "beta")
        & (comp["main_parameter_value"] == 50)
        & (comp["conclusion"].isin(["single_seed_hard_case", "seed_sensitive"]))
    ].copy()
    if len(hard):
        hard["rank"] = hard["conclusion"].eq("single_seed_hard_case").astype(int)
        add_selected(selected, hard.sort_values(["rank", "fp64_median_best_l2"], ascending=[False, True]), "сложный режим: FP64 выглядит лучше, но нужны дополнительные seed")

    similar = comp[
        (comp["task_name"] == "burgers1d")
        & (comp["main_parameter_name"] == "nu")
        & (comp["main_parameter_value"].round(6) == 0.002)
        & (comp["conclusion"] == "similar")
    ].copy()
    if len(similar) == 0:
        similar = comp[comp["conclusion"] == "similar"].copy()
    if len(similar):
        similar["dist"] = (similar["fp64_over_fp32_median"] - 1).abs()
        add_selected(selected, similar.sort_values("dist"), "контрольный пример: FP64 близок к FP32")

    neg = comp[
        (comp["task_name"] == "burgers1d")
        & (comp["main_parameter_name"] == "nu")
        & (comp["main_parameter_value"].round(6) == 0.001)
        & (comp["conclusion"] == "fp32_better")
    ].copy()
    if len(neg) == 0:
        neg = comp[comp["conclusion"] == "fp32_better"].copy()
    if len(neg):
        add_selected(selected, neg.sort_values("fp64_over_fp32_median"), "пример, где FP64 не дал преимущества")

    m8 = comp[
        (comp["task_name"] == "helmholtz1d")
        & (comp["main_parameter_name"] == "m")
        & (comp["main_parameter_value"] == 8)
        & (comp["conclusion"].isin(["seed_sensitive", "stable_fp64_better", "moderate_fp64_better"]))
    ].copy()
    if len(m8) and len(selected) < 5:
        add_selected(selected, m8.sort_values("fp64_over_fp32_median"), "дополнительный Helmholtz-кейс, но seed дают смешанную картину")

    if len(selected) < 5:
        more = stable.sort_values("fp64_over_fp32_median")
        for _, row in more.iterrows():
            if len(selected) >= 5:
                break
            add_selected(selected, row, "дополнительный устойчивый пример в пользу FP64")

    selected = selected[:5]

    if len(fp16):
        total = int(fp16["n_total"].sum())
        bad = int(fp16["n_bad"].sum())
        selected.append({
            "case_id": "fp16_summary",
            "task_name": "fp16",
            "main_parameter_name": "summary",
            "main_parameter_value": 0.0,
            "parameter": "все fp16-запуски",
            "variant": "отдельно",
            "n_fp32": np.nan,
            "n_fp64": np.nan,
            "fp32_median_best_l2": np.nan,
            "fp64_median_best_l2": np.nan,
            "ratio_median": np.nan,
            "fp32_bad_rate": np.nan,
            "fp64_bad_rate": np.nan,
            "conclusion": "fp16 failed or unstable",
            "confidence_label": "medium" if total else "needs_check",
            "why_selected": f"FP16 вынесен отдельно: {bad}/{total} запусков плохие или невалидные",
            "source_paths": "",
            "case_title": "FP16",
            "case_key": "",
        })

    selected = pd.DataFrame(selected)
    if len(selected):
        used = {}
        ids = []
        for x in selected["case_id"]:
            n = used.get(x, 0)
            used[x] = n + 1
            ids.append(x if n == 0 else f"{x}_{n + 1}")
        selected["case_id"] = ids

    for i, row in selected.iterrows():
        if row["task_name"] == "fp16":
            continue
        cur = runs[
            (runs["task_name"] == row["task_name"])
            & (runs["variant"] == row["variant"])
            & (runs["main_parameter_name"] + "=" + runs["main_parameter_value"].map(fmt_value) == row["parameter"])
            & (runs["dtype"].isin(["fp32", "fp64"]))
        ]
        selected.loc[i, "source_paths"] = "; ".join(cur["source_path"].drop_duplicates().head(20))
    return selected


def selected_runs_for_case(runs, row):
    if row["task_name"] == "fp16":
        return runs.iloc[0:0].copy()
    name, val = row["parameter"].split("=", 1)
    return runs[
        (runs["task_name"] == row["task_name"])
        & (runs["variant"] == row["variant"])
        & (runs["main_parameter_name"] == name)
        & (runs["main_parameter_value"].map(fmt_value) == val)
        & (runs["dtype"].isin(["fp32", "fp64", "fp16"]))
    ].copy()


def make_main_figures(selected, runs):
    sel = selected[selected["task_name"] != "fp16"].copy()
    if len(sel) == 0:
        return
    labels = sel["case_id"].tolist()
    x = np.arange(len(sel))

    fig, ax = plt.subplots(figsize=(max(8, len(sel) * 1.3), 4.5))
    ax.bar(x - 0.18, sel["fp32_median_best_l2"], width=0.36, label="FP32")
    ax.bar(x + 0.18, sel["fp64_median_best_l2"], width=0.36, label="FP64")
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "report_best_l2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(sel) * 1.3), 4))
    ax.bar(x, sel["ratio_median"])
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("FP64 / FP32 median best L2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "report_ratio.png", dpi=180)
    plt.close(fig)

    parts = []
    for _, row in sel.iterrows():
        cur = selected_runs_for_case(runs, row)
        cur = cur[cur["dtype"].isin(["fp32", "fp64"])]
        cur["case_id"] = row["case_id"]
        parts.append(cur)
    if parts:
        sc = pd.concat(parts, ignore_index=True)
        fig, ax = plt.subplots(figsize=(max(8, len(sel) * 1.3), 4.5))
        colors = {"fp32": "#4c78a8", "fp64": "#f58518"}
        pos = {v: i for i, v in enumerate(labels)}
        for dtype in ["fp32", "fp64"]:
            tmp = sc[sc["dtype"] == dtype]
            xs = [pos[v] + (-0.12 if dtype == "fp32" else 0.12) for v in tmp["case_id"]]
            ax.scatter(xs, tmp["best_l2_error"], label=dtype.upper(), color=colors[dtype], s=48)
        ax.set_yscale("log")
        ax.set_ylabel("relative L2 error")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "report_seed_scatter.png", dpi=180)
        plt.close(fig)


def plot_curves(selected, runs):
    for _, row in selected.iterrows():
        if row["task_name"] == "fp16":
            continue
        cur = selected_runs_for_case(runs, row)
        cur = cur[cur["metrics_path"].fillna("") != ""]
        if len(cur) == 0:
            continue

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ok = 0
        for _, run in cur.sort_values(["dtype", "seed"]).iterrows():
            p = root / str(run["metrics_path"])
            try:
                h = pd.read_csv(p)
            except Exception:
                continue
            if "step" not in h.columns:
                continue
            label = f"{str(run['dtype']).upper()} seed={int(run['seed']) if finite(run['seed']) else '?'}"
            loss_col = "total_loss" if "total_loss" in h.columns else None
            if loss_col is None:
                for c in h.columns:
                    if "loss" in c:
                        loss_col = c
                        break
            if loss_col is not None:
                y = pd.to_numeric(h[loss_col], errors="coerce")
                m = np.isfinite(y) & (y > 0)
                ax[0].plot(h.loc[m, "step"], y[m], label=label)
            if "l2_error" in h.columns:
                y = pd.to_numeric(h["l2_error"], errors="coerce")
                m = np.isfinite(y) & (y > 0)
                ax[1].plot(h.loc[m, "step"], y[m], label=label)
            ok += 1
        if ok == 0:
            plt.close(fig)
            continue
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
        fig.suptitle(row["case_title"])
        fig.tight_layout()
        fig.savefig(fig_dir / f"curves_{row['case_id']}.png", dpi=180)
        plt.close(fig)


def fp16_figure(fp16):
    if len(fp16) == 0:
        return
    tmp = fp16.copy().head(20)
    tmp["label"] = tmp["task_name"] + " " + tmp["main_parameter_name"].astype(str) + "=" + tmp["main_parameter_value"].map(fmt_value)
    fig, ax = plt.subplots(figsize=(max(8, len(tmp) * 0.7), 4))
    ax.bar(range(len(tmp)), tmp["bad_rate"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("bad run rate")
    ax.set_xticks(range(len(tmp)))
    ax.set_xticklabels(tmp["label"], rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fp16_summary.png", dpi=180)
    plt.close(fig)


def write_selected_notes(selected):
    lines = ["# Выбранные кейсы", ""]
    for _, row in selected.iterrows():
        lines.append(f"## {row['case_id']}")
        lines.append("")
        lines.append(f"- задача: {row['task_name']}")
        lines.append(f"- параметр: {row['parameter']}")
        lines.append(f"- вариант: `{row['variant']}`")
        lines.append(f"- вывод: {ru_conclusion(row['conclusion'])}")
        lines.append(f"- надёжность: {ru_confidence(row['confidence_label'])}")
        lines.append(f"- почему выбран: {row['why_selected']}")
        if str(row.get("source_paths", "")).strip():
            lines.append("- исходные run-папки:")
            for p in str(row["source_paths"]).split("; "):
                if p:
                    lines.append(f"  - `{p}`")
        lines.append("")
    write_text(notes_dir / "report_cases.md", lines)


def save_selected_runs(selected, runs):
    parts = []
    for _, row in selected.iterrows():
        if row["task_name"] == "fp16":
            continue
        cur = selected_runs_for_case(runs, row)
        cur = cur.copy()
        cur["case_id"] = row["case_id"]
        parts.append(cur)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(selected_dir / "selected_run_paths.csv", index=False)


def write_rerun_script():
    text = r'''
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))
import pinn_model


out_root = root / "results_selected_checks"
report_fig_dir = root / "report_results_clean" / "figures"
out_root.mkdir(exist_ok=True)
report_fig_dir.mkdir(parents=True, exist_ok=True)


def save_json(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def load_model(config, log_dir, device):
    dtype = pinn_model.get_dtype(config["dtype"])
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    model = pinn_model.PINN(input_dim=2, hid_size=config["hid_size"], num_layers=config["num_layers"]).to(device)
    p = log_dir / "model.pt"
    if not p.exists():
        return None
    state = torch.load(p, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def dense_convection(model, beta, device):
    n = 240
    x = torch.linspace(0, 2 * torch.pi, n, device=device).reshape(-1, 1)
    t = torch.linspace(0, 1, n, device=device).reshape(-1, 1)
    xx, tt = torch.meshgrid(x.reshape(-1), t.reshape(-1), indexing="ij")
    xv = xx.reshape(-1, 1)
    tv = tt.reshape(-1, 1)
    with torch.no_grad():
        pred = model(xv, tv)
        true = pinn_model.convection_1d_solution(xv, tv, beta)
    err = pred - true
    mae = torch.mean(torch.abs(err))
    rmse = torch.sqrt(torch.mean(err ** 2))
    rel_mae = mae / torch.mean(torch.abs(true))
    rel_rmse = rmse / torch.sqrt(torch.mean(true ** 2))
    rel_l2 = torch.sqrt(torch.sum(err ** 2)) / torch.sqrt(torch.sum(true ** 2))
    extra = {
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "relative_mae": float(rel_mae.detach().cpu()),
        "relative_rmse": float(rel_rmse.detach().cpu()),
        "relative_l2": float(rel_l2.detach().cpu()),
    }
    data = {
        "x": x.detach().cpu().numpy().reshape(-1),
        "t": t.detach().cpu().numpy().reshape(-1),
        "exact": true.detach().cpu().numpy().reshape(n, n),
        "pred": pred.detach().cpu().numpy().reshape(n, n),
        "abs_error": torch.abs(err).detach().cpu().numpy().reshape(n, n),
    }
    return extra, data


def save_maps(log_dir, data, title):
    extent = [data["t"].min(), data["t"].max(), data["x"].min(), data["x"].max()]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    im = ax[0].imshow(data["exact"], origin="lower", aspect="auto", extent=extent)
    ax[0].set_title("exact solution")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(data["pred"], origin="lower", aspect="auto", extent=extent)
    ax[1].set_title("prediction")
    ax[1].set_xlabel("t")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(data["abs_error"], origin="lower", aspect="auto", extent=extent)
    ax[2].set_title("absolute error")
    ax[2].set_xlabel("t")
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(log_dir / "solution_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(data["abs_error"], origin="lower", aspect="auto", extent=extent)
    ax.set_title("absolute error")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(log_dir / "error_map.png", dpi=180)
    plt.close(fig)
    np.savez(log_dir / "map_data.npz", **data)


def common_map(run_dirs):
    items = []
    exact = None
    extent = None
    for dtype, p in run_dirs:
        data_p = p / "map_data.npz"
        if not data_p.exists():
            continue
        data = np.load(data_p)
        if exact is None:
            exact = data["exact"]
            extent = [data["t"].min(), data["t"].max(), data["x"].min(), data["x"].max()]
        items.append((dtype, data["pred"], data["abs_error"]))
    if exact is None or len(items) == 0:
        return
    cols = 1 + 2 * len(items)
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        ax = [ax]
    im = ax[0].imshow(exact, origin="lower", aspect="auto", extent=extent)
    ax[0].set_title("exact solution")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    j = 1
    for dtype, pred, err in items:
        im = ax[j].imshow(pred, origin="lower", aspect="auto", extent=extent)
        ax[j].set_title(dtype.upper() + " prediction")
        ax[j].set_xlabel("t")
        fig.colorbar(im, ax=ax[j], fraction=0.046)
        j += 1
        im = ax[j].imshow(err, origin="lower", aspect="auto", extent=extent)
        ax[j].set_title(dtype.upper() + " absolute error")
        ax[j].set_xlabel("t")
        fig.colorbar(im, ax=ax[j], fraction=0.046)
        j += 1
    fig.tight_layout()
    fig.savefig(report_fig_dir / "convection_beta50_maps.png", dpi=180)
    plt.close(fig)


def run_one(config):
    log_dir = Path(config["log_dir"])
    if (log_dir / "summary.json").exists() and (log_dir / "metrics.csv").exists():
        return log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        pinn_model.run_experiment(config)
    except Exception as e:
        save_json(log_dir / "summary.json", {**config, "error": repr(e)})
    return log_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = {
        "task_name": "convection1d",
        "beta": 50.0,
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
    }
    jobs = []
    for seed in [1, 2]:
        for dtype in ["fp32", "fp64"]:
            jobs.append((dtype, seed))
    if device != "cpu":
        jobs.append(("fp16", 0))

    run_dirs = []
    for dtype, seed in jobs:
        cfg = dict(base)
        cfg["dtype"] = dtype
        cfg["seed"] = seed
        name = f"convection_beta50_{dtype}_seed{seed}"
        cfg["log_dir"] = str(out_root / name)
        log_dir = run_one(cfg)
        model = load_model(cfg, log_dir, torch.device(device))
        if model is None:
            continue
        extra, data = dense_convection(model, 50.0, torch.device(device))
        save_json(log_dir / "metrics_extra.json", extra)
        save_maps(log_dir, data, f"Convection beta=50, {dtype.upper()} seed={seed}")
        run_dirs.append((dtype, log_dir))
    common_map(run_dirs)


if __name__ == "__main__":
    main()
'''.strip()
    write_text(rerun_dir / "run_selected_checks.py", text.splitlines())


def write_missing(selected):
    lines = ["# Чего не хватает", ""]
    lines.append("Большой перезапуск всех экспериментов не нужен.")
    lines.append("")
    strong = selected[selected["confidence_label"] == "strong"]
    weak = selected[selected["confidence_label"] == "needs_check"]
    lines.append("## Уже достаточно надёжные кейсы")
    if len(strong):
        for _, row in strong.iterrows():
            lines.append(f"- `{row['case_id']}`: {ru_conclusion(row['conclusion'])}")
    else:
        lines.append("- таких кейсов не найдено")
    lines.append("")
    lines.append("## Кейсы, где нужен осторожный текст")
    if len(weak):
        for _, row in weak.iterrows():
            lines.append(f"- `{row['case_id']}`: {ru_conclusion(row['conclusion'])}; нужны дополнительные seed, чтобы называть результат устойчивым")
    else:
        lines.append("- таких кейсов не найдено")
    lines.append("")
    lines.append("## Желательно добавить")
    lines.append("- MAE/RMSE для convection beta=50 на плотной сетке")
    lines.append("- карты exact / prediction / error для convection beta=50")
    lines.append("")
    lines.append("## Минимальные дозапуски")
    lines.append("Запускать только если нужны карты или дополнительная проверка:")
    lines.append("`python report_results_clean/rerun_plan/run_selected_checks.py`")
    lines.append("")
    lines.append("Запланированные проверки:")
    lines.append("- convection_beta50_fp32_seed1")
    lines.append("- convection_beta50_fp64_seed1")
    lines.append("- convection_beta50_fp32_seed2")
    lines.append("- convection_beta50_fp64_seed2")
    lines.append("- convection_beta50_fp16_seed0, на CPU пропускается")
    write_text(rerun_dir / "missing_artifacts.md", lines)


def write_readme(runs, selected, fp16):
    invalid = int((~runs["is_valid"]).sum())
    bad = int(runs["is_bad"].sum())
    valid = int(runs["is_valid"].sum())
    lines = [
        "# Чистая сводка результатов",
        "",
        "Эта папка пересобирается только из настоящих run-папок с `summary.json` и `metrics.csv`.",
        "Старая папка `report_results/` не используется как источник данных.",
        "",
        "Просканированные источники:",
        "- `results_exp_*`",
        "- `final/final/`",
        "- `final/final 2/`",
        "- `experiments_raw/`, если там есть распакованные run-папки",
        "",
        f"Уникальных запусков найдено: {len(runs)}.",
        f"Валидных запусков: {valid}.",
        f"Невалидных запусков: {invalid}.",
        f"Плохих по выбранному порогу: {bad}.",
        "",
        "`bad_runs.csv` включает невалидные запуски и валидные запуски с большой ошибкой.",
        "Поэтому числа valid и bad могут пересекаться.",
        "",
        "## Выбранные кейсы",
    ]
    for _, row in selected.iterrows():
        lines.append(f"- `{row['case_id']}`: {ru_conclusion(row['conclusion'])} ({ru_confidence(row['confidence_label'])})")
    lines.extend([
        "",
        "## FP16",
        f"FP16-групп в сводке: {len(fp16)}. FP16 вынесен отдельно и не смешивается с основной таблицей FP32/FP64.",
        "",
        "## Главные таблицы",
        "- `tables/all_runs.csv`",
        "- `tables/run_quality.csv`",
        "- `tables/grouped_by_dtype.csv`",
        "- `tables/fp32_fp64_comparison.csv`",
        "- `tables/fp16_summary.csv`",
        "- `tables/report_cases.csv`",
        "",
        "## Главные графики",
        "- `figures/report_best_l2.png`",
        "- `figures/report_ratio.png`",
        "- `figures/report_seed_scatter.png`",
        "- `figures/curves_<case_id>.png`",
        "",
        "Главный вывод должен оставаться аккуратным: FP64 помогает в некоторых сложных режимах, но не всегда лучше FP32.",
    ])
    write_text(out_dir / "README.md", lines)


def clean_old_outputs():
    for folder in [table_dir, fig_dir, notes_dir, selected_dir, rerun_dir]:
        for p in folder.glob("*"):
            if p.is_file():
                p.unlink()


def main():
    clean_old_outputs()
    runs = collect_runs()
    runs = mark_quality(runs)
    runs.to_csv(table_dir / "all_runs.csv", index=False)
    runs.to_csv(table_dir / "run_quality.csv", index=False)
    runs[runs["is_valid"]].to_csv(table_dir / "valid_runs.csv", index=False)
    runs[runs["is_bad"]].to_csv(table_dir / "bad_runs.csv", index=False)

    grouped = grouped_by_dtype(runs)
    grouped.to_csv(table_dir / "grouped_by_dtype.csv", index=False)

    comp = comparison_table(runs)
    comp.to_csv(table_dir / "fp32_fp64_comparison.csv", index=False)
    comp[comp["conclusion"] == "stable_fp64_better"].to_csv(table_dir / "stable_fp64_better_cases.csv", index=False)
    comp[comp["conclusion"] == "single_seed_hard_case"].to_csv(table_dir / "single_seed_hard_cases.csv", index=False)
    comp[comp["conclusion"] == "seed_sensitive"].to_csv(table_dir / "seed_sensitive_cases.csv", index=False)
    comp[comp["conclusion"].isin(["similar", "fp32_better"])].to_csv(table_dir / "similar_or_negative_cases.csv", index=False)

    fp16 = fp16_table(runs)
    fp16.to_csv(table_dir / "fp16_summary.csv", index=False)

    selected = pick_report_cases(comp, runs, fp16)
    selected.to_csv(table_dir / "report_cases.csv", index=False)
    write_selected_notes(selected)
    save_selected_runs(selected, runs)

    make_main_figures(selected, runs)
    plot_curves(selected, runs)
    fp16_figure(fp16)
    write_missing(selected)
    write_rerun_script()
    write_readme(runs, selected, fp16)

    print(f"runs: {len(runs)}")
    print(f"valid: {int(runs['is_valid'].sum())}")
    print(f"bad: {int(runs['is_bad'].sum())}")
    print(f"comparison rows: {len(comp)}")
    print(f"selected cases: {len(selected)}")


if __name__ == "__main__":
    main()
