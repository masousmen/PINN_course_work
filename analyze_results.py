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
    return f"n={n}, медиана best L2={fmt_num(med)}, доля плохих={fmt_num(bad)}"


def public_conclusion(x):
    if x in ["single_seed_hard_case", "seed_sensitive"]:
        return "unstable_or_seed_sensitive"
    return x


def russian_conclusion(x):
    names = {
        "stable_fp64_better": "FP64 заметно лучше",
        "moderate_fp64_better": "FP64 немного лучше",
        "similar": "FP32 и FP64 близки",
        "fp32_better": "FP32 лучше",
        "unstable_or_seed_sensitive": "зависит от seed",
        "fp16 failed or unstable": "FP16 нестабилен",
    }
    return names.get(str(x), str(x))


def russian_confidence(x):
    names = {
        "strong": "сильная",
        "medium": "средняя",
        "weak_needs_rerun": "нужна проверка",
    }
    return names.get(str(x), str(x))


def russian_why(row):
    task = str(row.get("task_name", ""))
    parameter = str(row.get("parameter", ""))
    conclusion = str(row.get("conclusion", ""))
    if task == "helmholtz1d" and parameter == "m=12":
        return "Основной положительный пример: на этом режиме FP64 дал меньшую ошибку по медиане."
    if task == "convection1d" and parameter == "beta=50":
        return "Сложный режим: FP64 выглядит сильно лучше, но запусков мало, поэтому вывод осторожный."
    if task == "burgers1d" and parameter == "nu=0.002":
        return "Контрольный пример: FP32 и FP64 дали близкие ошибки."
    if task == "burgers1d" and parameter == "nu=0.001":
        return "Пример, где FP64 не дал преимущества."
    if task == "helmholtz1d" and parameter == "m=8":
        return "Пример с заметной зависимостью от seed."
    if task == "fp16":
        return "FP16 вынесен отдельно, потому что в этих запусках он часто давал плохие или невалидные метрики."
    return conclusion


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
            "why_selected": russian_why(row),
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
    write_selected_markdown(selected)


def write_selected_markdown(selected):
    lines = ["# Выбранные кейсы", ""]
    for _, row in selected.iterrows():
        lines.append(f"## {row['case_title']}")
        lines.append("")
        lines.append(f"- задача: `{row['task_name']}`")
        lines.append(f"- параметр: `{row['parameter']}`")
        lines.append(f"- вариант: `{row['variant']}`")
        lines.append(f"- почему выбран: {row['why_selected']}")
        if str(row.get("fp32_result", "")).strip():
            lines.append(f"- FP32: {row['fp32_result']}")
        if str(row.get("fp64_result", "")).strip():
            lines.append(f"- FP64: {row['fp64_result']}")
        lines.append(f"- вывод: {russian_conclusion(row['conclusion'])}")
        lines.append(f"- надёжность: {russian_confidence(row['confidence_label'])}")
        src = str(row.get("source_runs", "")).strip()
        if src and src.lower() != "nan":
            lines.append("- исходные run-папки:")
            for p in src.replace("|", ";").split(";"):
                p = p.strip()
                if p:
                    lines.append(f"  - `{p}`")
        lines.append("")
    (selected_dir / "selected_cases.md").write_text("\n".join(lines) + "\n")


def write_report_readme():
    runs = pd.read_csv(clean_dir / "tables" / "all_runs.csv")
    selected = pd.read_csv(table_dir / "selected_cases.csv")
    fp16 = pd.read_csv(clean_dir / "tables" / "fp16_summary.csv")
    valid = int(runs["is_valid"].sum())
    bad = int(runs["is_bad"].sum())
    invalid = int((~runs["is_valid"]).sum())

    lines = [
        "# Итоговые результаты",
        "",
        "В этой папке лежат таблицы и графики, которые я использую в отчёте.",
        "Сырые запуски не удалялись: они остались в старых папках `results_exp_*` и `final/`.",
        "Для выводов я ориентируюсь не на лучший seed, а на медиану, разброс и пометки о плохих запусках.",
        "",
        "## Что лежит в этой папке",
        "",
        "- `tables/` - таблицы после очистки логов.",
        "- `figures/` - графики для отчёта.",
        "- `selected_runs/` - выбранные запуски и пути к ним.",
        "- `rerun_plan/` - что можно дозапустить, если нужны дополнительные картинки или MAE/RMSE.",
        "",
        "## Сколько запусков найдено",
        "",
        f"- Всего уникальных run-папок: {len(runs)}.",
        f"- Валидных запусков: {valid}.",
        f"- Невалидных запусков: {invalid}.",
        f"- Плохих или нестабильных по выбранному порогу: {bad}.",
        "",
        "`bad_runs.csv` включает и невалидные запуски, и валидные запуски с большой ошибкой. Поэтому число bad может пересекаться с числом valid.",
        "",
        "## Главные таблицы",
        "",
        "- `tables/selected_cases.csv`",
        "- `tables/grouped_by_dtype.csv`",
        "- `tables/fp32_fp64_comparison.csv`",
        "- `tables/fp16_summary.csv`",
        "- `tables/run_quality.csv`",
        "",
        "## Главные графики",
        "",
        "- `figures/report_best_l2_by_dtype.png`",
        "- `figures/report_fp64_fp32_ratio.png`",
        "- `figures/report_seed_scatter.png`",
        "- `figures/report_convection_beta50_curves.png`",
        "- `figures/report_helmholtz_m12_curves.png`",
        "",
        "## Основные кейсы",
        "",
        "- Helmholtz, m=12 - основной пример, где FP64 заметно лучше.",
        "- Convection, beta=50 - потенциально сильный пример, но только один seed, поэтому вывод осторожный.",
        "- Burgers, nu=0.002 - пример, где FP32 и FP64 близки.",
        "- Burgers, nu=0.001 - пример, где FP64 не дал преимущества.",
        "- Helmholtz, m=8 - случай с зависимостью от seed.",
        "- FP16 - отдельный блок; в этих запусках часто нестабилен.",
        "",
        "## Что можно писать в отчёте",
        "",
        "- На Helmholtz при m=12 FP64 дал меньшую ошибку, чем FP32.",
        "- На convection при beta=50 FP64 выглядит сильно лучше, но данных мало, поэтому этот кейс лучше подавать как сложный режим, а не как устойчивый результат.",
        "- На Burgers преимущество FP64 не проявилось стабильно.",
        "- FP16 в этих запусках часто давал плохую сходимость или невалидные метрики.",
        "",
        "## Что нельзя писать",
        "",
        "- FP64 всегда лучше.",
        "- FP32 всегда ломается.",
        "- FP16 полноценно сравнен как устойчивый вариант.",
        "- Все результаты устойчивы по seed.",
        "",
        f"Отдельных FP16-групп в сводке: {len(fp16)}.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_missing_artifacts():
    lines = [
        "# Чего не хватает для аккуратного отчёта",
        "",
        "## Уже есть",
        "",
        "- таблицы best/final relative L2;",
        "- сравнение FP32 и FP64;",
        "- отдельная таблица по FP16;",
        "- графики loss/L2 для выбранных кейсов;",
        "- выбранные кейсы для отчёта.",
        "",
        "## Ещё желательно добавить",
        "",
        "- MAE/RMSE для выбранных convection-запусков;",
        "- карты exact / prediction / error для convection beta=50;",
        "- один selected check для FP16 на convection beta=50, если нужен пример неудачного запуска.",
        "",
        "## Большой перезапуск",
        "",
        "Большой перезапуск всех экспериментов не нужен.",
        "",
        "## Минимальные дозапуски",
        "",
        "- convection_beta50_fp32_seed0",
        "- convection_beta50_fp64_seed0",
        "- convection_beta50_fp16_seed0",
        "",
        "Перезапуск всех 33 runs для отчёта не нужен.",
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
