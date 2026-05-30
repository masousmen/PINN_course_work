# Эксперименты с точностью вычислений в PINN

Это код и результаты для курсовой работы. Здесь я сравниваю FP32, FP64 и FP16 при обучении Physics-Informed Neural Networks.

В экспериментах есть несколько задач:
- `convection1d`
- `helmholtz1d`
- `burgers1d`
- `heat1d`

Основной вывод аккуратный: FP64 помогает не всегда, но на некоторых сложных режимах даёт заметно меньшую ошибку. FP16 в текущих запусках часто нестабилен.

## Структура проекта

- `src/pinn_model.py` - основной код модели и обучения.
- `notebooks/results_summary.ipynb` - ноутбук для просмотра итоговых таблиц, графиков и коротких выводов.
- `report_results/tables` - итоговые таблицы.
- `report_results/figures` - итоговые графики.
- `report_results/selected_runs` - выбранные кейсы для отчёта.
- `report_results/rerun_plan/optional_checks.md` - небольшие проверки, если нужно усилить спорный кейс.
- `results_exp_*` - старые запуски экспериментов.
- `final/` - архив старых финальных запусков, не основной слой для чтения.

## Установка

```bash
pip install -r requirements.txt
```

## Как посмотреть результаты

Основной способ:

```bash
jupyter notebook notebooks/results_summary.ipynb
```

В ноутбуке есть общий обзор всех найденных запусков, отдельные блоки по Heat, Burgers, Helmholtz, Convection и FP16, итоговые кейсы для отчёта и основные графики.

## Как пересобрать таблицы

```bash
python analyze_results.py
```

Скрипт читает уже готовые `summary.json` и `metrics.csv`. Полный перезапуск экспериментов он не делает.

## Какие файлы использовать в отчёте

- `report_results/tables/report_cases.csv`
- `report_results/tables/task_overview.csv`
- `report_results/tables/selected_cases.csv`
- `report_results/tables/grouped_by_dtype.csv`
- `report_results/tables/fp32_fp64_comparison.csv`
- `report_results/tables/fp16_summary.csv`
- `report_results/figures/report_task_overview.png`
- `report_results/figures/report_best_l2_by_dtype.png`
- `report_results/figures/report_fp64_fp32_ratio.png`
- `report_results/figures/report_seed_scatter.png`
- `report_results/figures/report_convection_beta30_curves.png`
- `report_results/figures/report_convection_beta50_check.png`
- `report_results/figures/report_helmholtz_m12_curves.png`

## Замечания

- Плохие seed не скрывались.
- `convection beta=50` не используется как главный устойчивый вывод: там мало seed, поэтому он вынесен как кейс для проверки.
- FP16 анализируется отдельно от основной таблицы FP32/FP64.
- Вывод “FP64 всегда лучше” здесь не делается.
- Часть старых папок оставлена как архив экспериментов.
