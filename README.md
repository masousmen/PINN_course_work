# Эксперименты с точностью вычислений в PINN

Здесь лежит код и результаты экспериментов для курсовой. Я сравниваю FP32, FP64 и FP16 при обучении PINN.

## Структура проекта

- `src/pinn_model.py` - основной код модели и обучения
- `notebooks/results_summary.ipynb` - главный ноутбук для просмотра результатов
- `scripts/analyze_results.py` - сбор таблиц и графиков из готовых логов
- `report_results/tables` - итоговые таблицы для отчёта
- `report_results/figures` - итоговые графики
- `experiments_raw/` - запуски экспериментов

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Пересобрать таблицы

```bash
python scripts/analyze_results.py
```

## Главные таблицы

- `report_results/tables/report_main_cases.csv`
- `report_results/tables/report_helmholtz_cases.csv`
- `report_results/tables/report_diagnostic_cases.csv`
- `report_results/tables/task_overview.csv`
- `report_results/tables/fp32_fp64_comparison.csv`
- `report_results/tables/fp16_summary.csv`

## Главные графики

- `report_results/figures/report_main_best_l2_by_dtype.png`
- `report_results/figures/report_main_fp64_fp32_ratio.png`
- `report_results/figures/report_main_seed_scatter.png`
- `report_results/figures/report_helmholtz_ratio.png`
- `report_results/figures/report_helmholtz_rs_sweep.png`
- `report_results/figures/report_helmholtz_m12_curves.png`
