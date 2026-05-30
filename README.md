# Эксперименты с точностью вычислений в PINN

Здесь лежит код и результаты экспериментов для курсовой. Я сравниваю FP32, FP64 и FP16 при обучении PINN.

В логах есть задачи `heat1d`, `burgers1d`, `helmholtz1d` и `convection1d`. Главный экспериментальный блок - Helmholtz. 

## Структура проекта

- `src/pinn_model.py` - код модели и обучения
- `notebooks/results_summary.ipynb` - обзор таблиц, графиков и коротких выводов
- `report_results/tables` - итоговые таблицы
- `report_results/figures` - графики для отчёта
- `report_results/notes/additional_checks.md` - необязательные проверки
- `experiments_raw/` - архивные логи

## Установка

```bash
pip install -r requirements.txt
```

## Как посмотреть результаты

```bash
jupyter notebook notebooks/results_summary.ipynb
```

## Как пересобрать таблицы

```bash
python scripts/analyze_results.py
```

Скрипт читает готовые `summary.json` и `metrics.csv`. Обучение он не запускает.

## Что смотреть в первую очередь

- `report_results/tables/report_main_cases.csv`
- `report_results/tables/report_helmholtz_cases.csv`
- `report_results/tables/report_diagnostic_cases.csv`
- `report_results/tables/task_overview.csv`
- `report_results/tables/fp16_summary.csv`
- `report_results/figures/report_helmholtz_main_ratio.png`
- `report_results/figures/report_helmholtz_m12_curves.png`
- `report_results/figures/report_main_best_l2_by_dtype.png`
- `report_results/figures/report_main_fp64_fp32_ratio.png`
- `report_results/figures/report_main_seed_scatter.png`

## Замечания

- Плохие seed не скрывались.
- `convection beta=50` и старый `helmholtz_main m=8` вынесены в диагностические кейсы.
- FP16 анализируется отдельно от основной таблицы FP32/FP64.
- Вывод `FP64 всегда лучше` здесь не делается.
