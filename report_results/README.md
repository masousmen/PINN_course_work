# Итоговые результаты

В этой папке лежит чистый слой результатов для отчёта. Сырые логи не удалялись, но смотреть в первую очередь нужно на таблицы и графики из этой папки.

## Таблицы

- `tables/report_main_cases.csv` - главные кейсы для текста отчёта
- `tables/report_helmholtz_cases.csv` - отдельная таблица по Helmholtz
- `tables/report_diagnostic_cases.csv` - спорные и нестабильные случаи
- `tables/task_overview.csv` - общий обзор найденных запусков
- `tables/fp32_fp64_comparison.csv` - сравнение FP32 и FP64 по сопоставимым настройкам
- `tables/fp16_summary.csv` - отдельная сводка по FP16
- `tables/bad_runs.csv` - плохие и невалидные запуски
- `tables/all_runs_normalized.csv` - все найденные run-папки после нормализации

## Графики

- `figures/report_main_best_l2_by_dtype.png`
- `figures/report_main_fp64_fp32_ratio.png`
- `figures/report_main_seed_scatter.png`
- `figures/report_helmholtz_main_ratio.png`
- `figures/report_helmholtz_rs_sweep.png`
- `figures/report_helmholtz_m12_curves.png`
- `figures/report_convection_beta30_curves.png`
- `figures/report_convection_beta50_check.png`
- `figures/report_burgers_summary.png`
- `figures/report_fp16_summary.png`
- `figures/report_diagnostic_seed_sensitive.png`

## Как читать результаты

Helmholtz - основной блок. В нём есть несколько сопоставимых запусков, где FP64 даёт меньшую медианную ошибку, особенно при `m=12`.

Convection beta=30 - основной baseline для convection. Он нужен как аккуратный пример без сильной зависимости от одного плохого seed.

Convection beta=50 - diagnostic. Там FP64 выглядит сильно лучше, но запусков мало, поэтому этот кейс не стоит использовать как главный устойчивый аргумент.

Burgers - смешанный пример. На части запусков FP32 и FP64 близки, а на части FP64 не даёт преимущества.

FP16 - отдельный failure-блок. В этих логах он чаще даёт плохие или невалидные метрики, поэтому я не смешиваю его с основной таблицей FP32/FP64.

Главная формулировка для отчёта: FP64 помогает в некоторых сложных режимах, но не является универсально лучшим вариантом.
