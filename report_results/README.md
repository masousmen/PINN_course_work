# Итоговые результаты

Здесь лежит отчётный слой: таблицы и графики, собранные из уже готовых логов.
Сырые запуски не удалялись. Для выводов я смотрю на медиану, число seed и bad rate, а не на лучший отдельный запуск.

## Что лежит в папке

- `tables/all_runs_normalized.csv` - все найденные run-папки;
- `tables/task_overview.csv` - общий обзор по задачам, параметрам и dtype;
- `tables/fp32_fp64_comparison.csv` - сравнение FP32 и FP64 внутри одинаковых настроек;
- `tables/report_cases.csv` - главная таблица для отчёта;
- `tables/fp16_summary.csv` - отдельный обзор FP16;
- `figures/` - основные графики;
- `rerun_plan/optional_checks.md` - небольшие проверки, если их захочется добавить.

## Сколько данных найдено

- Всего run-папок: 439.
- Валидных запусков: 415.
- Плохих или нестабильных по порогу: 240.
- Задачи: burgers1d, convection1d, heat1d, helmholtz1d.

## Как читать результаты

Главный устойчивый положительный пример - Helmholtz m=12. Там есть по два валидных seed у FP32 и FP64, и FP64 лучше по медиане.
Convection beta=50 оставлен как предварительный hard-case: там только один seed, поэтому его нельзя использовать как главный устойчивый вывод.
Burgers получился смешанным: на части запусков FP32 и FP64 близки, на части FP32 лучше.
FP16 я не смешиваю с основной таблицей, потому что в этих запусках он часто даёт плохие или невалидные метрики.

## Основные файлы для отчёта

- `tables/report_cases.csv`
- `tables/task_overview.csv`
- `tables/fp32_fp64_comparison.csv`
- `tables/fp16_summary.csv`
- `figures/report_best_l2_by_dtype.png`
- `figures/report_fp64_fp32_ratio.png`
- `figures/report_seed_scatter.png`
- `figures/report_task_overview.png`

FP16-групп в отдельной сводке: 12.
