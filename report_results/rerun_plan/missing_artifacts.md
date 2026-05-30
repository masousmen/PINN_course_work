# Чего не хватает для аккуратного отчёта

## Уже есть

- таблицы best/final relative L2;
- сравнение FP32 и FP64;
- отдельная таблица по FP16;
- графики loss/L2 для выбранных кейсов;
- выбранные кейсы для отчёта.

## Ещё желательно добавить

- MAE/RMSE для выбранных convection-запусков;
- карты exact / prediction / error для convection beta=50;
- один selected check для FP16 на convection beta=50, если нужен пример неудачного запуска.

## Большой перезапуск

Большой перезапуск всех экспериментов не нужен.

## Минимальные дозапуски

- convection_beta50_fp32_seed0
- convection_beta50_fp64_seed0
- convection_beta50_fp16_seed0

Перезапуск всех 33 runs для отчёта не нужен.
