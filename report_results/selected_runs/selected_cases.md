# Выбранные кейсы

## heat_alpha01

- задача: `heat1d`
- параметр: `alpha=0.1`
- вариант: `heat1d`
- тип: простая проверка
- статус: ok
- FP32 median best L2: 0.0003753
- FP64 median best L2: 0.0003107
- FP64/FP32: 0.8279
- комментарий: Простая задача для проверки: ошибки у FP32 и FP64 маленькие, большой разницы тут ждать не нужно.
- исходные run-папки:
  - `results_exp_1_heat1d_precision/runs/heat1d_fp32_0`
  - `results_exp_1_heat1d_precision/runs/heat1d_fp32_1`
  - `results_exp_1_heat1d_precision/runs/heat1d_fp32_2`
  - `results_exp_1_heat1d_precision/runs/heat1d_fp64_0`
  - `results_exp_1_heat1d_precision/runs/heat1d_fp64_1`
  - `results_exp_1_heat1d_precision/runs/heat1d_fp64_2`

## helmholtz_m12

- задача: `helmholtz1d`
- параметр: `m=12`
- вариант: `helmholtz_resample_long`
- тип: устойчивый положительный
- статус: ok
- FP32 median best L2: 0.01253
- FP64 median best L2: 0.002114
- FP64/FP32: 0.1687
- комментарий: Основной Helmholtz-кейс: тут есть по два валидных seed и FP64 лучше по медиане.
- исходные run-папки:
  - `final/final 2/runs/run019_helmholtz1d_helmholtz_resample_long_m12_fp32_s0`
  - `final/final 2/runs/run020_helmholtz1d_helmholtz_resample_long_m12_fp32_s1`
  - `final/final 2/runs/run021_helmholtz1d_helmholtz_resample_long_m12_fp64_s0`
  - `final/final 2/runs/run022_helmholtz1d_helmholtz_resample_long_m12_fp64_s1`

## helmholtz_m8

- задача: `helmholtz1d`
- параметр: `m=8`
- вариант: `helmholtz_main`
- тип: зависит от seed
- статус: needs_check
- FP32 median best L2: 0.5019
- FP64 median best L2: 0.0008814
- FP64/FP32: 0.001756
- комментарий: Helmholtz m=8 полезен как пример зависимости от seed: один плохой FP32 seed сильно влияет на картину.
- исходные run-папки:
  - `final/final/runs/run013_helmholtz1d_helmholtz_main_m8_fp32_s0`
  - `final/final/runs/run014_helmholtz1d_helmholtz_main_m8_fp32_s1`
  - `final/final/runs/run015_helmholtz1d_helmholtz_main_m8_fp64_s0`
  - `final/final/runs/run016_helmholtz1d_helmholtz_main_m8_fp64_s1`

## burgers_nu0p002

- задача: `burgers1d`
- параметр: `nu=0.002`
- вариант: `burgers_more_points`
- тип: похоже
- статус: ok
- FP32 median best L2: 0.04878
- FP64 median best L2: 0.04652
- FP64/FP32: 0.9538
- комментарий: Burgers nu=0.002 показывает близкие результаты FP32 и FP64.
- исходные run-папки:
  - `final/final 2/runs/run007_burgers1d_burgers_more_points_nu0p002_fp32_s0`
  - `final/final 2/runs/run008_burgers1d_burgers_more_points_nu0p002_fp32_s1`
  - `final/final 2/runs/run009_burgers1d_burgers_more_points_nu0p002_fp64_s0`
  - `final/final 2/runs/run010_burgers1d_burgers_more_points_nu0p002_fp64_s1`

## burgers_nu0p001

- задача: `burgers1d`
- параметр: `nu=0.001`
- вариант: `burgers_more_points`
- тип: FP32 лучше
- статус: ok
- FP32 median best L2: 0.09485
- FP64 median best L2: 0.17
- FP64/FP32: 1.792
- комментарий: Burgers nu=0.001 оставлен как отрицательный пример: FP64 здесь не дал преимущества.
- исходные run-папки:
  - `final/final 2/runs/run001_burgers1d_burgers_more_points_nu0p001_fp32_s0`
  - `final/final 2/runs/run002_burgers1d_burgers_more_points_nu0p001_fp32_s1`
  - `final/final 2/runs/run003_burgers1d_burgers_more_points_nu0p001_fp64_s0`
  - `final/final 2/runs/run004_burgers1d_burgers_more_points_nu0p001_fp64_s1`

## convection_beta30

- задача: `convection1d`
- параметр: `beta=30`
- вариант: `convection_beta30_lbfgs_grid`
- тип: умеренно положительный
- статус: ok
- FP32 median best L2: 0.01094
- FP64 median best L2: 0.006625
- FP64/FP32: 0.6055
- комментарий: Convection beta=30 выглядит аккуратнее, потому что есть по два seed у FP32 и FP64.
- исходные run-папки:
  - `final/final 2/runs/run025_convection1d_convection_beta30_lbfgs_grid_beta30p0_fp32_s0`
  - `final/final 2/runs/run026_convection1d_convection_beta30_lbfgs_grid_beta30p0_fp32_s1`
  - `final/final 2/runs/run027_convection1d_convection_beta30_lbfgs_grid_beta30p0_fp64_s0`
  - `final/final 2/runs/run028_convection1d_convection_beta30_lbfgs_grid_beta30p0_fp64_s1`

## convection_beta50

- задача: `convection1d`
- параметр: `beta=50`
- вариант: `convection_beta50_wide_lbfgs`
- тип: зависит от seed
- статус: preliminary; needs_check
- FP32 median best L2: 0.6873
- FP64 median best L2: 0.007182
- FP64/FP32: 0.01045
- комментарий: Предварительный hard-case: FP64 выглядит сильно лучше, но есть только один seed. FP32 мог не сойтись из-за неудачного старта.
- исходные run-папки:
  - `final/final 2/runs/run031_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp32_s0`
  - `final/final 2/runs/run032_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp64_s0`

## fp16_summary

- задача: `fp16`
- параметр: `все fp16-запуски`
- вариант: `отдельно`
- тип: FP16 нестабилен
- статус: separate
- комментарий: FP16 вынесен отдельно: 22/22 запусков плохие или невалидные.

