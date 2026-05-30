# Выбранные кейсы

## Helmholtz, m=12

- задача: `helmholtz1d`
- параметр: `m=12`
- вариант: `helmholtz_resample_long`
- почему выбран: Основной положительный пример: на этом режиме FP64 дал меньшую ошибку по медиане.
- FP32: n=2.0, медиана best L2=0.01253, доля плохих=0
- FP64: n=2.0, медиана best L2=0.002114, доля плохих=0
- вывод: FP64 заметно лучше
- надёжность: сильная
- исходные run-папки:
  - `final/final 2/runs/run019_helmholtz1d_helmholtz_resample_long_m12_fp32_s0`
  - `final/final 2/runs/run020_helmholtz1d_helmholtz_resample_long_m12_fp32_s1`
  - `final/final 2/runs/run021_helmholtz1d_helmholtz_resample_long_m12_fp64_s0`
  - `final/final 2/runs/run022_helmholtz1d_helmholtz_resample_long_m12_fp64_s1`

## Convection, beta=50

- задача: `convection1d`
- параметр: `beta=50`
- вариант: `convection_beta50_wide_lbfgs`
- почему выбран: Сложный режим: FP64 выглядит сильно лучше, но запусков мало, поэтому вывод осторожный.
- FP32: n=1.0, медиана best L2=0.6873, доля плохих=1
- FP64: n=1.0, медиана best L2=0.007182, доля плохих=0
- вывод: зависит от seed
- надёжность: нужна проверка
- исходные run-папки:
  - `final/final 2/runs/run031_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp32_s0`
  - `final/final 2/runs/run032_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp64_s0`

## Burgers, nu=0.002

- задача: `burgers1d`
- параметр: `nu=0.002`
- вариант: `burgers_more_points`
- почему выбран: Контрольный пример: FP32 и FP64 дали близкие ошибки.
- FP32: n=2.0, медиана best L2=0.04878, доля плохих=0
- FP64: n=2.0, медиана best L2=0.04652, доля плохих=0
- вывод: FP32 и FP64 близки
- надёжность: сильная
- исходные run-папки:
  - `final/final 2/runs/run007_burgers1d_burgers_more_points_nu0p002_fp32_s0`
  - `final/final 2/runs/run008_burgers1d_burgers_more_points_nu0p002_fp32_s1`
  - `final/final 2/runs/run009_burgers1d_burgers_more_points_nu0p002_fp64_s0`
  - `final/final 2/runs/run010_burgers1d_burgers_more_points_nu0p002_fp64_s1`

## Burgers, nu=0.001

- задача: `burgers1d`
- параметр: `nu=0.001`
- вариант: `burgers_more_points`
- почему выбран: Пример, где FP64 не дал преимущества.
- FP32: n=2.0, медиана best L2=0.09485, доля плохих=0
- FP64: n=2.0, медиана best L2=0.17, доля плохих=0
- вывод: FP32 лучше
- надёжность: сильная
- исходные run-папки:
  - `final/final 2/runs/run001_burgers1d_burgers_more_points_nu0p001_fp32_s0`
  - `final/final 2/runs/run002_burgers1d_burgers_more_points_nu0p001_fp32_s1`
  - `final/final 2/runs/run003_burgers1d_burgers_more_points_nu0p001_fp64_s0`
  - `final/final 2/runs/run004_burgers1d_burgers_more_points_nu0p001_fp64_s1`

## Helmholtz, m=8

- задача: `helmholtz1d`
- параметр: `m=8`
- вариант: `helmholtz_main`
- почему выбран: Пример с заметной зависимостью от seed.
- FP32: n=2.0, медиана best L2=0.5019, доля плохих=0.5
- FP64: n=2.0, медиана best L2=0.0008814, доля плохих=0
- вывод: зависит от seed
- надёжность: средняя
- исходные run-папки:
  - `final/final/runs/run013_helmholtz1d_helmholtz_main_m8_fp32_s0`
  - `final/final/runs/run014_helmholtz1d_helmholtz_main_m8_fp32_s1`
  - `final/final/runs/run015_helmholtz1d_helmholtz_main_m8_fp64_s0`
  - `final/final/runs/run016_helmholtz1d_helmholtz_main_m8_fp64_s1`

## FP16

- задача: `fp16`
- параметр: `все fp16-запуски`
- вариант: `отдельно`
- почему выбран: FP16 вынесен отдельно, потому что в этих запусках он часто давал плохие или невалидные метрики.
- вывод: FP16 нестабилен
- надёжность: средняя

