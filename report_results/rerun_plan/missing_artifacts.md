# Missing artifacts and small rerun plan

## Missing metrics.csv
- none for selected cases

## Missing FP32/FP64 pairs
- none for selected cases

## Seed coverage
- convection1d beta=50 `convection_beta50_wide_lbfgs`: not enough repeated seeds

## Missing MAE/RMSE
- convection1d beta=50 `convection_beta50_wide_lbfgs`

## Missing solution/error maps
- convection1d beta=50 `convection_beta50_wide_lbfgs`

Large rerun is not needed.

Small selected checks are useful only for dense-map figures and MAE/RMSE.

Planned runs:
- convection_beta50_fp32_seed0
- convection_beta50_fp64_seed0
- convection_beta50_fp16_seed0, skipped automatically on CPU