# Selected cases

## 1. Конвекция, beta=50

- variant: `convection_beta50_wide_lbfgs`
- why selected: hard failure: fp32 is bad while fp64 reaches low error; needs careful wording
- FP32: n=1, median best L2=0.6873, bad rate=1
- FP64: n=1, median best L2=0.007182, bad rate=0
- conclusion: unstable_or_seed_sensitive
- source runs: final/final 2/runs/run031_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp32_s0|final/final 2/runs/run032_convection1d_convection_beta50_wide_lbfgs_beta50p0_fp64_s0

## 2. Гельмгольц, m=12

- variant: `helmholtz_resample_long`
- why selected: stable positive: median fp64 is better and both dtypes have valid repeated runs
- FP32: n=2, median best L2=0.01253, bad rate=0
- FP64: n=2, median best L2=0.002114, bad rate=0
- conclusion: fp64_clear_better
- source runs: final/final 2/runs/run019_helmholtz1d_helmholtz_resample_long_m12_fp32_s0|final/final 2/runs/run020_helmholtz1d_helmholtz_resample_long_m12_fp32_s1|final/final 2/runs/run021_helmholtz1d_helmholtz_resample_long_m12_fp64_s0|final/final 2/runs/run022_helmholtz1d_helmholtz_resample_long_m12_fp64_s1

## 3. Гельмгольц, m=12

- variant: `resample_proven_128`
- why selected: stable positive: median fp64 is better and both dtypes have valid repeated runs
- FP32: n=3, median best L2=0.01269, bad rate=0
- FP64: n=3, median best L2=0.003085, bad rate=0
- conclusion: fp64_clear_better
- source runs: results_exp_12_helmholts_more/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r017_m12_resample_proven_128_fp32_s0|results_exp_12_helmholts_more/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r018_m12_resample_proven_128_fp32_s1|results_exp_12_helmholts_more/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r019_m12_resample_proven_128_fp64_s0|results_exp_12_helmholts_more/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r020_m12_resample_proven_128_fp64_s1|results_exp_9_helmholtz1d_large_lbfgs/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r005_m12_resample_proven_128_fp32_s0|results_exp_9_helmholtz1d_large_lbfgs/results_exp_9_helmholtz1d_large_lbfgs/runs/exp9_r006_m12_resample_proven_128_fp64_s0

## 4. Гельмгольц, m=7

- variant: `helmholtz_rs`
- why selected: stable positive: median fp64 is better and both dtypes have valid repeated runs
- FP32: n=2, median best L2=0.001348, bad rate=0
- FP64: n=2, median best L2=0.0005144, bad rate=0
- conclusion: fp64_clear_better
- source runs: results_exp_6_helmholtz1d_resampling_final/runs/helmholtz_rs_m7_fp32_0|results_exp_6_helmholtz1d_resampling_final/runs/helmholtz_rs_m7_fp32_1|results_exp_6_helmholtz1d_resampling_final/runs/helmholtz_rs_m7_fp64_0|results_exp_6_helmholtz1d_resampling_final/runs/helmholtz_rs_m7_fp64_1

## 5. Бюргерс, nu=0.1

- variant: `nu_0p1`
- why selected: control case: fp64 is close to fp32
- FP32: n=3, median best L2=0.01967, bad rate=0
- FP64: n=3, median best L2=0.01995, bad rate=0
- conclusion: similar
- source runs: results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp32_0|results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp32_1|results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp32_2|results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp64_0|results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp64_1|results_exp_2_burgers1d_nu/runs/burgers1d_nu_0p1_fp64_2

## 6. Бюргерс, nu=0.001

- variant: `burgers_more_points`
- why selected: negative case: fp64 is not uniformly better
- FP32: n=2, median best L2=0.09485, bad rate=0
- FP64: n=2, median best L2=0.17, bad rate=0
- conclusion: fp32_better
- source runs: final/final 2/runs/run001_burgers1d_burgers_more_points_nu0p001_fp32_s0|final/final 2/runs/run002_burgers1d_burgers_more_points_nu0p001_fp32_s1|final/final 2/runs/run003_burgers1d_burgers_more_points_nu0p001_fp64_s0|final/final 2/runs/run004_burgers1d_burgers_more_points_nu0p001_fp64_s1

## 7. FP16

- variant: `separate dtype check`
- why selected: fp16 summary: keep failures separate from fp32/fp64 comparison
- conclusion: 22/22 fp16 runs are bad or invalid
