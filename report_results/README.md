# Report results

These files are the cleaned output layer for the report.
They are rebuilt from real run folders with `summary.json` and `metrics.csv`; old aggregate CSV files are not used as run sources.

Unique run folders found: 439.
Valid runs: 415.
Invalid runs: 24.
Bad or unstable by threshold: 240.

`bad_runs.csv` includes invalid runs and valid runs with high error, so bad and valid counts can overlap.

## Main tables

- `tables/selected_cases.csv`
- `tables/grouped_by_dtype.csv`
- `tables/fp32_fp64_comparison.csv`
- `tables/fp16_summary.csv`
- `tables/run_quality.csv`

## Main figures

- `figures/report_best_l2_by_dtype.png`
- `figures/report_fp64_fp32_ratio.png`
- `figures/report_seed_scatter.png`
- `figures/report_convection_beta50_curves.png`
- `figures/report_helmholtz_m12_curves.png`

## Selected cases

- `Helmholtz, m=12`: stable_fp64_better (strong)
- `Convection, beta=50`: unstable_or_seed_sensitive (weak_needs_rerun)
- `Burgers, nu=0.002`: similar (strong)
- `Burgers, nu=0.001`: fp32_better (strong)
- `Helmholtz, m=8`: unstable_or_seed_sensitive (medium)
- `FP16 summary`: fp16 failed or unstable (medium)

## Report wording

Can say:
- FP64 improves selected hard cases.
- FP64 is not uniformly better than FP32.
- FP16 is mostly unstable in these runs.
- Some cases are seed-sensitive.

Should not say:
- FP64 always wins.
- FP32 fails everywhere.
- FP16 was fully evaluated as a stable baseline.

FP16 groups in the separate summary: 12.
