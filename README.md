# PINN precision experiments

This project compares FP32, FP64 and FP16 precision for training physics-informed neural networks.

The experiments cover:
- `convection1d`
- `helmholtz1d`
- `burgers1d`
- `heat1d` as a sanity check

The main conclusion is intentionally modest: FP64 helps in some hard settings, but it is not always better than FP32. FP16 is mostly unstable in these runs.

## Project structure

- `src/pinn_model.py` — main PINN model, losses, exact/reference solutions and `run_experiment`.
- `notebooks/results_summary.ipynb` — notebook for looking through already generated logs.
- `report_results/tables` — main tables for the report.
- `report_results/figures` — main figures for the report.
- `report_results/selected_runs` — paths for selected report cases.
- `report_results/rerun_plan` — notes about small optional checks.
- `results_exp_*` and `final/` — archived old experiment runs, not the main layer to read first.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce analysis

Open the notebook:

```bash
jupyter notebook notebooks/results_summary.ipynb
```

Or rebuild the report tables and figures from existing logs:

```bash
python analyze_results.py
```

This reads existing `summary.json` and `metrics.csv` files only. It does not rerun the full experiments.

## Main report files

- `report_results/tables/selected_cases.csv`
- `report_results/tables/grouped_by_dtype.csv`
- `report_results/tables/fp32_fp64_comparison.csv`
- `report_results/tables/fp16_summary.csv`
- `report_results/figures/fp32_fp64_median_best_l2.png`
- `report_results/figures/fp64_over_fp32_ratio.png`
- `report_results/figures/seed_scatter_best_l2.png`

For the written report, prefer the cleaned figure names:

- `report_results/figures/report_best_l2_by_dtype.png`
- `report_results/figures/report_fp64_fp32_ratio.png`
- `report_results/figures/report_seed_scatter.png`
- `report_results/figures/report_convection_beta50_curves.png`
- `report_results/figures/report_helmholtz_m12_curves.png`

## Notes

- Bad seeds were not removed.
- FP16 is analyzed separately from the main FP32/FP64 comparison.
- The project does not claim that FP64 always wins.
