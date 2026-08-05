# Major-revision result summary

Generated from: `/root/autodl-tmp/ms-ATCN/results/revision_required`

Core tables:
- `main_multiseed_summary.csv`
- `main_pairwise_wilcoxon.csv`
- `ablation_summary.csv`
- `ablation_wilcoxon_holm.csv`
- `unsw_leakage_summary.csv`
- `unsw_leakage_wilcoxon_holm.csv`
- `window_sensitivity_summary.csv`
- `order_sanity_summary.csv`
- `order_sanity_wilcoxon_holm.csv`
- `cic_temporal_summary.csv`
- `hyperparameter_sensitivity_summary.csv`
- `efficiency_all_runs.csv`

All p-values are two-sided paired Wilcoxon signed-rank tests using seed-aligned runs.
Holm correction is applied within each dataset/model/metric comparison family.
With six seeds, statistical power remains limited; report effect sizes and mean ± standard deviation
alongside p-values and avoid universal superiority claims.
