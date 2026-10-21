# Figures

Figure generation from saved predictions and metrics. Only **computational** figures are produced;
no wet-lab panels.

```
figures/
├── scripts/09_reproduce_figures.py
└── (outputs written to ../figures/ at run time — git-ignored)
```

## Figures generated

| Figure | Generator function | Input |
|---|---|---|
| External population funnel (2,930 → 2,562 → 2,484 / 2,431 / 2,256) | `fig_external_funnel` | `../outputs/external_funnel_counts.csv` (step 07) |
| Comparator AUROC at ≤100 nM with per-comparator n | `fig_comparator_auroc` | `../outputs/comparator_metrics.csv` (step 07) |
| Held-out R² by split (DC50 / Dmax) | `fig_split_r2` | `../outputs/regression_metrics.csv` (step 05) |
| DC50 density + box: PROTAC-DB vs external | `fig_activity_density` | `external_benchmark/predictions/external_predictions.csv` + training table |
| Linker classifier grouped ROC (per class) | `fig_linker_roc` | `linker_multiclass/predictions/linker_oof_predictions.csv` |
| Partition independence by split | `fig_leakage_independence` | `../outputs/leakage_report.csv` (step 06) |

Each figure is written as **SVG with editable text** (`svg.fonttype = none`) plus a **300-dpi PNG**.

## How to run

```bash
cd figures/scripts
python 09_reproduce_figures.py
```
Figures are written to `/storage/savi/saveenas/Projects/SynGlue/reproducibility/figures/`
(runtime, git-ignored). Steps whose inputs are missing are skipped with a message.

## Manuscript figure mapping

| Manuscript panel | Status |
|---|---|
| Fig 1e–g (two systems: endpoint regression + linker classification) | architecture and both model families documented; legend correction required (4-head baseline; separate classifier) |
| Fig S4 (linker ROC / AUC panels) | regenerate with `fig_linker_roc`; legend must state the grouped protocol and that 0.80 is test-selected |
| Fig S5 (Transformer architecture) | `../endpoint_regression/configs/transformer.yaml`; label the 4-head, no-positional-embedding baseline |
| Fig S6 (external funnel) | `fig_external_funnel`; label 2,930 as the pre-overlap pool |
| Fig S7 (generalization panels) | `fig_split_r2`; replace Dmax values (0.681 / 0.048 / 0.173) and mark Random/Target as row-level |
| Fig S8 (model comparison denominators) | `fig_comparator_auroc`; report per-comparator n |
| Fig S9 (external comparator AUROCs) | `fig_comparator_auroc`; correct the 0.84 at ≤1 µM |
| Fig 2f / 2m (candidate ordering) | **not reproducible** — source values not located |

The wider plotting library (panels, venns, heatmaps, SI composites) remains in
`../legacy_code/plots/`.
