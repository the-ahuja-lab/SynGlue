# External benchmark and comparators

Everything needed to reproduce the **external degradation-model benchmark**: how the external
populations were built, what each comparator predicts, and which leakage rule applies to each.

```
external_benchmark/
├── scripts/     07_evaluate_external.py
├── splits/      external_test.csv
```

## 1. External populations (verified)

| Stage | Rule |
|---|---|
| Assembled endpoint-bearing pool | `final_dataset_E3_known` minus PROTAC-DB rows, deduplicated by canonical SMILES — **before** overlap control |
| Exact-overlap-free | removes compounds matching the PROTAC-DB training set exactly |
| max Tanimoto < 0.90 | chirality-aware Morgan (r=2, 2048-bit) vs training |
| max Tanimoto < 0.85 | as above |
| max Tanimoto < 0.70 | as above (strict novel set) |

`splits/external_test.csv` lists the external compounds with source, target, E3 and the maximum
Tanimoto similarity to training (the counts are produced at run time by step 07).

## 2. Comparators and leakage rules

| Comparator | Prediction column | Leakage rule applied |
|---|---|---|
| SynGlue whole-PROTAC | `gen_whole_pred_dc50_nM` | exact-in-SynGlue removed |
| SynGlue 3-token | `gen_pred_dc50_nM` | exact-in-SynGlue removed; decomposed 151-set |
| PDP | `pdp_active_proba` | none (baseline) |
| PDP-XGB | `pdp_xgb_proba` | none (baseline) |
| PROTAC-STAN | `protacstan_active_proba` | exact-in-STAN removed |
| Morgan-RF / Tanimoto-NN | baseline scripts | none |
| PROTAC-DC50 | component-decomposition pipeline | exact-in-SynGlue removed; 151-set |
| TACK-XGB / TACK-MLP | TACK ensembles | non-held-out evaluation subset |

## 3. Score orientation (a-priori)

- Regressors predicting DC50 → potency score = **−log10(predicted DC50 nM)** (higher = more potent).
- Classifiers → **P(active)**.
- Ground truth: DC50 ≤ threshold, restricted to **non-censored** DC50 labels.

## 4. How to run

```bash
cd external_benchmark/scripts
python 07_evaluate_external.py
```
Outputs (`external_funnel_counts.csv`, `comparator_metrics.csv`, `comparator_denominators.json`)
go to `../outputs/` (runtime, git-ignored).
