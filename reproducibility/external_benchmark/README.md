# External benchmark and comparators

Everything needed to reproduce the **external degradation-model benchmark**: how the external
populations were built, what each comparator predicts, which leakage rule applies to each, and the
resulting per-comparator denominators and AUROCs.

```
external_benchmark/
├── scripts/     07_evaluate_external.py
├── splits/      external_test.csv
└── predictions/ external_predictions.csv · comparator_predictions.csv
```

## 1. External populations (verified)

| Stage | n | Rule |
|---|---:|---|
| Assembled endpoint-bearing pool | **2,930** | `final_dataset_E3_known` minus PROTAC-DB rows, deduplicated by canonical SMILES — **before** overlap control |
| Exact-overlap-free | **2,562** | removed 952 compounds matching the PROTAC-DB training set exactly |
| max Tanimoto < 0.90 | **2,484** | chirality-aware Morgan (r=2, 2048-bit) vs training |
| max Tanimoto < 0.85 | **2,431** | as above |
| max Tanimoto < 0.70 | **2,256** | as above (strict novel set) |

`2,930` is the **assembled pool**, not the overlap-excluded test set. The strict 2,256 set was **not**
used for the comparator metrics.

`splits/external_test.csv` contains the 2,562 compounds with source, target, E3 and
`tmax_chiral` (maximum Tanimoto similarity to training).

## 2. Comparators and leakage rules

| Comparator | Prediction column | Leakage rule applied | n at ≤100 nM |
|---|---|---|---:|
| SynGlue whole-PROTAC | `gen_whole_pred_dc50_nM` | exact-in-SynGlue removed | 1,700 |
| SynGlue 3-token | `gen_pred_dc50_nM` | exact-in-SynGlue removed; decomposed 151-set | 71 |
| PDP | `pdp_active_proba` | none (baseline) | 1,809 |
| PDP-XGB | `pdp_xgb_proba` | none (baseline) | 1,809 |
| PROTAC-STAN | `protacstan_active_proba` | exact-in-STAN removed | 1,602 |
| Morgan-RF / Tanimoto-NN | (baseline scripts) | none | 1,809 |
| PROTAC-DC50 | (component-decomposition pipeline) | exact-in-SynGlue removed; 151-set | 71 |
| TACK-XGB / TACK-MLP | (TACK ensembles) | **label inverted in earlier reporting** | 1,481 |

**TACK caveat:** the reported n = 1,481 is the **non-held-out** evaluation set; it includes 109
compounds that are exact matches to SynGlue training. The held-out count is 328. TACK is therefore
excluded from the intersection computation here.

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
