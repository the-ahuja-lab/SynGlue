# SynGlue — reproducibility bundle (organised by model)

Reproduction code, configurations, splits, predictions, metrics, tests and environments for the
SynGlue manuscript.

## Layout — one folder per model

```
reproducibility/
├── README.md                        this file
├── MANIFEST.csv                     every file with SHA256 + original source path
├── RUN_STEPS.md                     the nine ordered steps and their outputs
├── run_all.sh                       runs the light steps in order
│
├── endpoint_regression/             MODEL 1 — component-aware Transformer + DC50/Dmax RF regressors
│   ├── README.md                    architecture, configs, splits, predictions, metrics, status
│   ├── scripts/                     01_prepare_components.py … 06_check_data_leakage.py
│   ├── configs/                     transformer.yaml · rf_dc50.yaml · rf_dmax.yaml
│   ├── splits/                      transformer_train_val.csv · rf_train_test.csv ·
│   │                                scaffold_test.csv · target_test.csv
│
├── linker_multiclass/               MODEL 2 — independent 4-class linker classifier
│   ├── README.md                    model, features, protocols, metrics, limitations
│   ├── scripts/                     08_train_linker_classifier.py
│   ├── configs/                     linker_classifier.yaml
│   ├── splits/                      linker_cv_folds.csv
│
├── external_benchmark/              comparator evaluation and external populations
│   ├── README.md                    populations, leakage rules, AUROCs, denominators
│   ├── scripts/                     07_evaluate_external.py
│   ├── splits/                      external_test.csv
│
├── figures/                         computational figure generation
│   ├── README.md                    figure list + manuscript panel mapping
│   └── scripts/                     09_reproduce_figures.py
│
├── shared/                          utilities, environments, artifact hashes, global index
│   ├── README.md                    path resolution, versions, integrity notes
│   ├── _common.py · ARTIFACTS.md · sha256_manifest.txt · mapped_index.csv
│   └── environment/                 environment.lock.yml · software_versions.json ·
│                                    environment.yml · requirements-legacy.txt · README.md
│
├── tests/                           test_checkpoint_load.py · test_split_independence.py ·
│                                    test_prediction_reproduction.py · test_metrics.py
│
├── legacy_code/                     full role-based code library
│   └── model_code/ analysis/ comparators/ data_prep/ inference/ pipeline/ plots/ misc/
│
├── outputs/                         runtime outputs (git-ignored)
└── figures/                         runtime figures (git-ignored)
```

## The two model families (do not conflate)

| | Endpoint regression | Linker multiclass |
|---|---|---|
| Folder | `endpoint_regression/` | `linker_multiclass/` |
| Model | 4-head, 2-layer Transformer (4,800→512, no positional embedding) + two RF regressors (500 trees, depth 20) | GradientBoostingClassifier on 4,800 features |
| Task | DC50 (log10 nM) and Dmax (%) regression | 4-class DC50-derived linker potency |
| Artifact | `multitask_transformer.pt` (9,026,563 params, SHA256 `630783ce…`) + `rf_dc50/rf_dmax.joblib` | `linker_classifier.pkl` (md5 `d5204351…`) |
| Evaluation | random / scaffold / target / E3 splits | original (test-selected) + grouped GroupKFold |
| Status | DC50 split metrics reproduced | protocol corrected; grouped AUROC 0.8035 (ChemBERTa) |

A third, separate family — an **8-head, 3-layer, positional-embedding** model (12,327,054 params) —
belongs to the two-stage analysis and is kept in `legacy_code/model_code/`.

## Quick start

```bash
export SYNGLUE_ROOT=/path/to/SynGlue
export SYNGLUE_PY_ROOT=/path/to/SynGlue_Py
export SYNGLUE_DATA_ROOT=/path/to/curated/tables
export GROVER_DIR=/path/to/grover_patched       # step 01 only
export GROVER_CKPT=/path/to/grover_fixed.pt     # step 01 only

bash reproducibility/run_all.sh                 # steps 05–09 on available artifacts
python reproducibility/tests/../tests -m pytest # verification suite
```
