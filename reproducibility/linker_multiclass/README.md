# Linker multiclass classifier

The **second, independent model family** in SynGlue. It shares no parameters, no training data
pipeline and no evaluation with the endpoint Transformer; it predicts a four-class
DC50-derived linker-potency label from molecular features.

```
linker_multiclass/
├── scripts/     08_train_linker_classifier.py
├── configs/     linker_classifier.yaml
├── splits/      linker_cv_folds.csv
└── (metrics written to ../outputs/ at run time)
```

## 1. Model and features

| Element | Value |
|---|---|
| Model | `GradientBoostingClassifier` (independent of the Transformer) |
| Hyperparameters | `n_estimators=228, max_depth=7, min_samples_leaf=9, min_samples_split=13, max_features='log2', learning_rate=0.05, random_state=42` |
| Preprocessing | mean imputation + standardisation, fitted **on the training partition only** |
| Deployed features | **4,800-dimensional GROVER** vectors |
| Features available in this bundle | **384-dimensional ChemBERTa** matrix (the GROVER matrix is not deposited) |
| Classes | 4 (DC50-derived), counts **580 / 580 / 581 / 579** |
| Records | **2,320 rows** = **846 unique linker identifiers** = **675 unique canonical structures** |
| Artifact | `linker_classifier.pkl`, md5 `d5204351e9c42dbfbe1bc108a478d09f` (three byte-identical copies) |

## 2. Splits

- `splits/linker_cv_folds.csv` — fold assignment for a **5-fold GroupKFold by linker identity**
  (deterministic), with `linker_id`, `fold`, `label`. Group overlap is **zero** in every fold
  (677/169, 677/169, 677/169, 677/169, 676/170).
- The original protocol used `train_test_split(test_size=0.2, random_state=42, stratify=y)` →
  1,856 / 464 rows.

## 3. Protocols (both implemented)

| Mode | Behaviour | Purpose |
|---|---|---|
| `--mode original` | RandomizedSearchCV (5-fold, `roc_auc_ovr`) on the training partition; **model selection by test AUC** | reproduces the deposited protocol so its bias is visible |
| `--mode grouped` | GroupKFold by linker identity, train-only preprocessing, fixed hyperparameters, out-of-fold predictions | valid grouped evaluation |

## 4. How to run

```bash
cd linker_multiclass/scripts
python 08_train_linker_classifier.py --mode grouped    # valid grouped evaluation
python 08_train_linker_classifier.py --mode original   # deposited (test-selected) protocol
```
Outputs (`linker_grouped_metrics.json`, `linker_grouped_oof.csv`, `linker_original_metrics.json`)
are written to `../outputs/` (runtime, git-ignored).
