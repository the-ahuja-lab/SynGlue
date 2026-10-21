# Linker multiclass classifier

The **second, independent model family** in SynGlue. It shares no parameters, no training data
pipeline and no evaluation with the endpoint Transformer; it predicts a four-class
DC50-derived linker-potency label from molecular features.

```
linker_multiclass/
├── scripts/     08_train_linker_classifier.py
├── configs/     linker_classifier.yaml
├── splits/      linker_cv_folds.csv
├── predictions/ linker_oof_predictions.csv
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

## 4. Predictions and metrics

- `predictions/linker_oof_predictions.csv` — out-of-fold probabilities per class, true label, pred, group.

| Evaluation | AUROC | Notes |
|---|---:|---|
| Original, test-selected (deployed GB) | **0.8013** | optimistically biased; selection used test AUC |
| Original, median of ten models on one fixed 464-row test set | **0.7975** | a repeated test-set evaluation, **not** cross-validation |
| Grouped, 5-fold by linker identity (ChemBERTa) | **0.8035** | macro OVR; macro-F1 0.5197; balanced acc 0.5181; MCC 0.3580; zero group overlap |
| Mordred representation, fold-level (reference) | 0.745 validation / 0.733 test | different representation |

Per-class sensitivity (grouped): 0.547 / 0.533 / 0.494 / 0.499; specificity 0.882 / 0.816 / 0.795 / 0.866.

## 5. How to run

```bash
cd linker_multiclass/scripts
python 08_train_linker_classifier.py --mode grouped    # valid grouped evaluation
python 08_train_linker_classifier.py --mode original   # deposited (test-selected) protocol
```
Outputs (`linker_grouped_metrics.json`, `linker_grouped_oof.csv`, `linker_original_metrics.json`)
are written to `../outputs/` (runtime, git-ignored).

## 6. Status and limitations

- **Verified:** artifact identity, hyperparameters, feature dimension (4,800 in the artifact),
  class distribution, grouped evaluation with zero group overlap.
- **Protocol corrections required in the manuscript:** (i) selection used **test AUC**; (ii) the
  "ten-fold cross-validation" is a **repeated fixed-test evaluation**; (iii) **Boruta was not run**
  (all 4,800 features retained); (iv) tuning used **RandomizedSearchCV**, not GridSearchCV.
- **Grouped evaluation is on ChemBERTa**, not the deployed GROVER representation — it is supporting
  evidence, not a substitute. A GROVER grouped evaluation requires the 4,800-d feature matrix.
- The pickle loads only with a numpy random-state shim under scikit-learn ≤ 1.1; see
  `../shared/environment/README.md`.
