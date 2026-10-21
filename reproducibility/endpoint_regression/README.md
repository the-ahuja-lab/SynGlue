# Endpoint regression — Transformer + DC50/Dmax Random Forests

This folder contains everything for the **degradation-endpoint prediction model**: the component-aware
Transformer and the two Random-Forest regressors. It is one of the two model families in SynGlue;
the other (linker multiclass) lives in `../linker_multiclass/`.

```
endpoint_regression/
├── scripts/     01_prepare_components.py … 06_check_data_leakage.py
├── configs/     transformer.yaml · rf_dc50.yaml · rf_dmax.yaml
├── splits/      transformer_train_val.csv · rf_train_test.csv · scaffold_test.csv · target_test.csv
```

## 1. Architecture 

| Element | Value |
|---|---|
| Input | `(N, 3, 4800)` — GROVER embeddings for [warhead, linker, E3-recruiter] |
| Projection | `Linear(4800 → 512)` |
| Encoder | `TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)`, **2 layers** |
| Positional embedding | **absent** |
| Pooling | `Linear(512 → 1)` + softmax over the 3 tokens |
| Heads | `512 → 256 → ReLU → Dropout(0.2) → 1` (DC50, Dmax) |
| Parameters | **9,026,563** (36 state-dict keys) |
| Checkpoint | `multitask_transformer.pt`, SHA256 `630783ce2f6f26e779e6aae004e0b6827135819ba3be3a2c16f458ea999d6e05` |
| Loss | masked MSE, `L = L_DC50 + 0.01·L_Dmax` |
| Optimiser | AdamW, lr 1e-4; batch 64 (train) / 128 (validation) |
| Schedule | up to 100 epochs, early stopping on validation loss, patience 15 |
| Split | 80:20, seed 42 |
| RF heads | `RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)` on the fused 512-d representation |

## 2. Pipeline stages

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_prepare_components.py` | component mapping (first-match, with `--structure-map` variant) + warhead/linker/E3 embeddings with compound IDs |
| 02 | `02_train_multitask_transformer.py` | train the 4-head Transformer with the masked multitask loss |
| 03 | `03_extract_embeddings.py` | produce/verify the fused 512-d matrices (`Z_all`, `Z_train`, `Z_test`) |
| 04 | `04_train_rf_regressors.py` | fit the DC50 and Dmax forests from the fused features |
| 05 | `05_evaluate_regression.py` | recompute R², RMSE, MAE, Spearman, Kendall, Pearson per split |
| 06 | `06_check_data_leakage.py` | compound-level overlap across train/test and against the external benchmark |

## 3. Splits 

| File | Content | Size |
|---|---|---|
| `splits/transformer_train_val.csv` | train / val / test assignment from the saved Random split | 9,252 rows |
| `splits/rf_train_test.csv` | RF train/test (the DC50-labelled subset, 80:20) | 1,531 / 383 |
| `splits/scaffold_test.csv` | scaffold-held-out test compounds | 843 |
| `splits/target_test.csv` | target-held-out test compounds | 1,447 |
| `../shared/mapped_index.csv` | full 9,252-row index with component IDs and canonical SMILES | 9,252 |

## 4. How to run

```bash
cd endpoint_regression/scripts
python 06_check_data_leakage.py        # runs on the recovered artifacts
python 05_evaluate_regression.py       # recomputes the split metrics
python 01_prepare_components.py        # heavy: needs GROVER_DIR + GROVER_CKPT
python 02_train_multitask_transformer.py --tensor X.npy --dc50 y50.npy --dmax ydm.npy
python 03_extract_embeddings.py --mode verify
python 04_train_rf_regressors.py
```
Outputs go to `../outputs/` (runtime, git-ignored).
