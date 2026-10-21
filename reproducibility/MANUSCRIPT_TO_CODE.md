# MANUSCRIPT_TO_CODE.md — claim → code → artifact → status

Every row maps a manuscript element to the code that implements it and the artifact that supports it.
Status vocabulary: **VERIFIED** (executed/confirmed in this audit) · **DOCUMENTED_ONLY** (described,
not independently run here) · **CONTRADICTED** (code or artifacts disagree with the claim) ·
**UNRESOLVED** (supporting evidence missing).

---

## 1. Methods → code

| Manuscript element | Code | Artifact / evidence | Status |
|---|---|---|---|
| Data curation, PROTAC-DB 3.0 harmonisation (9,380 entries; 6,110 unique SMILES) | `data_prep/` canonicalisation + `scripts/01_prepare_components.py` | `Upadted_PROTAC_features.csv` (10,330 rows / 5,203 compounds); `splits/_mapped_index.csv` | VERIFIED (counts); manuscript says 6,111 → CONTRADICTED (off-by-one) |
| Component decomposition (warhead / linker / E3) | `data_prep/05_Models_attention_dataset_builder.py`; `scripts/01_prepare_components.py` | `splits/_mapped_index.csv` (9,252 mapped rows) | VERIFIED (mapping rule); first-match underdetermined → CONTRADICTED as a component-identity claim |
| GROVER representation (4,800-d per component) | `scripts/01_prepare_components.py` (save_features + fingerprint, `--fingerprint_source both`) | `models/sha256_manifest.txt` (`grover_fixed.pt`) | DOCUMENTED_ONLY (checkpoint not retrained here) |
| Component-aware Transformer (4,800→512, 4 heads, 2 layers, no positional embedding, attention pooling) | `scripts/02_train_multitask_transformer.py`; `scripts/model_code/masked_multitask_trainer.py` | `models/ARTIFACTS.md` (36 keys, 9,026,563 params, SHA256 `630783ce…`) | VERIFIED (strict load); an 8-head variant exists for the two-stage family → report per family |
| Masked multitask objective `L = L_DC50 + 0.01·L_Dmax` | `scripts/02_train_multitask_transformer.py` (`masked_mse`) | executed cell source in `scripts/model_code/` | VERIFIED in code; the *deposited* trainer was single-task → CONTRADICTED for the old deposit |
| AdamW 1e-4, batch 64/128, 100 epochs, patience 15, seed 42, 80:20 split | `scripts/02_train_multitask_transformer.py`; `configs/transformer.yaml` | training logs in `revision/models/final_protac_model/` | VERIFIED |
| Fused 512-d representation for downstream prediction | `scripts/03_extract_embeddings.py` | `Z_all.npy` (9,252×512), `Z_train.npy` (1,531×512), `Z_test.npy` (383×512) | VERIFIED |
| Endpoint Random Forests (500 trees, depth 20) | `scripts/04_train_rf_regressors.py`; `configs/rf_dc50.yaml`, `rf_dmax.yaml` | `models/ARTIFACTS.md` (two RF families, both 500/20/512) | VERIFIED (parameters) |
| Held-out evaluation (random / scaffold / target / E3) | `scripts/05_evaluate_regression.py`; `scripts/06_check_data_leakage.py` | `splits/*_test.csv`; saved metrics table | VERIFIED (DC50 R² 0.757 / 0.566 / 0.826); random & target splits are **row-level** → CONTRADICTED as compound-held-out |
| External benchmark construction (pool → exact-overlap-free → Tanimoto thresholds) | `scripts/07_evaluate_external.py` | `splits/external_test.csv`; funnel 2,930 → 2,562 → 2,484 / 2,431 / 2,256 | VERIFIED; "2,930 = overlap-excluded test set" → CONTRADICTED wording |
| Comparator evaluation (SynGlue, PDP, PDP-XGB, PROTAC-STAN, Morgan-RF, Tanimoto-NN, PROTAC-DC50, TACK) | `comparators/`; `scripts/07_evaluate_external.py` | `predictions/comparator_predictions.csv` | VERIFIED (per-comparator AUROC reproduced after applying each leakage rule) |
| Linker multiclass classification (4 classes) | `scripts/08_train_linker_classifier.py`; `configs/linker_classifier.yaml`; `scripts/model_code/multi_class_linker.py` | `predictions/linker_oof_predictions.csv`; `splits/linker_cv_folds.csv` | VERIFIED (model + data); protocol CONTRADICTED (test-AUC selection; "10-fold" = repeated fixed-test evaluation) |
| ADMET profiling | `pipeline/06_Generative_Pipeline_generator.py` (ADMET-AI bridge) | `environment/software_versions.json` (`admet_ai==1.4.0`) | VERIFIED package; manuscript cites OpenADMET → CONTRADICTED citation |
| Generative pipeline (Link-INVENT + TRIE) | `pipeline/06_Generative_Pipeline_generator.py`, `pipeline/03_TRIE_Index_*.py` | TRIE dictionary reference | DOCUMENTED_ONLY (not re-run here) |
| Target mapping / polypharmacology types | `pipeline/04_Target_Mapping_Target_Mapping.py` | — | DOCUMENTED_ONLY |

## 2. Results → code

| Manuscript result | Code / data | Status |
|---|---|---|
| DC50 R² random 0.76, scaffold 0.57, target 0.83 | `scripts/05_evaluate_regression.py` on the recovered metrics table | **VERIFIED** (0.7568 / 0.5665 / 0.8264) |
| Dmax R² random 0.82, scaffold 0.60, target 0.63 | same | **UNRESOLVED** — verified Dmax R² are 0.681 / 0.048 / 0.173; 0.8325 is a Pearson correlation and 0.5999 a different (both-model) test |
| External AUROC 0.76 at ≤100 nM | `scripts/07_evaluate_external.py` | **VERIFIED** (0.7556, n=71, three-token) |
| External AUROC 0.84 at ≤1 µM | same | **CONTRADICTED** — not reproducible (whole-PROTAC 0.666, n=1,700; three-token 0.550, n=71) |
| Linker median test ROC-AUC 0.80 ("ten-fold CV") | `scripts/08_train_linker_classifier.py --mode original/grouped` | **VERIFIED value / CONTRADICTED label** — 0.8013 test-selected; the ten-fold table is a repeated fixed-test evaluation (median 0.7975); grouped ChemBERTa AUROC 0.8035 |
| Endpoint model trained on 2,446 DC50 / 1,728 Dmax records | `scripts/06_check_data_leakage.py`; notebook log | **CONTRADICTED as the training population** — the tensor holds 142,727 rows / 12,843 DC50 / 8,520 Dmax valid labels (4,585 unique compounds); 2,446 / 1,728 are curated source records |
| External validation independent of training | `scripts/06_check_data_leakage.py` | **PARTIAL** — external overlap is 0 for the compound-level pipeline; the v4 retrain merged 19 DC₅₀ / 14 Dmax external compounds (32 / 31 rows) |

## 3. Figure → code

| Figure | Generator | Status |
|---|---|---|
| External funnel | `scripts/09_reproduce_figures.py::fig_external_funnel` | reproducible |
| Comparator AUROC | `::fig_comparator_auroc` | reproducible |
| Held-out R² by split | `::fig_split_r2` | reproducible |
| DC50 density/box (train vs external) | `::fig_activity_density` | reproducible |
| Linker ROC (grouped) | `::fig_linker_roc` | reproducible (ChemBERTa) |
| Partition independence | `::fig_leakage_independence` | reproducible |
| Fig 2f/2m (candidate ordering) | not located | **UNRESOLVED** — source values missing; deposited example artifact favours cmp2 on DC₅₀ |
| Fig S4 (linker panels) | `scripts/08…`, `plots/` | legend requires correction (repeated fixed-test evaluation, no Boruta) |
| Fig S5 (Transformer architecture) | `configs/transformer.yaml` | label 4-head baseline; 8-head variant separate |
| Fig S6 (external funnel) | `scripts/09…` | label 2,930 as pre-overlap pool |
| Fig S7 (generalization panels) | `scripts/05…` | replace Dmax values; mark random/target as row-level |
| Fig S9 (external comparators) | `scripts/07…` | correct the 0.84 claim; report per-comparator n |

## 4. Known manuscript↔code mismatches (summary)

1. **6,111 vs 6,110** unique SMILES (dedup rule unstated).
2. **Endpoint counts** mix rules: DC50 has `N.D.` removed, Dmax does not.
3. **2,320 "PROTACs"** is a row count of the linker-classification table (846 identifiers; 675 structures).
4. **2,749 "unique linker structures"** is an identifier count (2,494 unique structures).
5. **Attention heads** — 4 for the released model; 8 for the two-stage family.
6. **Training population** — 2,446 / 1,728 records ≠ the 142,727-row training tensor.
7. **Linker protocol** — test-AUC selection; no Boruta; RandomizedSearchCV; "10-fold" is a repeated fixed-test evaluation.
8. **ADMET** — `admet_ai` 1.4.0, not OpenADMET.
9. **AUROC 0.84 at ≤1 µM** — not reproducible.
10. **Dmax R² 0.82 / 0.60 / 0.63** — not reproducible as R².
11. **Random / target splits** are row-level (921 / 522 shared compounds).
12. **v4 leakage** — 19 DC₅₀ / 14 Dmax external compounds in training.
