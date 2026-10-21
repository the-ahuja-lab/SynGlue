# Model artifacts

Hashes are computed from the working copies; no binaries are stored in this repository.

| Artifact | Role | SHA256 (prefix) | Source path |
|---|---|---|---|
| `multitask_transformer.pt` | released endpoint encoder (4 heads, 2 layers, 9,026,563 params) | `630783ce2f6f26e7…` | `/storage/savi/saveenas/Projects/SynGlue_Py/models/multitask_transformer.pt` |
| `protac_attention_model.pt` | compound-level pipeline encoder (32 keys, 8,894,978 params) | `3574b3b79bb39882…` | `/storage/savi/saveenas/Projects/SynGlue/revision/models/final_protac_model/protac_attention_model.pt` |
| `rf_dc50.joblib` | production DC50 RF (500/20/512) | `fc17095499b376db…` | `/storage/savi/saveenas/Projects/SynGlue_Py/models/rf_dc50.joblib` |
| `rf_dmax.joblib` | production Dmax RF (500/20/512) | `5df8d3f0f5346f98…` | `/storage/savi/saveenas/Projects/SynGlue_Py/models/rf_dmax.joblib` |
| `stage2_rf_dc50.joblib` | compound-level DC50 RF (500/20/512) | `ddd93eaddd4bff81…` | `/storage/savi/saveenas/Projects/SynGlue/revision/models/final_protac_model/stage2_rf_dc50.joblib` |
| `stage2_rf_dmax.joblib` | compound-level Dmax RF (500/20/512) | `f42798532c0269d2…` | `/storage/savi/saveenas/Projects/SynGlue/revision/models/final_protac_model/stage2_rf_dmax.joblib` |
| `grover_fixed.pt` | GROVER featuriser checkpoint | `6256228ac5b80ce9…` | `/storage/savi/saveenas/Projects/SynGlue_Py/models/grover_fixed.pt` |
| `linker_classifier.pkl` | linker multiclass classifier (4,800 features) | `8796a89259da25d4…` | `/storage/savi/saveenas/Projects/SynGlue/Reproducibility/Zenodo/zenodo_upload/06_Generative_Pipeline_models_linker_classifier.pkl` |

## Notes
- Two Random-Forest families exist (production `rf_*` and compound-level `stage2_rf_*`); both are 500 trees / depth 20 / 512 features.
- Three endpoint checkpoint families exist: released (4-head), compound-level (4-head), two-stage (8-head).
- The two archived Zenodo copies of `multitask_transformer.pt` are corrupt (zip central-directory failure); the working copy is listed above.
- The linker classifier loads only with the numpy random-state shim documented in `environment/software_versions.json`.
