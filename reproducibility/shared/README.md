# Shared resources

Utilities, environments, artifact hashes and the global index used by every model folder.

```
shared/
├── _common.py                     path/config resolution (SYNGLUE_ROOT, SYNGLUE_PY_ROOT, …)
├── ARTIFACTS.md                   artifact descriptions (checkpoints, RFs, classifier)
├── sha256_manifest.txt            computed SHA256 for every model artifact
├── mapped_index.csv               global 9,252-row index: compound_id, components, canonical SMILES
└── environment/
    ├── environment.lock.yml       pinned main environment
    ├── software_versions.json     observed versions + caveats
    ├── environment.yml            base conda environment
    ├── requirements-legacy.txt    legacy pins for archived pickles
    └── README.md                  environment notes (RF pickle generations, linker shim)
```

## Path resolution

`_common.py` finds the bundle root by walking upward for `MANUSCRIPT_TO_CODE.md`, then exposes:

| Variable | Default | Purpose |
|---|---|---|
| `SYNGLUE_ROOT` | `/storage/savi/saveenas/Projects/SynGlue` | project root |
| `SYNGLUE_PY_ROOT` | `/storage/savi/saveenas/Projects/SynGlue_Py` | models directory |
| `SYNGLUE_DATA_ROOT` | `server_53 …/Attention/data` | curated tables |
| `GROVER_DIR`, `GROVER_CKPT` | — | step 01 only |

`RESULTS` → `<bundle>/outputs/` and `FIGURES` → `<bundle>/figures/` (both runtime, git-ignored).

## Observed versions

| Package | Main environment | Legacy environment |
|---|---|---|
| Python | 3.10.20 | 3.10.20 |
| numpy | 2.2.6 | 1.21.5 |
| scikit-learn | 1.7.2 | 1.0.2 |
| scipy | 1.15.3 | 1.7.3 |
| pandas | 2.3.3 | — |
| matplotlib | 3.10.9 | — |
| torch | 2.12.1+cu130 | — |
| rdkit | 2024.09.6 | — |
| xgboost | 3.2.0 | — |

## Artifact integrity notes

- Two archived Zenodo copies of `multitask_transformer.pt` are **corrupt** (zip central-directory
  failure); the working copy is `630783ce…` (36 keys, 9,026,563 parameters).
- Two Random-Forest families exist (production `rf_*`, compound-level `stage2_rf_*`), both 500 trees /
  depth 20 / 512 features.
- The linker classifier requires a numpy random-state shim under scikit-learn ≤ 1.1.
