# Environment notes for reproducibility

## 1. Main environment (`environment.yml`)
Used by the training, inference, analysis and plotting scripts. Verified working combination in the
audit: Python 3.10, torch 2.12 (CPU), scikit-learn 1.7.2, numpy 2.2.6, rdkit 2024.09.

## 2. Random-Forest artifacts — two pickle generations

| Artifact family | Pickled with | Loads under modern env? |
|---|---|---|
| `rf_dc50.joblib`, `rf_dmax.joblib` (production, `SynGlue_Py/models/`) | scikit-learn 1.2.2 | **yes**, with `InconsistentVersionWarning` |
| `stage2_rf_dc50.joblib`, `stage2_rf_dmax.joblib` (compound-level pipeline) | older (pre-1.2 node dtype) | **no** (`ValueError: node array … incompatible dtype`) → needs the legacy environment |

Legacy environment (`requirements-legacy.txt`) for the `stage2_*` family:

```
numpy==1.21.5
scikit-learn==1.0.2
joblib==1.1.0
scipy==1.7.3
pandas==1.3.5
```

## 3. Linker-classifier artifact
`linker_classifier.pkl` (md5 `d5204351e9c42dbfbe1bc108a478d09f`) could not be loaded directly in
either available environment:

| Environment | Observed error |
|---|---|
| scikit-learn 1.7.2 / numpy 2.2.6 | `ModuleNotFoundError: sklearn.ensemble._gb_losses` |
| scikit-learn 1.0.2 / numpy 1.21.5 | `TypeError: __randomstate_ctor() takes from 0 to 1 positional arguments but 2 were given` |
| scikit-learn 1.0.2 / numpy 1.26 | `ValueError: numpy.dtype size changed` (ABI mismatch) |

**Working load path used in the audit** (scikit-learn 1.0.2 + numpy 1.21.5 + the shim below).
Note the two incompatibilities are independent: the missing `sklearn.ensemble._gb_losses` module
requires scikit-learn ≤ 1.1, while the `__randomstate_ctor` arity error requires a newer numpy —
the shim bridges the second without changing the first.

```python
import numpy.random._pickle as npp
_orig = npp.__randomstate_ctor
npp.__randomstate_ctor = lambda *a, **k: (_orig(a[0]) if a else _orig())
import joblib
model = joblib.load("linker_classifier.pkl")
```
This recovers the fitted model (4,800 features; 4 classes; GradientBoosting 228/7/9/13/log2/0.05).
For a clean, shim-free reproduction, re-serialise the model under a pinned modern environment.

## 4. ADMET (`admet.yml`)
`admet_ai==1.4.0` in a dedicated environment; the generator invokes it through a subprocess bridge.

## 5. Comparator environments
Each comparator runner expects its own environment (e.g. `pdp`, `protac-stan`, `protacdc50`,
`DeLinker-env`, `pgmg`, `shapelinker`). See the header docstring of each script in
`comparators/`.

## 6. GROVER embeddings
Embeddings are produced with a patched GROVER checkout:

```bash
python <grover>/scripts/save_features.py --data_path <smiles.csv> --save_path <features.npz> \
       --features_generator rdkit_2d_normalized --restart
python <grover>/main.py fingerprint --data_path <smiles.csv> --features_path <features.npz> \
       --checkpoint_path <grover_fixed.pt> --fingerprint_source both --output <fps.npz>
```
The resulting vectors are 4,800-dimensional. Pin the GROVER commit and checkpoint hash for
reproducibility.
