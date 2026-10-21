# Legacy / role-based code library

The full SynGlue code library, organised by role rather than by model. These files are kept so that
every analysis, plot, data-preparation and comparator script lives in one place; the ordered
reproduction path is in `../endpoint_regression/`, `../linker_multiclass/`,
`../external_benchmark/` and `../figures/`.

| Folder | Contents |
|---|---|
| `model_code/` | model and training scripts (masked multitask trainer, attention model, two-stage, XGBoost, multi-class linker, reproduction scripts) |
| `analysis/` | metrics, OOD tiers, split performance, interpretability, audit scripts |
| `data_prep/` | dataset builders, component mapping, tensor construction, split export, decomposition |
| `inference/` | frozen scoring, whole-PROTAC scoring, external scoring entry points |
| `comparators/` | PDP, PDP-XGB, PROTAC-STAN, PROTAC-DC50, TACK, Morgan-RF, Tanimoto-NN runners |
| `pipeline/` | TRIE index, target mapping, generative pipeline, embedding benchmark |
| `plots/` | figure and panel generators (venns, heatmaps, SI composites, box plots) |
| `misc/` | smaller utility scripts not matching a primary pattern |

`MANIFEST.csv` at the bundle root maps every file to its original repository-relative path with a
SHA256, so each script is traceable to the exact source revision.
