#!/usr/bin/env python3
"""_common.py — shared configuration and helpers for the SynGlue reproduction steps.

All steps import this module. Paths resolve from environment variables so the pipeline can run
outside the original workstation:

    SYNGLUE_ROOT   project root (default: /storage/savi/saveenas/Projects/SynGlue)
    SYNGLUE_PY_ROOT   models/artifacts root (default: /storage/savi/saveenas/Projects/SynGlue_Py)
    SYNGLUE_DATA_ROOT  curated data tables (default: <server_53 Attention/data path>)
    GROVER_DIR     patched GROVER checkout (required only for embedding extraction)
    GROVER_CKPT    GROVER checkpoint (default: $SYNGLUE_PY_ROOT/models/grover_fixed.pt)

Every step writes machine-readable outputs to ./results and figures to ./figures.
No results are committed; the folders exist for reproducibility runs.
"""
import os
import sys
import json
import hashlib

ROOT = os.environ.get("SYNGLUE_ROOT", "/storage/savi/saveenas/Projects/SynGlue")
PY_ROOT = os.environ.get("SYNGLUE_PY_ROOT", "/storage/savi/saveenas/Projects/SynGlue_Py")
DATA_ROOT = os.environ.get(
    "SYNGLUE_DATA_ROOT",
    "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data",
)
GROVER_DIR = os.environ.get("GROVER_DIR", "/tmp/ws1_grover/grover_patched")
GROVER_CKPT = os.environ.get("GROVER_CKPT", os.path.join(PY_ROOT, "models", "grover_fixed.pt"))

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """Walk upward until the bundle root (marked by MANUSCRIPT_TO_CODE.md) is found."""
    d = start
    for _ in range(6):
        if os.path.exists(os.path.join(d, "MANUSCRIPT_TO_CODE.md")):
            return d
        d = os.path.dirname(d)
    return start


REPO_ROOT = _find_repo_root(HERE)
RESULTS = os.path.join(REPO_ROOT, "outputs")   # runtime outputs (git-ignored)
FIGURES = os.path.join(REPO_ROOT, "figures")   # runtime figures (git-ignored)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---- artifact locations used across steps -------------------------------------------------
PATHS = {
    "training_table": os.path.join(DATA_ROOT, "Upadted_PROTAC_features.csv"),
    "warhead_table": os.path.join(DATA_ROOT, "Updated_warhead.csv"),
    "linker_table": os.path.join(DATA_ROOT, "linker_updated_features.csv"),
    "e3_table": os.path.join(DATA_ROOT, "E3_updated_features.csv"),
    "checkpoint_released": os.path.join(PY_ROOT, "models", "multitask_transformer.pt"),
    "checkpoint_compound_level": os.path.join(
        ROOT, "revision", "models", "final_protac_model", "protac_attention_model.pt"),
    "checkpoint_two_stage": os.path.join(ROOT, "revision", "models", "final_two_stage_transformer.pt"),
    "rf_dc50": os.path.join(PY_ROOT, "models", "rf_dc50.joblib"),
    "rf_dmax": os.path.join(PY_ROOT, "models", "rf_dmax.joblib"),
    "stage2_rf_dc50": os.path.join(ROOT, "revision", "models", "final_protac_model", "stage2_rf_dc50.joblib"),
    "stage2_rf_dmax": os.path.join(ROOT, "revision", "models", "final_protac_model", "stage2_rf_dmax.joblib"),
    "Z_all": os.path.join(ROOT, "revision", "models", "final_protac_model", "Z_all.npy"),
    "Z_train": os.path.join(ROOT, "revision", "models", "final_protac_model", "Z_train.npy"),
    "Z_test": os.path.join(ROOT, "revision", "models", "final_protac_model", "Z_test.npy"),
    "split_indices": os.path.join(ROOT, "revision", "models", "final_protac_model", "four_split_indices.npz"),
    "split_metrics": os.path.join(ROOT, "revision", "models", "final_protac_model", "four_split_generalization.csv"),
    "external_pool": os.path.join(ROOT, "revision", "benchmarking", "external_not_in_training.csv"),
    "external_predictions": os.path.join(
        ROOT, "revision", "rebuttal_analyses", "08_external_nonprotacdb", "outputs", "final_predictions.csv"),
    "external_funnel": os.path.join(
        ROOT, "revision", "rebuttal_analyses", "17_source_to_final", "final_external_tack_chiral.csv"),
    "linker_features_chemberta": os.path.join(
        ROOT, "Reproducibility", "Savi", "PROTAC", "PROTAC_featuriser", "featuresChemberta.csv"),
    "linker_classifier": os.path.join(
        ROOT, "Reproducibility", "Zenodo", "zenodo_upload",
        "06_Generative_Pipeline_models_linker_classifier.pkl"),
    "mapped_order": os.path.join(RESULTS, "mapped_compound_order.csv"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path, what=None):
    """Fail with a clear message when an input artifact is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing input: {path}\n"
            f"  ({what or 'required artifact'})\n"
            f"  set the corresponding environment variable (see _common.py header)."
        )
    return path


def save_json(obj, name):
    p = os.path.join(RESULTS, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", p)
    return p


def save_csv(df, name):
    p = os.path.join(RESULTS, name)
    df.to_csv(p, index=False)
    print("wrote", p)
    return p


def save_fig(fig, name):
    """Save a figure as both SVG (editable text) and PNG (300 dpi)."""
    import matplotlib
    matplotlib.use("Agg")
    svg = os.path.join(FIGURES, name + ".svg")
    png = os.path.join(FIGURES, name + ".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
    print("wrote", svg)
    print("wrote", png)
    return svg, png


SEED = 42
