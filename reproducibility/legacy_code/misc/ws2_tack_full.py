#!/usr/bin/env python3
"""
TACK (KDD'26) DC50 + Dmax XGBoost ensembles on the external set.
Loads each fold's datamodule from hparams+state (fitted encoders), featurizes
samples, predicts with the fold Booster, denormalizes, weighted-ensembles.
Outputs: benchmarking/ws1/tiers/tack_predictions.csv
Usage: conda run -n synglu_test_package python ws2_tack_external.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (driver/torch mismatch)
import sys, warnings, json, types
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --- shim broken optional deps (XGBoost-only path) ---
try:
    import curvlinops
    if not hasattr(curvlinops, "_base"):
        m = types.ModuleType("curvlinops._base")
        m._LinearOperator = type("_LinearOperator", (), {})
        sys.modules["curvlinops._base"] = m
except Exception:
    pass
try:
    import laplace
except Exception:
    fake = types.ModuleType("laplace")
    fake.Laplace = type("Laplace", (), {})
    sys.modules["laplace"] = fake

sys.path.insert(0, "/tmp/tack_repo")
import re, inspect, torch
import xgboost as xgb
from tackai.data.datamodule import DegradationComplexDataModule, load_config_from_yaml

def load_dm(hpath, spath):
    sig = inspect.signature(DegradationComplexDataModule.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    cfg = load_config_from_yaml(hpath)
    # translate old-style feature specs (model-release yaml) to constructor flags
    mf = str(cfg.get("mol_features", ""))
    if "fingerprint" in mf: cfg["use_fingerprints"] = True
    if "descriptor" in mf: cfg["use_descriptors"] = True
    pf = str(cfg.get("poi_features", ""))
    if "sequence" in pf: cfg["use_poi_sequence_embedding"] = True
    lf = str(cfg.get("ligase_features", ""))
    if "name" in lf: cfg["use_ligase_name_embedding"] = True
    cf = str(cfg.get("cell_features", ""))
    if "description" in cf: cfg["use_cell_description_embedding"] = True
    cfg = {k: v for k, v in cfg.items() if k in valid}
    dm = DegradationComplexDataModule(dataset=None, **cfg)
    dm.load_state_dict(torch.load(spath, map_location="cpu", weights_only=False))
    return dm

def fold_of(f):
    mm = re.search(r"fold=(\d+)", f)
    return int(mm.group(1)) if mm else None

EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
TACK_RAW = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet"
OUT = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws1/tiers/tack_predictions.csv"
MODELS = "/tmp/tack_models"

# ─── 1. External pool + sequences ────────────────────
ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
uniq = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
tack = pq.read_table(TACK_RAW).to_pandas()
tseq = tack.drop_duplicates("SMILES", keep="first")
held = tack.groupby("SMILES")["SMILES_Held_Out"].first()
m = uniq.merge(tseq, left_on="canonical_SMILES", right_on="SMILES", how="left", suffixes=("", "_t"))
m["held_out"] = m["canonical_SMILES"].map(held)

samples = []
for _, r in m.iterrows():
    samples.append({
        "SMILES": r["canonical_SMILES"],
        "Degradation_Target_Gene": str(r.get("Protein_of_Interest", "")),
        "Degradation_Target_Sequence": str(r["POI_Sequence"]) if pd.notna(r.get("POI_Sequence")) else None,
        "Recruiter_Gene": str(r.get("E3_Ligase", "")),
        "Recruiter_Sequence": str(r["Ligase_Sequence"]) if pd.notna(r.get("Ligase_Sequence")) else None,
        "Cell_Line_ID": str(r["Cell_Line_ID"]) if pd.notna(r.get("Cell_Line_ID")) else "Unknown",
        "Assay_Time": float(r["Assay_Time"]) if pd.notna(r.get("Assay_Time")) else 24.0,
        "Assay": str(r["Assay"]) if pd.notna(r.get("Assay")) else "Unknown",
    })
valid_idx = [i for i, s in enumerate(samples) if s["Degradation_Target_Sequence"] and s["Recruiter_Sequence"]]
valid = [samples[i] for i in valid_idx]
print(f"external (no PROTAC-DB): {len(uniq)} | samples with sequences: {len(valid)}")

def run_ensemble(task):
    """Return per-sample weighted-mean prediction (denormalized)."""
    ckpt_dir = os.path.join(MODELS, f"{task}_best_arch", "checkpoints")
    wfile = os.path.join(MODELS, f"{task}_best_arch", f"ensemble_weights_{task}_best_architecture.json")
    with open(wfile) as f:
        wdict = json.load(f)
    wdict = wdict.get("weights", wdict) if isinstance(wdict, dict) else wdict
    print(f"[{task}] weights: {len(wdict)} entries")

    all_files = os.listdir(ckpt_dir)
    dms = [f for f in all_files if f.endswith("_hparams.yaml")]
    sts = [f for f in all_files if f.endswith("_state.pt")]
    preds = np.zeros(len(valid))
    wsum = 0.0
    nfold = 0
    for fname in sorted(all_files):
        if not fname.startswith("model=xgboost_" + task) or not fname.endswith(".json"):
            continue
        fnum = fold_of(fname)
        hpath = os.path.join(ckpt_dir, next((f for f in dms if fold_of(f) == fnum), ""))
        spath = os.path.join(ckpt_dir, next((f for f in sts if fold_of(f) == fnum), ""))
        if not (os.path.exists(hpath) and os.path.exists(spath)):
            print(f"  skip fold {fnum}: hparams/state missing")
            continue
        dm = load_dm(hpath, spath)
        feats = dm.featurize_samples_batch(valid, return_tensor="xgb")
        X = np.vstack([f[0] for f in feats]).astype(np.float32)
        names = list(feats[0][1])
        booster = xgb.Booster()
        booster.load_model(os.path.join(ckpt_dir, fname))
        dmat = xgb.DMatrix(X, feature_names=names)
        raw = booster.predict(dmat)
        # quantile models -> use median column
        if raw.ndim > 1:
            raw = raw[:, raw.shape[1] // 2]
        lab_key = next((k for k in dm.label_transformers if k.lower() == task.lower()), None)
        inv = dm.label_transformers[lab_key].inverse_transform(raw.reshape(-1, 1))
        w = wdict.get(fname.replace(".json", ""), wdict.get(fname, 1.0 / len(wdict)))
        preds += w * inv.ravel()
        wsum += w

    return preds / max(wsum, 1e-9)

m["tack_pred_dc50_nM"] = np.nan
m["tack_pred_dmax_pct"] = np.nan
print("Running DC50 ensemble...")
dc50 = run_ensemble("dc50")
m.loc[m.index.isin([m.index[i] for i in valid_idx]), "tack_pred_dc50_nM"] = dc50
print("Running Dmax ensemble...")
dmax = run_ensemble("dmax")
m.loc[m.index.isin([m.index[i] for i in valid_idx]), "tack_pred_dmax_pct"] = dmax

keep = ["canonical_SMILES", "E3_Ligase", "POI_UniProt", "Database", "held_out",
        "tack_pred_dc50_nM", "tack_pred_dmax_pct"]
m[keep].to_csv(OUT, index=False)
print(f"\nSaved {OUT}: DC50 n={m['tack_pred_dc50_nM'].notna().sum()} | Dmax n={m['tack_pred_dmax_pct'].notna().sum()}")
print(m[["canonical_SMILES", "tack_pred_dc50_nM", "tack_pred_dmax_pct"]].head(5).to_string())
