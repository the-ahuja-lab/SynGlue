#!/usr/bin/env python3
"""
TACK extra ensembles — binary activity (XGBoost) + MLP DC50 + MLP Dmax (manual load).
Outputs: benchmarking/ws1/tiers/tack_extra_predictions.csv
Usage: conda run -n synglu_test_package python ws2_tack_extra.py
"""
import os, sys, types, warnings, re, inspect, json
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import curvlinops
    if not hasattr(curvlinops, "_base"):
        m = types.ModuleType("curvlinops._base"); m._LinearOperator = type("_LinearOperator", (), {})
        sys.modules["curvlinops._base"] = m
except Exception:
    pass
try:
    import laplace
except Exception:
    fake = types.ModuleType("laplace"); fake.Laplace = type("Laplace", (), {})
    sys.modules["laplace"] = fake
sys.path.insert(0, "/tmp/tack_repo")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import xgboost as xgb
from tackai.data.datamodule import DegradationComplexDataModule, load_config_from_yaml
from tackai.models.mlp_model import MLPModel

def load_dm(hpath, spath):
    sig = inspect.signature(DegradationComplexDataModule.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    cfg = load_config_from_yaml(hpath)
    mf = str(cfg.get("mol_features", ""))
    if "fingerprint" in mf: cfg["use_fingerprints"] = True
    if "descriptor" in mf: cfg["use_descriptors"] = True
    pf = str(cfg.get("poi_features", ""))
    if "sequence" in pf: cfg["use_poi_sequence_embedding"] = True
    if "onehot" in pf or "name" in pf: cfg["use_poi_name_embedding"] = True
    if "precomputed" in pf: cfg["use_poi_precomputed_embedding"] = True
    lf = str(cfg.get("ligase_features", ""))
    if "name" in lf: cfg["use_ligase_name_embedding"] = True
    cf = str(cfg.get("cell_features", ""))
    if "description" in cf: cfg["use_cell_description_embedding"] = True
    if "onehot" in cf: cfg["use_cell_name_embedding"] = True
    cfg = {k: v for k, v in cfg.items() if k in valid}
    dm = DegradationComplexDataModule(dataset=None, **cfg)
    dm.load_state_dict(torch.load(spath, map_location="cpu", weights_only=False))
    return dm

def fold_of(f):
    mm = re.search(r"fold=(\d+)", f)
    return int(mm.group(1)) if mm else None

def featurize_pt(dm, valid):
    out = dm.featurize_samples_batch(valid, return_tensor="pt")
    if isinstance(out, dict):
        return out
    batch = {}
    for k in out[0].keys():
        batch[k] = torch.stack([o[k] for o in out])
    return batch

EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
TACK_RAW = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet"
OUT = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws1/tiers/tack_extra_predictions.csv"
MODELS = "/tmp/tack_models"

ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
uniq = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
tack = pq.read_table(TACK_RAW).to_pandas()
tseq = tack.drop_duplicates("SMILES", keep="first")
held = tack.groupby("SMILES")["SMILES_Held_Out"].first()
m = uniq.merge(tseq, left_on="canonical_SMILES", right_on="SMILES", how="left", suffixes=("", "_t"))
m["held_out"] = m["canonical_SMILES"].map(held)
samples = [{"SMILES": r["canonical_SMILES"], "Degradation_Target_Gene": str(r.get("Protein_of_Interest", "")),
            "Degradation_Target_Sequence": str(r["POI_Sequence"]) if pd.notna(r.get("POI_Sequence")) else None,
            "Recruiter_Gene": str(r.get("E3_Ligase", "")), "Recruiter_Sequence": str(r["Ligase_Sequence"]) if pd.notna(r.get("Ligase_Sequence")) else None,
            "Cell_Line_ID": str(r["Cell_Line_ID"]) if pd.notna(r.get("Cell_Line_ID")) else "Unknown",
            "Assay_Time": float(r["Assay_Time"]) if pd.notna(r.get("Assay_Time")) else 24.0,
            "Assay": str(r["Assay"]) if pd.notna(r.get("Assay")) else "Unknown"} for _, r in m.iterrows()]
valid = [s for s in samples if s["Degradation_Target_Sequence"] and s["Recruiter_Sequence"]]
valid_idx = [i for i, s in enumerate(samples) if s["Degradation_Target_Sequence"] and s["Recruiter_Sequence"]]
print(f"samples with sequences: {len(valid)}")

def match_pair(fname, dms, sts, ckpt_dir):
    """Match hparams/state by fold number + data= config + label."""
    fnum = fold_of(fname)
    mcfg = re.search(r"data=([^ ]+?)_labels_", fname)
    cfg = mcfg.group(1) if mcfg else None
    mlab = re.search(r"labels_([A-Za-z0-9]+)-", fname)
    lab = mlab.group(1) if mlab else None
    def pick(files):
        cands = [f for f in files if fold_of(f) == fnum]
        if lab is not None:
            cands = [f for f in cands if lab in f]
        if cfg is not None:
            cands = [f for f in cands if cfg in f]
        return cands[0] if cands else ""
    return os.path.join(ckpt_dir, pick(dms)), os.path.join(ckpt_dir, pick(sts))

def xgb_ensemble(task_dir, prefix, label_key=None, take_proba=False):
    ckpt_dir = os.path.join(MODELS, task_dir, "checkpoints")
    allf = os.listdir(ckpt_dir)
    dms = [f for f in allf if f.endswith("_hparams.yaml")]
    sts = [f for f in allf if f.endswith("_state.pt")]
    mods = sorted(f for f in allf if f.startswith("model=xgboost_" + prefix) and f.endswith(".json"))
    wfile = os.path.join(MODELS, task_dir, [f for f in os.listdir(os.path.join(MODELS, task_dir)) if f.startswith("ensemble_weights")][0])
    wdict = json.load(open(wfile)).get("weights", {})
    preds = np.zeros(len(valid)); wsum = 0.0
    for fname in mods:
        fnum = fold_of(fname)
        hpath, spath = match_pair(fname, dms, sts, ckpt_dir)
        if not (os.path.exists(hpath) and os.path.exists(spath)):
            continue
        dm = load_dm(hpath, spath)
        feats = dm.featurize_samples_batch(valid, return_tensor="xgb")
        X = np.vstack([f[0] for f in feats]).astype(np.float32)
        names = list(feats[0][1])
        booster = xgb.Booster(); booster.load_model(os.path.join(ckpt_dir, fname))
        raw = booster.predict(xgb.DMatrix(X, feature_names=names))
        if raw.ndim > 1: raw = raw[:, raw.shape[1] // 2]
        if not take_proba:
            lab_key = next((k for k in dm.label_transformers if k.lower() == label_key.lower()), None)
            if lab_key is None:
                raw = np.clip(raw, 0, 1)  # binary logistic output
            else:
                raw = dm.label_transformers[lab_key].inverse_transform(raw.reshape(-1, 1)).ravel()
        w = wdict.get(fname.replace(".json", ""), 1.0 / len(wdict))
        preds += w * raw; wsum += w
    return preds / max(wsum, 1e-9)

def mlp_ensemble(task_dir, label_key, n_samples=1):
    ckpt_dir = os.path.join(MODELS, task_dir, "checkpoints")
    allf = os.listdir(ckpt_dir)
    dms = [f for f in allf if f.endswith("_hparams.yaml")]
    sts = [f for f in allf if f.endswith("_state.pt")]
    ckpts = sorted(f for f in allf if f.endswith(".ckpt"))
    wfile = os.path.join(MODELS, task_dir, [f for f in os.listdir(os.path.join(MODELS, task_dir)) if f.startswith("ensemble_weights")][0])
    wdict = json.load(open(wfile)).get("weights", {})
    preds = np.zeros(len(valid)); wsum = 0.0
    for ck in ckpts:
        fnum = fold_of(ck)
        hpath, spath = match_pair(ck, dms, sts, ckpt_dir)
        if not (os.path.exists(hpath) and os.path.exists(spath)):
            print(f"  skip MLP {os.path.basename(ck)[:60]} (no matching datamodule)")
            continue
        dm = load_dm(hpath, spath)
        sd = torch.load(os.path.join(ckpt_dir, ck), map_location="cpu", weights_only=False)
        hp = sd.get("hyper_parameters", {})
        input_dim = int(hp.get("input_dim", np.sum(list(dm.feature_dims.values()))))
        model = MLPModel(input_dim=input_dim,
                         hidden_dim=hp.get("hidden_dim", 256),
                         n_targets=hp.get("n_targets", 1),
                         mlp_depth=hp.get("mlp_depth", 2),
                         head_depth=hp.get("head_depth", 1),
                         dropout=hp.get("dropout", 0.1),
                         head_dropout=hp.get("head_dropout", 0.1),
                         task_type=hp.get("task_type", "point"),
                         activation=hp.get("activation", "silu"),
                         norm_type=hp.get("norm_type", None))
        sd_model = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd["state_dict"].items()}
        model.load_state_dict(sd_model)
        model.eval()
        batch = featurize_pt(dm, valid)
        with torch.no_grad():
            out = model(batch)  # (n, output_dim, n_targets)
            mu = out[:, 0, 0].numpy()  # DER gamma = mean
        lab_key = next((k for k in dm.label_transformers if k.lower() == label_key.lower()), None)
        if lab_key is not None:
            mu = dm.label_transformers[lab_key].inverse_transform(mu.reshape(-1, 1)).ravel()
        w = wdict.get(ck.replace(".ckpt", ""), wdict.get(ck, 1.0 / len(wdict)))
        preds += w * mu; wsum += w
    return preds / max(wsum, 1e-9)

m["tack_bin_proba"] = np.nan
m["tack_mlp_dc50_nM"] = np.nan
m["tack_mlp_dmax_pct"] = np.nan
loc = [m.index[i] for i in valid_idx]

# bin ensemble skipped: requires author-cluster Boltz POI embeddings
# (poi_embeddings_file=/proj/berzelius-2026-62/...boltz_embeddings...npz) not distributed.
print("Running TACK MLP DC50 (caruana)...")
m.loc[loc, "tack_mlp_dc50_nM"] = mlp_ensemble("dc50_caruana", "dc50")
m[["canonical_SMILES", "held_out", "tack_mlp_dc50_nM"]].to_csv(OUT, index=False)
print("  dc50 partial saved")

print("Running TACK MLP Dmax (caruana)...")
m.loc[loc, "tack_mlp_dmax_pct"] = mlp_ensemble("dmax_caruana", "dmax")

keep_cols = ["canonical_SMILES", "held_out", "tack_bin_proba", "tack_mlp_dc50_nM", "tack_mlp_dmax_pct"]
m[keep].to_csv(OUT, index=False)
print(f"Saved {OUT}: bin n={m['tack_bin_proba'].notna().sum()} | mlp_dc50 n={m['tack_mlp_dc50_nM'].notna().sum()} | mlp_dmax n={m['tack_mlp_dmax_pct'].notna().sum()}")
print(m[["canonical_SMILES", "tack_bin_proba", "tack_mlp_dc50_nM", "tack_mlp_dmax_pct"]].head(4).to_string())
