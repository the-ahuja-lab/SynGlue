#!/usr/bin/env python3
"""
WS1/WS2 — SynGlue WHOLE-PROTAC mode on ALL external compounds.
Input: GROVER(full PROTAC SMILES) repeated in the 3-token window
  X[i] = [GROVER(full), GROVER(full), GROVER(full)]  (no decomposition needed)
Models: frozen (multitask_transformer.pt + rf_*), generalized (5-fold ensemble,
  per-fold RFs regenerated deterministically).
Outputs: benchmarking/ws1/tiers/synglue_wholeprotac_predictions.csv
Usage: conda run -n synglu_test_package python ws2_synglue_wholeprotac.py
"""
import os, sys, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
MODEL_DIR = os.path.join(REV, "models")
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"
GROVER_CKPT = os.path.join(SYNGLUE, "models", "grover_fixed.pt")
GROVER_PATCHED = "/tmp/ws1_grover/grover_patched"
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
os.makedirs(TIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, REV)
from reproduce_original_model import MultiTaskProtacModel, load_data

# ─── 1. Full external set ────────────────────────────
ext = pd.read_csv(EXT)
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
}))
sc = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
sc = sc.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
              left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
print(f"external unique: {len(sc)}")

# ─── 2. GROVER embed full PROTACs (whole-molecule mode) ───
cache = os.path.join(TIR, "emb_whole")
os.makedirs(cache, exist_ok=True)
fps_path = os.path.join(cache, "full_fps.npz")
if os.path.exists(fps_path):
    F = np.load(fps_path)["fps"][:, :4800]
    assert F.shape[0] == len(sc), f"cache mismatch {F.shape[0]} vs {len(sc)}"
else:
    fn = os.path.join(cache, "full_smiles.csv")
    pd.DataFrame({"smiles": sc["canonical_SMILES"]}).to_csv(fn, index=False)
    feat = os.path.join(cache, "full_features.npz")
    fps = fps_path
    subprocess.run(f"python {GROVER_PATCHED}/scripts/save_features.py --data_path {fn} "
                   f"--save_path {feat} --features_generator rdkit_2d_normalized --restart",
                   shell=True, check=True, capture_output=True)
    subprocess.run(f"python {GROVER_PATCHED}/main.py fingerprint --data_path {fn} "
                   f"--features_path {feat} --checkpoint_path {GROVER_CKPT} "
                   f"--fingerprint_source both --output {fps}",
                   shell=True, check=True, capture_output=True)
    F = np.load(fps)["fps"][:, :4800]
print(f"full-PROTAC GROVER embeddings: {F.shape}")

Xe = np.repeat(F[:, None, :], 3, axis=1).astype(np.float32)  # (n,3,4800)

def extract(model, X, batch=64):
    model.eval(); Z = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            h = model.proj(torch.tensor(X[b:b+batch], dtype=torch.float32).to(DEVICE))
            h = model.transformer(h)
            attn = torch.softmax(model.attn_pool(h), dim=1)
            Z.append((h * attn).sum(dim=1).cpu().numpy())
    return np.vstack(Z)

# ─── 3. Regenerate fold splits (generalized model) ───
print("Regenerating training folds...")
Xtr, y50tr, ymtr, metatr = load_data()
sig = pd.DataFrame({"w": [Xtr[i, 0].tobytes() for i in range(len(Xtr))],
                    "l": [Xtr[i, 1].tobytes() for i in range(len(Xtr))],
                    "e": [Xtr[i, 2].tobytes() for i in range(len(Xtr))]})
dup = sig.duplicated(keep="first")
Xtr = Xtr[~dup]; y50tr = y50tr[~dup]; ymtr = ymtr[~dup]; metatr = metatr[~dup].reset_index(drop=True)
scaff = []
for s in metatr["SMILES"].values:
    m = Chem.MolFromSmiles(str(s))
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        scaf = None
    scaff.append(scaf or f"noscaf_{len(scaff)}")
groups = np.array(scaff)
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
folds = list(gss.split(Xtr, groups=groups))

def make_loader(idx, shuffle=False):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(Xtr[idx], dtype=torch.float32),
                       torch.tensor(y50tr[idx], dtype=torch.float32),
                       torch.tensor(ymtr[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=64, shuffle=shuffle)

# ─── 4. Frozen model predictions ─────────────────────
print("Frozen transformer -> Z...")
fz = MultiTaskProtacModel(input_dim=4800).to(DEVICE)
fz.load_state_dict(torch.load(os.path.join(SYNGLUE, "models", "multitask_transformer.pt"),
                              map_location=DEVICE, weights_only=True))
Zfz = extract(fz, Xe)
np.save(os.path.join(cache, "Z_frozen_whole.npy"), Zfz)

# ─── 5. Generalized ensemble (per-fold RFs) ──────────
print("Generalized ensemble...")
pe50, pem, Ze_all = [], [], []
for k in range(5):
    m = MultiTaskProtacModel(input_dim=4800).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"generalized_transformer_fold{k}.pt"),
                                 map_location=DEVICE, weights_only=True)["state_dict"])
    m.eval()
    tr_i, _ = folds[k]
    tr_g = groups[tr_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42 + k)
    tr2, _ = next(iter(gss2.split(np.arange(len(tr_i)), groups=tr_g)))
    tr_idx = tr_i[tr2]
    Ztr = extract(m, Xtr[tr_idx])
    mask50, maskmax = np.isfinite(y50tr[tr_idx]), np.isfinite(ymtr[tr_idx])
    rf50 = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
    rf50.fit(Ztr[mask50], y50tr[tr_idx][mask50])
    rfmax = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
    rfmax.fit(Ztr[maskmax], ymtr[tr_idx][maskmax])
    Zk = extract(m, Xe)
    Ze_all.append(Zk)
    pe50.append(rf50.predict(Zk)); pem.append(rfmax.predict(Zk))
    print(f"  fold {k} done")

sc["gen_whole_pred_dc50_nM"] = 10 ** np.mean(pe50, axis=0)
sc["gen_whole_pred_dmax_pct"] = np.mean(pem, axis=0)
np.save(os.path.join(cache, "Z_gen_whole.npy"), np.stack(Ze_all))

# frozen RFs need sklearn 1.0.2 env -> save Z + table, predict separately
sc[["canonical_SMILES", "E3_Ligase", "POI_UniProt", "dc50_median", "dmax_max", "dc50_censored",
    "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]].to_csv(
    os.path.join(TIR, "synglue_wholeprotac_gen.csv"), index=False)
np.save(os.path.join(cache, "Z_frozen_whole.npy"), Zfz)
sc[["canonical_SMILES"]].to_csv(os.path.join(cache, "scoring_ids.csv"), index=False)
print(f"Saved: gen predictions ({len(sc)} rows) + Z for frozen RF step")
print("Next: frozen RF predictions -> ws2_frozen_rf_whole.py (rf_predict env)")
