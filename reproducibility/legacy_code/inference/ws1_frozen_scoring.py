#!/usr/bin/env python3
"""
WS1 steps 9-11 — FROZEN SynGlue scoring on external decomposed PROTACs.
========================================================================
Inputs: ws1_components.csv (decomposed) + ws1_ood_tiers.csv (tiers) + external endpoints
Model : FROZEN multitask_transformer.pt + rf_dc50.joblib + rf_dmax.joblib (hashes in WS1 spec)
Input : X[i] = [GROVER(warhead), GROVER(linker), GROVER(E3)] (4800-dim each)
Output: ws1_predictions.csv (per compound: components, tier, endpoints, predictions)
        ws1_metrics.md (R2/MAE/RMSE/Spearman rho + bootstrap 95% CI per tier/endpoint)
Usage: python ws1_frozen_scoring.py [--max_compounds N]
"""
import os, sys, subprocess, warnings, time, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"
GROVER_DIR = "/tmp/ws1_grover/grover_patched"  # patched copy (weights_only=False) of SynGlue_Py/repos/grover
GROVER_CKPT = os.path.join(SYNGLUE, "models", "grover_fixed.pt")
MODEL_CKPT = os.path.join(SYNGLUE, "models", "multitask_transformer.pt")
RF_DC50 = os.path.join(SYNGLUE, "models", "rf_dc50.joblib")
RF_DMAX = os.path.join(SYNGLUE, "models", "rf_dmax.joblib")
WORK = "/tmp/ws1_grover"
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, REV)
from reproduce_original_model import MultiTaskProtacModel  # exact original class (no pos_embed)

N_BOOT = 2000

# ─── 1. Build scoring table ──────────────────────────
print("Building scoring set...")
comp = pd.read_csv(os.path.join(OUT, "ws1_components_v2.csv"))
tiers = pd.read_csv(os.path.join(OUT, "ws1_ood_tiers.csv"))
ext = pd.read_csv(EXT)

sel = comp.merge(tiers[["canonical_smiles", "tier", "max_tanimoto"]], on="canonical_smiles", how="left")
# require clean single-fragment linker + fully decomposed + non-exact
sel = sel[sel["linker_smiles"].notna()
          & sel["linker_single_fragment"].fillna(False)
          & sel["e3_smiles"].notna()
          & sel["warhead_smiles"].notna()
          & (sel["tier"] != "exact_in_train")].copy()

# endpoints (all records per compound)
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "n_dc50": int((g["Value_Type"] == "DC50").sum()),
    "n_dmax": int((g["Value_Type"] == "Dmax").sum()),
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
    "dmax_censored": any(g.loc[g["Value_Type"] == "Dmax", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
    "cell_lines": ";".join(g["Cell_Line"].dropna().astype(str).unique()[:3]),
    "assay": ";".join(g["Assay"].dropna().astype(str).unique()[:2]),
}))
sel = sel.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                on="canonical_smiles", how="left")
sel = sel[sel["n_dc50"].fillna(0).gt(0) | sel["n_dmax"].fillna(0).gt(0)].reset_index(drop=True)
print(f"  scoring set: {len(sel)} compounds (non-exact, decomposed, with endpoints)")

# ─── 2. GROVER embedding ─────────────────────────────
def grover_embed(smiles_list, tag):
    """Embed SMILES via grover pipeline -> (n,4800) float32. Mirrors module_4 run_ai_predictions."""
    fn = os.path.join(WORK, f"{tag}_smiles.csv")
    pd.DataFrame({"smiles": smiles_list}).to_csv(fn, index=False)
    feat = os.path.join(WORK, f"{tag}_features.npz")
    fps = os.path.join(WORK, f"{tag}_fps.npz")
    subprocess.run(
        f"python {GROVER_DIR}/scripts/save_features.py --data_path {fn} --save_path {feat} "
        f"--features_generator rdkit_2d_normalized --restart",
        shell=True, check=True, capture_output=True)
    subprocess.run(
        f"python {GROVER_DIR}/main.py fingerprint --data_path {fn} --features_path {feat} "
        f"--checkpoint_path {GROVER_CKPT} --fingerprint_source both --output {fps}",
        shell=True, check=True, capture_output=True)
    return np.load(fps)["fps"][:, :4800]

# unique components
w_unique = sorted(sel["warhead_smiles"].dropna().unique())
l_unique = sorted(sel["linker_smiles"].dropna().unique())
e_unique = sorted(sel["e3_smiles"].dropna().unique())
print(f"  unique components: warhead {len(w_unique)}, linker {len(l_unique)}, E3 {len(e_unique)}")
print("Embedding warheads..."); t0 = time.time()
W = grover_embed(w_unique, "warhead"); print(f"    {time.time()-t0:.0f}s")
print("Embedding linkers..."); t0 = time.time()
L = grover_embed(l_unique, "linker"); print(f"    {time.time()-t0:.0f}s")
print("Embedding E3..."); t0 = time.time()
E = grover_embed(e_unique, "e3"); print(f"    {time.time()-t0:.0f}s")
w2i = {s: i for i, s in enumerate(w_unique)}
l2i = {s: i for i, s in enumerate(l_unique)}
e2i = {s: i for i, s in enumerate(e_unique)}

# ─── 3. Frozen transformer -> fused embeddings Z ──
print("Loading frozen transformer...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskProtacModel(input_dim=4800).to(device)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=device, weights_only=True))
model.eval()

n = len(sel)
X = np.zeros((n, 3, 4800), dtype=np.float32)
for i, r in sel.iterrows():
    X[i, 0] = W[w2i[r["warhead_smiles"]]]
    X[i, 1] = L[l2i[r["linker_smiles"]]]
    X[i, 2] = E[e2i[r["e3_smiles"]]]

Z = []
with torch.no_grad():
    for b in range(0, n, 64):
        xb = torch.tensor(X[b:b+64], dtype=torch.float32).to(device)
        h = model.proj(xb)
        h = model.transformer(h)
        attn = torch.softmax(model.attn_pool(h), dim=1)
        Z.append((h * attn).sum(dim=1).cpu().numpy())
Z = np.vstack(Z)

# Save intermediate artifacts for RF step (rf_predict env: numpy 1.21/sklearn 1.0.2)
np.save(os.path.join(OUT, "ws1_Z.npy"), Z)
sel.to_csv(os.path.join(OUT, "ws1_scoring_set.csv"), index=False)
print(f"Saved Z ({Z.shape}) + scoring set ({len(sel)} rows) -> {OUT}")
print("Next: run  ws1_rf_metrics.py  in env rf_predict (python -s)")
