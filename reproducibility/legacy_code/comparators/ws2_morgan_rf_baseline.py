#!/usr/bin/env python3
"""
Morgan-fingerprint Random Forest baseline (descriptor-ML control).
Trains on the SAME training data as SynGlue (full-PROTAC SMILES, log10 DC50 labels),
predicts on all 3,518 external PROTACs. Tests: does the component-aware transformer
add value over a plain fingerprint model?
Usage: conda run -n synglu_test_package python ws2_morgan_rf_baseline.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor
RDLogger.DisableLog("rdApp.*")

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
sys.path.insert(0, REV)
from reproduce_original_model import load_data

print("Loading training data...")
Xtr, y50tr, ymtr, metatr = load_data()
# unique (compound, components) rows with DC50 label
sig = pd.DataFrame({"w": [Xtr[i, 0].tobytes() for i in range(len(Xtr))],
                    "l": [Xtr[i, 1].tobytes() for i in range(len(Xtr))],
                    "e": [Xtr[i, 2].tobytes() for i in range(len(Xtr))]})
dup = sig.duplicated(keep="first")
metatr = metatr[~dup].reset_index(drop=True)
y50tr = y50tr[~dup]
mask = np.isfinite(y50tr)
tr_smiles = metatr["SMILES"].values[mask]
tr_y = y50tr[mask]
print(f"train: {len(tr_smiles)} labeled compounds")

def fps(smiles_list, n=2048):
    out, idx = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n))
            idx.append(i)
    return out, idx

print("Training fingerprints...")
tr_fp, tr_idx = fps(tr_smiles)
rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(np.array(tr_fp), tr_y[tr_idx])
print("RF trained")

ext = pd.read_csv("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv")
sc = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
ext_fp, ext_idx = fps(sc["canonical_SMILES"])
sc["morgan_rf_pred_logdc50"] = np.nan
sc.loc[ext_idx, "morgan_rf_pred_logdc50"] = rf.predict(np.array(ext_fp))
sc[["canonical_SMILES", "morgan_rf_pred_logdc50"]].to_csv(os.path.join(TIR, "morgan_rf_baseline.csv"), index=False)
print(f"predicted {len(ext_idx)}/{len(sc)} compounds -> morgan_rf_baseline.csv")
