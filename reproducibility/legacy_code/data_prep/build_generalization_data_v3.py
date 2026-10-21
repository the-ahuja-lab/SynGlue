#!/usr/bin/env python3
"""
Build consistent-encoding dataset:
- W token: matched warhead GROVER (by POI uniprot) — SAME as external inference
- L token: FULL PROTAC GROVER embedding — SAME as external inference
- E token: matched E3 GROVER (by E3 ligase) — SAME as external inference

Training and external inference now use identical encoding conventions.
Requires GROVER fingerprints for all labeled training PROTACs.
Output: revision/benchmarking/generalization_data_v3/
"""
import os, sys, json, subprocess, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
GROVER_DIR = f"{BASE}/../SynGlue_Py/repos/grover"
MODEL_DIR = f"{BASE}/../SynGlue_Py/models"
OUT_DIR = f"{REV}/benchmarking/generalization_data_v3"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def _parse(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.", "N/D", "NA", "N/A", ""): return np.nan
    if s.startswith(">"): s = s[1:]
    elif s.startswith("<"): s = s[1:]
    vals = []
    for p in s.split("/"):
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", ""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or None
    except:
        return None

def main():
    print("Loading training data...")
    protac_df = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False)
    warhead_df = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)

    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()

    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    wh_by_uniprot, wh_by_target, e3_by_name = {}, {}, {}
    for _, r in warhead_df.iterrows():
        u = str(r.get('warhead_Uniprot', '')).strip().upper()
        t = str(r.get('warhead_Target', '')).strip().upper()
        if u and u != 'NAN': wh_by_uniprot.setdefault(u, r[w_cols].values.astype(np.float32))
        if t and t != 'NAN': wh_by_target.setdefault(t, r[w_cols].values.astype(np.float32))
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase', '')).strip().upper()
        if n and n != 'NAN': e3_by_name.setdefault(n, r[e_cols].values.astype(np.float32))

    # Collect labeled compounds
    rows = []
    for _, p in protac_df.iterrows():
        u = str(p.get('warhead_Uniprot', '')).strip().upper()
        t = str(p.get('warhead_Target', '')).strip().upper()
        wv = wh_by_uniprot.get(u)
        if wv is None: wv = wh_by_target.get(t)
        if wv is None: continue
        en = str(p.get('E3 ligase', '')).strip().upper()
        ev = e3_by_name.get(en)
        if ev is None: continue
        dc50 = _parse(p.get('DC50 (nM)', np.nan))
        dmax = _parse(p.get('Dmax (%)', np.nan))
        if pd.isna(dc50) and pd.isna(dmax): continue
        rows.append({"W": wv, "E": ev, "DC50": dc50, "Dmax": dmax,
                     "SMILES": p.get('PROTAC_SMILES',''),
                     "target_uniprot": u, "target_name": t, "e3": en,
                     "compound_id": p.get('PROTAC_Compound_ID','')})
    print(f"Labeled compounds: {len(rows)}")

    # GROVER fps for full PROTAC SMILES
    smiles_list = [r['SMILES'] for r in rows]
    fps_file = f"{OUT_DIR}/train_full_protac_fps.npz"
    if os.path.exists(fps_file):
        fps = np.load(fps_file)['fps'][:, :4800]
        print(f"Loaded cached fps: {fps.shape}")
    else:
        print("Extracting GROVER fingerprints for training PROTACs...")
        smi_file = f"{OUT_DIR}/train_smiles.csv"
        pd.DataFrame({'smiles': smiles_list}).to_csv(smi_file, index=False)
        feats = f"{OUT_DIR}/train_features.npz"
        t0 = time.time()
        grover_env = os.environ.copy()
        grover_env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        r1 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/scripts/save_features.py "
            f"--data_path {smi_file} --save_path {feats} --features_generator rdkit_2d_normalized --restart",
            shell=True, capture_output=True, text=True, timeout=3600, env=grover_env)
        if r1.returncode != 0:
            print(f"save_features failed: {r1.stderr[-300:]}"); sys.exit(1)
        r2 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/main.py fingerprint "
            f"--data_path {smi_file} --features_path {feats} "
            f"--checkpoint_path {MODEL_DIR}/grover_fixed.pt --fingerprint_source both --output {fps_file}",
            shell=True, capture_output=True, text=True, timeout=7200, env=grover_env)
        if r2.returncode != 0:
            print(f"GROVER failed: {r2.stderr[-300:]}"); sys.exit(1)
        fps = np.load(fps_file)['fps'][:, :4800]
        print(f"FPs: {fps.shape} ({time.time()-t0:.0f}s)")

    assert len(fps) == len(rows), f"fps {len(fps)} != rows {len(rows)}"

    # Assemble full-dim X per task
    for task, key in [("dc50", "DC50"), ("dmax", "Dmax")]:
        keep = [i for i, r in enumerate(rows) if pd.notna(r[key])]
        X = np.zeros((len(keep), 3, 4800), dtype=np.float32)
        for j, i in enumerate(keep):
            X[j, 0, :] = rows[i]['W']
            X[j, 1, :] = fps[i]          # full PROTAC embedding
            X[j, 2, :] = rows[i]['E']
        y = np.array([rows[i][key] for i in keep], dtype=np.float32)
        if task == 'dc50':
            y = np.clip(y, 1e-3, 1e5)
            y = np.log10(y)
        else:
            y = np.clip(y, 0, 100)
        print(f"{task}: X={X.shape}, y=[{y.min():.3f}, {y.max():.3f}]")
        np.save(f"{OUT_DIR}/X_{task}_full.npy", X)
        np.save(f"{OUT_DIR}/y_{task}.npy", y)
        meta = pd.DataFrame([{**{k: rows[i][k] for k in ('SMILES','target_uniprot','target_name','e3','compound_id')},
                             **{key: rows[i][key]}} for i in keep])
        meta['scaffold'] = meta['SMILES'].apply(murcko_scaffold)
        meta.to_csv(f"{OUT_DIR}/metadata_{task}.csv", index=False)

    # PCA on DC50 tokens
    Xdc = np.load(f"{OUT_DIR}/X_dc50_full.npy")
    pca = PCA(n_components=512, random_state=42)
    pca.fit(Xdc.reshape(-1, 4800))
    print(f"PCA explained var: {pca.explained_variance_ratio_.sum():.3f}")
    joblib.dump(pca, f"{OUT_DIR}/pca_512.joblib")
    for task in ["dc50", "dmax"]:
        Xf = np.load(f"{OUT_DIR}/X_{task}_full.npy")
        Xp = pca.transform(Xf.reshape(-1, 4800)).reshape(Xf.shape[0], 3, 512).astype(np.float32)
        np.save(f"{OUT_DIR}/X_{task}_pca.npy", Xp)
    print(f"\nSaved to {OUT_DIR}")

if __name__ == '__main__':
    main()
