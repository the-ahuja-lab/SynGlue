#!/usr/bin/env python3
"""
Build extended component-aware datasets — SEPARATE DC50 and Dmax tasks
to maximize sample count, plus PCA-reduced features.

- DC50 task: all PROTACs with DC50 label (log10 nM)
- Dmax task: all PROTACs with Dmax label (%)

Output: revision/benchmarking/generalization_data_v2/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT_DIR = f"{REV}/benchmarking/generalization_data_v2"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.decomposition import PCA

def _parse(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.", "N/D", "NA", "N/A", ""): return np.nan
    if s.startswith(">"): s = s[1:]
    elif s.startswith("<"): s = s[1:]
    parts = s.split("/")
    vals = []
    for p in parts:
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", ""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return sc if sc else None
    except:
        return None

def main():
    print("Loading data...")
    protac_df  = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False)
    warhead_df = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    linker_df  = pd.read_csv(f"{DATA_DIR}/linker_updated_features.csv", low_memory=False)
    e3_df      = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)

    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()

    def prefix_grover(df, prefix):
        return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df  = prefix_grover(linker_df, "L_")
    e3_df      = prefix_grover(e3_df, "E_")

    w_grover = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_grover = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_grover = [c for c in e3_df.columns if c.startswith("E_Grover_")]

    wh_by_uniprot, wh_by_target = {}, {}
    for _, r in warhead_df.iterrows():
        u = str(r.get('warhead_Uniprot', '')).strip().upper()
        t = str(r.get('warhead_Target', '')).strip().upper()
        if u and u != 'NAN': wh_by_uniprot.setdefault(u, r[w_grover].values.astype(np.float32))
        if t and t != 'NAN': wh_by_target.setdefault(t, r[w_grover].values.astype(np.float32))
    lk_by_id = {}
    for _, r in linker_df.iterrows():
        lid = r.get('linker_Compound_ID', None)
        if pd.notna(lid): lk_by_id.setdefault(str(lid).strip(), r[l_grover].values.astype(np.float32))
    e3_by_name = {}
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase', '')).strip().upper()
        if n and n != 'NAN': e3_by_name.setdefault(n, r[e_grover].values.astype(np.float32))

    # Collect ALL rows with any label
    rows = []
    for _, p in protac_df.iterrows():
        u = str(p.get('warhead_Uniprot', '')).strip().upper()
        t = str(p.get('warhead_Target', '')).strip().upper()
        wv = wh_by_uniprot.get(u)
        if wv is None: wv = wh_by_target.get(t)
        if wv is None: continue
        lid = str(p.get('Linker ID', '')).strip()
        lv = lk_by_id.get(lid)
        if lv is None: continue
        en = str(p.get('E3 ligase', '')).strip().upper()
        ev = e3_by_name.get(en)
        if ev is None: continue
        dc50 = _parse(p.get('DC50 (nM)', np.nan))
        dmax = _parse(p.get('Dmax (%)', np.nan))
        if pd.isna(dc50) and pd.isna(dmax): continue
        rows.append({
            "W_vec": wv, "L_vec": lv, "E_vec": ev,
            "DC50": dc50, "Dmax": dmax,
            "SMILES": p.get('PROTAC_SMILES', ''),
            "target_uniprot": u, "target_name": t,
            "e3": en, "linker_id": lid,
            "compound_id": p.get('PROTAC_Compound_ID', ''),
        })
    print(f"Total rows with any label: {len(rows)}")

    meta = pd.DataFrame([{k: r[k] for k in ('SMILES','target_uniprot','target_name','e3','linker_id','compound_id','DC50','Dmax')} for r in rows])
    meta['scaffold'] = meta['SMILES'].apply(murcko_scaffold)

    # ── DC50 task ──
    dc50_mask = meta['DC50'].notna()
    X_dc50 = np.stack([np.stack([r['W_vec'], r['L_vec'], r['E_vec']]) for r, keep in zip(rows, dc50_mask) if keep]).astype(np.float32)
    y_dc50_raw = meta.loc[dc50_mask, 'DC50'].values.astype(np.float32)
    # clean: clip zeros to 1e-3, cap at 1e5
    y_dc50_raw = np.clip(y_dc50_raw, 1e-3, 1e5)
    y_dc50_log = np.log10(y_dc50_raw)
    meta_dc50 = meta[dc50_mask].reset_index(drop=True)
    print(f"DC50 task: X={X_dc50.shape}, y range=[{y_dc50_log.min():.2f}, {y_dc50_log.max():.2f}]")

    # ── Dmax task ──
    dmax_mask = meta['Dmax'].notna()
    X_dmax = np.stack([np.stack([r['W_vec'], r['L_vec'], r['E_vec']]) for r, keep in zip(rows, dmax_mask) if keep]).astype(np.float32)
    y_dmax_raw = meta.loc[dmax_mask, 'Dmax'].values.astype(np.float32)
    y_dmax_raw = np.clip(y_dmax_raw, 0, 100)
    meta_dmax = meta[dmax_mask].reset_index(drop=True)
    print(f"Dmax task: X={X_dmax.shape}, y range=[{y_dmax_raw.min():.0f}, {y_dmax_raw.max():.0f}]")

    # ── PCA fit on concatenated training tokens (fit on DC50 task tokens) ──
    # Flatten W/L/E tokens to fit PCA on the 4800-dim space
    print("\nFitting PCA on training tokens...")
    tokens = X_dc50.reshape(-1, 4800)
    N_COMP = 512
    pca = PCA(n_components=N_COMP, random_state=42)
    pca.fit(tokens)
    print(f"  PCA: {N_COMP} components, explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # Apply to both tasks (each token independently)
    X_dc50_pca = pca.transform(X_dc50.reshape(-1, 4800)).reshape(X_dc50.shape[0], 3, N_COMP).astype(np.float32)
    X_dmax_pca = pca.transform(X_dmax.reshape(-1, 4800)).reshape(X_dmax.shape[0], 3, N_COMP).astype(np.float32)
    print(f"  PCA'd: DC50 {X_dc50_pca.shape}, Dmax {X_dmax_pca.shape}")

    # Save
    np.save(f"{OUT_DIR}/X_dc50_pca.npy", X_dc50_pca)
    np.save(f"{OUT_DIR}/y_dc50_log.npy", y_dc50_log)
    meta_dc50.to_csv(f"{OUT_DIR}/metadata_dc50.csv", index=False)
    np.save(f"{OUT_DIR}/X_dmax_pca.npy", X_dmax_pca)
    np.save(f"{OUT_DIR}/y_dmax_raw.npy", y_dmax_raw)
    meta_dmax.to_csv(f"{OUT_DIR}/metadata_dmax.csv", index=False)

    # Save PCA + full-dim arrays too (for external transform)
    np.save(f"{OUT_DIR}/X_dc50_full.npy", X_dc50)
    np.save(f"{OUT_DIR}/X_dmax_full.npy", X_dmax)
    import joblib
    joblib.dump(pca, f"{OUT_DIR}/pca_512.joblib")

    # scalers (log-space mean/std for DC50; raw for Dmax)
    scalers = {
        "dc50": {"mean": float(y_dc50_log.mean()), "std": float(y_dc50_log.std())},
        "dmax": {"mean": float(y_dmax_raw.mean()), "std": float(y_dmax_raw.std())},
    }
    json.dump(scalers, open(f"{OUT_DIR}/scalers.json", "w"))
    print(f"\nSaved to {OUT_DIR}")
    print(f"DC50: μ={scalers['dc50']['mean']:.3f} σ={scalers['dc50']['std']:.3f} (log10)")
    print(f"Dmax: μ={scalers['dmax']['mean']:.1f} σ={scalers['dmax']['std']:.1f}")
    print("\nTargets DC50:", meta_dc50['target_uniprot'].value_counts().head(8).to_dict())
    print("E3 DC50:", meta_dc50['e3'].value_counts().head(5).to_dict())

if __name__ == '__main__':
    main()
