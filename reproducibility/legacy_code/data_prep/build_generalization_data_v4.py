#!/usr/bin/env python3
"""
Build v4 dataset — TRUE W/L/E decomposition on both sides:
- Training: W (matched warhead GROVER), L (TRUE linker GROVER from linker_updated_features.csv), E (matched E3 GROVER)
- External: W (matched by POI uniprot), L (TRUE decomposed linker GROVER), E (matched by E3 ligase)
- Plus Dmax-saturated (<=90%) filtered variant for the Dmax task
- Plus PCA fit on the COMBINED token space

Output: revision/benchmarking/generalization_data_v4/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT_DIR = f"{REV}/benchmarking/generalization_data_v4"
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
    print("Loading data...")
    protac_df = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False)
    warhead_df = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    linker_df = pd.read_csv(f"{DATA_DIR}/linker_updated_features.csv", low_memory=False)
    e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)

    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()

    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    l_cols = [c for c in linker_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    print(f"GROVER dims: W={len(w_cols)} L={len(l_cols)} E={len(e_cols)}")

    wh_by_uniprot, wh_by_target = {}, {}
    for _, r in warhead_df.iterrows():
        u = str(r.get('warhead_Uniprot', '')).strip().upper()
        t = str(r.get('warhead_Target', '')).strip().upper()
        if u and u != 'NAN': wh_by_uniprot.setdefault(u, r[w_cols].values.astype(np.float32))
        if t and t != 'NAN': wh_by_target.setdefault(t, r[w_cols].values.astype(np.float32))
    lk_by_id = {}
    for _, r in linker_df.iterrows():
        lid = r.get('linker_Compound_ID', None)
        if pd.notna(lid): lk_by_id.setdefault(str(lid).strip(), r[l_cols].values.astype(np.float32))
    e3_by_name = {}
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase', '')).strip().upper()
        if n and n != 'NAN': e3_by_name.setdefault(n, r[e_cols].values.astype(np.float32))

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
        rows.append({"W": wv, "L": lv, "E": ev, "DC50": dc50, "Dmax": dmax,
                     "SMILES": p.get('PROTAC_SMILES',''), "target_uniprot": u,
                     "target_name": t, "e3": en, "compound_id": p.get('PROTAC_Compound_ID','')})
    print(f"Labeled rows: {len(rows)}")

    meta = pd.DataFrame([{k: r[k] for k in ('SMILES','target_uniprot','target_name','e3','compound_id','DC50','Dmax')} for r in rows])
    meta['scaffold'] = meta['SMILES'].apply(murcko_scaffold)

    # ── DC50 task (all labels) ──
    m50 = meta['DC50'].notna()
    X50 = np.stack([np.stack([r['W'], r['L'], r['E']]) for r, k in zip(rows, m50) if k]).astype(np.float32)
    y50 = np.clip(meta.loc[m50,'DC50'].values, 1e-3, 1e5).astype(np.float32)
    y50 = np.log10(y50).astype(np.float32)
    meta50 = meta[m50].reset_index(drop=True)
    print(f"DC50: X={X50.shape}")

    # ── Dmax full task ──
    mm = meta['Dmax'].notna()
    Xm = np.stack([np.stack([r['W'], r['L'], r['E']]) for r, k in zip(rows, mm) if k]).astype(np.float32)
    ym = np.clip(meta.loc[mm,'Dmax'].values, 0, 100).astype(np.float32)
    metam = meta[mm].reset_index(drop=True)
    print(f"Dmax full: X={Xm.shape}")

    # ── Dmax filtered task (<= 90%, drop saturated) ──
    mf = (meta['Dmax'].notna()) & (meta['Dmax'] <= 90)
    Xmf = np.stack([np.stack([r['W'], r['L'], r['E']]) for r, k in zip(rows, mf) if k]).astype(np.float32)
    ymf = np.clip(meta.loc[mf,'Dmax'].values, 0, 100).astype(np.float32)
    metaf = meta[mf].reset_index(drop=True)
    print(f"Dmax <=90%: X={Xmf.shape} (drop {(~mf).sum()-0} saturated + missing)")

    # ── PCA fit on combined tokens ──
    tokens = np.concatenate([X50.reshape(-1, 4800), Xm.reshape(-1, 4800)], axis=0)
    pca = PCA(n_components=512, random_state=42)
    pca.fit(tokens)
    print(f"PCA explained var: {pca.explained_variance_ratio_.sum():.3f}")
    joblib.dump(pca, f"{OUT_DIR}/pca_512.joblib")

    def pca_transform(X):
        return pca.transform(X.reshape(-1, 4800)).reshape(X.shape[0], 3, 512).astype(np.float32)

    np.save(f"{OUT_DIR}/X_dc50_pca.npy", pca_transform(X50)); np.save(f"{OUT_DIR}/y_dc50.npy", y50)
    meta50.to_csv(f"{OUT_DIR}/metadata_dc50.csv", index=False)
    np.save(f"{OUT_DIR}/X_dmax_pca.npy", pca_transform(Xm)); np.save(f"{OUT_DIR}/y_dmax.npy", ym)
    metam.to_csv(f"{OUT_DIR}/metadata_dmax.csv", index=False)
    np.save(f"{OUT_DIR}/X_dmax90_pca.npy", pca_transform(Xmf)); np.save(f"{OUT_DIR}/y_dmax90.npy", ymf)
    metaf.to_csv(f"{OUT_DIR}/metadata_dmax90.csv", index=False)

    print(f"\nSaved to {OUT_DIR}")
    print(f"DC50 targets: {meta50['target_uniprot'].value_counts().head(6).to_dict()}")
    print(f"Dmax<=90 median: {np.median(ymf):.1f}")

if __name__ == '__main__':
    main()
