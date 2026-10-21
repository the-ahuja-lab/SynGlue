#!/usr/bin/env python3
"""
Build full component-aware training tensors with metadata for generalization splits.

For each training PROTAC with DC50 AND Dmax labels:
  - W_vec: GROVER embedding of warhead (matched by warhead_Uniprot/Target)
  - L_vec: GROVER embedding of linker (matched by linker_Compound_ID)
  - E_vec: GROVER embedding of E3 ligand (matched by E3 ligase)
  - y_dc50 (log10 nM, z-scored), y_dmax (%, z-scored)
  - metadata: PROTAC_SMILES, target uniprot, E3 ligase, scaffold (Murcko), Linker ID

Output: revision/benchmarking/generalization_data/
"""
import os, warnings, sys
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT_DIR = f"{REV}/benchmarking/generalization_data"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def load_full_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {os.path.basename(path)} → {df.shape}")
    return df

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
    protac_df  = load_full_data(f"{DATA_DIR}/Upadted_PROTAC_features.csv")
    warhead_df = load_full_data(f"{DATA_DIR}/Updated_warhead.csv")
    linker_df  = load_full_data(f"{DATA_DIR}/linker_updated_features.csv")
    e3_df      = load_full_data(f"{DATA_DIR}/E3_updated_features.csv")

    # Normalize keys
    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()

    # Prefix GROVER cols
    def prefix_grover(df, prefix):
        return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df  = prefix_grover(linker_df, "L_")
    e3_df      = prefix_grover(e3_df, "E_")

    w_grover = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_grover = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_grover = [c for c in e3_df.columns if c.startswith("E_Grover_")]
    print(f"  GROVER dims: W={len(w_grover)} L={len(l_grover)} E={len(e_grover)}")

    # Index lookups
    wh_by_uniprot = {}
    wh_by_target = {}
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

    # Iterate PROTACs
    rows = []
    n_wh = n_lk = n_e3 = n_dc50 = n_dmax = 0
    for _, p in protac_df.iterrows():
        # warhead lookup
        u = str(p.get('warhead_Uniprot', '')).strip().upper()
        t = str(p.get('warhead_Target', '')).strip().upper()
        wv = wh_by_uniprot.get(u)
        if wv is None:
            wv = wh_by_target.get(t)
        if wv is None: continue
        n_wh += 1
        # linker lookup
        lid = str(p.get('Linker ID', '')).strip()
        lv = lk_by_id.get(lid)
        if lv is None: continue
        n_lk += 1
        # E3 lookup
        en = str(p.get('E3 ligase', '')).strip().upper()
        ev = e3_by_name.get(en)
        if ev is None: continue
        n_e3 += 1
        # labels
        dc50 = _parse(p.get('DC50 (nM)', np.nan))
        dmax = _parse(p.get('Dmax (%)', np.nan))
        if pd.notna(dc50): n_dc50 += 1
        if pd.notna(dmax): n_dmax += 1
        if pd.isna(dc50) or pd.isna(dmax): continue
        rows.append({
            "W_vec": wv, "L_vec": lv, "E_vec": ev,
            "DC50": dc50, "Dmax": dmax,
            "SMILES": p.get('PROTAC_SMILES', ''),
            "target_uniprot": u, "target_name": t,
            "e3": en, "linker_id": lid,
            "compound_id": p.get('PROTAC_Compound_ID', ''),
        })

    print(f"\n  Matched warhead: {n_wh}, linker: {n_lk}, E3: {n_e3}")
    print(f"  DC50-labeled: {n_dc50}, Dmax-labeled: {n_dmax}")
    print(f"  Final samples (both labels): {len(rows)}")

    if not rows:
        print("No samples — aborting")
        sys.exit(1)

    meta = pd.DataFrame([{k: r[k] for k in ('SMILES','target_uniprot','target_name','e3','linker_id','compound_id','DC50','Dmax')} for r in rows])
    meta['scaffold'] = meta['SMILES'].apply(murcko_scaffold)
    print(f"  Scaffolds computed: {meta['scaffold'].notna().sum()}/{len(meta)}")

    X = np.stack([np.stack([r['W_vec'], r['L_vec'], r['E_vec']]) for r in rows]).astype(np.float32)
    y_dc50 = np.log10(np.clip(meta['DC50'].values, 1e-3, None)).astype(np.float32)
    y_dmax = meta['Dmax'].values.astype(np.float32)

    # z-score (fit on full data; note: for CV we should refit per fold — handled in training)
    dc50_m, dc50_s = y_dc50.mean(), y_dc50.std()
    dmax_m, dmax_s = y_dmax.mean(), y_dmax.std()
    y_dc50_z = ((y_dc50 - dc50_m) / dc50_s).astype(np.float32)
    y_dmax_z = ((y_dmax - dmax_m) / dmax_s).astype(np.float32)
    scalers = {"dc50": (dc50_m, dc50_s), "dmax": (dmax_m, dmax_s)}

    np.save(f"{OUT_DIR}/X.npy", X)
    np.save(f"{OUT_DIR}/y_dc50_z.npy", y_dc50_z)
    np.save(f"{OUT_DIR}/y_dmax_z.npy", y_dmax_z)
    np.save(f"{OUT_DIR}/y_dc50_raw.npy", y_dc50)
    np.save(f"{OUT_DIR}/y_dmax_raw.npy", y_dmax)
    meta.to_csv(f"{OUT_DIR}/metadata.csv", index=False)
    with open(f"{OUT_DIR}/scalers.json", "w") as f:
        import json
        json.dump({k: [float(x) for x in v] for k, v in scalers.items()}, f)

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  X: {X.shape}")
    print(f"  y_dc50_z: {y_dc50_z.shape}  (μ={dc50_m:.2f} σ={dc50_s:.2f})")
    print(f"  y_dmax_z: {y_dmax_z.shape}  (μ={dmax_m:.1f} σ={dmax_s:.1f})")
    print(f"  metadata: {meta.shape}")

    # Quick target/E3 breakdown
    print("\n  Targets:", meta['target_uniprot'].value_counts().head(8).to_dict())
    print("  E3:", meta['e3'].value_counts().to_dict())

if __name__ == '__main__':
    main()
