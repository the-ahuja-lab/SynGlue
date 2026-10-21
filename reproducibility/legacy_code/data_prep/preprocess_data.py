#!/usr/bin/env python3
"""
Preprocess PROTAC data into compact numpy arrays for WP2 generalization testing.
Avoids OOM by processing in stages: (1) metadata merge, (2) GROVER extraction per component.
"""
import os, sys, time, warnings, gc, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT_DIR  = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

# ─── STAGE 1: Load metadata only (no GROVER columns) ───
print("=" * 60)
print("STAGE 1: Loading metadata...")
protac = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False,
                     usecols=lambda c: not c.startswith("Grover_"))
print(f"  PROTAC: {len(protac)} rows, {time.time()-t0:.1f}s")

# Normalize keys
protac["warhead_Uniprot"] = protac["warhead_Uniprot"].astype(str).str.strip().str.upper()
protac["warhead_Target"]  = protac["warhead_Target"].astype(str).str.strip().str.upper()
protac["E3 ligase"]       = protac["E3 ligase"].astype(str).str.strip().str.upper()
protac["Linker ID"]       = protac["Linker ID"].astype(str).str.strip().str.upper()
protac["PROTAC_Compound_ID"] = protac["PROTAC_Compound_ID"].astype(str).str.strip().str.upper()

# ─── STAGE 2: Load warhead GROVER + metadata ───
print("STAGE 2: Loading warhead GROVER...")
warhead = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
warhead["warhead_Uniprot"] = warhead["warhead_Uniprot"].astype(str).str.strip().str.upper()
warhead["warhead_Target"]  = warhead["warhead_Target"].astype(str).str.strip().str.upper()
w_g = [c for c in warhead.columns if c.startswith("Grover_")]
print(f"  Warhead: {len(warhead)} rows, {len(w_g)} GROVER dims, {time.time()-t0:.1f}s")

# Build warhead lookup: dict of (uniprot, target) → GROVER vector
# Use Uniprot as primary key, Target as fallback
warhead_by_uniprot = {}
warhead_by_target  = {}
for _, r in warhead.iterrows():
    w_vec = r[w_g].values.astype(np.float32)
    up = r["warhead_Uniprot"]
    tg = r["warhead_Target"]
    if up and up != "NAN" and up != "": warhead_by_uniprot[up] = w_vec
    if tg and tg != "NAN" and tg != "": warhead_by_target[tg] = w_vec

del warhead
gc.collect()

# ─── STAGE 3: Load linker GROVER ───
print("STAGE 3: Loading linker GROVER...")
linker = pd.read_csv(f"{DATA_DIR}/linker_updated_features.csv", low_memory=False)
linker["linker_Compound_ID"] = linker["linker_Compound_ID"].astype(str).str.strip().str.upper()
l_g = [c for c in linker.columns if c.startswith("Grover_")]
print(f"  Linker: {len(linker)} rows, {len(l_g)} GROVER dims, {time.time()-t0:.1f}s")

linker_by_id = {}
for _, r in linker.iterrows():
    lid = r["linker_Compound_ID"]
    linker_by_id[lid] = r[l_g].values.astype(np.float32)

del linker
gc.collect()

# ─── STAGE 4: Load E3 GROVER ───
print("STAGE 4: Loading E3 GROVER...")
e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)
e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()
e_g = [c for c in e3_df.columns if c.startswith("Grover_")]
print(f"  E3: {len(e3_df)} rows, {len(e_g)} GROVER dims, {time.time()-t0:.1f}s")

e3_by_name = {}
for _, r in e3_df.iterrows():
    e3_by_name[r["E3_Ligase"]] = r[e_g].values.astype(np.float32)

del e3_df
gc.collect()

assert len(w_g) == len(l_g) == len(e_g), f"GROVER dim mismatch: {len(w_g)} vs {len(l_g)} vs {len(e_g)}"
GROVER_DIM = len(w_g)

# ─── STAGE 5: Match and build dataset ───
print("STAGE 5: Matching components...")

def parse_dc50(v):
    if pd.isna(v) or str(v).strip().upper() in ("N.D.", "N/D", "NA", "N/A", "", "NAN", "NONE"):
        return np.nan
    s = str(v).strip()
    if s.startswith(">"): s = s[1:]
    elif s.startswith("<"): s = s[1:]
    parts = s.split("/")
    vals = []
    for p in parts:
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", "", "NAN", "NONE"): continue
        try:
            vals.append(float(p))
        except:
            continue
    return np.median(vals) if vals else np.nan

# Pre-parse DC50
dc50_parsed = protac["DC50 (nM)"].apply(parse_dc50).values
dmax_parsed = pd.to_numeric(protac["Dmax (%)"], errors="coerce").values

rows = []
matched = 0
total = len(protac)
for i, (_, p) in enumerate(protac.iterrows()):
    if i % 2000 == 0:
        print(f"  Matching {i}/{total}... ({matched} matched so far)", end="\r")
    
    # Warhead lookup: Uniprot first, then Target
    up = p["warhead_Uniprot"]
    tg = p["warhead_Target"]
    w_vec = warhead_by_uniprot.get(up)
    if w_vec is None:
        # Try by target name
        w_vec = warhead_by_target.get(tg)
    if w_vec is None:
        continue
    
    # Linker lookup
    lid = p["Linker ID"]
    l_vec = linker_by_id.get(lid)
    if l_vec is None:
        continue
    
    # E3 lookup
    e3_name = p["E3 ligase"]
    e_vec = e3_by_name.get(e3_name)
    if e_vec is None:
        continue
    
    # Labels
    dc50_val = dc50_parsed[i]
    dmax_val = dmax_parsed[i]
    if np.isnan(dc50_val) or np.isnan(dmax_val):
        continue
    
    rows.append({
        "w_vec": w_vec,
        "l_vec": l_vec,
        "e_vec": e_vec,
        "dc50": dc50_val,
        "dmax": dmax_val,
        "target": tg,
        "e3_name": e3_name,
        "smiles": str(p.get("PROTAC_SMILES", "")),
        "protac_id": str(p.get("PROTAC_Compound_ID", "")),
    })
    matched += 1

print(f"\n  Matched {matched} samples with all components + labels ({time.time()-t0:.1f}s)")

n = len(rows)
X = np.zeros((n, 3, GROVER_DIM), dtype=np.float32)
y_dc50 = np.zeros(n, dtype=np.float32)
y_dmax = np.zeros(n, dtype=np.float32)
meta_list = []

for i, r in enumerate(rows):
    X[i, 0] = r["w_vec"]
    X[i, 1] = r["l_vec"]
    X[i, 2] = r["e_vec"]
    y_dc50[i] = np.log10(max(r["dc50"], 1e-3))
    y_dmax[i] = r["dmax"]
    meta_list.append({
        "target": r["target"],
        "e3_name": r["e3_name"],
        "smiles": r["smiles"],
        "protac_id": r["protac_id"],
    })

meta = pd.DataFrame(meta_list)

# Compute scaffolds
print("Computing Murcko scaffolds...")
scaffolds = []
for i, smi in enumerate(meta["smiles"].values):
    if i % 500 == 0:
        print(f"  Scaffold {i}/{n}...", end="\r")
    if pd.isna(smi) or str(smi).strip() in ("", "nan", "none"):
        scaffolds.append(None)
        continue
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        scaffolds.append(None)
        continue
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(sc if sc else None)
    except:
        scaffolds.append(None)

meta["scaffold"] = scaffolds
n_scaffolded = meta["scaffold"].notna().sum()
print(f"  Scaffolds: {n_scaffolded}/{n} valid, {meta['scaffold'].nunique()} unique")

# ─── STAGE 6: Save ───
np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y_dc50.npy"), y_dc50)
np.save(os.path.join(OUT_DIR, "y_dmax.npy"), y_dmax)
meta.to_pickle(os.path.join(OUT_DIR, "meta.pkl"))

# Summary
dc50_m, dc50_s = y_dc50.mean(), y_dc50.std()
dmax_m, dmax_s = y_dmax.mean(), y_dmax.std()

print(f"\n{'=' * 60}")
print(f"DATA PREPROCESSING COMPLETE")
print(f"{'=' * 60}")
print(f"  Samples: {n}")
print(f"  GROVER dim: {GROVER_DIM}")
print(f"  X shape: {X.shape} ({X.nbytes/1e6:.1f} MB)")
print(f"  y_dc50 (log10 nM): mean={dc50_m:.2f}, std={dc50_s:.2f}")
print(f"  y_dmax (%): mean={dmax_m:.1f}, std={dmax_s:.1f}")
print(f"  Targets: {meta['target'].nunique()}")
print(f"  E3 ligases: {meta['e3_name'].nunique()} → {sorted(meta['e3_name'].unique())}")
print(f"  Scaffolds: {meta['scaffold'].nunique()} unique")
print(f"  Total time: {time.time()-t0:.1f}s ({time.time()-t0:.1f}s)")

print(f"\n  Saved:")
print(f"    {OUT_DIR}/X.npy")
print(f"    {OUT_DIR}/y_dc50.npy")
print(f"    {OUT_DIR}/y_dmax.npy")
print(f"    {OUT_DIR}/meta.pkl")

# Quick E3 distribution
print(f"\n  E3 distribution:")
e3_counts = meta["e3_name"].value_counts()
for e3_name, count in e3_counts.items():
    print(f"    {e3_name}: {count}")

# Quick target distribution (top)
print(f"\n  Top targets:")
for t, c in meta["target"].value_counts().head(10).items():
    print(f"    {t}: {c}")
