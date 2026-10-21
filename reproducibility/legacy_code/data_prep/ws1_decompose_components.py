#!/usr/bin/env python3
"""
WS1 steps 4+7 — train-consistent PROTAC decomposition for external set.
Strategy (mirrors training construction):
  1. E3 recruiter : substructure match external PROTAC vs training E3 ligand library
                    (e3_ligand.csv, 117 ligands, by Target name); fallback = per-family MCS core
  2. Warhead      : excise E3 fragment; substructure match remainder vs training warhead
                    library (Updated_warhead.csv, filtered by POI UniProt when available)
  3. Linker       : remainder after E3 + warhead excision
Outputs per unique compound: component SMILES + novelty flags (component seen in train).

Usage: python ws1_decompose_components.py
Output: benchmarking/ws1/ws1_components.csv + summary
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
os.makedirs(OUT, exist_ok=True)

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def remove_substruct(mol, query, atom_map=None):
    """Remove first substructure match; return remainder SMILES (or None)."""
    if mol is None: return None
    match = mol.GetSubstructMatch(query)
    if not match: return None
    em = Chem.RWMol(mol)
    for a in sorted(match, reverse=True):
        em.RemoveAtom(a)
    try:
        Chem.SanitizeMol(em.GetMol())
        return Chem.MolToSmiles(em.GetMol())
    except Exception:
        return None

def mcs_of(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m]
    if not mols: return None
    try:
        res = rdFMCS.FindMCS(mols, timeout=30, ringMatchesRingOnly=False,
                             completeRingsOnly=False, bondCompare=rdFMCS.BondCompare.CompareAny)
        q = Chem.MolFromSmarts(res.smartsString)
        return q
    except Exception:
        return None

print("Loading training component libraries...")
# ─── E3 library ──────────────────────────────────────
e3_lig = pd.read_csv(f"{DATA}/e3_ligand.csv")
e3_lig = e3_lig.dropna(subset=["Smiles"]).copy()
e3_lig["Smiles"] = e3_lig["Smiles"].astype(str).apply(canon)
e3_lig = e3_lig.dropna(subset=["Smiles"])
e3_by_target = {}
for _, r in e3_lig.iterrows():
    tgt = str(r.get("Target", "")).strip().upper()
    e3_by_target.setdefault(tgt, []).append(r["Smiles"])
print(f"  E3 ligands: {len(e3_lig)} for {list(e3_by_target.keys())}")

# per-family MCS cores
e3_cores = {}
for tgt, smis in e3_by_target.items():
    q = mcs_of(smis)
    if q is not None:
        e3_cores[tgt] = q
        print(f"  {tgt}: MCS core from {len(smis)} ligands")

# ─── Warhead library ─────────────────────────────────
wh = pd.read_csv(f"{DATA}/Updated_warhead.csv", low_memory=False,
                 usecols=lambda c: c.startswith("Grover_") is False)
print(f"  warhead rows: {len(wh)}, cols: {list(wh.columns)}")
wh_smiles_col = "warhead_Canonical_SMILES" if "warhead_Canonical_SMILES" in wh.columns else None
warheads_by_poi = {}
if wh_smiles_col:
    wh2 = wh.dropna(subset=[wh_smiles_col]).copy()
    wh2[wh_smiles_col] = wh2[wh_smiles_col].astype(str).apply(canon)
    wh2 = wh2.dropna(subset=[wh_smiles_col])
    for _, r in wh2.iterrows():
        poi = str(r.get("warhead_Uniprot", "")).strip().upper()
        smi = r[wh_smiles_col]
        warheads_by_poi.setdefault(poi, set()).add(smi)
    print(f"  warhead SMILES: {len(wh2)} (across {len(warheads_by_poi)} POIs)")

# ─── Decompose ───────────────────────────────────────
print("\nDecomposing external compounds...")
ext = pd.read_csv(EXT).drop_duplicates("canonical_SMILES").reset_index(drop=True)
rows = []
for i, r in ext.iterrows():
    smi = r["canonical_SMILES"]
    mol = Chem.MolFromSmiles(smi)
    e3_name = str(r.get("E3_Ligase", "")).strip().upper()
    poi_up = str(r.get("POI_UniProt", "")).strip().upper()
    if mol is None:
        rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, POI_UniProt=poi_up,
                         e3_smiles=None, e3_match="parse_failed", warhead_smiles=None,
                         linker_smiles=None, e3_in_train=None, warhead_in_train=None))
        continue

    # 1) E3 recruiter: full ligand substructure first
    e3_frag, e3_status = None, "unmatched"
    for tgt in (e3_name,):
        for lsmi in e3_by_target.get(tgt, []):
            q = Chem.MolFromSmiles(lsmi)
            if q and mol.HasSubstructMatch(q):
                e3_frag, e3_status = lsmi, "full_ligand_match"
                break
        if e3_status == "full_ligand_match": break
    # 1b) MCS core fallback
    if e3_status == "unmatched":
        q = e3_cores.get(e3_name)
        if q is not None:
            match = mol.GetSubstructMatch(q)
            if match:
                # extract matched core as SMILES
                em = Chem.RWMol(mol)
                keep = [i for i in range(mol.GetNumAtoms()) if i not in match]
                for a in sorted(keep, reverse=True):
                    em.RemoveAtom(a)
                try:
                    Chem.SanitizeMol(em.GetMol())
                    e3_frag = Chem.MolToSmiles(em.GetMol())
                    e3_status = "mcs_core_match"
                except Exception:
                    e3_status = "mcs_fail"
    e3_in_train = (e3_frag in e3_by_target.get(e3_name, [])) if e3_frag else None

    # 2) Warhead: excise E3, then match remainder vs POI warhead library
    remainder = None
    if e3_frag:
        q = Chem.MolFromSmiles(e3_frag)
        remainder = remove_substruct(mol, q) if q else None
    wh_smi, wh_status = None, "unmatched"
    if remainder:
        rm = Chem.MolFromSmiles(remainder)
        cands = set()
        for k, v in warheads_by_poi.items():
            if poi_up and k == poi_up:
                cands |= v
        if not cands:  # fallback: all warheads (expensive; sample if huge)
            for k, v in warheads_by_poi.items():
                cands |= v
        for wsmi in list(cands)[:4000]:
            q = Chem.MolFromSmiles(wsmi)
            if q and rm.HasSubstructMatch(q):
                wh_smi, wh_status = wsmi, "warhead_match"
                break
    wh_in_train = wh_smi in {s for v in warheads_by_poi.values() for s in v} if wh_smi else None

    # 3) Linker = remainder after warhead excision
    linker = None
    if remainder and wh_smi:
        rm = Chem.MolFromSmiles(remainder)
        q = Chem.MolFromSmiles(wh_smi)
        linker = remove_substruct(rm, q) if q else None

    rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, POI_UniProt=poi_up,
                     e3_smiles=e3_frag, e3_match=e3_status, e3_in_train=e3_in_train,
                     warhead_smiles=wh_smi, warhead_match=wh_status, warhead_in_train=wh_in_train,
                     linker_smiles=linker))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(ext)}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ws1_components.csv"), index=False)

print("\n=== DECOMPOSITION SUMMARY ===")
print(f"Compounds: {len(df)}")
print(df["e3_match"].value_counts().to_string())
print()
print("E3 matched:", int((df['e3_match'].isin(['full_ligand_match','mcs_core_match'])).sum()),
      f"({100*df['e3_match'].isin(['full_ligand_match','mcs_core_match']).mean():.1f}%)")
print("E3 full-ligand in train:", int(df['e3_in_train'].fillna(False).sum()))
print("Warhead matched:", int((df['warhead_match']=='warhead_match').sum()))
print("Full [E3+warhead+linker] decomposed:", int(df['linker_smiles'].notna().sum()))
print(f"\nSaved: {OUT}/ws1_components.csv")
