#!/usr/bin/env python3
"""
WS1 step 4 (v2) — train-consistent PROTAC decomposition with BOND-BREAKING.
Fixes v1 artifact: atom-removal left dangling fragments in linkers.
Method: substructure match E3 ligand / warhead, BREAK bonds crossing the fragment
boundary, then take connected fragments (E3 fragment = one containing the match;
linker/warhead = remaining connected components, merged).

Output: benchmarking/ws1/ws1_components_v2.csv + validation stats
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

def fragment_out(mol, match_atoms):
    """Break bonds crossing the match boundary; return list of fragment SMILES."""
    match_set = set(match_atoms)
    rm = Chem.RWMol(mol)
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            rm.RemoveBond(a1, a2)
    try:
        dotted = Chem.MolToSmiles(rm.GetMol())
    except Exception:
        return None
    frags = [s for s in dotted.split(".") if s]
    return frags or None

def split_by_query(mol, query_smiles_list, query_smarts=None):
    """Break bonds around first query substructure match; return (query_frag, rest)."""
    frags = None
    for qs in query_smiles_list:
        q = Chem.MolFromSmiles(qs)
        if q and mol.HasSubstructMatch(q):
            frags = fragment_out(mol, mol.GetSubstructMatch(q))
            if frags:
                # find the fragment containing the query
                for f in frags:
                    fm = Chem.MolFromSmiles(f)
                    if fm and fm.HasSubstructMatch(q):
                        rest = [x for x in frags if x != f]
                        return f, (".".join(rest) if rest else None)
                return frags[0], None
    if query_smarts is not None:
        q = query_smarts
        if mol.HasSubstructMatch(q):
            frags = fragment_out(mol, mol.GetSubstructMatch(q))
            if frags:
                for f in frags:
                    fm = Chem.MolFromSmiles(f)
                    if fm and fm.HasSubstructMatch(q):
                        rest = [x for x in frags if x != f]
                        return f, (".".join(rest) if rest else None)
                return frags[0], None
    return None, None

def mcs_of(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m]
    if not mols: return None
    try:
        res = rdFMCS.FindMCS(mols, timeout=30, ringMatchesRingOnly=False,
                             completeRingsOnly=False, bondCompare=rdFMCS.BondCompare.CompareAny)
        return Chem.MolFromSmarts(res.smartsString)
    except Exception:
        return None

print("Loading training component libraries...")
e3_lig = pd.read_csv(f"{DATA}/e3_ligand.csv").dropna(subset=["Smiles"]).copy()
e3_lig["Smiles"] = e3_lig["Smiles"].astype(str).apply(canon)
e3_lig = e3_lig.dropna(subset=["Smiles"])
e3_by_target = {}
for _, r in e3_lig.iterrows():
    tgt = str(r.get("Target", "")).strip().upper()
    e3_by_target.setdefault(tgt, []).append(r["Smiles"])
print(f"  E3 ligands: {len(e3_lig)} targets: {list(e3_by_target.keys())}")
e3_cores = {tgt: mcs_of(smis) for tgt, smis in e3_by_target.items()}

wh = pd.read_csv(f"{DATA}/Updated_warhead.csv", low_memory=False,
                 usecols=lambda c: c.startswith("Grover_") is False)
wh2 = wh.dropna(subset=["warhead_Canonical_SMILES"]).copy()
wh2["warhead_Canonical_SMILES"] = wh2["warhead_Canonical_SMILES"].astype(str).apply(canon)
wh2 = wh2.dropna(subset=["warhead_Canonical_SMILES"])
warheads_by_poi = {}
for _, r in wh2.iterrows():
    poi = str(r.get("warhead_Uniprot", "")).strip().upper()
    warheads_by_poi.setdefault(poi, set()).add(r["warhead_Canonical_SMILES"])
print(f"  warheads: {len(wh2)} across {len(warheads_by_poi)} POIs")

print("Decomposing (bond-breaking)...")
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
                         linker_smiles=None, e3_in_train=None, warhead_in_train=None,
                         linker_single_fragment=None)); continue

    # 1) E3: full ligand substructure (or MCS core)
    e3_frag, e3_status = None, "unmatched"
    e3_q = None
    for lsmi in e3_by_target.get(e3_name, []):
        q = Chem.MolFromSmiles(lsmi)
        if q and mol.HasSubstructMatch(q):
            e3_q = q; e3_status = "full_ligand_match"; break
    if e3_q is None and e3_cores.get(e3_name) is not None and mol.HasSubstructMatch(e3_cores[e3_name]):
        e3_q = e3_cores[e3_name]; e3_status = "mcs_core_match"
    e3_in_train = None
    if e3_q is not None:
        e3_frag, remainder = split_by_query(mol, [Chem.MolToSmiles(e3_q)] if e3_status == "full_ligand_match" else [], e3_q if e3_status == "mcs_core_match" else None)
        if e3_frag is None and e3_status == "full_ligand_match":
            e3_frag, remainder = split_by_query(mol, [], e3_q)
        if e3_frag is not None:
            e3_in_train = e3_frag in e3_by_target.get(e3_name, [])

    # 2) warhead: match remainder
    wh_smi, wh_status = None, "unmatched"
    if remainder:
        rm = Chem.MolFromSmiles(remainder)
        cands = set()
        for k, v in warheads_by_poi.items():
            if poi_up and k == poi_up:
                cands |= v
        if not cands:
            for v in warheads_by_poi.values():
                cands |= v
        for wsmi in list(cands)[:4000]:
            q = Chem.MolFromSmiles(wsmi)
            if q and rm and rm.HasSubstructMatch(q):
                wh_smi, wh_status = wsmi, "warhead_match"
                break
    wh_in_train = wh_smi in {s for v in warheads_by_poi.values() for s in v} if wh_smi else None

    # 3) linker = remainder after warhead excision
    linker, linker_single = None, None
    if remainder and wh_smi:
        rm = Chem.MolFromSmiles(remainder)
        q = Chem.MolFromSmiles(wh_smi)
        if rm and q and rm.HasSubstructMatch(q):
            linker = split_by_query(rm, [wh_smi])[1]
    linker_single = (linker is not None and "." not in linker) if linker else None

    rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, POI_UniProt=poi_up,
                     e3_smiles=e3_frag, e3_match=e3_status, e3_in_train=e3_in_train,
                     warhead_smiles=wh_smi, warhead_match=wh_status, warhead_in_train=wh_in_train,
                     linker_smiles=linker, linker_single_fragment=linker_single))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(ext)}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ws1_components_v2.csv"), index=False)

print("\n=== DECOMPOSITION v2 SUMMARY ===")
print(f"Compounds: {len(df)}")
print(df["e3_match"].value_counts().to_string())
e3ok = df["e3_match"].isin(["full_ligand_match", "mcs_core_match"])
print(f"\nE3 matched: {e3ok.sum()} ({100*e3ok.mean():.1f}%)")
print(f"Warhead matched: {(df['warhead_match']=='warhead_match').sum()}")
full = df[df["linker_smiles"].notna()]
print(f"Full [E3+warhead+linker]: {len(full)}")
print(f"  of which linker is single fragment: {full['linker_single_fragment'].sum()}")
print(f"  linker artifacts (multi-fragment): {(~full['linker_single_fragment'].fillna(False)).sum()}")
print(f"\nSaved: {OUT}/ws1_components_v2.csv")
