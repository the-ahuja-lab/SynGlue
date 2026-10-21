#!/usr/bin/env python3
"""
WS1 step 4/7 — provisional PROTAC decomposition: E3-recruiter identification.
Matches known E3-ligand cores by substructure:
  CRBN : 2,6-dioxopiperidin-3-yl (glutarimide) core [thalidomide/lenalidomide/pomalidomide family]
  VHL  : (4-hydroxyproline)-amide VH032/VH298 family core
  other: IAP/MDM2/XIAP/FEM1B/UBR1/RNF4 flagged unresolved (small N in dataset)

Outputs per unique compound: E3_recruiter_smiles (matched core fragment), remainder SMILES,
match status. PROVISIONAL — validate before use in scoring.

Usage: python ws1_decompose_e3.py
Output: benchmarking/ws1/ws1_decomposition.csv + summary printed
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, DataStructs

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"

# ─── E3 core SMARTS ──────────────────────────────────
CORE_SMARTS = {
    # 2,6-dioxopiperidine (glutarimide) ring — CRBN family
    "CRBN_glutarimide": "O=C1NCCCC1=O",
    # phthalimide (alternative CRBN anchor in some degrons)
    "CRBN_phthalimide": "O=C1NC(=O)c2ccccc12",
    # VHL: hydroxyproline amide core of VH032/VH298 (tert-leucine cap variant)
    "VHL_hydroxyproline": "C1CN(CC1O)C(=O)C(NC(=O)OC(C)(C)C)C(C)(C)C",
    # VHL looser: pyrrolidine-2-carboxamide with 4-OH
    "VHL_loose": "O=C1CC(O)CN1C(=O)",
}

E3_EXPECT = {  # expected E3 name per core (dataset uses these names)
    "CRBN_glutarimide": "CRBN",
    "CRBN_phthalimide": "CRBN",
    "VHL_hydroxyproline": "VHL",
    "VHL_loose": "VHL",
}

def match_cores(mol):
    """Return (core_name, core_mol_match) for the best (largest) matched core."""
    hits = []
    for name, sma in CORE_SMARTS.items():
        q = Chem.MolFromSmarts(sma)
        if q is None:
            continue
        if mol.HasSubstructMatch(q):
            # get matched atoms of the largest match
            matches = mol.GetSubstructMatches(q)
            if matches:
                hits.append((name, matches))
    if not hits:
        return None, None
    # prefer largest core match (more atoms = more specific)
    best = max(hits, key=lambda h: len(h[1][0]))
    return best[0], best[1][0]

def excise(mol, atom_idxs):
    """Return the E3 core fragment SMILES and the remainder SMILES."""
    em = Chem.RWMol(mol)
    for aidx in sorted(atom_idxs, reverse=True):
        em.RemoveAtom(aidx)
    remainder = em.GetMol()
    try:
        Chem.SanitizeMol(remainder)
    except Exception:
        remainder = Chem.Mol(remainder)
    core_frag = Chem.MolFromSmiles("")
    # core as a fragment: copy atoms
    core_mol = Chem.RWMol(mol)
    keep = [i for i in range(mol.GetNumAtoms()) if i not in atom_idxs]
    for aidx in sorted(keep, reverse=True):
        core_mol.RemoveAtom(aidx)
    try:
        Chem.SanitizeMol(core_mol.GetMol())
        core_smiles = Chem.MolToSmiles(core_mol.GetMol())
    except Exception:
        core_smiles = ""
    try:
        rem_smiles = Chem.MolToSmiles(remainder)
    except Exception:
        rem_smiles = ""
    return core_smiles, rem_smiles

def main():
    ext = pd.read_csv(EXT).drop_duplicates("canonical_SMILES").reset_index(drop=True)
    rows = []
    for _, r in ext.iterrows():
        smi = r["canonical_SMILES"]
        mol = Chem.MolFromSmiles(smi)
        e3_name = str(r.get("E3_Ligase", "")).strip().upper()
        if mol is None:
            rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, core=None,
                             match_ok=False, core_smiles="", remainder_smiles="", note="parse_failed"))
            continue
        core_name, atom_idxs = match_cores(mol)
        if core_name is None:
            rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, core=None,
                             match_ok=False, core_smiles="", remainder_smiles="", note="no_core_match"))
            continue
        core_smiles, rem_smiles = excise(mol, atom_idxs)
        expected = E3_EXPECT.get(core_name, "")
        consistent = (expected == e3_name)
        rows.append(dict(canonical_smiles=smi, E3_Ligase=e3_name, core=core_name,
                         match_ok=True, core_smiles=core_smiles, remainder_smiles=rem_smiles,
                         consistent_with_metadata=consistent, note=""))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ws1_decomposition.csv"), index=False)

    print(f"Compounds: {len(df)}")
    print(df["core"].fillna("unresolved").value_counts().to_string())
    ok = df[df["match_ok"]]
    print(f"\nMatched: {len(ok)} ({100*len(ok)/len(df):.1f}%)")
    print("Consistency (core vs metadata E3):",
          ok["consistent_with_metadata"].sum(), "/", len(ok))
    print("\nE3-name × core cross-tab (top):")
    ct = pd.crosstab(df["E3_Ligase"], df["core"].fillna("unresolved"))
    print(ct.to_string())
    print(f"\nSaved: {OUT}/ws1_decomposition.csv")

if __name__ == "__main__":
    main()
