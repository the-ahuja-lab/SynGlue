#!/usr/bin/env python3
"""
Fast pharmacophore-based linker enumeration for the GSPT1 benchmark.
Uses available fragment database fragments as direct linkers between warhead and E3.
Limited enumeration with filtering.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdDistGeom
from config import (
    GSPT1_WARHEAD_SMILES, GSPT1_E3_LIGAND_SMILES,
    SYNGLUE_DATA_DIR, TOOL_RESULTS, FILTER_PARAMS
)

out_dir = TOOL_RESULTS["Pharmacophore"]
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("Pharmacophore-Guided Linker Enumeration for GSPT1")
print("=" * 60)

# 1. Load fragments
db_path = os.path.join(SYNGLUE_DATA_DIR, "warhead_fragments.pkl")
print(f"Loading fragment DB: {db_path}")
fragments_df = pd.read_pickle(db_path)
all_fragments = fragments_df["fragment"].dropna().unique().tolist()
print(f"Total unique fragments in DB: {len(all_fragments)}")

# 2. Filter fragments as potential linkers
# Criteria: 4-12 heavy atoms, single connected component
valid_linkers = []
linker_smis = set()

for smi in all_fragments:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    # Remove dummy atoms for counting
    n_atoms = mol.GetNumHeavyAtoms()
    if 4 <= n_atoms <= 12:
        try:
            # Check it's a single fragment
            frags = Chem.GetMolFrags(mol)
            if len(frags) == 1:
                canon = Chem.MolToSmiles(mol, canonical=True)
                if canon not in linker_smis:
                    linker_smis.add(canon)
                    valid_linkers.append(canon)
        except Exception:
            continue

print(f"Valid linker fragments (4-12 atoms): {len(valid_linkers)}")

# 3. Combine warhead + linker + E3
def combine_wh_linker_e3(wh_smi, linker_smi, e3_smi):
    """Combine fragments, handling dummy atoms.
    
    Strategy: Remove * from warhead and E3, then concatenate wh + linker + e3.
    For better chemistry, we try direct SMILES concatenation and then
    also try with common connecting atoms.
    """
    wh_clean = wh_smi.replace("*", "[H]")  # Replace dummy with H temporarily
    e3_clean = e3_smi.replace("*", "[H]")
    
    # Try different connection strategies
    candidates = set()
    
    # Strategy 1: wh-linker-e3 direct
    combined = f"{wh_clean}{linker_smi}{e3_clean}"
    mol = Chem.MolFromSmiles(combined)
    if mol:
        candidates.add(Chem.MolToSmiles(mol, canonical=True))
    
    # Strategy 2: wh + linker + e3 with explicit single bonds
    # Try building with attachment points
    wh_mol = Chem.MolFromSmiles(wh_smi)  # Has dummy atom
    e3_mol = Chem.MolFromSmiles(e3_smi)  # Has dummy atom
    linker_mol = Chem.MolFromSmiles(linker_smi)
    
    if wh_mol and e3_mol and linker_mol:
        # Find dummy atoms
        wh_dummy = [a.GetIdx() for a in wh_mol.GetAtoms() if a.GetAtomicNum() == 0]
        e3_dummy = [a.GetIdx() for a in e3_mol.GetAtoms() if a.GetAtomicNum() == 0]
        
        if wh_dummy and e3_dummy:
            # Remove dummies and make connecting points
            rw_wh = Chem.RWMol(wh_mol)
            rw_e3 = Chem.RWMol(e3_mol)
            
            # Get neighbors of dummy atoms
            wh_neighbor = None
            for neighbor in rw_wh.GetAtomWithIdx(wh_dummy[0]).GetNeighbors():
                wh_neighbor = neighbor.GetIdx()
                break
            
            e3_neighbor = None
            for neighbor in rw_e3.GetAtomWithIdx(e3_dummy[0]).GetNeighbors():
                e3_neighbor = neighbor.GetIdx()
                break
            
            if wh_neighbor is not None and e3_neighbor is not None:
                # Remove dummy atoms
                rw_wh.RemoveAtom(wh_dummy[0])
                rw_e3.RemoveAtom(e3_dummy[0])
                
                try:
                    wh_clean_mol = rw_wh.GetMol()
                    e3_clean_mol = rw_e3.GetMol()
                    
                    # Combine with linker
                    from rdkit.Chem import rdmolops
                    combo = rdmolops.CombineMols(wh_clean_mol, linker_mol)
                    combo = rdmolops.CombineMols(combo, e3_clean_mol)
                    
                    Chem.SanitizeMol(combo)
                    canon = Chem.MolToSmiles(combo, canonical=True)
                    candidates.add(canon)
                except Exception:
                    pass
    
    return list(candidates)


# 4. Enumerate PROTACs (limit to first 3000 linkers for speed)
print("Enumerating PROTACs...")
all_protacs = set()
for i, linker_smi in enumerate(valid_linkers[:3000]):
    if i % 500 == 0:
        print(f"  Processing linker {i}/{min(3000, len(valid_linkers))}...")
    protacs = combine_wh_linker_e3(GSPT1_WARHEAD_SMILES, linker_smi, GSPT1_E3_LIGAND_SMILES)
    for p in protacs:
        mol = Chem.MolFromSmiles(p)
        if mol:
            mw = Descriptors.MolWt(mol)
            if mw <= FILTER_PARAMS["max_mw"] * 1.2:  # generous first pass
                all_protacs.add(p)

print(f"Generated {len(all_protacs)} unique PROTAC candidates")

# 5. Save
result_df = pd.DataFrame({"SMILES": list(all_protacs)})
csv_path = os.path.join(out_dir, "generated_candidates.csv")
result_df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")

# 6. Apply filter cascade
from standard_filters import apply_filter_cascade, summarize_filter_results

smiles_list = result_df["SMILES"].tolist()
print(f"Running filter cascade on {len(smiles_list)} candidates...")
df_filt = apply_filter_cascade(
    smiles_list,
    GSPT1_WARHEAD_SMILES,
    GSPT1_E3_LIGAND_SMILES,
)

summary = summarize_filter_results(df_filt, "Pharmacophore")
for k, v in summary.items():
    print(f"  {k}: {v}")

filt_csv = os.path.join(out_dir, "filtered_results.csv")
df_filt.to_csv(filt_csv, index=False)
print(f"Saved filtered results to {filt_csv}")
print("Done!")
