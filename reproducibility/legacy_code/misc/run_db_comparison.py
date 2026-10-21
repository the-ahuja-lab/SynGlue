#!/usr/bin/env python3
"""
Cross-database comparison: PROTAC-DB vs PROTACpedia vs TPDdb vs PROTAC-Patent vs TACK

Levels of comparison:
  1. Count (unique PROTACs by canonical SMILES)
  2. Targets (POI)
  3. Linker (component-level)
  4. E3 ligase
  5. Cell lines
  6. Assays
  7. Other (physchem properties, publication years, database overlap)

Output: protac_db_justification/{results,figures,data}
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
EXT = f"{BASE}/external_benchmark_data"
OUT = f"{REV}/protac_db_justification"
RES = f"{OUT}/results"; FIG = f"{OUT}/figures"; DATA = f"{OUT}/data"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon(s):
    if not isinstance(s, str) or not s: return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or None
    except:
        return None

def physchem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return {
        'MW': Descriptors.MolWt(mol),
        'logP': Crippen.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
    }

def main():
    # ══════════ LOAD SOURCES ══════════
    print("Loading sources...")
    # TACK (with Database provenance) — TPDdb is TACK's core source
    tack = pd.read_parquet(f"{EXT}/tack_raw/data_train-00000-of-00001.parquet")
    tack['canon'] = tack['SMILES'].apply(canon)

    # TACK = whole dataset (TPDdb is its main constituent); subsets for comparison
    tack_db = tack[tack['Database'].str.contains('PROTAC-DB', na=False)].copy()
    tack_pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)].copy()
    tack_tpddb = tack  # TPDdb is TACK itself

    # Training PROTAC-DB (with features) — use Upadted for linker IDs, PROTAC_DB for the rest
    pdb = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/PROTAC_DB_with_features.csv", low_memory=False)
    upd = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv", low_memory=False,
                      usecols=['PROTAC_SMILES', 'Linker ID', 'E3 ligase', 'Assay (DC50/Dmax)'])
    upd_map = upd.drop_duplicates(subset='PROTAC_SMILES').set_index('PROTAC_SMILES')
    pdb['canon'] = pdb['Smiles'].apply(canon)
    pdb['_linker_id'] = pdb['Smiles'].map(upd_map['Linker ID'])
    pdb['_assay_dc50dmax'] = pdb['Smiles'].map(upd_map['Assay (DC50/Dmax)'])

    # PROTACpedia full (TACK subset is what we have; also check for pedia-specific rows)
    # Patent (OFFICIAL PROTAC-PatentDB from Figshare, Scientific Data 2025)
    pat = pd.read_csv(f"{OUT}/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv", low_memory=False)
    pat['canon'] = pat['SMILES'].apply(canon)

    # Linker library (training)
    linker_df = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/linker_updated_features.csv", low_memory=False)
    linker_by_id = {str(int(lid)): smi for lid, smi in zip(linker_df['linker_Compound_ID'], linker_df['linker_Smiles'])}

    # ══════════ DATABASE DEFINITIONS ══════════
    dbs = {
        "PROTAC-DB": {
            "canon": set(pdb['canon'].dropna()),
            "n_records": len(pdb),
            "target": pdb['Target'],
            "e3": pdb['E3 ligase'],
            "cell_line": pd.Series([np.nan]*len(pdb)),
            "assay": pdb['_assay_dc50dmax'],
            "smiles_series": pdb['Smiles'],
            "linker_id": pdb['_linker_id'],
            "year": pd.Series([np.nan]*len(pdb)),
            "ref": pdb['PDB'],
        },
        "TACK (incl. PROTAC-DB 3.0 subset)": {
            "canon": set(tack_db['canon'].dropna()),
            "n_records": len(tack_db),
            "target": tack_db['POI_Name'],
            "e3": tack_db['Ligase_Name'],
            "cell_line": tack_db['Cell_Line'],
            "assay": tack_db['Assay'],
            "smiles_series": tack_db['SMILES'],
            "linker_id": None,
            "year": pd.Series([np.nan]*len(tack_db)),
            "ref": tack_db['Reference'],
        },
        "PROTACpedia": {
            "canon": set(tack_pedia['canon'].dropna()),
            "n_records": len(tack_pedia),
            "target": tack_pedia['POI_Name'],
            "e3": tack_pedia['Ligase_Name'],
            "cell_line": tack_pedia['Cell_Line'],
            "assay": tack_pedia['Assay'],
            "smiles_series": tack_pedia['SMILES'],
            "linker_id": None,
            "year": pd.Series([np.nan]*len(tack_pedia)),
            "ref": tack_pedia['Reference'],
        },
        "TACK (full)": {
            "canon": set(tack_tpddb['canon'].dropna()),
            "n_records": len(tack_tpddb),
            "target": tack_tpddb['POI_Name'],
            "e3": tack_tpddb['Ligase_Name'],
            "cell_line": tack_tpddb['Cell_Line'],
            "assay": tack_tpddb['Assay'],
            "smiles_series": tack_tpddb['SMILES'],
            "linker_id": None,
            "year": pd.Series([np.nan]*len(tack_tpddb)),
            "ref": tack_tpddb['Reference'],
        },
        "PROTAC-Patent": {
            "canon": set(pat['canon'].dropna()),
            "n_records": len(pat),
            "target": pat['Target'],
            "e3": pd.Series([np.nan]*len(pat)),
            "cell_line": pd.Series([np.nan]*len(pat)),
            "assay": pd.Series([np.nan]*len(pat)),
            "smiles_series": pat['SMILES'],
            "linker_id": None,
            "year": pat['Patent Publication Year'],
            "ref": pat['Patent Number'],
        },
    }

    # ══════════ 1. COUNT ══════════
    print("\n=== 1. COUNT (unique PROTACs) ===")
    count_rows = []
    for name, d in dbs.items():
        n_uniq = len(d['canon'])
        n_rec = d['n_records']
        print(f"  {name:20s}: {n_rec:6d} records, {n_uniq:6d} unique")
        count_rows.append({"database": name, "records": n_rec, "unique_protacs": n_uniq})
    pd.DataFrame(count_rows).to_csv(f"{RES}/1_counts.csv", index=False)

    # Overlap matrix
    names = list(dbs.keys())
    overlap = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in names:
        for b in names:
            overlap.loc[a, b] = len(dbs[a]['canon'] & dbs[b]['canon'])
    overlap.to_csv(f"{RES}/2_overlap_matrix.csv")
    print("\n  Overlap matrix (unique SMILES):")
    print(overlap.to_string())

    # ══════════ 2. TARGETS ══════════
    print("\n=== 2. TARGETS ===")
    target_rows = []
    for name, d in dbs.items():
        tgt = d['target'].dropna().astype(str)
        tgt = tgt[tgt != 'nan'].replace({'': np.nan}).dropna()
        n_t = tgt.nunique()
        print(f"  {name:20s}: {n_t:5d} unique targets")
        target_rows.append({"database": name, "unique_targets": n_t})
        # top targets
        d['_target_counts'] = tgt.value_counts().head(10)
    pd.DataFrame(target_rows).to_csv(f"{RES}/3_targets.csv", index=False)

    # ══════════ 3. E3 LIGASE ══════════
    print("\n=== 3. E3 LIGASE ===")
    e3_rows = []
    e3_top = {}
    for name, d in dbs.items():
        e3 = d['e3'].dropna().astype(str)
        e3 = e3[e3 != 'nan'].replace({'': np.nan}).dropna()
        n_e3 = e3.nunique()
        print(f"  {name:20s}: {n_e3:3d} unique E3 ligases")
        e3_rows.append({"database": name, "unique_e3": n_e3})
        e3_top[name] = e3.value_counts().head(6)
    pd.DataFrame(e3_rows).to_csv(f"{RES}/4_e3.csv", index=False)

    # ══════════ 4. CELL LINES ══════════
    print("\n=== 4. CELL LINES ===")
    cell_rows = []
    for name, d in dbs.items():
        cl = d['cell_line'].dropna().astype(str)
        cl = cl[cl != 'nan'].replace({'': np.nan}).dropna()
        n_cl = cl.nunique()
        print(f"  {name:20s}: {n_cl:5d} unique cell lines")
        cell_rows.append({"database": name, "unique_cell_lines": n_cl})
    pd.DataFrame(cell_rows).to_csv(f"{RES}/5_celllines.csv", index=False)

    # ══════════ 5. ASSAYS ══════════
    print("\n=== 5. ASSAYS ===")
    assay_rows = []
    for name, d in dbs.items():
        a = d['assay'].dropna().astype(str)
        a = a[a != 'nan'].replace({'': np.nan}).dropna()
        n_a = a.nunique()
        print(f"  {name:20s}: {n_a:5d} unique assay types")
        assay_rows.append({"database": name, "unique_assays": n_a})
    pd.DataFrame(assay_rows).to_csv(f"{RES}/6_assays.csv", index=False)

    # ══════════ 6. LINKER (component) ══════════
    print("\n=== 6. LINKER ===")
    # PROTAC-DB: explicit linker IDs
    pdb_linkers = pdb['_linker_id'].dropna()
    pdb_linkers = [str(int(float(l))) if pd.notna(l) else None for l in pdb_linkers]
    pdb_linker_smiles = [linker_by_id.get(l) for l in pdb_linkers]
    pdb_linker_smiles = [l for l in pdb_linker_smiles if l and Chem.MolFromSmiles(l)]
    n_linker_unique = len(set(canon(l) for l in pdb_linker_smiles))
    linker_len = [Chem.MolFromSmiles(l).GetNumHeavyAtoms() for l in pdb_linker_smiles]
    print(f"  PROTAC-DB: {n_linker_unique} unique linker SMILES (explicit component annotation)")
    print(f"    linker heavy-atom: median={np.median(linker_len):.0f}, range=[{min(linker_len)},{max(linker_len)}]")

    # Other DBs: scaffold diversity as linker-proxy + note component info availability
    linker_rows = [{"database": "PROTAC-DB", "unique_linkers": n_linker_unique,
                    "linker_heavy_atom_median": np.median(linker_len), "component_info": "explicit warhead/linker/E3"}]
    for name in ["TACK (incl. PROTAC-DB 3.0 subset)", "PROTACpedia", "TACK (full)", "PROTAC-Patent"]:
        d = dbs[name]
        sc = d['smiles_series'].apply(murcko).dropna()
        n_sc = sc.nunique()
        linker_rows.append({"database": name, "unique_linkers": n_sc,
                            "linker_heavy_atom_median": np.nan,
                            "component_info": "no explicit component annotation (needs decomposition)"})
        print(f"  {name:20s}: {n_sc:5d} unique scaffolds (linker proxy)")
    pd.DataFrame(linker_rows).to_csv(f"{RES}/7_linker.csv", index=False)

    # ══════════ 7. OTHER: physchem + years ══════════
    print("\n=== 7. OTHER ===")
    phys_rows = []
    for name, d in dbs.items():
        smis = d['smiles_series'].dropna().head(2000)  # sample for speed
        props = [physchem(s) for s in smis]
        props = [p for p in props if p]
        if props:
            dfp = pd.DataFrame(props)
            phys_rows.append({"database": name,
                              "MW_median": dfp['MW'].median(), "logP_median": dfp['logP'].median(),
                              "TPSA_median": dfp['TPSA'].median(), "RotB_median": dfp['RotB'].median(),
                              "HBA_median": dfp['HBA'].median(), "HBD_median": dfp['HBD'].median()})
    pd.DataFrame(phys_rows).to_csv(f"{RES}/8_physchem.csv", index=False)

    # Publication years (patent)
    yr = pat['Patent Publication Year'].dropna()
    print(f"  PROTAC-Patent: year range [{yr.min():.0f}, {yr.max():.0f}], median {yr.median():.0f}")

    # Save canon sets for downstream analysis
    import pickle
    with open(f"{DATA}/db_canon_sets.pkl", "wb") as f:
        pickle.dump({k: v['canon'] for k, v in dbs.items()}, f)
    with open(f"{DATA}/db_canon_sets.json", "w") as f:
        json.dump({k: len(v['canon']) for k, v in dbs.items()}, f, indent=2)

    print("\nSaved all level-comparison CSVs to", RES)

if __name__ == '__main__':
    main()
