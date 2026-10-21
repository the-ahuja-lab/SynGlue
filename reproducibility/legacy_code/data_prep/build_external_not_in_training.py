#!/usr/bin/env python3
"""
Build external benchmark set: compounds from final_dataset_E3_known.csv
that are NOT in the SynGlue training data (canonical SMILES exclusion).

Output: external_not_in_training.csv in revision/benchmarking/
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

TRAIN_FILE = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv"
EXT_FILE = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
OUT_FILE = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/external_not_in_training.csv"

def canon(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def main():
    # ---- 1. Training SMILES ----
    print("Loading training data...")
    train = pd.read_csv(TRAIN_FILE, usecols=['id', 'PROTAC_SMILES', 'PROTAC_Compound_ID', 'PROTAC_Name', 'warhead_Uniprot', 'warhead_Target', 'E3 ligase'], low_memory=False)
    train_smiles = train['PROTAC_SMILES'].dropna().tolist()
    print(f"  Training rows: {len(train)}")
    print(f"  Training SMILES (raw): {len(train_smiles)}")

    # Canonicalize
    train_canon = set()
    canon_fail = 0
    for s in train_smiles:
        c = canon(s)
        if c:
            train_canon.add(c)
        else:
            canon_fail += 1
    print(f"  Training canonical SMILES: {len(train_canon)} (failures: {canon_fail})")

    # ---- 2. External benchmark data ----
    print("Loading external benchmark data...")
    ext = pd.read_csv(EXT_FILE, low_memory=False)
    print(f"  External rows: {len(ext)}")

    ext['canonical_for_match'] = ext['SMILES'].apply(canon)
    ext['in_training'] = ext['canonical_for_match'].isin(train_canon)

    n_overlap = ext['in_training'].sum()
    n_external = (~ext['in_training']).sum()
    print(f"  Overlap with training: {n_overlap} rows")
    print(f"  NOT in training: {n_external} rows")

    # ---- 3. Save non-training subset ----
    out = ext[~ext['in_training']].drop(columns=['canonical_for_match', 'in_training'])
    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(f"  Rows: {len(out)}")
    print(f"  Unique SMILES: {out['SMILES'].nunique()}")

    # Summary stats
    print("\n=== Value_Type breakdown (non-training) ===")
    print(out['Value_Type'].value_counts().to_string())
    print("\n=== E3_Ligase breakdown (non-training) ===")
    print(out['E3_Ligase'].value_counts().head(10).to_string())
    print("\n=== POI_UniProt unique (non-training):", out['POI_UniProt'].dropna().nunique())

    # Also save a deduplicated compound-level version (one row per unique SMILES)
    dedup = out.drop_duplicates(subset=['canonical_SMILES'] if 'canonical_SMILES' in out.columns else ['SMILES'])
    dedup_out = OUT_FILE.replace('.csv', '_unique.csv')
    dedup.to_csv(dedup_out, index=False)
    print(f"\nSaved deduplicated: {dedup_out}")
    print(f"  Rows: {len(dedup)}")

if __name__ == '__main__':
    main()
