#!/usr/bin/env python3
"""
Run PDP (PROTAC-Degradation-Predictor) on external benchmark compounds
NOT in training data. Batch mode with CPU patching.
"""
import warnings, time, sys, os, json
warnings.filterwarnings('ignore')

PDP_ENV = "/home/saveenas/miniconda3/envs/pdp"
BASE = "/storage/savi/saveenas/Projects/SynGlue"
sys.path.insert(0, f"{BASE}/validation_tools/PROTAC-Degradation-Predictor")

import numpy as np
import pandas as pd

# CPU patch
import protac_degradation_predictor.pytorch_models as pm
orig_load = pm.load_model
def patched_load(ckpt_path):
    import torch
    model = pm.PROTAC_Model.load_from_checkpoint(ckpt_path, map_location='cpu')
    return model.eval()
pm.load_model = patched_load
import protac_degradation_predictor.protac_degradation_predictor as pdp_mod
pdp_mod.load_model = patched_load
import protac_degradation_predictor as pdp

INPUT = f"{BASE}/revision/benchmarking/external_not_in_training.csv"
OUT = f"{BASE}/revision/benchmarking/tool_results/pdp_external_predictions.csv"
OUT_JSON = f"{BASE}/revision/benchmarking/tool_results/pdp_external_predictions.json"

def main():
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"Input rows: {len(df)}")

    # Deduplicate to unique (SMILES, E3_Ligase, POI_UniProt, Cell_Line) combos
    # Use E3_Ligase name directly (PDP expects names like CRBN/VHL/IAP)
    df['e3_name'] = df['E3_Ligase'].fillna('CRBN')
    # Map PDP-compatible names
    name_map = {'Ubr1': 'Ubr1', 'UBR1': 'Ubr1'}
    df['e3_name'] = df['e3_name'].replace(name_map)

    keep_cols = ['SMILES', 'canonical_SMILES', 'POI_UniProt', 'E3_UniProt', 'e3_name', 'E3_Ligase',
                 'Cell_Line', 'Value_Type', 'Value', 'Value_Unit', 'Protein_of_Interest', 'Curated_Name']
    sub = df[keep_cols].copy()
    sub['POI_UniProt'] = sub['POI_UniProt'].fillna('P01116')  # default if missing

    # PDP cell line fallback: use 22Rv1 for lines not in PDP DB
    avail_cells = set(pdp.avail_cell_lines())
    def fix_cell(c):
        if pd.isna(c) or c not in avail_cells:
            return '22Rv1'
        return c
    sub['cell_used'] = sub['Cell_Line'].apply(fix_cell)
    n_fallback = (sub['cell_used'] != sub['Cell_Line'].fillna('22Rv1')).sum()
    print(f"Cell lines with PDP fallback: {n_fallback}")

    # Run PDP in batches on unique rows
    # Group by (SMILES, e3_name, POI_UniProt) — one prediction per unique PROTAC-target-E3
    sub = sub.drop_duplicates(subset=['canonical_SMILES', 'e3_name', 'POI_UniProt'])
    print(f"Unique (SMILES, E3, POI) combos: {len(sub)}")

    BATCH = 32
    results = []
    t0 = time.time()
    for i in range(0, len(sub), BATCH):
        chunk = sub.iloc[i:i+BATCH]
        try:
            pred = pdp.get_protac_active_proba(
                chunk['SMILES'].tolist(),
                chunk['e3_name'].tolist(),
                chunk['POI_UniProt'].tolist(),
                chunk['cell_used'].tolist(),
                device='cpu', use_models_from_cv=False
            )
            for j, (_, row) in enumerate(chunk.iterrows()):
                results.append({
                    'SMILES': row['SMILES'],
                    'canonical_SMILES': row['canonical_SMILES'],
                    'Protein_of_Interest': row['Protein_of_Interest'],
                    'POI_UniProt': row['POI_UniProt'],
                    'E3_Ligase': row['E3_Ligase'],
                    'e3_name_used': row['e3_name'],
                    'Cell_Line': row['Cell_Line'],
                    'cell_used': row['cell_used'],
                    'Value_Type': row['Value_Type'],
                    'Value': row['Value'],
                    'Curated_Name': row['Curated_Name'],
                    'pdp_active_mean': float(pred['mean'][j]),
                    'pdp_majority_vote': bool(pred['majority_vote'][j]),
                })
        except Exception as e:
            print(f"  Batch {i//BATCH} error: {str(e)[:150]}")
            for _, row in chunk.iterrows():
                results.append({
                    'SMILES': row['SMILES'],
                    'canonical_SMILES': row['canonical_SMILES'],
                    'Protein_of_Interest': row['Protein_of_Interest'],
                    'POI_UniProt': row['POI_UniProt'],
                    'E3_Ligase': row['E3_Ligase'],
                    'e3_name_used': row['e3_name'],
                    'Cell_Line': row['Cell_Line'],
                    'cell_used': row['cell_used'],
                    'Value_Type': row['Value_Type'],
                    'Value': row['Value'],
                    'Curated_Name': row['Curated_Name'],
                    'pdp_active_mean': None,
                    'pdp_majority_vote': None,
                    'error': str(e)[:200],
                })
        if (i // BATCH) % 10 == 0:
            print(f"  Progress: {min(i+BATCH, len(sub))}/{len(sub)} ({time.time()-t0:.0f}s)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Successful: {out_df['pdp_active_mean'].notna().sum()}")
    print(f"  Errors: {out_df['pdp_active_mean'].isna().sum()}")

    # JSON too
    out_df.to_json(OUT_JSON, orient='records', indent=2)
    print(f"Saved JSON: {OUT_JSON}")

if __name__ == '__main__':
    main()
