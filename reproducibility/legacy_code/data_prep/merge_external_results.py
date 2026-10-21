#!/usr/bin/env python3
"""
Merge predictions from PDP, PROTAC-STAN, SynGlue on external compounds
NOT in training data. Build unified comparison table + agreement metrics.
"""
import os, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
OUT_DIR = f"{BASE}/revision/benchmarking/tool_results"

import numpy as np
import pandas as pd

def main():
    # ---- Load each tool's predictions ----
    pdp = pd.read_csv(f"{OUT_DIR}/pdp_external_predictions.csv")
    stan = pd.read_csv(f"{OUT_DIR}/protac_stan_external_predictions.csv")
    synglue = pd.read_csv(f"{OUT_DIR}/synglue_external_predictions.csv")
    source = pd.read_csv(f"{BASE}/revision/benchmarking/external_not_in_training.csv", low_memory=False)

    print(f"PDP rows: {len(pdp)}")
    print(f"PROTAC-STAN rows: {len(stan)}")
    print(f"SynGlue rows: {len(synglue)}")
    print(f"Source rows: {len(source)}")

    # ---- Canonical key ----
    def canon(s):
        from rdkit import Chem
        if not isinstance(s, str) or not s:
            return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s

    for df, col in [(pdp, 'canonical_SMILES'), (stan, 'SMILES_source'), (synglue, 'canonical_SMILES')]:
        df['canon'] = df[col].apply(canon)

    # ---- Merge on canonical SMILES ----
    merged = source.copy()
    merged['canon'] = merged['canonical_SMILES'].apply(canon)

    # PDP: one row per (canon, POI, E3) — aggregate by taking max value across duplicates
    pdp_agg = pdp.groupby('canon').agg(
        pdp_active_mean=('pdp_active_mean', 'mean'),
        pdp_majority_vote=('pdp_majority_vote', 'mean'),
        pdp_n=('pdp_active_mean', 'count')
    ).reset_index()
    merged = merged.merge(pdp_agg, on='canon', how='left')

    # PROTAC-STAN: aggregate
    stan_agg = stan.groupby('canon').agg(
        stan_p_active=('stan_p_active', 'mean'),
        stan_pred_class=('stan_pred_class', 'mean'),
        stan_n=('stan_p_active', 'count')
    ).reset_index()
    merged = merged.merge(stan_agg, on='canon', how='left')

    # SynGlue: aggregate
    synglue_agg = synglue.groupby('canon').agg(
        synglue_tf_dc50_nM=('synglue_tf_dc50_nM', 'mean'),
        synglue_tf_dmax_pct=('synglue_tf_dmax_pct', 'mean'),
        synglue_rf_dc50_nM=('synglue_rf_dc50_nM', 'mean'),
        synglue_rf_dmax_pct=('synglue_rf_dmax_pct', 'mean'),
    ).reset_index()
    merged = merged.merge(synglue_agg, on='canon', how='left')

    print(f"\nMerged rows: {len(merged)}")
    print(f"Rows with PDP: {merged['pdp_active_mean'].notna().sum()}")
    print(f"Rows with PROTAC-STAN: {merged['stan_p_active'].notna().sum()}")
    print(f"Rows with SynGlue: {merged['synglue_tf_dc50_nM'].notna().sum()}")

    # ---- Save merged ----
    out_file = f"{OUT_DIR}/external_merged_predictions.csv"
    merged.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

    # ---- Agreement metrics ----
    print("\n" + "="*60)
    print("AGREEMENT METRICS (rows with all 3 tools)")
    print("="*60)
    both = merged[merged['pdp_active_mean'].notna() & merged['stan_p_active'].notna() & merged['synglue_tf_dc50_nM'].notna()].copy()
    print(f"N with all 3 tools: {len(both)}")

    # PDP vote (majority > 0.5) vs PROTAC-STAN class (1=active)
    both['pdp_active_flag'] = both['pdp_active_mean'] > 0.5
    both['stan_active_flag'] = both['stan_pred_class'] > 0.5
    if len(both) > 0:
        agree = (both['pdp_active_flag'] == both['stan_active_flag']).mean()
        print(f"PDP vs PROTAC-STAN agreement: {agree:.1%}")

    # Correlation between PDP prob and STAN prob
    corr = both[['pdp_active_mean', 'stan_p_active']].corr().iloc[0, 1]
    print(f"PDP prob vs STAN prob correlation: {corr:.3f}")

    # DC50 relationship: compounds with known DC50 — check SynGlue prediction correlation
    dc50 = both[both['Value_Type'] == 'DC50'].copy()
    dc50['log_value'] = np.log10(pd.to_numeric(dc50['Value'], errors='coerce'))
    dc50 = dc50[dc50['log_value'].notna() & dc50['synglue_tf_dc50_nM'].notna()]
    if len(dc50) > 5:
        corr_dc50 = dc50[['log_value', 'synglue_tf_dc50_nM']].corr()
        corr_rf = dc50[['log_value', 'synglue_rf_dc50_nM']].corr()
        print(f"\nDC50 vs SynGlue transformer (log-log): r = {corr_dc50.iloc[0,1]:.3f} (n={len(dc50)})")
        print(f"DC50 vs SynGlue RF (log-log): r = {corr_rf.iloc[0,1]:.3f}")

    # Value range summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"PDP mean active prob: {merged['pdp_active_mean'].mean():.3f}")
    print(f"PROTAC-STAN mean P(active): {merged['stan_p_active'].mean():.3f}")
    print(f"SynGlue TF DC50 mean: {merged['synglue_tf_dc50_nM'].mean():.0f} nM")
    print(f"SynGlue RF DC50 mean: {merged['synglue_rf_dc50_nM'].mean():.0f} nM")

if __name__ == '__main__':
    main()
