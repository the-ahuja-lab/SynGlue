#!/usr/bin/env python3
"""
Final merge: external_not_in_training + PDP + PROTAC-STAN + SynGlue(v2)
Build unified comparison CSV + metrics table.
"""
import os, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
OUT_DIR = f"{BASE}/revision/benchmarking/tool_results"

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr, pearsonr

def canon(s):
    if not isinstance(s, str) or not s: return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, canonical=True) if mol else s

def main():
    source = pd.read_csv(f"{BASE}/revision/benchmarking/external_not_in_training.csv", low_memory=False)
    pdp = pd.read_csv(f"{OUT_DIR}/pdp_external_predictions.csv")
    stan = pd.read_csv(f"{OUT_DIR}/protac_stan_external_predictions.csv")
    sg = pd.read_csv(f"{OUT_DIR}/synglue_external_predictions_v2.csv")

    source['canon'] = source['canonical_SMILES'].apply(canon)
    pdp['canon'] = pdp['canonical_SMILES'].apply(canon)
    stan['canon'] = stan['SMILES_source'].apply(canon)
    sg['canon'] = sg['canonical_SMILES'].apply(canon)

    # Aggregate per (canon, POI, E3) — keep best DC50 per compound
    pdp_agg = pdp.groupby('canon').agg(pdp_active_mean=('pdp_active_mean','mean'),
                                       pdp_vote=('pdp_majority_vote','mean'),
                                       pdp_n=('pdp_active_mean','count')).reset_index()
    stan_agg = stan.groupby('canon').agg(stan_p_active=('stan_p_active','mean'),
                                         stan_class=('stan_pred_class','mean'),
                                         stan_n=('stan_p_active','count')).reset_index()
    sg_agg = sg.groupby('canon').agg(sg_tf_dc50=('synglue_tf_dc50_nM','mean'),
                                     sg_tf_dmax=('synglue_tf_dmax_pct','mean'),
                                     sg_rf_dc50=('synglue_rf_dc50_nM','mean'),
                                     sg_rf_dmax=('synglue_rf_dmax_pct','mean')).reset_index()

    merged = source.merge(pdp_agg, on='canon', how='left') \
                   .merge(stan_agg, on='canon', how='left') \
                   .merge(sg_agg, on='canon', how='left')
    print(f"Merged: {len(merged)} rows")
    print(f"  PDP: {merged['pdp_active_mean'].notna().sum()}")
    print(f"  STAN: {merged['stan_p_active'].notna().sum()}")
    print(f"  SynGlue: {merged['sg_tf_dc50'].notna().sum()}")

    # Save
    out_file = f"{OUT_DIR}/external_merged_predictions_v2.csv"
    merged.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    # ---- Metrics vs reported DC50 (exact values only) ----
    dc50 = merged[merged['Value_Type'] == 'DC50'].copy()
    dc50['value_num'] = pd.to_numeric(dc50['Value'], errors='coerce')
    dc50 = dc50[dc50['value_num'].notna()]
    exact = dc50[dc50['Value_Operator'].isna()].copy()
    exact['logv'] = np.log10(exact['value_num'])
    print(f"\n=== Correlation with reported DC50 (exact, n={len(exact)}) ===")
    print("Direction: NEGATIVE r = more potent (lower DC50) → higher activity score (correct)")

    rows = []
    # PDP
    d = exact[exact['pdp_active_mean'].notna()]
    r_p, p_p = pearsonr(d['logv'], d['pdp_active_mean'])
    r_s, p_s = spearmanr(d['logv'], d['pdp_active_mean'])
    rows.append(('PDP active-prob', len(d), r_p, p_p, r_s, p_s))
    # STAN
    d = exact[exact['stan_p_active'].notna()]
    r_p, p_p = pearsonr(d['logv'], d['stan_p_active'])
    r_s, p_s = spearmanr(d['logv'], d['stan_p_active'])
    rows.append(('PROTAC-STAN P(active)', len(d), r_p, p_p, r_s, p_s))
    # SynGlue TF
    d = exact[exact['sg_tf_dc50'].notna()].copy()
    d['logp'] = np.log10(d['sg_tf_dc50'])
    r_p, p_p = pearsonr(d['logv'], d['logp'])
    r_s, p_s = spearmanr(d['logv'], d['logp'])
    rows.append(('SynGlue-TF DC50', len(d), r_p, p_p, r_s, p_s))
    # SynGlue RF
    d = exact[exact['sg_rf_dc50'].notna()].copy()
    d['logp'] = np.log10(d['sg_rf_dc50'])
    r_p, p_p = pearsonr(d['logv'], d['logp'])
    r_s, p_s = spearmanr(d['logv'], d['logp'])
    rows.append(('SynGlue-RF DC50', len(d), r_p, p_p, r_s, p_s))

    print(f"{'Tool':<22} {'n':>5} {'Pearson r':>10} {'p':>10} {'Spearman':>9} {'p':>10}")
    for name, n, rp, pp, rs, ps in rows:
        print(f"{name:<22} {n:>5} {rp:>10.3f} {pp:>10.1e} {rs:>9.3f} {ps:>10.1e}")

    # ---- Cross-tool agreement ----
    both = merged[merged['pdp_active_mean'].notna() & merged['stan_p_active'].notna()].copy()
    both['pdp_flag'] = both['pdp_active_mean'] > 0.5
    both['stan_flag'] = both['stan_p_active'] > 0.5
    agree = (both['pdp_flag'] == both['stan_flag']).mean()
    r_ps = spearmanr(both['pdp_active_mean'], both['stan_p_active'])[0]
    print(f"\n=== Cross-tool agreement (n={len(both)}) ===")
    print(f"PDP vs STAN binary agreement: {agree:.1%}")
    print(f"PDP vs STAN prob Spearman: {r_ps:.3f}")

    # Save metrics
    metrics = pd.DataFrame(rows, columns=['Tool','n','pearson_r','pearson_p','spearman_r','spearman_p'])
    metrics.to_csv(f"{OUT_DIR}/external_metrics_summary.csv", index=False)
    print(f"\nSaved metrics: {OUT_DIR}/external_metrics_summary.csv")

if __name__ == '__main__':
    main()
