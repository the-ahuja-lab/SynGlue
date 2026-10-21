#!/usr/bin/env python3
"""
Generate descriptive stats + analysis plots for the external benchmark
(external_not_in_training set, 4,645 rows / 2,640 compounds).

Output: revision/figures/external_benchmark/
"""
import os, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
MERGED = f"{BASE}/revision/benchmarking/tool_results/external_merged_predictions_v2.csv"
OUTDIR = f"{BASE}/revision/figures/external_benchmark"
os.makedirs(OUTDIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, pearsonr

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

C_PDP = '#1f77b4'
C_STAN = '#ff7f0e'
C_SG = '#2ca02c'
C_SG_RF = '#d62728'

def main():
    m = pd.read_csv(MERGED, low_memory=False)
    print(f"Loaded {len(m)} rows")

    # ---- 1. DESCRIPTIVE: value distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    v = pd.to_numeric(m['Value'], errors='coerce')
    dc50 = v[m['Value_Type'] == 'DC50'].dropna()
    dmax = v[m['Value_Type'] == 'Dmax'].dropna()

    axes[0].hist(np.log10(dc50), bins=40, color=C_PDP, alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('log10(DC50 / nM)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Reported DC50 (n={len(dc50)})')
    axes[0].axvline(np.log10(dc50.median()), color='red', ls='--', lw=1, label=f'median={dc50.median():.1f} nM')
    axes[0].legend()

    axes[1].hist(dmax, bins=40, color=C_STAN, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Dmax (%)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Reported Dmax (n={len(dmax)})')
    axes[1].axvline(dmax.median(), color='red', ls='--', lw=1, label=f'median={dmax.median():.1f}%')
    axes[1].legend()
    fig.suptitle('External set — reported degradation values (NOT in training)', fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_reported_values_distributions.png")
    plt.close(fig)

    # ---- 2. DESCRIPTIVE: target distribution ----
    poi_counts = m['Protein_of_Interest'].value_counts()
    top_poi = poi_counts.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_poi.index[::-1], top_poi.values[::-1], color=C_SG, alpha=0.85)
    ax.set_xlabel('Number of records')
    ax.set_title(f'Target protein distribution (top 15 of {len(poi_counts)} targets, n={len(m)})')
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_target_distribution.png")
    plt.close(fig)

    # ---- 3. DESCRIPTIVE: E3 + cell line ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    e3_counts = m['E3_Ligase'].value_counts()
    axes[0].bar(e3_counts.index, e3_counts.values, color=[C_PDP, C_STAN, C_SG, C_SG_RF])
    axes[0].set_ylabel('Count')
    axes[0].set_title('E3 ligase distribution')
    axes[0].tick_params(axis='x', rotation=30)

    cell_counts = m['Cell_Line'].value_counts().head(10)
    axes[1].barh(cell_counts.index[::-1], cell_counts.values[::-1], color='#7f7f7f', alpha=0.85)
    axes[1].set_xlabel('Count')
    axes[1].set_title('Top 10 cell lines')
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_e3_cellline_distribution.png")
    plt.close(fig)

    # ---- 4. TOOL PREDICTION DISTRIBUTIONS ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # PDP
    axes[0,0].hist(m['pdp_active_mean'].dropna(), bins=40, color=C_PDP, alpha=0.85, edgecolor='white')
    axes[0,0].axvline(0.5, color='red', ls='--', lw=1)
    axes[0,0].set_xlabel('PDP P(active)')
    axes[0,0].set_title(f'PDP (n={m["pdp_active_mean"].notna().sum()})')
    # STAN
    axes[0,1].hist(m['stan_p_active'].dropna(), bins=40, color=C_STAN, alpha=0.85, edgecolor='white')
    axes[0,1].axvline(0.5, color='red', ls='--', lw=1)
    axes[0,1].set_xlabel('PROTAC-STAN P(active)')
    axes[0,1].set_title(f'PROTAC-STAN (n={m["stan_p_active"].notna().sum()})')
    # SynGlue TF
    sg_dc = m['sg_tf_dc50'].dropna()
    axes[1,0].hist(np.log10(sg_dc), bins=40, color=C_SG, alpha=0.85, edgecolor='white')
    axes[1,0].set_xlabel('log10(SynGlue-TF DC50 / nM)')
    axes[1,0].set_title(f'SynGlue-TF DC50 (n={len(sg_dc)})')
    # SynGlue RF
    sg_rf = m['sg_rf_dc50'].dropna()
    axes[1,1].hist(np.log10(sg_rf), bins=40, color=C_SG_RF, alpha=0.85, edgecolor='white')
    axes[1,1].set_xlabel('log10(SynGlue-RF DC50 / nM)')
    axes[1,1].set_title(f'SynGlue-RF DC50 (n={len(sg_rf)})')
    fig.suptitle('Tool prediction distributions (external set)', fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_tool_prediction_distributions.png")
    plt.close(fig)

    # ---- 5. CORRELATION SCATTERS vs reported DC50 ----
    # Build exact-DC50 (no operator) compound-level frame
    exact = m[m['Value_Type'] == 'DC50'].copy()
    exact['value_num'] = pd.to_numeric(exact['Value'], errors='coerce')
    exact = exact[exact['value_num'].notna() & exact['Value_Operator'].isna()].copy()
    exact['logv'] = np.log10(exact['value_num'])
    exact = exact.drop_duplicates(subset=['canon'])
    print(f"Exact DC50 compounds: {len(exact)}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # PDP
    d = exact[exact['pdp_active_mean'].notna()]
    r = pearsonr(d['logv'], d['pdp_active_mean'])[0]
    rho = spearmanr(d['logv'], d['pdp_active_mean'])[0]
    axes[0,0].scatter(d['logv'], d['pdp_active_mean'], s=8, alpha=0.4, color=C_PDP)
    axes[0,0].set_xlabel('log10(reported DC50 / nM)')
    axes[0,0].set_ylabel('PDP P(active)')
    axes[0,0].set_title(f'PDP  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})')

    # STAN
    d = exact[exact['stan_p_active'].notna()]
    r = pearsonr(d['logv'], d['stan_p_active'])[0]
    rho = spearmanr(d['logv'], d['stan_p_active'])[0]
    axes[0,1].scatter(d['logv'], d['stan_p_active'], s=8, alpha=0.4, color=C_STAN)
    axes[0,1].set_xlabel('log10(reported DC50 / nM)')
    axes[0,1].set_ylabel('PROTAC-STAN P(active)')
    axes[0,1].set_title(f'PROTAC-STAN  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})')

    # SynGlue TF
    d = exact[exact['sg_tf_dc50'].notna()].copy()
    d['logp'] = np.log10(d['sg_tf_dc50'])
    r = pearsonr(d['logv'], d['logp'])[0]
    rho = spearmanr(d['logv'], d['logp'])[0]
    axes[1,0].scatter(d['logv'], d['logp'], s=8, alpha=0.4, color=C_SG)
    axes[1,0].plot([0, 5], [0, 5], 'r--', lw=1, alpha=0.6)
    axes[1,0].set_xlabel('log10(reported DC50 / nM)')
    axes[1,0].set_ylabel('log10(SynGlue-TF DC50 / nM)')
    axes[1,0].set_title(f'SynGlue-TF  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})')

    # SynGlue RF
    d = exact[exact['sg_rf_dc50'].notna()].copy()
    d['logp'] = np.log10(d['sg_rf_dc50'])
    r = pearsonr(d['logv'], d['logp'])[0]
    rho = spearmanr(d['logv'], d['logp'])[0]
    axes[1,1].scatter(d['logv'], d['logp'], s=8, alpha=0.4, color=C_SG_RF)
    axes[1,1].plot([0, 5], [0, 5], 'r--', lw=1, alpha=0.6)
    axes[1,1].set_xlabel('log10(reported DC50 / nM)')
    axes[1,1].set_ylabel('log10(SynGlue-RF DC50 / nM)')
    axes[1,1].set_title(f'SynGlue-RF  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})')

    fig.suptitle('Predicted vs reported DC50 (exact values, external non-training set)', fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig5_dc50_correlation_scatters.png")
    plt.close(fig)

    # ---- 6. TOOL COMPARISON BAR (Pearson r) ----
    metrics = pd.read_csv(f"{BASE}/revision/benchmarking/tool_results/external_metrics_summary_compound.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = metrics['Tool'].tolist()
    r_vals = metrics['pearson_r'].tolist()
    p_vals = metrics['pearson_p'].tolist()
    colors = [C_PDP, C_STAN, C_SG, C_SG_RF]
    bars = ax.bar(names, r_vals, color=colors, alpha=0.9, edgecolor='black', lw=0.5)
    for i, (b, r, p) in enumerate(zip(bars, r_vals, p_vals)):
        sig = '***' if p < 1e-10 else ('**' if p < 1e-4 else ('*' if p < 0.05 else 'n.s.'))
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + (0.01 if r >= 0 else -0.03),
                f'{r:.3f} {sig}', ha='center', va='bottom' if r >= 0 else 'top', fontsize=10)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Pearson r with reported DC50')
    ax.set_title('Tool correlation with reported DC50 (compound-level)\n(negative = correct direction: potent → active)')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig6_tool_comparison_pearson.png")
    plt.close(fig)

    # ---- 7. CROSS-TOOL AGREEMENT (PDP vs STAN) ----
    both = m[m['pdp_active_mean'].notna() & m['stan_p_active'].notna()].copy()
    both['pdp_flag'] = both['pdp_active_mean'] > 0.5
    both['stan_flag'] = both['stan_p_active'] > 0.5
    agree = (both['pdp_flag'] == both['stan_flag']).mean()
    rho = spearmanr(both['pdp_active_mean'], both['stan_p_active'])[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(both['pdp_active_mean'], both['stan_p_active'], s=8, alpha=0.35, color='#555555')
    axes[0].axvline(0.5, color='red', ls='--', lw=1)
    axes[0].axhline(0.5, color='red', ls='--', lw=1)
    axes[0].set_xlabel('PDP P(active)')
    axes[0].set_ylabel('PROTAC-STAN P(active)')
    axes[0].set_title(f'Cross-tool agreement (ρ={rho:.3f}, n={len(both)})')

    # confusion counts
    tp = ((both['pdp_flag']) & (both['stan_flag'])).sum()
    tn = ((~both['pdp_flag']) & (~both['stan_flag'])).sum()
    fp = ((both['pdp_flag']) & (~both['stan_flag'])).sum()
    fn = ((~both['pdp_flag']) & (both['stan_flag'])).sum()
    axes[1].bar(['both\nactive', 'both\ninactive', 'PDP only\nactive', 'STAN only\nactive'],
                [tp, tn, fp, fn], color=[C_PDP, 'grey', C_PDP, C_STAN], alpha=0.85)
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Binary agreement = {agree:.1%}')
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig7_cross_tool_agreement.png")
    plt.close(fig)

    # ---- 8. SynGlue DC50 vs Dmax (model internal consistency) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    d = m[m['sg_tf_dc50'].notna() & m['sg_tf_dmax'].notna()]
    sc = ax.scatter(np.log10(d['sg_tf_dc50']), d['sg_tf_dmax'], s=6, alpha=0.3, c=d['sg_rf_dc50'],
                    cmap='viridis', norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('log10(SynGlue-TF DC50 / nM)')
    ax.set_ylabel('SynGlue-TF Dmax (%)')
    ax.set_title('SynGlue internal predictions (color = RF DC50)')
    cb = fig.colorbar(sc, ax=ax, label='SynGlue-RF DC50 (nM)')
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig8_synglue_internal.png")
    plt.close(fig)

    print(f"\nAll figures saved to {OUTDIR}")
    for f in sorted(os.listdir(OUTDIR)):
        print(f"  {f}  ({os.path.getsize(os.path.join(OUTDIR, f))/1024:.0f} KB)")

if __name__ == '__main__':
    main()
