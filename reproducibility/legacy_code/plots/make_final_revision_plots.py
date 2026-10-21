#!/usr/bin/env python3
"""
FINAL REVISION PLOTS — frozen ProtacAttentionModel
A. Better results plots (DC50 & Dmax): scatter, residual, metric bars
B. Unique interpretability: attention + ablation + per-split performance
C. 4-split generalization plots
Output: figures/final_results/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/final_results"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    # ── Load results ──
    four = pd.read_csv(f"{OUT}/four_split_generalization.csv")
    interp = json.load(open(f"{OUT}/interpretability.json"))
    attn = np.load(f"{OUT}/attention_weights.npy").squeeze(-1)
    both = json.load(open(f"{OUT}/final_metrics_both.json"))
    preds = {t: pd.read_csv(f"{OUT}/preds_{t}_test.csv") for t in ['dc50','dmax']}

    labels = ["Warhead", "Linker", "E3 ligase"]

    # ═══ A1: DC50 predicted vs actual (test) ═══
    fig, ax = plt.subplots(figsize=(7, 6))
    d = preds['dc50']
    ax.scatter(d['true'], d['pred'], s=14, alpha=0.5, color='#2c7fb8', edgecolor='none')
    lims = [min(d['true'].min(), d['pred'].min()), max(d['true'].max(), d['pred'].max())]
    ax.plot(lims, lims, 'r--', lw=1.2, label='y = x')
    r2 = both['dc50']['R2']; rho = both['dc50']['rho']
    ax.set_xlabel('Reported log10(DC50 / nM)')
    ax.set_ylabel('Predicted log10(DC50 / nM)')
    ax.set_title(f'DC50 — frozen ProtacAttentionModel + RF\nR²={r2:.3f}  ρ={rho:.3f}  (test n={len(d)})')
    ax.legend()
    fig.tight_layout(); save(fig, "A1_dc50_scatter")

    # ═══ A2: Dmax predicted vs actual ═══
    fig, ax = plt.subplots(figsize=(7, 6))
    d = preds['dmax']
    ax.scatter(d['true'], d['pred'], s=14, alpha=0.5, color='#d95f0e', edgecolor='none')
    lims = [min(d['true'].min(), d['pred'].min()), max(d['true'].max(), d['pred'].max())]
    ax.plot(lims, lims, 'r--', lw=1.2)
    r2 = both['dmax']['R2']; rho = both['dmax']['rho']
    ax.set_xlabel('Reported Dmax (%)')
    ax.set_ylabel('Predicted Dmax (%)')
    ax.set_title(f'Dmax — frozen model\nR²={r2:.3f}  ρ={rho:.3f}  (test n={len(d)})')
    fig.tight_layout(); save(fig, "A2_dmax_scatter")

    # ═══ A3: metric comparison bar (both endpoints) ═══
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    eps = ['dc50', 'dmax']
    metric_keys = [('R2', 'R²'), ('MAE', 'MAE'), ('rho', 'Spearman ρ')]
    for ax, (k, lab) in zip(axes, metric_keys):
        v50 = both['dc50'].get(k, np.nan); v95 = both['dmax'].get(k, np.nan)
        bars = ax.bar(['DC50', 'Dmax'], [v50, v95], color=['#2c7fb8', '#d95f0e'], alpha=0.9)
        for b, v in zip(bars, [v50, v95]):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=10, weight='bold')
        ax.set_ylabel(lab); ax.set_title(f'{lab} (test)')
    fig.suptitle('Both-endpoint performance — frozen ProtacAttentionModel', fontsize=13)
    fig.tight_layout(); save(fig, "A3_endpoint_metrics")

    # ═══ B1: unique interpretability — attention (donut + bar) ═══
    a = interp['attention']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    vals = [a['W'], a['L'], a['E']]
    ax.pie(vals, labels=labels, autopct='%.1f%%', colors=['#2c7fb8','#d95f0e','#41ab5d'],
           startangle=90, wedgeprops=dict(width=0.45, edgecolor='white'))
    ax.set_title('Attention allocation (donut)')
    ax = axes[1]
    stds = [attn[:,0].std(), attn[:,1].std(), attn[:,2].std()]
    bars = ax.bar(labels, vals, yerr=stds, capsize=6, color=['#2c7fb8','#d95f0e','#41ab5d'], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=11, weight='bold')
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Component attention (mean ± std)')
    fig.suptitle('Interpretability — component attention (frozen model)', fontsize=13)
    fig.tight_layout(); save(fig, "B1_attention_donut_bar")

    # ═══ B2: ablation (unique: horizontal + delta) ═══
    ab = interp['ablation']
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vals = [ab['Warhead'], ab['Linker'], ab['E3']]
    bars = ax.barh(labels[::-1], vals[::-1], color=['#41ab5d','#d95f0e','#2c7fb8'][::-1], alpha=0.9)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v+0.01, b.get_y()+b.get_height()/2, f'{v:.2f}', va='center', fontsize=11, weight='bold')
    ax.set_xlabel('Mean |Δ log10(DC50)| — zeroing component token')
    ax.set_title('Component ablation — prediction sensitivity')
    fig.tight_layout(); save(fig, "B2_ablation")

    # ═══ B3: attention heatmap (unique: sorted by linker weight) ═══
    fig, ax = plt.subplots(figsize=(9, 5))
    n = min(300, attn.shape[0])
    order = np.argsort(attn[:n, 1])  # sort by linker
    im = ax.imshow(attn[:n][order].T, aspect='auto', cmap='viridis')
    ax.set_yticks([0,1,2]); ax.set_yticklabels(labels)
    ax.set_xlabel('Sample (sorted by linker attention)')
    ax.set_title('Attention heatmap — linker-sorted (frozen model)')
    fig.colorbar(im, ax=ax, label='Attention')
    fig.tight_layout(); save(fig, "B3_attention_heatmap_sorted")

    # ═══ C1: 4-split generalization — DC50 ═══
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ep in zip(axes, ['DC50', 'Dmax']):
        sub = four[four['endpoint'] == ep]
        splits = sub['split'].tolist()
        x = np.arange(len(splits))
        ax.bar(x, sub['rho'], color=['#2c7fb8' if v > 0 else '#d62728' for v in sub['rho']], alpha=0.9)
        ax.axhline(0, color='black', lw=0.8)
        for i, (r, n) in enumerate(zip(sub['rho'], sub['n'])):
            ax.text(i, r+0.02, f'ρ={r:.2f}\nn={n}', ha='center', fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha='right')
        ax.set_ylabel('Spearman ρ'); ax.set_title(f'{ep} — 4-split generalization (frozen model)')
    fig.tight_layout(); save(fig, "C1_four_split_rho")

    # ═══ C2: R² by split ═══
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ep in zip(axes, ['DC50', 'Dmax']):
        sub = four[four['endpoint'] == ep]
        x = np.arange(len(sub))
        ax.bar(x, sub['R2'], color=['#2c7fb8' if v > 0 else '#d62728' for v in sub['R2']], alpha=0.9)
        ax.axhline(0, color='black', lw=0.8)
        for i, v in enumerate(sub['R2']):
            ax.text(i, v + (0.02 if v >= 0 else -0.05), f'{v:.2f}', ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(sub['split'], rotation=20, ha='right')
        ax.set_ylabel('R²'); ax.set_title(f'{ep} — R² by split (frozen model)')
    fig.tight_layout(); save(fig, "C2_four_split_R2")

    # ═══ C3: split counts bar ═══
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ep in ['DC50', 'Dmax']:
        sub = four[four['endpoint'] == ep]
        ax.bar([f"{s}\n({ep})" for s in sub['split']], sub['n'],
               color=['#2c7fb8' if ep=='DC50' else '#d95f0e']*len(sub), alpha=0.85)
    ax.set_ylabel('Test compounds'); ax.set_title('4-split test counts (frozen model)')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(); save(fig, "C3_split_counts")

    print(f"\nAll final plots saved to {FIG}")

if __name__ == '__main__':
    main()
