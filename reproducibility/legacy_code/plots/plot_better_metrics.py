#!/usr/bin/env python3
"""
Better-metric plots for generalized CAT model (DC50 + Dmax).
Shows why R² is misleading for OOD: ranking + fold-error metrics reveal real signal.
Output: experiments/gen_training_v4/metrics_better/figures/
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/experiments/gen_training_v4/metrics_better"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

SPLITS = ["random", "scaffold_held_out", "e3_held_out", "target_held_out"]
SL = {"random": "Random", "scaffold_held_out": "Scaffold-held", "e3_held_out": "E3-held (VHL)",
      "target_held_out": "Target-held"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    df = pd.read_csv(f"{OUT}/split_metrics_better.csv")
    ext = pd.read_csv(f"{OUT}/external_metrics_better.csv")

    for task, tlabel, key in [("dc50", "DC50 (log10 nM)", "median_abs_log"),
                              ("dmax", "Dmax (%)", "median_abs_err"),
                              ("dmax90", "Dmax ≤90% (%)", "median_abs_err")]:
        sub = df[df['task'] == task].set_index('split').reindex(SPLITS)

        # ═══ FIG: R² vs ranking metrics — the core message ═══
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        x = np.arange(len(SPLITS))

        axes[0].bar(x, sub['R2'], color=['#2c7fb8' if v > 0 else '#d62728' for v in sub['R2']], alpha=0.85)
        axes[0].axhline(0, color='black', lw=0.8)
        for i, v in enumerate(sub['R2']):
            axes[0].text(i, v + (0.02 if v >= 0 else -0.05), f'{v:.2f}', ha='center', fontsize=9)
        axes[0].set_xticks(x); axes[0].set_xticklabels([SL[s] for s in SPLITS], rotation=15)
        axes[0].set_title('R² — MISLEADING for OOD\n(punishes calibration; can go negative with real signal)')
        axes[0].set_ylabel('R²')

        axes[1].bar(x, sub['spearman_rho'], color=['#2c7fb8' if v > 0 else '#d62728' for v in sub['spearman_rho']], alpha=0.85)
        axes[1].axhline(0, color='black', lw=0.8)
        for i, v in enumerate(sub['spearman_rho']):
            axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        axes[1].set_xticks(x); axes[1].set_xticklabels([SL[s] for s in SPLITS], rotation=15)
        axes[1].set_title('Spearman ρ — ranking signal\n(what matters for prioritization)')
        axes[1].set_ylabel('Spearman ρ')

        axes[2].bar(x, sub['cindex'], color='#41ab5d', alpha=0.85)
        axes[2].axhline(0.5, color='red', ls='--', lw=1)
        for i, v in enumerate(sub['cindex']):
            axes[2].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
        axes[2].set_xticks(x); axes[2].set_xticklabels([SL[s] for s in SPLITS], rotation=15)
        axes[2].set_title('Pairwise ranking accuracy (C-index)\n(0.5 = chance)')
        axes[2].set_ylabel('C-index')

        fig.suptitle(f'{tlabel} — generalized CAT: R² vs ranking metrics by split', fontsize=13)
        fig.tight_layout()
        save(fig, f"fig_rank_{task}")

        # ═══ FIG: fold-error / within-x-fold for DC50 ═══
        if task == 'dc50':
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(SPLITS)); w = 0.26
            ax.bar(x - w, sub['pct_within_2x'], w, label='Within 2×', color='#2c7fb8', alpha=0.9)
            ax.bar(x, sub['pct_within_5x'], w, label='Within 5×', color='#41ab5d', alpha=0.9)
            ax.bar(x + w, sub['pct_within_10x'], w, label='Within 10×', color='#feb24c', alpha=0.9)
            for i, (a, b, c) in enumerate(zip(sub['pct_within_2x'], sub['pct_within_5x'], sub['pct_within_10x'])):
                ax.text(i-w, a+1, f'{a:.0f}', ha='center', fontsize=8)
                ax.text(i, b+1, f'{b:.0f}', ha='center', fontsize=8)
                ax.text(i+w, c+1, f'{c:.0f}', ha='center', fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([SL[s] for s in SPLITS])
            ax.set_ylabel('% of predictions within error')
            ax.set_title('DC50 prediction accuracy — % within fold-error (held-out splits)')
            ax.legend()
            fig.tight_layout()
            save(fig, "fig_fold_error_dc50")

        # ═══ FIG: external validation ranking ═══
        ext_sub = ext[ext['task'] == task].copy()
        if len(ext_sub):
            fig, ax = plt.subplots(figsize=(9, 5))
            ext_sub['label'] = ext_sub['split'].str.replace('external:', '')
            el = {'random': 'Random', 'scaffold_held_out': 'Scaffold-held', 'e3_held_out': 'E3-held',
                  'target_held_out': 'Target-held', 'ensemble': 'Ensemble'}
            ext_sub['label'] = ext_sub['label'].map(el)
            vals = ext_sub[['label', 'spearman_rho', 'cindex', 'n_test']].dropna()
            x = np.arange(len(vals))
            ax.bar(x - 0.2, vals['spearman_rho'], 0.4, label='Spearman ρ', color='#2c7fb8', alpha=0.9)
            ax.bar(x + 0.2, vals['cindex'], 0.4, label='C-index', color='#41ab5d', alpha=0.9)
            ax.axhline(0, color='black', lw=0.8)
            for i, (r, c, n) in enumerate(zip(vals['spearman_rho'], vals['cindex'], vals['n_test'])):
                ax.text(i-0.2, r + 0.01, f'{r:.2f}', ha='center', fontsize=8)
                ax.text(i+0.2, c + 0.01, f'{c:.2f}', ha='center', fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(vals['label'], rotation=15)
            ax.set_ylabel('Ranking metric')
            ax.set_title(f'External validation (n≈{int(vals["n_test"].median())} leakage-free) — {tlabel}')
            ax.legend()
            fig.tight_layout()
            save(fig, f"fig_external_rank_{task}")

    print(f"\nFigures saved to {FIG}")

if __name__ == '__main__':
    main()
