#!/usr/bin/env python3
"""
fig1_developability (v2) — ALL tools are pocket-guided de novo warhead generators.
No misleading paradigm split: single paradigm shading, all warhead generators for GSPT1.

fig4_stage_survival (replaces confusing cascade funnel) — stage-by-stage survival rate:
% of generated candidates surviving each filter stage (Validity → Physchem → SA → Developability)
as grouped bars per tool.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
ROOT = f"{REV}/benchmarking/gcoupler_vs_tools"
RES = f"{ROOT}/results"; FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol"]
COL = {"Gcoupler": "#2c7fb8", "Pharmacophore": "#41ab5d",
       "DiffSBDD": "#d95f0e", "Pocket2Mol": "#d62728"}

def main():
    master = pd.read_csv(f"{RES}/MASTER_COMPARISON.csv").set_index('tool')
    tools = [t for t in TOOLS if t in master.index]

    # ══════════════════ FIG 1 (v2): all warhead generators ══════════════════
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = [float(master.loc[t, 'developability%']) for t in tools]
    bars = ax.bar(tools, vals, width=0.62, color=[COL[t] for t in tools], alpha=0.92, zorder=3,
                  edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 1.2, f'{v:.1f}%', ha='center', fontsize=11, weight='bold')

    # Single paradigm: all are pocket-guided de novo warhead generators
    ax.axvspan(-0.45, len(tools)-0.55, color='#2c7fb8', alpha=0.05, zorder=0)

    # Annotate generated counts below names
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    for i, t in enumerate(tools):
        n = int(master.loc[t, 'n_generated'])
        ax.text(i, -8, f'{n:,} generated', ha='center', fontsize=9, color='#555555')
    ax.set_ylim(0, 62)

    ax.set_ylabel('Developability pass rate (%)')
    ax.set_title('Pocket-guided de novo warhead generators — downstream developability pass rate\n'
                 '(GSPT1, PDB 5HXB; validity + physchem + SA + linker filters)')
    # legend: single paradigm note
    handles = [Patch(facecolor='#2c7fb8', alpha=0.1, label='all: pocket-guided de novo (GSPT1 warhead generation)')]
    ax.legend(handles=handles, fontsize=9, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_developability.png")
    fig.savefig(f"{FIG}/fig1_developability.svg")
    plt.close(fig)
    print("fig1_developability regenerated — all pocket-guided de novo warhead generators")

    # ══════════════════ FIG 4 (new): stage survival ══════════════════
    # Stage rates (% of generated surviving each stage)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    stages = ['Validity', 'Physchem', 'SA ≤ 5.0', 'Developability']
    x = np.arange(len(tools))
    width = 0.2

    for j, (stage, col_m, color) in enumerate([('Validity', 'validity%', '#969696'),
                                                 ('Physchem', 'physchem_pass%', '#feb24c'),
                                                 ('SA ≤ 5.0', 'SA_pass%', '#31a354'),
                                                 ('Developability', 'developability%', '#2c7fb8')]):
        vals = []
        for t in tools:
            v = master.loc[t, col_m]
            vals.append(float(v))
        bars = ax.bar(x + (j-1.5)*width, vals, width, label=stage, color=color, alpha=0.9,
                      edgecolor='black', linewidth=0.4, zorder=3)
        for b, v in zip(bars, vals):
            if v > 2:
                ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.0f}%', ha='center', fontsize=7.5, rotation=90)

    # stack generated counts as annotations
    for i, t in enumerate(tools):
        ax.text(i, -18, f"{int(master.loc[t,'n_generated']):,}", ha='center', fontsize=10, color='#555555', weight='bold')

    ax.set_xticks(x); ax.set_xticklabels(tools, fontsize=11)
    ax.set_ylabel('% of generated candidates passing stage')
    ax.set_ylim(0, 112)
    ax.set_title('Stage-by-stage filter survival — pocket-guided de novo warhead generators (GSPT1)')
    ax.legend(fontsize=8.5, ncol=4, loc='upper center')
    ax.text(0.0, -26, 'bottom numbers = total generated candidates', transform=ax.get_xaxis_transform(),
            fontsize=8.5, color='#777777')
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_stage_survival.png")
    fig.savefig(f"{FIG}/fig4_stage_survival.svg")
    plt.close(fig)
    print("fig4_stage_survival created (replaces confusing funnel)")

    # remove old funnel figure
    for ext in ['png', 'svg']:
        old = f"{FIG}/fig4_cascade_funnel.{ext}"
        if os.path.exists(old):
            os.remove(old)
            print(f"removed {old}")

if __name__ == '__main__':
    main()
