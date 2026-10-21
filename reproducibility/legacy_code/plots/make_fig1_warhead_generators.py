#!/usr/bin/env python3
"""
fig1_developability — warhead generators only (Gcoupler, Pharmacophore, DiffSBDD, Pocket2Mol)
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

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

# Warhead generators only
TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol"]
COL = {"Gcoupler": "#2c7fb8", "Pharmacophore": "#41ab5d",
       "DiffSBDD": "#d95f0e", "Pocket2Mol": "#d62728"}
PARADIGM = {"Gcoupler": "Assembly (fixed warhead)", "Pharmacophore": "Assembly (fixed warhead)",
            "DiffSBDD": "Pocket-guided de novo", "Pocket2Mol": "Pocket-guided de novo"}
PARA_COLOR = {"Assembly (fixed warhead)": "#2c7fb8", "Pocket-guided de novo": "#d95f0e"}

def main():
    master = pd.read_csv(f"{RES}/MASTER_COMPARISON.csv").set_index('tool')
    tools = [t for t in TOOLS if t in master.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Pass-rate bars
    vals = [float(master.loc[t, 'developability%']) for t in tools]
    bars = ax.bar(tools, vals, width=0.6, color=[COL[t] for t in tools], alpha=0.92, zorder=3,
                  edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 1.5, f'{v:.1f}%', ha='center', fontsize=11, weight='bold')

    # Paradigm shading
    groups = {}
    for i, t in enumerate(tools):
        groups.setdefault(PARADIGM[t], []).append(i)
    for p, idxs in groups.items():
        ax.axvspan(min(idxs)-0.45, max(idxs)+0.45, color=PARA_COLOR[p], alpha=0.08, zorder=0)

    ax.set_ylabel('Developability pass rate (%)')
    ax.set_ylim(0, 60)
    ax.set_title('Warhead-generator comparison — downstream developability pass rate\n'
                 '(GSPT1, PDB 5HXB; validity + physchem + SA + linker filters)')
    handles = [Patch(facecolor=PARA_COLOR[p], alpha=0.15, label=p) for p in PARA_COLOR]
    ax.legend(handles=handles, fontsize=9, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_developability.png")
    fig.savefig(f"{FIG}/fig1_developability.svg")
    plt.close(fig)
    print("fig1_developability.png + .svg regenerated (4 warhead generators)")

    # Print table
    print("\nTool | Generated | Developability-pass | Rate")
    for t in tools:
        print(f"{t:12s} | {int(master.loc[t,'n_generated']):6d} | {int(master.loc[t,'developability_pass']):6d} | {master.loc[t,'developability%']:.1f}%")

if __name__ == '__main__':
    main()
