#!/usr/bin/env python3
"""
Clinical PROTAC scatter (light colors) — predicted vs reported DC50 (log-log),
with identity + 3×/10× bands, light pastel palette.
Output: figures/final_plot/clinical_scatter_light.svg/png
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures/final_plot"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11.5, 'axes.titlesize': 14, 'axes.labelsize': 12.5,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linestyle': '--'})

# Light pastel palette per compound
COLORS = {
    "ARV-110":  ("#a8cbe8", "#5a8db8"),  # light blue
    "ARV-471":  ("#f2c9a0", "#d19a55"),  # light orange
    "NX-2127":  ("#b9d9b0", "#6fa868"),  # light green
    "KT-474":   ("#c3b8e0", "#8f7fb8"),  # light purple
    "CFT8634":  ("#e0b8c8", "#b87a94"),  # light rose
    "ZLC491":   ("#bcd4d0", "#6f9a94"),  # light teal
}

data = [
    ("ARV-110",  "AR",        6.1, 1.0),
    ("ARV-471",  "ER\u03b1",   15.1, 1.0),
    ("NX-2127",  "BTK",       28.0, 4.0),
    ("KT-474",   "IRAK4",     13.1, 2.1),
    ("CFT8634",  "BRD9",      11.2, 2.7),
    ("ZLC491",   "CDK12/13",   0.9, 28.0),
]

def main():
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # light shaded bands behind
    ax.axhspan(0.1, 10.0, color='#fdf6e3', alpha=0.5, zorder=0, label='within 10\u00d7')
    ax.axhspan(0.3, 3.0,  color='#eef7ee', alpha=0.6, zorder=0, label='within 3\u00d7')

    # identity + band lines (light)
    lims = [0.3, 100]
    ax.plot(lims, lims, color='#999999', ls='--', lw=1.5, zorder=1, label='y = x')
    ax.plot(lims, [3*x for x in lims],  color='#cccccc', ls=':', lw=1.2, zorder=1)
    ax.plot(lims, [x/3 for x in lims],  color='#cccccc', ls=':', lw=1.2, zorder=1)
    ax.plot(lims, [10*x for x in lims], color='#dddddd', ls=':', lw=1, zorder=1)
    ax.plot(lims, [x/10 for x in lims], color='#dddddd', ls=':', lw=1, zorder=1)

    # points (light fill, muted edge)
    for name, target, pred, rep in data:
        fill, edge = COLORS[name]
        ax.scatter(rep, pred, s=300, color=fill, edgecolor=edge, linewidth=1.8,
                   alpha=0.95, zorder=3)
        ax.annotate(f"{name}\n({target})", (rep, pred), textcoords='offset points',
                    xytext=(14, 10), fontsize=11, weight='bold', color='#333333', zorder=4)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.3, 120); ax.set_ylim(0.3, 120)
    ax.set_xlabel('Reported DC50 (nM) \u2014 literature / clinical')
    ax.set_ylabel('SynGlue predicted DC50 (nM)')
    ax.set_title('Clinical PROTACs \u2014 SynGlue predicted vs reported DC50\n(single-digit to low-tens nM)', fontsize=13)

    # light legend
    handles = [plt.Line2D([0],[0], color='#999999', ls='--', lw=1.5, label='y = x'),
               plt.Line2D([0],[0], color='#cccccc', ls=':', lw=1.2, label='\u00b13\u00d7'),
               plt.Line2D([0],[0], color='#dddddd', ls=':', lw=1, label='\u00b110\u00d7')]
    ax.legend(handles=handles, fontsize=10, loc='upper left', framealpha=0.9, edgecolor='#dddddd')

    fig.tight_layout()
    fig.savefig(f"{FIG}/clinical_scatter_light.png")
    fig.savefig(f"{FIG}/clinical_scatter_light.svg")
    plt.close(fig)
    print("Saved: clinical_scatter_light.png + .svg")

    # small annotation of fold errors on the side
    print("\nFold ratios (predicted/reported):")
    for name, target, pred, rep in data:
        print(f"  {name:9s} {pred:5.1f}/{rep:5.1f} = {pred/rep:5.2f}\u00d7")

if __name__ == '__main__':
    main()
