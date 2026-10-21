#!/usr/bin/env python3
"""
Combine the 4 rebuttal Panel A Venns + venn_e3 into a single 4-in-a-row figure
with subtle (muted) colors. Output: figures/final_plot/

Panels (left → right):
  A1 Training vs External scaffolds
  A2 Training vs Temporal scaffolds
  A3 Training vs External targets
  A4 Training vs External E3 ligases
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures/final_plot"
os.makedirs(FIG, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11.5, 'figure.dpi': 200,
                     'savefig.bbox': 'tight'})

# Subtle (muted) color pairs
# [training, external] per panel — soft pastels with muted edges
PALETTES = {
    'A1':  ('#8fb3d9', '#e8b98a', '#5a7fa8', '#c98a4a'),  # soft blue / soft orange
    'A2':  ('#a9c6e3', '#b9d4b1', '#6d93bd', '#7aa871'),  # soft blue / soft green
    'A3':  ('#c3b8e0', '#e8b98a', '#8f7fb8', '#c98a4a'),  # soft purple / soft orange
    'A4':  ('#a9c6e3', '#e0b8c8', '#6d93bd', '#b87a94'),  # soft blue / soft rose
}

PANELS = [
    ("A1", "Scaffolds (external)", 4880, 1930, 70, "Training", "External\nscaffolds"),
    ("A2", "Scaffolds (temporal)",  4880, 103,  2,  "Training", "Temporal\nscaffolds"),
    ("A3", "Targets",               190,  53,  44, "Training", "External\ntargets"),
    ("A4", "E3 ligases",            20,   4,   3,  "Training", "External\nE3"),
]

def draw_venn(ax, ta, tb, both, c_train, c_ext, ec_train, ec_ext, lab_a, lab_b, title):
    # circles
    c1 = Circle((0.42, 0.52), 0.34, facecolor=c_train, edgecolor=ec_train, lw=1.5, alpha=0.85, zorder=2)
    c2 = Circle((0.68, 0.52), 0.34, facecolor=c_ext,  edgecolor=ec_ext,  lw=1.5, alpha=0.85, zorder=2)
    ax.add_patch(c1); ax.add_patch(c2)
    # numbers
    only_a = ta - both; only_b = tb - both
    ax.text(0.33, 0.52, f"{only_a:,}", ha='center', va='center', fontsize=17, weight='bold', color='#333', zorder=3)
    ax.text(0.55, 0.52, f"{both:,}",  ha='center', va='center', fontsize=15, weight='bold', color='#222', zorder=3)
    ax.text(0.77, 0.52, f"{only_b:,}", ha='center', va='center', fontsize=17, weight='bold', color='#333', zorder=3)
    # labels
    ax.text(0.42, 0.90, lab_a, ha='center', fontsize=10.5, weight='bold', color=ec_train)
    ax.text(0.42, 0.845, f"n = {ta:,}", ha='center', fontsize=9, color='#555')
    ax.text(0.68, 0.90, lab_b, ha='center', fontsize=10.5, weight='bold', color=ec_ext)
    ax.text(0.68, 0.845, f"n = {tb:,}", ha='center', fontsize=9, color='#555')
    ax.text(0.55, 0.13, f"shared = {both:,}", ha='center', fontsize=10, weight='bold', color='#444')
    ax.text(0.55, 0.065, title, ha='center', fontsize=11, weight='bold', color='#222')
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.05); ax.axis('off')

def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for ax, (key, title, ta, tb, both, la, lb) in zip(axes, PANELS):
        c_t, c_e, e_t, e_e = PALETTES[key]
        draw_venn(ax, ta, tb, both, c_t, c_e, e_t, e_e, la, lb, f"{key} · {title}")
    fig.suptitle("Novelty / Leakage Stress-Test — Full Curated Set (training vs external)",
                 fontsize=14, weight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG}/panelA_4in1_venn.png")
    fig.savefig(f"{FIG}/panelA_4in1_venn.svg")
    plt.close(fig)
    print(f"Saved: {FIG}/panelA_4in1_venn.png + .svg")

if __name__ == '__main__':
    main()
