#!/usr/bin/env python3
"""
Rebuild interpretability figures with LIGHT colors + consistent W/L/E scheme:
  B1 attention donut+bar   → light colors → final_plot/
  B2 ablation              → VERTICAL bar, same colors
  B3 attention heatmap     → VERTICAL layout, same colors
Consistent scheme:
  Warhead  = light blue   #a8cbe8
  Linker   = light orange #f2c9a0
  E3 ligase= light green  #b9d9b0
Output: figures/final_plot/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/final_plot"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linestyle': '--'})

LABELS = ["Warhead", "Linker", "E3 ligase"]
COLORS = {"Warhead": "#a8cbe8", "Linker": "#f2c9a0", "E3 ligase": "#b9d9b0"}   # light pastels
EDGES  = {"Warhead": "#5a8db8", "Linker": "#d19a55", "E3 ligase": "#6fa868"}   # muted edges

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    interp = json.load(open(f"{OUT}/interpretability.json"))
    attn = np.load(f"{OUT}/attention_weights.npy").squeeze(-1)  # (n,3)
    a = interp['attention']; ab = interp['ablation']
    vals = [a['W'], a['L'], a['E']]
    stds = [attn[:,0].std(), attn[:,1].std(), attn[:,2].std()]
    abl  = [ab['Warhead'], ab['Linker'], ab['E3']]

    # ═══ B1: attention — donut + bar, LIGHT colors ═══
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(vals, labels=LABELS, autopct='%.1f%%', startangle=90,
                       colors=[COLORS[l] for l in LABELS],
                       wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for w, l in zip(wedges, LABELS):
        w.set_edgecolor(EDGES[l]); w.set_linewidth(1.5)
    ax.set_title('Attention allocation')
    ax = axes[1]
    bars = ax.bar(LABELS, vals, yerr=stds, capsize=6,
                  color=[COLORS[l] for l in LABELS],
                  edgecolor=[EDGES[l] for l in LABELS], linewidth=1.2, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=11, weight='bold')
    ax.set_ylabel('Mean attention weight'); ax.set_ylim(0, 1.15)
    ax.set_title('Component attention (mean ± std)')
    fig.suptitle('Interpretability — component attention (frozen model)', fontsize=13)
    fig.tight_layout()
    save(fig, "B1_attention_donut_bar_light")

    # ═══ B2: ablation — VERTICAL bars (horizontal layout = labels on y), same colors ═══
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(LABELS, abl,
                  color=[COLORS[l] for l in LABELS],
                  edgecolor=[EDGES[l] for l in LABELS], linewidth=1.2, alpha=0.9)
    for b, v in zip(bars, abl):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=11, weight='bold')
    ax.set_xlabel('Component (zeroed token)')
    ax.set_ylabel('Mean |Δ log10(DC50)|')
    ax.set_title('Component ablation — prediction sensitivity (frozen model)')
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    save(fig, "B2_ablation_vertical")

    # ═══ B3: attention heatmap — VERTICAL (components on x are too wide; use transposed heatmap) ═══
    fig, ax = plt.subplots(figsize=(7, 7))
    n = min(300, attn.shape[0])
    order = np.argsort(attn[:n, 1])
    data = attn[:n][order].T  # (3, n)
    im = ax.imshow(data, aspect='auto', cmap='YlGnBu')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(LABELS)
    # color the y-tick labels per component scheme
    for t, l in zip(ax.get_yticklabels(), LABELS):
        t.set_color(EDGES[l]); t.set_fontweight('bold')
    ax.set_xlabel('Sample (sorted by linker attention)')
    ax.set_ylabel('Component')
    ax.set_title(f'Attention heatmap — {n} samples\n(linker-sorted, frozen model)')
    cbar = fig.colorbar(im, ax=ax, label='Attention weight', shrink=0.85)
    fig.tight_layout()
    save(fig, "B3_attention_heatmap_vertical")

    print(f"\nAll light-color interpretability figures saved to {FIG}")

if __name__ == '__main__':
    main()
