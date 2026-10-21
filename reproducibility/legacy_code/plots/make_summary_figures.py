#!/usr/bin/env python3
"""
Publication-quality figures for Gcoupler vs. representative methods benchmark (GSPT1, 5HXB).
Tools: Gcoupler, Pharmacophore, DiffSBDD, Pocket2Mol, DiffLinker, DeLinker (ShapeLinker dropped).
Paradigm grouping: PROTAC-assembly (Gcoupler, Pharmacophore) vs pocket-guided (DiffSBDD, Pocket2Mol)
                  vs linker-generation (DiffLinker, DeLinker).
"""
import os, sys, json, warnings
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

plt.rcParams.update({
    'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol", "DiffLinker", "DeLinker"]
COL = {"Gcoupler": "#2c7fb8", "Pharmacophore": "#41ab5d", "DiffSBDD": "#d95f0e",
       "Pocket2Mol": "#d62728", "DiffLinker": "#756bb1", "DeLinker": "#636363"}
PARADIGM = {"Gcoupler": "PROTAC assembly", "Pharmacophore": "PROTAC assembly",
            "DiffSBDD": "Pocket-guided", "Pocket2Mol": "Pocket-guided",
            "DiffLinker": "Linker generation", "DeLinker": "Linker generation"}
PARA_COLOR = {"PROTAC assembly": "#2c7fb8", "Pocket-guided": "#d95f0e", "Linker generation": "#756bb1"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def add_para_background(ax, tools, alpha=0.06):
    """Shade paradigm groups."""
    seen = {}
    for i, t in enumerate(tools):
        p = PARADIGM[t]
        if p not in seen:
            seen[p] = i
    # group boundaries
    groups = {}
    for i, t in enumerate(tools):
        groups.setdefault(PARADIGM[t], []).append(i)
    for p, idxs in groups.items():
        ax.axvspan(min(idxs)-0.4, max(idxs)+0.4, color=PARA_COLOR[p], alpha=alpha, zorder=0)
    return groups

def main():
    master = pd.read_csv(f"{RES}/MASTER_COMPARISON.csv").set_index('tool')
    master = master.loc[[t for t in TOOLS if t in master.index]]
    dock = pd.read_csv(f"{RES}/level8_docking.csv").set_index('tool')
    dock2 = pd.read_csv(f"{RES}/level8b_docking_fixed.csv").set_index('tool')
    terr = pd.read_csv(f"{RES}/level9_ternary.csv").set_index('tool')
    pdp = pd.read_csv(f"{RES}/level7_pdp_potency.csv").set_index('tool')

    # ───────────────── FIG 1: Developability pass rate (waterfall-style) ─────────────────
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tools = [t for t in TOOLS if t in master.index]
    n_gen = [int(master.loc[t, 'n_generated']) for t in tools]
    n_dev = [int(master.loc[t, 'developability_pass']) for t in tools]
    pct = [master.loc[t, 'developability%'] for t in tools]

    x = np.arange(len(tools))
    ax.bar(x, n_gen, width=0.6, color='#d9d9d9', label='Generated', zorder=3)
    ax.bar(x, n_dev, width=0.6, color=[COL[t] for t in tools], label='Developability-passed', zorder=3)
    for i, (g, d, p) in enumerate(zip(n_gen, n_dev, pct)):
        ax.text(i, g + 40, f'{g:,}', ha='center', fontsize=9, color='#333333')
        if d > 0:
            ax.text(i, d/2, f'{d}\n({p:.1f}%)', ha='center', va='center', fontsize=9,
                    color='white', weight='bold', zorder=4)
        else:
            ax.text(i, 15, '0%', ha='center', fontsize=9, color='#d62728', weight='bold')
    groups = add_para_background(ax, tools)
    ax.set_xticks(x); ax.set_xticklabels(tools, fontsize=10.5)
    ax.set_ylabel('Number of candidates')
    ax.set_ylim(0, max(n_gen)*1.15)
    ax.set_title('Gcoupler vs. representative methods — generated vs. developability-passed\n(GSPT1, PDB 5HXB chain A; validity + physchem + SA + linker filters)')
    # legend for paradigm shading
    handles = [Patch(facecolor=PARA_COLOR[p], alpha=0.15, label=p) for p in PARA_COLOR]
    handles += [Patch(facecolor='#d9d9d9', label='Generated'), Patch(facecolor=COL['Gcoupler'], label='Developability-passed')]
    ax.legend(handles=handles, fontsize=8.5, loc='upper right')
    save(fig, "fig1_developability")

    # ───────────────── FIG 2: Docking affinity ─────────────────
    fig, ax = plt.subplots(figsize=(9, 5.2))
    dock_vals = {}
    for t in tools:
        v = None
        if t in dock.index and pd.notna(dock.loc[t, 'affinity_mean_kcal']):
            v = float(dock.loc[t, 'affinity_mean_kcal'])
        elif t in dock2.index and pd.notna(dock2.loc[t, 'affinity_mean_kcal']):
            v = float(dock2.loc[t, 'affinity_mean_kcal'])
        dock_vals[t] = v

    d_tools = [t for t in tools if dock_vals[t] is not None]
    vals = [dock_vals[t] for t in d_tools]
    bests = []
    for t in d_tools:
        b = None
        if t in dock.index and pd.notna(dock.loc[t, 'affinity_best_kcal']):
            b = float(dock.loc[t, 'affinity_best_kcal'])
        elif t in dock2.index and pd.notna(dock2.loc[t, 'affinity_best_kcal']):
            b = float(dock2.loc[t, 'affinity_best_kcal'])
        bests.append(b)

    x = np.arange(len(d_tools))
    bars = ax.bar(x, vals, width=0.55, color=[COL[t] for t in d_tools], alpha=0.9, zorder=3)
    ax.axhline(-6, color='red', ls='--', lw=1.2, zorder=2)
    ax.text(len(d_tools)-0.45, -5.85, 'binding threshold (−6 kcal/mol)', color='red', fontsize=8.5, ha='right')
    for i, (v, b) in enumerate(zip(vals, bests)):
        ax.text(i, v - 0.35, f'{v:.1f}', ha='center', va='top', fontsize=10, weight='bold', color='white')
        if b is not None and abs(b - v) > 0.2:
            ax.text(i, v + 0.25, f'best {b:.1f}', ha='center', fontsize=8, color='#555555')
    ax.set_xticks(x); ax.set_xticklabels(d_tools, fontsize=10.5)
    ax.set_ylabel('Mean smina affinity (kcal/mol)')
    ax.set_ylim(min(vals) - 1.5, 0.5)
    ax.set_title('Docking compatibility — smina vs. GSPT1 pocket (5HXB chain A)')
    add_para_background(ax, d_tools)
    save(fig, "fig2_docking")

    # ───────────────── FIG 3: Multi-panel evaluation radar/bar summary ─────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # 3a. Validity / uniqueness / novelty (near-100% bars)
    ax = axes[0]
    metrics = ['validity%', 'novelty%']
    x = np.arange(len(tools))
    width = 0.35
    for j, m in enumerate(metrics):
        vals = [master.loc[t, m] if t in master.index else 0 for t in tools]
        ax.bar(x + (j-0.5)*width, vals, width, label=m.replace('%',''), color=['#41ab5d', '#2c7fb8'][j], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tools, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel('%')
    ax.set_title('(a) Validity & novelty')
    ax.legend(fontsize=8)

    # 3b. SA pass rate
    ax = axes[1]
    sa_vals = [master.loc[t, 'SA_pass%'] if t in master.index else 0 for t in tools]
    bars = ax.bar(tools, sa_vals, color=[COL[t] for t in tools], alpha=0.9)
    for b, v in zip(bars, sa_vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=25)
    ax.set_ylim(0, 110); ax.set_ylabel('SA ≤ 5.0 (%)')
    ax.set_title('(b) Synthetic accessibility')

    # 3c. Ternary linker-span plausibility (PROTAC-assembly tools)
    ax = axes[2]
    t_tools = [t for t in TOOLS if t in terr.index]
    if t_tools:
        vals = [terr.loc[t, 'span_pct'] for t in t_tools]
        bars = ax.bar(t_tools, vals, color=[COL[t] for t in t_tools], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        ax.set_ylim(0, 110)
    ax.set_ylabel('Linker-span plausible (%)')
    ax.set_title('(c) Ternary-complex plausibility (linker span)')

    # 3d. PDP predicted potency
    ax = axes[3]
    p_tools = [t for t in TOOLS if t in pdp.index and pd.notna(pdp.loc[t, 'mean_p_active'])]
    if p_tools:
        vals = [pdp.loc[t, 'mean_p_active'] for t in p_tools]
        bars = ax.bar(p_tools, vals, color=[COL[t] for t in p_tools], alpha=0.9)
        ax.axhline(0.5, color='red', ls='--', lw=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}', ha='center', fontsize=9)
        ax.set_ylim(0, 0.7)
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylabel('Mean PDP P(active)')
    ax.set_title('(d) Predicted degradation potency (PDP, GSPT1/CRBN)')

    fig.suptitle('Gcoupler benchmark evaluation (GSPT1, PDB 5HXB) — multi-metric summary', fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, "fig3_multimetric_summary")

    # ───────────────── FIG 4: Cascade funnel (Gcoupler vs others) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # For each tool: cumulative pass through validity → developability
    stages = ['Valid', 'Physchem', 'SA', 'Developability']
    funnel = {}
    for t in tools:
        v = master.loc[t]
        funnel[t] = [int(v['n_valid']), int(v['n_valid']*v['physchem_pass%']/100),
                     int(v['n_valid']*v['SA_pass%']/100), int(v['developability_pass'])]
    for t in tools:
        vals = funnel[t]
        ax.plot(stages, vals, marker='o', linewidth=2.2, color=COL[t], label=f"{t} ({vals[-1]})")
        ax.fill_between(stages, vals, color=COL[t], alpha=0.05)
    ax.set_yscale('log')
    ax.set_ylabel('Candidates (log scale)')
    ax.set_title('Filter-cascade funnel per tool (GSPT1)')
    ax.legend(fontsize=8.5)
    save(fig, "fig4_cascade_funnel")

    print(f"\nFigures saved to {FIG}")
    for f in sorted(os.listdir(FIG)):
        if f.endswith('.png'):
            print(f"  {f}")

if __name__ == '__main__':
    main()
