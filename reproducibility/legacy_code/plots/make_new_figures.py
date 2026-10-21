#!/usr/bin/env python3
"""
New figures for Gcoupler benchmark (GSPT1, 5HXB):
  fig1_new: developability PASS-RATE (normalized %) with paradigm shading
  fig5_warhead_similarity: Tanimoto vs Gcoupler-selected warhead (bar + distribution)
  fig6_warhead_similarity_dist: box/violin of per-candidate similarity
  fig7_generation_similarity_scatter: similarity vs docking (2D view)
  fig8_tool_summary_radar: normalized multi-metric radar
"""
import os, sys, json, warnings, glob
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
from math import pi

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol", "DiffLinker", "DeLinker"]
COL = {"Gcoupler": "#2c7fb8", "Pharmacophore": "#41ab5d", "DiffSBDD": "#d95f0e",
       "Pocket2Mol": "#d62728", "DiffLinker": "#756bb1", "DeLinker": "#636363"}
PARADIGM = {"Gcoupler": "PROTAC assembly (fixed warhead)", "Pharmacophore": "PROTAC assembly (fixed warhead)",
            "DiffSBDD": "Pocket-guided de novo", "Pocket2Mol": "Pocket-guided de novo",
            "DiffLinker": "Linker generation", "DeLinker": "Linker generation"}
PARA_COLOR = {"PROTAC assembly (fixed warhead)": "#2c7fb8", "Pocket-guided de novo": "#d95f0e",
              "Linker generation": "#756bb1"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    master = pd.read_csv(f"{RES}/MASTER_COMPARISON.csv").set_index('tool')
    sim = pd.read_csv(f"{RES}/warhead_similarity_vs_gcoupler.csv").set_index('tool')
    dock = pd.read_csv(f"{RES}/level8_docking.csv").set_index('tool')
    dock2 = pd.read_csv(f"{RES}/level8b_docking_fixed.csv").set_index('tool')
    tools = [t for t in TOOLS if t in master.index]

    # ════ FIG 1 (NEW): Developability pass-rate with paradigm shading ════
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    vals = [float(master.loc[t, 'developability%']) for t in tools]
    bars = ax.bar(tools, vals, width=0.6, color=[COL[t] for t in tools], alpha=0.92, zorder=3,
                  edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10, weight='bold')
    # paradigm shading
    groups = {}
    for i, t in enumerate(tools):
        groups.setdefault(PARADIGM[t], []).append(i)
    for p, idxs in groups.items():
        ax.axvspan(min(idxs)-0.45, max(idxs)+0.45, color=PARA_COLOR[p], alpha=0.07, zorder=0)
    ax.set_ylabel('Developability pass rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Downstream developability pass rate by method\n(GSPT1, PDB 5HXB; validity + physchem + SA + linker filters)')
    handles = [Patch(facecolor=PARA_COLOR[p], alpha=0.12, label=p) for p in PARA_COLOR]
    ax.legend(handles=handles, fontsize=8.5, loc='upper right', framealpha=0.9)
    save(fig, "fig1_developability")

    # ════ FIG 5: Warhead similarity vs Gcoupler-selected warhead ════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    s_tools = [t for t in tools if t in sim.index]
    means = [sim.loc[t, 'mean_tanimoto'] for t in s_tools]
    meds = [sim.loc[t, 'median_tanimoto'] for t in s_tools]
    x = np.arange(len(s_tools))
    ax.bar(x - 0.2, means, 0.4, color=[COL[t] for t in s_tools], alpha=0.9, label='Mean Tanimoto')
    ax.bar(x + 0.2, meds, 0.4, color=[COL[t] for t in s_tools], alpha=0.45, label='Median Tanimoto')
    for i, (m, d) in enumerate(zip(means, meds)):
        ax.text(i-0.2, m+0.02, f'{m:.2f}', ha='center', fontsize=9)
        ax.text(i+0.2, d+0.02, f'{d:.2f}', ha='center', fontsize=8, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(s_tools, rotation=20, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Morgan (r=2, 2048-bit) Tanimoto to Gcoupler warhead')
    ax.set_title('Warhead similarity to Gcoupler-selected GSPT1 warhead\n(Nc1nnc(C#CC2CC2)s1, thiadiazole)')
    ax.legend(fontsize=8)
    # annotation for fixed vs de novo
    ax.text(0.5, 1.05, 'fixed warhead (design)', fontsize=9, color='#2c7fb8', ha='center', transform=ax.get_xaxis_transform())

    ax = axes[1]
    ax.axis('off')
    ax.text(0.02, 0.95,
            "Interpretation:\n\n"
            "• Gcoupler & Pharmacophore: Tanimoto = 1.0 —\n"
            "  they ASSEMBLE linkers onto the same selected\n"
            "  GSPT1 warhead (fixed by design).\n\n"
            "• DiffSBDD / Pocket2Mol: mean ~0.05 — they\n"
            "  generate entirely NEW chemotypes with no\n"
            "  thiadiazole resemblance.\n\n"
            "• DeLinker / DiffLinker: low (0.06–0.07) —\n"
            "  they output linkers, not warheads.\n\n"
            "⇒ 'Similarity' reflects design paradigm, not\n"
            "  quality: Gcoupler fixes a validated warhead\n"
            "  (fragment compatibility), de novo tools explore\n"
            "  different warhead space.",
            fontsize=10, va='top', family='monospace')
    fig.suptitle('How similar are generated candidates to the Gcoupler-selected GSPT1 warhead?', fontsize=13)
    fig.tight_layout()
    save(fig, "fig5_warhead_similarity")

    # ════ FIG 6: per-candidate similarity distribution (violin/box) ════
    # recompute per-candidate sims
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    GCR2 = f"{REV}/experiments/gcoupler_benchmark/results"
    SELECTED = "Nc1nnc(C#CC2CC2)s1"
    wh_mol = Chem.MolFromSmiles(SELECTED)
    f_wh = AllChem.GetMorganFingerprintAsBitVect(wh_mol, 2, 2048)

    all_sims = {}
    for tool, f in [("Gcoupler", f"{GCR2}/gcoupler/filtered_results.csv"),
                    ("Pharmacophore", f"{GCR2}/pharmacophore/generated_candidates.csv"),
                    ("DiffSBDD", f"{GCR2}/diffsbdd/samples.sdf"),
                    ("Pocket2Mol", glob.glob(f"{GCR2}/pocket2mol/*/SMILES.txt")[0]),
                    ("DeLinker", f"{GCR2}/warheads/summary/warhead_delinker.csv"),
                    ("DiffLinker", f"{GCR2}/warheads/summary/warhead_difflinker.csv")]:
        if tool == 'Gcoupler':
            smis = pd.read_csv(f)['canonical_smiles'].tolist()
        elif tool == 'Pharmacophore':
            smis = pd.read_csv(f)['SMILES'].tolist()
        elif tool == 'DiffSBDD':
            smis = [Chem.MolToSmiles(m) for m in Chem.SDMolSupplier(f) if m is not None]
        elif tool == 'Pocket2Mol':
            smis = pd.read_csv(f, header=None)[0].tolist()
        elif tool == 'DeLinker':
            smis = pd.read_csv(f)['canonical_smiles'].tolist()
        else:
            smis = pd.read_csv(f)['canonical_smiles'].tolist()
        sims = []
        for s in smis[:300]:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            if m.HasSubstructMatch(wh_mol):
                sims.append(1.0)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                sims.append(DataStructs.TanimotoSimilarity(f_wh, fp))
        all_sims[tool] = sims

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_tools = [t for t in TOOLS if t in all_sims and len(all_sims[t]) > 0]
    data = [all_sims[t] for t in plot_tools]
    bp = ax.boxplot(data, tick_labels=plot_tools, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', lw=1.5))
    for patch, t in zip(bp['boxes'], plot_tools):
        patch.set_facecolor(COL[t]); patch.set_alpha(0.75)
    ax.axhline(0.3, color='red', ls='--', lw=1, label='0.3 similarity threshold')
    ax.set_ylabel('Per-candidate Tanimoto to Gcoupler warhead')
    ax.set_title('Distribution of warhead similarity (each point = one generated candidate)')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    save(fig, "fig6_warhead_similarity_distribution")

    # ════ FIG 7: similarity vs docking (2D scatter) ════
    # dock affinity (mean) per tool vs mean similarity
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in tools:
        if t not in sim.index: continue
        sm = sim.loc[t, 'mean_tanimoto']
        dv = None
        if t in dock.index and pd.notna(dock.loc[t, 'affinity_mean_kcal']):
            dv = float(dock.loc[t, 'affinity_mean_kcal'])
        elif t in dock2.index and pd.notna(dock2.loc[t, 'affinity_mean_kcal']):
            dv = float(dock2.loc[t, 'affinity_mean_kcal'])
        if dv is None: continue
        ax.scatter(sm, dv, s=160, color=COL[t], edgecolor='black', linewidth=0.8, zorder=3, label=f"{t} ({dv:.1f} kcal/mol)")
        ax.annotate(t, (sm, dv), textcoords='offset points', xytext=(6, -14), fontsize=9)
    ax.axhline(-6, color='red', ls='--', lw=1)
    ax.set_xlabel('Mean warhead similarity (Tanimoto vs Gcoupler warhead)')
    ax.set_ylabel('Mean smina affinity (kcal/mol)')
    ax.set_title('Design-space view: warhead similarity vs. GSPT1 docking')
    ax.legend(fontsize=8, loc='lower left')
    save(fig, "fig7_similarity_vs_docking")

    # ════ FIG 8: Radar chart (normalized metrics) ════
    metrics = ['validity%', 'novelty%', 'SA_pass%', 'developability%']
    labels = ['Validity', 'Novelty', 'SA pass', 'Developability']
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(projection='polar'))
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    for t in tools:
        vals = [float(master.loc[t, m]) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COL[t], label=t)
        ax.fill(angles, vals, color=COL[t], alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Method profile — normalized evaluation metrics (GSPT1)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save(fig, "fig8_radar")

    print(f"\nNew figures saved to {FIG}")

if __name__ == '__main__':
    main()
