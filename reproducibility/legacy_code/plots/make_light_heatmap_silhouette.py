#!/usr/bin/env python3
"""
Re-produce tool chemical-space separation + per-tool silhouette with requested styling:
  (1) LIGHT-colored pairwise separation heatmap (mean Jaccard distance).
  (2) Better non-heatmap silhouette plot (diverging horizontal bar, sorted, error bars,
      global reference line).
Saves ONLY to tool_analysis/ . Also writes sep_matrix.csv for reproducibility.
"""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
RDLogger.DisableLog('rdApp.*')

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = f"{REV}/figures/final_plot/tool_analysis"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol", "DiffLinker", "DeLinker"]
COLORS = {"Gcoupler": "#7fb3d5", "Pharmacophore": "#f0b27a", "DiffSBDD": "#82cba4",
          "Pocket2Mol": "#af9cd6", "DiffLinker": "#d98ba6", "DeLinker": "#7fbdb6"}

def load_tool_smiles():
    GCR = f"{REV}/experiments/gcoupler_benchmark/results"
    data = {}
    g = pd.read_csv(f"{GCR}/gcoupler/filtered_results.csv")
    data['Gcoupler'] = g['canonical_smiles'].dropna().tolist()
    ph = pd.read_csv(f"{GCR}/pharmacophore/generated_candidates.csv")
    data['Pharmacophore'] = ph['SMILES'].dropna().tolist()
    mols = [m for m in Chem.SDMolSupplier(f"{GCR}/diffsbdd/samples.sdf") if m is not None]
    data['DiffSBDD'] = [Chem.MolToSmiles(m) for m in mols]
    p2m = pd.read_csv(glob.glob(f"{GCR}/pocket2mol/*/SMILES.txt")[0], header=None)[0].dropna().tolist()
    data['Pocket2Mol'] = p2m
    dl = pd.read_csv(f"{GCR}/warheads/summary/warhead_delinker.csv")
    data['DeLinker'] = dl['canonical_smiles'].dropna().tolist()
    dlk = pd.read_csv(f"{GCR}/warheads/summary/warhead_difflinker.csv")
    data['DiffLinker'] = dlk['canonical_smiles'].dropna().tolist()
    clean = {}
    for t, smis in data.items():
        canons = set()
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                canons.add(Chem.MolToSmiles(m, canonical=True))
        clean[t] = sorted(canons)
    return clean

def fp_matrix(smiles_list):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles_list]
    X = np.zeros((len(fps), 2048), dtype=np.int8)
    for i, f in enumerate(fps):
        for idx in f.GetOnBits():
            X[i, idx] = 1
    return X

smiles = load_tool_smiles()
fps = {t: fp_matrix(smiles[t]) for t in TOOLS}
for t in TOOLS:
    print(f"  {t}: {fps[t].shape}", flush=True)

all_smis = [s for t in TOOLS for s in smiles[t]]
all_tool = [t for t in TOOLS for _ in smiles[t]]
X_all = np.vstack([fps[t] for t in TOOLS])
n_total = len(X_all)
step = max(1, n_total // 3000)
idx_use = np.arange(0, n_total, step)
X_sub = X_all[idx_use]
tool_sub = [all_tool[i] for i in idx_use]
D_jac = pairwise_distances(X_sub, metric='jaccard')
lab_sub = np.array([TOOLS.index(t) for t in tool_sub])
print(f"subsample: {len(idx_use)} molecules", flush=True)

# ---- silhouette ----
sil_orig = silhouette_score(D_jac, lab_sub, metric='precomputed')
s_vals = silhouette_samples(D_jac, lab_sub, metric='precomputed')
per_tool = {}
for t in TOOLS:
    m = lab_sub == TOOLS.index(t)
    per_tool[t] = dict(mean=float(s_vals[m].mean()), median=float(np.median(s_vals[m])),
                       std=float(s_vals[m].std()), n=int(m.sum()), vals=s_vals[m])

# ---- 1) LIGHT separation heatmap ----
n_t = len(TOOLS)
sep = np.full((n_t, n_t), np.nan)
for i in range(n_t):
    for j in range(n_t):
        if i == j: continue
        a_i = np.where(np.array(tool_sub) == TOOLS[i])[0]
        a_j = np.where(np.array(tool_sub) == TOOLS[j])[0]
        if len(a_i) == 0 or len(a_j) == 0:
            continue
        d_ij = D_jac[np.ix_(a_i, a_j)]
        sep[i, j] = float(d_ij.mean())
sep_df = pd.DataFrame(sep, index=TOOLS, columns=TOOLS)
sep_df.to_csv(f"{OUT}/sep_matrix.csv")
print("Separation matrix:\n", sep_df.round(3).to_string(), flush=True)

fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(sep, cmap='YlGnBu', vmin=sep[~np.isnan(sep)].min(), vmax=sep[~np.isnan(sep)].max())
ax.set_xticks(range(n_t)); ax.set_xticklabels(TOOLS, rotation=45, ha='right')
ax.set_yticks(range(n_t)); ax.set_yticklabels(TOOLS)
for i in range(n_t):
    for j in range(n_t):
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='#666'); continue
        if not np.isnan(sep[i, j]):
            ax.text(j, i, f'{sep[i,j]:.2f}', ha='center', va='center', fontsize=9, color='black')
ax.set_title(f'Mean pairwise Jaccard distance (chemical-space separation)\n'
             f'global silhouette = {sil_orig:.3f}')
fig.colorbar(im, ax=ax, label='Mean Jaccard distance (0=identical, 1=distinct)')
fig.tight_layout()
fig.savefig(f"{OUT}/heatmap_pairwise_separation_light.svg")
fig.savefig(f"{OUT}/heatmap_pairwise_separation_light.png")
plt.close(fig)
print("Saved heatmap_pairwise_separation_light", flush=True)

# ---- 2) BETTER silhouette plot (non-heatmap diverging bars) ----
order = sorted(TOOLS, key=lambda t: per_tool[t]['mean'])
vals = [per_tool[t]['mean'] for t in order]
stds = [per_tool[t]['std'] for t in order]
ns = [per_tool[t]['n'] for t in order]
cmaps_c = [COLORS[t] for t in order]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
y = np.arange(len(order))
# mask with light gray where negative
colors_bar = ['#e07b6a' if v < 0 else '#6fae80' for v in vals]
for yi, (v, s, c) in enumerate(zip(vals, stds, cmaps_c)):
    ax.barh(yi, v, height=0.6, color=c, edgecolor='#4a4a4a', lw=0.8, alpha=0.95)
    ax.errorbar(v, yi, xerr=s, fmt='none', ecolor='#333', elinewidth=1.0, capsize=3)
    ax.text(v if v >= 0 else v - s - 0.03, yi, f'{v:.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9, color='black')
ax.axvline(sil_orig, color='#d62728', ls='--', lw=1.6, label=f'global = {sil_orig:.3f}')
ax.axvline(0, color='#999', lw=1.0)
ylab = [f'{t}  (n={per_tool[t][chr(110)]})' for t in order]
ax.set_yticks(y); ax.set_yticklabels(ylab)
ax.set_xlabel('Mean silhouette coefficient (original fingerprint space)')
ax.set_title('Per-tool silhouette — molecules cluster by generating tool')
ax.set_xlim(min(vals) - 0.3, max(vals) + 0.35)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.25, ls='--')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/silhouette_score_bars.svg")
fig.savefig(f"{OUT}/silhouette_score_bars.png")
plt.close(fig)
print("Saved silhouette_score_bars", flush=True)

pd.DataFrame([{"Tool": t, "mean_sil": per_tool[t]['mean'], "median": per_tool[t]['median'],
               "std": per_tool[t]['std'], "n": per_tool[t]['n']} for t in TOOLS]).to_csv(
    f"{OUT}/silhouette_bars_data.csv", index=False)
print("\nAll done. Files written to", OUT)
