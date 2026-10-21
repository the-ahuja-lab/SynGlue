#!/usr/bin/env python3
"""
Frozen Model Analysis + Generalization Figures (v3, consistent encoding)

Figures:
  fig1: Attention weights per component (all splits)
  fig2: Component ablation sensitivity (frozen random-split model)
  fig3: External DC50 correlation scatter (pred vs reported, 4 splits)
  fig4: External Dmax correlation scatter
  fig5: Generalization bar — internal R2 vs external r per split
  fig6: Attention heatmap W→L→E transitions (frozen model, sample)
Output: revision/figures/generalization/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = f"{REV}/benchmarking/generalization_data_v3"
MODEL_DIR = f"{REV}/experiments/gen_training_v3"
FIG_DIR = f"{REV}/figures/generalization"
os.makedirs(FIG_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

class CAT(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, n_heads=8, n_layers=2, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(3, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, x, return_attn=False):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * attn).sum(dim=1)
        y = self.head(fused).squeeze(-1)
        return (y, attn.squeeze(-1)) if return_attn else y
    def embed(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn).sum(dim=1)

SPLITS = ["random", "target_held_out", "e3_held_out", "scaffold_held_out"]
SPLIT_LABELS = {"random": "Random", "target_held_out": "Target-held", "e3_held_out": "E3-held", "scaffold_held_out": "Scaffold-held"}
C = {'W': '#1f77b4', 'L': '#ff7f0e', 'E': '#2ca02c'}

def main():
    # Load external predictions + stats
    ext = pd.read_csv(f"{MODEL_DIR}/external_predictions_all_splits.csv")
    stats = pd.read_csv(f"{MODEL_DIR}/external_validation_stats.csv")
    summ = pd.read_csv(f"{MODEL_DIR}/generalization_summary_v2.csv")

    # Reported values for scatter
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    from rdkit import Chem
    def canon(s):
        if not isinstance(s, str) or not s: return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s
    rep['canon'] = rep['canonical_SMILES'].apply(canon)
    dc50_rep = rep[rep['Value_Type']=='DC50'][['canon','Value','Value_Operator']].copy()
    dc50_rep['val'] = pd.to_numeric(dc50_rep['Value'], errors='coerce')
    dc50_rep = dc50_rep[dc50_rep['val'].notna() & dc50_rep['Value_Operator'].isna() & (dc50_rep['val']>0)]
    dc50_rep = dc50_rep.drop_duplicates(subset=['canon'])
    dmax_rep = rep[rep['Value_Type']=='Dmax'][['canon','Value','Value_Operator']].copy()
    dmax_rep['val'] = pd.to_numeric(dmax_rep['Value'], errors='coerce')
    dmax_rep = dmax_rep[dmax_rep['val'].notna() & dmax_rep['Value_Operator'].isna()]
    dmax_rep = dmax_rep.drop_duplicates(subset=['canon'])

    # ================= FIG 1: Attention ================
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SPLITS)); w = 0.25
    for j, comp in enumerate(['W','L','E']):
        vals = [summ.loc[(summ['task']=='dc50') & (summ['split']==s), f'attn_{comp}'].iloc[0] for s in SPLITS]
        ax.bar(x + (j-1)*w, vals, w, label=comp, color=C[comp], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Component attention (frozen models, DC50 task)')
    ax.legend(title='Component')
    ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig1_attention.png"); plt.close(fig)
    print("fig1_attention.png")

    # ================= FIG 2: Ablation (frozen random model) ================
    X_dc = np.load(f"{DATA_DIR}/X_dc50_pca.npy")
    y_dc = np.load(f"{DATA_DIR}/y_dc50.npy")
    sc = (float(y_dc.mean()), float(y_dc.std()))
    model = CAT().to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/model_dc50_random.pt", map_location=DEVICE))
    model.eval()
    Xt = torch.tensor(X_dc, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        base, attn_all = model(Xt, return_attn=True)
        base = base.cpu().numpy(); attn_all = attn_all.cpu().numpy()
        abl = {}
        for comp, idx in [("Warhead", 0), ("Linker", 1), ("E3 ligand", 2)]:
            Xa = Xt.clone(); Xa[:, idx, :] = 0
            ya = model(Xa).cpu().numpy()
            abl[comp] = {"delta_z": float(np.mean(np.abs(ya - base))),
                         "rmse_z": float(np.sqrt(np.mean((ya - base)**2)))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comps = list(abl.keys())
    axes[0].bar(comps, [abl[c]['delta_z'] for c in comps], color=[C['W'], C['L'], C['E']], alpha=0.85)
    axes[0].set_ylabel('Mean |Δprediction| (z-space)')
    axes[0].set_title('Component ablation sensitivity (frozen random model, DC50)')
    axes[1].bar(comps, [abl[c]['rmse_z'] for c in comps], color=[C['W'], C['L'], C['E']], alpha=0.85)
    axes[1].set_ylabel('RMSE of Δprediction (z-space)')
    axes[1].set_title('Ablation RMSE')
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig2_ablation.png"); plt.close(fig)
    print("fig2_ablation.png")

    # ================= FIG 3: External DC50 scatter ================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, split in enumerate(SPLITS):
        ax = axes[i//2][i%2]
        pred_col = f'pred_dc50_{split}'
        d = ext[['canon', pred_col]].merge(dc50_rep, on='canon', how='inner')
        d = d[d[pred_col] > 0]
        d['logp'] = np.log10(d[pred_col]); d['logt'] = np.log10(d['val'])
        r = pearsonr(d['logt'], d['logp'])[0]
        rho = spearmanr(d['logt'], d['logp'])[0]
        ax.scatter(d['logt'], d['logp'], s=6, alpha=0.35, color='#2ca02c')
        ax.plot([0, 5], [0, 5], 'r--', lw=1, alpha=0.5)
        ax.set_xlabel('Reported log10(DC50 / nM)')
        ax.set_ylabel('Predicted log10(DC50 / nM)')
        ax.set_title(f"{SPLIT_LABELS[split]} model  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})")
    fig.suptitle('External DC50: predicted vs reported (consistent encoding, frozen models)', fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig3_external_dc50_scatter.png"); plt.close(fig)
    print("fig3_external_dc50_scatter.png")

    # ================= FIG 4: External Dmax scatter ================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, split in enumerate(SPLITS):
        ax = axes[i//2][i%2]
        pred_col = f'pred_dmax_{split}'
        d = ext[['canon', pred_col]].merge(dmax_rep, on='canon', how='inner')
        r = pearsonr(d['val'], d[pred_col])[0]
        rho = spearmanr(d['val'], d[pred_col])[0]
        ax.scatter(d['val'], d[pred_col], s=6, alpha=0.35, color='#ff7f0e')
        ax.set_xlabel('Reported Dmax (%)')
        ax.set_ylabel('Predicted Dmax (%)')
        ax.set_title(f"{SPLIT_LABELS[split]} model  (r={r:.3f}, ρ={rho:.3f}, n={len(d)})")
    fig.suptitle('External Dmax: predicted vs reported (frozen models)', fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig4_external_dmax_scatter.png"); plt.close(fig)
    print("fig4_external_dmax_scatter.png")

    # ================= FIG 5: Internal vs external ================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, task in enumerate(['dc50', 'dmax']):
        ax = axes[j]
        sub = summ[summ['task'] == task]
        r2_vals = sub.set_index('split')['R2'].reindex(SPLITS)
        ext_stats = stats[stats['task'] == task].set_index('split')['pearson_r'].reindex(SPLITS)
        x = np.arange(len(SPLITS)); w = 0.35
        ax.bar(x - w/2, r2_vals.values, w, label='Internal R² (held-out)', color='#1f77b4', alpha=0.85)
        ax.bar(x + w/2, ext_stats.values, w, label='External r', color='#d62728', alpha=0.85)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=15)
        ax.set_ylabel('Score')
        ax.set_title(f'{task.upper()} — internal vs external')
        ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig5_internal_vs_external.png"); plt.close(fig)
    print("fig5_internal_vs_external.png")

    # ================= FIG 6: Attention heatmap per sample (frozen) ================
    fig, ax = plt.subplots(figsize=(8, 5))
    n_show = 500
    attn_sample = attn_all[:n_show]  # (n, 3)
    im = ax.imshow(attn_sample.T, aspect='auto', cmap='YlGnBu')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['Warhead', 'Linker', 'E3'])
    ax.set_xlabel('Sample index')
    ax.set_title('Attention weights per component (first 500 training samples, frozen model)')
    fig.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig6_attention_heatmap.png"); plt.close(fig)
    print("fig6_attention_heatmap.png")

    print(f"\nAll figures saved to {FIG_DIR}")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")

if __name__ == '__main__':
    main()
