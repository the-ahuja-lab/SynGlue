#!/usr/bin/env python3
"""
External validation of frozen CAT models (v3) — proper per-task comparison:
- DC50 model → predicted log10(DC50) vs reported DC50
- Dmax model  → predicted Dmax (%) vs reported Dmax (%)
Includes p-values and per-target breakdown.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = f"{REV}/benchmarking/generalization_data_v3"
MODEL_DIR = f"{REV}/experiments/gen_training_v3"
OUT_DIR = f"{REV}/experiments/gen_training_v3"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        return (y, attn) if return_attn else y

def build_external_inputs(pca):
    """Consistent encoding: W=matched by POI, L=full PROTAC fp, E=matched by E3."""
    from rdkit import Chem
    def canon(s):
        if not isinstance(s, str) or not s: return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s

    fps = np.load(f"{REV}/benchmarking/tool_results/synglue_external_smiles_fps.npz")['fps'][:, :4800]
    ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    ext = ext.drop_duplicates(subset=['SMILES']).copy()

    wh_df = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/E3_updated_features.csv", low_memory=False)
    w_cols = [c for c in wh_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    wh_by_u, e3_by_n = {}, {}
    for _, r in wh_df.iterrows():
        u = str(r.get('warhead_Uniprot','')).strip().upper()
        if u and u != 'NAN': wh_by_u.setdefault(u, r[w_cols].values.astype(np.float32))
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase','')).strip().upper()
        if n and n != 'NAN': e3_by_n.setdefault(n, r[e_cols].values.astype(np.float32))

    X = np.zeros((len(ext), 3, 4800), dtype=np.float32)
    for i, (_, r) in enumerate(ext.iterrows()):
        poi = str(r.get('POI_UniProt','')).strip().upper()
        X[i, 0, :] = wh_by_u.get(poi, fps[i])
        X[i, 1, :] = fps[i]
        en = str(r.get('E3_Ligase','')).strip().upper()
        X[i, 2, :] = e3_by_n.get(en, fps[i])
    Xp = pca.transform(X.reshape(-1, 4800)).reshape(len(ext), 3, -1).astype(np.float32)
    ext['canon'] = ext['canonical_SMILES'].apply(canon)
    return Xp, ext

def main():
    pca = joblib.load(f"{DATA_DIR}/pca_512.joblib")
    Xp, ext = build_external_inputs(pca)
    print(f"External compounds: {len(ext)}")

    # Reported values (per compound, exact only)
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    rep['canon'] = rep['canonical_SMILES'].apply(lambda s: __import__('rdkit').Chem.MolToSmiles(__import__('rdkit').Chem.MolFromSmiles(s), canonical=True) if isinstance(s,str) and s and __import__('rdkit').Chem.MolFromSmiles(s) else None)

    tasks = {"dc50": ("y_dc50.npy", "DC50"), "dmax": ("y_dmax.npy", "Dmax")}
    splits = ["random", "target_held_out", "e3_held_out", "scaffold_held_out"]
    rows = []

    for task, (yfile, vtype) in tasks.items():
        y_tr = np.load(f"{DATA_DIR}/{yfile}")
        sc = (float(y_tr.mean()), float(y_tr.std()))

        # Reported values for this task
        vals = rep[rep['Value_Type'] == vtype][['canon', 'Value', 'Value_Operator']].copy()
        vals['val'] = pd.to_numeric(vals['Value'], errors='coerce')
        vals = vals[vals['val'].notna() & vals['Value_Operator'].isna()].drop_duplicates(subset=['canon'])
        vals = vals[vals['val'] > 0] if task == 'dc50' else vals
        print(f"\n=== {task.upper()} | reported values: {len(vals)}")

        for split in splits:
            model = CAT().to(DEVICE)
            model.load_state_dict(torch.load(f"{MODEL_DIR}/model_{task}_{split}.pt", map_location=DEVICE))
            model.eval()
            with torch.no_grad():
                yp = model(torch.tensor(Xp, dtype=torch.float32).to(DEVICE)).cpu().numpy() * sc[1] + sc[0]
            ext[f'pred_{task}_{split}'] = yp

            pred_col = f'pred_{task}_{split}'
            merged = ext[['canon','SMILES','Protein_of_Interest','POI_UniProt','E3_Ligase', pred_col]].merge(vals, on='canon', how='inner')
            if task == 'dc50':
                d = merged[merged[pred_col] > 0].copy()
                d['logp'] = np.log10(d[pred_col])
                d['logt'] = np.log10(d['val'])
            else:
                d = merged.copy()
                d['logp'] = d[pred_col]
                d['logt'] = d['val']
            d = d[np.isfinite(d['logp']) & np.isfinite(d['logt'])]

            r, p_r = pearsonr(d['logt'], d['logp'])
            rho, p_rho = spearmanr(d['logt'], d['logp'])
            rows.append({"task": task, "split": split, "n": len(d),
                         "pearson_r": r, "pearson_p": p_r, "spearman_rho": rho, "spearman_p": p_rho,
                         "pred_mean": float(d['logp'].mean()), "true_mean": float(d['logt'].mean())})
            print(f"  [{split:18s}] n={len(d):4d}  r={r:+.3f} (p={p_r:.1e})  ρ={rho:+.3f} (p={p_rho:.1e})")

    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT_DIR}/external_validation_stats.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/external_validation_stats.csv")

    # Save per-compound predictions for plotting
    cols = ['canon','SMILES','Protein_of_Interest','POI_UniProt','E3_Ligase'] + [f'pred_{t}_{s}' for t in tasks for s in splits]
    ext[cols].to_csv(f"{OUT_DIR}/external_predictions_all_splits.csv", index=False)
    print("Saved external_predictions_all_splits.csv")

if __name__ == '__main__':
    main()
