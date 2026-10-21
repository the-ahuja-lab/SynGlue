#!/usr/bin/env python3
"""
4-SPLIT GENERALIZATION TEST on the SAME frozen ProtacAttentionModel
(random / scaffold-held / target-held / E3-held) + better plots + unique interpretability.

Splits defined on full mapped set; evaluated per endpoint-labeled subset.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/final_results"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau, pearsonr
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ProtacAttentionModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        h = self.proj(x); h = self.transformer(h)
        a = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * a).sum(dim=1)
        return self.regressor(fused).squeeze(-1), a
    def fused_embeddings(self, x):
        h = self.proj(x); h = self.transformer(h)
        a = torch.softmax(self.attn_pool(h), dim=1)
        return (h * a).sum(dim=1), a

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
    except: return None

def metrics(yt, yp, task):
    m = {'R2': float(r2_score(yt,yp)), 'RMSE': float(np.sqrt(mean_squared_error(yt,yp))),
         'MAE': float(mean_absolute_error(yt,yp)), 'rho': float(spearmanr(yt,yp)[0]),
         'tau': float(kendalltau(yt,yp)[0]), 'pearson': float(pearsonr(yt,yp)[0])}
    if task == 'dc50':
        d = np.abs(yt-yp); m['med_abs_log']=float(np.median(d))
        m['within_2x']=float(np.mean(d<=np.log10(2))*100); m['within_5x']=float(np.mean(d<=np.log10(5))*100)
        m['within_10x']=float(np.mean(d<=1.0)*100)
    else:
        m['med_abs_err']=float(np.median(np.abs(yt-yp))); m['within_10pct']=float(np.mean(np.abs(yt-yp)<=10)*100)
    return m

def main():
    # Load saved model + embeddings + data
    import joblib
    model = ProtacAttentionModel(input_dim=4800).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT}/protac_attention_model_both.pt", map_location=DEVICE))
    model.eval()
    Z = np.load(f"{OUT}/Z_all.npy")
    # rebuild dataset for metadata
    sys.path.insert(0, REV)
    from train_final_both_endpoints import build_dataset
    X, df = build_dataset()
    logdc = df['logDC50'].values.astype(float)
    dmax = df['Dmax'].values.astype(float)
    df['scaffold'] = df['smiles'].apply(murcko)

    # Splits (on full set)
    idx = np.arange(len(df))
    rng = np.random.RandomState(42)
    splits = {}

    # 1. Random
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    splits['Random'] = te

    # 2. Scaffold-held-out
    sc_counts = df['scaffold'].value_counts()
    cand = [s for s in sc_counts.index if pd.notna(s) and sc_counts[s] >= 3]
    n_hold = max(1, int(len(cand)*0.2))
    held_s = set(rng.choice(cand, size=n_hold, replace=False))
    splits['Scaffold-held'] = np.where(df['scaffold'].isin(held_s))[0]

    # 3. Target-held-out
    tg_counts = df['target'].value_counts()
    cand_t = [t for t in tg_counts.index if str(t)!='NAN' and tg_counts[t] >= 5]
    n_hold_t = max(1, int(len(cand_t)*0.2))
    held_t = set(rng.choice(cand_t, size=n_hold_t, replace=False))
    splits['Target-held'] = np.where(df['target'].isin(held_t))[0]

    # 4. E3-held-out (VHL)
    splits['E3-held (VHL)'] = np.where(df['e3'] == 'VHL')[0]

    rows = []
    for split_name, te_idx in splits.items():
        for task, y, name in [('dc50', logdc, 'DC50'), ('dmax', dmax, 'Dmax')]:
            mask = np.isfinite(y)
            te_f = te_idx[mask[te_idx]]
            if len(te_f) < 8:
                print(f"[{split_name}] {name}: n={len(te_f)} too few", flush=True); continue
            tr_f = np.setdiff1d(np.where(mask)[0], te_f)
            # RF trained on train subset (same frozen transformer embeddings)
            rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
            rf.fit(Z[tr_f], y[tr_f])
            prf = rf.predict(Z[te_f])
            m = metrics(y[te_f], prf, task)
            rows.append({"split": split_name, "endpoint": name, "n": len(te_f), **m})
            print(f"[{split_name}] {name}: n={len(te_f)} R²={m['R2']:.3f} ρ={m['rho']:.3f} "
                  f"MAE={m['MAE']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/four_split_generalization.csv", index=False)
    print(f"\nSaved: {OUT}/four_split_generalization.csv", flush=True)
    print(res.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
