#!/usr/bin/env python3
"""
TRUE GENERALIZATION TEST — external data (NOT in training) on FINAL frozen model.

4 splits (all from external 2,640, none in training):
  S1 All external (leakage-free)
  S2 Novel scaffold (2,555 — scaffolds not in training)
  S3 Novel target (119 — targets not in training)
  S4 Novel E3 (2 — E3 UBR1 not in training)

Predict DC50 & Dmax with the final frozen ProtacAttentionModel
(shared transformer + per-endpoint RF stage-2), compute metrics.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/final_results"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau, pearsonr
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

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
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = {'R2': float(r2_score(yt,yp)), 'RMSE': float(np.sqrt(mean_squared_error(yt,yp))),
         'MAE': float(mean_absolute_error(yt,yp)), 'rho': float(spearmanr(yt,yp)[0]),
         'tau': float(kendalltau(yt,yp)[0]), 'pearson': float(pearsonr(yt,yp)[0])}
    if task == 'dc50':
        d = np.abs(yt - yp)
        m['med_abs_log'] = float(np.median(d))
        m['within_2x'] = float(np.mean(d <= np.log10(2))*100)
        m['within_5x'] = float(np.mean(d <= np.log10(5))*100)
        m['within_10x'] = float(np.mean(d <= 1.0)*100)
    else:
        m['med_abs_err'] = float(np.median(np.abs(yt-yp)))
        m['within_10pct'] = float(np.mean(np.abs(yt-yp) <= 10)*100)
        m['within_20pct'] = float(np.mean(np.abs(yt-yp) <= 20)*100)
    return m

def main():
    # ── Load final frozen model (trained on TRAINING data only) ──
    model = ProtacAttentionModel(input_dim=4800).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT}/protac_attention_model_both.pt", map_location=DEVICE))
    model.eval()
    rf_dc50 = joblib.load(f"{OUT}/stage2_rf_dc50.joblib")
    rf_dmax = joblib.load(f"{OUT}/stage2_rf_dmax.joblib")
    print("Loaded final frozen model + stage-2 RFs", flush=True)

    # ── External encoding (3×4800, matches training input distribution) ──
    X_ext = np.load(f"{REV}/benchmarking/generalization_data_v4/X_external_full.npy")
    ext_meta = pd.read_csv(f"{REV}/benchmarking/generalization_data_v4/external_meta.csv")
    print(f"External encoding: {X_ext.shape}", flush=True)

    # Training reference for novelty flags
    train = pd.read_csv('/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv', low_memory=False)
    train['scaffold'] = train['PROTAC_SMILES'].apply(murcko)
    train['target'] = train['warhead_Uniprot'].astype(str).str.strip().str.upper()
    train['e3'] = train['E3 ligase'].astype(str).str.strip().str.upper()
    tr_scaff = set(train['scaffold'].dropna())
    tr_targets = set(train['target'][train['target']!='NAN'])
    tr_e3 = set(train['e3'][train['e3']!='NAN'])

    # Novelty flags (ext_meta order = X_ext order)
    ext_meta['scaffold'] = ext_meta['SMILES'].apply(murcko)
    ext_meta['target'] = ext_meta['POI_UniProt'].astype(str).str.strip().str.upper()
    ext_meta['e3'] = ext_meta['E3_Ligase'].astype(str).str.strip().str.upper()
    ext_meta['novel_scaffold'] = ~ext_meta['scaffold'].isin(tr_scaff)
    ext_meta['novel_target'] = ~ext_meta['target'].isin(tr_targets)
    ext_meta['novel_e3'] = ~ext_meta['e3'].isin(tr_e3)

    # Reported values (per compound, exact DC50/Dmax)
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    rep['canon'] = rep['canonical_SMILES'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) if Chem.MolFromSmiles(s) else None)
    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    ext_meta['canon'] = ext_meta['SMILES'].apply(canon)

    dc50v = rep[rep['Value_Type']=='DC50'][['canon','Value','Value_Operator']].copy()
    dc50v['val'] = pd.to_numeric(dc50v['Value'], errors='coerce')
    dc50v = dc50v[dc50v['val'].notna() & dc50v['Value_Operator'].isna() & (dc50v['val']>0)].drop_duplicates(subset=['canon'])
    dmaxv = rep[rep['Value_Type']=='Dmax'][['canon','Value','Value_Operator']].copy()
    dmaxv['val'] = pd.to_numeric(dmaxv['Value'], errors='coerce')
    dmaxv = dmaxv[dmaxv['val'].notna() & dmaxv['Value_Operator'].isna()].drop_duplicates(subset=['canon'])

    # ── Frozen inference on ALL external ──
    with torch.no_grad():
        pred_dc50_log = model(torch.tensor(X_ext, dtype=torch.float32).to(DEVICE))[0].detach().cpu().numpy()
    with torch.no_grad():
        Z_ext = model.fused_embeddings(torch.tensor(X_ext, dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
    pred_dmax = rf_dmax.predict(Z_ext)

    # ── 4 splits ──
    splits = {
        "S1_All_external": np.ones(len(ext_meta), bool),
        "S2_Novel_scaffold": ext_meta['novel_scaffold'].values,
        "S3_Novel_target": ext_meta['novel_target'].values,
        "S4_Novel_E3": ext_meta['novel_e3'].values,
    }

    all_rows = []
    for split_name, mask in splits.items():
        sub = ext_meta[mask].copy()
        sub['pred_dc50'] = pred_dc50_log[mask]  # log10 DC50
        sub['pred_dmax'] = pred_dmax[mask]
        # DC50 metrics
        d = sub[['canon','pred_dc50']].merge(dc50v, on='canon', how='inner')
        d = d[d['pred_dc50'] > -5]  # sanity
        if len(d) >= 8:
            m50 = metrics(d['val'].values, 10**d['pred_dc50'].values, 'dc50')  # convert to nM
            row50 = {"split": split_name, "endpoint": "DC50", "n": len(d), **m50}
            all_rows.append(row50)
            print(f"[{split_name}] DC50 n={len(d)} R²={m50['R2']:+.3f} ρ={m50['rho']:+.3f} "
                  f"med|Δlog|={m50['med_abs_log']:.2f} within10x={m50['within_10x']:.0f}%", flush=True)
        # Dmax metrics
        d = sub[['canon','pred_dmax']].merge(dmaxv, on='canon', how='inner')
        if len(d) >= 8:
            m95 = metrics(d['val'].values, d['pred_dmax'].values, 'dmax')
            row95 = {"split": split_name, "endpoint": "Dmax", "n": len(d), **m95}
            all_rows.append(row95)
            print(f"[{split_name}] Dmax n={len(d)} R²={m95['R2']:+.3f} ρ={m95['rho']:+.3f} "
                  f"med|err|={m95['med_abs_err']:.1f}pp", flush=True)

    res = pd.DataFrame(all_rows)
    res.to_csv(f"{OUT}/external_generalization_frozen.csv", index=False)
    print(f"\nSaved: {OUT}/external_generalization_frozen.csv", flush=True)
    print(res[['split','endpoint','n','R2','rho','MAE']].to_string(index=False), flush=True)

    # Save external predictions
    out = ext_meta[['SMILES','canon','POI_UniProt','E3_Ligase','novel_scaffold','novel_target','novel_e3']].copy()
    out['pred_DC50_nM'] = 10**pred_dc50_log
    out['pred_Dmax_pct'] = pred_dmax
    out.to_csv(f"{OUT}/external_predictions_frozen.csv", index=False)
    print(f"Saved external predictions: {OUT}/external_predictions_frozen.csv", flush=True)

if __name__ == '__main__':
    main()
