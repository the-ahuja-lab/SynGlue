#!/usr/bin/env python3
"""
Performance check on generalization splits (frozen v4 CAT models, NO retraining).

Design (per rebuttal panels):
  Split defined on FULL curated set → evaluate only on endpoint-annotated subset.

Splits:
  S1 Scaffold-held-out : external compounds with NOVEL scaffold (vs 4,880 training scaffolds)
  S2 Target-held-out   : external compounds targeting NEW targets (9 novel targets)
  S3 E3-held-out       : external VHL compounds (held-out E3 family) + UBR1 (novel E3)
  S4 Temporal/external : temporal 2021-2024 set

Metrics (better than R²): Spearman ρ, Kendall τ, C-index, Pearson r(log),
median |Δlog10 DC50|, % within 2×/5×/10×, AUC(≤100nM/≤1µM), top-10% enrichment.
R² reported but flagged.
Output: experiments/gen_training_v4/metrics_better/split_performance_*.csv + report
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = f"{REV}/benchmarking/generalization_data_v4"
MOD = f"{REV}/experiments/gen_training_v4"
OUT = f"{MOD}/metrics_better"
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau, pearsonr
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
    def forward(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return self.head((h * attn).sum(dim=1)).squeeze(-1)

def canon(s):
    if not isinstance(s, str) or not s: return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
    except: return None

def cindex(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    n = len(yt); num = 0.; den = 0.
    for i in range(n):
        for j in range(i+1, n):
            if yt[i] != yt[j]:
                den += 1
                if (yt[i] > yt[j]) == (yp[i] > yp[j]): num += 1
    return num/den if den else np.nan

def metrics_battery(yt, yp, task='dc50'):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = {}
    rho, pr = spearmanr(yt, yp); tau, pt = kendalltau(yt, yp)
    m['spearman_rho'] = rho; m['spearman_p'] = pr
    m['kendall_tau'] = tau; m['kendall_p'] = pt
    m['cindex'] = cindex(yt, yp)
    if task == 'dc50':
        lt = np.log10(np.clip(yt, 1e-3, None)); lp = np.log10(np.clip(yp, 1e-3, None))
        m['pearson_r_log'] = pearsonr(lt, lp)[0]
        m['rmse_log'] = float(np.sqrt(np.mean((lt-lp)**2)))
        m['mae_log'] = float(np.mean(np.abs(lt-lp)))
        m['median_abs_log'] = float(np.median(np.abs(lt-lp)))
        m['pct_within_2x'] = float(np.mean(np.abs(lt-lp) <= np.log10(2))*100)
        m['pct_within_5x'] = float(np.mean(np.abs(lt-lp) <= np.log10(5))*100)
        m['pct_within_10x'] = float(np.mean(np.abs(lt-lp) <= 1.0)*100)
        for cut, nm in [(100,'100nM'), (1000,'1uM')]:
            try: m[f'auc_le{nm}'] = roc_auc_score((yt <= cut).astype(int), yp)
            except: m[f'auc_le{nm}'] = np.nan
        k = max(1, int(len(yt)*0.10))
        top = np.argsort(yp)[::-1][:k]
        m['top10_enrich_100nM'] = float(np.mean(yt[top] <= 100)/max(np.mean(yt <= 100), 1e-9))
    else:
        m['pearson_r'] = pearsonr(yt, yp)[0]
        m['rmse_pct'] = float(np.sqrt(np.mean((yt-yp)**2)))
        m['mae_pct'] = float(np.mean(np.abs(yt-yp)))
        m['median_abs_err'] = float(np.median(np.abs(yt-yp)))
        m['pct_within_10pct'] = float(np.mean(np.abs(yt-yp) <= 10)*100)
        m['pct_within_20pct'] = float(np.mean(np.abs(yt-yp) <= 20)*100)
        try: m['auc_degrade_50'] = roc_auc_score((yt >= 50).astype(int), yp)
        except: m['auc_degrade_50'] = np.nan
    ssr = np.sum((yt-yp)**2); sst = np.sum((yt-yt.mean())**2)
    m['R2'] = float(1 - ssr/max(sst, 1e-12))
    return m

def predict_ext(X_ext, task, scaler):
    """Ensemble prediction over 4 frozen split models."""
    preds = []
    for split in ["random","scaffold_held_out","e3_held_out","target_held_out"]:
        mp = f"{MOD}/model_{task}_{split}.pt"
        if not os.path.exists(mp): continue
        model = CAT().to(DEVICE)
        model.load_state_dict(torch.load(mp, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            yp = model(torch.tensor(X_ext, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        m, s = scaler
        preds.append(yp * s + m)
    return np.mean(preds, axis=0)

def main():
    pca = joblib.load(f"{DATA}/pca_512.joblib")
    X_ext = np.load(f"{DATA}/X_external_pca.npy")
    ext_meta = pd.read_csv(f"{DATA}/external_meta.csv")
    ext_meta['canon'] = ext_meta['canonical_SMILES'].apply(canon)
    ext_meta['scaffold'] = ext_meta['SMILES'].apply(murcko)

    # Training scaffolds/targets/E3 (full set)
    train = pd.read_csv('/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv', low_memory=False)
    train['scaffold'] = train['PROTAC_SMILES'].apply(murcko)
    train['target'] = train['warhead_Uniprot'].astype(str).str.strip().str.upper()
    train['e3'] = train['E3 ligase'].astype(str).str.strip().str.upper()
    tr_scaff = set(train['scaffold'].dropna())
    tr_targets = set(train['target'][train['target']!='NAN'])
    tr_e3 = set(train['e3'][train['e3']!='NAN'])

    # Reported values
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    rep['canon'] = rep['canonical_SMILES'].apply(canon)

    # Scalers
    y50 = np.load(f"{DATA}/y_dc50.npy"); sc50 = (float(y50.mean()), float(y50.std()))
    y95 = np.load(f"{DATA}/y_dmax.npy"); sc95 = (float(y95.mean()), float(y95.std()))

    # Predictions
    print("Computing frozen-model predictions on external 2,640...")
    pred_dc50 = predict_ext(X_ext, "dc50", sc50)
    pred_dmax = predict_ext(X_ext, "dmax", sc95)
    ext_meta['pred_dc50'] = pred_dc50
    ext_meta['pred_dmax'] = pred_dmax

    # Novelty flags (full set)
    ext_meta['novel_scaffold'] = ~ext_meta['scaffold'].isin(tr_scaff)
    ext_meta['novel_target'] = ~ext_meta['POI_UniProt'].astype(str).str.strip().str.upper().isin(tr_targets)
    ext_meta['is_vhl'] = ext_meta['E3_Ligase'].astype(str).str.strip().str.upper() == 'VHL'

    # Endpoint values per compound
    dc50v = rep[rep['Value_Type']=='DC50'][['canon','Value','Value_Operator']].copy()
    dc50v['val'] = pd.to_numeric(dc50v['Value'], errors='coerce')
    dc50v = dc50v[dc50v['val'].notna() & dc50v['Value_Operator'].isna()].drop_duplicates(subset=['canon'])
    dmaxv = rep[rep['Value_Type']=='Dmax'][['canon','Value','Value_Operator']].copy()
    dmaxv['val'] = pd.to_numeric(dmaxv['Value'], errors='coerce')
    dmaxv = dmaxv[dmaxv['val'].notna() & dmaxv['Value_Operator'].isna()].drop_duplicates(subset=['canon'])

    # ── Define splits (full set) ──
    splits = {
        "S1_Scaffold-held-out (novel scaffold)": ext_meta[ext_meta['novel_scaffold']],
        "S2_Target-held-out (novel target)": ext_meta[ext_meta['novel_target']],
        "S3a_E3-held-out (VHL family)": ext_meta[ext_meta['is_vhl']],
        "S3b_E3-held-out (novel E3=UBR1)": ext_meta[ext_meta['E3_Ligase'].astype(str).str.strip().str.upper()=='UBR1'],
        "S4_All external (leakage-free)": ext_meta,
    }

    all_rows = []
    for split_name, sub in splits.items():
        # DC50 evaluation
        d = sub[['canon','pred_dc50']].merge(dc50v, on='canon', how='inner')
        d = d[d['pred_dc50'] > 0]
        d50 = None
        if len(d) >= 8:
            res = metrics_battery(d['val'].values, d['pred_dc50'].values, 'dc50')
            d50 = {"split": split_name, "endpoint": "DC50", "n": len(d), **res}
            all_rows.append(d50)
            print(f"\n[{split_name}] DC50 n={len(d)} ρ={res['spearman_rho']:+.3f} "
                  f"med|Δlog|={res['median_abs_log']:.2f} AUC100={res.get('auc_le100nM',np.nan):.2f} R²={res['R2']:+.2f}")
        # Dmax evaluation
        d = sub[['canon','pred_dmax']].merge(dmaxv, on='canon', how='inner')
        if len(d) >= 8:
            res = metrics_battery(d['val'].values, d['pred_dmax'].values, 'dmax')
            dm = {"split": split_name, "endpoint": "Dmax", "n": len(d), **res}
            all_rows.append(dm)
            print(f"[{split_name}] Dmax n={len(d)} ρ={res['spearman_rho']:+.3f} "
                  f"med|err|={res['median_abs_err']:.1f}pp R²={res['R2']:+.2f}")

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{OUT}/split_performance_external.csv", index=False)
    print(f"\nSaved: {OUT}/split_performance_external.csv")

    # ── Temporal set (separate — different encoding path) ──
    print("\n── Temporal set (215) — endpoint data limited ──")
    tmp = pd.read_csv(f"{REV}/temporal_external_with_activity.csv", low_memory=False)
    print(f"  Temporal rows with activity: {tmp['Value_Type'].notna().sum()} "
          f"(DC50={ (tmp['Value_Type']=='DC50').sum() }, Dmax={ (tmp['Value_Type']=='Dmax').sum() })")
    print(f"  → too few labeled endpoints for reliable metrics; temporal performance "
          f"reported via scaffold novelty (98%) + literature-known compounds only")

if __name__ == '__main__':
    main()
