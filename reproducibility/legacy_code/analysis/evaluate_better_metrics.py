#!/usr/bin/env python3
"""
Generalized CAT model — comprehensive evaluation with BETTER metrics than R².

For each task (DC50, Dmax, Dmax90) × split (random, scaffold-held, E3-held, target-held),
compute a battery of metrics appropriate for OOD generalization:
  - Ranking:   Spearman ρ, Kendall τ, pairwise ranking accuracy (C-index)
  - Correlation: Pearson r on log-scale DC50 / raw Dmax
  - Error:     RMSE, MAE, median |Δlog10 DC50| (fold-error), % within 2×/5×/10×
  - Enrichment: AUC(≤100nM), AUC(≤1µM), top-10% enrichment factor
  - R² (reported but flagged as unreliable for OOD)

Also: per-split metric table + external validation (2,640 compounds) with same battery.
Output: revision/experiments/gen_training_v4/metrics_better/
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau, pearsonr

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

SPLITS = ["random", "scaffold_held_out", "e3_held_out", "target_held_out"]

def cindex(y_true, y_pred):
    """Pairwise ranking accuracy (concordance index)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    num = 0.0; den = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:
                den += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    num += 1
    return num / den if den > 0 else np.nan

def metrics_battery(y_true, y_pred, task='dc50'):
    """Compute comprehensive metrics in ORIGINAL units."""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    m = {}
    # Ranking (robust to calibration)
    rho, p_rho = spearmanr(y_true, y_pred)
    tau, p_tau = kendalltau(y_true, y_pred)
    m['spearman_rho'] = rho; m['spearman_p'] = p_rho
    m['kendall_tau'] = tau; m['kendall_p'] = p_tau
    m['cindex'] = cindex(y_true, y_pred)

    if task == 'dc50':
        # log-scale
        lt = np.log10(np.clip(y_true, 1e-3, None)); lp = np.log10(np.clip(y_pred, 1e-3, None))
        r_p, _ = pearsonr(lt, lp)
        m['pearson_r_log'] = r_p
        m['rmse_log'] = float(np.sqrt(np.mean((lt - lp)**2)))
        m['mae_log'] = float(np.mean(np.abs(lt - lp)))
        m['median_abs_log'] = float(np.median(np.abs(lt - lp)))   # fold error (1.0 = 10x)
        m['pct_within_2x'] = float(np.mean(np.abs(lt - lp) <= np.log10(2)) * 100)
        m['pct_within_5x'] = float(np.mean(np.abs(lt - lp) <= np.log10(5)) * 100)
        m['pct_within_10x'] = float(np.mean(np.abs(lt - lp) <= 1.0) * 100)
        # enrichment: active = DC50 ≤ 100nM / ≤ 1µM
        for cutoff, name in [(100, '100nM'), (1000, '1uM')]:
            try:
                m[f'auc_le{name}'] = roc_auc_score((y_true <= cutoff).astype(int), y_pred)
            except Exception:
                m[f'auc_le{name}'] = np.nan
        # top-10% enrichment (baseline = fraction active)
        k = max(1, int(len(y_true) * 0.10))
        top_idx = np.argsort(y_pred)[::-1][:k]
        frac_active_top = np.mean(y_true[top_idx] <= 100)
        frac_active_all = np.mean(y_true <= 100)
        m['top10_enrich_100nM'] = float(frac_active_top / max(frac_active_all, 1e-9))
    else:
        # Dmax (%), also log-ish transform for fold-type metrics
        lt = y_true; lp = y_pred
        r_p, _ = pearsonr(lt, lp)
        m['pearson_r'] = r_p
        m['rmse_pct'] = float(np.sqrt(np.mean((lt - lp)**2)))
        m['mae_pct'] = float(np.mean(np.abs(lt - lp)))
        m['median_abs_err'] = float(np.median(np.abs(lt - lp)))
        m['pct_within_10pct'] = float(np.mean(np.abs(lt - lp) <= 10) * 100)
        m['pct_within_20pct'] = float(np.mean(np.abs(lt - lp) <= 20) * 100)
        try:
            m['auc_degrade_50'] = roc_auc_score((y_true >= 50).astype(int), y_pred)
        except Exception:
            m['auc_degrade_50'] = np.nan

    # R² (flagged — unreliable for OOD)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    m['R2'] = float(1 - ss_res / max(ss_tot, 1e-12))
    return m

def evaluate_split(X, y, meta, tr, va, te, model_path, task, sc):
    """Load frozen model, predict on test split, compute metrics."""
    model = CAT().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X[te], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    yt = y[te].astype(float)
    # rescale predictions from z to original; TRUE labels are already in original units
    m, s = sc
    yp_orig = yp * s + m
    yt_orig = yt
    res = metrics_battery(yt_orig, yp_orig, task)
    return res

def main():
    pca = joblib.load(f"{DATA}/pca_512.joblib")

    # Re-derive split functions (same as train_generalization_v4.py, seed 42)
    RANDOM_STATE = 42
    def random_split(meta):
        idx = np.arange(len(meta))
        tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
        tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
        return tr, va, te
    def target_held_out(meta):
        targets = meta['target_uniprot'].value_counts()
        rng = np.random.RandomState(RANDOM_STATE)
        candidates = [t for t in targets.index if str(t) != 'NAN' and targets[t] >= 5]
        n_hold = max(1, int(len(candidates) * 0.2))
        held = set(rng.choice(candidates, size=n_hold, replace=False))
        te = np.where(meta['target_uniprot'].isin(held))[0]
        trva = np.where(~meta['target_uniprot'].isin(held))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    def e3_held_out(meta):
        held_e3 = ['VHL']
        te = np.where(meta['e3'].isin(held_e3))[0]
        trva = np.where(~meta['e3'].isin(held_e3))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    def scaffold_held_out(meta):
        sc_counts = meta['scaffold'].value_counts()
        rng = np.random.RandomState(RANDOM_STATE)
        candidates = [s for s in sc_counts.index if pd.notna(s) and sc_counts[s] >= 3]
        n_hold = max(1, int(len(candidates) * 0.2))
        held = set(rng.choice(candidates, size=n_hold, replace=False))
        te = np.where(meta['scaffold'].isin(held))[0]
        trva = np.where(~meta['scaffold'].isin(held))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    SPLIT_FNS = {"random": random_split, "scaffold_held_out": scaffold_held_out,
                 "e3_held_out": e3_held_out, "target_held_out": target_held_out}

    # Load external encoding (v4 consistent)
    X_ext = np.load(f"{DATA}/X_external_pca.npy")
    ext_meta = pd.read_csv(f"{DATA}/external_meta.csv")
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    from rdkit import Chem
    def canon(s):
        if not isinstance(s, str) or not s: return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s
    rep['canon'] = rep['canonical_SMILES'].apply(canon)

    tasks = {
        "dc50": {"yfile": "y_dc50.npy", "metafile": "metadata_dc50.csv",
                 "extra_y": "y_dc50_extra.npy", "extra_meta": "metadata_dc50_extra.csv",
                 "extra_X": "X_dc50_extra_full.npy"},
        "dmax": {"yfile": "y_dmax.npy", "metafile": "metadata_dmax.csv",
                 "extra_y": "y_dmax_extra.npy", "extra_meta": "metadata_dmax_extra.csv",
                 "extra_X": "X_dmax_extra_full.npy"},
        "dmax90": {"yfile": "y_dmax90.npy", "metafile": "metadata_dmax90.csv",
                   "extra_y": None, "extra_meta": None, "extra_X": None},
    }
    vtype = {"dc50": "DC50", "dmax": "Dmax", "dmax90": "Dmax"}

    all_rows = []
    ext_rows = []

    for task, D in tasks.items():
        X = np.load(f"{DATA}/X_{task}_pca.npy")
        y = np.load(f"{DATA}/{D['yfile']}")
        meta = pd.read_csv(f"{DATA}/{D['metafile']}")
        if D['extra_y'] and os.path.exists(f"{DATA}/{D['extra_y']}"):
            Xe = np.load(f"{DATA}/{D['extra_X']}")
            Xe_p = pca.transform(Xe.reshape(-1, 4800)).reshape(Xe.shape[0], 3, 512).astype(np.float32)
            ye = np.load(f"{DATA}/{D['extra_y']}")
            me = pd.read_csv(f"{DATA}/{D['extra_meta']}")
            X = np.concatenate([X, Xe_p]); y = np.concatenate([y, ye])
            meta = pd.concat([meta, me], ignore_index=True)
        sc = (float(y.mean()), float(y.std()))
        y_z = ((y - sc[0]) / sc[1]).astype(np.float32)
        print(f"\n=== TASK {task} (n={len(X)}) ===")

        for split in SPLITS:
            tr, va, te = SPLIT_FNS[split](meta)
            model_path = f"{MOD}/model_{task}_{split}.pt"
            if not os.path.exists(model_path):
                print(f"  missing {model_path}"); continue
            res = evaluate_split(X, y, meta, tr, va, te, model_path, task, sc)
            res = {"task": task, "split": split, "n_test": len(te), **res}
            all_rows.append(res)
            print(f"  [{split:18s}] n={len(te):4d} ρ={res['spearman_rho']:+.3f} "
                  f"τ={res['kendall_tau']:+.3f} C={res['cindex']:.3f} "
                  f"med|Δlog|={res.get('median_abs_log', res.get('median_abs_err', np.nan)):.2f} R²={res['R2']:+.2f}")

        # ── External validation (same battery) ──
        print(f"  ── external (n={len(X_ext)}) ──")
        vals = rep[rep['Value_Type'] == vtype[task]][['canon', 'Value', 'Value_Operator']].copy()
        vals['val'] = pd.to_numeric(vals['Value'], errors='coerce')
        vals = vals[vals['val'].notna() & vals['Value_Operator'].isna()].drop_duplicates(subset=['canon'])
        if task == 'dc50':
            vals = vals[vals['val'] > 0]

        ext_preds = {}
        for split in SPLITS:
            model_path = f"{MOD}/model_{task}_{split}.pt"
            if not os.path.exists(model_path): continue
            model = CAT().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            with torch.no_grad():
                yp = model(torch.tensor(X_ext, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            yp_orig = yp * sc[1] + sc[0]
            ext_preds[split] = yp_orig
        # ensemble
        if ext_preds:
            ext_preds['ensemble'] = np.mean([ext_preds[s] for s in ext_preds], axis=0)
            for split, yp_orig in ext_preds.items():
                d = pd.DataFrame({'canon': ext_meta['canonical_SMILES'].apply(canon), 'pred': yp_orig}).merge(vals, on='canon', how='inner')
                d = d[np.isfinite(d['pred'])]
                if task == 'dc50':
                    d = d[d['pred'] > 0]
                if len(d) < 10: continue
                res = metrics_battery(d['val'].values, d['pred'].values, task)
                res = {"task": task, "split": "external:" + split, "n_test": len(d), **res}
                ext_rows.append(res)
                key = 'median_abs_log' if task == 'dc50' else 'median_abs_err'
                print(f"  [ext:{split:12s}] n={len(d):4d} ρ={res['spearman_rho']:+.3f} "
                      f"τ={res['kendall_tau']:+.3f} med|Δlog|={res.get(key, np.nan):.2f} "
                      f"AUC100nM={res.get('auc_le100nM', np.nan):.2f} R²={res['R2']:+.2f}")

    pd.DataFrame(all_rows).to_csv(f"{OUT}/split_metrics_better.csv", index=False)
    pd.DataFrame(ext_rows).to_csv(f"{OUT}/external_metrics_better.csv", index=False)
    print(f"\nSaved: {OUT}/split_metrics_better.csv + external_metrics_better.csv")

if __name__ == '__main__':
    main()
