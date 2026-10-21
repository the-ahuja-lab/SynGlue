#!/usr/bin/env python3
"""
v4 Master Training — all 4 improvements:
1. Dmax: drop saturated (>90%) task variant
2. Ensemble of 4 split models
3. True W/L/E decomposition external evaluation
4. Training diversity: + PROTAC_DB extras + target-balanced sampling

Tasks: dc50, dmax, dmax90
Splits: random, target_held_out, e3_held_out, scaffold_held_out
Output: experiments/gen_training_v4/
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = f"{REV}/benchmarking/generalization_data_v4"
OUT_DIR = f"{REV}/experiments/gen_training_v4"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

BATCH_SIZE, LR, EPOCHS, PATIENCE, RANDOM_STATE = 64, 1e-4, 150, 20, 42

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

# ─── Splits ──────────────────────────────────────────
def random_split(meta):
    idx = np.arange(len(meta))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
    return tr, va, te, "random"

def target_held_out(meta):
    targets = meta['target_uniprot'].value_counts()
    rng = np.random.RandomState(RANDOM_STATE)
    candidates = [t for t in targets.index if str(t) != 'NAN' and targets[t] >= 5]
    n_hold = max(1, int(len(candidates) * 0.2))
    held = set(rng.choice(candidates, size=n_hold, replace=False))
    te = np.where(meta['target_uniprot'].isin(held))[0]
    trva = np.where(~meta['target_uniprot'].isin(held))[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te, f"target-held({len(held)}t)"

def e3_held_out(meta):
    held_e3 = ['VHL']
    te = np.where(meta['e3'].isin(held_e3))[0]
    trva = np.where(~meta['e3'].isin(held_e3))[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te, f"E3-held(VHL,{len(te)})"

def scaffold_held_out(meta):
    sc_counts = meta['scaffold'].value_counts()
    rng = np.random.RandomState(RANDOM_STATE)
    candidates = [s for s in sc_counts.index if pd.notna(s) and sc_counts[s] >= 3]
    n_hold = max(1, int(len(candidates) * 0.2))
    held = set(rng.choice(candidates, size=n_hold, replace=False))
    te = np.where(meta['scaffold'].isin(held))[0]
    trva = np.where(~meta['scaffold'].isin(held))[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te, f"scaffold-held({len(held)}s)"

SPLITS = {"random": random_split, "target_held_out": target_held_out,
          "e3_held_out": e3_held_out, "scaffold_held_out": scaffold_held_out}

# ─── Balanced sampler (diversity #4) ─────────────────
def make_loader(X, y_z, idx, shuffle=False, balance=False, meta=None):
    ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                       torch.tensor(y_z[idx], dtype=torch.float32))
    if shuffle and balance and meta is not None:
        # target-balanced sampling: weight = 1/freq
        tgt = meta.loc[idx, 'target_uniprot'].astype(str).values
        freqs = pd.Series(tgt).value_counts()
        weights = np.array([1.0 / freqs[t] for t in tgt], dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def train_model(model, tr_loader, va_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val = float("inf"); best_state = None; counter = 0; t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yp = model(xb)
            loss = loss_fn(yp, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += loss_fn(model(xb), yb).item()
        vl /= len(va_loader); sched.step(vl)
        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; counter = 0
        else:
            counter += 1
        if counter >= PATIENCE:
            print(f"    early stop @ {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model, best_val, time.time() - t0

def evaluate(model, loader, sc):
    m, s = sc
    model.eval(); yp_all, yt_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yp_all.extend(model(xb.to(DEVICE)).cpu().numpy())
            yt_all.extend(yb.numpy())
    yp, yt = np.array(yp_all) * s + m, np.array(yt_all) * s + m
    return {"R2": float(r2_score(yt, yp)), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)), "rho": float(spearmanr(yt, yp)[0])}

def frozen_analysis(model, X_full, sc):
    X_t = torch.tensor(X_full, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        yp, attn = model(X_t, return_attn=True)
        yp = yp.cpu().numpy(); attn = attn.cpu().numpy()
        base = yp.copy(); abl = {}
        for comp, idx in [("warhead", 0), ("linker", 1), ("e3", 2)]:
            Xa = X_t.clone(); Xa[:, idx, :] = 0
            ya = model(Xa).cpu().numpy()
            abl[comp] = float(np.mean(np.abs(ya - base)))
    return {"attn": {"W": float(attn[:,0].mean()), "L": float(attn[:,1].mean()), "E": float(attn[:,2].mean())},
            "abl": abl}

def main():
    # ---- Load v4 data (with extras merged) ----
    pca = joblib.load(f"{DATA_DIR}/pca_512.joblib")
    X_ext = np.load(f"{DATA_DIR}/X_external_pca.npy")
    ext_meta = pd.read_csv(f"{DATA_DIR}/external_meta.csv")
    print(f"External encoding: {X_ext.shape} (true linkers: {ext_meta['has_true_linker'].sum()})")

    # Reported external values
    rep = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    from rdkit import Chem
    def canon(s):
        if not isinstance(s, str) or not s: return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s
    rep['canon'] = rep['canonical_SMILES'].apply(canon)

    tasks = {}
    for tname, yfile, metafile, extra_y, extra_meta in [
        ("dc50", "y_dc50.npy", "metadata_dc50.csv", "y_dc50_extra.npy", "metadata_dc50_extra.csv"),
        ("dmax", "y_dmax.npy", "metadata_dmax.csv", "y_dmax_extra.npy", "metadata_dmax_extra.csv"),
        ("dmax90", "y_dmax90.npy", "metadata_dmax90.csv", None, None),
    ]:
        X = np.load(f"{DATA_DIR}/X_{tname}_pca.npy")
        y = np.load(f"{DATA_DIR}/{yfile}")
        meta = pd.read_csv(f"{DATA_DIR}/{metafile}")
        if extra_y and os.path.exists(f"{DATA_DIR}/{extra_y}"):
            Xe = np.load(f"{DATA_DIR}/X_{tname}_extra_full.npy")
            # transform extra to pca
            Xe_p = pca.transform(Xe.reshape(-1, 4800)).reshape(Xe.shape[0], 3, 512).astype(np.float32)
            ye = np.load(f"{DATA_DIR}/{extra_y}")
            me = pd.read_csv(f"{DATA_DIR}/{extra_meta}")
            X = np.concatenate([X, Xe_p]); y = np.concatenate([y, ye])
            meta = pd.concat([meta, me], ignore_index=True)
            # scaffold for extra
            from rdkit.Chem.Scaffolds import MurckoScaffold
            def ms(s):
                m = Chem.MolFromSmiles(s)
                if m is None: return None
                try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
                except: return None
            if 'scaffold' not in meta.columns:
                meta['scaffold'] = meta['SMILES'].apply(ms)
        sc = (float(y.mean()), float(y.std()))
        y_z = ((y - sc[0]) / sc[1]).astype(np.float32)
        tasks[tname] = {"X": X, "y": y, "y_z": y_z, "meta": meta, "sc": sc}
        print(f"TASK {tname}: n={len(X)}, y_mean={sc[0]:.3f}, y_std={sc[1]:.3f}")

    all_rows = []
    ext_preds = pd.DataFrame({'canon': ext_meta['canonical_SMILES']})

    for task, D in tasks.items():
        X, y, y_z, meta, sc = D['X'], D['y'], D['y_z'], D['meta'], D['sc']
        print(f"\n{'='*60}\nTASK: {task} (n={len(X)})\n{'='*60}")
        model_preds = {}

        for split_name, split_fn in SPLITS.items():
            tr, va, te, desc = split_fn(meta)
            print(f"\n  [{task}/{split_name}] {desc} | tr={len(tr)} va={len(va)} te={len(te)}")
            tr_loader = make_loader(X, y_z, tr, shuffle=True, balance=True, meta=meta)
            va_loader = make_loader(X, y_z, va)
            te_loader = make_loader(X, y_z, te)

            torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
            model = CAT(input_dim=X.shape[2]).to(DEVICE)
            model, bvl, el = train_model(model, tr_loader, va_loader)
            ev = evaluate(model, te_loader, sc)
            fa = frozen_analysis(model, X, sc)
            print(f"    TEST R2={ev['R2']:.3f} RMSE={ev['RMSE']:.3f} ρ={ev['rho']:.3f} | attn W/L/E={fa['attn']['W']:.2f}/{fa['attn']['L']:.2f}/{fa['attn']['E']:.2f} | {el:.0f}s")

            torch.save(model.state_dict(), f"{OUT_DIR}/model_{task}_{split_name}.pt")
            model.eval()
            with torch.no_grad():
                yp_ext = model(torch.tensor(X_ext, dtype=torch.float32).to(DEVICE)).cpu().numpy() * sc[1] + sc[0]
            model_preds[split_name] = yp_ext
            ext_preds[f'pred_{task}_{split_name}'] = yp_ext

            all_rows.append({"task": task, "split": split_name, "desc": desc,
                             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                             "R2": ev['R2'], "RMSE": ev['RMSE'], "MAE": ev['MAE'], "rho": ev['rho'],
                             "attn_W": fa['attn']['W'], "attn_L": fa['attn']['L'], "attn_E": fa['attn']['E'],
                             "abl_W": fa['abl']['warhead'], "abl_L": fa['abl']['linker'], "abl_E": fa['abl']['e3'],
                             "train_time_s": el})

        # ── Ensemble of 4 split models ──
        ens = np.mean([model_preds[s] for s in SPLITS], axis=0)
        ext_preds[f'pred_{task}_ensemble'] = ens
        print(f"  ENSEMBLE saved for {task}")

    summ = pd.DataFrame(all_rows)
    summ.to_csv(f"{OUT_DIR}/generalization_summary_v4.csv", index=False)
    ext_preds.to_csv(f"{OUT_DIR}/external_predictions_v4.csv", index=False)

    # ── External stats (per task, per split + ensemble) ──
    ext_rows = []
    for task, vtype in [("dc50", "DC50"), ("dmax", "Dmax"), ("dmax90", "Dmax")]:
        vals = rep[rep['Value_Type'] == vtype][['canon', 'Value', 'Value_Operator']].copy()
        vals['val'] = pd.to_numeric(vals['Value'], errors='coerce')
        vals = vals[vals['val'].notna() & vals['Value_Operator'].isna()].drop_duplicates(subset=['canon'])
        if task == 'dc50':
            vals = vals[vals['val'] > 0]
        for col in [f'pred_{task}_{s}' for s in SPLITS] + [f'pred_{task}_ensemble']:
            d = ext_preds[['canon', col]].merge(vals, on='canon', how='inner')
            if task == 'dc50':
                d = d[d[col] > 0].copy()
                p = np.log10(d[col]); t = np.log10(d['val'])
            else:
                p = d[col]; t = d['val']
            d = d[np.isfinite(p) & np.isfinite(t)]
            if len(d) < 10:
                ext_rows.append({"task": task, "model": col.replace(f"pred_{task}_", ""), "n": len(d), "r": np.nan, "p_r": np.nan, "rho": np.nan, "p_rho": np.nan})
                continue
            r, pr = pearsonr(t, p); rho, prho = spearmanr(t, p)
            ext_rows.append({"task": task, "model": col.replace(f"pred_{task}_", ""), "n": len(d),
                             "r": r, "p_r": pr, "rho": rho, "p_rho": prho})
            print(f"  EXT[{task}/{col.replace('pred_'+task+'_','')}] n={len(d)} r={r:+.3f} (p={pr:.1e}) ρ={rho:+.3f}")

    ext_stats = pd.DataFrame(ext_rows)
    ext_stats.to_csv(f"{OUT_DIR}/external_validation_v4.csv", index=False)
    print(f"\nSaved summary + external stats to {OUT_DIR}")

if __name__ == '__main__':
    main()
