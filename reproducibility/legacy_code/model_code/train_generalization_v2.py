#!/usr/bin/env python3
"""
Component-Aware Transformer v2 — Separate-task training with PCA-reduced input.
Generalization splits (random / target-held / E3-held / scaffold-held) +
external validation on the 2,640 external compounds (matched W/E vectors,
full-PROTAC as linker, PCA-transformed).

Frozen model analysis per split: attention, ablations, external correlation.
Output: revision/experiments/gen_training_v3/
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = f"{REV}/benchmarking/generalization_data_v3"
OUT_DIR = f"{REV}/experiments/gen_training_v3"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

BATCH_SIZE, LR, EPOCHS, PATIENCE, RANDOM_STATE = 64, 1e-4, 150, 20, 42

# ─── Component-Aware Transformer (PCA input) ─────────
class CAT(nn.Module):
    """Component-Aware Transformer: [W|L|E] tokens + positional identity."""
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
        if return_attn:
            return y, attn
        return y

    def embed(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn).sum(dim=1)

# ─── Splits ──────────────────────────────────────────
def random_split(meta):
    idx = np.arange(len(meta))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
    return tr, va, te, "random"

def target_held_out(meta):
    targets = meta['target_uniprot'].value_counts()
    rng = np.random.RandomState(RANDOM_STATE)
    candidates = [t for t in targets.index if t != 'NAN' and targets[t] >= 5]
    n_hold = max(1, int(len(candidates) * 0.2))
    held = set(rng.choice(candidates, size=n_hold, replace=False))
    te = np.where(meta['target_uniprot'].isin(held))[0]
    trva = np.where(~meta['target_uniprot'].isin(held))[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te, f"target-held({len(held)} t)"

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
    return tr, va, te, f"scaffold-held({len(held)} s)"

SPLITS = {
    "random": random_split,
    "target_held_out": target_held_out,
    "e3_held_out": e3_held_out,
    "scaffold_held_out": scaffold_held_out,
}

# ─── Training ────────────────────────────────────────
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

# ─── Frozen analysis ─────────────────────────────────
def frozen_analysis(model, X_full, y_full, sc):
    """Attention, ablations, internal correlation on full training set."""
    X_t = torch.tensor(X_full, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        yp, attn = model(X_t, return_attn=True)
        yp = yp.cpu().numpy(); attn = attn.cpu().numpy()
        base = yp.copy()
        abl = {}
        for comp, idx in [("warhead", 0), ("linker", 1), ("e3", 2)]:
            Xa = X_t.clone(); Xa[:, idx, :] = 0
            ya = model(Xa).cpu().numpy()
            abl[comp] = {"delta_z": float(np.mean(np.abs(ya - base))),
                         "rmse_z": float(np.sqrt(np.mean((ya - base)**2)))}
        Z = model.embed(X_t).cpu().numpy()
        # internal correlation in rescaled space
        m, s = sc
        yt = y_full * s + m
        yp_r = yp * s + m
        r_int = float(pearsonr(yt, yp_r)[0])
    return {
        "attn": {"W": float(attn[:,0].mean()), "L": float(attn[:,1].mean()), "E": float(attn[:,2].mean())},
        "abl": abl,
        "emb_norm_mean": float(np.mean(np.linalg.norm(Z, axis=1))),
        "internal_r": r_int,
    }

def external_validation(model, pca, scaler, tag):
    """Predict on external non-training set using matched W/E + full-PROTAC linker."""
    from rdkit import Chem
    def canon(s):
        if not isinstance(s, str) or not s: return None
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s

    # External GROVER fps (cached) + training component vectors
    fps = np.load(f"{REV}/benchmarking/tool_results/synglue_external_smiles_fps.npz")['fps'][:, :4800]
    ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    ext = ext.drop_duplicates(subset=['SMILES']).copy()

    warhead_df = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/E3_updated_features.csv", low_memory=False)
    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    wh_by_uniprot, e3_by_name = {}, {}
    for _, r in warhead_df.iterrows():
        u = str(r.get('warhead_Uniprot','')).strip().upper()
        if u and u != 'NAN': wh_by_uniprot.setdefault(u, r[w_cols].values.astype(np.float32))
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase','')).strip().upper()
        if n and n != 'NAN': e3_by_name.setdefault(n, r[e_cols].values.astype(np.float32))

    X_new = np.zeros((len(ext), 3, 4800), dtype=np.float32)
    for i, (_, r) in enumerate(ext.iterrows()):
        poi = str(r.get('POI_UniProt','')).strip().upper()
        X_new[i, 0, :] = wh_by_uniprot.get(poi, fps[i])
        X_new[i, 1, :] = fps[i]
        en = str(r.get('E3_Ligase','')).strip().upper()
        X_new[i, 2, :] = e3_by_name.get(en, fps[i])

    X_pca = pca.transform(X_new.reshape(-1, 4800)).reshape(len(ext), 3, -1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_pca, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    m, s = scaler
    yp_rescaled = yp * s + m

    ext['pred'] = yp_rescaled
    ext['canon'] = ext['canonical_SMILES'].apply(canon)

    # Merge with reported DC50 (exact only)
    dc50 = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    dc50 = dc50[dc50['Value_Type'] == 'DC50'][['canonical_SMILES', 'Value', 'Value_Operator']].copy()
    dc50['canon'] = dc50['canonical_SMILES'].apply(canon)
    dc50['val'] = pd.to_numeric(dc50['Value'], errors='coerce')
    dc50 = dc50[dc50['val'].notna() & dc50['Value_Operator'].isna()].drop_duplicates(subset=['canon'])

    merged = ext.merge(dc50, on='canon', how='inner')
    merged = merged[merged['val'] > 0]
    if len(merged) < 10:
        return {"n": 0, "r": None, "rho": None}

    if 'dc50' in tag.lower():
        merged = merged[merged['pred'] > 0]
        p = np.log10(merged['pred']); t = np.log10(merged['val'])
        r = float(pearsonr(t, p)[0]); rho = float(spearmanr(t, p)[0])
    else:
        p = merged['pred']; t = np.clip(merged['val'], 0, 100)
        r = float(pearsonr(t, p)[0]); rho = float(spearmanr(t, p)[0])
    return {"n": len(merged), "r": r, "rho": rho, "pred_mean": float(merged['pred'].mean())}

# ─── Main ────────────────────────────────────────────
def main():
    pca = joblib.load(f"{DATA_DIR}/pca_512.joblib")

    tasks = {
        "dc50": {"X": np.load(f"{DATA_DIR}/X_dc50_pca.npy"),
                 "y": np.load(f"{DATA_DIR}/y_dc50.npy"),
                 "meta": pd.read_csv(f"{DATA_DIR}/metadata_dc50.csv")},
        "dmax": {"X": np.load(f"{DATA_DIR}/X_dmax_pca.npy"),
                 "y": np.load(f"{DATA_DIR}/y_dmax.npy"),
                 "meta": pd.read_csv(f"{DATA_DIR}/metadata_dmax.csv")},
    }

    all_rows = []
    for task, D in tasks.items():
        X, y, meta = D['X'], D['y'], D['meta']
        sc = (float(y.mean()), float(y.std()))
        y_z = ((y - sc[0]) / sc[1]).astype(np.float32)
        print(f"\n{'='*60}\nTASK: {task}  (n={len(X)}, y_mean={sc[0]:.3f}, y_std={sc[1]:.3f})\n{'='*60}")

        for split_name, split_fn in SPLITS.items():
            tr, va, te, desc = split_fn(meta)
            print(f"\n  [{task}/{split_name}] {desc} | tr={len(tr)} va={len(va)} te={len(te)}")

            def mk(idx, shuffle=False):
                ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                                   torch.tensor(y_z[idx], dtype=torch.float32))
                return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

            torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
            model = CAT(input_dim=X.shape[2]).to(DEVICE)
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model, bvl, el = train_model(model, mk(tr, True), mk(va))
            ev = evaluate(model, mk(te), sc)
            print(f"    TEST R2={ev['R2']:.3f} RMSE={ev['RMSE']:.3f} MAE={ev['MAE']:.3f} ρ={ev['rho']:.3f} ({n_p:,} params, {el:.0f}s)")

            fa = frozen_analysis(model, X, y_z, sc)
            print(f"    attn W/L/E = {fa['attn']['W']:.3f}/{fa['attn']['L']:.3f}/{fa['attn']['E']:.3f} | internal r={fa['internal_r']:.3f}")

            ev_ext = external_validation(model, pca, sc, task)
            print(f"    EXTERNAL: n={ev_ext['n']} r={ev_ext['r']} ρ={ev_ext['rho']}")

            # Save frozen model
            torch.save(model.state_dict(), f"{OUT_DIR}/model_{task}_{split_name}.pt")

            all_rows.append({
                "task": task, "split": split_name, "desc": desc,
                "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_params": n_p,
                "R2": ev['R2'], "RMSE": ev['RMSE'], "MAE": ev['MAE'], "rho": ev['rho'],
                "attn_W": fa['attn']['W'], "attn_L": fa['attn']['L'], "attn_E": fa['attn']['E'],
                "internal_r": fa['internal_r'],
                "ext_n": ev_ext['n'], "ext_r": ev_ext['r'], "ext_rho": ev_ext['rho'],
                "train_time_s": el,
            })

    summ = pd.DataFrame(all_rows)
    summ.to_csv(f"{OUT_DIR}/generalization_summary_v2.csv", index=False)
    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    cols = ['task', 'split', 'n_test', 'R2', 'RMSE', 'rho', 'ext_n', 'ext_r', 'ext_rho']
    print(summ[cols].to_string(index=False))
    print(f"\nSaved: {OUT_DIR}/generalization_summary_v2.csv")

if __name__ == '__main__':
    main()
