#!/usr/bin/env python3
"""
REPRODUCE ORIGINAL SYNGLUE TRANSFORMER — same model, same values as previously reported
=======================================================================================
Rebuilds the ORIGINAL MultiTaskProtacModel (NO positional encoding, n_heads=4) exactly as
defined in SynGlue_Py/module_4.py (lines 395-414), trains it on the same data pipeline and
hyperparameters used for the previously reported generalization results, and verifies the
reported values:

    Previously reported (benchmarking/generalization_results.csv, 2026-07-14):
      Random 80/20 split   : DC50 R² = 0.342 | Dmax R² = 0.178
      Scaffold-held-out    : DC50 R² = 0.207 | Dmax R² = 0.088
      E3-held-out (VHL)    : DC50 R² = -0.153 | Dmax R² = -0.256
      Target-held-out      : DC50 R² = -0.639 | Dmax R² = -1.179

Pipeline (exact match with original):
  Stage 1: Train Transformer (n_heads=4, 2 layers, d=512, FF=2048, dropout=0.1) as feature
           extractor -> fused 512-dim embeddings via attention pooling
  Stage 2: Random Forest (500 trees, max_depth=20) on fused embeddings -> predictions

Config: BATCH_SIZE=64, LR=1e-4, EPOCHS=25, PATIENCE=8, RANDOM_STATE=42, N_TRIALS=2
Loss: MSE (sum of DC50 + Dmax terms on finite targets), AdamW, grad clip 1.0,
      ReduceLROnPlateau(patience=5, factor=0.5)
Data: server_53/Projects/Synglue/Attention/data (Upadted_PROTAC_features.csv + GROVER CSVs)
      DC50 -> log10(nM); Dmax (%) as-is; median DC50 / max Dmax per compound.

Usage:
    python reproduce_original_model.py                 # random 80/20 split only (fast)
    python reproduce_original_model.py --all_splits    # all 4 generalization splits
    python reproduce_original_model.py --use_tensors   # use prebuilt X.npy/y_*.npy instead
Outputs:
    benchmarking/reproduced_original_results.csv
    benchmarking/reproduced_original_detailed.csv
    models/reproduced_multitask_transformer_trialN.pt
"""

import os, sys, time, warnings, argparse, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ─── Paths ───────────────────────────────────────────
REV_DIR   = os.path.dirname(os.path.abspath(__file__))
BM_DIR    = os.path.join(REV_DIR, "benchmarking")
FIG_DIR   = os.path.join(REV_DIR, "figures")
MODEL_DIR = os.path.join(REV_DIR, "models")
DATA_DIR  = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
os.makedirs(BM_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  Torch threads: {torch.get_num_threads()}\n")

# ─── Config (EXACT match to reported runs) ──────────
BATCH_SIZE   = 64
LR           = 1e-4
EPOCHS       = 25
PATIENCE     = 8
RANDOM_STATE = 42
N_TRIALS     = 2
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH    = 20

# ─── 1. ORIGINAL MODEL CLASS (module_4.py, NO pos_embed) ───
class MultiTaskProtacModel(nn.Module):
    """Exact copy of SynGlue_Py/module_4.py lines 395-414 (original, pre-WP1)."""
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn_scores = self.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1)
        return self.head_dc50(fused), self.head_dmax(fused), attn_weights

    def extract_embeddings(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn_weights = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn_weights).sum(dim=1)


# ─── 2. DATA PIPELINE (fast dict-based, EXACT original semantics) ───
def load_data():
    """Matches WP1_generalization_test.py semantics exactly:
    - component matching: warhead (Uniprot→Target fallback), linker ID, E3 name
    - labels aggregated per PROTAC_Compound_ID: DC50 = median, Dmax = max
    - samples WITHOUT labels are KEPT (NaN, masked in loss)
    - y_dc50 = log10(nM) where raw>0 & finite, else NaN
    Uses dictionary lookups instead of per-row dataframe filters (fast path).
    """
    t0 = time.time()

    protac = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False,
                         usecols=lambda c: not c.startswith("Grover_"))
    for c in ["warhead_Uniprot", "warhead_Target", "E3 ligase", "Linker ID", "PROTAC_Compound_ID"]:
        if c in protac.columns:
            protac[c] = protac[c].astype(str).str.strip().str.upper()

    warhead = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    warhead["warhead_Uniprot"] = warhead["warhead_Uniprot"].astype(str).str.strip().str.upper()
    warhead["warhead_Target"]  = warhead["warhead_Target"].astype(str).str.strip().str.upper()
    w_g = [c for c in warhead.columns if c.startswith("Grover_")]
    w_by_uniprot, w_by_target = {}, {}
    for _, r in warhead.iterrows():
        vec = r[w_g].values.astype(np.float32)
        if r["warhead_Uniprot"] not in ("NAN", "", "NONE"): w_by_uniprot[r["warhead_Uniprot"]] = vec
        if r["warhead_Target"] not in ("NAN", "", "NONE"):  w_by_target[r["warhead_Target"]] = vec

    linker = pd.read_csv(f"{DATA_DIR}/linker_updated_features.csv", low_memory=False)
    linker["linker_Compound_ID"] = linker["linker_Compound_ID"].astype(str).str.strip().str.upper()
    l_g = [c for c in linker.columns if c.startswith("Grover_")]
    l_by_id = {r["linker_Compound_ID"]: r[l_g].values.astype(np.float32) for _, r in linker.iterrows()}

    e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()
    e_g = [c for c in e3_df.columns if c.startswith("Grover_")]
    e_by_name = {r["E3_Ligase"]: r[e_g].values.astype(np.float32) for _, r in e3_df.iterrows()}

    assert len(w_g) == len(l_g) == len(e_g), f"GROVER dim mismatch: {len(w_g)}/{len(l_g)}/{len(e_g)}"

    # Per-compound label aggregation (median DC50 / max Dmax) — exact original logic
    label_df = protac[["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]].copy()
    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"]  = pd.to_numeric(label_df["Dmax (%)"],  errors="coerce")
    labels = (label_df.groupby("PROTAC_Compound_ID")
                      .agg({"DC50 (nM)": "median", "Dmax (%)": "max"})
                      .reset_index())
    label_map = {r["PROTAC_Compound_ID"]: (r["DC50 (nM)"], r["Dmax (%)"])
                 for _, r in labels.iterrows()}

    rows = []
    for _, p in protac.iterrows():
        w_vec = w_by_uniprot.get(p["warhead_Uniprot"])
        if w_vec is None:
            w_vec = w_by_target.get(p["warhead_Target"])
        if w_vec is None: continue
        l_vec = l_by_id.get(p["Linker ID"])
        if l_vec is None: continue
        e_vec = e_by_name.get(p["E3 ligase"])
        if e_vec is None: continue
        dc50_raw, dmax_raw = label_map.get(p["PROTAC_Compound_ID"], (np.nan, np.nan))
        rows.append((w_vec, l_vec, e_vec, dc50_raw, dmax_raw,
                     str(p.get("warhead_Target", "")), str(p.get("E3 ligase", "")),
                     str(p.get("PROTAC_SMILES", "")), str(p.get("PROTAC_Compound_ID", ""))))

    n = len(rows)
    X = np.stack([np.stack([r[0], r[1], r[2]]) for r in rows]).astype(np.float32)
    dc50_raw = np.array([r[3] for r in rows], dtype=float)
    y_dc50 = np.where((dc50_raw > 0) & (~np.isnan(dc50_raw)), np.log10(dc50_raw), np.nan)
    y_dmax = np.array([r[4] for r in rows], dtype=float)
    meta = pd.DataFrame({"PROTAC_Compound_ID": [r[8] for r in rows],
                         "Target": [r[5] for r in rows],
                         "E3_Ligase": [r[6] for r in rows],
                         "SMILES": [r[7] for r in rows]})
    print(f"  Loaded {n} matched rows in {time.time()-t0:.1f}s")
    print(f"  X: {X.shape}  DC50 valid: {np.isfinite(y_dc50).sum()}  Dmax valid: {np.isfinite(y_dmax).sum()}")
    return X, y_dc50, y_dmax, meta


# ─── 3. TRAINING (identical to original) ────────────
def train_transformer(model, tr_loader, va_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val, best_state, counter = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for xb, y50, ym in tr_loader:
            xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
            p50, pm, _ = model(xb)
            loss = 0
            m50 = torch.isfinite(y50)
            if m50.any(): loss += loss_fn(p50[m50], y50[m50])
            mm = torch.isfinite(ym)
            if mm.any(): loss += loss_fn(pm[mm], ym[mm])
            if loss == 0: continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, y50, ym in va_loader:
                xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
                p50, pm, _ = model(xb)
                loss = 0
                m50 = torch.isfinite(y50)
                if m50.any(): loss += loss_fn(p50[m50], y50[m50])
                mm = torch.isfinite(ym)
                if mm.any(): loss += loss_fn(pm[mm], ym[mm])
                vl += loss
        vl /= len(va_loader)
        sched.step(vl)
        if vl < best_val:
            best_val, best_state, counter = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, time.time() - t0


def extract_embeddings(model, loader):
    model.eval()
    Z, Y50, Ym = [], [], []
    with torch.no_grad():
        for xb, y50, ym in loader:
            xb = xb.to(DEVICE)
            fused = model.extract_embeddings(xb)
            Z.append(fused.cpu().numpy()); Y50.append(y50.numpy()); Ym.append(ym.numpy())
    return np.vstack(Z), np.hstack(Y50), np.hstack(Ym)


def train_rf(Z_tr, y_tr):
    mask = np.isfinite(y_tr)
    if mask.sum() == 0: return None, 0
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                               n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    rf.fit(Z_tr[mask], y_tr[mask])
    return rf, mask.sum()


def evaluate_rf(rf, Z_te, y_te):
    mask = np.isfinite(y_te)
    if mask.sum() == 0 or rf is None:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "n": 0}
    pred = rf.predict(Z_te[mask])
    return {
        "R2": r2_score(y_te[mask], pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te[mask], pred))),
        "MAE": mean_absolute_error(y_te[mask], pred),
        "n": int(mask.sum()),
    }


def make_loaders(X, y50, ym, idx_tr, idx_va):
    Xt = torch.tensor(X[idx_tr], dtype=torch.float32)
    Xv = torch.tensor(X[idx_va], dtype=torch.float32)
    tr = TensorDataset(Xt, torch.tensor(y50[idx_tr], dtype=torch.float32),
                       torch.tensor(ym[idx_tr], dtype=torch.float32))
    va = TensorDataset(Xv, torch.tensor(y50[idx_va], dtype=torch.float32),
                       torch.tensor(ym[idx_va], dtype=torch.float32))
    return DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True), DataLoader(va, batch_size=BATCH_SIZE)


# ─── 4. SPLIT STRATEGIES ────────────────────────────
def split_random(X, y50, ym, meta):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    tr, va = train_test_split(tr, test_size=0.125, random_state=RANDOM_STATE, shuffle=True)  # 80/10/10
    return tr, va, te


def split_scaffold(X, y50, ym, meta):
    smis = meta["SMILES"].values
    scaffolds = []
    for s in smis:
        try:
            m = Chem.MolFromSmiles(s)
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
        except Exception:
            sc = None
        scaffolds.append(sc)
    uniq = sorted({s for s in scaffolds if s})
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(uniq)
    n_test = max(1, int(0.2 * len(uniq)))
    test_scaffolds, rest = set(uniq[:n_test]), uniq[n_test:]
    te  = np.array([i for i, s in enumerate(scaffolds) if s in test_scaffolds])
    rem = np.array([i for i, s in enumerate(scaffolds) if s not in test_scaffolds])
    rng2 = np.random.RandomState(RANDOM_STATE)
    va_idx = rng2.choice(len(rem), size=int(0.125 * len(rem)), replace=False)
    va = rem[va_idx]; tr = np.delete(rem, va_idx)
    return tr, va, te


def split_e3(X, y50, ym, meta):
    e3 = meta["E3_Ligase"].values
    test_e3 = "VHL"
    te  = np.array([i for i, e in enumerate(e3) if e == test_e3])
    rem = np.array([i for i, e in enumerate(e3) if e != test_e3])
    rng = np.random.RandomState(RANDOM_STATE)
    va_idx = rng.choice(len(rem), size=int(0.125 * len(rem)), replace=False)
    va = rem[va_idx]; tr = np.delete(rem, va_idx)
    return tr, va, te


def split_target(X, y50, ym, meta):
    tg = meta["Target"].values
    uniq = sorted({str(t) for t in tg})
    counts = pd.Series([str(t) for t in tg]).value_counts()
    rare = [u for u in uniq if counts[u] <= 15]
    test_tg = set(rare[: max(1, len(rare) // 2)])
    te  = np.array([i for i, t in enumerate(tg) if str(t) in test_tg])
    rem = np.array([i for i, t in enumerate(tg) if str(t) not in test_tg])
    rng = np.random.RandomState(RANDOM_STATE)
    va_idx = rng.choice(len(rem), size=int(0.125 * len(rem)), replace=False)
    va = rem[va_idx]; tr = np.delete(rem, va_idx)
    return tr, va, te


SPLITS = {
    "Random 80/20": split_random,
    "Scaffold-held-out": split_scaffold,
    "E3-held-out (VHL)": split_e3,
    "Target-held-out": split_target,
}

REPORTED = {  # from benchmarking/generalization_results.csv (2026-07-14)
    "Random 80/20":          {"dc50_R2": 0.342, "dmax_R2": 0.178},
    "Scaffold-held-out":     {"dc50_R2": 0.207, "dmax_R2": 0.088},
    "E3-held-out (VHL)":     {"dc50_R2": -0.153, "dmax_R2": -0.256},
    "Target-held-out":       {"dc50_R2": -0.639, "dmax_R2": -1.179},
}


def run_split(name, X, y50, ym, meta, split_fn, trials=N_TRIALS):
    print(f"\n=== Split: {name} ===")
    rows = []
    for t in range(trials):
        tr_i, va_i, te_i = split_fn(X, y50, ym, meta)
        tr_loader, va_loader = make_loaders(X, y50, ym, tr_i, va_i)
        model = MultiTaskProtacModel(input_dim=X.shape[2]).to(DEVICE)
        print(f"  Trial {t+1}/{trials}: train={len(tr_i)} val={len(va_i)} test={len(te_i)} "
              f"params={sum(p.numel() for p in model.parameters()):,}")
        model, dt = train_transformer(model, tr_loader, va_loader)

        # Save checkpoint (original architecture, best val state)
        ckpt_path = os.path.join(MODEL_DIR, f"reproduced_multitask_transformer_trial{t+1}.pt")
        torch.save({"state_dict": model.state_dict(),
                    "split": name, "trial": t + 1,
                    "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
                    "architecture": "MultiTaskProtacModel(input_dim=4800, hidden_dim=512, n_heads=4), "
                                    "2 layers, FF=2048, attn-pool, dual heads, NO pos_embed",
                    "train_time_s": dt}, ckpt_path)

        te_loader = DataLoader(TensorDataset(
            torch.tensor(X[te_i], dtype=torch.float32),
            torch.tensor(y50[te_i], dtype=torch.float32),
            torch.tensor(ym[te_i], dtype=torch.float32)), batch_size=BATCH_SIZE)
        Ztr, y50_tr, ym_tr = extract_embeddings(model, tr_loader)
        Zte, y50_te, ym_te = extract_embeddings(model, te_loader)

        rf50, _ = train_rf(Ztr, y50_tr)
        rfmax, _ = train_rf(Ztr, ym_tr)
        r50 = evaluate_rf(rf50, Zte, y50_te)
        rmax = evaluate_rf(rfmax, Zte, ym_te)
        rows.append({"split": name, "trial": t + 1,
                     "dc50_R2": r50["R2"], "dc50_RMSE": r50["RMSE"], "dc50_MAE": r50["MAE"], "dc50_n": r50["n"],
                     "dmax_R2": rmax["R2"], "dmax_RMSE": rmax["RMSE"], "dmax_MAE": rmax["MAE"], "dmax_n": rmax["n"]})
        print(f"    -> DC50 R²={r50['R2']:.4f}  Dmax R²={rmax['R2']:.4f}")
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    agg = df.groupby("split").agg(
        dc50_R2=("dc50_R2", "mean"), dc50_R2_std=("dc50_R2", "std"),
        dc50_RMSE=("dc50_RMSE", "mean"), dc50_MAE=("dc50_MAE", "mean"),
        dmax_R2=("dmax_R2", "mean"), dmax_R2_std=("dmax_R2", "std"),
        dmax_RMSE=("dmax_RMSE", "mean"), dmax_MAE=("dmax_MAE", "mean"),
        dc50_n=("dc50_n", "mean"), dmax_n=("dmax_n", "mean")).reset_index()
    rep = REPORTED.get(name, {})
    agg["reported_dc50_R2"] = rep.get("dc50_R2", np.nan)
    agg["reported_dmax_R2"] = rep.get("dmax_R2", np.nan)
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_splits", action="store_true", help="run all 4 generalization splits")
    ap.add_argument("--use_tensors", action="store_true", help="use prebuilt benchmarking/X.npy tensors (random split only)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print("Loading data...")
    X, y50, ym, meta = load_data()

    splits = SPLITS if args.all_splits else {"Random 80/20": SPLITS["Random 80/20"]}
    det, aggs = [], []
    for name, fn in splits.items():
        d, a = run_split(name, X, y50, ym, meta, fn)
        det.append(d); aggs.append(a)
    det_df = pd.concat(det, ignore_index=True)
    agg_df = pd.concat(aggs, ignore_index=True)

    det_df.to_csv(os.path.join(BM_DIR, "reproduced_original_detailed.csv"), index=False)
    agg_df.to_csv(os.path.join(BM_DIR, "reproduced_original_results.csv"), index=False)

    print("\n================ REPRODUCED vs REPORTED ================")
    print(f"{'Split':<22}{'Rep DC50 R2':>12}{'Rep Dmax R2':>12}   {'Rpt DC50 R2':>12}{'Rpt Dmax R2':>12}")
    for _, r in agg_df.iterrows():
        print(f"{r['split']:<22}{r['dc50_R2']:>12.3f}{r['dmax_R2']:>12.3f}   "
              f"{r['reported_dc50_R2']:>12.3f}{r['reported_dmax_R2']:>12.3f}")
    print("=========================================================")


if __name__ == "__main__":
    main()
