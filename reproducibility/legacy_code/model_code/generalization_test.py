#!/usr/bin/env python3
"""
WP1: Transformer + RF Two-Stage Pipeline + Rigorous Generalization Testing
===========================================================================
Matches the original SynGlue pipeline exactly:
  Stage 1: Train Transformer (n_heads=4) as feature extractor → 512-dim fused embeddings
  Stage 2: Train Random Forest (500 trees, max_depth=20) on fused embeddings → predictions

Four generalization strategies:
  - Random 80/20 split (baseline — should reproduce R² ~0.90/0.87)
  - Target-held-out
  - E3-held-out
  - Scaffold-held-out

Outputs:
  - benchmarking/generalization_results.csv
  - benchmarking/generalization_detailed.csv
  - figures/generalization_comparison.png
"""

import os, sys, time, warnings, gc, json
warnings.filterwarnings("ignore")
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ─── Paths ───────────────────────────────────────────
DATA_DIR   = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
REV_DIR    = "/storage/savi/saveenas/Projects/SynGlue/revision"
BM_DIR     = os.path.join(REV_DIR, "benchmarking")
FIG_DIR    = os.path.join(REV_DIR, "figures")
os.makedirs(BM_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# ─── Config ──────────────────────────────────────────
BATCH_SIZE   = 64
LR           = 1e-4
EPOCHS       = 25
PATIENCE     = 8
RANDOM_STATE = 42
N_TRIALS     = 2  # per split strategy

RF_N_ESTIMATORS = 500
RF_MAX_DEPTH    = 20

# ─── 1. DATA PIPELINE (exact match with original notebook) ───
def load_data():
    def load(p):
        return pd.read_csv(p, low_memory=False)
    
    protac_df  = load(f"{DATA_DIR}/Upadted_PROTAC_features.csv")
    warhead_df = load(f"{DATA_DIR}/Updated_warhead.csv")
    linker_df  = load(f"{DATA_DIR}/linker_updated_features.csv")
    e3_df      = load(f"{DATA_DIR}/E3_updated_features.csv")
    
    # Normalize join keys
    def norm(df, cols):
        for c in cols:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
        return df
    protac_df  = norm(protac_df,  ["warhead_Uniprot", "warhead_Target", "E3 ligase"])
    warhead_df = norm(warhead_df, ["warhead_Uniprot", "warhead_Target"])
    e3_df      = norm(e3_df,      ["E3_Ligase"])
    
    # Prefix GROVER columns for disambiguation
    warhead_df = warhead_df.rename(columns={c: f"W_{c}" for c in warhead_df.columns if c.startswith("Grover_")})
    linker_df  = linker_df.rename(columns={c: f"L_{c}" for c in linker_df.columns if c.startswith("Grover_")})
    e3_df      = e3_df.rename(columns={c: f"E_{c}" for c in e3_df.columns if c.startswith("Grover_")})
    
    # Build component map
    rows = []
    for _, p in protac_df.iterrows():
        wh = warhead_df[(warhead_df["warhead_Uniprot"] == p["warhead_Uniprot"]) | 
                        (warhead_df["warhead_Target"] == p["warhead_Target"])]
        if wh.empty: continue
        lk = linker_df[linker_df["linker_Compound_ID"] == p["Linker ID"]]
        if lk.empty: continue
        e3 = e3_df[e3_df["E3_Ligase"] == p["E3 ligase"]]
        if e3.empty: continue
        rows.append({
            "PROTAC_Compound_ID": p["PROTAC_Compound_ID"],
            "warhead_Compound_ID": wh.iloc[0]["warhead_Compound_ID"],
            "linker_Compound_ID": p["Linker ID"],
            "E3_Compound_ID": e3.iloc[0]["E3_Compound_ID"],
            "Target": str(p.get("warhead_Target", "")),
            "E3_Ligase": str(p.get("E3 ligase", "")),
            "SMILES": str(p.get("PROTAC_SMILES", "")),
        })
    
    protac_map = pd.DataFrame(rows)
    
    # Merge GROVER features
    w_g = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_g = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_g = [c for c in e3_df.columns if c.startswith("E_Grover_")]
    
    protac_features = (
        protac_map
        .merge(warhead_df[["warhead_Compound_ID"] + w_g], on="warhead_Compound_ID", how="inner")
        .merge(linker_df[["linker_Compound_ID"] + l_g],   on="linker_Compound_ID", how="inner")
        .merge(e3_df[["E3_Compound_ID"] + e_g],           on="E3_Compound_ID", how="inner")
    )
    
    # Build X tensor
    X = np.stack([
        np.stack([
            r[w_g].values.astype(np.float32),
            r[l_g].values.astype(np.float32),
            r[e_g].values.astype(np.float32),
        ])
        for _, r in protac_features.iterrows()
    ])
    
    # Labels (exact original logic)
    label_df = protac_df[["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]].copy()
    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"]  = pd.to_numeric(label_df["Dmax (%)"],  errors="coerce")
    
    protac_labels = (
        label_df
        .groupby("PROTAC_Compound_ID")
        .agg({"DC50 (nM)": "median", "Dmax (%)": "max"})
        .reset_index()
    )
    
    final_df = protac_features.merge(protac_labels, on="PROTAC_Compound_ID", how="left")
    
    # Extract y arrays
    dc50_raw = final_df["DC50 (nM)"].values.astype(float)
    y_dc50   = np.where((dc50_raw > 0) & (~np.isnan(dc50_raw)), np.log10(dc50_raw), np.nan)
    y_dmax   = final_df["Dmax (%)"].values.astype(float)
    
    meta = final_df[["PROTAC_Compound_ID", "Target", "E3_Ligase", "SMILES"]].copy()
    
    print(f"  X: {X.shape} ({X.shape[0]} samples, {X.shape[2]} GROVER dim)")
    print(f"  DC50 valid: {np.isfinite(y_dc50).sum()}, Dmax valid: {np.isfinite(y_dmax).sum()}")
    print(f"  Targets: {meta['Target'].nunique()}, E3 ligases: {meta['E3_Ligase'].nunique()}")
    
    return X, y_dc50, y_dmax, meta


# ─── 2. TRANSFORMER MODEL (n_heads=4 to match original) ───
class MultiTaskProtacModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        # ★ Positional Encoding
        self.pos_embed = nn.Embedding(3, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1,
                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * attn).sum(dim=1)
        return self.head_dc50(fused).squeeze(-1), self.head_dmax(fused).squeeze(-1), fused

    def extract_embeddings(self, x):
        """Return fused 512-dim embeddings (for Stage 2 RF)."""
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn).sum(dim=1)


# ─── 3. STAGE 1: TRAIN TRANSFORMER ──────────────────
def train_transformer(model, tr_loader, va_loader):
    """Train transformer as feature extractor (multi-task)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    
    best_val = float("inf")
    best_state = None
    counter = 0
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        tl = 0
        for xb, y50, ym in tr_loader:
            xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
            p50, pm, _ = model(xb)
            # Only compute loss on valid targets
            loss = 0
            m50 = torch.isfinite(y50)
            if m50.any():
                loss += loss_fn(p50[m50], y50[m50])
            mm = torch.isfinite(ym)
            if mm.any():
                loss += loss_fn(pm[mm], ym[mm])
            if loss == 0: continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        
        model.eval()
        vl = 0
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
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                break
    
    model.load_state_dict(best_state)
    return model, time.time() - t0


# ─── 4. STAGE 2: EXTRACT EMBEDDINGS + TRAIN RF ─────
def extract_embeddings(model, loader):
    """Extract 512-dim fused embeddings from the transformer."""
    model.eval()
    Z, Y50, Ym = [], [], []
    with torch.no_grad():
        for xb, y50, ym in loader:
            xb = xb.to(DEVICE)
            fused = model.extract_embeddings(xb)
            Z.append(fused.cpu().numpy())
            Y50.append(y50.numpy())
            Ym.append(ym.numpy())
    return np.vstack(Z), np.hstack(Y50), np.hstack(Ym)


def train_rf(Z_tr, y_tr):
    """Train Random Forest on fused embeddings."""
    mask = np.isfinite(y_tr)
    if mask.sum() == 0:
        return None, 0
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    rf.fit(Z_tr[mask], y_tr[mask])
    return rf, mask.sum()


def evaluate_rf(rf, Z_te, y_te):
    """Evaluate RF predictions vs ground truth."""
    mask = np.isfinite(y_te)
    if mask.sum() == 0 or rf is None:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "n": 0}
    pred = rf.predict(Z_te[mask])
    return {
        "R2":   float(r2_score(y_te[mask], pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te[mask], pred))),
        "MAE":  float(mean_absolute_error(y_te[mask], pred)),
        "n":    int(mask.sum()),
    }


# ─── 5. SPLIT STRATEGIES ────────────────────────────
def random_split(X, y50, ym, meta):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
    return tr, va, te

def target_held_out_split(X, y50, ym, meta):
    targets = meta["Target"].values
    unique, counts = np.unique(targets, return_counts=True)
    sorted_idx = np.argsort(counts)
    n_hold = max(1, len(unique) // 7)
    held = set(unique[sorted_idx[:n_hold]])
    te_mask = np.array([t in held for t in targets])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    print(f"  (held-out {n_hold} targets, {len(te)} test samples)")
    return tr, va, te

def e3_held_out_split(X, y50, ym, meta):
    e3s = meta["E3_Ligase"].values
    unique, counts = np.unique(e3s, return_counts=True)
    sorted_idx = np.argsort(-counts)
    if len(sorted_idx) >= 2:
        held = {unique[sorted_idx[1]]}
    else:
        held = {unique[sorted_idx[0]]}
    te_mask = np.array([e in held for e in e3s])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
    print(f"  (held-out {held}, {len(te)} test samples)")
    return tr, va, te

def scaffold_held_out_split(X, y50, ym, meta):
    smiles = meta["SMILES"].values
    valid_idx, scaffolds = [], []
    for i, smi in enumerate(smiles):
        if pd.isna(smi) or str(smi).strip() in ("", "nan"): continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None: continue
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if sc and sc != "": scaffolds.append(sc); valid_idx.append(i)
    if len(scaffolds) < 5: return random_split(X, y50, ym, meta)
    sc2idx = defaultdict(list)
    for sc, i in zip(scaffolds, valid_idx): sc2idx[sc].append(i)
    uniq_sc = list(sc2idx.keys())
    rng = np.random.RandomState(RANDOM_STATE); rng.shuffle(uniq_sc)
    n_test = max(1, int(len(uniq_sc) * 0.2))
    test_sc = set(uniq_sc[:n_test])
    te_idx = [i for sc in test_sc for i in sc2idx[sc]]
    trva_idx = [i for sc in uniq_sc[n_test:] for i in sc2idx[sc]]
    tr, va = train_test_split(trva_idx, test_size=0.15, random_state=RANDOM_STATE)
    print(f"  (held-out {n_test} scaffolds, {len(te_idx)} test samples)")
    return np.array(tr), np.array(va), np.array(te_idx)


# ─── 6. FULL PIPELINE PER SPLIT ──────────────────────
def run_split_pipeline(X, y50, ym, meta, name, split_fn):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    
    trial_results_dc50 = []
    trial_results_dmax = []
    
    for t in range(N_TRIALS):
        print(f"\n  Trial {t+1}/{N_TRIALS}")
        torch.manual_seed(RANDOM_STATE + t*10)
        np.random.seed(RANDOM_STATE + t*10)
        
        # Split
        tr_i, va_i, te_i = split_fn(X, y50, ym, meta)
        
        def dl(idx, shuffle=False):
            ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                               torch.tensor(y50[idx], dtype=torch.float32),
                               torch.tensor(ym[idx], dtype=torch.float32))
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
        
        tr_loader = dl(tr_i, True)
        va_loader = dl(va_i)
        te_loader = dl(te_i)
        
        # Stage 1: Train Transformer
        model = MultiTaskProtacModel(input_dim=X.shape[2]).to(DEVICE)
        t1 = time.time()
        model, elapsed_tr = train_transformer(model, tr_loader, va_loader)
        t2 = time.time()
        
        # Stage 2: Extract embeddings
        Z_tr, y50_tr, ym_tr = extract_embeddings(model, tr_loader)
        Z_te, y50_te, ym_te = extract_embeddings(model, te_loader)
        
        # Stage 3: Train RF
        rf_dc50, n50 = train_rf(Z_tr, y50_tr)
        rf_dmax, nm  = train_rf(Z_tr, ym_tr)
        
        # Stage 4: Evaluate
        res50 = evaluate_rf(rf_dc50, Z_te, y50_te)
        resm  = evaluate_rf(rf_dmax, Z_te, ym_te)
        
        print(f"    Transformer: {elapsed_tr:.0f}s | DC50 RF R²={res50['R2']:.4f} (n={res50['n']}) | Dmax RF R²={resm['R2']:.4f} (n={resm['n']})", flush=True)
        
        trial_results_dc50.append(res50)
        trial_results_dmax.append(resm)
        
        del model; torch.cuda.empty_cache(); gc.collect()
    
    # Aggregate
    def avg_std(results, key):
        vals = [r[key] for r in results if np.isfinite(r[key])]
        if not vals: return 0, 0
        return float(np.mean(vals)), float(np.std(vals))
    
    dc50_r2_m, dc50_r2_s = avg_std(trial_results_dc50, "R2")
    dc50_rmse_m, dc50_rmse_s = avg_std(trial_results_dc50, "RMSE")
    dc50_mae_m, dc50_mae_s = avg_std(trial_results_dc50, "MAE")
    dmax_r2_m, dmax_r2_s = avg_std(trial_results_dmax, "R2")
    dmax_rmse_m, dmax_rmse_s = avg_std(trial_results_dmax, "RMSE")
    dmax_mae_m, dmax_mae_s = avg_std(trial_results_dmax, "MAE")
    
    return {
        "split": name,
        "dc50_R2_mean": dc50_r2_m, "dc50_R2_std": dc50_r2_s,
        "dc50_RMSE_mean": dc50_rmse_m, "dc50_RMSE_std": dc50_rmse_s,
        "dc50_MAE_mean": dc50_mae_m, "dc50_MAE_std": dc50_mae_s,
        "dmax_R2_mean": dmax_r2_m, "dmax_R2_std": dmax_r2_s,
        "dmax_RMSE_mean": dmax_rmse_m, "dmax_RMSE_std": dmax_rmse_s,
        "dmax_MAE_mean": dmax_mae_m, "dmax_MAE_std": dmax_mae_s,
        "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
    }, {"dc50": trial_results_dc50, "dmax": trial_results_dmax}


# ─── 7. MAIN ─────────────────────────────────────────
def main():
    print("╔" + "═"*54 + "╗")
    print("║  WP1: Two-Stage Transformer+RF + Generalization Testing  ║")
    print("╚" + "═"*54 + "╝\n")
    
    t0 = time.time()
    
    # Load data
    X, y50, ym, meta = load_data()
    print(f"  Data loaded in {time.time()-t0:.0f}s")
    
    strategies = [
        ("Random 80/20 split", random_split),
        ("Target-held-out", target_held_out_split),
        ("E3-held-out", e3_held_out_split),
        ("Scaffold-held-out", scaffold_held_out_split),
    ]
    
    all_results = []
    all_detailed = {}
    
    for name, fn in strategies:
        res, det = run_split_pipeline(X, y50, ym, meta, name, fn)
        all_results.append(res)
        all_detailed[name] = det
    
    # Results table
    df = pd.DataFrame(all_results)
    cols = ["split", "n_train", "n_val", "n_test",
            "dc50_R2_mean", "dc50_R2_std", "dc50_RMSE_mean", "dc50_RMSE_std",
            "dmax_R2_mean", "dmax_R2_std", "dmax_RMSE_mean", "dmax_RMSE_std"]
    
    print("\n\n" + "="*70)
    print("  GENERALIZATION RESULTS — Two-Stage Transformer + Random Forest")
    print("="*70)
    print()
    print(df[cols].to_string(index=False))
    
    df.to_csv(os.path.join(BM_DIR, "generalization_results.csv"), index=False)
    
    # Detailed per-trial
    rows = []
    for name, det in all_detailed.items():
        for t, (r50, rm) in enumerate(zip(det["dc50"], det["dmax"])):
            rows.append({
                "split": name, "trial": t+1,
                "dc50_R2": r50["R2"], "dc50_RMSE": r50["RMSE"], "dc50_MAE": r50["MAE"], "dc50_n": r50["n"],
                "dmax_R2": rm["R2"],  "dmax_RMSE": rm["RMSE"],  "dmax_MAE": rm["MAE"],  "dmax_n": rm["n"],
            })
    pd.DataFrame(rows).to_csv(os.path.join(BM_DIR, "generalization_detailed.csv"), index=False)
    
    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        splits = df["split"].values
        x = np.arange(len(splits))
        colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]
        labels_short = [s[:14]+".." if len(s)>14 else s for s in splits]
        
        for row, (target, title) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
            for col, metric in enumerate(["R2", "RMSE", "MAE"]):
                ax = axes[row, col]
                key = f"{target}_{metric}_mean"
                sk = f"{target}_{metric}_std"
                if key not in df.columns: continue
                vals = df[key].values; errs = df[sk].values
                bars = ax.bar(x, vals, 0.5, yerr=errs, capsize=5, color=colors)
                ax.set_xticks(x); ax.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=9)
                ax.set_title(f"{title} {metric}")
                if metric == "R2": ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
                ax.grid(alpha=0.3, axis="y")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(v)*0.02+0.005,
                            f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
        
        plt.suptitle("SynGlue (Transformer + RF): Generalization Across Split Strategies", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "generalization_comparison.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  ✅ Plot saved")
    except Exception as e:
        print(f"  ⚠ Plot error: {e}")
    
    print(f"\n✅ WP1 complete. Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
