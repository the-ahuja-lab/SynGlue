#!/usr/bin/env python3
"""
WP1a: Baseline vs Positional Encoding — Controlled Comparison
==============================================================
Compares two models on identical random splits:
  1. Baseline: Transformer without positional encoding (permutation-invariant)
  2. PosEmbed: Transformer with learnable 3×512 positional embeddings

Outputs:
  - revision/benchmarking/pos_embed_comparison.csv
  - revision/figures/pos_embed_comparison.png
"""

import os, sys, time, warnings, gc, json
warnings.filterwarnings("ignore")
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
REVISION_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision"
BM_DIR = os.path.join(REVISION_DIR, "benchmarking")
FIG_DIR = os.path.join(REVISION_DIR, "figures")
os.makedirs(BM_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

BATCH_SIZE, LR, EPOCHS, PATIENCE, RANDOM_STATE, N_TRIALS = 64, 1e-4, 30, 8, 42, 3

# ─── Data ────────────────────────────────────────────
def load_full_data(path):
    df = pd.read_csv(path, low_memory=False)
    if df.empty: raise ValueError(f"Empty: {path}")
    print(f"  Loaded {os.path.basename(path)} → {df.shape}")
    return df

def _parse(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.", "N/D", "NA", "N/A", ""): return np.nan
    if s.startswith(">"): s = s[1:]
    elif s.startswith("<"): s = s[1:]
    parts = s.split("/")
    vals = []
    for p in parts:
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", ""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

def build_dataset():
    protac_df  = load_full_data(f"{DATA_DIR}/Upadted_PROTAC_features.csv")
    warhead_df = load_full_data(f"{DATA_DIR}/Updated_warhead.csv")
    linker_df  = load_full_data(f"{DATA_DIR}/linker_updated_features.csv")
    e3_df      = load_full_data(f"{DATA_DIR}/E3_updated_features.csv")
    
    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    for c in ["E3 ligase"]:
        if c in protac_df.columns: protac_df[c] = protac_df[c].astype(str).str.strip().str.upper()
    for c in ["E3_Ligase"]:
        if c in e3_df.columns: e3_df[c] = e3_df[c].astype(str).str.strip().str.upper()
    
    def prefix_grover(df, prefix):
        return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df  = prefix_grover(linker_df, "L_")
    e3_df      = prefix_grover(e3_df, "E_")
    
    w_grover = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_grover = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_grover = [c for c in e3_df.columns if c.startswith("E_Grover_")]
    
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
            "W_vec": wh.iloc[0][w_grover].values.astype(np.float32),
            "L_vec": lk.iloc[0][l_grover].values.astype(np.float32),
            "E_vec": e3.iloc[0][e_grover].values.astype(np.float32),
            "DC50_raw": p.get("DC50 (nM)", np.nan),
            "Dmax_raw": p.get("Dmax (%)", np.nan),
        })
    
    protac_map = pd.DataFrame(rows)
    protac_map["DC50"] = protac_map["DC50_raw"].apply(_parse)
    protac_map["Dmax"] = protac_map["Dmax_raw"].apply(_parse)
    valid = protac_map.dropna(subset=["DC50", "Dmax"]).copy()
    
    X = np.stack([np.stack([r["W_vec"], r["L_vec"], r["E_vec"]]) for _, r in valid.iterrows()])
    y_dc50 = np.log10(np.clip(valid["DC50"].values, 1e-3, None)).astype(np.float32)
    y_dmax = valid["Dmax"].values.astype(np.float32)
    
    dc50_m, dc50_s = y_dc50.mean(), y_dc50.std()
    dmax_m, dmax_s = y_dmax.mean(), y_dmax.std()
    y_dc50_z = ((y_dc50 - dc50_m) / dc50_s).astype(np.float32)
    y_dmax_z = ((y_dmax - dmax_m) / dmax_s).astype(np.float32)
    scalers = {"dc50": (dc50_m, dc50_s), "dmax": (dmax_m, dmax_s)}
    
    print(f"  Dataset: {len(X)} samples, GROVER dim: {X.shape[2]}")
    print(f"  DC50: μ={dc50_m:.2f} σ={dc50_s:.2f} (log10 nM) | Dmax: μ={dmax_m:.1f} σ={dmax_s:.1f} (%)")
    return X, y_dc50_z, y_dmax_z, scalers

# ─── Models ──────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * attn).sum(dim=1)
        return self.head_dc50(fused).squeeze(-1), self.head_dmax(fused).squeeze(-1), attn

class PosEmbedModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(3, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * attn).sum(dim=1)
        return self.head_dc50(fused).squeeze(-1), self.head_dmax(fused).squeeze(-1), attn

def train_model(model, tr_loader, va_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val = float("inf"); best_state = None; counter = 0; t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for xb, y50, ym in tr_loader:
            xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
            p50, pm, _ = model(xb)
            loss = loss_fn(p50, y50) + loss_fn(pm, ym)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, y50, ym in va_loader:
                xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
                p50, pm, _ = model(xb)
                vl += (loss_fn(p50, y50) + loss_fn(pm, ym)).item()
        vl /= len(va_loader); sched.step(vl)
        if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}; counter = 0
        else: counter += 1
        if counter >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, best_val, time.time() - t0

def evaluate_rescaled(model, loader, scalers):
    dc50_m, dc50_s = scalers["dc50"]; dmax_m, dmax_s = scalers["dmax"]
    model.eval(); p50, t50, pm, tm = [], [], [], []
    with torch.no_grad():
        for xb, y50, ym in loader:
            xb = xb.to(DEVICE); pr50, prm, _ = model(xb)
            p50.extend((pr50.cpu().numpy() * dc50_s + dc50_m)); t50.extend((y50.numpy() * dc50_s + dc50_m))
            pm.extend((prm.cpu().numpy() * dmax_s + dmax_m));  tm.extend((ym.numpy() * dmax_s + dmax_m))
    p50, t50, pm, tm = [np.array(x) for x in (p50, t50, pm, tm)]
    return {
        "dc50": {"R2": float(r2_score(t50, p50)), "RMSE": float(np.sqrt(mean_squared_error(t50, p50))), "MAE": float(mean_absolute_error(t50, p50))},
        "dmax": {"R2": float(r2_score(tm, pm)),  "RMSE": float(np.sqrt(mean_squared_error(tm, pm))),  "MAE": float(mean_absolute_error(tm, pm))},
    }

# ─── Main ────────────────────────────────────────────
print("═"*60)
print("  Baseline vs PosEmbed — Controlled Comparison")
print("═"*60)

X, y50_z, ym_z, scalers = build_dataset()

idx = np.arange(len(X))
tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
tr_i, va_i = train_test_split(tr_i, test_size=0.15/0.8, random_state=RANDOM_STATE)
print(f"  Train: {len(tr_i)}, Val: {len(va_i)}, Test: {len(te_i)}")

def dl(idx, shuffle=False):
    ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                       torch.tensor(y50_z[idx], dtype=torch.float32),
                       torch.tensor(ym_z[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

tr_loader = dl(tr_i, True); va_loader = dl(va_i); te_loader = dl(te_i)

all_results = []

for ModelClass, name in [(BaselineModel, "baseline"), (PosEmbedModel, "pos_embed")]:
    trial_metrics = []
    for t in range(N_TRIALS):
        print(f"\n  {name} — Trial {t+1}/{N_TRIALS}")
        torch.manual_seed(RANDOM_STATE + t*10)
        np.random.seed(RANDOM_STATE + t*10)
        model = ModelClass(input_dim=X.shape[2]).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, best_vl, elapsed = train_model(model, tr_loader, va_loader)
        m = evaluate_rescaled(model, te_loader, scalers)
        print(f"    DC50 R²={m['dc50']['R2']:.4f}  Dmax R²={m['dmax']['R2']:.4f}  ({elapsed:.0f}s)")
        trial_metrics.append(m)
        del model; torch.cuda.empty_cache(); gc.collect()
    
    all_results.append({
        "model": name,
        "dc50_R2": np.mean([m["dc50"]["R2"] for m in trial_metrics]),
        "dc50_R2_std": np.std([m["dc50"]["R2"] for m in trial_metrics]),
        "dc50_RMSE": np.mean([m["dc50"]["RMSE"] for m in trial_metrics]),
        "dc50_RMSE_std": np.std([m["dc50"]["RMSE"] for m in trial_metrics]),
        "dc50_MAE": np.mean([m["dc50"]["MAE"] for m in trial_metrics]),
        "dmax_R2": np.mean([m["dmax"]["R2"] for m in trial_metrics]),
        "dmax_R2_std": np.std([m["dmax"]["R2"] for m in trial_metrics]),
        "dmax_RMSE": np.mean([m["dmax"]["RMSE"] for m in trial_metrics]),
        "dmax_RMSE_std": np.std([m["dmax"]["RMSE"] for m in trial_metrics]),
        "dmax_MAE": np.mean([m["dmax"]["MAE"] for m in trial_metrics]),
        "n_params": n_p,
        "trial_details": trial_metrics,
    })

# Print comparison
bl = all_results[0]; pe = all_results[1]
print("\n" + "="*60)
print("  COMPARISON: Baseline vs Positional Encoding")
print("="*60)
print(f"{'Metric':<20} {'Baseline':<12} {'PosEmbed':<12} {'Δ':<12}")
print("-"*56)
for metric in ["dc50_R2", "dc50_RMSE", "dc50_MAE", "dmax_R2", "dmax_RMSE", "dmax_MAE"]:
    bv = bl[metric]; pv = pe[metric]
    delta = pv - bv if "R2" in metric else bv - pv
    arrow = "↑" if delta > 0 else "↓"
    print(f"{metric:<20} {bv:<12.4f} {pv:<12.4f} {arrow}{abs(delta):.4f}")
print(f"{'Params':<20} {bl['n_params']:<12,} {pe['n_params']:<12,} +{pe['n_params']-bl['n_params']:,}")

# Save
df = pd.DataFrame([{k: v for k, v in r.items() if k != "trial_details"} for r in all_results])
df.to_csv(os.path.join(BM_DIR, "pos_embed_comparison.csv"), index=False)

# Detail
rows = []
for r in all_results:
    for t, m in enumerate(r["trial_details"]):
        rows.append({"model": r["model"], "trial": t+1,
                     **{f"dc50_{k}": v for k,v in m["dc50"].items()},
                     **{f"dmax_{k}": v for k,v in m["dmax"].items()}})
pd.DataFrame(rows).to_csv(os.path.join(BM_DIR, "pos_embed_detailed.csv"), index=False)

# Plot
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    
    for idx, (target, label) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
        ax = axes[idx]
        metrics_names = ["R2", "RMSE", "MAE"]
        bl_vals = [bl.get(f"{target}_{m}", bl.get(f"{target}_{m.lower()}", 0)) for m in metrics_names]
        pe_vals = [pe.get(f"{target}_{m}", pe.get(f"{target}_{m.lower()}", 0)) for m in metrics_names]
        
        x = np.arange(len(metrics_names)); w = 0.3
        ax.bar(x - w/2, bl_vals, w, label="Without PosEmbed", color="#E74C3C", alpha=0.85)
        ax.bar(x + w/2, pe_vals, w, label="With PosEmbed", color="#2ECC71", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(metrics_names)
        ax.set_title(f"{label} Prediction"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        for i, (b, p) in enumerate(zip(bl_vals, pe_vals)):
            better = "↑" if (metrics_names[i] == "R2" and p > b) or (metrics_names[i] != "R2" and p < b) else "↓"
            ax.text(i, max(b, p)*1.05, better, ha="center", fontsize=14, fontweight="bold",
                   color="#2ECC71" if better == "↑" else "#E74C3C")
    
    ax = axes[2]
    deltas = [pe["dc50_R2"] - bl["dc50_R2"], pe["dmax_R2"] - bl["dmax_R2"]]
    ax.bar([0, 1], deltas, color=["#3498DB", "#9B59B6"], alpha=0.85, width=0.5)
    ax.axhline(y=0, color="black", ls="--", alpha=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["DC50 R²", "Dmax R²"])
    ax.set_ylabel("Δ (PosEmbed − Baseline)"); ax.set_title("Improvement from PosEmbed"); ax.grid(alpha=0.3, axis="y")
    for i, d in enumerate(deltas):
        ax.text(i, d + 0.002 if d >= 0 else d - 0.02, f"{d:+.4f}", ha="center", fontsize=11, fontweight="bold")
    
    plt.suptitle("SynGlue: Impact of Learnable Positional Embeddings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pos_embed_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot: pos_embed_comparison.png")
except Exception as e:
    print(f"  Plot error: {e}")

print("\n✅ WP1a complete.")
