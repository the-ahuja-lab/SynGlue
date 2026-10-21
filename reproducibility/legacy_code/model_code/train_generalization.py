#!/usr/bin/env python3
"""
Component-Aware Embedding Transformer — Generalization Training + Frozen Model Analysis

Pipeline:
1. Load component-aware tensors (W/L/E GROVER embeddings + DC50/Dmax labels)
2. Train ComponentAwareTransformer (pos_embed) under 4 split strategies:
   - random (baseline)
   - target-held-out
   - E3-held-out
   - scaffold-held-out (Murcko)
3. Save frozen model + fused embeddings per split
4. Analyze frozen model: attention weights, component ablation, embedding stats

Output: revision/experiments/gen_training/
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = f"{REV}/benchmarking/generalization_data"
OUT_DIR = f"{REV}/experiments/gen_training"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

BATCH_SIZE, LR, EPOCHS, PATIENCE, RANDOM_STATE = 64, 1e-4, 120, 15, 42

# ─── Component-Aware Transformer ─────────────────────
class ComponentAwareTransformer(nn.Module):
    """Component-aware embedding transformer: W/L/E tokens + positional identity."""
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(3, hidden_dim)  # component identity (W/L/E)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x, return_attn=False):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn_scores = self.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1)
        p50 = self.head_dc50(fused).squeeze(-1)
        pm = self.head_dmax(fused).squeeze(-1)
        if return_attn:
            return p50, pm, attn_weights
        return p50, pm

    def embed(self, x):
        """Return fused embedding (frozen feature extractor)."""
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn_weights = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn_weights).sum(dim=1)

# ─── Data ────────────────────────────────────────────
def load_data():
    X = np.load(f"{DATA_DIR}/X.npy")
    y50 = np.load(f"{DATA_DIR}/y_dc50_z.npy")
    ym = np.load(f"{DATA_DIR}/y_dmax_z.npy")
    meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    scalers = json.load(open(f"{DATA_DIR}/scalers.json"))
    print(f"  X: {X.shape}, meta: {meta.shape}")
    return X, y50, ym, meta, scalers

# ─── Split strategies ────────────────────────────────
def random_split(X, y50, ym, meta):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
    return tr, va, te, "random"

def target_held_out(X, y50, ym, meta):
    """Hold out entire target proteins (all PROTACs for held targets)."""
    targets = meta['target_uniprot'].value_counts()
    rng = np.random.RandomState(RANDOM_STATE)
    # Hold out ~20% of targets (by count), >=5 samples each
    candidates = targets[targets >= 5].index.tolist()
    n_hold = max(1, int(len(candidates) * 0.2))
    held = set(rng.choice(candidates, size=n_hold, replace=False))
    te_idx = np.where(meta['target_uniprot'].isin(held))[0]
    trva_idx = np.where(~meta['target_uniprot'].isin(held))[0]
    tr, va = train_test_split(trva_idx, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te_idx, f"target-held({len(held)} targets)"

def e3_held_out(X, y50, ym, meta):
    """Hold out one E3 ligase family (VHL)."""
    held_e3 = ['VHL']
    te_idx = np.where(meta['e3'].isin(held_e3))[0]
    trva_idx = np.where(~meta['e3'].isin(held_e3))[0]
    tr, va = train_test_split(trva_idx, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te_idx, f"E3-held(VHL, {len(te_idx)} samples)"

def scaffold_held_out(X, y50, ym, meta):
    """Hold out entire Murcko scaffolds."""
    sc_counts = meta['scaffold'].value_counts()
    rng = np.random.RandomState(RANDOM_STATE)
    candidates = sc_counts[sc_counts >= 3].index.tolist()
    n_hold = max(1, int(len(candidates) * 0.2))
    held = set(rng.choice(candidates, size=n_hold, replace=False))
    te_idx = np.where(meta['scaffold'].isin(held))[0]
    trva_idx = np.where(~meta['scaffold'].isin(held))[0]
    tr, va = train_test_split(trva_idx, test_size=0.15, random_state=RANDOM_STATE)
    return tr, va, te_idx, f"scaffold-held({len(held)} scaffolds)"

SPLITS = {
    "random": random_split,
    "target_held_out": target_held_out,
    "e3_held_out": e3_held_out,
    "scaffold_held_out": scaffold_held_out,
}

# ─── Training ────────────────────────────────────────
def train_model(model, tr_loader, va_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val = float("inf"); best_state = None; counter = 0; t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for xb, y50, ym in tr_loader:
            xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
            p50, pm = model(xb)
            loss = loss_fn(p50, y50) + loss_fn(pm, ym)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, y50, ym in va_loader:
                xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
                p50, pm = model(xb)
                vl += (loss_fn(p50, y50) + loss_fn(pm, ym)).item()
        vl /= len(va_loader); sched.step(vl)
        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; counter = 0
        else:
            counter += 1
        if counter >= PATIENCE:
            print(f"    early stop @ epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model, best_val, time.time() - t0

def evaluate(model, loader, scalers):
    dc50_m, dc50_s = scalers["dc50"]; dmax_m, dmax_s = scalers["dmax"]
    model.eval(); p50, t50, pm, tm = [], [], [], []
    with torch.no_grad():
        for xb, y50, ym in loader:
            xb = xb.to(DEVICE)
            pr50, prm = model(xb)
            p50.extend((pr50.cpu().numpy() * dc50_s + dc50_m))
            t50.extend((y50.numpy() * dc50_s + dc50_m))
            pm.extend((prm.cpu().numpy() * dmax_s + dmax_m))
            tm.extend((ym.numpy() * dmax_s + dmax_m))
    p50, t50, pm, tm = [np.array(x) for x in (p50, t50, pm, tm)]
    return {
        "dc50": {"R2": float(r2_score(t50, p50)), "RMSE": float(np.sqrt(mean_squared_error(t50, p50))), "MAE": float(mean_absolute_error(t50, p50))},
        "dmax": {"R2": float(r2_score(tm, pm)),  "RMSE": float(np.sqrt(mean_squared_error(tm, pm))),  "MAE": float(mean_absolute_error(tm, pm))},
    }

# ─── Frozen model analysis ───────────────────────────
def frozen_analysis(model, X_full, split_name, scalers):
    """Analyze frozen model on full data: attention, ablations, embedding stats."""
    X_t = torch.tensor(X_full, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        # Attention weights
        _, _, attn = model(X_t, return_attn=True)
        attn = attn.cpu().numpy()  # (N, 3)
        # Component ablation: zero out each component token
        results = {}
        base_dc50, base_dmax = [], []
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256]
            p50, pm = model(b)
            base_dc50.extend(p50.cpu().numpy()); base_dmax.extend(pm.cpu().numpy())
        base_dc50, base_dmax = np.array(base_dc50), np.array(base_dmax)

        for comp, idx in [("warhead", 0), ("linker", 1), ("e3", 2)]:
            X_abl = X_t.clone()
            X_abl[:, idx, :] = 0
            d50, dm = [], []
            for i in range(0, len(X_abl), 256):
                p50, pm = model(X_abl[i:i+256])
                d50.extend(p50.cpu().numpy()); dm.extend(pm.cpu().numpy())
            d50, dm = np.array(d50), np.array(dm)
            results[f"abl_{comp}"] = {
                "delta_dc50_z": float(np.mean(np.abs(d50 - base_dc50))),
                "delta_dmax_z": float(np.mean(np.abs(dm - base_dmax))),
                "rmse_dc50_z": float(np.sqrt(np.mean((d50 - base_dc50)**2))),
            }

        # Embedding stats
        Z = model.embed(X_t).cpu().numpy()
        results["embedding"] = {
            "shape": list(Z.shape),
            "mean_norm": float(np.mean(np.linalg.norm(Z, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(Z, axis=1))),
        }
        results["attention"] = {
            "warhead": float(attn[:, 0].mean()), "linker": float(attn[:, 1].mean()), "e3": float(attn[:, 2].mean()),
        }
        return results, attn

# ─── Main ────────────────────────────────────────────
def main():
    X, y50, ym, meta, scalers = load_data()
    summary = []

    for split_name, split_fn in SPLITS.items():
        print(f"\n{'='*60}\n  Split: {split_name}\n{'='*60}")
        tr, va, te, desc = split_fn(X, y50, ym, meta)
        print(f"  Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}  ({desc})")

        # Refit scalers on train only (proper protocol)
        y50_tr = y50[tr]; ym_tr = ym[tr]
        dc50_m, dc50_s = y50_tr.mean(), y50_tr.std()
        dmax_m, dmax_s = ym_tr.mean(), ym_tr.std()
        split_scalers = {"dc50": (dc50_m, dc50_s), "dmax": (dmax_m, dmax_s)}

        def make_loader(idx, shuffle=False):
            ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                               torch.tensor(y50[idx], dtype=torch.float32),
                               torch.tensor(ym[idx], dtype=torch.float32))
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

        tr_loader = make_loader(tr, True); va_loader = make_loader(va); te_loader = make_loader(te)

        torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
        model = ComponentAwareTransformer(input_dim=X.shape[2]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, best_vl, elapsed = train_model(model, tr_loader, va_loader)
        print(f"  Params: {n_params:,} | best val loss: {best_vl:.4f} | time: {elapsed:.0f}s")

        # Evaluate (z-space metrics are comparable; rescale for report)
        te_metrics_z = evaluate(model, te_loader, split_scalers)
        # also rescale to original units
        te_metrics = evaluate(model, te_loader, {"dc50": (scalers['dc50'][0], scalers['dc50'][1]), "dmax": (scalers['dmax'][0], scalers['dmax'][1])})

        print(f"  TEST  DC50: R2={te_metrics['dc50']['R2']:.3f} RMSE={te_metrics['dc50']['RMSE']:.3f} (log10 nM)")
        print(f"  TEST  Dmax: R2={te_metrics['dmax']['R2']:.3f} RMSE={te_metrics['dmax']['RMSE']:.3f} (%)")

        # Save frozen model
        model_path = f"{OUT_DIR}/model_{split_name}.pt"
        torch.save(model.state_dict(), model_path)
        np.save(f"{OUT_DIR}/tr_{split_name}.npy", np.asarray(tr))
        np.save(f"{OUT_DIR}/va_{split_name}.npy", np.asarray(va))
        np.save(f"{OUT_DIR}/te_{split_name}.npy", np.asarray(te))

        # Frozen analysis
        analysis, attn = frozen_analysis(model, X, split_name, scalers)
        print(f"  Attention (mean): W={analysis['attention']['warhead']:.3f} "
              f"L={analysis['attention']['linker']:.3f} E={analysis['attention']['e3']:.3f}")
        print(f"  Ablation |ΔDC50|: W={analysis['abl_warhead']['delta_dc50_z']:.4f} "
              f"L={analysis['abl_linker']['delta_dc50_z']:.4f} E={analysis['abl_e3']['delta_dc50_z']:.4f}")

        summary.append({
            "split": split_name, "desc": desc, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "n_params": n_params, "best_val_loss": best_vl, "train_time_s": elapsed,
            "dc50_R2": te_metrics['dc50']['R2'], "dc50_RMSE": te_metrics['dc50']['RMSE'], "dc50_MAE": te_metrics['dc50']['MAE'],
            "dmax_R2": te_metrics['dmax']['R2'], "dmax_RMSE": te_metrics['dmax']['RMSE'], "dmax_MAE": te_metrics['dmax']['MAE'],
            "attn_W": analysis['attention']['warhead'], "attn_L": analysis['attention']['linker'], "attn_E": analysis['attention']['e3'],
            "abl_W_dc50": analysis['abl_warhead']['delta_dc50_z'], "abl_L_dc50": analysis['abl_linker']['delta_dc50_z'], "abl_E_dc50": analysis['abl_e3']['delta_dc50_z'],
        })

    summ = pd.DataFrame(summary)
    summ.to_csv(f"{OUT_DIR}/generalization_summary.csv", index=False)
    print(f"\n\n{'='*60}\nGENERALIZATION SUMMARY\n{'='*60}")
    print(summ[['split', 'n_test', 'dc50_R2', 'dc50_RMSE', 'dmax_R2', 'dmax_RMSE', 'attn_W', 'attn_L', 'attn_E']].to_string(index=False))
    print(f"\nSaved: {OUT_DIR}/generalization_summary.csv")

if __name__ == '__main__':
    main()
