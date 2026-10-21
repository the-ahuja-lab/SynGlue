#!/usr/bin/env python3
"""
WP2: Generalization Testing for SynGlue Transformer
====================================================
Four split strategies:
  1. Random 80/20 (baseline)
  2. Target-held-out
  3. E3-held-out
  4. Scaffold-held-out

Loads preprocessed data from preprocess_data.py.
Trains improved transformer with pos_embed, Pre-LN, component gating.
Includes E3 adversarial loss for domain-invariant features.
"""
import os, sys, time, warnings, gc, json, math
warnings.filterwarnings("ignore")
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking"
OUT_DIR  = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/wp2_results"
FIG_DIR  = "/storage/savi/saveenas/Projects/SynGlue/revision/figures"
MODEL_DIR= "/storage/savi/saveenas/Projects/SynGlue/revision/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─── Config ───────────────────────────────────────────
CFG = {
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "epochs": 200,
    "patience": 30,
    "grad_clip": 1.0,
    "hidden_dim": 512,
    "n_heads": 8,
    "n_layers": 3,
    "dropout": 0.15,
    "ff_dim": 2048,
    "random_state": 42,
    "n_trials": 2,
    "e3_adv_lambda": 0.1,
    "e3_adv_hidden": 128,
}


# ─── 1. DATA LOADING ─────────────────────────────────
def load_data():
    """Load preprocessed arrays from preprocess_data.py."""
    t0 = time.time()
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_dc50 = np.load(os.path.join(DATA_DIR, "y_dc50.npy"))
    y_dmax = np.load(os.path.join(DATA_DIR, "y_dmax.npy"))
    meta = pd.read_pickle(os.path.join(DATA_DIR, "meta.pkl"))

    # Z-score normalize targets
    dc50_m, dc50_s = y_dc50.mean(), y_dc50.std()
    dmax_m, dmax_s = y_dmax.mean(), y_dmax.std()
    y_dc50_z = ((y_dc50 - dc50_m) / (dc50_s + 1e-8)).astype(np.float32)
    y_dmax_z = ((y_dmax - dmax_m) / (dmax_s + 1e-8)).astype(np.float32)

    scalers = {"dc50": (dc50_m, dc50_s), "dmax": (dmax_m, dmax_s)}

    # E3 label mapping for adversarial loss
    e3_labels = sorted(meta["e3_name"].unique())
    e3_to_idx = {e: i for i, e in enumerate(e3_labels)}
    meta["E3_idx"] = meta["e3_name"].map(e3_to_idx).values
    n_e3 = len(e3_labels)

    print(f"  Loaded {len(X)} samples in {time.time()-t0:.1f}s")
    print(f"  X: {X.shape}, y_dc50: {y_dc50.shape}, y_dmax: {y_dmax.shape}")
    print(f"  Targets: {meta['target'].nunique()}, E3: {n_e3}, Scaffolds: {meta['scaffold'].nunique()}")
    print(f"  E3 classes: {e3_labels}")
    print(f"  Scalers — DC50: μ={dc50_m:.2f} σ={dc50_s:.3f}, Dmax: μ={dmax_m:.1f} σ={dmax_s:.3f}")

    return X, y_dc50_z, y_dmax_z, meta, scalers, n_e3


# ─── 2. IMPROVED TRANSFORMER ─────────────────────────
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class ImprovedProtacTransformer(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8, n_layers=3,
                 dropout=0.15, ff_dim=2048, n_e3_classes=9, e3_adv_hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Embedding(3, hidden_dim)
        self.component_gate = nn.Parameter(torch.ones(3, hidden_dim) * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_q = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 1))
        self.head_dmax = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 1))
        self.log_var_dc50 = nn.Parameter(torch.zeros(1))
        self.log_var_dmax = nn.Parameter(torch.zeros(1))
        self.e3_classifier = nn.Sequential(
            nn.Linear(hidden_dim, e3_adv_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(e3_adv_hidden, n_e3_classes))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x, e3_adv_lambda=0.0):
        B = x.shape[0]
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        h = self.norm(h)
        gate = torch.sigmoid(self.component_gate)
        h = h * gate.unsqueeze(0)
        attn_weights = torch.softmax(self.attn_q(h), dim=1)
        fused = (h * attn_weights).sum(dim=1)
        dc50_pred = self.head_dc50(fused).squeeze(-1)
        dmax_pred = self.head_dmax(fused).squeeze(-1)
        h_rev = GradientReversal.apply(fused, e3_adv_lambda) if e3_adv_lambda > 0 else fused.detach()
        e3_logits = self.e3_classifier(h_rev)
        return dc50_pred, dmax_pred, {
            "fused": fused, "attn": attn_weights, "e3_logits": e3_logits}

    def extract_embeddings(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        h = self.norm(h)
        gate = torch.sigmoid(self.component_gate)
        h = h * gate.unsqueeze(0)
        attn_weights = torch.softmax(self.attn_q(h), dim=1)
        return (h * attn_weights).sum(dim=1)


# ─── 3. LOSS FUNCTIONS ───────────────────────────────
def uncertainty_loss(pred, target, log_var):
    precision = torch.exp(-log_var)
    return 0.5 * precision * F.mse_loss(pred, target, reduction="mean") + 0.5 * log_var

def e3_loss(e3_logits, e3_labels, n_classes):
    if e3_logits is None: return torch.tensor(0.0, device=DEVICE)
    valid = e3_labels < n_classes
    if not valid.any(): return torch.tensor(0.0, device=DEVICE)
    return F.cross_entropy(e3_logits[valid], e3_labels[valid])


# ─── 4. SPLIT STRATEGIES ─────────────────────────────
def random_split(X, meta):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=CFG["random_state"])
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=CFG["random_state"])
    return tr, va, te

def target_held_out_split(X, meta):
    targets = meta["target"].values
    unique, counts = np.unique(targets, return_counts=True)
    sorted_idx = np.argsort(counts)
    n_hold = max(1, len(unique) // 7)
    held = set(unique[sorted_idx[:n_hold]])
    te_mask = np.array([t in held for t in targets])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=CFG["random_state"])
    print(f"  Held-out {n_hold} targets ({len(te)} test samples): {sorted(list(held))}")
    return tr, va, te

def e3_held_out_split(X, meta):
    e3s = meta["e3_name"].values
    unique, counts = np.unique(e3s, return_counts=True)
    sorted_by_count = np.argsort(-counts)
    # Hold out VHL (second most common) — hardest generalization
    held = {unique[sorted_by_count[1]]} if len(sorted_by_count) > 1 else {unique[1]}
    te_mask = np.array([e in held for e in e3s])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=CFG["random_state"])
    print(f"  Held-out E3: {held} ({len(te)} test samples)")
    return tr, va, te

def scaffold_held_out_split(X, meta):
    scaffolds = meta["scaffold"].values
    valid_mask = np.array([sc is not None for sc in scaffolds])
    if valid_mask.sum() < 20:
        return random_split(X, meta)
    valid_idx = np.where(valid_mask)[0]
    valid_sc = [scaffolds[i] for i in valid_idx]
    sc2idx = defaultdict(list)
    for sc, i in zip(valid_sc, valid_idx): sc2idx[sc].append(i)
    uniq_sc = list(sc2idx.keys())
    rng = np.random.RandomState(CFG["random_state"])
    rng.shuffle(uniq_sc)
    n_test = max(1, int(len(uniq_sc) * 0.2))
    test_sc = set(uniq_sc[:n_test])
    te_idx = [i for sc in test_sc for i in sc2idx[sc]]
    trva_idx = [i for sc in uniq_sc[n_test:] for i in sc2idx[sc]]
    tr, va = train_test_split(trva_idx, test_size=0.15, random_state=CFG["random_state"])
    print(f"  Held-out {n_test} scaffolds, {len(te_idx)} test samples")
    return np.array(tr), np.array(va), np.array(te_idx)


# ─── 5. TRAINING ─────────────────────────────────────
def train_epoch(model, loader, opt, epoch, n_e3):
    model.train()
    total_loss = 0
    n_batches = 0
    e3_lambda = min(CFG["e3_adv_lambda"],
                    CFG["e3_adv_lambda"] * epoch / 10)

    for xb, y50, ym, e3_idx in loader:
        xb, y50, ym, e3_idx = (xb.to(DEVICE), y50.to(DEVICE),
                                ym.to(DEVICE), e3_idx.to(DEVICE))
        p50, pm, aux = model(xb, e3_adv_lambda=e3_lambda)
        l_dc50 = uncertainty_loss(p50, y50, model.log_var_dc50)
        l_dmax = uncertainty_loss(pm, ym, model.log_var_dmax)
        l_task = l_dc50 + l_dmax
        l_e3 = e3_loss(aux["e3_logits"], e3_idx, n_e3)
        loss = l_task + l_e3
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, n_e3):
    model.eval()
    all_p50, all_t50, all_pm, all_tm = [], [], [], []
    for xb, y50, ym, _ in loader:
        xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
        p50, pm, _ = model(xb, e3_adv_lambda=0.0)
        all_p50.append(p50.cpu()); all_t50.append(y50.cpu())
        all_pm.append(pm.cpu());   all_tm.append(ym.cpu())
    p50 = torch.cat(all_p50).numpy()
    t50 = torch.cat(all_t50).numpy()
    pm  = torch.cat(all_pm).numpy()
    tm  = torch.cat(all_tm).numpy()
    return {
        "loss": float(np.mean((p50 - t50)**2) + np.mean((pm - tm)**2)),
        "dc50_R2": float(r2_score(t50, p50)),
        "dc50_RMSE": float(np.sqrt(mean_squared_error(t50, p50))),
        "dc50_MAE": float(mean_absolute_error(t50, p50)),
        "dmax_R2": float(r2_score(tm, pm)),
        "dmax_RMSE": float(np.sqrt(mean_squared_error(tm, pm))),
        "dmax_MAE": float(mean_absolute_error(tm, pm)),
        "n": len(t50),
    }


@torch.no_grad()
def evaluate_rescaled(model, loader, scalers):
    dc50_m, dc50_s = scalers["dc50"]
    dmax_m, dmax_s = scalers["dmax"]
    model.eval()
    all_p50, all_t50, all_pm, all_tm = [], [], [], []
    for xb, y50, ym, _ in loader:
        xb = xb.to(DEVICE)
        p50, pm, _ = model(xb, e3_adv_lambda=0.0)
        all_p50.append(p50.cpu()); all_t50.append(y50.cpu())
        all_pm.append(pm.cpu());   all_tm.append(ym.cpu())
    p50 = torch.cat(all_p50).numpy() * dc50_s + dc50_m
    t50 = torch.cat(all_t50).numpy() * dc50_s + dc50_m
    pm  = torch.cat(all_pm).numpy()  * dmax_s + dmax_m
    tm  = torch.cat(all_tm).numpy()  * dmax_s + dmax_m
    return {
        "dc50_R2": float(r2_score(t50, p50)),
        "dc50_RMSE": float(np.sqrt(mean_squared_error(t50, p50))),
        "dc50_MAE": float(mean_absolute_error(t50, p50)),
        "dmax_R2": float(r2_score(tm, pm)),
        "dmax_RMSE": float(np.sqrt(mean_squared_error(tm, pm))),
        "dmax_MAE": float(mean_absolute_error(tm, pm)),
    }


# ─── 6. FULL PIPELINE ────────────────────────────────
def run_split(X, y50_z, ym_z, meta, scalers, n_e3, split_name, split_fn, trial):
    print(f"\n{'=' * 55}")
    print(f"  {split_name} — Trial {trial}")
    print(f"{'=' * 55}")

    seed = CFG["random_state"] + trial * 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_i, va_i, te_i = split_fn(X, meta)

    def make_loader(idx, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(y50_z[idx], dtype=torch.float32),
            torch.tensor(ym_z[idx], dtype=torch.float32),
            torch.tensor(meta.iloc[idx]["E3_idx"].values, dtype=torch.long))
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle)

    tr_loader = make_loader(tr_i, True)
    va_loader = make_loader(va_i)
    te_loader = make_loader(te_i)

    model = ImprovedProtacTransformer(
        input_dim=4800, hidden_dim=CFG["hidden_dim"],
        n_heads=CFG["n_heads"], n_layers=CFG["n_layers"],
        dropout=CFG["dropout"], ff_dim=CFG["ff_dim"],
        n_e3_classes=n_e3, e3_adv_hidden=CFG["e3_adv_hidden"]).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []
    t0 = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, tr_loader, opt, epoch, n_e3)
        val_metrics = validate(model, va_loader, n_e3)
        sched.step(epoch)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dc50_R2": val_metrics["dc50_R2"],
            "val_dmax_R2": val_metrics["dmax_R2"],
            "lr": opt.param_groups[0]["lr"],
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} "
                  f"(DC50 R²={val_metrics['dc50_R2']:.4f}, "
                  f"Dmax R²={val_metrics['dmax_R2']:.4f})")

        if patience_counter >= CFG["patience"]:
            print(f"    ⏹ Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    te_metrics = evaluate_rescaled(model, te_loader, scalers)
    elapsed = time.time() - t0

    # Per-E3 breakdown on test set
    e3_breakdown = {}
    e3_names = meta["e3_name"].unique()
    for e3_name in e3_names:
        mask = meta.iloc[te_i]["e3_name"].values == e3_name
        if mask.sum() < 3: continue
        e3_loader = make_loader(te_i[mask])
        e3m = evaluate_rescaled(model, e3_loader, scalers)
        e3_breakdown[e3_name] = {"n": int(mask.sum()), "dc50_R2": e3m["dc50_R2"], "dmax_R2": e3m["dmax_R2"]}

    result = {
        "split": split_name, "trial": trial,
        "n_params": n_params, "n_train": len(tr_i),
        "n_val": len(va_i), "n_test": len(te_i),
        "epochs_trained": len(history),
        "best_val_loss": float(best_val_loss),
        "dc50_R2": te_metrics["dc50_R2"],
        "dc50_RMSE": te_metrics["dc50_RMSE"],
        "dc50_MAE": te_metrics["dc50_MAE"],
        "dmax_R2": te_metrics["dmax_R2"],
        "dmax_RMSE": te_metrics["dmax_RMSE"],
        "dmax_MAE": te_metrics["dmax_MAE"],
        "elapsed_min": elapsed / 60,
        "e3_breakdown": e3_breakdown,
    }

    # Save checkpoint
    safe_name = split_name.replace(" ", "_").replace("/", "_")
    ckpt_path = os.path.join(MODEL_DIR, f"wp2_{safe_name}_t{trial}.pt")
    torch.save({"state_dict": best_state, "result": result, "config": CFG}, ckpt_path)
    print(f"  ✅ Trial done — DC50 R²={te_metrics['dc50_R2']:.4f}, "
          f"Dmax R²={te_metrics['dmax_R2']:.4f} ({elapsed/60:.1f} min)")
    print(f"     Checkpoint: {ckpt_path}")

    return result, history, model


# ─── 7. MAIN ─────────────────────────────────────────
def main():
    print("╔" + "═" * 55 + "╗")
    print("║  WP2: Generalization Testing — Improved Transformer  ║")
    print("║  4 splits × 2 trials · Pre-LN · Adversarial E3       ║")
    print("╚" + "═" * 55 + "╝\n")

    global_t0 = time.time()

    # Load data
    X, y50_z, ym_z, meta, scalers, n_e3 = load_data()
    print(f"  Data: {len(X)} samples, {meta['target'].nunique()} targets, {n_e3} E3, "
          f"{meta['scaffold'].nunique()} scaffolds")

    strategies = [
        ("Random 80/20", random_split),
        ("Target-held-out", target_held_out_split),
        ("E3-held-out", e3_held_out_split),
        ("Scaffold-held-out", scaffold_held_out_split),
    ]

    all_results = []
    all_histories = {}

    for split_name, split_fn in strategies:
        print(f"\n{'─' * 55}")
        print(f"  SPLIT: {split_name}")
        print(f"{'─' * 55}")

        trial_results = []
        trial_histories = []

        for t in range(1, CFG["n_trials"] + 1):
            result, history, model = run_split(
                X, y50_z, ym_z, meta, scalers, n_e3,
                split_name, split_fn, t)

            trial_results.append(result)
            trial_histories.append(history)
            all_results.append(result)

            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

        all_histories[split_name] = trial_histories

    # ── Compile results ──
    print(f"\n\n{'=' * 70}")
    print("  WP2 GENERALIZATION RESULTS — Improved Transformer")
    print(f"{'=' * 70}")

    df = pd.DataFrame(all_results)
    print(f"\n{'Split':25s} {'n_train':>8s} {'n_test':>7s} {'DC50 R²':>10s} {'Dmax R²':>10s} {'Epochs':>7s} {'Time(m)':>8s}")
    print(f"{'-'*75}")
    for _, r in df.iterrows():
        print(f"{r['split']:25s} {r['n_train']:8d} {r['n_test']:7d} "
              f"{r['dc50_R2']:10.4f} {r['dmax_R2']:10.4f} "
              f"{r['epochs_trained']:7d} {r['elapsed_min']:8.1f}")

    # Save master results
    df.to_csv(os.path.join(OUT_DIR, "wp2_master_results.csv"), index=False)

    # Aggregate per split
    print(f"\n{'─' * 70}")
    print("  AGGREGATED (mean ± std across trials)")
    print(f"{'─' * 70}")
    agg_rows = []
    for split_name, _ in strategies:
        trials = [r for r in all_results if r["split"] == split_name]
        if not trials: continue
        def avg_std(key):
            vals = [t[key] for t in trials if np.isfinite(t.get(key, 0))]
            return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else (vals[0], 0)

        dc50_r2, dc50_r2_std = avg_std("dc50_R2")
        dmax_r2, dmax_r2_std = avg_std("dmax_R2")
        dc50_rmse, _ = avg_std("dc50_RMSE")
        dmax_rmse, _ = avg_std("dmax_RMSE")

        agg_rows.append({
            "split": split_name,
            "dc50_R2_mean": dc50_r2, "dc50_R2_std": dc50_r2_std,
            "dmax_R2_mean": dmax_r2, "dmax_R2_std": dmax_r2_std,
            "dc50_RMSE": dc50_rmse, "dmax_RMSE": dmax_rmse,
            "n_train": trials[0]["n_train"],
            "n_test": trials[0]["n_test"],
        })
        print(f"  {split_name:20s}: DC50 R²={dc50_r2:.4f}±{dc50_r2_std:.4f}  "
              f"Dmax R²={dmax_r2:.4f}±{dmax_r2_std:.4f}  "
              f"RMSE DC50={dc50_rmse:.3f} Dmax={dmax_rmse:.2f}")

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(OUT_DIR, "wp2_aggregated_results.csv"), index=False)

    # ── Figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Training curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (split_name, histories) in enumerate(all_histories.items()):
            ax = axes[idx // 2, idx % 2]
            h = histories[0]
            epochs = [e["epoch"] for e in h]
            ax.plot(epochs, [e["train_loss"] for e in h], label="Train Loss", alpha=0.8)
            ax.plot(epochs, [e["val_loss"] for e in h], label="Val Loss", alpha=0.8)
            ax.plot(epochs, [e["val_dc50_R2"] for e in h], label="Val DC50 R²", alpha=0.8)
            ax.plot(epochs, [e["val_dmax_R2"] for e in h], label="Val Dmax R²", alpha=0.8)
            ax.set_title(split_name); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.suptitle("WP2: Training Curves (Improved Transformer)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "wp2_training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Generalization bar plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]
        for ax_idx, (target, title) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
            ax = axes[ax_idx]
            splits = [r["split"] for r in agg_rows]
            means = [r[f"{target}_R2_mean"] for r in agg_rows]
            stds  = [r[f"{target}_R2_std"] for r in agg_rows]
            x = np.arange(len(splits))
            bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels([s[:14] for s in splits], rotation=20, ha="right", fontsize=9)
            ax.set_title(f"{title} R² by Split Strategy")
            ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
            ax.set_ylabel("R²"); ax.grid(alpha=0.3, axis="y")
            for bar, v in zip(bars, means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(v)*0.02+0.005,
                        f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.suptitle("WP2: Generalization Across Split Strategies", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "wp2_generalization.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 3. Comparison with old WP1 results
        old_path = os.path.join(DATA_DIR, "generalization_results.csv")
        if os.path.exists(old_path):
            old_df = pd.read_csv(old_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax_idx, (target, title) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
                ax = axes[ax_idx]
                old_vals, new_vals, labels = [], [], []
                for o_row in old_df.itertuples():
                    s = o_row.split
                    new_row = [r for r in agg_rows if s.startswith(r["split"][:8])]
                    if not new_row: continue
                    old_vals.append(getattr(o_row, f"{target}_R2_mean", 0))
                    new_vals.append(new_row[0][f"{target}_R2_mean"])
                    labels.append(s[:14])
                x = np.arange(len(labels)); w = 0.3
                ax.bar(x-w/2, old_vals, w, label="Old (25-epoch TF+RF)", color="#E74C3C", alpha=0.7)
                ax.bar(x+w/2, new_vals, w, label="Improved (200-epoch)", color="#2ECC71", alpha=0.7)
                ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                ax.set_title(f"{title} R²: Old vs Improved")
                ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
                ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
            plt.suptitle("WP2: Old (Two-Stage TF+RF) vs Improved Transformer", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "wp2_old_vs_new.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"\n  ✅ Figures saved to {FIG_DIR}/")

    except Exception as e:
        print(f"  ⚠ Figure generation error: {e}")
        import traceback; traceback.print_exc()

    total_min = (time.time() - global_t0) / 60
    print(f"\n{'=' * 55}")
    print(f"  ✅ WP2 GENERALIZATION TESTING COMPLETE")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Results: {OUT_DIR}/")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Models: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
