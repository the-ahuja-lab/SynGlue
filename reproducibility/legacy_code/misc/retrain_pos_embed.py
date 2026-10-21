#!/usr/bin/env python3
"""
WP1: Retrain MultiTaskProtacModel WITH positional encoding on real data.
Compares: baseline (existing checkpoint) vs. pos_embed (retrained).

This script:
  1. Loads training data (X.npy, y_dc50.npy, y_dmax.npy) from the real pipeline
  2. Builds both baseline and pos_embed models
  3. Trains both from scratch with identical settings
  4. Evaluates on held-out test set
  5. Reports ΔR², ΔRMSE, ΔMAE

Usage:
  conda activate synglu_test_package
  python revision/patches/retrain_pos_embed.py

Outputs:
  revision/patches/pos_embed_results.json
  revision/figures/retraining_comparison.png
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─── Paths ─────────────────────────────────────────
SYNGLUE_PY = "/storage/savi/saveenas/Projects/SynGlue_Py"
DATA_DIR   = os.path.join(SYNGLUE_PY, "data")
MODEL_DIR  = os.path.join(SYNGLUE_PY, "models")
OUTPUT_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision/patches"
FIGURE_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Config ─────────────────────────────────────────
BATCH_SIZE  = 64
LR          = 1e-4
EPOCHS      = 100
PATIENCE    = 10
RANDOM_STATE = 42

# ─── Load Data ──────────────────────────────────────
def load_data():
    """Try loading from standard locations."""
    candidates = [
        os.path.join(DATA_DIR, "..", "Architecture_Code", "05_Models", "X.npy"),
        os.path.join(DATA_DIR, "X.npy"),
        os.path.join(SYNGLUE_PY, "X.npy"),
        os.path.join(SYNGLUE_PY, "data", "X.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            X = np.load(path)
            print(f"Loaded X from {path}: {X.shape}")
            break
    else:
        # Check if we can generate from the dataset pipeline
        print("X.npy not found. Need to generate from dataset builder.")
        print("Try running 05_Models_attention_dataset_builder.py first.")
        sys.exit(1)
    
    for path, label in [
        (os.path.join(os.path.dirname(path), "y_dc50.npy"), "y_dc50"),
        (os.path.join(os.path.dirname(path), "y_dmax.npy"), "y_dmax"),
        (os.path.join(DATA_DIR, "..", "05_Models", "y_dc50.npy"), "y_dc50"),
        (os.path.join(DATA_DIR, "..", "05_Models", "y_dmax.npy"), "y_dmax"),
    ]:
        if os.path.exists(path) and label == "y_dc50":
            y_dc50 = np.load(path)
            print(f"Loaded y_dc50 from {path}: {y_dc50.shape}")
        if os.path.exists(path) and label == "y_dmax":
            y_dmax = np.load(path)
            print(f"Loaded y_dmax from {path}: {y_dmax.shape}")
    
    if "y_dc50" not in dir() or "y_dmax" not in dir():
        print("y_dc50.npy or y_dmax.npy not found")
        sys.exit(1)
    
    return X, y_dc50, y_dmax


# ─── Models ─────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * attn).sum(dim=1)
        return self.head_dc50(fused), self.head_dmax(fused), attn

class PosEmbedModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(3, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
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
        return self.head_dc50(fused), self.head_dmax(fused), attn


# ─── Training ───────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    counter = 0
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, y50, ym in train_loader:
            xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
            p50, pm, _ = model(xb)
            loss = loss_fn(p50.squeeze(), y50) + loss_fn(pm.squeeze(), ym)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, y50, ym in val_loader:
                xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
                p50, pm, _ = model(xb)
                val_loss += (loss_fn(p50.squeeze(), y50) + loss_fn(pm.squeeze(), ym)).item()
        val_loss /= len(val_loader)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  [{tag}] Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    return model, best_loss, history, elapsed


def evaluate(model, loader):
    model.eval()
    p50, t50, pm, tm = [], [], [], []
    with torch.no_grad():
        for xb, y50, ym in loader:
            xb = xb.to(DEVICE)
            pr50, prm, _ = model(xb)
            p50.extend(pr50.squeeze().cpu().numpy())
            t50.extend(y50.numpy())
            pm.extend(prm.squeeze().cpu().numpy())
            tm.extend(ym.numpy())
    p50 = np.array(p50); t50 = np.array(t50)
    pm = np.array(pm); tm = np.array(tm)
    return {
        "dc50": {
            "R2": float(r2_score(t50, p50)),
            "RMSE": float(np.sqrt(mean_squared_error(t50, p50))),
            "MAE": float(mean_absolute_error(t50, p50)),
        },
        "dmax": {
            "R2": float(r2_score(tm, pm)),
            "RMSE": float(np.sqrt(mean_squared_error(tm, pm))),
            "MAE": float(mean_absolute_error(tm, pm)),
        }
    }


# ─── Main ───────────────────────────────────────────
def main():
    print("═══ WP1: Positional Encoding Retraining ═══")
    print()
    
    # Load data
    X, y_dc50, y_dmax = load_data()
    
    # Split: train/val/test
    X_tr, X_te, y50_tr, y50_te, ym_tr, ym_te = train_test_split(
        X, y_dc50, y_dmax, test_size=0.15, random_state=RANDOM_STATE
    )
    X_tr, X_va, y50_tr, y50_va, ym_tr, ym_va = train_test_split(
        X_tr, y50_tr, ym_tr, test_size=0.15/0.85, random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")
    
    def make_loader(x, y50, ym, shuffle=False):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y50, dtype=torch.float32),
            torch.tensor(ym, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
    
    tr_loader = make_loader(X_tr, y50_tr, ym_tr, shuffle=True)
    va_loader = make_loader(X_va, y50_va, ym_va)
    te_loader = make_loader(X_te, y50_te, ym_te)
    
    results = {}
    
    for model_class, name in [
        (BaselineModel, "baseline"),
        (PosEmbedModel, "pos_embed"),
    ]:
        print(f"\n─── Training {name} ───")
        model = model_class().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        
        model, best_loss, history, elapsed = train_model(
            model, tr_loader, va_loader, tag=name
        )
        
        ev = evaluate(model, te_loader)
        results[name] = {
            "n_params": n_params,
            "best_val_loss": best_loss,
            "train_time_s": round(elapsed, 1),
            "test": ev,
        }
        
        print(f"  Results: DC50 R²={ev['dc50']['R2']:.4f}, Dmax R²={ev['dmax']['R2']:.4f}")
    
    # Compare
    delta_dc50 = results["pos_embed"]["test"]["dc50"]["R2"] - results["baseline"]["test"]["dc50"]["R2"]
    delta_dmax = results["pos_embed"]["test"]["dmax"]["R2"] - results["baseline"]["test"]["dmax"]["R2"]
    
    print("\n═══ COMPARISON ═══")
    print(f"  DC50 R²:  baseline={results['baseline']['test']['dc50']['R2']:.4f} → "
          f"pos_embed={results['pos_embed']['test']['dc50']['R2']:.4f} (Δ={delta_dc50:+.4f})")
    print(f"  DC50 RMSE: baseline={results['baseline']['test']['dc50']['RMSE']:.4f} → "
          f"pos_embed={results['pos_embed']['test']['dc50']['RMSE']:.4f}")
    print(f"  Dmax R²:  baseline={results['baseline']['test']['dmax']['R2']:.4f} → "
          f"pos_embed={results['pos_embed']['test']['dmax']['R2']:.4f} (Δ={delta_dmax:+.4f})")
    print(f"  Dmax RMSE: baseline={results['baseline']['test']['dmax']['RMSE']:.4f} → "
          f"pos_embed={results['pos_embed']['test']['dmax']['RMSE']:.4f}")
    print(f"  Parameters: baseline={results['baseline']['n_params']:,}, "
          f"pos_embed={results['pos_embed']['n_params']:,} "
          f"(+{results['pos_embed']['n_params'] - results['baseline']['n_params']:,})")
    
    # Save results
    results["delta"] = {"dc50_R2": delta_dc50, "dmax_R2": delta_dmax}
    results_path = os.path.join(OUTPUT_DIR, "pos_embed_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    
    # Save models
    for name in ["baseline", "pos_embed"]:
        torch.save(results[name].get("state_dict", None), 
                   os.path.join(OUTPUT_DIR, f"model_{name}.pt"))
    
    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # Metrics comparison
        metrics = ["R2", "RMSE", "MAE"]
        for idx, (target, label) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
            ax = axes[idx]
            bl_vals = [results["baseline"]["test"][target][m] for m in metrics]
            pe_vals = [results["pos_embed"]["test"][target][m] for m in metrics]
            
            x = np.arange(len(metrics))
            w = 0.35
            ax.bar(x - w/2, bl_vals, w, label="Without PosEmbed", color="#E74C3C", alpha=0.85)
            ax.bar(x + w/2, pe_vals, w, label="With PosEmbed", color="#2ECC71", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_title(f"{label} Prediction", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            
            # Annotate
            for i, (b, p) in enumerate(zip(bl_vals, pe_vals)):
                if metrics[i] == "R2":
                    better = "↑" if p > b else "↓"
                else:
                    better = "↓" if p < b else "↑"
                ax.text(i, max(b, p)*1.05, better, ha="center", fontsize=14, fontweight="bold",
                       color="#2ECC71" if better == "↑" else "#E74C3C")
        
        # Delta bar
        ax = axes[2]
        deltas = [delta_dc50, delta_dmax]
        labels = ["DC50 R²", "Dmax R²"]
        colors = ["#3498DB", "#9B59B6"]
        ax.bar([0, 1], deltas, color=colors, alpha=0.85, width=0.5)
        ax.axhline(y=0, color="black", ls="--", alpha=0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Δ (PosEmbed − Baseline)")
        ax.set_title("Improvement from Positional Encoding", fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        
        # Add value labels
        for i, d in enumerate(deltas):
            ax.text(i, d + 0.001 if d >= 0 else d - 0.015, f"{d:+.4f}", 
                   ha="center", fontsize=11, fontweight="bold")
        
        plt.suptitle("SynGlue: Impact of Learnable Positional Embeddings (Real Data)", 
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plot_path = os.path.join(FIGURE_DIR, "retraining_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {plot_path}")
    except Exception as e:
        print(f"  Plotting error: {e}")
    
    print("\n  ✓ WP1 complete.")


if __name__ == "__main__":
    main()
