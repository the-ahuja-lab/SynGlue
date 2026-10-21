#!/usr/bin/env python3
"""
WP2: Improved Transformer Model — Generalized for E3 & Scaffold
================================================================
Addresses four goals together:
  1. Transformer architecture improvements (pos_embed, deeper, LayerNorm, gating)
  2. E3 generalization via adversarial regularization (domain confusion)
  3. Scaffold-held-out evaluation & scaffold-aware training
  4. 200-epoch training with cosine annealing, gradient accumulation

Pipeline:
  - Loads GROVER features for [warhead, linker, E3] with labels (DC50, Dmax)
  - Four splitting strategies: random, target-held-out, E3-held-out, scaffold-held-out
  - Trains an improved transformer with:
      * Learnable positional embeddings (3 × hidden_dim)
      * 4-layer TransformerEncoder with Pre-LN
      * Component-specific gating (learned gates per component)
      * Auxiliary E3 classification with GradientReversal → domain confusion
      * Uncertainty-weighted multi-task loss
  - Evaluates every N epochs, saves best checkpoint
  - Generates comprehensive results table + figures

Outputs:
  - benchmarking/wp2_results/{split}_results.json per split
  - benchmarking/wp2_results/master_summary.csv
  - figures/wp2_training_curves.png, wp2_generalization.png, wp2_e3_confusion.png
  - models/wp2_best_{split}.pt (checkpoints)
"""

import os, sys, time, warnings, gc, json, math, copy
warnings.filterwarnings("ignore")
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)

# ─── Paths ───────────────────────────────────────────
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
REV_DIR  = "/storage/savi/saveenas/Projects/SynGlue/revision"
BM_DIR   = os.path.join(REV_DIR, "benchmarking", "wp2_results")
FIG_DIR  = os.path.join(REV_DIR, "figures")
MODEL_DIR= os.path.join(REV_DIR, "models")
os.makedirs(BM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters ──────────────────────────────────
CFG = {
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "epochs": 200,
    "warmup_epochs": 10,
    "patience": 30,            # early stopping patience
    "grad_clip": 1.0,
    "grad_accum_steps": 2,     # effective batch = 128
    "hidden_dim": 512,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.15,
    "ff_dim": 2048,
    "random_state": 42,
    "n_trials": 2,             # per split strategy
    # E3 adversarial
    "e3_adv_lambda": 0.1,      # weight for E3 domain confusion loss
    "e3_adv_hidden": 128,
    # Loss weighting
    "dc50_loss_weight": 1.0,
    "dmax_loss_weight": 1.0,
}

# ─── 1. DATA LOADING (same preprocessing as WP1) ──────
def load_data():
    """Load and merge GROVER features, return X, y_dc50, y_dmax, meta."""
    def load_csv(path):
        df = pd.read_csv(path, low_memory=False)
        print(f"  {os.path.basename(path)}: {df.shape}")
        return df

    protac_df  = load_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv")
    warhead_df = load_csv(f"{DATA_DIR}/Updated_warhead.csv")
    linker_df  = load_csv(f"{DATA_DIR}/linker_updated_features.csv")
    e3_df      = load_csv(f"{DATA_DIR}/E3_updated_features.csv")

    # Normalize join keys
    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    for c in ["E3 ligase"]:
        if c in protac_df.columns: protac_df[c] = protac_df[c].astype(str).str.strip().str.upper()
    for c in ["E3_Ligase"]:
        if c in e3_df.columns: e3_df[c] = e3_df[c].astype(str).str.strip().str.upper()

    # Prefix GROVER columns
    def prefix_grover(df, prefix):
        return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df  = prefix_grover(linker_df, "L_")
    e3_df      = prefix_grover(e3_df, "E_")

    w_g = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_g = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_g = [c for c in e3_df.columns if c.startswith("E_Grover_")]

    # Build component map
    rows = []
    for _, p in protac_df.iterrows():
        wh = warhead_df[(warhead_df["warhead_Uniprot"] == p["warhead_Uniprot"]) |
                        (warhead_df["warhead_Target"] == p["warhead_Target"])]
        if wh.empty: continue
        lk = linker_df[linker_df["linker_Compound_ID"] == p["Linker ID"]]
        if lk.empty: continue
        e3_match = e3_df[e3_df["E3_Ligase"] == p["E3 ligase"]]
        if e3_match.empty: continue
        rows.append({
            "W_vec": wh.iloc[0][w_g].values.astype(np.float32),
            "L_vec": lk.iloc[0][l_g].values.astype(np.float32),
            "E_vec": e3_match.iloc[0][e_g].values.astype(np.float32),
            "Target": str(p.get("warhead_Target", "")),
            "E3_Ligase": str(p.get("E3 ligase", "")),
            "PROTAC_Compound_ID": str(p.get("PROTAC_Compound_ID", "")),
            "SMILES": str(p.get("PROTAC_SMILES", "")),
            "DC50_raw": p.get("DC50 (nM)", np.nan),
            "Dmax_raw": p.get("Dmax (%)", np.nan),
        })

    protac_map = pd.DataFrame(rows)

    # Parse DC50: handle ">", "<", "/" formats
    def parse_dc50(v):
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

    protac_map["DC50"] = protac_map["DC50_raw"].apply(parse_dc50)
    protac_map["Dmax"] = pd.to_numeric(protac_map["Dmax_raw"], errors="coerce")

    # Keep only samples with both labels
    valid = protac_map.dropna(subset=["DC50", "Dmax"]).copy()
    print(f"\n  Valid samples with both labels: {len(valid)} (out of {len(protac_map)} matched)")

    # Build tensors
    X = np.stack([
        np.stack([r["W_vec"], r["L_vec"], r["E_vec"]])
        for _, r in valid.iterrows()
    ])
    
    y_dc50 = np.log10(np.clip(valid["DC50"].values, 1e-3, None)).astype(np.float32)
    y_dmax = valid["Dmax"].values.astype(np.float32)

    # Z-score normalize targets
    dc50_mean, dc50_std = y_dc50.mean(), y_dc50.std()
    dmax_mean, dmax_std = y_dmax.mean(), y_dmax.std()
    y_dc50_z = ((y_dc50 - dc50_mean) / (dc50_std + 1e-8)).astype(np.float32)
    y_dmax_z = ((y_dmax - dmax_mean) / (dmax_std + 1e-8)).astype(np.float32)

    scalers = {
        "dc50": (dc50_mean, dc50_std),
        "dmax": (dmax_mean, dmax_std),
    }

    meta = valid[["PROTAC_Compound_ID", "Target", "E3_Ligase", "SMILES"]].copy()
    
    # E3 label mapping for adversarial loss
    e3_labels = meta["E3_Ligase"].unique()
    e3_to_idx = {e: i for i, e in enumerate(e3_labels)}
    meta["E3_idx"] = meta["E3_Ligase"].map(e3_to_idx).values

    print(f"  X shape: {X.shape}  ({X.shape[0]} samples, {X.shape[1]} components, {X.shape[2]} features)")
    print(f"  DC50: μ={dc50_mean:.2f} σ={dc50_std:.2f} (log10 nM)")
    print(f"  Dmax: μ={dmax_mean:.1f} σ={dmax_std:.1f} (%)")
    print(f"  Unique targets: {meta['Target'].nunique()}")
    print(f"  Unique E3 ligases: {meta['E3_Ligase'].nunique()} → {e3_labels.tolist()}")
    print(f"  Unique scaffolds: computing...")

    # Precompute scaffolds
    scaffolds_list = []
    valid_scaffold_idx = []
    for i, smi in enumerate(meta["SMILES"].values):
        if pd.isna(smi) or str(smi).strip() in ("", "nan"):
            scaffolds_list.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            scaffolds_list.append(None)
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds_list.append(sc if sc else None)
        except:
            scaffolds_list.append(None)
    meta["Scaffold"] = scaffolds_list
    n_scaffolded = meta["Scaffold"].notna().sum()
    n_unique_sc = meta["Scaffold"].nunique()
    print(f"  Scaffolds: {n_scaffolded}/{len(meta)} have valid scaffolds, {n_unique_sc} unique")

    return X, y_dc50_z, y_dmax_z, meta, scalers


# ─── 2. IMPROVED TRANSFORMER ARCHITECTURE ──────────────
class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain confusion."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class ImprovedProtacTransformer(nn.Module):
    """
    Improved multi-task PROTAC transformer with:
      - Learnable positional embeddings (3 × hidden_dim)
      - Pre-LayerNorm TransformerEncoder (4 layers, 8 heads)
      - Component-specific gating (learned gate per component)
      - Auxiliary E3 classifier for domain confusion (adversarial)
      - Uncertainty-weighted multi-task output heads
    """
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8, n_layers=4,
                 dropout=0.15, ff_dim=2048, n_e3_classes=3, e3_adv_hidden=128):
        super().__init__()

        # Input projection: 4800 → hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable positional embeddings for [warhead, linker, E3]
        self.pos_embed = nn.Embedding(3, hidden_dim)

        # Component gating: learned scalar gates per component (after transformer)
        self.component_gate = nn.Parameter(torch.ones(3, hidden_dim) * 0.5)

        # Pre-LN TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,  # Pre-LN for stable deep training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Post-transformer LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

        # Attention pooling
        self.attn_q = nn.Linear(hidden_dim, 1)

        # Multi-task output heads with learnable uncertainty weighting
        self.head_dc50 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self.head_dmax = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

        # Learnable uncertainty weights (log variance)
        self.log_var_dc50 = nn.Parameter(torch.zeros(1))
        self.log_var_dmax = nn.Parameter(torch.zeros(1))

        # E3 adversarial classifier
        self.e3_classifier = nn.Sequential(
            nn.Linear(hidden_dim, e3_adv_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(e3_adv_hidden, n_e3_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x, e3_adv_lambda=0.0):
        """
        Args:
            x: (B, 3, 4800)  GROVER embeddings for [warhead, linker, E3]
            e3_adv_lambda: weight for gradient reversal (0.0 = no adversarial)
        Returns:
            dc50_pred, dmax_pred, aux_out (dict with fused, attn, e3_logits, e3_loss)
        """
        B = x.shape[0]

        # Project to hidden_dim
        h = self.proj(x)  # (B, 3, hidden_dim)

        # Add positional embeddings
        pos = self.pos_embed(torch.arange(3, device=x.device))  # (3, hidden_dim)
        h = h + pos.unsqueeze(0)

        # Transformer (Pre-LN inside each layer)
        h = self.transformer(h)  # (B, 3, hidden_dim)
        h = self.norm(h)

        # Component gating (element-wise)
        gate = torch.sigmoid(self.component_gate)  # (3, hidden_dim)
        h = h * gate.unsqueeze(0)

        # Attention pooling
        attn_scores = self.attn_q(h)  # (B, 3, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1)  # (B, hidden_dim)

        # Multi-task predictions
        dc50_pred = self.head_dc50(fused).squeeze(-1)
        dmax_pred = self.head_dmax(fused).squeeze(-1)

        # E3 adversarial classification
        if e3_adv_lambda > 0:
            h_rev = GradientReversal.apply(fused, e3_adv_lambda)
        else:
            h_rev = fused.detach()
        e3_logits = self.e3_classifier(h_rev)

        return dc50_pred, dmax_pred, {
            "fused": fused,
            "attn": attn_weights,
            "e3_logits": e3_logits,
            "e3_adv_lambda": e3_adv_lambda,
        }


# ─── 3. LOSS FUNCTIONS ────────────────────────────────
def uncertainty_weighted_loss(pred, target, log_var):
    """Multi-task loss with learnable uncertainty weighting (Kendall et al. 2018)."""
    precision = torch.exp(-log_var)
    return 0.5 * precision * F.mse_loss(pred, target, reduction="mean") + 0.5 * log_var


def e3_adversarial_loss(e3_logits, e3_labels, n_classes):
    """Cross-entropy for E3 classification (domain confusion)."""
    if e3_logits is None or e3_labels is None:
        return torch.tensor(0.0, device=DEVICE)
    # Handle cases where not all classes are in the batch
    valid = e3_labels < n_classes
    if not valid.any():
        return torch.tensor(0.0, device=DEVICE)
    return F.cross_entropy(e3_logits[valid], e3_labels[valid])


# ─── 4. TRAINING LOOP (200 epochs) ────────────────────
def train_epoch(model, loader, optimizer, scheduler, epoch, cfg, n_e3_classes):
    """Single training epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    total_dc50 = 0
    total_dmax = 0
    total_e3_adv = 0
    n_batches = 0
    opt = optimizer

    # Linearly ramp up adversarial lambda over warmup
    e3_adv_lambda = min(cfg["e3_adv_lambda"],
                        cfg["e3_adv_lambda"] * epoch / max(cfg["warmup_epochs"], 1))

    optimizer.zero_grad()

    for batch_idx, (xb, y50, ym, e3_idx) in enumerate(loader):
        xb, y50, ym, e3_idx = (xb.to(DEVICE), y50.to(DEVICE),
                                ym.to(DEVICE), e3_idx.to(DEVICE))

        # Forward
        p50, pm, aux = model(xb, e3_adv_lambda=e3_adv_lambda)

        # Multi-task uncertainty-weighted loss
        loss_dc50 = uncertainty_weighted_loss(p50, y50, model.log_var_dc50)
        loss_dmax = uncertainty_weighted_loss(pm, ym, model.log_var_dmax)
        loss_task = cfg["dc50_loss_weight"] * loss_dc50 + cfg["dmax_loss_weight"] * loss_dmax

        # E3 adversarial loss
        loss_e3 = e3_adversarial_loss(aux["e3_logits"], e3_idx, n_e3_classes)

        # Total loss
        loss = loss_task + loss_e3

        # Scale for gradient accumulation
        loss = loss / cfg["grad_accum_steps"]
        loss.backward()

        if (batch_idx + 1) % cfg["grad_accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg["grad_accum_steps"]
        total_dc50 += loss_dc50.item()
        total_dmax += loss_dmax.item()
        total_e3_adv += loss_e3.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_dc50 = total_dc50 / n_batches
    avg_dmax = total_dmax / n_batches
    avg_e3 = total_e3_adv / n_batches

    # Step scheduler (if using per-epoch scheduler, e.g., CosineAnnealingWarmRestarts)
    if hasattr(scheduler, 'step') and not isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(epoch)

    return avg_loss, avg_dc50, avg_dmax, avg_e3


@torch.no_grad()
def validate(model, loader, n_e3_classes):
    """Validation loop — returns metrics dict."""
    model.eval()
    total_loss = 0
    all_p50, all_t50, all_pm, all_tm = [], [], [], []
    n_batches = 0

    for xb, y50, ym, e3_idx in loader:
        xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
        p50, pm, aux = model(xb, e3_adv_lambda=0.0)  # no adversarial at val time

        loss_dc50 = uncertainty_weighted_loss(p50, y50, model.log_var_dc50)
        loss_dmax = uncertainty_weighted_loss(pm, ym, model.log_var_dmax)
        total_loss += (loss_dc50 + loss_dmax).item()

        all_p50.append(p50.cpu())
        all_t50.append(y50.cpu())
        all_pm.append(pm.cpu())
        all_tm.append(ym.cpu())
        n_batches += 1

    avg_loss = total_loss / n_batches

    p50 = torch.cat(all_p50); t50 = torch.cat(all_t50)
    pm  = torch.cat(all_pm);  tm  = torch.cat(all_tm)

    return {
        "loss": avg_loss,
        "dc50_R2": r2_score(t50.numpy(), p50.numpy()),
        "dc50_RMSE": float(np.sqrt(mean_squared_error(t50.numpy(), p50.numpy()))),
        "dc50_MAE": float(mean_absolute_error(t50.numpy(), p50.numpy())),
        "dmax_R2": r2_score(tm.numpy(), pm.numpy()),
        "dmax_RMSE": float(np.sqrt(mean_squared_error(tm.numpy(), pm.numpy()))),
        "dmax_MAE": float(mean_absolute_error(tm.numpy(), pm.numpy())),
        "n": len(t50),
    }


def rescaled_metrics(model, loader, scalers):
    """Evaluate on original (un-normalized) scale."""
    dc50_m, dc50_s = scalers["dc50"]
    dmax_m, dmax_s = scalers["dmax"]
    model.eval()
    all_p50, all_t50, all_pm, all_tm = [], [], [], []

    with torch.no_grad():
        for xb, y50, ym, _ in loader:
            xb = xb.to(DEVICE)
            p50, pm, _ = model(xb, e3_adv_lambda=0.0)
            all_p50.append(p50.cpu()); all_t50.append(y50.cpu())
            all_pm.append(pm.cpu());   all_tm.append(ym.cpu())

    p50 = torch.cat(all_p50) * dc50_s + dc50_m
    t50 = torch.cat(all_t50) * dc50_s + dc50_m
    pm  = torch.cat(all_pm)  * dmax_s + dmax_m
    tm  = torch.cat(all_tm)  * dmax_s + dmax_m

    return {
        "dc50_R2": float(r2_score(t50.numpy(), p50.numpy())),
        "dc50_RMSE": float(np.sqrt(mean_squared_error(t50.numpy(), p50.numpy()))),
        "dc50_MAE": float(mean_absolute_error(t50.numpy(), p50.numpy())),
        "dmax_R2": float(r2_score(tm.numpy(), pm.numpy())),
        "dmax_RMSE": float(np.sqrt(mean_squared_error(tm.numpy(), pm.numpy()))),
        "dmax_MAE": float(mean_absolute_error(tm.numpy(), pm.numpy())),
    }


# ─── 5. SPLIT STRATEGIES ──────────────────────────────
def random_split(X, meta):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=CFG["random_state"])
    tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=CFG["random_state"])
    return tr, va, te

def target_held_out_split(X, meta):
    targets = meta["Target"].values
    unique, counts = np.unique(targets, return_counts=True)
    # Hold out smaller targets (harder generalization)
    sorted_idx = np.argsort(counts)
    n_hold = max(1, len(unique) // 7)  # ~14% of targets held out
    held = set(unique[sorted_idx[:n_hold]])
    te_mask = np.array([t in held for t in targets])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=CFG["random_state"])
    print(f"  Held-out targets: {held}")
    return tr, va, te

def e3_held_out_split(X, meta):
    """Hold out ALL samples for the second most common E3 ligase."""
    e3s = meta["E3_Ligase"].values
    unique, counts = np.unique(e3s, return_counts=True)
    sorted_by_count = np.argsort(-counts)
    # Hold out one E3 — choose the one that gives substantial test set
    held = {unique[sorted_by_count[1]]}  # second most common
    te_mask = np.array([e in held for e in e3s])
    trva = np.where(~te_mask)[0]; te = np.where(te_mask)[0]
    tr, va = train_test_split(trva, test_size=0.15, random_state=CFG["random_state"])
    print(f"  Held-out E3: {held}")
    return tr, va, te

def scaffold_held_out_split(X, meta):
    """Hold out ~20% of Murcko scaffolds."""
    scaffolds = meta["Scaffold"].values
    valid_mask = np.array([sc is not None for sc in scaffolds])
    if valid_mask.sum() < 20:
        print("  ⚠ Too few valid scaffolds, falling back to random split")
        return random_split(X, meta)

    valid_idx = np.where(valid_mask)[0]
    valid_sc = [scaffolds[i] for i in valid_idx]

    sc2idx = defaultdict(list)
    for sc, i in zip(valid_sc, valid_idx):
        sc2idx[sc].append(i)

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


# ─── 6. EVALUATION ────────────────────────────────────
def evaluate_model(model, X, y50_z, ym_z, meta, tr_i, va_i, te_i, scalers, split_name):
    """Full evaluation: train, validate, test on original scale."""
    def make_loader(idx, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(y50_z[idx], dtype=torch.float32),
            torch.tensor(ym_z[idx], dtype=torch.float32),
            torch.tensor(meta.iloc[idx]["E3_idx"].values, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle)

    tr_loader = make_loader(tr_i, True)
    va_loader = make_loader(va_i)
    te_loader = make_loader(te_i)

    n_e3 = meta["E3_idx"].nunique()

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    t0 = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, l_dc50, l_dmax, l_e3 = train_epoch(
            model, tr_loader, optimizer, scheduler, epoch, CFG, n_e3)

        val_metrics = validate(model, va_loader, n_e3)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dc50_R2": val_metrics["dc50_R2"],
            "val_dmax_R2": val_metrics["dmax_R2"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} "
                  f"(DC50 R²={val_metrics['dc50_R2']:.4f}, "
                  f"Dmax R²={val_metrics['dmax_R2']:.4f}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= CFG["patience"]:
            print(f"    ⏹ Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_state)

    # Test set evaluation (rescaled)
    model.eval()
    te_metrics = rescaled_metrics(model, te_loader, scalers)

    # Per-E3 breakdown
    e3_breakdown = {}
    e3_names = meta["E3_Ligase"].unique()
    e3_to_name = dict(zip(meta["E3_idx"], meta["E3_Ligase"]))
    
    for e3_name in e3_names:
        e3_mask = meta.iloc[te_i]["E3_Ligase"].values == e3_name
        if e3_mask.sum() < 3:
            continue
        e3_idx_subset = te_i[e3_mask]
        e3_loader = make_loader(e3_idx_subset)
        e3_metrics = rescaled_metrics(model, e3_loader, scalers)
        e3_breakdown[e3_name] = {
            "n": int(e3_mask.sum()),
            "dc50_R2": e3_metrics["dc50_R2"],
            "dmax_R2": e3_metrics["dmax_R2"],
        }

    elapsed = time.time() - t0

    return {
        "split": split_name,
        "n_train": len(tr_i),
        "n_val": len(va_i),
        "n_test": len(te_i),
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
        "history": history,
    }, best_state


# ─── 7. MAIN PIPELINE ─────────────────────────────────
def main():
    print("╔" + "═" * 58 + "╗")
    print("║  WP2: Improved Transformer — E3-Generalized, Scaffold-Aware  ║")
    print("║         200 Epochs · Pre-LN · Adversarial E3 · Gating        ║")
    print("╚" + "═" * 58 + "╝\n")

    global_t0 = time.time()

    # ── Load data ──
    X, y50_z, ym_z, meta, scalers = load_data()
    input_dim = X.shape[2]
    n_e3 = meta["E3_idx"].nunique()
    print(f"\n  E3 classes for adversarial: {n_e3}")

    # ── Split strategies ──
    strategies = [
        ("Random 80/20", random_split),
        ("Target-held-out", target_held_out_split),
        ("E3-held-out", e3_held_out_split),
        ("Scaffold-held-out", scaffold_held_out_split),
    ]

    all_summaries = []
    all_detailed = {}

    for split_name, split_fn in strategies:
        print(f"\n{'=' * 58}")
        print(f"  Split: {split_name}")
        print(f"{'=' * 58}")

        trial_results = []
        trial_models = []

        for t in range(CFG["n_trials"]):
            print(f"\n  Trial {t + 1}/{CFG['n_trials']}")
            seed = CFG["random_state"] + t * 10
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Split
            tr_i, va_i, te_i = split_fn(X, meta)

            # Build model
            model = ImprovedProtacTransformer(
                input_dim=input_dim,
                hidden_dim=CFG["hidden_dim"],
                n_heads=CFG["n_heads"],
                n_layers=CFG["n_layers"],
                dropout=CFG["dropout"],
                ff_dim=CFG["ff_dim"],
                n_e3_classes=n_e3,
                e3_adv_hidden=CFG["e3_adv_hidden"],
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Model parameters: {n_params:,}")

            # Train & evaluate
            result, best_state = evaluate_model(
                model, X, y50_z, ym_z, meta, tr_i, va_i, te_i, scalers, split_name)

            result["n_params"] = n_params
            result["trial"] = t + 1
            trial_results.append(result)

            # Save checkpoint
            ckpt_path = os.path.join(MODEL_DIR, f"wp2_{split_name.replace(' ', '_')}_t{t+1}.pt")
            torch.save({
                "model_state_dict": best_state,
                "config": CFG,
                "result": {k: v for k, v in result.items() if k != "history"},
                "scalers": scalers,
            }, ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Aggregate trial results
        def avg_std(results, key):
            vals = [r[key] for r in results if np.isfinite(r.get(key, np.nan))]
            if not vals: return 0, 0
            return float(np.mean(vals)), float(np.std(vals))

        summary = {
            "split": split_name,
            "n_params": trial_results[0]["n_params"],
            "n_train": trial_results[0]["n_train"],
            "n_val": trial_results[0]["n_val"],
            "n_test": trial_results[0]["n_test"],
        }
        for metric in ["dc50_R2", "dc50_RMSE", "dc50_MAE",
                        "dmax_R2", "dmax_RMSE", "dmax_MAE"]:
            m, s = avg_std(trial_results, metric)
            summary[f"{metric}_mean"] = m
            summary[f"{metric}_std"] = s
        
        summary["epochs_trained"] = int(np.mean([r["epochs_trained"] for r in trial_results]))
        summary["elapsed_min"] = np.mean([r["elapsed_min"] for r in trial_results])

        all_summaries.append(summary)
        all_detailed[split_name] = trial_results

        # Print summary
        print(f"\n  ═══ {split_name} — Summary ═══")
        print(f"    DC50  R²={summary['dc50_R2_mean']:.4f}±{summary['dc50_R2_std']:.4f}  "
              f"RMSE={summary['dc50_RMSE_mean']:.4f}  MAE={summary['dc50_MAE_mean']:.4f}")
        print(f"    Dmax  R²={summary['dmax_R2_mean']:.4f}±{summary['dmax_R2_std']:.4f}  "
              f"RMSE={summary['dmax_RMSE_mean']:.4f}  MAE={summary['dmax_MAE_mean']:.4f}")
        print(f"    Samples: train={summary['n_train']} val={summary['n_val']} test={summary['n_test']}")
        print(f"    Epochs trained: {summary['epochs_trained']} ({summary['elapsed_min']:.1f} min)")

    # ── Master results table ──
    print(f"\n\n{'=' * 70}")
    print("  MASTER RESULTS — Improved Transformer (200 epochs)")
    print(f"{'=' * 70}")

    df = pd.DataFrame(all_summaries)
    display_cols = ["split", "n_train", "n_val", "n_test", "epochs_trained",
                    "dc50_R2_mean", "dc50_R2_std", "dc50_RMSE_mean", "dc50_RMSE_std",
                    "dmax_R2_mean", "dmax_R2_std", "dmax_RMSE_mean", "dmax_RMSE_std",
                    "elapsed_min"]
    print(df[display_cols].to_string(index=False))

    # Save master CSV
    df.to_csv(os.path.join(BM_DIR, "master_summary.csv"), index=False)

    # Save per-trial details
    detail_rows = []
    for split_name, trials in all_detailed.items():
        for t_res in trials:
            row = {"split": split_name, "trial": t_res["trial"]}
            for k in ["dc50_R2", "dc50_RMSE", "dc50_MAE",
                       "dmax_R2", "dmax_RMSE", "dmax_MAE",
                       "n_train", "n_val", "n_test", "epochs_trained", "elapsed_min"]:
                row[k] = t_res.get(k, 0)
            detail_rows.append(row)
    pd.DataFrame(detail_rows).to_csv(os.path.join(BM_DIR, "trial_details.csv"), index=False)

    # ── Save per-split JSON (with history) ──
    for split_name, trials in all_detailed.items():
        safe_name = split_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(BM_DIR, f"{safe_name}_results.json")
        # Save a copy without full history to keep file size manageable
        compact = []
        for t_res in trials:
            t = {k: v for k, v in t_res.items() if k != "history"}
            t["history"] = t_res["history"][::5]  # every 5th epoch
            compact.append(t)
        with open(path, "w") as f:
            json.dump(compact, f, indent=2, default=str)

    # ── Generate figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Training curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (split_name, trials) in enumerate(all_detailed.items()):
            ax = axes[idx // 2, idx % 2]
            t = trials[0]  # first trial
            hist = t["history"]
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["train_loss"] for h in hist],
                    label="Train Loss", alpha=0.8)
            ax.plot(epochs, [h["val_loss"] for h in hist],
                    label="Val Loss", alpha=0.8)
            ax.plot(epochs, [h["val_dc50_R2"] for h in hist],
                    label="Val DC50 R²", alpha=0.8)
            ax.plot(epochs, [h["val_dmax_R2"] for h in hist],
                    label="Val Dmax R²", alpha=0.8)
            ax.set_title(split_name)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle("WP2: Training Curves (Improved Transformer, 200 Epochs)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "wp2_training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ✅ Training curves saved")

        # 2. Generalization comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        splits = df["split"].values
        x = np.arange(len(splits))
        colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]

        for ax_idx, (target, title) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
            ax = axes[ax_idx]
            means = df[f"{target}_R2_mean"].values
            stds  = df[f"{target}_R2_std"].values
            bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels([s[:12] + ".." if len(s) > 12 else s for s in splits],
                               rotation=20, ha="right", fontsize=9)
            ax.set_title(f"{title} R² by Split Strategy")
            ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
            ax.set_ylabel("R²")
            ax.grid(alpha=0.3, axis="y")
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(v) * 0.02 + 0.01,
                        f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

        plt.suptitle("WP2: E3 & Scaffold Generalization (Improved Model)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "wp2_generalization.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Generalization plot saved")

        # 3. Comparison with old model
        old_results_path = os.path.join(REV_DIR, "benchmarking", "generalization_results.csv")
        if os.path.exists(old_results_path):
            old_df = pd.read_csv(old_results_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax_idx, (target, title) in enumerate([("dc50", "DC50"), ("dmax", "Dmax")]):
                ax = axes[ax_idx]
                old_vals = []
                new_vals = []
                labels = []
                for s in old_df["split"].values:
                    new_row = df[df["split"].str.contains(s.split(" ")[0], na=False)]
                    if new_row.empty:
                        continue
                    old_r2 = old_df[old_df["split"] == s][f"{target}_R2_mean"].values
                    new_r2 = new_row[f"{target}_R2_mean"].values
                    if len(old_r2) and len(new_r2):
                        old_vals.append(old_r2[0])
                        new_vals.append(new_r2[0])
                        labels.append(s[:14])

                x = np.arange(len(labels))
                w = 0.3
                ax.bar(x - w / 2, old_vals, w, label="Old (25 epochs)", color="#E74C3C", alpha=0.7)
                ax.bar(x + w / 2, new_vals, w, label="Improved (200 epochs)", color="#2ECC71", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                ax.set_title(f"{title} R²: Old vs Improved")
                ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3, axis="y")

            plt.suptitle("WP2: Old Model vs Improved Model", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "wp2_old_vs_new.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✅ Old vs new comparison plot saved")

    except Exception as e:
        print(f"  ⚠ Plot error: {e}")
        import traceback
        traceback.print_exc()

    total_min = (time.time() - global_t0) / 60
    print(f"\n{'=' * 58}")
    print(f"  ✅ WP2 complete. Total time: {total_min:.1f} min")
    print(f"  Results: {BM_DIR}/")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Models:  {MODEL_DIR}/")

    # Print actionable takeaways
    print(f"\n{'─' * 58}")
    print("  KEY TAKEAWAYS")
    print(f"{'─' * 58}")
    for r in all_summaries:
        dc50 = r["dc50_R2_mean"]
        dmax = r["dmax_R2_mean"]
        tag = "✅" if dc50 > 0.3 and dmax > 0.3 else ("⚠️" if dc50 > 0 else "❌")
        print(f"  {tag} {r['split']:20s}: DC50 R²={dc50:.4f}  Dmax R²={dmax:.4f}")


if __name__ == "__main__":
    main()
