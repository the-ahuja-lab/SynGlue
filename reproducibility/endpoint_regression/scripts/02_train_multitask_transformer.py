#!/usr/bin/env python3
"""02_train_multitask_transformer.py  [P0]

Executed Attention-Copy1.ipynb training cells -> a clean, runnable masked multitask trainer.

Architecture (exactly as the released checkpoint):
    MultiTaskProtacModel(input_dim=4800, hidden_dim=512, n_heads=4)
      proj: Linear(4800 -> 512)
      encoder: nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True) x 2 layers
      attn_pool: Linear(512 -> 1) + softmax over the 3 component tokens
      heads: 512 -> 256 -> ReLU -> Dropout(0.2) -> 1  (DC50, Dmax)
    No positional embedding.

Training (exactly as executed):
    loss  = masked MSE, L = L_DC50 + 0.01 * L_Dmax   (non-finite targets masked out)
    optim = AdamW(lr=1e-4); batch 64 train / 128 val; up to 100 epochs;
            early stopping on validation loss, patience 15; 80/20 split, seed 42.

Inputs
  --tensor   npz/npy of shape (N, 3, 4800)          (from 01_prepare_components.py + GROVER)
  --dc50     npy of shape (N,)  log10(nM) or NaN
  --dmax     npy of shape (N,)  % or NaN

Outputs (results/)
  checkpoint_multitask.pt   trained weights
  training_log.json         epoch losses, best epoch, final metrics, config
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)


class MultiTaskProtacModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        w = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * w).sum(dim=1)
        return self.head_dc50(fused), self.head_dmax(fused), w

    def embed(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        w = torch.softmax(self.attn_pool(h), dim=1)
        return (h * w).sum(dim=1)


def masked_mse(pred, target):
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.mse_loss(pred[mask], target[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True, help="(N,3,4800) .npy or .npz with key X")
    ap.add_argument("--dc50", required=True, help="(N,) log10 DC50 (nM), NaN allowed")
    ap.add_argument("--dmax", required=True, help="(N,) Dmax (%), NaN allowed")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dmax-weight", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "checkpoint_multitask.pt"))
    args = ap.parse_args()

    def load_arr(p):
        if p.endswith(".npz"):
            z = np.load(p)
            return z[list(z.keys())[0]]
        return np.load(p)

    X = load_arr(args.tensor).astype(np.float32)
    y50 = load_arr(args.dc50).astype(np.float32)
    ydm = load_arr(args.dmax).astype(np.float32)
    assert X.shape[1:] == (3, 4800), f"unexpected tensor shape {X.shape}"
    assert len(X) == len(y50) == len(ydm)
    print(f"tensor {X.shape} | DC50 finite {int(np.isfinite(y50).sum())} | "
          f"Dmax finite {int(np.isfinite(ydm).sum())}")

    idx = np.random.RandomState(C.SEED).permutation(len(X))
    n_val = int(0.2 * len(X))
    va, tr = idx[:n_val], idx[n_val:]

    def loader(ix, bs, shuffle):
        ds = TensorDataset(torch.tensor(X[ix]), torch.tensor(y50[ix]), torch.tensor(ydm[ix]))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    tr_ld, va_ld = loader(tr, args.batch, True), loader(va, args.val_batch, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskProtacModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={dev} | params={sum(p.numel() for p in model.parameters()):,}")

    best, best_state, counter, log = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for xb, b50, bm in tr_ld:
            xb, b50, bm = xb.to(dev), b50.to(dev), bm.to(dev)
            opt.zero_grad()
            p50, pm, _ = model(xb)
            loss = masked_mse(p50.squeeze(-1), b50) + args.dmax_weight * masked_mse(pm.squeeze(-1), bm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, b50, bm in va_ld:
                xb, b50, bm = xb.to(dev), b50.to(dev), bm.to(dev)
                p50, pm, _ = model(xb)
                vl += (masked_mse(p50.squeeze(-1), b50) +
                       args.dmax_weight * masked_mse(pm.squeeze(-1), bm)).item() * len(xb)
        tl /= len(tr); vl /= len(va)
        log.append(dict(epoch=epoch, train_loss=tl, val_loss=vl))
        print(f"epoch {epoch:03d} | train {tl:.4f} | val {vl:.4f}")
        if vl < best:
            best, best_state, counter = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"early stop at epoch {epoch} (patience {args.patience})")
                break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out)
    C.save_json(dict(config=vars(args), best_val_loss=best, epochs_run=len(log),
                     minutes=round((time.time() - t0) / 60, 2), log=log), "training_log.json")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
