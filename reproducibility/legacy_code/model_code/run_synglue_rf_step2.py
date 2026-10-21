#!/usr/bin/env python3
"""
Step 2: Load SynGlue RF models (sklearn 1.2.2, use Magnet env)
on the fused 512-dim embeddings from step 1.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
OUT_DIR = f"{BASE}/revision/benchmarking/tool_results"
SYNGLUE_PY = "/storage/savi/saveenas/Projects/SynGlue_Py"
MODEL_DIR = f"{SYNGLUE_PY}/models"

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

class MultiTaskProtacModel(nn.Module):
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

def main():
    # Load fingerprints and reassemble tensor
    fps = np.load(f"{OUT_DIR}/synglue_external_smiles_fps.npz")['fps'][:, :4800]
    print(f"FPs: {fps.shape}")

    sub = pd.read_csv(f"{OUT_DIR}/synglue_external_smiles.csv")
    print(f"Compounds: {len(sub)}")

    warhead_df = pd.read_csv(f"{SYNGLUE_PY}/data/grover_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{SYNGLUE_PY}/data/grover_e3.csv", low_memory=False)
    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    const_warhead = warhead_df[w_cols].iloc[0].values.astype(np.float32)
    const_e3 = e3_df[e_cols].iloc[0].values.astype(np.float32)

    X_new = np.zeros((len(sub), 3, 4800), dtype=np.float32)
    X_new[:, 0, :] = const_warhead
    X_new[:, 1, :] = fps
    X_new[:, 2, :] = const_e3

    device = torch.device("cpu")
    model = MultiTaskProtacModel(input_dim=4800).to(device)
    state = torch.load(f"{MODEL_DIR}/multitask_transformer.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    X_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    with torch.no_grad():
        h = model.proj(X_tensor)
        h = model.transformer(h)
        attn_scores = model.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        Z_new = (h * attn_weights).sum(dim=1).cpu().numpy()
    print(f"Fused: {Z_new.shape}")

    # RF models (Magnet env has sklearn 1.2.2)
    print("Loading RF models...")
    rf_dc50 = joblib.load(f"{MODEL_DIR}/rf_dc50.joblib")
    rf_dmax = joblib.load(f"{MODEL_DIR}/rf_dmax.joblib")
    pred_dc50 = 10 ** rf_dc50.predict(Z_new)
    pred_dmax = rf_dmax.predict(Z_new)
    print(f"RF DC50: mean={pred_dc50.mean():.1f} nM, median={np.median(pred_dc50):.1f}")
    print(f"RF Dmax: mean={pred_dmax.mean():.1f}%")

    # Merge into existing results
    out = pd.read_csv(f"{OUT_DIR}/synglue_external_predictions.csv")
    out['synglue_rf_dc50_nM'] = pred_dc50
    out['synglue_rf_dmax_pct'] = pred_dmax
    out.to_csv(f"{OUT_DIR}/synglue_external_predictions.csv", index=False)
    print(f"Saved with RF: {OUT_DIR}/synglue_external_predictions.csv")

if __name__ == '__main__':
    main()
