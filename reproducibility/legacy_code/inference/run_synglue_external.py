#!/usr/bin/env python3
"""
Run SynGlue (MultiTask Transformer + RF) on external benchmark compounds
NOT in training data.

Pipeline:
1. GROVER fingerprint extraction for each compound SMILES
2. Assemble 3-token tensor [warhead_vec, linker_fp, e3_vec] (using training data
   constant vectors per the module_4.py convention)
3. Transformer forward → fused 512-dim
4. RF DC50/Dmax prediction
"""
import os, sys, json, subprocess, time, warnings
warnings.filterwarnings('ignore')
# Allow legacy torch.load (argparse.Namespace in GROVER checkpoint)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

BASE = "/storage/savi/saveenas/Projects/SynGlue"
SYNGLUE_PY = "/storage/savi/saveenas/Projects/SynGlue_Py"
GROVER_DIR = f"{SYNGLUE_PY}/repos/grover"
MODEL_DIR = f"{SYNGLUE_PY}/models"
DATA_DIR = f"{SYNGLUE_PY}/data"
OUT_DIR = f"{BASE}/revision/benchmarking/tool_results"
OUT = f"{OUT_DIR}/synglue_external_predictions.csv"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ---- MultiTaskProtacModel (mirror of module_4.py, no pos_embed for now) ----
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


def extract_grover_fps(smiles_list, batch_smiles_file):
    """Run GROVER fingerprint extraction on a list of SMILES."""
    features_file = batch_smiles_file.replace('.csv', '_features.npz')
    fingerprint_file = batch_smiles_file.replace('.csv', '_fps.npz')

    cmd1 = (
        f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/scripts/save_features.py "
        f"--data_path {batch_smiles_file} --save_path {features_file} "
        f"--features_generator rdkit_2d_normalized --restart"
    )
    r1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=600)
    if r1.returncode != 0:
        print(f"  save_features failed: {r1.stderr[-300:]}")
        return None

    cmd2 = (
        f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/main.py fingerprint "
        f"--data_path {batch_smiles_file} --features_path {features_file} "
        f"--checkpoint_path {MODEL_DIR}/grover_fixed.pt --fingerprint_source both "
        f"--output {fingerprint_file}"
    )
    r2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=1800)
    if r2.returncode != 0:
        print(f"  GROVER fingerprint failed: {r2.stderr[-300:]}")
        return None

    fps = np.load(fingerprint_file)['fps'][:, :4800]
    return fps


def main():
    INPUT = f"{BASE}/revision/benchmarking/external_not_in_training.csv"
    df = pd.read_csv(INPUT, low_memory=False)

    # Deduplicate to unique SMILES
    sub = df.drop_duplicates(subset=['SMILES']).copy()
    print(f"Unique compounds: {len(sub)}")

    # Load constant warhead/E3 vectors (module_4.py convention: first row of training CSVs)
    warhead_df = pd.read_csv(f"{DATA_DIR}/grover_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{DATA_DIR}/grover_e3.csv", low_memory=False)
    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    print(f"Warhead cols: {len(w_cols)}, E3 cols: {len(e_cols)}")
    const_warhead = warhead_df[w_cols].iloc[0].values.astype(np.float32)
    const_e3 = e3_df[e_cols].iloc[0].values.astype(np.float32)
    print(f"Const warhead vec shape: {const_warhead.shape}")

    # ---- GROVER fingerprints ----
    os.makedirs(OUT_DIR, exist_ok=True)
    batch_file = f"{OUT_DIR}/synglue_external_smiles.csv"
    pd.DataFrame({'smiles': sub['SMILES']}).to_csv(batch_file, index=False)

    print("Extracting GROVER fingerprints...")
    t0 = time.time()
    fps = extract_grover_fps(sub['SMILES'].tolist(), batch_file)
    if fps is None:
        print("GROVER extraction failed — aborting")
        sys.exit(1)
    print(f"  FPs: {fps.shape} ({time.time()-t0:.0f}s)")

    # ---- Assemble 3-token tensor ----
    X_new = np.zeros((len(sub), 3, 4800), dtype=np.float32)
    X_new[:, 0, :] = const_warhead
    X_new[:, 1, :] = fps
    X_new[:, 2, :] = const_e3
    print(f"Tensor: {X_new.shape}")

    # ---- Transformer forward ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = MultiTaskProtacModel(input_dim=4800).to(device)
    # Load with weights_only (handles older pickles)
    state = torch.load(f"{MODEL_DIR}/multitask_transformer.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    X_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    with torch.no_grad():
        h = model.proj(X_tensor)
        h = model.transformer(h)
        attn_scores = model.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        Z_new = (h * attn_weights).sum(dim=1).cpu().numpy()
        pred_log_dc50, pred_dmax, _ = model(X_tensor)
        pred_log_dc50 = pred_log_dc50.cpu().numpy().flatten()
        pred_dmax = pred_dmax.cpu().numpy().flatten()
    print(f"Fused embeddings: {Z_new.shape}")

    # ---- RF models (may need sklearn compat) ----
    try:
        rf_dc50 = joblib.load(f"{MODEL_DIR}/rf_dc50.joblib")
        rf_dmax = joblib.load(f"{MODEL_DIR}/rf_dmax.joblib")
        pred_rf_dc50 = 10 ** rf_dc50.predict(Z_new)
        pred_rf_dmax = rf_dmax.predict(Z_new)
        rf_ok = True
    except Exception as e:
        print(f"RF load failed ({str(e)[:80]}) — using transformer head predictions")
        pred_rf_dc50 = np.full(len(sub), np.nan)
        pred_rf_dmax = np.full(len(sub), np.nan)
        rf_ok = False

    # ---- Compile output ----
    out = sub[['SMILES', 'canonical_SMILES', 'Protein_of_Interest', 'POI_UniProt',
               'E3_Ligase', 'E3_UniProt', 'Cell_Line', 'Value_Type', 'Value',
               'Curated_Name']].copy().reset_index(drop=True)
    out['synglue_tf_log_dc50'] = pred_log_dc50
    out['synglue_tf_dc50_nM'] = 10 ** pred_log_dc50
    out['synglue_tf_dmax_pct'] = pred_dmax
    if rf_ok:
        out['synglue_rf_dc50_nM'] = pred_rf_dc50
        out['synglue_rf_dmax_pct'] = pred_rf_dmax
    out.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")
    print(f"Rows: {len(out)}")
    print(f"\nDC50 (transformer): mean={out['synglue_tf_dc50_nM'].mean():.1f} nM, median={out['synglue_tf_dc50_nM'].median():.1f} nM")
    print(f"Dmax (transformer): mean={out['synglue_tf_dmax_pct'].mean():.1f}%")

if __name__ == '__main__':
    main()
