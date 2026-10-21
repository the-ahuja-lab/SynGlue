#!/usr/bin/env python3
"""
SynGlue external predictions v2 — faithful to training tensor convention:
- warhead token: GROVER vec of training warhead matched by POI Uniprot
- linker token:  GROVER vec of the full external PROTAC SMILES
- E3 token:      GROVER vec of training E3 ligand matched by E3 ligase name

This mirrors how WP1a built X (per-compound W/L/E vectors).
"""
import os, sys, json, subprocess, time, warnings
warnings.filterwarnings('ignore')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

BASE = "/storage/savi/saveenas/Projects/SynGlue"
SYNGLUE_PY = "/storage/savi/saveenas/Projects/SynGlue_Py"
GROVER_DIR = f"{SYNGLUE_PY}/repos/grover"
MODEL_DIR = f"{SYNGLUE_PY}/models"
TRAIN_DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT_DIR = f"{BASE}/revision/benchmarking/tool_results"
OUT = f"{OUT_DIR}/synglue_external_predictions_v2.csv"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

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
    INPUT = f"{BASE}/revision/benchmarking/external_not_in_training.csv"
    df = pd.read_csv(INPUT, low_memory=False)
    sub = df.drop_duplicates(subset=['SMILES']).copy()
    print(f"Unique compounds: {len(sub)}")

    # ---- Load training component vectors ----
    warhead_df = pd.read_csv(f"{TRAIN_DATA}/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{TRAIN_DATA}/E3_updated_features.csv", low_memory=False)
    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    print(f"Warhead vectors: {len(warhead_df)}, E3 vectors: {len(e3_df)}")

    # Index by target uniprot / E3 ligase
    wh_by_uniprot = {}
    for _, r in warhead_df.iterrows():
        u = str(r.get('warhead_Uniprot', '')).strip().upper()
        if u and u != 'NAN':
            wh_by_uniprot.setdefault(u, r[w_cols].values.astype(np.float32))
    e3_by_name = {}
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase', '')).strip()
        if n and n != 'nan':
            e3_by_name.setdefault(n, r[e_cols].values.astype(np.float32))
    print(f"Warhead by Uniprot: {len(wh_by_uniprot)} keys")
    print(f"E3 by ligase: {len(e3_by_name)} keys: {list(e3_by_name.keys())[:10]}")

    # ---- GROVER fingerprints for external PROTACs ----
    fps_file = f"{OUT_DIR}/synglue_external_smiles_fps.npz"
    if os.path.exists(fps_file):
        fps = np.load(fps_file)['fps'][:, :4800]
        print(f"Loaded cached GROVER FPs: {fps.shape}")
    else:
        print("Extracting GROVER fingerprints...")
        batch_file = f"{OUT_DIR}/synglue_external_smiles.csv"
        pd.DataFrame({'smiles': sub['SMILES']}).to_csv(batch_file, index=False)
        features_file = f"{OUT_DIR}/synglue_external_smiles_features.npz"
        r1 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/scripts/save_features.py "
            f"--data_path {batch_file} --save_path {features_file} --features_generator rdkit_2d_normalized --restart",
            shell=True, capture_output=True, text=True, timeout=600)
        if r1.returncode != 0:
            print(f"save_features failed: {r1.stderr[-200:]}")
            sys.exit(1)
        r2 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/main.py fingerprint "
            f"--data_path {batch_file} --features_path {features_file} "
            f"--checkpoint_path {MODEL_DIR}/grover_fixed.pt --fingerprint_source both --output {fps_file}",
            shell=True, capture_output=True, text=True, timeout=1800)
        if r2.returncode != 0:
            print(f"GROVER failed: {r2.stderr[-200:]}")
            sys.exit(1)
        fps = np.load(fps_file)['fps'][:, :4800]
        print(f"FPs: {fps.shape}")

    # ---- Assemble per-compound tensors ----
    X_new = np.zeros((len(sub), 3, 4800), dtype=np.float32)
    n_wh = n_e3 = 0
    for i, (_, r) in enumerate(sub.iterrows()):
        # warhead: match by POI uniprot
        poi = str(r.get('POI_UniProt', '')).strip().upper()
        if poi in wh_by_uniprot:
            X_new[i, 0, :] = wh_by_uniprot[poi]
            n_wh += 1
        else:
            X_new[i, 0, :] = fps[i]  # fallback: use PROTAC fp
        # linker: full PROTAC
        X_new[i, 1, :] = fps[i]
        # E3: match by ligase name
        e3name = str(r.get('E3_Ligase', '')).strip()
        if e3name in e3_by_name:
            X_new[i, 2, :] = e3_by_name[e3name]
            n_e3 += 1
        else:
            X_new[i, 2, :] = fps[i]
    print(f"Warhead matched: {n_wh}/{len(sub)}, E3 matched: {n_e3}/{len(sub)}")

    # ---- Transformer forward ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
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
        pred_log_dc50, pred_dmax, _ = model(X_tensor)
        pred_log_dc50 = pred_log_dc50.cpu().numpy().flatten()
        pred_dmax = pred_dmax.cpu().numpy().flatten()
    print(f"Fused: {Z_new.shape}")

    # ---- RF (sklearn 1.2.2 in Magnet env) ----
    try:
        rf_dc50 = joblib.load(f"{MODEL_DIR}/rf_dc50.joblib")
        rf_dmax = joblib.load(f"{MODEL_DIR}/rf_dmax.joblib")
        pred_rf_dc50 = 10 ** rf_dc50.predict(Z_new)
        pred_rf_dmax = rf_dmax.predict(Z_new)
        print("RF OK")
    except Exception as e:
        print(f"RF failed: {str(e)[:100]}")
        pred_rf_dc50 = np.full(len(sub), np.nan)
        pred_rf_dmax = np.full(len(sub), np.nan)

    # Save Z for potential re-use
    np.save(f"{OUT_DIR}/synglue_external_Z_v2.npy", Z_new)

    out = sub[['SMILES', 'canonical_SMILES', 'Protein_of_Interest', 'POI_UniProt',
               'E3_Ligase', 'E3_UniProt', 'Cell_Line', 'Value_Type', 'Value',
               'Curated_Name']].copy().reset_index(drop=True)
    out['synglue_tf_log_dc50'] = pred_log_dc50
    out['synglue_tf_dc50_nM'] = 10 ** pred_log_dc50
    out['synglue_tf_dmax_pct'] = pred_dmax
    out['synglue_rf_dc50_nM'] = pred_rf_dc50
    out['synglue_rf_dmax_pct'] = pred_rf_dmax
    out.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} ({len(out)} rows)")
    print(f"DC50 TF: mean={out['synglue_tf_dc50_nM'].mean():.0f}, median={out['synglue_tf_dc50_nM'].median():.0f}")
    print(f"Dmax TF: mean={out['synglue_tf_dmax_pct'].mean():.1f}")

if __name__ == '__main__':
    main()
