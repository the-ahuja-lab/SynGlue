#!/usr/bin/env python3
"""
SynGlue PROTAC Predictor — component-aware transformer inference.

Usage:
    python predict_protac.py "SMILES" [--e3 VHL|CRBN|...] [--target P15170] [--csv file.csv] [--out out.csv]

Takes a FULL PROTAC SMILES, decomposes it into warhead/linker/E3 (substructure
matching against the training component libraries), computes GROVER embeddings,
PCA-transforms, and predicts DC50 (nM) and Dmax (%) via the generalized
component-aware transformer ensemble (v4).

Outputs:
  - DC50 (nM), Dmax (%) per split model + ensemble mean ± std
  - component decomposition report (warhead/E3/linker SMILES, match sizes)
"""
import os, sys, json, warnings, subprocess, tempfile, argparse, glob
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = f"{REV}/benchmarking/generalization_data_v4"
MODEL_DIR = f"{REV}/experiments/gen_training_v4"
SYNGLUE_PY = f"{BASE}/../SynGlue_Py"
GROVER_DIR = f"{SYNGLUE_PY}/repos/grover"
GROVER_CKPT = f"{SYNGLUE_PY}/models/grover_fixed.pt"
TRAIN_DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model (must match train_generalization_v4.py) ──
class CAT(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, n_heads=8, n_layers=2, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(3, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, x):
        h = self.proj(x)
        h = h + self.pos_embed(torch.arange(3, device=x.device))
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return self.head((h * attn).sum(dim=1)).squeeze(-1)

# ── Loaded lazily ──
_LIBS = None
def get_libs():
    global _LIBS
    if _LIBS is not None:
        return _LIBS
    wh = pd.read_csv(f"{TRAIN_DATA}/Updated_warhead.csv", low_memory=False)
    e3 = pd.read_csv(f"{TRAIN_DATA}/E3_updated_features.csv", low_memory=False)
    linker = pd.read_csv(f"{TRAIN_DATA}/linker_updated_features.csv", low_memory=False)
    w_cols = [c for c in wh.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3.columns if c.startswith("Grover_")]
    l_cols = [c for c in linker.columns if c.startswith("Grover_")]

    # libraries of (SMILES, embedding, name)
    wh_lib = []
    for _, r in wh.iterrows():
        s = str(r.get('warhead_Canonical_SMILES', ''))
        if pd.notna(s) and s and s != 'nan':
            wh_lib.append((s, r[w_cols].values.astype(np.float32),
                           str(r.get('warhead_Target', '')), str(r.get('warhead_Uniprot', ''))))
    e3_lib = []
    for _, r in e3.iterrows():
        s = str(r.get('E3_Canonical_SMILES', ''))
        if pd.notna(s) and s and s != 'nan':
            e3_lib.append((s, r[e_cols].values.astype(np.float32), str(r.get('E3_Ligase', ''))))
    linker_by_id = {}
    for _, r in linker.iterrows():
        lid = r.get('linker_Compound_ID', None)
        if pd.notna(lid):
            linker_by_id[str(int(lid))] = (str(r.get('linker_Smiles','')), r[l_cols].values.astype(np.float32))

    _LIBS = {"wh": wh_lib, "e3": e3_lib, "linker_by_id": linker_by_id}
    return _LIBS

def grover_fp(smiles_list):
    """Compute GROVER 4800-dim fingerprints for SMILES via subprocess (synglu env)."""
    if not smiles_list:
        return np.zeros((0, 4800), dtype=np.float32)
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        f.write("smiles\n" + "\n".join(smiles_list))
        smi_file = f.name
    feats = smi_file.replace('.csv', '_feat.npz')
    out = smi_file.replace('.csv', '_fps.npz')
    env = os.environ.copy()
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    PY = "/home/saveenas/miniconda3/envs/synglu_test_package/bin/python"
    r1 = subprocess.run(
        f"{PY} {GROVER_DIR}/scripts/save_features.py --data_path {smi_file} --save_path {feats} "
        f"--features_generator rdkit_2d_normalized --restart",
        shell=True, capture_output=True, text=True, timeout=600, env=env)
    if r1.returncode != 0:
        for f in [smi_file, feats, out]:
            if os.path.exists(f): os.unlink(f)
        raise RuntimeError(f"save_features failed: {r1.stderr[-200:]}")
    r2 = subprocess.run(
        f"{PY} {GROVER_DIR}/main.py fingerprint --data_path {smi_file} --features_path {feats} "
        f"--checkpoint_path {GROVER_CKPT} --fingerprint_source both --output {out}",
        shell=True, capture_output=True, text=True, timeout=1800, env=env)
    if r2.returncode != 0:
        for f in [smi_file, feats, out]:
            if os.path.exists(f): os.unlink(f)
        raise RuntimeError(f"GROVER failed: {r2.stderr[-200:]}")
    fps = np.load(out)['fps'][:, :4800]
    for f in [smi_file, feats, out]:
        if os.path.exists(f): os.unlink(f)
    return fps.astype(np.float32)

def libs_wh_with_matches(mol):
    """Yield warhead library entries with their substructure match sets."""
    libs = get_libs()
    out = []
    for s, emb, target, unip in libs['wh']:
        wm = Chem.MolFromSmiles(s)
        if wm is None or wm.GetNumAtoms() > mol.GetNumAtoms():
            continue
        m = mol.GetSubstructMatch(wm)
        if m:
            out.append((s, emb, target, unip, set(m)))
    return out


def decompose_protac(smiles):
    """Decompose a full PROTAC into warhead/E3/linker using training libraries."""
    libs = get_libs()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "invalid", "smiles": smiles}
    n_atoms = mol.GetNumAtoms()

    # match warhead (largest substructure) — must NOT overlap the E3 match atoms
    best_w = None
    for s, emb, target, unip in libs['wh']:
        wm = Chem.MolFromSmiles(s)
        if wm is None or wm.GetNumAtoms() > n_atoms: continue
        wmatch = mol.GetSubstructMatch(wm)
        if not wmatch: continue
        if best_w is None or wm.GetNumAtoms() > Chem.MolFromSmiles(best_w[0]).GetNumAtoms():
            best_w = (s, emb, target, unip, set(wmatch))
    # match E3
    best_e = None
    for s, emb, lig in libs['e3']:
        em = Chem.MolFromSmiles(s)
        if em is None or em.GetNumAtoms() > n_atoms: continue
        ematch = mol.GetSubstructMatch(em)
        if not ematch: continue
        if best_e is None or em.GetNumAtoms() > Chem.MolFromSmiles(best_e[0]).GetNumAtoms():
            best_e = (s, emb, lig, set(ematch))

    # reject overlapping warhead/E3 matches; also exclude warheads targeting the matched E3
    # (e.g., VHL-ligand-as-warhead entries should not be used when E3 is VHL)
    e3_family = best_e[2].upper() if best_e is not None else ''
    if best_w is not None and best_e is not None:
        overlap = bool(best_w[4] & best_e[3])
        w_targets_e3 = e3_family and (e3_family in best_w[2].upper() or e3_family in best_w[3].upper())
        if overlap or w_targets_e3:
            for s, emb, target, unip, wmatch_set in sorted(libs_wh_with_matches(mol), key=lambda x: -x[4].__len__()):
                w_targets_e3_i = e3_family and e3_family in str(target).upper()
                if not (wmatch_set & best_e[3]) and not w_targets_e3_i:
                    best_w = (s, emb, target, unip, wmatch_set)
                    break

    if best_w is None or best_e is None:
        return {"status": "no_match", "smiles": smiles,
                "warhead_found": best_w is not None, "e3_found": best_e is not None}

    # linker = remainder after removing warhead + E3
    try:
        rw = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(best_w[0]))
        rwe = Chem.DeleteSubstructs(rw, Chem.MolFromSmiles(best_e[0]))
        linker_smi = Chem.MolToSmiles(rwe) if rwe and rwe.GetNumAtoms() > 0 else None
    except Exception:
        linker_smi = None

    return {"status": "ok", "smiles": smiles,
            "warhead_smiles": best_w[0], "warhead_target": best_w[2], "warhead_uniprot": best_w[3],
            "warhead_emb": best_w[1], "warhead_atoms": Chem.MolFromSmiles(best_w[0]).GetNumAtoms(),
            "warhead_reliable": not (e3_family and e3_family in best_w[2].upper()),
            "e3_smiles": best_e[0], "e3_ligase": best_e[2], "e3_emb": best_e[1],
            "e3_atoms": Chem.MolFromSmiles(best_e[0]).GetNumAtoms(),
            "linker_smiles": linker_smi,
            "linker_atoms": (Chem.MolFromSmiles(linker_smi).GetNumAtoms() if linker_smi and Chem.MolFromSmiles(linker_smi) else 0)}

def build_tensor(decomp, pca):
    """Build (3, 512) tensor from decomposition. Linker embedding computed via GROVER."""
    if decomp['status'] != 'ok':
        return None
    if decomp['linker_smiles'] is None:
        return None
    lfp = grover_fp([decomp['linker_smiles']])
    if lfp.shape[0] == 0:
        return None
    X = np.zeros((1, 3, 4800), dtype=np.float32)
    X[0, 0, :] = decomp['warhead_emb']
    X[0, 1, :] = lfp[0]
    X[0, 2, :] = decomp['e3_emb']
    Xp = pca.transform(X.reshape(-1, 4800)).reshape(1, 3, 512).astype(np.float32)
    return Xp


def build_tensor_whole(smiles, pca, e3_hint=None):
    """Build (3,512) tensor using v3 convention: full-PROTAC as L token.
    W = first-row training warhead vector (v3 convention), E = matched or first E3."""
    libs = get_libs()
    wh = pd.read_csv(f"{TRAIN_DATA}/Updated_warhead.csv", low_memory=False)
    e3df = pd.read_csv(f"{TRAIN_DATA}/E3_updated_features.csv", low_memory=False)
    w_cols = [c for c in wh.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3df.columns if c.startswith("Grover_")]
    wv = wh[w_cols].iloc[0].values.astype(np.float32)
    if e3_hint:
        e3_row = e3df[e3df['E3_Ligase'].str.upper() == str(e3_hint).upper()]
        if len(e3_row) == 0:
            e3v = e3df[e_cols].iloc[0].values.astype(np.float32)
        else:
            e3v = e3_row.iloc[0][e_cols].values.astype(np.float32)
    else:
        e3v = e3df[e_cols].iloc[0].values.astype(np.float32)
    fp = grover_fp([smiles])
    if fp.shape[0] == 0:
        return None
    X = np.zeros((1, 3, 4800), dtype=np.float32)
    X[0, 0, :] = wv
    X[0, 1, :] = fp[0]
    X[0, 2, :] = e3v
    return pca.transform(X.reshape(-1, 4800)).reshape(1, 3, 512).astype(np.float32)

def predict(Xp, task, scaler):
    """Predict with 4 split models + ensemble. Returns (per-model preds, mean, std)."""
    preds = []
    for split in ["random", "scaffold_held_out", "e3_held_out", "target_held_out"]:
        mp = f"{MODEL_DIR}/model_{task}_{split}.pt"
        if not os.path.exists(mp):
            continue
        model = CAT().to(DEVICE)
        model.load_state_dict(torch.load(mp, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            yp = model(torch.tensor(Xp, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        m, s = scaler
        preds.append(float((yp[0] * s + m)))
    preds = np.array(preds)
    return preds, float(preds.mean()), float(preds.std())

def predict_one(smiles, e3_hint=None, target_hint=None, verbose=True):
    pca = joblib.load(f"{DATA}/pca_512.joblib")
    decomp = decompose_protac(smiles)
    if verbose:
        print(f"Decomposition status: {decomp['status']}")
        if decomp['status'] == 'ok':
            print(f"  warhead ({decomp['warhead_atoms']} atoms, {decomp['warhead_target']}/{decomp['warhead_uniprot']}): {decomp['warhead_smiles']}")
            print(f"  E3 ligand ({decomp['e3_atoms']} atoms, {decomp['e3_ligase']}): {decomp['e3_smiles']}")
            print(f"  linker ({decomp['linker_atoms']} atoms): {decomp['linker_smiles']}")

    # scalers (same for v3/v4 — identical label transforms)
    y50 = np.load(f"{DATA}/y_dc50.npy")
    sc50 = (float(y50.mean()), float(y50.std()))
    y95 = np.load(f"{DATA}/y_dmax.npy")
    sc95 = (float(y95.mean()), float(y95.std()))

    # Mode 1: v4 decomposed (requires reliable warhead + E3 library match + linker extraction)
    Xp = build_tensor(decomp, pca) if decomp['status'] == 'ok' and decomp.get('warhead_reliable', False) else None
    if Xp is not None:
        p50, m50, s50 = predict(Xp, "dc50", sc50)
        p95, m95, s95 = predict(Xp, "dmax", sc95)
        mode = "decomposed (v4)"
    else:
        # Mode 2: v3 whole-PROTAC (full PROTAC as linker token)
        Xw = build_tensor_whole(smiles, pca, e3_hint)
        if Xw is None:
            return {"status": "embedding_failed", "smiles": smiles, **decomp}
        p50, m50, s50 = predict(Xw, "dc50", sc50)
        p95, m95, s95 = predict(Xw, "dmax", sc95)
        mode = "whole-PROTAC (v3)"

    result = {
        "status": "ok", "smiles": smiles, "mode": mode,
        "DC50_nM_ensemble": float(10 ** m50),
        "DC50_log10_ensemble": float(m50),
        "DC50_nM_std": float(10 ** (m50 + s50) - 10 ** m50) / 2,
        "DC50_nM_per_model": [float(10 ** p) for p in p50],
        "Dmax_pct_ensemble": float(m95),
        "Dmax_pct_std": float(s95),
        "Dmax_pct_per_model": [float(p) for p in p95],
    }
    if decomp['status'] == 'ok':
        result.update({k: decomp[k] for k in ["warhead_smiles", "e3_smiles", "linker_smiles", "e3_ligase", "warhead_target"]})
    if verbose:
        print(f"\nMode: {mode}")
        print(f"Predicted DC50 = {10**m50:.1f} nM  (log10 = {m50:.3f} ± {s50:.3f})")
        print(f"Predicted Dmax = {m95:.1f}%  (± {s95:.1f})")
        print(f"Per-model DC50 (nM): {[f'{10**p:.1f}' for p in p50]}")
        print(f"Per-model Dmax (%): {[f'{p:.1f}' for p in p95]}")
    return result

def predict_csv(csv_path, smiles_col, out_path, e3_col=None, target_col=None):
    df = pd.read_csv(csv_path, low_memory=False)
    rows = []
    for _, r in df.iterrows():
        smi = r[smiles_col]
        hint = r[e3_col] if (e3_col and e3_col in df.columns) else None
        res = predict_one(smi, e3_hint=hint, verbose=False)
        if res.get('status') != 'ok':
            rows.append({"SMILES": smi, "status": res['status']})
            continue
        rows.append({"SMILES": smi, "status": "ok", "mode": res['mode'],
                     "DC50_nM": res['DC50_nM_ensemble'], "DC50_log10": res['DC50_log10_ensemble'],
                     "Dmax_pct": res['Dmax_pct_ensemble'], "Dmax_std": res['Dmax_pct_std'],
                     "warhead": res.get('warhead_smiles'), "E3_ligase": res.get('e3_ligase'),
                     "linker": res.get('linker_smiles'), "warhead_target": res.get('warhead_target')})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions → {out_path}")
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="SynGlue PROTAC predictor (component-aware transformer)")
    ap.add_argument("smiles", nargs="?", help="Full PROTAC SMILES")
    ap.add_argument("--e3", default=None, help="E3 ligase hint (CRBN/VHL/...) for whole-PROTAC mode")
    ap.add_argument("--target", default=None, help="Target UniProt hint")
    ap.add_argument("--csv", help="CSV file with SMILES column for batch")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--e3-col", default=None, help="Column name for E3 hint in CSV mode")
    ap.add_argument("--out", default=None, help="Output CSV for batch mode")
    args = ap.parse_args()

    if args.csv:
        out = args.out or args.csv.replace('.csv', '_synglue_pred.csv')
        predict_csv(args.csv, args.smiles_col, out, e3_col=args.e3_col)
    elif args.smiles:
        predict_one(args.smiles, e3_hint=args.e3)
    else:
        ap.print_help()
