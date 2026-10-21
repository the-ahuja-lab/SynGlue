#!/usr/bin/env python3
"""
Build v4 EXTERNAL encoding + EXTENDED training data (diversity).

Part A — External v4 encoding:
  For external compounds with a valid decomposed linker (1,056):
    L = decomposed linker GROVER (true linker)
  Fallback (no match): L = full-PROTAC GROVER
  W/E matched as before. PCA transform via v4 pca.

Part B — Extended training (diversity):
  Add PROTAC_DB_with_features rows NOT in current training with DC50/Dmax labels.
  Decompose W/L/E via substructure matching against warhead/E3 libraries,
  extract linker, compute GROVER.
  Output combined metadata + full-dim arrays for PCA retrain.
"""
import os, sys, json, subprocess, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
GROVER_DIR = f"{BASE}/../SynGlue_Py/repos/grover"
MODEL_DIR = f"{BASE}/../SynGlue_Py/models"
V3_DIR = f"{REV}/benchmarking/generalization_data_v3"
V4_DIR = f"{REV}/benchmarking/generalization_data_v4"
os.makedirs(V4_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon(s):
    if not isinstance(s, str) or not s: return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def _parse(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.", "N/D", "NA", "N/A", ""): return np.nan
    if s.startswith(">"): s = s[1:]
    elif s.startswith("<"): s = s[1:]
    vals = []
    for p in s.split("/"):
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", ""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

def main():
    pca = joblib.load(f"{V4_DIR}/pca_512.joblib")

    # ========== PART A: External v4 encoding ==========
    print("=== PART A: External v4 encoding ===")
    fps_full = np.load(f"{REV}/benchmarking/tool_results/synglue_external_smiles_fps.npz")['fps'][:, :4800]
    ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    ext = ext.drop_duplicates(subset=['SMILES']).copy()

    # Decomposed linkers
    dec = pd.read_csv(f"{V4_DIR}/external_decomposed_valid.csv")
    linker_fps = np.load(f"{V4_DIR}/linker_fps.npy")
    # Map: canonical SMILES -> linker fp index
    dec['canon'] = dec['SMILES'].apply(canon)
    fp_map = {c: i for i, c in enumerate(dec['canon'])}

    wh_df = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)
    w_cols = [c for c in wh_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    wh_by_u, e3_by_n = {}, {}
    for _, r in wh_df.iterrows():
        u = str(r.get('warhead_Uniprot','')).strip().upper()
        if u and u != 'NAN': wh_by_u.setdefault(u, r[w_cols].values.astype(np.float32))
    for _, r in e3_df.iterrows():
        n = str(r.get('E3_Ligase','')).strip().upper()
        if n and n != 'NAN': e3_by_n.setdefault(n, r[e_cols].values.astype(np.float32))

    X_ext_full = np.zeros((len(ext), 3, 4800), dtype=np.float32)
    n_true = 0
    ext['has_true_linker'] = False
    for i, (_, r) in enumerate(ext.iterrows()):
        c = canon(r['SMILES'])
        poi = str(r.get('POI_UniProt','')).strip().upper()
        en = str(r.get('E3_Ligase','')).strip().upper()
        X_ext_full[i, 0, :] = wh_by_u.get(poi, fps_full[i])
        if c in fp_map:
            X_ext_full[i, 1, :] = linker_fps[fp_map[c]]   # TRUE linker
            ext.loc[ext.index[i], 'has_true_linker'] = True
            n_true += 1
        else:
            X_ext_full[i, 1, :] = fps_full[i]              # fallback full PROTAC
        X_ext_full[i, 2, :] = e3_by_n.get(en, fps_full[i])
    print(f"External: {len(ext)} compounds, {n_true} with true decomposed linker")
    X_ext_pca = pca.transform(X_ext_full.reshape(-1, 4800)).reshape(len(ext), 3, 512).astype(np.float32)
    np.save(f"{V4_DIR}/X_external_pca.npy", X_ext_pca)
    np.save(f"{V4_DIR}/X_external_full.npy", X_ext_full)
    ext.to_csv(f"{V4_DIR}/external_meta.csv", index=False)
    print(f"Saved external encoding: {X_ext_pca.shape}")

    # ========== PART B: Extended training data ==========
    print("\n=== PART B: Extended training (diversity) ===")
    pdb = pd.read_csv(f"{DATA_DIR}/PROTAC_DB_with_features.csv", low_memory=False)
    upd = pd.read_csv(f"{DATA_DIR}/Upadted_PROTAC_features.csv", low_memory=False,
                      usecols=['PROTAC_SMILES'])
    s_upd = set(upd['PROTAC_SMILES'].dropna().apply(canon))
    extra = pdb[~pdb['Smiles'].apply(canon).isin(s_upd)].copy()
    print(f"PROTAC_DB extra rows: {len(extra)}")
    extra['DC50'] = extra['DC50 (nM)'].apply(_parse)
    extra['Dmax'] = extra['Dmax (%)'].apply(_parse)
    labeled = extra[(extra['DC50'].notna()) | (extra['Dmax'].notna())]
    print(f"  with labels: {len(labeled)}")

    if len(labeled) > 0:
        # Decompose via substructure matching
        wh_mols = [(s, Chem.MolFromSmiles(s)) for s in pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv")['warhead_Canonical_SMILES'].dropna() if Chem.MolFromSmiles(s)]
        e3_mols = [(s, Chem.MolFromSmiles(s)) for s in pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv")['E3_Canonical_SMILES'].dropna() if Chem.MolFromSmiles(s)]
        wh_by_smi = {}
        for _, r in pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False).iterrows():
            cs = canon(r.get('warhead_Canonical_SMILES',''))
            if cs: wh_by_smi.setdefault(cs, r[w_cols].values.astype(np.float32))
        e3_by_smi = {}
        for _, r in pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False).iterrows():
            cs = canon(r.get('E3_Canonical_SMILES',''))
            if cs: e3_by_smi.setdefault(cs, r[e_cols].values.astype(np.float32))

        rows = []
        for _, p in labeled.iterrows():
            mol = Chem.MolFromSmiles(p['Smiles'])
            if mol is None: continue
            best_wh, best_e3 = None, None
            for s, wmol in wh_mols:
                if wmol.GetNumAtoms() <= mol.GetNumAtoms() and mol.HasSubstructMatch(wmol):
                    if best_wh is None or wmol.GetNumAtoms() > Chem.MolFromSmiles(best_wh).GetNumAtoms():
                        best_wh = s
            for s, emol in e3_mols:
                if emol.GetNumAtoms() <= mol.GetNumAtoms() and mol.HasSubstructMatch(emol):
                    if best_e3 is None or emol.GetNumAtoms() > Chem.MolFromSmiles(best_e3).GetNumAtoms():
                        best_e3 = s
            if best_wh is None or best_e3 is None: continue
            rw = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(best_wh))
            rwe = Chem.DeleteSubstructs(rw, Chem.MolFromSmiles(best_e3))
            linker_smi = Chem.MolToSmiles(rwe) if rwe and rwe.GetNumAtoms() > 0 else None
            if linker_smi is None: continue
            rows.append({"W": wh_by_smi.get(canon(best_wh)), "E": e3_by_smi.get(canon(best_e3)),
                         "linker_smi": linker_smi, "DC50": p['DC50'], "Dmax": p['Dmax'],
                         "SMILES": p['Smiles'], "target_uniprot": str(p.get('Uniprot','')).strip().upper(),
                         "target_name": p.get('Target',''), "e3": str(p.get('E3 ligase','')).strip().upper(),
                         "compound_id": p.get('Compound ID','')})
        rows = [r for r in rows if r['W'] is not None and r['E'] is not None]
        print(f"  Decomposed with W/E matched: {len(rows)}")

        if len(rows) > 0:
            # GROVER for linkers
            linkers = [r['linker_smi'] for r in rows]
            smi_file = f"{V4_DIR}/extra_linker_smiles.csv"
            pd.DataFrame({'smiles': linkers}).to_csv(smi_file, index=False)
            feats = f"{V4_DIR}/extra_linker_features.npz"; fps_file = f"{V4_DIR}/extra_linker_fps.npz"
            env = os.environ.copy(); env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
            r1 = subprocess.run(
                f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/scripts/save_features.py "
                f"--data_path {smi_file} --save_path {feats} --features_generator rdkit_2d_normalized --restart",
                shell=True, capture_output=True, text=True, timeout=1800, env=env)
            if r1.returncode != 0:
                print(f"  save_features failed: {r1.stderr[-200:]}"); sys.exit(1)
            r2 = subprocess.run(
                f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/main.py fingerprint "
                f"--data_path {smi_file} --features_path {feats} "
                f"--checkpoint_path {MODEL_DIR}/grover_fixed.pt --fingerprint_source both --output {fps_file}",
                shell=True, capture_output=True, text=True, timeout=3600, env=env)
            if r2.returncode != 0:
                print(f"  GROVER failed: {r2.stderr[-200:]}"); sys.exit(1)
            linker_fps_extra = np.load(fps_file)['fps'][:, :4800]
            print(f"  Extra linker fps: {linker_fps_extra.shape}")

            # Save extra full-dim X + meta
            for task, key in [("dc50", "DC50"), ("dmax", "Dmax")]:
                keep = [(i, r) for i, r in enumerate(rows) if pd.notna(r[key])]
                if not keep: continue
                X = np.zeros((len(keep), 3, 4800), dtype=np.float32)
                for j, (i, r) in enumerate(keep):
                    X[j, 0, :] = r['W']; X[j, 1, :] = linker_fps_extra[i]; X[j, 2, :] = r['E']
                y = np.array([r[key] for _, r in keep], dtype=np.float32)
                if task == 'dc50':
                    y = np.log10(np.clip(y, 1e-3, 1e5)).astype(np.float32)
                else:
                    y = np.clip(y, 0, 100).astype(np.float32)
                meta = pd.DataFrame([{**{k: r[k] for k in ('SMILES','target_uniprot','target_name','e3','compound_id')},
                                     **{key: r[key]}} for _, r in keep])
                np.save(f"{V4_DIR}/X_{task}_extra_full.npy", X)
                np.save(f"{V4_DIR}/y_{task}_extra.npy", y)
                meta.to_csv(f"{V4_DIR}/metadata_{task}_extra.csv", index=False)
                print(f"  {task} extra: X={X.shape}")

    print("\nPart A+B complete. Files in", V4_DIR)

if __name__ == '__main__':
    main()
