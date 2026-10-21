#!/usr/bin/env python3
"""
True W/L/E decomposition of external PROTACs via substructure matching.

For each external PROTAC:
1. Match warhead SMILES (from training Updated_warhead.csv) as substructure
2. Match E3 ligand SMILES (from training E3_updated_features.csv) as substructure
3. Linker = remainder of the molecule after removing matched warhead + E3 atoms
4. Compute GROVER embeddings for warhead (training), true linker, E3 (training)

Output: external_decomposed.csv + GROVER fps for true linkers
"""
import os, sys, json, subprocess, time, warnings
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
GROVER_DIR = f"{BASE}/../SynGlue_Py/repos/grover"
MODEL_DIR = f"{BASE}/../SynGlue_Py/models"
OUT_DIR = f"{REV}/benchmarking/generalization_data_v4"
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon(s):
    if not isinstance(s, str) or not s: return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def main():
    # Load component libraries
    wh_df = pd.read_csv(f"{DATA_DIR}/Updated_warhead.csv", low_memory=False)
    e3_df = pd.read_csv(f"{DATA_DIR}/E3_updated_features.csv", low_memory=False)
    wh_smiles = wh_df['warhead_Canonical_SMILES'].dropna().tolist()
    e3_smiles = e3_df['E3_Canonical_SMILES'].dropna().tolist()
    print(f"Warhead library: {len(wh_smiles)}, E3 library: {len(e3_smiles)}")

    # Precompute mols
    wh_mols = [(s, Chem.MolFromSmiles(s)) for s in wh_smiles if Chem.MolFromSmiles(s)]
    e3_mols = [(s, Chem.MolFromSmiles(s)) for s in e3_smiles if Chem.MolFromSmiles(s)]
    print(f"Valid warhead mols: {len(wh_mols)}, E3 mols: {len(e3_mols)}")

    # External compounds
    ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
    ext = ext.drop_duplicates(subset=['SMILES']).copy()
    print(f"External compounds: {len(ext)}")

    results = []
    t0 = time.time()
    for i, (_, r) in enumerate(ext.iterrows()):
        smi = r['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"SMILES": smi, "status": "invalid", "wh_smi": None, "e3_smi": None, "linker_smi": None})
            continue

        # Match warhead (largest substructure match preferred)
        best_wh = None; best_n = 0
        for s, wmol in wh_mols:
            if wmol.GetNumAtoms() > mol.GetNumAtoms(): continue
            if mol.HasSubstructMatch(wmol):
                n = wmol.GetNumAtoms()
                if n > best_n:
                    best_n = n; best_wh = s
        # Match E3 ligand
        best_e3 = None; best_n3 = 0
        for s, emol in e3_mols:
            if emol.GetNumAtoms() > mol.GetNumAtoms(): continue
            if mol.HasSubstructMatch(emol):
                n = emol.GetNumAtoms()
                if n > best_n3:
                    best_n3 = n; best_e3 = s

        if best_wh is None or best_e3 is None:
            results.append({"SMILES": smi, "status": "no_match",
                            "wh_smi": best_wh, "e3_smi": best_e3, "linker_smi": None})
            continue

        # Extract linker: remove warhead + E3 atoms
        try:
            rw = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(best_wh))
            rwe = Chem.DeleteSubstructs(rw, Chem.MolFromSmiles(best_e3))
            linker = Chem.MolToSmiles(rwe) if rwe and rwe.GetNumAtoms() > 0 else None
        except:
            linker = None

        results.append({"SMILES": smi, "status": "matched",
                        "wh_smi": best_wh, "e3_smi": best_e3,
                        "wh_atoms": best_n, "e3_atoms": best_n3,
                        "linker_smi": linker})

        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(ext)} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(results)
    print(f"\nMatched: {(res['status']=='matched').sum()}, no_match: {(res['status']=='no_match').sum()}, invalid: {(res['status']=='invalid').sum()}")
    print(f"Linkers extracted: {res['linker_smi'].notna().sum()}")

    res.to_csv(f"{OUT_DIR}/external_decomposed.csv", index=False)
    print(f"Saved: {OUT_DIR}/external_decomposed.csv")

    # GROVER fps for linkers
    linkers = res[res['linker_smi'].notna()]['linker_smi'].tolist()
    print(f"\nLinkers for GROVER: {len(linkers)}")
    if len(linkers) > 0:
        smi_file = f"{OUT_DIR}/linker_smiles.csv"
        pd.DataFrame({'smiles': linkers}).to_csv(smi_file, index=False)
        feats = f"{OUT_DIR}/linker_features.npz"
        fps_file = f"{OUT_DIR}/linker_fps.npz"
        env = os.environ.copy(); env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        r1 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/scripts/save_features.py "
            f"--data_path {smi_file} --save_path {feats} --features_generator rdkit_2d_normalized --restart",
            shell=True, capture_output=True, text=True, timeout=1800, env=env)
        if r1.returncode != 0:
            print(f"save_features failed: {r1.stderr[-300:]}"); sys.exit(1)
        r2 = subprocess.run(
            f"/home/saveenas/miniconda3/envs/synglu_test_package/bin/python {GROVER_DIR}/main.py fingerprint "
            f"--data_path {smi_file} --features_path {feats} "
            f"--checkpoint_path {MODEL_DIR}/grover_fixed.pt --fingerprint_source both --output {fps_file}",
            shell=True, capture_output=True, text=True, timeout=3600, env=env)
        if r2.returncode != 0:
            print(f"GROVER failed: {r2.stderr[-300:]}"); sys.exit(1)
        fps = np.load(fps_file)['fps'][:, :4800]
        print(f"Linker fps: {fps.shape}")
        np.save(f"{OUT_DIR}/linker_fps.npy", fps)

if __name__ == '__main__':
    main()
