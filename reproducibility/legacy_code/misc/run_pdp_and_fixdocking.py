#!/usr/bin/env python3
"""
Level 7: Predicted degradation potency via PDP (PROTAC-Degradation-Predictor)
Level 8b: Re-dock pharmacophore (largest fragment) + DiffLinker (assembled) with smina
"""
import os, sys, json, warnings, subprocess, glob, tempfile
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
ROOT = f"{REV}/benchmarking/gcoupler_vs_tools"
RES = f"{ROOT}/results"
os.makedirs(RES, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

SMINA = "/home/saveenas/miniconda3/envs/smina_test/bin/smina"
RECEPTOR = f"{RES}/receptor_5HXB.pdbqt"
POCKET_CENTER = (3.8, 28.9, 168.9)

def largest_fragment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        rw = Chem.RWMol(mol)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(dummies, reverse=True):
            rw.RemoveAtom(idx)
        m2 = rw.GetMol()
        frags = Chem.GetMolFrags(m2, asMols=True)
        if not frags: return None
        largest = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)[0]
        return Chem.MolToSmiles(largest)
    except Exception:
        try:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags: return None
            largest = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)[0]
            return Chem.MolToSmiles(largest)
        except Exception:
            return None

def dock_smina_frag(smiles, tag, idx):
    smi = largest_fragment(smiles)
    if smi is None: return None
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0: return None
    AllChem.MMFFOptimizeMolecule(mol)
    pdb = f"{RES}/tmp_{tag}_{idx}.pdb"
    Chem.MolToPDBFile(mol, pdb)
    pdbqt = f"{RES}/tmp_{tag}_{idx}.pdbqt"
    obabel = "/home/saveenas/miniconda3/envs/smina_test/bin/obabel"
    r = subprocess.run(f"{obabel} {pdb} -O {pdbqt}", shell=True, capture_output=True, text=True, timeout=120)
    if not os.path.exists(pdbqt): return None
    out = f"{RES}/dock_{tag}_{idx}.sdf"
    cx, cy, cz = POCKET_CENTER
    r2 = subprocess.run(f"{SMINA} -r {RECEPTOR} -l {pdbqt} --center_x {cx} --center_y {cy} --center_z {cz} "
                        f"--size_x 22 --size_y 22 --size_z 22 --num_modes 1 --out {out}",
                        shell=True, capture_output=True, text=True, timeout=300)
    if os.path.exists(out):
        try:
            m = next(Chem.SDMolSupplier(out))
            if m is not None:
                return float(m.GetProp("minimizedAffinity"))
        except: pass
    return None

def run_pdp(smiles_list, e3='CRBN', uniprot='P15170', cell='22Rv1'):
    """Run PDP via subprocess (pdp env)."""
    if not smiles_list: return []
    import pickle
    script = f'''
import warnings, sys, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "{BASE}/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor.pytorch_models as pm
orig = pm.load_model
def patched(ckpt):
    import torch
    m = pm.PROTAC_Model.load_from_checkpoint(ckpt, map_location="cpu")
    return m.eval()
pm.load_model = patched
import protac_degradation_predictor.protac_degradation_predictor as pdp_mod
pdp_mod.load_model = patched
import protac_degradation_predictor as pdp
smis = {json.dumps(smiles_list)}
preds = []
for s in smis:
    try:
        p = pdp.get_protac_active_proba(s, "{e3}", "{uniprot}", "{cell}", device="cpu", use_models_from_cv=False)
        preds.append(float(p["mean"]))
    except Exception as e:
        preds.append(None)
print(json.dumps(preds))
'''
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(script); tmp = f.name
    r = subprocess.run(["/home/saveenas/miniconda3/envs/pdp/bin/python", tmp],
                       capture_output=True, text=True, timeout=1800)
    os.unlink(tmp)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except:
        print("PDP stderr:", r.stderr[-300:])
        return []

def main():
    GCR = f"{REV}/experiments/gcoupler_benchmark/results"
    ph = pd.read_csv(f"{GCR}/pharmacophore/generated_candidates.csv")

    # ── Level 7: PDP potency on sample (15/tool, PROTAC-type tools) ──
    print("=== Level 7: PDP potency ===")
    tools_smiles = {
        'Gcoupler': pd.read_csv(f"{GCR}/gcoupler/filtered_results.csv")['canonical_smiles'].tolist(),
        'Pharmacophore': ph['SMILES'].tolist(),
    }
    pdp_rows = []
    for tool, smis in tools_smiles.items():
        sample = smis[:15]
        preds = run_pdp(sample)
        valid_preds = [p for p in preds if p is not None]
        pdp_rows.append({
            "tool": tool, "n_tested": len(sample), "n_predicted": len(valid_preds),
            "mean_p_active": np.mean(valid_preds) if valid_preds else np.nan,
            "median_p_active": np.median(valid_preds) if valid_preds else np.nan,
            "n_active_p50": sum(1 for p in valid_preds if p > 0.5) if valid_preds else 0,
        })
        print(f"  {tool}: mean P(active)={np.mean(valid_preds):.3f} ({len(valid_preds)}/{len(sample)})" if valid_preds else f"  {tool}: no preds")
    pd.DataFrame(pdp_rows).to_csv(f"{RES}/level7_pdp_potency.csv", index=False)
    print(f"  Saved: {RES}/level7_pdp_potency.csv")

    # ── Level 8b: dock pharmacophore (largest fragment) + DiffLinker assembled ──
    print("\n=== Level 8b: pharmacophore + difflinker docking (fragment method) ===")
    dock_rows = []
    for tool, smis in [('Pharmacophore', ph['SMILES'].tolist()),
                       ('DiffLinker', pd.read_csv(f"{GCR}/warheads/summary/warhead_difflinker.csv")['canonical_smiles'].tolist())]:
        sample = smis[:20]
        scores = []
        for i, s in enumerate(sample):
            aff = dock_smina_frag(s, tool.replace(' ', '_'), i)
            if aff is not None:
                scores.append(aff)
        dock_rows.append({"tool": tool, "n_docked": len(scores), "n_tried": len(sample),
                          "affinity_mean_kcal": np.mean(scores) if scores else np.nan,
                          "affinity_best_kcal": min(scores) if scores else np.nan,
                          "n_below_minus6": sum(1 for a in scores if a < -6)})
        print(f"  {tool}: docked {len(scores)}/{len(sample)}, mean={np.mean(scores):.1f}" if scores else f"  {tool}: no scores")
    pd.DataFrame(dock_rows).to_csv(f"{RES}/level8b_docking_fixed.csv", index=False)
    print(f"  Saved: {RES}/level8b_docking_fixed.csv")

if __name__ == '__main__':
    main()
