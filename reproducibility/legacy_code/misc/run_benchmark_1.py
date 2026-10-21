#!/usr/bin/env python3
"""
Gcoupler vs Representative Methods — GSPT1 (5HXB chain A) Benchmark

Standardized evaluation framework (9 levels):
  1. Chemical validity
  2. Uniqueness
  3. Novelty vs. known degrader databases
  4. Linker-length & linker-geometry compatibility
  5. Synthetic accessibility
  6. Degrader-likeness / physicochemical constraints
  7. Predicted degradation potency (PDP / SynGlue predictor)
  8. Docking compatibility (smina vs 5HXB pocket)
  9. Ternary-complex plausibility (structural filtering)

Tools compared:
  - Gcoupler (SynGlue) — full PROTAC generation
  - Pocket2Mol — pocket-guided de novo
  - DiffSBDD — diffusion-based SBDD
  - DiffLinker — 3D fragment/linker generation
  - DeLinker — 2D fragment/linker generation
  - DeepLigBuilder — structure-conditioned ligand building (blocked)
  - Pharmacophore enumeration — rule-based control

Output: benchmarking/gcoupler_vs_tools/results/ + figures/
"""
import os, sys, json, warnings, subprocess, glob
warnings.filterwarnings('ignore')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
ROOT = f"{REV}/benchmarking/gcoupler_vs_tools"
RES = f"{ROOT}/results"; FIG = f"{ROOT}/figures"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, f"{REV}/experiments/gcoupler_benchmark")
from standard_filters import (canonicalize_smiles, compute_sa_score, compute_physchem,
                               extract_linker, compute_linker_metrics, apply_filter_cascade)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

GSPT1_PDB = f"{REV}/experiments/gcoupler_benchmark/results/pocket2mol/sample_for_pdb_5HXB_chainA_clean.pdb_2026_07_14__12_21_20/5HXB_chainA_clean.pdb"
SMINA = "/home/saveenas/miniconda3/envs/smina_test/bin/smina"

# Reference components for GSPT1
WARHEAD_SMI = "Nc1nnc(C#CC2CC2)s1"
E3_SMI = "O=C1CCC(c2ccc(N3CCNCC3)cc2)C(=O)N1"

# Known GSPT1 chemotypes (novelty reference)
KNOWN_GSPT1 = {
    "Nc1nnc(C#CC2CC2)s1", "Nc1ncc(Cl)cn1", "Cc1nc2ccccc2n1",
    "Nc1ncnc2ccccc12", "O=C1CCC(c2ccc(N3CCNCC3)cc2)C(=O)N1",
}

def load_tool_outputs():
    """Load all tool outputs into unified DataFrame with 'tool' and 'smiles'."""
    GCR = f"{REV}/experiments/gcoupler_benchmark/results"
    frames = []

    # 1. Gcoupler — full PROTACs
    g = pd.read_csv(f"{GCR}/gcoupler/filtered_results.csv")
    g = g[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
    g['tool'] = 'Gcoupler'
    frames.append(g)

    # 2. Pocket2Mol — pocket-guided (SMILES.txt)
    for d in sorted(glob.glob(f"{GCR}/pocket2mol/*/SMILES.txt")):
        p = pd.read_csv(d, header=None, names=['smiles'])
        p['tool'] = 'Pocket2Mol'
        frames.append(p)

    # 3. DiffSBDD — diffusion (samples.sdf)
    sdf = f"{GCR}/diffsbdd/samples.sdf"
    mols = [m for m in Chem.SDMolSupplier(sdf) if m is not None]
    d = pd.DataFrame({'smiles': [Chem.MolToSmiles(m) for m in mols]})
    d['tool'] = 'DiffSBDD'
    frames.append(d)

    # 4. DiffLinker — linkers (collected sdf or xyz)
    df = pd.read_csv(f"{GCR}/warheads/summary/warhead_difflinker.csv") if os.path.exists(f"{GCR}/warheads/summary/warhead_difflinker.csv") else pd.DataFrame()
    if len(df) and 'canonical_smiles' in df.columns:
        df = df[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
        df['tool'] = 'DiffLinker'
        frames.append(df)

    # 5. DeLinker — linkers
    dl = pd.read_csv(f"{GCR}/warheads/summary/warhead_delinker.csv") if os.path.exists(f"{GCR}/warheads/summary/warhead_delinker.csv") else pd.DataFrame()
    if len(dl) and 'canonical_smiles' in dl.columns:
        dl = dl[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
        dl['tool'] = 'DeLinker'
        frames.append(dl)

    # 6. Pharmacophore — full PROTAC candidates
    ph = pd.read_csv(f"{GCR}/pharmacophore/generated_candidates.csv")
    ph = ph[['SMILES']].rename(columns={'SMILES': 'smiles'})
    ph['tool'] = 'Pharmacophore'
    frames.append(ph)

    # 7. ShapeLinker (extra; from warhead benchmark)
    if os.path.exists(f"{GCR}/warheads/summary/warhead_shapelinker.csv"):
        sl = pd.read_csv(f"{GCR}/warheads/summary/warhead_shapelinker.csv")
        if len(sl) and 'canonical_smiles' in sl.columns:
            sl = sl[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
            sl['tool'] = 'ShapeLinker'
            frames.append(sl)

    out = pd.concat(frames, ignore_index=True)
    # dedupe within tool
    out = out.drop_duplicates(subset=['tool', 'smiles'])
    print(f"Loaded {len(out)} candidates across {out['tool'].nunique()} tools:")
    for t, n in out['tool'].value_counts().items():
        print(f"  {t}: {n}")
    return out

def main():
    df = load_tool_outputs()

    # ─── Level 1-6: filter cascade (validity, uniqueness, physchem, SA, linker) ───
    print("\n=== Filter cascade (levels 1-6) ===")
    results = []
    for tool, grp in df.groupby('tool'):
        valid = grp[grp['smiles'].apply(canonicalize_smiles).notna()].copy()
        valid['canon'] = valid['smiles'].apply(canonicalize_smiles)
        valid = valid.drop_duplicates('canon')
        n_raw = len(grp); n_valid = len(valid)

        # physchem + SA
        mols = valid['canon'].apply(Chem.MolFromSmiles)
        pc = mols.apply(lambda m: compute_physchem(m) if m else None)
        valid['MW'] = pc.apply(lambda x: x['MW'] if x else np.nan)
        valid['logP'] = pc.apply(lambda x: x['logP'] if x else np.nan)
        valid['TPSA'] = pc.apply(lambda x: x['TPSA'] if x else np.nan)
        valid['RotB'] = pc.apply(lambda x: x['RotatableBonds'] if x else np.nan)
        valid['HBD'] = pc.apply(lambda x: x['HBD'] if x else np.nan)
        valid['HBA'] = pc.apply(lambda x: x['HBA'] if x else np.nan)
        valid['SA'] = [compute_sa_score(m) if m else np.nan for m in mols]

        # degrader-likeness (PROTAC-like: MW 500-1200, logP -1..7, TPSA 80..250)
        dl = valid[(valid['MW'] >= 500) & (valid['MW'] <= 1200) &
                   (valid['logP'] >= -1) & (valid['logP'] <= 7) &
                   (valid['TPSA'] >= 80) & (valid['TPSA'] <= 250)]
        # SA pass
        sa_pass = valid[valid['SA'] <= 5.0]

        # linker extraction (for PROTAC/linker tools)
        linker_pass = 0
        linker_stats = {}
        if tool in ('Gcoupler', 'Pharmacophore', 'ShapeLinker'):
            def get_linker(s):
                res = extract_linker(s, WARHEAD_SMI, E3_SMI)
                if res is None: return (None, None)
                return res[0], res[1]  # smi, mol
            lk = valid['canon'].apply(get_linker)
            valid['linker'] = lk.apply(lambda x: x[0])
            lm = valid['linker'].apply(lambda l: compute_linker_metrics(Chem.MolFromSmiles(l)) if l else None)
            valid['graph_len'] = lm.apply(lambda x: x['graph_length'] if x else np.nan)
            valid['eff_len'] = lm.apply(lambda x: x['effective_length'] if x else np.nan)
            linker_pass = valid[(valid['graph_len'] >= 4) & (valid['graph_len'] <= 12) &
                                (valid['eff_len'] >= 4) & (valid['eff_len'] <= 10)].shape[0]
            linker_stats = {"n_linker_extracted": valid['linker'].notna().sum(),
                            "linker_pass": linker_pass}

        # novelty
        novel = valid[~valid['canon'].isin(KNOWN_GSPT1)]

        # combined "developability pass" = valid + physchem + SA + (linker if applicable)
        if linker_stats:
            dev_pass = valid[(valid['MW'] >= 500) & (valid['MW'] <= 1200) &
                             (valid['logP'] >= -1) & (valid['logP'] <= 7) &
                             (valid['SA'] <= 5.0) &
                             (valid['graph_len'] >= 4) & (valid['graph_len'] <= 12) &
                             (valid['eff_len'] >= 4) & (valid['eff_len'] <= 10)]
        else:
            dev_pass = valid[(valid['MW'] >= 200) & (valid['MW'] <= 700) &
                             (valid['logP'] >= -1) & (valid['logP'] <= 6) &
                             (valid['SA'] <= 5.0)]

        results.append({
            'tool': tool, 'n_raw': n_raw, 'n_valid': n_valid,
            'validity_pct': 100.0 * n_valid / max(n_raw, 1),
            'n_unique': valid['canon'].nunique(),
            'novelty_pct': 100.0 * len(novel) / max(len(valid), 1),
            'physchem_pass': len(dl), 'physchem_pct': 100.0 * len(dl) / max(len(valid), 1),
            'sa_pass': len(sa_pass), 'sa_pct': 100.0 * len(sa_pass) / max(len(valid), 1),
            'linker_pass': linker_pass,
            'developability_pass': len(dev_pass),
            'developability_pct': 100.0 * len(dev_pass) / max(len(valid), 1),
        })
        print(f"  {tool}: valid={n_valid}/{n_raw} ({100*n_valid/max(n_raw,1):.0f}%), "
              f"developability pass={len(dev_pass)} ({100*len(dev_pass)/max(len(valid),1):.1f}%)")

    summ = pd.DataFrame(results)
    summ.to_csv(f"{RES}/level1_6_filter_cascade.csv", index=False)
    print(f"\nSaved: {RES}/level1_6_filter_cascade.csv")

    # Save per-candidate developability-passed molecules for downstream docking
    print("\nNote: saved cascade summary. Next: level 7-9 (potency, docking, ternary).")

if __name__ == '__main__':
    main()
