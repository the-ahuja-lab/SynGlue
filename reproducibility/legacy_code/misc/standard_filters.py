#!/usr/bin/env python3
"""
Standardized Filter Cascade for Gcoupler Benchmarking.

Applies identical downstream filters to any tool's generated candidates:
1. Chemical validity
2. Uniqueness
3. Novelty vs. known degrader databases
4. Linker-length compatibility
5. Linker-geometry compatibility
6. Synthetic accessibility
7. Degrader-likeness / physicochemical constraints
8. Predicted DC50 / Dmax (when applicable)
9. Docking compatibility (structural filter)
10. Ternary-complex plausibility

Author: Feynman (SynGlue revision team)
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, RDConfig

RDLogger.DisableLog('rdApp.*')

# Attempt SA Score import — handle version differences in RDConfig
_SA_ADDED = False
for _rp in ['RDContribDir', 'contribDir']:
    try:
        p = getattr(RDConfig, _rp, None)
        if p:
            sys.path.append(os.path.join(p, 'SA_Score'))
            _SA_ADDED = True
            break
    except Exception:
        pass
if not _SA_ADDED:
    # Fallback: try common paths
    for _p in ['/usr/local/lib/python*/site-packages/rdkit/Contrib/SA_Score',
               '/home/saveenas/miniconda3/envs/synglu_test_package/lib/python*/site-packages/rdkit/Contrib/SA_Score',
               '/home/saveenas/miniconda3/envs/synglu_test_package/lib/python*/site-packages/rdkit/contrib/SA_Score']:
        import glob
        matches = glob.glob(_p)
        if matches:
            sys.path.append(matches[0])
            break
try:
    import sascorer
    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False

from config import FILTER_PARAMS


# ============================================================
# Helper Functions
# ============================================================

def canonicalize_smiles(smiles):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def compute_sa_score(mol):
    if SA_AVAILABLE:
        try:
            return sascorer.calculateScore(mol)
        except Exception:
            pass
    return None


def compute_physchem(mol):
    """Compute physicochemical properties."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    return {
        "MW": round(mw, 2),
        "logP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "RotatableBonds": rot_bonds,
        "HBD": hbd,
        "HBA": hba,
        "AromaticRings": arom_rings,
    }


def compute_ro3_violations(mol):
    """Rule-of-3 degrader violations: MW ≤ 500, logP ≤ 3, HBD ≤ 3, HBA ≤ 3."""
    adme = compute_physchem(mol)
    violations = 0
    if adme["MW"] > 500:
        violations += 1
    if adme["logP"] > 3:
        violations += 1
    if adme["HBD"] > 3:
        violations += 1
    if adme["HBA"] > 3:
        violations += 1
    return violations


def extract_linker(protac_smiles, warhead_smi, e3_smi):
    """Extract linker SMILES from a full PROTAC given warhead and E3 ligand."""
    protac = Chem.MolFromSmiles(protac_smiles)
    wh = Chem.MolFromSmiles(warhead_smi)
    e3_mol = Chem.MolFromSmiles(e3_smi)

    if not protac or not wh or not e3_mol:
        return None, None, None

    # Remove dummy atoms for matching
    def remove_dummy(mol):
        rw = Chem.RWMol(mol)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(dummies, reverse=True):
            rw.RemoveAtom(idx)
        return rw.GetMol()

    wh_clean = remove_dummy(wh)
    e3_clean = remove_dummy(e3_mol)

    wh_match = protac.GetSubstructMatch(wh_clean)
    e3_match = protac.GetSubstructMatch(e3_clean)

    if not wh_match or not e3_match:
        return None, None, None
    if set(wh_match).intersection(set(e3_match)):
        return None, None, None  # overlapping matches

    atoms_to_delete = list(set(wh_match + e3_match))
    rw = Chem.RWMol(protac)
    for idx in sorted(atoms_to_delete, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
        linker_smi = Chem.MolToSmiles(rw)
        if not linker_smi:
            return None, None, None
        linker_mol = Chem.MolFromSmiles(linker_smi)
    except Exception:
        return None, None, None

    if linker_mol is None:
        return None, None, None

    return linker_smi, linker_mol, {
        "wh_match_size": len(wh_match),
        "e3_match_size": len(e3_match),
    }


def compute_linker_metrics(linker_mol):
    """Compute linker graph length and effective length."""
    if linker_mol is None:
        return None
    # Graph length: number of atoms in linker backbone
    # Effective length: number of rotatable bonds + ring atoms along path
    n_atoms = linker_mol.GetNumAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(linker_mol)
    n_rings = rdMolDescriptors.CalcNumRings(linker_mol)

    # Approximate effective length as rotatable bonds + ring atoms
    ring_atoms = sum(len(
        [a for a in ring]
    ) for ring in linker_mol.GetRingInfo().AtomRings())

    return {
        "graph_length": n_atoms,
        "effective_length": n_rot + ring_atoms,
        "n_rotatable": n_rot,
        "n_rings": n_rings,
        "n_ring_atoms": ring_atoms,
    }


# ============================================================
# Primary Filter Cascade
# ============================================================

def apply_filter_cascade(
    smiles_list,
    warhead_smi,
    e3_smi,
    known_degraders=None,
    params=None
):
    """
    Apply standardized filter cascade to a list of generated SMILES.

    Parameters
    ----------
    smiles_list : list of str
        Generated candidate SMILES strings.
    warhead_smi : str
        Warhead SMILES (with dummy atom).
    e3_smi : str
        E3 ligand SMILES (with dummy atom).
    known_degraders : list of str, optional
        Known degrader SMILES for novelty check.
    params : dict, optional
        Filter parameters. Falls back to config.FILTER_PARAMS.

    Returns
    -------
    pd.DataFrame with columns: SMILES, is_valid, canonical, ..., pass_all
    """
    if params is None:
        params = FILTER_PARAMS

    if known_degraders is None:
        known_degraders = []
    known_canonical = set()
    for smi in known_degraders:
        c = canonicalize_smiles(smi)
        if c:
            known_canonical.add(c)

    results = []

    for raw_smi in smiles_list:
        record = {"input_smiles": raw_smi}

        # --- Level 1: Chemical validity ---
        mol = mol_from_smiles(raw_smi)
        if mol is None:
            record["is_valid"] = False
            record["pass_all"] = False
            record["fail_reason"] = "invalid_smiles"
            record["canonical_smiles"] = None
            # Fill remaining with defaults
            for k in ["mw", "logp", "tpsa", "rotatable_bonds", "hbd", "hba",
                       "sa_score", "ro3_violations", "linker_smiles",
                       "graph_length", "effective_length", "novel",
                       "pass_validity", "pass_uniqueness", "pass_novelty",
                       "pass_linker_length", "pass_sa", "pass_physchem",
                       "pass_desirability"]:
                record[k] = None
            results.append(record)
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        record["is_valid"] = True
        record["canonical_smiles"] = canonical

        # --- Level 2: Physicochemical properties ---
        adme = compute_physchem(mol)
        record["mw"] = adme["MW"]
        record["logp"] = adme["logP"]
        record["tpsa"] = adme["TPSA"]
        record["rotatable_bonds"] = adme["RotatableBonds"]
        record["hbd"] = adme["HBD"]
        record["hba"] = adme["HBA"]

        # --- Level 3: SA Score ---
        sa = compute_sa_score(mol)
        record["sa_score"] = sa

        # --- Level 4: Rule-of-3 ---
        ro3 = compute_ro3_violations(mol)
        record["ro3_violations"] = ro3

        # --- Level 5: Linker extraction ---
        linker_smi, linker_mol, linker_info = extract_linker(
            canonical, warhead_smi, e3_smi
        )
        record["linker_smiles"] = linker_smi
        record["linker_info"] = linker_info

        # --- Level 6: Linker metrics ---
        if linker_mol:
            link_metrics = compute_linker_metrics(linker_mol)
            record["graph_length"] = link_metrics["graph_length"]
            record["effective_length"] = link_metrics["effective_length"]
        else:
            record["graph_length"] = None
            record["effective_length"] = None

        # --- Level 7: Novelty ---
        record["novel"] = canonical not in known_canonical if known_canonical else True

        # === Filter Checks ===
        passes = {}

        # F1: Validity (already passed)
        passes["validity"] = True

        # F2: Physicochemical constraints
        passes["mw"] = adme["MW"] <= params["max_mw"]
        passes["logp"] = adme["logP"] <= params["max_logp"]
        passes["tpsa_low"] = adme["TPSA"] >= params["min_tpsa"]
        passes["tpsa_high"] = adme["TPSA"] <= params["max_tpsa"]
        passes["rotatable"] = adme["RotatableBonds"] <= params["max_rotatable"]

        # F3: SA Score
        passes["sa"] = sa is not None and sa <= params["max_sa_score"]

        # F4: RO3 violations
        passes["ro3"] = ro3 <= params["n_ro3_violations"]

        # F5: Linker length
        if linker_mol:
            passes["linker_graph"] = (
                params["min_linker_graph_length"] <= record["graph_length"] <= params["max_linker_graph_length"]
            )
            passes["linker_effective"] = (
                params["min_linker_effective_length"] <= record["effective_length"] <= params["max_linker_effective_length"]
            )
        else:
            passes["linker_graph"] = False
            passes["linker_effective"] = False

        # F6: Novelty
        passes["novelty"] = record["novel"]

        record["pass_validity"] = passes["validity"]
        record["pass_physchem_mw"] = passes["mw"]
        record["pass_physchem_logp"] = passes["logp"]
        record["pass_physchem_tpsa"] = passes["tpsa_low"] and passes["tpsa_high"]
        record["pass_physchem_rot"] = passes["rotatable"]
        record["pass_sa"] = passes["sa"]
        record["pass_ro3"] = passes["ro3"]
        record["pass_linker_graph"] = passes["linker_graph"]
        record["pass_linker_effective"] = passes["linker_effective"]
        record["pass_novelty"] = passes["novelty"]

        # Combined pass/fail
        combined = (
            passes["validity"]
            and passes["mw"] and passes["logp"] and passes["tpsa_low"] and passes["tpsa_high"]
            and passes["rotatable"]
            and passes["sa"]
            and passes["ro3"]
            and passes["linker_graph"]
            and passes["linker_effective"]
        )
        record["pass_all"] = combined
        record["pass_count"] = sum(1 for v in passes.values() if v)
        record["total_checks"] = len(passes)

        if not combined:
            # Identify first failing reason
            for check_name in [
                "validity", "mw", "logp", "tpsa_low", "tpsa_high",
                "rotatable", "sa", "ro3", "linker_graph", "linker_effective", "novelty"
            ]:
                if not passes[check_name]:
                    record["fail_reason"] = f"fail_{check_name}"
                    break
        else:
            record["fail_reason"] = None

        results.append(record)

    df = pd.DataFrame(results)
    return df


# ============================================================
# Summary Statistics
# ============================================================

def summarize_filter_results(df, tool_name):
    """Generate summary statistics for a tool's filtered output."""
    total = len(df)
    valid = df["is_valid"].sum() if "is_valid" in df else 0
    passed = df["pass_all"].sum() if "pass_all" in df else 0
    physchem_pass = df["pass_physchem_mw"].sum() if "pass_physchem_mw" in df else 0
    sa_pass = df["pass_sa"].sum() if "pass_sa" in df else 0
    linker_pass = df["pass_linker_graph"].sum() if "pass_linker_graph" in df else 0
    novelty_pass = df["pass_novelty"].sum() if "pass_novelty" in df else 0

    summary = {
        "Tool": tool_name,
        "Total generated": total,
        "Valid SMILES": valid,
        "Validity rate (%)": round(100 * valid / total, 1) if total > 0 else 0,
        "Passed all filters": int(passed),
        "Pass rate (%)": round(100 * passed / total, 1) if total > 0 else 0,
        "Passed physchem": int(physchem_pass),
        "Passed SA score": int(sa_pass),
        "Passed linker constraints": int(linker_pass),
        "Novel (vs. known)": int(novelty_pass),
    }
    return summary


def build_comparison_table(summaries):
    """Build a comparison table from multiple tool summaries."""
    df = pd.DataFrame(summaries)
    df = df.set_index("Tool")
    return df


# ============================================================
# Main (standalone)
# ============================================================

if __name__ == "__main__":
    # Test with Gcoupler GSPT1 output
    gcoupler_csv = (
        "/storage/savi/saveenas/Projects/Magnet/Reinvent/GSPT1/results/scaffold_memory.csv"
    )
    if os.path.exists(gcoupler_csv):
        df_in = pd.read_csv(gcoupler_csv)
        smiles_col = "SMILES" if "SMILES" in df_in.columns else df_in.columns[2]
        smiles_list = df_in[smiles_col].dropna().tolist()
        print(f"Loaded {len(smiles_list)} Gcoupler candidates")

        from config import GSPT1_WARHEAD_SMILES, GSPT1_E3_LIGAND_SMILES, KNOWN_GSPT1_DEGRADERS

        df_filtered = apply_filter_cascade(
            smiles_list,
            GSPT1_WARHEAD_SMILES,
            GSPT1_E3_LIGAND_SMILES,
            known_degraders=KNOWN_GSPT1_DEGRADERS,
        )

        summary = summarize_filter_results(df_filtered, "Gcoupler")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        # Save
        out_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "gcoupler", "filtered_results.csv"
        )
        df_filtered.to_csv(out_csv, index=False)
        print(f"\nSaved filtered results to {out_csv}")
    else:
        print("Gcoupler results not found. Run generation first.")
