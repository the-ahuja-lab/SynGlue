#!/usr/bin/env python3
"""
Standardized Warhead Evaluation for GSPT1 de novo generation outputs.

Applies a warhead-oriented filter cascade to every tool's generated molecules:
1. Chemical validity (RDKit parse)
2. Uniqueness (canonical SMILES dedup)
3. Novelty vs. known GSPT1 degrader warheads
4. Physicochemical warhead constraints:
   - MW 150-700 Da (warhead-like, not full PROTAC)
   - logP -1 to 6
   - TPSA 20-180
   - rotatable bonds <= 12
   - HBD <= 5, HBA <= 10
5. SA score <= 5.0
6. Rule-of-3 violations <= 2
7. Warhead-likeness (has >=1 ring, contains N/O/S heteroatom)
8. For linker/PROTAC tools: linker-length + geometry checks applied separately.

Output: results/warheads/summary/warhead_comparison.csv + per-tool filtered CSVs
"""
import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, RDConfig

RDLogger.DisableLog('rdApp.*')

# SA score
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(ROOT, "summary")
os.makedirs(SUMMARY, exist_ok=True)

# Known GSPT1 warhead/degrader chemotypes (for novelty check)
KNOWN_GSPT1 = {
    "Nc1nnc(C#CC2CC2)s1",          # Gcoupler-selected thiadiazole warhead
    "Nc1ncc(Cl)cn1",               # 4-amino-2-chloropyrimidine (CC-90009-like)
    "Cc1nc2ccccc2n1",              # quinazoline-like
    "Nc1ncnc2ccccc12",             # 4-aminoquinazoline
    "O=C1CCC(c2ccc(N3CCNCC3)cc2)C(=O)N1",  # pomalidomide/glutarimide core
}

FILTER_PARAMS = {
    "mw_min": 150, "mw_max": 700,
    "logp_min": -1.0, "logp_max": 6.0,
    "tpsa_min": 20, "tpsa_max": 180,
    "max_rot": 12, "max_hbd": 5, "max_hba": 10,
    "max_sa": 5.0, "n_ro3_violations": 2,
}

def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)

def analyze(smi):
    """Return dict of properties + pass flags for a single warhead candidate."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {"valid": False}
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    rot = rdMolDescriptors.CalcNumRotatableBonds(m)
    hbd = rdMolDescriptors.CalcNumHBD(m)
    hba = rdMolDescriptors.CalcNumHBA(m)
    rings = rdMolDescriptors.CalcNumRings(m)
    arom = rdMolDescriptors.CalcNumAromaticRings(m)
    sa = sascorer.calculateScore(m) if SA_AVAILABLE else None
    ro3 = 0
    if mw > 300: ro3 += 1
    if logp > 3: ro3 += 1
    if tpsa > 60: ro3 += 1
    het = any(a.GetSymbol() in ("N", "O", "S") for a in m.GetAtoms())
    return {
        "valid": True, "MW": round(mw, 2), "logP": round(logp, 2),
        "TPSA": round(tpsa, 2), "RotBonds": rot, "HBD": hbd, "HBA": hba,
        "Rings": rings, "AromRings": arom, "SA": round(sa, 3) if sa else None,
        "RO3_viol": ro3, "has_ring": rings >= 1, "has_heteroatom": het,
    }

def warhead_filters(p):
    """Pass/fail for warhead-oriented constraints."""
    if not p.get("valid"):
        return False, "invalid"
    if not (FILTER_PARAMS["mw_min"] <= p["MW"] <= FILTER_PARAMS["mw_max"]):
        return False, f"mw_out_{p['MW']}"
    if not (FILTER_PARAMS["logp_min"] <= p["logP"] <= FILTER_PARAMS["logp_max"]):
        return False, "logp_out"
    if not (FILTER_PARAMS["tpsa_min"] <= p["TPSA"] <= FILTER_PARAMS["tpsa_max"]):
        return False, "tpsa_out"
    if p["RotBonds"] > FILTER_PARAMS["max_rot"]:
        return False, "rot_out"
    if p["HBD"] > FILTER_PARAMS["max_hbd"] or p["HBA"] > FILTER_PARAMS["max_hba"]:
        return False, "hbd_hba_out"
    if p.get("SA") is not None and p["SA"] > FILTER_PARAMS["max_sa"]:
        return False, "sa_out"
    if p["RO3_viol"] > FILTER_PARAMS["n_ro3_violations"]:
        return False, "ro3_out"
    if not p["has_ring"] or not p["has_heteroatom"]:
        return False, "not_warhead_like"
    return True, "pass"

# ============================================================
# Load per-tool outputs
# ============================================================

def load_tool_outputs():
    """Return {tool: list_of_smiles} from each tool's results dir."""
    out = {}

    # Gcoupler (full PROTACs; warhead part is embedded)
    gc = os.path.join(ROOT, "gcoupler", "scaffold_memory.csv")
    if os.path.exists(gc):
        df = pd.read_csv(gc)
        smis = [s for s in df["SMILES"].dropna().tolist()]
        out["Gcoupler"] = smis

    # Pocket2Mol SDFs
    p2m = []
    for d in sorted(glob.glob(os.path.join(ROOT, "pocket2mol", "sample_for_pdb_*"))):
        for sdf in sorted(glob.glob(os.path.join(d, "SDF", "*.sdf"))) + sorted(glob.glob(os.path.join(d, "*.sdf"))):
            try:
                supp = Chem.SDMolSupplier(sdf)
                for m in supp:
                    if m is not None:
                        p2m.append(Chem.MolToSmiles(m))
            except Exception:
                pass
    # also log.txt Success: lines
    for logf in glob.glob(os.path.join(ROOT, "pocket2mol", "sample_for_pdb_*", "log.txt")):
        for line in open(logf):
            if "Success: " in line:
                p2m.append(line.split("Success: ")[1].strip())
    out["Pocket2Mol"] = p2m

    # DiffSBDD SDF
    ds = os.path.join(ROOT, "diffsbdd", "samples.sdf")
    if os.path.exists(ds):
        supp = Chem.SDMolSupplier(ds)
        out["DiffSBDD"] = [Chem.MolToSmiles(m) for m in supp if m is not None]

    # PGMG result txt
    pg = []
    for f in glob.glob(os.path.join(ROOT, "pgmg", "*result*.txt")) + glob.glob(os.path.join(ROOT, "pgmg", "*", "*result*.txt")):
        for line in open(f):
            line = line.strip()
            if line and Chem.MolFromSmiles(line):
                pg.append(line)
    out["PGMG"] = pg

    # DeepICL SDFs
    di = []
    for f in glob.glob(os.path.join(ROOT, "deepicl", "results", "*.sdf")):
        try:
            supp = Chem.SDMolSupplier(f)
            for m in supp:
                if m is not None:
                    di.append(Chem.MolToSmiles(m))
        except Exception:
            pass
    out["DeepICL"] = di

    # DiffLinker XYZ -> SDF (obabel collected/all.sdf)
    dl = []
    for sdf_all in [os.path.join(ROOT, "difflinker", "collected", "all.sdf"),
                    os.path.join(ROOT, "difflinker", "all.sdf")]:
        if os.path.exists(sdf_all):
            supp = Chem.SDMolSupplier(sdf_all)
            for m in supp:
                if m is not None:
                    dl.append(Chem.MolToSmiles(m))
            break
    out["DiffLinker"] = dl

    # DeLinker .smi (columns: frags frags linker)
    de = []
    for f in glob.glob(os.path.join(ROOT, "delinker", "*.smi")):
        for line in open(f):
            toks = line.strip().split()
            if len(toks) >= 3:
                de.append(toks[2])
    out["DeLinker"] = de

    # ShapeLinker sampled.csv (molecules, input, linker, likelihood)
    sh = []
    sc = os.path.join(ROOT, "shapelinker", "sampled.csv")
    if os.path.exists(sc):
        df = pd.read_csv(sc)
        for col in ("molecules", "SMILES", "output"):
            if col in df.columns:
                sh.extend(df[col].dropna().astype(str).tolist())
                break
    out["ShapeLinker"] = sh

    # DiffPROTACs XYZ -> SDF -> SMILES (converted separately)
    dp = []
    conv = os.path.join(ROOT, "diffprotacs", "converted_smiles.csv")
    if os.path.exists(conv):
        df = pd.read_csv(conv)
        dp = df["smiles"].dropna().tolist()
    out["DiffPROTACs"] = dp

    # Pharmacophore enumeration
    ph = []
    pc = os.path.join(ROOT, "pharmacophore", "generated_candidates.csv")
    if os.path.exists(pc):
        df = pd.read_csv(pc)
        ph = df.iloc[:, 0].dropna().tolist()
    out["Pharmacophore"] = ph

    return out

def main():
    tools = load_tool_outputs()
    rows = []
    per_tool = {}
    for tool, smis in tools.items():
        # dedupe
        seen = set()
        uniq = []
        for s in smis:
            c = canonical(s)
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        n_raw = len(smis)
        n_valid = sum(1 for s in smis if canonical(s))
        n_uniq = len(uniq)
        novel = [s for s in uniq if s not in KNOWN_GSPT1]
        passed = []
        fails = Counter()
        for s in uniq:
            p = analyze(s)
            ok, reason = warhead_filters(p)
            if ok:
                passed.append(s)
            else:
                fails[reason] += 1
        rows.append({
            "Tool": tool, "Raw": n_raw, "Valid": n_valid, "Validity%": round(100*n_valid/max(n_raw,1),1),
            "Unique": n_uniq, "Novel": len(novel), "Novelty%": round(100*len(novel)/max(n_uniq,1),1),
            "PassedWarheadFilters": len(passed), "PassRate%": round(100*len(passed)/max(n_uniq,1),1),
            "TopFailure": fails.most_common(1)[0][0] if fails else "-",
        })
        per_tool[tool] = {"smiles": uniq, "passed": passed, "fails": dict(fails)}
        pd.DataFrame({"canonical_smiles": uniq}).to_csv(
            os.path.join(SUMMARY, f"warhead_{tool.replace(' ','').lower()}.csv"), index=False)

    res = pd.DataFrame(rows).sort_values("PassRate%", ascending=False)
    res.to_csv(os.path.join(SUMMARY, "warhead_comparison.csv"), index=False)
    print(res.to_string(index=False))
    print("\nSaved per-tool lists →", SUMMARY)
    # Save top passes
    all_passed = []
    for t, d in per_tool.items():
        for s in d["passed"]:
            all_passed.append((t, s))
    pd.DataFrame(all_passed, columns=["tool", "smiles"]).to_csv(
        os.path.join(SUMMARY, "warhead_all_passed.csv"), index=False)
    print(f"Total warhead-filter-passing molecules across tools: {len(all_passed)}")

if __name__ == "__main__":
    main()
