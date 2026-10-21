#!/usr/bin/env python3
"""
GSPT1 De Novo Warhead Generation — Final Evaluation + Comparison

Runs two complementary evaluations on every tool's output:
  A) Warhead-oriented cascade (for de novo pocket-binder outputs)
     validity -> uniqueness -> novelty -> physchem -> SA -> Ro3 -> warhead-likeness
  B) PROTAC linker cascade (for full-PROTAC/linker tools, reuses standard_filters)
     extract linker -> linker length/geometry -> physchem -> SA

Also reports per-tool raw counts, and marks blocked tools.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), ".."))  # gcoupler_benchmark/
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, RDConfig

RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False

from standard_filters import extract_linker, compute_linker_metrics

SUMMARY = os.path.join(ROOT, "summary")
os.makedirs(SUMMARY, exist_ok=True)

WARHEAD_SMI = "*Nc1nnc(C#CC2CC2)s1"
E3_SMI = "*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"

KNOWN_GSPT1 = {
    "Nc1nnc(C#CC2CC2)s1",
    "Nc1ncc(Cl)cn1",
    "Cc1nc2ccccc2n1",
    "Nc1ncnc2ccccc12",
    "O=C1CCC(c2ccc(N3CCNCC3)cc2)C(=O)N1",
}

F = {"mw_min": 150, "mw_max": 700, "logp_min": -1.0, "logp_max": 6.0,
     "tpsa_min": 20, "tpsa_max": 180, "max_rot": 12, "max_hbd": 5, "max_hba": 10,
     "max_sa": 5.0, "n_ro3": 2}

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def analyze(smi):
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
    sa = sascorer.calculateScore(m) if SA_AVAILABLE else 99.0
    ro3 = int(mw > 300) + int(logp > 3) + int(tpsa > 60)
    het = any(a.GetSymbol() in ("N", "O", "S") for a in m.GetAtoms())
    return {"valid": True, "MW": mw, "logP": logp, "TPSA": tpsa, "Rot": rot,
            "HBD": hbd, "HBA": hba, "Rings": rings, "SA": sa, "RO3": ro3,
            "ring": rings >= 1, "het": het}

def warhead_pass(p):
    if not p.get("valid"):
        return False, "invalid"
    if not (F["mw_min"] <= p["MW"] <= F["mw_max"]): return False, "mw"
    if not (F["logp_min"] <= p["logP"] <= F["logp_max"]): return False, "logp"
    if not (F["tpsa_min"] <= p["TPSA"] <= F["tpsa_max"]): return False, "tpsa"
    if p["Rot"] > F["max_rot"]: return False, "rot"
    if p["HBD"] > F["max_hbd"] or p["HBA"] > F["max_hba"]: return False, "hbdhba"
    if p["SA"] > F["max_sa"]: return False, "sa"
    if p["RO3"] > F["n_ro3"]: return False, "ro3"
    if not p["ring"] or not p["het"]: return False, "not_warhead_like"
    return True, "pass"

def load_all():
    out = {}

    gc = os.path.join(ROOT, "gcoupler", "scaffold_memory.csv")
    if os.path.exists(gc):
        df = pd.read_csv(gc)
        out["Gcoupler"] = df["SMILES"].dropna().tolist()

    p2m = []
    for d in sorted(glob.glob(os.path.join(ROOT, "pocket2mol", "sample_for_pdb_*"))):
        for logf in glob.glob(os.path.join(d, "log.txt")):
            for line in open(logf):
                if "Success: " in line:
                    p2m.append(line.split("Success: ")[1].strip())
        for sdf in sorted(glob.glob(os.path.join(d, "SDF", "*.sdf"))):
            try:
                for m in Chem.SDMolSupplier(sdf):
                    if m is not None:
                        p2m.append(Chem.MolToSmiles(m))
            except Exception:
                pass
    out["Pocket2Mol"] = p2m

    ds = os.path.join(ROOT, "diffsbdd", "samples.sdf")
    if os.path.exists(ds):
        out["DiffSBDD"] = [Chem.MolToSmiles(m) for m in Chem.SDMolSupplier(ds) if m is not None]

    pg = []
    for f in glob.glob(os.path.join(ROOT, "pgmg", "*result*.txt")):
        for line in open(f):
            s = line.strip()
            if s and Chem.MolFromSmiles(s):
                pg.append(s)
    out["PGMG"] = pg

    di = []
    for f in glob.glob(os.path.join(ROOT, "deepicl", "results", "*.sdf")):
        try:
            for m in Chem.SDMolSupplier(f):
                if m is not None:
                    di.append(Chem.MolToSmiles(m))
        except Exception:
            pass
    out["DeepICL"] = di

    dl = []
    for sdf_all in [os.path.join(ROOT, "difflinker", "collected", "all.sdf"),
                    os.path.join(ROOT, "difflinker", "all.sdf")]:
        if os.path.exists(sdf_all):
            for m in Chem.SDMolSupplier(sdf_all):
                if m is not None:
                    dl.append(Chem.MolToSmiles(m))
            break
    out["DiffLinker"] = dl

    de = []
    for f in glob.glob(os.path.join(ROOT, "delinker", "*.smi")):
        for line in open(f):
            toks = line.strip().split()
            if len(toks) >= 3:
                de.append(toks[2])
    out["DeLinker"] = de

    sh = []
    sc = os.path.join(ROOT, "shapelinker", "sampled.csv")
    if os.path.exists(sc):
        df = pd.read_csv(sc)
        for col in ("molecules", "SMILES"):
            if col in df.columns:
                sh.extend(df[col].dropna().astype(str).tolist())
                break
    out["ShapeLinker"] = sh

    dp = []
    conv = os.path.join(ROOT, "diffprotacs", "converted_smiles.csv")
    if os.path.exists(conv):
        df = pd.read_csv(conv)
        dp = df["smiles"].dropna().tolist()
    # count raw XYZ regardless
    n_dp_xyz = len(glob.glob(os.path.join(ROOT, "diffprotacs", "GSPT1_CRBN", "*.xyz")))
    out["DiffPROTACs"] = dp
    out["_diffprotacs_xyz"] = n_dp_xyz

    ph = []
    pc = os.path.join(ROOT, "pharmacophore", "generated_candidates.csv")
    if os.path.exists(pc):
        df = pd.read_csv(pc)
        ph = df.iloc[:, 0].dropna().tolist()
    out["Pharmacophore"] = ph

    return out

def main():
    data = load_all()
    rows = []
    per_tool = {}

    for tool in ["Gcoupler", "Pocket2Mol", "DiffSBDD", "PGMG", "DeepICL",
                 "DiffLinker", "DeLinker", "ShapeLinker", "DiffPROTACs",
                 "Pharmacophore"]:
        smis = data.get(tool, [])
        if tool == "DiffPROTACs" and not smis:
            n_xyz = data.get("_diffprotacs_xyz", 0)
            rows.append({"Tool": tool, "Raw": n_xyz, "Valid": 0, "Validity%": 0.0,
                         "Unique": 0, "Novel": 0, "Novelty%": 0.0,
                         "WarheadPass": 0, "WarheadPassRate%": 0.0,
                         "LinkerPass": 0, "LinkerPassRate%": 0.0,
                         "TopFail": "XYZ->SMILES blocked"})
            continue
        seen = set(); uniq = []
        for s in smis:
            c = canon(s)
            if c and c not in seen:
                seen.add(c); uniq.append(c)
        n_raw = len(smis)
        n_valid = sum(1 for s in smis if canon(s))
        novel = [s for s in uniq if s not in KNOWN_GSPT1]
        wp = []; wfails = Counter(); lpass = 0; lfails = Counter()
        for s in uniq:
            p = analyze(s)
            ok, why = warhead_pass(p)
            if ok: wp.append(s)
            else: wfails[why] += 1
            # PROTAC linker cascade
            lk, lm, info = extract_linker(s, WARHEAD_SMI, E3_SMI)
            if lk is None:
                lfails["no_linker_match"] += 1
                continue
            lm2 = Chem.MolFromSmiles(lk)
            if lm2 is None:
                lfails["linker_invalid"] += 1
                continue
            met = compute_linker_metrics(lm2)
            if met is None:
                lfails["no_metrics"] += 1
                continue
            if not (4 <= met["graph_length"] <= 12):
                lfails["linker_len"] += 1; continue
            if not (4 <= met["effective_length"] <= 10):
                lfails["linker_eff"] += 1; continue
            lp = analyze(lk)
            if lp.get("valid") and lp["SA"] <= 5.0:
                lpass += 1
            else:
                lfails["linker_sa"] += 1
        rows.append({
            "Tool": tool, "Raw": n_raw, "Valid": n_valid,
            "Validity%": round(100 * n_valid / max(n_raw, 1), 1),
            "Unique": len(uniq), "Novel": len(novel),
            "Novelty%": round(100 * len(novel) / max(len(uniq), 1), 1),
            "WarheadPass": len(wp),
            "WarheadPassRate%": round(100 * len(wp) / max(len(uniq), 1), 1),
            "LinkerPass": lpass,
            "LinkerPassRate%": round(100 * lpass / max(len(uniq), 1), 1),
            "TopFail": wfails.most_common(1)[0][0] if wfails else "-",
        })
        per_tool[tool] = {"smiles": uniq, "passed": wp}

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(SUMMARY, "warhead_comparison.csv"), index=False)
    print("=" * 100)
    print("GSPT1 DE NOVO WARHEAD GENERATION — ALL TOOLS COMPARISON")
    print("=" * 100)
    print(res.to_string(index=False))

    allp = [(t, s) for t, d in per_tool.items() for s in d["passed"]]
    pd.DataFrame(allp, columns=["tool", "smiles"]).to_csv(
        os.path.join(SUMMARY, "warhead_all_passed.csv"), index=False)
    print("\nTotal warhead-filter-passing molecules:", len(allp))
    print("Saved →", os.path.join(SUMMARY, "warhead_comparison.csv"))

if __name__ == "__main__":
    main()
