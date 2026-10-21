#!/usr/bin/env python3
"""
WS2 generative benchmark — collect outputs, apply standard filter cascade,
build comparison table + figure (Gcoupler vs DiffPROTACs/DiffLinker/DeLinker/
PGMG/ShapeLinker on GSPT1).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/revision/experiments/gcoupler_benchmark")
from config import GSPT1_WARHEAD_SMILES, GSPT1_E3_LIGAND_SMILES, KNOWN_GSPT1_DEGRADERS, FILTER_PARAMS, RESULTS_DIR
from standard_filters import apply_filter_cascade

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
GEN = os.path.join(REV, "benchmarking", "ws2_gen")
FIG = os.path.join(REV, "figures")
os.makedirs(FIG, exist_ok=True)

WH = GSPT1_WARHEAD_SMILES   # *Nc1nnc(C#CC2CC2)s1
E3 = GSPT1_E3_LIGAND_SMILES  # *N1CCN(...)CC1

def load_delinker(path):
    """DeLinker output: 'input | ground_truth | generated' per line."""
    smis = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                smis.append(parts[2])
            elif len(parts) == 1:
                smis.append(parts[0])
    return smis

def load_shapelinker(path):
    df = pd.read_csv(path)
    return df["molecules"].tolist(), df.get("linker", pd.Series(dtype=str)).tolist()

def load_pgmg(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

print("Loading raw outputs...")
delinker = load_delinker(os.path.join(GEN, "delinker_raw.smi"))
shape_mols, shape_linkers = load_shapelinker(os.path.join(GEN, "shapelinker_sampled.csv"))
pgmg = load_pgmg(os.path.join(GEN, "pgmg_out", "gspt1_phar_result.txt"))
print(f"  DeLinker: {len(delinker)} | ShapeLinker: {len(shape_mols)} | PGMG: {len(pgmg)}")
print(f"  DiffPROTACs XYZ: {len(os.listdir(os.path.join(GEN,'diffprotacs_out','GSPT1_CRBN')))} | DiffLinker XYZ: {len(os.listdir(os.path.join(GEN,'difflinker_out')))}")

# ─── Apply filter cascade ────────────────────────────
def run(smiles_list, tool, use_linker=True, known=None):
    df = apply_filter_cascade(smiles_list, WH, E3, known_degraders=known, params=FILTER_PARAMS)
    df["tool"] = tool
    return df

known = KNOWN_GSPT1_DEGRADERS
res = {}
res["DeLinker"] = run(delinker, "DeLinker", use_linker=True, known=known)
res["ShapeLinker"] = run(shape_mols, "ShapeLinker", use_linker=True, known=known)
# PGMG: de novo standalone molecules — linker extraction not applicable
res["PGMG"] = run(pgmg, "PGMG", use_linker=False, known=known)
if "linker_smiles" in res["PGMG"].columns:
    res["PGMG"]["pass_linker_graph"] = True
    res["PGMG"]["pass_linker_effective"] = True
    res["PGMG"]["pass_all"] = res["PGMG"]["pass_validity"] & res["PGMG"]["pass_physchem_mw"] & \
        res["PGMG"]["pass_physchem_logp"] & res["PGMG"]["pass_physchem_tpsa"] & res["PGMG"]["pass_physchem_rot"] & \
        res["PGMG"]["pass_sa"] & res["PGMG"]["pass_ro3"] & res["PGMG"]["pass_novelty"]
    print("PGMG linker filters overridden (de novo generator)")

os.makedirs(os.path.join(GEN, "collected"), exist_ok=True)
for t, df in res.items():
    df.to_csv(os.path.join(GEN, "collected", f"{t.lower()}_filtered.csv"), index=False)

# ─── Comparison table ────────────────────────────────
# Gcoupler + Pharmacophore from existing experiment summaries
gc = pd.read_csv(os.path.join(RESULTS_DIR, "gcoupler", "filtered_results.csv"))
ph = pd.read_csv(os.path.join(RESULTS_DIR, "pharmacophore", "filtered_results.csv"))

def summ(df, tool):
    n = len(df)
    valid = int(df["pass_validity"].sum()) if "pass_validity" in df.columns else n
    uniq = df["canonical_smiles"].nunique() if "canonical_smiles" in df.columns else n
    sa = int(df["pass_sa"].sum()) if "pass_sa" in df.columns else np.nan
    allpass = int(df["pass_all"].sum()) if "pass_all" in df.columns else np.nan
    return dict(tool=tool, generated=n, valid=valid, unique=uniq, sa_pass=sa,
                pass_all=allpass, pass_rate=(100 * allpass / n) if allpass == allpass else np.nan)

rows = []
rows.append(summ(gc, "Gcoupler (SynGlue)"))
rows.append(summ(ph, "Pharmacophore enum"))
for t, df in res.items():
    s = summ(df, t)
    if "pass_linker_graph" in df.columns:
        s["linker_ok"] = int(df["pass_linker_graph"].fillna(False).sum())
    rows.append(s)
# 3D tools: counts only (XYZ->SMILES bond perception fails — documented B-003)
rows.append(dict(tool="DiffPROTACs", generated=88, valid="n/a (XYZ)", unique="n/a", sa_pass="n/a",
                 pass_all="n/a", pass_rate=np.nan))
rows.append(dict(tool="DiffLinker", generated=30, valid="n/a (XYZ)", unique="n/a", sa_pass="n/a",
                 pass_all="n/a", pass_rate=np.nan))

cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(GEN, "ws2_gen_comparison.csv"), index=False)
print("\n" + cmp.to_string(index=False))

# ─── Figure ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tools = [r["tool"] for r in rows if isinstance(r.get("pass_rate", None), (int, float)) and r["pass_rate"] == r["pass_rate"]]
rates = [r["pass_rate"] for r in rows if isinstance(r.get("pass_rate", None), (int, float)) and r["pass_rate"] == r["pass_rate"]]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(tools, rates, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(tools)], alpha=0.85)
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.4, f"{r:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Pass rate (%) — standard degrader filter cascade")
ax.set_title("WS2 generative benchmark (GSPT1) — filter pass rates")
ax.set_ylim(0, max(rates) * 1.25)
plt.xticks(rotation=12, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "ws2_gen_comparison.png"), dpi=300)
plt.savefig(os.path.join(FIG, "ws2_gen_comparison.svg"))
print(f"\nFigure: {FIG}/ws2_gen_comparison.png")

# ─── Report ──────────────────────────────────────────
lines = ["# WS2 Generative Benchmark — GSPT1 (Gcoupler vs tools)", "",
         "Common inputs: warhead `*Nc1nnc(C#CC2CC2)s1`, E3 `*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1` (CRBN)",
         "Filter cascade: validity → MW≤1000 → logP≤5 → TPSA 40–200 → rot≤15 → SA≤4 → RO3≤2 → linker 4–12/4–10 → novelty", "",
         "| Tool | Generated | Valid | Unique | SA pass | Linker-ok | Pass-all | Rate |",
         "|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['tool']} | {r['generated']} | {r['valid']} | {r['unique']} | {r['sa_pass']} | "
                 f"{r.get('linker_ok', 'n/a')} | {r['pass_all']} | {r['pass_rate']:.1f}% |" if r['pass_rate'] == r['pass_rate']
                 else f"| {r['tool']} | {r['generated']} | {r['valid']} | {r['unique']} | n/a | n/a | n/a | n/a |")
lines += ["", "## Notes", "",
          "- DiffPROTACs (88) and DiffLinker (30): generated 3D XYZ successfully; XYZ→SMILES bond perception fails (documented B-003) — SMILES-level metrics n/a, reported as coverage only",
          "- PGMG: de novo molecule generator (pharmacophore-conditioned, no warhead/E3 context) — linker filters not applicable",
          "- ShapeLinker: link-invent prior sampling (no RL agent for CRBN/GSPT1; trained agent available for VHL/5T35 only)",
          "- DeLinker: pretrained ZINC model, CPU mode",
          "- Novelty check uses placeholder known-degrader list (config KNOWN_GSPT1_DEGRADERS) — validate before final", ""]
open(os.path.join(GEN, "ws2_gen_report.md"), "w").write("\n".join(lines))
print("\n".join(lines[-14:]))
