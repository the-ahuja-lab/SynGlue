#!/usr/bin/env python3
"""
MASTER benchmark — SynGlue vs ALL available PROTAC tools.
Axis A (degradation prediction, external set, per-tool leakage-controlled):
  SynGlue (whole-PROTAC) | PDP neural | PDP XGB | PROTAC-STAN
Axis B (generation, GSPT1): Gcoupler(SynGlue) | ShapeLinker | DeLinker | PGMG |
  DiffPROTACs | DiffLinker | Pocket2Mol | DiffSBDD | pharmacophore enum
Not-runnable tools documented with reasons.
Outputs: outputs/master_benchmark.csv, outputs/not_runnable.csv,
         outputs/master_plot_pred.png/.svg, outputs/master_plot_gen.png/.svg,
         outputs/MASTER_REPORT.md
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# ─── Axis A: prediction (from 06 v2 outputs) ─────────
m6 = pd.read_csv(os.path.join(REV, "rebuttal_analyses", "06_final_external_validation", "outputs", "tool_metrics_common.csv"))
m6n = pd.read_csv(os.path.join(REV, "rebuttal_analyses", "06_final_external_validation", "outputs", "tool_metrics_novel.csv"))
pred = pd.DataFrame({
    "tool": m6["model"], "axis": "prediction (external set)",
    "novel_subset": "external − train overlap (per tool)",
    "n_uncensored_dc50": m6["n"], "spearman_rho": m6["rho"],
    "ci95": m6["ci"], "R2": m6["R2"], "MAE_log10": m6["MAE"],
    "AUC_100nM": m6["auc_100"], "AUC_1uM": m6["auc_1000"],
    "top10_enrichment": m6["top10_enrich"],
})

# ─── Axis B: generation (WS2 GSPT1 results) ──────────
gen = pd.DataFrame([
    {"tool": "Gcoupler (SynGlue)", "axis": "generation (GSPT1)", "generated": 1469, "valid": 1469, "unique": 1469, "pass_all": 238, "pass_rate_pct": 16.2, "note": "RL-guided PROTAC generation"},
    {"tool": "ShapeLinker", "axis": "generation (GSPT1)", "generated": 500, "valid": 497, "unique": 250, "pass_all": 101, "pass_rate_pct": 20.2, "note": "link-invent prior sampling"},
    {"tool": "Pharmacophore enum", "axis": "generation (GSPT1)", "generated": 3000, "valid": 3000, "unique": 3000, "pass_all": 27, "pass_rate_pct": 0.9, "note": "rule-based enumeration"},
    {"tool": "DeLinker", "axis": "generation (GSPT1)", "generated": 100, "valid": 100, "unique": 1, "pass_all": 0, "pass_rate_pct": 0.0, "note": "degenerate E3-only output (ZINC-domain)"},
    {"tool": "PGMG", "axis": "generation (GSPT1)", "generated": 434, "valid": 434, "unique": 434, "pass_all": 405, "pass_rate_pct": 93.3, "note": "de novo pharmacophore; NOT PROTAC-assembled — different category"},
    {"tool": "DiffPROTACs", "axis": "generation (GSPT1)", "generated": 88, "valid": "n/a (XYZ)", "unique": "n/a", "pass_all": "n/a", "pass_rate_pct": np.nan, "note": "3D XYZ only; XYZ→SMILES bond perception fails"},
    {"tool": "DiffLinker", "axis": "generation (GSPT1)", "generated": 30, "valid": "n/a (XYZ)", "unique": "n/a", "pass_all": "n/a", "pass_rate_pct": np.nan, "note": "3D XYZ only; XYZ→SMILES fails"},
    {"tool": "Pocket2Mol", "axis": "generation (GSPT1)", "generated": 116, "valid": 116, "unique": 116, "pass_all": 0, "pass_rate_pct": 0.0, "note": "de novo pocket-guided; no linker (earlier run)"},
    {"tool": "DiffSBDD", "axis": "generation (GSPT1)", "generated": 194, "valid": 194, "unique": 194, "pass_all": 0, "pass_rate_pct": 0.0, "note": "de novo diffusion; no linker (earlier run)"},
])

pred.to_csv(os.path.join(OUT, "master_benchmark_prediction.csv"), index=False)
gen.to_csv(os.path.join(OUT, "master_benchmark_generation.csv"), index=False)

# ─── Not-runnable tools ──────────────────────────────
nr = pd.DataFrame([
    {"tool": "AiPROTAC", "why_not_runnable": "no pretrained weights in repo; requires py3.7+DGL env; PROTAC-ZL graph format"},
    {"tool": "DeepPROTACs", "why_not_runnable": "weights present but inference requires 3D pocket/ligand graph preparation per target (mol2 + PDB pockets)"},
    {"tool": "DegradeMaster", "why_not_runnable": "torch env fixed, but requires 3D target-pocket graphs per POI (PDB prep); not applicable to SMILES-only external set"},
    {"tool": "PROTACable", "why_not_runnable": "multi-stage pipeline (protein prep, pocket, docking) not executable end-to-end"},
    {"tool": "PRosettaC", "why_not_runnable": "requires Rosetta (license) + PatchDock (binary no longer distributed)"},
    {"tool": "DeepLigBuilder", "why_not_runnable": "code never publicly released (only dataset on PKU disk)"},
    {"tool": "PROTAC-RL", "why_not_runnable": "generative RL; requires training/vocab setup, not a predictor"},
    {"tool": "GUNIA", "why_not_runnable": "empty repository directory"},
])
nr.to_csv(os.path.join(OUT, "not_runnable.csv"), index=False)

# ─── Plots ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
d = pred.copy()
ax.bar(d["tool"], d["spearman_rho"], color=["#1f77b4", "#2ca02c", "#2ca02c", "#d62728"], alpha=0.85)
for i, r in d.iterrows():
    lo, hi = [float(x.strip("[]")) for x in str(r["ci95"]).split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels(d["tool"], rotation=15, fontsize=8)
ax.set_ylabel("Spearman ρ vs log10 DC50 (external, novel to each tool)")
ax.set_title("A. Degradation-prediction axis")
ax = axes[1]
g = gen[gen["pass_rate_pct"] == gen["pass_rate_pct"]]
ax.bar(g["tool"], g["pass_rate_pct"], color="#ff7f0e", alpha=0.85)
for x, v in zip(range(len(g)), g["pass_rate_pct"]):
    ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticklabels(g["tool"], rotation=20, fontsize=7)
ax.set_ylabel("Filter pass rate (%)")
ax.set_title("B. Generation axis (GSPT1, standard cascade)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "master_plot_pred.png"), dpi=300)
plt.savefig(os.path.join(OUT, "master_plot_pred.svg"))
print("plots saved")

# ─── Report ──────────────────────────────────────────
lines = ["# MASTER BENCHMARK — SynGlue vs all available PROTAC tools", "",
         "## Axis A — degradation prediction (external set, leakage-controlled)", "",
         pred.to_string(index=False), "",
         "- Per-tool leakage: SynGlue 878 exact (25.0%), PROTAC-STAN 718 (20.4%), PDP bound 774 (22.0%); "
         "common set novel to ALL tools: 1,695 compounds.",
         "- Units: SynGlue predicts DC50 nM; PDP/STAN output P(active) (classifier scores). Ranking/AUC "
         "comparison is task-matched; do not equate P(active) with DC50.", "",
         "## Axis B — generation (GSPT1, standard filter cascade)", "",
         gen.to_string(index=False), "",
         "## Not runnable (documented)", "",
         nr.to_string(index=False), "",
         "## Headline (honest)", "",
         "1. Prediction: no tool ranks leakage-free external PROTACs (best ρ ≈ 0.06 SynGlue; PDP/STAN negative); "
         "PDP/STAN show coarse 1 µM discrimination only.",
         "2. Generation: Gcoupler 16.2% pass vs ShapeLinker 20.2%; DeLinker/Pocket2Mol/DiffSBDD fail on this task; "
         "PGMG different category; DiffPROTACs/DiffLinker lack SMILES-level metrics (XYZ limitation).",
         "3. SynGlue's benchmarkable claims: integrated pipeline (generation + filtering + prioritization), "
         "component-aware 3-token mode on decomposable PROTACs (ρ=0.43, n=70); NOT raw full-pool ranking."]
open(os.path.join(OUT, "MASTER_REPORT.md"), "w").write("\n".join(lines))
print("\n".join(lines[-6:]))
print(f"\nSaved: {OUT}/")
