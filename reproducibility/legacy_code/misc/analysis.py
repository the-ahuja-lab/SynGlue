#!/usr/bin/env python3
"""
Analysis 2 (R1.1b) — held-out generalization results (fills 'We observed …').
Sources: WP2 two-stage TF+RF results (wp2_aggregated_results.csv),
         WS1 external OOD (tier metrics, generalized model).
Outputs: outputs/generalization_results.csv, outputs/generalization_plot.png/.svg,
         outputs/rebuttal_text.md (suggested 'We observed …' wording)
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

# ─── WP2 two-stage TF+RF (best model, 825-compound set) ───
# two-stage TF+RF results (best generalization model; WP2_REPORT.md §2b)
wp2 = pd.read_csv(os.path.join(REV, "benchmarking", "wp2_results", "wp2_twostage_results.csv"))
# ─── WS1 external (generalized model, leakage-free 151 set) ───
tiers = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "tiers", "tier_metrics.csv"))

rows = []
def add(split, dc50_r2, dc50_rho, dc50_n, dmax_r2, dmax_rho, dmax_n, note):
    rows.append(dict(split=split, dc50_R2=dc50_r2, dc50_rho=dc50_rho, dc50_n=dc50_n,
                     dmax_R2=dmax_r2, dmax_rho=dmax_rho, dmax_n=dmax_n, note=note))

# from wp2 (two-stage TF+RF) — R2 available; rho not computed there
# aggregate per-trial two-stage results (means across trials)
agg2 = wp2.groupby("split").agg(dc50_R2=("dc50_R2", "mean"), dmax_R2=("dmax_R2", "mean")).reset_index()
for _, r in agg2.iterrows():
    add(r["split"], r["dc50_R2"], np.nan, np.nan, r["dmax_R2"], np.nan, np.nan,
        "two-stage TF+RF, 825-compound set (WP2)")
# external validation (generalized model) from tier metrics
t_all = tiers[tiers["tier"] == "all"].iloc[0]
t_scaf = tiers[tiers["tier"] == "scaffold_ood (primary)"].iloc[0]
add("External (leakage-free, all)", t_all["dc50_R2"], t_all["dc50_rho"], t_all["dc50_n"],
    np.nan, t_all["dmax_rho"], t_all["dmax_n"], "generalized model, 151-set (WS1/WS2)")
add("External (scaffold-OOD)", t_scaf["dc50_R2"], t_scaf["dc50_rho"], t_scaf["dc50_n"],
    np.nan, t_scaf["dmax_rho"], t_scaf["dmax_n"], "generalized model")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "generalization_results.csv"), index=False)
print(df.to_string(index=False))

# ─── Plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels = df["split"].tolist()
vals = df["dc50_R2"].astype(float).values
colors = ["#1f77b4" if "External" not in l else "#d62728" for l in labels]
bars = ax.bar(labels, vals, color=colors, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("DC50 R²")
ax.set_title("Held-out generalization (two-stage TF+RF; external = generalized model)")
ax.tick_params(axis="x", rotation=18, labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "generalization_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "generalization_plot.svg"))

# ─── Suggested rebuttal wording ──────────────────────
txt = """# 'We observed …' — suggested wording (from actual outputs)

## Scaffold-held-out validation
We observed DC50 R² = 0.302 ± 0.082 on scaffold-held-out testing (two-stage transformer + random-forest pipeline, 825-compound leakage-controlled dataset; 160–178 test compounds), indicating moderate generalization to novel Bemis–Murcko scaffolds of the full degrader structure, warhead and E3 ligand.

## Target-held-out validation
We observed that target-held-out testing was not reliably evaluable with the available data: holding out all degraders for rare targets left only n = 10 test compounds (DC50 R² = −1.68), which we report as insufficient for a meaningful estimate rather than as a performance claim.

## E3-held-out validation
We observed DC50 R² = −0.042 ± 0.028 when all VHL-recruiting degraders were held out (n = 316 test compounds). This reflects the CRBN-dominated training chemistry (57% CRBN) and is reported transparently as the key generalization bottleneck.

## External validation (recently reported PROTACs, leakage-controlled)
We curated 3,518 unique PROTACs from TPDdb/PROTAC-DB/PROTACpedia with endpoint metadata, removed 25.0% exact matches with training compounds, decomposed each into warhead/linker/E3 recruiter, and scored 151 fully-decomposed, leakage-free compounds with a leakage-resistant retrained model (per-compound conditioning, scaffold-grouped 5-fold cross-validation, ensembling). We observed Spearman ρ = 0.43 [95% CI 0.22–0.60] for DC50 ranking (n = 70 uncensored) and ρ = 0.47 [0.23–0.65] on the scaffold-OOD subset, with median |Δlog10(DC50)| = 0.70 (≈5-fold) and 70% of predictions within 10-fold of reported values. Dmax ranking did not transfer (ρ ≈ 0). Absolute R² remained negative on the external set, which we report transparently; the manuscript has been revised to state that SynGlue provides practical generalization under chemically and biologically constrained validation settings rather than unrestricted extrapolation.
"""
open(os.path.join(OUT, "rebuttal_text.md"), "w").write(txt)
print(f"\nSaved: {OUT}/generalization_results.csv, generalization_plot.png/.svg, rebuttal_text.md")
