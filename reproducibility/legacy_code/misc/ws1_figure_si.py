#!/usr/bin/env python3
"""
WS1 deliverable — OOD figure + SI per-compound table.
Figure: (A) Spearman rho per tier with 95% CI (DC50 + Dmax, N annotated)
        (B) predicted vs known DC50 (log10), colored by tier, 10x band
SI table: per-compound frozen + generalized predictions vs known endpoints.
Outputs: figures/ws1_ood_figure.png/.svg, benchmarking/ws1/si_ws1_per_compound.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
FIG = os.path.join(REV, "figures")
os.makedirs(FIG, exist_ok=True)

sc = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
fr = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_predictions.csv"))

# ─── SI per-compound table ───────────────────────────
si = sc[["canonical_smiles", "tier", "scaffold_ood_flag", "chemical_ood", "chemical_ood_soft",
         "chemical_in", "component_ood_e3", "POI_unseen", "E3_unseen", "deep_ood",
         "E3", "POI_UniProt", "cell_lines", "assay", "e3_match", "warhead_match",
         "dc50_median", "dmax_max", "dc50_censored",
         "gen_pred_dc50_nM", "gen_pred_dmax_pct", "max_tanimoto"]].copy()
si["frozen_pred_dc50_nM"] = si["canonical_smiles"].map(
    fr.set_index("canonical_smiles")["pred_dc50_nM"]) if "pred_dc50_nM" in fr.columns else np.nan
si["frozen_pred_dmax_pct"] = si["canonical_smiles"].map(
    fr.set_index("canonical_smiles")["pred_dmax_pct"]) if "pred_dmax_pct" in fr.columns else np.nan
si["abs_log10_err_gen"] = np.abs(np.log10(si["gen_pred_dc50_nM"]) - np.log10(si["dc50_median"]))
si = si.rename(columns={"canonical_smiles": "SMILES", "E3": "E3_Ligase"})
si.to_csv(os.path.join(REV, "benchmarking", "ws1", "si_ws1_per_compound.csv"), index=False)
print(f"SI table: {len(si)} compounds -> si_ws1_per_compound.csv")

# ─── Metrics for figure ──────────────────────────────
def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

tiers = [("All (non-exact)", sc.index >= 0),
         ("Scaffold-OOD", sc["tier"] == "scaffold_ood"),
         ("Chemical-in", sc["tier"] == "chemical_in"),
         ("Tanimoto 0.4-0.6", sc["chemical_ood_soft"])]
rows = []
for name, mask in tiers:
    sub = sc[mask]
    d50 = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)]
    dm = sub[sub["dmax_max"].notna()]
    if len(d50) >= 5:
        y = np.log10(d50["dc50_median"].values.astype(float)); p = np.log10(d50["gen_pred_dc50_nM"].values.astype(float))
        rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
        rows.append(dict(tier=name, ep="DC50", rho=rho, lo=ci[0], hi=ci[1], n=len(d50)))
    if len(dm) >= 5:
        y = dm["dmax_max"].values.astype(float); p = dm["gen_pred_dmax_pct"].values.astype(float)
        rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
        rows.append(dict(tier=name, ep="Dmax", rho=rho, lo=ci[0], hi=ci[1], n=len(dm)))
mdf = pd.DataFrame(rows)

# ─── Figure ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
x = np.arange(len(mdf)); colors = ["#1f77b4" if e == "DC50" else "#d62728" for e in mdf["ep"]]
ax.bar(x, mdf["rho"], yerr=[mdf["rho"] - mdf["lo"], mdf["hi"] - mdf["rho"]],
       capsize=4, color=colors, alpha=0.85, error_kw=dict(lw=1.2))
for xi, r, n in zip(x, mdf["rho"], mdf["n"]):
    ax.text(xi, r + 0.03, f"n={n}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{t['tier']}\n({t['ep']})" for _, t in mdf.iterrows()], fontsize=8)
ax.set_ylabel("Spearman ρ (generalized model)")
ax.set_title("A. External ranking by OOD tier")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#1f77b4", label="DC50"), Patch(color="#d62728", label="Dmax")], loc="lower left")

ax = axes[1]
d = sc[sc["dc50_median"].notna() & ~sc["dc50_censored"].fillna(False)].copy()
cmap = {"scaffold_ood": "#ff7f0e", "chemical_in": "#2ca02c", "chemical_ood_soft": "#9467bd"}
for t in d["tier"].unique():
    sub = d[d["tier"] == t]
    ax.scatter(np.log10(sub["dc50_median"]), np.log10(sub["gen_pred_dc50_nM"]),
               s=28, alpha=0.8, label=t, color=cmap.get(t, "#7f7f7f"))
lim = [np.log10(1), np.log10(5e4)]
ax.plot(lim, lim, "k--", lw=1, label="y=x")
ax.plot(lim, np.array(lim) - 1, "k:", lw=0.8, label="10×")
ax.plot(lim, np.array(lim) + 1, "k:", lw=0.8)
ax.set_xlabel("Reported DC50 (log10 nM)"); ax.set_ylabel("Predicted DC50 (log10 nM)")
ax.set_title("B. Predicted vs known DC50 (uncensored)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "ws1_ood_figure.png"), dpi=300)
plt.savefig(os.path.join(FIG, "ws1_ood_figure.svg"))
print(f"Figure saved: {FIG}/ws1_ood_figure.png/.svg")
print(mdf.to_string(index=False))
