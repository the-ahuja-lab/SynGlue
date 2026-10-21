#!/usr/bin/env python3
"""
Analysis 5 (R1.3) — systematic external benchmark on a common retrospective test set.
Models: SynGlue frozen, SynGlue generalized (leakage-resistant retrain), PDP.
Inputs: benchmarking/ws1/tiers/tier_scoring_set.csv (predictions),
        pdp_predictions.csv (PDP), ws1_predictions.csv (frozen).
Metrics: DC50 ranking (Spearman + 95% CI), R2/MAE, active/inactive AUC
(class-size gated >=5/class), top-k enrichment, Dmax ranking.
Outputs: outputs/external_benchmark_metrics.csv, outputs/external_benchmark_plot.png/.svg,
         outputs/external_predictions.csv (per-compound, all models)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

gen = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
pdp = pd.read_csv(os.path.join(TIR, "pdp_predictions.csv"))
fr = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_predictions.csv"))
stan = pd.read_csv(os.path.join(TIR, "protacstan_predictions.csv"))

comp = gen[["canonical_smiles", "tier", "dc50_median", "dmax_max", "dc50_censored"]].merge(
    pdp[["SMILES", "pdp_active_proba"]], left_on="canonical_smiles", right_on="SMILES", how="left")
comp = comp.merge(fr[["canonical_smiles", "pred_dc50_nM", "pred_dmax_pct"]], on="canonical_smiles",
                  how="left", suffixes=("", "_f"))
comp = comp.rename(columns={"pred_dc50_nM": "frozen_pred_dc50_nM", "pred_dmax_pct": "frozen_pred_dmax_pct"})
comp = comp.merge(gen[["canonical_smiles", "gen_pred_dc50_nM", "gen_pred_dmax_pct"]],
                  on="canonical_smiles", how="left")
comp = comp.merge(stan[["canonical_smiles", "protacstan_active_proba"]], on="canonical_smiles", how="left")
comp.to_csv(os.path.join(OUT, "external_predictions.csv"), index=False)

d50 = comp[comp["dc50_median"].notna() & ~comp["dc50_censored"].fillna(False)].copy()

def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

MODELS = [("SynGlue frozen", "frozen_pred_dc50_nM", True, "frozen_pred_dmax_pct"),
          ("SynGlue generalized", "gen_pred_dc50_nM", True, "gen_pred_dmax_pct"),
          ("PDP (P active)", "pdp_active_proba", False, None),
          ("PROTAC-STAN (P active)", "protacstan_active_proba", False, None)]

rows = []
for name, col, log, dcol in MODELS:
    m = d50.dropna(subset=[col])
    y = np.log10(m["dc50_median"].values.astype(float))
    p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    # top-10% enrichment: fraction of active (DC50<=100nM) in top 10% by predicted potency
    k = max(1, int(0.1 * len(m)))
    top = m.iloc[np.argsort(p)[:k]] if log else m.iloc[np.argsort(-p)[:k]]
    enrich = float((top["dc50_median"] <= 100).mean()) / max(float((m["dc50_median"] <= 100).mean()), 1e-6)
    # AUC at 100nM (class-gated)
    lb = (m["dc50_median"] <= 100).astype(int)
    auc = roc_auc_score(lb, p) if min((lb == 0).sum(), (lb == 1).sum()) >= 5 else np.nan
    # Dmax ranking
    drho = np.nan
    if dcol and dcol in comp.columns:
        dm = comp[comp["dmax_max"].notna()].dropna(subset=[dcol])
        if len(dm) >= 5:
            drho = stats.spearmanr(dm["dmax_max"], dm[dcol])[0]
    rows.append(dict(model=name, n=len(m), rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]",
                     R2=r2_score(y, p), MAE=float(mean_absolute_error(y, p)),
                     med_abs_log=float(np.median(np.abs(p - y))),
                     auc_100nM=auc, top10_enrichment=enrich, dmax_rho=drho))

md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "external_benchmark_metrics.csv"), index=False)
print(md.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.bar(md["model"], md["rho"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
for i, r in md.iterrows():
    lo, hi = [float(x.strip("[]")) for x in r["ci"].split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
    ax.plot([i - 0.1, i + 0.1], [lo, lo], color="k", lw=1)
    ax.plot([i - 0.1, i + 0.1], [hi, hi], color="k", lw=1)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(md))); ax.set_xticklabels(md["model"], rotation=12, fontsize=8)
ax.set_ylabel("Spearman ρ vs log10 DC50"); ax.set_title("DC50 ranking (n=70)")
ax = axes[1]
ax.bar(md["model"], md["auc_100nM"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticklabels(md["model"], rotation=12, fontsize=8); ax.set_ylim(0, 1)
ax.set_ylabel("AUC (active = DC50 ≤ 100 nM)"); ax.set_title("Active/inactive discrimination")
ax = axes[2]
ax.bar(md["model"], md["top10_enrichment"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticklabels(md["model"], rotation=12, fontsize=8)
ax.set_ylabel("Top-10% enrichment (active ≤100 nM)"); ax.set_title("Prioritization")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "external_benchmark_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "external_benchmark_plot.svg"))
print(f"\nSaved: {OUT}/external_benchmark_metrics.csv, external_benchmark_plot.png/.svg, external_predictions.csv")
