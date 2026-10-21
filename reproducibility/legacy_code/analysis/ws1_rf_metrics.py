#!/usr/bin/env python3
"""
WS1 part 2 — frozen RF predictions + metrics.
Run in `rf_predict` env (numpy 1.21.5, scikit-learn 1.0.2, joblib) with `python -s`
to avoid user-site numpy 2.x shadowing:
    conda run -n rf_predict python -s ws1_rf_metrics.py

Reads:  benchmarking/ws1/ws1_Z.npy, ws1_scoring_set.csv
Output: benchmarking/ws1/ws1_predictions.csv, ws1_metrics.md
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"
RF_DC50 = os.path.join(SYNGLUE, "models", "rf_dc50.joblib")
RF_DMAX = os.path.join(SYNGLUE, "models", "rf_dmax.joblib")
N_BOOT = 2000

Z = np.load(os.path.join(OUT, "ws1_Z.npy"))
sel = pd.read_csv(os.path.join(OUT, "ws1_scoring_set.csv"))
print(f"Z: {Z.shape} | scoring set: {len(sel)}")

rf50 = joblib.load(RF_DC50)
rfmax = joblib.load(RF_DMAX)
sel["pred_dc50_nM"] = 10 ** rf50.predict(Z)   # log10 -> nM
sel["pred_dmax_pct"] = rfmax.predict(Z)
sel.to_csv(os.path.join(OUT, "ws1_predictions.csv"), index=False)

def bootstrap_ci(x, y, metric, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        try:
            vals.append(metric(x[idx], y[idx]))
        except Exception:
            pass
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)

def spearman(x, y):
    return stats.spearmanr(x, y)[0]

def metrics_table(sub, name):
    lines = [f"### {name} (N={len(sub)})", "",
             "| Endpoint | N | R² | RMSE | MAE | Spearman ρ | ρ 95% CI |",
             "|----------|---|----|----|----|----|----|"]
    for col, pred, label, log in [("dc50_median", "pred_dc50_nM", "DC50 (nM)", True),
                                  ("dmax_max", "pred_dmax_pct", "Dmax (%)", False)]:
        mask = sub[col].notna() & sub[pred].notna() & (~sub.get("dc50_censored", pd.Series(False, index=sub.index)).fillna(False)) if col == "dc50_median" else (sub[col].notna() & sub[pred].notna())
        y = np.log10(sub.loc[mask, col].values.astype(float)) if log else sub.loc[mask, col].values.astype(float)
        p = np.log10(sub.loc[mask, pred].values.astype(float)) if log else sub.loc[mask, pred].values.astype(float)
        if len(y) < 5:
            lines.append(f"| {label} | {len(y)} | n/a | | | | |"); continue
        r2 = r2_score(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        mae = mean_absolute_error(y, p)
        rho = spearman(y, p)
        ci = bootstrap_ci(y, p, spearman)
        lines.append(f"| {label} | {len(y)} | {r2:.3f} | {rmse:.3f} | {mae:.3f} | {rho:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines.append("")
    return "\n".join(lines)

md = ["# WS1 Frozen SynGlue Scoring — External Decomposed PROTACs", "",
      f"- Scoring set: {len(sel)} non-exact, fully decomposed, endpoint-bearing compounds",
      f"- Exact/censored excluded from regression; censored concordance reported separately", ""]

md.append(metrics_table(sel, "All scoring set (uncensored)"))
for tier in ["scaffold_ood", "chemical_in"]:
    sub = sel[sel["tier"] == tier]
    if len(sub):
        md.append(metrics_table(sub, f"Tier: {tier}"))

csub = sel[sel.get("dc50_censored", pd.Series(False, index=sel.index)).fillna(False)]
if len(csub):
    hi_hi = int(((csub["dc50_median"] > 1000) & (csub["pred_dc50_nM"] > 1000)).sum())
    md.append(f"### Censored DC50 concordance (N={len(csub)})\n\n- reported >1000 nM AND predicted >1000 nM: {hi_hi}\n")

# assay-matched series: same cell line+assay with >=3 compounds
ser = sel.dropna(subset=["cell_lines", "assay"]).copy()
ser["series"] = ser["cell_lines"] + " | " + ser["assay"]
big = ser.groupby("series").filter(lambda g: g["dc50_median"].notna().sum() >= 3)
md.append("### Assay-matched series ranking (within-series Spearman, DC50)")
if len(big):
    rhos = []
    for name, g in big.groupby("series"):
        m = g["dc50_median"].notna()
        if m.sum() < 3: continue
        rho = spearman(np.log10(g.loc[m, "dc50_median"].values.astype(float)),
                       np.log10(g.loc[m, "pred_dc50_nM"].values.astype(float)))
        rhos.append((name, rho, int(m.sum())))
    for name, rho, n in sorted(rhos, key=lambda t: -t[1]):
        md.append(f"- {name} (n={n}): ρ={rho:.3f}")
    md.append(f"\n- Series with n≥3: {len(rhos)} | mean ρ: {np.mean([r[1] for r in rhos]):.3f}")
else:
    md.append("- No series with ≥3 scored compounds")

open(os.path.join(OUT, "ws1_metrics.md"), "w").write("\n".join(md))
print("\n".join(md))
print(f"\nSaved: {OUT}/ws1_predictions.csv + ws1_metrics.md")
