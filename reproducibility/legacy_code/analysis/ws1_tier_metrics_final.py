#!/usr/bin/env python3
"""Regenerate tier metrics with primary-tier view + 'are we nearby' stats."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TIR = "benchmarking/ws1/tiers"
sc = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))

def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

def mrow(sub, endpoint, pred, log=False):
    m = sub[endpoint].notna() & sub[pred].notna()
    if log:
        y = np.log10(sub.loc[m, endpoint].values.astype(float)); p = np.log10(sub.loc[m, pred].values.astype(float))
    else:
        y = sub.loc[m, endpoint].values.astype(float); p = sub.loc[m, pred].values.astype(float)
    if len(y) < 5: return dict(n=len(y))
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    return dict(n=len(y), R2=r2_score(y, p), rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]",
                RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                med_abs_log=float(np.median(np.abs(np.log10(sub.loc[m, pred].values.astype(float)) -
                                                   np.log10(sub.loc[m, endpoint].values.astype(float))))),
                pct1=float(np.mean(np.abs(np.log10(sub.loc[m, pred].values.astype(float)) -
                                         np.log10(sub.loc[m, endpoint].values.astype(float))) <= 1.0)))

rows = []
for name, mask in [
    ("all", pd.Series(True, index=sc.index)),
    ("scaffold_ood (primary)", sc["tier"] == "scaffold_ood"),
    ("chemical_in (primary)", sc["tier"] == "chemical_in"),
    ("chemical_ood_soft (flag)", sc["chemical_ood_soft"]),
]:
    sub = sc[mask].copy()
    d50 = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)]
    dm = sub[sub["dmax_max"].notna()]
    r50 = mrow(d50, "dc50_median", "gen_pred_dc50_nM", log=True)
    rmax = mrow(dm, "dmax_max", "gen_pred_dmax_pct")
    rows.append(dict(tier=name, n=len(sub), dc50_n=r50.get("n", 0), dc50_R2=r50.get("R2", np.nan),
                     dc50_rho=r50.get("rho", np.nan), dc50_ci=r50.get("ci", ""),
                     dc50_med_abs_log=r50.get("med_abs_log", np.nan), dc50_pct1log=r50.get("pct1", np.nan),
                     dmax_n=rmax.get("n", 0), dmax_R2=rmax.get("R2", np.nan),
                     dmax_rho=rmax.get("rho", np.nan), dmax_ci=rmax.get("ci", "")))
md = pd.DataFrame(rows)
md.to_csv(os.path.join(TIR, "tier_metrics.csv"), index=False)
lines = ["# WS1 Tiered Inference — Generalized Model (5-fold ensemble, predict-then-average)", "",
         f"- Scoring set: {len(sc)} leakage-free decomposed compounds; SAME generalized model for all tiers",
         "- 'are we nearby': med |Δlog10(DC50)| (0=exact; <0.5≈within 3×; <1≈within 10×) and % within 1 log10", "",
         "| Tier | n | DC50 n | R² | ρ [95% CI] | med|Δlog10| | %≤1log | Dmax n | R² | ρ [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in md.iterrows():
    lines.append(f"| {r['tier']} | {r['n']} | {r['dc50_n']} | {r['dc50_R2']:.3f} | {r['dc50_rho']:.3f} {r['dc50_ci']} | "
                 f"{r['dc50_med_abs_log']:.2f} | {100*r['dc50_pct1log']:.0f}% | {r['dmax_n']} | {r['dmax_R2']:.3f} | {r['dmax_rho']:.3f} {r['dmax_ci']} |")
lines += ["", "## Deep-OOD tiers: not evaluable in scorable set (honest)", "",
          f"- chemical_ood (T<0.4): {int(sc['chemical_ood'].sum())} scored compounds",
          f"- POI_unseen: {int(sc['POI_unseen'].sum())} | E3_unseen: {int(sc['E3_unseen'].sum())} | component_ood_e3: {int(sc['component_ood_e3'].sum())}",
          "- The scorable set (decomposed + endpoints + non-exact) is chemically close to train (all T≥0.4); deepest tiers require more/wider external data.", ""]
open(os.path.join(TIR, "tier_metrics.md"), "w").write("\n".join(lines))
print("\n".join(lines))
