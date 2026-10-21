#!/usr/bin/env python3
"""
WS2 — common-set comparison of degradation-prediction models on the WS1
external scoring set (151 leakage-free decomposed PROTACs).
Models:
  SynGlue frozen      (multitask_transformer.pt + rf_*)      -> DC50 nM, Dmax %
  SynGlue generalized (5-fold ensemble, per-compound cond.)  -> DC50 nM, Dmax %
  PDP                 (PROTAC-Degradation-Predictor)         -> P(active) [classifier]
Coverage + failure notes for PROTAC-STAN / DeepPROTACs / DegradeMaster / AiPROTAC.
Outputs: benchmarking/ws1/tiers/ws2_comparison.csv, ws2_metrics.csv, ws2_metrics.md,
         figures/ws2_model_comparison.png/.svg
"""
import os, warnings
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
FIG = os.path.join(REV, "figures")
os.makedirs(FIG, exist_ok=True)

gen = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
pdp = pd.read_csv(os.path.join(TIR, "pdp_predictions.csv"))
fr = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_predictions.csv"))

comp = gen[["canonical_smiles", "tier", "dc50_median", "dmax_max", "dc50_censored"]].merge(
    pdp[["SMILES", "pdp_active_proba"]], left_on="canonical_smiles", right_on="SMILES", how="left")
comp = comp.merge(fr[["canonical_smiles", "pred_dc50_nM", "pred_dmax_pct"]],
                  on="canonical_smiles", how="left", suffixes=("", "_frozen"))
comp["frozen_pred_dc50_nM"] = comp["pred_dc50_nM"]
comp["frozen_pred_dmax_pct"] = comp["pred_dmax_pct"]
comp = comp.drop(columns=["pred_dc50_nM", "pred_dmax_pct"])
comp = comp.merge(gen[["canonical_smiles", "gen_pred_dc50_nM", "gen_pred_dmax_pct"]],
                  on="canonical_smiles", how="left")
comp.to_csv(os.path.join(TIR, "ws2_comparison.csv"), index=False)

d50 = comp[comp["dc50_median"].notna() & ~comp["dc50_censored"].fillna(False)].copy()
d50["log10_known"] = np.log10(d50["dc50_median"])

def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

def summ(df, col_pred, log_pred=False, label=""):
    if log_pred:
        p = np.log10(df[col_pred].values.astype(float))
    else:
        p = df[col_pred].values.astype(float)
    y = df["log10_known"].values.astype(float)
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    return dict(model=label, n=len(df), rho=rho, lo=ci[0], hi=ci[1],
                R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)),
                med_abs_log=float(np.median(np.abs(p - y))))

rows = []
m = d50.dropna(subset=["frozen_pred_dc50_nM"])
rows.append(summ(m, "frozen_pred_dc50_nM", log_pred=True, label="SynGlue frozen"))
m = d50.dropna(subset=["gen_pred_dc50_nM"])
rows.append(summ(m, "gen_pred_dc50_nM", log_pred=True, label="SynGlue generalized"))
m = d50.dropna(subset=["pdp_active_proba"])
rows.append(summ(m, "pdp_active_proba", log_pred=False, label="PDP"))

# classification view: active = DC50 <= thr; AUC only if both classes >= 5
for thr in [30, 100]:
    yb = (d50["dc50_median"] <= thr).astype(int)
    if min((yb == 0).sum(), (yb == 1).sum()) < 5:
        continue
    for col, lab in [("frozen_pred_dc50_nM", "SynGlue frozen"), ("gen_pred_dc50_nM", "SynGlue generalized"), ("pdp_active_proba", "PDP")]:
        m = d50.dropna(subset=[col])
        if m["dc50_median"].nunique() > 1:
            s = np.log10(m[col].values.astype(float)) if col.startswith("frozen") or col.startswith("gen") else m[col].values.astype(float)
            lb = (m["dc50_median"] <= thr).astype(int)
            if min((lb == 0).sum(), (lb == 1).sum()) < 5:
                continue
            try:
                auc = roc_auc_score(lb, s)
            except Exception:
                auc = np.nan
            rows.append(dict(model=f"{lab} (active<= {thr} nM)", n=len(m), rho=np.nan, lo=np.nan,
                             hi=np.nan, R2=np.nan, RMSE=np.nan, MAE=np.nan, med_abs_log=np.nan,
                             AUC=auc))
mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(TIR, "ws2_metrics.csv"), index=False)

# Dmax: only SynGlue models have Dmax predictions (PDP has no Dmax head)
dm = comp[comp["dmax_max"].notna()].copy()
rows_dm = []
for col, lab in [("frozen_pred_dmax_pct", "SynGlue frozen"), ("gen_pred_dmax_pct", "SynGlue generalized")]:
    m = dm.dropna(subset=[col])
    rho = stats.spearmanr(m["dmax_max"], m[col])[0]; ci = bci(m["dmax_max"].values.astype(float), m[col].values.astype(float))
    rows_dm.append(dict(model=lab, n=len(m), rho=rho, lo=ci[0], hi=ci[1],
                        R2=r2_score(m["dmax_max"], m[col]), RMSE=float(np.sqrt(mean_squared_error(m["dmax_max"], m[col]))),
                        MAE=float(mean_absolute_error(m["dmax_max"], m[col])), med_abs_log=np.nan))
dmdf = pd.DataFrame(rows_dm)
dmdf.to_csv(os.path.join(TIR, "ws2_metrics_dmax.csv"), index=False)

# ─── Figure ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
dc = mdf[mdf["rho"].notna() & mdf["AUC"].isna()]
ax.bar(dc["model"], dc["rho"], yerr=[dc["rho"] - dc["lo"], dc["hi"] - dc["rho"]],
       capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
for xi, r, n in zip(range(len(dc)), dc["rho"], dc["n"]):
    ax.text(xi, r + 0.02, f"n={n}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels(dc["model"], rotation=15, fontsize=8)
ax.set_ylabel("Spearman ρ vs log10 DC50")
ax.set_title("A. DC50 ranking (uncensored, n=70)")

ax = axes[1]
aucdf = mdf[mdf["AUC"].notna()]
for thr in [100, 1000]:
    sub = aucdf[aucdf["model"].str.contains(str(thr))]
    ax.bar([f"{m}\n({thr} nM)" for m in sub["model"].str.replace(f" (active<= {thr} nM)", "")], sub["AUC"],
           alpha=0.75, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=8)
ax.set_ylabel("AUC (active = DC50 ≤ threshold)")
ax.set_title("B. Active/inactive discrimination")

ax = axes[2]
x = d50["log10_known"]
ax.scatter(x, np.log10(d50["frozen_pred_dc50_nM"]), s=20, alpha=0.6, label="SynGlue frozen", color="#1f77b4")
ax.scatter(x, np.log10(d50["gen_pred_dc50_nM"]), s=20, alpha=0.6, label="SynGlue gen", color="#ff7f0e")
ax.plot([1, 5], [1, 5], "k--", lw=1)
ax.plot([1, 5], [0, 4], "k:", lw=0.7); ax.plot([1, 5], [2, 6], "k:", lw=0.7)
ax.set_xlabel("Reported DC50 (log10 nM)"); ax.set_ylabel("Predicted (log10 nM)")
ax.set_title("C. SynGlue predictions vs known")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "ws2_model_comparison.png"), dpi=300)
plt.savefig(os.path.join(FIG, "ws2_model_comparison.svg"))
print("Figure saved:", FIG + "/ws2_model_comparison.png")

# ─── Report ──────────────────────────────────────────
lines = ["# WS2 — Common-Set Model Comparison (external 151 set)", "",
         "| Model | Type | Coverage | DC50 ρ [95% CI] | R² | MAE(log10) | med|Δlog10| | AUC(≤30nM) | AUC(≤100nM) | Dmax ρ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
# lookup: (model_base, thr) -> AUC
auclu = {}
for _, r in aucdf.iterrows():
    for thr in [30, 100]:
        suf = f" (active<= {thr} nM)"
        if r["model"].endswith(suf):
            auclu[(r["model"][: -len(suf)], thr)] = r["AUC"]
for _, r in mdf[mdf["rho"].notna()].iterrows():
    drho = dmdf[dmdf["model"] == r["model"]]["rho"]
    drho_v = f"{drho.iloc[0]:.3f}" if len(drho) else "n/a"
    a30 = auclu.get((r["model"], 30), np.nan)
    a100 = auclu.get((r["model"], 100), np.nan)
    lines.append(f"| {r['model']} | | {r['n']} | {r['rho']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] | {r['R2']:.3f} | "
                 f"{r['MAE']:.3f} | {r['med_abs_log']:.2f} | {a30:.2f} | {a100:.2f} | {drho_v} |")
lines += ["", "## Coverage / failures", "",
          "- SynGlue frozen: 151/151 (DC50+Dmax)", "- SynGlue generalized: 151/151 (DC50+Dmax)",
          "- PDP: 151/151 active probability (E3 vocab ok; cell-line fallback to HeLa where needed)",
          "- PROTAC-STAN: NOT RUN — needs PROTAC graph construction + ESM protein embeddings + checkpoint; 2-class classifier (not directly comparable to DC50)",
          "- DeepPROTACs: NOT RUN — requires 3D mol2 structures (pocket+ligand+linker)",
          "- DegradeMaster: NOT RUN — env torch broken",
          "- AiPROTAC: NOT RUN — Python 3.7+DGL env, PROTAC-ZL-format input, AR-focused training",
          "", "Note: PDP outputs P(active) — a classifier score, NOT a DC50. Ranking/AUC comparisons only."]
open(os.path.join(TIR, "ws2_metrics.md"), "w").write("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved: {TIR}/ws2_comparison.csv, ws2_metrics.csv, ws2_metrics.md")
