#!/usr/bin/env python3
"""
FINAL direct comparison — SynGlue vs TACK (KDD'26) on DC50 & Dmax (same units).
Leakage control per tool:
  - external pool: TPDdb+PROTACpedia (no PROTAC-DB)
  - SynGlue: exact matches vs SynGlue train removed
  - TACK: trained on TACK train split -> evaluate on SMILES_Held_Out=True subset
    (compounds TACK never trained on), also report full-pool numbers
  - PDP/STAN included as classifier references (P(active), labeled)
Outputs: outputs/final_direct_comparison.csv, direct_metrics.csv,
         direct_plot_dc50.png/.svg, direct_plot_dmax.png/.svg, FINAL_REPORT.md
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
STAN_TRAIN = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN/data/PROTAC-fine"

def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

# ─── master (no PROTAC-DB) ───────────────────────────
ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
}))
master = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
master = master.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

# train sets
tr = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
syn_train = {c for c in map(canon, tr["PROTAC_SMILES"].dropna().astype(str).unique()) if c}
stan_smiles = []
for f in ["train_compound_smiles.csv", "test_compound_smiles.csv"]:
    stan_smiles += pd.read_csv(os.path.join(STAN_TRAIN, f), skiprows=1, header=None, names=["s"])["s"].dropna().astype(str).tolist()
stan_train = {c for c in map(canon, stan_smiles) if c}
master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)

# predictions
tack = pd.read_csv(os.path.join(TIR, "tack_predictions.csv"))
tack_extra = pd.read_csv(os.path.join(TIR, "tack_extra_predictions.csv"))
pdc50 = pd.read_csv(os.path.join(TIR, "protacdc50_predictions.csv"))
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))

master = master.merge(tack[["canonical_SMILES", "tack_pred_dc50_nM", "tack_pred_dmax_pct", "held_out"]], on="canonical_SMILES", how="left")
master = master.merge(tack_extra[["canonical_SMILES", "tack_mlp_dc50_nM"]], on="canonical_SMILES", how="left")
pdc = pdc50[["canonical_smiles", "protacdc50_dc50_nM"]].rename(columns={"canonical_smiles": "pdc50_key"})
master = master.merge(pdc, left_on="canonical_SMILES", right_on="pdc50_key", how="left").drop(columns=["pdc50_key"], errors="ignore")
master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]], on="canonical_SMILES", how="left")
master = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left", suffixes=("", "_t3"))
master = master.merge(pdp_full[["SMILES", "pdp_active_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
master["pdp_overlap_bound"] = master["exact_in_stan"]  # PROTAC-DB already excluded

master.to_csv(os.path.join(OUT, "final_direct_comparison.csv"), index=False)
print(f"master: {len(master)} | TACK scored: {master['tack_pred_dc50_nM'].notna().sum()} | TACK held-out: {int(master['held_out'].fillna(False).sum())}")

# ─── metrics ─────────────────────────────────────────
def bci(x, y, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(fn(x[ix], y[ix]))
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

def metrics(sub, col, log=False, dcol=None):
    m = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    out = dict(n=len(m))
    if len(m) >= 5:
        y = np.log10(m["dc50_median"].values.astype(float))
        p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
        rho = stats.spearmanr(y, p)[0]; ci = bci(y, p, lambda a, b: stats.spearmanr(a, b)[0])
        out.update(rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]", tau=stats.kendalltau(y, p)[0],
                   R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                   MAE=float(mean_absolute_error(y, p)))
    if dcol is not None:
        dm = sub[sub["dmax_max"].notna()].dropna(subset=[dcol])
        if len(dm) >= 5:
            out["dmax_rho"] = stats.spearmanr(dm["dmax_max"], dm[dcol])[0]
            out["dmax_MAE"] = float(mean_absolute_error(dm["dmax_max"], dm[dcol]))
    return out

# subsets
common = master[~master["exact_in_synglue"] & ~master["exact_in_stan"]]
tack_ho = master[master["held_out"].fillna(False) & ~master["exact_in_synglue"]]
print(f"common (novel to SynGlue+STAN): {len(common)} | TACK-held-out & SynGlue-novel: {len(tack_ho)}")

rows = []
def add_rows(sub, tag):
    rows.append(dict(subset=tag, **{"model": "SynGlue 3-token (DC50)", **metrics(sub, "gen_pred_dc50_nM", log=True)}))
    rows.append(dict(subset=tag, **{"model": "SynGlue whole-PROTAC", **metrics(sub, "gen_whole_pred_dc50_nM", log=True, dcol="gen_whole_pred_dmax_pct")}))
    rows.append(dict(subset=tag, **{"model": "TACK (DC50 regressor)", **metrics(sub, "tack_pred_dc50_nM", log=True, dcol="tack_pred_dmax_pct")}))
    rows.append(dict(subset=tag, **{"model": "TACK-MLP (DC50 regressor)", **metrics(sub, "tack_mlp_dc50_nM", log=True)}))
    rows.append(dict(subset=tag, **{"model": "PROTAC-DC50 (GNN+FP, Warwick)", **metrics(sub, "protacdc50_dc50_nM", log=True)}))
    rows.append(dict(subset=tag, **{"model": "PDP (P active)", **metrics(sub, "pdp_active_proba")}))
    rows.append(dict(subset=tag, **{"model": "PROTAC-STAN (P active)", **metrics(sub, "protacstan_active_proba")}))

add_rows(common, "common (novel to SynGlue & STAN)")
add_rows(tack_ho, "TACK-held-out & SynGlue-novel")

md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "direct_metrics.csv"), index=False)
print("\n" + md.to_string(index=False))

# ─── Plots (same units) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for col, lab, c in [("gen_whole_pred_dc50_nM", "SynGlue whole", "#1f77b4"),
                    ("tack_pred_dc50_nM", "TACK", "#2ca02c"),
                    ("gen_pred_dc50_nM", "SynGlue 3-token", "#ff7f0e")]:
    m = common[common["dc50_median"].notna() & ~common["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(m) >= 5:
        ax.scatter(np.log10(m["dc50_median"]), np.log10(m[col]), s=14, alpha=0.55, label=f"{lab} (n={len(m)})", color=c)
lim = [0, 5]
ax.plot(lim, lim, "k--", lw=1)
ax.plot(lim, np.array(lim) - 1, "k:", lw=0.7); ax.plot(lim, np.array(lim) + 1, "k:", lw=0.7)
ax.set_xlabel("Reported DC50 (log10 nM)"); ax.set_ylabel("Predicted DC50 (log10 nM)")
ax.set_title("DC50 — same units, common leakage-free set"); ax.legend(fontsize=8)
ax = axes[1]
for col, lab, c in [("gen_whole_pred_dmax_pct", "SynGlue whole", "#1f77b4"),
                    ("tack_pred_dmax_pct", "TACK", "#2ca02c")]:
    m = common[common["dmax_max"].notna()].dropna(subset=[col])
    if len(m) >= 5:
        ax.scatter(m["dmax_max"], m[col], s=14, alpha=0.55, label=f"{lab} (n={len(m)})", color=c)
lim = [0, 100]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("Reported Dmax (%)"); ax.set_ylabel("Predicted Dmax (%)")
ax.set_title("Dmax — same units"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "direct_plot_dc50.png"), dpi=300)
plt.savefig(os.path.join(OUT, "direct_plot_dc50.svg"))
print(f"\nSaved: {OUT}/")
