#!/usr/bin/env python3
"""
Analysis 5 v2 (R1.3) — SYSTEMATIC EXTERNAL BENCHMARK on the FULL external pool.
Common test set: final_dataset_E3_known.csv (3,518 unique PROTACs; 2,961 with DC50).
Models (each on its covered subset; coverage reported):
  SynGlue frozen / generalized  : 151 fully-decomposed leakage-free compounds
  PDP                            : full set (E3-vocab supported, cell-line fallback)
  PROTAC-STAN                    : 3,208 (ESM-covered POI+E3)
  Baselines: median DC50; Tanimoto-NN (nearest training compound DC50)
Metrics per model: N, Spearman rho [95% CI], R2, MAE(log10), med|Δlog10|,
  AUC(≤100nM & ≤1µM, class-gated), top-10% enrichment, Dmax rho where applicable.
Intersection (151): all models on identical compounds.
Outputs: outputs/fullset_metrics.csv, fullset_predictions.csv, fullset_plot.png/.svg,
         intersection_metrics.csv, coverage_table.csv
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
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"

# ─── 1. Full external master table ───────────────────
ext = pd.read_csv(EXT)
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
}))
master = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
master = master.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

# ─── 2. Model predictions ────────────────────────────
# SynGlue (151 decomposed set) — mark which of the 151
sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
master["in_synglue151"] = master["canonical_SMILES"].isin(sc151["canonical_smiles"])
frozen = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_predictions.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
fz_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_frozen.csv"))
morgan = pd.read_csv(os.path.join(TIR, "morgan_rf_baseline.csv"))

master = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM", "gen_pred_dmax_pct"]],
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left", suffixes=("", "_s"))
master = master.merge(frozen[["canonical_smiles", "pred_dc50_nM", "pred_dmax_pct"]],
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left", suffixes=("", "_f"))
master = master.merge(pdp_full[["SMILES", "pdp_active_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]], on="canonical_smiles", how="left")
master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]],
                      left_on="canonical_SMILES", right_on="canonical_SMILES", how="left", suffixes=("", "_gw"))
master = master.merge(fz_whole[["canonical_SMILES", "frozen_whole_pred_dc50_nM", "frozen_whole_pred_dmax_pct"]],
                      on="canonical_SMILES", how="left")
master = master.merge(morgan[["canonical_SMILES", "morgan_rf_pred_logdc50"]], on="canonical_SMILES", how="left")

# ─── 3. Baselines ────────────────────────────────────
# Tanimoto-NN: nearest training compound (with DC50) by Morgan-2 similarity
print("Computing Tanimoto-NN baseline (train ~5.2k compounds)...")
sys.path.insert(0, REV)
from reproduce_original_model import load_data
Xtr, y50tr, ymtr, metatr = load_data()
mask_tr = np.isfinite(y50tr)
tr_smiles = metatr["SMILES"].values[mask_tr]
tr_dc50 = y50tr[mask_tr]
train_fps, train_vals = [], []
for s, v in zip(tr_smiles, tr_dc50):
    m = Chem.MolFromSmiles(str(s))
    if m is not None:
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        train_vals.append(v)
print(f"  train NN pool: {len(train_fps)}")

nn_pred = np.full(len(master), np.nan)
for i, smi in enumerate(master["canonical_SMILES"]):
    m = Chem.MolFromSmiles(smi)
    if m is None: continue
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    j = int(np.argmax(sims))
    nn_pred[i] = train_vals[j]
master["nn_pred_logdc50"] = nn_pred
master["median_pred_logdc50"] = np.log10(np.nanmedian(10 ** y50tr[np.isfinite(y50tr)]))

master.to_csv(os.path.join(OUT, "fullset_predictions.csv"), index=False)
print(f"master: {len(master)} | with DC50: {master['dc50_median'].notna().sum()}")

# ─── 4. Metrics ──────────────────────────────────────
def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

MODELS = [
    ("SynGlue frozen (3-token)", "pred_dc50_nM", True, "pred_dmax_pct", "in_synglue151"),
    ("SynGlue generalized (3-token)", "gen_pred_dc50_nM", True, "gen_pred_dmax_pct", "in_synglue151"),
    ("SynGlue frozen (whole-PROTAC)", "frozen_whole_pred_dc50_nM", True, "frozen_whole_pred_dmax_pct", None),
    ("SynGlue generalized (whole-PROTAC)", "gen_whole_pred_dc50_nM", True, "gen_whole_pred_dmax_pct", None),
    ("PDP (P active)", "pdp_active_proba", False, None, None),
    ("PROTAC-STAN (P active)", "protacstan_active_proba", False, None, None),
    ("Morgan-RF baseline", "morgan_rf_pred_logdc50", False, None, None),
    ("Tanimoto-NN baseline", "nn_pred_logdc50", False, None, None),
    ("Median baseline", "median_pred_logdc50", False, None, None),
]

def metrics(sub, col, log, dcol=None):
    m = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(m) < 5: return dict(n=0)
    y = np.log10(m["dc50_median"].values.astype(float))
    p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    k = max(1, int(0.1 * len(m)))
    top = m.iloc[np.argsort(p)[:k]] if log else m.iloc[np.argsort(-p)[:k]]
    base = float((m["dc50_median"] <= 100).mean())
    enrich = float((top["dc50_median"] <= 100).mean()) / max(base, 1e-6)
    out = dict(n=len(m), rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]", R2=r2_score(y, p),
               MAE=float(mean_absolute_error(y, p)),
               med_abs_log=float(np.median(np.abs(p - y))),
               top10_enrich=enrich)
    for thr in [100, 1000]:
        lb = (m["dc50_median"] <= thr).astype(int)
        out[f"auc_{thr}"] = roc_auc_score(lb, p) if min((lb == 0).sum(), (lb == 1).sum()) >= 5 else np.nan
    if dcol is not None:
        dm = sub[sub["dmax_max"].notna()].dropna(subset=[dcol])
        out["dmax_rho"] = stats.spearmanr(dm["dmax_max"], dm[dcol])[0] if len(dm) >= 5 else np.nan
    return out

rows = []
for name, col, log, dcol, flag in MODELS:
    if flag:
        sub = master[master[flag]]
    else:
        sub = master
    m = metrics(sub, col, log, dcol)
    rows.append(dict(model=name, coverage=int(sub["canonical_SMILES"].nunique()), **m))
md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "fullset_metrics.csv"), index=False)
print("\n" + md.to_string(index=False))

# ─── 5. Intersection (151, all models comparable) ────
inter = master[master["in_synglue151"]].copy()
rows_i = []
for name, col, log, dcol, _ in MODELS:
    m = metrics(inter, col, log, dcol)
    rows_i.append(dict(model=name, n=m.get("n", 0), rho=m.get("rho", np.nan),
                       ci=m.get("ci", ""), R2=m.get("R2", np.nan), MAE=m.get("MAE", np.nan),
                       top10_enrich=m.get("top10_enrich", np.nan)))
idi = pd.DataFrame(rows_i)
idi.to_csv(os.path.join(OUT, "intersection_metrics.csv"), index=False)

# coverage table
cov = pd.DataFrame([dict(tool=r["model"], n_compounds_covered=r["coverage"], n_with_dc50=sum(
    master[(master[flag] if flag else master.index >= 0)]["dc50_median"].notna()) if False else np.nan)
                    for r in rows])
cov.to_csv(os.path.join(OUT, "coverage_table.csv"), index=False)
cov = md[["model", "coverage", "n"]].rename(columns={"coverage": "covered_compounds", "n": "scored_with_dc50"})
cov.to_csv(os.path.join(OUT, "coverage_table.csv"), index=False)
print("\nCoverage:\n" + cov.to_string(index=False))

# ─── 6. Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
d = md[md["n"] >= 5]
ax.bar(d["model"], d["rho"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(d)], alpha=0.85)
for i, r in d.iterrows():
    lo, hi = [float(x.strip("[]")) for x in r["ci"].split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(d))); ax.set_xticklabels(d["model"], rotation=15, fontsize=8)
ax.set_ylabel("Spearman ρ vs log10 DC50"); ax.set_title("Full external set — DC50 ranking")
ax = axes[1]
dd = d[d["auc_100"].notna()]
ax.bar(dd["model"], dd["auc_100"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(dd)], alpha=0.85)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticklabels(dd["model"], rotation=15, fontsize=8); ax.set_ylim(0, 1)
ax.set_ylabel("AUC (active = DC50 ≤ 100 nM)"); ax.set_title("Active/inactive discrimination")
ax = axes[2]
ax.bar(d["model"], d["top10_enrich"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(d)], alpha=0.85)
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticklabels(d["model"], rotation=15, fontsize=8)
ax.set_ylabel("Top-10% enrichment (active ≤100 nM)"); ax.set_title("Prioritization")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fullset_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "fullset_plot.svg"))
print(f"\nSaved: {OUT}/")
