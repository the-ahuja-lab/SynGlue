#!/usr/bin/env python3
"""
11 — DIRECTIONALITY-FIXED cross-tool benchmark (per reviewer guidance).
A-priori orientation (established BEFORE looking at results):
  - Models predicting DC50 (SynGlue, TACK-XGB, TACK-MLP, PROTAC-DC50, Morgan-RF, Tanimoto-NN):
    potency_score = -log10(pred_DC50_nM)   (higher = more potent)
  - Classifiers (PDP, PDP-XGB, PROTAC-STAN): score = P(active) (higher = more potent)
  - Ground truth potency = -log10(known DC50 nM); y_active(thr) = DC50 <= thr.
Metrics:
  A) Spearman rho (regressors only, + expected for useful models)
  B) AUROC + AUPRC at DC50<=10 / <=100 / <=1000 nM (ALL models, oriented)
  C) EF10% + Precision@10% + Recall@10% at the same thresholds (ALL models)
Panel figure: A) SynGlue regression generalization R2 (splits) | B) AUROC (<=1uM & <=100nM)
             C) EF10% (<=100nM)
Outputs: outputs/oriented_metrics.csv, syn_glue_splits.csv, panel_figure.png/.svg, REPORT.md
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, average_precision_score)

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

# ─── 1. Master (no PROTAC-DB) + predictions ──────────
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

tr = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
syn_train = {c for c in map(canon, tr["PROTAC_SMILES"].dropna().astype(str).unique()) if c}
stan_smiles = []
for f in ["train_compound_smiles.csv", "test_compound_smiles.csv"]:
    stan_smiles += pd.read_csv(os.path.join(STAN_TRAIN, f), skiprows=1, header=None, names=["s"])["s"].dropna().astype(str).tolist()
stan_train = {c for c in map(canon, stan_smiles) if c}
master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)

tack = pd.read_csv(os.path.join(TIR, "tack_predictions.csv"))
tack_extra = pd.read_csv(os.path.join(TIR, "tack_extra_predictions.csv"))
pdc50 = pd.read_csv(os.path.join(TIR, "protacdc50_predictions.csv"))
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
morgan = pd.read_csv(os.path.join(TIR, "morgan_rf_baseline.csv"))
full_prev = pd.read_csv(os.path.join(REV, "rebuttal_analyses", "05_external_benchmark", "outputs", "fullset_predictions.csv"))

master = master.merge(tack[["canonical_SMILES", "tack_pred_dc50_nM", "tack_pred_dmax_pct", "held_out"]], on="canonical_SMILES", how="left")
master = master.merge(tack_extra[["canonical_SMILES", "tack_mlp_dc50_nM"]], on="canonical_SMILES", how="left")
pc = pdc50[["canonical_smiles", "protacdc50_dc50_nM"]].rename(columns={"canonical_smiles": "k1"})
master = master.merge(pc, left_on="canonical_SMILES", right_on="k1", how="left").drop(columns=["k1"], errors="ignore")
master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM"]], on="canonical_SMILES", how="left")
s3 = sc151[["canonical_smiles", "gen_pred_dc50_nM"]].rename(columns={"canonical_smiles": "k2"})
master = master.merge(s3, left_on="canonical_SMILES", right_on="k2", how="left").drop(columns=["k2"], errors="ignore")
master = master.merge(pdp_full[["SMILES", "pdp_active_proba", "pdp_xgb_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
sf = stan_full[["canonical_smiles", "protacstan_active_proba"]].rename(columns={"canonical_smiles": "k3"})
master = master.merge(sf, left_on="canonical_SMILES", right_on="k3", how="left").drop(columns=["k3"], errors="ignore")
master = master.merge(morgan[["canonical_SMILES", "morgan_rf_pred_logdc50"]], on="canonical_SMILES", how="left")
fp = full_prev[["canonical_SMILES", "nn_pred_logdc50"]]
master = master.merge(fp, on="canonical_SMILES", how="left")

# ─── 2. A-priori orientation (higher = more potent) ──
def orient_dc50(s):
    return -np.log10(np.asarray(s, dtype=float))  # -log10(DC50 nM)

master["potency_gt"] = orient_dc50(master["dc50_median"])
MODELS = {
    "SynGlue 3-token":            ("gen_pred_dc50_nM", "dc50", "exact_in_synglue"),
    "SynGlue whole-PROTAC":       ("gen_whole_pred_dc50_nM", "dc50", "exact_in_synglue"),
    "TACK-XGB DC50":              ("tack_pred_dc50_nM", "dc50", "held_out_novel"),
    "TACK-MLP DC50":              ("tack_mlp_dc50_nM", "dc50", "held_out_novel"),
    "PROTAC-DC50":                ("protacdc50_dc50_nM", "dc50", "exact_in_synglue"),
    "Morgan-RF":                  ("morgan_rf_pred_logdc50", "logdc50", None),
    "Tanimoto-NN":                ("nn_pred_logdc50", "logdc50", None),
    "PDP":                        ("pdp_active_proba", "proba", None),
    "PDP-XGB":                    ("pdp_xgb_proba", "proba", None),
    "PROTAC-STAN":                ("protacstan_active_proba", "proba", "exact_in_stan"),
}
master["held_out_novel"] = master["held_out"].fillna(False) & ~master["exact_in_synglue"]

# leakage filter per model
def subset(model, key):
    flag = MODELS[key][2]
    if flag is None:
        return master
    return master[~master[flag].fillna(False)]

# ─── 3. Metrics ──────────────────────────────────────
THRESH = {10: "<=10nM", 100: "<=100nM", 1000: "<=1uM"}
rows = []
for key, (col, kind, _) in MODELS.items():
    sub = subset(master, key)
    u = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(u) < 5:
        rows.append(dict(model=key, n=0)); continue
    if kind == "dc50":
        score = orient_dc50(u[col].values)
    elif kind == "logdc50":
        score = -np.asarray(u[col].values, dtype=float)  # -log10 DC50
    else:
        score = np.asarray(u[col].values, dtype=float)   # P(active), higher = active
    ygt = u["potency_gt"].values
    r = dict(model=key, n=len(u),
             spearman_rho=stats.spearmanr(ygt, score)[0])
    for thr, tname in THRESH.items():
        y = (u["dc50_median"].values <= thr).astype(int)
        if min((y == 0).sum(), (y == 1).sum()) >= 5:
            r[f"auroc_{tname}"] = roc_auc_score(y, score)
            r[f"auprc_{tname}"] = average_precision_score(y, score)
        else:
            r[f"auroc_{tname}"] = np.nan
            r[f"auprc_{tname}"] = np.nan
        # EF10% + Precision/Recall@10%
        k = max(1, int(0.1 * len(u)))
        order = np.argsort(-score)[:k]
        hit_all = y.mean()
        hit_top = y[order].mean()
        r[f"ef10_{tname}"] = hit_top / max(hit_all, 1e-9)
        r[f"prec10_{tname}"] = hit_top
        r[f"rec10_{tname}"] = y[order].sum() / max(y.sum(), 1)
    rows.append(r)

md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "oriented_metrics.csv"), index=False)
print(md.to_string(index=False))

# ─── 4. SynGlue regression generalization (splits) ───
wp2 = pd.read_csv(os.path.join(REV, "benchmarking", "wp2_results", "wp2_twostage_results.csv"))
agg = wp2.groupby("split").agg(dc50_R2=("dc50_R2", "mean"), dmax_R2=("dmax_R2", "mean")).reset_index()
rows_s = []
for _, r in agg.iterrows():
    rows_s.append(dict(split=r["split"], DC50_R2=r["dc50_R2"], Dmax_R2=r["dmax_R2"]))
rows_s.append(dict(split="External (whole-PROTAC, n=1809)", DC50_R2=np.nan, Dmax_R2=np.nan))
sd = pd.DataFrame(rows_s)
sd.to_csv(os.path.join(OUT, "syn_glue_splits.csv"), index=False)

# ─── 5. Three-panel figure ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
splits = ["Random 80/20", "Scaffold-held-out", "E3-held-out (VHL)", "Target-held-out"]
def r2_for(s):
    row = sd[sd["split"] == s]
    return float(row["DC50_R2"].iloc[0]) if len(row) else np.nan
r2s = [r2_for(s) for s in splits]
ax.bar(splits, r2s, color="#1f77b4", alpha=0.85)
for x, v in zip(range(4), r2s):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels([s.split("-")[0] for s in splits], rotation=12, fontsize=8)
ax.set_ylabel("DC50 R² (two-stage TF+RF)")
ax.set_title("A. SynGlue regression generalization")

ax = axes[1]
d = md[md["n"] >= 5]
for thr, lab, c in [(1000, "≤1 µM", "#2ca02c"), (100, "≤100 nM", "#d62728")]:
    col = f"auroc_{'<=1uM' if thr==1000 else '<=100nM'}"
    ax.plot(d["model"], d[col], marker="o", label=lab, color=c)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticklabels(d["model"], rotation=20, fontsize=7)
ax.set_ylabel("AUROC (oriented: higher = more potent)")
ax.set_title("B. Active/inactive discrimination (all tools)")
ax.legend(fontsize=8)

ax = axes[2]
col = "ef10_<=100nM"
d2 = d[d[col].notna()]
ax.bar(d2["model"], d2[col], color="#9467bd", alpha=0.85)
ax.axhline(1, color="k", ls="--", lw=0.8, label="random = 1")
ax.set_xticklabels(d2["model"], rotation=20, fontsize=7)
ax.set_ylabel("EF10% (DC50 ≤ 100 nM)")
ax.set_title("C. Candidate prioritization (top 10%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "panel_figure.png"), dpi=300)
plt.savefig(os.path.join(OUT, "panel_figure.svg"))
print(f"\nSaved: {OUT}/")
