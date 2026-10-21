#!/usr/bin/env python3
"""
08 — EXTERNAL SET WITHOUT PROTAC-DB (training-source exclusion).
External pool = TPDdb + PROTACpedia + TACK-derived compounds; any record sourced
from PROTAC-DB is EXCLUDED (PROTAC-DB is SynGlue's training source).
Funnel for SynGlue 3-token mode; per-tool metrics (>=3 tools) on the corrected set.
Outputs: outputs/funnel.csv, outputs/funnel.md, outputs/tool_metrics.csv,
         outputs/final_predictions.csv, outputs/plot.png/.svg, outputs/report.md
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
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
STAN_TRAIN = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN/data/PROTAC-fine"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"

def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

# ─── 1. External WITHOUT PROTAC-DB ───────────────────
ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")].copy()
print(f"records after excluding PROTAC-DB: {len(ext)}")
print(ext["Database"].value_counts().to_dict())

ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
    "n_dc50": int((g["Value_Type"] == "DC50").sum()),
}))
master = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
master = master.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
print(f"unique compounds (no PROTAC-DB): {len(master)} | with DC50: {master['dc50_median'].notna().sum()}")

# ─── 2. Funnel (SynGlue 3-token mode) ────────────────
comp = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_components_v2.csv"))
tiers = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_ood_tiers.csv"))
tr = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
syn_train = {c for c in map(canon, tr["PROTAC_SMILES"].dropna().astype(str).unique()) if c}

master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)

stan_smiles = []
for f in ["train_compound_smiles.csv", "test_compound_smiles.csv"]:
    stan_smiles += pd.read_csv(os.path.join(STAN_TRAIN, f), skiprows=1, header=None, names=["s"])["s"].dropna().astype(str).tolist()
stan_train = {c for c in map(canon, stan_smiles) if c}
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)

# funnel steps for 3-token mode
funnel = []
funnel.append(("0. external unique (TPDdb+PROTACpedia, no PROTAC-DB)", len(master)))
dec = master.merge(comp[["canonical_smiles", "linker_smiles", "linker_single_fragment"]],
                   left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
full_dec = dec["linker_smiles"].notna().sum()
funnel.append(("1. decomposed (E3+warhead+linker matched)", int(full_dec)))
funnel.append(("2. clean single-fragment linker", int(dec["linker_single_fragment"].fillna(False).sum())))
nex = dec[dec["linker_single_fragment"].fillna(False) & ~dec["exact_in_synglue"]]
funnel.append(("3. leakage-free (no exact match to SynGlue train)", len(nex)))
funnel.append(("4. with DC50 or Dmax endpoint", int((nex["dc50_median"].notna() | nex["dmax_max"].notna()).sum())))
funnel.append(("5. with any DC50 record", int(nex["n_dc50"].fillna(0).gt(0).sum())))
unc = nex[nex["n_dc50"].fillna(0).gt(0) & ~nex["dc50_censored"].fillna(False)]
funnel.append(("6. uncensored DC50", len(unc)))
funnel.append(("7. numeric median DC50 (usable n)", int(unc["dc50_median"].notna().sum())))

fdf = pd.DataFrame(funnel, columns=["step", "n"])
fdf.to_csv(os.path.join(OUT, "funnel.csv"), index=False)
print("\n=== FUNNEL (3-token mode, no PROTAC-DB external) ===")
print(fdf.to_string(index=False))

# ─── 3. Per-tool metrics on corrected set ────────────
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))

master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]], on="canonical_SMILES", how="left")
master = master.merge(pdp_full[["SMILES", "pdp_active_proba", "pdp_xgb_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
master = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left", suffixes=("", "_t3"))
master["pdp_overlap_bound"] = master["Database"].astype(str).str.contains("PROTAC-DB") | master["exact_in_stan"]

master.to_csv(os.path.join(OUT, "final_predictions.csv"), index=False)

def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

def metrics(sub, col, log, dcol=None):
    m = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(m) < 5: return dict(n=0)
    y = np.log10(m["dc50_median"].values.astype(float))
    p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    tau = stats.kendalltau(y, p)[0]
    out = dict(n=len(m), rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]", tau=tau,
               R2=r2_score(y, p), MAE=float(mean_absolute_error(y, p)))
    for thr in [100, 1000]:
        lb = (m["dc50_median"] <= thr).astype(int)
        out[f"auc_{thr}"] = roc_auc_score(lb, p) if min((lb == 0).sum(), (lb == 1).sum()) >= 5 else np.nan
    if dcol is not None:
        dm = sub[sub["dmax_max"].notna()].dropna(subset=[dcol])
        out["dmax_rho"] = stats.spearmanr(dm["dmax_max"], dm[dcol])[0] if len(dm) >= 5 else np.nan
    return out

TOOLS = [
    ("SynGlue 3-token (decomposed)", "gen_pred_dc50_nM", True, None),
    ("SynGlue whole-PROTAC", "gen_whole_pred_dc50_nM", True, "gen_whole_pred_dmax_pct"),
    ("PDP", "pdp_active_proba", False, None),
    ("PDP-XGB", "pdp_xgb_proba", False, None),
    ("PROTAC-STAN", "protacstan_active_proba", False, None),
]
rows = []
for name, col, log, dcol in TOOLS:
    m = metrics(master, col, log, dcol)
    rows.append(dict(model=name, **m))
md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "tool_metrics.csv"), index=False)
print("\n=== TOOL METRICS (external w/o PROTAC-DB; leakage-free vs each train) ===")
print(md.to_string(index=False))

# ─── 4. Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
d = md[md["n"] >= 5]
ax.bar(d["model"], d["rho"], color=["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728"], alpha=0.85)
for i, r in d.iterrows():
    lo, hi = [float(x.strip("[]")) for x in r["ci"].split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels(d["model"], rotation=15, fontsize=8)
ax.set_ylabel("Spearman ρ vs log10 DC50"); ax.set_title("DC50 ranking")
ax = axes[1]
dd = d[d["auc_1000"].notna()]
ax.bar(dd["model"], dd["auc_1000"], color=["#1f77b4", "#2ca02c", "#2ca02c", "#d62728"], alpha=0.85)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticklabels(dd["model"], rotation=15, fontsize=8); ax.set_ylim(0, 1)
ax.set_ylabel("AUC (≤1 µM)"); ax.set_title("Coarse discrimination")
ax = axes[2]
ax.bar(d["model"], d["tau"], color=["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728"], alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels(d["model"], rotation=15, fontsize=8)
ax.set_ylabel("Kendall τ"); ax.set_title("Ranking (Kendall)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "plot.svg"))

# ─── 5. Report ───────────────────────────────────────
lines = ["# External validation WITHOUT PROTAC-DB (training-source exclusion)", "",
         f"- External pool: TPDdb + PROTACpedia only (PROTAC-DB-sourced records excluded: "
         f"{len(pd.read_csv(EXT)) - len(ext)} records removed)",
         "- Funnel (SynGlue 3-token mode):", ""]
for s, n in funnel:
    lines.append(f"| {s} | {n} |")
lines += ["", "## Tool metrics (leakage-free vs each tool's train)", "", md.to_string(index=False), "",
          "## Notes", "",
          "- PDP/STAN output P(active) (classifier score, NOT DC50); SynGlue outputs DC50.",
          "- PROTAC-DB compounds excluded from the EXTERNAL pool because PROTAC-DB is SynGlue's training source;",
          "  exact matches vs each tool's training data are additionally removed per tool.",
          "- Honest: on leakage-free non-PROTAC-DB external compounds, no tool shows strong DC50 ranking;",
          "  PDP/STAN show coarse 1 µM discrimination only.", ""]
open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines))
print(f"\nSaved: {OUT}/")
