#!/usr/bin/env python3
"""
FINAL external tool validation v2 (R1.3) — PER-TOOL LEAKAGE CHECK.
Principles (per user):
 1. Only real tools compared: SynGlue (one model), PDP, PROTAC-STAN.
    (Morgan-RF / Tanimoto-NN internal baselines REMOVED from the tool table;
     median kept only as a labeled reference.)
 2. Each tool is evaluated on compounds NOVEL TO THAT TOOL's training data:
      - SynGlue train: server_53 Attention/data (exact match removal, 878)
      - PROTAC-STAN train: data/PROTAC-fine train+test SMILES (exact match removal)
      - PDP: training SMILES not distributed (PROTAC-DB-derived) -> exact overlap
        not computable; evaluated on the same novelty subsets, bound documented.
 3. Task-matched units: DC50 ranking (Spearman rho, all scores) and active/inactive
    AUC (all scores); PDP/STAN output P(active) (classifier), SynGlue outputs DC50.
Outputs: outputs/per_tool_leakage.csv, tool_metrics_novel.csv,
         tool_metrics_common.csv (novel to ALL tools), plot, report.
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

# ─── 1. External master ──────────────────────────────
ext = pd.read_csv(EXT)
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
}))
master = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
master = master.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
print(f"external unique: {len(master)}")

# ─── 2. Per-tool training overlap ────────────────────
# SynGlue training set (canonical)
tr = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False,
                 usecols=lambda c: not c.startswith("Grover_"))
syn_train = set()
for s in tr["PROTAC_SMILES"].dropna().astype(str).unique():
    c = canon(s)
    if c: syn_train.add(c)
print(f"SynGlue train compounds: {len(syn_train)}")

# PROTAC-STAN training set (PROTAC-fine train+test)
stan_smiles = []
for f in ["train_compound_smiles.csv", "test_compound_smiles.csv"]:
    d = pd.read_csv(os.path.join(STAN_TRAIN, f), header=None, names=["smiles"])
    stan_smiles.extend(d["smiles"].dropna().astype(str).tolist())
stan_train = set()
for s in stan_smiles:
    c = canon(s)
    if c: stan_train.add(c)
print(f"PROTAC-STAN train+test compounds: {len(stan_train)}")

master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)
master["exact_in_any"] = master["exact_in_synglue"] | master["exact_in_stan"]

# PDP: training SMILES not distributed; PROTAC-DB-derived. External compounds
# sourced from PROTAC-DB family (Database col) are the overlap bound; also any
# compound in STAN's PROTAC-fine (PDP and STAN both PROTAC-DB-derived).
master["pdp_overlap_bound"] = master["Database"].astype(str).str.contains("PROTAC-DB") | master["exact_in_stan"]

print("\n=== PER-TOOL LEAKAGE ===")
print(f"exact in SynGlue train: {int(master['exact_in_synglue'].sum())} ({100*master['exact_in_synglue'].mean():.1f}%)")
print(f"exact in PROTAC-STAN train: {int(master['exact_in_stan'].sum())} ({100*master['exact_in_stan'].mean():.1f}%)")
print(f"exact in any (SynGlue ∪ STAN): {int(master['exact_in_any'].sum())} ({100*master['exact_in_any'].mean():.1f}%)")
print(f"PDP overlap bound (PROTAC-DB-sourced ∪ STAN-train): {int(master['pdp_overlap_bound'].sum())}")

leak = pd.DataFrame({
    "tool": ["SynGlue", "PROTAC-STAN", "PDP"],
    "training_data": ["PROTAC-DB-era (server_53, 5,200 cmpds)", "PROTAC-fine (PROTAC-DB 2.0, 1,409 cmpds)",
                      "PROTAC-DB-derived (Ribes 2024; SMILES not distributed)"],
    "exact_matches_in_external": [int(master["exact_in_synglue"].sum()), int(master["exact_in_stan"].sum()),
                                  "n/a (bound: %d)" % int(master["pdp_overlap_bound"].sum())],
    "pct_of_external": [round(100 * master["exact_in_synglue"].mean(), 1),
                        round(100 * master["exact_in_stan"].mean(), 1),
                        round(100 * master["pdp_overlap_bound"].mean(), 1)],
})
leak.to_csv(os.path.join(OUT, "per_tool_leakage.csv"), index=False)
print(leak.to_string(index=False))

# ─── 3. Model predictions ────────────────────────────
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]],
                      on="canonical_SMILES", how="left")
master = master.merge(pdp_full[["SMILES", "pdp_active_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]],
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

# ─── 4. Metrics on per-tool novel subsets ────────────
def bci(x, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

def metrics(sub, col, log):
    m = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(m) < 5: return dict(n=0)
    y = np.log10(m["dc50_median"].values.astype(float))
    p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
    rho = stats.spearmanr(y, p)[0]; ci = bci(y, p)
    k = max(1, int(0.1 * len(m)))
    top = m.iloc[np.argsort(p)[:k]] if log else m.iloc[np.argsort(-p)[:k]]
    base = float((m["dc50_median"] <= 100).mean())
    out = dict(n=len(m), rho=rho, ci=f"[{ci[0]:.3f},{ci[1]:.3f}]", R2=r2_score(y, p),
               MAE=float(mean_absolute_error(y, p)), top10_enrich=float((top["dc50_median"] <= 100).mean()) / max(base, 1e-6))
    for thr in [100, 1000]:
        lb = (m["dc50_median"] <= thr).astype(int)
        out[f"auc_{thr}"] = roc_auc_score(lb, p) if min((lb == 0).sum(), (lb == 1).sum()) >= 5 else np.nan
    return out

TOOLS = [
    ("SynGlue (whole-PROTAC)", "gen_whole_pred_dc50_nM", True, "exact_in_synglue"),
    ("PDP", "pdp_active_proba", False, "pdp_overlap_bound"),
    ("PROTAC-STAN", "protacstan_active_proba", False, "exact_in_stan"),
]

rows, rows_common = [], []
common = master[~master["exact_in_any"]].copy()
for name, col, log, leak_col in TOOLS:
    nov = master[~master[leak_col]].copy()
    m = metrics(nov, col, log)
    rows.append(dict(model=name, novel_set=f"external − {leak_col} (n={len(nov)})", **m))
    mc = metrics(common, col, log)
    rows_common.append(dict(model=name, **mc))

md = pd.DataFrame(rows)
md.to_csv(os.path.join(OUT, "tool_metrics_novel.csv"), index=False)
mdc = pd.DataFrame(rows_common)
mdc.to_csv(os.path.join(OUT, "tool_metrics_common.csv"), index=False)
print("\n=== METRICS on per-tool NOVEL subsets ===\n" + md.to_string(index=False))
print("\n=== METRICS on common set (novel to ALL tools) ===\n" + mdc.to_string(index=False))

# ─── 5. Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, df_, title in [(axes[0], md, "Per-tool novel subsets — ρ"),
                       (axes[1], md, "Per-tool novel subsets — AUC(1µM)"),
                       (axes[2], mdc, "Common set (novel to all) — ρ")]:
    if title.endswith("ρ"):
        ax.bar(df_["model"], df_["rho"], color=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.85)
        for i, r in df_.iterrows():
            lo, hi = [float(x.strip("[]")) for x in r["ci"].split(",")]
            ax.plot([i, i], [lo, hi], color="k", lw=1.2)
        ax.axhline(0, color="k", lw=0.8)
    else:
        ax.bar(df_["model"], df_["auc_1000"], color=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.85)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 1)
    ax.set_xticklabels(df_["model"], rotation=12, fontsize=8)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tool_comparison_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "tool_comparison_plot.svg"))
print(f"\nSaved: {OUT}/")
