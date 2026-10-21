#!/usr/bin/env python3
"""
Rank analysis — Spearman rho + Kendall tau (with bootstrap 95% CIs) for all models.
Orientation (a-priori): regressors score = -log10(pred DC50 nM); classifiers score = P(active);
ground truth = -log10(known DC50 nM). Higher = more potent. Positive rho/tau = useful model.
Leakage-controlled per tool (no-PROTAC-DB pool; TACK excluded per user).
Outputs: outputs/rank_analysis.csv, rank_plot.png/.svg
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
STAN_TRAIN = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN/data/PROTAC-fine"

def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

# master (no PROTAC-DB) + predictions
ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
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

sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
master = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM"]].rename(columns={"canonical_smiles": "k1"}),
                      left_on="canonical_SMILES", right_on="k1", how="left").drop(columns=["k1"], errors="ignore")
pdc = pd.read_csv(os.path.join(TIR, "protacdc50_predictions.csv"))
master = master.merge(pdc[["canonical_smiles", "protacdc50_dc50_nM"]].rename(columns={"canonical_smiles": "k2"}),
                      left_on="canonical_SMILES", right_on="k2", how="left").drop(columns=["k2"], errors="ignore")
pdp = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
master = master.merge(pdp[["SMILES", "pdp_active_proba", "pdp_xgb_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
stan = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
master = master.merge(stan[["canonical_smiles", "protacstan_active_proba"]].rename(columns={"canonical_smiles": "k3"}),
                      left_on="canonical_SMILES", right_on="k3", how="left").drop(columns=["k3"], errors="ignore")
morgan = pd.read_csv(os.path.join(TIR, "morgan_rf_baseline.csv"))
master = master.merge(morgan[["canonical_SMILES", "morgan_rf_pred_logdc50"]], on="canonical_SMILES", how="left")
fp = pd.read_csv(os.path.join(REV, "rebuttal_analyses", "05_external_benchmark", "outputs", "fullset_predictions.csv"))
master = master.merge(fp[["canonical_SMILES", "nn_pred_logdc50"]], on="canonical_SMILES", how="left")

master["potency_gt"] = -np.log10(master["dc50_median"].values.astype(float))

MODELS = {
    "SynGlue 3-token": ("gen_pred_dc50_nM", "dc50", "exact_in_synglue"),
    "PROTAC-DC50": ("protacdc50_dc50_nM", "dc50", None),
    "Morgan-RF": ("morgan_rf_pred_logdc50", "logdc50", None),
    "Tanimoto-NN": ("nn_pred_logdc50", "logdc50", None),
    "PDP": ("pdp_active_proba", "proba", None),
    "PDP-XGB": ("pdp_xgb_proba", "proba", None),
    "PROTAC-STAN": ("protacstan_active_proba", "proba", "exact_in_stan"),
}

def bci(x, y, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(fn(x[ix], y[ix]))
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

rows = []
for name, (col, kind, flag) in MODELS.items():
    sub = master if flag is None else master[~master[flag].fillna(False)]
    u = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(u) < 5:
        rows.append(dict(model=name, n=0)); continue
    if kind == "dc50":
        score = -np.log10(u[col].values.astype(float))
    elif kind == "logdc50":
        score = -u[col].values.astype(float)
    else:
        score = u[col].values.astype(float)
    y = u["potency_gt"].values
    rho = stats.spearmanr(y, score)[0]
    rho_ci = bci(y, score, lambda a, b: stats.spearmanr(a, b)[0])
    tau = stats.kendalltau(y, score)[0]
    tau_ci = bci(y, score, lambda a, b: stats.kendalltau(a, b)[0])
    rows.append(dict(model=name, n=len(u), spearman_rho=rho,
                     rho_ci_low=rho_ci[0], rho_ci_high=rho_ci[1],
                     kendall_tau=tau, tau_ci_low=tau_ci[0], tau_ci_high=tau_ci[1],
                     output="DC50 regressor" if kind != "proba" else "P(active) classifier"))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "rank_analysis.csv"), index=False)
print(df.to_string(index=False))

# plot
fig, ax = plt.subplots(figsize=(9.5, 5.5))
d = df[df["n"] >= 5]
x = np.arange(len(d))
w = 0.35
ax.bar(x - w / 2, d["spearman_rho"], w, yerr=[d["spearman_rho"] - d["rho_ci_low"], d["rho_ci_high"] - d["spearman_rho"]],
       capsize=4, label="Spearman ρ", color="#1f77b4", alpha=0.9)
ax.bar(x + w / 2, d["kendall_tau"], w, yerr=[d["kendall_tau"] - d["tau_ci_low"], d["tau_ci_high"] - d["kendall_tau"]],
       capsize=4, label="Kendall τ", color="#ff7f0e", alpha=0.9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(d["model"], rotation=20, fontsize=8)
ax.set_ylabel("Rank correlation vs observed potency (−log10 DC50)")
ax.set_title("Rank analysis — all models (oriented: higher = more potent)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rank_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "rank_plot.svg"))
print(f"\nSaved: {OUT}/rank_analysis.csv + rank_plot.png/.svg")
