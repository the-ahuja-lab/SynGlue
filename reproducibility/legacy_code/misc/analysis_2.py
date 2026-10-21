#!/usr/bin/env python3
"""
12 — Three-question revision analysis (reviewer-guided).
Q-A: Does SynGlue quantitatively generalize?   R², MAE, RMSE, Spearman/Kendall across
     Random → Scaffold → Target → E3 → External (TACK-XGB as external reference).
Q-B: Does SynGlue prioritize better?           AUROC + AUPRC + EF10% + Precision@10%
     (common task, oriented: higher = more potent; TACK-XGB primary comparator).
Q-C: Is the external test genuinely OOD?       scaffold novelty, max Tanimoto to training,
     target/E3 novelty, analog-series (Butina clusters).
Outputs: outputs/gen_quantitative.csv, prioritization_metrics.csv, ood_novelty.csv,
         panelA/B/C_*.png, three_questions_figure.png/.svg, REBUTTAL_TEXT.md
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

def scaffold_of(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None

# ═══════════ Q-A: quantitative generalization ═══════════
wp2 = pd.read_csv(os.path.join(REV, "benchmarking", "wp2_results", "wp2_twostage_results.csv"))
wpm = pd.read_csv(os.path.join(REV, "benchmarking", "wp2_results", "wp2_master_results.csv"))
rowsA = []
for s, g in wp2.groupby("split"):
    gm = wpm[wpm["split"] == s]
    rowsA.append(dict(split=s, R2=float(g["dc50_R2"].mean()),
                      RMSE=float(gm["dc50_RMSE"].mean()) if len(gm) else np.nan,
                      MAE=float(gm["dc50_MAE"].mean()) if len(gm) else np.nan,
                      rho=np.nan, tau=np.nan,
                      n_test=int(g["n_test"].mean())))

# external split metrics (SynGlue whole + 3-token; TACK-XGB reference on its held-out subset)
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
master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)
master = master.merge(pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))[["canonical_SMILES", "gen_whole_pred_dc50_nM"]], on="canonical_SMILES", how="left")
tack = pd.read_csv(os.path.join(TIR, "tack_predictions.csv"))
master = master.merge(tack[["canonical_SMILES", "tack_pred_dc50_nM", "held_out"]], on="canonical_SMILES", how="left")

def ext_metrics(col, subset):
    u = subset[subset["dc50_median"].notna() & ~subset["dc50_censored"].fillna(False)].dropna(subset=[col])
    y = np.log10(u["dc50_median"].values.astype(float))
    p = np.log10(u[col].values.astype(float))
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), rho=stats.spearmanr(y, p)[0],
                tau=stats.kendalltau(y, p)[0], n_test=len(u))

sg_novel = master[~master["exact_in_synglue"]]
tack_ho = master[master["held_out"].fillna(False) & ~master["exact_in_synglue"]]
rowsA.append(dict(split="External (SynGlue 3-token)", **{k: np.nan for k in ["R2","RMSE","MAE","rho","tau","n_test"]}))
# 3-token via tier_scoring_set
sc151 = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
m3 = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
rowsA[-1] = dict(split="External (SynGlue 3-token)", **ext_metrics("gen_pred_dc50_nM", m3[~m3["exact_in_synglue"]]))
# TACK reference removed per user request
genA = pd.DataFrame(rowsA)
genA.to_csv(os.path.join(OUT, "gen_quantitative.csv"), index=False)

# ═══════════ Q-B: prioritization (oriented) ═══════════
master = master.merge(pd.read_csv(os.path.join(TIR, "tack_extra_predictions.csv"))[["canonical_SMILES", "tack_mlp_dc50_nM"]], on="canonical_SMILES", how="left")
master = master.merge(sc151[["canonical_smiles", "gen_pred_dc50_nM"]].rename(columns={"canonical_smiles": "k3"}), left_on="canonical_SMILES", right_on="k3", how="left").drop(columns=["k3"], errors="ignore")
pdc = pd.read_csv(os.path.join(TIR, "protacdc50_predictions.csv"))
master = master.merge(pdc[["canonical_smiles", "protacdc50_dc50_nM"]].rename(columns={"canonical_smiles": "k"}), left_on="canonical_SMILES", right_on="k", how="left").drop(columns=["k"], errors="ignore")
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
master = master.merge(pdp_full[["SMILES", "pdp_active_proba", "pdp_xgb_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]].rename(columns={"canonical_smiles": "k2"}), left_on="canonical_SMILES", right_on="k2", how="left").drop(columns=["k2"], errors="ignore")
morgan = pd.read_csv(os.path.join(TIR, "morgan_rf_baseline.csv"))
master = master.merge(morgan[["canonical_SMILES", "morgan_rf_pred_logdc50"]], on="canonical_SMILES", how="left")
fp = pd.read_csv(os.path.join(REV, "rebuttal_analyses", "05_external_benchmark", "outputs", "fullset_predictions.csv"))
master = master.merge(fp[["canonical_SMILES", "nn_pred_logdc50"]], on="canonical_SMILES", how="left")

stan_train = {c for c in map(canon, pd.read_csv(os.path.join(STAN_TRAIN, "train_compound_smiles.csv"), skiprows=1, header=None, names=["s"])["s"].astype(str)) if c} | \
             {c for c in map(canon, pd.read_csv(os.path.join(STAN_TRAIN, "test_compound_smiles.csv"), skiprows=1, header=None, names=["s"])["s"].astype(str)) if c}
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)
master["held_out_novel"] = master["held_out"].fillna(False) & ~master["exact_in_synglue"]

def orient(col, kind):
    if kind == "dc50": return -np.log10(np.asarray(col, dtype=float))
    if kind == "logdc50": return -np.asarray(col, dtype=float)
    return np.asarray(col, dtype=float)  # proba

MODELS = {
    "SynGlue 3-token": ("gen_pred_dc50_nM", "dc50", None),
    "PROTAC-DC50": ("protacdc50_dc50_nM", "dc50", None),
    "Morgan-RF": ("morgan_rf_pred_logdc50", "logdc50", None),
    "Tanimoto-NN": ("nn_pred_logdc50", "logdc50", None),
    "PDP": ("pdp_active_proba", "proba", None),
    "PDP-XGB": ("pdp_xgb_proba", "proba", None),
    "PROTAC-STAN": ("protacstan_active_proba", "proba", "exact_in_stan"),
}
TH = {10: "<=10nM", 100: "<=100nM", 1000: "<=1uM"}
rowsB = []
for name, (col, kind, flag) in MODELS.items():
    sub = master if flag is None else master[~master[flag].fillna(False)]
    u = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)].dropna(subset=[col])
    if len(u) < 5: continue
    score = orient(u[col].values, kind)
    r = dict(model=name, n=len(u), spearman_rho=stats.spearmanr(-np.log10(u["dc50_median"].values.astype(float)), score)[0])
    for thr, tname in TH.items():
        y = (u["dc50_median"].values <= thr).astype(int)
        n_act, n_inact = int(y.sum()), int((1 - y).sum())
        r[f"n_act_{tname}"] = n_act
        r[f"n_inact_{tname}"] = n_inact
        # compute AUC when both classes present (>=1); flag unreliable if <5 in a class
        if n_act >= 1 and n_inact >= 1:
            r[f"auroc_{tname}"] = roc_auc_score(y, score)
            r[f"auprc_{tname}"] = average_precision_score(y, score)
            r[f"unreliable_{tname}"] = min(n_act, n_inact) < 5
        else:
            r[f"auroc_{tname}"] = np.nan; r[f"auprc_{tname}"] = np.nan
            r[f"unreliable_{tname}"] = True
        k = max(1, int(0.1 * len(u)))
        top = np.argsort(-score)[:k]
        r[f"ef10_{tname}"] = y[top].mean() / max(y.mean(), 1e-9)
        r[f"prec10_{tname}"] = y[top].mean()
        r[f"rec10_{tname}"] = y[top].sum() / max(y.sum(), 1)
    rowsB.append(r)
genB = pd.DataFrame(rowsB)
genB.to_csv(os.path.join(OUT, "prioritization_metrics.csv"), index=False)

# ═══════════ Q-C: OOD characterization ═══════════
print("Computing OOD novelty...")
scaf_syn = set()
tani_syn = []
for s in list(syn_train):
    sc = scaffold_of(s)
    if sc: scaf_syn.add(sc)
    from rdkit.Chem import AllChem
    m = __import__("rdkit").Chem.MolFromSmiles(s)
    if m: tani_syn.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
from rdkit.Chem import AllChem, DataStructs
scaf_ext, tani_ext, novel_scaf = [], [], 0
targets = set()
for s in master["canonical_SMILES"]:
    m = __import__("rdkit").Chem.MolFromSmiles(s)
    if m is None: continue
    sc = scaffold_of(s)
    scaf_ext.append(sc)
    tani_ext.append(float(max(DataStructs.BulkTanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), tani_syn))))
    if sc is not None and sc not in scaf_syn: novel_scaf += 1
# Butina clusters from TACK raw
try:
    import pyarrow.parquet as pq
    tack_raw = pq.read_table("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet").to_pandas()
    bc = tack_raw[["SMILES", "SMILES_Butina_Cluster"]].drop_duplicates()
    bc["SMILES"] = bc["SMILES"].astype(str).apply(canon)
    bcm = dict(zip(bc["SMILES"], bc["SMILES_Butina_Cluster"]))
    clusters = master["canonical_SMILES"].map(bcm).dropna()
    n_clusters = clusters.nunique()
except Exception:
    n_clusters = np.nan
# target / E3 novelty vs SynGlue train
train_targets = set(tr["warhead_Target"].dropna().astype(str).str.upper())
train_e3 = set(tr["E3 ligase"].dropna().astype(str).str.upper())
ext_targets = set(master["Protein_of_Interest"].dropna().astype(str).str.upper())
ext_e3 = set(master["E3_Ligase"].dropna().astype(str).str.upper())
rowsC = [dict(metric="n_external_compounds", value=len(master)),
         dict(metric="scaffold_novel_pct", value=100 * novel_scaf / max(len(scaf_ext), 1)),
         dict(metric="max_tanimoto_to_train_median", value=float(np.median(tani_ext))),
         dict(metric="max_tanimoto_to_train_pct_lt0.4", value=100 * np.mean(np.array(tani_ext) < 0.4)),
         dict(metric="target_novel_vs_train", value=len(ext_targets - train_targets)),
         dict(metric="e3_novel_vs_train", value=len(ext_e3 - train_e3)),
         dict(metric="butina_analog_clusters", value=n_clusters)]
genC = pd.DataFrame(rowsC)
genC.to_csv(os.path.join(OUT, "ood_novelty.csv"), index=False)

# ═══════════ Figures ═══════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
ax = axes[0]
splits = ["Random 80/20", "Scaffold-held-out", "Target-held-out", "E3-held-out",
          "External (3-token)", "External (whole)"]
vals = []
for s in splits:
    row = genA[genA["split"] == s]
    vals.append(float(row["R2"].iloc[0]) if len(row) else np.nan)
ax.bar(range(len(vals)), vals, color=["#1f77b4"] * 4 + ["#ff7f0e", "#ff7f0e"], alpha=0.85)
for x, v in zip(range(len(vals)), vals):
    if v == v: ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticklabels(splits, rotation=18, fontsize=7)
ax.set_ylabel("DC50 R²"); ax.set_title("A. SynGlue quantitative generalization")

ax = axes[1]
d = genB[genB["n"] >= 5]
ax.plot(d["model"], d["auroc_<=100nM"], marker="o", label="AUROC ≤100 nM", color="#2ca02c")
dm = d[d["auroc_<=1uM"].notna()]
ax.plot(dm["model"], dm["auroc_<=1uM"], marker="s", label="AUROC ≤1 µM", color="#d62728")
# annotate unreliable (class < 5) points at 1uM
for _, r in dm.iterrows():
    if r.get("unreliable_<=1uM", False):
        ax.plot(r["model"], r["auroc_<=1uM"], marker="s", mfc="none", mec="#d62728", ms=9)
        ax.annotate(f"n_inact={int(r['n_inact_<=1uM'])}", (r["model"], r["auroc_<=1uM"]),
                    textcoords="offset points", xytext=(0, -16), fontsize=6.5, color="#d62728")
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_ylim(0.4, 0.9)
ax.set_xticklabels(d["model"], rotation=25, fontsize=6.5)
ax.set_ylabel("AUROC (oriented)"); ax.set_title("B. Prioritization — common task")
ax.legend(fontsize=8)

ax = axes[2]
d2 = d[d["ef10_<=100nM"].notna()]
ax.bar(d2["model"], d2["ef10_<=100nM"], color="#9467bd", alpha=0.85)
ax.axhline(1, color="k", ls="--", lw=0.8, label="random = 1")
ax.set_ylim(0, 3)
ax.set_xticklabels(d2["model"], rotation=25, fontsize=6.5)
ax.set_ylabel("EF10% (≤100 nM)"); ax.set_title("C. Enrichment of potent degraders")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "three_questions_figure.png"), dpi=300)
plt.savefig(os.path.join(OUT, "three_questions_figure.svg"))

for nm, f in [("panelA", "gen_quantitative.csv"), ("panelB", "prioritization_metrics.csv"), ("panelC", "ood_novelty.csv")]:
    pass
print("Saved:", OUT)
print("\n=== Q-A ===\n", genA.to_string(index=False))
print("\n=== Q-B (oriented) ===\n", genB[["model","n","spearman_rho","auroc_<=100nM","auprc_<=100nM","ef10_<=100nM","prec10_<=100nM"]].to_string(index=False))
print("\n=== Q-C ===\n", genC.to_string(index=False))
