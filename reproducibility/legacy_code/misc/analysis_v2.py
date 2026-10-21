#!/usr/bin/env python3
"""
MASTER benchmark v2 — ALL parameters from the R1.3 response, per tool vs SynGlue.
Prediction axis (external set, per-tool leakage-controlled):
  P1 degradation potency prediction  : DC50 R2/RMSE/MAE (log10) [SynGlue outputs DC50;
                                       PDP/STAN output P(active) — classifier scores, labeled]
                                       + Dmax metrics (SynGlue only)
  P2 active/inactive enrichment      : AUC(100nM), AUC(1uM)
  P3 ranking                         : Spearman rho + Kendall tau (with bootstrap CIs)
  P4 top-k enrichment                : top 1/5/10% enrichment of actives (<=100nM, <=1uM)
  P5 novelty/diversity               : scaffold novelty vs each tool's train; max Tanimoto
                                       to each tool's train; analog-series (Butina cluster) size
  P6 developability                  : SA, MW, cLogP, TPSA of the scored external set
Generation axis (GSPT1): pass rates + novelty + developability of generated candidates.
Outputs: outputs/params_prediction.csv, params_ranking.csv, params_topk.csv,
         params_novelty.csv, params_developability.csv, params_generation.csv,
         plots, MASTER_REPORT_v2.md
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
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
STAN_TRAIN = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN/data/PROTAC-fine"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None

def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

def develop(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return dict(mw=np.nan, logp=np.nan, tpsa=np.nan, sa=np.nan)
    try:
        sys.path.append(os.path.join(os.path.dirname(RDConfig.__file__) if False else "", ""))
    except Exception:
        pass
    return dict(mw=Descriptors.MolWt(m), logp=Crippen.MolLogP(m), tpsa=rdMolDescriptors.CalcTPSA(m), sa=np.nan)

# ─── Load external master + per-tool novel flags ─────
ext = pd.read_csv(EXT)
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
    stan_smiles += pd.read_csv(os.path.join(STAN_TRAIN, f), header=None, names=["s"])["s"].dropna().astype(str).tolist()
stan_train = {c for c in map(canon, stan_smiles) if c}

master["exact_in_synglue"] = master["canonical_SMILES"].isin(syn_train)
master["exact_in_stan"] = master["canonical_SMILES"].isin(stan_train)
master["pdp_overlap_bound"] = master["Database"].astype(str).str.contains("PROTAC-DB") | master["exact_in_stan"]
master["exact_in_any"] = master["exact_in_synglue"] | master["exact_in_stan"]

# novelty vs SynGlue train (scaffold + Tanimoto) — reuse tiers
tiers = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_ood_tiers.csv"))
master = master.merge(tiers[["canonical_smiles", "max_tanimoto", "scaffold_in_train"]],
                      left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

# novelty vs STAN train (scaffold + Tanimoto)
print("Computing STAN-train novelty...")
stan_scaff = {scaffold_of(s) for s in stan_train if scaffold_of(s)}
stan_fps = [morgan(s) for s in stan_train if morgan(s)]
mt_stan, sc_stan = [], []
for smi in master["canonical_SMILES"]:
    fp = morgan(smi); sc = scaffold_of(smi)
    mt = 0.0
    if fp is not None and stan_fps:
        mt = float(max(DataStructs.BulkTanimotoSimilarity(fp, stan_fps)))
    mt_stan.append(mt); sc_stan.append(sc is not None and sc in stan_scaff)
master["max_tanimoto_stan"] = mt_stan
master["scaffold_in_stan_train"] = sc_stan

# Butina analog clusters (from TACK raw parquet)
try:
    import pyarrow.parquet as pq
    tack = pq.read_table("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet").to_pandas()
    bc = tack[["SMILES", "SMILES_Butina_Cluster"]].drop_duplicates()
    bc["SMILES"] = bc["SMILES"].astype(str).apply(canon)
    bcmap = dict(zip(bc["SMILES"], bc["SMILES_Butina_Cluster"]))
    master["butina_cluster"] = master["canonical_SMILES"].map(bcmap)
except Exception as e:
    print("butina unavailable:", e)
    master["butina_cluster"] = np.nan
master["scaffold_cluster"] = master["SMILES_Scaffold_Cluster"] if "SMILES_Scaffold_Cluster" in master.columns else np.nan

# predictions
gen_whole = pd.read_csv(os.path.join(TIR, "synglue_wholeprotac_gen.csv"))
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
stan_full = pd.read_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"))
master = master.merge(gen_whole[["canonical_SMILES", "gen_whole_pred_dc50_nM", "gen_whole_pred_dmax_pct"]], on="canonical_SMILES", how="left")
master = master.merge(pdp_full[["SMILES", "pdp_active_proba", "pdp_xgb_proba"]], left_on="canonical_SMILES", right_on="SMILES", how="left")
master = master.merge(stan_full[["canonical_smiles", "protacstan_active_proba"]], left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

# developability of scored external compounds (SA via contrib path)
try:
    from rdkit.Chem import RDConfig
    sa_path = [p for p in [os.path.join(RDConfig.RDContribDir, "SA_Score")] if os.path.isdir(p)]
    if sa_path: sys.path.append(sa_path[0])
    import sascorer
    def sa_of(smi):
        m = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(m) if m else np.nan
except Exception:
    def sa_of(smi): return np.nan
print("Computing developability (MW/logP/TPSA/SA)...")
dv = []
for smi in master["canonical_SMILES"]:
    m = Chem.MolFromSmiles(smi)
    dv.append(dict(mw=Descriptors.MolWt(m) if m else np.nan,
                   logp=Crippen.MolLogP(m) if m else np.nan,
                   tpsa=rdMolDescriptors.CalcTPSA(m) if m else np.nan,
                   sa=sa_of(smi)))
ddf = pd.DataFrame(dv)
for c in ["mw", "logp", "tpsa", "sa"]:
    master[c] = ddf[c].values

# ─── Metrics ─────────────────────────────────────────
def bci_pair(x, y, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        ix = rng.randint(0, len(x), len(x))
        try: v.append(fn(x[ix], y[ix]))
        except Exception: pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

TOOLS = [
    ("SynGlue (whole-PROTAC)", "gen_whole_pred_dc50_nM", True, "gen_whole_pred_dmax_pct", "exact_in_synglue", "max_tanimoto", "scaffold_in_train"),
    ("PDP", "pdp_active_proba", False, None, "pdp_overlap_bound", "max_tanimoto", "scaffold_in_train"),
    ("PDP-XGB", "pdp_xgb_proba", False, None, "pdp_overlap_bound", "max_tanimoto", "scaffold_in_train"),
    ("PROTAC-STAN", "protacstan_active_proba", False, None, "exact_in_stan", "max_tanimoto_stan", "scaffold_in_stan_train"),
]

rows_rank, rows_pred, rows_topk, rows_dev, rows_nov = [], [], [], [], []
for name, col, log, dcol, leak_col, tani_col, scaf_col in TOOLS:
    nov = master[~master[leak_col]].copy()
    m = nov[nov["dc50_median"].notna() & ~nov["dc50_censored"].fillna(False)].dropna(subset=[col])
    y = np.log10(m["dc50_median"].values.astype(float))
    p = np.log10(m[col].values.astype(float)) if log else m[col].values.astype(float)
    # P3 ranking
    rho = stats.spearmanr(y, p)[0]; rho_ci = bci_pair(y, p, lambda a, b: stats.spearmanr(a, b)[0])
    tau = stats.kendalltau(y, p)[0]; tau_ci = bci_pair(y, p, lambda a, b: stats.kendalltau(a, b)[0])
    rows_rank.append(dict(model=name, n=len(m), spearman_rho=rho, rho_ci=f"[{rho_ci[0]:.3f},{rho_ci[1]:.3f}]",
                          kendall_tau=tau, tau_ci=f"[{tau_ci[0]:.3f},{tau_ci[1]:.3f}]"))
    # P1 potency prediction
    rows_pred.append(dict(model=name, n=len(m), DC50_R2=r2_score(y, p),
                          DC50_RMSE=float(np.sqrt(mean_squared_error(y, p))),
                          DC50_MAE=float(mean_absolute_error(y, p)),
                          output="DC50 nM (log10)" if log else "P(active) [classifier, NOT DC50]"))
    if dcol is not None:
        dm = nov[nov["dmax_max"].notna()].dropna(subset=[dcol])
        if len(dm) >= 5:
            rows_pred[-1]["Dmax_rho"] = stats.spearmanr(dm["dmax_max"], dm[dcol])[0]
            rows_pred[-1]["Dmax_MAE"] = float(mean_absolute_error(dm["dmax_max"], dm[dcol]))
    # P2 + P4: AUC + top-k enrichment
    row_tk = dict(model=name, n=len(m))
    for thr in [100, 1000]:
        lb = (m["dc50_median"] <= thr).astype(int)
        row_tk[f"AUC_{thr}"] = roc_auc_score(lb, p) if min((lb == 0).sum(), (lb == 1).sum()) >= 5 else np.nan
        base = lb.mean()
        order = np.argsort(p) if log else np.argsort(-p)  # most potent / most active first
        lb_arr = lb.values
        for k in [0.01, 0.05, 0.10]:
            kk = max(1, int(k * len(m)))
            top = lb_arr[order[:kk]]
            row_tk[f"top{int(k*100)}%_enrich_{thr}"] = float(top.mean()) / max(base, 1e-6)
    rows_topk.append(row_tk)
    # P5 novelty/diversity of the scored set
    rows_nov.append(dict(model=name, n_scored=len(nov),
                         exact_removed=int(master[leak_col].sum()),
                         median_tanimoto_to_own_train=float(np.nanmedian(nov[tani_col])),
                         pct_scaffold_novel_to_own_train=float(100 * (~nov[scaf_col].fillna(False)).mean()),
                         n_butina_clusters=nov["butina_cluster"].nunique() if nov["butina_cluster"].notna().any() else np.nan))
    # P6 developability of scored external set
    rows_dev.append(dict(model=name,
                         median_MW=float(np.nanmedian(nov["mw"])), median_logP=float(np.nanmedian(nov["logp"])),
                         median_TPSA=float(np.nanmedian(nov["tpsa"])),
                         median_SA=np.nanmedian(nov["sa"]) if nov["sa"].notna().any() else np.nan))

for name, df_, f in [("params_ranking", pd.DataFrame(rows_rank), "params_ranking.csv"),
                     ("params_prediction", pd.DataFrame(rows_pred), "params_prediction.csv"),
                     ("params_topk", pd.DataFrame(rows_topk), "params_topk.csv"),
                     ("params_novelty", pd.DataFrame(rows_nov), "params_novelty.csv"),
                     ("params_developability", pd.DataFrame(rows_dev), "params_developability.csv")]:
    df_.to_csv(os.path.join(OUT, f), index=False)

print("== P3 RANKING =="); print(pd.DataFrame(rows_rank).to_string(index=False))
print("\n== P1 PREDICTION =="); print(pd.DataFrame(rows_pred).to_string(index=False))
print("\n== P2+P4 AUC / top-k =="); print(pd.DataFrame(rows_topk).to_string(index=False))
print("\n== P5 NOVELTY =="); print(pd.DataFrame(rows_nov).to_string(index=False))
print("\n== P6 DEVELOPABILITY =="); print(pd.DataFrame(rows_dev).to_string(index=False))

# ─── Plots ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
rk = pd.DataFrame(rows_rank)
ax = axes[0, 0]
ax.bar(rk["model"], rk["spearman_rho"], color="#1f77b4", alpha=0.85)
for i, r in rk.iterrows():
    lo, hi = [float(x.strip("[]")) for x in r["rho_ci"].split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.8); ax.set_xticklabels(rk["model"], rotation=15, fontsize=7)
ax.set_title("Spearman ρ (DC50 ranking)"); ax.set_ylabel("ρ")
ax = axes[0, 1]
ax.bar(rk["model"], rk["kendall_tau"], color="#ff7f0e", alpha=0.85)
for i, r in rk.iterrows():
    lo, hi = [float(x.strip("[]")) for x in r["tau_ci"].split(",")]
    ax.plot([i, i], [lo, hi], color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.8); ax.set_xticklabels(rk["model"], rotation=15, fontsize=7)
ax.set_title("Kendall τ (DC50 ranking)"); ax.set_ylabel("τ")
ax = axes[1, 0]
tk = pd.DataFrame(rows_topk)
ax.plot(tk["model"], tk["top10%_enrich_100"], marker="o", label="top-10% (≤100nM)")
ax.plot(tk["model"], tk["top10%_enrich_1000"], marker="s", label="top-10% (≤1µM)")
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticklabels(tk["model"], rotation=15, fontsize=7); ax.legend(fontsize=8)
ax.set_title("Top-10% enrichment"); ax.set_ylabel("enrichment ratio")
ax = axes[1, 1]
tk2 = tk.set_index("model")
ax.plot(tk2.index, tk2["top1%_enrich_100"], marker="o", label="top-1% (≤100nM)")
ax.plot(tk2.index, tk2["top5%_enrich_100"], marker="s", label="top-5% (≤100nM)")
ax.plot(tk2.index, tk2["top10%_enrich_100"], marker="^", label="top-10% (≤100nM)")
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticklabels(tk2.index, rotation=15, fontsize=7); ax.legend(fontsize=8)
ax.set_title("Top-k enrichment (≤100 nM)"); ax.set_ylabel("enrichment ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "params_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "params_plot.svg"))
print(f"\nSaved: {OUT}/")
