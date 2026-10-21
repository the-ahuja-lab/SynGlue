#!/usr/bin/env python3
"""
WS1 tiered inference — per-OOD-tier datasets scored with the SAME generalized model.
===============================================================================
1. Build master tier table for all 3,518 external compounds (non-exclusive tiers):
   exact_in_train (EXCLUDED from scoring), scaffold_ood, chemical_ood (T<0.4),
   chemical_ood_soft (0.4-0.6), chemical_in (>=0.6), component_ood_e3 (E3 novel
   vs train library), POI_unseen, E3_unseen, deep_ood (scaffold_ood & T<0.4)
2. Extend GROVER embeddings to ALL decomposed non-exact compounds with endpoints
3. Score each tier with the SAME generalized 5-fold ensemble (per-compound
   conditioning); per-fold RFs regenerated deterministically (same splits/seed)
4. Per-tier metrics + "are we nearby" stats (MAE, median |log error|, % within
   0.5/1.0 log10 unit, Spearman rho + bootstrap CI)
Outputs (benchmarking/ws1/tiers/): tier_master.csv, tier_scoring_set.csv,
  per-tier *_inference.csv, tier_metrics.csv, tier_metrics.md
Usage: conda run -n synglu_test_package python ws1_tier_scoring.py
"""
import os, sys, time, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
TIR = os.path.join(OUT, "tiers")
MODEL_DIR = os.path.join(REV, "models")
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
GROVER_CKPT = "/storage/savi/saveenas/Projects/SynGlue_Py/models/grover_fixed.pt"
GROVER_PATCHED = "/tmp/ws1_grover/grover_patched"
os.makedirs(TIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, REV)
from reproduce_original_model import MultiTaskProtacModel, load_data, train_transformer

N_FOLDS, SEED = 5, 42

# ─── 0. Regenerate fold splits (same as training) ────
print("Regenerating training folds (deterministic)...")
Xtr, y50tr, ymtr, metatr = load_data()
sig = pd.DataFrame({"w": [Xtr[i, 0].tobytes() for i in range(len(Xtr))],
                    "l": [Xtr[i, 1].tobytes() for i in range(len(Xtr))],
                    "e": [Xtr[i, 2].tobytes() for i in range(len(Xtr))]})
dup = sig.duplicated(keep="first")
Xtr = Xtr[~dup]; y50tr = y50tr[~dup]; ymtr = ymtr[~dup]; metatr = metatr[~dup].reset_index(drop=True)
scaff = []
for s in metatr["SMILES"].values:
    m = Chem.MolFromSmiles(str(s))
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        sc = None
    scaff.append(sc or f"noscaf_{len(scaff)}")
groups = np.array(scaff)
gss = GroupShuffleSplit(n_splits=N_FOLDS, test_size=0.2, random_state=SEED)
folds = list(gss.split(Xtr, groups=groups))

def make_loader(idx, shuffle=False):
    ds = TensorDataset(torch.tensor(Xtr[idx], dtype=torch.float32),
                       torch.tensor(y50tr[idx], dtype=torch.float32),
                       torch.tensor(ymtr[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=64, shuffle=shuffle)

def extract(model, idx):
    model.eval(); Z = []
    with torch.no_grad():
        for xb, _, _ in make_loader(idx):
            h = model.proj(xb.to(DEVICE)); h = model.transformer(h)
            attn = torch.softmax(model.attn_pool(h), dim=1)
            Z.append((h * attn).sum(dim=1).cpu().numpy())
    return np.vstack(Z)

# ─── 1. Master tier table ────────────────────────────
print("Building master tier table...")
tiers = pd.read_csv(os.path.join(OUT, "ws1_ood_tiers.csv"))
tiers = tiers.drop(columns=[c for c in ["n_dc50", "n_dmax", "has_censored", "n_records"] if c in tiers.columns])
comp = pd.read_csv(os.path.join(OUT, "ws1_components_v2.csv"))
ext = pd.read_csv(EXT)
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "n_dc50": int((g["Value_Type"] == "DC50").sum()),
    "n_dmax": int((g["Value_Type"] == "Dmax").sum()),
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
    "cell_lines": ";".join(g["Cell_Line"].dropna().astype(str).unique()[:3]),
    "assay": ";".join(g["Assay"].dropna().astype(str).unique()[:2]),
}))
master = tiers.merge(comp[["canonical_smiles", "e3_match", "e3_in_train", "warhead_match",
                           "linker_smiles", "linker_single_fragment", "warhead_smiles", "e3_smiles"]],
                     on="canonical_smiles", how="left")
master = master.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                      on="canonical_smiles", how="left")
master["has_endpoint"] = master["n_dc50"].fillna(0).gt(0) | master["n_dmax"].fillna(0).gt(0)
master["component_ood_e3"] = master["e3_match"] == "mcs_core_match"
master["chemical_ood"] = master["max_tanimoto"] < 0.4
master["chemical_ood_soft"] = (master["max_tanimoto"] >= 0.4) & (master["max_tanimoto"] < 0.6)
master["chemical_in"] = master["max_tanimoto"] >= 0.6
master["scaffold_ood_flag"] = master["tier"] == "scaffold_ood"
master["exact_in_train_flag"] = master["tier"] == "exact_in_train"
master["deep_ood"] = master["scaffold_ood_flag"] & master["chemical_ood"]
master["POI_unseen"] = master["POI_seen"] == False
master["E3_unseen"] = master["E3_seen"] == False
master["scorable"] = (master["linker_smiles"].notna() & master["linker_single_fragment"].fillna(False)
                      & master["e3_smiles"].notna() & master["warhead_smiles"].notna()
                      & master["has_endpoint"] & (~master["exact_in_train_flag"]))
master.to_csv(os.path.join(TIR, "tier_master.csv"), index=False)
print(f"  master: {len(master)} | scorable: {master['scorable'].sum()}")

# ─── 2. Extend embeddings ────────────────────────────
def grover_embed(smiles_list, tag):
    cache = os.path.join(TIR, "emb", f"{tag}_fps.npz")
    if os.path.exists(cache):
        arr = np.load(cache)["fps"][:, :4800]
        assert arr.shape[0] == len(smiles_list), f"{tag} cache mismatch"
        return {s: arr[i] for i, s in enumerate(smiles_list)}
    os.makedirs(os.path.join(TIR, "emb"), exist_ok=True)
    fn = os.path.join(TIR, "emb", f"{tag}_smiles.csv")
    pd.DataFrame({"smiles": smiles_list}).to_csv(fn, index=False)
    feat = os.path.join(TIR, "emb", f"{tag}_features.npz")
    fps = os.path.join(TIR, "emb", f"{tag}_fps.npz")
    subprocess.run(f"python {GROVER_PATCHED}/scripts/save_features.py --data_path {fn} "
                  f"--save_path {feat} --features_generator rdkit_2d_normalized --restart",
                  shell=True, check=True, capture_output=True)
    subprocess.run(f"python {GROVER_PATCHED}/main.py fingerprint --data_path {fn} "
                  f"--features_path {feat} --checkpoint_path {GROVER_CKPT} "
                  f"--fingerprint_source both --output {fps}",
                  shell=True, check=True, capture_output=True)
    arr = np.load(fps)["fps"][:, :4800]
    return {s: arr[i] for i, s in enumerate(smiles_list)}

sc = master[master["scorable"]].copy().reset_index(drop=True)
w_uniq = sorted(sc["warhead_smiles"].dropna().unique())
l_uniq = sorted(sc["linker_smiles"].dropna().unique())
e_uniq = sorted(sc["e3_smiles"].dropna().unique())
print(f"  embedding {len(w_uniq)} warheads, {len(l_uniq)} linkers, {len(e_uniq)} E3...")
Wm, Lm, Em = grover_embed(w_uniq, "warhead"), grover_embed(l_uniq, "linker"), grover_embed(e_uniq, "e3")
Xe = np.zeros((len(sc), 3, 4800), dtype=np.float32)
for i, r in sc.iterrows():
    Xe[i, 0] = Wm[r["warhead_smiles"]]; Xe[i, 1] = Lm[r["linker_smiles"]]; Xe[i, 2] = Em[r["e3_smiles"]]

def extract_ext(model):
    model.eval(); Z = []
    with torch.no_grad():
        for b in range(0, len(sc), 64):
            h = model.proj(torch.tensor(Xe[b:b+64], dtype=torch.float32).to(DEVICE))
            h = model.transformer(h)
            attn = torch.softmax(model.attn_pool(h), dim=1)
            Z.append((h * attn).sum(dim=1).cpu().numpy())
    return np.vstack(Z)

# ─── 3. Score with generalized ensemble ──────────────
print("Scoring with generalized 5-fold ensemble (predict-then-average)...")
Zk_all, pe50, pem = [], [], []
for k in range(N_FOLDS):
    m = MultiTaskProtacModel(input_dim=4800).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"generalized_transformer_fold{k}.pt"),
                                 map_location=DEVICE, weights_only=True)["state_dict"])
    m.eval()
    # fold RFs: retrain on the fold's train split (deterministic)
    tr_i, _ = folds[k]
    tr_groups = groups[tr_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + k)
    tr2, va2 = next(iter(gss2.split(np.arange(len(tr_i)), groups=tr_groups)))
    tr_idx = tr_i[tr2]
    Ztr = extract(m, tr_idx)
    mask50, maskmax = np.isfinite(y50tr[tr_idx]), np.isfinite(ymtr[tr_idx])
    rf50 = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=SEED)
    rf50.fit(Ztr[mask50], y50tr[tr_idx][mask50])
    rfmax = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=SEED)
    rfmax.fit(Ztr[maskmax], ymtr[tr_idx][maskmax])
    Zk = extract_ext(m)
    Zk_all.append(Zk)
    pe50.append(rf50.predict(Zk)); pem.append(rfmax.predict(Zk))
    print(f"  fold {k} done")
Z_all = np.stack(Zk_all)                       # (folds, n, 512)
Ze = np.mean(Z_all, axis=0)                    # embedding-avg (for reference)
sc["gen_pred_dc50_nM"] = 10 ** np.mean(pe50, axis=0)      # predict-then-average (primary)
sc["gen_pred_dmax_pct"] = np.mean(pem, axis=0)
sc.to_csv(os.path.join(TIR, "tier_scoring_set.csv"), index=False)
print(f"  scored {len(sc)} compounds")

# (embedding-avg reference predictions, matching earlier generalized report)
import joblib as _jl
def _train_rf(Ztr2, y2):
    mm = np.isfinite(y2)
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=SEED)
    rf.fit(Ztr2[mm], y2[mm]); return rf
pe50_ea, pem_ea = [], []
for k in range(N_FOLDS):
    tr_i, _ = folds[k]
    tr_groups = groups[tr_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + k)
    tr2, va2 = next(iter(gss2.split(np.arange(len(tr_i)), groups=tr_groups)))
    tr_idx = tr_i[tr2]
    mk = MultiTaskProtacModel(input_dim=4800).to(DEVICE)
    mk.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"generalized_transformer_fold{k}.pt"),
                                  map_location=DEVICE, weights_only=True)["state_dict"])
    mk.eval()
    Ztrk = extract(mk, tr_idx)
    pe50_ea.append(_train_rf(Ztrk, y50tr[tr_idx]).predict(Ze))
    pem_ea.append(_train_rf(Ztrk, ymtr[tr_idx]).predict(Ze))
sc["gen_pred_dc50_nM_embavg"] = 10 ** np.mean(pe50_ea, axis=0)
sc["gen_pred_dmax_pct_embavg"] = np.mean(pem_ea, axis=0)
sc.to_csv(os.path.join(TIR, "tier_scoring_set.csv"), index=False)
print("  embedding-avg reference predictions saved too")

# ─── 4. Per-tier inference CSVs + metrics ────────────
TIERS = [("scaffold_ood", "scaffold_ood_flag"), ("chemical_ood", "chemical_ood"),
         ("chemical_ood_soft", "chemical_ood_soft"), ("chemical_in", "chemical_in"),
         ("component_ood_e3", "component_ood_e3"), ("POI_unseen", "POI_unseen"),
         ("E3_unseen", "E3_unseen"), ("deep_ood", "deep_ood")]

def bootstrap_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n_boot):
        ix = rng.randint(0, len(x), len(x))
        try: vals.append(stats.spearmanr(x[ix], y[ix])[0])
        except Exception: pass
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)

def row_metrics(sub, endpoint, pred_col, log=False):
    if log:
        y = np.log10(sub[endpoint].values.astype(float)); p = np.log10(sub[pred_col].values.astype(float))
    else:
        y = sub[endpoint].values.astype(float); p = sub[pred_col].values.astype(float)
    n = len(y)
    r2 = r2_score(y, p); rmse = float(np.sqrt(mean_squared_error(y, p))); mae = mean_absolute_error(y, p)
    rho = stats.spearmanr(y, p)[0]
    ci = bootstrap_ci(y, p)
    med_abs_log = float(np.median(np.abs(np.log10(sub[pred_col].values.astype(float)) -
                                         np.log10(sub[endpoint].values.astype(float)))))
    pct_half = float(np.mean(np.abs(np.log10(sub[pred_col].values.astype(float)) -
                                    np.log10(sub[endpoint].values.astype(float))) <= 0.5))
    pct_one = float(np.mean(np.abs(np.log10(sub[pred_col].values.astype(float)) -
                                   np.log10(sub[endpoint].values.astype(float))) <= 1.0))
    return dict(n=n, R2=r2, RMSE=rmse, MAE=mae, rho=rho, ci_low=ci[0], ci_high=ci[1],
                med_abs_log10_err=med_abs_log, pct_within_0_5log=pct_half, pct_within_1_0log=pct_one)

metric_rows = []
for name, flag in TIERS:
    sub = sc[sc[flag]].copy()
    # uncensored DC50 subset
    d50 = sub[sub["dc50_median"].notna() & ~sub["dc50_censored"].fillna(False)]
    dmax = sub[sub["dmax_max"].notna()]
    r50 = row_metrics(d50, "dc50_median", "gen_pred_dc50_nM", log=True) if len(d50) >= 5 else {"n": len(d50)}
    rmax = row_metrics(dmax, "dmax_max", "gen_pred_dmax_pct") if len(dmax) >= 5 else {"n": len(dmax)}
    out = sub[["canonical_smiles", "E3", "POI_UniProt", "cell_lines", "assay",
               "dc50_median", "dmax_max", "dc50_censored", "gen_pred_dc50_nM", "gen_pred_dmax_pct",
               "max_tanimoto", "e3_match", "warhead_match"]].copy()
    out["abs_log10_dc50_err"] = np.abs(np.log10(out["gen_pred_dc50_nM"]) - np.log10(out["dc50_median"]))
    out.to_csv(os.path.join(TIR, f"tier_{name}_inference.csv"), index=False)
    metric_rows.append({"tier": name, "n_all": len(sub),
                        "dc50_n": r50.get("n", 0), "dc50_R2": r50.get("R2", np.nan),
                        "dc50_rho": r50.get("rho", np.nan), "dc50_ci": f"[{r50.get('ci_low', np.nan):.3f},{r50.get('ci_high', np.nan):.3f}]",
                        "dc50_MAE_log": r50.get("MAE", np.nan), "dc50_med_abs_log": r50.get("med_abs_log10_err", np.nan),
                        "dc50_pct_within_1log": r50.get("pct_within_1_0log", np.nan),
                        "dmax_n": rmax.get("n", 0), "dmax_R2": rmax.get("R2", np.nan),
                        "dmax_rho": rmax.get("rho", np.nan), "dmax_ci": f"[{rmax.get('ci_low', np.nan):.3f},{rmax.get('ci_high', np.nan):.3f}]",
                        "dmax_MAE": rmax.get("MAE", np.nan)})
    print(f"  {name}: n={len(sub)} | DC50 ρ={r50.get('rho', float('nan')):.3f} (n={r50.get('n',0)}) "
          f"med|Δlog10|={r50.get('med_abs_log10_err', float('nan')):.2f} | Dmax ρ={rmax.get('rho', float('nan')):.3f} (n={rmax.get('n',0)})")

md = pd.DataFrame(metric_rows)
md.to_csv(os.path.join(TIR, "tier_metrics.csv"), index=False)
# md report
lines = ["# WS1 Tiered Inference — Generalized Model (same model, per-tier datasets)", "",
         f"- Scoring set: {len(sc)} leakage-free decomposed compounds (per-compound conditioning, 5-fold ensemble)",
         "- Tiers non-exclusive; exact_in_train excluded everywhere", "",
         "| Tier | n | DC50 n | R² | ρ [95% CI] | med\\|Δlog10\\| | % within 1 log | Dmax n | R² | ρ [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in md.iterrows():
    lines.append(f"| {r['tier']} | {r['n_all']} | {r['dc50_n']} | {r['dc50_R2']:.3f} | {r['dc50_rho']:.3f} {r['dc50_ci']} | "
                 f"{r['dc50_med_abs_log']:.2f} | {100*r['dc50_pct_within_1log']:.0f}% | {r['dmax_n']} | {r['dmax_R2']:.3f} | {r['dmax_rho']:.3f} {r['dmax_ci']} |")
lines += ["", "Note: 'are we nearby' = med |Δlog10(DC50)| (0 = exact; <0.5 ≈ within 3-fold; <1.0 ≈ within 10-fold) and % within 1 log unit.", ""]
open(os.path.join(TIR, "tier_metrics.md"), "w").write("\n".join(lines))
print("\n".join(lines[-12:]))
print(f"\nSaved: {TIR}/")
