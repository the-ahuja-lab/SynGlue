#!/usr/bin/env python3
"""
GENERALIZED MODEL — leakage-resistant retrain (user-directed, post-frozen-test)
================================================================================
Fixes identified in WS1:
 1. Per-compound conditioning: X[i] = [GROVER(warhead_i), GROVER(linker_i), GROVER(E3_i)]
    (frozen pipeline used CONSTANT warhead/E3 vectors — a deployment flaw)
 2. Leakage-resistant model selection: Murcko-scaffold-grouped CV with early stopping
    on scaffold-held-out validation (no random splits)
 3. Fold ensemble at inference (average of fold models)
Architecture: identical to original MultiTaskProtacModel (comparability) + RF heads
  (as in the shipped pipeline: transformer fused embeddings -> RF).
Evaluation:
  - internal: scaffold-grouped CV (honest in-distribution estimate)
  - external: SAME protocol as WS1 (151 leakage-free decomposed compounds, same
    endpoints/metrics) -> direct comparison with frozen model.
Usage:
  conda run -n synglu_test_package python train_generalized_model.py
Outputs:
  models/generalized_transformer_fold{k}.pt, benchmarking/ws1/ws1_gen_predictions.csv,
  benchmarking/ws1/ws1_gen_metrics.md
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/revision")
from reproduce_original_model import MultiTaskProtacModel, load_data, train_transformer

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws1")
MODEL_DIR = os.path.join(REV, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
SEED = 42
RF_N, RF_DEPTH = 500, 20

# ─── 1. Training data (per-compound conditioning) ────
print("Loading training data (fast loader, original semantics)...")
X, y50, ym, meta = load_data()          # 9,066 matched rows
# dedupe identical (compound, w, l, e) rows
sig = pd.DataFrame({
    "w": X[:, 0].tobytes() if False else [X[i, 0].tobytes() for i in range(len(X))],
    "l": [X[i, 1].tobytes() for i in range(len(X))],
    "e": [X[i, 2].tobytes() for i in range(len(X))],
})
meta2 = meta.reset_index(drop=True)
dup = sig.duplicated(keep="first")
X = X[~dup]; y50 = y50[~dup]; ym = ym[~dup]; meta = meta2[~dup].reset_index(drop=True)
print(f"  after dedupe: {len(X)} unique (compound,component) rows")
lab = np.isfinite(y50) | np.isfinite(ym)
print(f"  rows with ≥1 label: {lab.sum()} | DC50 labels: {np.isfinite(y50).sum()} | Dmax: {np.isfinite(ym).sum()}")

# scaffolds (leakage groups)
scaff = []
for s in meta["SMILES"].values:
    m = Chem.MolFromSmiles(str(s))
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        sc = None
    scaff.append(sc or f"noscaf_{len(scaff)}")
groups = np.array(scaff)
print(f"  scaffold groups: {len(set(groups))}")

# ─── 2. Scaffold-grouped CV ──────────────────────────
gss = GroupShuffleSplit(n_splits=N_FOLDS, test_size=0.2, random_state=SEED)
folds = list(gss.split(X, groups=groups))
print(f"  {N_FOLDS} scaffold folds (test=20% scaffold groups)")

BATCH = 64

def make_loader(idx, shuffle=True):
    xb = torch.tensor(X[idx], dtype=torch.float32)
    ds = TensorDataset(xb, torch.tensor(y50[idx], dtype=torch.float32),
                       torch.tensor(ym[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

def train_rf(Ztr, y):
    mask = np.isfinite(y)
    if mask.sum() == 0: return None
    rf = RandomForestRegressor(n_estimators=RF_N, max_depth=RF_DEPTH, n_jobs=-1,
                               random_state=SEED)
    rf.fit(Ztr[mask], y[mask])
    return rf

def metrics(y, p):
    m = np.isfinite(y)
    if m.sum() < 5: return None
    r2 = r2_score(y[m], p[m])
    rmse = float(np.sqrt(mean_squared_error(y[m], p[m])))
    mae = mean_absolute_error(y[m], p[m])
    rho = stats.spearmanr(y[m], p[m])[0]
    return dict(R2=r2, RMSE=rmse, MAE=mae, rho=rho, n=int(m.sum()))

def extract(model, idx):
    model.eval()
    Z = []
    with torch.no_grad():
        for xb, _, _ in make_loader(idx, shuffle=False):
            h = model.proj(xb.to(DEVICE))
            h = model.transformer(h)
            attn = torch.softmax(model.attn_pool(h), dim=1)
            Z.append((h * attn).sum(dim=1).cpu().numpy())
    return np.vstack(Z)

# ─── 3. Train folds, collect internal metrics ────────
print("\nTraining fold models (scaffold-CV)...")
fold_models, fold_rfs50, fold_rfsmax = [], [], []
internal = []
for k, (tr_i, te_i) in enumerate(folds):
    # within-train scaffold val split (10% of scaffolds)
    tr_groups = groups[tr_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + k)
    tr2, va2 = next(iter(gss2.split(np.arange(len(tr_i)), groups=tr_groups)))
    tr_idx, va_idx = tr_i[tr2], tr_i[va2]
    model = MultiTaskProtacModel(input_dim=4800).to(DEVICE)
    model, dt = train_transformer(model, make_loader(tr_idx), make_loader(va_idx))
    torch.save({"state_dict": model.state_dict(), "fold": k, "n_train": len(tr_idx)},
               os.path.join(MODEL_DIR, f"generalized_transformer_fold{k}.pt"))
    Ztr = extract(model, tr_idx); Zva = extract(model, va_idx); Zte = extract(model, te_i)
    rf50 = train_rf(Ztr, y50[tr_idx]); rfmax = train_rf(Ztr, ym[tr_idx])
    # internal test metrics (per fold)
    p50 = rf50.predict(Zte) if rf50 else np.full(len(te_i), np.nan)
    pm = rfmax.predict(Zte) if rfmax else np.full(len(te_i), np.nan)
    m50 = metrics(y50[te_i], p50); mm = metrics(ym[te_i], pm)
    internal.append({"fold": k, "dc50": m50, "dmax": mm})
    print(f"  fold {k}: DC50 R²={m50['R2'] if m50 else None:.3f} ρ={m50['rho'] if m50 else None:.3f}"
          f" (n={m50['n'] if m50 else 0}) | Dmax R²={mm['R2'] if mm else None:.3f} ρ={mm['rho'] if mm else None:.3f}"
          f" (n={mm['n'] if mm else 0})")
    fold_models.append(model); fold_rfs50.append(rf50); fold_rfsmax.append(rfmax)

# ─── 4. External evaluation (same protocol as WS1) ───
print("\nScoring external set with generalized ensemble...")
sel = pd.read_csv(os.path.join(OUT, "ws1_scoring_set.csv"))
# rebuild unique-component -> npz row mapping (same sorted order as WS1 run)
def load_emb(tag):
    fps = np.load(f"/tmp/ws1_grover/{tag}_fps.npz")["fps"][:, :4800]
    uniq = sorted(sel[f"{tag}_smiles"].dropna().unique())
    assert len(uniq) == fps.shape[0], f"{tag}: {len(uniq)} vs {fps.shape[0]}"
    return {s: fps[i] for i, s in enumerate(uniq)}
Wm, Lm, Em = load_emb("warhead"), load_emb("linker"), load_emb("e3")
Xe = np.zeros((len(sel), 3, 4800), dtype=np.float32)
for i, r in sel.iterrows():
    Xe[i, 0] = Wm[r["warhead_smiles"]]
    Xe[i, 1] = Lm[r["linker_smiles"]]
    Xe[i, 2] = Em[r["e3_smiles"]]

def extract_ext(model):
    model.eval(); Z = []
    with torch.no_grad():
        for b in range(0, len(sel), 64):
            h = model.proj(torch.tensor(Xe[b:b+64], dtype=torch.float32).to(DEVICE))
            h = model.transformer(h)
            attn = torch.softmax(model.attn_pool(h), dim=1)
            Z.append((h * attn).sum(dim=1).cpu().numpy())
    return np.vstack(Z)

Ze = np.mean([extract_ext(m) for m in fold_models], axis=0)  # ensemble
pe50 = np.mean([rf.predict(Ze) if rf else np.full(len(sel), np.nan) for rf in fold_rfs50], axis=0)
pem = np.mean([rf.predict(Ze) if rf else np.full(len(sel), np.nan) for rf in fold_rfsmax], axis=0)
sel["gen_pred_dc50_nM"] = 10 ** pe50
sel["gen_pred_dmax_pct"] = pem
sel.to_csv(os.path.join(OUT, "ws1_gen_predictions.csv"), index=False)

# same metrics as ws1_rf_metrics.py
def metrics_table(sub, name):
    L = [f"### {name} (N={len(sub)})", "", "| Endpoint | N | R² | RMSE | MAE | Spearman ρ | ρ 95% CI |", "|----------|---|----|----|----|----|----|"]
    for col, pred, label, log, cens in [("dc50_median", "gen_pred_dc50_nM", "DC50 (nM)", True, "dc50_censored"),
                                        ("dmax_max", "gen_pred_dmax_pct", "Dmax (%)", False, None)]:
        mask = sub[col].notna() & sub[pred].notna()
        if cens:
            mask &= ~sub[cens].fillna(False)
        y = np.log10(sub.loc[mask, col].values.astype(float)) if log else sub.loc[mask, col].values.astype(float)
        p = np.log10(sub.loc[mask, pred].values.astype(float)) if log else sub.loc[mask, pred].values.astype(float)
        if len(y) < 5:
            L.append(f"| {label} | {len(y)} | n/a | | | | |"); continue
        r2 = r2_score(y, p); rmse = float(np.sqrt(mean_squared_error(y, p))); mae = mean_absolute_error(y, p)
        rho = stats.spearmanr(y, p)[0]
        rng = np.random.RandomState(0); vals = []
        for _ in range(2000):
            ix = rng.randint(0, len(y), len(y))
            try: vals.append(stats.spearmanr(y[ix], p[ix])[0])
            except Exception: pass
        ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
        L.append(f"| {label} | {len(y)} | {r2:.3f} | {rmse:.3f} | {mae:.3f} | {rho:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    L.append(""); return "\n".join(L)

md = ["# Generalized Model — External Scoring (same protocol as WS1)", "",
      "- Per-compound conditioning + scaffold-CV early stopping + 5-fold ensemble",
      "- Frozen-model comparison: DC50 ρ=−0.04 [−0.26,0.21], R²=−1.56 (n=70); Dmax ρ=−0.26, R²=−0.96 (n=88)", ""]
md.append(metrics_table(sel, "Generalized model — all"))
for tier in ["scaffold_ood", "chemical_in"]:
    sub = sel[sel["tier"] == tier]
    if len(sub): md.append(metrics_table(sub, f"Tier: {tier}"))
# internal CV summary
md.append("## Internal scaffold-CV (generalized model)")
for f in internal:
    d = f["dc50"] or {}; m = f["dmax"] or {}
    md.append(f"- fold {f['fold']}: DC50 R²={d.get('R2', float('nan')):.3f} ρ={d.get('rho', float('nan')):.3f} (n={d.get('n',0)}) | Dmax R²={m.get('R2', float('nan')):.3f} ρ={m.get('rho', float('nan')):.3f} (n={m.get('n',0)})")
open(os.path.join(OUT, "ws1_gen_metrics.md"), "w").write("\n".join(md))
print("\n".join(md))
print(f"\nSaved: {OUT}/ws1_gen_predictions.csv + ws1_gen_metrics.md")
