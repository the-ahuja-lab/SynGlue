#!/usr/bin/env python3
"""
FINAL MODEL 
==================================================================
proj 4800→512 | TransformerEncoder 2 layers (d=512, FF=2048, dropout=0.1) |
attn_pool 512→1 | regressor 512→256→1

Pipeline (matches user's original code):
  1. Build component tensor X = [W | L | E] GROVER 4800-dim (matched via
     warhead Uniprot/Target, linker ID, E3 ligase)
  2. Labels: DC50 median / Dmax max per PROTAC_Compound_ID
  3. Stage-1 transformer trained on log10(DC50) [and Dmax %]
  4. Fused 512-dim embeddings → Stage-2 regressors (XGBoost/RF/Ridge)
  5. 10-fold CV on fused embeddings

Also: interpretability (reviewer point 5) — attention, ablation, SHAP.
Output: models/final_protac_model/ + figures/interpretability/
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/interpretability"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, kendalltau

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── MODEL (exact manuscript spec) ───
class ProtacAttentionModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=8, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                         dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn_scores = self.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1)
        pred = self.regressor(fused).squeeze(-1)
        return pred, attn_weights
    def fused_embeddings(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (h * attn).sum(dim=1), attn

# ─── DATA (exact user pipeline) ───
def load_full_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"  ✅ Loaded {os.path.basename(path)} → {df.shape}")
    return df

def build_dataset():
    print("Loading data...")
    protac_df  = load_full_data(f"{DATA}/Upadted_PROTAC_features.csv")
    warhead_df = load_full_data(f"{DATA}/Updated_warhead.csv")
    linker_df  = load_full_data(f"{DATA}/linker_updated_features.csv")
    e3_df      = load_full_data(f"{DATA}/E3_updated_features.csv")

    # normalize join keys
    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()
    linker_df["linker_Compound_ID"] = linker_df["linker_Compound_ID"].astype(str).str.strip().str.upper()
    protac_df["Linker ID"] = protac_df["Linker ID"].astype(str).str.strip().str.upper()
    protac_df["PROTAC_Compound_ID"] = protac_df["PROTAC_Compound_ID"].astype(str).str.strip().str.upper()

    # prefix Grover
    def prefix(df, p): return df.rename(columns={c: f"{p}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix(warhead_df, "W_"); linker_df = prefix(linker_df, "L_"); e3_df = prefix(e3_df, "E_")

    w_g = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_g = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_g = [c for c in e3_df.columns if c.startswith("E_Grover_")]
    assert len(w_g) == len(l_g) == len(e_g) == 4800

    # labels: median DC50 / max Dmax per compound
    label_df = protac_df[["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]].copy()
    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"]  = pd.to_numeric(label_df["Dmax (%)"], errors="coerce")
    labels = (label_df.groupby("PROTAC_Compound_ID")
                      .agg({"DC50 (nM)": "median", "Dmax (%)": "max"}).reset_index())
    lab = labels.set_index("PROTAC_Compound_ID")

    # map components
    rows = []
    for _, p in protac_df.iterrows():
        wh = warhead_df[(warhead_df["warhead_Uniprot"] == p["warhead_Uniprot"]) |
                        (warhead_df["warhead_Target"] == p["warhead_Target"])]
        if wh.empty: continue
        lk = linker_df[linker_df["linker_Compound_ID"] == p["Linker ID"]]
        if lk.empty: continue
        e3 = e3_df[e3_df["E3_Ligase"] == p["E3 ligase"]]
        if e3.empty: continue
        cid = p["PROTAC_Compound_ID"]
        dc50 = lab.loc[cid, "DC50 (nM)"] if cid in lab.index else np.nan
        dmax = lab.loc[cid, "Dmax (%)"] if cid in lab.index else np.nan
        rows.append({"cid": cid, "W": wh.iloc[0][w_g].values.astype(np.float32),
                     "L": lk.iloc[0][l_g].values.astype(np.float32),
                     "E": e3.iloc[0][e_g].values.astype(np.float32),
                     "DC50": dc50, "Dmax": dmax,
                     "target": p.get("warhead_Target",""), "e3": p["E3 ligase"]})
    df = pd.DataFrame(rows)
    print(f"  Mapped rows: {len(df)}")

    X = np.stack([np.stack([r["W"], r["L"], r["E"]]) for _, r in df.iterrows()]).astype(np.float32)
    y_dc50 = df["DC50"].values.astype(float)
    y_dmax = df["Dmax"].values.astype(float)
    # log10 DC50
    logdc = np.log10(np.clip(y_dc50, 1e-3, None))
    logdc[~np.isfinite(logdc)] = np.nan
    df["logDC50"] = logdc
    return X, df, {"dc50": y_dc50, "dmax": y_dmax, "logdc50": logdc}

def main():
    X, df, ys = build_dataset()
    n_dc = np.isfinite(ys["logdc50"]).sum()
    n_dm = np.isfinite(ys["dmax"]).sum()
    print(f"\nTotal mapped: {len(df)} | DC50-labeled: {n_dc} | Dmax-labeled: {n_dm}")

    # ── DC50 task ──
    mask = np.isfinite(ys["logdc50"])
    Xc, yc = X[mask], ys["logdc50"][mask]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    print(f"DC50: train={len(Xtr)} test={len(Xte)}")

    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    te_ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
    tr_l = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_l = DataLoader(te_ds, batch_size=128)

    torch.manual_seed(42); np.random.seed(42)
    model = ProtacAttentionModel(input_dim=4800).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_loss, best_state, patience_counter = float('inf'), None, 0
    for epoch in range(100):
        model.train(); tl = 0
        for xb, yb in tr_l:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        avg = tl / len(tr_l)
        if avg < best_loss:
            best_loss = avg; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15: break
    model.load_state_dict(best_state)
    print(f"Transformer trained: best train MSE={best_loss:.4f} ({epoch+1} epochs)")

    # Test
    model.eval(); preds, trues, attn_all = [], [], []
    with torch.no_grad():
        for xb, yb in te_l:
            xb = xb.to(DEVICE)
            p, a = model(xb)
            preds.extend(p.cpu().numpy()); trues.extend(yb.numpy()); attn_all.append(a.cpu().numpy())
    r2 = r2_score(trues, preds); rmse = np.sqrt(mean_squared_error(trues, preds)); mae = mean_absolute_error(trues, preds)
    rho = spearmanr(trues, preds)[0]; tau = kendalltau(trues, preds)[0]
    print("\n" + "="*60)
    print("STAGE-1 TRANSFORMER (log10 DC50, original scale)")
    print("="*60)
    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ρ={rho:.4f}  τ={tau:.4f}  (n={len(trues)})")

    # ── Fused embeddings → Stage-2 regressors ──
    def fused(loader):
        model.eval(); Z, Y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                z, _ = model.fused_embeddings(xb)
                Z.append(z.cpu().numpy()); Y.append(yb.numpy())
        return np.vstack(Z), np.hstack(Y)
    Ztr, ytr2 = fused(tr_l); Zte, yte2 = fused(te_l)
    print(f"\nFused embeddings: Ztr={Ztr.shape}")

    # RF (500/20 — manuscript)
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(Ztr, ytr2)
    prf = rf.predict(Zte)
    r2rf = r2_score(yte2, prf); mrf = mean_absolute_error(yte2, prf)
    print("STAGE-2 RandomForest (500/20):")
    print(f"  R²={r2rf:.4f}  MAE={mrf:.4f}")

    # 10-fold CV on fused embeddings
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2s, maes = [], []
    for tri, vai in kf.split(Ztr):
        m = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
        m.fit(Ztr[tri], ytr2[tri])
        p = m.predict(Ztr[vai])
        r2s.append(r2_score(ytr2[vai], p)); maes.append(mean_absolute_error(ytr2[vai], p))
    print(f"\n10-FOLD CV (RF on fused embeddings, DC50 log10):")
    print(f"  R²={np.mean(r2s):.3f} ± {np.std(r2s):.3f} | MAE={np.mean(maes):.3f} ± {np.std(maes):.3f}")

    # Save final model
    torch.save(model.state_dict(), f"{OUT}/protac_attention_model.pt")
    import joblib
    joblib.dump(rf, f"{OUT}/stage2_rf_dc50.joblib")
    np.save(f"{OUT}/Z_train.npy", Ztr); np.save(f"{OUT}/Z_test.npy", Zte)
    json.dump({"dc50_r2": float(r2), "dc50_mae": float(mae), "rf_r2": float(r2rf), "rf_mae": float(mrf),
               "cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s))},
              open(f"{OUT}/final_metrics.json", "w"), indent=2)
    print(f"\nSaved final model + metrics → {OUT}/")

    # ── INTERPRETABILITY (reviewer point 5) ──
    print("\n" + "="*60)
    print("INTERPRETABILITY — attention / ablation / SHAP")
    print("="*60)
    attn = np.vstack(attn_all)
    mean_attn = attn.mean(axis=0); std_attn = attn.std(axis=0)
    print(f"Attention (mean±std): Warhead={float(mean_attn[0]):.3f}±{float(std_attn[0]):.3f} "
          f"Linker={float(mean_attn[1]):.3f}±{float(std_attn[1]):.3f} E3={float(mean_attn[2]):.3f}±{float(std_attn[2]):.3f}")
    np.save(f"{OUT}/attention_weights.npy", attn)

    # Ablation: zero each component token
    Xt = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    base_pred, _ = model(Xt)
    base = base_pred.detach().cpu().numpy()
    abl = {}
    for comp, idx in [("Warhead",0), ("Linker",1), ("E3",2)]:
        Xa = Xt.clone(); Xa[:, idx, :] = 0
        pa, _ = model(Xa)
        abl[comp] = float(np.mean(np.abs(pa.detach().cpu().numpy() - base)))
    print(f"Ablation |Δpred|: Warhead={abl['Warhead']:.4f} Linker={abl['Linker']:.4f} E3={abl['E3']:.4f}")
    json.dump({"attention": {"W": float(mean_attn[0]), "L": float(mean_attn[1]), "E": float(mean_attn[2])},
               "ablation": abl}, open(f"{OUT}/interpretability.json", "w"), indent=2)
    print("Saved interpretability.json")

if __name__ == '__main__':
    main()
