#!/usr/bin/env python3
"""
Run with nohup.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/final_results"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, kendalltau, pearsonr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ─── Model (manuscript arch) ───
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
        h = self.proj(x); h = self.transformer(h)
        a = torch.softmax(self.attn_pool(h), dim=1)
        fused = (h * a).sum(dim=1)
        return self.regressor(fused).squeeze(-1), a
    def fused_embeddings(self, x):
        h = self.proj(x); h = self.transformer(h)
        a = torch.softmax(self.attn_pool(h), dim=1)
        return (h * a).sum(dim=1), a

def load_full_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {os.path.basename(path)} → {df.shape}", flush=True)
    return df

def build_dataset():
    protac_df  = load_full_data(f"{DATA}/Upadted_PROTAC_features.csv")
    warhead_df = load_full_data(f"{DATA}/Updated_warhead.csv")
    linker_df  = load_full_data(f"{DATA}/linker_updated_features.csv")
    e3_df      = load_full_data(f"{DATA}/E3_updated_features.csv")
    for df in [protac_df, warhead_df]:
        for c in ["warhead_Uniprot", "warhead_Target"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    protac_df["E3 ligase"] = protac_df["E3 ligase"].astype(str).str.strip().str.upper()
    e3_df["E3_Ligase"] = e3_df["E3_Ligase"].astype(str).str.strip().str.upper()
    linker_df["linker_Compound_ID"] = linker_df["linker_Compound_ID"].astype(str).str.strip().str.upper()
    protac_df["Linker ID"] = protac_df["Linker ID"].astype(str).str.strip().str.upper()
    protac_df["PROTAC_Compound_ID"] = protac_df["PROTAC_Compound_ID"].astype(str).str.strip().str.upper()
    def prefix(df, p): return df.rename(columns={c: f"{p}{c}" for c in df.columns if c.startswith("Grover_")})
    warhead_df = prefix(warhead_df, "W_"); linker_df = prefix(linker_df, "L_"); e3_df = prefix(e3_df, "E_")
    w_g = [c for c in warhead_df.columns if c.startswith("W_Grover_")]
    l_g = [c for c in linker_df.columns if c.startswith("L_Grover_")]
    e_g = [c for c in e3_df.columns if c.startswith("E_Grover_")]

    label_df = protac_df[["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]].copy()
    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"]  = pd.to_numeric(label_df["Dmax (%)"], errors="coerce")
    labels = (label_df.groupby("PROTAC_Compound_ID")
                      .agg({"DC50 (nM)": "median", "Dmax (%)": "max"}).reset_index())
    lab = labels.set_index("PROTAC_Compound_ID")

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
                     "target": str(p.get("warhead_Target","")), "e3": p["E3 ligase"],
                     "smiles": str(p.get("PROTAC_SMILES",""))})
    df = pd.DataFrame(rows)
    X = np.stack([np.stack([r["W"], r["L"], r["E"]]) for _, r in df.iterrows()]).astype(np.float32)
    logdc = np.log10(np.clip(df["DC50"].values, 1e-3, None))
    logdc[~np.isfinite(logdc)] = np.nan
    df["logDC50"] = logdc
    return X, df

# ─── Training (single model, both heads via separate RF stages; transformer shared) ───
def train_transformer(X, y, idx_tr, idx_va, tag):
    tr_ds = TensorDataset(torch.tensor(X[idx_tr], dtype=torch.float32),
                          torch.tensor(y[idx_tr], dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X[idx_va], dtype=torch.float32),
                          torch.tensor(y[idx_va], dtype=torch.float32))
    tr_l = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_l = DataLoader(va_ds, batch_size=128)
    torch.manual_seed(42); np.random.seed(42)
    model = ProtacAttentionModel(input_dim=4800).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, counter = float('inf'), None, 0
    for epoch in range(100):
        model.train(); tl = 0
        for xb, yb in tr_l:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p, _ = model(xb)
            loss = loss_fn(p, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in va_l:
                p, _ = model(xb.to(DEVICE))
                vl += loss_fn(p, yb.to(DEVICE)).item()
        vl /= len(va_l)
        if vl < best_loss:
            best_loss, best_state, counter = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= 15: break
    model.load_state_dict(best_state)
    return model, best_loss

def fused_all(model, X, bs=256):
    model.eval(); Z = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            z, _ = model.fused_embeddings(torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE))
            Z.append(z.cpu().numpy())
    return np.vstack(Z)

def metrics(yt, yp, task):
    m = {}
    m['R2'] = float(r2_score(yt, yp))
    m['RMSE'] = float(np.sqrt(mean_squared_error(yt, yp)))
    m['MAE'] = float(mean_absolute_error(yt, yp))
    m['rho'] = float(spearmanr(yt, yp)[0])
    m['tau'] = float(kendalltau(yt, yp)[0])
    m['pearson'] = float(pearsonr(yt, yp)[0])
    if task == 'dc50':
        d = np.abs(yt - yp)
        m['med_abs_log'] = float(np.median(d))
        m['pct_within_2x'] = float(np.mean(d <= np.log10(2))*100)
        m['pct_within_5x'] = float(np.mean(d <= np.log10(5))*100)
        m['pct_within_10x'] = float(np.mean(d <= 1.0)*100)
    else:
        m['med_abs_err'] = float(np.median(np.abs(yt - yp)))
        m['pct_within_10pct'] = float(np.mean(np.abs(yt-yp) <= 10)*100)
    return m

def evaluate(model, X, y, idx, task):
    model.eval()
    with torch.no_grad():
        p, _ = model(torch.tensor(X[idx], dtype=torch.float32).to(DEVICE))
    return metrics(y[idx], p.cpu().numpy(), task)

def main():
    X, df = build_dataset()
    print(f"Mapped: {len(df)}", flush=True)
    logdc = df['logDC50'].values.astype(float)
    dmax  = df['Dmax'].values.astype(float)
    n_dc = np.isfinite(logdc).sum(); n_dm = np.isfinite(dmax).sum()
    print(f"DC50-labeled: {n_dc} | Dmax-labeled: {n_dm}", flush=True)

    # ── Shared transformer trained on DC50 (primary), used for both endpoints ──
    idx_all = np.arange(len(X))
    tr_i, te_i = train_test_split(idx_all, test_size=0.2, random_state=42)
    tr_i, va_i = train_test_split(tr_i, test_size=0.15/0.8, random_state=42)
    y_tr = logdc.copy(); y_tr[~np.isfinite(y_tr)] = 0  # masked proxy (only finite used via subset below)
    # use only finite for transformer training
    fin = np.isfinite(logdc)
    tr_fin = tr_i[fin[tr_i]]; va_fin = va_i[fin[va_i]]
    model, bv = train_transformer(X, logdc, tr_fin, va_fin, "DC50")
    print(f"Transformer trained (val {bv:.4f})", flush=True)

    # Fused embeddings for all
    Z = fused_all(model, X)
    np.save(f"{OUT}/Z_all.npy", Z)

    # ── Stage-2 RF per endpoint (shared transformer) ──
    results = {}
    for task, y, name, unit in [("dc50", logdc, "DC50", "log10 nM"), ("dmax", dmax, "Dmax", "%")]:
        mask = np.isfinite(y)
        Zf, yf = Z[mask], y[mask]
        idx_f = np.where(mask)[0]
        # align with tr/te indices
        tr_f = tr_i[np.isin(tr_i, idx_f)]; te_f = te_i[np.isin(te_i, idx_f)]
        # fallback: if empty, resplit
        if len(tr_f) < 20 or len(te_f) < 10:
            tr_f, te_f = train_test_split(idx_f, test_size=0.2, random_state=42)
        Ztr, Zte = Z[tr_f], Z[te_f]
        ytr, yte = y[tr_f], y[te_f]
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
        rf.fit(Ztr, ytr)
        prf = rf.predict(Zte)
        m = metrics(yte, prf, task)
        results[task] = {"n_train": len(tr_f), "n_test": len(te_f), **m}
        print(f"\n[{name}] STAGE-2 RF test: R²={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
              f"ρ={m['rho']:.4f} (n={len(te_f)})", flush=True)
        # 10-fold CV
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        r2s, maes = [], []
        for tri, vai in kf.split(Zf):
            mm = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
            mm.fit(Zf[tri], yf[tri])
            p = mm.predict(Zf[vai])
            r2s.append(r2_score(yf[vai], p)); maes.append(mean_absolute_error(yf[vai], p))
        results[task]['cv_r2'] = float(np.mean(r2s)); results[task]['cv_r2_std'] = float(np.std(r2s))
        results[task]['cv_mae'] = float(np.mean(maes)); results[task]['cv_mae_std'] = float(np.std(maes))
        print(f"  10-fold CV: R²={np.mean(r2s):.3f}±{np.std(r2s):.3f} MAE={np.mean(maes):.3f}±{np.std(maes):.3f}", flush=True)
        joblib_rf = rf if False else None
        import joblib
        joblib.dump(rf, f"{OUT}/stage2_rf_{task}.joblib")
        # save predictions for plots
        pd.DataFrame({"true": yte, "pred": prf}).to_csv(f"{OUT}/preds_{task}_test.csv", index=False)

    json.dump(results, open(f"{OUT}/final_metrics_both.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{OUT}/protac_attention_model_both.pt")
    print(f"\nSaved both-endpoint model + metrics → {OUT}/", flush=True)
    print(json.dumps(results, indent=2), flush=True)

if __name__ == '__main__':
    main()
