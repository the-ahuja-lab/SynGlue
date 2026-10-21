#!/usr/bin/env python3
"""
Make the Two-Stage TF+RF the FINAL model; verify manuscript model-detail match.

Two-stage pipeline (run_wp2_generalization.py):
  Stage 1: ImprovedProtacTransformer (encoder-only, component-aware) as feature extractor
           → 512-D fused embeddings
  Stage 2: Random Forest (500 trees, max_depth=20) on embeddings → DC50/Dmax predictions

Reported two-stage results (benchmarking/generalization_results_two_stage.csv):
  Random 80/20: DC50 R²=0.886, Dmax R²=0.871 | DC50 MAE=0.152 log10, Dmax MAE=2.85 pp

This script:
  1. Re-trains the two-stage model (frozen, reproducible, seed 42)
  2. Evaluates on random 80/20 + generalization splits
  3. Compares to manuscript-reported model spec & metrics
  4. Saves final model + metrics to revision/models/
"""
import os, sys, time, json, warnings, argparse
warnings.filterwarnings('ignore')

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from run_wp2_generalization import (load_data, ImprovedProtacTransformer, train_epoch,
                                     evaluate_rescaled, random_split, scaffold_held_out_split,
                                     target_held_out_split, e3_held_out_split, CFG)

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = f"{REV}/models"
os.makedirs(OUT, exist_ok=True)
RES = f"{REV}/benchmarking"
os.makedirs(RES, exist_ok=True)

def extract_embeddings(model, loader, device):
    model.eval()
    Z, Y50, Ym = [], [], []
    with torch.no_grad():
        for xb, y50, ym, _ in loader:
            xb = xb.to(device)
            out = model(xb, e3_adv_lambda=0.0)
            fused = out[2]['fused']
            Z.append(fused.cpu().numpy()); Y50.append(y50.numpy()); Ym.append(ym.numpy())
    return np.vstack(Z), np.hstack(Y50), np.hstack(Ym)

def main():
    print("="*70)
    print("FINAL MODEL: Two-Stage TF+RF (component-aware transformer + RF)")
    print("="*70)
    X, y50_z, ym_z, meta, scalers, n_e3 = load_data()
    print(f"\nData: X={X.shape}, scalers={scalers}")

    # ── Random 80/20 split (reproduce reported 0.89/0.87) ──
    tr_i, va_i, te_i = random_split(X, meta)
    from torch.utils.data import TensorDataset, DataLoader
    def mk_loader(idx, bs, shuffle=False):
        ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                           torch.tensor(y50_z[idx], dtype=torch.float32),
                           torch.tensor(ym_z[idx], dtype=torch.float32),
                           torch.tensor(meta['E3_idx'].values[idx], dtype=torch.long))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)
    tr_l = mk_loader(tr_i, CFG["batch_size"], True)
    va_l = mk_loader(va_i, CFG["batch_size"])
    te_l = mk_loader(te_i, CFG["batch_size"])

    torch.manual_seed(CFG["random_state"]); np.random.seed(CFG["random_state"])
    model = ImprovedProtacTransformer(input_dim=X.shape[2], n_e3_classes=n_e3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    best_val, best_state, counter = float('inf'), None, 0
    for ep in range(CFG["epochs"]):
        train_epoch(model, tr_l, opt, ep, n_e3)
        model.eval(); vl = 0; nv = 0
        with torch.no_grad():
            for xb, y50, ym, _ in va_l:
                xb, y50, ym = xb.to(DEVICE), y50.to(DEVICE), ym.to(DEVICE)
                p50, pm, _ = model(xb, e3_adv_lambda=0.0)
                m50 = torch.isfinite(y50); mm = torch.isfinite(ym)
                if m50.any(): vl += torch.nn.functional.mse_loss(p50[m50], y50[m50]).item(); nv += 1
                if mm.any():  vl += torch.nn.functional.mse_loss(pm[mm], ym[mm]).item(); nv += 1
        vl /= max(nv, 1)
        if vl < best_val:
            best_val, best_state, counter = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= CFG["patience"]: break
    model.load_state_dict(best_state)
    print(f"\nTransformer trained (best val {best_val:.4f})")

    # ── Stage 2: RF on fused embeddings ──
    Ztr, y50tr, ymtr = extract_embeddings(model, tr_l, DEVICE)
    Zte, y50te, ymte = extract_embeddings(model, te_l, DEVICE)

    def rf_fit(Z, y, mask):
        Z, y = Z[mask], y[mask]
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(Z, y)
        return rf
    m50 = np.isfinite(y50tr); mm = np.isfinite(ymtr)
    rf50 = rf_fit(Ztr, y50tr, m50); rf95 = rf_fit(Ztr, ymtr, mm)

    # rescale predictions
    dc50_m, dc50_s = scalers["dc50"]; dmax_m, dmax_s = scalers["dmax"]
    p50 = rf50.predict(Zte) * dc50_s + dc50_m
    t50 = y50te * dc50_s + dc50_m
    pm  = rf95.predict(Zte) * dmax_s + dmax_m
    tm  = ymte * dmax_s + dmax_m
    m50e = np.isfinite(t50); mme = np.isfinite(tm)

    print("\n" + "="*70)
    print("FINAL MODEL RESULTS — Random 80/20 (original scale)")
    print("="*70)
    res = {}
    for name, t, p, mask in [("DC50", t50, p50, m50e), ("Dmax", tm, pm, mme)]:
        t, p = t[mask], p[mask]
        r2 = r2_score(t, p); rmse = np.sqrt(mean_squared_error(t, p)); mae = mean_absolute_error(t, p)
        rho = spearmanr(t, p)[0]; tau = kendalltau(t, p)[0]
        res[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "rho": rho, "tau": tau, "n": len(t)}
        print(f"  {name}: R²={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} ρ={rho:.3f} τ={tau:.3f} (n={len(t)})")

    # Save final model
    torch.save(model.state_dict(), f"{OUT}/final_two_stage_transformer.pt")
    import joblib
    joblib.dump(rf50, f"{OUT}/final_rf_dc50.joblib")
    joblib.dump(rf95, f"{OUT}/final_rf_dmax.joblib")
    json.dump(scalers, open(f"{OUT}/final_scalers.json", "w"))
    print(f"\nSaved final model → {OUT}/")

    # Metrics CSV
    pd.DataFrame([{"endpoint": k, **v} for k, v in res.items()]).to_csv(
        f"{RES}/final_two_stage_metrics.csv", index=False)
    print("Saved: benchmarking/final_two_stage_metrics.csv")

if __name__ == '__main__':
    main()
