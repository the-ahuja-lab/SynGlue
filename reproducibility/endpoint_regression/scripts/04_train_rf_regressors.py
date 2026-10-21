#!/usr/bin/env python3
"""04_train_rf_regressors.py  [P0]

Fit the DC50 and Dmax Random Forests from their exact training data.

Protocol (as used in the compound-level pipeline that produced the reported split metrics):
    input  = fused 512-d embeddings (Z_train / Z_test from 03_extract_embeddings.py)
    target = DC50 as log10(nM); Dmax as %
    model  = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    split  = the saved split indices (four_split_indices.npz) when available

Because the released joblib artifacts were serialised with older scikit-learn versions, this step
**trains fresh models** rather than loading them, so it works in any modern environment.

Outputs (results/)
  rf_dc50_trained.joblib / rf_dmax_trained.joblib
  rf_training_report.json   parameters, n_train/n_test, train/test metrics, artifact hashes
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", default=os.path.join(C.RESULTS, "Z_all.npy"))
    ap.add_argument("--labels", default=None,
                    help="CSV with columns compound_id,DC50,Dmax in row order of Z "
                         "(default: rebuilt from the curated table via the mapped order)")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    Z = np.load(C.require(args.Z, "fused embeddings (run 03_extract_embeddings.py)"))
    print("Z:", Z.shape)

    if args.labels and os.path.exists(args.labels):
        lab = __import__("pandas").read_csv(args.labels)
        y50 = lab["DC50"].values.astype(float)
        ydm = lab["Dmax"].values.astype(float)
    else:
        # Rebuild labels in the same row order as the mapped table produced by 01/06.
        import pandas as pd
        mapped = os.path.join(C.RESULTS, "mapped_compound_order.csv")
        if not os.path.exists(mapped):
            raise SystemExit("run 01_prepare_components.py (or 06_check_data_leakage.py) first, "
                             "or pass --labels with a compound_id,DC50,Dmax CSV")
        m = pd.read_csv(mapped)
        tbl = pd.read_csv(C.require(C.PATHS["training_table"]), low_memory=False)
        agg = tbl.groupby("PROTAC_Compound_ID").agg(
            dc50=("DC50 (nM)", lambda s: __import__("pandas").to_numeric(
                s.astype(str).str.replace(r"[<>~]", "", regex=True), errors="coerce").median()),
            dmax=("Dmax (%)", lambda s: __import__("pandas").to_numeric(
                s.astype(str).str.replace(r"[<>~]", "", regex=True), errors="coerce").max()))
        m = m.merge(agg, left_on="compound_id", right_index=True, how="left")
        y50 = np.log10(m["dc50"].values.astype(float))
        ydm = m["dmax"].values.astype(float)
    assert len(y50) == len(Z), f"label/Z length mismatch: {len(y50)} vs {len(Z)}"

    rng = np.random.RandomState(C.SEED)
    n_test = int(args.test_frac * len(Z))
    idx = rng.permutation(len(Z))
    te, tr = idx[:n_test], idx[n_test:]
    if os.path.exists(C.PATHS["split_indices"]):
        z = np.load(C.PATHS["split_indices"], allow_pickle=True)
        if "tr_Random" in z and "te_Random" in z and len(z["tr_Random"]) + len(z["te_Random"]) == len(Z):
            tr, te = z["tr_Random"], z["te_Random"]
            print("using saved split indices (tr_Random / te_Random)")

    report = {}
    for tag, y in [("dc50", y50), ("dmax", ydm)]:
        ok = np.isfinite(y)
        tr_ok = np.array([i for i in tr if ok[i]])
        te_ok = np.array([i for i in te if ok[i]])
        if len(tr_ok) < 50:
            print(f"  [{tag}] insufficient labelled rows ({len(tr_ok)}); skipping")
            continue
        rf = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                   random_state=C.SEED, n_jobs=-1)
        rf.fit(Z[tr_ok], y[tr_ok])
        pred = rf.predict(Z[te_ok])
        m = dict(n_train=int(len(tr_ok)), n_test=int(len(te_ok)),
                 R2=float(r2_score(y[te_ok], pred)),
                 RMSE=float(mean_squared_error(y[te_ok], pred) ** 0.5),
                 MAE=float(mean_absolute_error(y[te_ok], pred)))
        report[tag] = m
        print(f"  [{tag}] n_train={m['n_train']} n_test={m['n_test']} "
              f"R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}")
        out = os.path.join(C.RESULTS, f"rf_{tag}_trained.joblib")
        import joblib
        joblib.dump(rf, out)
        m["artifact"] = out
        m["sha256"] = C.sha256(out)
        np.save(os.path.join(C.RESULTS, f"pred_{tag}_test.npy"), pred)

    report["parameters"] = dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                random_state=C.SEED, features=int(Z.shape[1]))
    C.save_json(report, "rf_training_report.json")


if __name__ == "__main__":
    main()
