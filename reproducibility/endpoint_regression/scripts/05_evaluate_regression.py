#!/usr/bin/env python3
"""05_evaluate_regression.py  [P0]

Recalculate random, scaffold and target DC50/Dmax R² (plus RMSE, MAE, Spearman, Pearson) from
predictions.

Preferred path: use the saved per-split metrics table
(`revision/models/final_protac_model/four_split_generalization.csv`) and the saved test predictions
(`preds_dc50_test.csv`, `preds_dmax_test.csv`) to recompute the headline numbers.

Alternative path (--from-artifacts): recompute predictions from Z + the trained RFs
(results/rf_*_trained.joblib from step 04) for each saved split in four_split_indices.npz.

Outputs (results/)
  regression_metrics.csv        split, endpoint, n, R2, RMSE, MAE, rho, tau, pearson
  regression_metrics_saved.csv  copy of the recovered table for direct comparison
"""
import argparse
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

SPLITS = ["Random", "Scaffold-held", "Target-held", "E3-held (VHL)"]


def metrics(y_true, y_pred):
    return dict(n=int(len(y_true)), R2=float(r2_score(y_true, y_pred)),
                RMSE=float(mean_squared_error(y_true, y_pred) ** 0.5),
                MAE=float(mean_absolute_error(y_true, y_pred)),
                rho=float(spearmanr(y_true, y_pred).statistic),
                tau=float(kendalltau(y_true, y_pred).statistic),
                pearson=float(pearsonr(y_true, y_pred)[0]))


def from_saved():
    C.require(C.PATHS["split_metrics"], "saved split metrics")
    df = pd.read_csv(C.PATHS["split_metrics"])
    shutil.copy2(C.PATHS["split_metrics"], os.path.join(C.RESULTS, "regression_metrics_saved.csv"))
    print(df.to_string(index=False))
    return df


def from_artifacts():
    import joblib
    Z = np.load(C.require(C.PATHS["Z_all"], "Z_all"))
    z = np.load(C.require(C.PATHS["split_indices"], "split indices"), allow_pickle=True)
    mapped = os.path.join(C.RESULTS, "mapped_compound_order.csv")
    if not os.path.exists(mapped):
        raise SystemExit("run 01/06 first to produce results/mapped_compound_order.csv")
    m = pd.read_csv(mapped)
    tbl = pd.read_csv(C.require(C.PATHS["training_table"]), low_memory=False)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[<>~]", "", regex=True), errors="coerce")
    agg = tbl.groupby("PROTAC_Compound_ID").agg(
        dc50=("DC50 (nM)", lambda s: num(s).median()),
        dmax=("Dmax (%)", lambda s: num(s).max()))
    m = m.merge(agg, left_on="compound_id", right_index=True, how="left")
    y50 = np.log10(m["dc50"].values.astype(float))
    ydm = m["dmax"].values.astype(float)

    rows = []
    for split in SPLITS:
        if f"te_{split}" not in z:
            continue
        te = z[f"te_{split}"]
        for tag, y in [("DC50", y50), ("Dmax", ydm)]:
            rf_path = os.path.join(C.RESULTS, f"rf_{'dc50' if tag == 'DC50' else 'dmax'}_trained.joblib")
            if not os.path.exists(rf_path):
                print(f"  [{split}/{tag}] no trained RF; run 04_train_rf_regressors.py")
                continue
            rf = joblib.load(rf_path)
            ok = np.array([i for i in te if np.isfinite(y[i])])
            if len(ok) < 10:
                continue
            pred = rf.predict(Z[ok])
            rows.append(dict(split=split, endpoint=tag, **metrics(y[ok], pred)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-artifacts", action="store_true",
                    help="recompute from Z + trained RFs instead of using the saved metrics table")
    args = ap.parse_args()
    df = from_artifacts() if args.from_artifacts else from_saved()
    C.save_csv(df, "regression_metrics.csv")
    print("\nreported vs saved comparison (DC50 R2):")
    for s in SPLITS:
        row = df[(df["split"] == s) & (df["endpoint"] == "DC50")]
        if len(row):
            print(f"  {s:16s} R2={row.iloc[0]['R2']:.4f} n={int(row.iloc[0]['n'])}")


if __name__ == "__main__":
    main()
