import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from joblib import dump
import os

# =========================================================
# CONFIG
# =========================================================

N_SPLITS = 10
RANDOM_STATE = 42
OUTPUT_DIR = "05_Models/cv_results/"
SAVE_MODELS = False   # set True if you want fold-wise models

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# MODEL CONFIG
# =========================================================

def get_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# =========================================================
# CROSS VALIDATION
# =========================================================

def run_cv(X, y):

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []

    print(f"\nRunning {N_SPLITS}-fold CV...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = get_model()

        model.fit(X_tr, y_tr)

        pred = model.predict(X_val)

        metrics = compute_metrics(y_val, pred)

        results.append({
            "Fold": fold,
            **metrics
        })

        print(
            f"Fold {fold:02d} | "
            f"R2: {metrics['R2']:.3f} | "
            f"RMSE: {metrics['RMSE']:.3f} | "
            f"MAE: {metrics['MAE']:.3f}"
        )

        # Optional: save model
        if SAVE_MODELS:
            dump(model, os.path.join(OUTPUT_DIR, f"xgb_fold_{fold}.joblib"))

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    summary = {
        "R2_mean": df["R2"].mean(),
        "R2_std": df["R2"].std(),
        "RMSE_mean": df["RMSE"].mean(),
        "RMSE_std": df["RMSE"].std(),
        "MAE_mean": df["MAE"].mean(),
        "MAE_std": df["MAE"].std(),
    }

    print("\n=== CV Summary ===")
    print(f"R2   : {summary['R2_mean']:.3f} ± {summary['R2_std']:.3f}")
    print(f"RMSE : {summary['RMSE_mean']:.3f} ± {summary['RMSE_std']:.3f}")
    print(f"MAE  : {summary['MAE_mean']:.3f} ± {summary['MAE_std']:.3f}")

    # Save results
    df.to_csv(os.path.join(OUTPUT_DIR, "fold_results.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    return df, summary

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    print("⚠ Provide X, y and call run_cv(X, y)")