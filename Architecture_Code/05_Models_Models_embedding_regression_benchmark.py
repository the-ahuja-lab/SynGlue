import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from joblib import dump

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = "05_Models/embedding_models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# EMBEDDING EXTRACTION
# =========================================================

def extract_fused_embeddings(model, loader):
    """
    Extract fused embeddings from trained attention model.
    Output:
        Z → (N, 512)
        y → (N,)
    """

    model.eval()

    Z, Y = [], []

    with torch.no_grad():
        for xb, yb in loader:

            xb = xb.to(DEVICE)

            # ---- Forward (explicit stages) ----
            h = model.proj(xb)                  # (B, 3, 512)
            h = model.transformer(h)            # (B, 3, 512)

            attn_scores = model.attn_pool(h)    # (B, 3, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)

            fused = (h * attn_weights).sum(dim=1)  # (B, 512)

            Z.append(fused.cpu().numpy())
            Y.append(yb.numpy())

    return np.vstack(Z), np.hstack(Y)

# =========================================================
# REGRESSOR DEFINITIONS
# =========================================================

def get_regressors():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=20, n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-4),
        "ElasticNet": ElasticNet(alpha=1e-4),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(max_depth=20, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
    }

# =========================================================
# METRIC FUNCTION
# =========================================================

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred)
    }

# =========================================================
# TRAIN + EVALUATE
# =========================================================

def run_regression_benchmark(Z_train, y_train, Z_test, y_test):

    regressors = get_regressors()
    results = []

    for name, model in regressors.items():

        print(f"\nTraining {name}...")

        model.fit(Z_train, y_train)

        pred_train = model.predict(Z_train)
        pred_test = model.predict(Z_test)

        train_metrics = compute_metrics(y_train, pred_train)
        test_metrics = compute_metrics(y_test, pred_test)

        results.append({
            "Model": name,
            **{f"{k}_Train": v for k, v in train_metrics.items()},
            **{f"{k}_Test": v for k, v in test_metrics.items()}
        })

        # Save model
        dump(model, os.path.join(OUTPUT_DIR, f"{name}.joblib"))

        print(f"✔ {name} | Test R2: {test_metrics['R2']:.3f}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("R2_Test", ascending=False)

    df_results.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)

    print("\n🏆 Best Model:")
    print(df_results.iloc[0])

    return df_results

# =========================================================
# MAIN PIPELINE
# =========================================================

def main(model, train_loader, test_loader):

    print("\nExtracting embeddings...")

    Z_train, y_train = extract_fused_embeddings(model, train_loader)
    Z_test, y_test = extract_fused_embeddings(model, test_loader)

    print("Z_train:", Z_train.shape)
    print("Z_test :", Z_test.shape)

    results = run_regression_benchmark(Z_train, y_train, Z_test, y_test)

    return results

# =========================================================
# ENTRY (EXAMPLE)
# =========================================================

if __name__ == "__main__":
    print("⚠ Run via main(model, train_loader, test_loader)")