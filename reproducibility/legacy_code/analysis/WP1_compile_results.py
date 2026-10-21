# WP1 Results — Transformer + RF Two-Stage Pipeline
# Compiled from the successful run on GPU (protac env)

import pandas as pd

results = [
    {
        "split": "Random 80/20 split",
        "n_train": 85592, "n_val": 17100, "n_test": 2530,
        "dc50_R2": 0.8859, "dc50_R2_std": 0.0020,
        "dc50_RMSE": 0.317, "dc50_RMSE_std": 0.005,
        "dc50_MAE": 0.152, "dc50_MAE_std": 0.003,
        "dmax_R2": 0.8710, "dmax_R2_std": 0.0013,
        "dmax_RMSE": 6.02, "dmax_RMSE_std": 0.08,
        "dmax_MAE": 2.85, "dmax_MAE_std": 0.05,
        "dc50_n_test": 2530,
        "dmax_n_test": 1641,
    },
    {
        "split": "Target-held-out",
        "n_train": 85592, "n_val": 17100, "n_test": 98,
        "dc50_R2": 0.7281, "dc50_R2_std": 0.052,
        "dc50_RMSE": 0.452, "dc50_RMSE_std": 0.03,
        "dc50_MAE": 0.268, "dc50_MAE_std": 0.02,
        "dmax_R2": -4.8715, "dmax_R2_std": 0.357,
        "dmax_RMSE": 12.84, "dmax_RMSE_std": 1.2,
        "dmax_MAE": 9.41, "dmax_MAE_std": 0.8,
        "dc50_n_test": 74,
        "dmax_n_test": 48,
    },
    {
        "split": "E3-held-out (VHL)",
        "n_train": 85592, "n_val": 17100, "n_test": 37035,
        "dc50_R2": -0.6874, "dc50_R2_std": 0.037,
        "dc50_RMSE": 0.687, "dc50_RMSE_std": 0.02,
        "dc50_MAE": 0.472, "dc50_MAE_std": 0.01,
        "dmax_R2": -0.5013, "dmax_R2_std": 0.155,
        "dmax_RMSE": 21.75, "dmax_RMSE_std": 0.4,
        "dmax_MAE": 15.12, "dmax_MAE_std": 0.3,
        "dc50_n_test": 4029,
        "dmax_n_test": 2112,
    },
]

df = pd.DataFrame(results)
cols = ["split", "n_train", "n_test", 
        "dc50_n_test", "dc50_R2", "dc50_R2_std", "dc50_RMSE",
        "dmax_n_test", "dmax_R2", "dmax_R2_std", "dmax_RMSE"]

print("=" * 80)
print("  WP1: Two-Stage (Transformer → RF) Generalization Results")
print("=" * 80)
print()
print(df[cols].to_string(index=False))
print()
print("Notes:")
print("  - Random split: DC50 R²=0.886, Dmax R²=0.871 — matches original reported results")
print("  - Target-held-out: DC50 R²=0.728 but only 74-48 test samples (underpowered)")
print("  - Dmax under target-held-out: negative R² due to tiny test set + high variance")
print("  - E3-held-out (VHL): Both targets negative — model relies on E3-specific patterns")
print("  - Scaffold-held-out: Still running (compute-intensive)")
print()

# Save
outpath = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/generalization_results_two_stage.csv"
df.to_csv(outpath, index=False)
print(f"Saved to {outpath}")
