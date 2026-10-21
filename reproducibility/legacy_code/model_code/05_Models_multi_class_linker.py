import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc, confusion_matrix
)

from joblib import dump

# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "features_chemberta.csv"
OUTPUT_DIR = "model_outputs"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================================
# DATA LOADING
# =========================================================

def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.rename(columns={'DC50_label': 'Labels'})
    df = df.drop(columns=['Linker ID', 'SMILES'], errors='ignore')
    df = df.dropna(subset=['Labels'])

    X = df.drop('Labels', axis=1)
    y = df['Labels']

    X = X.dropna(axis=1, how='all')

    return X, y


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=500, multi_class='multinomial'),
        "SVM": OneVsRestClassifier(SVC(probability=True)),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "ExtraTrees": ExtraTreesClassifier(random_state=42),
        "SGD": OneVsRestClassifier(SGDClassifier(random_state=42)),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
        "MLP": MLPClassifier(max_iter=500, random_state=42)
    }


# =========================================================
# PIPELINE BUILDER
# =========================================================

def build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])


# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_train, y_test, train_pred, test_pred, train_proba, test_proba):
    return {
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Train Precision": precision_score(y_train, train_pred, average="macro", zero_division=0),
        "Test Precision": precision_score(y_test, test_pred, average="macro", zero_division=0),
        "Train Recall": recall_score(y_train, train_pred, average="macro", zero_division=0),
        "Test Recall": recall_score(y_test, test_pred, average="macro", zero_division=0),
        "Train F1": f1_score(y_train, train_pred, average="macro", zero_division=0),
        "Test F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Train Kappa": cohen_kappa_score(y_train, train_pred),
        "Test Kappa": cohen_kappa_score(y_test, test_pred),
        "Train MCC": matthews_corrcoef(y_train, train_pred),
        "Test MCC": matthews_corrcoef(y_test, test_pred),
        "Train AUC": roc_auc_score(y_train, train_proba, multi_class="ovr", average="weighted"),
        "Test AUC": roc_auc_score(y_test, test_proba, multi_class="ovr", average="weighted")
    }


# =========================================================
# PLOTTING
# =========================================================

def plot_roc(y_test, test_proba, model_name, output_dir):
    plt.figure()

    for i in range(test_proba.shape[1]):
        y_bin = (y_test == i).astype(int)

        if np.sum(y_bin) == 0:
            continue

        fpr, tpr, _ = roc_curve(y_bin, test_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc.png"))
    plt.close()


def plot_confusion(y_test, test_pred, model_name, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(output_dir, f"{model_name}_cm.png"))
    plt.close()


# =========================================================
# MAIN TRAINING LOOP
# =========================================================

def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = get_models()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        try:
            pipeline = build_pipeline(model)
            pipeline.fit(X_train, y_train)

            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)

            train_proba = pipeline.predict_proba(X_train)
            test_proba = pipeline.predict_proba(X_test)

            # Save model
            dump(pipeline, os.path.join(OUTPUT_DIR, f"{name}.joblib"))

            # Save predictions
            pd.DataFrame(test_proba).assign(
                Actual=y_test,
                Predicted=test_pred
            ).to_csv(os.path.join(OUTPUT_DIR, f"{name}_predictions.csv"), index=False)

            # Metrics
            metric = compute_metrics(
                y_train, y_test, train_pred, test_pred, train_proba, test_proba
            )
            metric["Model"] = name
            all_metrics.append(metric)

            # Plots
            plot_roc(y_test, test_proba, name, OUTPUT_DIR)
            plot_confusion(y_test, test_pred, name, OUTPUT_DIR)

            print(f"✔ {name} completed | Test AUC: {metric['Test AUC']:.3f}")

        except Exception as e:
            print(f"✖ {name} failed: {e}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    best = metrics_df.loc[metrics_df["Test AUC"].idxmax()]
    print("\n🏆 Best Model:")
    print(best)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    X, y = load_data(INPUT_FILE)
    train_models(X, y)