#!/usr/bin/env python3
"""08_train_linker_classifier.py  [P1]

Linker multiclass classifier: document the original training and perform a valid grouped evaluation.

Two modes:

  --mode original   Reproduce the original protocol exactly for documentation purposes:
                    GradientBoostingClassifier(random_state=42) on the available feature matrix,
                    train_test_split(test_size=0.2, random_state=42, stratify=y),
                    RandomizedSearchCV(cv=5, scoring='roc_auc_ovr') on the training partition,
                    and selection by **test** AUC (as in the deposited code, line 204).
                    This deliberately reproduces the test-selected protocol so its bias is visible.

  --mode grouped    Valid grouped evaluation: GroupKFold by canonical linker identity, train-only
                    preprocessing, fixed or nested hyperparameters, out-of-fold predictions and
                    full metrics (macro OVR AUROC, macro-F1, balanced accuracy, MCC, confusion
                    matrix, per-class sensitivity/specificity, fold membership).

Note on features: the deployed model uses 4,800-dimensional GROVER features. If only the
ChemBERTa matrix is present (384 features), the grouped evaluation runs on ChemBERTa and the
output is labelled accordingly. The deployed GROVER model cannot be evaluated grouped until its
feature matrix is deposited.

Outputs (results/)
  linker_original_metrics.json      (mode=original) protocol, per-model test AUC, selected model
  linker_grouped_metrics.json       (mode=grouped) metrics + folds
  linker_grouped_oof.csv            (mode=grouped) out-of-fold probabilities, true labels, groups
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_features(path):
    df = pd.read_csv(C.require(path, "linker feature matrix"))
    label_col = "DC50_label" if "DC50_label" in df.columns else \
        next(c for c in df.columns if "label" in c.lower())
    group_col = next((c for c in df.columns if c.strip().lower() in ("linker id", "linker_id")), None)
    drop = [c for c in [label_col, group_col, "SMILES"] if c and c in df.columns]
    X = df.drop(columns=drop).values.astype(np.float32)
    y = df[label_col].astype(int).values
    groups = df[group_col].astype(str).values if group_col else np.arange(len(y)).astype(str)
    return X, y, groups, df, label_col, group_col


def mode_original(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=C.SEED, stratify=y)
    models = {
        "GradientBoosting": GradientBoostingClassifier(random_state=C.SEED),
        "RandomForest": RandomForestClassifier(random_state=C.SEED),
        "ExtraTrees": ExtraTreesClassifier(random_state=C.SEED),
    }
    grid = dict(classifier__n_estimators=[100, 228, 300], classifier__max_depth=[3, 5, 7],
                classifier__learning_rate=[0.05, 0.1], classifier__min_samples_leaf=[1, 9],
                classifier__min_samples_split=[2, 13], classifier__max_features=["log2", None])
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")),
                         ("scaler", StandardScaler()), ("classifier", clf)])
        if name == "GradientBoosting":
            search = RandomizedSearchCV(pipe, grid, n_iter=20, cv=5, scoring="roc_auc_ovr",
                                        random_state=C.SEED, n_jobs=-1)
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            params = {k: str(v) for k, v in search.best_params_.items()}
        else:
            best = pipe.fit(Xtr, ytr)
            params = {}
        auc = roc_auc_score(yte, best.predict_proba(Xte), multi_class="ovr", average="weighted")
        out[name] = dict(test_auc=float(auc), params=params)
        print(f"  [{name}] test AUC = {auc:.4f}")
    best_name = max(out, key=lambda k: out[k]["test_auc"])
    out["selected_by_test_auc"] = best_name
    out["protocol_note"] = ("reproduces the deposited protocol: model selection by TEST AUC "
                            "(05_Models_multi_class_linker.py:204); the reported value is therefore "
                            "an optimistically biased, test-selected estimate")
    return out


def mode_grouped(X, y, groups):
    clf = GradientBoostingClassifier(n_estimators=228, max_depth=7, min_samples_leaf=9,
                                     min_samples_split=13, max_features="log2", learning_rate=0.05,
                                     random_state=C.SEED)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(np.unique(y))))
    folds = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")),
                         ("scaler", StandardScaler()), ("classifier", clf)])
        pipe.fit(X[tr], y[tr])                      # preprocessing fitted on train only
        proba = pipe.predict_proba(X[te])
        for j, c in enumerate(pipe.named_steps["classifier"].classes_):
            oof[te, int(c)] = proba[:, j]
        pred = np.argmax(oof[te], axis=1)
        folds.append(dict(fold=k, n_train=int(len(tr)), n_test=int(len(te)),
                          train_groups=len(set(groups[tr])), test_groups=len(set(groups[te])),
                          group_overlap=len(set(groups[tr]) & set(groups[te])),
                          accuracy=float((pred == y[te]).mean())))
    pred = np.argmax(oof, axis=1)
    classes = list(range(oof.shape[1]))
    met = dict(protocol="GroupKFold by linker identity, train-only preprocessing, fixed hyperparameters",
               n=int(len(y)), n_features=int(X.shape[1]), n_groups=len(set(groups)),
               class_counts=np.bincount(y).tolist(),
               auroc_macro_ovr=float(roc_auc_score(y, oof, multi_class="ovr", average="macro")),
               auroc_weighted_ovr=float(roc_auc_score(y, oof, multi_class="ovr", average="weighted")),
               macro_f1=float(f1_score(y, pred, average="macro")),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               mcc=float(matthews_corrcoef(y, pred)),
               per_class_sensitivity={str(c): float(((pred == c) & (y == c)).sum() /
                                                    max((y == c).sum(), 1)) for c in classes},
               per_class_specificity={str(c): float(((pred != c) & (y != c)).sum() /
                                                    max((y != c).sum(), 1)) for c in classes},
               confusion_matrix=confusion_matrix(y, pred).tolist(), folds=folds)
    pd.DataFrame(oof, columns=[f"p_class{c}" for c in classes]).assign(
        true=y, pred=pred, group=groups).to_csv(os.path.join(C.RESULTS, "linker_grouped_oof.csv"),
                                                index=False)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "grouped"], default="grouped")
    ap.add_argument("--features", default=C.PATHS["linker_features_chemberta"])
    args = ap.parse_args()
    X, y, groups, df, label_col, group_col = load_features(args.features)
    rep = "ChemBERTa" if X.shape[1] < 1000 else "GROVER"
    print(f"features: {X.shape} | representation: {rep} | classes: {np.bincount(y).tolist()} "
          f"| unique linker ids: {len(set(groups))}")
    if args.mode == "original":
        out = mode_original(X, y)
        out["representation"] = rep
        C.save_json(out, "linker_original_metrics.json")
    else:
        met = mode_grouped(X, y, groups)
        met["representation"] = rep
        met["caveat"] = (f"representation = {rep}; the deployed model uses 4,800-d GROVER features. "
                         "This evaluation is not a substitute for a GROVER grouped evaluation.")
        C.save_json(met, "linker_grouped_metrics.json")
        print(json.dumps({k: v for k, v in met.items() if k not in ("folds", "confusion_matrix")},
                         indent=2))


if __name__ == "__main__":
    main()
