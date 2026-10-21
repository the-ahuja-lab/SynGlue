#!/usr/bin/env python3
"""Group-disjoint (by Linker ID) evaluation of the linker multiclass task on the ChemBERTa
feature matrix. EXPLICITLY NOT the deployed GROVER model; reported as a separate representation.
No tuning on test folds; fixed hyperparameters (deployed values). Saves OOF predictions + metrics.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix

OUT = "/storage/savi/saveenas/Projects/SynGlue/outputs/synglue_final_revision_evidence"
MR = f"{OUT}/machine_readable"; os.makedirs(MR, exist_ok=True)
P = "/storage/savi/saveenas/Projects/SynGlue/Reproducibility/Savi/PROTAC/PROTAC_featuriser/featuresChemberta.csv"
df = pd.read_csv(P)
label_col = "DC50_label"
y = df[label_col].astype(int).values
groups = df["Linker ID"].astype(str).values
X = df.drop(columns=[c for c in ["Linker ID", label_col, "SMILES"] if c in df.columns]).values.astype(np.float32)
print("rows", X.shape[0], "features", X.shape[1], "classes", np.bincount(y).tolist(),
      "unique linker IDs", len(set(groups)))

clf_params = dict(n_estimators=228, max_depth=7, min_samples_leaf=9, min_samples_split=13,
                  max_features="log2", learning_rate=0.05, random_state=42)
gkf = GroupKFold(n_splits=5)
oof = np.zeros((len(y), 4)); fold_rows = []
for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
    m = GradientBoostingClassifier(**clf_params); m.fit(X[tr], y[tr])
    proba = m.predict_proba(X[te])
    # align classes
    for j, c in enumerate(m.classes_): oof[te, int(c)] = proba[:, j]
    pred = np.argmax(oof[te], axis=1)
    fold_rows.append(dict(fold=k, n_train=len(tr), n_test=len(te),
                          train_groups=len(set(groups[tr])), test_groups=len(set(groups[te])),
                          test_group_overlap=len(set(groups[tr]) & set(groups[te])),
                          accuracy=float((pred == y[te]).mean())))
    print(f"fold {k}: train {len(tr)} / test {len(te)} | groups {len(set(groups[tr]))}/{len(set(groups[te]))} | "
          f"overlap {len(set(groups[tr]) & set(groups[te]))} | acc {(pred==y[te]).mean():.3f}")

pred = np.argmax(oof, axis=1)
metrics = dict(
    representation="ChemBERTa (NOT deployed GROVER)",
    split="5-fold GroupKFold by Linker ID (group-disjoint)",
    n=len(y), n_features=X.shape[1], n_groups=len(set(groups)),
    class_counts=np.bincount(y).tolist(),
    auroc_macro_ovr=float(roc_auc_score(y, oof, multi_class="ovr", average="macro")),
    auroc_weighted_ovr=float(roc_auc_score(y, oof, multi_class="ovr", average="weighted")),
    macro_f1=float(f1_score(y, pred, average="macro")),
    balanced_accuracy=float(balanced_accuracy_score(y, pred)),
    mcc=float(matthews_corrcoef(y, pred)),
    per_class_sensitivity={str(c): float(((pred == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(4)},
    per_class_specificity={str(c): float(((pred != c) & (y != c)).sum() / max((y != c).sum(), 1)) for c in range(4)},
    confusion_matrix=confusion_matrix(y, pred).tolist(),
    folds=fold_rows,
    protocol_note="No hyperparameter tuning inside folds; deployed hyperparameters used. Group-disjoint by Linker ID. ChemBERTa representation, not GROVER.")
json.dump(metrics, open(f"{MR}/linker_groupdisjoint_metrics.json", "w"), indent=2)
pd.DataFrame(oof, columns=[f"p_class{c}" for c in range(4)]).assign(
    true=y, pred=pred, linker_id=groups).to_csv(f"{MR}/linker_groupdisjoint_oof.csv", index=False)
pd.DataFrame(fold_rows).to_csv(f"{MR}/linker_groupdisjoint_folds.csv", index=False)
print(json.dumps({k: v for k, v in metrics.items() if k not in ("folds","confusion_matrix")}, indent=2))
