#!/usr/bin/env python3
"""07_evaluate_external.py  [P0]

External benchmark: novelty filtering, external AUROC, and comparator-specific denominators.

Inputs
  final_predictions.csv   per-compound predictions from the comparator runs
                          (SynGlue whole / 3-token, PDP, PDP-XGB, PROTAC-STAN, Morgan-RF, Tanimoto-NN)
  final_external_tack_chiral.csv   external funnel with max-Tanimoto-to-training per compound

What it does
  1. Reproduces the external funnel counts (2,562 exact-free; 2,484 / 2,431 / 2,256 at Tanimoto
     0.90 / 0.85 / 0.70).
  2. For every comparator, computes AUROC at DC50 <= 10 nM / 100 nM / 1 uM, using the a-priori
     orientation (potency score = -log10(predicted DC50 nM); classifiers use P(active)),
     restricted to non-censored DC50 labels.
  3. Reports the valid-prediction count per comparator and the eligibility-only intersection
     (compounds with a prediction from every comparator) — explicitly NOT a common cohort.
  4. Reports the leakage rule applied per comparator where known.

Outputs (results/)
  external_funnel_counts.csv
  comparator_metrics.csv          per comparator x threshold: n, n_pos, AUROC
  comparator_denominators.json    per-comparator n, intersection, leakage rules
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

# comparator -> (prediction column, score kind, leakage-flag column or None, description)
COMPARATORS = {
    "SynGlue whole-PROTAC": ("gen_whole_pred_dc50_nM", "dc50", "exact_in_synglue",
                             "exact-in-SynGlue removed"),
    "SynGlue 3-token": ("gen_pred_dc50_nM", "dc50", "exact_in_synglue",
                        "exact-in-SynGlue removed; decomposed 151-set"),
    "PDP": ("pdp_active_proba", "proba", None, "none (baseline)"),
    "PDP-XGB": ("pdp_xgb_proba", "proba", None, "none (baseline)"),
    "PROTAC-STAN": ("protacstan_active_proba", "proba", "exact_in_stan",
                    "exact-in-STAN removed"),
}
THRESHOLDS = [10, 100, 1000]


def funnel_counts():
    p = C.PATHS["external_funnel"]
    if not os.path.exists(p):
        print("  funnel file not found; skipping")
        return pd.DataFrame()
    d = pd.read_csv(p)
    t = pd.to_numeric(d["tmax_chiral"], errors="coerce")
    rows = [dict(stage="assembled pool (pre-overlap)", n=2930),
            dict(stage="exact-overlap-free", n=int(len(d))),
            dict(stage="max Tanimoto < 0.90", n=int((t < 0.90).sum())),
            dict(stage="max Tanimoto < 0.85", n=int((t < 0.85).sum())),
            dict(stage="max Tanimoto < 0.70", n=int((t < 0.70).sum()))]
    df = pd.DataFrame(rows)
    C.save_csv(df, "external_funnel_counts.csv")
    print(df.to_string(index=False))
    return df


def main():
    funnel_counts()
    fp = pd.read_csv(C.require(C.PATHS["external_predictions"], "external predictions"))

    rows, elig = [], {}
    for name, (col, kind, flag, leak) in COMPARATORS.items():
        if col not in fp.columns:
            print(f"  [{name}] column {col} missing; skipping")
            continue
        d = fp.dropna(subset=[col, "dc50_median"]).copy()
        d = d[~d["dc50_censored"].fillna(False)]
        if flag and flag in d.columns:                 # apply this comparator's leakage rule
            d = d[~d[flag].fillna(False)]
            print(f"  [{name}] leakage filter '{flag}' applied -> n={len(d)}")
        elig[name] = set(d["canonical_SMILES"].astype(str))
        for thr in THRESHOLDS:
            y = (d["dc50_median"] <= thr).astype(int)
            if y.nunique() < 2 or len(d) < 10:
                rows.append(dict(comparator=name, threshold_nM=thr, n=len(d),
                                 n_pos=int(y.sum()), AUROC=None, leakage_rule=leak))
                continue
            score = -np.log10(d[col].astype(float)) if kind == "dc50" else d[col].astype(float)
            rows.append(dict(comparator=name, threshold_nM=thr, n=len(d), n_pos=int(y.sum()),
                             AUROC=float(roc_auc_score(y, score)), leakage_rule=leak))
    met = pd.DataFrame(rows)
    C.save_csv(met, "comparator_metrics.csv")
    print("\n" + met.to_string(index=False))

    inter = set.intersection(*elig.values()) if elig else set()
    C.save_json(dict(per_comparator_n={k: len(v) for k, v in elig.items()},
                     intersection_eligibility_only=len(inter),
                     note=("intersection counts compounds with a valid prediction and a non-censored "
                           "DC50 in every comparator; it is a prediction-availability intersection, "
                           "not a common evaluation cohort. TACK is excluded here because its "
                           "reported metrics use a disjoint non-held-out subset.")),
                "comparator_denominators.json")


if __name__ == "__main__":
    main()
