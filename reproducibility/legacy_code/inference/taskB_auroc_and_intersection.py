#!/usr/bin/env python3
"""Task B/E: search for AUROC=0.84 at 1uM; compute true comparator intersection and per-model sets."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, itertools, json
from sklearn.metrics import roc_auc_score
OUT="/storage/savi/saveenas/Projects/SynGlue/outputs/synglue_final_revision_evidence"
MR=f"{OUT}/machine_readable"
fp=pd.read_csv("/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/08_external_nonprotacdb/outputs/final_predictions.csv", low_memory=False)
print("rows", len(fp), "cols", len(fp.columns))

def auc(sub, pred, thr):
    d=sub.dropna(subset=[pred,"dc50_median"]).copy()
    d["y"]=(d["dc50_median"]<=thr).astype(int)
    if d["y"].nunique()<2 or len(d)<10: return None, len(d), int(d["y"].sum())
    return roc_auc_score(d["y"], -np.log10(d[pred].astype(float))), len(d), int(d["y"].sum())

variants = {
 "excl_censored_excl_exact": lambda d: d[(~d["dc50_censored"].fillna(False)) & (~d["exact_in_synglue"].fillna(False))],
 "excl_censored_only": lambda d: d[~d["dc50_censored"].fillna(False)],
 "all_with_pred": lambda d: d,
 "excl_exact_only": lambda d: d[~d["exact_in_synglue"].fillna(False)],
}
rows=[]
for pred in ["gen_whole_pred_dc50_nM","gen_pred_dc50_nM","pdp_active_proba","protacstan_active_proba"]:
    if pred not in fp.columns: continue
    for vname, fn in variants.items():
        sub=fn(fp)
        for thr in [10,100,1000]:
            a,n,npos = auc(sub,pred,thr)
            rows.append(dict(predictor=pred, variant=vname, threshold_nM=thr, auroc=a, n=n, n_pos=npos))
res=pd.DataFrame(rows)
res.to_csv(f"{MR}/auroc_variant_search.csv", index=False)
hits=res[(res.auroc.notna()) & (res.auroc.between(0.83,0.85))]
print("\nAUROC in [0.83,0.85]:")
print(hits.to_string(index=False) if len(hits) else "  none")
print("\nSynGlue rows (whole, 3-token) at 1000 nM:")
print(res[(res.predictor.isin(['gen_whole_pred_dc50_nM','gen_pred_dc50_nM'])) & (res.threshold_nM==1000)].to_string(index=False))

# comparator intersection
models={"SynGlue whole":"gen_whole_pred_dc50_nM","SynGlue 3-token":"gen_pred_dc50_nM",
        "PDP":"pdp_active_proba","PDP-XGB":"pdp_xgb_proba","PROTAC-STAN":"protacstan_active_proba"}
elig={}
for m,col in models.items():
    if col not in fp.columns: continue
    d=fp.dropna(subset=[col,"dc50_median"])
    d=d[~d["dc50_censored"].fillna(False)]
    elig[m]=set(d["canonical_SMILES"].astype(str))
inter=set.intersection(*elig.values()) if elig else set()
out={"per_model_eligible_n":{k:len(v) for k,v in elig.items()},
     "intersection_all_models_n":len(inter),
     "intersection_note":"models with a valid prediction AND uncensored DC50; excludes leakage filters (none applied for baselines)"}
json.dump(out, open(f"{MR}/comparator_intersection.json","w"), indent=2)
print("\nper-model eligible n:", out["per_model_eligible_n"])
print("intersection n:", len(inter))
