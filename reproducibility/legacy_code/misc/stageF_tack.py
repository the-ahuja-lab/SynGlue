#!/usr/bin/env python3
"""TACK (KDD'26) XGB ensembles — DC50, Dmax, Activity — on the 6 submitted PROTACs.
Reuses ws2_tack_external.py datamodule/ensemble logic with /tmp/tack_repo + /tmp/tack_models."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, warnings, json, types, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq
try:
    import curvlinops
    if not hasattr(curvlinops, "_base"):
        m = types.ModuleType("curvlinops._base"); m._LinearOperator = type("_LinearOperator", (), {}); sys.modules["curvlinops._base"] = m
except Exception:
    pass
try:
    import laplace
except Exception:
    fake = types.ModuleType("laplace"); fake.Laplace = type("Laplace", (), {}); sys.modules["laplace"] = fake
sys.path.insert(0, "/tmp/tack_repo")
import torch, xgboost as xgb
from tackai.data.datamodule import DegradationComplexDataModule, load_config_from_yaml
OUT="/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/19_6protacs_toolrun"
MODELS="/tmp/tack_models"
TACK_RAW="/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet"
inp=pd.read_csv(f"{OUT}/input6.csv")
tack=pq.read_table(TACK_RAW).to_pandas()
# sequences: BRD4 POI (O60885) & VHL ligase (P40337) from TACK raw
brd4=tack[tack['POI_UniProt'].astype(str)=='O60885']
vhl=tack[tack['Ligase_UniProt'].astype(str)=='P40337']
poi_seq=brd4['POI_Sequence'].dropna().iloc[0]
lig_seq=vhl['Ligase_Sequence'].dropna().iloc[0]
print("BRD4 seq len:", len(poi_seq), "| VHL seq len:", len(lig_seq))
samples=[dict(SMILES=r['canonical_smiles'], Degradation_Target_Gene="BRD4",
              Degradation_Target_Sequence=poi_seq, Recruiter_Gene="VHL",
              Recruiter_Sequence=lig_seq, Cell_Line_ID="HeLa", Assay_Time=24.0, Assay="HiBiT")
         for _,r in inp.iterrows()]
print("samples:", len(samples))

def load_dm(hpath, spath):
    import inspect
    sig=inspect.signature(DegradationComplexDataModule.__init__); valid=set(sig.parameters.keys())-{"self"}
    cfg=load_config_from_yaml(hpath)
    mf=str(cfg.get("mol_features",""))
    if "fingerprint" in mf: cfg["use_fingerprints"]=True
    if "descriptor" in mf: cfg["use_descriptors"]=True
    pf=str(cfg.get("poi_features",""))
    if "sequence" in pf: cfg["use_poi_sequence_embedding"]=True
    lf=str(cfg.get("ligase_features",""))
    if "name" in lf: cfg["use_ligase_name_embedding"]=True
    cf=str(cfg.get("cell_features",""))
    if "description" in cf: cfg["use_cell_description_embedding"]=True
    cfg={k:v for k,v in cfg.items() if k in valid}
    dm=DegradationComplexDataModule(dataset=None, **cfg)
    dm.load_state_dict(torch.load(spath, map_location="cpu", weights_only=False))
    return dm
def fold_of(f):
    mm=re.search(r"fold=(\d+)", f); return int(mm.group(1)) if mm else None
def run_ensemble(task, scale="dc50"):
    ckpt_dir=os.path.join(MODELS, f"{task}_best_arch", "checkpoints")
    wfile=os.path.join(MODELS, f"{task}_best_arch", f"ensemble_weights_{task}_best_architecture.json")
    wdict=json.load(open(wfile)); wdict=wdict.get("weights", wdict) if isinstance(wdict, dict) else wdict
    all_files=os.listdir(ckpt_dir)
    dms=[f for f in all_files if f.endswith("_hparams.yaml")]
    sts=[f for f in all_files if f.endswith("_state.pt")]
    preds=np.zeros(len(samples)); wsum=0.0; nfolds=0
    for fname in sorted(all_files):
        if not fname.startswith("model=xgboost_"+task) or not fname.endswith(".json"): continue
        fnum=fold_of(fname)
        h=next((f for f in dms if fold_of(f)==fnum), ""); s=next((f for f in sts if fold_of(f)==fnum), "")
        if not (os.path.exists(os.path.join(ckpt_dir,h)) and os.path.exists(os.path.join(ckpt_dir,s))):
            print(f"  skip fold {fnum}"); continue
        dm=load_dm(os.path.join(ckpt_dir,h), os.path.join(ckpt_dir,s))
        feats=dm.featurize_samples_batch(samples, return_tensor="xgb")
        X=np.vstack([f[0] for f in feats]).astype(np.float32); names=list(feats[0][1])
        booster=xgb.Booster(); booster.load_model(os.path.join(ckpt_dir,fname))
        raw=booster.predict(xgb.DMatrix(X, feature_names=names))
        if raw.ndim>1: raw=raw[:, raw.shape[1]//2]
        lab_key=next((k for k in dm.label_transformers if k.lower()==task.lower()), None)
        if lab_key is None: lab_key=next(iter(dm.label_transformers))
        inv=dm.label_transformers[lab_key].inverse_transform(raw.reshape(-1,1))
        w=wdict.get(fname.replace(".json",""), wdict.get(fname, 1.0/len(wdict)))
        preds+=w*inv.ravel(); wsum+=w; nfolds+=1
    print(f"  [{task}] folds={nfolds}")
    return preds/max(wsum,1e-9)

res={}
res["dc50"]=run_ensemble("dc50")
res["dmax"]=run_ensemble("dmax")
try:
    res["bin"]=run_ensemble("bin")
except Exception as e:
    print("bin failed:", str(e)[:150]); res["bin"]=None
df=pd.DataFrame({"name":inp["name"],
                 "tack_dc50_nM":res["dc50"], "tack_dmax_pct":res["dmax"],
                 "tack_bin":res["bin"] if res["bin"] is not None else np.nan})
df.to_csv(f"{OUT}/tack.csv", index=False)
print(df.to_string(index=False))
print("TACK done")