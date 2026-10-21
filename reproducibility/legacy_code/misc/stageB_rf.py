#!/usr/bin/env python3
"""Stage B (rf_predict env): frozen RF (rf_dc50/rf_dmax) on SynGlue frozen Z embeddings."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
REV="/storage/savi/saveenas/Projects/SynGlue/revision"
OUT=f"{REV}/rebuttal_analyses/19_6protacs_toolrun"
Z=np.load(f"{OUT}/Z_frozen_whole.npy")
rf50=joblib.load("/storage/savi/saveenas/Projects/SynGlue_Py/models/rf_dc50.joblib")
rfmax=joblib.load("/storage/savi/saveenas/Projects/SynGlue_Py/models/rf_dmax.joblib")
p50=rf50.predict(Z); pm=rfmax.predict(Z)
names=pd.read_csv(f"{OUT}/input6.csv")["name"].tolist()
pd.DataFrame({"name":names,"synglue_rf_pred_logdc50":p50,"synglue_rf_pred_dmax":pm}).to_csv(f"{OUT}/synglue_rf.csv",index=False)
print("SynGlue frozen RF DC50 log10 nM:", np.round(p50,3), "-> nM:", np.round(10**p50,2))
print("SynGlue frozen RF Dmax %:", np.round(pm,2))