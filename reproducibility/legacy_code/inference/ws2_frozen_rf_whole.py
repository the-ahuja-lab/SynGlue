#!/usr/bin/env python3
"""Frozen RF predictions for whole-PROTAC mode (rf_predict env: sklearn 1.0.2)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
cache = os.path.join(TIR, "emb_whole")
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"

Z = np.load(os.path.join(cache, "Z_frozen_whole.npy"))
ids = pd.read_csv(os.path.join(cache, "scoring_ids.csv"))
rf50 = joblib.load(os.path.join(SYNGLUE, "models", "rf_dc50.joblib"))
rfmax = joblib.load(os.path.join(SYNGLUE, "models", "rf_dmax.joblib"))
ids["frozen_whole_pred_dc50_nM"] = 10 ** rf50.predict(Z)
ids["frozen_whole_pred_dmax_pct"] = rfmax.predict(Z)
ids.to_csv(os.path.join(TIR, "synglue_wholeprotac_frozen.csv"), index=False)
print("frozen whole-PROTAC predictions:", len(ids))
