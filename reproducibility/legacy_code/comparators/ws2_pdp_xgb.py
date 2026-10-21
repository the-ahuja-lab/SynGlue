#!/usr/bin/env python3
"""PDP XGBoost ensemble on the FULL external set (distinct model family)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor as pdp

TIR = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws1/tiers"
pdp_full = pd.read_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"))
mask = pdp_full["e3_supported"]
smis = pdp_full.loc[mask, "SMILES"].tolist()
e3s = pdp_full.loc[mask, "E3"].tolist()
pois = pdp_full.loc[mask, "POI_UniProt"].tolist()
cls = pdp_full.loc[mask, "Cell_Line"].tolist()
print(f"PDP-XGBoost on {len(smis)} compounds...")
res = pdp.get_protac_active_proba(smis, e3s, pois, cls, device="cuda", use_xgboost_models=True)
pdp_full["pdp_xgb_proba"] = np.nan
pdp_full.loc[mask, "pdp_xgb_proba"] = np.asarray(res["mean"])
pdp_full.to_csv(os.path.join(TIR, "pdp_fullset_predictions.csv"), index=False)
print("saved pdp_xgb_proba")
