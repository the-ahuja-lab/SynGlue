#!/usr/bin/env python3
"""
WS2 — PDP (PROTAC-Degradation-Predictor) on the WS1 external scoring set.
Outputs: benchmarking/ws1/tiers/pdp_predictions.csv (SMILES, E3, POI, cell line,
known DC50/Dmax, PDP active probability) + coverage report.
Usage: conda run -n pdp python ws2_pdp_external.py
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor as pdp

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")

sc = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
print(f"compounds: {len(sc)}")

avail_e3 = {e.upper() for e in pdp.avail_e3_ligases()}
avail_cl = set(pdp.avail_cell_lines())
print(f"PDP E3 vocab: {sorted(avail_e3)} | cell lines: {len(avail_cl)}")

# per compound: pick first valid cell line from metadata
rows = []
for _, r in sc.iterrows():
    smi = r["canonical_smiles"]
    e3 = str(r.get("E3", "")).strip().upper()
    poi = r.get("POI_UniProt", "")
    cl_raw = str(r.get("cell_lines", "")).split(";")[0].strip() if pd.notna(r.get("cell_lines")) else ""
    cl = cl_raw if cl_raw in avail_cl else "HeLa"
    e3_ok = e3 in avail_e3
    rows.append(dict(SMILES=smi, E3=e3, POI_UniProt=poi, Cell_Line=cl,
                     e3_supported=e3_ok, cellline_fallback=(cl != cl_raw)))

df = pd.DataFrame(rows)
print("E3 supported:", df["e3_supported"].sum(), "/", len(df))
print("Cell-line fallback to HeLa:", int(df["cellline_fallback"].sum()))

# run PDP batch (supported E3s only)
mask = df["e3_supported"]
smis = df.loc[mask, "SMILES"].tolist()
e3s = df.loc[mask, "E3"].tolist()
pois = df.loc[mask, "POI_UniProt"].tolist()
cls = df.loc[mask, "Cell_Line"].tolist()
print(f"Running PDP on {len(smis)} compounds...")
t0 = time.time()
res = pdp.get_protac_active_proba(smis, e3s, pois, cls, device="cuda")
print(f"done in {time.time()-t0:.0f}s | keys: {list(res.keys())}")
prob = np.asarray(res["mean"])
df["pdp_active_proba"] = np.nan
df.loc[mask, "pdp_active_proba"] = prob
df = df.merge(sc[["canonical_smiles", "dc50_median", "dmax_max", "dc50_censored", "tier"]],
              left_on="SMILES", right_on="canonical_smiles", how="left", suffixes=("", "_dup"))
df.to_csv(os.path.join(TIR, "pdp_predictions.csv"), index=False)
print(f"\nSaved: {TIR}/pdp_predictions.csv")
print(df[["SMILES", "E3", "POI_UniProt", "Cell_Line", "pdp_active_proba", "dc50_median"]].head(8).to_string())
