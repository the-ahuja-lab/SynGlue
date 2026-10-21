#!/usr/bin/env python3
"""
PDP on the FULL external set (all endpoint-bearing compounds).
Inputs: final_dataset_E3_known.csv (unique compounds, endpoints, metadata)
Outputs: benchmarking/ws1/tiers/pdp_fullset_predictions.csv
Usage: conda run -n pdp python ws2_pdp_fullset.py
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor as pdp

EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
OUT = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws1/tiers/pdp_fullset_predictions.csv"

ext = pd.read_csv(EXT)
uniq = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
# endpoints per compound
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
    "cell_lines": ";".join(g["Cell_Line"].dropna().astype(str).unique()[:3]),
}))
uniq = uniq.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}),
                  left_on="canonical_SMILES", right_on="canonical_smiles", how="left")

avail_e3 = {e.upper() for e in pdp.avail_e3_ligases()}
avail_cl = set(pdp.avail_cell_lines())
rows = []
for _, r in uniq.iterrows():
    e3 = str(r["E3_Ligase"]).strip().upper()
    cl_raw = str(r.get("cell_lines", "")).split(";")[0].strip() if pd.notna(r.get("cell_lines")) else ""
    rows.append(dict(SMILES=r["canonical_SMILES"], E3=e3, POI_UniProt=str(r["POI_UniProt"]),
                     Cell_Line=cl_raw if cl_raw in avail_cl else "HeLa",
                     e3_supported=e3 in avail_e3,
                     cellline_fallback=(cl_raw not in avail_cl)))
df = pd.DataFrame(rows)
print(f"total: {len(df)} | E3 supported: {df['e3_supported'].sum()} | cell-line fallback: {int(df['cellline_fallback'].sum())}")

mask = df["e3_supported"]
smis = df.loc[mask, "SMILES"].tolist()
e3s = df.loc[mask, "E3"].tolist()
pois = df.loc[mask, "POI_UniProt"].tolist()
cls = df.loc[mask, "Cell_Line"].tolist()
print(f"running PDP on {len(smis)} compounds (GPU)...")
t0 = time.time()
res = pdp.get_protac_active_proba(smis, e3s, pois, cls, device="cuda")
print(f"done in {time.time()-t0:.0f}s")
df["pdp_active_proba"] = np.nan
df.loc[mask, "pdp_active_proba"] = np.asarray(res["mean"])
df = df.merge(uniq[["canonical_SMILES", "dc50_median", "dmax_max", "dc50_censored"]],
              left_on="SMILES", right_on="canonical_SMILES", how="left", suffixes=("", "_u"))
df.to_csv(OUT, index=False)
print(f"Saved {OUT}")
print(df[["SMILES", "E3", "pdp_active_proba", "dc50_median"]].head(5).to_string())
