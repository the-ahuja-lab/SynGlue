#!/usr/bin/env python3
"""PDP (PROTAC-Degradation-Predictor) on the 6 submitted PROTACs. E3=VHL, POI=BRD4(O60885/Q01892), cell HeLa fallback."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor as pdp
REV="/storage/savi/saveenas/Projects/SynGlue/revision"
OUT=f"{REV}/rebuttal_analyses/19_6protacs_toolrun"
inp=pd.read_csv(f"{OUT}/input6.csv")
smis=inp["canonical_smiles"].tolist()
avail_e3={e.upper() for e in pdp.avail_e3_ligases()}
avail_cl=set(pdp.avail_cell_lines())
print("PDP E3 vocab:", sorted(avail_e3), "| n celllines:", len(avail_cl))
e3s=["VHL"]*len(smis); pois=["O60885"]*len(smis); cls=["HeLa"]*len(smis)
res=pdp.get_protac_active_proba(smis, e3s, pois, cls, device="cuda")
prob=np.asarray(res["mean"])
df=pd.DataFrame({"name":inp["name"],"canonical_smiles":smis,"pdp_active_proba":prob})
df.to_csv(f"{OUT}/pdp.csv",index=False)
print(df.to_string(index=False))
print("PDP done")