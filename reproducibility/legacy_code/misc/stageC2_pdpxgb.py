import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor")
import protac_degradation_predictor as pdp
OUT="/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/19_6protacs_toolrun"
inp=pd.read_csv(f"{OUT}/input6.csv")
smis=inp["canonical_smiles"].tolist()
res=pdp.get_protac_active_proba(smis, ["VHL"]*6, ["O60885"]*6, ["HeLa"]*6, device="cuda", use_xgboost_models=True)
df=pd.DataFrame({"name":inp["name"],"pdp_xgb_proba":np.asarray(res["mean"])})
df.to_csv(f"{OUT}/pdpxgb.csv",index=False)
print(df.to_string(index=False))
