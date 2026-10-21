import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys, os
sys.path.insert(0,"/storage/savi/saveenas/Projects/SynGlue/revision")
from reproduce_original_model import load_data
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem
def fps_list(smis,n=2048):
    X=[]
    for s in smis:
        m=Chem.MolFromSmiles(str(s)); f=AllChem.GetMorganFingerprintAsBitVect(m,2,n) if m else None
        X.append(np.zeros(n,int) if f is None else np.array(f))
    return np.vstack(X)
Xtr,y50tr,ymtr,metatr=load_data()
sig=pd.DataFrame({"w":[Xtr[i,0].tobytes() for i in range(len(Xtr))],"l":[Xtr[i,1].tobytes() for i in range(len(Xtr))],"e":[Xtr[i,2].tobytes() for i in range(len(Xtr))]})
dup=sig.duplicated(keep="first"); metatr=metatr[~dup].reset_index(drop=True); y50tr=y50tr[~dup]; ymtr=ymtr[~dup]
mk=np.isfinite(y50tr); tr_smiles=np.asarray(metatr["SMILES"].values)
dm_ok=np.isfinite(ymtr)
Xtr_fp=fps_list(tr_smiles[dm_ok]); yDM=ymtr[dm_ok]
rfm=RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1); rfm.fit(Xtr_fp,yDM)
names=pd.read_csv("input6.csv")["name"]; smis=pd.read_csv("input6.csv")["canonical_smiles"]
pm=rfm.predict(fps_list(smis))
d=pd.read_csv("morgan_rf.csv"); d["morgan_rf_pred_dmax"]=np.round(pm,2); d.to_csv("morgan_rf.csv",index=False)
print("Morgan-RF Dmax(%):", np.round(pm,2))
