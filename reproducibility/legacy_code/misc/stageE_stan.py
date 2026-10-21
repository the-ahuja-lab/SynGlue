"""PROTAC-STAN on the 6 submitted PROTACs (POI=BRD4 O60885, E3=VHL P40337)."""
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
ROOT="/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from data import PROTACData
from data_loader import PROTACLoader
from model import PROTAC_STAN
import toml
OUT="/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/19_6protacs_toolrun"
inp=pd.read_csv(f"{OUT}/input6.csv")
def props(smi):
    m=Chem.MolFromSmiles(smi)
    return [Descriptors.MolWt(m), Descriptors.ExactMolWt(m), Crippen.MolLogP(m),
            m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m)]
cols=['Smiles','Molecular Weight','Exact Mass','XLogP3','Heavy Atom Count','Ring Count',
      'Hydrogen Bond Acceptor Count','Hydrogen Bond Donor Count','Rotatable Bond Count',
      'Topological Polar Surface Area']
rows=[]
for _,r in inp.iterrows():
    rows.append([r["canonical_smiles"]]+props(r["canonical_smiles"]))
raw=pd.DataFrame(rows,columns=cols)
raw.insert(1,"Uniprot","O60885"); raw.insert(2,"E3 ligase Uniprot","P40337")
os.makedirs("data/ws1_full", exist_ok=True)
raw.to_csv("data/ws1_full/s6.csv", index=False)
print("raw rows:", len(raw))
pd_=PROTACData(root="data/ws1_full", name="s6")
print("processed exists:", os.path.exists("data/ws1_full/processed/s6/protac.pt"))
cfg=toml.load("config.toml")
model=PROTAC_STAN(cfg["model"]); model.load_state_dict(torch.load("saved_models/protac-stan.pt",map_location="cpu"))
model.eval()
_, loader=PROTACLoader(root="data/ws1_full", name="s6", batch_size=6, train_ratio=0.0)
probs=[]
with torch.no_grad():
    for d in loader:
        out,_=model(d["protac"], d["e3_ligase"], d["poi"], mode="eval")
        p=torch.softmax(out,dim=1)[:,1].cpu().numpy(); probs.extend(p)
df=pd.DataFrame({"name":inp["name"],"protacstan_active_proba":probs})
df.to_csv(f"{OUT}/protacstan.csv",index=False)
print(df.to_string(index=False))
print("STAN done")
