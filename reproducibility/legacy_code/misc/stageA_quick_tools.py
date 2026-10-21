#!/usr/bin/env python3
"""Stage A (synglu_test_package): Morgan-RF, Tanimoto-NN, SynGlue whole-PROTAC (transformer heads + Z)
for the 6 submitted PROTAC molecules. Saves intermediate CSVs/npz for later RF roundtrip."""
import os, sys, warnings, json, subprocess
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
REV="/storage/savi/saveenas/Projects/SynGlue/revision"
OUT=f"{REV}/rebuttal_analyses/19_6protacs_toolrun"
os.makedirs(OUT, exist_ok=True)
SYNGLUE="/storage/savi/saveenas/Projects/SynGlue_Py"
GROVER_CKPT=os.path.join(SYNGLUE,"models","grover_fixed.pt")
GROVER_PATCHED="/tmp/ws1_grover/grover_patched"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, REV)
from reproduce_original_model import MultiTaskProtacModel, load_data

M = {
 "Molecule 1":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)NCCOCCOCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
 "Molecule 2":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)CNCCOCCOCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)NCc1ccc(cc1)c1scnc1C)O",
 "Molecule 3":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)NCCCCCCCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
 "Molecule 4":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)NCCCCCCCCCCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
 "Molecule 5":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)c1cnn(c1)CCOCCOCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
 "Molecule 6":"COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)N1CCC(CC1)OCCOCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
}
smis=list(M.values()); names=list(M.keys())
pd.DataFrame({"name":names,"canonical_smiles":smis}).to_csv(f"{OUT}/input6.csv", index=False)

# ── GROVER embeddings for all 6 full PROTACs ──
fn=f"{OUT}/full_smiles.csv"; pd.DataFrame({"smiles":smis}).to_csv(fn,index=False)
feat=f"{OUT}/full_features.npz"; fps=f"{OUT}/full_fps.npz"
if not os.path.exists(fps):
    subprocess.run(f"python {GROVER_PATCHED}/scripts/save_features.py --data_path {fn} --save_path {feat} --features_generator rdkit_2d_normalized --restart", shell=True, check=True, capture_output=True)
    subprocess.run(f"python {GROVER_PATCHED}/main.py fingerprint --data_path {fn} --features_path {feat} --checkpoint_path {GROVER_CKPT} --fingerprint_source both --output {fps}", shell=True, check=True, capture_output=True)
F=np.load(fps)["fps"][:,:4800]
print("GROVER full embeddings:", F.shape)
Xe=np.repeat(F[:,None,:],3,axis=1).astype(np.float32)
np.save(f"{OUT}/Xe_whole.npy",Xe)

# ── frozen transformer ──
fz=MultiTaskProtacModel(input_dim=4800).to(DEVICE)
fz.load_state_dict(torch.load(os.path.join(SYNGLUE,"models","multitask_transformer.pt"), map_location=DEVICE, weights_only=True))
fz.eval(); Z=[]; head50=[]; headmax=[]
with torch.no_grad():
    for b in range(0,len(Xe),6):
        x=torch.tensor(Xe[b:b+6],dtype=torch.float32).to(DEVICE)
        h=fz.proj(x); h=fz.transformer(h); attn=torch.softmax(fz.attn_pool(h),dim=1)
        fused=(h*attn).sum(dim=1)
        Z.append(fused.cpu().numpy()); head50.append(fz.head_dc50(fused).cpu().numpy()); headmax.append(fz.head_dmax(fused).cpu().numpy())
Z=np.vstack(Z); head50=np.concatenate(head50); headmax=np.concatenate(headmax)
np.save(f"{OUT}/Z_frozen_whole.npy",Z)
print("transformer head DC50(log10 nM):", np.round(head50.ravel(),3))
print("transformer head Dmax(%):", np.round(headmax.ravel(),3))

# ── Morgan-RF (own baseline: train on training data, full-PROTAC Morgan) ──
Xtr,y50tr,ymtr,metatr=load_data()
sig=pd.DataFrame({"w":[Xtr[i,0].tobytes() for i in range(len(Xtr))],"l":[Xtr[i,1].tobytes() for i in range(len(Xtr))],"e":[Xtr[i,2].tobytes() for i in range(len(Xtr))]})
dup=sig.duplicated(keep="first"); metatr=metatr[~dup].reset_index(drop=True); y50tr=y50tr[~dup]; ymtr=ymtr[~dup]
mk=np.isfinite(y50tr); tr_smiles=np.asarray(metatr["SMILES"].values)[mk]; try50=y50tr[mk]; trym=ymtr[mk]
def fps_list(smis,n=2048):
    X=[]; 
    for s in smis:
        m=Chem.MolFromSmiles(str(s)); f=AllChem.GetMorganFingerprintAsBitVect(m,2,n) if m else None
        X.append(np.zeros(n,int) if f is None else np.array(f))
    return np.vstack(X)
print("Morgan-RF training on", len(tr_smiles), "compounds...")
Xtr_fp=fps_list(tr_smiles); Xte_fp=fps_list(smis)
rf50=RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1); rf50.fit(Xtr_fp, try50)
dmax_ok=np.isfinite(trym); rfm=RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1); rfm.fit(Xtr_fp[dmax_ok], trym[dmax_ok])
mp50=rf50.predict(Xte_fp); mpmax=rfm.predict(Xte_fp)
pd.DataFrame({"name":names,"morgan_rf_pred_logdc50":mp50,"morgan_rf_pred_dmax":mpmax}).to_csv(f"{OUT}/morgan_rf.csv",index=False)
print("Morgan-RF logDC50:", np.round(mp50,3), "Dmax:", np.round(mpmax,1))

# ── Tanimoto-NN (kNN median of training DC50) ──
tr_fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(s)),2,2048) for s in tr_smiles]
k=10
tn=[]
for s in smis:
    f=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048)
    sims=DataStructs.BulkTanimotoSimilarity(f,tr_fps)
    idx=np.argsort(sims)[-k:]
    tn.append(np.median(try50[idx]))
pd.DataFrame({"name":names,"tanimoto_nn_pred_logdc50":tn}).to_csv(f"{OUT}/tanimoto_nn.csv",index=False)
print("Tanimoto-NN logDC50:", np.round(tn,3))
print("Stage A done ->", OUT)