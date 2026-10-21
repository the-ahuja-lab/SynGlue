#!/usr/bin/env python3
"""Correct PROTAC-PatentDB analysis using the FULL official file (63,136 rows / 57,533 unique).
Compare to PROTAC-DB 3.0 (6,110 unique training ref) by exact + Tanimoto; build patent funnel.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
PAT=f"{REV}/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def fp(s):
    m=Chem.MolFromSmiles(str(s)); return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None

print("loading PROTAC-DB 3.0 ref...", flush=True)
pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_uniq=set(c for c in map(canon,pdb['Smiles'].dropna().astype(str)) if c)
tr_fps=[f for f in map(fp,pdb['Smiles'].dropna().astype(str)) if f is not None]
print(f"  DB3 ref: {len(tr_uniq)} unique, {len(tr_fps)} fps", flush=True)

print("loading FULL patent file...", flush=True)
pat=pd.read_csv(PAT, low_memory=False, usecols=['SMILES','Target','InChI Key'])
print(f"  patent rows: {len(pat)}", flush=True)
t0=time.time()
cans=[canon(s) for s in pat['SMILES'].dropna().astype(str)]
valid=[c for c in cans if c]
uniq=set(valid)
exact=sum(1 for c in uniq if c in tr_uniq)
novel=len(uniq)-exact
print(f"  valid canonical: {len(valid)} | unique: {len(uniq)} | exact-in-DB3: {exact} | novel: {novel}", flush=True)
print(f"  unique targets (patent): {pat['Target'].nunique()}", flush=True)

# Tanimoto max vs training for each UNIQUE novel compound
print("computing Tanimoto max-to-DB3 for unique patent compounds...", flush=True)
tmax={}
unique_smis=[]
for s in pat['SMILES'].dropna().astype(str):
    c=canon(s)
    if c is None or c in tmax: continue
    unique_smis.append((c,s))
    tmax[c]=None
print(f"  unique to process: {len(unique_smis)}", flush=True)
cnt=0
for c,s in unique_smis:
    f=fp(s)
    if f is not None:
        tmax[c]=max(DataStructs.BulkTanimotoSimilarity(f, tr_fps))
    cnt+=1
    if cnt%5000==0: print(f"    {cnt} done, {time.time()-t0:.0f}s", flush=True)
tm=np.array([tmax[c] for c,_ in unique_smis], dtype=float)
ge90=int(np.nansum(tm>=0.90)); ge85=int(np.nansum(tm>=0.85)); ge70=int(np.nansum(tm>=0.70)); lt70=int(np.nansum(tm<0.70))
print(f"  Tanimoto >=0.90: {ge90} | >=0.85: {ge85} | >=0.70: {ge70} | <0.70 (novel): {lt70}", flush=True)
print(f"  exact-in-DB3: {exact}", flush=True)
pd.DataFrame([{"metric":"patent rows","value":len(pat)},
              {"metric":"valid canonical","value":len(valid)},
              {"metric":"unique canonical","value":len(uniq)},
              {"metric":"unique targets","value":pat['Target'].nunique()},
              {"metric":"exact-in-DB3(name)","value":exact},
              {"metric":"novel-to-DB3","value":novel},
              {"metric":"Tani>=0.90","value":ge90},
              {"metric":"Tani>=0.85","value":ge85},
              {"metric":"Tani>=0.70","value":ge70},
              {"metric":"novel Tani<0.70","value":lt70}]).to_csv(f"{OUT}/patent_full_check.csv", index=False)
np.save(f"{OUT}/patent_tmax.npy", tm)
print("saved patent_full_check.csv + patent_tmax.npy", flush=True)
