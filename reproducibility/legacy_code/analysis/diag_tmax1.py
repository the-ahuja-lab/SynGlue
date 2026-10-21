#!/usr/bin/env python3
"""Diagnose why Tanimoto=1.0 (or >0.99) compounds remain in the 'filtered' final external set.
Hypothesis: Morgan fps (default) ignore chirality/tautomers -> stereoisomers / tautomers of
training molecules score 1.0 while their canonical SMILES differ (exact-match removal misses them).
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"
from rdkit.Chem.inchi import MolToInchiKey
import functools
@functools.lru_cache(maxsize=200000)
def _ik(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return 'err'
    try: return MolToInchiKey(m).split('-')[0]
    except: return 'err'
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def fp(s,chiral=False):
    m=Chem.MolFromSmiles(str(s))
    return AllChem.GetMorganFingerprintAsBitVect(m,2,2048,useChirality=chiral) if m else None

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_smis=pdb['Smiles'].dropna().astype(str).tolist()
tr_uniq=set(c for c in map(canon,tr_smis) if c)
tr_fps=[fp(s) for s in tr_smis]
tr_fps=[f for f in tr_fps if f is not None]
tr_fps_ch=[fp(s,True) for s in tr_smis]
tr_fps_ch=[f for f in tr_fps_ch if f is not None]

fm=pd.read_csv(f"{OUT}/final_external_with_meta.csv")
tack=fm[fm['source']=='TACK'].copy()
tack['tmax']=pd.to_numeric(tack['tmax'],errors='coerce')
print("TACK novel in final external:", len(tack))
hi=tack[tack['tmax']>=0.999]
print(f"  with tmax >= 0.999: {len(hi)}")
print(f"  with tmax >= 1.0-1e-6: {int((tack['tmax']>=0.999999).sum())}")

# sample high-sim fingerprints and compare with chirality-aware fingerprints
h=hi.sort_values('tmax',ascending=False).head(8)
for _,r in h.iterrows():
    s=r['canonical_SMILES']
    f=fp(s); fc=fp(s,True)
    if f is None: continue
    sims=DataStructs.BulkTanimotoSimilarity(f,tr_fps)
    i=int(np.argmax(sims)); best=tr_smis[i]
    sim=sims[i]
    simc=max(DataStructs.BulkTanimotoSimilarity(fc,tr_fps_ch)) if fc else np.nan
    cb=canon(best)
    same_canon = (canon(s)==cb)
    # inchi comparison
    try:
        ik_e=_ik(s)
        ik_t=_ik(best)
    except Exception:
        ik_e=ik_t='err'
    print(f"\n  ext : {s[:80]}")
    print(f"  best: {best[:80]}")
    print(f"  sim(achiral fp)={sim:.4f} | sim(chiral fp)={simc:.4f} | same_canonical={same_canon}")
    print(f"  inchi ext {ik_e.split('-')[0] if ik_e!='err' else 'err'} vs train {ik_t.split('-')[0] if ik_t!='err' else 'err'} (formula-layer equal => isomers?)")