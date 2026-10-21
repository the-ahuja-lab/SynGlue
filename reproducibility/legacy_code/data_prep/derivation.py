#!/usr/bin/env python3
"""Per-source -> final external derivation (PROTAC-DB 3.0 = 6,110 as membership reference).
Efficient: build per-source canon->meta maps once; single Tanimoto pass per unique compound.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def murcko(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
def fp(s):
    m=Chem.MolFromSmiles(str(s)); return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_uniq=set(c for c in map(canon,pdb['Smiles'].dropna().astype(str)) if c)
tr_fps=[f for f in map(fp,pdb['Smiles'].dropna().astype(str)) if f is not None]
print(f"PROTAC-DB 3.0 reference: {len(tr_uniq)} unique, {len(tr_fps)} fps")

tack=pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
pat=pd.read_csv(f"{DATA}/merged_patent_pubchem_only.csv", low_memory=False)
tp=tack[tack['Database'].str.contains('TPDdb',na=False)]
pe=tack[tack['Database'].str.contains('PROTACpedia',na=False)]
SOURCES=[("TACK",tack,'SMILES','POI_UniProt','Ligase_Name'),
         ("TPDdb",tp,'SMILES','POI_UniProt','Ligase_Name'),
         ("PROTACpedia",pe,'SMILES','POI_UniProt','Ligase_Name'),
         ("PROTAC-Patent",pat,'canonical_SMILES','Target',None)]

def tmax2train(f):
    return max(DataStructs.BulkTanimotoSimilarity(f, tr_fps)) if f is not None else np.nan

# Build per-source unique compound with meta + tmax
perrows=[]; final={}
src_meta={}
for name,df,smi,tcol,ecol in SOURCES:
    sub=df.drop_duplicates(smi)
    seen=set(); comps=[]
    meta_map={}
    for _,r in sub.iterrows():
        s=r[smi]; c=canon(s)
        if c is None or c in seen: continue
        seen.add(c)
        f=fp(s); tm=tmax2train(f)
        tgt = (r[tcol] if tcol and pd.notna(r.get(tcol)) else 'NA')
        e3v = (r[ecol] if ecol and pd.notna(r.get(ecol)) else 'NA')
        comps.append(dict(canon=c, smiles=s, tmax=tm))
        meta_map[c]=(tgt, e3v, name)
    src_meta[name]=meta_map
    cans=[x['canon'] for x in comps]; tmax=np.array([x['tmax'] for x in comps],dtype=float)
    exact=sum(1 for c in cans if c in tr_uniq)
    perrows.append(dict(source=name, records=len(df), unique=len(cans),
        in_PROTACDB3_exact=exact, novel_to_DB3=len(cans)-exact,
        Tani_ge90=int(np.nansum(tmax>=0.90)), Tani_ge85=int(np.nansum(tmax>=0.85)),
        Tani_ge70=int(np.nansum(tmax>=0.70)), novel_Tani_lt70=int(np.nansum(tmax<0.70)),
        targets=sub[tcol].nunique() if tcol in sub.columns else 0,
        e3=sub[ecol].nunique() if ecol in sub.columns else 0,
        scaffolds=len(set(c for c in map(murcko,sub[smi]) if c))))
    # add novel to final union
    for x in comps:
        if x['canon'] in tr_uniq: continue
        if x['canon'] not in final:
            final[x['canon']]=(name, x['tmax'], x['smiles'])

per=pd.DataFrame(perrows); per.to_csv(f"{OUT}/per_source_vs_DB3.csv", index=False)
print("\n=== per-source vs PROTAC-DB 3.0 ===")
print(per.to_string(index=False))
print(f"\nFINAL EXTERNAL union (novel to PROTAC-DB3): {len(final)} unique")
print(pd.Series([v[0] for v in final.values()]).value_counts().to_string())

# splits for final external
final_meta={}
for c,(name,tm,s) in final.items():
    tgt,e3v,_=src_meta[name][c]
    final_meta[c]=dict(source=name, tmax=tm, smiles=s, target=tgt, e3=e3v,
                       scaffold=murcko(s))
fm=pd.DataFrame([{'canonical_SMILES':c, **d} for c,d in final_meta.items()])
fm.to_csv(f"{OUT}/final_external_with_meta.csv", index=False)
print("\nTarget split (top20):"); print(fm['target'].value_counts().head(20).to_string())
print("\nE3 split:"); print(fm['e3'].value_counts().to_string())
print("unique scaffolds:", fm['scaffold'].nunique())
print("Saved to", OUT)
