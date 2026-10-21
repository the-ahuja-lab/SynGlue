#!/usr/bin/env python3
"""Task A: reconstruct the compound-level mapped ordering (9,252), map split indices -> compounds,
verify train/test disjointness, and compute external-overlap + label reuse per split."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, json
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
DATA="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
FM="/storage/savi/saveenas/Projects/SynGlue/revision/models/final_protac_model"
OUT="/storage/savi/saveenas/Projects/SynGlue/outputs/synglue_final_revision_evidence/machine_readable"
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def norm(x): return str(x).strip().upper()

protac=pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False)
wh=pd.read_csv(f"{DATA}/Updated_warhead.csv", low_memory=False)
lk=pd.read_csv(f"{DATA}/linker_updated_features.csv", low_memory=False)
e3=pd.read_csv(f"{DATA}/E3_updated_features.csv", low_memory=False)
for df,cols in [(protac,["warhead_Uniprot","warhead_Target","E3 ligase","Linker ID","PROTAC_Compound_ID"]),
                (wh,["warhead_Uniprot","warhead_Target"]),(lk,["linker_Compound_ID"]),(e3,["E3_Ligase"])]:
    for c in cols:
        if c in df.columns: df[c]=df[c].astype(str).str.strip().str.upper()

rows=[]
for _,p in protac.iterrows():
    cid=str(p["PROTAC_Compound_ID"])
    w=wh[(wh["warhead_Uniprot"]==p["warhead_Uniprot"])|(wh["warhead_Target"]==p["warhead_Target"])]
    l=lk[lk["linker_Compound_ID"]==p["Linker ID"]]
    e=e3[e3["E3_Ligase"]==p["E3 ligase"]]
    if w.empty or l.empty or e.empty: continue
    rows.append(dict(compound_id=cid, smiles=str(p["PROTAC_SMILES"]),
                     warhead_id=w.iloc[0]["warhead_Compound_ID"], linker_id=p["Linker ID"],
                     e3_id=e.iloc[0]["E3_Compound_ID"]))
mapped=pd.DataFrame(rows)
print("reconstructed mapped compounds:", len(mapped))
mapped["canon"]=[canon(s) for s in mapped["smiles"]]
mapped.to_csv(f"{OUT}/mapped_compound_order.csv", index=False)

z=np.load(f"{FM}/four_split_indices.npz", allow_pickle=True)
ext=pd.read_csv("/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/external_not_in_training.csv", low_memory=False)
ext_canon=set(c for c in map(canon, ext["SMILES"].dropna().astype(str)) if c)
print("external canon:", len(ext_canon))

report=[]
for split in ["Random","Scaffold-held","Target-held","E3-held (VHL)"]:
    tr=z[f"tr_{split}"]; te=z[f"te_{split}"]
    trc=set(mapped["canon"].iloc[tr].dropna()); tec=set(mapped["canon"].iloc[te].dropna())
    tr_comp=set(mapped["compound_id"].iloc[tr]); te_comp=set(mapped["compound_id"].iloc[te])
    report.append(dict(split=split, n_train=len(tr), n_test=len(te),
        shared_compound_ids_train_test=len(tr_comp & te_comp),
        test_rows_sharing_compound_with_train=int(mapped["compound_id"].iloc[te].isin(tr_comp).sum()),
        train_unique_compounds=len(trc), test_unique_compounds=len(tec),
        index_overlap_train_test=len(set(tr.tolist()) & set(te.tolist())),
        canon_overlap_train_test=len(trc & tec),
        test_overlap_external=len(tec & ext_canon),
        train_overlap_external=len(trc & ext_canon)))
rep=pd.DataFrame(report)
rep.to_csv(f"{OUT}/final_protac_model_split_independence.csv", index=False)
print(rep.to_string(index=False))
