#!/usr/bin/env python3
"""PROTAC-DC50 (Petrou & Minhas) on 6 PROTACs — ROBUST decomposition (RDKit-validated).
Fragments: e3 = VHL tail from 'N[C@@H](C(C)(C)C)'; warhead = head until first amide 'C(=O)N'
(or 'CCOCCO' for pyrazole-linked mol5); linker = remainder. Each validated by MolFromSmiles."""
import os, sys, warnings, pickle, json, urllib.request
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torchdrug import data, layers, models
from torchdrug.layers import geometry
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
OUT="/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/19_6protacs_toolrun"
CACHE="/tmp/protacdc50/cache"
def get_avalon(smiles):
    calc=FPCalculator("avalon"); tr=MoleculeTransformer(calc, n_jobs=0); return np.array(tr(smiles))
xgb_model=pickle.load(open("/tmp/protacdc50/gearnet_dc50_predictor.pkl","rb"))
gcm=layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
    edge_layers=[geometry.SpatialEdge(radius=10.0,min_distance=5),geometry.KNNEdge(k=10,min_distance=5),geometry.SequentialEdge(max_distance=2)],
    edge_feature="gearnet")
gearnet=models.GearNet(input_dim=21,hidden_dims=[512]*6,num_relation=7,edge_input_dim=59,
    batch_norm=True,activation="relu",concat_hidden=True,short_cut=True,readout="sum",num_angle_bin=8)
gearnet.load_state_dict(torch.load("/tmp/protacdc50/distance_gearnet_edge.pth",map_location="cpu")); gearnet.eval()
def protein_embedding(uniprot):
    pkl=os.path.join(CACHE,f"emb_{uniprot}.npy")
    if os.path.exists(pkl): return np.load(pkl)
    pdb=os.path.join(CACHE,f"{uniprot}.pdb")
    if not os.path.exists(pdb):
        try:
            with urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}",timeout=30) as r:
                info=json.load(r); url=info[0]["pdbUrl"]
        except Exception:
            url=f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"
        urllib.request.urlretrieve(url,pdb)
        if os.path.getsize(pdb)<1000: raise RuntimeError(f"AF dl failed {uniprot}")
    prot=data.Protein.from_pdb(pdb,atom_feature="position",bond_feature="length",residue_feature="symbol")
    pack=data.Protein.pack([prot])
    with torch.no_grad():
        t=gcm(pack); emb=gearnet.forward(t,t.residue_feature.float())["graph_feature"]
    np.save(pkl,emb.numpy()); return emb.numpy()

import re
def decompose(s):
    """Return (warhead, linker, e3) valid SMILES or None. Boundary = first chain pattern in head."""
    vhl_marker = "N[C@@H](C(C)(C)C)"
    if vhl_marker not in s:
        return None
    idx = s.index(vhl_marker)
    e3 = s[idx:]
    head = s[:idx]
    if not Chem.MolFromSmiles(e3):
        return None
    m = re.search(r"NCC{3,}|NCCOCCO|CCOCCO|N1CCC\(CC1\)", head)
    if m is None:
        return None
    cstart = m.start()
    war, link = head[:cstart], head[cstart:]
    if Chem.MolFromSmiles(war) and Chem.MolFromSmiles(link):
        return war, link, e3
    # alternate: include one more chain atom from warhead side if N-boundary fails
    war2 = head[:cstart+1]; link2 = head[cstart+1:]
    if Chem.MolFromSmiles(war2) and Chem.MolFromSmiles(link2) and Chem.MolFromSmiles(e3):
        return war2, link2, e3
    return None

inp=pd.read_csv(f"{OUT}/input6.csv")
print("BRD4(O60885) & VHL(P40337) embeddings...")
pe=protein_embedding("O60885"); ee=protein_embedding("P40337")
rows=[]
for _,r in inp.iterrows():
    s=r["canonical_smiles"]
    dec=decompose(s)
    if dec is None:
        rows.append(dict(name=r["name"], protacdc50_dc50_nM=np.nan, note="decomposition failed")); continue
    w,l,e3=dec
    wf=get_avalon([w]); lf=get_avalon([l]); ef=get_avalon([e3])
    vec=np.column_stack((pe,ee,wf,ef,lf))
    log50=xgb_model.predict(vec)[0]
    rows.append(dict(name=r["name"], protacdc50_dc50_nM=float(np.exp(log50)),
                     warhead_smiles=w, linker_smiles=l, e3_smiles=e3, note="ok"))
d=pd.DataFrame(rows); d.to_csv(f"{OUT}/protacdc50.csv", index=False)
print(d[["name","protacdc50_dc50_nM","note"]].to_string(index=False))
print("PROTAC-DC50 done")
