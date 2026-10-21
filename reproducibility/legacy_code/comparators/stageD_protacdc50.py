"""PROTAC-DC50 (Petrou & Minhas) on the 6 submitted PROTACs.
NOTE: heuristic warhead/linker/E3 decomposition (novel components); POI=BRD4(O60885), E3=VHL(P40337)."""
import os, sys, warnings, pickle, json, urllib.request
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, pandas as pd, torch
from torchdrug import data, layers, models
from torchdrug.layers import geometry
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
REV="/storage/savi/saveenas/Projects/SynGlue/revision"
OUT=f"{REV}/rebuttal_analyses/19_6protacs_toolrun"
CACHE="/tmp/protacdc50/cache"; os.makedirs(CACHE, exist_ok=True)
def get_avalon(smiles):
    calc=FPCalculator("avalon"); tr=MoleculeTransformer(calc, n_jobs=0)
    return np.array(tr(smiles))
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
        if os.path.getsize(pdb)<1000: raise RuntimeError(f"AF download failed {uniprot}")
    prot=data.Protein.from_pdb(pdb,atom_feature="position",bond_feature="length",residue_feature="symbol")
    pack=data.Protein.pack([prot])
    with torch.no_grad():
        t=gcm(pack); emb=gearnet.forward(t,t.residue_feature.float())["graph_feature"]
    np.save(pkl,emb.numpy()); return emb.numpy()

def decompose(s):
    """Heuristic: e3 = VHL tail from 'N[C@@H](C(C)(C)C)'; linker = middle chain + C(=O);
    warhead = head before linker."""
    idx=s.index("N[C@@H](C(C)(C)C)")
    e3=s[idx:]
    head=s[:idx]           # ends with ...C(=O)N
    # linker: last 'CC(=O)' inside the chain: take from after warhead-carbonyl-N
    # warhead = head up to and including its last aromatic C(=O) before the chain
    c=Chem.MolFromSmiles(head)
    # simple: warhead = head up to the LAST 'C(=O)' occurrence minus 'N'
    k=head.rfind("C(=O)")
    if k==-1: k=head.rfind("C=O")
    war=head[:k+4]
    linker=head[k+4:].lstrip("N")  # chain + its C(=O)? linker should include C(=O); approximate: chain w/o N
    linker2=linker+"C(=O)" if not linker.endswith("(=O)") else linker
    return war, linker2, e3
import re
from rdkit import Chem
inp=pd.read_csv(f"{OUT}/input6.csv")
print("EOI BRD4 embedding...")
pe=protein_embedding("O60885"); print("  BRD4 emb", pe.shape)
ee=protein_embedding("P40337"); print("  VHL emb", ee.shape)
rows=[]
for _,r in inp.iterrows():
    s=r["canonical_smiles"]
    try:
        w,l,e3=decompose(s)
        wf=get_avalon([w]); lf=get_avalon([l]); ef=get_avalon([e3])
        vec=np.column_stack((pe,ee,wf,ef,lf))
        log50=xgb_model.predict(vec)[0]
        rows.append(dict(name=r["name"],protacdc50_dc50_nM=float(np.exp(log50)),
                         warhead_smiles=w, linker_smiles=l, e3_smiles=e3))
    except Exception as ex:
        rows.append(dict(name=r["name"],protacdc50_dc50_nM=np.nan, note=str(ex)))
d=pd.DataFrame(rows); d.to_csv(f"{OUT}/protacdc50.csv",index=False)
print(d.to_string(index=False))
print("PROTAC-DC50 done")
