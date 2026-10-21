#!/usr/bin/env python3
"""Write DiffLinker fragments SDF without dummy atoms (heavy atoms, 3D)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/revision")
from rdkit import Chem
from rdkit.Chem import AllChem

def embed_nodummy(smi, seed=42):
    m = Chem.MolFromSmiles(smi)
    rm = Chem.RWMol(m)
    for a in rm.GetAtoms():
        if a.GetAtomicNum() == 0:
            rm.RemoveAtom(a.GetIdx()); break
    mol = rm.GetMol(); Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if ok != 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed + 1, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return Chem.RemoveHs(mol)

w = embed_nodummy("*Nc1nnc(C#CC2CC2)s1")
e = embed_nodummy("*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1")
w.SetProp("_Name", "GSPT1_warhead"); e.SetProp("_Name", "GSPT1_e3")
wtr = Chem.SDWriter("/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws2_gen/inputs/fragments_nodummy.sdf")
wtr.write(w); wtr.write(e); wtr.close()
print("written fragments_nodummy.sdf")
