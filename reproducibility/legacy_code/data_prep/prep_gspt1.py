#!/usr/bin/env python3
"""Build DeLinker GSPT1 fragment-pair input file (data/gspt1_frags.txt)."""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

wh_frag = "[*:1]Nc1nnc(C#CC2CC2)s1"
e3_frag = "[*:2]N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
smi_frags = wh_frag + "." + e3_frag

def embed(smi):
    m = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m)
    # RDKit 2018: ETKDG via AllChem.ETKDG()
    params = AllChem.ETKDG()
    params.randomSeed = 42
    AllChem.EmbedMolecule(m, params)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    return m, m.GetConformer()

def get_att_neighbor(smi):
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            return list(a.GetNeighbors())[0].GetIdx()
    return None

m1, c1 = embed(wh_frag)
m2, c2 = embed(e3_frag)
i1 = get_att_neighbor(wh_frag)
i2 = get_att_neighbor(e3_frag)
p1 = np.array(c1.GetAtomPosition(i1))
p2 = np.array(c2.GetAtomPosition(i2))
dist = float(np.linalg.norm(p1 - p2))
line = "%s %s %s %.2f 1.5\n" % (smi_frags, smi_frags, smi_frags, dist)
with open("data/gspt1_frags.txt", "w") as f:
    f.write(line)
print(line.strip())
