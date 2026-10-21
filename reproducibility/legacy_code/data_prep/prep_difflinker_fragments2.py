#!/usr/bin/env python3
"""
Build a DiffLinker input SDF with warhead and E3 ligand as disconnected
fragments in ONE molecule record, positioned ~8-10 A apart (typical PROTAC
warhead-to-E3 distance) in the GSPT1 pocket frame.
"""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

WARHEAD_SMILES = "Nc1nnc(C#CC2CC2)s1"
E3_SMILES = "N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
POCKET_CENTER = np.array([3.8, 28.9, 168.9])
SEP_AXIS = np.array([1.0, 0.0, 0.0])  # separate along x

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")

def embed(smi, seed=42):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    p = rdDistGeom.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(mol, p)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def translate(mol, center):
    conf = mol.GetConformer()
    pts = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    shift = np.array(center) - pts.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    return mol

if __name__ == "__main__":
    wh = embed(WARHEAD_SMILES)
    e3 = embed(E3_SMILES, seed=7)
    # Warhead near pocket center, E3 offset by 9 A along +x
    wh = translate(wh, POCKET_CENTER - SEP_AXIS * 4.5)
    e3 = translate(e3, POCKET_CENTER + SEP_AXIS * 4.5)
    combo = Chem.CombineMols(wh, e3)
    combo = Chem.RemoveHs(combo)
    conf = combo.GetConformer()
    frags = Chem.GetMolFrags(combo, asMols=True)
    c1 = np.array([conf.GetAtomPosition(a.GetIdx()) for a in frags[0].GetAtoms()]).mean(axis=0)
    c2 = np.array([conf.GetAtomPosition(a.GetIdx()) for a in frags[1].GetAtoms()]).mean(axis=0)
    print("fragments:", len(frags), "atoms:", combo.GetNumAtoms(), "centroid dist: %.1f A" % np.linalg.norm(c1 - c2))
    path = os.path.join(OUT, "fragments_combined.sdf")
    w = Chem.SDWriter(path)
    w.write(combo)
    w.close()
    print("wrote", path)
