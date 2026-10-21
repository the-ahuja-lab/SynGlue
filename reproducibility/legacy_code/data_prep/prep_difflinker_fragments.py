#!/usr/bin/env python3
"""
Build a clean combined-fragment SDF for DiffLinker:
warhead + E3 ligand as disconnected fragments in ONE molecule (RDKit),
with 3D coords, no H, positioned in the GSPT1 pocket cavity-1 frame.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

WARHEAD_SMILES = "Nc1nnc(C#CC2CC2)s1"   # dummy replaced with N attachment
E3_SMILES = "N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
POCKET_CENTER = [3.8, 28.9, 168.9]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")

def embed(smi, seed=42):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"embed failed: {smi}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def translate(mol, center):
    import numpy as np
    conf = mol.GetConformer()
    pts = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    shift = np.array(center) - pts.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    return mol

def combine(mol_a, mol_b):
    """Merge two mols into one disconnected molecule."""
    combo = Chem.CombineMols(mol_a, mol_b)
    return combo

if __name__ == "__main__":
    wh = embed(WARHEAD_SMILES)
    e3 = embed(E3_SMILES, seed=7)
    wh = translate(wh, POCKET_CENTER)
    e3 = translate(e3, POCKET_CENTER)
    combo = combine(wh, e3)
    combo = Chem.RemoveHs(combo)
    print("combined atoms:", combo.GetNumAtoms(), "frags:", Chem.GetMolFrags(combo))
    sdf_path = os.path.join(OUT, "fragments_combined.sdf")
    w = Chem.SDWriter(sdf_path)
    w.write(combo)
    w.close()
    print("wrote", sdf_path)
