#!/usr/bin/env python3
"""
Prepare shared GSPT1 inputs for warhead-generation tools:
1. DeepICL: warhead SDF translated to pocket cavity-1 center + protein pocket PDB
2. DiffPROTACs: fragment XYZ files (warhead + E3 ligand, translated to pocket)
3. Gcoupler pair string reference
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

WARHEAD_SMILES = "*Nc1nnc(C#CC2CC2)s1"
E3_SMILES = "*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
POCKET_CENTER = [3.8, 28.9, 168.9]  # cavity 1 center (5HXB)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
os.makedirs(OUT, exist_ok=True)

def embed_smiles(smi, name, center=None):
    """Embed 3D conformer, translate to center, write SDF + XYZ (no H in XYZ)."""
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"cannot parse {smi}"
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"embed failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    if center is not None:
        # translate centroid to pocket center
        import numpy as np
        pts = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        centroid = pts.mean(axis=0)
        shift = np.array(center) - centroid
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    # SDF
    sdf_path = os.path.join(OUT, name + ".sdf")
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()
    # XYZ (heavy atoms only, no Hs)
    xyz_path = os.path.join(OUT, name + ".xyz")
    with open(xyz_path, "w") as f:
        heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        f.write(f"{len(heavy)}\n{name}\n")
        for a in heavy:
            p = conf.GetAtomPosition(a.GetIdx())
            f.write(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}\n")
    print(f"[prep] {name}: SDF {sdf_path}, XYZ {xyz_path} ({len(heavy)} heavy atoms)")
    return sdf_path, xyz_path

if __name__ == "__main__":
    wh_sdf, wh_xyz = embed_smiles(WARHEAD_SMILES.replace("*", "N"), "warhead", POCKET_CENTER)
    e3_sdf, e3_xyz = embed_smiles(E3_SMILES.replace("*", "N"), "e3_ligand", POCKET_CENTER)
    # combined XYZ for DiffPROTACs: fragments concatenated
    lines = []
    for fn, tag in [(wh_xyz, "warhead"), (e3_xyz, "e3")]:
        with open(fn) as f:
            lines.append(f.read())
    combined = os.path.join(OUT, "fragments_combined.xyz")
    with open(combined, "w") as f:
        f.write(lines[0] + lines[1])
    print(f"[prep] combined fragments XYZ: {combined}")
