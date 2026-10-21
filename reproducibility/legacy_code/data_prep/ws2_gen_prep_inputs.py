#!/usr/bin/env python3
"""
WS2 generative benchmark — prepare common GSPT1 inputs for
DiffLinker (fragments.sdf 3D), DiffPROTACs (fragments.xyz), DeLinker (JSON),
ShapeLinker (config JSON), PGMG (pharmacophore .edgep).
Common GSPT1 pair: warhead *Nc1nnc(C#CC2CC2)s1 | CRBN E3 *N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1
Output: benchmarking/ws2_gen/inputs/
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(REV, "benchmarking", "ws2_gen", "inputs")
os.makedirs(OUT, exist_ok=True)

WARHEAD = "*Nc1nnc(C#CC2CC2)s1"
E3 = "*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"

def strip_attach(smi, tag):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(smi)
    # remove dummy atom
    em = Chem.RWMol(m)
    for a in em.GetAtoms():
        if a.GetAtomicNum() == 0:
            em.RemoveAtom(a.GetIdx())
            break
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    return mol, Chem.MolToSmiles(mol)

def embed3d(mol, seed=42):
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, randomSeed=seed + 1, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

print("GSPT1 warhead:", WARHEAD)
print("GSPT1 E3     :", E3)

w_mol, w_smi = strip_attach(WARHEAD, "W")
e_mol, e_smi = strip_attach(E3, "E")
print("stripped warhead:", w_smi)
print("stripped E3     :", e_smi)

# ─── DiffLinker: fragments.sdf (warhead + E3 with dummy anchors, 3D) ───
def _embed_with_dummy(smi, seed=42):
    m = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(m)
    ok = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if ok != 0: AllChem.EmbedMolecule(mol, randomSeed=seed + 1, useRandomCoords=True)
    try: AllChem.MMFFOptimizeMolecule(mol)
    except Exception: pass
    return Chem.RemoveHs(mol)
w3 = _embed_with_dummy(WARHEAD); e3_ = _embed_with_dummy(E3)
w3.SetProp("_Name", "GSPT1_warhead")
e3_.SetProp("_Name", "GSPT1_e3_CRBN")
writer = Chem.SDWriter(os.path.join(OUT, "fragments.sdf"))
writer.write(w3); writer.write(e3_); writer.close()
print("DiffLinker  -> fragments.sdf (with dummy anchors)")

# ─── DiffPROTACs: fragments.xyz (lines: ELEMENT x y z) ───
# DiffPROTACs XYZ: drop dummy atoms (vocab has no *); embed original fragments with dummy
def _embed_orig(smi, seed=42):
    m = Chem.MolFromSmiles(smi)
    m2 = Chem.RWMol(m)
    for a in m2.GetAtoms():
        if a.GetAtomicNum() == 0:
            m2.RemoveAtom(a.GetIdx()); break
    mol = m2.GetMol(); Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    try: AllChem.MMFFOptimizeMolecule(mol)
    except Exception: pass
    return Chem.RemoveHs(mol)
with open(os.path.join(OUT, "fragments.xyz"), "w") as f:
    for smi, tag in [(WARHEAD, "W"), (E3, "E")]:
        mol = _embed_orig(smi)
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1: continue
            p = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {p.x:.3f} {p.y:.3f} {p.z:.3f}\n")
print("DiffPROTACs -> fragments.xyz")

# ─── DeLinker: txt -> prepare_data.py (test mode) ───
# DeLinker fragment format: ORIGINAL fragment SMILES with [*:1] / [*:2] dummies
# (keep original atom order — canonicalization broke dummy placement)
dl_frags = "[*:1]Nc1nnc(C#CC2CC2)s1.[*:2]N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
with open(os.path.join(OUT, "delinker_input.txt"), "w") as f:
    f.write(f"{dl_frags} 4.82 2.25\n")
print("DeLinker    -> delinker_input.txt:", dl_frags)

# ─── ShapeLinker: sampling config (link-invent prior) ───
cfg = {
    "version": 3, "run_type": "sampling", "model_type": "link_invent",
    "logging": {"sender": "http://127.0.0.1", "recipient": "local",
                "logging_path": os.path.join(REV, "benchmarking", "ws2_gen", "shapelinker_progress.log"),
                "job_name": "GSPT1 sampling", "job_id": "GSPT1"},
    "parameters": {
        "model_path": "/storage/savi/saveenas/Projects/SynGlue/validation_tools/ShapeLinker/models/linkinvent.prior",
        "output_path": os.path.join(REV, "benchmarking", "ws2_gen", "shapelinker_sampled.csv"),
        "num_samples": 500, "batch_size": 32, "with_likelihood": True, "temperature": 1.0,
        "randomize_warheads": True,
        "warheads": [f"{WARHEAD}|{E3}"],
    },
}
with open(os.path.join(OUT, "shapelinker_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("ShapeLinker -> shapelinker_config.json")

# ─── PGMG: GSPT1 pharmacophore (.edgep) — constructed from warhead features ───
# feature points: AROM (thiazole ring), HACC (triazole/amine N), HDON (amine)
# approximate coordinates around warhead centroid; pairwise distances arbitrary but self-consistent
pts = [("AROM", [0.0, 0.0, 0.0]), ("HACC", [2.4, 1.2, 0.0]), ("HDON", [-2.1, 1.8, 0.5]), ("HACC", [3.2, -1.5, 1.0])]
n = len(pts)
with open(os.path.join(OUT, "gspt1_phar.edgep"), "w") as f:
    f.write(f"{n}\n")
    for i, (t, _) in enumerate(pts):
        f.write(f"{i+1} {t}\n")
    # pairwise distances (Euclidean on the coords)
    coords = np.array([p[1] for p in pts])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            f.write(f"{i+1} {j+1} {d:.2f}\n")
print("PGMG        -> gspt1_phar.edgep (constructed, approximate)")

# log
with open(os.path.join(OUT, "README_inputs.md"), "w") as f:
    f.write("# WS2 Generative Benchmark — Common GSPT1 Inputs\n\n")
    f.write(f"- Warhead: `{WARHEAD}` (attachment `*`)\n")
    f.write(f"- E3 (CRBN): `{E3}`\n")
    f.write(f"- DeLinker fragments: {dl_frags}\n")
    f.write("- PGMG pharmacophore: constructed from warhead features (AROM/HACC/HDON) — APPROXIMATE\n")
print("\nInputs written to", OUT)
