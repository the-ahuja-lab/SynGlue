#!/usr/bin/env python3
"""
GSPT1 De Novo Warhead Generation — Master Orchestrator

Runs ALL available generative tools against the GSPT1 (5HXB) pocket to
produce de novo warhead-like molecules / full PROTACs, then evaluates
with a standardized filter cascade.

Tools:
  1. Gcoupler (reference, Link-INVENT)   — existing output reused
  2. Pocket2Mol (pocket-guided de novo)  — GPU
  3. DiffSBDD (diffusion pocket)         — GPU
  4. PGMG (pharmacophore-guided)         — CPU
  5. DeepICL (interaction-conditioned)   — CPU
  6. DiffLinker (3D linker diffusion)    — GPU
  7. DeLinker (2D linker VAE)            — CPU (TF 1.10)
  8. ShapeLinker (Link-INVENT + shape)   — CPU
  9. DiffPROTACs (PROTAC linker diff.)   — GPU
 10. Pharmacophore enumeration (custom)  — CPU (existing output)

Blocked: PRosettaC (Rosetta license), DeepLigBuilder (code never released).

Usage:
    python run_warhead_generation.py --tool pocket2mol   # single tool
    python run_warhead_generation.py --all               # everything
    python run_warhead_generation.py --all --dry-run     # print commands only

Author: Feynman (SynGlue revision team)
"""
import argparse
import os
import subprocess
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = ROOT  # this file lives in results/warheads/
INPUTS = os.path.join(ROOT, "inputs")
LOGS = os.path.join(RESULTS, "logs")
os.makedirs(LOGS, exist_ok=True)

# ============================================================
# GSPT1 Inputs
# ============================================================
PDB = "/storage/savi/saveenas/Projects/Magnet/Gcoupler/GSPT1/5HXB_chainA_clean.pdb"
PDB_FULL = "/storage/savi/saveenas/Projects/Magnet/Gcoupler/GSPT1/5HXB.pdb"
POCKET_CENTER = "3.8,28.9,168.9"   # cavity 1
BBOX = 23.0
WARHEAD_SMI = "*Nc1nnc(C#CC2CC2)s1"
E3_SMI = "*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
# cavity 1 residues (5HXB chain A)
CAVITY1_RESIS = "442 443 444 445 456 457 458 459 460 497 498 499 517 553 554 555 556 557 558 610 611 612 613 614 615 616 618 626 627"

TOOLS_BASE = "/storage/savi/saveenas/Projects/SynGlue/validation_tools"
CONDA = "/home/saveenas/miniconda3/bin/conda"

# ============================================================
# Tool commands
# ============================================================

def cmd_pocket2mol():
    tool_dir = os.path.join(TOOLS_BASE, "Pocket2Mol")
    env = os.path.join(TOOLS_BASE, "envs", "Pocket2Mol")
    out = os.path.join(RESULTS, "pocket2mol")
    return [
        f"cd {tool_dir}",
        f"{CONDA} run -p {env} python sample_for_pdb.py "
        f"--pdb_path {PDB} --center {POCKET_CENTER} --bbox_size {BBOX} "
        f"--config ./configs/sample_for_pdb.yml --device cuda --outdir {out}",
    ]

def cmd_diffsbdd():
    tool_dir = os.path.join(TOOLS_BASE, "DiffSBDD")
    out = os.path.join(RESULTS, "diffsbdd", "samples.sdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ckpt = os.path.join(tool_dir, "checkpoints", "crossdocked_fullatom_cond.ckpt")
    return [
        f"cd {tool_dir}",
        f"{CONDA} run -n diffsbdd python generate_ligands.py {ckpt} "
        f"--pdbfile {PDB} --resi_list {CAVITY1_RESIS} "
        f"--outfile {out} --n_samples 300 --batch_size 30 --sanitize",
    ]

def cmd_pgmg():
    tool_dir = os.path.join(TOOLS_BASE, "PGMG")
    phar_dir = os.path.join(RESULTS, "pgmg", "pharmacophores")
    out = os.path.join(RESULTS, "pgmg")
    weights = os.path.join(tool_dir, "weights", "chembl_fold0_epoch32.pth")
    tok = os.path.join(tool_dir, "weights", "tokenizer.pkl")
    return [
        f"cd {tool_dir}",
        f"{CONDA} run -n pgmg python generate.py {phar_dir} {out} "
        f"{weights} {tok} --filter --device cpu --n_mol 1500 --batch_size 256",
    ]

def cmd_deepicl():
    tool_dir = os.path.join(TOOLS_BASE, "DeepICL")
    out = os.path.join(RESULTS, "deepicl")
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)
    seed_sdf = os.path.join(INPUTS, "warhead.sdf")
    return [
        f"cd {tool_dir}",
        # Prepare complex: merge ligand + protein with obabel
        f"{CONDA} run -n DeepICL bash -c 'cd {out} && "
        f"/home/saveenas/miniconda3/bin/obabel {seed_sdf} {PDB} -O {work}/complex.pdb -j'",
        # Processor + generate via a dedicated driver script
        f"{CONDA} run -n DeepICL python {ROOT}/run_deepicl_generate.py "
        f"--workdir {work} --pdb {PDB} --ligand {seed_sdf} --out {out} --n_sample 100",
    ]

def cmd_difflinker():
    tool_dir = os.path.join(TOOLS_BASE, "DiffLinker")
    out = os.path.join(RESULTS, "difflinker")
    frag_sdf = os.path.join(INPUTS, "fragments_combined.sdf")
    pocket_pdb = os.path.join(RESULTS, "difflinker", "pocket.pdb")
    model = os.path.join(tool_dir, "models", "geom_difflinker.ckpt")
    if not os.path.exists(model):
        model = os.path.join(tool_dir, "models", "geom_difflinker_new.pt")
    return [
        f"cd {tool_dir}",
        # Pocket = cavity-1 residues extracted from full PDB
        f"{CONDA} run -n difflinker python {ROOT}/prep_pocket.py "
        f"--pdb {PDB_FULL} --resis '{CAVITY1_RESIS}' --out {pocket_pdb}",
        f"{CONDA} run -n difflinker python generate_with_pocket.py "
        f"--fragments {frag_sdf} --pocket {pocket_pdb} --model {model} "
        f"--linker_size 4,10 --output {out} --n_samples 300 --n_steps 100",
        # Convert XYZ -> SDF -> SMILES with obabel
        f"/home/saveenas/miniconda3/bin/obabel {out}/*.xyz -O {out}/all.sdf",
    ]

def cmd_delinker():
    tool_dir = os.path.join(TOOLS_BASE, "DeLinker")
    out = os.path.join(RESULTS, "delinker")
    os.makedirs(out, exist_ok=True)
    # Build GSPT1 fragment-pair input (5-token format: mol linker frags dist angle)
    prep = os.path.join(out, "prep_gspt1.py")
    with open(prep, "w") as f:
        f.write('''#!/usr/bin/env python3
import sys
sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/validation_tools/DeLinker")
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

wh_frag = "[*:1]Nc1nnc(C#CC2CC2)s1"
e3_frag = "[*:2]N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
smi_frags = wh_frag + "." + e3_frag

def get_att(smi):
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            return list(a.GetNeighbors())[0].GetIdx()
    return None

def embed(smi):
    m = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m)
    p = rdDistGeom.ETKDGv3(); p.randomSeed = 42
    AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
    return m, m.GetConformer()

m1, c1 = embed(wh_frag); m2, c2 = embed(e3_frag)
idx1 = get_att(wh_frag) + 0  # dummy removed later; use neighbor atom pos
# attachment atom = neighbor of dummy
for a in m1.GetAtoms():
    if a.GetSymbol() == "*":
        n1 = list(a.GetNeighbors())[0]
for a in m2.GetAtoms():
    if a.GetSymbol() == "*":
        n2 = list(a.GetNeighbors())[0]
p1 = np.array(c1.GetAtomPosition(n1.GetIdx()))
p2 = np.array(c2.GetAtomPosition(n2.GetIdx()))
dist = float(np.linalg.norm(p1 - p2))
line = "%s %s %s %.2f 1.5\n" % (smi_frags, smi_frags, smi_frags, dist)
open("data/gspt1_frags.txt", "w").write(line)
print(line.strip())
''')
    # DeLinker config file (written by master script)
    cfg = os.path.join(out, "delinker_config.json")
    with open(cfg, "w") as f:
        json.dump({
            "generation": True,
            "number_of_generation_per_valid": 50,
            "batch_size": 1,
            "train_file": "data/molecules_gspt1.json",
            "valid_file": "data/molecules_gspt1.json",
        }, f, indent=2)
    return [
        f"cd {tool_dir}",
        f"CUDA_VISIBLE_DEVICES='' {CONDA} run -n DeLinker-env bash -c 'cd {tool_dir} && python {prep}'",
        f"CUDA_VISIBLE_DEVICES='' {CONDA} run -n DeLinker-env bash -c 'cd {tool_dir} && python data/prepare_data.py --data_path data/gspt1_frags.txt --dataset_name gspt1 --test_mode'",
        f"CUDA_VISIBLE_DEVICES='' {CONDA} run -n DeLinker-env bash -c 'cd {tool_dir} && python DeLinker.py --dataset gspt1 --restore models/pretrained_DeLinker_model.pickle --config-file {cfg}'",
    ]

def cmd_shapelinker():
    tool_dir = os.path.join(TOOLS_BASE, "ShapeLinker")
    out = os.path.join(RESULTS, "shapelinker")
    os.makedirs(out, exist_ok=True)
    cfg = os.path.join(out, "sampling_config.json")
    prior = os.path.join(tool_dir, "models", "linkinvent.prior")
    config = {
        "logging": {
            "job_id": "GSPT1",
            "job_name": "Sampling GSPT1 pair",
            "logging_path": os.path.join(out, "progress.log"),
            "recipient": "local",
            "sender": "http://127.0.0.1"
        },
        "model_type": "link_invent",
        "parameters": {
            "batch_size": 32,
            "model_path": os.path.join(tool_dir, "models", "linkinvent.prior"),
            "num_samples": 200,
            "output_path": os.path.join(out, "sampled.csv"),
            "randomize_warheads": True,
            "temperature": 1.5,
            "warheads": [
                f"{WARHEAD_SMI}|{E3_SMI}"
            ],
            "with_likelihood": True
        },
        "run_type": "sampling",
        "version": 3
    }
    with open(cfg, "w") as f:
        json.dump(config, f, indent=2)
    return [
        f"cd {tool_dir}",
        f"{CONDA} run -n shapelinker python Reinvent/input.py {cfg}",
    ]

def cmd_diffprotacs():
    tool_dir = os.path.join(TOOLS_BASE, "DiffPROTACs")
    out = os.path.join(RESULTS, "diffprotacs")
    os.makedirs(out, exist_ok=True)
    frag_xyz = os.path.join(INPUTS, "fragments_combined.xyz")
    return [
        f"cd {tool_dir}",
        f"{CONDA} run -n DiffPROTACs python {ROOT}/run_diffprotacs_generate.py "
        f"--model_path checkpoints/protacs_best.ckpt "
        f"--xyz_path {frag_xyz} --linker_len 8 --name GSPT1_CRBN "
        f"--output_dir {out} --n_samples 100 --batch_size 50",
    ]

def cmd_gcoupler():
    src = "/storage/savi/saveenas/Projects/Magnet/Reinvent/GSPT1/results/scaffold_memory.csv"
    dst = os.path.join(RESULTS, "gcoupler", "scaffold_memory.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        return [f"cp {src} {dst}"]
    return [f"echo 'Gcoupler source missing: {src}'"]

def cmd_pharmacophore():
    src = os.path.join(ROOT, "..", "pharmacophore", "generated_candidates.csv")
    dst = os.path.join(RESULTS, "pharmacophore", "generated_candidates.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        return [f"cp {src} {dst}"]
    return [f"echo 'Pharmacophore source missing: {src}'"]

TOOLS = {
    "gcoupler": cmd_gcoupler,
    "pocket2mol": cmd_pocket2mol,
    "diffsbdd": cmd_diffsbdd,
    "pgmg": cmd_pgmg,
    "deepicl": cmd_deepicl,
    "difflinker": cmd_difflinker,
    "delinker": cmd_delinker,
    "shapelinker": cmd_shapelinker,
    "diffprotacs": cmd_diffprotacs,
    "pharmacophore": cmd_pharmacophore,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=list(TOOLS.keys()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tools = list(TOOLS.keys()) if args.all else ([args.tool] if args.tool else [])
    if not tools:
        ap.error("Specify --tool NAME or --all")

    for t in tools:
        print("=" * 70)
        print(f"[{t}]")
        cmds = TOOLS[t]()
        for c in cmds:
            print(f"  $ {c}")
        if args.dry_run:
            continue
        log = os.path.join(LOGS, f"{t}.log")
        with open(log, "w") as f:
            for c in cmds:
                print(f"Running: {c}")
                f.write(f"$ {c}\n")
                r = subprocess.run(c, shell=True, stdout=f, stderr=subprocess.STDOUT)
                f.write(f"exit={r.returncode}\n")
                if r.returncode != 0:
                    print(f"[{t}] command failed (exit {r.returncode}): {c}")
        print(f"[{t}] done → {log}")

if __name__ == "__main__":
    main()
