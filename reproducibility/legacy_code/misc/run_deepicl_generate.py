#!/usr/bin/env python3
"""
DeepICL driver for GSPT1 de novo warhead generation (ligand-sparse mode).

Uses PocketDataProcessor (processor_no_ligand.py) — the official no-ligand
path for ligand-sparse targets — with the 5HXB cavity-1 pocket and the
cavity center as origin.
"""
import argparse
import os
import pickle
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--pocket", required=True, help="pocket PDB (cavity residues)")
    ap.add_argument("--center", default="3.8,28.9,168.9", help="pocket center x,y,z")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sample", type=int, default=100)
    args = ap.parse_args()

    DEEPICL_ROOT = "/storage/savi/saveenas/Projects/SynGlue/validation_tools/DeepICL"
    sys.path.insert(0, DEEPICL_ROOT)
    sys.path.insert(0, os.path.join(DEEPICL_ROOT, "script"))

    os.makedirs(args.workdir, exist_ok=True)

    # 1. Build pocket PDB (cavity-1 residues) if not present
    pocket_pdb = args.pocket
    if not os.path.exists(pocket_pdb):
        print("pocket PDB missing:", pocket_pdb)
        sys.exit(1)

    # 2. Process via PocketDataProcessor (no-ligand)
    from data.processor_no_ligand import PocketDataProcessor
    center = [float(v) for v in args.center.split(",")]
    proc = PocketDataProcessor(
        data_dir=os.path.dirname(pocket_pdb),
        save_dir=os.path.join(args.workdir, "proc/"),
        max_atom_num=50,
        max_add_atom_num=30,
        seed=0,
        tmp_dir="/tmp/deepicl_tmp/",
    )
    os.makedirs(proc.save_dir, exist_ok=True)
    os.makedirs(proc._tmp_dir, exist_ok=True)
    data = proc._processor(pocket_pdb, pocket_center=center)
    if data is None:
        print("ERROR: processor returned None")
        sys.exit(1)
    key = "GSPT1"
    with open(os.path.join(proc.save_dir, key), "wb") as w:
        pickle.dump(data, w)
    with open(os.path.join(args.workdir, "test_keys.pkl"), "wb") as w:
        pickle.dump([key], w)
    print("processed:", key, "->", os.path.join(proc.save_dir, key))

    # 3. Run generate.py
    result_dir = os.path.join(args.out, "results")
    os.makedirs(result_dir, exist_ok=True)
    gen_cmd = (
        f"python {DEEPICL_ROOT}/script/generate.py --ncpu 8 --k 8 "
        f"--data_dir {proc.save_dir} --key_dir {args.workdir} "
        f"--restart_dir {DEEPICL_ROOT}/script/best_model/save_best.pt "
        f"--result_dir {result_dir} "
        f"--num_layers 6 --num_dense_layers 3 --num_hidden_feature 128 --num_cond_feature 7 "
        f"--num_sample {args.n_sample} --max_num_add_atom 30 "
        f"--dist_one_hot_param1 0 10 25 --dist_one_hot_param2 0 15 300 "
        f"--temperature_factor1 0.05 --temperature_factor2 0.05 "
        f"--radial_limits 0.9 2.2 --pocket_coeff_max 10.0 --pocket_coeff_thr 2.5 --pocket_coeff_beta 0.91 "
        f"--conditional --use_condition --verbose"
    )
    print("$", gen_cmd)
    r = subprocess.run(gen_cmd, shell=True)
    print("generate.py exit:", r.returncode)

if __name__ == "__main__":
    main()
