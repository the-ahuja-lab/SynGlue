#!/usr/bin/env python3
"""
Convert DiffPROTACs XYZ outputs (fragments + linker) to SMILES via obabel,
then parse with RDKit. Writes converted_smiles.csv.
(Python 3.5 compatible)
"""
import os, glob, subprocess, csv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "diffprotacs")
OBABEL = "/home/saveenas/miniconda3/bin/obabel"

smiles_list = []
for xyz in sorted(glob.glob(os.path.join(OUT, "GSPT1_CRBN", "*.xyz"))):
    name = os.path.basename(xyz)
    sdf_tmp = "/tmp/dp_%s.sdf" % name.replace(".xyz", "")
    smi_tmp = "/tmp/dp_%s.smi" % name.replace(".xyz", "")
    r1 = subprocess.run(OBABEL + " " + xyz + " -O " + sdf_tmp + " 2>/dev/null", shell=True)
    if r1.returncode != 0:
        continue
    r2 = subprocess.run(OBABEL + " " + sdf_tmp + " -O " + smi_tmp + " 2>/dev/null", shell=True)
    if r2.returncode != 0:
        continue
    if os.path.exists(smi_tmp):
        for line in open(smi_tmp):
            toks = line.strip().split()
            if toks:
                smiles_list.append((name, toks[0]))

with open(os.path.join(OUT, "converted_smiles.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "smiles"])
    w.writerows(smiles_list)
print("converted %d molecules" % len(smiles_list))
for n, s in smiles_list[:5]:
    print(" ", n, s[:100])
