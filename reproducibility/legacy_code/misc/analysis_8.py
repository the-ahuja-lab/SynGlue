#!/usr/bin/env python3
"""
Analysis 4 (R1.2) — Gcoupler vs de novo/linker generation tools on GSPT1.
Canonical runner: revision/ws2_gen_collect.py (regenerates from raw tool outputs in
revision/benchmarking/ws2_gen/). This script re-runs it and copies outputs here.
"""
import os, shutil, subprocess, sys

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

env = dict(os.environ)
env["PYTHONPATH"] = REV
subprocess.run([sys.executable, os.path.join(REV, "ws2_gen_collect.py")], env=env, check=True)

GEN = os.path.join(REV, "benchmarking", "ws2_gen")
for f in ["ws2_gen_comparison.csv", "ws2_gen_report.md"]:
    shutil.copy(os.path.join(GEN, f), os.path.join(OUT, f))
shutil.copy(os.path.join(REV, "figures", "ws2_gen_comparison.png"), os.path.join(OUT, "comparison_plot.png"))
shutil.copy(os.path.join(REV, "figures", "ws2_gen_comparison.svg"), os.path.join(OUT, "comparison_plot.svg"))
for f in os.listdir(os.path.join(GEN, "collected")):
    shutil.copy(os.path.join(GEN, "collected", f), os.path.join(OUT, f))
print("Analysis 4 outputs copied to", OUT)
