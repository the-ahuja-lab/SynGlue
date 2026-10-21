#!/usr/bin/env python3
"""
Generate summary figures for the GSPT1 warhead generation comparison:
1. Pass-rate bar chart (warhead filter + linker cascade)
2. Top passing molecules per tool
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(ROOT, "summary")

df = pd.read_csv(os.path.join(SUMMARY, "warhead_comparison.csv"))
df = df.sort_values("WarheadPassRate%", ascending=False)

fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(df))
w = 0.38
b1 = ax.bar([i - w/2 for i in x], df["WarheadPassRate%"], w, label="Warhead filter pass rate %", color="#4C72B0")
b2 = ax.bar([i + w/2 for i in x], df["LinkerPassRate%"], w, label="PROTAC linker cascade pass rate %", color="#DD8452")
ax.set_xticks(list(x))
ax.set_xticklabels(df["Tool"], rotation=30, ha="right")
ax.set_ylabel("Pass rate (%)")
ax.set_title("GSPT1 (5HXB) de novo warhead generation — all-tool comparison (2026-08-12)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY, "warhead_pass_rates.png"), dpi=150)
print("saved", os.path.join(SUMMARY, "warhead_pass_rates.png"))

# Property comparison: MW distribution per tool
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
props = []
for tool in df["Tool"]:
    f = os.path.join(SUMMARY, f"warhead_{tool.lower().replace(' ','')}.csv")
    if not os.path.exists(f):
        continue
    p = pd.read_csv(f)
    p["tool"] = tool
    props.append(p)
allp = pd.concat(props)
from rdkit import Chem
from rdkit.Chem import Descriptors
def mw_of(s):
    m = Chem.MolFromSmiles(s)
    return Descriptors.MolWt(m) if m else None
mw = []
labels = []
for tool in df["Tool"]:
    sub = allp[allp["tool"] == tool]
    vals = sub["canonical_smiles"].map(mw_of).dropna().tolist()
    if vals:
        mw.append(vals)
        labels.append(tool)
ax2.boxplot(mw, labels=labels)
ax2.set_ylabel("Molecular weight (Da)")
ax2.set_title("MW distribution of unique generated molecules per tool")
ax2.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY, "warhead_mw_distributions.png"), dpi=150)
print("saved", os.path.join(SUMMARY, "warhead_mw_distributions.png"))
