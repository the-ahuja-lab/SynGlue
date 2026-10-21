#!/usr/bin/env python3
"""Leakage-control funnel: external pool -> exact(name) match -> Tanimoto cutoffs -> final novel set.
Uses external_pool_with_tmax.csv (has Tanimoto-max-to-training per compound) + training set (exact match).
Outputs funnel figure (svg/png) + steps CSV + MD.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
RDLogger.DisableLog('rdApp.*')

BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
REC=f"{REV}/rebuttal_analyses/15_count_reconciliation"
OUT=f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','figure.dpi':300,'savefig.bbox':'tight'})

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

pool=pd.read_csv(f"{REC}/external_pool_with_tmax.csv", low_memory=False)
tr=pd.read_csv(f"{S53}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
tr_train=set(c for c in map(canon, tr['PROTAC_SMILES'].dropna().astype(str)) if c)

mask=None
pc = pool['canonical_SMILES'].astype(str)
exact = pc.apply(canon).isin(tr_train)
tmax = pd.to_numeric(pool['tmax_sim_to_train'], errors='coerce')

N = len(pool)
steps = [
    ("All external pool", int(N)),
    ("Not exact\n(name) train match", int((N - exact.sum()))),
    ("Tanimoto < 0.90", int(((tmax < 0.90)).sum())),
    ("Tanimoto < 0.85", int(((tmax < 0.85)).sum())),
    ("Structurally novel\nTanimoto < 0.70", int(((tmax < 0.70)).sum())),
]
print("pool=",N,"exact matches=",int(exact.sum()))
for lbl,n in steps: print(f"  {lbl:30s} {n}")

# removed per step (difference, monotonic)
removed = [steps[0][1]-steps[i][1] for i in range(len(steps))]
step_txt = ["", "−306 exact", "−146 (0.90–1.0)", "−45 (0.85–0.90)", "−191 (0.70–0.85)"]

pd.DataFrame([{"stage":l,"count":n} for l,n in steps]).to_csv(f"{OUT}/leakage_funnel_steps.csv", index=False)

# ── funnel chart ──
fig, ax = plt.subplots(figsize=(6.5, 5.2))
levels = [n for _, n in steps]
labels = [l for l, _ in steps]
maxw = levels[0]
y_pos = np.arange(len(levels))[::-1]
widths = np.array(levels)/maxw
colors = ["#4C72B0","#8172B3","#55A868","#DD8452","#C44E52"]
for yi, (wi, n, lbl, rm) in enumerate(zip(widths, levels, labels, removed)):
    y = len(levels)-1-yi
    ax.barh(y, wi, height=0.62, color=colors[yi], alpha=0.9, edgecolor='white', left=(1-wi)/2)
    ax.text(0.5, y, f"{n:,}", ha='center', va='center', color='white', fontsize=11, weight='bold')
    ax.text(0.02, y+0.36, lbl, ha='left', va='bottom', fontsize=8.5, color='#111')
    if rm: ax.text(0.98, y-0.40, f"↓ remove {rm:,}", ha='right', va='top', fontsize=7, color='#555')
ax.set_xlim(0,1); ax.set_ylim(-0.7, len(levels)-0.1)
ax.invert_yaxis() if False else None
ax.axis('off')
ax.set_title("Leakage-control funnel — external PROTAC pool vs training set\n(exact + Tanimoto novelty)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/FUNNEL_leakage_control.svg")
fig.savefig(f"{OUT}/FUNNEL_leakage_control.png")
print("saved FUNNEL_leakage_control")
