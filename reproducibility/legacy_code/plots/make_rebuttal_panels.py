#!/usr/bin/env python3
"""
Final rebuttal panels:
  Panel A (SVG): novelty/leakage Venns on FULL curated set
     A1 training vs external scaffolds
     A2 training vs temporal scaffolds
     A3 training vs external targets
     A4 training vs external E3s
  Panel B (table): endpoint availability per split (Training/Temporal/External novel-scaffold)
  Panel C (side table): Have DC50 / Have Dmax / Both / Neither
Output: revision/figures/rebuttal_panelA_*.svg + revision/rebuttal_panels.md
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures"
os.makedirs(FIG, exist_ok=True)

import pandas as pd, numpy as np

def venn2_svg(a_total, b_total, both, label_a, label_b, fname, sub=""):
    only_a = a_total - both; only_b = b_total - both
    W, H = 700, 460
    cx1, cx2 = 285, 420; cy = 230; r = 160
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<circle cx="{cx1}" cy="{cy}" r="{r}" fill="#2c7fb8" fill-opacity="0.40" stroke="#2c7fb8" stroke-width="2.5"/>',
             f'<circle cx="{cx2}" cy="{cy}" r="{r}" fill="#d95f0e" fill-opacity="0.40" stroke="#d95f0e" stroke-width="2.5"/>',
             f'<text x="{cx1-r-15}" y="{cy}" text-anchor="end" font-size="21" font-weight="bold" fill="#2c7fb8">{label_a}</text>',
             f'<text x="{cx1-r-15}" y="{cy+28}" text-anchor="end" font-size="17" fill="#333">n = {a_total}</text>',
             f'<text x="{cx2+r+15}" y="{cy}" text-anchor="start" font-size="21" font-weight="bold" fill="#d95f0e">{label_b}</text>',
             f'<text x="{cx2+r+15}" y="{cy+28}" text-anchor="start" font-size="17" fill="#333">n = {b_total}</text>',
             f'<text x="{cx1 - r*0.55}" y="{cy}" text-anchor="middle" font-size="34" font-weight="bold" fill="#1a4a73">{only_a}</text>',
             f'<text x="{(cx1+cx2)/2}" y="{cy}" text-anchor="middle" font-size="34" font-weight="bold" fill="#333">{both}</text>',
             f'<text x="{cx2 + r*0.55}" y="{cy}" text-anchor="middle" font-size="34" font-weight="bold" fill="#8a3b00">{only_b}</text>',
             f'<text x="{W/2}" y="{H-40}" text-anchor="middle" font-size="15" fill="#444">shared = {both}  |  {label_a}-only = {only_a}  |  {label_b}-only = {only_b}</text>']
    if sub:
        lines.insert(1, f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" fill="#222" font-weight="bold">{sub}</text>')
    lines.append('</svg>')
    with open(fname,'w') as f:
        f.write('\n'.join(lines))
    print(f"  {os.path.basename(fname)}: {label_a}={a_total}, {label_b}={b_total}, shared={both}")

# ===== Counts (from full-set computation) =====
# Training
tr_compounds, tr_scaff, tr_targets, tr_e3 = 5203, 4880, 190, 20
# External full 2,640
ex_scaff_total, ex_scaff_shared = 1930, 70
ex_targets_total, ex_targets_shared = 53, 44
ex_e3_total, ex_e3_shared = 4, 3
# Temporal
tp_scaff_total, tp_scaff_shared = 103, 2

print("Panel A — Venn diagrams (FULL curated set, novelty characterization):")
venn2_svg(tr_scaff, ex_scaff_total, ex_scaff_shared,
          "Training scaffolds", "External scaffolds", f"{FIG}/rebuttal_panelA1_scaffolds.svg",
          "A1 · Scaffold novelty (full 5,203 vs 2,640 compounds)")
venn2_svg(tr_scaff, tp_scaff_total, tp_scaff_shared,
          "Training scaffolds", "Temporal scaffolds", f"{FIG}/rebuttal_panelA2_temporal_scaffolds.svg",
          "A2 · Temporal scaffold novelty (full 5,203 vs 215)")
venn2_svg(tr_targets, ex_targets_total, ex_targets_shared,
          "Training targets", "External targets", f"{FIG}/rebuttal_panelA3_targets.svg",
          "A3 · Target novelty (full set, 190 vs 53)")
venn2_svg(tr_e3, ex_e3_total, ex_e3_shared,
          "Training E3 ligases", "External E3 ligases", f"{FIG}/rebuttal_panelA4_e3.svg",
          "A4 · E3 ligase novelty (full set, 20 vs 4)")

print("\nPanel A complete.")

# ===== Panel B table data =====
# Training endpoint availability (from earlier analysis)
# Unique compounds with labels: DC50 1291 unique / Dmax 702 unique; records: 2446 DC50 / 1728 Dmax (manuscript), parsed 2433/1347
# External novel-scaffold 2,555: DC50 2215 / Dmax 1169 / both 832
# Temporal 215: mostly no endpoints (2 DC50, 6 Dmax from TACK)
rows = [
    {"Split": "Training (PROTAC-DB)", "Unique compounds": 5203,
     "DC50 records": 2446, "Dmax records": 1728,
     "Compounds w/ DC50": 1291, "Compounds w/ Dmax": 702, "Both": 519},
    {"Split": "Temporal 2021–24", "Unique compounds": 215,
     "DC50 records": 2, "Dmax records": 6,
     "Compounds w/ DC50": 2, "Compounds w/ Dmax": 6, "Both": 0},
    {"Split": "External novel-scaffold", "Unique compounds": 2555,
     "DC50 records": 2912, "Dmax records": 1590,
     "Compounds w/ DC50": 2215, "Compounds w/ Dmax": 1169, "Both": 832},
    {"Split": "External all (leakage-free)", "Unique compounds": 2640,
     "DC50 records": 2963, "Dmax records": 1678,
     "Compounds w/ DC50": 2248, "Compounds w/ Dmax": 1245, "Both": 857},
]
panelB = pd.DataFrame(rows)
panelB.to_csv(f"{REV}/rebuttal_panelB_endpoint_table.csv", index=False)
print("\nPanel B table saved: rebuttal_panelB_endpoint_table.csv")
print(panelB.to_string(index=False))

# ===== Panel C (side table) =====
rowsC = [
    {"Endpoint status": "Have DC50", "Count": 2248, "Pct": 85.2},
    {"Endpoint status": "DC50 only", "Count": 1391, "Pct": 52.7},
    {"Endpoint status": "Both DC50 + Dmax", "Count": 857, "Pct": 32.5},
    {"Endpoint status": "Have Dmax", "Count": 1245, "Pct": 47.2},
    {"Endpoint status": "Dmax only", "Count": 388, "Pct": 14.7},
    {"Endpoint status": "Neither", "Count": 4, "Pct": 0.2},
]
panelC = pd.DataFrame(rowsC)
panelC.to_csv(f"{REV}/rebuttal_panelC_endpoint_availability.csv", index=False)
print("\nPanel C table saved: rebuttal_panelC_endpoint_availability.csv")
print(panelC.to_string(index=False))
