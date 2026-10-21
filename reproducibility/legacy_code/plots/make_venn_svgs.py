#!/usr/bin/env python3
"""
SVG Venn diagrams for training vs external data:
  1. Scaffold overlap (full-degrader Murcko)
  2. Target overlap (POI UniProt)
  3. E3 ligase overlap
Output: revision/figures/venn_*
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures"
os.makedirs(FIG, exist_ok=True)

import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon(s):
    if not isinstance(s,str) or not s: return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
    except: return None

# Training
pdb = pd.read_csv('/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv',
                  low_memory=False, usecols=['PROTAC_SMILES','warhead_Uniprot','E3 ligase'])
pdb = pdb.dropna(subset=['PROTAC_SMILES']).copy()
pdb['scaffold'] = pdb['PROTAC_SMILES'].apply(murcko)
pdb['warhead_Uniprot'] = pdb['warhead_Uniprot'].astype(str).str.strip().str.upper()
pdb['E3 ligase'] = pdb['E3 ligase'].astype(str).str.strip().str.upper()
train_scaff = set(pdb['scaffold'].dropna())
train_targets = set(pdb['warhead_Uniprot'][pdb['warhead_Uniprot'] != 'NAN'])
train_e3 = set(pdb['E3 ligase'][pdb['E3 ligase'] != 'NAN'])

# External (2,640)
ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
ext = ext.drop_duplicates(subset=['SMILES']).copy()
ext['scaffold'] = ext['SMILES'].apply(murcko)
ext['POI_UniProt'] = ext['POI_UniProt'].astype(str).str.strip().str.upper()
ext['E3_Ligase'] = ext['E3_Ligase'].astype(str).str.strip().str.upper()
new_scaff = set(ext['scaffold'].dropna())
new_targets = set(ext['POI_UniProt'][ext['POI_UniProt'] != 'NAN'])
new_e3 = set(ext['E3_Ligase'][ext['E3_Ligase'] != 'NAN'])

# Temporal (215)
tmp = pd.read_csv(f"{REV}/temporal_external_protacs.csv", low_memory=False)
tmp['scaffold'] = tmp['SMILES'].apply(murcko)
tmp_scaff = set(tmp['scaffold'].dropna())

def venn_svg(a, b, label_a, label_b, fname, colors=('#2c7fb8','#d95f0e')):
    """Two-set Venn as inline SVG."""
    only_a = len(a - b); both = len(a & b); only_b = len(b - a)
    total_a = len(a); total_b = len(b)
    W, H = 640, 420
    cx1, cx2 = 260, 390
    cy = 210
    r = 145
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<circle cx="{cx1}" cy="{cy}" r="{r}" fill="{colors[0]}" fill-opacity="0.45" stroke="{colors[0]}" stroke-width="2"/>',
           f'<circle cx="{cx2}" cy="{cy}" r="{r}" fill="{colors[1]}" fill-opacity="0.45" stroke="{colors[1]}" stroke-width="2"/>',
           # labels
           f'<text x="{cx1-r-10}" y="{cy}" text-anchor="end" font-size="22" font-weight="bold" fill="{colors[0]}">{label_a}</text>',
           f'<text x="{cx1-r-10}" y="{cy+30}" text-anchor="end" font-size="18" fill="#333">n={total_a}</text>',
           f'<text x="{cx2+r+10}" y="{cy}" text-anchor="start" font-size="22" font-weight="bold" fill="{colors[1]}">{label_b}</text>',
           f'<text x="{cx2+r+10}" y="{cy+30}" text-anchor="start" font-size="18" fill="#333">n={total_b}</text>',
           # counts
           f'<text x="{cx1-r*0.55}" y="{cy}" text-anchor="middle" font-size="30" font-weight="bold">{only_a}</text>',
           f'<text x="{(cx1+cx2)/2}" y="{cy}" text-anchor="middle" font-size="30" font-weight="bold" fill="#333">{both}</text>',
           f'<text x="{cx2+r*0.55}" y="{cy}" text-anchor="middle" font-size="30" font-weight="bold">{only_b}</text>',
           f'<text x="{W/2}" y="{H-25}" text-anchor="middle" font-size="16" fill="#555">overlap (shared) = {both} | {label_a}-only = {only_a} | {label_b}-only = {only_b}</text>',
           '</svg>']
    with open(fname, 'w') as f:
        f.write('\n'.join(svg))
    print(f"  {fname}: only_a={only_a} both={both} only_b={only_b}")

print("Generating Venn diagrams...")
venn_svg(train_scaff, new_scaff, "Training scaffolds", "External scaffolds", f"{FIG}/venn_scaffolds.svg")
venn_svg(train_targets, new_targets, "Training targets", "External targets", f"{FIG}/venn_targets.svg")
venn_svg(train_e3, new_e3, "Training E3", "External E3", f"{FIG}/venn_e3.svg")
venn_svg(train_scaff, tmp_scaff, "Training scaffolds", "Temporal scaffolds", f"{FIG}/venn_scaffolds_temporal.svg")
print("\nDone — SVGs saved to", FIG)
EOF_MARKER = None
