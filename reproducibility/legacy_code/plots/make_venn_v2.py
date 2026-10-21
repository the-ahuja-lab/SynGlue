#!/usr/bin/env python3
"""
Proper Venn diagrams + counts — Training (PROTAC-DB) vs External sources (TACK/TPDdb/PROTACpedia/PatentDB/temporal)
Shows: compound-level overlap, DC50-record overlap, Dmax-record overlap
Output: revision/figures/venn_*_v2.svg
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures"
os.makedirs(FIG, exist_ok=True)

import pandas as pd, numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon(s):
    if not isinstance(s,str) or not s: return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None

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
    print(f"  {os.path.basename(fname)}: {label_a}={a_total}, {label_b}={b_total}, both={both}")

# ── Data ──
train = pd.read_csv('/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv', low_memory=False)
train['canon'] = train['PROTAC_SMILES'].apply(canon)
tr_canon = set(train['canon'].dropna())
tr_dc50 = set(train.loc[train['DC50 (nM)'].notna(),'canon'])
tr_dmax = set(train.loc[train['Dmax (%)'].notna(),'canon'])

tack = pd.read_parquet('/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet')
tack['canon'] = tack['SMILES'].apply(canon)

def sets_for(df):
    c = set(df['canon'].dropna())
    dc = set(df.loc[df['Value_Type']=='DC50','canon'])
    dm = set(df.loc[df['Value_Type']=='Dmax','canon'])
    return c, dc, dm

print("Generating Venn diagrams (compound level)...")
# 1. Training vs TACK full (compounds)
c_tack, _, _ = sets_for(tack)
venn2_svg(len(tr_canon), len(c_tack), len(tr_canon & c_tack),
          "Training (PROTAC-DB)", "TACK (external)", f"{FIG}/venn_train_vs_tack_compounds.svg",
          sub="Compound-level overlap (unique canonical SMILES)")

# 2. Training vs PROTACpedia
pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
c_pedia, _, _ = sets_for(pedia)
venn2_svg(len(tr_canon), len(c_pedia), len(tr_canon & c_pedia),
          "Training (PROTAC-DB)", "PROTACpedia", f"{FIG}/venn_train_vs_pedia_compounds.svg",
          sub="Compound-level overlap (unique canonical SMILES)")

# 3. Training vs TPDdb
tpddb = tack[tack['Database'].str.contains('TPDdb', na=False)]
c_tpddb, _, _ = sets_for(tpddb)
venn2_svg(len(tr_canon), len(c_tpddb), len(tr_canon & c_tpddb),
          "Training (PROTAC-DB)", "TPDdb", f"{FIG}/venn_train_vs_tpddb_compounds.svg",
          sub="Compound-level overlap (unique canonical SMILES)")

# 4. Training vs PatentDB
pat = pd.read_csv(f"{REV}/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv", low_memory=False)
pat['canon'] = pat['SMILES'].apply(canon)
c_pat = set(pat['canon'].dropna())
venn2_svg(len(tr_canon), len(c_pat), len(tr_canon & c_pat),
          "Training (PROTAC-DB)", "PROTAC-PatentDB", f"{FIG}/venn_train_vs_patent_compounds.svg",
          sub="Compound-level overlap (unique canonical SMILES)")

# 5. Temporal
tmp = pd.read_csv(f"{REV}/temporal_external_protacs.csv", low_memory=False)
tmp['canon'] = tmp['SMILES'].apply(canon)
c_tmp = set(tmp['canon'].dropna())
venn2_svg(len(tr_canon), len(c_tmp), len(tr_canon & c_tmp),
          "Training (PROTAC-DB)", "Temporal 2021-24", f"{FIG}/venn_train_vs_temporal_compounds.svg",
          sub="Compound-level overlap (unique canonical SMILES)")

# 6. DC50-activity overlap: Training vs TACK (after exclusion conceptually)
# Show raw DC50-label overlap (before exclusion) to justify exclusion
dc_tack = set(tack.loc[tack['Value_Type']=='DC50','canon'])
venn2_svg(len(tr_dc50), len(dc_tack), len(tr_dc50 & dc_tack),
          "Training DC50-labeled", "TACK DC50-labeled", f"{FIG}/venn_train_vs_tack_DC50.svg",
          sub="DC50-label overlap — overlap MUST be excluded before evaluation")

# 7. Dmax
dm_tack = set(tack.loc[tack['Value_Type']=='Dmax','canon'])
venn2_svg(len(tr_dmax), len(dm_tack), len(tr_dmax & dm_tack),
          "Training Dmax-labeled", "TACK Dmax-labeled", f"{FIG}/venn_train_vs_tack_Dmax.svg",
          sub="Dmax-label overlap — overlap MUST be excluded before evaluation")

print("\nDone — SVGs saved to", FIG)
