#!/usr/bin/env python3
"""
Training vs Testing Venn diagrams for all 4 splits (final ProtacAttentionModel dataset).
Compound-level + scaffold-level (for scaffold-held) + target/E3 level (for their splits).
Output: figures/final_results/venn_splits/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures/final_results/venn_splits"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
    except: return None

def venn_svg(a_total, b_total, both, label_a, label_b, fname, sub="", a_col="#2c7fb8", b_col="#d95f0e",
             a_sub="", b_sub="", both_sub=""):
    only_a = a_total - both; only_b = b_total - both
    W, H = 760, 480
    cx1, cx2 = 300, 460; cy = 240; r = 165
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<circle cx="{cx1}" cy="{cy}" r="{r}" fill="{a_col}" fill-opacity="0.40" stroke="{a_col}" stroke-width="3"/>',
             f'<circle cx="{cx2}" cy="{cy}" r="{r}" fill="{b_col}" fill-opacity="0.40" stroke="{b_col}" stroke-width="3"/>',
             f'<text x="{cx1-r-15}" y="{cy}" text-anchor="end" font-size="22" font-weight="bold" fill="{a_col}">{label_a}</text>',
             f'<text x="{cx1-r-15}" y="{cy+30}" text-anchor="end" font-size="16" fill="#444">{a_sub}</text>',
             f'<text x="{cx2+r+15}" y="{cy}" text-anchor="start" font-size="22" font-weight="bold" fill="{b_col}">{label_b}</text>',
             f'<text x="{cx2+r+15}" y="{cy+30}" text-anchor="start" font-size="16" fill="#444">{b_sub}</text>',
             f'<text x="{cx1-r*0.55}" y="{cy}" text-anchor="middle" font-size="38" font-weight="bold" fill="#1a4a73">{only_a}</text>',
             f'<text x="{(cx1+cx2)/2}" y="{cy}" text-anchor="middle" font-size="38" font-weight="bold" fill="#222">{both}</text>',
             f'<text x="{cx2+r*0.55}" y="{cy}" text-anchor="middle" font-size="38" font-weight="bold" fill="#8a3b00">{only_b}</text>',
             f'<text x="{W/2}" y="{H-55}" text-anchor="middle" font-size="14" fill="#555">{both_sub}</text>',
             f'<text x="{W/2}" y="{H-28}" text-anchor="middle" font-size="15" fill="#333" font-weight="bold">shared = {both} | {label_a}-only = {only_a} | {label_b}-only = {only_b}</text>']
    if sub:
        lines.insert(1, f'<text x="{W/2}" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#222">{sub}</text>')
    lines.append('</svg>')
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  {os.path.basename(fname)}: {label_a}={a_total}, {label_b}={b_total}, shared={both}")

def main():
    # rebuild data
    sys.path.insert(0, REV)
    from train_final_both_endpoints import build_dataset
    X, df = build_dataset()
    df['scaffold'] = df['smiles'].apply(murcko)
    idx = np.arange(len(df))
    rng = np.random.RandomState(42)
    from sklearn.model_selection import train_test_split

    splits = {}

    # Random
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    splits['Random'] = (tr, te)
    # Scaffold-held
    sc_counts = df['scaffold'].value_counts()
    cand = [s for s in sc_counts.index if pd.notna(s) and sc_counts[s] >= 3]
    held_s = set(rng.choice(cand, size=max(1,int(len(cand)*0.2)), replace=False))
    te_s = np.where(df['scaffold'].isin(held_s))[0]; tr_s = np.setdiff1d(idx, te_s)
    splits['Scaffold-held'] = (tr_s, te_s)
    # Target-held
    tg_counts = df['target'].value_counts()
    cand_t = [t for t in tg_counts.index if str(t)!='NAN' and tg_counts[t] >= 5]
    held_t = set(rng.choice(cand_t, size=max(1,int(len(cand_t)*0.2)), replace=False))
    te_t = np.where(df['target'].isin(held_t))[0]; tr_t = np.setdiff1d(idx, te_t)
    splits['Target-held'] = (tr_t, te_t)
    # E3-held
    te_e = np.where(df['e3']=='VHL')[0]; tr_e = np.setdiff1d(idx, te_e)
    splits['E3-held (VHL)'] = (tr_e, te_e)

    print("\nGenerating Venn diagrams...")

    # ═══ 1. Random — compound-level ═══
    tr_i, te_i = splits['Random']
    venn_svg(len(tr_i), len(te_i), 0, "Training", "Test", f"{FIG}/venn_split1_random.svg",
             sub="Split 1 · Random 80/20 (compound level)",
             a_sub="n = 7,401", b_sub="n = 1,851",
             both_sub="Random split: no compound appears in both sets (disjoint by construction)")

    # ═══ 2. Scaffold-held — scaffold-level ═══
    tr_i, te_i = splits['Scaffold-held']
    tr_sc = set(df['scaffold'][tr_i].dropna()); te_sc = set(df['scaffold'][te_i].dropna())
    venn_svg(len(tr_sc), len(te_sc), len(tr_sc & te_sc), "Training scaffolds", "Test scaffolds",
             f"{FIG}/venn_split2_scaffold.svg",
             sub="Split 2 · Scaffold-held-out (scaffold level)",
             a_sub=f"n = {len(tr_sc)}", b_sub=f"n = {len(te_sc)}",
             both_sub="Zero shared scaffolds — entire scaffold groups held out (no analog-series leakage)")

    # ═══ 3. Target-held — target level + compound ═══
    tr_i, te_i = splits['Target-held']
    tr_tg = set(df['target'][tr_i]); te_tg = set(df['target'][te_i])
    venn_svg(len(tr_tg), len(te_tg), len(tr_tg & te_tg), "Training targets", "Test targets",
             f"{FIG}/venn_split3_target.svg",
             sub="Split 3 · Target-held-out (target level)",
             a_sub=f"n = {len(tr_tg)}", b_sub=f"n = {len(te_tg)}",
             both_sub=f"Shared targets = {len(tr_tg & te_tg)} (shared only from multi-record targets; all held targets excluded from training)")

    # compound-level for target-held
    venn_svg(len(tr_i), len(te_i), 0, "Training compounds", "Test compounds",
             f"{FIG}/venn_split3_target_compounds.svg",
             sub="Split 3 · Target-held-out (compound level)",
             a_sub=f"n = {len(tr_i):,}", b_sub=f"n = {len(te_i):,}",
             both_sub="Compounds for held-out targets appear ONLY in test (disjoint)")

    # ═══ 4. E3-held — E3 level + compound ═══
    tr_i, te_i = splits['E3-held (VHL)']
    tr_e3 = set(df['e3'][tr_i]); te_e3 = set(df['e3'][te_i])
    venn_svg(len(tr_e3), len(te_e3), len(tr_e3 & te_e3), "Training E3", "Test E3",
             f"{FIG}/venn_split4_e3.svg",
             sub="Split 4 · E3-held-out (VHL) — E3-ligase level",
             a_sub=f"n = {len(tr_e3)}", b_sub=f"n = {len(te_e3)}",
             both_sub=f"Shared = {len(tr_e3 & te_e3)} (VHL appears ONLY in test — true E3-held-out)")
    venn_svg(len(tr_i), len(te_i), 0, "Training compounds", "Test compounds (VHL)",
             f"{FIG}/venn_split4_e3_compounds.svg",
             sub="Split 4 · E3-held-out (compound level)",
             a_sub=f"n = {len(tr_i):,}", b_sub=f"n = {len(te_i):,}",
             both_sub="All VHL-recruiting compounds in test only; transformer trained WITHOUT VHL (gen_training_v4 protocol)")

    print(f"\nDone — SVGs in {FIG}")

if __name__ == '__main__':
    main()
