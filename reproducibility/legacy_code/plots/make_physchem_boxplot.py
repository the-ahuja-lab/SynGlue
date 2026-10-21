#!/usr/bin/env python3
"""
Physchem property box plots: PROTAC-DB (training) vs other databases, per property,
with protocol-vs-other pairwise Mann-Whitney U (Wilcoxon rank-sum) statistics.
Layout: 2 rows x 3 cols (6 properties). Saved ONLY to figures/final_plot/.
Outputs:
  fig5_physchem_boxplot.png/.svg
  physchem_wilcoxon.csv
  physchem_results.md
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy import stats

RDLogger.DisableLog('rdApp.*')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
EXT = f"{BASE}/external_benchmark_data"
OUT = f"{REV}/figures/final_plot"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'figure.dpi': 300, 'savefig.bbox': 'tight'})

def props(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None: return None
    return {
        'MW': Descriptors.MolWt(m),
        'logP': Crippen.MolLogP(m),
        'TPSA': rdMolDescriptors.CalcTPSA(m),
        'HBD': rdMolDescriptors.CalcNumHBD(m),
        'HBA': rdMolDescriptors.CalcNumHBA(m),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(m),
    }

def canon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None

print("Loading data...")
tack = pd.read_parquet(f"{EXT}/tack_raw/data_train-00000-of-00001.parquet")
pdb_local = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/PROTAC_DB_with_features.csv", low_memory=False)
pat = pd.read_csv(f"{REV}/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv", low_memory=False)

tack_pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
tack_tpddb = tack[tack['Database'].str.contains('TPDdb', na=False)]
tack_db = tack[tack['Database'].str.contains('PROTAC-DB', na=False)]

DB = [
    ("PROTAC-DB (training)", pdb_local['Smiles'], 'Smiles'),
    ("TACK/PROTAC-DB 3.0", tack_db['SMILES'], 'SMILES'),
    ("TPDdb (TACK)", tack_tpddb['SMILES'], 'SMILES'),
    ("PROTACpedia", tack_pedia['SMILES'], 'SMILES'),
    ("PROTAC-Patent", pat['SMILES'], 'SMILES'),
]

# compute per-compound props (dedup by canonical, sample up to 2000)
proc = []
for name, series, _ in DB:
    seen = {}
    for s in series.dropna().astype(str):
        c = canon(s)
        if c is None or c in seen: continue
        seen[c] = True
        p = props(s)
        if p:
            p['database'] = name
            p['canon'] = c
            proc.append(p)
        if len(seen) >= 2000: break
dfp = pd.DataFrame(proc)
print("computed props:", len(dfp), "| per-db sizes:")
print(dfp['database'].value_counts().to_string())

PROP = [('MW', 'MW'), ('logP', 'logP'), ('TPSA', 'TPSA (Å²)'),
        ('RotB', 'Rotatable bonds'), ('HBA', 'HBA'), ('HBD', 'HBD')]

# pairwise ranks: PROTAC-DB (training) vs each other
from itertools import combinations
dbs = [d for d in dfp['database'].unique()]
ref = "PROTAC-DB (training)"
other = [d for d in dbs if d != ref]
rows = []
for col, _ in PROP:
    for o in other:
        x = dfp.loc[dfp['database'] == ref, col].dropna().values
        y = dfp.loc[dfp['database'] == o, col].dropna().values
        U, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        n1, n2 = len(x), len(y)
        eff = 2 * U / (n1 * n2) - 1
        rows.append(dict(property=col, group_a=ref, group_b=o, n_a=n1, n_b=n2,
                         median_a=round(float(np.median(x)), 2), median_b=round(float(np.median(y)), 2),
                         U=round(float(U), 1), p_value=p,
                         rank_biserial=round(float(eff), 3)))
wt = pd.DataFrame(rows)
wt['p_bonferroni'] = wt['p_value'] * 4  # 4 comparisons per property
wt['signif_0.05'] = wt['p_bonferroni'] < 0.05
wt.to_csv(f"{OUT}/physchem_wilcoxon.csv", index=False)

# ─── Box plots: 2 rows x 3 cols ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors = {'PROTAC-DB (training)': '#d62728',
          'TACK/PROTAC-DB 3.0': '#1f77b4',
          'TPDdb (TACK)': '#ff7f0e',
          'PROTACpedia': '#2ca02c',
          'PROTAC-Patent': '#9467bd'}
for ax, (col, title) in zip(axes.flat, PROP):
    xpos = []
    for i, d in enumerate(dbs):
        v = dfp.loc[dfp['database'] == d, col].dropna().values
        xpos.append((i, v, d))
    labels = [d for d in dbs]
    bp = ax.boxplot([v for _, v, _ in xpos], positions=[i for i, _, _ in xpos],
                    widths=0.55, patch_artist=True, medianprops=dict(color='black', lw=1.2),
                    showfliers=False)
    for patch, (_, _, d) in zip(bp['boxes'], xpos):
        patch.set_facecolor(colors[d]); patch.set_alpha(0.8)
    # jitter
    rng = np.random.RandomState(42)
    for i, (_, v, d) in enumerate(xpos):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, alpha=0.2, color='gray')
    ax.set_xticks([i for i, _, _ in xpos])
    short = [d.replace(' (training)', ' (train)').replace('PROTAC-DB 3.0', 'PDB3.0')
             .replace('PROTAC-DB (train)', 'PDB-train').replace('TACK/PROTAC-DB 3.0', 'TACK/PDB3.0')
             .replace('PROTACpedia', 'Pedia').replace('PROTAC-Patent', 'Patent')
             .replace('TPDdb (TACK)', 'TPDdb') for d in labels]
    ax.set_xticklabels(short, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel(title)
    ax.set_title(f"{title} — n={[len(v) for _, v, _ in xpos]}")
fig.suptitle('Physicochemical properties by database (box + jitter)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/fig5_physchem_boxplot.png")
fig.savefig(f"{OUT}/fig5_physchem_boxplot.svg")
plt.close(fig)

# ─── results MD ──────────────────────────────────────
lines = ["# Physchem box plots + Wilcoxon (PROTAC-DB training vs other databases)",
         "",
         f"- Method: box plots per property over 5 databases (sampled ≤2,000/db, dedup); two-sample Mann-Whitney U (Wilcoxon rank-sum), Bonferroni (×4 per property)",
         f"- Saved only to: `figures/final_plot/` (fig5_physchem_boxplot, physchem_wilcoxon.csv)",
         "",
         "## Pairwise stats (PROTAC-DB training vs each other; p=Bonferroni)", ""]
for col, _ in PROP:
    lines.append(f"### {col}")
    lines.append("| vs | U | p (adj) | signif | rank-biserial | median(other) | median(PDB-train) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in wt[wt['property'] == col].iterrows():
        lines.append(f"| {r['group_b']} | {r['U']:.0f} | {r['p_bonferroni']:.3g} | {'*' if r['signif_0.05'] else ''} | {r['rank_biserial']} | {r['median_b']} | {r['median_a']} |")
    lines.append("")
open(f"{OUT}/physchem_results.md", "w").write("\n".join(lines))

print(wt.to_string(index=False))
print(f"\nSaved to {OUT}: fig5_physchem_boxplot.png/.svg, physchem_wilcoxon.csv, physchem_results.md")
