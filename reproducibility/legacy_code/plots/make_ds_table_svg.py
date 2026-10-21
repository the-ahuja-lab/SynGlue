#!/usr/bin/env python3
"""Clear-cut SVG table: Dataset x (Records, Unique, Scaffolds, Targets, E3, Assay)
Rows: PROTAC-DB(3.0), TACK, TPDdb, PROTACpedia, PROTAC-Patent, External not-in-training (final).
Real text in SVG (svg.fonttype=path disabled).
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
RDLogger.DisableLog('rdApp.*')

BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs"; os.makedirs(OUT, exist_ok=True)

for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none'})

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def murcko(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
def nuniq(s): return len(set(c for c in map(canon,s.dropna().astype(str)) if c))
def nscaf(s): return len(set(c for c in map(murcko,s.dropna().astype(str)) if c))

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tack=pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
pat=pd.read_csv(f"{DATA}/merged_patent_pubchem_only.csv", low_memory=False)
ext=pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
tp=tack[tack['Database'].str.contains('TPDdb',na=False)]
pe=tack[tack['Database'].str.contains('PROTACpedia',na=False)]

def assemble(name, df, smi, targets_col, e3_col, assay_col):
    tgt = df[targets_col].nunique() if targets_col and targets_col in df.columns else 0
    e3  = df[e3_col].nunique() if e3_col and e3_col in df.columns else 0
    asy = df[assay_col].nunique() if assay_col and assay_col in df.columns and df[assay_col].notna().any() else 0
    return [name, f"{len(df):,}", f"{nuniq(df[smi]):,}", f"{nscaf(df[smi]):,}",
            f"{tgt:,}", f"{e3:,}", f"{asy:,}"]

rows = [
 assemble("PROTAC-DB (3.0)", pdb, 'Smiles', 'Target', 'E3 ligase', 'Assay (DC50/Dmax)'),
 assemble("TACK", tack, 'SMILES', 'POI_UniProt', 'Ligase_Name', 'Assay'),
 assemble("TPDdb", tp, 'SMILES', 'POI_UniProt', 'Ligase_Name', 'Assay'),
 assemble("PROTACpedia", pe, 'SMILES', 'POI_UniProt', 'Ligase_Name', 'Assay'),
 assemble("PROTAC-Patent", pat, 'canonical_SMILES', 'Target', None, None),
 assemble("External not-in-training (final)", ext.drop_duplicates('SMILES'), 'SMILES',
           'POI_UniProt', 'E3_Ligase', 'Assay'),
]
for r in rows: print(r)
cols=["Dataset","Records","Unique","Scaffolds","Targets","E3 ligases","Assay"]

# ── build SVG table ──
ncol, nrow = len(cols), len(rows)
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis('off')
cell_text = [cols] + rows
tab = ax.table(cellText=cell_text, loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.7)
# style
header_color = '#4C72B0'; alt = '#EEF1F7'
for (r,c), cell in tab.get_celld().items():
    cell.set_edgecolor('white'); cell.set_linewidth(1.2)
    if r==0:
        cell.set_facecolor(header_color); cell.set_text_props(color='white', weight='bold')
    else:
        if r%2==0: cell.set_facecolor(alt)
        if c==0: cell.set_text_props(weight='bold')
# widen first column
for r in range(nrow+1):
    tab[(r,0)].set_width(0.30)
fig.tight_layout()
fig.savefig(f"{OUT}/DS_TABLE_final_external.svg")
fig.savefig(f"{OUT}/DS_TABLE_final_external.png", dpi=300)
print("saved DS_TABLE_final_external.svg/.png")
print("text elements in svg:", int(__import__('subprocess').check_output(['grep','-o','<text',f'{OUT}/DS_TABLE_final_external.svg']).count(b'<text')))
