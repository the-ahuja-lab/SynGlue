#!/usr/bin/env python3
"""Corrected per-source Tanimoto distributions + patent funnel + source table + full OOD story.
Patent uses the FULL official file (57,533 unique). All figures = SVG with real text.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
RDLogger.DisableLog('rdApp.*')
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
PAT=f"{REV}/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight'})
SRC_C={"TACK":"#4C72B0","TPDdb":"#55A868","PROTACpedia":"#C44E52","PROTAC-Patent":"#8172B3"}
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def fp(s):
    m=Chem.MolFromSmiles(str(s)); return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_uniq=set(c for c in map(canon,pdb['Smiles'].dropna().astype(str)) if c)
tr_fps=[f for f in map(fp,pdb['Smiles'].dropna().astype(str)) if f is not None]
tack=pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
tp=tack[tack['Database'].str.contains('TPDdb',na=False)]; pe=tack[tack['Database'].str.contains('PROTACpedia',na=False)]
pat=pd.read_csv(PAT, low_memory=False, usecols=['SMILES','Target','InChI Key'])

def tmax_series(df, smi):
    seen={}; vals=[]
    for s in df[smi].dropna().astype(str):
        c=canon(s)
        if c is None or c in seen: continue
        seen[c]=1
        f=fp(s)
        tm=max(DataStructs.BulkTanimotoSimilarity(f,tr_fps)) if f is not None else np.nan
        vals.append(tm)
    return np.array(vals,dtype=float)

print("computing per-source tmax...", flush=True)
src_tmax={"TACK":tmax_series(tack,'SMILES'),"TPDdb":tmax_series(tp,'SMILES'),
          "PROTACpedia":tmax_series(pe,'SMILES'),"PROTAC-Patent":tmax_series(pat,'SMILES')}
for k,v in src_tmax.items(): print(f"  {k}: n={len(v)} <0.7={int(np.sum(v<0.7))}", flush=True)
np.savez(f"{OUT}/per_source_tmax.npz", **src_tmax)

# ── 1. Per-source Tanimoto histogram (SVG) ──
fig,ax=plt.subplots(figsize=(7.5,4.6))
bins=np.linspace(0,1,51)
for src in ["PROTAC-Patent","TACK","PROTACpedia","TPDdb"]:
    v=src_tmax[src]; mask=~np.isnan(v)
    ax.hist(v[mask],bins=bins,alpha=0.5,density=True,color=SRC_C[src],label=f"{src} (n={len(v):,})",edgecolor='white')
ax.axvline(0.70,color='#222',ls='--',lw=1.2); ax.text(0.70,ax.get_ylim()[1]*0.9,'0.70',fontsize=8)
ax.set_xlabel("Max Tanimoto similarity to PROTAC-DB 3.0 (Morgan r2, 2048 bit)")
ax.set_ylabel("density"); ax.set_title("Per-source novelty vs PROTAC-DB 3.0 training")
ax.legend(fontsize=7,frameon=False); ax.set_xlim(0,1.05)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_per_source_Tanimoto.svg"); fig.savefig(f"{OUT}/FIG_per_source_Tanimoto.png"); plt.close(fig)

# ── 2. Patent funnel (SVG) ──
pt=tmax_series(pat,'SMILES')
steps=[("All patent unique",len(pt)),
       ("Not exact\n(name) train match",int(np.sum(~pd.Series([False]*len(pt)).values) ) )]
# recompute exact per unique
seen={}; exactcnt=0
for s in pat['SMILES'].dropna().astype(str):
    c=canon(s)
    if c is None or c in seen: continue
    seen[c]=1
    if c in tr_uniq: exactcnt+=1
lev=[("All patent unique", len(pt)),
     ("Not exact(name) train match", len(pt)-exactcnt),
     ("Tanimoto < 0.90", int(np.sum(pt<0.90))),
     ("Tanimoto < 0.85", int(np.sum(pt<0.85))),
     ("Structurally novel\nTanimoto < 0.70", int(np.sum(pt<0.70)))]
print("patent funnel:", lev, flush=True)
levels=[n for _,n in lev]; labels=[l for l,_ in lev]; mx=levels[0]
fig,ax=plt.subplots(figsize=(6.5,5.2))
for yi,(wi,n,lbl) in enumerate(zip([n/mx for n in levels],levels,labels)):
    y=len(levels)-1-yi
    ax.barh(y,wi,height=0.62,color=plt.cm.Blues(0.4+0.5*yi/max(1,len(levels)-1)),edgecolor='white',left=(1-wi)/2)
    ax.text(0.5,y,f"{n:,}",ha='center',va='center',color='white',fontsize=10,weight='bold')
    ax.text(0.02,y+0.36,lbl,ha='left',va='bottom',fontsize=8.5,color='#111')
capture={}
ax.set_xlim(0,1); ax.set_ylim(-0.7,len(levels)-0.1); ax.axis('off')
ax.set_title("PROTAC-PatentDB — leakage funnel vs PROTAC-DB 3.0\n(63,136 rows → 57,533 unique)",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_patent_funnel.svg"); fig.savefig(f"{OUT}/FIG_patent_funnel.png"); plt.close(fig)

# ── 3. Source table (SVG) ──
rows=[
 ["PROTAC-DB (3.0)","9,380","6,110","5,698","441","21","527"],
 ["TACK","6,561","3,514","2,723","109","9","21"],
 ["TPDdb","4,293","2,426","1,753","23","2","21"],
 ["PROTACpedia","871","585","529","60","7","0"],
 ["PROTAC-Patent (full)","63,136","57,533","–","223","0*","0*"],
 ["External not-in-train (activity)","–","2,562","–","–","–","21"],
]
cols=["Source","Records","Unique","Scaffolds","Targets","E3","Assay"]
fig,ax=plt.subplots(figsize=(10,3.2)); ax.axis('off')
tab=ax.table(cellText=[cols]+rows,loc='center',cellLoc='center'); tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1,1.6)
for (r,c),cell in tab.get_celld().items():
    cell.set_edgecolor('white'); cell.set_linewidth(1.2)
    if r==0: cell.set_facecolor('#4C72B0'); cell.set_text_props(color='white',weight='bold')
    elif r%2==0: cell.set_facecolor('#EEF1F7')
    if c==0 and r>0: cell.set_text_props(weight='bold')
fig.tight_layout(); fig.savefig(f"{OUT}/TABLE_source_final.svg"); fig.savefig(f"{OUT}/TABLE_source_final.png"); plt.close(fig)
print("saved figures + table + per_source_tmax.npz")
