#!/usr/bin/env python3
"""Corrected FINAL-EXTERNAL funnel (activity set) + Tanimoto plot of the filtered remaining set.
Flow: All external unique (TACK + PROTAC-Patent) -> [DROP no-activity patent] -> TACK unique
-> [-exact PROTAC-DB3] -> novel -> Tanimoto <0.9/0.85/0.7 -> final activity test set.
Plus: Tanimoto-to-DB3 distribution of the REMAINING (filtered) final external set.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
RDLogger=None
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight'})

# TACK novel compounds with tmax (from the derivation that used correct TACK, old-patent)
fm=pd.read_csv(f"{OUT}/final_external_with_meta.csv")
tack_novel=fm[fm['source']=='TACK'].copy()
pv=pd.to_numeric(tack_novel['tmax'], errors='coerce').values
n_tack_novel=len(tack_novel)
print("TACK novel (activity-capable):", n_tack_novel)

# TACK unique total + exact matches
z=np.load(f"{OUT}/per_source_tmax.npz")
tack_all=z['TACK']; pat_all=z['PROTAC-Patent']
n_tack=len(tack_all); n_pat=len(pat_all)
n_exact=952  # from derivation (TACK exact-in-DB3)
n_pat62=len(pat_all)

# overlap TACK vs patent (exact canonical) for the union top stage
from rdkit import Chem
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
tack=pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
P='/storage/savi/saveenas/Projects/SynGlue/revision/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv'
pat=pd.read_csv(P, low_memory=False, usecols=['SMILES'])
tack_set={c for c in map(canon,tack['SMILES'].dropna().astype(str)) if c}
ovl=0; pat_set_unique=set()
for s in pat['SMILES'].dropna().astype(str):
    c=canon(s)
    if c is None: continue
    if c not in pat_set_unique:
        pat_set_unique.add(c)
        if c in tack_set: ovl+=1
union=len(pat_set_unique | tack_set) if False else len(pat_set_unique)+n_tack-ovl
print(f"overlap TACK∩Patent: {ovl} | union: {union}")

# funnel stages (activity final external)
l090=int(np.sum(pv<0.90)); l085=int(np.sum(pv<0.85)); l070=int(np.sum(pv<0.70))
stages=[("All external unique\n(TACK + Patent)", union),
        ("DROP: no activity endpoint\n(Patent {:,} removed)".format(n_pat), n_tack),
        ("DROP: exact PROTAC-DB3 matches\n(−{:,} in training)".format(n_exact), n_tack_novel),
        ("Tanimoto < 0.90", l090),
        ("Tanimoto < 0.85", l085),
        ("FINAL external activity set\nTanimoto < 0.70", l070)]
print("funnel:", [(l,n) for l,n in stages])

# ── 1) funnel plot ──
levels=[n for _,n in stages]; labels=[l for l,_ in stages]; mx=levels[0]
fig,ax=plt.subplots(figsize=(7.0,5.6))
cols=plt.cm.RdYlBu_r(np.linspace(0.15,0.85,len(levels)))
for yi,(wi,n,lbl) in enumerate(zip([n/mx for n in levels],levels,labels)):
    y=len(levels)-1-yi
    ax.barh(y,wi,height=0.6,color=cols[yi],edgecolor='white',left=(1-wi)/2)
    ax.text(0.5,y,f"{n:,}",ha='center',va='center',color='white',fontsize=10,weight='bold')
    ax.text(0.02,y+0.34,lbl,ha='left',va='bottom',fontsize=8,color='#111')
ax.set_xlim(0,1); ax.set_ylim(-0.7,len(levels)-0.1); ax.axis('off')
ax.set_title("Final external (activity test set) — funnel\nleakage control + no-activity drop",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_final_external_funnel.svg"); fig.savefig(f"{OUT}/FIG_final_external_funnel.png"); plt.close(fig)

# ── 2) Tanimoto plot of REMAINING final external (2,562) vs DB3 ──
fig,ax=plt.subplots(figsize=(7.5,4.6))
bins=np.linspace(0,1,41)
ax.hist(pv,bins=bins,color='#4C72B0',alpha=0.85,edgecolor='white',density=False)
ax.axvline(0.70,color='#C44E52',ls='--',lw=1.4); ax.axvline(0.85,color='#55A868',ls='--',lw=1.2)
ax.axvline(0.90,color='#8172B3',ls='--',lw=1.2)
for x,c in [(0.70,'#C44E52'),(0.85,'#55A868'),(0.90,'#8172B3')]:
    ax.text(x,ax.get_ylim()[1]*0.98,f"{x}",color=c,fontsize=8,ha='center')
ax.set_xlabel("Max Tanimoto similarity to PROTAC-DB 3.0 (Morgan r2, 2048 bit)")
ax.set_ylabel("compounds"); ax.set_title(f"Filtered final external set (n={n_tack_novel:,}): tanimoto-to-DB3\n(952 exact PROTAC-DB3 matches removed; patent dropped: no activity endpoint)")
ax.set_xlim(0,1)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_final_external_Tanimoto.svg"); fig.savefig(f"{OUT}/FIG_final_external_Tanimoto.png"); plt.close(fig)
pd.DataFrame([{"stage":l,"count":n} for l,n in stages]).to_csv(f"{OUT}/final_external_funnel_steps.csv", index=False)
print("saved funnel + tanimoto plots + steps csv")