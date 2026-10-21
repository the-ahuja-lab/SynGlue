#!/usr/bin/env python3
"""Venn: PROTAC-DB 3.0 (training) vs FINAL EXTERNAL activity set (chiral Tanimoto<0.70, n=2,256).
Three levels: compounds (exact), scaffolds (Murcko), targets. Real SVG text."""
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
S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','figure.dpi':300,'savefig.bbox':'tight'})
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_uniq=set(c for c in map(canon,pdb['Smiles'].dropna().astype(str)) if c)
tr_scaff=set(c for c in map(MurckoScaffold.MurckoScaffoldSmilesFromSmiles if hasattr(MurckoScaffold,'MurckoScaffoldSmilesFromSmiles') else (lambda s: None), []) ) if False else None
def murcko(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
tr_scaff=set(c for c in map(murcko,pdb['Smiles'].dropna().astype(str)) if c)
tr_tgt=set(pdb['Uniprot'].dropna().astype(str).str.strip().str.upper()[lambda x: x!='NAN']) if False else set(pdb['Uniprot'].dropna().astype(str).str.strip().str.upper())
tr_tgt=set(t for t in tr_tgt if t!='NAN')

# final external (2,256 chiral<0.7)
ch=pd.read_csv(f"{OUT}/final_external_tack_chiral.csv")
ch['tmax_chiral']=pd.to_numeric(ch['tmax_chiral'],errors='coerce')
fin=ch[ch['tmax_chiral']<0.70].copy()
print("final external (chiral<0.7):", len(fin))
ext_uniq=set(fin['canonical_SMILES'].dropna())
ext_scaff=set(fin['scaffold'].dropna())
ext_tgt=set(fin['target'].dropna().astype(str).str.strip().str.upper())
ext_tgt=set(t for t in ext_tgt if t not in ('NAN','NA'))
print(f"  ext uniq={len(ext_uniq)} scaffolds={len(ext_scaff)} targets={len(ext_tgt)}")

# overlap counts
def v3(a,b): return dict(oa=len(a-b), both=len(a&b), ob=len(b-a), A=len(a), B=len(b))
res={"compounds":v3(tr_uniq,ext_uniq),"scaffolds":v3(tr_scaff,ext_scaff),"targets":v3(tr_tgt,ext_tgt)}
for k,v in res.items(): print(f"  {k}: only_train={v['oa']} both={v['both']} only_ext={v['ob']}")
pd.DataFrame([{"level":k,**v} for k,v in res.items()]).to_csv(f"{OUT}/venn_final_external_vs_DB3.csv", index=False)

# 3-panel venn
fig,axes=plt.subplots(1,3,figsize=(12,4.4))
for ax,(title,k) in zip(axes,[("Compounds (exact)","compounds"),("Scaffolds (Murcko)","scaffolds"),("Targets (POI)","targets")]):
    v=res[k]; ax.add_patch(plt.Circle((0.80,0.5),0.34,color='#4C72B0',alpha=0.35))
    ax.add_patch(plt.Circle((1.20,0.5),0.34,color='#C44E52',alpha=0.35))
    ax.text(0.62,0.5,f"{v['oa']:,}",ha='center',fontsize=11,weight='bold')
    ax.text(1.00,0.5,f"{v['both']:,}",ha='center',fontsize=11,weight='bold')
    ax.text(1.38,0.5,f"{v['ob']:,}",ha='center',fontsize=11,weight='bold')
    ax.text(0.80,0.88,"PROTAC-DB",ha='center',color='#4C72B0',fontsize=8)
    ax.text(1.20,0.88,"Final external",ha='center',color='#C44E52',fontsize=8)
    ax.set_xlim(0.35,1.65); ax.set_ylim(0.05,0.98); ax.axis('off'); ax.set_aspect('equal')
    ax.set_title(f"{title}\n(DB {v['A']:,} vs ext {v['B']:,})",fontsize=9)
fig.suptitle("Venn: PROTAC-DB 3.0 (training) vs Final External activity set (n=2,256, Tanimoto<0.70)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f"{OUT}/FIG_venn_final_external.svg"); fig.savefig(f"{OUT}/FIG_venn_final_external.png"); plt.close(fig)
print("saved FIG_venn_final_external")