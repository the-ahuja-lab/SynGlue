#!/usr/bin/env python3
"""Venn: PROTAC-DB 3.0 (training) vs Random-split external test set (WP2 random 80/20 of the 825 set)."""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
RDLogger.DisableLog('rdApp.*')
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
BM=f"{REV}/benchmarking"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','figure.dpi':300,'savefig.bbox':'tight'})
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def norm(x): return str(x).strip().upper()

# training ref = PROTAC-DB 3.0 (6,110 unique)
pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_uniq=set(c for c in map(canon,pdb['Smiles'].dropna().astype(str)) if c)
# model training set (5,203) as secondary
trm=pd.read_csv(f"{S53}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
trm_uniq=set(c for c in map(canon,trm['PROTAC_SMILES'].dropna().astype(str)) if c)

meta=pd.read_pickle(f"{BM}/meta.pkl")
X=np.load(f"{BM}/X.npy"); print("meta/X rows:", len(meta), X.shape)
meta['canon']=[canon(s) for s in meta['smiles']]
# reproduce WP2 random 80/20 split (random_state=42)
idx=np.arange(len(meta))
tr,te=train_test_split(idx, test_size=0.2, random_state=42)
print("random split: train",len(tr),"test",len(te))
test_meta=meta.iloc[te]
te_uniq=set(meta.loc[te,'canon'].dropna())
oct82=len(te_uniq & tr_uniq)  # overlap random-test vs PROTAC-DB3
ote=len(te_uniq - tr_uniq)
print(f"random-split test: {len(te_uniq)} unique | in PROTAC-DB3: {oct82} | external-only: {ote}")
octm=len(te_uniq & trm_uniq)
print(f"  vs model-training(5,203): in={octm} out={len(te_uniq)-octm}")
# context: whole 825 set overlap
all825=set(meta['canon'].dropna())
print(f"825 set: {len(all825)} unique | in PROTAC-DB3: {len(all825 & tr_uniq)} | in model-training: {len(all825 & trm_uniq)}")

# ── Venn: PROTAC-DB3 (training) vs Random-split external test ──
fig,ax=plt.subplots(figsize=(7,5.2))
oa=len(tr_uniq-te_uniq); both=oct82; ob=ote
ax.add_patch(plt.Circle((0.38,0.5),0.30,color='#4C72B0',alpha=0.35))
ax.add_patch(plt.Circle((0.68,0.5),0.30,color='#C44E52',alpha=0.35))
ax.text(0.28,0.5,f"{oa:,}",ha='center',fontsize=12,weight='bold')
ax.text(0.53,0.5,f"{both:,}",ha='center',fontsize=12,weight='bold')
ax.text(0.78,0.5,f"{ob:,}",ha='center',fontsize=12,weight='bold')
ax.text(0.38,0.87,f"PROTAC-DB 3.0\n({len(tr_uniq):,})",ha='center',color='#4C72B0',fontsize=9)
ax.text(0.68,0.87,f"Random-split external test\n({len(te_uniq):,})",ha='center',color='#C44E52',fontsize=9)
ax.set_xlim(0,1.05); ax.set_ylim(0,0.95); ax.axis('off')
ax.set_aspect('equal')
ax.set_title("Random 80/20 split (WP2, 825-set): PROTAC-DB (training) vs random-split external test\n"
             "exact-canonical overlap (leakage check)",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_venn_random_split.svg"); fig.savefig(f"{OUT}/FIG_venn_random_split.png"); plt.close(fig)
pd.DataFrame([{"set":"random test unique","value":len(te_uniq)},
              {"set":"test ∩ PROTAC-DB3","value":oct82},
              {"set":"test external-only","value":ote},
              {"set":"test ∩ model-training(5203)","value":octm},
              {"set":"825 set unique","value":len(all825)},
              {"set":"825 ∩ PROTAC-DB3","value":len(all825 & tr_uniq)}]).to_csv(f"{OUT}/venn_random_split_counts.csv", index=False)
print("saved FIG_venn_random_split + counts csv")