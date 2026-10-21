#!/usr/bin/env python3
"""Fix A + better funnel:
- Recompute Tanimoto max-to-PROTAC-DB3 with CHIRAL-AWARE Morgan fps (useChirality=True)
  for the final external (TACK-novel) set.
- Rebuild cleaner funnel: patent branch (63,136 -> 57,533 -> 43,891 <0.7) + activity branch
  (60,769 -> 3,514 -> 2,562 exact-only ["unique in table"] -> optional -86 stereoisomers
  -> <0.90 -> <0.85 -> FINAL 2,238 <0.70).
- Regenerate Tanimoto histogram of the filtered/final set with chiral cutoffs.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
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

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def fp(s,chiral):
    m=Chem.MolFromSmiles(str(s));
    return AllChem.GetMorganFingerprintAsBitVect(m,2,2048,useChirality=chiral) if m else None

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr_fps_ch=[f for f in map(lambda s: fp(s,True), pdb['Smiles'].dropna().astype(str)) if f is not None]
print(f"chiral training fps: {len(tr_fps_ch)}")

fm=pd.read_csv(f"{OUT}/final_external_with_meta.csv")
tack=fm[fm['source']=='TACK'].copy()
print("TACK novel unique:", len(tack))
# chiral tmax per TACK-novel compound
vals=[]
for s in tack['canonical_SMILES'].astype(str):
    f=fp(s,True)
    if f is None: vals.append(np.nan); continue
    vals.append(max(DataStructs.BulkTanimotoSimilarity(f,tr_fps_ch)))
tv=np.array(vals,dtype=float)
tack['tmax_chiral']=tv
tack.to_csv(f"{OUT}/final_external_tack_chiral.csv", index=False)
n070=int(np.sum(tv<0.70)); n085=int(np.sum(tv<0.85)); n090=int(np.sum(tv<0.90))
print(f"chiral: <0.70={n070}  <0.85={n085}  <0.90={n090}  (achiral was 2238/2420/2447)")
n_stereo=int(np.sum(tv>=0.9999))
n_stereo86=86  # achiral Tanimoto=1.0 (stereoisomers/tautomers of training)
print("stereoisomer dups (chiral tmax=1.0):", n_stereo)

# ── funnel values (activity branch) ──
n_pool=60769; n_act=3514; n_exact_removed=952; n_exact_only=2562
stages=[
 ("PatentDB: 63,136 rows → 57,533 unique → 43,891 novel (Tanimoto<0.70)", None),
 ("1. All external pooled unique (TACK + Patent)", n_pool),
 ("2. Drop: no activity endpoint (Patent –57,533)", n_act),
 ("3. Drop: exact PROTAC-DB3 name match (–952)", n_exact_only),
 ("4. Tanimoto < 0.90 (chiral)", n090),
 ("5. Tanimoto < 0.85 (chiral)", n085),
 ("6. FINAL external activity set — Tanimoto < 0.70 (chiral)", n070),
]
levels=[n for _,n in stages[1:]]; labels=[l for l,_ in stages[1:]]
print(stages)

# ── BETTER funnel plot ──
fig,ax=plt.subplots(figsize=(8.2,6.4))
mx=max(levels)
pal=plt.cm.Blues(np.linspace(0.25,0.9,len(levels)))
extra=[0,0,1,0,0,0,1]  # highlight drop & final
for yi,(wi,n,lbl) in enumerate(zip([n/mx for n in levels],levels,labels)):
    y=len(levels)-1-yi
    col=pal[yi]
    ax.barh(y,wi,height=0.62,color=col,edgecolor='white',left=(1-wi)/2)
    ax.text(0.5,y,f"{n:,}",ha='center',va='center',color='white',fontsize=11 if yi==len(levels)-1 else 10,
            weight='bold')
    # label to the LEFT of bar in small font
    ax.text(0.015,y+0.34,lbl,ha='left',va='bottom',fontsize=8,color='#222')
ax.set_xlim(0,1); ax.set_ylim(-0.8,len(levels)-0.2); ax.axis('off')
ax.set_title("Final external activity test set — build funnel (chiral-aware Tanimoto)\n"
             "PatentDB branch: 63,136 → 57,533 → 43,891 <0.70 (coverage-only, no activity)",
             fontsize=10.5)
ax.text(0.5, -0.06, "Note: the 2,562 set retains {:,} stereoisomers/tautomers of training (achiral Tanimoto=1.0);\n"
        "these are removed by the <0.70 filter (their similarities 0.90–1.0).".format(n_stereo86),
        transform=ax.transAxes, ha='center', va='top', fontsize=8, color='#444')
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_final_external_funnel.svg"); fig.savefig(f"{OUT}/FIG_final_external_funnel.png"); plt.close(fig)

# ── Tanimoto histogram of kept set with chiral cutoffs ──
fig,ax=plt.subplots(figsize=(7.6,4.8))
bins=np.linspace(0,1,41)
ax.hist(tv,bins=bins,color='#4C72B0',alpha=0.85,edgecolor='white')
for x,c in [(0.70,'#C44E52'),(0.85,'#55A868'),(0.90,'#8172B3')]:
    ax.axvline(x,color=c,ls='--',lw=1.3)
    ax.text(x,ax.get_ylim()[1]*0.97,f"{x}",color=c,fontsize=8,ha='center')
ax.set_xlabel("Max Tanimoto to PROTAC-DB 3.0 (Morgan r2, 2048 bit, CHIRAL-aware)")
ax.set_ylabel("compounds")
ax.set_title(f"Final external (n={len(tack):,} after exact removal): Tanimoto-to-DB3\n"
             f"chiral-aware: <0.70 = {n070:,} · <0.85 = {n085:,} · <0.90 = {n090:,}\n"
             f"{n_stereo} are stereoisomers of training (tmax=1.0 achiral → <1.0 chiral)")
ax.set_xlim(0,1)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_final_external_Tanimoto.svg"); fig.savefig(f"{OUT}/FIG_final_external_Tanimoto.png"); plt.close(fig)
pd.DataFrame([{"stage":l,"count":("n/a" if n is None else n)} for l,n in stages]).to_csv(f"{OUT}/final_external_funnel_steps.csv", index=False)
print("saved better funnel + chiral tanimoto plot")