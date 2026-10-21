#!/usr/bin/env python3
"""Clear per-source -> final external figure (A4 portrait, real SVG text).
Panels:
  A: per-source grouped bars (Records / Unique / Novel-to-DB3)
  B: final-external target split (top 12)
  C: final-external E3 split
  D: scaffold & source callouts
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
RDLogger=None
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight'})
SRC_C={"TACK":"#4C72B0","TPDdb":"#55A868","PROTACpedia":"#C44E52","PROTAC-Patent":"#8172B3"}

per=pd.read_csv(f"{OUT}/per_source_vs_DB3.csv")
fm=pd.read_csv(f"{OUT}/final_external_with_meta.csv")

fig=plt.figure(figsize=(8.27,11.0))
fig.suptitle("Source → final external dataset (vs PROTAC-DB 3.0 train)", fontsize=14, fontweight='bold')
gs=fig.add_gridspec(4,1,height_ratios=[2.0,1.9,1.4,1.3],hspace=0.5)

# ── A: per-source grouped bars ──
axA=fig.add_subplot(gs[0])
srcs=per['source'].tolist(); x=np.arange(len(srcs)); w=0.26
for off,key,c in [(-0.28,"records",'#8CB0D9'),(0.0,"unique",'#4C72B0'),(0.28,"novel_to_DB3",'#C44E52')]:
    axA.bar(x+off, per[key], width=w, color=c, label={'records':'A: Records','unique':'B: Unique','novel_to_DB3':'C: Novel to DB3'}[key], edgecolor='white')
    for xi,v in zip(x+off, per[key]):
        axA.text(xi, v+80, f"{v:,}", ha='center', fontsize=6.5, rotation=0)
axA.set_xticks(x); axA.set_xticklabels(srcs)
axA.set_ylabel("count"); axA.set_title("(A) Per-source: records / unique / novel-to-PROTAC-DB3", loc='left')
axA.legend(fontsize=7, ncol=3, loc='upper left')

# ── B: final external target split ──
axB=fig.add_subplot(gs[1])
tc=fm['target'].value_counts().drop('NA',errors='ignore').head(12)
axB.barh(np.arange(len(tc))[::-1], tc.values, color='#4C72B0', alpha=0.9)
for yi,v in zip(np.arange(len(tc))[::-1], tc.values):
    axB.text(v+5, yi, f"{v:,}", va='center', fontsize=8)
axB.set_yticks(np.arange(len(tc))[::-1]); axB.set_yticklabels(tc.index, fontsize=7.5)
axB.set_xlabel("compounds"); axB.set_title("(B) Final external — target split (top 12)", loc='left')

# ── C: E3 split ──
axC=fig.add_subplot(gs[2], aspect='equal')
ec=fm['e3'].value_counts()
ec=ec[ec.index!='NA'] if 'NA' in ec.index else ec
vals=[ec.get('CRBN',0), ec.get('VHL',0), ec.get('IAP',0), ec.get('UBR1',0)]
lab=['CRBN','VHL','IAP','UBR1']
cols=['#4C72B0','#55A868','#DD8452','#8172B3']
w,_,aut=axC.pie(vals,colors=cols,startangle=90,counterclock=False,wedgeprops=dict(width=0.4,edgecolor='white'),
    autopct=lambda p:f"{p:.0f}%",pctdistance=0.78,textprops=dict(fontsize=8))
axC.text(0,0,f"{sum(vals)}\nwith E3",ha='center',va='center',fontsize=9,weight='bold')
axC.set_title("(C) Final external — E3 split (CRBN/VHL) | NA(patent)=3,859",fontsize=8.5)
axC.legend(w,lab,loc='center left',bbox_to_anchor=(1.0,0.5),fontsize=7,frameon=False)

# ── D: callouts ──
axD=fig.add_subplot(gs[3]); axD.axis('off')
info=[("Final external unique (novel to DB3)", len(fm)),
      ("Unique scaffolds", fm['scaffold'].nunique()),
      ("Of which PROTAC-Patent (coverage-only)", int((fm['source']=='PROTAC-Patent').sum()))]
yy=0.85
for lbl,val in info:
    axD.text(0.03,yy,f"{lbl}",transform=axD.transAxes,fontsize=9)
    axD.text(0.97,yy,f"{val:,}",transform=axD.transAxes,fontsize=10,weight='bold',ha='right')
    yy-=0.12
axD.set_title("(D) Final external summary",loc='left')
fig.savefig(f"{OUT}/SOURCE_TO_FINAL_figure.png")
fig.savefig(f"{OUT}/SOURCE_TO_FINAL_figure.svg")
print("saved SOURCE_TO_FINAL_figure")
