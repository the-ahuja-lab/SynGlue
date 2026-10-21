#!/usr/bin/env python3
"""PROTAC-DB (training) vs External — density + box comparison for log10 DC50 and Dmax.
External = final external activity set (external_not_in_training, exact train matches removed).
Outputs: FIG_DC50_Dmax_protacdb_vs_external.{svg,png} + stats CSV. Real SVG text.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
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
plt.rcParams.update({'font.family':'Arial','font.size':9.5,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight'})
TRAIN_C='#4C72B0'; EXT_C='#C44E52'

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
ext=pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)

def dcp(d):
    v=pd.to_numeric(d['DC50 (nM)'],errors='coerce').dropna(); v=v[v>0]; return np.log10(v.values)
def dmp(d):
    v=pd.to_numeric(d['Dmax (%)'],errors='coerce').dropna(); return v[(v>=0)&(v<=100)].values
def edc(d):
    x=d[d['Value_Type']=='DC50'].copy(); x['v']=pd.to_numeric(x['Value'],errors='coerce'); x=x[x['v']>0]
    return np.log10(x.groupby('canonical_SMILES')['v'].median().dropna().values)
def edm(d):
    x=d[d['Value_Type']=='Dmax'].copy(); x['v']=pd.to_numeric(x['Value'],errors='coerce'); x=x[(x['v']>=0)&(x['v']<=100)]
    return x.groupby('canonical_SMILES')['v'].max().dropna().values
dcA,dcB=dcp(pdb),edc(ext); dmA,dmB=dmp(pdb),edm(ext)

def quart(x): 
    q=np.percentile(x,[25,50,75]); return q
def W(a,b):
    U,p=stats.mannwhitneyu(a,b,alternative='two-sided'); st='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    return U,p,st,2*U/(len(a)*len(b))-1
sq=[]; stat_rows=[]
for lab,a,b,unit,medf in [("log10 DC50 (nM)",dcA,dcB,"",""),("Dmax (%)",dmA,dmB,"%","")]:
    qa,qb=quart(a),quart(b); U,p,st,c=W(a,b)
    sq.append(dict(endpoint=lab, group="PROTAC-DB", n=len(a), Q1=qa[0], median=qa[1], Q3=qa[2]))
    sq.append(dict(endpoint=lab, group="External", n=len(b), Q1=qb[0], median=qb[1], Q3=qb[2]))
    stat_rows.append(dict(endpoint=lab, U=U, p=p, sig=st, cliffs_delta=c, n_train=len(a), n_ext=len(b)))
    print(f"{lab}: PDB med={qa[1]:.3f} n={len(a)} | EXT med={qb[1]:.3f} n={len(b)} | p={p:.3g} {st} delta={c:+.3f}")
pd.DataFrame(sq).to_csv(f"{OUT}/density_quartiles.csv", index=False)
pd.DataFrame(stat_rows).to_csv(f"{OUT}/density_wilcoxon.csv", index=False)

# ── figure: 2x2 (density / box rows; DC50 / Dmax cols) ──
fig,axs=plt.subplots(2,2,figsize=(10,7.5))
def density(ax,a,b,xlab): 
    for data,c,lbl in [(a,TRAIN_C,"PROTAC-DB (train)"),(b,EXT_C,"External")]:
        if len(data)<2: continue
        k=gaussian_kde(data); xs=np.linspace(data.min()-0.1*np.ptp(data),data.max()+0.1*np.ptp(data),400)
        ax.plot(xs,k(xs),color=c,lw=2.2,label=lbl); ax.fill_between(xs,k(xs),color=c,alpha=0.22)
        ax.axvline(np.median(data),color=c,ls='--',lw=1.0,alpha=0.7)
    ax.set_xlabel(xlab); ax.set_ylabel("density"); ax.legend(fontsize=8,frameon=False)
def box(ax,a,b,xlab,Up,st,c=None):
    bp=ax.boxplot([a,b],labels=["PROTAC-DB\n(train)","External"],patch_artist=True,
                  medianprops=dict(color='k',lw=1.5),showfliers=False,widths=0.5)
    for p_,c_ in zip(bp['boxes'],[TRAIN_C,EXT_C]): p_.set_facecolor(c_); p_.set_alpha(0.85)
    rng=np.random.RandomState(42)
    for i,data in enumerate([a,b]):
        ax.scatter(i+1+rng.uniform(-0.18,0.18,len(data)),data,s=4,alpha=0.25,color='#222',linewidths=0)
    for i,data in enumerate([a,b]):
        med=np.median(data)
        ax.text(i+1,med,f" med={med:.3f}",va='center',ha='left',fontsize=7.5)
    ax.set_ylabel(xlab); ax.set_xticklabels(["PROTAC-DB\n(train)","External"])
    ax.set_title(f"Mann-Whitney U={int(Up):,}, p={Up:.3g} ({st}) | Cliff's δ={c:+.2f}",fontsize=8)
U1,p1,s1,c1=W(dcA,dcB); density(axs[0,0],dcA,dcB,"log10(DC50 / nM)")
U2,p2,s2,c2=W(dmA,dmB); density(axs[0,1],dmA,dmB,"Dmax (%)")
box(axs[1,0],dcA,dcB,"log10(DC50 / nM)",p1,s1,c1)
box(axs[1,1],dmA,dmB,"Dmax (%)",p2,s2,c2)
axs[0,0].set_title("DC50 — PROTAC-DB vs External",fontsize=10)
axs[0,1].set_title("Dmax — PROTAC-DB vs External",fontsize=10)
fig.suptitle("Activity distributions: PROTAC-DB (training) vs Final External set\n(density top, box-and-jitter bottom; median dashed)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f"{OUT}/FIG_DC50_Dmax_protacdb_vs_external.svg")
fig.savefig(f"{OUT}/FIG_DC50_Dmax_protacdb_vs_external.png")
print("saved FIG_DC50_Dmax_protacdb_vs_external")