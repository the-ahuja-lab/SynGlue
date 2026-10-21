#!/usr/bin/env python3
"""Clean A4 multi-panel SI figure v2.
- Square, coherent panels: (A) dataset table (B) venn (C) density+box w/ jitter (D) per-database donuts
- Proper axis labels, harmonious colorblind-safe palette, Arial-compatible font.
Donuts show each LEVEL (Targets / E3 / Scaffolds / Assays) split by DATABASE:
  PROTAC-DB, TACK, TPDdb, PROTACpedia, PROTAC-Patent (consistent color per database throughout).
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
RDLogger.DisableLog('rdApp.*')

BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
DATA=f"{BASE}/external_benchmark_data"; S53="/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT=f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs"; os.makedirs(OUT, exist_ok=True)

for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight',
                     'svg.fonttype':'none'})

# harmonious colorblind-safe palette, one color per DATABASE (used everywhere)
DB_ORDER = ["PROTAC-DB","TACK","TPDdb","PROTACpedia","PROTAC-Patent"]
DB_COLOR = {"PROTAC-DB":"#4C72B0","TACK":"#8172B3","TPDdb":"#55A868",
            "PROTACpedia":"#C44E52","PROTAC-Patent":"#DD8452"}
TRAIN_C="#4C72B0"; EXT_C="#C44E52"; SHARED_C="#CCB974"

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def murcko(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
def norm(s): return s.astype(str).str.strip().str.upper()

pdb=pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr =pd.read_csv(f"{S53}/Upadted_PROTAC_features.csv", low_memory=False, usecols=lambda c: not c.startswith("Grover_"))
tack=pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
pat=pd.read_csv(f"{DATA}/merged_patent_pubchem_only.csv", low_memory=False)
ext=pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)
fd=pd.read_csv(f"{DATA}/final_dataset_E3_known.csv", low_memory=False)

# ---------------- per-database elements ----------------
def scrf(df, smi): return set(c for c in map(murcko, df[smi].dropna().astype(str)) if c)
tp = tack[tack['Database'].str.contains('TPDdb',na=False)]
pe = tack[tack['Database'].str.contains('PROTACpedia',na=False)]
t3 = tack[tack['Database'].str.contains('PROTAC-DB',na=False)]
DB = {
 "PROTAC-DB": dict(targets=set(norm(pdb['Uniprot'][pdb['Uniprot'].notna()])),
                   e3=set(norm(pdb['E3 ligase'][pdb['E3 ligase'].notna()])),
                   scaffolds=scrf(pdb,'Smiles'),
                   assays=set(pdb['Assay (DC50/Dmax)'].dropna().astype(str).str.strip())),
 "TACK":      dict(targets=set(norm(tack['POI_UniProt'][tack['POI_UniProt'].notna()])),
                   e3=set(norm(tack['Ligase_Name'][tack['Ligase_Name'].notna()])),
                   scaffolds=scrf(tack,'SMILES'),
                   assays=set(tack['Assay'].dropna().astype(str).str.strip())),
 "TPDdb":     dict(targets=set(norm(tp['POI_UniProt'][tp['POI_UniProt'].notna()])),
                   e3=set(norm(tp['Ligase_Name'][tp['Ligase_Name'].notna()])),
                   scaffolds=scrf(tp,'SMILES'),
                   assays=set(tp['Assay'].dropna().astype(str).str.strip())),
 "PROTACpedia":dict(targets=set(norm(pe['POI_UniProt'][pe['POI_UniProt'].notna()])),
                   e3=set(norm(pe['Ligase_Name'][pe['Ligase_Name'].notna()])),
                   scaffolds=scrf(pe,'SMILES'),
                   assays=set()),
 "PROTAC-Patent":dict(targets=set(norm(pat['Target'][pat['Target'].notna()])), e3=set(),
                   scaffolds=scrf(pat,'canonical_SMILES'), assays=set()),
}
def dcount(level):
    return [len(DB[d][level]) for d in DB_ORDER]

# ---------------- table ----------------
def nuniq(s): return len(set(c for c in map(canon,s.dropna().astype(str)) if c))
def nscaf(s): return len(set(c for c in map(murcko,s.dropna().astype(str)) if c))
fd_np = fd[~fd['Database'].astype(str).str.contains('PROTAC-DB')].drop_duplicates('canonical_SMILES')
def R(name,df,smi): return [name,len(df),nuniq(df[smi]),nscaf(df[smi])]
trows=[R("PROTAC-DB local raw (training DB)",pdb,'Smiles'),
       R("Model training set",tr,'PROTAC_SMILES'),
       R("TACK (full)",tack,'SMILES'),
       R("TACK/TPDdb",tp,'SMILES'),
       R("TACK/PROTACpedia",pe,'SMILES'),
       R("TACK/PROTAC-DB 3.0",t3,'SMILES'),
       R("final_dataset (all merged)",fd,'canonical_SMILES'),
       R("EXTERNAL POOL (final−PROTAC-DB)",fd_np,'canonical_SMILES'),
       R("PROTAC-Patent (PubChem)",pat,'canonical_SMILES'),
       R("External not-in-training (final)",ext.drop_duplicates('SMILES'),'SMILES')]
header=["Dataset","Records","Unique","Scaffolds"]

# ---------------- activity ----------------
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
def W(a,b):
    U,p=stats.mannwhitneyu(a,b,alternative='two-sided'); st='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    cliff=2*U/(len(a)*len(b))-1
    return U,p,st,cliff
_=W(dcA,dcB); _=W(dmA,dmB)

# ═══════════ A4 FIGURE ═══════════
fig=plt.figure(figsize=(8.27,11.69))
fig.suptitle("Dataset accountability, coverage & activity density — SynGlue", fontsize=14, fontweight='bold')
gs=fig.add_gridspec(5,1,height_ratios=[2.15,1.45,2.2,2.2,1.1],hspace=0.42)

def panel_label(ax, letter, txt):
    ax.text(-0.02,1.08,f"{letter}",transform=ax.transAxes,fontsize=13,fontweight='bold',va='bottom',ha='left')

# ── (A) table ──
axA=fig.add_subplot(gs[0]); axA.axis('off')
tab=axA.table(cellText=trows,colLabels=header,loc='center',cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1,1.55)
for (r,c),cell in tab.get_celld().items():
    if r==0: cell.set_facecolor('#4C72B0'); cell.set_text_props(color='white',weight='bold')
    elif c==0: cell.set_text_props(weight='bold'); cell.set_facecolor('#EEF1F7')
axA.set_title("(A) Dataset accounting — from source to final external set",fontsize=10,loc='left')
panel_label(axA,"A","")

# ── (B) venn (square circles) ──
gsB=gs[1].subgridspec(1,3,wspace=0.35)
axB=[fig.add_subplot(gsB[i],aspect='equal') for i in range(3)]
tr_sc=scrf(pdb,'Smiles'); ex_sc=scrf(ext,'SMILES')
tr_tg=set(norm(pdb['Uniprot'][pdb['Uniprot'].notna()])); ex_tg=set(norm(ext['POI_UniProt'][ext['POI_UniProt'].notna()]))
tr_e3=set(norm(pdb['E3 ligase'][pdb['E3 ligase'].notna()])); ex_e3=set(norm(ext['E3_Ligase'][ext['E3_Ligase'].notna()]))
def venn(ax,a,b,title):
    oa,both,ob=len(a-b),len(a&b),len(b-a)
    ax.add_patch(plt.Circle((0.8,0.5),0.34,color=TRAIN_C,alpha=0.35))
    ax.add_patch(plt.Circle((1.2,0.5),0.34,color=EXT_C,alpha=0.35))
    ax.text(0.62,0.5,str(oa),ha='center',fontsize=10,weight='bold')
    ax.text(1.38,0.5,str(ob),ha='center',fontsize=10,weight='bold')
    ax.text(1.0,0.5,str(both),ha='center',fontsize=10,weight='bold')
    ax.text(0.8,0.88,"PROTAC-DB",ha='center',color=TRAIN_C,fontsize=7.5)
    ax.text(1.2,0.88,"External",ha='center',color=EXT_C,fontsize=7.5)
    ax.set_xlim(0.35,1.65); ax.set_ylim(0.05,0.95); ax.axis('off'); ax.set_title(title,fontsize=9)
for ax,(a,b),t in zip(axB,[(tr_sc,ex_sc),(tr_tg,ex_tg),(tr_e3,ex_e3)],["Scaffolds","Targets","E3"]):
    venn(ax,a,b,t)
fig.text(0.5,0.885,"(B) Coverage Venn — PROTAC-DB (training) vs External",fontsize=10,ha='center')

# ── (C) density + box w/ jitter ──
gsC=gs[2].subgridspec(2,2,hspace=0.55,wspace=0.28)
axC=[[fig.add_subplot(gsC[i,j]) for j in range(2)] for i in range(2)]
def densbox(axd,axb,a,b,xlab,Up,st,cliff,unit):
    for data,c,lbl in [(a,TRAIN_C,"PROTAC-DB"),(b,EXT_C,"External")]:
        if len(data)<2: continue
        k=gaussian_kde(data); xs=np.linspace(data.min()-0.1*(data.max()-data.min()),data.max()+0.1*(data.max()-data.min()),300)
        axd.plot(xs,k(xs),color=c,lw=2,label=lbl); axd.fill_between(xs,k(xs),color=c,alpha=0.25)
    axd.set_xlabel(xlab); axd.set_ylabel("density"); axd.legend(fontsize=7,loc='upper right',frameon=False)
    bp=axb.boxplot([a,b],labels=["PROTAC-DB\n(train)","External"],patch_artist=True,
                   medianprops=dict(color='k'),showfliers=False,widths=0.5)
    for p_,c_ in zip(bp['boxes'],[TRAIN_C,EXT_C]): p_.set_facecolor(c_); p_.set_alpha(0.85)
    rng=np.random.RandomState(42)
    for i,data in enumerate([a,b]):
        axb.scatter(i+1+rng.uniform(-0.16,0.16,len(data)),data,s=3,alpha=0.25,color='#222',linewidths=0)
    # median value labels on each box
    for i,data in enumerate([a,b]):
        med=np.median(data)
        axb.text(i+1, med, f" median={med:.2f}{unit}", va='center', ha='left', fontsize=7, color='#111')
    axb.set_ylabel(xlab); axb.set_xticklabels(["PROTAC-DB\n(train)","External"])
    axb.set_title(f"U={int(Up)}, p={Up:.3g} ({st})  |  effect (Cliff's δ)={cliff:+.2f}",fontsize=7.5)
U1,p1,s1,c1=W(dcA,dcB); densbox(axC[0][0],axC[1][0],dcA,dcB,"log10(DC50 / nM)",p1,s1,c1,"")
axC[0][0].set_title("DC50 density — PROTAC-DB vs External",fontsize=9)
U2,p2,s2,c2=W(dmA,dmB); densbox(axC[0][1],axC[1][1],dmA,dmB,"Dmax (%)",p2,s2,c2,"%")
axC[0][1].set_title("Dmax density — PROTAC-DB vs External",fontsize=9)
fig.text(0.5,0.60,"(C) Activity density (top) + box-and-jitter (bottom)",fontsize=10,ha='center')

# ── (D) per-database donuts (Targets / E3 / Scaffolds / Assays) ──
gsD=gs[3].subgridspec(2,2,wspace=0.25,hspace=0.35)
axD=[[fig.add_subplot(gsD[i,j],aspect='equal') for j in range(2)] for i in range(2)]
levels=[("Targets",[len(DB[d]['targets']) for d in DB_ORDER]),
        ("E3 ligases",[len(DB[d]['e3']) for d in DB_ORDER]),
        ("Scaffolds",[len(DB[d]['scaffolds']) for d in DB_ORDER]),
        ("Assay types",[len(DB[d]['assays']) for d in DB_ORDER])]
for ax,(lvl,vals) in zip([axD[0][0],axD[0][1],axD[1][0],axD[1][1]],levels):
    w,_,aut=ax.pie(vals,colors=[DB_COLOR[d] for d in DB_ORDER],startangle=90,counterclock=False,
        wedgeprops=dict(width=0.4,edgecolor='white'),autopct=lambda p: f"{p:.0f}%" if p>0 else "",
        pctdistance=0.82,textprops=dict(fontsize=6))
    ax.text(0,0,str(sum(vals)),ha='center',va='center',fontsize=10,weight='bold')
    ax.set_title(lvl,fontsize=9)
legend_handles=[Patch(color=DB_COLOR[d],label=d) for d in DB_ORDER]
fig.legend(handles=legend_handles,loc='lower center',ncol=5,frameon=False,fontsize=8,
           bbox_to_anchor=(0.5,0.045))
fig.text(0.5,0.32,"(D) Coverage by database — Targets / E3 / Scaffolds / Assays",fontsize=10,ha='center')

fig.savefig(f"{OUT}/SI_multipanel_A4_v2.png")
fig.savefig(f"{OUT}/SI_multipanel_A4_v2.svg")
print("saved SI_multipanel_A4_v2")
print("donut counts:", {l:vals for l,vals in levels})
