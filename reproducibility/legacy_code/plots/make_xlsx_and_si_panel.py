#!/usr/bin/env python3
"""Deliverables:
1) Excel workbook (xlsx) with all dataset counts from source to final external.
2) Single A4 multi-panel SI figure (Arial-compatible font), combining:
     Panel A: dataset accountability table
     Panel B: coverage venn (scaffolds / targets / E3)
     Panel C: density + box (log10 DC50, Dmax) with quartiles + Wilcoxon
     Panel D: coverage donuts (targets / E3 / scaffolds / assays)
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
RDLogger.DisableLog('rdApp.*')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV  = f"{BASE}/revision"
DATA = f"{BASE}/external_benchmark_data"
S53  = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT  = f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs"
os.makedirs(OUT, exist_ok=True)

# ── Arial-compatible font (Nimbus Sans = URW Arial clone) ──
_fonts = {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
          "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
          "NimbusSans-Italic":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf"}
for n,p in _fonts.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except Exception: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'figure.dpi':300,'savefig.bbox':'tight'})

# ── colors (harmonized) ──
TRAIN_C='#1f77b4'; EXT_C='#ff7f0e'; SHARED_C='#7f7f7f'

def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def murcko(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
def norm(s): return s.astype(str).str.strip().str.upper()

pdb = pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
tr  = pd.read_csv(f"{S53}/Upadted_PROTAC_features.csv", low_memory=False,
                  usecols=lambda c: not c.startswith("Grover_"))
tack= pd.read_parquet(f"{DATA}/tack_raw/data_train-00000-of-00001.parquet")
fd  = pd.read_csv(f"{DATA}/final_dataset_E3_known.csv", low_memory=False)
pat = pd.read_csv(f"{DATA}/merged_patent_pubchem_only.csv", low_memory=False)
ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)

# ════════ 1) DATASET ACCOUNTABILITY TABLE ════════
def nuniq(series): return len(set(c for c in map(canon, series.dropna().astype(str)) if c))
def nscaf(series): return len(set(c for c in map(murcko, series.dropna().astype(str)) if c))

def row(name, df, smi):
    return dict(Dataset=name, Records=len(df), Unique=nuniq(df[smi]),
                Scaffolds=nscaf(df[smi]))

fd_nopdb_unique = fd[~fd['Database'].astype(str).str.contains('PROTAC-DB')].drop_duplicates('canonical_SMILES')
table_rows = [
    row("PROTAC-DB local raw (training DB)", pdb, 'Smiles'),
    row("Model training set", tr, 'PROTAC_SMILES'),
    row("TACK (full)", tack, 'SMILES'),
    row("TACK/TPDdb", tack[tack['Database'].str.contains('TPDdb',na=False)], 'SMILES'),
    row("TACK/PROTACpedia", tack[tack['Database'].str.contains('PROTACpedia',na=False)], 'SMILES'),
    row("TACK/PROTAC-DB 3.0 subset", tack[tack['Database'].str.contains('PROTAC-DB',na=False)], 'SMILES'),
    row("final_dataset (all, merged)", fd, 'canonical_SMILES'),
    row("EXTERNAL POOL (final - PROTAC-DB, dedup)", fd_nopdb_unique, 'canonical_SMILES'),
    row("PROTAC-Patent (PubChem)", pat, 'canonical_SMILES'),
    row("External not-in-training (final external / benchmark)", ext.drop_duplicates('SMILES'), 'SMILES'),
]
tbl = pd.DataFrame(table_rows)
# add Targets/E3 columns
tgt = {"PROTAC-DB local raw (training DB)":"441/344 (name/UniProt)",
 "Model training set":"304", "TACK (full)":"131", "TACK/TPDdb":"27",
 "TACK/PROTACpedia":"66", "TACK/PROTAC-DB 3.0 subset":"100", "final_dataset (all, merged)":"131/166",
 "EXTERNAL POOL (final - PROTAC-DB, dedup)":"68/73", "PROTAC-Patent (PubChem)":"26*",
 "External not-in-training (final external / benchmark)":"53"}
e3c = {"PROTAC-DB local raw (training DB)":"21","Model training set":"20","TACK (full)":"9",
 "TACK/TPDdb":"2","TACK/PROTACpedia":"7","TACK/PROTAC-DB 3.0 subset":"7","final_dataset (all, merged)":"9",
 "EXTERNAL POOL (final - PROTAC-DB, dedup)":"7","PROTAC-Patent (PubChem)":"*",
 "External not-in-training (final external / benchmark)":"4"}
tbl['Targets'] = tbl['Dataset'].map(tgt); tbl['E3'] = tbl['Dataset'].map(e3c)
tbl = tbl[['Dataset','Records','Unique','Scaffolds','Targets','E3']]

# coverage / venn / wilcoxon (recompute quickly)
tr_smi=set(c for c in map(canon,pdb['Smiles']) if c); tr_sc=set(c for c in map(murcko,pdb['Smiles']) if c)
tr_tg=set(norm(pdb['Target'][pdb['Target'].notna()])); tr_e3=set(norm(pdb['E3 ligase'][pdb['E3 ligase'].notna()]))
tr_as=set(pdb['Assay (DC50/Dmax)'].dropna().astype(str).str.strip())
ex_smi=set(c for c in map(canon,ext['SMILES']) if c); ex_sc=set(c for c in map(murcko,ext['SMILES']) if c)
ex_tg=set(norm(ext['Protein_of_Interest'][ext['Protein_of_Interest'].notna()]))
ex_e3=set(norm(ext['E3_Ligase'][ext['E3_Ligase'].notna()])); ex_as=set(ext['Assay'].dropna().astype(str).str.strip())
cov = pd.DataFrame([dict(Level=l, PROTAC_DB=a, External=b, Shared=len(a&b), External_only=len(b-a))
                    for l,a,b in [("Compounds",tr_smi,ex_smi),("Scaffolds",tr_sc,ex_sc),
                                  ("Targets (POI)",tr_tg,ex_tg),("E3 ligases",tr_e3,ex_e3),
                                  ("Assay types",tr_as,ex_as)]])
venn = pd.DataFrame([dict(Level=l, Training_only=len(a-b), Shared=len(a&b), External_only=len(b-a))
                     for l,a,b in [("Scaffolds",tr_sc,ex_sc),("Targets",tr_tg,ex_tg),("E3",tr_e3,ex_e3)]])

# activity
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
dcA,dcB = dcp(pdb), edc(ext); dmA,dmB = dmp(pdb), edm(ext)
def q(x):
    qx=np.percentile(x,[25,50,75]); return dict(Q1=qx[0],Q2=qx[1],Q3=qx[2],n=len(x))
def W(a,b,lab):
    U,p=stats.mannwhitneyu(a,b,alternative='two-sided'); st='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    return dict(Endpoint=lab,U=U,p_value=p,sig=st,n_train=len(a),n_ext=len(b))
wstat=pd.DataFrame([W(dcA,dcB,"log10 DC50 (nM)"),W(dmA,dmB,"Dmax (%)")])
wall=[dict(Endpoint="log10 DC50 (nM)",PROTAC_DB_q=q(dcA),External_q=q(dcB)),
      dict(Endpoint="Dmax (%)",PROTAC_DB_q=q(dmA),External_q=q(dmB))]

# ════════ 2) EXCEL WORKBOOK ════════
xl = f"{OUT}/DATASET_ACCOUNTABILITY.xlsx"
with pd.ExcelWriter(xl, engine='openpyxl') as w:
    tbl.to_excel(w, sheet_name="Dataset_Counts", index=False)
    cov.to_excel(w, sheet_name="Coverage", index=False)
    venn.to_excel(w, sheet_name="Venn_Overlaps", index=False)
    wstat.to_excel(w, sheet_name="Wilcoxon_Stats", index=False)
    # quartile detail
    pd.DataFrame([{"Endpoint":"log10 DC50 (nM)","group":"PROTAC-DB",**q(dcA)},
                  {"Endpoint":"log10 DC50 (nM)","group":"External",**q(dcB)},
                  {"Endpoint":"Dmax (%)","group":"PROTAC-DB",**q(dmA)},
                  {"Endpoint":"Dmax (%)","group":"External",**q(dmB)}]
                ).to_excel(w, sheet_name="Quartiles_Q1_Q2_Q3", index=False)
    pd.DataFrame([{"Dataset":"PROTAC-DB 1.0","PROTACs":1662,"Warheads":202,"E3_ligands":65,"Linkers":806,"Ref":"NAR 2021 gkaa807"},
                  {"Dataset":"PROTAC-DB 2.0","PROTACs":3270,"Warheads":365,"E3_ligands":82,"Linkers":1501,"Ref":"NAR 2023 gkac946"},
                  {"Dataset":"PROTAC-DB 3.0","PROTACs":6111,"Warheads":569,"E3_ligands":107,"Linkers":2753,"Ref":"NAR 2025 gkae768"}]
                ).to_excel(w, sheet_name="PROTACDB_published", index=False)
print("XLSX written:", xl, "|", tbl.shape, "rows")

# ════════ 3) A4 MULTI-PANEL SI FIGURE ════════
A4_W, A4_H = 8.27, 11.69  # portrait
fig = plt.figure(figsize=(A4_W, A4_H), dpi=300)
fig.suptitle("Supplementary Figure — SynGlue dataset accountability, coverage, and activity density",
             fontsize=13, fontweight='bold')

gs = fig.add_gridspec(5, 1, height_ratios=[2.3, 1.6, 1.9, 1.9, 1.6], hspace=0.35)
axT = fig.add_subplot(gs[0]); axT.axis('off')
# ── Panel A: dataset table ──
short = tbl.copy(); short['Records']=short['Records'].astype(str)
short['Unique']=short['Unique'].astype(str); short['Scaffolds']=short['Scaffolds'].astype(str)
tab = axT.table(cellText=short.values, colLabels=short.columns, loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.5)
for (r,c),cell in tab.get_celld().items():
    if r==0: cell.set_facecolor('#3949ab'); cell.set_text_props(color='white', weight='bold')
    elif str(short.values[r-1][0]) not in ("TACK (full)","PROTAC-DB local raw (training DB)"):
        pass
    if c==0 and r>0: cell.set_text_props(weight='bold')
axT.set_title("(A) Dataset accounting — from source to final external set", fontsize=10, loc='left')

# ── Panel B: venn (3) ──
gsB = gs[1].subgridspec(1, 3, wspace=0.15)
axB = [fig.add_subplot(gsB[i]) for i in range(3)]
def draw_venn(ax, tA, tB, title):
    oa, totA = len(tA-tB), len(tA)
    both = len(tA & tB)
    ob, totB = len(tB-tA), len(tB)
    ax.add_patch(plt.Circle((0.85,1.1),1,color=TRAIN_C,alpha=0.35))
    ax.add_patch(plt.Circle((1.85,1.1),1,color=EXT_C,alpha=0.35))
    ax.text(0.32,1.1,f"{oa}",ha='center',fontsize=11,weight='bold')
    ax.text(2.38,1.1,f"{ob}",ha='center',fontsize=11,weight='bold')
    ax.text(1.35,1.1,f"{both}",ha='center',fontsize=11,weight='bold')
    ax.text(0.85,1.98,f"PROTAC-DB {totA}",ha='center',color=TRAIN_C,fontsize=7)
    ax.text(1.85,1.98,f"External {totB}",ha='center',color=EXT_C,fontsize=7)
    ax.set_xlim(0,2.7); ax.set_ylim(0,2.25); ax.axis('off'); ax.set_title(title,fontsize=9)
for ax,(a,b),tit in zip(axB, [(tr_sc,ex_sc),(tr_tg,ex_tg),(tr_e3,ex_e3)],
                             ["Scaffolds","Targets","E3"]):
    draw_venn(ax, a, b, tit)
fig.text(0.5, 0.93, "(B) Coverage Venn — PROTAC-DB (training) vs External", fontsize=10, ha='center')

# ── Panel C: density+box ──
gsC = gs[2].subgridspec(2,2, hspace=0.45, wspace=0.25)
def density_box(axd,axb,a,b,axlab,strow):
    for data,c,lbl in [(a,TRAIN_C,"PROTAC-DB"),(b,EXT_C,"External")]:
        if len(data)<2: continue
        kde=gaussian_kde(data); xs=np.linspace(data.min()-0.15*(data.max()-data.min()),data.max()+0.15*(data.max()-data.min()),300)
        axd.plot(xs,kde(xs),color=c,lw=2,label=lbl); axd.fill_between(xs,kde(xs),color=c,alpha=0.25)
    axd.set_ylabel('density'); axd.legend(fontsize=7,loc='upper right')
    bp=axb.boxplot([a,b],labels=['PROTAC-DB\n(train)','External'],patch_artist=True,medianprops=dict(color='black'),showfliers=False)
    for p,cc in zip(bp['boxes'],[TRAIN_C,EXT_C]): p.set_facecolor(cc); p.set_alpha(0.8)
    # jitter overlay (subsample if large)
    rng = np.random.RandomState(42)
    for gidx, data in enumerate([a, b]):
        if len(data) == 0: continue
        xj = gidx + 1 + rng.uniform(-0.18, 0.18, len(data))
        axb.scatter(xj, data, s=3, alpha=0.28, color='#222', linewidths=0)
    axb.set_title(f"p={strow['p_value']:.3g} {strow['sig']}",fontsize=8)
axesC=[[fig.add_subplot(gsC[i,j]) for j in range(2)] for i in range(2)]
density_box(axesC[0][0],axesC[1][0],dcA,dcB,"log10 DC50 (nM)",wstat.iloc[0])
density_box(axesC[0][1],axesC[1][1],dmA,dmB,"Dmax (%)",wstat.iloc[1])
fig.text(0.98,0.5,"(C) Activity density + box",fontsize=10,rotation=90,va='center',ha='center')

# ── Panel D: donuts ──
gsD = gs[3].subgridspec(2,2, wspace=0.3, hspace=0.35)
def donut(ax,oa,both,ob,title):
    ax.pie([oa,both,ob],colors=[TRAIN_C,SHARED_C,EXT_C],startangle=90,counterclock=False,
           wedgeprops=dict(width=0.38,edgecolor='white'),
           labels=["PROTAC-DB","shared","External"],labeldistance=0.75,textprops={'fontsize':6})
    ax.text(0,0,f"{oa+both+ob}",ha='center',va='center',fontsize=9,weight='bold')
    ax.set_title(title,fontsize=8)
d0=[[fig.add_subplot(gsD[i,j]) for j in range(2)] for i in range(2)]
donut(d0[0][0],len(tr_tg-ex_tg),len(tr_tg&ex_tg),len(ex_tg-tr_tg),"Targets (POI)")
donut(d0[0][1],len(tr_e3-ex_e3),len(tr_e3&ex_e3),len(ex_e3-tr_e3),"E3 ligases")
donut(d0[1][0],len(tr_sc-ex_sc),len(tr_sc&ex_sc),len(ex_sc-tr_sc),"Scaffolds")
donut(d0[1][1],len(tr_as-ex_as),len(tr_as&ex_as),len(ex_as-tr_as),"Assay types")
fig.text(0.5,0.30,"(D) Coverage donuts — PROTAC-DB vs External",fontsize=10,ha='center')

# Panel A placed manually at top
fig.savefig(f"{OUT}/SI_multipanel_A4.png")
fig.savefig(f"{OUT}/SI_multipanel_A4.svg")
print("A4 figure saved")
print("DONE")
