#!/usr/bin/env python3
"""Final coverage + density visualization set for the rebuttal.
Produces (all in outputs/):
  A. 3-panel Venn (scaffolds / targets / E3)  training(PROTAC-DB) vs External
  B. Density + box for log10 DC50 (nM) and Dmax (%) with Q1-Q4 and Wilcoxon rank-sum
  C. Donut plots (targets / E3 / scaffolds / assays) — % from each database, harmonized colors
  + PROTAC-DB 3.0 coverage table + wilcoxon stats CSV + REPORT.md
Harmonized color scheme used throughout:
  PROTAC-DB (training) = BLUE #1f77b4 | External = ORANGE #ff7f0e | Shared = GRAY #7f7f7f
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
RDLogger.DisableLog('rdApp.*')

BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV  = f"{BASE}/revision"
DATA = f"{BASE}/external_benchmark_data"
S53  = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT  = f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs"
os.makedirs(OUT, exist_ok=True)

TRAIN_C = '#1f77b4'; EXT_C = '#ff7f0e'; SHARED_C = '#7f7f7f'
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.bbox': 'tight'})

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except: return None
def norm(series): return series.astype(str).str.strip().str.upper()

# ── Load datasets ────────────────────────────────────
pdb = pd.read_csv(f"{S53}/PROTAC_DB_with_features.csv", low_memory=False)
ext = pd.read_csv(f"{REV}/benchmarking/external_not_in_training.csv", low_memory=False)

# training scaffolds/targets/e3
tr_smi = set(c for c in map(canon, pdb['Smiles'].dropna().astype(str)) if c)
tr_scaff = set(c for c in map(murcko, pdb['Smiles'].dropna().astype(str)) if c)
tr_tgt = set(norm(pdb['Target'][pdb['Target'].notna()]))
tr_e3 = set(norm(pdb['E3 ligase'][pdb['E3 ligase'].notna()]))
tr_assay = set(pdb['Assay (DC50/Dmax)'].dropna().astype(str).str.strip())

# external
ex_smi = set(c for c in map(canon, ext['SMILES'].dropna().astype(str)) if c)
ex_scaff = set(c for c in map(murcko, ext['SMILES'].dropna().astype(str)) if c)
ex_tgt = set(norm(ext['Protein_of_Interest'][ext['Protein_of_Interest'].notna()]))
ex_e3 = set(norm(ext['E3_Ligase'][ext['E3_Ligase'].notna()]))
ex_assay = set(ext['Assay'].dropna().astype(str).str.strip())

def venn3(a, b):
    return dict(only_a=len(a-b), both=len(a&b), only_b=len(b-a),
                a=len(a), b=len(b))

# ── PROTAC-DB 3.0 coverage table ─────────────────────
coverage = pd.DataFrame([
    dict(level="Compounds (unique PROTACs)", protectacdb=len(tr_smi), external=len(ex_smi),
         shared=len(tr_smi & ex_smi)),
    dict(level="Scaffolds (Murcko)", protectacdb=len(tr_scaff), external=len(ex_scaff),
         shared=len(tr_scaff & ex_scaff)),
    dict(level="Targets (POI, name)", protectacdb=len(tr_tgt), external=len(ex_tgt),
         shared=len(tr_tgt & ex_tgt)),
    dict(level="E3 ligases", protectacdb=len(tr_e3), external=len(ex_e3),
         shared=len(tr_e3 & ex_e3)),
    dict(level="Assay types", protectacdb=len(tr_assay), external=len(ex_assay),
         shared=len(tr_assay & ex_assay)),
])
coverage.to_csv(f"{OUT}/coverage_table.csv", index=False)

# ── ACTIVITY: log10 DC50 & Dmax ──────────────────────
def dc50_pdb(df):
    v = pd.to_numeric(df['DC50 (nM)'], errors='coerce').dropna()
    v = v[v > 0]
    return np.log10(v.values)
def dmax_pdb(df):
    v = pd.to_numeric(df['Dmax (%)'], errors='coerce').dropna()
    return v[(v>=0)&(v<=100)].values

def ext_dc50(df):
    d = df[df['Value_Type']=='DC50'].copy()
    d['val'] = pd.to_numeric(d['Value'], errors='coerce')
    d = d[(d['val']>0)]
    med = d.groupby('canonical_SMILES')['val'].median().dropna()
    return np.log10(med.values)
def ext_dmax(df):
    d = df[df['Value_Type']=='Dmax'].copy()
    d['val'] = pd.to_numeric(d['Value'], errors='coerce')
    d = d[(d['val']>=0)&(d['val']<=100)]
    mx = d.groupby('canonical_SMILES')['val'].max().dropna()
    return mx.values

dc_pdb = dc50_pdb(pdb); dm_pdb = dmax_pdb(pdb)
dc_ext = ext_dc50(ext); dm_ext = ext_dmax(ext)

def quartiles(x):
    q = np.percentile(x, [25,50,75])
    return dict(Q1=round(float(q[0]),3), Q2=round(float(q[1]),3), Q3=round(float(q[2]),3),
                Q4_min=round(float(q[2]),3), n=len(x), min=round(float(x.min()),3), max=round(float(x.max()),3))

def wilcox(a, b, lab):
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    stars = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    return dict(endpoint=lab, U=round(float(U),1), p_value=p, stars=stars, n_a=len(a), n_b=len(b))

stats_rows = []
for lab, a, b in [("log10 DC50 (nM)", dc_pdb, dc_ext), ("Dmax (%)", dm_pdb, dm_ext)]:
    stats_rows.append(wilcox(a, b, lab))
    print(f"{lab}: train n={len(a)} ext n={len(b)} | wilcoxon p={stats_rows[-1]['p_value']:.3g} {stats_rows[-1]['stars']}")
    print(f"   PROTAC-DB Q: {quartiles(a)}")
    print(f"   External Q : {quartiles(b)}")
wstat = pd.DataFrame(stats_rows); wstat.to_csv(f"{OUT}/wilcoxon_stats.csv", index=False)

# ── PLOT A: 3-panel Venn ─────────────────────────────
def draw_venn(ax, only_a, both, only_b, title):
    r = 1.0; cx1, cx2 = 0.85, 1.85; cy = 1.1
    ax.add_patch(plt.Circle((cx1,cy), r, color=TRAIN_C, alpha=0.35))
    ax.add_patch(plt.Circle((cx2,cy), r, color=EXT_C, alpha=0.35))
    ax.text(cx1-0.95, cy, f"{only_a}", ha='center', fontsize=14, weight='bold')
    ax.text(cx2+0.95, cy, f"{only_b}", ha='center', fontsize=14, weight='bold')
    ax.text((cx1+cx2)/2, cy, f"{both}", ha='center', fontsize=14, weight='bold')
    ax.text((cx1-0.5)/1, 1.95, f"PROTAC-DB\n{only_a+both}", ha='center', color=TRAIN_C, fontsize=9)
    ax.text((cx2+0.5)/1+0.0, 1.95, f"External\n{both+only_b}", ha='center', color=EXT_C, fontsize=9)
    ax.set_xlim(0,2.7); ax.set_ylim(0,2.2); ax.axis('off'); ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(1,3, figsize=(13,4.6))
for ax, title, a, b in zip(axes,
     ["Scaffolds","Targets (POI)","E3 ligases"],
     [tr_scaff, tr_tgt, tr_e3], [ex_scaff, ex_tgt, ex_e3]):
    v = venn3(a,b); draw_venn(ax, v['only_a'], v['both'], v['only_b'], title)
fig.suptitle("Coverage Venn — PROTAC-DB (training) vs External", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/plotA_venn.png"); fig.savefig(f"{OUT}/plotA_venn.svg"); plt.close(fig)

# ── PLOT B: density + box (log10 DC50 & Dmax) ───────
def density_box(axd, axb, a, b, lab, axlab, qa, qb, statrow):
    for ax, data, c, lbl in [(axd, a, TRAIN_C, "PROTAC-DB (training)"),
                             (axd, b, EXT_C, "External")]:
        if len(data)<2: continue
        kde = gaussian_kde(data)
        xs = np.linspace(data.min()-0.2*abs(data.max()-data.min()), data.max()+0.2*abs(data.max()-data.min()), 300)
        axd.plot(xs, kde(xs), color=c, lw=2, label=lbl)
        axd.fill_between(xs, kde(xs), color=c, alpha=0.25)
    axd.set_ylabel("density"); axd.set_xlabel(axlab); axd.legend(fontsize=8, loc='upper right')
    box = axb.boxplot([a,b], labels=["PROTAC-DB\n(train)","External"], patch_artist=True,
                      medianprops=dict(color='black'), showfliers=False)
    for patch, c in zip(box['boxes'], [TRAIN_C, EXT_C]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    axb.set_ylabel(axlab)
    p = statrow['p_value']
    axb.set_title(f"Wilcoxon rank-sum p={p:.3g} {'('+statrow['stars']+')'}", fontsize=9)

fig, axes = plt.subplots(2,2, figsize=(9,8))
# DC50
qa=quartiles(dc_pdb); qb=quartiles(dc_ext)
density_box(axes[0,0], axes[1,0], dc_pdb, dc_ext, "DC50", "log10(DC50 / nM)", qa, qb, stats_rows[0])
axes[0,0].set_title("log10 DC50 density — PROTAC-DB vs External")
# Dmax
qa=quartiles(dm_pdb); qb=quartiles(dm_ext)
density_box(axes[0,1], axes[1,1], dm_pdb, dm_ext, "Dmax", "Dmax (%)", qa, qb, stats_rows[1])
axes[0,1].set_title("Dmax density — PROTAC-DB vs External")
fig.tight_layout()
fig.savefig(f"{OUT}/plotB_density_box.png"); fig.savefig(f"{OUT}/plotB_density_box.svg"); plt.close(fig)

# ── PLOT C: donuts (target / E3 / scaffold / assay) ──
def donut(ax, only_a, both, only_b, title):
    sizes = [only_a, both, only_b]
    colors = [TRAIN_C, SHARED_C, EXT_C]
    lab = ["PROTAC-DB\nonly", "shared", "External\nonly"]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor='white'),
                       labels=lab, labeldistance=0.78, textprops={'fontsize':7})
    ax.text(0,0, f"{sum(sizes)}", ha='center', va='center', fontsize=11, weight='bold')
    ax.set_title(title, fontsize=10)

fig, axes = plt.subplots(2,2, figsize=(10,9))
donut(axes[0,0], *[venn3(tr_tgt,ex_tgt)[k] for k in ['only_a','both','only_b']], "Targets (POI)")
donut(axes[0,1], *[venn3(tr_e3,ex_e3)[k] for k in ['only_a','both','only_b']], "E3 ligases")
donut(axes[1,0], *[venn3(tr_scaff,ex_scaff)[k] for k in ['only_a','both','only_b']], "Scaffolds")
donut(axes[1,1], *[venn3(tr_assay,ex_assay)[k] for k in ['only_a','both','only_b']], "Assay types")
fig.suptitle("Coverage donuts — PROTAC-DB vs External (counts)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/plotC_donuts.png"); fig.savefig(f"{OUT}/plotC_donuts.svg"); plt.close(fig)
print("Saved all plots to", OUT)
