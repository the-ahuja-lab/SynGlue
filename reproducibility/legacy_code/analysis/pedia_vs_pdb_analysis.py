#!/usr/bin/env python3
"""
PROTACpedia vs PROTAC-DB — Distribution, Quartile, Variability & Statistical Analysis
DC50 (log10 nM) + Dmax (%): KDE densities, boxplots+jitter, Mann-Whitney U tests,
quartile enrichment, multi-panel figure (SVG/PDF/PNG600dpi), summary tables, interpretation.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures/final_plot"
os.makedirs(FIG, exist_ok=True)

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--'})

# ── Colors (consistent across panels) ──
C_PEDIA = '#f2c9a0'   # light orange (PROTACpedia)
C_PDB   = '#a8cbe8'   # light blue   (PROTAC-DB)
E_PEDIA = '#d19a55'
E_PDB   = '#5a8db8'

def parse_value(v):
    """Parse DC50/Dmax strings -> float (strip >,<,~, take median of ranges)."""
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.","N/D","NA","N/A",""): return np.nan
    s = re.sub(r'[><~=]', '', s)
    vals = []
    for p in re.split(r'[/;,]', s):
        p = p.strip()
        if p.upper() in ("N.D.","N/D","NA","N/A",""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

# ══════════════ 1. LOAD DATA ══════════════
print("Loading data...", flush=True)

# PROTAC-DB training (Upadted_PROTAC_features.csv)
pdb = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv", low_memory=False)
pdb_dc50 = pdb['DC50 (nM)'].apply(parse_value).dropna()
pdb_dmax = pdb['Dmax (%)'].apply(parse_value).dropna()
pdb_dc50 = pdb_dc50[pdb_dc50 > 0]
pdb_dmax = pdb_dmax[(pdb_dmax >= 0) & (pdb_dmax <= 100)]

# PROTACpedia (TACK Database column contains 'PROTACpedia')
tack = pd.read_parquet(f"{BASE}/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet")
pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)].copy()
pedia_dc50 = pd.to_numeric(pedia.loc[pedia['Value_Type']=='DC50','Value'], errors='coerce').dropna()
pedia_dmax = pd.to_numeric(pedia.loc[pedia['Value_Type']=='Dmax','Value'], errors='coerce').dropna()
pedia_dc50 = pedia_dc50[pedia_dc50 > 0]
pedia_dmax = pedia_dmax[(pedia_dmax >= 0) & (pedia_dmax <= 100)]

# log10 DC50
lp_dc50 = np.log10(pdb_dc50.values)
le_dc50 = np.log10(pedia_dc50.values)

print(f"DC50  — PROTACpedia n={len(le_dc50)} | PROTAC-DB n={len(lp_dc50)}", flush=True)
print(f"Dmax  — PROTACpedia n={len(pedia_dmax)} | PROTAC-DB n={len(pdb_dmax)}", flush=True)

# ══════════════ 2. STATISTICS ══════════════
def summarize(vals):
    v = np.asarray(vals, float)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "var": float(v.var(ddof=1)), "q1": float(q1), "q2": float(q2), "q3": float(q3),
            "iqr": float(q3-q1), "min": float(v.min()), "max": float(v.max())}

sum_dc50_pedia = summarize(le_dc50)
sum_dc50_pdb   = summarize(lp_dc50)
sum_dmax_pedia = summarize(pedia_dmax.values)
sum_dmax_pdb   = summarize(pdb_dmax.values)

# Mann-Whitney U (independent samples — PROTACpedia and PROTAC-DB are independent)
u_dc50, p_dc50 = stats.mannwhitneyu(le_dc50, lp_dc50, alternative='two-sided')
u_dmax, p_dmax = stats.mannwhitneyu(pedia_dmax.values, pdb_dmax.values, alternative='two-sided')

# Cliff's delta (non-parametric effect size)
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n, m = len(a), len(b)
    gt = np.sum([np.sum(a[i] > b) for i in range(n)])
    lt = np.sum([np.sum(a[i] < b) for i in range(n)])
    return (gt - lt) / (n * m)

cd_dc50 = cliffs_delta(le_dc50, lp_dc50)
cd_dmax = cliffs_delta(pedia_dmax.values, pdb_dmax.values)

# median differences (on original scales)
med_diff_dc50 = np.median(lp_dc50) - np.median(le_dc50)  # PROTAC-DB minus pedia (log10)
med_diff_dmax = np.median(pdb_dmax.values) - np.median(pedia_dmax.values)

def sig(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 0.05: return '*'
    return 'ns'

print("\n=== STATISTICAL TESTS ===", flush=True)
print(f"DC50: U={u_dc50:.1f} p={p_dc50:.3e} Cliff's δ={cd_dc50:+.3f} | medΔ(PROTAC-DB−pedia)={med_diff_dc50:+.3f} log10", flush=True)
print(f"Dmax: U={u_dmax:.1f} p={p_dmax:.3e} Cliff's δ={cd_dmax:+.3f} | medΔ(PROTAC-DB−pedia)={med_diff_dmax:+.2f} pp", flush=True)

# Quartile enrichment
def quartile_enrichment(a, b, lower_is_better, name):
    allv = np.concatenate([a, b])
    q = np.percentile(allv, [25, 50, 75])
    lab = ['Q1','Q2','Q3','Q4']
    rows = []
    for arr, dblab in [(a, 'PROTACpedia'), (b, 'PROTAC-DB')]:
        frac = []
        for i, (lo, hi) in enumerate(zip([-np.inf]+list(q), list(q)+[np.inf])):
            m = (arr >= lo) & (arr < hi) if i < 3 else (arr >= lo)
            frac.append(m.mean())
        rows.append((dblab, frac))
    # desirable tail
    des_a = np.mean(a <= q[0]) if lower_is_better else np.mean(a >= q[2])
    des_b = np.mean(b <= q[0]) if lower_is_better else np.mean(b >= q[2])
    return rows, des_a, des_b

en_dc50, des_dc50_p, des_dc50_b = quartile_enrichment(le_dc50, lp_dc50, True, 'DC50')
en_dmax, des_dmax_p, des_dmax_b = quartile_enrichment(pedia_dmax.values, pdb_dmax.values, False, 'Dmax')

print("\n=== QUARTILE ENRICHMENT (DC50, combined quartiles) ===", flush=True)
for lab, f in en_dc50:
    print(f"  {lab}: Q1={f[0]*100:.1f}% Q2={f[1]*100:.1f}% Q3={f[2]*100:.1f}% Q4={f[3]*100:.1f}%  (low-DC50/potent tail: {des_dc50_p if lab=='PROTACpedia' else des_dc50_b:.1%})", flush=True)
print("=== QUARTILE ENRICHMENT (Dmax) ===", flush=True)
for lab, f in en_dmax:
    print(f"  {lab}: Q1={f[0]*100:.1f}% Q2={f[1]*100:.1f}% Q3={f[2]*100:.1f}% Q4={f[3]*100:.1f}%  (high-Dmax tail: {des_dmax_p if lab=='PROTACpedia' else des_dmax_b:.1%})", flush=True)

# Save stats JSON
stats_out = {
    "dc50_pedia": sum_dc50_pedia, "dc50_pdb": sum_dc50_pdb,
    "dmax_pedia": sum_dmax_pedia, "dmax_pdb": sum_dmax_pdb,
    "tests": {
        "dc50": {"test": "Mann-Whitney U (independent)", "U": u_dc50, "p": p_dc50,
                 "cliffs_delta": cd_dc50, "median_diff_log10": med_diff_dc50,
                 "median_diff_nM": 10**med_diff_dc50, "sig": sig(p_dc50)},
        "dmax": {"test": "Mann-Whitney U (independent)", "U": u_dmax, "p": p_dmax,
                 "cliffs_delta": cd_dmax, "median_diff_pp": med_diff_dmax, "sig": sig(p_dmax)},
    },
    "quartile_enrichment": {
        "dc50_pedia": en_dc50[0][1], "dc50_pdb": en_dc50[1][1],
        "dmax_pedia": en_dmax[0][1], "dmax_pdb": en_dmax[1][1],
        "dc50_potent_tail_pedia": des_dc50_p, "dc50_potent_tail_pdb": des_dc50_b,
        "dmax_tail_pedia": des_dmax_p, "dmax_tail_pdb": des_dmax_b,
    },
}
json.dump(stats_out, open(f"{FIG}/pedia_vs_pdb_stats.json", "w"), indent=2, default=float)

# ── Summary tables (CSV) ──
tab1 = pd.DataFrame([
    {"Endpoint":"DC50 (log10 nM)","Database":"PROTACpedia","n":sum_dc50_pedia["n"],"Mean":round(sum_dc50_pedia["mean"],3),
     "SD":round(sum_dc50_pedia["sd"],3),"Variance":round(sum_dc50_pedia["var"],3),"Q1":round(sum_dc50_pedia["q1"],3),
     "Median":round(sum_dc50_pedia["q2"],3),"Q3":round(sum_dc50_pedia["q3"],3),"IQR":round(sum_dc50_pedia["iqr"],3)},
    {"Endpoint":"DC50 (log10 nM)","Database":"PROTAC-DB","n":sum_dc50_pdb["n"],"Mean":round(sum_dc50_pdb["mean"],3),
     "SD":round(sum_dc50_pdb["sd"],3),"Variance":round(sum_dc50_pdb["var"],3),"Q1":round(sum_dc50_pdb["q1"],3),
     "Median":round(sum_dc50_pdb["q2"],3),"Q3":round(sum_dc50_pdb["q3"],3),"IQR":round(sum_dc50_pdb["iqr"],3)},
    {"Endpoint":"Dmax (%)","Database":"PROTACpedia","n":sum_dmax_pedia["n"],"Mean":round(sum_dmax_pedia["mean"],2),
     "SD":round(sum_dmax_pedia["sd"],2),"Variance":round(sum_dmax_pedia["var"],2),"Q1":round(sum_dmax_pedia["q1"],1),
     "Median":round(sum_dmax_pedia["q2"],1),"Q3":round(sum_dmax_pedia["q3"],1),"IQR":round(sum_dmax_pedia["iqr"],1)},
    {"Endpoint":"Dmax (%)","Database":"PROTAC-DB","n":sum_dmax_pdb["n"],"Mean":round(sum_dmax_pdb["mean"],2),
     "SD":round(sum_dmax_pdb["sd"],2),"Variance":round(sum_dmax_pdb["var"],2),"Q1":round(sum_dmax_pdb["q1"],1),
     "Median":round(sum_dmax_pdb["q2"],1),"Q3":round(sum_dmax_pdb["q3"],1),"IQR":round(sum_dmax_pdb["iqr"],1)},
])
tab1.to_csv(f"{FIG}/pedia_vs_pdb_summary_table.csv", index=False)

tab2 = pd.DataFrame([
    {"Endpoint":"DC50","Test":"Mann-Whitney U (independent)","Statistic":u_dc50,"P-value":f"{p_dc50:.2e}",
     "Effect size (Cliff's δ)":round(cd_dc50,3),"Direction":"PROTAC-DB median lower (more potent)" if med_diff_dc50<0 else "PROTACpedia median lower","Sig":sig(p_dc50)},
    {"Endpoint":"Dmax","Test":"Mann-Whitney U (independent)","Statistic":u_dmax,"P-value":f"{p_dmax:.2e}",
     "Effect size (Cliff's δ)":round(cd_dmax,3),"Direction":"PROTAC-DB median higher (more efficacious)" if med_diff_dmax>0 else "PROTACpedia median higher","Sig":sig(p_dmax)},
])
tab2.to_csv(f"{FIG}/pedia_vs_pdb_test_table.csv", index=False)

print("\nSaved: summary_table.csv + test_table.csv + stats.json", flush=True)
