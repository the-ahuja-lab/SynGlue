#!/usr/bin/env python3
"""
Final 4-panel figure (Nature/MSB style):
A. DC50 KDE density (quartiles marked)
B. DC50 boxplot + jitter + significance
C. Dmax KDE density (quartiles)
D. Dmax boxplot + jitter + significance
Output: figures/final_plot/pedia_vs_pdb_4panel.{svg,pdf,png(600dpi)}
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

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22, 'grid.linestyle': '--',
                     'svg.fonttype': 'none'})

C_PEDIA, E_PEDIA = '#f2c9a0', '#d19a55'
C_PDB,   E_PDB   = '#a8cbe8', '#5a8db8'
C_Q       = '#cccccc'

def parse_value(v):
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

def sig(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 0.05: return '*'
    return 'ns'

# ── Load ──
pdb = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv", low_memory=False)
pdb_dc50 = pdb['DC50 (nM)'].apply(parse_value).dropna()
pdb_dmax = pdb['Dmax (%)'].apply(parse_value).dropna()
pdb_dc50 = pdb_dc50[pdb_dc50 > 0]
pdb_dmax = pdb_dmax[(pdb_dmax >= 0) & (pdb_dmax <= 100)]

tack = pd.read_parquet(f"{BASE}/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet")
pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
pedia_dc50 = pd.to_numeric(pedia.loc[pedia['Value_Type']=='DC50','Value'], errors='coerce').dropna()
pedia_dmax = pd.to_numeric(pedia.loc[pedia['Value_Type']=='Dmax','Value'], errors='coerce').dropna()
pedia_dc50 = pedia_dc50[pedia_dc50 > 0]
pedia_dmax = pedia_dmax[(pedia_dmax >= 0) & (pedia_dmax <= 100)]

lp_dc50 = np.log10(pdb_dc50.values); le_dc50 = np.log10(pedia_dc50.values)
lp_dmax = pdb_dmax.values; le_dmax = pedia_dmax.values

# tests
u50, p50 = stats.mannwhitneyu(le_dc50, lp_dc50, alternative='two-sided')
u95, p95 = stats.mannwhitneyu(le_dmax, lp_dmax, alternative='two-sided')

def draw_kde(ax, a, b, xlab, title, a_lab, b_lab, q=None, xlim=None):
    """Overlaid KDE with quartile marks + N annotation."""
    from scipy.stats import gaussian_kde
    def kd(v, bw=None):
        try:
            k = gaussian_kde(v, bw_method=0.3)
        except Exception:
            k = gaussian_kde(v)
        return k
    xs = np.linspace(min(a.min(), b.min()) - 0.2, max(a.max(), b.max()) + 0.2, 400)
    ax.fill_between(xs, kd(a)(xs), color=C_PEDIA, alpha=0.45, label=f'{a_lab} (n={len(a)})')
    ax.fill_between(xs, kd(b)(xs), color=C_PDB,   alpha=0.45, label=f'{b_lab} (n={len(b)})')
    ax.plot(xs, kd(a)(xs), color=E_PEDIA, lw=1.5)
    ax.plot(xs, kd(b)(xs), color=E_PDB,   lw=1.5)
    # quartiles of combined
    if q is not None:
        for qv, lab in zip(q, ['Q1','Q2','Q3']):
            ax.axvline(qv, color=C_Q, ls=':', lw=1)
            ax.text(qv, ax.get_ylim()[1]*0.95, f' {lab}', fontsize=7.5, color='#666')
    # medians
    for v, c in [(np.median(a), E_PEDIA), (np.median(b), E_PDB)]:
        ax.axvline(v, color=c, ls='--', lw=1.4)
    ax.set_xlabel(xlab); ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8.5, framealpha=0.9)
    if xlim: ax.set_xlim(xlim)

def draw_box(ax, a, b, ylab, title, sig_text, xlim=None):
    """Boxplot + jittered semi-transparent points."""
    bp = ax.boxplot([a, b], positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=True, medianprops=dict(color='black', lw=1.8),
                    whiskerprops=dict(color='#555', lw=1.2), capprops=dict(color='#555', lw=1.2))
    for patch, col in zip(bp['boxes'], [C_PEDIA, C_PDB]):
        patch.set_facecolor(col); patch.set_edgecolor('#555'); patch.set_linewidth(1.2)
    # jitter
    rng = np.random.RandomState(42)
    for i, (v, col) in enumerate([(a, E_PEDIA), (b, E_PDB)]):
        jx = rng.normal(0, 0.06, len(v))
        ax.scatter(i + jx, v, s=9, alpha=0.30, color=col, edgecolor='none', zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f'PROTACpedia\n(n={len(a)})', f'PROTAC-DB\n(n={len(b)})'])
    ax.set_ylabel(ylab); ax.set_title(title)
    # significance
    ymax = max(a.max(), b.max())
    ax.text(0.5, ymax + (ymax*0.04), sig_text, ha='center', fontsize=13, weight='bold')
    if xlim: ax.set_ylim(xlim)

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# A: DC50 KDE
q50 = np.percentile(np.concatenate([le_dc50, lp_dc50]), [25, 50, 75])
draw_kde(axes[0,0], le_dc50, lp_dc50, 'log10(DC50 / nM)', 'A · DC50 density (lower = more potent)',
         'PROTACpedia', 'PROTAC-DB', q=q50)
# B: DC50 box+jitter
draw_box(axes[0,1], le_dc50, lp_dc50, 'log10(DC50 / nM)', 'B · DC50 boxplot + jitter',
         f'{sig(p50)}  p = {p50:.1e}')
# C: Dmax KDE
q95 = np.percentile(np.concatenate([le_dmax, lp_dmax]), [25, 50, 75])
draw_kde(axes[1,0], le_dmax, lp_dmax, 'Dmax (%)', 'C · Dmax density (higher = more efficacious)',
         'PROTACpedia', 'PROTAC-DB', q=q95, xlim=(0, 100))
# D: Dmax box+jitter
draw_box(axes[1,1], le_dmax, lp_dmax, 'Dmax (%)', 'D · Dmax boxplot + jitter',
         f'{sig(p95)}  p = {p95:.1e}', xlim=(0, 110))

fig.suptitle('PROTACpedia vs PROTAC-DB — potency & degradation-efficacy distributions', fontsize=14, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIG}/pedia_vs_pdb_4panel.svg")
fig.savefig(f"{FIG}/pedia_vs_pdb_4panel.pdf")
fig.savefig(f"{FIG}/pedia_vs_pdb_4panel.png", dpi=600)
plt.close(fig)
print("Saved: pedia_vs_pdb_4panel.{svg,pdf,png(600dpi)}", flush=True)
