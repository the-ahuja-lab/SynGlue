#!/usr/bin/env python3
"""
Box plots + rank tests (Mann-Whitney U / Wilcoxon rank-sum) for DC50 and Dmax
distributions across the activity databases (same data as figA/figB_dc50/dmax_distributions).
Saves ONLY to: Projects/SynGlue/revision/figures/final_plot/
Outputs:
  figA_dc50_boxplot.svg/.png     (log10 DC50 box plots per database)
  figB_dmax_boxplot.svg/.png     (Dmax % box plots per database)
  wilcoxon_dc50_pairwise.csv     (pairwise Mann-Whitney U, DC50)
  wilcoxon_dmax_pairwise.csv     (pairwise Mann-Whitney U, Dmax)
  wilcoxon_results.md
"""
import os, sys, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

BASE = "/storage/savi/saveenas/Projects/SynGlue"
EXT = f"{BASE}/external_benchmark_data"
OUT = f"{BASE}/revision/figures/final_plot"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 300, 'savefig.bbox': 'tight'})

def parse(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s.upper() in ("N.D.", "N/D", "NA", "N/A", ""): return np.nan
    s2 = re.sub(r'[><~=]', '', s)
    parts = re.split(r'[/;]', s2)
    vals = []
    for p in parts:
        p = p.strip()
        if p.upper() in ("N.D.", "N/D", "NA", "N/A", ""): continue
        try: vals.append(float(p))
        except: continue
    return np.median(vals) if vals else np.nan

tack = pd.read_parquet(f"{EXT}/tack_raw/data_train-00000-of-00001.parquet")
pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
tpddb = tack[tack['Database'].str.contains('TPDdb', na=False)]
pdb_tack = tack[tack['Database'].str.contains('PROTAC-DB', na=False)]
pdb_local = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/PROTAC_DB_with_features.csv", low_memory=False)

def dc50_series(df, is_tack=True):
    if is_tack:
        v = pd.to_numeric(df.loc[df['Value_Type'] == 'DC50', 'Value'], errors='coerce').dropna()
    else:
        v = df['DC50 (nM)'].apply(parse).dropna()
    v = v[v > 0]
    return np.log10(v.values)   # log10 nM

def dmax_series(df, is_tack=True):
    if is_tack:
        v = pd.to_numeric(df.loc[df['Value_Type'] == 'Dmax', 'Value'], errors='coerce').dropna()
    else:
        v = df['Dmax (%)'].apply(parse).dropna()
    return v.values  # %

DB = [
    ("TACK (full)", tack, True),
    ("TPDdb (TACK core)", tpddb, True),
    ("TACK/PROTAC-DB 3.0", pdb_tack, True),
    ("PROTACpedia", pedia, True),
    ("PROTAC-DB (training)", pdb_local, False),
]

# ─── DC50 box plot ───────────────────────────────────
dc_sets = [(n, dc50_series(df, t)) for n, df, t in DB]
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(dc_sets))
bp = ax.boxplot([s for _, s in dc_sets], positions=pos, widths=0.55, patch_artist=True,
                medianprops=dict(color='black', lw=1.5), showfliers=False)
for patch, (n, s) in zip(bp['boxes'], dc_sets):
    patch.set_facecolor('#1f77b4'); patch.set_alpha(0.75)
# jitter strip
rng = np.random.RandomState(42)
for i, (n, s) in enumerate(dc_sets):
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s, s=6, alpha=0.25, color='gray')
ax.set_xticks(pos)
ax.set_xticklabels([f"{n}\n(n={len(s)})" for n, s in dc_sets], fontsize=8, rotation=0)
ax.set_ylabel('log10(DC50 / nM)')
ax.set_title('DC50 distribution by database (box + jitter)')
plt.tight_layout()
fig.savefig(f"{OUT}/figA_dc50_boxplot.png")
fig.savefig(f"{OUT}/figA_dc50_boxplot.svg")
plt.close(fig)

# ─── Dmax box plot ───────────────────────────────────
dm_sets = [(n, dmax_series(df, t)) for n, df, t in DB]
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([s for _, s in dm_sets], positions=pos, widths=0.55, patch_artist=True,
                medianprops=dict(color='black', lw=1.5), showfliers=False)
for patch, (n, s) in zip(bp['boxes'], dm_sets):
    patch.set_facecolor('#ff7f0e'); patch.set_alpha(0.75)
for i, (n, s) in enumerate(dm_sets):
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s, s=6, alpha=0.25, color='gray')
ax.set_xticks(pos)
ax.set_xticklabels([f"{n}\n(n={len(s)})" for n, s in dm_sets], fontsize=8)
ax.set_ylabel('Dmax (%)')
ax.set_title('Dmax distribution by database (box + jitter)')
plt.tight_layout()
fig.savefig(f"{OUT}/figB_dmax_boxplot.png")
fig.savefig(f"{OUT}/figB_dmax_boxplot.svg")
plt.close(fig)

# ─── Wilcoxon rank-sum (Mann-Whitney U) pairwise ─────
def pairwise(series_list, label):
    names = [s[0] for s in series_list]
    rows = []
    for a, b in combinations(range(len(series_list)), 2):
        x = series_list[a][1]; y = series_list[b][1]
        U, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        med_diff = np.median(x) - np.median(y)
        # rank-biserial / common language effect size (Cliff's delta approx)
        n1, n2 = len(x), len(y)
        eff = (2 * U) / (n1 * n2) - 1   # rank-biserial ~ measure of separation
        rows.append(dict(group_a=names[a], group_b=names[b],
                         n_a=n1, n_b=n2,
                         median_a=round(float(np.median(x)), 3),
                         median_b=round(float(np.median(y)), 3),
                         median_diff_a_minus_b=round(float(med_diff), 3),
                         U=round(float(U), 1), p_value=p,
                         rank_biserial=round(float(eff), 3)))
    d = pd.DataFrame(rows)
    # p adjusted (Bonferroni) and significance flags
    n_tests = len(d)
    d['p_bonferroni'] = d['p_value'] * n_tests
    d['signif_0.05'] = d['p_bonferroni'] < 0.05
    d['signif_0.01'] = d['p_bonferroni'] < 0.01
    d.to_csv(f"{OUT}/{label}_pairwise.csv", index=False)
    return d

dc_t = pairwise(dc_sets, "wilcoxon_dc50")
dm_t = pairwise(dm_sets, "wilcoxon_dmax")

# ─── Summary MD ──────────────────────────────────────
lines = ["# Box plots + Wilcoxon rank tests — DC50 & Dmax distributions by database",
         "",
         f"- Method: box plots with jitter; two-sample rank test (Mann-Whitney U, a.k.a. Wilcoxon rank-sum); Bonferroni-corrected",
         f"- Saved only to: `figures/final_plot/`",
         "",
         "## Sample sizes (DC50 / Dmax)",
         "| Database | DC50 n | Dmax n |",
         "|----------|--------|--------|"]
for n, df, t in DB:
    lines.append(f"| {n} | {len(dc50_series(df, t))} | {len(dmax_series(df, t))} |")
lines += ["", "## DC50 pairwise (Mann-Whitney U; p=Bonferroni)", "",
          "| A | B | U | p (adj) | signif | rank-biserial |", "|---|---|---|---|---|---|"]
for _, r in dc_t.iterrows():
    lines.append(f"| {r['group_a']} | {r['group_b']} | {r['U']:.0f} | {r['p_bonferroni']:.3g} | {'*' if r['signif_0.05'] else ''} | {r['rank_biserial']} |")
lines += ["", "## Dmax pairwise (Mann-Whitney U; p=Bonferroni)", "",
          "| A | B | U | p (adj) | signif | rank-biserial |", "|---|---|---|---|---|---|"]
for _, r in dm_t.iterrows():
    lines.append(f"| {r['group_a']} | {r['group_b']} | {r['U']:.0f} | {r['p_bonferroni']:.3g} | {'*' if r['signif_0.05'] else ''} | {r['rank_biserial']} |")
lines += ["", "* = significant at α=0.05 after Bonferroni. Positive rank-biserial = group A tends higher.",
          "", "Interpretation: distributions across databases are compared to establish whether the external "
          "benchmark pools differ from the training distribution (PROTAC-DB, local)." ]
open(f"{OUT}/wilcoxon_results.md", "w").write("\n".join(lines))
print(dc_t.to_string(index=False))
print()
print(dm_t.to_string(index=False))
print(f"\nSaved to {OUT}: figA_dc50_boxplot, figB_dmax_boxplot, wilcoxon_*_pairwise.csv, wilcoxon_results.md")
