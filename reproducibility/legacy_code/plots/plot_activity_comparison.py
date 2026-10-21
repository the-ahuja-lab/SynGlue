#!/usr/bin/env python3
"""
Activity-data comparison across TACK / TPDdb / PROTACpedia / PROTAC-DB.
Shows DC50 & Dmax distributions, operators, assay times, and coverage.
Output: protac_db_justification/figures/ (activity figures) + results/9_activity.csv
"""
import os, sys, re, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
EXT = f"{BASE}/external_benchmark_data"
OUT = f"{REV}/protac_db_justification"
RES = f"{OUT}/results"; FIG = f"{OUT}/figures"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

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

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    tack = pd.read_parquet(f"{EXT}/tack_raw/data_train-00000-of-00001.parquet")
    pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
    tpddb = tack[tack['Database'].str.contains('TPDdb', na=False)]
    pdb_tack = tack[tack['Database'].str.contains('PROTAC-DB', na=False)]
    pdb_local = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/PROTAC_DB_with_features.csv", low_memory=False)

    # ── Build activity summary ──
    def summarize(name, df, dc50_col='Value', dmax_col='Value', is_tack=True):
        if is_tack:
            dc = pd.to_numeric(df.loc[df['Value_Type']=='DC50', dc50_col], errors='coerce').dropna()
            dm = pd.to_numeric(df.loc[df['Value_Type']=='Dmax', dmax_col], errors='coerce').dropna()
            ops = df['Value_Operator'].value_counts(dropna=False)
            n_ops = ops.get('>',0)+ops.get('<',0)+ops.get('<=',0)+ops.get('>=',0)+ops.get('~',0)
            times = df['Assay_Time'].dropna()
            n_assay_time = df['Assay_Time'].notna().sum()
            cell = df['Cell_Line'].nunique()
        else:
            dc = df['DC50 (nM)'].apply(parse).dropna()
            dm = df['Dmax (%)'].apply(parse).dropna()
            n_ops = df['DC50 (nM)'].astype(str).str.contains('[><~]', regex=True).sum()
            times = pd.Series(dtype=float)
            n_assay_time = 0
            cell = 0
        return {
            "database": name,
            "n_records": len(df),
            "n_DC50": len(dc),
            "DC50_median_nM": round(float(dc.median()),1) if len(dc) else np.nan,
            "DC50_q25": round(float(dc.quantile(0.25)),1) if len(dc) else np.nan,
            "DC50_q75": round(float(dc.quantile(0.75)),1) if len(dc) else np.nan,
            "n_Dmax": len(dm),
            "Dmax_median_pct": round(float(dm.median()),1) if len(dm) else np.nan,
            "Dmax_q25": round(float(dm.quantile(0.25)),1) if len(dm) else np.nan,
            "Dmax_q75": round(float(dm.quantile(0.75)),1) if len(dm) else np.nan,
            "n_inequality_ops": int(n_ops),
            "n_assay_times": n_assay_time,
            "assay_time_unique": int(times.nunique()) if len(times) else 0,
            "n_cell_lines": int(cell),
        }

    rows = [
        summarize("TACK (full)", tack),
        summarize("TACK/TPDdb", tpddb),
        summarize("TACK/PROTAC-DB 3.0 subset", pdb_tack),
        summarize("PROTACpedia", pedia),
        summarize("PROTAC-DB (training, local)", pdb_local, is_tack=False),
    ]
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{RES}/9_activity_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nSaved: results/9_activity_summary.csv")

    # ═══════ FIG A: DC50 distributions (log10) ═══════
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    sets = [
        ("TACK (full)", tack[tack['Value_Type']=='DC50']['Value']),
        ("TPDdb (TACK core)", tpddb[tpddb['Value_Type']=='DC50']['Value']),
        ("TACK/PROTAC-DB 3.0", pdb_tack[pdb_tack['Value_Type']=='DC50']['Value']),
        ("PROTACpedia", pedia[pedia['Value_Type']=='DC50']['Value']),
        ("PROTAC-DB (training)", pdb_local['DC50 (nM)'].apply(parse)),
    ]
    for ax, (name, series) in zip(axes.flat, sets):
        v = pd.to_numeric(series, errors='coerce').dropna()
        v = v[v > 0]
        if len(v) == 0: ax.axis('off'); continue
        ax.hist(np.log10(v), bins=30, color='#1f77b4', alpha=0.85, edgecolor='white')
        ax.axvline(np.log10(v.median()), color='red', ls='--', lw=1.2, label=f'median={v.median():.1f} nM')
        ax.set_xlabel('log10(DC50 / nM)'); ax.set_ylabel('Count')
        ax.set_title(f'{name} (n={len(v)})')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.suptitle('DC50 distributions across activity databases', fontsize=14)
    fig.tight_layout()
    save(fig, "figA_dc50_distributions")

    # ═══════ FIG B: Dmax distributions ═══════
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    sets = [
        ("TACK (full)", tack[tack['Value_Type']=='Dmax']['Value']),
        ("TPDdb (TACK core)", tpddb[tpddb['Value_Type']=='Dmax']['Value']),
        ("TACK/PROTAC-DB 3.0", pdb_tack[pdb_tack['Value_Type']=='Dmax']['Value']),
        ("PROTACpedia", pedia[pedia['Value_Type']=='Dmax']['Value']),
        ("PROTAC-DB (training)", pdb_local['Dmax (%)'].apply(parse)),
    ]
    for ax, (name, series) in zip(axes.flat, sets):
        v = pd.to_numeric(series, errors='coerce').dropna()
        if len(v) == 0: ax.axis('off'); continue
        ax.hist(v, bins=30, color='#ff7f0e', alpha=0.85, edgecolor='white')
        ax.axvline(v.median(), color='red', ls='--', lw=1.2, label=f'median={v.median():.1f}%')
        ax.set_xlabel('Dmax (%)'); ax.set_ylabel('Count')
        ax.set_title(f'{name} (n={len(v)})')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.suptitle('Dmax distributions across activity databases', fontsize=14)
    fig.tight_layout()
    save(fig, "figB_dmax_distributions")

    # ═══════ FIG C: DC50 median comparison bar ═══════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    dbs = summ['database'].tolist()
    axes[0].bar(dbs, summ['DC50_median_nM'], color='#1f77b4', alpha=0.85)
    for i, v in enumerate(summ['DC50_median_nM']):
        if pd.notna(v): axes[0].text(i, v+5, f'{v:.0f}', ha='center', fontsize=9)
    axes[0].set_xticklabels(dbs, rotation=20, ha='right')
    axes[0].set_ylabel('Median DC50 (nM)')
    axes[0].set_title('Median DC50 by database (log scale note: y is linear)')
    axes[1].bar(dbs, summ['Dmax_median_pct'], color='#ff7f0e', alpha=0.85)
    for i, v in enumerate(summ['Dmax_median_pct']):
        if pd.notna(v): axes[1].text(i, v+1, f'{v:.0f}', ha='center', fontsize=9)
    axes[1].set_xticklabels(dbs, rotation=20, ha='right')
    axes[1].set_ylabel('Median Dmax (%)')
    axes[1].set_title('Median Dmax by database')
    fig.tight_layout()
    save(fig, "figC_activity_medians")

    # ═══════ FIG D: Activity endpoint coverage ═══════
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dbs)); w = 0.35
    ax.bar(x - w/2, summ['n_DC50'], w, label='DC50 endpoints', color='#1f77b4', alpha=0.9)
    ax.bar(x + w/2, summ['n_Dmax'], w, label='Dmax endpoints', color='#ff7f0e', alpha=0.9)
    for i, (a, b) in enumerate(zip(summ['n_DC50'], summ['n_Dmax'])):
        ax.text(i-w/2, a+40, f'{a}', ha='center', fontsize=9)
        ax.text(i+w/2, b+40, f'{b}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(dbs, rotation=20, ha='right')
    ax.set_ylabel('Endpoints')
    ax.set_title('Activity endpoint coverage per database')
    ax.legend()
    fig.tight_layout()
    save(fig, "figD_activity_coverage")

    # ═══════ FIG E: Heterogeneity — operators + assay times ═══════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(dbs, summ['n_inequality_ops'], color='#2ca02c', alpha=0.85)
    for i, v in enumerate(summ['n_inequality_ops']):
        axes[0].text(i, v+8, f'{v}', ha='center', fontsize=9)
    axes[0].set_xticklabels(dbs, rotation=20, ha='right')
    axes[0].set_ylabel('Records with >/< operators')
    axes[0].set_title('Value reporting heterogeneity (inequality ops)')
    axes[1].bar(dbs, summ['n_assay_times'], color='#9467bd', alpha=0.85)
    for i, v in enumerate(summ['n_assay_times']):
        axes[1].text(i, v+8, f'{v}', ha='center', fontsize=9)
    axes[1].set_xticklabels(dbs, rotation=20, ha='right')
    axes[1].set_ylabel('Records with assay-time')
    axes[1].set_title('Assay duration annotation')
    fig.tight_layout()
    save(fig, "figE_heterogeneity_ops_times")

    print(f"\nActivity figures saved to {FIG}")

if __name__ == '__main__':
    main()
