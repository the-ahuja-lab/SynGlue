#!/usr/bin/env python3
"""
Descriptive plots for cross-database comparison (PROTAC-DB vs TACK vs PROTACpedia vs Patent)
Output: protac_db_justification/figures/ (PNG + SVG)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/protac_db_justification"
RES = f"{OUT}/results"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

DB_ORDER = ["PROTAC-DB", "TACK (incl. PROTAC-DB 3.0 subset)", "PROTACpedia", "TACK (full)", "PROTAC-Patent"]
DB_SHORT = {"PROTAC-DB": "PROTAC-DB", "TACK (incl. PROTAC-DB 3.0 subset)": "TACK\n(PROTAC-DB 3.0 subset)",
            "PROTACpedia": "PROTACpedia", "TACK (full)": "TACK (full)", "PROTAC-Patent": "PROTAC-Patent"}
COLORS = {"PROTAC-DB": "#2ca02c", "TACK (incl. PROTAC-DB 3.0 subset)": "#1f77b4",
          "PROTACpedia": "#ff7f0e", "TACK (full)": "#9467bd", "PROTAC-Patent": "#d62728"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    counts = pd.read_csv(f"{RES}/1_counts.csv")
    overlap = pd.read_csv(f"{RES}/2_overlap_matrix.csv", index_col=0)
    targets = pd.read_csv(f"{RES}/3_targets.csv")
    e3c = pd.read_csv(f"{RES}/4_e3.csv")
    cells = pd.read_csv(f"{RES}/5_celllines.csv")
    assays = pd.read_csv(f"{RES}/6_assays.csv")
    linkers = pd.read_csv(f"{RES}/7_linker.csv")
    phys = pd.read_csv(f"{RES}/8_physchem.csv")

    # ═══════════ FIG 1: Unique PROTAC counts ═══════════
    fig, ax = plt.subplots(figsize=(10, 5))
    dbs = [c for c in counts['database'] if c in DB_ORDER]
    x = np.arange(len(dbs))
    bars1 = ax.bar(x - 0.18, [counts.loc[counts['database']==d,'records'].iloc[0] for d in dbs], 0.36,
                   label='Records', color='#7f7f7f', alpha=0.8)
    bars2 = ax.bar(x + 0.18, [counts.loc[counts['database']==d,'unique_protacs'].iloc[0] for d in dbs], 0.36,
                   label='Unique PROTACs', color=[COLORS[d] for d in dbs], alpha=0.9)
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+40, f'{int(b.get_height()):,}',
                ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([DB_SHORT[d] for d in dbs])
    ax.set_ylabel('Count')
    ax.set_title('Database size: records vs unique PROTACs (canonical SMILES)')
    ax.legend()
    save(fig, "fig1_counts")

    # ═══════════ FIG 2: Overlap heatmap ═══════════
    fig, ax = plt.subplots(figsize=(9, 7))
    odb = [c for c in overlap.index if c in DB_ORDER]
    ov = overlap.loc[odb, odb].astype(float)
    im = ax.imshow(ov.values, cmap='YlGnBu')
    ax.set_xticks(range(len(odb))); ax.set_xticklabels([DB_SHORT[d] for d in odb], rotation=30, ha='right')
    ax.set_yticks(range(len(odb))); ax.set_yticklabels([DB_SHORT[d] for d in odb])
    for i in range(len(odb)):
        for j in range(len(odb)):
            ax.text(j, i, f'{int(ov.values[i,j]):,}', ha='center', va='center', fontsize=10,
                    color='black' if ov.values[i,j] < np.nanmax(ov.values)/2 else 'white')
    ax.set_title('Unique-SMILES overlap between databases')
    fig.colorbar(im, ax=ax, label='Shared unique PROTACs')
    save(fig, "fig2_overlap_heatmap")

    # ═══════════ FIG 3: Target / E3 / cell-line / assay diversity ═══════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [("unique_targets", 'Targets (POI)', targets),
               ("unique_e3", 'E3 ligases', e3c),
               ("unique_cell_lines", 'Cell lines', cells),
               ("unique_assays", 'Assay types', assays)]
    for ax, (col, title, df) in zip(axes.flat, metrics):
        dbs_here = [c for c in df['database'] if c in DB_ORDER]
        vals = [df.loc[df['database']==d, col].iloc[0] for d in dbs_here]
        bars = ax.bar(range(len(dbs_here)), vals, color=[COLORS[d] for d in dbs_here], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(dbs_here))); ax.set_xticklabels([DB_SHORT[d] for d in dbs_here], rotation=25, ha='right')
        ax.set_ylabel('Unique entries')
        ax.set_title(f'Heterogeneity — {title}')
    fig.suptitle('Coverage across comparison levels', fontsize=14)
    fig.tight_layout()
    save(fig, "fig3_coverage_levels")

    # ═══════════ FIG 4: Linker/component annotation ═══════════
    fig, ax = plt.subplots(figsize=(10, 5))
    dbs_here = [c for c in linkers['database'] if c in DB_ORDER]
    vals = [linkers.loc[linkers['database']==d,'unique_linkers'].iloc[0] for d in dbs_here]
    labels = [linkers.loc[linkers['database']==d,'component_info'].iloc[0] for d in dbs_here]
    bars = ax.bar(range(len(dbs_here)), vals, color=[COLORS[d] for d in dbs_here], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+15, f'{v:,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(dbs_here))); ax.set_xticklabels([DB_SHORT[d] for d in dbs_here], rotation=20, ha='right')
    ax.set_ylabel('Unique linkers / scaffolds')
    ax.set_title('Linker-level coverage: PROTAC-DB has explicit linker SMILES (2,471);\nothers require computational decomposition (scaffold proxy shown)')
    for i, lab in enumerate(labels):
        if 'explicit' in lab:
            ax.text(i, 0, '✓ explicit warhead/linker/E3\nready for component-aware\nmodels', ha='center', va='bottom',
                    fontsize=9, color='#2ca02c', weight='bold')
    save(fig, "fig4_linker_annotation")

    # ═══════════ FIG 5: Physchem distributions (box) ═══════════
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    pc_cols = ['MW_median', 'logP_median', 'TPSA_median', 'RotB_median', 'HBA_median', 'HBD_median']
    pc_titles = ['MW', 'logP', 'TPSA (Å²)', 'Rotatable bonds', 'HBA', 'HBD']
    dbs_here = [c for c in phys['database'] if c in DB_ORDER]
    for ax, col, t in zip(axes.flat, pc_cols, pc_titles):
        vals = [phys.loc[phys['database']==d, col].iloc[0] for d in dbs_here]
        bars = ax.bar(range(len(dbs_here)), vals, color=[COLORS[d] for d in dbs_here], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5)
        ax.set_xticks(range(len(dbs_here))); ax.set_xticklabels([DB_SHORT[d] for d in dbs_here], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(t); ax.set_title(t)
    fig.suptitle('Physicochemical properties (median, sampled up to 2,000 compounds/database)', fontsize=14)
    fig.tight_layout()
    save(fig, "fig5_physchem")

    # ═══════════ FIG 6: Target diversity (top targets per DB) ═══════════
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    # reload raw data for target distributions
    tack = pd.read_parquet(f"{BASE}/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet")
    pdb = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/PROTAC_DB_with_features.csv", low_memory=False)
    pat = pd.read_csv(f"{REV}/protac_db_justification/data/PROTAC-PatentDB/PROTAC_Patent_Compounds_full.csv", low_memory=False)
    tack_db = tack[tack['Database'].str.contains('PROTAC-DB', na=False)]
    tack_pedia = tack[tack['Database'].str.contains('PROTACpedia', na=False)]
    top_data = [("PROTAC-DB", pdb['Target'].dropna().astype(str).replace('nan', '')),
                ("TACK (PROTAC-DB 3.0)", tack_db['POI_Name'].dropna().astype(str).replace('nan', '')),
                ("PROTACpedia", tack_pedia['POI_Name'].dropna().astype(str).replace('nan', ''))]
    for ax, (name, series) in zip(axes.flat[:3], top_data):
        series = series[series != '']
        vc = series.value_counts().head(10)
        ax.barh(vc.index[::-1], vc.values[::-1], color=COLORS[name] if name in COLORS else '#333', alpha=0.85)
        ax.set_xlabel('Count')
        ax.set_title(f'{name} — top targets')
    # patent targets
    vc = pat['Target'].dropna().astype(str).replace('nan','').value_counts().head(10)
    vc = vc[vc.index != '']
    ax = axes.flat[3]
    ax.barh(vc.index[::-1], vc.values[::-1], color=COLORS['PROTAC-Patent'], alpha=0.85)
    ax.set_xlabel('Count'); ax.set_title('PROTAC-Patent — top targets')
    axes.flat[4].axis('off'); axes.flat[5].axis('off')
    fig.suptitle('Target landscape per database', fontsize=14)
    fig.tight_layout()
    save(fig, "fig6_target_landscape")

    # ═══════════ FIG 7: E3 distribution ═══════════
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    e3_pdb = pdb['E3 ligase'].dropna().astype(str).replace('nan','')
    e3_pdb = e3_pdb[e3_pdb != '']
    vc1 = e3_pdb.value_counts().head(8)
    axes[0].bar(vc1.index, vc1.values, color='#2ca02c', alpha=0.85)
    axes[0].set_title('PROTAC-DB — E3 ligase distribution')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Count')
    vc2 = tack['Ligase_Name'].dropna().astype(str).replace('nan','')
    vc2 = vc2[vc2 != ''].value_counts().head(8)
    axes[1].bar(vc2.index, vc2.values, color='#9467bd', alpha=0.85)
    axes[1].set_title('TACK — E3 ligase distribution')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_ylabel('Count')
    fig.tight_layout()
    save(fig, "fig7_e3_distribution")

    # ═══════════ FIG 8: Assay heterogeneity (PROTAC-DB) ═══════════
    upd = pd.read_csv("/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data/Upadted_PROTAC_features.csv",
                      low_memory=False, usecols=['Assay (DC50/Dmax)'])
    a = upd['Assay (DC50/Dmax)'].dropna().astype(str)
    a = a[a != 'nan']
    vc = a.value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vc.index[::-1], vc.values[::-1], color='#2ca02c', alpha=0.85)
    ax.set_xlabel('Count')
    ax.set_title(f'PROTAC-DB — assay heterogeneity ({a.nunique()} unique assay descriptions)')
    fig.tight_layout()
    save(fig, "fig8_assay_heterogeneity")

    # ═══════════ FIG 9: Cell line heterogeneity (TACK) ═══════════
    cl = tack['Cell_Line'].dropna().astype(str)
    cl = cl[cl != 'nan']
    vc = cl.value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(vc.index[::-1], vc.values[::-1], color='#9467bd', alpha=0.85)
    ax.set_xlabel('Count')
    ax.set_title(f'TACK — cell line heterogeneity ({cl.nunique()} unique cell lines)')
    fig.tight_layout()
    save(fig, "fig9_cellline_heterogeneity")

    print(f"\nAll figures saved to {FIG}")

if __name__ == '__main__':
    main()
