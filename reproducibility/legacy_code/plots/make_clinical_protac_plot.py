#!/usr/bin/env python3
"""
Clinical PROTAC — SynGlue predicted DC50 vs reported (literature) DC50.
Predicted from frozen final ProtacAttentionModel; reported from literature/clinical data.

Reported DC50 (nM) sources:
  ARV-110  : 1.0   (VCaP, AACR 2021; <1 nM across cell lines)
  ARV-471  : 1.0   (ER+ breast cancer cell lines, ~1 nM DC50; PMC11325148)
  NX-2127  : 4.0   (TMD8, DC50=4 nM; <5 nM across cell lines)
  KT-474   : 2.1   (human PBMCs; 0.49-3.4 nM across cell types)
  CFT8634  : 2.7   (BRD9-HiBiT DC50=2.7-3 nM; C4 Therapeutics)
  ZLC491   : 28.0  (CDK13 MDA-MB-231; DC50 28-32 nM, J Med Chem 2024)

Predicted (SynGlue frozen model, whole-PROTAC mode):
  ARV-110  : 6.1
  ARV-471  : 15.1
  NX-2127  : 28.0
  KT-474   : 13.1
  CFT8634  : 11.2
  ZLC491   : 0.9

Output: figures/final_plot/clinical_protac_comparison
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
FIG = f"{REV}/figures/final_plot"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    data = [
        # name, target, predicted_nM, reported_nM, reported_note
        ("ARV-110", "AR",            6.1, 1.0,  "1 nM (VCaP)"),
        ("ARV-471", "ERα",           15.1, 1.0,  "~1 nM (ER+)"),
        ("NX-2127", "BTK",           28.0, 4.0,  "4 nM (TMD8)"),
        ("KT-474",  "IRAK4",         13.1, 2.1,  "2.1 nM (PBMC)"),
        ("CFT8634", "BRD9/SMARCA",   11.2, 2.7,  "2.7 nM (BRD9)"),
        ("ZLC491",  "CDK12/13",       0.9, 28.0, "28 nM (MDA-MB-231)"),
    ]
    names = [d[0] for d in data]
    targets = [d[1] for d in data]
    pred = np.array([d[2] for d in data])
    rep = np.array([d[3] for d in data])
    notes = [d[4] for d in data]

    # ── Figure 1: predicted vs reported log-log scatter ──
    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = ['#2c7fb8','#d95f0e','#41ab5d','#756bb1','#d62728','#8c564b']
    for i, (n, t, p, r) in enumerate(zip(names, targets, pred, rep)):
        ax.scatter(r, p, s=160, color=colors[i], edgecolor='black', linewidth=1.2, zorder=3)
        ax.annotate(f"{n}\n({t})", (r, p), textcoords='offset points', xytext=(12, 8),
                    fontsize=10.5, weight='bold', color='#222')
    # identity + 3x/10x bands
    lims = [0.3, 100]
    ax.plot(lims, lims, 'k--', lw=1.5, label='y = x (perfect)')
    ax.plot(lims, [3*x for x in lims], 'r:', lw=1.2, alpha=0.7, label='±3×')
    ax.plot(lims, [x/3 for x in lims], 'r:', lw=1.2, alpha=0.7)
    ax.plot(lims, [10*x for x in lims], 'grey', ls=':', lw=1, alpha=0.6, label='±10×')
    ax.plot(lims, [x/10 for x in lims], 'grey', ls=':', lw=1, alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Reported DC50 (nM) — literature/clinical')
    ax.set_ylabel('SynGlue predicted DC50 (nM)')
    ax.set_title('Clinical PROTACs — SynGlue predicted vs reported DC50')
    ax.set_xlim(0.3, 100); ax.set_ylim(0.3, 100)
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    save(fig, "clinical_protac_pred_vs_reported")

    # ── Figure 2: grouped bar (predicted vs reported per compound) ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    w = 0.36
    b1 = ax.bar(x - w/2, pred, w, label='SynGlue predicted', color='#2c7fb8', alpha=0.9,
                edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, rep, w, label='Reported (literature)', color='#f2c9a0', alpha=0.95,
                edgecolor='#d19a55', linewidth=0.8)
    for i, (p, r) in enumerate(zip(pred, rep)):
        ax.text(i - w/2, p*1.12, f'{p:.1f}', ha='center', fontsize=9.5, weight='bold', color='#1a4a73')
        ax.text(i + w/2, r*1.12, f'{r:.1f}', ha='center', fontsize=9.5, weight='bold', color='#8a3b00')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({t})" for n, t in zip(names, targets)], fontsize=10)
    ax.set_ylabel('DC50 (nM, log scale)')
    ax.set_title('Clinical PROTACs — predicted vs reported DC50 (single-digit to low-tens nM)')
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "clinical_protac_bar_predicted_reported")

    # ── Figure 3: fold-error (predicted/reported ratio) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ratio = pred / rep
    colors_ratio = ['#41ab5d' if 0.3 <= r <= 3 else ('#d95f0e' if 0.1 <= r <= 10 else '#d62728') for r in ratio]
    bars = ax.bar([f"{n}\n({t})" for n, t in zip(names, targets)], ratio,
                  color=colors_ratio, alpha=0.9, edgecolor='black', linewidth=0.6)
    ax.axhline(1.0, color='black', lw=1.2, label='perfect (ratio=1)')
    ax.axhspan(0.3, 3.0, color='green', alpha=0.08, label='within 3×')
    ax.axhspan(0.1, 10.0, color='orange', alpha=0.05, label='within 10×')
    for b, r in zip(bars, ratio):
        ax.text(b.get_x()+b.get_width()/2, r + 0.08, f'{r:.1f}×', ha='center', fontsize=10, weight='bold')
    ax.set_yscale('log')
    ax.set_ylabel('Predicted / Reported DC50 ratio (log scale)')
    ax.set_title('Clinical PROTACs — prediction fold-error (frozen SynGlue model)')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    save(fig, "clinical_protac_fold_error")

    # ── summary table ──
    import pandas as pd
    df = pd.DataFrame({"Compound": names, "Target": targets, "Predicted_DC50_nM": pred,
                       "Reported_DC50_nM": rep, "Fold_ratio": ratio, "Note": notes})
    df.to_csv(f"{FIG}/clinical_protac_comparison.csv", index=False)
    print("\nSaved: clinical_protac_comparison.csv")
    print(df.to_string(index=False))
    # stats
    within3 = np.mean((ratio >= 0.3) & (ratio <= 3.0)) * 100
    within10 = np.mean((ratio >= 0.1) & (ratio <= 10.0)) * 100
    geo_mean = np.exp(np.mean(np.log(ratio)))
    print(f"\n  within 3×: {within3:.0f}% | within 10×: {within10:.0f}% | geometric mean ratio: {geo_mean:.1f}")

if __name__ == '__main__':
    main()
