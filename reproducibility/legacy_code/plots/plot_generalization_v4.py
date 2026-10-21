#!/usr/bin/env python3
"""
v4 Generalization Plots — PNG + SVG
Covers all 4 improvements + v3 comparison.
Output: revision/figures/generalization_v4/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
V4 = f"{REV}/experiments/gen_training_v4"
V3 = f"{REV}/experiments/gen_training_v3"
FIG = f"{REV}/figures/generalization_v4"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 11.5,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

SPLITS = ["random", "target_held_out", "e3_held_out", "scaffold_held_out"]
SL = {"random": "Random", "target_held_out": "Target-held", "e3_held_out": "E3-held", "scaffold_held_out": "Scaffold-held"}
COL = {"dc50": "#2ca02c", "dmax": "#ff7f0e", "dmax90": "#9467bd"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    summ = pd.read_csv(f"{V4}/generalization_summary_v4.csv")
    ext = pd.read_csv(f"{V4}/external_validation_v4.csv")
    v3_summ = pd.read_csv(f"{V3}/generalization_summary_v2.csv")
    v3_ext = pd.read_csv(f"{V3}/external_validation_stats.csv")

    # ═══════════ FIG 1: Internal R2 per task/split (v4) ═══════════
    fig, ax = plt.subplots(figsize=(11, 5))
    tasks = ["dc50", "dmax", "dmax90"]
    x = np.arange(len(SPLITS)); w = 0.26
    for j, t in enumerate(tasks):
        vals = [summ.loc[(summ['task']==t)&(summ['split']==s),'R2'].iloc[0] for s in SPLITS]
        ax.bar(x + (j-1)*w, vals, w, label=f"{t.upper()} ({summ[summ['task']==t]['n_train'].iloc[0]} train)",
               color=COL[t], alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SL[s] for s in SPLITS])
    ax.set_ylabel('Internal held-out R²')
    ax.set_title('v4 Component-Aware Transformer — internal generalization (balanced sampling + extra data)')
    ax.legend()
    save(fig, "fig1_internal_R2_v4")

    # ═══════════ FIG 2: External correlation per task/model (v4) ═══════════
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for j, t in enumerate(tasks):
        ax = axes[j]
        sub = ext[ext['task'] == t]
        models = ["random", "target_held_out", "e3_held_out", "scaffold_held_out", "ensemble"]
        ml = [SL[m] if m != "ensemble" else "Ensemble" for m in models]
        vals = [sub.loc[sub['model']==m,'r'].iloc[0] if (sub['model']==m).any() else np.nan for m in models]
        ps = [sub.loc[sub['model']==m,'p_r'].iloc[0] if (sub['model']==m).any() else 1 for m in models]
        colors = ['#1f77b4']*4 + ['#d62728']
        bars = ax.bar(ml, vals, color=colors, alpha=0.85)
        for b, v, p in zip(bars, vals, ps):
            sig = '***' if p < 1e-4 else ('**' if p < 1e-3 else ('*' if p < 0.05 else 'n.s.'))
            ax.text(b.get_x()+b.get_width()/2, v+(0.01 if v>=0 else -0.02), sig,
                    ha='center', va='bottom' if v>=0 else 'top', fontsize=9)
        ax.axhline(0, color='black', lw=0.8)
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylabel('External Pearson r')
        ax.set_title(f'{t.upper()} — external validation')
    fig.suptitle('v4 external correlation (true decomposition encoding, 2,640 compounds)', fontsize=13)
    fig.tight_layout()
    save(fig, "fig2_external_r_v4")

    # ═══════════ FIG 3: v3 vs v4 external DC50 comparison ═══════════
    fig, ax = plt.subplots(figsize=(9, 5))
    v3_r = [v3_ext.loc[(v3_ext['task']=='dc50')&(v3_ext['split']==s),'pearson_r'].iloc[0] if ((v3_ext['task']=='dc50')&(v3_ext['split']==s)).any() else np.nan for s in SPLITS]
    v3_r += [v3_ext.loc[(v3_ext['task']=='dc50')&(v3_ext['split']=='random'),'pearson_r'].iloc[0]]  # placeholder
    v3_models = [SL[s] for s in SPLITS]
    v4_r = [ext.loc[(ext['task']=='dc50')&(ext['model']==s),'r'].iloc[0] for s in SPLITS]
    v4_r += [ext.loc[(ext['task']=='dc50')&(ext['model']=='ensemble'),'r'].iloc[0]]
    x = np.arange(5)
    ax.bar(x-0.2, v3_r, 0.4, label='v3 (full-PROTAC L)', color='#ff9896', alpha=0.9)
    ax.bar(x+0.2, v4_r, 0.4, label='v4 (true-decomp L)', color='#2ca02c', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SL[s] for s in SPLITS] + ['Ensemble'])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('External Pearson r (DC50)')
    ax.set_title('DC50 external correlation: v3 vs v4 encoding')
    ax.legend()
    save(fig, "fig3_v3_vs_v4_dc50")

    # ═══════════ FIG 4: Diversity improvement (target-held-out) ═══════════
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ['target_held_out', 'e3_held_out']
    gl = ['Target-held-out', 'E3-held-out']
    v3_r2 = [v3_summ.loc[(v3_summ['task']=='dc50')&(v3_summ['split']==g),'R2'].iloc[0] for g in groups]
    v4_r2 = [summ.loc[(summ['task']=='dc50')&(summ['split']==g),'R2'].iloc[0] for g in groups]
    x = np.arange(2); w = 0.3
    ax.bar(x-w/2, v3_r2, w, label='v3 (no balancing)', color='#ff9896')
    ax.bar(x+w/2, v4_r2, w, label='v4 (balanced + extra data)', color='#2ca02c')
    for i, (a, b) in enumerate(zip(v3_r2, v4_r2)):
        ax.text(i-w/2, a+(0.02 if a>=0 else -0.05), f'{a:.2f}', ha='center', fontsize=9)
        ax.text(i+w/2, b+(0.02 if b>=0 else -0.05), f'{b:.2f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(gl)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Internal held-out R² (DC50)')
    ax.set_title('Improvement #4: training diversity + balanced sampling')
    ax.legend()
    save(fig, "fig4_diversity_improvement")

    # ═══════════ FIG 5: Dmax saturation handling ═══════════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    y_dmax = np.load(f"{REV}/benchmarking/generalization_data_v4/y_dmax.npy")
    y_dmax90 = np.load(f"{REV}/benchmarking/generalization_data_v4/y_dmax90.npy")
    axes[0].hist(y_dmax, bins=25, color='#ff7f0e', alpha=0.8, edgecolor='white')
    axes[0].axvline(90, color='red', ls='--', lw=1.2, label='90% cutoff')
    axes[0].set_xlabel('Dmax (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Dmax full (n={len(y_dmax)}) — saturated >90%')
    axes[0].legend()
    axes[1].hist(y_dmax90, bins=25, color='#9467bd', alpha=0.8, edgecolor='white')
    axes[1].set_xlabel('Dmax (%)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Dmax ≤90% filtered (n={len(y_dmax90)})')
    fig.suptitle('Improvement #1: drop saturated Dmax values', fontsize=13)
    fig.tight_layout()
    save(fig, "fig5_dmax_saturation")

    # ═══════════ FIG 6: Ensemble summary ═══════════
    fig, ax = plt.subplots(figsize=(8, 5))
    t_labels = ['DC50', 'Dmax']
    ens_r = [ext.loc[(ext['task']=='dc50')&(ext['model']=='ensemble'),'r'].iloc[0],
             ext.loc[(ext['task']=='dmax')&(ext['model']=='ensemble'),'r'].iloc[0]]
    ens_p = [ext.loc[(ext['task']=='dc50')&(ext['model']=='ensemble'),'p_r'].iloc[0],
             ext.loc[(ext['task']=='dmax')&(ext['model']=='ensemble'),'p_r'].iloc[0]]
    bars = ax.bar(t_labels, ens_r, color=['#2ca02c', '#ff7f0e'], alpha=0.9)
    for b, v, p in zip(bars, ens_r, ens_p):
        sig = '***' if p < 1e-4 else ('**' if p < 1e-3 else ('*' if p < 0.05 else 'n.s.'))
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f} {sig} (p={p:.1e})', ha='center', fontsize=10)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('External Pearson r (ensemble of 4 split models)')
    ax.set_title('Improvement #2: ensemble external validation')
    save(fig, "fig6_ensemble")

    # ═══════════ FIG 7: Internal R2 all tasks/splits heatmap ═══════════
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = summ.pivot_table(index='task', columns='split', values='R2').reindex(columns=SPLITS)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(SPLITS))); ax.set_xticklabels([SL[s] for s in SPLITS])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([t.upper() for t in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center', fontsize=10,
                    color='black' if abs(pivot.values[i,j])<0.6 else 'white')
    ax.set_title('Internal held-out R² heatmap (v4)')
    fig.colorbar(im, ax=ax, label='R²')
    save(fig, "fig7_heatmap_R2")

    # ═══════════ FIG 8: Attention comparison v3 vs v4 (dc50 random) ═══════════
    fig, ax = plt.subplots(figsize=(8, 4.5))
    a_v3 = [v3_summ.loc[(v3_summ['task']=='dc50')&(v3_summ['split']=='random'),c].iloc[0] for c in ['attn_W','attn_L','attn_E']]
    a_v4 = [summ.loc[(summ['task']=='dc50')&(summ['split']=='random'),c].iloc[0] for c in ['attn_W','attn_L','attn_E']]
    x = np.arange(3); w = 0.3
    ax.bar(x-w/2, a_v3, w, label='v3 (full-PROTAC L)', color='#1f77b4', alpha=0.85)
    ax.bar(x+w/2, a_v4, w, label='v4 (true-decomp L)', color='#ff7f0e', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Warhead', 'Linker', 'E3'])
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Attention shift: true decomposition spreads attention across components')
    ax.legend()
    save(fig, "fig8_attention_v3_vs_v4")

    # ═══════════ SUMMARY CSV ═══════════
    # Merge v4 summary + external into one readable summary
    merged = []
    for t in tasks:
        for m in ["random", "target_held_out", "e3_held_out", "scaffold_held_out", "ensemble"]:
            ext_row = ext[(ext['task']==t)&(ext['model']==m)]
            if len(ext_row) == 0: continue
            e = ext_row.iloc[0]
            if m == 'ensemble':
                merged.append({"task": t, "model": "ensemble", "internal_R2": np.nan,
                               "ext_n": e['n'], "ext_r": e['r'], "ext_p": e['p_r'], "ext_rho": e['rho']})
            else:
                s = summ[(summ['task']==t)&(summ['split']==m)]
                merged.append({"task": t, "model": m, "internal_R2": s['R2'].iloc[0] if len(s) else np.nan,
                               "ext_n": e['n'], "ext_r": e['r'], "ext_p": e['p_r'], "ext_rho": e['rho']})
    summary = pd.DataFrame(merged)
    summary.to_csv(f"{FIG}/summary_v4.csv", index=False)
    print(f"\nSaved summary: {FIG}/summary_v4.csv")

    print(f"\nAll v4 figures saved to {FIG}")
    for f in sorted(os.listdir(FIG)):
        print(f"  {f}")

if __name__ == '__main__':
    main()
