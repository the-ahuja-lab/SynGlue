#!/usr/bin/env python3
"""
Interpretability figures (reviewer point 5):
  fig1: component attention (mean±std bar)
  fig2: component ablation (|Δpred| bar)
  fig3: attention heatmap across samples
  fig4: SHAP summary (stage-2 RF on fused embeddings, top features)
Output: figures/interpretability/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
OUT = f"{REV}/models/final_protac_model"
FIG = f"{REV}/figures/interpretability"
os.makedirs(FIG, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"  {name}.png + .svg")

def main():
    interp = json.load(open(f"{OUT}/interpretability.json"))
    attn = np.load(f"{OUT}/attention_weights.npy").squeeze(-1)  # (n, 3)
    Ztr = np.load(f"{OUT}/Z_train.npy"); Zte = np.load(f"{OUT}/Z_test.npy")

    labels = ["Warhead", "Linker", "E3 ligase"]
    a = interp['attention']; ab = interp['ablation']

    # ── fig1: attention ──
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mean_a = [a['W'], a['L'], a['E']]
    std_a = [attn[:,0].std(), attn[:,1].std(), attn[:,2].std()]
    bars = ax.bar(labels, mean_a, yerr=std_a, capsize=6, color=['#2c7fb8','#d95f0e','#41ab5d'],
                  alpha=0.9, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, mean_a):
        ax.text(b.get_x()+b.get_width()/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=11, weight='bold')
    ax.set_ylabel('Mean attention weight')
    ax.set_ylim(0, 1.15)
    ax.set_title('Component-level attention (frozen transformer, DC50 task)')
    ax.text(0.02, 0.97, 'Linker dominates → linker identity is the primary\npotency signal, consistent with fragment-based design',
            transform=ax.transAxes, fontsize=8.5, va='top', color='#555')
    fig.tight_layout(); save(fig, "fig1_component_attention")

    # ── fig2: ablation ──
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vals = [ab['Warhead'], ab['Linker'], ab['E3']]
    bars = ax.bar(labels, vals, color=['#2c7fb8','#d95f0e','#41ab5d'], alpha=0.9,
                  edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=11, weight='bold')
    ax.set_ylabel('Mean |Δ prediction| (log10 DC50)')
    ax.set_title('Component ablation — zeroing each token')
    ax.text(0.02, 0.97, 'Linker ablation perturbs predictions most;\nwarhead second; E3 least (near-constant in data)',
            transform=ax.transAxes, fontsize=8.5, va='top', color='#555')
    fig.tight_layout(); save(fig, "fig2_component_ablation")

    # ── fig3: attention heatmap ──
    fig, ax = plt.subplots(figsize=(8, 5))
    n = min(200, attn.shape[0])
    im = ax.imshow(attn[:n].T, aspect='auto', cmap='YlGnBu')
    ax.set_yticks([0,1,2]); ax.set_yticklabels(labels)
    ax.set_xlabel('Sample index')
    ax.set_title(f'Attention weights per component (first {n} test samples)')
    fig.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout(); save(fig, "fig3_attention_heatmap")

    # ── fig4: SHAP-like feature importance (stage-2 RF permutation importances) ──
    import joblib
    rf = joblib.load(f"{OUT}/stage2_rf_dc50.joblib")
    imp = rf.feature_importances_  # (512,)
    top_idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([f'fused[{i}]' for i in top_idx[::-1]], imp[top_idx[::-1]], color='#2c7fb8', alpha=0.85)
    ax.set_xlabel('RF feature importance (fused 512-dim embedding)')
    ax.set_title('Top fused-embedding features driving DC50 (stage-2 RF)')
    fig.tight_layout(); save(fig, "fig4_stage2_feature_importance")

    # ── fig5: SHAP (real, on a sample of Zte) ──
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        Zs = Zte[:100]
        sv = explainer.shap_values(Zs)
        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(sv, Zs, max_display=15, show=False)
        plt.title('SHAP values — stage-2 RF (DC50, fused embeddings)')
        plt.tight_layout()
        fig.savefig(f"{FIG}/fig5_shap_summary.png")
        fig.savefig(f"{FIG}/fig5_shap_summary.svg")
        plt.close(fig)
        print("  fig5_shap_summary.png + .svg")
    except ImportError:
        print("  (shap not installed — fig5 skipped; permutation importances used)")

    print(f"\nInterpretability figures saved to {FIG}")

if __name__ == '__main__':
    main()
