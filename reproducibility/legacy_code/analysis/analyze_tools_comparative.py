#!/usr/bin/env python3
"""
COMPREHENSIVE COMPARATIVE ANALYSIS — Molecule-Generation Tools
Gcoupler | Pharmacophore | DiffSBDD | Pocket2Mol | DiffLinker | DeLinker

Sections:
  1. Physicochemical boxplots + jitter (MW, logP, TPSA, HBD, HBA, RotB, HeavyAtoms, Fsp3, Rings, AromRings, QED, SA)
  2. Descriptive statistics table
  3. Kruskal-Wallis + Mann-Whitney pairwise + BH-FDR
  4. Pairwise effect-size heatmap
  5. Morgan fingerprint (r=2, 2048 bits) — common representation
  6. PCA
  7. UMAP
  8. t-SNE
  9. Silhouette (original, PCA, UMAP, t-SNE)
 10. Per-tool silhouette plot
 11. Pairwise chemical-space separation heatmap
 12. Diversity analysis (mean/median Tanimoto, internal diversity)
 13. Final multi-panel figure
 14. Interpretation
"""
import os, sys, json, warnings, itertools
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
ROOT = f"{REV}/benchmarking/gcoupler_vs_tools"
OUT = f"{REV}/figures/final_plot/tool_analysis"
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, DataStructs, RDConfig, QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
    SA_OK = True
except Exception:
    SA_OK = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import umap

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.2, 'grid.linestyle': '--',
                     'svg.fonttype': 'none', 'figure.dpi': 200})

TOOLS = ["Gcoupler", "Pharmacophore", "DiffSBDD", "Pocket2Mol", "DiffLinker", "DeLinker"]
COLORS = {
    "Gcoupler":     "#a8cbe8",  # light blue
    "Pharmacophore": "#f2c9a0",  # light orange
    "DiffSBDD":     "#b9d9b0",  # light green
    "Pocket2Mol":   "#c3b8e0",  # light purple
    "DiffLinker":   "#e0b8c8",  # light rose
    "DeLinker":     "#bcd4d0",  # light teal
}
EDGES = {
    "Gcoupler":     "#5a8db8",
    "Pharmacophore": "#d19a55",
    "DiffSBDD":     "#6fa868",
    "Pocket2Mol":   "#8f7fb8",
    "DiffLinker":   "#b87a94",
    "DeLinker":     "#6f9a94",
}

PROP_NAMES = {
    'MW': 'Molecular Weight (Da)', 'logP': 'cLogP', 'TPSA': 'TPSA (Å²)',
    'HBD': 'H-bond donors', 'HBA': 'H-bond acceptors', 'RotB': 'Rotatable bonds',
    'HeavyAtoms': 'Heavy atom count', 'Fsp3': 'Fraction Csp3', 'Rings': 'Ring count',
    'AromRings': 'Aromatic rings', 'QED': 'QED', 'SA': 'SA score',
}

def load_tool_smiles():
    """Load SMILES for each tool (valid, canonical, deduped within tool)."""
    GCR = f"{REV}/experiments/gcoupler_benchmark/results"
    import glob
    data = {}
    # Gcoupler
    g = pd.read_csv(f"{GCR}/gcoupler/filtered_results.csv")
    data['Gcoupler'] = g['canonical_smiles'].dropna().tolist()
    # Pharmacophore
    ph = pd.read_csv(f"{GCR}/pharmacophore/generated_candidates.csv")
    data['Pharmacophore'] = ph['SMILES'].dropna().tolist()
    # DiffSBDD
    mols = [m for m in Chem.SDMolSupplier(f"{GCR}/diffsbdd/samples.sdf") if m is not None]
    data['DiffSBDD'] = [Chem.MolToSmiles(m) for m in mols]
    # Pocket2Mol
    p2m = pd.read_csv(glob.glob(f"{GCR}/pocket2mol/*/SMILES.txt")[0], header=None)[0].dropna().tolist()
    data['Pocket2Mol'] = p2m
    # DeLinker
    dl = pd.read_csv(f"{GCR}/warheads/summary/warhead_delinker.csv")
    data['DeLinker'] = dl['canonical_smiles'].dropna().tolist()
    # DiffLinker
    dlk = pd.read_csv(f"{GCR}/warheads/summary/warhead_difflinker.csv")
    data['DiffLinker'] = dlk['canonical_smiles'].dropna().tolist()

    # canonicalize + dedupe within tool
    clean = {}
    for t, smis in data.items():
        canons = set()
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                canons.add(Chem.MolToSmiles(m, canonical=True))
        clean[t] = sorted(canons)
    return clean

def compute_props(smiles_list):
    """Compute physicochemical properties for a list of canonical SMILES."""
    rows = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        rows.append({
            'SMILES': s, 'MW': Descriptors.MolWt(m), 'logP': Crippen.MolLogP(m),
            'TPSA': rdMolDescriptors.CalcTPSA(m), 'HBD': rdMolDescriptors.CalcNumHBD(m),
            'HBA': rdMolDescriptors.CalcNumHBA(m), 'RotB': rdMolDescriptors.CalcNumRotatableBonds(m),
            'HeavyAtoms': m.GetNumHeavyAtoms(), 'Fsp3': rdMolDescriptors.CalcFractionCSP3(m),
            'Rings': rdMolDescriptors.CalcNumRings(m),
            'AromRings': rdMolDescriptors.CalcNumAromaticRings(m),
            'QED': Chem.QED.qed(m),
            'SA': sascorer.calculateScore(m) if SA_OK else np.nan,
        })
    return pd.DataFrame(rows)

def fp_matrix(smiles_list):
    """Morgan r=2 2048-bit binary fingerprint matrix."""
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles_list]
    X = np.zeros((len(fps), 2048), dtype=np.int8)
    for i, f in enumerate(fps):
        for b in f.GetOnBits():
            X[i, b] = 1
    return X

def box_jitter(ax, data_list, tools, prop, ylab):
    """Boxplot + jittered points for one property."""
    bp = ax.boxplot(data_list, positions=range(len(tools)), widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.6),
                    whiskerprops=dict(color='#555', lw=1.1), capprops=dict(color='#555', lw=1.1))
    rng = np.random.RandomState(42)
    for i, (patch, v, t) in enumerate(zip(bp['boxes'], data_list, tools)):
        patch.set_facecolor(COLORS[t]); patch.set_edgecolor(EDGES[t]); patch.set_linewidth(1.2)
        jx = rng.normal(0, 0.07, len(v))
        ax.scatter(i + jx, v, s=4, alpha=0.25, color=EDGES[t], edgecolor='none', zorder=3)
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels([f"{t}\n(n={len(v)})" for t, v in zip(tools, data_list)], fontsize=8)
    ax.set_ylabel(ylab)

def main():
    # ═══ 0. LOAD ═══
    print("Loading tool SMILES...", flush=True)
    smiles = load_tool_smiles()
    for t in TOOLS:
        print(f"  {t}: {len(smiles[t])} unique valid", flush=True)

    # ═══ 1. PROPERTIES ═══
    print("\nComputing physicochemical properties...", flush=True)
    props = {}
    for t in TOOLS:
        props[t] = compute_props(smiles[t])
        print(f"  {t}: {len(props[t])}", flush=True)

    prop_keys = ['MW','logP','TPSA','HBD','HBA','RotB','HeavyAtoms','Fsp3','Rings','AromRings','QED','SA']

    # ═══ 2. DESCRIPTIVE STATS ═══
    stat_rows = []
    for t in TOOLS:
        for pk in prop_keys:
            v = props[t][pk].dropna().values
            q1, q2, q3 = np.percentile(v, [25,50,75])
            stat_rows.append({"Property": PROP_NAMES[pk], "Tool": t, "N": len(v),
                              "Mean": round(v.mean(),3), "Median": round(q2,3),
                              "SD": round(v.std(ddof=1),3), "Variance": round(v.var(ddof=1),3),
                              "Q1": round(q1,3), "Q3": round(q3,3), "IQR": round(q3-q1,3),
                              "Min": round(v.min(),3), "Max": round(v.max(),3)})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(f"{OUT}/descriptive_stats.csv", index=False)
    print("\nSaved descriptive_stats.csv", flush=True)

    # ═══ 3. KRUSKAL-WALLIS + PAIRWISE ═══
    print("\nKruskal-Wallis + pairwise tests...", flush=True)
    kw_rows, pw_rows = [], []
    for pk in prop_keys:
        groups = [props[t][pk].dropna().values for t in TOOLS]
        H, p = stats.kruskal(*groups)
        kw_rows.append({"Property": PROP_NAMES[pk], "H": round(H,3), "df": len(TOOLS)-1,
                        "p": p, "sig": '***' if p<1e-3 else ('**' if p<1e-2 else ('*' if p<0.05 else 'ns'))})
        # pairwise Mann-Whitney U + BH-FDR
        pairs = list(itertools.combinations(range(len(TOOLS)), 2))
        pw_vals = []
        for i, j in pairs:
            u, pv = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            pw_vals.append((i, j, u, float(pv)))
        # BH-FDR (robust implementation)
        pvals_arr = np.array([x[3] for x in pw_vals], dtype=float)
        m = len(pvals_arr)
        order = np.argsort(pvals_arr)
        fdr_arr = np.empty(m)
        running = 1.0
        for k in reversed(range(m)):
            idx = order[k]
            running = min(pvals_arr[idx] * m / (k + 1), running)
            fdr_arr[idx] = running
        fdr_arr = np.minimum(fdr_arr, 1.0)
        for k, (i, j, u, pv) in enumerate(pw_vals):
            a, b = groups[i], groups[j]
            n, mm = len(a), len(b)
            gt = sum(np.sum(a[x] > b) for x in range(n))
            lt = sum(np.sum(a[x] < b) for x in range(n))
            cd = (gt - lt) / (n * mm)
            pw_rows.append({"Property": PROP_NAMES[pk], "Tool1": TOOLS[i], "Tool2": TOOLS[j],
                            "U": float(u), "raw_p": pv, "fdr_p": float(fdr_arr[k]),
                            "cliffs_delta": round(cd,3),
                            "sig": '***' if fdr_arr[k]<1e-3 else ('**' if fdr_arr[k]<1e-2 else ('*' if fdr_arr[k]<0.05 else 'ns'))})
    kw_df = pd.DataFrame(kw_rows)
    kw_df.to_csv(f"{OUT}/kruskal_wallis.csv", index=False)
    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(f"{OUT}/pairwise_tests.csv", index=False)
    print("Saved kruskal_wallis.csv + pairwise_tests.csv", flush=True)

    # ═══ 4. PAIRWISE EFFECT-SIZE HEATMAP (mean |Cliff's δ| across properties) ═══
    n_t = len(TOOLS)
    eff = np.zeros((n_t, n_t))
    for pk in prop_keys:
        groups = [props[t][pk].dropna().values for t in TOOLS]
        for i in range(n_t):
            for j in range(n_t):
                if i == j: continue
                a, b = groups[i], groups[j]
                n, mm = len(a), len(b)
                gt = sum(np.sum(a[x] > b) for x in range(n))
                lt = sum(np.sum(a[x] < b) for x in range(n))
                eff[i, j] += abs((gt - lt) / (n * mm))
    eff /= len(prop_keys)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(eff, cmap='YlGnBu')
    ax.set_xticks(range(n_t)); ax.set_xticklabels(TOOLS, rotation=45, ha='right')
    ax.set_yticks(range(n_t)); ax.set_yticklabels(TOOLS)
    for i in range(n_t):
        for j in range(n_t):
            ax.text(j, i, f'{eff[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if eff[i,j] > eff.max()*0.6 else 'black')
    ax.set_title('Mean |Cliff\'s δ| across physicochemical properties')
    fig.colorbar(im, ax=ax, label='Mean |effect size|')
    fig.tight_layout()
    fig.savefig(f"{OUT}/heatmap_pairwise_effect.svg")
    fig.savefig(f"{OUT}/heatmap_pairwise_effect.png")
    plt.close(fig)
    print("Saved heatmap_pairwise_effect", flush=True)

    # ═══ 5. FINGERPRINTS ═══
    print("\nComputing Morgan fingerprints...", flush=True)
    fps = {t: fp_matrix(smiles[t]) for t in TOOLS}
    for t in TOOLS:
        print(f"  {t}: {fps[t].shape}", flush=True)

    # ═══ 6. PCA ═══
    print("\nPCA...", flush=True)
    all_smis = [s for t in TOOLS for s in smiles[t]]
    all_tool = [t for t in TOOLS for _ in smiles[t]]
    X_all = np.vstack([fps[t] for t in TOOLS])
    pca = PCA(n_components=min(50, X_all.shape[0]-1), random_state=42)
    X_pca = pca.fit_transform(X_all.astype(float))
    var1 = pca.explained_variance_ratio_[0]*100
    var2 = pca.explained_variance_ratio_[1]*100
    cum = np.cumsum(pca.explained_variance_ratio_)
    cum90 = int(np.where(cum >= 0.90)[0][0] + 1) if np.any(cum >= 0.90) else len(cum)
    cum95 = int(np.where(cum >= 0.95)[0][0] + 1) if np.any(cum >= 0.95) else len(cum)
    print(f"  PC1={var1:.1f}% PC2={var2:.1f}% | PCs for 90%={cum90}, 95%={cum95}", flush=True)

    # ═══ 7. UMAP ═══
    print("UMAP...", flush=True)
    # Use Tanimoto distance (Jaccard) via pairwise + UMAP precomputed
    # Sampling for speed if > 3000 molecules
    n_total = len(X_all)
    step = max(1, n_total // 3000)
    idx_use = np.arange(0, n_total, step)
    D = 1 - np.array(DataStructs.BulkTanimotoSimilarity if False else 0)
    # Use sklearn pairwise Jaccard on the subsample
    from sklearn.metrics import pairwise_distances
    X_sub = X_all[idx_use]
    tool_sub = [all_tool[i] for i in idx_use]
    D_jac = pairwise_distances(X_sub, metric='jaccard')
    reducer = umap.UMAP(n_components=2, random_state=42, metric='precomputed', n_neighbors=30, min_dist=0.1)
    X_umap = reducer.fit_transform(D_jac)
    print(f"  UMAP on {len(idx_use)} molecules", flush=True)

    # ═══ 8. t-SNE ═══
    print("t-SNE...", flush=True)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(idx_use)//5 - 1)),
                learning_rate=200, max_iter=1000, random_state=42, metric='precomputed', init='random')
    X_tsne = tsne.fit_transform(D_jac)
    print(f"  t-SNE perplexity={tsne.perplexity} lr={tsne.learning_rate} iters={tsne.max_iter} seed=42", flush=True)

    # ═══ 9. SILHOUETTE ═══
    print("Silhouette scores...", flush=True)
    lab = np.array([TOOLS.index(t) for t in all_tool])
    # A. original fingerprint space (Jaccard distance)
    sil_orig = silhouette_score(D_jac, np.array([TOOLS.index(t) for t in tool_sub]), metric='precomputed')
    # B. PCA space (enough PCs for ~90%)
    lab_sub = np.array([TOOLS.index(t) for t in tool_sub])
    X_pca_sub = X_pca[idx_use, :cum95]
    sil_pca = silhouette_score(X_pca_sub, lab_sub, metric='euclidean')
    # C. UMAP space
    sil_umap = silhouette_score(X_umap, lab_sub, metric='euclidean')
    # D. t-SNE space
    sil_tsne = silhouette_score(X_tsne, lab_sub, metric='euclidean')
    print(f"  Silhouette — original: {sil_orig:.3f} | PCA: {sil_pca:.3f} | UMAP: {sil_umap:.3f} | t-SNE: {sil_tsne:.3f}", flush=True)

    # per-tool silhouette (original space)
    s_vals = silhouette_samples(D_jac, lab_sub, metric='precomputed')
    per_tool = {}
    for t in TOOLS:
        m = lab_sub == TOOLS.index(t)
        per_tool[t] = (float(s_vals[m].mean()), float(np.median(s_vals[m])), int(m.sum()))
    sil_rows = pd.DataFrame([{"Tool": t, "mean_sil": per_tool[t][0], "median_sil": per_tool[t][1], "n": per_tool[t][2]}
                             for t in TOOLS])
    sil_rows.loc[len(sil_rows)] = {"Tool": "GLOBAL", "mean_sil": sil_orig, "median_sil": np.median(s_vals), "n": len(s_vals)}
    sil_rows.to_csv(f"{OUT}/silhouette_scores.csv", index=False)
    print("Saved silhouette_scores.csv", flush=True)

    # ═══ 10. PER-TOOL SILHOUETTE PLOT ═══
    fig, ax = plt.subplots(figsize=(9, 5))
    # group by tool
    y_pos = []
    y_lab = []
    colors_list = []
    y_cur = 0
    for t in TOOLS:
        m = lab_sub == TOOLS.index(t)
        vals = s_vals[m]
        y_pos.extend(range(y_cur, y_cur + len(vals)))
        y_lab.extend([t]*len(vals))
        colors_list.extend([COLORS[t]]*len(vals))
        y_cur += len(vals)
    ax.barh(y_pos, s_vals, color=colors_list, edgecolor='none', height=1.0, alpha=0.8)
    ax.axvline(sil_orig, color='black', ls='--', lw=1.5, label=f'global = {sil_orig:.3f}')
    # tool separators
    y_cum = np.cumsum([per_tool[t][2] for t in TOOLS])
    for yy in y_cum[:-1]:
        ax.axhline(yy-0.5, color='#888', lw=0.8)
    ax.set_xlabel('Silhouette coefficient')
    ax.set_ylabel('Molecule (grouped by tool)')
    ax.set_title(f'Per-tool silhouette plot (original fingerprint space) — global {sil_orig:.3f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/silhouette_plot.svg")
    fig.savefig(f"{OUT}/silhouette_plot.png")
    plt.close(fig)
    print("Saved silhouette_plot", flush=True)

    # ═══ 11. PAIRWISE CHEMICAL-SPACE SEPARATION (mean Jaccard distance) ═══
    sep = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(n_t):
            if i == j: continue
            # mean Jaccard distance between tool i and tool j (subsample)
            a_i = np.where(np.array(tool_sub) == TOOLS[i])[0]
            a_j = np.where(np.array(tool_sub) == TOOLS[j])[0]
            if len(a_i) == 0 or len(a_j) == 0:
                sep[i, j] = np.nan; continue
            d_ij = D_jac[np.ix_(a_i, a_j)]
            sep[i, j] = float(d_ij.mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sep, cmap='YlOrRd')
    ax.set_xticks(range(n_t)); ax.set_xticklabels(TOOLS, rotation=45, ha='right')
    ax.set_yticks(range(n_t)); ax.set_yticklabels(TOOLS)
    for i in range(n_t):
        for j in range(n_t):
            if i != j and not np.isnan(sep[i,j]):
                ax.text(j, i, f'{sep[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if sep[i,j] > sep[~np.isnan(sep)].max()*0.7 else 'black')
    ax.set_title('Mean pairwise Jaccard distance (chemical-space separation)')
    fig.colorbar(im, ax=ax, label='Mean Jaccard distance (0=identical, 1=distinct)')
    fig.tight_layout()
    fig.savefig(f"{OUT}/heatmap_pairwise_separation.svg")
    fig.savefig(f"{OUT}/heatmap_pairwise_separation.png")
    plt.close(fig)
    print("Saved heatmap_pairwise_separation", flush=True)

    # ═══ 12. DIVERSITY ═══
    print("\nDiversity analysis...", flush=True)
    div_rows = []
    for t in TOOLS:
        fps_t = fps[t]
        if len(fps_t) > 500:
            idx_d = np.random.RandomState(42).choice(len(fps_t), 500, replace=False)
            fps_t = fps_t[idx_d]
        sims = []
        for i in range(len(fps_t)):
            for j in range(i+1, min(i+200, len(fps_t))):
                s = 1 - np.sum(np.bitwise_xor(fps_t[i], fps_t[j])) / 2048.0
                sims.append(s)
        div_rows.append({"Tool": t, "Mean Tanimoto": round(np.mean(sims),3),
                         "Median Tanimoto": round(np.median(sims),3),
                         "Internal diversity (1-mean)": round(1-np.mean(sims),3),
                         "pairs_sampled": len(sims)})
    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(f"{OUT}/diversity.csv", index=False)
    print("Saved diversity.csv", flush=True)
    print(div_df.to_string(index=False), flush=True)

    # ═══ 13. FINAL MULTI-PANEL FIGURE ═══
    print("\nBuilding final multi-panel figure...", flush=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    # A-C: MW, logP, TPSA boxplots
    for ax, pk in zip([axes[0,0], axes[0,1], axes[0,2]], ['MW','logP','TPSA']):
        box_jitter(ax, [props[t][pk].dropna().values for t in TOOLS], TOOLS, pk, PROP_NAMES[pk])
        ax.set_title(PROP_NAMES[pk])
    # D: PCA
    ax = axes[1,0]
    for t in TOOLS:
        m = np.array(tool_sub) == t
        ax.scatter(X_pca[idx_use[m],0], X_pca[idx_use[m],1], s=4, alpha=0.35, color=COLORS[t], label=t)
    ax.set_xlabel(f'PC1 ({var1:.1f}%)'); ax.set_ylabel(f'PC2 ({var2:.1f}%)')
    ax.set_title('PCA of generated chemical space')
    ax.legend(fontsize=7, markerscale=3)
    # E: UMAP
    ax = axes[1,1]
    for t in TOOLS:
        m = np.array(tool_sub) == t
        ax.scatter(X_umap[m,0], X_umap[m,1], s=4, alpha=0.35, color=COLORS[t], label=t)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP projection (Jaccard)')
    ax.legend(fontsize=7, markerscale=3)
    # F: t-SNE
    ax = axes[1,2]
    for t in TOOLS:
        m = np.array(tool_sub) == t
        ax.scatter(X_tsne[m,0], X_tsne[m,1], s=4, alpha=0.35, color=COLORS[t], label=t)
    ax.set_xlabel('t-SNE1'); ax.set_ylabel('t-SNE2')
    ax.set_title('t-SNE visualization')
    ax.legend(fontsize=7, markerscale=3)
    # G: silhouette (bar per tool + global)
    ax = axes[2,0]
    s_tools = [per_tool[t][0] for t in TOOLS]
    bars = ax.bar(TOOLS, s_tools, color=[COLORS[t] for t in TOOLS], edgecolor=[EDGES[t] for t in TOOLS])
    ax.axhline(sil_orig, color='black', ls='--', lw=1.5, label=f'global={sil_orig:.3f}')
    for b, v in zip(bars, s_tools):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('Mean silhouette (original space)')
    ax.set_title('Per-tool silhouette')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=8)
    # H: pairwise separation heatmap (reuse sep)
    ax = axes[2,1]
    im = ax.imshow(sep, cmap='YlOrRd')
    ax.set_xticks(range(n_t)); ax.set_xticklabels(TOOLS, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_t)); ax.set_yticklabels(TOOLS, fontsize=7)
    ax.set_title('Pairwise separation (Jaccard dist)')
    fig.colorbar(im, ax=ax)
    # I: diversity
    ax = axes[2,2]
    div = div_df.set_index('Tool')['Internal diversity (1-mean)'].reindex(TOOLS)
    bars = ax.bar(TOOLS, div.values, color=[COLORS[t] for t in TOOLS], edgecolor=[EDGES[t] for t in TOOLS])
    for b, v in zip(bars, div.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('Internal diversity (1 − mean Tanimoto)')
    ax.set_title('Within-tool structural diversity')
    ax.tick_params(axis='x', rotation=35)

    fig.suptitle('Comparative analysis of molecule-generation tools (GSPT1)', fontsize=15, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/tool_analysis_final_panel.svg")
    fig.savefig(f"{OUT}/tool_analysis_final_panel.png")
    plt.close(fig)
    print("Saved tool_analysis_final_panel", flush=True)

    # ═══ 14. INTERPRETATION ═══
    interp = {
        "pc1_var": var1, "pc2_var": var2, "cum_var_95_pcs": cum95,
        "silhouette": {"original": sil_orig, "pca": sil_pca, "umap": sil_umap, "tsne": sil_tsne},
        "per_tool_silhouette": {t: per_tool[t][0] for t in TOOLS},
        "kruskal": kw_df.to_dict('records'),
        "diversity": div_df.to_dict('records'),
    }
    json.dump(interp, open(f"{OUT}/interpretation.json", "w"), indent=2, default=float)
    print("\nAll outputs saved to:", OUT, flush=True)

if __name__ == '__main__':
    main()
