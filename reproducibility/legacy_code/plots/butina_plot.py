#!/usr/bin/env python3
"""
Better Butina analog-series visualization (replaces barplot):
  Panel A: 2D t-SNE of the external chemical space (Tanimoto-distance precomputed),
           colored by the 10 largest Butina clusters (others gray) + labels
  Panel B: rank-size distribution (log-log) + cumulative compound coverage by top-K clusters
Outputs: outputs/butina_plot.png/.svg, butina_cluster_table.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
uniq = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
smis = uniq["canonical_SMILES"].tolist()

import pyarrow.parquet as pq
tack_raw = pq.read_table("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet").to_pandas()
bc = tack_raw[["SMILES", "SMILES_Butina_Cluster"]].drop_duplicates()
bc["SMILES"] = bc["SMILES"].astype(str).apply(canon)
bcm = dict(zip(bc["SMILES"], bc["SMILES_Butina_Cluster"]))
clusters = pd.Series([bcm.get(s) for s in smis])
sizes = clusters.value_counts()
print(f"clusters: {len(sizes)} | covered: {clusters.notna().sum()}/{len(uniq)} | max: {sizes.max()}")

# fingerprints + Tanimoto distance matrix
fps = []
for s in smis:
    m = Chem.MolFromSmiles(s)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None else None)
fps = [f for f in fps if f is not None]
# restrict to fingerprints present
keep = [i for i, f in enumerate(fps)]
smis2 = [smis[i] for i in keep]
n = len(fps)
D = np.zeros((n, n))
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:])
    D[i, i:] = 1 - np.asarray(sims)
    D[i:, i] = D[i, i:]

# t-SNE on precomputed distances
from sklearn.manifold import TSNE
emb = TSNE(n_components=2, metric="precomputed", random_state=42, init="random",
           perplexity=40, max_iter=1000).fit_transform(D)

# top-10 clusters
top = sizes.head(10)
top_ids = set(top.index)
color_map = plt.cm.tab10
colors = {}
for k, cid in enumerate(top_ids):
    colors[cid] = color_map(k)

# ─── Panel A: chemical space colored by cluster ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axes[0]
mask_top = clusters.isin(top_ids).values
ax.scatter(emb[~mask_top, 0], emb[~mask_top, 1], s=8, c="#cccccc", alpha=0.5, label="other clusters")
handles = []
for cid in top_ids:
    m = (clusters.values == cid)
    h = ax.scatter(emb[m, 0], emb[m, 1], s=14, c=[colors[cid]], alpha=0.9,
                   label=f"cluster {cid} (n={sizes[cid]})")
    handles.append(h)
ax.legend(fontsize=7, markerscale=1.2, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"A. External chemical space (t-SNE, Tanimoto-2) — top-10 analog series highlighted\n"
             f"({len(sizes)} Butina clusters, {clusters.notna().sum()} compounds covered)")

# ─── Panel B: rank-size + cumulative coverage ────────
ax = axes[1]
rank = np.arange(1, len(sizes) + 1)
srt = sizes.sort_values(ascending=False).values
ax.loglog(rank, srt, marker="o", ms=4, color="#1f77b4", alpha=0.85)
ax.set_xlabel("Cluster rank (largest → smallest)")
ax.set_ylabel("Compounds per cluster (log)")
ax.set_title("B. Rank-size distribution (analog-series concentration)")
ax2 = ax.twinx()
cum = np.cumsum(srt) / max(srt.sum(), 1)
ax2.plot(rank, cum, color="#d62728", lw=1.5, label="cumulative coverage")
ax2.axhline(0.5, color="#d62728", ls=":", lw=0.8)
ax2.set_ylabel("Cumulative fraction of clustered compounds")
ax2.legend(fontsize=8, loc="lower right")
# annotate top clusters
for cid, s in top.items():
    r = int(np.where(sizes.sort_values(ascending=False).index == cid)[0][0]) + 1
    ax.annotate(f"{cid} (n={s})", (r, s), textcoords="offset points", xytext=(4, 6), fontsize=6.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "butina_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "butina_plot.svg"))

# cluster table
tbl = sizes.sort_values(ascending=False).reset_index()
tbl.columns = ["cluster", "n_compounds"]
tbl["cumulative_frac"] = np.cumsum(tbl["n_compounds"]) / tbl["n_compounds"].sum()
tbl.to_csv(os.path.join(OUT, "butina_cluster_table.csv"), index=False)
print(tbl.head(12).to_string(index=False))
print(f"\nSaved: {OUT}/butina_plot.png/.svg + butina_cluster_table.csv")
