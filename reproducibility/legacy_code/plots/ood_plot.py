#!/usr/bin/env python3
"""
OOD characterization plot — external benchmark novelty (independent of performance).
Panels:
  A. Distribution of maximum Tanimoto to SynGlue training (Morgan-2, 2048-bit),
     with 0.4 / 0.6 thresholds
  B. Bemis-Murcko scaffold overlap vs training (novel vs shared)
  C. Target & E3-ligase novelty vs training annotations
  D. Analog-series structure: Butina cluster size distribution (top clusters)
Outputs: outputs/ood_distributions.csv, ood_plot.png/.svg
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"

def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def scaffold_of(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None

# ─── external pool (no PROTAC-DB) ────────────────────
ext = pd.read_csv(EXT)
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
uniq = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
print(f"external (no PROTAC-DB): {len(uniq)}")

# max Tanimoto to training (from ws1 tiers, which computed vs SynGlue train)
tiers = pd.read_csv(os.path.join(REV, "benchmarking", "ws1", "ws1_ood_tiers.csv"))
m = uniq.merge(tiers[["canonical_smiles", "max_tanimoto", "scaffold_in_train"]],
               left_on="canonical_SMILES", right_on="canonical_smiles", how="left")
print(f"with tanimoto: {m['max_tanimoto'].notna().sum()}")

# training sets
tr = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False,
                 usecols=lambda c: not c.startswith("Grover_"))
syn_train = {c for c in map(canon, tr["PROTAC_SMILES"].dropna().astype(str).unique()) if c}
train_scaf = {scaffold_of(s) for s in syn_train if scaffold_of(s)}
train_targets = set(tr["warhead_Target"].dropna().astype(str).str.upper())
train_e3 = set(tr["E3 ligase"].dropna().astype(str).str.upper())

# scaffold novelty (recompute on pool)
scaf_novel, scaf_shared, scaf_none = 0, 0, 0
for s in uniq["canonical_SMILES"]:
    sc = scaffold_of(s)
    if sc is None: scaf_none += 1
    elif sc in train_scaf: scaf_shared += 1
    else: scaf_novel += 1
print(f"scaffolds: novel={scaf_novel}, shared={scaf_shared}, none={scaf_none}")

# target / E3 novelty
ext_targets = set(uniq["Protein_of_Interest"].dropna().astype(str).str.upper())
ext_e3 = set(uniq["E3_Ligase"].dropna().astype(str).str.upper())
target_novel = len(ext_targets - train_targets)
target_seen = len(ext_targets & train_targets)
e3_novel = len(ext_e3 - train_e3)
e3_seen = len(ext_e3 & train_e3)

# Butina clusters (from TACK raw)
try:
    import pyarrow.parquet as pq
    tack_raw = pq.read_table("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/tack_raw/data_train-00000-of-00001.parquet").to_pandas()
    bc = tack_raw[["SMILES", "SMILES_Butina_Cluster"]].drop_duplicates()
    bc["SMILES"] = bc["SMILES"].astype(str).apply(canon)
    bcm = dict(zip(bc["SMILES"], bc["SMILES_Butina_Cluster"]))
    clusters = uniq["canonical_SMILES"].map(bcm).dropna()
    sizes = clusters.value_counts()
    n_clusters = len(sizes)
    print(f"Butina clusters: {n_clusters} | median size: {sizes.median():.0f} | max size: {sizes.max()}")
except Exception as e:
    sizes = pd.Series(dtype=int); n_clusters = 0
    print("butina unavailable:", e)

# ─── distributions CSV ───────────────────────────────
tan = m["max_tanimoto"].dropna().values
dist = pd.DataFrame({
    "metric": ["n_external", "scaffold_novel_n", "scaffold_shared_n", "scaffold_unresolved_n",
               "scaffold_novel_pct", "target_novel_n", "target_seen_n", "e3_novel_n", "e3_seen_n",
               "n_butina_clusters", "median_max_tanimoto", "pct_tanimoto_lt0.2",
               "pct_tanimoto_0.2_0.4", "pct_tanimoto_0.4_0.6", "pct_tanimoto_ge0.6"],
    "value": [len(uniq), scaf_novel, scaf_shared, scaf_none,
              100 * scaf_novel / max(scaf_novel + scaf_shared, 1),
              target_novel, target_seen, e3_novel, e3_seen, n_clusters,
              float(np.median(tan)), 100 * np.mean(tan < 0.2),
              100 * np.mean((tan >= 0.2) & (tan < 0.4)),
              100 * np.mean((tan >= 0.4) & (tan < 0.6)),
              100 * np.mean(tan >= 0.6)]})
dist.to_csv(os.path.join(OUT, "ood_distributions.csv"), index=False)
print(dist.to_string(index=False))

# ─── figure ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
ax = axes[0, 0]
ax.hist(tan, bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
for th, lab in [(0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6")]:
    ax.axvline(th, color="#d62728", ls="--", lw=1)
    ax.text(th, ax.get_ylim()[1] * 0.95, lab, color="#d62728", fontsize=8)
ax.set_xlabel("Maximum Tanimoto to SynGlue training (Morgan-2, 2048-bit)")
ax.set_ylabel("Compounds")
ax.set_title(f"A. Chemical similarity to training (n={len(tan)}, median={np.median(tan):.2f})")

ax = axes[0, 1]
labels = ["Scaffold-novel", "Scaffold shared", "Unresolved"]
vals = [scaf_novel, scaf_shared, scaf_none]
bars = ax.bar(labels, vals, color=["#2ca02c", "#ff7f0e", "#7f7f7f"], alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v}", ha="center", fontsize=10)
ax.set_ylabel("Compounds")
ax.set_title("B. Bemis-Murcko scaffold overlap vs training")

ax = axes[1, 0]
cats = ["Targets\n(novel)", "Targets\n(seen)", "E3\n(novel)", "E3\n(seen)"]
vals2 = [target_novel, target_seen, e3_novel, e3_seen]
bars = ax.bar(cats, vals2, color=["#d62728", "#1f77b4", "#d62728", "#1f77b4"], alpha=0.85)
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}", ha="center", fontsize=10)
ax.set_ylabel("Count")
ax.set_title("C. Target & E3-ligase novelty vs training annotations")

ax = axes[1, 1]
top = sizes.head(15)
ax.bar(range(len(top)), top.values, color="#9467bd", alpha=0.85)
ax.set_xticks(range(len(top)))
ax.set_xticklabels([str(i) for i in top.index], rotation=45, fontsize=7)
ax.set_xlabel("Butina cluster ID (top 15)")
ax.set_ylabel("Compounds per cluster")
ax.set_title(f"D. Analog series: {n_clusters} Butina clusters (max size {sizes.max() if len(sizes) else 0})")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ood_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "ood_plot.svg"))
print(f"\nSaved: {OUT}/ood_plot.png/.svg + ood_distributions.csv")
