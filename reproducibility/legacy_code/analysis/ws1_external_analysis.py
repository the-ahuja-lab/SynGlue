#!/usr/bin/env python3
"""
WS1 Step 5-7: Leakage analysis — external validation set vs SynGlue training set.
Computes per-compound: exact-match status, Bemis-Murcko scaffold overlap,
max Tanimoto similarity to any training compound, and provisional chemical OOD tier.

Train reference: 5,200 unique PROTAC compounds from server_53 Attention/data
(Upadted_PROTAC_features.csv). External: revision/external_validation_clean.csv (481).

OOD tier definition (chemical levels only; component/POI/E3 tiers added later):
  exact         : canonical SMILES present in train          -> EXCLUDE (leakage)
  scaffold_ood  : Murcko scaffold unseen in train
  chemical_ood  : max Tanimoto < 0.4  (deep chemical novelty)
  chemical_ood_soft : 0.4 <= max Tanimoto < 0.6
  chemical_in   : max Tanimoto >= 0.6

Outputs:
  benchmarking/ws1/ws1_external_overlap.csv
  benchmarking/ws1/ws1_external_summary.md
"""
import os, sys, warnings, hashlib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = os.path.join(REV, "benchmarking", "ws1")
os.makedirs(OUT, exist_ok=True)

TANIMOTO_HARD = 0.4   # below = deep OOD (chemical)
TANIMOTO_SOFT = 0.6   # 0.4-0.6 = soft OOD

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m)

def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None

def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

print("Loading training compounds...")
train = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False,
                    usecols=lambda c: not c.startswith("Grover_"))
train_smiles = train["PROTAC_SMILES"].dropna().astype(str).unique()
print(f"  train rows: {len(train)}, unique SMILES: {len(train_smiles)}")

train_canon = {}
for s in train_smiles:
    c = canon(s)
    if c: train_canon[c] = True
print(f"  train canonical unique: {len(train_canon)}")

# Scaffolds + fingerprints for train (unique canonical)
train_scaffolds = set()
train_fps = []
for s in list(train_canon.keys()):
    sc = scaffold_of(s)
    if sc: train_scaffolds.add(sc)
    fp = morgan(s)
    if fp is not None: train_fps.append(fp)
print(f"  train scaffolds: {len(train_scaffolds)}, train fps: {len(train_fps)}")

print("Loading external set...")
ext = pd.read_csv(os.path.join(REV, "external_validation_clean.csv"))
print(f"  external: {len(ext)}")

rows = []
for i, r in ext.iterrows():
    smi = r["SMILES"]
    c = canon(smi)
    if c is None:
        rows.append({"cid": r["cid"], "name": r["Name"], "canonical_smiles": smi,
                     "exact_in_train": None, "scaffold": None, "scaffold_in_train": None,
                     "max_tanimoto": None, "tier": "parse_failed", "n_train_fp": len(train_fps)})
        continue
    sc = scaffold_of(c)
    fp = morgan(c)
    # max tanimoto vs train
    mt = 0.0
    if fp is not None:
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        mt = float(max(sims)) if sims else 0.0
    exact = c in train_canon
    sc_known = (sc is not None) and (sc in train_scaffolds)
    if exact:
        tier = "exact_in_train (EXCLUDE)"
    elif sc is not None and not sc_known:
        tier = "scaffold_ood"
    elif mt < TANIMOTO_HARD:
        tier = "chemical_ood (T<0.4)"
    elif mt < TANIMOTO_SOFT:
        tier = "chemical_ood_soft (0.4-0.6)"
    else:
        tier = "chemical_in (T>=0.6)"
    rows.append({"cid": r["cid"], "name": r["Name"], "canonical_smiles": c,
                 "exact_in_train": exact, "scaffold": sc, "scaffold_in_train": sc_known,
                 "max_tanimoto": mt, "tier": tier, "n_train_fp": len(train_fps)})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ws1_external_overlap.csv"), index=False)

# Summary
n = len(df)
n_exact = int(df["exact_in_train"].fillna(False).sum())
tier_counts = df["tier"].value_counts()
lines = [
    "# WS1 Leakage Analysis — External 481 vs Train",
    "",
    f"- External compounds: {n}",
    f"- Exact matches in train (EXCLUDE): {n_exact}",
    f"- Parse failures: {int((df['tier']=='parse_failed').sum())}",
    "",
    "## Tier assignment (chemical levels)",
    "",
    "| Tier | N |",
    "|------|---|",
]
for t, c in tier_counts.items():
    lines.append(f"| {t} | {c} |")
lines.append("")
clean = df[df["tier"] != "parse_failed"]
if len(clean):
    lines += [
        "## Max Tanimoto distribution (external → train)",
        "",
        f"- median: {clean['max_tanimoto'].median():.3f}",
        f"- mean: {clean['max_tanimoto'].mean():.3f}",
        f"- <0.2: {int((clean['max_tanimoto']<0.2).sum())} | 0.2-0.4: {int(((clean['max_tanimoto']>=0.2)&(clean['max_tanimoto']<0.4)).sum())} | "
        f"0.4-0.6: {int(((clean['max_tanimoto']>=0.4)&(clean['max_tanimoto']<0.6)).sum())} | >=0.6: {int((clean['max_tanimoto']>=0.6).sum())}",
        "",
        "## Note",
        "",
        "Component novelty (warhead/linker/E3), POI/E3 seen/unseen, and endpoint tiers will be added once",
        "decomposition and external endpoints are available. 481-set currently lacks DC50/Dmax endpoints.",
        "",
    ]
open(os.path.join(OUT, "ws1_external_summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved: {OUT}/ws1_external_overlap.csv")
