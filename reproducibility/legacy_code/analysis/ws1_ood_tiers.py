#!/usr/bin/env python3
"""
WS1 OOD analysis v2 — external endpoint dataset (final_dataset_E3_known.csv)
============================================================================
External: 6,565 endpoint records (DC50/Dmax) with POI, E3, cell line metadata
Train: 5,200 unique compounds from server_53 Attention/data

Tiers (per protocol WS1 step 8):
  exact_in_train        -> EXCLUDE (leakage)
  scaffold_ood          -> Murcko scaffold unseen in train
  chemical_ood          -> max Tanimoto < 0.4
  chemical_ood_soft     -> 0.4 <= T < 0.6
  chemical_in           -> T >= 0.6
  POI_seen / POI_unseen ; E3_seen / E3_unseen (recorded as flags, not exclusions)

Outputs:
  benchmarking/ws1/ws1_ood_tiers.csv          per unique compound
  benchmarking/ws1/ws1_ood_summary.md
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
EXT = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv"
OUT = os.path.join(REV, "benchmarking", "ws1")
os.makedirs(OUT, exist_ok=True)

T_HARD, T_SOFT = 0.4, 0.6

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None

def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

# ─── Train side ─────────────────────────────────────
print("Loading train...")
train = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False,
                    usecols=lambda c: not c.startswith("Grover_"))
train_smiles = train["PROTAC_SMILES"].dropna().astype(str).unique()
train_canon = {}
for s in train_smiles:
    c = canon(s)
    if c: train_canon[c] = True
train_scaffolds, train_fps = set(), []
for s in list(train_canon.keys()):
    sc = scaffold_of(s)
    if sc: train_scaffolds.add(sc)
    fp = morgan(s)
    if fp is not None: train_fps.append(fp)
train_poi = set(train["warhead_Uniprot"].dropna().astype(str).str.upper().unique())
train_e3  = set(train["E3 ligase"].dropna().astype(str).str.upper().unique())
print(f"  train: {len(train_canon)} canon, {len(train_scaffolds)} scaffolds, {len(train_poi)} POIs, {len(train_e3)} E3s")

# ─── External side ───────────────────────────────────
print("Loading external...")
ext_full = pd.read_csv(EXT)
# per-compound endpoint counts on FULL records (before dedup)
cnt = ext_full.groupby("canonical_SMILES").agg(
    n_records=("Value_Type", "size"),
    n_dc50=("Value_Type", lambda s: (s == "DC50").sum()),
    n_dmax=("Value_Type", lambda s: (s == "Dmax").sum()),
    has_censored=("Value_Operator", lambda s: s.astype(str).isin([">", "<", ">=", "<=", "~"]).any()),
)
ext = ext_full.drop_duplicates("canonical_SMILES").reset_index(drop=True)
print(f"  external records: {len(ext_full)} | unique compounds: {len(ext)}")

rows = []
for _, r in ext.iterrows():
    smi = r["canonical_SMILES"]
    c = canon(smi)
    if c is None:
        rows.append(dict(canonical_smiles=smi, tier="parse_failed")); continue
    sc = scaffold_of(c); fp = morgan(c)
    mt = 0.0
    if fp is not None:
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        mt = float(max(sims)) if sims else 0.0
    exact = c in train_canon
    sc_known = (sc is not None) and (sc in train_scaffolds)
    if exact: tier = "exact_in_train"
    elif sc is not None and not sc_known: tier = "scaffold_ood"
    elif mt < T_HARD: tier = "chemical_ood"
    elif mt < T_SOFT: tier = "chemical_ood_soft"
    else: tier = "chemical_in"
    rows.append(dict(
        cid=str(r.get("TPD_ID", "")), source=r.get("source_dataset", ""),
        canonical_smiles=c, scaffold=sc, max_tanimoto=mt,
        exact_in_train=exact, scaffold_in_train=sc_known, tier=tier,
        POI=r.get("Protein_of_Interest", ""), POI_UniProt=r.get("POI_UniProt", ""),
        E3=r.get("E3_Ligase", ""), E3_UniProt=r.get("E3_UniProt", ""),
        Cell_Line=r.get("Cell_Line", ""),
        POI_seen=(str(r.get("POI_UniProt", "")).upper() in train_poi) if pd.notna(r.get("POI_UniProt")) else None,
        E3_seen=(str(r.get("E3_Ligase", "")).upper() in train_e3) if pd.notna(r.get("E3_Ligase")) else None,
    ))

df = pd.DataFrame(rows)
# merge per-compound endpoint counts (computed on FULL records above)
df = df.merge(cnt.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}), on="canonical_smiles", how="left")
df.to_csv(os.path.join(OUT, "ws1_ood_tiers.csv"), index=False)

# ─── Summary ─────────────────────────────────────────
lines = ["# WS1 OOD Tiers — External Endpoint Dataset (final_dataset_E3_known.csv)",
         "",
         f"- External unique compounds: {len(df)}",
         f"- Records (all): {len(ext_full)} | DC50 records: {(ext_full['Value_Type']=='DC50').sum()} | Dmax: {(ext_full['Value_Type']=='Dmax').sum()}",
         f"- Unique w/ DC50: {df['n_dc50'].gt(0).sum()} | w/ Dmax: {df['n_dmax'].gt(0).sum()}",
         f"- Compounds w/ censored operators (>,<,~): {int(df['has_censored'].fillna(False).sum())}",
         "",
         "## Tier distribution (unique compounds)",
         "| Tier | N | DC50 | Dmax |",
         "|------|---|------|------|"]
for t, g in df.groupby("tier"):
    lines.append(f"| {t} | {len(g)} | {int(g['n_dc50'].gt(0).sum())} | {int(g['n_dmax'].gt(0).sum())} |")
lines += ["", "## POI / E3 seen-unseen (unique compounds)", "",
          "| Flag | N |", "|------|---|"]
for k, v in df["POI_seen"].value_counts(dropna=False).items():
    lines.append(f"| POI_seen={k} | {v} |")
for k, v in df["E3_seen"].value_counts(dropna=False).items():
    lines.append(f"| E3_seen={k} | {v} |")
lines += ["", "## Max Tanimoto (external→train)", "",
          f"- median {df['max_tanimoto'].median():.3f} | mean {df['max_tanimoto'].mean():.3f}",
          f"- <0.2: {int((df['max_tanimoto']<0.2).sum())} | 0.2-0.4: {int(((df['max_tanimoto']>=0.2)&(df['max_tanimoto']<0.4)).sum())} | "
          f"0.4-0.6: {int(((df['max_tanimoto']>=0.4)&(df['max_tanimoto']<0.6)).sum())} | >=0.6: {int((df['max_tanimoto']>=0.6).sum())}", ""]
open(os.path.join(OUT, "ws1_ood_summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved: {OUT}/ws1_ood_tiers.csv")
