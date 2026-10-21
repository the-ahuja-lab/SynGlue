#!/usr/bin/env python3
"""
Analysis 1 (R1.1a) — PROTAC-DB vs PROTACpedia vs patent coverage comparison.
Levels: unique compounds, targets (POI), E3 ligases, cell lines, assays.
Data: final_dataset_E3_known.csv (Database column = TPDdb/PROTAC-DB/PROTACpedia
combinations, TACK-curated) + patent compounds (PROTAC Patent Compounds xlsx CSV).
Outputs: outputs/coverage_comparison.csv, outputs/coverage_plot.png/.svg
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
DATA = "/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

ext = pd.read_csv(os.path.join(DATA, "final_dataset_E3_known.csv"))
patent = pd.read_csv(os.path.join(DATA, "merged_patent_pubchem_only.csv"), low_memory=False)

print(f"TACK/DB records: {len(ext)} | patent+pubchem rows: {len(patent)}")

# ─── Assign each record to a database family ─────────
def db_family(s):
    s = str(s)
    if "PROTAC-DB" in s and "PROTACpedia" in s and "TPDdb" in s:
        return "PROTAC-DB+PROTACpedia+TPDdb"
    if "PROTAC-DB" in s and "PROTACpedia" in s:
        return "PROTAC-DB+PROTACpedia"
    if "PROTAC-DB" in s and "TPDdb" in s:
        return "PROTAC-DB+TPDdb"
    if "PROTAC-DB" in s:
        return "PROTAC-DB"
    if "PROTACpedia" in s:
        return "PROTACpedia"
    if "TPDdb" in s:
        return "TPDdb"
    return "other"

ext["db"] = ext["Database"].apply(db_family)

# patent family: no endpoint metadata; count unique compounds + targets if available
patent_cols = set(patent.columns)
print("patent cols sample:", list(patent_cols)[:12])

rows = []
def add(dbname, df, smiles_col, poi_col=None, e3_col=None, cell_col=None, assay_col=None):
    uniq = df.drop_duplicates(smiles_col)
    rows.append(dict(
        database=dbname,
        n_compounds=len(uniq),
        n_targets=uniq[poi_col].nunique() if poi_col and poi_col in uniq.columns else "n/a",
        n_e3=uniq[e3_col].nunique() if e3_col and e3_col in uniq.columns else "n/a",
        n_cell_lines=df[cell_col].nunique() if cell_col and cell_col in df.columns else "n/a",
        n_assays=df[assay_col].nunique() if assay_col and assay_col in df.columns else "n/a",
        n_records=len(df),
    ))

for db in ["PROTAC-DB", "PROTACpedia", "TPDdb", "PROTAC-DB+PROTACpedia", "PROTAC-DB+TPDdb", "PROTAC-DB+PROTACpedia+TPDdb"]:
    sub = ext[ext["db"] == db]
    if len(sub):
        add(db, sub, "canonical_SMILES", "POI_UniProt", "E3_Ligase", "Cell_Line", "Assay")
# patent (no endpoint metadata; physchem only)
add("PROTAC-patent (PubChem)", patent, "canonical_SMILES" if "canonical_SMILES" in patent_cols else "SMILES",
    "Target" if "Target" in patent_cols else None,
    "E3_Ligase" if "E3_Ligase" in patent_cols else None,
    None, None)

cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(OUT, "coverage_comparison.csv"), index=False)
print("\n" + cmp.to_string(index=False))

# ─── Plot ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, title in [(axes[0], "n_compounds", "Unique compounds"),
                       (axes[1], "n_targets", "Targets (POI)"),
                       (axes[2], "n_e3", "E3 ligases")]:
    d = cmp[cmp[col] != "n/a"].copy()
    d[col] = pd.to_numeric(d[col])
    ax.bar(d["database"], d[col], color="#1f77b4", alpha=0.85)
    for x, v in zip(range(len(d)), d[col]):
        ax.text(x, v * 1.02, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20, labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "coverage_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "coverage_plot.svg"))
print(f"\nSaved: {OUT}/coverage_comparison.csv + coverage_plot.png/.svg")
