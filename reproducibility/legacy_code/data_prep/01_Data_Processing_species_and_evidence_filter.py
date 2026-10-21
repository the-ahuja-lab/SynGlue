import pandas as pd
import uuid

# ==============================
# CONFIG
# ==============================

INPUT_FILE = "00_Database/processed/gene_uniprot_mapped.csv"

OUTPUT_FILE = "00_Database/final/MagnetDB_final.csv"
SUMMARY_FILE = "00_Database/final/MagnetDB_summary.txt"

# Allowed species (NCBI Taxonomy IDs)
VALID_TAX_IDS = {
    9606: "Homo sapiens",
    10090: "Mus musculus",
    4932: "Saccharomyces cerevisiae"
}

# Experimental evidence keywords
VALID_ASSAY_TYPES = ["Kd", "Ki", "IC50", "EC50"]

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)
print(f"Initial dataset: {df.shape}")

# Expected columns:
# ['canonical_smiles', 'final_uniprot', 'gene_symbol',
#  'species', 'taxonomy_id', 'assay_type', 'assay_value',
#  'ligand_name', 'target_name', 'source']

# ==============================
# 1. SPECIES FILTERING
# ==============================

print("Filtering by species...")

df = df[df["taxonomy_id"].isin(VALID_TAX_IDS.keys())]

print(f"After species filter: {df.shape}")

# ==============================
# 2. EXPERIMENTAL EVIDENCE FILTER
# ==============================

print("Filtering for experimental evidence...")

df["assay_type"] = df["assay_type"].astype(str)

df_exp = df[
    df["assay_type"].str.contains("|".join(VALID_ASSAY_TYPES), case=False, na=False)
]

print(f"After experimental filter: {df_exp.shape}")

# ==============================
# 3. REMOVE INDIRECT / NON-BINDING DATA
# ==============================

# Remove inferred / predicted / text-mined
EXCLUDE_TERMS = ["predicted", "inferred", "text-mined", "computational", "network"]

mask_exclude = df_exp["source"].astype(str).str.contains("|".join(EXCLUDE_TERMS), case=False, na=False)

df_exp = df_exp[~mask_exclude]

print(f"After removing indirect data: {df_exp.shape}")

# ==============================
# 4. DEDUPLICATION
# ==============================

print("Deduplicating ligand-target pairs...")

# Aggregate multiple assay entries
agg_dict = {
    "assay_type": lambda x: list(set(x)),
    "assay_value": lambda x: list(x),
    "source": lambda x: list(set(x)),
    "ligand_name": "first",
    "target_name": "first",
    "gene_symbol": "first",
    "species": "first"
}

df_final = (
    df_exp
    .dropna(subset=["canonical_smiles", "final_uniprot"])
    .groupby(["canonical_smiles", "final_uniprot"])
    .agg(agg_dict)
    .reset_index()
)

print(f"After deduplication: {df_final.shape}")

# ==============================
# 5. ADD UNIQUE INTERACTION ID
# ==============================

print("Generating unique interaction IDs...")

df_final["interaction_id"] = [
    f"INT_{uuid.uuid4().hex[:12]}" for _ in range(len(df_final))
]

# ==============================
# 6. FINAL COLUMN ORDER
# ==============================

df_final = df_final[
    [
        "interaction_id",
        "canonical_smiles",
        "ligand_name",
        "final_uniprot",
        "target_name",
        "gene_symbol",
        "species",
        "source",
        "assay_type",
        "assay_value"
    ]
]

# ==============================
# 7. SAVE FINAL DATASET
# ==============================

df_final.to_csv(OUTPUT_FILE, index=False)
print(f"Final MagnetDB saved: {OUTPUT_FILE}")

# ==============================
# 8. SUMMARY STATISTICS
# ==============================

n_interactions = len(df_final)
n_compounds = df_final["canonical_smiles"].nunique()
n_targets = df_final["final_uniprot"].nunique()

with open(SUMMARY_FILE, "w") as f:
    f.write(f"Total interactions: {n_interactions}\n")
    f.write(f"Unique compounds: {n_compounds}\n")
    f.write(f"Unique targets: {n_targets}\n")

print("Summary saved.")

print("\n=== FINAL STATS ===")
print(f"Interactions: {n_interactions}")
print(f"Compounds: {n_compounds}")
print(f"Targets: {n_targets}")