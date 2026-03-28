import pandas as pd

INPUT_FILE = "00_Database/processed/target_harmonized.csv"
UNIPROT_MAP_FILE = "00_Database/processed/uniprot_mapping.csv"

OUTPUT_FILE = "00_Database/final/gene_uniprot_mapped.csv"

VALID_SPECIES = ["Homo sapiens", "Mus musculus", "Saccharomyces cerevisiae"]

df = pd.read_csv(INPUT_FILE)
map_df = pd.read_csv(UNIPROT_MAP_FILE)

print("Applying gene mapping...")

# Ensure species filter
df = df[df["species"].isin(VALID_SPECIES)]

# Merge gene info
df = df.merge(
    map_df[["uniprot_id", "gene_symbol", "gene_name", "species"]],
    how="left",
    left_on="final_uniprot",
    right_on="uniprot_id"
)

# Flags
df["flag_missing_gene"] = df["gene_symbol"].isna()

# Final columns
df_final = df[
    [
        "canonical_smiles",
        "final_uniprot",
        "gene_symbol",
        "gene_name",
        "species",
        "assay_id",
        "source"
    ]
]

df_final.to_csv(OUTPUT_FILE, index=False)

print(f"Saved final mapped file: {OUTPUT_FILE}, Shape: {df_final.shape}")