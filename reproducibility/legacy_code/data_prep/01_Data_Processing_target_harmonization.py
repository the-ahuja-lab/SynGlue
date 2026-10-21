import pandas as pd

INPUT_FILE = "00_Database/processed/smiles_standardized.csv"
UNIPROT_MAP_FILE = "00_Database/processed/uniprot_mapping.csv"

OUTPUT_FILE = "00_Database/processed/target_harmonized.csv"

# Allowed species
VALID_SPECIES = ["Homo sapiens", "Mus musculus", "Saccharomyces cerevisiae"]

df = pd.read_csv(INPUT_FILE)
map_df = pd.read_csv(UNIPROT_MAP_FILE)

print("Filtering species...")

df = df[df["species"].isin(VALID_SPECIES)]

# Merge UniProt mapping
df = df.merge(
    map_df,
    how="left",
    left_on="target_id",
    right_on="original_id"
)

# Flags
df["flag_unmapped"] = df["uniprot_id"].isna()

# Use canonical UniProt
df["final_uniprot"] = df["canonical_id"]

# Collapse isoforms (P12345-2 → P12345)
df["final_uniprot"] = df["final_uniprot"].astype(str).str.split("-").str[0]

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}, Shape: {df.shape}")