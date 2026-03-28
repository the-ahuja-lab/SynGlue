import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover


INPUT_FILE = "00_Database/processed/raw_interactions.csv"
OUTPUT_FILE = "00_Database/processed/smiles_standardized.csv"

remover = SaltRemover.SaltRemover()

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

df = pd.read_csv(INPUT_FILE)

print("Standardizing SMILES...")

df["canonical_smiles"] = df["ligand_smiles"].apply(standardize_smiles)

# Flags
df["flag_invalid_smiles"] = df["canonical_smiles"].isna()

# Drop invalid
df_clean = df.dropna(subset=["canonical_smiles"])

# Deduplicate
df_clean = df_clean.drop_duplicates(subset=["canonical_smiles"])

df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}, Shape: {df_clean.shape}")