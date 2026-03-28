import os
import sys
import csv
import time
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS, Draw
from IPython.display import display

# =====================================================================
# 1. TRIE & HASH MAP ARCHITECTURE
# =====================================================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self, hash_map):
        self.root = TrieNode()
        self.hash_map = hash_map # Store hash_map locally to avoid global scope issues

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def store_all_fragments(self, prefix, query_smile, query_frag_id, query_id, file):
        node = self.root
        frag = []
        for char in prefix:
            node = node.children[char]
            frag.append(char)
            
        with open(file, mode='a', newline='') as database:
            writer = csv.writer(database)
            self.dfs(frag, node, prefix, writer, query_frag_id, query_smile, query_id)

    def dfs(self, frag, node, prefix, writer, query_frag_id, query_smile, query_id):
        if node.is_end_of_word:
            smile = "".join(frag[::-1][:-1])
            magnet_id = self.hash_map.get(smile, "UNKNOWN")
            # Write: [reversed_prefix, matched_magnet_frag, query_frag_id, Magnet_ID, query_smile, query_smile_id]
            writer.writerow([prefix[::-1][:-1], smile, query_frag_id, magnet_id, query_smile, query_id])
            
        for child in node.children:
            frag.append(child)
            self.dfs(frag, node.children[child], prefix, writer, query_frag_id, query_smile, query_id)
            frag.pop()

# =====================================================================
# 2. CHEMINFORMATICS & SCORING FUNCTIONS
# =====================================================================
def get_atom_count_from_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        return len(mol.GetAtoms()) if mol else 0
    except:
        return 0

def percentage_calculator(part, whole):
    return (part / whole) * 100 if whole else 0

def get_tanimoto_similarity_e(query_mol, target_mol, radius=3, nbits=1048):
    qfp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius, nBits=nbits)
    tfp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius, nBits=nbits)
    return DataStructs.TanimotoSimilarity(qfp, tfp)

def get_tanimoto_similarity_r(query_mol, target_mol):
    qfp = Chem.RDKFingerprint(query_mol)
    tfp = Chem.RDKFingerprint(target_mol)
    return DataStructs.TanimotoSimilarity(qfp, tfp)

def get_mcs_smile(query_mol, target_mol):
    mcs = rdFMCS.FindMCS([query_mol, target_mol], timeout=5)
    return Chem.MolToSmiles(Chem.MolFromSmarts(mcs.smartsString))

def get_mcs_sm_score(query_mol, target_mol):
    try:
        mcs = rdFMCS.FindMCS([query_mol, target_mol], timeout=5)
        mcs_atoms = Chem.MolFromSmarts(mcs.smartsString).GetNumAtoms()
        return mcs_atoms / target_mol.GetNumAtoms() if target_mol.GetNumAtoms() else 0
    except:
        return 0

# =====================================================================
# 3. PIPELINE STEP A: MAP FRAGMENTS (TRIE SEARCH)
# =====================================================================
def run_fragment_mapping(query_fragment_pkl, final_magnet_csv, output_mapping_csv):
    print("\n--- Step 1: Building Trie & Mapping Fragments ---")
    sys.setrecursionlimit(10000)
    hash_map = {}
    trie = Trie(hash_map)

    # 1. Build Trie from Final_MagnetDB
    print("Building Trie from MagnetDB...")
    magnet_df = pd.read_csv(final_magnet_csv)
    for _, row in magnet_df.iterrows():
        frag = str(row['Fragment'])
        if frag != 'nan':
            trie.insert((frag + "$")[::-1])
            hash_map[frag] = row['Magnet Id']

    # 2. Load query fragments
    df_query = pd.read_pickle(query_fragment_pkl)
    if 'querySMILE' not in df_query.columns and 'Canonical_SMILES' in df_query.columns:
        df_query['querySMILE'] = df_query['Canonical_SMILES']

    # 3. Search Trie
    print("Searching Trie for query fragments...")
    with open(output_mapping_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["query_frag_smile", "matched_magnet_frag", "query_frag_id", "Magnet_ID", "query_smile", "query_smile_id"])

        for _, row in tqdm(df_query.iterrows(), total=len(df_query)):
            frag_smile = str(row['queryFragSMILE'])
            reversed_frag = (frag_smile + "$")[::-1]
            if trie.starts_with(reversed_frag):
                trie.store_all_fragments(
                    prefix=reversed_frag, 
                    query_smile=row['querySMILE'], 
                    query_frag_id=row['queryFragID'], 
                    query_id=row['queryID'], 
                    file=output_mapping_csv
                )
    print(f"✅ Fragment Mapping saved to {output_mapping_csv}")

# =====================================================================
# 4. PIPELINE STEP B: MAP TO TARGETS
# =====================================================================
def map_fragments_to_targets(mapping_file, final_magnet_dict_path, direct_binders_dict_path, output_file):
    print("\n--- Step 2: Mapping Fragments to Protein Targets ---")
    input_df = pd.read_csv(mapping_file)
    
    with open(final_magnet_dict_path, 'rb') as f:
        Final_MagnetDB_dict = pickle.load(f)
    with open(direct_binders_dict_path, 'rb') as f:
        direct_binders_dict = pickle.load(f)

    start = time.time()
    absent_source_ids = []

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Database_ID", "original_ligand_smile", "Target_ID", "query_frag_ID", 
            "query_frag_smile", "matched_magnet_frag", "Magnet_ID", "query_smile", 
            "query_ID", "query_smile_atom_count", "Query_Percentage", "Target_Percentage"
        ])

        for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
            magnet_id = row['Magnet_ID']
            if magnet_id not in Final_MagnetDB_dict:
                continue

            source_ids = Final_MagnetDB_dict[magnet_id].get('Source datbase IDs', [])
            query_smile_atom_count = get_atom_count_from_smiles(row['query_smile'])
            query_frag_atom_count = get_atom_count_from_smiles(row['query_frag_smile'])
            target_frag_atom_count = get_atom_count_from_smiles(row['matched_magnet_frag'])

            for source_id in source_ids:
                if source_id not in direct_binders_dict:
                    absent_source_ids.append(source_id)
                    continue

                target_id = direct_binders_dict[source_id]['Target ID']
                original_smile = direct_binders_dict[source_id]['SMILE']
                target_smile_atom_count = get_atom_count_from_smiles(original_smile)

                query_percent = percentage_calculator(query_frag_atom_count, query_smile_atom_count)
                target_percent = percentage_calculator(target_frag_atom_count, target_smile_atom_count)

                writer.writerow([
                    source_id, original_smile, target_id, row['query_frag_id'],
                    row['query_frag_smile'], row['matched_magnet_frag'], magnet_id,
                    row['query_smile'], row['query_smile_id'], query_smile_atom_count,
                    query_percent, target_percent
                ])

    print(f"✅ Target Mapping complete in {time.time() - start:.2f}s. Saved to {output_file}")

# =====================================================================
# 5. PIPELINE STEP C: FILTERING & VISUALIZATION
# =====================================================================
def filter_and_highlight(mapped_db_csv):
    print("\n--- Step 3: Threshold Filtering & Visualization ---")
    df = pd.read_csv(mapped_db_csv)
    
    # Apply your specific thresholds
    print(f"Initial row count: {len(df)}")
    df = df[df['query_smile_atom_count'] >= 10]
    df = df[df["Query_Percentage"] >= 25]
    final_filtered_df = df[df["Target_Percentage"] >= 75]
    print(f"Row count after stringent filtering: {len(final_filtered_df)}")
    
    # Quick Visualization of top 5 matches
    print("\nHighlighting Maximum Common Substructures (MCS) for top hits...")
    for _, row in final_filtered_df.head(5).iterrows():
        query_mol = Chem.MolFromSmiles(row['query_frag_smile'])
        target_mol = Chem.MolFromSmiles(row['matched_magnet_frag'])
        
        if query_mol and target_mol:
            mcs = rdFMCS.FindMCS([query_mol, target_mol], timeout=10, ringMatchesRingOnly=True, completeRingsOnly=True)
            if mcs.smartsString:
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                img = Draw.MolsToGridImage(
                    [query_mol, target_mol],
                    highlightAtomLists=[list(query_mol.GetSubstructMatch(mcs_mol)), list(target_mol.GetSubstructMatch(mcs_mol))],
                    legends=["Query Fragment", f"Target Match (Magnet ID: {row['Magnet_ID']})"],
                    molsPerRow=2,
                    subImgSize=(350, 350)
                )
                display(img)

# =====================================================================
# EXECUTION ROUTINE
# =====================================================================
if __name__ == "__main__":
    
    # Define File Paths
    QUERY_FRAGMENTS_PKL = "/storage/savi/saveenas/Projects/Magnet/Magnet_Package/Imppat_fragments_filtered.pkl"
    FINAL_MAGNET_CSV = "/storage/savi/saveenas/Projects/Magnet/Dataset/FINAL_MAGNET_DATABASE/Final_MagnetDB.csv"
    
    MAGNET_DICT_PKL = "/storage/savi/saveenas/Projects/Magnet/Magnet_Package/Working_Module_2/Workflow/data/Final_MagnetDB_Dictionary.pkl"
    DIRECT_BINDERS_PKL = "Direct_Binders_Dictionary.pkl"
    
    # Output Files
    MAPPING_CSV = "Query_Magnet_Mapping.csv"
    TARGET_MAPPED_CSV = "query_magnet_source_db_mapping.csv"

    # 1. Search Trie
    run_fragment_mapping(
        query_fragment_pkl=QUERY_FRAGMENTS_PKL, 
        final_magnet_csv=FINAL_MAGNET_CSV, 
        output_mapping_csv=MAPPING_CSV
    )
    
    # 2. Map to Targets
    map_fragments_to_targets(
        mapping_file=MAPPING_CSV,
        final_magnet_dict_path=MAGNET_DICT_PKL,
        direct_binders_dict_path=DIRECT_BINDERS_PKL,
        output_file=TARGET_MAPPED_CSV
    )
    
    # 3. Filter and Show High-Quality Hits
    filter_and_highlight(TARGET_MAPPED_CSV)