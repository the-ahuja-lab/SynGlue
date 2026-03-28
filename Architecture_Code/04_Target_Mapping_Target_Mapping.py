import os
import time
import pickle
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool
from rdkit import Chem
from rdkit.Chem import Recap, rdFMCS
from rdkit.Chem.rdmolops import DeleteSubstructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# =========================================================
# CONFIGURATION
# =========================================================

TRIE_PATH = "MagnetDB_Trie.pkl"
HASH_PATH = "Metadata_Hash.pkl"

INPUT_CSV = "input_smiles.csv"
OUTPUT_DIR = "Results/"
OUTPUT_FILE = "Mapping_Results.csv"

NUM_WORKERS = 10

# =========================================================
# LOAD DATABASE
# =========================================================

def load_database(trie_path, hash_path):
    with open(trie_path, "rb") as f:
        trie = pickle.load(f)
    with open(hash_path, "rb") as f:
        metadata = pickle.load(f)
    return trie, metadata


# =========================================================
# WORKER FUNCTION
# =========================================================

def hybrid_worker(row_data, trie, metadata):

    q_smiles, q_name = row_data

    mol = Chem.MolFromSmiles(q_smiles)
    if mol is None:
        return None

    q_atoms = mol.GetNumHeavyAtoms()

    # RECAP fragmentation
    recap_tree = Recap.RecapDecompose(mol)
    terminal_frags = [
        smi for smi in recap_tree.children.keys()
        if smi.count('*') == 1
    ]

    preliminary_hits = []

    # -------------------------
    # TRIE SEARCH
    # -------------------------
    for frag in terminal_frags:

        search_str = (frag + "$")[::-1]
        node = trie.root

        for char in search_str:
            if char not in node.children:
                node = None
                break
            node = node.children[char]

        if node is None:
            continue

        frag_mol = Chem.MolFromSmiles(frag)
        if frag_mol is None:
            continue

        q_frag_atoms = DeleteSubstructs(
            frag_mol, Chem.MolFromSmarts('[#0]')
        ).GetNumHeavyAtoms()

        for db_id in node.tag_ids:
            if db_id not in metadata:
                continue

            meta = metadata[db_id]
            t_atoms = meta["Target_Atom_Count"]

            f_q = (q_frag_atoms / q_atoms) * 100
            f_t = (q_frag_atoms / t_atoms) * 100

            if f_q >= 25.0 and f_t >= 50.0:
                preliminary_hits.append({
                    "Database_ID": db_id,
                    "Target_ID": meta["Target_ID"],
                    "Target_Name": meta["Target_Name"],
                    "Organism": meta["Organism"],
                    "Target_SMILES": meta["Original_SMILES"],
                    "Query_Fragment": frag,
                    "Fast_Score": f_q + f_t,
                    "Query_Name": q_name
                })

    if not preliminary_hits:
        return None

    # -------------------------
    # FAST FILTER (Top 300)
    # -------------------------
    df = (
        pd.DataFrame(preliminary_hits)
        .sort_values(by="Fast_Score", ascending=False)
        .drop_duplicates(subset=["Database_ID"])
        .head(300)
    )

    # -------------------------
    # MCS VALIDATION
    # -------------------------
    def validate(row):
        t_mol = Chem.MolFromSmiles(row["Target_SMILES"])
        if t_mol is None:
            return pd.Series([0, 0])

        mcs = rdFMCS.FindMCS(
            [mol, t_mol],
            timeout=1,
            ringMatchesRingOnly=True
        )

        if not mcs.smartsString:
            return pd.Series([0, 0])

        shared_atoms = Chem.MolFromSmarts(mcs.smartsString).GetNumHeavyAtoms()

        return pd.Series([
            round((shared_atoms / q_atoms) * 100, 2),
            round((shared_atoms / t_mol.GetNumHeavyAtoms()) * 100, 2)
        ])

    df[["Query_Percentage", "Target_Percentage"]] = df.apply(validate, axis=1)

    df = df[df["Target_Percentage"] >= 75.0]

    if df.empty:
        return None

    return df.drop(columns=["Fast_Score"])


# =========================================================
# MAIN ENGINE
# =========================================================

def run_engine(input_csv, output_dir, num_workers=10):

    print("\n=== SynGlue Hybrid Mapping Engine ===\n")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading database...")
    trie, metadata = load_database(TRIE_PATH, HASH_PATH)

    print("Reading input...")
    df = pd.read_csv(input_csv)

    smiles_col = "SMILES"
    name_col = "Name"

    tasks = [
        (str(row[smiles_col]), str(row[name_col]))
        for _, row in df.iterrows()
    ]

    print(f"Processing {len(tasks)} molecules with {num_workers} workers...")

    pool = ProcessPool(nodes=num_workers)

    t0 = time.time()

    results = list(
        tqdm(
            pool.imap(lambda x: hybrid_worker(x, trie, metadata), tasks),
            total=len(tasks),
            desc="Mapping"
        )
    )

    final_results = [r for r in results if r is not None]

    if final_results:
        final_df = (
            pd.concat(final_results)
            .sort_values(by="Target_Percentage", ascending=False)
        )

        output_path = os.path.join(output_dir, OUTPUT_FILE)
        final_df.to_csv(output_path, index=False)

        print(f"\nSaved results: {output_path}")
        print(f"Total matches: {final_df.shape[0]}")
    else:
        print("No matches found.")

    print(f"Completed in {time.time() - t0:.2f} seconds")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_engine(INPUT_CSV, OUTPUT_DIR, NUM_WORKERS)