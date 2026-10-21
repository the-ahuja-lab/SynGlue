import pandas as pd
import csv
import time

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def store_all_fragments(self, prefix, smile, hmdb_id, file, hash_map):
        node = self.root
        frag = []
        for char in prefix:
            node = node.children[char]
            frag.append(char)
        with open(file, mode='a', newline='') as database:
            writer = csv.writer(database)
            self.dfs(frag, node, prefix, writer, hmdb_id, smile, hash_map)

    def dfs(self, frag, node, prefix, writer, hmdb_id, smile, hash_map):
        if node.is_end_of_word:
            writer.writerow([
                prefix[::-1][:-1], 
                "".join(frag[::-1][:-1]), 
                hmdb_id, 
                hash_map.get("".join(frag[::-1][:-1]), "N/A"), 
                smile
            ])
        if not node.children:
            return
        for child in node.children:
            frag.append(child)
            self.dfs(frag, node.children[child], prefix, writer, hmdb_id, smile, hash_map)
            frag.pop()

# Initialize the Trie and other components
if __name__ == "__main__":
    start = time.time()

    trie = Trie()
    dataset = pd.read_csv("Final_MagnetDB.csv")
    hash_map = {}

    # Building the Trie
    for index, row in dataset.iterrows():
        frag = row['Fragment']
        id = row['Magnet Id']
        trie.insert((frag + "$")[::-1])
        hash_map[frag] = id

    end = time.time()
    print(f"Trie building time: {end - start} seconds")

    # Storing all fragments
    output_file = "output_fragments.csv"
    # Writing header
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Reversed_Prefix", "Fragment", "HMDB_ID", "Magnet_ID", "SMILES"])

    start = time.time()
    for index, row in dataset.iterrows():
        frag = (row['Fragment'] + "$")[::-1]
        hmdb_id = row['HMDB ID'] if 'HMDB ID' in row else "N/A"
        smile = row['SMILES'] if 'SMILES' in row else "N/A"
        trie.store_all_fragments(frag, smile, hmdb_id, output_file, hash_map)
    end = time.time()

    print(f"Fragment storage time: {end - start} seconds")



# Instructions:
# Save this script as trie_builder.py.
# Place your Final_MagnetDB.csv file in the same directory as the script.
# Run the script using python trie_builder.py.
# The output will be saved in output_fragments.csv.
