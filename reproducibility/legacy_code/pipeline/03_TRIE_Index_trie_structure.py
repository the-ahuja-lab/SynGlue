class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.tag_ids = set()


class LeanTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, db_id):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.tag_ids.add(db_id)

    def __getstate__(self):
        return self.root

    def __setstate__(self, state):
        self.root = state