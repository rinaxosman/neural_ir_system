import json
import os
from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def build_index(self, preprocessed_corpus_path):
        """Builds an inverted index from preprocessed corpus.
        \data\scifact\preprocessed_corpus.json"""
        if not os.path.exists(preprocessed_corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {preprocessed_corpus_path}")

        with open(preprocessed_corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        for doc in corpus:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]

            for token in tokens:
                self.index[token].add(doc_id)

        # Convert to lists
        for term in self.index:
            self.index[term] = list(self.index[term])

    def save_index(self, output_path):
        """Saves inverted index to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

        print(f"✔ Inverted index saved to {output_path}")

if __name__ == "__main__":
    preprocessed_corpus_path = "data\scifact\preprocessed_corpus.json"
    # output_path = "inverted_index.json"
    output_path = "data/scifact/inverted_index.json"

    indexer = InvertedIndex()
    indexer.build_index(preprocessed_corpus_path)
    indexer.save_index(output_path)
    print("✅ Inverted Index successfully built!")