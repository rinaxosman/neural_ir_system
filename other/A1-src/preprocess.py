import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self, stopwords_file=None, use_stemming=True):
        self.use_stemming = use_stemming

        # Download necessary NLTK resources
        nltk.download("punkt")
        nltk.download("stopwords")

        # Load stopwords from NLTK or a custom file
        self.stopwords = set(stopwords.words("english"))
        if stopwords_file:
            with open(stopwords_file, "r", encoding="utf-8") as f:
                self.stopwords.update(f.read().splitlines())

        self.stemmer = PorterStemmer()
        self.vocabulary = set()  # Store unique words

    def preprocess_text(self, text):
        """Tokenizes, removes punctuation & stopwords, and applies stemming."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]  # Remove stopwords
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]  # Apply porterstemming
        
        self.vocabulary.update(tokens)  # Add tokens to vocabulary
        return tokens

    def preprocess_corpus(self, corpus_path=None, output_path=None):
        """Processes corpus and saves preprocessed tokens."""
        if corpus_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            corpus_path = os.path.join(base_dir, "..", "data", "scifact", "corpus.jsonl")

        if output_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(base_dir, "..", "data", "scifact", "preprocessed_corpus.json")

        corpus = []
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = str(doc.get("_id", "")).strip()  # Extracts correct document ID
                    text = doc.get("text", "").strip()  # Extracts correct text

                    if not text:  # Skip empty documents
                        continue  

                    tokens = self.preprocess_text(text)
                    if tokens:  # Ensure tokens are not empty
                        corpus.append({"doc_id": doc_id, "tokens": tokens})

        else:
            raise FileNotFoundError(f" Corpus file not found: {corpus_path}")

        # Save the preprocessed corpus
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2)

        print(f"✔ Preprocessed corpus saved to {output_path}")
        print(f"✔ Total processed documents: {len(corpus)}")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess_corpus()
    print("✅ Preprocessing completed successfully.")
