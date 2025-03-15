import json
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Preprocess text (lowercase, remove special characters, tokenize, remove stopwords, stem)."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

test_queries_path = "data/scifact/qrels/test.tsv"
corpus_path = "data/scifact/corpus.jsonl"

# corpus.jsonl
corpus_dict_title = {} # Stores only title
corpus_dict_text = {} # Stores title, full text

with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip() 
        if not line:  # Skip empty lines
            continue
        try:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            title = preprocess_text(doc["title"])
            full_text = preprocess_text(doc["text"])
            
            corpus_dict_title[doc_id] = title  # Store only title
            corpus_dict_text[doc_id] = f"{title} {full_text}"  # Store title and full text
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {line[:100]}...")
            continue


df = pd.read_csv(test_queries_path, sep="\t", names=["query_id", "corpus_id", "score"])
df["query_id"] = pd.to_numeric(df["query_id"], errors='coerce')

# Filter only odd-numbered queries
odd_queries = df[df["query_id"] % 2 == 1]

unique_odd_queries = odd_queries.groupby("query_id")["corpus_id"].first().reset_index()

test_queries_title = {}
test_queries_text = {}

for _, row in unique_odd_queries.iterrows():
    query_id = int(row["query_id"])
    corpus_id = str(row["corpus_id"]) 
    
    if corpus_id in corpus_dict_title:
        test_queries_title[query_id] = {
            "query": corpus_dict_title[corpus_id],
            "run_name": f"run_{corpus_id}_title"
        }
    
    if corpus_id in corpus_dict_text:
        test_queries_text[query_id] = {
            "query": corpus_dict_text[corpus_id],
            "run_name": f"run_{corpus_id}_text"
        }

# Initialize TF-IDF Vectorizers
vectorizer_title = TfidfVectorizer()
vectorizer_text = TfidfVectorizer()

# Transform corpus into TF-IDF vectors
doc_vectors_title = vectorizer_title.fit_transform(list(corpus_dict_title.values()))
doc_vectors_text = vectorizer_text.fit_transform(list(corpus_dict_text.values()))

def retrieve_and_rank(query, query_id, run_name, vectorizer, doc_vectors, doc_ids, top_k=100):
    """Retrieve and rank documents based on cosine similarity."""
    query = preprocess_text(query)  # Preprocess query
    query_vector = vectorizer.transform([query])  # Convert to TF-IDF vector
    similarities = cosine_similarity(query_vector, doc_vectors)[0]  # Compute cosine similarity

    # Rank documents by similarity score
    ranked_docs = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k]

    # Formats results
    results = []
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}")
    
    return results

# Rretrieval for title only
all_results_title = []
for query_id, data in test_queries_title.items():
    all_results_title.extend(retrieve_and_rank(data["query"], query_id, data["run_name"],vectorizer_title, doc_vectors_title, list(corpus_dict_title.keys())))

# Rfor title and full text
all_results_text = []
for query_id, data in test_queries_text.items():
    all_results_text.extend(retrieve_and_rank(data["query"], query_id, data["run_name"],vectorizer_text, doc_vectors_text, list(corpus_dict_text.keys())))

results_title_path = "results/results_title.txt"
results_text_path = "results/results_text.txt"

with open(results_title_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_title) + "\n")

with open(results_text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_text) + "\n")

print(f"✅ Results saved to {results_title_path} and {results_text_path}")
