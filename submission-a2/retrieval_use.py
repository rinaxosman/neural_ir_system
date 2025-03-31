import os
import json
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# Load Universal Sentence Encoder (USE)
print("🔄 Loading Universal Sentence Encoder...")
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


corpus_path = "scifact/corpus.jsonl"
test_queries_path = "scifact/qrels/test.tsv"

results_dir = "m2-results"
os.makedirs(results_dir, exist_ok=True) 

results_title_path = os.path.join(results_dir, "results_title_use.txt")
results_text_path = os.path.join(results_dir, "results_text_use.txt")
results_both_path = os.path.join(results_dir, "results_both_use.txt")

os.makedirs("results", exist_ok=True)


def load_corpus(file_path):
    
    corpus_title, corpus_text, corpus_both = {}, {}, {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                doc_id = str(doc["_id"])
                title = doc["title"]
                text = doc["text"]
                
                corpus_title[doc_id] = title
                corpus_text[doc_id] = text
                corpus_both[doc_id] = f"{title} {text}"
            except json.JSONDecodeError as e:
                print(f"⚠ Skipping malformed JSON: {e}")

    return corpus_title, corpus_text, corpus_both


def load_queries(file_path):
    """Loads test queries and filters only the required ones."""
    df = pd.read_csv(file_path, sep="\t", names=["query_id", "corpus_id", "score"], skiprows=1)
    df["query_id"] = df["query_id"].astype(str)
    df["corpus_id"] = df["corpus_id"].astype(str)

    test_queries_title, test_queries_text, test_queries_both = {}, {}, {}

    for _, row in df.iterrows():
        query_id = row["query_id"]
        corpus_id = row["corpus_id"]

        if corpus_id in corpus_title:
            test_queries_title[query_id] = {"query": corpus_title[corpus_id], "run_name": f"run_{corpus_id}_title"}

        if corpus_id in corpus_text:
            test_queries_text[query_id] = {"query": corpus_text[corpus_id], "run_name": f"run_{corpus_id}_text"}

        if corpus_id in corpus_both:
            test_queries_both[query_id] = {"query": corpus_both[corpus_id], "run_name": f"run_{corpus_id}_both"}

    return test_queries_title, test_queries_text, test_queries_both


def encode_corpus(corpus_dict):
    """Encodes corpus documents using USE embeddings."""
    corpus_texts = list(corpus_dict.values())
    corpus_doc_ids = list(corpus_dict.keys())

    print(f"🔄 Encoding {len(corpus_texts)} documents...")
    corpus_embeddings = use_model(corpus_texts).numpy()

    return corpus_doc_ids, corpus_embeddings


def retrieve_and_rank_use(query, query_id, run_name, corpus_doc_ids, corpus_embeddings, top_k=100):
    """Retrieve top-K documents using USE embeddings."""
    query_embedding = use_model([query]).numpy()
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    ranked_docs = sorted(zip(corpus_doc_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}")

    return results


print("🔄 Loading dataset...")
corpus_title, corpus_text, corpus_both = load_corpus(corpus_path)
test_queries_title, test_queries_text, test_queries_both = load_queries(test_queries_path)


corpus_doc_ids_title, corpus_embeddings_title = encode_corpus(corpus_title)
corpus_doc_ids_text, corpus_embeddings_text = encode_corpus(corpus_text)
corpus_doc_ids_both, corpus_embeddings_both = encode_corpus(corpus_both)


all_results_title, all_results_text, all_results_both = [], [], []

for query_id, data in test_queries_title.items():
    all_results_title.extend(retrieve_and_rank_use(data["query"], query_id, data["run_name"], corpus_doc_ids_title, corpus_embeddings_title))

for query_id, data in test_queries_text.items():
    all_results_text.extend(retrieve_and_rank_use(data["query"], query_id, data["run_name"], corpus_doc_ids_text, corpus_embeddings_text))

for query_id, data in test_queries_both.items():
    all_results_both.extend(retrieve_and_rank_use(data["query"], query_id, data["run_name"], corpus_doc_ids_both, corpus_embeddings_both))

# Save results
with open(results_title_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_title) + "\n")

with open(results_text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_text) + "\n")

with open(results_both_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_both) + "\n")

print(f"✅ Results saved to:")
print(f"- {results_title_path}")
print(f"- {results_text_path}")
print(f"- {results_both_path}")
