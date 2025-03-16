import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

corpus_path = "data/scifact/corpus.jsonl"

corpus_dict_title = {}
corpus_dict_text = {}
corpus_dict_both = {} 

with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc["_id"])
        title = doc["title"]
        full_text = doc["text"]

        corpus_dict_title[doc_id] = title
        corpus_dict_text[doc_id] = full_text
        corpus_dict_both[doc_id] = f"{title} {full_text}"

# test queries
test_queries_path = "data/scifact/qrels/test.tsv"
df = pd.read_csv(test_queries_path, sep="\t", names=["query_id", "corpus_id", "score"])

df["query_id"] = pd.to_numeric(df["query_id"], errors="coerce")
odd_queries = df[df["query_id"] % 2 == 1]
unique_odd_queries = odd_queries.groupby("query_id")["corpus_id"].first().reset_index()

test_queries_title = {}
test_queries_text = {}
test_queries_both = {}

for _, row in unique_odd_queries.iterrows():
    query_id = str(row["query_id"])
    corpus_id = str(row["corpus_id"])

    if corpus_id in corpus_dict_title:
        test_queries_title[query_id] = {"query": corpus_dict_title[corpus_id], "run_name": f"run_{corpus_id}_title"}

    if corpus_id in corpus_dict_text:
        test_queries_text[query_id] = {"query": corpus_dict_text[corpus_id], "run_name": f"run_{corpus_id}_text"}

    if corpus_id in corpus_dict_both:
        test_queries_both[query_id] = {"query": corpus_dict_both[corpus_id], "run_name": f"run_{corpus_id}_both"}

# pre-trained BERT model 
model = SentenceTransformer("msmarco-distilbert-base-v3")

# Corpus to BERT embeddings
def encode_corpus(corpus_dict):
    corpus_texts = list(corpus_dict.values())
    corpus_doc_ids = list(corpus_dict.keys())
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    return corpus_doc_ids, corpus_embeddings

corpus_doc_ids_title, corpus_embeddings_title = encode_corpus(corpus_dict_title)
corpus_doc_ids_text, corpus_embeddings_text = encode_corpus(corpus_dict_text)
corpus_doc_ids_both, corpus_embeddings_both = encode_corpus(corpus_dict_both)

def retrieve_and_rank_bert(query, query_id, run_name, corpus_doc_ids, corpus_embeddings, top_k=100):
    """Retrieve top-K documents using BERT embeddings"""
    query_embedding = model.encode([query], convert_to_tensor=True)  # Encode query
    similarities = cosine_similarity(query_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy())[0]  # cosine similarity instead of dot product

    ranked_docs = sorted(zip(corpus_doc_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}")

    return results

# results using BERT embeddings
all_results_title = []
all_results_text = []
all_results_both = []

for query_id, data in test_queries_title.items():
    all_results_title.extend(retrieve_and_rank_bert(data["query"], query_id, data["run_name"], corpus_doc_ids_title, corpus_embeddings_title))

for query_id, data in test_queries_text.items():
    all_results_text.extend(retrieve_and_rank_bert(data["query"], query_id, data["run_name"], corpus_doc_ids_text, corpus_embeddings_text))

for query_id, data in test_queries_both.items():
    all_results_both.extend(retrieve_and_rank_bert(data["query"], query_id, data["run_name"], corpus_doc_ids_both, corpus_embeddings_both))

# Save to results directory
results_title_path = "results/results_title_bert.txt"
results_text_path = "results/results_text_bert.txt"
results_both_path = "results/results_both_bert.txt"

os.makedirs(os.path.dirname(results_title_path), exist_ok=True)

with open(results_title_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_title) + "\n")

with open(results_text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_text) + "\n")

with open(results_both_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_both) + "\n")

print(f"Results saved to:")
print(f"- {results_title_path}")
print(f"- {results_text_path}")
print(f"- {results_both_path}")
