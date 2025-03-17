import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from beir.retrieval.evaluation import EvaluateRetrieval

corpus_path = "data/scifact/corpus.jsonl"
corpus_dict = {}

with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc["_id"])
        title = doc["title"]
        text = doc["text"]
        corpus_dict[doc_id] = f"{title} {text}" 

queries_path = "data/scifact/queries.jsonl"
queries_dict = {}

with open(queries_path, "r", encoding="utf-8") as f:
    for line in f:
        query_obj = json.loads(line)
        query_id = str(query_obj["_id"])
        queries_dict[query_id] = query_obj["text"]

test_queries_path = "data/scifact/qrels/test.tsv"
df = pd.read_csv(test_queries_path, sep="\t", names=["query_id", "corpus_id", "score"], skiprows=1)
df["query_id"] = df["query_id"].astype(str)
df["corpus_id"] = df["corpus_id"].astype(str)

query_to_corpus = {row["query_id"]: row["corpus_id"] for _, row in df.iterrows()}

# BERT Model
model = SentenceTransformer("msmarco-distilbert-base-v3")

corpus_texts = list(corpus_dict.values())
corpus_doc_ids = list(corpus_dict.keys())
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

query_texts = [queries_dict[qid] for qid in queries_dict if qid in df["query_id"].values]
query_ids = [qid for qid in queries_dict if qid in df["query_id"].values]
query_embeddings = model.encode(query_texts, convert_to_tensor=True)

retrieved_results = {}

for i, query_id in enumerate(query_ids):
    query_embedding = query_embeddings[i].unsqueeze(0)
    similarities = cosine_similarity(query_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy())[0]
    
    ranked_docs = sorted(zip(corpus_doc_ids, similarities), key=lambda x: x[1], reverse=True)[:100]

    retrieved_results[query_id] = {doc_id: float(score) for doc_id, score in ranked_docs}

output_file = "Results_v4.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for query_id, doc_scores in retrieved_results.items():
        corpus_id = query_to_corpus.get(query_id, query_id)  # Default to query_id if missing
        run_name = f"run_{corpus_id}"  # Use corpus_id for run name

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n")

print(f"Results saved to {output_file}")

qrels = {}
for _, row in df.iterrows():
    query_id = str(row["query_id"])
    corpus_id = str(row["corpus_id"])
    score = int(row["score"])

    if query_id not in qrels:
        qrels[query_id] = {}
    qrels[query_id][corpus_id] = score

for query_id in qrels.keys():
    retrieved_results.setdefault(query_id, {})

# Evaluation
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels=qrels, results=retrieved_results, k_values=[1, 3, 5, 10])

print("\n**Evaluation Metrics:**")
print(metrics)
