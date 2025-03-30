import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from beir.retrieval.evaluation import EvaluateRetrieval

corpus_path = "scifact/corpus.jsonl"

corpus_dict_title, corpus_dict_text, corpus_dict_both = {}, {}, {}

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
test_queries_path = "scifact/qrels/test.tsv"
df = pd.read_csv(test_queries_path, sep="\t", names=["query_id", "corpus_id", "score"], skiprows=1)

df["query_id"] = df["query_id"].astype(str)
df["corpus_id"] = df["corpus_id"].astype(str)

test_queries_title, test_queries_text, test_queries_both = {}, {}, {}

for _, row in df.iterrows():
    query_id = row["query_id"]
    corpus_id = row["corpus_id"]

    if corpus_id in corpus_dict_title:
        test_queries_title[query_id] = {"query": corpus_dict_title[corpus_id], "run_name": f"run_{corpus_id}_title"}
    if corpus_id in corpus_dict_text:
        test_queries_text[query_id] = {"query": corpus_dict_text[corpus_id], "run_name": f"run_{corpus_id}_text"}
    if corpus_id in corpus_dict_both:
        test_queries_both[query_id] = {"query": corpus_dict_both[corpus_id], "run_name": f"run_{corpus_id}_both"}

# BERT Model
model = SentenceTransformer("msmarco-distilbert-base-v3")

# converts corpus to BERT embedding
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
    query_embedding = model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy())[0]

    ranked_docs = sorted(zip(corpus_doc_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k]

    results = {str(doc_id): int(score) for doc_id, score in ranked_docs}
    formatted_results = [
        f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}"
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1)
    ]
    
    return results, formatted_results

# retrieve results using BERT embeddings
retrieved_results_title, retrieved_results_text, retrieved_results_both = {}, {}, {}
all_results_title, all_results_text, all_results_both = [], [], []

for query_id, data in test_queries_title.items():
    retrieved_docs, formatted_docs = retrieve_and_rank_bert(
        data["query"], query_id, data["run_name"], corpus_doc_ids_title, corpus_embeddings_title
    )
    retrieved_results_title[query_id] = retrieved_docs
    all_results_title.extend(formatted_docs)

for query_id, data in test_queries_text.items():
    retrieved_docs, formatted_docs = retrieve_and_rank_bert(
        data["query"], query_id, data["run_name"], corpus_doc_ids_text, corpus_embeddings_text
    )
    retrieved_results_text[query_id] = retrieved_docs
    all_results_text.extend(formatted_docs)

for query_id, data in test_queries_both.items():
    retrieved_docs, formatted_docs = retrieve_and_rank_bert(
        data["query"], query_id, data["run_name"], corpus_doc_ids_both, corpus_embeddings_both
    )
    retrieved_results_both[query_id] = retrieved_docs
    all_results_both.extend(formatted_docs)

# qrels
qrels = {}
for _, row in df.iterrows():
    query_id = str(row["query_id"])
    corpus_id = str(row["corpus_id"])
    try:
        score = int(row["score"])  # Ensure integer score
    except ValueError:
        print(f" Warning: Invalid score '{row['score']}' for query {query_id}, skipping...")
        continue

    if query_id not in qrels:
        qrels[query_id] = {}
    qrels[query_id][corpus_id] = score

for query_id in qrels.keys():
    retrieved_results_title.setdefault(query_id, {})
    retrieved_results_text.setdefault(query_id, {})
    retrieved_results_both.setdefault(query_id, {})


# validation function below was generated using AI 
def validate_results(results_dict):
    for query_id, docs in results_dict.items():
        if not isinstance(docs, dict):
            print(f"⚠ Warning: Query ID {query_id} results are not in expected format")
            return False
    return True

print("\n Validating results before evaluation...")
if not all([
    validate_results(retrieved_results_title),
    validate_results(retrieved_results_text),
    validate_results(retrieved_results_both)
]):
    print(" Error: Results are not in the expected dictionary format. Fix the retrieval process.")
    exit(1)

# evaluate results
evaluator = EvaluateRetrieval()
metrics_title = evaluator.evaluate(qrels=qrels, results=retrieved_results_title, k_values=[1, 10])
metrics_text = evaluator.evaluate(qrels=qrels, results=retrieved_results_text, k_values=[1, 10])
metrics_both = evaluator.evaluate(qrels=qrels, results=retrieved_results_both, k_values=[1, 10])

print("\n **Evaluation Results:**")
print("\n Title Only:")
print(metrics_title)
print("\n Text Only:")
print(metrics_text)
print("\n Title + Text:")
print(metrics_both)

os.makedirs("m1-results", exist_ok=True)

with open("m1-results/results_title_bert.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_title) + "\n")

with open("m1-results/results_text_bert.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_text) + "\n")

with open("m1-results/results_both_bert.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_results_both) + "\n")

print(f"\n Results saved to m1-results/")
