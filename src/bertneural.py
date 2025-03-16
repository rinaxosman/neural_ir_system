import json
import pandas as pd
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.evaluation import EvaluateRetrieval
import os
import pandas as pd

# preprocessed corpus
with open("data/scifact/preprocessed_corpus.json", "r") as f:
    corpus_data = json.load(f)
    
# test queries
test_queries = pd.read_csv("data/scifact/qrels/test.tsv", sep="\t", names=["query-id", "corpus-id", "score"], skiprows=1)


# inverted index
with open("data/scifact/inverted_index.json", "r") as f:
    inverted_index = json.load(f)

print(f"Loaded {len(corpus_data)} documents, {len(test_queries)} test queries.")

# BEIR Corpus Format
corpus = {}
for doc in corpus_data:
    doc_id = doc["doc_id"]
    tokens = " ".join(doc["tokens"])
    corpus[doc_id] = {"text": tokens}

# Converts queries to BEIR format
queries = {}
for _, row in test_queries.iterrows():
    queries[str(row["query-id"])] = corpus.get(str(row["corpus-id"]), {}).get("text", "")

# Load  BERT based Model (pre trained)
model = SentenceBERT("msmarco-distilbert-base-v3")
retriever = DenseRetrievalExactSearch(model, batch_size=16)

#retrieved_results = retriever.search(corpus, queries, score_function="dot", top_k=100)
retrieved_results = retriever.search(corpus, queries, score_function="cos_sim", top_k=100)


# Evaluate performance (MAP and P@10)
evaluator = EvaluateRetrieval(retriever)

qrels = {}
for _, row in test_queries.iterrows():
    query_id = str(row["query-id"])
    corpus_id = str(row["corpus-id"])
    score = int(row["score"])

    if query_id not in qrels:
        qrels[query_id] = {}
    qrels[query_id][corpus_id] = score

metrics = evaluator.evaluate(qrels=qrels, results=retrieved_results, k_values=[1, 3, 5, 10])

print("Evaluation Metrics:", metrics)

# Saves eesults
output_file = "Results.txt"


test_queries_path = "data/scifact/qrels/test.tsv"
df = pd.read_csv(test_queries_path, sep="\t", names=["query_id", "corpus_id", "score"])
query_to_corpus = {row["query_id"]: row["corpus_id"] for _, row in df.iterrows()}


with open(output_file, "w", encoding="utf-8") as f:
    for query_id, doc_scores in retrieved_results.items():
        corpus_id = query_to_corpus.get(query_id, query_id)  # Default to query_id if missing
        run_name = f"run_{corpus_id}"  # ✅ Use corpus_id for run name

        # Skip queries with no retrieved documents
        if not doc_scores:
            continue  

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]  # Top 100 docs
        
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n")


print(f"Results saved to {output_file}")