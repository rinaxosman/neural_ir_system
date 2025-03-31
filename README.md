# Neural IR System
## CSI 4107: Assignment 2

This repository implements and compares two neural retrieval models for the SciFact dataset:

- **Model 1**: Sentence-BERT using `msmarco-distilbert-base-v3`
- **Model 2**: Universal Sentence Encoder (USE)

---

## 👥 Team Contributions

The tasks for this project were divided clearly between both team members:

- **Rina Osman** (Student ID: 300222206) was responsible for **Model 1 (BERT)**, which included:
  - Writing the `retrieval_bert.py` script
  - Generating embeddings using Sentence-BERT
  - Performing retrieval and ranking
  - Saving and formatting results
  - Conducting evaluation using BEIR
  - Writing the analysis for Model 1

- **Fatima Ghadbawi** (Student ID: 300301842) was responsible for **Model 2 (Universal Sentence Encoder)**, which included:
  - Writing the `retrieval_use.py` script
  - Generating embeddings using TensorFlow Hub’s USE model
  - Performing retrieval and ranking
  - Saving and formatting results
  - Evaluation
  - Writing the analysis for Model 2 

- Third teammate from assignment 1 did not contribute, and was therefore not included. 

---

## 📁 Directory Structure

```
NEURAL_IR_SYSTEM/
│
├── data/
│   └── scifact/
│       ├── corpus.jsonl
│       ├── qrels/
│       │   ├── test.tsv
│       │   └── train.tsv
│       ├── inverted_index.json
│       ├── preprocessed_corpus.json (not required)
│       ├── queries.json             (not required)
│       └── StopWords.txt            (not required)
│
├── m1-results/     # Outputs for Model 1 (SBERT)
│   ├── first_2_sample_queries_bert.txt
│   ├── results_title_bert.txt
│   ├── results_text_bert.txt
│   └── results_both_bert.txt
│
├── m2-results/     # Outputs for Model 2 (USE)
│   ├── results_title_use.txt
│   ├── results_text_use.txt
│   └── results_both_use.txt
│
├── src/
│   ├── retrieval_bert.py       # Model 1
│   ├── retrieval_use.py        # Model 2
│   ├── top_2_sample.py
│
├── README.md
├── requirements.txt
├── submission-a1.zip (not required)
└── submission-a2.zip
```

---

## 🧠 Model 1: Sentence-BERT (`retrieval_bert.py`)

### Functionality:
- Loads and encodes the SciFact corpus using `msmarco-distilbert-base-v3` via `sentence-transformers`.
- Retrieves top 100 documents based on cosine similarity.
- Computes evaluation metrics like NDCG, MAP, Recall, and Precision using BEIR's `EvaluateRetrieval`.
- Saves results into `m1-results/`.

---

## 💻 How to Run the Program

### 1. Install Dependencies

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

Dependencies include:
- `sentence-transformers`
- `tensorflow` and `tensorflow-hub`
- `sklearn`, `numpy`, `pandas`
- `pytrec_eval`, `beir`, etc.

### 2. Run the Models

To run Model 1 (BERT):

```bash
python src/retrieval_bert.py
```

To run Model 2 (USE):

```bash
python src/retrieval_use.py
```

---

## Algorithms and Optimizations

### Sentence-BERT:
- **Encoder**: `msmarco-distilbert-base-v3`
- **Similarity**: Cosine similarity on dense embeddings.
- **Advantages**: State-of-the-art semantic search, trained on MS MARCO dataset for retrieval tasks.
- **Optimization**: Batch encoding to speed up corpus vectorization.

### Universal Sentence Encoder:
- **Encoder**: USE from TensorFlow Hub
- **Similarity**: Cosine similarity.
- **Advantages**: Lightweight, versatile for semantic similarity tasks.
- **Trade-offs**: Generally lower performance on domain-specific IR benchmarks compared to SBERT.

---

## 📊 Evaluation Metrics Summary (MAP and P@10 Only)

### Model 1 (BERT)
```
- Title + Text:
  - MAP@100: 0.61009
  - P@10: 0.066
```

### Model 2 (USE)
```
- Title + Text:
  - MAP: 0.9743
  - P@10: 0.1073
```

---

## 📌 Sample Results

### Model 1: First 10 Answers for Query 1 and Query 3 (first_2_sample_queries_bert.txt):

```
1 Q0 31715818 1 1.0000 run_31715818_both
1 Q0 502797 2 0.5980 run_31715818_both
1 Q0 6219790 3 0.5788 run_31715818_both
1 Q0 28386343 4 0.5683 run_31715818_both
1 Q0 10982689 5 0.5377 run_31715818_both
1 Q0 20620012 6 0.5375 run_31715818_both
1 Q0 12887068 7 0.5228 run_31715818_both
1 Q0 11172205 8 0.5122 run_31715818_both
1 Q0 17021845 9 0.5081 run_31715818_both
1 Q0 3981613 10 0.5060 run_31715818_both

3 Q0 14717500 1 1.0000 run_14717500_both
3 Q0 23389795 2 0.6373 run_14717500_both
3 Q0 4378885 3 0.6335 run_14717500_both
3 Q0 2485101 4 0.6205 run_14717500_both
3 Q0 35329820 5 0.6068 run_14717500_both
3 Q0 11532028 6 0.6014 run_14717500_both
3 Q0 3222187 7 0.5987 run_14717500_both
3 Q0 2739854 8 0.5958 run_14717500_both
3 Q0 13940200 9 0.5950 run_14717500_both
3 Q0 33387953 10 0.5913 run_14717500_both
```

### Model 2: First 10 Answers for Query 1 and Query 3 (first_2_sample_queries_use.txt):

```
1 Q0 31715818 1 1.0000 run_31715818_both
1 Q0 502797 2 0.7541 run_31715818_both
1 Q0 15728433 3 0.7406 run_31715818_both
1 Q0 20524091 4 0.6851 run_31715818_both
1 Q0 226488 5 0.6789 run_31715818_both
1 Q0 5567223 6 0.6691 run_31715818_both
1 Q0 21387297 7 0.6618 run_31715818_both
1 Q0 21676556 8 0.6581 run_31715818_both
1 Q0 8891333 9 0.6551 run_31715818_both
1 Q0 1848452 10 0.6548 run_31715818_both

3 Q0 14717500 1 1.0000 run_14717500_both
3 Q0 2485101 2 0.6501 run_14717500_both
3 Q0 23389795 3 0.6140 run_14717500_both
3 Q0 13956305 4 0.6105 run_14717500_both
3 Q0 22067786 5 0.6048 run_14717500_both
3 Q0 2739854 6 0.5972 run_14717500_both
3 Q0 35329820 7 0.5939 run_14717500_both
3 Q0 33669399 8 0.5798 run_14717500_both
3 Q0 20471181 9 0.5784 run_14717500_both
3 Q0 13717103 10 0.5751 run_14717500_both
```

---

## 🧠 Discussion

### Performance Comparison:

- **BERT outperforms USE** in nearly all metrics across title, text, and title+text retrieval.
- USE is faster to run but sacrifices precision and recall, especially in texts like SciFact.

### Observations:

## 🧠 Discussion

### Model 1 (BERT)
The Neural BERT-based model performs better than the Classic IR (TF-IDF) model in ranking relevant documents and assigning similarity scores. It is especially effective at understanding the context of words through subword tokenization (BPE), making it robust to variations and misspellings. For both Query 1 and 3, BERT placed the correct document at rank 1 and showed high semantic confidence in the rest of the top 10 rankings. 

However, while BERT understands context well, its scores for MAP and P@10 were noticeably lower than USE’s, indicating room for improvement in ranking across the full top 10.

### Model 2 (USE)
USE significantly outperformed TF-IDF and even BERT in terms of MAP and P@10. It is better at capturing the full semantic meaning of a sentence, which allowed it to return highly relevant documents even when the wording differed from the query. USE ranked the correct documents at the top, with stronger confidence and overall better precision across the top 10.

Its architecture is simpler and faster than BERT while still being extremely effective in semantic understanding — making it a great baseline for sentence-level similarity.

### BERT vs. USE Summary

| Metric     | BERT (Title+Text) | USE (Title+Text) |
|------------|-------------------|------------------|
| MAP@100    | 0.61009           | 0.9743           |
| P@10       | 0.066             | 0.1073           |

USE outperformed BERT on both metrics. While BERT is more powerful on large datasets and fine-tuning tasks, USE provides faster inference and betterssemantic matching for smaller retrieval problems like SciFact.

---

## 🎯 Submission Notes

- All result files are saved in the `m1-results/` and `m2-results/` folders.
- `submission-a2.zip` includes necessary scripts, results, requirements, and README for reproduction. Only the results for title+text are included for simplicity, available in the root of the folder. 