# Neural IR System
## CSI 4107: Assignment 2

This repository implements and compares 2 NN models for the SciFact dataset:

- **Model 1**: Sentence-BERT using `msmarco-distilbert-base-v3`
- **Model 2**: Universal Sentence Encoder (USE)

We also tested a **2nd variation of model 1** using `multi-qa-mpnet-base-dot-v1` in `retrieval_bert-v2.py`, which outperformed both in overall MAP and P@10.

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
  - Implemented extra variation of model 1, scoring a higher MAP and p@10. 

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
│
├── m1-1-results/     # Outputs for Model 1 variation 2
│   ├── first_2_sample_queries_bert2.txt
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
- Saves results into.

## 🧠 Model 1 Variation: `multi-qa-mpnet-base-dot-v1` (`retrieval_bert-v2.py`)
- A stronger sentence embedding model from `sentence-transformers`, optimized for retrieval with dot-product similarity.
- This variation significantly outperformed the previous two models.
- Results saved in `m2-2-results/`.

## 🧠 Model 2: Universal Sentence Encoder (`retrieval_use.py`)

### Functionality:
- Loads and encodes the SciFact corpus using TensorFlow Hub’s Universal Sentence Encoder (USE).
- Computes document similarity using cosine similarity.
- Retrieves and ranks the top 100 most similar documents for each query.
- Saves results into `m2-results/` in TREC format.
- Evaluation metrics were computed externally using `pytrec_eval`

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

To run Model 3 (variation of model 1):

```bash
python src/retrieval_bert-v2.py 
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

### Model 1 (BERT)
- The Neural BERT-based model performed better than the Classic IR (TF-IDF) method when it came to ranking relevant documents and assigning similarity scores. For both Query 1 and Query 3, the top-ranked document was the same in both models, which means they both managed to correctly identify the most relevant one. But BERT gave higher similarity scores to the top-ranked results than TF-IDF did. Like, in Query 1, the second-ranked document had a similarity score of 0.5980 in BERT, while it only got 0.2538 in TF-IDF. The same thing happened in Query 3 where the second document was scored 0.6373 by BERT and 0.3357 by TF-IDF.

- This shows that the neural model is more confident in its rankings, and it pushes the relevant documents higher up the list. TF iDF sometimes ranks less useful documents higher just cause it depends on matching the exact keywords. Meanwhile, BERT can understand the meaning of the query, not just the words in it. So overall, BERT gives better rankings and is more useful in retrieving what the user actually meant.

- 1 key difference between BERT and TF-IDF is how they handle the text. BERT uses something called Byte Pair Encoding (BPE), which breaks down words into smaller parts so it can understand rare words or even typos. That helps it a lot. TF-IDF on the other hand just chops words down through stemming or tokenization, like turning "running" into "run", and that removes important context. Since TF-IDF doesn’t get word relationships or deeper meaning, it ends up being less accurate.

- As an experiment, i also tried a different pretrained model: multi-qa-mpnet-base-dot-v1, which is actually optimized for retrieval tasks using dot-product instead of cosine similarity. Surprisingly, the results were slightly worse than with the original BERT model we used (msmarco-distilbert-base-v3). For title + text, the MAP dropped a little to 0.6071 and P@10 dropped to 0.06567, while our original BERT model got 0.61009 and 0.066 respectively. So even though multi-qa-mpnet was supposed to be more retrieval focused, it didn’t give us better scores on this specific dataset.

### Model 2 (USE)
- From our tests, the Universal Sentence Encoder (USE) model did even better than TF-IDF and in some ways even better than BERT. Just like the others, it ranked the top document correctly for both Query 1 and Query 3. But what stood out is how confident USE seemed in its rankings. For example, in Query 1, the second document got a 0.7541 score in USE, compared to only 0.5980 in BERT. In Query 3, USE gave the second one 0.6501 while BERT had 0.6373.

- What’s also interesting is that USE and BERT didnt always agree on the bottom documents. That shows how differently they interpret what’s relevant. USE looks at the entire meaning of a sentence, while BERT tries to juggle meaning and sentence structure. TF-IDF just matches words, so it ends up missing the point sometimes and ranks unrelated stuff higher than it should.

- A good thing about USE is that it doesnt need exact wording to understand the query. It picks up the meaning behind what you're saying. So if a document phrases something differently than the query, USE still finds it useful. Like in Query 1, some documents with different wording were still ranked high because USE got the meaning right.

- The MAP score of 0.9743 for USE when using both the title and full text proves that giving more context helps a lot. It's a big step up from title only, and it beats TF-IDF by a lot. Overall, USE is super strong when it comes to understanding full sentence meaning. It’s way more effective than traditional models and gives way better search results. Definitely a solid model for improving document retrieval.

### BERT vs. USE Summary

| Metric     | BERT (Title+Text) | USE (Title+Text) |
|------------|-------------------|------------------|
| MAP@100    | 0.61009           | 0.9743           |
| P@10       | 0.066             | 0.1073           |

USE outperformed BERT on both MAP and P@10. Even though BERT is stronger in some large scale or fine tuned settings, USE had faster results and better semantic matching on this task.

## 🔍 Summary

The 2nd variation of model 1 using MPNet embeddings (`multi-qa-mpnet-base-dot-v1`) achieved the **highest MAP@100 and P@10**, showing major improvement over the original SBERT and USE models. Despite being pretrained similarly to BERT, this model is optimized for dense retrieval tasks and showed better precision in ranking relevant documents.

---

## 🎯 Submission Notes

- All result files are saved in the `m1-results/` and `m2-results/` folders.
- `submission-a2.zip` includes necessary scripts, results, requirements, and README for reproduction. Only the results for title+text are included for simplicity, available in the root of the folder. 

---

## 📚 References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv preprint arXiv:1908.10084*.
  - [https://www.sbert.net](https://www.sbert.net)
- Cer, D., Yang, Y., Kong, S. Y., Hua, N., Limtiaco, N., John, R. S., ... & Kurzweil, R. (2018). Universal Sentence Encoder. *arXiv preprint arXiv:1803.11175*.
  - [https://tfhub.dev/google/universal-sentence-encoder/4](https://tfhub.dev/google/universal-sentence-encoder/4)
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
  - [https://github.com/UKPLab/beir](https://github.com/UKPLab/beir)
- TREC Eval Metrics: Precision@K and MAP Calculation in Python.
  - [https://github.com/cvangysel/pytrec_eval](https://github.com/cvangysel/pytrec_eval)
- ChatGPT by OpenAI — Used to help generate and explainthe evaluation functions, especially for formatting qrels and retrieved results for `pytrec_eval`.
  - [https://openai.com/chatgpt](https://openai.com/chatgpt)
- Hugging Face - [`multi-qa-mpnet-base-dot-v1`](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)