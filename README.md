# CSI4107 - Assignment 1: Information Retrieval System

## Important Notes on Code Structure
For simplicity, the three main steps of the system can be found in the following scripts:

- Step 1 (Preprocessing) → src/preprocess.py (Output: data/scifact/preprocessed_corpus.json)
- Step 2 (Indexing) → src/index.py (Output: data/scifact/inverted_index.json)
- Step 3 (Retrieval & Ranking) → src/retrieval.py (Output: results/Results.txt)

Other helper methods in the repository assist with retrieving insights, analyzing results, and evaluating performance.

## Team Members
- Rina Osman - 300222206
- Fatima Ghadbawi - 300301842
- Uvil Dahanayake - 300199138

## Contributions

Fatima Ghadbawi
- Step 1 (Preprocessing): Developed the text preprocessing pipeline, including tokenization, stopword removal, stemming, and text cleaning.
- Processed the SciFact dataset and structured it into preprocessed_corpus.jsonl for indexing.
- Extracted the total vocabulary size and generated a sample of 100 tokens for reporting.
- Explained the algorithms, data structures, and optimizations used in preprocessing, indexing, and retrieval.
- Report Sections: #4, #5

Rina Osman
- Step 2 (Indexing): Built the inverted index and structured it for efficient document retrieval.
- Step 3 (Retrieval & Ranking): Implemented cosine similarity to rank documents based on relevance to queries.
- Processed and formatted the system output into Results.txt in TREC format.
- Debugged for trec eval
- Compared retrieval effectiveness between title-only and full-text queries.
- Report Sections: #3, #6, #7, #8


Uvil Dahanayake
- Ran trec_eval to compute Mean Average Precision (MAP) and analyzed evaluation metrics.
- Assisted in comparing retrieval performance for different query modes.
- Contributed to writing and structuring key report sections.
- Wrote sections for Introduction, Functionality of the Program, Query Results & Discussion, and Mean Average Precision (MAP) Score.
- Report Sections: #1, #2

---

## 1. Introduction

- This project aims to build a simple information retrieval system that can search a given collection of documents, in our case a collection provided by Scifact, and return a list of related documents each with its own ranking and score.

---

## 2. Functionality of the Program

- The first stage is preprocessing in which our program goes through each document removing and formatting text in a way that makes it easier to search through and create a more meaningful index. We removed punctuation, numbers, and stopwords while also using the Porter stemmer to normalize similar words that have different inflexional endings before finally saving the processed corpus. 
- We then move on to indexing where we go through each of the preprocessed documents and add tokens to our inverted index with the document ID in which they appear. 
- With our inverted index made we can now retrieve related documents and assign them a score based on cosine similarity.

---

## 3. How to Run the Program

### 3.1 Install Dependencies
Ensure you have Python 3.8+ installed. Then, install all required dependencies:

    pip install -r requirements.txt

The required dependencies include:  
- `nltk` (for text preprocessing)  
- `numpy` (for numerical operations)  
- `scipy` (for cosine similarity calculations)  
- `sklearn` (for TF-IDF vectorization)  
- `jsonlines` (for handling `.jsonl` files)  
- `pandas` (for handling tabular data)  

### 3.2 Preprocess the Corpus
Preprocessing extracts, tokenizes, and cleans the text data, then saves it as preprocessed_corpus.jsonl.

    python src/preprocess.py

✅ Expected Output:
data/scifact/preprocessed_corpus.jsonl

### 3.3 Create the Inverted Index
Indexing builds the inverted index, which is necessary for retrieval.

    python src/index.py

✅ Expected Output:
data/scifact/inverted_index.json

### 3.4 Run Document Retrieval and Ranking
Retrieval runs the search system using two different query modes:

Title-only retrieval (results_title.txt)
Title + Full-text retrieval (results_text.txt)

    python src/retrieval.py

✅ Expected Output:
- results/results_title.txt
- results/results_text.txt

The retrieval script: 

- ✔ Loads the preprocessed corpus and inverted index
- ✔ Selects only odd-numbered queries from test.tsv
- ✔ Retrieves relevant documents using TF-IDF + Cosine Similarity
- ✔ Outputs results in TREC format

### 3.5 Evaluate Results using trec_eval
To compute Mean Average Precision (MAP) and other evaluation metrics, run:

    trec_eval data/scifact/qrels/test.tsv results/results_title.txt
    trec_eval data/scifact/qrels/test.tsv results/results_text.txt

This will output evaluation metrics such as:
- Mean Average Precision (MAP)
- Precision at different ranks
- Recall scores

---

## 4. Algorithms, Data Structures, and Optimizations

### **Preprocessing**
For preprocessing, we implemented a pipeline that cleans and prepares the text for indexing and retrieval. The steps include:

- **Tokenization**: We used `nltk.word_tokenize()` to split the text into individual words.
- **Stopword Removal**: We used NLTK’s predefined list of English stopwords and also allowed for a custom stopword list to further filter out common words that do not contribute to meaningful retrieval.
- **Lowercasing**: All text was converted to lowercase to ensure case-insensitive matching.
- **Punctuation and Number Removal**: Using regex (`re.sub(r'[^\w\s]', '', text)`) to eliminate punctuation and numbers, ensuring only meaningful words are retained.
- **Stemming**: We applied the **Porter Stemmer** to reduce words to their root forms (e.g., “running” → “run”). This helps normalize variations of the same word.
- **Vocabulary Storage**: We stored unique words in a set to ensure an efficient lookup.

**Optimizations:**
✔ **Preprocessing is applied in a single pass** to improve efficiency.  
✔ **Regex is used for fast text cleaning.**  
✔ **Set operations** are used for stopword removal to speed up lookups.  
✔ **Processes documents line-by-line** to handle large datasets efficiently.

---

### **Indexing**
To efficiently retrieve relevant documents, we constructed an **inverted index**, which maps terms to document IDs.

- **Data Structure Used**: We implemented a **dictionary-based inverted index** (`defaultdict(set)`) where:
  - **Key**: A word from the corpus (token)
  - **Value**: A **set** of document IDs that contain the word.

- **Indexing Algorithm**:
  1. **Read the preprocessed corpus** from `preprocessed_corpus.json`.
  2. **Extract tokens from each document** and associate them with their `doc_id`.
  3. **Update the inverted index** by adding document IDs to the posting list for each token.
  4. **Convert sets to lists** before saving the final index as `inverted_index.json` for efficient storage.

- **Storage Format (JSON example)**:
```json
{
    "brain": ["4983", "72159", "152245"],
    "develop": ["4983", "106031"],
    "function": ["4983", "116792"]
}
```

**Optimizations:**
✔ **Uses `defaultdict(set)` to prevent missing keys and speed up insertions.**  
✔ **Stores document IDs in sets first to remove duplicates before converting to lists.**  
✔ **Processes documents one-by-one to minimize memory usage.**

---

### **Retrieval & Ranking**
We implemented **TF-IDF weighting combined with cosine similarity** to rank documents based on their relevance to a given query.

- **TF-IDF Vectorization**:
  - We used `TfidfVectorizer()` from `sklearn` to convert both **documents and queries into numerical vectors**.
  - **TF-IDF (Term Frequency-Inverse Document Frequency)** helps assign importance to terms based on their frequency in a document vs. the entire corpus.

- **Cosine Similarity Calculation**:
  - Queries are **transformed into TF-IDF vectors**.
  - **Cosine similarity** is computed between each query vector and all document vectors.
  - Documents are **ranked in descending order** based on their similarity scores.

- **Ranking Algorithm**:
  1. **Convert the query to a TF-IDF vector** using the same vectorizer as the documents.
  2. **Compute cosine similarity** between the query and all document vectors.
  3. **Sort the results by similarity score** in descending order.
  4. **Return the top 100 documents** for evaluation.

- **Output Format**:
```plaintext
1 Q0 4983 1 0.8032 run_name
1 Q0 152245 2 0.7586 run_name
1 Q0 72159 3 0.6517 run_name
```

**Optimizations:**
✔ **Uses `TfidfVectorizer()` for efficient text vectorization.**  
✔ **Precomputes document TF-IDF vectors** to avoid redundant calculations.  
✔ **Uses sparse matrix operations** to reduce memory overhead.  
✔ **Sorts only the top `100` results** instead of computing all similarities.  


---

## 5. Vocabulary Statistics

Our final vocabulary consists of **30,980 unique tokens**, generated after tokenization, stopword removal, and stemming. Below is a sample of 100 randomly selected tokens, showcasing the diversity of terms extracted from the dataset, including domain specific scientific terms and stemmed words, which are crucial for effective document retrieval.

- **Vocabulary Size:** 30980
- **Sample of 100 Tokens:** ['fmd', 'alcoholinduc', 'pressureoverload', 'rela', 'oligo', 'stanc', 'buyin', 'stereoisom', 'intradur', 'caudat', 'crcscs', 'pudefici', 'tast', 'dyt', 'redifferenti', 'drugadr', 'receptorhsp', 'transduct', 'cultureadapt', 'vacuol', 'phosphotyrosin', 'sodium', 'fluorodeoxyglucos', 'quadruplex', 'tsce', 'leukemiainiti', 'hypercalcem', 'femal', 'czechoslovakia', 'lessen', 'statur', 'phenomena', 'lateact', 'auscultatori', 'hungri', 'pomb', 'disproport', 'globus', 'cucumerina', 'subscriptionbas', 'cilengitid', 'hivseroposit', 'disclos', 'function', 'autophagydefici', 'ltd', 'nhejdepend', 'tumordriven', 'substratum', 'substantia', 'offici', 'ethnicityspecif', 'plu', 'tsctic', 'intract', 'bordetella', 'estron', 'selfassess', 'tmposit', 'ppilik', 'gabpba', 'endosteallin', 'fifteen', 'core', 'nfκbdepend', 'learn', 'pacapspecif', 'contextur', 'reductionoxid', 'oliguria', 'cfainduc', 'vecadherin', 'hivneg', 'abstractmicrorna', 'eufa', 'oscillometr', 'anthropomorph', 'retroperiton', 'scbvkaiyuan', 'dextran', 'account', 'restitut', 'cancerrecruit', 'codomin', 'hcmvpermiss', 'japonica', 'northeastern', 'zfns', 'anyth', 'eprostanoid', 'blastema', 'anticitrullin', 'spore', 'blooddifferenti', 'lymphotoxinalphabeta', 'endothelialhaematopoiet', 'sitedepend', 'adher', 'insitu', 'fecund']

---

## 6. Query Results & Discussion

    python src/top_2_sample.py

### Top 10 Results for First 2 Queries (Title-Only Retrieval) vs (Title + Full Text Retrieval)
- We displayed a sample of the top 5 each for simplicity, full results are found at 2_sample_queries.txt

#### Title only
Query ID: 1
- 1 Q0 31715818 1 1.0000 run_31715818_title
- 1 Q0 5415832 2 0.2493 run_31715818_title
- 1 Q0 87430549 3 0.2121 run_31715818_title
- 1 Q0 29321530 4 0.2110 run_31715818_title
- 1 Q0 41782935 5 0.1914 run_31715818_title

Query ID: 3
- 3 Q0 14717500 1 1.0000 run_14717500_title
- 3 Q0 24530130 2 0.3664 run_14717500_title
- 3 Q0 25643818 3 0.3477 run_14717500_title
- 3 Q0 2739854 4 0.3379 run_14717500_title
- 3 Q0 13777706 5 0.3284 run_14717500_title

#### Title + Full Text
Query ID: 1
- 1 Q0 31715818 1 1.0000 run_31715818_text
- 1 Q0 502797 2 0.2538 run_31715818_text
- 1 Q0 1848452 3 0.2374 run_31715818_text
- 1 Q0 87430549 4 0.2306 run_31715818_text
- 1 Q0 8891333 5 0.2251 run_31715818_text

Query ID: 3
- 3 Q0 14717500 1 0.9885 run_14717500_text
- 3 Q0 23389795 2 0.3357 run_14717500_text
- 3 Q0 2739854 3 0.3051 run_14717500_text
- 3 Q0 2485101 4 0.2934 run_14717500_text
- 3 Q0 15155862 5 0.2849 run_14717500_text

**Discussion:** The results of the full-text queries always had higher scores at the same ranking however they would result in an almost completely different ranking with most entries in the top 10 from a title only query ranking significantly lower on a full-text query if the rank at all.

- The results, saved in 2_sample_queries.txt, were generated using top_2_sample.py. In both title-only and title + full-text retrieval, the top-ranked document is always an exact match with a score of 1.0000. This makes sense because the query directly matches an existing document in the dataset.
- However, the order of the remaining documents changes between the two methods. Title-only retrieval ranks documents based on how similar their titles are to the query, prioritizing those with closely matching titles. Title + full-text retrieval, on the other hand, considers the entire content, which can cause documents with relevant text (but different titles) to rank higher.
- Title-only retrieval is more precise because it retrieves documents with strong title similarity, but it may miss relevant content that doesn’t have a matching title. Title + full-text retrieval improves recall by finding documents where the query terms appear anywhere in the text, but this can also lead to some less relevant documents ranking higher.

---

## 7. Mean Average Precision (MAP) Score

The Mean Average Precision (MAP) score was computed using trec_eval for different retrieval approaches:

- Title-Only Retrieval → 0.9585
- Title + Full Text Retrieval → 0.9585
- Best Overall Run → 0.9707
These results indicate that our retrieval system is highly effective at ranking relevant documents. The best run achieves 0.9707, showing strong precision in retrieving relevant results.

Observations:
- Both title-only and title + full-text retrieval achieve nearly identical MAP scores (0.9585), suggesting that adding full text does not significantly impact ranking effectiveness in this dataset.
- The best run (0.9707) suggests some improvements were made, likely due to better ranking or query processing optimizations.
- The high scores overall indicate that the model retrieves highly relevant documents early in the ranking process.

✅ The MAP computation results can be found at: results/trec-map-result.png.

---

## 8. Conclusion
In this assignment, we built an **Information Retrieval (IR) system** using the **vector space model**, with **TF-IDF weighting and cosine similarity** to rank documents from the SciFact dataset. Our **preprocessing pipeline** helped clean and organize the text by applying **tokenization, stopword removal, and stemming**, making indexing and retrieval more efficient. The **inverted index** allowed for fast lookups, and our **retrieval module** showed how different query methods (title only vs. full text) affected results.  

From our evaluation, we found that full-text retrieval ranked more relevant documents higher, improving recall but sometimes lowering precision. The Mean Average Precision (MAP) scores confirmed this, showing a balance between precision and recall. While our system performed well, there is room for improvement, such as using BM25 for better ranking, query expansion to improve recall, and incorporating synonyms or domain-specific terms.

Overall, this project gave us a hands-on understanding of **search algorithms, ranking strategies, and the real-world challenges of information retrieval**, making it a great learning experience.
