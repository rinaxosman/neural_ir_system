# CSI4107 - Assignment 2: Neural Information Retrieval System

## Important Note on Code Structure
For simplicity, the system can be found in the following script:

- retrieval_bert.py

        - Output: results/results_both_bert.txt
        - Output: results/results_title_bert.txt
        - Output: results/results_text_bert.txt

Other helper methods in the repository assist with retrieving insights, analyzing results, and evaluating performance.

## Team Members
- Rina Osman - 300222206
- Fatima Ghadbawi - 300301842
- Uvil Dahanayake - 300199138

## Contributions

Fatima Ghadbawi

Rina Osman
- Developed Model 1: Sentence-BERT, which utilizes pre-trained BERT embeddings for document retrieval.
- Implemented the retrieval pipeline in bertneural.py, ensuring efficient document ranking based on semantic similarity.
    - Processed and evaluated retrieval results, storing them in:
    - results/results_both_bert.txt
    - results/results_title_bert.txt
    - results/results_text_bert.txt
- Designed and executed evaluation metrics (MAP, P@10) to compare BERT's performance with classical retrieval methods from A1.

Uvil Dahanayake

---

## 1. Introduction

- In this assignment, we implemented an improved version of the Information Retrieval (IR) system from Assignment 1, focusing on neural information retrieval methods. We explored transformer-based retrieval models such as BERT-based models and Sentence-BERT (SBERT) to enhance retrieval effectiveness. The objective was to obtain improved evaluation scores using advanced neural models compared to the classical TF-IDF and exact match retrieval methods from Assignment 1.

---
##


## 2. Functionality of the Program

- Description

---

## 3. How to Run the Program

### 3.1 Install Dependencies
Ensure you have Python 3.8+ installed. Then, install all required dependencies:

    pip install -r requirements.txt

The required dependencies include:  
- `nltk` (for text preprocessing)  
- `numpy` (for numerical operations)
- etc ...

### 3.2 Preprocess the Corpus
Since this system relies on neural information retrieval methods, preprocessing is not required, due to its understading of 

### 3.5 Evaluate Results


This will output evaluation metrics such as:
- Mean Average Precision (MAP)
- Precision at different ranks (1,10)
- Recall scores

---

## 4. Algorithms, Data Structures, and Optimizations





### **Retrieval & Ranking**
- Description


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

**Vocabulary Size:** 30980
- **Sample of 100 Tokens:** ['fmd', 'alcoholinduc', 'pressureoverload', 'rela', 'oligo', 'stanc', 'buyin', 'stereoisom', 'intradur', 'caudat', 'crcscs', 'pudefici', 'tast', 'dyt', 'redifferenti', 'drugadr', 'receptorhsp', 'transduct', 'cultureadapt', 'vacuol', 'phosphotyrosin', 'sodium', 'fluorodeoxyglucos', 'quadruplex', 'tsce', 'leukemiainiti', 'hypercalcem', 'femal', 'czechoslovakia', 'lessen', 'statur', 'phenomena', 'lateact', 'auscultatori', 'hungri', 'pomb', 'disproport', 'globus', 'cucumerina', 'subscriptionbas', 'cilengitid', 'hivseroposit', 'disclos', 'function', 'autophagydefici', 'ltd', 'nhejdepend', 'tumordriven', 'substratum', 'substantia', 'offici', 'ethnicityspecif', 'plu', 'tsctic', 'intract', 'bordetella', 'estron', 'selfassess', 'tmposit', 'ppilik', 'gabpba', 'endosteallin', 'fifteen', 'core', 'nfκbdepend', 'learn', 'pacapspecif', 'contextur', 'reductionoxid', 'oliguria', 'cfainduc', 'vecadherin', 'hivneg', 'abstractmicrorna', 'eufa', 'oscillometr', 'anthropomorph', 'retroperiton', 'scbvkaiyuan', 'dextran', 'account', 'restitut', 'cancerrecruit', 'codomin', 'hcmvpermiss', 'japonica', 'northeastern', 'zfns', 'anyth', 'eprostanoid', 'blastema', 'anticitrullin', 'spore', 'blooddifferenti', 'lymphotoxinalphabeta', 'endothelialhaematopoiet', 'sitedepend', 'adher', 'insitu', 'fecund']

---

## 6. Query Results & Discussion

    python src/top_2_sample.py

### Top 10 Results for First 2 Queries
- Results are found at results/first_2_sample_queries.txt

### Model 1: 

### Query ID: 1
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

### Query ID: 3
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

###  **Discussion:** 
- The Neural BERT-based model performs better than the Classic IR (TF-IDF) model in ranking relevant documents and assigning similarity scores. For both Query 1 and Query 3, the top-ranked document is the same in both models, meaning they both correctly identify the most relevant document. However, the Neural model assigns much higher similarity scores to the top-ranked results compared to TF-IDF. For example, in Query 1, the second-ranked document has a similarity score of 0.5980 in BERT, while it only has 0.2538 in TF-IDF. This pattern is also seen in Query 3, where the second document is ranked at 0.6373 in BERT versus 0.3357 in TF-IDF.

- This means the NN is more confident in its rankings, and it also brings more relevant documents higher in the list. In contrast, TF-IDF assigns lower scores and sometimes ranks less relevant documents higher because it only relies on exact keyword matches, while BERT understands the meaning of words and context. Overall, BERT produces stronger retrieval results, giving more meaningful rankings compared to the basic keyword-matching approach of TF-IDF.

- One of the key differences between the Neural BERT-based model and the Classic IR (TF-IDF) model is how they process text. BERT uses Byte Pair Encoding (BPE), which allows it to break words into smaller subword units and handle out-of-vocabulary words efficiently. This means BERT can understand complex words, variations, and even misspellings without needing pre-processing like stemming or tokenization. On the other hand, TF-IDF relies on simple tokenized words that are often stemmed (e.g., "running" becomes "run"), which removes important context. Since TF-IDF only looks at exact word matches, it struggles to understand the relationships between words or their deeper meanings.

### Model 2: 

---

## 7. Mean Average Precision (MAP) and P@10 Scores

### Assignment 1 scores: 
The Mean Average Precision (MAP) score was computed using trec_eval for different retrieval approaches:

- MAP → 0.9585
- P@10 → 

### Assignment 2 scores - Model 1 Bert : 

- MAP → 
- P@10 → 

### Assignment 2 scores - Model 2 (name) : 

- MAP → 
- P@10 → 

Observations:


---

## 8. Conclusion
- Description
