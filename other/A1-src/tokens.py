import json
import random
import os

corpus_path = "data/scifact/preprocessed_corpus.json"
output_file = "results/tokens.txt"

def count_vocab(corpus_path):
    unique_tokens = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f) 

    for doc in corpus:
        tokens = doc.get("tokens", [])
        unique_tokens.update(tokens) 
    vocab_list = sorted(list(unique_tokens))
    vocab_size = len(vocab_list)

    # random sample of 100 tokens
    sample_tokens = random.sample(vocab_list, min(100, vocab_size))

    return vocab_size, sample_tokens

vocab_size, sample_tokens = count_vocab(corpus_path)

def save_to_file(vocab_size, sample_tokens, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total vocabulary size: {vocab_size}\n")
        f.write(f"Sample of 100 tokens: {sample_tokens}\n")

vocab_size, sample_tokens = count_vocab(corpus_path)
save_to_file(vocab_size, sample_tokens, output_file)

print(f"Results saved to {output_file}")