import os

results_title_path = "results/results_title.txt"
results_text_path = "results/results_text.txt"
output_path = "results/2_sample_queries.txt"

# Reads the first 10 results for the first 2 queries from ythe results file
def extract_top_10_results(file_path):
    extracted_results = {}
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return extracted_results

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue  
        query_id = parts[0]
        if query_id not in extracted_results:
            extracted_results[query_id] = []
        
        if len(extracted_results[query_id]) < 10:
            extracted_results[query_id].append(line.strip())

        if len(extracted_results) >= 2 and all(len(v) == 10 for v in extracted_results.values()):
            break

    return extracted_results

title_results = extract_top_10_results(results_title_path)
text_results = extract_top_10_results(results_text_path)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("## Top 10 Results for First 2 Queries (Title-Only Retrieval)\n\n")
    for query_id, results in title_results.items():
        f.write(f"### Query ID: {query_id}\n")
        f.write("\n".join(results) + "\n\n")

    f.write("## Top 10 Results for First 2 Queries (Title + Full Text Retrieval)\n\n")
    for query_id, results in text_results.items():
        f.write(f"### Query ID: {query_id}\n")
        f.write("\n".join(results) + "\n\n")

print(f"✅ Extracted top 10 results for the first 2 queries into {output_path}")
