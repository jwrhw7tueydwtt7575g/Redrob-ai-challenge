import json
import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

def build_bm25_index(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    print(f"Loading enriched texts from {input_path}...")
    corpus = []
    candidate_ids = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            candidate_ids.append(record['candidate_id'])
            # Tokenize by simple whitespace for BM25
            text = record['enriched_text'].lower()
            corpus.append(text.split())
            
    print(f"Loaded {len(corpus)} records. Building BM25 index...")
    bm25 = BM25Okapi(corpus)
    
    # Save both the index and the mapping of index to candidate_id
    output_data = {
        'bm25_index': bm25,
        'candidate_ids': candidate_ids
    }
    
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/processed/enriched_sample.jsonl", help="Path to enriched JSONL")
    parser.add_argument("--output", default="../../data/artifacts/bm25.pkl", help="Output path for pickle file")
    args = parser.parse_args()
    
    build_bm25_index(args.input, args.output)
