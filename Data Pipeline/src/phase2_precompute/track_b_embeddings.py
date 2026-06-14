import json
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_embeddings(input_file, output_file, batch_size=32):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    print(f"Loading enriched texts from {input_path}...")
    texts = []
    candidate_ids = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            candidate_ids.append(record['candidate_id'])
            texts.append(record['enriched_text'])
            
    print(f"Loaded {len(texts)} records. Initializing BGE-small-en-v1.5 model...")
    # BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    print(f"Encoding texts (this may take a while)...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Saving embeddings of shape {embeddings.shape} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save a dictionary mapping candidate IDs to their 384-dimensional vectors
    # We can save it as an npz file to include both arrays, or just the numpy array
    np.save(output_path, embeddings)
    
    # We also need to save the candidate IDs so we know which row is which
    ids_path = output_path.with_name("embedding_ids.npy")
    np.save(ids_path, np.array(candidate_ids))
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/processed/enriched_sample.jsonl", help="Path to enriched JSONL")
    parser.add_argument("--output", default="../../data/artifacts/embeddings.npy", help="Output path for numpy file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    args = parser.parse_args()
    
    build_embeddings(args.input, args.output, args.batch_size)
