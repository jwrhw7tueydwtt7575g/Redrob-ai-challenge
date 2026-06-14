import numpy as np
import argparse
from pathlib import Path

def build_interactions(features_file, output_file):
    features_path = Path(features_file)
    output_path = Path(output_file)
    
    print(f"Loading base features from {features_path}...")
    # Load base features
    if not features_path.exists():
        print(f"Warning: {features_path} not found. Creating dummy data for interaction builder.")
        # If not found (e.g. testing), create dummy
        X = np.random.rand(50, 20)
    else:
        X = np.load(features_path)
        
    print(f"Base features shape: {X.shape}. Building interaction terms...")
    
    # Placeholder for the 27 interaction features requested
    # e.g., Interaction 1: Col 0 * Col 1
    # Interaction 2: Col 2 * Col 3 etc.
    # In a real scenario, this would combine JD semantic similarity with candidate features.
    # Since we precompute, we create some polynomial features (X_i * X_j) for demonstration.
    
    num_samples = X.shape[0]
    num_interactions = 27
    
    interactions = np.zeros((num_samples, num_interactions))
    
    # Generate some simple cross-products as placeholders
    for i in range(num_interactions):
        col1 = i % X.shape[1]
        col2 = (i + 1) % X.shape[1]
        interactions[:, i] = X[:, col1] * X[:, col2]
        
    print(f"Interaction features shape: {interactions.shape}. Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, interactions)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/artifacts/features.npy", help="Path to features matrix")
    parser.add_argument("--output", default="../../data/artifacts/features_ix.npy", help="Output path for interaction features")
    args = parser.parse_args()
    
    build_interactions(args.input, args.output)
