import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def extract_features(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # We will read from the original candidates file (or sample) which has the structured redrob_signals
    # Assume the input here is the raw JSONL since we need the structured numeric fields,
    # or the enriched JSONL if we stored the raw JSON inside it. 
    # Let's assume input_file points to the original raw candidate JSONL.
    
    print(f"Loading raw candidate structured data from {input_path}...")
    candidate_ids = []
    features_list = []
    
    # List of expected numeric features
    feature_cols = [
        "profile_completeness_score", 
        "profile_views_received_30d",
        "applications_submitted_30d",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "github_activity_score",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
        "interview_completion_rate",
        "offer_acceptance_rate",
        "verified_email", # boolean to 0/1
        "verified_phone",
        "linkedin_connected",
        "open_to_work_flag",
        "willing_to_relocate",
        "salary_min", # Derived
        "salary_max"  # Derived
    ]
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # Check if JSON array or JSONL
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            candidates = json.load(f)
        else:
            candidates = [json.loads(line) for line in f if line.strip()]
            
        for record in candidates:
            candidate_ids.append(record['candidate_id'])
            signals = record.get('redrob_signals', {})
            
            # Extract basic features
            row = []
            for col in feature_cols[:18]:
                val = signals.get(col, 0)
                if isinstance(val, bool):
                    val = 1.0 if val else 0.0
                elif val is None:
                    val = 0.0
                row.append(float(val))
                
            # Extract salary
            salary = signals.get("expected_salary_range_inr_lpa", {})
            row.append(float(salary.get("min", 0.0)))
            row.append(float(salary.get("max", 0.0)))
            
            features_list.append(row)
            
    # Convert to numpy array
    X = np.array(features_list)
    
    # Simple imputation for -1 values (like offer_acceptance_rate or github_activity_score)
    # Replace -1 with column median
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = (col == -1.0)
        if mask.any():
            valid_vals = col[~mask]
            median_val = np.median(valid_vals) if len(valid_vals) > 0 else 0.0
            col[mask] = median_val
            
    print(f"Extracted features matrix of shape {X.shape}. Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, X)
    
    # Save column names and candidate ids
    meta_path = output_path.with_name("features_meta.npz")
    np.savez(meta_path, columns=feature_cols, candidate_ids=candidate_ids)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reading from raw candidates since it has the structured numeric data
    parser.add_argument("--input", default="../../data/raw/sample_candidates.json", help="Path to raw candidates JSON/JSONL")
    parser.add_argument("--output", default="../../data/artifacts/features.npy", help="Output path for numpy file")
    args = parser.parse_args()
    
    extract_features(args.input, args.output)
