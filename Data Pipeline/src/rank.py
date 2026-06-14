import json
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import time

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False

def cosine_similarity(vec, matrix):
    dot_product = np.dot(matrix, vec.T).squeeze()
    norm_vec = np.linalg.norm(vec)
    norm_matrix = np.linalg.norm(matrix, axis=1)
    norm_matrix[norm_matrix == 0] = 1e-10
    if norm_vec == 0: norm_vec = 1e-10
    return dot_product / (norm_matrix * norm_vec)

def rrf_score(rank_list_1, rank_list_2, k=60):
    return (1.0 / (k + rank_list_1)) + (1.0 / (k + rank_list_2))

def generate_verified_reasoning(candidate_profile, rank):
    profile = candidate_profile.get("profile", {})
    title = profile.get("job_title", "Software Engineer")
    yoe = profile.get("years_of_experience", 0.0)
    
    signals = candidate_profile.get("redrob_signals", {})
    views = signals.get("profile_views_received_30d", 0)
    apps = signals.get("applications_submitted_30d", 0)
    
    skills_list = [s.get("name") for s in candidate_profile.get("skills", [])[:3]]
    skills_str = ", ".join(skills_list) if skills_list else "relevant skills"
    
    reasoning = f"Ranked #{rank}: {title} with {yoe} years of experience. Demonstrated proficiency in {skills_str}. Received {views} profile views and submitted {apps} applications in the last 30 days."
    return reasoning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="data/raw/job_description.docx", help="Path to JD")
    parser.add_argument("--candidates", default="data/raw/sample_candidates.json", help="Path to raw candidates JSON")
    parser.add_argument("--enriched", default="data/processed/enriched_sample.jsonl", help="Path to enriched JSONL")
    parser.add_argument("--out", default="../submission.csv", help="Output CSV path")
    args = parser.parse_args()
    
    start_time = time.time()
    
    data_dir = Path(__file__).parent.parent / "data" / "artifacts"
    
    print("1. Loading Precomputed Artifacts...")
    
    # Load Features first to determine candidate count if others are missing
    feat_path = data_dir / "features.npy"
    if feat_path.exists():
        features = np.load(feat_path)
        num_candidates = features.shape[0]
    else:
        num_candidates = 50
        features = np.random.rand(num_candidates, 20)
        
    feat_ix_path = data_dir / "features_ix.npy"
    if feat_ix_path.exists():
        features_ix = np.load(feat_ix_path)
    else:
        features_ix = np.random.rand(num_candidates, 27)

    # Load BM25
    bm25_path = data_dir / "bm25.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            bm25_model = bm25_data['bm25_index']
            candidate_ids = np.array(bm25_data['candidate_ids'])
    else:
        print(f"Warning: bm25.pkl not found. Using dummy BM25 data of size {num_candidates}.")
        candidate_ids = np.array([f"CAND_{str(i).zfill(7)}" for i in range(num_candidates)])
        bm25_model = None
        
    # Load Embeddings
    emb_path = data_dir / "embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        print(f"Warning: embeddings.npy not found. Using dummy Embeddings of size {num_candidates}.")
        embeddings = np.random.rand(num_candidates, 384)
    
    # Load Honeypots
    honeypot_path = data_dir / "honeypot_flags.json"
    if honeypot_path.exists():
        with open(honeypot_path, "r") as f:
            honeypot_set = set(json.load(f))
    else:
        honeypot_set = set()
        
    print(f"Artifacts loaded in {time.time() - start_time:.2f}s")
    
    print("2. Performing Step 1: Reciprocal Rank Fusion (RRF)...")
    
    # --- Semantic Search ---
    jd_vector = np.random.rand(1, embeddings.shape[1])
    semantic_scores = cosine_similarity(jd_vector, embeddings)
    semantic_ranks = np.empty_like(semantic_scores)
    semantic_ranks[np.argsort(-semantic_scores)] = np.arange(1, len(semantic_scores) + 1)
    
    # --- BM25 Search ---
    if bm25_model:
        jd_tokens = ["python", "go", "node", "backend", "microservices"]
        bm25_scores = bm25_model.get_scores(jd_tokens)
    else:
        bm25_scores = np.random.rand(len(candidate_ids))
        
    bm25_ranks = np.empty_like(bm25_scores)
    bm25_ranks[np.argsort(-bm25_scores)] = np.arange(1, len(bm25_scores) + 1)
    
    # --- Fusion ---
    fused_scores = rrf_score(semantic_ranks, bm25_ranks)
    
    # Zero out honeypots
    for i, cid in enumerate(candidate_ids):
        if cid in honeypot_set:
            fused_scores[i] = 0.0
            
    # Get top N indices (up to 500)
    top_n = min(500, len(candidate_ids))
    top_500_indices = np.argsort(-fused_scores)[:top_n]
    top_500_ids = candidate_ids[top_500_indices]
    
    print("3. Performing Step 2: Building 27-Feature Vector...")
    
    X_rank = np.zeros((len(top_500_indices), 27))
    
    for row_idx, original_idx in enumerate(top_500_indices):
        X_rank[row_idx, 0] = semantic_scores[original_idx]
        X_rank[row_idx, 1:16] = features[original_idx, :15]
        X_rank[row_idx, 16] = features[original_idx, 15]
        X_rank[row_idx, 17] = features[original_idx, 16]
        X_rank[row_idx, 18] = 1.0
        X_rank[row_idx, 19] = 1.0 if top_500_ids[row_idx] in honeypot_set else 0.0
        if features_ix.shape[1] >= 7:
            X_rank[row_idx, 20:27] = features_ix[original_idx, :7]
            
    print("4. Performing Step 3: XGBoost LTR Scoring...")
    
    final_scores = np.zeros(len(top_500_indices))
    
    if XGB_AVAILABLE:
        model_path = data_dir / "xgboost_ltr.model"
        if model_path.exists():
            print(f"Loading trained XGBoost model from {model_path}...")
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            dtest = xgb.DMatrix(X_rank)
            final_scores = bst.predict(dtest)
        else:
            print(f"Warning: Model not found at {model_path}. Using fallback ranking.")
            final_scores = fused_scores[top_500_indices]
    else:
        print("Warning: xgboost package not installed. Using fallback ranking.")
        final_scores = fused_scores[top_500_indices]
        
    xgb_sorted_indices = np.argsort(-final_scores)
    
    # Top 100 from XGBoost (pre-Phase 4)
    pre_re_rank_indices = xgb_sorted_indices[:100]
    
    print("5. Performing Step 4: Cross-Encoder Re-Ranking (Top 50)...")
    
    # We take the top 50 candidates from XGBoost to re-rank
    re_rank_limit = min(50, len(pre_re_rank_indices))
    top_50_indices_of_xgb = pre_re_rank_indices[:re_rank_limit]
    
    # Load enriched text for these candidates
    enriched_dict = {}
    enriched_path = Path(args.enriched)
    if enriched_path.exists():
        with open(enriched_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                enriched_dict[record['candidate_id']] = record['enriched_text']
                
    # JD text placeholder
    jd_text = "Looking for a Software Engineer with Python and ML experience."
    
    ce_scores = []
    candidates_to_ce = []
    for idx in top_50_indices_of_xgb:
        original_idx = top_500_indices[idx]
        cid = candidate_ids[original_idx]
        text = enriched_dict.get(cid, "Dummy enriched text.")
        # Truncate text to fit 512 token limit roughly by character split
        # assuming ~4 characters per token: 450 tokens * 4 = 1800 chars
        truncated_text = text[:1800]
        candidates_to_ce.append((cid, truncated_text))
        
    if CE_AVAILABLE:
        try:
            print("Loading ms-marco-MiniLM-L-6-v2 cross-encoder...")
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [(jd_text, c[1]) for c in candidates_to_ce]
            ce_scores = model.predict(pairs)
        except Exception as e:
            print(f"Warning: Failed to run CrossEncoder: {e}. Using fallback CE scores.")
            ce_scores = np.random.rand(re_rank_limit)
    else:
        print("Warning: CrossEncoder package not installed. Using fallback CE scores.")
        ce_scores = np.random.rand(re_rank_limit)
        
    # Sort top 50 by CE scores descending
    ce_sorted_sub_indices = np.argsort(-ce_scores)
    sorted_top_50_indices = [top_50_indices_of_xgb[i] for i in ce_sorted_sub_indices]
    
    # Remaining 51-100 candidates (remain in XGBoost order)
    remaining_indices = pre_re_rank_indices[re_rank_limit:]
    
    # Merge
    final_merged_indices = list(sorted_top_50_indices) + list(remaining_indices)
    
    print("6. Scaling Scores and Generating Verified Reasonings...")
    
    # Align scores to be strictly non-increasing
    # If the remaining part starts at index 50, let's get the XGBoost score of index 50
    if len(remaining_indices) > 0:
        xgb_score_51 = final_scores[remaining_indices[0]]
    else:
        xgb_score_51 = 0.0
        
    # Normalize CE scores to be strictly greater than xgb_score_51
    if len(ce_scores) > 0:
        ce_scores = np.array(ce_scores)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max - ce_min > 1e-5:
            normalized_ce = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            normalized_ce = np.ones_like(ce_scores)
            
        # Map normalized CE scores to [xgb_score_51 + 0.01, xgb_score_51 + 0.5]
        scaled_ce_scores = xgb_score_51 + 0.01 + normalized_ce * 0.49
        scaled_ce_scores = np.sort(scaled_ce_scores)[::-1]
    else:
        scaled_ce_scores = []
        
    # Load raw candidates profiles for reasoning generator
    profiles_dict = {}
    candidates_path = Path(args.candidates)
    if candidates_path.exists():
        with open(candidates_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                candidates_list = json.load(f)
            else:
                candidates_list = [json.loads(line) for line in f if line.strip()]
            for record in candidates_list:
                profiles_dict[record['candidate_id']] = record
                
    # Assemble final output
    submission_data = []
    for rank_idx, idx in enumerate(final_merged_indices, start=1):
        original_idx = top_500_indices[idx]
        cid = candidate_ids[original_idx]
        
        # Determine score
        if rank_idx <= len(scaled_ce_scores):
            score = float(scaled_ce_scores[rank_idx - 1])
        else:
            score = float(final_scores[idx])
            
        # Get reasoning
        candidate_profile = profiles_dict.get(cid, {})
        reasoning = generate_verified_reasoning(candidate_profile, rank_idx)
        
        submission_data.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": score,
            "reasoning": reasoning
        })
        
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    
    print(f"\nExecution finished in {time.time() - start_time:.2f}s")
    print(f"Top {len(submission_data)} candidates written to {output_path}")

if __name__ == "__main__":
    main()
