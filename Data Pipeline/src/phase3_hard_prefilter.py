"""
Phase 3: Hard Pre-Filter + Retrieval (BM25 + Dense + RRF)
Extracted from improved_new_v2_fixed.ipynb Phase 3 cells.
Filters trap candidates, builds retrieval indices, and performs RRF fusion.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

log = logging.getLogger("redrob.phase3")

# ============================================================================
# 3a. Helper: Build retrieval text from candidate record
# ============================================================================
def build_retrieval_text(r: Dict) -> str:
    """Concatenate all textual fields from a candidate record for retrieval."""
    parts = [
        r.get("title", ""),
        r.get("headline", ""),
        r.get("summary", ""),
        " ".join(r.get("descs", [])),
        " ".join(s.get("name", "") for s in r.get("skills", [])),
        r.get("current_company", ""),
        r.get("current_industry", ""),
    ]
    return " ".join(p for p in parts if p)

# ============================================================================
# 3b. Hard pre-filter
# ============================================================================
def apply_hard_prefilter(records: List[Dict]) -> List[Dict]:
    """Remove honeypots, services-only, and trap titles BEFORE any scoring."""
    surviving_records = []
    filters = {"honeypot": 0, "services_only": 0, "title_score": 0}
    
    for r in records:
        if r["hp"] == 1:
            filters["honeypot"] += 1
            continue
        if r["services_only"] == 1:
            filters["services_only"] += 1
            continue
        if r["title_ai_score"] < -0.7:
            filters["title_score"] += 1
            continue
        surviving_records.append(r)
    
    log.info(f"  ✓ Pre-filter complete:")
    log.info(f"     Total:          {len(records)}")
    log.info(f"     Surviving:      {len(surviving_records)}")
    log.info(f"     Removed honeypots:   {filters['honeypot']}")
    log.info(f"     Removed services-only: {filters['services_only']}")
    log.info(f"     Removed trap titles: {filters['title_score']}")
    
    return surviving_records

# ============================================================================
# 3c. BM25 indexing
# ============================================================================
def build_bm25_index(surviving_records: List[Dict], tokenize_fn) -> Tuple[BM25Okapi, np.ndarray]:
    """Build BM25 index over surviving pool."""
    surviving_texts = [build_retrieval_text(r) for r in surviving_records]
    surviving_tokens = [tokenize_fn(text) for text in surviving_texts]
    bm25 = BM25Okapi(surviving_tokens)
    log.info(f"  ✓ BM25 index built: {len(surviving_tokens)} docs")
    return bm25, surviving_tokens

# ============================================================================
# 3d. Dense encoding
# ============================================================================
def encode_candidates_dense(surviving_records: List[Dict], jd_query_text: str, 
                            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                            batch_size: int = 32, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Encode all candidates and JD query with sentence-transformers."""
    log.info(f"  Loading encoder: {model_name} on {device} ...")
    encoder = SentenceTransformer(model_name, device=device)
    
    # Encode candidates in batches
    candidate_texts = [build_retrieval_text(r) for r in surviving_records]
    log.info(f"  Encoding {len(candidate_texts)} candidate texts in batches of {batch_size} ...")
    candidate_embeddings = encoder.encode(candidate_texts, batch_size=batch_size, 
                                         show_progress_bar=True, convert_to_tensor=False)
    
    # Encode JD query
    jd_embedding = encoder.encode(jd_query_text, convert_to_tensor=False)
    
    # Compute cosine similarity
    jd_embedding = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-8)
    candidate_embeddings = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    dense_scores = (candidate_embeddings @ jd_embedding).astype(np.float32)
    
    log.info(f"  ✓ Dense encoding complete: {len(dense_scores)} scores")
    
    return dense_scores, candidate_embeddings

# ============================================================================
# 3e. Score normalization
# ============================================================================
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s <= min_s:
        return np.ones_like(scores, dtype=np.float32) * 0.5
    normalized = (scores - min_s) / (max_s - min_s)
    return normalized.astype(np.float32)

# ============================================================================
# 3f. RRF (Reciprocal Rank Fusion)
# ============================================================================
def rrf_rank_fusion(bm25_scores: np.ndarray, dense_scores: np.ndarray, 
                   k: int = 60) -> np.ndarray:
    """Fuse BM25 and dense scores via RRF.
    rrf(d) = sum over retrieval systems S: 1 / (k + rank(d, S))
    """
    bm25_n = normalize_scores(bm25_scores)
    dense_n = normalize_scores(dense_scores)
    
    # Compute RRF
    bm25_ranks = np.argsort(-bm25_n) + 1  # +1 because RRF uses 1-based ranking
    dense_ranks = np.argsort(-dense_n) + 1
    
    rrf_scores = 1.0 / (k + bm25_ranks) + 1.0 / (k + dense_ranks)
    
    log.info(f"  ✓ RRF fusion complete: k={k}")
    log.info(f"     BM25 score range: [{bm25_scores.min():.3f}, {bm25_scores.max():.3f}]")
    log.info(f"     Dense score range: [{dense_scores.min():.3f}, {dense_scores.max():.3f}]")
    log.info(f"     RRF score range: [{rrf_scores.min():.6f}, {rrf_scores.max():.6f}]")
    
    return rrf_scores.astype(np.float32), bm25_n, dense_n

# ============================================================================
# Main Phase 3
# ============================================================================
def run_phase3(records: List[Dict], jd_query_text: str, jd_tokens: List[str],
               model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
               batch_size: int = 32, device: str = "cpu") -> Dict:
    """Execute Phase 3: hard pre-filter, BM25, dense encoding, RRF."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 3 — HARD PRE-FILTER + RETRIEVAL (BM25 + DENSE + RRF)")
    log.info("=" * 70)
    
    # Hard pre-filter
    surviving_records = apply_hard_prefilter(records)
    
    # BM25
    bm25, bm25_tokens = build_bm25_index(surviving_records, lambda t: t.split())
    bm25_scores = np.array(bm25.get_scores(jd_tokens), dtype=np.float32)
    
    # Dense encoding
    dense_scores, _ = encode_candidates_dense(surviving_records, jd_query_text, 
                                              model_name=model_name, batch_size=batch_size, device=device)
    
    # RRF fusion
    rrf_scores, bm25_n, dense_n = rrf_rank_fusion(bm25_scores, dense_scores, k=60)
    
    return {
        "surviving_records": surviving_records,
        "bm25_scores": bm25_scores,
        "bm25_n": bm25_n,
        "dense_scores": dense_scores,
        "dense_n": dense_n,
        "rrf_scores": rrf_scores,
        "rrf_n": rrf_scores,  # RRF scores are already normalized via fusion
    }
