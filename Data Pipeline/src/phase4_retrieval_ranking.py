"""
Phase 4: Retrieval Ranking — Select top candidates by RRF score
Extracted from improved_new_v2_fixed.ipynb Phase 4 cells.
Creates retrieval_records and associated score arrays for Phase 5 LTR training.
"""

import logging
from typing import List, Dict

import numpy as np

log = logging.getLogger("redrob.phase4")

# ============================================================================
# 4a. Select top candidates by RRF score
# ============================================================================
def run_phase4(surviving_records: List[Dict], bm25_scores: np.ndarray, bm25_n: np.ndarray,
               dense_scores: np.ndarray, dense_n: np.ndarray, rrf_scores: np.ndarray,
               top_k_retrieval: int = 2000) -> Dict:
    """Execute Phase 4: select top 2000 by RRF score, create retrieval pool."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 4 — RETRIEVAL RANKING")
    log.info("=" * 70)
    
    # Sort by RRF score, take top K
    top_retrieval_idx = np.argsort(-rrf_scores)[:top_k_retrieval]
    
    # Build retrieval pool with metadata
    retrieval_pool = []
    for global_idx, local_idx in enumerate(top_retrieval_idx):
        r = surviving_records[local_idx]
        retrieval_pool.append({
            "local_idx": local_idx,
            "record": r,
            "rrf_score": float(rrf_scores[local_idx]),
            "rrf_norm": float(rrf_scores[local_idx]),  # RRF is already normalized
            "bm25_score": float(bm25_scores[local_idx]),
            "bm25_norm": float(bm25_n[local_idx]),
            "dense_score": float(dense_scores[local_idx]),
            "dense_norm": float(dense_n[local_idx]),
        })
    
    # Extract arrays for Phase 5
    retrieval_records = [p["record"] for p in retrieval_pool]
    retrieval_rrf_n = np.array([p["rrf_norm"] for p in retrieval_pool], dtype=np.float32)
    retrieval_bm25_n = np.array([p["bm25_norm"] for p in retrieval_pool], dtype=np.float32)
    retrieval_dense_n = np.array([p["dense_norm"] for p in retrieval_pool], dtype=np.float32)
    
    # Log statistics
    log.info(f"  ✓ Retrieval ranking complete:")
    log.info(f"     Selected:       {len(retrieval_records)} / {len(surviving_records)}")
    log.info(f"     RRF score percentiles:")
    for p in [50, 90, 95, 99]:
        val = np.percentile(rrf_scores[top_retrieval_idx], p)
        log.info(f"       p{p:2d}: {val:.6f}")
    
    return {
        "retrieval_records": retrieval_records,
        "retrieval_rrf_n": retrieval_rrf_n,
        "retrieval_bm25_n": retrieval_bm25_n,
        "retrieval_dense_n": retrieval_dense_n,
        "retrieval_pool_info": retrieval_pool,  # For debugging
    }
