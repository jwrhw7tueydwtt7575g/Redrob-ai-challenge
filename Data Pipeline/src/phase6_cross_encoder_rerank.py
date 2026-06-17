"""
Phase 6: Cross-Encoder Re-Ranking
Extracted from improved_new_v2_fixed.ipynb Phase 6 cells.
Re-ranks top 200 LTR candidates using a cross-encoder model.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

log = logging.getLogger("redrob.phase6")

# ============================================================================
# 6a. Build cross-encoder pairs
# ============================================================================
def build_ce_pairs(jd_text: str, retrieval_records: List[Dict]) -> List[Tuple[str, str]]:
    """Build (JD text, candidate text) pairs for cross-encoder."""
    pairs = []
    for record in retrieval_records:
        # Build candidate text from profile
        parts = [
            record.get("title", ""),
            record.get("headline", ""),
            record.get("summary", ""),
            " ".join(record.get("descs", [])),
            " ".join(s.get("name", "") for s in record.get("skills", [])),
        ]
        candidate_text = " ".join(p for p in parts if p).strip()
        
        # Truncate to avoid token limit (ms-marco-MiniLM has 512 token limit)
        jd_text_trunc = jd_text[:1800]
        candidate_text_trunc = candidate_text[:1800]
        
        pairs.append((jd_text_trunc, candidate_text_trunc))
    
    return pairs

# ============================================================================
# 6b. Score normalization and monotonicity enforcement
# ============================================================================
def enforce_strict_descending(scores: np.ndarray) -> np.ndarray:
    """Ensure scores are strictly descending by applying small epsilon offsets."""
    scores = scores.copy()
    for i in range(1, len(scores)):
        if scores[i] >= scores[i-1]:
            scores[i] = scores[i-1] - 1e-4
    return scores

def normalize_and_calibrate_scores(ce_scores: np.ndarray, target_min: float = 0.55,
                                   target_max: float = 0.985) -> np.ndarray:
    """Normalize CE scores to [target_min, target_max] and enforce strict descending."""
    # Min-max normalize to [0, 1]
    min_s = np.min(ce_scores)
    max_s = np.max(ce_scores)
    if max_s <= min_s:
        normalized = np.ones_like(ce_scores) * 0.5
    else:
        normalized = (ce_scores - min_s) / (max_s - min_s)
    
    # Scale to [target_min, target_max]
    scaled = target_min + normalized * (target_max - target_min)
    
    # Enforce strict descending
    scaled = enforce_strict_descending(scaled)
    
    # Round to 4 decimals
    scaled = np.round(scaled, 4)
    
    return scaled.astype(np.float32)

# ============================================================================
# Main Phase 6
# ============================================================================
def run_phase6(retrieval_records: List[Dict], ltr_scores: np.ndarray, jd_ce_text: str,
               ce_window: int = 200, top_k: int = 100,
               model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
               device: str = "cpu") -> Dict:
    """Execute Phase 6: cross-encoder re-ranking on top LTR candidates."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 6 — CROSS-ENCODER RE-RANKING")
    log.info("=" * 70)
    
    # Select top CE_WINDOW by LTR score
    ce_window = min(ce_window, len(retrieval_records))
    ce_local_idx = np.argsort(-ltr_scores)[:ce_window]
    ce_records = [retrieval_records[i] for i in ce_local_idx]
    
    log.info(f"  Selected top {len(ce_records)} by LTR score for cross-encoder")
    
    # Build pairs
    log.info(f"  Building {len(ce_records)} CE pairs...")
    ce_pairs = build_ce_pairs(jd_ce_text, ce_records)
    
    # Load cross-encoder
    log.info(f"  Loading cross-encoder: {model_name} on {device} ...")
    model = CrossEncoder(model_name, max_length=512, device=device)
    
    # Predict scores
    log.info(f"  Computing CE scores...")
    ce_scores_raw = model.predict(ce_pairs, batch_size=32, show_progress_bar=True)
    ce_scores_raw = ce_scores_raw.astype(np.float32)
    
    # Select top TOP_K by CE score
    top_ce_idx = np.argsort(-ce_scores_raw)[:top_k]
    final_order = ce_local_idx[top_ce_idx]
    ce_scores_selected = ce_scores_raw[top_ce_idx]
    
    # Normalize and calibrate scores
    scores_final = normalize_and_calibrate_scores(ce_scores_selected)
    
    # Get final records
    final_records = [retrieval_records[i] for i in final_order]
    
    log.info(f"  ✓ Cross-encoder re-ranking complete:")
    log.info(f"     CE window: {ce_window}")
    log.info(f"     Final selection: {len(final_records)}")
    log.info(f"     Final score range: [{scores_final.min():.4f}, {scores_final.max():.4f}]")
    
    return {
        "final_order": final_order,
        "scores_final": scores_final,
        "ce_records": final_records,
    }
