"""
Phase 5: XGBoost Learning-to-Rank Training
Extracted from improved_new_v2_fixed.ipynb Phase 5 cells.
Builds feature matrix, computes heuristic silver labels, trains XGBoost rank:ndcg.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
import xgboost as xgb

log = logging.getLogger("redrob.phase5")

# ============================================================================
# 5a. Build feature matrix
# ============================================================================
def build_feature_matrix(retrieval_records: List[Dict], bm25_n: np.ndarray,
                        dense_n: np.ndarray, rrf_n: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Build 28-feature matrix from candidates + retrieval signals."""
    
    features_list = []
    feature_names = [
        "title_ai_score", "td_match", "yoe_log", "yoe_fit", "ai_role_pct", 
        "ai_months_log", "total_months_log", "n_skills", "n_ai_skills", "n_ai_adv",
        "n_ai_endorse", "desc_ai_norm", "services_only",
        "loc_score", "rr", "rr_score", "la_days_capped", "la_score", 
        "np_score", "otw", "gh_score", "pc", "v_both", "oa_score", "sbr_log", "sap_log",
        "bm25_n", "dense_n", "rrf_n"
    ]
    
    for i, record in enumerate(retrieval_records):
        # Static features
        title_ai = (record.get("title_ai_score", 0) + 1.0) / 2.0  # Shift [-1,1] to [0,1]
        td_match = record.get("td_match", 0.5)
        yoe = record.get("yoe", 0.1)
        yoe_log = np.log1p(yoe)
        yoe_fit = record.get("yoe_fit", 0)
        ai_role_pct = record.get("ai_role_pct", 0)
        ai_months = record.get("ai_months", 0.1)
        ai_months_log = np.log1p(ai_months)
        total_months = record.get("total_months", 0.1)
        total_months_log = np.log1p(total_months)
        n_skills = record.get("n_skills", 0) / 20.0  # Normalize
        n_ai_skills = record.get("n_ai_skills", 0)
        n_ai_adv = record.get("n_ai_adv", 0)
        n_ai_endorse = record.get("n_ai_endorse", 0)
        desc_ai_norm = min(1.0, record.get("desc_ai_n", 0) / 30.0)
        services_only = record.get("services_only", 0)
        loc_score = record.get("loc_score", 0.5)
        rr = record.get("rr", 0.5)
        rr_score = record.get("rr_score", 0.5)
        la_days_capped = min(365, record.get("la_days", 365))
        la_score = record.get("la_score", 0.5)
        np_score = record.get("np_score", 0.5)
        otw = record.get("otw", 0)
        gh_score = record.get("gh_score", 0)
        pc = record.get("pc", 0.5)
        v_both = record.get("v_both", 0.5)
        oa_score = record.get("oa_score", 0.5)
        sbr = record.get("sbr", 0.1)
        sbr_log = np.log1p(sbr)
        sap = record.get("sap", 0.1)
        sap_log = np.log1p(sap)
        
        # Retrieval signals
        bm25 = bm25_n[i]
        dense = dense_n[i]
        rrf = rrf_n[i]
        
        features = [
            title_ai, td_match, yoe_log, yoe_fit, ai_role_pct,
            ai_months_log, total_months_log, n_skills, n_ai_skills, n_ai_adv,
            n_ai_endorse, desc_ai_norm, services_only,
            loc_score, rr, rr_score, la_days_capped, la_score,
            np_score, otw, gh_score, pc, v_both, oa_score, sbr_log, sap_log,
            bm25, dense, rrf
        ]
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    log.info(f"  ✓ Feature matrix built: {X.shape[0]} samples × {X.shape[1]} features")
    return X, feature_names

# ============================================================================
# 5b. Heuristic silver label scorer
# ============================================================================
def heuristic_score(record: Dict) -> float:
    """Weighted combination of 10 signals for silver label generation."""
    title_ai = (record.get("title_ai_score", 0) + 1.0) / 2.0  # [0, 1]
    ai_role_pct = min(1.0, record.get("ai_role_pct", 0))
    desc_ai_norm = min(1.0, record.get("desc_ai_n", 0) / 30.0)
    n_ai_skills = min(1.0, record.get("n_ai_skills", 0) / 15.0)
    td_match = record.get("td_match", 0.5)
    rr_score = record.get("rr_score", 0.5)
    la_score = record.get("la_score", 0.5)
    np_score = record.get("np_score", 0.5)
    otw = record.get("otw", 0.0)
    loc_score = record.get("loc_score", 0.5)
    services_only = record.get("services_only", 0)
    
    score = (
        0.30 * title_ai +
        0.18 * ai_role_pct +
        0.10 * desc_ai_norm +
        0.08 * n_ai_skills +
        0.06 * td_match +
        0.05 * rr_score +
        0.04 * la_score +
        0.04 * np_score +
        0.04 * otw +
        0.03 * loc_score -
        0.20 * services_only
    )
    return max(0.0, min(1.0, score))

# ============================================================================
# 5c. Convert heuristic scores to 5-level relevance labels
# ============================================================================
def to_relevance_labels(heuristic_scores: np.ndarray) -> np.ndarray:
    """Convert heuristic scores to 5-level (0-4) labels based on percentiles."""
    p99 = np.percentile(heuristic_scores, 99)
    p95 = np.percentile(heuristic_scores, 95)
    p80 = np.percentile(heuristic_scores, 80)
    p50 = np.percentile(heuristic_scores, 50)
    
    labels = np.zeros_like(heuristic_scores, dtype=np.int32)
    labels[heuristic_scores >= p99] = 4
    labels[(heuristic_scores >= p95) & (heuristic_scores < p99)] = 3
    labels[(heuristic_scores >= p80) & (heuristic_scores < p95)] = 2
    labels[(heuristic_scores >= p50) & (heuristic_scores < p80)] = 1
    labels[heuristic_scores < p50] = 0
    
    log.info(f"  Label distribution:")
    for i in range(5):
        count = np.sum(labels == i)
        log.info(f"    Level {i}: {count} samples ({count/len(labels):.1%})")
    
    return labels

# ============================================================================
# 5d. Train XGBoost
# ============================================================================
def train_xgboost_ltr(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> xgb.Booster:
    """Train XGBoost rank:ndcg model on feature matrix + labels."""
    
    log.info(f"  Training XGBoost LTR model...")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # Parameters for rank:ndcg
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "seed": 42,
    }
    
    # Train
    booster = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=50)
    
    log.info(f"  ✓ XGBoost training complete: 200 rounds, rank:ndcg objective")
    
    return booster

# ============================================================================
# Main Phase 5
# ============================================================================
def run_phase5(retrieval_records: List[Dict], retrieval_bm25_n: np.ndarray,
               retrieval_dense_n: np.ndarray, retrieval_rrf_n: np.ndarray) -> Dict:
    """Execute Phase 5: feature engineering, silver labels, XGBoost training."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 5 — XGBOOST LTR TRAINING")
    log.info("=" * 70)
    
    # Build feature matrix
    X, feature_names = build_feature_matrix(retrieval_records, retrieval_bm25_n,
                                           retrieval_dense_n, retrieval_rrf_n)
    
    # Compute heuristic scores and convert to labels
    log.info(f"  Computing heuristic silver labels...")
    heurs = np.asarray([heuristic_score(r) for r in retrieval_records], dtype=np.float32)
    y = to_relevance_labels(heurs)
    
    # Train XGBoost
    booster = train_xgboost_ltr(X, y, feature_names)
    
    # Predict on training set
    ltr_scores = booster.predict(xgb.DMatrix(X)).astype(np.float32)
    
    log.info(f"  ✓ LTR predictions: min={ltr_scores.min():.4f}, max={ltr_scores.max():.4f}")
    
    return {
        "booster": booster,
        "ltr_scores": ltr_scores,
        "feature_names": feature_names,
        "X": X,
        "y": y,
    }
