"""
Enhanced Ranking Module
Incorporates response rate weighting, skill matching, and response rate signal
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


def apply_response_rate_boost(cross_encoder_scores: np.ndarray, response_rates: np.ndarray, boost_factor: float = 0.2) -> np.ndarray:
    """
    Apply response rate as a boost multiplier to cross-encoder scores.
    Higher response rate = stronger boost (up to 20% increase).
    
    Args:
        cross_encoder_scores: Array of CE scores (typically 0.0-1.0)
        response_rates: Array of response rates (0.0-1.0)
        boost_factor: Maximum boost percentage (default 0.2 = 20%)
    
    Returns:
        Boosted scores maintaining [0, 1] range
    """
    # Ensure response rates are in [0, 1]
    response_rates_normalized = np.clip(response_rates, 0, 1)
    
    # Apply boost: score * (1 + boost_factor * response_rate)
    # Example: score=0.75, response_rate=0.8 -> 0.75 * (1 + 0.2*0.8) = 0.75 * 1.16 = 0.87
    boosted_scores = cross_encoder_scores * (1 + boost_factor * response_rates_normalized)
    
    # Clip to [0, 1] to avoid scores > 1.0
    boosted_scores = np.clip(boosted_scores, 0, 1)
    
    log.info(f"Applied response rate boost (factor={boost_factor})")
    log.debug(f"Score range before boost: [{cross_encoder_scores.min():.4f}, {cross_encoder_scores.max():.4f}]")
    log.debug(f"Score range after boost: [{boosted_scores.min():.4f}, {boosted_scores.max():.4f}]")
    
    return boosted_scores


def apply_skill_match_boost(cross_encoder_scores: np.ndarray, skill_match_scores: np.ndarray, boost_factor: float = 0.15) -> np.ndarray:
    """
    Apply skill match quality as a boost to cross-encoder scores.
    Candidates with skills matching JD requirements get a boost.
    
    Args:
        cross_encoder_scores: Array of CE scores
        skill_match_scores: Array of skill match scores (0.0-1.0)
        boost_factor: Maximum boost percentage (default 0.15 = 15%)
    
    Returns:
        Skill-boosted scores
    """
    skill_match_normalized = np.clip(skill_match_scores, 0, 1)
    boosted_scores = cross_encoder_scores * (1 + boost_factor * skill_match_normalized)
    boosted_scores = np.clip(boosted_scores, 0, 1)
    
    log.info(f"Applied skill match boost (factor={boost_factor})")
    
    return boosted_scores


def normalize_scores_no_recompression(ce_scores: np.ndarray) -> np.ndarray:
    """
    Normalize CE scores without re-compression.
    Cross-encoder scores are already well-distributed [0, 1], so just clip.
    
    Args:
        ce_scores: Raw cross-encoder scores
    
    Returns:
        Clipped scores in [0, 1]
    """
    return np.clip(ce_scores, 0, 1)


def enforce_strict_descending_order(scores: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Enforce strictly descending score order.
    Rounds first, then applies epsilon to break ties.
    
    Args:
        scores: Array of scores
        epsilon: Small value to subtract for tied scores
    
    Returns:
        Scores with strict descending order
    """
    scores_final = np.round(scores, 4)
    
    for i in range(1, len(scores_final)):
        if scores_final[i] >= scores_final[i-1]:
            scores_final[i] = np.round(scores_final[i-1] - epsilon, 4)
    
    # Verify
    assert all(scores_final[i] > scores_final[i+1] for i in range(len(scores_final)-1)), \
        "Scores not strictly descending after epsilon fix!"
    
    log.info(f"✓ Scores enforced to strict descending order")
    
    return scores_final


def compute_blended_final_score(
    ce_score: float,
    response_rate_score: float = 0.0,
    skill_match_score: float = 0.5,
    ce_weight: float = 0.5,
    response_weight: float = 0.2,
    skill_weight: float = 0.3
) -> float:
    """
    Compute final blended score from multiple signals.
    
    Args:
        ce_score: Cross-encoder score (0.0-1.0)
        response_rate_score: Response rate signal (0.0-1.0)
        skill_match_score: Skill matching quality (0.0-1.0)
        ce_weight: Weight for CE score
        response_weight: Weight for response rate
        skill_weight: Weight for skill matching
    
    Returns:
        Blended final score (0.0-1.0)
    """
    # Normalize weights
    total_weight = ce_weight + response_weight + skill_weight
    ce_w = ce_weight / total_weight
    resp_w = response_weight / total_weight
    skill_w = skill_weight / total_weight
    
    blended = (ce_w * ce_score + 
               resp_w * response_rate_score + 
               skill_w * skill_match_score)
    
    return np.clip(blended, 0, 1)
