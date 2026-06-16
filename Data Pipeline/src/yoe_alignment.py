"""
Years of Experience Weighting Module
Weight candidates by experience level relative to JD requirements
"""

import logging
import numpy as np
import re

log = logging.getLogger(__name__)


def extract_yoe_requirement(jd_text: str) -> tuple:
    """
    Extract years of experience requirement from JD.
    
    Returns:
        (min_years, preferred_years, extracted_text)
    """
    jd_lower = jd_text.lower()
    
    # Look for patterns like "3+ years", "5 to 8 years", "8-10 years"
    patterns = [
        r'(\d+)\s*(?:to|\-)\s*(\d+)\s*(?:years?|yrs?)',  # "3 to 5 years"
        r'(\d+)\+\s*(?:years?|yrs?)',  # "5+ years"
        r'(?:at least|minimum|required)\s+(\d+)\s*(?:years?|yrs?)',  # "at least 3 years"
    ]
    
    min_years = 0
    preferred_years = 0
    
    for pattern in patterns:
        match = re.search(pattern, jd_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                min_years = int(groups[0])
                preferred_years = int(groups[1])
            elif len(groups) == 1:
                min_years = int(groups[0])
                preferred_years = int(groups[0]) + 2  # Assume +2 years for "preferred"
            break
    
    log.info(f"JD YoE requirement: minimum={min_years}, preferred={preferred_years}")
    
    return min_years, preferred_years, ""


def compute_yoe_alignment_score(candidate_yoe: float, min_years: int, preferred_years: int) -> float:
    """
    Score candidate based on YoE alignment with JD requirements.
    
    Args:
        candidate_yoe: Years of experience of candidate
        min_years: Minimum years required by JD
        preferred_years: Preferred years by JD
    
    Returns:
        Score 0.0-1.0, peaks at preferred_years, drops at extremes
    """
    candidate_yoe = max(0, candidate_yoe)
    
    if min_years == 0 and preferred_years == 0:
        # No requirement specified, neutral score
        return 0.5
    
    # Below minimum: penalty
    if candidate_yoe < min_years:
        return 0.3 + (0.2 * candidate_yoe / max(min_years, 1))
    
    # At minimum: okay
    if candidate_yoe < preferred_years:
        range_size = preferred_years - min_years
        progress = (candidate_yoe - min_years) / max(range_size, 1)
        return 0.5 + (0.4 * progress)
    
    # At preferred: perfect
    if candidate_yoe == preferred_years:
        return 1.0
    
    # Beyond preferred: slight bonus (experience is good)
    excess = candidate_yoe - preferred_years
    bonus = min(0.05, excess * 0.01)  # Cap bonus at 5%
    return min(1.0, 1.0 + bonus)


def apply_yoe_boost(
    cross_encoder_scores: np.ndarray,
    candidate_yoe_values: np.ndarray,
    jd_min_years: int,
    jd_preferred_years: int,
    boost_factor: float = 0.15
) -> np.ndarray:
    """
    Apply YoE alignment as a boost to CE scores.
    
    Args:
        cross_encoder_scores: Array of CE scores
        candidate_yoe_values: Array of YoE for each candidate
        jd_min_years: Minimum years required by JD
        jd_preferred_years: Preferred years for JD
        boost_factor: Maximum boost (15% default)
    
    Returns:
        Boosted scores
    """
    yoe_scores = np.array([
        compute_yoe_alignment_score(yoe, jd_min_years, jd_preferred_years)
        for yoe in candidate_yoe_values
    ])
    
    boosted = cross_encoder_scores * (1 + boost_factor * (yoe_scores - 0.5) * 2)
    boosted = np.clip(boosted, 0, 1)
    
    log.info(f"Applied YoE boost (factor={boost_factor})")
    
    return boosted
