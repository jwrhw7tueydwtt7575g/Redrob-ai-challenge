"""
Education Level & Achievement Signals Module
Extract and score education level, certifications, and quantified achievements
"""

import logging
import re
import numpy as np

log = logging.getLogger(__name__)

# Education level mapping
EDUCATION_LEVELS = {
    'phd': {'score': 1.0, 'keywords': ['phd', 'doctorate', 'dr.', 'ph.d']},
    'masters': {'score': 0.85, 'keywords': ['master', 'ms', 'mtech', 'mba', 'm.tech', 'm.s']},
    'bachelors': {'score': 0.70, 'keywords': ['bachelor', 'bs', 'btech', 'b.tech', 'b.s', 'b.a']},
    'diploma': {'score': 0.50, 'keywords': ['diploma', 'associate']},
    'bootcamp': {'score': 0.60, 'keywords': ['bootcamp', 'certification course']},
}

# Prestige universities
PRESTIGE_UNIVERSITIES = {
    'top_tier': {
        'score': 1.0,
        'keywords': ['mit', 'stanford', 'harvard', 'caltech', 'carnegie', 'berkley', 'berkeley', 'princeton']
    },
    'tier1': {
        'score': 0.85,
        'keywords': ['yale', 'columbia', 'upenn', 'penn', 'cornell', 'northwestern', 'duke', 'michigan', 'cmu']
    },
    'tier2': {
        'score': 0.70,
        'keywords': ['ut austin', 'georgia tech', 'illinois', 'wisconsin', 'ucla', 'penn state']
    },
}

# Achievement signal keywords
ACHIEVEMENT_KEYWORDS = {
    'quantified': ['built', 'developed', 'shipped', 'deployed', 'scaled', 'optimized', 'improved', 'reduced', 'increased'],
    'metrics': ['%', 'x', 'million', 'billion', 'thousands', 'faster', 'latency', 'throughput'],
    'leadership': ['led', 'managed', 'founded', 'directed', 'headed', 'pioneered'],
    'innovation': ['patent', 'designed', 'architected', 'framework', 'system'],
}


def extract_education_level(education_text: str) -> tuple:
    """
    Extract education level and institution from text.
    
    Returns:
        (education_level_score, institution_prestige_score)
    """
    if not education_text:
        return 0.5, 0.5  # Neutral if not provided
    
    education_lower = education_text.lower()
    
    # Find education level
    education_score = 0.5  # Default
    for level, info in EDUCATION_LEVELS.items():
        for keyword in info['keywords']:
            if keyword in education_lower:
                education_score = info['score']
                break
        if education_score != 0.5:
            break
    
    # Find university prestige
    prestige_score = 0.5  # Default
    for tier, info in PRESTIGE_UNIVERSITIES.items():
        for keyword in info['keywords']:
            if keyword in education_lower:
                prestige_score = info['score']
                break
        if prestige_score != 0.5:
            break
    
    log.debug(f"Education level: {education_score:.2f}, Prestige: {prestige_score:.2f}")
    
    return education_score, prestige_score


def extract_achievements(text: str) -> int:
    """
    Count quantified achievements in text.
    Higher count = more evidence of impact.
    
    Returns:
        Achievement count (0-100 scale)
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    achievement_count = 0
    
    # Count achievement keywords + metrics
    for keyword in ACHIEVEMENT_KEYWORDS['quantified']:
        achievement_count += text_lower.count(keyword)
    
    # Look for quantified metrics
    metric_patterns = [
        r'\d+%',  # "50%"
        r'\d+x',  # "10x"
        r'\d+\s*(?:million|billion|thousand)',  # "10 million"
        r'(?:improved|increased|reduced|faster)\s+\d+',  # "improved 30%"
    ]
    
    for pattern in metric_patterns:
        matches = re.findall(pattern, text_lower)
        achievement_count += len(matches)
    
    # Leadership achievements
    for keyword in ACHIEVEMENT_KEYWORDS['leadership']:
        achievement_count += text_lower.count(keyword) * 2  # Weight leadership higher
    
    # Innovation achievements
    for keyword in ACHIEVEMENT_KEYWORDS['innovation']:
        achievement_count += text_lower.count(keyword)
    
    return min(100, achievement_count)  # Cap at 100


def compute_achievement_score(achievement_count: int) -> float:
    """
    Convert achievement count to score (0.0-1.0).
    
    Args:
        achievement_count: Number of achievements found
    
    Returns:
        Score 0.0-1.0, with diminishing returns
    """
    # Sigmoid curve: more achievements = higher score, with diminishing returns
    # baseline: 5 achievements = 0.5, 20+ achievements = 0.95
    if achievement_count == 0:
        return 0.3
    
    score = 0.3 + 0.7 * (achievement_count / (achievement_count + 10))
    return min(1.0, score)


def compute_profile_quality_score(
    education_level_score: float,
    prestige_score: float,
    achievement_score: float
) -> float:
    """
    Combine education + prestige + achievements into overall profile quality.
    
    Args:
        education_level_score: 0.0-1.0
        prestige_score: 0.0-1.0
        achievement_score: 0.0-1.0
    
    Returns:
        Overall profile quality (0.0-1.0)
    """
    # Weighted: 40% education, 30% prestige, 30% achievements
    quality = (0.4 * education_level_score +
               0.3 * prestige_score +
               0.3 * achievement_score)
    
    return min(1.0, quality)


def apply_profile_quality_boost(
    cross_encoder_scores: np.ndarray,
    profile_quality_scores: np.ndarray,
    boost_factor: float = 0.10
) -> np.ndarray:
    """
    Apply profile quality as boost to CE scores.
    
    Args:
        cross_encoder_scores: Array of CE scores
        profile_quality_scores: Array of quality scores (0.0-1.0)
        boost_factor: Maximum boost (10% default)
    
    Returns:
        Boosted scores
    """
    quality_normalized = (profile_quality_scores - 0.5) * 2  # Normalize to [-1, 1]
    boosted = cross_encoder_scores * (1 + boost_factor * np.clip(quality_normalized, -1, 1))
    boosted = np.clip(boosted, 0, 1)
    
    log.info(f"Applied profile quality boost (factor={boost_factor})")
    
    return boosted
