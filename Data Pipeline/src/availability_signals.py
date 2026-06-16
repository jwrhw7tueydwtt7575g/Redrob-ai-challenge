"""
Skill Recency & Availability Signals Module
Weight recent skills higher, detect active job seekers
"""

import logging
import re
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)


def extract_skill_recency(experience_text: str) -> float:
    """
    Extract recency score based on most recent role dates.
    
    Args:
        experience_text: Candidate's experience history
    
    Returns:
        Recency score (0.0-1.0), where 1.0 = currently employed in relevant role
    """
    if not experience_text:
        return 0.5
    
    experience_lower = experience_text.lower()
    
    # Look for "present", "current", "ongoing"
    if any(kw in experience_lower for kw in ['present', 'current', 'ongoing', 'now']):
        return 1.0
    
    # Look for recent year (2024, 2025, 2026)
    current_year = datetime.now().year
    year_pattern = r'(20\d{2})'
    years = [int(m) for m in re.findall(year_pattern, experience_lower)]
    
    if years:
        most_recent_year = max(years)
        years_ago = current_year - most_recent_year
        
        if years_ago == 0:
            return 0.95
        elif years_ago == 1:
            return 0.85
        elif years_ago == 2:
            return 0.70
        elif years_ago <= 4:
            return 0.50
        else:
            return 0.30
    
    return 0.5


def detect_availability_signals(candidate_summary: str, headline: str) -> float:
    """
    Detect signals that candidate is actively looking for job.
    
    Args:
        candidate_summary: Candidate's summary/about section
        headline: LinkedIn headline or title
    
    Returns:
        Availability score (0.0-1.0), where 1.0 = very likely looking
    """
    if not candidate_summary and not headline:
        return 0.5
    
    full_text = (candidate_summary + " " + headline).lower()
    
    # Strong signals of job seeking
    seeking_keywords = [
        'looking for', 'seeking', 'open to', 'interested in', 'exploring',
        'available', 'opportunity', 'hire me', 'freelance', 'contract',
        'open to opportunities', 'actively looking', 'open positions'
    ]
    
    # Signals of NOT looking
    not_seeking = [
        'not looking', 'not interested', 'not available', 'focused on current',
        'focused on growing', 'dedicated to'
    ]
    
    # Count signals
    seeking_count = sum(full_text.count(kw) for kw in seeking_keywords)
    not_seeking_count = sum(full_text.count(kw) for kw in not_seeking)
    
    # Score: base 0.5, +0.1 per seeking signal, -0.1 per not seeking signal
    score = 0.5 + (0.1 * seeking_count) - (0.1 * not_seeking_count)
    
    return np.clip(score, 0.0, 1.0)


def detect_notice_period(headline: str, summary: str) -> float:
    """
    Estimate notice period from text clues.
    
    Returns:
        Score (0.0-1.0), where 1.0 = can start immediately
    """
    if not headline and not summary:
        return 0.5
    
    full_text = (headline + " " + summary).lower()
    
    # Immediate availability
    if any(kw in full_text for kw in ['immediate', 'available now', 'start now', 'ready to start']):
        return 1.0
    
    # 2 weeks notice (typical)
    if any(kw in full_text for kw in ['2 weeks', 'two weeks']):
        return 0.85
    
    # 1 month notice
    if any(kw in full_text for kw in ['1 month', 'one month', '30 days']):
        return 0.75
    
    # 2-3 months notice
    if any(kw in full_text for kw in ['2 months', 'three months', '3 months']):
        return 0.50
    
    # Longer or unspecified
    return 0.60


def compute_availability_score(
    is_actively_looking: float,
    notice_period_score: float,
    skill_recency: float
) -> float:
    """
    Combine availability signals into single score.
    
    Args:
        is_actively_looking: Job seeking signal (0.0-1.0)
        notice_period_score: Notice period score (0.0-1.0)
        skill_recency: Skill recency (0.0-1.0)
    
    Returns:
        Combined availability score (0.0-1.0)
    """
    # Weighted: 40% seeking, 35% notice, 25% recency
    combined = (0.4 * is_actively_looking +
                0.35 * notice_period_score +
                0.25 * skill_recency)
    
    return min(1.0, combined)


def apply_availability_boost(
    cross_encoder_scores: np.ndarray,
    availability_scores: np.ndarray,
    boost_factor: float = 0.08
) -> np.ndarray:
    """
    Apply availability as boost to CE scores.
    
    Args:
        cross_encoder_scores: Array of CE scores
        availability_scores: Array of availability scores (0.0-1.0)
        boost_factor: Maximum boost (8% default)
    
    Returns:
        Boosted scores
    """
    availability_normalized = (availability_scores - 0.5) * 2
    boosted = cross_encoder_scores * (1 + boost_factor * np.clip(availability_normalized, -1, 1))
    boosted = np.clip(boosted, 0, 1)
    
    log.info(f"Applied availability boost (factor={boost_factor})")
    
    return boosted
