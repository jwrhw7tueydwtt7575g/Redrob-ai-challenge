"""
Advanced JD Parsing Module
Extract detailed requirements: must-have vs nice-to-have, years required, salary, etc.
"""

import logging
import re
import numpy as np

log = logging.getLogger(__name__)


def parse_jd_comprehensive(jd_text: str) -> dict:
    """
    Parse JD to extract structured requirements.
    
    Returns:
        Dict with:
        - required_skills: list
        - nice_to_have_skills: list
        - years_minimum: int
        - years_preferred: int
        - education_required: str
        - location: str
        - job_level: str (entry/mid/senior/lead)
        - key_responsibilities: list
    """
    jd_lower = jd_text.lower()
    
    result = {
        'required_skills': [],
        'nice_to_have_skills': [],
        'years_minimum': 0,
        'years_preferred': 0,
        'education_required': '',
        'location': [],
        'job_level': 'mid',
        'key_responsibilities': [],
    }
    
    # Extract experience required
    exp_patterns = [
        r'(\d+)\s*(?:to|\-)\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+\s*(?:years?|yrs?)',
        r'(?:at least|minimum)\s+(\d+)\s*(?:years?|yrs?)',
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                result['years_minimum'] = int(groups[0])
                result['years_preferred'] = int(groups[1])
            else:
                result['years_minimum'] = int(groups[0])
                result['years_preferred'] = int(groups[0]) + 2
            break
    
    # Extract job level
    level_patterns = {
        'entry': ['entry level', 'junior', 'graduate'],
        'mid': ['mid level', 'mid-level', 'intermediate'],
        'senior': ['senior', 'staff'],
        'lead': ['lead', 'principal', 'head'],
    }
    
    for level, keywords in level_patterns.items():
        if any(kw in jd_lower for kw in keywords):
            result['job_level'] = level
            break
    
    # Extract education requirement
    edu_patterns = {
        'Bachelor': r"bachelor|bs|b\.s",
        'Master': r"master|ms|m\.s|mtech",
        'PhD': r"phd|ph\.d|doctorate",
    }
    
    for edu_level, pattern in edu_patterns.items():
        if re.search(pattern, jd_lower):
            result['education_required'] = edu_level
            break
    
    # Extract location
    location_patterns = [
        r'(?:location|based|located|remote|onsite)\s*:?\s*([a-zA-Z\s,]+)(?:\.|,)',
        r'([a-zA-Z]+),?\s+(?:US|USA|UK|India|Canada)',
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, jd_lower)
        if matches:
            result['location'] = [m.strip() for m in matches[:3]]
            break
    
    # Extract key responsibilities (bullet points)
    responsibility_patterns = [
        r'(?:responsibilities?|duties?|involve)[s]?\s*:?\s*([^\.]+)',
        r'(?:•|-|\*)\s*([^\.]+\.[^\.]{10,})',
    ]
    
    for pattern in responsibility_patterns:
        matches = re.findall(pattern, jd_lower)
        if matches:
            result['key_responsibilities'] = [m.strip() for m in matches[:5]]
            break
    
    log.info(f"Parsed JD: {result['years_minimum']}-{result['years_preferred']} YoE, "
             f"{result['job_level']} level, {len(result['required_skills'])} required skills")
    
    return result


def extract_requirements_section(jd_text: str, section_keyword: str = 'required') -> list:
    """
    Extract skills from specific section (required, nice-to-have, etc).
    
    Args:
        jd_text: Job description
        section_keyword: 'required', 'nice-to-have', 'preferred', etc.
    
    Returns:
        List of skills found in that section
    """
    # Look for section with keyword
    pattern = rf'(?:{section_keyword}[s]?)[:\s]+(.*?)(?:(?:preferred|nice|optional|additional|apply)|\n\n|\Z)'
    match = re.search(pattern, jd_text.lower(), re.DOTALL)
    
    if not match:
        return []
    
    section_text = match.group(1)
    
    # Extract comma/newline separated items
    skills = re.findall(r'([a-z\s\+\#\.]+)(?:,|\n|\s{2,})', section_text)
    skills = [s.strip() for s in skills if len(s.strip()) > 2]
    
    return skills[:20]  # Limit to top 20


def compute_jd_alignment_score(
    candidate_profiles: dict,
    jd_parsed: dict
) -> np.ndarray:
    """
    Compute overall JD alignment for multiple candidates.
    
    Args:
        candidate_profiles: Dict with candidate info
        jd_parsed: Parsed JD requirements
    
    Returns:
        Array of alignment scores (0.0-1.0)
    """
    n_candidates = len(candidate_profiles['yoe'])
    alignment_scores = np.ones(n_candidates) * 0.5
    
    # YoE alignment (40% weight)
    if jd_parsed['years_minimum'] > 0:
        yoe_alignment = np.array([
            compute_yoe_fit(yoe, jd_parsed['years_minimum'], jd_parsed['years_preferred'])
            for yoe in candidate_profiles['yoe']
        ])
        alignment_scores += 0.4 * (yoe_alignment - 0.5)
    
    # TODO: Add skill alignment, education alignment, etc.
    
    return np.clip(alignment_scores, 0, 1)


def compute_yoe_fit(candidate_yoe: float, min_years: int, preferred_years: int) -> float:
    """Helper to compute YoE fit (0.0-1.0)."""
    if candidate_yoe < min_years:
        return 0.3 + (0.2 * candidate_yoe / max(min_years, 1))
    elif candidate_yoe < preferred_years:
        return 0.5 + (0.4 * (candidate_yoe - min_years) / max(preferred_years - min_years, 1))
    elif candidate_yoe == preferred_years:
        return 1.0
    else:
        return min(1.0, 1.0 + 0.05)
