"""
Skill Matching Module
Match candidate skills to JD requirements instead of just counting
"""

import logging
from collections import Counter
import re

log = logging.getLogger(__name__)


# Common skill synonyms mapping
SKILL_SYNONYMS = {
    'python': ['python', 'py', 'python3'],
    'machine learning': ['machine learning', 'ml', 'deep learning', 'neural network', 'neural networks'],
    'tensorflow': ['tensorflow', 'tf', 'tensorflow.js'],
    'pytorch': ['pytorch', 'torch'],
    'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'tsql', 't-sql'],
    'java': ['java', 'j2ee'],
    'javascript': ['javascript', 'js', 'nodejs', 'node.js', 'typescript', 'ts'],
    'aws': ['aws', 'amazon web services'],
    'gcp': ['gcp', 'google cloud', 'google cloud platform'],
    'azure': ['azure', 'microsoft azure'],
    'docker': ['docker', 'containers'],
    'kubernetes': ['kubernetes', 'k8s'],
    'spark': ['spark', 'apache spark', 'pyspark'],
    'data analysis': ['data analysis', 'analytics', 'business analytics'],
    'statistical analysis': ['statistical analysis', 'statistics', 'statistical modeling'],
    'visualization': ['visualization', 'tableau', 'power bi', 'matplotlib', 'plotly'],
}


def extract_skills_from_text(text: str) -> list:
    """
    Extract skills mentioned in candidate text.
    
    Args:
        text: Candidate enriched text
    
    Returns:
        List of identified skills (with duplicates)
    """
    text_lower = text.lower()
    skills_found = []
    
    # Check skill synonyms
    for main_skill, synonyms in SKILL_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in text_lower:
                skills_found.append(main_skill)
                break  # Only add main skill once per candidate
    
    return skills_found


def extract_jd_requirements(jd_text: str) -> dict:
    """
    Extract structured requirements from JD.
    Looks for patterns like "3+ years", "Required:", "Must have:", etc.
    
    Args:
        jd_text: Job description text
    
    Returns:
        Dict with 'required_skills', 'nice_to_have_skills', 'experience_years'
    """
    jd_lower = jd_text.lower()
    
    requirements = {
        'required_skills': [],
        'nice_to_have_skills': [],
        'years_required': 0,
    }
    
    # Extract years of experience pattern
    years_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)?', jd_lower)
    if years_match:
        requirements['years_required'] = int(years_match.group(1))
    
    # Extract required skills (after "Required:", "Must have:", "Must know:")
    required_section = re.search(
        r'(?:required|must have|must know)[\s\:]+(.*?)(?:nice|preferred|optional|additional|\n\n)',
        jd_lower,
        re.DOTALL
    )
    if required_section:
        section_text = required_section.group(1)
        # Find skills in this section
        for main_skill, synonyms in SKILL_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in section_text:
                    requirements['required_skills'].append(main_skill)
                    break
    
    # Extract nice-to-have skills
    nice_section = re.search(
        r'(?:nice|preferred|optional|additional|plus|bonus)[\s\:]+(.*?)(?:\n\n|$)',
        jd_lower,
        re.DOTALL
    )
    if nice_section:
        section_text = nice_section.group(1)
        for main_skill, synonyms in SKILL_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in section_text:
                    requirements['nice_to_have_skills'].append(main_skill)
                    break
    
    # Remove duplicates
    requirements['required_skills'] = list(set(requirements['required_skills']))
    requirements['nice_to_have_skills'] = list(set(requirements['nice_to_have_skills']))
    
    log.info(f"Extracted JD requirements: {len(requirements['required_skills'])} required, "
             f"{len(requirements['nice_to_have_skills'])} nice-to-have skills")
    
    return requirements


def compute_skill_match_score(candidate_skills: list, jd_requirements: dict) -> float:
    """
    Compute how well candidate skills match JD requirements.
    
    Args:
        candidate_skills: List of skills extracted from candidate
        jd_requirements: Dict from extract_jd_requirements()
    
    Returns:
        Score between 0.0 and 1.0 indicating skill match quality
    """
    if not jd_requirements['required_skills']:
        # If no required skills found in JD, return 0.5 (neutral)
        return 0.5
    
    candidate_skill_set = set(candidate_skills)
    required_set = set(jd_requirements['required_skills'])
    nice_set = set(jd_requirements['nice_to_have_skills'])
    
    # Calculate match scores
    required_matches = len(candidate_skill_set & required_set)
    required_total = len(required_set)
    required_score = required_matches / required_total if required_total > 0 else 0
    
    # Nice-to-have skills provide bonus (up to 20% bonus)
    nice_matches = len(candidate_skill_set & nice_set)
    nice_total = len(nice_set) if len(nice_set) > 0 else 1
    nice_bonus = min(0.2, (nice_matches / nice_total) * 0.2)
    
    total_score = min(1.0, required_score + nice_bonus)
    
    log.debug(f"Skill match: {required_matches}/{required_total} required + "
              f"{nice_matches}/{nice_total} nice-to-have = {total_score:.2f}")
    
    return total_score
