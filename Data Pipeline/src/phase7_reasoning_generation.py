"""
Phase 7: Reasoning Generation (Anti-Hallucination)
Extracted from improved_new_v2_fixed.ipynb Phase 7 cells.
Generates reasoning strings using only real candidate fields.
"""

import logging
from typing import List, Dict

from constants_v3 import AI_SKILL_CANON, PRODUCT_INDUSTRIES

log = logging.getLogger("redrob.phase7")

# ============================================================================
# 7a. Extract real AI skills from candidate
# ============================================================================
def real_top_ai_skills(candidate: Dict, top_n: int = 3) -> List[str]:
    """Extract up to top_n AI skills with good proficiency/endorsements from candidate.skills."""
    skills = candidate.get("skills", [])
    ai_skills_with_scores = []
    
    for skill in skills:
        name = skill.get("name", "").lower().strip()
        if name not in AI_SKILL_CANON:
            continue
        
        # Score skill: proficiency + endorsements + duration
        proficiency = skill.get("proficiency", "")
        prof_score = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0.5}.get(proficiency, 0)
        endorsements = int(skill.get("endorsements", 0))
        duration_months = int(skill.get("duration_months", 0))
        
        # Composite score
        score = prof_score + endorsements * 0.1 + min(duration_months / 12.0, 1.0) * 0.2
        ai_skills_with_scores.append((skill.get("name", ""), score))
    
    # Sort by score, take top_n
    ai_skills_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in ai_skills_with_scores[:top_n]]

# ============================================================================
# 7b. Extract industry from candidate
# ============================================================================
def real_industry_for_reason(candidate: Dict) -> str:
    """Extract current_industry from candidate profile."""
    industry = candidate.get("current_industry", "").strip()
    if not industry:
        return ""
    return industry

# ============================================================================
# 7c. Build reasoning string
# ============================================================================
def build_reasoning(candidate: Dict, top_ai_skills: List[str], industry: str) -> str:
    """Build a structured reasoning string using only real fields."""
    title = candidate.get("title", "").strip()
    yoe = candidate.get("yoe", 0)
    ai_months = candidate.get("ai_months", 0)
    ai_role_pct = candidate.get("ai_role_pct", 0)
    desc_ai_n = candidate.get("desc_ai_n", 0)
    rr_score = candidate.get("rr_score", 0.5)
    la_score = candidate.get("la_score", 0.5)
    la_days = candidate.get("la_days", 365)
    otw = candidate.get("otw", 0)
    location = candidate.get("location", "").strip()
    
    # Build parts
    parts = []
    
    # Title + YoE
    parts.append(f"{title} with {yoe:.1f} yrs")
    
    # AI career indicator
    if ai_months > 0:
        ai_months_rounded = int(ai_months)
        ai_pct_str = f"{int(ai_role_pct*100)}%" if ai_role_pct > 0 else "some"
        parts.append(f"{ai_months_rounded}+ yrs in applied ML/AI roles")
    
    # Top AI skills
    if top_ai_skills:
        skills_str = ", ".join(top_ai_skills[:3])
        parts.append(f"skills: {skills_str}")
    
    # Description AI mentions
    if desc_ai_n > 0:
        parts.append(f"{desc_ai_n} AI-domain terms in career description")
    elif desc_ai_n == 0 and ai_months > 0:
        parts.append("AI-domain language in career history")
    
    # Behavioral signals
    behavioral = []
    if otw > 0.5:
        behavioral.append("open to work")
    if la_days <= 30:
        behavioral.append("active in last 30 days")
    elif la_days <= 90:
        behavioral.append("active in last 90 days")
    if 0.5 <= rr_score <= 0.9:
        behavioral.append(f"recruiter response rate {rr_score:.2f}")
    
    if behavioral:
        parts.append("; ".join(behavioral) + ".")
    else:
        parts.append("standard profile.")
    
    reasoning = "; ".join(parts)
    
    # Truncate to ~200 chars for CSV cleanliness
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."
    
    return reasoning

# ============================================================================
# Main Phase 7
# ============================================================================
def run_phase7(ce_records: List[Dict]) -> Dict:
    """Execute Phase 7: generate reasoning for all final candidates."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 7 — ANTI-HALLUCINATION REASONING GENERATION")
    log.info("=" * 70)
    
    reasonings = []
    for record in ce_records:
        top_skills = real_top_ai_skills(record, top_n=3)
        industry = real_industry_for_reason(record)
        reasoning = build_reasoning(record, top_skills, industry)
        reasonings.append(reasoning)
    
    log.info(f"  ✓ Generated {len(reasonings)} reasoning strings")
    
    return {
        "reasonings": reasonings,
    }
