"""
Phase 2: Streaming Candidate Load + Per-Candidate Feature Engineering
Extracted from improved_new_v2_fixed.ipynb Phase 2 cells.
Streams candidates.jsonl and computes 40+ features per candidate.
"""

import json
import logging
from datetime import datetime, date
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

from constants_v3 import (
    TODAY, TITLE_STRONG_POS, TITLE_POS, TITLE_ADJACENT, TITLE_STRONG_NEG,
    SERVICES_COMPANIES, PRODUCT_INDUSTRIES, AI_SKILL_CANON, DESCRIPTION_AI_TERMS,
    INDIA_PREFERRED_CITIES, INDIA_WELCOME_CITIES, INDIA_ALL_CITIES,
    JD_YOE_MIN, JD_YOE_PREFERRED
)

log = logging.getLogger("redrob.phase2")

# ============================================================================
# 2a. Title scoring
# ============================================================================
def title_ai_score(title: str) -> float:
    """Return a real number in [-1, +1] for the candidate's current title."""
    t = (title or "").lower().strip()
    if not t:
        return 0.0
    
    for needle in TITLE_STRONG_POS:
        if needle in t:
            return 1.0
    for needle in TITLE_POS:
        if needle in t:
            return 0.7
    for needle in TITLE_ADJACENT:
        if needle in t:
            if any(k in t for k in ("ml", "ai ", "ai-", "data", "machine learning", "deep learning", "nlp")):
                return 0.7
            return 0.4
    if any(k in t for k in ("junior software engineer", "intern", "trainee", "fresher")):
        return 0.2
    for needle in TITLE_STRONG_NEG:
        if needle in t:
            return -1.0
    if any(k in t for k in ("manager", "lead ", "head ", "director", "vp ", "vice president", "chief")):
        return -0.4
    return 0.0

# ============================================================================
# 2b. Title-description coherence
# ============================================================================
TITLE_TOKENS_TO_CHECK = [
    "engineer", "developer", "scientist", "analyst", "designer", "writer",
    "manager", "recruiter", "support", "sales", "accountant", "mechanic",
    "technician", "operator", "driver", "chef", "nurse", "doctor", "lawyer",
    "teacher", "professor", "consultant", "architect", "lead", "head",
    "founder", "co-founder", "researcher", "specialist",
]

def title_description_match_score(title: str, career_descriptions: List[str]) -> float:
    """Fraction of title tokens that appear in any career description text."""
    t = (title or "").lower()
    desc_blob = " ".join((d or "").lower() for d in career_descriptions)
    if not t or not desc_blob:
        return 0.0
    matched = 0
    checked = 0
    for token in TITLE_TOKENS_TO_CHECK:
        if token in t:
            checked += 1
            if token in desc_blob:
                matched += 1
    if checked == 0:
        return 0.5
    return matched / checked

# ============================================================================
# 2c. Services-company detection
# ============================================================================
def is_services_only_career(current_company: str, career_history: List[Dict]) -> bool:
    """True if current company is services/consulting AND no prior product role."""
    cur = (current_company or "").lower().strip()
    is_cur_services = any(s in cur for s in SERVICES_COMPANIES) or cur in SERVICES_COMPANIES
    if not is_cur_services:
        return False
    for role in career_history or []:
        if role.get("is_current"):
            continue
        co = (role.get("company") or "").lower().strip()
        if not co:
            continue
        if not any(s in co for s in SERVICES_COMPANIES) and co not in SERVICES_COMPANIES:
            return False
        industry = (role.get("industry") or "").lower().strip()
        if industry in PRODUCT_INDUSTRIES:
            return False
    return True

# ============================================================================
# 2d. Location scoring
# ============================================================================
def location_score(country: str, location: str) -> float:
    """1.0 for Pune/Noida, 0.85 for welcome cities, 0.5 elsewhere in India, 0.3 outside."""
    c = (country or "").lower().strip()
    loc = (location or "").lower().strip()
    if c != "india":
        return 0.3
    city = loc.split(",")[0].strip()
    if city in INDIA_PREFERRED_CITIES:
        return 1.0
    if city in INDIA_WELCOME_CITIES:
        return 0.85
    if city in INDIA_ALL_CITIES:
        return 0.6
    return 0.5

# ============================================================================
# 2e. AI-skill match (exact-match canon, no substring traps)
# ============================================================================
def ai_skill_match(skill_name: str) -> bool:
    """Exact-match check against AI_SKILL_CANON."""
    n = (skill_name or "").lower().strip()
    if not n:
        return False
    if n in AI_SKILL_CANON:
        return True
    if n in {"ml", "ai", "cv"}:
        return False
    return False

# ============================================================================
# 2f. Description-AI mentions
# ============================================================================
def description_ai_count(descriptions: List[str]) -> int:
    """Count how many distinct canonical AI terms appear in description blob."""
    blob = " ".join((d or "").lower() for d in descriptions)
    if not blob:
        return 0
    n = 0
    for term in DESCRIPTION_AI_TERMS:
        if term in blob:
            n += 1
    return n

# ============================================================================
# 2g. Career-AI role months
# ============================================================================
def ai_role_months_in_career(career_history: List[Dict]) -> Tuple[int, int]:
    """Return (ai_months, total_months) across the candidate's career history."""
    ai_total = 0
    grand_total = 0
    for role in career_history or []:
        title = (role.get("title") or "").lower()
        months = int(role.get("duration_months") or 0)
        grand_total += months
        s = title_ai_score(title)
        if s >= 0.4:
            ai_total += months
        elif s >= 0.0:
            ai_total += int(0.5 * months)
    return ai_total, grand_total

# ============================================================================
# 2h. Honeypot detection
# ============================================================================
def honeypot_score(candidate: Dict) -> int:
    """Return 1 if the candidate matches a documented honeypot pattern."""
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience") or 0)
    
    if yoe > 30:
        return 1
    
    career = candidate.get("career_history", [])
    total_months = sum(int(r.get("duration_months") or 0) for r in career)
    if yoe > 0 and total_months > (yoe * 12 + 60):
        return 1
    
    for s in candidate.get("skills", []):
        if (s.get("proficiency") == "expert"
            and int(s.get("endorsements") or 0) == 0
            and int(s.get("duration_months") or 0) == 0):
            return 1
    
    for r in career:
        try:
            s = r.get("start_date")
            e = r.get("end_date")
            if s and e and e < s:
                return 1
        except Exception:
            pass
    
    return 0

# ============================================================================
# 2i. Behavioral piecewise functions
# ============================================================================
def response_rate_score(r: float) -> float:
    """JD: 5% response rate = not actually available."""
    if r >= 0.5:
        return 1.0
    if r >= 0.3:
        return 0.7 + 0.3 * (r - 0.3) / 0.2
    if r >= 0.1:
        return 0.4 * (r - 0.1) / 0.2 + 0.3
    return 0.1

def last_active_score(days_ago: float) -> float:
    """Score based on days since last active."""
    if days_ago < 0:
        return 0.5
    if days_ago <= 30:
        return 1.0
    if days_ago <= 90:
        return 0.85
    if days_ago <= 180:
        return 0.65
    if days_ago <= 365:
        return 0.35
    return 0.10

def notice_period_score(days: int) -> float:
    """JD: 'sub-30-day notice ideal; 30+ day notice bar gets higher'."""
    if days <= 15:
        return 1.0
    if days <= 30:
        return 0.85
    if days <= 60:
        return 0.6
    if days <= 90:
        return 0.35
    return 0.15

def github_score_norm(s: float) -> float:
    """-1 = no GitHub linked → 0.0; otherwise piecewise."""
    if s < 0:
        return 0.0
    if s >= 60:
        return 1.0
    if s >= 20:
        return 0.4 + 0.6 * (s - 20) / 40
    return 0.2 * s / 20

# ============================================================================
# 2j. Date parsing
# ============================================================================
def parse_date_safe(s: str):
    """Parse date string safely."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

# ============================================================================
# 2k. Main streaming loader
# ============================================================================
def stream_candidates_with_features(candidates_path: str, expected_count: int = 100_000) -> List[Dict]:
    """Load candidates.jsonl and compute features for each. Returns list of enriched records."""
    records = []
    
    log.info("Streaming candidates.jsonl ...")
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=expected_count, desc="  load"):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            
            cid = c.get("candidate_id", "")
            if not cid:
                continue
            
            profile = c.get("profile", {}) or {}
            redrob = c.get("redrob_signals", {}) or {}
            career = c.get("career_history", []) or []
            skills = c.get("skills", []) or []
            
            # Title & scoring
            title = profile.get("current_title", "")
            t_score = title_ai_score(title)
            
            # YoE
            yoe = float(profile.get("years_of_experience") or 0)
            yoe_fit = max(0.0, 1.0 - abs(yoe - JD_YOE_PREFERRED) / max(1.0, JD_YOE_PREFERRED * 1.5))
            
            # Location
            loc_score = location_score(profile.get("country", ""), profile.get("location", ""))
            
            # Services
            cur_co = profile.get("current_company", "")
            services_only = is_services_only_career(cur_co, career)
            
            # Description
            descs = [r.get("description", "") for r in career]
            desc_ai_n = description_ai_count(descs)
            td_match = title_description_match_score(title, descs)
            
            # Skills
            n_skills = len(skills)
            n_ai_skills = 0
            n_ai_advanced_or_expert = 0
            n_ai_with_endorsements = 0
            ai_skill_duration_months = 0
            skill_duration_total = 0
            for s in skills:
                name = s.get("name", "")
                dur = int(s.get("duration_months") or 0)
                skill_duration_total += dur
                if ai_skill_match(name):
                    n_ai_skills += 1
                    ai_skill_duration_months += dur
                    if s.get("proficiency") in ("advanced", "expert"):
                        n_ai_advanced_or_expert += 1
                    if int(s.get("endorsements") or 0) >= 3:
                        n_ai_with_endorsements += 1
            
            # Career months
            ai_months, total_months = ai_role_months_in_career(career)
            ai_role_pct = ai_months / max(1, total_months)
            
            # Behavioral
            rr = float(redrob.get("recruiter_response_rate", 0.0) or 0.0)
            rr = max(0.0, min(1.0, rr))
            rr_score = response_rate_score(rr)
            
            la = parse_date_safe(redrob.get("last_active_date", ""))
            la_days = (TODAY - la).days if la else 9999
            la_score = last_active_score(la_days)
            
            np_days = int(redrob.get("notice_period_days", 90) or 90)
            np_score = notice_period_score(np_days)
            
            otw = 1.0 if redrob.get("open_to_work_flag", False) else 0.0
            
            gh = float(redrob.get("github_activity_score", -1) or -1)
            gh_score = github_score_norm(gh)
            
            pc = float(redrob.get("profile_completeness_score", 0) or 0) / 100.0
            
            v_email = 1.0 if redrob.get("verified_email", False) else 0.0
            v_phone = 1.0 if redrob.get("verified_phone", False) else 0.0
            v_both = (v_email + v_phone) / 2.0
            
            oa = float(redrob.get("offer_acceptance_rate", -1) or -1)
            oa_score = 0.5 if oa < 0 else oa
            
            sbr = int(redrob.get("saved_by_recruiters_30d", 0) or 0)
            sap = int(redrob.get("search_appearance_30d", 0) or 0)
            
            # Honeypot
            hp = honeypot_score(c)
            
            # Append enriched record
            records.append({
                "id": cid,
                "title": title,
                "title_ai_score": t_score,
                "yoe": yoe,
                "yoe_fit": yoe_fit,
                "loc_score": loc_score,
                "country": profile.get("country", ""),
                "location": profile.get("location", ""),
                "current_company": cur_co,
                "current_industry": profile.get("current_industry", ""),
                "services_only": int(services_only),
                "desc_ai_n": desc_ai_n,
                "td_match": td_match,
                "n_skills": n_skills,
                "n_ai_skills": n_ai_skills,
                "n_ai_adv": n_ai_advanced_or_expert,
                "n_ai_endorse": n_ai_with_endorsements,
                "ai_role_pct": ai_role_pct,
                "ai_months": ai_months,
                "total_months": total_months,
                "rr": rr,
                "rr_score": rr_score,
                "la_days": la_days,
                "la_score": la_score,
                "np_score": np_score,
                "otw": otw,
                "gh_score": gh_score,
                "pc": pc,
                "v_both": v_both,
                "oa_score": oa_score,
                "sbr": sbr,
                "sap": sap,
                "hp": hp,
                # For text indexing & reasoning
                "headline": profile.get("headline", ""),
                "summary": profile.get("summary", ""),
                "descs": descs,
                "skills": skills,
                "education": c.get("education", []),
            })
    
    log.info(f"  ✓ Loaded {len(records)} candidates")
    
    # Sanity checks
    n_honeypot = sum(r["hp"] for r in records)
    n_services = sum(r["services_only"] for r in records)
    n_strong_pos = sum(1 for r in records if r["title_ai_score"] >= 0.7)
    n_strong_neg = sum(1 for r in records if r["title_ai_score"] <= -0.5)
    log.info(f"  ✓ Honeypot candidates: {n_honeypot} ({n_honeypot/len(records):.1%})")
    log.info(f"  ✓ Services-only careers: {n_services} ({n_services/len(records):.1%})")
    log.info(f"  ✓ Strong-positive titles: {n_strong_pos} ({n_strong_pos/len(records):.1%})")
    log.info(f"  ✓ Strong-negative titles: {n_strong_neg} ({n_strong_neg/len(records):.1%})")
    
    return records
