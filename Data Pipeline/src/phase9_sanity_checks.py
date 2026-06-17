"""
Phase 9: Sanity Checks & Self-Report
Extracted from improved_new_v2_fixed.ipynb Phase 9 cells.
Performs final quality assurance checks on the submission.
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd

from constants_v3 import TITLE_STRONG_POS, TITLE_POS, TITLE_ADJACENT

log = logging.getLogger("redrob.phase9")

# ============================================================================
# 9a. Sanity gate checks
# ============================================================================
def run_sanity_checks(ce_records: List[Dict], df: pd.DataFrame) -> Dict:
    """Perform final quality checks on the submission."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 9 — SANITY CHECKS & SELF-REPORT")
    log.info("=" * 70)
    
    # Check 1: Trap titles (title_ai_score <= -0.5)
    n_trap = sum(1 for r in ce_records if r.get("title_ai_score", 0) <= -0.5)
    
    # Check 2: Services-only careers
    n_services = sum(1 for r in ce_records if r.get("services_only", 0) == 1)
    
    # Check 3: Honeypots
    n_honeypot = sum(1 for r in ce_records if r.get("hp", 0) == 1)
    
    # Check 4: Strong positive titles (>= 0.7)
    n_strong_pos = sum(1 for r in ce_records if r.get("title_ai_score", 0) >= 0.7)
    
    # Check 5: Pos or adjacent (>= 0.4)
    n_pos_or_adjacent = sum(1 for r in ce_records if r.get("title_ai_score", 0) >= 0.4)
    
    log.info(f"  Title quality checks:")
    log.info(f"    Trap titles (≤ -0.5):       {n_trap} (target: 0)")
    log.info(f"    Services-only careers:      {n_services} (target: 0)")
    log.info(f"    Honeypot candidates:        {n_honeypot} (target: 0)")
    log.info(f"    Strong-positive titles:     {n_strong_pos}")
    log.info(f"    Pos or adjacent titles:     {n_pos_or_adjacent} (target: ≥80)")
    
    # Check 6: Behavioral metrics
    yoes = [r.get("yoe", 0) for r in ce_records]
    ai_role_pcts = [r.get("ai_role_pct", 0) for r in ce_records]
    rrs = [r.get("rr", 0) for r in ce_records]
    la_days = [r.get("la_days", 365) for r in ce_records]
    
    log.info(f"  Behavioral signals:")
    log.info(f"    Avg YoE:                    {np.mean(yoes):.2f} years")
    log.info(f"    Avg AI role %:              {np.mean(ai_role_pcts):.1%}")
    log.info(f"    Avg recruiter response rate: {np.mean(rrs):.2%}")
    log.info(f"    Avg last-active days:       {np.mean(la_days):.0f} days")
    
    # Check 7: India location
    n_india = sum(1 for r in ce_records if r.get("country", "").lower() == "india")
    log.info(f"    India-located:              {n_india}/{len(ce_records)} ({n_india/len(ce_records):.1%})")
    
    # Check 8: Open to work
    n_otw = sum(1 for r in ce_records if r.get("otw", 0) > 0.5)
    log.info(f"    Open to work:               {n_otw}/{len(ce_records)} ({n_otw/len(ce_records):.1%})")
    
    # Final verdict
    pass_fail = {
        "trap_titles_zero": n_trap == 0,
        "services_zero": n_services == 0,
        "honeypots_zero": n_honeypot == 0,
        "pos_or_adjacent_high": n_pos_or_adjacent >= 80,
    }
    
    all_pass = all(pass_fail.values())
    
    log.info(f"\n  Sanity gate:")
    for check, result in pass_fail.items():
        status = "✓" if result else "✗"
        log.info(f"    {status} {check}")
    
    if all_pass:
        log.info(f"\n  ✅ ALL SANITY CHECKS PASSED")
    else:
        log.warning(f"\n  ⚠ Some sanity checks failed")
    
    return {
        "pass_fail": pass_fail,
        "all_pass": all_pass,
        "metrics": {
            "n_trap": n_trap,
            "n_services": n_services,
            "n_honeypot": n_honeypot,
            "n_strong_pos": n_strong_pos,
            "n_pos_or_adjacent": n_pos_or_adjacent,
            "avg_yoe": np.mean(yoes),
            "avg_ai_role_pct": np.mean(ai_role_pcts),
            "avg_recruiter_response": np.mean(rrs),
            "avg_last_active_days": np.mean(la_days),
            "n_india": n_india,
            "n_otw": n_otw,
        }
    }

# ============================================================================
# Main Phase 9
# ============================================================================
def run_phase9(ce_records: List[Dict], df: pd.DataFrame) -> Dict:
    """Execute Phase 9: sanity checks."""
    
    results = run_sanity_checks(ce_records, df)
    
    return results
