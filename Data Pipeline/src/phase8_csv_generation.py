"""
Phase 8: CSV Generation & Validation
Extracted from improved_new_v2_fixed.ipynb Phase 8 cells.
Creates output CSV with proper format and validation.
"""

import csv
import logging
from typing import List, Dict

import pandas as pd
import numpy as np

log = logging.getLogger("redrob.phase8")

# ============================================================================
# 8a. Build submission dataframe
# ============================================================================
def build_submission_dataframe(ce_records: List[Dict], scores_final: np.ndarray, 
                               reasonings: List[str]) -> pd.DataFrame:
    """Build submission dataframe: candidate_id, rank, score, reasoning."""
    
    data = []
    for i, (record, score, reasoning) in enumerate(zip(ce_records, scores_final, reasonings)):
        data.append({
            "candidate_id": record["id"],
            "rank": i + 1,
            "score": float(score),
            "reasoning": reasoning,
        })
    
    df = pd.DataFrame(data)
    return df

# ============================================================================
# 8b. Validate submission
# ============================================================================
def validate_submission(df: pd.DataFrame) -> bool:
    """Validate submission CSV."""
    errors = []
    
    # Check row count
    if len(df) != 100:
        errors.append(f"Expected 100 rows, got {len(df)}")
    
    # Check columns
    expected_cols = {"candidate_id", "rank", "score", "reasoning"}
    if set(df.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {set(df.columns)}")
    
    # Check uniqueness
    if len(df["candidate_id"].unique()) != len(df):
        errors.append("candidate_id contains duplicates")
    
    # Check rank sequence
    expected_ranks = list(range(1, len(df) + 1))
    if list(df["rank"]) != expected_ranks:
        errors.append("rank column is not sequential 1-100")
    
    # Check score range
    if (df["score"] < 0).any() or (df["score"] > 1).any():
        errors.append("Some scores are outside [0, 1]")
    
    # Check strict descending
    if not (df["score"].iloc[:-1] >= df["score"].iloc[1:]).all():
        errors.append("Scores are not strictly descending")
    
    # Check score decimals
    for score in df["score"]:
        score_str = f"{score:.4f}"
        if float(score_str) != score:
            errors.append(f"Score {score} does not have exactly 4 decimals")
    
    if errors:
        log.error("Validation failed:")
        for error in errors:
            log.error(f"  ✗ {error}")
        return False
    
    log.info("  ✓ All validation checks passed")
    return True

# ============================================================================
# Main Phase 8
# ============================================================================
def run_phase8(ce_records: List[Dict], scores_final: np.ndarray,
               reasonings: List[str], output_path: str) -> Dict:
    """Execute Phase 8: build and save submission CSV."""
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 8 — CSV GENERATION & VALIDATION")
    log.info("=" * 70)
    
    # Build dataframe
    df = build_submission_dataframe(ce_records, scores_final, reasonings)
    
    # Validate
    is_valid = validate_submission(df)
    if not is_valid:
        log.error("Submission validation failed. Not saving CSV.")
        return {"success": False, "dataframe": df}
    
    # Save to CSV
    df.to_csv(output_path, quoting=csv.QUOTE_NONNUMERIC, index=False)
    log.info(f"  ✓ Saved submission to {output_path}")
    
    # Pretty print
    log.info("\n  First 10 rows:")
    for _, row in df.head(10).iterrows():
        log.info(f"    {row['rank']:3d}. {row['candidate_id']:15s} {row['score']:.4f} | {row['reasoning'][:60]}...")
    
    log.info("\n  Last 5 rows:")
    for _, row in df.tail(5).iterrows():
        log.info(f"    {row['rank']:3d}. {row['candidate_id']:15s} {row['score']:.4f} | {row['reasoning'][:60]}...")
    
    return {
        "success": True,
        "dataframe": df,
        "output_path": output_path,
    }
