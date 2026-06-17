"""
Redrob v3.0 Data Pipeline Package
Complete modularization of the improved_new_v2_fixed.ipynb ranking pipeline.

Architecture: 9-phase ranking pipeline for Senior AI Engineer role matching.
- Phase 1: JD ingestion & parsing
- Phase 2: Feature engineering (40+ features per candidate)
- Phase 3: Hard pre-filter + BM25 + dense + RRF
- Phase 4: Retrieval pool selection (top 2000 by RRF)
- Phase 5: XGBoost LTR training on real silver labels
- Phase 6: Cross-encoder re-ranking (top 200 → top 100)
- Phase 7: Anti-hallucination reasoning generation
- Phase 8: CSV generation & validation
- Phase 9: Sanity checks & self-report

Main entry point: pipeline.run_pipeline()
"""

__version__ = "3.0.0"
__author__ = "Redrob AI Challenge"

from pipeline import run_pipeline
from constants_v3 import (
    TOP_K, CE_WINDOW, TOP_K_RETRIEVAL,
    TITLE_STRONG_POS, TITLE_POS, TITLE_ADJACENT, TITLE_STRONG_NEG,
    AI_SKILL_CANON, DESCRIPTION_AI_TERMS,
    SERVICES_COMPANIES, PRODUCT_INDUSTRIES,
    INDIA_PREFERRED_CITIES, INDIA_WELCOME_CITIES, INDIA_ALL_CITIES,
)

__all__ = [
    "run_pipeline",
    "TOP_K",
    "CE_WINDOW", 
    "TOP_K_RETRIEVAL",
    "TITLE_STRONG_POS",
    "TITLE_POS",
    "TITLE_ADJACENT",
    "TITLE_STRONG_NEG",
    "AI_SKILL_CANON",
    "DESCRIPTION_AI_TERMS",
    "SERVICES_COMPANIES",
    "PRODUCT_INDUSTRIES",
    "INDIA_PREFERRED_CITIES",
    "INDIA_WELCOME_CITIES",
    "INDIA_ALL_CITIES",
]
