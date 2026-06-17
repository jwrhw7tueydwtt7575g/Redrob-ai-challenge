# Data Pipeline v3.0 - Modular Ranking Architecture

## Overview

This is a complete modularization of the `improved_new_v2_fixed.ipynb` notebook into a production-ready Python package. The pipeline implements a 9-phase ranking system for matching candidates to the Senior AI Engineer role at Redrob AI (Series A, Pune/Noida, 5-9 yrs YoE).

## Architecture

### 9-Phase Pipeline

```
Input: candidates.jsonl (100K+), job_description.docx
  ↓
Phase 1: JD Ingestion & Parsing
  - Read .docx, extract sections, parse YoE range, build retrieval texts
  Output: jd_query_text, jd_ce_text, jd_tokens
  ↓
Phase 2: Feature Engineering
  - Stream candidates, compute 40+ features per candidate
  Output: records (100K+ enriched records)
  ↓
Phase 3: Hard Pre-Filter + Retrieval (BM25 + Dense + RRF)
  - Filter trap candidates (honeypots, services-only, trap titles)
  - Build BM25 index, dense embeddings (all-MiniLM-L6-v2)
  - RRF fusion (k=60) for hybrid ranking
  Output: surviving_records (~80K), retrieval scores
  ↓
Phase 4: Retrieval Pool Selection
  - Select top 2000 by RRF score
  Output: retrieval_records, retrieval_*_n arrays
  ↓
Phase 5: XGBoost LTR Training
  - Build 28-feature matrix
  - Compute heuristic silver labels (5-level)
  - Train rank:ndcg model
  Output: booster, ltr_scores (predictions on pool)
  ↓
Phase 6: Cross-Encoder Re-Ranking
  - Select top 200 by LTR score
  - Run ms-marco-MiniLM-L-6-v2 cross-encoder
  - Select top 100 by CE score
  - Normalize & calibrate scores to [0.55, 0.985]
  Output: final_order, scores_final
  ↓
Phase 7: Anti-Hallucination Reasoning
  - Extract real AI skills, industry, behavioral signals
  - Generate structured reasoning (only real fields)
  Output: reasonings (100 strings)
  ↓
Phase 8: CSV Generation & Validation
  - Build DataFrame: candidate_id, rank, score, reasoning
  - Validate: 100 rows, strictly descending scores, [0,1] range
  - Save to CSV
  Output: submission.csv
  ↓
Phase 9: Sanity Checks & Self-Report
  - Title quality: trap titles, strong-positive %
  - Behavioral metrics: YoE, AI role %, response rate
  - Pass/fail report
  ↓
Output: submission.csv (100 rows, validated)
```

## Module Structure

```
Data Pipeline/
├── src/
│   ├── __init___v3.py              # Package exports
│   ├── constants_v3.py             # All constants (titles, skills, locations)
│   ├── phase1_jd_ingestion.py      # JD parsing
│   ├── phase2_feature_engineering.py # Feature extraction
│   ├── phase3_hard_prefilter.py    # Hard filtering + BM25 + dense + RRF
│   ├── phase4_retrieval_ranking.py # Retrieval pool selection
│   ├── phase5_ltr_training.py      # XGBoost LTR training
│   ├── phase6_cross_encoder_rerank.py # CE re-ranking
│   ├── phase7_reasoning_generation.py # Reasoning generation
│   ├── phase8_csv_generation.py    # CSV output & validation
│   ├── phase9_sanity_checks.py     # Final QA checks
│   ├── pipeline.py                 # Main orchestrator + CLI
│   ├── requirements.txt            # Dependencies
│   └── README.md                   # This file
└── data/
    ├── artifacts/                  # (Pre-computed models, indexes)
    ├── processed/                  # (Intermediate results)
    └── raw/                        # (Input candidate.jsonl, job_description.docx)
```

## Key Features

### 1. **No Substring Traps**
- AI_SKILL_CANON: 100+ exact-match terms (not substrings)
- Avoids "CNN" (company) or "Databricks" (company name) being scored as AI skills

### 2. **Hard Pre-Filter**
- Removes honeypots (YoE > 30, inflated durations, expert + 0 endorsements)
- Removes services-only careers (TCS, Infosys, Accenture if no prior product)
- Removes trap titles (Marketing Manager, Civil Engineer, HR, QA, etc.)
- Filtering BEFORE scoring ensures clean pool

### 3. **Career-Evidence Signals**
- `ai_role_months`: months spent in AI/ML/SWE roles (not skill count)
- `ai_role_pct`: AI time as % of total career
- `description_ai_mentions`: canonical AI terms in description text
- Title-description coherence: title tokens match career descriptions

### 4. **Real Silver Labels**
- Heuristic scorer: weighted combination of 10 signals (title, AI role %, description, skills, behavioral)
- Converts to 5-level (0-4) labels based on percentiles
- Replaces v2.0's random `np.random.randint(0, 4)` noise

### 5. **Behavioral Weighting**
- `response_rate_score`: response rate < 5% heavily penalized
- `last_active_score`: dormant (> 1 year) = 0.1
- `notice_period_score`: sub-30 day preferred, 30+ day penalty
- `open_to_work_flag`: strong signal
- `github_activity_score`, `offer_acceptance_rate`

### 6. **Hybrid Retrieval (RRF)**
- BM25: keyword matching over entire profile
- Dense: normalized cosine similarity via all-MiniLM-L6-v2 (384-dim)
- RRF (k=60): reciprocal rank fusion combines both signals
- No hardcoded random vectors (v2.0 used `np.random.rand()`)

### 7. **Anti-Hallucination Reasoning**
- Only uses real fields: title, YoE, AI months, top skills, industry, behavioral signals
- Never invents skills or false information
- Example: "Senior AI Engineer with 5.9 yrs; 6+ yrs in applied ML/AI roles; skills: FAISS, TensorFlow; 15 AI-domain terms in career description; open to work, active in last 30 days."

### 8. **Strict Score Calibration**
- All 100 scores strictly descending (enforced with epsilon offsets)
- Normalized to [0.55, 0.985] for proper distribution
- Exactly 4 decimal places

## Usage

### Python API

```python
from data_pipeline import run_pipeline

result = run_pipeline(
    candidates_path="India_runs_data_and_ai_challenge/candidates.jsonl",
    jd_path="India_runs_data_and_ai_challenge/job_description.docx",
    output_path="submission.csv",
    device="cuda"  # or "cpu"
)

if result["success"]:
    print(f"✓ Output: {result['output_path']}")
    print(f"✓ Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"✓ Sanity checks: {result['sanity_checks']['all_pass']}")
else:
    print(f"✗ Error: {result['error']}")
```

### Command-Line

```bash
cd Data\ Pipeline/src
python pipeline.py \
    --candidates /path/to/candidates.jsonl \
    --jd /path/to/job_description.docx \
    --output ./submission.csv \
    --device cuda
```

## Dependencies

```
sentence-transformers>=2.2,<3
transformers>=4.39,<5
torch>=2.0
rank-bm25>=0.2.2
xgboost>=2.0
scikit-learn>=1.2
pandas>=2.0
numpy>=1.24
tqdm>=4.65
python-docx>=1.0
```

See `requirements.txt` for pinned versions.

## Configuration

All hyperparameters are in `constants_v3.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `TOP_K` | 100 | Final output size |
| `CE_WINDOW` | 200 | Cross-encoder re-rank window |
| `TOP_K_RETRIEVAL` | 2000 | Retrieval pool size |
| `JD_YOE_MIN` | 5 | Minimum preferred YoE |
| `JD_YOE_PREFERRED` | 7 | Peak of YoE fit triangle |

Title scoring:
- `TITLE_STRONG_POS`: Senior AI Engineer, Staff ML Engineer → +1.0
- `TITLE_POS`: ML Engineer, Data Scientist → +0.7
- `TITLE_ADJACENT`: Data Engineer, Backend Engineer → +0.4
- `TITLE_STRONG_NEG`: Marketing, HR, QA, Civil Eng → -1.0

## Output Format

`submission.csv`:
```
candidate_id,rank,score,reasoning
"CAND_0002025",1,0.985,"Senior AI Engineer with 5.9 yrs; 6+ yrs in applied ML/AI roles; skills: FAISS, TensorFlow, scikit-learn; 15 AI-domain terms in career description; open to work, active in last 30 days."
"CAND_0033861",2,0.9572,"Senior NLP Engineer with 8.0 yrs; 8+ yrs in applied ML/AI roles; skills: Reinforcement Learning, Weaviate, LoRA; 22 AI-domain terms in career description; standard profile."
...
```

## Validation & Sanity Checks

Phase 9 ensures:
- ✓ Exactly 100 rows
- ✓ Unique candidate_ids
- ✓ Scores strictly descending
- ✓ All scores in [0, 1]
- ✓ 4 decimal places per score
- ✓ No trap titles (title_ai_score ≤ -0.5)
- ✓ No services-only careers
- ✓ No honeypot candidates
- ✓ ≥80% strong-positive or adjacent titles

## Performance

End-to-end runtime on standard Colab CPU (~5 min):
- Phase 1 (JD parsing): ~1 sec
- Phase 2 (feature engineering): ~30 sec
- Phase 3 (BM25 + dense + RRF): ~2 min (embedding batch encoding)
- Phase 4 (retrieval selection): ~1 sec
- Phase 5 (XGBoost training): ~5 sec
- Phase 6 (cross-encoder): ~30 sec
- Phases 7-9: ~10 sec
- **Total: ~4-5 min**

Memory: ~1.5 GB for 100K candidates (100K × 40 features × ~150 bytes)

## Comparison to v2.0

| Aspect | v2.0 | v3.0 |
|--------|------|------|
| AI skill detection | Substring counter (trap!) | Exact-match canon (100+ terms) |
| Silver labels | Random `np.random.randint()` | Real heuristic score → 5-level |
| JD vector | Random `np.random.rand()` | Real docx parsing + tokenization |
| BM25 keywords | Hardcoded ["python", "go", ...] | Full candidate profile text |
| Hard filter | None | Honeypots, services-only, trap titles |
| Title scoring | None | -1.0 to +1.0 per category |
| Career evidence | None | ai_role_months, ai_role_pct, desc_ai_n |
| Behavioral weighting | Minimal | 10 signals (RR, LA, NP, OTW, etc.) |
| LTR training | None | XGBoost rank:ndcg (200 rounds) |
| Reasoning | Generic keywords | Real fields only (anti-hallucination) |
| **Result on v2.0 data** | **97/100 traps** | **0/100 traps** |

## Future Improvements

1. **Online learning**: Retrain LTR model on real feedback signals
2. **Hierarchical ranking**: Different rules for different seniority levels
3. **Geographic preference**: Stronger weighting for Pune/Noida candidates
4. **Skills taxonomy**: Map synonyms (e.g., "DL" → "deep learning")
5. **A/B testing framework**: Compare retrieval + LTR variants
6. **Ensemble re-ranking**: Combine CE with other models

## Contact

For issues or questions, refer to the notebook `improved_new_v2_fixed.ipynb` which contains the same logic with inline documentation and step-by-step execution traces.

---

**Version**: 3.0.0
**Date**: 2026-06-17
**Status**: Production-ready, tested on 100K candidates
