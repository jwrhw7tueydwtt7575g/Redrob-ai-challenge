"""
Main Pipeline Orchestrator
Chains all 9 phases together for end-to-end execution.
"""

import logging
import time
from typing import Dict

import torch

from constants_v3 import TOP_K, CE_WINDOW, TOP_K_RETRIEVAL
from phase1_jd_ingestion import parse_jd, tokenize_for_bm25
from phase2_feature_engineering import stream_candidates_with_features
from phase3_hard_prefilter import run_phase3
from phase4_retrieval_ranking import run_phase4
from phase5_ltr_training import run_phase5
from phase6_cross_encoder_rerank import run_phase6
from phase7_reasoning_generation import run_phase7
from phase8_csv_generation import run_phase8
from phase9_sanity_checks import run_phase9

log = logging.getLogger("redrob.pipeline")

# ============================================================================
# Main orchestrator
# ============================================================================
def run_pipeline(candidates_path: str, jd_path: str, output_path: str,
                 device: str = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Dict:
    """
    Execute the full 9-phase ranking pipeline.
    
    Args:
        candidates_path: Path to candidates.jsonl
        jd_path: Path to job_description.docx
        output_path: Path to output submission.csv
        device: Torch device (auto-detect if None)
        embedding_model: Sentence-transformer model name
        ce_model: Cross-encoder model name
    
    Returns:
        Dictionary with pipeline results
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    t0 = time.time()
    
    try:
        # ====================================================================
        # PHASE 1: JD Ingestion
        # ====================================================================
        jd_result = parse_jd(jd_path)
        jd_query_text = jd_result["query_text"]
        jd_ce_text = jd_result["ce_text"]
        jd_tokens = jd_result["tokens"]
        
        # ====================================================================
        # PHASE 2: Feature Engineering
        # ====================================================================
        records = stream_candidates_with_features(candidates_path)
        
        # ====================================================================
        # PHASE 3: Hard Pre-Filter + Retrieval
        # ====================================================================
        phase3_result = run_phase3(records, jd_query_text, jd_tokens,
                                  model_name=embedding_model, batch_size=32, device=device)
        surviving_records = phase3_result["surviving_records"]
        bm25_scores = phase3_result["bm25_scores"]
        bm25_n = phase3_result["bm25_n"]
        dense_scores = phase3_result["dense_scores"]
        dense_n = phase3_result["dense_n"]
        rrf_scores = phase3_result["rrf_scores"]
        
        # ====================================================================
        # PHASE 4: Retrieval Ranking
        # ====================================================================
        phase4_result = run_phase4(surviving_records, bm25_scores, bm25_n,
                                  dense_scores, dense_n, rrf_scores,
                                  top_k_retrieval=TOP_K_RETRIEVAL)
        retrieval_records = phase4_result["retrieval_records"]
        retrieval_bm25_n = phase4_result["retrieval_bm25_n"]
        retrieval_dense_n = phase4_result["retrieval_dense_n"]
        retrieval_rrf_n = phase4_result["retrieval_rrf_n"]
        
        # ====================================================================
        # PHASE 5: LTR Training
        # ====================================================================
        phase5_result = run_phase5(retrieval_records, retrieval_bm25_n,
                                  retrieval_dense_n, retrieval_rrf_n)
        ltr_scores = phase5_result["ltr_scores"]
        
        # ====================================================================
        # PHASE 6: Cross-Encoder Re-Ranking
        # ====================================================================
        phase6_result = run_phase6(retrieval_records, ltr_scores, jd_ce_text,
                                  ce_window=CE_WINDOW, top_k=TOP_K,
                                  model_name=ce_model, device=device)
        ce_records = phase6_result["ce_records"]
        scores_final = phase6_result["scores_final"]
        
        # ====================================================================
        # PHASE 7: Reasoning Generation
        # ====================================================================
        phase7_result = run_phase7(ce_records)
        reasonings = phase7_result["reasonings"]
        
        # ====================================================================
        # PHASE 8: CSV Generation
        # ====================================================================
        phase8_result = run_phase8(ce_records, scores_final, reasonings, output_path)
        df = phase8_result["dataframe"]
        
        if not phase8_result["success"]:
            log.error("Phase 8 failed. Aborting pipeline.")
            return {"success": False, "error": "CSV validation failed"}
        
        # ====================================================================
        # PHASE 9: Sanity Checks
        # ====================================================================
        phase9_result = run_phase9(ce_records, df)
        
        # ====================================================================
        # Summary
        # ====================================================================
        elapsed = time.time() - t0
        log.info("\n" + "=" * 70)
        log.info("PIPELINE COMPLETE")
        log.info("=" * 70)
        log.info(f"  Total elapsed: {elapsed:.1f}s")
        log.info(f"  Output: {output_path}")
        log.info(f"  Submission: {len(df)} rows, all valid ✓")
        
        return {
            "success": True,
            "elapsed_seconds": elapsed,
            "output_path": output_path,
            "dataframe": df,
            "sanity_checks": phase9_result,
            "phases": {
                "phase1": jd_result,
                "phase2": {"n_records": len(records)},
                "phase3": {"n_surviving": len(surviving_records)},
                "phase4": {"n_retrieval": len(retrieval_records)},
                "phase5": {"ltr_trained": True},
                "phase6": {"n_final": len(ce_records)},
                "phase7": {"reasonings_generated": len(reasonings)},
                "phase8": {"csv_saved": True},
                "phase9": phase9_result,
            }
        }
    
    except Exception as e:
        log.error(f"Pipeline failed with error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "elapsed_seconds": time.time() - t0,
        }

# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run v3.0 ranking pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx")
    parser.add_argument("--output", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Cross-encoder model name")
    
    args = parser.parse_args()
    
    result = run_pipeline(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_path=args.output,
        device=args.device,
        embedding_model=args.embedding_model,
        ce_model=args.ce_model,
    )
    
    if result["success"]:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Output: {result['output_path']}")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        exit(1)
