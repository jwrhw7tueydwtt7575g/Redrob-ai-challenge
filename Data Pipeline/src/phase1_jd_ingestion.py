"""
Phase 1: JD Ingestion & Structured Parsing
Extracted from improved_new_v2_fixed.ipynb Phase 1 cells.
Reads the real .docx file, extracts sections, and prepares texts for retrieval.
"""

import re
import logging
from typing import Dict, List, Tuple

log = logging.getLogger("redrob.phase1")

# ============================================================================
# 1a. Read the real .docx
# ============================================================================
def read_docx_text(path: str) -> str:
    """Read all paragraph text from a .docx, preserving order."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text is not None).strip()
    except Exception as e:
        log.warning(f"Failed to read DOCX: {e}. Returning empty string.")
        return ""

# ============================================================================
# 1b. Section splitter
# ============================================================================
SECTIONS = {
    "must_have": "Things you absolutely need",
    "nice_to_have": "Things we'd like you to have but won't reject you for",
    "do_not_want": "Things we explicitly do NOT want",
    "ideal_candidate": "How to read between the lines",
    "logistics": "On location, comp, and logistics",
    "yoe": "What we mean by",
    "role": "What you'd actually be doing",
}

def split_into_sections(text: str) -> Dict[str, str]:
    """Return a dict of section_key -> section_text. Unrecognized text is dropped."""
    out = {k: "" for k in SECTIONS}
    lines = text.split("\n")
    current_key = None
    for line in lines:
        matched = None
        for key, header_phrase in SECTIONS.items():
            if header_phrase.lower() in line.lower() and len(line) < 120:
                matched = key
                break
        if matched is not None:
            current_key = matched
            continue
        if current_key is not None:
            out[current_key] += line + "\n"
    return out

# ============================================================================
# 1c. YoE range extraction
# ============================================================================
def extract_yoe_range(text: str) -> Tuple[int, int]:
    """Return (min_yoe, preferred_yoe) parsed from '5-9 years' style phrases."""
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\s*\+\s*years", text)
    if m:
        return int(m.group(1)), int(m.group(1)) + 3
    return 4, 9

# ============================================================================
# 1d. Must-have skill groups
# ============================================================================
def extract_jd_must_have_bullets(must_have_text: str) -> List[str]:
    """Split a JD section into individual bullet lines."""
    out = []
    for line in must_have_text.split("\n"):
        s = line.strip(" -\t•●*0123456789.").strip()
        if len(s) > 10 and not s.lower().startswith(("things", "production")):
            out.append(s)
    return out

JD_MUST_HAVE_GROUPS = {
    "embedding_retrieval": [
        "embeddings-based retrieval", "embedding drift", "index refresh",
        "retrieval-quality regression", "sentence-transformers", "openai embeddings",
        "bge", "e5",
    ],
    "vector_db": [
        "vector databases", "hybrid search", "pinecone", "weaviate", "qdrant",
        "milvus", "opensearch", "elasticsearch", "faiss",
    ],
    "python": ["strong python", "code quality"],
    "eval_ir": [
        "evaluation framework", "ndcg", "mrr", "map", "offline-to-online",
        "offline to online", "a/b test", "ranking system",
    ],
}

JD_NICE_GROUPS = {
    "llm_finetune": ["llm fine-tuning", "lora", "qlora", "peft"],
    "ltr_xgb": ["learning-to-rank", "xgboost", "neural"],
    "hr_tech": ["hr-tech", "recruiting tech", "marketplace"],
    "distributed": ["distributed systems", "large-scale inference"],
    "oss": ["open-source", "ai/ml space"],
}

# ============================================================================
# 1e. Disqualifier patterns
# ============================================================================
DISQUALIFIER_PATTERNS = [
    r"title-chasers?",
    r"framework enthusiasts?",
    r"consulting firms?",
    r"computer vision,? speech,? or robotics",
    r"closed-source proprietary",
    r"18 months",
    r"12 months.*langchain",
    r"pure research",
]

# ============================================================================
# 1f. Build JD texts for retrieval
# ============================================================================
def build_jd_query_text(full_text: str, must_have_bullets: List[str], sec: Dict[str, str]) -> str:
    """Compose a focused JD text suitable for BM25 + dense retrieval.
    Emphasizes must-haves so retrieval is biased toward what the JD actually wants."""
    parts = [
        "Senior AI Engineer. Founding team. Series A. Hybrid retrieval, ranking, LLMs.",
        " ".join(must_have_bullets),
        sec.get("ideal_candidate", ""),
        sec.get("must_have", ""),
    ]
    return " ".join(p for p in parts if p).strip()

def build_jd_full_text_for_ce(full_text: str) -> str:
    """Use a truncated version of the full JD for cross-encoder pairs
    (ms-marco-MiniLM-L-6-v2 has a 512-token limit)."""
    return full_text[:6000]

# ============================================================================
# 1g. BM25 tokenizer
# ============================================================================
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")

def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase, alphanum + hyphen, drop 1-2 char tokens."""
    return _TOKEN_RE.findall(text.lower())

# ============================================================================
# Main: parse JD end-to-end
# ============================================================================
def parse_jd(jd_path: str) -> Dict:
    """Parse a JD .docx file and return all extracted components."""
    jd_text = read_docx_text(jd_path)
    if not jd_text:
        log.warning(f"JD file at {jd_path} is empty or unreadable.")
        return {
            "full_text": "",
            "sections": {k: "" for k in SECTIONS},
            "yoe_min": 5,
            "yoe_preferred": 7,
            "must_have_bullets": [],
            "query_text": "",
            "ce_text": "",
            "tokens": [],
        }
    
    sec = split_into_sections(jd_text)
    yoe_min, yoe_preferred = extract_yoe_range(jd_text)
    must_have = extract_jd_must_have_bullets(sec["must_have"])
    query_text = build_jd_query_text(jd_text, must_have, sec)
    ce_text = build_jd_full_text_for_ce(jd_text)
    tokens = tokenize_for_bm25(query_text)
    
    log.info(f"  ✓ JD parsed: {len(jd_text)} chars, YoE {yoe_min}-{yoe_preferred}, {len(must_have)} bullets")
    
    return {
        "full_text": jd_text,
        "sections": sec,
        "yoe_min": yoe_min,
        "yoe_preferred": yoe_preferred,
        "must_have_bullets": must_have,
        "query_text": query_text,
        "ce_text": ce_text,
        "tokens": tokens,
    }
