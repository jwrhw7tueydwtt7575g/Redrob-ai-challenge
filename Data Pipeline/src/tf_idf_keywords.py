"""
TF-IDF Keyword Extraction Module
Extract discriminative keywords from JD to avoid generic terms like 'transformers'
"""

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

log = logging.getLogger(__name__)


def extract_tfidf_keywords(jd_text: str, top_k: int = 30) -> list:
    """
    Extract top TF-IDF keywords from job description.
    Filters out generic stopwords and ensures discriminative keyword selection.
    
    Args:
        jd_text: Job description text
        top_k: Number of top keywords to extract
    
    Returns:
        List of top TF-IDF keywords (sorted by importance)
    """
    try:
        # Custom stopwords to filter out generic tech terms
        stopwords = {
            "the", "and", "for", "with", "to", "of", "in", "a", "an", "is",
            "are", "was", "were", "be", "been", "has", "have", "had", "on",
            "at", "by", "from", "as", "or", "not", "this", "that", "will",
            "can", "our", "you", "your", "their", "we", "us", "who", "which",
            "must", "should", "also", "work", "role", "team", "strong", "using",
            "use", "used", "experience", "ability", "skills", "good", "well",
            "it", "they", "them", "there", "what", "where", "when", "why",
            "include", "including", "like", "need", "required", "job", "position",
            "candidate", "applicant", "responsible", "support", "provide", "help",
            "create", "develop", "build", "design", "implement", "write", "code",
            "data", "system", "application", "software", "technology", "digital",
            "business", "company", "organization", "team", "people", "environment",
        }
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=list(stopwords),
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,
            max_df=0.95,
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([jd_text])
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort by score (descending)
        top_indices = np.argsort(scores)[::-1][:top_k]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        
        log.info(f"Extracted {len(keywords)} TF-IDF keywords from JD")
        log.debug(f"Top keywords: {keywords[:10]}")
        
        return keywords
    
    except Exception as e:
        log.warning(f"TF-IDF extraction failed: {e}. Falling back to simple extraction.")
        # Fallback: return empty list, let caller use default extraction
        return []


def match_keywords_to_candidate(candidate_text: str, jd_keywords: list, top_k: int = 4) -> list:
    """
    Match JD keywords that appear in candidate text.
    
    Args:
        candidate_text: Candidate enriched text
        jd_keywords: List of JD keywords from TF-IDF
        top_k: Maximum number of matched keywords to return
    
    Returns:
        List of matched keywords (up to top_k)
    """
    candidate_lower = candidate_text.lower()
    matched = []
    
    for keyword in jd_keywords:
        if keyword.lower() in candidate_lower:
            matched.append(keyword)
            if len(matched) >= top_k:
                break
    
    return matched
