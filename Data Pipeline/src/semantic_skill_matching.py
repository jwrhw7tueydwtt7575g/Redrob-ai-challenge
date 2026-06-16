"""
Semantic Skill Matching Module
Match skills using embeddings (e.g., "neural networks" ≈ "deep learning")
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util

log = logging.getLogger(__name__)

# Skill groups (semantically similar)
SKILL_FAMILIES = {
    'deep_learning': ['deep learning', 'neural networks', 'convolutional', 'cnn', 'rnn', 'lstm', 'transformer'],
    'machine_learning': ['machine learning', 'ml', 'supervised learning', 'unsupervised', 'classification', 'regression'],
    'python': ['python', 'py', 'python3', 'cpython', 'pandas', 'numpy', 'scipy'],
    'tensorflow': ['tensorflow', 'tf', 'keras', 'tfx'],
    'pytorch': ['pytorch', 'torch', 'fastai'],
    'data_processing': ['data processing', 'etl', 'pipelines', 'spark', 'hadoop', 'mapreduce'],
    'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'cassandra', 'elasticsearch'],
    'cloud_aws': ['aws', 'amazon', 'ec2', 's3', 'sagemaker', 'lambda'],
    'cloud_gcp': ['gcp', 'google cloud', 'bigquery', 'cloud ml'],
    'cloud_azure': ['azure', 'microsoft azure', 'cosmos db'],
    'containers': ['docker', 'kubernetes', 'k8s', 'container', 'orchestration'],
    'devops': ['devops', 'ci/cd', 'jenkins', 'gitlab', 'github actions', 'terraform'],
    'nlp': ['nlp', 'natural language', 'bert', 'gpt', 'transformers', 'language model'],
    'computer_vision': ['computer vision', 'cv', 'image', 'object detection', 'yolo'],
}


def create_skill_embedding_index(embedding_model) -> dict:
    """
    Create embeddings for all skill families.
    
    Returns:
        Dict: skill_family -> embedding
    """
    skill_embeddings = {}
    
    for family_name, skills in SKILL_FAMILIES.items():
        # Embed all skills in family
        embeddings = embedding_model.encode(skills, convert_to_numpy=True)
        # Average embedding for family
        family_embedding = embeddings.mean(axis=0)
        skill_embeddings[family_name] = family_embedding
    
    log.info(f"Created embeddings for {len(skill_embeddings)} skill families")
    
    return skill_embeddings


def compute_semantic_skill_match(
    candidate_skills: list,
    jd_requirements: dict,
    skill_embeddings: dict,
    embedding_model
) -> float:
    """
    Compute skill match using semantic similarity.
    
    Args:
        candidate_skills: List of candidate's skills
        jd_requirements: Dict with required_skills, nice_to_have_skills
        skill_embeddings: Pre-computed skill family embeddings
        embedding_model: SentenceTransformer model
    
    Returns:
        Semantic match score (0.0-1.0)
    """
    if not candidate_skills or not jd_requirements.get('required_skills'):
        return 0.5
    
    # Embed candidate skills
    candidate_embeddings = embedding_model.encode(candidate_skills, convert_to_numpy=True)
    
    # Embed JD requirements
    required_embeddings = embedding_model.encode(
        jd_requirements['required_skills'],
        convert_to_numpy=True
    )
    
    # Compute similarities
    similarities = util.cos_sim(candidate_embeddings, required_embeddings).cpu().numpy()
    
    # For each required skill, find best candidate skill match
    best_matches = similarities.max(axis=0)
    required_match_score = best_matches.mean()
    
    # Nice-to-have bonus
    if jd_requirements.get('nice_to_have_skills'):
        nice_embeddings = embedding_model.encode(
            jd_requirements['nice_to_have_skills'],
            convert_to_numpy=True
        )
        nice_similarities = util.cos_sim(candidate_embeddings, nice_embeddings).cpu().numpy()
        nice_matches = nice_similarities.max(axis=0)
        nice_score = nice_matches.mean()
        bonus = min(0.2, nice_score * 0.2)
    else:
        bonus = 0
    
    total_score = min(1.0, required_match_score + bonus)
    
    return total_score
