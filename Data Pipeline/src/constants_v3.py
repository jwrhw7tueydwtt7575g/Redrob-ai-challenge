"""
Constants for v3.0 Redrob AI Challenge Ranking Pipeline.
Centralized definitions for all scoring constants, skill canons, company lists, and location buckets.
Extracted from improved_new_v2_fixed.ipynb Phase 0 (constants).
"""

from datetime import date

# ============================================================================
# REFERENCE DATE (for behavioral signal recency calculations)
# ============================================================================
TODAY = date(2026, 6, 17)

# ============================================================================
# HYPERPARAMETERS (Colab/local runtime)
# ============================================================================
TOP_K = 100                       # Final output size
CE_WINDOW = 200                   # Cross-encoder re-rank window
LTR_TRAIN_FRAC = 1.0              # Use entire surviving pool as one LTR group
TOP_K_RETRIEVAL = 2000            # Retrieval pool size after RRF ranking

# ============================================================================
# TITLE SCORING CONSTANTS
# ============================================================================
TITLE_STRONG_POS = {
    "senior ai engineer", "ai research engineer", "staff ml engineer",
    "principal ml engineer", "staff machine learning engineer",
    "principal machine learning engineer", "lead ai engineer",
    "lead ml engineer", "senior applied scientist", "staff applied scientist",
}

TITLE_POS = {
    "ml engineer", "ai engineer", "machine learning engineer",
    "data scientist", "senior software engineer (ml)",
    "ai specialist", "ai research", "research engineer (ai)",
    "applied scientist", "research scientist", "nlp engineer",
    "senior nlp engineer", "deep learning engineer",
}

TITLE_ADJACENT = {
    "data engineer", "analytics engineer", "backend engineer",
    "software engineer", "senior software engineer",
}

TITLE_STRONG_NEG = {
    "marketing manager", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "graphic designer", "content writer",
    "customer support", "sales executive", "operations manager",
    "project manager", "business analyst", "sales manager",
    "human resources", "recruiter", "talent acquisition",
    "admin", "office manager", "logistics", "warehouse",
    "quality assurance", "qa engineer", "test engineer",
    "electrical engineer", "electronics engineer",
    "production engineer", "manufacturing engineer",
    "account manager", "relationship manager", "branch manager",
    "fashion designer", "interior designer", "ui designer",
    "ux designer", "copywriter", "journalist", "editor",
    "teacher", "lecturer", "professor", "trainer",
    "doctor", "nurse", "pharmacist", "dentist",
    "lawyer", "legal", "advocate", "ca", "chartered accountant",
    "company secretary", "auditor",
    "bank officer", "banker", "financial analyst",
    "mechanic", "technician", "operator", "driver",
    "chef", "cook", "waiter", "bartender", "hotel",
    "receptionist", "front desk", "office assistant",
    "data entry", "back office", "bpo", "voice", "process associate",
    "civil", "mechanical", "electrical",
    "marketing", "sales", "hr ", "human resources",
}

TITLE_TOKENS_TO_CHECK = [
    "engineer", "developer", "scientist", "analyst", "designer", "writer",
    "manager", "recruiter", "support", "sales", "accountant", "mechanic",
    "technician", "operator", "driver", "chef", "nurse", "doctor", "lawyer",
    "teacher", "professor", "consultant", "architect", "lead", "head",
    "founder", "co-founder", "researcher", "specialist",
]

# ============================================================================
# COMPANY & INDUSTRY LISTS
# ============================================================================
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "ltimindtree", "hcl", "tech mahindra", "mphasis",
    "persistent", "genpact", "ibm global services", "ibm india",
    "larsen & toubro infotech", "larsen toubro", "l&t infotech",
    "syntel", "atos", "dxc", "fis global", "fis", "igate",
    "patni", "hexaware", "mastek", "persistent systems",
    "tata consultancy", "tata elxsi", "tata communications",
    "wipro technologies", "infosys limited", "cognizant technology",
    "mindtree limited",
}

PRODUCT_INDUSTRIES = {
    "software", "ai/ml", "ai / ml", "ai", "ml", "saas", "fintech",
    "edtech", "e-commerce", "ecommerce", "healthtech", "health tech",
    "conversational ai", "adtech", "ad tech", "gaming", "media",
    "logistics tech", "foodtech", "food delivery", "travel",
    "consumer electronics", "consumer internet", "marketplace",
    "consumer products", "social", "iot", "robotics", "automotive",
    "cybersecurity", "data", "analytics",
}

# ============================================================================
# AI/ML SKILL CANON (Exact match, not substring)
# ============================================================================
AI_SKILL_CANON = {
    "machine learning", "deep learning", "neural networks", "neural network",
    "nlp", "nlu", "natural language processing", "natural language",
    "computer vision", "machine vision", "image classification",
    "object detection", "semantic segmentation", "image segmentation",
    "speech recognition", "asr", "text-to-speech", "tts",
    "llm", "large language models", "large language model",
    "transformers", "transformer", "bert", "gpt", "roberta",
    "llama", "mistral", "claude", "gemini", "chatgpt", "falcon",
    "fine-tuning", "fine tuning", "fine-tune", "finetuning",
    "lora", "qlora", "peft", "rlhf", "prompt engineering",
    "rag", "retrieval-augmented generation",
    "embeddings", "embedding", "sentence embeddings",
    "vector search", "semantic search", "vector database", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma",
    "elasticsearch", "opensearch", "vespa",
    "langchain", "llamaindex", "haystack",
    "huggingface", "hugging face", "transformers library",
    "tensorflow", "pytorch", "torch", "jax", "keras",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "learning to rank", "learning-to-rank", "lambdaMART", "lambdarank",
    "ndcg", "mrr", "map", "recall@k", "ndcg@k", "mrr@k",
    "recommendation system", "recommender system", "recommendation engine",
    "search ranking", "neural ranking", "learning to rank",
    "information retrieval", "ir", "text ranking", "text retrieval",
    "knowledge graph", "knowledge graphs",
    "openai", "anthropic", "cohere", "ai21",
    "stable diffusion", "diffusion models", "gan", "gans", "vaes",
    "reinforcement learning", "rl", "policy gradient", "q-learning",
    "spark ml", "spark mllib", "mahout", "graph neural networks",
    "gnn", "graph learning", "graph embeddings",
    "mlops", "mlflow", "kubeflow", "sagemaker", "vertex ai",
    "model deployment", "model serving", "model monitoring",
    "data labeling", "data annotation", "labeling",
    "feature store", "feast", "tecton",
    "model evaluation", "model interpretability", "explainability",
    "shap", "lime", "fairness", "ml safety",
    "speech synthesis", "voice cloning", "speaker recognition",
    "ocr", "optical character recognition",
    "translation", "machine translation", "neural translation",
    "summarization", "text summarization", "abstractive summarization",
    "question answering", "qa system", "chatbot", "conversational ai",
    "dialogue systems", "dialog systems",
    "aws sagemaker", "azure ml", "gcp vertex",
    "weights & biases", "wandb", "tensorboard",
    "ray", "bentoml", "seldon", "fastapi", "flask",
    "docker", "kubernetes", "airflow", "prefect", "dagster",
    "kafka", "spark", "hadoop", "flink", "beam",
    "snowflake", "bigquery", "redshift", "databricks", "dbt",
    "etl", "elt", "data pipeline", "data pipelines",
    "data engineering", "feature engineering",
    "python", "r ", " r,", "(r)", "julia",
    "pandas", "numpy", "scipy",
    "transformer models", "transformer architecture",
    "encoder-decoder", "encoder decoder", "seq2seq",
}

# ============================================================================
# AI/ML TERMS IN DESCRIPTIONS
# ============================================================================
DESCRIPTION_AI_TERMS = [
    "embedding", "embeddings", "retrieval", "reranking", "re-ranking",
    "ranking", "ranker", "fine-tun", "fine tun", "lora", "qlora", "peft",
    "rag", "retrieval-augmented", "retrieval augmented",
    "vector", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "sentence-transformer", "sentence transformer", "bge", "e5", "sbert",
    "llama", "mistral", "gpt", "bert", "roberta", "t5",
    "transformer", "transformers",
    "ndcg", "mrr", "map@", "ndcg@", "learning-to-rank", "learning to rank",
    "lambdaMART", "xgboost ranker", "ltr model",
    "recommendation system", "recommender", "search system", "search engine",
    "personalization", "personalisation",
    "neural network", "neural net", "deep learning", "deep model",
    "machine learning model", "ml model", "ml system", "ml pipeline",
    "model serving", "model deployment", "model inference",
    "feature store", "feature engineering", "feature pipeline",
    "training pipeline", "inference pipeline", "offline eval",
    "offline evaluation", "online experiment", "a/b test", "ab test",
    "evaluation framework", "eval framework",
    "airflow", "spark", "kafka", "kubernetes", "docker", "mlflow",
    "distributed training", "distributed inference", "model parallel",
    "data parallel", "parameter server", "ray", "horovod",
    "pytorch", "tensorflow", "jax", "huggingface",
    "llm", "large language", "language model", "foundation model",
    "nlp", "natural language", "computer vision",
]

# ============================================================================
# INDIA LOCATION BUCKETS
# ============================================================================
INDIA_PREFERRED_CITIES = {"pune", "noida"}
INDIA_WELCOME_CITIES = {
    "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
    "chennai", "bangalore", "bengaluru", "kolkata",
}
INDIA_ALL_CITIES = INDIA_PREFERRED_CITIES | INDIA_WELCOME_CITIES | {
    "ahmedabad", "jaipur", "lucknow", "indore", "bhopal", "nagpur",
    "visakhapatnam", "vizag", "coimbatore", "kochi", "trivandrum",
    "thiruvananthapuram", "chandigarh", "surat", "vadodara",
}

# ============================================================================
# JD-SPECIFIC PARAMETERS
# ============================================================================
JD_YOE_MIN = 5
JD_YOE_PREFERRED = 7  # Triangle peak for YoE fit scoring
