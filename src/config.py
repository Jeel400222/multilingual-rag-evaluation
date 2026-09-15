"""
config.py

Centralized configuration for the Multilingual RAG Evaluation &
LLM Hallucination Detection System.

Every other module in this project imports paths, constants, and
settings from here rather than hardcoding them locally.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
API_KEY = os.getenv("API_KEY", "")

# ---------------------------------------------------------------------------
# Project root and directory paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Specific file paths used across modules
QUESTIONS_CSV = RAW_DATA_DIR / "questions.csv"
KNOWLEDGE_BASE_CSV = KNOWLEDGE_BASE_DIR / "knowledge_base.csv"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index.index"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "kb_embeddings.npy"
KB_METADATA_CACHE_PATH = PROCESSED_DATA_DIR / "kb_metadata.pkl"
EXPERIMENT_RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"

# Ensure all required directories exist (safe to call multiple times)
for directory in [
    RAW_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    CHARTS_DIR,
    TABLES_DIR,
    NOTEBOOKS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ["English", "Hindi", "Gujarati"]

# ISO-style short codes used internally in data files (question_id prefixes,
# knowledge base language column, etc.)
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
}

# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

SUPPORTED_CATEGORIES = ["Geography", "History", "Science"]

# ---------------------------------------------------------------------------
# Embedding / retrieval configuration
# ---------------------------------------------------------------------------

# intfloat/multilingual-e5-base requires "query: " / "passage: " prefixes
# on inputs. This is handled in embeddings.py, but the constants live here
# so retriever.py and embeddings.py stay in sync.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "

# Number of top documents to retrieve per query
TOP_K_RETRIEVAL = 3

# ---------------------------------------------------------------------------
# Hallucination detection configuration
# ---------------------------------------------------------------------------
#
# IMPORTANT (academic honesty note):
# These thresholds define a SEMANTIC SIMILARITY HEURISTIC, not logical
# entailment. A claim scoring above SUPPORTED_THRESHOLD is semantically
# close to some retrieved passage — it is NOT verified as logically
# entailed by that passage. This distinction is documented here and
# surfaced in the UI / README so results are not overstated.
#
# These starting values are placeholders. In Phase 5, we will inspect the
# actual similarity score distribution on real claims and recalibrate
# these thresholds based on that data, not guesswork.

CLAIM_SUPPORTED_THRESHOLD = 0.80
CLAIM_PARTIAL_THRESHOLD = 0.65
# Below CLAIM_PARTIAL_THRESHOLD -> classified as Unsupported
# (An additional "Unable to Determine" state is reserved for cases such as
#  empty claims or retrieval failures, handled in claim_analyzer.py)

# ---------------------------------------------------------------------------
# Evaluation labels (fixed vocabulary — do not invent new labels elsewhere)
# ---------------------------------------------------------------------------

EVALUATION_LABELS = [
    "Correct",
    "Partially Correct",
    "Incorrect",
    "Unsupported-Hallucinated",
    "Unable to Evaluate",
]

CLAIM_STATUS_LABELS = [
    "Supported",
    "Partially Supported",
    "Unsupported",
    "Unable to Determine",
]

HALLUCINATION_RISK_LEVELS = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Groq API configuration
# ---------------------------------------------------------------------------

GROQ_MAX_RETRIES = 3
GROQ_TIMEOUT_SECONDS = 30
GROQ_TEMPERATURE = 0.0  # deterministic-ish factual responses
GROQ_MAX_TOKENS = 512


def validate_config() -> None:
    """
    Validates that critical configuration is present before the app
    or any pipeline component runs. Raises a clear, user-friendly
    error instead of failing deep inside an API call.
    """
    if not API_KEY or API_KEY == "your_api_key_here":
        raise ValueError(
            "Missing or invalid API_KEY. Please create a '.env' file "
            "(copy '.env.example' to '.env') and set your real Groq "
            "API key in the API_KEY field."
        )
    if LLM_PROVIDER.lower() != "groq":
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
            "This project currently only supports 'groq'."
        )
        
# ---------------------------------------------------------------------------
# Phase 3 additions: retrieval quality check output paths
# ---------------------------------------------------------------------------
RETRIEVAL_QUALITY_DETAILS_CSV = TABLES_DIR / "retrieval_quality_details.csv"
RETRIEVAL_QUALITY_SUMMARY_CSV = TABLES_DIR / "retrieval_quality_summary.csv"

# ---------------------------------------------------------------------------
# Phase 4 addition: retry backoff for Groq API calls
# ---------------------------------------------------------------------------
GROQ_RETRY_BACKOFF_SECONDS = 2  # base delay; actual wait = base * (2 ** attempt)

# ---------------------------------------------------------------------------
# Phase 5 additions: hallucination risk thresholds
# ---------------------------------------------------------------------------
# Risk bucket is based on the proportion of a response's determinate claims
# (i.e. excluding "Unable to Determine") classified as Unsupported.
HALLUCINATION_RISK_LOW_MAX_UNSUPPORTED_RATE = 0.20
HALLUCINATION_RISK_MEDIUM_MAX_UNSUPPORTED_RATE = 0.50

# ---------------------------------------------------------------------------
# Phase 6 additions: correctness matching + evaluation output paths
# ---------------------------------------------------------------------------
# Fraction of the reference answer's significant words that must appear in
# the response for "Relaxed" correctness. A documented heuristic threshold,
# not a statistically fitted value.
RELAXED_TOKEN_OVERLAP_THRESHOLD = 0.6

RESPONSE_METRICS_SUMMARY_CSV = TABLES_DIR / "response_metrics_summary.csv"
ACCURACY_BY_LANGUAGE_CSV = TABLES_DIR / "accuracy_by_language.csv"
ACCURACY_BY_CATEGORY_CSV = TABLES_DIR / "accuracy_by_category.csv"
RAG_METRICS_SUMMARY_CSV = TABLES_DIR / "rag_metrics_summary.csv"
HALLUCINATION_RISK_DISTRIBUTION_CSV = TABLES_DIR / "hallucination_risk_distribution.csv"