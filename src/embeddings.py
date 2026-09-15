"""
embeddings.py

Wraps the multilingual-e5-base sentence embedding model.

intfloat/multilingual-e5-base expects a task-specific prefix on every
input string:
    - "query: "   for search queries
    - "passage: " for documents / passages being indexed

Omitting these prefixes measurably degrades retrieval quality for e5-family
models, so encode_queries() and encode_passages() apply them automatically.
Callers should never need to prepend prefixes themselves.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config

_model_instance: Optional[SentenceTransformer] = None


def load_embedding_model() -> SentenceTransformer:
    """
    Loads (or returns a cached instance of) the multilingual embedding model.

    The model is loaded once per process and reused across calls, since
    loading intfloat/multilingual-e5-base from disk/network takes several
    seconds and only needs to happen once. On first run, this downloads
    ~1.1GB from the Hugging Face Hub and caches it locally (typically under
    C:\\Users\\<you>\\.cache\\huggingface on Windows).
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    try:
        _model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{config.EMBEDDING_MODEL_NAME}'. "
            "This usually means either (1) no internet connection was "
            "available for the first-time model download, (2) a corporate "
            "proxy/firewall blocked the download from huggingface.co, or "
            "(3) 'sentence-transformers' / 'torch' are not installed "
            f"correctly. Original error: {exc}"
        ) from exc

    return _model_instance


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalizes each row so that FAISS IndexFlatIP (inner product) is
    mathematically equivalent to cosine similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # avoid division by zero in a degenerate case
    return vectors / norms


def _encode(texts: List[str], prefix: str) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot encode an empty list of texts.")

    invalid_positions = [
        i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()
    ]
    if invalid_positions:
        raise ValueError(
            f"Cannot encode empty or non-string text at index positions: "
            f"{invalid_positions}"
        )

    model = load_embedding_model()
    prefixed_texts = [f"{prefix}{t}" for t in texts]

    raw_embeddings = model.encode(
        prefixed_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,  # we normalize explicitly below
    )

    embeddings = raw_embeddings.astype("float32")
    embeddings = _l2_normalize(embeddings)
    return embeddings


def encode_queries(queries: List[str]) -> np.ndarray:
    """Encodes a list of search queries using the 'query: ' prefix."""
    return _encode(queries, config.EMBEDDING_QUERY_PREFIX)


def encode_passages(passages: List[str]) -> np.ndarray:
    """Encodes a list of knowledge base passages using the 'passage: ' prefix."""
    return _encode(passages, config.EMBEDDING_PASSAGE_PREFIX)


def encode_single_query(query: str) -> np.ndarray:
    """Convenience wrapper returning a single query embedding, shape (dim,)."""
    return encode_queries([query])[0]


if __name__ == "__main__":
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    print("(First run downloads the model -- this may take a few minutes.)")
    model = load_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded successfully. Embedding dimension: {dim}")

    sample_queries = [
        "What is the capital of India?",
        "भारत की राजधानी क्या है?",
        "ભારતની રાજધાની શું છે?",
    ]
    query_embeddings = encode_queries(sample_queries)
    print(f"\nEncoded {len(sample_queries)} sample queries. Shape: {query_embeddings.shape}")

    norms = np.linalg.norm(query_embeddings, axis=1)
    print(f"L2 norms after normalization (should all be ~1.0): {norms}")

    sample_passage = ["New Delhi is the capital of India."]
    passage_embedding = encode_passages(sample_passage)
    similarity = float(np.dot(query_embeddings[0], passage_embedding[0]))
    print(f"\nCosine similarity (EN query vs matching EN passage): {similarity:.4f}")