"""
retriever.py

Builds a FAISS vector index over the multilingual knowledge base and
provides top-k retrieval for a given query and language.

Index type: faiss.IndexFlatIP (exact inner-product search). Since all
embeddings are L2-normalized (see embeddings.py), inner product is
mathematically equivalent to cosine similarity. IndexFlatIP is an exact,
brute-force index -- appropriate here because the knowledge base has only
54 documents. There is no practical benefit to an approximate index
(e.g. IVF, HNSW) at this scale, and exact search keeps results fully
reproducible.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

from src import config, data_loader, embeddings


@dataclass
class RetrievedDocument:
    document_id: str
    content: str
    category: str
    language: str
    source: str
    linked_fact_id: Optional[str]
    is_distractor: Optional[bool]
    similarity_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    """
    Wraps a FAISS IndexFlatIP index over the knowledge base, with helpers
    to build, persist, load, and query it.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Index construction / persistence
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Builds the FAISS index from the knowledge base and saves it to disk,
        along with document metadata. If a cached index already exists and
        force_rebuild is False, loads from cache instead of recomputing.

        If you edit knowledge_base.csv, call build_index(force_rebuild=True)
        or delete the files under data/processed/ so the index is rebuilt.
        """
        if not force_rebuild and self._cache_exists():
            print("Cached FAISS index found. Loading from disk...")
            self.load_index()
            return

        print("Building FAISS index from knowledge base...")
        kb_df = data_loader.load_knowledge_base()

        passages = kb_df["content"].tolist()
        passage_embeddings = embeddings.encode_passages(passages)

        dimension = passage_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(passage_embeddings)

        self.index = index
        self.metadata = kb_df.reset_index(drop=True)

        self._save_cache(passage_embeddings)
        print(
            f"Built and cached FAISS index with {index.ntotal} documents "
            f"(dimension={dimension})."
        )

    def load_index(self) -> None:
        """Loads a previously built FAISS index and metadata from disk."""
        if not self._cache_exists():
            raise FileNotFoundError(
                "No cached FAISS index found. Run build_index() first "
                "(e.g. `python -m src.retriever`)."
            )

        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        with open(config.KB_METADATA_CACHE_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def _cache_exists(self) -> bool:
        return (
            config.FAISS_INDEX_PATH.exists()
            and config.KB_METADATA_CACHE_PATH.exists()
        )

    def _save_cache(self, passage_embeddings: np.ndarray) -> None:
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        with open(config.KB_METADATA_CACHE_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        np.save(config.EMBEDDINGS_CACHE_PATH, passage_embeddings)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        language: str,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> List[RetrievedDocument]:
        """
        Retrieves the top_k most similar knowledge base documents for a
        query, restricted to documents in the same language as the query.

        Restricting to the query's language is intentional: retrieved
        context is later handed to the LLM to generate an answer in that
        same language, so cross-language context would be operationally
        useless even if semantically similar.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError(
                "Index not loaded. Call build_index() or load_index() first."
            )
        if language not in config.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Must be one of "
                f"{config.SUPPORTED_LANGUAGES}."
            )
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = embeddings.encode_single_query(query).reshape(1, -1)

        # Search the full index, then filter to the requested language.
        # The knowledge base is small (54 docs), so searching everything
        # and filtering afterward is simpler than maintaining separate
        # per-language indices, with no meaningful performance cost.
        search_k = self.index.ntotal
        scores, indices = self.index.search(query_embedding, search_k)

        results: List[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            if row["language"] != language:
                continue
            results.append(
                RetrievedDocument(
                    document_id=row["document_id"],
                    content=row["content"],
                    category=row["category"],
                    language=row["language"],
                    source=row["source"],
                    linked_fact_id=row.get("linked_fact_id"),
                    is_distractor=bool(row.get("is_distractor", False)),
                    similarity_score=float(score),
                )
            )
            if len(results) == top_k:
                break

        return results


def _print_retrieval_result(
    question: str, language: str, results: List[RetrievedDocument]
) -> None:
    print(f"\nQuery ({language}): {question}")
    print("-" * 70)
    if not results:
        print("  No documents retrieved.")
        return
    for rank, doc in enumerate(results, start=1):
        distractor_tag = " [DISTRACTOR]" if doc.is_distractor else ""
        print(
            f"  Rank {rank} | score={doc.similarity_score:.4f} | "
            f"doc_id={doc.document_id} | category={doc.category}"
            f"{distractor_tag}"
        )
        print(f"           content: {doc.content}")


if __name__ == "__main__":
    retriever = Retriever()
    retriever.build_index()

    _print_retrieval_result(
        "What is the capital of India?",
        "English",
        retriever.retrieve("What is the capital of India?", "English", top_k=3),
    )
    _print_retrieval_result(
        "भारत की राजधानी क्या है?",
        "Hindi",
        retriever.retrieve("भारत की राजधानी क्या है?", "Hindi", top_k=3),
    )
    _print_retrieval_result(
        "ભારતની રાજધાની શું છે?",
        "Gujarati",
        retriever.retrieve("ભારતની રાજધાની શું છે?", "Gujarati", top_k=3),
    )