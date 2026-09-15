"""
rag_pipeline.py

Orchestrates a single end-to-end pipeline run for one question:
retrieval -> Non-RAG generation -> RAG generation -> claim analysis for
both -> (optional) correctness checking against a reference answer.

This is the module the Streamlit app calls for interactive "Try the
System" queries. It mirrors the steps in evaluator.py (which runs the
same logic in batch across the fixed 45-question dataset) but treats a
reference answer as optional, since interactive questions typed by a
user may have no known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src import config, llm_client
from src.evaluator import evaluate_correctness, assign_evaluation_label
from src.hallucination_detector import assess_response, ResponseGroundednessResult
from src.llm_client import LLMResponse
from src.retriever import Retriever, RetrievedDocument


@dataclass
class PipelineResult:
    question: str
    language: str
    retrieved_documents: List[RetrievedDocument]
    non_rag_response: LLMResponse
    rag_response: LLMResponse
    non_rag_groundedness: Optional[ResponseGroundednessResult]
    rag_groundedness: Optional[ResponseGroundednessResult]
    reference_answer: Optional[str]
    non_rag_strict_correct: Optional[bool]
    non_rag_relaxed_correct: Optional[bool]
    rag_strict_correct: Optional[bool]
    rag_relaxed_correct: Optional[bool]
    non_rag_evaluation_label: Optional[str]
    rag_evaluation_label: Optional[str]


def run_pipeline(
    retriever: Retriever,
    question: str,
    language: str,
    reference_answer: Optional[str] = None,
    top_k: int = config.TOP_K_RETRIEVAL,
) -> PipelineResult:
    """
    Runs the full pipeline for one question. Raises ValueError for an
    empty question or unsupported language -- callers (e.g. the Streamlit
    app) should catch this and show a friendly message rather than crash.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if language not in config.SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Must be one of {config.SUPPORTED_LANGUAGES}."
        )

    retrieved_documents = retriever.retrieve(question, language, top_k=top_k)
    context_texts = [d.content for d in retrieved_documents]

    non_rag_response = llm_client.generate_non_rag_response(question, language)
    rag_response = llm_client.generate_rag_response(question, language, context_texts)

    non_rag_groundedness = (
        assess_response(non_rag_response.response_text, retrieved_documents)
        if non_rag_response.success
        else None
    )
    rag_groundedness = (
        assess_response(rag_response.response_text, retrieved_documents)
        if rag_response.success
        else None
    )

    non_rag_strict = non_rag_relaxed = None
    rag_strict = rag_relaxed = None
    non_rag_label = rag_label = None

    if reference_answer:
        if non_rag_response.success:
            non_rag_strict, non_rag_relaxed = evaluate_correctness(
                non_rag_response.response_text, reference_answer
            )
        non_rag_risk = (
            non_rag_groundedness.hallucination_risk if non_rag_groundedness else "Unable to Determine"
        )
        non_rag_label = assign_evaluation_label(
            non_rag_response.success, bool(non_rag_strict), bool(non_rag_relaxed), non_rag_risk
        )

        if rag_response.success:
            rag_strict, rag_relaxed = evaluate_correctness(
                rag_response.response_text, reference_answer
            )
        rag_risk = (
            rag_groundedness.hallucination_risk if rag_groundedness else "Unable to Determine"
        )
        rag_label = assign_evaluation_label(
            rag_response.success, bool(rag_strict), bool(rag_relaxed), rag_risk
        )

    return PipelineResult(
        question=question,
        language=language,
        retrieved_documents=retrieved_documents,
        non_rag_response=non_rag_response,
        rag_response=rag_response,
        non_rag_groundedness=non_rag_groundedness,
        rag_groundedness=rag_groundedness,
        reference_answer=reference_answer,
        non_rag_strict_correct=non_rag_strict,
        non_rag_relaxed_correct=non_rag_relaxed,
        rag_strict_correct=rag_strict,
        rag_relaxed_correct=rag_relaxed,
        non_rag_evaluation_label=non_rag_label,
        rag_evaluation_label=rag_label,
    )
    