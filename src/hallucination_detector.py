"""
hallucination_detector.py

Classifies each claim in an LLM response as Supported / Partially
Supported / Unsupported / Unable to Determine against a set of retrieved
context documents, using multilingual sentence-embedding cosine
similarity as the matching signal, then aggregates into a groundedness
score and a hallucination risk category.

METHODOLOGY NOTE (read before interpreting any results this module produces):
This is a SEMANTIC SIMILARITY HEURISTIC, not logical entailment or
automated fact-checking. A claim scoring above CLAIM_SUPPORTED_THRESHOLD
is semantically close in wording/meaning to some retrieved passage -- it
is NOT independently verified as logically implied by that passage. A low
score means no close semantic match was found in the retrieved context --
it does NOT mean the claim is factually wrong. True NLI (entailment)
models exist for English but do not reliably support Hindi and Gujarati,
which is why this project uses one similarity-based method consistently
across all three languages rather than silently downgrading accuracy for
two of them. "Unsupported by retrieved context" and "factually incorrect"
are evaluated as separate, independent dimensions in this project (see
evaluator.py in Phase 6) and must never be conflated when displaying
results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from src import config, embeddings
from src.retriever import RetrievedDocument

METHODOLOGY_NOTE = (
    "Claim support is determined by semantic similarity between "
    "multilingual sentence embeddings (cosine similarity), NOT logical "
    "entailment or automated fact-checking. A high similarity score means "
    "a claim's wording is semantically close to some retrieved passage -- "
    "it does not guarantee the claim is logically implied by that "
    "passage. A low score means no close semantic match was found in the "
    "retrieved context -- it does NOT mean the claim is factually "
    "incorrect. 'Unsupported by retrieved context' and 'factually "
    "incorrect' are evaluated separately in this project and must not be "
    "conflated."
)


@dataclass
class ClaimAssessment:
    claim_text: str
    status: str
    similarity_score: Optional[float]
    best_evidence_document_id: Optional[str]
    best_evidence_text: Optional[str]
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResponseGroundednessResult:
    claims: List[ClaimAssessment]
    total_claims: int
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    unable_to_determine_count: int
    groundedness_score: Optional[float]
    hallucination_risk: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d


def assess_claim(
    claim_text: str, context_documents: List[RetrievedDocument]
) -> ClaimAssessment:
    """
    Compares a single claim against retrieved context documents and
    classifies it based on maximum cosine similarity.

    The claim is embedded with the query prefix and context documents
    with the passage prefix -- mirroring the retriever's own convention
    of treating a claim as something seeking supporting evidence among
    passages, consistent with how the embedding model was trained.
    """
    if not claim_text or not claim_text.strip():
        return ClaimAssessment(
            claim_text=claim_text,
            status="Unable to Determine",
            similarity_score=None,
            best_evidence_document_id=None,
            best_evidence_text=None,
            reasoning="The claim text was empty and could not be assessed.",
        )

    if not context_documents:
        return ClaimAssessment(
            claim_text=claim_text,
            status="Unable to Determine",
            similarity_score=None,
            best_evidence_document_id=None,
            best_evidence_text=None,
            reasoning=(
                "No context documents were retrieved, so this claim could "
                "not be checked against any evidence."
            ),
        )

    claim_embedding = embeddings.encode_queries([claim_text])[0]
    context_texts = [doc.content for doc in context_documents]
    context_embeddings = embeddings.encode_passages(context_texts)

    similarities = context_embeddings @ claim_embedding
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    best_doc = context_documents[best_idx]

    if best_score >= config.CLAIM_SUPPORTED_THRESHOLD:
        status = "Supported"
        reasoning = (
            f"This claim is semantically very close (similarity "
            f"{best_score:.2f}) to retrieved document "
            f"{best_doc.document_id}."
        )
    elif best_score >= config.CLAIM_PARTIAL_THRESHOLD:
        status = "Partially Supported"
        reasoning = (
            f"This claim has some semantic overlap (similarity "
            f"{best_score:.2f}) with retrieved document "
            f"{best_doc.document_id}, but the match is not strong enough "
            f"to consider it fully supported."
        )
    else:
        status = "Unsupported"
        reasoning = (
            f"No retrieved document was semantically close enough to "
            f"this claim (best similarity {best_score:.2f}, with document "
            f"{best_doc.document_id}). This means the claim was not found "
            f"in the retrieved context -- it does not necessarily mean "
            f"the claim is factually wrong."
        )

    return ClaimAssessment(
        claim_text=claim_text,
        status=status,
        similarity_score=best_score,
        best_evidence_document_id=best_doc.document_id,
        best_evidence_text=best_doc.content,
        reasoning=reasoning,
    )


def _compute_hallucination_risk(unsupported_rate: float) -> str:
    if unsupported_rate <= config.HALLUCINATION_RISK_LOW_MAX_UNSUPPORTED_RATE:
        return "Low"
    elif unsupported_rate <= config.HALLUCINATION_RISK_MEDIUM_MAX_UNSUPPORTED_RATE:
        return "Medium"
    else:
        return "High"


def assess_response(
    response_text: str, context_documents: List[RetrievedDocument]
) -> ResponseGroundednessResult:
    """
    Splits a response into claims, assesses each against the provided
    context documents, and aggregates into a groundedness score and a
    hallucination risk category.

    Groundedness score = (supported_count + 0.5 * partially_supported_count)
    / determinate_claim_count (excludes "Unable to Determine" claims from
    the denominator). The 0.5 weight for partial support is a deliberate,
    documented heuristic choice -- not a statistically fitted parameter --
    so partial matches count for something without being treated as
    equivalent to full support.
    """
    from src.claim_analyzer import split_into_claims

    claim_texts = split_into_claims(response_text)

    if not claim_texts:
        return ResponseGroundednessResult(
            claims=[],
            total_claims=0,
            supported_count=0,
            partially_supported_count=0,
            unsupported_count=0,
            unable_to_determine_count=0,
            groundedness_score=None,
            hallucination_risk="Unable to Determine",
        )

    assessments = [assess_claim(claim, context_documents) for claim in claim_texts]

    supported_count = sum(1 for a in assessments if a.status == "Supported")
    partial_count = sum(1 for a in assessments if a.status == "Partially Supported")
    unsupported_count = sum(1 for a in assessments if a.status == "Unsupported")
    undetermined_count = sum(1 for a in assessments if a.status == "Unable to Determine")

    determinate_total = supported_count + partial_count + unsupported_count

    if determinate_total == 0:
        groundedness_score = None
        hallucination_risk = "Unable to Determine"
    else:
        groundedness_score = (supported_count + 0.5 * partial_count) / determinate_total
        unsupported_rate = unsupported_count / determinate_total
        hallucination_risk = _compute_hallucination_risk(unsupported_rate)

    return ResponseGroundednessResult(
        claims=assessments,
        total_claims=len(assessments),
        supported_count=supported_count,
        partially_supported_count=partial_count,
        unsupported_count=unsupported_count,
        unable_to_determine_count=undetermined_count,
        groundedness_score=groundedness_score,
        hallucination_risk=hallucination_risk,
    )


if __name__ == "__main__":
    from src.retriever import Retriever
    from src import llm_client

    print("Running end-to-end claim analysis demo...\n")
    retriever = Retriever()
    retriever.build_index()

    demo_question = "What is the capital of India?"
    demo_language = "English"

    context_docs = retriever.retrieve(demo_question, demo_language, top_k=3)
    print(f"Retrieved {len(context_docs)} context documents.\n")

    print("=" * 70)
    print("RAG RESPONSE CLAIM ANALYSIS")
    print("=" * 70)
    rag_response = llm_client.generate_rag_response(
        demo_question, demo_language, [d.content for d in context_docs]
    )
    if rag_response.success:
        print(f"Response: {rag_response.response_text}\n")
        result = assess_response(rag_response.response_text, context_docs)
        for i, claim in enumerate(result.claims, start=1):
            print(f"Claim {i}: {claim.claim_text}")
            print(f"  Status: {claim.status} (score={claim.similarity_score})")
            print(f"  Reasoning: {claim.reasoning}\n")
        print(f"Groundedness score: {result.groundedness_score}")
        print(f"Hallucination risk: {result.hallucination_risk}")
    else:
        print(f"RAG generation failed: {rag_response.error_message}")

    print()
    print("=" * 70)
    print("NON-RAG RESPONSE CLAIM ANALYSIS (checked against same context)")
    print("=" * 70)
    non_rag_response = llm_client.generate_non_rag_response(demo_question, demo_language)
    if non_rag_response.success:
        print(f"Response: {non_rag_response.response_text}\n")
        result = assess_response(non_rag_response.response_text, context_docs)
        for i, claim in enumerate(result.claims, start=1):
            print(f"Claim {i}: {claim.claim_text}")
            print(f"  Status: {claim.status} (score={claim.similarity_score})")
            print(f"  Reasoning: {claim.reasoning}\n")
        print(f"Groundedness score: {result.groundedness_score}")
        print(f"Hallucination risk: {result.hallucination_risk}")
    else:
        print(f"Non-RAG generation failed: {non_rag_response.error_message}")

    print()
    print("METHODOLOGY NOTE:")
    print(METHODOLOGY_NOTE)