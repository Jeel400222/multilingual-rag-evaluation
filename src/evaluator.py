"""
evaluator.py

Runs the full 45-question experiment (RAG + Non-RAG generation, claim
analysis, hallucination detection, correctness checking) and computes all
evaluation metrics.

METHODOLOGY NOTE: automated correctness checking here is a NORMALIZED TEXT
MATCHING heuristic (exact/near-exact term presence), not human judgment or
semantic fact-checking. It is deliberately kept separate from
hallucination_detector.py's context-groundedness check -- a response can
be "Incorrect" (does not match the reference answer) while still being
well-grounded in the retrieved context (e.g. it echoes a real passage but
gets a specific detail wrong), and it can be "Unsupported-Hallucinated"
(wrong AND not semantically traceable to any retrieved passage). These are
never conflated: see assign_evaluation_label() below.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd

from src import config, data_loader, llm_client
from src.hallucination_detector import assess_response
from src.retriever import Retriever

# ---------------------------------------------------------------------------
# Correctness checking (normalized text matching heuristic)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercases, strips parenthetical content and punctuation, collapses whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)  # drop parenthetical alt-text, e.g. "(स्वर्ण)"
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_correctness(response_text: str, reference_answer: str) -> Tuple[bool, bool]:
    """
    Returns (strict_correct, relaxed_correct).

    strict_correct: normalized reference answer appears verbatim in the
        normalized response.
    relaxed_correct: strict_correct, OR at least
        config.RELAXED_TOKEN_OVERLAP_THRESHOLD of the reference answer's
        significant words appear in the response (handles rephrasing).
    """
    normalized_response = _normalize_text(response_text)
    normalized_reference = _normalize_text(reference_answer)

    if not normalized_reference or not normalized_response:
        return False, False

    strict_correct = normalized_reference in normalized_response

    reference_tokens = set(normalized_reference.split())
    response_tokens = set(normalized_response.split())
    overlap_ratio = (
        len(reference_tokens & response_tokens) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    relaxed_correct = strict_correct or overlap_ratio >= config.RELAXED_TOKEN_OVERLAP_THRESHOLD

    return strict_correct, relaxed_correct


def assign_evaluation_label(
    generation_success: bool,
    strict_correct: bool,
    relaxed_correct: bool,
    hallucination_risk: str,
) -> str:
    """
    Assigns one of config.EVALUATION_LABELS.

    "Incorrect" vs "Unsupported-Hallucinated" is the key distinction this
    project must not collapse: both mean the answer didn't match the
    reference, but "Incorrect" still had reasonable grounding in the
    retrieved context (low/medium hallucination risk), while
    "Unsupported-Hallucinated" means the wrong answer also wasn't
    traceable to the retrieved context at all (high risk).
    """
    if not generation_success:
        return "Unable to Evaluate"
    if strict_correct:
        return "Correct"
    if relaxed_correct:
        return "Partially Correct"
    if hallucination_risk == "High":
        return "Unsupported-Hallucinated"
    return "Incorrect"


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------


def _extract_expected_fact_id(question_id: str) -> str:
    return question_id.rsplit("_", 1)[0]


def run_full_experiment() -> pd.DataFrame:
    """
    Runs RAG + Non-RAG generation, claim analysis, and correctness checking
    for all 45 questions. Returns one row per question with rag_/non_rag_
    prefixed columns for anything that differs between the two response
    types, so the comparison is preserved rather than collapsed.
    """
    questions_df = data_loader.load_questions()
    retriever = Retriever()
    retriever.build_index()

    rows = []
    for i, q in questions_df.iterrows():
        print(f"[{i + 1}/{len(questions_df)}] {q['question_id']} ({q['language']})")

        expected_fact_id = _extract_expected_fact_id(q["question_id"])
        context_docs = retriever.retrieve(q["question"], q["language"], top_k=config.TOP_K_RETRIEVAL)
        top1_retrieval_correct = (
            bool(context_docs) and context_docs[0].linked_fact_id == expected_fact_id
        )

        retrieved_context_json = json.dumps(
            [
                {
                    "document_id": d.document_id,
                    "content": d.content,
                    "category": d.category,
                    "similarity_score": d.similarity_score,
                }
                for d in context_docs
            ],
            ensure_ascii=False,
        )
        retrieval_scores_json = json.dumps([d.similarity_score for d in context_docs])
        context_texts = [d.content for d in context_docs]

        # --- Non-RAG ---
        non_rag_llm = llm_client.generate_non_rag_response(q["question"], q["language"])
        if non_rag_llm.success:
            non_rag_assessment = assess_response(non_rag_llm.response_text, context_docs)
            non_rag_strict, non_rag_relaxed = evaluate_correctness(
                non_rag_llm.response_text, q["reference_answer"]
            )
        else:
            non_rag_assessment = None
            non_rag_strict, non_rag_relaxed = False, False
        non_rag_risk = non_rag_assessment.hallucination_risk if non_rag_assessment else "Unable to Determine"
        non_rag_label = assign_evaluation_label(
            non_rag_llm.success, non_rag_strict, non_rag_relaxed, non_rag_risk
        )

        # --- RAG ---
        rag_llm = llm_client.generate_rag_response(q["question"], q["language"], context_texts)
        if rag_llm.success:
            rag_assessment = assess_response(rag_llm.response_text, context_docs)
            rag_strict, rag_relaxed = evaluate_correctness(rag_llm.response_text, q["reference_answer"])
        else:
            rag_assessment = None
            rag_strict, rag_relaxed = False, False
        rag_risk = rag_assessment.hallucination_risk if rag_assessment else "Unable to Determine"
        rag_label = assign_evaluation_label(rag_llm.success, rag_strict, rag_relaxed, rag_risk)

        rows.append(
            {
                "experiment_id": f"EXP_{q['question_id']}",
                "question_id": q["question_id"],
                "category": q["category"],
                "language": q["language"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "retrieved_context": retrieved_context_json,
                "retrieval_scores": retrieval_scores_json,
                "top1_retrieval_correct": top1_retrieval_correct,
                # Non-RAG
                "non_rag_response": non_rag_llm.response_text,
                "non_rag_generation_success": non_rag_llm.success,
                "non_rag_error_message": non_rag_llm.error_message,
                "non_rag_claim_analysis": json.dumps(
                    non_rag_assessment.to_dict()["claims"] if non_rag_assessment else [],
                    ensure_ascii=False,
                ),
                "non_rag_groundedness_score": non_rag_assessment.groundedness_score
                if non_rag_assessment
                else None,
                "non_rag_hallucination_risk": non_rag_risk,
                "non_rag_strict_correct": non_rag_strict,
                "non_rag_relaxed_correct": non_rag_relaxed,
                "non_rag_evaluation_label": non_rag_label,
                # RAG
                "rag_response": rag_llm.response_text,
                "rag_generation_success": rag_llm.success,
                "rag_error_message": rag_llm.error_message,
                "rag_claim_analysis": json.dumps(
                    rag_assessment.to_dict()["claims"] if rag_assessment else [],
                    ensure_ascii=False,
                ),
                "rag_groundedness_score": rag_assessment.groundedness_score if rag_assessment else None,
                "rag_hallucination_risk": rag_risk,
                "rag_strict_correct": rag_strict,
                "rag_relaxed_correct": rag_relaxed,
                "rag_evaluation_label": rag_label,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Long-format helper (for metric computation across both response types)
# ---------------------------------------------------------------------------


def _to_long_format(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes the wide (one row per question) experiment results into long
    format (one row per question PER response type), so metrics functions
    can filter by response_type without duplicating logic.
    """
    shared_cols = ["question_id", "category", "language", "top1_retrieval_correct"]

    non_rag_df = results_df[shared_cols].copy()
    non_rag_df["response_type"] = "non_rag"
    non_rag_df["evaluation_label"] = results_df["non_rag_evaluation_label"]
    non_rag_df["groundedness_score"] = results_df["non_rag_groundedness_score"]
    non_rag_df["hallucination_risk"] = results_df["non_rag_hallucination_risk"]
    non_rag_df["claim_analysis"] = results_df["non_rag_claim_analysis"]

    rag_df = results_df[shared_cols].copy()
    rag_df["response_type"] = "rag"
    rag_df["evaluation_label"] = results_df["rag_evaluation_label"]
    rag_df["groundedness_score"] = results_df["rag_groundedness_score"]
    rag_df["hallucination_risk"] = results_df["rag_hallucination_risk"]
    rag_df["claim_analysis"] = results_df["rag_claim_analysis"]

    return pd.concat([non_rag_df, rag_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def compute_response_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response_type, group in long_df.groupby("response_type"):
        total = len(group)
        evaluated = group[group["evaluation_label"] != "Unable to Evaluate"]
        label_counts = group["evaluation_label"].value_counts().to_dict()

        strict_accuracy = (group["evaluation_label"] == "Correct").sum() / total if total else 0.0
        relaxed_accuracy = (
            group["evaluation_label"].isin(["Correct", "Partially Correct"]).sum() / total
            if total
            else 0.0
        )

        rows.append(
            {
                "response_type": response_type,
                "total_responses": total,
                "total_evaluated": len(evaluated),
                "correct": label_counts.get("Correct", 0),
                "partially_correct": label_counts.get("Partially Correct", 0),
                "incorrect": label_counts.get("Incorrect", 0),
                "unsupported_hallucinated": label_counts.get("Unsupported-Hallucinated", 0),
                "unable_to_evaluate": label_counts.get("Unable to Evaluate", 0),
                "strict_accuracy": strict_accuracy,
                "relaxed_accuracy": relaxed_accuracy,
            }
        )
    return pd.DataFrame(rows)


def compute_accuracy_by_language(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (response_type, language), group in long_df.groupby(["response_type", "language"]):
        total = len(group)
        strict_accuracy = (group["evaluation_label"] == "Correct").sum() / total if total else 0.0
        relaxed_accuracy = (
            group["evaluation_label"].isin(["Correct", "Partially Correct"]).sum() / total
            if total
            else 0.0
        )
        hallucination_rate = (
            (group["evaluation_label"] == "Unsupported-Hallucinated").sum() / total if total else 0.0
        )
        rows.append(
            {
                "response_type": response_type,
                "language": language,
                "strict_accuracy": strict_accuracy,
                "relaxed_accuracy": relaxed_accuracy,
                "hallucination_rate": hallucination_rate,
            }
        )
    return pd.DataFrame(rows)


def compute_accuracy_by_category(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (response_type, category), group in long_df.groupby(["response_type", "category"]):
        total = len(group)
        strict_accuracy = (group["evaluation_label"] == "Correct").sum() / total if total else 0.0
        relaxed_accuracy = (
            group["evaluation_label"].isin(["Correct", "Partially Correct"]).sum() / total
            if total
            else 0.0
        )
        hallucination_rate = (
            (group["evaluation_label"] == "Unsupported-Hallucinated").sum() / total if total else 0.0
        )
        rows.append(
            {
                "response_type": response_type,
                "category": category,
                "strict_accuracy": strict_accuracy,
                "relaxed_accuracy": relaxed_accuracy,
                "hallucination_rate": hallucination_rate,
            }
        )
    return pd.DataFrame(rows)


def compute_rag_metrics_summary(long_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Claim-level supported/partial/unsupported rates (computed across every
    individual claim, not per-response), average groundedness, and
    retrieval success rate (top-1 fact match, shared across response types
    since retrieval happens once per question).
    """
    retrieval_success_rate = results_df["top1_retrieval_correct"].mean()

    rows = []
    for response_type, group in long_df.groupby("response_type"):
        avg_groundedness = group["groundedness_score"].dropna().mean()

        all_claims = []
        for claim_json in group["claim_analysis"]:
            all_claims.extend(json.loads(claim_json))

        total_claims = len(all_claims)
        if total_claims > 0:
            supported = sum(1 for c in all_claims if c["status"] == "Supported")
            partial = sum(1 for c in all_claims if c["status"] == "Partially Supported")
            unsupported = sum(1 for c in all_claims if c["status"] == "Unsupported")
            supported_rate = supported / total_claims
            unsupported_rate = unsupported / total_claims
            partial_rate = partial / total_claims
        else:
            supported_rate = partial_rate = unsupported_rate = None

        rows.append(
            {
                "response_type": response_type,
                "avg_groundedness_score": avg_groundedness,
                "supported_claim_rate": supported_rate,
                "partially_supported_claim_rate": partial_rate,
                "unsupported_claim_rate": unsupported_rate,
                "retrieval_success_rate": retrieval_success_rate,
                "total_claims_analyzed": total_claims,
            }
        )
    return pd.DataFrame(rows)


def compute_hallucination_risk_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response_type, group in long_df.groupby("response_type"):
        total = len(group)
        risk_counts = group["hallucination_risk"].value_counts().to_dict()
        for risk_level in ["Low", "Medium", "High", "Unable to Determine"]:
            count = risk_counts.get(risk_level, 0)
            rows.append(
                {
                    "response_type": response_type,
                    "risk_level": risk_level,
                    "count": count,
                    "percentage": (count / total * 100) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Running full experiment across 45 questions (RAG + Non-RAG)...")
    print("This makes 90 Groq API calls and may take a few minutes.\n")

    results_df = run_full_experiment()
    results_df.to_csv(config.EXPERIMENT_RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved full experiment results ({len(results_df)} rows) to: {config.EXPERIMENT_RESULTS_CSV}")

    long_df = _to_long_format(results_df)

    response_metrics = compute_response_metrics(long_df)
    accuracy_by_language = compute_accuracy_by_language(long_df)
    accuracy_by_category = compute_accuracy_by_category(long_df)
    rag_metrics_summary = compute_rag_metrics_summary(long_df, results_df)
    hallucination_risk_distribution = compute_hallucination_risk_distribution(long_df)

    response_metrics.to_csv(config.RESPONSE_METRICS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    accuracy_by_language.to_csv(config.ACCURACY_BY_LANGUAGE_CSV, index=False, encoding="utf-8-sig")
    accuracy_by_category.to_csv(config.ACCURACY_BY_CATEGORY_CSV, index=False, encoding="utf-8-sig")
    rag_metrics_summary.to_csv(config.RAG_METRICS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    hallucination_risk_distribution.to_csv(
        config.HALLUCINATION_RISK_DISTRIBUTION_CSV, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("RESPONSE METRICS (Non-RAG vs RAG)")
    print("=" * 70)
    print(response_metrics.to_string(index=False))

    print("\n" + "=" * 70)
    print("ACCURACY BY LANGUAGE")
    print("=" * 70)
    print(accuracy_by_language.to_string(index=False))

    print("\n" + "=" * 70)
    print("ACCURACY BY CATEGORY")
    print("=" * 70)
    print(accuracy_by_category.to_string(index=False))

    print("\n" + "=" * 70)
    print("RAG METRICS SUMMARY")
    print("=" * 70)
    print(rag_metrics_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("HALLUCINATION RISK DISTRIBUTION")
    print("=" * 70)
    print(hallucination_risk_distribution.to_string(index=False))

    print(f"\nAll metric tables saved to: {config.TABLES_DIR}")


if __name__ == "__main__":
    main()