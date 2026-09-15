"""
calibrate_thresholds.py

Generates RAG responses for all 45 questions, splits each into claims,
computes each claim's best similarity score against its own retrieved
context, and reports the resulting score distribution.

Purpose: CLAIM_SUPPORTED_THRESHOLD / CLAIM_PARTIAL_THRESHOLD in config.py
were set as reasonable starting points, not derived from data. Since these
are RAG responses generated FROM their own retrieved context, their claims
SHOULD mostly be genuinely well-supported -- so this distribution shows
where "supported" scores actually land for this embedding model on this
data, which is the honest basis for keeping or adjusting the thresholds.

This makes 45 real Groq API calls. Run with:
    python -m src.calibrate_thresholds
"""

import numpy as np
import pandas as pd

from src import config, data_loader, llm_client
from src.claim_analyzer import split_into_claims
from src.hallucination_detector import assess_claim
from src.retriever import Retriever


def main() -> None:
    questions_df = data_loader.load_questions()
    retriever = Retriever()
    retriever.build_index()

    rows = []
    for i, row in questions_df.iterrows():
        print(f"[{i + 1}/{len(questions_df)}] {row['question_id']} ({row['language']})")

        context_docs = retriever.retrieve(row["question"], row["language"], top_k=3)
        context_texts = [d.content for d in context_docs]

        rag_response = llm_client.generate_rag_response(
            row["question"], row["language"], context_texts
        )
        if not rag_response.success:
            print(f"  Skipped (generation failed): {rag_response.error_message}")
            continue

        claims = split_into_claims(rag_response.response_text)
        for claim_text in claims:
            assessment = assess_claim(claim_text, context_docs)
            rows.append(
                {
                    "question_id": row["question_id"],
                    "language": row["language"],
                    "category": row["category"],
                    "claim_text": claim_text,
                    "similarity_score": assessment.similarity_score,
                    "status_at_current_thresholds": assessment.status,
                }
            )

    scores_df = pd.DataFrame(rows)
    output_path = config.TABLES_DIR / "claim_similarity_scores.csv"
    scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    scores = scores_df["similarity_score"].dropna().to_numpy()
    print()
    print("=" * 70)
    print("CLAIM SIMILARITY SCORE DISTRIBUTION (RAG responses vs own retrieved context)")
    print("=" * 70)
    print(f"Total claims analyzed: {len(scores)}")
    print(f"Min:    {scores.min():.4f}")
    print(f"P25:    {np.percentile(scores, 25):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"P75:    {np.percentile(scores, 75):.4f}")
    print(f"Max:    {scores.max():.4f}")
    print()
    print(f"Current CLAIM_SUPPORTED_THRESHOLD = {config.CLAIM_SUPPORTED_THRESHOLD}")
    print(f"Current CLAIM_PARTIAL_THRESHOLD   = {config.CLAIM_PARTIAL_THRESHOLD}")
    print()
    print("Status breakdown at current thresholds:")
    print(scores_df["status_at_current_thresholds"].value_counts().to_string())
    print(f"\nFull results saved to: {output_path}")
    print(
        "\nReview the distribution above. If most RAG claims (which SHOULD "
        "be well-supported, since they were generated from this exact "
        "context) score well above CLAIM_SUPPORTED_THRESHOLD, the current "
        "threshold is reasonable. If many genuinely-grounded claims fall "
        "below it, consider lowering CLAIM_SUPPORTED_THRESHOLD in "
        "config.py to better match this embedding model's actual "
        "similarity scale."
    )


if __name__ == "__main__":
    main()