"""
evaluate_retrieval.py

Runs top-1 retrieval across all 45 questions and measures how often the
top retrieved document matches the fact the question was written about.

This is a RETRIEVAL quality check only -- it says nothing about whether
the LLM's eventual answer is correct or grounded. Those checks come in
later phases (claim analysis, hallucination detection, evaluation).

Run with:
    python -m src.evaluate_retrieval
"""

import pandas as pd

from src import config, data_loader
from src.retriever import Retriever


def extract_expected_fact_id(question_id: str) -> str:
    """
    question_id format is '{FACT_ID}_{lang_code}', e.g. 'GEO01_en'.
    Returns the FACT_ID part.
    """
    return question_id.rsplit("_", 1)[0]


def run_retrieval_quality_check() -> pd.DataFrame:
    questions_df = data_loader.load_questions()

    retriever = Retriever()
    retriever.build_index()

    rows = []
    for _, row in questions_df.iterrows():
        expected_fact_id = extract_expected_fact_id(row["question_id"])
        results = retriever.retrieve(row["question"], row["language"], top_k=1)

        if results:
            top_result = results[0]
            retrieved_fact_id = top_result.linked_fact_id
            similarity_score = top_result.similarity_score
            retrieved_doc_id = top_result.document_id
        else:
            retrieved_fact_id = None
            similarity_score = None
            retrieved_doc_id = None

        is_correct = retrieved_fact_id == expected_fact_id

        rows.append(
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "language": row["language"],
                "question": row["question"],
                "expected_fact_id": expected_fact_id,
                "retrieved_document_id": retrieved_doc_id,
                "retrieved_fact_id": retrieved_fact_id,
                "similarity_score": similarity_score,
                "top1_correct": is_correct,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(details_df: pd.DataFrame) -> pd.DataFrame:
    overall_accuracy = details_df["top1_correct"].mean()
    by_language = details_df.groupby("language")["top1_correct"].mean()
    by_category = details_df.groupby("category")["top1_correct"].mean()

    summary_rows = [
        {"breakdown": "Overall", "group": "All", "top1_accuracy": overall_accuracy}
    ]
    for language, accuracy in by_language.items():
        summary_rows.append(
            {"breakdown": "Language", "group": language, "top1_accuracy": accuracy}
        )
    for category, accuracy in by_category.items():
        summary_rows.append(
            {"breakdown": "Category", "group": category, "top1_accuracy": accuracy}
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    details_df = run_retrieval_quality_check()
    summary_df = summarize_results(details_df)

    details_df.to_csv(
        config.RETRIEVAL_QUALITY_DETAILS_CSV, index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        config.RETRIEVAL_QUALITY_SUMMARY_CSV, index=False, encoding="utf-8-sig"
    )

    print("=" * 70)
    print("RETRIEVAL QUALITY CHECK (Top-1 accuracy)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()
    print(f"Detailed per-question results saved to: {config.RETRIEVAL_QUALITY_DETAILS_CSV}")
    print(f"Summary saved to: {config.RETRIEVAL_QUALITY_SUMMARY_CSV}")

    incorrect = details_df[~details_df["top1_correct"]]
    if not incorrect.empty:
        print()
        print(f"{len(incorrect)} question(s) had incorrect top-1 retrieval:")
        print(
            incorrect[
                ["question_id", "language", "category", "expected_fact_id", "retrieved_fact_id"]
            ].to_string(index=False)
        )
    else:
        print("\nAll 45 questions retrieved the correct fact at top-1.")


if __name__ == "__main__":
    main()