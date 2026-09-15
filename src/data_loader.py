"""
data_loader.py

Loads and validates the questions dataset and knowledge base CSVs.
All later modules (retriever, rag_pipeline, evaluator) should load data
through this module rather than reading CSVs directly, so validation
logic lives in exactly one place.
"""

from pathlib import Path

import pandas as pd

from src import config

REQUIRED_QUESTION_COLUMNS = [
    "question_id",
    "category",
    "language",
    "question",
    "reference_answer",
]

REQUIRED_KB_COLUMNS = [
    "document_id",
    "category",
    "language",
    "content",
    "source",
]


def _require_file(path: Path, build_command: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            f"Generate it by running: {build_command}"
        )


def _validate_columns(df: pd.DataFrame, required_columns: list, file_label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _validate_no_nulls(df: pd.DataFrame, columns: list, file_label: str) -> None:
    for col in columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"{file_label} has {null_count} missing value(s) in column '{col}'. "
                "All rows must be fully populated."
            )


def _validate_allowed_values(
    df: pd.DataFrame, column: str, allowed_values: list, file_label: str
) -> None:
    invalid = set(df[column].unique()) - set(allowed_values)
    if invalid:
        raise ValueError(
            f"{file_label} column '{column}' contains unsupported values: {invalid}. "
            f"Allowed values are: {allowed_values}"
        )


def load_questions() -> pd.DataFrame:
    """
    Loads and validates data/raw/questions.csv.

    Returns:
        DataFrame with columns: question_id, category, language, question,
        reference_answer.

    Raises:
        FileNotFoundError: if the CSV does not exist yet.
        ValueError: if required columns are missing, values are null, or
            category/language values fall outside the supported sets.
    """
    _require_file(config.QUESTIONS_CSV, "python -m src.build_seed_data")

    df = pd.read_csv(config.QUESTIONS_CSV, encoding="utf-8-sig")

    _validate_columns(df, REQUIRED_QUESTION_COLUMNS, "questions.csv")
    _validate_no_nulls(df, REQUIRED_QUESTION_COLUMNS, "questions.csv")
    _validate_allowed_values(
        df, "category", config.SUPPORTED_CATEGORIES, "questions.csv"
    )
    _validate_allowed_values(
        df, "language", config.SUPPORTED_LANGUAGES, "questions.csv"
    )

    duplicate_ids = df["question_id"][df["question_id"].duplicated()].tolist()
    if duplicate_ids:
        raise ValueError(f"questions.csv contains duplicate question_id values: {duplicate_ids}")

    return df


def load_knowledge_base() -> pd.DataFrame:
    """
    Loads and validates data/knowledge_base/knowledge_base.csv.

    Returns:
        DataFrame with columns: document_id, category, language, content,
        source (plus optional linked_fact_id, is_distractor if present).

    Raises:
        FileNotFoundError: if the CSV does not exist yet.
        ValueError: if required columns are missing, values are null, or
            category/language values fall outside the supported sets.
    """
    _require_file(config.KNOWLEDGE_BASE_CSV, "python -m src.build_seed_data")

    df = pd.read_csv(config.KNOWLEDGE_BASE_CSV, encoding="utf-8-sig")

    _validate_columns(df, REQUIRED_KB_COLUMNS, "knowledge_base.csv")
    _validate_no_nulls(df, REQUIRED_KB_COLUMNS, "knowledge_base.csv")
    _validate_allowed_values(
        df, "category", config.SUPPORTED_CATEGORIES, "knowledge_base.csv"
    )
    _validate_allowed_values(
        df, "language", config.SUPPORTED_LANGUAGES, "knowledge_base.csv"
    )

    duplicate_ids = df["document_id"][df["document_id"].duplicated()].tolist()
    if duplicate_ids:
        raise ValueError(
            f"knowledge_base.csv contains duplicate document_id values: {duplicate_ids}"
        )

    return df


def get_questions_by_language(questions_df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Filters questions to a single language. Validates the language first."""
    if language not in config.SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Must be one of {config.SUPPORTED_LANGUAGES}."
        )
    return questions_df[questions_df["language"] == language].reset_index(drop=True)


def get_kb_by_language(kb_df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Filters knowledge base documents to a single language."""
    if language not in config.SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Must be one of {config.SUPPORTED_LANGUAGES}."
        )
    return kb_df[kb_df["language"] == language].reset_index(drop=True)


def print_dataset_summary() -> None:
    """Prints a quick sanity-check summary of both datasets."""
    questions_df = load_questions()
    kb_df = load_knowledge_base()

    print("=" * 60)
    print("QUESTIONS DATASET SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(questions_df)}")
    print("\nBy language:")
    print(questions_df["language"].value_counts().to_string())
    print("\nBy category:")
    print(questions_df["category"].value_counts().to_string())

    print()
    print("=" * 60)
    print("KNOWLEDGE BASE SUMMARY")
    print("=" * 60)
    print(f"Total documents: {len(kb_df)}")
    print("\nBy language:")
    print(kb_df["language"].value_counts().to_string())
    print("\nBy category:")
    print(kb_df["category"].value_counts().to_string())

    if "is_distractor" in kb_df.columns:
        print(f"\nDistractor documents: {int(kb_df['is_distractor'].sum())}")


if __name__ == "__main__":
    print_dataset_summary()