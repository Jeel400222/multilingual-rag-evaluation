"""
visualization.py

Generates the project's charts from data/results/experiment_results.csv
and the metric tables saved by evaluator.py / evaluate_retrieval.py.
All charts are saved as PNGs to outputs/charts/.

Note on interpreting these charts: this project's dataset is small (45
questions) and deliberately clean and well-separated (see build_seed_data.py).
Charts showing 100% accuracy or 100% "Low" hallucination risk reflect that
dataset design honestly -- they are not evidence that the underlying
methods (retrieval, claim-similarity scoring) are error-free in general,
only that they perform well on this particular, limited evaluation set.
This caveat is intentionally repeated in the README.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

RAG_COLOR = "#0891b2"
NON_RAG_COLOR = "#f59e0b"
RISK_COLORS = {
    "Low": "#16a34a",
    "Medium": "#f59e0b",
    "High": "#dc2626",
    "Unable to Determine": "#9ca3af",
}
CLAIM_STATUS_COLORS = {
    "Supported": "#16a34a",
    "Partially Supported": "#f59e0b",
    "Unsupported": "#dc2626",
    "Unable to Determine": "#9ca3af",
}
DPI = 150


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _save(fig, filename: str) -> str:
    path = config.CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _load_experiment_results() -> pd.DataFrame:
    if not config.EXPERIMENT_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{config.EXPERIMENT_RESULTS_CSV} not found. Run "
            "`python -m src.evaluator` first."
        )
    return pd.read_csv(config.EXPERIMENT_RESULTS_CSV, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Chart 1: Accuracy by Language (Strict, RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_accuracy_by_language() -> str:
    df = pd.read_csv(config.ACCURACY_BY_LANGUAGE_CSV, encoding="utf-8-sig")
    languages = config.SUPPORTED_LANGUAGES
    x = np.arange(len(languages))
    width = 0.35

    non_rag_vals = [
        df[(df.response_type == "non_rag") & (df.language == lang)]["strict_accuracy"].iloc[0]
        for lang in languages
    ]
    rag_vals = [
        df[(df.response_type == "rag") & (df.language == lang)]["strict_accuracy"].iloc[0]
        for lang in languages
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, non_rag_vals, width, label="Non-RAG", color=NON_RAG_COLOR)
    ax.bar(x + width / 2, rag_vals, width, label="RAG", color=RAG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylim(0, 1.1)
    _style_axes(ax, "Strict Accuracy by Language", "Language", "Strict Accuracy")
    ax.legend()
    return _save(fig, "01_accuracy_by_language.png")


# ---------------------------------------------------------------------------
# Chart 2: RAG vs Non-RAG Overall Accuracy (Strict + Relaxed)
# ---------------------------------------------------------------------------


def chart_rag_vs_non_rag_accuracy() -> str:
    df = pd.read_csv(config.RESPONSE_METRICS_SUMMARY_CSV, encoding="utf-8-sig")
    metrics = ["strict_accuracy", "relaxed_accuracy"]
    labels = ["Strict Accuracy", "Relaxed Accuracy"]
    x = np.arange(len(metrics))
    width = 0.35

    non_rag_vals = df[df.response_type == "non_rag"][metrics].iloc[0].tolist()
    rag_vals = df[df.response_type == "rag"][metrics].iloc[0].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, non_rag_vals, width, label="Non-RAG", color=NON_RAG_COLOR)
    ax.bar(x + width / 2, rag_vals, width, label="RAG", color=RAG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    _style_axes(ax, "RAG vs Non-RAG: Overall Accuracy", "Metric", "Accuracy")
    ax.legend()
    return _save(fig, "02_rag_vs_non_rag_accuracy.png")


# ---------------------------------------------------------------------------
# Chart 3: Hallucination Risk Distribution (RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_hallucination_risk_distribution() -> str:
    df = pd.read_csv(config.HALLUCINATION_RISK_DISTRIBUTION_CSV, encoding="utf-8-sig")
    response_types = ["non_rag", "rag"]
    risk_levels = ["Low", "Medium", "High", "Unable to Determine"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(response_types))
    for risk in risk_levels:
        vals = [
            df[(df.response_type == rt) & (df.risk_level == risk)]["percentage"].iloc[0]
            for rt in response_types
        ]
        ax.bar(
            response_types, vals, bottom=bottoms, label=risk, color=RISK_COLORS[risk]
        )
        bottoms += np.array(vals)

    ax.set_xticks(range(len(response_types)))
    ax.set_xticklabels(["Non-RAG", "RAG"])
    ax.set_ylim(0, 110)
    _style_axes(ax, "Hallucination Risk Distribution", "Response Type", "Percentage of Responses")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "03_hallucination_risk_distribution.png")


# ---------------------------------------------------------------------------
# Chart 4: Groundedness Score by Language (RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_groundedness_by_language() -> str:
    results_df = _load_experiment_results()
    languages = config.SUPPORTED_LANGUAGES
    x = np.arange(len(languages))
    width = 0.35

    non_rag_vals = [
        results_df[results_df.language == lang]["non_rag_groundedness_score"].mean()
        for lang in languages
    ]
    rag_vals = [
        results_df[results_df.language == lang]["rag_groundedness_score"].mean()
        for lang in languages
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, non_rag_vals, width, label="Non-RAG", color=NON_RAG_COLOR)
    ax.bar(x + width / 2, rag_vals, width, label="RAG", color=RAG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylim(0, 1.1)
    _style_axes(
        ax,
        "Average Groundedness Score by Language",
        "Language",
        "Groundedness Score (claim-similarity heuristic)",
    )
    ax.legend()
    return _save(fig, "04_groundedness_by_language.png")


# ---------------------------------------------------------------------------
# Chart 5: Accuracy by Category (Strict, RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_accuracy_by_category() -> str:
    df = pd.read_csv(config.ACCURACY_BY_CATEGORY_CSV, encoding="utf-8-sig")
    categories = config.SUPPORTED_CATEGORIES
    x = np.arange(len(categories))
    width = 0.35

    non_rag_vals = [
        df[(df.response_type == "non_rag") & (df.category == cat)]["strict_accuracy"].iloc[0]
        for cat in categories
    ]
    rag_vals = [
        df[(df.response_type == "rag") & (df.category == cat)]["strict_accuracy"].iloc[0]
        for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, non_rag_vals, width, label="Non-RAG", color=NON_RAG_COLOR)
    ax.bar(x + width / 2, rag_vals, width, label="RAG", color=RAG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    _style_axes(ax, "Strict Accuracy by Category", "Category", "Strict Accuracy")
    ax.legend()
    return _save(fig, "05_accuracy_by_category.png")


# ---------------------------------------------------------------------------
# Chart 6: Retrieval Similarity Scores (Top-1, by Language)
# ---------------------------------------------------------------------------


def chart_retrieval_similarity_scores() -> str:
    if not config.RETRIEVAL_QUALITY_DETAILS_CSV.exists():
        raise FileNotFoundError(
            f"{config.RETRIEVAL_QUALITY_DETAILS_CSV} not found. Run "
            "`python -m src.evaluate_retrieval` first."
        )
    df = pd.read_csv(config.RETRIEVAL_QUALITY_DETAILS_CSV, encoding="utf-8-sig")
    languages = config.SUPPORTED_LANGUAGES
    data = [df[df.language == lang]["similarity_score"].dropna().tolist() for lang in languages]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(data, labels=languages, patch_artist=True, widths=0.5)
    for patch in box["boxes"]:
        patch.set_facecolor(RAG_COLOR)
        patch.set_alpha(0.6)
    _style_axes(
        ax,
        "Top-1 Retrieval Similarity Scores by Language",
        "Language",
        "Cosine Similarity",
    )
    return _save(fig, "06_retrieval_similarity_scores.png")


# ---------------------------------------------------------------------------
# Chart 7: Claim Classification Distribution (RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_claim_classification_distribution() -> str:
    results_df = _load_experiment_results()
    statuses = ["Supported", "Partially Supported", "Unsupported", "Unable to Determine"]
    response_types = ["non_rag", "rag"]

    counts = {rt: {s: 0 for s in statuses} for rt in response_types}
    for rt, col in [("non_rag", "non_rag_claim_analysis"), ("rag", "rag_claim_analysis")]:
        for claim_json in results_df[col].dropna():
            for claim in json.loads(claim_json):
                counts[rt][claim["status"]] += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(response_types))
    for status in statuses:
        vals = [counts[rt][status] for rt in response_types]
        ax.bar(
            response_types, vals, bottom=bottoms, label=status, color=CLAIM_STATUS_COLORS[status]
        )
        bottoms += np.array(vals)

    ax.set_xticks(range(len(response_types)))
    ax.set_xticklabels(["Non-RAG", "RAG"])
    _style_axes(
        ax, "Claim Classification Distribution", "Response Type", "Number of Claims"
    )
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "07_claim_classification_distribution.png")


# ---------------------------------------------------------------------------
# Chart 8: Language x Category Accuracy Heatmap (RAG vs Non-RAG)
# ---------------------------------------------------------------------------


def chart_language_category_heatmap() -> str:
    results_df = _load_experiment_results()
    languages = config.SUPPORTED_LANGUAGES
    categories = config.SUPPORTED_CATEGORIES

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (response_type, col) in zip(
        axes, [("Non-RAG", "non_rag_strict_correct"), ("RAG", "rag_strict_correct")]
    ):
        matrix = np.zeros((len(languages), len(categories)))
        for i, lang in enumerate(languages):
            for j, cat in enumerate(categories):
                subset = results_df[(results_df.language == lang) & (results_df.category == cat)]
                matrix[i, j] = subset[col].mean() if len(subset) else np.nan

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(languages)
        ax.set_title(f"{response_type} Strict Accuracy", fontsize=12, fontweight="bold")

        for i in range(len(languages)):
            for j in range(len(categories)):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10, color="black"
                )

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="Strict Accuracy")
    fig.suptitle("Language x Category Strict Accuracy Heatmap", fontsize=14, fontweight="bold")
    return _save(fig, "08_language_category_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_all_charts() -> None:
    chart_functions = [
        chart_accuracy_by_language,
        chart_rag_vs_non_rag_accuracy,
        chart_hallucination_risk_distribution,
        chart_groundedness_by_language,
        chart_accuracy_by_category,
        chart_retrieval_similarity_scores,
        chart_claim_classification_distribution,
        chart_language_category_heatmap,
    ]

    saved_paths = []
    for fn in chart_functions:
        print(f"Generating: {fn.__name__}...")
        path = fn()
        saved_paths.append(path)
        print(f"  Saved to: {path}")

    print(f"\nAll {len(saved_paths)} charts saved to: {config.CHARTS_DIR}")


if __name__ == "__main__":
    generate_all_charts()