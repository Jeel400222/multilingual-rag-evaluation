"""
build_notebooks.py

Generates the three Jupyter notebooks for this project using nbformat.
Building notebooks programmatically (rather than hand-writing .ipynb JSON)
keeps them valid and reviewable in version control diffs. After running
this once, the notebooks are ordinary .ipynb files -- open and edit them
freely in Jupyter afterward; re-running this script will overwrite them.

Run with:
    python -m src.build_notebooks
"""

import nbformat as nbf

from src import config

PATH_SETUP_CODE = """\
import sys
from pathlib import Path

# Notebooks run from notebooks/, but the src package lives at the project
# root -- add the project root to sys.path so `from src import ...` works.
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option("display.max_colwidth", 100)
"""


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 01 · Data Exploration\n"
            "Multilingual RAG Evaluation & LLM Hallucination Detection System\n\n"
            "This notebook explores the questions dataset and multilingual "
            "knowledge base built in Phase 2 (`src/build_seed_data.py`)."
        ),
        _code(PATH_SETUP_CODE),
        _code(
            "from src import data_loader\n\n"
            "questions_df = data_loader.load_questions()\n"
            "kb_df = data_loader.load_knowledge_base()\n\n"
            "print(f'Questions: {len(questions_df)}')\n"
            "print(f'Knowledge base documents: {len(kb_df)}')"
        ),
        _md(
            "## Dataset design\n"
            "This project uses **15 underlying facts** (5 Geography, 5 History, "
            "5 Science), each asked in all 3 languages -> **45 questions total**. "
            "Every question's `question_id` (e.g. `GEO01_en`) encodes its "
            "`fact_id` (`GEO01`) and language code (`en`), which lets us compare "
            "retrieval quality and accuracy for the *same fact* across languages "
            "-- a stronger design than comparing unrelated per-language question "
            "sets."
        ),
        _code(
            "print('Questions per language:')\n"
            "display(questions_df['language'].value_counts())\n\n"
            "print('\\nQuestions per category:')\n"
            "display(questions_df['category'].value_counts())"
        ),
        _code("questions_df.sample(6, random_state=42)"),
        _md(
            "## Knowledge base\n"
            "The knowledge base contains one matching document per fact per "
            "language (45 docs) plus **9 distractor documents** (topically "
            "related but not answering any question) that make retrieval "
            "non-trivial instead of a guaranteed top-1 match."
        ),
        _code(
            "print('KB documents per language:')\n"
            "display(kb_df['language'].value_counts())\n\n"
            "print(f\"\\nDistractor documents: {int(kb_df['is_distractor'].sum())}\")\n"
            "kb_df[kb_df['is_distractor']][['document_id', 'category', 'language', 'content']]"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "questions_df['language'].value_counts().plot(kind='bar', ax=axes[0], "
            "color='#0891b2', title='Questions by Language')\n"
            "questions_df['category'].value_counts().plot(kind='bar', ax=axes[1], "
            "color='#f59e0b', title='Questions by Category')\n"
            "for ax in axes:\n"
            "    ax.tick_params(axis='x', rotation=0)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md(
            "## Notes\n"
            "- All knowledge base content is **original reference material written "
            "for this project** (see `source` column) -- it does not claim to be "
            "scraped from any external publication.\n"
            "- Facts were chosen to be simple, verifiable, and unambiguous, "
            "making this a controlled evaluation environment rather than an "
            "open-domain benchmark."
        ),
    ]
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 02 · Retrieval Analysis\n"
            "Explores the FAISS retriever built in Phase 3 "
            "(`src/embeddings.py`, `src/retriever.py`) across all three "
            "supported languages."
        ),
        _code(PATH_SETUP_CODE),
        _code(
            "from src.retriever import Retriever\n\n"
            "retriever = Retriever()\n"
            "retriever.build_index()\n"
            "print(f'Index size: {retriever.index.ntotal} documents')"
        ),
        _md(
            "## Sample multilingual retrieval\n"
            "The same underlying question (capital of India) asked in all "
            "three languages, restricted to same-language retrieval."
        ),
        _code(
            "samples = [\n"
            "    ('What is the capital of India?', 'English'),\n"
            "    ('भारत की राजधानी क्या है?', 'Hindi'),\n"
            "    ('ભારતની રાજધાની શું છે?', 'Gujarati'),\n"
            "]\n\n"
            "for question, language in samples:\n"
            "    print(f'--- {language}: {question} ---')\n"
            "    results = retriever.retrieve(question, language, top_k=3)\n"
            "    for rank, doc in enumerate(results, start=1):\n"
            "        print(f'  Rank {rank} | score={doc.similarity_score:.4f} | "
            "{doc.document_id} | {doc.content[:70]}...')\n"
            "    print()"
        ),
        _md(
            "## Retrieval quality check (all 45 questions)\n"
            "Loaded from `outputs/tables/retrieval_quality_details.csv` and "
            "`retrieval_quality_summary.csv`, generated by "
            "`src/evaluate_retrieval.py` (Phase 3). Top-1 accuracy compares the "
            "top retrieved document's linked fact against the fact the question "
            "was written about."
        ),
        _code(
            "from src import config\n\n"
            "details_df = pd.read_csv(config.RETRIEVAL_QUALITY_DETAILS_CSV, encoding='utf-8-sig')\n"
            "summary_df = pd.read_csv(config.RETRIEVAL_QUALITY_SUMMARY_CSV, encoding='utf-8-sig')\n"
            "summary_df"
        ),
        _code(
            "incorrect = details_df[~details_df['top1_correct']]\n"
            "print(f'Questions with incorrect top-1 retrieval: {len(incorrect)}')\n"
            "incorrect[['question_id', 'language', 'category', 'expected_fact_id', 'retrieved_fact_id']]"
        ),
        _md("## Similarity score distribution by language"),
        _code(
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "languages = config.SUPPORTED_LANGUAGES\n"
            "data = [details_df[details_df.language == lang]['similarity_score'].dropna() "
            "for lang in languages]\n"
            "ax.boxplot(data, tick_labels=languages, patch_artist=True)\n"
            "ax.set_title('Top-1 Retrieval Similarity Score by Language')\n"
            "ax.set_ylabel('Cosine Similarity')\n"
            "plt.show()"
        ),
        _md(
            "## Notes\n"
            "- Retrieval uses `intfloat/multilingual-e5-base` embeddings with "
            "`query: ` / `passage: ` prefixes, L2-normalized, searched via FAISS "
            "`IndexFlatIP` (exact inner-product search, equivalent to cosine "
            "similarity on normalized vectors).\n"
            "- `IndexFlatIP` is appropriate at this scale (54 documents) -- an "
            "approximate index (IVF/HNSW) would add complexity with no "
            "practical benefit here.\n"
            "- Any retrieval errors reflect real embedding-model behavior on "
            "this dataset, not implementation bugs -- worth documenting "
            "honestly rather than hiding."
        ),
    ]
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 03 · RAG Evaluation\n"
            "Loads the full 45-question experiment (`src/evaluator.py`, "
            "Phase 6) and reviews accuracy, groundedness, and hallucination "
            "risk metrics, RAG vs Non-RAG."
        ),
        _code(PATH_SETUP_CODE),
        _code(
            "from src import config\n\n"
            "results_df = pd.read_csv(config.EXPERIMENT_RESULTS_CSV, encoding='utf-8-sig')\n"
            "response_metrics = pd.read_csv(config.RESPONSE_METRICS_SUMMARY_CSV, encoding='utf-8-sig')\n"
            "accuracy_by_language = pd.read_csv(config.ACCURACY_BY_LANGUAGE_CSV, encoding='utf-8-sig')\n"
            "accuracy_by_category = pd.read_csv(config.ACCURACY_BY_CATEGORY_CSV, encoding='utf-8-sig')\n"
            "rag_metrics = pd.read_csv(config.RAG_METRICS_SUMMARY_CSV, encoding='utf-8-sig')\n"
            "risk_distribution = pd.read_csv(config.HALLUCINATION_RISK_DISTRIBUTION_CSV, encoding='utf-8-sig')\n\n"
            "print(f'Experiment rows: {len(results_df)}')"
        ),
        _md("## Response metrics (RAG vs Non-RAG)"),
        _code("response_metrics"),
        _md("## Accuracy by language"),
        _code("accuracy_by_language"),
        _md("## Accuracy by category"),
        _code("accuracy_by_category"),
        _md("## RAG metrics summary (groundedness, claim rates, retrieval success)"),
        _code("rag_metrics"),
        _md("## Hallucination risk distribution"),
        _code("risk_distribution"),
        _md("## Charts (generated by `src/visualization.py`, Phase 7)"),
        _code(
            "chart_files = sorted(config.CHARTS_DIR.glob('*.png'))\n"
            "for chart_path in chart_files:\n"
            "    display(Image(filename=str(chart_path)))"
        ),
        _md(
            "## Example: inspecting one question end-to-end\n"
            "A single row from the experiment results, including retrieved "
            "context, both responses, and claim-level analysis."
        ),
        _code(
            "import json\n\n"
            "row = results_df[results_df['question_id'] == 'HIS01_gu'].iloc[0]\n"
            "print('Question:', row['question'])\n"
            "print('Reference answer:', row['reference_answer'])\n"
            "print()\n"
            "print('Non-RAG response:', row['non_rag_response'])\n"
            "print('Non-RAG label:', row['non_rag_evaluation_label'])\n"
            "print()\n"
            "print('RAG response:', row['rag_response'])\n"
            "print('RAG label:', row['rag_evaluation_label'])\n"
            "print()\n"
            "print('RAG claim analysis:')\n"
            "for claim in json.loads(row['rag_claim_analysis']):\n"
            "    print(f\"  [{claim['status']}] {claim['claim_text']} (score={claim['similarity_score']})\")"
        ),
        _md(
            "## Limitations (read before interpreting the metrics above)\n\n"
            "**1. Small, clean, well-separated dataset.** 45 questions across "
            "15 facts is enough to demonstrate the pipeline and produce a real, "
            "explainable RAG-vs-Non-RAG gap, but it is not a large-scale "
            "benchmark. Near-100% retrieval accuracy and uniformly 'Low' "
            "hallucination risk reflect this dataset's deliberate design "
            "(topically distinct facts, short answers), not proof the "
            "underlying methods are error-free on harder or noisier data.\n\n"
            "**2. Correctness checking is a lexical heuristic.** "
            "`evaluate_correctness()` uses normalized text matching (exact "
            "match for Strict, ~60% significant-word overlap for Relaxed) "
            "against the reference answer -- not semantic understanding. It "
            "will not catch every valid paraphrase and can be fooled by "
            "superficial word overlap in principle, though this did not appear "
            "to be an issue on this dataset's short, unambiguous answers.\n\n"
            "**3. Hallucination detection is a semantic similarity heuristic, "
            "not entailment.** A claim scoring above `CLAIM_SUPPORTED_THRESHOLD` "
            "is semantically close in wording to some retrieved passage -- this "
            "is not verified logical entailment. A claim scoring low is "
            "**unsupported by the retrieved context**, which is explicitly *not* "
            "the same as being **factually incorrect** -- these two dimensions "
            "are tracked independently throughout this project and should never "
            "be conflated when reading results.\n\n"
            "**4. Thresholds were calibrated on the 'supported' end of the "
            "distribution.** `src/calibrate_thresholds.py` measured similarity "
            "scores for RAG claims generated directly from their own retrieved "
            "context (a near-best-case scenario), confirming genuinely-grounded "
            "claims score well above `CLAIM_SUPPORTED_THRESHOLD = 0.80`. This "
            "dataset did not naturally produce many low-similarity claims, so "
            "the Medium/High hallucination-risk buckets are implemented and "
            "correct but not heavily stress-tested here."
        ),
        _md(
            "## Future improvements\n"
            "- Expand the dataset with harder, more ambiguous, or intentionally "
            "conflicting facts to stress-test the Medium/High hallucination-risk "
            "buckets.\n"
            "- Add an NLI-based entailment check for English as a secondary "
            "signal alongside the similarity heuristic (with an honest note "
            "that Hindi/Gujarati would remain similarity-only).\n"
            "- Introduce adversarial or out-of-domain questions to test "
            "retrieval failure modes and observe how groundedness degrades."
        ),
    ]
    return nb


def main() -> None:
    notebooks = {
        "01_data_exploration.ipynb": build_notebook_01(),
        "02_retrieval_analysis.ipynb": build_notebook_02(),
        "03_rag_evaluation.ipynb": build_notebook_03(),
    }

    for filename, nb in notebooks.items():
        path = config.NOTEBOOKS_DIR / filename
        nbf.write(nb, str(path))
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()