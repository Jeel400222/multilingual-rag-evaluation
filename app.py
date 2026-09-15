"""
app.py

Multilingual RAG Evaluation & LLM Hallucination Detection System
Streamlit application: an AI reliability dashboard, not a chatbot.

Three views (sidebar navigation):
    Home                  - project overview and pipeline explanation
    Try the System         - interactive question analysis (RAG vs Non-RAG)
    Evaluation Dashboard   - aggregate metrics and charts from the 45-question experiment
"""

import streamlit as st
import pandas as pd

from src import config, data_loader
from src.retriever import Retriever
from src.rag_pipeline import run_pipeline, PipelineResult

# ---------------------------------------------------------------------------
# Page config + minimal styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multilingual RAG Evaluation & Hallucination Detection",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .risk-low { color: #16a34a; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-high { color: #dc2626; font-weight: 600; }
    .risk-unable { color: #6b7280; font-weight: 600; }
    .pipeline-step {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_CSS_CLASS = {
    "Low": "risk-low",
    "Medium": "risk-medium",
    "High": "risk-high",
    "Unable to Determine": "risk-unable",
}
RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Unable to Determine": "⚪"}
CLAIM_ICON = {
    "Supported": "✅",
    "Partially Supported": "🟡",
    "Unsupported": "⚠️",
    "Unable to Determine": "❔",
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model and building retrieval index...")
def get_retriever() -> Retriever:
    retriever = Retriever()
    retriever.build_index()
    return retriever


@st.cache_data
def get_questions_df() -> pd.DataFrame:
    return data_loader.load_questions()


def startup_check() -> bool:
    """Validates config and knowledge base availability before rendering the app."""
    try:
        config.validate_config()
    except ValueError as exc:
        st.error(
            "**Configuration error.** " + str(exc)
        )
        return False

    try:
        data_loader.load_knowledge_base()
        data_loader.load_questions()
    except FileNotFoundError as exc:
        st.error(f"**Missing data files.** {exc}")
        return False
    except ValueError as exc:
        st.error(f"**Data validation error.** {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------


def render_home() -> None:
    st.title("Multilingual RAG Evaluation & LLM Hallucination Detection System")
    st.markdown(
        "*An interactive system for evaluating retrieval quality, response "
        "groundedness, and hallucination risk across English, Hindi, and Gujarati.*"
    )
    st.markdown("---")

    cols = st.columns(4)
    cards = [
        ("🌐", "3 Languages", "English, Hindi, Gujarati"),
        ("📚", "RAG Powered", "FAISS + multilingual embeddings"),
        ("🤖", "LLM Evaluation", "Groq · openai/gpt-oss-20b"),
        ("🔍", "Hallucination Detection", "Claim-level groundedness"),
    ]
    for col, (icon, title, subtitle) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                        <div style="font-size:1.8rem;">{icon}</div>
                        <div style="font-weight:600; margin-top:0.3rem;">{title}</div>
                        <div style="color:#64748b; font-size:0.85rem;">{subtitle}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### System Pipeline")
    steps = [
        "User Question",
        "Language Selection",
        "Multilingual Knowledge Base",
        "Embedding + Vector Search (FAISS)",
        "Retrieved Context",
        "Non-RAG Response  +  RAG Response",
        "Claim-Level Analysis",
        "Context Groundedness Check",
        "Hallucination Risk Classification",
        "Final Comparison + Metrics",
    ]
    for step in steps:
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center; color:#94a3b8;">↓</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "### Methodology note\n"
        "This system uses **semantic similarity between multilingual sentence "
        "embeddings** to check whether a claim is grounded in retrieved context — "
        "this is a heuristic, not automated logical entailment or fact-checking. "
        "A claim marked *Unsupported* means it wasn't semantically traceable to the "
        "retrieved documents; it does **not** mean the claim is factually false. "
        "**Context groundedness** and **factual correctness** are tracked as "
        "separate, independent dimensions throughout this project — never conflated."
    )


# ---------------------------------------------------------------------------
# Try the System page
# ---------------------------------------------------------------------------


def render_claim_inspector(result_groundedness, label: str) -> None:
    if result_groundedness is None:
        st.info(f"No {label} response was generated, so no claims could be analyzed.")
        return
    if result_groundedness.total_claims == 0:
        st.info(f"The {label} response did not contain any analyzable claims.")
        return

    for i, claim in enumerate(result_groundedness.claims, start=1):
        icon = CLAIM_ICON.get(claim.status, "❔")
        with st.expander(f"Claim {i}: {claim.claim_text}  {icon} {claim.status}"):
            if claim.similarity_score is not None:
                st.write(f"**Similarity score:** {claim.similarity_score:.3f}")
            if claim.best_evidence_document_id:
                st.write(f"**Closest evidence:** `{claim.best_evidence_document_id}`")
                st.caption(claim.best_evidence_text)
            st.write(f"**Why:** {claim.reasoning}")
            if claim.status == "Unsupported":
                st.caption(
                    "Note: 'Unsupported by retrieved context' is not the same as "
                    "'factually incorrect' — this claim simply wasn't traceable to "
                    "the documents retrieved for this query."
                )


def render_response_block(title: str, llm_response, groundedness, extra_note: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(extra_note)
    if not llm_response.success:
        st.error(f"Generation failed: {llm_response.error_message}")
        return

    st.write(f"**Model:** `{llm_response.model}`")
    st.write(f"**Response:** {llm_response.response_text}")

    if groundedness and groundedness.groundedness_score is not None:
        risk = groundedness.hallucination_risk
        icon = RISK_ICON.get(risk, "⚪")
        css_class = RISK_CSS_CLASS.get(risk, "risk-unable")
        col1, col2 = st.columns(2)
        col1.metric("Groundedness Score", f"{groundedness.groundedness_score:.2f}")
        col2.markdown(
            f'Hallucination Risk: <span class="{css_class}">{icon} {risk}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Groundedness could not be computed (no analyzable claims).")


def render_try_the_system() -> None:
    st.title("Try the System")
    st.caption(
        "Ask a factual question in English, Hindi, or Gujarati and see the full "
        "pipeline: retrieval, RAG vs Non-RAG generation, and claim-level "
        "hallucination analysis."
    )

    questions_df = get_questions_df()

    language = st.selectbox("Select Language", config.SUPPORTED_LANGUAGES)
    lang_questions = questions_df[questions_df["language"] == language]

    example_options = ["(Type my own question)"] + lang_questions["question"].tolist()
    selected_example = st.selectbox("Or pick an example question", example_options)

    if selected_example != "(Type my own question)":
        default_question = selected_example
        matched_row = lang_questions[lang_questions["question"] == selected_example].iloc[0]
        reference_answer = matched_row["reference_answer"]
    else:
        default_question = ""
        reference_answer = None

    question_text = st.text_input("Enter Question", value=default_question)

    analyze_clicked = st.button("🔍 Analyze Question", type="primary")

    if analyze_clicked:
        if not question_text or not question_text.strip():
            st.warning("Please enter a question before analyzing.")
        else:
            try:
                retriever = get_retriever()
                with st.spinner("Running retrieval, generation, and claim analysis..."):
                    result = run_pipeline(
                        retriever, question_text, language, reference_answer=reference_answer
                    )
                st.session_state["pipeline_result"] = result
            except ValueError as exc:
                st.warning(str(exc))
            except FileNotFoundError as exc:
                st.error(f"Missing required data: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"An unexpected error occurred: {exc}")

    result: PipelineResult = st.session_state.get("pipeline_result")
    if result is None:
        return

    st.markdown("---")
    st.markdown("### 📝 Question")
    st.write(f"**{result.question}**  ·  Language: {result.language}")
    if result.reference_answer:
        st.caption(f"Reference answer (for correctness checking): {result.reference_answer}")
    else:
        st.caption(
            "No reference answer available for this free-form question — "
            "correctness cannot be checked, but groundedness and hallucination "
            "risk are still shown below."
        )

    st.markdown("### 📚 Retrieved Knowledge")
    if not result.retrieved_documents:
        st.info("No documents were retrieved for this query.")
    for doc in result.retrieved_documents:
        with st.expander(f"`{doc.document_id}` · {doc.category} · score={doc.similarity_score:.3f}"):
            st.write(doc.content)
            st.caption(f"Source: {doc.source}")

    st.markdown("### 🤖 Non-RAG vs 🧠 RAG Responses")
    col1, col2 = st.columns(2)
    with col1:
        render_response_block(
            "🤖 Non-RAG Response",
            result.non_rag_response,
            result.non_rag_groundedness,
            "Generated from the model's own knowledge only. Groundedness here is "
            "checked incidentally against the same retrieved context — the model "
            "never saw it while answering.",
        )
        if result.non_rag_strict_correct is not None:
            st.write(f"**Evaluation label:** {result.non_rag_evaluation_label}")
    with col2:
        render_response_block(
            "🧠 RAG Response",
            result.rag_response,
            result.rag_groundedness,
            "Generated using the retrieved context above as grounding.",
        )
        if result.rag_strict_correct is not None:
            st.write(f"**Evaluation label:** {result.rag_evaluation_label}")

    st.markdown("### 🔍 Claim Inspector")
    tab1, tab2 = st.tabs(["Non-RAG Claims", "RAG Claims"])
    with tab1:
        render_claim_inspector(result.non_rag_groundedness, "Non-RAG")
    with tab2:
        render_claim_inspector(result.rag_groundedness, "RAG")


# ---------------------------------------------------------------------------
# Evaluation Dashboard page
# ---------------------------------------------------------------------------


def render_dashboard() -> None:
    st.title("Evaluation Dashboard")
    st.caption("Aggregate results from the 45-question multilingual experiment.")

    if not config.EXPERIMENT_RESULTS_CSV.exists():
        st.warning(
            "No experiment results found yet. Run `python -m src.evaluator` "
            "in your terminal first, then refresh this page."
        )
        return

    response_metrics = pd.read_csv(config.RESPONSE_METRICS_SUMMARY_CSV, encoding="utf-8-sig")
    accuracy_by_language = pd.read_csv(config.ACCURACY_BY_LANGUAGE_CSV, encoding="utf-8-sig")
    accuracy_by_category = pd.read_csv(config.ACCURACY_BY_CATEGORY_CSV, encoding="utf-8-sig")
    rag_metrics = pd.read_csv(config.RAG_METRICS_SUMMARY_CSV, encoding="utf-8-sig")
    risk_distribution = pd.read_csv(config.HALLUCINATION_RISK_DISTRIBUTION_CSV, encoding="utf-8-sig")

    rag_row = response_metrics[response_metrics.response_type == "rag"].iloc[0]
    non_rag_row = response_metrics[response_metrics.response_type == "non_rag"].iloc[0]
    rag_ground_row = rag_metrics[rag_metrics.response_type == "rag"].iloc[0]

    st.markdown("### Overall Metrics (RAG)")
    cols = st.columns(4)
    cols[0].metric(
        "Strict Accuracy", f"{rag_row.strict_accuracy:.1%}",
        delta=f"{(rag_row.strict_accuracy - non_rag_row.strict_accuracy):+.1%} vs Non-RAG",
    )
    cols[1].metric(
        "Relaxed Accuracy", f"{rag_row.relaxed_accuracy:.1%}",
        delta=f"{(rag_row.relaxed_accuracy - non_rag_row.relaxed_accuracy):+.1%} vs Non-RAG",
    )
    cols[2].metric(
        "Avg. Groundedness",
        f"{rag_ground_row.avg_groundedness_score:.2f}" if pd.notna(rag_ground_row.avg_groundedness_score) else "N/A",
    )
    cols[3].metric("Retrieval Success Rate", f"{rag_ground_row.retrieval_success_rate:.1%}")

    st.markdown("---")
    st.markdown("### Language Performance")
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        st.image(str(config.CHARTS_DIR / "01_accuracy_by_language.png"), use_column_width=True)
    with lang_col2:
        st.image(str(config.CHARTS_DIR / "04_groundedness_by_language.png"), use_column_width=True)
    st.dataframe(accuracy_by_language, use_container_width=True)

    st.markdown("### Category Performance")
    st.image(str(config.CHARTS_DIR / "05_accuracy_by_category.png"), use_column_width=True)
    st.dataframe(accuracy_by_category, use_container_width=True)

    st.markdown("### RAG vs Non-RAG")
    rvn_col1, rvn_col2 = st.columns(2)
    with rvn_col1:
        st.image(str(config.CHARTS_DIR / "02_rag_vs_non_rag_accuracy.png"), use_column_width=True)
    with rvn_col2:
        st.image(str(config.CHARTS_DIR / "03_hallucination_risk_distribution.png"), use_column_width=True)
    st.dataframe(response_metrics, use_container_width=True)
    st.dataframe(risk_distribution, use_container_width=True)
    st.markdown("### Retrieval Quality & Claim Classification")
    rq_col1, rq_col2 = st.columns(2)
    with rq_col1:
        st.image(str(config.CHARTS_DIR / "06_retrieval_similarity_scores.png"), use_column_width=True)
    with rq_col2:
        st.image(str(config.CHARTS_DIR / "07_claim_classification_distribution.png"), use_column_width=True)

    st.markdown("### Language × Category Heatmap")
    st.image(str(config.CHARTS_DIR / "08_language_category_heatmap.png"), use_column_width=True)

    st.markdown("---")
    st.caption(
        "Limitation: this evaluation set is small (45 questions) and deliberately "
        "clean/well-separated. Near-100% accuracy and uniformly 'Low' hallucination "
        "risk reflect this dataset's design, not proof the underlying methods are "
        "error-free on harder, noisier data."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not startup_check():
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["🏠 Home", "💬 Try the System", "📊 Evaluation Dashboard"]
    )

    if page == "🏠 Home":
        render_home()
    elif page == "💬 Try the System":
        render_try_the_system()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()