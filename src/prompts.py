"""
prompts.py

Centralized prompt templates for Non-RAG and RAG LLM calls.

Keeping prompts here (rather than inline in llm_client.py) means every
prompt change is made in one place and stays consistent across the
pipeline, the Streamlit app, and any evaluation scripts.

Both prompt types explicitly instruct the model to answer in the same
language as the question and to keep the answer to 1-3 complete
sentences. The sentence constraint is deliberate: Phase 5's claim
analyzer splits responses into sentence-level claims, and short,
well-formed sentences make that split far more reliable than a long
rambling paragraph.
"""

from typing import List


def build_non_rag_system_prompt(language: str) -> str:
    return (
        f"You are a factual question-answering assistant. Answer the "
        f"user's question in {language}, using only your own general "
        f"knowledge. Do not mention whether you are or are not using any "
        f"external context. Write your answer as 1 to 3 complete, clear "
        f"sentences. Do not add disclaimers, hedging, or extra "
        f"commentary beyond the direct factual answer."
    )


def build_non_rag_user_prompt(question: str) -> str:
    return question


def build_rag_system_prompt(language: str) -> str:
    return (
        f"You are a factual question-answering assistant. Answer the "
        f"user's question in {language}, using ONLY the information "
        f"provided in the context below. If the context does not contain "
        f"enough information to answer confidently, say so honestly in "
        f"{language} rather than guessing or relying on outside "
        f"knowledge. Write your answer as 1 to 3 complete, clear "
        f"sentences. Do not add disclaimers or commentary beyond the "
        f"direct factual answer."
    )


def build_rag_user_prompt(question: str, context_documents: List[str]) -> str:
    if context_documents:
        context_block = "\n".join(
            f"[{i + 1}] {doc}" for i, doc in enumerate(context_documents)
        )
    else:
        context_block = "(No relevant context was retrieved.)"

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )