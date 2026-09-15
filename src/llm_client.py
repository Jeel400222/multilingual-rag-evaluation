"""
llm_client.py

Thin, defensive wrapper around the Groq API for generating Non-RAG and
RAG responses.

Design notes:
- API key / provider validation happens once via config.validate_config()
  and raises immediately -- a missing key is a setup problem the user
  should see right away, not something to silently work around.
- Individual generation calls (generate_non_rag_response /
  generate_rag_response) NEVER raise for ordinary API failures (rate
  limits, timeouts, transient errors). They retry with exponential
  backoff, and if all retries are exhausted, they return an LLMResponse
  with success=False and a human-readable error_message. This keeps the
  Streamlit app and evaluation scripts from crashing on a single flaky
  API call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional

from groq import (
    Groq,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)

from src import config, prompts

_client_instance: Optional[Groq] = None


@dataclass
class LLMResponse:
    response_text: Optional[str]
    prompt_type: str  # "non_rag" or "rag"
    language: str
    model: str
    success: bool
    error_message: Optional[str]
    latency_seconds: Optional[float]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_groq_client() -> Groq:
    """
    Returns a cached Groq client instance. Validates configuration on
    first call so a missing/invalid API key fails fast with a clear
    message rather than deep inside a network call.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    config.validate_config()
    _client_instance = Groq(api_key=config.API_KEY)
    return _client_instance


def _call_groq_with_retries(
    system_prompt: str, user_prompt: str
) -> tuple[Optional[str], Optional[str], float]:
    """
    Calls the Groq chat completion endpoint with exponential backoff retry.

    Returns:
        (response_text, error_message, latency_seconds)
        response_text is None if all retries failed; error_message
        describes why in that case.
    """
    client = get_groq_client()
    last_error_message = None

    for attempt in range(config.GROQ_MAX_RETRIES):
        start_time = time.monotonic()
        try:
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
                timeout=config.GROQ_TIMEOUT_SECONDS,
            )
            latency = time.monotonic() - start_time

            if not completion.choices:
                last_error_message = "Groq API returned no response choices."
                continue

            text = completion.choices[0].message.content
            if not text or not text.strip():
                last_error_message = "Groq API returned an empty response."
                continue

            return text.strip(), None, latency

        except RateLimitError as exc:
            last_error_message = f"Rate limit hit: {exc}"
        except APITimeoutError as exc:
            last_error_message = f"Request timed out: {exc}"
        except APIConnectionError as exc:
            last_error_message = f"Connection error: {exc}"
        except APIStatusError as exc:
            # Covers 4xx/5xx errors from the API (e.g. invalid model name,
            # malformed request, server-side errors).
            last_error_message = f"API returned an error status: {exc}"
            if exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                # Client errors other than rate limiting won't be fixed by
                # retrying (e.g. bad model name, invalid request) -- fail fast.
                break
        except Exception as exc:  # noqa: BLE001 - defensive catch-all
            last_error_message = f"Unexpected error calling Groq API: {exc}"

        if attempt < config.GROQ_MAX_RETRIES - 1:
            backoff_seconds = config.GROQ_RETRY_BACKOFF_SECONDS * (2 ** attempt)
            time.sleep(backoff_seconds)

    return None, last_error_message, 0.0


def generate_non_rag_response(question: str, language: str) -> LLMResponse:
    """Generates an answer using only the LLM's own knowledge (no retrieved context)."""
    if language not in config.SUPPORTED_LANGUAGES:
        return LLMResponse(
            response_text=None,
            prompt_type="non_rag",
            language=language,
            model=config.LLM_MODEL,
            success=False,
            error_message=(
                f"Unsupported language '{language}'. Must be one of "
                f"{config.SUPPORTED_LANGUAGES}."
            ),
            latency_seconds=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if not question or not question.strip():
        return LLMResponse(
            response_text=None,
            prompt_type="non_rag",
            language=language,
            model=config.LLM_MODEL,
            success=False,
            error_message="Question cannot be empty.",
            latency_seconds=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    system_prompt = prompts.build_non_rag_system_prompt(language)
    user_prompt = prompts.build_non_rag_user_prompt(question)

    response_text, error_message, latency = _call_groq_with_retries(
        system_prompt, user_prompt
    )

    return LLMResponse(
        response_text=response_text,
        prompt_type="non_rag",
        language=language,
        model=config.LLM_MODEL,
        success=response_text is not None,
        error_message=error_message,
        latency_seconds=latency if response_text is not None else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def generate_rag_response(
    question: str, language: str, context_documents: List[str]
) -> LLMResponse:
    """Generates an answer grounded in the provided retrieved context documents."""
    if language not in config.SUPPORTED_LANGUAGES:
        return LLMResponse(
            response_text=None,
            prompt_type="rag",
            language=language,
            model=config.LLM_MODEL,
            success=False,
            error_message=(
                f"Unsupported language '{language}'. Must be one of "
                f"{config.SUPPORTED_LANGUAGES}."
            ),
            latency_seconds=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if not question or not question.strip():
        return LLMResponse(
            response_text=None,
            prompt_type="rag",
            language=language,
            model=config.LLM_MODEL,
            success=False,
            error_message="Question cannot be empty.",
            latency_seconds=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    system_prompt = prompts.build_rag_system_prompt(language)
    user_prompt = prompts.build_rag_user_prompt(question, context_documents)

    response_text, error_message, latency = _call_groq_with_retries(
        system_prompt, user_prompt
    )

    return LLMResponse(
        response_text=response_text,
        prompt_type="rag",
        language=language,
        model=config.LLM_MODEL,
        success=response_text is not None,
        error_message=error_message,
        latency_seconds=latency if response_text is not None else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    print(f"Testing Groq LLM client with model: {config.LLM_MODEL}\n")

    test_question_en = "What is the capital of India?"
    test_context_en = [
        "New Delhi is the capital of India. It is also the seat of all "
        "three branches of the Government of India."
    ]

    print("=" * 70)
    print("NON-RAG RESPONSE (English)")
    print("=" * 70)
    non_rag_result = generate_non_rag_response(test_question_en, "English")
    if non_rag_result.success:
        print(f"Response: {non_rag_result.response_text}")
        print(f"Latency: {non_rag_result.latency_seconds:.2f}s")
    else:
        print(f"FAILED: {non_rag_result.error_message}")

    print()
    print("=" * 70)
    print("RAG RESPONSE (English)")
    print("=" * 70)
    rag_result = generate_rag_response(test_question_en, "English", test_context_en)
    if rag_result.success:
        print(f"Response: {rag_result.response_text}")
        print(f"Latency: {rag_result.latency_seconds:.2f}s")
    else:
        print(f"FAILED: {rag_result.error_message}")

    print()
    print("=" * 70)
    print("NON-RAG RESPONSE (Hindi)")
    print("=" * 70)
    test_question_hi = "भारत की राजधानी क्या है?"
    non_rag_result_hi = generate_non_rag_response(test_question_hi, "Hindi")
    if non_rag_result_hi.success:
        print(f"Response: {non_rag_result_hi.response_text}")
        print(f"Latency: {non_rag_result_hi.latency_seconds:.2f}s")
    else:
        print(f"FAILED: {non_rag_result_hi.error_message}")