"""
claim_analyzer.py

Splits an LLM response into individual factual claims (sentences).

This is a lightweight, rule-based sentence splitter -- not an NLP
sentence-segmentation model. Language-specific sentence terminators are
included: Latin '.', '!', '?' for English, plus the Devanagari danda '।'
and double danda '॥' used in Hindi (and occasionally Gujarati). A rule-
based split is reliable here because prompts.py constrains every LLM
response to 1-3 short, well-formed sentences, so punctuation boundaries
are a dependable signal.
"""

import re
from typing import List

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?।॥])\s+")
_MIN_CLAIM_LENGTH = 3  # filters stray punctuation-only fragments


def split_into_claims(text: str) -> List[str]:
    """
    Splits response text into a list of claim strings.
    Returns an empty list if text is empty or whitespace-only.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()
    raw_pieces = _SENTENCE_SPLIT_PATTERN.split(stripped)
    claims = [piece.strip() for piece in raw_pieces if piece.strip()]
    claims = [c for c in claims if len(c) >= _MIN_CLAIM_LENGTH]
    return claims


if __name__ == "__main__":
    samples = [
        "New Delhi is the capital of India.",
        "भारत की राजधानी नई दिल्ली है। यह भारत सरकार की सीट भी है।",
        "The capital is New Delhi! It has been the capital since 1911.",
        "",
        "   ",
    ]
    for sample in samples:
        print(f"Input:  {sample!r}")
        print(f"Claims: {split_into_claims(sample)}")
        print()