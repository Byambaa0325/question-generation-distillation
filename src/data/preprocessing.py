"""Text preprocessing and concept extraction utilities."""

import re
from typing import Optional


def preprocess_context(text: str) -> str:
    """
    Clean and normalize context text for concept extraction.

    Args:
        text: Raw educational text

    Returns:
        Cleaned text suitable for concept extraction
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation
    text = re.sub(r"[.!?]{2,}", ".", text)

    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_concepts_from_text(
    text: str,
    teacher_model,
    max_concepts: int = 5,
) -> list[str]:
    """
    Extract atomic concepts from educational text using teacher LLM.

    Args:
        text: Educational context text
        teacher_model: Teacher model instance with generate() method
        max_concepts: Maximum number of concepts to extract

    Returns:
        List of concept strings
    """
    prompt = f"""Extract {max_concepts} atomic, testable concepts from this educational text.

Requirements for each concept:
- Must be expressible as a single clear statement
- Must be testable by at least one question
- Should be independent of learner level wording
- Should capture a distinct piece of knowledge

Text:
{text}

Return only the concepts as a numbered list, one per line.
"""

    response = teacher_model.generate(prompt)
    concepts = parse_concept_list(response)

    return concepts[:max_concepts]


def parse_concept_list(response: str) -> list[str]:
    """Parse numbered list of concepts from LLM response."""
    concepts = []

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove numbering (e.g., "1.", "1)", "- ")
        cleaned = re.sub(r"^[\d]+[.)\-]\s*", "", line)
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)

        if cleaned:
            concepts.append(cleaned)

    return concepts


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks for processing.

    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within last 20% of chunk
            search_start = end - int(max_chunk_size * 0.2)
            sentence_end = -1

            for marker in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = text.rfind(marker, search_start, end)
                if pos > sentence_end:
                    sentence_end = pos + len(marker)

            if sentence_end > search_start:
                end = sentence_end

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def validate_concept(
    concept: str,
    context: str,
    nli_model=None,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """
    Validate that a concept is entailed by the context using NLI.

    Args:
        concept: Concept statement to validate
        context: Source context text
        nli_model: Optional NLI model for validation
        threshold: Minimum entailment score

    Returns:
        Tuple of (is_valid, entailment_score)
    """
    if nli_model is None:
        # Without NLI model, assume valid
        return True, 1.0

    # NLI expects (premise, hypothesis) format
    result = nli_model.predict(premise=context, hypothesis=concept)

    # Extract entailment score
    entailment_score = result.get("entailment", 0.0)

    return entailment_score >= threshold, entailment_score
