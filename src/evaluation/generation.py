"""
Batch generation metrics: ROUGE-L, METEOR, and T5 perplexity.

All heavy imports (``evaluate``, ``numpy``) are deferred to call time.

Functions
---------
compute_rouge_l(predictions, references)
    ROUGE-L F-measure via HuggingFace ``evaluate``.
compute_meteor(predictions, references)
    METEOR score via HuggingFace ``evaluate``.
compute_perplexity(model_obj, items)
    Mean perplexity on reference questions (requires model with
    ``get_perplexity(topic, context, question)``).
"""

from __future__ import annotations

from typing import Any


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """
    Compute ROUGE-L F-measure via HuggingFace evaluate.

    Parameters
    ----------
    predictions: Generated question strings.
    references:  Reference question strings.

    Returns
    -------
    Mean ROUGE-L F-measure float in [0, 1].
    """
    if not predictions:
        return 0.0
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    out = rouge.compute(predictions=predictions, references=references)
    return float(out["rougeL"])


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """
    Compute METEOR score via HuggingFace evaluate.

    Parameters
    ----------
    predictions: Generated question strings.
    references:  Reference question strings.

    Returns
    -------
    Mean METEOR float in [0, 1].
    """
    if not predictions:
        return 0.0
    import evaluate as hf_evaluate

    meteor = hf_evaluate.load("meteor")
    out = meteor.compute(predictions=predictions, references=references)
    return float(out["meteor"])


def compute_perplexity(model_obj: Any, items: list[dict]) -> float:
    """
    Compute mean perplexity on reference questions.

    Calls ``model_obj.get_perplexity(topic, context, question)`` for each item.
    Items that raise exceptions are silently skipped.
    Returns ``float('inf')`` if the model has no ``get_perplexity`` method or
    no items could be scored.

    Parameters
    ----------
    model_obj: Model with a ``get_perplexity`` method (e.g. ``T5Model``).
    items:     List of dicts with keys ``"topic"``, ``"text"``, ``"question"``.

    Returns
    -------
    Mean perplexity float, or ``float('inf')``.
    """
    if not hasattr(model_obj, "get_perplexity"):
        return float("inf")

    import numpy as np

    scores: list[float] = []
    for item in items:
        try:
            p = model_obj.get_perplexity(item["topic"], item["text"], item["question"])
            scores.append(p)
        except Exception:
            pass

    return float(np.mean(scores)) if scores else float("inf")
