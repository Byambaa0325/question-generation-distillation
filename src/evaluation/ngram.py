"""
Per-sample n-gram metrics for question generation evaluation.

Functions
---------
bleu_ngrams(reference, hypothesis)
    Word-level BLEU-1/2/3/4 using SmoothingFunction.method2 (correct method).
bleu_ngrams_char(reference, hypothesis)
    Character-level BLEU (paper's original method — for direct comparison only).
token_f1(reference, hypothesis)
    Token-level F1 using NLTK word_tokenize.
"""

from __future__ import annotations


def bleu_ngrams(reference: str, hypothesis: str) -> tuple[float, float, float, float]:
    """
    Word-level BLEU-1/2/3/4 (correct method).

    Uses ``SmoothingFunction.method2``, matching ``evaluate_full_metrics.py``.

    Parameters
    ----------
    reference:  Ground-truth question string.
    hypothesis: Generated question string.

    Returns
    -------
    (bleu1, bleu2, bleu3, bleu4) floats in [0, 1].
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoothing = SmoothingFunction().method2
    ref_toks = reference.lower().split()
    hyp_toks = hypothesis.lower().split()

    return (
        sentence_bleu([ref_toks], hyp_toks, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        sentence_bleu([ref_toks], hyp_toks, weights=(0, 1, 0, 0), smoothing_function=smoothing),
        sentence_bleu([ref_toks], hyp_toks, weights=(0, 0, 1, 0), smoothing_function=smoothing),
        sentence_bleu([ref_toks], hyp_toks, weights=(0, 0, 0, 1), smoothing_function=smoothing),
    )


def bleu_ngrams_char(reference: str, hypothesis: str) -> tuple[float, float, float, float]:
    """
    Character-level BLEU-1/2/3/4 — the paper's original (buggy) method.

    The paper passed raw strings to ``sentence_bleu``, causing each *character*
    to be treated as a token.  This function replicates that behaviour so results
    can be compared directly with the paper's Table 2 numbers.

    Parameters
    ----------
    reference:  Ground-truth question string.
    hypothesis: Generated question string.

    Returns
    -------
    (bleu1, bleu2, bleu3, bleu4) floats in [0, 1].
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoothing = SmoothingFunction().method2
    # Passing strings (not token lists) → each character treated as a token
    return (
        sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0), smoothing_function=smoothing),
        sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0), smoothing_function=smoothing),
        sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1), smoothing_function=smoothing),
    )


def token_f1(reference: str, hypothesis: str) -> float:
    """
    Token-level F1 score (paper's method).

    Uses NLTK ``word_tokenize`` for tokenisation.

    Parameters
    ----------
    reference:  Ground-truth question.
    hypothesis: Generated question.

    Returns
    -------
    F1 float in [0, 1].
    """
    from nltk.tokenize import word_tokenize

    ref_toks = set(word_tokenize(reference.lower()))
    hyp_toks = set(word_tokenize(hypothesis.lower()))

    if not ref_toks or not hyp_toks:
        return 0.0

    common = ref_toks & hyp_toks
    if not common:
        return 0.0

    precision = len(common) / len(hyp_toks)
    recall = len(common) / len(ref_toks)
    return 2 * precision * recall / (precision + recall)
