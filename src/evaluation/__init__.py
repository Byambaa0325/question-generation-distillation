"""
Evaluation utilities for question generation.

Sub-modules
-----------
ngram       Per-sample BLEU-1/2/3/4 (word-level + paper char-level) and token F1.
generation  Batch ROUGE-L, METEOR (via HF evaluate), and T5 perplexity.
models      T5Model, OllamaBaseline, GeminiBaseline + load_model() factory.
metrics     BERTScore, WikiSemRel, EvaluationResults, and composite evaluate_questions().
"""

from .ngram import bleu_ngrams, bleu_ngrams_char, token_f1
from .generation import compute_rouge_l, compute_meteor, compute_perplexity
from .models import T5Model, OllamaBaseline, GeminiBaseline, load_model
from .metrics import (
    EvaluationResults,
    compute_rouge,
    compute_bleu,
    compute_bert_score,
    compute_semantic_similarity,
    evaluate_concept_coverage,
    compute_wikisemrel,
    compute_bertscore_topic_aligned,
    evaluate_questions,
)

__all__ = [
    # ngram
    "bleu_ngrams",
    "bleu_ngrams_char",
    "token_f1",
    # generation
    "compute_rouge_l",
    "compute_meteor",
    "compute_perplexity",
    # models
    "T5Model",
    "OllamaBaseline",
    "GeminiBaseline",
    "load_model",
    # metrics
    "EvaluationResults",
    "compute_rouge",
    "compute_bleu",
    "compute_bert_score",
    "compute_semantic_similarity",
    "evaluate_concept_coverage",
    "compute_wikisemrel",
    "compute_bertscore_topic_aligned",
    "evaluate_questions",
]
