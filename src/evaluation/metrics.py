"""Evaluation metrics for question generation."""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
from pathlib import Path

import numpy as np


@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    rouge_scores: dict = field(default_factory=dict)
    bleu_score: float = 0.0
    bert_score: dict = field(default_factory=dict)
    concept_coverage: float = 0.0
    level_appropriateness: float = 0.0
    custom_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rouge": self.rouge_scores,
            "bleu": self.bleu_score,
            "bert_score": self.bert_score,
            "concept_coverage": self.concept_coverage,
            "level_appropriateness": self.level_appropriateness,
            "custom": self.custom_metrics,
        }

    def save(self, path: str | Path):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """
    Compute ROUGE scores.

    Args:
        predictions: Generated questions
        references: Reference questions

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        return {k: float(np.mean(v)) for k, v in scores.items()}

    except ImportError:
        print("rouge_score not installed. Skipping ROUGE computation.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> float:
    """
    Compute BLEU score.

    Args:
        predictions: Generated questions
        references: Reference questions

    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smoother = SmoothingFunction().method1
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoother)
            scores.append(score)

        return float(np.mean(scores))

    except ImportError:
        print("nltk not installed. Skipping BLEU computation.")
        return 0.0


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
) -> dict:
    """
    Compute BERTScore.

    Args:
        predictions: Generated questions
        references: Reference questions
        model_type: BERT model to use

    Returns:
        Dictionary with precision, recall, f1
    """
    try:
        from bert_score import score

        P, R, F1 = score(
            predictions,
            references,
            model_type=model_type,
            verbose=False,
        )

        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }

    except ImportError:
        print("bert_score not installed. Skipping BERTScore computation.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_semantic_similarity(
    predictions: list[str],
    references: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Compute semantic similarity using sentence embeddings.

    Args:
        predictions: Generated questions
        references: Reference questions
        model_name: Sentence transformer model

    Returns:
        Average cosine similarity
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        pred_embeddings = model.encode(predictions, normalize_embeddings=True)
        ref_embeddings = model.encode(references, normalize_embeddings=True)

        # Compute cosine similarity (embeddings are normalized)
        similarities = np.sum(pred_embeddings * ref_embeddings, axis=1)

        return float(np.mean(similarities))

    except ImportError:
        print("sentence_transformers not installed. Skipping semantic similarity.")
        return 0.0


def evaluate_concept_coverage(
    questions: list[str],
    concepts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
) -> float:
    """
    Evaluate whether questions cover their target concepts.

    Args:
        questions: Generated questions
        concepts: Target concepts
        model_name: Sentence transformer model
        threshold: Minimum similarity for coverage

    Returns:
        Proportion of questions covering their concept
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        q_embeddings = model.encode(questions, normalize_embeddings=True)
        c_embeddings = model.encode(concepts, normalize_embeddings=True)

        similarities = np.sum(q_embeddings * c_embeddings, axis=1)
        coverage = np.mean(similarities >= threshold)

        return float(coverage)

    except ImportError:
        print("sentence_transformers not installed. Skipping concept coverage.")
        return 0.0


# ---------------------------------------------------------------------------
# WikiSemRel — paper's primary evaluation metric (Table 4)
# Reference: eval_relatedness.ipynb (Topic-controllable-Question-Generator)
# ---------------------------------------------------------------------------

def _wat_annotate(text: str, token: str, max_retries: int = 3) -> list[dict]:
    """
    Annotate text with WAT entity linking.

    Uses the exact method parameters from the paper's eval_relatedness.ipynb
    (cell-1) for reproducibility.

    Args:
        text:        Input text (question or topic title).
        token:       WAT API gcube-token.
        max_retries: Number of retry attempts on failure.

    Returns:
        List of dicts with wiki_id, wiki_title, rho.
    """
    import requests

    url = "https://wat.d4science.org/wat/tag/tag"
    # Paper's exact annotation pipeline (eval_relatedness.ipynb cell-1)
    payload = [
        ("gcube-token", token),
        ("text", text),
        ("lang", "en"),
        ("tokenizer", "nlp4j"),
        ("debug", 9),
        ("method",
         "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,"
         "prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,"
         "voting:relatedness=lm,ranker:model=0046.model,"
         "confidence:model=pruner-wiki.linear"),
    ]

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "wiki_id":    ann["id"],
                        "wiki_title": ann["title"],
                        "rho":        ann["rho"],
                    }
                    for ann in data.get("annotations", [])
                ]
            elif resp.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception:
            break

    return []


def _wat_relatedness(
    wiki_ids: list[int],
    token: str,
    relatedness_type: str = "jaccard",
    max_retries: int = 3,
) -> float:
    """
    Call WAT relatedness/graph API and return max pairwise relatedness.

    Per the paper (eval_relatedness.ipynb cell-5 and cell-8):
      - Sends combined list of question-entity IDs + topic-entity IDs.
      - Returns max relatedness across all returned pairs.
      - Returns 0.0 if fewer than 2 IDs or API error.

    Args:
        wiki_ids:         List of Wikipedia page IDs (question entities + topic entity).
        token:            WAT API gcube-token.
        relatedness_type: 'jaccard' (outbound link overlap, best per paper Table 2)
                          or 'lm' (word2vec / language model co-occurrence).

    Returns:
        Max pairwise relatedness score in [0, 1].
    """
    import requests

    if len(wiki_ids) < 2:
        return 0.0

    url = "https://wat.d4science.org/wat/relatedness/graph"
    params = {
        "gcube-token":  token,
        "lang":         "en",
        "ids":          wiki_ids,
        "relatedness":  relatedness_type,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get("pairs", [])
                if pairs:
                    return float(max(p["relatedness"] for p in pairs))
                return 0.0
            elif resp.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception:
            break

    return 0.0


def compute_wikisemrel(
    questions_t: list[str],
    questions_t_prime: list[str],
    topics: list[str],
    alt_topics: list[str],
    token: str,
    relatedness_type: str = "jaccard",
    api_delay: float = 0.1,
) -> dict:
    """
    Compute WikiSemRel topic-alignment scores (paper Table 4).

    Per the paper (eval_relatedness.ipynb):
      1. Wikify each generated question with WAT to get Wikipedia entity IDs.
      2. Wikify each topic title with WAT to get its Wikipedia entity ID.
      3. Call WAT relatedness/graph API with (question IDs + topic ID).
      4. Take max pairwise relatedness score.
      5. Score = 1.0 if any question entity wiki_id equals the topic wiki_id
         (exact match — paper's update_and_save_scores, cell-8).
      6. Repeat for the alternative topic (t') to get q̂_t' scores.
      7. Report mean_t, mean_t_prime, and their difference (key metric).

    Args:
        questions_t:       Questions generated with prescribed topic t (q̂_t).
        questions_t_prime: Questions generated with alternative topic t' (q̂_t').
        topics:            Prescribed topic Wikipedia titles (e.g. "Mary_(mother_of_Jesus)").
        alt_topics:        Alternative topic Wikipedia titles (topic2 from MixSQuAD).
        token:             WAT API gcube-token (from config/wat_config.json).
        relatedness_type:  'jaccard' (default, best per Table 2 MAE=0.23)
                           or 'lm' (w2v, embedding-based co-occurrence).
        api_delay:         Seconds to sleep between API calls (be polite to the API).

    Returns:
        dict with:
          scores_t        — per-sample scores for q̂_t
          scores_t_prime  — per-sample scores for q̂_t'
          mean_t          — mean WikiSemRel(q̂_t, t)
          mean_t_prime    — mean WikiSemRel(q̂_t', t')
          mean_diff       — mean_t - mean_t_prime  (paper's headline metric)
          relatedness_type

    Example:
        results = compute_wikisemrel(
            questions_t       = ["What is photosynthesis?", ...],
            questions_t_prime = ["What is gravity?", ...],
            topics            = ["Photosynthesis", ...],
            alt_topics        = ["Gravity", ...],
            token             = "YOUR-GCUBE-TOKEN",
        )
        print(f"WikiSimRel diff: {results['mean_diff']:.3f}")
        # Paper's TopicQG: 0.595, TopicQG2X: 0.680
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    scores_t       = []
    scores_t_prime = []

    iterator = _tqdm(
        zip(questions_t, questions_t_prime, topics, alt_topics),
        total=len(questions_t),
        desc=f"WikiSemRel ({relatedness_type})",
    )

    for q_t, q_tp, topic, alt_topic in iterator:

        # ---- Prescribed topic: WikiSemRel(q̂_t, t) ----
        q_t_anns    = _wat_annotate(q_t,    token)
        topic_anns  = _wat_annotate(topic,  token)
        time.sleep(api_delay)

        q_t_ids    = [a["wiki_id"] for a in q_t_anns]
        topic_ids  = [a["wiki_id"] for a in topic_anns]

        if set(q_t_ids) & set(topic_ids):
            # Exact match: question directly mentions the topic entity
            score_t = 1.0
        elif q_t_ids and topic_ids:
            score_t = _wat_relatedness(q_t_ids + topic_ids, token, relatedness_type)
            time.sleep(api_delay)
        else:
            score_t = 0.0

        scores_t.append(score_t)

        # ---- Alternative topic: WikiSemRel(q̂_t', t') ----
        q_tp_anns      = _wat_annotate(q_tp,      token)
        alt_topic_anns = _wat_annotate(alt_topic, token)
        time.sleep(api_delay)

        q_tp_ids      = [a["wiki_id"] for a in q_tp_anns]
        alt_topic_ids = [a["wiki_id"] for a in alt_topic_anns]

        if set(q_tp_ids) & set(alt_topic_ids):
            score_tp = 1.0
        elif q_tp_ids and alt_topic_ids:
            score_tp = _wat_relatedness(q_tp_ids + alt_topic_ids, token, relatedness_type)
            time.sleep(api_delay)
        else:
            score_tp = 0.0

        scores_t_prime.append(score_tp)

    mean_t      = float(np.mean(scores_t))       if scores_t       else 0.0
    mean_t_prime = float(np.mean(scores_t_prime)) if scores_t_prime else 0.0

    return {
        "scores_t":        scores_t,
        "scores_t_prime":  scores_t_prime,
        "mean_t":          mean_t,
        "mean_t_prime":    mean_t_prime,
        "mean_diff":       mean_t - mean_t_prime,
        "relatedness_type": relatedness_type,
    }


# ---------------------------------------------------------------------------
# BERTScore for topic alignment (paper Table 4)
# ---------------------------------------------------------------------------

def compute_bertscore_topic_aligned(
    questions_t: list[str],
    questions_t_prime: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    remove_stopwords: bool = True,
) -> dict:
    """
    Compute BERTScore for topic alignment evaluation (paper Table 4).

    Per the paper: "BERTScore: stopwords excluded before computation."

    Computes BERTScore F1 between:
      - q̂_t (prescribed topic question) vs reference question → f1_t
      - q̂_t' (alternative topic question) vs reference question → f1_t_prime

    The difference f1_t - f1_t_prime is the alignment signal.

    Args:
        questions_t:       Questions generated with prescribed topic.
        questions_t_prime: Questions generated with alternative topic.
        references:        Reference (ground-truth) questions.
        model_type:        BERT model for BERTScore.
        remove_stopwords:  Remove stopwords before scoring (paper does this).

    Returns:
        dict with f1_t, f1_t_prime, diff, and per-sample lists.
    """
    try:
        from bert_score import score as bert_score_fn
        import nltk

        if remove_stopwords:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words("english"))

            def remove_sw(text):
                return " ".join(
                    w for w in text.lower().split() if w not in stop_words
                )

            questions_t       = [remove_sw(q) for q in questions_t]
            questions_t_prime = [remove_sw(q) for q in questions_t_prime]
            references        = [remove_sw(r) for r in references]

        _, _, F1_t = bert_score_fn(
            questions_t, references, model_type=model_type, verbose=False
        )
        _, _, F1_tp = bert_score_fn(
            questions_t_prime, references, model_type=model_type, verbose=False
        )

        f1_t      = float(F1_t.mean())
        f1_t_prime = float(F1_tp.mean())

        return {
            "f1_t":         f1_t,
            "f1_t_prime":   f1_t_prime,
            "diff":         f1_t - f1_t_prime,
            "scores_t":     F1_t.tolist(),
            "scores_t_prime": F1_tp.tolist(),
        }

    except ImportError:
        print("bert_score not installed. Skipping BERTScore topic alignment.")
        return {"f1_t": 0.0, "f1_t_prime": 0.0, "diff": 0.0,
                "scores_t": [], "scores_t_prime": []}


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def evaluate_questions(
    student_questions: list[str],
    teacher_questions: list[str],
    concepts: Optional[list[str]] = None,
    metrics: list[str] | None = None,
) -> EvaluationResults:
    """
    Comprehensive evaluation of generated questions.

    Args:
        student_questions: Questions from student model
        teacher_questions: Questions from teacher model (reference)
        concepts: Optional list of target concepts
        metrics: List of metrics to compute

    Returns:
        EvaluationResults object
    """
    if metrics is None:
        metrics = ["rouge", "bleu", "bert_score"]

    results = EvaluationResults()

    if "rouge" in metrics:
        results.rouge_scores = compute_rouge(student_questions, teacher_questions)

    if "bleu" in metrics:
        results.bleu_score = compute_bleu(student_questions, teacher_questions)

    if "bert_score" in metrics:
        results.bert_score = compute_bert_score(student_questions, teacher_questions)

    if concepts and "concept_coverage" in metrics:
        results.concept_coverage = evaluate_concept_coverage(
            student_questions, concepts
        )

    # Semantic similarity as additional metric
    results.custom_metrics["semantic_similarity"] = compute_semantic_similarity(
        student_questions, teacher_questions
    )

    return results
