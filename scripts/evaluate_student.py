#!/usr/bin/env python3
"""Evaluate student model against teacher."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from config import get_settings
from src.models.student import T5Student
from src.evaluation.metrics import evaluate_questions, EvaluationResults


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained student model"
    )
    parser.add_argument(
        "--test-data", type=str, required=True,
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for generation"
    )
    args = parser.parse_args()

    # Load settings
    settings = get_settings(args.config)

    # Device setup
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load student model
    print(f"Loading model from {args.model}")
    student = T5Student.load(args.model, device=device)
    student.eval_mode()

    # Load test data
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data) as f:
        test_data = json.load(f)

    # Generate questions with student
    print("Generating questions with student model...")
    student_questions = []
    teacher_questions = []
    concepts = []

    for i in tqdm(range(0, len(test_data), args.batch_size)):
        batch = test_data[i:i + args.batch_size]

        batch_concepts = [item["concept_text"] for item in batch]
        batch_levels = [item["learner_level"] for item in batch]
        batch_types = [item.get("question_type", "recall") for item in batch]

        # Generate with student
        generated = student.generate_batch(
            concepts=batch_concepts,
            levels=batch_levels,
            question_types=batch_types,
        )

        student_questions.extend(generated)
        teacher_questions.extend([item["question"] for item in batch])
        concepts.extend(batch_concepts)

    # Evaluate
    print("Computing evaluation metrics...")
    results = evaluate_questions(
        student_questions=student_questions,
        teacher_questions=teacher_questions,
        concepts=concepts,
        metrics=["rouge", "bleu", "bert_score", "concept_coverage"],
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"ROUGE-1: {results.rouge_scores.get('rouge1', 0):.4f}")
    print(f"ROUGE-2: {results.rouge_scores.get('rouge2', 0):.4f}")
    print(f"ROUGE-L: {results.rouge_scores.get('rougeL', 0):.4f}")
    print(f"BLEU: {results.bleu_score:.4f}")
    print(f"BERTScore F1: {results.bert_score.get('f1', 0):.4f}")
    print(f"Concept Coverage: {results.concept_coverage:.4f}")
    print(f"Semantic Similarity: {results.custom_metrics.get('semantic_similarity', 0):.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "metrics": results.to_dict(),
        "samples": [
            {
                "concept": c,
                "teacher": t,
                "student": s,
            }
            for c, t, s in zip(concepts[:20], teacher_questions[:20], student_questions[:20])
        ],
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
