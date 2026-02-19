#!/usr/bin/env python3
"""Generate training data using teacher model."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from config import get_settings
from src.models.teacher import get_teacher
from src.data.concept_bank import ConceptBank
from src.generation.question_generator import QuestionGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate training data for distillation")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--contexts", type=str, required=True, help="Path to contexts JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="data/generated/training_data.json",
        help="Output path for generated data"
    )
    parser.add_argument(
        "--concept-bank", type=str, default=None,
        help="Path to existing concept bank (optional)"
    )
    parser.add_argument(
        "--save-concepts", type=str, default="data/concepts/concept_bank.json",
        help="Path to save concept bank"
    )
    args = parser.parse_args()

    # Load settings
    settings = get_settings(args.config)

    # Initialize teacher
    print(f"Initializing teacher model ({settings.teacher.backend})...")
    if settings.teacher.backend == "ollama":
        teacher = get_teacher(
            "ollama",
            model_name=settings.teacher.ollama.model_name,
            base_url=settings.teacher.ollama.base_url,
            temperature=settings.teacher.ollama.temperature,
            max_tokens=settings.teacher.ollama.max_tokens,
        )
    else:
        teacher = get_teacher(
            "gemini",
            model_name=settings.teacher.gemini.model_name,
            temperature=settings.teacher.gemini.temperature,
            max_tokens=settings.teacher.gemini.max_tokens,
        )

    # Load or create concept bank
    if args.concept_bank and Path(args.concept_bank).exists():
        print(f"Loading concept bank from {args.concept_bank}...")
        concept_bank = ConceptBank.load(args.concept_bank)
    else:
        print("Creating new concept bank...")
        concept_bank = ConceptBank(
            embedding_model=settings.concepts.embedding_model,
            similarity_threshold=settings.concepts.similarity_threshold,
        )

    # Load contexts
    print(f"Loading contexts from {args.contexts}...")
    with open(args.contexts) as f:
        contexts = json.load(f)

    # Extract concepts from contexts
    print("Extracting concepts from contexts...")
    for ctx in tqdm(contexts, desc="Extracting concepts"):
        text = ctx.get("text", ctx.get("content", ""))
        if not text:
            continue

        concepts = teacher.extract_concepts(
            text, max_concepts=settings.concepts.max_concepts_per_context
        )

        for concept_text in concepts:
            concept_bank.add(
                text=concept_text,
                source_context=text[:500],  # Truncate for storage
                deduplicate=True,
            )

    print(f"Concept bank contains {len(concept_bank)} unique concepts")

    # Save concept bank
    Path(args.save_concepts).parent.mkdir(parents=True, exist_ok=True)
    concept_bank.save(args.save_concepts)
    print(f"Saved concept bank to {args.save_concepts}")

    # Generate questions
    print("Generating questions...")
    generator = QuestionGenerator(
        teacher=teacher,
        levels=settings.learner_levels,
        question_types=settings.question_types,
    )

    questions = generator.generate_dataset(
        concept_bank=concept_bank,
        samples_per_concept_level=settings.data_generation.samples_per_concept_level,
        progress_callback=lambda c, t: print(f"\rProgress: {c}/{t}", end=""),
    )
    print()

    # Save generated data
    generator.save_dataset(questions, args.output)
    print(f"Saved {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
