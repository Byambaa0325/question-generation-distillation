"""High-level question generation interface."""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

from ..models.teacher import BaseTeacher
from ..data.concept_bank import Concept, ConceptBank
from .prompts import get_prompt, LEVEL_GUIDANCE, TYPE_GUIDANCE


@dataclass
class GeneratedQuestion:
    """A generated question with metadata."""

    concept_id: str
    concept_text: str
    learner_level: str
    question_type: str
    question: str
    context: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "concept_text": self.concept_text,
            "learner_level": self.learner_level,
            "question_type": self.question_type,
            "question": self.question,
            "context": self.context,
        }


class QuestionGenerator:
    """Generate questions using teacher model."""

    def __init__(
        self,
        teacher: BaseTeacher,
        levels: list[str] | None = None,
        question_types: list[str] | None = None,
    ):
        """
        Initialize question generator.

        Args:
            teacher: Teacher model instance
            levels: List of learner levels
            question_types: List of question types
        """
        self.teacher = teacher
        self.levels = levels or ["beginner", "intermediate", "advanced"]
        self.question_types = question_types or ["recall", "explain", "apply"]

    def generate_for_concept(
        self,
        concept: Concept | str,
        level: str,
        question_type: str = "recall",
        context: Optional[str] = None,
    ) -> GeneratedQuestion:
        """
        Generate a single question for a concept.

        Args:
            concept: Concept object or concept text
            level: Learner level
            question_type: Type of question
            context: Optional source context

        Returns:
            GeneratedQuestion object
        """
        if isinstance(concept, Concept):
            concept_id = concept.id
            concept_text = concept.text
            context = context or concept.source_context
        else:
            concept_id = "manual"
            concept_text = concept

        question = self.teacher.generate_question(
            concept=concept_text,
            level=level,
            question_type=question_type,
            context=context,
        )

        return GeneratedQuestion(
            concept_id=concept_id,
            concept_text=concept_text,
            learner_level=level,
            question_type=question_type,
            question=question,
            context=context,
        )

    def generate_all_variations(
        self,
        concept: Concept | str,
        context: Optional[str] = None,
    ) -> list[GeneratedQuestion]:
        """
        Generate questions for all level and type combinations.

        Args:
            concept: Concept object or text
            context: Optional source context

        Returns:
            List of GeneratedQuestion objects
        """
        questions = []
        for level in self.levels:
            for qtype in self.question_types:
                q = self.generate_for_concept(concept, level, qtype, context)
                questions.append(q)
        return questions

    def generate_dataset(
        self,
        concept_bank: ConceptBank,
        samples_per_concept_level: int = 3,
        progress_callback=None,
    ) -> list[GeneratedQuestion]:
        """
        Generate a full dataset from a concept bank.

        Args:
            concept_bank: ConceptBank with concepts
            samples_per_concept_level: Questions per (concept, level) pair
            progress_callback: Optional callback(current, total)

        Returns:
            List of GeneratedQuestion objects
        """
        questions = []
        total_concepts = len(concept_bank)
        current = 0

        for concept in concept_bank:
            for level in self.levels:
                for qtype in self.question_types[:samples_per_concept_level]:
                    q = self.generate_for_concept(concept, level, qtype)
                    questions.append(q)

            current += 1
            if progress_callback:
                progress_callback(current, total_concepts)

        return questions

    def save_dataset(
        self,
        questions: list[GeneratedQuestion],
        path: str | Path,
    ):
        """Save generated questions to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [q.to_dict() for q in questions]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_dataset(path: str | Path) -> list[GeneratedQuestion]:
        """Load questions from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return [
            GeneratedQuestion(
                concept_id=item["concept_id"],
                concept_text=item["concept_text"],
                learner_level=item["learner_level"],
                question_type=item["question_type"],
                question=item["question"],
                context=item.get("context"),
            )
            for item in data
        ]
