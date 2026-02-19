"""Prompt templates for question generation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """A prompt template for question generation."""

    name: str
    template: str
    description: str

    def format(self, **kwargs) -> str:
        """Format the template with provided arguments."""
        return self.template.format(**kwargs)


# Level-specific guidance
LEVEL_GUIDANCE = {
    "beginner": (
        "Use simple vocabulary and straightforward phrasing. "
        "Assume minimal prior knowledge. Avoid jargon."
    ),
    "intermediate": (
        "Use standard terminology. Assume foundational understanding. "
        "Can include some technical terms with context."
    ),
    "advanced": (
        "Use precise technical language. Assume strong background. "
        "Can include nuanced, complex reasoning."
    ),
}

# Question type guidance
TYPE_GUIDANCE = {
    "recall": "Ask a direct factual question that tests memory of the concept.",
    "explain": "Ask a question requiring explanation of why or how something works.",
    "apply": "Ask a question requiring application of the concept to a scenario.",
    "compare": "Ask a question requiring comparison with related concepts.",
    "analyze": "Ask a question requiring analysis of implications or consequences.",
}


PROMPT_TEMPLATES = {
    "concept_extraction": PromptTemplate(
        name="concept_extraction",
        description="Extract atomic concepts from educational text",
        template="""Extract up to {max_concepts} atomic, testable concepts from this educational text.

Requirements for each concept:
- Express as a clear, single statement
- Must be testable with a question
- Should be independent of difficulty level
- Capture distinct knowledge pieces

Text:
{text}

List the concepts, one per line, numbered:""",
    ),
    "question_generation": PromptTemplate(
        name="question_generation",
        description="Generate a question for a concept at a level",
        template="""Generate a single educational question to assess understanding of the following concept.

Concept: {concept}

Learner Level: {level}
{level_guidance}

Question Type: {question_type}
{type_guidance}

Requirements:
- Question must directly assess the given concept
- Question must be appropriate for the specified learner level
- Question must be answerable from the concept or common knowledge at that level
- Generate only the question, no answer or explanation

Question:""",
    ),
    "question_generation_with_context": PromptTemplate(
        name="question_generation_with_context",
        description="Generate a question with source context reference",
        template="""Generate a single educational question to assess understanding of the following concept.

Concept: {concept}

Source Context:
{context}

Learner Level: {level}
{level_guidance}

Question Type: {question_type}
{type_guidance}

Requirements:
- Question must directly assess the given concept
- Question must be appropriate for the specified learner level
- Question must be answerable from the context provided
- Generate only the question, no answer or explanation

Question:""",
    ),
    "batch_question_generation": PromptTemplate(
        name="batch_question_generation",
        description="Generate multiple questions in one call",
        template="""Generate {num_questions} educational questions for the following concept, varying by learner level.

Concept: {concept}

Generate one question for each level:
- Beginner: Simple vocabulary, minimal prior knowledge assumed
- Intermediate: Standard terminology, foundational knowledge assumed
- Advanced: Technical language, strong background assumed

Format each question as:
[LEVEL]: Question text

Questions:""",
    ),
}


def get_prompt(
    template_name: str,
    concept: str,
    level: Optional[str] = None,
    question_type: Optional[str] = None,
    context: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Get a formatted prompt for question generation.

    Args:
        template_name: Name of the template to use
        concept: Concept text
        level: Learner level
        question_type: Type of question
        context: Optional source context
        **kwargs: Additional template arguments

    Returns:
        Formatted prompt string
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}")

    # Build format arguments
    format_args = {
        "concept": concept,
        **kwargs,
    }

    if level:
        format_args["level"] = level
        format_args["level_guidance"] = LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE["intermediate"])

    if question_type:
        format_args["question_type"] = question_type
        format_args["type_guidance"] = TYPE_GUIDANCE.get(question_type, TYPE_GUIDANCE["recall"])

    if context:
        format_args["context"] = context

    return template.format(**format_args)
