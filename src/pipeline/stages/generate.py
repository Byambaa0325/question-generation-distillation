"""
Stage 7 — Beam-search question generation with reranking.

Logic extracted from scripts/generate_questions.py.
Uses the paper's methodology:
  - num_beams=10, num_return_sequences=8
  - Select best candidate via sentence-transformer similarity

All heavy imports (torch, transformers) are deferred inside run().
"""

from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig


def run(
    config: PipelineConfig,
    topic: str,
    context: str,
    model_path: Optional[str | Path] = None,
    mode: str = "topic",
    num_beams: int = 10,
    num_return: int = 8,
) -> str:
    """
    Generate a question for a topic + context using the fine-tuned T5 model.

    Parameters
    ----------
    topic:
        Wikipedia concept title to condition on.
    context:
        The source passage.
    model_path:
        Explicit model directory.  If ``None``, uses
        ``config.model_dir(mode) / "best_model"``.
    mode:
        Model mode used when *model_path* is ``None``.
    num_beams:
        Beam width (paper uses 10).
    num_return:
        Number of candidate sequences (paper uses 8).

    Returns
    -------
    Best generated question string.
    """
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from scripts.generate_questions import generate_questions

    resolved_path = Path(model_path) if model_path else config.model_dir(mode) / "best_model"

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model not found: {resolved_path}\nRun the train stage first."
        )

    print(f"[RUN] generate — topic={topic!r}, model={resolved_path}")

    model = T5ForConditionalGeneration.from_pretrained(str(resolved_path))
    tokenizer = T5Tokenizer.from_pretrained(str(resolved_path), legacy=False)

    results = generate_questions(
        model,
        tokenizer,
        contexts=[context],
        topics=[topic],
        max_length=config.training.max_output_len,
        num_beams=num_beams,
        num_return=num_return,
        use_beam_selection=True,
    )

    question = results[0]["generated_question"]
    print(f"  -> {question}")
    return question
