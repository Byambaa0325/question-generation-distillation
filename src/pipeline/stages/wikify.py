"""
Stage 2 — Annotate texts and questions with Wikipedia entities.

Uses src.wikification.get_wikifier() to dispatch to Wikifier.org or WAT.
"""

from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig
from src.wikification import get_wikifier


def run(
    config: PipelineConfig,
    dataset: str = "all",
    tool: Optional[str] = None,
    target: str = "all",
) -> dict[str, Path]:
    """
    Wikify texts and/or questions for the given dataset(s).

    Parameters
    ----------
    dataset:
        ``"squad"``, ``"khanq"``, or ``"all"``.
    tool:
        Override the wikification tool (``"wikifier"`` or ``"wat"``).
        Defaults to ``config.wikification.tool``.
    target:
        ``"texts"``, ``"questions"``, or ``"all"``.

    Returns
    -------
    dict mapping logical names to output paths.
    """
    effective_tool = (tool or config.wikification.tool).lower()
    wikifier = get_wikifier(effective_tool, config)

    out_dir = config.processed_dir(effective_tool)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(config.paths.processed)
    dataset = dataset.lower()
    target = target.lower()

    results: dict[str, Path] = {}

    for ds in _datasets(dataset):
        if target in ("texts", "all"):
            results.update(
                _wikify_file(
                    wikifier,
                    input_path=processed_dir / f"ready_{ds}_text.json",
                    output_path=out_dir / f"wikified_{ds}_text.json",
                    chunk_texts=True,
                    label=f"{ds} texts",
                    save_every=config.wikification.save_every,
                )
            )

        if target in ("questions", "all"):
            results.update(
                _wikify_file(
                    wikifier,
                    input_path=processed_dir / f"ready_{ds}_question.json",
                    output_path=out_dir / f"wikified_{ds}_question.json",
                    chunk_texts=False,
                    label=f"{ds} questions",
                    save_every=config.wikification.save_every,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _datasets(dataset: str) -> list[str]:
    if dataset == "all":
        return ["squad", "khanq"]
    return [dataset]


def _wikify_file(
    wikifier,
    input_path: Path,
    output_path: Path,
    chunk_texts: bool,
    label: str,
    save_every: int,
) -> dict[str, Path]:
    import json

    key = output_path.stem  # e.g. "wikified_squad_text"

    if output_path.exists():
        print(f"[SKIP] already exists: {output_path}")
        return {key: output_path}

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            f"Run 'pipeline.convert()' first."
        )

    print(f"[RUN] wikify: {label}")

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    wikifier.annotate_batch(
        items,
        output_path=output_path,
        save_every=save_every,
        chunk_texts=chunk_texts,
    )

    return {key: output_path}
