"""
Stage 3 — Select best topic per QA pair.

Logic extracted from scripts/select_topics_paper.py.
Intersection of Wikipedia concepts between question and passage;
highest pageRank (Wikifier) or rho (WAT) wins.
"""

import json
from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig


def run(
    config: PipelineConfig,
    dataset: str = "all",
    tool: Optional[str] = None,
) -> dict[str, Path]:
    """
    Select topics from wikified data.

    Parameters
    ----------
    dataset:
        ``"squad"``, ``"khanq"``, or ``"all"``.
    tool:
        Override the wikification tool to determine subdirectory and score field.

    Returns
    -------
    dict mapping logical names to output paths:
      - ``"{ds}_enriched"``:  all entries (topic may be "NA")
      - ``"{ds}_filtered"``:  entries with a real topic
    """
    effective_tool = (tool or config.wikification.tool).lower()
    score_field = "pageRank" if effective_tool == "wikifier" else "rho"

    wiki_dir = config.processed_dir(effective_tool)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    for ds in _datasets(dataset):
        results.update(_select(wiki_dir, ds, score_field))

    return results


# ---------------------------------------------------------------------------
# Per-dataset topic selection
# ---------------------------------------------------------------------------


def _select(wiki_dir: Path, ds: str, score_field: str) -> dict[str, Path]:
    question_file = wiki_dir / f"wikified_{ds}_question.json"
    text_file = wiki_dir / f"wikified_{ds}_text.json"
    enriched_out = wiki_dir / f"enriched_{ds}.json"
    filtered_out = wiki_dir / f"enriched_{ds}_filtered.json"

    if enriched_out.exists() and filtered_out.exists():
        print(f"[SKIP] already exists: {enriched_out}")
        print(f"[SKIP] already exists: {filtered_out}")
        return {f"{ds}_enriched": enriched_out, f"{ds}_filtered": filtered_out}

    for path in [question_file, text_file]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}\nRun wikify stage first."
            )

    print(f"[RUN] topics: {ds} (score_field={score_field})")

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(text_file, "r", encoding="utf-8") as f:
        texts = json.load(f)

    text_index = {item["id"]: item for item in texts if "id" in item}

    enriched = []
    na_count = 0

    for i, question in enumerate(questions):
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1}/{len(questions)}")

        text_id = question.get("text_id")
        text_item = text_index.get(text_id)
        if not text_item:
            continue

        topic, score = _best_match(
            question.get("annotations", {}),
            text_item.get("annotations", []),
            score_field,
        )

        if topic == "NA":
            na_count += 1

        enriched.append(
            {
                "id": question.get("id"),
                "text_id": text_id,
                "question": question.get("text"),
                "text": text_item.get("text"),
                "topic": topic,
                f"topic_{score_field}": score,
            }
        )

    _write_json(enriched, enriched_out)

    filtered = [e for e in enriched if e["topic"] != "NA"]
    _write_json(filtered, filtered_out)

    pct_na = 100 * na_count / max(len(enriched), 1)
    print(
        f"  {ds}: {len(enriched)} total, {len(filtered)} with topics "
        f"({na_count} NA = {pct_na:.1f}%)"
    )

    return {f"{ds}_enriched": enriched_out, f"{ds}_filtered": filtered_out}


# ---------------------------------------------------------------------------
# Core logic (mirrors select_topics_paper.py)
# ---------------------------------------------------------------------------


def _best_match(
    question_annotations,
    text_annotations,
    score_field: str,
) -> tuple[str, float]:
    """Return (topic_title, score) or ("NA", 0) for a QA pair."""

    # Build set of titles from question annotations
    question_titles: dict[str, float] = {}
    if isinstance(question_annotations, dict):
        for ann in question_annotations.get("annotation_data", []):
            title = ann.get("title")
            if title:
                question_titles[title] = ann.get(score_field, 0)

    # Build set of titles from text annotations (chunked or single)
    text_title_set: set[str] = set()
    if isinstance(text_annotations, list):
        for chunk in text_annotations:
            if isinstance(chunk, dict):
                for ann in chunk.get("annotation_data", []):
                    if ann.get("title"):
                        text_title_set.add(ann["title"])
    elif isinstance(text_annotations, dict):
        for ann in text_annotations.get("annotation_data", []):
            if ann.get("title"):
                text_title_set.add(ann["title"])

    # Intersection
    common = [
        (title, score)
        for title, score in question_titles.items()
        if title in text_title_set
    ]
    if not common:
        return ("NA", 0)

    return max(common, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _datasets(dataset: str) -> list[str]:
    return ["squad", "khanq"] if dataset == "all" else [dataset]


def _write_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
