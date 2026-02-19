"""
Stage 1 — Convert raw datasets to wikifier-ready flat JSON.

Extracts from:
  - scripts/convert_squad.py
  - scripts/convert_khanq.py
"""

import json
from pathlib import Path

from src.pipeline.config import PipelineConfig


def run(config: PipelineConfig, dataset: str = "all") -> dict[str, Path]:
    """
    Convert raw SQuAD / KhanQ JSON → flat ``{id, text}`` JSONs.

    Parameters
    ----------
    dataset:
        ``"squad"``, ``"khanq"``, or ``"all"``.

    Returns
    -------
    dict mapping logical names to output paths, e.g.
    ``{"squad_text": Path(...), "squad_question": Path(...)}``.
    """
    dataset = dataset.lower()
    out_dir = Path(config.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    if dataset in ("squad", "all"):
        results.update(_convert_squad(config, out_dir))

    if dataset in ("khanq", "all"):
        results.update(_convert_khanq(config, out_dir))

    return results


# ---------------------------------------------------------------------------
# SQuAD
# ---------------------------------------------------------------------------


def _convert_squad(config: PipelineConfig, out_dir: Path) -> dict[str, Path]:
    out_text = out_dir / "ready_squad_text.json"
    out_question = out_dir / "ready_squad_question.json"

    if out_text.exists() and out_question.exists():
        print(f"[SKIP] already exists: {out_text}")
        print(f"[SKIP] already exists: {out_question}")
        return {"squad_text": out_text, "squad_question": out_question}

    print("[RUN] convert: squad")

    squad_file = config.raw_path("train-v1.1.json")
    if not squad_file.exists():
        raise FileNotFoundError(f"SQuAD file not found: {squad_file}")

    with open(squad_file, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    text_entries = []
    question_entries = []
    text_id_counter = 0

    for article in squad_data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            text_entry = {
                "id": f"text_{text_id_counter}",
                "title": title,
                "text": context,
            }
            text_entries.append(text_entry)
            text_id = text_entry["id"]
            text_id_counter += 1

            for qa in paragraph["qas"]:
                question_entries.append(
                    {
                        "id": qa["id"],
                        "text_id": text_id,
                        "title": title,
                        "text": qa["question"],
                    }
                )

    _write_json(text_entries, out_text)
    _write_json(question_entries, out_question)

    print(f"  SQuAD: {len(text_entries)} contexts, {len(question_entries)} questions")
    return {"squad_text": out_text, "squad_question": out_question}


# ---------------------------------------------------------------------------
# KhanQ
# ---------------------------------------------------------------------------


def _convert_khanq(config: PipelineConfig, out_dir: Path) -> dict[str, Path]:
    out_text = out_dir / "ready_khanq_text.json"
    out_question = out_dir / "ready_khanq_question.json"

    if out_text.exists() and out_question.exists():
        print(f"[SKIP] already exists: {out_text}")
        print(f"[SKIP] already exists: {out_question}")
        return {"khanq_text": out_text, "khanq_question": out_question}

    print("[RUN] convert: khanq")

    khanq_file = config.raw_path("KhanQ.json")
    if not khanq_file.exists():
        raise FileNotFoundError(f"KhanQ file not found: {khanq_file}")

    with open(khanq_file, "r", encoding="utf-8") as f:
        khanq_data = json.load(f)

    text_entries = []
    question_entries = []

    for i, entry in enumerate(khanq_data):
        source = entry.get("Source", "Unknown")
        context = entry.get("Context", "")
        question = entry.get("Question", "")

        text_entry = {"id": f"khanq_text_{i}", "source": source, "text": context}
        text_entries.append(text_entry)

        question_entries.append(
            {
                "id": f"khanq_q_{i}",
                "text_id": text_entry["id"],
                "source": source,
                "text": question,
            }
        )

    _write_json(text_entries, out_text)
    _write_json(question_entries, out_question)

    print(f"  KhanQ: {len(text_entries)} contexts, {len(question_entries)} questions")
    return {"khanq_text": out_text, "khanq_question": out_question}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
