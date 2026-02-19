"""
Stage 4 — Build train/val/test CSVs.

Delegates to functions in scripts/create_training_dataset.py.

Supported modes
---------------
baseline   — plain context, no mixing (70/15/15 split)
mixsquad   — 10,000 randomly mixed context pairs
mixsquad2x — MixSQuAD doubled by reversing context order
mixkhanq   — ~653 mixed KhanQ entries, no split (evaluation set)
"""

import json
import os
from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig


def run(
    config: PipelineConfig,
    dataset: str = "squad",
    mode: str = "mixsquad",
    tool: Optional[str] = None,
) -> dict[str, Path]:
    """
    Build training CSVs from wikified enriched data.

    Parameters
    ----------
    dataset:
        ``"squad"`` or ``"khanq"``.
    mode:
        ``"baseline"``, ``"mixsquad"``, ``"mixsquad2x"``, or ``"mixkhanq"``.
    tool:
        Wikification tool to determine the input directory.

    Returns
    -------
    dict mapping ``"train"``, ``"val"``, ``"test"`` (or ``"data"``) to paths.
    """
    from scripts.create_training_dataset import (
        create_mixed_dataset,
        double_dataset,
        filter_by_token_length,
        create_splits,
        save_to_csv,
    )
    from transformers import T5Tokenizer

    mode = mode.lower()
    effective_tool = (tool or config.wikification.tool).lower()

    # Input: enriched filtered JSON
    wiki_dir = config.processed_dir(effective_tool)
    input_file = wiki_dir / f"enriched_{dataset}_filtered.json"

    # Output directory
    out_dir = config.training_dir(dataset, mode)

    # Check idempotency
    if mode == "mixkhanq":
        expected = out_dir / "data.csv"
    else:
        expected = out_dir / "train.csv"

    if expected.exists():
        print(f"[SKIP] already exists: {expected}")
        return _collect_outputs(out_dir, mode)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found: {input_file}\nRun wikify + topics stages first."
        )

    print(f"[RUN] dataset: {dataset}/{mode}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (used for token-length filtering and mixed-dataset creation)
    print("Loading T5 tokenizer…")
    tokenizer = T5Tokenizer.from_pretrained(config.training.model_name)

    # Load source data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} enriched entries from {input_file}")

    # Build dataset
    dc = config.dataset

    if mode in ("mixsquad", "mixsquad2x", "mixkhanq"):
        num_samples = dc.khanq_mix_samples if mode == "mixkhanq" else dc.mix_samples
        data = create_mixed_dataset(
            data, tokenizer, num_samples, dc.max_tokens, dc.seed
        )
        if mode == "mixsquad2x":
            data = double_dataset(data)
    else:
        # Baseline: filter by token length
        data = filter_by_token_length(data, tokenizer, dc.max_tokens)

    # Save
    if mode == "mixkhanq":
        path = out_dir / "data.csv"
        save_to_csv(data, path)
        return {"data": path}

    train_data, val_data, test_data = create_splits(
        data, dc.train_ratio, dc.val_ratio, dc.test_ratio
    )

    paths = {}
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        p = out_dir / f"{split_name}.csv"
        save_to_csv(split_data, p)
        paths[split_name] = p

    print(f"[DONE] dataset saved to {out_dir}/")
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_outputs(out_dir: Path, mode: str) -> dict[str, Path]:
    if mode == "mixkhanq":
        p = out_dir / "data.csv"
        return {"data": p} if p.exists() else {}

    result = {}
    for split in ("train", "val", "test"):
        p = out_dir / f"{split}.csv"
        if p.exists():
            result[split] = p
    return result
