"""Dataset classes for knowledge distillation training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


@dataclass
class DistillationSample:
    """A single training sample for distillation."""

    concept_id: str
    concept_text: str
    learner_level: str
    question_type: str
    question: str  # teacher-generated question


class DistillationDataset(Dataset):
    """Dataset for training student model via distillation."""

    def __init__(
        self,
        samples: list[DistillationSample],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_output_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        return len(self.samples)

    def _format_input(self, sample: DistillationSample) -> str:
        """Format input for the student model."""
        # T5-style input format
        return (
            f"generate question: "
            f"concept: {sample.concept_text} "
            f"level: {sample.learner_level} "
            f"type: {sample.question_type}"
        )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        input_text = self._format_input(sample)
        target_text = sample.question

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token id with -100 for loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "concept_id": sample.concept_id,
            "learner_level": sample.learner_level,
        }

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_output_length: int = 128,
    ) -> "DistillationDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        samples = [
            DistillationSample(
                concept_id=item["concept_id"],
                concept_text=item["concept_text"],
                learner_level=item["learner_level"],
                question_type=item.get("question_type", "recall"),
                question=item["question"],
            )
            for item in data
        ]

        return cls(samples, tokenizer, max_input_length, max_output_length)

    def save(self, path: str | Path):
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "concept_id": s.concept_id,
                "concept_text": s.concept_text,
                "learner_level": s.learner_level,
                "question_type": s.question_type,
                "question": s.question,
            }
            for s in self.samples
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def create_dataloader(
    dataset: DistillationDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for distillation dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for batching."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "concept_ids": [item["concept_id"] for item in batch],
        "learner_levels": [item["learner_level"] for item in batch],
    }
