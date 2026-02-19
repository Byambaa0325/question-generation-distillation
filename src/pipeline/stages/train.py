"""
Stage 5 — Fine-tune T5-small using HuggingFace Trainer.

Standardised on the paper's T5 format:
  Input:  <topic> {topic} <context> {text}
  Target: {question}

Special tokens: <sep>, <space>
Saves best checkpoint (lowest val loss) + final model.

All heavy imports (torch, transformers) are deferred inside run() so that
``import src.pipeline`` stays fast even without GPU / transformers installed.
"""

from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig


def run(
    config: PipelineConfig,
    mode: str = "topic",
    dataset: str = "squad",
    tool: Optional[str] = None,
) -> Path:
    """
    Fine-tune T5-small on the training CSVs produced by the dataset stage.

    Parameters
    ----------
    mode:
        ``"baseline"``, ``"topic"`` (MixSQuAD), or ``"topic2x"`` (MixSQuAD2X).
    dataset:
        Source dataset name, used to locate training CSVs.

    Returns
    -------
    Path to the best model directory.
    """
    # Lazy heavy imports — only loaded when training is actually requested
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        T5ForConditionalGeneration,
        T5Tokenizer,
    )

    # ------------------------------------------------------------------
    # Inner dataset class (defined here to avoid top-level torch import)
    # ------------------------------------------------------------------

    class QGDataset(Dataset):
        """CSV dataset: text, topic, question columns → T5 input/target."""

        def __init__(self, data_file, tokenizer, max_input_len, max_output_len):
            df = pd.read_csv(data_file)
            self.tokenizer = tokenizer
            self.max_input_len = max_input_len
            self.max_output_len = max_output_len
            self.examples = [
                {
                    "input_text": f"<topic> {row['topic']} <context> {row['text']} ",
                    "target_text": str(row["question"]),
                }
                for _, row in df.iterrows()
            ]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            model_inputs = self.tokenizer(
                ex["input_text"],
                max_length=self.max_input_len,
                padding="max_length",
                truncation=True,
            )
            labels = self.tokenizer(
                ex["target_text"],
                max_length=self.max_output_len,
                padding="max_length",
                truncation=True,
            ).input_ids
            labels = [
                (lbl if lbl != self.tokenizer.pad_token_id else -100)
                for lbl in labels
            ]
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels,
            }

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------

    stage_mode_map = {
        "baseline": "baseline",
        "topic":    "mixsquad",
        "topic2x":  "mixsquad2x",
    }
    ds_mode = stage_mode_map.get(mode, mode)

    train_dir = config.training_dir(dataset, ds_mode)
    train_csv = train_dir / "train.csv"
    val_csv   = train_dir / "val.csv"

    model_out = config.model_dir(mode)
    best_dir  = model_out / "best_model"

    if best_dir.exists() and any(best_dir.iterdir()):
        print(f"[SKIP] already exists: {best_dir}")
        return best_dir

    for p in [train_csv, val_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"Training data not found: {p}\nRun dataset stage first."
            )

    print(f"[RUN] train: mode={mode}, dataset={dataset}")

    tc = config.training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}  |  Model: {tc.model_name}")

    # Load tokenizer + model
    tokenizer = T5Tokenizer.from_pretrained(tc.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(tc.model_name)

    num_added = tokenizer.add_tokens(tc.special_tokens)
    if num_added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Added special tokens: {tc.special_tokens}")

    train_dataset = QGDataset(train_csv, tokenizer, tc.max_input_len, tc.max_output_len)
    val_dataset   = QGDataset(val_csv,   tokenizer, tc.max_input_len, tc.max_output_len)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model_out.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_out),
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.batch,
        per_device_eval_batch_size=tc.batch,
        learning_rate=tc.lr,
        warmup_steps=tc.warmup_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        predict_with_generate=False,
        logging_steps=50,
        dataloader_num_workers=0,  # Windows compatibility
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save best model explicitly under a fixed name
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    print(f"[DONE] Best model saved to {best_dir}")
    return best_dir
