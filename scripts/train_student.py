#!/usr/bin/env python3
"""Train student model via knowledge distillation (supports full fine-tuning and QLoRA)."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from config import get_settings
from src.data.dataset import DistillationDataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--train-data", type=str, required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--eval-data", type=str, default=None,
        help="Path to evaluation data JSON (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["full", "qlora"], default=None,
        help="Training mode: full fine-tuning or qlora (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load settings
    settings = get_settings(args.config)

    # Override settings with CLI args
    output_dir = args.output_dir or settings.training.output_dir
    training_mode = args.mode or getattr(settings.training, "mode", "qlora")
    num_epochs = args.epochs or settings.training.num_epochs
    batch_size = args.batch_size or settings.training.batch_size
    learning_rate = args.learning_rate or settings.training.learning_rate

    # Device setup
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training mode: {training_mode}")

    # Load tokenizer
    model_name = settings.student.model_name
    print(f"Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    print(f"Loading training data from {args.train_data}")
    train_dataset = DistillationDataset.from_json(
        args.train_data,
        tokenizer=tokenizer,
        max_input_length=settings.student.max_input_length,
        max_output_length=settings.student.max_output_length,
    )
    train_dataloader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    eval_dataloader = None
    if args.eval_data:
        print(f"Loading evaluation data from {args.eval_data}")
        eval_dataset = DistillationDataset.from_json(
            args.eval_data,
            tokenizer=tokenizer,
            max_input_length=settings.student.max_input_length,
            max_output_length=settings.student.max_output_length,
        )
        eval_dataloader = create_dataloader(
            eval_dataset, batch_size=batch_size, shuffle=False
        )

    # Train with appropriate method
    if training_mode == "qlora":
        train_qlora(
            model_name=model_name,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            settings=settings,
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
    else:
        train_full(
            model_name=model_name,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            settings=settings,
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )


def train_qlora(
    model_name,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    settings,
    output_dir,
    num_epochs,
    learning_rate,
):
    """Train with QLoRA."""
    from src.distillation import QLoRAConfig, QLoRATrainer

    # Build QLoRA config from settings
    qlora_settings = getattr(settings, "qlora", None)
    if qlora_settings:
        qlora_config = QLoRAConfig(
            load_in_4bit=getattr(qlora_settings, "load_in_4bit", True),
            bnb_4bit_quant_type=getattr(qlora_settings, "bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(qlora_settings, "bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_use_double_quant=getattr(qlora_settings, "bnb_4bit_use_double_quant", True),
            lora_r=getattr(qlora_settings, "lora_r", 16),
            lora_alpha=getattr(qlora_settings, "lora_alpha", 32),
            lora_dropout=getattr(qlora_settings, "lora_dropout", 0.05),
            gradient_checkpointing=getattr(qlora_settings, "gradient_checkpointing", True),
        )
    else:
        qlora_config = QLoRAConfig()

    # Initialize trainer
    trainer = QLoRATrainer(
        model_name=model_name,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        qlora_config=qlora_config,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_ratio=settings.training.warmup_ratio,
        weight_decay=settings.training.weight_decay,
        max_grad_norm=settings.training.max_grad_norm,
        gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        output_dir=output_dir,
        save_strategy=settings.training.save_strategy,
        save_total_limit=settings.training.save_total_limit,
        logging_steps=settings.training.logging_steps,
        eval_steps=settings.training.eval_steps,
        save_merged=getattr(settings.training, "save_merged", False),
    )

    # Train
    print("\nStarting QLoRA training...")
    print(f"  LoRA rank: {qlora_config.lora_r}")
    print(f"  LoRA alpha: {qlora_config.lora_alpha}")
    print(f"  4-bit quantization: {qlora_config.load_in_4bit}")

    history = trainer.train()

    print("\nTraining complete!")
    print(f"Best eval loss: {trainer.best_eval_loss:.4f}")
    print(f"Adapters saved to: {output_dir}")


def train_full(
    model_name,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    settings,
    output_dir,
    num_epochs,
    learning_rate,
    device,
):
    """Train with full fine-tuning."""
    from src.distillation.trainer import DistillationTrainer

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    trainer = DistillationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_ratio=settings.training.warmup_ratio,
        weight_decay=settings.training.weight_decay,
        max_grad_norm=settings.training.max_grad_norm,
        gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        mixed_precision=settings.mixed_precision,
        output_dir=output_dir,
        save_strategy=settings.training.save_strategy,
        save_total_limit=settings.training.save_total_limit,
        logging_steps=settings.training.logging_steps,
        eval_steps=settings.training.eval_steps,
        early_stopping_patience=settings.training.early_stopping_patience,
        device=device,
    )

    print("\nStarting full fine-tuning...")
    history = trainer.train()

    print("\nTraining complete!")
    print(f"Best eval loss: {trainer.best_eval_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
