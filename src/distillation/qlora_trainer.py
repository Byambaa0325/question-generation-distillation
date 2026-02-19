"""QLoRA-specific trainer for memory-efficient fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .qlora import QLoRAConfig, setup_qlora_model, load_model_for_qlora, save_qlora_adapters


@dataclass
class QLoRATrainingMetrics:
    """Training metrics container."""

    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0


class QLoRATrainer:
    """Trainer for QLoRA fine-tuning of T5 models."""

    def __init__(
        self,
        model_name: str,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        qlora_config: Optional[QLoRAConfig] = None,
        learning_rate: float = 2e-4,  # Higher LR typical for LoRA
        num_epochs: int = 3,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.3,
        gradient_accumulation_steps: int = 4,
        output_dir: str = "outputs/qlora",
        save_strategy: str = "epoch",
        save_total_limit: int = 3,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_merged: bool = False,  # Save merged model or just adapters
        callbacks: Optional[list[Callable]] = None,
    ):
        """
        Initialize QLoRA trainer.

        Args:
            model_name: HuggingFace model name to fine-tune
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            qlora_config: QLoRA configuration (uses defaults if None)
            learning_rate: Learning rate (2e-4 is typical for LoRA)
            num_epochs: Number of training epochs
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm for clipping
            gradient_accumulation_steps: Gradient accumulation steps
            output_dir: Output directory for checkpoints
            save_strategy: When to save ("epoch" or "steps")
            save_total_limit: Max checkpoints to keep
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_merged: If True, save merged model; if False, save only adapters
            callbacks: Optional callback functions
        """
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.qlora_config = qlora_config or QLoRAConfig()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_merged = save_merged
        self.callbacks = callbacks or []

        # Load and setup model with QLoRA
        print(f"Loading {model_name} with 4-bit quantization...")
        self.model, _ = load_model_for_qlora(model_name, self.qlora_config)

        print("Applying LoRA adapters...")
        self.model, self.lora_config = setup_qlora_model(
            self.model, self.qlora_config, model_type="t5"
        )

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.training_history: list[QLoRATrainingMetrics] = []

        # Initialize optimizer
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            optimizer_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Calculate total steps
        total_steps = len(self.train_dataloader) * self.num_epochs
        total_steps = total_steps // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train(self) -> list[QLoRATrainingMetrics]:
        """
        Run QLoRA training loop.

        Returns:
            Training history
        """
        print(f"Starting QLoRA training")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Train batches: {len(self.train_dataloader)}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")

        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                # Gradient update
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

                # Evaluation during training
                if (
                    self.eval_dataloader
                    and self.save_strategy == "steps"
                    and self.global_step % self.eval_steps == 0
                    and self.global_step > 0
                ):
                    eval_loss = self.evaluate()
                    self._handle_checkpoint(epoch, eval_loss)
                    self.model.train()

            # End of epoch
            avg_train_loss = epoch_loss / num_batches

            # Evaluation at end of epoch
            eval_loss = None
            if self.eval_dataloader:
                eval_loss = self.evaluate()

            metrics = QLoRATrainingMetrics(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                eval_loss=eval_loss,
                learning_rate=self.scheduler.get_last_lr()[0],
            )
            self.training_history.append(metrics)

            print(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f}", end="")
            if eval_loss:
                print(f", Eval loss: {eval_loss:.4f}")
            else:
                print()

            # Save checkpoint
            if self.save_strategy == "epoch":
                self._handle_checkpoint(epoch, eval_loss or avg_train_loss)

            # Run callbacks
            for callback in self.callbacks:
                callback(self, metrics)

        # Save final model
        self._save_checkpoint("final")
        self._save_training_history()

        return self.training_history

    def evaluate(self) -> float:
        """
        Run evaluation.

        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _handle_checkpoint(self, epoch: int, loss: float):
        """Handle checkpoint saving."""
        if loss < self.best_eval_loss:
            self.best_eval_loss = loss
            self._save_checkpoint("best")

        self._save_checkpoint(f"epoch_{epoch + 1}")
        self._cleanup_checkpoints()

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters (or merged model)
        if self.save_merged:
            from .qlora import merge_and_save_qlora
            merge_and_save_qlora(self.model, checkpoint_dir, self.tokenizer)
        else:
            save_qlora_adapters(self.model, checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config and state
        config = {
            "model_name": self.model_name,
            "qlora_config": {
                "lora_r": self.qlora_config.lora_r,
                "lora_alpha": self.qlora_config.lora_alpha,
                "lora_dropout": self.qlora_config.lora_dropout,
            },
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }
        with open(checkpoint_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir()
             if d.is_dir() and d.name.startswith("epoch_")],
            key=lambda x: int(x.name.split("_")[1]),
        )

        while len(checkpoints) > self.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)

    def _save_training_history(self):
        """Save training history to JSON."""
        history = [
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "eval_loss": m.eval_loss,
                "learning_rate": m.learning_rate,
            }
            for m in self.training_history
        ]
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
