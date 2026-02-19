"""Training loop for knowledge distillation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .losses import DistillationLoss


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0


class DistillationTrainer:
    """Trainer for T5 student model distillation."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        output_dir: str = "outputs/distillation",
        save_strategy: str = "epoch",
        save_total_limit: int = 3,
        logging_steps: int = 50,
        eval_steps: int = 500,
        early_stopping_patience: int = 3,
        device: Optional[str] = None,
        callbacks: Optional[list[Callable]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: T5 model to train
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm for clipping
            gradient_accumulation_steps: Gradient accumulation steps
            mixed_precision: Use mixed precision training
            output_dir: Output directory for checkpoints
            save_strategy: When to save ("epoch" or "steps")
            save_total_limit: Max checkpoints to keep
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            early_stopping_patience: Early stopping patience
            device: Device to use
            callbacks: Optional callback functions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.callbacks = callbacks or []

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0
        self.training_history: list[TrainingMetrics] = []

        # Initialize optimizer and scheduler
        self._setup_optimizer()

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # No weight decay for bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Calculate total steps
        total_steps = len(self.train_dataloader) * self.num_epochs
        total_steps = total_steps // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train(self) -> list[TrainingMetrics]:
        """
        Run training loop.

        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_dataloader)}")

        loss_fn = DistillationLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                # Gradient update
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_metrics(epoch, loss.item())

                # Evaluation during training
                if (
                    self.eval_dataloader
                    and self.save_strategy == "steps"
                    and self.global_step % self.eval_steps == 0
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

            metrics = TrainingMetrics(
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

            # Early stopping
            if self._check_early_stopping(eval_loss or avg_train_loss):
                print("Early stopping triggered")
                break

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
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

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
            self.patience_counter = 0
            self._save_checkpoint("best")
        else:
            self.patience_counter += 1

        self._save_checkpoint(f"epoch_{epoch + 1}")
        self._cleanup_checkpoints()

    def _check_early_stopping(self, loss: float) -> bool:
        """Check if early stopping should trigger."""
        return self.patience_counter >= self.early_stopping_patience

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "patience_counter": self.patience_counter,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
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

    def _log_metrics(self, epoch: int, loss: float):
        """Log training metrics."""
        # Extensible for wandb/tensorboard integration
        pass
