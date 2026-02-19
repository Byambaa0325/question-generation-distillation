"""Loss functions for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for sequence-to-sequence knowledge distillation.

    This loss combines:
    1. Standard cross-entropy loss on target tokens
    2. Optional KL divergence for soft label distillation (if teacher logits available)
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (1-alpha for hard labels)
            ignore_index: Index to ignore in cross-entropy
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model output logits [batch, seq_len, vocab]
            labels: Target token IDs [batch, seq_len]
            teacher_logits: Optional teacher logits for soft distillation

        Returns:
            Dictionary with loss components
        """
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)

        if teacher_logits is None:
            # No teacher logits, use only hard labels
            return {
                "loss": hard_loss,
                "hard_loss": hard_loss,
                "soft_loss": torch.tensor(0.0, device=student_logits.device),
            }

        # Soft label loss (KL divergence with temperature)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)

        # Create mask for valid positions
        mask = labels_flat != self.ignore_index
        if mask.sum() == 0:
            return {
                "loss": hard_loss,
                "hard_loss": hard_loss,
                "soft_loss": torch.tensor(0.0, device=student_logits.device),
            }

        # Apply mask
        student_logits_masked = student_logits_flat[mask]
        teacher_logits_masked = teacher_logits_flat[mask]

        # Compute soft targets
        soft_targets = F.softmax(teacher_logits_masked / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits_masked / self.temperature, dim=-1)

        # KL divergence loss
        soft_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
        }


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for sequence generation."""

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Initialize label smoothing loss.

        Args:
            smoothing: Label smoothing factor
            ignore_index: Index to ignore
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab]
            labels: Target token IDs [batch, seq_len]

        Returns:
            Smoothed cross-entropy loss
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Create mask
        mask = labels_flat != self.ignore_index

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Apply mask
        logits_masked = logits_flat[mask]
        labels_masked = labels_flat[mask]

        # Compute log probabilities
        log_probs = F.log_softmax(logits_masked, dim=-1)

        # Create smoothed targets
        smooth_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
        smooth_targets.scatter_(1, labels_masked.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()

        return loss
