#!/usr/bin/env python3
"""
Fine-tune T5-small for question generation.
Supports both baseline and topic-conditioned training.
"""

import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np

class QuestionGenerationDataset(Dataset):
    """Dataset for question generation."""

    def __init__(self, data_file, tokenizer, max_input_length=512, max_target_length=64):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            item['input'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            item['target'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Replace padding token id with -100 for loss calculation
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune T5-small for question generation')
    parser.add_argument('--train-file', required=True, help='Training data JSON')
    parser.add_argument('--val-file', required=True, help='Validation data JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    parser.add_argument('--model-name', default='t5-small', help='Base model name')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-input-length', type=int, default=512, help='Max input length')
    parser.add_argument('--max-target-length', type=int, default=64, help='Max target length')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("="*70)
    print("T5-SMALL FINE-TUNING")
    print("="*70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    # Load tokenizer and model
    print(f"\nLoading {args.model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    # Add special tokens if needed
    special_tokens = ['<sep>']
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens: {special_tokens}")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = QuestionGenerationDataset(
        args.train_file,
        tokenizer,
        args.max_input_length,
        args.max_target_length
    )
    val_dataset = QuestionGenerationDataset(
        args.val_file,
        tokenizer,
        args.max_input_length,
        args.max_target_length
    )

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val Loss:   {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = output_dir / f'checkpoint-epoch-{epoch+1}'
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / 'best_model'
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best model! Saved to {best_dir}")

    # Save final model
    final_dir = output_dir / 'final_model'
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"  - best_model/ (lowest val loss)")
    print(f"  - final_model/ (last epoch)")

if __name__ == "__main__":
    main()
