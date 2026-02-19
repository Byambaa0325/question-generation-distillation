"""
Fine-tune T5-small for topic-controlled question generation.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import pandas as pd
import copy
from tqdm import tqdm
import argparse
import os

class QGDataset(Dataset):
    """Question Generation Dataset."""

    def __init__(self, tokenizer, file_path, max_len_input=200, max_len_output=45):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path)
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.inputs = []
        self.targets = []
        self._load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100  # Ignore padding in loss

        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'labels': labels
        }

    def _load_data(self):
        """Load and tokenize data."""
        print(f"Loading {len(self.data)} examples from dataset...")

        for idx in tqdm(range(len(self.data)), desc="Tokenizing"):
            context = str(self.data.loc[idx, 'text'])
            topic = str(self.data.loc[idx, 'topic'])
            question = str(self.data.loc[idx, 'question'])

            # Paper's input format: '<topic> {} <context> {} '
            input_text = '<topic> {} <context> {} '.format(topic, context)

            # Tokenize input
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len_input,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Tokenize target
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [question],
                max_length=self.max_len_output,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

class T5FineTuner(pl.LightningModule):
    """PyTorch Lightning module for T5 fine-tuning."""

    def __init__(self, model, tokenizer, train_dataset, val_dataset, batch_size=64, lr=0.001):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, eps=1e-08)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune T5 for question generation')
    parser.add_argument('--train-data', required=True, help='Training CSV file')
    parser.add_argument('--val-data', required=True, help='Validation CSV file')
    parser.add_argument('--output-dir', default='models/t5-qg', help='Output directory for model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-input-len', type=int, default=200, help='Max input token length')
    parser.add_argument('--max-output-len', type=int, default=45, help='Max output token length')

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model and tokenizer
    print('Loading T5-small model and tokenizer...')
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    # Add special tokens
    print('Adding special tokens...')
    SPECIAL_TOKENS = ['<sep>', '<space>']
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    print('Loading datasets...')
    train_dataset = QGDataset(
        tokenizer,
        args.train_data,
        max_len_input=args.max_input_len,
        max_len_output=args.max_output_len
    )
    val_dataset = QGDataset(
        tokenizer,
        args.val_data,
        max_len_input=args.max_input_len,
        max_len_output=args.max_output_len
    )

    print(f'Train dataset: {len(train_dataset)} examples')
    print(f'Val dataset: {len(val_dataset)} examples')

    # Create Lightning module
    print('Initializing training module...')
    lightning_model = T5FineTuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='t5-qg-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10
    )

    # Train
    print('\nStarting training...')
    trainer.fit(lightning_model)

    # Save final model
    print(f'\nSaving final model to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print('\n[DONE] Training complete!')
    print(f'Model saved to: {args.output_dir}')

if __name__ == "__main__":
    main()
