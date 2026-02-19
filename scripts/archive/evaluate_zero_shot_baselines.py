#!/usr/bin/env python3
"""
Zero-shot baseline evaluation using T5-base.
Compare with paper's reported baselines.
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load
from tqdm import tqdm

# Paper's reported baselines (from Table 2)
PAPER_BASELINES = {
    'SQuAD': {
        'T5-base (zero-shot)': 16.51,
        'T5-small (fine-tuned)': 18.23,
        'Ours (T5-small + topic)': 19.45
    },
    'KhanQ': {
        'T5-base (zero-shot)': 9.32,
        'T5-small (fine-tuned)': 11.45,
        'Ours (T5-small + topic)': 13.78
    }
}

def load_filtered_data(filepath, max_samples=None):
    """Load filtered enriched data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    return data

def generate_questions_zero_shot(data, model, tokenizer, batch_size=4, max_length=64):
    """
    Generate questions using T5-base zero-shot.
    Input: context only (no topic conditioning)
    """
    print(f"\nGenerating {len(data)} questions with T5-base (zero-shot)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    generated_questions = []
    references = []

    for i in tqdm(range(0, len(data), batch_size), desc="Generating"):
        batch = data[i:i + batch_size]

        # Prepare inputs (context only, no topic)
        inputs = [f"generate question: {item['text']}" for item in batch]

        # Tokenize
        encoding = tokenizer(
            inputs,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        batch_questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_questions.extend(batch_questions)

        # Collect references
        batch_refs = [item['question'] for item in batch]
        references.extend(batch_refs)

    return generated_questions, references

def calculate_bleu(predictions, references):
    """Calculate BLEU score using word-level tokenization (paper's method)."""
    from nltk.translate.bleu_score import corpus_bleu

    # Tokenize at word level
    predictions_tokenized = [pred.split() for pred in predictions]
    references_tokenized = [[ref.split()] for ref in references]  # List of lists of references

    # Calculate BLEU using nltk (same as paper)
    bleu_score = corpus_bleu(references_tokenized, predictions_tokenized)

    return bleu_score * 100  # Convert to percentage

def evaluate_dataset(data_path, dataset_name, model, tokenizer, max_samples=None):
    """Evaluate zero-shot baseline on a dataset."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*70}")

    # Load data
    data = load_filtered_data(data_path, max_samples=max_samples)
    print(f"Loaded {len(data)} filtered items")

    if len(data) == 0:
        print(f"No data available for {dataset_name}")
        return None

    # Generate questions
    predictions, references = generate_questions_zero_shot(data, model, tokenizer)

    # Calculate BLEU
    bleu_score = calculate_bleu(predictions, references)

    print(f"\nResults:")
    print(f"  BLEU Score: {bleu_score:.2f}")

    # Compare with paper
    paper_zero_shot = PAPER_BASELINES.get(dataset_name, {}).get('T5-base (zero-shot)')
    if paper_zero_shot:
        print(f"  Paper's T5-base (zero-shot): {paper_zero_shot:.2f}")
        diff = bleu_score - paper_zero_shot
        print(f"  Difference: {diff:+.2f}")

    return {
        'dataset': dataset_name,
        'method': 'T5-base (zero-shot)',
        'bleu': bleu_score,
        'num_samples': len(data),
        'paper_baseline': paper_zero_shot
    }

def create_results_table(results, output_csv):
    """Create comparison table with paper's baselines."""

    # Create rows for each dataset
    rows = []

    for dataset_name in ['SQuAD', 'KhanQ']:
        paper_baselines = PAPER_BASELINES.get(dataset_name, {})

        # Find our result
        our_result = next((r for r in results if r['dataset'] == dataset_name), None)
        our_bleu = our_result['bleu'] if our_result else None

        # Add rows
        rows.append({
            'Dataset': dataset_name,
            'Method': 'T5-base (zero-shot)',
            'Our BLEU': f"{our_bleu:.2f}" if our_bleu else 'N/A',
            'Paper BLEU': f"{paper_baselines.get('T5-base (zero-shot)', 0):.2f}",
            'Difference': f"{(our_bleu - paper_baselines.get('T5-base (zero-shot)', 0)):+.2f}" if our_bleu else 'N/A'
        })

        rows.append({
            'Dataset': dataset_name,
            'Method': 'T5-small (fine-tuned)',
            'Our BLEU': 'TODO',
            'Paper BLEU': f"{paper_baselines.get('T5-small (fine-tuned)', 0):.2f}",
            'Difference': 'TODO'
        })

        rows.append({
            'Dataset': dataset_name,
            'Method': 'Ours (T5-small + topic)',
            'Our BLEU': 'TODO',
            'Paper BLEU': f"{paper_baselines.get('Ours (T5-small + topic)', 0):.2f}",
            'Difference': 'TODO'
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*70}\n")

    # Display table
    print(df.to_string(index=False))

    return df

def main():
    print("\n" + "="*70)
    print("ZERO-SHOT BASELINE EVALUATION")
    print("="*70)
    print("\nComparing T5-base zero-shot with paper's reported baselines")

    # Load model
    print("\nLoading T5-base model...")
    model_name = "google-t5/t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets to evaluate
    datasets = [
        ('data/processed/wat/enriched_squad_filtered.json', 'SQuAD', 1000),  # Sample 1000 for speed
        ('data/processed/wat/enriched_khanq_filtered.json', 'KhanQ', None),   # Use all (only 47)
    ]

    results = []

    for data_path, dataset_name, max_samples in datasets:
        if Path(data_path).exists():
            result = evaluate_dataset(data_path, dataset_name, model, tokenizer, max_samples)
            if result:
                results.append(result)
        else:
            print(f"\nSkipping {dataset_name}: File not found - {data_path}")

    # Create comparison table
    output_csv = 'results/baseline_comparison.csv'
    Path('results').mkdir(exist_ok=True)

    df = create_results_table(results, output_csv)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Zero-shot baselines complete!

Next steps:
1. Fine-tune T5-small on filtered datasets
2. Fine-tune T5-small with topic conditioning
3. Compare all methods with paper's results
""")

if __name__ == "__main__":
    main()
